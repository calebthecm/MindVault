"""
Memory storage layer — enrichment tables on top of raw chunks.

Tables:
  memory_compressed  — document/session summaries + Qdrant embeddings
  memory_entities    — extracted persons, projects, decisions, goals, facts
  memory_links       — relationships between chunks
  memory_importance  — access counts and importance scores

Also manages brain_compressed_public/private Qdrant collections.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768


class MemoryStore:
    def __init__(self, db_path: Path, qdrant: QdrantClient, compressed_collections: tuple[str, str]):
        self.db_path = Path(db_path)
        self.qdrant = qdrant
        self.col_compressed_public, self.col_compressed_private = compressed_collections
        self._init_tables()
        self._init_collections()

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_compressed (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    chunk_ids TEXT NOT NULL,
                    token_estimate INTEGER,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS memory_entities (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value TEXT,
                    source_id TEXT NOT NULL,
                    chunk_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(entity_type, lower(name), source_id)
                );

                CREATE TABLE IF NOT EXISTS memory_links (
                    id TEXT PRIMARY KEY,
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(from_id, to_id, link_type)
                );

                CREATE TABLE IF NOT EXISTS memory_importance (
                    chunk_id TEXT PRIMARY KEY,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    reinforcement_count INTEGER DEFAULT 0,
                    suppressed INTEGER DEFAULT 0
                );
            """)

    def _init_collections(self) -> None:
        existing = {c.name for c in self.qdrant.get_collections().collections}
        for name in [self.col_compressed_public, self.col_compressed_private]:
            if name not in existing:
                self.qdrant.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                logger.info(f"Created compressed collection: {name}")

    # Stable namespace for deterministic compressed-memory IDs.
    _COMPRESSED_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    def store_compressed(
        self,
        source_id: str,
        source_type: str,
        summary: str,
        chunk_ids: list[str],
        vector: list[float],
        collection: str,
        metadata: Optional[dict] = None,
    ) -> str:
        # Derive a stable ID from source_id so re-processing replaces the old record
        # instead of accumulating duplicate summaries for the same source.
        compressed_id = str(uuid.uuid5(self._COMPRESSED_NS, source_id))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_compressed
                    (id, source_id, source_type, summary, chunk_ids, token_estimate)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                compressed_id, source_id, source_type, summary,
                json.dumps(chunk_ids), len(summary) // 4,
            ))

        # Include created_at so the retriever can compute recency for compressed results
        payload = {
            "summary": summary,
            "source_id": source_id,
            "source_type": source_type,
            "created_at": (metadata or {}).get("created_at", _now_iso()),
        }
        if metadata:
            payload.update(metadata)

        self.qdrant.upsert(
            collection_name=collection,
            points=[PointStruct(id=compressed_id, vector=vector, payload=payload)],
        )
        return compressed_id

    def store_entities(self, entities: list[dict], source_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for entity in entities:
                conn.execute("""
                    INSERT OR IGNORE INTO memory_entities (id, entity_type, name, value, source_id, chunk_id, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    entity.get("type", "fact"),
                    entity.get("name", ""),
                    entity.get("value", ""),
                    source_id,
                    entity.get("chunk_id"),
                    entity.get("confidence", 1.0),
                ))

    def get_entities_for_source(self, source_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT entity_type, name, value FROM memory_entities WHERE source_id = ?",
                (source_id,)
            ).fetchall()
        return [{"type": r[0], "name": r[1], "value": r[2]} for r in rows]

    def find_entities_by_name(self, names: list[str]) -> list[dict]:
        if not names:
            return []
        placeholders = ",".join("?" * len(names))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT source_id, entity_type, name, value FROM memory_entities "
                f"WHERE lower(name) IN ({placeholders})",
                [n.lower() for n in names],
            ).fetchall()
        return [{"source_id": r[0], "type": r[1], "name": r[2], "value": r[3]} for r in rows]

    def update_importance(self, chunk_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memory_importance (chunk_id, importance, access_count, last_accessed)
                VALUES (?, 0.55, 1, datetime('now'))
                ON CONFLICT(chunk_id) DO UPDATE SET
                    access_count = access_count + 1,
                    last_accessed = datetime('now'),
                    importance = min(1.0, importance + 0.05)
            """, (chunk_id,))

    def get_importance(self, chunk_id: str) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT importance FROM memory_importance WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return row[0] if row else 0.5

    def suppress_chunks(self, chunk_ids: list[str]) -> int:
        """Mark chunks as suppressed so they are excluded from future retrieval."""
        with sqlite3.connect(self.db_path) as conn:
            for cid in chunk_ids:
                conn.execute("""
                    INSERT INTO memory_importance (chunk_id, suppressed)
                    VALUES (?, 1)
                    ON CONFLICT(chunk_id) DO UPDATE SET suppressed = 1
                """, (cid,))
        return len(chunk_ids)

    def get_suppressed_ids(self) -> set[str]:
        """Return set of all suppressed chunk_ids."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM memory_importance WHERE suppressed = 1"
            ).fetchall()
        return {r[0] for r in rows}

    # ── Memory links ──────────────────────────────────────────────────────────

    def store_links(self, links: list[dict]) -> int:
        """
        Insert links between source IDs. Each link: {from_id, to_id, link_type, strength}.
        Skips duplicates (same from_id + to_id + link_type).
        Returns count of new links inserted.
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            for link in links:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO memory_links (id, from_id, to_id, link_type, strength)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        str(uuid.uuid4()),
                        link["from_id"],
                        link["to_id"],
                        link["link_type"],
                        link.get("strength", 1.0),
                    ))
                    inserted += conn.execute("SELECT changes()").fetchone()[0]
                except sqlite3.Error as e:
                    logger.warning(f"Failed to insert link: {e}")
        return inserted

    def get_links_for_source(
        self,
        source_id: str,
        link_type: Optional[str] = None,
        min_strength: float = 0.0,
    ) -> list[dict]:
        """Return all sources linked to source_id (in either direction)."""
        with sqlite3.connect(self.db_path) as conn:
            if link_type:
                rows = conn.execute("""
                    SELECT to_id AS linked_id, link_type, strength FROM memory_links
                    WHERE from_id = ? AND link_type = ? AND strength >= ?
                    UNION
                    SELECT from_id AS linked_id, link_type, strength FROM memory_links
                    WHERE to_id = ? AND link_type = ? AND strength >= ?
                """, (source_id, link_type, min_strength, source_id, link_type, min_strength)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT to_id AS linked_id, link_type, strength FROM memory_links
                    WHERE from_id = ? AND strength >= ?
                    UNION
                    SELECT from_id AS linked_id, link_type, strength FROM memory_links
                    WHERE to_id = ? AND strength >= ?
                """, (source_id, min_strength, source_id, min_strength)).fetchall()
        return [{"linked_id": r[0], "link_type": r[1], "strength": r[2]} for r in rows]

    def get_entity_source_map(self) -> dict[str, list[str]]:
        """Return {normalized_entity_name: [source_ids]} for all entities."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT lower(name), source_id FROM memory_entities"
            ).fetchall()
        result: dict[str, list[str]] = {}
        for name, source_id in rows:
            result.setdefault(name, []).append(source_id)
        return result

    def delete_compressed(self, compressed_id: str, collection: str) -> None:
        """Remove a compressed memory from SQLite and Qdrant."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory_compressed WHERE id = ?", (compressed_id,))
        try:
            self.qdrant.delete(
                collection_name=collection,
                points_selector=[compressed_id],
            )
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant point {compressed_id}: {e}")
