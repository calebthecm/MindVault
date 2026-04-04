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
from pathlib import Path
from typing import Optional

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
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS memory_links (
                    id TEXT PRIMARY KEY,
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS memory_importance (
                    chunk_id TEXT PRIMARY KEY,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    reinforcement_count INTEGER DEFAULT 0
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
        compressed_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_compressed
                    (id, source_id, source_type, summary, chunk_ids, token_estimate)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                compressed_id, source_id, source_type, summary,
                json.dumps(chunk_ids), len(summary) // 4,
            ))

        payload = {"summary": summary, "source_id": source_id, "source_type": source_type}
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
                    INSERT INTO memory_entities (id, entity_type, name, value, source_id, chunk_id, confidence)
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
