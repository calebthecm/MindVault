"""
Storage layer for the Brain RAG system.

Two storage backends work together:
  1. SQLite (brain.db) — ingestion tracking, deduplication, source attribution
  2. Qdrant (in-memory or persisted) — vector search

Privacy: My Brain and Private Brain are in separate Qdrant collections.
Conversation data goes into the 'public' collection.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.models import Chunk, PrivacyLevel, VaultName

logger = logging.getLogger(__name__)

# Collection names — one per privacy boundary
COLLECTION_PUBLIC = "brain_public"
COLLECTION_PRIVATE = "brain_private"

EMBEDDING_DIM = 768  # nomic-embed-text output dimension


def _collection_for(privacy_level: PrivacyLevel, vault: VaultName) -> str:
    """Route a chunk to the correct Qdrant collection."""
    if privacy_level == PrivacyLevel.PRIVATE or vault == VaultName.PRIVATE_BRAIN:
        return COLLECTION_PRIVATE
    return COLLECTION_PUBLIC


class BrainStore:
    """
    Unified storage interface. Initialize once and use throughout the pipeline.

    Args:
        db_path: Path to SQLite database file (created if not exists)
        qdrant_path: Directory for Qdrant persistent storage, or None for in-memory
    """

    def __init__(self, db_path: Path, qdrant_path: Optional[Path] = None):
        self.db_path = Path(db_path)
        self._init_sqlite()

        if qdrant_path:
            Path(qdrant_path).mkdir(parents=True, exist_ok=True)
            self.qdrant = QdrantClient(path=str(qdrant_path))
        else:
            self.qdrant = QdrantClient(":memory:")

        self._init_collections()

    def _init_sqlite(self) -> None:
        """Create tables for ingestion tracking."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ingested_documents (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    vault TEXT NOT NULL,
                    privacy_level TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    export_batch TEXT,
                    ingested_at TEXT DEFAULT (datetime('now')),
                    metadata_json TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS ingested_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    qdrant_point_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    privacy_level TEXT NOT NULL,
                    freshness_score REAL DEFAULT 1.0,
                    ingested_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (document_id) REFERENCES ingested_documents(id)
                );

                CREATE TABLE IF NOT EXISTS export_batches (
                    batch_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    path TEXT NOT NULL,
                    doc_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    ingested_at TEXT DEFAULT (datetime('now'))
                );
            """)
        logger.info(f"SQLite initialized at {self.db_path}")

    def _init_collections(self) -> None:
        """Create Qdrant collections if they don't exist."""
        existing = {c.name for c in self.qdrant.get_collections().collections}

        for name in [COLLECTION_PUBLIC, COLLECTION_PRIVATE]:
            if name not in existing:
                self.qdrant.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {name}")

    def is_document_ingested(self, document_id: str) -> bool:
        """Check if a document has already been indexed (dedup guard)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM ingested_documents WHERE id = ?", (document_id,)
            ).fetchone()
        return row is not None

    def upsert_chunks(
        self,
        chunk_vector_pairs: list[tuple[Chunk, list[float]]],
        export_batch: str = "",
    ) -> None:
        """
        Store chunks into Qdrant and record them in SQLite.
        Skips chunks whose parent document is already indexed.
        """
        if not chunk_vector_pairs:
            return

        # Group by collection
        by_collection: dict[str, list[tuple[Chunk, list[float]]]] = {
            COLLECTION_PUBLIC: [],
            COLLECTION_PRIVATE: [],
        }
        for chunk, vector in chunk_vector_pairs:
            col = _collection_for(chunk.privacy_level, chunk.vault)
            by_collection[col].append((chunk, vector))

        with sqlite3.connect(self.db_path) as conn:
            for collection, pairs in by_collection.items():
                if not pairs:
                    continue

                points = []
                chunk_rows = []
                doc_rows = []

                seen_docs: set[str] = set()

                for chunk, vector in pairs:
                    # Track unique documents in this batch
                    if chunk.document_id not in seen_docs:
                        seen_docs.add(chunk.document_id)
                        doc_rows.append((
                            chunk.document_id,
                            chunk.source_type.value,
                            chunk.vault.value,
                            chunk.privacy_level.value,
                            chunk.title,
                            chunk.created_at.isoformat(),
                            chunk.updated_at.isoformat(),
                            export_batch,
                            json.dumps(chunk.metadata),
                        ))

                    # Qdrant point — payload carries full attribution metadata
                    payload = {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "source_type": chunk.source_type.value,
                        "vault": chunk.vault.value,
                        "privacy_level": chunk.privacy_level.value,
                        "title": chunk.title,
                        "text": chunk.text,
                        "speaker": chunk.speaker,
                        "conversation_uuid": chunk.conversation_uuid,
                        "note_path": chunk.note_path,
                        "freshness_score": chunk.freshness_score,
                        "created_at": chunk.created_at.isoformat(),
                        "updated_at": chunk.updated_at.isoformat(),
                        "index": chunk.index,
                    }
                    points.append(PointStruct(id=chunk.id, vector=vector, payload=payload))

                    chunk_rows.append((
                        chunk.id,
                        chunk.document_id,
                        collection,
                        chunk.id,
                        chunk.source_type.value,
                        chunk.privacy_level.value,
                        chunk.freshness_score,
                    ))

                # Write to Qdrant
                self.qdrant.upsert(collection_name=collection, points=points)

                # Write to SQLite
                conn.executemany("""
                    INSERT OR IGNORE INTO ingested_documents
                        (id, source_type, vault, privacy_level, title, created_at, updated_at, export_batch, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, doc_rows)

                conn.executemany("""
                    INSERT OR IGNORE INTO ingested_chunks
                        (id, document_id, collection, qdrant_point_id, source_type, privacy_level, freshness_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, chunk_rows)

                # Update chunk counts
                for doc_id in seen_docs:
                    count = sum(1 for c, _ in pairs if c.document_id == doc_id)
                    conn.execute(
                        "UPDATE ingested_documents SET chunk_count = chunk_count + ? WHERE id = ?",
                        (count, doc_id)
                    )

                logger.info(f"Upserted {len(points)} chunks into collection '{collection}'")

    def record_batch(self, batch_id: str, source: str, path: str, doc_count: int, chunk_count: int) -> None:
        """Track which export batches have been processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO export_batches (batch_id, source, path, doc_count, chunk_count)
                VALUES (?, ?, ?, ?, ?)
            """, (batch_id, source, path, doc_count, chunk_count))

    def is_batch_ingested(self, batch_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT batch_id FROM export_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        return row is not None

    def stats(self) -> dict:
        """Return summary statistics about what's ingested."""
        with sqlite3.connect(self.db_path) as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM ingested_documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM ingested_chunks").fetchone()[0]
            batch_count = conn.execute("SELECT COUNT(*) FROM export_batches").fetchone()[0]
            by_type = conn.execute(
                "SELECT source_type, COUNT(*) FROM ingested_documents GROUP BY source_type"
            ).fetchall()

        public_count = self.qdrant.count(collection_name=COLLECTION_PUBLIC).count
        private_count = self.qdrant.count(collection_name=COLLECTION_PRIVATE).count

        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "batches": batch_count,
            "by_source_type": dict(by_type),
            "qdrant_public_vectors": public_count,
            "qdrant_private_vectors": private_count,
        }
