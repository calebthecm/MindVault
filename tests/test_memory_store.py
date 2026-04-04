"""Tests for src/memory/store.py — uses in-memory Qdrant and temp SQLite."""

import tempfile
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.memory.store import MemoryStore

COL_PUBLIC = "brain_compressed_public"
COL_PRIVATE = "brain_compressed_private"


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    qdrant = QdrantClient(":memory:")
    s = MemoryStore(
        db_path=db_path,
        qdrant=qdrant,
        compressed_collections=(COL_PUBLIC, COL_PRIVATE),
    )
    yield s


def test_tables_created(store):
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "memory_compressed" in tables
    assert "memory_entities" in tables
    assert "memory_links" in tables
    assert "memory_importance" in tables


def test_store_and_retrieve_entities(store):
    store.store_entities(
        [{"type": "project", "name": "MindVault", "value": "second brain system"}],
        source_id="doc_123",
    )
    entities = store.get_entities_for_source("doc_123")
    assert len(entities) == 1
    assert entities[0]["name"] == "MindVault"
    assert entities[0]["type"] == "project"


def test_find_entities_by_name(store):
    store.store_entities(
        [{"type": "decision", "name": "charge $500/month", "value": "retainer model"}],
        source_id="session_abc",
    )
    results = store.find_entities_by_name(["charge $500/month"])
    assert len(results) == 1
    assert results[0]["source_id"] == "session_abc"


def test_importance_updates(store):
    score_before = store.get_importance("chunk_xyz")
    assert score_before == 0.5  # default
    store.update_importance("chunk_xyz")
    score_after = store.get_importance("chunk_xyz")
    assert score_after > 0.5


def test_compressed_collections_created(store):
    collections = {c.name for c in store.qdrant.get_collections().collections}
    assert COL_PUBLIC in collections
    assert COL_PRIVATE in collections
