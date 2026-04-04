"""
Hybrid retriever — combines embedding similarity, entity overlap, recency, importance.

Score = 0.5 * embedding_sim + 0.2 * entity_overlap + 0.2 * recency + 0.1 * importance

Retrieval order:
  1. Search compressed collection (summaries — prefer these)
  2. If top compressed score < threshold, also pull raw chunks
  3. Boost chunks whose source has entity matches for query entities
  4. Update importance scores for every retrieved chunk
"""

import logging
from datetime import datetime, timezone
from math import exp
from typing import Optional

from qdrant_client import QdrantClient

from src.memory.store import MemoryStore

logger = logging.getLogger(__name__)

W_EMBEDDING = 0.5
W_ENTITY = 0.2
W_RECENCY = 0.2
W_IMPORTANCE = 0.1


def _recency_score(created_at_iso: str, half_life_days: int = 180) -> float:
    """Exponential decay from created_at. Returns 0.5 on parse failure."""
    try:
        dt = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        days_old = max(0, (datetime.now(timezone.utc) - dt).days)
        return exp(-0.693 * days_old / half_life_days)
    except Exception:
        return 0.5


def hybrid_score(
    embedding_sim: float,
    entity_score: float,
    recency: float,
    importance: float,
) -> float:
    return (
        W_EMBEDDING * embedding_sim
        + W_ENTITY * entity_score
        + W_RECENCY * recency
        + W_IMPORTANCE * importance
    )


def retrieve(
    query_vector: list[float],
    qdrant: QdrantClient,
    memory_store: MemoryStore,
    raw_collection: str,
    compressed_collection: str,
    top_k: int = 8,
    compressed_threshold: float = 0.75,
    query_entities: Optional[list[str]] = None,
) -> list[dict]:
    """
    Hybrid retrieval: compressed-first with raw fallback.

    Returns list of chunk dicts sorted by hybrid score.
    Each dict has: chunk_id, source_id, text, title, source_type,
                   created_at, score, layer ("compressed" | "raw")
    """
    query_entities = query_entities or []
    chunks: list[dict] = []

    # --- Compressed search (always) ---
    compressed_hits = qdrant.query_points(
        collection_name=compressed_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    top_compressed_score = compressed_hits[0].score if compressed_hits else 0.0

    # Pre-fetch entity source IDs for query entity matching
    entity_source_ids: set[str] = set()
    if query_entities:
        matched = memory_store.find_entities_by_name(query_entities)
        entity_source_ids = {e["source_id"] for e in matched}

    for hit in compressed_hits:
        payload = hit.payload or {}
        source_id = payload.get("source_id", "")
        recency = _recency_score(payload.get("created_at", ""))
        entity_score = 1.0 if source_id in entity_source_ids else 0.0
        importance = memory_store.get_importance(str(hit.id))
        score = hybrid_score(hit.score, entity_score, recency, importance)

        chunks.append({
            "chunk_id": str(hit.id),
            "source_id": source_id,
            "text": payload.get("summary", ""),
            "title": payload.get("title", source_id),
            "source_type": payload.get("source_type", ""),
            "created_at": payload.get("created_at", ""),
            "score": round(score, 3),
            "layer": "compressed",
        })
        memory_store.update_importance(str(hit.id))

    # --- Raw fallback (only when compressed confidence is low) ---
    if top_compressed_score < compressed_threshold:
        raw_hits = qdrant.query_points(
            collection_name=raw_collection,
            query=query_vector,
            limit=max(top_k // 2, 4),
            with_payload=True,
        ).points

        for hit in raw_hits:
            payload = hit.payload or {}
            source_id = payload.get("document_id", "")
            recency = _recency_score(payload.get("created_at", ""))
            entity_score = 1.0 if source_id in entity_source_ids else 0.0
            importance = memory_store.get_importance(str(hit.id))
            score = hybrid_score(hit.score, entity_score, recency, importance)

            chunks.append({
                "chunk_id": str(hit.id),
                "source_id": source_id,
                "text": payload.get("text", ""),
                "title": payload.get("title", ""),
                "source_type": payload.get("source_type", ""),
                "created_at": payload.get("created_at", ""),
                "speaker": payload.get("speaker"),
                "score": round(score, 3),
                "layer": "raw",
            })
            memory_store.update_importance(str(hit.id))

    # --- Deduplicate by source and rank ---
    seen_sources: set[str] = set()
    result: list[dict] = []
    for chunk in sorted(chunks, key=lambda c: c["score"], reverse=True):
        src = chunk["source_id"]
        if src not in seen_sources:
            seen_sources.add(src)
            result.append(chunk)
        if len(result) >= top_k:
            break

    return result
