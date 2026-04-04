"""
Hybrid retriever — combines embedding similarity, entity overlap, recency, importance,
and graph traversal via memory_links.

Score = 0.5 * embedding_sim + 0.2 * entity_overlap + 0.2 * recency + 0.1 * importance

Retrieval order:
  1. Search compressed collection (summaries — prefer these)
  2. If top compressed score < threshold, also pull raw chunks
  3. Walk memory_links from top direct hits to find linked neighbors (graph expansion)
  4. Score linked neighbors with a small embedding discount (LINK_WEIGHT)
  5. Update importance scores for every retrieved chunk
"""

import logging
from datetime import datetime, timezone
from math import exp
from typing import Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from src.memory.store import MemoryStore

logger = logging.getLogger(__name__)

W_EMBEDDING = 0.5
W_ENTITY = 0.2
W_RECENCY = 0.2
W_IMPORTANCE = 0.1

# How much to discount the embedding score for graph-traversal neighbors.
# A linked node scores slightly lower than a direct hit at the same cosine distance.
LINK_WEIGHT = 0.85

# Maximum linked-neighbor expansions per retrieve() call.
MAX_LINKED_EXPANSIONS = 4


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


def _fetch_linked_chunks(
    query_vector: list[float],
    linked_source_ids: list[str],
    qdrant: QdrantClient,
    memory_store: MemoryStore,
    compressed_collection: str,
    raw_collection: str,
    entity_source_ids: set[str],
) -> list[dict]:
    """
    Fetch compressed (preferred) or raw chunks for a set of linked source IDs,
    scoring them against the query vector to get real embedding similarity.
    Returns chunk dicts with layer="linked".
    """
    if not linked_source_ids:
        return []

    source_filter = Filter(
        must=[FieldCondition(key="source_id", match=MatchAny(any=linked_source_ids))]
    )

    # Try compressed collection first
    try:
        hits = qdrant.query_points(
            collection_name=compressed_collection,
            query=query_vector,
            query_filter=source_filter,
            limit=len(linked_source_ids),
            with_payload=True,
        ).points
    except Exception:
        hits = []

    # Fall back to raw collection for any source_ids not found in compressed
    found_sources = {(h.payload or {}).get("source_id", "") for h in hits}
    missing = [s for s in linked_source_ids if s not in found_sources]

    if missing:
        raw_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchAny(any=missing))]
        )
        try:
            raw_hits = qdrant.query_points(
                collection_name=raw_collection,
                query=query_vector,
                query_filter=raw_filter,
                limit=len(missing),
                with_payload=True,
            ).points
            hits = list(hits) + list(raw_hits)
        except Exception:
            pass

    result = []
    for hit in hits:
        payload = hit.payload or {}
        is_compressed = "summary" in payload
        source_id = payload.get("source_id") or payload.get("document_id", "")
        recency = _recency_score(payload.get("created_at", ""))
        entity_score = 1.0 if source_id in entity_source_ids else 0.0
        importance = memory_store.get_importance(str(hit.id))

        # Discount embedding similarity for graph neighbors
        score = hybrid_score(hit.score * LINK_WEIGHT, entity_score, recency, importance)

        chunk = {
            "chunk_id": str(hit.id),
            "source_id": source_id,
            "text": payload.get("summary" if is_compressed else "text", ""),
            "title": payload.get("title", source_id),
            "source_type": payload.get("source_type", ""),
            "created_at": payload.get("created_at", ""),
            "score": round(score, 3),
            "layer": "linked",
        }
        if not is_compressed:
            chunk["speaker"] = payload.get("speaker")
        result.append(chunk)
        memory_store.update_importance(str(hit.id))

    return result


def retrieve(
    query_vector: list[float],
    qdrant: QdrantClient,
    memory_store: MemoryStore,
    raw_collection: str,
    compressed_collection: str,
    top_k: int = 8,
    compressed_threshold: float = 0.75,
    query_entities: Optional[list[str]] = None,
    expand_links: bool = True,
    date_after: Optional[datetime] = None,
    date_before: Optional[datetime] = None,
) -> list[dict]:
    """
    Hybrid retrieval: compressed-first, raw fallback, graph expansion.

    Steps:
      1. Search compressed collection (summaries)
      2. If top score < threshold, also search raw chunks
      3. For top direct hits, walk memory_links and fetch linked neighbors
      4. Score everything with hybrid_score; linked nodes get a small discount
      5. Deduplicate by source_id, return top_k sorted by score

    Returns list of chunk dicts. Each dict has:
      chunk_id, source_id, text, title, source_type, created_at, score,
      layer ("compressed" | "raw" | "linked")
    """
    query_entities = query_entities or []
    chunks: list[dict] = []
    suppressed = memory_store.get_suppressed_ids()
    # Fetch extra results when date filtering is active — many will be filtered out
    fetch_k = top_k * 4 if (date_after or date_before) else top_k

    # --- Pre-fetch entity source IDs for query entity matching ---
    entity_source_ids: set[str] = set()
    if query_entities:
        matched = memory_store.find_entities_by_name(query_entities)
        entity_source_ids = {e["source_id"] for e in matched}

    # --- Step 1: Compressed search (always) ---
    compressed_hits = qdrant.query_points(
        collection_name=compressed_collection,
        query=query_vector,
        limit=fetch_k,
        with_payload=True,
    ).points

    top_compressed_score = compressed_hits[0].score if compressed_hits else 0.0

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

    # --- Step 2: Raw fallback (only when compressed confidence is low) ---
    if top_compressed_score < compressed_threshold:
        raw_hits = qdrant.query_points(
            collection_name=raw_collection,
            query=query_vector,
            limit=max(fetch_k // 2, 4),
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

    # --- Step 3: Graph expansion via memory_links ---
    if expand_links and chunks:
        # Walk links from the top direct hits (up to top_k // 2 sources)
        direct_sources = list(dict.fromkeys(c["source_id"] for c in chunks))[: top_k // 2]
        seen_sources = {c["source_id"] for c in chunks}

        candidate_linked: dict[str, float] = {}  # source_id → max link strength
        for source_id in direct_sources:
            for link in memory_store.get_links_for_source(source_id):
                linked_id = link["linked_id"]
                if linked_id and linked_id not in seen_sources:
                    # Keep strongest link if the same target appears via multiple sources
                    strength = link.get("strength", 1.0)
                    if strength > candidate_linked.get(linked_id, 0.0):
                        candidate_linked[linked_id] = strength

        # Sort by link strength, cap at MAX_LINKED_EXPANSIONS
        top_linked = sorted(candidate_linked, key=lambda k: candidate_linked[k], reverse=True)
        top_linked = top_linked[:MAX_LINKED_EXPANSIONS]

        if top_linked:
            linked_chunks = _fetch_linked_chunks(
                query_vector=query_vector,
                linked_source_ids=top_linked,
                qdrant=qdrant,
                memory_store=memory_store,
                compressed_collection=compressed_collection,
                raw_collection=raw_collection,
                entity_source_ids=entity_source_ids,
            )
            chunks.extend(linked_chunks)
            logger.debug(f"Graph expansion added {len(linked_chunks)} linked chunks")

    # --- Step 4a: Filter by date range if requested ---
    if date_after or date_before:
        date_filtered = []
        for chunk in chunks:
            raw_dt = chunk.get("created_at", "")
            if not raw_dt:
                date_filtered.append(chunk)  # keep if no date
                continue
            try:
                dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                if date_after and dt < date_after:
                    continue
                if date_before and dt > date_before:
                    continue
                date_filtered.append(chunk)
            except Exception:
                date_filtered.append(chunk)
        chunks = date_filtered

    # --- Step 4b: Deduplicate by source, filter suppressed, rank ---
    seen_sources_final: set[str] = set()
    result: list[dict] = []
    for chunk in sorted(chunks, key=lambda c: c["score"], reverse=True):
        if chunk["chunk_id"] in suppressed:
            continue
        src = chunk["source_id"]
        if src not in seen_sources_final:
            seen_sources_final.add(src)
            result.append(chunk)
        if len(result) >= top_k:
            break

    return result
