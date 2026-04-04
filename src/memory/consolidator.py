"""
src/memory/consolidator.py — Merge near-duplicate compressed memories.

Algorithm:
  1. Scroll all compressed memories from Qdrant in batches
  2. For each unprocessed memory, query its nearest neighbors
  3. If a neighbor's cosine similarity >= threshold, they are near-duplicates
  4. Merge: ask the LLM to write a single combined summary
  5. Store the merged summary, delete the originals

This reduces redundancy over time as the same topics get ingested multiple times.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient

from src.memory.store import MemoryStore
from src.llm import compress_session

logger = logging.getLogger(__name__)

# Cosine similarity above which two compressed memories are considered duplicates
DEFAULT_CONSOLIDATION_THRESHOLD = 0.92

# Maximum number of compressed memories to consolidate per run (prevents very long runs)
MAX_CONSOLIDATIONS_PER_RUN = 50


def _merge_summaries(
    summaries: list[str],
    model: str,
    base_url: str,
) -> Optional[str]:
    """Ask the LLM to combine multiple summaries into one concise summary."""
    from src.llm import _call_ollama

    combined = "\n\n---\n\n".join(summaries)
    prompt = f"""The following are multiple summaries of related conversations or documents.
Merge them into a single, concise summary that captures all key points without repetition.

Summaries:
{combined}

Write a unified summary in 2-5 sentences. Second person ("You discussed...", "You decided...").
No bullet points. Prose only. Under 200 words.

Merged summary:"""

    return _call_ollama(prompt, model=model, base_url=base_url, timeout=60.0)


def consolidate(
    memory_store: MemoryStore,
    qdrant: QdrantClient,
    collection: str,
    threshold: float = DEFAULT_CONSOLIDATION_THRESHOLD,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    dry_run: bool = False,
) -> dict:
    """
    Find and merge near-duplicate compressed memories in the given collection.

    Returns {"checked": int, "merged": int, "skipped": int}
    """
    from mindvault.config import LLM_MODEL, OLLAMA_BASE
    model = model or LLM_MODEL
    base_url = base_url or OLLAMA_BASE

    # Scroll all compressed memories
    scroll_result = qdrant.scroll(
        collection_name=collection,
        with_vectors=True,
        with_payload=True,
        limit=500,
    )
    all_points = scroll_result[0]

    if not all_points:
        logger.info(f"[Consolidator] No compressed memories in '{collection}'")
        return {"checked": 0, "merged": 0, "skipped": 0}

    logger.info(f"[Consolidator] Checking {len(all_points)} compressed memories for near-duplicates...")

    processed_ids: set[str] = set()
    merged = 0
    skipped = 0

    for point in all_points:
        pid = str(point.id)
        if pid in processed_ids:
            continue
        if merged >= MAX_CONSOLIDATIONS_PER_RUN:
            logger.info(f"[Consolidator] Reached max consolidations ({MAX_CONSOLIDATIONS_PER_RUN}), stopping")
            break

        if point.vector is None:
            skipped += 1
            continue

        # Find nearest neighbors
        try:
            neighbors = qdrant.query_points(
                collection_name=collection,
                query=point.vector,
                limit=5,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning(f"Query failed for {pid}: {e}")
            skipped += 1
            continue

        # Filter to high-similarity unprocessed neighbors (exclude self)
        duplicates = [
            h for h in neighbors
            if str(h.id) != pid
            and str(h.id) not in processed_ids
            and h.score >= threshold
        ]

        if not duplicates:
            continue

        # Build merge group: original + duplicates
        group = [point] + duplicates
        group_ids = [str(p.id) for p in group]
        summaries = [p.payload.get("summary", "") for p in group if p.payload]
        summaries = [s for s in summaries if s.strip()]

        if len(summaries) < 2:
            skipped += 1
            continue

        source_ids = [p.payload.get("source_id", "") for p in group if p.payload]
        source_type = (group[0].payload or {}).get("source_type", "")
        created_at = (group[0].payload or {}).get("created_at", "")

        logger.info(
            f"[Consolidator] Merging {len(group)} memories "
            f"(similarity >= {threshold}) for source: {source_ids[0]}"
        )

        if dry_run:
            logger.info(f"  [dry-run] Would merge: {group_ids}")
            processed_ids.update(group_ids)
            merged += 1
            continue

        # Generate merged summary
        merged_summary = _merge_summaries(summaries, model=model, base_url=base_url)
        if not merged_summary:
            logger.warning(f"[Consolidator] LLM merge failed for group {group_ids}, skipping")
            skipped += 1
            continue

        # Embed the merged summary
        try:
            from src.ingestion.embedder import embed_chunks
            from src.models import Chunk, SourceType, VaultName, PrivacyLevel
            from datetime import datetime, timezone
            dummy_chunk = Chunk(
                id="temp_consolidate",
                document_id="consolidate",
                source_type=SourceType.ANTHROPIC_CONVERSATION,
                vault=VaultName.NONE,
                privacy_level=PrivacyLevel.PUBLIC,
                text=merged_summary,
                index=0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            pairs = embed_chunks([dummy_chunk])
            if not pairs:
                skipped += 1
                continue
            _, merged_vector = pairs[0]
        except Exception as e:
            logger.error(f"[Consolidator] Embedding failed: {e}")
            skipped += 1
            continue

        # Delete the old compressed memories
        for old_id in group_ids:
            memory_store.delete_compressed(old_id, collection)

        # Store the merged summary
        primary_source_id = source_ids[0] if source_ids else "consolidated"
        memory_store.store_compressed(
            source_id=primary_source_id,
            source_type=source_type,
            summary=merged_summary,
            chunk_ids=group_ids,  # track which originals were merged
            vector=merged_vector,
            collection=collection,
            metadata={
                "created_at": created_at,
                "consolidated_from": group_ids,
                "consolidated": True,
            },
        )

        processed_ids.update(group_ids)
        merged += 1

    logger.info(
        f"[Consolidator] Done — checked {len(all_points)}, "
        f"merged {merged} groups, skipped {skipped}"
    )
    return {"checked": len(all_points), "merged": merged, "skipped": skipped}


def run_consolidation(
    memory_store: MemoryStore,
    qdrant: QdrantClient,
    collections: list[str],
    threshold: float = DEFAULT_CONSOLIDATION_THRESHOLD,
    dry_run: bool = False,
) -> dict:
    """Run consolidation over all given collections. Returns combined stats."""
    from mindvault.config import LLM_MODEL, OLLAMA_BASE

    total = {"checked": 0, "merged": 0, "skipped": 0}
    for collection in collections:
        stats = consolidate(
            memory_store=memory_store,
            qdrant=qdrant,
            collection=collection,
            threshold=threshold,
            model=LLM_MODEL,
            base_url=OLLAMA_BASE,
            dry_run=dry_run,
        )
        for k in total:
            total[k] += stats[k]
    return total
