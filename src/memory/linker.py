"""
src/memory/linker.py — Populate memory_links by finding related source IDs.

Link types:
  entity_overlap   — two sources mention the same named entities (projects, people, decisions)
  time_proximity   — two sources created within a short time window

Run after ingestion to build the link graph. Safe to re-run (INSERT OR IGNORE).
"""

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional

from src.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Entity overlap: minimum shared entities to create a link
MIN_SHARED_ENTITIES = 2

# Time proximity: window in days for considering sources "related by time"
TIME_WINDOW_DAYS = 7


def build_entity_links(
    memory_store: MemoryStore,
    min_shared: int = MIN_SHARED_ENTITIES,
) -> int:
    """
    Build entity_overlap links between source IDs that share >= min_shared entities.

    Uses the memory_entities table populated by extract_entities_from_turn().
    Returns the number of new links inserted.
    """
    entity_map = memory_store.get_entity_source_map()
    if not entity_map:
        logger.info("[Linker] No entities found — run ingestion first")
        return 0

    # Build source_id → set of entity names
    source_entities: dict[str, set[str]] = defaultdict(set)
    for entity_name, source_ids in entity_map.items():
        for source_id in source_ids:
            source_entities[source_id].add(entity_name)

    source_ids = list(source_entities.keys())
    if len(source_ids) < 2:
        return 0

    links = []
    for src_a, src_b in combinations(source_ids, 2):
        shared = source_entities[src_a] & source_entities[src_b]
        if len(shared) >= min_shared:
            # Strength = normalized overlap (Jaccard-like, capped at 1.0)
            union_size = len(source_entities[src_a] | source_entities[src_b])
            strength = round(len(shared) / union_size, 3) if union_size > 0 else 1.0
            links.append({
                "from_id": src_a,
                "to_id": src_b,
                "link_type": "entity_overlap",
                "strength": strength,
            })

    if not links:
        logger.info("[Linker] No entity overlap links found")
        return 0

    inserted = memory_store.store_links(links)
    logger.info(f"[Linker] Built {inserted} entity_overlap links from {len(source_ids)} sources")
    return inserted


def build_wikilink_links(
    memory_store: MemoryStore,
    wikilink_map: Optional[dict[str, list[str]]] = None,
) -> int:
    """
    Build wikilink links from Obsidian [[note]] references.

    wikilink_map: {source_id: [linked_note_title, ...]}
    If not provided, this is a no-op (wikilinks must be extracted at ingest time).
    Returns number of new links inserted.
    """
    if not wikilink_map:
        return 0

    # Build title → source_id reverse index
    # We need note titles to map back to source IDs
    # This requires access to the document store; caller must provide the map
    title_to_source: dict[str, str] = {}
    for source_id in wikilink_map:
        # Assume source_id encodes the note title for Obsidian notes
        title_to_source[source_id] = source_id  # placeholder

    links = []
    for source_id, linked_titles in wikilink_map.items():
        for title in linked_titles:
            target_id = title_to_source.get(title.lower())
            if target_id and target_id != source_id:
                links.append({
                    "from_id": source_id,
                    "to_id": target_id,
                    "link_type": "wikilink",
                    "strength": 1.0,
                })

    if not links:
        return 0

    inserted = memory_store.store_links(links)
    logger.info(f"[Linker] Built {inserted} wikilink links")
    return inserted


def run_linker(memory_store: MemoryStore, min_shared_entities: int = MIN_SHARED_ENTITIES) -> dict:
    """
    Run all link builders. Returns a summary dict.
    Safe to call multiple times — INSERT OR IGNORE prevents duplicates.
    """
    entity_links = build_entity_links(memory_store, min_shared=min_shared_entities)
    return {
        "entity_links": entity_links,
        "total": entity_links,
    }
