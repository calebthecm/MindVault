"""
Ingestion pipeline orchestrator.

Usage:
    from src.pipeline import run_ingestion, run_obsidian_ingestion
    run_ingestion(export_dir=Path("data-2026-04-03-..."), store=store)
    run_obsidian_ingestion(vault_dir=Path("My Brain"), store=store)

Both pipelines are idempotent — running them twice on the same content
is safe; already-indexed content is skipped.
"""

import logging
from pathlib import Path

from src.adapters.anthropic import load_export
from src.adapters.obsidian import load_vault
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.ingestion.store import BrainStore
from src.models import PrivacyLevel, VaultName

logger = logging.getLogger(__name__)


def run_ingestion(export_dir: Path, store: BrainStore, force: bool = False) -> dict:
    """
    Run the full ingestion pipeline for one Anthropic export batch.

    Returns a summary dict with counts.
    """
    export_dir = Path(export_dir)
    batch_id = export_dir.name

    if not force and store.is_batch_ingested(batch_id):
        logger.info(f"Batch '{batch_id}' already ingested. Use force=True to re-index.")
        return {"skipped": True, "batch_id": batch_id}

    logger.info(f"=== Ingesting batch: {batch_id} ===")

    # Step 1: Parse
    docs = load_export(export_dir)
    if not docs:
        logger.warning("No documents found in export. Check the directory contents.")
        return {"docs": 0, "chunks": 0}

    # Step 2: Filter already-ingested docs (per-document dedup within a batch)
    if not force:
        new_docs = [d for d in docs if not store.is_document_ingested(d.id)]
        skipped = len(docs) - len(new_docs)
        if skipped:
            logger.info(f"Skipping {skipped} already-indexed documents")
        docs = new_docs

    if not docs:
        logger.info("All documents already indexed.")
        return {"skipped": True, "batch_id": batch_id}

    # Step 3: Chunk
    chunks = chunk_documents(docs)

    # Step 4: Embed
    chunk_vector_pairs = embed_chunks(chunks)

    # Step 5: Store
    store.upsert_chunks(chunk_vector_pairs, export_batch=batch_id)
    store.record_batch(
        batch_id=batch_id,
        source="anthropic",
        path=str(export_dir),
        doc_count=len(docs),
        chunk_count=len(chunks),
    )

    summary = {
        "batch_id": batch_id,
        "docs_processed": len(docs),
        "chunks_created": len(chunks),
        "chunks_embedded": len(chunk_vector_pairs),
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


def run_obsidian_ingestion(
    vault_dir: Path,
    store: BrainStore,
    vault_name: VaultName = VaultName.MY_BRAIN,
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC,
    force: bool = False,
    skip_stale: bool = False,
) -> dict:
    """
    Run the ingestion pipeline for one Obsidian vault.

    vault_name / privacy_level are inferred from the directory name if not provided:
      - "My Brain" or "my_brain"   → MY_BRAIN / PUBLIC
      - "Private Brain" or similar → PRIVATE_BRAIN / PRIVATE

    Returns a summary dict with counts.
    """
    vault_dir = Path(vault_dir)
    vault_id = f"obsidian:{vault_dir.name}"

    # Auto-infer privacy from vault name if caller uses defaults
    lower = vault_dir.name.lower()
    if "private" in lower:
        vault_name = VaultName.PRIVATE_BRAIN
        privacy_level = PrivacyLevel.PRIVATE
    elif vault_name == VaultName.MY_BRAIN:
        privacy_level = PrivacyLevel.PUBLIC

    if not force and store.is_batch_ingested(vault_id):
        logger.info(f"Vault '{vault_dir.name}' already ingested. Use force=True to re-index.")
        return {"skipped": True, "vault": vault_dir.name}

    logger.info(f"=== Ingesting Obsidian vault: {vault_dir.name} ===")

    # Step 1: Parse
    docs = load_vault(vault_dir, vault_name=vault_name, privacy_level=privacy_level, skip_stale=skip_stale)
    if not docs:
        logger.warning(f"No notes found in {vault_dir.name}")
        return {"docs": 0, "chunks": 0}

    # Step 2: Filter already-ingested docs
    if not force:
        new_docs = [d for d in docs if not store.is_document_ingested(d.id)]
        skipped = len(docs) - len(new_docs)
        if skipped:
            logger.info(f"Skipping {skipped} already-indexed notes")
        docs = new_docs

    if not docs:
        logger.info("All notes already indexed.")
        return {"skipped": True, "vault": vault_dir.name}

    # Step 3: Chunk
    chunks = chunk_documents(docs)

    # Step 4: Embed
    chunk_vector_pairs = embed_chunks(chunks)

    # Step 5: Store
    store.upsert_chunks(chunk_vector_pairs, export_batch=vault_id)
    store.record_batch(
        batch_id=vault_id,
        source="obsidian",
        path=str(vault_dir),
        doc_count=len(docs),
        chunk_count=len(chunks),
    )

    summary = {
        "vault": vault_dir.name,
        "docs_processed": len(docs),
        "chunks_created": len(chunks),
        "chunks_embedded": len(chunk_vector_pairs),
    }
    logger.info(f"Vault ingestion complete: {summary}")
    return summary


def discover_export_dirs(base_dir: Path) -> list[Path]:
    """
    Find all Anthropic export batch directories under base_dir.
    Pattern: data-{timestamp}-batch-{n}/
    """
    base_dir = Path(base_dir)
    dirs = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("data-") and "batch" in d.name
    ])
    logger.info(f"Found {len(dirs)} export batch(es) in {base_dir}")
    return dirs
