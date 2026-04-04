"""
Ingestion pipeline orchestrator.

Usage:
    from src.pipeline import run_ingestion
    run_ingestion(export_dir=Path("data-2026-04-03-..."), store=store)

The pipeline is idempotent — running it twice on the same export batch
is safe; already-indexed content is skipped.
"""

import logging
from pathlib import Path

from src.adapters.anthropic import load_export
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.ingestion.store import BrainStore

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
