"""
ingest.py — Index all export data into vectors + generate Obsidian notes.

Steps:
  1. Auto-discovers every export directory in Brain/ (any folder with .json files)
  2. Detects the format of each export (Anthropic, OpenAI, or auto-detected via llama3.2)
  3. Parses conversations into normalized Document objects
  4. Chunks each document into embeddable pieces
  5. Embeds chunks using nomic-embed-text via Ollama
  6. Stores vectors in Qdrant + metadata in SQLite (idempotent — safe to re-run)
  7. Generates Obsidian .md notes with LLM summaries and category decisions

Usage:
    python ingest.py                 # full pipeline (index + notes)
    python ingest.py --index-only    # skip note generation
    python ingest.py --notes-only    # skip vector indexing
    python ingest.py --force         # re-index even if already processed
    python ingest.py --stats         # show current index stats and exit
    python ingest.py --no-llm        # disable llama3.2 (faster, keyword-only)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mindvault.config as cfg
from mindvault.config import DB_PATH, QDRANT_PATH, VAULT_MY_BRAIN
from mindvault.generate_notes import generate_notes
from src.adapters.anthropic import load_export as load_anthropic_export
from src.export_detector import find_export_dirs, load_conversations_from_dir
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.ingestion.store import BrainStore
from src.models import Document, PrivacyLevel, SourceType, VaultName

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def conversations_to_documents(convos: list[dict], export_batch: str) -> list[Document]:
    """Convert normalized conversation dicts → Document objects for the ingestion pipeline."""
    import hashlib
    from datetime import datetime, timezone

    def parse_iso(ts: str) -> datetime:
        if not ts:
            return datetime.now(timezone.utc)
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def extract_text(msg: dict) -> str:
        parts = []
        for block in msg.get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                parts.append(block["text"].strip())
        if not parts and msg.get("text", "").strip():
            parts.append(msg["text"].strip())
        return "\n".join(parts)

    docs = []
    for convo in convos:
        uuid = convo.get("uuid", "")
        name = convo.get("name", "Untitled")
        summary = convo.get("summary", "")
        created_at = parse_iso(convo.get("created_at", ""))
        updated_at = parse_iso(convo.get("updated_at", convo.get("created_at", "")))

        turns = []
        for msg in convo.get("chat_messages", []):
            text = extract_text(msg)
            speaker = msg.get("sender", "unknown")
            if text:
                turns.append(f"[{speaker.upper()}]\n{text}")

        if not turns:
            continue

        body = "\n\n".join(turns)
        if summary:
            body = f"Summary: {summary}\n\n---\n\n{body}"

        docs.append(Document(
            id=f"conv_{uuid}",
            source_type=SourceType.ANTHROPIC_CONVERSATION,
            vault=VaultName.NONE,
            privacy_level=PrivacyLevel.PUBLIC,
            title=name,
            body=body,
            created_at=created_at,
            updated_at=updated_at,
            conversation_uuid=uuid,
            metadata={
                "summary": summary,
                "export_batch": export_batch,
                "message_count": len(convo.get("chat_messages", [])),
            },
        ))
    return docs


def run_indexing(store: BrainStore, force: bool = False, folder: str | None = None) -> dict:
    """Index export directories into Qdrant + SQLite.

    If folder is given, index only that directory.
    Otherwise auto-discover all data-* dirs in BRAIN_DIR.
    """
    from pathlib import Path as _Path
    if folder:
        target = _Path(folder).resolve()
        if not target.exists():
            logger.error(f"Folder not found: {target}")
            return {"docs": 0, "chunks": 0}
        export_dirs = [target]
    else:
        export_dirs = find_export_dirs(cfg.BRAIN_DIR)
    if not export_dirs:
        logger.warning("No export directories found")
        return {"docs": 0, "chunks": 0}

    total_docs = 0
    total_chunks = 0

    for export_dir in export_dirs:
        batch_id = export_dir.name

        if not force and store.is_batch_ingested(batch_id):
            logger.info(f"Batch '{batch_id}' already indexed. Use --force to re-index.")
            continue

        logger.info(f"=== Indexing: {batch_id} ===")

        convos = load_conversations_from_dir(export_dir)
        if not convos:
            logger.warning(f"No conversations loaded from {batch_id}")
            continue

        docs = conversations_to_documents(convos, export_batch=batch_id)
        new_docs = [d for d in docs if not store.is_document_ingested(d.id)] if not force else docs

        if not new_docs:
            logger.info("All documents already indexed")
            store.record_batch(batch_id, "auto", str(export_dir), 0, 0)
            continue

        chunks = chunk_documents(new_docs)
        pairs = embed_chunks(chunks)
        store.upsert_chunks(pairs, export_batch=batch_id)
        store.record_batch(batch_id, "auto", str(export_dir), len(new_docs), len(chunks))

        total_docs += len(new_docs)
        total_chunks += len(chunks)
        logger.info(f"Indexed {len(new_docs)} docs → {len(chunks)} chunks from {batch_id}")

    return {"docs": total_docs, "chunks": total_chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain ingestion pipeline")
    parser.add_argument("folder", nargs="?", default=None, help="Specific export folder to ingest (default: auto-discover all)")
    parser.add_argument("--index-only", action="store_true", help="Skip note generation")
    parser.add_argument("--notes-only", action="store_true", help="Skip vector indexing")
    parser.add_argument("--force", action="store_true", help="Re-index already-processed batches")
    parser.add_argument("--stats", action="store_true", help="Show index stats and exit")
    parser.add_argument("--no-llm", action="store_true", help="Disable llama3.2 (keyword categorization only)")
    parser.add_argument("--in-memory", action="store_true", help="Use in-memory Qdrant (no persistence)")
    args = parser.parse_args()

    # Apply --no-llm flag to config at runtime
    if args.no_llm:
        cfg.USE_LLM_SUMMARIZATION = False
        cfg.USE_LLM_CATEGORIZATION = False
        logger.info("LLM disabled — using keyword rules only")

    qdrant_path = None if args.in_memory else QDRANT_PATH
    store = BrainStore(db_path=DB_PATH, qdrant_path=qdrant_path)

    if args.stats:
        stats = store.stats()
        print("\nBrain Index Statistics")
        print("=" * 40)
        for key, val in stats.items():
            if isinstance(val, dict):
                print(f"{key}:")
                for k, v in val.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {val}")
        return

    # ── Step 1: Vector indexing ───────────────────────────────────────────────
    if not args.notes_only:
        result = run_indexing(store, force=args.force, folder=args.folder)
        stats = store.stats()
        print(f"\nIndex: {stats['documents']} docs, {stats['chunks']} chunks, "
              f"{stats['qdrant_public_vectors']} public vectors")

    # ── Step 2: Generate Obsidian notes ───────────────────────────────────────
    if not args.index_only:
        print(f"\nGenerating Obsidian notes...\n")
        note_count = generate_notes(vault=VAULT_MY_BRAIN)
        print(f"\nDone. {note_count} notes written to '{VAULT_MY_BRAIN.name}/'.")


if __name__ == "__main__":
    main()
