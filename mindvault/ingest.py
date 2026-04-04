"""
ingest.py — Index all data into vectors + generate Obsidian notes.

Handles any combination of content in a directory:
  - JSON conversation exports (Anthropic, OpenAI, or LLM-auto-detected format)
  - PDF files (.pdf)
  - Plain text and Markdown files (.txt, .md)
  - Obsidian vaults (My Brain / Private Brain)

Steps:
  1. Auto-discovers every qualifying directory in Brain/ (JSON, PDF, or text files)
  2. Detects JSON format via known parsers or llama3.2 for unknown formats
  3. Normalizes all content into Document objects
  4. Chunks documents into embeddable pieces
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
from mindvault.config import (
    COLLECTION_COMPRESSED_PUBLIC,
    COLLECTION_COMPRESSED_PRIVATE,
    DB_PATH,
    QDRANT_PATH,
    VAULT_MY_BRAIN,
    VAULT_PRIVATE,
)
from mindvault.generate_notes import generate_notes
from src.adapters.pdf import load_pdfs_from_dir
from src.export_detector import find_export_dirs, load_conversations_from_dir
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.ingestion.store import BrainStore
from src.memory.linker import run_linker
from src.memory.store import MemoryStore
from src.models import Document, PrivacyLevel, SourceType, VaultName
from src.pipeline import run_obsidian_ingestion

# Sub-module loggers are noisy at INFO level — only surface warnings and errors.
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("ingest")

SEP = "─" * 52


def _head(msg: str) -> None:
    print(f"\n{SEP}")
    print(f"  {msg}")
    print(SEP)


def _step(msg: str) -> None:
    print(f"\n  {msg}")


def _ok(label: str, docs: int, chunks: int) -> None:
    print(f"    {label:<16} {docs:>3} docs  →  {chunks:>4} chunks")


def _warn(msg: str) -> None:
    print(f"  [!] {msg}")


def _err(msg: str) -> None:
    print(f"\n  [ERROR] {msg}\n")


def load_plain_text_from_dir(
    directory: Path,
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC,
) -> list[Document]:
    """Load .txt and standalone .md files from a directory as plain-text Documents."""
    import hashlib
    from datetime import datetime, timezone

    docs = []
    for path in sorted(directory.glob("*.txt")) + sorted(directory.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            _warn(f"Could not read '{path.name}': {e}")
            continue
        if not text:
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        doc_id = "txt_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        docs.append(Document(
            id=doc_id,
            source_type=SourceType.PLAIN_TEXT,
            vault=VaultName.NONE,
            privacy_level=privacy_level,
            title=path.stem,
            body=text,
            created_at=mtime,
            updated_at=mtime,
            metadata={"file_path": str(path)},
        ))
    return docs


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


def run_vault_indexing(store: BrainStore, force: bool = False) -> dict:
    """Ingest My Brain and Private Brain Obsidian vaults."""
    total = {"docs": 0, "chunks": 0}
    vaults = [
        (VAULT_MY_BRAIN,  VaultName.MY_BRAIN,      PrivacyLevel.PUBLIC,  "My Brain"),
        (VAULT_PRIVATE,   VaultName.PRIVATE_BRAIN,  PrivacyLevel.PRIVATE, "Private Brain"),
    ]
    for vault_dir, vault_name, privacy_level, label in vaults:
        if not vault_dir.exists():
            continue
        result = run_obsidian_ingestion(
            vault_dir=vault_dir,
            store=store,
            vault_name=vault_name,
            privacy_level=privacy_level,
            force=force,
        )
        d = result.get("docs_processed", 0)
        c = result.get("chunks_created", 0)
        if d:
            _ok(label, d, c)
        total["docs"]   += d
        total["chunks"] += c
    return total


def _index_doc_list(
    docs: list[Document],
    store: BrainStore,
    batch_id: str,
    force: bool,
) -> tuple[int, int]:
    """Embed and store a list of Documents. Returns (doc_count, chunk_count)."""
    new_docs = [d for d in docs if not store.is_document_ingested(d.id)] if not force else docs
    if not new_docs:
        return 0, 0
    chunks = chunk_documents(new_docs)
    pairs = embed_chunks(chunks)
    store.upsert_chunks(pairs, export_batch=batch_id)
    return len(new_docs), len(chunks)


def run_indexing(store: BrainStore, force: bool = False, folder: str | None = None) -> dict:
    """Index a directory into Qdrant + SQLite.

    Handles any combination of:
    - JSON conversation exports (Anthropic, OpenAI, or LLM-detected generic format)
    - PDF files
    - Plain text and Markdown files (.txt, .md)

    If folder is given, index only that directory.
    Otherwise auto-discover all qualifying dirs in BRAIN_DIR.
    """
    from pathlib import Path as _Path
    if folder:
        target = _Path(folder).resolve()
        if not target.exists():
            _err(f"Folder not found: {target}")
            return {"docs": 0, "chunks": 0}
        export_dirs = [target]
    else:
        export_dirs = find_export_dirs(cfg.BRAIN_DIR)

    if not export_dirs:
        return {"docs": 0, "chunks": 0}

    print(f"  Found {len(export_dirs)} folder(s) to scan")
    total_docs = 0
    total_chunks = 0

    for i, export_dir in enumerate(export_dirs, 1):
        batch_id = export_dir.name

        if not force and store.is_batch_ingested(batch_id):
            print(f"  [{i}/{len(export_dirs)}] {batch_id}  (already indexed — skip with --force to re-run)")
            continue

        print(f"\n  [{i}/{len(export_dirs)}] {batch_id}")
        batch_docs = 0
        batch_chunks = 0

        # ── JSON conversation exports ─────────────────────────────────────────
        convos = load_conversations_from_dir(export_dir)
        if convos:
            conv_docs = conversations_to_documents(convos, export_batch=batch_id)
            d, c = _index_doc_list(conv_docs, store, batch_id, force)
            batch_docs += d
            batch_chunks += c
            if d:
                _ok("Conversations", d, c)

        # ── PDF files ─────────────────────────────────────────────────────────
        pdf_docs = load_pdfs_from_dir(export_dir)
        if pdf_docs:
            d, c = _index_doc_list(pdf_docs, store, batch_id, force)
            batch_docs += d
            batch_chunks += c
            if d:
                _ok("PDFs", d, c)

        # ── Plain text / Markdown files ───────────────────────────────────────
        txt_docs = load_plain_text_from_dir(export_dir)
        if txt_docs:
            d, c = _index_doc_list(txt_docs, store, batch_id, force)
            batch_docs += d
            batch_chunks += c
            if d:
                _ok("Text files", d, c)

        if not convos and not pdf_docs and not txt_docs:
            _warn(f"No recognizable content found in '{batch_id}'")
            _warn("Expected: .json exports, .pdf files, or .txt/.md files")
            continue

        if batch_docs == 0:
            print(f"    (nothing new to index)")
            store.record_batch(batch_id, "auto", str(export_dir), 0, 0)
            continue

        store.record_batch(batch_id, "auto", str(export_dir), batch_docs, batch_chunks)
        total_docs += batch_docs
        total_chunks += batch_chunks

    return {"docs": total_docs, "chunks": total_chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description="MindVault ingestion pipeline")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Specific folder to ingest (default: auto-discover all)")
    parser.add_argument("--index-only",   action="store_true", help="Skip Obsidian note generation")
    parser.add_argument("--notes-only",   action="store_true", help="Skip vector indexing")
    parser.add_argument("--no-vaults",    action="store_true", help="Skip Obsidian vault ingestion")
    parser.add_argument("--force",        action="store_true", help="Re-index already-processed content")
    parser.add_argument("--stats",        action="store_true", help="Show index stats and exit")
    parser.add_argument("--no-llm",       action="store_true", help="Disable LLM calls (faster, keyword-only)")
    parser.add_argument("--consolidate",  action="store_true",
                        help="Merge near-duplicate memories after indexing")
    parser.add_argument("--in-memory",    action="store_true", help="Use in-memory Qdrant (no persistence)")
    args = parser.parse_args()

    if args.no_llm:
        cfg.USE_LLM_SUMMARIZATION = False
        cfg.USE_LLM_CATEGORIZATION = False

    qdrant_path = None if args.in_memory else QDRANT_PATH

    try:
        store = BrainStore(db_path=DB_PATH, qdrant_path=qdrant_path)
    except Exception as e:
        _err(f"Could not open the brain index: {e}")
        _err("Try running: python mindvault.py ingest --in-memory  to test without storage")
        sys.exit(1)

    # ── Stats only ────────────────────────────────────────────────────────────
    if args.stats:
        stats = store.stats()
        _head("Brain Index")
        print(f"  Documents    {stats['documents']}")
        print(f"  Chunks       {stats['chunks']}")
        print(f"  Batches      {stats['batches']}")
        print(f"  Public vecs  {stats['qdrant_public_vectors']}")
        print(f"  Private vecs {stats['qdrant_private_vectors']}")
        if stats.get("by_source_type"):
            print("\n  By source type:")
            for k, v in stats["by_source_type"].items():
                print(f"    {k}: {v}")
        print()
        return

    _head("MindVault — Indexing")
    if args.no_llm:
        print("  (LLM disabled — using keyword rules only)\n")

    # ── Step 1: Vector indexing ───────────────────────────────────────────────
    if not args.notes_only:
        result = run_indexing(store, force=args.force, folder=args.folder)
        if result["docs"] == 0 and result["chunks"] == 0:
            print("  Nothing new to index.")
            print("  Drop a folder with .json, .pdf, or .txt files into Brain/ and re-run.")

        # Obsidian vaults (skipped when a specific folder is targeted)
        if not args.no_vaults and not args.folder:
            vaults_exist = VAULT_MY_BRAIN.exists() or VAULT_PRIVATE.exists()
            if vaults_exist:
                _step("Obsidian vaults")
                vault_result = run_vault_indexing(store, force=args.force)
                result["docs"]   += vault_result["docs"]
                result["chunks"] += vault_result["chunks"]

        # ── Step 2: Memory links ──────────────────────────────────────────────
        memory_store = MemoryStore(
            db_path=DB_PATH,
            qdrant=store.qdrant,
            compressed_collections=(COLLECTION_COMPRESSED_PUBLIC, COLLECTION_COMPRESSED_PRIVATE),
        )
        _step("Building memory links...")
        try:
            link_stats = run_linker(memory_store)
            edges = link_stats.get("entity_links", 0)
            print(f"  done  ({edges} edges)")
        except Exception as e:
            _warn(f"Link building failed: {e}")

        # ── Step 3: Optional consolidation ────────────────────────────────────
        if args.consolidate:
            _step("Consolidating duplicate memories...")
            try:
                from src.memory.consolidator import run_consolidation
                c_stats = run_consolidation(
                    memory_store=memory_store,
                    qdrant=store.qdrant,
                    collections=[COLLECTION_COMPRESSED_PUBLIC, COLLECTION_COMPRESSED_PRIVATE],
                )
                print(f"  done  (checked {c_stats['checked']}, "
                      f"merged {c_stats['merged']}, skipped {c_stats['skipped']})")
            except Exception as e:
                _warn(f"Consolidation failed: {e}")

        # ── Summary ───────────────────────────────────────────────────────────
        stats = store.stats()
        print(f"\n{SEP}")
        print(f"  {stats['documents']} docs  ·  {stats['chunks']} chunks  ·  "
              f"{stats['qdrant_public_vectors']} public  ·  "
              f"{stats['qdrant_private_vectors']} private vectors")
        print(SEP)

    # ── Step 4: Generate Obsidian notes ───────────────────────────────────────
    if not args.index_only:
        _step(f"Generating notes in '{VAULT_MY_BRAIN.name}/'...")
        try:
            note_count = generate_notes(vault=VAULT_MY_BRAIN)
            print(f"  done  ({note_count} notes written)")
        except Exception as e:
            _warn(f"Note generation failed: {e}")
            _warn("Your index is fine — notes are optional. Re-run with: python mindvault.py notes")

    print(f"\n  All done. Start chatting: python mindvault.py chat\n")


if __name__ == "__main__":
    main()
