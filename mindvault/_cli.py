"""
mindvault/_cli.py — Shared CLI logic for both `python mindvault.py` and
`python -m mindvault`. Do not run this file directly.
"""

import sys

HELP_TEXT = {
    "main": """\

MindVault — local-first second brain

Commands:
  chat          Interactive REPL (default if no command given)
  ingest        Index export data, Obsidian vaults, and build memory links
  notes         Regenerate Obsidian notes only
  consolidate   Merge near-duplicate compressed memories
  setup         First-run configuration wizard
  stats         Show index and session statistics
  sessions      List resumable chat sessions
  help          Show this help, or: help <command>

Run: mindvault <command> [options]
     python -m mindvault <command> [options]
""",
    "chat": """\

chat — Talk to your second brain

Usage:
  mindvault chat
  mindvault chat --resume
  mindvault chat --resume <session-id>

Options:
  --resume          Resume the last session
  --resume <id>     Resume a specific session by ID

Commands during a session:
  /web <query>      Search the web (DuckDuckGo, no API needed)
  /search <term>    Search memory without LLM
  /note <text>      Quick-capture a note
  /forget <topic>   Suppress matching chunks from retrieval
  /sources          Show sources used in last answer
  /remember <fact>  Save a specific fact to this session
  /private          Toggle private vault inclusion on/off
  /mode [name]      Show or switch reasoning mode
  /resume           Interactive session picker
  /clear            Clear conversation history
  /quit, /exit      End session
  /help             Show all commands
""",
    "ingest": """\

ingest — Index data into the brain

Usage:
  mindvault ingest                    # auto-discover exports + ingest vaults
  mindvault ingest ./my-export/       # ingest a specific folder only
  mindvault ingest --force            # re-index even if already processed
  mindvault ingest --no-llm           # skip LLM calls (keyword rules only)
  mindvault ingest --no-vaults        # skip Obsidian vault ingestion
  mindvault ingest --consolidate      # merge near-duplicate memories after indexing
  mindvault ingest --stats            # show index stats and exit
  mindvault ingest --notes-only       # regenerate notes without re-embedding
  mindvault ingest --index-only       # embed only, skip note generation
""",
    "sessions": """\

sessions — List resumable chat sessions

Usage:
  mindvault sessions

Resume with:
  mindvault chat --resume <session-id>
""",
    "stats": """\

stats — Show index and session statistics

Usage:
  mindvault stats
""",
    "notes": """\

notes — Regenerate Obsidian notes

Usage:
  mindvault notes
""",
    "setup": """\

setup — First-run configuration wizard

Usage:
  mindvault setup
""",
}


def cmd_help(args: list[str]) -> None:
    command = args[0] if args else "main"
    print(HELP_TEXT.get(command, HELP_TEXT["main"]))


def cmd_chat(args: list[str]) -> None:
    from mindvault.chat import run_chat

    resume_id = None
    resume_last = False
    i = 0
    while i < len(args):
        if args[i] == "--resume":
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                resume_id = args[i + 1]
                i += 2
            else:
                resume_last = True
                i += 1
        else:
            i += 1

    run_chat(resume_session_id=resume_id, resume_last=resume_last)


def cmd_ingest(args: list[str]) -> None:
    sys.argv = ["ingest"] + args
    from mindvault.ingest import main as ingest_main
    ingest_main()


def cmd_notes(args: list[str]) -> None:
    from mindvault.generate_notes import generate_notes
    from mindvault.config import VAULT_MY_BRAIN
    print("Regenerating Obsidian notes...")
    count = generate_notes(vault=VAULT_MY_BRAIN)
    print(f"Done. {count} notes written.")


def cmd_setup(args: list[str]) -> None:
    from mindvault.onboard import main as onboard_main
    onboard_main()


def cmd_stats(args: list[str]) -> None:
    from mindvault.config import DB_PATH, QDRANT_PATH, SESSIONS_DIR
    from src.ingestion.store import BrainStore
    from src.sessions.manager import list_sessions

    store = BrainStore(db_path=DB_PATH, qdrant_path=QDRANT_PATH)
    s = store.stats()

    print("\nBrain Index")
    print("=" * 40)
    print(f"Documents:    {s['documents']}")
    print(f"Chunks:       {s['chunks']}")
    print(f"Batches:      {s['batches']}")
    print(f"Public vecs:  {s['qdrant_public_vectors']}")
    print(f"Private vecs: {s['qdrant_private_vectors']}")
    if s.get("by_source_type"):
        print("\nBy source type:")
        for k, v in s["by_source_type"].items():
            print(f"  {k}: {v}")

    sessions = list_sessions(SESSIONS_DIR) if SESSIONS_DIR.exists() else []
    print(f"\nSessions:     {len(sessions)}")
    print()


def cmd_sessions(args: list[str]) -> None:
    from mindvault.config import SESSIONS_DIR
    from src.sessions.manager import list_sessions

    sessions = list_sessions(SESSIONS_DIR) if SESSIONS_DIR.exists() else []
    if not sessions:
        print("\nNo sessions yet.\n  Start one: mindvault chat\n")
        return

    sep = "─" * 92
    print(f"\n{'ID':<33} {'Date':<20} {'Turns':>5}  {'Status':<10} Preview")
    print(sep)
    for s in sessions[:20]:
        date = s["started_at"][:19].replace("T", " ")
        preview = s.get("preview", "")[:35]
        print(f"{s['session_id']:<33} {date:<20} {s.get('turn_count', 0):>5}  {s.get('status','raw'):<10} {preview}")
    if len(sessions) > 20:
        print(f"  ... and {len(sessions) - 20} more")
    print()


def cmd_consolidate(args: list[str]) -> None:
    dry_run = "--dry-run" in args
    from mindvault.config import (
        DB_PATH, QDRANT_PATH,
        COLLECTION_COMPRESSED_PUBLIC, COLLECTION_COMPRESSED_PRIVATE,
        LLM_MODEL, OLLAMA_BASE,
    )
    from qdrant_client import QdrantClient
    from src.memory.store import MemoryStore
    from src.memory.consolidator import run_consolidation

    if not QDRANT_PATH.exists():
        print("No index found. Run 'mindvault ingest' first.")
        sys.exit(1)

    qdrant = QdrantClient(path=str(QDRANT_PATH))
    memory_store = MemoryStore(
        db_path=DB_PATH,
        qdrant=qdrant,
        compressed_collections=(COLLECTION_COMPRESSED_PUBLIC, COLLECTION_COMPRESSED_PRIVATE),
    )

    if dry_run:
        print("\n[Dry run — no changes will be made]\n")

    print("Scanning for near-duplicate compressed memories...")
    stats = run_consolidation(
        memory_store=memory_store,
        qdrant=qdrant,
        collections=[COLLECTION_COMPRESSED_PUBLIC, COLLECTION_COMPRESSED_PRIVATE],
        dry_run=dry_run,
    )
    print(f"\nDone — checked {stats['checked']}, "
          f"merged {stats['merged']} groups, skipped {stats['skipped']}\n")


COMMANDS = {
    "chat": cmd_chat,
    "ingest": cmd_ingest,
    "notes": cmd_notes,
    "setup": cmd_setup,
    "stats": cmd_stats,
    "sessions": cmd_sessions,
    "consolidate": cmd_consolidate,
    "help": cmd_help,
}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        cmd_chat([])
        return

    command = args[0].lower()
    if command in COMMANDS:
        COMMANDS[command](args[1:])
    elif command.startswith("-"):
        cmd_chat(args)
    else:
        print(f"Unknown command: '{command}'")
        print("Run: mindvault help")
        sys.exit(1)
