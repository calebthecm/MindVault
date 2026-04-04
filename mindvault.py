"""
mindvault.py — Unified CLI entry point for MindVault.

Usage:
    python mindvault.py                         # chat (default)
    python mindvault.py chat                    # interactive REPL
    python mindvault.py chat --resume           # resume last session
    python mindvault.py chat --resume <id>      # resume specific session
    python mindvault.py ingest                  # auto-discover and index all exports
    python mindvault.py ingest ./folder/        # index a specific folder
    python mindvault.py ingest --force          # re-index even if already processed
    python mindvault.py ingest --no-llm         # skip LLM calls (keyword rules only)
    python mindvault.py notes                   # regenerate Obsidian notes
    python mindvault.py setup                   # first-run configuration wizard
    python mindvault.py stats                   # show index and session statistics
    python mindvault.py sessions                # list resumable sessions
    python mindvault.py help [command]          # show help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

HELP_TEXT = {
    "main": """\

MindVault — local-first second brain

Commands:
  chat       Interactive REPL (default if no command given)
  ingest     Index export data into the brain
  notes      Regenerate Obsidian notes only
  setup      First-run configuration wizard
  stats      Show index and session statistics
  sessions   List resumable chat sessions
  help       Show this help, or: help <command>

Run: python mindvault.py <command> [options]
""",
    "chat": """\

chat — Talk to your second brain

Usage:
  python mindvault.py chat
  python mindvault.py chat --resume
  python mindvault.py chat --resume <session-id>

Options:
  --resume          Resume the last session
  --resume <id>     Resume a specific session by ID

Commands during a session:
  /quit, /exit      End session (compresses and saves automatically)
  /clear            Clear conversation history
  /private          Toggle private vault inclusion on/off
  /sources          Show sources used in last answer
  /remember <fact>  Save a specific fact to this session
""",
    "ingest": """\

ingest — Index data into the brain

Usage:
  python mindvault.py ingest                    # auto-discover all data-* dirs
  python mindvault.py ingest ./my-export/       # ingest a specific folder
  python mindvault.py ingest --force            # re-index even if already processed
  python mindvault.py ingest --no-llm           # skip LLM calls (keyword rules only)
  python mindvault.py ingest --stats            # show index stats and exit
  python mindvault.py ingest --notes-only       # regenerate notes without re-embedding
  python mindvault.py ingest --index-only       # embed only, skip note generation
""",
    "sessions": """\

sessions — List resumable chat sessions

Usage:
  python mindvault.py sessions

Shows ID, date, status, and preview for recent sessions.
Resume with:
  python mindvault.py chat --resume <session-id>
""",
    "stats": """\

stats — Show index and session statistics

Usage:
  python mindvault.py stats

Displays document count, chunk count, Qdrant vector counts, and session count.
""",
    "notes": """\

notes — Regenerate Obsidian notes

Usage:
  python mindvault.py notes

Re-runs the note generation pipeline without re-embedding.
Same as: python mindvault.py ingest --notes-only
""",
    "setup": """\

setup — First-run configuration wizard

Usage:
  python mindvault.py setup

Guides you through model selection, Ollama detection, dependency install,
directory creation, and .gitignore generation.
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
    sys.argv = ["ingest.py"] + args
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
        print("\nNo sessions yet.\n  Start one: python mindvault.py chat\n")
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


COMMANDS = {
    "chat": cmd_chat,
    "ingest": cmd_ingest,
    "notes": cmd_notes,
    "setup": cmd_setup,
    "stats": cmd_stats,
    "sessions": cmd_sessions,
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
        print("Run: python mindvault.py help")
        sys.exit(1)


if __name__ == "__main__":
    main()
