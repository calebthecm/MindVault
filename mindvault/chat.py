"""
chat.py — Talk to your second brain.

Sessions are saved as gzip-compressed JSON. At session end, the LLM
compresses the conversation into a summary and extracts entities for
future retrieval. Sessions are resumable.

Usage:
    python chat.py                      # new session
    python chat.py --resume             # resume last session
    python chat.py --resume <id>        # resume specific session
    python chat.py "single question"    # single query (no session saved)

Commands during chat:
    /quit or /exit   — end session (compresses and saves)
    /clear           — clear conversation history
    /private         — toggle private vault on/off
    /sources         — show sources from last answer
    /resume          — interactive session picker (arrow keys + Enter)
    /remember <fact> — save a specific fact to this session
    /mode [name]     — print current mode or switch to named mode

Keys:
    Shift+Tab        — cycle through reasoning modes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from qdrant_client import QdrantClient

from mindvault.config import (
    CHAT_INCLUDE_PRIVATE,
    CHAT_TOP_K,
    COLLECTION_COMPRESSED_PUBLIC,
    COLLECTION_COMPRESSED_PRIVATE,
    COMPRESSED_SCORE_THRESHOLD,
    DB_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_BASE,
    QDRANT_PATH,
    SESSIONS_DIR,
)

from mindvault.council import run_council
from mindvault.modes import Mode, get_config
from mindvault.tui import BrainPrompt, print_bar, print_welcome, print_mode_switch, print_response, print_thinking
from mindvault.version import fetch_in_background
from src.ingestion.store import COLLECTION_PUBLIC, COLLECTION_PRIVATE
from src.llm import compress_session
from src.memory.extractor import extract_entities_from_turn, deduplicate_entities
from src.memory.retriever import retrieve
from src.memory.store import MemoryStore
from src.sessions.manager import Session, load_session, load_last_session


def embed_query(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": [text]},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def format_sources(chunks: list[dict]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for chunk in chunks:
        title = chunk.get("title", "Unknown")
        date = chunk.get("created_at", "")[:10]
        score = chunk.get("score", 0)
        layer = chunk.get("layer", "raw")
        if title not in seen:
            seen.add(title)
            lines.append(f"  - {title} ({date}) [score: {score}, layer: {layer}]")
    return "\n".join(lines) if lines else "  (none)"


def _process_session_end(session: Session, memory_store: MemoryStore) -> None:
    if not session.turns:
        session.save_and_index()
        return

    print("\n[Processing session — compressing and extracting memories...]")

    summary = compress_session(session.turns, model=LLM_MODEL, base_url=OLLAMA_BASE)
    if summary:
        session.summary = summary
        try:
            vector = embed_query(summary)
            memory_store.store_compressed(
                source_id=session.session_id,
                source_type="chat_session",
                summary=summary,
                chunk_ids=[],
                vector=vector,
                collection=COLLECTION_COMPRESSED_PUBLIC,
                metadata={"title": f"Chat {session.started_at[:10]}"},
            )
        except Exception as e:
            print(f"  [Warning: could not embed summary: {e}]")

    deduped = deduplicate_entities(session.entities)
    if deduped:
        memory_store.store_entities(deduped, source_id=session.session_id)
    session.entities = deduped
    session.status = "processed"
    session.save_and_index()

    preview = summary[:80] if summary else "none"
    print(f"[Session saved. {len(deduped)} entities captured. Preview: {preview}...]\n")


def run_chat(
    single_query: str | None = None,
    resume_session_id: str | None = None,
    resume_last: bool = False,
) -> None:
    # ── Version check (non-blocking) ──────────────────────────────────────────
    fetch_in_background()

    # ── Preflight ──────────────────────────────────────────────────────────────
    try:
        httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0).raise_for_status()
    except httpx.HTTPError:
        print(f"Cannot reach Ollama at {OLLAMA_BASE}. Run: ollama serve")
        sys.exit(1)

    if not QDRANT_PATH.exists():
        print("No index found. Run 'python mindvault.py ingest' first.")
        sys.exit(1)

    qdrant = QdrantClient(path=str(QDRANT_PATH))
    memory_store = MemoryStore(
        db_path=DB_PATH,
        qdrant=qdrant,
        compressed_collections=(COLLECTION_COMPRESSED_PUBLIC, COLLECTION_COMPRESSED_PRIVATE),
    )

    include_private = CHAT_INCLUDE_PRIVATE
    history: list[dict] = []
    last_chunks: list[dict] = []
    session_entities: list[dict] = []

    # ── Session setup ──────────────────────────────────────────────────────────
    session: Session | None = None
    if not single_query:
        if resume_session_id:
            session = load_session(SESSIONS_DIR, resume_session_id)
            if not session:
                print(f"Session '{resume_session_id}' not found.")
                sys.exit(1)
            history = [{"role": t["role"], "content": t["content"]} for t in session.turns]
            session_entities = list(session.entities)
            print(f"\n[Resumed session from {session.started_at[:19]}. {len(session.turns)} prior turns loaded.]")
        elif resume_last:
            session = load_last_session(SESSIONS_DIR)
            if not session:
                session = Session(SESSIONS_DIR, model=LLM_MODEL)
            else:
                history = [{"role": t["role"], "content": t["content"]} for t in session.turns]
                session_entities = list(session.entities)
                print(f"\n[Resumed last session from {session.started_at[:19]}. {len(session.turns)} prior turns loaded.]")
        else:
            session = Session(SESSIONS_DIR, model=LLM_MODEL)

    # ── Build prompt UI ────────────────────────────────────────────────────────
    def on_mode_change(mode: Mode) -> None:
        config = get_config(mode)
        print_mode_switch(config)

    prompt_ui = BrainPrompt(on_mode_change=on_mode_change)

    # ── First-run name prompt (if not set during onboarding) ──────────────────
    from mindvault import user_config
    if not user_config.get_name():
        print()
        try:
            entered = input("  What's your name? (shown in the welcome screen): ").strip()
        except (EOFError, KeyboardInterrupt):
            entered = ""
        if entered:
            user_config.set_name(entered)

    # ── Welcome box ────────────────────────────────────────────────────────────
    # Wait briefly for background version check to complete before rendering welcome box.
    # The daemon thread started at the top of run_chat — by now Ollama preflight and
    # Qdrant init have consumed enough time that the fetch is usually done, but we
    # give it up to 1.5 s more to be sure the release banner shows correctly.
    from mindvault.version import latest_version as _latest
    _latest(wait=True, timeout=1.5)

    from src.sessions.manager import list_sessions
    recent = list_sessions(SESSIONS_DIR)[:5] if SESSIONS_DIR.exists() else []
    print_welcome(
        sessions=recent,
        model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        work_dir=str(QDRANT_PATH.parent),
    )
    print("Shift+Tab cycles modes · /resume to switch sessions · /quit to exit\n")

    # ── Core ask function ──────────────────────────────────────────────────────
    def ask(query: str) -> None:
        nonlocal last_chunks, session_entities

        mode = prompt_ui.mode
        config = get_config(mode)

        try:
            vector = embed_query(query)
        except Exception as e:
            print(f"\n[Error embedding query: {e}]\n")
            return

        # EXPLORE mode: enable graph link expansion
        expand_links = mode == Mode.EXPLORE

        chunks = retrieve(
            query_vector=vector,
            qdrant=qdrant,
            memory_store=memory_store,
            raw_collection=COLLECTION_PUBLIC,
            compressed_collection=COLLECTION_COMPRESSED_PUBLIC,
            top_k=CHAT_TOP_K,
            compressed_threshold=COMPRESSED_SCORE_THRESHOLD,
            expand_links=expand_links,
        )
        if include_private:
            from src.ingestion.store import COLLECTION_PRIVATE
            private_chunks = retrieve(
                query_vector=vector,
                qdrant=qdrant,
                memory_store=memory_store,
                raw_collection=COLLECTION_PRIVATE,
                compressed_collection=COLLECTION_COMPRESSED_PRIVATE,
                top_k=CHAT_TOP_K,
                compressed_threshold=COMPRESSED_SCORE_THRESHOLD,
                expand_links=expand_links,
            )
            combined = chunks + private_chunks
            combined.sort(key=lambda c: c["score"], reverse=True)
            chunks = combined[:CHAT_TOP_K]
        last_chunks = chunks

        if not chunks:
            print_response("Brain", "Nothing relevant found in your brain for that query.")
            return

        # Show thinking lines for council modes
        if mode in (Mode.PLAN, Mode.DECIDE, Mode.DEBATE, Mode.REFLECT, Mode.EXPLORE):
            from mindvault.council import COUNCIL
            members_by_mode = {
                Mode.PLAN: ["The Analyst", "The Pragmatist", "The Devil"],
                Mode.DECIDE: [m.name for m in COUNCIL],
                Mode.DEBATE: ["The Visionary", "The Devil", "The Analyst"],
                Mode.REFLECT: ["The Historian", "The Visionary", "The Analyst"],
                Mode.EXPLORE: ["The Visionary", "The Historian", "The Pragmatist"],
            }
            for name in members_by_mode.get(mode, []):
                print_thinking(name)

        response = run_council(
            mode=mode,
            query=query,
            chunks=chunks,
            model=LLM_MODEL,
            base_url=OLLAMA_BASE,
            history=history if mode == Mode.CHAT else None,
        )

        if response:
            print_bar()
            print_response(f"Brain [{config.label}]", response)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

            if session:
                session.add_turn("user", query)
                session.add_turn("assistant", response)
                turn_entities = extract_entities_from_turn(
                    user_turn=query,
                    assistant_turn=response,
                    model=LLM_MODEL,
                    base_url=OLLAMA_BASE,
                )
                session_entities.extend(turn_entities)
                session.entities = session_entities
        else:
            print("\n[LLM did not respond — check if Ollama is running]\n")

    # ── Single-query mode ──────────────────────────────────────────────────────
    if single_query:
        ask(single_query)
        return

    # ── Interactive REPL ───────────────────────────────────────────────────────
    while True:
        user_input = prompt_ui.ask()

        # None = user quit via Ctrl+C / Ctrl+D
        if user_input is None:
            print("\nEnding session.")
            if session:
                try:
                    _process_session_end(session, memory_store)
                except KeyboardInterrupt:
                    print("\n[Session save interrupted — partial data may be lost]")
                    session.save_and_index()
            break

        if not user_input:
            continue

        # ── Slash commands ─────────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit"):
            print("Ending session.")
            if session:
                try:
                    _process_session_end(session, memory_store)
                except KeyboardInterrupt:
                    print("\n[Session save interrupted — partial data may be lost]")
                    session.save_and_index()
            break

        if user_input.lower() == "/clear":
            history.clear()
            print("[Conversation history cleared]\n")
            continue

        if user_input.lower() == "/private":
            include_private = not include_private
            state = "included" if include_private else "excluded"
            print(f"[Private vault now {state}]\n")
            continue

        if user_input.lower() == "/sources":
            if last_chunks:
                print(f"\nSources from last answer:\n{format_sources(last_chunks)}\n")
            else:
                print("[No previous query]\n")
            continue

        if user_input.lower() == "/resume":
            from mindvault.session_picker import pick_session
            all_sessions = list_sessions(SESSIONS_DIR) if SESSIONS_DIR.exists() else []
            if not all_sessions:
                print("[No sessions found]\n")
                continue
            # Save current session state before potentially switching
            if session and session.turns:
                session.save_and_index()
            chosen = pick_session(all_sessions)
            if chosen is None:
                print("[Cancelled]\n")
                continue
            loaded = load_session(SESSIONS_DIR, chosen["session_id"])
            if not loaded:
                print(f"[Could not load session {chosen['session_id'][:12]}...]\n")
                continue
            # Swap current session for the resumed one
            session = loaded
            history.clear()
            history.extend(
                {"role": t["role"], "content": t["content"]} for t in session.turns
            )
            session_entities = list(session.entities)
            date = session.started_at[:19].replace("T", " ")
            print(f"\n[Resumed session from {date}. {len(session.turns)} turns loaded.]\n")
            continue

        if user_input.lower().startswith("/remember "):
            fact_text = user_input[10:].strip()
            if fact_text and session:
                session_entities.append({"type": "fact", "name": fact_text, "value": ""})
                session.entities = session_entities
                print(f"[Remembered: {fact_text}]\n")
            continue

        if user_input.lower().startswith("/mode"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                config = get_config(prompt_ui.mode)
                print(f"[Current mode: {config.icon} {config.label} — {config.description}]\n")
            else:
                target = parts[1].strip().upper()
                try:
                    new_mode = Mode(target)
                    prompt_ui.current_mode = new_mode
                    on_mode_change(new_mode)
                except ValueError:
                    valid = ", ".join(m.value for m in Mode)
                    print(f"[Unknown mode '{target}'. Valid: {valid}]\n")
            continue

        ask(user_input)


if __name__ == "__main__":
    args = sys.argv[1:]
    resume_id = None
    resume_last = False
    single = None

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
            single = " ".join(args[i:])
            break

    run_chat(single_query=single, resume_session_id=resume_id, resume_last=resume_last)
