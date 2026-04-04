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
    /remember <fact> — save a specific fact to this session
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
from src.ingestion.store import COLLECTION_PUBLIC, COLLECTION_PRIVATE
from src.llm import chat_with_brain, compress_session
from src.memory.extractor import extract_entities_from_turn, deduplicate_entities
from src.memory.retriever import retrieve
from src.memory.store import MemoryStore
from src.sessions.manager import Session, load_session, load_last_session

SEPARATOR = "─" * 60


def embed_query(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": [text]},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def format_sources(chunks: list[dict]) -> str:
    seen = set()
    lines = []
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
    """Compress session and store in memory layer at session end."""
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
    print(f"[Session saved. {len(deduped)} entities captured. Preview: {preview}...]")


def run_chat(
    single_query: str | None = None,
    resume_session_id: str | None = None,
    resume_last: bool = False,
) -> None:
    # Check Ollama
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

    # Load or create session
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
                print("No previous sessions found. Starting new session.")
                session = Session(SESSIONS_DIR, model=LLM_MODEL)
            else:
                history = [{"role": t["role"], "content": t["content"]} for t in session.turns]
                session_entities = list(session.entities)
                print(f"\n[Resumed last session from {session.started_at[:19]}. {len(session.turns)} prior turns loaded.]")
        else:
            session = Session(SESSIONS_DIR, model=LLM_MODEL)

    print(f"\n{SEPARATOR}")
    print("  MINDVAULT CHAT")
    print(f"  Model: {LLM_MODEL}  |  Index: {QDRANT_PATH.name}")
    print(f"  Private vault: {'included' if include_private else 'excluded'} (/private to toggle)")
    print(f"  Commands: /quit  /clear  /private  /sources  /remember <fact>")
    print(SEPARATOR)

    def ask(query: str) -> None:
        nonlocal last_chunks, session_entities

        try:
            vector = embed_query(query)
        except Exception as e:
            print(f"\n[Error embedding query: {e}]")
            return

        chunks = retrieve(
            query_vector=vector,
            qdrant=qdrant,
            memory_store=memory_store,
            raw_collection=COLLECTION_PUBLIC,
            compressed_collection=COLLECTION_COMPRESSED_PUBLIC,
            top_k=CHAT_TOP_K,
            compressed_threshold=COMPRESSED_SCORE_THRESHOLD,
        )
        last_chunks = chunks

        if not chunks:
            print("\nBrain: Nothing relevant found for that query.")
            return

        response = chat_with_brain(
            query=query,
            context_chunks=chunks,
            model=LLM_MODEL,
            base_url=OLLAMA_BASE,
            conversation_history=history,
        )

        if response:
            print(f"\nBrain: {response}\n")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

            if session:
                session.add_turn("user", query)
                session.add_turn("assistant", response)

                # Streaming entity extraction per turn
                turn_entities = extract_entities_from_turn(
                    user_turn=query,
                    assistant_turn=response,
                    model=LLM_MODEL,
                    base_url=OLLAMA_BASE,
                )
                session_entities.extend(turn_entities)
                session.entities = session_entities
        else:
            print("\nBrain: [LLM did not respond — check if Ollama is running]\n")

    # Single-query mode (no session saved)
    if single_query:
        print(f"\nYou: {single_query}")
        ask(single_query)
        return

    # Interactive REPL
    print("\nStart asking. Your brain is ready.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            if session:
                _process_session_end(session, memory_store)
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Ending session.")
            if session:
                _process_session_end(session, memory_store)
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

        if user_input.lower().startswith("/remember "):
            fact_text = user_input[10:].strip()
            if fact_text and session:
                session_entities.append({"type": "fact", "name": fact_text, "value": ""})
                session.entities = session_entities
                print(f"[Remembered: {fact_text}]\n")
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
