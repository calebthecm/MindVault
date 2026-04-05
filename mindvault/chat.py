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
    /search <term>   — keyword search without LLM, shows raw scored results
    /note <text>     — quick-capture a note (saved to notes/, indexed on next ingest)
    /forget <topic>  — suppress matching chunks from future chat and search

Keys:
    Shift+Tab        — cycle through reasoning modes
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
from datetime import datetime, timezone

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
    SUGGEST_FOLLOWUPS,
    WEB_SEARCH_AUTO_THRESHOLD,
    WEB_SEARCH_MAX_RESULTS,
    WRITE_SESSIONS_TO_VAULT,
)

from mindvault.council import run_council
from mindvault.modes import Mode, get_config
from mindvault.tui import (
    BakingAnimation,
    BrainPrompt,
    print_bar,
    print_markdown_response,
    print_mode_switch,
    print_response,
    print_thinking,
    print_welcome,
)
from mindvault.version import fetch_in_background
from src.ingestion.store import COLLECTION_PUBLIC, COLLECTION_PRIVATE
from src.llm import compress_session
from src.memory.extractor import extract_entities_from_turn, deduplicate_entities
from src.memory.retriever import retrieve
from src.memory.store import MemoryStore
from mindvault.sessions.manager import Session, load_session, load_last_session


def _shrink_for_embedding(text: str, max_chars: int) -> str:
    """Trim oversized queries while keeping the start and end intact."""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars < 200:
        return normalized[:max_chars]
    head = int(max_chars * 0.7)
    tail = max_chars - head - len("\n...\n")
    return normalized[:head] + "\n...\n" + normalized[-max(tail, 0):]


def _extract_pasted_blocks(text: str) -> list[str]:
    pattern = re.compile(r"User pasted content \(\d+ lines\):\n```text\n([\s\S]*?)\n```")
    return [block.strip() for block in pattern.findall(text)]


def condensed_retrieval_query(text: str, max_chars: int = 2400) -> str:
    """
    Build a retrieval-focused query from a potentially huge user message.

    Keep the user's explicit ask, then compress any large pasted payloads into a
    short search surrogate so vector search stays robust and semantically useful.
    """
    normalized = text.strip()
    if len(normalized) <= max_chars and len(normalized.splitlines()) <= 24:
        return normalized

    pasted_blocks = _extract_pasted_blocks(normalized)
    outside = re.sub(r"User pasted content \(\d+ lines\):\n```text\n[\s\S]*?\n```", " ", normalized)
    outside = re.sub(r"\s+", " ", outside).strip()

    parts: list[str] = []
    if outside:
        parts.append(outside[:900])

    for idx, block in enumerate(pasted_blocks[:2], start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        excerpt = lines[:8]
        if len(lines) > 12:
            excerpt += lines[-4:]
        excerpt = [line[:180] for line in excerpt]
        summary = " | ".join(excerpt)
        parts.append(f"Pasted block {idx} summary: {summary}")

    if not parts:
        return _shrink_for_embedding(normalized, max_chars)

    condensed = "\n".join(parts)
    return _shrink_for_embedding(condensed, max_chars)


def embed_query(text: str) -> list[float]:
    from mindvault.config import LLM_API_KEY, LLM_BACKEND

    if LLM_BACKEND == "ollama":
        candidates = [text]
        for max_chars in (12000, 8000, 4000, 2000):
            shrunk = _shrink_for_embedding(text, max_chars)
            if shrunk not in candidates:
                candidates.append(shrunk)

        last_error: Exception | None = None
        for candidate in candidates:
            payloads = [
                {"model": EMBEDDING_MODEL, "input": [candidate]},
                {"model": EMBEDDING_MODEL, "input": candidate},
            ]
            for payload in payloads:
                try:
                    resp = httpx.post(
                        f"{OLLAMA_BASE}/api/embed",
                        json=payload,
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    embeddings = resp.json()["embeddings"]
                    if not embeddings:
                        raise RuntimeError("Ollama returned no embeddings")
                    return embeddings[0]
                except (httpx.HTTPError, KeyError, RuntimeError) as exc:
                    last_error = exc
                    if isinstance(exc, httpx.HTTPStatusError):
                        detail = exc.response.text.lower()
                        if "context length" not in detail and "input length exceeds" not in detail:
                            break
                    continue
        if isinstance(last_error, httpx.HTTPStatusError):
            detail = last_error.response.text[:400]
            raise RuntimeError(
                f"Ollama embedding failed for model '{EMBEDDING_MODEL}'. "
                f"Response: {detail}"
            ) from last_error
        raise RuntimeError(
            f"Ollama embedding failed for model '{EMBEDDING_MODEL}'. "
            "Make sure the model is pulled and Ollama is running."
        ) from last_error

    resp = httpx.post(
        f"{OLLAMA_BASE}/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text},
        headers={"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    if score >= 0.6:
        return "MED"
    return "LOW"


def format_sources(chunks: list[dict]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for chunk in chunks:
        title = chunk.get("title", "Unknown")
        date = chunk.get("created_at", "")[:10]
        score = chunk.get("score", 0)
        layer = chunk.get("layer", "raw")
        conf = _confidence_label(score)
        if title not in seen:
            seen.add(title)
            lines.append(f"  [{conf}] {title} ({date}) [{layer}]")
    return "\n".join(lines) if lines else "  (none)"


def _print_transcript(turns: list[dict], max_turns: int = 20) -> None:
    """Print prior session turns to screen so the user can read them on resume."""
    if not turns:
        return
    total = len(turns)
    visible = turns[-max_turns:] if total > max_turns else turns
    skipped = total - len(visible)
    print()
    print_bar()
    if skipped:
        print(f"  [... {skipped} earlier turn(s) not shown — use /clear to reset context]\n")
    for turn in visible:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        label = "You" if role == "user" else "Brain"
        print_response(label, content)
    print_bar()
    print()


def _process_session_end(session: Session, memory_store: MemoryStore) -> None:
    if not session.turns:
        # Nothing happened — discard silently, don't pollute the session list
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

    # Optionally write summary back to Obsidian vault
    if summary and WRITE_SESSIONS_TO_VAULT:
        try:
            from mindvault.config import VAULT_MY_BRAIN
            vault_sessions = VAULT_MY_BRAIN / "MindVault Sessions"
            vault_sessions.mkdir(parents=True, exist_ok=True)
            date_str = session.started_at[:10]
            note_path = vault_sessions / f"{date_str} Session.md"
            tags = " ".join(
                f"#{e['name'].replace(' ', '-')}"
                for e in deduped[:5]
                if e.get("name")
            )
            content = f"# Session {date_str}\n\n{summary}\n\n{tags}\n"
            # Append if multiple sessions on same day
            if note_path.exists():
                content = "\n---\n\n" + content
                note_path.write_text(note_path.read_text() + content)
            else:
                note_path.write_text(content)
            print(f"  [Wrote session note to My Brain/MindVault Sessions/{note_path.name}]")
        except Exception as e:
            print(f"  [Warning: could not write vault note: {e}]")

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
            history = [{"role": t["role"], "content": t["content"]} for t in session.turns[-40:]]
            session_entities = list(session.entities)
            _print_transcript(session.turns)
            print(f"[Resumed session from {session.started_at[:19].replace('T', ' ')} — {len(session.turns) // 2} exchange(s)]\n")
        elif resume_last:
            session = load_last_session(SESSIONS_DIR)
            if not session:
                session = Session(SESSIONS_DIR, model=LLM_MODEL)
            else:
                history = [{"role": t["role"], "content": t["content"]} for t in session.turns[-40:]]
                session_entities = list(session.entities)
                _print_transcript(session.turns)
                print(f"[Resumed last session from {session.started_at[:19].replace('T', ' ')} — {len(session.turns) // 2} exchange(s)]\n")
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

    from mindvault.sessions.manager import list_sessions
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

        retrieval_query = condensed_retrieval_query(query)

        # Parse time filter before embedding so the model sees a clean query
        from src.memory.time_filter import parse_time_filter
        embed_text, date_after, date_before = parse_time_filter(retrieval_query)
        if date_after:
            after_str = date_after.strftime("%Y-%m-%d")
            before_str = (date_before or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
            print(f"  [Time filter: {after_str} → {before_str}]\n")

        try:
            vector = embed_query(embed_text)
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
            date_after=date_after,
            date_before=date_before,
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
                date_after=date_after,
                date_before=date_before,
            )
            combined = chunks + private_chunks
            combined.sort(key=lambda c: c["score"], reverse=True)
            chunks = combined[:CHAT_TOP_K]
        last_chunks = chunks

        # Auto web search when memory is weak or empty
        web_used = False
        if WEB_SEARCH_AUTO_THRESHOLD > 0:
            top_score = max((c["score"] for c in chunks), default=0.0)
            if top_score < WEB_SEARCH_AUTO_THRESHOLD:
                from src.adapters.web import web_search
                web_chunks = web_search(embed_text, max_results=WEB_SEARCH_MAX_RESULTS)
                if web_chunks:
                    chunks = web_chunks + chunks
                    last_chunks = chunks
                    web_used = True
                    print(f"  [Low memory confidence ({top_score:.2f}) — searching the web...]\n")

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

        # CHAT mode: animate while generating, render full markdown at end
        if mode == Mode.CHAT:
            print_bar()
            sys.stdout.write(f"\nBrain [{config.label}]: ")
            sys.stdout.flush()
            streamed_any = False
            def _on_token(t: str) -> None:
                nonlocal streamed_any
                streamed_any = True
                sys.stdout.write(t)
                sys.stdout.flush()
            response = run_council(
                mode=mode,
                query=query,
                chunks=chunks,
                model=LLM_MODEL,
                base_url=OLLAMA_BASE,
                history=history,
                on_token=_on_token,
            )
            if streamed_any:
                sys.stdout.write("\n\n")
                sys.stdout.flush()
        else:
            # Council modes: show thinking indicators then wait for full response
            anim = BakingAnimation()
            anim.start()
            response = run_council(
                mode=mode,
                query=query,
                chunks=chunks,
                model=LLM_MODEL,
                base_url=OLLAMA_BASE,
            )
            elapsed = anim.stop()

        if response:
            label = f"Brain [{config.label}]"
            if mode == Mode.CHAT:
                if not streamed_any:
                    print_bar()
                    print_markdown_response(label, response)
            else:
                print_bar()
                print_markdown_response(label, response)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

            if session:
                session.add_turn("user", query)
                session.add_turn("assistant", response)

                # Extract entities in the background — don't block the next prompt
                _q, _a = query, response
                def _extract_bg(q=_q, a=_a) -> None:
                    entities = extract_entities_from_turn(
                        user_turn=q,
                        assistant_turn=a,
                        model=LLM_MODEL,
                        base_url=OLLAMA_BASE,
                    )
                    if entities:
                        session_entities.extend(entities)
                        session.entities = session_entities

                threading.Thread(target=_extract_bg, daemon=True).start()

            # Follow-up suggestions (CHAT mode only, background thread)
            if mode == Mode.CHAT and SUGGEST_FOLLOWUPS:
                _rq, _rr = query, response
                def _suggest_bg(q=_rq, r=_rr) -> None:
                    from src.llm import suggest_followups
                    suggestions = suggest_followups(q, r, model=LLM_MODEL, base_url=OLLAMA_BASE)
                    if suggestions:
                        def _print_suggestions() -> None:
                            print("  Related:")
                            for s in suggestions:
                                print(f"    · {s}")
                            print()
                        _print_suggestions()
                threading.Thread(target=_suggest_bg, daemon=True).start()
        else:
            print("\n[LLM did not respond — check if Ollama is running]\n")

    # ── Single-query mode ──────────────────────────────────────────────────────
    if single_query:
        ask(single_query)
        return

    # ── Interactive REPL ───────────────────────────────────────────────────────
    from prompt_toolkit.patch_stdout import patch_stdout as _patch_stdout
    while True:
        with _patch_stdout():
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
        if user_input.lower() in ("/help", "/?"):
            print("""
Commands:
  /web <query>     search the web (DuckDuckGo, no API needed)
  /search <term>   search memory without LLM — shows scored results
  /note <text>     quick-capture a note (indexed on next ingest)
  /forget <topic>  suppress matching chunks from future retrieval
  /sources         show sources from last answer
  /remember <fact> save a fact to this session
  /private         toggle private vault on/off
  /mode [name]     show or switch reasoning mode (CHAT/PLAN/DECIDE/DEBATE/REFLECT/EXPLORE)
  /resume          interactive session picker
  /clear           clear conversation history
  /quit or /exit   end session and save
""")
            continue

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
                {"role": t["role"], "content": t["content"]} for t in session.turns[-40:]
            )
            session_entities = list(session.entities)
            _print_transcript(session.turns)
            date = session.started_at[:19].replace("T", " ")
            print(f"[Resumed session from {date} — {len(session.turns) // 2} exchange(s)]\n")
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

        if user_input.lower().startswith("/search "):
            term = user_input[8:].strip()
            if not term:
                print("[Usage: /search <term>]\n")
                continue
            try:
                vec = embed_query(term)
            except Exception as e:
                print(f"[Embed error: {e}]\n")
                continue
            results = retrieve(
                query_vector=vec,
                qdrant=qdrant,
                memory_store=memory_store,
                raw_collection=COLLECTION_PUBLIC,
                compressed_collection=COLLECTION_COMPRESSED_PUBLIC,
                top_k=10,
                compressed_threshold=COMPRESSED_SCORE_THRESHOLD,
                expand_links=False,
            )
            if not results:
                print("[No results found]\n")
            else:
                print(f"\nSearch results for '{term}':")
                for r in results:
                    title = r.get("title", "?")
                    date = r.get("created_at", "")[:10]
                    score = r.get("score", 0)
                    layer = r.get("layer", "")
                    snippet = r.get("text", "")[:120].replace("\n", " ")
                    print(f"  [{score:.3f}] {title} ({date}) [{layer}]")
                    print(f"         {snippet}...")
                print()
            continue

        if user_input.lower().startswith("/note "):
            note_text = user_input[6:].strip()
            if not note_text:
                print("[Usage: /note <text>]\n")
                continue
            from mindvault.config import BRAIN_DIR
            notes_dir = BRAIN_DIR / "notes"
            notes_dir.mkdir(exist_ok=True)
            from datetime import datetime as _dt
            fname = _dt.now().strftime("%Y-%m-%dT%H-%M-%S") + ".md"
            (notes_dir / fname).write_text(note_text + "\n")
            print(f"[Saved to notes/{fname} — run 'ingest' to index it]\n")
            continue

        if user_input.lower().startswith("/forget "):
            topic = user_input[8:].strip()
            if not topic:
                print("[Usage: /forget <topic>]\n")
                continue
            try:
                vec = embed_query(topic)
            except Exception as e:
                print(f"[Embed error: {e}]\n")
                continue
            candidates = retrieve(
                query_vector=vec,
                qdrant=qdrant,
                memory_store=memory_store,
                raw_collection=COLLECTION_PUBLIC,
                compressed_collection=COLLECTION_COMPRESSED_PUBLIC,
                top_k=5,
                compressed_threshold=COMPRESSED_SCORE_THRESHOLD,
                expand_links=False,
            )
            if not candidates:
                print("[Nothing found to forget]\n")
                continue
            print(f"\nSuppressing {len(candidates)} result(s) for '{topic}':")
            chunk_ids = []
            for c in candidates:
                print(f"  - {c.get('title', '?')} ({c.get('created_at', '')[:10]})")
                chunk_ids.append(c["chunk_id"])
            memory_store.suppress_chunks(chunk_ids)
            print(f"[Done — these will no longer surface in chat or search]\n")
            continue

        if user_input.lower().startswith("/web "):
            web_query = user_input[5:].strip()
            if not web_query:
                print("[Usage: /web <query>]\n")
                continue
            # Force web search and feed results to the LLM — bypass memory entirely
            from src.adapters.web import web_search as _web_search
            print(f"  [Searching the web for: {web_query}]\n")
            web_chunks = _web_search(web_query, max_results=WEB_SEARCH_MAX_RESULTS)
            if not web_chunks:
                print("[No web results found — check your internet connection]\n")
                continue
            # Reuse ask() logic but inject web chunks directly
            from mindvault.modes import get_config as _get_config
            _mode = prompt_ui.mode
            _config = _get_config(_mode)
            anim = BakingAnimation()
            anim.start()
            response = run_council(
                mode=_mode,
                query=web_query,
                chunks=web_chunks,
                model=LLM_MODEL,
                base_url=OLLAMA_BASE,
                history=history if _mode == Mode.CHAT else None,
            )
            elapsed = anim.stop()
            if response:
                print_bar()
                print(f"✻ Baked for {elapsed:.1f}s  [web]")
                print_markdown_response(f"Brain [{_config.label}]", response)
                history.append({"role": "user", "content": web_query})
                history.append({"role": "assistant", "content": response})
                last_chunks = web_chunks
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
