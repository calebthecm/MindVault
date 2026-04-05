"""
web/server.py — Local FastAPI web interface for MindVault.

Serves a browser-based chat UI at http://localhost:7432
Start with: mindvault web
"""

from __future__ import annotations

import logging
import json
import sys
import threading
from queue import Queue
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient

# Ensure project root is on path when run directly
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="MindVault", docs_url=None, redoc_url=None)

# Serve static files (HTML, etc.)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    mode: str = "CHAT"
    include_private: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode: str


class StatusResponse(BaseModel):
    status: str
    version: str


def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "title": c.get("title") or c.get("source_id", ""),
            "score": round(c.get("score", 0), 3),
            "layer": c.get("layer", ""),
            "created_at": c.get("created_at", ""),
        }
        for c in chunks[:5]
    ]


@lru_cache(maxsize=1)
def _web_services():
    from mindvault.chat import embed_query
    from mindvault.config import (
        CHAT_TOP_K,
        CHAT_INCLUDE_PRIVATE,
        COMPRESSED_SCORE_THRESHOLD,
        DB_PATH,
        OLLAMA_BASE,
        LLM_MODEL,
        QDRANT_PATH,
        COLLECTION_COMPRESSED_PUBLIC,
        COLLECTION_COMPRESSED_PRIVATE,
    )
    from src.ingestion.store import COLLECTION_PUBLIC, COLLECTION_PRIVATE
    from src.llm import chat_with_brain
    from src.memory.retriever import retrieve
    from src.memory.store import MemoryStore

    qdrant = QdrantClient(path=str(QDRANT_PATH))
    memory_store = MemoryStore(
        db_path=DB_PATH,
        qdrant=qdrant,
        compressed_collections=(COLLECTION_COMPRESSED_PUBLIC, COLLECTION_COMPRESSED_PRIVATE),
    )
    return {
        "embed_query": embed_query,
        "chat_with_brain": chat_with_brain,
        "retrieve": retrieve,
        "qdrant": qdrant,
        "memory_store": memory_store,
        "chat_top_k": CHAT_TOP_K,
        "chat_include_private": CHAT_INCLUDE_PRIVATE,
        "compressed_threshold": COMPRESSED_SCORE_THRESHOLD,
        "ollama_base": OLLAMA_BASE,
        "llm_model": LLM_MODEL,
        "collection_public": COLLECTION_PUBLIC,
        "collection_private": COLLECTION_PRIVATE,
        "collection_compressed_public": COLLECTION_COMPRESSED_PUBLIC,
        "collection_compressed_private": COLLECTION_COMPRESSED_PRIVATE,
    }


def _retrieve_for_web(query: str, include_private: bool, top_k: int) -> list[dict]:
    services = _web_services()
    from mindvault.chat import condensed_retrieval_query

    retrieval_query = condensed_retrieval_query(query)
    vector = services["embed_query"](retrieval_query)

    chunks = services["retrieve"](
        query_vector=vector,
        qdrant=services["qdrant"],
        memory_store=services["memory_store"],
        raw_collection=services["collection_public"],
        compressed_collection=services["collection_compressed_public"],
        top_k=top_k,
        compressed_threshold=services["compressed_threshold"],
        expand_links=False,
    )

    if include_private:
        private_chunks = services["retrieve"](
            query_vector=vector,
            qdrant=services["qdrant"],
            memory_store=services["memory_store"],
            raw_collection=services["collection_private"],
            compressed_collection=services["collection_compressed_private"],
            top_k=top_k,
            compressed_threshold=services["compressed_threshold"],
            expand_links=False,
        )
        chunks = sorted(chunks + private_chunks, key=lambda c: c["score"], reverse=True)[:top_k]

    return chunks


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = _static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text())
    return HTMLResponse("<h1>MindVault Web UI</h1><p>static/index.html not found.</p>")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/status", response_model=StatusResponse)
async def status():
    from mindvault.version import CURRENT_VERSION
    return StatusResponse(status="ok", version=CURRENT_VERSION)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        services = _web_services()
        include_private = req.include_private or services["chat_include_private"]
        chunks = _retrieve_for_web(req.query, include_private=include_private, top_k=services["chat_top_k"])

        from mindvault.council import run_council
        from mindvault.modes import Mode

        try:
            mode = Mode(req.mode.upper())
        except ValueError:
            mode = Mode.CHAT

        if mode == Mode.CHAT:
            answer = services["chat_with_brain"](
                query=req.query,
                context_chunks=chunks,
                model=services["llm_model"],
                base_url=services["ollama_base"],
                conversation_history=[],
            )
        else:
            answer = run_council(
                mode=mode,
                query=req.query,
                chunks=chunks,
                model=services["llm_model"],
                base_url=services["ollama_base"],
            )

        sources = _sources_from_chunks(chunks)

        return ChatResponse(answer=answer, sources=sources, mode=req.mode)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    def event_stream():
        queue: Queue[tuple[str, object]] = Queue()

        def _run() -> None:
            try:
                services = _web_services()
                include_private = req.include_private or services["chat_include_private"]
                chunks = _retrieve_for_web(req.query, include_private=include_private, top_k=services["chat_top_k"])

                from mindvault.council import run_council
                from mindvault.modes import Mode

                try:
                    mode = Mode(req.mode.upper())
                except ValueError:
                    mode = Mode.CHAT

                answer_parts: list[str] = []

                def _on_token(token: str) -> None:
                    answer_parts.append(token)
                    queue.put(("token", token))

                if mode == Mode.CHAT:
                    answer = run_council(
                        mode=mode,
                        query=req.query,
                        chunks=chunks,
                        model=services["llm_model"],
                        base_url=services["ollama_base"],
                        history=[],
                        on_token=_on_token,
                    )
                else:
                    answer = run_council(
                        mode=mode,
                        query=req.query,
                        chunks=chunks,
                        model=services["llm_model"],
                        base_url=services["ollama_base"],
                    )
                    answer_parts.append(answer)
                    queue.put(("token", answer))

                queue.put((
                    "done",
                    {
                        "answer": "".join(answer_parts) if mode == Mode.CHAT else answer,
                        "sources": _sources_from_chunks(chunks),
                        "mode": mode.value,
                    },
                ))
            except Exception as exc:
                logger.error(f"Chat stream error: {exc}", exc_info=True)
                queue.put(("error", str(exc)))
            finally:
                queue.put(("close", None))

        threading.Thread(target=_run, daemon=True).start()

        while True:
            event, payload = queue.get()
            if event == "close":
                break
            yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/search")
async def search(q: str, k: int = 8):
    try:
        chunks = _retrieve_for_web(q, include_private=False, top_k=k)
        return {
            "results": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "title": c.get("title") or c.get("source_id", ""),
                    "text": c.get("text", "")[:300],
                    "score": round(c.get("score", 0), 3),
                    "layer": c.get("layer", ""),
                    "created_at": c.get("created_at", ""),
                }
                for c in chunks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
