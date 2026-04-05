"""
web/server.py — Local FastAPI web interface for MindVault.

Serves a browser-based chat UI at http://localhost:7432
Start with: mindvault web
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on path when run directly
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = _static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text())
    return HTMLResponse("<h1>MindVault Web UI</h1><p>static/index.html not found.</p>")


@app.get("/api/status", response_model=StatusResponse)
async def status():
    from mindvault.version import CURRENT_VERSION
    return StatusResponse(status="ok", version=CURRENT_VERSION)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        from mindvault.config import CHAT_TOP_K, CHAT_INCLUDE_PRIVATE
        from src.memory.retriever import retrieve
        from src.llm import chat_with_brain

        chunks = retrieve(
            req.query,
            top_k=CHAT_TOP_K,
            include_private=req.include_private or CHAT_INCLUDE_PRIVATE,
        )

        answer = chat_with_brain(
            query=req.query,
            chunks=chunks,
            history=[],
        )

        sources = [
            {
                "title": c.get("title") or c.get("source_id", ""),
                "score": round(c.get("score", 0), 3),
                "layer": c.get("layer", ""),
                "created_at": c.get("created_at", ""),
            }
            for c in chunks[:5]
        ]

        return ChatResponse(answer=answer, sources=sources, mode=req.mode)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search(q: str, k: int = 8):
    try:
        from src.memory.retriever import retrieve
        chunks = retrieve(q, top_k=k)
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
