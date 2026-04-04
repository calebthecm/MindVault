"""
src/adapters/web.py — Local web search via DuckDuckGo (no API key required).

Returns results as chunk dicts compatible with the retriever output format
so they can be mixed with memory chunks and passed directly to the LLM.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo and return results formatted as retriever-style chunk dicts.

    Each result has: chunk_id, source_id (url), text, title, created_at,
    score (fixed 0.7), layer="web", url.

    Returns [] on any error — callers should handle gracefully.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed: pip install duckduckgo-search")
        return []

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        logger.info(f"DuckDuckGo search failed: {e}")
        return []

    now = datetime.now(timezone.utc).isoformat()
    chunks = []
    for r in raw:
        url = r.get("href", "")
        title = r.get("title", url)
        snippet = r.get("body", "").strip()
        if not snippet:
            continue
        chunk_id = f"web_{abs(hash(url))}"
        chunks.append({
            "chunk_id": chunk_id,
            "source_id": url,
            "text": f"{title}\n{snippet}",
            "title": title,
            "url": url,
            "source_type": "web",
            "created_at": now,
            "score": 0.7,
            "layer": "web",
        })

    return chunks


def fetch_page(url: str, max_chars: int = 8000) -> Optional[str]:
    """
    Fetch a URL and extract clean article text via trafilatura.
    Returns None on failure.
    """
    try:
        import httpx
        import trafilatura
        resp = httpx.get(url, timeout=10.0, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 MindVault/1.0"})
        resp.raise_for_status()
        text = trafilatura.extract(resp.text) or ""
        return text[:max_chars] if text else None
    except Exception as e:
        logger.info(f"Failed to fetch {url}: {e}")
        return None
