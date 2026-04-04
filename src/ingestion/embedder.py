"""
Embedder: generates embedding vectors for Chunks.

Supports:
  - Ollama /api/embed  (nomic-embed-text, mxbai-embed-large, etc.)
  - OpenAI-compatible /embeddings  (text-embedding-3-small, etc.)

Backend is read from mindvault.config at call time.
"""

import logging
import time
from typing import Optional

import httpx

from src.models import Chunk

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0


def _embed_batch_ollama(
    texts: list[str],
    model: str,
    base_url: str,
    client: httpx.Client,
) -> list[list[float]]:
    """Call Ollama /api/embed for a batch of texts."""
    payload = {"model": model, "input": texts}
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = client.post(f"{base_url}/api/embed", json=payload, timeout=60.0)
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except httpx.HTTPError as e:
            if attempt < RETRY_ATTEMPTS - 1:
                logger.warning(f"Embed attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Ollama embedding failed after {RETRY_ATTEMPTS} attempts. "
                    f"Make sure Ollama is running and '{model}' is pulled.\n"
                    f"Run: ollama pull {model}"
                ) from e


def _embed_batch_openai(
    texts: list[str],
    model: str,
    base_url: str,
    api_key: str,
    client: httpx.Client,
) -> list[list[float]]:
    """Call OpenAI-compatible /embeddings endpoint."""
    payload = {"model": model, "input": texts}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = client.post(
                f"{base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            # Sort by index to preserve order
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        except (httpx.HTTPError, KeyError) as e:
            if attempt < RETRY_ATTEMPTS - 1:
                logger.warning(f"Embed attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"OpenAI embedding failed after {RETRY_ATTEMPTS} attempts: {e}"
                ) from e


def embed_chunks(
    chunks: list[Chunk],
    model: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
) -> list[tuple[Chunk, list[float]]]:
    """
    Embed a list of Chunks using the configured backend.
    Returns list of (chunk, embedding_vector) pairs in the same order as input.
    """
    if not chunks:
        return []

    from mindvault.config import EMBEDDING_MODEL, OLLAMA_BASE, LLM_BACKEND, LLM_API_KEY
    if model is None:
        model = EMBEDDING_MODEL
    backend = LLM_BACKEND
    base_url = OLLAMA_BASE
    api_key = LLM_API_KEY

    results: list[tuple[Chunk, list[float]]] = []

    with httpx.Client() as client:
        if backend == "ollama":
            try:
                client.get(f"{base_url}/api/tags", timeout=5.0).raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(
                    f"Cannot reach Ollama at {base_url}. Start with `ollama serve`."
                ) from e

        total = len(chunks)
        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [chunk.text for chunk in batch]

            logger.info(
                f"Embedding batch {batch_start // batch_size + 1}/"
                f"{(total + batch_size - 1) // batch_size} "
                f"({len(texts)} chunks)..."
            )

            if backend == "ollama":
                vectors = _embed_batch_ollama(texts, model, base_url, client)
            else:
                vectors = _embed_batch_openai(texts, model, base_url, api_key, client)

            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}"
                )

            results.extend(zip(batch, vectors))

    logger.info(f"Embedded {len(results)} chunks with model '{model}'")
    return results
