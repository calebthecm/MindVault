"""
Embedder: calls Ollama's embedding API to produce vectors for Chunks.

Uses nomic-embed-text by default (768 dimensions, good quality/speed tradeoff).
Falls back gracefully with clear error messages if Ollama is unavailable.
"""

import logging
import time
from typing import Optional

import httpx

from src.models import Chunk

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"
BATCH_SIZE = 32         # Chunks per Ollama request
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0       # seconds


def _embed_batch(texts: list[str], model: str, client: httpx.Client) -> list[list[float]]:
    """Call Ollama /api/embed for a batch of texts. Returns list of embedding vectors."""
    payload = {"model": model, "input": texts}
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.post(
                f"{OLLAMA_BASE}/api/embed",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except httpx.HTTPError as e:
            if attempt < RETRY_ATTEMPTS - 1:
                logger.warning(f"Embed attempt {attempt + 1} failed: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Ollama embedding failed after {RETRY_ATTEMPTS} attempts. "
                    f"Make sure Ollama is running and '{model}' is pulled.\n"
                    f"Run: ollama pull {model}"
                ) from e


def embed_chunks(
    chunks: list[Chunk],
    model: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> list[tuple[Chunk, list[float]]]:
    """
    Embed a list of Chunks.
    Returns list of (chunk, embedding_vector) pairs in the same order as input.
    """
    if not chunks:
        return []

    results: list[tuple[Chunk, list[float]]] = []

    with httpx.Client() as client:
        # Verify Ollama is reachable
        try:
            client.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0).raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                "Cannot reach Ollama at localhost:11434. Start Ollama with `ollama serve`."
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

            vectors = _embed_batch(texts, model, client)

            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}"
                )

            results.extend(zip(batch, vectors))

    logger.info(f"Embedded {len(results)} chunks with model '{model}'")
    return results
