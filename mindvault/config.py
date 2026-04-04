"""
config.py — Single source of truth for all paths, model names, and thresholds.

Edit this file to change where data lives, which models to use,
or how aggressively the system categorizes and discovers topics.
"""

import os
from pathlib import Path

# Load .env from the Brain root (API keys live there, not in code)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# ─── Directory layout ─────────────────────────────────────────────────────────

# Root of the Brain project (parent of the mindvault/ package)
BRAIN_DIR = Path(__file__).parent.parent

# Obsidian vaults
VAULT_MY_BRAIN = BRAIN_DIR / "My Brain"
VAULT_PRIVATE = BRAIN_DIR / "Private Brain"

# Storage
DB_PATH = BRAIN_DIR / "brain.db"
QDRANT_PATH = BRAIN_DIR / ".qdrant"

# Directories that are never treated as export sources
EXCLUDED_DIRS = {
    ".venv", ".qdrant", "src", "mindvault", "__pycache__", ".git",
    "My Brain", "Private Brain",
}

# ─── Models ───────────────────────────────────────────────────────────────────

# LLM backend: "ollama" for local Ollama, "openai" for OpenAI-compatible APIs
LLM_BACKEND = "ollama"

# Base URL for the LLM backend
# Ollama default: http://localhost:11434
# OpenAI: https://api.openai.com/v1
OLLAMA_BASE = "http://localhost:11434"

# API key for OpenAI-compatible backends — set LLM_API_KEY in .env
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# Model used for: summarizing conversations, deciding categories, chat interface
LLM_MODEL = "llama3.2"

# Model used for: generating embeddings (vector search)
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768  # nomic-embed-text output dimension; OpenAI text-embedding-3-small = 1536

# ─── Ingestion settings ───────────────────────────────────────────────────────

# How many chunks to embed per Ollama API call
EMBED_BATCH_SIZE = 32

# Max characters per conversation turn before splitting into sub-chunks
MAX_TURN_CHARS = 2000

# ─── Summarization ────────────────────────────────────────────────────────────

# Whether to use LLM (llama3.2) to summarize conversations into notes.
# If False, raw transcripts are written instead (faster, no LLM calls).
USE_LLM_SUMMARIZATION = True

# Max transcript characters sent to LLM for summarization
# Keeps token usage bounded. Long convos get truncated before summarization.
MAX_TRANSCRIPT_CHARS_FOR_LLM = 4000

# ─── Categorization ───────────────────────────────────────────────────────────

# Whether to use LLM (llama3.2) to decide which folder each conversation goes in.
# If False, falls back to keyword matching rules.
USE_LLM_CATEGORIZATION = True

# ─── Discovery ────────────────────────────────────────────────────────────────

# Minimum keyword hits for a conversation to be considered "well-matched"
# by existing rules. Below this, it enters the discovery pool.
WEAK_MATCH_THRESHOLD = 2

# Min conversations in a cluster to earn its own category
MIN_CLUSTER_SIZE = 3

# Min percentage of total conversations a cluster must represent
MIN_CLUSTER_PCT = 0.06

# ─── Chat interface ───────────────────────────────────────────────────────────

# Number of vector search results to retrieve per query
CHAT_TOP_K = 8

# Whether to include private vault content in chat responses by default
CHAT_INCLUDE_PRIVATE = False

# ─── Session storage ──────────────────────────────────────────────────────────

# Directory where chat sessions are stored as compressed JSON
SESSIONS_DIR = BRAIN_DIR / "sessions"

# ─── Memory layers ────────────────────────────────────────────────────────────

# Max characters of context sent to LLM per query (summaries preferred)
MAX_CONTEXT_CHARS = 3000

# Below this compressed similarity score, also fetch raw chunks as fallback
COMPRESSED_SCORE_THRESHOLD = 0.75

# Qdrant collections for compressed (summary) memory
COLLECTION_COMPRESSED_PUBLIC = "brain_compressed_public"
COLLECTION_COMPRESSED_PRIVATE = "brain_compressed_private"
