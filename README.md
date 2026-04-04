# MindVault

A local-first second brain that turns your AI conversation exports, Obsidian notes, and documents into a searchable, conversational memory system.

Everything runs on your machine. No data leaves.

---

## What it does

- **Ingests** Anthropic conversation exports, Obsidian vaults, and PDFs
- **Indexes** content into a multi-layer memory system (raw → compressed → structured → linked)
- **Remembers** entities, decisions, and goals extracted from every chat
- **Retrieves** using hybrid scoring — summaries first, raw text only when needed
- **Chats** interactively with six reasoning modes powered by a council of AI voices
- **Saves sessions** — resume any previous conversation exactly where you left off

---

## Quick start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) with two models:

```bash
ollama pull nomic-embed-text   # vector search
ollama pull llama3.2           # chat and summarization
ollama serve                   # start Ollama if not running
```

### Install

```bash
git clone https://github.com/calebthecm/MindVault
cd MindVault
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### First run

```bash
python mindvault.py setup      # configure models and create folders
```

### Add your data

Drop your Anthropic export folder into the project directory (any folder starting with `data-`). PDFs go anywhere — point the ingester at them manually.

- **Anthropic:** `claude.ai → Settings → Export Data`

### Index and chat

```bash
python mindvault.py ingest     # index everything
python mindvault.py chat       # start talking to your brain
```

---

## Commands

```
python mindvault.py                         # chat (default)
python mindvault.py chat                    # interactive REPL
python mindvault.py chat --resume           # resume last session
python mindvault.py chat --resume <id>      # resume specific session
python mindvault.py ingest                  # auto-discover and index all exports
python mindvault.py ingest ./folder/        # index a specific folder
python mindvault.py ingest --force          # re-index even if already processed
python mindvault.py notes                   # regenerate Obsidian notes
python mindvault.py setup                   # first-run configuration wizard
python mindvault.py stats                   # show index and session statistics
python mindvault.py sessions                # list resumable sessions
```

**During a chat session:**

```
Shift+Tab          cycle reasoning mode
/mode [name]       show or switch mode (CHAT, PLAN, DECIDE, DEBATE, REFLECT, EXPLORE)
/quit, /exit       end session (compresses and saves automatically)
/clear             clear conversation history
/private           toggle private vault inclusion
/sources           show which memories were used in the last answer
/remember <fact>   immediately save a fact to this session
```

---

## Reasoning modes

MindVault has six modes, cycled with **Shift+Tab** in the prompt bar.

| Mode | What it does |
|---|---|
| 💬 CHAT | Standard RAG — retrieve memories, synthesize an answer |
| 📋 PLAN | Break the task into structured, actionable steps |
| 🗳 DECIDE | Five-voice council votes; tally + majority verdict shown |
| ⚖ DEBATE | FOR vs AGAINST, then a moderated verdict |
| 🔍 REFLECT | Deep synthesis — what does your brain really know about this? |
| 🕸 EXPLORE | Graph traversal — follows memory links to surface surprises |

The council is five internal voices with distinct personalities:

| Voice | Orientation |
|---|---|
| 📊 The Analyst | Evidence-first, skeptical, quantitative |
| 🚀 The Visionary | Big-picture, creative, optimistic |
| 🔧 The Pragmatist | What's actionable right now |
| 😈 The Devil | Challenges every assumption, finds the flaw |
| 📜 The Historian | Patterns across time; what past memory reveals |

---

## How it works

### Memory layers

| Layer | What | Used for |
|---|---|---|
| Raw | Original text chunks | Fallback when summaries aren't confident enough |
| Compressed | LLM-generated summaries per session/document | Primary retrieval context |
| Structured | Extracted entities (persons, projects, decisions, goals) | Entity-boosted retrieval |
| Linked | Relationships between memories via shared entities + wikilinks | Graph traversal in EXPLORE mode |

### Retrieval scoring

```
score = 0.5 × embedding_similarity
      + 0.2 × entity_overlap
      + 0.2 × recency
      + 0.1 × importance
```

Compressed summaries are searched first. Raw chunks are only fetched when confidence drops below the threshold. EXPLORE mode additionally walks `memory_links` to pull in related neighbors.

### Session lifecycle

1. **During chat:** turns saved live + entities extracted per exchange
2. **On exit:** LLM compresses the session into a 2–4 sentence summary
3. **Summary embedded** and stored in the compressed memory layer
4. **Resume anytime** with `--resume`

---

## Configuration

All settings in `mindvault/config.py`:

| Variable | Default | What it controls |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Model for summarization, chat, extraction |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Vector search embeddings |
| `CHAT_TOP_K` | `8` | Chunks retrieved per query |
| `COMPRESSED_SCORE_THRESHOLD` | `0.75` | Below this, also fetch raw chunks |
| `MAX_CONTEXT_CHARS` | `3000` | Max context sent to LLM per query |
| `CHAT_INCLUDE_PRIVATE` | `False` | Include private vault by default |

---

## Storage

| Path | What |
|---|---|
| `brain.db` | SQLite: ingestion tracking, entities, links, importance scores |
| `.qdrant/` | Qdrant: vector index (raw + compressed collections) |
| `sessions/` | Compressed chat sessions (`.json.gz`) |
| `My Brain/` | Obsidian vault — business, projects, general knowledge |
| `Private Brain/` | Obsidian vault — personal content (separate collection) |
| `data-*/` | Export folders (excluded from git) |

---

## Privacy

- All processing is local by default.
- `My Brain` and `Private Brain` are in **separate Qdrant collections** — private content is never implicitly included in responses.
- `.gitignore` excludes all personal data: vaults, exports, sessions, databases.

---

## Requirements

```
qdrant-client
httpx
python-dotenv
pypdf
prompt_toolkit
```
