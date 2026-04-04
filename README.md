# MindVault

A local-first second brain that turns your AI conversation exports, notes, and documents into a searchable, conversational memory system.

Everything runs on your machine. No data leaves.

---

## What it does

- **Ingests** Anthropic, OpenAI, and other AI chat exports automatically
- **Indexes** your Obsidian notes alongside your conversations
- **Remembers** entities, decisions, and goals extracted from every chat
- **Retrieves** using hybrid scoring — summaries first, raw text as fallback
- **Chats** interactively so you can ask questions across your own history
- **Saves sessions** — resume any previous conversation exactly where you left off

---

## Quick start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) with two models:

```bash
ollama pull nomic-embed-text   # for vector search
ollama pull llama3.2           # for chat and summarization
ollama serve                   # start Ollama if not running
```

### Install

```bash
git clone https://github.com/calebthecm/mindvault
cd mindvault
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### First run

```bash
python mindvault.py setup      # configure models and create folders
```

### Add your data

Drop your export folder into the project directory:

- **Anthropic:** download from `claude.ai → Settings → Export Data`
- **OpenAI:** download from `chat.openai.com → Settings → Data Controls → Export`

Any folder with `.json` files is auto-detected.

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
python mindvault.py ingest --no-llm         # skip LLM (keyword rules only, faster)
python mindvault.py notes                   # regenerate Obsidian notes
python mindvault.py setup                   # first-run configuration wizard
python mindvault.py stats                   # show index and session statistics
python mindvault.py sessions                # list resumable sessions
python mindvault.py help [command]          # show help
```

**During a chat session:**

```
/quit, /exit       end session (compresses and saves automatically)
/clear             clear conversation history
/private           toggle private vault inclusion
/sources           show which memories were used in the last answer
/remember <fact>   immediately save a fact to this session
```

---

## How it works

### Memory layers

MindVault stores knowledge in four layers to keep context small for local models:

| Layer | What | Used for |
|---|---|---|
| Raw | Original text chunks | Fallback when summaries aren't confident enough |
| Compressed | LLM-generated summaries per session/document | Primary retrieval context |
| Structured | Extracted entities (persons, projects, decisions, goals) | Entity-boosted retrieval |
| Linked | Relationships between memories | Future: graph traversal |

### Retrieval scoring

```
score = 0.5 × embedding_similarity
      + 0.2 × entity_overlap
      + 0.2 × recency
      + 0.1 × importance
```

Compressed summaries are searched first. If confidence is below 0.75, raw chunks are also fetched and merged.

### Session lifecycle

1. **During chat:** raw turns saved live + entities extracted per exchange
2. **On exit:** LLM compresses the session into a 2–4 sentence summary
3. **Summary embedded** and stored in compressed memory layer
4. **Resume anytime** with `--resume` to pick up where you left off

### Pipeline

```
Export folders (any format)
        │
        ▼
Auto-detection (Anthropic / OpenAI / LLM-inspected)
        │
        ├──► ingest: chunk → embed → Qdrant + SQLite
        │
        └──► notes: LLM summarizes → Obsidian .md files
                    category discovery → graph coloring
```

---

## Configuration

All settings in `config.py`:

| Variable | Default | What it controls |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Model for summarization, chat, extraction |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Vector search embeddings |
| `CHAT_TOP_K` | `8` | Chunks retrieved per query |
| `COMPRESSED_SCORE_THRESHOLD` | `0.75` | Below this, also fetch raw chunks |
| `MAX_CONTEXT_CHARS` | `3000` | Max context sent to LLM per query |
| `SESSIONS_DIR` | `sessions/` | Where chat sessions are stored |
| `CHAT_INCLUDE_PRIVATE` | `False` | Include private vault by default |

---

## Storage

| Path | What |
|---|---|
| `brain.db` | SQLite: ingestion tracking, entities, importance scores |
| `.qdrant/` | Qdrant: vector index (raw + compressed collections) |
| `sessions/` | Compressed chat sessions (`.json.gz`) |
| `My Brain/` | Obsidian vault — business, projects, general knowledge |
| `Private Brain/` | Obsidian vault — personal content (separate collection) |
| `data-*/` | Export folders (excluded from git — see `.gitignore`) |

---

## Privacy

- All processing is local by default. No API calls unless you configure an external endpoint via `python mindvault.py setup`.
- `My Brain` and `Private Brain` are in **separate Qdrant collections** — private content is never implicitly included in responses.
- `.gitignore` excludes all personal data: vaults, exports, sessions, databases.

---

## Requirements

```
qdrant-client
httpx
```

Install: `pip install -r requirements.txt`

---

## License

MIT
