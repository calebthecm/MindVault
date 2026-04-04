# HOW TO RUN — Brain Second Brain System

## What this is

A local RAG (retrieval-augmented generation) system that turns your AI conversation exports and Obsidian notes into a searchable, conversational second brain. You can ask it questions and it answers using only your own past thoughts, decisions, and conversations.

Everything runs locally. No data leaves your machine.

---

## Prerequisites

- Python 3.11+ (you have 3.14)
- Ollama running with two models pulled:
  - `ollama pull nomic-embed-text` — for vector search (already done)
  - `ollama pull llama3.2` — for summarization and chat (already done)

Start Ollama if it's not running: `ollama serve`

---

## Scripts

### `python ingest.py` — Index data + generate notes

The main pipeline. Run this whenever you add new export data.

```
python ingest.py                 # full pipeline: index vectors + generate notes
python ingest.py --notes-only    # regenerate Obsidian notes only (no re-embedding)
python ingest.py --index-only    # rebuild vector index only (no notes)
python ingest.py --force         # re-index everything even if already processed
python ingest.py --stats         # show current index stats and exit
python ingest.py --no-llm        # skip llama3.2 (faster, uses keyword rules only)
python ingest.py --in-memory     # use in-memory Qdrant (nothing saved to disk)
```

### `python chat.py` — Talk to your second brain

Interactive REPL. Ask questions, get answers from your own data.

```
python chat.py                                    # interactive mode
python chat.py "what projects am I working on?"  # single query
```

Commands during chat:
- `/quit` — exit
- `/clear` — clear conversation history (start fresh)
- `/private` — toggle Private Brain content on/off
- `/sources` — show which memories were used in the last answer

### `python generate_notes.py` — Regenerate notes only

Same as `python ingest.py --notes-only`. Useful if you want to change how notes look without re-embedding.

---

## Adding new export data

**Anthropic export:**
1. Download your data from claude.ai
2. Drop the export folder (e.g. `data-2026-05-01-batch-0000/`) into `Brain/`
3. Run `python ingest.py`

The system auto-detects any folder with `.json` files as an export source. You do not need to rename anything or update any config.

**OpenAI/ChatGPT export:**
1. Download from chat.openai.com → Settings → Data Controls → Export
2. Drop the folder into `Brain/`
3. Run `python ingest.py`

If the format is not recognized, llama3.2 will inspect the file structure and figure out how to parse it automatically.

---

## Configuration

All settings live in `config.py`. The key ones:

| Variable | Default | What it controls |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Model used for summarization, categorization, chat |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model used for vector search |
| `USE_LLM_SUMMARIZATION` | `True` | Whether llama3.2 writes clean notes (vs raw transcripts) |
| `USE_LLM_CATEGORIZATION` | `True` | Whether llama3.2 decides categories (vs keyword rules) |
| `CHAT_TOP_K` | `8` | Number of memory chunks retrieved per chat query |
| `CHAT_INCLUDE_PRIVATE` | `False` | Whether Private Brain is included in chat by default |
| `MAX_TRANSCRIPT_CHARS_FOR_LLM` | `4000` | How much of a transcript goes to llama3.2 |
| `WEAK_MATCH_THRESHOLD` | `2` | Keyword hits below this = enters category discovery pool |
| `MIN_CLUSTER_SIZE` | `3` | Min conversations to create a new auto-discovered category |

---

## Storage files

| File / Folder | What it is |
|---|---|
| `brain.db` | SQLite database tracking ingested documents, chunks, and batches |
| `.qdrant/` | Qdrant vector database (persistent, survives restarts) |
| `My Brain/` | Obsidian vault — business, projects, general knowledge |
| `Private Brain/` | Obsidian vault — private content (separate Qdrant collection) |
| `data-*/` | Anthropic or other export batch directories |

---

## How the pipeline works

```
Export folders (any format)
        │
        ▼
export_detector.py
  Detects format (Anthropic / OpenAI / unknown via llama3.2)
  Extracts conversations into normalized format
        │
        ├──► ingest.py (vector path)
        │       chunk → embed (nomic-embed-text) → store (Qdrant + SQLite)
        │
        └──► generate_notes.py (notes path)
                llama3.2 categorizes all conversations in one batch call
                llama3.2 summarizes each conversation into a knowledge note
                category_discovery.py finds new topic clusters in uncategorized data
                Writes .md files into My Brain/ with frontmatter + wikilinks
                Updates .obsidian/graph.json with color groups
```

## How chat works

```
You type a question
        │
        ▼
nomic-embed-text embeds the query
        │
        ▼
Qdrant searches the brain_public collection (+ brain_private if /private is on)
Returns top 8 most relevant chunks
        │
        ▼
llama3.2 reads the chunks as context + your question
Answers using only what's in your brain
        │
        ▼
You see the answer. Use /sources to see which memories it used.
```

---

## Virtual environment

The `.venv/` folder contains all Python dependencies. Always run scripts with:

```
.venv/bin/python ingest.py
.venv/bin/python chat.py
```

Or activate the venv first: `source .venv/bin/activate`

Installed packages: `qdrant-client`, `httpx`
