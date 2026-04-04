"""
generate_notes.py — Build Obsidian .md notes from all discovered export data.

What it does:
  1. Auto-discovers every export directory in Brain/ (any folder with .json files)
  2. Detects the export format (Anthropic, OpenAI, or unknown via llama3.2)
  3. Uses llama3.2 to decide which category each conversation belongs in
  4. Uses llama3.2 to write real knowledge summaries instead of raw transcripts
  5. Runs category discovery to find new topic clusters worth their own folder
  6. Writes all notes into the configured vault with proper frontmatter + wikilinks
  7. Updates graph.json with color groups for each category

Run: python generate_notes.py
Or automatically via: python ingest.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mindvault.config import (
    BRAIN_DIR,
    LLM_MODEL,
    MAX_TRANSCRIPT_CHARS_FOR_LLM,
    OLLAMA_BASE,
    USE_LLM_CATEGORIZATION,
    USE_LLM_SUMMARIZATION,
    VAULT_MY_BRAIN,
)
from src.category_discovery import add_colors_to_graph, discover_categories
from src.export_detector import find_export_dirs, load_conversations_from_dir
from src.llm import categorize_conversations, summarize_conversation

VAULT_DIR = VAULT_MY_BRAIN
GRAPH_JSON = VAULT_DIR / ".obsidian" / "graph.json"


FALLBACK_CATEGORY = "General"
FALLBACK_TAGS = ["general"]

# Used by category_discovery.py to know which categories already exist
CATEGORY_RULES: list = []


# ─── Helpers ──────────────────────────────────────────────────────────────────

def slug(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_]+", "-", text.strip())
    return text[:60]


def fmt_date(iso: str) -> str:
    return iso[:10] if iso else ""


def fmt_datetime(iso: str) -> str:
    from datetime import datetime
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso[:16]


def extract_message_text(msg: dict) -> str:
    parts = []
    for block in msg.get("content", []):
        if block.get("type") == "text" and block.get("text", "").strip():
            parts.append(block["text"].strip())
    if not parts and msg.get("text", "").strip():
        parts.append(msg["text"].strip())
    return "\n".join(parts)


def build_transcript(convo: dict) -> str:
    lines = []
    for msg in convo.get("chat_messages", []):
        text = extract_message_text(msg)
        if not text:
            continue
        speaker = msg.get("sender", "unknown").upper()
        lines.append(f"[{speaker}]\n{text}")
    return "\n\n---\n\n".join(lines)


# ─── Note writers ─────────────────────────────────────────────────────────────

def write_conversation_note(
    convo: dict,
    category: str,
    tags: list[str],
    vault: Path,
    llm_summary: str | None = None,
) -> Path:
    uuid = convo["uuid"]
    name = convo.get("name", "Untitled")
    claude_summary = convo.get("summary", "").strip()
    created = convo.get("created_at", "")
    updated = convo.get("updated_at", created)

    # Body: prefer LLM-written knowledge note, fall back to transcript
    if llm_summary:
        body_section = f"## Knowledge Note\n\n{llm_summary}"
    else:
        transcript = build_transcript(convo)
        body_section = f"## Transcript\n\n{transcript or '_No transcript content._'}"

    # Related wikilinks
    related_map = {
        "Business & Agency": ["[[Areas/Business & Agency]]"],
        "Automation & Tools": ["[[Areas/Automation & Tools]]"],
        "Development": ["[[Areas/Development]]"],
        "Learning": ["[[Areas/Learning]]"],
        "Legal": ["[[Areas/Legal]]"],
        "Personal": ["[[Me/About Me]]"],
    }
    related_links = related_map.get(category, [])
    related_section = ("\n## Related\n\n" + "  ".join(related_links)) if related_links else ""

    tag_yaml = "\n".join(f"  - {t}" for t in tags)
    content = f"""---
title: "{name}"
date: {fmt_date(created)}
updated: {fmt_date(updated)}
source: anthropic-conversation
source_id: {uuid}
category: {category}
tags:
{tag_yaml}
---

# {name}

> **Date:** {fmt_datetime(created)}

{f"## Summary{chr(10)}{chr(10)}{claude_summary}{chr(10)}" if claude_summary else ""}
{body_section}
{related_section}
"""

    folder = vault / "Conversations" / category
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug(name)}.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_profile_note(memory_text: str, vault: Path) -> Path:
    folder = vault / "Me"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "About Me.md"
    path.write_text(f"""---
title: "About Me"
type: profile
source: anthropic-memory
tags:
  - profile
  - personal
---

# About Me

_Auto-generated from Claude's memory snapshot. Edit freely._

{memory_text}
""", encoding="utf-8")
    return path


def write_goals_note(vault: Path) -> Path:
    folder = vault / "Me"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "Goals & Ambitions.md"
    if path.exists():
        return path
    path.write_text("""---
title: "Goals & Ambitions"
type: goals
tags:
  - goals
  - planning
---

# Goals & Ambitions

_Add your goals here._

## Related

[[Me/About Me|About Me]]  [[Brain Index]]
""", encoding="utf-8")
    return path


def write_area_note(title: str, filename: str, body: str, tags: list[str], vault: Path) -> Path:
    folder = vault / "Areas"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{filename}.md"
    tag_yaml = "\n".join(f"  - {t}" for t in tags)
    path.write_text(f"""---
title: "{title}"
type: area
tags:
{tag_yaml}
---

# {title}

{body}
""", encoding="utf-8")
    return path


def write_index_note(
    conversation_notes: list[tuple[str, Path, str]],
    vault: Path,
) -> Path:
    by_category: dict[str, list[tuple[str, str]]] = {}
    for name, path, category in conversation_notes:
        rel = f"Conversations/{category}/{path.stem}"
        by_category.setdefault(category, []).append((name, rel))

    known_order = ["Business & Agency", "Automation & Tools", "Development", "Learning", "Legal", "Personal"]
    discovered = sorted(set(by_category.keys()) - set(known_order))

    conv_sections = ""
    for cat in known_order + discovered:
        if cat not in by_category:
            continue
        label = f"{cat} _(auto-discovered)_" if cat in discovered else cat
        conv_sections += f"\n### {label}\n\n"
        for name, rel in sorted(by_category[cat]):
            conv_sections += f"- [[{rel}|{name}]]\n"

    path = vault / "Brain Index.md"
    path.write_text(f"""---
title: "Brain Index"
type: moc
tags:
  - index
  - moc
---

# Brain Index

_Map of Content — entry point for your second brain._

## Me

- [[Me/About Me|About Me]]
- [[Me/Goals & Ambitions|Goals & Ambitions]]

## Conversations
{conv_sections}
""", encoding="utf-8")
    return path


# ─── Static area notes ────────────────────────────────────────────────────────

AREAS: list = []


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_notes(vault: Path = VAULT_DIR) -> int:
    """
    Generate all notes from all discovered export directories.
    Returns total note count.
    """
    vault = Path(vault)

    # ── Discover all export directories ──────────────────────────────────────
    export_dirs = find_export_dirs(BRAIN_DIR)
    if not export_dirs:
        print("No export directories found in Brain/")
        return 0

    # ── Load all conversations from all exports ───────────────────────────────
    all_conversations: list[dict] = []
    for export_dir in export_dirs:
        convos = load_conversations_from_dir(export_dir)
        all_conversations.extend(convos)

    print(f"\nLoaded {len(all_conversations)} conversations from {len(export_dirs)} export(s)")

    # ── Get LLM categorization (or fall back to keyword rules) ───────────────
    llm_categories: dict[str, dict] | None = None
    if USE_LLM_CATEGORIZATION:
        print(f"\nAsking {LLM_MODEL} to categorize conversations...")
        llm_categories = categorize_conversations(
            all_conversations, model=LLM_MODEL, base_url=OLLAMA_BASE
        )
        if llm_categories:
            print(f"LLM categorized {len(llm_categories)} conversations")
        else:
            print("LLM categorization unavailable, using keyword rules")

    # ── Write conversation notes ──────────────────────────────────────────────
    conversation_notes: list[tuple[str, Path, str]] = []
    print()

    for convo in all_conversations:
        uuid = convo["uuid"]
        name = convo.get("name", "Untitled")

        # Get category from LLM result, fall back to General
        if llm_categories and uuid in llm_categories:
            cat_data = llm_categories[uuid]
            category = cat_data.get("category", FALLBACK_CATEGORY)
            tags = cat_data.get("tags", FALLBACK_TAGS)
        else:
            category, tags = FALLBACK_CATEGORY, FALLBACK_TAGS

        # Get LLM summary (or None for raw transcript)
        llm_summary = None
        if USE_LLM_SUMMARIZATION:
            transcript = build_transcript(convo)
            if transcript.strip():
                llm_summary = summarize_conversation(
                    title=name,
                    transcript=transcript,
                    model=LLM_MODEL,
                    base_url=OLLAMA_BASE,
                    max_chars=MAX_TRANSCRIPT_CHARS_FOR_LLM,
                )

        summary_label = "LLM" if llm_summary else "raw"
        print(f"  [conv/{category}] {slug(name)}.md  ({summary_label})")

        path = write_conversation_note(convo, category, tags, vault, llm_summary)
        conversation_notes.append((name, path, category))

    # ── Category discovery ────────────────────────────────────────────────────
    # Collect category names already assigned to conversations
    existing_names = {cat for _, _, cat in conversation_notes} | {FALLBACK_CATEGORY}
    new_categories = discover_categories(
        conversations=all_conversations,
        category_rules=CATEGORY_RULES,
        existing_category_names=existing_names,
    )

    for cat in new_categories:
        for convo in cat["conversations"]:
            name = convo.get("name", "Untitled")
            path = write_conversation_note(
                convo, cat["name"], [cat["tag"], "discovered"], vault
            )
            conversation_notes.append((name, path, cat["name"]))
            print(f"  [discovered/{cat['name']}] {path.name}")

    if new_categories:
        add_colors_to_graph(new_categories, GRAPH_JSON)

    # ── Memory notes ─────────────────────────────────────────────────────────
    # Load memories from all Anthropic exports
    for export_dir in export_dirs:
        mem_path = export_dir / "memories.json"
        if not mem_path.exists():
            continue
        with open(mem_path) as f:
            memories = json.load(f)
        memory_text = memories[0].get("conversations_memory", "") if memories else ""
        if memory_text:
            path = write_profile_note(memory_text, vault)
            print(f"  [me] {path.name}")
            break  # Only write once even if multiple exports have memories

    path = write_goals_note(vault)
    print(f"  [me] {path.name}")

    # ── Area notes ────────────────────────────────────────────────────────────
    for title, filename, body, tags in AREAS:
        path = write_area_note(title, filename, body, tags, vault)
        print(f"  [area] {path.name}")

    # ── Index MOC ─────────────────────────────────────────────────────────────
    path = write_index_note(conversation_notes, vault)
    print(f"  [moc] {path.name}")

    total = len(conversation_notes) + len(AREAS) + 3
    return total


if __name__ == "__main__":
    print(f"Generating notes in '{VAULT_DIR}/'...\n")
    total = generate_notes()
    print(f"\nDone. {total} notes written.")
