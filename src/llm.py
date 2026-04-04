"""
src/llm.py — Ollama wrapper for llama3.2.

Used for:
  - Summarizing conversations into clean Obsidian notes
  - Deciding which category a conversation belongs in
  - Detecting unknown export formats
  - Powering the chat interface (retrieve → synthesize → respond)

All calls are synchronous and include basic retry logic.
If Ollama is unreachable, functions return None and callers fall back gracefully.
"""

import json
import logging
import re
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = 2
RETRY_DELAY = 1.5


def _call_ollama(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> Optional[str]:
    """
    Call Ollama's /api/generate endpoint.
    Returns the response text, or None if the call fails.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = httpx.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except httpx.HTTPError as e:
            if attempt < RETRY_ATTEMPTS - 1:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"LLM call failed after {RETRY_ATTEMPTS} attempts: {e}")
                return None


def _extract_json(text: str) -> Optional[dict | list]:
    """Pull JSON out of an LLM response that may have surrounding prose."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON block
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find raw JSON object or array
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


# ─── Summarization ────────────────────────────────────────────────────────────

def summarize_conversation(
    title: str,
    transcript: str,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    max_chars: int = 4000,
) -> Optional[str]:
    """
    Use llama3.2 to write a clean, dense knowledge note from a conversation.

    Returns Markdown-formatted note content, or None if LLM unavailable.
    The note is written in second person ("You explored...", "You built...")
    so it reads as a personal memory when retrieved later.
    """
    truncated = transcript[:max_chars]
    if len(transcript) > max_chars:
        truncated += "\n\n[...transcript truncated for summary...]"

    prompt = f"""You are writing a personal knowledge note for a second brain system.
The note should capture the essence of this conversation so the person can recall it later.

Conversation title: {title}

Transcript:
{truncated}

Write a concise Obsidian-style note with these sections:
## Key Points
(3-5 bullet points of the most important insights, decisions, or information)

## What Was Built / Decided
(If any tool, script, document, or plan was created — what it was)

## Follow-up Ideas
(Any next steps, open questions, or ideas that came up — skip if none)

Rules:
- Write in second person: "You asked...", "You built...", "You decided..."
- Be specific, not vague. Include actual names, numbers, tool names if mentioned.
- Do not pad with filler. If a section has nothing meaningful, omit it.
- No em dashes. No emojis.
- Keep total length under 300 words."""

    result = _call_ollama(prompt, model=model, base_url=base_url, timeout=90.0)
    if not result:
        logger.warning(f"LLM summarization failed for '{title}', using fallback")
    return result


# ─── Categorization ───────────────────────────────────────────────────────────

def categorize_conversations(
    conversations: list[dict],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> Optional[dict[str, dict]]:
    """
    Pass all conversation titles + summaries to llama3.2 at once.
    Ask it to decide: folder name, tags, and whether any new categories are needed.

    Returns {uuid: {"category": str, "tags": [str]}} or None if LLM unavailable.

    Batching all conversations in one call is intentional — it lets the model
    see the full picture and make consistent grouping decisions, rather than
    categorizing each conversation in isolation.
    """
    items = []
    for c in conversations:
        summary_preview = (c.get("summary") or "")[:200]
        items.append({
            "uuid": c["uuid"],
            "title": c.get("name", "Untitled"),
            "summary": summary_preview,
        })

    prompt = f"""You are organizing conversations into a personal knowledge vault (Obsidian second brain).

Here are all the conversations to categorize:

{json.dumps(items, indent=2)}

Rules:
- Group similar conversations under shared folder names
- Use descriptive, broad categories (e.g. "Business & Agency", "Learning", "Personal", "Development")
- Separate business/work topics from personal/random/homework topics clearly
- Keep the number of categories reasonable (ideally 4-8 total)
- Tags should be lowercase, hyphen-separated, 1-3 tags per conversation

Respond ONLY with a JSON object mapping each UUID to its category and tags:
{{
  "uuid-here": {{"category": "Category Name", "tags": ["tag1", "tag2"]}},
  ...
}}"""

    result = _call_ollama(prompt, model=model, base_url=base_url, timeout=120.0)
    if not result:
        return None

    parsed = _extract_json(result)
    if not isinstance(parsed, dict):
        logger.warning("LLM categorization returned unexpected format, falling back to keywords")
        return None

    return parsed


# ─── Export format detection ──────────────────────────────────────────────────

def detect_export_format(
    export_dir_path: str,
    file_previews: dict[str, str],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> Optional[dict]:
    """
    Given a directory's file list and JSON previews, use llama3.2 to identify
    the export format and describe how to extract conversations from it.

    Returns a dict describing the format, or None if detection fails.

    This is what allows the system to handle OpenAI exports, Gemini exports,
    or any other format without hardcoded parsers — the LLM reads the structure
    and explains it.
    """
    prompt = f"""You are analyzing an AI conversation export directory to understand its data format.

Directory: {export_dir_path}
Files found: {list(file_previews.keys())}

File content previews (first 600 chars each):
{json.dumps(file_previews, indent=2)}

Identify the format and describe how to extract conversations. Respond ONLY as JSON:
{{
  "source": "anthropic | openai | gemini | unknown",
  "conversations_file": "filename.json",
  "conversations_path": "$.conversations" or "top-level array" etc,
  "title_field": "field name for conversation title",
  "messages_field": "field name for messages array",
  "message_text_field": "field name or path for message text content",
  "message_role_field": "field name for sender/role",
  "timestamp_field": "field name for created_at or equivalent",
  "notes": "any other important structural details"
}}"""

    result = _call_ollama(prompt, model=model, base_url=base_url, timeout=60.0)
    if not result:
        return None

    parsed = _extract_json(result)
    if not isinstance(parsed, dict):
        return None

    return parsed


# ─── Chat interface ───────────────────────────────────────────────────────────

def chat_with_brain(
    query: str,
    context_chunks: list[dict],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    conversation_history: Optional[list[dict]] = None,
) -> Optional[str]:
    """
    Use llama3.2 to answer a query using retrieved context chunks.

    context_chunks: list of dicts with keys: text, title, source_type, created_at
    conversation_history: list of {"role": "user"|"assistant", "content": str}
                          for multi-turn conversation support

    The model is instructed to answer AS the user's second brain — drawing
    only on the retrieved context and speaking from the user's own perspective.
    """
    if not context_chunks:
        context_text = "No relevant memories found in your brain for this query."
    else:
        parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("title", "Unknown")
            date = chunk.get("created_at", "")[:10]
            text = chunk.get("text", "").strip()
            parts.append(f"[Memory {i}] Source: {source} ({date})\n{text}")
        context_text = "\n\n---\n\n".join(parts)

    # Build conversation history string if multi-turn
    history_text = ""
    if conversation_history:
        for turn in conversation_history[-6:]:  # last 3 exchanges
            role = "You" if turn["role"] == "user" else "Brain"
            history_text += f"{role}: {turn['content']}\n"

    system = """You are a personal second brain — a knowledge assistant built entirely from the user's own past conversations and notes.

Your job: answer questions using ONLY the memories provided. Do not make things up.
If the memories don't contain enough to answer, say so directly.
Speak as if you remember these things — you ARE the user's memory.
Be specific. Reference actual details from the memories when relevant.
Do not use em dashes. Do not use emojis. Keep responses focused and direct."""

    prompt = f"""Relevant memories from your brain:

{context_text}

{f"Previous conversation:{chr(10)}{history_text}" if history_text else ""}

Question: {query}

Answer using your memories above:"""

    return _call_ollama(prompt, model=model, system=system, base_url=base_url, timeout=120.0)


# ─── Session compression ──────────────────────────────────────────────────────

def compress_session(
    turns: list[dict],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    max_chars: int = 6000,
) -> Optional[str]:
    """
    Generate a 2-4 sentence summary of a full chat session.
    Returns None if LLM unavailable or session has no turns.
    """
    if not turns:
        return None

    transcript_parts = []
    char_count = 0
    for turn in turns:
        line = f"{turn['role'].upper()}: {turn['content']}"
        if char_count + len(line) > max_chars:
            transcript_parts.append("[...truncated...]")
            break
        transcript_parts.append(line)
        char_count += len(line)

    transcript = "\n".join(transcript_parts)

    prompt = f"""Summarize this chat session in 2-4 sentences. Capture:
- What topics were discussed
- Any decisions made or conclusions reached
- Any projects, tools, or ideas mentioned

Be specific. Second person: "You discussed...", "You decided..."
No bullet points. Prose only. Under 150 words.

Session:
{transcript}

Summary:"""

    return _call_ollama(prompt, model=model, base_url=base_url, timeout=60.0)
