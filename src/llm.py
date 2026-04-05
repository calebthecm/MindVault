"""
src/llm.py — Unified LLM wrapper supporting Ollama and OpenAI-compatible APIs.

Used for:
  - Summarizing conversations into clean Obsidian notes
  - Deciding which category a conversation belongs in
  - Detecting unknown export formats
  - Powering the chat interface (retrieve → synthesize → respond)
  - Extracting entities from conversation turns
  - Compressing sessions into dense summaries

All calls are synchronous and include basic retry logic.
If the LLM is unreachable, functions return None and callers fall back gracefully.
"""

import json
import logging
import re
import time
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = 2
RETRY_DELAY = 1.5


def _call_llm(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    api_key: str = "",
    backend: str = "ollama",
    timeout: float = 120.0,
) -> Optional[str]:
    """
    Unified LLM call supporting Ollama and OpenAI-compatible APIs.

    backend="ollama"  → POST {base_url}/api/generate
    backend="openai"  → POST {base_url}/chat/completions  (OpenAI, Groq, LM Studio, vLLM, etc.)

    Returns the response text, or None if the call fails.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            if backend == "ollama":
                payload: dict = {"model": model, "prompt": prompt, "stream": False}
                if system:
                    payload["system"] = system
                resp = httpx.post(
                    f"{base_url}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()

            else:
                # OpenAI-compatible chat/completions
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                payload = {"model": model, "messages": messages, "stream": False}
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                resp = httpx.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

        except (httpx.HTTPError, KeyError, IndexError) as e:
            if attempt < RETRY_ATTEMPTS - 1:
                logger.info(f"LLM call attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                logger.info(f"LLM call failed after {RETRY_ATTEMPTS} attempts: {e}")
                return None

    return None


def _call_ollama(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> Optional[str]:
    """
    Backward-compatible wrapper. Dispatches to _call_llm using config backend.
    Existing callers (extractor, etc.) automatically pick up the configured backend.
    """
    try:
        from mindvault.config import LLM_BACKEND, LLM_API_KEY
        backend = LLM_BACKEND
        api_key = LLM_API_KEY
    except ImportError:
        backend = "ollama"
        api_key = ""
    return _call_llm(prompt, model, system, base_url, api_key, backend, timeout)


def stream_ollama(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    on_token: Optional[Callable[[str], None]] = None,
    timeout: float = 180.0,
) -> str:
    """
    Stream an Ollama response token by token.

    Calls on_token(token_str) for each piece of text as it arrives.
    Returns the full accumulated response string.
    Falls back to non-streaming _call_ollama on any error.
    """
    try:
        from mindvault.config import LLM_BACKEND, LLM_API_KEY
        backend = LLM_BACKEND
        api_key = LLM_API_KEY
    except ImportError:
        backend = "ollama"
        api_key = ""

    full_text = ""

    if backend == "ollama":
        payload: dict = {"model": model, "prompt": prompt, "stream": True}
        if system:
            payload["system"] = system
        try:
            with httpx.stream(
                "POST", f"{base_url}/api/generate", json=payload, timeout=timeout
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("response", "")
                    if token:
                        full_text += token
                        if on_token:
                            on_token(token)
                    if data.get("done"):
                        break
            return full_text.strip()
        except Exception as e:
            logger.info(f"Streaming failed, falling back to blocking call: {e}")
            result = _call_ollama(prompt, model=model, system=system, base_url=base_url, timeout=timeout)
            if result and on_token:
                on_token(result)
            return result or ""

    else:
        # OpenAI-compatible streaming
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages, "stream": True}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            with httpx.stream(
                "POST", f"{base_url}/chat/completions",
                json=payload, headers=headers, timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        data = json.loads(line)
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            full_text += token
                            if on_token:
                                on_token(token)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            return full_text.strip()
        except Exception as e:
            logger.info(f"Streaming failed, falling back to blocking call: {e}")
            result = _call_ollama(prompt, model=model, system=system, base_url=base_url, timeout=timeout)
            if result and on_token:
                on_token(result)
            return result or ""


def _extract_json(text: str) -> Optional[dict | list]:
    """Pull JSON out of an LLM response that may have surrounding prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
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
    Write a clean, dense knowledge note from a conversation.
    Returns Markdown-formatted note content, or None if LLM unavailable.
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
    Pass all conversation titles + summaries to the LLM at once and ask it to
    assign a folder name and tags to each one.

    Batching all conversations in one call is intentional — the model sees the
    full picture and makes consistent grouping decisions instead of treating each
    conversation in isolation.

    Returns {uuid: {"category": str, "tags": [str]}} or None if LLM unavailable.
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
        logger.warning("LLM categorization returned unexpected format")
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
    Given a directory's file list and JSON previews, use the LLM to identify
    the export format and describe how to extract conversations from it.

    Returns a dict describing the format, or None if detection fails.
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
  "conversations_path": "$.conversations or top-level array etc",
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
    on_token: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Answer a query using retrieved context chunks.

    context_chunks: list of dicts with keys: text, title, source_type, created_at
    conversation_history: list of {"role": "user"|"assistant", "content": str}
    """
    if not context_chunks:
        context_text = "No relevant memories found in your brain for this query."
    else:
        parts = []
        for chunk in context_chunks:
            source = chunk.get("title", "Unknown")
            date = chunk.get("created_at", "")[:10]
            text = chunk.get("text", "").strip()
            parts.append(f"[{source} — {date}]\n{text}")
        context_text = "\n\n---\n\n".join(parts)

    history_text = ""
    if conversation_history:
        for turn in conversation_history[-6:]:
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

    if on_token:
        result = stream_ollama(prompt, model=model, system=system, base_url=base_url, on_token=on_token)
        return result or None
    return _call_ollama(prompt, model=model, system=system, base_url=base_url, timeout=120.0)


# ─── Follow-up suggestions ───────────────────────────────────────────────────

def suggest_followups(
    query: str,
    response: str,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> list[str]:
    """
    Given a Q&A exchange, return 2-3 short follow-up questions the user might want to ask.
    Returns [] on failure or if nothing meaningful suggested.
    """
    prompt = f"""Given this Q&A from a personal second brain chat, suggest 2-3 short follow-up questions.

Q: {query[:400]}
A: {response[:400]}

Rules:
- Each question under 12 words
- Make them specific, not generic ("tell me more")
- Return ONLY a JSON array of strings, e.g. ["...", "...", "..."]
- Return [] if no natural follow-ups exist

JSON array:"""

    result = _call_ollama(prompt, model=model, base_url=base_url, timeout=30.0)
    if not result:
        return []
    parsed = _extract_json(result)
    if not isinstance(parsed, list):
        return []
    return [str(q) for q in parsed if isinstance(q, str) and q.strip()][:3]


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

    def _clip(text: str, limit: int = 280) -> str:
        compact = " ".join(str(text).split())
        return compact[:limit]

    transcript_parts = []
    char_count = 0
    truncated = False
    for turn in turns:
        line = f"{turn['role'].upper()}: {turn['content']}"
        if char_count + len(line) > max_chars:
            truncated = True
            break
        transcript_parts.append(line)
        char_count += len(line)

    if truncated:
        first_turns = turns[:4]
        last_turns = turns[-8:] if len(turns) > 8 else turns
        sampled_lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for turn in first_turns + last_turns:
            key = (turn.get("role", ""), turn.get("content", ""))
            if key in seen:
                continue
            seen.add(key)
            sampled_lines.append(f"{turn['role'].upper()}: {_clip(turn['content'])}")
        transcript_parts = sampled_lines + ["[...middle of long session omitted for brevity...]"]

    transcript = "\n".join(transcript_parts)

    prompt = f"""Summarize this chat session in 2-4 sentences. Capture:
- What topics were discussed
- Any decisions made or conclusions reached
- Any projects, tools, or ideas mentioned

Be specific. Second person: "You discussed...", "You decided..."
No bullet points. Prose only. Under 150 words.
If the session excerpt is partial or truncated, summarize the visible parts only.
Do not apologize, ask for more context, or mention truncation unless it directly changes a conclusion.

Session:
{transcript}

Summary:"""

    result = _call_ollama(prompt, model=model, base_url=base_url, timeout=60.0)
    if not result:
        return None
    lowered = result.lower()
    refusal_markers = (
        "i'm sorry",
        "i’m sorry",
        "not enough information",
        "could you share the full",
        "provided was truncated",
    )
    if any(marker in lowered for marker in refusal_markers):
        summary_bits: list[str] = []
        user_topics = [
            _clip(turn["content"], 120)
            for turn in turns
            if turn.get("role") == "user" and turn.get("content")
        ]
        assistant_points = [
            _clip(turn["content"], 120)
            for turn in turns
            if turn.get("role") == "assistant" and turn.get("content")
        ]
        if user_topics:
            summary_bits.append(f"You discussed {user_topics[0]}")
        if len(user_topics) > 1:
            summary_bits.append(f"You also covered {user_topics[1]}")
        if assistant_points:
            summary_bits.append(f"Key response details included {assistant_points[0]}")
        fallback = ". ".join(summary_bits).strip()
        return (fallback + ".")[:280] if fallback else "You discussed a long session with multiple topics and outcomes."
    return result
