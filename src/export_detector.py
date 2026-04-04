"""
src/export_detector.py — Auto-detect and parse any AI conversation export.

Instead of hardcoded parsers per service, this module:
  1. Scans the Brain directory for any folder containing JSON files
  2. Peeks at the file contents
  3. Uses llama3.2 to identify the format and describe the data structure
  4. Extracts conversations using the LLM's instructions
  5. Falls back to known-format parsers (Anthropic, OpenAI) if LLM detection fails

This means adding OpenAI data is as simple as dropping the export folder in Brain/.
The system will figure out the rest.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from mindvault.config import BRAIN_DIR, EXCLUDED_DIRS, LLM_MODEL, OLLAMA_BASE
from src.llm import detect_export_format
from src.models import Document, PrivacyLevel, SourceType, VaultName

logger = logging.getLogger(__name__)

# How many bytes to preview from each JSON file for format detection
PREVIEW_BYTES = 800


# ─── Directory discovery ──────────────────────────────────────────────────────

def find_export_dirs(base_dir: Path = BRAIN_DIR) -> list[Path]:
    """
    Find all directories under base_dir that contain indexable content.
    A directory qualifies if it has at least one .json, .pdf, .txt, or .md file.
    Known non-export directories (vaults, source code, etc.) are excluded.
    """
    exports = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name in EXCLUDED_DIRS:
            continue
        indexable = (
            list(entry.glob("*.json"))
            + list(entry.glob("*.pdf"))
            + list(entry.glob("*.txt"))
            + list(entry.glob("*.md"))
        )
        if indexable:
            exports.append(entry)
            logger.info(f"Found export dir: {entry.name} ({len(indexable)} indexable files)")
    return exports


# ─── Known-format parsers ─────────────────────────────────────────────────────

def _parse_anthropic(export_dir: Path) -> list[dict]:
    """
    Parse Anthropic Claude export format.
    conversations.json → list of raw conversation dicts.
    """
    path = export_dir / "conversations.json"
    if not path.exists():
        return []
    with open(path) as f:
        convos = json.load(f)
    logger.info(f"[Anthropic] Loaded {len(convos)} conversations from {export_dir.name}")
    return convos


def _parse_openai(export_dir: Path) -> list[dict]:
    """
    Parse OpenAI ChatGPT export format.

    OpenAI schema:
    [
      {
        "title": "...",
        "create_time": 1234567890.0,
        "update_time": 1234567890.0,
        "mapping": {
          "msg-id": {
            "message": {
              "author": {"role": "user"|"assistant"|"system"},
              "content": {"parts": ["text here"]},
              "create_time": 1234567890.0
            }
          }
        }
      }
    ]
    """
    path = export_dir / "conversations.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)

    normalized = []
    for convo in raw:
        title = convo.get("title", "Untitled")
        create_time = convo.get("create_time", 0)
        update_time = convo.get("update_time", create_time)

        from datetime import datetime, timezone
        def ts_to_iso(ts):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""

        messages = []
        mapping = convo.get("mapping", {})
        # Build ordered message list by traversing parent chain
        # Simple approach: sort by create_time
        raw_msgs = []
        for node in mapping.values():
            msg = node.get("message")
            if not msg:
                continue
            role = msg.get("author", {}).get("role", "")
            if role not in ("user", "assistant"):
                continue
            parts = msg.get("content", {}).get("parts", [])
            text = " ".join(str(p) for p in parts if isinstance(p, str)).strip()
            if not text:
                continue
            raw_msgs.append({
                "role": role,
                "text": text,
                "ts": msg.get("create_time", 0) or 0,
            })

        raw_msgs.sort(key=lambda m: m["ts"])
        for m in raw_msgs:
            messages.append({
                "sender": m["role"],
                "text": m["text"],
                "created_at": ts_to_iso(m["ts"]),
            })

        normalized.append({
            "uuid": convo.get("id", title),
            "name": title,
            "summary": "",
            "created_at": ts_to_iso(create_time),
            "updated_at": ts_to_iso(update_time),
            "chat_messages": [
                {
                    "uuid": f"msg_{i}",
                    "sender": m["sender"],
                    "text": m["text"],
                    "content": [{"type": "text", "text": m["text"]}],
                    "created_at": m["created_at"],
                    "updated_at": m["created_at"],
                    "attachments": [],
                    "files": [],
                }
                for i, m in enumerate(messages)
            ],
        })

    logger.info(f"[OpenAI] Loaded {len(normalized)} conversations from {export_dir.name}")
    return normalized


# ─── LLM-powered generic parser ───────────────────────────────────────────────

def _preview_files(export_dir: Path) -> dict[str, str]:
    """Read the first PREVIEW_BYTES of each JSON file for LLM inspection."""
    previews = {}
    for f in export_dir.glob("*.json"):
        try:
            with open(f) as fp:
                previews[f.name] = fp.read(PREVIEW_BYTES)
        except Exception:
            pass
    return previews


def _detect_and_parse_generic(export_dir: Path) -> Optional[list[dict]]:
    """
    Use llama3.2 to understand an unknown export format, then extract conversations.
    Returns normalized conversation list or None if detection fails.
    """
    previews = _preview_files(export_dir)
    if not previews:
        return None

    logger.info(f"Asking LLM to detect format of '{export_dir.name}'...")
    fmt = detect_export_format(
        export_dir_path=str(export_dir),
        file_previews=previews,
        model=LLM_MODEL,
        base_url=OLLAMA_BASE,
    )

    if not fmt:
        logger.warning("LLM format detection failed")
        return None

    logger.info(f"LLM detected format: {fmt}")
    source = fmt.get("source", "unknown").lower()

    # Route to known parser if LLM identified the source
    if "anthropic" in source:
        return _parse_anthropic(export_dir)
    if "openai" in source or "chatgpt" in source:
        return _parse_openai(export_dir)

    # Unknown format — try to use LLM's field descriptions to extract data
    logger.warning(f"Unknown format '{source}' — attempting generic extraction")
    conv_file = fmt.get("conversations_file", "conversations.json")
    path = export_dir / conv_file
    if not path.exists():
        logger.error(f"Conversations file '{conv_file}' not found in {export_dir}")
        return None

    with open(path) as f:
        raw = json.load(f)

    # The LLM told us the field names — try to use them
    title_field = fmt.get("title_field", "name")
    messages_field = fmt.get("messages_field", "chat_messages")
    text_field = fmt.get("message_text_field", "text")
    role_field = fmt.get("message_role_field", "sender")
    ts_field = fmt.get("timestamp_field", "created_at")

    if not isinstance(raw, list):
        raw = [raw]

    normalized = []
    for item in raw:
        title = item.get(title_field, item.get("title", item.get("name", "Untitled")))
        messages_raw = item.get(messages_field, [])
        messages = []
        for m in messages_raw:
            text = m.get(text_field, m.get("text", "")).strip()
            role = m.get(role_field, m.get("sender", m.get("role", "unknown")))
            if text:
                messages.append({
                    "uuid": m.get("id", m.get("uuid", "")),
                    "sender": role,
                    "text": text,
                    "content": [{"type": "text", "text": text}],
                    "created_at": m.get(ts_field, ""),
                    "updated_at": m.get(ts_field, ""),
                    "attachments": [],
                    "files": [],
                })
        normalized.append({
            "uuid": item.get("id", item.get("uuid", title)),
            "name": title,
            "summary": item.get("summary", ""),
            "created_at": item.get(ts_field, ""),
            "updated_at": item.get("updated_at", item.get(ts_field, "")),
            "chat_messages": messages,
        })

    logger.info(f"[Generic] Extracted {len(normalized)} conversations via LLM-guided parsing")
    return normalized


# ─── Main entry point ─────────────────────────────────────────────────────────

def load_conversations_from_dir(export_dir: Path) -> list[dict]:
    """
    Load conversations from any export directory.
    Tries known formats first (fast), then falls back to LLM detection.
    Returns a normalized list of conversation dicts.
    """
    export_dir = Path(export_dir)
    files = {f.name for f in export_dir.glob("*.json")}

    # Anthropic exports always have 'users.json' alongside conversations
    if "users.json" in files and "conversations.json" in files:
        return _parse_anthropic(export_dir)

    # OpenAI exports have 'conversations.json' but no 'users.json'
    # They also typically have a 'user.json' (singular)
    if "conversations.json" in files and "user.json" in files:
        return _parse_openai(export_dir)

    # Unknown — let llama3.2 figure it out
    result = _detect_and_parse_generic(export_dir)
    if result is not None:
        return result

    # Last resort: if there's any conversations.json, try Anthropic format
    if "conversations.json" in files:
        logger.warning(f"Could not detect format, trying Anthropic parser as last resort")
        return _parse_anthropic(export_dir)

    logger.error(f"Could not load conversations from {export_dir.name}")
    return []
