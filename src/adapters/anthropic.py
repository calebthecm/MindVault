"""
Anthropic export adapter.

Parses the files inside a data-{timestamp}-batch-{n}/ directory and
produces normalized Document objects. Handles:
  - conversations.json  → one Document per conversation (turns concatenated as body)
  - memories.json       → one Document for conversations_memory + one per project memory
  - projects.json       → one Document per project (prompt_template + docs)
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models import Document, PrivacyLevel, SourceType, VaultName

logger = logging.getLogger(__name__)


def _parse_iso(ts: str) -> datetime:
    """Parse ISO8601 string → aware datetime (UTC)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _extract_message_text(message: dict) -> str:
    """
    Extract plain text from a chat message.
    The top-level `text` field is often empty — always use content[].text.
    """
    parts = []
    for block in message.get("content", []):
        if block.get("type") == "text" and block.get("text", "").strip():
            parts.append(block["text"].strip())
    # Fallback to top-level text if content array was empty
    if not parts and message.get("text", "").strip():
        parts.append(message["text"].strip())
    return "\n".join(parts)


def _freshness_score(updated_at: datetime) -> float:
    """Exponential decay: half-life 180 days for conversations."""
    now = datetime.now(timezone.utc)
    days_old = (now - updated_at).days
    import math
    return math.exp(-0.693 * days_old / 180)


def parse_conversations(export_dir: Path) -> list[Document]:
    """Parse conversations.json → list of Document."""
    path = export_dir / "conversations.json"
    if not path.exists():
        logger.warning(f"conversations.json not found in {export_dir}")
        return []

    with open(path) as f:
        convos = json.load(f)

    documents = []
    for convo in convos:
        uuid = convo["uuid"]
        name = convo.get("name", "Untitled conversation")
        summary = convo.get("summary", "")
        created_at = _parse_iso(convo["created_at"])
        updated_at = _parse_iso(convo["updated_at"])

        # Build body: interleaved speaker turns
        turns = []
        for msg in convo.get("chat_messages", []):
            speaker = msg.get("sender", "unknown")
            text = _extract_message_text(msg)
            if text:
                turns.append(f"[{speaker.upper()}]\n{text}")

        if not turns:
            logger.debug(f"Skipping empty conversation: {name} ({uuid})")
            continue

        body = "\n\n".join(turns)
        if summary:
            body = f"Summary: {summary}\n\n---\n\n{body}"

        doc = Document(
            id=f"anthropic_conv_{uuid}",
            source_type=SourceType.ANTHROPIC_CONVERSATION,
            vault=VaultName.NONE,
            privacy_level=PrivacyLevel.PUBLIC,
            title=name,
            body=body,
            created_at=created_at,
            updated_at=updated_at,
            conversation_uuid=uuid,
            metadata={
                "summary": summary,
                "message_count": len(convo.get("chat_messages", [])),
                "has_attachments": any(
                    msg.get("attachments") or msg.get("files")
                    for msg in convo.get("chat_messages", [])
                ),
            },
        )
        documents.append(doc)

    logger.info(f"Parsed {len(documents)} conversations from {export_dir.name}")
    return documents


def parse_memories(export_dir: Path) -> list[Document]:
    """Parse memories.json → Document(s)."""
    path = export_dir / "memories.json"
    if not path.exists():
        return []

    with open(path) as f:
        entries = json.load(f)

    documents = []
    for entry in entries:
        account_uuid = entry.get("account_uuid", "unknown")

        # Conversations memory — the long freeform Claude-maintained memory
        conv_memory = entry.get("conversations_memory", "").strip()
        if conv_memory:
            stable_id = hashlib.sha256(f"anthropic_memory_{account_uuid}".encode()).hexdigest()[:16]
            documents.append(Document(
                id=f"anthropic_memory_{stable_id}",
                source_type=SourceType.ANTHROPIC_MEMORY,
                vault=VaultName.NONE,
                privacy_level=PrivacyLevel.PUBLIC,
                title="Claude Memory — Conversations",
                body=conv_memory,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                metadata={"account_uuid": account_uuid, "memory_type": "conversations"},
            ))

        # Per-project memories
        for project_uuid, memory_text in entry.get("project_memories", {}).items():
            if not memory_text or not memory_text.strip():
                continue
            stable_id = hashlib.sha256(f"anthropic_proj_memory_{project_uuid}".encode()).hexdigest()[:16]
            documents.append(Document(
                id=f"anthropic_proj_memory_{stable_id}",
                source_type=SourceType.ANTHROPIC_MEMORY,
                vault=VaultName.NONE,
                privacy_level=PrivacyLevel.PUBLIC,
                title=f"Claude Memory — Project {project_uuid[:8]}",
                body=memory_text.strip(),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                project_uuid=project_uuid,
                metadata={"account_uuid": account_uuid, "memory_type": "project"},
            ))

    logger.info(f"Parsed {len(documents)} memory documents from {export_dir.name}")
    return documents


def parse_projects(export_dir: Path) -> list[Document]:
    """Parse projects.json → Document per project."""
    path = export_dir / "projects.json"
    if not path.exists():
        return []

    with open(path) as f:
        projects = json.load(f)

    documents = []
    for proj in projects:
        uuid = proj["uuid"]
        name = proj.get("name", "Untitled project")
        description = proj.get("description", "").strip()
        prompt_template = proj.get("prompt_template", "").strip()
        created_at = _parse_iso(proj["created_at"])
        updated_at = _parse_iso(proj["updated_at"])

        parts = []
        if description:
            parts.append(f"Description: {description}")
        if prompt_template:
            parts.append(f"System prompt:\n{prompt_template}")
        for doc in proj.get("docs", []):
            if doc.get("content", "").strip():
                parts.append(f"Document — {doc.get('filename', 'unnamed')}:\n{doc['content'].strip()}")

        if not parts:
            continue

        body = "\n\n".join(parts)
        # Skip the default "How to use Claude" starter project
        if proj.get("is_starter_project"):
            logger.debug(f"Skipping starter project: {name}")
            continue

        documents.append(Document(
            id=f"anthropic_project_{uuid}",
            source_type=SourceType.ANTHROPIC_PROJECT,
            vault=VaultName.NONE,
            privacy_level=PrivacyLevel.PUBLIC,
            title=f"Project: {name}",
            body=body,
            created_at=created_at,
            updated_at=updated_at,
            project_uuid=uuid,
            metadata={
                "is_private": proj.get("is_private", False),
                "doc_count": len(proj.get("docs", [])),
            },
        ))

    logger.info(f"Parsed {len(documents)} projects from {export_dir.name}")
    return documents


def load_export(export_dir: Path) -> list[Document]:
    """
    Load all documents from one Anthropic export batch directory.
    Returns combined list of all document types.
    """
    export_dir = Path(export_dir)
    if not export_dir.is_dir():
        raise ValueError(f"Not a directory: {export_dir}")

    docs = []
    docs.extend(parse_conversations(export_dir))
    docs.extend(parse_memories(export_dir))
    docs.extend(parse_projects(export_dir))
    logger.info(f"Total documents from {export_dir.name}: {len(docs)}")
    return docs
