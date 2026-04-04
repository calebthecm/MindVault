"""
Core data models for the Brain RAG system.
All ingestion adapters normalize their output to Document.
Chunking produces Chunk objects, which are what gets embedded and stored.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    ANTHROPIC_CONVERSATION = "anthropic_conversation"
    ANTHROPIC_MEMORY = "anthropic_memory"
    ANTHROPIC_PROJECT = "anthropic_project"
    OPENAI_CONVERSATION = "openai_conversation"
    OBSIDIAN_NOTE = "obsidian_note"


class VaultName(str, Enum):
    MY_BRAIN = "my_brain"
    PRIVATE_BRAIN = "private_brain"
    # Conversation data is not vault-scoped
    NONE = "none"


class PrivacyLevel(str, Enum):
    PUBLIC = "public"      # My Brain, general conversation data
    PRIVATE = "private"    # Private Brain content — requires explicit opt-in to retrieve


@dataclass
class Document:
    """
    Normalized representation of any ingested content.
    Produced by adapters before chunking.
    """
    id: str                          # Stable unique ID (conversation uuid, note path hash, etc.)
    source_type: SourceType
    vault: VaultName
    privacy_level: PrivacyLevel
    title: str
    body: str                        # Full text content
    created_at: datetime
    updated_at: datetime
    metadata: dict = field(default_factory=dict)
    # Anthropic-specific
    conversation_uuid: Optional[str] = None
    project_uuid: Optional[str] = None
    # Obsidian-specific
    note_path: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """
    A single embeddable unit derived from a Document.
    This is what gets stored in the vector DB.
    """
    id: str                          # chunk_id = f"{doc_id}__chunk_{index}"
    document_id: str                 # Parent document ID
    source_type: SourceType
    vault: VaultName
    privacy_level: PrivacyLevel
    text: str                        # The actual text to embed
    index: int                       # Position within document
    created_at: datetime
    updated_at: datetime
    freshness_score: float = 1.0     # 0.0 (stale) to 1.0 (fresh)
    metadata: dict = field(default_factory=dict)
    # Source attribution fields — always preserved
    title: str = ""
    conversation_uuid: Optional[str] = None
    note_path: Optional[str] = None
    speaker: Optional[str] = None    # "human" | "assistant" for conversations
