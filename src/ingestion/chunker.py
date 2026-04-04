"""
Chunker: splits Documents into embeddable Chunks.

Strategy by source type:
- Conversations: each speaker turn is one chunk (natural boundary).
  Turns longer than MAX_TURN_CHARS get split at paragraph boundaries.
- Memories/Projects: paragraph-based chunking with overlap.
- Obsidian notes: heading-section based chunking.

Chunks carry full source attribution so retrieval always knows where a
piece of text came from.
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from math import exp

from src.models import Chunk, Document, SourceType  # PDF_DOCUMENT added to SourceType

# Stable namespace UUID for deterministic chunk ID generation
_CHUNK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

logger = logging.getLogger(__name__)

MAX_TURN_CHARS = 2000       # Max chars per conversation turn chunk
MAX_PARA_CHARS = 1500       # Max chars for paragraph chunks
OVERLAP_CHARS = 150         # Overlap between paragraph chunks


def _freshness_score(updated_at: datetime) -> float:
    """Exponential decay. Conversations: half-life 180d. Notes: 365d (set by caller)."""
    now = datetime.now(timezone.utc)
    days_old = max(0, (now - updated_at).days)
    return exp(-0.693 * days_old / 180)


def _chunk_id(document_id: str, index: int) -> str:
    """Generate a deterministic UUID for a chunk (Qdrant requires valid UUIDs)."""
    return str(uuid.uuid5(_CHUNK_NS, f"{document_id}__chunk_{index}"))


def _split_at_paragraphs(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text into chunks at paragraph boundaries with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            # Keep last paragraph for overlap
            overlap_paras = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) <= overlap_chars:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current = overlap_paras
            current_len = overlap_len
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


def _chunk_conversation(doc: Document) -> list[Chunk]:
    """One chunk per speaker turn. Long turns get paragraph-split."""
    chunks = []
    index = 0
    freshness = _freshness_score(doc.updated_at)

    # Parse out speaker turns from the body
    # Body format: "[HUMAN]\ntext\n\n[ASSISTANT]\ntext\n\n..."
    # Also handles the leading Summary block
    turns = re.split(r"\n\n(?=\[(HUMAN|ASSISTANT)\]\n)", doc.body)

    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue

        # Extract speaker label
        speaker_match = re.match(r"^\[(HUMAN|ASSISTANT)\]\n", turn)
        speaker = None
        text = turn
        if speaker_match:
            speaker = speaker_match.group(1).lower()
            text = turn[speaker_match.end():].strip()

        if not text:
            continue

        # Split long turns at paragraph boundaries
        sub_texts = _split_at_paragraphs(text, MAX_TURN_CHARS, OVERLAP_CHARS) if len(text) > MAX_TURN_CHARS else [text]

        for sub_text in sub_texts:
            prefix = f"[{speaker.upper()}] " if speaker else ""
            chunks.append(Chunk(
                id=_chunk_id(doc.id, index),
                document_id=doc.id,
                source_type=doc.source_type,
                vault=doc.vault,
                privacy_level=doc.privacy_level,
                text=f"{prefix}{sub_text}",
                index=index,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                freshness_score=freshness,
                title=doc.title,
                conversation_uuid=doc.conversation_uuid,
                speaker=speaker,
                metadata={**doc.metadata, "turn_split": len(sub_texts) > 1},
            ))
            index += 1

    return chunks


def _chunk_by_paragraphs(doc: Document, half_life_days: int = 180) -> list[Chunk]:
    """Paragraph-based chunking for memories, projects, and notes."""
    now = datetime.now(timezone.utc)
    days_old = max(0, (now - doc.updated_at).days)
    freshness = exp(-0.693 * days_old / half_life_days)

    texts = _split_at_paragraphs(doc.body, MAX_PARA_CHARS, OVERLAP_CHARS)
    chunks = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        chunks.append(Chunk(
            id=_chunk_id(doc.id, i),
            document_id=doc.id,
            source_type=doc.source_type,
            vault=doc.vault,
            privacy_level=doc.privacy_level,
            text=text,
            index=i,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            freshness_score=freshness,
            title=doc.title,
            conversation_uuid=doc.conversation_uuid,
            note_path=doc.note_path,
            metadata=doc.metadata,
        ))
    return chunks


def _chunk_by_headings(doc: Document) -> list[Chunk]:
    """
    Split Obsidian notes at heading boundaries (h1/h2/h3).
    Each section (heading + body until next heading) becomes one chunk.
    Falls back to paragraph splitting within sections that are too long.
    """
    now = datetime.now(timezone.utc)
    days_old = max(0, (now - doc.updated_at).days)
    freshness = exp(-0.693 * days_old / 365)  # notes: 365-day half-life

    # Strip YAML frontmatter
    body = doc.body
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end != -1:
            body = body[end + 4:].lstrip("\n")

    # Split on headings (keep delimiter with each section)
    sections = re.split(r"(?m)(?=^#{1,3} )", body)
    sections = [s.strip() for s in sections if s.strip()]

    if not sections:
        return _chunk_by_paragraphs(doc, half_life_days=365)

    chunks = []
    index = 0
    for section in sections:
        # If section is too long, paragraph-split it
        if len(section) > MAX_PARA_CHARS:
            sub_texts = _split_at_paragraphs(section, MAX_PARA_CHARS, OVERLAP_CHARS)
        else:
            sub_texts = [section]

        for text in sub_texts:
            if not text.strip():
                continue
            chunks.append(Chunk(
                id=_chunk_id(doc.id, index),
                document_id=doc.id,
                source_type=doc.source_type,
                vault=doc.vault,
                privacy_level=doc.privacy_level,
                text=text,
                index=index,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                freshness_score=freshness,
                title=doc.title,
                note_path=doc.note_path,
                metadata=doc.metadata,
            ))
            index += 1

    return chunks or _chunk_by_paragraphs(doc, half_life_days=365)


def _chunk_pdf(doc: Document) -> list[Chunk]:
    """
    Split PDF documents by page (each [Page N] block is one chunk).
    Pages that are too long get paragraph-split within the page.
    """
    now = datetime.now(timezone.utc)
    days_old = max(0, (now - doc.updated_at).days)
    freshness = exp(-0.693 * days_old / 365)

    page_blocks = re.split(r"\n\n(?=\[Page \d+\]\n)", doc.body)
    chunks = []
    index = 0

    for block in page_blocks:
        block = block.strip()
        if not block:
            continue

        if len(block) > MAX_PARA_CHARS:
            sub_texts = _split_at_paragraphs(block, MAX_PARA_CHARS, OVERLAP_CHARS)
        else:
            sub_texts = [block]

        for text in sub_texts:
            if not text.strip():
                continue
            chunks.append(Chunk(
                id=_chunk_id(doc.id, index),
                document_id=doc.id,
                source_type=doc.source_type,
                vault=doc.vault,
                privacy_level=doc.privacy_level,
                text=text,
                index=index,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                freshness_score=freshness,
                title=doc.title,
                metadata=doc.metadata,
            ))
            index += 1

    return chunks or _chunk_by_paragraphs(doc, half_life_days=365)


def chunk_document(doc: Document) -> list[Chunk]:
    """Dispatch to the right chunking strategy based on source type."""
    if doc.source_type == SourceType.ANTHROPIC_CONVERSATION:
        chunks = _chunk_conversation(doc)
    elif doc.source_type == SourceType.OBSIDIAN_NOTE:
        chunks = _chunk_by_headings(doc)
    elif doc.source_type == SourceType.PDF_DOCUMENT:
        chunks = _chunk_pdf(doc)
    else:
        # memories, projects, generic
        chunks = _chunk_by_paragraphs(doc, half_life_days=180)

    logger.debug(f"Chunked '{doc.title}' → {len(chunks)} chunks")
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    logger.info(f"Total chunks from {len(docs)} documents: {len(all_chunks)}")
    return all_chunks
