"""
src/adapters/obsidian.py — Load Obsidian vault notes into Document objects.

Handles:
  - YAML frontmatter (title, tags, status, date fields)
  - Wikilinks [[Note Title]] and [[Note Title|Alias]]
  - Inline tags (#tag)
  - Backlink index (derived by scanning all notes for links pointing to each note)
  - Folder path as implicit hierarchy in metadata
  - Staleness detection (>12 months old, no inbound links, no status: active)

Document ID: sha256 of the relative path (stable across re-runs).
Privacy routing:
  - My Brain vault    → VaultName.MY_BRAIN,    PrivacyLevel.PUBLIC
  - Private Brain vault → VaultName.PRIVATE_BRAIN, PrivacyLevel.PRIVATE

The body is the raw markdown (frontmatter included) — the chunker strips
frontmatter before splitting at headings.
"""

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.models import Document, PrivacyLevel, SourceType, VaultName

logger = logging.getLogger(__name__)

# How old a note must be to be considered "potentially stale"
STALE_AGE_DAYS = 365


# ─── Frontmatter ──────────────────────────────────────────────────────────────

def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter from a markdown file.
    Returns (frontmatter_dict, body_without_frontmatter).
    If no frontmatter, returns ({}, original content).

    Handles:
      - Scalar values:  key: value
      - Inline lists:   key: [a, b, c]
      - Block lists:    key:\n  - a\n  - b
    """
    if not content.startswith("---"):
        return {}, content

    end = content.find("\n---", 3)
    if end == -1:
        return {}, content

    fm_text = content[3:end].strip()
    body = content[end + 4:].lstrip("\n")

    fm: dict = {}
    current_key: str | None = None

    for line in fm_text.splitlines():
        stripped = line.strip()

        # Block list item: "  - value"
        if stripped.startswith("- ") and current_key is not None:
            item = stripped[2:].strip().strip('"').strip("'")
            if item:
                if not isinstance(fm.get(current_key), list):
                    fm[current_key] = []
                fm[current_key].append(item)
            continue

        if ":" not in line:
            current_key = None
            continue

        key, _, val = line.partition(":")
        key = key.strip()
        if not key or key.startswith(" "):
            # Nested value under current_key — skip
            continue

        val = val.strip()
        current_key = key

        if val.startswith("[") and val.endswith("]"):
            # Inline list
            items = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",")]
            fm[key] = [i for i in items if i]
        elif val == "":
            # Value will follow as block list items or be empty
            fm[key] = []
        else:
            fm[key] = val.strip('"').strip("'")
            current_key = None  # scalar — no list items expected

    return fm, body


# ─── Wikilink extraction ───────────────────────────────────────────────────────

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

def _extract_wikilinks(content: str) -> list[str]:
    """Return list of linked note titles from [[...]] patterns."""
    return _WIKILINK_RE.findall(content)


# ─── Tag extraction ────────────────────────────────────────────────────────────

_INLINE_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_/-]*)")

def _extract_inline_tags(content: str) -> list[str]:
    return _INLINE_TAG_RE.findall(content)


# ─── Date parsing ─────────────────────────────────────────────────────────────

_DATE_FIELDS = ("date", "created", "created_at", "updated", "updated_at", "modified")

def _parse_date(value: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(value.split("T")[0], fmt.split("T")[0])
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _note_timestamps(fm: dict, file_path: Path) -> tuple[datetime, datetime]:
    """
    Derive created_at and updated_at from frontmatter or file mtime.
    Returns (created_at, updated_at) as timezone-aware datetimes.
    """
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    for field in ("date", "created", "created_at"):
        if field in fm and isinstance(fm[field], str):
            created_at = _parse_date(fm[field])
            if created_at:
                break

    for field in ("updated", "updated_at", "modified"):
        if field in fm and isinstance(fm[field], str):
            updated_at = _parse_date(fm[field])
            if updated_at:
                break

    if not created_at or not updated_at:
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        created_at = created_at or mtime
        updated_at = updated_at or mtime

    return created_at, updated_at


# ─── Staleness ─────────────────────────────────────────────────────────────────

def _is_stale(
    updated_at: datetime,
    fm: dict,
    inbound_link_count: int,
) -> bool:
    """
    A note is stale if:
      - Updated more than STALE_AGE_DAYS ago
      - AND has no inbound wikilinks
      - AND has no frontmatter `status: active`
    """
    age = (datetime.now(timezone.utc) - updated_at).days
    if age <= STALE_AGE_DAYS:
        return False
    if fm.get("status", "").lower() == "active":
        return False
    if inbound_link_count > 0:
        return False
    return True


# ─── Main loader ──────────────────────────────────────────────────────────────

def _doc_id(vault_dir: Path, file_path: Path) -> str:
    rel = str(file_path.relative_to(vault_dir))
    return "obsidian_" + hashlib.sha256(rel.encode()).hexdigest()[:16]


def load_vault(
    vault_dir: Path,
    vault_name: VaultName,
    privacy_level: PrivacyLevel,
    skip_stale: bool = False,
) -> list[Document]:
    """
    Load all markdown notes from an Obsidian vault directory.

    Args:
        vault_dir: Root path of the vault (e.g. Path("My Brain"))
        vault_name: VaultName.MY_BRAIN or VaultName.PRIVATE_BRAIN
        privacy_level: PrivacyLevel.PUBLIC or PrivacyLevel.PRIVATE
        skip_stale: If True, stale notes are excluded entirely (default: include but mark)

    Returns list of Document objects, one per note.
    """
    vault_dir = Path(vault_dir)
    if not vault_dir.exists():
        logger.warning(f"Vault directory not found: {vault_dir}")
        return []

    # Collect all markdown files (skip .obsidian/ config dir)
    md_files = [
        p for p in vault_dir.rglob("*.md")
        if ".obsidian" not in p.parts
    ]

    if not md_files:
        logger.warning(f"No markdown files found in {vault_dir.name}")
        return []

    logger.info(f"Found {len(md_files)} notes in {vault_dir.name}")

    # First pass: build title → doc_id index for backlink resolution
    title_to_id: dict[str, str] = {}
    note_data: list[tuple[Path, str, dict, str]] = []  # (path, content, fm, body)

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            continue

        fm, body = _parse_frontmatter(content)
        title = fm.get("title") or fm.get("name") or file_path.stem
        doc_id = _doc_id(vault_dir, file_path)
        title_to_id[title.lower()] = doc_id
        title_to_id[file_path.stem.lower()] = doc_id
        note_data.append((file_path, content, fm, body))

    # Second pass: build inbound link count per doc_id
    inbound_counts: dict[str, int] = {}
    for file_path, content, fm, body in note_data:
        for linked_title in _extract_wikilinks(content):
            target_id = title_to_id.get(linked_title.lower())
            if target_id:
                inbound_counts[target_id] = inbound_counts.get(target_id, 0) + 1

    # Third pass: build Document objects
    docs: list[Document] = []
    stale_count = 0

    for file_path, content, fm, body in note_data:
        doc_id = _doc_id(vault_dir, file_path)
        title = fm.get("title") or fm.get("name") or file_path.stem
        created_at, updated_at = _note_timestamps(fm, file_path)

        # Tags: frontmatter + inline
        fm_tags = fm.get("tags", [])
        if isinstance(fm_tags, str):
            fm_tags = [fm_tags]
        inline_tags = _extract_inline_tags(content)
        all_tags = list(dict.fromkeys(t.lower() for t in fm_tags + inline_tags))

        wikilinks = _extract_wikilinks(content)
        inbound = inbound_counts.get(doc_id, 0)
        stale = _is_stale(updated_at, fm, inbound)

        if stale:
            stale_count += 1
            if skip_stale:
                continue

        rel_path = str(file_path.relative_to(vault_dir))
        folder = str(file_path.parent.relative_to(vault_dir)) if file_path.parent != vault_dir else ""

        docs.append(Document(
            id=doc_id,
            source_type=SourceType.OBSIDIAN_NOTE,
            vault=vault_name,
            privacy_level=privacy_level,
            title=title,
            body=content,       # raw markdown (chunker strips frontmatter)
            created_at=created_at,
            updated_at=updated_at,
            note_path=rel_path,
            tags=all_tags,
            wikilinks=wikilinks,
            metadata={
                "folder": folder,
                "inbound_links": inbound,
                "stale": stale,
                "status": fm.get("status", ""),
                "vault": vault_name.value,
            },
        ))

    logger.info(
        f"Loaded {len(docs)} notes from {vault_dir.name} "
        f"({stale_count} stale — {'excluded' if skip_stale else 'included'})"
    )
    return docs
