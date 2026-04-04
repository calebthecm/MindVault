"""
src/adapters/pdf.py — Load PDF files into Document objects.

Uses pypdf for text extraction. One Document per PDF file.
Title = filename stem. Source type = PDF_DOCUMENT.

Usage:
    from src.adapters.pdf import load_pdf
    doc = load_pdf(Path("my-doc.pdf"))
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models import Document, PrivacyLevel, SourceType, VaultName

logger = logging.getLogger(__name__)


def _file_mtime(path: Path) -> datetime:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def load_pdf(
    path: Path,
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC,
    vault: VaultName = VaultName.NONE,
) -> Optional[Document]:
    """
    Extract text from a PDF and return a Document, or None if extraction fails.

    Each page is separated by a form-feed marker so the chunker can split on pages.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return None

    path = Path(path)
    if not path.exists():
        logger.error(f"PDF not found: {path}")
        return None

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        logger.error(f"Cannot open PDF {path.name}: {e}")
        return None

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")
        except Exception as e:
            logger.warning(f"Failed to extract page {i + 1} from {path.name}: {e}")

    if not pages:
        logger.warning(f"No text extracted from {path.name}")
        return None

    body = "\n\n".join(pages)
    doc_id = "pdf_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    mtime = _file_mtime(path)

    logger.info(f"[PDF] Loaded {len(pages)} pages from {path.name}")
    return Document(
        id=doc_id,
        source_type=SourceType.PDF_DOCUMENT,
        vault=vault,
        privacy_level=privacy_level,
        title=path.stem,
        body=body,
        created_at=mtime,
        updated_at=mtime,
        metadata={"file_path": str(path), "page_count": len(pages)},
    )


def load_pdfs_from_dir(
    directory: Path,
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC,
    vault: VaultName = VaultName.NONE,
) -> list[Document]:
    """Load all PDFs from a directory (non-recursive)."""
    directory = Path(directory)
    docs = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        doc = load_pdf(pdf_path, privacy_level=privacy_level, vault=vault)
        if doc:
            docs.append(doc)
    logger.info(f"Loaded {len(docs)} PDFs from {directory.name}/")
    return docs
