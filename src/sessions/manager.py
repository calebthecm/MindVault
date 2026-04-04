"""
Session manager — chat session lifecycle.

Sessions saved as gzip-compressed JSON in sessions/<id>.json.gz.
index.json tracks metadata for fast listing without decompressing.

Session status:
  raw       — turns captured, not yet LLM-processed
  processed — summary and entities filled in by LLM at session end
"""

import gzip
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_filename(session_id: str) -> str:
    return f"{session_id}.json.gz"


def _index_path(sessions_dir: Path) -> Path:
    return sessions_dir / "index.json"


def _load_index(sessions_dir: Path) -> list[dict]:
    path = _index_path(sessions_dir)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _save_index(sessions_dir: Path, index: list[dict]) -> None:
    _index_path(sessions_dir).write_text(json.dumps(index, indent=2))


class Session:
    """
    A single chat session. Add turns as they happen, save at session end.
    """

    def __init__(self, sessions_dir: Path, model: str):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            + "_" + uuid.uuid4().hex[:6]
        )
        self.started_at = _now_iso()
        self.model = model
        self.turns: list[dict] = []
        self.summary = ""
        self.entities: list[dict] = []
        self.status = "raw"

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content, "ts": _now_iso()})

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": _now_iso(),
            "model": self.model,
            "status": self.status,
            "turns": self.turns,
            "summary": self.summary,
            "entities": self.entities,
        }

    def save(self) -> Path:
        path = self.sessions_dir / _session_filename(self.session_id)
        data = json.dumps(self.to_dict()).encode()
        path.write_bytes(gzip.compress(data, compresslevel=6))
        return path

    def save_and_index(self) -> Path:
        path = self.save()
        index = _load_index(self.sessions_dir)

        preview = ""
        if self.summary:
            preview = self.summary[:120]
        elif self.turns:
            preview = self.turns[0]["content"][:80]

        entry = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "model": self.model,
            "status": self.status,
            "preview": preview,
            "turn_count": len(self.turns),
        }
        updated = [e for e in index if e["session_id"] != self.session_id]
        updated.append(entry)
        _save_index(self.sessions_dir, updated)
        return path


def load_session(sessions_dir: Path, session_id: str) -> Optional[Session]:
    path = Path(sessions_dir) / _session_filename(session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(gzip.decompress(path.read_bytes()))
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        return None

    session = Session(Path(sessions_dir), model=data.get("model", ""))
    session.session_id = data["session_id"]
    session.started_at = data["started_at"]
    session.turns = data.get("turns", [])
    session.summary = data.get("summary", "")
    session.entities = data.get("entities", [])
    session.status = data.get("status", "raw")
    return session


def load_last_session(sessions_dir: Path) -> Optional[Session]:
    index = _load_index(Path(sessions_dir))
    if not index:
        return None
    latest = max(index, key=lambda e: e["started_at"])
    return load_session(Path(sessions_dir), latest["session_id"])


def list_sessions(sessions_dir: Path) -> list[dict]:
    index = _load_index(Path(sessions_dir))
    return sorted(index, key=lambda e: e["started_at"], reverse=True)
