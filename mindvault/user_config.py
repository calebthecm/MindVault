"""
mindvault/user_config.py — Persistent user preferences.

Stored in {BRAIN_DIR}/.mindvault.json (gitignored, local-only).
Keeps user name, setup status, and any other personal prefs out of
version-controlled files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent.parent / ".mindvault.json"

_DEFAULTS: dict[str, Any] = {
    "user_name": "",
    "setup_complete": False,
}


def _load_raw() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_raw(data: dict[str, Any]) -> None:
    _CONFIG_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get(key: str, default: Any = None) -> Any:
    """Read a single preference. Falls back to _DEFAULTS then `default`."""
    data = _load_raw()
    if key in data:
        return data[key]
    return _DEFAULTS.get(key, default)


def set(key: str, value: Any) -> None:  # noqa: A001
    """Write a single preference."""
    data = _load_raw()
    data[key] = value
    _save_raw(data)


def all_prefs() -> dict[str, Any]:
    """Return all stored preferences merged with defaults."""
    return {**_DEFAULTS, **_load_raw()}


# ── Convenience accessors ─────────────────────────────────────────────────────

def get_name() -> str:
    return get("user_name", "")


def set_name(name: str) -> None:
    set("user_name", name.strip())


def is_setup_complete() -> bool:
    return bool(get("setup_complete", False))


def mark_setup_complete() -> None:
    set("setup_complete", True)
