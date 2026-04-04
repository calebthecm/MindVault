"""
mindvault/version.py — Semver version tracking for MindVault.

Version format: MAJOR.MINOR.PATCH
  MAJOR — breaking changes to storage format or public API
  MINOR — new features (new modes, new adapters, new memory layers)
  PATCH — bug fixes, minor improvements

Latest version is fetched from the update API at startup.
If the API is unreachable the check is silently skipped.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# The version this running instance is at
CURRENT_VERSION = "0.5.1"

# Remote update API — returns {"version": "x.x.x"}
UPDATE_API_URL = "https://mindvaultapiupdate.cryptclouds.org/"

# Cached result of the last fetch (populated once per process, in background)
_fetched_latest: Optional[str] = None
_fetch_done = threading.Event()


def _fetch_latest() -> None:
    """Fetch the latest version from the update API (runs in a background thread)."""
    global _fetched_latest
    try:
        import httpx
        resp = httpx.get(UPDATE_API_URL, timeout=4.0)
        resp.raise_for_status()
        data = resp.json()
        _fetched_latest = str(data.get("version", "")).strip()
    except Exception as e:
        logger.debug(f"Version check failed: {e}")
    finally:
        _fetch_done.set()


def fetch_in_background() -> None:
    """
    Kick off the version check in a daemon thread.
    Call once at startup — does not block.
    """
    t = threading.Thread(target=_fetch_latest, daemon=True)
    t.start()


def latest_version(wait: bool = False, timeout: float = 4.0) -> Optional[str]:
    """
    Return the fetched latest version string, or None if unavailable.

    wait=True blocks until the background fetch completes (or timeout).
    wait=False returns immediately with whatever is cached so far.
    """
    if wait:
        _fetch_done.wait(timeout=timeout)
    return _fetched_latest


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.strip().split("."))
    except ValueError:
        return (0,)


def update_available() -> Optional[str]:
    """
    Return the latest version string if it's strictly newer than CURRENT_VERSION.
    Returns None if already up to date or the check hasn't completed.
    """
    remote = latest_version()
    if not remote:
        return None
    if _version_tuple(remote) > _version_tuple(CURRENT_VERSION):
        return remote
    return None


def release_label() -> Optional[str]:
    """Return 'New Release x.x.x' if update available, else None."""
    ver = update_available()
    return f"New Release {ver}" if ver else None
