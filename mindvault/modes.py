"""
mindvault/modes.py — Chat mode definitions and cycling logic.

Modes cycle with Shift+Tab. Each mode changes how retrieved memories
are processed and how the council reasons about the answer.

Modes:
  CHAT     — Standard RAG: retrieve → synthesize → respond
  PLAN     — Break the query into a structured step-by-step plan
  DECIDE   — Council votes; majority view wins with dissenting notes
  DEBATE   — Two opposing voices argue; you see both sides
  REFLECT  — Deep introspective synthesis: what does my brain really know?
  EXPLORE  — Graph-traversal deep dive; follows links to surface surprises
"""

from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    CHAT = "CHAT"
    PLAN = "PLAN"
    DECIDE = "DECIDE"
    DEBATE = "DEBATE"
    REFLECT = "REFLECT"
    EXPLORE = "EXPLORE"


@dataclass(frozen=True)
class ModeConfig:
    name: Mode
    label: str          # Short display label for the status bar
    description: str    # One-line description shown on mode switch
    icon: str           # Single character icon


MODES: list[ModeConfig] = [
    ModeConfig(
        name=Mode.CHAT,
        label="CHAT",
        description="Standard RAG — retrieve memories and synthesize an answer.",
        icon="💬",
    ),
    ModeConfig(
        name=Mode.PLAN,
        label="PLAN",
        description="Structured planning — break the task into actionable steps.",
        icon="📋",
    ),
    ModeConfig(
        name=Mode.DECIDE,
        label="DECIDE",
        description="Council vote — multiple perspectives weigh in, majority rules.",
        icon="🗳",
    ),
    ModeConfig(
        name=Mode.DEBATE,
        label="DEBATE",
        description="Opposing views — for and against, let both sides speak.",
        icon="⚖",
    ),
    ModeConfig(
        name=Mode.REFLECT,
        label="REFLECT",
        description="Deep synthesis — what does your brain really know about this?",
        icon="🔍",
    ),
    ModeConfig(
        name=Mode.EXPLORE,
        label="EXPLORE",
        description="Graph traversal — follow links to surface surprising connections.",
        icon="🕸",
    ),
]

_MODE_CYCLE = [m.name for m in MODES]
_MODE_CONFIG_MAP = {m.name: m for m in MODES}


def get_config(mode: Mode) -> ModeConfig:
    return _MODE_CONFIG_MAP[mode]


def next_mode(current: Mode) -> Mode:
    """Advance to the next mode (wraps around)."""
    idx = _MODE_CYCLE.index(current)
    return _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]


def prev_mode(current: Mode) -> Mode:
    """Go to the previous mode (wraps around)."""
    idx = _MODE_CYCLE.index(current)
    return _MODE_CYCLE[(idx - 1) % len(_MODE_CYCLE)]
