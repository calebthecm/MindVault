"""
mindvault/session_picker.py — Interactive session picker.

Shows a scrollable list of sessions. Navigate with arrow keys, select
with Enter, cancel with Escape or Q.

Returns the selected session dict, or None if cancelled.
"""

from __future__ import annotations

import shutil
from typing import Optional

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

_PICKER_STYLE = Style.from_dict({
    "header":      "fg:#ff6a00 bold",
    "subheader":   "fg:#555555",
    "border":      "fg:#333333",
    "selected":    "fg:#ffffff bg:#ff6a00 bold",
    "normal":      "fg:#aaaaaa",
    "date":        "fg:#888888",
    "status-p":    "fg:#16a34a",       # processed — green
    "status-r":    "fg:#d97706",       # raw — amber
    "preview":     "fg:#cccccc",
    "footer":      "fg:#555555 italic",
    "turns":       "fg:#6b7280",
})

_CURSOR = "▶ "
_BLANK  = "  "


def _fmt_status(status: str) -> tuple[str, str]:
    """Return (style_class, text) for a session status."""
    if status == "processed":
        return "class:status-p", "✓ processed"
    return "class:status-r", "○ raw"


def _render_list(sessions: list[dict], cursor: int, cols: int) -> FormattedText:
    """Build the full FormattedText for the picker body."""
    parts: list[tuple[str, str]] = []

    sep = "─" * cols

    # ── Header ───────────────────────────────────────────────────────────────
    parts += [
        ("class:header",    "  Resume a session\n"),
        ("class:subheader", f"  {len(sessions)} session{'s' if len(sessions) != 1 else ''} found\n"),
        ("class:border",    sep + "\n"),
    ]

    # ── Session rows ──────────────────────────────────────────────────────────
    for i, s in enumerate(sessions):
        selected = i == cursor
        prefix_style = "class:selected" if selected else "class:normal"
        prefix = _CURSOR if selected else _BLANK

        date  = (s.get("started_at") or "")[:10]
        turns = s.get("turn_count", 0)
        status = s.get("status", "raw")
        raw_preview = (s.get("preview") or "").strip().replace("\n", " ")
        status_cls, status_txt = _fmt_status(status)

        # Build a fixed-width row: cursor + date + turns + status + preview
        # prefix(2) + date(10) + "  " + turns(8) + "  " + status(13) + "  " + preview
        used = 2 + 10 + 2 + 8 + 2 + 13 + 2
        preview_width = max(cols - used - 2, 10)
        preview = raw_preview[:preview_width]

        parts += [
            (prefix_style, prefix),
            ("class:date",   date),
            ("class:border", "  "),
            ("class:turns",  f"{turns:>3} turns"),
            ("class:border", "  "),
            (status_cls,     f"{status_txt:<13}"),
            ("class:border", "  "),
            ("class:preview" if not selected else "class:selected", preview),
            ("", "\n"),
        ]

    # ── Footer ────────────────────────────────────────────────────────────────
    parts += [
        ("class:border", sep + "\n"),
        ("class:footer",
         "  ↑↓ navigate   Enter select   Esc / q cancel\n"),
    ]

    return FormattedText(parts)


def pick_session(sessions: list[dict]) -> Optional[dict]:
    """
    Show the interactive session picker.
    Returns the chosen session dict, or None if cancelled.
    """
    if not sessions:
        return None

    cols = max(shutil.get_terminal_size((100, 24)).columns, 60)
    state = {"cursor": 0, "result": None, "done": False}
    n = len(sessions)

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:
        state["cursor"] = (state["cursor"] - 1) % n

    @kb.add("down")
    def _down(event) -> None:
        state["cursor"] = (state["cursor"] + 1) % n

    @kb.add("enter")
    def _enter(event) -> None:
        state["result"] = sessions[state["cursor"]]
        state["done"] = True
        event.app.exit()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event) -> None:
        state["done"] = True
        event.app.exit()

    def _get_text() -> FormattedText:
        return _render_list(sessions, state["cursor"], cols)

    control  = FormattedTextControl(_get_text, focusable=True)
    body     = Window(content=control, wrap_lines=False)
    layout   = Layout(HSplit([body]))

    app: Application = Application(
        layout=layout,
        key_bindings=kb,
        style=_PICKER_STYLE,
        full_screen=False,
        mouse_support=False,
        refresh_interval=None,
    )

    app.run()
    return state["result"]
