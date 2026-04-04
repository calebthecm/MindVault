"""
mindvault/tui.py — Claude Code-style TUI for MindVault.

Layout:
  ╭─── MindVault v0.3.0 ────────────────────────────────────────────────────────╮
  │                         │ Getting started                                    │
  │   Welcome back, Caleb!  │ ingest    index your data                         │
  │                         │ chat      start chatting                           │
  │      (brain logo)       │ Shift+Tab cycle modes                              │
  │                         │ ─────────────────────────────────────────────────  │
  │  MindVault v0.3.0       │ Recent sessions                                    │
  │  llama3.2 · nomic       │ 2026-04-03  "What is MindVault?"                  │
  │                         │                                                    │
  │  Personal use — free    │                                                    │
  │  Commercial — contact   │                                                    │
  │  /Users/caleb/Brain     │                                                    │
  ╰────────────────────────────────────────────────────────────────────────────╯

  ─────────────────────────────────────────────────────────────────────────────
  ❯ <user input>
  ─────────────────────────────────────────────────────────────────────────────
  [MODE]                                               New Release x.x.x

Key bindings:
  Shift+Tab      — cycle to next mode
  Enter          — submit
  Ctrl+C / Ctrl+D — quit
"""

from __future__ import annotations

import getpass
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from mindvault.modes import Mode, ModeConfig, get_config, next_mode
from mindvault.version import release_label

# ─── Style ────────────────────────────────────────────────────────────────────

TUI_STYLE = Style.from_dict({
    "bar":          "fg:#555555",
    "prompt":       "fg:#ff6a00 bold",          # Blaze orange — matches brand
    "mode-label":   "fg:#ff8a2a bold",           # Blaze2
    "mode-desc":    "fg:#888888 italic",
    "release":      "fg:#16a34a bold",           # green — new release
    "separator":    "fg:#333333",
    "think":        "fg:#6b7280 italic",         # council "thinking" lines
    # welcome box
    "box-border":   "fg:#444444",
    "box-title":    "fg:#ff6a00 bold",
    "box-name":     "fg:#ffffff bold",
    "box-logo":     "fg:#ff8a2a",
    "box-meta":     "fg:#888888",
    "box-license":  "fg:#6b7280 italic",
    "box-tip-head": "fg:#ff8a2a bold",
    "box-tip":      "fg:#aaaaaa",
    "box-session":  "fg:#888888",
    "box-divider":  "fg:#333333",
})


# ─── Bar rendering ─────────────────────────────────────────────────────────────

def _bar(width: int | None = None) -> str:
    w = width or shutil.get_terminal_size((100, 24)).columns
    return "─" * w


def _status_line(mode_config: ModeConfig, width: int | None = None) -> FormattedText:
    """Bottom status bar: [MODE] on left, release notice on right."""
    w = width or shutil.get_terminal_size((100, 24)).columns
    left = f" {mode_config.icon} {mode_config.label} "
    right_text = release_label()
    right = f" {right_text} " if right_text else ""
    gap = w - len(left) - len(right)
    middle = " " * max(gap, 0)
    parts: list[tuple[str, str]] = [
        ("class:mode-label", left),
        ("class:bar", middle),
    ]
    if right:
        parts.append(("class:release", right))
    return FormattedText(parts)


# ─── Output helpers ────────────────────────────────────────────────────────────

def print_bar(width: int | None = None) -> None:
    from prompt_toolkit import print_formatted_text
    print_formatted_text(
        FormattedText([("class:bar", _bar(width))]),
        style=TUI_STYLE,
        end="\n",
    )


def print_mode_switch(config: ModeConfig) -> None:
    from prompt_toolkit import print_formatted_text
    print_formatted_text(
        FormattedText([
            ("class:bar", "  "),
            ("class:mode-label", f"{config.icon}  Switched to {config.label}"),
            ("class:bar", " — "),
            ("class:mode-desc", config.description),
            ("class:bar", "\n"),
        ]),
        style=TUI_STYLE,
        end="",
    )


def print_thinking(member_name: str) -> None:
    """Print a 'council member is thinking' line in dim italic."""
    from prompt_toolkit import print_formatted_text
    print_formatted_text(
        FormattedText([("class:think", f"  ↳ {member_name} is reasoning...\n")]),
        style=TUI_STYLE,
        end="",
    )


def print_response(label: str, text: str) -> None:
    """Print the brain/council response with a coloured label."""
    from prompt_toolkit import print_formatted_text
    print_formatted_text(
        FormattedText([("class:prompt", f"\n{label}: ")]),
        style=TUI_STYLE,
        end="",
    )
    print(text)
    print()


def _get_username() -> str:
    """Best-effort: git config user.name → system user."""
    try:
        name = subprocess.check_output(
            ["git", "config", "user.name"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if name:
            return name
    except Exception:
        pass
    return getpass.getuser().capitalize()


def print_welcome(
    sessions: list[dict],
    model: str,
    embedding_model: str,
) -> None:
    """
    Render the Claude Code-style two-panel welcome box.

    Left panel  — logo, greeting, version, model, license, cwd
    Right panel — getting-started tips, recent sessions
    """
    from prompt_toolkit import print_formatted_text
    from mindvault.version import CURRENT_VERSION

    cols = shutil.get_terminal_size((120, 24)).columns
    cols = max(cols, 80)  # minimum sensible width

    # ── Panel dimensions ────────────────────────────────────────
    LEFT_W = 42   # content chars inside left cell (excluding border/padding)
    # Row format: │ {left:<LEFT_W} │ {right:<RIGHT_W} │
    # total cols = 1 + 1 + LEFT_W + 1 + 1 + 1 + RIGHT_W + 1 + 1 = LEFT_W + RIGHT_W + 7
    RIGHT_W = max(cols - LEFT_W - 7, 20)

    # ── Brain ASCII logo (5 lines, narrow chars only) ───────────
    LOGO = [
        "    ▗▄▄▄▄▖    ",
        "   ▟██████▙   ",
        "   ▜██████▛   ",
        "    ▀▄▄▄▄▀    ",
        "      ▐▌      ",
    ]

    # ── Left panel lines (content only, no padding/border) ──────
    name = _get_username()
    greeting = f"Welcome back, {name}!"
    left_lines: list[tuple[str, str]] = [
        ("", ""),
        ("box-name", greeting.center(LEFT_W)),
        ("", ""),
        *[("box-logo", line.center(LEFT_W)) for line in LOGO],
        ("", ""),
        ("box-meta",    f"  MindVault v{CURRENT_VERSION}"),
        ("box-meta",    f"  {model} · {embedding_model}"),
        ("", ""),
        ("box-license", "  Personal use — free"),
        ("box-license", "  Commercial use — contact me"),
        ("", ""),
        ("box-meta",    f"  {str(Path.cwd())}"),
        ("", ""),
    ]

    # ── Right panel lines ────────────────────────────────────────
    tip_sep = "─" * (RIGHT_W - 1)

    right_lines: list[tuple[str, str]] = [
        ("box-tip-head", " Getting started"),
        ("box-divider",  " " + tip_sep),
        ("box-tip", " ingest         index your data"),
        ("box-tip", " chat           start chatting"),
        ("box-tip", " Shift+Tab      cycle modes"),
        ("box-tip", " /mode [name]   switch reasoning mode"),
        ("", ""),
        ("box-tip-head", " Recent sessions"),
        ("box-divider",  " " + tip_sep),
    ]

    if sessions:
        for s in sessions[:5]:
            date = s.get("started_at", "")[:10]
            preview = (s.get("preview") or "")[:RIGHT_W - 16].strip()
            status = s.get("status", "raw")
            entry = f" {date}  {preview}"
            right_lines.append(("box-session", entry[:RIGHT_W - 1]))
    else:
        right_lines.append(("box-session", " No recent sessions"))

    # ── Pad to equal height ──────────────────────────────────────
    n_rows = max(len(left_lines), len(right_lines))
    while len(left_lines) < n_rows:
        left_lines.append(("", ""))
    while len(right_lines) < n_rows:
        right_lines.append(("", ""))

    # ── Render ────────────────────────────────────────────────────
    title = f"MindVault v{CURRENT_VERSION}"
    title_section = f"─── {title} "
    top_fill = "─" * (cols - 2 - len(title_section))
    top_border = "╭" + title_section + top_fill + "╮"
    bot_border = "╰" + "─" * (cols - 2) + "╯"

    ft = FormattedText

    # Top border
    print_formatted_text(ft([
        ("class:box-title", "╭─── "),
        ("class:box-title", title),
        ("class:box-border", " " + "─" * (cols - 2 - 5 - len(title)) + "╮"),
    ]), style=TUI_STYLE)

    # Content rows
    for (lclass, lcontent), (rclass, rcontent) in zip(left_lines, right_lines):
        # Truncate and pad content to exact widths
        ltext = (lcontent or "")[:LEFT_W].ljust(LEFT_W)
        rtext = (rcontent or "")[:RIGHT_W].ljust(RIGHT_W)
        lclass = lclass or "box-meta"
        rclass = rclass or "box-meta"
        print_formatted_text(ft([
            ("class:box-border", "│ "),
            (f"class:{lclass}", ltext),
            ("class:box-border", " │ "),
            (f"class:{rclass}", rtext),
            ("class:box-border", " │"),
        ]), style=TUI_STYLE)

    # Bottom border
    print_formatted_text(ft([
        ("class:box-border", bot_border),
    ]), style=TUI_STYLE)
    print()


# ─── Input session ─────────────────────────────────────────────────────────────

class BrainPrompt:
    """
    Interactive prompt session with mode cycling and Claude Code-style chrome.

    Usage:
        prompt = BrainPrompt(on_mode_change=my_callback)
        text = prompt.ask()   # returns user input or None on quit
    """

    def __init__(self, on_mode_change: Callable[[Mode], None] | None = None):
        self.current_mode: Mode = Mode.CHAT
        self._on_mode_change = on_mode_change
        self._session: PromptSession = self._build_session()

    def _build_session(self) -> PromptSession:
        kb = KeyBindings()

        @kb.add("s-tab")
        def _cycle_mode(event) -> None:
            self.current_mode = next_mode(self.current_mode)
            config = get_config(self.current_mode)
            # Clear current line and reprint
            event.app.current_buffer.reset()
            if self._on_mode_change:
                self._on_mode_change(self.current_mode)
            else:
                # Default: print mode switch inline
                print()
                print_mode_switch(config)

        return PromptSession(
            key_bindings=kb,
            style=TUI_STYLE,
            multiline=False,
        )

    def _bottom_toolbar(self) -> FormattedText:
        """Rendered as the prompt_toolkit bottom toolbar."""
        return _status_line(get_config(self.current_mode))

    def ask(self) -> str | None:
        """
        Show the two-bar prompt and wait for input.
        Returns the entered text, or None if the user quit (Ctrl+C / Ctrl+D).
        """
        bar = _bar()

        try:
            result = self._session.prompt(
                FormattedText([
                    ("class:bar", bar + "\n"),
                    ("class:prompt", "❯ "),
                ]),
                bottom_toolbar=self._bottom_toolbar,
                style=TUI_STYLE,
            )
        except (EOFError, KeyboardInterrupt):
            return None

        return result.strip() if result else ""

    @property
    def mode(self) -> Mode:
        return self.current_mode
