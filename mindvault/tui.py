"""
mindvault/tui.py — Claude Code-style TUI for MindVault.

Layout:
  ────────────────────────── top bar ──────────────────────────
  ❯ <user input>
  ────────────────────────── bottom bar ───────────────────────
  [MODE]                               New Release x.x.x (if any)

Key bindings:
  Shift+Tab      — cycle to next mode
  Enter          — submit
  Ctrl+C / Ctrl+D — quit
"""

from __future__ import annotations

import shutil
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


def print_header(model: str, index_name: str, include_private: bool) -> None:
    from prompt_toolkit import print_formatted_text
    bar = _bar()
    print_formatted_text(
        FormattedText([("class:bar", bar)]),
        style=TUI_STYLE,
    )
    print_formatted_text(
        FormattedText([
            ("class:mode-label", "  MINDVAULT  "),
            ("class:mode-desc", f"model: {model}  |  index: {index_name}  |  "
             f"private: {'on' if include_private else 'off'}  |  shift+tab: cycle mode"),
        ]),
        style=TUI_STYLE,
    )
    print_formatted_text(
        FormattedText([("class:bar", bar)]),
        style=TUI_STYLE,
    )


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
