"""
mindvault/tui.py — Claude Code-style TUI for MindVault.

Layout:
  ╭─── MindVault v0.5.0 ────────────────────────────────────────────────────────╮
  │                         │ Getting started                                    │
  │   Welcome back, Caleb!  │ ingest    index your data                         │
  │                         │ chat      start chatting                           │
  │      (brain logo)       │ Shift+Tab cycle modes                              │
  │                         │ ─────────────────────────────────────────────────  │
  │  MindVault v0.5.0       │ Recent sessions                                    │
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
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import run_in_terminal
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


def print_markdown_response(label: str, text: str) -> None:
    """Render a Brain response as markdown using Rich."""
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console(highlight=False)
    console.print(f"\n[bold #ff6a00]{label}:[/bold #ff6a00]")
    console.print(Markdown(text))
    console.print()


# ─── Baking animation ──────────────────────────────────────────────────────────

_BAKING_WORDS = [
    "Synthesizing", "Brewing", "Cooking", "Thinking", "Distilling",
    "Manifesting", "Forging", "Conjuring", "Weaving", "Crystallizing",
    "Decoding", "Channeling", "Processing", "Summoning", "Baking",
    "Fusing", "Crafting", "Compiling", "Rendering", "Calculating",
]

# Cycle through orange → red → yellow brand palette
_BAKING_COLORS = ["#ff6a00", "#dc2626", "#ff8a2a", "#fbbf24", "#b45309", "#ff6a00"]


class BakingAnimation:
    """
    Cycles animated one-word status labels with ✻ in orange/red/yellow
    while the model is generating a response.

    Usage:
        anim = BakingAnimation()
        anim.start()
        ...do LLM call...
        elapsed = anim.stop()   # clears line, returns seconds elapsed
        print(f"✻ Baked for {elapsed:.1f}s")
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop animation, clear the line, return elapsed seconds."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        elapsed = time.monotonic() - self._start_time
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
        return elapsed

    def _run(self) -> None:
        from rich.console import Console
        console = Console(highlight=False)
        i = 0
        while not self._stop.is_set():
            word = _BAKING_WORDS[i % len(_BAKING_WORDS)]
            color = _BAKING_COLORS[i % len(_BAKING_COLORS)]
            console.print(f"[{color}]✻ {word}...[/{color}]", end="\r")
            time.sleep(0.28)
            i += 1


def _get_username() -> str:
    """
    Resolution order:
      1. .mindvault.json user_name  (set during onboarding or first-chat prompt)
      2. git config user.name
      3. system login name (capitalized)
    """
    try:
        from mindvault.user_config import get_name
        stored = get_name()
        if stored:
            return stored
    except Exception:
        pass
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
    work_dir: str | None = None,
) -> None:
    """
    Render the Claude Code-style two-panel welcome box.

    Everything is derived from arguments or runtime state — nothing hardcoded.

    Left panel  — logo, greeting, version, model, license notice, working dir
    Right panel — getting-started tips, recent sessions with dates + previews
    """
    from prompt_toolkit import print_formatted_text
    from mindvault.version import CURRENT_VERSION, release_label

    # ── Terminal geometry ────────────────────────────────────────
    # Cap at 160 — beyond that the two-panel layout looks too sparse
    MAX_BOX_COLS = 160
    cols = min(max(shutil.get_terminal_size((120, 24)).columns, 80), MAX_BOX_COLS)

    # Row format: │ {left:<LEFT_W} │ {right:<RIGHT_W} │
    # Border chars per row: "│ " + " │ " + " │" = 7
    LEFT_W = 42
    RIGHT_W = max(cols - LEFT_W - 7, 20)

    # ── Dynamic values ───────────────────────────────────────────
    name = _get_username()
    cwd = work_dir or str(Path.cwd())
    version_label = f"MindVault v{CURRENT_VERSION}"
    update = release_label()  # "New Release x.x.x" or None

    # ── Brain ASCII logo ─────────────────────────────────────────
    LOGO = [
        "    ▗▄▄▄▄▖    ",
        "   ▟██████▙   ",
        "   ▜██████▛   ",
        "    ▀▄▄▄▄▀    ",
        "      ▐▌      ",
    ]

    # ── Left panel ───────────────────────────────────────────────
    greeting = f"Welcome back, {name}!" if name else "Welcome to MindVault!"
    left_lines: list[tuple[str, str]] = [
        ("", ""),
        ("box-name",    greeting.center(LEFT_W)),
        ("", ""),
        *[("box-logo", line.center(LEFT_W)) for line in LOGO],
        ("", ""),
        ("box-meta",    f"  {version_label}"),
        ("box-meta",    f"  {model} · {embedding_model}"),
        ("", ""),
        ("box-license", "  Personal use — free"),
        ("box-license", "  Commercial use — contact me"),
        ("", ""),
        ("box-meta",    f"  {cwd}"),
    ]
    if update:
        left_lines.append(("release", f"  ★ {update}"))
    left_lines.append(("", ""))

    # ── Right panel ──────────────────────────────────────────────
    divider = " " + "─" * (RIGHT_W - 1)

    right_lines: list[tuple[str, str]] = [
        ("box-tip-head", " Getting started"),
        ("box-divider",  divider),
        ("box-tip", " ingest         index your data"),
        ("box-tip", " chat           start chatting"),
        ("box-tip", " Shift+Tab      cycle modes"),
        ("box-tip", " /mode [name]   switch reasoning mode"),
        ("", ""),
        ("box-tip-head", " Recent sessions"),
        ("box-divider",  divider),
    ]

    if sessions:
        for s in sessions[:5]:
            date = (s.get("started_at") or "")[:10]
            raw_preview = (s.get("preview") or "").strip()
            # Truncate preview to fit right panel (leave space for date + spacing)
            max_preview = RIGHT_W - len(date) - 4
            preview = raw_preview[:max(max_preview, 10)]
            right_lines.append(("box-session", f" {date}  {preview}"))
    else:
        right_lines.append(("box-session", " No recent sessions"))

    # ── Equalize panel heights ───────────────────────────────────
    n_rows = max(len(left_lines), len(right_lines))
    while len(left_lines) < n_rows:
        left_lines.append(("", ""))
    while len(right_lines) < n_rows:
        right_lines.append(("", ""))

    # ── Render ───────────────────────────────────────────────────
    ft = FormattedText

    # Top border: ╭─── MindVault vX.X.X ─...─╮
    # fixed: ╭(1) ─(1) ─(1) ─(1) space(1) title space(1) fill ╮(1) = 7 + len + fill
    title_fill = max(0, cols - 7 - len(version_label))
    print_formatted_text(ft([
        ("class:box-title",  "╭─── "),
        ("class:box-title",  version_label),
        ("class:box-border", " " + "─" * title_fill + "╮"),
    ]), style=TUI_STYLE)

    # Content rows
    for (lclass, lcontent), (rclass, rcontent) in zip(left_lines, right_lines):
        ltext = (lcontent or "")[:LEFT_W].ljust(LEFT_W)
        rtext = (rcontent or "")[:RIGHT_W].ljust(RIGHT_W)
        print_formatted_text(ft([
            ("class:box-border",          "│ "),
            (f"class:{lclass or 'box-meta'}", ltext),
            ("class:box-border",          " │ "),
            (f"class:{rclass or 'box-meta'}", rtext),
            ("class:box-border",          " │"),
        ]), style=TUI_STYLE)

    # Bottom border
    print_formatted_text(ft([
        ("class:box-border", "╰" + "─" * (cols - 2) + "╯"),
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

    # Seconds within which a second Ctrl+C press confirms exit
    _CTRLC_WINDOW = 2.0

    def __init__(self, on_mode_change: Callable[[Mode], None] | None = None):
        self.current_mode: Mode = Mode.CHAT
        self._on_mode_change = on_mode_change
        self._last_ctrlc: float = 0.0   # monotonic timestamp of last Ctrl+C
        self._session: PromptSession = self._build_session()

    def _build_session(self) -> PromptSession:
        kb = KeyBindings()

        @kb.add("s-tab")
        def _cycle_mode(event) -> None:
            self.current_mode = next_mode(self.current_mode)
            config = get_config(self.current_mode)
            event.app.current_buffer.reset()
            if self._on_mode_change:
                self._on_mode_change(self.current_mode)
            else:
                print()
                print_mode_switch(config)

        @kb.add("c-c")
        def _handle_ctrlc(event) -> None:
            """
            First press: show warning and stay in prompt.
            Second press within _CTRLC_WINDOW seconds: exit.
            """
            from prompt_toolkit import print_formatted_text as pft
            now = time.monotonic()
            if now - self._last_ctrlc < self._CTRLC_WINDOW:
                # Confirmed — exit
                event.app.exit(exception=KeyboardInterrupt())
            else:
                self._last_ctrlc = now
                event.app.current_buffer.reset()

                def _warn() -> None:
                    pft(
                        FormattedText([
                            ("class:mode-desc", "\n  (Press Ctrl+C again to exit)\n"),
                        ]),
                        style=TUI_STYLE,
                    )

                run_in_terminal(_warn)

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
        Returns the entered text, or None if the user confirmed quit.
        First Ctrl+C shows a warning; second Ctrl+C within 2s exits.
        Ctrl+D (EOF) exits immediately.
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
        except KeyboardInterrupt:
            # Raised only on confirmed double-tap
            return None
        except EOFError:
            # Ctrl+D — exit immediately (intentional, no double-tap needed)
            return None

        return result.strip() if result else ""

    @property
    def mode(self) -> Mode:
        return self.current_mode
