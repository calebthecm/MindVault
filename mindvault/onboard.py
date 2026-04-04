"""
onboard.py — First-run setup for MindVault.

Guides you through:
  - Python version check
  - Dependency installation
  - Model backend selection (Ollama / OpenAI-compatible / custom)
  - Ollama model detection and pull suggestions
  - Directory structure creation
  - .gitignore generation
  - config.py patching with your choices

Run once before anything else:
    python onboard.py
"""

import re
import subprocess
import sys
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent   # Brain root
CONFIG_PATH = ROOT / "mindvault" / "config.py"
GITIGNORE_PATH = ROOT / ".gitignore"

REQUIRED_DIRS = [
    ROOT / "My Brain",
    ROOT / "Private Brain",
]

REQUIRED_PACKAGES = ["qdrant-client", "httpx"]

RECOMMENDED_EMBED_MODEL = "nomic-embed-text"
RECOMMENDED_LLM_MODEL = "llama3.2"

SEP = "─" * 60


# ─── Helpers ──────────────────────────────────────────────────────────────────


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str) -> None:
    print(f"  [ok]  {msg}")


def warn(msg: str) -> None:
    print(f"  [!!]  {msg}")


def info(msg: str) -> None:
    print(f"        {msg}")


def ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        answer = input(f"\n  > {prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    return answer if answer else default


def ask_yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    answer = ask(f"{prompt} ({hint})", "y" if default else "n").lower()
    return answer in ("y", "yes")


# ─── Step 1: Python version ───────────────────────────────────────────────────


def check_python() -> None:
    header("Step 1 — Python version")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}"
    if (major, minor) < (3, 11):
        warn(f"Python {version_str} detected. MindVault requires 3.11 or later.")
        info("Install a newer Python: https://www.python.org/downloads/")
        sys.exit(1)
    ok(f"Python {version_str}")


# ─── Step 2: Dependencies ─────────────────────────────────────────────────────


def check_dependencies() -> None:
    header("Step 3 — Python packages")

    missing = []
    for pkg in REQUIRED_PACKAGES:
        import importlib

        module_name = pkg.replace("-", "_")
        try:
            importlib.import_module(module_name)
            ok(f"{pkg}")
        except ImportError:
            warn(f"{pkg} not found")
            missing.append(pkg)

    if missing:
        if ask_yes(f"Install {len(missing)} missing package(s)?"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + missing,
                stdout=subprocess.DEVNULL,
            )
            ok("Packages installed")
        else:
            warn("Skipped — some features may not work.")


# ─── Step 3: Model backend ────────────────────────────────────────────────────


def _check_ollama(base_url: str) -> tuple[bool, list[str]]:
    """Returns (is_running, list_of_model_names)."""
    try:
        import httpx

        resp = httpx.get(f"{base_url}/api/tags", timeout=4.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, models
    except Exception:
        return False, []


def _ollama_pull_hint(model: str) -> None:
    info(f"  Pull it with:  ollama pull {model}")


def configure_ollama(base_url: str) -> tuple[str, str, str]:
    """
    Walk the user through Ollama model selection.
    Returns (ollama_base_url, llm_model, embedding_model).
    """
    running, available = _check_ollama(base_url)

    if not running:
        warn(f"Ollama not reachable at {base_url}")
        info("To install Ollama:  https://ollama.com/download")
        info("To start Ollama:    ollama serve")
        info("Then re-run this script, or skip and edit config.py manually.")
        print()
        if not ask_yes("Continue without Ollama? (you can configure it later)"):
            sys.exit(0)
        llm = ask("LLM model name to use later", RECOMMENDED_LLM_MODEL)
        embed = ask("Embedding model name to use later", RECOMMENDED_EMBED_MODEL)
        return base_url, llm, embed

    ok(f"Ollama is running at {base_url}")

    if available:
        info("Available models:")
        for m in available:
            info(f"  - {m}")
    else:
        info("No models pulled yet.")

    # Check for recommended models
    needs_pull = []
    for model in [RECOMMENDED_LLM_MODEL, RECOMMENDED_EMBED_MODEL]:
        # Match either exact name or name:tag
        if not any(m == model or m.startswith(model + ":") for m in available):
            warn(f"Recommended model '{model}' not found")
            needs_pull.append(model)

    if needs_pull:
        if ask_yes(f"Pull {len(needs_pull)} recommended model(s) now? (takes a few minutes)"):
            for model in needs_pull:
                print(f"\n  Pulling {model}...")
                result = subprocess.run(["ollama", "pull", model])
                if result.returncode == 0:
                    ok(f"Pulled {model}")
                else:
                    warn(f"Pull failed for {model} — pull it manually: ollama pull {model}")
        else:
            for model in needs_pull:
                _ollama_pull_hint(model)

    llm = ask("LLM model", RECOMMENDED_LLM_MODEL)
    embed = ask("Embedding model", RECOMMENDED_EMBED_MODEL)
    return base_url, llm, embed


def select_model_backend() -> tuple[str, str, str, str]:
    """
    Let user choose a backend.
    Returns (backend_type, ollama_base_url, llm_model, embedding_model).
    """
    header("Step 4 — Model backend")
    info("Choose how MindVault will run AI models:\n")
    info("  1) Ollama  (local, recommended — no data leaves your machine)")
    info("  2) OpenAI-compatible API  (e.g. OpenAI, Groq, LM Studio, vLLM)")
    info("  3) Custom endpoint")

    choice = ask("Your choice", "1")

    if choice == "1":
        base_url = ask("Ollama URL", "http://localhost:11434")
        ollama_base, llm, embed = configure_ollama(base_url)
        return "ollama", ollama_base, llm, embed

    elif choice == "2":
        base_url = ask("API base URL (OpenAI-compatible)", "https://api.openai.com/v1")
        llm = ask("LLM model name", "gpt-4o-mini")
        embed = ask("Embedding model name", "text-embedding-3-small")
        warn("OpenAI-compatible mode: set your API key in .env as LLM_API_KEY=sk-...")
        _write_env_template({"LLM_API_KEY": ""})
        return "openai", base_url, llm, embed

    else:
        base_url = ask("Custom endpoint base URL", "http://localhost:11434")
        llm = ask("LLM model name", RECOMMENDED_LLM_MODEL)
        embed = ask("Embedding model name", RECOMMENDED_EMBED_MODEL)
        return "custom", base_url, llm, embed


def _write_env_template(keys: dict[str, str]) -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        ok(".env already exists — not overwriting")
        return
    lines = [f"{k}={v}\n" for k, v in keys.items()]
    env_path.write_text("".join(lines))
    ok(f"Created {env_path.name} — fill in your values")


# ─── Step 4: Patch config.py ──────────────────────────────────────────────────


def patch_config(backend: str, ollama_base: str, llm_model: str, embedding_model: str) -> None:
    """Update LLM_BACKEND, OLLAMA_BASE, LLM_MODEL, EMBEDDING_MODEL in config.py in-place."""
    header("Step 5 — Writing config")

    if not CONFIG_PATH.exists():
        warn("config.py not found — skipping config patch")
        info("Set LLM_BACKEND, OLLAMA_BASE, LLM_MODEL, and EMBEDDING_MODEL manually.")
        return

    text = CONFIG_PATH.read_text()

    # Map backend choice to config value
    backend_value = "ollama" if backend == "ollama" else "openai"

    replacements = {
        "LLM_BACKEND": f'LLM_BACKEND = "{backend_value}"',
        "OLLAMA_BASE": f'OLLAMA_BASE = "{ollama_base}"',
        "LLM_MODEL": f'LLM_MODEL = "{llm_model}"',
        "EMBEDDING_MODEL": f'EMBEDDING_MODEL = "{embedding_model}"',
    }

    for var, new_line in replacements.items():
        pattern = rf'^{var}\s*=\s*.*$'
        text, count = re.subn(pattern, new_line, text, flags=re.MULTILINE)
        if count:
            ok(f"{var} = {new_line.split('=', 1)[1].strip()}")
        else:
            warn(f"Could not find {var} in config.py — add it manually")

    CONFIG_PATH.write_text(text)


# ─── Step 5: Directories ──────────────────────────────────────────────────────


def create_directories() -> None:
    header("Step 6 — Directories")
    for d in REQUIRED_DIRS:
        if d.exists():
            ok(f"{d.name}/  (exists)")
        else:
            d.mkdir(parents=True)
            ok(f"{d.name}/  (created)")

    # mindvault/__init__.py so it's a proper package
    init_file = ROOT / "mindvault" / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""MindVault — local-first second brain system."""\n')


# ─── Step 6: .gitignore ───────────────────────────────────────────────────────


GITIGNORE_CONTENT = """\
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/

# MindVault storage (never commit your personal data)
brain.db
.qdrant/
*.db
*.db-shm
*.db-wal

# Obsidian vaults (personal notes — keep local)
My Brain/
Private Brain/

# Export data (may contain sensitive conversation history)
data-*/

# Environment secrets
.env
.env.*

# macOS
.DS_Store

# Editor
.idea/
.vscode/
*.swp
"""


def create_gitignore() -> None:
    header("Step 7 — .gitignore")
    if GITIGNORE_PATH.exists():
        ok(".gitignore already exists — not overwriting")
        return
    GITIGNORE_PATH.write_text(GITIGNORE_CONTENT)
    ok(".gitignore created")


# ─── Step 7: First-run guidance ───────────────────────────────────────────────


def print_next_steps() -> None:
    header("Setup complete — next steps")
    print("""
  1. Add content to Brain/
       Drop any folder containing your data — it can be:
         - Anthropic export  (claude.ai → Settings → Export Data)
         - OpenAI export     (chat.openai.com → Settings → Data Controls → Export)
         - PDF files, text files, or any JSON export

  2. Index your data:
       python mindvault.py ingest

  3. Start chatting with your brain:
       python mindvault.py chat

  Useful flags:
       python mindvault.py ingest --no-llm   # faster, skips LLM calls
       python mindvault.py stats             # show what's indexed
       python mindvault.py ingest --force    # re-index everything
""")
    print(SEP)
    print("  MindVault is ready. Your brain is local, private, yours.")
    print(f"{SEP}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────


def collect_user_info() -> None:
    """Ask for the user's name and save it to user config."""
    from mindvault import user_config

    header("Step 2 — About you")
    current = user_config.get_name()
    prompt = "Your first name (shown in the welcome screen)"
    name = ask(prompt, default=current or "")
    if name:
        user_config.set_name(name)
        ok(f"Name saved: {name}")
    else:
        warn("No name entered — you can set it later with: python mindvault.py setup")


def main() -> None:
    print(f"\n{SEP}")
    print("  MINDVAULT — First Run Setup")
    print(SEP)

    check_python()
    collect_user_info()
    check_dependencies()

    backend, ollama_base, llm_model, embedding_model = select_model_backend()

    patch_config(backend, ollama_base, llm_model, embedding_model)
    create_directories()
    create_gitignore()

    from mindvault import user_config
    user_config.mark_setup_complete()

    print_next_steps()


if __name__ == "__main__":
    main()
