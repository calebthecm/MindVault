"""
mindvault/__main__.py — Entry point for `python -m mindvault`.

Enables:
    python -m mindvault          # chat (default)
    python -m mindvault chat
    python -m mindvault ingest
    python -m mindvault setup
    ... (all commands)

After `pip install -e .` you can also run just:
    mindvault
    mindvault chat
    mindvault ingest
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running in development
# (so src.* imports resolve). When installed via pip this is handled
# by the package layout.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import and run the shared CLI entrypoint
from mindvault._cli import main  # noqa: E402

if __name__ == "__main__":
    main()
