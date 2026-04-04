"""
mindvault.py — Legacy entry point. Prefer: python -m mindvault

Kept for backward compatibility. All CLI logic lives in mindvault/_cli.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mindvault._cli import main  # noqa: E402

if __name__ == "__main__":
    main()
