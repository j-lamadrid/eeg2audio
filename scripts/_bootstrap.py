"""Make the local src package importable when scripts are run directly."""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root

