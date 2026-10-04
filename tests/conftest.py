"""Keep local modules importable from pytest's console entry point."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
