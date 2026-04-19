"""Test bootstrap — install the import shim and discover tools exactly once."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import shim  # noqa: E402

shim.install_shim()

from tools.registry import discover_builtin_tools  # noqa: E402

discover_builtin_tools()
