"""
Backward-compatible shim for legacy imports.

Utilities were migrated to `common.scripts.utils`.
This module re-exports the public symbols to preserve legacy imports.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.scripts.utils import compute_gl_coefficients

__all__ = ["compute_gl_coefficients"]
