"""Package containing TALON's robot asset configurations.

Mirrors jaykorea/Isaac-RL-Two-wheel-Legged-Bot's assets/<robot>/__init__.py
pattern: per-robot config modules (a1.py) import TALON_ASSETS_DATA_DIR from
the shared talon_rl.assets package instead of each re-deriving it — see
talon_rl/assets/__init__.py.
"""

from __future__ import annotations

from .. import TALON_ASSETS_DATA_DIR  # noqa: F401 — re-exported for a1.py's `from . import TALON_ASSETS_DATA_DIR`

##
# Configuration for different assets.
##

from .a1 import *  # noqa: F401,F403
