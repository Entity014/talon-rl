"""Package containing TALON's robot asset configurations.

Mirrors jaykorea/Isaac-RL-Two-wheel-Legged-Bot's assets/<robot>/__init__.py
pattern: per-robot config modules (a1.py) import TALON_ASSETS_DATA_DIR from
the shared talon_rl.assets package instead of each re-deriving it — see
talon_rl/assets/__init__.py.
"""

from __future__ import annotations

from .. import (  # noqa: F401 — re-exported for a1.py's `from . import TALON_ASSETS_DATA_DIR`, and to keep unitree_a1.* resolving all four names as before this refactor
    TALON_ASSETS_DATA_DIR,
    TALON_ASSETS_EXT_DIR,
    TALON_ASSETS_METADATA,
    __version__,
)

##
# Configuration for different assets.
##

from .a1 import *  # noqa: F401,F403
