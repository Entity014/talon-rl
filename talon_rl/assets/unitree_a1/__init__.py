"""Package containing TALON's robot asset configurations.

Mirrors jaykorea/Isaac-RL-Two-wheel-Legged-Bot's assets/<robot>/__init__.py
pattern: EXT_DIR/DATA_DIR path constants derived from this file's location
plus extension.toml metadata, so per-robot config modules (a1.py) build
usd_path from a shared constant instead of each re-deriving it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

TALON_ASSETS_EXT_DIR = Path(__file__).resolve().parent.parent
"""Path to the assets extension source directory (talon_rl/assets/)."""

TALON_ASSETS_DATA_DIR = TALON_ASSETS_EXT_DIR / "data"
"""Path to the assets data directory (talon_rl/assets/data/)."""

with open(TALON_ASSETS_EXT_DIR / "config" / "extension.toml", "rb") as _f:
    TALON_ASSETS_METADATA = tomllib.load(_f)
"""Extension metadata dictionary parsed from config/extension.toml."""

__version__ = TALON_ASSETS_METADATA["package"]["version"]


##
# Configuration for different assets.
##

from .a1 import *  # noqa: F401,F403
