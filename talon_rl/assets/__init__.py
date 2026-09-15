"""Shared path/metadata constants for talon_rl's vendored robot assets.

Hoisted out of unitree_a1/__init__.py (2026-09-15) when a second robot
(tienkung2_lite) was added — the values were always robot-agnostic
(EXT_DIR/DATA_DIR resolve relative to this file's own location, not any
one robot's), only the code that computed them lived inside one robot's
own package. Every per-robot __init__.py re-exports these instead of
re-deriving them — see
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

TALON_ASSETS_EXT_DIR = Path(__file__).resolve().parent
"""Path to the assets extension source directory (talon_rl/assets/)."""

TALON_ASSETS_DATA_DIR = TALON_ASSETS_EXT_DIR / "data"
"""Path to the assets data directory (talon_rl/assets/data/)."""

with open(TALON_ASSETS_EXT_DIR / "config" / "extension.toml", "rb") as _f:
    TALON_ASSETS_METADATA = tomllib.load(_f)
"""Extension metadata dictionary parsed from config/extension.toml."""

__version__ = TALON_ASSETS_METADATA["package"]["version"]
