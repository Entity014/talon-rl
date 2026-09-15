"""Package containing TienKung2 Lite's asset configuration.

Separate research application (bimanual box-carry) reusing this repo's
generic Multi-Objective RMA infrastructure, not part of this thesis's own
A1 locomotion scope — see
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md.
"""

from __future__ import annotations

from .. import TALON_ASSETS_DATA_DIR  # noqa: F401 — re-exported for tienkung.py's `from . import TALON_ASSETS_DATA_DIR`

from .tienkung import *  # noqa: F401,F403
