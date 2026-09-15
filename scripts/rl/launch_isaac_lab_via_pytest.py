"""Launches train_prelim.py's --env isaac_lab path through pytest instead of
`python scripts/rl/train_prelim.py` directly.

Found 2026-09-15: on this machine, `python scripts/rl/train_prelim.py
--env isaac_lab ...` deterministically died mid-scene-construction (Isaac
Sim's bare `SimulationApp({"headless": True})` shut itself down silently,
no traceback, right after the PhysX GPU-pipeline extension started) —
reproduced at num_envs=4 and 64, with/without torch imported before
SimulationApp, and even with a minimal ~20-line repro script. The *exact
same* env-construction code run through `pytest` (tests/test_a1_env.py,
and this file) succeeded reliably every time.

train_prelim.py now uses the official `isaaclab.app.AppLauncher` instead of
bare SimulationApp (matching jaykorea/Isaac-RL-Two-wheel-Legged-Bot's own
scripts/co_rl/train.py) — a plain-python repro with AppLauncher got past
the point bare SimulationApp died at (reached actual PhysX scene creation,
with a real informative exception instead of a silent one), which is
strong evidence AppLauncher was the real fix. Not independently confirmed
standalone yet — the machine's single 8GB GPU was occupied by a real
training run at the time, so the AppLauncher repro failed on VRAM
contention, not the original silent-death bug. Re-verify
`python scripts/rl/train_prelim.py --env isaac_lab ...` directly once the
GPU is free; if it holds, this pytest-workaround launcher is no longer
needed for new runs (kept for now since it's what's driving the run
started under it).

Run with train_prelim.py's own argparse flags passed through the
TALON_TRAIN_ARGS env var (pytest's own CLI parser would otherwise choke on
flags like --env that aren't its own):

    TALON_TRAIN_ARGS="--env isaac_lab --num_envs 1024 --updates 5000 \
        --logs_root logs/talon_rl --run_name full_a1_run --save_every 100" \
        pytest scripts/rl/launch_isaac_lab_via_pytest.py -v -s

Only picked up by an explicit `pytest scripts/rl/...` invocation, never by
`pytest tests/` (CLAUDE.md's documented command) or a bare `pytest` from
repo root scoped to tests/ — this file lives outside tests/ on purpose.
"""

import os
import shlex
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import pytest

pytest.importorskip("isaacsim")


def test_launch_train_prelim_isaac_lab():
    sys.argv = [sys.argv[0]] + shlex.split(os.environ.get("TALON_TRAIN_ARGS", ""))

    from rl.train_prelim import main

    main()
