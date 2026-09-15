"""Run-directory management for train_prelim.py/play.py — mirrors
jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
logs/<exp>/<algo>/<timestamp>_<run_name>/ convention, simplified for this
repo's single-task/single-algorithm prelim scope: logs/talon_rl/<run_name>/
instead of a 3-level exp/algo/timestamp hierarchy (we don't have multiple
experiments/algorithms to disambiguate between yet — see
scripts/rl/core/algorithms/__init__.py's own docstring on why there's only
one algorithm here so far).
"""

from __future__ import annotations

import dataclasses
import os
from datetime import datetime

import yaml


def make_run_dir(logs_root: str, run_name: str | None = None) -> str:
    """Creates and returns logs_root/run_name (run_name defaults to a
    %Y-%m-%d_%H-%M-%S timestamp, matching the reference project's naming,
    just without its extra <run_name> suffix since we don't have a
    multi-experiment naming scheme yet)."""
    name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(logs_root, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def dump_config(run_dir: str, **cfgs) -> None:
    """Writes every passed dataclass instance to run_dir/config.yaml, one
    top-level key per kwarg name, e.g.
    dump_config(run_dir, obs=obs_cfg, reward=reward_cfg)."""
    data = {name: dataclasses.asdict(cfg) for name, cfg in cfgs.items()}
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def resolve_checkpoint(logs_root: str, load_run: str) -> str:
    """Returns the checkpoint.pt path inside logs_root/<run>, where <run>
    is either an exact run-directory name or "last" (the
    most-recently-modified subdirectory of logs_root) — the load_run="last"
    convenience the reference project's get_checkpoint_path() also offers,
    reimplemented here rather than ported since that helper lives in
    isaaclab_tasks.utils and is coupled to Isaac Lab's agent_cfg/hydra
    conventions this repo doesn't use."""
    if load_run == "last":
        candidates = [d for d in os.listdir(logs_root) if os.path.isdir(os.path.join(logs_root, d))]
        if not candidates:
            raise FileNotFoundError(f"no run directories found under {logs_root}")
        candidates.sort(key=lambda d: os.path.getmtime(os.path.join(logs_root, d)))
        run_name = candidates[-1]
    else:
        run_name = load_run
    path = os.path.join(logs_root, run_name, "checkpoint.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no checkpoint.pt found at {path}")
    return path
