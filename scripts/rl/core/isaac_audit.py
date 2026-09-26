"""Rolling a policy in Isaac Lab and writing a report.

Every Isaac audit script boots the simulator, builds the flat Unitree A1 task
with the repo's own USD asset, rolls something over a set of preferences and
seeds, writes one JSON into a run directory, and has to close the env and the
app whether or not it succeeded. Only the rollout differs, so that is what a
subclass supplies.

`--out` writes the report somewhere else, which is what makes it possible to
re-run an audit and compare it against the recorded one: `runs/` is gitignored,
so overwriting a report loses it for good.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
# talon_rl is imported from source, not from site-packages, in the Isaac
# environment — the scripts used to do this for themselves
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUNS = REPO / "runs"
A1_USD = REPO / "talon_rl/assets/data/Robots/unitree_a1/a1.usd"
TASK = "Isaac-Velocity-Flat-Unitree-A1-v0"


def obs_tensor(x):
    """The policy group of an observation, as a tensor."""
    if isinstance(x, dict):
        x = x.get("policy", next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)


class IsaacAudit:
    run: str = ""              # run directory under runs/
    report: str = ""           # report filename written into it
    task: str = TASK
    num_envs: int = 8
    seed: int = 0              # the env's construction seed, not a rollout seed

    def __init__(self, out: str | Path | None = None):
        self.out = Path(out) if out else self.dir
        self.out.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return RUNS / self.run

    # --- what a subclass fills in ---

    def rollout(self, env, obs) -> dict:
        """Roll the policy and return the report. `obs` is the first reset's."""
        raise NotImplementedError

    # --- shared ---

    def build_env(self):
        """The flat A1 task, seeded and pointed at this repo's own USD asset."""
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  (registers the task ids)
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import (
            UnitreeA1FlatEnvCfg,
        )

        cfg = UnitreeA1FlatEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.seed = self.seed
        cfg.scene.robot.spawn.usd_path = str(A1_USD)
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=self.seed)
        return env, obs_tensor(obs).cuda()

    def write(self, report: dict) -> None:
        (self.out / self.report).write_text(json.dumps(report, indent=2) + "\n")

    def execute(self) -> dict:
        from isaaclab.app import AppLauncher

        # AppLauncher parses sys.argv, so hide our own flags from it
        argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = argv
        env = None
        try:
            env, obs = self.build_env()
            return self.rollout(env, obs)
        except BaseException:
            # app.close() below can take the process down before a traceback
            # reaches stdout, which looks like a silent success
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            raise
        finally:
            if env is not None:
                env.close()
            app.close()

    @classmethod
    def main(cls) -> None:
        doc = (cls.__doc__ or "").strip().splitlines()
        ap = argparse.ArgumentParser(description=doc[0] if doc else f"audit {cls.run}")
        ap.add_argument("--out", help="write the report here instead of the run directory")
        cls(ap.parse_args().out).execute()
