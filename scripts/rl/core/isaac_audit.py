"""Rolling a policy in Isaac Lab and writing a report.

Every Isaac audit script boots the simulator, builds the flat Unitree A1 task
with the repo's own USD asset, rolls something over a set of preferences and
seeds, writes one JSON into a run directory, and has to close the env and the
app whether or not it succeeded. Only the rollout differs, so that is what a
subclass supplies.
"""

from __future__ import annotations

import sys
import traceback

import torch

from rl.core.run_report import ARTIFACTS, REPO, RUNS, RunReport

__all__ = ["IsaacAudit", "obs_tensor", "ARTIFACTS", "REPO", "RUNS", "A1_USD", "TASK"]

A1_USD = REPO / "talon_rl/assets/data/Robots/unitree_a1/a1.usd"
TASK = "Isaac-Velocity-Flat-Unitree-A1-v0"


def obs_tensor(x):
    """The policy group of an observation, as a tensor."""
    if isinstance(x, dict):
        x = x.get("policy", next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)


class IsaacAudit(RunReport):
    task: str = TASK
    num_envs: int = 8
    seed: int = 0              # the env's construction seed, not a rollout seed
    reset_seed: int | None = None   # None means reuse `seed` for the first reset
    set_cfg_seed: bool = True       # a couple of snapshots never set it
    set_usd_path: bool = True       # a few audits ran against Isaac's own asset

    # --- what a subclass fills in ---

    def rollout(self, env, obs) -> dict:
        """Roll the policy and return the report. `obs` is the first reset's."""
        raise NotImplementedError

    def configure(self, cfg) -> None:
        """Last chance to change the env config. A nominal snapshot uses this to
        switch off the randomisation that would otherwise perturb what it reads."""

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
        if self.set_cfg_seed:
            cfg.seed = self.seed
        if self.set_usd_path:
            cfg.scene.robot.spawn.usd_path = str(A1_USD)
        self.configure(cfg)
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=self.reset_seed if self.reset_seed is not None else self.seed)
        return env, obs_tensor(obs).cuda()

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
        cls(cls.parse_args().out).execute()
