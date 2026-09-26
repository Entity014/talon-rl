#!/usr/bin/env python3
"""V4-C env gate: Isaac-Talon-A1-V4C-v0 equals stock Isaac-Velocity-Flat-Unitree-A1-v0 except the declared additions, and its e_t is 12-D.

Diffs the full built configs (to_dict) and fails on any difference outside
the allow-list: the added privileged observation group, and the robot USD
path (the repo's a1.usd, which every M0/V3 script also set). Then checks the
live env: physics timing, action scale and actuator gains (the action
contract), e_t shape, and which e_t channels vary (only the trunk mass, from
stock add_base_mass, should).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/

import torch

from rl.core.diagnostics.isaac_audit import IsaacAudit

ALLOWED = ("observations.privileged", "scene.robot.spawn.usd_path")
E_T_NAMES = ("friction", "kp", "kd", "leg_length", "joint_range", "terrain_height", "dynamic_friction",
             "joint_damping", "payload_mass", "payload_com_x", "payload_com_y", "payload_com_z")
EXPECTED_VARYING = {"payload_mass"}
STEPS = 8


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                out.append((p, a.get(k, "<absent>"), b.get(k, "<absent>")))
            else:
                out += _diff(a[k], b[k], p)
        return out
    return [] if a == b else [(path, a, b)]


class V4CEnvParity(IsaacAudit):
    """V4-C env parity gate."""
    task = "Isaac-Talon-A1-V4C-v0"
    run = "teacher_v4_c0a2_v4c_env_parity-2026-09-26"
    report = "report.json"
    num_envs = 64

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg

        cfg = TalonV4CEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.seed = self.seed
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=self.seed)
        return env, obs

    def rollout(self, env, obs) -> dict:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg

        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg

        # Fresh, unbuilt cfgs on both sides: building the scene rewrites prim
        # paths and terrain fields in place, which would show up as diffs.
        stock, v4c = UnitreeA1FlatEnvCfg(), TalonV4CEnvCfg()
        for c in (stock, v4c):
            c.scene.num_envs, c.seed = self.num_envs, self.seed
        diffs = _diff(stock.to_dict(), v4c.to_dict())
        unexpected = [d for d in diffs if not d[0].startswith(ALLOWED)]
        checks = {"config_matches_stock_except_allowed": not unexpected}

        u = env.unwrapped
        term = u.action_manager.get_term("joint_pos")
        act = next(iter(u.scene["robot"].actuators.values()))
        timing = {"sim_dt": u.cfg.sim.dt, "decimation": u.cfg.decimation, "step_dt": u.step_dt,
                  "max_episode_length": int(u.max_episode_length)}
        action = {"scale": float(term._scale) if not torch.is_tensor(term._scale) else float(term._scale.unique()[0]),
                  "kp": act.stiffness.unique().tolist(), "kd": act.damping.unique().tolist()}
        checks["timing_stock"] = timing == {"sim_dt": 0.005, "decimation": 4, "step_dt": 0.02, "max_episode_length": 1000}
        checks["action_contract_canonical"] = action["scale"] == 0.25 and action["kp"] == [25.0] and action["kd"] == [0.5]

        ext = []
        policy_shape = tuple(obs["policy"].shape)
        for _ in range(STEPS):
            o, *_ = env.step(torch.zeros(self.num_envs, u.action_manager.total_action_dim, device=u.device))
            ext.append(o["privileged"])
        e = torch.cat(ext)
        std = e.std(0)
        varying = {n for n, s in zip(E_T_NAMES, std.tolist()) if s > 1e-6}
        checks["policy_obs_48d"] = policy_shape == (self.num_envs, 48)
        checks["e_t_12d"] = e.shape[1] == 12
        checks["e_t_finite"] = bool(torch.isfinite(e).all())
        checks["e_t_varying_channels_expected"] = varying == EXPECTED_VARYING

        out = {
            "config_diffs": [{"path": p, "stock": str(a), "v4c": str(b)} for p, a, b in diffs],
            "unexpected_config_diffs": [{"path": p, "stock": str(a), "v4c": str(b)} for p, a, b in unexpected],
            "timing": timing, "action": action,
            "e_t_mean": dict(zip(E_T_NAMES, e.mean(0).tolist())), "e_t_std": dict(zip(E_T_NAMES, std.tolist())),
            "e_t_varying": sorted(varying),
            "reward_terms": list(u.reward_manager.active_terms),
            "checks": checks, "pass": all(checks.values()),
        }
        self.write(out)
        print(checks, "PASS" if out["pass"] else "FAIL", flush=True)
        if not out["pass"]:
            raise SystemExit(1)
        return out


if __name__ == "__main__":
    V4CEnvParity.main()
