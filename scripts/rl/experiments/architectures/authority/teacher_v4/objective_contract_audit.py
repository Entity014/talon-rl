#!/usr/bin/env python3
"""V4-C0a objective-contract port check: the T/A/O/S reward terms on Isaac-Talon-A1-v0 match the stock A1 flat terms in config and kernel output.

Checks, in order: (1) per term, the Talon RewardsCfg weight and params equal
the stock UnitreeA1FlatEnvCfg ones, with the same function for A/O/S and the
ported v_command_buf kernel for T; (2) on live rollouts, every manager term
equals the stock kernel evaluated on the same state through a CommandManager
shim that returns v_command_buf, on lanes that did not reset this step;
(3) the V3 objective grouping builds a finite [N,4] vector from the Talon
manager. Random-action objective magnitudes are reported as a preview only:
divisors follow the T3-B protocol (M0 policy) and wait for the action contract.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/

import numpy as np
import torch

from rl.core.diagnostics.isaac_audit import IsaacAudit

TERMS = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "ang_vel_xy_l2", "flat_orientation_l2", "action_rate_l2")
PORTED = {"track_lin_vel_xy_exp", "track_ang_vel_z_exp"}
STEPS = 64
TOL = 1e-6


class ObjectiveContractAudit(IsaacAudit):
    """V4-C0a objective-contract port check."""
    task = "Isaac-Talon-A1-v0"
    run = "teacher_v4_c0a_objective_contract-2026-09-26"
    report = "report.json"
    num_envs = 64

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.seed = self.seed
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg).unwrapped
        return env, env.reset()

    def rollout(self, env, tr) -> dict:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.rewards.objectives import OBJECTIVE_ORDER, OBJECTIVE_TERMS, normalized_objective_vector, raw_objective_vector

        stock = UnitreeA1FlatEnvCfg().rewards
        talon = self.cfg.rewards
        checks, cfg_rows = {}, {}
        for n in TERMS:
            s, t = getattr(stock, n), getattr(talon, n)
            sp = {k: v for k, v in s.params.items() if k != "command_name"}
            same_func = (t.func is s.func) if n not in PORTED else t.func.__module__.endswith("a1_env.mdp.rewards")
            cfg_rows[n] = {"stock_weight": s.weight, "talon_weight": t.weight, "stock_params": {k: str(v) for k, v in sp.items()},
                           "talon_params": {k: str(v) for k, v in t.params.items()}, "func": f"{t.func.__module__}.{t.func.__name__}"}
            checks[f"cfg_{n}"] = bool(s.weight == t.weight and {k: str(v) for k, v in sp.items()} == {k: str(v) for k, v in t.params.items()} and same_func)
        # objectives.py grouping must use exactly these terms
        checks["objective_terms_match"] = sorted(x for g in OBJECTIVE_TERMS.values() for x in g) == sorted(TERMS)

        mgr = env.reward_manager
        names = list(mgr.active_terms)
        checks["manager_has_only_objective_terms"] = sorted(names) == sorted(TERMS)
        shim = SimpleNamespace(scene=env.scene, action_manager=env.action_manager,
                               command_manager=SimpleNamespace(get_command=lambda _name: env.v_command_buf))
        stock_funcs = {n: getattr(stock, n) for n in TERMS}

        gen = np.random.default_rng(self.seed)
        max_err = {n: 0.0 for n in TERMS}
        compared = 0
        raw_rows, vec_rows = [], []
        for _ in range(STEPS):
            a = gen.uniform(-1, 1, (env.num_envs, env.action_dim)).astype(np.float32)
            tr, done = env.step(a)
            raw = mgr._step_reward.detach()
            keep = torch.as_tensor(~done, device=raw.device)
            if keep.any():
                compared += int(keep.sum())
                for i, n in enumerate(names):
                    sc = stock_funcs[n]
                    ref = sc.func(shim, **sc.params) * sc.weight  # the shim ignores command_name
                    max_err[n] = max(max_err[n], float((raw[keep, i] - ref[keep]).abs().max()))
            raw_np = raw.cpu().numpy()
            terms = {n: raw_np[:, i] for i, n in enumerate(names)}
            raw_rows.append(raw_objective_vector(terms, shape=(env.num_envs,)))
            vec_rows.append(normalized_objective_vector(terms, shape=(env.num_envs,)))
        for n in TERMS:
            checks[f"kernel_{n}"] = max_err[n] < TOL
        vec = np.concatenate(vec_rows)
        rawv = np.concatenate(raw_rows)
        checks["objective_vector_shape_finite"] = bool(vec.shape[1] == 4 and np.isfinite(vec).all())

        out = {
            "config": cfg_rows,
            "kernel_max_abs_err": max_err, "kernel_samples_compared": compared,
            "manager_term_order": names,
            "preview_random_action_abs_mean_raw": dict(zip(OBJECTIVE_ORDER, np.abs(rawv).mean(0).tolist())),
            "preview_note": "uniform[-1,1] actions on the current Talon action contract; NOT divisors",
            "checks": checks, "pass": all(checks.values()),
        }
        self.write(out)
        print(checks, "PASS" if out["pass"] else "FAIL", flush=True)
        if not out["pass"]:
            raise SystemExit(1)
        return out


if __name__ == "__main__":
    ObjectiveContractAudit.main()
