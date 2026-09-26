#!/usr/bin/env python3
"""Why does the smoothness-heavy policy lose a lane, and does it get worse?

Rolls the S-heavy branch at four snapshots on the reset that fails, recording
action rate, action norm, angular velocity, tilt, vertical velocity and
tracking error per step. Each is summarised over the lanes that fell and the
lanes that did not, so a physical signature of the failure can be separated
from the population average.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import RUNS, IsaacAudit, obs_tensor

SRC = "post_v2_t5_c25_actor_updating25-2026-09-23"
SNAPS = (0, 5, 10, 25)
SEED = 2840503
PREFERENCE = np.array([.1, .1, .1, .7], np.float32)
HORIZON = 64
METRIC_KEYS = ("action_rate", "action_norm", "ang_vel_xy", "tilt_deg",
               "abs_lin_vel_z", "tracking_abs_error")


def tilt(q):
    """Angle between the body's up axis and the world's, in degrees."""
    _, x, y, _ = [q[:, i] for i in range(4)]
    return torch.rad2deg(torch.acos((1 - 2 * (x * x + y * y)).clamp(-1, 1)))


class SmoothnessSafetyAudit(IsaacAudit):
    """Physical signature of the smoothness-heavy branch's lane failure."""

    run = "post_v2_t5_c27_smoothness_safety-2026-09-23"
    report = "audit.json"
    schema = "c27_smoothness_safety_v1"

    def snapshot(self, env, robot, od, ad, snap):
        from talon_rl.t4_actor_critic import T4SharedActorCritic

        n = self.num_envs
        m = T4SharedActorCritic(od, ad).cuda()
        m.load_state_dict(torch.load(RUNS / SRC / f"S_snap_{snap}.pt", map_location="cuda",
                                     weights_only=False)["model"])
        m.eval()
        w = torch.tensor(PREFERENCE, device="cuda").repeat(n, 1)
        cur, _ = env.reset(seed=SEED)
        cur = obs_tensor(cur).cuda()
        prev = torch.zeros((n, ad), device="cuda")
        first_done = [None] * n
        rows = []
        with torch.no_grad():
            for t in range(HORIZON):
                a = m.act_inference_with_preference(cur, w)
                nxt, _, te, tr, _ = env.step(a)
                done = (te | tr).cpu().numpy()
                data = robot.data
                cmd = env.unwrapped.command_manager.get_command("base_velocity")
                rows.append({
                    "t": t,
                    "action_rate": torch.linalg.vector_norm(a - prev, dim=-1).cpu().tolist(),
                    "action_norm": torch.linalg.vector_norm(a, dim=-1).cpu().tolist(),
                    "ang_vel_xy": torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1).cpu().tolist(),
                    "tilt_deg": tilt(data.root_quat_w).cpu().tolist(),
                    "abs_lin_vel_z": data.root_lin_vel_b[:, 2].abs().cpu().tolist(),
                    "tracking_abs_error": ((data.root_lin_vel_b[:, 0] - cmd[:, 0]).abs()
                                           + (data.root_ang_vel_b[:, 2] - cmd[:, 2]).abs()).cpu().tolist(),
                    "done": done.astype(int).tolist()})
                for i, d in enumerate(done):
                    if d and first_done[i] is None:
                        first_done[i] = t
                prev = a
                cur = obs_tensor(nxt).cuda()

        failed = [i for i, x in enumerate(first_done) if x is not None]
        survivors = [i for i in range(n) if i not in failed]
        metrics = {}
        for key in METRIC_KEYS:
            arr = np.array([r[key] for r in rows], float)
            metrics[key] = {"all_mean": float(arr.mean()),
                            "failed_env_mean": float(arr[:, failed].mean()) if failed else None,
                            "survivor_mean": float(arr[:, survivors].mean()) if survivors else None}
        return {"first_done_step": first_done, "failed_envs": failed,
                "survival": float(1 - len(failed) / n), "metrics": metrics, "rows": rows}

    def rollout(self, env, obs):
        robot = env.unwrapped.scene["robot"]
        od = obs.shape[-1]
        ad = env.unwrapped.action_manager.total_action_dim
        out = {"schema": self.schema, "seed": SEED, "snapshots": {}}
        for snap in SNAPS:
            out["snapshots"][str(snap)] = self.snapshot(env, robot, od, ad, snap)
        self.write(out)
        for s, x in out["snapshots"].items():
            print("\n", s, "survival", x["survival"], "failed", x["failed_envs"],
                  "done", x["first_done_step"])
            for k, v in x["metrics"].items():
                print(k, round(v["all_mean"], 4), v["failed_env_mean"], v["survivor_mean"])
        return out


if __name__ == "__main__":
    SmoothnessSafetyAudit.main()
