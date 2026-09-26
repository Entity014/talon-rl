#!/usr/bin/env python3
"""V4-B1-Fix1 benchmark of one Isaac-Talon-A1-v0 config: startup, VRAM, env throughput and leg-length variant spread under replicate_physics on/off.

One process per (replicate_physics, num_envs) so VRAM readings start clean.
VRAM is whole-GPU nvidia-smi, sampled every 50 ms from a thread, because
PhysX memory is invisible to torch.cuda. The correctness gate checks the
physics, not the USD: PhysX FR thigh length must equal 0.2 * e_t[3] per env.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/

import numpy as np

from rl.core.diagnostics.isaac_audit import IsaacAudit

WARMUP = 20
STEPS = 200


def _gpu_used_total_mib() -> tuple[int, int]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.split(",")
    return int(q[0]), int(q[1])


class _PeakVram(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, _gpu_used_total_mib()[0])
            time.sleep(0.05)


class SpawnBenchmark(IsaacAudit):
    """V4-B1-Fix1 spawn benchmark for one config."""
    task = "Isaac-Talon-A1-v0"
    run = "teacher_v4_b1_fix1_spawn_benchmark-2026-09-26"

    def __init__(self, out, num_envs: int, replicate: bool):
        super().__init__(out)
        self.num_envs = num_envs
        self.replicate = replicate
        self.report = f"replicate{int(replicate)}_n{num_envs}.json"

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.scene.replicate_physics = self.replicate
        cfg.seed = self.seed
        self.cfg = cfg
        self.baseline_mib, self.total_mib = _gpu_used_total_mib()
        self.monitor = _PeakVram(); self.monitor.start()
        t0 = time.perf_counter()
        env = gym.make(self.task, cfg=cfg).unwrapped
        tr = env.reset()
        self.startup_s = time.perf_counter() - t0
        return env, tr

    def rollout(self, env, tr) -> dict:
        import torch
        from talon_rl.tasks.locomotion.a1_env import mdp

        zero = np.zeros((env.num_envs, env.action_dim), np.float32)
        for _ in range(WARMUP):
            env.step(zero)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            env.step(zero)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        steady_mib = _gpu_used_total_mib()[0]
        self.monitor.stop.set(); self.monitor.join()

        asset = env.scene["robot"]
        names = asset.body_names
        # Fresh read, not the extrinsic's cache, so a stale cache can't mask a mismatch.
        env.__dict__.pop("_leg_scale_cache", None)
        reported = mdp.leg_length_extrinsic(env).squeeze(-1)
        env.reset()  # default pose, so thigh length is joint geometry only
        pos = asset.data.body_pos_w
        thigh = (pos[:, names.index("FR_calf")] - pos[:, names.index("FR_thigh")]).norm(dim=-1)
        err = (thigh - 0.2 * reported).abs()
        dist = Counter(round(float(x), 4) for x in reported.cpu())

        out = {
            "num_envs": self.num_envs, "replicate_physics": self.replicate,
            "startup_s": round(self.startup_s, 2),
            "vram_mib": {"total": self.total_mib, "baseline_before_env": self.baseline_mib,
                         "peak": self.monitor.peak, "steady": steady_mib,
                         "free_at_peak": self.total_mib - self.monitor.peak},
            "steps_per_s": STEPS / dt, "env_samples_per_s": STEPS * self.num_envs / dt,
            "variant_distribution": {str(k): v for k, v in sorted(dist.items())},
            "unique_variants": len(dist),
            "physx_thigh_vs_reported_max_abs_err_m": float(err.max()),
            "e_t_leg_length_correct": bool(err.max() < 1e-3),
        }
        self.write(out)
        print(out, flush=True)
        return out


if __name__ == "__main__":
    a = SpawnBenchmark.parse_args(
        (("--num-envs",), {"type": int, "required": True}),
        (("--replicate",), {"type": int, "choices": (0, 1), "required": True}),
    )
    SpawnBenchmark(a.out, a.num_envs, bool(a.replicate)).execute()
