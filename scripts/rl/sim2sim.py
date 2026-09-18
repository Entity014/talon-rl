#!/usr/bin/env python3
"""Sim2Sim entry point — runs a TorchScript-exported policy (see play.py's
--export) against the vendored A1 MuJoCo model.

    python scripts/rl/sim2sim.py --policy policy.pt --steps 200

00_Proposal §3.4's Sim-to-Sim Validation is [CORE] scope for this thesis;
this script is the mechanism only (no Isaac Sim comparison yet — see
core/sim2sim.py's module docstring for exactly what's unverified and why).
Runs with no Isaac Sim/isaaclab dependency at all — that's the point of
using MuJoCo here.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
import torch

from talon_rl.config import RewardVectorCfg
from rl.core.sim2sim import rollout

_DEFAULT_A1_MUJOCO_XML = "talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="TorchScript policy exported by play.py --export.")
    parser.add_argument("--mujoco_xml", type=str, default=_DEFAULT_A1_MUJOCO_XML)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0], help="v_x v_y omega_z")
    reward_cfg = RewardVectorCfg()
    parser.add_argument(
        "--preference", type=float, nargs=reward_cfg.dim,
        default=[1.0 / reward_cfg.dim] * reward_cfg.dim,
        help=f"w for {reward_cfg.term_names} — must sum to 1.",
    )
    args = parser.parse_args()

    policy = torch.jit.load(args.policy)
    model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # the "home" standing keyframe

    stats = rollout(
        policy, model, data, steps=args.steps,
        command=np.array(args.command, dtype=np.float32),
        preference=np.array(args.preference, dtype=np.float32),
    )

    print(f"ran {args.steps} steps | all_finite={stats['all_finite']} "
          f"height: final={stats['final_height']:.3f} min={stats['min_height']:.3f} max={stats['max_height']:.3f}")


if __name__ == "__main__":
    main()
