#!/usr/bin/env python3
"""Overlays 3 hip_intervention_trace.py .npz outputs (none/R_follows_L/
L_follows_R, same checkpoint+seed+lane) on one set of panels, so the
causal ordering of what changes first under the intervention is visible
directly -- pure numpy/matplotlib, no Isaac Sim needed (kept separate
from hip_intervention_trace.py since that one needs a live env per mode
and this repo's other scripts all create exactly one Isaac Lab env per
process; three .npz files in, one comparison figure out avoids testing
whether re-creating multiple envs in one process is safe).

    python scripts/rl/diagnostics.py combine-hip-intervention-traces \
        --none logs/.../seed0_none.npz --r_follows_l logs/.../seed0_R_follows_L.npz \
        --l_follows_r logs/.../seed0_L_follows_R.npz --out logs/.../seed0_combined.png
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--none", required=True)
    parser.add_argument("--r_follows_l", required=True)
    parser.add_argument("--l_follows_r", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {
        "none": (np.load(args.none), "-"),
        "R_follows_L": (np.load(args.r_follows_l), "--"),
        "L_follows_R": (np.load(args.l_follows_r), ":"),
    }

    panels = [
        ("height", "m"), ("v_z", "m/s"), ("pitch", "rad"), ("pitch_rate", "rad/s"),
        ("hip_L_torque", "Nm"), ("hip_R_torque", "Nm"),
        ("hip_L_qdot", "rad/s"), ("hip_R_qdot", "rad/s"),
        ("thigh_dq_target", "rad"), ("calf_dq_target", "rad"),
        ("n_feet_contact", "count"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 26), sharex=True)
    for ax, (name, unit) in zip(axes, panels):
        for mode, (data, ls) in runs.items():
            t = np.arange(len(data[name]))
            ax.plot(t, data[name], linestyle=ls, label=mode, alpha=0.85)
        ax.set_ylabel(f"{name}\n({unit})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Fall markers per mode, colored to match that mode's line style legend.
    colors = {"none": "C0", "R_follows_L": "C1", "L_follows_R": "C2"}
    for ax, _ in zip(axes, panels):
        for mode, (data, _) in runs.items():
            for fs in np.where(data["fell"])[0]:
                ax.axvline(fs, color=colors[mode], linestyle=(0, (1, 5)), alpha=0.3)

    axes[-1].set_xlabel("step (faint vertical lines = fall events, colored per mode)")
    fig.suptitle("hip_symmetry_intervention trace comparison")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
