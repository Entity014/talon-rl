#!/usr/bin/env python3
"""Overlays 2 hip_intervention_trace.py .npz outputs (different
CHECKPOINTS -- e.g. K=5 baseline vs K=5+hip_activation, same seed/lane --
run with --mode none, i.e. no action override, just each checkpoint's own
policy) to answer Experiment 2A.5's follow-up question directly:

    hip activity -> functional gait/contact -> v_x    (real mechanism)
        vs
    hip activity -> reward satisfaction only           (loophole)

Pure numpy/matplotlib, no Isaac Sim -- same reasoning as
combine_hip_intervention_traces.py for keeping this separate from the
env-touching trace script.

    python scripts/rl/combine_checkpoint_traces.py \
        --baseline logs/.../seed1_baseline.npz --treatment logs/.../seed1_hipact.npz \
        --out logs/.../seed1_baseline_vs_hipact.png
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {"baseline (K=5)": (np.load(args.baseline), "-", "C0"), "+hip_activation": (np.load(args.treatment), "--", "C1")}

    panels = [
        ("height", "m"), ("v_z", "m/s"), ("v_x", "m/s"),
        ("hip_L_qdot", "rad/s"), ("hip_R_qdot", "rad/s"),
        ("contact_L", "frac"), ("contact_R", "frac"),
        ("n_feet_contact", "count"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 20), sharex=True)
    for ax, (name, unit) in zip(axes, panels):
        for label, (data, ls, color) in runs.items():
            t = np.arange(len(data[name]))
            ax.plot(t, data[name], linestyle=ls, color=color, label=label, alpha=0.85)
        if name == "v_x":
            ax.axhline(float(runs["baseline (K=5)"][0]["command_vx"]), color="green", linestyle=":", alpha=0.7, label="command")
        ax.set_ylabel(f"{name}\n({unit})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax, _ in zip(axes, panels):
        for label, (data, _, color) in runs.items():
            for fs in np.where(data["fell"])[0]:
                ax.axvline(fs, color=color, linestyle=(0, (1, 5)), alpha=0.3)

    axes[-1].set_xlabel("step (faint vertical lines = fall events, colored per checkpoint)")
    fig.suptitle("baseline vs hip_activation checkpoint trace comparison")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
