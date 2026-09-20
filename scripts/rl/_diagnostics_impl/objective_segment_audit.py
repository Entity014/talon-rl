#!/usr/bin/env python3
"""Objective/advantage audit, segmented by actual v_x -- the branch opened
after Case 3 (talon-thesis 2026-09-20's training-timestep causality test)
showed training at the numerically-validated dt=0.01 fixes stability
(falls/torque-saturation/contact all improve, all 3 seeds) but NOT
forward-velocity command tracking (v_x stays near/below the command in
every seed). That separates two previously-conflated bottlenecks; this
script investigates the SURVIVING one: why does an objective-conditioned
policy converge to a stable, low-v_x attractor when the progress
objective and command signal are both demonstrably present (2C already
showed the actor is command-sensitive in isolation)?

Distinguishes three hypotheses (do NOT retrain or touch reward
coefficients based on this script's output alone -- it's meant to pick
which of these branches deserves that, not to justify a coefficient
change on its own):

  A. Progress objective's realized advantage is genuinely small in
     stable/low-v_x segments relative to other objectives -- look at
     objective weighting / advantage normalization next.
  B. Progress advantage has real magnitude/direction (pushing toward
     more v_x) even in stable segments, but the policy doesn't act on
     it -- local optimum / exploration / policy learning dynamics.
  C. Progress advantage is real AND the policy's actions differ between
     groups (comparable or larger action magnitude in stable segments),
     but physical v_x still doesn't follow -- back to contact-
     conditioned controllability / action-to-motion pathway (a
     DIFFERENT question than 2C/2D/2E's controller-tracks-target-
     position question -- this would be about whether a given ACTION
     actually produces forward motion at all, not tracking accuracy).

Method: reuses MOPPOTrainer._collect_rollout() verbatim for pipeline
fidelity (same monkey-patched forced-w trick as advantage_decomposition.py
-- see that script's docstring for why), over a --steps window long
enough to reach the trained policy's actual attractor (default 200,
matching attractor_phase_metrics.py's Phase C horizon, not the 24-step
PPO minibatch window advantage_decomposition.py normally uses). GAE is
computed the same way (gae_per_objective + normalize_per_objective) so
the "normalized advantage" reported here is exactly what the policy
gradient would see under this w.

Per-step v_x/height/v_z/foot-contact/action-magnitude are NOT part of
_collect_rollout's return dict, so env.step is wrapped (not modified) to
record them as they pass through -- restored afterward, no change to the
env or trainer objects survives this script.

Segments each (t>=--warmup, lane) sample into:
  group A "stable"  : |v_x_actual| < --stable_vx_thresh
  group B "forward"  : v_x_actual > --forward_vx_thresh
(samples in between are excluded from both -- an ambiguous band, not
worth attributing to either attractor). Reports, per group: sample
count, raw reward per objective, MEAN SIGNED normalized advantage per
objective (direction, not just magnitude -- is progress still pushing
toward more v_x even in stable segments?), mean |normalized advantage|
per objective, w_i * mean|adv_i| (each objective's realized share of
the weighted-sum policy gradient, same convention as
advantage_decomposition.py), plus context: height, |v_z|, n_feet_contact,
action magnitude, mean v_x (actual) vs command -- so a segment is never
called "stable" from v_x alone without also reporting whether it's
physically stable (height/contact/v_z), per the audit's own design
requirement.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py objective-segment-audit \
        runs/phase1_hipact_dt01_seed1_2026-09-20/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 \
        --steps 200 --sim_dt 0.01
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from talon_rl.config import (
    ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg,
)

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
import rl.core.algorithms.moppo as moppo_mod
from rl.core.losses import normalize_per_objective
from rl.core.storage.rollout_storage import gae_per_objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4, help="Force this w (RewardVectorCfg.term_names order), need not sum to 1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200, help="Rollout window length -- long enough to reach the trained policy's attractor, not the 24-step PPO minibatch window")
    parser.add_argument("--warmup", type=int, default=15, help="Exclude steps before this (initial-drop transient, same convention as attractor_phase_metrics.py's Phase B/C split)")
    parser.add_argument("--stable_vx_thresh", type=float, default=0.15)
    parser.add_argument("--forward_vx_thresh", type=float, default=0.35)
    parser.add_argument(
        "--fall_exclude_window", type=int, default=5,
        help="Exclude samples within +-N steps of ANY terminal-fall step (per lane), not just the "
             "exact fall step -- a single-step exclusion lets near-fall transient dynamics (high "
             "|v_z|, low contact) leak into the low-v_x 'stable' bucket and get misread as a "
             "standing attractor. 5 chosen as the primary value (same conservative pre-fall window "
             "used for the progress-leak fix, Experiment 2A.3); this arc's own fall-cycle length is "
             "~13-15 steps, so 5 trims the transient without eating the whole cycle. Pass 0 to "
             "disable (old behavior -- exact-step-only exclusion).",
    )
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--sim_dt", type=float, default=0.02)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_encoder", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app  # noqa: F841 -- kept alive for the process lifetime

    import gymnasium as gym
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401 -- registers Isaac-Talon-A1-v0
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()  # noqa: F841 -- kept for parity with advantage_decomposition.py's imports
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg()
    extrinsics_cfg = None if args.no_encoder else ExtrinsicsCfg()

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.dt = args.sim_dt
    cfg.decimation = args.decimation
    cfg.sim.render_interval = cfg.decimation
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    moppo_cfg = MOPPOConfig(num_steps=args.steps)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(checkpoint_path)
    print(f"loaded checkpoint (t={trainer._t}, penalty_k={trainer._penalty_k:.4f})")

    w_forced = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)
    trainer.w = w_forced.copy()
    env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    orig_sample = moppo_mod.sample_preference_vector

    def _fixed_sample(rng, reward_cfg_, pref_cfg_, n, step=0):
        return w_forced[:n].copy()

    moppo_mod.sample_preference_vector = _fixed_sample

    # Wrap env.step (not modified, restored after) to record per-step
    # telemetry _collect_rollout's own return dict doesn't carry.
    T, N = args.steps, args.num_envs
    v_x_rec = np.zeros((T, N), dtype=np.float32)
    height_rec = np.zeros((T, N), dtype=np.float32)
    vz_rec = np.zeros((T, N), dtype=np.float32)
    contact_rec = np.zeros((T, N), dtype=np.float32)
    action_mag_rec = np.zeros((T, N), dtype=np.float32)
    _step_idx = [0]
    orig_step = env.step

    def _wrapped_step(action):
        transition, done = orig_step(action)
        t = _step_idx[0]
        v_x_rec[t] = transition["v_actual"][:, 0]
        height_rec[t] = transition["height"]
        vz_rec[t] = transition["v_z"]
        cf = transition.get("foot_contact_force")
        contact_rec[t] = (cf > 1.0).sum(axis=-1) if cf is not None else np.nan
        action_mag_rec[t] = np.abs(action).mean(axis=-1)
        _step_idx[0] += 1
        return transition, done

    env.step = _wrapped_step
    try:
        r = trainer._collect_rollout()
    finally:
        env.step = orig_step
        moppo_mod.sample_preference_vector = orig_sample

    values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)
    adv_raw = gae_per_objective(r["rewards"], values_with_final, r["dones"], moppo_cfg.gamma, moppo_cfg.gae_lambda)
    adv_norm = normalize_per_objective(adv_raw.reshape(T * N, -1)).reshape(T, N, -1)

    names = reward_cfg.term_names
    w_normed = np.array(args.w, dtype=np.float32) / sum(args.w)

    # terminal_fall-style exclusion (don't attribute the exact fall step to
    # either group -- its v_x is a transient artifact of the fall itself,
    # same reasoning prefall_window_analysis.py/attractor_phase_metrics.py
    # already apply elsewhere in this arc).
    terminal_mask = r["dones"]  # (T, N) -- real termination (terminal_fall) only, see _collect_rollout's own comment
    fall_exclude = terminal_mask.copy()
    for shift in range(1, args.fall_exclude_window + 1):
        fall_exclude[:-shift] |= terminal_mask[shift:]   # a fall within N steps AHEAD
        fall_exclude[shift:] |= terminal_mask[:-shift]   # a fall within N steps BEHIND
    valid = np.ones((T, N), dtype=bool)
    valid[:args.warmup] = False
    valid[fall_exclude] = False

    abs_vx = np.abs(v_x_rec)
    group_a = valid & (abs_vx < args.stable_vx_thresh)          # "stable" / low-v_x
    group_b = valid & (v_x_rec > args.forward_vx_thresh)        # "forward-moving"

    def report(mask: np.ndarray, label: str) -> None:
        n = int(mask.sum())
        print(f"\n=== group {label}: n={n} samples ({n / max(int(valid.sum()), 1) * 100:.1f}% of valid) ===")
        if n == 0:
            print("  (empty)")
            return
        print(f"  context: v_x(actual)={v_x_rec[mask].mean():.4f} (cmd={args.command[0]})  "
              f"height={height_rec[mask].mean():.4f}  |v_z|={np.abs(vz_rec[mask]).mean():.4f}  "
              f"n_feet_contact={contact_rec[mask].mean():.4f}  action_mag={action_mag_rec[mask].mean():.4f}")
        print(f"  {'term':<12}{'raw reward':>14}{'signed adv':>14}{'|adv|':>14}{'w_i*|adv_i|':>14}")
        contributions = []
        for k, name in enumerate(names):
            raw_reward = r["rewards"][:, :, k][mask].mean()
            signed_adv = adv_norm[:, :, k][mask].mean()
            abs_adv = np.abs(adv_norm[:, :, k][mask]).mean()
            contrib = w_normed[k] * abs_adv
            contributions.append(contrib)
            print(f"  {name:<12}{raw_reward:>14.4f}{signed_adv:>14.4f}{abs_adv:>14.4f}{contrib:>14.4f}")
        total = sum(contributions)
        if total > 0:
            for k, name in enumerate(names):
                print(f"    {name}: {contributions[k] / total * 100:.1f}% share of weighted-sum magnitude")

    print(f"\nw = {dict(zip(names, args.w))}  command={args.command}  (T={T}, N={N}, warmup={args.warmup}, "
          f"fall_exclude_window={args.fall_exclude_window})")
    print(f"thresholds: stable=|v_x|<{args.stable_vx_thresh}  forward=v_x>{args.forward_vx_thresh}")
    report(group_a, "A (stable / low-v_x)")
    report(group_b, "B (forward-moving)")


if __name__ == "__main__":
    main()
