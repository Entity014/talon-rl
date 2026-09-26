#!/usr/bin/env python3
"""Stochastic action-direction diversity probe -- follow-up to
objective_perturbation_probe.py (deterministic action barely reorients
under a command perturbation, cos_sim>=0.99) and a direct checkpoint
inspection (log_std is STATE-INDEPENDENT by architecture -- `std =
self.log_std.exp()` has zero dependence on actor_obs_w -- and identical
across every Case 3 seed AND the Case 1 baseline: -0.1968, pinned at its
own annealing ceiling, std=exp(-0.1968)=0.822, not collapsed near zero).

That ruled out "entropy has collapsed specifically at this state" (there
is no such thing as state-local entropy in this architecture) but left
open whether a nominally-substantial pre-tanh std=0.822 actually produces
diverse ACTION DIRECTIONS once squashed through tanh at a stable-
attractor state's mean, or whether the squashing/actor_mean geometry
collapses stochastic samples back toward nearly the same direction
regardless. Three outcomes distinguish where the real bottleneck is:

  1. High post-tanh direction diversity (samples spread widely around
     the deterministic mean) -- exploration genuinely reaches different
     directions here; entropy-collapse is NOT the explanation, look at
     PPO actor-optimization/mean-action geometry instead.
  2. Std produces magnitude variation but direction stays locked
     (cos_sim to deterministic mean stays high, e.g. >0.95-0.99, across
     most samples) -- nominal Gaussian exploration isn't translating
     into locomotion-relevant directional exploration AFTER tanh --
     points at the squashing/action-mapping as a directional bottleneck.
  3. Pre-tanh (u) samples themselves have low direction diversity, before
     tanh is even applied -- points at the actor distribution/mean
     geometry itself (the network's raw output landscape), not the
     squashing.

Reports BOTH pre-tanh (u) and post-tanh (action) direction diversity
separately per frozen stable state, to tell 2 and 3 apart, per the
user's explicit requirement -- collapsing them together would hide
exactly which stage (raw Gaussian sample vs squashing) direction
diversity is lost at, if it is.

No physical rollout involved beyond the one used to LOCATE stable
states (same technique as objective_segment_audit.py/
objective_perturbation_probe.py) -- sampling and squashing happen
entirely via the loaded model, off the simulated trajectory.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py stochastic-diversity-probe \
        runs/phase1_hipact_dt01_seed1_2026-09-20/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 \
        --steps 200 --sim_dt 0.01 --n_states 15 --n_samples 256
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


def _pairwise_cos_sim_mean(x: torch.Tensor) -> float:
    """x: (N, D). Mean cosine similarity over all N*(N-1)/2 unordered pairs."""
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    sim = x_norm @ x_norm.T  # (N, N)
    n = x.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return sim[iu[0], iu[1]].mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4, help="Force this w (RewardVectorCfg.term_names order), need not sum to 1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--stable_vx_thresh", type=float, default=0.15)
    parser.add_argument("--fall_exclude_window", type=int, default=5)
    parser.add_argument("--n_states", type=int, default=15, help="Number of distinct frozen stable states to probe")
    parser.add_argument("--n_samples", type=int, default=256, help="Stochastic action samples drawn per state")
    parser.add_argument("--cos_thresholds", type=float, nargs="+", default=[0.9, 0.7, 0.5])
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
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()  # noqa: F841
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
    print(f"loaded checkpoint (t={trainer._t})  log_std={trainer.model.log_std.detach().cpu().numpy().round(4)}")

    w_forced = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)
    trainer.w = w_forced.copy()
    env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    orig_sample = moppo_mod.sample_preference_vector

    def _fixed_sample(rng, reward_cfg_, pref_cfg_, n, step=0):
        return w_forced[:n].copy()

    moppo_mod.sample_preference_vector = _fixed_sample

    T, N = args.steps, args.num_envs
    v_x_rec = np.zeros((T, N), dtype=np.float32)
    _step_idx = [0]
    orig_step = env.step

    def _wrapped_step(action):
        transition, done = orig_step(action)
        v_x_rec[_step_idx[0]] = transition["v_actual"][:, 0]
        _step_idx[0] += 1
        return transition, done

    env.step = _wrapped_step
    try:
        r = trainer._collect_rollout()
    finally:
        env.step = orig_step
        moppo_mod.sample_preference_vector = orig_sample

    terminal_mask = r["dones"]
    fall_exclude = terminal_mask.copy()
    for shift in range(1, args.fall_exclude_window + 1):
        fall_exclude[:-shift] |= terminal_mask[shift:]
        fall_exclude[shift:] |= terminal_mask[:-shift]
    valid = np.ones((T, N), dtype=bool)
    valid[:args.warmup] = False
    valid[fall_exclude] = False
    stable_mask = valid & (np.abs(v_x_rec) < args.stable_vx_thresh)

    t_idx, n_idx = np.nonzero(stable_mask)
    n_available = len(t_idx)
    print(f"{n_available} stable samples available")
    if n_available == 0:
        print("no stable samples found -- nothing to probe")
        return

    rng = np.random.default_rng(args.seed)
    n_states = min(args.n_states, n_available)
    sel = rng.choice(n_available, size=n_states, replace=False)
    t_sel, n_sel = t_idx[sel], n_idx[sel]

    device = trainer.device
    model = trainer.model
    action_clip = model.ACTION_CLIP
    K = args.n_samples

    agg = {"pretanh_pairwise": [], "posttanh_pairwise": [], "pretanh_cos_to_mean": [], "posttanh_cos_to_mean": [],
           "posttanh_std_per_dim": [], "frac_below": {th: [] for th in args.cos_thresholds}}

    for i in range(n_states):
        obs_row = torch.from_numpy(r["actor_obs"][t_sel[i], n_sel[i]]).to(device).unsqueeze(0)  # (1, D)
        obs_batch = obs_row.repeat(K, 1)  # (K, D)
        with torch.no_grad():
            dist = model._pre_tanh_dist(obs_batch)
            mean = dist.mean[0]  # (action_dim,) -- same for every row in this batch (same state)
            u = dist.sample()  # (K, action_dim) -- pre-tanh samples
            action = torch.tanh(u) * action_clip  # (K, action_dim) -- post-tanh samples
            det_action = torch.tanh(mean) * action_clip  # deterministic reference

        pretanh_pairwise = _pairwise_cos_sim_mean(u)
        posttanh_pairwise = _pairwise_cos_sim_mean(action)

        mean_norm = mean / (mean.norm() + 1e-8)
        u_norm = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
        pretanh_cos_to_mean = (u_norm @ mean_norm).mean().item()

        det_norm = det_action / (det_action.norm() + 1e-8)
        action_norm = action / (action.norm(dim=-1, keepdim=True) + 1e-8)
        cos_to_det = action_norm @ det_norm  # (K,)
        posttanh_cos_to_mean = cos_to_det.mean().item()

        posttanh_std = action.std(dim=0).mean().item()  # mean over action dims of per-dim std across K samples

        agg["pretanh_pairwise"].append(pretanh_pairwise)
        agg["posttanh_pairwise"].append(posttanh_pairwise)
        agg["pretanh_cos_to_mean"].append(pretanh_cos_to_mean)
        agg["posttanh_cos_to_mean"].append(posttanh_cos_to_mean)
        agg["posttanh_std_per_dim"].append(posttanh_std)
        for th in args.cos_thresholds:
            agg["frac_below"][th].append((cos_to_det < th).float().mean().item())

    print(f"\n=== {n_states} frozen stable states x {K} stochastic samples each ===")
    print(f"pre-tanh (u)  pairwise cos_sim:  mean={np.mean(agg['pretanh_pairwise']):.4f}  "
          f"(1.0=all samples same direction, 0=orthogonal)")
    print(f"post-tanh (action) pairwise cos_sim: mean={np.mean(agg['posttanh_pairwise']):.4f}")
    print(f"pre-tanh (u) cos_sim to mean:     mean={np.mean(agg['pretanh_cos_to_mean']):.4f}")
    print(f"post-tanh (action) cos_sim to deterministic action: mean={np.mean(agg['posttanh_cos_to_mean']):.4f}")
    print(f"post-tanh per-dim std (empirical, across samples): mean={np.mean(agg['posttanh_std_per_dim']):.4f}")
    for th in args.cos_thresholds:
        print(f"fraction of samples with post-tanh cos_sim(action, det_action) < {th}: "
              f"mean={np.mean(agg['frac_below'][th]):.4f}")


if __name__ == "__main__":
    main()
