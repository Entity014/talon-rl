#!/usr/bin/env python3
"""One-update PPO/D3PO instrumentation -- the branch objective_segment_audit
/objective_perturbation_probe/stochastic_diversity_probe narrowed the
local-attractor question to: reward weighting, critic blindness, entropy
collapse, and tanh squashing are all ruled out. The remaining hypothesis
is PPO actor optimization/mean-update geometry -- the actor has real
stochastic exploration and the critic clearly signals better directions
exist, but the deterministic mean doesn't migrate toward them. This
script inspects ONE real update() call's internals directly (no
retraining, no multi-update loop) to distinguish where in the pipeline
that migration is failing:

  1. ADVANTAGE SIGN -- do forward-producing samples actually get positive
     progress advantage, and stable samples negative/near-zero, in a
     REAL training batch (not the forced-w diagnostic rollouts used so
     far)? (objective_segment_audit.py already answered this under a
     forced w; here it's checked under the real, per-lane resampled w
     _collect_rollout uses in actual training.)
  2. IMPORTANCE RATIO r=exp(logp_new-logp_old) and PPO CLIP FRACTION --
     per group (forward vs stable), what fraction of samples have their
     surrogate objective REDUCED by clipping, and is that concentrated
     on the positive-advantage (forward) samples specifically?
  3. GRADIENT DECOMPOSITION -- backward() the D3PO clip loss using ONLY
     forward-group samples vs ONLY stable-group samples (two separate,
     independent backward passes, no optimizer step), compare actor
     parameter gradient norms. If forward-only gradient is near zero
     despite positive advantage -- clipping/batch-geometry is consuming
     it before it ever reaches the parameters.
  4. ACTUAL PARAMETER UPDATE -- runs ONE real optimizer step (full loss:
     clip_loss + diversity + entropy + mean_reg, matching update()'s own
     per-minibatch loss exactly), then re-evaluates the SAME frozen
     stable states' deterministic mean action before vs after, reporting
     cosine similarity. If gradient was non-trivial (step 3) but this
     cosine stays ~1.0, the optimizer/LR/parameterization is absorbing
     the gradient without moving the mean.

Decision tree (do not skip past unclear results to pick a hypothesis):
  forward samples have positive advantage
    -> gradient (step 3) near-zero despite that -> PPO clipping/batch geometry
    -> gradient present, but parameter update (step 4) barely moves mean -> optimizer/LR/parameterization
    -> gradient present AND mean moves, but not toward the forward direction -> actor architecture/action mapping

This calls trainer._collect_rollout() and reproduces update()'s own
batch-construction logic verbatim (advantage/return computation, w
extraction from stored actor_obs, minibatch loss terms) rather than
calling update() itself, because update() doesn't expose per-sample
ratio/advantage/clip data or let gradient be computed on a masked
subset -- this is intentionally NOT a rewrite of the training algorithm,
every formula mirrors moppo.py's update()/losses.py's d3po_actor_loss
line for line.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py ppo-update-diagnostic \
        runs/phase1_hipact_dt01_seed1_2026-09-20/checkpoints/checkpoint.pt \
        --steps 200 --sim_dt 0.01
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from talon_rl.config import (
    ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg,
)

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
from rl.core.objectives.losses import d3po_actor_loss, diversity_regularizer_loss, normalize_per_objective
from rl.core.rollout.gae_functional import gae_per_objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200, help="Rollout window; the REAL num_steps=24 training window is too short to reliably contain both stable and forward-moving samples for a stable checkpoint -- this uses a longer window purely to find enough of both groups, then treats it as one big minibatch for the update math (matches num_minibatches=1 semantics, not the real 4)")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--stable_vx_thresh", type=float, default=0.15)
    parser.add_argument("--forward_vx_thresh", type=float, default=0.35)
    parser.add_argument("--fall_exclude_window", type=int, default=5)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--sim_dt", type=float, default=0.02)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_encoder", action="store_true")
    args = parser.parse_args()

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
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t})")
    env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    # Real training's own w: per-lane resampled preference (NOT forced) --
    # deliberately NOT monkey-patching sample_preference_vector here, so
    # this rollout uses exactly the w-sampling machinery real training does.
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

    # ---- reproduce update()'s own batch construction verbatim ----
    values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)
    adv = gae_per_objective(r["rewards"], values_with_final, r["dones"], moppo_cfg.gamma, moppo_cfg.gae_lambda)
    returns = adv + r["values"]
    w_used = r["actor_obs"][:, :, -reward_cfg.dim:]
    adv_t = torch.from_numpy(normalize_per_objective(adv.reshape(T * N, -1))).to(trainer.device)
    w_t = torch.from_numpy(w_used.reshape(T * N, -1).astype(np.float32)).to(trainer.device)
    actor_obs_t = torch.from_numpy(r["actor_obs"].reshape(T * N, -1)).to(trainer.device)
    critic_obs_t = torch.from_numpy(r["critic_obs"].reshape(T * N, -1)).to(trainer.device)
    actions_t = torch.from_numpy(r["actions"].reshape(T * N, -1)).to(trainer.device)
    logp_old_t = torch.from_numpy(r["logp"].reshape(T * N)).to(trainer.device)
    returns_t = torch.from_numpy(returns.reshape(T * N, -1).astype(np.float32)).to(trainer.device)

    # ---- group masks (same convention as objective_segment_audit.py) ----
    terminal_mask = r["dones"]
    fall_exclude = terminal_mask.copy()
    for shift in range(1, args.fall_exclude_window + 1):
        fall_exclude[:-shift] |= terminal_mask[shift:]
        fall_exclude[shift:] |= terminal_mask[:-shift]
    valid = np.ones((T, N), dtype=bool)
    valid[:args.warmup] = False
    valid[fall_exclude] = False
    abs_vx = np.abs(v_x_rec)
    group_a = (valid & (abs_vx < args.stable_vx_thresh)).reshape(-1)
    group_b = (valid & (v_x_rec > args.forward_vx_thresh)).reshape(-1)
    print(f"group A (stable): {group_a.sum()} samples  group B (forward): {group_b.sum()} samples "
          f"(of {T * N} total)")
    if group_a.sum() == 0 or group_b.sum() == 0:
        print("one or both groups empty -- cannot run the diagnostic, try a longer --steps or different --seed")
        return

    idx_a = torch.from_numpy(np.nonzero(group_a)[0]).to(trainer.device)
    idx_b = torch.from_numpy(np.nonzero(group_b)[0]).to(trainer.device)
    progress_idx = reward_cfg.term_names.index("progress")

    # ---- 1: advantage sign under REAL (not forced) w ----
    print("\n=== 1. advantage sign (normalized, real per-lane w) ===")
    for name, idx in (("A (stable)", idx_a), ("B (forward)", idx_b)):
        prog_adv = adv_t[idx, progress_idx]
        print(f"  group {name}: mean progress adv={prog_adv.mean().item():.4f}  "
              f"frac positive={float((prog_adv > 0).float().mean()):.4f}")

    # ---- 2: importance ratio + clip fraction (single forward pass, no grad needed for these stats) ----
    with torch.no_grad():
        logp_new_full = trainer._logp(actor_obs_t, actions_t)
        ratio_full = torch.exp(logp_new_full - logp_old_t)
        clipped_ratio_full = torch.clamp(ratio_full, 1 - moppo_cfg.clip_eps, 1 + moppo_cfg.clip_eps)
        # per-objective clipped-vs-unclipped surrogate, progress column only
        unclipped_term = ratio_full * adv_t[:, progress_idx]
        clipped_term = clipped_ratio_full * adv_t[:, progress_idx]
        is_clip_binding = clipped_term < unclipped_term  # True where clipping REDUCED the surrogate (the min() picked the clipped branch)

    print("\n=== 2. importance ratio + PPO clip fraction (progress objective) ===")
    for name, idx in (("A (stable)", idx_a), ("B (forward)", idx_b)):
        r_g = ratio_full[idx]
        clip_frac = float(is_clip_binding[idx].float().mean())
        # among positive-advantage samples specifically, what fraction got clipped down?
        pos_mask = adv_t[idx, progress_idx] > 0
        clip_frac_pos = float(is_clip_binding[idx][pos_mask].float().mean()) if pos_mask.any() else float("nan")
        print(f"  group {name}: ratio mean={r_g.mean().item():.4f} std={r_g.std().item():.4f}  "
              f"clip_fraction={clip_frac:.4f}  clip_fraction(among A>0 samples)={clip_frac_pos:.4f}")

    # ---- 3: gradient decomposition -- forward-only vs stable-only backward, no optimizer step ----
    def group_grad_norm(idx: torch.Tensor) -> float:
        trainer.optim.zero_grad()
        logp_new = trainer._logp(actor_obs_t[idx], actions_t[idx])
        ratio = torch.exp(logp_new - logp_old_t[idx])
        loss = d3po_actor_loss(ratio, adv_t[idx], w_t[idx], moppo_cfg.clip_eps)
        loss.backward()
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        trainer.optim.zero_grad()
        return total_norm ** 0.5

    grad_norm_a = group_grad_norm(idx_a)
    grad_norm_b = group_grad_norm(idx_b)
    print("\n=== 3. actor gradient norm, isolated per group (clip_loss only, no optimizer step) ===")
    print(f"  group A (stable)  ||grad||={grad_norm_a:.6f}")
    print(f"  group B (forward) ||grad||={grad_norm_b:.6f}")

    # ---- 4: one real optimizer step (full loss, matching update()'s own minibatch loss), before/after mean-action cosine at frozen probe states ----
    n_probe = min(50, int(idx_a.shape[0]))
    probe_idx = idx_a[torch.randperm(idx_a.shape[0], device=trainer.device)[:n_probe]]
    probe_obs = actor_obs_t[probe_idx]
    with torch.no_grad():
        mean_before = trainer._act_inference(probe_obs).clone()

    trainer.optim.zero_grad()
    logp_new = trainer._logp(actor_obs_t, actions_t)
    ratio = torch.exp(logp_new - logp_old_t)
    clip_loss = d3po_actor_loss(ratio, adv_t, w_t, moppo_cfg.clip_eps)
    w_prime = torch.from_numpy(trainer._sample_diversity_w(T * N)).to(trainer.device)
    actor_obs_prime = actor_obs_t.clone()
    actor_obs_prime[:, -reward_cfg.dim:] = w_prime
    mean_w = trainer._act_inference(actor_obs_t)
    mean_w_prime = trainer._act_inference(actor_obs_prime)
    diversity_loss = diversity_regularizer_loss(
        mean_w, mean_w_prime, w_t, w_prime, trainer.model.log_std.exp(), moppo_cfg.diversity_alpha,
    )
    policy_loss = clip_loss + moppo_cfg.diversity_lambda * diversity_loss
    values_pred = trainer._value(critic_obs_t)
    value_loss = nn.functional.mse_loss(values_pred, returns_t)
    entropy_bonus = trainer._entropy(actor_obs_t).mean()
    mean_reg = trainer._raw_mean(actor_obs_t).pow(2).mean()
    full_loss = (
        policy_loss + 0.5 * value_loss - moppo_cfg.entropy_coef * entropy_bonus + moppo_cfg.mean_reg_coef * mean_reg
    )
    full_loss.backward()
    full_grad_norm = sum(p.grad.norm().item() ** 2 for p in trainer.model.parameters() if p.grad is not None) ** 0.5
    trainer.optim.step()
    with torch.no_grad():
        trainer.model.log_std.clamp_(trainer.model.LOG_STD_MIN, trainer._log_std_max)

    with torch.no_grad():
        mean_after = trainer._act_inference(probe_obs).clone()

    delta_mean = mean_after - mean_before
    cos_sim = torch.sum(mean_before * mean_after, dim=-1) / (
        mean_before.norm(dim=-1) * mean_after.norm(dim=-1) + 1e-8
    )
    print(f"\n=== 4. ONE real optimizer step (full loss, ||grad||={full_grad_norm:.6f}) -- "
          f"probe states BEFORE vs AFTER ===")
    print(f"  n_probe={n_probe} frozen stable states")
    print(f"  |Δmean_action| (L2): mean={delta_mean.norm(dim=-1).mean().item():.6f}  "
          f"max={delta_mean.norm(dim=-1).max().item():.6f}")
    print(f"  cos_sim(mean_before, mean_after): mean={cos_sim.mean().item():.6f}  "
          f"min={cos_sim.min().item():.6f}  (1.0=no directional change from this single update)")


if __name__ == "__main__":
    main()
