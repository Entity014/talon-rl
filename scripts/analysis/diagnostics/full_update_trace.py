#!/usr/bin/env python3
"""Full update() trace -- follow-up to mean_update_decomposition.py's single-
minibatch result (both Case 3 seeds: actor gradient is TANGENTIAL-dominant
at a single step, not radial-dominant as hypothesized; entropy contributes
exactly 0.0 to Δmu, mean_reg is the radial-biased term). That result only
characterized ONE minibatch step in isolation. This script asks the
follow-up question directly: across a REAL full update() call (all
epochs_per_update epochs x num_minibatches minibatches, same reshuffling,
same Adam instance carrying momentum/variance across steps), does the
tangential drift measured at step 1 accumulate, or does it get cancelled
out by later minibatches/epochs?

No retrain: works on a deep copy of the loaded checkpoint's model (+encoder
if present), a fresh Adam with the same hyperparameters as the real
trainer.optim, and replays moppo.py's own update() minibatch loop verbatim
(same GAE/normalize_per_objective/loss construction/log_std clamp) on ONE
real rollout collected from the live checkpoint -- the real trainer and its
checkpoint file are never modified.

Per minibatch step, logs:
  - ||Δmu|| and tan_frac (radial/tangential split of THIS step's Δmu on the
    frozen probe set, against mu_before_this_step) -- same decomposition as
    mean_update_decomposition.py, but for every step in the real loop, not
    just the first.
  - cos(mu_t, mu_0) -- cumulative angular drift from the pre-update mean.
  - actor_grad_norm / mean_reg_grad_norm -- L2 norm of d(policy_loss)/d(actor
    params) and d(mean_reg_coef*mean_reg)/d(actor params) respectively,
    computed via torch.autograd.grad(..., retain_graph=True) SEPARATELY from
    the real combined-loss backward+step (so instrumentation never changes
    what the optimizer actually does).
  - ratio_mean and clip_fraction (|ratio-1| > clip_eps, PRE-clip -- same
    quantity flagged in an earlier session note as "clipping not active at
    the inspected pre-update minibatch," now tracked across every minibatch
    of a real update, not just one).

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py full-update-trace \
        runs/phase1_hipact_dt01_seed0_2026-09-20/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 \
        --steps 200 --sim_dt 0.01
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn

from talon_rl.config import (
    ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg,
)

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
import rl.core.algorithms.moppo as moppo_mod
from rl.core.objectives.losses import d3po_actor_loss, diversity_regularizer_loss, normalize_per_objective
from rl.core.rollout.gae_functional import gae_per_objective


def _radial_tangential_mean(delta_mu: np.ndarray, mu_before: np.ndarray) -> tuple[float, float, float]:
    mu_before_sq = np.sum(mu_before * mu_before, axis=-1, keepdims=True)
    mu_before_sq_safe = np.where(mu_before_sq < 1e-12, 1.0, mu_before_sq)
    coeff = np.sum(delta_mu * mu_before, axis=-1, keepdims=True) / mu_before_sq_safe
    radial_vec = coeff * mu_before
    tangential_vec = delta_mu - radial_vec
    delta_norm = np.linalg.norm(delta_mu, axis=-1)
    radial_norm = np.linalg.norm(radial_vec, axis=-1)
    tangential_norm = np.linalg.norm(tangential_vec, axis=-1)
    tan_frac = np.median(tangential_norm / np.maximum(delta_norm, 1e-8))
    return float(delta_norm.mean()), float(radial_norm.mean()), float(tan_frac)


def _row_cos(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return float(np.mean(num / den))


def _grad_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total_sq = sum(g.pow(2).sum().item() for g in grads if g is not None)
    return float(total_sq ** 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4, help="Force this w (RewardVectorCfg.term_names order), need not sum to 1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--stable_vx_thresh", type=float, default=0.15)
    parser.add_argument("--fall_exclude_window", type=int, default=5)
    parser.add_argument("--n_probe", type=int, default=500, help="Frozen stable probe states to track mu on across the whole update")
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

    obs_cfg = ObservationSpaceCfg()  # noqa: F841
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
    print(f"loaded checkpoint (t={trainer._t}, penalty_k={trainer._penalty_k:.4f})")

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

    values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)
    adv = gae_per_objective(r["rewards"], values_with_final, r["dones"], trainer.cfg.gamma, trainer.cfg.gae_lambda)
    returns = adv + r["values"]
    w_used = r["actor_obs"][:, :, -reward_cfg.dim :]
    adv_t = torch.from_numpy(normalize_per_objective(adv.reshape(T * N, -1))).to(trainer.device)
    w_t = torch.from_numpy(w_used.reshape(T * N, -1).astype(np.float32)).to(trainer.device)
    actor_obs_t = torch.from_numpy(r["actor_obs"].reshape(T * N, -1)).to(trainer.device)
    critic_obs_t = torch.from_numpy(r["critic_obs"].reshape(T * N, -1)).to(trainer.device)
    actions_t = torch.from_numpy(r["actions"].reshape(T * N, -1)).to(trainer.device)
    logp_old_t = torch.from_numpy(r["logp"].reshape(T * N)).to(trainer.device)
    returns_t = torch.from_numpy(returns.reshape(T * N, -1).astype(np.float32)).to(trainer.device)

    # Frozen probe set: stable low-v_x states from the whole rollout.
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
    print(f"\n{n_available} stable probe states available (warmup={args.warmup}, fall_exclude_window={args.fall_exclude_window})")
    if n_available == 0:
        print("no stable samples found -- nothing to probe")
        return
    rng = np.random.default_rng(args.seed)
    n_probe = min(args.n_probe, n_available)
    sel = rng.choice(n_available, size=n_probe, replace=False)
    t_sel, n_sel = t_idx[sel], n_idx[sel]
    probe_actor_obs = torch.from_numpy(r["actor_obs"][t_sel, n_sel]).to(trainer.device)

    # Deep-copy model (+ encoder if present) so the real checkpoint/trainer
    # is never touched; fresh Adam over the same param set moppo.py's
    # __init__ builds, same hyperparameters as trainer.optim.
    model_c = copy.deepcopy(trainer.model)
    encoder_c = copy.deepcopy(trainer.encoder) if trainer.encoder else None
    params = list(model_c.parameters()) + (list(encoder_c.parameters()) if encoder_c else [])
    optim_c = torch.optim.Adam(params, lr=trainer.cfg.lr, weight_decay=trainer.cfg.weight_decay)
    actor_params = list(model_c.actor_body.parameters()) + list(model_c.actor_mean.parameters()) + [model_c.log_std]

    if trainer.encoder:
        extrinsics_normed = trainer.extrinsics_norm.normalize(r["extrinsics"].reshape(T * N, -1), center=True)
        extrinsics_t = torch.from_numpy(extrinsics_normed).float().to(trainer.device)
    else:
        extrinsics_t = None

    with torch.no_grad():
        mu_0 = model_c.raw_mean(probe_actor_obs).clone()
    mu_0_np = mu_0.cpu().numpy()
    mu_prev = mu_0.clone()
    delta_prev = None  # previous minibatch's Delta_mu, for bearing_cos -- carries across epoch boundaries too (same update, same question)
    path_len = np.zeros(mu_0_np.shape[0], dtype=np.float64)
    log_std_max = trainer.model.LOG_STD_MAX  # frozen ceiling for this diagnostic (no anneal state carried in)

    total = T * N
    minibatch_size = total // trainer.cfg.num_minibatches
    torch.manual_seed(args.seed)

    print(f"\nreplaying real update(): {trainer.cfg.epochs_per_update} epochs x {trainer.cfg.num_minibatches} minibatches ({minibatch_size} samples each)\n")
    header = (
        f"{'ep':>3}{'mb':>3}{'||d_mu||':>10}{'tan_frac':>10}{'cos(t,0)':>10}{'cos(dk,dk-1)':>14}{'D_k/P_k':>10}"
        f"{'actor_gnorm':>13}{'mreg_gnorm':>12}{'ratio_mean':>11}{'clip_frac':>10}"
    )
    print(header)

    step = 0
    for epoch in range(trainer.cfg.epochs_per_update):
        perm = torch.randperm(total, device=trainer.device)
        for mb in range(trainer.cfg.num_minibatches):
            idx = perm[mb * minibatch_size : (mb + 1) * minibatch_size]
            actor_obs_mb, critic_obs_mb = actor_obs_t[idx], critic_obs_t[idx]
            actions_mb, logp_old_mb = actions_t[idx], logp_old_t[idx]
            returns_mb, adv_mb, w_mb = returns_t[idx], adv_t[idx], w_t[idx]

            if encoder_c:
                extrinsics_mb = extrinsics_t[idx]
                z_t_live = encoder_c(extrinsics_mb)
                p, lat = trainer.stack.policy_obs_dim, trainer.extrinsics_cfg.adaptation_latent_dim
                actor_obs_live = torch.cat([actor_obs_mb[:, :p], z_t_live, actor_obs_mb[:, p + lat :]], dim=-1)
                cp = trainer.stack.critic_obs_dim
                critic_obs_live = torch.cat([critic_obs_mb[:, :cp], z_t_live, critic_obs_mb[:, cp + lat :]], dim=-1)
            else:
                actor_obs_live, critic_obs_live = actor_obs_mb, critic_obs_mb

            logp_new = model_c.logp(actor_obs_live, actions_mb)
            ratio = torch.exp(logp_new - logp_old_mb)
            clip_loss = d3po_actor_loss(ratio, adv_mb, w_mb, trainer.cfg.clip_eps)

            w_prime_mb = torch.from_numpy(trainer._sample_diversity_w(idx.shape[0])).to(trainer.device)
            actor_obs_prime_mb = actor_obs_live.clone()
            actor_obs_prime_mb[:, -reward_cfg.dim :] = w_prime_mb
            mean_w = model_c.act_inference(actor_obs_live)
            mean_w_prime = model_c.act_inference(actor_obs_prime_mb)
            diversity_loss = diversity_regularizer_loss(
                mean_w, mean_w_prime, w_mb, w_prime_mb, model_c.log_std.exp(), trainer.cfg.diversity_alpha
            )
            policy_loss = clip_loss + trainer.cfg.diversity_lambda * diversity_loss

            values_pred = model_c.value(critic_obs_live)
            value_loss = nn.functional.mse_loss(values_pred, returns_mb)
            entropy_bonus = model_c.entropy(actor_obs_live).mean()
            mean_reg = model_c.raw_mean(actor_obs_live).pow(2).mean()

            loss = (
                policy_loss + 0.5 * value_loss - trainer.cfg.entropy_coef * entropy_bonus
                + trainer.cfg.mean_reg_coef * mean_reg
            )

            # Instrumentation ONLY -- separate autograd.grad calls, never
            # touch model_c.parameters().grad or optim_c's state.
            actor_gnorm = _grad_norm(policy_loss, actor_params)
            mreg_gnorm = _grad_norm(trainer.cfg.mean_reg_coef * mean_reg, actor_params)

            optim_c.zero_grad()
            loss.backward()
            optim_c.step()
            with torch.no_grad():
                model_c.log_std.clamp_(model_c.LOG_STD_MIN, log_std_max)

            with torch.no_grad():
                mu_t = model_c.raw_mean(probe_actor_obs)
                delta_mu = (mu_t - mu_prev).cpu().numpy()
                mu_prev_np = mu_prev.cpu().numpy()
                delta_norm, radial_norm, tan_frac = _radial_tangential_mean(delta_mu, mu_prev_np)
                cos_t0 = torch.sum(mu_t * mu_0, dim=-1) / (
                    torch.norm(mu_t, dim=-1) * torch.norm(mu_0, dim=-1) + 1e-8
                )
                clip_frac = float((torch.abs(ratio - 1.0) > trainer.cfg.clip_eps).float().mean().item())

            bearing_cos = _row_cos(delta_mu, delta_prev) if delta_prev is not None else float("nan")
            path_len += np.linalg.norm(delta_mu, axis=-1)
            mu_t_np = mu_t.cpu().numpy()
            net_disp = np.linalg.norm(mu_t_np - mu_0_np, axis=-1)
            dp_ratio = float(np.median(net_disp / np.maximum(path_len, 1e-8)))

            print(
                f"{epoch:>3}{mb:>3}{delta_norm:>10.5f}{tan_frac:>10.4f}{cos_t0.mean().item():>10.5f}"
                f"{bearing_cos:>14.5f}{dp_ratio:>10.4f}"
                f"{actor_gnorm:>13.5f}{mreg_gnorm:>12.5f}{ratio.mean().item():>11.5f}{clip_frac:>10.4f}"
            )
            mu_prev = mu_t.clone()
            delta_prev = delta_mu
            step += 1

    with torch.no_grad():
        final_cos = torch.sum(mu_prev * mu_0, dim=-1) / (
            torch.norm(mu_prev, dim=-1) * torch.norm(mu_0, dim=-1) + 1e-8
        )
        total_delta = (mu_prev - mu_0).cpu().numpy()
        mu_0_np = mu_0.cpu().numpy()
        total_norm, total_radial, total_tan_frac = _radial_tangential_mean(total_delta, mu_0_np)

    print(f"\n=== full update() summary over {step} minibatch steps ===")
    print(f"cumulative ||mu_final - mu_0|| = {total_norm:.5f}  tan_frac (cumulative) = {total_tan_frac:.4f}  "
          f"cos(mu_final, mu_0) mean = {final_cos.mean().item():.5f}")
    print("Compare cumulative tan_frac to the per-step tan_frac values above: if cumulative << per-step,")
    print("successive minibatches are cancelling each other's tangential drift, not accumulating it.")


if __name__ == "__main__":
    main()
