#!/usr/bin/env python3
"""Radial/tangential decomposition of one real PPO minibatch step's effect
on actor_mean's raw (pre-tanh) output -- the next branch after
objective_perturbation_probe.py / stochastic_diversity_probe.py closed the
reward-weighting, critic-blindness, entropy-collapse, and tanh-squashing
hypotheses in sequence. Remaining hypothesis: actor mean / PPO optimization
geometry -- the policy has real exploration and the critic gives a correct
directional signal, but does the actual parameter UPDATE move mu toward the
value-favored direction (tangential/rotational) or just scale it along the
direction it already has (radial/magnitude)?

Method, no training loop touched, no retrain:
  1. Reuses MOPPOTrainer._collect_rollout() verbatim (same env.step-wrapping
     trick as objective_perturbation_probe.py) to get one real rollout from
     the loaded checkpoint's live policy.
  2. Reproduces update()'s own GAE/normalize_per_objective/minibatch-index
     math exactly (same functions, same call sites) to build ONE real
     minibatch -- the same data an actual training step would consume.
  3. Identifies a frozen PROBE set: "stable" (|v_x| < --stable_vx_thresh)
     states from the same rollout (same warmup/fall-exclusion masking as
     objective_perturbation_probe.py) -- these are NOT necessarily in the
     training minibatch; mu_before/mu_after is measured on this fixed set
     regardless of which states the minibatch happened to sample, so the
     decomposition answers "how does THIS update move the mean at the
     attractor states we care about," not just at the states it trained on.
  4. For each of {full, actor_only, mean_reg_only, entropy_only}: deep-copies
     the model (fresh Adam, same lr/weight_decay as trainer.optim), computes
     ONLY that loss term from the SAME minibatch, does exactly one
     zero_grad/backward/step, and measures raw_mean() on the probe set
     before vs after. actor_body/actor_mean/log_std are the only params any
     of these terms can reach -- critic_body/critic_head get zero gradient
     from all four (value_loss is the only critic-reaching term and is
     therefore skipped entirely: it cannot affect Delta-mu since actor and
     critic share no trunk, see actor_critic.py's ActorCritic.__init__).
  5. Delta_mu = mu_after - mu_before, per probe state, decomposed into the
     component parallel to mu_before (radial / magnitude change) and the
     component orthogonal to it (tangential / direction change).

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py mean-update-decomposition \
        runs/phase1_hipact_dt01_seed0_2026-09-20/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 \
        --steps 200 --sim_dt 0.01
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch

from talon_rl.config import (
    ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg,
)

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
import rl.core.algorithms.moppo as moppo_mod
from rl.core.objectives.losses import d3po_actor_loss, diversity_regularizer_loss, normalize_per_objective
from rl.core.rollout.gae_functional import gae_per_objective

LOSS_MODES = ("full", "actor_only", "mean_reg_only", "entropy_only")


def _radial_tangential(delta_mu: np.ndarray, mu_before: np.ndarray) -> dict:
    """Per-row decomposition of delta_mu against mu_before's direction.
    radial = projection of delta_mu onto mu_before (magnitude change along
    the existing direction); tangential = the orthogonal remainder
    (rotation toward a new direction)."""
    mu_before_sq = np.sum(mu_before * mu_before, axis=-1, keepdims=True)
    mu_before_sq_safe = np.where(mu_before_sq < 1e-12, 1.0, mu_before_sq)
    coeff = np.sum(delta_mu * mu_before, axis=-1, keepdims=True) / mu_before_sq_safe
    radial_vec = coeff * mu_before
    tangential_vec = delta_mu - radial_vec
    delta_norm = np.linalg.norm(delta_mu, axis=-1)
    radial_norm = np.linalg.norm(radial_vec, axis=-1)
    tangential_norm = np.linalg.norm(tangential_vec, axis=-1)
    return {"delta_norm": delta_norm, "radial_norm": radial_norm, "tangential_norm": tangential_norm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4, help="Force this w (RewardVectorCfg.term_names order), need not sum to 1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--stable_vx_thresh", type=float, default=0.15)
    parser.add_argument("--fall_exclude_window", type=int, default=5)
    parser.add_argument("--n_probe", type=int, default=1000, help="Max number of frozen stable (t, lane) states to measure mu_before/mu_after on")
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

    # Exact reproduction of update()'s own GAE/return/normalize math, so
    # the minibatch built below is the same data a real training step
    # would consume -- not a synthetic approximation.
    values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)
    adv = gae_per_objective(r["rewards"], values_with_final, r["dones"], trainer.cfg.gamma, trainer.cfg.gae_lambda)
    w_used = r["actor_obs"][:, :, -reward_cfg.dim :]
    adv_t = torch.from_numpy(normalize_per_objective(adv.reshape(T * N, -1))).to(trainer.device)
    w_t = torch.from_numpy(w_used.reshape(T * N, -1).astype(np.float32)).to(trainer.device)
    actor_obs_t = torch.from_numpy(r["actor_obs"].reshape(T * N, -1)).to(trainer.device)
    actions_t = torch.from_numpy(r["actions"].reshape(T * N, -1)).to(trainer.device)
    logp_old_t = torch.from_numpy(r["logp"].reshape(T * N)).to(trainer.device)

    total = T * N
    minibatch_size = total // trainer.cfg.num_minibatches
    torch.manual_seed(args.seed)
    perm = torch.randperm(total, device=trainer.device)
    idx = perm[:minibatch_size]
    actor_obs_mb = actor_obs_t[idx]
    actions_mb = actions_t[idx]
    logp_old_mb = logp_old_t[idx]
    adv_mb = adv_t[idx]
    w_mb = w_t[idx]
    w_prime_mb = torch.from_numpy(trainer._sample_diversity_w(idx.shape[0])).to(trainer.device)

    # Frozen probe set: stable low-v_x states from the whole rollout, same
    # masking objective_perturbation_probe.py uses -- independent of which
    # states landed in the training minibatch above.
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

    print(f"probing {n_probe} frozen stable states, training minibatch size {minibatch_size} (num_minibatches={trainer.cfg.num_minibatches})")

    results = {}
    for mode in LOSS_MODES:
        model_c = copy.deepcopy(trainer.model)
        optim_c = torch.optim.Adam(model_c.parameters(), lr=trainer.cfg.lr, weight_decay=trainer.cfg.weight_decay)

        with torch.no_grad():
            mu_before = model_c.raw_mean(probe_actor_obs)

        logp_new = model_c.logp(actor_obs_mb, actions_mb)
        ratio = torch.exp(logp_new - logp_old_mb)
        clip_loss = d3po_actor_loss(ratio, adv_mb, w_mb, trainer.cfg.clip_eps)

        actor_obs_prime_mb = actor_obs_mb.clone()
        actor_obs_prime_mb[:, -reward_cfg.dim :] = w_prime_mb
        mean_w = model_c.act_inference(actor_obs_mb)
        mean_w_prime = model_c.act_inference(actor_obs_prime_mb)
        diversity_loss = diversity_regularizer_loss(
            mean_w, mean_w_prime, w_mb, w_prime_mb, model_c.log_std.exp(), trainer.cfg.diversity_alpha
        )
        policy_loss = clip_loss + trainer.cfg.diversity_lambda * diversity_loss

        entropy_bonus = model_c.entropy(actor_obs_mb).mean()
        mean_reg = model_c.raw_mean(actor_obs_mb).pow(2).mean()

        if mode == "full":
            loss = policy_loss - trainer.cfg.entropy_coef * entropy_bonus + trainer.cfg.mean_reg_coef * mean_reg
        elif mode == "actor_only":
            loss = policy_loss
        elif mode == "mean_reg_only":
            loss = trainer.cfg.mean_reg_coef * mean_reg
        elif mode == "entropy_only":
            loss = -trainer.cfg.entropy_coef * entropy_bonus

        optim_c.zero_grad()
        loss.backward()
        optim_c.step()

        with torch.no_grad():
            mu_after = model_c.raw_mean(probe_actor_obs)
            cos_sim = torch.sum(mu_before * mu_after, dim=-1) / (
                torch.norm(mu_before, dim=-1) * torch.norm(mu_after, dim=-1) + 1e-8
            )

        delta_mu = (mu_after - mu_before).cpu().numpy()
        mu_before_np = mu_before.cpu().numpy()
        decomp = _radial_tangential(delta_mu, mu_before_np)
        results[mode] = {**decomp, "cos_sim": cos_sim.cpu().numpy(), "loss_value": float(loss.item())}

    print(f"\n{'mode':<16}{'loss':>10}{'||d_mu||':>12}{'radial':>10}{'tangent':>10}{'tan_frac':>10}{'cos(before,after)':>20}")
    for mode in LOSS_MODES:
        res = results[mode]
        tan_frac = res["tangential_norm"] / np.maximum(res["delta_norm"], 1e-8)
        print(
            f"{mode:<16}{res['loss_value']:>10.5f}{res['delta_norm'].mean():>12.5f}"
            f"{res['radial_norm'].mean():>10.5f}{res['tangential_norm'].mean():>10.5f}"
            f"{np.median(tan_frac):>10.4f}{res['cos_sim'].mean():>20.5f}"
        )
    print("\n(radial = |Delta_mu| projected onto mu_before's own direction -- magnitude change;")
    print(" tangent = orthogonal remainder -- direction change. tan_frac is per-state median, not mean-of-ratio.")
    print(" 'full' combines policy+entropy+mean_reg exactly as moppo.py's update() does; value_loss is omitted")
    print(" everywhere since actor and critic share no trunk (actor_critic.py) -- it cannot reach these params.)")


if __name__ == "__main__":
    main()
