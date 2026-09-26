#!/usr/bin/env python3
"""Minibatch-composition trace -- follow-up to compare_gradients.py's Test 8
(2026-09-20 branch). Layers 1/2 ruled out state-distribution shift and
comparable-state valuation shift as explanations for cross-update bearing
diffusion. Test 4/5 found impact/balance's OWN gradient direction wanders
far more update-to-update in seed0 than seed1, with the E-I-B triangle's
intra-update geometry itself seed-invariant. Test 8 ruled out simple
rollout-distribution drift (forward OR reverse) as the driver. This script
asks the question Test 8 pointed to: does the instability originate WITHIN
a single update() call, across its epochs_per_update x num_minibatches
minibatch/epoch structure, rather than between rollouts at all? Three
questions (this branch's "Test 10"):
  A. minibatch-local -- does cos(g_mb, g_mb-1) wander wildly WITHIN an
     update (minibatch composition/order as candidate)?
  B. epoch-local -- is each epoch internally stable but the direction shifts
     at epoch BOUNDARIES (reshuffling/repeated-data interaction)?
  C. is the final Delta_theta dominated by a few outlier minibatches (large
     norm, divergent direction), which would explain why the FIRST
     minibatch (full_update_trace.py / compare_gradients.py's g_total) is a
     poor predictor of the whole update's net movement (cos ~ -0.11, both
     seeds)?
Then: does this WITHIN-update minibatch behavior differ between seed0 and
seed1 the same way Test 5's update-level E/I/B stability did?

Runs the REAL trainer.update() --num_updates times (same real-training
technique as multi_update_trace.py -- mutates the in-memory model, never
re-saves the checkpoint file). Adds THREE taps on top of that script's
existing d3po_actor_loss/diversity_regularizer_loss wraps:
  1. torch.randperm -- tightly scoped around EACH trainer.update() call
     only (patched immediately before, restored in finally immediately
     after, real return value passed through unmodified, nothing else
     called from inside the wrapper) -- the ONLY way to recover which
     (t, n) rollout samples landed in which minibatch, since `idx` is a
     local variable inside update() never otherwise exposed. Audited after
     every update: exactly epochs_per_update calls captured, and every
     captured permutation is a genuine permutation of arange(T*N).
  2. d3po_actor_loss / diversity_regularizer_loss -- extended from
     multi_update_trace.py's epoch0/mb0-only capture to EVERY minibatch of
     EVERY epoch, computing the same exact per-objective gradient
     decomposition (see that script's own docstring for why it's exact,
     not approximate).
  3. Per-minibatch state lookup via idx // N_envs, idx % N_envs into the
     SAME Layer-1 raw descriptor arrays (vx/height/vz/contact_count/
     term_base_contact/term_time_out) already used elsewhere in this
     branch -- never actor_obs (normalized, confounds with running-stat
     drift).

Saves per update, per scalar metric, a (epochs_per_update, num_minibatches)
grid -- epoch and minibatch stay separate axes deliberately (never
flattened together), so within-epoch vs across-epoch instability can be
told apart downstream. Full gradient VECTORS are held only transiently
(needed for cos_to_final_delta_theta, computed after the whole update
finishes) and discarded before saving -- only scalars (norms, cosines,
advantage stats, clip fraction, state means) hit disk, plus the raw
minibatch membership indices (cheap relative to gradient vectors, kept for
audit/reconstruction on this smoke run; --no_save_indices to drop them).

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py minibatch-composition-trace \\
        runs/phase1_hipact_dt01_seed0_2026-09-20/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 \\
        --num_updates 10 --sim_dt 0.01 --seed 0 \\
        --save_descriptors /tmp/.../seed0_minibatch_10update.npz
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
import rl.core.objectives.losses as losses_mod
from rl.core.rollout.gae_functional import gae_per_objective

GRADIENT_NAMES = ("progress", "efficiency", "impact", "balance", "diversity", "total")
OBJECTIVES = ("progress", "efficiency", "impact", "balance")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def _qstats(arr: np.ndarray) -> tuple[float, float, float, float, float]:
    """mean, std, p10, p50, p90 -- mean/std alone can hide a skewed or heavy-tailed
    distribution, which is exactly what's in question for GAE advantage/return."""
    return (
        float(arr.mean()), float(arr.std()),
        float(np.percentile(arr, 10)), float(np.percentile(arr, 50)), float(np.percentile(arr, 90)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4, help="Force this w (RewardVectorCfg.term_names order), need not sum to 1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--num_updates", type=int, default=10)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--sim_dt", type=float, default=0.02)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_encoder", action="store_true")
    parser.add_argument("--save_descriptors", default=None, help="Path to save per-update-per-minibatch (.npz) -- omit to skip saving")
    parser.add_argument("--no_save_indices", action="store_true", help="Drop raw minibatch-membership indices from the saved file (keep only scalar summaries)")
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401
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
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t}, penalty_k={trainer._penalty_k:.4f})")

    w_single = (np.array(args.w, dtype=np.float32) / sum(args.w)).astype(np.float32)
    w_forced = np.tile(w_single, (args.num_envs, 1))
    trainer.w = w_forced.copy()
    env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    orig_sample = moppo_mod.sample_preference_vector

    def _fixed_sample(rng, reward_cfg_, pref_cfg_, n, step=0):
        return np.tile(w_single, (n, 1))

    moppo_mod.sample_preference_vector = _fixed_sample

    # Layer-1 raw state descriptors, same fields/technique as
    # multi_update_trace.py -- pulled from `transition` inside env.step,
    # never from actor_obs (normalized, would confound with running-stat
    # drift). Cleared at the top of every update, so by the time the
    # minibatch loop for THIS update runs, this is exactly that update's
    # own (T, N) rollout.
    CONTACT_THRESHOLD_N = 1.0
    descriptor_buf: dict[str, list[np.ndarray]] = {
        "vx": [], "height": [], "vz": [], "pitch": [], "pitch_rate": [], "contact_count": [],
        "term_base_contact": [], "term_time_out": [],
    }
    orig_step = env.step

    def _wrapped_step(action):
        transition, done = orig_step(action)
        descriptor_buf["vx"].append(transition["v_actual"][:, 0].copy())
        descriptor_buf["height"].append(transition["height"].copy())
        descriptor_buf["vz"].append(transition["v_z"].copy())
        descriptor_buf["pitch"].append(transition["roll_pitch"][:, 1].copy())
        descriptor_buf["pitch_rate"].append(transition["roll_pitch_rate"][:, 1].copy())
        descriptor_buf["contact_count"].append((transition["foot_contact_force"] > CONTACT_THRESHOLD_N).sum(axis=-1).astype(np.float32))
        descriptor_buf["term_base_contact"].append(np.asarray(transition.get("term_base_contact", np.zeros_like(done)), dtype=np.float32).copy())
        descriptor_buf["term_time_out"].append(np.asarray(transition.get("term_time_out", np.zeros_like(done)), dtype=np.float32).copy())
        return transition, done

    env.step = _wrapped_step

    # Test 12B (2026-09-20, advantage-construction distribution trace) --
    # wraps trainer._collect_rollout ADDITIVELY (same technique as
    # multi_update_trace.py's Layer 2: calls the real one, copies its
    # return, hands it back unchanged) to recover r["rewards"], r["values"],
    # r["dones"] (== terminal_fall, real termination only), r["final_value"]
    # -- raw ingredients update() itself uses to compute the SAME raw
    # advantage/return via gae_per_objective() right after this call
    # returns, as a LOCAL variable it never exposes. Recomputed here with
    # the identical pure function + gamma/gae_lambda, so there is no
    # implementation discrepancy with what the real update() actually used
    # for this exact minibatch loop (see compare_valuation.py's docstring
    # for the same argument, applied there post-hoc; here it's computed
    # inline so the minibatch wrappers below can index straight into it via
    # the SAME idx recovered from the torch.randperm tap).
    rollout_capture: dict[str, np.ndarray] = {}
    orig_collect_rollout = trainer._collect_rollout

    def _wrapped_collect_rollout():
        r = orig_collect_rollout()
        values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)
        raw_adv = gae_per_objective(r["rewards"], values_with_final, r["dones"], trainer.cfg.gamma, trainer.cfg.gae_lambda)
        raw_return = raw_adv + r["values"]
        T_r, N_r, K_r = r["rewards"].shape
        rollout_capture["reward_flat"] = r["rewards"].reshape(T_r * N_r, K_r).copy()
        rollout_capture["raw_adv_flat"] = raw_adv.reshape(T_r * N_r, K_r).astype(np.float32)
        rollout_capture["return_flat"] = raw_return.reshape(T_r * N_r, K_r).astype(np.float32)
        rollout_capture["terminal_fall"] = r["dones"].copy()  # (T, N) -- for the termination-invariant audit below
        return r

    trainer._collect_rollout = _wrapped_collect_rollout

    actor_params = list(trainer.model.actor_body.parameters()) + list(trainer.model.actor_mean.parameters()) + [trainer.model.log_std]

    def _flatten(tensors) -> np.ndarray:
        return torch.cat([t.reshape(-1) for t in tensors]).detach().cpu().numpy().astype(np.float32)

    def _flatten_grad(loss: torch.Tensor) -> np.ndarray:
        grads = torch.autograd.grad(loss, actor_params, retain_graph=True, allow_unused=True)
        return _flatten([g if g is not None else torch.zeros_like(p) for g, p in zip(grads, actor_params)])

    T_steps, N_envs = args.steps, args.num_envs
    total = T_steps * N_envs
    num_minibatches = trainer.cfg.num_minibatches
    epochs_per_update = trainer.cfg.epochs_per_update
    mb_size = total // num_minibatches

    # Test 13-0 (2026-09-20, shuffle-advantage counterfactual): pure numpy
    # RNG, NEVER torch.randperm -- the outer torch.randperm tap around
    # trainer.update() (below) is actively capturing every call for the
    # idx-reconstruction audit; calling torch.randperm again in here would
    # get captured too and corrupt that audit's epoch/minibatch accounting.
    N_SHUFFLES = 3
    shuffle_rng = np.random.default_rng(args.seed + 10_000)

    # Test 13A/13B (2026-09-20, per-sample contribution): closed-form,
    # zero-extra-autograd-cost per-sample scalar s_i = w_i * A_i * mask_i,
    # where mask_i is PPO-clip's known zero-gradient indicator
    # (d(min(ratio*A, clip(ratio)*A))/d(ratio) = A when NOT((ratio>1+eps &
    # A>0) or (ratio<1-eps & A<0)), else exactly 0). This is the EXACT
    # advantage-side contribution each sample carries into the gradient
    # after clip-gating -- it deliberately omits the state-dependent score-
    # function factor d(ratio_i)/d(theta), which differs per sample and is
    # too expensive to get exactly for every sample (would need B backward
    # passes per minibatch). Validated below on a random subsample against
    # the TRUE per-sample gradient norm, update 1 only.
    TOPK_FRACS = (0.01, 0.05, 0.10)
    VALIDATION_SAMPLES = 16
    VALIDATION_OBJECTIVES = ("impact", "balance")
    validation_log: list[dict] = []
    current_update = [0]  # set at top of the k-loop below

    # Experiment 3A (2026-09-20, "advantage-side coefficient clipping using
    # the validated proxy" -- explicitly NOT "per-sample gradient clipping",
    # see the locked methodology note). c_j (one per objective) is computed
    # ONCE from update 1's pooled |s_i,j| across all its minibatches, frozen
    # for updates 2-10 -- never recomputed from later updates (that would
    # let the intervention adapt to the very distribution it's meant to be
    # tested against). Update 1 itself has no clipped counterfactual (chose
    # this over an unfrozen bootstrap threshold -- matches
    # threshold_source_update=1, threshold_frozen=True exactly).
    THRESHOLD_PERCENTILES = (99, 97.5, 95)
    threshold_calibration: dict[str, list[np.ndarray]] = {o: [] for o in OBJECTIVES}
    frozen_thresholds: dict[str, dict[float, float]] = {}  # obj -> {pct: c_j}, set once after update 1

    # Experiment 3A.1 (2026-09-20, temporal counterfactual -- Axis 2): does
    # the CLIPPED gradient's cross-update temporal cosine improve over the
    # REAL gradient's, at the SAME first-minibatch (epoch0/mb0) comparison
    # point Test 4/5/12A already established? Persists across update
    # boundaries (unlike prev_vec/prev_vec_unclip, which reset every
    # update for WITHIN-update comparison) -- only written/read at
    # epoch0/mb0, never cleared mid-run.
    prev_real_first_mb: dict[str, np.ndarray] = {}
    prev_clip_first_mb: dict[float, dict[str, np.ndarray]] = {pct: {} for pct in THRESHOLD_PERCENTILES}
    temporal_cos_log: list[dict] = []  # one row per update (from update 2 real-only, update 3 onward real+clip)
    CLIP3A_PER_OBJ_STATS = ("cos", "norm_ratio", "frac_clipped", "mean_clip_scale",
                             "top1pct_concentration", "top5pct_concentration",
                             "frac_positive_top1pct", "frac_positive_top5pct")
    CLIP3A_TOTAL_STATS = ("cos", "norm_ratio")

    step_idx = [0]  # 0..epochs_per_update*num_minibatches-1, reset each update
    captured_perms: list[torch.Tensor] = []  # reset each update, populated ONLY around trainer.update()
    prev_vec: dict[str, np.ndarray] = {}  # previous minibatch's gradient vectors, reset each update
    prev_vec_unclip: dict[str, np.ndarray] = {}  # Test 12A (2026-09-20) -- same, for the UNCLIPPED counterparts
    minibatch_records: list[dict] = []  # reset each update
    pending = [None]  # bridges d3po's per-objective work to diversity's g_total/g_diversity completion, same minibatch

    def _lookup_state(field: str, t_idx: np.ndarray, n_idx: np.ndarray) -> np.ndarray:
        arr = np.stack(descriptor_buf[field])  # (T, N) -- full for this update by the time any minibatch runs
        return arr[t_idx, n_idx]

    def _segment_position() -> np.ndarray:
        """(T, N) steps-since-last-episode-end, BEFORE this step (0 = first
        step of a fresh episode) -- done = terminal_fall OR timeout, both
        restart the lane. Recomputed fresh each update from this update's
        own rollout (cheap, T=~24/N=~256)."""
        fall = rollout_capture["terminal_fall"].astype(bool)
        timeout = np.stack(descriptor_buf["term_time_out"]).astype(bool)
        done = fall | timeout
        T_, N_ = done.shape
        pos = np.zeros((T_, N_), dtype=np.float32)
        counter = np.zeros(N_, dtype=np.float32)
        for t in range(T_):
            pos[t] = counter
            counter = np.where(done[t], 0.0, counter + 1.0)
        return pos

    orig_d3po = losses_mod.d3po_actor_loss

    def _wrapped_d3po(ratio, adv, w, clip_eps):
        with torch.no_grad():
            clip_frac = float((torch.abs(ratio - 1.0) > clip_eps).float().mean().item())
        clip_loss = orig_d3po(ratio, adv, w, clip_eps)

        epoch = step_idx[0] // num_minibatches
        mb = step_idx[0] % num_minibatches
        idx = captured_perms[epoch][mb * mb_size : (mb + 1) * mb_size].cpu().numpy()
        t_idx, n_idx = idx // N_envs, idx % N_envs

        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        unclipped_objective = ratio[:, None] * adv  # the raw PPO surrogate, before the min(unclip, clip) that makes it the "clip" candidate
        per_objective = torch.min(unclipped_objective, clipped_ratio[:, None] * adv)  # what the real loss actually uses (clip_loss above)
        obj_vecs: dict[str, np.ndarray] = {}
        obj_vecs_unclip: dict[str, np.ndarray] = {}
        adv_stats: dict[str, tuple[float, float, float]] = {}
        for i, name in enumerate(reward_cfg.term_names):
            loss_i = -(w[:, i] * per_objective[:, i]).mean()
            obj_vecs[name] = _flatten_grad(loss_i)
            loss_i_unclip = -(w[:, i] * unclipped_objective[:, i]).mean()
            obj_vecs_unclip[name] = _flatten_grad(loss_i_unclip)
            adv_i = adv[:, i].detach().cpu().numpy()
            adv_stats[name] = (float(adv_i.mean()), float(adv_i.std()), float((adv_i > 0).mean()))
        clip_loss_unclip = -(w * unclipped_objective).sum(-1).mean()  # the un-clipped counterpart to clip_loss itself, for g_total's unclip version

        # Test 13-0 (2026-09-20, shuffle-advantage counterfactual): each
        # shuffle applies ONE joint permutation across the (B, K) advantage
        # matrix -- same perm for every objective column, since a sample's
        # advantage vector is one joint quantity tied to that sample's
        # state/action, and `ratio` (the policy score-function contribution)
        # is shared across objectives for a given sample too. Tests whether
        # THIS objective's gradient depends on which specific sample's
        # policy-gradient contribution its advantage got paired with, or
        # whether the aggregate is robust to that pairing (marginal
        # advantage distribution alone would then be sufficient to explain
        # the gradient, matching what Test 12B already measured).
        shuffle_cos: dict[str, list[float]] = {name: [] for name in (*reward_cfg.term_names, "total_clip")}
        for _ in range(N_SHUFFLES):
            perm_np = shuffle_rng.permutation(adv.shape[0])
            perm_t = torch.from_numpy(perm_np).to(adv.device)
            adv_shuffled = adv[perm_t]  # (B, K) -- joint shuffle, ratio/w untouched (still tied to the REAL sample)
            per_objective_shuffled = torch.min(ratio[:, None] * adv_shuffled, clipped_ratio[:, None] * adv_shuffled)
            for i, name in enumerate(reward_cfg.term_names):
                loss_i_shuffled = -(w[:, i] * per_objective_shuffled[:, i]).mean()
                g_shuffled = _flatten_grad(loss_i_shuffled)
                shuffle_cos[name].append(_cos(obj_vecs[name], g_shuffled))
            clip_loss_shuffled = -(w * per_objective_shuffled).sum(-1).mean()
            g_total_clip_shuffled = _flatten_grad(clip_loss_shuffled)
            g_total_clip_real = sum(obj_vecs.values())  # exact by the same linearity identity Test 4/5 already verified -- no extra grad call needed
            shuffle_cos["total_clip"].append(_cos(g_total_clip_real, g_total_clip_shuffled))

        with torch.no_grad():
            sample_contribution: dict[str, np.ndarray] = {}
            sample_contribution_t: dict[str, torch.Tensor] = {}  # torch (detached) version, reused by Experiment 3A's clipping below
            for i, name in enumerate(reward_cfg.term_names):
                a_i = adv[:, i]
                upper_clip_active = (ratio > (1 + clip_eps)) & (a_i > 0)
                lower_clip_active = (ratio < (1 - clip_eps)) & (a_i < 0)
                mask = ~(upper_clip_active | lower_clip_active)
                s_t = w[:, i] * a_i * mask.float()
                sample_contribution_t[name] = s_t
                sample_contribution[name] = s_t.cpu().numpy()

        contribution_stats: dict[str, dict[str, float]] = {}
        # Test 13B (2026-09-20, follow-up to 13A): saves RAW top-1%/top-5%
        # sample identity (flat T*N indices, mapped through this
        # minibatch's own `idx`) and their sign split -- deliberately NOT
        # binned/Jaccard'd/persistence-tracked here. Binning needs GLOBAL
        # quantile edges over the whole saved run (same principle Layer 1/2
        # already established), which isn't available mid-collection;
        # region-occupancy/persistence/entropy belongs in a post-hoc
        # analysis pass over these indices + the raw (T,N) state arrays
        # saved once per update below, not baked into live instrumentation.
        topk_indices: dict[str, dict[int, np.ndarray]] = {}
        topk_signs: dict[str, dict[int, np.ndarray]] = {}  # 2026-09-20 asymmetry analysis -- sign(s_i) for the SAME top-K samples as topk_indices, real/unclipped only, zero extra cost
        for name, s in sample_contribution.items():
            abs_s = np.abs(s)
            total = abs_s.sum()
            order = np.argsort(-abs_s)
            sorted_abs = abs_s[order]
            cum = np.cumsum(sorted_abs)
            stats = {"mean": float(s.mean()), "std": float(s.std()), "frac_masked_zero": float((abs_s < 1e-12).mean())}
            topk_indices[name] = {}
            topk_signs[name] = {}
            for frac in TOPK_FRACS:
                k_n = max(1, int(round(frac * len(s))))
                stats[f"top{int(frac*100)}pct_concentration"] = float(cum[k_n - 1] / total) if total > 1e-12 else float("nan")
                if frac in (0.01, 0.05):
                    top_positions = order[:k_n]
                    stats[f"frac_positive_top{int(frac*100)}pct"] = float((s[top_positions] > 0).mean())
                    topk_indices[name][int(frac * 100)] = idx[top_positions]  # flat T*N indices, NOT within-minibatch positions
                    topk_signs[name][int(frac * 100)] = np.sign(s[top_positions]).astype(np.int8)  # same order as topk_indices, +/-1 (0 only if s exactly 0)
            contribution_stats[name] = stats

        # Experiment 3A calibration (update 1 ONLY): accumulate |s_i,j| from
        # every minibatch of update 1, pooled per objective -- thresholds
        # are computed from this pool AFTER update 1 finishes (see the
        # k-loop below), then frozen for updates 2-10.
        if current_update[0] == 1:
            for name in OBJECTIVES:
                threshold_calibration[name].append(np.abs(sample_contribution[name]))

        # Experiment 3A clipping counterfactual (updates 2-10 only, using
        # update 1's frozen thresholds): r_i,j = min(1, c_j/|s_i,j|), r=1
        # when s_i,j==0. Scales each sample's LOSS TERM by r_i,j.detach()
        # before the mean -- since d(w_j*per_objective_j)/d(ratio) == s_i,j
        # in the active (unclipped-by-PPO) region, this makes the resulting
        # gradient's per-sample coefficient exactly r_i,j * s_i,j, i.e. the
        # locked "advantage-side coefficient clipping" definition -- NOT a
        # literal per-sample gradient-norm clip. Same states/actions/
        # old-logp/policy/ratio/PPO-clip/entropy/diversity/weights as the
        # REAL loss -- only this counterfactual's per-sample weight changes,
        # and it is NEVER used for the real backward()/optimizer.step().
        clip_results: dict[str, dict] = {}
        clip_vecs_first_mb: dict[float, dict[str, np.ndarray]] = {pct: {} for pct in THRESHOLD_PERCENTILES}  # Experiment 3A.1 -- only populated at epoch0/mb0 (step_idx[0]==0)
        is_first_mb = step_idx[0] == 0
        if current_update[0] >= 2 and frozen_thresholds:
            clipped_term_by_pct: dict[float, dict[str, torch.Tensor]] = {pct: {} for pct in THRESHOLD_PERCENTILES}
            for pct in THRESHOLD_PERCENTILES:
                for i, name in enumerate(reward_cfg.term_names):
                    s_t = sample_contribution_t[name]
                    c_j = frozen_thresholds[name][pct]
                    with torch.no_grad():
                        abs_s = s_t.abs()
                        r = torch.where(abs_s > 1e-12, torch.clamp(c_j / abs_s, max=1.0), torch.ones_like(s_t))
                    clipped_term = r * (w[:, i] * per_objective[:, i])
                    clipped_term_by_pct[pct][name] = clipped_term
                    loss_j_clipped = -clipped_term.mean()
                    g_j_clipped = _flatten_grad(loss_j_clipped)
                    if is_first_mb:
                        clip_vecs_first_mb[pct][name] = g_j_clipped

                    with torch.no_grad():
                        s_clipped_np = (r * s_t).cpu().numpy()
                    abs_sc = np.abs(s_clipped_np)
                    total_sc = abs_sc.sum()
                    order_sc = np.argsort(-abs_sc)
                    cum_sc = np.cumsum(abs_sc[order_sc])
                    entry = {
                        "cos": _cos(obj_vecs[name], g_j_clipped),
                        "norm_ratio": float(np.linalg.norm(g_j_clipped) / (np.linalg.norm(obj_vecs[name]) + 1e-12)),
                        "frac_clipped": float((r < 0.999).float().mean().item()),
                        "mean_clip_scale": float(r.mean().item()),
                    }
                    for frac in (0.01, 0.05):
                        k_n = max(1, int(round(frac * len(s_clipped_np))))
                        entry[f"top{int(frac*100)}pct_concentration"] = float(cum_sc[k_n - 1] / total_sc) if total_sc > 1e-12 else float("nan")
                        entry[f"frac_positive_top{int(frac*100)}pct"] = float((s_clipped_np[order_sc[:k_n]] > 0).mean())
                    clip_results[f"{pct}_{name}"] = entry

                # Total: built from the four independently-clipped objective
                # terms SUMMED first, then differentiated once -- not by
                # clipping an already-combined total coefficient (locked
                # methodological guardrail).
                clip_loss_total = -sum(clipped_term_by_pct[pct].values()).mean()
                g_total_clipped = _flatten_grad(clip_loss_total)
                g_total_real = sum(obj_vecs.values())
                clip_results[f"{pct}_total"] = {
                    "cos": _cos(g_total_real, g_total_clipped),
                    "norm_ratio": float(np.linalg.norm(g_total_clipped) / (np.linalg.norm(g_total_real) + 1e-12)),
                }
                if is_first_mb:
                    clip_vecs_first_mb[pct]["total"] = g_total_clipped

        # Test 13 validation (update 1 ONLY -- once is enough to trust s_i's
        # ranking for the rest of the run, not worth repeating every
        # minibatch/update): TRUE per-sample gradient norm via individual
        # backward passes on a small random subsample, compared against s_i
        # for the SAME samples. shuffle_rng (numpy), never torch.randperm.
        if current_update[0] == 1:
            for name in VALIDATION_OBJECTIVES:
                i = reward_cfg.term_names.index(name)
                sample_idx = shuffle_rng.choice(adv.shape[0], size=min(VALIDATION_SAMPLES, adv.shape[0]), replace=False)
                for j in sample_idx:
                    loss_j = -(w[j, i] * per_objective[j, i])
                    grads = torch.autograd.grad(loss_j, actor_params, retain_graph=True, allow_unused=True)
                    true_norm = float(sum(g.pow(2).sum().item() for g in grads if g is not None) ** 0.5)
                    validation_log.append({
                        "objective": name, "s_i": float(sample_contribution[name][j]), "true_norm": true_norm,
                        "epoch": step_idx[0] // num_minibatches, "mb": step_idx[0] % num_minibatches,
                    })

        pending[0] = {
            "shuffle_cos": shuffle_cos, "contribution_stats": contribution_stats, "topk_indices": topk_indices, "topk_signs": topk_signs,
            "clip_results": clip_results, "clip_vecs_first_mb": clip_vecs_first_mb, "is_first_mb": is_first_mb,
            "clip_loss": clip_loss, "clip_loss_unclip": clip_loss_unclip,
            "obj_vecs": obj_vecs, "obj_vecs_unclip": obj_vecs_unclip, "adv_stats": adv_stats, "clip_frac": clip_frac,
            "t_idx": t_idx, "n_idx": n_idx, "epoch": epoch, "mb": mb, "idx": idx,
            "adv_np": adv.detach().cpu().numpy(),  # (mb_size, K) -- NORMALIZED advantage, exactly what the real loss uses, for Test 12B's normalized_adv quantiles
        }
        return clip_loss

    moppo_mod.d3po_actor_loss = _wrapped_d3po

    orig_diversity = losses_mod.diversity_regularizer_loss

    def _wrapped_diversity(mean_w, mean_w_prime, w, w_prime, std, alpha):
        diversity_loss = orig_diversity(mean_w, mean_w_prime, w, w_prime, std, alpha)
        p = pending[0]
        scaled_diversity = trainer.cfg.diversity_lambda * diversity_loss
        g_diversity = _flatten_grad(scaled_diversity)
        policy_loss_reconstructed = p["clip_loss"] + scaled_diversity
        g_total = _flatten_grad(policy_loss_reconstructed)
        # Unclipped counterpart -- diversity_loss is untouched by ratio-clipping
        # (it's a separate KL-style term, not a function of ratio at all), so
        # the SAME g_diversity applies to both the clip and unclip totals --
        # only the clip_loss component differs between them.
        policy_loss_unclip_reconstructed = p["clip_loss_unclip"] + scaled_diversity
        g_total_unclip = _flatten_grad(policy_loss_unclip_reconstructed)

        vecs = {**p["obj_vecs"], "diversity": g_diversity, "total": g_total}

        record: dict = {"epoch": p["epoch"], "mb": p["mb"], "clip_frac": p["clip_frac"], "idx": p["idx"]}
        # Test 11 (2026-09-20, clipping-induced rotation): for each objective
        # (+ total) compares the ACTUAL clipped surrogate's gradient against
        # its unclipped counterpart -- rotation = 1-cos(unclip,clip) isolates
        # DIRECTION change from clipping (high clip_fraction alone doesn't
        # imply rotation; clipping can rescale a gradient's magnitude while
        # barely turning it -- that's exactly what norm_ratio is for).
        rotation_targets = {**p["obj_vecs_unclip"], "total": g_total_unclip}
        rotation_targets_clip = {**p["obj_vecs"], "total": g_total}
        for name in rotation_targets:
            u, c = rotation_targets[name], rotation_targets_clip[name]
            record[f"rotation_{name}"] = 1.0 - _cos(u, c)
            record[f"norm_ratio_{name}"] = float(np.linalg.norm(c) / (np.linalg.norm(u) + 1e-12))
            record[f"grad_norm_unclip_{name}"] = float(np.linalg.norm(u))
            # Test 12A (2026-09-20, follow-up to Test 11): does the seed0/
            # seed1 temporal-stability gap Test 10 found in the CLIPPED
            # gradient already exist in the gradient BEFORE clipping ever
            # touches it? cos_prev_unclip is the exact unclipped analogue of
            # the existing cos_prev (clipped) below.
            record[f"cos_prev_unclip_{name}"] = _cos(u, prev_vec_unclip[name]) if name in prev_vec_unclip else float("nan")
        prev_vec_unclip.clear()
        prev_vec_unclip.update(rotation_targets)
        record["mean_vx"] = float(_lookup_state("vx", p["t_idx"], p["n_idx"]).mean())
        record["mean_height"] = float(_lookup_state("height", p["t_idx"], p["n_idx"]).mean())
        record["mean_abs_vz"] = float(np.abs(_lookup_state("vz", p["t_idx"], p["n_idx"])).mean())
        record["mean_contact"] = float(_lookup_state("contact_count", p["t_idx"], p["n_idx"]).mean())
        record["fall_frac"] = float(_lookup_state("term_base_contact", p["t_idx"], p["n_idx"]).mean())
        record["timeout_frac"] = float(_lookup_state("term_time_out", p["t_idx"], p["n_idx"]).mean())
        record["segment_position"] = float(_segment_position()[p["t_idx"], p["n_idx"]].mean())
        for name, stats in p["contribution_stats"].items():
            for stat_name, val in stats.items():
                record[f"contrib_{stat_name}_{name}"] = val
        for name, by_frac in p["topk_indices"].items():
            for pct, flat_idx in by_frac.items():
                record[f"top{pct}pct_idx_{name}"] = flat_idx
        for name, by_frac in p["topk_signs"].items():
            for pct, signs in by_frac.items():
                record[f"top{pct}pct_sign_{name}"] = signs
        # Experiment 3A fields default to NaN (update 1 has no clip_results yet --
        # thresholds aren't calibrated until update 1 finishes) so every minibatch
        # record has the same key set for the grid-building step below.
        for pct in THRESHOLD_PERCENTILES:
            for name in OBJECTIVES:
                for stat_name in CLIP3A_PER_OBJ_STATS:
                    record[f"clip3a_{stat_name}_{pct}_{name}"] = float("nan")
            for stat_name in CLIP3A_TOTAL_STATS:
                record[f"clip3a_{stat_name}_{pct}_total"] = float("nan")
        for key, entry in p["clip_results"].items():
            for stat_name, val in entry.items():
                record[f"clip3a_{stat_name}_{key}"] = val
        for name, cos_list in p["shuffle_cos"].items():
            record[f"shuffle_cos_mean_{name}"] = float(np.mean(cos_list))
            record[f"shuffle_cos_std_{name}"] = float(np.std(cos_list))
        for name, (m, s, f) in p["adv_stats"].items():
            record[f"adv_mean_{name}"] = m
            record[f"adv_std_{name}"] = s
            record[f"adv_posfrac_{name}"] = f

        # Test 12B (2026-09-20, advantage-construction distribution trace):
        # per-minibatch, per-objective distribution stats at EVERY layer --
        # raw reward, raw return, raw (pre-normalize_per_objective) GAE
        # advantage, and normalized advantage (the last sourced from `adv`
        # itself, exactly what the real loss consumes, not recomputed).
        # Pure data collection only -- no temporal/cosine/SMD analysis baked
        # in here, that's a separate downstream pass over the saved grids.
        idx = p["idx"]
        for i, name in enumerate(reward_cfg.term_names):
            for layer, flat in (("reward", "reward_flat"), ("return", "return_flat"), ("raw_adv", "raw_adv_flat")):
                m, s, p10, p50, p90 = _qstats(rollout_capture[flat][idx, i])
                record[f"{layer}_mean_{name}"] = m
                record[f"{layer}_std_{name}"] = s
                record[f"{layer}_p10_{name}"] = p10
                record[f"{layer}_p50_{name}"] = p50
                record[f"{layer}_p90_{name}"] = p90
            m, s, p10, p50, p90 = _qstats(p["adv_np"][:, i])
            record[f"norm_adv_p10_{name}"] = p10
            record[f"norm_adv_p50_{name}"] = p50
            record[f"norm_adv_p90_{name}"] = p90
        for gname in GRADIENT_NAMES:
            record[f"grad_norm_{gname}"] = float(np.linalg.norm(vecs[gname]))
            record[f"cos_prev_{gname}"] = _cos(vecs[gname], prev_vec[gname]) if gname in prev_vec else float("nan")
        record["_vectors"] = vecs  # transient -- deleted after cos_to_final_delta below, never saved

        # Experiment 3A.1 (Axis 2, temporal counterfactual): only at
        # epoch0/mb0, compare THIS update's first-minibatch gradient
        # (real and, from update >=3, clipped) against the PREVIOUS
        # update's first-minibatch gradient at the same position -- the
        # exact cross-update convention Test 4/5/12A already used.
        # Defaults NaN (most (epoch,mb) positions never populate this).
        for name in (*OBJECTIVES, "total"):
            record[f"temporal_cos_real_{name}"] = float("nan")
        for pct in THRESHOLD_PERCENTILES:
            for name in (*OBJECTIVES, "total"):
                record[f"temporal_cos_clip_{pct}_{name}"] = float("nan")
        if p["is_first_mb"]:
            temporal_row = {"update": current_update[0]}
            for name in (*OBJECTIVES, "total"):
                if name in prev_real_first_mb:
                    c = _cos(prev_real_first_mb[name], vecs[name])
                    record[f"temporal_cos_real_{name}"] = c
                    temporal_row[f"real_{name}"] = c
                prev_real_first_mb[name] = vecs[name]
            for pct in THRESHOLD_PERCENTILES:
                for name in (*OBJECTIVES, "total"):
                    if name in p["clip_vecs_first_mb"][pct]:
                        if name in prev_clip_first_mb[pct]:
                            c = _cos(prev_clip_first_mb[pct][name], p["clip_vecs_first_mb"][pct][name])
                            record[f"temporal_cos_clip_{pct}_{name}"] = c
                            temporal_row[f"clip_{pct}_{name}"] = c
                        prev_clip_first_mb[pct][name] = p["clip_vecs_first_mb"][pct][name]
            temporal_cos_log.append(temporal_row)

        prev_vec.clear()
        prev_vec.update(vecs)
        minibatch_records.append(record)
        step_idx[0] += 1
        return diversity_loss

    moppo_mod.diversity_regularizer_loss = _wrapped_diversity

    orig_randperm = torch.randperm

    print(f"\nrunning {args.num_updates} real trainer.update() calls "
          f"({epochs_per_update} epochs x {num_minibatches} minibatches, {mb_size} samples each)\n")

    per_update_grids: dict[int, dict[str, np.ndarray]] = {}
    ROTATION_NAMES = (*OBJECTIVES, "total")  # diversity excluded -- untouched by ratio-clipping, rotation trivially 0
    scalar_fields = (
        [f"grad_norm_{g}" for g in GRADIENT_NAMES] + [f"cos_prev_{g}" for g in GRADIENT_NAMES]
        + [f"cos_final_{g}" for g in GRADIENT_NAMES] + [f"adv_mean_{o}" for o in OBJECTIVES]
        + [f"adv_std_{o}" for o in OBJECTIVES] + [f"adv_posfrac_{o}" for o in OBJECTIVES]
        + ["mean_vx", "mean_height", "mean_abs_vz", "mean_contact", "fall_frac", "timeout_frac", "clip_frac", "segment_position"]
        + [f"rotation_{n}" for n in ROTATION_NAMES] + [f"norm_ratio_{n}" for n in ROTATION_NAMES]
        + [f"grad_norm_unclip_{n}" for n in ROTATION_NAMES] + [f"cos_prev_unclip_{n}" for n in ROTATION_NAMES]
        + [f"{layer}_{stat}_{o}" for layer in ("reward", "return", "raw_adv") for stat in ("mean", "std", "p10", "p50", "p90") for o in OBJECTIVES]
        + [f"norm_adv_{stat}_{o}" for stat in ("p10", "p50", "p90") for o in OBJECTIVES]
        + [f"shuffle_cos_mean_{n}" for n in (*OBJECTIVES, "total_clip")] + [f"shuffle_cos_std_{n}" for n in (*OBJECTIVES, "total_clip")]
        + [f"contrib_{stat}_{o}" for stat in ("mean", "std", "frac_masked_zero", "top1pct_concentration", "top5pct_concentration", "top10pct_concentration", "frac_positive_top1pct", "frac_positive_top5pct") for o in OBJECTIVES]
        + [f"clip3a_{stat}_{pct}_{o}" for pct in THRESHOLD_PERCENTILES for stat in CLIP3A_PER_OBJ_STATS for o in OBJECTIVES]
        + [f"clip3a_{stat}_{pct}_total" for pct in THRESHOLD_PERCENTILES for stat in CLIP3A_TOTAL_STATS]
        + [f"temporal_cos_real_{n}" for n in (*OBJECTIVES, "total")]
        + [f"temporal_cos_clip_{pct}_{n}" for pct in THRESHOLD_PERCENTILES for n in (*OBJECTIVES, "total")]
    )

    try:
        for k in range(1, args.num_updates + 1):
            current_update[0] = k
            for v in descriptor_buf.values():
                v.clear()
            step_idx[0] = 0
            captured_perms.clear()
            prev_vec.clear()
            prev_vec_unclip.clear()
            minibatch_records.clear()

            actor_params_before = _flatten([p.detach() for p in actor_params])

            def _traced_randperm(*a, **kw):
                out = orig_randperm(*a, **kw)
                captured_perms.append(out.detach().clone())
                return out

            torch.randperm = _traced_randperm
            try:
                trainer.update()
            finally:
                torch.randperm = orig_randperm

            actor_params_after = _flatten([p.detach() for p in actor_params])
            delta_theta = actor_params_after - actor_params_before

            # Audit (per explicit request): exactly epochs_per_update calls,
            # every captured permutation a genuine permutation of
            # arange(total), every epoch's minibatches cover every sample
            # exactly once.
            assert len(captured_perms) == epochs_per_update, (
                f"update {k}: expected {epochs_per_update} randperm calls, captured {len(captured_perms)}"
            )
            for e, perm in enumerate(captured_perms):
                perm_np = perm.cpu().numpy()
                assert np.array_equal(np.sort(perm_np), np.arange(total)), f"update {k} epoch {e}: captured tensor is not a permutation of arange({total})"
            assert len(minibatch_records) == epochs_per_update * num_minibatches, (
                f"update {k}: expected {epochs_per_update * num_minibatches} minibatch records, got {len(minibatch_records)}"
            )
            for e in range(epochs_per_update):
                epoch_idx = np.concatenate([r["idx"] for r in minibatch_records if r["epoch"] == e])
                assert np.array_equal(np.sort(epoch_idx), np.arange(total)), f"update {k} epoch {e}: minibatch indices don't cover arange({total}) exactly once"

            # Termination-semantics invariant (per explicit request): prove,
            # not assume, that the `dones` GAE actually bootstrap-masks on
            # (rollout_capture["terminal_fall"], from r["dones"] inside
            # _collect_rollout) is exactly this env's term_base_contact
            # field, NOT timeout -- printed once per update, not asserted,
            # since a task with obstacles enabled would legitimately break
            # this equality (terminal_fall = obstacle_reached | base_contact
            # there) without being a bug.
            term_match = np.array_equal(rollout_capture["terminal_fall"], np.stack(descriptor_buf["term_base_contact"]).astype(bool))
            if k == 1:
                print(f"[termination invariant] terminal_fall == term_base_contact (update 1): {term_match}")

            # cos_to_final_delta -- now that delta_theta is known, then
            # discard the transient full gradient vectors (never saved).
            for r in minibatch_records:
                for gname in GRADIENT_NAMES:
                    r[f"cos_final_{gname}"] = _cos(r["_vectors"][gname], delta_theta)
                del r["_vectors"]

            print(f"update {k}: {len(minibatch_records)} minibatch records, ||delta_theta||={np.linalg.norm(delta_theta):.5f}, "
                  f"cos(mb0_total, delta_theta)={minibatch_records[0]['cos_final_total']:+.4f}")

            # Experiment 3A: freeze c_j right after update 1 finishes --
            # threshold_source_update=1, threshold_frozen=True,
            # threshold_scope=per_objective, threshold_pool=all minibatches
            # in update 1. Never recomputed after this point.
            if k == 1:
                for name in OBJECTIVES:
                    pooled = np.concatenate(threshold_calibration[name])
                    frozen_thresholds[name] = {pct: float(np.percentile(pooled, pct)) for pct in THRESHOLD_PERCENTILES}
                print(f"[Experiment 3A] frozen thresholds (from update 1, n={len(pooled)} samples/objective):")
                for name in OBJECTIVES:
                    print(f"  {name}: " + ", ".join(f"P{pct}={frozen_thresholds[name][pct]:.5f}" for pct in THRESHOLD_PERCENTILES))

            if args.save_descriptors:
                grids: dict[str, np.ndarray] = {}
                for field in scalar_fields:
                    grid = np.full((epochs_per_update, num_minibatches), np.nan, dtype=np.float32)
                    for r in minibatch_records:
                        grid[r["epoch"], r["mb"]] = r[field]
                    grids[field] = grid
                if not args.no_save_indices:
                    idx_grid = np.zeros((epochs_per_update, num_minibatches, mb_size), dtype=np.int64)
                    for r in minibatch_records:
                        idx_grid[r["epoch"], r["mb"]] = r["idx"]
                    grids["idx"] = idx_grid
                # Test 13B -- top-1%/top-5% sample identity per objective,
                # fixed-size (K constant given fixed mb_size), same grid
                # pattern as idx above.
                for name in OBJECTIVES:
                    for pct, k_n in ((1, max(1, round(0.01 * mb_size))), (5, max(1, round(0.05 * mb_size)))):
                        topk_grid = np.zeros((epochs_per_update, num_minibatches, k_n), dtype=np.int64)
                        sign_grid = np.zeros((epochs_per_update, num_minibatches, k_n), dtype=np.int8)
                        for r in minibatch_records:
                            topk_grid[r["epoch"], r["mb"]] = r[f"top{pct}pct_idx_{name}"]
                            sign_grid[r["epoch"], r["mb"]] = r[f"top{pct}pct_sign_{name}"]
                        grids[f"top{pct}pct_idx_{name}"] = topk_grid
                        grids[f"top{pct}pct_sign_{name}"] = sign_grid
                # Test 13B -- full (T, N) raw state arrays for THIS update,
                # saved once (not per-minibatch) -- lets post-hoc analysis
                # look up any top-K sample's exact state via idx // N, idx % N,
                # with region bins computed from GLOBAL quantile edges over
                # the whole saved run (same principle as Layer 1/2).
                for field in ("vx", "height", "vz", "pitch", "pitch_rate", "contact_count"):
                    grids[f"raw_{field}"] = np.stack(descriptor_buf[field]).astype(np.float32)
                per_update_grids[k] = grids
    finally:
        env.step = orig_step
        trainer._collect_rollout = orig_collect_rollout
        moppo_mod.sample_preference_vector = orig_sample
        moppo_mod.d3po_actor_loss = orig_d3po
        moppo_mod.diversity_regularizer_loss = orig_diversity
        torch.randperm = orig_randperm

    if args.save_descriptors:
        flat = {f"{k}__{field}": arr for k, fields in per_update_grids.items() for field, arr in fields.items()}
        # Test 13 validation log (update 1 only) -- s_i vs TRUE per-sample
        # gradient norm, per objective, flat arrays (not gridded -- this is
        # a one-off proxy check, not a per-update/minibatch trace).
        for name in VALIDATION_OBJECTIVES:
            rows = [r for r in validation_log if r["objective"] == name]
            if rows:
                flat[f"validation__s_i_{name}"] = np.array([r["s_i"] for r in rows], dtype=np.float32)
                flat[f"validation__true_norm_{name}"] = np.array([r["true_norm"] for r in rows], dtype=np.float32)
        np.savez(args.save_descriptors, **flat)
        print(f"\nsaved per-update-per-minibatch grids -> {args.save_descriptors}")


if __name__ == "__main__":
    main()
