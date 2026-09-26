#!/usr/bin/env python3
"""Multi-update bearing trace -- follow-up to full_update_trace.py's result
(both Case 3 seeds: a single real update() produces a genuine, non-cancelled
tangential-dominant shift in actor_mean at stable-attractor states). That
answered "what does one real update do"; it did not answer whether that
tangential direction holds a consistent BEARING across many updates or
wanders, which is the only way to explain why 500 updates of this still
converge to the stable low-v_x attractor instead of locomotion. Three
possible patterns, per this branch's plan:
  A. bearing roughly consistent, v_x moving with it -- a scale finding (needs
     more than 500 updates), not a qualitative bug.
  B. bearing wanders update-to-update (cos(delta_mu_k, delta_mu_{k-1}) low or
     sign-flipping) -- PPO noise / preference-sampling variance is the
     mechanism, not the geometry of any single update.
  C. bearing consistent AND v_x doesn't follow it -- the actor is genuinely
     moving mu in a stable direction but it doesn't translate into motion;
     points back to the action->joint->contact->motion pathway, not PPO
     optimization at all.

Runs the REAL trainer object's REAL .update() method --num_updates times in
a row (mutates the in-memory model exactly like training would; the loaded
checkpoint FILE is never re-saved, so nothing on disk changes). Per update,
logs: cos(mu_k, mu_0) [cumulative drift from the pre-update mean], tan_frac
of THIS update's own Delta_mu against mu_{k-1} [radial/tangential split,
same decomposition as the other two scripts in this branch],
cos(Delta_mu_k, Delta_mu_{k-1}) [bearing consistency between consecutive
updates -- the key one cumulative cosine alone can hide, since a rotate-out-
and-rotate-back pattern would still show high cumulative cos], ||Delta_mu||,
mean |v_x| achieved during that update's own rollout, and clip_fraction
(captured by wrapping losses.d3po_actor_loss for the duration of the run,
restored after).

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py multi-update-trace \
        runs/phase1_hipact_dt01_seed0_2026-09-20/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 \
        --num_updates 10 --sim_dt 0.01
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


def _radial_tangential_mean(delta_mu: np.ndarray, mu_before: np.ndarray) -> tuple[float, float]:
    mu_before_sq = np.sum(mu_before * mu_before, axis=-1, keepdims=True)
    mu_before_sq_safe = np.where(mu_before_sq < 1e-12, 1.0, mu_before_sq)
    coeff = np.sum(delta_mu * mu_before, axis=-1, keepdims=True) / mu_before_sq_safe
    radial_vec = coeff * mu_before
    tangential_vec = delta_mu - radial_vec
    delta_norm = np.linalg.norm(delta_mu, axis=-1)
    tangential_norm = np.linalg.norm(tangential_vec, axis=-1)
    tan_frac = np.median(tangential_norm / np.maximum(delta_norm, 1e-8))
    return float(delta_norm.mean()), float(tan_frac)


def _row_cos(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return float(np.mean(num / den))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4, help="Force this w (RewardVectorCfg.term_names order), need not sum to 1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=24, help="MOPPOConfig.num_steps per update -- default matches training config, NOT the longer eval-only rollout length other probes in this branch use")
    parser.add_argument("--num_updates", type=int, default=10)
    parser.add_argument("--n_probe", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--stable_vx_thresh", type=float, default=0.15)
    parser.add_argument("--fall_exclude_window", type=int, default=5)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--sim_dt", type=float, default=0.02)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_encoder", action="store_true")
    parser.add_argument("--save_descriptors", default=None, help="Path to save per-update raw state descriptors (.npz) for compare_consecutive_rollouts.py -- omit to skip saving")
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

    w_single = (np.array(args.w, dtype=np.float32) / sum(args.w)).astype(np.float32)
    w_forced = np.tile(w_single, (args.num_envs, 1))
    trainer.w = w_forced.copy()
    env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    orig_sample = moppo_mod.sample_preference_vector

    def _fixed_sample(rng, reward_cfg_, pref_cfg_, n, step=0):
        # n varies (num_envs at rollout-collection time, minibatch_size --
        # a DIFFERENT number, e.g. T*N/num_minibatches -- inside update()'s
        # diversity-regularizer draw), so tile to whatever n is asked for
        # rather than slicing a fixed-size w_forced array (that silently
        # under-filled and crashed update()'s shape-broadcast the first
        # time this was tried against the real trainer.update() loop).
        return np.tile(w_single, (n, 1))

    moppo_mod.sample_preference_vector = _fixed_sample

    vx_buf: list[np.ndarray] = []
    # Raw state descriptors for the state-distribution-shift diagnostic
    # (Layer 1, 2026-09-20 branch) -- pulled straight from `transition`,
    # never from actor_obs (actor_obs is obs_norm-normalized, which would
    # confound any shift found here with running-stat drift, exactly the
    # variable this diagnostic must NOT touch yet). CONTACT_THRESHOLD_N=1.0
    # matches the convention already used in attractor_phase_metrics.py /
    # command_gait_comparison.py / hip_functional_correlation.py etc.
    CONTACT_THRESHOLD_N = 1.0
    descriptor_buf: dict[str, list[np.ndarray]] = {
        "vx": [], "height": [], "vz": [], "pitch": [], "pitch_rate": [],
        "contact_count": [], "action_norm": [], "done": [],
    }
    # term_time_out/term_base_contact: raw a1_env.py termination-reason
    # flags (a1_env.py:122,124), added ONLY if this env exposes them --
    # never synthesized from contact_count or any other proxy (a
    # nonzero contact_count is not the same claim as "this step ended the
    # episode"). DummyTalonEnv has no such keys, so this stays empty there.
    has_term_flags = False
    orig_step = env.step

    def _wrapped_step(action):
        nonlocal has_term_flags
        transition, done = orig_step(action)
        vx_buf.append(transition["v_actual"][:, 0].copy())
        descriptor_buf["vx"].append(transition["v_actual"][:, 0].copy())
        descriptor_buf["height"].append(transition["height"].copy())
        descriptor_buf["vz"].append(transition["v_z"].copy())
        descriptor_buf["pitch"].append(transition["roll_pitch"][:, 1].copy())
        descriptor_buf["pitch_rate"].append(transition["roll_pitch_rate"][:, 1].copy())
        descriptor_buf["contact_count"].append((transition["foot_contact_force"] > CONTACT_THRESHOLD_N).sum(axis=-1).astype(np.float32))
        descriptor_buf["action_norm"].append(np.linalg.norm(action, axis=-1))
        descriptor_buf["done"].append(np.asarray(done, dtype=np.float32).copy())
        if "term_time_out" in transition and "term_base_contact" in transition:
            has_term_flags = True
            descriptor_buf.setdefault("term_time_out", []).append(np.asarray(transition["term_time_out"], dtype=np.float32).copy())
            descriptor_buf.setdefault("term_base_contact", []).append(np.asarray(transition["term_base_contact"], dtype=np.float32).copy())
        return transition, done

    env.step = _wrapped_step

    # Layer 2 (2026-09-20 branch, valuation/advantage comparison) -- wraps
    # trainer._collect_rollout ADDITIVELY (calls the real one, copies its
    # return, hands it back unchanged) so update()'s actual PPO step is
    # untouched; advantage/return are NOT computed here -- update() already
    # does that internally as a local var it never exposes, so
    # compare_valuation.py recomputes it post-hoc with the exact same
    # gae_per_objective() training uses, from these raw ingredients.
    # r["dones"] here is `terminal_fall` (real termination only -- see
    # a1_env.py's terminal_fall wiring and moppo.py's own comment on why
    # done vs terminal_fall matter for GAE bootstrap masking) -- NOT the
    # same array as descriptor_buf["done"] above, which is the raw env.step
    # done (timeout+fall conflated). Saved under its own "terminal_fall" key
    # specifically so nothing downstream can mix the two up.
    rollout_capture: dict[str, np.ndarray] = {}
    orig_collect_rollout = trainer._collect_rollout

    def _wrapped_collect_rollout():
        r = orig_collect_rollout()
        rollout_capture["values"] = r["values"].copy()
        rollout_capture["rewards"] = r["rewards"].copy()
        rollout_capture["terminal_fall"] = r["dones"].copy()
        rollout_capture["final_value"] = r["final_value"].copy()
        return r

    trainer._collect_rollout = _wrapped_collect_rollout

    clip_fracs_buf: list[float] = []

    # Layer 3 (2026-09-20 branch, gradient-formation diagnostic) -- per
    # update k, epoch 0 / minibatch 0 ONLY (this repo's existing single-
    # minibatch convention, see mean_update_decomposition.py /
    # full_update_trace.py), captures:
    #   g_<objective> for each of reward_cfg.term_names -- EXACT linear
    #     per-objective decomposition of clip_loss (d3po_actor_loss:
    #     clip_loss = -mean(sum_i w_i * per_objective_i), so
    #     sum_i(g_<objective_i>) == grad(clip_loss) by construction --
    #     verified numerically in this script's smoke test, not assumed).
    #   g_diversity -- grad of the REAL diversity_lambda*diversity_loss
    #     term as it actually enters policy_loss, not raw diversity_loss.
    #   g_total -- grad of the REAL reconstructed policy_loss
    #     (clip_loss + diversity_lambda*diversity_loss), i.e. exactly what
    #     backward() differentiates for the actor -- not an approximation.
    # ALL via torch.autograd.grad(..., retain_graph=True) -- NEVER
    # retain_graph=False, even on the last capture: the real loss.backward()
    # (value_loss/entropy/mean_reg layered on top of these same tensors)
    # still needs the graph afterward, and this instrumentation must never
    # touch optimizer.zero_grad()/backward()/step(), which run byte-for-byte
    # as they would without this file. actor_params' ordering is fixed once
    # and reused for delta_theta below so every cosine downstream compares
    # like-for-like.
    actor_params = list(trainer.model.actor_body.parameters()) + list(trainer.model.actor_mean.parameters()) + [trainer.model.log_std]

    def _flatten(tensors) -> np.ndarray:
        return torch.cat([t.reshape(-1) for t in tensors]).detach().cpu().numpy().astype(np.float32)

    def _flatten_grad(loss: torch.Tensor) -> np.ndarray:
        grads = torch.autograd.grad(loss, actor_params, retain_graph=True, allow_unused=True)
        return _flatten([g if g is not None else torch.zeros_like(p) for g, p in zip(grads, actor_params)])

    minibatch_seq = [0]  # reset to 0 right before each trainer.update() call, below
    grad_capture: dict[str, np.ndarray] = {}
    captured_clip_loss: list[torch.Tensor | None] = [None]

    orig_d3po = losses_mod.d3po_actor_loss

    def _wrapped_d3po(ratio, adv, w, clip_eps):
        with torch.no_grad():
            clip_fracs_buf.append(float((torch.abs(ratio - 1.0) > clip_eps).float().mean().item()))
        minibatch_seq[0] += 1
        clip_loss = orig_d3po(ratio, adv, w, clip_eps)
        if minibatch_seq[0] == 1:
            clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            per_objective = torch.min(ratio[:, None] * adv, clipped_ratio[:, None] * adv)  # (B, K) -- identical construction to d3po_actor_loss's own internals
            for i, name in enumerate(reward_cfg.term_names):
                loss_i = -(w[:, i] * per_objective[:, i]).mean()
                grad_capture[f"g_{name}"] = _flatten_grad(loss_i)
            captured_clip_loss[0] = clip_loss
        return clip_loss

    moppo_mod.d3po_actor_loss = _wrapped_d3po

    orig_diversity = losses_mod.diversity_regularizer_loss

    def _wrapped_diversity(mean_w, mean_w_prime, w, w_prime, std, alpha):
        diversity_loss = orig_diversity(mean_w, mean_w_prime, w, w_prime, std, alpha)
        if minibatch_seq[0] == 1 and captured_clip_loss[0] is not None:
            scaled_diversity = trainer.cfg.diversity_lambda * diversity_loss
            grad_capture["g_diversity"] = _flatten_grad(scaled_diversity)
            policy_loss_reconstructed = captured_clip_loss[0] + scaled_diversity
            grad_capture["g_total"] = _flatten_grad(policy_loss_reconstructed)
        return diversity_loss

    moppo_mod.diversity_regularizer_loss = _wrapped_diversity

    def _build_probe_set() -> torch.Tensor:
        """One rollout (steps=args.steps, same as an update() would collect)
        under the current live policy, used only to pick a frozen probe
        state set -- consumes it via the same wrapped env.step, so vx_buf
        is cleared before/after to not contaminate update 1's v_x stat."""
        vx_buf.clear()
        r0 = trainer._collect_rollout()
        vx_buf.clear()
        terminal_mask = r0["dones"]
        fall_exclude = terminal_mask.copy()
        for shift in range(1, args.fall_exclude_window + 1):
            fall_exclude[:-shift] |= terminal_mask[shift:]
            fall_exclude[shift:] |= terminal_mask[:-shift]
        T0, N0 = terminal_mask.shape
        valid = np.ones((T0, N0), dtype=bool)
        valid[: min(args.warmup, T0 - 1)] = False
        valid[fall_exclude] = False
        return r0, valid

    try:
        r0, valid0 = _build_probe_set()
        # v_actual isn't stored in the rollout dict itself; reconstruct the
        # stable mask straight from actor_obs's command-tracking behavior is
        # unnecessary here -- just use whatever samples survived warmup/fall
        # exclusion regardless of v_x (probe set is meant to be a fixed
        # state anchor, not re-filtered by v_x every update since v_x itself
        # is what we're now trying to explain the trend of).
        t_idx, n_idx = np.nonzero(valid0)
        n_available = len(t_idx)
        print(f"\n{n_available} valid probe-anchor states available from initial rollout")
        rng = np.random.default_rng(args.seed)
        n_probe = min(args.n_probe, n_available)
        sel = rng.choice(n_available, size=n_probe, replace=False)
        t_sel, n_sel = t_idx[sel], n_idx[sel]
        probe_actor_obs = torch.from_numpy(r0["actor_obs"][t_sel, n_sel]).to(trainer.device)

        with torch.no_grad():
            mu_0 = trainer.model.raw_mean(probe_actor_obs).clone()
        mu_0_np = mu_0.cpu().numpy()
        mu_prev = mu_0.clone()
        delta_prev = None
        path_len = np.zeros(n_probe, dtype=np.float64)  # per-probe-state cumulative sum of ||Delta_mu_i|| across updates

        print(f"\nrunning {args.num_updates} real trainer.update() calls (num_steps={args.steps} each, num_envs={args.num_envs})\n")
        header = (
            f"{'upd':>4}{'||d_mu||':>10}{'tan_frac':>10}{'cos(k,0)':>10}{'cos(dk,dk-1)':>14}"
            f"{'D_k/P_k':>10}{'mean|vx|':>10}{'clip_frac':>10}"
        )
        print(header)

        per_update_descriptors: dict[int, dict[str, np.ndarray]] = {}

        for k in range(1, args.num_updates + 1):
            vx_buf.clear()
            clip_fracs_buf.clear()
            for v in descriptor_buf.values():
                v.clear()
            minibatch_seq[0] = 0
            grad_capture.clear()
            captured_clip_loss[0] = None
            theta_before = _flatten([p.detach() for p in actor_params])
            trainer.update()
            theta_after = _flatten([p.detach() for p in actor_params])
            grad_capture["delta_theta"] = theta_after - theta_before

            if args.save_descriptors:
                per_update_descriptors[k] = {name: np.stack(vals) for name, vals in descriptor_buf.items()}  # each (T, N)
                per_update_descriptors[k].update(rollout_capture)  # values (T,N,K), rewards (T,N,K), terminal_fall (T,N), final_value (N,K) -- already correctly shaped, one _collect_rollout() call per update
                per_update_descriptors[k].update(grad_capture)  # g_<objective>, g_diversity, g_total, delta_theta -- each a flat (P,) actor-param-count vector, epoch0/mb0 only

            with torch.no_grad():
                mu_k = trainer.model.raw_mean(probe_actor_obs)
                delta_mu = (mu_k - mu_prev).cpu().numpy()
                mu_prev_np = mu_prev.cpu().numpy()
                delta_norm, tan_frac = _radial_tangential_mean(delta_mu, mu_prev_np)
                cos_k0 = torch.sum(mu_k * mu_0, dim=-1) / (
                    torch.norm(mu_k, dim=-1) * torch.norm(mu_0, dim=-1) + 1e-8
                )
                cos_k0_mean = float(cos_k0.mean().item())

            bearing_cos = _row_cos(delta_mu, delta_prev) if delta_prev is not None else float("nan")
            mean_abs_vx = float(np.mean(np.abs(np.concatenate(vx_buf)))) if vx_buf else float("nan")
            clip_frac_mean = float(np.mean(clip_fracs_buf)) if clip_fracs_buf else float("nan")

            # Path length P_k (per probe state, cumulative sum of step norms)
            # vs net displacement D_k (per probe state, straight-line
            # distance from mu_0) -- D_k/P_k ~ 1 means the walk is going one
            # direction; D_k/P_k << 1 means it's wandering/backtracking in
            # output space even though each step itself has real magnitude.
            path_len += np.linalg.norm(delta_mu, axis=-1)
            mu_k_np = mu_k.cpu().numpy()
            net_disp = np.linalg.norm(mu_k_np - mu_0_np, axis=-1)
            dp_ratio = float(np.median(net_disp / np.maximum(path_len, 1e-8)))

            print(
                f"{k:>4}{delta_norm:>10.5f}{tan_frac:>10.4f}{cos_k0_mean:>10.5f}{bearing_cos:>14.5f}"
                f"{dp_ratio:>10.4f}{mean_abs_vx:>10.4f}{clip_frac_mean:>10.4f}"
            )

            mu_prev = mu_k.clone()
            delta_prev = delta_mu

    finally:
        env.step = orig_step
        trainer._collect_rollout = orig_collect_rollout
        moppo_mod.sample_preference_vector = orig_sample
        moppo_mod.d3po_actor_loss = orig_d3po
        moppo_mod.diversity_regularizer_loss = orig_diversity

    if args.save_descriptors:
        # Flat dict of "{k}__{field}" -> (T, N) array -- np.savez can't nest,
        # compare_consecutive_rollouts.py reassembles per-update dicts by
        # splitting keys on "__".
        flat = {
            f"{k}__{field}": arr
            for k, fields in per_update_descriptors.items()
            for field, arr in fields.items()
        }
        np.savez(args.save_descriptors, **flat)
        print(f"\nsaved per-update raw state descriptors -> {args.save_descriptors}")

    print("\npattern read: A) bearing_cos consistently positive + mean|vx| rising with it -> scale finding, needs more updates.")
    print("B) bearing_cos low/sign-flipping while tan_frac stays high each update -> direction wanders, PPO/preference noise mechanism.")
    print("C) bearing_cos consistently positive but mean|vx| flat -> mu moves in a stable direction that doesn't translate to motion; look at action->joint->contact pathway next, not PPO optimization.")


if __name__ == "__main__":
    main()
