#!/usr/bin/env python3
"""V4-C trainer: TeacherV4 from scratch on Isaac-Talon-A1-V4C-v0 with objective-set PPO in the M0 shell.

Frozen contracts: docs/verdicts/teacher_v4/ (env = stock A1 flat + e_t,
T/A/O/S with the T3-B divisors, 4096 x 24, 5 epochs x 4 minibatches, M0 PPO
constants, 300 iterations). Rollouts persist across iterations as in
rsl_rl; each env draws a new objective set when its episode ends. e_t goes
through a centered RunningNormalizer whose stats are frozen during a rollout
and updated after it. Writes metrics.jsonl, checkpoints and summary.json to
--out. Stops (exit 1) on any non-finite statistic or a log-std outside
[-5, 2]; log-std is gated, not clamped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/

import numpy as np
import torch

from rl.core.diagnostics.isaac_audit import IsaacAudit  # first: puts the repo root (talon_rl) on sys.path
from rl.core.algorithms.objective_set_ppo import PPOConfig, gae, normalize_advantages, sample_objective_sets, update
from rl.core.normalization.running import RunningNormalizer

LOG_STD_GATE = (-5.0, 2.0)
DEAD_CRITIC_STREAK = 5  # consecutive iterations with a constant critic feature c_t
# FB-2 preference regions over w_R (R, O preference): O-vertex [0, .15), O+ [.15, .40), C [.40, .60), R+ [.60, .85), R-vertex [.85, 1]
LAGR_EDGES = (0.15, 0.40, 0.60, 0.85)
LAGR_REGIONS = ("O-vertex", "O+", "C", "R+", "R-vertex")


def region_of(w_r):
    return torch.bucketize(w_r, torch.tensor(LAGR_EDGES, device=w_r.device), right=True)


RESTART_SEED_OFFSET = 1_000_000  # budget audit: a resumed run draws env/sampler randomness from seed + offset


class TrainV4C(IsaacAudit):
    """V4-C TeacherV4 trainer."""
    task = "Isaac-Talon-A1-V4C-v0"
    report = "summary.json"

    def __init__(self, out, a):
        if not out:
            raise SystemExit("--out is required: runs/ is the only copy of a training run")
        super().__init__(out)
        self.a = a
        self.num_envs = a.num_envs
        self.seed = a.seed
        # a controlled restart, not an exact continuation: env, sampler and RNG state are not in the checkpoint
        self.run_seed = a.seed + RESTART_SEED_OFFSET if a.resume else a.seed
        # F8 task-anchored support: every sampled set contains this objective (not a rule of the final architecture)
        self.required = a.objectives.index(a.require_objective) if a.require_objective else None
        self.cardinalities = tuple(int(x) for x in a.cardinalities.split(","))
        self.objectives = a.objectives
        if not self.objectives or any(x not in "TAOSV" for x in self.objectives) or len(set(self.objectives)) != len(self.objectives):
            raise SystemExit("--objectives must be distinct letters from TAOSV, e.g. TAO")
        if ("V" in self.objectives) != (a.v_objective != "none"):
            raise SystemExit("objective V needs --v-objective V1|V3, and --v-objective needs V in --objectives")

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg

        if self.a.v_objective == "V3":  # F8: body_height_osc_l2 lives in its own env variant
            from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CV3EnvCfg
            cfg, self.task = TalonV4CV3EnvCfg(), "Isaac-Talon-A1-V4C-V3-v0"
        elif self.a.s_objective == "action_jerk":
            from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CS1EnvCfg
            cfg, self.task = TalonV4CS1EnvCfg(), "Isaac-Talon-A1-V4C-S1-v0"
        else:
            cfg = TalonV4CEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.seed = self.run_seed  # before gym.make, or identical launches diverge
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=self.run_seed)
        return env, obs

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.rewards.objectives import NORMALIZATION_DIVISORS, OBJECTIVE_TERMS

        import dataclasses
        a, dev = self.a, "cuda"
        cfg = dataclasses.replace(PPOConfig(), desired_kl=a.desired_kl)  # F5 A2: relaxes the adaptive-KL LR controller (thresholds d/2, 2d)
        u_env = env.unwrapped
        N, H = self.num_envs, cfg.num_steps
        torch.manual_seed(self.run_seed)
        gen = torch.Generator().manual_seed(self.run_seed)

        mgr = u_env.reward_manager
        names = list(mgr.active_terms)
        # [K, n_terms] summing matrix for the T/A/O/S grouping, divided by the frozen divisors
        groups = [list(t) for t in OBJECTIVE_TERMS.values()]
        divisors = np.asarray(NORMALIZATION_DIVISORS, dtype=np.float64).copy()
        if a.s_objective == "action_jerk":  # V4-C2S-R1: S1 replaces action_rate_l2 as the S objective
            from talon_rl.rewards.objectives import S1_DIVISOR, S1_TERM
            groups[3] = [S1_TERM]; divisors[3] = S1_DIVISOR
        # objective subset in the order given (V4-C3 drops S: "TAO"; F8 adds V: "TAOV")
        if a.v_objective != "none":
            from talon_rl.rewards.objectives import V_REALIZATIONS
            vt, vd = V_REALIZATIONS[a.v_objective]
            groups = groups + [[vt]]; divisors = np.append(divisors, vd)
        keep = ["TAOSV".index(x) for x in self.objectives]
        groups = [groups[k] for k in keep]; divisors = divisors[keep]
        if a.lagrange_tmin is not None:
            # FB-2: the multiplied stream must be the constrained quantity. The task stream is linear tracking only
            # (same T divisor, so lambda0 keeps its FB-1 meaning); yaw tracking is in no stream in FB-2.
            groups[0] = ["track_lin_vel_xy_exp"]
        K = len(keep)
        S = torch.zeros(K, len(names), device=dev)
        for k, terms in enumerate(groups):
            for t in terms:
                S[k, names.index(t)] = 1.0
        S /= torch.as_tensor(divisors, dtype=torch.float32, device=dev).unsqueeze(-1)
        # F3: preference-invariant substrate, same coefficient in every objective row
        from talon_rl.rewards.objectives import shared_vector
        sv = torch.as_tensor(shared_vector(names, a.shared), dtype=torch.float32, device=dev)
        if a.shared != "none":
            S = S + sv.unsqueeze(0)
        shared_steps = []

        model = TeacherV4(num_objectives=K).to(dev)
        aopt = torch.optim.Adam(model.actor_parameters(), lr=cfg.lr)
        copt = torch.optim.Adam(model.critic_parameters(), lr=cfg.lr)
        lr = cfg.lr
        norm = RunningNormalizer(12, center=True)
        norm.update(obs["privileged"].cpu().numpy())
        it0 = 0
        if a.resume:
            ck = torch.load(a.resume, map_location=dev, weights_only=False)
            want = {"objectives": self.objectives, "shared": a.shared, "cardinalities": list(self.cardinalities), "seed": a.seed}
            got = {k: ck.get(k, "none" if k == "shared" else None) for k in want}
            if got != want:
                raise SystemExit(f"--resume checkpoint does not match this run: {got} != {want}")
            model.load_state_dict(ck["model"]); aopt.load_state_dict(ck["actor_opt"]); copt.load_state_dict(ck["critic_opt"])
            lr = ck["lr"]; norm.load_state_dict(ck["extrinsics_normalizer"]); it0 = ck["iteration"]
            for o in (aopt, copt):
                for g_ in o.param_groups:
                    g_["lr"] = lr
        # FB fixed-task formulation: the first objective is the task (T), never in the conditioning set; its loss
        # weight is alpha_T and the preference streams share 1 - alpha_T by w.
        task = a.task_alpha is not None or a.lagrange_tmin is not None
        # FB-2: per-preference-region dual variables for the constraint  track_lin >= tmin  (regions by w_R, frozen)
        lagr = a.lagrange_tmin is not None
        if lagr:
            if K != 3:
                raise SystemExit("--lagrange-tmin expects --objectives TAO (task T, preferences R, O)")
            lam = torch.full((len(LAGR_EDGES) + 1,), a.lagrange_lambda0, device=dev)
            tl_col = names.index("track_lin_vel_xy_exp")
        Kp = K - 1 if task else K
        qids = torch.arange(K, device=dev).expand(N, -1).contiguous()
        ids = qids[:, 1:].contiguous() if task else qids
        w, mask = sample_objective_sets(N, self.cardinalities, gen, dev, num_objectives=Kp, required=self.required)
        ep_ret = torch.zeros(N, K, device=dev)
        ep_len = torch.zeros(N, device=dev)
        metrics = open(self.out / "metrics.jsonl", "a")
        start = time.time()

        def norm_e(o):
            return torch.as_tensor(norm.transform(o["privileged"].cpu().numpy()), dtype=torch.float32, device=dev)

        dead_streak = 0
        for it in range(it0 + 1, a.iterations + 1):
            buf = {k: [] for k in ("obs", "env", "w", "mask", "u", "old_logp", "old_mu", "old_sigma", "old_values", "r", "d")}
            if lagr:
                tl_sum = torch.zeros(len(lam), device=dev); tl_cnt = torch.zeros(len(lam), device=dev)
            raw_e, done_ret, done_len = [], [], []
            model.eval()
            with torch.no_grad():  # not inference_mode: buffers feed the autograd update
                for _ in range(H):
                    x = obs["policy"]
                    raw_e.append(obs["privileged"].cpu().numpy())
                    e = norm_e(obs)
                    dist = model._dist(x, e, ids, w)
                    uu = dist.sample()
                    lp = (dist.log_prob(uu) - model._log_det_jacobian(uu)).sum(-1)
                    v = model.query_values(x, e, ids, w, qids)
                    obs, _, term, trunc, _ = env.step(torch.tanh(uu) * model.ACTION_CLIP)
                    r = (mgr._step_reward @ S.T) * u_env.step_dt
                    shared_steps.append(float((mgr._step_reward @ sv).mean()) * u_env.step_dt)
                    if lagr:  # online linear tracking per region, weighted step value (same units as replay tl)
                        reg = region_of(w[:, 0])
                        tl_sum.index_add_(0, reg, mgr._step_reward[:, tl_col]); tl_cnt.index_add_(0, reg, torch.ones_like(reg, dtype=torch.float32))
                    ep_ret += r; ep_len += 1
                    r = r + cfg.gamma * v * trunc.unsqueeze(-1).float()  # rsl_rl time-out bootstrap
                    d = term | trunc
                    for k, val in (("obs", x), ("env", e), ("w", w), ("mask", mask), ("u", uu), ("old_logp", lp),
                                   ("old_mu", dist.loc), ("old_sigma", dist.scale), ("old_values", v), ("r", r), ("d", d)):
                        buf[k].append(val.clone())
                    if d.any():
                        di = d.nonzero().squeeze(-1)
                        done_ret.append(ep_ret[di].clone()); done_len.append(ep_len[di].clone())
                        ep_ret[di] = 0; ep_len[di] = 0
                        nw, nm = sample_objective_sets(len(di), self.cardinalities, gen, dev, num_objectives=Kp, required=self.required)
                        w = w.clone(); mask = mask.clone()
                        w[di], mask[di] = nw, nm
                last_v = model.query_values(obs["policy"], norm_e(obs), ids, w, qids)
            T = {k: torch.stack(v) for k, v in buf.items()}
            returns, adv = gae(T["r"], T["old_values"], last_v, T["d"], cfg.gamma, cfg.lam)
            flat = {k: T[k].flatten(0, 1) for k in ("obs", "env", "w", "mask", "u", "old_logp", "old_mu", "old_sigma", "old_values")}
            flat["ids"] = ids.repeat(H, 1)
            flat["returns"] = returns.flatten(0, 1)
            if task:
                flat["query_ids"] = qids.repeat(H, 1)
                if lagr:  # L = w_R R + w_O O + lambda_region(w) T ; the dual step follows the update
                    flat["loss_w"] = torch.cat([lam[region_of(flat["w"][:, 0])].unsqueeze(-1), flat["w"]], -1)
                else:
                    flat["loss_w"] = torch.cat([torch.full_like(flat["w"][:, :1], a.task_alpha), (1 - a.task_alpha) * flat["w"]], -1)
                flat["loss_mask"] = torch.ones_like(flat["loss_w"], dtype=torch.bool)
            flat["adv"] = normalize_advantages(adv.flatten(0, 1), flat.get("loss_w", flat["w"]), flat.get("loss_mask", flat["mask"]))
            model.train()
            lr, st = update(model, aopt, copt, flat, cfg, lr, gen)
            if lagr:  # projected dual update (descent in lambda for max L = J_pref + lambda (J_lin - tmin)); unsampled regions unchanged
                seen = tl_cnt > 0
                tl_reg = torch.where(seen, tl_sum / tl_cnt.clamp_min(1), torch.full_like(tl_sum, float("nan")))
                lam = torch.where(seen, (lam + a.lagrange_eta * (a.lagrange_tmin - tl_reg)).clamp(0.0, a.lagrange_cap), lam)
                self.lam_state = lam.tolist()
            norm.update(np.concatenate(raw_e))

            with torch.no_grad():
                m = flat.get("loss_mask", flat["mask"]).float()
                res = (flat["returns"] - flat["old_values"]) * m
                ret_c = (flat["returns"] - (flat["returns"] * m).sum(0) / m.sum(0).clamp_min(1)) * m
                ev = (1 - res.pow(2).sum(0) / ret_c.pow(2).sum(0).clamp_min(1e-12)).tolist()
                # authority on 1024 rollout states: action change when only the set, or only e_t, changes
                pick = torch.randperm(flat["obs"].shape[0], generator=gen)[:1024].to(dev)
                xo, eo, io, wo = flat["obs"][pick], flat["env"][pick], flat["ids"][pick], flat["w"][pick]
                a0 = model.act_inference(xo, eo, io, wo)
                w2, _ = sample_objective_sets(len(pick), self.cardinalities, gen, dev, num_objectives=Kp, required=self.required)
                pref_auth = float((model.act_inference(xo, eo, io, w2) - a0).norm(dim=-1).median())
                plant_auth = float((model.act_inference(xo, eo.roll(1, 0), io, wo) - a0).norm(dim=-1).median())
                # Integrity: a dead critic body (every last-layer ELU unit saturated, so c_t is
                # constant and its gradients vanish) happened in 1/22 V4-C runs. Gate, not fix.
                c_std = float(model.critic_features(xo, eo, io, wo).std(0).max())
            dead_streak = dead_streak + 1 if c_std < 1e-5 else 0
            ls = model.log_std.detach()
            rec = {"iteration": it, "env_samples": it * N * H, "lr": lr, **st,
                   **({"lagrange_lambda": lam.tolist(), "lagrange_tl_online": tl_reg.tolist()} if lagr else {}),
                   "log_std": {"min": float(ls.min()), "mean": float(ls.mean()), "max": float(ls.max())},
                   "reward_per_step": T["r"].mean((0, 1)).tolist(), "explained_variance": ev,
                   "shared_reward_per_step": float(np.mean(shared_steps[-H:])),
                   "preference_authority": pref_auth, "plant_authority": plant_auth, "critic_feature_std_max": c_std,
                   "termination_fraction": float(T["d"].float().mean()),
                   "episodes_finished": int(sum(len(x) for x in done_len)),
                   "wall_s": round(time.time() - start, 1)}
            if done_len:
                dl = torch.cat(done_len); dr = torch.cat(done_ret)
                rec["episode_length_mean"] = float(dl.mean())
                rec["episode_return_mean"] = dr.mean(0).tolist()
            metrics.write(json.dumps(rec) + "\n"); metrics.flush()
            print("ITER", json.dumps({k: rec[k] for k in ("iteration", "lr", "kl", "clip_frac", "value", "explained_variance", "preference_authority", "log_std", "episode_length_mean") if k in rec}), flush=True)

            finite = all(np.isfinite(v) for v in st.values()) and bool(torch.isfinite(ls).all())
            if dead_streak >= DEAD_CRITIC_STREAK:
                self._save(model, aopt, copt, lr, norm, it, "stopped")
                raise RuntimeError(f"stop gate at iteration {it}: critic body dead (c_t std < 1e-5) for {dead_streak} iterations")
            if not finite or ls.min() < LOG_STD_GATE[0] or ls.max() > LOG_STD_GATE[1]:
                self._save(model, aopt, copt, lr, norm, it, "stopped")
                raise RuntimeError(f"stop gate at iteration {it}: finite={finite} log_std=[{float(ls.min())}, {float(ls.max())}]")
            if (a.save_every and it % a.save_every == 0) or it == a.iterations:
                self._save(model, aopt, copt, lr, norm, it, f"model_{it}")
        metrics.close()
        out = {"task": self.task, "objectives": self.objectives, "s_objective": a.s_objective, "shared": a.shared, "v_objective": a.v_objective, "require_objective": a.require_objective, "task_alpha": a.task_alpha, "lagrange": None if a.lagrange_tmin is None else
               {"tmin": a.lagrange_tmin, "lambda0": a.lagrange_lambda0, "eta": a.lagrange_eta, "cap": a.lagrange_cap, "edges": list(LAGR_EDGES)}, "num_envs": N, "seed": self.seed, "resume": a.resume, "run_seed": self.run_seed, "cardinalities": list(self.cardinalities),
               "iterations": a.iterations, "env_samples": a.iterations * N * H, "ppo_config": cfg.__dict__,
               "final_lr": lr, "wall_s": round(time.time() - start, 1)}
        self.write(out)
        return out

    def _save(self, model, aopt, copt, lr, norm, it, name):
        torch.save({"model": model.state_dict(), "actor_opt": aopt.state_dict(), "critic_opt": copt.state_dict(),
                    "lr": lr, "extrinsics_normalizer": norm.state_dict(), "iteration": it,
                    "cardinalities": list(self.cardinalities), "objectives": self.objectives, "shared": self.a.shared, "v_objective": self.a.v_objective, "task_alpha": self.a.task_alpha, "lagrange_tmin": self.a.lagrange_tmin,
                    "lagrange_lambda": getattr(self, "lam_state", None), "seed": self.seed}, self.out / f"{name}.pt")


if __name__ == "__main__":
    args = TrainV4C.parse_args(
        (("--iterations",), {"type": int, "default": 300}),
        (("--num-envs",), {"type": int, "default": 4096}),
        (("--seed",), {"type": int, "default": 0}),
        (("--cardinalities",), {"required": True, "help": "comma list of training set sizes, e.g. 1,2,3,4"}),
        (("--save-every",), {"type": int, "default": 50}),
        (("--s-objective",), {"choices": ("action_rate", "action_jerk"), "default": "action_rate"}),
        (("--objectives",), {"default": "TAOS", "help": "objective subset in TAOS order, e.g. TAO for V4-C3"}),
        (("--desired-kl",), {"type": float, "default": 0.01, "help": "adaptive-KL target; 0.01 is the M0 value (F5 A2 uses 0.02)"}),
        (("--lagrange-tmin",), {"type": float, "default": None, "help": "FB-2: constraint track_lin >= tmin with per-region duals (task T, preferences R, O)"}),
        (("--lagrange-lambda0",), {"type": float, "default": 0.786}),
        (("--lagrange-eta",), {"type": float, "default": 0.15}),
        (("--lagrange-cap",), {"type": float, "default": 20.0}),
        (("--task-alpha",), {"type": float, "default": None, "help": "FB: first objective is a fixed-weight task term with this alpha_T; the rest are preferences"}),
        (("--require-objective",), {"default": None, "help": "F8: every sampled objective set contains this letter, e.g. T"}),
        (("--v-objective",), {"choices": ("none", "V1", "V3"), "default": "none", "help": "F8 Vertical Stability realization (objectives.V_REALIZATIONS)"}),
        (("--resume",), {"default": None, "help": "budget audit: controlled restart from this checkpoint (model, optimizers, LR, normalizer)"}),
        (("--shared",), {"choices": ("none", "linz", "torque_acc", "air", "all"), "default": "none",
                         "help": "F3 preference-invariant substrate arm (talon_rl.rewards.objectives.SHARED_ARMS)"}),
    )
    TrainV4C(args.out, args).execute()
