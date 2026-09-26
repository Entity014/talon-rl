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
        self.cardinalities = tuple(int(x) for x in a.cardinalities.split(","))

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg

        cfg = TalonV4CEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.seed = self.seed  # before gym.make, or identical launches diverge
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=self.seed)
        return env, obs

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.rewards.objectives import NORMALIZATION_DIVISORS, OBJECTIVE_TERMS

        cfg, a, dev = PPOConfig(), self.a, "cuda"
        u_env = env.unwrapped
        N, H = self.num_envs, cfg.num_steps
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)

        mgr = u_env.reward_manager
        names = list(mgr.active_terms)
        # [K, n_terms] summing matrix for the T/A/O/S grouping, divided by the frozen divisors
        S = torch.zeros(4, len(names), device=dev)
        for k, terms in enumerate(OBJECTIVE_TERMS.values()):
            for t in terms:
                S[k, names.index(t)] = 1.0
        S /= torch.as_tensor(np.asarray(NORMALIZATION_DIVISORS), device=dev).unsqueeze(-1)

        model = TeacherV4().to(dev)
        aopt = torch.optim.Adam(model.actor_parameters(), lr=cfg.lr)
        copt = torch.optim.Adam(model.critic_parameters(), lr=cfg.lr)
        lr = cfg.lr
        norm = RunningNormalizer(12, center=True)
        norm.update(obs["privileged"].cpu().numpy())
        ids = torch.arange(4, device=dev).expand(N, -1).contiguous()
        w, mask = sample_objective_sets(N, self.cardinalities, gen, dev)
        ep_ret = torch.zeros(N, 4, device=dev)
        ep_len = torch.zeros(N, device=dev)
        metrics = open(self.out / "metrics.jsonl", "a")
        start = time.time()

        def norm_e(o):
            return torch.as_tensor(norm.transform(o["privileged"].cpu().numpy()), dtype=torch.float32, device=dev)

        for it in range(1, a.iterations + 1):
            buf = {k: [] for k in ("obs", "env", "w", "mask", "u", "old_logp", "old_mu", "old_sigma", "old_values", "r", "d")}
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
                    v = model.query_values(x, e, ids, w, ids)
                    obs, _, term, trunc, _ = env.step(torch.tanh(uu) * model.ACTION_CLIP)
                    r = (mgr._step_reward @ S.T) * u_env.step_dt
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
                        nw, nm = sample_objective_sets(len(di), self.cardinalities, gen, dev)
                        w = w.clone(); mask = mask.clone()
                        w[di], mask[di] = nw, nm
                last_v = model.query_values(obs["policy"], norm_e(obs), ids, w, ids)
            T = {k: torch.stack(v) for k, v in buf.items()}
            returns, adv = gae(T["r"], T["old_values"], last_v, T["d"], cfg.gamma, cfg.lam)
            flat = {k: T[k].flatten(0, 1) for k in ("obs", "env", "w", "mask", "u", "old_logp", "old_mu", "old_sigma", "old_values")}
            flat["ids"] = ids.repeat(H, 1)
            flat["returns"] = returns.flatten(0, 1)
            flat["adv"] = normalize_advantages(adv.flatten(0, 1), flat["w"], flat["mask"])
            model.train()
            lr, st = update(model, aopt, copt, flat, cfg, lr, gen)
            norm.update(np.concatenate(raw_e))

            with torch.no_grad():
                m = flat["mask"].float()
                res = (flat["returns"] - flat["old_values"]) * m
                ret_c = (flat["returns"] - (flat["returns"] * m).sum(0) / m.sum(0).clamp_min(1)) * m
                ev = (1 - res.pow(2).sum(0) / ret_c.pow(2).sum(0).clamp_min(1e-12)).tolist()
                # authority on 1024 rollout states: action change when only the set, or only e_t, changes
                pick = torch.randperm(flat["obs"].shape[0], generator=gen)[:1024].to(dev)
                xo, eo, io, wo = flat["obs"][pick], flat["env"][pick], flat["ids"][pick], flat["w"][pick]
                a0 = model.act_inference(xo, eo, io, wo)
                w2, _ = sample_objective_sets(len(pick), self.cardinalities, gen, dev)
                pref_auth = float((model.act_inference(xo, eo, io, w2) - a0).norm(dim=-1).median())
                plant_auth = float((model.act_inference(xo, eo.roll(1, 0), io, wo) - a0).norm(dim=-1).median())
            ls = model.log_std.detach()
            rec = {"iteration": it, "env_samples": it * N * H, "lr": lr, **st,
                   "log_std": {"min": float(ls.min()), "mean": float(ls.mean()), "max": float(ls.max())},
                   "reward_per_step": T["r"].mean((0, 1)).tolist(), "explained_variance": ev,
                   "preference_authority": pref_auth, "plant_authority": plant_auth,
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
            if not finite or ls.min() < LOG_STD_GATE[0] or ls.max() > LOG_STD_GATE[1]:
                self._save(model, aopt, copt, lr, norm, it, "stopped")
                raise RuntimeError(f"stop gate at iteration {it}: finite={finite} log_std=[{float(ls.min())}, {float(ls.max())}]")
            if (a.save_every and it % a.save_every == 0) or it == a.iterations:
                self._save(model, aopt, copt, lr, norm, it, f"model_{it}")
        metrics.close()
        out = {"task": self.task, "num_envs": N, "seed": self.seed, "cardinalities": list(self.cardinalities),
               "iterations": a.iterations, "env_samples": a.iterations * N * H, "ppo_config": cfg.__dict__,
               "final_lr": lr, "wall_s": round(time.time() - start, 1)}
        self.write(out)
        return out

    def _save(self, model, aopt, copt, lr, norm, it, name):
        torch.save({"model": model.state_dict(), "actor_opt": aopt.state_dict(), "critic_opt": copt.state_dict(),
                    "lr": lr, "extrinsics_normalizer": norm.state_dict(), "iteration": it,
                    "cardinalities": list(self.cardinalities), "seed": self.seed}, self.out / f"{name}.pt")


if __name__ == "__main__":
    args = TrainV4C.parse_args(
        (("--iterations",), {"type": int, "default": 300}),
        (("--num-envs",), {"type": int, "default": 4096}),
        (("--seed",), {"type": int, "default": 0}),
        (("--cardinalities",), {"required": True, "help": "comma list of training set sizes, e.g. 1,2,3,4"}),
        (("--save-every",), {"type": int, "default": 50}),
    )
    TrainV4C(args.out, args).execute()
