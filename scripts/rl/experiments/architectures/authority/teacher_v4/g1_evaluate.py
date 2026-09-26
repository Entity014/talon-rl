#!/usr/bin/env python3
"""V4-C G1 evaluation of one fold/seed at iteration 300: the V3 G1 protocol (objective_set_g1_evaluate.py) ported to TeacherV4 on Isaac-Talon-A1-V4C-v0.

Unchanged from V3 G1: 8 envs x 64 steps x 4 suites per preference, the same
suite seeds, endpoint/center/continuum/interior/critic/survival criteria and
thresholds, all m=2,3,4 sets, fixed probe states, permutation tolerance 1e-6.
Changed, as declared in docs/contracts/teacher_v4/teacher-v4-c-g1-contract.md:
the model API (e_t from the env, normalized with the checkpoint's own
normalizer), and the authority reference (V4 has no G0 init, so the
reference is the V3 G0 model, i.e. the Phase-1 authority level, on the same
probes and sets; probe e_t is the physical nominal stock plant, i.e. the
live constant channels with the trunk at its pre-randomization mass, passed
through the checkpoint's frozen normalizer). The authority of the V4 init
(rebuilt from the training seed) is reported, never gated.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/

import numpy as np
import torch

from rl.core.diagnostics.isaac_audit import REPO, IsaacAudit, obs_tensor  # first: repo root on sys.path
from rl.core.normalization.running import RunningNormalizer

PROBE = REPO / "runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
G0 = REPO / "runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
NENV = 8; STEPS = 64; SUITES = 4; G = .99
ORDER = ("T", "A", "O", "S")
PHYS = {"T": "tracking_error", "A": "ang_vel_xy", "O": "tilt_deg", "S": "action_rate"}
FOLDS = {"G1-2": {"seen": (3, 4), "holdout": 2}, "G1-3": {"seen": (2, 4), "holdout": 3}}
ALPHAS = (0., .25, .5, .75, 1.)
AUTH_THR = .75; TOL = 1e-6


def tilt_deg(q):
    _, x, y, _ = [q[:, i] for i in range(4)]
    return torch.rad2deg(torch.acos((1 - 2 * (x * x + y * y)).clamp(-1, 1)))


def ev(y, p):
    y = np.asarray(y, float).reshape(-1); p = np.asarray(p, float).reshape(-1)
    return float(1 - np.var(y - p) / (np.var(y) + 1e-12))


def segret(R, D, st, en):
    out = np.zeros_like(R[st:en]); run = np.zeros_like(R[0])
    for t in range(en - 1, st - 1, -1):
        run = R[t] + G * run * (~D[t])[:, None]; out[t - st] = run
    return out


def mono(vals):
    vals = np.asarray(vals, float); d = np.diff(vals); target = np.sign(vals[-1] - vals[0])
    return 0.0 if target == 0 else float(np.mean(np.sign(d) == target))


def between(vals):
    vals = np.asarray(vals, float); lo = min(vals[0], vals[-1]); hi = max(vals[0], vals[-1])
    return float(np.mean((vals[1:-1] >= lo - 1e-9) & (vals[1:-1] <= hi + 1e-9)))


def center_w(m): return np.full(m, 1 / m, np.float32)


def heavy_w(m, h):
    w = np.full(m, .30 / (m - 1), np.float32); w[h] = .70; return w


def interp(a, b, alpha): return ((1 - alpha) * a + alpha * b).astype(np.float32)


class G1Evaluate(IsaacAudit):
    """V4-C G1 evaluation."""
    task = "Isaac-Talon-A1-V4C-v0"
    num_envs = NENV
    seed = 0
    report = "g1_evaluation.json"

    def __init__(self, fold: str, train_seed: int, iteration: int):
        self.run_dir = REPO / f"runs/teacher_v4_c_{fold.lower().replace('-', '_')}_seed{train_seed}-2026-09-27"
        super().__init__(self.run_dir)
        self.fold, self.train_seed, self.ck = fold, train_seed, self.run_dir / f"model_{iteration}.pt"
        if (self.run_dir / self.report).exists():
            raise SystemExit(f"{self.run_dir / self.report} exists; refusing to overwrite")

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg
        cfg = TalonV4CEnvCfg(); cfg.scene.num_envs = NENV; cfg.seed = 0
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=0)
        return env, obs

    # --- model adapters ---
    def e_norm(self, obs):
        return torch.as_tensor(self.norm.transform(obs["privileged"].cpu().numpy()), dtype=torch.float32, device="cuda")

    def act(self, x, e, ids, w):
        return self.model.act_inference(x, e, ids, w)

    def evaluate(self, env, ids, w_np, seed):
        ids_t = torch.tensor(ids, device="cuda").repeat(NENV, 1)
        w = torch.tensor(w_np, device="cuda").repeat(NENV, 1)
        obs, _ = env.reset(seed=seed)
        mgr, robot = env.unwrapped.reward_manager, env.unwrapped.scene["robot"]
        R, D, V, phys = [], [], [], []
        prev = torch.zeros((NENV, env.unwrapped.action_manager.total_action_dim), device="cuda"); done_any = np.zeros(NENV, bool)
        with torch.no_grad():
            for _ in range(STEPS):
                x, e = obs["policy"], self.e_norm(obs)
                V.append(self.model.query_values(x, e, ids_t, w, ids_t).cpu().numpy())
                a = self.act(x, e, ids_t, w)
                obs, _, te, tr, _ = env.step(a)
                raw = mgr._step_reward.detach().cpu().numpy()
                full = self.nov({n: raw[:, i] for i, n in enumerate(mgr.active_terms)}, shape=(NENV,)) * env.unwrapped.step_dt
                R.append(full[:, list(ids)])
                dd = (te | tr).cpu().numpy(); D.append(dd); done_any |= dd
                data = robot.data; cmd = env.unwrapped.command_manager.get_command("base_velocity")
                vx = float((data.root_lin_vel_b[:, 0] - cmd[:, 0]).abs().mean()); wz = float((data.root_ang_vel_b[:, 2] - cmd[:, 2]).abs().mean())
                phys.append({"tracking_error": vx + wz, "ang_vel_xy": float(torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1).mean()),
                             "tilt_deg": float(tilt_deg(data.root_quat_w).mean()), "action_rate": float(torch.linalg.vector_norm(a - prev, dim=-1).mean())})
                prev = a
        R = np.asarray(R); D = np.asarray(D, bool); V = np.asarray(V); H0 = segret(R, D, 0, 32); H1 = segret(R, D, 32, 64)
        p = {k: float(np.mean([x[k] for x in phys])) for k in phys[0]}
        m = len(ids)
        return {"active_objective_mean": R.mean((0, 1)).tolist(), "physical": p, "survival": float(1 - done_any.mean()),
                "critic": {"ev": [ev(H0[:, :, j], V[:32, :, j]) for j in range(m)] + [ev(H1[:, :, j], V[32:, :, j]) for j in range(m)],
                           "bias": [float(np.mean(V[:32, :, j] - H0[:, :, j])) for j in range(m)] + [float(np.mean(V[32:, :, j] - H1[:, :, j])) for j in range(m)]}}

    def authority(self, ids, model=None):
        from talon_rl.models.authority.objective_set import canonical_tokens
        from torch.func import jacrev, vmap
        model = model or self.model
        probe, e0 = self.probe, self.probe_e
        dev = probe.device; m = len(ids); P = len(probe)
        base = canonical_tokens(device=dev)[torch.tensor(ids, device=dev)]
        ids_t = torch.tensor(ids, device=dev).repeat(P, 1)
        prefs = [center_w(m)] + [heavy_w(m, h) for h in range(m)]
        v4 = lambda o, w: model.act_inference(o, e0[:o.shape[0]], ids_t[:o.shape[0]], w)  # noqa: E731
        g0 = lambda o, w: self.g0.act_inference_from_set(o, base.unsqueeze(0).repeat(o.shape[0], 1, 1), w)  # noqa: E731

        def pair(f):
            with torch.no_grad():
                out = [f(probe, torch.tensor(wv, device=dev).repeat(P, 1)) for wv in prefs]
            return float(np.mean([float(torch.linalg.vector_norm(out[i] - out[j], dim=-1).mean()) for i in range(len(out)) for j in range(i + 1, len(out))]))
        wc = torch.tensor(center_w(m), device=dev)
        Dm = torch.eye(m, device=dev)[:, :m - 1] - torch.eye(m, device=dev)[:, [-1]]
        Q, _ = torch.linalg.qr(Dm, mode="reduced")

        def jn(f):
            J = vmap(jacrev(lambda o, w: f(o.unsqueeze(0), w.unsqueeze(0)).squeeze(0), argnums=1), in_dims=(0, None))(probe, wc) @ Q
            return float(torch.linalg.matrix_norm(J, ord="fro", dim=(1, 2)).mean().detach())
        p, p0, j, j0 = pair(v4), pair(g0), jn(v4), jn(g0)
        return {"pairwise": p, "pairwise_ref": p0, "pairwise_retention": p / (p0 + 1e-12),
                "tangent": j, "tangent_ref": j0, "tangent_retention": j / (j0 + 1e-12)}

    def permutation_drift(self, ids):
        probe, e0 = self.probe, self.probe_e
        m = len(ids); P = len(probe)
        ids_t = torch.tensor(ids, device=probe.device).repeat(P, 1); w = torch.tensor(center_w(m), device=probe.device).repeat(P, 1)
        with torch.no_grad():
            ar = self.model.act_inference(probe, e0, ids_t, w); vr = self.model.query_values(probe, e0, ids_t, w, ids_t)
            da = dv = 0.0
            for q in itertools.permutations(range(m)):
                q = list(q)
                a = self.model.act_inference(probe, e0, ids_t[:, q], w[:, q]); v = self.model.query_values(probe, e0, ids_t[:, q], w[:, q], ids_t)
                da = max(da, float((a - ar).abs().max())); dv = max(dv, float((v - vr).abs().max()))
        return {"action": da, "value": dv}

    def eval_set(self, env, ids, set_index, role):
        m = len(ids); labs = [ORDER[i] for i in ids]; rows = []; cw = center_w(m)
        for suite in range(SUITES):
            seed = 860001 + set_index * 100 + suite
            c = self.evaluate(env, ids, cw, seed); c.update({"suite": suite, "label": "C"}); rows.append(c)
            for h, lab in enumerate(labs):
                q = self.evaluate(env, ids, heavy_w(m, h), seed); q.update({"suite": suite, "label": lab}); rows.append(q)
        endpoints = {}
        for h, lab in enumerate(labs):
            oo, pp, ss, do, dp = [], [], [], [], []
            for suite in range(SUITES):
                r = next(x for x in rows if x["suite"] == suite and x["label"] == lab); c = next(x for x in rows if x["suite"] == suite and x["label"] == "C")
                x = r["active_objective_mean"][h] - c["active_objective_mean"][h]; y = r["physical"][PHYS[lab]] - c["physical"][PHYS[lab]]
                oo.append(x > 0); pp.append(y < 0); ss.append(r["survival"]); do.append(x); dp.append(y)
            endpoints[lab] = {"objective_correct_fraction": float(np.mean(oo)), "physical_correct_fraction": float(np.mean(pp)),
                              "mean_objective_delta": float(np.mean(do)), "mean_physical_delta": float(np.mean(dp)), "min_survival": float(np.min(ss)),
                              "pass": bool(np.mean(oo) >= .75 and np.mean(pp) >= .75 and np.min(ss) >= .95)}
        cont = []
        for pi, (ha, hb) in enumerate(itertools.combinations(range(m), 2)):
            wa = heavy_w(m, ha); wb = heavy_w(m, hb)
            for suite in range(SUITES):
                seed = 870001 + set_index * 1000 + pi * 10 + suite
                rr = [self.evaluate(env, ids, interp(wa, wb, alpha), seed) for alpha in ALPHAS]
                for h in (ha, hb):
                    lab = labs[h]; ov = [x["active_objective_mean"][h] for x in rr]; pv = [x["physical"][PHYS[lab]] for x in rr]
                    cont.append({"axis": lab, "objective_monotonic": mono(ov), "objective_between": between(ov), "physical_monotonic": mono(pv), "physical_between": between(pv)})
        mono_f = float(np.mean([(x["objective_monotonic"] + x["physical_monotonic"]) / 2 for x in cont])) if cont else 1.0
        between_f = float(np.mean([(x["objective_between"] + x["physical_between"]) / 2 for x in cont])) if cont else 1.0
        center_between = []
        for suite in range(SUITES):
            hs = [next(x for x in rows if x["suite"] == suite and x["label"] == lab) for lab in labs]; cc = next(x for x in rows if x["suite"] == suite and x["label"] == "C")
            for h in range(m):
                vals = [x["active_objective_mean"][h] for x in hs]; cv = cc["active_objective_mean"][h]
                center_between.append(min(vals) - 1e-9 <= cv <= max(vals) + 1e-9)
        cev, cb, surv = [], [], []
        for x in rows: cev += x["critic"]["ev"]; cb += x["critic"]["bias"]; surv.append(x["survival"])
        critic = {"ev_mean": float(np.mean(cev)), "negative_fraction": float(np.mean(np.asarray(cev) < 0)), "mean_abs_bias": float(np.mean(np.abs(cb)))}
        auth = self.authority(ids); perm = self.permutation_drift(ids)
        init = self.authority(ids, self.init_model)
        auth["init_pairwise"], auth["init_tangent"] = init["pairwise"], init["tangent"]  # descriptive only
        auth["pairwise_growth_from_init"] = auth["pairwise"] / (init["pairwise"] + 1e-12)
        rng = np.random.default_rng(20260925 + set_index); interiors = []
        for k in range(3):
            wv = rng.dirichlet(np.ones(m)).astype(np.float32)
            qs = [self.evaluate(env, ids, wv, 880001 + set_index * 100 + k * 10 + suite) for suite in range(SUITES)]
            interiors.append({"weights": wv.tolist(), "min_survival": float(min(q["survival"] for q in qs))})
        required = [lab for lab in labs if lab in ("T", "A", "O")]
        criteria = {"required_semantics": all(endpoints[x]["pass"] for x in required),
                    "center_compromise": float(np.mean(center_between)) >= .75,
                    "continuum_monotonicity": mono_f >= .65, "continuum_between": between_f >= .65,
                    "critic_valid": critic["ev_mean"] > 0 and critic["negative_fraction"] <= .25,
                    "endpoint_survival": min(surv) >= .95,
                    "authority_pairwise": auth["pairwise_retention"] >= AUTH_THR,
                    "authority_tangent": auth["tangent_retention"] >= AUTH_THR,
                    "permutation": perm["action"] <= TOL and perm["value"] <= TOL}
        return {"ids": list(ids), "labels": labs, "cardinality": m, "role": role, "endpoint": endpoints,
                "center_compromise_fraction": float(np.mean(center_between)), "continuum": {"monotonicity": mono_f, "between": between_f},
                "critic": critic, "authority": auth, "permutation_drift": perm, "interiors": interiors,
                "min_endpoint_survival": float(min(surv)), "criteria": criteria, "pass": bool(all(criteria.values()))}

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.rewards.objectives import normalized_objective_vector
        self.nov = normalized_objective_vector
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        ad = env.unwrapped.action_manager.total_action_dim
        self.g0 = ObjectiveSetAuthorityIsolatedWideCritic(48, ad).cuda()
        self.g0.load_state_dict(torch.load(G0, map_location="cuda", weights_only=False)["model"]); self.g0.eval()
        self.probe = torch.tensor(np.load(PROBE)["obs"], device="cuda", dtype=torch.float32)
        # Physical nominal stock plant, not normalized zeros (= the training mean,
        # which includes the +1 kg mean of add_base_mass). In V4-C every e_t
        # channel but trunk mass is constant; take env 0 and reset the mass.
        robot = env.unwrapped.scene["robot"]
        e_live = obs["privileged"]
        others = [i for i in range(12) if i != 8]
        if float(e_live[:, others].std(0).max()) > 1e-6:
            raise RuntimeError("an e_t channel other than trunk mass varies; nominal plant is not defined this way")
        e_raw = e_live[0].clone()
        e_raw[8] = robot.data.default_mass[0, robot.find_bodies("trunk")[0][0]].to(e_raw.device)
        self.e_ref_raw = e_raw.tolist()
        e_ref = torch.as_tensor(self.norm.transform(e_raw.cpu().numpy()[None]), dtype=torch.float32, device="cuda")
        self.probe_e = e_ref.repeat(len(self.probe), 1)
        torch.manual_seed(self.train_seed)  # train_v4c.py builds TeacherV4 right after this call
        self.init_model = TeacherV4().cuda().eval()
        sets = []; si = 0
        for m in (2, 3, 4):
            role = "heldout" if m == FOLDS[self.fold]["holdout"] else ("seen" if m in FOLDS[self.fold]["seen"] else "other")
            for ids in itertools.combinations(range(4), m):
                sets.append(self.eval_set(env, ids, si, role)); print("SET", self.fold, self.train_seed, ids, role, sets[-1]["pass"], flush=True); si += 1
        held = [x for x in sets if x["role"] == "heldout"]; seen = [x for x in sets if x["role"] == "seen"]; anchor = next(x for x in sets if x["cardinality"] == 4)
        held_pass = all(x["pass"] for x in held)
        anchor_pass = anchor["criteria"]["required_semantics"] and anchor["criteria"]["critic_valid"] and anchor["criteria"]["endpoint_survival"]
        rep = {"schema": "teacher_v4_c_g1_evaluation_v1", "fold": self.fold, "seed": self.train_seed,
               "checkpoint": str(self.ck.relative_to(REPO)), "checkpoint_sha256": self.sha(self.ck), "authority_reference": str(G0.relative_to(REPO)),
               "probe_e_t_raw": self.e_ref_raw, "probe_e_t_normalized": self.probe_e[0].tolist(),
               "sets": sets, "summary": {"heldout_set_count": len(held), "heldout_pass_count": sum(x["pass"] for x in held),
                                         "seen_pass_count": sum(x["pass"] for x in seen), "seen_set_count": len(seen),
                                         "heldout_cardinality_pass": held_pass, "full_set_anchor_pass": bool(anchor_pass),
                                         "fold_seed_pass": bool(held_pass and anchor_pass)}}
        self.write(rep)
        print("FINAL", rep["summary"], flush=True)
        return rep


if __name__ == "__main__":
    a = G1Evaluate.parse_args((("--fold",), {"choices": tuple(FOLDS), "required": True}),
                              (("--seed",), {"type": int, "required": True}),
                              (("--iteration",), {"type": int, "default": 300}))
    G1Evaluate(a.fold, a.seed, a.iteration).execute()
