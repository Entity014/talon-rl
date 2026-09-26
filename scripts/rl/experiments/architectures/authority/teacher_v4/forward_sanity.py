#!/usr/bin/env python3
"""V4-B forward and invariance sanity for the untrained TeacherV4 on live Isaac-Talon-A1-v0 obs and e_t.

Collects real 48-D policy obs and 12-D extrinsics, normalizes e_t with the
same centered running normalizer MOPPO uses, and checks: finite latents and
outputs, permutation and zero-padding invariance, non-zero plant authority
(e_t -> z_t -> action), non-zero preference authority (w -> z_w -> action),
and how far apart the DeepSets latent keeps distinct preference mixtures.
Numbers are an init baseline, not a trained verdict.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/

import numpy as np
import torch

from rl.core.diagnostics.isaac_audit import IsaacAudit
from rl.core.normalization.running import RunningNormalizer

STEPS = 32
PERM_TOL = 1e-5


def _stats(x: torch.Tensor) -> dict:
    x = x.detach().float().cpu()
    return {"min": float(x.min()), "median": float(x.median()), "mean": float(x.mean()), "max": float(x.max())}


def _random_simplex(n: int, m: int, gen: torch.Generator, device) -> torch.Tensor:
    """Uniform on the simplex (Dirichlet(1)), drawn from a seeded generator."""
    g = -torch.rand(n, m, generator=gen).log()
    return (g / g.sum(-1, keepdim=True)).to(device)


class TeacherV4ForwardSanity(IsaacAudit):
    """V4-B forward/invariance sanity for TeacherV4."""
    task = "Isaac-Talon-A1-v0"
    run = "teacher_v4_b_forward_sanity-2026-09-26"
    report = "report.json"
    num_envs = 64

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401  (registers Isaac-Talon-A1-v0)
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.seed = self.seed
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg).unwrapped
        return env, env.reset()

    def rollout(self, env, tr) -> dict:
        from talon_rl.config import ExtrinsicsCfg
        from talon_rl.models.authority.teacher_v4 import TeacherV4

        # --- live data: zero-action rollout, resets resample e_t ---
        obs, ext = [tr["obs"]], [tr["extrinsics"]]
        zero = np.zeros((env.num_envs, env.action_dim), np.float32)
        for _ in range(STEPS):
            tr, _ = env.step(zero)
            obs.append(tr["obs"]); ext.append(tr["extrinsics"])
        obs = np.concatenate(obs); ext = np.concatenate(ext)

        norm = RunningNormalizer(ExtrinsicsCfg().dim, center=True)
        norm.update(ext)
        ext_n = norm.transform(ext)

        dev = "cuda"
        x = torch.as_tensor(obs, device=dev)
        e = torch.as_tensor(ext_n, dtype=torch.float32, device=dev)
        n = x.shape[0]
        torch.manual_seed(self.seed)
        m = TeacherV4(obs_dim=x.shape[1], env_dim=e.shape[1]).to(dev).eval()
        gen = torch.Generator().manual_seed(self.seed)
        ids = torch.arange(4, device=dev).expand(n, -1).contiguous()
        w = _random_simplex(n, 4, gen, dev)

        out: dict = {
            "num_samples": n, "obs_shape": list(obs.shape), "extrinsics_shape": list(ext.shape),
            "raw_extrinsics": {"mean": ext.mean(0).tolist(), "std": ext.std(0).tolist(),
                               "min": ext.min(0).tolist(), "max": ext.max(0).tolist()},
            "normalized_extrinsics": {"mean": ext_n.mean(0).tolist(), "std": ext_n.std(0).tolist()},
            "normalizer_state": {k: np.asarray(v).tolist() for k, v in norm.state_dict().items()},
        }
        checks: dict = {}
        checks["obs_is_48d"] = obs.shape[1] == 48
        checks["extrinsics_is_12d"] = ext.shape[1] == 12
        # Constant channels normalize to 0; every varying channel must come out centered and unit-scale.
        varying = ext.std(0) > 1e-6
        out["constant_extrinsic_channels"] = np.flatnonzero(~varying).tolist()
        checks["extrinsics_normalized"] = bool(
            np.all(np.abs(ext_n.mean(0)) < 1e-3) and np.allclose(ext_n.std(0)[varying], 1.0, atol=1e-2))

        with torch.no_grad():
            h = m.state_trunk(x); z = m.env_encoder(e); zw = m.actor_set_encoder(ids, w)
            u = m.pre_tanh_mean(x, e, ids, w); a = torch.tanh(u)
            v = m.query_values(x, e, ids, w)
            checks["all_finite"] = all(bool(torch.isfinite(t).all()) for t in (h, z, zw, u, a, v))
            out["latent_scale"] = {k: _stats(t.abs()) for k, t in (("h_t", h), ("z_t", z), ("z_w", zw), ("u", u))}

            # --- permutation invariance: all 24 orderings ---
            perm = {"action": 0.0, "value": 0.0, "z_w": 0.0}
            for p in itertools.permutations(range(4)):
                p = torch.tensor(p, device=dev)
                perm["action"] = max(perm["action"], float((m.act_inference(x, e, ids[:, p], w[:, p]) - a).abs().max()))
                perm["value"] = max(perm["value"], float((m.query_values(x, e, ids[:, p], w[:, p], ids) - v).abs().max()))
                perm["z_w"] = max(perm["z_w"], float((m.actor_set_encoder(ids[:, p], w[:, p]) - zw).abs().max()))
            out["permutation_max_abs_diff"] = perm
            checks["permutation_invariant"] = max(perm.values()) < PERM_TOL

            # --- zero-weight padding: {(T,.6),(O,.4)} vs same plus (A,0),(S,0) ---
            ids2 = torch.tensor([0, 2], device=dev).expand(n, -1)
            w2 = torch.tensor([.6, .4], device=dev).expand(n, -1)
            idsp = torch.tensor([0, 2, 1, 3], device=dev).expand(n, -1)
            wp = torch.tensor([.6, .4, 0., 0.], device=dev).expand(n, -1)
            pad = {"action": float((m.act_inference(x, e, ids2, w2) - m.act_inference(x, e, idsp, wp)).abs().max()),
                   "value": float((m.query_values(x, e, ids2, w2) - m.query_values(x, e, idsp, wp, ids2)).abs().max())}
            out["padding_max_abs_diff"] = pad
            checks["padding_invariant"] = max(pad.values()) < PERM_TOL

            # --- plant authority: same obs/preference, e_t from another sample ---
            e_alt = e.roll(1, 0)
            dz = (m.env_encoder(e_alt) - z).norm(dim=-1)
            da_env = (m.act_inference(x, e_alt, ids, w) - a).norm(dim=-1)
            out["plant_authority"] = {"dz_t": _stats(dz), "d_action": _stats(da_env)}
            same_e = (e_alt - e).abs().max(-1).values < 1e-6
            checks["plant_authority_nonzero"] = bool((da_env[~same_e] > 0).all() and (dz[~same_e] > 0).all())

            # --- preference authority: same obs/e_t, another preference ---
            w_alt = _random_simplex(n, 4, gen, dev)
            dzw = (m.actor_set_encoder(ids, w_alt) - zw).norm(dim=-1)
            da_w = (m.act_inference(x, e, ids, w_alt) - a).norm(dim=-1)
            out["preference_authority"] = {"dz_w": _stats(dzw), "d_action": _stats(da_w),
                                           "d_w": _stats((w_alt - w).norm(dim=-1))}
            checks["preference_authority_nonzero"] = bool((da_w > 0).all() and (dzw > 0).all())
            out["preference_to_plant_action_ratio"] = float(da_w.median() / da_env.median())

            # --- set aliasing: how close z_w gets for distinct mixtures ---
            k = 1024
            wa = _random_simplex(k, 4, gen, dev)
            za = m.actor_set_encoder(torch.arange(4, device=dev).expand(k, -1).contiguous(), wa)
            dw = torch.cdist(wa, wa); dzs = torch.cdist(za, za)
            iu = torch.triu_indices(k, k, 1, device=dev)
            ratio = dzs[iu[0], iu[1]] / dw[iu[0], iu[1]]
            far = dw[iu[0], iu[1]] > 0.2
            sv = torch.linalg.svdvals(za - za.mean(0))
            out["set_aliasing"] = {
                "pairs": int(ratio.numel()),
                "dz_w_over_dw": {"min": float(ratio.min()), "p01": float(ratio.quantile(.01)),
                                 "median": float(ratio.median()), "max": float(ratio.max())},
                "min_dz_w_for_dw_gt_0.2": float(dzs[iu[0], iu[1]][far].min()),
                "z_w_singular_values": sv.tolist(),
                # The simplex has 3 degrees of freedom; a rank-3 image keeps mixtures locally distinguishable.
                "z_w_rank_rel_1e-3": int((sv > sv[0] * 1e-3).sum()),
            }
            checks["z_w_rank_ge_3"] = out["set_aliasing"]["z_w_rank_rel_1e-3"] >= 3

        out["checks"] = checks
        out["pass"] = all(checks.values())
        self.write(out)
        print({k: v for k, v in checks.items()}, "PASS" if out["pass"] else "FAIL", flush=True)
        return out


if __name__ == "__main__":
    TeacherV4ForwardSanity.main()
