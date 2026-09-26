#!/usr/bin/env python3
"""How hard the tail penalty would pull, per action coordinate and checkpoint.

Read-only: it builds the tail loss on fixed probe states and measures its
gradient, without applying anything.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import RUNS, OfflineAudit

PREFS = {"T": [.7, .1, .1, .1], "A": [.1, .7, .1, .1], "O": [.1, .1, .7, .1],
         "S": [.1, .1, .1, .7], "C": [.25] * 4}
CHECKPOINTS = {
    "u0": "authority_isolated_coordinate_headroom_control-2026-09-25/model_0.pt",
    "control10": "authority_isolated_coordinate_headroom_control-2026-09-25/model_10.pt",
    "treatment10": "authority_isolated_coordinate_headroom_treatment-2026-09-25/model_10.pt",
}
TAU_PROBE = "authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
PROBE_STATES = "update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
COORDS = ["FL_hip", "FR_hip", "RL_hip", "RR_hip",
          "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
          "FL_calf", "FR_calf", "RL_calf", "RR_calf"]


def actor_params(m):
    return [p for n, p in m.named_parameters()
            if n.startswith("actor_") or n == "log_std" or n.startswith("family_")]


def grad_norm(params):
    gs = [p.grad.reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(gs)).cpu()) if gs else 0.


class TailGradientAudit(OfflineAudit):
    """How hard the tail penalty would pull, per action coordinate and checkpoint."""

    run = "authority_isolated_tail_gradient_audit-2026-09-25"
    report = "tail_gradient_audit.json"

    def analyze(self):
        from talon_rl.authority_isolated_wide_critic import AuthorityIsolatedWideCritic

        tau = torch.tensor(json.loads((RUNS / TAU_PROBE).read_text())["tau"], device="cuda")
        probe = torch.tensor(np.load(RUNS / PROBE_STATES)["obs"], device="cuda")
        rep = {}
        for lab, path in CHECKPOINTS.items():
            m = AuthorityIsolatedWideCritic(probe.shape[1], 12).cuda()
            m.load_state_dict(torch.load(RUNS / path, map_location="cuda",
                                         weights_only=False)["model"])
            m.train()
            z = torch.cat([m._actor_mean_with_preference(
                probe, torch.tensor(wv, device="cuda").repeat(len(probe), 1))
                for wv in PREFS.values()])
            excess = torch.relu(z.abs() - tau)
            loss = ((excess / (tau + 1e-6)) ** 2).mean()
            frac = (z.abs() > tau).float().mean(0)
            dz = 2 * excess / (tau + 1e-6) ** 2 / z.numel()
            ps = actor_params(m)
            m.zero_grad(set_to_none=True)
            loss.backward()
            gn = grad_norm(ps)
            rep[lab] = {"tail_loss": float(loss.detach().cpu()),
                        "raw_tail_grad_norm": gn,
                        "effective_lambda001_grad_norm": .01 * gn,
                        "exceed_fraction_by_coord": {COORDS[j]: float(frac[j].cpu()) for j in range(12)},
                        "mean_abs_dL_dz_by_coord": {COORDS[j]: float(dz[:, j].mean().cpu()) for j in range(12)}}
        return rep

    def summarize(self, report):
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    TailGradientAudit.main()
