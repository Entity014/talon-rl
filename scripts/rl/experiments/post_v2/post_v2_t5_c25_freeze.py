#!/usr/bin/env python3
"""Freeze C25: the critic repair under actor learning."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.freeze import Freeze

SRC = "scripts/rl/experiments/post_v2"
SOURCES = (f"{SRC}/post_v2_t5_c25_actor_updating_reset_support_25.py",
           f"{SRC}/post_v2_t5_c25_replay_audit_25.py",
           f"{SRC}/post_v2_t5_c25_causal_u25_confirm.py")


class C25Freeze(Freeze):
    """Freeze C25: the critic repair under actor learning."""

    run = "post_v2_t5_c25_actor_updating25-2026-09-23"
    schema = "t5_c25_actor_updating_reset_support_synthesis_v1"
    status = "C25_CLOSED_CRITIC_REPAIR_SURVIVES_ACTOR_LEARNING_BUT_SEMANTIC_CAUSAL_CREDIT_DRIFTS"
    artifacts = ("train.json", "replay_audit.json", "causal_u25_confirm.json", "synthesis.json")

    def grad_stats(self, train, u):
        """Off-diagonal gradient cosines and drift, pooled over the four heads."""
        cs, adr, hdr, gn = [], [], [], []
        for lab in ("T", "A", "O", "S"):
            x = train["specialists"][lab][u - 1]
            m = np.array(x["objective_grad_cosine"])
            cs += m[np.triu_indices(4, 1)].tolist()
            adr.append(x["actor_param_drift"])
            hdr.append(x["head_solution_drift"])
            gn += x["objective_grad_norm"]
        return {"offdiag_cos_mean": float(np.mean(cs)),
                "offdiag_cos_min": float(np.min(cs)),
                "offdiag_cos_max": float(np.max(cs)),
                "actor_param_drift_mean": float(np.mean(adr)),
                "head_solution_drift_mean": float(np.mean(hdr)),
                "grad_norm_mean": float(np.mean(gn))}

    def body(self):
        train = self.load("train.json")
        return {
            "evidence": {
                "fresh_replay": self.load("replay_audit.json")["aggregate"],
                "u25_gradient_geometry": self.grad_stats(train, 25),
                "u10_gradient_geometry": self.grad_stats(train, 10),
                "u25_causal": self.load("causal_u25_confirm.json"),
            },
            "findings": {
                "critic": "Reset-diverse critic repair remains effective under actor learning through u25: fresh H32 EV ~0.189, MC64 EV ~0.416, Orientation H32 ~0.082, H32 mean |bias| ~0.061, MC64 negative fraction 0%.",
                "tracking": "Tracking H32 is near neutral at u25 (~-0.002), consistent with C24's nonstructural short-horizon residual diagnosis.",
                "gradient_separability": "Objective gradients remain distinct through u25; mean off-diagonal cosine stays near zero and PPO ratio invariance remains ~1e-5.",
                "semantic_credit": "Physical causal semantics do not persist. At u25 Angular perturbation is wrong-sign on average for H1-H16 and Orientation is correct locally H1-H4 but wrong-sign for H8-H32.",
                "safety": "One Smoothness fresh suite shows survival 0.875 at u10/u25, while other suites and dedicated additional Smoothness probes survive at 1.0. This is a warning but not a global collapse.",
                "interpretation": "The critic-side foundation survives the coupled loop. The remaining blocker has moved downstream: distinct, numerically valid objective gradients cease to map reliably to intended closed-loop physical semantics as the actor policy evolves.",
            },
            "decision": {
                "C25": "CLOSED — critic coupled-loop PASS / semantic-credit persistence FAIL",
                "critic_repair_contract": "RETAIN",
                "actor_distribution_shift_as_value_failure": "REJECTED under reset-diverse support",
                "full_T4": "BLOCKED",
                "V2": "OFF",
                "next": "C26 actor-policy semantic-drift audit, diagnostic-only. Track A/O objective-gradient causal response across checkpoints u0/u1/u5/u10/u25 on matched states and horizons, while measuring state visitation/contact/action saturation changes. Determine when and why a still-distinct objective gradient loses physical meaning before changing reward, PPO, or critic again.",
            },
            "provenance": {
                "train_sha256": self.sha(self.dir / "train.json"),
                "replay_sha256": self.sha(self.dir / "replay_audit.json"),
                "causal_u25_sha256": self.sha(self.dir / "causal_u25_confirm.json"),
                "train_script_sha256": self.sha(SOURCES[0]),
                "replay_script_sha256": self.sha(SOURCES[1]),
                "causal_script_sha256": self.sha(SOURCES[2]),
            },
        }

    def manifest_extra(self):
        return {"sources": {s: {"sha256": self.sha(s)} for s in SOURCES}}

    def summary(self, syn):
        return {"status": syn["status"],
                "u25": syn["evidence"]["fresh_replay"]["25"],
                "grad": syn["evidence"]["u25_gradient_geometry"],
                "next": syn["decision"]["next"]}


if __name__ == "__main__":
    C25Freeze.main()
