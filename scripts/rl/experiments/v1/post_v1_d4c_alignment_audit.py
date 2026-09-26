#!/usr/bin/env python3
"""Do D4-B's preference-dependent actions point where the D1 specialists do?

Measurement only, no parameters updated. For each preference it compares
D4-B's action against that preference's specialist, and checks the difference
between any two preferences matches the difference between the corresponding
specialists in size and direction.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import RUNS, IsaacAudit, obs_tensor

PREFS = {"P": np.array([.8, .1, .1], dtype=np.float32),
         "B": np.array([.1, .8, .1], dtype=np.float32),
         "E": np.array([.1, .1, .8], dtype=np.float32)}
PAIRS = [("P", "B"), ("P", "E"), ("B", "E")]
D4B = "post_v1_d4b-2026-09-22/d4b_terminal.pt"
D1 = "post_v1_d1-2026-09-22/specialist_{}_terminal.pt"
BASE_SEED = 47001


class D4CAlignmentAudit(IsaacAudit):
    """D4-B action alignment against the fixed-preference D1 specialists."""

    run = "post_v1_d4c-2026-09-22"
    report = "d4c.json"
    schema = "post_v1_d4c_alignment_audit_v1"
    reset_seed = BASE_SEED
    set_cfg_seed = False
    set_usd_path = False
    steps = 32
    reset_suites = 4

    def mark(self, event, **kw):
        path = self.out / "d4c.lifecycle.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"event": event, "unix": time.time(), **kw}, sort_keys=True) + "\n")

    def pref_tensor(self, label):
        return torch.as_tensor(np.repeat(PREFS[label][None, :], self.num_envs, 0), device="cuda")

    def suite(self, env, d4, spec, suite):
        cur, _ = env.reset(seed=BASE_SEED + suite)
        cur = obs_tensor(cur).cuda()
        dist = {f"{a}_vs_{b}": [] for a, b in PAIRS}
        cos = {f"{a}_vs_{b}": [] for a, b in PAIRS}
        align, dof = [], []
        with torch.no_grad():
            for _ in range(self.steps):
                d4a = {k: torch.clamp(d4.act_inference_with_preference(cur, self.pref_tensor(k)), -1, 1)
                       for k in PREFS}
                sa = {k: torch.clamp(spec[k].act_inference_with_preference(cur, self.pref_tensor(k)), -1, 1)
                      for k in PREFS}
                for a in PREFS:
                    own = torch.linalg.vector_norm(d4a[a] - sa[a], dim=-1)
                    others = torch.stack([torch.linalg.vector_norm(d4a[a] - sa[b], dim=-1)
                                          for b in PREFS if b != a], 0)
                    align.append({"preference": a, "own_distance": float(own.mean()),
                                  "other_distance_mean": float(others.mean()),
                                  "own_is_nearest": bool((own < others.min(0).values).float().mean())})
                    dof.append({"preference": a,
                                "own_per_dof": (d4a[a] - sa[a]).abs().mean(0).detach().cpu().tolist()})
                for a, b in PAIRS:
                    u, v = d4a[a] - d4a[b], sa[a] - sa[b]
                    dist[f"{a}_vs_{b}"].append(float(torch.linalg.vector_norm(u - v, dim=-1).mean()))
                    cos[f"{a}_vs_{b}"].append(
                        float(torch.nn.functional.cosine_similarity(u, v, dim=-1).mean()))
                nxt, *_ = env.step(d4a["P"])
                cur = obs_tensor(nxt).cuda()
        return {"reset_suite": suite,
                "pairwise_difference_distance": {k: float(np.mean(v)) for k, v in dist.items()},
                "pairwise_difference_cosine": {k: float(np.mean(v)) for k, v in cos.items()},
                "preference_alignment": align, "per_dof": dof}

    def rollout(self, env, obs):
        from talon_rl.d4b_aux_actor_critic import D4BAuxActorCritic
        from talon_rl.v1c_actor_critic import V1CSharedActorCritic

        self.mark("APP_INIT_OK")
        self.mark("ENV_CREATED")
        ad = env.unwrapped.action_manager.total_action_dim
        d4 = D4BAuxActorCritic(obs.shape[-1], ad).cuda()
        d4.load_state_dict(torch.load(RUNS / D4B, map_location="cuda",
                                      weights_only=False)["model"])
        d4.eval()
        spec = {}
        for label in PREFS:
            m = V1CSharedActorCritic(obs.shape[-1], ad).cuda()
            m.load_state_dict(torch.load(RUNS / D1.format(label), map_location="cuda",
                                         weights_only=False)["model"])
            m.eval()
            spec[label] = m
        self.mark("CHECKPOINTS_LOADED", count=4)

        rows = [self.suite(env, d4, spec, s) for s in range(self.reset_suites)]
        report = {"schema": self.schema, "status": "MEASUREMENT_COMPLETE",
                  "measurement_only": True, "d4b_checkpoint": f"runs/{D4B}",
                  "d1_checkpoints": {k: f"runs/{D1.format(k)}" for k in PREFS},
                  "reset_suites": self.reset_suites, "steps": self.steps, "rows": rows,
                  "note": "D4-C compares D4-B preference-dependent action changes against "
                          "fixed-preference specialist action references; no parameters are updated."}
        self.write(report)
        self.mark("ARTIFACT_WRITTEN", path=str(self.out / self.report))
        self.mark("RUN_DONE", status=report["status"])
        print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))
        return report

    def execute(self):
        self.mark("RUN_STARTED", protocol="POST-V1-D4C", measurement_only=True)
        try:
            return super().execute()
        except BaseException as exc:
            self.write({"status": "ERROR", "error": str(exc),
                        "traceback": traceback.format_exc()}, "d4c.ERROR.json")
            self.mark("ERROR", error=str(exc))
            raise


if __name__ == "__main__":
    D4CAlignmentAudit.main()
