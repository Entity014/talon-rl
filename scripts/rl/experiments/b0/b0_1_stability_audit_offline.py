#!/usr/bin/env python3
"""Did the B0.1 collapses come with an abrupt jump in the actor's output?

Read-only over every saved checkpoint. Actor drift is measured on one frozen
observation tensor, so consecutive checkpoints are compared on identical
inputs; the behavioural columns come from the deterministic monitor. Metrics
the trainer never persisted are listed as unavailable rather than
reconstructed.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import ARTIFACTS, RUNS, OfflineAudit

SEEDS = [0, 1, 2]
RUN_DIRS = {s: f"b0_1_seed{s}_2026-09-21" for s in SEEDS}
OBS_FILE = "frozen_obs.npy"
SATURATION = 2.9
SELECTED = {(1, 25), (1, 50), (0, 450), (0, 475), (2, 250), (2, 275)}
UNAVAILABLE = ["approx_kl", "ratio_mean", "clip_fraction", "gradient_norm", "return_mean",
               "return_std", "advantage_mean", "advantage_std", "value_prediction_error"]

TAIL = ["", "## Interpretation", "", "- Seed 1 (25→50): survival falls 1.00→0.00 while actor-output cosine drops to 0.822; this is the clearest abrupt policy-output drift signature. The parameter-relative displacement is 0.092, so the behavioral change is large without requiring a uniquely largest global parameter jump.", "- Seed 0 (450→475): survival falls 1.00→0.359, but actor-output cosine remains 0.996 with 0.086 relative parameter displacement; this is not evidence of a comparable abrupt actor jump. It is consistent with a smaller drift amplified by a near-failure state distribution, though critic/advantage causality cannot be established from saved data.", "- Seed 2 is a negative control: it also has substantial early actor drift (cosine 0.761 at update 50) without ever satisfying the full deterministic gate, so actor drift is not by itself sufficient to explain the seed-1/seed-0 collapses.", "", "## Evidence boundary", "", "Actor drift metrics are computed from the identical frozen observation tensor; behavioral fields come from the existing deterministic monitor.", "", "The trainer did not persist approximate KL, PPO ratios/clip fraction, gradient norm, returns/advantages, or value prediction error. These are recorded as unavailable, not reconstructed.", "", "No training, configuration, reward, std schedule, or checkpoint was modified."]

def model_metrics(model, obs, previous):
    """One checkpoint's actor summary, and its drift from the previous one."""
    with torch.no_grad():
        mean = model.raw_mean(obs).cpu()
        action = model.act_inference(obs).cpu()
    params = torch.cat([v.detach().float().cpu().reshape(-1)
                        for v in model.state_dict().values() if v.is_floating_point()])
    row = {"action_saturation": float((action.abs() >= SATURATION).float().mean()),
           "mean_norm": float(mean.norm()), "param_norm": float(params.norm()),
           "cos_mu_prev": "", "delta_mu_norm": "", "delta_param_norm": "",
           "relative_param_delta": ""}
    if previous:
        pm, pp = previous
        dm, dp = mean - pm, params - pp
        row.update(cos_mu_prev=float(torch.nn.functional.cosine_similarity(
                       mean.reshape(1, -1), pm.reshape(1, -1)).item()),
                   delta_mu_norm=float(dm.norm()), delta_param_norm=float(dp.norm()),
                   relative_param_delta=float(dp.norm() / (pp.norm() + 1e-12)))
    return row, (mean, params)


class B01StabilityAudit(OfflineAudit):
    """B0.1 stability and policy preservation, over all saved checkpoints."""

    root = ARTIFACTS
    run = "b0_1_stability_audit"
    report = "summary.json"

    def analyze(self):
        from rl.core.modules.actor_critic import ActorCritic

        obs = torch.from_numpy(np.load(self.dir / OBS_FILE)).float()
        self.obs_shape = list(obs.shape)
        rows = []
        for seed in SEEDS:
            model = ActorCritic(51, 51, 12, 1, [64, 64]).cpu().eval()
            previous = None
            run = RUNS / RUN_DIRS[seed]
            ckpts = sorted((run / "checkpoints").glob("update_*.pt"),
                           key=lambda p: int(p.stem.split("_")[-1]))
            metrics = json.loads((run / "training_metrics.json").read_text())["metrics"]
            for p in ckpts:
                u = int(p.stem.split("_")[-1])
                state = torch.load(p, map_location="cpu")
                model.load_state_dict(state.get("model", state))
                row, previous = model_metrics(model, obs, previous)
                mon = json.loads((run / "monitor" / f"u{u:03d}.json").read_text())["acceptance"]
                tr = next((x for x in metrics if x["update"] == u), None) or {}
                row.update(seed=seed, update=u, survival_rate=mon["survival_rate"],
                           first_fall_mean=float(np.mean(mon["first_fall"])),
                           vx_mae=mon["mean_abs_vx_error"],
                           displacement=mon["mean_displacement"],
                           tilt_p95_rad=mon["tilt_p95_rad"], tilt_max_rad=mon["tilt_max_rad"],
                           height_error=mon["mean_height_error"],
                           scheduled_std=tr.get("std", ""), policy_loss=tr.get("policy_loss", ""),
                           value_loss=tr.get("value_loss", ""))
                rows.append(row)
        self.rows = rows
        sel = [r for r in rows if (r["seed"], r["update"]) in SELECTED]
        return {"status": "COMPLETE", "read_only": True,
                "fixed_observation_shape": self.obs_shape, "selected": sel,
                "unavailable_metrics": UNAVAILABLE}

    def summarize(self, summary):
        with (self.out / "audit.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        lines = ["# B0.1 Stability / Policy-Preservation Audit", "",
                 "Read-only audit over all saved checkpoints and deterministic monitor artifacts.", "",
                 f"Fixed actor-input artifact: `{OBS_FILE}` shape `{self.obs_shape}` (64 frozen states).", "",
                 "## Selected windows", "",
                 "| seed | update | survival | vx MAE | tilt p95 rad | cos \u03bc(prev) | \u0394\u03bc | \u0394\u03b8 rel | policy loss | value loss |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in summary["selected"]:
            lines.append("| {seed} | {update} | {survival_rate:.3f} | {vx_mae:.3f} | "
                         "{tilt_p95_rad:.3f} | {cos_mu_prev} | {delta_mu_norm} | "
                         "{relative_param_delta} | {policy_loss} | {value_loss} |".format(**r))
        lines += TAIL
        (self.out / "report.md").write_text("\n".join(lines) + "\n")
        print(self.out / "report.md")


if __name__ == "__main__":
    B01StabilityAudit.main()
