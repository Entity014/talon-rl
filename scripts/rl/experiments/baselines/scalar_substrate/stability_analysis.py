"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_b0_1_collapse_pair_audit():
    """Run former b0_1_collapse_pair_audit.py stage."""
    """Read-only same-state pair comparison for the two B0.1 collapse windows."""
    from pathlib import Path
    import json, sys
    import numpy as np
    import torch
    ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT/'scripts'))
    from rl.core.modules.actor_critic import ActorCritic
    OUT=ROOT/'artifacts'/'b0_1_stability_audit'; OBS=torch.from_numpy(np.load(OUT/'frozen_obs.npy')).float()
    PAIRS=[('seed1_25_to_50',1,25,50),('seed0_450_to_475',0,450,475),('seed2_negative_control',2,250,275)]
    def load(seed,u):
        m=ActorCritic(51,51,12,1,[64,64]).eval(); p=ROOT/f'runs/b0_1_seed{seed}_2026-09-21/checkpoints/update_{u:03d}.pt'; m.load_state_dict(torch.load(p,map_location='cpu')['model']); return m
    def main():
        result={"schema":"b0_1_collapse_pair_audit_v1","read_only":True,"observation_shape":list(OBS.shape),"pairs":{}}
        for name,seed,a,b in PAIRS:
            ma,mb=load(seed,a),load(seed,b)
            with torch.no_grad(): aa=ma.act_inference(OBS); ab=mb.act_inference(OBS); mua=ma.raw_mean(OBS); mub=mb.raw_mean(OBS)
            d=ab-aa
            result['pairs'][name]={"seed":seed,"updates":[a,b],"first_action_l2_mean":float(d.norm(dim=1).mean()),"first_action_l2_p95":float(torch.quantile(d.norm(dim=1),.95)),"first_action_max_abs":float(d.abs().max()),"action_norm_mean_before":float(aa.norm(dim=1).mean()),"action_norm_mean_after":float(ab.norm(dim=1).mean()),"saturation_before":float((aa.abs()>=2.9).float().mean()),"saturation_after":float((ab.abs()>=2.9).float().mean()),"raw_mean_l2_mean":float((mub-mua).norm(dim=1).mean())}
        result['limitations']=["This artifact compares the first deterministic action on identical frozen reset observations.","The existing B0.1 artifacts contain no per-step simulator states or near-boundary trajectory replay; therefore first 10-20 physical actions, tilt growth, height, and contact cannot be reconstructed read-only.","No trajectory was generated and no training/configuration was changed."]
        (OUT/'collapse_pair_summary.json').write_text(json.dumps(result,indent=2)+"\n")
        lines=['# Collapse pair same-state audit','','First deterministic action comparison on identical frozen reset observations.','']
        for n,x in result['pairs'].items(): lines.append(f"- **{n}**: mean action ΔL2={x['first_action_l2_mean']:.4f}, p95={x['first_action_l2_p95']:.4f}, max |Δa|={x['first_action_max_abs']:.4f}, saturation {x['saturation_before']:.3f}→{x['saturation_after']:.3f}.")
        lines += ['', 'Near-boundary seed0 trajectory states and per-step physical traces were not present in the frozen artifacts, so those requested comparisons remain unmeasured.']
        (OUT/'collapse_pair_report.md').write_text('\n'.join(lines)+'\n'); print(OUT/'collapse_pair_report.md')
    if True: main()

def run_b0_1_learning_curve_audit():
    """Run former b0_1_learning_curve_audit.py stage."""
    """Did B0.1 ever learn deterministic locomotion, or never learn it at all?
    
    Read-only over the frozen monitor artifacts; never opens a simulator. Plots
    each acceptance metric against the training update for all three seeds and
    marks the final-gate thresholds, so a transient pass is visible as such.
    """
    import csv
    import json
    import math
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import ARTIFACTS, RUNS, OfflineAudit
    
    UPDATES = list(range(0, 501, 25))
    SEEDS = range(3)
    GATE = {"survival": .9, "vx_mae": .15, "tilt_p95_deg": 15, "tilt_max_deg": 30}
    METRICS = [("survival", "Survival"), ("first_fall_mean", "Mean first-fall step"),
               ("vx_mae", "vx MAE (m/s)"), ("displacement", "Mean displacement (m)"),
               ("tilt_p95_deg", "Tilt p95 (deg)"), ("tilt_max_deg", "Tilt max (deg)"),
               ("height_error", "Height error (m)"), ("std", "Scheduled std")]
    INTERPRETATION = {
        "answer": "B0.1 did learn deterministic capability transiently; it did not simply fail to learn locomotion.",
        "evidence": "seed1 passes every final-gate metric at update 25; seed0 passes every final-gate metric at updates 400, 425, and 450, then collapses at 475 while std is already fixed at 0.10.",
        "caveat": "seed2 never satisfies all metrics simultaneously, so the instability is seed-sensitive rather than a uniformly solved locomotion task.",
        "next_decision_scope": "Investigate read-only policy/training-stability evidence before any new formulation; do not retune reward or std from this audit alone."}
    
    
    def scheduled_std(u):
        """The std schedule B0.1 trained under: hold, then anneal over updates 100-400."""
        return .82 - .72 * min(max((u - 100) / 300, 0), 1)
    
    
    class B01LearningCurveAudit(OfflineAudit):
        """B0.1 deterministic learning-curve audit over the frozen monitor artifacts."""
    
        root = ARTIFACTS
        run = "b0_1_learning_curve_audit"
        report = "summary.json"
        sort_keys = True
        schema = "b0_1_learning_curve_audit_v1"
    
        def row(self, seed, update):
            a = json.loads((RUNS / f"b0_1_seed{seed}_2026-09-21" / "monitor"
                            / f"u{update:03d}.json").read_text())["acceptance"]
            row = {"seed": seed, "update": update, "std": scheduled_std(update),
                   "survival": a["survival_rate"],
                   "first_fall_mean": sum(a["first_fall"]) / len(a["first_fall"]),
                   "vx_mae": a["mean_abs_vx_error"],
                   "displacement": a["mean_displacement"],
                   "tilt_p95_deg": math.degrees(a["tilt_p95_rad"]),
                   "tilt_max_deg": math.degrees(a["tilt_max_rad"]),
                   "height_error": a["mean_height_error"]}
            row["gate_pass"] = (row["survival"] >= GATE["survival"]
                                and row["vx_mae"] <= GATE["vx_mae"]
                                and row["tilt_p95_deg"] <= GATE["tilt_p95_deg"]
                                and row["tilt_max_deg"] <= GATE["tilt_max_deg"])
            return row
    
        def analyze(self):
            self.rows = []
            summary = {"schema": self.schema, "read_only": True, "seeds": []}
            for seed in SEEDS:
                curve = [self.row(seed, u) for u in UPDATES]
                self.rows.extend(curve)
                passing = [r["update"] for r in curve if r["gate_pass"]]
                peak = max(curve, key=lambda r: (r["gate_pass"], r["survival"],
                                                 -r["vx_mae"], -r["tilt_p95_deg"]))
                collapse = (max(passing) + 25) if passing and max(passing) < 500 else None
                summary["seeds"].append({
                    "seed": seed, "full_gate_updates": passing, "best_update": peak["update"],
                    "best": peak, "collapse_after_last_full_gate": collapse,
                    "collapse_std_phase": ("post-anneal plateau" if collapse and collapse >= 425
                                           else "std hold/anneal" if collapse else None)})
            summary["interpretation"] = INTERPRETATION
            return summary
    
        def summarize(self, summary):
            with (self.out / "learning_curve.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            self.plot()
            lines = ["# B0.1 deterministic learning-curve audit", "",
                     "Read-only analysis of the frozen monitor artifacts.", "",
                     "## Finding", "", summary["interpretation"]["answer"], "",
                     summary["interpretation"]["evidence"], "",
                     summary["interpretation"]["caveat"], "",
                     "## Per-seed full-gate windows", ""]
            for item in summary["seeds"]:
                lines.append(f"- Seed {item['seed']}: {item['full_gate_updates'] or 'none'}; "
                             f"best update {item['best_update']}; collapse after last full gate: "
                             f"{item['collapse_after_last_full_gate']}.")
            (self.out / "report.md").write_text("\n".join(lines) + "\n")
            print(self.out)
    
        def plot(self):
            import matplotlib.pyplot as plt
    
            fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
            for ax, (key, label) in zip(axes.flat, METRICS):
                for seed in SEEDS:
                    r = [x for x in self.rows if x["seed"] == seed]
                    ax.plot([x["update"] for x in r], [x[key] for x in r],
                            marker="o", label=f"seed {seed}")
                if key in GATE:
                    ax.axhline(GATE[key], color="black", linestyle="--", linewidth=1)
                ax.set_title(label)
                ax.grid(alpha=.3)
                ax.legend(fontsize=8)
            for ax in axes[-1]:
                ax.set_xlabel("Training update")
            fig.tight_layout()
            fig.savefig(self.out / "learning_curve.png", dpi=160)
            plt.close(fig)
    
    
    if True:
        B01LearningCurveAudit.main()

def run_b0_1_stability_audit():
    """Run former b0_1_stability_audit.py stage."""
    """Read-only actor/policy-preservation audit over frozen B0.1 checkpoints."""
    import csv, json, math, os, subprocess, sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    RUNS = ROOT / "runs"
    OUT = ROOT / "artifacts" / "b0_1_stability_audit"
    UPDATES = list(range(0, 501, 25))
    
    def _nominalize(cfg):
        from talon_rl.assets.unitree_a1.a1 import TALON_A1_CFG
        cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None; cfg.scene.robot = TALON_A1_CFG.replace()
        cfg.events.randomize_payload_mass.params["mass_distribution_params"] = (0., 0.)
        cfg.events.randomize_payload_com.params["com_range"] = {"x": (0., 0.), "y": (0., 0.), "z": (0., 0.)}
        cfg.events.randomize_friction.params.update(static_friction_range=(1., 1.), dynamic_friction_range=(1., 1.), restitution_range=(0., 0.))
        cfg.events.randomize_motor_power.params.update(stiffness_distribution_params=(1., 1.), damping_distribution_params=(1., 1.))
        cfg.events.randomize_joint_range.params["scale_range"] = (1., 1.); cfg.events.push_robot = None
        cfg.curriculum.terrain_levels = None; cfg.terminations.obstacle_reached = None; cfg.episode_length_s = 22.; cfg.stand_phase_s = 0.
    
    def _install(base, states):
        ids = torch.arange(64, device=base.device); robot = base.scene["robot"]
        root = torch.from_numpy(states["root_state"]).to(base.device); pos = torch.from_numpy(states["joint_pos"]).to(base.device); vel = torch.from_numpy(states["joint_vel"]).to(base.device)
        robot.write_root_pose_to_sim(root[:, :7], env_ids=ids); robot.write_root_velocity_to_sim(root[:, 7:], env_ids=ids); robot.write_joint_state_to_sim(pos, vel, env_ids=ids); base.scene.write_data_to_sim(); base.sim.forward()
    
    def _flat_state(state):
        return torch.cat([v.detach().float().cpu().reshape(-1) for k, v in state.items() if torch.is_tensor(v) and v.is_floating_point()])
    
    def main():
        OUT.mkdir(parents=True, exist_ok=True); app = base = None
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym
            import talon_rl.tasks.locomotion.a1_env
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from rl.core.modules.actor_critic import ActorCritic
            cfg = IsaacLabTalonEnvCfg(); cfg.scene.num_envs = 64; cfg.seed = 17; cfg.sim.dt = .01; cfg.decimation = 1; _nominalize(cfg)
            base = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped; env = B0TalonEnv(base); env.reset(); env._command()
            with np.load(ROOT / "artifacts/b0_smoke/b0-monitor-frozen-reset-states.npz") as z: states = {k: z[k].copy() for k in z.files}
            _install(base, states); obs = base._transition(base.observation_manager.compute())["obs"]
            model = ActorCritic(base.obs_dim, base.obs_dim, base.action_dim, 1, [64, 64]).to(base.device); model.eval()
            records = []
            for seed in range(3):
                previous = None
                for update in UPDATES:
                    path = RUNS / f"b0_1_seed{seed}_2026-09-21/checkpoints/update_{update:03d}.pt"
                    state = torch.load(path, map_location=base.device); model.load_state_dict(state["model"])
                    with torch.no_grad():
                        x = torch.as_tensor(obs, device=base.device, dtype=torch.float32); mean = model.raw_mean(x).cpu(); action = model.act_inference(x).cpu()
                    params = _flat_state(state["model"])
                    row = {"seed": seed, "update": update, "action_saturation": float((action.abs() >= 2.9).float().mean()), "mean_norm": float(mean.norm()), "param_norm": float(params.norm())}
                    if previous is None:
                        row.update({"cos_mu_prev": None, "delta_mu_norm": None, "delta_param_norm": None, "relative_param_delta": None})
                    else:
                        prior_mean, prior_params = previous
                        row.update({"cos_mu_prev": float(torch.nn.functional.cosine_similarity(mean.reshape(1, -1), prior_mean.reshape(1, -1)).item()),
                                    "delta_mu_norm": float((mean - prior_mean).norm()), "delta_param_norm": float((params - prior_params).norm()),
                                    "relative_param_delta": float((params - prior_params).norm() / (prior_params.norm() + 1e-12))})
                    previous = (mean, params); records.append(row)
            # Join the already-recorded behavioral and scalar-loss values.
            for row in records:
                seed, update = row["seed"], row["update"]; run = RUNS / f"b0_1_seed{seed}_2026-09-21"
                b = json.loads((run / "monitor" / f"u{update:03d}.json").read_text())["acceptance"]
                train = json.loads((run / "training_metrics.json").read_text())["metrics"][max(update - 1, 0)]
                row.update({"survival": b["survival_rate"], "first_fall_mean": float(np.mean(b["first_fall"])), "vx_mae": b["mean_abs_vx_error"], "displacement": b["mean_displacement"], "tilt_p95_deg": math.degrees(b["tilt_p95_rad"]), "tilt_max_deg": math.degrees(b["tilt_max_rad"]), "value_loss": train["value_loss"], "policy_loss": train["policy_loss"], "reward_mean": train["reward_mean"], "std": .82 - .72 * min(max((update - 100) / 300, 0), 1)})
            with (OUT / "audit.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
            windows = {"seed1_25_to_50": [(1, 25), (1, 50)], "seed0_450_to_475": [(0, 450), (0, 475)], "seed2_negative_control": [(2, 250), (2, 275)]}
            selected = {name: [next(r for r in records if r["seed"] == s and r["update"] == u) for s, u in pair] for name, pair in windows.items()}
            summary = {"schema": "b0_1_stability_audit_v1", "read_only": True, "unavailable_from_frozen_artifacts": ["approx_kl", "ppo_ratio", "clip_fraction", "gradient_norm", "returns", "advantage_mean_std", "value_prediction_error"], "windows": selected,
                       "interpretation": {"seed0": "large behavior collapse after update 450→475 should be compared with actor/parameter drift; std is already 0.10 plateau.", "seed1": "capability collapse after update 25→50 occurs while std remains 0.82; variance schedule alone cannot explain it.", "seed2": "negative control: no full gate window."}}
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 2, figsize=(13, 9))
            for seed in range(3):
                r = [x for x in records if x["seed"] == seed]; x = [z["update"] for z in r]
                ax[0,0].plot(x, [z["cos_mu_prev"] if z["cos_mu_prev"] is not None else 1 for z in r], marker="o", label=f"seed {seed}")
                ax[0,1].plot(x, [z["relative_param_delta"] if z["relative_param_delta"] is not None else 0 for z in r], marker="o", label=f"seed {seed}")
                ax[1,0].plot(x, [z["survival"] for z in r], marker="o", label=f"seed {seed}")
                ax[1,1].plot(x, [z["vx_mae"] for z in r], marker="o", label=f"seed {seed}")
            for a, title in zip(ax.flat, ["cos(mu_k, mu_prev)", "relative parameter displacement", "survival", "vx MAE"]): a.set_title(title); a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel("update")
            fig.tight_layout(); fig.savefig(OUT / "stability_curve.png", dpi=160); plt.close(fig)
            (OUT / "report.md").write_text("# B0.1 stability / policy-preservation audit\n\nRead-only checkpoint audit.\n\nApproximate KL, PPO ratio, clip fraction, gradient norm, returns, advantages, and value prediction error were not recorded by the frozen runner and are therefore not inferred. See `summary.json` for the three collapse windows and `stability_curve.png` for actor/parameter/behavior trajectories.\n")
            print(OUT)
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    
    def main2():
        """Collect fixed-state actor traces through the already validated worker."""
        OUT.mkdir(parents=True, exist_ok=True); records = []
        worker_script = ROOT / "scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py"
        env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'scripts'}")
        for seed in range(3):
            run = RUNS / f"b0_1_seed{seed}_2026-09-21"; model_dir = run / "checkpoints"; result = OUT / f"actor_seed{seed}.json"
            first = model_dir / "update_000.pt"
            subprocess.run([sys.executable, str(worker_script), "--worker", str(first), str(result), "--reuse-frozen", "--actor-audit", str(model_dir)], cwd=ROOT, env=env, check=True)
            actor_rows = json.loads(result.read_text())["actor_rows"]
            train = json.loads((run / "training_metrics.json").read_text())["metrics"]
            for row in actor_rows:
                update = row["update"]; b = json.loads((run / "monitor" / f"u{update:03d}.json").read_text())["acceptance"]; t = train[max(update - 1, 0)]
                row.update({"seed": seed, "survival": b["survival_rate"], "first_fall_mean": float(np.mean(b["first_fall"])), "vx_mae": b["mean_abs_vx_error"], "displacement": b["mean_displacement"], "tilt_p95_deg": math.degrees(b["tilt_p95_rad"]), "tilt_max_deg": math.degrees(b["tilt_max_rad"]), "value_loss": t["value_loss"], "policy_loss": t["policy_loss"], "reward_mean": t["reward_mean"], "std": .82 - .72 * min(max((update - 100) / 300, 0), 1)})
                records.append(row)
        with (OUT / "audit.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        windows = {"seed1_25_to_50": [(1, 25), (1, 50)], "seed0_450_to_475": [(0, 450), (0, 475)], "seed2_negative_control": [(2, 250), (2, 275)]}
        selected = {name: [next(r for r in records if r["seed"] == s and r["update"] == u) for s, u in pair] for name, pair in windows.items()}
        summary = {"schema": "b0_1_stability_audit_v1", "read_only": True, "unavailable_from_frozen_artifacts": ["approx_kl", "ppo_ratio", "clip_fraction", "gradient_norm", "returns", "advantage_mean_std", "value_prediction_error"], "windows": selected, "interpretation": {"seed0": "compare actor/parameter drift at 450→475; std is already 0.10 plateau.", "seed1": "compare actor/parameter drift at 25→50 while std remains 0.82.", "seed2": "negative control: no full gate window."}}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        for seed in range(3):
            r = [x for x in records if x["seed"] == seed]; x = [z["update"] for z in r]
            ax[0,0].plot(x, [z["cos_mu_prev"] if z["cos_mu_prev"] is not None else 1 for z in r], marker="o", label=f"seed {seed}"); ax[0,1].plot(x, [z["relative_param_delta"] if z["relative_param_delta"] is not None else 0 for z in r], marker="o", label=f"seed {seed}"); ax[1,0].plot(x, [z["survival"] for z in r], marker="o", label=f"seed {seed}"); ax[1,1].plot(x, [z["vx_mae"] for z in r], marker="o", label=f"seed {seed}")
        for a, title in zip(ax.flat, ["cos(mu_k, mu_prev)", "relative parameter displacement", "survival", "vx MAE"]): a.set_title(title); a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel("update")
        fig.tight_layout(); fig.savefig(OUT / "stability_curve.png", dpi=160); plt.close(fig)
        (OUT / "report.md").write_text("# B0.1 stability / policy-preservation audit\n\nRead-only actor traces use the frozen 64 reset states through the validated monitor worker. KL, PPO ratio, clip fraction, gradient norm, returns, advantages, and value prediction error were not recorded by the frozen runner and are not inferred.\n")
        print(OUT)
    
    if True: main2()

def run_b0_1_stability_audit_offline():
    """Run former b0_1_stability_audit_offline.py stage."""
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
    
    from rl.core.diagnostics.offline_audit import ARTIFACTS, RUNS, OfflineAudit
    
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
    
    
    if True:
        B01StabilityAudit.main()

def run_b0_1_verdict():
    """Run former b0_1_verdict.py stage."""
    """Aggregate the frozen B0.1 gate; this script never retrains or retunes."""
    import hashlib, json, math
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    OUT = ROOT / "artifacts" / "b0_1_freeze"
    EXPECTED_MONITORS = list(range(0, 501, 25))
    
    def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
    def load(path: Path): return json.loads(path.read_text())
    
    def main() -> None:
        frozen = load(OUT / "FREEZE.json"); rows = []; early = []
        for seed in frozen["effective_config"]["seeds"]:
            run = ROOT / "runs" / f"b0_1_seed{seed}_2026-09-21"
            done, cfg, copied, train = (load(run / name) for name in ("RUN_DONE.json", "config.json", "freeze_manifest_copy.json", "training_metrics.json"))
            assert done["status"] == "RUN_DONE" and done["exit_code"] == 0 and done["final_update"] == 500
            assert copied == frozen and not (run / "ERROR.json").exists()
            assert cfg["num_envs"] == 4096 and cfg["updates"] == 500 and cfg["command"] == [0.5, 0.0, 0.0]
            assert len(train["metrics"]) == 500 and [item["update"] for item in train["monitor"]] == EXPECTED_MONITORS
            for index, item in enumerate(train["metrics"]):
                expected = .82 - .72 * min(max((index - 100) / 300, 0), 1)
                assert math.isclose(item["std"], expected, abs_tol=1e-12)
                assert all(math.isfinite(item[key]) for key in ("std", "reward_mean", "policy_loss", "value_loss"))
            monitors = {u: load(run / "monitor" / f"u{u:03d}.json") for u in EXPECTED_MONITORS}
            assert all(item["pass"] for item in monitors.values())
            m0, m250, m500 = (monitors[u]["acceptance"] for u in (0, 250, 500))
            final = {"survival_rate": m500["survival_rate"], "mean_abs_vx_error": m500["mean_abs_vx_error"],
                     "tilt_p95_deg": math.degrees(m500["tilt_p95_rad"]), "tilt_max_deg": math.degrees(m500["tilt_max_rad"])}
            gate = final["survival_rate"] >= .9 and final["mean_abs_vx_error"] <= .15 and final["tilt_p95_deg"] <= 15 and final["tilt_max_deg"] <= 30
            rows.append({"seed": seed, "run": str(run), "checkpoint_sha256": sha(run / "checkpoints" / "final.pt"), "final": final, "final_gate_pass": gate})
            early.append({"seed": seed, "survival_at_0": m0["survival_rate"], "survival_at_250": m250["survival_rate"], "improvement_pp": 100 * (m250["survival_rate"] - m0["survival_rate"])})
        early_stop = all(row["survival_at_250"] < .25 for row in early) and not any(row["improvement_pp"] >= 20 for row in early)
        verdict = {"schema": "b0_1_verdict_v1", "training_sanity_pass": True, "seeds": rows,
                   "early_review": {"condition_met": early_stop, "action": "record-only; all seeds already reached update 500", "seeds": early},
                   "final_decision": "PASS" if all(row["final_gate_pass"] for row in rows) else "FAIL",
                   "rule": "All seeds must pass update-500 deterministic gate; a failure keeps B0.2-B0.6 closed and requires a new formulation decision.",
                   "freeze_manifest_sha256": sha(OUT / "FREEZE.json")}
        (OUT / "B0_1_VERDICT.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(OUT / "B0_1_VERDICT.json")
    
    if True: main()

def run_freeze_b0_1():
    """Run former freeze_b0_1.py stage."""
    """Write B0.1's immutable formulation manifest from validated inputs."""
    
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path
    
    
    ROOT = Path(__file__).resolve().parents[4]
    OUT = ROOT / "artifacts" / "b0_1_freeze"
    MANIFEST = OUT / "FREEZE.json"
    FILES = {
        "reward": "talon_rl/rewards/baselines.py",
        "trainer": "scripts/rl/core/algorithms/scalar_ppo.py",
        "environment_wrapper": "talon_rl/wrappers/scalar_reward_env.py",
        "monitor": "scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py",
        "terminal_smoke": "scripts/rl/experiments/baselines/scalar_substrate/smoke_tests.py",
        "ppo_smoke": "scripts/rl/experiments/baselines/scalar_substrate/smoke_tests.py",
        "train_runner": "scripts/rl/experiments/common/utilities/train_b0.py",
        "freeze_record": "docs/baselines/scalar_substrate/b0-1-freeze-draft.md",
    }
    ARTIFACTS = {
        "terminal_semantics": "artifacts/b0_smoke/b0-terminal-smoke.json",
        "ppo_smoke": "artifacts/b0_smoke/b0-ppo-smoke.json",
        "monitor_isolation": "artifacts/b0_smoke/b0-monitor-isolation.json",
        "frozen_reset_states": "artifacts/b0_smoke/b0-monitor-frozen-reset-states.npz",
    }
    
    
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    
    
    def main() -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        paths = {name: ROOT / value for name, value in FILES.items()} | {name: ROOT / value for name, value in ARTIFACTS.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing: raise FileNotFoundError("freeze input missing: " + ", ".join(missing))
        manifest = {
            "schema": "b0_1_freeze_v1", "status": "FROZEN", "training_authorized": True,
            "git_head": run("git", "rev-parse", "HEAD"),
            "working_tree_porcelain": run("git", "status", "--porcelain").splitlines(),
            "sha256": {name: sha256(path) for name, path in paths.items()},
            "effective_config": {
                "seeds": [0, 1, 2], "num_envs": 4096, "updates": 500, "rollout_steps": 16, "hidden_dims": [64, 64],
                "command": [0.5, 0.0, 0.0], "sim_dt": 0.01, "decimation": 1,
                "terrain": "plane", "nominal_morphology": True, "domain_randomization": False,
                "preference_or_moppo": False, "reward_normalization": False,
                "reward": "exp(-((vx-0.5)/0.25)^2)-0.25*(roll^2+pitch^2)-2*(height-z_nominal)^2-10*terminal_base_contact",
                "exploration": {"mode": "scheduled_fixed_std", "initial": 0.82, "final": 0.10, "hold_updates": 100, "decay_updates": 300},
            },
            "monitor": {"updates": list(range(0, 501, 25)), "frozen_states": 64, "horizon": 500,
                        "policy": "tanh(actor_mean)", "separate_process": True},
            "decision_rules": {"early_review_update": 250, "final_checkpoint_update": 500,
                               "no_mid_run_reward_schedule_threshold_or_budget_changes": True},
            "python_executable": sys.executable,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(MANIFEST)
    
    
    if True: main()

STAGES = {
    "b0_1_collapse_pair_audit": run_b0_1_collapse_pair_audit,
    "b0_1_learning_curve_audit": run_b0_1_learning_curve_audit,
    "b0_1_stability_audit": run_b0_1_stability_audit,
    "b0_1_stability_audit_offline": run_b0_1_stability_audit_offline,
    "b0_1_verdict": run_b0_1_verdict,
    "freeze_b0_1": run_freeze_b0_1,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
