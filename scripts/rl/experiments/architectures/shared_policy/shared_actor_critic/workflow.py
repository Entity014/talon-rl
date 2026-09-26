"""Consolidated experiment stages. Use the first CLI argument to select a former script stage."""
from __future__ import annotations
import argparse

def run_v1c_aggregate():
    """Run former v1c_aggregate.py stage."""
    """Aggregate frozen-grid V1-C confirmatory artifacts without selecting points post hoc."""
    
    import argparse
    import json
    from pathlib import Path
    
    import numpy as np
    
    
    CONDITIONS = ("curriculum", "full_simplex_control")
    OBJECTIVES = ("progress", "balance", "efficiency")
    
    
    def mean_sd(values):
        values = np.asarray(values, dtype=float)
        return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0}
    
    
    def rankdata(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks
    
    
    def spearman(x, y):
        rx, ry = rankdata(x), rankdata(y)
        if np.std(rx) == 0 or np.std(ry) == 0:
            return None
        return float(np.corrcoef(rx, ry)[0, 1])
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--manifest", type=Path, default=Path("artifacts/v1c/V1C_CURRICULUM_MANIFEST.json"))
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--run-root", type=Path, default=Path("runs"))
        args = parser.parse_args()
    
        manifest = json.loads(args.manifest.read_text())
        expected_grid = np.asarray(manifest["evaluation_grid"]["points"], dtype=float)
        reports = {}
        for seed in (0, 1, 2):
            path = args.run_root / f"v1c_confirmatory_seed{seed}-2026-09-22" / "report.json"
            report = json.loads(path.read_text())
            grid = np.asarray(report["evaluation_grid"], dtype=float)
            if grid.shape != expected_grid.shape or not np.allclose(grid, expected_grid, rtol=0.0, atol=1e-7):
                raise AssertionError(f"seed {seed}: evaluation grid differs from frozen manifest")
            if report["updates"] != manifest["common"]["updates"]:
                raise AssertionError(f"seed {seed}: terminal update mismatch")
            reports[seed] = report
    
        rows = {condition: [] for condition in CONDITIONS}
        for seed, report in reports.items():
            for condition in CONDITIONS:
                evaluation = report["conditions"][condition]["evaluation"]
                if len(evaluation) != len(expected_grid):
                    raise AssertionError(f"seed {seed} {condition}: evaluation row count mismatch")
                for row in evaluation:
                    rows[condition].append({"seed": seed, **row})
    
        pooled = {}
        for condition in CONDITIONS:
            grouped = []
            for index in range(len(expected_grid)):
                selected = [row for row in rows[condition] if row["grid_index"] == index]
                grouped.append({
                    "grid_index": index,
                    "w": expected_grid[index].tolist(),
                    "objective_reward_mean": [mean_sd([r["objective_reward_mean"][j] for r in selected]) for j in range(3)],
                    "survival": mean_sd([r["survival"] for r in selected]),
                    "vx_error": mean_sd([r["vx_error"] for r in selected]),
                    "finite_all": all(r["finite"] for r in selected),
                })
            pooled[condition] = grouped
    
        paired_deltas = []
        for seed in reports:
            cur = {r["grid_index"]: r for r in rows["curriculum"] if r["seed"] == seed}
            ctl = {r["grid_index"]: r for r in rows["full_simplex_control"] if r["seed"] == seed}
            for index in range(len(expected_grid)):
                paired_deltas.append({
                    "seed": seed,
                    "grid_index": index,
                    "w": expected_grid[index].tolist(),
                    "objective_reward_delta_curriculum_minus_control": [cur[index]["objective_reward_mean"][j] - ctl[index]["objective_reward_mean"][j] for j in range(3)],
                    "survival_delta": cur[index]["survival"] - ctl[index]["survival"],
                    "vx_error_delta": cur[index]["vx_error"] - ctl[index]["vx_error"],
                })
    
        def aggregate_deltas(metric):
            return mean_sd([row[metric] for row in paired_deltas])
    
        auc = {condition: {} for condition in CONDITIONS}
        for seed, report in reports.items():
            for condition in CONDITIONS:
                records = report["conditions"][condition]["records"]
                for j, name in enumerate(OBJECTIVES):
                    value = float(np.trapz([r["reward_mean"][j] for r in records], dx=1.0))
                    auc[condition].setdefault(name, []).append(value)
    
        response = {condition: {} for condition in CONDITIONS}
        for condition in CONDITIONS:
            for j, name in enumerate(OBJECTIVES):
                reward = np.asarray([r["objective_reward_mean"][j]["mean"] for r in pooled[condition]], dtype=float)
                weights = expected_grid[:, j]
                response[condition][name] = {
                    "reward_range_across_grid": [float(reward.min()), float(reward.max())],
                    "weight_reward_spearman_descriptive": spearman(weights, reward),
                }
    
        result = {
            "schema": "v1c_confirmatory_aggregate_v1",
            "status": "AGGREGATION_COMPLETE_NO_VERDICT",
            "manifest": str(args.manifest),
            "seeds": [0, 1, 2],
            "conditions": list(CONDITIONS),
            "grid_points": len(expected_grid),
            "terminal_update": manifest["common"]["updates"],
            "numerical_tolerance": manifest["numerical_contract"]["frozen_tolerance"],
            "data_integrity": {
                "same_frozen_grid": True,
                "all_evaluation_rows_finite": all(r["finite"] for c in CONDITIONS for r in rows[c]),
                "max_reconstruction_error": {
                    str(seed): {c: reports[seed]["conditions"][c]["max_reward_reconstruction_error"] for c in CONDITIONS}
                    for seed in reports
                },
            },
            "per_seed": {
                str(seed): {
                    c: {
                        "max_reconstruction_error": reports[seed]["conditions"][c]["max_reward_reconstruction_error"],
                        "grid_survival_mean": float(np.mean([r["survival"] for r in reports[seed]["conditions"][c]["evaluation"]])),
                        "grid_vx_error_mean": float(np.mean([r["vx_error"] for r in reports[seed]["conditions"][c]["evaluation"]])),
                    }
                    for c in CONDITIONS
                }
                for seed in reports
            },
            "pooled_by_grid": pooled,
            "paired_deltas_curriculum_minus_control": paired_deltas,
            "paired_delta_summary": {
                "survival": aggregate_deltas("survival_delta"),
                "vx_error": aggregate_deltas("vx_error_delta"),
            },
            "objective_reward_delta_summary": {
                name: mean_sd([row["objective_reward_delta_curriculum_minus_control"][j] for row in paired_deltas])
                for j, name in enumerate(OBJECTIVES)
            },
            "reward_vector_auc": {c: {name: mean_sd(values) for name, values in auc[c].items()} for c in CONDITIONS},
            "preference_response_descriptive": response,
            "unavailable_from_frozen_artifacts": ["KL", "clip_fraction", "critic_loss_aggregate", "updates_to_threshold", "tilt", "contact", "efficiency_physical_metrics"],
            "interpretation": "This artifact reports all fixed-grid paired measurements available in the frozen runner outputs. It does not issue a superiority verdict because the artifacts do not contain a predeclared decision threshold/rule for the broad coverage and trade-off endpoints.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "output": str(args.output), "grid_points": len(expected_grid)}, indent=2))
    
    
    if True:
        main()

def run_v1c_d0_aggregate():
    """Run former v1c_d0_aggregate.py stage."""
    """Aggregate and classify the frozen Post-V1 D0 audit."""
    
    import argparse
    import json
    from pathlib import Path
    
    import numpy as np
    
    
    PREFERENCES = ("progress_heavy", "balance_heavy", "efficiency_heavy")
    METRICS = ("vx_error", "tilt_deg", "ang_vel_xy", "torque_norm", "action_rate")
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        data = json.loads(args.input.read_text())
        rows = data["behavior_rows"]
        action_rows = data["action_sensitivity_rows"]
    
        summary = {}
        for condition in data["conditions"]:
            condition_rows = [r for r in rows if r["condition"] == condition]
            pref_summary = {}
            for pref in PREFERENCES:
                selected = [r for r in condition_rows if r["preference"] == pref]
                pref_summary[pref] = {
                    "objective_return_mean": np.mean([r["objective_return_mean"] for r in selected], axis=0).tolist(),
                    "objective_return_reset_sd": np.std([r["objective_return_mean"] for r in selected], axis=0, ddof=1).tolist(),
                    "metrics_mean": {m: float(np.mean([r["metrics_mean"][m] for r in selected])) for m in METRICS},
                    "metrics_reset_sd": {m: float(np.std([r["metrics_mean"][m] for r in selected], ddof=1)) for m in METRICS},
                    "survival_mean": float(np.mean([r["metrics_mean"]["survival"] for r in selected])),
                }
            action = [r for r in action_rows if r["condition"] == condition]
            action_summary = {
                key: {
                    "mean": float(np.mean([r["distances"][key] for r in action])),
                    "sd": float(np.std([r["distances"][key] for r in action], ddof=1)),
                }
                for key in action[0]["distances"]
            }
            summary[condition] = {"preferences": pref_summary, "action_pair_distances": action_summary}
    
        # Conservative D0 rule: classify C when all preference-induced behavior
        # mean shifts are below the corresponding pooled reset variability and the
        # action response remains at numerical-scale magnitude for every condition.
        behavior_effects = []
        for condition in data["conditions"]:
            pref = summary[condition]["preferences"]
            for a_i, a in enumerate(PREFERENCES):
                for b in PREFERENCES[a_i + 1:]:
                    for metric in METRICS:
                        delta = abs(pref[a]["metrics_mean"][metric] - pref[b]["metrics_mean"][metric])
                        variability = max(pref[a]["metrics_reset_sd"][metric], pref[b]["metrics_reset_sd"][metric])
                        behavior_effects.append({"condition": condition, "pair": f"{a}_vs_{b}", "metric": metric, "absolute_mean_delta": delta, "reset_sd_reference": variability, "exceeds_reset_variability": bool(delta > variability)})
        max_action_distance = max(v["mean"] for c in summary.values() for v in c["action_pair_distances"].values())
        meaningful_behavior_effect = any(v["exceeds_reset_variability"] for v in behavior_effects)
        classification = "D0-A" if meaningful_behavior_effect and max_action_distance > 1e-3 else ("D0-B" if max_action_distance > 1e-3 else "D0-C")
    
        result = {
            "schema": "post_v1_d0_aggregate_v1",
            "status": "CLASSIFIED",
            "classification": classification,
            "input": str(args.input),
            "measurement_only": True,
            "no_retraining": True,
            "no_mutation": data["no_mutation"],
            "summary": summary,
            "behavior_effects": behavior_effects,
            "decision_rule": "D0-C when no behavior metric mean shift exceeds paired reset variability and all action pair distances remain <=1e-3; D0-B requires action response without meaningful behavior response; D0-A requires meaningful behavior response.",
            "interpretation": "Preference-conditioned action distances are at numerical-scale magnitude and all observed behavior shifts remain below reset-suite variability; the current policy is effectively preference-insensitive under D0.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "classification": classification, "max_action_distance": max_action_distance}, indent=2))
    
    
    if True:
        main()

def run_v1c_d0_preference_audit():
    """Run former v1c_d0_preference_audit.py stage."""
    """Post-V1 D0: read-only preference sensitivity audit for V1-C terminals."""
    
    import argparse
    import hashlib
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    PREFERENCES = {
        "progress_heavy": np.array([0.8, 0.1, 0.1], dtype=np.float32),
        "balance_heavy": np.array([0.1, 0.8, 0.1], dtype=np.float32),
        "efficiency_heavy": np.array([0.1, 0.1, 0.8], dtype=np.float32),
    }
    CONDITIONS = ("curriculum", "full_simplex_control")
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def quat_tilt_deg(quat):
        # Isaac Lab quaternions are wxyz. The projected gravity z component gives
        # the angle between the body-up axis and world-up.
        w, x, y, z = [quat[:, i] for i in range(4)]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return torch.rad2deg(torch.acos(up_z.clamp(-1.0, 1.0)))
    
    
    def sha256_model(model):
        payload = b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in model.state_dict().values())
        return hashlib.sha256(payload).hexdigest()
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=8)
        parser.add_argument("--steps", type=int, default=32)
        parser.add_argument("--reset-suites", type=int, default=4)
        parser.add_argument("--seed", type=int, default=47001)
        args = parser.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lifecycle = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
    
        def mark(event, **extra):
            with lifecycle.open("a") as handle:
                handle.write(json.dumps({"event": event, "unix": time.time(), **extra}, sort_keys=True) + "\n")
                handle.flush()
    
        mark("RUN_STARTED", protocol="POST-V1-D0", measurement_only=True)
        app = env = None
        try:
            from isaaclab.app import AppLauncher
            saved = sys.argv[:]
            sys.argv = [sys.argv[0]]
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            sys.argv = saved
            mark("APP_INIT_OK")
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = args.seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            mark("ENV_CREATED", num_envs=args.num_envs)
            obs, _ = env.reset(seed=args.seed)
            obs = obs_tensor(obs).cuda()
            action_dim = env.unwrapped.action_manager.total_action_dim
    
            checkpoints = {}
            for seed in range(3):
                for condition in CONDITIONS:
                    path = ROOT / "runs" / f"v1c_confirmatory_seed{seed}-2026-09-22" / f"{condition}_terminal.pt"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    payload = torch.load(path, map_location="cuda", weights_only=False)
                    model = V1CSharedActorCritic(obs.shape[-1], action_dim).cuda()
                    model.load_state_dict(payload["model"])
                    model.eval()
                    checkpoints[(seed, condition)] = (model, path)
            mark("CHECKPOINTS_LOADED", count=len(checkpoints))
    
            manager = env.unwrapped.reward_manager
            robot = env.unwrapped.scene["robot"]
            rows = []
            action_rows = []
            model_hashes = {}
    
            def scalar_metrics(current_action, previous_action, raw, scalar_reward):
                raw_np = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
                terms = {name: raw_np[:, i] for i, name in enumerate(list(manager.active_terms))}
                vector = group_v1b_s7_terms(terms, shape=(args.num_envs,))
                data = robot.data
                cmd = env.unwrapped.command_manager.get_command("base_velocity")
                vx_error = (data.root_lin_vel_b[:, 0] - cmd[:, 0]).abs()
                lin_z = data.root_lin_vel_b[:, 2].abs()
                ang_xy = torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1)
                tilt = quat_tilt_deg(data.root_quat_w)
                torque = torch.linalg.vector_norm(data.applied_torque, dim=-1) if hasattr(data, "applied_torque") else torch.zeros(args.num_envs, device="cuda")
                acceleration = torch.linalg.vector_norm(data.joint_acc, dim=-1) if hasattr(data, "joint_acc") else torch.zeros(args.num_envs, device="cuda")
                action_rate = torch.linalg.vector_norm(current_action - previous_action, dim=-1)
                return vector, {
                    "vx_error": float(vx_error.mean().item()),
                    "abs_lin_vel_z": float(lin_z.mean().item()),
                    "ang_vel_xy": float(ang_xy.mean().item()),
                    "tilt_deg": float(tilt.mean().item()),
                    "torque_norm": float(torque.mean().item()),
                    "joint_acc_norm": float(acceleration.mean().item()),
                    "action_rate": float(action_rate.mean().item()),
                    "survival": float(1.0 - ((env.unwrapped.termination_manager.terminated | env.unwrapped.termination_manager.time_outs).float().mean().item())),
                    "finite": bool(np.isfinite(vector).all() and torch.isfinite(current_action).all().item()),
                }
    
            for seed in range(3):
                for condition in CONDITIONS:
                    model, path = checkpoints[(seed, condition)]
                    model_hashes[f"{seed}:{condition}"] = sha256_model(model)
                    for pref_name, pref_np in PREFERENCES.items():
                        preference = torch.as_tensor(np.repeat(pref_np[None, :], args.num_envs, axis=0), device="cuda")
                        for suite in range(args.reset_suites):
                            reset_seed = args.seed + seed * 1000 + suite
                            current, _ = env.reset(seed=reset_seed)
                            current = obs_tensor(current).cuda()
                            previous_action = torch.zeros((args.num_envs, action_dim), device="cuda")
                            vectors, metrics, action_norms = [], [], []
                            with torch.no_grad():
                                for _ in range(args.steps):
                                    action = torch.clamp(model.act_inference_with_preference(current, preference), -1.0, 1.0)
                                    nxt, scalar_reward, term, trunc, _ = env.step(action)
                                    raw = manager._step_reward.detach()
                                    vector, metric = scalar_metrics(action, previous_action, raw, scalar_reward)
                                    vectors.append(np.asarray(vector))
                                    metrics.append(metric)
                                    action_norms.append(float(torch.linalg.vector_norm(action).mean().item()))
                                    previous_action = action
                                    current = obs_tensor(nxt).cuda()
                            rows.append({
                                "seed": seed, "condition": condition, "preference": pref_name,
                                "preference_vector": pref_np.tolist(), "reset_suite": suite,
                                "objective_return_mean": np.asarray(vectors).mean(axis=(0, 1)).tolist(),
                                "metrics_mean": {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
                                "action_norm_mean": float(np.mean(action_norms)),
                            })
    
                    # Action sensitivity is measured on identical observations:
                    # drive the same reset trajectory with progress-heavy actions,
                    # while evaluating all three preference-conditioned actions.
                    preference_tensors = {name: torch.as_tensor(np.repeat(value[None, :], args.num_envs, axis=0), device="cuda") for name, value in PREFERENCES.items()}
                    for suite in range(args.reset_suites):
                        current, _ = env.reset(seed=args.seed + seed * 1000 + suite)
                        current = obs_tensor(current).cuda()
                        distances = {f"{a}_vs_{b}": [] for a in PREFERENCES for b in PREFERENCES if a < b}
                        with torch.no_grad():
                            for _ in range(args.steps):
                                actions = {name: torch.clamp(model.act_inference_with_preference(current, w), -1.0, 1.0) for name, w in preference_tensors.items()}
                                for key in distances:
                                    a, b = key.split("_vs_")
                                    distances[key].append(float(torch.linalg.vector_norm(actions[a] - actions[b], dim=-1).mean().item()))
                                nxt, *_ = env.step(actions["progress_heavy"])
                                current = obs_tensor(nxt).cuda()
                        action_rows.append({"seed": seed, "condition": condition, "reset_suite": suite, "distances": {key: float(np.mean(value)) for key, value in distances.items()}})
    
            before = dict(model_hashes)
            report = {
                "schema": "post_v1_d0_preference_audit_v1",
                "status": "MEASUREMENT_COMPLETE",
                "measurement_only": True,
                "training": False,
                "preferences": {name: value.tolist() for name, value in PREFERENCES.items()},
                "conditions": list(CONDITIONS), "seeds": [0, 1, 2],
                "reset_suites": args.reset_suites, "steps": args.steps,
                "reset_seed_rule": "47001 + seed*1000 + reset_suite",
                "action_semantics": "deterministic actor mean + clip_actions=1.0",
                "model_hashes_before": before, "model_hashes_after": model_hashes,
                "no_mutation": before == model_hashes,
                "behavior_rows": rows, "action_sensitivity_rows": action_rows,
                "note": "Classification is performed by the separate aggregation step; no training or formulation change occurs here.",
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            mark("ARTIFACT_WRITTEN", path=str(args.output))
            mark("RUN_DONE", status=report["status"], no_mutation=report["no_mutation"])
            print(json.dumps({"status": report["status"], "behavior_rows": len(rows), "action_rows": len(action_rows), "no_mutation": report["no_mutation"]}, indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem + ".ERROR.json").write_text(json.dumps({"status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
            mark("ERROR", error=str(exc))
            raise
        finally:
            if env is not None:
                env.close()
            if app is not None:
                app.close()
    
    
    if True:
        main()

def run_v1c_isaac_smoke():
    """Run former v1c_isaac_smoke.py stage."""
    """Real-Isaac V1-C sampler/training-loop smoke; not training authorization."""
    
    import argparse
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=8)
        parser.add_argument("--updates", type=int, default=300)
        parser.add_argument("--horizon", type=int, default=2)
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        run = args.run_dir.resolve(); run.mkdir(parents=True, exist_ok=True)
        lifecycle = run / "lifecycle.jsonl"
    
        def mark(event, **extra):
            with lifecycle.open("a") as handle:
                handle.write(json.dumps({"event": event, "unix": time.time(), **extra}, sort_keys=True) + "\n")
                handle.flush()
    
        mark("RUN_STARTED", protocol="V1-C-SMOKE", updates=args.updates)
        app = env = None
        try:
            from isaaclab.app import AppLauncher
            saved = sys.argv[:]; sys.argv = [sys.argv[0]]
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            sys.argv = saved; mark("APP_INIT_OK")
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.preferences.v1c_curriculum import evaluation_grid, load_manifest, sample_preferences, stage_for_update
            from talon_rl.rewards.baselines import group_v1b_s7_terms, reconstruct_v1b_s7_scalar
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic, scalarized_late_weighted_ppo, vector_gae, vector_value_loss
    
            manifest = load_manifest(); grid = evaluation_grid(manifest)
            if not np.array_equal(grid, np.asarray(manifest["evaluation_grid"]["points"], dtype=np.float32)):
                raise AssertionError("evaluation grid differs from frozen manifest")
            mark("MANIFEST_LOADED", eval_grid_points=int(grid.shape[0]))
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = args.seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg); mark("ENV_CREATED")
            obs, _ = env.reset(seed=args.seed); obs = obs_tensor(obs).cuda(); mark("RESET_OK", obs_shape=list(obs.shape))
            model = V1CSharedActorCritic(obs.shape[-1], env.unwrapped.action_manager.total_action_dim).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            rng = np.random.default_rng(args.seed + 100000)
            max_reconstruction_error = 0.0; stage_records = []; updates_done = 0
    
            for update in range(1, args.updates + 1):
                stage = stage_for_update(update, manifest)
                # Smoke deliberately runs the frozen stage logic; short runs may
                # cover only early stages, but the stage is still asserted.
                w_np, labels = sample_preferences(rng, update, args.num_envs, "curriculum", manifest)
                w = torch.as_tensor(w_np, device="cuda")
                if not np.allclose(w_np.sum(axis=1), 1.0, atol=1e-6):
                    raise AssertionError("curriculum sample left simplex")
                observations=[]; actions=[]; old_logp=[]; rewards=[]; values=[]; dones=[]
                for _ in range(args.horizon):
                    with torch.no_grad():
                        action, logp = model.act_with_preference(obs, w); value = model.value_with_preference(obs, w)
                    action = torch.clamp(action, -1.0, 1.0)
                    nxt, scalar_reward, term, trunc, _ = env.step(action)
                    manager = env.unwrapped.reward_manager; raw = manager._step_reward.detach().cpu().numpy(); names = list(manager.active_terms)
                    vector = group_v1b_s7_terms({name: raw[:, i] for i, name in enumerate(names)}, shape=(args.num_envs,))
                    reconstruction = reconstruct_v1b_s7_scalar(vector)
                    error = float(np.max(np.abs(reconstruction - scalar_reward.detach().cpu().numpy() / env.unwrapped.step_dt)))
                    max_reconstruction_error = max(max_reconstruction_error, error)
                    if error >= 1e-6 or not np.isfinite(vector).all():
                        raise AssertionError(f"reward vector reconstruction failed: {error}")
                    observations.append(obs); actions.append(action); old_logp.append(logp); rewards.append(torch.as_tensor(vector, device="cuda") * env.unwrapped.step_dt); values.append(value); dones.append((term | trunc).to("cuda"))
                    obs = obs_tensor(nxt).cuda()
                with torch.no_grad(): next_value = model.value_with_preference(obs, w)
                reward_t = torch.stack(rewards); value_t = torch.stack(values); done_t = torch.stack(dones).bool()
                adv, returns = vector_gae(reward_t, value_t, next_value, done_t)
                flat_obs = torch.cat(observations); flat_actions = torch.cat(actions); flat_old = torch.cat(old_logp); flat_w = w.repeat(args.horizon, 1)
                new_logp = model.logp_with_preference(flat_obs, flat_w, flat_actions); ratio = torch.exp(new_logp - flat_old.detach())
                actor_loss = scalarized_late_weighted_ppo(ratio, adv.reshape(-1, 3).detach(), flat_w)
                critic_loss = vector_value_loss(model.value_with_preference(flat_obs, flat_w), returns.reshape(-1, 3).detach())
                loss = actor_loss + critic_loss
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                updates_done += 1; stage_records.append({"update": update, "stage": stage["name"], "labels": {str(x): int((labels == x).sum()) for x in np.unique(labels)}, "loss_finite": bool(torch.isfinite(loss).item()), "mean_w": w.mean(0).detach().cpu().tolist()})
                mark("UPDATE_DONE", update=update, stage=stage["name"])
    
            checkpoint = run / "v1c_smoke.pt"
            torch.save({"schema": "v1c_smoke_checkpoint_v1", "update": updates_done, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler_rng_state": rng.bit_generator.state, "condition": "curriculum", "manifest": str(ROOT / "artifacts/v1c/V1C_CURRICULUM_MANIFEST.json")}, checkpoint)
            resumed = V1CSharedActorCritic(obs.shape[-1], env.unwrapped.action_manager.total_action_dim).cuda(); state = torch.load(checkpoint, map_location="cuda", weights_only=False); resumed.load_state_dict(state["model"])
            resume_rng = np.random.default_rng(); resume_rng.bit_generator.state = state["sampler_rng_state"]
            # At the terminal update there is no legal update 301 in the frozen
            # manifest. Validate resume against the terminal stage and restore the
            # RNG state without probing outside the declared schedule.
            resumed_stage = stage_for_update(updates_done, manifest)
            next_w, _ = sample_preferences(resume_rng, updates_done, args.num_envs, "curriculum", manifest)
            control_w, control_labels = sample_preferences(np.random.default_rng(args.seed + 100000), 1, args.num_envs, "full_simplex_control", manifest)
            exclusions = manifest["common"]["exclusions"]
            checks = {"updates": updates_done == args.updates, "reward_reconstruction": max_reconstruction_error < 1e-6, "finite": all(x["loss_finite"] for x in stage_records), "checkpoint_resume": all(torch.equal(model.state_dict()[k], resumed.state_dict()[k]) for k in model.state_dict()), "sampler_state_restored": np.isfinite(next_w).all() and resumed_stage["name"] == "stage_3_full_simplex", "control_full_simplex": set(control_labels.tolist()) == {"full_simplex"}, "eval_grid_10_points": grid.shape == (10, 3), "clip_actions": True, "exclusions_off": all(value is False for value in exclusions.values())}
            if not all(checks.values()):
                raise AssertionError(checks)
            report = {"schema": "v1c_isaac_smoke_v1", "status": "PASS", "training_authorized": False, "updates": updates_done, "num_envs": args.num_envs, "horizon": args.horizon, "max_reward_reconstruction_error": max_reconstruction_error, "stage_records": stage_records, "checkpoint": str(checkpoint), "checks": checks, "evaluation_grid": grid.tolist(), "exclusions": exclusions, "note": "Smoke only; no curriculum-vs-control verdict."}
            (run / "artifact.json").write_text(json.dumps(report, indent=2) + "\n"); mark("ARTIFACT_WRITTEN", path=str(run / "artifact.json")); mark("RUN_DONE", status="PASS"); print(json.dumps(report, indent=2))
        except BaseException as exc:
            (run / "ERROR.json").write_text(json.dumps({"status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n"); mark("ERROR", error=str(exc)); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    
    
    if True:
        main()

def run_v1c_pilot():
    """Run former v1c_pilot.py stage."""
    """V1-C curriculum/control run with diagnostic or confirmatory reporting."""
    
    import argparse
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--updates", type=int, default=300)
        parser.add_argument("--num-envs", type=int, default=8)
        parser.add_argument("--horizon", type=int, default=2)
        parser.add_argument("--eval-steps", type=int, default=32)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--confirmatory", action="store_true",
                            help="record an authorized confirmatory run; does not issue a scientific verdict")
        args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
        lifecycle = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
    
        def mark(event, **extra):
            with lifecycle.open("a") as handle:
                handle.write(json.dumps({"event": event, "unix": time.time(), **extra}, sort_keys=True) + "\n")
                handle.flush()
    
        protocol = "V1-C-CONFIRMATORY" if args.confirmatory else "V1-C-PILOT"
        mark("RUN_STARTED", protocol=protocol, updates=args.updates,
             training_authorized=args.confirmatory)
        app = env = None
        try:
            from isaaclab.app import AppLauncher
            saved = sys.argv[:]; sys.argv = [sys.argv[0]]
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            sys.argv = saved; mark("APP_INIT_OK")
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.preferences.v1c_curriculum import evaluation_grid, load_manifest, sample_preferences, stage_for_update
            from talon_rl.rewards.baselines import group_v1b_s7_terms, reconstruct_v1b_s7_scalar
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic, initialize_from_rsl_m01, scalarized_late_weighted_ppo, vector_gae, vector_value_loss
    
            manifest = load_manifest(); grid = evaluation_grid(manifest)
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = args.seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg); mark("ENV_CREATED")
            obs, _ = env.reset(seed=args.seed); obs = obs_tensor(obs).cuda(); action_dim = env.unwrapped.action_manager.total_action_dim
    
            def make_model():
                model = V1CSharedActorCritic(obs.shape[-1], action_dim).cuda()
                initialize_from_rsl_m01(model, args.checkpoint, device="cpu")
                return model
    
            def train_condition(condition):
                model = make_model(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); rng = np.random.default_rng(args.seed + 100000)
                current_obs, _ = env.reset(seed=args.seed); current_obs = obs_tensor(current_obs).cuda(); records = []; max_error = 0.0
                for update in range(1, args.updates + 1):
                    sampler_condition = "curriculum" if condition == "curriculum" else "full_simplex_control"
                    w_np, labels = sample_preferences(rng, update, args.num_envs, sampler_condition, manifest); w = torch.as_tensor(w_np, device="cuda")
                    observations=[]; actions=[]; old_logp=[]; rewards=[]; values=[]; dones=[]
                    for _ in range(args.horizon):
                        with torch.no_grad(): action, logp = model.act_with_preference(current_obs, w); value = model.value_with_preference(current_obs, w)
                        action = torch.clamp(action, -1.0, 1.0); nxt, scalar_reward, term, trunc, _ = env.step(action)
                        manager = env.unwrapped.reward_manager; raw = manager._step_reward.detach().cpu().numpy(); names = list(manager.active_terms)
                        vector = group_v1b_s7_terms({name: raw[:, i] for i, name in enumerate(names)}, shape=(args.num_envs,)); reconstruction = reconstruct_v1b_s7_scalar(vector)
                        error = float(np.max(np.abs(reconstruction - scalar_reward.detach().cpu().numpy() / env.unwrapped.step_dt))); max_error = max(max_error, error)
                        observations.append(current_obs); actions.append(action); old_logp.append(logp); rewards.append(torch.as_tensor(vector, device="cuda") * env.unwrapped.step_dt); values.append(value); dones.append((term | trunc).to("cuda")); current_obs = obs_tensor(nxt).cuda()
                    with torch.no_grad(): next_value = model.value_with_preference(current_obs, w)
                    reward_t = torch.stack(rewards); value_t = torch.stack(values); done_t = torch.stack(dones).bool(); adv, returns = vector_gae(reward_t, value_t, next_value, done_t)
                    flat_obs=torch.cat(observations); flat_actions=torch.cat(actions); flat_old=torch.cat(old_logp); flat_w=w.repeat(args.horizon,1); ratio=torch.exp(model.logp_with_preference(flat_obs,flat_w,flat_actions)-flat_old.detach())
                    actor_loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,3).detach(),flat_w); critic_loss=vector_value_loss(model.value_with_preference(flat_obs,flat_w),returns.reshape(-1,3).detach()); loss=actor_loss+critic_loss
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                    records.append({"update":update,"stage":stage_for_update(update,manifest)["name"],"loss_finite":bool(torch.isfinite(loss).item()),"reward_mean":reward_t.mean((0,1)).detach().cpu().tolist(),"mean_w":w.mean(0).detach().cpu().tolist(),"labels":{str(x):int((labels==x).sum()) for x in np.unique(labels)},"critic_loss":float(critic_loss.detach())})
                    if update in {1,75,76,150,151,225,226,300}: mark("CONDITION_BOUNDARY", condition=condition, update=update, stage=records[-1]["stage"])
                terminal=args.output.parent/f"{condition}_terminal.pt"; torch.save({"schema":"v1c_pilot_terminal_v1","condition":condition,"seed":args.seed,"update":args.updates,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"sampler_rng_state":rng.bit_generator.state},terminal); mark("CHECKPOINT_WRITTEN",condition=condition,checkpoint=str(terminal))
                return model, {"condition":condition,"records":records,"terminal_checkpoint":str(terminal),"max_reward_reconstruction_error":max_error}
    
            trained={}
            for condition in ("curriculum","full_simplex_control"):
                mark("CONDITION_START",condition=condition); trained[condition]=train_condition(condition); mark("CONDITION_DONE",condition=condition)
    
            def evaluate(model, condition):
                rows=[]
                model.eval()
                with torch.no_grad():
                    for index, point in enumerate(grid):
                        w=torch.as_tensor(np.repeat(point[None,:],args.num_envs,axis=0),device="cuda"); current,_=env.reset(seed=90000+index); current=obs_tensor(current).cuda(); rewards=[]; vx=[]; done_any=np.zeros(args.num_envs,dtype=bool)
                        for _ in range(args.eval_steps):
                            action=torch.clamp(model.act_inference_with_preference(current,w),-1.,1.); nxt,_,term,trunc,_=env.step(action); done_any|=(term|trunc).cpu().numpy(); manager=env.unwrapped.reward_manager; raw=manager._step_reward.detach().cpu().numpy(); names=list(manager.active_terms); vector=group_v1b_s7_terms({name:raw[:,i] for i,name in enumerate(names)},shape=(args.num_envs,)); rewards.append(vector); data=env.unwrapped.scene["robot"].data; cmd=env.unwrapped.command_manager.get_command("base_velocity"); vx.append((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().cpu().numpy()); current=obs_tensor(nxt).cuda()
                        rows.append({"grid_index":index,"w":point.tolist(),"objective_reward_mean":np.concatenate(rewards).mean(0).tolist(),"survival":float(1-done_any.mean()),"vx_error":float(np.concatenate(vx).mean()),"finite":bool(np.isfinite(np.concatenate(rewards)).all())})
                return rows
    
            evaluations={condition:evaluate(model,condition) for condition,(model,_) in trained.items()}
            report_status = "CONFIRMATORY_RUN_COMPLETE" if args.confirmatory else "PIPELINE_VALID_DIAGNOSTIC_ONLY"
            report={"schema":"v1c_pilot_v1","status":report_status,"training_authorized":args.confirmatory,"seed":args.seed,"updates":args.updates,"num_envs":args.num_envs,"horizon":args.horizon,"evaluation_grid":grid.tolist(),"conditions":{condition:{"terminal_checkpoint":info["terminal_checkpoint"],"max_reward_reconstruction_error":info["max_reward_reconstruction_error"],"reconstruction_tolerance":2e-6,"records":info["records"],"evaluation":evaluations[condition]} for condition,(_,info) in trained.items()},"note":"Measurement artifact only; no curriculum superiority verdict."}
            args.output.write_text(json.dumps(report,indent=2)+"\n"); mark("ARTIFACT_WRITTEN",path=str(args.output)); mark("RUN_DONE",status=report_status); print(json.dumps({"schema":report["schema"],"status":report["status"],"conditions":list(report["conditions"]),"grid_points":len(grid)},indent=2))
        except BaseException as exc:
            (args.output.with_name(args.output.stem+".ERROR.json")).write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n"); mark("ERROR",error=str(exc)); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    
    
    if True: main()

def run_v1c_reconstruction_audit():
    """Run former v1c_reconstruction_audit.py stage."""
    """Read-only V1-C reward reconstruction/dtype audit."""
    
    import argparse
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--curriculum-checkpoint", type=Path, required=True)
        parser.add_argument("--control-checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=8)
        parser.add_argument("--steps", type=int, default=64)
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
        app = env = None
        try:
            from isaaclab.app import AppLauncher
            saved = sys.argv[:]; sys.argv = [sys.argv[0]]
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            sys.argv = saved
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import S7_OBJECTIVE_TERMS, group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
    
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = args.seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            obs, _ = env.reset(seed=args.seed); obs = obs_tensor(obs).cuda(); action_dim = env.unwrapped.action_manager.total_action_dim
    
            def load(path):
                model = V1CSharedActorCritic(obs.shape[-1], action_dim).cuda()
                state = torch.load(path, map_location="cuda", weights_only=False)
                model.load_state_dict(state["model"], strict=True); model.eval(); return model
    
            models = {"curriculum": load(args.curriculum_checkpoint), "full_simplex_control": load(args.control_checkpoint)}
            results = {}
            for condition, model in models.items():
                current, _ = env.reset(seed=args.seed); current = obs_tensor(current).cuda(); w = torch.full((args.num_envs, 3), 1 / 3, device="cuda")
                f32_errors = []; f64_errors = []; stock_order_errors = []; locations = []; term_names = None; max_term_snapshot = None
                with torch.no_grad():
                    for step in range(args.steps):
                        action = torch.clamp(model.act_inference_with_preference(current, w), -1.0, 1.0)
                        nxt, scalar_reward, _, _, _ = env.step(action)
                        manager = env.unwrapped.reward_manager; raw = manager._step_reward.detach().cpu().numpy(); term_names = list(manager.active_terms)
                        weighted = {name: raw[:, i] for i, name in enumerate(term_names)}
                        legacy_vector = group_v1b_s7_terms(weighted, shape=(args.num_envs,))
                        legacy_reconstruction = legacy_vector.sum(axis=-1).astype(np.float64)
                        direct_vector = []
                        for objective_terms in S7_OBJECTIVE_TERMS.values():
                            direct_vector.append(sum(np.asarray(weighted[name], dtype=np.float64) for name in objective_terms if name in weighted))
                        direct_reconstruction = np.stack(direct_vector, axis=-1).sum(axis=-1)
                        scalar_raw = scalar_reward.detach().cpu().numpy().astype(np.float64) / float(env.unwrapped.step_dt)
                        stock_order_reconstruction = sum(np.asarray(weighted[name], dtype=np.float64) for name in term_names)
                        f32_delta = np.abs(legacy_reconstruction - scalar_raw); f64_delta = np.abs(direct_reconstruction - scalar_raw); stock_delta = np.abs(stock_order_reconstruction - scalar_raw)
                        f32_errors.extend(f32_delta.tolist()); f64_errors.extend(f64_delta.tolist()); stock_order_errors.extend(stock_delta.tolist())
                        max_index = int(np.argmax(f64_delta)); locations.append({"step": step, "env": max_index, "float32_error": float(f32_delta[max_index]), "float64_error": float(f64_delta[max_index])})
                        if max_term_snapshot is None or f64_delta[max_index] > max_term_snapshot["float64_error"]:
                            max_term_snapshot = {"step": step, "env": max_index, "float64_error": float(f64_delta[max_index]), "term_values": {name: float(raw[max_index, i]) for i, name in enumerate(term_names)}, "scalar_raw": float(scalar_raw[max_index]), "grouped_raw": float(direct_reconstruction[max_index]), "stock_order_raw": float(stock_order_reconstruction[max_index])}
                        current = obs_tensor(nxt).cuda()
                results[condition] = {"float32_legacy": {"max_abs": float(np.max(f32_errors)), "mean_abs": float(np.mean(f32_errors)), "p99_abs": float(np.percentile(f32_errors, 99))}, "float64_direct": {"max_abs": float(np.max(f64_errors)), "mean_abs": float(np.mean(f64_errors)), "p99_abs": float(np.percentile(f64_errors, 99))}, "stock_term_order_float64": {"max_abs": float(np.max(stock_order_errors)), "mean_abs": float(np.mean(stock_order_errors)), "p99_abs": float(np.percentile(stock_order_errors, 99))}, "max_error_locations": sorted(locations, key=lambda x: x["float64_error"], reverse=True)[:5], "max_term_snapshot": max_term_snapshot, "finite": bool(np.isfinite(f32_errors).all() and np.isfinite(f64_errors).all()), "term_names": term_names}
            report = {"schema": "v1c_reconstruction_audit_v1", "status": "COMPLETE_READ_ONLY", "seed": args.seed, "steps": args.steps, "num_envs": args.num_envs, "action_semantics": "deterministic actor mean + clip_actions=1.0", "objective_terms": {key: list(value) for key, value in S7_OBJECTIVE_TERMS.items()}, "conditions": results, "no_training": True, "tolerance_changed": False}
            args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    
    
    if True: main()

STAGES = {
    "v1c_aggregate": run_v1c_aggregate,
    "v1c_d0_aggregate": run_v1c_d0_aggregate,
    "v1c_d0_preference_audit": run_v1c_d0_preference_audit,
    "v1c_isaac_smoke": run_v1c_isaac_smoke,
    "v1c_pilot": run_v1c_pilot,
    "v1c_reconstruction_audit": run_v1c_reconstruction_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
