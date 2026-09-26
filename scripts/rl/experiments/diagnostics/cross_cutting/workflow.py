"""Consolidated experiment stages. Use the first CLI argument to select a former script stage."""
from __future__ import annotations
import argparse

def run_context_incremental_retention_audit():
    """Run former context_incremental_retention_audit.py stage."""
    """Do context features predict gate retention beyond the mean margins alone?
    
    Read-only. Fits leave-one-seed-out logistic regressions on the two mean
    margins, then on those plus seven context features, and asks whether the
    larger model earns its keep on every criterion at once. A nonparametric check
    inside each margin stratum backs the parametric one up, so a gain cannot come
    from the fit alone.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import REPO, RUNS, OfflineAudit
    
    SRC = "semantic_gate_factorization_audit-2026-09-24/semantic_gate_factorization_report.json"
    CONTRACT = "docs/contracts/diagnostics/context-incremental-retention-contract.md"
    SEEDS = (980001, 981001, 982001)
    AXES = ("T", "A", "O", "S")
    BASE_FEATS = ["obj_mean_margin", "phys_mean_margin"]
    CTX_FEATS = ["obj_phase_pos_frac", "phys_phase_pos_frac",
                 "obj_worst_suite_phase_margin", "phys_worst_suite_phase_margin",
                 "obj_std_margin", "phys_std_margin", "obj_phys_sign_agreement_fraction"]
    SUITE_PHASE_CELLS = 12.0
    ROBUST_CTX = .75
    
    
    def safe_auc(y, p):
        y, p = np.asarray(y, int), np.asarray(p, float)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        wins = sum(1.0 if p[i] > p[j] else 0.5 if p[i] == p[j] else 0.0 for i in pos for j in neg)
        return float(wins / (len(pos) * len(neg)))
    
    
    def safe_ap(y, p):
        y, p = np.asarray(y, int), np.asarray(p, float)
        if y.sum() == 0:
            return float("nan")
        ys = y[np.argsort(-p, kind="mergesort")]
        prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
        return float(np.sum(prec * ys) / y.sum())
    
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    
    
    def fit_logistic_l2(x, y, c=1.0, steps=4000, lr=0.05):
        x, y = np.asarray(x, float), np.asarray(y, float)
        w, b, lam = np.zeros(x.shape[1]), 0.0, 1.0 / c
        for _ in range(steps):
            err = sigmoid(x @ w + b) - y
            w -= lr * (x.T @ err / len(y) + lam * w / len(y))
            b -= lr * float(err.mean())
        return w, b
    
    
    def fit_predict(train, test, features):
        xtr = np.array([[x[f] for f in features] for x in train], float)
        xte = np.array([[x[f] for f in features] for x in test], float)
        ytr = np.array([x["retained"] for x in train], int)
        mu, sd = xtr.mean(0), xtr.std(0)
        sd[sd < 1e-12] = 1.0
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        if len(set(ytr)) < 2:
            return np.full(len(test), float(ytr.mean()))
        w, b = fit_logistic_l2(xtr, ytr, c=1.0)
        return sigmoid(xte @ w + b)
    
    
    def metrics(rows, p):
        y = np.array([r["retained"] for r in rows], int)
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return {"n": len(rows), "positive_retained": int(y.sum()), "forgotten": int((1 - y).sum()),
                "roc_auc": safe_auc(y, p), "average_precision": safe_ap(y, p),
                "brier": float(np.mean((p - y) ** 2)),
                "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    
    
    class ContextIncrementalRetentionAudit(OfflineAudit):
        """Incremental value of context features for predicting gate retention."""
    
        run = "context_incremental_retention_audit-2026-09-24"
        report = "context_incremental_retention_report.json"
        schema = "context_incremental_retention_audit_v1"
    
        def rows(self):
            states = json.loads((RUNS / SRC).read_text())["states"]
            out = []
            for s in SEEDS:
                arr = states[str(s)]
                for i in range(len(arr) - 1):
                    for axis in AXES:
                        src, dst = arr[i]["axes"][axis], arr[i + 1]["axes"][axis]
                        if not src["PASS"]:
                            continue    # only a state that passed can be forgotten
                        out.append({
                            "seed": s, "from": arr[i]["label"], "to": arr[i + 1]["label"],
                            "axis": axis, "retained": int(dst["PASS"]),
                            "obj_mean_margin": float(src["obj_mean_margin"]),
                            "phys_mean_margin": float(src["phys_mean_margin"]),
                            "obj_phase_pos_frac": 1.0 - float(src["obj_negative_suite_phase_cells"]) / SUITE_PHASE_CELLS,
                            "phys_phase_pos_frac": 1.0 - float(src["phys_negative_suite_phase_cells"]) / SUITE_PHASE_CELLS,
                            "obj_worst_suite_phase_margin": float(src["obj_worst_suite_phase_margin"]),
                            "phys_worst_suite_phase_margin": float(src["phys_worst_suite_phase_margin"]),
                            "obj_std_margin": float(src["obj_std_margin"]),
                            "phys_std_margin": float(src["phys_std_margin"]),
                            "obj_phys_sign_agreement_fraction": float(src["obj_phys_sign_agreement_fraction"])})
            return out
    
        def strata(self, rows):
            """Within each seed's upper/lower margin half, does context still separate?"""
            comp = []
            for s in SEEDS:
                rr = [r for r in rows if r["seed"] == s]
                vals = np.array([[r["obj_mean_margin"], r["phys_mean_margin"]] for r in rr], float)
                mu, sd = vals.mean(0), vals.std(0)
                sd[sd < 1e-12] = 1
                b = ((vals - mu) / sd).sum(1)
                med = float(np.median(b))
                for r, bv in zip(rr, b):
                    q = dict(r)
                    q["stratum"] = "upper" if bv >= med else "lower"
                    q["ctx_robust"] = bool(r["obj_phase_pos_frac"] >= ROBUST_CTX
                                           and r["phys_phase_pos_frac"] >= ROBUST_CTX)
                    comp.append(q)
            out = {}
            for st in ("lower", "upper"):
                ss = [x for x in comp if x["stratum"] == st]
                a = [x for x in ss if x["ctx_robust"]]
                b = [x for x in ss if not x["ctx_robust"]]
                ra = float(np.mean([x["retained"] for x in a])) if a else float("nan")
                rb = float(np.mean([x["retained"] for x in b])) if b else float("nan")
                out[st] = {"n": len(ss), "robust_n": len(a), "nonrobust_n": len(b),
                           "robust_retention": ra, "nonrobust_retention": rb,
                           "difference": ra - rb if np.isfinite(ra) and np.isfinite(rb) else float("nan")}
            return out
    
        def analyze(self):
            rows = self.rows()
            pred_base, pred_ctx, ordered, per_seed = [], [], [], {}
            for held in SEEDS:
                tr = [r for r in rows if r["seed"] != held]
                te = [r for r in rows if r["seed"] == held]
                pb = fit_predict(tr, te, BASE_FEATS)
                pc = fit_predict(tr, te, BASE_FEATS + CTX_FEATS)
                mb, mc = metrics(te, pb), metrics(te, pc)
                per_seed[str(held)] = {
                    "base": mb, "base_ctx": mc,
                    "auc_delta": (mc["roc_auc"] - mb["roc_auc"]
                                  if np.isfinite(mc["roc_auc"]) and np.isfinite(mb["roc_auc"])
                                  else float("nan")),
                    "brier_delta": mc["brier"] - mb["brier"]}
                ordered.extend(te)
                pred_base.extend(pb.tolist())
                pred_ctx.extend(pc.tolist())
            pooled_base, pooled_ctx = metrics(ordered, pred_base), metrics(ordered, pred_ctx)
            strata = self.strata(rows)
    
            n = len(rows)
            forget = sum(1 - r["retained"] for r in rows)
            under = n < 20 or forget < 8
            seed_nonworse = True
            for v in per_seed.values():
                if np.isfinite(v["auc_delta"]):
                    seed_nonworse &= v["auc_delta"] >= -1e-12
                else:
                    seed_nonworse &= v["brier_delta"] <= 1e-12
            diffs = [strata[x]["difference"] for x in ("lower", "upper")]
            nonparam = (all(np.isfinite(x) for x in diffs)
                        and max(diffs) >= .15 and min(diffs) >= -1e-12)
            criteria = {
                "sample_not_underpowered": not under,
                "auc_gain_ge_0p10": pooled_ctx["roc_auc"] >= pooled_base["roc_auc"] + .10,
                "ap_gain_ge_0p10": pooled_ctx["average_precision"] >= pooled_base["average_precision"] + .10,
                "brier_improvement_ge_0p02": pooled_ctx["brier"] <= pooled_base["brier"] - .02,
                "logloss_improves": pooled_ctx["log_loss"] < pooled_base["log_loss"],
                "nonworse_all_3_seeds": bool(seed_nonworse),
                "nonparametric_context_gain": bool(nonparam)}
            if under:
                status = "UNDERPOWERED / INCONCLUSIVE"
            elif all(criteria.values()):
                status = "INCREMENTAL CONTEXT VALUE SUPPORTED"
            else:
                any_gain = any([criteria["auc_gain_ge_0p10"], criteria["ap_gain_ge_0p10"],
                                criteria["brier_improvement_ge_0p02"],
                                criteria["nonparametric_context_gain"]])
                status = "INCONCLUSIVE" if any_gain else "NO INCREMENTAL VALUE"
            return {"schema": self.schema, "status": status, "read_only": True,
                    "pass_origin_n": n, "forgetting_n": forget, "retained_n": n - forget,
                    "pooled": {"base": pooled_base, "base_ctx": pooled_ctx,
                               "auc_gain": pooled_ctx["roc_auc"] - pooled_base["roc_auc"],
                               "ap_gain": pooled_ctx["average_precision"] - pooled_base["average_precision"],
                               "brier_improvement": pooled_base["brier"] - pooled_ctx["brier"],
                               "logloss_improvement": pooled_base["log_loss"] - pooled_ctx["log_loss"]},
                    "per_seed": per_seed, "nonparametric_strata": strata, "criteria": criteria,
                    "decision": {"context_intervention_design_authorized":
                                 status == "INCREMENTAL CONTEXT VALUE SUPPORTED"},
                    "rows": rows}
    
        def execute(self):
            rep = super().execute()
            self.write({"status": "FROZEN_BY_HASH", "decision": rep["status"],
                        "report_sha256": self.sha(self.out / self.report),
                        "script_sha256": self.sha(Path(__file__).resolve()),
                        "contract_sha256": self.sha(REPO / CONTRACT),
                        "source_sha256": self.sha(RUNS / SRC)},
                       "PROVENANCE_MANIFEST.json")
            return rep
    
        def summarize(self, rep):
            print(json.dumps({k: rep[k] for k in
                              ["status", "pass_origin_n", "forgetting_n", "retained_n", "pooled",
                               "per_seed", "nonparametric_strata", "criteria"]}, indent=2))
    
    
    if True:
        ContextIncrementalRetentionAudit.main()

def run_d1_same_state_probe():
    """Run former d1_same_state_probe.py stage."""
    """Read-only same-state action-basin probe for D1 diagnostics."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    import argparse, json, os
    from pathlib import Path
    import numpy as np
    import torch
    
    from rl.experiments.common.utilities.final_locomotion_eval import _select, _nominalize_cfg, _set_initial_state, _sha256
    from talon_rl.config import ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg
    from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
    
    def main():
        p=argparse.ArgumentParser(); p.add_argument("checkpoint",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--seed",type=int,default=1001); p.add_argument("--steps",type=int,default=30); a=p.parse_args()
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app
        import gymnasium as gym; import isaaclab; import talon_rl.tasks.locomotion.a1_env
        import yaml
        stored=yaml.safe_load((a.checkpoint.resolve().parents[1]/"config.yaml").read_text())
        obs=ObservationSpaceCfg(**_select(ObservationSpaceCfg,stored["obs"])); act=ActionSpaceCfg(**_select(ActionSpaceCfg,stored["action"]))
        rew=RewardVectorCfg(**_select(RewardVectorCfg,stored["reward"])); pref=PreferenceCfg(**_select(PreferenceCfg,stored["preference"]))
        stack=ObservationStackCfg(**_select(ObservationStackCfg,stored["stack"])); mc=MOPPOConfig(**_select(MOPPOConfig,stored["moppo"]))
        cfg=__import__('talon_rl.tasks.locomotion.a1_env.a1_env_cfg',fromlist=['IsaacLabTalonEnvCfg']).IsaacLabTalonEnvCfg(); cfg.scene.num_envs=int(os.environ.get('D1_LANES','8')); cfg.seed=a.seed; cfg.sim.dt=.01; cfg.decimation=1; cfg.sim.render_interval=1; _nominalize_cfg(cfg)
        env=gym.make("Isaac-Talon-A1-v0",cfg=cfg,render_mode=None).unwrapped
        tr=MOPPOTrainer(env,obs,rew,pref,moppo_cfg=mc,stack_cfg=stack,extrinsics_cfg=ExtrinsicsCfg(),seed=a.seed); tr.load(str(a.checkpoint)); tr.model.eval()
        rng=np.random.default_rng(a.seed); env.reset(); _set_initial_state(env,rng); env.v_command_buf[:]=0
        tr.push_obs(env._transition(env.observation_manager.compute())["obs"],update_normalizer=False)
        mean=tr.act_inference().copy(); actor_obs=torch.from_numpy(tr._actor_obs()).to(tr.device); torch.manual_seed(int(os.environ.get('D1_SAMPLE_SEED','7001')))
        with torch.no_grad(): sample=tr.model.act(actor_obs)[0].detach().cpu().numpy()
        mean_norm=np.linalg.norm(mean,axis=1,keepdims=True).clip(min=1e-6)
        sample_norm=np.linalg.norm(sample,axis=1,keepdims=True).clip(min=1e-6)
        direction_only=sample/sample_norm*mean_norm
        variants={"mean":mean,"radial_0.70":mean*.70,"radial_0.85":mean*.85,"sample_same_state":sample,"direction_only_sample_norm":direction_only}
        out=[]
        for name,first in variants.items():
            env.reset(); _set_initial_state(env,np.random.default_rng(a.seed)); env.v_command_buf[:]=0
            tr.push_obs(env._transition(env.observation_manager.compute())["obs"],update_normalizer=False)
            alive=np.ones(env.num_envs,bool); max_tilt=np.zeros(env.num_envs); min_h=np.full(env.num_envs,np.inf); contact=np.zeros(env.num_envs); fail=np.full(env.num_envs,a.steps)
            for k in range(a.steps):
                action=first if k==0 else tr.act_inference(); trans,done=env.step(action); frame=trans
                q=frame["root_quat_w"]; w,x,y,z=[q[:,i] for i in range(4)]; roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)); pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1));
                max_tilt=np.maximum(max_tilt,np.maximum(np.abs(roll),np.abs(pitch))); min_h=np.minimum(min_h,frame["root_pos_w"][:,2]); contact+=frame.get("undesired_contact_count",np.zeros(env.num_envs));
                newly=alive & done; fail[newly]=k+1; alive &= ~done; tr.push_obs(trans["obs"],done_mask=done,update_normalizer=False)
            rec={"variant":name,"survivors":int(alive.sum()),"first_fail_min":int(fail.min()),"first_fail_mean":float(fail.mean()),"mean_action_norm":float(np.linalg.norm(first,axis=1).mean()),"mean_saturation_frac":float((np.abs(first)>=.95*3).mean()),"max_tilt_rad":float(max_tilt.max()),"min_height_m":float(min_h.min()),"undesired_contact_mean":float(contact.mean()/a.steps)}
            delta=first-mean; dot=np.sum(first*mean,axis=1); cos=dot/(np.linalg.norm(first,axis=1)*np.linalg.norm(mean,axis=1)+1e-8)
            rec["per_lane"]=[{"lane":int(i),"first_fail_step":int(fail[i]),"survivor":bool(alive[i]),"delta_a":delta[i].tolist(),"abs_delta_a":np.abs(delta[i]).tolist(),"cosine_to_mean":float(cos[i]),"norm_ratio":float(np.linalg.norm(first[i])/(np.linalg.norm(mean[i])+1e-8)),"saturation_margin":float(3.0-np.max(np.abs(first[i])))} for i in range(env.num_envs)]
            out.append(rec)
        result={"protocol":"D1.1 same-state action-basin probe","checkpoint":str(a.checkpoint),"checkpoint_sha256":_sha256(a.checkpoint),"eval_seed":a.seed,"num_envs":env.num_envs,"steps":a.steps,"variants":out,"read_only":True}
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n"); print(a.output)
        env.close(); app.close()
    if True: main()

def run_final_eval_manifest():
    """Run former final_eval_manifest.py stage."""
    """Build the immutable-input manifest for Final Locomotion Evaluation v1."""
    
    
    import argparse
    import hashlib
    import json
    import platform
    import subprocess
    from pathlib import Path
    
    import torch
    import yaml
    
    
    ROOT = Path(__file__).resolve().parents[4]
    PROTOCOL = ROOT / "docs/protocols/evaluation/final-locomotion-evaluation-protocol.md"
    RUNS = [ROOT / f"runs/phase1_hipact_dt01_seed{i}_2026-09-20" for i in range(3)]
    
    
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    
    
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--smoke-json", type=Path)
        args = parser.parse_args()
    
        candidates = []
        for seed, run in enumerate(RUNS):
            checkpoint = run / "checkpoints/checkpoint.pt"
            config_path = run / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            rollout_steps = int(config["moppo"]["num_steps"])
            env_steps = int(state["t"])
            if env_steps % rollout_steps:
                raise SystemExit(f"{checkpoint}: t={env_steps} is not divisible by num_steps={rollout_steps}")
            candidates.append(
                {
                    "training_seed": seed,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256(checkpoint),
                    "config": str(config_path.relative_to(ROOT)),
                    "config_sha256": sha256(config_path),
                    "checkpoint_env_steps": env_steps,
                    "rollout_steps_per_update": rollout_steps,
                    "training_updates": env_steps // rollout_steps,
                    "encoder_present": state.get("encoder") is not None,
                    "contrib_clip_percentile": state.get("contrib_clip_percentile"),
                    "objective_names": config["reward"]["term_names"],
                    "objective_dim": len(config["reward"]["term_names"]),
                    "action_dim": int(config["action"]["dim"]),
                    "effective_config": config,
                }
            )
    
        dirty_diff = git("diff", "--binary", "HEAD")
        untracked = [
            name for name in git("ls-files", "--others", "--exclude-standard").splitlines()
            if not name.startswith("artifacts/")
        ]
        untracked_hashes = {
            name: sha256(ROOT / name) for name in untracked if (ROOT / name).is_file()
        }
        smoke = json.loads(args.smoke_json.read_text()) if args.smoke_json else None
        manifest = {
            "schema_version": 1,
            "criteria_label": "project-defined milestone criteria",
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": sha256(PROTOCOL),
            "git_commit": git("rev-parse", "HEAD"),
            "git_status_porcelain": git("status", "--porcelain=v1").splitlines(),
            "git_tracked_diff_sha256": hashlib.sha256(dirty_diff.encode()).hexdigest(),
            "untracked_files": untracked,
            "untracked_file_sha256": untracked_hashes,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "candidates": candidates,
            "matrix": {
                "commands_vx_mps": [-0.25, 0.0, 0.25, 0.5, 0.75],
                "preferences": {
                    "uniform": [0.25, 0.25, 0.25, 0.25],
                    "progress_heavy": [0.55, 0.15, 0.15, 0.15],
                    "efficiency_heavy": [0.15, 0.55, 0.10, 0.20],
                    "impact_heavy": [0.15, 0.10, 0.55, 0.20],
                    "balance_heavy": [0.15, 0.15, 0.10, 0.60],
                },
                "evaluation_reset_seeds": [1001, 1002, 1003],
                "lanes_per_cell": 64,
                "settling_seconds": 2.0,
                "command_seconds": 20.0,
                "policy_step_seconds": 0.01,
                "expected_episode_count": 14_400,
            },
            "execution": {
                "simulator": "Isaac Sim 5.1" if smoke else None,
                "isaaclab": smoke.get("isaaclab_version") if smoke else None,
                "gpu": "NVIDIA GeForce RTX 3070 Ti / driver 580.178.04" if smoke else None,
                "effective_limits": {
                    "action_abs": smoke.get("action_clip"),
                    "torque_nm": smoke.get("torque_limit_nm"),
                    "body_mass_kg": [smoke.get("body_mass_kg_min"), smoke.get("body_mass_kg_max")],
                } if smoke else None,
                "termination_definitions": ["time_out", "obstacle_reached (disabled)", "base_contact"],
                "instrumentation_smoke_check": smoke,
                "status": "PREFLIGHT_PASSED" if smoke else "INCOMPLETE_UNTIL_RUNNER_PREFLIGHT",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(args.output)
    
    
    if True:
        main()

def run_robust_collapse_mechanism_audit():
    """Run former robust_collapse_mechanism_audit.py stage."""
    from pathlib import Path
    import json, hashlib
    import numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    CORP=ROOT/'runs/semantic_collapse_corpus-2026-09-24/semantic_collapse_corpus.json'
    HOLD=ROOT/'runs/relational_persistence_heldout-2026-09-24/heldout_report.json'
    FAC=ROOT/'runs/semantic_gate_factorization_audit-2026-09-24/semantic_gate_factorization_report.json'
    ORD=ROOT/'runs/preference_behavior_ordering_audit-2026-09-24/preference_behavior_ordering_report.json'
    ROB=ROOT/'runs/semantic_gate_robustness_audit-2026-09-24/semantic_gate_robustness_report.json'
    CACHE=ROOT/'runs/preference_behavior_ordering_audit-2026-09-24/raw_cache'
    OUT=ROOT/'runs/robust_collapse_mechanism_audit-2026-09-24'
    AXES=('T','A','O','S'); TOL=1e-10
    CONT_DESC=[
     'J_heavy','obj_adv_mean','phys_adv_mean','obj_adv_late','phys_adv_late','obj_adv_late_minus_early','phys_adv_late_minus_early',
     'obj_adv_slope','phys_adv_slope','obj_adv_final16','phys_adv_final16','obj_adv_worst16','phys_adv_worst16','obj_adv_positive_fraction','phys_adv_positive_fraction',
     'joint_primary_order_acc','joint_reset_consistent_pair_fraction']
    CTX_FLAGS=['worst_suite_collapse','phase_specific_collapse','objective_physical_disagreement_growth','heterogeneity_growth']
    QUARTERS=[(0,16),(16,32),(32,48),(48,64)]
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def key(seed,fr,to,axis): return (int(seed),fr,to,axis)
    def gate_margin(vals): return float(np.sort(np.asarray(vals,float))[1])
    
    def main():
        corp=json.load(open(CORP)); hold=json.load(open(HOLD)); fac=json.load(open(FAC)); ordrep=json.load(open(ORD)); rob=json.load(open(ROB))
        robust_keys={key(e['seed'],e['from'],e['to'],e['axis']) for e in corp['robust_semantic_collapse']}
        hold_s={key(s['seed'],s['from'],s['to'],s['axis']):s for s in hold['samples']}
        ord_s={key(s['seed'],s['from'],s['to'],s['axis']):s for s in ordrep['samples']}
        fac_s={key(s['seed'],s['from'],s['to'],s['axis']):s for s in fac['all_transitions']}
    
        # state lookup and G_sem for all states from factorization + frozen scales
        scales=rob['scale']; stateG={}
        for sd,arr in fac['states'].items():
            for st in arr:
                for a in AXES:
                    q=st['axes'][a]
                    zo=[float(x['obj_margin'])/scales[a]['obj'] for x in q['suite_factors']]
                    zp=[float(x['phys_margin'])/scales[a]['phys'] for x in q['suite_factors']]
                    stateG[(int(sd),st['label'],a)]=min(gate_margin(zo),gate_margin(zp))
    
        # ordering checkpoint path lookup
        path_lookup={}
        for sd,arr in ordrep['states'].items():
            for st in arr: path_lookup[(int(sd),st['label'])]=st['checkpoint']
        cache_by_path={}
        for p in CACHE.glob('*.npz'):
            z=np.load(p,allow_pickle=False); cache_by_path[str(z['checkpoint_path'])]=p
    
        # frozen continuous descriptors merged from heldout and ordering
        rows=[]
        for k in robust_keys:
            hs=hold_s[k]; os=ord_s[k]; fs=fac_s[k]
            ch=hs['changes']
            vals={
                'J_heavy':ch['J_heavy'],'obj_adv_mean':ch['obj_adv_mean'],'phys_adv_mean':ch['phys_adv_mean'],
                'obj_adv_late':ch['obj_adv_late'],'phys_adv_late':ch['phys_adv_late'],
                'obj_adv_late_minus_early':ch['obj_adv_late_minus_early'],'phys_adv_late_minus_early':ch['phys_adv_late_minus_early'],
                'obj_adv_slope':ch['obj_adv_slope'],'phys_adv_slope':ch['phys_adv_slope'],
                'obj_adv_final16':ch['obj_adv_final16'],'phys_adv_final16':ch['phys_adv_final16'],
                'obj_adv_worst16':ch['obj_adv_worst16'],'phys_adv_worst16':ch['phys_adv_worst16'],
                'obj_adv_positive_fraction':ch['obj_adv_positive_fraction'],'phys_adv_positive_fraction':ch['phys_adv_positive_fraction'],
                'joint_primary_order_acc':os['delta']['joint_primary_order_acc'],
                'joint_reset_consistent_pair_fraction':os['delta']['joint_reset_consistent_pair_fraction']}
            row={'seed':k[0],'from':k[1],'to':k[2],'axis':k[3],'source_G':stateG[(k[0],k[1],k[3])],'target_G':stateG[(k[0],k[2],k[3])],
                 'descriptors':vals,'flags':{f:bool(fs['flags'][f]) for f in CTX_FLAGS},'target_limiting_channel':next(e['target_limiting_channel'] for e in corp['robust_semantic_collapse'] if key(e['seed'],e['from'],e['to'],e['axis'])==k)}
            rows.append(row)
        rows=sorted(rows,key=lambda x:(x['seed'],x['from'],x['axis']))
    
        descriptor_summary={}
        for d in CONT_DESC:
            vv=[r['descriptors'][d] for r in rows]
            descriptor_summary[d]={'deterioration_n':int(sum(v < -TOL for v in vv)),'deterioration_fraction':float(np.mean(np.asarray(vv)<-TOL)),'median_delta':float(np.median(vv))}
        for f in CTX_FLAGS:
            vv=[r['flags'][f] for r in rows]
            descriptor_summary[f]={'deterioration_n':int(sum(vv)),'deterioration_fraction':float(np.mean(vv)),'median_delta':None}
    
        # control pool PASS->PASS and descriptor merged
        controls=[]
        for k,hs in hold_s.items():
            if not (hs['pass_from'] and hs['pass_to']): continue
            os=ord_s[k];fs=fac_s[k];ch=hs['changes']
            vals={'J_heavy':ch['J_heavy'],'obj_adv_mean':ch['obj_adv_mean'],'phys_adv_mean':ch['phys_adv_mean'],'obj_adv_late':ch['obj_adv_late'],'phys_adv_late':ch['phys_adv_late'],
                  'obj_adv_late_minus_early':ch['obj_adv_late_minus_early'],'phys_adv_late_minus_early':ch['phys_adv_late_minus_early'],'obj_adv_slope':ch['obj_adv_slope'],'phys_adv_slope':ch['phys_adv_slope'],
                  'obj_adv_final16':ch['obj_adv_final16'],'phys_adv_final16':ch['phys_adv_final16'],'obj_adv_worst16':ch['obj_adv_worst16'],'phys_adv_worst16':ch['phys_adv_worst16'],
                  'obj_adv_positive_fraction':ch['obj_adv_positive_fraction'],'phys_adv_positive_fraction':ch['phys_adv_positive_fraction'],
                  'joint_primary_order_acc':os['delta']['joint_primary_order_acc'],'joint_reset_consistent_pair_fraction':os['delta']['joint_reset_consistent_pair_fraction']}
            controls.append({'seed':k[0],'from':k[1],'to':k[2],'axis':k[3],'source_G':stateG[(k[0],k[1],k[3])],'descriptors':vals,'flags':{f:bool(fs['flags'][f]) for f in CTX_FLAGS}})
    
        # deterministic matching: same-axis nearest sourceG first; no reuse until exhausted
        unused=set(range(len(controls))); matches=[]
        for r in rows:
            same=[i for i in unused if controls[i]['axis']==r['axis']]
            cand=same if same else list(unused)
            reused=False
            if not cand:
                same=[i for i,c in enumerate(controls) if c['axis']==r['axis']]
                cand=same if same else list(range(len(controls))); reused=True
            i=min(cand,key=lambda i:(abs(controls[i]['source_G']-r['source_G']),controls[i]['seed'],controls[i]['from'],controls[i]['axis']))
            if i in unused:unused.remove(i)
            c=controls[i]
            matches.append({'robust':{k:r[k] for k in ['seed','from','to','axis','source_G']},'control':{k:c[k] for k in ['seed','from','to','axis','source_G']},'source_G_abs_diff':abs(c['source_G']-r['source_G']),'reused':reused,'control_index':i})
    
        matched=[controls[m['control_index']] for m in matches]
        matched_summary={}; specific=[]
        for d in CONT_DESC:
            rv=[r['descriptors'][d] for r in rows];cv=[c['descriptors'][d] for c in matched]
            rf=float(np.mean(np.asarray(rv)<-TOL));cf=float(np.mean(np.asarray(cv)<-TOL));diff=rf-cf
            matched_summary[d]={'robust_deterioration_fraction':rf,'matched_retained_deterioration_fraction':cf,'difference':diff,'robust_median_delta':float(np.median(rv)),'retained_median_delta':float(np.median(cv))}
            if rf>=.75 and cf<=.50 and diff>=.25:specific.append(d)
        for f in CTX_FLAGS:
            rv=[r['flags'][f] for r in rows];cv=[c['flags'][f] for c in matched];rf=float(np.mean(rv));cf=float(np.mean(cv));diff=rf-cf
            matched_summary[f]={'robust_deterioration_fraction':rf,'matched_retained_deterioration_fraction':cf,'difference':diff,'robust_median_delta':None,'retained_median_delta':None}
            if rf>=.75 and cf<=.50 and diff>=.25:specific.append(f)
    
        # quarter trajectories from q=.70 heavy and .25 center; axis index = signal index
        quarter_events=[]
        for r in rows:
            ai=AXES.index(r['axis']);ps=cache_by_path[path_lookup[(r['seed'],r['from'])]];pt=cache_by_path[path_lookup[(r['seed'],r['to'])]]
            zs=np.load(ps,allow_pickle=False);zt=np.load(pt,allow_pickle=False)
            # [axis_eval,suite,q,steps,signal]
            src_o=zs['obj'][ai,:,4,:,ai]-zs['obj'][ai,:,1,:,ai]
            tgt_o=zt['obj'][ai,:,4,:,ai]-zt['obj'][ai,:,1,:,ai]
            src_p=zs['phys'][ai,:,1,:,ai]-zs['phys'][ai,:,4,:,ai]
            tgt_p=zt['phys'][ai,:,1,:,ai]-zt['phys'][ai,:,4,:,ai]
            od=[];pd=[]
            for lo,hi in QUARTERS:
                od.append(float(np.mean(tgt_o[:,lo:hi])-np.mean(src_o[:,lo:hi])))
                pd.append(float(np.mean(tgt_p[:,lo:hi])-np.mean(src_p[:,lo:hi])))
            eo=next((i+1 for i,v in enumerate(od) if v<-TOL),None);ep=next((i+1 for i,v in enumerate(pd) if v<-TOL),None)
            if eo is None and ep is None:lead='none'
            elif eo is None:lead='physical-only'
            elif ep is None:lead='objective-only'
            elif eo<ep:lead='objective-first'
            elif ep<eo:lead='physical-first'
            else:lead='simultaneous'
            quarter_events.append({'seed':r['seed'],'from':r['from'],'to':r['to'],'axis':r['axis'],'objective_delta_by_quarter':od,'physical_delta_by_quarter':pd,'objective_earliest_deterioration_quarter':eo,'physical_earliest_deterioration_quarter':ep,'temporal_lead':lead,'target_limiting_channel':r['target_limiting_channel']})
        qsum={}
        for ch in ('objective','physical'):
            arr=np.array([e[f'{ch}_delta_by_quarter'] for e in quarter_events])
            qsum[ch]={'deterioration_count_by_quarter':[int(np.sum(arr[:,i]<-TOL)) for i in range(4)],'median_delta_by_quarter':[float(np.median(arr[:,i])) for i in range(4)],'q1_deterioration_fraction':float(np.mean(arr[:,0]<-TOL)),'only_after_q1_fraction':float(np.mean((arr[:,0]>=-TOL)&np.any(arr[:,1:]<-TOL,axis=1)))}
        from collections import Counter
        qsum['earliest_objective']=dict(Counter(str(e['objective_earliest_deterioration_quarter']) for e in quarter_events))
        qsum['earliest_physical']=dict(Counter(str(e['physical_earliest_deterioration_quarter']) for e in quarter_events))
        qsum['temporal_lead']=dict(Counter(e['temporal_lead'] for e in quarter_events))
    
        # split target limiting channel
        split={}
        for lim in ('objective','physical'):
            rr=[r for r in rows if r['target_limiting_channel']==lim]
            qq=[e for e in quarter_events if e['target_limiting_channel']==lim]
            split[lim]={'n':len(rr),'descriptor_deterioration':{d:float(np.mean([x['descriptors'][d]<-TOL for x in rr])) if rr else None for d in CONT_DESC},'context_flags':{f:float(np.mean([x['flags'][f] for x in rr])) if rr else None for f in CTX_FLAGS},'temporal_lead':dict(Counter(e['temporal_lead'] for e in qq))}
    
        common=[]
        for d,s in matched_summary.items():
            if s['robust_deterioration_fraction']>=7/8 and s['matched_retained_deterioration_fraction']<7/8 and s['difference']>0: common.append(d)
        status='COMMON ROBUST-COLLAPSE PRECURSOR' if common else ('MULTIPLE FAILURE MODES' if split['objective']['n'] and split['physical']['n'] else 'NO COMMON PRECURSOR')
        report={'schema':'robust_collapse_mechanism_audit_v1','status':status,'read_only':True,'robust_n':len(rows),'retained_control_pool_n':len(controls),'descriptor_summary':descriptor_summary,'matches':matches,'matched_control_summary':matched_summary,'specific_collapse_candidates_descriptive':specific,'quarter_events':quarter_events,'quarter_summary':qsum,'limiting_channel_split':split,'common_precursor_descriptors':common,'decision':{'new_training_method_authorized':False}}
        OUT.mkdir(parents=True,exist_ok=True);out=OUT/'robust_collapse_mechanism_report.json';out.write_text(json.dumps(report,indent=2)+'\n')
        (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':status,'report_sha256':sha(out),'contract_sha256':sha(ROOT/'docs/contracts/diagnostics/robust-collapse-mechanism-audit-contract.md'),'sources':{str(p.relative_to(ROOT)):sha(p) for p in [CORP,HOLD,FAC,ORD,ROB]}},indent=2)+'\n')
        print(json.dumps({'status':status,'robust_n':len(rows),'retained_control_pool_n':len(controls),'descriptor_summary':descriptor_summary,'specific_candidates':specific,'matches':matches,'quarter_summary':qsum,'limiting_channel_split':split,'common_precursor_descriptors':common},indent=2))
    if True: main()

def run_state_manifold_localization_audit():
    """Run former state_manifold_localization_audit.py stage."""
    from pathlib import Path
    import json,hashlib,sys,math
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    RUN=ROOT/'runs/state_manifold_localization_audit-2026-09-24';RUN.mkdir(parents=True,exist_ok=True)
    V18=ROOT/'runs/prospective_slope_validation-2026-09-24/prospective_slope_validation_report.json'
    SC=ROOT/'runs/update_visitation_interaction_audit-2026-09-24/state_cache'
    ORDER=('T','A','O','S');EPS=1e-12;K=16;SEED=260924;NINIT=20;MINST=16
    HEAVY={'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32)}
    CENTER=np.array([.25,.25,.25,.25],np.float32)
    PRIMARY=('R_change','R_rotation','J_change_rel','J_rotation')
    SECONDARY=('A_center','A_heavy')
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def mh(path):
     d=torch.load(path,map_location='cpu',weights_only=False)['model'];h=hashlib.sha256()
     for k in sorted(d):h.update(k.encode());h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
     return h.hexdigest()
    def auc(y,s):
     y=np.asarray(y,int);s=np.asarray(s,float);P=np.where(y==1)[0];N=np.where(y==0)[0]
     if len(P)==0 or len(N)==0:return float('nan')
     z=0.0
     for i in P:
      for j in N:z+=1 if s[i]>s[j] else .5 if s[i]==s[j] else 0
     return float(z/(len(P)*len(N)))
    def safe_cos_rows(a,b):
     a=np.asarray(a,float);b=np.asarray(b,float);num=np.sum(a*b,axis=tuple(range(1,a.ndim)));den=np.sqrt(np.sum(a*a,axis=tuple(range(1,a.ndim)))*np.sum(b*b,axis=tuple(range(1,b.ndim))))
     return num/(den+EPS)
    def pca_fit_transform(X):
     mu=X.mean(0);sd=X.std(0);sd=np.where(sd<1e-8,1.0,sd);Z=(X-mu)/sd
     cov=(Z.T@Z)/(len(Z)-1);eigval,eigvec=np.linalg.eigh(cov);idx=np.argsort(eigval)[::-1];eigval=eigval[idx];eigvec=eigvec[:,idx]
     frac=np.cumsum(eigval)/eigval.sum();n=int(np.searchsorted(frac,.90)+1);n=max(4,min(16,n));Y=Z@eigvec[:,:n]
     return Y.astype(np.float32),mu.astype(np.float32),sd.astype(np.float32),eigvec[:,:n].astype(np.float32),eigval.astype(np.float32),n,float(frac[n-1])
    def kmeans_gpu(X,k=16,n_init=20,seed=260924,max_iter=100):
     x=torch.tensor(X,dtype=torch.float32,device='cuda');n=len(x);best=None;best_in=float('inf');g=torch.Generator(device='cuda');g.manual_seed(seed)
     for init in range(n_init):
      # deterministic random centers from fixed generator; no labels
      perm=torch.randperm(n,generator=g,device='cuda')[:k];c=x[perm].clone();last=None
      for it in range(max_iter):
       d2=(x*x).sum(1,keepdim=True)+(c*c).sum(1)[None,:]-2*x@c.T;lab=d2.argmin(1)
       if last is not None and torch.equal(lab,last):break
       last=lab
       sums=torch.zeros_like(c);sums.index_add_(0,lab,x);cnt=torch.bincount(lab,minlength=k).float().unsqueeze(1)
       empty=(cnt.squeeze(1)==0)
       c=sums/cnt.clamp_min(1)
       if empty.any():c[empty]=x[torch.randperm(n,generator=g,device='cuda')[:int(empty.sum())]]
      inertia=float(((x-c[lab])**2).sum().cpu())
      if inertia<best_in:best_in=inertia;best=(lab.cpu().numpy().astype(np.int16),c.cpu().numpy(),init,it+1)
     return best[0],best[1],best_in,best[2],best[3]
    def load_model(path,obs_dim):
     from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
     m=V2BSingleSiteFiLMActorCritic(obs_dim,12).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval();return m
    def state_metrics(ms,mt,obs,axis):
     from torch.func import jacrev,vmap
     o=torch.tensor(obs,dtype=torch.float32,device='cuda');wh=torch.tensor(HEAVY[axis],device='cuda').repeat(len(o),1);wc=torch.tensor(CENTER,device='cuda').repeat(len(o),1)
     with torch.no_grad():
      sh=ms.act_inference_with_preference(o,wh);sc=ms.act_inference_with_preference(o,wc);th=mt.act_inference_with_preference(o,wh);tc=mt.act_inference_with_preference(o,wc)
      rs=sh-sc;rt=th-tc
      A_center=torch.linalg.vector_norm(tc-sc,dim=1).cpu().numpy();A_heavy=torch.linalg.vector_norm(th-sh,dim=1).cpu().numpy();R_change=torch.linalg.vector_norm(rt-rs,dim=1).cpu().numpy()
      R_rotation=(1-(rs*rt).sum(1)/(torch.linalg.vector_norm(rs,dim=1)*torch.linalg.vector_norm(rt,dim=1)+EPS)).cpu().numpy()
     w0=torch.tensor(CENTER,dtype=torch.float32,device='cuda')
     def fs(x,w):return ms.act_inference_with_preference(x.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
     def ft(x,w):return mt.act_inference_with_preference(x.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
     js=[];jt=[]
     for k in range(0,len(o),128):
      js.append(vmap(jacrev(fs,argnums=1),in_dims=(0,None))(o[k:k+128],w0).detach());jt.append(vmap(jacrev(ft,argnums=1),in_dims=(0,None))(o[k:k+128],w0).detach())
     js=torch.cat(js);jt=torch.cat(jt);diff=jt-js;ns=torch.linalg.vector_norm(js.reshape(len(js),-1),dim=1);nd=torch.linalg.vector_norm(diff.reshape(len(diff),-1),dim=1)
     J_change_rel=(nd/(ns+EPS)).cpu().numpy();J_rotation=(1-(js.reshape(len(js),-1)*jt.reshape(len(jt),-1)).sum(1)/(torch.linalg.vector_norm(js.reshape(len(js),-1),dim=1)*torch.linalg.vector_norm(jt.reshape(len(jt),-1),dim=1)+EPS)).cpu().numpy()
     return {'A_center':A_center,'A_heavy':A_heavy,'R_change':R_change,'R_rotation':R_rotation,'J_change_rel':J_change_rel,'J_rotation':J_rotation}
    def main():
     rep=json.load(open(V18));samples=rep['samples'];states=rep['states'];cp={}
     for sd,arr in states.items():
      for st in arr:cp[(int(sd),st['label'])]=ROOT/st['checkpoint']
     # gather source-state matrix only
     trans=[];allX=[];offset=0
     for s in samples:
      ps=cp[(s['seed'],s['from'])];f=SC/f'{mh(ps)}_{s["axis"]}.npz';z=np.load(f,allow_pickle=False);obs=z['obs'];flat=obs.reshape(-1,obs.shape[-1]);n=len(flat)
      phase=np.tile(np.repeat(np.arange(4),16*8),4) # suite-major: per suite Q1..Q4 each 16*8
      assert len(phase)==n
      trans.append({'seed':s['seed'],'from':s['from'],'to':s['to'],'axis':s['axis'],'Y':s['Y_robust_collapse'],'source_G':s['source_G_sem'],'ps':ps,'pt':cp[(s['seed'],s['to'])],'obs':flat,'phase':phase,'start':offset,'stop':offset+n})
      allX.append(flat);offset+=n
     X=np.concatenate(allX,0);Yp,mu,sd,pcs,eig,npc,expl=pca_fit_transform(X);labels,centers,inertia,binit,kit=kmeans_gpu(Yp,K,NINIT,SEED)
     np.savez_compressed(RUN/'manifold_partition.npz',mean=mu,std=sd,pcs=pcs,eigenvalues=eig,n_components=np.array(npc),explained=np.array(expl),labels=labels,centers=centers,inertia=np.array(inertia),best_init=np.array(binit),kmeans_iters=np.array(kit))
     # state context per cluster, label blind
     context={}
     for c in range(K):
      xx=X[labels==c];ph=[]
      # gather phases using offsets
      for t in trans:ph.append(t['phase'][labels[t['start']:t['stop']]==c])
      ph=np.concatenate(ph) if ph else np.array([],int)
      med=lambda v:float(np.median(v)) if len(v) else None
      context[str(c)]={'n_states':int(len(xx)),'phase_fraction':[float(np.mean(ph==q)) for q in range(4)] if len(ph) else [0]*4,
       'base_lin_speed_median':med(np.linalg.norm(xx[:,0:3],axis=1)),'base_ang_speed_median':med(np.linalg.norm(xx[:,3:6],axis=1)),'gravity_xy_median':med(np.linalg.norm(xx[:,6:8],axis=1)),
       'command_xyz_median':np.median(xx[:,9:12],axis=0).tolist() if len(xx) else None,'joint_pos_norm_median':med(np.linalg.norm(xx[:,12:24],axis=1)),'joint_vel_norm_median':med(np.linalg.norm(xx[:,24:36],axis=1)),'prev_action_norm_median':med(np.linalg.norm(xx[:,36:48],axis=1))}
     # models cache + transition metrics
     mods={};rows=[]
     for ti,t in enumerate(trans,1):
      print('TRANS',ti,'/',len(trans),t['seed'],t['from'],'->',t['to'],t['axis'],'Y',t['Y'],flush=True)
      for p in (t['ps'],t['pt']):
       if str(p) not in mods:mods[str(p)]=load_model(p,X.shape[1])
      met=state_metrics(mods[str(t['ps'])],mods[str(t['pt'])],t['obs'],t['axis']);lab=labels[t['start']:t['stop']]
      globalm={m:float(np.mean(v)) for m,v in met.items()};clusters={}
      for c in range(K):
       ix=np.where(lab==c)[0];entry={'n':int(len(ix)),'occupancy':float(len(ix)/len(lab)),'phase_counts':[int(np.sum(t['phase'][ix]==q)) for q in range(4)]}
       if len(ix)>=MINST:
        for m,v in met.items():entry[m]={'mean':float(np.mean(v[ix])),'p90':float(np.percentile(v[ix],90)),'mass':float((len(ix)/len(lab))*np.mean(v[ix]))}
       clusters[str(c)]=entry
      rows.append({'seed':t['seed'],'from':t['from'],'to':t['to'],'axis':t['axis'],'Y_robust_collapse':t['Y'],'source_G_sem':t['source_G'],'global':globalm,'clusters':clusters})
     # global controls
     y=np.array([r['Y_robust_collapse'] for r in rows]);global_auc={m:auc(y,[r['global'][m] for r in rows]) for m in PRIMARY+SECONDARY}
     regional=[];passing=[];approach=[]
     for c in range(K):
      for m in PRIMARY:
       elig=[r for r in rows if r['clusters'][str(c)]['n']>=MINST];pos=[r for r in elig if r['Y_robust_collapse']];neg=[r for r in elig if not r['Y_robust_collapse']]
       if not pos or not neg:continue
       yp=np.array([r['Y_robust_collapse'] for r in elig]);meanv=np.array([r['clusters'][str(c)][m]['mean'] for r in elig]);p90=np.array([r['clusters'][str(c)][m]['p90'] for r in elig]);mass=np.array([r['clusters'][str(c)][m]['mass'] for r in elig])
       A=auc(yp,meanv);Ap=auc(yp,p90);Am=auc(yp,mass);nv=np.array([r['clusters'][str(c)][m]['mean'] for r in neg]);pv=np.array([r['clusters'][str(c)][m]['mean'] for r in pos]);medn=float(np.median(nv));iqr=float(np.percentile(nv,75)-np.percentile(nv,25));effect=(float(np.median(pv))-medn)/max(iqr,EPS);count=int(np.sum(pv>medn))
       seedok=True;seeddata={}
       for seed in (983001,984001,985001):
        pp=[r['clusters'][str(c)][m]['mean'] for r in pos if r['seed']==seed];nn=[r['clusters'][str(c)][m]['mean'] for r in neg if r['seed']==seed];ok=(len(pp)>0 and len(nn)>0 and np.median(pp)>np.median(nn));seedok &= ok;seeddata[str(seed)]={'robust_n':len(pp),'retained_n':len(nn),'direction_ok':bool(ok),'robust_median':float(np.median(pp)) if pp else None,'retained_median':float(np.median(nn)) if nn else None}
       crit={'robust_eligible_ge8':len(pos)>=8,'retained_eligible_ge8':len(neg)>=8,'auc_ge0p75':A>=.75,'effect_ge0p5_iqr':effect>=.5,'robust_above_retained_median_ge8':count>=8,'seed_consistent_3of3':bool(seedok),'beats_global_auc_by0p05':A>=global_auc[m]+.05,'p90_or_mass_auc_ge0p70':max(Ap,Am)>=.70}
       rec={'cluster':c,'metric':m,'robust_eligible':len(pos),'retained_eligible':len(neg),'auc_mean':A,'auc_p90':Ap,'auc_mass':Am,'global_auc':global_auc[m],'auc_gain':A-global_auc[m],'robust_median':float(np.median(pv)),'retained_median':medn,'retained_iqr':iqr,'effect_iqr_units':effect,'robust_above_retained_median_n':count,'seed_consistent':bool(seedok),'per_seed':seeddata,'criteria':crit,'passes':all(crit.values())}
       regional.append(rec)
       if rec['passes']:passing.append(rec)
       if A>=.70 and seedok and A>=global_auc[m]+.03:approach.append(rec)
     # occupancy differences transition-level
     occupancy={}
     for c in range(K):
      p=[r['clusters'][str(c)]['occupancy'] for r in rows if r['Y_robust_collapse']];n=[r['clusters'][str(c)]['occupancy'] for r in rows if not r['Y_robust_collapse']]
      occupancy[str(c)]={'robust_median':float(np.median(p)),'retained_median':float(np.median(n)),'auc':auc(y,[r['clusters'][str(c)]['occupancy'] for r in rows])}
     # phase profile only for passing/approach clusters; rewrite mass distribution in robust states
     cand=sorted(set([r['cluster'] for r in passing+approach]));phase_profiles={}
     for c in cand:
      phase_profiles[str(c)]={}
      for m in PRIMARY:
       masses=np.zeros(4);tot=0.0
       for r,t in zip(rows,trans):
        if not r['Y_robust_collapse']:continue
        lab=labels[t['start']:t['stop']];met=state_metrics(mods[str(t['ps'])],mods[str(t['pt'])],t['obs'],t['axis'])[m]
        for q in range(4):
         ix=(lab==c)&(t['phase']==q);val=float(np.sum(met[ix])) if np.any(ix) else 0;masses[q]+=val;tot+=val
       phase_profiles[str(c)][m]={'mass_fraction':(masses/(tot+EPS)).tolist(),'phase_concentrated':bool(np.max(masses/(tot+EPS))>=.40)}
     status='REPRODUCIBLE LOCALIZED REWRITE REGION' if passing else ('DISTRIBUTED / WEAK LOCALIZATION' if approach else 'NO REPRODUCIBLE LOCALIZATION')
     report={'schema':'state_manifold_localization_audit_v1','status':status,'primary_n':len(rows),'robust_n':int(y.sum()),'retained_n':int(len(y)-y.sum()),'manifold':{'K':K,'pca_components':npc,'pca_explained_variance':expl,'kmeans_inertia':inertia,'best_init':binit,'iterations':kit},'global_auc':global_auc,'cluster_context':context,'cluster_occupancy':occupancy,'regional_tests':regional,'passing_regions':passing,'approaching_regions':approach,'phase_profiles':phase_profiles,'rows':rows,'decision':{'training_method_authorized':False}}
     op=RUN/'state_manifold_localization_report.json';op.write_text(json.dumps(report,indent=2)+'\n');(RUN/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':status,'report_sha256':sha(op),'contract_sha256':sha(ROOT/'docs/contracts/diagnostics/state-manifold-localization-audit-contract.md'),'script_sha256':sha(Path(__file__).resolve()),'v18_sha256':sha(V18),'v20_state_cache_files':len(list(SC.glob('*.npz')))},indent=2)+'\n')
     print(json.dumps({'status':status,'manifold':report['manifold'],'global_auc':global_auc,'passing_regions':passing,'approaching_regions':approach},indent=2))
    if True:main()

STAGES = {
    "context_incremental_retention_audit": run_context_incremental_retention_audit,
    "d1_same_state_probe": run_d1_same_state_probe,
    "final_eval_manifest": run_final_eval_manifest,
    "robust_collapse_mechanism_audit": run_robust_collapse_mechanism_audit,
    "state_manifold_localization_audit": run_state_manifold_localization_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
