"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_l0_c0_audit():
    """Run former l0_c0_audit.py stage."""
    """Why does L0-B seed 2 reach a good basin while seeds 0 and 1 do not?
    
    Read-only. Compares the scalar training telemetry, the final command-cell
    behaviour and a coarse actor-parameter norm across the three seeds. It cannot
    attribute causality, because L0-B logged aggregate reward rather than the
    reward-term decomposition; that limit is stated in the report it writes.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import ARTIFACTS, RUNS, OfflineAudit
    
    SEEDS = [(0, "l0b_seed0_2026-09-21", "l0b_seed0_eval"),
             (1, "l0b_seed1_2026-09-21", "l0b_seed1_eval"),
             (2, "l0b_seed2_2026-09-22", "l0b_seed2_eval")]
    CELL_KEYS = ("survival_rate", "mean_abs_vx_error", "tilt_p95_deg",
                 "tilt_max_deg", "base_contact_rate")
    INTERPRETATION = (
        "Final command-conditioned behavior separates strongly while all runs remain "
        "finite and use the same frozen PPO contract. This supports seed-dependent "
        "basin selection/task sensitivity, but cannot attribute causality to a "
        "specific reward term because L0-B did not log decomposed reward terms.")
    
    
    def span(x):
        return {"first": float(x[0]), "last": float(x[-1]),
                "min": float(x.min()), "max": float(x.max())}
    
    
    class L0C0ResidualAudit(OfflineAudit):
        """L0-B seed-2 pass versus seed-0/1 fail, from what was logged."""
    
        root = ARTIFACTS
        run = "l0_c0_audit"
        report = "L0C0_AUDIT.json"
        sort_keys = True
        schema = "l0_c0_residual_audit_v1"
    
        def seed_record(self, seed, run_name, eval_name):
            rows = json.loads((RUNS / run_name / "training_metrics.json").read_text())["metrics"]
            ev = json.loads((ARTIFACTS / eval_name / "evaluation.json").read_text())
            reward = np.array([r["reward_mean"] for r in rows], float)
            kl = np.array([r["analytic_kl"] for r in rows], float)
            lr = np.array([r["learning_rate"] for r in rows], float)
            std = np.array([r["learned_std"] for r in rows], float)
            return {
                "run": run_name,
                "updates": len(rows),
                "reward_mean": {"first25": float(reward[:25].mean()),
                                "last25": float(reward[-25:].mean()),
                                "min": float(reward.min()), "max": float(reward.max())},
                "analytic_kl": {"mean": float(kl.mean()),
                                "p95": float(np.percentile(kl, 95)), "max": float(kl.max())},
                "learning_rate": span(lr),
                "learned_std": span(std),
                "cells": {k: {x: v[x] for x in CELL_KEYS} for k, v in ev["cells"].items()},
                "telemetry_finite": bool(np.isfinite(reward).all() and np.isfinite(kl).all()
                                         and np.isfinite(lr).all() and np.isfinite(std).all())}
    
        def analyze(self):
            seeds = {str(s): self.seed_record(s, r, e) for s, r, e in SEEDS}
            # actor parameter norm at the last checkpoint, as a coarse drift signal
            for seed, run_name, _ in SEEDS:
                ck = torch.load(RUNS / run_name / "checkpoints/update_500.pt", map_location="cpu")
                vals = [float(v.float().norm()) for k, v in ck["model"].items()
                        if k.startswith("actor_") and torch.is_floating_point(v)]
                seeds[str(seed)]["actor_param_l2_sum"] = float(sum(vals))
                seeds[str(seed)]["checkpoint_schema"] = ck.get("schema")
            return {
                "schema": self.schema,
                "status": "COMPLETE_READ_ONLY",
                "question": "why does L0-B seed2 reach a good basin while seed0/1 do not?",
                "seed2_is_pass_control": True,
                "available_telemetry": ["reward_mean", "analytic_kl", "adaptive_learning_rate",
                                        "learned_std", "checkpoint_actor_parameter_norm",
                                        "final_command_cells"],
                "not_recorded_in_l0b": ["reward_term_decomposition",
                                        "episode_termination_timing_by_update",
                                        "joint_action_statistics",
                                        "height_tilt_contact_training_traces",
                                        "critic_value_error",
                                        "observation/action distribution traces"],
                "seeds": seeds,
                "interpretation": INTERPRETATION}
    
        def summarize(self, report):
            lines = ["# L0-C0 residual audit", "", "Status: **COMPLETE — read-only**", "",
                     report["interpretation"], "", "## Final cell summary", ""]
            for seed, d in report["seeds"].items():
                lines.append(f"### seed {seed}")
                for cmd, m in d["cells"].items():
                    lines.append(
                        f"- vx={cmd}: survival={m['survival_rate']:.3f}, "
                        f"vx MAE={m['mean_abs_vx_error']:.3f}, "
                        f"tilt p95={m['tilt_p95_deg']:.2f}°, max tilt={m['tilt_max_deg']:.2f}°, "
                        f"base contact={m['base_contact_rate']:.3f}")
            lines += ["", "## Evidence limits", "",
                      "- L0-B logged aggregate scalar reward, not the underlying reward-term decomposition.",
                      "- No per-update trajectory/contact/joint-action telemetry was stored, so the first physical divergence cannot be localized retrospectively.",
                      "- The next causal data collection should add these fields prospectively; this audit does not authorize a new formulation or training run."]
            (self.out / "report.md").write_text("\n".join(lines) + "\n")
            print(self.out / self.report)
    
    
    if True:
        L0C0ResidualAudit.main()

def run_l0_mapping_audit():
    """Run former l0_mapping_audit.py stage."""
    """Emit the frozen, source-backed L0 reference/task mapping audit.
    
    This is intentionally read-only: it does not construct Isaac Lab or launch a
    simulation.  The audit records the source hashes and the equivalence claims
    that must be resolved before an L0 training authorization.
    """
    
    import hashlib
    import json
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    OUT = ROOT / "artifacts/l0_mapping_audit"
    
    
    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()
    
    
    def main() -> None:
        stock = Path("/home/xero/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity")
        sources = {
            "stock_velocity_cfg": stock / "velocity_env_cfg.py",
            "stock_a1_rough_cfg": stock / "config/a1/rough_env_cfg.py",
            "stock_a1_flat_cfg": stock / "config/a1/flat_env_cfg.py",
            "stock_rsl_ppo_cfg": stock / "config/a1/agents/rsl_rl_ppo_cfg.py",
            "custom_a1_cfg": ROOT / "talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py",
            "custom_a1_env": ROOT / "talon_rl/tasks/locomotion/a1_env/a1_env.py",
            "gate0b_runner": ROOT / "scripts/rl/experiments/baselines/scalar_substrate/substrate_gate.py",
            "b0_reward": ROOT / "talon_rl/rewards/baselines.py",
            "b0_wrapper": ROOT / "talon_rl/wrappers/scalar_reward_env.py",
        }
        missing = [str(p) for p in sources.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("missing audit source(s): " + ", ".join(missing))
    
        mapping = [
            {
                "field": "task_identity",
                "stock_reference": "Isaac Lab Unitree A1 flat velocity task; manager-based reward/termination/command managers",
                "gate0b": "custom Isaac-Talon A1 wrapped by B0 scalar reward; nominalized flat plane",
                "status": "NOT_EQUIVALENT",
                "implication": "A stock result cannot by itself identify a B0 reward/env failure.",
            },
            {
                "field": "policy_observation",
                "stock_reference": "flat policy concatenation: base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12) = 45; corruption enabled in training; height scan disabled on flat",
                "gate0b": "51 dims: absolute joint_pos(12), joint_vel(12), roll_pitch(2), binary foot_contact(4), last_action(12), v_command(3), base_ang_vel(3), projected_gravity(3); corruption disabled",
                "status": "NOT_EQUIVALENT",
                "implication": "Observation semantics and noise differ; this is not an algorithm-only comparison.",
            },
            {
                "field": "joint_observation_semantics",
                "stock_reference": "relative joint position to default pose; joint velocity relative",
                "gate0b": "absolute joint position; scaled joint velocity",
                "status": "NOT_EQUIVALENT",
                "implication": "The actor sees a different coordinate system even when dimensions are similar.",
            },
            {
                "field": "action",
                "stock_reference": "joint-position target, default offset, scale 0.25 in A1 rough/flat config",
                "gate0b": "joint-position target, custom scale 0.15, custom ActorCritic action clipping path",
                "status": "NOT_EQUIVALENT",
                "implication": "Action authority and effective PD target range differ.",
            },
            {
                "field": "reset_state",
                "stock_reference": "random x/y ±0.5, yaw ±pi, zero velocity and joint scale 1.0 in A1 rough override",
                "gate0b": "custom reset_scene_to_default plus fresh frozen Gate-0B reset artifact; no stock reset randomization is retained by nominalize",
                "status": "PARTIAL",
                "implication": "The frozen reset suite is an explicit contract, not stock reset behavior.",
            },
            {
                "field": "command",
                "stock_reference": "UniformVelocityCommand manager, sampled ranges and held for 10 s; nominal stock ranges include vx/vy/yaw",
                "gate0b": "fixed forward cells vx={0.25,0.50,0.75}, vy=0, yaw=0; one command held per rollout and cycled by update index",
                "status": "NOT_EQUIVALENT",
                "implication": "Command coverage must be reported separately from stock reproduction.",
            },
            {
                "field": "reward",
                "stock_reference": "weighted velocity tracking, angular tracking, vertical/angular velocity, torque, acceleration, action-rate, feet-air-time, contacts and flat orientation terms",
                "gate0b": "scalar B0 reward: exp forward-vx tracking + roll/pitch penalty + height penalty + -10 terminal fall penalty",
                "status": "NOT_EQUIVALENT",
                "implication": "Option A and Option B answer different causal questions; stock reward must not be silently mixed into B0.",
            },
            {
                "field": "termination_contact",
                "stock_reference": "illegal contact on configured base/trunk contact sensor, threshold 1.0, plus timeout",
                "gate0b": "trunk/base-contact termination plus timeout; terminal fall is read from pre-reset transition",
                "status": "CLOSE_BUT_VERIFY",
                "implication": "Body name, sensor history and pre/post-reset timing must remain explicitly checked.",
            },
            {
                "field": "horizon_and_control_rate",
                "stock_reference": "20 s episodes; physics dt 0.005, decimation 4, effective control period 0.02 s",
                "gate0b": "22 s nominalized episodes; Gate-0B runner overrides dt 0.01 and decimation 1, effective control period 0.01 s",
                "status": "NOT_EQUIVALENT",
                "implication": "Dynamics exposure and number of control decisions differ.",
            },
            {
                "field": "contact_semantics",
                "stock_reference": "all-body contact sensor history 3, air-time tracking; stock feet/thigh contact reward terms",
                "gate0b": "custom foot/trunk sensors history 1; scalar reward only charges trunk base-contact fall",
                "status": "NOT_EQUIVALENT",
                "implication": "Contact-derived observations/rewards are different, even if termination intent is similar.",
            },
            {
                "field": "randomization_curriculum",
                "stock_reference": "training config retains material/mass/reset randomization and interval pushes; flat removes terrain generator/curriculum but not all events",
                "gate0b": "nominalized plane, fixed robot properties, no push, no terrain curriculum, fixed command cells",
                "status": "NOT_EQUIVALENT",
                "implication": "Stock training is not a controlled reference for the nominal Gate-0B task unless disabled explicitly.",
            },
            {
                "field": "ppo_recipe",
                "stock_reference": "rsl_rl: rollout 24, 5 epochs/4 minibatches, learned std init 1.0, adaptive KL desired 0.01, lr 1e-3, gamma .99, value .5, entropy .01",
                "gate0b": "custom scalar PPO: rollout 16, full-batch update, fixed scheduled std .82→.10, lr 3e-4, gamma .998, value .5, entropy .001",
                "status": "NOT_EQUIVALENT",
                "implication": "This is the cleanest candidate variable for L0-B, provided task/reward stay fixed.",
            },
        ]
        report = {
            "schema": "l0_environment_reward_mapping_audit_v1",
            "status": "COMPLETE",
            "training_authorized": False,
            "conclusion": "Stock Isaac Lab A1 flat and Gate-0B are not field-equivalent. L0-A is a stack/task sanity reproduction; L0-B is the causal reference-PPO-on-our-task experiment.",
            "recommended_l0_0": "L0-B",
            "mapping": mapping,
            "source_sha256": {name: sha256(path) for name, path in sources.items()},
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "mapping.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"path": str(OUT / "mapping.json"), "status": report["status"], "recommended": report["recommended_l0_0"]}, indent=2))
    
    
    if True:
        main()

def run_l0_reference_inventory():
    """Run former l0_reference_inventory.py stage."""
    from pathlib import Path
    import hashlib,json
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'l0_reference_inventory'; OUT.mkdir(parents=True,exist_ok=True)
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
        cfg=Path('/home/xero/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/agents/rsl_rl_ppo_cfg.py'); flat=Path('/home/xero/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/flat_env_cfg.py'); ppo=Path('/home/xero/isaac-lab-env/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py')
        data={'schema':'l0_reference_inventory_v1','read_only':True,'reference':'Isaac Lab UnitreeA1FlatPPORunnerCfg + installed rsl_rl PPO','files':{'runner_cfg':str(cfg),'flat_env_cfg':str(flat),'ppo_impl':str(ppo)},'sha256':{'runner_cfg':sha(cfg),'flat_env_cfg':sha(flat),'ppo_impl':sha(ppo)},'settings':{'num_steps_per_env':24,'actor_hidden_dims':[128,128,128],'critic_hidden_dims':[128,128,128],'activation':'elu','init_noise_std':1.0,'actor_obs_normalization':False,'critic_obs_normalization':False,'num_learning_epochs':5,'num_mini_batches':4,'learning_rate':1e-3,'schedule':'adaptive','desired_kl':.01,'clip_param':.2,'value_loss_coef':1.0,'entropy_coef':.01,'gamma':.99,'lam':.95,'max_grad_norm':1.0},'status':'inventory_only','training_authorized':False}
        (OUT/'inventory.json').write_text(json.dumps(data,indent=2)+'\n'); print(OUT/'inventory.json')
    if True: main()

def run_l0a_screen_report():
    """Run former l0a_screen_report.py stage."""
    """Record the read-only result of the stock Isaac Lab L0-A screen."""
    import json, hashlib
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ROOT=Path(__file__).resolve().parents[4]
    LOG=Path('/home/xero/IsaacLab/logs/rsl_rl/unitree_a1_flat/2026-09-22_00-15-44_l0a_seed0_screen')
    def main():
        e=EventAccumulator(str(LOG)); e.Reload()
        def last(tag): return float(e.Scalars(tag)[-1].value)
        report={'schema':'l0a_stock_screen_v1','task':'Isaac-Velocity-Flat-Unitree-A1-v0','recipe':'installed Isaac Lab rsl_rl UnitreeA1FlatPPORunnerCfg','seed':0,'iterations':300,'status':'SCREEN_PASS','metrics':{'mean_reward':last('Train/mean_reward'),'mean_episode_length':last('Train/mean_episode_length'),'velocity_xy_error':last('Metrics/base_velocity/error_vel_xy'),'base_contact_rate':last('Episode_Termination/base_contact'),'timeout_rate':last('Episode_Termination/time_out'),'learned_std_start':float(e.Scalars('Policy/mean_noise_std')[0].value),'learned_std_final':last('Policy/mean_noise_std')},'log_dir':str(LOG),'interpretation':'Stock A1 flat task learns a stable nominal locomotion signal on seed0; proceed to 3-seed stock robustness confirmation. This is not a Gate-0B thesis verdict.'}
        out=ROOT/'artifacts/l0a_screen'; out.mkdir(parents=True,exist_ok=True); (out/'L0A_SCREEN.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2))
    if True: main()

STAGES = {
    "l0_c0_audit": run_l0_c0_audit,
    "l0_mapping_audit": run_l0_mapping_audit,
    "l0_reference_inventory": run_l0_reference_inventory,
    "l0a_screen_report": run_l0a_screen_report,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
