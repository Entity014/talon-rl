"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_freeze_v1a_e0():
    """Run former freeze_v1a_e0.py stage."""
    """Freeze the prospective V1A-E0 evaluation protocol.
    
    This file only inventories protocol inputs and hashes; it does not load model
    weights or run an evaluation.  V1-A formulation remains immutable.
    """
    
    import hashlib
    import json
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    
    
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def main() -> None:
        freeze = json.loads((ROOT / "artifacts/v1a/V1A_FREEZE.json").read_text())
        source = json.loads((ROOT / "artifacts/v1a/V1A_M01_SOURCE_BINDING.json").read_text())
        stock_cfg = Path(
            "/home/xero/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
            "manager_based/locomotion/velocity/config/a1/flat_env_cfg.py"
        )
        runner_cfg = Path(
            "/home/xero/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
            "manager_based/locomotion/velocity/config/a1/agents/rsl_rl_ppo_cfg.py"
        )
        protocol_script = ROOT / "scripts/rl/experiments/common/utilities/v1a_e0_eval.py"
        if not protocol_script.exists():
            raise FileNotFoundError(protocol_script)
    
        protocol = {
            "schema": "v1a_e0_protocol_v1",
            "protocol_id": "V1A-E0",
            "status": "FROZEN_PROSPECTIVE",
            "purpose": "Paired stock-protocol preservation evaluation for M0.1 and V1-A",
            "training_formulation_unchanged": True,
            "source_formulation_hash": freeze["hashes"]["v1a_formulation_hash"],
            "source_baseline_hash": source["source_baseline_hash"],
            "environment": {
                "task": "Isaac-Velocity-Flat-Unitree-A1-v0",
                "stock_config": str(stock_cfg),
                "stock_config_sha256": sha(stock_cfg),
                "runner_config": str(runner_cfg),
                "runner_config_sha256": sha(runner_cfg),
                "backend": "Isaac Lab + rsl_rl",
            },
            "reset_suite": {
                "generation": "gymnasium reset(seed=reset_seed) on stock A1 env",
                "reset_seed_base": 17001,
                "seed_mapping": "reset_seed = reset_seed_base + evaluation_seed",
                "num_envs": 64,
                "paired_reset_required": True,
                "initial_observation_hashes_recorded": True,
            },
            "rollout": {
                "horizon_steps": 500,
                "command_source": "stock command manager",
                "action": "deterministic rsl_rl actor mean, inherited action semantics",
                "checkpoint": "terminal model_299.pt only",
                "seed_matching": True,
                "pairs": "M0.1 seed s and V1-A seed s share reset seed and suite",
            },
            "metrics": [
                "survival",
                "velocity_tracking_error",
                "tilt_p95_deg",
                "max_tilt_deg",
                "termination_rate",
                "base_contact_rate",
                "finite_state",
                "evaluator_non_mutation",
            ],
            "comparison": {
                "baseline_re_evaluated_under_e0": True,
                "absolute_thresholds": freeze["evaluation"]["thresholds"],
                "threshold_provenance": "recorded_from_v1a_freeze; provenance must be reported, not reconstructed",
                "paired_degradation_rule": "not applied until prospective margins are explicitly frozen",
                "no_posthoc_threshold_selection": True,
            },
            "integrity": {
                "checkpoint_hash_validation": True,
                "config_hash_validation": True,
                "actor_critic_std_snapshot_before_after": True,
                "normalizer_snapshot_before_after": True,
                "rng_state_snapshot_before_after": True,
                "training_mode": False,
                "optimizer_step": False,
            },
            "evaluator": {
                "script": str(protocol_script),
                "script_sha256": sha(protocol_script),
                "must_run_before_verdict": True,
            },
            "verdict": {
                "terminal_checkpoints_only": True,
                "all_three_seed_pairs_required": True,
                "pass": "all seed pairs satisfy the frozen E0 rule",
                "fail": "any seed pair violates the frozen E0 rule",
                "otherwise": "INCONCLUSIVE_PROTOCOL_OR_INTEGRITY_FAILURE",
            },
            "v1b_authorized": False,
        }
        out = ROOT / "artifacts/v1a/V1A_E0_PROTOCOL.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        print(out)
    
    
    if True:
        main()

def run_freeze_v1a_e05():
    """Run former freeze_v1a_e05.py stage."""
    """Freeze the baseline-only V1A-E0.5 calibration protocol."""
    
    import hashlib
    import json
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    
    
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def main() -> None:
        e0 = json.loads((ROOT / "artifacts/v1a/V1A_E0_PROTOCOL.json").read_text())
        evaluator = ROOT / "scripts/rl/experiments/architectures/shared_policy/v1_family/v1a/v1a_e05_calibrate.py"
        protocol = {
            "schema": "v1a_e05_protocol_v1",
            "protocol_id": "V1A-E0.5",
            "status": "FROZEN_PROSPECTIVE",
            "purpose": "Estimate M0.1-only stock-A1 evaluation variability before E1 margin freeze",
            "subjects": "M0.1 terminal checkpoints only; V1-A checkpoints forbidden",
            "source_baseline_hash": e0["source_baseline_hash"],
            "environment": e0["environment"],
            "suite_design": {
                "suite_count": 8,
                "states_per_suite": 64,
                "reset_seed_base": 27001,
                "seed_mapping": "reset_seed = reset_seed_base + suite_index * 101 + evaluation_seed",
                "evaluation_seeds": [0, 1, 2],
                "fresh_relative_to": ["V1A-E0", "V1A-E1"],
                "paired_across_m01_seeds": True,
            },
            "rollout": {
                "horizon_steps": 500,
                "action": "deterministic raw rsl_rl actor mean",
                "checkpoint": "M0.1 model_299.pt only",
                "normalizer_optimizer_rng_mutation": False,
            },
            "metrics": [
                "survival",
                "velocity_tracking_error",
                "tilt_p95_deg",
                "max_tilt_deg",
                "termination_rate",
                "base_contact_rate",
                "finite_state",
                "evaluator_non_mutation",
            ],
            "outputs": {
                "purpose": "descriptive baseline variability only",
                "margin_selection": "not performed by this protocol",
                "engineering_allowance": "not specified by this protocol",
                "binary_verdict": "forbidden",
            },
            "integrity": {
                "checkpoint_hash_validation": True,
                "lifecycle_markers": True,
                "v1a_checkpoint_access": False,
                "e0_checkpoint_reuse": False,
            },
            "evaluator": {
                "script": str(evaluator),
                "script_sha256": sha(evaluator),
                "e0_reset_seed_base": e0["reset_suite"]["reset_seed_base"],
                "e05_reset_seed_base": 27001,
            },
            "next_step": "independent engineering allowance review and prospective E1 margin freeze",
            "v1b_authorized": False,
        }
        out = ROOT / "artifacts/v1a/V1A_E05_PROTOCOL.json"
        out.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        print(out)
    
    
    if True:
        main()

def run_freeze_v1a_e06():
    """Run former freeze_v1a_e06.py stage."""
    """Freeze the baseline-only V1A-E0.6 deployment-contract audit."""
    
    import hashlib
    import json
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    
    
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def main() -> None:
        e05 = json.loads((ROOT / "artifacts/v1a/V1A_E05_PROTOCOL.json").read_text())
        wrapper = Path("/home/xero/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py")
        actor = Path("/home/xero/isaac-lab-env/lib/python3.11/site-packages/rsl_rl/modules/actor_critic.py")
        evaluator = ROOT / "scripts/rl/experiments/architectures/shared_policy/v1_family/v1a/v1a_e06_audit.py"
        out = {
            "schema": "v1a_e06_protocol_v1",
            "protocol_id": "V1A-E0.6",
            "status": "FROZEN_PROSPECTIVE",
            "purpose": "Verify M0.1 deployment/inference semantics before any E1 gate",
            "scope": "M0.1 terminal checkpoints only; V1-A checkpoint access forbidden",
            "environment": e05["environment"],
            "evaluation": {
                "seeds": [0, 1, 2],
                "num_envs": 64,
                "steps": 500,
                "reset_seed_base": 37001,
                "deterministic_actor": "rsl_rl ActorCritic.act_inference",
                "reference_action_path": "RslRlVecEnvWrapper.step with clip_actions=1.0",
                "raw_vs_clipped_action_statistics": True,
            },
            "audit_questions": [
                "Does M0.1 inference use action clipping in the stock path?",
                "Does the stock-clipped path restore deterministic competence?",
                "Are observation grouping, command, reset, termination, and std semantics aligned?",
            ],
            "integrity": {
                "v1a_checkpoint_access": False,
                "training": False,
                "optimizer_step": False,
                "state_nonmutation": True,
            },
            "hashes": {
                "vecenv_wrapper": sha(wrapper),
                "rsl_actor_critic": sha(actor),
                "evaluator": sha(evaluator),
            },
            "verdict": "AUDIT_ONLY_NO_V1A_VERDICT",
            "v1b_authorized": False,
        }
        path = ROOT / "artifacts/v1a/V1A_E06_PROTOCOL.json"
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(path)
    
    
    if True:
        main()

def run_freeze_v1a_e1r():
    """Run former freeze_v1a_e1r.py stage."""
    """Freeze the corrected V1A-E1R measurement contract."""
    
    import hashlib
    import json
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    
    
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def main() -> None:
        e06 = json.loads((ROOT / "artifacts/v1a/V1A_E06_PROTOCOL.json").read_text())
        evaluator = ROOT / "scripts/rl/experiments/architectures/shared_policy/v1_family/v1a/v1a_e1r_eval.py"
        protocol = {
            "schema": "v1a_e1r_protocol_v1",
            "protocol_id": "V1A-E1R",
            "status": "FROZEN_MEASUREMENT_PENDING_MARGIN_CALIBRATION",
            "purpose": "Corrected paired preservation evaluation under exact stock deployment semantics",
            "causal_question": "Does V1-A preserve M0.1 locomotion competence when both use the same stock action path?",
            "inheritance": {
                "environment": e06["environment"],
                "action_semantics": "raw rsl_rl deterministic actor mean followed by clip_actions=1.0",
                "command_reset_termination": "stock M0.1 semantics",
                "distribution_backend": "rsl_rl",
            },
            "evaluation": {
                "seed_matched_pairs": True,
                "seeds": [0, 1, 2],
                "num_envs": 64,
                "steps": 500,
                "reset_seed_base": 47001,
                "fresh_relative_to": ["V1A-E0", "V1A-E0.5", "V1A-E0.6"],
                "terminal_checkpoint": "model_299.pt only",
                "subjects": ["M0.1", "V1-A"],
            },
            "metrics": ["survival", "velocity_tracking_error", "tilt_p95_deg", "max_tilt_deg", "termination_rate", "base_contact_rate", "finite_state", "evaluator_non_mutation"],
            "integrity": {
                "checkpoint_hash_validation": True,
                "paired_reset_hash_validation": True,
                "training": False,
                "optimizer_step": False,
                "state_nonmutation": True,
            },
            "decision": {
                "margin_source": "corrected M0.1-only calibration required before binary verdict",
                "absolute_safety_floor": "must be independently justified or explicitly deferred",
                "posthoc_thresholding": False,
                "verdict": "NOT_AUTHORIZED_UNTIL_MARGIN_CONTRACT_IS_FROZEN",
            },
            "invalidated_evidence": {
                "e0": "descriptive only; raw unclipped action path mismatch",
                "e05": "not valid for competence or margin calibration; raw unclipped action path mismatch",
            },
            "evaluator": {"script": str(evaluator), "script_sha256": sha(evaluator)},
            "v1b_authorized": False,
        }
        out = ROOT / "artifacts/v1a/V1A_E1R_PROTOCOL.json"
        out.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        print(out)
    
    
    if True:
        main()

def run_freeze_v1a_e1r_cal():
    """Run former freeze_v1a_e1r_cal.py stage."""
    import hashlib, json
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[4]
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
        e1r=json.loads((ROOT/'artifacts/v1a/V1A_E1R_PROTOCOL.json').read_text())
        ev=ROOT/'scripts/rl/experiments/architectures/shared_policy/v1_family/v1a/v1a_e1r_calibrate.py'
        out={
          'schema':'v1a_e1r_cal_protocol_v1','protocol_id':'V1A-E1R-CAL','status':'FROZEN_PROSPECTIVE',
          'purpose':'Corrected M0.1-only variability calibration for E1R margins',
          'subjects':'M0.1 terminal checkpoints only; V1-A forbidden',
          'action_semantics':'raw rsl_rl deterministic actor mean followed by clip_actions=1.0',
          'environment':e1r['inheritance']['environment'],
          'suite_design':{'suite_count':8,'states_per_suite':64,'evaluation_seeds':[0,1,2],
            'reset_seed_base':57001,'seed_mapping':'reset_seed=base+suite_index*101+evaluation_seed',
            'fresh_relative_to':['V1A-E0','V1A-E0.5','V1A-E0.6','V1A-E1R']},
          'rollout':{'steps':500,'terminal_checkpoint':'model_299.pt','deterministic':True,'training':False},
          'metrics':['survival','velocity_tracking_error','tilt_p95_deg','max_tilt_deg','termination_rate','base_contact_rate','finite_state','evaluator_non_mutation'],
          'outputs':{'descriptive_only':True,'margin_selection':False,'binary_verdict':'forbidden'},
          'evaluator':{'script':str(ev),'script_sha256':sha(ev)},'v1b_authorized':False}
        p=ROOT/'artifacts/v1a/V1A_E1R_CAL_PROTOCOL.json';p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(p)
    if True: main()

STAGES = {
    "freeze_v1a_e0": run_freeze_v1a_e0,
    "freeze_v1a_e05": run_freeze_v1a_e05,
    "freeze_v1a_e06": run_freeze_v1a_e06,
    "freeze_v1a_e1r": run_freeze_v1a_e1r,
    "freeze_v1a_e1r_cal": run_freeze_v1a_e1r_cal,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
