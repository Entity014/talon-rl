#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[4]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    canon=json.load(open(ROOT/"runs/authority_isolated_h2a_u30_semantic_validity-2026-09-25/semantic_report.json"))
    ctrl=json.load(open(ROOT/"runs/phase5_e2_source_control/semantic_report.json"))
    trt=json.load(open(ROOT/"runs/phase5_e2_source_ensemble/semantic_report.json"))
    ctr=json.load(open(ROOT/"runs/phase5_e2_control_seed75001/training_report.json"))
    trr=json.load(open(ROOT/"runs/phase5_e2_ensemble_seed75001/training_report.json"))
    init=ROOT/"artifacts/phase5_e2_initialization_manifest.json"
    result={
      "schema":"phase5_e2_screen_verdict_v1",
      "status":"FAIL_SOURCE_NO_REGRESSION",
      "date":"2026-09-26",
      "training_completed":{"control":True,"treatment":True,"fixed_endpoint_update":30},
      "source_reference":{
        "endpoint_pass":canon["endpoint_pass"],"critic":canon["critic"],
        "continuum_monotonicity":canon["continuum_summary"]["monotonicity_fraction"],
        "continuum_endpoint_between":canon["continuum_summary"]["endpoint_between_fraction"],
        "center_compromise":canon["center_compromise"]["between_heavy_envelope_fraction"]},
      "matched_control":{
        "endpoint_pass":ctrl["endpoint_pass"],"critic":ctrl["critic"],
        "continuum_monotonicity":ctrl["continuum_summary"]["monotonicity_fraction"],
        "continuum_endpoint_between":ctrl["continuum_summary"]["endpoint_between_fraction"],
        "center_compromise":ctrl["center_compromise"]["between_heavy_envelope_fraction"],
        "authority_training_summary":ctr["summary"]},
      "ensemble_treatment":{
        "endpoint_pass":trt["endpoint_pass"],"critic":trt["critic"],
        "continuum_monotonicity":trt["continuum_summary"]["monotonicity_fraction"],
        "continuum_endpoint_between":trt["continuum_summary"]["endpoint_between_fraction"],
        "center_compromise":trt["center_compromise"]["between_heavy_envelope_fraction"],
        "authority_training_summary":trr["summary"]},
      "primary_gate":{
        "T_source_preserved":bool(trt["endpoint_pass"]["T"]),
        "O_source_preserved":bool(trt["endpoint_pass"]["O"]),
        "global_pairwise_authority_preserved":bool(trr["summary"]["fixed_probe_pairwise_retention"]>=0.90),
        "tangent_authority_preserved":bool(trr["summary"]["fixed_probe_tangent_retention"]>=0.90),
        "critic_valid":bool(trt["critic"]["h32_ev_mean"]>0 and trt["critic"]["negative_fraction"]<=.25),
        "source_domain_gate_pass":False,
        "failure_reason":"Ensemble treatment regressed O semantics on canonical source-domain evaluation; O passed in both frozen source reference and matched control."
      },
      "evaluation_reached":{
        "source_domain_no_regression":True,
        "post_training_inensemble_semantic_robustness":False,
        "E3_heldout":False,
        "E4_mujoco":False},
      "sampler_provenance":{
        "realized_draws":657,"draw_index_start":0,"draw_index_end":656,
        "duplicate_draw_ids":0,"minimum_Linf_to_any_heldout_tuple":0.04775577783584595,
        "heldout_leakage":False},
      "decision":{
        "P5_E2":"FAIL",
        "P5_E3":"BLOCKED",
        "P5_E4":"BLOCKED",
        "D4_hardware":"BLOCKED",
        "phase5_escalation":"STOP under predeclared source no-regression rule"
      },
      "artifact_hashes":{
        "initialization_manifest":sha(init),
        "control_model_30":sha(ROOT/"runs/phase5_e2_control_seed75001/model_30.pt"),
        "treatment_model_30":sha(ROOT/"runs/phase5_e2_ensemble_seed75001/model_30.pt"),
        "control_training_report":sha(ROOT/"runs/phase5_e2_control_seed75001/training_report.json"),
        "treatment_training_report":sha(ROOT/"runs/phase5_e2_ensemble_seed75001/training_report.json"),
        "control_source_report":sha(ROOT/"runs/phase5_e2_source_control/semantic_report.json"),
        "treatment_source_report":sha(ROOT/"runs/phase5_e2_source_ensemble/semantic_report.json")
      }
    }
    out=ROOT/"artifacts/phase5_e2_screen_verdict.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    (ROOT/"artifacts/phase5_e2_screen_verdict.sha256").write_text(sha(out)+"  phase5_e2_screen_verdict.json\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
