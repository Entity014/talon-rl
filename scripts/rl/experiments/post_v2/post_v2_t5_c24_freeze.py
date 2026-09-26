#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
R=Path("runs/post_v2_t5_c24_tracking_residual-2026-09-23")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.load(open(R/"audit.json")); r=json.load(open(R/"reset_breakdown.json"))
t=a["aggregate"]["target_stats"]["Tracking"]
f=a["aggregate"]["fit_summary"]["Tracking"]
syn={"schema":"t5_c24_tracking_residual_synthesis_v1",
"status":"C24_CLOSED_TRACKING_RESIDUAL_NONSTRUCTURAL_SHORT_HORIZON_TARGET_GEOMETRY",
"evidence":{"tracking_h32_mc64_corr":t["h32_mc64_corr"],"tracking_h32_var":t["h32_var"],"tracking_mc64_var":t["mc64_var"],"tracking_h32_ridge1_ev":f["h32_ridge_sweep"]["1.0"]["h32_ev_mean"],"tracking_h32_best_sweep_ev":max(v["h32_ev_mean"] for v in f["h32_ridge_sweep"].values()),"tracking_mc64_ridge1_ev":f["mc64_ridge1_ev_mean"],"reset_breakdown":r["aggregate"]},
"decision":{"C24":"CLOSED","shared_body_tracking_capacity":"SUFFICIENT but H32 predictability weak","ridge_tuning":"NOT JUSTIFIED","tracking_h32_residual":"NONSTRUCTURAL / horizon-target-sensitive","critic_repair_principle":"RETAIN C23 reset-diverse support","actor_updates":"AUTHORIZED for next pilot only","full_T4":"STILL BLOCKED","V2":"OFF","next":"C25 actor-updating reset-diverse critic-support pilot. Keep C23 representative reset/command/state support, shared frozen critic body, ridge lambda=1, H32 target and PPO semantics fixed; re-enable actor updates only. Primary gate: A/O semantic gradients and fresh value generalization must remain stable under policy shift."}}
(R/"synthesis.json").write_text(json.dumps(syn,indent=2)+"\n")
manifest={"status":"FROZEN_BY_HASH","artifacts":{f:{"sha256":sha(R/f)} for f in ("audit.json","reset_breakdown.json","synthesis.json")},"sources":{"main":{"sha256":sha("scripts/rl/post_v2_t5_c24_tracking_residual_audit.py")},"reset":{"sha256":sha("scripts/rl/post_v2_t5_c24_reset_breakdown.py")}}}
(R/"PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"status":syn["status"],"evidence":syn["evidence"],"next":syn["decision"]["next"]},indent=2))
