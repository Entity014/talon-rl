#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
R=Path("runs/post_v2_t5_c27_smoothness_safety-2026-09-23")
C25=Path("runs/post_v2_t5_c25_actor_updating25-2026-09-23")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.load(open(R/"audit.json")); c=json.load(open(R/"counterfactual.json"))
s=json.load(open(C25/"u25_safety_confirm.json"))
syn={"schema":"c27_smoothness_safety_synthesis_v1",
"status":"C27_CLOSED_SMOOTHNESS_BRANCH_TRAJECTORY_SAFETY_RISK_LOCAL_GRADIENT_NOT_SUFFICIENT",
"evidence":{"u25_safety":s,"failed_seed_trace":a["snapshots"]["25"],"counterfactual":c},
"decision":{"C27":"CLOSED","smoothness_safety_failure":"REAL / branch-specific / trajectory-level","critic_failure":"REJECTED","local_smoothness_gradient_as_root_cause":"NOT SUPPORTED","full_T4":"BLOCKED","V2":"OFF","next":"C28 Smoothness basin/safety audit: characterize S-heavy policy state visitation and recovery margin on the failing reset versus survivors, then decide whether smoothness should remain a free MORL axis or require a safety constraint/regularizer. No critic changes."}}
(R/"synthesis.json").write_text(json.dumps(syn,indent=2)+"\n")
manifest={"status":"FROZEN_BY_HASH","artifacts":{f:{"sha256":sha(R/f)} for f in ("audit.json","counterfactual.json","synthesis.json")},"c25_refs":{"u25_safety":{"sha256":sha(C25/"u25_safety_confirm.json")}}}
(R/"PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"status":syn["status"],"next":syn["decision"]["next"]},indent=2))
