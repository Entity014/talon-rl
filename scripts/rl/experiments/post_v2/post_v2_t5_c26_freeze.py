#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
C25=Path("runs/post_v2_t5_c25_actor_updating25-2026-09-23")
C26=Path("runs/post_v2_t5_c26_angular_credit-2026-09-23")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rep=json.load(open(C25/"replay_audit25.json"))
safety=json.load(open(C25/"u25_safety_confirm.json"))
sem=json.load(open(C25/"semantic_u25.json"))
aud=json.load(open(C26/"audit.json"))
con=json.load(open(C26/"consensus.json"))
syn={"schema":"c26_angular_credit_stability_synthesis_v1",
"status":"C26_CLOSED_ANGULAR_TRANSIENT_LOW_CONSENSUS_NOT_CRITIC_FAILURE",
"evidence":{"value_u25":rep["aggregate"]["25"],"safety_u25":safety,"semantic_u25":sem,"angular_stability":{k:v["summary"] for k,v in aud["snapshots"].items()},"consensus":con["snapshots"]},
"decision":{"C25_durability":"critic repair PASS through u25","angular_credit":"TRANSIENT instability; recovered by u25","critic_failure":"REJECTED","smoothness_safety":"REMAINING blocker","full_T4":"BLOCKED","V2":"OFF","next":"C27 Smoothness safety residual audit at u10/u25: isolate failed reset suite physical trajectory and determine whether safety loss is branch-specific actor behavior or evaluation noise; no critic changes."}}
(C26/"synthesis.json").write_text(json.dumps(syn,indent=2)+"\n")
manifest={"status":"FROZEN_BY_HASH","artifacts":{f:{"sha256":sha(C26/f)} for f in ("audit.json","consensus.json","synthesis.json")},
"c25_refs":{"replay25":{"sha256":sha(C25/"replay_audit25.json")},"safety25":{"sha256":sha(C25/"u25_safety_confirm.json")},"semantic25":{"sha256":sha(C25/"semantic_u25.json")}}}
(C26/"PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"status":syn["status"],"next":syn["decision"]["next"]},indent=2))
