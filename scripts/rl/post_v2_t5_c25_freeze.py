#!/usr/bin/env python3
import json,hashlib,numpy as np
from pathlib import Path
R=Path("runs/post_v2_t5_c25_actor_updating25-2026-09-23")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rep=json.load(open(R/"replay_audit.json"));tr=json.load(open(R/"train.json"))
safe=json.load(open(R/"u25_safety_confirm.json"))
mid=json.load(open(R/"mid_causal_semantics.json"))["snapshots"]
u25=json.load(open(R/"u25_causal_semantics.json"))["snapshots"]["25"]
def delta32(block,branch):
 r=next(x for x in block["perturbations"] if x["branch"]==branch);m=r["metric"]
 vals=[q["perturbed"]["32"][m]-q["baseline"]["32"][m] for q in r["suites"]]
 return {"mean":float(np.mean(vals)),"improve_fraction":float(np.mean(np.asarray(vals)<0)),"values":vals}
def gradgeom(update):
 cs=[];gn=[];ad=[];hd=[]
 for lab in ("T","A","O","S"):
  x=tr["specialists"][lab][update-1];M=np.asarray(x["objective_grad_cosine"])
  cs+=M[np.triu_indices(4,1)].tolist();gn+=x["objective_grad_norm"];ad.append(x["actor_param_drift"]);hd.append(x["head_solution_drift"])
 return {"offdiag_cos_mean":float(np.mean(cs)),"offdiag_cos_min":float(np.min(cs)),"offdiag_cos_max":float(np.max(cs)),
 "grad_norm_mean":float(np.mean(gn)),"actor_param_drift_mean":float(np.mean(ad)),"head_solution_drift_mean":float(np.mean(hd))}
syn={"schema":"t5_c25_synthesis_v1",
"status":"C25_CLOSED_CRITIC_REPAIR_SURVIVES_ACTOR_LEARNING_BUT_LATE_ORIENTATION_CREDIT_DRIFT",
"fresh_value":{"u1":rep["aggregate"]["1"],"u5":rep["aggregate"]["5"],"u10":rep["aggregate"]["10"],"u25":rep["aggregate"]["25"]},
"causal_semantics":{"u15":{"A":delta32(mid["15"],"A"),"O":delta32(mid["15"],"O")},"u20":{"A":delta32(mid["20"],"A"),"O":delta32(mid["20"],"O")},"u25":{"A":delta32(u25,"A"),"O":delta32(u25,"O")}},
"gradient_geometry":{"u10":gradgeom(10),"u25":gradgeom(25)},"u25_safety":safe,
"decision":{"C25":"CLOSED — critic/value gate PASS; semantic-credit persistence FAIL/PARTIAL",
"critic_repair_under_actor_shift":"PASS","fresh_H32_MC64":"PASS","orientation_value_EV":"PASS","objective_gradient_separability":"RETAINED",
"angular_causal_credit":"MOSTLY RETAINED","orientation_causal_credit":"LATE FLIP by u25","safety":"PARTIAL: Smoothness has one 0.875 survival suite out of 8","full_T4":"BLOCKED","V2":"OFF",
"next":"C26 late semantic-credit drift audit focused on u20->u25. Keep critic repair fixed and diagnose why Orientation PPO/advantage gradient ceases to reduce tilt despite positive fresh value EV: compare per-objective advantage-return alignment, critic bias/TD residual on Orientation states, state/command visitation shift, and raw-vs-Adam actor update geometry at u20/u25. No architecture/reward changes yet."},
"provenance":{"train_sha256":sha(R/"train.json"),"replay_sha256":sha(R/"replay_audit.json"),"mid_causal_sha256":sha(R/"mid_causal_semantics.json"),"u25_causal_sha256":sha(R/"u25_causal_semantics.json"),"u25_safety_sha256":sha(R/"u25_safety_confirm.json"),
"terminal_snapshots":{lab:sha(R/f"{lab}_snap_25.pt") for lab in ("T","A","O","S")}}}
(R/"synthesis.json").write_text(json.dumps(syn,indent=2)+"\n")
manifest={"status":"FROZEN_BY_HASH","artifacts":{f:{"sha256":sha(R/f)} for f in ("train.json","replay_audit.json","mid_causal_semantics.json","u25_causal_semantics.json","u25_safety_confirm.json","synthesis.json")},
"terminal_snapshots":{lab:{"sha256":sha(R/f"{lab}_snap_25.pt")} for lab in ("T","A","O","S")}}
(R/"PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"status":syn["status"],"u25":syn["fresh_value"]["u25"],"causal":syn["causal_semantics"],"next":syn["decision"]["next"]},indent=2))
