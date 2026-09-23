#!/usr/bin/env python3
import json,hashlib,numpy as np
from pathlib import Path
R=Path("runs/post_v2_t5_c25_actor_updating-2026-09-23")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tr=json.load(open(R/"train.json"));ev=json.load(open(R/"eval.json"));u10=json.load(open(R/"u10_perturb_confirm.json"));ang10=json.load(open(R/"angular_credit_confirm.json"));ang0=json.load(open(R/"angular_u0_control.json"))
sep={}
for u in (1,5,10):
 mats=[]
 for lab in ("T","A","O","S"): mats.append(np.array(tr["specialists"][lab][u-1]["objective_grad_cosine"],float))
 M=np.mean(mats,axis=0);off=M[~np.eye(4,dtype=bool)]
 sep[str(u)]={"offdiag_mean":float(off.mean()),"offdiag_max":float(off.max()),"offdiag_min":float(off.min())}
syn={"schema":"t5_c25_actor_updating_reset_support_synthesis_v1",
"status":"C25_CLOSED_VALUE_GENERALIZATION_PASS_SEMANTIC_CREDIT_FAIL_ANGULAR",
"evidence":{"value":ev["aggregate"],"gradient_separability":sep,"u10_perturb_confirm":u10,
"angular_u0":{"gradient_batch_cosine_mean":ang0["gradient_batch_cosine_mean"],"summary":ang0["summary"]},
"angular_u10":{"gradient_batch_cosine_mean":ang10["gradient_batch_cosine_mean"],"summary":ang10["summary"]}},
"decision":{"critic_reset_diverse_repair_under_actor_learning":"PASS","fresh_value_generalization":"PASS","orientation_credit":"MOSTLY PRESERVED","angular_credit":"FAIL / unstable","overall_C25":"PARTIAL FAIL","full_T4":"BLOCKED","V2":"OFF","next":"C26 Angular-credit stability audit under the repaired critic. Keep C23/C25 critic support fixed and diagnose why Angular advantage/gradient direction loses causal reliability after actor updates: compare per-batch advantage SNR, gradient cosine, objective-return sensitivity and short-vs-long-horizon credit at u0/u5/u10 before changing PPO or reward."},
"provenance":{"train_sha256":sha(R/"train.json"),"eval_sha256":sha(R/"eval.json"),"u10_perturb_sha256":sha(R/"u10_perturb_confirm.json"),"angular_u10_sha256":sha(R/"angular_credit_confirm.json"),"angular_u0_sha256":sha(R/"angular_u0_control.json")}}
(R/"synthesis.json").write_text(json.dumps(syn,indent=2)+"\n")
manifest={"status":"FROZEN_BY_HASH","artifacts":{f:{"sha256":sha(R/f)} for f in ("train.json","eval.json","u10_perturb_confirm.json","angular_credit_confirm.json","angular_u0_control.json","synthesis.json")},"sources":{s:{"sha256":sha(s)} for s in ("scripts/rl/post_v2_t5_c25_actor_updating_reset_support.py","scripts/rl/post_v2_t5_c25_eval.py","scripts/rl/post_v2_t5_c25_u10_perturb_confirm.py","scripts/rl/post_v2_t5_c25_angular_credit_confirm.py","scripts/rl/post_v2_t5_c25_angular_u0_control.py")}}
(R/"PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"status":syn["status"],"u10_value":ev["aggregate"]["10"],"sep":sep,"angular_u0":syn["evidence"]["angular_u0"],"angular_u10":syn["evidence"]["angular_u10"],"next":syn["decision"]["next"]},indent=2))
