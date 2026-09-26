#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
R=Path("runs/post_v2_t5_c23_reset_diverse-2026-09-23")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.load(open(R/"audit.json"))
syn={"schema":"t5_c23_reset_diverse_synthesis_v1",
"status":"C23_CLOSED_RESET_DIVERSE_SUPPORT_PRINCIPLE_PASS_ALL_HEAD_ROBUSTNESS_PARTIAL",
"aggregate":a["aggregate"],
"findings":{
"primary":"At identical support count, ridge lambda, frozen actor/body and H32 target, independent reset/seeded support changes fresh H32 EV from -1.211 to +0.196 and Orientation from -1.142 to +0.236.",
"mc64":"Fresh MC64 EV changes from -0.249 to +0.464; reset-diverse has 0% negative MC64 head evaluations and survival 1.0.",
"round_consistency":"Reset-diverse H32 mean remains positive in all four rounds (+0.233,+0.159,+0.141,+0.252); Orientation is also positive in all four rounds.",
"mechanism":"Head-solution drift is not materially lower (2.239 vs 2.184 excluding first round). The repair therefore comes primarily from representative support/generalization, not merely a more stationary optimum.",
"residual":"Tracking remains the weak head: mean H32 EV about -0.092 with high negative fraction, while Angular, Orientation and Smoothness are positive. Thus the support principle passes but all-head robustness is not yet complete."},
"decision":{"C23":"CLOSED — reset-diverse support principle PASS; all-head robustness PARTIAL",
"representative_reset_command_state_support":"SUPPORTED",
"separate_critics":"NOT JUSTIFIED","body_anchor":"NOT JUSTIFIED",
"actor_updates":"STILL OFF for one more gate","full_T4":"BLOCKED","V2":"OFF",
"next":"C24 Tracking-head residual audit under the reset-diverse repair. Keep the C23 support mechanism fixed and diagnose why Tracking H32 EV remains slightly negative while A/O/S generalize. First determine whether this is target variance/horizon mismatch or a head-specific feature-fit issue before re-enabling actor updates."},
"provenance":{"audit_sha256":sha(R/"audit.json"),"pilot_script_sha256":sha("scripts/rl/post_v2_t5_c23_reset_diverse_pilot.py")}}
(R/"synthesis.json").write_text(json.dumps(syn,indent=2)+"\n")
manifest={"status":"FROZEN_BY_HASH","artifacts":{f:{"sha256":sha(R/f)} for f in ("audit.json","synthesis.json")},"sources":{"pilot":{"sha256":sha("scripts/rl/post_v2_t5_c23_reset_diverse_pilot.py")}}}
(R/"PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"status":syn["status"],"aggregate":a["aggregate"],"next":syn["decision"]["next"]},indent=2))
