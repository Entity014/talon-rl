#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
D1=ROOT/"runs/post_v1_d1-2026-09-22/d1.json"
AGG=ROOT/"runs/post_v1_d1-2026-09-22/aggregate.json"
T0=ROOT/"runs/post_v2_t0_trajectory_credit-2026-09-23/audit.json"
OUT=ROOT/"runs/post_v2_t1_specialist_semantics-2026-09-23"
OUT.mkdir(parents=True,exist_ok=True)
PREFS={"P":np.array([.8,.1,.1]),"B":np.array([.1,.8,.1]),"E":np.array([.1,.1,.8])}
LABS=["P","B","E"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

d1=json.loads(D1.read_text())
agg=json.loads(AGG.read_text())
t0=json.loads(T0.read_text())

# terminal objective reward-rate vectors from original D1 aggregate
terminal=np.stack([np.array(agg["summary"][lab]["objective_return_mean"],float) for lab in LABS],0)
cross=np.zeros((3,3))
contrib={}
for i,plab in enumerate(LABS):
    contrib[plab]={}
    for j,wlab in enumerate(LABS):
        w=PREFS[wlab]
        weighted=terminal[i]*w
        cross[i,j]=weighted.sum()
        denom=np.sum(np.abs(weighted))+1e-12
        contrib[plab][wlab]={
            "weighted_terms":weighted.tolist(),
            "scalarized_reward_rate":float(weighted.sum()),
            "absolute_contribution_share":(np.abs(weighted)/denom).tolist(),
        }

# training-time grouped reward vectors (already multiplied by step_dt in training)
training={}
for lab in LABS:
    rec=d1["records"][lab]
    arr=np.array([r["reward_mean"] for r in rec],float)
    w=PREFS[lab]
    windows={}
    for name,sl in {
        "early_1_25":slice(0,25),"mid_126_175":slice(125,175),"late_276_300":slice(275,300)
    }.items():
        x=arr[sl].mean(0); wc=x*w
        windows[name]={
          "objective_stepdt_mean":x.tolist(),
          "weighted_contributions":wc.tolist(),
          "scalarized_stepdt_mean":float(wc.sum()),
          "absolute_contribution_share":(np.abs(wc)/(np.abs(wc).sum()+1e-12)).tolist()
        }
    training[lab]=windows

# Physical semantic rankings from D1 aggregate
metrics={lab:agg["summary"][lab]["metrics_mean"] for lab in LABS}
physical={
 "progress_vx_error_best":min(LABS,key=lambda x:metrics[x]["vx_error"]),
 "balance_tilt_best":min(LABS,key=lambda x:metrics[x]["tilt_deg"]),
 "balance_ang_vel_best":min(LABS,key=lambda x:metrics[x]["ang_vel_xy"]),
 "efficiency_torque_best":min(LABS,key=lambda x:metrics[x]["torque_norm"]),
 "efficiency_action_rate_best":min(LABS,key=lambda x:metrics[x]["action_rate"]),
}

# Cross-eval winners under each evaluation preference.
cross_winners={LABS[j]:LABS[int(np.argmax(cross[:,j]))] for j in range(3)}
cross_margin={}
for j,wlab in enumerate(LABS):
    order=np.argsort(-cross[:,j]); cross_margin[wlab]={
      "winner":LABS[int(order[0])],"runner_up":LABS[int(order[1])],
      "margin":float(cross[order[0],j]-cross[order[1],j])
    }

# Reward-scale diagnostics across policies at terminal
obj_ranges=terminal.max(0)-terminal.min(0)
obj_abs_mean=np.abs(terminal).mean(0)
scale={
 "terminal_objective_abs_mean":obj_abs_mean.tolist(),
 "terminal_objective_policy_range":obj_ranges.tolist(),
 "range_over_abs_mean":(obj_ranges/(obj_abs_mean+1e-12)).tolist()
}

report={
 "schema":"v2_t1_specialist_reward_semantics_audit_v1",
 "status":"MEASUREMENT_COMPLETE","measurement_only":True,
 "provenance":{"d1_sha256":sha(D1),"aggregate_sha256":sha(AGG),"t0_sha256":sha(T0)},
 "objective_order":["progress","balance","efficiency"],
 "fixed_preferences":{k:v.tolist() for k,v in PREFS.items()},
 "metric_semantics_note":"D1 aggregate objective_return_mean is a terminal mean weighted reward-rate per env-step, not an episodic return; training reward_mean is the same grouped reward multiplied by step_dt before GAE.",
 "terminal_objective_reward_rate_matrix":{"rows_policy":LABS,"columns_objective":["P","B","E"],"values":terminal.tolist()},
 "cross_evaluation_scalarized_matrix":{"rows_policy":LABS,"columns_eval_preference":LABS,"values":cross.tolist(),"winners":cross_winners,"margins":cross_margin},
 "terminal_scalarized_contributions":contrib,
 "training_time":training,
 "physical_metrics":metrics,
 "physical_best":physical,
 "reward_scale":scale,
 "t0_summary":t0["summary"]["D1"],
}
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({
 "cross_matrix":cross.tolist(),
 "winners":cross_winners,
 "physical_best":physical,
 "reward_scale":scale,
 "late_training":{k:v["late_276_300"] for k,v in training.items()}
},indent=2))
