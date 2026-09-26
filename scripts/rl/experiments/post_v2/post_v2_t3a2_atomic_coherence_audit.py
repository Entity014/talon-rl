#!/usr/bin/env python3
"""T3-A2 read-only atomic semantic-coordinate/coherence audit."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
T3=ROOT/"runs/post_v2_t3a_regrouping-2026-09-23/audit.json"
S6=[ROOT/f"runs/v1b_s6_objective_decomposition_seed{s}-2026-09-22/audit.json" for s in (0,1,2)]
OUT=ROOT/"runs/post_v2_t3a2_atomic_coherence-2026-09-23"; OUT.mkdir(parents=True,exist_ok=True)
LABS=("P","B","E")
ATOMIC={
 "velocity_tracking":["track_lin_vel_xy_exp","track_ang_vel_z_exp"],
 "vertical_stability":["lin_vel_z_l2"],
 "angular_stability":["ang_vel_xy_l2"],
 "orientation_stability":["flat_orientation_l2"],
 "effort":["dof_torques_l2"],
 "joint_smoothness":["dof_acc_l2"],
 "control_smoothness":["action_rate_l2"],
 "gait_contact":["feet_air_time"],
}
PHYS_TARGET={
 "velocity_tracking":["vx_error","wz_error"],
 "vertical_stability":["abs_lin_vel_z"],
 "angular_stability":["abs_ang_vel_xy"],
 "orientation_stability":["tilt_deg"],
 "effort":["torque_l2"],
 "joint_smoothness":[],
 "control_smoothness":["action_rate_l2"],
 "gait_contact":[],
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.std()==0 or y.std()==0:return 0.0
    return float(np.corrcoef(x,y)[0,1])

t3=json.loads(T3.read_text())
s6=[json.loads(p.read_text()) for p in S6]

# matched-policy atomic statistics from T3-A term means/std/nonzero
stats={}
for name,terms in ATOMIC.items():
    policy_means={lab:[] for lab in LABS}
    stds=[]; nonzeros=[]
    for r in t3["suite_policy_rows"]:
        vals=[r["term_mean"][t] for t in terms]
        policy_means[r["policy"]].append(float(sum(vals)))
        if len(terms)==1:
            stds.append(r["term_std"][terms[0]]); nonzeros.append(r["term_nonzero"][terms[0]])
        else:
            # Exact std/nonzero for velocity_tracking is available from T3-A group stats.
            stds.append(r["group_std"]["progress"]); nonzeros.append(r["group_nonzero"]["progress"])
    avg={lab:float(np.mean(v)) for lab,v in policy_means.items()}
    scale=float(np.mean(np.abs([x for v in policy_means.values() for x in v])))
    prange=float(max(avg.values())-min(avg.values()))
    winners=[]
    for suite in range(4):
        vals={lab:float(sum(next(r for r in t3["suite_policy_rows"] if r["suite"]==suite and r["policy"]==lab)["term_mean"][t] for t in terms)) for lab in LABS}
        winners.append(max(vals,key=vals.get))
    stats[name]={
      "terms":terms,"policy_mean":avg,"range_over_scale":prange/(scale+1e-12),
      "abs_mean_scale":scale,"mean_std":float(np.mean(stds)),"mean_nonzero_fraction":float(np.mean(nonzeros)),
      "winner_fraction":{lab:float(np.mean([w==lab for w in winners])) for lab in LABS}
    }

# multi-seed atomic physical alignment and evidence graph.
# velocity tracking uses the grouped progress alignment already computed by T3-A matched audit;
# singleton atomics use S6 per-term correlations across seeds.
phys={}
for name,terms in ATOMIC.items():
    phys[name]={}
    if name=="velocity_tracking":
        phys[name]={"vx_error":t3["mean_group_physical_corr"]["progress"]["vx_error"],
                    "wz_error":t3["mean_group_physical_corr"]["progress"]["wz_error"]}
    elif len(terms)==1:
        t=terms[0]
        for metric in s6[0]["physical_metric_order"]:
            vals=[]
            for d in s6:
                ti=d["reward_term_order"].index(t); pi=d["physical_metric_order"].index(metric)
                vals.append(d["term_physical_pearson"][ti][pi])
            phys[name][metric]={"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals))}
# semantic target alignment summary
alignment={}
for name,targets in PHYS_TARGET.items():
    if name=="velocity_tracking":
        vals=[abs(phys[name][m]) for m in targets]
    else:
        vals=[abs(phys[name][m]["mean"]) for m in targets] if targets else []
    alignment[name]={"target_metrics":targets,"mean_abs_target_alignment":float(np.mean(vals)) if vals else None}

# Pairwise atomic correlations from S6 multi-seed.
# velocity_tracking correlations cannot be reconstructed exactly from term-only correlation matrices,
# so use T3-A progress group correlation where the counterpart is an existing T3-A group only;
# all singleton-singleton pairs are exact S6 term correlations.
names=list(ATOMIC)
graph=[]
for i,a in enumerate(names):
  for b in names[i+1:]:
    vals=[]; source=None
    if len(ATOMIC[a])==1 and len(ATOMIC[b])==1:
      ta,tb=ATOMIC[a][0],ATOMIC[b][0]
      for d in s6:
        ia=d["reward_term_order"].index(ta); ib=d["reward_term_order"].index(tb)
        vals.append(d["term_term_pearson"][ia][ib])
      source="S6 multi-seed term correlation"
    elif a=="velocity_tracking" or b=="velocity_tracking":
      other=b if a=="velocity_tracking" else a
      groupmap={"vertical_stability":"stability","angular_stability":"stability","orientation_stability":"stability",
                "effort":"effort","joint_smoothness":"smoothness","control_smoothness":"smoothness","gait_contact":"gait_contact_aux"}
      # only coarse parent-group relation is available for velocity tracking.
      gi=t3["core_group_order"].index("progress"); gj=t3["core_group_order"].index(groupmap[other])
      vals=[t3["mean_group_pairwise_corr"][gi][gj]]
      source="T3-A coarse progress-to-parent-group correlation"
    graph.append({"a":a,"b":b,"corr_mean":float(np.mean(vals)),"abs_corr_mean":float(np.mean(np.abs(vals))),
                  "corr_min":float(np.min(vals)),"corr_max":float(np.max(vals)),"source":source})

# recombination candidates require both semantic affinity and correlation.
# Use a conservative evidence label rather than hard threshold as a design decision.
for e in graph:
    a,b=e["a"],e["b"]
    same_domain=((a in {"vertical_stability","angular_stability","orientation_stability"} and b in {"vertical_stability","angular_stability","orientation_stability"})
                 or (a in {"joint_smoothness","control_smoothness"} and b in {"joint_smoothness","control_smoothness"}))
    if same_domain and e["abs_corr_mean"]>=0.5:
        e["recombine_evidence"]="STRONG"
    elif same_domain and e["abs_corr_mean"]>=0.3:
        e["recombine_evidence"]="MODERATE"
    elif same_domain:
        e["recombine_evidence"]="WEAK"
    else:
        e["recombine_evidence"]="NOT_PROPOSED"

report={
 "schema":"v2_t3a2_atomic_coherence_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
 "atomic_coordinates":ATOMIC,
 "provenance":{"t3a":str(T3.relative_to(ROOT)),"t3a_sha256":sha(T3),
               "s6":[{"path":str(p.relative_to(ROOT)),"sha256":sha(p)} for p in S6]},
 "matched_policy_atomic_stats":stats,
 "physical_alignment":phys,
 "target_alignment_summary":alignment,
 "atomic_evidence_graph":graph,
 "limitations":[
  "Velocity-tracking cross-correlations to singleton atomics are only available through T3-A parent-group correlations; no raw matched sample matrix was persisted for exact atomic progress correlations.",
  "D1 policies were trained on the old objective buckets, so current matched policy separation is sensitivity evidence, not proof of optimizability of the new atomic coordinate.",
  "No scaling or normalization is applied in T3-A2."
 ]
}
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"stats":stats,"alignment":alignment,
                  "same_domain_edges":[e for e in graph if e["recombine_evidence"]!="NOT_PROPOSED"]},indent=2))
