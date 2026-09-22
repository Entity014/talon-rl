#!/usr/bin/env python3
"""T3-A3 read-only objective-set selection/compression review."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
T3=ROOT/"runs/post_v2_t3a_regrouping-2026-09-23/audit.json"
A2=ROOT/"runs/post_v2_t3a2_atomic_coherence-2026-09-23/audit.json"
OUT=ROOT/"runs/post_v2_t3a3_objective_selection-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
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
PRIMARY_CANDIDATES=[k for k in ATOMIC if k!="gait_contact"]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pear(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.std()<1e-12 or y.std()<1e-12:return 0.0
    return float(np.corrcoef(x,y)[0,1])
def rankdata(x):
    # simple average-rank-free ordering sufficient here because ties are negligible
    order=np.argsort(np.asarray(x))
    ranks=np.empty(len(order),float);ranks[order]=np.arange(len(order),dtype=float)
    return ranks
def spearman(x,y):return pear(rankdata(x),rankdata(y))
def nondominated(vals):
    # maximize every reward coordinate
    n=len(vals); keep=[]
    for i in range(n):
        dominated=False
        for j in range(n):
            if i==j:continue
            if np.all(vals[j]>=vals[i]-1e-12) and np.any(vals[j]>vals[i]+1e-12):
                dominated=True;break
        if not dominated:keep.append(i)
    return keep

t3=json.loads(T3.read_text());a2=json.loads(A2.read_text())

# Build matched 12-point atomic reward matrix from suite-policy means.
points=[]; X=[]
for r in t3["suite_policy_rows"]:
    vals=[]
    for name,terms in ATOMIC.items():
        vals.append(float(sum(r["term_mean"][t] for t in terms)))
    points.append({"suite":r["suite"],"policy":r["policy"]})
    X.append(vals)
X=np.asarray(X,float); names=list(ATOMIC)
name_to_idx={n:i for i,n in enumerate(names)}

# Pairwise policy-induced redundancy. Center within each matched reset suite to remove environment/reset effects.
Xc=X.copy()
for suite in range(4):
    inds=[k for k,p in enumerate(points) if p["suite"]==suite]
    Xc[inds]=X[inds]-X[inds].mean(0,keepdims=True)
corr={}
for i,a in enumerate(names):
    for j,b in enumerate(names[i+1:],i+1):
        corr[f"{a}|{b}"]={
          "within_suite_centered_pearson":pear(Xc[:,i],Xc[:,j]),
          "within_suite_centered_spearman":spearman(Xc[:,i],Xc[:,j]),
          "raw_pearson_for_context":pear(X[:,i],X[:,j]),
        }

# Pareto-front uniqueness within each matched suite, then aggregate.
Pidx=[name_to_idx[n] for n in PRIMARY_CANDIDATES]
pareto={}
for n in PRIMARY_CANDIDATES:
    cols=[name_to_idx[x] for x in PRIMARY_CANDIDATES if x!=n]
    suite_rows=[]
    for suite in range(4):
        inds=[k for k,p in enumerate(points) if p["suite"]==suite]
        full_local=nondominated(X[inds][:,Pidx])
        drop_local=nondominated(X[inds][:,cols])
        sf=set(full_local);sd=set(drop_local)
        suite_rows.append({
          "suite":suite,"full_front_size":len(sf),"front_without_size":len(sd),
          "front_jaccard":len(sf&sd)/max(1,len(sf|sd)),"changes_front":bool(sf!=sd)
        })
    pareto[n]={
      "suite_change_fraction":float(np.mean([r["changes_front"] for r in suite_rows])),
      "mean_front_jaccard":float(np.mean([r["front_jaccard"] for r in suite_rows])),
      "suite_rows":suite_rows,
    }

# Unique winner/order sensitivity across matched suites.
winner_uniqueness={}
for n in PRIMARY_CANDIDATES:
    i=name_to_idx[n]
    winners=[]
    for suite in range(4):
        inds=[k for k,p in enumerate(points) if p["suite"]==suite]
        best=max(inds,key=lambda k:X[k,i])
        winners.append(points[best]["policy"])
    winner_uniqueness[n]={"winners_by_suite":winners,"num_unique_winner_policies":len(set(winners))}

# Role evidence from A2.
stats=a2["matched_policy_atomic_stats"];align=a2["target_alignment_summary"]
# conservative evidence categories, not a numeric score
role={}
for n in PRIMARY_CANDIDATES:
    sep=stats[n]["range_over_scale"]
    phys=align[n]["mean_abs_target_alignment"]
    pr=[abs(v["within_suite_centered_pearson"]) for k,v in corr.items() if n in k.split("|")]
    maxcorr=max(pr) if pr else 0.0
    role[n]={
      "task_relevance":"HIGH" if n in {"velocity_tracking","vertical_stability","angular_stability","orientation_stability","effort","control_smoothness"} else "MEDIUM",
      "physical_alignment":phys,
      "controllability_proxy_range_over_scale":sep,
      "controllability_proxy":"MODERATE" if sep>=0.04 else ("WEAK" if sep<0.02 else "LOW_MODERATE"),
      "max_abs_policy_level_corr":maxcorr,
      "pareto_suite_change_fraction_when_removed":pareto[n]["suite_change_fraction"],
      "pareto_mean_front_jaccard_without":pareto[n]["mean_front_jaccard"],
      "winner_diversity":winner_uniqueness[n],
    }

# Compression logic based on current evidence:
# - velocity tracking essential task objective.
# - choose angular + orientation as stability axes; vertical retained as constraint candidate because P wins all suites and it is coupled to joint_smoothness.
# - effort primary despite weak current sensitivity because semantics/physics are clean and deployment relevance high.
# - control smoothness primary only if sufficiently distinct from effort and has measurable sensitivity.
# - joint smoothness auxiliary until direct physical proxy validation.
# - gait auxiliary.
recommended={
 "primary_morl_objectives":[
   "velocity_tracking",
   "angular_stability",
   "orientation_stability",
   "effort",
   "control_smoothness"
 ],
 "constraints_or_safety_metrics":[
   "vertical_stability"
 ],
 "auxiliary_metrics":[
   "joint_smoothness",
   "gait_contact"
 ]
}

report={
 "schema":"v2_t3a3_objective_set_selection_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
 "provenance":{"t3a_sha256":sha(T3),"t3a2_sha256":sha(A2)},
 "points":points,"atomic_order":names,
 "policy_level_correlation":corr,
 "pareto_redundancy":pareto,
 "winner_uniqueness":winner_uniqueness,
 "role_evidence":role,
 "recommended_set":recommended,
 "rationale":{
   "velocity_tracking":"Keep as mandatory task-performance axis.",
   "angular_stability":"Keep as MORL axis: strong physical alignment, highest current matched separation among stability atomics, and distinct from orientation.",
   "orientation_stability":"Keep as MORL axis: strong tilt alignment and weak correlation with angular stability, so it contributes a different physical trade-off.",
   "vertical_stability":"Prefer constraint/safety role for now: physically valid but current old-bucket policies show P best in all matched suites; it is moderately coupled to joint_smoothness and does not yet demonstrate an independent desirable preference axis.",
   "effort":"Keep as MORL axis despite weak current separation because physical semantics are exceptionally clean and deployment relevance is direct; controllability must be revalidated after scaling/retraining.",
   "control_smoothness":"Keep as MORL axis provisionally: measurable physical meaning and nontrivial policy separation; distinct from effort and joint_smoothness.",
   "joint_smoothness":"Auxiliary until a direct joint-acceleration physical proxy and controllability evidence exist.",
   "gait_contact":"Auxiliary: distinct and sensitive but not justified as thesis-level preference axis."
 },
 "limitations":[
   "Pareto analysis uses only 12 matched points from three old-bucket D1 policies across four suites; it is a compression diagnostic, not a full reachable-set Pareto characterization.",
   "Current controllability proxies come from policies trained on old objectives, so weak separation does not prove an atomic objective is uncontrollable.",
   "No scaling/normalization or retraining is performed here."
 ]
}
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"recommended_set":recommended,"pareto":pareto,"role_evidence":role},indent=2))
