#!/usr/bin/env python3
"""T3-A4 read-only controllability / reachable-set audit."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
T3=ROOT/"runs/post_v2_t3a_regrouping-2026-09-23/audit.json"
A3=ROOT/"runs/post_v2_t3a3_objective_selection-2026-09-23/audit.json"
OUT=ROOT/"runs/post_v2_t3a4_controllability-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
CANDS=["angular_stability","orientation_stability","effort","control_smoothness"]
ATOMIC={
 "velocity_tracking":["track_lin_vel_xy_exp","track_ang_vel_z_exp"],
 "angular_stability":["ang_vel_xy_l2"],
 "orientation_stability":["flat_orientation_l2"],
 "effort":["dof_torques_l2"],
 "control_smoothness":["action_rate_l2"],
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit_residual(y,x):
    x=np.asarray(x,float);y=np.asarray(y,float)
    A=np.stack([np.ones_like(x),x],1)
    beta=np.linalg.lstsq(A,y,rcond=None)[0]
    pred=A@beta
    return y-pred,beta
def slope(a,b):
    da=b[0]-a[0]; db=b[1]-a[1]
    return None if abs(da)<1e-12 else float(db/da)

t3=json.loads(T3.read_text());a3=json.loads(A3.read_text())

# 12 matched points, all reward coordinates "higher is better".
rows=[]
for r in t3["suite_policy_rows"]:
    x={n:float(sum(r["term_mean"][t] for t in terms)) for n,terms in ATOMIC.items()}
    rows.append({"suite":r["suite"],"policy":r["policy"],**x})

# Normalize each coordinate by global abs-mean scale for dimensionless slopes/widths.
scale={n:float(np.mean(np.abs([r[n] for r in rows])))+1e-12 for n in ATOMIC}
Z=[]
for r in rows:
    Z.append({"suite":r["suite"],"policy":r["policy"],**{n:r[n]/scale[n] for n in ATOMIC}})

result={}
track=np.array([r["velocity_tracking"] for r in Z],float)
for cand in CANDS:
    y=np.array([r[cand] for r in Z],float)

    # Conditional sensitivity: remove linear dependence on tracking globally and within-suite centered.
    resid,beta=fit_residual(y,track)
    raw_std=float(np.std(y)); resid_std=float(np.std(resid))
    residual_fraction=resid_std/(raw_std+1e-12)

    # Within-suite centered residual variation.
    yc=[];tc=[]
    for s in range(4):
        rs=[r for r in Z if r["suite"]==s]
        yy=np.array([r[cand] for r in rs]); tt=np.array([r["velocity_tracking"] for r in rs])
        yc.extend((yy-yy.mean()).tolist());tc.extend((tt-tt.mean()).tolist())
    yc=np.array(yc);tc=np.array(tc)
    cresid,cbeta=fit_residual(yc,tc)
    centered_residual_fraction=float(np.std(cresid)/(np.std(yc)+1e-12))

    # Pairwise finite differences inside each matched suite.
    pair_rows=[]
    labs=("P","B","E")
    for s in range(4):
        rs={r["policy"]:r for r in Z if r["suite"]==s}
        for i,a in enumerate(labs):
            for b in labs[i+1:]:
                dt=rs[b]["velocity_tracking"]-rs[a]["velocity_tracking"]
                dc=rs[b][cand]-rs[a][cand]
                pair_rows.append({"suite":s,"a":a,"b":b,"delta_tracking":float(dt),"delta_candidate":float(dc),
                                  "abs_tradeoff_ratio":float(abs(dc)/(abs(dt)+1e-12)),
                                  "candidate_improves":bool(dc>0),"tracking_improves":bool(dt>0)})

    # Reachable width under tracking tolerances relative to best tracking policy within suite.
    tol_results={}
    for tol in (0.01,0.025,0.05,0.10):
        widths=[];counts=[];improve_exists=[]
        for s in range(4):
            rs=[r for r in Z if r["suite"]==s]
            best_track=max(r["velocity_tracking"] for r in rs)
            eligible=[r for r in rs if r["velocity_tracking"]>=best_track-tol]
            vals=[r[cand] for r in eligible]
            widths.append(float(max(vals)-min(vals)) if len(vals)>=2 else 0.0)
            counts.append(len(vals))
            bestcand=max(r[cand] for r in rs)
            improve_exists.append(bool(any(r[cand]>=bestcand-1e-12 for r in eligible)))
        tol_results[str(tol)]={
          "mean_width":float(np.mean(widths)),
          "max_width":float(np.max(widths)),
          "mean_eligible_count":float(np.mean(counts)),
          "fraction_suites_best_candidate_reachable_within_tracking_tol":float(np.mean(improve_exists)),
        }

    # Local Pareto availability: candidate can improve versus another policy while tracking loss <= tol.
    local={}
    for tol in (0.01,0.025,0.05,0.10):
        successes=0;total=0
        for pr in pair_rows:
            # either direction; candidate-improving move with tracking loss no worse than tol
            for dt,dc in ((pr["delta_tracking"],pr["delta_candidate"]),(-pr["delta_tracking"],-pr["delta_candidate"])):
                if dc>0:
                    total+=1
                    if dt>=-tol:successes+=1
        local[str(tol)]={"fraction_candidate_improvements_with_tracking_loss_within_tol":float(successes/max(1,total)),
                         "num_improvement_moves":total}

    result[cand]={
      "conditional_sensitivity":{
        "global_tracking_beta":beta.tolist(),
        "residual_std_fraction_of_raw":float(residual_fraction),
        "within_suite_tracking_beta":cbeta.tolist(),
        "within_suite_residual_std_fraction_of_raw":centered_residual_fraction,
      },
      "finite_difference_pairs":pair_rows,
      "tracking_tolerance_width":tol_results,
      "local_tradeoff_availability":local,
      "current_policy_range_over_scale":a3["role_evidence"][cand]["controllability_proxy_range_over_scale"],
      "policy_level_max_abs_corr":a3["role_evidence"][cand]["max_abs_policy_level_corr"],
    }

# Conservative role transition rules.
roles={}
for cand,x in result.items():
    rf=x["conditional_sensitivity"]["within_suite_residual_std_fraction_of_raw"]
    sep=x["current_policy_range_over_scale"]
    w5=x["tracking_tolerance_width"]["0.05"]["mean_width"]
    avail=x["local_tradeoff_availability"]["0.05"]["fraction_candidate_improvements_with_tracking_loss_within_tol"]
    if rf>=0.5 and sep>=0.03 and (w5>=0.01 or avail>=0.5):
        role="KEEP_AS_AXIS"
    elif rf<0.25 and x["policy_level_max_abs_corr"]>=0.9:
        role="INSUFFICIENT_EVIDENCE"
    elif cand=="effort" and sep<0.02:
        role="INSUFFICIENT_EVIDENCE"
    elif cand=="control_smoothness" and rf<0.4:
        role="MOVE_TO_REGULARIZER"
    else:
        role="INSUFFICIENT_EVIDENCE"
    roles[cand]=role

report={
 "schema":"v2_t3a4_controllability_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
 "provenance":{"t3a_sha256":sha(T3),"t3a3_sha256":sha(A3)},
 "normalization":"Each reward coordinate divided by its global absolute mean scale; no reward/training changes.",
 "tracking_tolerances":[0.01,0.025,0.05,0.10],
 "candidate_results":result,
 "role_transition":roles,
 "limitations":[
   "Reachable set contains only three old-bucket D1 policies per suite, so this audit can establish local evidence but cannot prove global controllability.",
   "Linear residualization tests conditional variation, not causal intervention.",
   "No retraining, reward scaling, architecture changes, or policy interpolation are performed."
 ]
}
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"roles":roles,"summary":{k:{
 "resid_frac":v["conditional_sensitivity"]["within_suite_residual_std_fraction_of_raw"],
 "sep":v["current_policy_range_over_scale"],
 "width05":v["tracking_tolerance_width"]["0.05"]["mean_width"],
 "avail05":v["local_tradeoff_availability"]["0.05"]["fraction_candidate_improvements_with_tracking_loss_within_tol"]
} for k,v in result.items()}},indent=2))
