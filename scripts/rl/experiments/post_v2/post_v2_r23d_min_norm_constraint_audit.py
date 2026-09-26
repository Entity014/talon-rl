#!/usr/bin/env python3
"""V2-R2.3-D read-only minimum-norm pairwise semantic orientation correction audit."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[4]
EPS=1e-12
MARGIN=1e-6
PAIRS=(("P","B","PB"),("P","E","PE"),("B","E","BE"))
TARGETS={"PB":np.array([1.,-1.]),"PE":np.array([1.,1.]),"BE":np.array([0.,1.])}
IDX={"P":0,"B":2,"E":4}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cosine(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+EPS))
def pair_vec(flat,a,b):
    ia,ib=IDX[a],IDX[b]
    return flat[ib:ib+2]-flat[ia:ia+2]

def solve_min_correction(u0):
    # x is correction to flattened [P(2), B(2), E(2)]
    cons=[]
    for a,b,name in PAIRS:
        t=TARGETS[name]
        for j in range(2):
            if t[j]==0: continue
            ia,ib=IDX[a]+j,IDX[b]+j
            s=float(t[j])
            # require s * ((u_b + x_b) - (u_a + x_a)) >= MARGIN
            def fun(x, ia=ia, ib=ib, s=s):
                return s*((u0[ib]+x[ib])-(u0[ia]+x[ia]))-MARGIN
            cons.append({"type":"ineq","fun":fun})
    res=minimize(lambda x:0.5*float(x@x),np.zeros_like(u0),jac=lambda x:x,
                 constraints=cons,method="SLSQP",
                 options={"ftol":1e-12,"maxiter":500,"disp":False})
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--r23a-audit",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    src=json.loads(args.r23a_audit.read_text())
    rows=[]
    for snap in src["snapshots"]:
        update=snap["update"]
        # suite-level solves
        for suite in range(4):
            pref={}
            for p in ("P","B","E"):
                pref[p]=np.asarray(snap["preferences"][p][suite]["ppo_update_dir_out"],float)
            u0=np.concatenate([pref["P"],pref["B"],pref["E"]])
            res=solve_min_correction(u0)
            corr=res.x if res.success else np.full_like(u0,np.nan)
            uc=u0+corr if res.success else np.full_like(u0,np.nan)
            pair_info={}
            for a,b,name in PAIRS:
                pv=pair_vec(uc,a,b);t=TARGETS[name]
                margins=[]
                ok=True
                for j in range(2):
                    if t[j]==0: continue
                    m=float(t[j]*pv[j]);margins.append(m);ok &= m>=MARGIN-1e-9
                pair_info[name]={
                  "corrected_vector":pv.tolist(),
                  "target_sign":[int(x) for x in t],
                  "min_signed_margin":min(margins) if margins else None,
                  "constraint_ok":bool(ok),
                }
            ppo_norm=float(np.linalg.norm(u0));corr_norm=float(np.linalg.norm(corr)) if res.success else None
            shared_corr=np.linalg.norm(corr[[0,2,4]]) if res.success else None
            be_corr=np.linalg.norm(corr[[1,3,5]]) if res.success else None
            rows.append({
              "update":update,"suite":suite,"feasible":bool(res.success),
              "solver_status":res.message,
              "ppo_pressure_flat":u0.tolist(),
              "correction_flat":corr.tolist() if res.success else None,
              "corrected_pressure_flat":uc.tolist() if res.success else None,
              "ppo_norm":ppo_norm,"correction_norm":corr_norm,
              "correction_ratio":None if not res.success else corr_norm/(ppo_norm+EPS),
              "corrected_vs_ppo_cosine":None if not res.success else cosine(uc,u0),
              "shared_correction_norm":shared_corr,"BE_correction_norm":be_corr,
              "BE_fraction_of_correction":None if not res.success else be_corr/(corr_norm+EPS),
              "pairs":pair_info
            })
        # aggregate-mean solve too
        pref={}
        for p in ("P","B","E"):
            pref[p]=np.mean([np.asarray(r["ppo_update_dir_out"],float) for r in snap["preferences"][p]],axis=0)
        u0=np.concatenate([pref["P"],pref["B"],pref["E"]])
        res=solve_min_correction(u0);corr=res.x if res.success else np.full_like(u0,np.nan);uc=u0+corr if res.success else np.full_like(u0,np.nan)
        pair_info={}
        for a,b,name in PAIRS:
            pv=pair_vec(uc,a,b);t=TARGETS[name];margins=[];ok=True
            for j in range(2):
                if t[j]==0: continue
                m=float(t[j]*pv[j]);margins.append(m);ok &= m>=MARGIN-1e-9
            pair_info[name]={"corrected_vector":pv.tolist(),"target_sign":[int(x) for x in t],
                             "min_signed_margin":min(margins) if margins else None,"constraint_ok":bool(ok)}
        ppo_norm=float(np.linalg.norm(u0));corr_norm=float(np.linalg.norm(corr)) if res.success else None
        rows.append({
          "update":update,"suite":"aggregate","feasible":bool(res.success),"solver_status":res.message,
          "ppo_pressure_flat":u0.tolist(),"correction_flat":corr.tolist() if res.success else None,
          "corrected_pressure_flat":uc.tolist() if res.success else None,
          "ppo_norm":ppo_norm,"correction_norm":corr_norm,
          "correction_ratio":None if not res.success else corr_norm/(ppo_norm+EPS),
          "corrected_vs_ppo_cosine":None if not res.success else cosine(uc,u0),
          "shared_correction_norm":None if not res.success else float(np.linalg.norm(corr[[0,2,4]])),
          "BE_correction_norm":None if not res.success else float(np.linalg.norm(corr[[1,3,5]])),
          "BE_fraction_of_correction":None if not res.success else float(np.linalg.norm(corr[[1,3,5]])/(corr_norm+EPS)),
          "pairs":pair_info
        })

    summary={}
    for update in sorted({r["update"] for r in rows}):
        sr=[r for r in rows if r["update"]==update and r["suite"]!="aggregate"]
        ar=next(r for r in rows if r["update"]==update and r["suite"]=="aggregate")
        summary[str(update)]={
          "suite_feasibility_fraction":float(np.mean([r["feasible"] for r in sr])),
          "suite_all_constraints_fraction":float(np.mean([r["feasible"] and all(x["constraint_ok"] for x in r["pairs"].values()) for r in sr])),
          "correction_ratio_mean":float(np.mean([r["correction_ratio"] for r in sr])),
          "correction_ratio_max":float(np.max([r["correction_ratio"] for r in sr])),
          "corrected_vs_ppo_cosine_mean":float(np.mean([r["corrected_vs_ppo_cosine"] for r in sr])),
          "corrected_vs_ppo_cosine_min":float(np.min([r["corrected_vs_ppo_cosine"] for r in sr])),
          "BE_fraction_of_correction_mean":float(np.mean([r["BE_fraction_of_correction"] for r in sr])),
          "aggregate":ar
        }
    report={
      "schema":"post_v2_r23d_min_norm_constraint_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
      "source_r23a_audit":str(args.r23a_audit),"source_r23a_sha256":sha(args.r23a_audit),
      "margin":MARGIN,
      "optimization":"min 0.5||delta u||^2 subject to D1 PB/PE/BE coefficient half-space constraints",
      "rows":rows,"summary":summary,
      "note":"Read-only coefficient-output pressure correction. No optimizer/model updates."
    }
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"updates":list(summary)},indent=2))
if __name__=="__main__":main()
