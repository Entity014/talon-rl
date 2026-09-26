#!/usr/bin/env python3
"""V2-R2.3-C read-only normalized semantic guidance audit."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
RHO=(0.01,0.025,0.05,0.1,0.2,0.3,0.5,0.75,1.0)
EPS=1e-12
TARGETS={"PB":np.array([1.,-1.]),"PE":np.array([1.,1.]),"BE":np.array([0.,1.])}
PAIRS=(("P","B","PB"),("P","E","PE"),("B","E","BE"))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sign(x,tol=1e-12): return 0 if abs(x)<=tol else (1 if x>0 else -1)
def pair_ok(v,t):
    if t[0] and sign(v[0])!=int(t[0]): return False
    if t[1] and sign(v[1])!=int(t[1]): return False
    return True
def cosine(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+EPS))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--r23a-audit",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    d=json.loads(args.r23a_audit.read_text())

    snaps=[]
    for s in d["snapshots"]:
        update=s["update"]; rho_rows={}
        for rho in RHO:
            pref_comb={}
            pref_cos={}
            for pref,rs in s["preferences"].items():
                comb=[];cos=[]
                for r in rs:
                    p=np.asarray(r["ppo_update_dir_out"],float)
                    g=np.asarray(r["coeff_update_dir_out"],float)
                    gn=np.linalg.norm(g)
                    guide=np.zeros_like(g) if gn<EPS else rho*np.linalg.norm(p)*g/(gn+EPS)
                    total=p+guide
                    comb.append(total);cos.append(cosine(total,p))
                pref_comb[pref]=comb;pref_cos[pref]=cos

            pairs={};all_agg=True;all_suite=True
            for a,b,name in PAIRS:
                arr=np.asarray(pref_comb[b])-np.asarray(pref_comb[a])
                mean=arr.mean(0);target=TARGETS[name]
                suite_ok=[pair_ok(v,target) for v in arr]
                agg_ok=pair_ok(mean,target)
                all_agg &= agg_ok
                all_suite &= all(suite_ok)
                pairs[name]={
                    "mean_vector":mean.tolist(),
                    "sign":[sign(x) for x in mean],
                    "target_sign":[int(x) for x in target],
                    "aggregate_ok":bool(agg_ok),
                    "suite_match_fraction":float(np.mean(suite_ok)),
                    "mean_cosine_to_target":float(np.mean([cosine(v,target) for v in arr])),
                }
            # objective-pressure preservation: compare total guidance-adjusted pressure to original PPO pressure
            all_cos=[x for pref in pref_cos.values() for x in pref]
            rho_rows[str(rho)]={
                "pairs":pairs,
                "all_aggregate_ok":bool(all_agg),
                "all_suite_ok":bool(all_suite),
                "mean_combined_vs_ppo_cosine":float(np.mean(all_cos)),
                "min_combined_vs_ppo_cosine":float(np.min(all_cos)),
                "guide_norm_budget_ratio":rho,
            }
        snaps.append({"update":update,"rho_candidates":rho_rows})

    effective=[]
    for rho in RHO:
        k=str(rho)
        if all(s["rho_candidates"][k]["all_aggregate_ok"] for s in snaps):
            effective.append(rho)
    effective_suite=[]
    for rho in RHO:
        k=str(rho)
        if all(s["rho_candidates"][k]["all_suite_ok"] for s in snaps):
            effective_suite.append(rho)

    report={
      "schema":"post_v2_r23c_normalized_guidance_audit_v1",
      "status":"MEASUREMENT_COMPLETE","measurement_only":True,
      "source_r23a_audit":str(args.r23a_audit),
      "source_r23a_sha256":sha(args.r23a_audit),
      "rho_candidates":list(RHO),
      "snapshots":snaps,
      "minimum_all_snapshot_aggregate_orientation_rho":effective[0] if effective else None,
      "minimum_all_snapshot_all_suite_orientation_rho":effective_suite[0] if effective_suite else None,
      "acceptance_rule":{
        "orientation":"PB/PE/BE aggregate signs correct at updates 0/1/5/10",
        "budget":"guide norm fixed to rho * ||PPO output-pressure|| per preference/suite",
        "ppo_preservation":"report combined-vs-PPO cosine; no training authorization in this audit"
      },
      "note":"Pure read-only algebra on R2.3-A coefficient-output pressures. No optimizer/model updates."
    }
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"min_aggregate_rho":report["minimum_all_snapshot_aggregate_orientation_rho"],"min_all_suite_rho":report["minimum_all_snapshot_all_suite_orientation_rho"]},indent=2))
if __name__=="__main__":main()
