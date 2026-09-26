#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[4]
C14=ROOT/"runs/post_v2_t5_c14_representation_drift-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c16_head_progress-2026-09-23"
ORDER=("T","A","O","S");SNAPS=(0,10,25);RIDGE=1.0;PCR=32
def cos(a,b):
 a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
 return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def ridge(F,Y,l2):
 A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0;sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return A@sol
def pcr(F,Y,r):
 mu=F.mean(0,keepdims=True);X=F-mu;U,S,Vt=np.linalg.svd(X,full_matrices=False);Vr=Vt[:r].T;A=np.c_[X@Vr,np.ones(len(X))]
 sol=np.linalg.lstsq(A,Y,rcond=None)[0];return A@sol
def metric(p0,pt,pstar):
 d=pt-p0;ds=pstar-p0
 return {"cosine":cos(d,ds),"norm_progress_ratio":float(np.linalg.norm(d)/(np.linalg.norm(ds)+1e-12)),
         "projected_progress_ratio":float((d.reshape(-1)@ds.reshape(-1))/(np.linalg.norm(ds)**2+1e-12)),
         "orthogonal_fraction":float(np.linalg.norm(d-((d.reshape(-1)@ds.reshape(-1))/(np.linalg.norm(ds)**2+1e-12))*ds)/(np.linalg.norm(d)+1e-12))}
report={"schema":"t5_c16_function_space_progress_v1","specialists":{}}
for lab in ORDER:
 z=np.load(C14/f"{lab}_matched.npz");Y=z["Y"].astype(float);F=z["F25"].astype(float)
 W={s:z[f"W{s}"].astype(float) for s in SNAPS};b={s:z[f"b{s}"].astype(float) for s in SNAPS}
 P={s:F@W[s].T+b[s] for s in SNAPS}
 refs={"ridge":ridge(F,Y,RIDGE),"pcr":pcr(F,Y,PCR)}
 q={}
 for rn,ps in refs.items():
  q[rn]={}
  for t in (10,25):
   q[rn][str(t)]={"all":metric(P[0],P[t],ps),"per_head":[metric(P[0][:,j:j+1],P[t][:,j:j+1],ps[:,j:j+1]) for j in range(4)]}
 report["specialists"][lab]=q
agg={}
for rn in ("ridge","pcr"):
 agg[rn]={}
 for t in ("10","25"):
  vals=[report["specialists"][lab][rn][t]["all"] for lab in ORDER]
  agg[rn][t]={"cosine_mean":float(np.mean([x["cosine"] for x in vals])),"cosine_median":float(np.median([x["cosine"] for x in vals])),
              "norm_progress_ratio_mean":float(np.mean([x["norm_progress_ratio"] for x in vals])),
              "projected_progress_ratio_mean":float(np.mean([x["projected_progress_ratio"] for x in vals])),
              "orthogonal_fraction_mean":float(np.mean([x["orthogonal_fraction"] for x in vals]))}
 report.setdefault("aggregate_per_head",{}).setdefault(rn,{})
 for j in range(4):
  for t in ("10","25"):
   vals=[report["specialists"][lab][rn][t]["per_head"][j] for lab in ORDER]
   report["aggregate_per_head"][rn].setdefault(str(j),{})[t]={"cosine_mean":float(np.mean([x["cosine"] for x in vals])),
    "projected_progress_ratio_mean":float(np.mean([x["projected_progress_ratio"] for x in vals]))}
report["aggregate"]=agg
(OUT/"function_space.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"aggregate":agg,"per_head":report["aggregate_per_head"]},indent=2))
