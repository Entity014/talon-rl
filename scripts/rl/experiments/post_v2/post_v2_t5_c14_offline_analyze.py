#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/"runs/post_v2_t5_c14_representation_drift-2026-09-23"
ORDER=("T","A","O","S");SNAPS=(0,10,25);RANKS=(16,32,48)
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
 return float(1-np.var(y-p)/(np.var(y)+1e-12))
def cka(X,Y):
 X=X-X.mean(0);Y=Y-Y.mean(0)
 hs=np.linalg.norm(X.T@Y,'fro')**2
 return float(hs/(np.linalg.norm(X.T@X,'fro')*np.linalg.norm(Y.T@Y,'fro')+1e-12))
def subspace_overlap(X,Y,r):
 X=X-X.mean(0);Y=Y-Y.mean(0)
 _,_,Vx=np.linalg.svd(X,full_matrices=False);_,_,Vy=np.linalg.svd(Y,full_matrices=False)
 Qx=Vx[:r].T;Qy=Vy[:r].T;s=np.linalg.svd(Qx.T@Qy,compute_uv=False)
 return {"mean_cos":float(np.mean(s)),"min_cos":float(np.min(s)),"mean_angle_deg":float(np.degrees(np.arccos(np.clip(s,-1,1))).mean())}
def spectrum(F):
 X=F-F.mean(0);s=np.linalg.svd(X,compute_uv=False)
 return {"cond":float(s.max()/(s.min()+1e-12)),"rank_1e2":int(np.sum(s>1e-2*s.max())),"rank_1e3":int(np.sum(s>1e-3*s.max())),
         "top16_energy":float(np.sum(s[:16]**2)/np.sum(s**2)),"top32_energy":float(np.sum(s[:32]**2)/np.sum(s**2))}
report={"schema":"t5_c14_representation_drift_v1","specialists":{}}
for lab in ORDER:
 z=np.load(OUT/f"{lab}_matched.npz")
 Y=z["Y"].astype(np.float64)
 F={s:z[f"F{s}"].astype(np.float64) for s in SNAPS}
 W={s:z[f"W{s}"].astype(np.float64) for s in SNAPS}
 b={s:z[f"b{s}"].astype(np.float64) for s in SNAPS}
 P={s:z[f"P{s}"].astype(np.float64) for s in SNAPS}
 pair={}
 for i,j in ((0,10),(10,25),(0,25)):
  d={"cka":cka(F[i],F[j]),"feature_relative_l2":float(np.linalg.norm(F[j]-F[i])/(np.linalg.norm(F[i])+1e-12)),
     "mean_feature_cos":float(np.mean(np.sum(F[i]*F[j],1)/(np.linalg.norm(F[i],axis=1)*np.linalg.norm(F[j],axis=1)+1e-12)))}
  for r in RANKS:d[f"subspace_r{r}"]=subspace_overlap(F[i],F[j],r)
  pair[f"{i}->{j}"]=d
 # matched same-body head transplants
 trans={}
 for bs in SNAPS:
  trans[str(bs)]={}
  for hs in SNAPS:
   pred=F[bs]@W[hs].T+b[hs]
   trans[str(bs)][str(hs)]={"ev":[ev(Y[:,k],pred[:,k]) for k in range(4)],"mse":[float(np.mean((pred[:,k]-Y[:,k])**2)) for k in range(4)]}
 # best possible linear head on each frozen body
 best={}
 for s in SNAPS:
  A=np.c_[F[s],np.ones(len(F[s]))];sol=np.linalg.lstsq(A,Y,rcond=None)[0];pred=A@sol
  best[str(s)]={"ev":[ev(Y[:,k],pred[:,k]) for k in range(4)],"solution_norm":float(np.linalg.norm(sol[:-1]))}
 # actual predictions
 actual={str(s):{"ev":[ev(Y[:,k],P[s][:,k]) for k in range(4)]} for s in SNAPS}
 # head drift
 hd={}
 for i,j in ((0,10),(10,25),(0,25)):
  hd[f"{i}->{j}"]={"weight_l2":float(np.linalg.norm(W[j]-W[i])),"bias_l2":float(np.linalg.norm(b[j]-b[i])),
                   "relative_weight_l2":float(np.linalg.norm(W[j]-W[i])/(np.linalg.norm(W[i])+1e-12))}
 report["specialists"][lab]={"feature_pairwise":pair,"spectrum":{str(s):spectrum(F[s]) for s in SNAPS},
                             "transplant":trans,"best_linear":best,"actual":actual,"head_drift":hd}
# aggregate
agg={}
for key in ("0->10","10->25","0->25"):
 vals=[report["specialists"][lab]["feature_pairwise"][key] for lab in ORDER]
 agg.setdefault("feature_pairwise",{})[key]={
  "cka_mean":float(np.mean([x["cka"] for x in vals])),
  "feature_relative_l2_mean":float(np.mean([x["feature_relative_l2"] for x in vals])),
  "mean_feature_cos_mean":float(np.mean([x["mean_feature_cos"] for x in vals])),
  "subspace_r16_mean_cos":float(np.mean([x["subspace_r16"]["mean_cos"] for x in vals])),
  "subspace_r32_mean_cos":float(np.mean([x["subspace_r32"]["mean_cos"] for x in vals])),
  "subspace_r48_mean_cos":float(np.mean([x["subspace_r48"]["mean_cos"] for x in vals]))
 }
# spectrum evolution
for s in SNAPS:
 sp=[report["specialists"][lab]["spectrum"][str(s)] for lab in ORDER]
 agg.setdefault("spectrum",{})[str(s)]={"cond_median":float(np.median([x["cond"] for x in sp])),
  "rank_1e2_mean":float(np.mean([x["rank_1e2"] for x in sp])),"top16_energy_mean":float(np.mean([x["top16_energy"] for x in sp])),
  "top32_energy_mean":float(np.mean([x["top32_energy"] for x in sp]))}
# actual vs transplants: aggregate all 16 head objectives
for bs in SNAPS:
 for hs in SNAPS:
  vals=[]
  for lab in ORDER: vals+=report["specialists"][lab]["transplant"][str(bs)][str(hs)]["ev"]
  agg.setdefault("transplant_ev_mean",{}).setdefault(str(bs),{})[str(hs)]=float(np.mean(vals))
# best and actual
for s in SNAPS:
 be=[];ac=[]
 for lab in ORDER:
  be+=report["specialists"][lab]["best_linear"][str(s)]["ev"];ac+=report["specialists"][lab]["actual"][str(s)]["ev"]
 agg.setdefault("best_linear_ev_mean",{})[str(s)]=float(np.mean(be));agg.setdefault("actual_ev_mean",{})[str(s)]=float(np.mean(ac))
# head drift
for key in ("0->10","10->25","0->25"):
 hs=[report["specialists"][lab]["head_drift"][key] for lab in ORDER]
 agg.setdefault("head_drift",{})[key]={"weight_l2_mean":float(np.mean([x["weight_l2"] for x in hs])),
  "relative_weight_l2_mean":float(np.mean([x["relative_weight_l2"] for x in hs]))}
report["aggregate"]=agg
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(agg,indent=2))
