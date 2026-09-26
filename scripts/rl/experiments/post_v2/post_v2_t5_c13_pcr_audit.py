#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
ORDER=("T","A","O","S");NB=6;RANKS=(16,32,48,64,96,128)
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
 return float(1-np.var(y-p)/(np.var(y)+1e-12))
def metrics(Y,P):return {"ev":[ev(Y[:,j],P[:,j]) for j in range(4)]}
out={"ranks":RANKS,"specialists":{}}
for lab in ORDER:
 z=np.load(OUT/f"{lab}_features_targets.npz")
 B=[(z[f"F{k}"].astype(np.float64),z[f"Y{k}"].astype(np.float64)) for k in range(NB)]
 labout={}
 for rank in RANKS:
  rows=[]
  for k in range(NB-1):
   F,Y=B[k];Fn,Yn=B[k+1];mu=F.mean(0);X=F-mu;Xn=Fn-mu
   U,S,Vt=np.linalg.svd(X,full_matrices=False);r=min(rank,len(S));Vr=Vt[:r].T
   Z=X@Vr;Zn=Xn@Vr
   A=np.c_[Z,np.ones(len(Z))];An=np.c_[Zn,np.ones(len(Zn))]
   sol=np.linalg.lstsq(A,Y,rcond=None)[0]
   rows.append({"pair":f"{k}->{k+1}","self":metrics(Y,A@sol),"prior_on_next":metrics(Yn,An@sol),
                "retained_singular_energy":float(np.sum(S[:r]**2)/np.sum(S**2))})
  labout[str(rank)]=rows
 out["specialists"][lab]=labout
agg={}
for rank in RANKS:
 se=[];ne=[];energy=[]
 for lab in ORDER:
  for row in out["specialists"][lab][str(rank)]:
   se+=row["self"]["ev"];ne+=row["prior_on_next"]["ev"];energy.append(row["retained_singular_energy"])
 agg[str(rank)]={"self_ev_mean":float(np.mean(se)),"next_ev_mean":float(np.mean(ne)),
                 "next_ev_negative_fraction":float(np.mean(np.array(ne)<0)),
                 "retained_singular_energy_mean":float(np.mean(energy))}
out["aggregate"]=agg;(OUT/"pcr_audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
