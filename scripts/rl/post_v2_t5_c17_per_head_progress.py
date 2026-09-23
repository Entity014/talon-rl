#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c17_head_tracking-2026-09-23"
ORDER=("T","A","O","S");NB=6
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
def ridge(F,Y,l2):
 A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0;sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
def prog(F,W0,b0,W,b,Ws,bs,j):
 p0=F@W0[j]+b0[j];pt=F@W[j]+b[j];ps=F@Ws[j]+bs[j];d=pt-p0;ds=ps-p0
 return float(d@ds/(ds@ds+1e-12))
out={"schema":"c17_per_head_progress_v1","specialists":{}}
for lab in ORDER:
 z=np.load(SRC/f"{lab}_features_targets.npz");B=[(z[f"F{k}"].astype(float),z[f"Y{k}"].astype(float)) for k in range(NB)]
 W0=z["head_weight"].astype(float);b0=z["head_bias"].astype(float)
 Fall=np.concatenate([x[0] for x in B]);Yall=np.concatenate([x[1] for x in B]);Ws,bs=ridge(Fall,Yall,1.0)
 rows=[]
 for win in (1,2,3,4):
  pe=[[],[],[],[]];nev=[[],[],[],[]]
  for end in range(1,NB):
   sel=B[max(0,end-win):end];Fw=np.concatenate([x[0] for x in sel]);Yw=np.concatenate([x[1] for x in sel]);Fn,Yn=B[end]
   W,b=ridge(Fw,Yw,1.0)
   for j in range(4):
    pe[j].append(prog(Fall,W0,b0,W,b,Ws,bs,j));nev[j].append(ev(Yn[:,j],Fn@W[j]+b[j]))
  rows.append({"window":win,"projected_progress_by_head":[float(np.mean(x)) for x in pe],"next_ev_by_head":[float(np.mean(x)) for x in nev]})
 out["specialists"][lab]=rows
agg={}
for win in (1,2,3,4):
 pe=[[],[],[],[]];ne=[[],[],[],[]]
 for lab in ORDER:
  r=next(x for x in out["specialists"][lab] if x["window"]==win)
  for j in range(4):pe[j].append(r["projected_progress_by_head"][j]);ne[j].append(r["next_ev_by_head"][j])
 agg[str(win)]={"projected_progress_by_head":[float(np.mean(x)) for x in pe],"next_ev_by_head":[float(np.mean(x)) for x in ne]}
out["aggregate"]=agg
(OUT/"per_head_progress.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(agg,indent=2))
