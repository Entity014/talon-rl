#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"runs/post_v2_t5_c20_coverage-2026-09-23"
ORDER=("T","A","O","S");RIDGE=1.0
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
def fit(F,Y):
 A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0;sol=np.linalg.solve(A.T@A+RIDGE*I,A.T@Y);return sol[:-1].T,sol[-1]
def evaluate(W,b,items):
 es=[];ms=[]
 for F,Y in items:
  P=F@W.T+b
  es += [ev(Y[:,j],P[:,j]) for j in range(4)]
  ms += [float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]
 return {"ev_mean":float(np.mean(es)),"ev_negative_fraction":float(np.mean(np.array(es)<0)),"mse_mean":float(np.mean(ms)),
         "ev_by_head":[float(np.mean(es[j::4])) for j in range(4)]}
def select_diverse(S,k):
 # use command+feature mean/std only, excluding final 8 target summary coordinates
 X=S[:,:-8].astype(float)
 X=(X-X.mean(0))/(X.std(0)+1e-6)
 # deterministic farthest point: start farthest from centroid
 d0=np.sum(X*X,1);sel=[int(np.argmax(d0))]
 mind=np.sum((X-X[sel[0]])**2,1)
 while len(sel)<k:
  mind[sel]=-1
  q=int(np.argmax(mind));sel.append(q)
  mind=np.minimum(mind,np.sum((X-X[q])**2,1))
 return sorted(sel)
def summary_distance(trainS,testS):
 # report standardized nearest-summary distance; command-only and feature-only
 out={}
 for name,sl in [("command",slice(0,6)),("feature",slice(6,-8)),("target",slice(-8,None))]:
  A=trainS[:,sl].astype(float);B=testS[:,sl].astype(float);mu=A.mean(0);sd=A.std(0)+1e-6;Az=(A-mu)/sd;Bz=(B-mu)/sd
  ds=[]
  for b in Bz: ds.append(float(np.sqrt(np.min(np.mean((Az-b)**2,axis=1)))))
  out[name+"_nn_std_distance_mean"]=float(np.mean(ds))
  out[name+"_nn_std_distance_max"]=float(np.max(ds))
 return out
report={"schema":"c20_coverage_audit_v1","specialists":{}}
for lab in ORDER:
 z=np.load(OUT/f"{lab}_pool.npz")
 train=[(z[f"train_F{i}"].astype(float),z[f"train_Y{i}"].astype(float)) for i in range(16)]
 test=[(z[f"test_F{i}"].astype(float),z[f"test_Y{i}"].astype(float)) for i in range(8)]
 S=np.stack([z[f"train_S{i}"] for i in range(16)]);St=np.stack([z[f"test_S{i}"] for i in range(8)])
 supports={
  "recent3":list(range(13,16)),
  "recent6":list(range(10,16)),
  "recent12":list(range(4,16)),
  "all16":list(range(16)),
  "diverse6":select_diverse(S,6),
  "diverse12":select_diverse(S,12)
 }
 q={}
 for name,idx in supports.items():
  F=np.concatenate([train[i][0] for i in idx]);Y=np.concatenate([train[i][1] for i in idx]);W,b=fit(F,Y)
  q[name]={"indices":idx,"train":evaluate(W,b,[train[i] for i in idx]),"fresh":evaluate(W,b,test),
           "weight_norm":float(np.linalg.norm(W)),"bias_norm":float(np.linalg.norm(b)),
           "coverage":summary_distance(S[idx],St)}
 # rolling solution drift for temporal windows
 for win in (3,6,12):
  sols=[]
  for end in range(win,17):
   idx=list(range(end-win,end));F=np.concatenate([train[i][0] for i in idx]);Y=np.concatenate([train[i][1] for i in idx]);W,b=fit(F,Y);sols.append(np.r_[W.reshape(-1),b])
  dr=[np.linalg.norm(sols[i]-sols[i-1]) for i in range(1,len(sols))]
  q[f"recent{win}"]["rolling_solution_drift_mean"]=float(np.mean(dr)) if dr else 0.0
  q[f"recent{win}"]["rolling_solution_drift_p90"]=float(np.quantile(dr,.9)) if dr else 0.0
 report["specialists"][lab]=q
agg={}
for name in ("recent3","recent6","recent12","all16","diverse6","diverse12"):
 fresh=[];train_ev=[];wn=[];cmd=[];feat=[];targ=[];by=[[] for _ in range(4)]
 drift=[]
 for lab in ORDER:
  q=report["specialists"][lab][name];fresh.append(q["fresh"]["ev_mean"]);train_ev.append(q["train"]["ev_mean"]);wn.append(q["weight_norm"])
  cmd.append(q["coverage"]["command_nn_std_distance_mean"]);feat.append(q["coverage"]["feature_nn_std_distance_mean"]);targ.append(q["coverage"]["target_nn_std_distance_mean"])
  for j,x in enumerate(q["fresh"]["ev_by_head"]):by[j].append(x)
  if "rolling_solution_drift_mean" in q:drift.append(q["rolling_solution_drift_mean"])
 agg[name]={"train_ev_mean":float(np.mean(train_ev)),"fresh_ev_mean":float(np.mean(fresh)),
            "fresh_ev_by_head":[float(np.mean(x)) for x in by],
            "weight_norm_mean":float(np.mean(wn)),
            "command_nn_distance_mean":float(np.mean(cmd)),"feature_nn_distance_mean":float(np.mean(feat)),"target_nn_distance_mean":float(np.mean(targ))}
 if drift:agg[name]["rolling_solution_drift_mean"]=float(np.mean(drift))
report["aggregate"]=agg
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(agg,indent=2))
