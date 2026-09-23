#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,numpy as np,torch

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c17_head_tracking-2026-09-23"
ORDER=("T","A","O","S"); NB=6
WINDOWS=(1,2,3,4)
RIDGES=(0.1,1.0)
ADAM_STEPS=(1,5,10,25)
LR=1e-4

def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))

def metrics(Y,P):
    return {"ev":[ev(Y[:,j],P[:,j]) for j in range(4)],
            "mse":[float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]}

def ridge_solution(F,Y,l2):
    A=np.c_[F,np.ones(len(F))]
    I=np.eye(A.shape[1]);I[-1,-1]=0
    sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
    return sol[:-1].T,sol[-1]

def func_progress(Fref,W0,b0,W,b,Wstar,bstar):
    p0=Fref@W0.T+b0;pt=Fref@W.T+b;pstar=Fref@Wstar.T+bstar
    d=pt-p0;ds=pstar-p0
    def c(a,b):
        a=a.reshape(-1);b=b.reshape(-1)
        return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    return {
      "cosine":c(d,ds),
      "norm_progress_ratio":float(np.linalg.norm(d)/(np.linalg.norm(ds)+1e-12)),
      "projected_progress_ratio":float((d.reshape(-1)@ds.reshape(-1))/(np.linalg.norm(ds)**2+1e-12))
    }

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

OUT.mkdir(parents=True,exist_ok=True)
report={"schema":"t5_c17_controlled_head_tracking_v1","specialists":{}}
for lab in ORDER:
    z=np.load(SRC/f"{lab}_features_targets.npz")
    batches=[(z[f"F{k}"].astype(np.float32),z[f"Y{k}"].astype(np.float32)) for k in range(NB)]
    W0=z["head_weight"].astype(np.float32);b0=z["head_bias"].astype(np.float32)
    # stable reference from all 6 recent batches with ridge=1
    Fall=np.concatenate([x[0] for x in batches],0);Yall=np.concatenate([x[1] for x in batches],0)
    Wstar,bstar=ridge_solution(Fall.astype(np.float64),Yall.astype(np.float64),1.0)
    # evaluate progress on all-window feature support to avoid one-batch coordinate bias
    Fref=Fall.astype(np.float64)
    labout={"reference":{"ridge_all6_lambda1":{"weight_norm":float(np.linalg.norm(Wstar))}},"windows":[]}
    for end in range(1,NB):
        nextF,nextY=batches[end]
        for win in WINDOWS:
            start=max(0,end-win);sel=batches[start:end]
            Fw=np.concatenate([x[0] for x in sel],0);Yw=np.concatenate([x[1] for x in sel],0)
            row={"end_batch":end-1,"next_batch":end,"window":win,"actual_window_size":len(sel),"methods":{}}
            # baseline current head
            row["methods"]["baseline"]={
                "window":metrics(Yw,Fw@W0.T+b0),"next":metrics(nextY,nextF@W0.T+b0),
                "progress":func_progress(Fref,W0,b0,W0,b0,Wstar,bstar)
            }
            # Adam head-only from same initial head, fit window
            for steps in ADAM_STEPS:
                W=torch.nn.Parameter(torch.tensor(W0.copy()));b=torch.nn.Parameter(torch.tensor(b0.copy()))
                opt=torch.optim.Adam([W,b],lr=LR)
                Ft=torch.tensor(Fw);Yt=torch.tensor(Yw)
                for _ in range(steps):
                    pred=Ft@W.T+b;loss=((pred-Yt)**2).mean()
                    opt.zero_grad();loss.backward();opt.step()
                Wn=W.detach().numpy();bn=b.detach().numpy()
                row["methods"][f"adam_{steps}"]={
                    "window":metrics(Yw,Fw@Wn.T+bn),"next":metrics(nextY,nextF@Wn.T+bn),
                    "progress":func_progress(Fref,W0,b0,Wn,bn,Wstar,bstar),
                    "weight_norm":float(np.linalg.norm(Wn))
                }
            # ridge closed form on window
            for l2 in RIDGES:
                Wr,br=ridge_solution(Fw.astype(np.float64),Yw.astype(np.float64),l2)
                row["methods"][f"ridge_{l2}"]={
                    "window":metrics(Yw,Fw@Wr.T+br),"next":metrics(nextY,nextF@Wr.T+br),
                    "progress":func_progress(Fref,W0,b0,Wr,br,Wstar,bstar),
                    "weight_norm":float(np.linalg.norm(Wr))
                }
            labout["windows"].append(row)
    report["specialists"][lab]=labout

# aggregate by method/window
agg={}
for win in WINDOWS:
    agg[str(win)]={}
    methods=["baseline"]+[f"adam_{s}" for s in ADAM_STEPS]+[f"ridge_{r}" for r in RIDGES]
    for method in methods:
        next_ev=[];window_ev=[];prog=[];cosines=[];next_mse=[]
        for lab in ORDER:
            for row in report["specialists"][lab]["windows"]:
                if row["window"]!=win:continue
                m=row["methods"][method]
                next_ev+=m["next"]["ev"];window_ev+=m["window"]["ev"];next_mse+=m["next"]["mse"]
                prog.append(m["progress"]["projected_progress_ratio"]);cosines.append(m["progress"]["cosine"])
        agg[str(win)][method]={
            "window_ev_mean":float(np.mean(window_ev)),
            "next_ev_mean":float(np.mean(next_ev)),
            "next_ev_negative_fraction":float(np.mean(np.array(next_ev)<0)),
            "next_mse_mean":float(np.mean(next_mse)),
            "projected_progress_mean":float(np.mean(prog)),
            "function_cosine_mean":float(np.mean(cosines))
        }
report["aggregate"]=agg
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(agg,indent=2))
