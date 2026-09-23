#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np,torch

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c15_spectral_update-2026-09-23"
ORDER=("T","A","O","S"); NB=6; LR=1e-4; THRESH=1e-2

def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def metrics(Y,P):
    return {"ev":[ev(Y[:,j],P[:,j]) for j in range(4)],
            "mse":[float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]}
def projector(F):
    X=F-F.mean(0,keepdims=True);_,S,Vt=np.linalg.svd(X.astype(np.float64),full_matrices=False)
    r=int(np.sum(S>=THRESH*S.max()));Vr=Vt[:r].T.astype(np.float32);Ps=Vr@Vr.T
    return Ps,np.eye(F.shape[1],dtype=np.float32)-Ps,r

report={"schema":"t5_c15_sequential_projection_v1","specialists":{}}
for lab in ORDER:
    z=np.load(SRC/f"{lab}_features_targets.npz")
    B=[(z[f"F{k}"].astype(np.float32),z[f"Y{k}"].astype(np.float32)) for k in range(NB)]
    W0=z["head_weight"].astype(np.float32);b0=z["head_bias"].astype(np.float32)
    branch={}
    for mode in ("full","strong_only","weak_only"):
        W=torch.nn.Parameter(torch.tensor(W0.copy()));b=torch.nn.Parameter(torch.tensor(b0.copy()))
        opt=torch.optim.Adam([W,b],lr=LR)
        trace=[]
        for k,(F,Y) in enumerate(B):
            Ft=torch.tensor(F);Yt=torch.tensor(Y)
            Ps,Pw,r=projector(F)
            preW=W.detach().clone();preb=b.detach().clone()
            pred=Ft@W.T+b;loss=((pred-Yt)**2).mean()
            opt.zero_grad();loss.backward();opt.step()
            postW=W.detach().clone();postb=b.detach().clone()
            dW=(postW-preW).numpy();db=(postb-preb).numpy()
            # project the APPLIED Adam weight delta; keep full bias step in all branches
            if mode=="strong_only": dW=dW@Ps
            elif mode=="weak_only": dW=dW@Pw
            with torch.no_grad():
                W.copy_(preW+torch.tensor(dW));b.copy_(preb+torch.tensor(db))
            cur=metrics(Y,(F@W.detach().numpy().T+b.detach().numpy()))
            nxt=None
            if k+1<NB:
                Fn,Yn=B[k+1];nxt=metrics(Yn,(Fn@W.detach().numpy().T+b.detach().numpy()))
            # retrospective average EV on all seen batches
            seen=[]
            for q in range(k+1):
                Fq,Yq=B[q];seen+=metrics(Yq,Fq@W.detach().numpy().T+b.detach().numpy())["ev"]
            trace.append({"batch":k,"effective_rank":r,"current":cur,"next":nxt,"seen_ev_mean":float(np.mean(seen)),
                          "weight_norm":float(W.norm()),"bias_norm":float(b.norm())})
        all_ev=[];all_mse=[]
        for F,Y in B:
            mm=metrics(Y,F@W.detach().numpy().T+b.detach().numpy());all_ev+=mm["ev"];all_mse+=mm["mse"]
        branch[mode]={"trace":trace,"final_all_ev_mean":float(np.mean(all_ev)),"final_all_mse_mean":float(np.mean(all_mse)),
                      "final_weight_norm":float(W.norm())}
    report["specialists"][lab]=branch

agg={}
for mode in ("full","strong_only","weak_only"):
    cur=[];nxt=[];seen=[];final=[];norm=[]
    for lab in ORDER:
        q=report["specialists"][lab][mode];final.append(q["final_all_ev_mean"]);norm.append(q["final_weight_norm"])
        for tr in q["trace"]:
            cur+=tr["current"]["ev"];seen.append(tr["seen_ev_mean"])
            if tr["next"] is not None:nxt+=tr["next"]["ev"]
    agg[mode]={"current_ev_mean":float(np.mean(cur)),"next_ev_mean":float(np.mean(nxt)),
               "next_ev_negative_fraction":float(np.mean(np.array(nxt)<0)),
               "seen_ev_mean":float(np.mean(seen)),"final_all_ev_mean":float(np.mean(final)),
               "final_weight_norm_mean":float(np.mean(norm))}
report["aggregate"]=agg
(OUT/"sequential_projection.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(agg,indent=2))
