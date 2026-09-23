#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,numpy as np,torch

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c15_spectral_update-2026-09-23"
ORDER=("T","A","O","S"); NB=6; LR=1e-4; THRESH=1e-2

def ev(y,p):
    y=np.asarray(y,float).reshape(-1); p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))

def metrics(Y,P):
    return {
        "ev":[ev(Y[:,j],P[:,j]) for j in range(4)],
        "mse":[float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]
    }

def cos(a,b):
    a=np.asarray(a).reshape(-1); b=np.asarray(b).reshape(-1)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

OUT.mkdir(parents=True,exist_ok=True)
report={"schema":"t5_c15_spectral_update_audit_v1","threshold_relative_to_smax":THRESH,
        "adam_lr":LR,"bias_policy":"same full bias update in strong-only and weak-only; bias-only control reported","specialists":{}}

for lab in ORDER:
    z=np.load(SRC/f"{lab}_features_targets.npz")
    batches=[(z[f"F{k}"].astype(np.float32),z[f"Y{k}"].astype(np.float32)) for k in range(NB)]
    W0=z["head_weight"].astype(np.float32); b0=z["head_bias"].astype(np.float32)
    rows=[]
    for k in range(NB-1):
        F,Y=batches[k]; Fn,Yn=batches[k+1]
        # spectral basis from current batch feature matrix
        X=F-F.mean(0,keepdims=True)
        _,S,Vt=np.linalg.svd(X.astype(np.float64),full_matrices=False)
        r=int(np.sum(S>=THRESH*S.max()))
        Vr=Vt[:r].T.astype(np.float32)
        Pstrong=Vr@Vr.T
        Pweak=np.eye(F.shape[1],dtype=np.float32)-Pstrong

        Ft=torch.tensor(F); Yt=torch.tensor(Y)
        W=torch.nn.Parameter(torch.tensor(W0.copy())); b=torch.nn.Parameter(torch.tensor(b0.copy()))
        opt=torch.optim.Adam([W,b],lr=LR)
        pred=Ft@W.T+b
        loss=((pred-Yt)**2).mean()
        opt.zero_grad(); loss.backward()
        gW=W.grad.detach().numpy().copy(); gb=b.grad.detach().numpy().copy()
        opt.step()
        dW=(W.detach().numpy()-W0); db=(b.detach().numpy()-b0)

        # decompose gradient and actual one-step Adam delta into feature singular subspaces
        gWs=gW@Pstrong; gWw=gW@Pweak
        dWs=dW@Pstrong; dWw=dW@Pweak
        # counterfactual states: keep same full bias update except bias-only
        variants={
            "base":(W0,b0),
            "full":(W0+dW,b0+db),
            "strong_only":(W0+dWs,b0+db),
            "weak_only":(W0+dWw,b0+db),
            "bias_only":(W0,b0+db),
            "weight_full_no_bias":(W0+dW,b0)
        }
        evals={}
        for name,(WV,bV) in variants.items():
            evals[name]={
                "current":metrics(Y,F@WV.T+bV),
                "next":metrics(Yn,Fn@WV.T+bV)
            }

        # per-head energy
        per_head=[]
        for j in range(4):
            gs=np.linalg.norm(gWs[j]); gw=np.linalg.norm(gWw[j])
            ds=np.linalg.norm(dWs[j]); dw=np.linalg.norm(dWw[j])
            per_head.append({
                "head":j,
                "gradient_strong_norm":float(gs),"gradient_weak_norm":float(gw),
                "gradient_weak_energy_fraction":float(gw*gw/(gs*gs+gw*gw+1e-12)),
                "update_strong_norm":float(ds),"update_weak_norm":float(dw),
                "update_weak_energy_fraction":float(dw*dw/(ds*ds+dw*dw+1e-12)),
                "strong_to_weak_update_norm_ratio":float(ds/(dw+1e-12))
            })
        rows.append({
            "pair":f"{k}->{k+1}","effective_rank":r,"singular_value_ratio_boundary":float(S[r-1]/S[0]) if r>0 else None,
            "gradient_weak_energy_fraction":float(np.linalg.norm(gWw)**2/(np.linalg.norm(gW)**2+1e-12)),
            "update_weak_energy_fraction":float(np.linalg.norm(dWw)**2/(np.linalg.norm(dW)**2+1e-12)),
            "gradient_strong_cosine_full":cos(gWs,gW),
            "update_strong_cosine_full":cos(dWs,dW),
            "gradient_weak_cosine_full":cos(gWw,gW),
            "update_weak_cosine_full":cos(dWw,dW),
            "bias_update_norm":float(np.linalg.norm(db)),
            "full_weight_update_norm":float(np.linalg.norm(dW)),
            "per_head":per_head,"eval":evals
        })
    report["specialists"][lab]=rows

# aggregate deltas relative to base
agg={}
weakE=[];gradWeak=[];strongCos=[];weakCos=[];ranks=[]
head_weak=[[] for _ in range(4)]
variant_stats={v:{"cur_ev_delta":[],"next_ev_delta":[],"cur_mse_improve":[],"next_mse_improve":[]} for v in ("full","strong_only","weak_only","bias_only","weight_full_no_bias")}
for lab in ORDER:
    for row in report["specialists"][lab]:
        weakE.append(row["update_weak_energy_fraction"]);gradWeak.append(row["gradient_weak_energy_fraction"])
        strongCos.append(row["update_strong_cosine_full"]);weakCos.append(row["update_weak_cosine_full"]);ranks.append(row["effective_rank"])
        for h in row["per_head"]:head_weak[h["head"]].append(h["update_weak_energy_fraction"])
        basec=row["eval"]["base"]["current"];basen=row["eval"]["base"]["next"]
        for v in variant_stats:
            vc=row["eval"][v]["current"];vn=row["eval"][v]["next"]
            variant_stats[v]["cur_ev_delta"] += (np.array(vc["ev"])-np.array(basec["ev"])).tolist()
            variant_stats[v]["next_ev_delta"] += (np.array(vn["ev"])-np.array(basen["ev"])).tolist()
            variant_stats[v]["cur_mse_improve"] += ((np.array(basec["mse"])-np.array(vc["mse"]))/(np.array(basec["mse"])+1e-12)).tolist()
            variant_stats[v]["next_mse_improve"] += ((np.array(basen["mse"])-np.array(vn["mse"]))/(np.array(basen["mse"])+1e-12)).tolist()

agg["effective_rank_mean"]=float(np.mean(ranks))
agg["gradient_weak_energy_fraction_mean"]=float(np.mean(gradWeak))
agg["update_weak_energy_fraction_mean"]=float(np.mean(weakE))
agg["update_strong_cosine_full_mean"]=float(np.mean(strongCos))
agg["update_weak_cosine_full_mean"]=float(np.mean(weakCos))
agg["per_head_update_weak_energy_fraction_mean"]=[float(np.mean(x)) for x in head_weak]
agg["variants"]={}
for v,x in variant_stats.items():
    agg["variants"][v]={
        "current_ev_delta_mean":float(np.mean(x["cur_ev_delta"])),
        "next_ev_delta_mean":float(np.mean(x["next_ev_delta"])),
        "current_mse_improvement_mean":float(np.mean(x["cur_mse_improve"])),
        "next_mse_improvement_mean":float(np.mean(x["next_mse_improve"])),
        "next_ev_improve_fraction":float(np.mean(np.array(x["next_ev_delta"])>0)),
        "next_mse_improve_fraction":float(np.mean(np.array(x["next_mse_improve"])>0))
    }
# causal contrasts
agg["contrasts"]={
    "strong_minus_weak_next_ev_delta":agg["variants"]["strong_only"]["next_ev_delta_mean"]-agg["variants"]["weak_only"]["next_ev_delta_mean"],
    "strong_minus_weak_current_ev_delta":agg["variants"]["strong_only"]["current_ev_delta_mean"]-agg["variants"]["weak_only"]["current_ev_delta_mean"],
    "full_minus_strong_next_ev_delta":agg["variants"]["full"]["next_ev_delta_mean"]-agg["variants"]["strong_only"]["next_ev_delta_mean"]
}
report["aggregate"]=agg
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(agg,indent=2))
