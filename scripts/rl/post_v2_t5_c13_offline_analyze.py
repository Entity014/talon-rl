#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
ORDER=("T","A","O","S");NB=6
RIDGES=(0.0,1e-4,1e-3,1e-2,1e-1,1.0)
SEQ_L2=(0.0,1e-4,1e-3,1e-2,1e-1,1.0)
def ev(y,p):
    y=np.asarray(y,float).reshape(-1); p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def metrics(Y,P):
    return {"ev":[ev(Y[:,j],P[:,j]) for j in range(4)],"mse":[float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]}
def solve_ridge(A,Y,l2):
    if l2==0: return np.linalg.lstsq(A,Y,rcond=None)[0]
    I=np.eye(A.shape[1]);I[-1,-1]=0
    return np.linalg.solve(A.T@A+l2*I,A.T@Y)
def whiten_fit(F,Y,Fn,rel_floor):
    mu=F.mean(0,keepdims=True);X=F-mu
    C=(X.T@X)/max(1,len(F)-1)
    evals,evecs=np.linalg.eigh(C);mx=max(float(evals.max()),1e-12);floor=rel_floor*mx
    inv=1/np.sqrt(np.maximum(evals,floor));W=evecs@np.diag(inv)@evecs.T
    Fb=(F-mu)@W;Fnext=(Fn-mu)@W
    A=np.c_[Fb,np.ones(len(Fb))];An=np.c_[Fnext,np.ones(len(Fnext))]
    sol=np.linalg.lstsq(A,Y,rcond=None)[0]
    s=np.linalg.svd(Fb-Fb.mean(0),compute_uv=False)
    return metrics(Y,A@sol),An@sol,float(s.max()/(s.min()+1e-12))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={"schema":"t5_c13_conditioning_regularization_v2","specialists":{}}
for lab in ORDER:
    z=np.load(OUT/f"{lab}_features_targets.npz")
    batches=[(z[f"F{k}"].astype(np.float64),z[f"Y{k}"].astype(np.float64)) for k in range(NB)]
    baseW=z["head_weight"].astype(np.float32);baseb=z["head_bias"].astype(np.float32)
    spectra=[]
    for k,(F,Y) in enumerate(batches):
        X=F-F.mean(0);U,S,Vh=np.linalg.svd(X,full_matrices=False);Yc=Y-Y.mean(0);coef=U.T@Yc
        small=S<1e-2*S.max()
        spectra.append({"batch":k,"condition_number":float(S.max()/(S.min()+1e-12)),
                        "effective_rank_1e-3":int(np.sum(S>1e-3*S.max())),"effective_rank_1e-2":int(np.sum(S>1e-2*S.max())),
                        "target_energy_fraction_small_sv_lt_1e-2":[float(np.sum(coef[small,j]**2)/(np.sum(coef[:,j]**2)+1e-12)) for j in range(4)],
                        "singular_values":S.tolist()})
    ridge={}
    for l2 in RIDGES:
        rows=[]
        for k in range(NB-1):
            F,Y=batches[k];Fn,Yn=batches[k+1]
            A=np.c_[F,np.ones(len(F))];An=np.c_[Fn,np.ones(len(Fn))]
            sol=solve_ridge(A,Y,l2);soln=solve_ridge(An,Yn,l2)
            rows.append({"pair":f"{k}->{k+1}","self":metrics(Y,A@sol),"prior_on_next":metrics(Yn,An@sol),
                         "next_opt":metrics(Yn,An@soln),"solution_norm":float(np.linalg.norm(sol[:-1])),
                         "solution_drift":float(np.linalg.norm(soln-sol))})
        ridge[str(l2)]=rows
    seq={}
    for l2 in SEQ_L2:
        W=torch.nn.Parameter(torch.tensor(baseW));b=torch.nn.Parameter(torch.tensor(baseb));opt=torch.optim.Adam([W,b],lr=1e-4)
        trace=[]
        for k,(F,Y) in enumerate(batches):
            Ft=torch.tensor(F,dtype=torch.float32);Yt=torch.tensor(Y,dtype=torch.float32)
            with torch.no_grad():pb=Ft@W.T+b
            before=metrics(Y,pb.numpy())
            pred=Ft@W.T+b;loss=((pred-Yt)**2).mean()+float(l2)*(W**2).mean()
            opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():pa=Ft@W.T+b
            after=metrics(Y,pa.numpy());nextm=None
            if k+1<NB:
                Fn,Yn=batches[k+1];Pnext=torch.tensor(Fn,dtype=torch.float32)@W.T+b;nextm=metrics(Yn,Pnext.detach().numpy())
            trace.append({"batch":k,"loss":float(loss.detach()),"current_before":before,"current_after":after,"next_after":nextm,
                          "weight_norm":float(W.norm()),"bias_norm":float(b.norm())})
        all_ev=[];all_mse=[]
        with torch.no_grad():
            for F,Y in batches:
                P=torch.tensor(F,dtype=torch.float32)@W.T+b;m=metrics(Y,P.numpy());all_ev+=m["ev"];all_mse+=m["mse"]
        seq[str(l2)]={"trace":trace,"final_all_batches_ev_mean":float(np.mean(all_ev)),"final_all_batches_mse_mean":float(np.mean(all_mse)),
                      "final_weight_norm":float(W.norm())}
    white={}
    for rf in (1e-6,1e-4,1e-3,1e-2):
        rows=[]
        for k in range(NB-1):
            F,Y=batches[k];Fn,Yn=batches[k+1]
            selfm,Pnext,cond=whiten_fit(F,Y,Fn,rf)
            rows.append({"pair":f"{k}->{k+1}","self":selfm,"prior_on_next":metrics(Yn,Pnext),"whitened_condition_number":cond})
        white[str(rf)]=rows
    report["specialists"][lab]={"spectrum":spectra,"ridge":ridge,"sequential_head_l2":seq,"whitening":white}
agg={"ridge":{},"sequential_head_l2":{},"whitening":{}}
for l2 in RIDGES:
    se=[];ne=[];dr=[];norm=[]
    for lab in ORDER:
        for r in report["specialists"][lab]["ridge"][str(l2)]:
            se+=r["self"]["ev"];ne+=r["prior_on_next"]["ev"];dr.append(r["solution_drift"]);norm.append(r["solution_norm"])
    agg["ridge"][str(l2)]={"self_ev_mean":float(np.mean(se)),"next_ev_mean":float(np.mean(ne)),"next_ev_negative_fraction":float(np.mean(np.array(ne)<0)),
                           "solution_drift_mean":float(np.mean(dr)),"solution_norm_mean":float(np.mean(norm))}
for l2 in SEQ_L2:
    cur=[];nxt=[];final=[];norm=[]
    for lab in ORDER:
        q=report["specialists"][lab]["sequential_head_l2"][str(l2)];final.append(q["final_all_batches_ev_mean"]);norm.append(q["final_weight_norm"])
        for tr in q["trace"]:
            cur+=tr["current_after"]["ev"]
            if tr["next_after"] is not None:nxt+=tr["next_after"]["ev"]
    agg["sequential_head_l2"][str(l2)]={"current_after_ev_mean":float(np.mean(cur)),"next_after_ev_mean":float(np.mean(nxt)),
                                        "next_after_ev_negative_fraction":float(np.mean(np.array(nxt)<0)),
                                        "final_all_batches_ev_mean":float(np.mean(final)),"final_weight_norm_mean":float(np.mean(norm))}
for rf in (1e-6,1e-4,1e-3,1e-2):
    se=[];ne=[];co=[]
    for lab in ORDER:
        for r in report["specialists"][lab]["whitening"][str(rf)]:
            se+=r["self"]["ev"];ne+=r["prior_on_next"]["ev"];co.append(r["whitened_condition_number"])
    agg["whitening"][str(rf)]={"self_ev_mean":float(np.mean(se)),"next_ev_mean":float(np.mean(ne)),"next_ev_negative_fraction":float(np.mean(np.array(ne)<0)),
                               "condition_number_median":float(np.median(co))}
conds=[];rank=[];small=[[],[],[],[]]
for lab in ORDER:
    for s in report["specialists"][lab]["spectrum"]:
        conds.append(s["condition_number"]);rank.append(s["effective_rank_1e-2"])
        for j in range(4):small[j].append(s["target_energy_fraction_small_sv_lt_1e-2"][j])
agg["spectrum"]={"condition_number_median":float(np.median(conds)),"condition_number_max":float(np.max(conds)),
                 "effective_rank_1e-2_mean":float(np.mean(rank)),"small_sv_target_energy_fraction_by_head":[float(np.mean(x)) for x in small]}
report["aggregate"]=agg
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(agg,indent=2))
