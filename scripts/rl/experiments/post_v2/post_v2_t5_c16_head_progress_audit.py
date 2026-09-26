#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,numpy as np

ROOT=Path(__file__).resolve().parents[4]
C14=ROOT/"runs/post_v2_t5_c14_representation_drift-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c16_head_progress-2026-09-23"
ORDER=("T","A","O","S"); SNAPS=(0,10,25); RIDGE=1.0; PCR_RANK=32

def cos(a,b):
    a=np.asarray(a).reshape(-1); b=np.asarray(b).reshape(-1)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def ridge_solution(F,Y,l2):
    A=np.c_[F,np.ones(len(F))]
    I=np.eye(A.shape[1]);I[-1,-1]=0
    sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
    return sol[:-1].T,sol[-1]

def pcr_solution(F,Y,rank):
    mu=F.mean(0,keepdims=True);X=F-mu
    U,S,Vt=np.linalg.svd(X,full_matrices=False);r=min(rank,len(S));Vr=Vt[:r].T
    Z=X@Vr;A=np.c_[Z,np.ones(len(Z))]
    sol=np.linalg.lstsq(A,Y,rcond=None)[0]
    Wred=sol[:-1].T
    b=sol[-1] - Wred@(mu.reshape(-1)@Vr)
    W=Wred@Vr.T
    return W,b,S

def displacement_metrics(W0,b0,Wt,bt,Wstar,bstar):
    d_on=np.r_[ (Wt-W0).reshape(-1), (bt-b0).reshape(-1) ]
    d_st=np.r_[ (Wstar-W0).reshape(-1), (bstar-b0).reshape(-1) ]
    n_on=np.linalg.norm(d_on);n_st=np.linalg.norm(d_st)
    c=cos(d_on,d_st)
    rho_norm=float(n_on/(n_st+1e-12))
    rho_par=float((d_on@d_st)/(d_st@d_st+1e-12))
    orth=np.linalg.norm(d_on-rho_par*d_st)
    return {
        "online_norm":float(n_on),"target_norm":float(n_st),"cosine":c,
        "norm_progress_ratio":rho_norm,"projected_progress_ratio":rho_par,
        "orthogonal_fraction_of_online":float(orth/(n_on+1e-12))
    }

def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

OUT.mkdir(parents=True,exist_ok=True)
report={"schema":"t5_c16_head_progress_v1","stable_refs":{"ridge_lambda":RIDGE,"pcr_rank":PCR_RANK},"specialists":{}}
for lab in ORDER:
    z=np.load(C14/f"{lab}_matched.npz")
    Y=z["Y"].astype(np.float64)
    F={s:z[f"F{s}"].astype(np.float64) for s in SNAPS}
    W={s:z[f"W{s}"].astype(np.float64) for s in SNAPS}
    b={s:z[f"b{s}"].astype(np.float64) for s in SNAPS}
    # freeze target references on u25 body + matched target; robust solution class only
    Wr,br=ridge_solution(F[25],Y,RIDGE)
    Wp,bp,S=pcr_solution(F[25],Y,PCR_RANK)
    labout={"ridge_ref":{},"pcr_ref":{},"actual_ev":{}}
    for t in (10,25):
        labout["ridge_ref"][str(t)]=displacement_metrics(W[0],b[0],W[t],b[t],Wr,br)
        labout["pcr_ref"][str(t)]=displacement_metrics(W[0],b[0],W[t],b[t],Wp,bp)
    for s in SNAPS:
        pred=F[s]@W[s].T+b[s]
        labout["actual_ev"][str(s)]=[ev(Y[:,j],pred[:,j]) for j in range(4)]
    # per-head same metrics
    ph={}
    for j in range(4):
        ph[str(j)]={"ridge":{},"pcr":{}}
        for t in (10,25):
            ph[str(j)]["ridge"][str(t)]=displacement_metrics(W[0][j:j+1],b[0][j:j+1],W[t][j:j+1],b[t][j:j+1],Wr[j:j+1],br[j:j+1])
            ph[str(j)]["pcr"][str(t)]=displacement_metrics(W[0][j:j+1],b[0][j:j+1],W[t][j:j+1],b[t][j:j+1],Wp[j:j+1],bp[j:j+1])
    labout["per_head"]=ph
    labout["reference_ev"]={
      "ridge_on_u25":[ev(Y[:,j],(F[25]@Wr.T+br)[:,j]) for j in range(4)],
      "pcr32_on_u25":[ev(Y[:,j],(F[25]@Wp.T+bp)[:,j]) for j in range(4)]
    }
    report["specialists"][lab]=labout

agg={"ridge":{},"pcr":{}}
for refkey,outkey in (("ridge_ref","ridge"),("pcr_ref","pcr")):
    for t in ("10","25"):
        vals=[report["specialists"][lab][refkey][t] for lab in ORDER]
        agg[outkey][t]={
          "cosine_mean":float(np.mean([x["cosine"] for x in vals])),
          "cosine_median":float(np.median([x["cosine"] for x in vals])),
          "cosine_negative_fraction":float(np.mean(np.array([x["cosine"] for x in vals])<0)),
          "norm_progress_ratio_mean":float(np.mean([x["norm_progress_ratio"] for x in vals])),
          "projected_progress_ratio_mean":float(np.mean([x["projected_progress_ratio"] for x in vals])),
          "projected_progress_ratio_median":float(np.median([x["projected_progress_ratio"] for x in vals])),
          "orthogonal_fraction_mean":float(np.mean([x["orthogonal_fraction_of_online"] for x in vals]))
        }
# per-head aggregate
for refname in ("ridge","pcr"):
    agg.setdefault("per_head",{}).setdefault(refname,{})
    for j in range(4):
      for t in ("10","25"):
        vals=[report["specialists"][lab]["per_head"][str(j)][refname][t] for lab in ORDER]
        agg["per_head"][refname].setdefault(str(j),{})[t]={
          "cosine_mean":float(np.mean([x["cosine"] for x in vals])),
          "projected_progress_ratio_mean":float(np.mean([x["projected_progress_ratio"] for x in vals])),
          "norm_progress_ratio_mean":float(np.mean([x["norm_progress_ratio"] for x in vals]))
        }
report["aggregate"]=agg
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(agg,indent=2))
