#!/usr/bin/env python3
from pathlib import Path
import itertools,json,sys,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic,canonical_tokens
from talon_rl.rewards.objectives import normalized_objective_vector
G0=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
OUT=ROOT/"runs/objective_set_g1_c0_critic_substrate-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
FOLDS={"G1-2":(3,4),"G1-3":(2,4)}
NENV=8;H=64;G=.99;LAM=1.0;ORDER=("T","A","O","S")

def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def trunc(R,D):
    out=torch.zeros_like(R);run=torch.zeros_like(R[0])
    for t in range(len(R)-1,-1,-1):
        run=R[t]+G*run*(~D[t]).to(R.dtype).unsqueeze(-1);out[t]=run
    return out
def pref_list(m):
    out=[np.full(m,1/m,np.float32)]
    for h in range(m):
        w=np.full(m,.30/(m-1),np.float32);w[h]=.70;out.append(w)
    return out
def ridge(F,Y,lam=1.0):
    X=np.c_[F,np.ones(len(F))];I=np.eye(X.shape[1]);I[-1,-1]=0
    B=np.linalg.solve(X.T@X+lam*I,X.T@Y)
    return B[:-1],B[-1]
def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def collect(env,mgr,model,ids,wv,seed):
    ids=np.asarray(ids,np.int64);m=len(ids);dev=torch.device("cuda")
    tok=canonical_tokens(device=dev)[torch.tensor(ids,device=dev)].unsqueeze(0).repeat(NENV,1,1)
    w=torch.tensor(wv,device=dev).repeat(NENV,1)
    obs,_=env.reset(seed=seed);obs=ot(obs).cuda();O=[];R=[];D=[]
    with torch.no_grad():
        for _ in range(H):
            O.append(obs)
            a=model.act_inference_from_set(obs,tok,w);nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            full=torch.tensor(normalized_objective_vector(terms(raw,names),shape=(NENV,)),device=dev)*env.unwrapped.step_dt
            R.append(full[:,ids]);D.append((te|tr).cuda());obs=ot(nxt).cuda()
        obs=torch.stack(O);R=torch.stack(R);D=torch.stack(D).bool();Y=trunc(R,D)
        F=model.critic_features_from_set(obs.reshape(-1,obs.shape[-1]),tok.repeat(H,1,1),w.repeat(H,1)).reshape(H,NENV,-1)
    return F.cpu().numpy(),Y.cpu().numpy()

def fit_fold(env,mgr,fold,model):
    feats={i:[] for i in range(4)};ys={i:[] for i in range(4)};counter=0
    for m in FOLDS[fold]:
        for ids in itertools.combinations(range(4),m):
            for pi,wv in enumerate(pref_list(m)):
                F,Y=collect(env,mgr,model,ids,wv,910001+counter);counter+=1
                # balanced early+late support, active tokens only
                ff=F.reshape(-1,F.shape[-1])
                yy=Y.reshape(-1,m)
                for j,obj in enumerate(ids):
                    feats[obj].append(ff);ys[obj].append(yy[:,j])
    with torch.no_grad():
        for obj in range(4):
            F=np.concatenate(feats[obj]);Y=np.concatenate(ys[obj])
            W,b=ridge(F,Y,LAM)
            model.value_basis_weight[obj].copy_(torch.tensor(W,device="cuda",dtype=model.value_basis_weight.dtype))
            model.value_basis_bias[obj].copy_(torch.tensor(b,device="cuda",dtype=model.value_basis_bias.dtype))
def validate_fold(env,mgr,fold,model):
    vals=[];counter=0
    for m in FOLDS[fold]:
        for ids in itertools.combinations(range(4),m):
            for pi,wv in enumerate(pref_list(m)):
                F,Y=collect(env,mgr,model,ids,wv,930001+counter);counter+=1
                ids_t=torch.tensor(ids,device="cuda")
                tok=canonical_tokens(device="cuda")[ids_t]
                # direct token-query prediction from collected feature bank
                ff=torch.tensor(F.reshape(-1,F.shape[-1]),device="cuda",dtype=torch.float32)
                qw=model.value_basis_weight[ids_t];qb=model.value_basis_bias[ids_t]
                P=(ff@qw.T+qb).detach().cpu().numpy().reshape(H,NENV,m)
                for j,obj in enumerate(ids):
                    vals.append({"cardinality":m,"set":"".join(ORDER[x] for x in ids),"objective":ORDER[obj],
                                 "ev":ev(Y[:,:,j],P[:,:,j]),"bias":float(np.mean(P[:,:,j]-Y[:,:,j]))})
    E=np.array([x["ev"] for x in vals])
    return vals,{"ev_mean":float(E.mean()),"negative_fraction":float(np.mean(E<0)),
                 "mean_abs_bias":float(np.mean(np.abs([x["bias"] for x in vals]))),
                 "pass":bool(E.mean()>0 and np.mean(E<0)<=.25)}

def main():
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
        base=torch.load(G0,map_location="cuda",weights_only=False)["model"];report={}
        for fold in FOLDS:
            m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();m.load_state_dict(base);m.eval()
            before={k:v.detach().clone() for k,v in m.state_dict().items() if k.startswith("actor_") or k.startswith("family_") or k=="log_std"}
            fit_fold(env,mgr,fold,m);vals,summ=validate_fold(env,mgr,fold,m)
            drift=max(float((m.state_dict()[k]-v).abs().max().cpu()) for k,v in before.items())
            torch.save({"model":m.state_dict(),"fold":fold},OUT/f"{fold.lower().replace('-','_')}_critic_refreshed.pt")
            report[fold]={"actor_max_drift":drift,"critic":summ,"details":vals}
            print(fold,"actor_drift",drift,"critic",summ,flush=True)
        decision={"C0_pass":bool(all(report[f]["actor_max_drift"]<=1e-6 and report[f]["critic"]["pass"] for f in report))}
        rep={"schema":"objective_set_g1_c0_critic_substrate_v1","folds":report,"decision":decision}
        (OUT/"critic_substrate_report.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",decision,flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
