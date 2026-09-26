#!/usr/bin/env python3
from __future__ import annotations

# scripts/ on sys.path so the absolute rl.experiments.* imports below
# resolve when this file is run directly, as these scripts always are.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pathlib import Path
import itertools,json,sys,copy
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]

import rl.experiments.common.utilities.objective_set_g1_credit_vs_mc_audit as cvm
import rl.experiments.common.utilities.objective_set_g1_c0_critic_substrate as c0
import rl.experiments.common.utilities.objective_set_g1_train as g1
from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic,canonical_tokens
from talon_rl.rewards.objectives import normalized_objective_vector

INIT=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
OUT=ROOT/"runs/objective_set_g1_trajectory_semantic_credit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
ORDER=("T","A","O","S");PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
HMAX=64;HS=(8,16,32,64);SUITES=4;SEED=73101

def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)

def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}

def tilt_deg(q):
    _,x,y,_=[q[:,i] for i in range(4)]
    return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))

def prefs(m):
    center=np.full(m,1/m,np.float32);heavy={}
    for h in range(m):
        w=np.full(m,.30/(m-1),np.float32);w[h]=.70;heavy[h]=w
    return center,heavy

def virtual_from_grad(base,g):
    return cvm.virtual_model(base,g)[0]
def build_directions(env,mgr,fold):
    base=ObjectiveSetAuthorityIsolatedWideCritic(48,12).cuda()
    base.load_state_dict(torch.load(INIT,map_location="cuda",weights_only=False)["model"]);base.eval()
    c0.fit_fold(env,mgr,fold,base)
    batches=[];credits=[]
    for bi,card in enumerate(g1.FOLDS[fold]["train"]):
        b=cvm.make_batch(env,mgr,base,fold,card,bi);batches.append(b)
        gae,mc=cvm.credit_for_batch(base,b);credits.append({"critic":gae,"mc":mc})
    gm,_=cvm.gradients(base,batches,credits,"mc")
    gc,_=cvm.gradients(base,batches,credits,"critic")
    dirs={"mixed_mc":gm,"mixed_critic":gc}
    for obj in range(4):
        q,_=cvm.gradients(base,batches,credits,"mc",obj)
        if q is not None:dirs[ORDER[obj]]=q
    models={"base":base}
    for k,g in dirs.items():models[k]=virtual_from_grad(base,g)
    return base,dirs,models

def rollout(env,mgr,robot,model,ids,wv,seed):
    m=len(ids);dev=torch.device("cuda")
    tok=canonical_tokens(device=dev)[torch.tensor(ids,device=dev)].unsqueeze(0).repeat(g1.NENV,1,1)
    w=torch.tensor(wv,device=dev,dtype=torch.float32).unsqueeze(0).repeat(g1.NENV,1)
    obs,_=env.reset(seed=seed);obs=ot(obs).cuda();prev=torch.zeros((g1.NENV,env.unwrapped.action_manager.total_action_dim),device=dev)
    R=[];P=[];D=[]
    with torch.no_grad():
        for _ in range(HMAX):
            a=model.act_inference_from_set(obs,tok,w);nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            full=normalized_objective_vector(terms(raw,names),shape=(g1.NENV,))*env.unwrapped.step_dt
            data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            P.append({
                "tracking_error":float(((data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()).mean()),
                "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
            R.append(full[:,list(ids)].mean(0));D.append((te|tr).cpu().numpy());prev=a;obs=ot(nxt).cuda()
    R=np.asarray(R,float);D=np.asarray(D,bool)
    out={}
    for h in HS:
        out[h]={"return":R[:h].sum(0).tolist(),
                "physical":{k:float(np.mean([x[k] for x in P[:h]])) for k in P[0]},
                "survival":float(1-D[:h].any(0).mean())}
    return out
def set_metrics(env,mgr,robot,models,ids,set_index):
    m=len(ids);labs=[ORDER[i] for i in ids];cw,hw=prefs(m)
    raw={name:{lab:[] for lab in ["C"]+labs} for name in models}
    for name,model in models.items():
        for suite in range(SUITES):
            seed=960001+set_index*100+suite
            raw[name]["C"].append(rollout(env,mgr,robot,model,ids,cw,seed))
            for h,lab in enumerate(labs):
                raw[name][lab].append(rollout(env,mgr,robot,model,ids,hw[h],seed))
    margins={}
    for name in models:
        margins[name]={}
        for lab in labs:
            j=labs.index(lab);pk=PHYS[lab];margins[name][lab]={}
            for H in HS:
                nr=[];pr=[];sv=[]
                for suite in range(SUITES):
                    c=raw[name]["C"][suite][H];r=raw[name][lab][suite][H]
                    nr.append(r["return"][j]-c["return"][j])
                    pr.append(c["physical"][pk]-r["physical"][pk])
                    sv.append(min(c["survival"],r["survival"]))
                margins[name][lab][H]={"normalized":float(np.mean(nr)),"physical":float(np.mean(pr)),"min_survival":float(min(sv))}
    return {"ids":list(ids),"labels":labs,"margins":margins}

def aggregate(fold_sets,model_name,obj,H):
    vals=[]
    for s in fold_sets:
        if obj not in s["labels"]:continue
        b=s["margins"]["base"][obj][H];v=s["margins"][model_name][obj][H]
        vals.append({"dn":v["normalized"]-b["normalized"],"dp":v["physical"]-b["physical"],
                     "bn":b["normalized"],"bp":b["physical"],"survival":v["min_survival"]})
    if not vals:return None
    dn=float(np.mean([x["dn"] for x in vals]));dp=float(np.mean([x["dp"] for x in vals]))
    bn=float(np.mean([x["bn"] for x in vals]));bp=float(np.mean([x["bp"] for x in vals]))
    tn=max(1e-6,.05*abs(bn));tp=max(1e-6,.05*abs(bp))
    cls="improve" if dn>tn and dp>=-tp else ("degrade" if dn<-tn or dp<-tp else "neutral")
    return {"delta_normalized":dn,"delta_physical":dp,"base_normalized":bn,"base_physical":bp,
            "deadband_normalized":tn,"deadband_physical":tp,"class":cls,
            "min_survival":float(min(x["survival"] for x in vals)),"n_sets":len(vals)}
def run_fold(env,mgr,robot,fold):
    base,dirs,models=build_directions(env,mgr,fold)
    sets=[];idx=0
    for m in g1.FOLDS[fold]["train"]:
        for ids in itertools.combinations(range(4),m):
            sets.append(set_metrics(env,mgr,robot,models,ids,idx))
            print("SET",fold,ids,"done",flush=True);idx+=1
    matrix={}
    for H in HS:
        matrix[str(H)]={}
        for obj in ORDER:
            matrix[str(H)][obj]={}
            for d in ("T","A","O","S","mixed_mc","mixed_critic"):
                if d not in models:continue
                q=aggregate(sets,d,obj,H)
                if q is not None:matrix[str(H)][obj][d]=q
    primary={}
    for H in HS:
        t=matrix[str(H)]["T"]
        primary[str(H)]={k:t.get(k) for k in ("T","mixed_mc","mixed_critic")}
    return {"sets":sets,"sacrifice_matrix":matrix,"primary_T":primary}

def classify(folds):
    def q(f,H,d):return folds[f]["primary_T"][str(H)][d]
    # Evaluate H32/H64 for stable cross-fold patterns.
    late=(32,64)
    A=True
    for f in folds:
        ok_any=False
        for H in late:
            pt=q(f,H,"T");mx=q(f,H,"mixed_mc")
            if pt and mx and pt["class"]!="degrade" and mx["class"]=="degrade" and mx["delta_normalized"]<pt["delta_normalized"]:
                ok_any=True
        A &= ok_any
    B=True
    for f in folds:
        ok_any=False
        for H in late:
            pt=q(f,H,"T")
            if pt and pt["delta_normalized"]>pt["deadband_normalized"] and pt["delta_physical"]<-pt["deadband_physical"]:
                ok_any=True
        B &= ok_any
    C=True
    for f in folds:
        ok_any=False
        for H in late:
            mc=q(f,H,"mixed_mc");cr=q(f,H,"mixed_critic")
            if mc and cr and cr["delta_normalized"]<mc["delta_normalized"] and cr["delta_physical"]<mc["delta_physical"] and mc["class"]=="degrade":
                ok_any=True
        C &= ok_any
    # Stability: same primary T class over H32/H64 for each direction/fold.
    stable=True
    for f in folds:
        for d in ("T","mixed_mc","mixed_critic"):
            cs=[q(f,H,d)["class"] for H in late if q(f,H,d)]
            stable &= len(set(cs))<=1
    if A:verdict="A_MULTI_OBJECTIVE_INTERACTION"
    elif B:verdict="B_RETURN_SEMANTIC_MISMATCH"
    elif C:verdict="C_CRITIC_PLUS_INTERACTION"
    else:verdict="D_NO_COMPACT_GRADIENT_EXPLANATION"
    if not stable and verdict!="D_NO_COMPACT_GRADIENT_EXPLANATION":
        verdict="D_NO_COMPACT_GRADIENT_EXPLANATION"
    return {"verdict":verdict,"A":bool(A),"B":bool(B),"C":bool(C),"late_horizon_stable":bool(stable),
            "g1_retraining_authorized":False}

def main():
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=g1.NENV;cfg.seed=SEED
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=SEED)
        mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        folds={}
        for fold in ("G1-2","G1-3"):
            folds[fold]=run_fold(env,mgr,robot,fold)
            print("FOLD",fold,"PRIMARY",json.dumps(folds[fold]["primary_T"],indent=2),flush=True)
        decision=classify(folds)
        rep={"schema":"objective_set_g1_trajectory_semantic_credit_v1","folds":folds,"decision":decision}
        (OUT/"trajectory_semantic_credit.json").write_text(json.dumps(rep,indent=2)+"\n")
        print("FINAL",json.dumps(decision),flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
