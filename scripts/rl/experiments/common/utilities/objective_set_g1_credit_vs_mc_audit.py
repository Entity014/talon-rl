#!/usr/bin/env python3
from __future__ import annotations

# scripts/ on sys.path so the absolute rl.experiments.* imports below
# resolve when this file is run directly, as these scripts always are.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pathlib import Path
import copy,itertools,json,sys
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]

from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic,canonical_tokens
from talon_rl.rewards.objectives import normalized_objective_vector
import rl.experiments.common.utilities.objective_set_g1_train as g1
import rl.experiments.common.utilities.objective_set_g1_c0_critic_substrate as c0
import rl.experiments.common.utilities.authority_isolated_h2a_u30_semantic_validity as h2

INIT=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
OUT=ROOT/"runs/objective_set_g1_credit_vs_mc_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
ORDER=("T","A","O","S");H=64;NENV=8;GAMMA=.99;LAM=.95;EPS=1e-12
SEED=73101

def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)

def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}

def actor_named(model):
    return [(n,p) for n,p in model.named_parameters()
            if n.startswith("actor_") or n.startswith("family_") or n=="log_std"]

def flatten_grads(params,grads):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])

def cosine(a,b):
    return float((a@b/(a.norm()*b.norm()+EPS)).detach().cpu())

def mc_returns(r,d):
    out=torch.zeros_like(r);run=torch.zeros_like(r[0])
    for t in range(H-1,-1,-1):
        run=r[t]+GAMMA*run*(~d[t]).to(r.dtype).unsqueeze(-1)
        out[t]=run
    return out

def gae64(r,v,nv,d):
    out=torch.zeros_like(r);last=torch.zeros_like(nv)
    for t in range(H-1,-1,-1):
        boot=nv if t==H-1 else v[t+1]
        nt=(~d[t]).to(r.dtype).unsqueeze(-1)
        delta=r[t]+GAMMA*boot*nt-v[t]
        last=delta+GAMMA*LAM*nt*last
        out[t]=last
    return out
def make_batch(env,mgr,model,fold,card,batch_index):
    rng=np.random.default_rng(SEED*100003+card*7919+batch_index*101)
    combos=list(itertools.combinations(range(4),card));ids=[];ws=[]
    for lane in range(NENV):
        sub=np.asarray(combos[int(rng.integers(len(combos)))],np.int64);ids.append(sub)
        q=float(rng.random())
        if q<.20:w=np.full(card,1/card,np.float32)
        elif q<.60:
            h=int(rng.integers(card));w=np.full(card,.30/(card-1),np.float32);w[h]=.70
        else:w=rng.dirichlet(np.ones(card)).astype(np.float32)
        ws.append(w)
    ids=torch.tensor(np.stack(ids),device="cuda",dtype=torch.long)
    w=torch.tensor(np.stack(ws),device="cuda",dtype=torch.float32)
    tok=canonical_tokens(device="cuda")[ids]
    assert card in g1.FOLDS[fold]["train"] and card!=g1.FOLDS[fold]["holdout"]
    obs,_=env.reset(seed=SEED+700000+card*100+batch_index);obs=ot(obs).cuda()
    O=[];U=[];LP=[];R=[];D=[]
    with torch.no_grad():
        for _ in range(H):
            a,lp,u=model.act_with_set_latent(obs,tok,w)
            nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            full=torch.tensor(normalized_objective_vector(terms(raw,names),shape=(NENV,)),device="cuda")*env.unwrapped.step_dt
            O.append(obs);U.append(u);LP.append(lp);R.append(torch.gather(full,1,ids));D.append((te|tr).cuda())
            obs=ot(nxt).cuda()
    return {"card":card,"ids":ids,"tok":tok,"w":w,"obs":torch.stack(O),"u":torch.stack(U),
            "oldlp":torch.stack(LP),"r":torch.stack(R),"d":torch.stack(D).bool(),"next_obs":obs}

def credit_for_batch(model,b):
    card=b["card"];B=H*NENV
    obs=b["obs"].reshape(B,-1);tok=b["tok"].repeat(H,1,1);w=b["w"].repeat(H,1)
    with torch.no_grad():
        v=model.query_values_from_set(obs,tok,w,tok).reshape(H,NENV,card)
        nv=model.query_values_from_set(b["next_obs"],b["tok"],b["w"],b["tok"])
    return gae64(b["r"],v,nv,b["d"]),mc_returns(b["r"],b["d"])

def ppo_loss_from_adv(model,b,adv,mixed=True,obj=None):
    card=b["card"];B=H*NENV
    obs=b["obs"].reshape(B,-1);tok=b["tok"].repeat(H,1,1);w=b["w"].repeat(H,1)
    lp=model.logp_from_pre_tanh_from_set(obs,tok,w,b["u"].reshape(B,-1))
    ratio=torch.exp(lp-b["oldlp"].reshape(B).detach())
    cl=ratio.clamp(1-g1.CLIP,1+g1.CLIP)
    A=adv.reshape(B,card).detach()
    po=torch.minimum(ratio[:,None]*A,cl[:,None]*A)
    if mixed:return -(card*(w*po).sum(-1)).mean()
    ids=b["ids"].repeat(H,1)
    mask=(ids==obj)
    if not bool(mask.any()):return None
    return -po[mask].mean()

def gradients(model,batches,credits,arm,obj=None):
    named=actor_named(model);params=[p for _,p in named];losses=[]
    for b,c in zip(batches,credits):
        adv=c[arm]
        loss=ppo_loss_from_adv(model,b,adv,mixed=obj is None,obj=obj)
        if loss is not None:losses.append(loss)
    if not losses:return None,None
    loss=sum(losses)/len(losses)
    gs=torch.autograd.grad(loss,params,allow_unused=True)
    return flatten_grads(params,gs),float(loss.detach().cpu())
def virtual_model(base,flat_grad):
    m=copy.deepcopy(base)
    named=actor_named(m);theta=torch.cat([p.detach().reshape(-1) for _,p in named])
    eta=1e-4*float(theta.norm().cpu())
    gnorm=float(flat_grad.norm().cpu())
    step=(-eta/(gnorm+EPS))*flat_grad
    off=0
    with torch.no_grad():
        for _,p in named:
            n=p.numel();p.add_(step[off:off+n].view_as(p));off+=n
    m.eval()
    return m,eta

def functional_effect(base,mc,cr,probe):
    prefs={"C":[.25]*4,"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7]}
    rows={}
    for lab,wv in prefs.items():
        w=torch.tensor(wv,device=probe.device,dtype=torch.float32).repeat(len(probe),1)
        tok=canonical_tokens(device=probe.device).unsqueeze(0).repeat(len(probe),1,1)
        with torch.no_grad():
            mb=base.pre_tanh_mean_from_set(probe,tok,w);ab=base.act_inference_from_set(probe,tok,w)
            mm=mc.pre_tanh_mean_from_set(probe,tok,w);am=mc.act_inference_from_set(probe,tok,w)
            mc2=cr.pre_tanh_mean_from_set(probe,tok,w);ac=cr.act_inference_from_set(probe,tok,w)
        dm=(mm-mb).reshape(-1);dc=(mc2-mb).reshape(-1);dam=(am-ab).reshape(-1);dac=(ac-ab).reshape(-1)
        rows[lab]={
            "mc_pre_rms":float(torch.sqrt((dm*dm).mean()).cpu()),"mc_pre_max":float(dm.abs().max().cpu()),
            "critic_pre_rms":float(torch.sqrt((dc*dc).mean()).cpu()),"critic_pre_max":float(dc.abs().max().cpu()),
            "mc_action_rms":float(torch.sqrt((dam*dam).mean()).cpu()),"mc_action_max":float(dam.abs().max().cpu()),
            "critic_action_rms":float(torch.sqrt((dac*dac).mean()).cpu()),"critic_action_max":float(dac.abs().max().cpu()),
            "pre_delta_cosine":cosine(dm,dc),"action_delta_cosine":cosine(dam,dac)}
    return rows
def semantic_margins(env,mgr,robot,model):
    rows=[]
    for suite in range(4):
        seed=840001+suite
        for lab in ("T","A","O","S","C"):
            q=h2.evaluate(env,model,mgr,robot,h2.PREFS[lab],seed)
            q.update({"suite":suite,"label":lab});rows.append(q)
    out={}
    for lab in h2.ORDER:
        j=h2.IDX[lab];pk=h2.PHYS[lab];nm=[];pm=[]
        for suite in range(4):
            r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
            c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
            nm.append(r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j])
            pm.append(c["physical"][pk]-r["physical"][pk])
        out[lab]={"normalized_margin":float(np.mean(nm)),"physical_margin":float(np.mean(pm))}
    return out

def semantic_delta(base,new):
    out={}
    for lab in h2.ORDER:
        bn=base[lab]["normalized_margin"];bp=base[lab]["physical_margin"]
        dn=new[lab]["normalized_margin"]-bn;dp=new[lab]["physical_margin"]-bp
        tn=max(1e-6,.05*abs(bn));tp=max(1e-6,.05*abs(bp))
        preserve=(dn>=-tn and dp>=-tp and (dn>tn or dp>tp))
        degrade=(dn<-tn or dp<-tp)
        out[lab]={"delta_normalized":dn,"delta_physical":dp,"deadband_normalized":tn,"deadband_physical":tp,
                  "preserve_or_improve":bool(preserve),"degrade":bool(degrade)}
    return out
def run_fold(env,mgr,robot,fold):
    base=ObjectiveSetAuthorityIsolatedWideCritic(48,12).cuda()
    base.load_state_dict(torch.load(INIT,map_location="cuda",weights_only=False)["model"]);base.eval()
    # Fit only critic value basis using validated allowed-cardinality support.
    c0.fit_fold(env,mgr,fold,base)
    batches=[];credits=[]
    for bi,card in enumerate(g1.FOLDS[fold]["train"]):
        b=make_batch(env,mgr,base,fold,card,bi);batches.append(b)
        gae,mc=credit_for_batch(base,b);credits.append({"critic":gae,"mc":mc})
    gc,lc=gradients(base,batches,credits,"critic");gm,lm=gradients(base,batches,credits,"mc")
    mixed={"cosine":cosine(gc,gm),"critic_norm":float(gc.norm().cpu()),"mc_norm":float(gm.norm().cpu()),
           "relative_difference":float(((gc-gm).norm()/(gm.norm()+EPS)).cpu()),"critic_loss":lc,"mc_loss":lm}
    per={}
    for obj in range(4):
        a,_=gradients(base,batches,credits,"critic",obj);b,_=gradients(base,batches,credits,"mc",obj)
        if a is None or b is None:continue
        per[ORDER[obj]]={"critic_mc_cosine":cosine(a,b),"critic_norm":float(a.norm().cpu()),"mc_norm":float(b.norm().cpu()),
                         "critic_to_mixed_cosine":cosine(a,gc),"mc_to_mixed_cosine":cosine(b,gm)}
    vm,eta=virtual_model(base,gm);vc,_=virtual_model(base,gc)
    probe=torch.tensor(np.load(PROBE)["obs"],device="cuda",dtype=torch.float32)
    func=functional_effect(base,vm,vc,probe)
    sb=semantic_margins(env,mgr,robot,base);sm=semantic_margins(env,mgr,robot,vm);sc=semantic_margins(env,mgr,robot,vc)
    dm=semantic_delta(sb,sm);dc=semantic_delta(sb,sc)
    mc_pres=sum(dm[x]["preserve_or_improve"] for x in ("T","A","O"));mc_deg=sum(dm[x]["degrade"] for x in ("T","A","O"))
    cr_pres=sum(dc[x]["preserve_or_improve"] for x in ("T","A","O"));cr_deg=sum(dc[x]["degrade"] for x in ("T","A","O"))
    return {"mixed_gradient":mixed,"per_objective":per,"virtual_step_norm":eta,"functional_effect":func,
            "semantic":{"base":sb,"mc":sm,"critic":sc,"mc_delta":dm,"critic_delta":dc,
                        "mc_preserve_count_TAO":mc_pres,"mc_degrade_count_TAO":mc_deg,
                        "critic_preserve_count_TAO":cr_pres,"critic_degrade_count_TAO":cr_deg}}

def main():
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]]
    app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=SEED
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        o,_=env.reset(seed=SEED);mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        folds={}
        for fold in ("G1-2","G1-3"):
            folds[fold]=run_fold(env,mgr,robot,fold)
            print(fold,json.dumps({"mixed":folds[fold]["mixed_gradient"],"semantic":{k:folds[fold]["semantic"][k] for k in
                ("mc_preserve_count_TAO","mc_degrade_count_TAO","critic_preserve_count_TAO","critic_degrade_count_TAO")}},indent=2),flush=True)
        c=[folds[f]["mixed_gradient"]["cosine"] for f in folds]
        critic_blocker=all(folds[f]["mixed_gradient"]["cosine"]<.5 and folds[f]["semantic"]["mc_preserve_count_TAO"]>=2 and folds[f]["semantic"]["critic_degrade_count_TAO"]>=2 for f in folds)
        geometry=all(folds[f]["mixed_gradient"]["cosine"]>=.8 and folds[f]["semantic"]["mc_degrade_count_TAO"]>=2 and folds[f]["semantic"]["critic_degrade_count_TAO"]>=2 for f in folds)
        both_degrade=all(folds[f]["semantic"]["mc_degrade_count_TAO"]>=2 and folds[f]["semantic"]["critic_degrade_count_TAO"]>=2 for f in folds)
        if critic_blocker:verdict="CRITIC_DERIVED_CREDIT_PROXIMATE_BLOCKER"
        elif geometry:verdict="ACTIVE_SET_PPO_GEOMETRY_BLOCKER"
        elif both_degrade:verdict="CRITIC_NOT_SUFFICIENT_EXPLANATION"
        else:verdict="INCONCLUSIVE_ESCALATE_TRAJECTORY_CREDIT"
        rep={"schema":"objective_set_g1_credit_vs_mc_audit_v1","folds":folds,"decision":{"verdict":verdict,"g1_retraining_authorized":False}}
        (OUT/"credit_vs_mc_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
        print("FINAL",json.dumps(rep["decision"]),flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
