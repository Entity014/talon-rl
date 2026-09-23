#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,hashlib,traceback
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ORDER=("T","A","O","S"); SNAPS=(0,1,5,10,25,50,100,200,300)
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def flat_grad(gs,params):
    out=[]
    for g,p in zip(gs,params):out.append((torch.zeros_like(p) if g is None else g).reshape(-1))
    return torch.cat(out)
def cos(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
def corrmat(x):
    x=np.asarray(x,float);return np.corrcoef(x,rowvar=False).tolist()
def param_vec(params):return torch.cat([p.detach().reshape(-1) for p in params])
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--updates",type=int,default=300);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=2);ap.add_argument("--eval-steps",type=int,default=32);ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"));ap.add_argument("--critic-head-init",choices=("scalar","zero"),default="scalar");ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--critic-lr",type=float,default=1e-3);args=ap.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    env=None
    try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
      from talon_rl.t3b_objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      all_logs={};snap_paths={}
      for idx,label in enumerate(ORDER):
        torch.manual_seed(31000+idx);np.random.seed(31000+idx)
        m=T4SharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init=args.critic_head_init)
        actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
        critic_params=[p for n,p in m.named_parameters() if n.startswith("critic_")]
        opt=torch.optim.Adam([{"params":actor_params,"lr":args.actor_lr},{"params":critic_params,"lr":args.critic_lr}])
        w=torch.as_tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda");cur,_=env.reset(seed=310001+idx*1000);cur=obs_tensor(cur).cuda()
        logs=[]
        def save_snap(tag):
            p=args.output.parent/f"{label}_snap_{tag}.pt";torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag},p);snap_paths[f"{label}:{tag}"]=str(p)
        save_snap(0)
        for update in range(1,args.updates+1):
            ob=[];ac=[];pre=[];old=[];rw=[];val=[];dn=[]
            for _ in range(args.horizon):
                with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
                nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                ob.append(cur);ac.append(a);pre.append(u);old.append(lp);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
            with torch.no_grad():nv=m.value_with_preference(cur,w)
            rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt)
            fo=torch.cat(ob);fa=torch.cat(ac);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1)
            logp=m.logp_from_pre_tanh_with_preference(fo,fw,fu);ratio=torch.exp(logp-fold.detach())
            ratio_maxerr=float((ratio-1).abs().max().detach())
            if ratio_maxerr>1e-6 or not torch.isfinite(ratio).all():
                raise RuntimeError(f"PPO pre-update ratio invariant failed: {ratio_maxerr}")
            al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw);cl=vector_value_loss(m.value_with_preference(fo,fw),ret.reshape(-1,4).detach());loss=al+cl
            snap_tag=0 if update==1 else update-1
            do_diag=snap_tag in SNAPS
            if do_diag:
                advflat=adv.reshape(-1,4).detach();gobj=[]
                for j in range(4):
                    lj=-(ratio*advflat[:,j]).mean()
                    gobj.append(flat_grad(torch.autograd.grad(lj,actor_params,retain_graph=True,allow_unused=True),actor_params).detach())
                gnorm=[float(g.norm()) for g in gobj];gcos=[[cos(gobj[i],gobj[j]) for j in range(4)] for i in range(4)]
                comb={}
                for plab in ORDER:
                    ww=PREFS[plab];gc=sum(float(4*ww[j])*gobj[j] for j in range(4));comb[plab]=gc
                ccos={a:{b:cos(comb[a],comb[b]) for b in ORDER} for a in ORDER}
                before=param_vec(actor_params).clone()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if do_diag:
                after=param_vec(actor_params);delta=after-before;actual=comb[label]
                logs.append({"snapshot":snap_tag,"adv_mean":advflat.mean(0).cpu().tolist(),"adv_std":advflat.std(0).cpu().tolist(),"adv_corr":corrmat(advflat.cpu().numpy()),"objective_grad_norm":gnorm,"objective_grad_cosine":gcos,"combined_grad_norm":{k:float(v.norm()) for k,v in comb.items()},"combined_grad_cosine":ccos,"actual_update_norm":float(delta.norm()),"actual_update_vs_negative_combined_grad_cosine":cos(delta,-actual)})
            if update in SNAPS:save_snap(update)
        all_logs[label]=logs
      # matched snapshot action divergence on common initial observations, no rollout confound
      action_diag=[]
      active_snaps=tuple(s for s in SNAPS if s<=args.updates)
      for snap in active_snaps:
        cur,_=env.reset(seed=340001);cur=obs_tensor(cur).cuda()
        models={}
        for lab in ORDER:
            m=T4SharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(args.output.parent/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
        ws={lab:torch.as_tensor(np.repeat(PREFS[lab][None,:],args.num_envs,axis=0),device="cuda") for lab in ORDER}
        ds={}; acts={}
        with torch.no_grad():
            for lab in ORDER:acts[lab]=models[lab].act_inference_with_preference(cur,ws[lab])
        for i,a in enumerate(ORDER):
            for b in ORDER[i+1:]:ds[f"{a}_{b}"]=float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean())
        action_diag.append({"snapshot":snap,"pair_action_distance":ds})
      report={"schema":"t5_gradient_separability_audit_v1","status":"MEASUREMENT_COMPLETE","instrumented_replay":True,"critic_head_init":args.critic_head_init,"actor_lr":args.actor_lr,"critic_lr":args.critic_lr,"snapshots":list(active_snaps),"preferences":{k:v.tolist() for k,v in PREFS.items()},"specialist_logs":all_logs,"action_divergence":action_diag,"snapshot_paths":snap_paths}
      args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"status":report["status"],"action_divergence":action_diag},indent=2))
    except BaseException as e:
      args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
    finally:
      if env is not None:env.close()
      app.close()
if __name__=="__main__":main()
