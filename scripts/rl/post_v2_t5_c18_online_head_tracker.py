#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,traceback
from pathlib import Path
from collections import deque
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ORDER=("T","A","O","S");SNAPS=(0,1,5,10,25)
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
GAMMA=.99
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def flat_grad(gs,params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,params)])
def cos(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
def corrmat(x):return np.corrcoef(np.asarray(x,float),rowvar=False).tolist()
def trunc_mc(rt,dt):
    out=torch.zeros_like(rt);running=torch.zeros_like(rt[-1])
    for t in range(len(rt)-1,-1,-1):
        running=rt[t]+GAMMA*running*(~dt[t]).unsqueeze(-1);out[t]=running
    return out
def ev_np(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def ridge_fit(F,Y,l2=1.0):
    # float64 CPU for stable solve; no regularization on bias
    Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
    A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
    sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
    return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--tracker",choices=("adam1","ridge3"),required=True)
    ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
    ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
    ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--critic-lr",type=float,default=1e-4)
    ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--recent-window",type=int,default=3)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    env=None
    try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
      from talon_rl.t3b_objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      all_logs={};snap_paths={}
      for idx,label in enumerate(ORDER):
        torch.manual_seed(41000+idx);np.random.seed(41000+idx)
        m=T4SharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
        # freeze body; online repair changes head tracker only
        for n,p in m.named_parameters():
            if n.startswith("critic_body"):p.requires_grad_(False)
        actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
        head_params=[m.critic_head.weight,m.critic_head.bias]
        actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
        head_opt=torch.optim.Adam(head_params,lr=args.critic_lr)
        recent=deque(maxlen=args.recent_window)
        w=torch.as_tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
        cur,_=env.reset(seed=410001+idx*1000);cur=obs_tensor(cur).cuda()
        logs=[]
        def save_snap(tag):
            p=args.output.parent/f"{label}_snap_{tag}.pt"
            torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"tracker":args.tracker},p);snap_paths[f"{label}:{tag}"]=str(p)
        save_snap(0)
        for update in range(1,args.updates+1):
            ob=[];pre=[];old=[];rw=[];dn=[];surv=np.ones(args.num_envs,bool)
            for _ in range(args.horizon):
                with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
                nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((term|trunc).cuda())
                surv &= ~(term|trunc).cpu().numpy();cur=obs_tensor(nxt).cuda()
            fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1)
            rt=torch.stack(rw);dt=torch.stack(dn).bool();target_ret=trunc_mc(rt,dt).reshape(-1,4).detach()
            # frozen body features for explicit head tracker
            with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
            recent.append((F.detach().cpu(),target_ret.detach().cpu()))
            # pre-refresh value quality
            with torch.no_grad():vpre=m.critic_head(F);pre_ev=[ev_np(target_ret[:,j].cpu(),vpre[:,j].cpu()) for j in range(4)]
            # controlled head update
            if args.tracker=="adam1":
                head_opt.zero_grad(set_to_none=True);pred=m.critic_head(F);loss=((pred-target_ret)**2).mean();loss.backward();head_opt.step()
            else:
                FF=torch.cat([x[0] for x in recent],0).cuda();YY=torch.cat([x[1] for x in recent],0).cuda()
                Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit)
            with torch.no_grad():
                vpost=m.critic_head(F);post_ev=[ev_np(target_ret[:,j].cpu(),vpost[:,j].cpu()) for j in range(4)]
                post_bias=[float((vpost[:,j]-target_ret[:,j]).mean()) for j in range(4)]
            # actor advantage uses refreshed value head; body remains frozen
            with torch.no_grad():
                vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4)
                nv=m.value_with_preference(cur,w)
                adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
            logp=m.logp_from_pre_tanh_with_preference(fo,fw,fu);ratio=torch.exp(logp-fold.detach())
            ratio_maxerr=float((ratio-1).abs().max())
            if ratio_maxerr>1e-4 or not torch.isfinite(ratio).all():raise RuntimeError(f"ratio invariant failed {ratio_maxerr}")
            advflat=adv.reshape(-1,4).detach()
            # semantic actor gradient geometry BEFORE actor step
            gobj=[]
            for j in range(4):
                lj=-(ratio*advflat[:,j]).mean()
                gobj.append(flat_grad(torch.autograd.grad(lj,actor_params,retain_graph=True,allow_unused=True),actor_params).detach())
            al=scalarized_late_weighted_ppo(ratio,advflat,fw)
            actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
            snap_tag=0 if update==1 else update-1
            if snap_tag in SNAPS:
                logs.append({
                  "snapshot":snap_tag,"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"post_bias":post_bias,
                  "adv_mean":advflat.mean(0).cpu().tolist(),"adv_std":advflat.std(0).cpu().tolist(),"adv_corr":corrmat(advflat.cpu().numpy()),
                  "objective_grad_norm":[float(g.norm()) for g in gobj],
                  "objective_grad_cosine":[[cos(gobj[i],gobj[j]) for j in range(4)] for i in range(4)],
                  "survival":float(surv.mean()),"recent_buffer_len":len(recent),
                  "head_weight_norm":float(m.critic_head.weight.norm()),"head_bias_norm":float(m.critic_head.bias.norm())
                })
            if update in SNAPS:save_snap(update)
        all_logs[label]=logs
      report={"schema":"t5_c18_online_head_tracker_v1","status":"MEASUREMENT_COMPLETE","tracker":args.tracker,"critic_body_frozen":True,
              "horizon":args.horizon,"updates":args.updates,"env_steps":args.horizon*args.updates,"actor_lr":args.actor_lr,"critic_lr":args.critic_lr,
              "ridge_lambda":args.ridge_lambda,"recent_window":args.recent_window,"specialist_logs":all_logs,"snapshot_paths":snap_paths}
      args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"status":report["status"],"tracker":args.tracker},indent=2))
    except BaseException as e:
      args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
    finally:
      if env is not None:env.close()
      app.close()
if __name__=="__main__":main()
