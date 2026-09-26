#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,traceback
from pathlib import Path
from collections import deque
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ORDER=("T","A","O","S")
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
G=.99
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def trunc_mc(rt,dt):
    out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
    for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
    return out
def ev_np(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def ridge_fit(F,Y,l2):
    Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
    A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
    sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
    return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--actor-update",choices=("on","off"),required=True)
    ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
    ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
    ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--recent-window",type=int,default=3)
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
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      all_logs={}
      for idx,label in enumerate(ORDER):
        torch.manual_seed(51000+idx);np.random.seed(51000+idx)
        m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
        for n,p in m.named_parameters():
            if n.startswith("critic_body"):p.requires_grad_(False)
        actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
        actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
        w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
        recent=deque(maxlen=args.recent_window)
        cur,_=env.reset(seed=510001+idx*1000);cur=obs_tensor(cur).cuda()
        logs=[]
        def save_snap(tag):
            torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"actor_update":args.actor_update},
                       args.output.parent/f"{label}_snap_{tag}.pt")
        save_snap(0)
        prev_F=None;prev_Y=None;prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
        prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params])
        for update in range(1,args.updates+1):
            ob=[];pre=[];old=[];rw=[];dn=[];acts=[]
            for _ in range(args.horizon):
                with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());acts.append(a)
                cur=obs_tensor(nxt).cuda()
            fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1);fa=torch.cat(acts)
            rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
            with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
            recent.append((F.detach().cpu(),Y.detach().cpu()))
            with torch.no_grad():pre=m.critic_head(F);pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
            FF=torch.cat([x[0] for x in recent],0).cuda();YY=torch.cat([x[1] for x in recent],0).cuda()
            Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
            with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit);post=m.critic_head(F)
            post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
            # drift metrics before optional actor update
            Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
            feat_mean=F.mean(0);feat_std=F.std(0);tar_mean=Y.mean(0);tar_std=Y.std(0);act_mean=fa.mean(0);act_std=fa.std(0)
            row={"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"buffer_len":len(recent),
                 "feature_mean_norm":float(feat_mean.norm()),"feature_std_norm":float(feat_std.norm()),
                 "target_mean":tar_mean.cpu().tolist(),"target_std":tar_std.cpu().tolist(),
                 "action_mean_norm":float(act_mean.norm()),"action_std_norm":float(act_std.norm()),
                 "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                 "head_bias_drift":float((bnow-prev_b).norm())}
            if prev_F is not None:
                row["feature_mean_drift"]=float((feat_mean-prev_F[0]).norm())
                row["feature_std_drift"]=float((feat_std-prev_F[1]).norm())
                row["target_mean_drift"]=float((tar_mean-prev_Y[0]).norm())
                row["target_std_drift"]=float((tar_std-prev_Y[1]).norm())
            # optional actor update from refreshed critic
            if args.actor_update=="on":
                with torch.no_grad():
                    vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4);nv=m.value_with_preference(cur,w)
                    adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
                if float((ratio-1).abs().max())>1e-4:raise RuntimeError("ratio invariant")
                al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
            actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
            row["actor_param_drift"]=float((actor_now-prev_actor).norm())
            prev_actor=actor_now.clone();prev_W=Wnow.clone();prev_b=bnow.clone();prev_F=(feat_mean.clone(),feat_std.clone());prev_Y=(tar_mean.clone(),tar_std.clone())
            logs.append(row);save_snap(update)
        all_logs[label]=logs
      out={"schema":"t5_c19_actor_coupling_train_v1","status":"COMPLETE","actor_update":args.actor_update,"recent_window":args.recent_window,
           "ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,"specialist_logs":all_logs}
      args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","actor_update":args.actor_update},indent=2))
    except BaseException as e:
      args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
    finally:
      if env is not None:env.close()
      app.close()
if __name__=="__main__":main()
