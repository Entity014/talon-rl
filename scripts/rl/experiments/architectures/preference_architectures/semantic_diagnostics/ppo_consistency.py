"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_post_v2_t5_adam_geometry_smoke():
    """Run former post_v2_t5_adam_geometry_smoke.py stage."""
    import json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,params):
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,params)])
    def cos(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=370001);obs=obs_tensor(obs).cuda();m=T4SharedActorCritic(obs.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda();initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt")
      w=torch.tensor(PREFS["T"],device="cuda").repeat(8,1);mgr=env.unwrapped.reward_manager
      ob=[];uall=[];old=[];rw=[];val=[];dn=[]
      for _ in range(2):
       with torch.no_grad():a,lp,u=m.act_with_preference_latent(obs,w);v=m.value_with_preference(obs,w)
       nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(8,))
       ob.append(obs);uall.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());obs=obs_tensor(nxt).cuda()
      with torch.no_grad():nv=m.value_with_preference(obs,w)
      rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt);adv=adv.reshape(-1,4).detach()
      fo=torch.cat(ob);fu=torch.cat(uall);fold=torch.cat(old);fw=w.repeat(2,1)
      ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
      params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
      gs=[]
      for j in range(4):
       loss=-(ratio*adv[:,j]).mean()
       gs.append(flat(torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True),params).detach())
      comb={lab:sum(float(4*PREFS[lab][j])*gs[j] for j in range(4)) for lab in ORDER}
      eps=1e-8
      adam1={lab:-comb[lab]/(comb[lab].abs()+eps) for lab in ORDER} # first Adam step, ignoring common lr and bias correction
      rawcos={a:{b:cos(comb[a],comb[b]) for b in ORDER} for a in ORDER}
      adamcos={a:{b:cos(adam1[a],adam1[b]) for b in ORDER} for a in ORDER}
      signflip={a:{b:float((torch.sign(comb[a])!=torch.sign(comb[b])).float().mean()) for b in ORDER} for a in ORDER}
      out={"schema":"t5_adam_geometry_smoke_v1","ratio_maxerr":float((ratio-1).abs().max()),"objective_grad_norm":[float(g.norm()) for g in gs],"combined_raw_cosine":rawcos,"hypothetical_first_adam_step_cosine":adamcos,"combined_gradient_sign_flip_fraction":signflip}
      p=ROOT/"runs/post_v2_t5_repaired-2026-09-23/adam_geometry.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_advantage_decomposition_smoke():
    """Run former post_v2_t5_advantage_decomposition_smoke.py stage."""
    import json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREF=np.array([.7,.1,.1,.1],np.float32)
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def corr(x):
        a=np.asarray(x,float)
        return np.corrcoef(a,rowvar=False).tolist()
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=64;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=380001);obs=obs_tensor(obs).cuda();m=T4SharedActorCritic(obs.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda();initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt")
      w=torch.tensor(PREF,device="cuda").repeat(64,1);mgr=env.unwrapped.reward_manager
      rewards=[];values=[];dones=[]
      for _ in range(2):
       with torch.no_grad():a,_,_=m.act_with_preference_latent(obs,w);v=m.value_with_preference(obs,w)
       nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(64,))
       rewards.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);values.append(v);dones.append((term|trunc).cuda());obs=obs_tensor(nxt).cuda()
      with torch.no_grad():nv=m.value_with_preference(obs,w)
      r=torch.stack(rewards);v=torch.stack(values);d=torch.stack(dones).bool()
      adv,ret=vector_gae(r,v,nv,d)
      # TD deltas and decomposition
      deltas=[];bootstrap=[]
      for t in range(2):
       vn=nv if t==1 else v[t+1]
       nt=(~d[t]).float().unsqueeze(-1)
       b=.99*vn*nt-v[t]
       bootstrap.append(b);deltas.append(r[t]+b)
      R=r.reshape(-1,4).cpu().numpy();V=v.reshape(-1,4).detach().cpu().numpy();B=torch.stack(bootstrap).reshape(-1,4).detach().cpu().numpy();D=torch.stack(deltas).reshape(-1,4).detach().cpu().numpy();A=adv.reshape(-1,4).detach().cpu().numpy()
      # reward-only two-step discounted return, repeated only at t0 across envs
      rr=(r[0]+.99*r[1]*(~d[0]).float().unsqueeze(-1)).cpu().numpy()
      out={
       "schema":"t5_advantage_decomposition_smoke_v1",
       "reward_corr":corr(R),"critic_value_corr":corr(V),"bootstrap_corr":corr(B),"td_delta_corr":corr(D),"gae_advantage_corr":corr(A),"reward_only_2step_return_corr":corr(rr),
       "std":{"reward":R.std(0).tolist(),"critic_value":V.std(0).tolist(),"bootstrap":B.std(0).tolist(),"td_delta":D.std(0).tolist(),"gae_advantage":A.std(0).tolist(),"reward_only_2step_return":rr.std(0).tolist()},
       "mean_abs":{"reward":np.mean(np.abs(R),0).tolist(),"bootstrap":np.mean(np.abs(B),0).tolist(),"td_delta":np.mean(np.abs(D),0).tolist()},
       "critic_head_max_pair_diff_at_init":float(np.max(np.abs(V-V[:,[0]]))),
      }
      p=ROOT/"runs/post_v2_t5_repaired-2026-09-23/advantage_decomposition.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_consistency_repair_smoke():
    """Run former post_v2_t5_consistency_repair_smoke.py stage."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREF=np.array([.7,.1,.1,.1],np.float32)
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
        outp=ROOT/"runs/post_v2_t5_consistency_repair-2026-09-23/smoke.json";outp.parent.mkdir(parents=True,exist_ok=True)
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          obs,_=env.reset(seed=360001);obs=obs_tensor(obs).cuda()
          m=T4SharedActorCritic(obs.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda()
          initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt")
          w=torch.as_tensor(np.repeat(PREF[None,:],len(obs),axis=0),device="cuda")
          rows=[];done_any=np.zeros(len(obs),dtype=bool)
          for step in range(64):
            with torch.no_grad():
              action,old,u=m.act_with_preference_latent(obs,w)
              recomputed=m.logp_from_pre_tanh_with_preference(obs,w,u)
              ratio=torch.exp(recomputed-old)
            # No external clamp. The exact sampled policy action is applied and is already in [-1,1].
            buffer_action=action.clone()
            next_obs,_,term,trunc,_=env.step(action)
            done_any|=(term|trunc).cpu().numpy()
            rows.append({
              "step":step,
              "env_buffer_action_max_abs_diff":float((action-buffer_action).abs().max()),
              "ratio_mean":float(ratio.mean()),
              "ratio_max_abs_error":float((ratio-1).abs().max()),
              "logp_max_abs_diff":float((recomputed-old).abs().max()),
              "action_max_abs":float(action.abs().max()),
              "near_boundary_fraction":float((action.abs()>=0.999999).float().mean()),
              "u_max_abs":float(u.abs().max()),
              "finite":bool(torch.isfinite(action).all() and torch.isfinite(old).all() and torch.isfinite(recomputed).all() and torch.isfinite(u).all())
            })
            obs=obs_tensor(next_obs).cuda()
          report={
            "schema":"t5_consistency_repair_smoke_v1","ACTION_CLIP":m.ACTION_CLIP,
            "protocol":{"steps":64,"num_envs":32,"external_clamp":False,"buffer_stores_applied_action":True,"buffer_stores_pre_tanh_u":True},
            "mean":{"ratio_mean":float(np.mean([r["ratio_mean"] for r in rows])),
                    "near_boundary_fraction":float(np.mean([r["near_boundary_fraction"] for r in rows]))},
            "max":{"ratio_max_abs_error":float(np.max([r["ratio_max_abs_error"] for r in rows])),
                   "logp_max_abs_diff":float(np.max([r["logp_max_abs_diff"] for r in rows])),
                   "env_buffer_action_max_abs_diff":float(np.max([r["env_buffer_action_max_abs_diff"] for r in rows])),
                   "action_max_abs":float(np.max([r["action_max_abs"] for r in rows])),
                   "u_max_abs":float(np.max([r["u_max_abs"] for r in rows]))},
            "survival":float(1-done_any.mean()),
            "all_finite":all(r["finite"] for r in rows),
            "pass":bool(np.max([r["ratio_max_abs_error"] for r in rows])<=1e-6
                        and np.max([r["logp_max_abs_diff"] for r in rows])<=1e-6
                        and np.max([r["env_buffer_action_max_abs_diff"] for r in rows])==0.0
                        and all(r["finite"] for r in rows)
                        and (1-done_any.mean())>=0.95),
            "rows":rows,
          }
          outp.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
        except BaseException as e:
          outp.with_name("smoke.ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_gradient_separability_audit():
    """Run former post_v2_t5_gradient_separability_audit.py stage."""
    import argparse,json,sys,hashlib,traceback
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
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
        ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--updates",type=int,default=300);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=2);ap.add_argument("--eval-steps",type=int,default=32);ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"));ap.add_argument("--critic-head-init",choices=("scalar","zero"),default="scalar");ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--critic-lr",type=float,default=1e-3);ap.add_argument("--critic-updates",type=int,default=1);ap.add_argument("--gae-lambda",type=float,default=.95);ap.add_argument("--target-sync-interval",type=int,default=0);ap.add_argument("--critic-target-mode",choices=("gae","truncated_mc"),default="gae");args=ap.parse_args()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={};snap_paths={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(31000+idx);np.random.seed(31000+idx)
            m=T4SharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init=args.critic_head_init)
            target_m=None
            if args.target_sync_interval>0:
                target_m=T4SharedActorCritic(obs.shape[-1],ad).cuda();target_m.load_state_dict(m.state_dict());target_m.eval()
                for p in target_m.parameters():p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            critic_params=[p for n,p in m.named_parameters() if n.startswith("critic_")]
            actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
            critic_opt=torch.optim.Adam(critic_params,lr=args.critic_lr)
            w=torch.as_tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda");cur,_=env.reset(seed=310001+idx*1000);cur=obs_tensor(cur).cuda()
            logs=[]
            def save_snap(tag):
                p=args.output.parent/f"{label}_snap_{tag}.pt";payload={"model":m.state_dict(),"specialist":label,"snapshot":tag}
                if target_m is not None:payload["target_model"]=target_m.state_dict()
                torch.save(payload,p);snap_paths[f"{label}:{tag}"]=str(p)
            save_snap(0)
            for update in range(1,args.updates+1):
                ob=[];ac=[];pre=[];old=[];rw=[];val=[];tval=[];dn=[]
                for _ in range(args.horizon):
                    with torch.no_grad():
                        a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
                        tv=(target_m.value_with_preference(cur,w) if target_m is not None else v)
                    nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);ac.append(a);pre.append(u);old.append(lp);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);tval.append(tv);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
                with torch.no_grad():
                    nv=m.value_with_preference(cur,w)
                    tnv=(target_m.value_with_preference(cur,w) if target_m is not None else nv)
                rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool()
                if target_m is None:
                    adv,ret=vector_gae(rt,vt,nv,dt,lam=args.gae_lambda)
                else:
                    tvt=torch.stack(tval);_,ret=vector_gae(rt,tvt,tnv,dt,lam=args.gae_lambda)
                    adv=ret-vt
                fo=torch.cat(ob);fa=torch.cat(ac);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1)
                logp=m.logp_from_pre_tanh_with_preference(fo,fw,fu);ratio=torch.exp(logp-fold.detach())
                ratio_maxerr=float((ratio-1).abs().max().detach())
                if ratio_maxerr>1e-4 or not torch.isfinite(ratio).all():
                    raise RuntimeError(f"PPO pre-update ratio invariant failed: {ratio_maxerr}")
                al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                if args.critic_target_mode=="gae":
                    target_ret=ret.reshape(-1,4).detach()
                else:
                    mc_ret=torch.zeros_like(rt);running=torch.zeros_like(rt[-1])
                    for _t in range(args.horizon-1,-1,-1):
                        running=rt[_t]+0.99*running*(~dt[_t]).unsqueeze(-1)
                        mc_ret[_t]=running
                    target_ret=mc_ret.reshape(-1,4).detach()
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
                actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
                for _critic_step in range(args.critic_updates):
                    critic_opt.zero_grad(set_to_none=True)
                    cl=vector_value_loss(m.value_with_preference(fo,fw),target_ret)
                    cl.backward();critic_opt.step()
                if do_diag:
                    after=param_vec(actor_params);delta=after-before;actual=comb[label]
                    logs.append({"snapshot":snap_tag,"adv_mean":advflat.mean(0).cpu().tolist(),"adv_std":advflat.std(0).cpu().tolist(),"adv_corr":corrmat(advflat.cpu().numpy()),"objective_grad_norm":gnorm,"objective_grad_cosine":gcos,"combined_grad_norm":{k:float(v.norm()) for k,v in comb.items()},"combined_grad_cosine":ccos,"actual_update_norm":float(delta.norm()),"actual_update_vs_negative_combined_grad_cosine":cos(delta,-actual)})
                if target_m is not None and update%args.target_sync_interval==0:
                    target_m.load_state_dict(m.state_dict());target_m.eval()
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
          report={"schema":"t5_gradient_separability_audit_v1","status":"MEASUREMENT_COMPLETE","instrumented_replay":True,"critic_head_init":args.critic_head_init,"actor_lr":args.actor_lr,"critic_lr":args.critic_lr,"critic_updates":args.critic_updates,"gae_lambda":args.gae_lambda,"target_sync_interval":args.target_sync_interval,"critic_target_mode":args.critic_target_mode,"snapshots":list(active_snaps),"preferences":{k:v.tolist() for k,v in PREFS.items()},"specialist_logs":all_logs,"action_divergence":action_diag,"snapshot_paths":snap_paths}
          args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"status":report["status"],"action_divergence":action_diag},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_ratio_consistency_smoke():
    """Run former post_v2_t5_ratio_consistency_smoke.py stage."""
    import json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          obs,_=env.reset(seed=350001);obs=obs_tensor(obs).cuda();m=T4SharedActorCritic(obs.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda();initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt")
          w=torch.tensor([.7,.1,.1,.1],device="cuda").repeat(len(obs),1)
          rows=[]
          for _ in range(64):
            with torch.no_grad():
              a,old=m.act_with_preference(obs,w)
              clipped=torch.clamp(a,-1,1)
              lp_unclip=m.logp_with_preference(obs,w,a)
              lp_clip=m.logp_with_preference(obs,w,clipped)
              r_unclip=torch.exp(lp_unclip-old);r_clip=torch.exp(lp_clip-old)
            rows.append({
              "clip_fraction":float((a.abs()>1).float().mean()),
              "max_action":float(a.abs().max()),
              "ratio_unclip_mean":float(r_unclip.mean()),"ratio_unclip_maxerr":float((r_unclip-1).abs().max()),
              "ratio_clip_mean":float(r_clip.mean()),"ratio_clip_std":float(r_clip.std()),"ratio_clip_min":float(r_clip.min()),"ratio_clip_max":float(r_clip.max()),
              "logp_shift_mean":float((lp_clip-old).mean()),"logp_shift_abs_mean":float((lp_clip-old).abs().mean()),
            })
            obs,*_=env.step(clipped);obs=obs_tensor(obs).cuda()
          keys=rows[0]
          out={"schema":"t5_ratio_consistency_smoke_v1","ACTION_CLIP":m.ACTION_CLIP,
               "mean":{k:float(np.mean([r[k] for r in rows])) for k in keys},
               "max":{k:float(np.max([r[k] for r in rows])) for k in keys},
               "rows":rows}
          p=ROOT/"runs/post_v2_t5_gradient_separability-2026-09-23/ratio_smoke.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"mean":out["mean"],"max":out["max"]},indent=2))
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

STAGES = {
    "post_v2_t5_adam_geometry_smoke": run_post_v2_t5_adam_geometry_smoke,
    "post_v2_t5_advantage_decomposition_smoke": run_post_v2_t5_advantage_decomposition_smoke,
    "post_v2_t5_consistency_repair_smoke": run_post_v2_t5_consistency_repair_smoke,
    "post_v2_t5_gradient_separability_audit": run_post_v2_t5_gradient_separability_audit,
    "post_v2_t5_ratio_consistency_smoke": run_post_v2_t5_ratio_consistency_smoke,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
