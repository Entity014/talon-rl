"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_post_v2_t5_c0_zero_critic_audit():
    """Run former post_v2_t5_c0_zero_critic_audit.py stage."""
    import json,sys,hashlib,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cosine(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
    def corr(x):return np.corrcoef(np.asarray(x,float),rowvar=False)
    def offmean(m):return float(np.mean([m[i,j] for i in range(4) for j in range(i+1,4)]))
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
        outdir=ROOT/"runs/post_v2_t5_c0_zero_critic-2026-09-23";outdir.mkdir(parents=True,exist_ok=True)
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae
          from talon_rl.rewards.objectives import normalized_objective_vector
          ckpt=ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt"
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=64;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          obs,_=env.reset(seed=390001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          torch.manual_seed(9001)
          scalar=T4SharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(scalar,ckpt,critic_head_init="scalar")
          zero=T4SharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(zero,ckpt,critic_head_init="zero")
          w=torch.tensor(PREFS["T"],device="cuda").repeat(len(obs),1)
          with torch.no_grad():
            a_scalar=scalar.act_inference_with_preference(obs,w);a_zero=zero.act_inference_with_preference(obs,w)
            v_zero=zero.value_with_preference(obs,w);v_scalar=scalar.value_with_preference(obs,w)
          actor_maxdiff=float((a_scalar-a_zero).abs().max());zero_value_maxabs=float(v_zero.abs().max())
          # Two-step rollout with repaired action/logp semantics.
          ob=[];pre=[];old=[];rw=[];val=[];dn=[];cur=obs
          for _ in range(2):
            with torch.no_grad():a,lp,u=zero.act_with_preference_latent(cur,w);v=zero.value_with_preference(cur,w)
            nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            vec=normalized_objective_vector(terms(raw,names),shape=(len(obs),))
            ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
          with torch.no_grad():nv=zero.value_with_preference(cur,w)
          r=torch.stack(rw);v=torch.stack(val);d=torch.stack(dn).bool();adv,_=vector_gae(r,v,nv,d)
          fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(2,1)
          recomputed=zero.logp_from_pre_tanh_with_preference(fo,fw,fu);ratio=torch.exp(recomputed-fold.detach())
          ratio_maxerr=float((ratio-1).abs().max());logp_maxdiff=float((recomputed-fold).abs().max())
          # Gradient geometry on same batch.
          ps=[p for n,p in zero.named_parameters() if n.startswith("actor_") or n=="log_std"]
          af=adv.reshape(-1,4).detach();gobj=[]
          for j in range(4):
            lj=-(ratio*af[:,j]).mean()
            gobj.append(flat(torch.autograd.grad(lj,ps,retain_graph=True,allow_unused=True),ps).detach())
          gcos=np.array([[cosine(gobj[i],gobj[j]) for j in range(4)] for i in range(4)])
          comb={}
          for lab in ORDER:
            ww=PREFS[lab];comb[lab]=sum(float(4*ww[j])*gobj[j] for j in range(4))
          ccos=np.array([[cosine(comb[a],comb[b]) for b in ORDER] for a in ORDER])
          rcorr=corr(r.reshape(-1,4).cpu().numpy());acorr=corr(af.cpu().numpy())
          # Save/resume exactness.
          cp=outdir/"c0_zero_critic_checkpoint.pt";torch.save({"model":zero.state_dict(),"critic_head_init":"zero"},cp)
          resumed=T4SharedActorCritic(obs.shape[-1],ad).cuda();resumed.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);resumed.eval()
          test_obs=fo[:32];test_w=fw[:32];test_u=fu[:32]
          with torch.no_grad():
            resume_action_diff=float((resumed.act_inference_with_preference(test_obs,test_w)-zero.act_inference_with_preference(test_obs,test_w)).abs().max())
            resume_value_diff=float((resumed.value_with_preference(test_obs,test_w)-zero.value_with_preference(test_obs,test_w)).abs().max())
            resume_logp_diff=float((resumed.logp_from_pre_tanh_with_preference(test_obs,test_w,test_u)-zero.logp_from_pre_tanh_with_preference(test_obs,test_w,test_u)).abs().max())
          # Historical scalar-head comparator on same rewards/dones: use scalar critic values on the same stored observations.
          with torch.no_grad():
            sv=torch.stack([scalar.value_with_preference(o,w) for o in ob]);snv=scalar.value_with_preference(cur,w)
          sadv,_=vector_gae(r,sv,snv,d);scorr=corr(sadv.reshape(-1,4).cpu().numpy())
          metrics={
            "actor_max_abs_diff_zero_vs_scalar_init":actor_maxdiff,
            "zero_critic_value_max_abs":zero_value_maxabs,
            "ratio_max_abs_error":ratio_maxerr,
            "logp_max_abs_diff":logp_maxdiff,
            "reward_corr_offdiag_mean":offmean(rcorr),
            "scalar_head_gae_corr_offdiag_mean_same_rollout":offmean(scorr),
            "zero_head_gae_corr_offdiag_mean":offmean(acorr),
            "objective_gradient_cosine_offdiag_mean":offmean(gcos),
            "combined_gradient_cosine_offdiag_mean":offmean(ccos),
            "resume_action_max_abs_diff":resume_action_diff,
            "resume_value_max_abs_diff":resume_value_diff,
            "resume_logp_max_abs_diff":resume_logp_diff,
          }
          gates={
            "actor_unchanged":actor_maxdiff==0.0,
            "zero_head_exact":zero_value_maxabs==0.0,
            "ratio_invariant":ratio_maxerr<=1e-5 and logp_maxdiff<=1e-5,
            "resume_exact":max(resume_action_diff,resume_value_diff,resume_logp_diff)==0.0,
            "advantage_separability_improved":offmean(acorr) < offmean(scorr)-0.25,
            "gradient_direction_separated":offmean(gcos) < 0.90,
            "combined_preference_direction_separated":offmean(ccos) < 0.95,
          }
          report={
            "schema":"t5_c0_zero_critic_audit_v1","status":"PASS" if all(gates.values()) else "FAIL",
            "intervention":"ONLY critic output head initialization scalar-copy -> zero; critic body/actor/reward/preferences/PPO semantics unchanged",
            "metrics":metrics,"gates":gates,
            "matrices":{"reward_corr":rcorr.tolist(),"scalar_head_gae_corr_same_rollout":scorr.tolist(),"zero_head_gae_corr":acorr.tolist(),"objective_gradient_cosine":gcos.tolist(),"combined_gradient_cosine":ccos.tolist()},
            "checkpoint":{"path":str(cp),"sha256":sha(cp)},
          }
          (outdir/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
          print(json.dumps({"status":report["status"],"metrics":metrics,"gates":gates},indent=2))
        except BaseException as e:
          (outdir/"ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c2_gradient_physical_audit():
    """Run former post_v2_t5_c2_gradient_physical_audit.py stage."""
    import copy,json,sys,traceback
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    OBJ_IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS_KEY={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    SNAPS=(0,10,50)
    CRITIC_SNAPS=(0,10,50,100)
    HORIZONS=(1,2,4,8,16,32)
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def set_from_flat(ps,base,delta):
        o=0
        with torch.no_grad():
            for p,b in zip(ps,base):
                n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def clone_params(ps):return [p.detach().clone() for p in ps]
    def explained_variance(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float);v=np.var(y)
        return float(1-np.var(y-p)/(v+1e-12))
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metrics(env,model,w,seed,max_h=32):
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"]
        prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
        rows=[]
        with torch.no_grad():
          for t in range(1,max_h+1):
            a=model.act_inference_with_preference(cur,w)
            nxt,_,term,trunc,_=env.step(a);data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            rows.append({
              "t":t,
              "vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),
              "wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),
              "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
              "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
              "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
              "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean()),
              "survival_step":float(1-(term|trunc).float().mean()),
            })
            prev=a;cur=obs_tensor(nxt).cuda()
        out={}
        for H in HORIZONS:
          seg=rows[:H]
          out[str(H)]={k:float(np.mean([r[k] for r in seg])) for k in rows[0] if k!="t"}
        return out
    
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     outdir=ROOT/"runs/post_v2_t5_c2_gradient_physical-2026-09-23";outdir.mkdir(parents=True,exist_ok=True)
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=16;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
    
      # Critic accuracy: evaluate head predictions against 32-step Monte-Carlo normalized returns.
      critic={}
      for lab in ("A","O","S"):
        critic[lab]={}
        w=torch.tensor(PREFS[lab],device="cuda").repeat(16,1)
        for snap in CRITIC_SNAPS:
          cp=ROOT/f"runs/post_v2_t5_c1_zero_critic-2026-09-23/{lab}_snap_{snap}.pt"
          m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
          cur,_=env.reset(seed=410000+OBJ_IDX[lab]*1000+snap);cur=obs_tensor(cur).cuda()
          vals=[];rews=[];dones=[]
          with torch.no_grad():
            for _ in range(32):
              vals.append(m.value_with_preference(cur,w).cpu().numpy())
              a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
              raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
              rews.append(normalized_objective_vector(terms(raw,names),shape=(16,))*env.unwrapped.step_dt)
              dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
          R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals)
          mc=np.zeros_like(R);run=np.zeros_like(R[0])
          for t in range(31,-1,-1):
            run=R[t]+.99*run*(~D[t])[:,None];mc[t]=run
          td_bias=[];ev=[]
          for j in range(4):
            ev.append(explained_variance(mc[:,:,j].reshape(-1),V[:,:,j].reshape(-1)))
            td_bias.append(float(np.mean(V[:,:,j]-mc[:,:,j])))
          critic[lab][str(snap)]={"explained_variance":ev,"value_minus_mc_bias":td_bias,"value_std":V.reshape(-1,4).std(0).tolist(),"mc_std":mc.reshape(-1,4).std(0).tolist()}
    
      interventions=[]
      # Reconstruct one batch per branch/snapshot and compare own raw gradient vs actual Adam actor step.
      for lab in ("A","O","S"):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(16,1);j=OBJ_IDX[lab]
        for snap in SNAPS:
          cp=ROOT/f"runs/post_v2_t5_c1_zero_critic-2026-09-23/{lab}_snap_{snap}.pt"
          base_model=T4SharedActorCritic(od,ad).cuda();base_model.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
          opt=torch.optim.Adam(base_model.parameters(),lr=1e-3)
          # matched short batch for gradient/update construction
          cur,_=env.reset(seed=420000+OBJ_IDX[lab]*1000+snap);cur=obs_tensor(cur).cuda()
          ob=[];pre=[];old=[];rw=[];val=[];dn=[]
          for _ in range(2):
            with torch.no_grad():a,lp,u=base_model.act_with_preference_latent(cur,w);v=base_model.value_with_preference(cur,w)
            nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            vec=normalized_objective_vector(terms(raw,names),shape=(16,))
            ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
          with torch.no_grad():nv=base_model.value_with_preference(cur,w)
          rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt)
          fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(2,1)
          ratio=torch.exp(base_model.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
          actor_ps=[p for n,p in base_model.named_parameters() if n.startswith("actor_")] # deterministic policy only
          base_actor=clone_params(actor_ps)
          own_loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean()
          own_g=flat(torch.autograd.grad(own_loss,actor_ps,retain_graph=True,allow_unused=True),actor_ps).detach()
          # actual joint loss optimizer step, then extract actor delta
          al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
          cl=vector_value_loss(base_model.value_with_preference(fo,fw),ret.reshape(-1,4).detach())
          opt.zero_grad(set_to_none=True);(al+cl).backward();opt.step()
          adam_delta=torch.cat([(p.detach()-b).reshape(-1) for p,b in zip(actor_ps,base_actor)])
          adam_norm=float(adam_delta.norm());raw_dir=-own_g/(own_g.norm()+1e-12)
          raw_delta=raw_dir*adam_norm
          # Build three models from same checkpoint: baseline, raw-SGD own objective, actual Adam actor delta.
          models={}
          for kind,delta in (("baseline",torch.zeros_like(raw_delta)),("raw_own_sgd",raw_delta),("actual_adam",adam_delta)):
            mm=T4SharedActorCritic(od,ad).cuda();mm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);aps=[p for n,p in mm.named_parameters() if n.startswith("actor_")]
            set_from_flat(aps,clone_params(aps),delta);mm.eval();models[kind]=mm
          # Two matched reset suites; same initial distribution per condition.
          suite_rows=[]
          for suite in range(2):
            seed=430000+OBJ_IDX[lab]*1000+snap*10+suite
            rr={kind:rollout_metrics(env,mm,w,seed,32) for kind,mm in models.items()}
            suite_rows.append({"suite":suite,"seed":seed,"rollouts":rr})
          interventions.append({"specialist":lab,"snapshot":snap,"own_objective_index":j,"adam_actor_step_norm":adam_norm,"own_gradient_norm":float(own_g.norm()),"suites":suite_rows})
      report={"schema":"t5_c2_gradient_physical_audit_v1","status":"MEASUREMENT_COMPLETE","horizons":list(HORIZONS),"critic_accuracy":critic,"interventions":interventions,
              "notes":{"raw_own_sgd":"negative own-objective loss gradient normalized to exact actual Adam actor-step norm","actual_adam":"actual actor parameter delta from one joint PPO+critic Adam step on the same batch","baseline":"same checkpoint with no perturbation"}}
      (outdir/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
      print(json.dumps({"status":report["status"],"num_interventions":len(interventions),"critic":critic},indent=2))
     except BaseException as e:
      (outdir/"ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c3_optimizer_counterfactual():
    """Run former post_v2_t5_c3_optimizer_counterfactual.py stage."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S");SNAPS=(1,5,10);LRS=(1e-3,3e-4,1e-4,3e-5);G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     outdir=ROOT/"runs/post_v2_t5_c3_value_learning-2026-09-23";outdir.mkdir(parents=True,exist_ok=True)
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      rows=[]
      for bi,lab in enumerate(ORDER):
       w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
       for snap in SNAPS:
        cp=ROOT/f"runs/post_v2_t5_c1_zero_critic-2026-09-23/{lab}_snap_{snap}.pt"
        base=T4SharedActorCritic(od,ad).cuda();base.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);base.eval()
        cur,_=env.reset(seed=450000+bi*1000+snap);cur=obs_tensor(cur).cuda()
        obs0=cur
        with torch.no_grad():
          a=base.act_inference_with_preference(cur,w);v0=base.value_with_preference(cur,w)
        n1,_,d0,t0,_=env.step(a);n1=obs_tensor(n1).cuda()
        raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);r0=torch.tensor(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt,device="cuda")
        with torch.no_grad():a1=base.act_inference_with_preference(n1,w)
        n2,_,d1,t1,_=env.step(a1);n2=obs_tensor(n2).cuda()
        raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);r1=torch.tensor(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt,device="cuda")
        with torch.no_grad():v2=base.value_with_preference(n2,w)
        done0=(d0|t0).cuda();done1=(d1|t1).cuda();nt0=(~done0).float().unsqueeze(-1);nt1=(~done1).float().unsqueeze(-1)
        target=r0+G*nt0*r1+(G**2)*nt0*nt1*v2
        for mode in ("all","head_only"):
         for lr in LRS:
          m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
          params=[p for n,p in m.named_parameters() if (n.startswith("critic_") if mode=="all" else n.startswith("critic_head"))]
          opt=torch.optim.Adam(params,lr=lr)
          before=m.value_with_preference(obs0,w)
          lb=((before-target.detach())**2).mean(0)
          total=lb.mean()
          opt.zero_grad(set_to_none=True);total.backward();opt.step()
          after=m.value_with_preference(obs0,w)
          la=((after-target.detach())**2).mean(0)
          rows.append({"branch":lab,"snapshot":snap,"mode":mode,"lr":lr,
                       "loss_before":lb.detach().cpu().tolist(),"loss_after":la.detach().cpu().tolist(),
                       "change":(la-lb).detach().cpu().tolist(),
                       "all_heads_improved":bool(torch.all(la<lb)),
                       "mean_relative_change":float(((la-lb)/(lb.abs()+1e-12)).mean())})
      out={"schema":"t5_c3_optimizer_counterfactual_v1","rows":rows}
      (outdir/"optimizer_counterfactual.json").write_text(json.dumps(out,indent=2)+"\n")
      agg={}
      for mode in ("all","head_only"):
       agg[mode]={}
       for lr in LRS:
        rr=[r for r in rows if r["mode"]==mode and r["lr"]==lr]
        changes=np.array([x for r in rr for x in r["change"]],float)
        agg[mode][str(lr)]={"case_all_heads_improved_fraction":float(np.mean([r["all_heads_improved"] for r in rr])),
                            "head_improvement_fraction":float(np.mean(changes<0)),
                            "mean_relative_change":float(np.mean([r["mean_relative_change"] for r in rr]))}
      print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c3_value_learning_diagnosis():
    """Run former post_v2_t5_c3_value_learning_diagnosis.py stage."""
    """C3 diagnostic-only audit of four-head critic/value learning."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
    }
    SNAPS=(0,1,5,10,25,50,100)
    NSTEPS=(1,2,4,8,16,32)
    GAMMA=.99
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names):
        return {n:raw[:,i] for i,n in enumerate(names)}
    
    def explained_variance(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float);v=np.var(y)
        return float(1-np.var(y-p)/(v+1e-12))
    
    def flat(gs,ps):
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    
    def cosine(a,b):
        return float((a@b)/(a.norm()*b.norm()+1e-12))
    
    def nstep_targets(rew,done,values,n):
        # rew [T,N,O], done [T,N], values [T+1,N,O]
        T,N,O=rew.shape
        out=np.zeros_like(rew,dtype=np.float64)
        for t in range(T):
            ret=np.zeros((N,O),dtype=np.float64)
            disc=np.ones((N,1),dtype=np.float64)
            alive=np.ones((N,1),dtype=np.float64)
            end=min(T,t+n)
            for k in range(t,end):
                ret += disc*alive*rew[k]
                alive *= (~done[k])[:,None]
                disc *= GAMMA
            if t+n <= T:
                ret += disc*alive*values[t+n]
            out[t]=ret
        return out
    
    def mc_targets(rew,done):
        T,N,O=rew.shape
        out=np.zeros_like(rew,dtype=np.float64)
        run=np.zeros((N,O),dtype=np.float64)
        for t in range(T-1,-1,-1):
            run=rew[t]+GAMMA*run*(~done[t])[:,None]
            out[t]=run
        return out
    
    def main():
      from isaaclab.app import AppLauncher
      app=AppLauncher({"headless":True,"enable_cameras":False}).app
      env=None
      outdir=ROOT/"runs/post_v2_t5_c3_value_learning-2026-09-23";outdir.mkdir(parents=True,exist_ok=True)
      try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.models.foundations.four_objective import T4SharedActorCritic
        from talon_rl.rewards.objectives import normalized_objective_vector
    
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda()
        od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
    
        report={"schema":"t5_c3_value_learning_diagnosis_v1","status":"MEASUREMENT_COMPLETE",
                "snapshots":list(SNAPS),"nsteps":list(NSTEPS),"branches":{}}
    
        for bi,lab in enumerate(ORDER):
          w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
          branch={}
          for snap in SNAPS:
            cp=ROOT/f"runs/post_v2_t5_c1_zero_critic-2026-09-23/{lab}_snap_{snap}.pt"
            m=T4SharedActorCritic(od,ad).cuda()
            m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
    
            # Same reset protocol across snapshots within branch.
            cur,_=env.reset(seed=440000+bi*1000);cur=obs_tensor(cur).cuda()
            obs_seq=[];rew=[];done=[];vals=[]
            with torch.no_grad():
              for t in range(32):
                obs_seq.append(cur)
                vals.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,term,trunc,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                rew.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt)
                done.append((term|trunc).cpu().numpy().astype(bool))
                cur=obs_tensor(nxt).cuda()
              vals.append(m.value_with_preference(cur,w).cpu().numpy())
    
            R=np.asarray(rew,float);D=np.asarray(done,bool);V=np.asarray(vals,float)
            MC=mc_targets(R,D)
    
            target_family={}
            for n in NSTEPS:
              Tn=nstep_targets(R,D,V,n)
              # Compare only positions with a full n-step lookahead so edge truncation does not bias comparison.
              usable=max(1,32-n+1)
              sl=slice(0,usable)
              per=[]
              for j in range(4):
                err=Tn[sl,:,j]-MC[sl,:,j]
                per.append({
                  "mae_vs_mc":float(np.mean(np.abs(err))),
                  "rmse_vs_mc":float(np.sqrt(np.mean(err*err))),
                  "bias_vs_mc":float(np.mean(err)),
                  "corr_with_mc":float(np.corrcoef(Tn[sl,:,j].reshape(-1),MC[sl,:,j].reshape(-1))[0,1]),
                  "target_std":float(np.std(Tn[sl,:,j])),
                })
              target_family[str(n)]=per
    
            value_quality=[]
            for j in range(4):
              y=MC[:,:,j].reshape(-1);p=V[:-1,:,j].reshape(-1)
              value_quality.append({
                "ev_vs_mc32":explained_variance(y,p),
                "bias_value_minus_mc":float(np.mean(p-y)),
                "value_std":float(np.std(p)),
                "mc_std":float(np.std(y)),
                "value_abs_mean":float(np.mean(np.abs(p))),
                "mc_abs_mean":float(np.mean(np.abs(y))),
              })
    
            # Critic gradient interference on the actual 2-step training-style target built from first 2 transitions.
            # Target = r0 + gamma r1 + gamma^2 V(s2), respecting done.
            o0=obs_seq[0];o1=obs_seq[1]
            r0=torch.tensor(R[0],device="cuda",dtype=torch.float32)
            r1=torch.tensor(R[1],device="cuda",dtype=torch.float32)
            d0=torch.tensor(D[0],device="cuda")
            d1=torch.tensor(D[1],device="cuda")
            with torch.no_grad():
              v2=m.value_with_preference(obs_seq[2],w)
              nt0=(~d0).float().unsqueeze(-1);nt1=(~d1).float().unsqueeze(-1)
              target2=r0 + GAMMA*nt0*r1 + (GAMMA**2)*nt0*nt1*v2
            pred=m.value_with_preference(o0,w)
    
            body_ps=[p for n,p in m.named_parameters() if n.startswith("critic_body")]
            head_w=m.critic_head.weight
            head_b=m.critic_head.bias
            body_g=[];head_g=[];head_loss=[]
            for j in range(4):
              lj=(pred[:,j]-target2[:,j].detach()).pow(2).mean()
              head_loss.append(float(lj.detach()))
              gb=torch.autograd.grad(lj,body_ps,retain_graph=True,allow_unused=True)
              body_g.append(flat(gb,body_ps).detach())
              gh=torch.autograd.grad(lj,[head_w,head_b],retain_graph=True,allow_unused=True)
              # Extract only row j influence from full head gradients for a comparable per-head vector.
              gw=gh[0][j].reshape(-1);gbi=gh[1][j].reshape(-1)
              head_g.append(torch.cat([gw,gbi]).detach())
    
            body_cos=np.array([[cosine(body_g[i],body_g[j]) for j in range(4)] for i in range(4)])
            body_norm=[float(g.norm()) for g in body_g]
            head_norm=[float(g.norm()) for g in head_g]
    
            # Hypothetical one critic-only Adam step on this fixed batch, measure same-batch loss movement.
            clone=T4SharedActorCritic(od,ad).cuda();clone.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
            opt=torch.optim.Adam([p for n,p in clone.named_parameters() if n.startswith("critic_")],lr=1e-3)
            before=clone.value_with_preference(o0,w)
            losses_before=[float(((before[:,j]-target2[:,j]).pow(2).mean()).detach()) for j in range(4)]
            total=((before-target2.detach()).pow(2)).mean()
            params=[p for n,p in clone.named_parameters() if n.startswith("critic_")]
            old=[p.detach().clone() for p in params]
            opt.zero_grad(set_to_none=True);total.backward();opt.step()
            after=clone.value_with_preference(o0,w)
            losses_after=[float(((after[:,j]-target2[:,j]).pow(2).mean()).detach()) for j in range(4)]
            step_norm=float(torch.sqrt(sum(((p-o)**2).sum() for p,o in zip(params,old))))
            param_norm=float(torch.sqrt(sum((o**2).sum() for o in old)))
            loss_change=[a-b for a,b in zip(losses_after,losses_before)]
    
            branch[str(snap)]={
              "value_quality_vs_mc32":value_quality,
              "target_family":target_family,
              "training_like_2step":{
                "head_loss":head_loss,
                "target_mean":target2.detach().mean(0).cpu().tolist(),
                "target_std":target2.detach().std(0).cpu().tolist(),
                "prediction_mean":pred.detach().mean(0).cpu().tolist(),
                "prediction_std":pred.detach().std(0).cpu().tolist(),
              },
              "shared_body_interference":{
                "body_grad_norm":body_norm,
                "head_grad_norm":head_norm,
                "body_grad_cosine":body_cos.tolist(),
                "mean_offdiag_body_cosine":float(np.mean([body_cos[i,j] for i in range(4) for j in range(i+1,4)])),
                "negative_body_pair_fraction":float(np.mean([body_cos[i,j]<0 for i in range(4) for j in range(i+1,4)])),
              },
              "critic_optimizer_same_batch":{
                "lr":1e-3,
                "loss_before_per_head":losses_before,
                "loss_after_per_head":losses_after,
                "loss_change_after_minus_before":loss_change,
                "all_heads_improved":bool(all(x<0 for x in loss_change)),
                "critic_param_step_norm":step_norm,
                "critic_param_norm":param_norm,
                "relative_step_norm":step_norm/(param_norm+1e-12),
              }
            }
          report["branches"][lab]=branch
    
        (outdir/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"status":report["status"],
          "summary":{lab:{s:{
            "ev":[round(x["ev_vs_mc32"],3) for x in report["branches"][lab][s]["value_quality_vs_mc32"]],
            "2step_mae":[round(x["mae_vs_mc"],4) for x in report["branches"][lab][s]["target_family"]["2"]],
            "16step_mae":[round(x["mae_vs_mc"],4) for x in report["branches"][lab][s]["target_family"]["16"]],
            "body_cos":round(report["branches"][lab][s]["shared_body_interference"]["mean_offdiag_body_cosine"],3),
            "same_batch_improved":report["branches"][lab][s]["critic_optimizer_same_batch"]["all_heads_improved"],
            "rel_step":report["branches"][lab][s]["critic_optimizer_same_batch"]["relative_step_norm"],
          } for s in ("0","10","50","100")} for lab in ORDER}},indent=2))
      except BaseException as e:
        (outdir/"ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
      finally:
        if env is not None:env.close()
        app.close()
    if True:main()

def run_post_v2_t5_c4_samebatch_check():
    """Run former post_v2_t5_c4_samebatch_check.py stage."""
    from pathlib import Path
    import sys, json, numpy as np, torch
    ROOT=Path("/home/xero/Master's Degree/Thesis/talon-rl");sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S"); SNAPS=(10,25,50,100); G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
    try:
     import gymnasium as gym,isaaclab_tasks
     from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
     from talon_rl.models.foundations.four_objective import T4SharedActorCritic
     from talon_rl.rewards.objectives import normalized_objective_vector
     cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
     o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
     rows=[]
     for bi,lab in enumerate(ORDER):
      w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
      for snap in SNAPS:
       cp=ROOT/f"runs/post_v2_t5_c4_critic_lr1e4-2026-09-23/{lab}_snap_{snap}.pt"
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
       cur,_=env.reset(seed=490000+bi*1000+snap);cur=obs_tensor(cur).cuda();obs0=cur
       with torch.no_grad(): a=m.act_inference_with_preference(cur,w)
       n1,_,d0,t0,_=env.step(a);n1=obs_tensor(n1).cuda();raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);r0=torch.tensor(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt,device="cuda")
       with torch.no_grad(): a1=m.act_inference_with_preference(n1,w)
       n2,_,d1,t1,_=env.step(a1);n2=obs_tensor(n2).cuda();raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);r1=torch.tensor(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt,device="cuda")
       with torch.no_grad(): v2=m.value_with_preference(n2,w)
       nt0=(~(d0|t0).cuda()).float().unsqueeze(-1);nt1=(~(d1|t1).cuda()).float().unsqueeze(-1)
       target=r0+G*nt0*r1+(G**2)*nt0*nt1*v2
       params=[p for n,p in m.named_parameters() if n.startswith("critic_")]
       opt=torch.optim.Adam(params,lr=1e-4)
       before=m.value_with_preference(obs0,w);lb=((before-target.detach())**2).mean(0)
       opt.zero_grad(set_to_none=True);lb.mean().backward();opt.step()
       after=m.value_with_preference(obs0,w);la=((after-target.detach())**2).mean(0)
       rows.append({"branch":lab,"snapshot":snap,"before":lb.detach().cpu().tolist(),"after":la.detach().cpu().tolist(),"change":(la-lb).detach().cpu().tolist(),"all_improved":bool(torch.all(la<lb))})
     out={"rows":rows,"all_case_success":float(np.mean([r["all_improved"] for r in rows])),"head_improve_fraction":float(np.mean([x<0 for r in rows for x in r["change"]]))}
     p=ROOT/"runs/post_v2_t5_c4_critic_lr1e4-2026-09-23/samebatch_critic_lr_check.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
    finally:
     if env is not None:env.close()
     app.close()

def run_post_v2_t5_c4_value_semantics_audit():
    """Run former post_v2_t5_c4_value_semantics_audit.py stage."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    OBJ_IDX={"T":0,"A":1,"O":2,"S":3}; H=(1,2,4,8,16,32); G=.99
    SNAPS=(0,10,25,50,100)
    PERT_SNAPS=(10,25,50)
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
        o=0
        with torch.no_grad():
          for p,b in zip(ps,base):
            n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
        rows=[]
        with torch.no_grad():
          for t in range(1,33):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);data=robot.data
            rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"survival":float(1-(term|trunc).float().mean())})
            prev=a;cur=obs_tensor(nxt).cuda()
        return {str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H}
    
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     outdir=ROOT/"runs/post_v2_t5_c4_critic_lr1e4-2026-09-23";outdir.mkdir(parents=True,exist_ok=True)
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      critic={};pert=[]
      for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);critic[lab]={}
        for snap in SNAPS:
          cp=outdir/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
          cur,_=env.reset(seed=460000+bi*1000+snap);cur=obs_tensor(cur).cuda();vals=[];rews=[];dones=[]
          with torch.no_grad():
            for _ in range(32):
              vals.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
              raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rews.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
          R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals);MC=np.zeros_like(R);run=np.zeros_like(R[0])
          for t in range(31,-1,-1):run=R[t]+G*run*(~D[t])[:,None];MC[t]=run
          critic[lab][str(snap)]={"ev":[ev(MC[:,:,j].reshape(-1),V[:,:,j].reshape(-1)) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
        if lab in ("A","O"):
          j=OBJ_IDX[lab]
          for snap in PERT_SNAPS:
            cp=outdir/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
            cur,_=env.reset(seed=470000+bi*1000+snap);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
            for _ in range(2):
              with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
              nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
              ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
            with torch.no_grad():nv=m.value_with_preference(cur,w)
            rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt)
            fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(2,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
            aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
            loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
            # fixed small relative step to compare direction only
            basevec=torch.cat([p.detach().reshape(-1) for p in aps]);step_norm=1e-4*(basevec.norm()+1e-12);delta=-g/(g.norm()+1e-12)*step_norm
            pertm=T4SharedActorCritic(od,ad).cuda();pertm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pertm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pertm.eval();m.eval()
            suites=[]
            metric="ang_vel_xy" if lab=="A" else "tilt_deg"
            for ss in range(2):
              seed=480000+bi*1000+snap*10+ss;b=rollout(env,m,w,seed);p=rollout(env,pertm,w,seed)
              suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
            pert.append({"branch":lab,"snapshot":snap,"grad_norm":float(g.norm()),"step_norm":float(step_norm),"suites":suites})
      out={"schema":"t5_c4_value_semantics_audit_v1","critic":critic,"perturbations":pert}
      (outdir/"value_semantics_audit.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({"critic":critic,"perturbation_count":len(pert)},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c5_h16_endpoint_eval():
    """Run former post_v2_t5_c5_h16_endpoint_eval.py stage."""
    import json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c5_h16-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      models={}
      for lab in ORDER:
        m=T4SharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_100.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
      rows=[]
      for suite in range(4):
        seed=320001+suite
        for lab in ORDER:
          m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((8,ad),device="cuda");norm=[];phys=[];done=np.zeros(8,bool)
          with torch.no_grad():
           for _ in range(64):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);norm.append(normalized_objective_vector(terms(raw,names),shape=(8,)).mean(0));data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            phys.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
            done|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
          n=np.asarray(norm).mean(0);pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
          rows.append({"suite":suite,"policy":lab,"normalized_objective_mean":n.tolist(),"scalarized":{e:float(n@PREFS[e]) for e in ORDER},"physical":pm,"survival":float(1-done.mean())})
      ow={};pw={};sw={};pk={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
      for i,lab in enumerate(ORDER):
        a=[];b=[];c=[]
        for suite in range(4):
          rr=[r for r in rows if r["suite"]==suite]
          a.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==lab)
          b.append((min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==lab) if lab=="T" else (min(rr,key=lambda r:r["physical"][pk[lab]])["policy"]==lab))
          c.append(max(rr,key=lambda r:r["scalarized"][lab])["policy"]==lab)
        ow[lab]=float(np.mean(a));pw[lab]=float(np.mean(b));sw[lab]=float(np.mean(c))
      vb={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in rows if r["policy"]==p])) for p in ORDER};vr={p:vb[p]/(vb["T"]+1e-12) for p in ORDER};mins=min(r["survival"] for r in rows)
      gates={"objective_winner_fraction":ow,"physical_winner_fraction":pw,"diagonal_scalarized_winner_fraction":sw,"min_survival":mins,"vertical_ratio_to_tracking_policy":vr,"safety_pass":bool(mins>=.95 and max(vr.values())<=2)}
      out={"schema":"t5_c5_h16_endpoint_eval_v1","rows":rows,"gates":gates};(RUN/"endpoint_eval.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(gates,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c5_h16_value_endpoint_audit():
    """Run former post_v2_t5_c5_h16_value_endpoint_audit.py stage."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c5_h16-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3};SNAPS=(0,10,25,50,100);PERT=(10,25,50,100);H=(1,2,4,8,16,32);G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
        o=0
        with torch.no_grad():
          for p,b in zip(ps,base):
            n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
        rows=[];done_any=np.zeros(len(cur),bool)
        with torch.no_grad():
          for _ in range(32):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);data=robot.data
            rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
            done_any|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
        metrics={str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H}
        return metrics,float(1-done_any.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector,raw_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      critic={};pert=[]
      for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);critic[lab]={}
        for snap in SNAPS:
          cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
          cur,_=env.reset(seed=500000+bi*1000+snap);cur=obs_tensor(cur).cuda();vals=[];rews=[];dones=[]
          with torch.no_grad():
           for _ in range(32):
            vals.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rews.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
          R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals);MC=np.zeros_like(R);run=np.zeros_like(R[0])
          for t in range(31,-1,-1):run=R[t]+G*run*(~D[t])[:,None];MC[t]=run
          critic[lab][str(snap)]={"ev":[ev(MC[:,:,j].reshape(-1),V[:,:,j].reshape(-1)) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
        if lab in ("A","O"):
          j=IDX[lab]
          for snap in PERT:
            cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
            cur,_=env.reset(seed=510000+bi*1000+snap);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
            for _ in range(16):
              with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
              nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
              ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
            with torch.no_grad():nv=m.value_with_preference(cur,w)
            rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt)
            fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
            aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
            loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
            basevec=torch.cat([p.detach().reshape(-1) for p in aps]);stepnorm=1e-4*(basevec.norm()+1e-12);delta=-g/(g.norm()+1e-12)*stepnorm
            pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval();m.eval()
            metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
            for ss in range(2):
              seed=520000+bi*1000+snap*10+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
            pert.append({"branch":lab,"snapshot":snap,"grad_norm":float(g.norm()),"step_norm":float(stepnorm),"suites":suites})
      # terminal paired endpoint evaluation, 8 envs to match prior gate
      cfg2=UnitreeA1FlatEnvCfg();cfg2.scene.num_envs=8;cfg2.seed=0
      env.close();env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg2);obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      rows=[]
      models={}
      for lab in ORDER:
        m=T4SharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_100.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
      for suite in range(4):
        seed=320001+suite
        for lab in ORDER:
          m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((8,ad),device="cuda");norm=[];phys=[];done=np.zeros(8,bool)
          with torch.no_grad():
           for _ in range(64):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);norm.append(normalized_objective_vector(terms(raw,names),shape=(8,)).mean(0));data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            phys.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
            done|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
          n=np.asarray(norm).mean(0);pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
          rows.append({"suite":suite,"policy":lab,"normalized_objective_mean":n.tolist(),"scalarized":{e:float(n@PREFS[e]) for e in ORDER},"physical":pm,"survival":float(1-done.mean())})
      ow={};pw={};sw={};pk={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
      for i,lab in enumerate(ORDER):
        a=[];b=[];c=[]
        for suite in range(4):
          rr=[r for r in rows if r["suite"]==suite]
          a.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==lab)
          b.append((min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==lab) if lab=="T" else (min(rr,key=lambda r:r["physical"][pk[lab]])["policy"]==lab))
          c.append(max(rr,key=lambda r:r["scalarized"][lab])["policy"]==lab)
        ow[lab]=float(np.mean(a));pw[lab]=float(np.mean(b));sw[lab]=float(np.mean(c))
      vb={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in rows if r["policy"]==p])) for p in ORDER};vr={p:vb[p]/(vb["T"]+1e-12) for p in ORDER};mins=min(r["survival"] for r in rows)
      gates={"objective_winner_fraction":ow,"physical_winner_fraction":pw,"diagonal_scalarized_winner_fraction":sw,"min_survival":mins,"vertical_ratio_to_tracking_policy":vr,"safety_pass":bool(mins>=.95 and max(vr.values())<=2)}
      out={"schema":"t5_c5_h16_value_endpoint_audit_v1","critic":critic,"perturbations":pert,"endpoint_rows":rows,"gates":gates}
      (RUN/"value_endpoint_audit.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({"critic":critic,"gates":gates,"perturbations":len(pert)},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c5_h16_value_semantics_audit():
    """Run former post_v2_t5_c5_h16_value_semantics_audit.py stage."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c5_h16-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3};SNAPS=(0,10,25,50,100);PERT=(10,25,50,100);H=(1,2,4,8,16,32);G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
        o=0
        with torch.no_grad():
          for p,b in zip(ps,base):
            n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
        rows=[];done_any=np.zeros(len(cur),bool)
        with torch.no_grad():
          for _ in range(32):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);data=robot.data
            rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
            done_any|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
        metrics={str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H}
        return metrics,float(1-done_any.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector,raw_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      critic={};pert=[]
      for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);critic[lab]={}
        for snap in SNAPS:
          cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
          cur,_=env.reset(seed=500000+bi*1000+snap);cur=obs_tensor(cur).cuda();vals=[];rews=[];dones=[]
          with torch.no_grad():
           for _ in range(32):
            vals.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rews.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
          R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals);MC=np.zeros_like(R);run=np.zeros_like(R[0])
          for t in range(31,-1,-1):run=R[t]+G*run*(~D[t])[:,None];MC[t]=run
          critic[lab][str(snap)]={"ev":[ev(MC[:,:,j].reshape(-1),V[:,:,j].reshape(-1)) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
        if lab in ("A","O"):
          j=IDX[lab]
          for snap in PERT:
            cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
            cur,_=env.reset(seed=510000+bi*1000+snap);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
            for _ in range(16):
              with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
              nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
              ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
            with torch.no_grad():nv=m.value_with_preference(cur,w)
            rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt)
            fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
            aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
            loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
            basevec=torch.cat([p.detach().reshape(-1) for p in aps]);stepnorm=1e-4*(basevec.norm()+1e-12);delta=-g/(g.norm()+1e-12)*stepnorm
            pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval();m.eval()
            metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
            for ss in range(2):
              seed=520000+bi*1000+snap*10+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
            pert.append({"branch":lab,"snapshot":snap,"grad_norm":float(g.norm()),"step_norm":float(stepnorm),"suites":suites})
      out={"schema":"t5_c5_h16_value_semantics_audit_v1","critic":critic,"perturbations":pert}
      (RUN/"value_semantics_audit.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({"critic":critic,"perturbations":len(pert)},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c6_critic_target_representation_audit():
    """Run former post_v2_t5_c6_critic_target_representation_audit.py stage."""
    import json,sys,traceback,copy
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c5_h16-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c6_critic_target_repr-2026-09-23"
    ORDER=("T","A","O","S"); SNAPS=(0,10,25,50,100); G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);v=np.var(y)
        return float(1-np.var(y-p)/(v+1e-12))
    def corr(a,b):
        a=np.asarray(a,float).reshape(-1);b=np.asarray(b,float).reshape(-1)
        if np.std(a)<1e-12 or np.std(b)<1e-12:return 0.0
        return float(np.corrcoef(a,b)[0,1])
    def mc(rew,done):
        T,N,O=rew.shape;out=np.zeros_like(rew,dtype=np.float64);run=np.zeros((N,O),np.float64)
        for t in range(T-1,-1,-1):
            run=rew[t]+G*run*(~done[t])[:,None];out[t]=run
        return out
    def cosine(a,b):
        a=np.asarray(a,float).reshape(-1);b=np.asarray(b,float).reshape(-1)
        return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    
    def main():
      from isaaclab.app import AppLauncher
      app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
      OUT.mkdir(parents=True,exist_ok=True)
      try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
        from talon_rl.rewards.objectives import normalized_objective_vector
    
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
        report={"schema":"t5_c6_critic_target_repr_v1","status":"MEASUREMENT_COMPLETE","A_fixed_fit":{},"B_target_validity":{},"C_online_drift":{}}
    
        datasets={}
        # Collect matched 64-step replay per branch/snapshot. Actor deterministic to reduce sampling noise.
        for bi,lab in enumerate(ORDER):
          w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);datasets[lab]={}
          for snap in SNAPS:
            cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
            cur,_=env.reset(seed=600000+bi*1000);cur=obs_tensor(cur).cuda()
            obs=[];r=[];term=[];trunc=[];vals=[];nextvals=[];info_samples=[];next_obs=[]
            with torch.no_grad():
              for t in range(64):
                obs.append(cur.detach().cpu())
                vals.append(m.value_with_preference(cur,w).detach().cpu())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,info=env.step(a);nxt=obs_tensor(nxt).cuda()
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                r.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt)
                term.append(te.cpu().numpy().astype(bool));trunc.append(tr.cpu().numpy().astype(bool))
                nextvals.append(m.value_with_preference(nxt,w).detach().cpu())
                next_obs.append(nxt.detach().cpu())
                if (te|tr).any():
                  info_samples.append({"t":t,"term":int(te.sum()),"trunc":int(tr.sum()),"keys":sorted(list(info.keys())) if isinstance(info,dict) else str(type(info))})
                cur=nxt
            O=torch.stack(obs).float(); NO=torch.stack(next_obs).float(); R=np.asarray(r,float);TE=np.asarray(term,bool);TR=np.asarray(trunc,bool);D=TE|TR
            V=torch.stack(vals).numpy();NV=torch.stack(nextvals).numpy()
            MC_done=mc(R,D);MC_term=mc(R,TE)
            # one-step target variants using observed next-state value
            T_current=R+G*NV*(~D)[:,:,None]
            T_term=R+G*NV*(~TE)[:,:,None]
            datasets[lab][snap]={"obs":O,"next_obs":NO,"R":R,"TE":TE,"TR":TR,"D":D,"V":V,"NV":NV,"MC_done":MC_done,"MC_term":MC_term}
            # Current GAE implementation from exact rollout.
            rt=torch.tensor(R,dtype=torch.float32,device="cuda");vt=torch.tensor(V,dtype=torch.float32,device="cuda");dt=torch.tensor(D,device="cuda")
            # vector_gae expects a single next value for end of segment; use last observed NV.
            adv,gae_ret=vector_gae(rt,vt,torch.tensor(NV[-1],dtype=torch.float32,device="cuda"),dt)
            gae=gae_ret.cpu().numpy()
            report["B_target_validity"][f"{lab}:{snap}"]={
              "termination_count":int(TE.sum()),"truncation_count":int(TR.sum()),
              "done_count":int(D.sum()),"event_samples":info_samples[:12],
              "current_1step_vs_mc_done_mae":[float(np.mean(np.abs(T_current[:,:,j]-MC_done[:,:,j]))) for j in range(4)],
              "term_only_1step_vs_mc_term_mae":[float(np.mean(np.abs(T_term[:,:,j]-MC_term[:,:,j]))) for j in range(4)],
              "gae_vs_mc_done_mae":[float(np.mean(np.abs(gae[:,:,j]-MC_done[:,:,j]))) for j in range(4)],
              "gae_vs_mc_term_mae":[float(np.mean(np.abs(gae[:,:,j]-MC_term[:,:,j]))) for j in range(4)],
              "gae_bias_vs_mc_done":[float(np.mean(gae[:,:,j]-MC_done[:,:,j])) for j in range(4)],
              "value_ev_vs_mc_done":[ev(MC_done[:,:,j],V[:,:,j]) for j in range(4)],
              "value_ev_vs_mc_term":[ev(MC_term[:,:,j],V[:,:,j]) for j in range(4)],
              "mc_done_vs_mc_term_mae":[float(np.mean(np.abs(MC_done[:,:,j]-MC_term[:,:,j]))) for j in range(4)],
            }
    
        # C6-A: frozen-target fit. Use snap 50 and 100 where online EV is poor.
        for lab in ORDER:
          report["A_fixed_fit"][lab]={}
          wbase=torch.tensor(PREFS[lab],device="cuda")
          for snap in (50,100):
            d=datasets[lab][snap];X=d["obs"].reshape(-1,od).cuda();Y=torch.tensor(d["MC_done"].reshape(-1,4),dtype=torch.float32,device="cuda");W=wbase.repeat(len(X),1)
            cp=RUN/f"{lab}_snap_{snap}.pt"
            base=T4SharedActorCritic(od,ad).cuda();base.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);base.eval()
            with torch.no_grad(): p0=base.value_with_preference(X,W)
            init_ev=[ev(Y[:,j].cpu().numpy(),p0[:,j].cpu().numpy()) for j in range(4)]
    
            # Analytical best linear head on frozen critic-body features.
            with torch.no_grad():
              feats=base.critic_body(base._with_w(X,W))
              A=torch.cat([feats,torch.ones((len(feats),1),device="cuda")],dim=1)
              sol=torch.linalg.lstsq(A,Y).solution
              pred_lin=A@sol
            lin_ev=[ev(Y[:,j].cpu().numpy(),pred_lin[:,j].cpu().numpy()) for j in range(4)]
            lin_rmse=[float(torch.sqrt(((pred_lin[:,j]-Y[:,j])**2).mean())) for j in range(4)]
    
            # Head-only Adam fit on fixed targets.
            hm=T4SharedActorCritic(od,ad).cuda();hm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
            for n,p in hm.named_parameters(): p.requires_grad_(n.startswith("critic_head"))
            opt=torch.optim.Adam([p for p in hm.parameters() if p.requires_grad],lr=1e-3)
            head_curve=[]
            for k in range(501):
              pred=hm.value_with_preference(X,W);loss=((pred-Y)**2).mean()
              if k in (0,1,10,50,100,250,500):
                head_curve.append({"step":k,"loss":float(loss.detach()),"ev":[ev(Y[:,j].cpu().numpy(),pred[:,j].detach().cpu().numpy()) for j in range(4)]})
              if k<500: opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    
            # Full critic fixed-dataset fit, actor frozen. Use stable LR 1e-4.
            fm=T4SharedActorCritic(od,ad).cuda();fm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
            pars=[p for n,p in fm.named_parameters() if n.startswith("critic_")]
            opt2=torch.optim.Adam(pars,lr=1e-4);full_curve=[]
            for k in range(1001):
              pred=fm.value_with_preference(X,W);loss=((pred-Y)**2).mean()
              if k in (0,1,10,50,100,250,500,1000):
                full_curve.append({"step":k,"loss":float(loss.detach()),"ev":[ev(Y[:,j].cpu().numpy(),pred[:,j].detach().cpu().numpy()) for j in range(4)]})
              if k<1000: opt2.zero_grad(set_to_none=True);loss.backward();opt2.step()
            report["A_fixed_fit"][lab][str(snap)]={
              "dataset_n":int(len(X)),"initial_ev":init_ev,
              "analytic_frozen_body_linear_head_ev":lin_ev,"analytic_frozen_body_linear_head_rmse":lin_rmse,
              "head_only_curve":head_curve,"full_critic_curve":full_curve,
            }
    
        # C6-C online drift: matched seed means state-target distributions across snapshots can be compared.
        for lab in ORDER:
          rows={}
          ref=datasets[lab][0]
          for snap in SNAPS:
            d=datasets[lab][snap]
            # Distribution drift in obs and target moments.
            obs_mean=d["obs"].numpy().mean((0,1));ref_mean=ref["obs"].numpy().mean((0,1))
            obs_std=d["obs"].numpy().std((0,1));ref_std=ref["obs"].numpy().std((0,1))
            target=d["MC_done"];rt=ref["MC_done"]
            rows[str(snap)]={
              "obs_mean_shift_l2_from_u0":float(np.linalg.norm(obs_mean-ref_mean)),
              "obs_std_shift_l2_from_u0":float(np.linalg.norm(obs_std-ref_std)),
              "mc_mean":[float(target[:,:,j].mean()) for j in range(4)],
              "mc_std":[float(target[:,:,j].std()) for j in range(4)],
              "mc_mean_shift_from_u0":[float(target[:,:,j].mean()-rt[:,:,j].mean()) for j in range(4)],
              "mc_std_ratio_to_u0":[float(target[:,:,j].std()/(rt[:,:,j].std()+1e-12)) for j in range(4)],
            }
          # Consecutive target drift summary
          cons={}
          for a,b in zip(SNAPS[:-1],SNAPS[1:]):
            da,db=datasets[lab][a]["MC_done"],datasets[lab][b]["MC_done"]
            cons[f"{a}->{b}"]={
              "mean_shift_l2":float(np.linalg.norm(db.mean((0,1))-da.mean((0,1)))),
              "std_shift_l2":float(np.linalg.norm(db.std((0,1))-da.std((0,1)))),
            }
          report["C_online_drift"][lab]={"snapshots":rows,"consecutive":cons}
    
        (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"status":report["status"],
          "timeouts":{k:{"term":v["termination_count"],"trunc":v["truncation_count"]} for k,v in list(report["B_target_validity"].items())[:8]},
          "fit_terminal":{lab:{s:report["A_fixed_fit"][lab][s]["full_critic_curve"][-1]["ev"] for s in ("50","100")} for lab in ORDER}},indent=2))
      except BaseException as e:
        (OUT/"ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
      finally:
        if env is not None:env.close()
        app.close()
    if True:main()

def run_post_v2_t5_c6_timeout_target_audit():
    """Run former post_v2_t5_c6_timeout_target_audit.py stage."""
    """What target does the critic get when an episode times out rather than fails?
    
    The implementation zeroes the bootstrap on termination and truncation alike.
    This records, for every episode end, both the target as written and the target
    a truncation-aware rule would give, along with the observation the env returns
    at the boundary, so the two can be compared without changing anything.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.isaac_audit import RUNS, IsaacAudit, obs_tensor
    
    SRC = "post_v2_t5_c5_h16-2026-09-23"
    PREFERENCE = np.array([.7, .1, .1, .1], np.float32)
    GAMMA = .99
    MIN_STEPS = 1100
    MAX_EVENTS = 64
    FIRST_EVENTS = 8
    
    
    def terms(raw, names):
        return {n: raw[:, i] for i, n in enumerate(names)}
    
    
    class TimeoutTargetAudit(IsaacAudit):
        """Critic target at episode boundaries, termination versus truncation."""
    
        run = "post_v2_t5_c6_critic_target_repr-2026-09-23"
        report = "timeout_target_audit.json"
        schema = "t5_c6_timeout_target_audit_v1"
        reset_seed = 700000
        set_usd_path = False       # this audit ran against Isaac's own A1 asset
    
        def rollout(self, env, obs):
            from talon_rl.rewards.objectives import normalized_objective_vector
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
    
            u = env.unwrapped
            ad = u.action_manager.total_action_dim
            m = T4SharedActorCritic(obs.shape[-1], ad).cuda()
            m.load_state_dict(torch.load(RUNS / SRC / "T_snap_100.pt", map_location="cuda",
                                         weights_only=False)["model"])
            m.eval()
            w = torch.tensor(PREFERENCE, device="cuda").repeat(self.num_envs, 1)
            mgr = u.reward_manager
            maxlen = int(getattr(u, "max_episode_length", -1))
            dt = float(u.step_dt)
    
            events = []
            for t in range(max(maxlen + 20, MIN_STEPS)):
                with torch.no_grad():
                    v = m.value_with_preference(obs, w)
                    a = m.act_inference_with_preference(obs, w)
                nxt, _, term, trunc, info = env.step(a)
                nxt = obs_tensor(nxt).cuda()
                raw = mgr._step_reward.detach().cpu().numpy()
                r = normalized_objective_vector(terms(raw, list(mgr.active_terms)),
                                                shape=(self.num_envs,)) * dt
                with torch.no_grad():
                    nv = m.value_with_preference(nxt, w)
                mask = (term | trunc).cpu().numpy()
                for i in np.where(mask)[0]:
                    # as implemented the bootstrap is zeroed for both cases
                    cur_target = r[i].copy()
                    alt_target = r[i] + GAMMA * nv[i].detach().cpu().numpy() * (not bool(term[i]))
                    events.append({
                        "t": t, "env": int(i), "term": bool(term[i]), "trunc": bool(trunc[i]),
                        "reward": r[i].tolist(),
                        "v_before": v[i].detach().cpu().tolist(),
                        "v_returned_nextobs": nv[i].detach().cpu().tolist(),
                        "current_target_done_zero": cur_target.tolist(),
                        "term_only_target_using_returned_obs": alt_target.tolist(),
                        "returned_obs_l2": float(nxt[i].norm()),
                        "pre_obs_l2": float(obs[i].norm()),
                        "obs_jump_l2": float((nxt[i] - obs[i]).norm()),
                        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else str(type(info))})
                obs = nxt
    
            out = {"schema": self.schema, "max_episode_length": maxlen, "step_dt": dt,
                   "event_count": len(events),
                   "term_count": sum(e["term"] for e in events),
                   "trunc_count": sum(e["trunc"] for e in events),
                   "events": events[:MAX_EVENTS]}
            self.write(out)
            print(json.dumps({"max_episode_length": maxlen, "event_count": len(events),
                              "term": out["term_count"], "trunc": out["trunc_count"],
                              "first_events": events[:FIRST_EVENTS]}, indent=2))
            return out
    
    
    if True:
        TimeoutTargetAudit.main()

def run_post_v2_t5_c7_endpoint_eval():
    """Run former post_v2_t5_c7_endpoint_eval.py stage."""
    import json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c7_critic_updates10-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      models={}
      for lab in ORDER:
        m=T4SharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_50.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
      rows=[]
      for suite in range(4):
        seed=320001+suite
        for lab in ORDER:
          m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((8,ad),device="cuda");norm=[];phys=[];done=np.zeros(8,bool)
          with torch.no_grad():
           for _ in range(64):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);norm.append(normalized_objective_vector(terms(raw,names),shape=(8,)).mean(0));data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            phys.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
            done|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
          n=np.asarray(norm).mean(0);pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
          rows.append({"suite":suite,"policy":lab,"normalized_objective_mean":n.tolist(),"scalarized":{e:float(n@PREFS[e]) for e in ORDER},"physical":pm,"survival":float(1-done.mean())})
      ow={};pw={};sw={};pk={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
      for i,lab in enumerate(ORDER):
        a=[];b=[];c=[]
        for suite in range(4):
          rr=[r for r in rows if r["suite"]==suite]
          a.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==lab)
          b.append((min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==lab) if lab=="T" else (min(rr,key=lambda r:r["physical"][pk[lab]])["policy"]==lab))
          c.append(max(rr,key=lambda r:r["scalarized"][lab])["policy"]==lab)
        ow[lab]=float(np.mean(a));pw[lab]=float(np.mean(b));sw[lab]=float(np.mean(c))
      vb={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in rows if r["policy"]==p])) for p in ORDER};vr={p:vb[p]/(vb["T"]+1e-12) for p in ORDER};mins=min(r["survival"] for r in rows)
      gates={"objective_winner_fraction":ow,"physical_winner_fraction":pw,"diagonal_scalarized_winner_fraction":sw,"min_survival":mins,"vertical_ratio_to_tracking_policy":vr,"safety_pass":bool(mins>=.95 and max(vr.values())<=2)}
      out={"schema":"t5_c7_updates10_endpoint_eval_v1","rows":rows,"gates":gates};(RUN/"endpoint_eval.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(gates,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c7_target_gap_audit():
    """Run former post_v2_t5_c7_target_gap_audit.py stage."""
    """Target gap for C5 at one critic update against C7 at ten."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.target_gap_audit import TargetGapAudit
    
    
    class C7TargetGapAudit(TargetGapAudit):
        """Target gap for C5 at one critic update against C7 at ten."""
    
        run = "post_v2_t5_c7_critic_updates10-2026-09-23"
        arms = {"C5_u1": ("post_v2_t5_c5_h16-2026-09-23", None),
                "C7_u10": ("post_v2_t5_c7_critic_updates10-2026-09-23", None)}
        seed_base = 800000
    
    
    if True:
        C7TargetGapAudit.main()

def run_post_v2_t5_c7_value_semantics_audit():
    """Run former post_v2_t5_c7_value_semantics_audit.py stage."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c7_critic_updates10-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3};SNAPS=(0,10,25,50);PERT=(10,25,50);H=(1,2,4,8,16,32);G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
        o=0
        with torch.no_grad():
          for p,b in zip(ps,base):
            n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
        rows=[];done_any=np.zeros(len(cur),bool)
        with torch.no_grad():
          for _ in range(32):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);data=robot.data
            rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
            done_any|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
        metrics={str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H}
        return metrics,float(1-done_any.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector,raw_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      critic={};pert=[]
      for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);critic[lab]={}
        for snap in SNAPS:
          cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
          cur,_=env.reset(seed=500000+bi*1000+snap);cur=obs_tensor(cur).cuda();vals=[];rews=[];dones=[]
          with torch.no_grad():
           for _ in range(32):
            vals.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rews.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
          R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals);MC=np.zeros_like(R);run=np.zeros_like(R[0])
          for t in range(31,-1,-1):run=R[t]+G*run*(~D[t])[:,None];MC[t]=run
          critic[lab][str(snap)]={"ev":[ev(MC[:,:,j].reshape(-1),V[:,:,j].reshape(-1)) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
        if lab in ("A","O"):
          j=IDX[lab]
          for snap in PERT:
            cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
            cur,_=env.reset(seed=510000+bi*1000+snap);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
            for _ in range(16):
              with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
              nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
              ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
            with torch.no_grad():nv=m.value_with_preference(cur,w)
            rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt)
            fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
            aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
            loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
            basevec=torch.cat([p.detach().reshape(-1) for p in aps]);stepnorm=1e-4*(basevec.norm()+1e-12);delta=-g/(g.norm()+1e-12)*stepnorm
            pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval();m.eval()
            metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
            for ss in range(2):
              seed=520000+bi*1000+snap*10+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
            pert.append({"branch":lab,"snapshot":snap,"grad_norm":float(g.norm()),"step_norm":float(stepnorm),"suites":suites})
      out={"schema":"t5_c7_updates10_value_semantics_audit_v1","critic":critic,"perturbations":pert}
      (RUN/"value_semantics_audit.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({"critic":critic,"perturbations":len(pert)},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c8_endpoint_eval():
    """Run former post_v2_t5_c8_endpoint_eval.py stage."""
    import json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c8_lambda1-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      models={}
      for lab in ORDER:
        m=T4SharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_50.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
      rows=[]
      for suite in range(4):
        seed=320001+suite
        for lab in ORDER:
          m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((8,ad),device="cuda");norm=[];phys=[];done=np.zeros(8,bool)
          with torch.no_grad():
           for _ in range(64):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);norm.append(normalized_objective_vector(terms(raw,names),shape=(8,)).mean(0));data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            phys.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
            done|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
          n=np.asarray(norm).mean(0);pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
          rows.append({"suite":suite,"policy":lab,"normalized_objective_mean":n.tolist(),"scalarized":{e:float(n@PREFS[e]) for e in ORDER},"physical":pm,"survival":float(1-done.mean())})
      ow={};pw={};sw={};pk={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
      for i,lab in enumerate(ORDER):
        a=[];b=[];c=[]
        for suite in range(4):
          rr=[r for r in rows if r["suite"]==suite]
          a.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==lab)
          b.append((min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==lab) if lab=="T" else (min(rr,key=lambda r:r["physical"][pk[lab]])["policy"]==lab))
          c.append(max(rr,key=lambda r:r["scalarized"][lab])["policy"]==lab)
        ow[lab]=float(np.mean(a));pw[lab]=float(np.mean(b));sw[lab]=float(np.mean(c))
      vb={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in rows if r["policy"]==p])) for p in ORDER};vr={p:vb[p]/(vb["T"]+1e-12) for p in ORDER};mins=min(r["survival"] for r in rows)
      gates={"objective_winner_fraction":ow,"physical_winner_fraction":pw,"diagonal_scalarized_winner_fraction":sw,"min_survival":mins,"vertical_ratio_to_tracking_policy":vr,"safety_pass":bool(mins>=.95 and max(vr.values())<=2)}
      out={"schema":"t5_c8_lambda1_endpoint_eval_v1","rows":rows,"gates":gates};(RUN/"endpoint_eval.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(gates,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c8_exact_h16_target_gap():
    """Run former post_v2_t5_c8_exact_h16_target_gap.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"C5_lam095":(ROOT/"runs/post_v2_t5_c5_h16-2026-09-23",.95),"C8_lam1":(ROOT/"runs/post_v2_t5_c8_lambda1-2026-09-23",1.0)}
    ORDER=("T","A","O","S");SNAPS=(10,25,50);G=.99;BLOCK=16
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def mc(R,D):
     out=np.zeros_like(R);run=np.zeros_like(R[0])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={}
      for tag,(run,lam) in RUNS.items():
       out[tag]={}
       for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);out[tag][lab]={}
        for snap in SNAPS:
         m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
         cur,_=env.reset(seed=810000+bi*1000+snap);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];NV=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);nxt=obs_tensor(nxt).cuda()
           raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());NV.append(m.value_with_preference(nxt,w).cpu().numpy());cur=nxt
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);NV=np.asarray(NV);MC=mc(R,D);RET=np.zeros_like(R)
         for st in range(0,64,BLOCK):
          en=st+BLOCK
          rt=torch.tensor(R[st:en],dtype=torch.float32,device="cuda")
          vt=torch.tensor(V[st:en],dtype=torch.float32,device="cuda")
          dt=torch.tensor(D[st:en],device="cuda")
          nxt=torch.tensor(NV[en-1],dtype=torch.float32,device="cuda")
          _,ret=vector_gae(rt,vt,nxt,dt,lam=lam);RET[st:en]=ret.cpu().numpy()
         out[tag][lab][str(snap)]={
           "exact_h16_target_mc_mae":[float(np.mean(np.abs(RET[:,:,j]-MC[:,:,j]))) for j in range(4)],
           "exact_h16_target_mc_bias":[float(np.mean(RET[:,:,j]-MC[:,:,j])) for j in range(4)]
         }
      p=ROOT/"runs/post_v2_t5_c8_lambda1-2026-09-23/exact_h16_target_gap.json";p.write_text(json.dumps(out,indent=2)+"\n")
      for s in SNAPS:
       for tag in RUNS:
        a=[]
        for lab in ORDER:a+=out[tag][lab][str(s)]["exact_h16_target_mc_mae"]
        print(s,tag,round(float(np.mean(a)),4))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c8_target_gap_audit():
    """Run former post_v2_t5_c8_target_gap_audit.py stage."""
    """Target gap at GAE lambda 0.95 against lambda 1."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.target_gap_audit import TargetGapAudit
    
    
    class C8TargetGapAudit(TargetGapAudit):
        """Target gap at GAE lambda 0.95 against lambda 1."""
    
        run = "post_v2_t5_c8_lambda1-2026-09-23"
        arms = {"C5_lam095": ("post_v2_t5_c5_h16-2026-09-23", .95),
                "C8_lam1": ("post_v2_t5_c8_lambda1-2026-09-23", 1.0)}
        seed_base = 800000
    
    
    if True:
        C8TargetGapAudit.main()

def run_post_v2_t5_c8_value_semantics_audit():
    """Run former post_v2_t5_c8_value_semantics_audit.py stage."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c8_lambda1-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3};SNAPS=(0,10,25,50);PERT=(10,25,50);H=(1,2,4,8,16,32);G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
        o=0
        with torch.no_grad():
          for p,b in zip(ps,base):
            n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
        rows=[];done_any=np.zeros(len(cur),bool)
        with torch.no_grad():
          for _ in range(32):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);data=robot.data
            rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
            done_any|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
        metrics={str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H}
        return metrics,float(1-done_any.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector,raw_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      critic={};pert=[]
      for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);critic[lab]={}
        for snap in SNAPS:
          cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
          cur,_=env.reset(seed=500000+bi*1000+snap);cur=obs_tensor(cur).cuda();vals=[];rews=[];dones=[]
          with torch.no_grad():
           for _ in range(32):
            vals.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rews.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
          R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals);MC=np.zeros_like(R);run=np.zeros_like(R[0])
          for t in range(31,-1,-1):run=R[t]+G*run*(~D[t])[:,None];MC[t]=run
          critic[lab][str(snap)]={"ev":[ev(MC[:,:,j].reshape(-1),V[:,:,j].reshape(-1)) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
        if lab in ("A","O"):
          j=IDX[lab]
          for snap in PERT:
            cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
            cur,_=env.reset(seed=510000+bi*1000+snap);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
            for _ in range(16):
              with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
              nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
              ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
            with torch.no_grad():nv=m.value_with_preference(cur,w)
            rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=1.0)
            fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
            aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
            loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
            basevec=torch.cat([p.detach().reshape(-1) for p in aps]);stepnorm=1e-4*(basevec.norm()+1e-12);delta=-g/(g.norm()+1e-12)*stepnorm
            pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval();m.eval()
            metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
            for ss in range(2):
              seed=520000+bi*1000+snap*10+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
            pert.append({"branch":lab,"snapshot":snap,"grad_norm":float(g.norm()),"step_norm":float(stepnorm),"suites":suites})
      out={"schema":"t5_c8_lambda1_value_semantics_audit_v1","critic":critic,"perturbations":pert}
      (RUN/"value_semantics_audit.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({"critic":critic,"perturbations":len(pert)},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c9_endpoint_eval():
    """Run former post_v2_t5_c9_endpoint_eval.py stage."""
    import json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c9_delayed_target10-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      models={}
      for lab in ORDER:
        m=T4SharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_50.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
      rows=[]
      for suite in range(4):
        seed=320001+suite
        for lab in ORDER:
          m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((8,ad),device="cuda");norm=[];phys=[];done=np.zeros(8,bool)
          with torch.no_grad():
           for _ in range(64):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);norm.append(normalized_objective_vector(terms(raw,names),shape=(8,)).mean(0));data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            phys.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
            done|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
          n=np.asarray(norm).mean(0);pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
          rows.append({"suite":suite,"policy":lab,"normalized_objective_mean":n.tolist(),"scalarized":{e:float(n@PREFS[e]) for e in ORDER},"physical":pm,"survival":float(1-done.mean())})
      ow={};pw={};sw={};pk={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
      for i,lab in enumerate(ORDER):
        a=[];b=[];c=[]
        for suite in range(4):
          rr=[r for r in rows if r["suite"]==suite]
          a.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==lab)
          b.append((min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==lab) if lab=="T" else (min(rr,key=lambda r:r["physical"][pk[lab]])["policy"]==lab))
          c.append(max(rr,key=lambda r:r["scalarized"][lab])["policy"]==lab)
        ow[lab]=float(np.mean(a));pw[lab]=float(np.mean(b));sw[lab]=float(np.mean(c))
      vb={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in rows if r["policy"]==p])) for p in ORDER};vr={p:vb[p]/(vb["T"]+1e-12) for p in ORDER};mins=min(r["survival"] for r in rows)
      gates={"objective_winner_fraction":ow,"physical_winner_fraction":pw,"diagonal_scalarized_winner_fraction":sw,"min_survival":mins,"vertical_ratio_to_tracking_policy":vr,"safety_pass":bool(mins>=.95 and max(vr.values())<=2)}
      out={"schema":"t5_c9_delayed_target10_endpoint_eval_v1","rows":rows,"gates":gates};(RUN/"endpoint_eval.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(gates,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c9_exact_h16_target_gap():
    """Run former post_v2_t5_c9_exact_h16_target_gap.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"C5_online":(ROOT/"runs/post_v2_t5_c5_h16-2026-09-23","online"),"C9_delayed10":(ROOT/"runs/post_v2_t5_c9_delayed_target10-2026-09-23","target")}
    ORDER=("T","A","O","S");SNAPS=(10,25,50);G=.99;BLOCK=16
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def mc(R,D):
     out=np.zeros_like(R);run=np.zeros_like(R[0])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={}
      for tag,(run,mode) in RUNS.items():
       out[tag]={}
       for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);out[tag][lab]={}
        for snap in SNAPS:
         payload=torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)
         m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(payload["model"]);m.eval()
         bm=m
         if mode=="target":
          bm=T4SharedActorCritic(od,ad).cuda();bm.load_state_dict(payload["target_model"]);bm.eval()
         cur,_=env.reset(seed=810000+bi*1000+snap);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];NV=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(bm.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);nxt=obs_tensor(nxt).cuda()
           raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());NV.append(bm.value_with_preference(nxt,w).cpu().numpy());cur=nxt
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);NV=np.asarray(NV);MC=mc(R,D);RET=np.zeros_like(R)
         for st in range(0,64,BLOCK):
          en=st+BLOCK
          rt=torch.tensor(R[st:en],dtype=torch.float32,device="cuda")
          vt=torch.tensor(V[st:en],dtype=torch.float32,device="cuda")
          dt=torch.tensor(D[st:en],device="cuda")
          nxt=torch.tensor(NV[en-1],dtype=torch.float32,device="cuda")
          _,ret=vector_gae(rt,vt,nxt,dt,lam=.95);RET[st:en]=ret.cpu().numpy()
         out[tag][lab][str(snap)]={
           "exact_h16_target_mc_mae":[float(np.mean(np.abs(RET[:,:,j]-MC[:,:,j]))) for j in range(4)],
           "exact_h16_target_mc_bias":[float(np.mean(RET[:,:,j]-MC[:,:,j])) for j in range(4)]
         }
      p=ROOT/"runs/post_v2_t5_c9_delayed_target10-2026-09-23/exact_h16_target_gap.json";p.write_text(json.dumps(out,indent=2)+"\n")
      for s in SNAPS:
       for tag in RUNS:
        a=[]
        for lab in ORDER:a+=out[tag][lab][str(s)]["exact_h16_target_mc_mae"]
        print(s,tag,round(float(np.mean(a)),4))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c9_value_semantics_audit():
    """Run former post_v2_t5_c9_value_semantics_audit.py stage."""
    import json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c9_delayed_target10-2026-09-23"
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3};SNAPS=(0,10,25,50);PERT=(10,25,50);H=(1,2,4,8,16,32);G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
        o=0
        with torch.no_grad():
          for p,b in zip(ps,base):
            n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
        rows=[];done_any=np.zeros(len(cur),bool)
        with torch.no_grad():
          for _ in range(32):
            a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);data=robot.data
            rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
            done_any|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
        metrics={str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H}
        return metrics,float(1-done_any.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector,raw_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      critic={};pert=[]
      for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);critic[lab]={}
        for snap in SNAPS:
          cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
          cur,_=env.reset(seed=500000+bi*1000+snap);cur=obs_tensor(cur).cuda();vals=[];rews=[];dones=[]
          with torch.no_grad():
           for _ in range(32):
            vals.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rews.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
          R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals);MC=np.zeros_like(R);run=np.zeros_like(R[0])
          for t in range(31,-1,-1):run=R[t]+G*run*(~D[t])[:,None];MC[t]=run
          critic[lab][str(snap)]={"ev":[ev(MC[:,:,j].reshape(-1),V[:,:,j].reshape(-1)) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
        if lab in ("A","O"):
          j=IDX[lab]
          for snap in PERT:
            cp=RUN/f"{lab}_snap_{snap}.pt";payload=torch.load(cp,map_location="cuda",weights_only=False)
            m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(payload["model"])
            tm=T4SharedActorCritic(od,ad).cuda();tm.load_state_dict(payload["target_model"]);tm.eval()
            cur,_=env.reset(seed=510000+bi*1000+snap);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];tval=[];dn=[]
            for _ in range(16):
              with torch.no_grad():
                a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w);tv=tm.value_with_preference(cur,w)
              nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
              ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);tval.append(tv);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
            with torch.no_grad():tnv=tm.value_with_preference(cur,w)
            rt=torch.stack(rw);vt=torch.stack(val);tvt=torch.stack(tval);dt=torch.stack(dn).bool();_,ret=vector_gae(rt,tvt,tnv,dt,lam=.95);adv=ret-vt
            fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
            aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
            loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
            basevec=torch.cat([p.detach().reshape(-1) for p in aps]);stepnorm=1e-4*(basevec.norm()+1e-12);delta=-g/(g.norm()+1e-12)*stepnorm
            pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval();m.eval()
            metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
            for ss in range(2):
              seed=520000+bi*1000+snap*10+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
            pert.append({"branch":lab,"snapshot":snap,"grad_norm":float(g.norm()),"step_norm":float(stepnorm),"suites":suites})
      out={"schema":"t5_c9_delayed_target10_value_semantics_audit_v1","critic":critic,"perturbations":pert}
      (RUN/"value_semantics_audit.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({"critic":critic,"perturbations":len(pert)},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

STAGES = {
    "post_v2_t5_c0_zero_critic_audit": run_post_v2_t5_c0_zero_critic_audit,
    "post_v2_t5_c2_gradient_physical_audit": run_post_v2_t5_c2_gradient_physical_audit,
    "post_v2_t5_c3_optimizer_counterfactual": run_post_v2_t5_c3_optimizer_counterfactual,
    "post_v2_t5_c3_value_learning_diagnosis": run_post_v2_t5_c3_value_learning_diagnosis,
    "post_v2_t5_c4_samebatch_check": run_post_v2_t5_c4_samebatch_check,
    "post_v2_t5_c4_value_semantics_audit": run_post_v2_t5_c4_value_semantics_audit,
    "post_v2_t5_c5_h16_endpoint_eval": run_post_v2_t5_c5_h16_endpoint_eval,
    "post_v2_t5_c5_h16_value_endpoint_audit": run_post_v2_t5_c5_h16_value_endpoint_audit,
    "post_v2_t5_c5_h16_value_semantics_audit": run_post_v2_t5_c5_h16_value_semantics_audit,
    "post_v2_t5_c6_critic_target_representation_audit": run_post_v2_t5_c6_critic_target_representation_audit,
    "post_v2_t5_c6_timeout_target_audit": run_post_v2_t5_c6_timeout_target_audit,
    "post_v2_t5_c7_endpoint_eval": run_post_v2_t5_c7_endpoint_eval,
    "post_v2_t5_c7_target_gap_audit": run_post_v2_t5_c7_target_gap_audit,
    "post_v2_t5_c7_value_semantics_audit": run_post_v2_t5_c7_value_semantics_audit,
    "post_v2_t5_c8_endpoint_eval": run_post_v2_t5_c8_endpoint_eval,
    "post_v2_t5_c8_exact_h16_target_gap": run_post_v2_t5_c8_exact_h16_target_gap,
    "post_v2_t5_c8_target_gap_audit": run_post_v2_t5_c8_target_gap_audit,
    "post_v2_t5_c8_value_semantics_audit": run_post_v2_t5_c8_value_semantics_audit,
    "post_v2_t5_c9_endpoint_eval": run_post_v2_t5_c9_endpoint_eval,
    "post_v2_t5_c9_exact_h16_target_gap": run_post_v2_t5_c9_exact_h16_target_gap,
    "post_v2_t5_c9_value_semantics_audit": run_post_v2_t5_c9_value_semantics_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
