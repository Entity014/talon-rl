#!/usr/bin/env python3
from __future__ import annotations
import json,sys,hashlib,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
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
      from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae
      from talon_rl.t3b_objectives import normalized_objective_vector
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
if __name__=="__main__":main()
