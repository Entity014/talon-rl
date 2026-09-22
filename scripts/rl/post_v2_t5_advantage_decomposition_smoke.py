#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
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
  from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector
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
if __name__=="__main__":main()
