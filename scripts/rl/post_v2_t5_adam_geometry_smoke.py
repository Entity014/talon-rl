#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
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
  from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector
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
if __name__=="__main__":main()
