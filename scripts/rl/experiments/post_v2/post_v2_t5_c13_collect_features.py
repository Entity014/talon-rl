#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
ORDER=("T","A","O","S");NB=6;H=32;G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def trunc_mc(R,D):
 out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
 for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
 return out
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  from talon_rl.t3b_objectives import normalized_objective_vector
  OUT.mkdir(parents=True,exist_ok=True)
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
   w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=996000+bi*1000);cur=obs_tensor(cur).cuda()
   data={}
   for k in range(NB):
    obs=[];rw=[];dn=[]
    with torch.no_grad():
     for _ in range(H):
      obs.append(cur);a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
      raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
      rw.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
    X=torch.cat(obs);Wp=w.repeat(H,1);Y=trunc_mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
    with torch.no_grad():F=m.critic_body(m._with_w(X,Wp))
    data[f"F{k}"]=F.cpu().numpy();data[f"Y{k}"]=Y.cpu().numpy()
   data["head_weight"]=m.critic_head.weight.detach().cpu().numpy();data["head_bias"]=m.critic_head.bias.detach().cpu().numpy()
   np.savez_compressed(OUT/f"{lab}_features_targets.npz",**data)
   print(lab,"saved",data["F0"].shape,data["Y0"].shape,flush=True)
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
