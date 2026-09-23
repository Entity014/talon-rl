#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
OUT=ROOT/"runs/post_v2_t5_c20_coverage-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
ORDER=("T","A","O","S");G=.99;H=32;NTRAIN=16;NTEST=8
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def mc(R,D):
 out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
 for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
 return out
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=16;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  meta={"schema":"c20_pool_v1","horizon":H,"ntrain":NTRAIN,"ntest":NTEST,"num_envs":16,"specialists":{}}
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(od,ad).cuda();initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero");m.eval()
   w=torch.tensor(PREFS[lab],device="cuda").repeat(16,1)
   arrays={};rows=[]
   for split,nroll,base_seed in (("train",NTRAIN,710000+bi*10000),("test",NTEST,910000+bi*10000)):
    splitrows=[]
    for r in range(nroll):
     cur,_=env.reset(seed=base_seed+r*97);cur=obs_tensor(cur).cuda();obs=[];R=[];D=[];cmds=[]
     with torch.no_grad():
      for _ in range(H):
       obs.append(cur);cmds.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(16,))
       R.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);D.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
     X=torch.cat(obs);Wp=w.repeat(H,1);Y=mc(torch.stack(R),torch.stack(D).bool()).reshape(-1,4)
     with torch.no_grad():F=m.critic_body(m._with_w(X,Wp))
     C=np.concatenate(cmds,0)
     arrays[f"{split}_F{r}"]=F.cpu().numpy();arrays[f"{split}_Y{r}"]=Y.cpu().numpy()
     summary=np.r_[C.mean(0),C.std(0),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),Y.mean(0).cpu().numpy(),Y.std(0).cpu().numpy()]
     arrays[f"{split}_S{r}"]=summary.astype(np.float32)
     splitrows.append({"rollout":r,"seed":base_seed+r*97,"command_mean":C.mean(0).tolist(),"command_std":C.std(0).tolist(),
                       "feature_mean_norm":float(F.mean(0).norm()),"feature_std_norm":float(F.std(0).norm()),
                       "target_mean":Y.mean(0).cpu().tolist(),"target_std":Y.std(0).cpu().tolist()})
    rows.append((split,splitrows))
   np.savez_compressed(OUT/f"{lab}_pool.npz",**arrays)
   meta["specialists"][lab]={k:v for k,v in rows}
   print(lab,"saved",flush=True)
  (OUT/"pool_meta.json").write_text(json.dumps(meta,indent=2)+"\n")
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
