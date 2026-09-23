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
 from talon_rl.t4_actor_critic import T4SharedActorCritic
 from talon_rl.t3b_objectives import normalized_objective_vector
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
