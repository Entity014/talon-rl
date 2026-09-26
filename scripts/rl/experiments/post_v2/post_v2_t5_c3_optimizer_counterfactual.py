#!/usr/bin/env python3
from __future__ import annotations
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
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  from talon_rl.t3b_objectives import normalized_objective_vector
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
if __name__=="__main__":main()
