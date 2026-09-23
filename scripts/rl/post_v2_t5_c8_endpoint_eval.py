#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
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
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  from talon_rl.t3b_objectives import normalized_objective_vector
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
if __name__=="__main__":main()
