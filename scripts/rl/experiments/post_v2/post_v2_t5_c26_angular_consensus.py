#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c26_angular_credit-2026-09-23"
exec(open(ROOT/"scripts/rl/experiments/post_v2/post_v2_t5_c25_eval.py").read().split("def main():")[0])
SCALES=(0.0,1.0,2.0,4.0)
def rollout_full(env,m,w,mgr,seed):
 from talon_rl.t3b_objectives import normalized_objective_vector
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];ang=[];obj=[];done=np.zeros(NENV,bool)
 with torch.no_grad():
  for _ in range(32):
   a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
   vec=normalized_objective_vector(terms(raw,names),shape=(NENV,));obj.append(vec[:,1].mean())
   ang.append(float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[:,:2],dim=-1).mean()))
   done|=(te|tr).cpu().numpy();cur=obs_tensor(nxt).cuda()
 return {"ang_vel_xy":float(np.mean(ang)),"angular_objective":float(np.mean(obj)),"survival":float(1-done.mean())}
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={"schema":"c26_angular_consensus_v1","snapshots":{}}
  for snap in (10,25):
   m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
   gs=[grad_perturb(env,m,w,mgr,1,3210000+snap*10000+i*211) for i in range(8)]
   cos=[]
   for i in range(8):
    for j in range(i+1,8):cos.append(float(torch.dot(gs[i],gs[j])/(gs[i].norm()*gs[j].norm()+1e-12)))
   g=torch.stack(gs).mean(0);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base]);unit=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
   models={}
   for sc in SCALES:
    pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"])
    paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")]
    if sc>0:setflat(paps,[p.detach().clone() for p in paps],unit*sc)
    pm.eval();models[sc]=pm
   suites=[]
   for ss in range(8):
    seed=3310000+snap*10000+ss*149;row={"suite":ss}
    for sc in SCALES:row[str(sc)]=rollout_full(env,models[sc],w,mgr,seed)
    suites.append(row)
   summary={}
   for sc in SCALES:
    if sc==0:continue
    da=[];dr=[]
    for row in suites:
     da.append(row[str(sc)]["ang_vel_xy"]-row["0.0"]["ang_vel_xy"]);dr.append(row[str(sc)]["angular_objective"]-row["0.0"]["angular_objective"])
    summary[str(sc)]={"delta_ang_mean":float(np.mean(da)),"delta_ang_median":float(np.median(da)),"physical_correct_fraction":float(np.mean(np.array(da)<0)),
      "delta_objective_mean":float(np.mean(dr)),"objective_correct_fraction":float(np.mean(np.array(dr)>0))}
   out["snapshots"][str(snap)]={"gradient_batch_pairwise_cosine_mean":float(np.mean(cos)),"gradient_batch_pairwise_cosine_std":float(np.std(cos)),
    "gradient_negative_pair_fraction":float(np.mean(np.array(cos)<0)),"mean_gradient_norm":float(g.norm()),"summary":summary}
  (OUT/"consensus.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
