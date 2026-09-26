#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23"
exec(open(ROOT/"scripts/rl/experiments/post_v2/post_v2_t5_c25_eval.py").read().split("def main():")[0])
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
  cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={}
  for lab in ("A","O"):
   bi=ORDER.index(lab);j=IDX[lab];m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_10.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
   w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);g=grad_perturb(env,m,w,mgr,j,2810000+bi*1000)
   aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
   pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_10.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,[p.detach().clone() for p in paps],delta);pm.eval();metric="ang_vel_xy" if lab=="A" else "tilt_deg";rows=[]
   for ss in range(8):
    seed=2910000+bi*1000+ss*137;b,sb=phys_roll(env,m,w,seed);p,sp=phys_roll(env,pm,w,seed);rows.append({"suite":ss,"delta":p[metric]-b[metric],"base_survival":sb,"pert_survival":sp})
   ds=[x["delta"] for x in rows];out[lab]={"metric":metric,"mean_delta":float(np.mean(ds)),"median_delta":float(np.median(ds)),"correct_fraction":float(np.mean(np.array(ds)<0)),"rows":rows}
  (RUN/"u10_perturb_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
