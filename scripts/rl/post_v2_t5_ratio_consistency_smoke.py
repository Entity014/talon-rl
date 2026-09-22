#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def main():
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
    try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=350001);obs=obs_tensor(obs).cuda();m=T4SharedActorCritic(obs.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda();initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt")
      w=torch.tensor([.7,.1,.1,.1],device="cuda").repeat(len(obs),1)
      rows=[]
      for _ in range(64):
        with torch.no_grad():
          a,old=m.act_with_preference(obs,w)
          clipped=torch.clamp(a,-1,1)
          lp_unclip=m.logp_with_preference(obs,w,a)
          lp_clip=m.logp_with_preference(obs,w,clipped)
          r_unclip=torch.exp(lp_unclip-old);r_clip=torch.exp(lp_clip-old)
        rows.append({
          "clip_fraction":float((a.abs()>1).float().mean()),
          "max_action":float(a.abs().max()),
          "ratio_unclip_mean":float(r_unclip.mean()),"ratio_unclip_maxerr":float((r_unclip-1).abs().max()),
          "ratio_clip_mean":float(r_clip.mean()),"ratio_clip_std":float(r_clip.std()),"ratio_clip_min":float(r_clip.min()),"ratio_clip_max":float(r_clip.max()),
          "logp_shift_mean":float((lp_clip-old).mean()),"logp_shift_abs_mean":float((lp_clip-old).abs().mean()),
        })
        obs,*_=env.step(clipped);obs=obs_tensor(obs).cuda()
      keys=rows[0]
      out={"schema":"t5_ratio_consistency_smoke_v1","ACTION_CLIP":m.ACTION_CLIP,
           "mean":{k:float(np.mean([r[k] for r in rows])) for k in keys},
           "max":{k:float(np.max([r[k] for r in rows])) for k in keys},
           "rows":rows}
      p=ROOT/"runs/post_v2_t5_gradient_separability-2026-09-23/ratio_smoke.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"mean":out["mean"],"max":out["max"]},indent=2))
    finally:
      if env is not None:env.close()
      app.close()
if __name__=="__main__":main()
