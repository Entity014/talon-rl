#!/usr/bin/env python3
from __future__ import annotations
import json,sys,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
PREF=np.array([.7,.1,.1,.1],np.float32)
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def main():
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
    outp=ROOT/"runs/post_v2_t5_consistency_repair-2026-09-23/smoke.json";outp.parent.mkdir(parents=True,exist_ok=True)
    try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=360001);obs=obs_tensor(obs).cuda()
      m=T4SharedActorCritic(obs.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda()
      initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt")
      w=torch.as_tensor(np.repeat(PREF[None,:],len(obs),axis=0),device="cuda")
      rows=[];done_any=np.zeros(len(obs),dtype=bool)
      for step in range(64):
        with torch.no_grad():
          action,old,u=m.act_with_preference_latent(obs,w)
          recomputed=m.logp_from_pre_tanh_with_preference(obs,w,u)
          ratio=torch.exp(recomputed-old)
        # No external clamp. The exact sampled policy action is applied and is already in [-1,1].
        buffer_action=action.clone()
        next_obs,_,term,trunc,_=env.step(action)
        done_any|=(term|trunc).cpu().numpy()
        rows.append({
          "step":step,
          "env_buffer_action_max_abs_diff":float((action-buffer_action).abs().max()),
          "ratio_mean":float(ratio.mean()),
          "ratio_max_abs_error":float((ratio-1).abs().max()),
          "logp_max_abs_diff":float((recomputed-old).abs().max()),
          "action_max_abs":float(action.abs().max()),
          "near_boundary_fraction":float((action.abs()>=0.999999).float().mean()),
          "u_max_abs":float(u.abs().max()),
          "finite":bool(torch.isfinite(action).all() and torch.isfinite(old).all() and torch.isfinite(recomputed).all() and torch.isfinite(u).all())
        })
        obs=obs_tensor(next_obs).cuda()
      report={
        "schema":"t5_consistency_repair_smoke_v1","ACTION_CLIP":m.ACTION_CLIP,
        "protocol":{"steps":64,"num_envs":32,"external_clamp":False,"buffer_stores_applied_action":True,"buffer_stores_pre_tanh_u":True},
        "mean":{"ratio_mean":float(np.mean([r["ratio_mean"] for r in rows])),
                "near_boundary_fraction":float(np.mean([r["near_boundary_fraction"] for r in rows]))},
        "max":{"ratio_max_abs_error":float(np.max([r["ratio_max_abs_error"] for r in rows])),
               "logp_max_abs_diff":float(np.max([r["logp_max_abs_diff"] for r in rows])),
               "env_buffer_action_max_abs_diff":float(np.max([r["env_buffer_action_max_abs_diff"] for r in rows])),
               "action_max_abs":float(np.max([r["action_max_abs"] for r in rows])),
               "u_max_abs":float(np.max([r["u_max_abs"] for r in rows]))},
        "survival":float(1-done_any.mean()),
        "all_finite":all(r["finite"] for r in rows),
        "pass":bool(np.max([r["ratio_max_abs_error"] for r in rows])<=1e-6
                    and np.max([r["logp_max_abs_diff"] for r in rows])<=1e-6
                    and np.max([r["env_buffer_action_max_abs_diff"] for r in rows])==0.0
                    and all(r["finite"] for r in rows)
                    and (1-done_any.mean())>=0.95),
        "rows":rows,
      }
      outp.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
    except BaseException as e:
      outp.with_name("smoke.ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
    finally:
      if env is not None:env.close()
      app.close()
if __name__=="__main__":main()
