#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c5_h16-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c6_critic_target_repr-2026-09-23"
P=np.array([.7,.1,.1,.1],np.float32);G=.99
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  obs,_=env.reset(seed=700000);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
  m=T4SharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(RUN/"T_snap_100.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
  w=torch.tensor(P,device="cuda").repeat(8,1);mgr=env.unwrapped.reward_manager
  maxlen=int(getattr(env.unwrapped,"max_episode_length",-1));dt=float(env.unwrapped.step_dt)
  events=[]; prev_obs=obs.clone(); prev_v=None
  for t in range(max(maxlen+20,1100)):
   with torch.no_grad():
    v=m.value_with_preference(obs,w);a=m.act_inference_with_preference(obs,w)
   nxt,_,term,trunc,info=env.step(a);nxt=obs_tensor(nxt).cuda()
   raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);r=normalized_objective_vector(terms(raw,names),shape=(8,))*dt
   with torch.no_grad():nv=m.value_with_preference(nxt,w)
   mask=(term|trunc).cpu().numpy()
   if mask.any():
    for i in np.where(mask)[0]:
      # Current implementation zeros bootstrap on both term and trunc.
      cur_target=r[i].copy()
      alt_target=r[i]+G*nv[i].detach().cpu().numpy()*(not bool(term[i]))
      events.append({
       "t":t,"env":int(i),"term":bool(term[i]),"trunc":bool(trunc[i]),
       "reward":r[i].tolist(),"v_before":v[i].detach().cpu().tolist(),"v_returned_nextobs":nv[i].detach().cpu().tolist(),
       "current_target_done_zero":cur_target.tolist(),"term_only_target_using_returned_obs":alt_target.tolist(),
       "returned_obs_l2":float(nxt[i].norm()),"pre_obs_l2":float(obs[i].norm()),
       "obs_jump_l2":float((nxt[i]-obs[i]).norm()),
       "info_keys":sorted(list(info.keys())) if isinstance(info,dict) else str(type(info))
      })
   obs=nxt
  out={"schema":"t5_c6_timeout_target_audit_v1","max_episode_length":maxlen,"step_dt":dt,
       "event_count":len(events),"term_count":sum(e["term"] for e in events),"trunc_count":sum(e["trunc"] for e in events),
       "events":events[:64]}
  (OUT/"timeout_target_audit.json").write_text(json.dumps(out,indent=2)+"\n")
  print(json.dumps({"max_episode_length":maxlen,"event_count":len(events),"term":out["term_count"],"trunc":out["trunc_count"],"first_events":events[:8]},indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
