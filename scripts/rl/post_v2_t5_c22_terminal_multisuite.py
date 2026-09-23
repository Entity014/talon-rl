#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUNS={"consecutive12":ROOT/"runs/post_v2_t5_c21_recent12warm-2026-09-23","diverse12":ROOT/"runs/post_v2_t5_c22_diverse12-2026-09-23"}
ORDER=("T","A","O","S");G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
def ret(R,D,seg=None):
 out=np.zeros_like(R)
 if seg is None:
  run=np.zeros_like(R[0])
  for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
 else:
  for st in range(0,len(R),seg):
   en=min(st+seg,len(R));run=np.zeros_like(R[0])
   for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
 return out
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={"schema":"t5_c22_terminal_multisuite_v1","arms":{}}
  for arm,run in RUNS.items():
   ar={}
   for bi,lab in enumerate(ORDER):
    m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
    w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);suites=[]
    for si in range(4):
     seed=1410000+bi*1000+si*97;cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];F=[];C=[]
     with torch.no_grad():
      for t in range(64):
       feat=m.critic_body(m._with_w(cur,w));V.append(m.critic_head(feat).cpu().numpy())
       if t<32:F.append(feat.cpu().numpy());C.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
     R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None);FF=np.concatenate(F,0);CC=np.concatenate(C,0)
     # first H32 segment target summary to match training H32 segments
     Y0=H32[:32].reshape(-1,4)
     suites.append({"suite":si,"seed":seed,"h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
       "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean()),
       "fresh_summary":np.r_[CC.mean(0),CC.std(0),FF.mean(0),FF.std(0)].tolist(),
       "target_summary":np.r_[Y0.mean(0),Y0.std(0)].tolist()})
    ar[lab]=suites
   out["arms"][arm]=ar
  agg={}
  for arm in RUNS:
   e=[];mc=[];b=[];sv=[];by=[[] for _ in range(4)]
   for lab in ORDER:
    for s in out["arms"][arm][lab]:
     e+=s["h32_ev"];mc+=s["mc64_ev"];b+=s["h32_bias"];sv.append(s["survival"])
     for j,x in enumerate(s["h32_ev"]):by[j].append(x)
   agg[arm]={"h32_ev_mean":float(np.mean(e)),"h32_ev_std":float(np.std(e)),"h32_negative_fraction":float(np.mean(np.array(e)<0)),
    "h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(mc)),"mc64_negative_fraction":float(np.mean(np.array(mc)<0)),
    "h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
  out["aggregate"]=agg
  p=RUNS["diverse12"]/"terminal_multisuite.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
