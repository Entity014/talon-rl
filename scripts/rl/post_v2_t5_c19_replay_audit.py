#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUNS={"frozen":ROOT/"runs/post_v2_t5_c19_actor_frozen-2026-09-23","moving":ROOT/"runs/post_v2_t5_c19_actor_moving-2026-09-23"}
ORDER=("T","A","O","S");SNAPS=tuple(range(26));G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
 return float(1-np.var(y-p)/(np.var(y)+1e-12))
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
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={"schema":"t5_c19_replay_audit_v1","snapshots":list(SNAPS),"arms":{}}
  for arm,run in RUNS.items():
   armout={}
   for bi,lab in enumerate(ORDER):
    w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);rows=[];base_pred=None;base_feat_mean=None;base_feat_std=None;base_action_mean=None;base_action_std=None
    for snap in SNAPS:
     m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
     cur,_=env.reset(seed=610000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];F=[];A=[]
     with torch.no_grad():
      for _ in range(64):
       fw=w;mfeat=m.critic_body(m._with_w(cur,fw));v=m.critic_head(mfeat);a=m.act_inference_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());V.append(v.cpu().numpy());F.append(mfeat.cpu().numpy());A.append(a.cpu().numpy());cur=obs_tensor(nxt).cuda()
     R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);F=np.concatenate(F,0);A=np.concatenate(A,0);H32=ret(R,D,32);MC=ret(R,D,None)
     fm=F.mean(0);fs=F.std(0);am=A.mean(0);ast=A.std(0);pred=V.reshape(-1,4)
     if base_pred is None:base_pred=pred.copy();base_feat_mean=fm.copy();base_feat_std=fs.copy();base_action_mean=am.copy();base_action_std=ast.copy()
     rows.append({
      "snapshot":snap,
      "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
      "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
      "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
      "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)],
      "feature_mean_drift_from_u0":float(np.linalg.norm(fm-base_feat_mean)),
      "feature_std_drift_from_u0":float(np.linalg.norm(fs-base_feat_std)),
      "action_mean_drift_from_u0":float(np.linalg.norm(am-base_action_mean)),
      "action_std_drift_from_u0":float(np.linalg.norm(ast-base_action_std)),
      "prediction_drift_from_u0":float(np.linalg.norm(pred-base_pred)/(np.linalg.norm(base_pred)+1e-12)),
      "survival":float(1-D.any(0).mean())
     })
    armout[lab]=rows
   out["arms"][arm]=armout
  # paired aggregate trajectories
  agg={}
  for arm in RUNS:
   traj=[]
   for snap in SNAPS:
    h=[];mc=[];hb=[];mb=[];fd=[];fs=[];ad=[];asd=[];pd=[];sv=[]
    for lab in ORDER:
     r=out["arms"][arm][lab][snap];h+=r["h32_ev"];mc+=r["mc64_ev"];hb+=r["h32_bias"];mb+=r["mc64_bias"];fd.append(r["feature_mean_drift_from_u0"]);fs.append(r["feature_std_drift_from_u0"]);ad.append(r["action_mean_drift_from_u0"]);asd.append(r["action_std_drift_from_u0"]);pd.append(r["prediction_drift_from_u0"]);sv.append(r["survival"])
    traj.append({"snapshot":snap,"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),
                 "mc64_ev_mean":float(np.mean(mc)),"mc64_negative_fraction":float(np.mean(np.array(mc)<0)),
                 "h32_mean_abs_bias":float(np.mean(np.abs(hb))),"mc64_mean_abs_bias":float(np.mean(np.abs(mb))),
                 "feature_mean_drift_mean":float(np.mean(fd)),"feature_std_drift_mean":float(np.mean(fs)),
                 "action_mean_drift_mean":float(np.mean(ad)),"action_std_drift_mean":float(np.mean(asd)),
                 "prediction_drift_mean":float(np.mean(pd)),"min_survival":float(np.min(sv))})
   agg[arm]=traj
  out["aggregate"]=agg
  target=RUNS["moving"]/"replay_audit.json";target.write_text(json.dumps(out,indent=2)+"\n")
  for arm in ("frozen","moving"):
   print("\\n",arm)
   for s in (0,1,2,3,5,10,15,20,25):
    r=agg[arm][s];print(s,"H32",round(r["h32_ev_mean"],3),"MC",round(r["mc64_ev_mean"],3),"feat",round(r["feature_mean_drift_mean"],3),"act",round(r["action_mean_drift_mean"],3),"pred",round(r["prediction_drift_mean"],3),"surv",round(r["min_survival"],3))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
