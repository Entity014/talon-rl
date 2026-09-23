#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c12_moving_batch-2026-09-23"
ORDER=("T","A","O","S");NB=6;H=32;G=.99;RIDGES=(0.0,1e-6,1e-4,1e-3,1e-2,1e-1)
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def trunc_mc(R,D):
 out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
 for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
 return out
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
 return float(1-np.var(y-p)/(np.var(y)+1e-12))
def solve(A,Y,l2):
 if l2==0:
  return torch.linalg.lstsq(A,Y).solution
 I=torch.eye(A.shape[1],device=A.device,dtype=A.dtype);I[-1,-1]=0
 return torch.linalg.solve(A.T@A+l2*I,A.T@Y)
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={"ridge_values":RIDGES,"specialists":{}}
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
   w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=990000+bi*1000);cur=obs_tensor(cur).cuda();B=[]
   for k in range(NB):
    obs=[];rw=[];dn=[]
    with torch.no_grad():
     for _ in range(H):
      obs.append(cur);a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
      rw.append(torch.tensor(normalized_objective_vector(terms(raw,names),shape=(32,)),device="cuda",dtype=torch.float32)*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
    X=torch.cat(obs);W=w.repeat(H,1);Y=trunc_mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
    with torch.no_grad():
     F=m.critic_body(m._with_w(X,W));A=torch.cat([F,torch.ones((len(F),1),device="cuda")],1)
    B.append((A,Y))
   labout={}
   for l2 in RIDGES:
    pairs=[]
    for k in range(NB-1):
     A,Y=B[k];An,Yn=B[k+1];sol=solve(A,Y,l2);soln=solve(An,Yn,l2)
     with torch.no_grad():
      pc=A@sol;pn=An@sol;pnopt=An@soln
     pairs.append({
      "pair":f"{k}->{k+1}",
      "self_ev":[ev(Y[:,j].cpu(),pc[:,j].cpu()) for j in range(4)],
      "prior_on_next_ev":[ev(Yn[:,j].cpu(),pn[:,j].cpu()) for j in range(4)],
      "next_opt_ev":[ev(Yn[:,j].cpu(),pnopt[:,j].cpu()) for j in range(4)],
      "solution_norm":float(sol.norm()),"solution_drift":float((soln-sol).norm())
     })
    labout[str(l2)]=pairs
   out["specialists"][lab]=labout
  agg={}
  for l2 in RIDGES:
   selfe=[];nextv=[];opte=[];norm=[];drift=[]
   for lab in ORDER:
    for p in out["specialists"][lab][str(l2)]:
     selfe+=p["self_ev"];nextv+=p["prior_on_next_ev"];opte+=p["next_opt_ev"];norm.append(p["solution_norm"]);drift.append(p["solution_drift"])
   agg[str(l2)]={"self_ev_mean":float(np.mean(selfe)),"prior_on_next_ev_mean":float(np.mean(nextv)),
                  "prior_on_next_negative_fraction":float(np.mean(np.array(nextv)<0)),
                  "next_opt_ev_mean":float(np.mean(opte)),"solution_norm_mean":float(np.mean(norm)),"solution_drift_mean":float(np.mean(drift))}
  out["aggregate"]=agg;(OUT/"ridge_head_transfer.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
