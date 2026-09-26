#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c12_moving_batch-2026-09-23"
ORDER=("T","A","O","S");NB=6;H=32;G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def trunc_mc(R,D):
 out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
 for t in range(len(R)-1,-1,-1):
  run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
 return out
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
 return float(1-np.var(y-p)/(np.var(y)+1e-12))
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
  out={"specialists":{}}
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
   w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=980000+bi*1000);cur=obs_tensor(cur).cuda();B=[]
   for k in range(NB):
    obs=[];rw=[];dn=[]
    with torch.no_grad():
     for _ in range(H):
      obs.append(cur);a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
      rw.append(torch.tensor(normalized_objective_vector(terms(raw,names),shape=(32,)),device="cuda",dtype=torch.float32)*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
    X=torch.cat(obs);W=w.repeat(H,1);Y=trunc_mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
    with torch.no_grad():
      F=m.critic_body(m._with_w(X,W));A=torch.cat([F,torch.ones((len(F),1),device="cuda")],1)
      sol=torch.linalg.lstsq(A,Y).solution;pred=A@sol;s=torch.linalg.svdvals(A)
    B.append({"A":A,"Y":Y,"sol":sol,"self_ev":[ev(Y[:,j].cpu(),pred[:,j].cpu()) for j in range(4)],
              "cond":float(s.max()/(s.min()+1e-12)),"effective_rank":int((s>1e-6*s.max()).sum())})
   pairs=[]
   for k in range(NB-1):
    curB,nxt=B[k],B[k+1]
    with torch.no_grad():
      p_cur_on_next=nxt["A"]@curB["sol"];p_next_on_next=nxt["A"]@nxt["sol"]
    pairs.append({"pair":f"{k}->{k+1}",
      "cond_current":curB["cond"],"cond_next":nxt["cond"],"rank_current":curB["effective_rank"],"rank_next":nxt["effective_rank"],
      "self_ev_current":curB["self_ev"],"self_ev_next":nxt["self_ev"],
      "current_optimum_on_next_ev":[ev(nxt["Y"][:,j].cpu(),p_cur_on_next[:,j].cpu()) for j in range(4)],
      "next_optimum_on_next_ev":[ev(nxt["Y"][:,j].cpu(),p_next_on_next[:,j].cpu()) for j in range(4)],
      "prediction_disagreement_rmse":[float(torch.sqrt(((p_cur_on_next[:,j]-p_next_on_next[:,j])**2).mean())) for j in range(4)]
    })
   out["specialists"][lab]=pairs
  vals=[];gaps=[];conds=[]
  for lab in ORDER:
   for p in out["specialists"][lab]:
    vals+=p["current_optimum_on_next_ev"];gaps+=(np.array(p["next_optimum_on_next_ev"])-np.array(p["current_optimum_on_next_ev"])).tolist();conds += [p["cond_current"],p["cond_next"]]
  out["aggregate"]={"prior_batch_optimum_on_next_ev_mean":float(np.mean(vals)),"next_batch_optimum_ev_advantage_mean":float(np.mean(gaps)),
                    "prior_optimum_next_ev_negative_fraction":float(np.mean(np.array(vals)<0)),
                    "design_condition_number_median":float(np.median(conds)),"design_condition_number_max":float(np.max(conds))}
  (OUT/"optimal_head_transfer.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["aggregate"],indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
