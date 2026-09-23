#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
OUT=ROOT/'runs/post_v2_t5_c20_coverage-2026-09-23';ORDER=('T','A','O','S');SUP=('recent3','recent6','recent12','all16','diverse12');G=.99
PREFS={'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
def ret(R,D):
 out=np.zeros_like(R);run=np.zeros_like(R[0])
 for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
 return out
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({'headless':True,'enable_cameras':False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=16;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  sol=np.load(OUT/'head_solutions.npz');res={}
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(od,ad).cuda();initialize_from_rsl_m01(m,ROOT/'runs/m0_1_seed0_2026-09-22/model_299.pt',device='cpu',critic_head_init='zero');m.eval();w=torch.tensor(PREFS[lab],device='cuda').repeat(16,1)
   # collect one fresh 64 rollout and body features/policy actions once
   cur,_=env.reset(seed=990000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];F=[]
   with torch.no_grad():
    for _ in range(64):
     F.append(m.critic_body(m._with_w(cur,w)).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
     R.append(normalized_objective_vector(terms(raw,names),shape=(16,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
   R=np.asarray(R);D=np.asarray(D,bool);F=np.asarray(F);MC=ret(R,D);q={}
   for s in SUP:
    W=sol[f'{lab}_{s}_W'];b=sol[f'{lab}_{s}_b'];P=np.einsum('ted,hd->teh',F,W)+b
    q[s]={'mc64_ev':[ev(MC[:,:,j],P[:,:,j]) for j in range(4)],'mc64_bias':[float(np.mean(P[:,:,j]-MC[:,:,j])) for j in range(4)]}
   res[lab]=q
  agg={}
  for s in SUP:
   e=[];b=[]
   for lab in ORDER:e+=res[lab][s]['mc64_ev'];b+=res[lab][s]['mc64_bias']
   agg[s]={'mc64_ev_mean':float(np.mean(e)),'mc64_negative_fraction':float(np.mean(np.array(e)<0)),'mc64_mean_abs_bias':float(np.mean(np.abs(b)))}
  out={'schema':'c20_mc64_eval_v1','specialists':res,'aggregate':agg};(OUT/'mc64_eval.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(agg,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=='__main__':main()
