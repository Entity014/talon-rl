#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";NENV=8;G=.99
PREFS={"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
def tilt(q):
 _,x,y,_=[q[:,i] for i in range(4)];return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def setflat(ps,delta):
 o=0
 with torch.no_grad():
  for p in ps:
   n=p.numel();p.add_(delta[o:o+n].view_as(p));o+=n
def rollout(env,m,w,seed):
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];rows=[]
 with torch.no_grad():
  for _ in range(32):
   a=m.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a);d=robot.data
   rows.append((float(torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1).mean()),float(tilt(d.root_quat_w).mean())));cur=obs_tensor(nxt).cuda()
 return rows
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={}
  for lab,j,mi in (("A",1,0),("O",2,1)):
   m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
   cur,_=env.reset(seed=2610000+j*1000);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
   for _ in range(16):
    with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
    ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
   with torch.no_grad():nv=m.value_with_preference(cur,w)
   adv,_=vector_gae(torch.stack(rw),torch.stack(val),nv,torch.stack(dn).bool(),lam=.95);fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1)
   ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach());aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
   loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach();basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
   pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);setflat([p for n,p in pm.named_parameters() if n.startswith("actor_")],delta);pm.eval()
   ds={h:[] for h in (1,2,4,8,16,32)}
   for ss in range(10):
    seed=2620000+j*1000+ss;b=rollout(env,m,w,seed);p=rollout(env,pm,w,seed)
    for h in ds: ds[h].append(float(np.mean([x[mi] for x in p[:h]])-np.mean([x[mi] for x in b[:h]])))
   out[lab]={str(h):{"mean_delta":float(np.mean(v)),"median_delta":float(np.median(v)),"improve_fraction":float(np.mean(np.array(v)<0))} for h,v in ds.items()}
  (RUN/"causal_u25_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
