#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c26_angular_credit-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
SNAPS=(0,5,10,25);G=.99;H=32;NENV=8;J=1
WREF=np.array([.1,.7,.1,.1],np.float32)
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
def corr(a,b):
 a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
 return float(np.corrcoef(a,b)[0,1])
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  report={"schema":"c26_angular_credit_stability_v1","snapshots":{}}
  for snap in SNAPS:
   m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
   w=torch.tensor(WREF,device="cuda").repeat(NENV,1);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
   batchrows=[];grads=[]
   for bi in range(8):
    cur,_=env.reset(seed=3110000+snap*10000+bi*211);cur=obs_tensor(cur).cuda()
    ob=[];pre=[];old=[];rw=[];val=[];dn=[]
    for _ in range(H):
     with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
     nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
     vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
     ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
    with torch.no_grad():nv=m.value_with_preference(cur,w)
    rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt,lam=.95)
    fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(H,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
    av=adv.reshape(-1,4)[:,J].detach();loss=-(ratio*av).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach();grads.append(g)
    target=ret.reshape(-1,4)[:,J].detach().cpu().numpy();pred=vt.reshape(-1,4)[:,J].detach().cpu().numpy();rew=rt.reshape(-1,4)[:,J].detach().cpu().numpy()
    batchrows.append({"batch":bi,"adv_mean":float(av.mean()),"adv_std":float(av.std()),"adv_rms":float(torch.sqrt(torch.mean(av*av))),
      "adv_absmean":float(av.abs().mean()),"adv_mean_over_std":float(abs(av.mean())/(av.std()+1e-12)),
      "grad_norm":float(g.norm()),"value_ev":float(1-np.var(target-pred)/(np.var(target)+1e-12)),
      "reward_adv_corr":corr(rew,av.cpu().numpy()),"return_adv_corr":corr(target,av.cpu().numpy())})
   cos=[]
   for i in range(len(grads)):
    for j in range(i+1,len(grads)):cos.append(float(torch.dot(grads[i],grads[j])/(grads[i].norm()*grads[j].norm()+1e-12)))
   report["snapshots"][str(snap)]={"batches":batchrows,"summary":{
    "adv_std_mean":float(np.mean([x["adv_std"] for x in batchrows])),
    "adv_rms_mean":float(np.mean([x["adv_rms"] for x in batchrows])),
    "adv_snr_mean":float(np.mean([x["adv_mean_over_std"] for x in batchrows])),
    "grad_norm_mean":float(np.mean([x["grad_norm"] for x in batchrows])),
    "grad_norm_cv":float(np.std([x["grad_norm"] for x in batchrows])/(np.mean([x["grad_norm"] for x in batchrows])+1e-12)),
    "gradient_pairwise_cos_mean":float(np.mean(cos)),"gradient_pairwise_cos_std":float(np.std(cos)),
    "gradient_pairwise_cos_negative_fraction":float(np.mean(np.array(cos)<0)),
    "value_ev_mean":float(np.mean([x["value_ev"] for x in batchrows])),
    "reward_adv_corr_mean":float(np.mean([x["reward_adv_corr"] for x in batchrows])),
    "return_adv_corr_mean":float(np.mean([x["return_adv_corr"] for x in batchrows]))}}
  (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps({k:v["summary"] for k,v in report["snapshots"].items()},indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
