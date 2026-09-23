#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
OUT=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
ORDER=("T","A","O","S");G=.99;H=32;NENV=8;NSUP=12
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def trunc_mc(rt,dt):
 out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
 for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
 return out
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
 return float(1-np.var(y-p)/(np.var(y)+1e-12))
def ridge(F,Y,l2=1.0):
 A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
 sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
 return sol[:-1].T,sol[-1]
def flat_grad(loss,params,retain=True):
 gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
 return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,params)])
def collect(env,m,w,mgr,seed,need_latent=False):
 from talon_rl.t3b_objectives import normalized_objective_vector
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
 ob=[];pre=[];old=[];rw=[];dn=[]
 with torch.no_grad():
  for _ in range(H):
   if need_latent:a,lp,u=m.act_with_preference_latent(cur,w)
   else:a=m.act_inference_with_preference(cur,w);lp=u=None
   nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
   vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
   ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda())
   if need_latent:pre.append(u);old.append(lp)
   cur=obs_tensor(nxt).cuda()
 rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
 fo=torch.cat(ob);fw=w.repeat(H,1)
 with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
 out={"obs":fo,"w":fw,"Y":Y,"F":F,"rt":rt,"dt":dt,"next_obs":cur}
 if need_latent:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
 return out
def cosine_matrix(gs):
 M=[]
 for a in gs:
  row=[]
  for b in gs:row.append(float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12)))
  M.append(row)
 return M
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
  cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda()
  ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  report={"schema":"c25_actor_updating_reset_support_v1","updates":25,"support_size":NSUP,"actor_lr":1e-3,"ridge_lambda":1.0,"specialists":{}}
  for bi,lab in enumerate(ORDER):
   torch.manual_seed(61000+bi);np.random.seed(61000+bi)
   m=T4SharedActorCritic(o.shape[-1],ad).cuda()
   initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero")
   for n,p in m.named_parameters():
    if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
   actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
   opt=torch.optim.Adam(actor_params,lr=1e-3);w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
   rows=[];torch.save({"model":m.state_dict(),"snapshot":0,"specialist":lab},OUT/f"{lab}_snap_0.pt")
   prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params]);prev_sol=None
   for uidx in range(1,26):
    mainb=collect(env,m,w,mgr,1810000+bi*100000+uidx*131,need_latent=True)
    Fs=[];Ys=[]
    for k in range(NSUP):
     q=collect(env,m,w,mgr,1910000+bi*100000+uidx*1000+k*37,need_latent=False)
     Fs.append(q["F"].cpu().numpy());Ys.append(q["Y"].cpu().numpy())
    F=np.concatenate(Fs);Y=np.concatenate(Ys);W,b=ridge(F,Y,1.0)
    with torch.no_grad():
     m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
     m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
     vt=m.value_with_preference(mainb["obs"],mainb["w"]).reshape(H,NENV,4)
     nv=m.value_with_preference(mainb["next_obs"],w)
     adv,_=vector_gae(mainb["rt"],vt,nv,mainb["dt"],lam=.95)
    ratio=torch.exp(m.logp_from_pre_tanh_with_preference(mainb["obs"],mainb["w"],mainb["u"])-mainb["old"].detach())
    ratio_err=float((ratio-1).abs().max())
    if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
    av=adv.reshape(-1,4).detach()
    obj_grads=[]
    for j in range(4):
     li=-(ratio*av[:,j]).mean();obj_grads.append(flat_grad(li,actor_params,retain=True).detach())
    cm=cosine_matrix(obj_grads)
    loss=scalarized_late_weighted_ppo(ratio,av,mainb["w"])
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor_params,1.0);opt.step()
    actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
    sol=np.r_[W.reshape(-1),b];sdr=0.0 if prev_sol is None else float(np.linalg.norm(sol-prev_sol));prev_sol=sol
    row={"update":uidx,"ratio_maxerr":ratio_err,"actor_loss":float(loss.detach()),"actor_param_drift":float((actor_now-prev_actor).norm()),
         "objective_grad_norm":[float(g.norm()) for g in obj_grads],"objective_grad_cosine":cm,"head_solution_drift":sdr}
    prev_actor=actor_now.clone();rows.append(row)
    torch.save({"model":m.state_dict(),"snapshot":uidx,"specialist":lab},OUT/f"{lab}_snap_{uidx}.pt")
   report["specialists"][lab]=rows
  (OUT/"train.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","updates":25},indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
