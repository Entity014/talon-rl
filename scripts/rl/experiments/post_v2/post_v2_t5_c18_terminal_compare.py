#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUNS={"control":ROOT/"runs/post_v2_t5_c18_control-2026-09-23","ridge3":ROOT/"runs/post_v2_t5_c18_ridge3-2026-09-23"}
ORDER=("T","A","O","S");PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
IDX={"T":0,"A":1,"O":2,"S":3};H=(1,2,4,8,16,32);G=.99
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
def setflat(ps,base,delta):
 o=0
 with torch.no_grad():
  for p,b in zip(ps,base):
   n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
def cloneps(ps):return [p.detach().clone() for p in ps]
def tilt(q):
 _,x,y,_=[q[:,i] for i in range(4)]
 return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def rollout_metric(env,m,w,seed):
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
 rows=[];done=np.zeros(len(cur),bool)
 with torch.no_grad():
  for _ in range(32):
   a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
   rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
   done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
 return {str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H},float(1-done.mean())
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
  out={}
  for tag,run in RUNS.items():
   critic={};pert=[];endrows=[]
   models={}
   for lab in ORDER:
    m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
   # critic terminal EV on same fresh 64-step rollouts
   for bi,lab in enumerate(ORDER):
    m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
    cur,_=env.reset(seed=1110000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
    with torch.no_grad():
     for _ in range(64):
      V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
      raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
    R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);MC=np.zeros_like(R);rr=np.zeros_like(R[0])
    for t in range(63,-1,-1):rr=R[t]+G*rr*(~D[t])[:,None];MC[t]=rr
    critic[lab]={"ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
   # A/O raw gradient perturbations at terminal
   for lab in ("A","O"):
    bi=ORDER.index(lab);m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);j=IDX[lab]
    cur,_=env.reset(seed=1120000+bi*1000);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
    for _ in range(16):
     with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
     nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
     ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
    with torch.no_grad():nv=m.value_with_preference(cur,w)
    rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
    fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
    aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
    loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
    basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
    pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval()
    metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
    for ss in range(3):
     seed=1130000+bi*1000+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
    pert.append({"branch":lab,"grad_norm":float(g.norm()),"suites":suites})
   # endpoint matched suites
   for suite in range(4):
    seed=1140000+suite
    for lab in ORDER:
     m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((32,ad),device="cuda");norm=[];phys=[];done=np.zeros(32,bool)
     with torch.no_grad():
      for _ in range(64):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);norm.append(normalized_objective_vector(terms(raw,names),shape=(32,)).mean(0));data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
       phys.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
       done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     n=np.asarray(norm).mean(0);pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
     endrows.append({"suite":suite,"policy":lab,"normalized_objective_mean":n.tolist(),"scalarized":{e:float(n@PREFS[e]) for e in ORDER},"physical":pm,"survival":float(1-done.mean())})
   ow={};pw={};sw={};pk={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
   for i,lab in enumerate(ORDER):
    aa=[];bb=[];cc=[]
    for suite in range(4):
     rr=[r for r in endrows if r["suite"]==suite]
     aa.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==lab)
     bb.append((min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==lab) if lab=="T" else min(rr,key=lambda r:r["physical"][pk[lab]])["policy"]==lab)
     cc.append(max(rr,key=lambda r:r["scalarized"][lab])["policy"]==lab)
    ow[lab]=float(np.mean(aa));pw[lab]=float(np.mean(bb));sw[lab]=float(np.mean(cc))
   vb={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in endrows if r["policy"]==p])) for p in ORDER};vr={p:vb[p]/(vb["T"]+1e-12) for p in ORDER}
   gates={"objective_winner_fraction":ow,"physical_winner_fraction":pw,"diagonal_scalarized_winner_fraction":sw,"min_survival":min(r["survival"] for r in endrows),"vertical_ratio_to_tracking_policy":vr}
   out[tag]={"critic":critic,"perturbations":pert,"endpoint_rows":endrows,"gates":gates}
  target=RUNS["ridge3"]/"terminal_compare.json";target.write_text(json.dumps(out,indent=2)+"\n")
  print(json.dumps({k:{"gates":v["gates"],"critic":v["critic"],"pert_n":len(v["perturbations"])} for k,v in out.items()},indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
