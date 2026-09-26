#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,traceback
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ORDER=("T","A","O","S")
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
OBJ_IDX={"T":0,"A":1,"O":2,"S":3}
PHYS_KEY={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
SNAPS=(0,10,50)
CRITIC_SNAPS=(0,10,50,100)
HORIZONS=(1,2,4,8,16,32)

def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
def set_from_flat(ps,base,delta):
    o=0
    with torch.no_grad():
        for p,b in zip(ps,base):
            n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
def clone_params(ps):return [p.detach().clone() for p in ps]
def explained_variance(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);v=np.var(y)
    return float(1-np.var(y-p)/(v+1e-12))
def tilt_deg(q):
    _,x,y,_=[q[:,i] for i in range(4)]
    return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def rollout_metrics(env,model,w,seed,max_h=32):
    cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"]
    prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
    rows=[]
    with torch.no_grad():
      for t in range(1,max_h+1):
        a=model.act_inference_with_preference(cur,w)
        nxt,_,term,trunc,_=env.step(a);data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        rows.append({
          "t":t,
          "vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),
          "wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),
          "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
          "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
          "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
          "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean()),
          "survival_step":float(1-(term|trunc).float().mean()),
        })
        prev=a;cur=obs_tensor(nxt).cuda()
    out={}
    for H in HORIZONS:
      seg=rows[:H]
      out[str(H)]={k:float(np.mean([r[k] for r in seg])) for k in rows[0] if k!="t"}
    return out

def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 outdir=ROOT/"runs/post_v2_t5_c2_gradient_physical-2026-09-23";outdir.mkdir(parents=True,exist_ok=True)
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=16;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager

  # Critic accuracy: evaluate head predictions against 32-step Monte-Carlo normalized returns.
  critic={}
  for lab in ("A","O","S"):
    critic[lab]={}
    w=torch.tensor(PREFS[lab],device="cuda").repeat(16,1)
    for snap in CRITIC_SNAPS:
      cp=ROOT/f"runs/post_v2_t5_c1_zero_critic-2026-09-23/{lab}_snap_{snap}.pt"
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
      cur,_=env.reset(seed=410000+OBJ_IDX[lab]*1000+snap);cur=obs_tensor(cur).cuda()
      vals=[];rews=[];dones=[]
      with torch.no_grad():
        for _ in range(32):
          vals.append(m.value_with_preference(cur,w).cpu().numpy())
          a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
          raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
          rews.append(normalized_objective_vector(terms(raw,names),shape=(16,))*env.unwrapped.step_dt)
          dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
      R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals)
      mc=np.zeros_like(R);run=np.zeros_like(R[0])
      for t in range(31,-1,-1):
        run=R[t]+.99*run*(~D[t])[:,None];mc[t]=run
      td_bias=[];ev=[]
      for j in range(4):
        ev.append(explained_variance(mc[:,:,j].reshape(-1),V[:,:,j].reshape(-1)))
        td_bias.append(float(np.mean(V[:,:,j]-mc[:,:,j])))
      critic[lab][str(snap)]={"explained_variance":ev,"value_minus_mc_bias":td_bias,"value_std":V.reshape(-1,4).std(0).tolist(),"mc_std":mc.reshape(-1,4).std(0).tolist()}

  interventions=[]
  # Reconstruct one batch per branch/snapshot and compare own raw gradient vs actual Adam actor step.
  for lab in ("A","O","S"):
    w=torch.tensor(PREFS[lab],device="cuda").repeat(16,1);j=OBJ_IDX[lab]
    for snap in SNAPS:
      cp=ROOT/f"runs/post_v2_t5_c1_zero_critic-2026-09-23/{lab}_snap_{snap}.pt"
      base_model=T4SharedActorCritic(od,ad).cuda();base_model.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
      opt=torch.optim.Adam(base_model.parameters(),lr=1e-3)
      # matched short batch for gradient/update construction
      cur,_=env.reset(seed=420000+OBJ_IDX[lab]*1000+snap);cur=obs_tensor(cur).cuda()
      ob=[];pre=[];old=[];rw=[];val=[];dn=[]
      for _ in range(2):
        with torch.no_grad():a,lp,u=base_model.act_with_preference_latent(cur,w);v=base_model.value_with_preference(cur,w)
        nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
        vec=normalized_objective_vector(terms(raw,names),shape=(16,))
        ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
      with torch.no_grad():nv=base_model.value_with_preference(cur,w)
      rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt)
      fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(2,1)
      ratio=torch.exp(base_model.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
      actor_ps=[p for n,p in base_model.named_parameters() if n.startswith("actor_")] # deterministic policy only
      base_actor=clone_params(actor_ps)
      own_loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean()
      own_g=flat(torch.autograd.grad(own_loss,actor_ps,retain_graph=True,allow_unused=True),actor_ps).detach()
      # actual joint loss optimizer step, then extract actor delta
      al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
      cl=vector_value_loss(base_model.value_with_preference(fo,fw),ret.reshape(-1,4).detach())
      opt.zero_grad(set_to_none=True);(al+cl).backward();opt.step()
      adam_delta=torch.cat([(p.detach()-b).reshape(-1) for p,b in zip(actor_ps,base_actor)])
      adam_norm=float(adam_delta.norm());raw_dir=-own_g/(own_g.norm()+1e-12)
      raw_delta=raw_dir*adam_norm
      # Build three models from same checkpoint: baseline, raw-SGD own objective, actual Adam actor delta.
      models={}
      for kind,delta in (("baseline",torch.zeros_like(raw_delta)),("raw_own_sgd",raw_delta),("actual_adam",adam_delta)):
        mm=T4SharedActorCritic(od,ad).cuda();mm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);aps=[p for n,p in mm.named_parameters() if n.startswith("actor_")]
        set_from_flat(aps,clone_params(aps),delta);mm.eval();models[kind]=mm
      # Two matched reset suites; same initial distribution per condition.
      suite_rows=[]
      for suite in range(2):
        seed=430000+OBJ_IDX[lab]*1000+snap*10+suite
        rr={kind:rollout_metrics(env,mm,w,seed,32) for kind,mm in models.items()}
        suite_rows.append({"suite":suite,"seed":seed,"rollouts":rr})
      interventions.append({"specialist":lab,"snapshot":snap,"own_objective_index":j,"adam_actor_step_norm":adam_norm,"own_gradient_norm":float(own_g.norm()),"suites":suite_rows})
  report={"schema":"t5_c2_gradient_physical_audit_v1","status":"MEASUREMENT_COMPLETE","horizons":list(HORIZONS),"critic_accuracy":critic,"interventions":interventions,
          "notes":{"raw_own_sgd":"negative own-objective loss gradient normalized to exact actual Adam actor-step norm","actual_adam":"actual actor parameter delta from one joint PPO+critic Adam step on the same batch","baseline":"same checkpoint with no perturbation"}}
  (outdir/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps({"status":report["status"],"num_interventions":len(interventions),"critic":critic},indent=2))
 except BaseException as e:
  (outdir/"ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
