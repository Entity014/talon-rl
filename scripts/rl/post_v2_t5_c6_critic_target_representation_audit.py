#!/usr/bin/env python3
from __future__ import annotations
import json,sys,traceback,copy
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c5_h16-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c6_critic_target_repr-2026-09-23"
ORDER=("T","A","O","S"); SNAPS=(0,10,25,50,100); G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}

def obs_tensor(x):
    if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);v=np.var(y)
    return float(1-np.var(y-p)/(v+1e-12))
def corr(a,b):
    a=np.asarray(a,float).reshape(-1);b=np.asarray(b,float).reshape(-1)
    if np.std(a)<1e-12 or np.std(b)<1e-12:return 0.0
    return float(np.corrcoef(a,b)[0,1])
def mc(rew,done):
    T,N,O=rew.shape;out=np.zeros_like(rew,dtype=np.float64);run=np.zeros((N,O),np.float64)
    for t in range(T-1,-1,-1):
        run=rew[t]+G*run*(~done[t])[:,None];out[t]=run
    return out
def cosine(a,b):
    a=np.asarray(a,float).reshape(-1);b=np.asarray(b,float).reshape(-1)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def main():
  from isaaclab.app import AppLauncher
  app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
  OUT.mkdir(parents=True,exist_ok=True)
  try:
    import gymnasium as gym,isaaclab_tasks
    from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
    from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae
    from talon_rl.t3b_objectives import normalized_objective_vector

    cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0
    env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
    o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
    report={"schema":"t5_c6_critic_target_repr_v1","status":"MEASUREMENT_COMPLETE","A_fixed_fit":{},"B_target_validity":{},"C_online_drift":{}}

    datasets={}
    # Collect matched 64-step replay per branch/snapshot. Actor deterministic to reduce sampling noise.
    for bi,lab in enumerate(ORDER):
      w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);datasets[lab]={}
      for snap in SNAPS:
        cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
        cur,_=env.reset(seed=600000+bi*1000);cur=obs_tensor(cur).cuda()
        obs=[];r=[];term=[];trunc=[];vals=[];nextvals=[];info_samples=[];next_obs=[]
        with torch.no_grad():
          for t in range(64):
            obs.append(cur.detach().cpu())
            vals.append(m.value_with_preference(cur,w).detach().cpu())
            a=m.act_inference_with_preference(cur,w)
            nxt,_,te,tr,info=env.step(a);nxt=obs_tensor(nxt).cuda()
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            r.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt)
            term.append(te.cpu().numpy().astype(bool));trunc.append(tr.cpu().numpy().astype(bool))
            nextvals.append(m.value_with_preference(nxt,w).detach().cpu())
            next_obs.append(nxt.detach().cpu())
            if (te|tr).any():
              info_samples.append({"t":t,"term":int(te.sum()),"trunc":int(tr.sum()),"keys":sorted(list(info.keys())) if isinstance(info,dict) else str(type(info))})
            cur=nxt
        O=torch.stack(obs).float(); NO=torch.stack(next_obs).float(); R=np.asarray(r,float);TE=np.asarray(term,bool);TR=np.asarray(trunc,bool);D=TE|TR
        V=torch.stack(vals).numpy();NV=torch.stack(nextvals).numpy()
        MC_done=mc(R,D);MC_term=mc(R,TE)
        # one-step target variants using observed next-state value
        T_current=R+G*NV*(~D)[:,:,None]
        T_term=R+G*NV*(~TE)[:,:,None]
        datasets[lab][snap]={"obs":O,"next_obs":NO,"R":R,"TE":TE,"TR":TR,"D":D,"V":V,"NV":NV,"MC_done":MC_done,"MC_term":MC_term}
        # Current GAE implementation from exact rollout.
        rt=torch.tensor(R,dtype=torch.float32,device="cuda");vt=torch.tensor(V,dtype=torch.float32,device="cuda");dt=torch.tensor(D,device="cuda")
        # vector_gae expects a single next value for end of segment; use last observed NV.
        adv,gae_ret=vector_gae(rt,vt,torch.tensor(NV[-1],dtype=torch.float32,device="cuda"),dt)
        gae=gae_ret.cpu().numpy()
        report["B_target_validity"][f"{lab}:{snap}"]={
          "termination_count":int(TE.sum()),"truncation_count":int(TR.sum()),
          "done_count":int(D.sum()),"event_samples":info_samples[:12],
          "current_1step_vs_mc_done_mae":[float(np.mean(np.abs(T_current[:,:,j]-MC_done[:,:,j]))) for j in range(4)],
          "term_only_1step_vs_mc_term_mae":[float(np.mean(np.abs(T_term[:,:,j]-MC_term[:,:,j]))) for j in range(4)],
          "gae_vs_mc_done_mae":[float(np.mean(np.abs(gae[:,:,j]-MC_done[:,:,j]))) for j in range(4)],
          "gae_vs_mc_term_mae":[float(np.mean(np.abs(gae[:,:,j]-MC_term[:,:,j]))) for j in range(4)],
          "gae_bias_vs_mc_done":[float(np.mean(gae[:,:,j]-MC_done[:,:,j])) for j in range(4)],
          "value_ev_vs_mc_done":[ev(MC_done[:,:,j],V[:,:,j]) for j in range(4)],
          "value_ev_vs_mc_term":[ev(MC_term[:,:,j],V[:,:,j]) for j in range(4)],
          "mc_done_vs_mc_term_mae":[float(np.mean(np.abs(MC_done[:,:,j]-MC_term[:,:,j]))) for j in range(4)],
        }

    # C6-A: frozen-target fit. Use snap 50 and 100 where online EV is poor.
    for lab in ORDER:
      report["A_fixed_fit"][lab]={}
      wbase=torch.tensor(PREFS[lab],device="cuda")
      for snap in (50,100):
        d=datasets[lab][snap];X=d["obs"].reshape(-1,od).cuda();Y=torch.tensor(d["MC_done"].reshape(-1,4),dtype=torch.float32,device="cuda");W=wbase.repeat(len(X),1)
        cp=RUN/f"{lab}_snap_{snap}.pt"
        base=T4SharedActorCritic(od,ad).cuda();base.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);base.eval()
        with torch.no_grad(): p0=base.value_with_preference(X,W)
        init_ev=[ev(Y[:,j].cpu().numpy(),p0[:,j].cpu().numpy()) for j in range(4)]

        # Analytical best linear head on frozen critic-body features.
        with torch.no_grad():
          feats=base.critic_body(base._with_w(X,W))
          A=torch.cat([feats,torch.ones((len(feats),1),device="cuda")],dim=1)
          sol=torch.linalg.lstsq(A,Y).solution
          pred_lin=A@sol
        lin_ev=[ev(Y[:,j].cpu().numpy(),pred_lin[:,j].cpu().numpy()) for j in range(4)]
        lin_rmse=[float(torch.sqrt(((pred_lin[:,j]-Y[:,j])**2).mean())) for j in range(4)]

        # Head-only Adam fit on fixed targets.
        hm=T4SharedActorCritic(od,ad).cuda();hm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
        for n,p in hm.named_parameters(): p.requires_grad_(n.startswith("critic_head"))
        opt=torch.optim.Adam([p for p in hm.parameters() if p.requires_grad],lr=1e-3)
        head_curve=[]
        for k in range(501):
          pred=hm.value_with_preference(X,W);loss=((pred-Y)**2).mean()
          if k in (0,1,10,50,100,250,500):
            head_curve.append({"step":k,"loss":float(loss.detach()),"ev":[ev(Y[:,j].cpu().numpy(),pred[:,j].detach().cpu().numpy()) for j in range(4)]})
          if k<500: opt.zero_grad(set_to_none=True);loss.backward();opt.step()

        # Full critic fixed-dataset fit, actor frozen. Use stable LR 1e-4.
        fm=T4SharedActorCritic(od,ad).cuda();fm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
        pars=[p for n,p in fm.named_parameters() if n.startswith("critic_")]
        opt2=torch.optim.Adam(pars,lr=1e-4);full_curve=[]
        for k in range(1001):
          pred=fm.value_with_preference(X,W);loss=((pred-Y)**2).mean()
          if k in (0,1,10,50,100,250,500,1000):
            full_curve.append({"step":k,"loss":float(loss.detach()),"ev":[ev(Y[:,j].cpu().numpy(),pred[:,j].detach().cpu().numpy()) for j in range(4)]})
          if k<1000: opt2.zero_grad(set_to_none=True);loss.backward();opt2.step()
        report["A_fixed_fit"][lab][str(snap)]={
          "dataset_n":int(len(X)),"initial_ev":init_ev,
          "analytic_frozen_body_linear_head_ev":lin_ev,"analytic_frozen_body_linear_head_rmse":lin_rmse,
          "head_only_curve":head_curve,"full_critic_curve":full_curve,
        }

    # C6-C online drift: matched seed means state-target distributions across snapshots can be compared.
    for lab in ORDER:
      rows={}
      ref=datasets[lab][0]
      for snap in SNAPS:
        d=datasets[lab][snap]
        # Distribution drift in obs and target moments.
        obs_mean=d["obs"].numpy().mean((0,1));ref_mean=ref["obs"].numpy().mean((0,1))
        obs_std=d["obs"].numpy().std((0,1));ref_std=ref["obs"].numpy().std((0,1))
        target=d["MC_done"];rt=ref["MC_done"]
        rows[str(snap)]={
          "obs_mean_shift_l2_from_u0":float(np.linalg.norm(obs_mean-ref_mean)),
          "obs_std_shift_l2_from_u0":float(np.linalg.norm(obs_std-ref_std)),
          "mc_mean":[float(target[:,:,j].mean()) for j in range(4)],
          "mc_std":[float(target[:,:,j].std()) for j in range(4)],
          "mc_mean_shift_from_u0":[float(target[:,:,j].mean()-rt[:,:,j].mean()) for j in range(4)],
          "mc_std_ratio_to_u0":[float(target[:,:,j].std()/(rt[:,:,j].std()+1e-12)) for j in range(4)],
        }
      # Consecutive target drift summary
      cons={}
      for a,b in zip(SNAPS[:-1],SNAPS[1:]):
        da,db=datasets[lab][a]["MC_done"],datasets[lab][b]["MC_done"]
        cons[f"{a}->{b}"]={
          "mean_shift_l2":float(np.linalg.norm(db.mean((0,1))-da.mean((0,1)))),
          "std_shift_l2":float(np.linalg.norm(db.std((0,1))-da.std((0,1)))),
        }
      report["C_online_drift"][lab]={"snapshots":rows,"consecutive":cons}

    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],
      "timeouts":{k:{"term":v["termination_count"],"trunc":v["truncation_count"]} for k,v in list(report["B_target_validity"].items())[:8]},
      "fit_terminal":{lab:{s:report["A_fixed_fit"][lab][s]["full_critic_curve"][-1]["ev"] for s in ("50","100")} for lab in ORDER}},indent=2))
  except BaseException as e:
    (OUT/"ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
  finally:
    if env is not None:env.close()
    app.close()
if __name__=="__main__":main()
