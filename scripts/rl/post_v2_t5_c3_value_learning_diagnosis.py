#!/usr/bin/env python3
"""C3 diagnostic-only audit of four-head critic/value learning."""
from __future__ import annotations
import json,sys,traceback
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ORDER=("T","A","O","S")
PREFS={
 "T":np.array([.7,.1,.1,.1],np.float32),
 "A":np.array([.1,.7,.1,.1],np.float32),
 "O":np.array([.1,.1,.7,.1],np.float32),
 "S":np.array([.1,.1,.1,.7],np.float32),
}
SNAPS=(0,1,5,10,25,50,100)
NSTEPS=(1,2,4,8,16,32)
GAMMA=.99

def obs_tensor(x):
    if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)

def terms(raw,names):
    return {n:raw[:,i] for i,n in enumerate(names)}

def explained_variance(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);v=np.var(y)
    return float(1-np.var(y-p)/(v+1e-12))

def flat(gs,ps):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])

def cosine(a,b):
    return float((a@b)/(a.norm()*b.norm()+1e-12))

def nstep_targets(rew,done,values,n):
    # rew [T,N,O], done [T,N], values [T+1,N,O]
    T,N,O=rew.shape
    out=np.zeros_like(rew,dtype=np.float64)
    for t in range(T):
        ret=np.zeros((N,O),dtype=np.float64)
        disc=np.ones((N,1),dtype=np.float64)
        alive=np.ones((N,1),dtype=np.float64)
        end=min(T,t+n)
        for k in range(t,end):
            ret += disc*alive*rew[k]
            alive *= (~done[k])[:,None]
            disc *= GAMMA
        if t+n <= T:
            ret += disc*alive*values[t+n]
        out[t]=ret
    return out

def mc_targets(rew,done):
    T,N,O=rew.shape
    out=np.zeros_like(rew,dtype=np.float64)
    run=np.zeros((N,O),dtype=np.float64)
    for t in range(T-1,-1,-1):
        run=rew[t]+GAMMA*run*(~done[t])[:,None]
        out[t]=run
    return out

def main():
  from isaaclab.app import AppLauncher
  app=AppLauncher({"headless":True,"enable_cameras":False}).app
  env=None
  outdir=ROOT/"runs/post_v2_t5_c3_value_learning-2026-09-23";outdir.mkdir(parents=True,exist_ok=True)
  try:
    import gymnasium as gym,isaaclab_tasks
    from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
    from talon_rl.t4_actor_critic import T4SharedActorCritic
    from talon_rl.t3b_objectives import normalized_objective_vector

    cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0
    env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
    obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda()
    od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager

    report={"schema":"t5_c3_value_learning_diagnosis_v1","status":"MEASUREMENT_COMPLETE",
            "snapshots":list(SNAPS),"nsteps":list(NSTEPS),"branches":{}}

    for bi,lab in enumerate(ORDER):
      w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
      branch={}
      for snap in SNAPS:
        cp=ROOT/f"runs/post_v2_t5_c1_zero_critic-2026-09-23/{lab}_snap_{snap}.pt"
        m=T4SharedActorCritic(od,ad).cuda()
        m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()

        # Same reset protocol across snapshots within branch.
        cur,_=env.reset(seed=440000+bi*1000);cur=obs_tensor(cur).cuda()
        obs_seq=[];rew=[];done=[];vals=[]
        with torch.no_grad():
          for t in range(32):
            obs_seq.append(cur)
            vals.append(m.value_with_preference(cur,w).cpu().numpy())
            a=m.act_inference_with_preference(cur,w)
            nxt,_,term,trunc,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            rew.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt)
            done.append((term|trunc).cpu().numpy().astype(bool))
            cur=obs_tensor(nxt).cuda()
          vals.append(m.value_with_preference(cur,w).cpu().numpy())

        R=np.asarray(rew,float);D=np.asarray(done,bool);V=np.asarray(vals,float)
        MC=mc_targets(R,D)

        target_family={}
        for n in NSTEPS:
          Tn=nstep_targets(R,D,V,n)
          # Compare only positions with a full n-step lookahead so edge truncation does not bias comparison.
          usable=max(1,32-n+1)
          sl=slice(0,usable)
          per=[]
          for j in range(4):
            err=Tn[sl,:,j]-MC[sl,:,j]
            per.append({
              "mae_vs_mc":float(np.mean(np.abs(err))),
              "rmse_vs_mc":float(np.sqrt(np.mean(err*err))),
              "bias_vs_mc":float(np.mean(err)),
              "corr_with_mc":float(np.corrcoef(Tn[sl,:,j].reshape(-1),MC[sl,:,j].reshape(-1))[0,1]),
              "target_std":float(np.std(Tn[sl,:,j])),
            })
          target_family[str(n)]=per

        value_quality=[]
        for j in range(4):
          y=MC[:,:,j].reshape(-1);p=V[:-1,:,j].reshape(-1)
          value_quality.append({
            "ev_vs_mc32":explained_variance(y,p),
            "bias_value_minus_mc":float(np.mean(p-y)),
            "value_std":float(np.std(p)),
            "mc_std":float(np.std(y)),
            "value_abs_mean":float(np.mean(np.abs(p))),
            "mc_abs_mean":float(np.mean(np.abs(y))),
          })

        # Critic gradient interference on the actual 2-step training-style target built from first 2 transitions.
        # Target = r0 + gamma r1 + gamma^2 V(s2), respecting done.
        o0=obs_seq[0];o1=obs_seq[1]
        r0=torch.tensor(R[0],device="cuda",dtype=torch.float32)
        r1=torch.tensor(R[1],device="cuda",dtype=torch.float32)
        d0=torch.tensor(D[0],device="cuda")
        d1=torch.tensor(D[1],device="cuda")
        with torch.no_grad():
          v2=m.value_with_preference(obs_seq[2],w)
          nt0=(~d0).float().unsqueeze(-1);nt1=(~d1).float().unsqueeze(-1)
          target2=r0 + GAMMA*nt0*r1 + (GAMMA**2)*nt0*nt1*v2
        pred=m.value_with_preference(o0,w)

        body_ps=[p for n,p in m.named_parameters() if n.startswith("critic_body")]
        head_w=m.critic_head.weight
        head_b=m.critic_head.bias
        body_g=[];head_g=[];head_loss=[]
        for j in range(4):
          lj=(pred[:,j]-target2[:,j].detach()).pow(2).mean()
          head_loss.append(float(lj.detach()))
          gb=torch.autograd.grad(lj,body_ps,retain_graph=True,allow_unused=True)
          body_g.append(flat(gb,body_ps).detach())
          gh=torch.autograd.grad(lj,[head_w,head_b],retain_graph=True,allow_unused=True)
          # Extract only row j influence from full head gradients for a comparable per-head vector.
          gw=gh[0][j].reshape(-1);gbi=gh[1][j].reshape(-1)
          head_g.append(torch.cat([gw,gbi]).detach())

        body_cos=np.array([[cosine(body_g[i],body_g[j]) for j in range(4)] for i in range(4)])
        body_norm=[float(g.norm()) for g in body_g]
        head_norm=[float(g.norm()) for g in head_g]

        # Hypothetical one critic-only Adam step on this fixed batch, measure same-batch loss movement.
        clone=T4SharedActorCritic(od,ad).cuda();clone.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
        opt=torch.optim.Adam([p for n,p in clone.named_parameters() if n.startswith("critic_")],lr=1e-3)
        before=clone.value_with_preference(o0,w)
        losses_before=[float(((before[:,j]-target2[:,j]).pow(2).mean()).detach()) for j in range(4)]
        total=((before-target2.detach()).pow(2)).mean()
        params=[p for n,p in clone.named_parameters() if n.startswith("critic_")]
        old=[p.detach().clone() for p in params]
        opt.zero_grad(set_to_none=True);total.backward();opt.step()
        after=clone.value_with_preference(o0,w)
        losses_after=[float(((after[:,j]-target2[:,j]).pow(2).mean()).detach()) for j in range(4)]
        step_norm=float(torch.sqrt(sum(((p-o)**2).sum() for p,o in zip(params,old))))
        param_norm=float(torch.sqrt(sum((o**2).sum() for o in old)))
        loss_change=[a-b for a,b in zip(losses_after,losses_before)]

        branch[str(snap)]={
          "value_quality_vs_mc32":value_quality,
          "target_family":target_family,
          "training_like_2step":{
            "head_loss":head_loss,
            "target_mean":target2.detach().mean(0).cpu().tolist(),
            "target_std":target2.detach().std(0).cpu().tolist(),
            "prediction_mean":pred.detach().mean(0).cpu().tolist(),
            "prediction_std":pred.detach().std(0).cpu().tolist(),
          },
          "shared_body_interference":{
            "body_grad_norm":body_norm,
            "head_grad_norm":head_norm,
            "body_grad_cosine":body_cos.tolist(),
            "mean_offdiag_body_cosine":float(np.mean([body_cos[i,j] for i in range(4) for j in range(i+1,4)])),
            "negative_body_pair_fraction":float(np.mean([body_cos[i,j]<0 for i in range(4) for j in range(i+1,4)])),
          },
          "critic_optimizer_same_batch":{
            "lr":1e-3,
            "loss_before_per_head":losses_before,
            "loss_after_per_head":losses_after,
            "loss_change_after_minus_before":loss_change,
            "all_heads_improved":bool(all(x<0 for x in loss_change)),
            "critic_param_step_norm":step_norm,
            "critic_param_norm":param_norm,
            "relative_step_norm":step_norm/(param_norm+1e-12),
          }
        }
      report["branches"][lab]=branch

    (outdir/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],
      "summary":{lab:{s:{
        "ev":[round(x["ev_vs_mc32"],3) for x in report["branches"][lab][s]["value_quality_vs_mc32"]],
        "2step_mae":[round(x["mae_vs_mc"],4) for x in report["branches"][lab][s]["target_family"]["2"]],
        "16step_mae":[round(x["mae_vs_mc"],4) for x in report["branches"][lab][s]["target_family"]["16"]],
        "body_cos":round(report["branches"][lab][s]["shared_body_interference"]["mean_offdiag_body_cosine"],3),
        "same_batch_improved":report["branches"][lab][s]["critic_optimizer_same_batch"]["all_heads_improved"],
        "rel_step":report["branches"][lab][s]["critic_optimizer_same_batch"]["relative_step_norm"],
      } for s in ("0","10","50","100")} for lab in ORDER}},indent=2))
  except BaseException as e:
    (outdir/"ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
  finally:
    if env is not None:env.close()
    app.close()
if __name__=="__main__":main()
