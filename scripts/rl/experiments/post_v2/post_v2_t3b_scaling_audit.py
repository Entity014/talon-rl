#!/usr/bin/env python3
"""T3-B read-only scaling/normalization audit for frozen 4D MORL vector."""
from __future__ import annotations
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
OBJ=("velocity_tracking","angular_stability","orientation_stability","control_smoothness")
TERMS={
 "velocity_tracking":("track_lin_vel_xy_exp","track_ang_vel_z_exp"),
 "angular_stability":("ang_vel_xy_l2",),
 "orientation_stability":("flat_orientation_l2",),
 "control_smoothness":("action_rate_l2",),
}
PHYS_REF={
 # reward-space magnitude at a documented unit/reference condition.
 # tracking: max of frozen positive terms 1.5+0.75.
 # angular: |w_xy|^2 = 1 -> 0.05.
 # orientation: projected-gravity xy norm^2 = 1 -> 2.5.
 # control: summed squared action change = 1 -> 0.01.
 "velocity_tracking":2.25,
 "angular_stability":0.05,
 "orientation_stability":2.5,
 "control_smoothness":0.01,
}
EPS=1e-8
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def objective_vec(raw,names):
    idx={n:i for i,n in enumerate(names)}
    cols=[]
    for o in OBJ:
        cols.append(sum(raw[:,idx[t]] for t in TERMS[o]))
    return np.stack(cols,1).astype(np.float32)
def discounted_rtg(rew,done,gamma=.99):
    # [T,N,O]
    out=np.zeros_like(rew,dtype=np.float64);running=np.zeros_like(rew[0],dtype=np.float64)
    for t in range(len(rew)-1,-1,-1):
        running=rew[t]+gamma*running*(~done[t])[:,None]
        out[t]=running
    return out
def flat_grad(grads):
    xs=[g.reshape(-1) for g in grads if g is not None]
    return torch.cat(xs) if xs else torch.zeros(1,device="cuda")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
    ap.add_argument("--num-envs",type=int,default=16)
    ap.add_argument("--steps",type=int,default=192)
    ap.add_argument("--reset-seeds",type=int,nargs="+",default=[230001,230101,230201])
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.v1c_actor_critic import V1CSharedActorCritic,initialize_from_rsl_m01
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.reset_seeds[0]
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        obs,_=env.reset(seed=args.reset_seeds[0]);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
        model=V1CSharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(model,args.checkpoint,device="cpu");model.eval()
        w=torch.tensor([1.,0.,0.],device="cuda").repeat(args.num_envs,1)
        all_reward=[]; seed_rows=[]; grad_records=[]
        actor_params=[p for n,p in model.named_parameters() if n.startswith("actor_") or n=="log_std"]
        for si,seed in enumerate(args.reset_seeds):
            cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
            obs_b=[];act_b=[];rew_b=[];done_b=[]
            for _ in range(args.steps):
                with torch.no_grad():a,_=model.act_with_preference(cur,w)
                nxt,_,term,trunc,_=env.step(torch.clamp(a,-1,1))
                mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy().astype(np.float64);names=list(mgr.active_terms)
                vec=objective_vec(raw,names)
                obs_b.append(cur.detach().cpu());act_b.append(a.detach().cpu());rew_b.append(vec);done_b.append((term|trunc).cpu().numpy().astype(bool))
                cur=obs_tensor(nxt).cuda()
            R=np.asarray(rew_b);D=np.asarray(done_b);all_reward.append(R.reshape(-1,4))
            seed_rows.append({"seed":seed,"mean":R.mean((0,1)).tolist(),"std":R.std((0,1)).tolist(),
                              "nonzero_fraction":(np.abs(R)>0).mean((0,1)).tolist()})
            # Keep rollout for later gradient audit.
            grad_records.append((torch.cat(obs_b).cuda(),torch.cat(act_b).cuda(),R,D))
        A=np.concatenate(all_reward,0)
        # distribution statistics
        q=np.quantile(A,[.01,.05,.25,.5,.75,.95,.99],axis=0)
        med=np.median(A,0);mad=np.median(np.abs(A-med),0)
        scales={
          "raw":np.ones(4),
          "std":A.std(0),
          "iqr_sigma":(np.quantile(A,.75,axis=0)-np.quantile(A,.25,axis=0))/1.349,
          "mad_sigma":1.4826*mad,
          "abs_mean":np.mean(np.abs(A),0),
          "physical_reference":np.array([PHYS_REF[o] for o in OBJ],float),
        }
        scales={k:np.maximum(v,EPS) for k,v in scales.items()}
        dist={
          "mean":A.mean(0).tolist(),"std":A.std(0).tolist(),"abs_mean":np.mean(np.abs(A),0).tolist(),
          "nonzero_fraction":(np.abs(A)>0).mean(0).tolist(),
          "quantiles":{str(p):q[i].tolist() for i,p in enumerate([.01,.05,.25,.5,.75,.95,.99])},
          "mad":mad.tolist(),"iqr":(np.quantile(A,.75,axis=0)-np.quantile(A,.25,axis=0)).tolist(),
          "seed_rows":seed_rows,
        }
        candidates={}
        for cname,scale in scales.items():
            norm=A/scale
            # uniform scalar contribution in reward space
            abs_contrib=np.mean(np.abs(0.25*norm),0)
            reward_share=abs_contrib/(abs_contrib.sum()+1e-12)
            seed_reward_shares=[]
            grad_norms_seed=[]
            adv_std_seed=[]
            grad_cos_seed=[]
            for rec_i,(fo,fa,R,D) in enumerate(grad_records):
                rtg=discounted_rtg(R/scale,D)
                # advantage proxy: center RTG per objective over the sampled rollout.
                adv=rtg.reshape(-1,4);adv=adv-adv.mean(0,keepdims=True)
                adv_std_seed.append(adv.std(0).tolist())
                logp=model.logp_with_preference(fo,w.repeat(args.steps,1),fa)
                gvec=[]
                for j in range(4):
                    aj=torch.as_tensor(adv[:,j],device="cuda",dtype=logp.dtype)
                    loss=-(logp*aj.detach()).mean()
                    g=flat_grad(torch.autograd.grad(loss,actor_params,retain_graph=True,allow_unused=True))
                    gvec.append(g)
                norms=np.array([float(x.norm()) for x in gvec])
                grad_norms_seed.append(norms.tolist())
                # pairwise gradient cosine
                C=np.eye(4)
                for i in range(4):
                    for j in range(i+1,4):
                        C[i,j]=C[j,i]=float((gvec[i]@gvec[j])/(gvec[i].norm()*gvec[j].norm()+1e-12))
                grad_cos_seed.append(C.tolist())
                rr=R.reshape(-1,4)/scale
                ac=np.mean(np.abs(.25*rr),0);seed_reward_shares.append((ac/(ac.sum()+1e-12)).tolist())
            G=np.asarray(grad_norms_seed)
            gmean=G.mean(0);gshare=gmean/(gmean.sum()+1e-12)
            Gshare=G/(G.sum(1,keepdims=True)+1e-12)
            candidates[cname]={
              "divisor":scale.tolist(),
              "reward_abs_contribution_share":reward_share.tolist(),
              "reward_dominance_max_share":float(reward_share.max()),
              "reward_share_seed_std":np.std(np.asarray(seed_reward_shares),axis=0).tolist(),
              "advantage_proxy_std_mean":np.mean(np.asarray(adv_std_seed),axis=0).tolist(),
              "actor_gradient_norm_mean":gmean.tolist(),
              "actor_gradient_share":gshare.tolist(),
              "actor_gradient_share_seed_std":Gshare.std(0).tolist(),
              "actor_gradient_norm_seed_cv":(G.std(0)/(G.mean(0)+1e-12)).tolist(),
              "actor_gradient_dominance_max_share":float(gshare.max()),
              "actor_gradient_max_min_ratio":float(gmean.max()/(gmean.min()+1e-12)),
              "actor_gradient_pairwise_cosine_mean":np.mean(np.asarray(grad_cos_seed),axis=0).tolist(),
            }
        report={
          "schema":"v2_t3b_scaling_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
          "objective_order":list(OBJ),"sign_convention":"All four are weighted rewards with higher=better; three penalties are negative-valued and improve toward zero.",
          "source_checkpoint":str(args.checkpoint),"source_checkpoint_sha256":sha(args.checkpoint),
          "protocol":{"num_envs":args.num_envs,"steps":args.steps,"reset_seeds":args.reset_seeds,"stochastic_policy_actions":True},
          "raw_distribution":dist,"normalization_candidates":candidates,
          "physical_reference_definition":PHYS_REF,
          "advantage_note":"Advantage scale uses centered discounted return-to-go as a read-only policy-gradient proxy; no critic training or optimizer step is performed.",
          "effort_excluded":True,
        }
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"distribution":dist,"candidate_summary":{k:{
          "divisor":v["divisor"],"reward_share":v["reward_abs_contribution_share"],"grad_share":v["actor_gradient_share"],
          "grad_ratio":v["actor_gradient_max_min_ratio"],"reward_dom":v["reward_dominance_max_share"],"grad_dom":v["actor_gradient_dominance_max_share"]
        } for k,v in candidates.items()}},indent=2))
    except BaseException as exc:
        args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");raise
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
