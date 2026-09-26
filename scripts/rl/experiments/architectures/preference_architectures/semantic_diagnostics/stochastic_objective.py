"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_post_v2_t5_c40_derivative_equivalence():
    """Run former post_v2_t5_c40_derivative_equivalence.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c40_derivative_equivalence-2026-09-23"
    OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32)
    J=1;NENV=8;HS=(.03,.015,.0075,.00375)
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1)
        return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def flat(gs,ps):
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def rel_l2(a,b): return float((a-b).norm()/(b.norm()+1e-12))
    
    def reset_t2(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for _ in range(2):
            with torch.no_grad(): act=m.act_inference_with_preference(cur,w)
            nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
        return cur.detach()
    def step_reward(env,action):
        from talon_rl.rewards.objectives import normalized_objective_vector
        env.step(action)
        raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy()
        names=list(env.unwrapped.reward_manager.active_terms)
        vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        return torch.tensor(vec[:,J],device="cuda",dtype=torch.float32)*env.unwrapped.step_dt
    
    def fd_action(env,m,w,seed,a0,h):
        ad=a0.shape[1];cols=[]
        for d in range(ad):
            ap=a0.clone();am=a0.clone();ap[:,d]+=h;am[:,d]-=h
            _=reset_t2(env,m,w,seed);rp=step_reward(env,ap)
            _=reset_t2(env,m,w,seed);rm=step_reward(env,am)
            cols.append((rp-rm)/(2*h))
        return torch.stack(cols,dim=1)
    
    def fd_mu(env,m,w,seed,loc,h):
        ad=loc.shape[1];cols=[]
        for d in range(ad):
            up=loc.clone();um=loc.clone();up[:,d]+=h;um[:,d]-=h
            ap=torch.tanh(up)*m.ACTION_CLIP
            am=torch.tanh(um)*m.ACTION_CLIP
            _=reset_t2(env,m,w,seed);rp=step_reward(env,ap)
            _=reset_t2(env,m,w,seed);rm=step_reward(env,am)
            cols.append((rp-rm)/(2*h))
        return torch.stack(cols,dim=1)
    
    def process_slope(env,a0,h):
        am=env.unwrapped.action_manager;term=am._terms["joint_pos"]
        slopes=[]
        for d in range(a0.shape[1]):
            ap=a0.clone();amv=a0.clone();ap[:,d]+=h;amv[:,d]-=h
            am.process_action(ap);pp=term.processed_actions.detach().clone()
            am.process_action(amv);pm=term.processed_actions.detach().clone()
            slopes.append(((pp-pm)/(2*h))[:,d])
        am.process_action(a0)
        return torch.stack(slopes,dim=1),repr(term.cfg)
    def c34_ref(snap,suite):
        g=[]
        for d in range(12):
            p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"))
            m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
            g.append((p["angular_obj"]-m["angular_obj"])/.06)
        return np.asarray(g,np.float32)
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--snap",type=int,required=True)
        ap.add_argument("--suite",type=int,required=True)
        a=ap.parse_args()
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1]
            ad=env.unwrapped.action_manager.total_action_dim
            m=T4SharedActorCritic(od,ad).cuda()
            state=torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)
            m.load_state_dict(state["model"]);m.eval()
            w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
            seed=7310000+a.snap*10000+a.suite*211
            O=reset_t2(env,m,w,seed)
            with torch.no_grad():
                loc=m._pre_tanh_dist(m._with_w(O,w)).loc
                a0=torch.tanh(loc)*m.ACTION_CLIP
                jac=m.ACTION_CLIP*(1.0-torch.tanh(loc).pow(2))
            proc_slope,action_cfg=process_slope(env,a0,HS[0])
            out={"schema":"c40_zero_variance_derivative_equivalence_v1",
                 "snap":a.snap,"suite":a.suite,"seed":seed,"h":list(HS),
                 "action_path":{"policy_action_clip":float(m.ACTION_CLIP),
                                "action_term_cfg":action_cfg,
                                "processed_diag_slope_mean":proc_slope.mean(0).cpu().tolist(),
                                "processed_diag_slope_min":proc_slope.min(0).values.cpu().tolist(),
                                "processed_diag_slope_max":proc_slope.max(0).values.cpu().tolist()},
                 "loc":loc.cpu().tolist(),"policy_action":a0.cpu().tolist(),
                 "tanh_jacobian_diag":jac.cpu().tolist(),"steps":{}}
            prev_err=None;ga_h03=None
            for h in HS:
                ga=fd_action(env,m,w,seed,a0,h)
                if abs(h-.03)<1e-12: ga_h03=ga.detach()
                gmu=fd_mu(env,m,w,seed,loc,h)
                chain=jac*ga
                ga_mean=ga.mean(0);gmu_mean=gmu.mean(0);chain_mean=chain.mean(0)
                naive_chain=jac.mean(0)*ga_mean
                err=torch.linalg.vector_norm(gmu-chain,dim=0)
                den=torch.linalg.vector_norm(chain,dim=0)+1e-12
                per_dof_rel=err/den
                mask=(torch.abs(gmu)>1e-8)|(torch.abs(chain)>1e-8)
                sign=float(((torch.sign(gmu)==torch.sign(chain))|(~mask)).float().mean())
                total_err=rel_l2(gmu,chain)
                order=None if prev_err is None else float(np.log2(prev_err/(total_err+1e-30)))
                prev_err=total_err
                rec={"cos_flat":cos(gmu,chain),"norm_ratio_flat":float(gmu.norm()/(chain.norm()+1e-12)),
                     "relative_l2_flat":total_err,"sign_agreement":sign,
                     "observed_order_from_prev_h":order,
                     "cos_mean":cos(gmu_mean,chain_mean),
                     "norm_ratio_mean":float(gmu_mean.norm()/(chain_mean.norm()+1e-12)),
                     "mean_relative_l2":rel_l2(gmu_mean,chain_mean),
                     "aggregation_naive_vs_correct_cos":cos(naive_chain,chain_mean),
                     "aggregation_naive_vs_correct_rel_l2":rel_l2(naive_chain,chain_mean),
                     "per_dof_relative_error":per_dof_rel.cpu().tolist(),
                     "g_action_mean":ga_mean.cpu().tolist(),
                     "g_mu_fd_mean":gmu_mean.cpu().tolist(),
                     "g_mu_chain_mean":chain_mean.cpu().tolist(),
                     "g_mu_naive_aggregate_chain":naive_chain.cpu().tolist()}
                if abs(h-.03)<1e-12:
                    ref=torch.tensor(c34_ref(a.snap,a.suite),device="cuda")*env.unwrapped.step_dt
                    rec["c34_vs_c40_action_cos"]=cos(ref,ga_mean)
                    rec["c34_vs_c40_action_norm_ratio"]=float(ref.norm()/(ga_mean.norm()+1e-12))
                out["steps"][str(h)]=rec
                print("H",h,rec,flush=True)
            actor_ps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
            a_graph=m.act_inference_with_preference(O,w)
            g_old=flat(torch.autograd.grad((a_graph*ga_h03.mean(0)).sum(-1).mean(),actor_ps,retain_graph=True,allow_unused=True),actor_ps).detach()
            g_correct=flat(torch.autograd.grad((a_graph*ga_h03).sum(-1).mean(),actor_ps,allow_unused=True),actor_ps).detach()
            out["actor_parameter_chain"]={"old_aggregate_vs_correct_cos":cos(g_old,g_correct),
                "old_aggregate_vs_correct_rel_l2":rel_l2(g_old,g_correct),
                "old_norm":float(g_old.norm()),"correct_norm":float(g_correct.norm())}
            rels=np.asarray([out["steps"][str(h)]["per_dof_relative_error"] for h in HS])
            out["convergence"]={"per_dof_monotone_nonincreasing":
                [bool(np.all(np.diff(rels[:,d])<=1e-9)) for d in range(ad)],
                "dofs_error_not_reducing":[int(d) for d in range(ad) if rels[-1,d]>=rels[0,d]]}
            fn=OUT/f"audit_u{a.snap}_s{a.suite}.json"
            fn.write_text(json.dumps(out,indent=2)+"\n")
            print("WROTE",fn,flush=True)
        finally:
            if env is not None: env.close()
            app.close()
    if True: main()

def run_post_v2_t5_c41_corrected_gradient_compare():
    """Run former post_v2_t5_c41_corrected_gradient_compare.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c41_corrected_gradient_compare-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;NENV=8;HFD=0.03
    BUDGETS=(64,256,1024,4096)
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps): return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b): return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def reset_t2(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for _ in range(2):
            with torch.no_grad():a=m.act_inference_with_preference(cur,w)
            nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        return cur
    def step_reward(env,action):
        from talon_rl.rewards.objectives import normalized_objective_vector
        env.step(action)
        raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy()
        names=list(env.unwrapped.reward_manager.active_terms)
        vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        return torch.tensor(vec[:,J],device="cuda",dtype=torch.float32)*env.unwrapped.step_dt
    def fd_action_per_env(env,m,w,seed,a0,h):
        cols=[]
        for d in range(a0.shape[1]):
            ap=a0.clone();am=a0.clone();ap[:,d]+=h;am[:,d]-=h
            _=reset_t2(env,m,w,seed);rp=step_reward(env,ap)
            _=reset_t2(env,m,w,seed);rm=step_reward(env,am)
            cols.append((rp-rm)/(2*h))
        return torch.stack(cols,dim=1)
    def project_mean(m,ps,O,w,g,eta=1e-4):
        gn=float(g.norm())
        if gn<1e-12:return torch.zeros(m.actor_mean.out_features,device="cuda")
        d=g/(g.norm()+1e-12)
        o=0
        with torch.no_grad():
            for p in ps:
                n=p.numel();p.add_(eta*d[o:o+n].view_as(p));o+=n
            ap=m.act_inference_with_preference(O,w).mean(0)
            o=0
            for p in ps:
                n=p.numel();p.add_(-2*eta*d[o:o+n].view_as(p));o+=n
            am=m.act_inference_with_preference(O,w).mean(0)
            o=0
            for p in ps:
                n=p.numel();p.add_(eta*d[o:o+n].view_as(p));o+=n
        return ((ap-am)/(2*eta))*gn
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--draws",type=int,default=512);a=ap.parse_args()
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
            m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
            named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"];ps=[p for _,p in named]
            actor_named=[(n,p) for n,p in named if n.startswith("actor_")];actor_ps=[p for _,p in actor_named]
            w=torch.tensor(WREF,device="cuda").repeat(NENV,1);seed=7310000+a.snap*10000+a.suite*211
            O0=reset_t2(env,m,w,seed).detach()
            with torch.no_grad():
                loc0=m._pre_tanh_dist(m._with_w(O0,w)).loc
                a0=torch.tanh(loc0)*m.ACTION_CLIP
            ga_i=fd_action_per_env(env,m,w,seed,a0,HFD).detach()
            print("C41_STAGE ga_i_done",float(ga_i.norm()),flush=True)
            # Correct deterministic batch gradient: preserve per-env pairing before mean.
            a_graph=m.act_inference_with_preference(O0,w)
            gdet_actor=flat(torch.autograd.grad((a_graph*ga_i).sum(-1).mean(),actor_ps,retain_graph=True,allow_unused=True),actor_ps).detach()
            print("C41_STAGE gdet_actor_done",float(gdet_actor.norm()),flush=True)
            # merge zero log_std block so geometry is comparable to score/path gradients
            parts=[];off=0
            for n,p in named:
                if n=="log_std":parts.append(torch.zeros_like(p).reshape(-1))
                else:
                    nn=p.numel();parts.append(gdet_actor[off:off+nn]);off+=nn
            gdet=torch.cat(parts)
            print("C41_STAGE gdet_merge_done",float(gdet.norm()),flush=True)
            # Keep old invalid comparator for audit only. Do this before any
            # in-place parameter projection so the retained graph stays valid.
            ga_bar=ga_i.mean(0)
            gdet_old=flat(torch.autograd.grad((a_graph*ga_bar).sum(-1).mean(),ps,allow_unused=True),ps).detach()
            print("C41_STAGE old_done",float(gdet_old.norm()),flush=True)
            ddet=project_mean(m,ps,O0,w,gdet)
            out={"schema":"c41_corrected_gradient_compare_v1","snap":a.snap,"suite":a.suite,"draws":a.draws,
                 "deterministic":{"correct_norm":float(gdet.norm()),"old_norm":float(gdet_old.norm()),
                                  "old_vs_correct_cos":cos(gdet_old,gdet),
                                  "ga_mean":ga_bar.cpu().tolist(),
                                  "ga_per_env":ga_i.cpu().tolist()},"results":{}}
            score_sum=torch.zeros_like(gdet);path_sum=torch.zeros_like(gdet)
            gen=torch.Generator(device="cuda");gen.manual_seed(9910000+a.snap*10000+a.suite*100)
            for draw in range(a.draws):
                z=torch.randn((NENV,ad),device="cuda",generator=gen)
                dl=(torch.randint(0,2,(NENV,ad),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
                ds=(torch.randint(0,2,(ad,),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
                # score-function
                _=reset_t2(env,m,w,seed)
                dist0=m._pre_tanh_dist(m._with_w(O0,w));loc=dist0.loc;std=dist0.scale
                u=(loc.detach()+std.detach()*z).detach();act=torch.tanh(u)*m.ACTION_CLIP
                r=step_reward(env,act.detach())
                logp=(dist0.log_prob(u)-m._log_det_jacobian(u)).sum(-1)
                gscore=flat(torch.autograd.grad((logp*r.detach()).mean(),ps,retain_graph=True,allow_unused=True),ps).detach()
                # pathwise actor mean FD
                _=reset_t2(env,m,w,seed)
                rp=step_reward(env,torch.tanh(loc.detach()+std.detach()*z+HFD*dl)*m.ACTION_CLIP)
                _=reset_t2(env,m,w,seed)
                rm=step_reward(env,torch.tanh(loc.detach()+std.detach()*z-HFD*dl)*m.ACTION_CLIP)
                gu=((rp-rm)/(2*HFD))[:,None]*dl
                gloc=m._pre_tanh_dist(m._with_w(O0,w)).loc
                gactor=flat(torch.autograd.grad((gloc*gu.detach()).sum(-1).mean(),actor_ps,allow_unused=True),actor_ps).detach()
                # pathwise log_std FD
                _=reset_t2(env,m,w,seed);stdp=std.detach()*torch.exp(HFD*ds)
                rsp=step_reward(env,torch.tanh(loc.detach()+stdp*z)*m.ACTION_CLIP)
                _=reset_t2(env,m,w,seed);stdm=std.detach()*torch.exp(-HFD*ds)
                rsm=step_reward(env,torch.tanh(loc.detach()+stdm*z)*m.ACTION_CLIP)
                glog=((rsp-rsm).mean()/(2*HFD))*ds
                parts=[];off=0
                for n,p in named:
                    if n=="log_std":parts.append(glog.reshape(-1))
                    else:
                        nn=p.numel();parts.append(gactor[off:off+nn]);off+=nn
                gpath=torch.cat(parts)
                score_sum+=gscore;path_sum+=gpath
                ns=(draw+1)*NENV
                if ns in BUDGETS:
                    gs=score_sum/(draw+1);gp=path_sum/(draw+1)
                    dscr=project_mean(m,ps,O0,w,gs);dpth=project_mean(m,ps,O0,w,gp)
                    out["results"][str(ns)]={"score_cos_correct_det":cos(gs,gdet),
                        "path_cos_correct_det":cos(gp,gdet),"score_cos_path":cos(gs,gp),
                        "score_cos_old_det":cos(gs,gdet_old),"path_cos_old_det":cos(gp,gdet_old),
                        "score_norm":float(gs.norm()),"path_norm":float(gp.norm()),"det_norm":float(gdet.norm()),
                        "score_action_cos_det_action":cos(dscr,ddet),"path_action_cos_det_action":cos(dpth,ddet)}
                    print("BUDGET",ns,out["results"][str(ns)],flush=True)
            fn=OUT/f"audit_u{a.snap}_s{a.suite}.json";fn.write_text(json.dumps(out,indent=2)+"\n");print("WROTE",fn,flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_post_v2_t5_c42_highsample_confidence():
    """Run former post_v2_t5_c42_highsample_confidence.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c42_highsample_confidence-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;NENV=8;HFD=0.03
    BUDGETS=(4096,8192,16384)
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos_np(a,b):
        na=np.linalg.norm(a);nb=np.linalg.norm(b)
        return float(np.dot(a,b)/(na*nb+1e-12))
    def reset_t2(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for _ in range(2):
            with torch.no_grad():act=m.act_inference_with_preference(cur,w)
            nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
        return cur
    def step_reward(env,action):
        from talon_rl.rewards.objectives import normalized_objective_vector
        env.step(action)
        raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy()
        names=list(env.unwrapped.reward_manager.active_terms)
        vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        return torch.tensor(vec[:,J],device="cuda",dtype=torch.float32)*env.unwrapped.step_dt
    def fd_action_per_env(env,m,w,seed,a0,h):
        cols=[]
        for d in range(a0.shape[1]):
            ap=a0.clone();am=a0.clone();ap[:,d]+=h;am[:,d]-=h
            _=reset_t2(env,m,w,seed);rp=step_reward(env,ap)
            _=reset_t2(env,m,w,seed);rm=step_reward(env,am)
            cols.append((rp-rm)/(2*h))
        return torch.stack(cols,dim=1)
    def bootstrap_ci(X,det,reps=1000,seed=0):
        # Bootstrap over per-draw gradient vectors; each draw already averages NENV envs.
        rng=np.random.default_rng(seed);n=len(X);vals=np.empty(reps,np.float64)
        for k in range(reps):
            idx=rng.integers(0,n,size=n);vals[k]=cos_np(X[idx].mean(0),det)
        return {"mean":float(vals.mean()),"lo95":float(np.percentile(vals,2.5)),
                "hi95":float(np.percentile(vals,97.5)),"sd":float(vals.std(ddof=1))}
    def snr_stats(X,det):
        mu=X.mean(0);noise=X-mu
        vartrace=float(np.mean(np.sum(noise*noise,axis=1)))
        sem_rms=float(np.sqrt(vartrace/len(X)))
        return {"estimate_norm":float(np.linalg.norm(mu)),"det_norm":float(np.linalg.norm(det)),
                "draw_vartrace":vartrace,"sem_rms":sem_rms,
                "det_norm_over_sem_rms":float(np.linalg.norm(det)/(sem_rms+1e-12)),
                "estimate_norm_over_sem_rms":float(np.linalg.norm(mu)/(sem_rms+1e-12))}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--suite",type=int,required=True,choices=[0,1])
        ap.add_argument("--bootstrap",type=int,default=1000);a=ap.parse_args();snap=25
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
            m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/"A_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
            named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"];ps=[p for _,p in named]
            actor_ps=[p for n,p in named if n.startswith("actor_")]
            w=torch.tensor(WREF,device="cuda").repeat(NENV,1);seed=7310000+25*10000+a.suite*211
            O=reset_t2(env,m,w,seed).detach()
            with torch.no_grad():
                loc0=m._pre_tanh_dist(m._with_w(O,w)).loc;a0=torch.tanh(loc0)*m.ACTION_CLIP
            ga_i=fd_action_per_env(env,m,w,seed,a0,HFD).detach()
            agraph=m.act_inference_with_preference(O,w)
            gd_actor=flat(torch.autograd.grad((agraph*ga_i).sum(-1).mean(),actor_ps,allow_unused=True),actor_ps).detach()
            parts=[];off=0
            for n,p in named:
                if n=="log_std":parts.append(torch.zeros_like(p).reshape(-1))
                else:
                    nn=p.numel();parts.append(gd_actor[off:off+nn]);off+=nn
            gdet=torch.cat(parts).cpu().numpy()
            max_draws=max(BUDGETS)//NENV
            score_draws=[];path_draws=[]
            gen=torch.Generator(device="cuda");gen.manual_seed(9910000+25*10000+a.suite*100)
            out={"schema":"c42_highsample_confidence_v1","snap":25,"suite":a.suite,
                 "deterministic":{"norm":float(np.linalg.norm(gdet)),"ga_per_env_norms":torch.linalg.vector_norm(ga_i,dim=1).cpu().tolist()},
                 "budgets":{}}
            for draw in range(max_draws):
                z=torch.randn((NENV,ad),device="cuda",generator=gen)
                dl=(torch.randint(0,2,(NENV,ad),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
                ds=(torch.randint(0,2,(ad,),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
                _=reset_t2(env,m,w,seed)
                dist=m._pre_tanh_dist(m._with_w(O,w));loc=dist.loc;std=dist.scale
                u=(loc.detach()+std.detach()*z).detach();act=torch.tanh(u)*m.ACTION_CLIP
                r=step_reward(env,act.detach());logp=(dist.log_prob(u)-m._log_det_jacobian(u)).sum(-1)
                gs=flat(torch.autograd.grad((logp*r.detach()).mean(),ps,retain_graph=True,allow_unused=True),ps).detach().cpu().numpy()
                _=reset_t2(env,m,w,seed);rp=step_reward(env,torch.tanh(loc.detach()+std.detach()*z+HFD*dl)*m.ACTION_CLIP)
                _=reset_t2(env,m,w,seed);rm=step_reward(env,torch.tanh(loc.detach()+std.detach()*z-HFD*dl)*m.ACTION_CLIP)
                gu=((rp-rm)/(2*HFD))[:,None]*dl
                gloc=m._pre_tanh_dist(m._with_w(O,w)).loc
                ga=flat(torch.autograd.grad((gloc*gu.detach()).sum(-1).mean(),actor_ps,allow_unused=True),actor_ps).detach()
                _=reset_t2(env,m,w,seed);rsp=step_reward(env,torch.tanh(loc.detach()+(std.detach()*torch.exp(HFD*ds))*z)*m.ACTION_CLIP)
                _=reset_t2(env,m,w,seed);rsm=step_reward(env,torch.tanh(loc.detach()+(std.detach()*torch.exp(-HFD*ds))*z)*m.ACTION_CLIP)
                glog=((rsp-rsm).mean()/(2*HFD))*ds
                parts=[];off=0
                for n,p in named:
                    if n=="log_std":parts.append(glog.reshape(-1))
                    else:
                        nn=p.numel();parts.append(ga[off:off+nn]);off+=nn
                gp=torch.cat(parts).cpu().numpy()
                score_draws.append(gs);path_draws.append(gp)
                ns=(draw+1)*NENV
                if ns in BUDGETS:
                    S=np.stack(score_draws);P=np.stack(path_draws)
                    sm=S.mean(0);pm=P.mean(0)
                    rec={"score_cos_det":cos_np(sm,gdet),"path_cos_det":cos_np(pm,gdet),
                         "score_cos_path":cos_np(sm,pm),
                         "score_ci95":bootstrap_ci(S,gdet,a.bootstrap,7000+ns+a.suite),
                         "path_ci95":bootstrap_ci(P,gdet,a.bootstrap,9000+ns+a.suite),
                         "score_snr":snr_stats(S,gdet),"path_snr":snr_stats(P,gdet)}
                    out["budgets"][str(ns)]=rec
                    print("BUDGET",ns,json.dumps(rec),flush=True)
            fn=OUT/f"audit_u25_s{a.suite}.json";fn.write_text(json.dumps(out,indent=2)+"\n");print("WROTE",fn,flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_post_v2_t5_c43_corrected_variance_limit():
    """Run former post_v2_t5_c43_corrected_variance_limit.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c43_corrected_variance_limit-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;NENV=8;HFD=0.03
    SCALES=(1.0,.5,.25,.125,.0625);DRAWS=512
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def reset_t2(env,m,w,seed):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for _ in range(2):
            with torch.no_grad():act=m.act_inference_with_preference(cur,w)
            nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
        return cur
    def step_reward(env,action):
        from talon_rl.rewards.objectives import normalized_objective_vector
        env.step(action)
        raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy()
        names=list(env.unwrapped.reward_manager.active_terms)
        vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        return torch.tensor(vec[:,J],device="cuda",dtype=torch.float32)*env.unwrapped.step_dt
    def fd_action_per_env(env,m,w,seed,a0,h):
        cols=[]
        for d in range(a0.shape[1]):
            ap=a0.clone();am=a0.clone();ap[:,d]+=h;am[:,d]-=h
            _=reset_t2(env,m,w,seed);rp=step_reward(env,ap)
            _=reset_t2(env,m,w,seed);rm=step_reward(env,am)
            cols.append((rp-rm)/(2*h))
        return torch.stack(cols,dim=1)
    def bootstrap_cos(X,det,reps=600,seed=0):
        rng=np.random.default_rng(seed);n=len(X);vals=np.empty(reps)
        for k in range(reps):
            idx=rng.integers(0,n,size=n);g=X[idx].mean(0)
            vals[k]=float(np.dot(g,det)/(np.linalg.norm(g)*np.linalg.norm(det)+1e-12))
        return {"lo95":float(np.percentile(vals,2.5)),"hi95":float(np.percentile(vals,97.5)),
                "mean":float(vals.mean()),"sd":float(vals.std(ddof=1))}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--suite",type=int,required=True,choices=[0,1]);ap.add_argument("--draws",type=int,default=DRAWS);a=ap.parse_args()
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
            m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/"A_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
            named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"];ps=[p for _,p in named]
            actor_ps=[p for n,p in named if n.startswith("actor_")]
            w=torch.tensor(WREF,device="cuda").repeat(NENV,1);seed=7310000+25*10000+a.suite*211
            O=reset_t2(env,m,w,seed).detach()
            dist0=m._pre_tanh_dist(m._with_w(O,w));loc0=dist0.loc.detach();std0=dist0.scale.detach()
            a0=torch.tanh(loc0)*m.ACTION_CLIP
            ga_i=fd_action_per_env(env,m,w,seed,a0,HFD).detach()
            agraph=m.act_inference_with_preference(O,w)
            gd_actor=flat(torch.autograd.grad((agraph*ga_i).sum(-1).mean(),actor_ps,allow_unused=True),actor_ps).detach()
            parts=[];off=0
            for n,p in named:
                if n=="log_std":parts.append(torch.zeros_like(p).reshape(-1))
                else:
                    nn=p.numel();parts.append(gd_actor[off:off+nn]);off+=nn
            gdet=torch.cat(parts)
            out={"schema":"c43_corrected_variance_limit_v1","snap":25,"suite":a.suite,"draws":a.draws,
                 "det_norm":float(gdet.norm()),"scales":{}}
            gen=torch.Generator(device="cuda");gen.manual_seed(12000000+a.suite*1000)
            Z=[torch.randn((NENV,ad),device="cuda",generator=gen) for _ in range(a.draws)]
            DL=[(torch.randint(0,2,(NENV,ad),device="cuda",generator=gen,dtype=torch.int64)*2-1).float() for _ in range(a.draws)]
            DS=[(torch.randint(0,2,(ad,),device="cuda",generator=gen,dtype=torch.int64)*2-1).float() for _ in range(a.draws)]
            det_np=gdet.cpu().numpy()
            for sc in SCALES:
                rows=[]
                for z,dl,ds in zip(Z,DL,DS):
                    std=std0*sc
                    _=reset_t2(env,m,w,seed)
                    rp=step_reward(env,torch.tanh(loc0+std*z+HFD*dl)*m.ACTION_CLIP)
                    _=reset_t2(env,m,w,seed)
                    rm=step_reward(env,torch.tanh(loc0+std*z-HFD*dl)*m.ACTION_CLIP)
                    gu=((rp-rm)/(2*HFD))[:,None]*dl
                    gloc=m._pre_tanh_dist(m._with_w(O,w)).loc
                    gactor=flat(torch.autograd.grad((gloc*gu.detach()).sum(-1).mean(),actor_ps,allow_unused=True),actor_ps).detach()
                    _=reset_t2(env,m,w,seed)
                    rsp=step_reward(env,torch.tanh(loc0+(std*torch.exp(HFD*ds))*z)*m.ACTION_CLIP)
                    _=reset_t2(env,m,w,seed)
                    rsm=step_reward(env,torch.tanh(loc0+(std*torch.exp(-HFD*ds))*z)*m.ACTION_CLIP)
                    glog=((rsp-rsm).mean()/(2*HFD))*ds
                    parts=[];off=0
                    for n,p in named:
                        if n=="log_std":parts.append(glog.reshape(-1))
                        else:
                            nn=p.numel();parts.append(gactor[off:off+nn]);off+=nn
                    rows.append(torch.cat(parts).cpu().numpy())
                X=np.stack(rows);gm=X.mean(0)
                c=float(np.dot(gm,det_np)/(np.linalg.norm(gm)*np.linalg.norm(det_np)+1e-12))
                rec={"cos_det":c,"one_minus_cos":1.0-c,"sigma_scale":sc,"sigma2_scale":sc*sc,
                     "path_norm":float(np.linalg.norm(gm)),"ci95":bootstrap_cos(X,det_np,600,13000+int(sc*10000)+a.suite)}
                out["scales"][str(sc)]=rec
                print("SCALE",sc,json.dumps(rec),flush=True)
            xs=np.array([s*s for s in SCALES],float);ys=np.array([out["scales"][str(s)]["one_minus_cos"] for s in SCALES],float)
            A=np.vstack([xs,np.ones_like(xs)]).T;coef=np.linalg.lstsq(A,ys,rcond=None)[0];pred=A@coef
            ssr=float(np.sum((ys-pred)**2));sst=float(np.sum((ys-ys.mean())**2))
            out["sigma2_trend"]={"slope":float(coef[0]),"intercept_at_zero_sigma2":float(coef[1]),
                                 "r2":float(1-ssr/(sst+1e-12))}
            fn=OUT/f"audit_u25_s{a.suite}.json";fn.write_text(json.dumps(out,indent=2)+"\n");print("WROTE",fn,flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "post_v2_t5_c40_derivative_equivalence": run_post_v2_t5_c40_derivative_equivalence,
    "post_v2_t5_c41_corrected_gradient_compare": run_post_v2_t5_c41_corrected_gradient_compare,
    "post_v2_t5_c42_highsample_confidence": run_post_v2_t5_c42_highsample_confidence,
    "post_v2_t5_c43_corrected_variance_limit": run_post_v2_t5_c43_corrected_variance_limit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
