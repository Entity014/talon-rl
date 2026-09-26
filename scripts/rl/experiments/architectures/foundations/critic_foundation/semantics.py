"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_rv1_b_preference_sensitivity_screen():
    """Run former rv1_b_preference_sensitivity_screen.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    OUT_DEFAULT=ROOT/"runs/rv1_b_preference_sensitivity-2026-09-23"
    ORDER=("T","A","O","S","C")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
     "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;NENV=8;SUPPORT=12;POOL_MAX=24
    SNAPS=(0,10,25,50,75)
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
        return sol[:-1].T,sol[-1]
    
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        d0=np.mean(X*X,axis=1);sel=[int(np.argmax(d0))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        arr=np.stack([PREFS[x] for x in labs])
        return labs,torch.tensor(arr,device=device)
    
    def collect(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=obs_tensor(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
        fo=torch.cat(ob);fw=w.repeat(H,1)
        with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
        C=np.concatenate(cmd,0)
        summary=np.r_[C.mean(0),C.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy()]
        out={"obs":fo,"w":fw,"Y":Y,"F":F,"rt":rt,"dt":dt,"next_obs":cur,
             "summary":summary.astype(np.float32),"termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def fit_head(m,pool,summaries):
        idx=select_diverse(summaries,SUPPORT)
        F=np.concatenate([pool[i][0] for i in idx]);Y=np.concatenate([pool[i][1] for i in idx])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return idx
    
    def pairwise_dist(actions):
        labs=list(actions);vals=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:
                vals.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals))}
    def preference_sensitivity(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        pair=pairwise_dist(acts)
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w)
        rows=[]
        for j in range(a.shape[1]):
            g=torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0]
            rows.append(g)
        jac=torch.stack(rows,dim=1)
        first=m.actor_body[0].weight[:,m.physical_obs_dim:]
        return {"pairwise_action_distance":pair,
                "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
                "preference_weight_norm":float(first.norm().detach().cpu())}
    
    def fresh_critic(env,m,mgr,seed):
        hs=[];bias=[];surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            q=collect(env,m,w,mgr,seed+qi*101,False)
            with torch.no_grad():V=m.value_with_preference(q["obs"],q["w"])
            Y=q["Y"]
            hs.extend(ev(Y[:,j].cpu(),V[:,j].cpu()) for j in range(4))
            bias.extend(float((V[:,j]-Y[:,j]).mean().cpu()) for j in range(4))
            surv.append(1-q["termination_fraction"])
        return {"h32_ev_mean":float(np.mean(hs)),"h32_negative_fraction":float(np.mean(np.array(hs)<0)),
                "mean_abs_bias":float(np.mean(np.abs(bias))),"min_survival":float(np.min(surv))}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75)
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);ap.add_argument("--seed",type=int,default=73001)
        args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=obs_tensor(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(INIT,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
            pool=[];summaries=[]
            # Warm the critic with independent reset-diverse support before actor update 1.
            for k in range(SUPPORT):
                _,wk=pref_batch(k,torch.device("cuda"))
                q=collect(env,m,wk,mgr,args.seed+10000+k*137,False)
                pool.append((q["F"].cpu().numpy(),q["Y"].cpu().numpy()));summaries.append(q["summary"])
            fit_head(m,pool,summaries)
            rows=[];snaps={}
            def audit(tag):
                m.eval();sens=preference_sensitivity(m,probe)
                fresh=fresh_critic(env,m,mgr,args.seed+500000+int(tag)*1000)
                snaps[str(tag)]={"sensitivity":sens,"fresh_critic":fresh}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt")
                m.train()
            audit(0)
            for uidx in range(1,args.updates+1):
                labs,w=pref_batch(uidx,torch.device("cuda"))
                mainb=collect(env,m,w,mgr,args.seed+uidx*211,True)
                # Add one independent reset support candidate and keep a bounded pool.
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                q=collect(env,m,ws,mgr,args.seed+200000+uidx*223,False)
                pool.append((q["F"].cpu().numpy(),q["Y"].cpu().numpy()));summaries.append(q["summary"])
                if len(pool)>POOL_MAX:pool.pop(0);summaries.pop(0)
                selected=fit_head(m,pool,summaries)
                with torch.no_grad():
                    vt=m.value_with_preference(mainb["obs"],mainb["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(mainb["next_obs"],w)
                    adv,_=vector_gae(mainb["rt"],vt,nv,mainb["dt"],lam=.95)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(mainb["obs"],mainb["w"],mainb["u"])-mainb["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"PPO ratio invariant failed: {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),mainb["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pref_grad=m.actor_body[0].weight.grad[:,m.physical_obs_dim:]
                pref_grad_norm=float(pref_grad.norm().detach().cpu())
                total_grad=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu())
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                rows.append({"update":uidx,"preferences":labs,"loss":float(loss.detach().cpu()),
                             "ratio_maxerr":ratio_err,"preference_input_grad_norm":pref_grad_norm,
                             "actor_grad_norm_preclip":total_grad,"termination_fraction":mainb["termination_fraction"],
                             "selected_support_indices":selected})
                if uidx in SNAPS:audit(uidx)
            report={"schema":"rv1_b_preference_sensitivity_screen_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"RV1-B one-seed short sensitivity screen","objective_semantics_gate":False,
                    "rows":rows,"snapshots":snaps}
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            fr=snaps[str(args.updates)]["fresh_critic"]
            grad=np.asarray([r["preference_input_grad_norm"] for r in rows])
            term=np.asarray([r["termination_fraction"] for r in rows])
            ratio=np.asarray([r["ratio_maxerr"] for r in rows])
            # Screen criteria intentionally test use of w + foundation preservation only.
            criteria={
                "preference_weights_moved":sf["preference_weight_norm"]>1e-5,
                "fixed_state_action_separation":sf["pairwise_action_distance"]["mean"]>1e-3,
                "preference_jacobian_nonzero":sf["jacobian_fro_mean"]>1e-3,
                "preference_gradient_observed":float(np.median(grad))>1e-6,
                "ppo_ratio_invariant":float(np.max(ratio))<=1e-4,
                "locomotion_not_collapsed":float(np.mean(term[-10:]))<0.5,
                "fresh_critic_not_collapsed":fr["h32_ev_mean"]>-0.5 and fr["h32_negative_fraction"]<0.75,
            }
            passed=all(criteria.values())
            report["summary"]={"status":"RV1-B PASS" if passed else "RV1-B FAIL","criteria":criteria,
                "initial_sensitivity":s0,"final_sensitivity":sf,"final_fresh_critic":fr,
                "median_preference_grad_norm":float(np.median(grad)),
                "last10_termination_fraction":float(np.mean(term[-10:])),
                "max_ratio_error":float(np.max(ratio)),
                "rv1_c_authorized":bool(passed)}
            out=args.output_dir/"rv1_b_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_c_semantic_response_audit():
    """Run former rv1_c_semantic_response_audit.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_b_preference_sensitivity-2026-09-23/model_75.pt"
    OUT_DEFAULT=ROOT/"runs/rv1_c_semantic_response-2026-09-23"
    NENV=8;STEPS=64;SUITES=4;G=.99
    ORDER=("T","A","O","S")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
     "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    ALPHAS=(0.0,.25,.5,.75,1.0)
    PATHS=(("T","A"),("T","O"),("T","S"))
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def segret(R,D,st,en):
        out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
        for t in range(en-1,st-1,-1):
            run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
        return out
    
    def evaluate(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        done_any=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(STEPS):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done_any|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"vx_error":vx,"wz_error":wz,"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
                             "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
                prev=a;cur=obs_tensor(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V)
        H0=segret(R,D,0,32);H1=segret(R,D,32,64)
        pmean={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
        obj=R.mean((0,1))
        return {"normalized_objective_mean":obj.tolist(),"physical":pmean,
                "survival":float(1-done_any.mean()),
                "critic":{"first_h32_ev":[ev(H0[:,:,j],V[:32,:,j]) for j in range(4)],
                          "second_h32_ev":[ev(H1[:,:,j],V[32:,:,j]) for j in range(4)],
                          "first_h32_bias":[float(np.mean(V[:32,:,j]-H0[:,:,j])) for j in range(4)],
                          "second_h32_bias":[float(np.mean(V[32:,:,j]-H1[:,:,j])) for j in range(4)]}}
    
    def interp(a,b,alpha):return (1-alpha)*PREFS[a]+alpha*PREFS[b]
    
    def monotonic_fraction(vals):
        vals=np.asarray(vals,float);d=np.diff(vals);target=np.sign(vals[-1]-vals[0])
        if target==0:return 0.0
        return float(np.mean(np.sign(d)==target))
    
    def between_fraction(vals):
        vals=np.asarray(vals,float);lo=min(vals[0],vals[-1]);hi=max(vals[0],vals[-1])
        return float(np.mean((vals[1:-1]>=lo-1e-9)&(vals[1:-1]<=hi+1e-9)))
    def sha(p):
        h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT)
        args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            # Matched endpoints + center: exact same reset seed across every preference in a suite.
            for suite in range(SUITES):
                seed=840001+suite
                for lab in ("T","A","O","S","C"):
                    q=evaluate(env,m,mgr,robot,PREFS[lab],seed)
                    q.update({"suite":suite,"kind":"endpoint","label":lab,"w":PREFS[lab].tolist()});rows.append(q)
            # Continuum paths, again matched by suite and path.
            continuum=[]
            for pi,(a,b) in enumerate(PATHS):
                for suite in range(SUITES):
                    seed=850001+pi*1000+suite
                    vals=[]
                    for alpha in ALPHAS:
                        w=interp(a,b,alpha)
                        q=evaluate(env,m,mgr,robot,w,seed)
                        q.update({"path":f"{a}-{b}","a":a,"b":b,"suite":suite,"alpha":alpha,"w":w.tolist()})
                        vals.append(q);continuum.append(q)
            # Endpoint directional correctness vs matched center.
            endpoint={}
            for lab in ORDER:
                j=IDX[lab];pk=PHYS[lab];obj_ok=[];phys_ok=[];surv=[]
                del_obj=[];del_phys=[]
                for suite in range(SUITES):
                    r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                    c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                    do=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                    dp=r["physical"][pk]-c["physical"][pk]
                    # Objectives are higher-is-better; physical proxies are lower-is-better.
                    obj_ok.append(do>0);phys_ok.append(dp<0);surv.append(r["survival"])
                    del_obj.append(do);del_phys.append(dp)
                endpoint[lab]={"objective_correct_fraction":float(np.mean(obj_ok)),
                               "physical_correct_fraction":float(np.mean(phys_ok)),
                               "mean_objective_delta_vs_center":float(np.mean(del_obj)),
                               "mean_physical_delta_vs_center":float(np.mean(del_phys)),
                               "min_survival":float(np.min(surv))}
            cont_stats=[]
            for a,b in PATHS:
                for suite in range(SUITES):
                    rr=sorted([x for x in continuum if x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                    for lab in (a,b):
                        j=IDX[lab];pk=PHYS[lab]
                        ov=[x["normalized_objective_mean"][j] for x in rr]
                        pv=[x["physical"][pk] for x in rr]
                        cont_stats.append({"path":f"{a}-{b}","suite":suite,"axis":lab,
                            "objective_monotonic":monotonic_fraction(ov),
                            "objective_between":between_fraction(ov),
                            # lower physical is better, but monotonic_fraction only tests coherence between endpoints.
                            "physical_monotonic":monotonic_fraction(pv),
                            "physical_between":between_fraction(pv)})
            mono=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont_stats]))
            between=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont_stats]))
            # Cross-axis trade-off/collateral summary.
            centers=[x for x in rows if x["label"]=="C"]
            center_track=float(np.mean([x["physical"]["tracking_error"] for x in centers]))
            max_track_ratio=0.0
            for lab in ORDER:
                rr=[x for x in rows if x["label"]==lab]
                tr=float(np.mean([x["physical"]["tracking_error"] for x in rr]))
                max_track_ratio=max(max_track_ratio,tr/(center_track+1e-12))
            all_surv=[x["survival"] for x in rows]+[x["survival"] for x in continuum]
            # Fresh critic aggregate over all endpoint/center matched evaluations.
            cev=[];cbias=[]
            for x in rows:
                cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"]
                cbias+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
            critic={"h32_ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),
                    "mean_abs_bias":float(np.mean(np.abs(cbias)))}
            endpoint_pass={lab:(endpoint[lab]["objective_correct_fraction"]>=.75 and
                                endpoint[lab]["physical_correct_fraction"]>=.75 and
                                endpoint[lab]["min_survival"]>=.95) for lab in ORDER}
            criteria={
                "all_endpoint_axes_correct":bool(all(endpoint_pass.values())),
                "continuum_monotonicity":mono>=.65,
                "continuum_endpoint_between":between>=.65,
                "survival_preserved":float(np.min(all_surv))>=.95,
                "tracking_collateral_bounded":max_track_ratio<=2.0,
                "fresh_critic_not_collapsed":critic["h32_ev_mean"]>0.0 and critic["negative_fraction"]<=.25,
            }
            passed=all(criteria.values())
            report={"schema":"rv1_c_semantic_response_audit_v1","status":"RV1-C PASS" if passed else "RV1-C FAIL",
                "measurement_only":True,"training_updates":0,"checkpoint":str(CKPT.relative_to(ROOT)),
                "protocol":{"matched_reset":True,"suites":SUITES,"steps":STEPS,"alphas":list(ALPHAS),
                            "paths":[f"{a}-{b}" for a,b in PATHS],"thresholds":{"endpoint_fraction":.75,"continuum_fraction":.65,
                            "min_survival":.95,"max_tracking_ratio_to_center":2.0}},
                "endpoint":endpoint,"endpoint_pass":endpoint_pass,"continuum_summary":{"monotonicity_fraction":mono,
                            "endpoint_between_fraction":between,"details":cont_stats},
                "collateral":{"center_tracking_error":center_track,"max_tracking_ratio_to_center":max_track_ratio},
                "critic":critic,"min_survival_all":float(np.min(all_surv)),"criteria":criteria,
                "decision":{"rv1_d_authorized":bool(passed),"v2_authorized":bool(not passed)},
                "endpoint_rows":rows,"continuum_rows":continuum}
            out=args.output_dir/"rv1_c_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:report[k] for k in ("status","endpoint","endpoint_pass","continuum_summary","collateral","critic","min_survival_all","criteria","decision") if k in report and k!="continuum_summary"} | {"continuum_summary":{"monotonicity_fraction":mono,"endpoint_between_fraction":between}},indent=2))
            prov={"status":"FROZEN_BY_HASH","gate":"RV1-C","verdict":report["status"],"measurement_only":True,
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},str(CKPT.relative_to(ROOT)):{"sha256":sha(CKPT)},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())},
                               "runs/rv1_b_preference_sensitivity-2026-09-23/PROVENANCE_MANIFEST.json":{"sha256":sha(ROOT/"runs/rv1_b_preference_sensitivity-2026-09-23/PROVENANCE_MANIFEST.json")}}}
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_c_semantic_response_phase_balanced():
    """Run former rv1_c_semantic_response_phase_balanced.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
    OUT_DEFAULT=ROOT/"runs/rv1_c_semantic_response_phase_balanced-2026-09-23"
    NENV=8;STEPS=64;SUITES=4;G=.99
    ORDER=("T","A","O","S")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
     "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    ALPHAS=(0.0,.25,.5,.75,1.0)
    PATHS=(("T","A"),("T","O"),("T","S"))
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def segret(R,D,st,en):
        out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
        for t in range(en-1,st-1,-1):
            run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
        return out
    
    def evaluate(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        done_any=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(STEPS):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done_any|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"vx_error":vx,"wz_error":wz,"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
                             "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
                prev=a;cur=obs_tensor(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V)
        H0=segret(R,D,0,32);H1=segret(R,D,32,64)
        pmean={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
        obj=R.mean((0,1))
        return {"normalized_objective_mean":obj.tolist(),"physical":pmean,
                "survival":float(1-done_any.mean()),
                "critic":{"first_h32_ev":[ev(H0[:,:,j],V[:32,:,j]) for j in range(4)],
                          "second_h32_ev":[ev(H1[:,:,j],V[32:,:,j]) for j in range(4)],
                          "first_h32_bias":[float(np.mean(V[:32,:,j]-H0[:,:,j])) for j in range(4)],
                          "second_h32_bias":[float(np.mean(V[32:,:,j]-H1[:,:,j])) for j in range(4)]}}
    
    def interp(a,b,alpha):return (1-alpha)*PREFS[a]+alpha*PREFS[b]
    
    def monotonic_fraction(vals):
        vals=np.asarray(vals,float);d=np.diff(vals);target=np.sign(vals[-1]-vals[0])
        if target==0:return 0.0
        return float(np.mean(np.sign(d)==target))
    
    def between_fraction(vals):
        vals=np.asarray(vals,float);lo=min(vals[0],vals[-1]);hi=max(vals[0],vals[-1])
        return float(np.mean((vals[1:-1]>=lo-1e-9)&(vals[1:-1]<=hi+1e-9)))
    def sha(p):
        h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT)
        args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            # Matched endpoints + center: exact same reset seed across every preference in a suite.
            for suite in range(SUITES):
                seed=840001+suite
                for lab in ("T","A","O","S","C"):
                    q=evaluate(env,m,mgr,robot,PREFS[lab],seed)
                    q.update({"suite":suite,"kind":"endpoint","label":lab,"w":PREFS[lab].tolist()});rows.append(q)
            # Continuum paths, again matched by suite and path.
            continuum=[]
            for pi,(a,b) in enumerate(PATHS):
                for suite in range(SUITES):
                    seed=850001+pi*1000+suite
                    vals=[]
                    for alpha in ALPHAS:
                        w=interp(a,b,alpha)
                        q=evaluate(env,m,mgr,robot,w,seed)
                        q.update({"path":f"{a}-{b}","a":a,"b":b,"suite":suite,"alpha":alpha,"w":w.tolist()})
                        vals.append(q);continuum.append(q)
            # Endpoint directional correctness vs matched center.
            endpoint={}
            for lab in ORDER:
                j=IDX[lab];pk=PHYS[lab];obj_ok=[];phys_ok=[];surv=[]
                del_obj=[];del_phys=[]
                for suite in range(SUITES):
                    r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                    c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                    do=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                    dp=r["physical"][pk]-c["physical"][pk]
                    # Objectives are higher-is-better; physical proxies are lower-is-better.
                    obj_ok.append(do>0);phys_ok.append(dp<0);surv.append(r["survival"])
                    del_obj.append(do);del_phys.append(dp)
                endpoint[lab]={"objective_correct_fraction":float(np.mean(obj_ok)),
                               "physical_correct_fraction":float(np.mean(phys_ok)),
                               "mean_objective_delta_vs_center":float(np.mean(del_obj)),
                               "mean_physical_delta_vs_center":float(np.mean(del_phys)),
                               "min_survival":float(np.min(surv))}
            cont_stats=[]
            for a,b in PATHS:
                for suite in range(SUITES):
                    rr=sorted([x for x in continuum if x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                    for lab in (a,b):
                        j=IDX[lab];pk=PHYS[lab]
                        ov=[x["normalized_objective_mean"][j] for x in rr]
                        pv=[x["physical"][pk] for x in rr]
                        cont_stats.append({"path":f"{a}-{b}","suite":suite,"axis":lab,
                            "objective_monotonic":monotonic_fraction(ov),
                            "objective_between":between_fraction(ov),
                            # lower physical is better, but monotonic_fraction only tests coherence between endpoints.
                            "physical_monotonic":monotonic_fraction(pv),
                            "physical_between":between_fraction(pv)})
            mono=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont_stats]))
            between=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont_stats]))
            # Cross-axis trade-off/collateral summary.
            centers=[x for x in rows if x["label"]=="C"]
            center_track=float(np.mean([x["physical"]["tracking_error"] for x in centers]))
            max_track_ratio=0.0
            for lab in ORDER:
                rr=[x for x in rows if x["label"]==lab]
                tr=float(np.mean([x["physical"]["tracking_error"] for x in rr]))
                max_track_ratio=max(max_track_ratio,tr/(center_track+1e-12))
            all_surv=[x["survival"] for x in rows]+[x["survival"] for x in continuum]
            # Fresh critic aggregate over all endpoint/center matched evaluations.
            cev=[];cbias=[]
            for x in rows:
                cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"]
                cbias+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
            critic={"h32_ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),
                    "mean_abs_bias":float(np.mean(np.abs(cbias)))}
            endpoint_pass={lab:(endpoint[lab]["objective_correct_fraction"]>=.75 and
                                endpoint[lab]["physical_correct_fraction"]>=.75 and
                                endpoint[lab]["min_survival"]>=.95) for lab in ORDER}
            criteria={
                "all_endpoint_axes_correct":bool(all(endpoint_pass.values())),
                "continuum_monotonicity":mono>=.65,
                "continuum_endpoint_between":between>=.65,
                "survival_preserved":float(np.min(all_surv))>=.95,
                "tracking_collateral_bounded":max_track_ratio<=2.0,
                "fresh_critic_not_collapsed":critic["h32_ev_mean"]>0.0 and critic["negative_fraction"]<=.25,
            }
            passed=all(criteria.values())
            report={"schema":"rv1_c_semantic_response_audit_phase_balanced_v1","status":"RV1-C PASS" if passed else "RV1-C FAIL",
                "measurement_only":True,"training_updates":0,"checkpoint":str(CKPT.relative_to(ROOT)),
                "protocol":{"matched_reset":True,"suites":SUITES,"steps":STEPS,"alphas":list(ALPHAS),
                            "paths":[f"{a}-{b}" for a,b in PATHS],"thresholds":{"endpoint_fraction":.75,"continuum_fraction":.65,
                            "min_survival":.95,"max_tracking_ratio_to_center":2.0}},
                "endpoint":endpoint,"endpoint_pass":endpoint_pass,"continuum_summary":{"monotonicity_fraction":mono,
                            "endpoint_between_fraction":between,"details":cont_stats},
                "collateral":{"center_tracking_error":center_track,"max_tracking_ratio_to_center":max_track_ratio},
                "critic":critic,"min_survival_all":float(np.min(all_surv)),"criteria":criteria,
                "decision":{"rv1_d_authorized":bool(passed),"v2_authorized":bool(not passed)},
                "endpoint_rows":rows,"continuum_rows":continuum}
            out=args.output_dir/"rv1_c_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:report[k] for k in ("status","endpoint","endpoint_pass","continuum_summary","collateral","critic","min_survival_all","criteria","decision") if k in report and k!="continuum_summary"} | {"continuum_summary":{"monotonicity_fraction":mono,"endpoint_between_fraction":between}},indent=2))
            prov={"status":"FROZEN_BY_HASH","gate":"RV1-C","verdict":report["status"],"measurement_only":True,
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},str(CKPT.relative_to(ROOT)):{"sha256":sha(CKPT)},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())},
                               "runs/rv1_b_preference_sensitivity-2026-09-23/PROVENANCE_MANIFEST.json":{"sha256":sha(ROOT/"runs/rv1_b_preference_sensitivity-2026-09-23/PROVENANCE_MANIFEST.json")}}}
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_c_semantic_response_repaired_persistence():
    """Run former rv1_c_semantic_response_repaired_persistence.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_persistence_repair_v2-2026-09-23/model_75.pt"
    OUT_DEFAULT=ROOT/"runs/rv1_c_semantic_response_repaired_persistence-2026-09-23"
    NENV=8;STEPS=64;SUITES=4;G=.99
    ORDER=("T","A","O","S")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
     "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    ALPHAS=(0.0,.25,.5,.75,1.0)
    PATHS=(("T","A"),("T","O"),("T","S"))
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def segret(R,D,st,en):
        out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
        for t in range(en-1,st-1,-1):
            run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
        return out
    
    def evaluate(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        done_any=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(STEPS):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done_any|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"vx_error":vx,"wz_error":wz,"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
                             "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
                prev=a;cur=obs_tensor(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V)
        H0=segret(R,D,0,32);H1=segret(R,D,32,64)
        pmean={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
        obj=R.mean((0,1))
        return {"normalized_objective_mean":obj.tolist(),"physical":pmean,
                "survival":float(1-done_any.mean()),
                "critic":{"first_h32_ev":[ev(H0[:,:,j],V[:32,:,j]) for j in range(4)],
                          "second_h32_ev":[ev(H1[:,:,j],V[32:,:,j]) for j in range(4)],
                          "first_h32_bias":[float(np.mean(V[:32,:,j]-H0[:,:,j])) for j in range(4)],
                          "second_h32_bias":[float(np.mean(V[32:,:,j]-H1[:,:,j])) for j in range(4)]}}
    
    def interp(a,b,alpha):return (1-alpha)*PREFS[a]+alpha*PREFS[b]
    
    def monotonic_fraction(vals):
        vals=np.asarray(vals,float);d=np.diff(vals);target=np.sign(vals[-1]-vals[0])
        if target==0:return 0.0
        return float(np.mean(np.sign(d)==target))
    
    def between_fraction(vals):
        vals=np.asarray(vals,float);lo=min(vals[0],vals[-1]);hi=max(vals[0],vals[-1])
        return float(np.mean((vals[1:-1]>=lo-1e-9)&(vals[1:-1]<=hi+1e-9)))
    def sha(p):
        h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT)
        args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            # Matched endpoints + center: exact same reset seed across every preference in a suite.
            for suite in range(SUITES):
                seed=840001+suite
                for lab in ("T","A","O","S","C"):
                    q=evaluate(env,m,mgr,robot,PREFS[lab],seed)
                    q.update({"suite":suite,"kind":"endpoint","label":lab,"w":PREFS[lab].tolist()});rows.append(q)
            # Continuum paths, again matched by suite and path.
            continuum=[]
            for pi,(a,b) in enumerate(PATHS):
                for suite in range(SUITES):
                    seed=850001+pi*1000+suite
                    vals=[]
                    for alpha in ALPHAS:
                        w=interp(a,b,alpha)
                        q=evaluate(env,m,mgr,robot,w,seed)
                        q.update({"path":f"{a}-{b}","a":a,"b":b,"suite":suite,"alpha":alpha,"w":w.tolist()})
                        vals.append(q);continuum.append(q)
            # Endpoint directional correctness vs matched center.
            endpoint={}
            for lab in ORDER:
                j=IDX[lab];pk=PHYS[lab];obj_ok=[];phys_ok=[];surv=[]
                del_obj=[];del_phys=[]
                for suite in range(SUITES):
                    r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                    c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                    do=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                    dp=r["physical"][pk]-c["physical"][pk]
                    # Objectives are higher-is-better; physical proxies are lower-is-better.
                    obj_ok.append(do>0);phys_ok.append(dp<0);surv.append(r["survival"])
                    del_obj.append(do);del_phys.append(dp)
                endpoint[lab]={"objective_correct_fraction":float(np.mean(obj_ok)),
                               "physical_correct_fraction":float(np.mean(phys_ok)),
                               "mean_objective_delta_vs_center":float(np.mean(del_obj)),
                               "mean_physical_delta_vs_center":float(np.mean(del_phys)),
                               "min_survival":float(np.min(surv))}
            cont_stats=[]
            for a,b in PATHS:
                for suite in range(SUITES):
                    rr=sorted([x for x in continuum if x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                    for lab in (a,b):
                        j=IDX[lab];pk=PHYS[lab]
                        ov=[x["normalized_objective_mean"][j] for x in rr]
                        pv=[x["physical"][pk] for x in rr]
                        cont_stats.append({"path":f"{a}-{b}","suite":suite,"axis":lab,
                            "objective_monotonic":monotonic_fraction(ov),
                            "objective_between":between_fraction(ov),
                            # lower physical is better, but monotonic_fraction only tests coherence between endpoints.
                            "physical_monotonic":monotonic_fraction(pv),
                            "physical_between":between_fraction(pv)})
            mono=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont_stats]))
            between=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont_stats]))
            # Cross-axis trade-off/collateral summary.
            centers=[x for x in rows if x["label"]=="C"]
            center_track=float(np.mean([x["physical"]["tracking_error"] for x in centers]))
            max_track_ratio=0.0
            for lab in ORDER:
                rr=[x for x in rows if x["label"]==lab]
                tr=float(np.mean([x["physical"]["tracking_error"] for x in rr]))
                max_track_ratio=max(max_track_ratio,tr/(center_track+1e-12))
            all_surv=[x["survival"] for x in rows]+[x["survival"] for x in continuum]
            # Fresh critic aggregate over all endpoint/center matched evaluations.
            cev=[];cbias=[]
            for x in rows:
                cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"]
                cbias+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
            critic={"h32_ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),
                    "mean_abs_bias":float(np.mean(np.abs(cbias)))}
            endpoint_pass={lab:(endpoint[lab]["objective_correct_fraction"]>=.75 and
                                endpoint[lab]["physical_correct_fraction"]>=.75 and
                                endpoint[lab]["min_survival"]>=.95) for lab in ORDER}
            criteria={
                "all_endpoint_axes_correct":bool(all(endpoint_pass.values())),
                "continuum_monotonicity":mono>=.65,
                "continuum_endpoint_between":between>=.65,
                "survival_preserved":float(np.min(all_surv))>=.95,
                "tracking_collateral_bounded":max_track_ratio<=2.0,
                "fresh_critic_not_collapsed":critic["h32_ev_mean"]>0.0 and critic["negative_fraction"]<=.25,
            }
            passed=all(criteria.values())
            report={"schema":"rv1_c_semantic_response_audit_repaired_persistence_v1","status":"RV1-C PASS" if passed else "RV1-C FAIL",
                "measurement_only":True,"training_updates":0,"checkpoint":str(CKPT.relative_to(ROOT)),
                "protocol":{"matched_reset":True,"suites":SUITES,"steps":STEPS,"alphas":list(ALPHAS),
                            "paths":[f"{a}-{b}" for a,b in PATHS],"thresholds":{"endpoint_fraction":.75,"continuum_fraction":.65,
                            "min_survival":.95,"max_tracking_ratio_to_center":2.0}},
                "endpoint":endpoint,"endpoint_pass":endpoint_pass,"continuum_summary":{"monotonicity_fraction":mono,
                            "endpoint_between_fraction":between,"details":cont_stats},
                "collateral":{"center_tracking_error":center_track,"max_tracking_ratio_to_center":max_track_ratio},
                "critic":critic,"min_survival_all":float(np.min(all_surv)),"criteria":criteria,
                "decision":{"rv1_d_authorized":bool(passed),"v2_authorized":bool(not passed)},
                "endpoint_rows":rows,"continuum_rows":continuum}
            out=args.output_dir/"rv1_c_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:report[k] for k in ("status","endpoint","endpoint_pass","continuum_summary","collateral","critic","min_survival_all","criteria","decision") if k in report and k!="continuum_summary"} | {"continuum_summary":{"monotonicity_fraction":mono,"endpoint_between_fraction":between}},indent=2))
            prov={"status":"FROZEN_BY_HASH","gate":"RV1-C","verdict":report["status"],"measurement_only":True,
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},str(CKPT.relative_to(ROOT)):{"sha256":sha(CKPT)},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())},
                               "runs/rv1_b_preference_sensitivity-2026-09-23/PROVENANCE_MANIFEST.json":{"sha256":sha(ROOT/"runs/rv1_b_preference_sensitivity-2026-09-23/PROVENANCE_MANIFEST.json")}}}
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_local_target_mapping_same_preference():
    """Run former rv1_local_target_mapping_same_preference.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
    REP=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/phase_repair_report.json"
    OUT=ROOT/"runs/rv1_local_target_mapping_same_preference-2026-09-23"
    NENV=8;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    ORDER=("T","A","O","S","C");ALPHAS=(0,.25,.5,.75,1.0);PATHS=(("T","A"),("T","O"),("T","S"))
    HEADS=("Tracking","Angular","Orientation","Smoothness")
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    def interp(a,b,t):return (1-t)*PREFS[a]+t*PREFS[b]
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def collect64(env,m,w_np,seed,label):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager
        w=torch.tensor(w_np,device="cuda") if np.asarray(w_np).ndim==2 else torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];R=[];D=[];cmd=[]
        with torch.no_grad():
            for _ in range(64):
                obs.append(cur);cmd.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector({n:raw[:,i] for i,n in enumerate(names)},shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);D.append((te|tr).cuda());cur=ot(nxt).cuda()
        O=torch.stack(obs)[32:64].reshape(-1,obs[0].shape[-1]);W=w.repeat(32,1)
        Y=trunc(torch.stack(R)[32:64],torch.stack(D).bool()[32:64]).reshape(-1,4)
        C=np.asarray(cmd)[32:64].reshape(-1,3)
        with torch.no_grad():
            F=m.critic_body(m._with_w(O,W));V=m.critic_head(F)
        return {"feature":F.cpu().numpy(),"value":V.cpu().numpy(),"target":Y.cpu().numpy(),
                "command":C,"w":W.cpu().numpy(),"label":np.array([label]*len(F),object)}
    def corr(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        if np.std(a)<1e-12 or np.std(b)<1e-12:return 0.0
        return float(np.corrcoef(a,b)[0,1])
    def regress(y,*xs):
        X=np.c_[np.ones(len(y)),*[np.asarray(x,float) for x in xs]];y=np.asarray(y,float)
        beta=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@beta
        ssr=np.sum((y-pred)**2);sst=np.sum((y-y.mean())**2)
        return {"coef":beta.tolist(),"r2":float(1-ssr/(sst+1e-12))}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);a=ap.parse_args()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            tr=json.load(open(REP));row75=tr["rows"][-1];anchor_specs=tr["anchor_specs"];late=row75["selected"]["late"]
            sup=[]
            for loc in late["anchor"]:
                k,seed=anchor_specs[loc];sup.append(collect64(env,m,pref_batch(k,torch.device("cuda")).cpu().numpy(),seed,f"a{k}"))
            for idx in late["adaptive"]:
                u=52+idx;seed=73001+200000+u*223;sup.append(collect64(env,m,pref_batch(u+17,torch.device("cuda")).cpu().numpy(),seed,f"r{u}"))
            ep=[];ct=[]
            for suite in range(4):
                seed=840001+suite
                for lab in ORDER:ep.append(collect64(env,m,PREFS[lab],seed,f"ep_{lab}_{suite}"))
            for pi,(x,y) in enumerate(PATHS):
                for suite in range(4):
                    seed=850001+pi*1000+suite
                    for al in ALPHAS:ct.append(collect64(env,m,interp(x,y,al),seed,f"{x}{y}_{al}_{suite}"))
            def cat(rows,key):return np.concatenate([r[key] for r in rows],axis=0)
            Sf,Sv,Sy,Sc,Sw=[cat(sup,k) for k in ("feature","value","target","command","w")]
            mu=Sf.mean(0);sd=Sf.std(0)+1e-8;SZ=(Sf-mu)/sd
            St=torch.tensor(SZ,dtype=torch.float32,device="cuda")
            result={"schema":"rv1_local_target_mapping_same_preference_v1","measurement_only":True,"endpoint_by_preference":{}}
            for lab in ORDER:
                rows=[r for r in ep if r["label"][0].startswith(f"ep_{lab}_")]
                Qf,Qv,Qy,Qc,Qw=[cat(rows,k) for k in ("feature","value","target","command","w")]
                # exact same-preference support subset
                mask=np.all(np.isclose(Sw,Qw[0],atol=1e-7),axis=1)
                Sf_l,Sy_l,Sc_l= Sf[mask],Sy[mask],Sc[mask]
                mu_l=Sf_l.mean(0);sd_l=Sf_l.std(0)+1e-8
                SZ_l=(Sf_l-mu_l)/sd_l;QZ=(Qf-mu_l)/sd_l
                St_l=torch.tensor(SZ_l,dtype=torch.float32,device="cuda")
                don=[];fd=[]
                for i in range(0,len(QZ),512):
                    Qt=torch.tensor(QZ[i:i+512],dtype=torch.float32,device="cuda")
                    D=torch.cdist(Qt,St_l)/np.sqrt(Sf_l.shape[1]);v,ix=D.min(1)
                    don.append(ix.cpu().numpy());fd.append(v.cpu().numpy())
                don=np.concatenate(don);fd=np.concatenate(fd)
                dy=np.abs(Qy-Sy_l[don]);res=np.abs(Qv-Qy);cmd=np.linalg.norm(Qc-Sc_l[don],axis=1)
                close=fd<=np.quantile(fd,.25);far=fd>=np.quantile(fd,.75)
                heads=[]
                for j,h in enumerate(HEADS):
                    heads.append({"head":h,
                        "corr_residual_feature_distance":corr(res[:,j],fd),
                        "corr_residual_command_mismatch":corr(res[:,j],cmd),
                        "corr_target_delta_feature_distance":corr(dy[:,j],fd),
                        "close_neighbor":{"feature_dist_mean":float(fd[close].mean()),"target_delta_mean":float(dy[close,j].mean()),
                                          "residual_mean":float(res[close,j].mean()),"command_mismatch_mean":float(cmd[close].mean())},
                        "far_neighbor":{"feature_dist_mean":float(fd[far].mean()),"target_delta_mean":float(dy[far,j].mean()),
                                        "residual_mean":float(res[far,j].mean()),"command_mismatch_mean":float(cmd[far].mean())},
                        "regression_residual_on_feature_command":regress(res[:,j],fd,cmd),
                        "regression_target_delta_on_feature":regress(dy[:,j],fd)})
                result["endpoint_by_preference"][lab]={"support_n":int(mask.sum()),"semantic_n":len(Qf),
                    "feature_distance":{"mean":float(fd.mean()),"p25":float(np.quantile(fd,.25)),"p50":float(np.quantile(fd,.5)),"p75":float(np.quantile(fd,.75))},
                    "command_mismatch_mean":float(cmd.mean()),"heads":heads}
            # aggregate concise summary
            summary={}
            for hidx,hname in enumerate(HEADS):
                close_res=[];far_res=[];close_dy=[];far_dy=[];corrs=[]
                for lab in ORDER:
                    h=result["endpoint_by_preference"][lab]["heads"][hidx]
                    close_res.append(h["close_neighbor"]["residual_mean"]);far_res.append(h["far_neighbor"]["residual_mean"])
                    close_dy.append(h["close_neighbor"]["target_delta_mean"]);far_dy.append(h["far_neighbor"]["target_delta_mean"])
                    corrs.append(h["corr_residual_feature_distance"])
                summary[hname]={"close_residual_mean":float(np.mean(close_res)),"far_residual_mean":float(np.mean(far_res)),
                                "close_target_delta_mean":float(np.mean(close_dy)),"far_target_delta_mean":float(np.mean(far_dy)),
                                "mean_corr_residual_feature_distance":float(np.mean(corrs))}
            result["summary"]=summary
            out=a.output_dir/"local_mapping_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":hashlib.sha256(out.read_bytes()).hexdigest()},indent=2)+"\n")
            print(json.dumps(result["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "rv1_b_preference_sensitivity_screen": run_rv1_b_preference_sensitivity_screen,
    "rv1_c_semantic_response_audit": run_rv1_c_semantic_response_audit,
    "rv1_c_semantic_response_phase_balanced": run_rv1_c_semantic_response_phase_balanced,
    "rv1_c_semantic_response_repaired_persistence": run_rv1_c_semantic_response_repaired_persistence,
    "rv1_local_target_mapping_same_preference": run_rv1_local_target_mapping_same_preference,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
