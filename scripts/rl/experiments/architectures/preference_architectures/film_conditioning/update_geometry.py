"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_v2b_adam_history_isolation():
    """Run former v2b_adam_history_isolation.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    V2B_INIT=ROOT/"runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2b_adam_history_isolation"
    ORDER=("T","A","O","S","C")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;SUP_H=64;NENV=8;POOL_MAX=24
    ANCHOR_CANDIDATES=12;ANCHOR_PHASE_K=3;ADAPT_PHASE_K=3
    SNAPS=(0,10,25,50,75)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        sel=[int(np.argmax(np.mean(X*X,axis=1)))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_actor(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        with torch.no_grad():
            for _ in range(H):
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool()
        out={"obs":torch.cat(ob),"w":w.repeat(H,1),"rt":rt,"dt":dt,"next_obs":cur,
             "termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def collect_support64(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();obs=torch.stack(ob) # [64,E,D]
        C=np.stack(cmd) # [64,E,3]
        phases=[]
        for pi,(st,en) in enumerate(((0,32),(32,64))):
            Y=trunc(rt[st:en],dt[st:en]).reshape(-1,4).detach()
            po=obs[st:en].reshape(-1,obs.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).detach()
            cc=C[st:en].reshape(-1,C.shape[-1])
            summary=np.r_[cc.mean(0),cc.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),float(pi)]
            phases.append({"F":F.cpu().numpy(),"Y":Y.cpu().numpy(),"summary":summary.astype(np.float32),
                           "phase":"early" if pi==0 else "late"})
        return phases
    
    def fit_expanded_current_policy(m,anchor_units,adaptive_pools):
        support=[]
        counts={}
        for phase in ("early","late"):
            au=anchor_units[phase]
            ad=adaptive_pools[phase]
            support += [(u["F"],u["Y"]) for u in au]
            support += [(u["F"],u["Y"]) for u in ad]
            counts[phase]={"anchor_units":len(au),"adaptive_units":len(ad)}
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return counts
    
    def pairwise_dist(actions):
        labs=list(actions);v=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:v.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def sensitivity(m,probe):
        total={};masked={};film_g={};film_b={};emb={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                total[lab]=m.act_inference_with_preference(probe,w)
                # FiLM-masked baseline = exact V2-A computation using current shared actor + embedding params.
                h=m._actor_hidden(probe,w)
                e=m.preference_embedding(w)
                masked[lab]=torch.tanh(m.actor_mean(torch.cat((h,e),dim=-1)))*m.ACTION_CLIP
                gb=m.preference_film(w);g,b=torch.chunk(gb,2,dim=-1)
                film_g[lab]=g;film_b[lab]=b;emb[lab]=e
    
        pair_total=pairwise_dist(total)
        pair_masked=pairwise_dist(masked)
        gamma_sep=pairwise_dist(film_g)
        beta_sep=pairwise_dist(film_b)
    
        authority=float(np.mean([
            torch.linalg.vector_norm(total[k]-masked[k],dim=1).mean().cpu() for k in ORDER
        ]))
    
        # Total d a / d w.
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w);rows=[]
        for j in range(a.shape[1]):
            rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
    
        # FiLM-masked d a / d w: RV1 direct + embedding only.
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        hm=m._actor_hidden(probe,wm);em=m.preference_embedding(wm)
        am=torch.tanh(m.actor_mean(torch.cat((hm,em),dim=-1)))*m.ACTION_CLIP
        mrows=[]
        for j in range(am.shape[1]):
            mrows.append(torch.autograd.grad(am[:,j].mean(),wm,retain_graph=True)[0])
        mjac=torch.stack(mrows,dim=1)
    
        wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1)
        with torch.no_grad():
            gbc=m.preference_film(wc);gc,bc=torch.chunk(gbc,2,dim=-1)
        return {
            "pairwise_action_distance":pair_total,
            "film_masked_pairwise_action_distance":pair_masked,
            "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_masked_jacobian_fro_mean":float(torch.linalg.matrix_norm(mjac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_action_authority_mean":authority,
            "film_weight_norm":float(m.preference_film.weight.norm().detach().cpu()),
            "film_bias_norm":float(m.preference_film.bias.norm().detach().cpu()),
            "gamma_abs_mean_center":float(gc.abs().mean().cpu()),
            "beta_abs_mean_center":float(bc.abs().mean().cpu()),
            "gamma_pairwise_distance":gamma_sep,
            "beta_pairwise_distance":beta_sep,
            "embedding_feature_pairwise_distance":pairwise_dist(emb),
            "preference_weight_norm":float(m.actor_body[0].weight[:,m.physical_obs_dim:].norm().detach().cpu())
        }
    
    def fresh_phase_audit(env,m,mgr,seed):
        phase_ev={"early":[],"late":[]};phase_bias={"early":[],"late":[]};surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            units=collect_support64(env,m,w,mgr,seed+qi*101)
            for unit in units:
                with torch.no_grad():
                    # Features already current-policy and head is linear.
                    F=torch.tensor(unit["F"],device="cuda",dtype=m.critic_head.weight.dtype)
                    V=m.critic_head(F).cpu().numpy()
                Y=unit["Y"];ph=unit["phase"]
                phase_ev[ph].extend(ev(Y[:,j],V[:,j]) for j in range(4))
                phase_bias[ph].extend(float(np.mean(V[:,j]-Y[:,j])) for j in range(4))
            surv.append(1.0) # deterministic support rollouts are diagnostic; detailed terminations in independent revalidation
        out={}
        for ph in ("early","late"):
            a=np.asarray(phase_ev[ph]);b=np.asarray(phase_bias[ph])
            out[ph]={"ev_mean":float(a.mean()),"negative_fraction":float((a<0).mean()),"mean_abs_bias":float(np.abs(b).mean())}
        out["combined_negative_fraction"]=float(np.mean(np.r_[np.asarray(phase_ev["early"])<0,np.asarray(phase_ev["late"])<0]))
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--gae-lambda",type=float,choices=[1.0],default=1.0)
        ap.add_argument("--adam-reset-update",type=int,default=0)
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);args=ap.parse_args()
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
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            init_path=V2B_INIT
            m.load_state_dict(torch.load(init_path,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            # Freeze anchor seed/preference specs once.
            cand=[]
            for k in range(ANCHOR_CANDIDATES):
                _,w=pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            # Select 6 seed specs on combined early+late summary geometry.
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
    
            adaptive_pools={"early":[],"late":[]};adaptive_summaries={"early":[],"late":[]}
            # initialize adaptive pools from all initialization candidates
            for _,_,units in cand:
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
    
            def current_anchor_units():
                out={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=pref_batch(k,torch.device("cuda"))
                    for u in collect_support64(env,m,w,mgr,seed):
                        out[u["phase"]].append(u)
                return out
            fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
    
            rows=[];snaps={}
            def audit(tag):
                m.eval();snaps[str(tag)]={"sensitivity":sensitivity(m,probe),
                    "phase_critic":fresh_phase_audit(env,m,mgr,args.seed+500000+int(tag)*1000)}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt");m.train()
            audit(0)
    
            for uidx in range(1,args.updates+1):
                adam_reset_applied=False
                if args.adam_reset_update>0 and uidx==args.adam_reset_update+1:
                    opt.state.clear()
                    adam_reset_applied=True
                labs,w=pref_batch(uidx,torch.device("cuda"))
                main=collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                # one recent adaptive 64-step support candidate contributes one early + one late unit
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                units=collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
                    if len(adaptive_pools[ph])>POOL_MAX:
                        adaptive_pools[ph].pop(0);adaptive_summaries[ph].pop(0)
                selected=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=args.gae_lambda)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pg=float(m.actor_body[0].weight.grad[:,m.physical_obs_dim:].norm().detach().cpu())
                fwgrad=float(m.preference_film.weight.grad.norm().detach().cpu()) if m.preference_film.weight.grad is not None else 0.0
                fbgrad=float(m.preference_film.bias.grad.norm().detach().cpu()) if m.preference_film.bias.grad is not None else 0.0
                # log optimizer memory and actual parameter delta around the real update
                grad_preclip=total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu())
                before=[x.detach().clone() for x in actor_params]
                m1_before=[];m2_before=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_before.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_before.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                delta_sq=0.0
                for bb,par in zip(before,actor_params):
                    delta_sq += float(((par.detach()-bb)**2).sum().cpu())
                actual_delta_norm=float(delta_sq**0.5)
                m1_after=[];m2_after=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_after.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_after.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                obs=main["obs"].detach()
                rt=main["rt"].detach()
                visitation={
                  "obs_mean_norm":float(obs.mean(0).norm().cpu()),
                  "obs_std_mean":float(obs.std(0).mean().cpu()),
                  "objective_mean":rt.mean((0,1)).cpu().tolist(),
                  "termination_fraction":main["termination_fraction"]}
    
                post_refresh=None
                # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
                if uidx in SNAPS:
                    post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                             "preference_input_grad_norm":pg,"film_weight_grad_norm":fwgrad,"film_bias_grad_norm":fbgrad,"actor_grad_norm_preclip":grad_preclip,
                             "actual_param_delta_norm":actual_delta_norm,
                             "adam_m1_norm_sum_before":float(sum(m1_before)),"adam_m2_norm_sum_before":float(sum(m2_before)),
                             "adam_m1_norm_sum_after":float(sum(m1_after)),"adam_m2_norm_sum_after":float(sum(m2_after)),
                             "adam_reset_applied":adam_reset_applied,"visitation":visitation,
                             "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
                if uidx in SNAPS:audit(uidx)
    
            report={"schema":"v2b_adam_history_isolation_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"optimizer-history vs visitation isolation under frozen V2-B + Foundation V2",
                    "actor_ppo_objectives_changed":False,"architecture":"v2b","gae_lambda":args.gae_lambda,"adam_reset_update":args.adam_reset_update,"anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            final=snaps[str(args.updates)]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            crit=final["phase_critic"];sens=final["sensitivity"]
            criteria={
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
            report["summary"]={
              "status":"LAMBDA-PILOT FOUNDATION PASS" if passed else "LAMBDA-PILOT FOUNDATION FAIL",
              "gae_lambda":args.gae_lambda,
              "criteria":criteria,
              "final":final,
              "last10_termination_fraction":float(term[-10:].mean()),
              "max_ratio_error":float(ratio.max())}
            (args.output_dir/"adam_history_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_adam_m1_eachstep_isolation():
    """Run former v2b_adam_m1_eachstep_isolation.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    V2B_INIT=ROOT/"runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2b_adam_history_isolation"
    ORDER=("T","A","O","S","C")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;SUP_H=64;NENV=8;POOL_MAX=24
    ANCHOR_CANDIDATES=12;ANCHOR_PHASE_K=3;ADAPT_PHASE_K=3
    SNAPS=(0,10,25,50,75)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        sel=[int(np.argmax(np.mean(X*X,axis=1)))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_actor(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        with torch.no_grad():
            for _ in range(H):
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool()
        out={"obs":torch.cat(ob),"w":w.repeat(H,1),"rt":rt,"dt":dt,"next_obs":cur,
             "termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def collect_support64(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();obs=torch.stack(ob) # [64,E,D]
        C=np.stack(cmd) # [64,E,3]
        phases=[]
        for pi,(st,en) in enumerate(((0,32),(32,64))):
            Y=trunc(rt[st:en],dt[st:en]).reshape(-1,4).detach()
            po=obs[st:en].reshape(-1,obs.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).detach()
            cc=C[st:en].reshape(-1,C.shape[-1])
            summary=np.r_[cc.mean(0),cc.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),float(pi)]
            phases.append({"F":F.cpu().numpy(),"Y":Y.cpu().numpy(),"summary":summary.astype(np.float32),
                           "phase":"early" if pi==0 else "late"})
        return phases
    
    def fit_expanded_current_policy(m,anchor_units,adaptive_pools):
        support=[]
        counts={}
        for phase in ("early","late"):
            au=anchor_units[phase]
            ad=adaptive_pools[phase]
            support += [(u["F"],u["Y"]) for u in au]
            support += [(u["F"],u["Y"]) for u in ad]
            counts[phase]={"anchor_units":len(au),"adaptive_units":len(ad)}
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return counts
    
    def pairwise_dist(actions):
        labs=list(actions);v=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:v.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def sensitivity(m,probe):
        total={};masked={};film_g={};film_b={};emb={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                total[lab]=m.act_inference_with_preference(probe,w)
                # FiLM-masked baseline = exact V2-A computation using current shared actor + embedding params.
                h=m._actor_hidden(probe,w)
                e=m.preference_embedding(w)
                masked[lab]=torch.tanh(m.actor_mean(torch.cat((h,e),dim=-1)))*m.ACTION_CLIP
                gb=m.preference_film(w);g,b=torch.chunk(gb,2,dim=-1)
                film_g[lab]=g;film_b[lab]=b;emb[lab]=e
    
        pair_total=pairwise_dist(total)
        pair_masked=pairwise_dist(masked)
        gamma_sep=pairwise_dist(film_g)
        beta_sep=pairwise_dist(film_b)
    
        authority=float(np.mean([
            torch.linalg.vector_norm(total[k]-masked[k],dim=1).mean().cpu() for k in ORDER
        ]))
    
        # Total d a / d w.
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w);rows=[]
        for j in range(a.shape[1]):
            rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
    
        # FiLM-masked d a / d w: RV1 direct + embedding only.
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        hm=m._actor_hidden(probe,wm);em=m.preference_embedding(wm)
        am=torch.tanh(m.actor_mean(torch.cat((hm,em),dim=-1)))*m.ACTION_CLIP
        mrows=[]
        for j in range(am.shape[1]):
            mrows.append(torch.autograd.grad(am[:,j].mean(),wm,retain_graph=True)[0])
        mjac=torch.stack(mrows,dim=1)
    
        wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1)
        with torch.no_grad():
            gbc=m.preference_film(wc);gc,bc=torch.chunk(gbc,2,dim=-1)
        return {
            "pairwise_action_distance":pair_total,
            "film_masked_pairwise_action_distance":pair_masked,
            "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_masked_jacobian_fro_mean":float(torch.linalg.matrix_norm(mjac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_action_authority_mean":authority,
            "film_weight_norm":float(m.preference_film.weight.norm().detach().cpu()),
            "film_bias_norm":float(m.preference_film.bias.norm().detach().cpu()),
            "gamma_abs_mean_center":float(gc.abs().mean().cpu()),
            "beta_abs_mean_center":float(bc.abs().mean().cpu()),
            "gamma_pairwise_distance":gamma_sep,
            "beta_pairwise_distance":beta_sep,
            "embedding_feature_pairwise_distance":pairwise_dist(emb),
            "preference_weight_norm":float(m.actor_body[0].weight[:,m.physical_obs_dim:].norm().detach().cpu())
        }
    
    def fresh_phase_audit(env,m,mgr,seed):
        phase_ev={"early":[],"late":[]};phase_bias={"early":[],"late":[]};surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            units=collect_support64(env,m,w,mgr,seed+qi*101)
            for unit in units:
                with torch.no_grad():
                    # Features already current-policy and head is linear.
                    F=torch.tensor(unit["F"],device="cuda",dtype=m.critic_head.weight.dtype)
                    V=m.critic_head(F).cpu().numpy()
                Y=unit["Y"];ph=unit["phase"]
                phase_ev[ph].extend(ev(Y[:,j],V[:,j]) for j in range(4))
                phase_bias[ph].extend(float(np.mean(V[:,j]-Y[:,j])) for j in range(4))
            surv.append(1.0) # deterministic support rollouts are diagnostic; detailed terminations in independent revalidation
        out={}
        for ph in ("early","late"):
            a=np.asarray(phase_ev[ph]);b=np.asarray(phase_bias[ph])
            out[ph]={"ev_mean":float(a.mean()),"negative_fraction":float((a<0).mean()),"mean_abs_bias":float(np.abs(b).mean())}
        out["combined_negative_fraction"]=float(np.mean(np.r_[np.asarray(phase_ev["early"])<0,np.asarray(phase_ev["late"])<0]))
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--gae-lambda",type=float,choices=[1.0],default=1.0)
        ap.add_argument("--adam-reset-update",type=int,default=0)
        ap.add_argument("--reset-mode",choices=["none","m1_eachstep"],default="none")
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);args=ap.parse_args()
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
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            init_path=V2B_INIT
            m.load_state_dict(torch.load(init_path,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            # Freeze anchor seed/preference specs once.
            cand=[]
            for k in range(ANCHOR_CANDIDATES):
                _,w=pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            # Select 6 seed specs on combined early+late summary geometry.
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
    
            adaptive_pools={"early":[],"late":[]};adaptive_summaries={"early":[],"late":[]}
            # initialize adaptive pools from all initialization candidates
            for _,_,units in cand:
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
    
            def current_anchor_units():
                out={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=pref_batch(k,torch.device("cuda"))
                    for u in collect_support64(env,m,w,mgr,seed):
                        out[u["phase"]].append(u)
                return out
            fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
    
            rows=[];snaps={}
            def audit(tag):
                m.eval();snaps[str(tag)]={"sensitivity":sensitivity(m,probe),
                    "phase_critic":fresh_phase_audit(env,m,mgr,args.seed+500000+int(tag)*1000)}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt");m.train()
            audit(0)
    
            for uidx in range(1,args.updates+1):
                adam_reset_applied=False
                if args.adam_reset_update>0 and uidx>args.adam_reset_update and args.reset_mode=="m1_eachstep":
                    for par in actor_params:
                        st=opt.state.get(par)
                        if st:
                            if "exp_avg" in st: st["exp_avg"].zero_()
                    adam_reset_applied=True
                labs,w=pref_batch(uidx,torch.device("cuda"))
                main=collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                # one recent adaptive 64-step support candidate contributes one early + one late unit
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                units=collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
                    if len(adaptive_pools[ph])>POOL_MAX:
                        adaptive_pools[ph].pop(0);adaptive_summaries[ph].pop(0)
                selected=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=args.gae_lambda)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pg=float(m.actor_body[0].weight.grad[:,m.physical_obs_dim:].norm().detach().cpu())
                fwgrad=float(m.preference_film.weight.grad.norm().detach().cpu()) if m.preference_film.weight.grad is not None else 0.0
                fbgrad=float(m.preference_film.bias.grad.norm().detach().cpu()) if m.preference_film.bias.grad is not None else 0.0
                # log optimizer memory and actual parameter delta around the real update
                grad_preclip=total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu())
                before=[x.detach().clone() for x in actor_params]
                m1_before=[];m2_before=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_before.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_before.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                delta_sq=0.0
                for bb,par in zip(before,actor_params):
                    delta_sq += float(((par.detach()-bb)**2).sum().cpu())
                actual_delta_norm=float(delta_sq**0.5)
                m1_after=[];m2_after=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_after.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_after.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                obs=main["obs"].detach()
                rt=main["rt"].detach()
                visitation={
                  "obs_mean_norm":float(obs.mean(0).norm().cpu()),
                  "obs_std_mean":float(obs.std(0).mean().cpu()),
                  "objective_mean":rt.mean((0,1)).cpu().tolist(),
                  "termination_fraction":main["termination_fraction"]}
    
                post_refresh=None
                # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
                if uidx in SNAPS:
                    post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                             "preference_input_grad_norm":pg,"film_weight_grad_norm":fwgrad,"film_bias_grad_norm":fbgrad,"actor_grad_norm_preclip":grad_preclip,
                             "actual_param_delta_norm":actual_delta_norm,
                             "adam_m1_norm_sum_before":float(sum(m1_before)),"adam_m2_norm_sum_before":float(sum(m2_before)),
                             "adam_m1_norm_sum_after":float(sum(m1_after)),"adam_m2_norm_sum_after":float(sum(m2_after)),
                             "adam_reset_applied":adam_reset_applied,"visitation":visitation,
                             "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
                if uidx in SNAPS:audit(uidx)
    
            report={"schema":"v2b_adam_m1_eachstep_isolation_v4","seed":args.seed,"updates":args.updates,
                    "training_scope":"optimizer-history vs visitation isolation under frozen V2-B + Foundation V2",
                    "actor_ppo_objectives_changed":False,"architecture":"v2b","gae_lambda":args.gae_lambda,"adam_reset_update":args.adam_reset_update,"reset_mode":args.reset_mode,"anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            final=snaps[str(args.updates)]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            crit=final["phase_critic"];sens=final["sensitivity"]
            criteria={
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
            report["summary"]={
              "status":"LAMBDA-PILOT FOUNDATION PASS" if passed else "LAMBDA-PILOT FOUNDATION FAIL",
              "gae_lambda":args.gae_lambda,
              "criteria":criteria,
              "final":final,
              "last10_termination_fraction":float(term[-10:].mean()),
              "max_ratio_error":float(ratio.max())}
            (args.output_dir/"adam_history_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_adam_m1_isolation():
    """Run former v2b_adam_m1_isolation.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    V2B_INIT=ROOT/"runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2b_adam_history_isolation"
    ORDER=("T","A","O","S","C")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;SUP_H=64;NENV=8;POOL_MAX=24
    ANCHOR_CANDIDATES=12;ANCHOR_PHASE_K=3;ADAPT_PHASE_K=3
    SNAPS=(0,10,25,50,75)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        sel=[int(np.argmax(np.mean(X*X,axis=1)))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_actor(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        with torch.no_grad():
            for _ in range(H):
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool()
        out={"obs":torch.cat(ob),"w":w.repeat(H,1),"rt":rt,"dt":dt,"next_obs":cur,
             "termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def collect_support64(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();obs=torch.stack(ob) # [64,E,D]
        C=np.stack(cmd) # [64,E,3]
        phases=[]
        for pi,(st,en) in enumerate(((0,32),(32,64))):
            Y=trunc(rt[st:en],dt[st:en]).reshape(-1,4).detach()
            po=obs[st:en].reshape(-1,obs.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).detach()
            cc=C[st:en].reshape(-1,C.shape[-1])
            summary=np.r_[cc.mean(0),cc.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),float(pi)]
            phases.append({"F":F.cpu().numpy(),"Y":Y.cpu().numpy(),"summary":summary.astype(np.float32),
                           "phase":"early" if pi==0 else "late"})
        return phases
    
    def fit_expanded_current_policy(m,anchor_units,adaptive_pools):
        support=[]
        counts={}
        for phase in ("early","late"):
            au=anchor_units[phase]
            ad=adaptive_pools[phase]
            support += [(u["F"],u["Y"]) for u in au]
            support += [(u["F"],u["Y"]) for u in ad]
            counts[phase]={"anchor_units":len(au),"adaptive_units":len(ad)}
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return counts
    
    def pairwise_dist(actions):
        labs=list(actions);v=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:v.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def sensitivity(m,probe):
        total={};masked={};film_g={};film_b={};emb={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                total[lab]=m.act_inference_with_preference(probe,w)
                # FiLM-masked baseline = exact V2-A computation using current shared actor + embedding params.
                h=m._actor_hidden(probe,w)
                e=m.preference_embedding(w)
                masked[lab]=torch.tanh(m.actor_mean(torch.cat((h,e),dim=-1)))*m.ACTION_CLIP
                gb=m.preference_film(w);g,b=torch.chunk(gb,2,dim=-1)
                film_g[lab]=g;film_b[lab]=b;emb[lab]=e
    
        pair_total=pairwise_dist(total)
        pair_masked=pairwise_dist(masked)
        gamma_sep=pairwise_dist(film_g)
        beta_sep=pairwise_dist(film_b)
    
        authority=float(np.mean([
            torch.linalg.vector_norm(total[k]-masked[k],dim=1).mean().cpu() for k in ORDER
        ]))
    
        # Total d a / d w.
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w);rows=[]
        for j in range(a.shape[1]):
            rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
    
        # FiLM-masked d a / d w: RV1 direct + embedding only.
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        hm=m._actor_hidden(probe,wm);em=m.preference_embedding(wm)
        am=torch.tanh(m.actor_mean(torch.cat((hm,em),dim=-1)))*m.ACTION_CLIP
        mrows=[]
        for j in range(am.shape[1]):
            mrows.append(torch.autograd.grad(am[:,j].mean(),wm,retain_graph=True)[0])
        mjac=torch.stack(mrows,dim=1)
    
        wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1)
        with torch.no_grad():
            gbc=m.preference_film(wc);gc,bc=torch.chunk(gbc,2,dim=-1)
        return {
            "pairwise_action_distance":pair_total,
            "film_masked_pairwise_action_distance":pair_masked,
            "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_masked_jacobian_fro_mean":float(torch.linalg.matrix_norm(mjac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_action_authority_mean":authority,
            "film_weight_norm":float(m.preference_film.weight.norm().detach().cpu()),
            "film_bias_norm":float(m.preference_film.bias.norm().detach().cpu()),
            "gamma_abs_mean_center":float(gc.abs().mean().cpu()),
            "beta_abs_mean_center":float(bc.abs().mean().cpu()),
            "gamma_pairwise_distance":gamma_sep,
            "beta_pairwise_distance":beta_sep,
            "embedding_feature_pairwise_distance":pairwise_dist(emb),
            "preference_weight_norm":float(m.actor_body[0].weight[:,m.physical_obs_dim:].norm().detach().cpu())
        }
    
    def fresh_phase_audit(env,m,mgr,seed):
        phase_ev={"early":[],"late":[]};phase_bias={"early":[],"late":[]};surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            units=collect_support64(env,m,w,mgr,seed+qi*101)
            for unit in units:
                with torch.no_grad():
                    # Features already current-policy and head is linear.
                    F=torch.tensor(unit["F"],device="cuda",dtype=m.critic_head.weight.dtype)
                    V=m.critic_head(F).cpu().numpy()
                Y=unit["Y"];ph=unit["phase"]
                phase_ev[ph].extend(ev(Y[:,j],V[:,j]) for j in range(4))
                phase_bias[ph].extend(float(np.mean(V[:,j]-Y[:,j])) for j in range(4))
            surv.append(1.0) # deterministic support rollouts are diagnostic; detailed terminations in independent revalidation
        out={}
        for ph in ("early","late"):
            a=np.asarray(phase_ev[ph]);b=np.asarray(phase_bias[ph])
            out[ph]={"ev_mean":float(a.mean()),"negative_fraction":float((a<0).mean()),"mean_abs_bias":float(np.abs(b).mean())}
        out["combined_negative_fraction"]=float(np.mean(np.r_[np.asarray(phase_ev["early"])<0,np.asarray(phase_ev["late"])<0]))
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--gae-lambda",type=float,choices=[1.0],default=1.0)
        ap.add_argument("--adam-reset-update",type=int,default=0)
        ap.add_argument("--reset-mode",choices=["none","m1_only"],default="none")
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);args=ap.parse_args()
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
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            init_path=V2B_INIT
            m.load_state_dict(torch.load(init_path,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            # Freeze anchor seed/preference specs once.
            cand=[]
            for k in range(ANCHOR_CANDIDATES):
                _,w=pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            # Select 6 seed specs on combined early+late summary geometry.
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
    
            adaptive_pools={"early":[],"late":[]};adaptive_summaries={"early":[],"late":[]}
            # initialize adaptive pools from all initialization candidates
            for _,_,units in cand:
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
    
            def current_anchor_units():
                out={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=pref_batch(k,torch.device("cuda"))
                    for u in collect_support64(env,m,w,mgr,seed):
                        out[u["phase"]].append(u)
                return out
            fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
    
            rows=[];snaps={}
            def audit(tag):
                m.eval();snaps[str(tag)]={"sensitivity":sensitivity(m,probe),
                    "phase_critic":fresh_phase_audit(env,m,mgr,args.seed+500000+int(tag)*1000)}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt");m.train()
            audit(0)
    
            for uidx in range(1,args.updates+1):
                adam_reset_applied=False
                if args.adam_reset_update>0 and uidx==args.adam_reset_update+1 and args.reset_mode=="m1_only":
                    for par in actor_params:
                        st=opt.state.get(par)
                        if st:
                            if "exp_avg" in st: st["exp_avg"].zero_()
                    adam_reset_applied=True
                labs,w=pref_batch(uidx,torch.device("cuda"))
                main=collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                # one recent adaptive 64-step support candidate contributes one early + one late unit
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                units=collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
                    if len(adaptive_pools[ph])>POOL_MAX:
                        adaptive_pools[ph].pop(0);adaptive_summaries[ph].pop(0)
                selected=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=args.gae_lambda)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pg=float(m.actor_body[0].weight.grad[:,m.physical_obs_dim:].norm().detach().cpu())
                fwgrad=float(m.preference_film.weight.grad.norm().detach().cpu()) if m.preference_film.weight.grad is not None else 0.0
                fbgrad=float(m.preference_film.bias.grad.norm().detach().cpu()) if m.preference_film.bias.grad is not None else 0.0
                # log optimizer memory and actual parameter delta around the real update
                grad_preclip=total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu())
                before=[x.detach().clone() for x in actor_params]
                m1_before=[];m2_before=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_before.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_before.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                delta_sq=0.0
                for bb,par in zip(before,actor_params):
                    delta_sq += float(((par.detach()-bb)**2).sum().cpu())
                actual_delta_norm=float(delta_sq**0.5)
                m1_after=[];m2_after=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_after.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_after.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                obs=main["obs"].detach()
                rt=main["rt"].detach()
                visitation={
                  "obs_mean_norm":float(obs.mean(0).norm().cpu()),
                  "obs_std_mean":float(obs.std(0).mean().cpu()),
                  "objective_mean":rt.mean((0,1)).cpu().tolist(),
                  "termination_fraction":main["termination_fraction"]}
    
                post_refresh=None
                # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
                if uidx in SNAPS:
                    post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                             "preference_input_grad_norm":pg,"film_weight_grad_norm":fwgrad,"film_bias_grad_norm":fbgrad,"actor_grad_norm_preclip":grad_preclip,
                             "actual_param_delta_norm":actual_delta_norm,
                             "adam_m1_norm_sum_before":float(sum(m1_before)),"adam_m2_norm_sum_before":float(sum(m2_before)),
                             "adam_m1_norm_sum_after":float(sum(m1_after)),"adam_m2_norm_sum_after":float(sum(m2_after)),
                             "adam_reset_applied":adam_reset_applied,"visitation":visitation,
                             "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
                if uidx in SNAPS:audit(uidx)
    
            report={"schema":"v2b_adam_m1_isolation_v3","seed":args.seed,"updates":args.updates,
                    "training_scope":"optimizer-history vs visitation isolation under frozen V2-B + Foundation V2",
                    "actor_ppo_objectives_changed":False,"architecture":"v2b","gae_lambda":args.gae_lambda,"adam_reset_update":args.adam_reset_update,"reset_mode":args.reset_mode,"anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            final=snaps[str(args.updates)]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            crit=final["phase_critic"];sens=final["sensitivity"]
            criteria={
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
            report["summary"]={
              "status":"LAMBDA-PILOT FOUNDATION PASS" if passed else "LAMBDA-PILOT FOUNDATION FAIL",
              "gae_lambda":args.gae_lambda,
              "criteria":criteria,
              "final":final,
              "last10_termination_fraction":float(term[-10:].mean()),
              "max_ratio_error":float(ratio.max())}
            (args.output_dir/"adam_history_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_adam_moment_reset_preserve_step():
    """Run former v2b_adam_moment_reset_preserve_step.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    V2B_INIT=ROOT/"runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2b_adam_moment_reset_preserve_step"
    ORDER=("T","A","O","S","C")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;SUP_H=64;NENV=8;POOL_MAX=24
    ANCHOR_CANDIDATES=12;ANCHOR_PHASE_K=3;ADAPT_PHASE_K=3
    SNAPS=(0,10,25,50,75)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        sel=[int(np.argmax(np.mean(X*X,axis=1)))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_actor(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        with torch.no_grad():
            for _ in range(H):
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool()
        out={"obs":torch.cat(ob),"w":w.repeat(H,1),"rt":rt,"dt":dt,"next_obs":cur,
             "termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def collect_support64(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();obs=torch.stack(ob) # [64,E,D]
        C=np.stack(cmd) # [64,E,3]
        phases=[]
        for pi,(st,en) in enumerate(((0,32),(32,64))):
            Y=trunc(rt[st:en],dt[st:en]).reshape(-1,4).detach()
            po=obs[st:en].reshape(-1,obs.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).detach()
            cc=C[st:en].reshape(-1,C.shape[-1])
            summary=np.r_[cc.mean(0),cc.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),float(pi)]
            phases.append({"F":F.cpu().numpy(),"Y":Y.cpu().numpy(),"summary":summary.astype(np.float32),
                           "phase":"early" if pi==0 else "late"})
        return phases
    
    def fit_expanded_current_policy(m,anchor_units,adaptive_pools):
        support=[]
        counts={}
        for phase in ("early","late"):
            au=anchor_units[phase]
            ad=adaptive_pools[phase]
            support += [(u["F"],u["Y"]) for u in au]
            support += [(u["F"],u["Y"]) for u in ad]
            counts[phase]={"anchor_units":len(au),"adaptive_units":len(ad)}
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return counts
    
    def pairwise_dist(actions):
        labs=list(actions);v=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:v.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def sensitivity(m,probe):
        total={};masked={};film_g={};film_b={};emb={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                total[lab]=m.act_inference_with_preference(probe,w)
                # FiLM-masked baseline = exact V2-A computation using current shared actor + embedding params.
                h=m._actor_hidden(probe,w)
                e=m.preference_embedding(w)
                masked[lab]=torch.tanh(m.actor_mean(torch.cat((h,e),dim=-1)))*m.ACTION_CLIP
                gb=m.preference_film(w);g,b=torch.chunk(gb,2,dim=-1)
                film_g[lab]=g;film_b[lab]=b;emb[lab]=e
    
        pair_total=pairwise_dist(total)
        pair_masked=pairwise_dist(masked)
        gamma_sep=pairwise_dist(film_g)
        beta_sep=pairwise_dist(film_b)
    
        authority=float(np.mean([
            torch.linalg.vector_norm(total[k]-masked[k],dim=1).mean().cpu() for k in ORDER
        ]))
    
        # Total d a / d w.
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w);rows=[]
        for j in range(a.shape[1]):
            rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
    
        # FiLM-masked d a / d w: RV1 direct + embedding only.
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        hm=m._actor_hidden(probe,wm);em=m.preference_embedding(wm)
        am=torch.tanh(m.actor_mean(torch.cat((hm,em),dim=-1)))*m.ACTION_CLIP
        mrows=[]
        for j in range(am.shape[1]):
            mrows.append(torch.autograd.grad(am[:,j].mean(),wm,retain_graph=True)[0])
        mjac=torch.stack(mrows,dim=1)
    
        wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1)
        with torch.no_grad():
            gbc=m.preference_film(wc);gc,bc=torch.chunk(gbc,2,dim=-1)
        return {
            "pairwise_action_distance":pair_total,
            "film_masked_pairwise_action_distance":pair_masked,
            "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_masked_jacobian_fro_mean":float(torch.linalg.matrix_norm(mjac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_action_authority_mean":authority,
            "film_weight_norm":float(m.preference_film.weight.norm().detach().cpu()),
            "film_bias_norm":float(m.preference_film.bias.norm().detach().cpu()),
            "gamma_abs_mean_center":float(gc.abs().mean().cpu()),
            "beta_abs_mean_center":float(bc.abs().mean().cpu()),
            "gamma_pairwise_distance":gamma_sep,
            "beta_pairwise_distance":beta_sep,
            "embedding_feature_pairwise_distance":pairwise_dist(emb),
            "preference_weight_norm":float(m.actor_body[0].weight[:,m.physical_obs_dim:].norm().detach().cpu())
        }
    
    def fresh_phase_audit(env,m,mgr,seed):
        phase_ev={"early":[],"late":[]};phase_bias={"early":[],"late":[]};surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            units=collect_support64(env,m,w,mgr,seed+qi*101)
            for unit in units:
                with torch.no_grad():
                    # Features already current-policy and head is linear.
                    F=torch.tensor(unit["F"],device="cuda",dtype=m.critic_head.weight.dtype)
                    V=m.critic_head(F).cpu().numpy()
                Y=unit["Y"];ph=unit["phase"]
                phase_ev[ph].extend(ev(Y[:,j],V[:,j]) for j in range(4))
                phase_bias[ph].extend(float(np.mean(V[:,j]-Y[:,j])) for j in range(4))
            surv.append(1.0) # deterministic support rollouts are diagnostic; detailed terminations in independent revalidation
        out={}
        for ph in ("early","late"):
            a=np.asarray(phase_ev[ph]);b=np.asarray(phase_bias[ph])
            out[ph]={"ev_mean":float(a.mean()),"negative_fraction":float((a<0).mean()),"mean_abs_bias":float(np.abs(b).mean())}
        out["combined_negative_fraction"]=float(np.mean(np.r_[np.asarray(phase_ev["early"])<0,np.asarray(phase_ev["late"])<0]))
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--gae-lambda",type=float,choices=[1.0],default=1.0)
        ap.add_argument("--adam-reset-update",type=int,default=0)
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);args=ap.parse_args()
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
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            init_path=V2B_INIT
            m.load_state_dict(torch.load(init_path,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            # Freeze anchor seed/preference specs once.
            cand=[]
            for k in range(ANCHOR_CANDIDATES):
                _,w=pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            # Select 6 seed specs on combined early+late summary geometry.
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
    
            adaptive_pools={"early":[],"late":[]};adaptive_summaries={"early":[],"late":[]}
            # initialize adaptive pools from all initialization candidates
            for _,_,units in cand:
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
    
            def current_anchor_units():
                out={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=pref_batch(k,torch.device("cuda"))
                    for u in collect_support64(env,m,w,mgr,seed):
                        out[u["phase"]].append(u)
                return out
            fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
    
            rows=[];snaps={}
            def audit(tag):
                m.eval();snaps[str(tag)]={"sensitivity":sensitivity(m,probe),
                    "phase_critic":fresh_phase_audit(env,m,mgr,args.seed+500000+int(tag)*1000)}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt");m.train()
            audit(0)
    
            for uidx in range(1,args.updates+1):
                adam_reset_applied=False
                if args.adam_reset_update>0 and uidx==args.adam_reset_update+1:
                    for par in actor_params:
                        st=opt.state.get(par,{})
                        if "exp_avg" in st: st["exp_avg"].zero_()
                        if "exp_avg_sq" in st: st["exp_avg_sq"].zero_()
                        if "max_exp_avg_sq" in st: st["max_exp_avg_sq"].zero_()
                        # Preserve Adam step counter / bias-correction clock.
                    adam_reset_applied=True
                labs,w=pref_batch(uidx,torch.device("cuda"))
                main=collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                # one recent adaptive 64-step support candidate contributes one early + one late unit
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                units=collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
                    if len(adaptive_pools[ph])>POOL_MAX:
                        adaptive_pools[ph].pop(0);adaptive_summaries[ph].pop(0)
                selected=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=args.gae_lambda)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pg=float(m.actor_body[0].weight.grad[:,m.physical_obs_dim:].norm().detach().cpu())
                fwgrad=float(m.preference_film.weight.grad.norm().detach().cpu()) if m.preference_film.weight.grad is not None else 0.0
                fbgrad=float(m.preference_film.bias.grad.norm().detach().cpu()) if m.preference_film.bias.grad is not None else 0.0
                # log optimizer memory and actual parameter delta around the real update
                grad_preclip=total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu())
                before=[x.detach().clone() for x in actor_params]
                m1_before=[];m2_before=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_before.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_before.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                delta_sq=0.0
                for bb,par in zip(before,actor_params):
                    delta_sq += float(((par.detach()-bb)**2).sum().cpu())
                actual_delta_norm=float(delta_sq**0.5)
                m1_after=[];m2_after=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_after.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_after.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                obs=main["obs"].detach()
                rt=main["rt"].detach()
                visitation={
                  "obs_mean_norm":float(obs.mean(0).norm().cpu()),
                  "obs_std_mean":float(obs.std(0).mean().cpu()),
                  "objective_mean":rt.mean((0,1)).cpu().tolist(),
                  "termination_fraction":main["termination_fraction"]}
    
                post_refresh=None
                # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
                if uidx in SNAPS:
                    post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                             "preference_input_grad_norm":pg,"film_weight_grad_norm":fwgrad,"film_bias_grad_norm":fbgrad,"actor_grad_norm_preclip":grad_preclip,
                             "actual_param_delta_norm":actual_delta_norm,
                             "adam_m1_norm_sum_before":float(sum(m1_before)),"adam_m2_norm_sum_before":float(sum(m2_before)),
                             "adam_m1_norm_sum_after":float(sum(m1_after)),"adam_m2_norm_sum_after":float(sum(m2_after)),
                             "adam_reset_applied":adam_reset_applied,"visitation":visitation,
                             "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
                if uidx in SNAPS:audit(uidx)
    
            report={"schema":"v2b_adam_moment_reset_preserve_step_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"optimizer-history vs visitation isolation under frozen V2-B + Foundation V2",
                    "actor_ppo_objectives_changed":False,"architecture":"v2b","gae_lambda":args.gae_lambda,"adam_reset_update":args.adam_reset_update,"adam_reset_mode":"zero_moments_preserve_step","anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            final=snaps[str(args.updates)]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            crit=final["phase_critic"];sens=final["sensitivity"]
            criteria={
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
            report["summary"]={
              "status":"LAMBDA-PILOT FOUNDATION PASS" if passed else "LAMBDA-PILOT FOUNDATION FAIL",
              "gae_lambda":args.gae_lambda,
              "criteria":criteria,
              "final":final,
              "last10_termination_fraction":float(term[-10:].mean()),
              "max_ratio_error":float(ratio.max())}
            (args.output_dir/"adam_history_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_adam_moments_isolation():
    """Run former v2b_adam_moments_isolation.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    V2B_INIT=ROOT/"runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2b_adam_history_isolation"
    ORDER=("T","A","O","S","C")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;SUP_H=64;NENV=8;POOL_MAX=24
    ANCHOR_CANDIDATES=12;ANCHOR_PHASE_K=3;ADAPT_PHASE_K=3
    SNAPS=(0,10,25,50,75)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        sel=[int(np.argmax(np.mean(X*X,axis=1)))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_actor(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        with torch.no_grad():
            for _ in range(H):
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool()
        out={"obs":torch.cat(ob),"w":w.repeat(H,1),"rt":rt,"dt":dt,"next_obs":cur,
             "termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def collect_support64(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();obs=torch.stack(ob) # [64,E,D]
        C=np.stack(cmd) # [64,E,3]
        phases=[]
        for pi,(st,en) in enumerate(((0,32),(32,64))):
            Y=trunc(rt[st:en],dt[st:en]).reshape(-1,4).detach()
            po=obs[st:en].reshape(-1,obs.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).detach()
            cc=C[st:en].reshape(-1,C.shape[-1])
            summary=np.r_[cc.mean(0),cc.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),float(pi)]
            phases.append({"F":F.cpu().numpy(),"Y":Y.cpu().numpy(),"summary":summary.astype(np.float32),
                           "phase":"early" if pi==0 else "late"})
        return phases
    
    def fit_expanded_current_policy(m,anchor_units,adaptive_pools):
        support=[]
        counts={}
        for phase in ("early","late"):
            au=anchor_units[phase]
            ad=adaptive_pools[phase]
            support += [(u["F"],u["Y"]) for u in au]
            support += [(u["F"],u["Y"]) for u in ad]
            counts[phase]={"anchor_units":len(au),"adaptive_units":len(ad)}
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return counts
    
    def pairwise_dist(actions):
        labs=list(actions);v=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:v.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def sensitivity(m,probe):
        total={};masked={};film_g={};film_b={};emb={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                total[lab]=m.act_inference_with_preference(probe,w)
                # FiLM-masked baseline = exact V2-A computation using current shared actor + embedding params.
                h=m._actor_hidden(probe,w)
                e=m.preference_embedding(w)
                masked[lab]=torch.tanh(m.actor_mean(torch.cat((h,e),dim=-1)))*m.ACTION_CLIP
                gb=m.preference_film(w);g,b=torch.chunk(gb,2,dim=-1)
                film_g[lab]=g;film_b[lab]=b;emb[lab]=e
    
        pair_total=pairwise_dist(total)
        pair_masked=pairwise_dist(masked)
        gamma_sep=pairwise_dist(film_g)
        beta_sep=pairwise_dist(film_b)
    
        authority=float(np.mean([
            torch.linalg.vector_norm(total[k]-masked[k],dim=1).mean().cpu() for k in ORDER
        ]))
    
        # Total d a / d w.
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w);rows=[]
        for j in range(a.shape[1]):
            rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
    
        # FiLM-masked d a / d w: RV1 direct + embedding only.
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        hm=m._actor_hidden(probe,wm);em=m.preference_embedding(wm)
        am=torch.tanh(m.actor_mean(torch.cat((hm,em),dim=-1)))*m.ACTION_CLIP
        mrows=[]
        for j in range(am.shape[1]):
            mrows.append(torch.autograd.grad(am[:,j].mean(),wm,retain_graph=True)[0])
        mjac=torch.stack(mrows,dim=1)
    
        wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1)
        with torch.no_grad():
            gbc=m.preference_film(wc);gc,bc=torch.chunk(gbc,2,dim=-1)
        return {
            "pairwise_action_distance":pair_total,
            "film_masked_pairwise_action_distance":pair_masked,
            "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_masked_jacobian_fro_mean":float(torch.linalg.matrix_norm(mjac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "film_action_authority_mean":authority,
            "film_weight_norm":float(m.preference_film.weight.norm().detach().cpu()),
            "film_bias_norm":float(m.preference_film.bias.norm().detach().cpu()),
            "gamma_abs_mean_center":float(gc.abs().mean().cpu()),
            "beta_abs_mean_center":float(bc.abs().mean().cpu()),
            "gamma_pairwise_distance":gamma_sep,
            "beta_pairwise_distance":beta_sep,
            "embedding_feature_pairwise_distance":pairwise_dist(emb),
            "preference_weight_norm":float(m.actor_body[0].weight[:,m.physical_obs_dim:].norm().detach().cpu())
        }
    
    def fresh_phase_audit(env,m,mgr,seed):
        phase_ev={"early":[],"late":[]};phase_bias={"early":[],"late":[]};surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            units=collect_support64(env,m,w,mgr,seed+qi*101)
            for unit in units:
                with torch.no_grad():
                    # Features already current-policy and head is linear.
                    F=torch.tensor(unit["F"],device="cuda",dtype=m.critic_head.weight.dtype)
                    V=m.critic_head(F).cpu().numpy()
                Y=unit["Y"];ph=unit["phase"]
                phase_ev[ph].extend(ev(Y[:,j],V[:,j]) for j in range(4))
                phase_bias[ph].extend(float(np.mean(V[:,j]-Y[:,j])) for j in range(4))
            surv.append(1.0) # deterministic support rollouts are diagnostic; detailed terminations in independent revalidation
        out={}
        for ph in ("early","late"):
            a=np.asarray(phase_ev[ph]);b=np.asarray(phase_bias[ph])
            out[ph]={"ev_mean":float(a.mean()),"negative_fraction":float((a<0).mean()),"mean_abs_bias":float(np.abs(b).mean())}
        out["combined_negative_fraction"]=float(np.mean(np.r_[np.asarray(phase_ev["early"])<0,np.asarray(phase_ev["late"])<0]))
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--gae-lambda",type=float,choices=[1.0],default=1.0)
        ap.add_argument("--adam-reset-update",type=int,default=0)
        ap.add_argument("--reset-mode",choices=["none","moments_only"],default="none")
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);args=ap.parse_args()
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
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            init_path=V2B_INIT
            m.load_state_dict(torch.load(init_path,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            # Freeze anchor seed/preference specs once.
            cand=[]
            for k in range(ANCHOR_CANDIDATES):
                _,w=pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            # Select 6 seed specs on combined early+late summary geometry.
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
    
            adaptive_pools={"early":[],"late":[]};adaptive_summaries={"early":[],"late":[]}
            # initialize adaptive pools from all initialization candidates
            for _,_,units in cand:
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
    
            def current_anchor_units():
                out={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=pref_batch(k,torch.device("cuda"))
                    for u in collect_support64(env,m,w,mgr,seed):
                        out[u["phase"]].append(u)
                return out
            fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
    
            rows=[];snaps={}
            def audit(tag):
                m.eval();snaps[str(tag)]={"sensitivity":sensitivity(m,probe),
                    "phase_critic":fresh_phase_audit(env,m,mgr,args.seed+500000+int(tag)*1000)}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt");m.train()
            audit(0)
    
            for uidx in range(1,args.updates+1):
                adam_reset_applied=False
                if args.adam_reset_update>0 and uidx==args.adam_reset_update+1 and args.reset_mode=="moments_only":
                    for par in actor_params:
                        st=opt.state.get(par)
                        if st:
                            if "exp_avg" in st: st["exp_avg"].zero_()
                            if "exp_avg_sq" in st: st["exp_avg_sq"].zero_()
                            if "max_exp_avg_sq" in st: st["max_exp_avg_sq"].zero_()
                    adam_reset_applied=True
                labs,w=pref_batch(uidx,torch.device("cuda"))
                main=collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                # one recent adaptive 64-step support candidate contributes one early + one late unit
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                units=collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
                    if len(adaptive_pools[ph])>POOL_MAX:
                        adaptive_pools[ph].pop(0);adaptive_summaries[ph].pop(0)
                selected=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=args.gae_lambda)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pg=float(m.actor_body[0].weight.grad[:,m.physical_obs_dim:].norm().detach().cpu())
                fwgrad=float(m.preference_film.weight.grad.norm().detach().cpu()) if m.preference_film.weight.grad is not None else 0.0
                fbgrad=float(m.preference_film.bias.grad.norm().detach().cpu()) if m.preference_film.bias.grad is not None else 0.0
                # log optimizer memory and actual parameter delta around the real update
                grad_preclip=total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu())
                before=[x.detach().clone() for x in actor_params]
                m1_before=[];m2_before=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_before.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_before.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                delta_sq=0.0
                for bb,par in zip(before,actor_params):
                    delta_sq += float(((par.detach()-bb)**2).sum().cpu())
                actual_delta_norm=float(delta_sq**0.5)
                m1_after=[];m2_after=[]
                for par in actor_params:
                    st=opt.state.get(par,{})
                    m1_after.append(float(st.get("exp_avg",torch.zeros_like(par)).norm().detach().cpu()))
                    m2_after.append(float(st.get("exp_avg_sq",torch.zeros_like(par)).norm().detach().cpu()))
                obs=main["obs"].detach()
                rt=main["rt"].detach()
                visitation={
                  "obs_mean_norm":float(obs.mean(0).norm().cpu()),
                  "obs_std_mean":float(obs.std(0).mean().cpu()),
                  "objective_mean":rt.mean((0,1)).cpu().tolist(),
                  "termination_fraction":main["termination_fraction"]}
    
                post_refresh=None
                # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
                if uidx in SNAPS:
                    post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                             "preference_input_grad_norm":pg,"film_weight_grad_norm":fwgrad,"film_bias_grad_norm":fbgrad,"actor_grad_norm_preclip":grad_preclip,
                             "actual_param_delta_norm":actual_delta_norm,
                             "adam_m1_norm_sum_before":float(sum(m1_before)),"adam_m2_norm_sum_before":float(sum(m2_before)),
                             "adam_m1_norm_sum_after":float(sum(m1_after)),"adam_m2_norm_sum_after":float(sum(m2_after)),
                             "adam_reset_applied":adam_reset_applied,"visitation":visitation,
                             "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
                if uidx in SNAPS:audit(uidx)
    
            report={"schema":"v2b_adam_moments_isolation_v2","seed":args.seed,"updates":args.updates,
                    "training_scope":"optimizer-history vs visitation isolation under frozen V2-B + Foundation V2",
                    "actor_ppo_objectives_changed":False,"architecture":"v2b","gae_lambda":args.gae_lambda,"adam_reset_update":args.adam_reset_update,"reset_mode":args.reset_mode,"anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            final=snaps[str(args.updates)]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            crit=final["phase_critic"];sens=final["sensitivity"]
            criteria={
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
            report["summary"]={
              "status":"LAMBDA-PILOT FOUNDATION PASS" if passed else "LAMBDA-PILOT FOUNDATION FAIL",
              "gae_lambda":args.gae_lambda,
              "criteria":criteria,
              "final":final,
              "last10_termination_fraction":float(term[-10:].mean()),
              "max_ratio_error":float(ratio.max())}
            (args.output_dir/"adam_history_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_mixed_training_batch_geometry_audit():
    """Run former v2b_mixed_training_batch_geometry_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from rl.experiments.common.utilities.v2b_fixed_stream_multiupdate_audit import ot,pref_batch,collect_actor,geometry,ORDER,groups
    STAGES=(10,25,50,75);SEEDS=(940001,940002,940003,940004)
    ARMS={"095":0.95,"100":1.0}
    OUT=ROOT/"runs/v2b_mixed_training_batch_geometry_audit-2026-09-24"
    
    def agg(rows):
        def rec(vals):
            v0=vals[0]
            if isinstance(v0,dict):return {k:rec([v[k] for v in vals]) for k in v0}
            if isinstance(v0,(int,float,np.floating)):
                a=np.asarray(vals,float);return {"mean":float(a.mean()),"std":float(a.std())}
            return v0
        return rec(rows)
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            raw={arm:{} for arm in ARMS}
            for arm,actual_lam in ARMS.items():
                for st in STAGES:
                    ck=ROOT/f"runs/v2b_lambda{arm}_pilot-2026-09-23/model_{st}.pt"
                    m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
                    m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
                    raw[arm][str(st)]={"0.95":[],"1.0":[]}
                    for si,seed in enumerate(SEEDS):
                        _,w=pref_batch(st+si+1,torch.device("cuda"))
                        b=collect_actor(env,m,w,mgr,seed)
                        for lam in (0.95,1.0):
                            g,ratio=geometry(m,b,lam)
                            raw[arm][str(st)][str(lam)].append({"geometry":g,"ratio_error":ratio})
            aggregate={arm:{st:{lam:agg(rows) for lam,rows in lams.items()} for st,lams in stages.items()} for arm,stages in raw.items()}
            compact={}
            for st in map(str,STAGES):
                compact[st]={}
                c=aggregate["095"][st]["0.95"]["geometry"]
                t=aggregate["100"][st]["1.0"]["geometry"]
                for gn in ("all_actor","shared_body","direct_input","embedding","film","actor_head"):
                    cg=c[gn];tg=t[gn]
                    compact[st][gn]={
                        "control_share":cg["share"],
                        "treatment_share":tg["share"],
                        "share_delta":{lab:tg["share"][lab]["mean"]-cg["share"][lab]["mean"] for lab in ORDER},
                        "control_combined_cos":cg["combined_cos"],
                        "treatment_combined_cos":tg["combined_cos"],
                        "combined_cos_delta":{lab:tg["combined_cos"][lab]["mean"]-cg["combined_cos"][lab]["mean"] for lab in ORDER},
                        "control_pairwise":cg["pairwise_cos"],
                        "treatment_pairwise":tg["pairwise_cos"],
                        "pairwise_delta":{k:tg["pairwise_cos"][k]["mean"]-cg["pairwise_cos"][k]["mean"] for k in cg["pairwise_cos"]},
                        "combined_norm_ratio":tg["combined_norm"]["mean"]/(cg["combined_norm"]["mean"]+1e-12)}
            OUT.mkdir(parents=True,exist_ok=True)
            report={"schema":"v2b_mixed_training_batch_geometry_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "stages":list(STAGES),"seeds":list(SEEDS),"fresh_on_policy_batches":True,
                    "batch_structure":"exact training-style mixed endpoint preferences, 2 envs per T/A/O/S",
                    "arms":{"095":"actual lambda=0.95 checkpoints","100":"actual lambda=1.0 checkpoints"},
                    "aggregate":aggregate,"compact":compact}
            out=OUT/"mixed_training_batch_geometry_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (OUT/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps(compact,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_paired_joint_update_geometry_audit():
    """Run former v2b_paired_joint_update_geometry_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_paired_joint_update_geometry_audit-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32)}
    LAMS=(0.95,1.0)
    G=.99;H=32;NENV=8
    SEEDS=(930001,930002,930003,930004)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1);den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def flat_grads(loss,params):
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])
    def pairwise(grads):
        out={}
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i: out[f"{a}-{b}"]=cos(grads[a],grads[b])
        return out
    
    def collect(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w.repeat(H,1)}
    
    def gae(reward,value,nextv,done,lam):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*lam*nt*last;adv[t]=last
        return adv
    
    def compute_for_lambda(m,batch,pref_lab,lam):
        w=batch["w"];obs=batch["obs"]
        with torch.no_grad():
            vt=m.value_with_preference(obs,w).reshape(H,NENV,4)
            nv=m.value_with_preference(batch["next_obs"],torch.tensor(PREFS[pref_lab],device="cuda").repeat(NENV,1))
            A=gae(batch["rt"],vt,nv,batch["dt"],lam).reshape(-1,4).detach()
    
        actor_in=m._with_w(obs,w)
        h=m.actor_body(actor_in)
        gb=m.preference_film(w);gamma,beta=torch.chunk(gb,2,dim=-1)
        hfilm=(1+gamma)*h+beta
        emb=m.preference_embedding(w)
        feat=torch.cat((hfilm,emb),dim=-1)
        mean=m.actor_mean(feat)
        std=m.log_std.exp().expand_as(mean)
        dist=torch.distributions.Normal(mean,std)
        logp=(dist.log_prob(batch["u"])-m._log_det_jacobian(batch["u"])).sum(-1)
        ratio=torch.exp(logp-batch["old"].detach())
        clipped=ratio.clamp(.8,1.2)
    
        losses={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clipped*A[:,j])
            losses[lab]=-po.mean()
        weighted_components_loss={lab:4*float(PREFS[pref_lab][j])*losses[lab] for j,lab in enumerate(ORDER)}
        combined=sum(weighted_components_loss.values())
    
        groups={
          "all_actor":[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")],
          "shared_body":[p for n,p in m.named_parameters() if n.startswith("actor_body")],
          "direct_input":[m.actor_body[0].weight],
          "embedding":[p for n,p in m.named_parameters() if n.startswith("preference_embedding")],
          "film":[p for n,p in m.named_parameters() if n.startswith("preference_film")],
          "actor_head":[p for n,p in m.named_parameters() if n.startswith("actor_mean")],
        }
        out={"groups":{}}
        for gn,ps in groups.items():
            og={lab:flat_grads(losses[lab],ps) for lab in ORDER}
            wg={lab:4*float(PREFS[pref_lab][j])*og[lab] for j,lab in enumerate(ORDER)}
            cg=sum(wg.values())
            raw_norm={lab:float(og[lab].norm().cpu()) for lab in ORDER}
            weighted_norm={lab:float(wg[lab].norm().cpu()) for lab in ORDER}
            sum_weighted=sum(weighted_norm.values())+1e-12
            heavy=pref_lab
            others=sum((wg[x] for x in ORDER if x!=heavy),torch.zeros_like(cg))
            out["groups"][gn]={
              "raw_norm":raw_norm,
              "weighted_norm":weighted_norm,
              "weighted_norm_share":{lab:weighted_norm[lab]/sum_weighted for lab in ORDER},
              "pairwise_cosine":pairwise(og),
              "combined_norm":float(cg.norm().cpu()),
              "combined_cos_to_objective":{lab:cos(cg,og[lab]) for lab in ORDER},
              "combined_cos_to_heavy":cos(cg,og[heavy]),
              "heavy_vs_others_cos":cos(wg[heavy],others),
              "override_ratio":float(others.norm()/(wg[heavy].norm()+1e-12)),
            }
        return out
    
    def aggregate(rows):
        def rec(vals):
            v0=vals[0]
            if isinstance(v0,dict):return {k:rec([v[k] for v in vals]) for k in v0}
            if isinstance(v0,(int,float,np.floating)):
                a=np.asarray(vals,float);return {"mean":float(a.mean()),"std":float(a.std())}
            return v0
        return rec(rows)
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
    
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
    
            raw={lab:{str(l):[] for l in LAMS} for lab in ORDER}
            paired_delta={lab:[] for lab in ORDER}
            for seed in SEEDS:
                for pref in ORDER:
                    w=torch.tensor(PREFS[pref],device="cuda").repeat(NENV,1)
                    batch=collect(env,m,w,mgr,seed)
                    res={l:compute_for_lambda(m,batch,pref,l) for l in LAMS}
                    for l in LAMS: raw[pref][str(l)].append(res[l])
                    # exact paired comparisons at each seed/batch
                    d={}
                    for gn in res[0.95]["groups"]:
                        g95=res[0.95]["groups"][gn];g1=res[1.0]["groups"][gn]
                        d[gn]={
                          "combined_cos_heavy_delta":g1["combined_cos_to_heavy"]-g95["combined_cos_to_heavy"],
                          "override_ratio_delta":g1["override_ratio"]-g95["override_ratio"],
                          "weighted_share_delta":{obj:g1["weighted_norm_share"][obj]-g95["weighted_norm_share"][obj] for obj in ORDER},
                          "raw_norm_ratio_1_over_095":{obj:g1["raw_norm"][obj]/(g95["raw_norm"][obj]+1e-12) for obj in ORDER},
                          "combined_norm_ratio_1_over_095":g1["combined_norm"]/(g95["combined_norm"]+1e-12),
                          "pairwise_cos_delta":{k:g1["pairwise_cosine"][k]-g95["pairwise_cosine"][k] for k in g95["pairwise_cosine"]},
                        }
                    paired_delta[pref].append(d)
    
            agg={pref:{str(l):aggregate(raw[pref][str(l)]) for l in LAMS} for pref in ORDER}
            delta={pref:aggregate(paired_delta[pref]) for pref in ORDER}
    
            compact={}
            for pref in ORDER:
                compact[pref]={}
                for gn in ("all_actor","shared_body","direct_input","embedding","film","actor_head"):
                    compact[pref][gn]={
                      "lambda095":{
                        "combined_cos_to_heavy":agg[pref]["0.95"]["groups"][gn]["combined_cos_to_heavy"],
                        "override_ratio":agg[pref]["0.95"]["groups"][gn]["override_ratio"],
                        "weighted_norm_share":agg[pref]["0.95"]["groups"][gn]["weighted_norm_share"],
                        "raw_norm":agg[pref]["0.95"]["groups"][gn]["raw_norm"],
                        "pairwise_cosine":agg[pref]["0.95"]["groups"][gn]["pairwise_cosine"]},
                      "lambda100":{
                        "combined_cos_to_heavy":agg[pref]["1.0"]["groups"][gn]["combined_cos_to_heavy"],
                        "override_ratio":agg[pref]["1.0"]["groups"][gn]["override_ratio"],
                        "weighted_norm_share":agg[pref]["1.0"]["groups"][gn]["weighted_norm_share"],
                        "raw_norm":agg[pref]["1.0"]["groups"][gn]["raw_norm"],
                        "pairwise_cosine":agg[pref]["1.0"]["groups"][gn]["pairwise_cosine"]},
                      "paired_delta":delta[pref][gn],
                    }
    
            report={"schema":"v2b_paired_joint_update_geometry_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "checkpoint":str(a.checkpoint.relative_to(ROOT)),"lambdas":[0.95,1.0],
                    "same_batch_for_lambdas":True,"preferences":{k:v.tolist() for k,v in PREFS.items()},"seeds":list(SEEDS),
                    "aggregate":agg,"paired_delta":delta,"compact":compact}
            out=a.output_dir/"joint_update_geometry_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps(compact,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_repeated_update_composition_audit():
    """Run former v2b_repeated_update_composition_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_repeated_update_composition_audit-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    ALT_SEQ=("A","O","S","T","A","O","S","T")
    FULL_STEPS=64;ENDPOINT_SUITES=4
    BASE_SEED=980001
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def apply_flat_step(m,direction,step_norm):
        nps=actor_named_params(m);off=0
        with torch.no_grad():
            for _,p in nps:
                n=p.numel();p.add_(direction[off:off+n].view_as(p)*step_norm);off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        vals=[r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]
        return float(np.mean(vals))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w_env.repeat(H,1),"w_env":w_env}
    
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        raw={}
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            raw[lab]=-po.mean()
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return raw,weighted,ratio
    
    def update_direction(m,b,arm,update_idx):
        nps=actor_named_params(m);raw,weighted,ratio=objective_losses(m,b)
        if arm=="A_only":
            label="A";loss=weighted["A"]
        elif arm=="mixed":
            label="MIX";loss=sum(weighted.values())
        elif arm=="alternating":
            label=ALT_SEQ[update_idx-1];loss=weighted[label]
        else:raise ValueError(arm)
        g=flat_grad(loss,nps).detach()
        d=-g/(g.norm()+1e-12)
        return label,loss,float(g.norm().cpu()),d,float((ratio-1).abs().max().detach().cpu())
    
    def rollout(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(FULL_STEPS):
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();done_any|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    
    def endpoint_eval(env,m,mgr,robot):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];dobj=[];dphy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                do=r["objective_mean"][j]-c["objective_mean"][j];dp=r["physical"][pk]-c["physical"][pk]
                obj.append(do>0);phy.append(dp<0);dobj.append(do);dphy.append(dp);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,
                     "semantic_score":0.5*(of+pf),"pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),
                     "mean_objective_delta_vs_center":float(np.mean(dobj)),
                     "mean_physical_delta_vs_center":float(np.mean(dphy)),
                     "min_survival":float(min(surv))}
        return ep
    
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        vals={}
        ds=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:
                    v=float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu())
                    vals[f"{a}-{b}"]=v;ds.append(v)
        return {"pairwise":vals,"mean_pairwise":float(np.mean(ds))}
    
    def displacement_from_init(m,init_state):
        vals=[]
        for n,p in actor_named_params(m):
            vals.append((p.detach()-init_state[n].to(p.device)).reshape(-1))
        return float(torch.cat(vals).norm().cpu())
    
    def mixed_w(device,update_idx):
        labs=[ORDER[(update_idx+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init_state=copy.deepcopy(base.state_dict())
            # fixed probe from base A-heavy states
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
            pb=collect_batch(env,base,mgr,BASE_SEED,wb)
            probe=pb["obs"][:64].detach().clone()
            step_norm=STEP_SCALE*nominal_step()
    
            baseline_ep=endpoint_eval(env,base,mgr,robot)
            baseline_sep=preference_separation(base,probe)
            arms={}
            models={}
            prev_dirs={}
            for arm in ("A_only","mixed","alternating"):
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init_state);m.eval()
                models[arm]=m;prev_dirs[arm]=None;arms[arm]=[]
            # semantic retention state for forgetting accounting
            prev_scores={arm:{lab:baseline_ep[lab]["semantic_score"] for lab in ORDER} for arm in models}
    
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                for arm,m in models.items():
                    if arm=="mixed":
                        labs,w=mixed_w(torch.device("cuda"),k)
                    elif arm=="A_only":
                        labs=["A"]*NENV;w=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
                    else:
                        lab=ALT_SEQ[k-1];labs=[lab]*NENV;w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
                    batch=collect_batch(env,m,mgr,seed,w)
                    label,loss,gnorm,direction,ratioerr=update_direction(m,batch,arm,k)
                    update_cos=None if prev_dirs[arm] is None else cos(direction,prev_dirs[arm])
                    apply_flat_step(m,direction,step_norm)
                    prev_dirs[arm]=direction.clone()
                    ep=endpoint_eval(env,m,mgr,robot)
                    sep=preference_separation(m,probe)
                    scores={lab:ep[lab]["semantic_score"] for lab in ORDER}
                    forgetting={lab:prev_scores[arm][lab]-scores[lab] for lab in ORDER}
                    prev_scores[arm]=scores
                    row={"update":k,"update_label":label,"batch_labels":labs,
                         "step_norm":step_norm,"raw_grad_norm":gnorm,"ratio_maxerr":ratioerr,
                         "update_cos_prev":update_cos,
                         "parameter_displacement_from_theta0":displacement_from_init(m,init_state),
                         "preference_separation":sep,"endpoint":ep,
                         "semantic_scores":scores,"forgetting_from_previous":forgetting}
                    arms[arm].append(row)
                    ck=args.output_dir/f"{arm}_u{k}.pt"
                    torch.save({"model":m.state_dict(),"arm":arm,"update":k,"step_norm":step_norm},ck)
                # persist partial report after every shared update index
                partial={"schema":"v2b_repeated_update_composition_audit_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
                         "lambda":LAM,"updates":K,"step_scale":STEP_SCALE,"step_norm":step_norm,
                         "alternating_sequence":list(ALT_SEQ),"baseline_endpoint":baseline_ep,
                         "baseline_preference_separation":baseline_sep,"arms":arms}
                (args.output_dir/"repeated_update_composition_partial.json").write_text(json.dumps(partial,indent=2)+"\n")
    
            # Retention/forgetting matrices by causing update label.
            forgetting_summary={}
            for arm,rows in arms.items():
                by_cause={}
                for r in rows:
                    cause=r["update_label"]
                    by_cause.setdefault(cause,{lab:[] for lab in ORDER})
                    for lab in ORDER:by_cause[cause][lab].append(r["forgetting_from_previous"][lab])
                forgetting_summary[arm]={cause:{lab:float(np.mean(vals[lab])) for lab in ORDER} for cause,vals in by_cause.items()}
    
            report={"schema":"v2b_repeated_update_composition_audit_v1","measurement_only":True,
                    "base_checkpoint":str(CKPT.relative_to(ROOT)),"lambda":LAM,"updates":K,
                    "step_scale":STEP_SCALE,"step_norm":step_norm,
                    "nominal_step_norm":nominal_step(),"alternating_sequence":list(ALT_SEQ),
                    "fresh_on_policy_batch_each_update":True,"matched_reset_seed_by_update":True,
                    "baseline_endpoint":baseline_ep,"baseline_preference_separation":baseline_sep,
                    "arms":arms,"forgetting_summary":forgetting_summary}
            out=args.output_dir/"repeated_update_composition_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({
              "status":"FROZEN_BY_HASH","report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),
              "base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            # compact stdout
            compact={"baseline":{lab:(baseline_ep[lab]["semantic_score"],baseline_ep[lab]["pass"]) for lab in ORDER}}
            for arm,rows in arms.items():
                compact[arm]=[{"u":r["update"],"label":r["update_label"],
                               "scores":r["semantic_scores"],
                               "passes":{lab:r["endpoint"][lab]["pass"] for lab in ORDER},
                               "forgetting":r["forgetting_from_previous"],
                               "disp":r["parameter_displacement_from_theta0"],
                               "cos_prev":r["update_cos_prev"]} for r in rows]
            print(json.dumps(compact,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "v2b_adam_history_isolation": run_v2b_adam_history_isolation,
    "v2b_adam_m1_eachstep_isolation": run_v2b_adam_m1_eachstep_isolation,
    "v2b_adam_m1_isolation": run_v2b_adam_m1_isolation,
    "v2b_adam_moment_reset_preserve_step": run_v2b_adam_moment_reset_preserve_step,
    "v2b_adam_moments_isolation": run_v2b_adam_moments_isolation,
    "v2b_mixed_training_batch_geometry_audit": run_v2b_mixed_training_batch_geometry_audit,
    "v2b_paired_joint_update_geometry_audit": run_v2b_paired_joint_update_geometry_audit,
    "v2b_repeated_update_composition_audit": run_v2b_repeated_update_composition_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
