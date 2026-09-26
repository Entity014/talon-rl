#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,json,sys
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
AI_INIT=ROOT/"runs/authority_isolated_h0-2026-09-25/authority_isolated_h0_init.pt"
OUT_DEFAULT=ROOT/"runs/authority_isolated_h1-2026-09-25"
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
def _pairwise_tensor_distance(xs):
    labs=list(xs);vals=[]
    for i,a in enumerate(labs):
        for b in labs[i+1:]:
            vals.append(float(torch.linalg.vector_norm((xs[a]-xs[b]).reshape(len(xs[a]),-1),dim=1).mean().detach().cpu()))
    return {"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals))}

def _svd_summary(X):
    X=np.asarray(X,np.float64)
    X=X-X.mean(0,keepdims=True)
    s=np.linalg.svd(X,compute_uv=False)
    s=s.tolist()
    ratio=float(s[1]/(s[0]+1e-12)) if len(s)>1 else 0.0
    eff=int(sum(v >= (s[0]*0.05 if s and s[0]>0 else np.inf) for v in s))
    return {"singular_values":[float(v) for v in s],"s2_over_s1":ratio,"effective_rank_5pct":eff}

def _cosine_matrix(flat):
    X=np.asarray(flat,np.float64)
    N=np.linalg.norm(X,axis=1,keepdims=True)+1e-12
    C=(X@X.T)/(N@N.T)
    return C

def sensitivity(m,probe):
    acts={};coeffs={};params={}
    with torch.no_grad():
        for lab in ORDER:
            w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
            acts[lab]=m.act_inference_with_preference(probe,w)
            coeffs[lab]=m.family_coefficients(w)
            params[lab]=m.generated_parameter_vector(w)
    pair=pairwise_dist(acts)
    cmean={k:coeffs[k].mean(0).cpu().numpy() for k in ORDER}
    pmean={k:params[k].mean(0).cpu().numpy() for k in ORDER}
    heavy=("T","A","O","S")
    cd=[];pd=[]
    for i,a in enumerate(heavy):
        for b in heavy[i+1:]:
            cd.append(float(np.linalg.norm(cmean[a]-cmean[b])))
            pd.append(float(np.linalg.norm(pmean[a]-pmean[b])))
    P=np.stack([pmean[k] for k in ORDER]);Pbar=P.mean(0);Pc=P-Pbar
    ps=np.linalg.svd(Pc,compute_uv=False);p_eff=int(np.sum(ps >= (ps[0]*.05 if ps[0]>0 else np.inf)))
    p_s2=float(ps[1]/(ps[0]+1e-12)) if len(ps)>1 else 0.0
    p_single=float(ps[0]**2/(np.sum(ps**2)+1e-12));ec=float(np.linalg.norm(Pbar));es=float(np.sqrt(np.mean(np.sum(Pc**2,axis=1))));pfrac=es/(ec+es+1e-12)
    # Centered functional preference geometry; common state-only action cancels.
    A=np.stack([acts[k].cpu().numpy() for k in ORDER]);Ac=A-A.mean(0,keepdims=True);Af=Ac.reshape(len(ORDER),-1)
    fs=np.linalg.svd(Af,compute_uv=False);f_eff=int(np.sum(fs >= (fs[0]*.05 if fs[0]>0 else np.inf)));f_s2=float(fs[1]/(fs[0]+1e-12)) if len(fs)>1 else 0.0
    common=A.mean(0);fc=float(np.sqrt(np.mean(np.sum(common**2,axis=1))));fsp=float(np.sqrt(np.mean(np.sum(Ac**2,axis=2))));ffrac=fsp/(fc+fsp+1e-12)
    # Simplex-tangent action Jacobian at center.
    from torch.func import jacrev,vmap
    wc=torch.tensor(PREFS["C"],device="cuda")
    D=torch.tensor([[1.,-1,0,0],[1.,0,-1,0],[1.,0,0,-1]],device="cuda").T;Q,_=torch.linalg.qr(D,mode="reduced")
    def afun(o,w):return m.act_inference_with_preference(o.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
    J=vmap(jacrev(afun,argnums=1),in_dims=(0,None))(probe,wc)@Q
    jnorm=float(torch.linalg.matrix_norm(J,ord="fro",dim=(1,2)).mean().detach().cpu())
    def cf(x):return m.family_coefficients(x.unsqueeze(0)).squeeze(0)
    def pf(x):return m.generated_parameter_vector(x.unsqueeze(0)).squeeze(0)
    Jc=torch.autograd.functional.jacobian(cf,wc,create_graph=False)@Q
    Jp=torch.autograd.functional.jacobian(pf,wc,create_graph=False)@Q
    return {"pairwise_action_distance":pair,"tangent_jacobian_fro_mean":jnorm,
      "coefficient_pairwise_distance_heavy":{"mean":float(np.mean(cd)),"min":float(np.min(cd)),"max":float(np.max(cd))},
      "generated_parameter_pairwise_distance_heavy":{"mean":float(np.mean(pd)),"min":float(np.min(pd)),"max":float(np.max(pd))},
      "centered_parameter_geometry":{"s2_over_s1":p_s2,"effective_rank_5pct":p_eff,"largest_energy_fraction":p_single,"common_energy":ec,"specific_energy":es,"specific_fraction":pfrac,"singular_values":ps.tolist()},
      "centered_functional_geometry":{"s2_over_s1":f_s2,"effective_rank_5pct":f_eff,"common_rms":fc,"specific_rms":fsp,"specific_fraction":ffrac,"specific_over_common":fsp/(fc+1e-12),"singular_values":fs.tolist()},
      "coefficient_tangent_jacobian_fro":float(torch.linalg.matrix_norm(Jc,ord="fro").detach().cpu()),
      "parameter_tangent_jacobian_fro":float(torch.linalg.matrix_norm(Jp,ord="fro").detach().cpu())}

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
    ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001)
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
        from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
        ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
        m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
        init_path=AI_INIT
        m.load_state_dict(torch.load(init_path,map_location="cuda",weights_only=False)["model"]);m.train()
        for n,p in m.named_parameters():
            if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
        actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
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
                adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
            ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
            ratio_err=float((ratio-1).abs().max().detach().cpu())
            if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
            loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
            opt.zero_grad(set_to_none=True);loss.backward()
            pg=0.0
            outw=m.family_hyper[2].weight.grad
            outb=m.family_hyper[2].bias.grad
            hgrad=float(torch.sqrt((outw.detach()**2).sum()+(outb.detach()**2).sum()).cpu()) if outw is not None and outb is not None else 0.0
            basis_grads=[p.grad.norm() for n,p in m.named_parameters() if n.startswith("family_B_") and p.grad is not None]
            bgrad=float(torch.stack(basis_grads).norm().detach().cpu()) if basis_grads else 0.0
            total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
            with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
            post_refresh=None
            # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
            if uidx in SNAPS:
                post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
            rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                         "preference_input_grad_norm":pg,"family_output_grad_norm":hgrad,"family_basis_grad_norm":bgrad,"actor_grad_norm_preclip":total,
                         "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
            if uidx in SNAPS:audit(uidx)

        final=snaps[str(args.updates)];initial=snaps["0"]
        term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
        hg=np.array([r["family_output_grad_norm"] for r in rows]);bg=np.array([r["family_basis_grad_norm"] for r in rows])
        crit=final["phase_critic"];sens=final["sensitivity"]
        prior=[snaps[str(k)]["sensitivity"] for k in (10,25,50)]
        pair0=initial["sensitivity"]["pairwise_action_distance"]["mean"];pair75=sens["pairwise_action_distance"]["mean"];pairpeak=max(x["pairwise_action_distance"]["mean"] for x in prior)
        j0=initial["sensitivity"]["tangent_jacobian_fro_mean"];j75=sens["tangent_jacobian_fro_mean"];jpeak=max(x["tangent_jacobian_fro_mean"] for x in prior)
        pair_peak_ret=pair75/(pairpeak+1e-12);pair_init_ret=pair75/(pair0+1e-12);j_peak_ret=j75/(jpeak+1e-12);j_init_ret=j75/(j0+1e-12)
        pgm=sens["centered_parameter_geometry"];fg=sens["centered_functional_geometry"]
        criteria={
          "pairwise_authority_peak_retention":pair_peak_ret>=.75,
          "tangent_jacobian_peak_retention":j_peak_ret>=.75,
          "pairwise_authority_initial_retention":pair_init_ret>=.75,
          "tangent_jacobian_initial_retention":j_init_ret>=.75,
          "centered_parameter_geometry":pgm["s2_over_s1"]>=.10 and pgm["effective_rank_5pct"]>=2 and pgm["specific_fraction"]>=.15,
          "centered_functional_geometry":fg["effective_rank_5pct"]>=2 and fg["specific_fraction"]>=.15,
          "coefficient_tangent_jacobian_nonzero":sens["coefficient_tangent_jacobian_fro"]>1e-3,
          "parameter_tangent_jacobian_nonzero":sens["parameter_tangent_jacobian_fro"]>1e-3,
          "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
          "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
          "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
          "ppo_ratio_invariant":float(ratio.max())<=1e-4,
          "survival_preserved":float(term[-10:].mean())<.5,
          "family_output_gradient_observed":float(np.max(hg))>1e-7,
          "family_basis_gradient_observed":float(np.max(bg))>1e-7}
        foundation_keys={"early_critic_preserved","late_critic_preserved","combined_negative_fraction","ppo_ratio_invariant","survival_preserved"}
        authority_keys={"pairwise_authority_peak_retention","tangent_jacobian_peak_retention","pairwise_authority_initial_retention","tangent_jacobian_initial_retention"}
        geom_keys={"centered_parameter_geometry","centered_functional_geometry","coefficient_tangent_jacobian_nonzero","parameter_tangent_jacobian_nonzero"}
        passed=all(criteria.values())
        if passed:status="AI-H1 PASS"
        elif any(not criteria[k] for k in foundation_keys):status="AI-H1 FAIL — FOUNDATION"
        elif any(not criteria[k] for k in authority_keys):status="AI-H1 FAIL — AUTHORITY DRIFT"
        else:status="AI-H1 FAIL — FAMILY DEGENERATION"
        report={"schema":"authority_isolated_h1_v1","seed":args.seed,"updates":args.updates,"training_scope":"AI-H1 sole-path authority stability; semantics blocked","actor_ppo_objectives_changed":False,"architecture":"authority_isolated","reference_checkpoint":str(AI_INIT.relative_to(ROOT)),"anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
        report["summary"]={"status":status,"criteria":criteria,"initial":initial,"final":final,"authority_retention":{"pair_peak":pairpeak,"pair_u0":pair0,"pair_u75":pair75,"pair_peak_retention":pair_peak_ret,"pair_initial_retention":pair_init_ret,"jac_peak":jpeak,"jac_u0":j0,"jac_u75":j75,"jac_peak_retention":j_peak_ret,"jac_initial_retention":j_init_ret},"max_family_output_grad_norm":float(np.max(hg)),"max_family_basis_grad_norm":float(np.max(bg)),"last10_termination_fraction":float(term[-10:].mean()),"max_ratio_error":float(ratio.max()),"h2_authorized":bool(passed),"semantic_judgement_performed":False}
        (args.output_dir/"authority_isolated_h1_report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report["summary"],indent=2))
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
