#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,json,sys
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
V2B_INIT=ROOT/"runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
OUT_DEFAULT=ROOT/"runs/v2b_lambda_training_pilot"
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
    ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--gae-lambda",type=float,choices=[0.95,1.0],required=True)
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
            total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
            with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
            post_refresh=None
            # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
            if uidx in SNAPS:
                post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
            rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                         "preference_input_grad_norm":pg,"film_weight_grad_norm":fwgrad,"film_bias_grad_norm":fbgrad,"actor_grad_norm_preclip":total,
                         "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
            if uidx in SNAPS:audit(uidx)

        report={"schema":"v2b_lambda_training_pilot_v1","seed":args.seed,"updates":args.updates,
                "training_scope":"paired lambda-only temporal-credit training pilot under frozen V2-B + Foundation V2",
                "actor_ppo_objectives_changed":False,"architecture":"v2b","gae_lambda":args.gae_lambda,"anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
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
        (args.output_dir/"lambda_pilot_report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report["summary"],indent=2))
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
