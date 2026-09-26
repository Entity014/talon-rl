"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_v2b0_function_preserving_gate():
    """Run former v2b0_function_preserving_gate.py stage."""
    from pathlib import Path
    import argparse,hashlib,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    OUT=ROOT/"runs/v2b0_function_preserving_gate-2026-09-23"
    WREF=np.asarray([.25,.25,.25,.25],np.float32)
    NENV=8;STEPS=64;TOL=1e-6
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def ma(x):return float(torch.max(torch.abs(x)).detach().cpu()) if x.numel() else 0.0
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open("rb") as f:
            for b in iter(lambda:f.read(1<<20),b""):h.update(b)
        return h.hexdigest()
    def state_hash(m,exclude_prefix=None):
        h=hashlib.sha256()
        for n,t in m.state_dict().items():
            if exclude_prefix and n.startswith(exclude_prefix): continue
            h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    
    def scalarization_check():
        from talon_rl.rewards.objectives import OBJECTIVE_ORDER,NORMALIZATION_DIVISORS,normalize_objectives,scalarize
        raw=np.asarray([[1.7194554805755615,-.15590913593769073,-.01563369482755661,-.08311229199171066]],np.float32)
        norm=normalize_objectives(raw);exp=np.asarray([[1.,-1.,-1.,-1.]],np.float32)
        got=float(scalarize(norm,WREF)[0]);ref=float((exp*WREF).sum())
        return {"objective_order":list(OBJECTIVE_ORDER),"normalization_divisors":NORMALIZATION_DIVISORS.tolist(),
                "normalized_probe_max_abs_error":float(np.max(np.abs(norm-exp))),
                "scalarization_abs_error":abs(got-ref),
                "pass":bool(np.max(np.abs(norm-exp))<=1e-7 and abs(got-ref)<=1e-7)}
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);ap.add_argument("--seed",type=int,default=424242)
        a=ap.parse_args()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
    
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.preference_embedding import V2AMinimalEmbeddingActorCritic
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic,initialize_from_v2a
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=a.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=a.seed);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            state=torch.load(V2A_INIT,map_location="cuda",weights_only=False)
    
            ref=V2AMinimalEmbeddingActorCritic(o.shape[-1],ad).cuda();ref.load_state_dict(state["model"]);ref.eval()
            torch.manual_seed(a.seed);np.random.seed(a.seed)
            v2b=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();initialize_from_v2a(v2b,state);v2b.eval()
            w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
    
            with torch.no_grad():
                rmean=ref.actor_mean(ref._actor_features_v2a(o,w))
                bmean=v2b.actor_mean(v2b._actor_features_v2a(o,w))
                ra=ref.act_inference_with_preference(o,w);ba=v2b.act_inference_with_preference(o,w)
                rv=ref.value_with_preference(o,w);bv=v2b.value_with_preference(o,w)
                gb=v2b.preference_film(w);gamma,beta=torch.chunk(gb,2,dim=-1)
                re=ref.preference_embedding(w);be=v2b.preference_embedding(w)
    
            checks={}
            checks["deterministic_action_identity"]={"max_abs_diff":ma(ba-ra),"pass":ma(ba-ra)<=TOL}
            checks["pre_tanh_mean_identity"]={"max_abs_diff":ma(bmean-rmean),"pass":ma(bmean-rmean)<=TOL}
            checks["critic_identity"]={"max_abs_diff":ma(bv-rv),"pass":ma(bv-rv)<=TOL}
            checks["embedding_path_identity"]={"max_abs_diff":ma(be-re),"pass":ma(be-re)<=TOL}
            checks["film_identity_init"]={
                "gamma_max_abs":ma(gamma),"beta_max_abs":ma(beta),
                "film_weight_norm":float(v2b.preference_film.weight.norm().detach().cpu()),
                "film_bias_norm":float(v2b.preference_film.bias.norm().detach().cpu()),
                "pass":ma(gamma)==0.0 and ma(beta)==0.0 and float(v2b.preference_film.weight.norm().cpu())==0.0 and float(v2b.preference_film.bias.norm().cpu())==0.0}
    
            # Same pre-tanh latent -> exact same transformed log-prob.
            torch.manual_seed(a.seed+1)
            with torch.no_grad():
                dist=ref._pre_tanh_dist_with_preference(o,w);u=dist.sample()
                rlog=ref.logp_from_pre_tanh_with_preference(o,w,u)
                blog=v2b.logp_from_pre_tanh_with_preference(o,w,u)
            checks["same_latent_logprob_identity"]={"max_abs_diff":ma(blog-rlog),"pass":ma(blog-rlog)<=TOL}
    
            # PPO invariant on B itself.
            with torch.no_grad():
                act,old,u2=v2b.act_with_preference_latent(o,w)
                new=v2b.logp_from_pre_tanh_with_preference(o,w,u2)
            ratio=torch.exp(new-old)
            checks["ppo_ratio_invariant"]={"max_abs_ratio_minus_1":ma(ratio-1),"pass":ma(ratio-1)<=1e-6}
    
            # Env applied action identity.
            _=env.reset(seed=a.seed+7);env.step(act)
            raw=env.unwrapped.action_manager._terms["joint_pos"].raw_actions.detach()
            checks["env_raw_action_identity"]={"max_abs_diff":ma(raw-act),"pass":ma(raw-act)<=TOL}
    
            # Checkpoint round trip.
            ck=a.output_dir/"v2b0_init.pt"
            torch.save({"model":v2b.state_dict(),"reference":"v2a0_init","training_steps":0,
                        "architecture":"RV1 direct + V2-A embedding + one-site final-hidden FiLM"},ck)
            v2br=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            v2br.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);v2br.eval()
            with torch.no_grad():
                ba2=v2br.act_inference_with_preference(o,w);bv2=v2br.value_with_preference(o,w)
            checks["checkpoint_roundtrip"]={
                "action_max_abs_diff":ma(ba2-ba),"value_max_abs_diff":ma(bv2-bv),
                "state_hash_equal":state_hash(v2b)==state_hash(v2br),
                "pass":ma(ba2-ba)<=TOL and ma(bv2-bv)<=TOL and state_hash(v2b)==state_hash(v2br)}
    
            checks["objective_contract"]=scalarization_check()
            checks["all_finite"]={"pass":bool(torch.isfinite(ba).all() and torch.isfinite(bv).all() and torch.isfinite(old).all())}
    
            # Paired no-update smoke vs exact V2-A reference.
            cur,_=env.reset(seed=a.seed+99);cur=ot(cur).cuda();mx=0.;terms=0;finite=True
            for _ in range(STEPS):
                with torch.no_grad():
                    ab=v2b.act_inference_with_preference(cur,w)
                    ar=ref.act_inference_with_preference(cur,w)
                mx=max(mx,ma(ab-ar));finite=finite and bool(torch.isfinite(ab).all())
                nxt,_,te,tr,_=env.step(ab);terms+=int((te|tr).sum().item());cur=ot(nxt).cuda()
            checks["no_update_smoke"]={"steps":STEPS,"num_envs":NENV,"max_action_diff_vs_v2a":mx,
                                       "termination_events":terms,"all_finite":finite,"pass":mx<=TOL and finite}
    
            # Treatment isolation: exactly preference_film is new relative to V2-A.
            ref_names=set(dict(ref.named_parameters()))
            b_names=set(dict(v2b.named_parameters()))
            new_names=sorted(b_names-ref_names)
            expected={"preference_film.weight","preference_film.bias"}
            checks["single_site_film_only"]={
                "new_parameter_names":new_names,
                "expected_new_parameter_names":sorted(expected),
                "pass":set(new_names)==expected}
    
            # Existing V2-A tensors must be exactly copied.
            exact_existing=True;max_existing=0.0
            rs=ref.state_dict();bs=v2b.state_dict()
            for k,v in rs.items():
                if k in bs and bs[k].shape==v.shape:
                    d=float((bs[k]-v).abs().max().cpu());max_existing=max(max_existing,d)
                    exact_existing=exact_existing and d==0.0
            checks["v2a_parameter_identity"]={"max_abs_diff":max_existing,"pass":exact_existing}
    
            passed=all(v.get("pass",False) for v in checks.values())
            report={"schema":"v2b0_function_preserving_gate_v1",
                    "status":"V2-B0 PASS" if passed else "V2-B0 FAIL",
                    "training_enabled":False,"optimizer_steps":0,
                    "reference_checkpoint":str(V2A_INIT.relative_to(ROOT)),
                    "reference_definition":"frozen V2-A0 function-preserving initialization",
                    "treatment":"retain RV1 direct path + V2-A embedding; add one final-hidden FiLM site with zero gamma/beta identity init",
                    "checks":checks,
                    "authorization":{"v2b1_authorized":bool(passed),"training_in_this_gate":False},
                    "checkpoint":{"path":str(ck.relative_to(ROOT)),"sha256":sha(ck)},
                    "provenance":{"v2a_init_sha256":sha(V2A_INIT),
                                  "v2b_module_sha256":sha(ROOT/"talon_rl/models/conditioning/single_site_film.py"),
                                  "gate_script_sha256":sha(Path(__file__).resolve()),
                                  "contract_sha256":sha(ROOT/"docs/contracts/preference_architectures/v2b-contract.md")}}
            out=a.output_dir/"v2b0_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(
                {"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(ck)},indent=2)+"\n")
            print(json.dumps({"status":report["status"],"authorization":report["authorization"],"checks":checks},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b1_film_authority_screen():
    """Run former v2b1_film_authority_screen.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    V2B_INIT=ROOT/"runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2b1_film_authority-2026-09-23"
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
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
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
    
            report={"schema":"v2b1_film_authority_screen_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"V2-B1 authority-only screen under frozen Foundation V2",
                    "actor_ppo_objectives_changed":False,"architecture":"v2b","anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            final=snaps[str(args.updates)];initial=snaps["0"]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            fwg=np.array([r["film_weight_grad_norm"] for r in rows]);fbg=np.array([r["film_bias_grad_norm"] for r in rows])
            crit=final["phase_critic"];sens=final["sensitivity"]
            pair_total=sens["pairwise_action_distance"]["mean"]
            pair_mask=sens["film_masked_pairwise_action_distance"]["mean"]
            jac_total=sens["jacobian_fro_mean"]
            jac_mask=sens["film_masked_jacobian_fro_mean"]
            criteria={
              "film_weight_nonzero":sens["film_weight_norm"]>1e-5,
              "film_bias_nonzero":sens["film_bias_norm"]>1e-5,
              "film_gradient_observed":float(np.max(fwg))>1e-7 and float(np.max(fbg))>1e-7,
              "film_modulation_nonzero":sens["gamma_abs_mean_center"]>1e-5 and sens["beta_abs_mean_center"]>1e-5,
              "film_preference_separation":sens["gamma_pairwise_distance"]["mean"]>1e-5 and sens["beta_pairwise_distance"]["mean"]>1e-5,
              "film_adds_action_separation":pair_total>pair_mask+1e-4,
              "film_adds_jacobian_authority":jac_total>jac_mask+1e-4,
              "film_action_authority_nonzero":sens["film_action_authority_mean"]>1e-4,
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
            report["summary"]={
              "status":"V2-B1 PASS" if passed else "V2-B1 FAIL",
              "criteria":criteria,
              "initial":initial,
              "final":final,
              "max_film_weight_grad_norm":float(np.max(fwg)),
              "median_film_weight_grad_norm":float(np.median(fwg)),
              "max_film_bias_grad_norm":float(np.max(fbg)),
              "median_film_bias_grad_norm":float(np.median(fbg)),
              "last10_termination_fraction":float(term[-10:].mean()),
              "max_ratio_error":float(ratio.max()),
              "v2b2_authorized":bool(passed)}
            (args.output_dir/"v2b1_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_absolute_A_center_check():
    """Run former v2b_absolute_A_center_check.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    NENV=8;STEPS=64
    PREFS={"A":np.array([.1,.7,.1,.1],np.float32),"C":np.array([.25,.25,.25,.25],np.float32)}
    CASES={
    "base":ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/A_only_scale_0p0.pt",
    "A_0p5":ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/A_only_scale_0p5.pt",
    "A_1p0":ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/A_only_scale_1p0.pt",
    "A_2p0":ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/A_only_scale_2p0.pt",
    "C_0p25":ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/combined_scale_0p25.pt",
    "C_0p5":ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/combined_scale_0p5.pt",
    "C_1p0":ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/combined_scale_1p0.pt",
    "C_2p0":ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/combined_scale_2p0.pt",
    }
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def run(env,m,robot,w_np,seed):
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        vals=[];surv=np.ones(NENV,bool)
        with torch.no_grad():
            for _ in range(STEPS):
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                vals.append(float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[:,:2],dim=-1).mean()))
                surv &= ~(te|tr).cpu().numpy()
                cur=ot(nxt).cuda()
        return {"ang_vel_xy":float(np.mean(vals)),"survival":float(surv.mean())}
    def main():
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
          o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;robot=env.unwrapped.scene["robot"]
          out={}
          for name,ck in CASES.items():
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
            out[name]={"A":[],"C":[]}
            for suite in range(4):
              seed=840001+suite
              for lab in ("A","C"):out[name][lab].append(run(env,m,robot,PREFS[lab],seed))
            out[name]={"A_ang":float(np.mean([x["ang_vel_xy"] for x in out[name]["A"]])),
                       "C_ang":float(np.mean([x["ang_vel_xy"] for x in out[name]["C"]])),
                       "delta_A_minus_C":float(np.mean([x["ang_vel_xy"] for x in out[name]["A"]])-np.mean([x["ang_vel_xy"] for x in out[name]["C"]])),
                       "survival_min":float(min([x["survival"] for x in out[name]["A"]+out[name]["C"]]))}
          path=ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24/absolute_A_center_check.json"
          path.write_text(json.dumps(out,indent=2)+"\n")
          print(json.dumps(out,indent=2))
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_v2b_action_trust_region_gate():
    """Run former v2b_action_trust_region_gate.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_action_trust_region_gate-2026-09-24"
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
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    ANCHOR_SEEDS={"T":995001,"A":996001,"O":997001,"S":998001}
    ANCHOR_STATES=64
    TRUST_FRAC=0.25
    CORRECT_ITERS=8
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flatten_params(m):
        return torch.cat([p.detach().reshape(-1) for _,p in actor_named_params(m)])
    def set_flat_params(m,vec):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.copy_(vec[off:off+n].view_as(p));off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=False,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
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
    
    def collect_anchor_states(env,m,seed,lab):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        states=[]
        with torch.no_grad():
            for _ in range(max(ANCHOR_STATES//NENV,1)):
                states.append(cur.detach().clone())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        s=torch.cat(states,0)[:ANCHOR_STATES]
        wfull=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1)
        with torch.no_grad():
            aref=m.act_inference_with_preference(s,wfull).detach().clone()
            allacts={}
            for other in ORDER:
                wo=torch.tensor(PREFS[other],device="cuda").repeat(len(s),1)
                allacts[other]=m.act_inference_with_preference(s,wo)
            seps=[]
            for i,a in enumerate(ORDER):
                for j,b in enumerate(ORDER):
                    if j>i:
                        seps.append(float(torch.linalg.vector_norm(allacts[a]-allacts[b],dim=-1).mean().cpu()))
        semantic_scale=float(np.mean(seps))
        radius=TRUST_FRAC*max(semantic_scale,1e-4)
        return {"states":s,"pref":wfull,"actions":aref,
                "semantic_action_separation_scale":semantic_scale,
                "radius":radius}
    
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
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    
    def mixed_w(device,update_idx):
        labs=[ORDER[(update_idx+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def mixed_direction(m,b):
        nps=actor_named_params(m);weighted,ratio=objective_losses(m,b)
        loss=sum(weighted.values());g=flat_grad(loss,nps).detach()
        return -g/(g.norm()+1e-12),float(g.norm().cpu()),float((ratio-1).abs().max().detach().cpu())
    
    def anchor_drift(m,anchors):
        out={}
        with torch.no_grad():
            for lab,a in anchors.items():
                cur=m.act_inference_with_preference(a["states"],a["pref"])
                drift=torch.linalg.vector_norm(cur-a["actions"],dim=-1)
                out[lab]={"mean":float(drift.mean().cpu()),"max":float(drift.max().cpu()),
                          "radius":a["radius"],"normalized_mean":float(drift.mean().cpu())/(a["radius"]+1e-12)}
        return out
    
    def anchor_loss(m,anchors):
        if not anchors:return torch.tensor(0.,device="cuda")
        terms_=[]
        for lab,a in anchors.items():
            cur=m.act_inference_with_preference(a["states"],a["pref"])
            r=a["radius"]
            terms_.append(((cur-a["actions"])**2).mean()/(r*r+1e-12))
        return sum(terms_)/len(terms_)
    
    def finite_action_trust_step(m,raw_d,step_norm,anchors):
        theta0=flatten_params(m)
        raw_theta=theta0+step_norm*raw_d
        set_flat_params(m,raw_theta)
        raw_drift=anchor_drift(m,anchors)
        if not anchors or all(v["mean"]<=v["radius"] for v in raw_drift.values()):
            return raw_d,{"corrected":False,"raw_drift":raw_drift,"final_drift":raw_drift,
                          "raw_to_applied_cosine":1.0,"iterations":0}
        # Direct finite correction on actual deterministic reference-action error.
        x=raw_theta.clone()
        for it in range(CORRECT_ITERS):
            set_flat_params(m,x)
            dr=anchor_drift(m,anchors)
            if all(v["mean"]<=v["radius"] for v in dr.values()):break
            loss=anchor_loss(m,anchors)
            g=flat_grad(loss,actor_named_params(m)).detach()
            if float(g.norm())<1e-12:break
            # correction sized relative to trust violation but capped to 0.5 nominal step
            worst=max(v["normalized_mean"] for v in dr.values())
            corr=min(0.5*step_norm, step_norm*0.25*max(worst-1.0,0.1))
            x=x-corr*g/(g.norm()+1e-12)
            # enforce exact total displacement norm from theta0
            disp=x-theta0
            if float(disp.norm())>1e-12:x=theta0+step_norm*disp/disp.norm()
        set_flat_params(m,x)
        applied=(x-theta0)/(step_norm+1e-12)
        final_drift=anchor_drift(m,anchors)
        return applied,{"corrected":True,"raw_drift":raw_drift,"final_drift":final_drift,
                        "raw_to_applied_cosine":cos(raw_d,applied),"iterations":it+1}
    
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
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,
                     "semantic_score":0.5*(of+pf),"pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),
                     "min_survival":float(min(surv))}
        return ep
    
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),"final_score":float(scores[-1]),
                      "first_pass_checkpoint":fp,"retained_pass_fraction_after_first_pass":ret}
        return out
    
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
            init=copy.deepcopy(base.state_dict());step_norm=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
            probe=collect_batch(env,base,mgr,BASE_SEED,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot)
            models={};rows={};anchors={};activation={}
            for arm in ("control","action_trust"):
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval();models[arm]=m;rows[arm]=[]
            # baseline retained anchors from PASS axes
            for lab in ORDER:
                activation[lab]=None
                if base_ep[lab]["pass"]:
                    anchors[lab]=collect_anchor_states(env,models["action_trust"],ANCHOR_SEEDS[lab],lab);activation[lab]=0
            prev_dirs={a:None for a in models}
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                for arm,m in models.items():
                    labs,w=mixed_w(torch.device("cuda"),k)
                    b=collect_batch(env,m,mgr,seed,w)
                    raw_d,gnorm,ratioerr=mixed_direction(m,b)
                    trust={"corrected":False,"raw_drift":{},"final_drift":{},"raw_to_applied_cosine":1.0,"iterations":0}
                    active_before=sorted(anchors) if arm=="action_trust" else []
                    if arm=="action_trust":
                        # finite_action_trust_step commits parameters itself
                        applied_d,trust=finite_action_trust_step(m,raw_d,step_norm,anchors)
                    else:
                        theta=flatten_params(m);set_flat_params(m,theta+step_norm*raw_d);applied_d=raw_d
                    update_cos=None if prev_dirs[arm] is None else cos(applied_d,prev_dirs[arm]);prev_dirs[arm]=applied_d.detach().clone()
                    ep=endpoint_eval(env,m,mgr,robot)
                    if arm=="action_trust":
                        for lab in ORDER:
                            if ep[lab]["pass"] and lab not in anchors:
                                anchors[lab]=collect_anchor_states(env,m,ANCHOR_SEEDS[lab],lab);activation[lab]=k
                    row={"update":k,"active_anchors_before_update":active_before,"raw_grad_norm":gnorm,"ratio_maxerr":ratioerr,
                         "trust":trust,"update_cos_prev":update_cos,
                         "parameter_displacement":float(torch.linalg.vector_norm(flatten_params(m)-torch.cat([init[n].reshape(-1).cuda() for n,_ in actor_named_params(m)] )).cpu()),
                         "preference_separation":preference_separation(m,probe),
                         "endpoint":ep,"semantic_scores":{lab:ep[lab]["semantic_score"] for lab in ORDER}}
                    rows[arm].append(row)
                    torch.save({"model":m.state_dict(),"arm":arm,"update":k},args.output_dir/f"{arm}_u{k}.pt")
                (args.output_dir/"action_trust_partial.json").write_text(json.dumps({"activation":activation,"active":sorted(anchors),"arms":rows},indent=2)+"\n")
            report={"schema":"v2b_action_trust_region_gate_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "trust_frac_of_semantic_action_separation":TRUST_FRAC,"step_norm":step_norm,"updates":K,
                    "baseline_endpoint":base_ep,"activation":activation,"arms":rows,
                    "retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            out=args.output_dir/"action_trust_region_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"],"last":{a:r[-1]["semantic_scores"] for a,r in rows.items()}},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_action_trust_region_gate_v3():
    """Run former v2b_action_trust_region_gate_v3.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_action_trust_region_gate_v3-2026-09-24"
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
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    ANCHOR_SEEDS={"T":995001,"A":996001,"O":997001,"S":998001}
    ANCHOR_STATES=64
    TRUST_FRAC=0.25
    CORRECT_ITERS=16
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flatten_params(m):
        return torch.cat([p.detach().reshape(-1) for _,p in actor_named_params(m)])
    def set_flat_params(m,vec):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.copy_(vec[off:off+n].view_as(p));off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=False,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
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
    
    def collect_anchor_states(env,m,seed,lab):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        states=[]
        with torch.no_grad():
            for _ in range(max(ANCHOR_STATES//NENV,1)):
                states.append(cur.detach().clone())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        s=torch.cat(states,0)[:ANCHOR_STATES]
        wfull=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1)
        with torch.no_grad():
            aref=m.act_inference_with_preference(s,wfull).detach().clone()
            allacts={}
            for other in ORDER:
                wo=torch.tensor(PREFS[other],device="cuda").repeat(len(s),1)
                allacts[other]=m.act_inference_with_preference(s,wo)
            seps=[]
            for i,a in enumerate(ORDER):
                for j,b in enumerate(ORDER):
                    if j>i:
                        seps.append(float(torch.linalg.vector_norm(allacts[a]-allacts[b],dim=-1).mean().cpu()))
        semantic_scale=float(np.mean(seps))
        radius=TRUST_FRAC*max(semantic_scale,1e-4)
        return {"states":s,"pref":wfull,"actions":aref,
                "semantic_action_separation_scale":semantic_scale,
                "radius":radius}
    
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
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    
    def mixed_w(device,update_idx):
        labs=[ORDER[(update_idx+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def mixed_direction(m,b):
        nps=actor_named_params(m);weighted,ratio=objective_losses(m,b)
        loss=sum(weighted.values());g=flat_grad(loss,nps).detach()
        return -g/(g.norm()+1e-12),float(g.norm().cpu()),float((ratio-1).abs().max().detach().cpu())
    
    def anchor_drift(m,anchors):
        out={}
        with torch.no_grad():
            for lab,a in anchors.items():
                cur=m.act_inference_with_preference(a["states"],a["pref"])
                drift=torch.linalg.vector_norm(cur-a["actions"],dim=-1)
                out[lab]={"mean":float(drift.mean().cpu()),"max":float(drift.max().cpu()),
                          "radius":a["radius"],"normalized_mean":float(drift.mean().cpu())/(a["radius"]+1e-12)}
        return out
    
    def anchor_loss(m,anchors):
        if not anchors:return torch.tensor(0.,device="cuda")
        terms_=[]
        for lab,a in anchors.items():
            cur=m.act_inference_with_preference(a["states"],a["pref"])
            r=a["radius"]
            terms_.append(((cur-a["actions"])**2).mean()/(r*r+1e-12))
        return sum(terms_)/len(terms_)
    
    def finite_action_trust_step(m,raw_d,step_norm,anchors):
        theta0=flatten_params(m)
        raw_theta=theta0+step_norm*raw_d
        set_flat_params(m,raw_theta)
        raw_drift=anchor_drift(m,anchors)
        if not anchors or all(v["mean"]<=v["radius"] for v in raw_drift.values()):
            return raw_d,step_norm,{"corrected":False,"backtracked":False,"raw_drift":raw_drift,
                          "final_drift":raw_drift,"raw_to_applied_cosine":1.0,
                          "iterations":0,"step_ratio":1.0}
        # First correct direction at full nominal step.
        x=raw_theta.clone()
        it=0
        for it in range(CORRECT_ITERS):
            set_flat_params(m,x)
            dr=anchor_drift(m,anchors)
            if all(v["mean"]<=v["radius"] for v in dr.values()):break
            loss=anchor_loss(m,anchors)
            g=flat_grad(loss,actor_named_params(m)).detach()
            if float(g.norm())<1e-12:break
            worst=max(v["normalized_mean"] for v in dr.values())
            corr=min(0.5*step_norm, step_norm*0.25*max(worst-1.0,0.1))
            x=x-corr*g/(g.norm()+1e-12)
            disp=x-theta0
            if float(disp.norm())>1e-12:x=theta0+step_norm*disp/disp.norm()
        # If full-norm feasible point was not found, backtrack the corrected displacement.
        disp=x-theta0
        if float(disp.norm())<1e-12:disp=raw_d*step_norm
        direction=disp/(disp.norm()+1e-12)
        actual=step_norm
        backtracked=False
        set_flat_params(m,theta0+actual*direction)
        final_drift=anchor_drift(m,anchors)
        while anchors and not all(v["mean"]<=v["radius"] for v in final_drift.values()) and actual>step_norm/128:
            actual*=0.5
            backtracked=True
            set_flat_params(m,theta0+actual*direction)
            final_drift=anchor_drift(m,anchors)
        applied=direction
        return applied,actual,{"corrected":True,"backtracked":backtracked,
                        "raw_drift":raw_drift,"final_drift":final_drift,
                        "raw_to_applied_cosine":cos(raw_d,applied),"iterations":it+1,
                        "step_ratio":actual/(step_norm+1e-12)}
    
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
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,
                     "semantic_score":0.5*(of+pf),"pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),
                     "min_survival":float(min(surv))}
        return ep
    
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),"final_score":float(scores[-1]),
                      "first_pass_checkpoint":fp,"retained_pass_fraction_after_first_pass":ret}
        return out
    
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
            init=copy.deepcopy(base.state_dict());step_norm=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
            probe=collect_batch(env,base,mgr,BASE_SEED,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot)
            models={};rows={};anchors={};activation={}
            for arm in ("control","action_trust"):
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval();models[arm]=m;rows[arm]=[]
            for lab in ORDER:
                activation[lab]=None
                if base_ep[lab]["pass"]:
                    anchors[lab]=collect_anchor_states(env,models["action_trust"],ANCHOR_SEEDS[lab],lab);activation[lab]=0
            prev_dirs={a:None for a in models}
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                # Treatment first determines the feasible trust-region step norm.
                mt=models["action_trust"]
                labs_t,w_t=mixed_w(torch.device("cuda"),k)
                bt=collect_batch(env,mt,mgr,seed,w_t)
                raw_t,gn_t,ratio_t=mixed_direction(mt,bt)
                active_before=sorted(anchors)
                applied_t,actual_step,trust=finite_action_trust_step(mt,raw_t,step_norm,anchors)
                cosprev_t=None if prev_dirs["action_trust"] is None else cos(applied_t,prev_dirs["action_trust"])
                prev_dirs["action_trust"]=applied_t.detach().clone()
                ep_t=endpoint_eval(env,mt,mgr,robot)
                for lab in ORDER:
                    if ep_t[lab]["pass"] and lab not in anchors:
                        anchors[lab]=collect_anchor_states(env,mt,ANCHOR_SEEDS[lab],lab);activation[lab]=k
                rows["action_trust"].append({"update":k,"active_anchors_before_update":active_before,
                    "raw_grad_norm":gn_t,"ratio_maxerr":ratio_t,"trust":trust,
                    "effective_step_norm":actual_step,"update_cos_prev":cosprev_t,
                    "parameter_displacement":float(torch.linalg.vector_norm(flatten_params(mt)-torch.cat([init[n].reshape(-1).cuda() for n,_ in actor_named_params(mt)])).cpu()),
                    "preference_separation":preference_separation(mt,probe),"endpoint":ep_t,
                    "semantic_scores":{lab:ep_t[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mt.state_dict(),"arm":"action_trust","update":k,"effective_step_norm":actual_step},args.output_dir/f"action_trust_u{k}.pt")
    
                # Magnitude-matched control: same fresh mixed protocol and same effective step norm, no anchor correction.
                mc=models["control"]
                labs_c,w_c=mixed_w(torch.device("cuda"),k)
                bc=collect_batch(env,mc,mgr,seed,w_c)
                raw_c,gn_c,ratio_c=mixed_direction(mc,bc)
                theta_c=flatten_params(mc);set_flat_params(mc,theta_c+actual_step*raw_c)
                cosprev_c=None if prev_dirs["control"] is None else cos(raw_c,prev_dirs["control"])
                prev_dirs["control"]=raw_c.detach().clone()
                ep_c=endpoint_eval(env,mc,mgr,robot)
                rows["control"].append({"update":k,"active_anchors_before_update":[],
                    "raw_grad_norm":gn_c,"ratio_maxerr":ratio_c,
                    "trust":{"corrected":False,"backtracked":False,"raw_drift":{},"final_drift":{},
                             "raw_to_applied_cosine":1.0,"iterations":0,"step_ratio":actual_step/(step_norm+1e-12)},
                    "effective_step_norm":actual_step,"update_cos_prev":cosprev_c,
                    "parameter_displacement":float(torch.linalg.vector_norm(flatten_params(mc)-torch.cat([init[n].reshape(-1).cuda() for n,_ in actor_named_params(mc)])).cpu()),
                    "preference_separation":preference_separation(mc,probe),"endpoint":ep_c,
                    "semantic_scores":{lab:ep_c[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mc.state_dict(),"arm":"control","update":k,"effective_step_norm":actual_step},args.output_dir/f"control_u{k}.pt")
                (args.output_dir/"action_trust_partial.json").write_text(json.dumps({"activation":activation,"active":sorted(anchors),"arms":rows},indent=2)+"\n")
            report={"schema":"v2b_action_trust_region_gate_v3","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "trust_frac_of_semantic_action_separation":TRUST_FRAC,"nominal_step_norm":step_norm,"updates":K,
                    "baseline_endpoint":base_ep,"activation":activation,"arms":rows,
                    "retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            out=args.output_dir/"action_trust_region_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"],"last":{a:r[-1]["semantic_scores"] for a,r in rows.items()}},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "v2b0_function_preserving_gate": run_v2b0_function_preserving_gate,
    "v2b1_film_authority_screen": run_v2b1_film_authority_screen,
    "v2b_absolute_A_center_check": run_v2b_absolute_A_center_check,
    "v2b_action_trust_region_gate": run_v2b_action_trust_region_gate,
    "v2b_action_trust_region_gate_v3": run_v2b_action_trust_region_gate_v3,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
