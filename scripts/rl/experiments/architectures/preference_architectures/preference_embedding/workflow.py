"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_v2a0_function_preserving_gate():
    """Run former v2a0_function_preserving_gate.py stage."""
    from pathlib import Path
    import argparse,hashlib,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    OUT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23"
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
    def state_hash(m):
        h=hashlib.sha256()
        for n,t in m.state_dict().items():
            h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    def ref_mean(m,obs,w):
        return m.actor_mean(m.actor_body(m._with_w(obs,w)))
    def v2_mean(m,obs,w):
        return m.actor_mean(m._actor_features_v2a(obs,w))
    
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
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            from talon_rl.models.foundations.preference_embedding import V2AMinimalEmbeddingActorCritic,initialize_from_rv1
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=a.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=a.seed);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            state=torch.load(RV1_INIT,map_location="cuda",weights_only=False)
    
            ref=T4SharedActorCritic(o.shape[-1],ad).cuda();ref.load_state_dict(state["model"]);ref.eval()
            torch.manual_seed(a.seed);np.random.seed(a.seed)
            v2=V2AMinimalEmbeddingActorCritic(o.shape[-1],ad).cuda();initialize_from_rv1(v2,state);v2.eval()
            w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
    
            with torch.no_grad():
                rmean=ref_mean(ref,o,w);vmean=v2_mean(v2,o,w)
                ra=torch.tanh(rmean)*ref.ACTION_CLIP
                va=v2.act_inference_with_preference(o,w)
                rv=ref.value_with_preference(o,w);vv=v2.value_with_preference(o,w)
                emb=v2.preference_embedding(w)
                embed_cols=v2.actor_mean.weight[:,-v2.EMBED_DIM:]
            checks={}
            checks["deterministic_action_identity"]={"max_abs_diff":ma(va-ra),"pass":ma(va-ra)<=TOL}
            checks["pre_tanh_mean_identity"]={"max_abs_diff":ma(vmean-rmean),"pass":ma(vmean-rmean)<=TOL}
            checks["critic_identity"]={"max_abs_diff":ma(vv-rv),"pass":ma(vv-rv)<=TOL}
            checks["embedding_readout_zero"]={"readout_l2_norm":float(embed_cols.norm().cpu()),
                                             "embedding_feature_l2_norm":float(emb.norm().cpu()),
                                             "pass":float(embed_cols.norm().cpu())==0.0}
    
            # Same latent u => same action/log-prob at init.
            torch.manual_seed(a.seed+1)
            with torch.no_grad():
                dist=ref._pre_tanh_dist(ref._with_w(o,w));u=dist.sample()
                rlog=ref.logp_from_pre_tanh_with_preference(o,w,u)
                vlog=v2.logp_from_pre_tanh_with_preference(o,w,u)
                ua=torch.tanh(u)*ref.ACTION_CLIP
            checks["same_latent_logprob_identity"]={"max_abs_diff":ma(vlog-rlog),"pass":ma(vlog-rlog)<=TOL}
    
            # PPO invariant on V2 itself.
            with torch.no_grad():
                act,old,u2=v2.act_with_preference_latent(o,w)
                new=v2.logp_from_pre_tanh_with_preference(o,w,u2)
            ratio=torch.exp(new-old)
            checks["ppo_ratio_invariant"]={"max_abs_ratio_minus_1":ma(ratio-1),"pass":ma(ratio-1)<=1e-6}
    
            # Environment action identity.
            _=env.reset(seed=a.seed+7)
            env.step(act)
            term=env.unwrapped.action_manager._terms["joint_pos"]
            raw=term.raw_actions.detach()
            checks["env_raw_action_identity"]={"max_abs_diff":ma(raw-act),"pass":ma(raw-act)<=TOL}
    
            # Checkpoint round-trip.
            before=state_hash(v2);ck=a.output_dir/"v2a0_init.pt"
            torch.save({"model":v2.state_dict(),"reference":"rv1_a_init","training_steps":0,
                        "architecture":"RV1 direct conditioning + 16D preference embedding concatenated before actor head"},ck)
            v2r=V2AMinimalEmbeddingActorCritic(o.shape[-1],ad).cuda()
            v2r.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);v2r.eval()
            with torch.no_grad():
                va2=v2r.act_inference_with_preference(o,w);vv2=v2r.value_with_preference(o,w)
            checks["checkpoint_roundtrip"]={"action_max_abs_diff":ma(va2-va),"value_max_abs_diff":ma(vv2-vv),
                                            "state_hash_equal":before==state_hash(v2r),
                                            "pass":ma(va2-va)<=TOL and ma(vv2-vv)<=TOL and before==state_hash(v2r)}
    
            checks["objective_contract"]=scalarization_check()
            checks["all_finite"]={"pass":bool(torch.isfinite(va).all() and torch.isfinite(vv).all() and torch.isfinite(old).all())}
    
            # No-update paired smoke versus RV1 reference on the exact same visited states.
            cur,_=env.reset(seed=a.seed+99);cur=ot(cur).cuda();mx=0.;terms=0;finite=True
            for _ in range(STEPS):
                with torch.no_grad():
                    av=v2.act_inference_with_preference(cur,w)
                    ar=ref.act_inference_with_preference(cur,w)
                mx=max(mx,ma(av-ar));finite=finite and bool(torch.isfinite(av).all())
                nxt,_,te,tr,_=env.step(av);terms+=int((te|tr).sum().item());cur=ot(nxt).cuda()
            checks["no_update_smoke"]={"steps":STEPS,"num_envs":NENV,"max_action_diff_vs_rv1":mx,
                                       "termination_events":terms,"all_finite":finite,
                                       "pass":mx<=TOL and finite}
    
            # Architecture exclusions.
            names=[n for n,_ in v2.named_parameters()]
            forbidden=("film","adapter","router","expert","aux","reconstruct")
            hits=[n for n in names if any(x in n.lower() for x in forbidden)]
            checks["minimal_v2a_only"]={"forbidden_parameter_hits":hits,
                                        "embedding_parameters":[n for n in names if "preference_embedding" in n],
                                        "pass":len(hits)==0}
    
            passed=all(v.get("pass",False) for v in checks.values())
            report={"schema":"v2a0_function_preserving_gate_v1","status":"V2-A0 PASS" if passed else "V2-A0 FAIL",
                    "training_enabled":False,"optimizer_steps":0,"reference_checkpoint":str(RV1_INIT.relative_to(ROOT)),
                    "reference_definition":"RV1-A frozen repaired initialization",
                    "treatment":"retain RV1 direct conditioning; add 16D learned preference embedding concatenated before actor head; new head columns zero-init",
                    "checks":checks,"authorization":{"v2a1_authorized":bool(passed),"training_in_this_gate":False},
                    "checkpoint":{"path":str(ck.relative_to(ROOT)),"sha256":sha(ck)},
                    "provenance":{"rv1_init_sha256":sha(RV1_INIT),
                                  "v2a_module_sha256":sha(ROOT/"talon_rl/models/foundations/preference_embedding.py"),
                                  "gate_script_sha256":sha(Path(__file__).resolve())}}
            out=a.output_dir/"v2a0_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH",
                "report_sha256":sha(out),"checkpoint_sha256":sha(ck)},indent=2)+"\n")
            print(json.dumps({"status":report["status"],"authorization":report["authorization"],"checks":checks},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2a1_actor_frozen_head_refresh_audit():
    """Run former v2a1_actor_frozen_head_refresh_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2a1_authority_sensitivity-2026-09-23/model_75.pt"
    REP=ROOT/"runs/v2a1_authority_sensitivity-2026-09-23/v2a1_report.json"
    OUT=ROOT/"runs/v2a1_actor_frozen_head_refresh-2026-09-23"
    NENV=8;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    ORDER=("T","A","O","S","C")
    HEADS=("Tracking","Angular","Orientation","Smoothness")
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
        return sol[:-1],sol[-1]
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def collect64(env,m,w_np,seed,label):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager
        w=torch.tensor(w_np,device="cuda") if np.asarray(w_np).ndim==2 else torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(64):
                obs.append(cur)
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector({n:raw[:,i] for i,n in enumerate(names)},shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);D.append((te|tr).cuda())
                cur=ot(nxt).cuda()
        O=torch.stack(obs);R=torch.stack(R);D=torch.stack(D).bool()
        out=[]
        for phase,(st,en) in (("first",(0,32)),("second",(32,64))):
            po=O[st:en].reshape(-1,O.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).cpu().numpy()
            Y=trunc(R[st:en],D[st:en]).reshape(-1,4).cpu().numpy()
            out.append({"phase":phase,"F":F,"Y":Y,"label":label})
        return out
    def fit_units(units):
        F=np.concatenate([u["F"] for u in units]);Y=np.concatenate([u["Y"] for u in units])
        return ridge(F,Y,1.0),{"samples":len(F),"support_mae":[float(np.mean(np.abs((F@ridge(F,Y,1.0)[0]+ridge(F,Y,1.0)[1])[:,j]-Y[:,j]))) for j in range(4)]}
    def pred(F,fit): return F@fit[0]+fit[1]
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
            from talon_rl.models.foundations.preference_embedding import V2AMinimalEmbeddingActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            m=V2AMinimalEmbeddingActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            tr=json.load(open(REP));row75=tr["rows"][-1];sel=row75["selected"];anchor_specs=tr["anchor_specs"]
    
            # Recollect all current-policy anchor/adaptive support.
            anchors=[]
            for k,seed in anchor_specs:
                anchors += [dict(u,source="anchor",spec=k) for u in collect64(env,m,pref_batch(k,torch.device("cuda")).cpu().numpy(),seed,f"a{k}")]
            adapt=[]
            for u in range(52,76):
                seed=73001+200000+u*223
                adapt += [dict(z,source="adaptive",update=u) for z in collect64(env,m,pref_batch(u+17,torch.device("cuda")).cpu().numpy(),seed,f"u{u}")]
            selected=[]
            for phase_key,phase in (("early","first"),("late","second")):
                au=[x for x in anchors if x["phase"]==phase]
                adu=[x for x in adapt if x["phase"]==phase]
                for i in sel[phase_key]["anchor"]:selected.append(au[i])
                for i in sel[phase_key]["adaptive"]:selected.append(adu[i])
            expanded=anchors+adapt
    
            fitA,metaA=fit_units(selected)
            fitB,metaB=fit_units(expanded)
    
            # Exact matched RV1-C endpoint suites, same seeds/preference set.
            eval_units=[]
            for suite in range(4):
                seed=840001+suite
                for lab in ORDER:
                    eval_units += [dict(x,suite=suite,preference=lab) for x in collect64(env,m,PREFS[lab],seed,f"{lab}_s{suite}")]
    
            def evaluate_head(name,fit):
                by_phase={};all_ev=[];all_bias=[]
                for phase in ("first","second"):
                    rows=[x for x in eval_units if x["phase"]==phase]
                    per_head=[];per_pref={}
                    for j,h in enumerate(HEADS):
                        vals=[];bias=[]
                        for r in rows:
                            P=pred(r["F"],fit)
                            vals.append(ev(r["Y"][:,j],P[:,j]))
                            bias.append(float(np.mean(P[:,j]-r["Y"][:,j])))
                        per_head.append({"head":h,"ev_mean":float(np.mean(vals)),
                                         "negative_fraction":float(np.mean(np.asarray(vals)<0)),
                                         "mean_abs_bias":float(np.mean(np.abs(bias)))})
                        all_ev += vals;all_bias += bias
                    for lab in ORDER:
                        rr=[x for x in rows if x["preference"]==lab]
                        vals=[]
                        for r in rr:
                            P=pred(r["F"],fit)
                            vals += [ev(r["Y"][:,j],P[:,j]) for j in range(4)]
                        per_pref[lab]={"ev_mean":float(np.mean(vals)),
                                       "negative_fraction":float(np.mean(np.asarray(vals)<0))}
                    by_phase[phase]={"by_head":per_head,"by_preference":per_pref,
                                     "aggregate_ev_mean":float(np.mean([x["ev_mean"] for x in per_head])),
                                     "aggregate_negative_fraction":float(np.mean([x["negative_fraction"] for x in per_head])),
                                     "aggregate_mean_abs_bias":float(np.mean([x["mean_abs_bias"] for x in per_head]))}
                return {"name":name,"phases":by_phase,
                        "overall_ev_mean":float(np.mean(all_ev)),
                        "overall_negative_fraction":float(np.mean(np.asarray(all_ev)<0)),
                        "overall_mean_abs_bias":float(np.mean(np.abs(all_bias)))}
    
            # Current checkpoint head as baseline, represented as numpy fit.
            Wc=m.critic_head.weight.detach().cpu().numpy().T
            bc=m.critic_head.bias.detach().cpu().numpy()
            baseline=evaluate_head("current_checkpoint_head",(Wc,bc))
            A=evaluate_head("A_selected12_current_policy",fitA)
            B=evaluate_head("B_expanded_current_policy",fitB)
    
            # Predeclared late gate.
            def gate(x):
                first=x["phases"]["first"];second=x["phases"]["second"]
                bh={z["head"]:z for z in second["by_head"]}
                return {
                    "first_preserved": first["aggregate_ev_mean"]>0 and first["aggregate_negative_fraction"]<=.25,
                    "second_tracking_ev_positive": bh["Tracking"]["ev_mean"]>0,
                    "second_orientation_ev_positive": bh["Orientation"]["ev_mean"]>0,
                    "second_negative_fraction": second["aggregate_negative_fraction"]<=.25,
                    "second_bias_bounded": second["aggregate_mean_abs_bias"]<=.10,
                }
            gates={}
            for x in (baseline,A,B):
                g=gate(x);g["pass"]=all(g.values());gates[x["name"]]=g
    
            # Interpretation branch.
            if gates[A["name"]]["pass"]:
                verdict="A_PASS_FRESHNESS_LAG_SUFFICIENT"
            elif gates[B["name"]]["pass"]:
                verdict="B_PASS_SUPPORT_BREADTH_PRIMARY"
            else:
                verdict="A_B_FAIL_TRUE_SUPPORT_DISTRIBUTION_GAP_REMAINS"
    
            result={"schema":"v2a1_actor_frozen_head_refresh_audit_v1","measurement_only":True,
                    "actor_frozen":True,"critic_body_frozen":True,"optimizer_steps":0,
                    "support":{"selected12":metaA,"expanded":metaB,"selected_indices":sel},
                    "baseline":baseline,"A_selected12":A,"B_expanded":B,
                    "gates":gates,"verdict":verdict}
            out=a.output_dir/"head_refresh_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({
                "status":"FROZEN_BY_HASH","report_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
                "checkpoint_sha256":hashlib.sha256(CKPT.read_bytes()).hexdigest()},indent=2)+"\n")
            print(json.dumps({"verdict":verdict,"gates":gates,
                              "second_phase":{"baseline":baseline["phases"]["second"],
                                              "A":A["phases"]["second"],"B":B["phases"]["second"]}},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2a1_authority_sensitivity_screen():
    """Run former v2a1_authority_sensitivity_screen.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2a1_authority_sensitivity-2026-09-23"
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
    
    def fit_phase_balanced(m,anchor_units,adaptive_pools,adaptive_summaries):
        support=[];sel={}
        for phase in ("early","late"):
            au=anchor_units[phase]; asum=[x["summary"] for x in au]
            ai=select_diverse(asum,ANCHOR_PHASE_K)
            ri=select_diverse(adaptive_summaries[phase],ADAPT_PHASE_K)
            support += [(au[i]["F"],au[i]["Y"]) for i in ai]
            support += [(adaptive_pools[phase][i]["F"],adaptive_pools[phase][i]["Y"]) for i in ri]
            sel[phase]={"anchor":ai,"adaptive":ri}
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return sel
    
    def pairwise_dist(actions):
        labs=list(actions);v=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:v.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def sensitivity(m,probe):
        total={};masked={};embs={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                total[lab]=m.act_inference_with_preference(probe,w)
                h=m.actor_body(m._with_w(probe,w))
                z=torch.zeros((len(probe),m.EMBED_DIM),device=probe.device,dtype=probe.dtype)
                masked[lab]=torch.tanh(m.actor_mean(torch.cat((h,z),dim=-1)))*m.ACTION_CLIP
                embs[lab]=m.preference_embedding(w)
        pair_total=pairwise_dist(total);pair_masked=pairwise_dist(masked);pair_emb=pairwise_dist(embs)
        authority=float(np.mean([torch.linalg.vector_norm(total[k]-masked[k],dim=1).mean().cpu() for k in ORDER]))
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w)
        rows=[]
        for j in range(a.shape[1]): rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
        # Direct-only/masked embedding path retains RV1 path but removes V2 embedding authority.
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        h=m.actor_body(m._with_w(probe,wm))
        z=torch.zeros((len(probe),m.EMBED_DIM),device=probe.device,dtype=probe.dtype)
        am=torch.tanh(m.actor_mean(torch.cat((h,z),dim=-1)))*m.ACTION_CLIP
        mrows=[]
        for j in range(am.shape[1]): mrows.append(torch.autograd.grad(am[:,j].mean(),wm,retain_graph=True)[0])
        mjac=torch.stack(mrows,dim=1)
        direct_cols=m.actor_body[0].weight[:,m.physical_obs_dim:]
        embed_cols=m.actor_mean.weight[:,-m.EMBED_DIM:]
        return {"pairwise_action_distance":pair_total,
                "masked_embedding_pairwise_action_distance":pair_masked,
                "embedding_feature_pairwise_distance":pair_emb,
                "embedding_action_authority_mean":authority,
                "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
                "masked_embedding_jacobian_fro_mean":float(torch.linalg.matrix_norm(mjac,ord="fro",dim=(1,2)).mean().detach().cpu()),
                "direct_preference_weight_norm":float(direct_cols.norm().detach().cpu()),
                "embedding_readout_norm":float(embed_cols.norm().detach().cpu()),
                "embedding_parameter_norm":float(torch.sqrt(sum((p.detach()**2).sum() for p in m.preference_embedding.parameters())).cpu())}
    
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
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            from talon_rl.models.foundations.preference_embedding import V2AMinimalEmbeddingActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2AMinimalEmbeddingActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(INIT,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n.startswith("preference_embedding") or n=="log_std"]
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
            fit_phase_balanced(m,current_anchor_units(),adaptive_pools,adaptive_summaries)
    
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
                selected=fit_phase_balanced(m,current_anchor_units(),adaptive_pools,adaptive_summaries)
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
                eg_terms=[p.grad for p in m.preference_embedding.parameters() if p.grad is not None]
                eg=float(torch.sqrt(sum((g.detach()**2).sum() for g in eg_terms)).cpu()) if eg_terms else 0.0
                rg=float(m.actor_mean.weight.grad[:,-m.EMBED_DIM:].norm().detach().cpu())
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                              "preference_input_grad_norm":pg,"embedding_parameter_grad_norm":eg,
                             "embedding_readout_grad_norm":rg,"actor_grad_norm_preclip":total,
                             "termination_fraction":main["termination_fraction"],"selected":selected})
                if uidx in SNAPS:audit(uidx)
    
            report={"schema":"v2a1_authority_sensitivity_screen_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"V2-A1: same validated foundation; treatment only actor preference embedding authority",
                    "actor_ppo_objectives_changed":False,"conditioning_treatment":"RV1 direct path + 16D learned embedding readout","anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            final=snaps[str(args.updates)];initial=snaps["0"]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            embg=np.array([r["embedding_parameter_grad_norm"] for r in rows]);readg=np.array([r["embedding_readout_grad_norm"] for r in rows])
            crit=final["phase_critic"];sens=final["sensitivity"];s0=initial["sensitivity"]
            total_pair=sens["pairwise_action_distance"]["mean"];masked_pair=sens["masked_embedding_pairwise_action_distance"]["mean"]
            total_j=sens["jacobian_fro_mean"];masked_j=sens["masked_embedding_jacobian_fro_mean"]
            criteria={
              "embedding_readout_nonzero":sens["embedding_readout_norm"]>1e-5,
              "embedding_gradient_observed":float(np.max(embg))>1e-7 and float(np.median(readg))>1e-6,
              "fixed_state_action_separation":total_pair>1e-3,
              "preference_jacobian_nonzero":total_j>1e-3,
              "embedding_adds_action_separation":total_pair>masked_pair+1e-4,
              "embedding_adds_jacobian_authority":total_j>masked_j+1e-4,
              "embedding_action_authority_nonzero":sens["embedding_action_authority_mean"]>1e-4,
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
            report["summary"]={"status":"V2-A1 PASS" if passed else "V2-A1 FAIL",
                "criteria":criteria,"initial":initial,"final":final,
                "max_embedding_parameter_grad_norm":float(np.max(embg)),
                "median_embedding_parameter_grad_norm":float(np.median(embg)),
                "median_embedding_readout_grad_norm":float(np.median(readg)),
                "last10_termination_fraction":float(term[-10:].mean()),
                "max_ratio_error":float(ratio.max()),
                "v2a2_authorized":bool(passed)}
            (args.output_dir/"v2a1_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "v2a0_function_preserving_gate": run_v2a0_function_preserving_gate,
    "v2a1_actor_frozen_head_refresh_audit": run_v2a1_actor_frozen_head_refresh_audit,
    "v2a1_authority_sensitivity_screen": run_v2a1_authority_sensitivity_screen,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
