"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_rv1_a_implementation_gate():
    """Run former rv1_a_implementation_gate.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, math, sys, tempfile
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT_DEFAULT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23"
    CHECKPOINT=ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt"
    WREF=np.asarray([.25,.25,.25,.25],dtype=np.float32)
    NENV=8
    SMOKE_STEPS=64
    TOL=1e-6
    
    def sha256(path:Path)->str:
        h=hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda:f.read(1<<20),b""): h.update(b)
        return h.hexdigest()
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def maxabs(x): return float(torch.max(torch.abs(x)).detach().cpu()) if x.numel() else 0.0
    
    def shadow_actor_mean(model,obs):
        # Preference columns are zero-initialized. The repaired baseline is the
        # identical actor evaluated on physical observation columns only.
        x=obs
        for i in (0,2,4):
            layer=model.actor_body[i]
            if i==0:
                x=torch.nn.functional.linear(x,layer.weight[:,:model.physical_obs_dim],layer.bias)
            else:
                x=layer(x)
            x=torch.nn.functional.elu(x)
        return model.actor_mean(x)
    
    def snapshot_hash(model):
        h=hashlib.sha256()
        for n,t in model.state_dict().items():
            h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    
    def scalarization_check():
        from talon_rl.rewards.objectives import OBJECTIVE_ORDER,NORMALIZATION_DIVISORS,normalize_objectives,scalarize
        raw=np.asarray([[1.7194554805755615,-.15590913593769073,-.01563369482755661,-.08311229199171066]],np.float32)
        norm=normalize_objectives(raw)
        expected=np.asarray([[1.,-1.,-1.,-1.]],np.float32)
        got=float(scalarize(norm,WREF)[0])
        ref=float((expected*WREF).sum())
        return {
            "objective_order":list(OBJECTIVE_ORDER),
            "normalization_divisors":NORMALIZATION_DIVISORS.tolist(),
            "normalized_probe_max_abs_error":float(np.max(np.abs(norm-expected))),
            "scalarization_abs_error":abs(got-ref),
            "pass":bool(np.max(np.abs(norm-expected))<=1e-7 and abs(got-ref)<=1e-7),
        }
    
    def paired_no_update_smoke(env,model,w,seed,steps):
        # Both lanes use the same model/state trajectory. The shadow action is
        # checked at every visited state before the actual repaired action steps env.
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        max_action_diff=0.0;finite=True;terminated=0
        for _ in range(steps):
            with torch.no_grad():
                a=model.act_inference_with_preference(cur,w)
                shadow=torch.tanh(shadow_actor_mean(model,cur))*model.ACTION_CLIP
            max_action_diff=max(max_action_diff,maxabs(a-shadow))
            finite=finite and bool(torch.isfinite(a).all() and torch.isfinite(shadow).all())
            nxt,_,te,tr,_=env.step(a)
            terminated+=int((te|tr).sum().item());cur=obs_tensor(nxt).cuda()
        return {
            "steps":steps,"num_envs":NENV,"max_action_diff":max_action_diff,
            "all_finite":finite,"termination_events":terminated,
            "function_preserved":bool(max_action_diff<=TOL and finite),
        }
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT)
        ap.add_argument("--seed",type=int,default=424242)
        args=ap.parse_args()
        if not args.output_dir.is_absolute():
            args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,OBJECTIVE_ORDER
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=obs_tensor(o).cuda()
            ad=env.unwrapped.action_manager.total_action_dim
            torch.manual_seed(args.seed);np.random.seed(args.seed)
            model=T4SharedActorCritic(o.shape[-1],ad).cuda()
            initialize_from_rsl_m01(model,CHECKPOINT,device="cpu",critic_head_init="zero")
            model.eval()
            w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
    
            with torch.no_grad():
                a=model.act_inference_with_preference(o,w)
                shadow=torch.tanh(shadow_actor_mean(model,o))*model.ACTION_CLIP
                actor_diff=maxabs(a-shadow)
                first=model.actor_body[0].weight
                pref_col_norm=float(first[:,model.physical_obs_dim:].norm().cpu())
                value=model.value_with_preference(o,w)
                critic_zero=maxabs(value)
                critic_shape=list(value.shape)
    
            # Stochastic action/log-prob identity using the exact same pre-tanh latent.
            torch.manual_seed(args.seed+1)
            with torch.no_grad():
                action,stored_logp,u=model.act_with_preference_latent(o,w)
                recomputed=model.logp_from_pre_tanh_with_preference(o,w,u)
                action_from_u=torch.tanh(u)*model.ACTION_CLIP
                logp_diff=maxabs(stored_logp-recomputed)
                latent_action_diff=maxabs(action-action_from_u)
                all_finite=bool(torch.isfinite(action).all() and torch.isfinite(stored_logp).all() and torch.isfinite(value).all())
    
            # Action-manager identity: raw action received by the environment must
            # equal the policy action used to define the stored latent/log-prob.
            _=env.reset(seed=args.seed+7)
            with torch.no_grad():
                action2,logp2,u2=model.act_with_preference_latent(o,w)
            env.step(action2)
            term=env.unwrapped.action_manager._terms["joint_pos"]
            raw_received=term.raw_actions.detach()
            env_raw_diff=maxabs(raw_received-action2)
            env_latent_diff=maxabs(raw_received-(torch.tanh(u2)*model.ACTION_CLIP))
            processed=term.processed_actions.detach()
            processed_finite=bool(torch.isfinite(processed).all())
    
            # Save/load round trip must not change outputs or model state.
            before_state=snapshot_hash(model)
            ckpt=args.output_dir/"rv1_a_init.pt"
            torch.save({"model":model.state_dict(),"reference_preference":WREF.tolist(),"training_steps":0},ckpt)
            reload_model=T4SharedActorCritic(o.shape[-1],ad).cuda()
            state=torch.load(ckpt,map_location="cuda",weights_only=False)
            reload_model.load_state_dict(state["model"]);reload_model.eval()
            with torch.no_grad():
                reload_action=reload_model.act_inference_with_preference(o,w)
                reload_value=reload_model.value_with_preference(o,w)
            roundtrip_action_diff=maxabs(reload_action-a)
            roundtrip_value_diff=maxabs(reload_value-value)
            roundtrip_state_equal=(before_state==snapshot_hash(reload_model))
    
            scalar=scalarization_check()
            smoke=paired_no_update_smoke(env,model,w,args.seed+99,SMOKE_STEPS)
            checks={
                "deterministic_reference_action":{"max_abs_diff":actor_diff,"pass":actor_diff<=TOL},
                "preference_columns_zero":{"l2_norm":pref_col_norm,"pass":pref_col_norm==0.0},
                "critic_zero_init":{"shape":critic_shape,"max_abs_output":critic_zero,"pass":critic_shape==[NENV,4] and critic_zero==0.0},
                "stochastic_logprob_identity":{"max_abs_diff":logp_diff,"pass":logp_diff<=TOL},
                "latent_to_action_identity":{"max_abs_diff":latent_action_diff,"pass":latent_action_diff<=TOL},
                "env_raw_action_identity":{"max_abs_diff":env_raw_diff,"latent_max_abs_diff":env_latent_diff,
                                           "processed_finite":processed_finite,
                                           "pass":env_raw_diff<=TOL and env_latent_diff<=TOL and processed_finite},
                "checkpoint_roundtrip":{"action_max_abs_diff":roundtrip_action_diff,"value_max_abs_diff":roundtrip_value_diff,
                                        "state_hash_equal":roundtrip_state_equal,
                                        "pass":roundtrip_action_diff<=TOL and roundtrip_value_diff<=TOL and roundtrip_state_equal},
                "objective_contract":scalar,
                "all_finite":{"pass":all_finite},
                "no_update_smoke":smoke,
            }
            pass_all=all(bool(v.get("pass",v.get("function_preserved",False))) for v in checks.values())
            # Explicit architecture exclusion audit.
            param_names=[n for n,_ in model.named_parameters()]
            forbidden=("film","latent","adapter","router","expert","aux","reconstruct")
            forbidden_hits=[n for n in param_names if any(x in n.lower() for x in forbidden)]
            checks["minimal_architecture"]={"forbidden_parameter_hits":forbidden_hits,"pass":len(forbidden_hits)==0}
            pass_all=pass_all and len(forbidden_hits)==0
    
            sources=[
                ROOT/"docs/contracts/general/master-synthesis-repaired-foundation-contract.md",
                ROOT/"talon_rl/models/foundations/four_objective.py",
                ROOT/"talon_rl/t3b_objectives.py",
                Path(__file__).resolve(),
                CHECKPOINT,
            ]
            provenance={str(p.relative_to(ROOT)):sha256(p) for p in sources}
            report={
                "schema":"rv1_a_implementation_gate_v1",
                "status":"RV1-A PASS" if pass_all else "RV1-A FAIL",
                "training_enabled":False,"optimizer_steps":0,
                "reference_definition":"shadow repaired actor: same physical actor weights and ACTION_CLIP=1, with preference contribution exactly zero",
                "reference_preference":WREF.tolist(),
                "objective_order":list(OBJECTIVE_ORDER),
                "action_clip":float(model.ACTION_CLIP),
                "checks":checks,"provenance":provenance,
                "checkpoint":{"path":str(ckpt.relative_to(ROOT)),"sha256":sha256(ckpt)},
                "authorization":{"rv1_b_authorized":bool(pass_all),"training_in_this_gate":False},
            }
            out=args.output_dir/"rv1_a_report.json"
            out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({"status":report["status"],"report":str(out),"rv1_b_authorized":bool(pass_all)},indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True: main()

def run_rv1_conditional_matched_donor_attribution():
    """Run former rv1_conditional_matched_donor_attribution.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
    REP=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/phase_repair_report.json"
    OUT=ROOT/"runs/rv1_conditional_matched_donor_attribution-2026-09-23"
    NENV=8;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    ORDER=("T","A","O","S","C")
    ALPHAS=(0,.25,.5,.75,1.0);PATHS=(("T","A"),("T","O"),("T","S"))
    BLOCKS={
     "base_lin_vel":slice(0,3),
     "base_ang_vel":slice(3,6),
     "projected_gravity":slice(6,9),
     "velocity_command":slice(9,12),
     "joint_pos":slice(12,24),
     "joint_vel":slice(24,36),
     "previous_action":slice(36,48),
    }
    CANDIDATES={
     "joint_pos":["joint_pos"],
     "velocity_command":["velocity_command"],
     "preference":["preference"],
     "joint_pos+velocity_command":["joint_pos","velocity_command"],
    }
    
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
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        obs=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(64):
                obs.append(cur.detach().cpu().numpy())
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=np.zeros((NENV,4),np.float32)
                from talon_rl.rewards.objectives import normalized_objective_vector
                vec=normalized_objective_vector({n:raw[:,i] for i,n in enumerate(names)},shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);D.append((te|tr).cuda())
                cur=ot(nxt).cuda()
        O=np.asarray(obs)[32:64].reshape(-1,48)
        W=np.repeat(w.cpu().numpy()[None,:,:],32,axis=0).reshape(-1,4)
        Y=trunc(torch.stack(R)[32:64],torch.stack(D).bool()[32:64]).reshape(-1,4).cpu().numpy()
        return {"obs":O,"w":W,"target":Y,"label":label}
    
    def features_values(m,obs,w):
        with torch.no_grad():
            O=torch.tensor(obs,dtype=torch.float32,device="cuda")
            W=torch.tensor(w,dtype=torch.float32,device="cuda")
            F=m.critic_body(m._with_w(O,W))
            V=m.critic_head(F)
        return F.cpu().numpy(),V.cpu().numpy()
    
    def support_feature_ref(Sf):
        mu=Sf.mean(0);sd=Sf.std(0)+1e-8;Z=(Sf-mu)/sd
        dev="cuda"
        T=torch.tensor(Z,dtype=torch.float32,device=dev);D=torch.cdist(T,T);D.fill_diagonal_(float("inf"))
        loo=(D.min(1).values/np.sqrt(Sf.shape[1])).cpu().numpy()
        return mu,sd,float(np.quantile(loo,.95)),float(np.mean(loo))
    
    def feature_dist(Sf,Qf,mu,sd):
        Sz=(Sf-mu)/sd;Qz=(Qf-mu)/sd
        St=torch.tensor(Sz,dtype=torch.float32,device="cuda")
        out=[]
        for i in range(0,len(Qz),512):
            Qt=torch.tensor(Qz[i:i+512],dtype=torch.float32,device="cuda")
            out.append((torch.cdist(Qt,St).min(1).values/np.sqrt(Sf.shape[1])).cpu().numpy())
        return np.concatenate(out)
    
    def build_match_matrix(obs,w,exclude):
        cols=[]
        for name,sl in BLOCKS.items():
            if name not in exclude: cols.append(obs[:,sl])
        if "preference" not in exclude: cols.append(w)
        return np.concatenate(cols,axis=1)
    
    def conditional_donors(Sobs,Sw,Qobs,Qw,exclude,maxq=None):
        SX=build_match_matrix(Sobs,Sw,exclude);QX=build_match_matrix(Qobs,Qw,exclude)
        mu=SX.mean(0);sd=SX.std(0)+1e-8;SZ=(SX-mu)/sd;QZ=(QX-mu)/sd
        St=torch.tensor(SZ,dtype=torch.float32,device="cuda")
        donors=[];dists=[]
        for i in range(0,len(QZ),512):
            Qt=torch.tensor(QZ[i:i+512],dtype=torch.float32,device="cuda")
            D=torch.cdist(Qt,St)/np.sqrt(SZ.shape[1])
            v,ix=D.min(1);donors.append(ix.cpu().numpy());dists.append(v.cpu().numpy())
        return np.concatenate(donors),np.concatenate(dists)
    
    def apply_replace(Qobs,Qw,Sobs,Sw,donor,replace):
        O=Qobs.copy();W=Qw.copy()
        for name in replace:
            if name=="preference":W[:]=Sw[donor]
            else:O[:,BLOCKS[name]]=Sobs[donor][:,BLOCKS[name]]
        return O,W
    
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    
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
            Sobs,Sw,Sy=cat(sup,"obs"),cat(sup,"w"),cat(sup,"target")
            Sf,Sv=features_values(m,Sobs,Sw);mu,sd,thr,loo_mean=support_feature_ref(Sf)
            result={"schema":"rv1_conditional_matched_donor_attribution_v1","measurement_only":True,
                    "support":{"states":len(Sobs),"feature_loo_p95":thr,"feature_loo_mean":loo_mean},"groups":{}}
            for gname,rows in (("endpoint",ep),("continuum",ct),("all",ep+ct)):
                Qobs,Qw,Qy=cat(rows,"obs"),cat(rows,"w"),cat(rows,"target")
                Qf,Qv=features_values(m,Qobs,Qw);base_d=feature_dist(Sf,Qf,mu,sd)
                base={"feature_oos":float(np.mean(base_d>thr)),"feature_nn_mean":float(base_d.mean()),
                      "feature_nn_p50":float(np.quantile(base_d,.5)),
                      "value_ev_mean":float(np.mean([ev(Qy[:,j],Qv[:,j]) for j in range(4)])),
                      "value_mae_mean":float(np.mean(np.abs(Qv-Qy)))}
                out={"baseline":base,"candidates":{}}
                for cname,replace in CANDIDATES.items():
                    exclude=set(replace)
                    donor,md=conditional_donors(Sobs,Sw,Qobs,Qw,exclude)
                    O2,W2=apply_replace(Qobs,Qw,Sobs,Sw,donor,replace)
                    F2,V2=features_values(m,O2,W2);d2=feature_dist(Sf,F2,mu,sd)
                    out["candidates"][cname]={
                        "donor_match_mean":float(md.mean()),"donor_match_p95":float(np.quantile(md,.95)),
                        "feature_oos":float(np.mean(d2>thr)),
                        "feature_oos_drop":float(base["feature_oos"]-np.mean(d2>thr)),
                        "feature_nn_mean":float(d2.mean()),
                        "feature_nn_mean_drop":float(base["feature_nn_mean"]-d2.mean()),
                        "value_ev_mean":float(np.mean([ev(Qy[:,j],V2[:,j]) for j in range(4)])),
                        "value_ev_gain":float(np.mean([ev(Qy[:,j],V2[:,j]) for j in range(4)])-base["value_ev_mean"]),
                        "value_mae_mean":float(np.mean(np.abs(V2-Qy))),
                        "value_mae_drop":float(base["value_mae_mean"]-np.mean(np.abs(V2-Qy))),
                    }
                result["groups"][gname]=out
            rank=sorted(result["groups"]["all"]["candidates"].items(),key=lambda kv:kv[1]["feature_oos_drop"],reverse=True)
            result["summary"]={"all_baseline":result["groups"]["all"]["baseline"],
                               "ranking":[{"candidate":k,**v} for k,v in rank],
                               "top_candidate":rank[0][0]}
            out=a.output_dir/"conditional_attribution_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":hashlib.sha256(out.read_bytes()).hexdigest()},indent=2)+"\n")
            print(json.dumps(result["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_fixed_feature_mapping_capacity_audit():
    """Run former rv1_fixed_feature_mapping_capacity_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    import torch.nn as nn
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
    REP=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/phase_repair_report.json"
    OUT=ROOT/"runs/rv1_fixed_feature_mapping_capacity-2026-09-23"
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
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(64):
                obs.append(cur)
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector({n:raw[:,i] for i,n in enumerate(names)},shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);D.append((te|tr).cuda());cur=ot(nxt).cuda()
        O=torch.stack(obs)[32:64].reshape(-1,obs[0].shape[-1]);W=w.repeat(32,1)
        Y=trunc(torch.stack(R)[32:64],torch.stack(D).bool()[32:64]).reshape(-1,4)
        with torch.no_grad():
            F=m.critic_body(m._with_w(O,W));V=m.critic_head(F)
        return {"feature":F.cpu().numpy(),"current":V.cpu().numpy(),"target":Y.cpu().numpy(),"w":W.cpu().numpy(),"label":label}
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
        return sol[:-1],sol[-1]
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def metrics(Y,P,mask=None):
        if mask is not None:Y=Y[mask];P=P[mask]
        return {"ev":[ev(Y[:,j],P[:,j]) for j in range(4)],
                "mae":[float(np.mean(np.abs(Y[:,j]-P[:,j]))) for j in range(4)]}
    def nearest_feature_distance(SF,QF):
        mu=SF.mean(0);sd=SF.std(0)+1e-8;SZ=(SF-mu)/sd;QZ=(QF-mu)/sd
        St=torch.tensor(SZ,dtype=torch.float32,device="cuda");vals=[]
        for i in range(0,len(QZ),512):
            Qt=torch.tensor(QZ[i:i+512],dtype=torch.float32,device="cuda")
            vals.append((torch.cdist(Qt,St).min(1).values/np.sqrt(SF.shape[1])).cpu().numpy())
        return np.concatenate(vals)
    class Probe(nn.Module):
        def __init__(self,d):
            super().__init__();self.net=nn.Sequential(nn.Linear(d,64),nn.ELU(),nn.Linear(64,32),nn.ELU(),nn.Linear(32,4))
        def forward(self,x):return self.net(x)
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
            SF,SY=cat(sup,"feature"),cat(sup,"target")
            # fresh linear ridge on exact real support
            W,b=ridge(SF,SY,1.0)
            # nonlinear probe: diagnostic only, trained on support with deterministic 80/20 split
            rng=np.random.default_rng(1234);idx=np.arange(len(SF));rng.shuffle(idx);cut=int(.8*len(idx));tridx,validx=idx[:cut],idx[cut:]
            Xtr=torch.tensor(SF[tridx],dtype=torch.float32,device="cuda");Ytr=torch.tensor(SY[tridx],dtype=torch.float32,device="cuda")
            Xva=torch.tensor(SF[validx],dtype=torch.float32,device="cuda");Yva=torch.tensor(SY[validx],dtype=torch.float32,device="cuda")
            torch.manual_seed(1234);probe=Probe(SF.shape[1]).cuda();opt=torch.optim.Adam(probe.parameters(),lr=1e-3,weight_decay=1e-5)
            best=None;best_state=None
            for epc in range(300):
                probe.train();pred=probe(Xtr);loss=((pred-Ytr)**2).mean();opt.zero_grad();loss.backward();opt.step()
                if epc%5==0:
                    probe.eval()
                    with torch.no_grad():vl=float(((probe(Xva)-Yva)**2).mean().cpu())
                    if best is None or vl<best:best=vl;best_state={k:v.detach().cpu().clone() for k,v in probe.state_dict().items()}
            probe.load_state_dict(best_state);probe.eval()
            result={"schema":"rv1_fixed_feature_mapping_capacity_v1","measurement_only":True,
                    "probe":{"support_validation_mse":best},"groups":{}}
            for gname,rows in (("endpoint",ep),("continuum",ct),("all",ep+ct)):
                QF,QY,Qcur=cat(rows,"feature"),cat(rows,"target"),cat(rows,"current")
                Qridge=QF@W+b
                with torch.no_grad():Qprobe=probe(torch.tensor(QF,dtype=torch.float32,device="cuda")).cpu().numpy()
                fd=nearest_feature_distance(SF,QF);close=fd<=np.quantile(fd,.25);far=fd>=np.quantile(fd,.75)
                result["groups"][gname]={
                    "feature_distance":{"p25":float(np.quantile(fd,.25)),"p75":float(np.quantile(fd,.75))},
                    "current":{"all":metrics(QY,Qcur),"close":metrics(QY,Qcur,close),"far":metrics(QY,Qcur,far)},
                    "fresh_ridge":{"all":metrics(QY,Qridge),"close":metrics(QY,Qridge,close),"far":metrics(QY,Qridge,far)},
                    "nonlinear_probe":{"all":metrics(QY,Qprobe),"close":metrics(QY,Qprobe,close),"far":metrics(QY,Qprobe,far)}}
            # compact decision hints per head on ALL set
            hints=[]
            g=result["groups"]["all"]
            for j,h in enumerate(HEADS):
                c=g["current"]["close"]["mae"][j];r=g["fresh_ridge"]["close"]["mae"][j];n=g["nonlinear_probe"]["close"]["mae"][j]
                hints.append({"head":h,"current_close_mae":c,"ridge_close_mae":r,"probe_close_mae":n,
                              "ridge_gain_frac":float((c-r)/(c+1e-12)),"probe_gain_over_ridge_frac":float((r-n)/(r+1e-12))})
            result["summary"]={"close_set_capacity_hints":hints}
            out=a.output_dir/"capacity_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":hashlib.sha256(out.read_bytes()).hexdigest()},indent=2)+"\n")
            print(json.dumps(result["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_local_target_mapping_audit():
    """Run former rv1_local_target_mapping_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
    REP=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/phase_repair_report.json"
    OUT=ROOT/"runs/rv1_local_target_mapping_audit-2026-09-23"
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
            result={"schema":"rv1_local_target_mapping_audit_v1","measurement_only":True,"groups":{}}
            for gname,rows in (("endpoint",ep),("continuum",ct),("all",ep+ct)):
                Qf,Qv,Qy,Qc,Qw=[cat(rows,k) for k in ("feature","value","target","command","w")]
                QZ=(Qf-mu)/sd;don=[];fd=[]
                for i in range(0,len(QZ),512):
                    Qt=torch.tensor(QZ[i:i+512],dtype=torch.float32,device="cuda")
                    D=torch.cdist(Qt,St)/np.sqrt(Sf.shape[1]);v,ix=D.min(1)
                    don.append(ix.cpu().numpy());fd.append(v.cpu().numpy())
                don=np.concatenate(don);fd=np.concatenate(fd)
                dy=np.abs(Qy-Sy[don]);res=np.abs(Qv-Qy)
                cmd=np.linalg.norm(Qc-Sc[don],axis=1);pref=np.linalg.norm(Qw-Sw[don],axis=1)
                close=fd<=np.quantile(fd,.25);far=fd>=np.quantile(fd,.75)
                heads=[]
                for j,h in enumerate(HEADS):
                    heads.append({"head":h,
                        "corr_residual_feature_distance":corr(res[:,j],fd),
                        "corr_residual_command_mismatch":corr(res[:,j],cmd),
                        "corr_target_delta_feature_distance":corr(dy[:,j],fd),
                        "close_neighbor":{"n":int(close.sum()),"feature_dist_mean":float(fd[close].mean()),
                                          "target_delta_mean":float(dy[close,j].mean()),"residual_mean":float(res[close,j].mean()),
                                          "command_mismatch_mean":float(cmd[close].mean())},
                        "far_neighbor":{"n":int(far.sum()),"feature_dist_mean":float(fd[far].mean()),
                                        "target_delta_mean":float(dy[far,j].mean()),"residual_mean":float(res[far,j].mean()),
                                        "command_mismatch_mean":float(cmd[far].mean())},
                        "regression_residual_on_feature_command":regress(res[:,j],fd,cmd),
                        "regression_target_delta_on_feature":regress(dy[:,j],fd)})
                result["groups"][gname]={"n":len(Qf),"feature_distance":{"mean":float(fd.mean()),"p25":float(np.quantile(fd,.25)),
                     "p50":float(np.quantile(fd,.5)),"p75":float(np.quantile(fd,.75))},
                     "command_mismatch":{"mean":float(cmd.mean()),"p50":float(np.quantile(cmd,.5))},
                     "preference_mismatch":{"mean":float(pref.mean())},"heads":heads}
            out=a.output_dir/"local_mapping_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":hashlib.sha256(out.read_bytes()).hexdigest()},indent=2)+"\n")
            print(json.dumps(result["groups"]["all"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_observation_block_attribution_audit():
    """Run former rv1_observation_block_attribution_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
    REP=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/phase_repair_report.json"
    OUT=ROOT/"runs/rv1_observation_block_attribution-2026-09-23"
    NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    ORDER=("T","A","O","S","C")
    ALPHAS=(0,.25,.5,.75,1.0);PATHS=(("T","A"),("T","O"),("T","S"))
    BLOCKS={
     "base_lin_vel":slice(0,3),
     "base_ang_vel":slice(3,6),
     "projected_gravity":slice(6,9),
     "velocity_command":slice(9,12),
     "joint_pos":slice(12,24),
     "joint_vel":slice(24,36),
     "previous_action":slice(36,48),
    }
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    def interp(a,b,t):return (1-t)*PREFS[a]+t*PREFS[b]
    def collect_obs64(env,m,w_np,seed,label):
        w=torch.tensor(w_np,device="cuda") if np.asarray(w_np).ndim==2 else torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[]
        with torch.no_grad():
            for _ in range(64):
                obs.append(cur.detach().cpu().numpy())
                a=m.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        O=np.asarray(obs)[32:64].reshape(-1,48);W=np.repeat(w.cpu().numpy()[None,:,:],32,axis=0).reshape(-1,4)
        return {"obs":O,"w":W,"label":label}
    def nn_stats(S,Q):
        S=np.asarray(S,float);Q=np.asarray(Q,float);mu=S.mean(0);sd=S.std(0)+1e-8
        Sz=(S-mu)/sd;Qz=(Q-mu)/sd
        dev="cuda" if torch.cuda.is_available() else "cpu"
        St=torch.tensor(Sz,dtype=torch.float32,device=dev)
        dss=torch.cdist(St,St);dss.fill_diagonal_(float("inf"))
        base=(dss.min(1).values/np.sqrt(S.shape[1])).cpu().numpy()
        qs=[]
        for i in range(0,len(Qz),512):
            Qt=torch.tensor(Qz[i:i+512],dtype=torch.float32,device=dev)
            qs.append((torch.cdist(Qt,St).min(1).values/np.sqrt(S.shape[1])).cpu().numpy())
        q=np.concatenate(qs);thr=float(np.quantile(base,.95))
        return {"support_p95":thr,"semantic_p50":float(np.quantile(q,.5)),
                "semantic_p95":float(np.quantile(q,.95)),"oos_fraction":float(np.mean(q>thr))}
    def cov_mismatch(S,Q):
        S=np.asarray(S,float);Q=np.asarray(Q,float);mu=S.mean(0);sd=S.std(0)+1e-8
        Sz=(S-mu)/sd;Qz=(Q-mu)/sd
        Cs=np.cov(Sz,rowvar=False);Cq=np.cov(Qz,rowvar=False)
        if np.ndim(Cs)==0: return 0.0
        return float(np.linalg.norm(Cq-Cs,"fro")/(np.linalg.norm(Cs,"fro")+1e-12))
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
                k,seed=anchor_specs[loc];sup.append(collect_obs64(env,m,pref_batch(k,torch.device("cuda")).cpu().numpy(),seed,f"a{k}"))
            for idx in late["adaptive"]:
                u=52+idx;seed=73001+200000+u*223;sup.append(collect_obs64(env,m,pref_batch(u+17,torch.device("cuda")).cpu().numpy(),seed,f"r{u}"))
            sem_ep=[];sem_ct=[]
            for suite in range(4):
                seed=840001+suite
                for lab in ORDER:sem_ep.append(collect_obs64(env,m,PREFS[lab],seed,f"ep_{lab}_{suite}"))
            for pi,(x,y) in enumerate(PATHS):
                for suite in range(4):
                    seed=850001+pi*1000+suite
                    for al in ALPHAS:sem_ct.append(collect_obs64(env,m,interp(x,y,al),seed,f"{x}{y}_{al}_{suite}"))
            def cat(rows,key):return np.concatenate([r[key] for r in rows],axis=0)
            Sobs,Sw=cat(sup,"obs"),cat(sup,"w")
            result={"schema":"rv1_observation_block_attribution_v1","measurement_only":True,"blocks":{}}
            for name,sl in BLOCKS.items():
                result["blocks"][name]={}
                for grp,rows in (("endpoint",sem_ep),("continuum",sem_ct),("all",sem_ep+sem_ct)):
                    Q=cat(rows,"obs")[:,sl];S=Sobs[:,sl]
                    result["blocks"][name][grp]={"nn":nn_stats(S,Q),"covariance_mismatch":cov_mismatch(S,Q)}
            result["blocks"]["preference"]={}
            for grp,rows in (("endpoint",sem_ep),("continuum",sem_ct),("all",sem_ep+sem_ct)):
                result["blocks"]["preference"][grp]={"nn":nn_stats(Sw,cat(rows,"w")),"covariance_mismatch":cov_mismatch(Sw,cat(rows,"w"))}
            # baseline feature OOS and block-replacement intervention
            def feats(obs,w):
                with torch.no_grad():
                    O=torch.tensor(obs,dtype=torch.float32,device="cuda");W=torch.tensor(w,dtype=torch.float32,device="cuda")
                    return m.critic_body(m._with_w(O,W)).cpu().numpy()
            Sf=feats(Sobs,Sw)
            rng=np.random.default_rng(123)
            repl={}
            for grp,rows in (("endpoint",sem_ep),("continuum",sem_ct),("all",sem_ep+sem_ct)):
                Qobs,Qw=cat(rows,"obs"),cat(rows,"w")
                base=nn_stats(Sf,feats(Qobs,Qw));repl[grp]={"baseline_feature_oos":base["oos_fraction"],"blocks":{}}
                # nearest random support donor, fixed seed; test each raw block replacement independently
                donor=rng.integers(0,len(Sobs),size=len(Qobs))
                for name,sl in BLOCKS.items():
                    X=Qobs.copy();X[:,sl]=Sobs[donor,sl]
                    st=nn_stats(Sf,feats(X,Qw))
                    repl[grp]["blocks"][name]={"feature_oos":st["oos_fraction"],
                        "oos_drop":float(base["oos_fraction"]-st["oos_fraction"])}
                W=Qw.copy();W[:]=Sw[donor]
                st=nn_stats(Sf,feats(Qobs,W));repl[grp]["blocks"]["preference"]={"feature_oos":st["oos_fraction"],
                    "oos_drop":float(base["oos_fraction"]-st["oos_fraction"])}
            result["replacement_attribution"]=repl
            all_drops=repl["all"]["blocks"];ranking=sorted(all_drops.items(),key=lambda kv:kv[1]["oos_drop"],reverse=True)
            result["summary"]={"all_feature_baseline_oos":repl["all"]["baseline_feature_oos"],
                               "replacement_ranking":[{"block":k,**v} for k,v in ranking],
                               "top_driver":ranking[0][0]}
            out=a.output_dir/"attribution_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":hashlib.sha256(out.read_bytes()).hexdigest()},indent=2)+"\n")
            print(json.dumps(result["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "rv1_a_implementation_gate": run_rv1_a_implementation_gate,
    "rv1_conditional_matched_donor_attribution": run_rv1_conditional_matched_donor_attribution,
    "rv1_fixed_feature_mapping_capacity_audit": run_rv1_fixed_feature_mapping_capacity_audit,
    "rv1_local_target_mapping_audit": run_rv1_local_target_mapping_audit,
    "rv1_observation_block_attribution_audit": run_rv1_observation_block_attribution_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
