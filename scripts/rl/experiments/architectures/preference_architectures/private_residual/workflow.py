"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_v2k0_function_preserving_gate():
    """Run former v2k0_function_preserving_gate.py stage."""
    
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    V2B_INIT = ROOT / "runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
    OUT = ROOT / "runs/v2k0_function_preserving_gate-2026-09-24"
    
    WREF = np.asarray([0.25,0.25,0.25,0.25],np.float32)
    PREFS = np.asarray([
        [0.70,0.10,0.10,0.10],
        [0.10,0.70,0.10,0.10],
        [0.10,0.10,0.70,0.10],
        [0.10,0.10,0.10,0.70],
        [0.25,0.25,0.25,0.25]],np.float32)
    NENV=8
    STEPS=64
    TOL=1e-6
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def ma(x):
        return float(torch.max(torch.abs(x)).detach().cpu()) if x.numel() else 0.0
    
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda:f.read(1<<20),b""): h.update(block)
        return h.hexdigest()
    
    def state_hash(model):
        h=hashlib.sha256()
        for n,t in model.state_dict().items():
            h.update(n.encode()); h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    
    def scalarization_check():
        from talon_rl.rewards.objectives import OBJECTIVE_ORDER,NORMALIZATION_DIVISORS,normalize_objectives,scalarize
        raw=np.asarray([[1.7194554805755615,-0.15590913593769073,-0.01563369482755661,-0.08311229199171066]],np.float32)
        norm=normalize_objectives(raw); exp=np.asarray([[1.,-1.,-1.,-1.]],np.float32)
        got=float(scalarize(norm,WREF)[0]); ref=float((exp*WREF).sum())
        err=float(np.max(np.abs(norm-exp)))
        return {"objective_order":list(OBJECTIVE_ORDER),"normalization_divisors":NORMALIZATION_DIVISORS.tolist(),
                "normalized_probe_max_abs_error":err,"scalarization_abs_error":abs(got-ref),
                "pass":bool(err<=1e-7 and abs(got-ref)<=1e-7)}
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output-dir",type=Path,default=OUT)
        ap.add_argument("--seed",type=int,default=424242)
        args=ap.parse_args()
        if not args.output_dir.is_absolute(): args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
    
        from isaaclab.app import AppLauncher
        saved=sys.argv[:]; sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app
        sys.argv=saved
        env=None
        try:
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            from talon_rl.models.conditioning.private_residual_residual import V2KHyperPrivateResidualActorCritic,initialize_from_v2b
    
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=NENV; cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=args.seed); obs=ot(obs).cuda()
            ad=env.unwrapped.action_manager.total_action_dim
    
            state=torch.load(V2B_INIT,map_location="cuda",weights_only=False)
            ref=V2BSingleSiteFiLMActorCritic(obs.shape[-1],ad).cuda()
            ref.load_state_dict(state["model"]); ref.eval()
    
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            hyp=V2KHyperPrivateResidualActorCritic(obs.shape[-1],ad).cuda()
            initialize_from_v2b(hyp,state); hyp.eval()
    
            w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
            wp=torch.tensor(PREFS,device="cuda")
            with torch.no_grad():
                ref_f=ref._actor_features_v2a(obs,w)
                ref_mean=ref.actor_mean(ref_f)
                h_f=hyp._actor_features_v2a(obs,w)
                h_mean=hyp.actor_mean(h_f)
                ref_action=ref.act_inference_with_preference(obs,w)
                h_action=hyp.act_inference_with_preference(obs,w)
                ref_value=ref.value_with_preference(obs,w)
                h_value=hyp.value_with_preference(obs,w)
    
            checks={}
            checks["deterministic_action_identity"]={"max_abs_diff":ma(h_action-ref_action),"pass":ma(h_action-ref_action)<=TOL}
            checks["pre_tanh_mean_identity"]={"max_abs_diff":ma(h_mean-ref_mean),"pass":ma(h_mean-ref_mean)<=TOL}
            checks["critic_identity"]={"max_abs_diff":ma(h_value-ref_value),"pass":ma(h_value-ref_value)<=TOL}
    
            coeff=hyp.module_coefficients(wp)
            hfilm=hyp._film_hidden(obs,w)
            module_outs=hyp.private_module_outputs(hfilm)
            combined=hyp._combined_private_residual(hfilm,w)
            checks["zero_coefficient_identity_init"]={
                "coefficient_max_abs":ma(coeff),
                "combined_private_residual_max_abs":ma(combined),
                "pass":ma(coeff)==0.0 and ma(combined)==0.0}
            mod_norm=[float(torch.linalg.vector_norm(module_outs[:,j],dim=-1).mean().cpu()) for j in range(hyp.NUM_MODULES)]
            div=[]
            for i in range(hyp.NUM_MODULES):
                for j in range(i+1,hyp.NUM_MODULES):
                    div.append(float(torch.linalg.vector_norm(module_outs[:,i]-module_outs[:,j],dim=-1).mean().cpu()))
            checks["private_modules_live_and_diverse"]={
                "mean_output_norms":mod_norm,
                "pairwise_diversity_mean":float(np.mean(div)),
                "pairwise_diversity_min":float(np.min(div)),
                "pass":min(mod_norm)>0.0 and float(np.min(div))>0.0}
    
            torch.manual_seed(args.seed+1)
            with torch.no_grad():
                dist=ref._pre_tanh_dist_with_preference(obs,w); latent=dist.sample()
                rlp=ref.logp_from_pre_tanh_with_preference(obs,w,latent)
                hlp=hyp.logp_from_pre_tanh_with_preference(obs,w,latent)
            checks["same_latent_logprob_identity"]={"max_abs_diff":ma(hlp-rlp),"pass":ma(hlp-rlp)<=TOL}
    
            with torch.no_grad():
                action,old,latent2=hyp.act_with_preference_latent(obs,w)
                new=hyp.logp_from_pre_tanh_with_preference(obs,w,latent2)
            ratio=torch.exp(new-old)
            checks["ppo_ratio_invariant"]={"max_abs_ratio_minus_1":ma(ratio-1),"pass":ma(ratio-1)<=1e-6}
    
            _=env.reset(seed=args.seed+7); env.step(action)
            raw=env.unwrapped.action_manager._terms["joint_pos"].raw_actions.detach()
            checks["env_raw_action_identity"]={"max_abs_diff":ma(raw-action),"pass":ma(raw-action)<=TOL}
    
            rs=ref.state_dict(); hs=hyp.state_dict()
            max_existing=0.0; exact=True
            for k,v in rs.items():
                if k in hs and hs[k].shape==v.shape:
                    d=float((hs[k]-v).abs().max().cpu()); max_existing=max(max_existing,d); exact=exact and d==0.0
            checks["v2b_parameter_identity"]={"max_abs_diff":max_existing,"pass":exact}
    
            ref_names=set(dict(ref.named_parameters())); h_names=set(dict(hyp.named_parameters()))
            new_names=sorted(h_names-ref_names)
            expected={"preference_coeff.0.weight","preference_coeff.0.bias",
                      "preference_coeff.2.weight","preference_coeff.2.bias"}
            for i in range(hyp.NUM_MODULES):
                expected.update({f"private_modules.{i}.fc1.weight",f"private_modules.{i}.fc1.bias",
                                 f"private_modules.{i}.fc2.weight",f"private_modules.{i}.fc2.bias"})
            checks["treatment_isolation"]={"new_parameter_names":new_names,
                "expected_new_parameter_names":sorted(expected),"pass":set(new_names)==expected}
    
            checkpoint=args.output_dir/"v2k0_init.pt"; before=state_hash(hyp)
            torch.save({"model":hyp.state_dict(),"reference":"v2b0_init","training_steps":0,
                        "architecture":"V2-B + hyper-conditioned private residual modules"},checkpoint)
            restored=V2KHyperPrivateResidualActorCritic(obs.shape[-1],ad).cuda()
            restored.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=False)["model"]); restored.eval()
            with torch.no_grad():
                ra=restored.act_inference_with_preference(obs,w); rv=restored.value_with_preference(obs,w)
            checks["checkpoint_roundtrip"]={"action_max_abs_diff":ma(ra-h_action),"value_max_abs_diff":ma(rv-h_value),
                "state_hash_equal":before==state_hash(restored),
                "pass":ma(ra-h_action)<=TOL and ma(rv-h_value)<=TOL and before==state_hash(restored)}
    
            checks["objective_contract"]=scalarization_check()
            checks["all_finite"]={"pass":bool(torch.isfinite(h_action).all() and torch.isfinite(h_value).all() and torch.isfinite(old).all())}
    
            cur,_=env.reset(seed=args.seed+99); cur=ot(cur).cuda(); maxdiff=0.0; terms=0; finite=True
            for _ in range(STEPS):
                with torch.no_grad():
                    ha=hyp.act_inference_with_preference(cur,w); va=ref.act_inference_with_preference(cur,w)
                maxdiff=max(maxdiff,ma(ha-va)); finite=finite and bool(torch.isfinite(ha).all())
                nxt,_,te,tr,_=env.step(ha); terms+=int((te|tr).sum().item()); cur=ot(nxt).cuda()
            checks["no_update_smoke"]={"steps":STEPS,"num_envs":NENV,"max_action_diff_vs_v2b":maxdiff,
                "termination_events":terms,"all_finite":finite,"pass":maxdiff<=TOL and finite}
    
            passed=all(v.get("pass",False) for v in checks.values())
            report={"schema":"v2k0_function_preserving_gate_v1",
                    "status":"V2-K0 PASS" if passed else "V2-K0 FAIL",
                    "training_enabled":False,"optimizer_steps":0,
                    "reference_checkpoint":str(V2B_INIT.relative_to(ROOT)),
                    "treatment":"preference-conditioned coefficients over four unlabeled private residual modules",
                    "checks":checks,
                    "authorization":{"v2k1_authorized":bool(passed),"training_in_this_gate":False},
                    "checkpoint":{"path":str(checkpoint.relative_to(ROOT)),"sha256":sha(checkpoint)},
                    "provenance":{"v2b_init_sha256":sha(V2B_INIT),
                        "v2k_module_sha256":sha(ROOT/"talon_rl/models/conditioning/private_residual.py"),
                        "gate_script_sha256":sha(Path(__file__).resolve()),
                        "contract_sha256":sha(ROOT/"docs/contracts/preference_architectures/v2k-contract.md")}}
            out=args.output_dir/"v2k0_report.json"; out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({"status":report["status"],"authorization":report["authorization"],"checks":checks},indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    
    if True: main()

def run_v2k1_private_subspace_screen():
    """Run former v2k1_private_subspace_screen.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    V2K_INIT=ROOT/"runs/v2k0_function_preserving_gate-2026-09-24/v2k0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2k1_private_subspace-2026-09-24"
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
    
    def _svd_summary(X):
        X=np.asarray(X,np.float64)
        X=X-X.mean(0,keepdims=True)
        sv=np.linalg.svd(X,compute_uv=False)
        ratio=float(sv[1]/(sv[0]+1e-12)) if len(sv)>1 else 0.0
        eff=int(np.sum(sv >= (sv[0]*0.05 if len(sv) and sv[0]>0 else np.inf)))
        return {"singular_values":sv.tolist(),"s2_over_s1":ratio,"effective_rank_5pct":eff}
    
    def sensitivity(m,probe):
        total={};masked={};coeffs={};residuals={};shares={};single_mask_effects={};module_out_norms={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                h=m._film_hidden(probe,w);e=m.preference_embedding(w)
                c=m.module_coefficients(w);outs=m.private_module_outputs(h)
                contrib=c.unsqueeze(-1)*outs
                r=contrib.sum(1)
                total[lab]=torch.tanh(m.actor_mean(torch.cat((h+r,e),dim=-1)))*m.ACTION_CLIP
                masked[lab]=torch.tanh(m.actor_mean(torch.cat((h,e),dim=-1)))*m.ACTION_CLIP
                coeffs[lab]=c;residuals[lab]=r
                cn=torch.linalg.vector_norm(contrib,dim=-1)
                shares[lab]=(cn/(cn.sum(-1,keepdim=True)+1e-12)).mean(0).cpu().tolist()
                module_out_norms[lab]=torch.linalg.vector_norm(outs,dim=-1).mean(0).cpu().tolist()
                effects=[]
                for j in range(m.NUM_MODULES):
                    rj=r-contrib[:,j]
                    aj=torch.tanh(m.actor_mean(torch.cat((h+rj,e),dim=-1)))*m.ACTION_CLIP
                    effects.append(float(torch.linalg.vector_norm(total[lab]-aj,dim=-1).mean().cpu()))
                single_mask_effects[lab]=effects
    
        pair_total=pairwise_dist(total);pair_masked=pairwise_dist(masked)
        authority=float(np.mean([torch.linalg.vector_norm(total[k]-masked[k],dim=1).mean().cpu() for k in ORDER]))
        heavy=("T","A","O","S")
        cmean={k:coeffs[k].mean(0).cpu().numpy() for k in ORDER}
        rflat={k:residuals[k].reshape(-1).cpu().numpy() for k in ORDER}
    
        coeff_pair=[];resid_pair=[];cos_pairs=[]
        for i,a in enumerate(heavy):
            for b in heavy[i+1:]:
                coeff_pair.append(float(np.linalg.norm(cmean[a]-cmean[b])))
                ra=rflat[a];rb=rflat[b]
                resid_pair.append(float(np.linalg.norm(ra-rb)/np.sqrt(len(probe))))
                cos=float(np.dot(ra,rb)/(np.linalg.norm(ra)*np.linalg.norm(rb)+1e-12));cos_pairs.append(cos)
        coeff_svd=_svd_summary(np.stack([cmean[k] for k in heavy]))
        resid_svd=_svd_summary(np.stack([rflat[k] for k in heavy]))
        cosmat=np.eye(4)
        for i,a in enumerate(heavy):
            for j,b in enumerate(heavy):
                ra=rflat[a];rb=rflat[b]
                cosmat[i,j]=float(np.dot(ra,rb)/(np.linalg.norm(ra)*np.linalg.norm(rb)+1e-12))
        share_mat=np.asarray([shares[k] for k in heavy],float)
        share_d=[]
        for i in range(4):
            for j in range(i+1,4):share_d.append(float(np.linalg.norm(share_mat[i]-share_mat[j])))
        mask_mat=np.asarray([single_mask_effects[k] for k in heavy],float)
    
        # Action preference Jacobians.
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w);rows=[]
        for j in range(a.shape[1]):rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        hm=m._film_hidden(probe,wm);em=m.preference_embedding(wm)
        am=torch.tanh(m.actor_mean(torch.cat((hm,em),dim=-1)))*m.ACTION_CLIP;mrows=[]
        for j in range(am.shape[1]):mrows.append(torch.autograd.grad(am[:,j].mean(),wm,retain_graph=True)[0])
        mjac=torch.stack(mrows,dim=1)
    
        # Local coefficient Jacobian at center.
        wc=torch.tensor(PREFS["C"],device="cuda",requires_grad=True)
        def cf(x):return m.module_coefficients(x.unsqueeze(0)).squeeze(0)
        Jc=torch.autograd.functional.jacobian(cf,wc,create_graph=False).detach()
        sc=torch.linalg.svdvals(Jc).cpu().numpy()
        jc={"fro_norm":float(torch.linalg.matrix_norm(Jc,ord="fro").cpu()),"singular_values":sc.tolist(),
            "s2_over_s1":float(sc[1]/(sc[0]+1e-12)) if len(sc)>1 else 0.0,
            "effective_rank_5pct":int(np.sum(sc >= (sc[0]*.05 if sc[0]>0 else np.inf)))}
    
        # Raw private-module output diversity independent of coefficients.
        div=[]
        with torch.no_grad():
            for lab in heavy:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                h=m._film_hidden(probe,w);outs=m.private_module_outputs(h)
                for i in range(m.NUM_MODULES):
                    for j in range(i+1,m.NUM_MODULES):
                        div.append(float(torch.linalg.vector_norm(outs[:,i]-outs[:,j],dim=-1).mean().cpu()))
    
        return {
          "pairwise_action_distance":pair_total,
          "private_masked_pairwise_action_distance":pair_masked,
          "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
          "private_masked_jacobian_fro_mean":float(torch.linalg.matrix_norm(mjac,ord="fro",dim=(1,2)).mean().detach().cpu()),
          "private_action_authority_mean":authority,
          "coefficient_matrix":{k:cmean[k].tolist() for k in ORDER},
          "coefficient_pairwise_distance_heavy":{"mean":float(np.mean(coeff_pair)),"min":float(np.min(coeff_pair)),"max":float(np.max(coeff_pair))},
          "coefficient_svd_heavy":coeff_svd,
          "coefficient_jacobian":jc,
          "private_residual_pairwise_distance_heavy":{"mean":float(np.mean(resid_pair)),"min":float(np.min(resid_pair)),"max":float(np.max(resid_pair))},
          "private_residual_cosine_matrix_heavy":{"rows":list(heavy),"values":cosmat.tolist()},
          "private_residual_direction_diversity":{"mean_one_minus_cos":float(np.mean([1-x for x in cos_pairs])),"min_one_minus_cos":float(np.min([1-x for x in cos_pairs])),"max_one_minus_cos":float(np.max([1-x for x in cos_pairs]))},
          "private_residual_svd_heavy":resid_svd,
          "module_contribution_shares":{"rows":list(heavy),"values":share_mat.tolist()},
          "module_share_pairwise_distance":{"mean":float(np.mean(share_d)),"min":float(np.min(share_d)),"max":float(np.max(share_d))},
          "max_module_share":float(share_mat.max()),
          "modules_above_0p10_any_pref":int(np.sum(np.max(share_mat,axis=0)>0.10)),
          "single_module_mask_effect_matrix":{"rows":list(heavy),"values":mask_mat.tolist()},
          "single_module_mask_pref_std_mean":float(mask_mat.std(axis=0).mean()),
          "private_module_output_norms":module_out_norms,
          "private_module_output_diversity":{"mean":float(np.mean(div)),"min":float(np.min(div)),"max":float(np.max(div))},
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
            from talon_rl.models.conditioning.private_residual_residual import V2KHyperPrivateResidualActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2KHyperPrivateResidualActorCritic(o.shape[-1],ad).cuda()
            init_path=V2K_INIT
            m.load_state_dict(torch.load(init_path,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film") or n.startswith("preference_coeff") or n.startswith("private_modules")]
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
                outw=m.preference_coeff[2].weight.grad
                outb=m.preference_coeff[2].bias.grad
                hgrad=float(torch.sqrt((outw.detach()**2).sum()+(outb.detach()**2).sum()).cpu()) if outw is not None and outb is not None else 0.0
                module_grads=[]
                for mod in m.private_modules:
                    vals=[p.grad.detach().pow(2).sum() for p in mod.parameters() if p.grad is not None]
                    module_grads.append(float(torch.sqrt(sum(vals)).cpu()) if vals else 0.0)
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                post_refresh=None
                # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
                if uidx in SNAPS:
                    post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                             "preference_input_grad_norm":pg,"coefficient_output_grad_norm":hgrad,"private_module_grad_norms":module_grads,"actor_grad_norm_preclip":total,
                             "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
                if uidx in SNAPS:audit(uidx)
    
            final=snaps[str(args.updates)];initial=snaps["0"]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            cg=np.array([r["coefficient_output_grad_norm"] for r in rows])
            mg=np.asarray([r["private_module_grad_norms"] for r in rows],float)
            crit=final["phase_critic"];sens=final["sensitivity"]
            pair_total=sens["pairwise_action_distance"]["mean"]
            pair_mask=sens["private_masked_pairwise_action_distance"]["mean"]
            jac_total=sens["jacobian_fro_mean"]
            jac_mask=sens["private_masked_jacobian_fro_mean"]
            mask_reduces=(pair_total>pair_mask+1e-4) or (jac_total>jac_mask+1e-4)
    
            criteria={
              "coefficient_preference_separation":sens["coefficient_pairwise_distance_heavy"]["mean"]>=0.02,
              "coefficient_manifold_rank_gt1":sens["coefficient_svd_heavy"]["s2_over_s1"]>=0.05,
              "coefficient_jacobian_nonzero":sens["coefficient_jacobian"]["fro_norm"]>1e-3,
              "private_residual_preference_separation":sens["private_residual_pairwise_distance_heavy"]["mean"]>=0.01,
              "private_residual_direction_diversity":sens["private_residual_direction_diversity"]["mean_one_minus_cos"]>=0.02,
              "private_residual_manifold_rank_gt1":sens["private_residual_svd_heavy"]["s2_over_s1"]>=0.05,
              "multiple_modules_contribute":sens["modules_above_0p10_any_pref"]>=2,
              "module_share_preference_separation":sens["module_share_pairwise_distance"]["mean"]>=0.02,
              "private_action_authority_nonzero":sens["private_action_authority_mean"]>1e-4,
              "private_mask_reduces_preference_authority":bool(mask_reduces),
              "single_module_mask_preference_specific":sens["single_module_mask_pref_std_mean"]>1e-5,
              "coefficient_output_gradient_observed":float(np.max(cg))>1e-7,
              "private_module_gradient_observed":float(np.max(mg))>1e-7,
              "private_modules_remain_diverse":sens["private_module_output_diversity"]["mean"]>1e-3,
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
    
            report={"schema":"v2k1_private_subspace_authority_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"V2-K1 private-subspace authority/diversity only; semantics blocked",
                    "actor_ppo_objectives_changed":False,"architecture":"v2k",
                    "reference_checkpoint":str(V2K_INIT.relative_to(ROOT)),
                    "anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            report["summary"]={
              "status":"V2-K1 PASS" if passed else "V2-K1 FAIL",
              "criteria":criteria,"initial":initial,"final":final,
              "max_coefficient_output_grad_norm":float(np.max(cg)),
              "median_coefficient_output_grad_norm":float(np.median(cg)),
              "max_private_module_grad_norms":mg.max(axis=0).tolist(),
              "median_private_module_grad_norms":np.median(mg,axis=0).tolist(),
              "last10_termination_fraction":float(term[-10:].mean()),
              "max_ratio_error":float(ratio.max()),
              "v2k2_authorized":bool(passed),
              "semantic_judgement_performed":False}
            (args.output_dir/"v2k1_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2k2_semantic_eval():
    """Run former v2k2_semantic_eval.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=None
    OUT_DEFAULT=ROOT/"runs/foundation_v2_semantic_eval"
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
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT)
        args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        if not args.checkpoint.is_absolute():args.checkpoint=(ROOT/args.checkpoint).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            from talon_rl.models.conditioning.private_residual_residual import V2KHyperPrivateResidualActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            m=V2KHyperPrivateResidualActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(args.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
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
            report={"schema":"v2k2_semantic_eval_v1","status":"V2-K2 PASS" if passed else "V2-K2 FAIL",
                "measurement_only":True,"training_updates":0,"checkpoint":str(args.checkpoint.relative_to(ROOT)),
                "protocol":{"matched_reset":True,"suites":SUITES,"steps":STEPS,"alphas":list(ALPHAS),
                            "paths":[f"{a}-{b}" for a,b in PATHS],"thresholds":{"endpoint_fraction":.75,"continuum_fraction":.65,
                            "min_survival":.95,"max_tracking_ratio_to_center":2.0}},
                "endpoint":endpoint,"endpoint_pass":endpoint_pass,"continuum_summary":{"monotonicity_fraction":mono,
                            "endpoint_between_fraction":between,"details":cont_stats},
                "collateral":{"center_tracking_error":center_track,"max_tracking_ratio_to_center":max_track_ratio},
                "critic":critic,"min_survival_all":float(np.min(all_surv)),"criteria":criteria,
                "decision":{"semantic_pass":bool(passed),"architecture":"v2k","v2k3_authorized":bool(passed)},
                "endpoint_rows":rows,"continuum_rows":continuum}
            out=args.output_dir/"semantic_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:report[k] for k in ("status","endpoint","endpoint_pass","continuum_summary","collateral","critic","min_survival_all","criteria","decision") if k in report and k!="continuum_summary"} | {"continuum_summary":{"monotonicity_fraction":mono,"endpoint_between_fraction":between}},indent=2))
            prov={"status":"FROZEN_BY_HASH","gate":"FOUNDATION-V2-SEMANTIC","verdict":report["status"],"measurement_only":True,
                  "architecture":"v2k",
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},
                               str(args.checkpoint.relative_to(ROOT)):{"sha256":sha(args.checkpoint)},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())}}}
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "v2k0_function_preserving_gate": run_v2k0_function_preserving_gate,
    "v2k1_private_subspace_screen": run_v2k1_private_subspace_screen,
    "v2k2_semantic_eval": run_v2k2_semantic_eval,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
