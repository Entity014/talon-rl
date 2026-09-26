"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_objective_set_g1_c0_critic_isolation():
    """Run former objective_set_g1_c0_critic_isolation.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.objective_set_g1_train as g1
    import rl.experiments.common.utilities.authority_isolated_h2a_u30_semantic_validity as h2
    from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic
    INIT=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
    CONTRACT=ROOT/"docs/contracts/objective_set/objective-set-g1-c0-critic-isolation-contract.md"
    OUT=ROOT/"runs/objective_set_g1_c0_critic_isolation-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    RUNS=[(f,s) for f in ("G1-2","G1-3") for s in (73101,73102,73103)]
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def actor_snapshot(m):
        return {n:p.detach().cpu().clone() for n,p in m.named_parameters() if n.startswith("actor_") or n.startswith("family_") or n=="log_std"}
    def actor_drift(a,b):
        return max(float((a[k]-b[k]).abs().max()) for k in a)
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=g1.NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            init_state=torch.load(INIT,map_location="cuda",weights_only=False)["model"];rows=[]
            for fold,seed in RUNS:
                torch.manual_seed(seed);np.random.seed(seed)
                m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();m.load_state_dict(init_state);m.train()
                before=actor_snapshot(m)
                for n,p in m.named_parameters():
                    if n.startswith("actor_") or n.startswith("family_") or n=="log_std":p.requires_grad_(False)
                cp=[p for n,p in m.named_parameters() if n.startswith("critic_body") or n.startswith("value_basis")]
                opt=torch.optim.Adam(cp,lr=1e-3);losses=[];cards=[]
                for u in range(1,31):
                    b=g1.collect(env,mgr,m,fold,seed,u);cards.append(b["card"])
                    assert b["card"] in g1.FOLDS[fold]["train"] and b["card"]!=g1.FOLDS[fold]["holdout"]
                    adv,ret=g1.gae(b["r"],b["v"],b["nextv"],b["d"])
                    B=g1.H*g1.NENV;M=b["card"];obs=b["obs"].reshape(B,-1)
                    tok=b["tok"].repeat(g1.H,1,1);w=b["w"].repeat(g1.H,1)
                    cv=m.query_values_from_set(obs.detach(),tok.detach(),w.detach(),tok.detach())
                    loss=(cv-ret.reshape(B,M).detach()).pow(2).mean()
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(cp,1.0);opt.step();losses.append(float(loss.detach().cpu()))
                after=actor_snapshot(m);dr=actor_drift(before,after)
                cev=[];cb=[]
                m.eval()
                for suite in range(4):
                    rs=840001+suite
                    for lab in ("T","A","O","S","C"):
                        q=h2.evaluate(env,m,mgr,robot,h2.PREFS[lab],rs)
                        cev+=q["critic"]["first_h32_ev"]+q["critic"]["second_h32_ev"];cb+=q["critic"]["first_h32_bias"]+q["critic"]["second_h32_bias"]
                critic={"ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),"mean_abs_bias":float(np.mean(np.abs(cb)))}
                rec={"fold":fold,"seed":seed,"actor_max_abs_drift":dr,"cards_seen":sorted(set(cards)),"mean_critic_loss":float(np.mean(losses)),
                     "final_critic_loss":losses[-1],"critic":critic,
                     "pass":bool(dr==0.0 and critic["ev_mean"]>0 and critic["negative_fraction"]<=.25)}
                rows.append(rec);print(fold,seed,json.dumps(rec),flush=True)
                torch.save({"model":m.state_dict(),"fold":fold,"seed":seed},OUT/f"{fold.lower().replace('-','_')}_seed{seed}_model30.pt")
            rep={"schema":"objective_set_g1_c0_critic_isolation_v1","rows":rows,
                 "summary":{"pass_runs":sum(x["pass"] for x in rows),"total_runs":len(rows),"all_actor_exact":all(x["actor_max_abs_drift"]==0 for x in rows),
                            "all_critic_pass":all(x["pass"] for x in rows)}}
            rp=OUT/"critic_isolation.json";rp.write_text(json.dumps(rep,indent=2)+"\n")
            (OUT/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","contract_sha256":sha(CONTRACT),"init_sha256":sha(INIT),
              "script_sha256":sha(Path(__file__).resolve()),"report_sha256":sha(rp)},indent=2)+"\n")
            print("FINAL",json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_objective_set_g1_c1_train():
    """Run former objective_set_g1_c1_train.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,itertools,json,sys
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic,canonical_tokens
    from talon_rl.rewards.objectives import normalized_objective_vector
    import rl.experiments.common.utilities.objective_set_g1_train as g1
    INIT=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
    FOLDS=g1.FOLDS;NENV=g1.NENV;H=g1.H;EPS=g1.EPS
    SUP_H=64;RIDGE=1.0
    
    def ridge(F,Y,lam=1.0):
        X=np.c_[F,np.ones(len(F))];I=np.eye(X.shape[1]);I[-1,-1]=0
        B=np.linalg.solve(X.T@X+lam*I,X.T@Y);return B[:-1],B[-1]
    
    def support_batch(env,mgr,model,fold,card,seed,update):
        rng=np.random.default_rng(seed*31337+update*1009+card*17)
        combos=list(itertools.combinations(range(4),card));ids=[];ws=[]
        for lane in range(NENV):
            sub=np.array(combos[(lane+update)%len(combos)],np.int64);ids.append(sub)
            mode=lane%(card+1)
            if mode==0:w=np.full(card,1/card,np.float32)
            else:
                h=(mode-1)%card;w=np.full(card,.30/(card-1),np.float32);w[h]=.70
            ws.append(w)
        ids=torch.tensor(np.stack(ids),device="cuda",dtype=torch.long)
        tok=canonical_tokens(device="cuda")[ids];w=torch.tensor(np.stack(ws),device="cuda")
        obs,_=env.reset(seed=seed+500000+update*97+card);obs=g1.ot(obs).cuda()
        O=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                O.append(obs);a=model.act_inference_from_set(obs,tok,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                full=torch.tensor(normalized_objective_vector(g1.terms(raw,names),shape=(NENV,)),device="cuda")*env.unwrapped.step_dt
                R.append(torch.gather(full,1,ids));D.append((te|tr).cuda());obs=g1.ot(nxt).cuda()
            O=torch.stack(O);R=torch.stack(R);D=torch.stack(D).bool()
            Y=torch.zeros_like(R);run=torch.zeros_like(R[0])
            for t in range(SUP_H-1,-1,-1):
                run=R[t]+g1.GAMMA*run*(~D[t]).to(R.dtype).unsqueeze(-1);Y[t]=run
            F=model.critic_features_from_set(O.reshape(-1,O.shape[-1]),tok.repeat(SUP_H,1,1),w.repeat(SUP_H,1)).reshape(SUP_H,NENV,-1)
        return ids,F.cpu().numpy(),Y.cpu().numpy()
    def refresh_critic(env,mgr,model,fold,seed,update):
        feats={i:[] for i in range(4)};ys={i:[] for i in range(4)}
        for card in FOLDS[fold]["train"]:
            ids,F,Y=support_batch(env,mgr,model,fold,card,seed,update)
            ids=ids.cpu().numpy()
            for lane in range(NENV):
                for j,obj in enumerate(ids[lane]):
                    feats[int(obj)].append(F[:,lane,:]);ys[int(obj)].append(Y[:,lane,j])
        with torch.no_grad():
            for obj in range(4):
                if not feats[obj]:continue
                FF=np.concatenate(feats[obj],0);YY=np.concatenate(ys[obj],0);W,b=ridge(FF,YY,RIDGE)
                model.value_basis_weight[obj].copy_(torch.tensor(W,device="cuda",dtype=model.value_basis_weight.dtype))
                model.value_basis_bias[obj].copy_(torch.tensor(b,device="cuda",dtype=model.value_basis_bias.dtype))
    
    def collect_main(env,mgr,model,fold,seed,update):
        card,ids,tok,w,modes=g1.sample_sets(fold,seed,update,torch.device("cuda"))
        obs,_=env.reset(seed=seed+update*211);obs=g1.ot(obs).cuda()
        O=[];U=[];LP=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,u=model.act_with_set_latent(obs,tok,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                full=torch.tensor(normalized_objective_vector(g1.terms(raw,names),shape=(NENV,)),device="cuda")*env.unwrapped.step_dt
                O.append(obs);U.append(u);LP.append(lp);R.append(torch.gather(full,1,ids));D.append((te|tr).cuda());obs=g1.ot(nxt).cuda()
        return {"card":card,"ids":ids,"tok":tok,"w":w,"modes":modes,"obs":torch.stack(O),"u":torch.stack(U),
                "oldlp":torch.stack(LP),"r":torch.stack(R),"d":torch.stack(D).bool(),"next_obs":obs,
                "termination_fraction":float(torch.stack(D).any(0).float().mean().cpu())}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--fold",choices=tuple(FOLDS),required=True);ap.add_argument("--seed",type=int,default=73101);ap.add_argument("--updates",type=int,default=10)
        a=ap.parse_args()
        if a.updates!=10:raise ValueError("C1 screen frozen at 10 updates")
        out=ROOT/f"runs/objective_set_{a.fold.lower().replace('-','_')}_c1_seed{a.seed}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(a.seed);np.random.seed(a.seed)
        tau=torch.tensor(json.load(open(g1.TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=a.seed;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=a.seed);o=g1.ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            state=torch.load(INIT,map_location="cuda",weights_only=False)["model"]
            model=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();model.load_state_dict(state);model.train()
            ref=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();ref.load_state_dict(state);ref.eval()
            for p in ref.parameters():p.requires_grad_(False)
            for n,p in model.named_parameters():
                if n.startswith("critic_body") or n.startswith("value_basis"):p.requires_grad_(False)
            actor_params=[p for n,p in model.named_parameters() if n.startswith("actor_") or n.startswith("family_") or n=="log_std"]
            opt=torch.optim.Adam(actor_params,lr=1e-3);rows=[]
            torch.save({"model":model.state_dict(),"update":0},out/"model_0.pt")
            for uidx in range(1,a.updates+1):
                main=collect_main(env,mgr,model,a.fold,a.seed,uidx)
                refresh_critic(env,mgr,model,a.fold,a.seed,uidx)
                card=main["card"];B=H*NENV;M=card
                obs=main["obs"].reshape(B,-1);tok=main["tok"].repeat(H,1,1);w=main["w"].repeat(H,1)
                with torch.no_grad():
                    v=model.query_values_from_set(obs,tok,w,tok).reshape(H,NENV,M)
                    nv=model.query_values_from_set(main["next_obs"],main["tok"],main["w"],main["tok"])
                    adv,_=g1.gae(main["r"],v,nv,main["d"])
                u=main["u"].reshape(B,-1);old=main["oldlp"].reshape(B)
                lp=model.logp_from_pre_tanh_from_set(obs,tok,w,u);ratio=torch.exp(lp-old.detach());ratio_err=float((ratio-1).abs().max().cpu())
                aa=adv.reshape(B,M).detach();cl=ratio.clamp(1-g1.CLIP,1+g1.CLIP);po=torch.minimum(ratio[:,None]*aa,cl[:,None]*aa)
                ppo=-(M*(w*po).sum(-1)).mean();tail,tf=g1.tail_metrics(model,obs,tok,w,tau)
                retain=g1.edge_loss(model,ref,main["obs"][0],main["ids"])
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True);gt=torch.autograd.grad(tail,actor_params,retain_graph=True,allow_unused=True);gr=torch.autograd.grad(retain,actor_params,allow_unused=True)
                gp2=sum((x.detach()**2).sum() for x in gp if x is not None);gt2=sum((x.detach()**2).sum() for x in gt if x is not None)
                dot=sum((x.detach()*y.detach()).sum() for x,y in zip(gp,gt) if x is not None and y is not None);coef=torch.minimum(dot/(gt2+EPS),torch.zeros((),device="cuda"))
                base=[];p2=torch.zeros((),device="cuda")
                for p,x,y in zip(actor_params,gp,gt):
                    x=torch.zeros_like(p) if x is None else x;y=torch.zeros_like(p) if y is None else y;q=x-coef*y;base.append([q,y]);p2+=(q.detach()**2).sum()
                gtn=g1.grad_norm(gt);scale=g1.KAPPA*torch.sqrt(p2+EPS)/(gtn+EPS);gb=[q+scale*y for q,y in base];gbn=g1.grad_norm(gb);grn=g1.grad_norm(gr)
                alpha=min(g1.BETA0,g1.RHO*float(gbn.cpu())/(float(grn.cpu())+EPS))
                opt.zero_grad(set_to_none=True)
                for p,x,r in zip(actor_params,gb,gr):p.grad=x+(alpha*r if r is not None else 0)
                torch.nn.utils.clip_grad_norm_(actor_params,1.0);opt.step()
                with torch.no_grad():model.log_std.clamp_(-1.6,0)
                rec={"update":uidx,"cardinality":card,"ppo_loss":float(ppo.detach().cpu()),"retain_loss":float(retain.detach().cpu()),"retain_alpha":alpha,
                     "ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"]}
                rows.append(rec);print("UPDATE",a.fold,uidx,json.dumps(rec),flush=True)
            torch.save({"model":model.state_dict(),"update":10,"fold":a.fold,"seed":a.seed},out/"model_10.pt")
            (out/"training_report.json").write_text(json.dumps({"fold":a.fold,"seed":a.seed,"rows":rows,"leakage_free":True},indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_objective_set_g1_c1_u10_gate():
    """Run former objective_set_g1_c1_u10_gate.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h2a_u30_semantic_validity as h2
    from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic
    from rl.experiments.common.utilities.objective_set_g1_evaluate import authority,permutation_drift
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    G0=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
    OUT=ROOT/"runs/objective_set_g1_c1_u10_gate-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=h2.NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"];probe=torch.tensor(np.load(PROBE)["obs"],device="cuda")
            g0=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();g0.load_state_dict(torch.load(G0,map_location="cuda",weights_only=False)["model"]);g0.eval()
            report={}
            for fold in ("G1-2","G1-3"):
                ck=ROOT/f"runs/objective_set_{fold.lower().replace('-','_')}_c1_seed73101-2026-09-25/model_10.pt"
                m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
                rows=[]
                for suite in range(4):
                    seed=840001+suite
                    for lab in ("T","A","O","S","C"):
                        q=h2.evaluate(env,m,mgr,robot,h2.PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
                ep={}
                for lab in h2.ORDER:
                    j=h2.IDX[lab];pk=h2.PHYS[lab];oo=[];pp=[];sv=[]
                    for suite in range(4):
                        r=next(x for x in rows if x["suite"]==suite and x["label"]==lab);c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                        oo.append(r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]>0);pp.append(r["physical"][pk]-c["physical"][pk]<0);sv.append(r["survival"])
                    ep[lab]={"objective":float(np.mean(oo)),"physical":float(np.mean(pp)),"min_survival":float(min(sv)),
                             "pass":bool(np.mean(oo)>=.75 and np.mean(pp)>=.75 and min(sv)>=.95)}
                cev=[];cb=[]
                for x in rows:cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"];cb+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
                critic={"ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),"mean_abs_bias":float(np.mean(np.abs(cb)))}
                auth=authority(m,g0,probe,(0,1,2,3));perm=permutation_drift(m,probe,(0,1,2,3))
                critpass=critic["ev_mean"]>0 and critic["negative_fraction"]<=.25
                passed=all(ep[x]["pass"] for x in ("T","A","O")) and critpass and auth["pairwise_retention"]>=.75 and auth["tangent_retention"]>=.75 and max(perm.values())<=1e-6
                report[fold]={"endpoint":ep,"critic":critic,"authority":auth,"permutation":perm,"pass":bool(passed)}
                print(fold,json.dumps(report[fold]),flush=True)
            rep={"schema":"objective_set_g1_c1_u10_gate_v1","folds":report,"decision":{"full30_authorized":all(x["pass"] for x in report.values())}}
            (OUT/"u10_gate.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",rep["decision"],flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_objective_set_g1_c2_posthoc_critic_refit():
    """Run former objective_set_g1_c2_posthoc_critic_refit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h2a_u30_semantic_validity as h2
    import rl.experiments.common.utilities.objective_set_g1_c0_critic_substrate as c0
    from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic
    OUT=ROOT/"runs/objective_set_g1_c2_posthoc_critic_refit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def anchor(env,mgr,robot,m):
        rows=[]
        for suite in range(4):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=h2.evaluate(env,m,mgr,robot,h2.PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in h2.ORDER:
            j=h2.IDX[lab];pk=h2.PHYS[lab];oo=[];pp=[];sv=[]
            for suite in range(4):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab);c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                oo.append(r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]>0)
                pp.append(r["physical"][pk]-c["physical"][pk]<0);sv.append(r["survival"])
            ep[lab]={"objective":float(np.mean(oo)),"physical":float(np.mean(pp)),"survival":float(min(sv)),
                     "pass":bool(np.mean(oo)>=.75 and np.mean(pp)>=.75 and min(sv)>=.95)}
        evs=[];bs=[]
        for x in rows:evs+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"];bs+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
        cr={"ev_mean":float(np.mean(evs)),"negative_fraction":float(np.mean(np.asarray(evs)<0)),"mean_abs_bias":float(np.mean(np.abs(bs)))}
        return {"endpoint":ep,"TAO_pass":all(ep[x]["pass"] for x in ("T","A","O")),
                "critic":cr,"critic_pass":cr["ev_mean"]>0 and cr["negative_fraction"]<=.25}
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=h2.NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"];rep={}
            for fold in ("G1-2","G1-3"):
                ck=ROOT/f"runs/objective_set_{fold.lower().replace('-','_')}_c1_seed73101-2026-09-25/model_10.pt"
                m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
                actor_before={k:v.detach().clone() for k,v in m.state_dict().items() if k.startswith("actor_") or k.startswith("family_") or k=="log_std"}
                pre=anchor(env,mgr,robot,m)
                c0.fit_fold(env,mgr,fold,m)
                post=anchor(env,mgr,robot,m)
                drift=max(float((m.state_dict()[k]-v).abs().max().cpu()) for k,v in actor_before.items())
                rep[fold]={"pre":pre,"post":post,"actor_drift":drift}
                print(fold,json.dumps(rep[fold]),flush=True)
            (OUT/"posthoc_refit.json").write_text(json.dumps({"schema":"objective_set_g1_c2_posthoc_refit_v1","folds":rep},indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "objective_set_g1_c0_critic_isolation": run_objective_set_g1_c0_critic_isolation,
    "objective_set_g1_c1_train": run_objective_set_g1_c1_train,
    "objective_set_g1_c1_u10_gate": run_objective_set_g1_c1_u10_gate,
    "objective_set_g1_c2_posthoc_critic_refit": run_objective_set_g1_c2_posthoc_critic_refit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
