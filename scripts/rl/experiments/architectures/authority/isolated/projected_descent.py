"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_projected_descent_endpoint_audit():
    """Run former authority_isolated_projected_descent_endpoint_audit.py stage."""
    """Endpoint survival for the projected-tail-descent run at update 3."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.projected_endpoint_audit import ProjectedEndpointAudit
    
    
    class ProjectedDescentEndpointAudit(ProjectedEndpointAudit):
        """Endpoint survival for the projected-tail-descent run at update 3."""
    
        run = "authority_isolated_projected_tail_descent-2026-09-25"
        checkpoint = "model_3.pt"
        report = "projected_endpoint_audit.json"
    
    
    if True:
        ProjectedDescentEndpointAudit.main()

def run_authority_isolated_projected_descent_u10_endpoint_audit():
    """Run former authority_isolated_projected_descent_u10_endpoint_audit.py stage."""
    """Endpoint survival for the projected-tail-descent run at update 10."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.projected_endpoint_audit import ProjectedEndpointAudit
    
    
    class ProjectedDescentU10EndpointAudit(ProjectedEndpointAudit):
        """Endpoint survival for the projected-tail-descent run at update 10."""
    
        run = "authority_isolated_projected_tail_descent-2026-09-25"
        checkpoint = "model_10.pt"
        report = "projected_descent_u10_endpoint_audit.json"
    
    
    if True:
        ProjectedDescentU10EndpointAudit.main()

def run_authority_isolated_projected_endpoint_audit():
    """Run former authority_isolated_projected_endpoint_audit.py stage."""
    """Endpoint survival for the tail-projected-gradient run."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.projected_endpoint_audit import ProjectedEndpointAudit
    
    
    class ProjectedGradientEndpointAudit(ProjectedEndpointAudit):
        """Endpoint survival for the tail-projected-gradient run."""
    
        run = "authority_isolated_tail_projected_gradient-2026-09-25"
        checkpoint = "model_3.pt"
        report = "projected_endpoint_audit.json"
    
    
    if True:
        ProjectedGradientEndpointAudit.main()

def run_authority_isolated_projected_tail_descent_train():
    """Run former authority_isolated_projected_tail_descent_train.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,json,sys
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    BASE=ROOT/"runs/authority_isolated_critic_capacity-2026-09-25/wide_critic_fixed_policy.pt"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    NENV=h1.NENV;H=h1.H
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    def new_model(obs_dim,ad):
        m=AuthorityIsolatedWideCritic(obs_dim,ad).cuda()
        m.load_state_dict(torch.load(BASE,map_location="cuda",weights_only=False)["model"]);return m
    def audit(m,probe,tau,out,tag,snaps,args,lam):
        m.eval();sens=h1.sensitivity(m,probe)
        P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        wm=torch.tensor(h1.PREFS["C"],device="cuda").repeat(len(P),1)
        with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
        snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
        torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed,"arm":args.arm,"lambda_tail":lam},out/f"model_{tag}.pt")
        m.train()
    def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
          "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
          "numpy_rng":np.random.get_state()},path)
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["projected_descent"],default="projected_descent")
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--target-updates",type=int,default=3)
        ap.add_argument("--chunk-updates",type=int,default=3);args=ap.parse_args()
        lam=0.0;kappa=0.05
        out=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25"
        out.mkdir(parents=True,exist_ok=True);state_path=out/"resume_state.pt"
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=args.seed);o=h1.ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=new_model(o.shape[-1],ad);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False)
                m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
                print("RESUME",cur,flush=True)
                print("RESUME_STATE_LOADED",len(rows),list(snaps.keys()),flush=True)
            else:
                cand=[]
                for k in range(h1.ANCHOR_CANDIDATES):
                    _,w=h1.pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                    units=h1.collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
                comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
                spec_idx=h1.select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
                adaptive_pools={"early":[],"late":[]}
                for _,_,units in cand:
                    for u in units:adaptive_pools[u["phase"]].append(u)
                rows=[];snaps={};cur=0
                audit(m,probe,tau,out,0,snaps,args,lam)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur)
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
            end=min(args.target_updates,cur+args.chunk_updates)
            print("LOOP_RANGE",cur+1,end,flush=True)
            for uidx in range(cur+1,end+1):
                print("BEGIN_UPDATE",uidx,flush=True)
                labs,w=h1.pref_batch(uidx,torch.device("cuda"))
                print("BEFORE_COLLECT_ACTOR",uidx,args.seed+uidx*211,flush=True)
                main=h1.collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                print("AFTER_COLLECT_ACTOR",uidx,flush=True)
                with torch.no_grad():
                    lp_pre=m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])
                    ratio_pre=float((torch.exp(lp_pre-main["old"])-1).abs().max().cpu())
                    lp_chunks=[]
                    for st in range(0,len(main["obs"]),NENV):
                        lp_chunks.append(m.logp_from_pre_tanh_with_preference(main["obs"][st:st+NENV],main["w"][st:st+NENV],main["u"][st:st+NENV]))
                    lp_chunk=torch.cat(lp_chunks)
                    ratio_chunk=float((torch.exp(lp_chunk-main["old"])-1).abs().max().cpu())
                print("RATIO_PRE_REFRESH",uidx,ratio_pre,"CHUNK8",ratio_chunk,flush=True)
                _,ws=h1.pref_batch(uidx+17,torch.device("cuda"))
                print("BEFORE_COLLECT_SUPPORT",uidx,args.seed+200000+uidx*223,flush=True)
                units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                print("AFTER_COLLECT_SUPPORT",uidx,flush=True)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                print("BEFORE_ANCHOR_RECOLLECT",uidx,flush=True)
                anchors_now=current_anchor_units()
                print("AFTER_ANCHOR_RECOLLECT",uidx,flush=True)
                print("BEFORE_CRITIC_REFRESH",uidx,flush=True)
                h1.fit_expanded_current_policy(m,anchors_now,adaptive_pools)
                print("AFTER_CRITIC_REFRESH",uidx,flush=True)
                print("BEFORE_VALUE",uidx,flush=True)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    print("AFTER_VALUE",uidx,flush=True)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                    print("AFTER_GAE",uidx,flush=True)
                print("BEFORE_RATIO",uidx,flush=True)
                lp_parts=[]
                for st in range(0,len(main["obs"]),NENV):
                    lp_parts.append(m.logp_from_pre_tanh_with_preference(main["obs"][st:st+NENV],main["w"][st:st+NENV],main["u"][st:st+NENV]))
                ratio=torch.exp(torch.cat(lp_parts)-main["old"].detach())
                print("AFTER_RATIO",uidx,flush=True)
                print("BEFORE_RATIO_ERR",uidx,flush=True)
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                print("AFTER_RATIO_ERR",uidx,ratio_err,flush=True)
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                print("BEFORE_PPO_LOSS",uidx,flush=True)
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                print("AFTER_PPO_LOSS",uidx,flush=True)
                print("BEFORE_TAIL",uidx,flush=True)
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                print("AFTER_TAIL",uidx,flush=True)
                print("BEFORE_GRAD_BUDGET",uidx,flush=True)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None)
                gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2);gtn=torch.sqrt(gt2)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                cos=dot/(gpn*gtn+1e-12)
                coeff=torch.minimum(dot/(gt2+1e-12),torch.zeros_like(dot))
                proj_parts=[];proj2=torch.zeros((),device="cuda");removed2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a
                    bb=torch.zeros_like(p) if b is None else b
                    corr=coeff*bb
                    pp=aa-corr
                    proj_parts.append((pp,bb,corr))
                    proj2=proj2+(pp.detach()**2).sum()
                    removed2=removed2+(corr.detach()**2).sum()
                projn=torch.sqrt(proj2)
                tail_scale=kappa*projn/(gtn+1e-12)
                opt.zero_grad(set_to_none=True)
                for p,(pp,bb,corr) in zip(actor_params,proj_parts):
                    p.grad=pp+tail_scale*bb
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                print("AFTER_STEP",uidx,flush=True)
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                     "tail_fraction":float(tailfrac.detach().cpu()),"kappa":kappa,
                     "ppo_grad_norm":float(gpn.cpu()),"tail_grad_norm":float(gtn.cpu()),
                     "grad_cosine":float(cos.cpu()),"projection_coeff":float(coeff.cpu()),
                     "removed_grad_norm":float(torch.sqrt(removed2).cpu()),
                     "removed_over_ppo":float((torch.sqrt(removed2)/(gpn+1e-12)).cpu()),
                     "projected_grad_norm":float(projn.cpu()),"tail_descent_scale":float(tail_scale.cpu()),
                     "tail_descent_over_projected":float((tail_scale*gtn/(projn+1e-12)).cpu()),
                     "grad_norm_preclip":total,"ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"]}
                rows.append(row);print("UPDATE",uidx,json.dumps(row),flush=True)
                if uidx==args.target_updates:audit(m,probe,tau,out,uidx,snaps,args,lam)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx)
            print("CHUNK_DONE",end,flush=True)
            if end==args.target_updates:
                s0=snaps["0"]["sensitivity"];sf=snaps[str(end)]["sensitivity"]
                pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
                jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
                rep={"schema":"authority_isolated_projected_tail_descent_v1","arm":args.arm,"kappa":kappa,
                     "seed":args.seed,"updates":end,"tau_source":str(TAU_PATH.relative_to(ROOT)),
                     "base_checkpoint":str(BASE.relative_to(ROOT)),"rows":rows,"snapshots":snaps,
                     "summary":{"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
                       "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                       "probe_tail_loss_ratio":snaps[str(end)]["probe_tail_loss"]/(snaps["0"]["probe_tail_loss"]+1e-12),
                       "probe_tail_fraction_delta":snaps[str(end)]["probe_tail_fraction"]-snaps["0"]["probe_tail_fraction"]}}
                (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_tail_authority_gradient_geometry():
    """Run former authority_isolated_tail_authority_gradient_geometry.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK=ROOT/"runs/authority_isolated_coordinate_headroom_control-2026-09-25/model_0.pt"
    O=torch.tensor(np.load(ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz")["obs"],device="cuda")
    tau=torch.tensor(json.load(open(ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"))["tau"],device="cuda")
    P={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    m=AuthorityIsolatedWideCritic(O.shape[1],12).cuda()
    m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"])
    m.train()
    named=[(n,p) for n,p in m.named_parameters() if (n.startswith("actor_") or n.startswith("family_") or n=="log_std") and p.requires_grad]
    def flatgrad(loss):
        m.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        parts=[]
        for _,p in named:
            parts.append((p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1))
        return torch.cat(parts)
    Z=[];A={}
    for lab,wv in P.items():
        w=torch.tensor(wv,device="cuda").repeat(len(O),1)
        z=m._actor_mean_with_preference(O,w)
        Z.append(z);A[lab]=torch.tanh(z)
    z=torch.cat(Z)
    ex=torch.relu(z.abs()-tau)
    Ltail=((ex/(tau+1e-6))**2).mean()
    vals=[];labs=list(P)
    for i,a in enumerate(labs):
        for b in labs[i+1:]:
            vals.append(torch.linalg.vector_norm(A[a]-A[b],dim=1).mean())
    Lauth=-torch.stack(vals).mean()
    gt=flatgrad(Ltail);ga=flatgrad(Lauth)
    def cosine(x,y):
        return float((torch.dot(x,y)/(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y)+1e-12)).detach().cpu())
    groups={}
    off=0
    slices=[]
    for n,p in named:
        slices.append((n,off,off+p.numel()))
        off+=p.numel()
    for pref in ("actor_","family_"):
        idx=[]
        for n,a,b in slices:
            if n.startswith(pref):idx.extend(range(a,b))
        ix=torch.tensor(idx,device="cuda",dtype=torch.long)
        x=gt[ix];y=ga[ix]
        groups[pref]={"tail_norm":float(torch.linalg.vector_norm(x).detach().cpu()),"authority_norm":float(torch.linalg.vector_norm(y).detach().cpu()),"cosine":cosine(x,y)}
    rep={"tail_loss":float(Ltail.detach().cpu()),"authority_loss_neg_pairwise":float(Lauth.detach().cpu()),
    "tail_grad_norm":float(torch.linalg.vector_norm(gt).detach().cpu()),"authority_grad_norm":float(torch.linalg.vector_norm(ga).detach().cpu()),
    "cosine_tail_vs_authority":cosine(gt,ga),"groups":groups}
    print(json.dumps(rep,indent=2))
    out=ROOT/"runs/authority_isolated_tail_gradient_audit-2026-09-25"
    (out/"tail_authority_geometry.json").write_text(json.dumps(rep,indent=2)+"\n")

def run_authority_isolated_tail_budget_calibration():
    """Run former authority_isolated_tail_budget_calibration.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CK=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt";SEEDS=(840001,840002,840003,840004)
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4};NENV=8;H=64
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
            vals=[]
            for seed in SEEDS:
                for lab,wv in PREFS.items():
                    w=torch.tensor(wv,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
                    for _ in range(H):
                        with torch.no_grad():
                            z=m._actor_mean_with_preference(obs,w);a=torch.tanh(z)
                        vals.append(z.abs().cpu().numpy())
                        nxt,_,_,_,_=env.step(a);obs=ot(nxt).cuda()
            X=np.concatenate(vals,axis=0)
            names=list(env.unwrapped.scene["robot"].data.joint_names)
            q90=np.quantile(X,.90,axis=0);q95=np.quantile(X,.95,axis=0);q975=np.quantile(X,.975,axis=0)
            rep={"schema":"authority_isolated_tail_budget_v1","source":"u50 robust actor semantic support","joint_names":names,
                 "n_samples":int(len(X)),"q90":q90.tolist(),"q95":q95.tolist(),"q975":q975.tolist(),
                 "tau":q95.tolist()}
            (OUT/"tail_budget.json").write_text(json.dumps(rep,indent=2)+"\n")
            print(json.dumps(rep,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_tail_budget_from_probe():
    """Run former authority_isolated_tail_budget_from_probe.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CK=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    NAMES=["FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint","FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint","FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint"]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    obs=np.load(PROBE)["obs"].astype(np.float32)
    m=AuthorityIsolatedActorCritic(obs.shape[1],12).cuda()
    m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
    O=torch.tensor(obs,device="cuda");vals=[]
    with torch.no_grad():
        for wv in PREFS.values():
            w=torch.tensor(wv,device="cuda").repeat(len(O),1)
            vals.append(m._actor_mean_with_preference(O,w).abs().cpu().numpy())
    X=np.concatenate(vals,axis=0)
    q90=np.quantile(X,.90,axis=0);q95=np.quantile(X,.95,axis=0);q975=np.quantile(X,.975,axis=0)
    rep={"schema":"authority_isolated_tail_budget_probe_v1","source":"u50 robust actor fixed probe corpus","joint_names":NAMES,
         "n_samples":int(len(X)),"q90":q90.tolist(),"q95":q95.tolist(),"q975":q975.tolist(),"tau":q95.tolist()}
    (OUT/"tail_budget_probe.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2))

def run_authority_isolated_tail_budget_probe_calibration():
    """Run former authority_isolated_tail_budget_probe_calibration.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CK=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    OUT=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    probe=torch.tensor(np.load(PROBE)["obs"],dtype=torch.float32)
    st=torch.load(CK,map_location="cpu",weights_only=False)["model"]
    m=AuthorityIsolatedActorCritic(probe.shape[-1],12);m.load_state_dict(st);m.eval()
    vals=[]
    with torch.no_grad():
        for wv in PREFS.values():
            w=torch.tensor(wv,dtype=torch.float32).repeat(len(probe),1)
            vals.append(m._actor_mean_with_preference(probe,w).abs().numpy())
    X=np.concatenate(vals)
    q90=np.quantile(X,.90,axis=0);q95=np.quantile(X,.95,axis=0);q975=np.quantile(X,.975,axis=0)
    rep={"schema":"authority_isolated_tail_budget_probe_v1","source":"u50 robust actor fixed probe x T/A/O/S/C",
    "n_samples":len(X),"q90":q90.tolist(),"q95":q95.tolist(),"q975":q975.tolist(),"tau":q95.tolist()}
    (OUT/"tail_budget.json").write_text(json.dumps(rep,indent=2)+"\n");print(json.dumps(rep,indent=2))

def run_authority_isolated_tail_gradient_audit():
    """Run former authority_isolated_tail_gradient_audit.py stage."""
    """How hard the tail penalty would pull, per action coordinate and checkpoint.
    
    Read-only: it builds the tail loss on fixed probe states and measures its
    gradient, without applying anything.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import RUNS, OfflineAudit
    
    PREFS = {"T": [.7, .1, .1, .1], "A": [.1, .7, .1, .1], "O": [.1, .1, .7, .1],
             "S": [.1, .1, .1, .7], "C": [.25] * 4}
    CHECKPOINTS = {
        "u0": "authority_isolated_coordinate_headroom_control-2026-09-25/model_0.pt",
        "control10": "authority_isolated_coordinate_headroom_control-2026-09-25/model_10.pt",
        "treatment10": "authority_isolated_coordinate_headroom_treatment-2026-09-25/model_10.pt",
    }
    TAU_PROBE = "authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE_STATES = "update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    COORDS = ["FL_hip", "FR_hip", "RL_hip", "RR_hip",
              "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
              "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    
    
    def actor_params(m):
        return [p for n, p in m.named_parameters()
                if n.startswith("actor_") or n == "log_std" or n.startswith("family_")]
    
    
    def grad_norm(params):
        gs = [p.grad.reshape(-1) for p in params if p.grad is not None]
        return float(torch.linalg.vector_norm(torch.cat(gs)).cpu()) if gs else 0.
    
    
    class TailGradientAudit(OfflineAudit):
        """How hard the tail penalty would pull, per action coordinate and checkpoint."""
    
        run = "authority_isolated_tail_gradient_audit-2026-09-25"
        report = "tail_gradient_audit.json"
    
        def analyze(self):
            from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
            tau = torch.tensor(json.loads((RUNS / TAU_PROBE).read_text())["tau"], device="cuda")
            probe = torch.tensor(np.load(RUNS / PROBE_STATES)["obs"], device="cuda")
            rep = {}
            for lab, path in CHECKPOINTS.items():
                m = AuthorityIsolatedWideCritic(probe.shape[1], 12).cuda()
                m.load_state_dict(torch.load(RUNS / path, map_location="cuda",
                                             weights_only=False)["model"])
                m.train()
                z = torch.cat([m._actor_mean_with_preference(
                    probe, torch.tensor(wv, device="cuda").repeat(len(probe), 1))
                    for wv in PREFS.values()])
                excess = torch.relu(z.abs() - tau)
                loss = ((excess / (tau + 1e-6)) ** 2).mean()
                frac = (z.abs() > tau).float().mean(0)
                dz = 2 * excess / (tau + 1e-6) ** 2 / z.numel()
                ps = actor_params(m)
                m.zero_grad(set_to_none=True)
                loss.backward()
                gn = grad_norm(ps)
                rep[lab] = {"tail_loss": float(loss.detach().cpu()),
                            "raw_tail_grad_norm": gn,
                            "effective_lambda001_grad_norm": .01 * gn,
                            "exceed_fraction_by_coord": {COORDS[j]: float(frac[j].cpu()) for j in range(12)},
                            "mean_abs_dL_dz_by_coord": {COORDS[j]: float(dz[:, j].mean().cpu()) for j in range(12)}}
            return rep
    
        def summarize(self, report):
            print(json.dumps(report, indent=2))
    
    
    if True:
        TailGradientAudit.main()

def run_authority_isolated_tail_projected_gradient_train():
    """Run former authority_isolated_tail_projected_gradient_train.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,json,sys
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    BASE=ROOT/"runs/authority_isolated_critic_capacity-2026-09-25/wide_critic_fixed_policy.pt"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    NENV=h1.NENV;H=h1.H
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    def new_model(obs_dim,ad):
        m=AuthorityIsolatedWideCritic(obs_dim,ad).cuda()
        m.load_state_dict(torch.load(BASE,map_location="cuda",weights_only=False)["model"]);return m
    def audit(m,probe,tau,out,tag,snaps,args,lam):
        m.eval();sens=h1.sensitivity(m,probe)
        P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        wm=torch.tensor(h1.PREFS["C"],device="cuda").repeat(len(P),1)
        with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
        snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
        torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed,"arm":args.arm,"lambda_tail":lam},out/f"model_{tag}.pt")
        m.train()
    def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
          "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
          "numpy_rng":np.random.get_state()},path)
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["projected"],default="projected")
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--target-updates",type=int,default=3)
        ap.add_argument("--chunk-updates",type=int,default=3);args=ap.parse_args()
        lam=0.0
        out=ROOT/"runs/authority_isolated_tail_projected_gradient-2026-09-25"
        out.mkdir(parents=True,exist_ok=True);state_path=out/"resume_state.pt"
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=args.seed);o=h1.ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=new_model(o.shape[-1],ad);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False)
                m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
                print("RESUME",cur,flush=True)
                print("RESUME_STATE_LOADED",len(rows),list(snaps.keys()),flush=True)
            else:
                cand=[]
                for k in range(h1.ANCHOR_CANDIDATES):
                    _,w=h1.pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                    units=h1.collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
                comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
                spec_idx=h1.select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
                adaptive_pools={"early":[],"late":[]}
                for _,_,units in cand:
                    for u in units:adaptive_pools[u["phase"]].append(u)
                rows=[];snaps={};cur=0
                audit(m,probe,tau,out,0,snaps,args,lam)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur)
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
            end=min(args.target_updates,cur+args.chunk_updates)
            print("LOOP_RANGE",cur+1,end,flush=True)
            for uidx in range(cur+1,end+1):
                print("BEGIN_UPDATE",uidx,flush=True)
                labs,w=h1.pref_batch(uidx,torch.device("cuda"))
                print("BEFORE_COLLECT_ACTOR",uidx,args.seed+uidx*211,flush=True)
                main=h1.collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                print("AFTER_COLLECT_ACTOR",uidx,flush=True)
                with torch.no_grad():
                    lp_pre=m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])
                    ratio_pre=float((torch.exp(lp_pre-main["old"])-1).abs().max().cpu())
                    lp_chunks=[]
                    for st in range(0,len(main["obs"]),NENV):
                        lp_chunks.append(m.logp_from_pre_tanh_with_preference(main["obs"][st:st+NENV],main["w"][st:st+NENV],main["u"][st:st+NENV]))
                    lp_chunk=torch.cat(lp_chunks)
                    ratio_chunk=float((torch.exp(lp_chunk-main["old"])-1).abs().max().cpu())
                print("RATIO_PRE_REFRESH",uidx,ratio_pre,"CHUNK8",ratio_chunk,flush=True)
                _,ws=h1.pref_batch(uidx+17,torch.device("cuda"))
                print("BEFORE_COLLECT_SUPPORT",uidx,args.seed+200000+uidx*223,flush=True)
                units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                print("AFTER_COLLECT_SUPPORT",uidx,flush=True)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                print("BEFORE_ANCHOR_RECOLLECT",uidx,flush=True)
                anchors_now=current_anchor_units()
                print("AFTER_ANCHOR_RECOLLECT",uidx,flush=True)
                print("BEFORE_CRITIC_REFRESH",uidx,flush=True)
                h1.fit_expanded_current_policy(m,anchors_now,adaptive_pools)
                print("AFTER_CRITIC_REFRESH",uidx,flush=True)
                print("BEFORE_VALUE",uidx,flush=True)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    print("AFTER_VALUE",uidx,flush=True)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                    print("AFTER_GAE",uidx,flush=True)
                print("BEFORE_RATIO",uidx,flush=True)
                lp_parts=[]
                for st in range(0,len(main["obs"]),NENV):
                    lp_parts.append(m.logp_from_pre_tanh_with_preference(main["obs"][st:st+NENV],main["w"][st:st+NENV],main["u"][st:st+NENV]))
                ratio=torch.exp(torch.cat(lp_parts)-main["old"].detach())
                print("AFTER_RATIO",uidx,flush=True)
                print("BEFORE_RATIO_ERR",uidx,flush=True)
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                print("AFTER_RATIO_ERR",uidx,ratio_err,flush=True)
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                print("BEFORE_PPO_LOSS",uidx,flush=True)
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                print("AFTER_PPO_LOSS",uidx,flush=True)
                print("BEFORE_TAIL",uidx,flush=True)
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                print("AFTER_TAIL",uidx,flush=True)
                print("BEFORE_GRAD_BUDGET",uidx,flush=True)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None)
                gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2);gtn=torch.sqrt(gt2)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                cos=dot/(gpn*gtn+1e-12)
                coeff=torch.minimum(dot/(gt2+1e-12),torch.zeros_like(dot))
                opt.zero_grad(set_to_none=True)
                removed2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a
                    bb=torch.zeros_like(p) if b is None else b
                    corr=coeff*bb
                    p.grad=aa-corr
                    removed2=removed2+(corr.detach()**2).sum()
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                print("AFTER_STEP",uidx,flush=True)
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                     "tail_fraction":float(tailfrac.detach().cpu()),
                     "ppo_grad_norm":float(gpn.cpu()),"tail_grad_norm":float(gtn.cpu()),
                     "grad_cosine":float(cos.cpu()),"projection_coeff":float(coeff.cpu()),
                     "removed_grad_norm":float(torch.sqrt(removed2).cpu()),
                     "removed_over_ppo":float((torch.sqrt(removed2)/(gpn+1e-12)).cpu()),
                     "grad_norm_preclip":total,"ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"]}
                rows.append(row);print("UPDATE",uidx,json.dumps(row),flush=True)
                if uidx==args.target_updates:audit(m,probe,tau,out,uidx,snaps,args,lam)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx)
            print("CHUNK_DONE",end,flush=True)
            if end==args.target_updates:
                s0=snaps["0"]["sensitivity"];sf=snaps[str(end)]["sensitivity"]
                pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
                jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
                rep={"schema":"authority_isolated_tail_projected_gradient_v1","arm":args.arm,
                     "seed":args.seed,"updates":end,"tau_source":str(TAU_PATH.relative_to(ROOT)),
                     "base_checkpoint":str(BASE.relative_to(ROOT)),"rows":rows,"snapshots":snaps,
                     "summary":{"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
                       "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                       "probe_tail_loss_ratio":snaps[str(end)]["probe_tail_loss"]/(snaps["0"]["probe_tail_loss"]+1e-12),
                       "probe_tail_fraction_delta":snaps[str(end)]["probe_tail_fraction"]-snaps["0"]["probe_tail_fraction"]}}
                (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "authority_isolated_projected_descent_endpoint_audit": run_authority_isolated_projected_descent_endpoint_audit,
    "authority_isolated_projected_descent_u10_endpoint_audit": run_authority_isolated_projected_descent_u10_endpoint_audit,
    "authority_isolated_projected_endpoint_audit": run_authority_isolated_projected_endpoint_audit,
    "authority_isolated_projected_tail_descent_train": run_authority_isolated_projected_tail_descent_train,
    "authority_isolated_tail_authority_gradient_geometry": run_authority_isolated_tail_authority_gradient_geometry,
    "authority_isolated_tail_budget_calibration": run_authority_isolated_tail_budget_calibration,
    "authority_isolated_tail_budget_from_probe": run_authority_isolated_tail_budget_from_probe,
    "authority_isolated_tail_budget_probe_calibration": run_authority_isolated_tail_budget_probe_calibration,
    "authority_isolated_tail_gradient_audit": run_authority_isolated_tail_gradient_audit,
    "authority_isolated_tail_projected_gradient_train": run_authority_isolated_tail_projected_gradient_train,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
