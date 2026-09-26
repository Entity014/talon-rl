"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_control20_critic_revalidation():
    """Run former authority_isolated_control20_critic_revalidation.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    OUT=ROOT/"runs/authority_isolated_control20_critic_revalidation-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S","C");G=.99;NENV=8;H=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),"C":np.array([.25]*4,np.float32)}
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
        out=np.zeros_like(R)
        if seg is None:
            run=np.zeros_like(R[0])
            for t in range(len(R)-1,-1,-1):
                run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):
                    run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def eval_roll(env,m,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done|=dd;cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        return {"survival":float(1-done.mean()),
          "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
          "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
          "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
          "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            for li,lab in enumerate(ORDER):
                for suite in range(4):
                    seed=9300000+li*1000+suite*113
                    q=eval_roll(env,m,PREFS[lab],seed);q.update({"preference":lab,"suite":suite,"seed":seed});rows.append(q);print(q,flush=True)
            h=np.array([x["h32_ev"] for x in rows]);mc=np.array([x["mc64_ev"] for x in rows])
            hb=np.array([x["h32_bias"] for x in rows]);mb=np.array([x["mc64_bias"] for x in rows])
            agg={"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
                 "mc64_ev_mean":float(mc.mean()),"mc64_negative_fraction":float((mc<0).mean()),
                 "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
                 "min_survival":float(min(x["survival"] for x in rows))}
            by_pref={}
            for lab in ORDER:
                rr=[x for x in rows if x["preference"]==lab]
                hh=np.array([x["h32_ev"] for x in rr]);mm=np.array([x["mc64_ev"] for x in rr])
                by_pref[lab]={"h32_ev_mean":float(hh.mean()),"h32_negative_fraction":float((hh<0).mean()),
                              "mc64_ev_mean":float(mm.mean()),"mc64_negative_fraction":float((mm<0).mean()),
                              "min_survival":float(min(x["survival"] for x in rr))}
            gate=bool(agg["h32_ev_mean"]>0 and agg["mc64_ev_mean"]>0 and agg["h32_negative_fraction"]<=.25 and agg["mc64_negative_fraction"]<=.25 and agg["min_survival"]>=.95)
            rep={"schema":"authority_isolated_control20_critic_revalidation_v1","rows":rows,"aggregate":agg,"by_preference":by_pref,"critic_gate":gate}
            (OUT/"critic_revalidation.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("SUMMARY",json.dumps(rep,indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_coverage_continuation_train():
    """Run former authority_isolated_coverage_continuation_train.py stage."""
    
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
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["control","coverage","state","jointvel","friction","safe"],required=True)
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--target-updates",type=int,default=20)
        ap.add_argument("--chunk-updates",type=int,default=5);args=ap.parse_args()
        lam=0.0;kappa=0.05
        out=ROOT/f"runs/authority_isolated_coverage2_{args.arm}-2026-09-25"
        out.mkdir(parents=True,exist_ok=True);state_path=out/"resume_state.pt"
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            print("DBG_CFG_START",flush=True)
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            print("DBG_CFG_BASE_DONE",flush=True)
            state_cov=args.arm in ("coverage","state")
            friction_cov=args.arm in ("coverage","friction","safe")
            jointvel_cov=args.arm in ("coverage","jointvel","safe")
            if state_cov:
                # Broaden root-state coverage only.
                cfg.events.reset_base.params["pose_range"]={
                    "x":(-0.5,0.5),"y":(-0.5,0.5),"z":(-0.03,0.03),
                    "roll":(-0.12,0.12),"pitch":(-0.12,0.12),"yaw":(-3.14,3.14)}
                cfg.events.reset_base.params["velocity_range"]={
                    "x":(-0.75,0.75),"y":(-0.75,0.75),"z":(-0.75,0.75),
                    "roll":(-0.75,0.75),"pitch":(-0.75,0.75),"yaw":(-0.75,0.75)}
                print("DBG_RESET_BASE_MUTATED",flush=True)
            if friction_cov:
                # Resample contact dynamics per reset instead of fixed startup material.
                cfg.events.physics_material.mode="reset"
                cfg.events.physics_material.params["static_friction_range"]=(0.6,1.4)
                cfg.events.physics_material.params["dynamic_friction_range"]=(0.5,1.2)
                cfg.events.physics_material.params["restitution_range"]=(0.0,0.1)
                print("DBG_MATERIAL_MUTATED",flush=True)
            if jointvel_cov:
                # Preserve original joint-position scale distribution exactly and add only velocity coverage.
                def _reset_joints_scale_plus_velocity(env,env_ids,position_range=(0.5,1.5),velocity_range=(-1.5,1.5)):
                    robot=env.scene["robot"];iter_ids=env_ids
                    pos=robot.data.default_joint_pos[iter_ids].clone()
                    pos *= torch.empty_like(pos).uniform_(position_range[0],position_range[1])
                    lim=robot.data.soft_joint_pos_limits[iter_ids];pos=pos.clamp(lim[...,0],lim[...,1])
                    vel=torch.empty_like(robot.data.default_joint_vel[iter_ids]).uniform_(velocity_range[0],velocity_range[1])
                    vlim=robot.data.soft_joint_vel_limits[iter_ids];vel=vel.clamp(-vlim,vlim)
                    robot.write_joint_state_to_sim(pos,vel,env_ids=env_ids)
                cfg.events.reset_robot_joints.func=_reset_joints_scale_plus_velocity
                cfg.events.reset_robot_joints.params={"position_range":(0.5,1.5),"velocity_range":(-1.5,1.5)}
                print("DBG_JOINT_RESET_MUTATED",flush=True)
            print("DBG_COVERAGE_MODE",args.arm,flush=True)
            print("DBG_BEFORE_GYM_MAKE",flush=True)
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            print("DBG_AFTER_GYM_MAKE",flush=True)
            o,_=env.reset(seed=args.seed)
            print("DBG_AFTER_RESET",flush=True)
            o=h1.ot(o).cuda();probe=o.detach().clone()
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
                start_key="10"
                s0=snaps[start_key]["sensitivity"];sf=snaps[str(end)]["sensitivity"]
                pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
                jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
                rep={"schema":"authority_isolated_coverage_continuation_v1","arm":args.arm,"kappa":kappa,
                     "seed":args.seed,"start_update":10,"updates":end,"tau_source":str(TAU_PATH.relative_to(ROOT)),
                     "base_checkpoint":"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt","rows":rows,"snapshots":snaps,
                     "summary":{"pairwise_retention_from_u10":pair_ret,"jacobian_retention_from_u10":jac_ret,
                       "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                       "probe_tail_loss_ratio_from_u10":snaps[str(end)]["probe_tail_loss"]/(snaps[start_key]["probe_tail_loss"]+1e-12),
                       "probe_tail_fraction_delta_from_u10":snaps[str(end)]["probe_tail_fraction"]-snaps[start_key]["probe_tail_fraction"]}}
                (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_coverage_endpoint_eval():
    """Run former authority_isolated_coverage_endpoint_eval.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
     "control":ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt",
     "safe":ROOT/"runs/authority_isolated_coverage2_safe-2026-09-25/model_20.pt"}
    OUT=ROOT/"runs/authority_isolated_coverage_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEEDS={"suite2":840003,"suite3":840004,"held4":850101,"held5":850202,"held6":850303}
    NENV=8;H=64
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def termmap(env):
        out={};tm=env.unwrapped.termination_manager
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except:pass
        return out
    def run(env,m,wv,seed):
        w=torch.tensor(wv,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        done=np.zeros(NENV,bool);ft=np.full(NENV,-1,int);why=[[] for _ in range(NENV)]
        for t in range(H):
            with torch.no_grad():a=m.act_inference_with_preference(obs,w)
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool);tm=termmap(env)
            for i in range(NENV):
                if dd[i] and not done[i]:ft[i]=t;why[i]=[n for n,v in tm.items() if v[i]]
            done|=dd;obs=ot(nxt).cuda()
        return {"survival":float(1-done.mean()),"fail_count":int(done.sum()),"fail_t":ft.tolist(),"reason":why}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            rep={"arms":{}}
            for arm,ck in CK.items():
                m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
                rows=[]
                for sname,seed in SEEDS.items():
                    for pref,wv in PREFS.items():
                        q=run(env,m,wv,seed);q.update({"suite":sname,"seed":seed,"preference":pref});rows.append(q);print(arm,sname,pref,q,flush=True)
                semantic=[x for x in rows if x["suite"] in ("suite2","suite3")]
                held=[x for x in rows if x["suite"].startswith("held")]
                rep["arms"][arm]={
                    "rows":rows,
                    "semantic_min_survival":float(min(x["survival"] for x in semantic)),
                    "semantic_failed_lanes":int(sum(x["fail_count"] for x in semantic)),
                    "semantic_mean_survival":float(np.mean([x["survival"] for x in semantic])),
                    "heldout_min_survival":float(min(x["survival"] for x in held)),
                    "heldout_failed_lanes":int(sum(x["fail_count"] for x in held)),
                    "heldout_mean_survival":float(np.mean([x["survival"] for x in held]))}
            c=rep["arms"]["control"];s=rep["arms"]["safe"]
            rep["comparison"]={
                "semantic_failed_lane_delta_safe_minus_control":s["semantic_failed_lanes"]-c["semantic_failed_lanes"],
                "semantic_mean_survival_delta":s["semantic_mean_survival"]-c["semantic_mean_survival"],
                "heldout_failed_lane_delta_safe_minus_control":s["heldout_failed_lanes"]-c["heldout_failed_lanes"],
                "heldout_mean_survival_delta":s["heldout_mean_survival"]-c["heldout_mean_survival"]}
            (OUT/"coverage_endpoint_eval.json").write_text(json.dumps(rep,indent=2)+"\n");print("SUMMARY",json.dumps(rep["comparison"],indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_coverage_generalization_eval():
    """Run former authority_isolated_coverage_generalization_eval.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
    "u10":ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt",
    "control20":ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt",
    "coverage20":ROOT/"runs/authority_isolated_coverage2_coverage-2026-09-25/model_20.pt"}
    OUT=ROOT/"runs/authority_isolated_coverage_generalization_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEEDS={"suite2":840003,"suite3":840004,"suite4":850101,"suite5":850202,"suite6":850303}
    NENV=8;H=64
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def termmap(env):
        out={};tm=env.unwrapped.termination_manager
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except:pass
        return out
    def run(env,m,seed,wv):
        w=torch.tensor(wv,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        done=np.zeros(NENV,bool);ft=np.full(NENV,-1,int);why=[[] for _ in range(NENV)]
        for t in range(H):
            with torch.no_grad():a=m.act_inference_with_preference(obs,w)
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool);tm=termmap(env)
            for i in range(NENV):
                if dd[i] and not done[i]:
                    ft[i]=t;why[i]=[n for n,v in tm.items() if v[i]]
            done|=dd;obs=ot(nxt).cuda()
        return {"survival":float(1-done.mean()),"fail_count":int(done.sum()),"fail_t":ft.tolist(),"reason":why}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            models={}
            for lab,path in CK.items():
                m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda()
                m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
            rep={"rows":[],"summary":{}}
            for ml,m in models.items():
                vals=[];frozen=[];held=[]
                for suite,seed in SEEDS.items():
                    for pref,wv in PREFS.items():
                        q=run(env,m,seed,wv);row={"model":ml,"suite":suite,"preference":pref,**q};rep["rows"].append(row)
                        vals.append(q["survival"]);(frozen if suite in ("suite2","suite3") else held).append(q["survival"])
                        print(row,flush=True)
                rep["summary"][ml]={"min_all":float(min(vals)),"mean_all":float(np.mean(vals)),
                    "min_frozen":float(min(frozen)),"mean_frozen":float(np.mean(frozen)),
                    "min_heldout":float(min(held)),"mean_heldout":float(np.mean(held)),
                    "failed_lanes_frozen":int(sum(r["fail_count"] for r in rep["rows"] if r["model"]==ml and r["suite"] in ("suite2","suite3"))),
                    "failed_lanes_heldout":int(sum(r["fail_count"] for r in rep["rows"] if r["model"]==ml and r["suite"] in ("suite4","suite5","suite6")))}
            (OUT/"coverage_generalization_eval.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("SUMMARY",json.dumps(rep["summary"],indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_critic_capacity_gate():
    """Run former authority_isolated_critic_capacity_gate.py stage."""
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    BASE=ROOT/'runs/authority_isolated_h1-2026-09-25/model_75.pt'
    OUT=ROOT/'runs/authority_isolated_critic_capacity-2026-09-25';OUT.mkdir(parents=True,exist_ok=True)
    ORDER=('T','A','O','S','C');PREFS={'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32),'C':np.array([.25]*4,np.float32)}
    NENV=8;G=.99;H=64;HEADS=('Tracking','Angular','Orientation','Smoothness')
    def ot(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ret(R,D,seg=None):
        out=np.zeros_like(R)
        if seg is None:
            run=np.zeros_like(R[0])
            for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def collect(env,actor,w_np,seed,mgr):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device='cuda').repeat(NENV,1);cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        O=[];R=[];D=[];done=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(H):
                O.append(cur.cpu().numpy())
                a=actor.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done|=dd;cur=ot(nxt).cuda()
        O=np.asarray(O);R=np.asarray(R);D=np.asarray(D,bool)
        W=np.broadcast_to(w_np,(H,NENV,4)).copy()
        return {'obs':O,'w':W,'h32':ret(R,D,32),'mc64':ret(R,D,None),'survival':float(1-done.mean())}
    
    def eval_dataset(model,rows):
        vals=[];by_head=[[] for _ in range(4)];by_pref={k:[] for k in ORDER};surv=[]
        for r in rows:
            O=torch.tensor(r['obs'].reshape(-1,48),device='cuda')
            W=torch.tensor(r['w'].reshape(-1,4),device='cuda')
            with torch.no_grad():V=model.value_with_preference(O,W).cpu().numpy().reshape(H,NENV,4)
            he=[];me=[]
            for j in range(4):
                h=ev(r['h32'][:,:,j],V[:,:,j]);m=ev(r['mc64'][:,:,j],V[:,:,j])
                he.append(h);me.append(m);by_head[j].append((h,m))
            by_pref[r['preference']].append((he,me));vals += [(h,m) for h,m in zip(he,me)];surv.append(r['survival'])
        hv=np.array([x[0] for x in vals]);mv=np.array([x[1] for x in vals])
        out={'h32_ev_mean':float(hv.mean()),'h32_negative_fraction':float((hv<0).mean()),'mc64_ev_mean':float(mv.mean()),'mc64_negative_fraction':float((mv<0).mean()),'min_survival':float(min(surv))}
        out['by_head']=[{'head':HEADS[j],'h32_ev_mean':float(np.mean([x[0] for x in by_head[j]])),'mc64_ev_mean':float(np.mean([x[1] for x in by_head[j]]))} for j in range(4)]
        out['by_preference']={k:{'h32_ev_mean':float(np.mean([z for a,_ in v for z in a])),'mc64_ev_mean':float(np.mean([z for _,b in v for z in b]))} for k,v in by_pref.items()}
        return out
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic,initialize_actor_exact_from_ai_h1
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            st=torch.load(BASE,map_location='cuda',weights_only=False)['model']
            ctrl=AuthorityIsolatedActorCritic(48,ad).cuda();ctrl.load_state_dict(st);ctrl.eval()
            tr=AuthorityIsolatedWideCritic(48,ad).cuda();initialize_actor_exact_from_ai_h1(tr,st);tr.eval()
            # AI-C0 actor invariance
            probe=torch.tensor(np.load(ROOT/'runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz')['obs'],device='cuda')
            max_action=0.;max_logp=0.
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device='cuda').repeat(len(probe),1)
                with torch.no_grad():
                    a0=ctrl.act_inference_with_preference(probe,w);a1=tr.act_inference_with_preference(probe,w)
                max_action=max(max_action,float((a0-a1).abs().max().cpu()))
                torch.manual_seed(444);_,lp0,u0=ctrl.act_with_preference_latent(probe,w)
                torch.manual_seed(444);_,lp1,u1=tr.act_with_preference_latent(probe,w)
                max_logp=max(max_logp,float((lp0-lp1).abs().max().detach().cpu()),float((u0-u1).abs().max().detach().cpu()))
            c0=(max_action<=1e-6 and max_logp<=1e-6)
            if not c0: raise RuntimeError('AI-C0 actor invariance failed')
            # fixed-policy datasets
            fit=[];held=[];sem=[]
            for ri in range(8):
                seed=950000+ri*173
                target=fit if ri<6 else held
                for lab in ORDER:
                    q=collect(env,ctrl,PREFS[lab],seed,mgr);q['preference']=lab;target.append(q)
            for su,seed in enumerate((840001,840002,840003,840004)):
                for lab in ORDER:
                    q=collect(env,ctrl,PREFS[lab],seed,mgr);q.update({'preference':lab,'suite':su});sem.append(q)
            # train widened critic on H32 only
            X=[];W=[];Y=[]
            for r in fit:
                X.append(r['obs'].reshape(-1,48));W.append(r['w'].reshape(-1,4));Y.append(r['h32'].reshape(-1,4))
            X=np.concatenate(X).astype(np.float32);W=np.concatenate(W).astype(np.float32);Y=np.concatenate(Y).astype(np.float32)
            opt=torch.optim.Adam(list(tr.critic_body.parameters())+list(tr.critic_head.parameters()),lr=3e-4)
            rng=np.random.default_rng(26092511);tr.train();losses=[]
            for step in range(1,5001):
                idx=rng.integers(0,len(X),size=2048)
                xo=torch.tensor(X[idx],device='cuda');ww=torch.tensor(W[idx],device='cuda');yy=torch.tensor(Y[idx],device='cuda')
                pv=tr.value_with_preference(xo,ww);loss=(pv-yy).pow(2).mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step%250==0:losses.append({'step':step,'loss':float(loss.detach().cpu())})
            tr.eval()
            ck=OUT/'wide_critic_fixed_policy.pt';torch.save({'model':tr.state_dict(),'actor_source':str(BASE.relative_to(ROOT))},ck)
            ctrl_eval={'heldout_dense':eval_dataset(ctrl,held),'semantic':eval_dataset(ctrl,sem)}
            treat_eval={'heldout_dense':eval_dataset(tr,held),'semantic':eval_dataset(tr,sem)}
            q=treat_eval
            semq=q['semantic'];heldq=q['heldout_dense']
            heads={x['head']:x for x in semq['by_head']}
            passed=(heldq['h32_ev_mean']>0 and heldq['mc64_ev_mean']>0 and heldq['h32_negative_fraction']<=.25 and heldq['mc64_negative_fraction']<=.25 and heldq['min_survival']>=.95 and
                    semq['h32_ev_mean']>0 and semq['mc64_ev_mean']>0 and semq['h32_negative_fraction']<=.25 and semq['mc64_negative_fraction']<=.25 and semq['min_survival']>=.95 and
                    heads['Angular']['h32_ev_mean']>0 and heads['Smoothness']['h32_ev_mean']>0)
            rep={'schema':'authority_isolated_critic_capacity_v1','AI_C0':{'pass':c0,'max_action_error':max_action,'max_stochastic_or_logp_error':max_logp},
                 'control_old_critic':ctrl_eval,'treatment_wide_critic':treat_eval,'loss_trace':losses,
                 'decision':{'AI_C1_pass':bool(passed),'AI_C2_authorized':bool(passed),'actor_changed':False}}
            rp=OUT/'authority_isolated_critic_capacity_report.json';rp.write_text(json.dumps(rep,indent=2)+'\n')
            (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':'AI-C1 PASS' if passed else 'AI-C1 FAIL','report_sha256':sha(rp),'contract_sha256':sha(ROOT/'docs/contracts/authority/authority-isolated-critic-capacity-contract.md'),'script_sha256':sha(Path(__file__).resolve()),'checkpoint_sha256':sha(ck)},indent=2)+'\n')
            print(json.dumps(rep['decision'],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_critic_support_audit():
    """Run former authority_isolated_critic_support_audit.py stage."""
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    RUN=ROOT/'runs/authority_isolated_h1-2026-09-25'
    OUT=ROOT/'runs/authority_isolated_critic_support-2026-09-25';OUT.mkdir(parents=True,exist_ok=True)
    CKPTS={50:RUN/'model_50.pt',75:RUN/'model_75.pt'}
    ORDER=('T','A','O','S','C')
    PREFS={'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),
    'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32),'C':np.array([.25]*4,np.float32)}
    NENV=8;G=.99;SUP_H=64;HEADS=('Tracking','Angular','Orientation','Smoothness')
    
    def ot(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
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
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
        return sol[:-1],sol[-1]
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64)
        X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        sel=[int(np.argmax(np.mean(X*X,axis=1)))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(k,device):
        labs=[ORDER[(k+i)%len(ORDER)] for i in range(NENV)]
        return torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect64(env,m,w,seed,label,mgr):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                cmd.append(env.unwrapped.command_manager.get_command('base_velocity').cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur);rw.append(torch.tensor(vec,device='cuda')*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        O=torch.stack(obs);R=torch.stack(rw);D=torch.stack(dn).bool();C=np.stack(cmd)
        units=[]
        for pi,(st,en) in enumerate(((0,32),(32,64))):
            Y=trunc(R[st:en],D[st:en]).reshape(-1,4).cpu().numpy()
            po=O[st:en].reshape(-1,O.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).cpu().numpy()
            cc=C[st:en].reshape(-1,C.shape[-1])
            summary=np.r_[cc.mean(0),cc.std(0),w.mean(0).cpu().numpy(),F.mean(0),F.std(0),float(pi)]
            units.append({'phase':'early' if pi==0 else 'late','F':F,'Y':Y,
                          'summary':summary.astype(np.float32),'label':label})
        return units
    
    def fit_units(units):
        F=np.concatenate([u['F'] for u in units]);Y=np.concatenate([u['Y'] for u in units])
        return ridge(F,Y,1.0),{'samples':len(F),'units':len(units)}
    def pred(F,fit):return F@fit[0]+fit[1]
    
    def evaluate(units,fit):
        out={};allv=[]
        for phase in ('early','late'):
            rr=[u for u in units if u['phase']==phase];heads=[];prefs={};suites={}
            for j,h in enumerate(HEADS):
                vals=[];bias=[]
                for r in rr:
                    P=pred(r['F'],fit);vals.append(ev(r['Y'][:,j],P[:,j]));bias.append(float(np.mean(P[:,j]-r['Y'][:,j])))
                heads.append({'head':h,'ev_mean':float(np.mean(vals)),'negative_fraction':float(np.mean(np.array(vals)<0)),
                              'mean_abs_bias':float(np.mean(np.abs(bias)))})
                allv += vals
            for lab in ORDER:
                z=[r for r in rr if r.get('preference')==lab];vals=[]
                for r in z:
                    P=pred(r['F'],fit);vals += [ev(r['Y'][:,j],P[:,j]) for j in range(4)]
                prefs[lab]={'ev_mean':float(np.mean(vals)),'negative_fraction':float(np.mean(np.array(vals)<0))}
            for su in range(4):
                z=[r for r in rr if r.get('suite')==su];vals=[]
                for r in z:
                    P=pred(r['F'],fit);vals += [ev(r['Y'][:,j],P[:,j]) for j in range(4)]
                suites[str(su)]={'ev_mean':float(np.mean(vals)),'negative_fraction':float(np.mean(np.array(vals)<0))}
            out[phase]={'ev_mean':float(np.mean([x['ev_mean'] for x in heads])),
                        'negative_fraction':float(np.mean([x['negative_fraction'] for x in heads])),
                        'mean_abs_bias':float(np.mean([x['mean_abs_bias'] for x in heads])),
                        'by_head':heads,'by_preference':prefs,'by_suite':suites}
        out['combined_negative_fraction']=float(np.mean(np.array(allv)<0))
        return out
    
    def geom(support,eval_units):
        ans={}
        for phase in ('early','late'):
            S=np.concatenate([u['F'] for u in support if u['phase']==phase])
            E=np.concatenate([u['F'] for u in eval_units if u['phase']==phase])
            # deterministic evenly-spaced cap for distance diagnostics
            S=S[np.linspace(0,len(S)-1,min(4096,len(S)),dtype=int)]
            E=E[np.linspace(0,len(E)-1,min(4096,len(E)),dtype=int)]
            mu=S.mean(0);sd=S.std(0)+1e-6;Sz=(S-mu)/sd;Ez=(E-mu)/sd
            # nearest standardized support distance, chunked
            mins=[]
            for i in range(0,len(Ez),256):
                q=Ez[i:i+256]
                d=((q[:,None,:]-Sz[None,:,:])**2).mean(2)
                mins.append(np.sqrt(d.min(1)))
            nearest=np.concatenate(mins)
            cov=(Sz.T@Sz)/max(len(Sz)-1,1)
            val,vec=np.linalg.eigh(cov);idx=np.argsort(val)[::-1];val=val[idx];vec=vec[:,idx]
            keep=val>1e-6;val=val[keep];vec=vec[:,keep]
            Z=Ez@vec
            maha=np.sqrt(np.mean((Z**2)/(val[None,:]+1e-6),axis=1))
            ans[phase]={'nearest_standardized_mean':float(nearest.mean()),
                        'nearest_standardized_p95':float(np.percentile(nearest,95)),
                        'pca_whitened_distance_mean':float(maha.mean()),
                        'feature_mean_shift_std_norm':float(np.linalg.norm(Ez.mean(0))/np.sqrt(Ez.shape[1]))}
        return ans
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            report={'schema':'authority_isolated_critic_support_compatibility_v1','checkpoints':{}}
            for upd,cp in CKPTS.items():
                print('CHECKPOINT',upd,flush=True)
                m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(cp,map_location='cuda',weights_only=False)['model']);m.eval()
                # evaluation corpus
                eval_units=[]
                for su,seed in enumerate((840001,840002,840003,840004)):
                    for lab in ORDER:
                        w=torch.tensor(PREFS[lab],device='cuda').repeat(NENV,1)
                        for u in collect64(env,m,w,seed,f'eval_{lab}_{su}',mgr):
                            u.update({'preference':lab,'suite':su});eval_units.append(u)
                # anchor and adaptive candidate pools, independent seeds
                anchor=[];adaptive=[]
                for k in range(12):
                    w=pref_batch(k,torch.device('cuda'))
                    anchor += collect64(env,m,w,910000+upd*100+k*137,f'a{k}',mgr)
                    adaptive += collect64(env,m,w,920000+upd*100+k*149,f'd{k}',mgr)
                selected=[]
                for ph in ('early','late'):
                    A=[u for u in anchor if u['phase']==ph];D=[u for u in adaptive if u['phase']==ph]
                    for i in select_diverse([u['summary'] for u in A],3):selected.append(A[i])
                    for i in select_diverse([u['summary'] for u in D],3):selected.append(D[i])
                expanded=anchor+adaptive
                # dense current-policy support: 8 reset seeds x 5 anchor preferences
                dense=[];dense_train=[];dense_hold=[]
                for ri in range(8):
                    seed=930000+upd*100+ri*173
                    target=dense_train if ri<6 else dense_hold
                    for lab in ORDER:
                        w=torch.tensor(PREFS[lab],device='cuda').repeat(NENV,1)
                        us=collect64(env,m,w,seed,f'dense_{ri}_{lab}',mgr)
                        for u in us:u.update({'preference':lab,'dense_reset':ri})
                        dense += us;target += us
                Wcur=m.critic_head.weight.detach().cpu().numpy().T
                bcur=m.critic_head.bias.detach().cpu().numpy()
                fits={'CURRENT_HEAD':(Wcur,bcur),'SELECTED12':fit_units(selected)[0],
                      'EXPANDED':fit_units(expanded)[0],'DENSE_CURRENT':fit_units(dense)[0],
                      'DENSE_75_TRAIN':fit_units(dense_train)[0]}
                regimes={k:evaluate(eval_units,v) for k,v in fits.items() if k!='DENSE_75_TRAIN'}
                body_probe_eval=evaluate(dense_hold,fits['DENSE_75_TRAIN'])
                body_probe_semantic=evaluate(eval_units,fits['DENSE_75_TRAIN'])
                report['checkpoints'][str(upd)]={'regimes':regimes,
                  'body_probe':{'heldout_dense':body_probe_eval,'semantic_suites':body_probe_semantic},
                  'feature_geometry':{'SELECTED12':geom(selected,eval_units),
                                      'EXPANDED':geom(expanded,eval_units),
                                      'DENSE_CURRENT':geom(dense,eval_units)},
                  'support_counts':{'selected_units':len(selected),'expanded_units':len(expanded),
                                    'dense_units':len(dense),'dense_train_units':len(dense_train),'dense_hold_units':len(dense_hold)}}
            u75=report['checkpoints']['75']
            restored=[]
            for name in ('EXPANDED','DENSE_CURRENT'):
                r=u75['regimes'][name]
                ok=(r['early']['ev_mean']>0 and r['late']['ev_mean']>0 and
                    r['early']['negative_fraction']<=.25 and r['late']['negative_fraction']<=.25 and
                    r['combined_negative_fraction']<=.25)
                restored.append((name,ok))
            bp=u75['body_probe']['heldout_dense']
            bpok=(bp['early']['ev_mean']>0 and bp['late']['ev_mean']>0 and
                  bp['early']['negative_fraction']<=.25 and bp['late']['negative_fraction']<=.25 and
                  bp['combined_negative_fraction']<=.25)
            if any(ok for _,ok in restored) and bpok:
                verdict='SUPPORT COMPATIBILITY PROBLEM'
            elif (not restored[-1][1]) and (not bpok):
                verdict='CRITIC BODY INSUFFICIENT'
            else:
                verdict='INCONCLUSIVE'
            report['decision']={'verdict':verdict,'restored_regimes':dict(restored),
                                'dense_body_probe_pass':bpok,'actor_changed':False,'rl_training_authorized':False}
            rp=OUT/'authority_isolated_critic_support_report.json';rp.write_text(json.dumps(report,indent=2)+'\n')
            (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':verdict,
              'report_sha256':sha(rp),'contract_sha256':sha(ROOT/'docs/contracts/authority/authority-isolated-critic-support-compatibility-contract.md'),
              'script_sha256':sha(Path(__file__).resolve()),'u50_sha256':sha(CKPTS[50]),'u75_sha256':sha(CKPTS[75])},indent=2)+'\n')
            print(json.dumps(report['decision'],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "authority_isolated_control20_critic_revalidation": run_authority_isolated_control20_critic_revalidation,
    "authority_isolated_coverage_continuation_train": run_authority_isolated_coverage_continuation_train,
    "authority_isolated_coverage_endpoint_eval": run_authority_isolated_coverage_endpoint_eval,
    "authority_isolated_coverage_generalization_eval": run_authority_isolated_coverage_generalization_eval,
    "authority_isolated_critic_capacity_gate": run_authority_isolated_critic_capacity_gate,
    "authority_isolated_critic_support_audit": run_authority_isolated_critic_support_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
