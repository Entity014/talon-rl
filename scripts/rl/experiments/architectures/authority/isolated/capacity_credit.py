"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_ai_c2_capacity_init():
    """Run former authority_isolated_ai_c2_capacity_init.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SRC=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    OUT=ROOT/"runs/authority_isolated_ai_c2_capacity_init-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    SEEDS=[961000+i*173 for i in range(8)]
    ORDER=h1.ORDER;PREFS=h1.PREFS;NENV=h1.NENV
    STEPS=6000;BATCH=2048
    
    def copy_actor(dst,src):
     t=dst.state_dict()
     for k in t:
      if k.startswith("actor_") or k.startswith("family_") or k=="log_std":t[k].copy_(src[k])
     dst.load_state_dict(t)
    
    def tangent(m,obs,w):
     B=torch.tensor([[1.,-1,0,0],[1,0,-1,0],[1,0,0,-1]],device=obs.device)
     vals=[]
     for st in range(0,len(obs),64):
      oo=obs[st:st+64];ww=w[st:st+64].clone().detach().requires_grad_(True)
      v=m.value_with_preference(oo,ww);J=[]
      for j in range(4):
       g=torch.autograd.grad(v[:,j].sum(),ww,retain_graph=True)[0]
       J.append(g@B.T)
      vals.append(torch.stack(J,1))
     return torch.cat(vals)
    
    def corr(a,b):
     a=a.reshape(-1);b=b.reshape(-1)
     return float(np.corrcoef(a,b)[0,1])
    
    def main():
     from isaaclab.app import AppLauncher
     sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=h1.ot(o).cuda()
      src=torch.load(SRC,map_location="cuda",weights_only=False)["model"]
      teacher=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();teacher.load_state_dict(src);teacher.eval()
      torch.manual_seed(26092542)
      student=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();copy_actor(student,src);student.train()
      # frozen-policy state support from teacher actor, plus random simplex w labels per observed state
      obsbuf=[]
      with torch.no_grad():
       for seed in SEEDS:
        for lab in ORDER:
         w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);cur,_=env.reset(seed=seed);cur=h1.ot(cur).cuda()
         for t in range(64):
          obsbuf.append(cur.cpu());a=teacher.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a);cur=h1.ot(nxt).cuda()
      X=torch.cat(obsbuf).float()
      rng=np.random.default_rng(26092542)
      # 2 preference draws per state: one Dirichlet + one canonical cycling preference
      Wd=rng.dirichlet(np.ones(4),size=len(X)).astype(np.float32)
      Wc=np.stack([np.asarray(PREFS[ORDER[i%5]],np.float32) for i in range(len(X))])
      X2=torch.cat([X,X]);W2=torch.tensor(np.concatenate([Wd,Wc])).float()
      with torch.no_grad():
       Y=[]
       for st in range(0,len(X2),4096):
        Y.append(teacher.value_with_preference(X2[st:st+4096].cuda(),W2[st:st+4096].cuda()).cpu())
       Y=torch.cat(Y)
      params=list(student.critic_body.parameters())+list(student.critic_head.parameters())
      opt=torch.optim.Adam(params,lr=3e-4);trace=[]
      for step in range(1,STEPS+1):
       idx=rng.integers(0,len(X2),size=BATCH)
       pv=student.value_with_preference(X2[idx].cuda(),W2[idx].cuda());yy=Y[idx].cuda()
       loss=(pv-yy).pow(2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
       if step%500==0:trace.append({"step":step,"loss":float(loss.detach().cpu())});print(trace[-1],flush=True)
      # exact actor unchanged
      ae=0.
      for k,v in student.state_dict().items():
       if k.startswith("actor_") or k.startswith("family_") or k=="log_std":ae=max(ae,float((v-src[k]).abs().max().cpu()))
      P=torch.tensor(np.load(PROBE)["obs"],device="cuda",dtype=torch.float32)
      stats=[]
      for lab in ORDER:
       w=torch.tensor(PREFS[lab],device="cuda").repeat(len(P),1)
       with torch.no_grad():a=teacher.value_with_preference(P,w);b=student.value_with_preference(P,w)
       J0=tangent(teacher,P,w).detach();J1=tangent(student,P,w).detach()
       d=b-a
       stats.append({"preference":lab,"rmse":float(torch.sqrt((d*d).mean()).cpu()),
         "corr":corr(a.cpu().numpy(),b.cpu().numpy()),
         "tangent_cosine":float((J0.flatten()@J1.flatten()/(J0.norm()*J1.norm()+1e-12)).cpu()),
         "tangent_norm_ratio":float((J1.norm()/(J0.norm()+1e-12)).cpu())})
      agg={k:float(np.mean([q[k] for q in stats])) for k in ("rmse","corr","tangent_cosine","tangent_norm_ratio")}
      passed=bool(ae<=1e-7 and agg["corr"]>=.98 and agg["tangent_cosine"]>=.90)
      rep={"schema":"ai_c2_capacity_init_v1","actor_error":ae,"train_trace":trace,"probe":stats,"aggregate":agg,"init_gate":passed}
      torch.save({"model":student.state_dict(),"teacher":str(SRC.relative_to(ROOT))},OUT/"narrow_distilled.pt")
      (OUT/"capacity_init.json").write_text(json.dumps(rep,indent=2)+"\n")
      print("FINAL",json.dumps(rep,indent=2),flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_authority_isolated_ai_c2_continuity_control():
    """Run former authority_isolated_ai_c2_continuity_control.py stage."""
    
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
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["continuity"],default="continuity")
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--target-updates",type=int,default=30)
        ap.add_argument("--chunk-updates",type=int,default=10);args=ap.parse_args()
        lam=0.0;kappa=0.05
        out=ROOT/"runs/authority_isolated_ai_c2_continuity-2026-09-25"
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
                start_key="20"
                s0=snaps[start_key]["sensitivity"];sf=snaps[str(end)]["sensitivity"]
                pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
                jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
                rep={"schema":"authority_isolated_ai_c2_continuity_v1","arm":args.arm,"kappa":kappa,
                     "seed":args.seed,"start_update":20,"updates":end,"tau_source":str(TAU_PATH.relative_to(ROOT)),
                     "base_checkpoint":"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt","rows":rows,"snapshots":snaps,
                     "summary":{"pairwise_retention_from_u20":pair_ret,"jacobian_retention_from_u20":jac_ret,
                       "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                       "probe_tail_loss_ratio_from_u20":snaps[str(end)]["probe_tail_loss"]/(snaps[start_key]["probe_tail_loss"]+1e-12),
                       "probe_tail_fraction_delta_from_u20":snaps[str(end)]["probe_tail_fraction"]-snaps[start_key]["probe_tail_fraction"]}}
                (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_c2_credit_geometry_audit():
    """Run former authority_isolated_ai_c2_credit_geometry_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
    
    CK={
     "inherited_wide":ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt",
     "refit_wide":ROOT/"runs/authority_isolated_ai_c2_clean2_wide-2026-09-25/model_0.pt",
     "refit_narrow":ROOT/"runs/authority_isolated_ai_c2_clean2_narrow-2026-09-25/model_0.pt"}
    OUT=ROOT/"runs/authority_isolated_ai_c2_credit_geometry_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    NENV=h1.NENV;H=h1.H
    UPDATES=[1,2,3,4,5]
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    
    def actor_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
    
    def flat_grad(gs,params):
        out=[]
        for g,(_,p) in zip(gs,params):
            out.append((torch.zeros_like(p) if g is None else g).reshape(-1))
        return torch.cat(out)
    
    def cosine(a,b):
        return float((a@b/(a.norm()*b.norm()+1e-12)).detach().cpu())
    
    def authority_metric(m,probe):
        labs=("T","A","O","S")
        acts=[]
        for lab in labs:
            w=torch.tensor(h1.PREFS[lab],device="cuda").repeat(len(probe),1)
            acts.append(m.act_inference_with_preference(probe,w))
        vals=[]
        for i in range(len(acts)):
            for j in range(i+1,len(acts)):
                vals.append(torch.linalg.vector_norm(acts[i]-acts[j],dim=1).mean())
        return torch.stack(vals).mean()
    
    def family_share(gvec,params):
        num=0;den=0;off=0
        for n,p in params:
            nels=p.numel();q=gvec[off:off+nels];e=float((q*q).sum().detach().cpu());den+=e
            if n.startswith("family_"):num+=e
            off+=nels
        return float(num/(den+1e-12))
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=73001
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=73001);o=h1.ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            models={}
            for k,path in CK.items():
                cls=AuthorityIsolatedActorCritic if k=="refit_narrow" else AuthorityIsolatedWideCritic
                m=cls(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();models[k]=m
            # exact actor invariance
            ref=models["inherited_wide"];maxerr={}
            for k,m in models.items():
                e=0.
                for (n,p),(n2,q) in zip(actor_params(ref),actor_params(m)):
                    assert n==n2;e=max(e,float((p-q).abs().max().detach().cpu()))
                maxerr[k]=e
            probe=torch.tensor(np.load(PROBE)["obs"],device="cuda",dtype=torch.float32)
            p_ref=actor_params(ref)
            A=authority_metric(ref,probe)
            gA=torch.autograd.grad(A,[p for _,p in p_ref],retain_graph=False,allow_unused=True)
            gAflat=flat_grad(gA,p_ref).detach()
            rows=[]
            for uidx in UPDATES:
                _,w=h1.pref_batch(uidx,torch.device("cuda"))
                # rollout once using inherited actor; actor is exact across all three models
                main=h1.collect_actor(env,ref,w,mgr,73001+uidx*211,True)
                grads={};vals={};advs={}
                for k,m in models.items():
                    params=actor_params(m)
                    with torch.no_grad():
                        vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                        nv=m.value_with_preference(main["next_obs"],w)
                        adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                    # ratio=1 exactly at shared actor
                    ratio=torch.ones(len(main["obs"]),device="cuda")
                    loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                    gs=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
                    gv=flat_grad(gs,params).detach()
                    grads[k]=gv;vals[k]=vt.detach().cpu().numpy();advs[k]=adv.detach().cpu().numpy()
                base=grads["inherited_wide"]
                rec={"update":uidx}
                for k in ("refit_wide","refit_narrow"):
                    v0=vals["inherited_wide"].reshape(-1,4);v1=vals[k].reshape(-1,4)
                    a0=advs["inherited_wide"].reshape(-1,4);a1=advs[k].reshape(-1,4)
                    rec[k]={
                      "grad_cosine_to_inherited":cosine(base,grads[k]),
                      "grad_norm_ratio":float((grads[k].norm()/(base.norm()+1e-12)).cpu()),
                      "family_grad_share":family_share(grads[k],actor_params(models[k])),
                      "value_rmse":float(np.sqrt(np.mean((v1-v0)**2))),
                      "adv_corr":float(np.corrcoef(a0.reshape(-1),a1.reshape(-1))[0,1]),
                      "authority_direction_cosine":float((-(gAflat@grads[k])/(gAflat.norm()*grads[k].norm()+1e-12)).cpu())
                    }
                rec["inherited_wide"]={
                  "family_grad_share":family_share(base,p_ref),
                  "authority_direction_cosine":float((-(gAflat@base)/(gAflat.norm()*base.norm()+1e-12)).cpu())
                }
                rows.append(rec);print(json.dumps(rec),flush=True)
            agg={}
            for k in ("refit_wide","refit_narrow"):
                agg[k]={x:float(np.mean([r[k][x] for r in rows])) for x in rows[0][k]}
            agg["inherited_wide"]={x:float(np.mean([r["inherited_wide"][x] for r in rows])) for x in rows[0]["inherited_wide"]}
            rep={"schema":"ai_c2_credit_geometry_audit_v1","actor_max_error":maxerr,
                 "authority_metric":float(A.detach().cpu()),"rows":rows,"aggregate":agg}
            (OUT/"credit_geometry_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("SUMMARY",json.dumps(agg,indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_c2_edge_endpoint_audit():
    """Run former authority_isolated_ai_c2_edge_endpoint_audit.py stage."""
    """Simplex-edge endpoint authority for the edge-C2 narrow/wide pair."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.simplex_edge_endpoint import SimplexEdgeEndpointAudit
    
    
    class EdgeEndpointAudit(SimplexEdgeEndpointAudit):
        """Simplex-edge endpoint authority for the edge-C2 narrow/wide pair."""
    
        run = "authority_isolated_ai_c2_edge_endpoint_audit-2026-09-25"
        narrow = ("narrow", "authority_isolated_ai_c2_edgec2_narrow-2026-09-25/model_10.pt")
        wide = ("wide", "authority_isolated_ai_c2_edgec2_wide-2026-09-25/model_10.pt")
    
    
    if True:
        EdgeEndpointAudit.main()

def run_authority_isolated_ai_c2_edge_noregression_eval():
    """Run former authority_isolated_ai_c2_edge_noregression_eval.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
    "narrow":ROOT/"runs/authority_isolated_ai_c2_edgec2_narrow-2026-09-25/model_10.pt",
    "wide":ROOT/"runs/authority_isolated_ai_c2_edgec2_wide-2026-09-25/model_10.pt"}
    OUT=ROOT/"runs/authority_isolated_ai_c2_edge_noregression_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S","C");PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEM={"suite2":840003,"suite3":840004};HELD={"suite4":850101,"suite5":850202,"suite6":850303}
    NENV=8;H=64;G=.99
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
    def rollout(env,m,wv,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool);reasons=[[] for _ in range(NENV)]
        with torch.no_grad():
            for t in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();done|=dd
                tm=env.unwrapped.termination_manager
                for i in range(NENV):
                    if dd[i]:
                        reasons[i]=[n for n in tm.active_terms if bool(tm.get_term(n)[i].detach().cpu())]
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                D.append(dd);cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        return {"survival":float(1-done.mean()),"fail_count":int(done.sum()),"reasons":reasons,
          "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
          "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
          "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
          "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def summarize(rows):
        h=np.array([r["h32_ev"] for r in rows]);m=np.array([r["mc64_ev"] for r in rows])
        hb=np.array([r["h32_bias"] for r in rows]);mb=np.array([r["mc64_bias"] for r in rows])
        return {"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
          "mc64_ev_mean":float(m.mean()),"mc64_negative_fraction":float((m<0).mean()),
          "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
          "min_survival":float(min(r["survival"] for r in rows)),"failed_lanes":int(sum(r["fail_count"] for r in rows))}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            rep={"schema":"authority_isolated_ai_c2_endpoint_eval_v1","models":{}}
            for arm,path in CK.items():
                cls=AuthorityIsolatedActorCritic if arm=="narrow" else AuthorityIsolatedWideCritic
                m=cls(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval()
                sem=[];held=[];fresh=[]
                for suite,seed in SEM.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});sem.append(q)
                for suite,seed in HELD.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});held.append(q)
                for li,lab in enumerate(ORDER):
                    for si in range(4):
                        seed=9700000+li*1000+si*113
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":si,"preference":lab,"seed":seed});fresh.append(q)
                fs=summarize(fresh);ss=summarize(sem);hs=summarize(held)
                critic_gate=bool(fs["h32_ev_mean"]>0 and fs["mc64_ev_mean"]>0 and fs["h32_negative_fraction"]<=.25 and fs["mc64_negative_fraction"]<=.25 and fs["min_survival"]>=.95)
                rep["models"][arm]={"semantic":ss,"heldout":hs,"fresh_critic":fs,"critic_gate":critic_gate,
                                    "semantic_rows":sem,"heldout_rows":held,"fresh_rows":fresh}
                print("SUMMARY",arm,json.dumps(rep["models"][arm],default=str)[:2000],flush=True)
            (OUT/"endpoint_eval.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_c2_edge_paired_train():
    """Run former authority_isolated_ai_c2_edge_paired_train.py stage."""
    
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
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SOURCE_CK=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    SOURCE_STATE=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    AUTH_SUPPORT=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    NENV=h1.NENV;H=h1.H;G=.99
    ORDER=h1.ORDER;PREFS=h1.PREFS
    PREFIT_SEEDS=[960000+i*173 for i in range(6)]
    PREFIT_STEPS=2500;PREFIT_BATCH=1024
    KAPPA=.05
    RHO=.25;BETA0=2.497041993384243;GAMMA=.90
    ALL_PREFS=("T","A","O","S","C")
    EDGE_PAIRS=tuple((ALL_PREFS[i],ALL_PREFS[j]) for i in range(len(ALL_PREFS)) for j in range(i+1,len(ALL_PREFS)))
    
    def ot(x):
        return h1.ot(x)
    
    def copy_actor_exact(dst,src_state):
        tgt=dst.state_dict()
        for k in tgt:
            if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
                tgt[k].copy_(src_state[k])
        dst.load_state_dict(tgt)
    
    def terms(raw,names):
        return {n:raw[:,i] for i,n in enumerate(names)}
    
    def trunc_h32(rt,dt):
        # Phase-local H32 targets: bootstrap accumulator is reset every 32 steps.
        out=torch.zeros_like(rt)
        for st in range(0,len(rt),32):
            en=min(st+32,len(rt));run=torch.zeros_like(rt[-1])
            for t in range(en-1,st-1,-1):
                run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    
    def collect_prefit(env,m,mgr):
        from talon_rl.rewards.objectives import normalized_objective_vector
        X=[];W=[];Y=[]
        m.eval()
        for seed in PREFIT_SEEDS:
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];rw=[];dn=[]
                with torch.no_grad():
                    for _ in range(64):
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,te,tr,_=env.step(a)
                        raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                        vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                        obs.append(cur.cpu());rw.append(torch.tensor(vec)*env.unwrapped.step_dt)
                        dn.append((te|tr).cpu());cur=ot(nxt).cuda()
                rt=torch.stack(rw);dt=torch.stack(dn).bool();yy=trunc_h32(rt,dt)
                X.append(torch.cat(obs));W.append(w.cpu().repeat(64,1));Y.append(yy.reshape(-1,4))
        return torch.cat(X).float(),torch.cat(W).float(),torch.cat(Y).float()
    
    def prefit_critic(m,X,W,Y):
        params=list(m.critic_body.parameters())+list(m.critic_head.parameters())
        opt=torch.optim.Adam(params,lr=3e-4)
        rng=np.random.default_rng(2609252201)
        m.train();trace=[]
        for step in range(1,PREFIT_STEPS+1):
            idx=rng.integers(0,len(X),size=PREFIT_BATCH)
            xo=X[idx].cuda();ww=W[idx].cuda();yy=Y[idx].cuda()
            pv=m.value_with_preference(xo,ww);loss=(pv-yy).pow(2).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%250==0:trace.append({"step":step,"loss":float(loss.detach().cpu())})
        return trace
    
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    
    def audit(m,probe,tau,out,tag,snaps,arm):
        m.eval();sens=h1.sensitivity(m,probe)
        P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(P),1)
        with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
        snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
        torch.save({"model":m.state_dict(),"update":int(tag),"arm":arm},out/f"model_{tag}.pt")
        m.train()
    
    def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update,prefit_trace):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
          "prefit_trace":prefit_trace,"torch_rng":torch.get_rng_state(),
          "cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state()},path)
    
    def balanced_indices(origin,phase,gidx):
        rng=np.random.default_rng(2609252401+gidx);ids=[]
        for oi in range(5):
            for pi in range(3):
                pool=np.flatnonzero((origin==oi)&(phase==pi))
                ids.extend(rng.choice(pool,size=8,replace=False).tolist())
        return np.asarray(ids,np.int64)
    
    def edge_floor_loss(m,ref,obs):
        acts={};refs={};n=len(obs)
        for lab in ALL_PREFS:
            w=torch.tensor(PREFS[lab],device="cuda").repeat(n,1)
            acts[lab]=m.act_inference_with_preference(obs,w)
            with torch.no_grad():
                refs[lab]=ref.act_inference_with_preference(obs,w)
        terms=[]
        for i,j in EDGE_PAIRS:
            d=torch.linalg.vector_norm(acts[i]-acts[j],dim=1)
            dr=torch.linalg.vector_norm(refs[i]-refs[j],dim=1)
            terms.append(torch.relu(GAMMA*dr-d).pow(2))
        return torch.stack(terms,dim=1).mean()
    
    def grad_norm(gs):
        return torch.sqrt(sum((g.detach()**2).sum() for g in gs if g is not None)+1e-12)
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--arm",choices=["narrow","wide"],required=True)
        ap.add_argument("--seed",type=int,default=73001)
        ap.add_argument("--updates",type=int,default=10)
        ap.add_argument("--run-tag",default="clean2")
        args=ap.parse_args()
        out=ROOT/f"runs/authority_isolated_ai_c2_{args.run_tag}_{args.arm}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        state_path=out/"resume_state.pt"
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
    
            source=torch.load(SOURCE_CK,map_location="cuda",weights_only=False)["model"]
            source_resume=torch.load(SOURCE_STATE,map_location="cpu",weights_only=False)
            auth=np.load(AUTH_SUPPORT);auth_obs=torch.tensor(auth["obs"],device="cuda");auth_origin=auth["origin"];auth_phase=auth["phase"]
            ref=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();ref.load_state_dict(source);ref.eval()
            for p in ref.parameters():p.requires_grad_(False)
            cls=AuthorityIsolatedActorCritic if args.arm=="narrow" else AuthorityIsolatedWideCritic
            torch.manual_seed(26092520)
            m=cls(o.shape[-1],ad).cuda();copy_actor_exact(m,source)
    
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False)
                m.load_state_dict(st["model"])
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
                opt=torch.optim.Adam(actor_params,lr=1e-3);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
                prefit_trace=st["prefit_trace"]
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
                print("RESUME",args.arm,cur,flush=True)
            else:
                # Same frozen robust-u20 policy dataset for both critic architectures.
                X,W,Y=collect_prefit(env,m,mgr)
                print("PREFIT_DATA",args.arm,len(X),flush=True)
                prefit_trace=prefit_critic(m,X,W,Y)
                # Actor must remain exact after critic-only equilibration.
                chk=m.state_dict();max_actor=0.0
                for k,v in source.items():
                    if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
                        max_actor=max(max_actor,float((chk[k]-v).abs().max().detach().cpu()))
                if max_actor>1e-7:raise RuntimeError(f"actor changed during prefit {max_actor}")
                print("PREFIT_DONE",args.arm,prefit_trace[-1],"actor_err",max_actor,flush=True)
    
                # Freeze critic representation during actor-learning phase; linear head is refreshed analytically.
                for n,p in m.named_parameters():
                    if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
                opt=torch.optim.Adam(actor_params,lr=1e-3)
                # Load exact robust-u20 actor optimizer state; actor parameter ordering is architecture-invariant.
                opt.load_state_dict(source_resume["optimizer"])
    
                # Matched support topology; each arm builds features through its own critic representation.
                cand=[]
                for k in range(h1.ANCHOR_CANDIDATES):
                    _,w=h1.pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                    units=h1.collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
                # Exact same anchor/reset schedule for both critic architectures.
                anchor_specs=[tuple(x) for x in source_resume["anchor_specs"]]
                adaptive_pools={"early":[],"late":[]}
                for _,_,units in cand:
                    for u in units:adaptive_pools[u["phase"]].append(u)
                rows=[];snaps={};cur=0
                audit(m,probe,tau,out,0,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur,prefit_trace)
    
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
    
            # Ensure critic params remain frozen after resume too.
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
    
            # This is continuation from the robust global update-20 checkpoint.
            # Local AI-C2 steps 1..10 must therefore use the *new* global update
            # schedule 21..30 rather than replaying reset/support seeds from 1..10.
            GLOBAL_START_UPDATE=20
            for uidx in range(cur+1,args.updates+1):
                gidx=GLOBAL_START_UPDATE+uidx
                labs,w=h1.pref_batch(gidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,args.seed+gidx*211,True)
                with torch.no_grad():
                    lp_pre=m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])
                    ratio_pre=float((torch.exp(lp_pre-main["old"])-1).abs().max().cpu())
                _,ws=h1.pref_batch(gidx+17,torch.device("cuda"))
                units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+gidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                anchors_now=current_anchor_units();h1.fit_expanded_current_policy(m,anchors_now,adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                lp_parts=[]
                for st in range(0,len(main["obs"]),NENV):
                    lp_parts.append(m.logp_from_pre_tanh_with_preference(main["obs"][st:st+NENV],main["w"][st:st+NENV],main["u"][st:st+NENV]))
                ratio=torch.exp(torch.cat(lp_parts)-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,retain_graph=True,allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2);gtn=torch.sqrt(gt2)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                cos=dot/(gpn*gtn+1e-12);coeff=torch.minimum(dot/(gt2+1e-12),torch.zeros_like(dot))
                proj_parts=[];proj2=torch.zeros((),device="cuda");removed2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a;bb=torch.zeros_like(p) if b is None else b
                    corr=coeff*bb;pp=aa-corr;proj_parts.append((pp,bb,corr))
                    proj2+=(pp.detach()**2).sum();removed2+=(corr.detach()**2).sum()
                projn=torch.sqrt(proj2+1e-12);tail_scale=KAPPA*projn/(gtn+1e-12)
                gbase=[pp+tail_scale*bb for pp,bb,_ in proj_parts]
                gbn=grad_norm(gbase)
                ids=balanced_indices(auth_origin,auth_phase,gidx)
                edge=edge_floor_loss(m,ref,auth_obs[ids])
                ge=torch.autograd.grad(edge,actor_params,allow_unused=True)
                gen=grad_norm(ge)
                alpha=min(BETA0,RHO*float(gbn.detach().cpu())/(float(gen.detach().cpu())+1e-12))
                opt.zero_grad(set_to_none=True)
                for i,p in enumerate(actor_params):
                    gg=gbase[i]
                    if ge[i] is not None:gg=gg+alpha*ge[i]
                    p.grad=gg
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                     "tail_fraction":float(tailfrac.detach().cpu()),"edge_loss":float(edge.detach().cpu()),
                     "edge_alpha":float(alpha),"edge_grad_norm":float(gen.detach().cpu()),
                     "weighted_edge_over_base":float(alpha*float(gen.detach().cpu())/(float(gbn.detach().cpu())+1e-12)),
                     "ratio_pre_refresh":ratio_pre,"ratio_maxerr":ratio_err,
                     "termination_fraction":main["termination_fraction"],"ppo_grad_norm":float(gpn.cpu()),
                     "tail_grad_norm":float(gtn.cpu()),"grad_cosine":float(cos.cpu()),
                     "removed_over_ppo":float((torch.sqrt(removed2)/(gpn+1e-12)).cpu()),
                     "tail_descent_over_projected":float((tail_scale*gtn/(projn+1e-12)).cpu()),
                     "grad_norm_preclip":total}
                rows.append(row);print("UPDATE",args.arm,uidx,json.dumps(row),flush=True)
                if uidx==args.updates:audit(m,probe,tau,out,uidx,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx,prefit_trace)
    
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
            jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
            rep={"schema":"authority_isolated_ai_c2_paired_train_v1","arm":args.arm,"updates":args.updates,
                 "source_checkpoint":str(SOURCE_CK.relative_to(ROOT)),"prefit_steps":PREFIT_STEPS,
                 "summary":{"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
                            "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                            "max_ratio_error":float(max(r["ratio_maxerr"] for r in rows[-args.updates:])),
                            "max_termination_fraction":float(max(r["termination_fraction"] for r in rows[-args.updates:]))},
                 "rows":rows,"snapshots":snaps,"prefit_trace":prefit_trace}
            (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("FINAL",args.arm,json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    
    if True:main()

def run_authority_isolated_ai_c2_endpoint_audit():
    """Run former authority_isolated_ai_c2_endpoint_audit.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
     "narrow":ROOT/"runs/authority_isolated_ai_c2_clean2_narrow-2026-09-25/model_10.pt",
     "wide":ROOT/"runs/authority_isolated_ai_c2_clean2_wide-2026-09-25/model_10.pt"}
    TR={
     "narrow":ROOT/"runs/authority_isolated_ai_c2_clean2_narrow-2026-09-25/training_report.json",
     "wide":ROOT/"runs/authority_isolated_ai_c2_clean2_wide-2026-09-25/training_report.json"}
    OUT=ROOT/"runs/authority_isolated_ai_c2_clean2_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S","C");G=.99;NENV=8;H=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),"C":np.array([.25]*4,np.float32)}
    ROBUST={"suite2":840003,"suite3":840004,"held4":850101,"held5":850202,"held6":850303}
    FRESH=[9300000+i*113 for i in range(4)]
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
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def rollout(env,m,w_np,seed,need_values=False):
     from talon_rl.rewards.objectives import normalized_objective_vector
     mgr=env.unwrapped.reward_manager;w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
     cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool);ft=np.full(NENV,-1,int)
     with torch.no_grad():
      for t in range(H):
       if need_values:V.append(m.value_with_preference(cur,w).cpu().numpy())
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
       dd=(te|tr).cpu().numpy().astype(bool);new=(ft<0)&dd;ft[new]=t;D.append(dd);done|=dd;cur=ot(nxt).cuda()
     q={"survival":float(1-done.mean()),"fail_count":int(done.sum()),"fail_t":ft.tolist()}
     if need_values:
      R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
      q.update({"h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
                "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
                "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]})
     return q
    def main():
     from isaaclab.app import AppLauncher
     sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      rep={"schema":"authority_isolated_ai_c2_clean2_endpoint_v1","arms":{}}
      for arm,ck in CK.items():
       cls=AuthorityIsolatedActorCritic if arm=="narrow" else AuthorityIsolatedWideCritic
       m=cls(o.shape[-1],12).cuda();m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
       train=json.load(open(TR[arm]));rob=[];fresh=[]
       for sname,seed in ROBUST.items():
        for pref in ORDER:
         q=rollout(env,m,PREFS[pref],seed,False);q.update({"suite":sname,"preference":pref,"seed":seed});rob.append(q)
         print("ROB",arm,sname,pref,q["survival"],q["fail_count"],flush=True)
       for pi,pref in enumerate(ORDER):
        for si,base in enumerate(FRESH):
         seed=base+pi*1000
         q=rollout(env,m,PREFS[pref],seed,True);q.update({"preference":pref,"suite":si,"seed":seed});fresh.append(q)
         print("FRESH",arm,pref,si,q["survival"],round(float(np.mean(q["h32_ev"])),3),round(float(np.mean(q["mc64_ev"])),3),flush=True)
       sem=[x for x in rob if x["suite"] in ("suite2","suite3")];held=[x for x in rob if x["suite"].startswith("held")]
       h=np.array([x["h32_ev"] for x in fresh]);mc=np.array([x["mc64_ev"] for x in fresh]);hb=np.array([x["h32_bias"] for x in fresh]);mb=np.array([x["mc64_bias"] for x in fresh])
       crit={"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),"mc64_ev_mean":float(mc.mean()),"mc64_negative_fraction":float((mc<0).mean()),
             "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),"min_survival":float(min(x["survival"] for x in fresh))}
       auth=train["summary"]["authority_gate"];ratio=train["summary"]["max_ratio_error"]
       gates={"authority":bool(auth),
              "critic":bool(crit["h32_ev_mean"]>0 and crit["mc64_ev_mean"]>0 and crit["h32_negative_fraction"]<=.25 and crit["mc64_negative_fraction"]<=.25 and crit["min_survival"]>=.95),
              "semantic_survival":bool(min(x["survival"] for x in sem)>=1.0),
              "heldout_survival":bool(min(x["survival"] for x in held)>=.95),
              "ppo_ratio":bool(ratio<=1e-4)}
       gates["pass"]=all(gates.values())
       rep["arms"][arm]={"training_summary":train["summary"],"robustness_rows":rob,"fresh_rows":fresh,
                         "semantic_min_survival":float(min(x["survival"] for x in sem)),"semantic_failed_lanes":int(sum(x["fail_count"] for x in sem)),
                         "heldout_min_survival":float(min(x["survival"] for x in held)),"heldout_failed_lanes":int(sum(x["fail_count"] for x in held)),
                         "critic":crit,"gates":gates}
      rep["decision"]={"wide_pass":rep["arms"]["wide"]["gates"]["pass"],
                       "narrow_pass":rep["arms"]["narrow"]["gates"]["pass"],
                       "AI_C2_pass":rep["arms"]["wide"]["gates"]["pass"],
                       "AI_H2_authorized":rep["arms"]["wide"]["gates"]["pass"]}
      (OUT/"ai_c2_endpoint_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
      print("DECISION",json.dumps(rep["decision"],indent=2),flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_authority_isolated_ai_c2_endpoint_eval():
    """Run former authority_isolated_ai_c2_endpoint_eval.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
    "narrow":ROOT/"runs/authority_isolated_ai_c2_clean2_narrow-2026-09-25/model_10.pt",
    "wide":ROOT/"runs/authority_isolated_ai_c2_clean2_wide-2026-09-25/model_10.pt"}
    OUT=ROOT/"runs/authority_isolated_ai_c2_endpoint_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S","C");PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEM={"suite2":840003,"suite3":840004};HELD={"suite4":850101,"suite5":850202,"suite6":850303}
    NENV=8;H=64;G=.99
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
    def rollout(env,m,wv,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool);reasons=[[] for _ in range(NENV)]
        with torch.no_grad():
            for t in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();done|=dd
                tm=env.unwrapped.termination_manager
                for i in range(NENV):
                    if dd[i]:
                        reasons[i]=[n for n in tm.active_terms if bool(tm.get_term(n)[i].detach().cpu())]
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                D.append(dd);cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        return {"survival":float(1-done.mean()),"fail_count":int(done.sum()),"reasons":reasons,
          "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
          "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
          "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
          "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def summarize(rows):
        h=np.array([r["h32_ev"] for r in rows]);m=np.array([r["mc64_ev"] for r in rows])
        hb=np.array([r["h32_bias"] for r in rows]);mb=np.array([r["mc64_bias"] for r in rows])
        return {"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
          "mc64_ev_mean":float(m.mean()),"mc64_negative_fraction":float((m<0).mean()),
          "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
          "min_survival":float(min(r["survival"] for r in rows)),"failed_lanes":int(sum(r["fail_count"] for r in rows))}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            rep={"schema":"authority_isolated_ai_c2_endpoint_eval_v1","models":{}}
            for arm,path in CK.items():
                cls=AuthorityIsolatedActorCritic if arm=="narrow" else AuthorityIsolatedWideCritic
                m=cls(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval()
                sem=[];held=[];fresh=[]
                for suite,seed in SEM.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});sem.append(q)
                for suite,seed in HELD.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});held.append(q)
                for li,lab in enumerate(ORDER):
                    for si in range(4):
                        seed=9700000+li*1000+si*113
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":si,"preference":lab,"seed":seed});fresh.append(q)
                fs=summarize(fresh);ss=summarize(sem);hs=summarize(held)
                critic_gate=bool(fs["h32_ev_mean"]>0 and fs["mc64_ev_mean"]>0 and fs["h32_negative_fraction"]<=.25 and fs["mc64_negative_fraction"]<=.25 and fs["min_survival"]>=.95)
                rep["models"][arm]={"semantic":ss,"heldout":hs,"fresh_critic":fs,"critic_gate":critic_gate,
                                    "semantic_rows":sem,"heldout_rows":held,"fresh_rows":fresh}
                print("SUMMARY",arm,json.dumps(rep["models"][arm],default=str)[:2000],flush=True)
            (OUT/"endpoint_eval.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_c2_formal_edge_authority_audit():
    """Run former authority_isolated_ai_c2_formal_edge_authority_audit.py stage."""
    """Simplex-edge endpoint authority for the formal narrow/wide edge pair."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.simplex_edge_endpoint import SimplexEdgeEndpointAudit
    
    
    class FormalEdgeAuthorityAudit(SimplexEdgeEndpointAudit):
        """Simplex-edge endpoint authority for the formal narrow/wide edge pair."""
    
        run = "authority_isolated_ai_c2_formal_edge_authority_audit-2026-09-25"
        narrow = ("control", "authority_isolated_ai_c2_formal_edge_narrow-2026-09-25/model_10.pt")
        wide = ("edge", "authority_isolated_ai_c2_formal_edge_wide-2026-09-25/model_10.pt")
    
    
    if True:
        FormalEdgeAuthorityAudit.main()

def run_authority_isolated_ai_c2_formal_edge_endpoint_eval():
    """Run former authority_isolated_ai_c2_formal_edge_endpoint_eval.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
    "narrow":ROOT/"runs/authority_isolated_ai_c2_formal_edge_narrow-2026-09-25/model_10.pt",
    "wide":ROOT/"runs/authority_isolated_ai_c2_formal_edge_wide-2026-09-25/model_10.pt"}
    OUT=ROOT/"runs/authority_isolated_ai_c2_formal_edge_endpoint_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S","C");PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEM={"suite2":840003,"suite3":840004};HELD={"suite4":850101,"suite5":850202,"suite6":850303}
    NENV=8;H=64;G=.99
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
    def rollout(env,m,wv,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool);reasons=[[] for _ in range(NENV)]
        with torch.no_grad():
            for t in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();done|=dd
                tm=env.unwrapped.termination_manager
                for i in range(NENV):
                    if dd[i]:
                        reasons[i]=[n for n in tm.active_terms if bool(tm.get_term(n)[i].detach().cpu())]
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                D.append(dd);cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        return {"survival":float(1-done.mean()),"fail_count":int(done.sum()),"reasons":reasons,
          "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
          "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
          "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
          "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def summarize(rows):
        h=np.array([r["h32_ev"] for r in rows]);m=np.array([r["mc64_ev"] for r in rows])
        hb=np.array([r["h32_bias"] for r in rows]);mb=np.array([r["mc64_bias"] for r in rows])
        return {"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
          "mc64_ev_mean":float(m.mean()),"mc64_negative_fraction":float((m<0).mean()),
          "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
          "min_survival":float(min(r["survival"] for r in rows)),"failed_lanes":int(sum(r["fail_count"] for r in rows))}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            rep={"schema":"authority_isolated_ai_c2_endpoint_eval_v1","models":{}}
            for arm,path in CK.items():
                cls=AuthorityIsolatedActorCritic if arm=="narrow" else AuthorityIsolatedWideCritic
                m=cls(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval()
                sem=[];held=[];fresh=[]
                for suite,seed in SEM.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});sem.append(q)
                for suite,seed in HELD.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});held.append(q)
                for li,lab in enumerate(ORDER):
                    for si in range(4):
                        seed=9700000+li*1000+si*113
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":si,"preference":lab,"seed":seed});fresh.append(q)
                fs=summarize(fresh);ss=summarize(sem);hs=summarize(held)
                critic_gate=bool(fs["h32_ev_mean"]>0 and fs["mc64_ev_mean"]>0 and fs["h32_negative_fraction"]<=.25 and fs["mc64_negative_fraction"]<=.25 and fs["min_survival"]>=.95)
                rep["models"][arm]={"semantic":ss,"heldout":hs,"fresh_critic":fs,"critic_gate":critic_gate,
                                    "semantic_rows":sem,"heldout_rows":held,"fresh_rows":fresh}
                print("SUMMARY",arm,json.dumps(rep["models"][arm],default=str)[:2000],flush=True)
            (OUT/"endpoint_eval.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_c2_formal_edge_train():
    """Run former authority_isolated_ai_c2_formal_edge_train.py stage."""
    
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
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SOURCE_CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    SOURCE_STATE=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/resume_state.pt"
    REF_CK=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    SUPPORT=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    NENV=h1.NENV;H=h1.H;G=.99
    ORDER=h1.ORDER;PREFS=h1.PREFS
    PREFIT_SEEDS=[960000+i*173 for i in range(6)]
    PREFIT_STEPS=2500;PREFIT_BATCH=1024
    KAPPA=.05;RHO=.25;BETA0=2.497041993384243;GAMMA=.90;EPS=1e-12
    ALL_PREFS=("T","A","O","S","C")
    EDGE_PAIRS=tuple((ALL_PREFS[i],ALL_PREFS[j]) for i in range(len(ALL_PREFS)) for j in range(i+1,len(ALL_PREFS)))
    
    def ot(x):
        return h1.ot(x)
    
    def copy_actor_exact(dst,src_state):
        tgt=dst.state_dict()
        for k in tgt:
            if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
                tgt[k].copy_(src_state[k])
        dst.load_state_dict(tgt)
    
    def terms(raw,names):
        return {n:raw[:,i] for i,n in enumerate(names)}
    
    def trunc_h32(rt,dt):
        # Phase-local H32 targets: bootstrap accumulator is reset every 32 steps.
        out=torch.zeros_like(rt)
        for st in range(0,len(rt),32):
            en=min(st+32,len(rt));run=torch.zeros_like(rt[-1])
            for t in range(en-1,st-1,-1):
                run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    
    def collect_prefit(env,m,mgr):
        from talon_rl.rewards.objectives import normalized_objective_vector
        X=[];W=[];Y=[]
        m.eval()
        for seed in PREFIT_SEEDS:
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];rw=[];dn=[]
                with torch.no_grad():
                    for _ in range(64):
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,te,tr,_=env.step(a)
                        raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                        vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                        obs.append(cur.cpu());rw.append(torch.tensor(vec)*env.unwrapped.step_dt)
                        dn.append((te|tr).cpu());cur=ot(nxt).cuda()
                rt=torch.stack(rw);dt=torch.stack(dn).bool();yy=trunc_h32(rt,dt)
                X.append(torch.cat(obs));W.append(w.cpu().repeat(64,1));Y.append(yy.reshape(-1,4))
        return torch.cat(X).float(),torch.cat(W).float(),torch.cat(Y).float()
    
    def prefit_critic(m,X,W,Y):
        params=list(m.critic_body.parameters())+list(m.critic_head.parameters())
        opt=torch.optim.Adam(params,lr=3e-4)
        rng=np.random.default_rng(2609252201)
        m.train();trace=[]
        for step in range(1,PREFIT_STEPS+1):
            idx=rng.integers(0,len(X),size=PREFIT_BATCH)
            xo=X[idx].cuda();ww=W[idx].cuda();yy=Y[idx].cuda()
            pv=m.value_with_preference(xo,ww);loss=(pv-yy).pow(2).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%250==0:trace.append({"step":step,"loss":float(loss.detach().cpu())})
        return trace
    
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    
    def balanced_indices(origin,phase,global_update):
        rng=np.random.default_rng(2609252401+global_update);ids=[]
        for oi in range(5):
            for pi in range(3):
                pool=np.flatnonzero((origin==oi)&(phase==pi))
                ids.extend(rng.choice(pool,size=8,replace=False).tolist())
        return np.asarray(ids,np.int64)
    
    def edge_floor_loss(m,ref,obs):
        n=len(obs);acts={};refs={}
        for lab in ALL_PREFS:
            w=torch.tensor(PREFS[lab],device="cuda").repeat(n,1)
            acts[lab]=m.act_inference_with_preference(obs,w)
            with torch.no_grad():
                refs[lab]=ref.act_inference_with_preference(obs,w)
        terms=[]
        for i,j in EDGE_PAIRS:
            d=torch.linalg.vector_norm(acts[i]-acts[j],dim=1)
            dr=torch.linalg.vector_norm(refs[i]-refs[j],dim=1)
            terms.append(torch.relu(GAMMA*dr-d).pow(2))
        return torch.stack(terms,dim=1).mean()
    
    def grad_norm(gs):
        return torch.sqrt(sum((g.detach()**2).sum() for g in gs if g is not None)+EPS)
    
    def audit(m,probe,tau,out,tag,snaps,arm):
        m.eval();sens=h1.sensitivity(m,probe)
        P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(P),1)
        with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
        snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
        torch.save({"model":m.state_dict(),"update":int(tag),"arm":arm},out/f"model_{tag}.pt")
        m.train()
    
    def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update,prefit_trace):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
          "prefit_trace":prefit_trace,"torch_rng":torch.get_rng_state(),
          "cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state()},path)
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--arm",choices=["narrow","wide"],required=True)
        ap.add_argument("--seed",type=int,default=73001)
        ap.add_argument("--updates",type=int,default=10)
        ap.add_argument("--run-tag",default="formal_edge")
        args=ap.parse_args()
        out=ROOT/f"runs/authority_isolated_ai_c2_{args.run_tag}_{args.arm}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        state_path=out/"resume_state.pt"
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
    
            source=torch.load(SOURCE_CK,map_location="cuda",weights_only=False)["model"]
            source_resume=torch.load(SOURCE_STATE,map_location="cpu",weights_only=False)
            ref=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda()
            ref.load_state_dict(torch.load(REF_CK,map_location="cuda",weights_only=False)["model"]);ref.eval()
            for p in ref.parameters():p.requires_grad_(False)
            sd=np.load(SUPPORT);ref_obs=torch.tensor(sd["obs"],device="cuda");origin=sd["origin"];phase=sd["phase"]
            cls=AuthorityIsolatedActorCritic if args.arm=="narrow" else AuthorityIsolatedWideCritic
            torch.manual_seed(26092520)
            m=cls(o.shape[-1],ad).cuda();copy_actor_exact(m,source)
    
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False)
                m.load_state_dict(st["model"])
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
                opt=torch.optim.Adam(actor_params,lr=1e-3);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
                prefit_trace=st["prefit_trace"]
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
                print("RESUME",args.arm,cur,flush=True)
            else:
                # Same frozen robust-u20 policy dataset for both critic architectures.
                X,W,Y=collect_prefit(env,m,mgr)
                print("PREFIT_DATA",args.arm,len(X),flush=True)
                prefit_trace=prefit_critic(m,X,W,Y)
                # Actor must remain exact after critic-only equilibration.
                chk=m.state_dict();max_actor=0.0
                for k,v in source.items():
                    if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
                        max_actor=max(max_actor,float((chk[k]-v).abs().max().detach().cpu()))
                if max_actor>1e-7:raise RuntimeError(f"actor changed during prefit {max_actor}")
                print("PREFIT_DONE",args.arm,prefit_trace[-1],"actor_err",max_actor,flush=True)
    
                # Freeze critic representation during actor-learning phase; linear head is refreshed analytically.
                for n,p in m.named_parameters():
                    if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
                opt=torch.optim.Adam(actor_params,lr=1e-3)
                # Load exact robust-u20 actor optimizer state; actor parameter ordering is architecture-invariant.
                opt.load_state_dict(source_resume["optimizer"])
    
                # Matched support topology; each arm builds features through its own critic representation.
                cand=[]
                for k in range(h1.ANCHOR_CANDIDATES):
                    _,w=h1.pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                    units=h1.collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
                # Exact same anchor/reset schedule for both critic architectures.
                anchor_specs=[tuple(x) for x in source_resume["anchor_specs"]]
                adaptive_pools={"early":[],"late":[]}
                for _,_,units in cand:
                    for u in units:adaptive_pools[u["phase"]].append(u)
                rows=[];snaps={};cur=0
                audit(m,probe,tau,out,0,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur,prefit_trace)
    
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
    
            # Ensure critic params remain frozen after resume too.
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
    
            # Formal AI-C2 continues from durability-valid global update 30.
            GLOBAL_START_UPDATE=30
            for uidx in range(cur+1,args.updates+1):
                gidx=GLOBAL_START_UPDATE+uidx
                labs,w=h1.pref_batch(gidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,args.seed+gidx*211,True)
                with torch.no_grad():
                    lp_pre=m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])
                    ratio_pre=float((torch.exp(lp_pre-main["old"])-1).abs().max().cpu())
                _,ws=h1.pref_batch(gidx+17,torch.device("cuda"))
                units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+gidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                anchors_now=current_anchor_units();h1.fit_expanded_current_policy(m,anchors_now,adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                lp_parts=[]
                for st in range(0,len(main["obs"]),NENV):
                    lp_parts.append(m.logp_from_pre_tanh_with_preference(main["obs"][st:st+NENV],main["w"][st:st+NENV],main["u"][st:st+NENV]))
                ratio=torch.exp(torch.cat(lp_parts)-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2);gtn=torch.sqrt(gt2)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                cos=dot/(gpn*gtn+1e-12);coeff=torch.minimum(dot/(gt2+1e-12),torch.zeros_like(dot))
                proj_parts=[];proj2=torch.zeros((),device="cuda");removed2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a;bb=torch.zeros_like(p) if b is None else b
                    corr=coeff*bb;pp=aa-corr;proj_parts.append((pp,bb,corr))
                    proj2+=(pp.detach()**2).sum();removed2+=(corr.detach()**2).sum()
                projn=torch.sqrt(proj2);tail_scale=KAPPA*projn/(gtn+1e-12)
                gbase=[pp+tail_scale*bb for pp,bb,_ in proj_parts]
                gbn=grad_norm(gbase)
    
                ids=balanced_indices(origin,phase,gidx)
                edge_loss=edge_floor_loss(m,ref,ref_obs[ids])
                ge=torch.autograd.grad(edge_loss,actor_params,allow_unused=True)
                gen=grad_norm(ge)
                alpha=min(BETA0,RHO*float(gbn.detach().cpu())/(float(gen.detach().cpu())+EPS))
    
                opt.zero_grad(set_to_none=True)
                for p,gb,gg in zip(actor_params,gbase,ge):
                    p.grad=gb if gg is None else gb+alpha*gg
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"global_update":gidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                     "tail_fraction":float(tailfrac.detach().cpu()),"edge_loss":float(edge_loss.detach().cpu()),
                     "edge_alpha":float(alpha),"edge_grad_norm":float(gen.detach().cpu()),
                     "weighted_edge_over_base":float(alpha*float(gen.detach().cpu())/(float(gbn.detach().cpu())+EPS)),
                     "ratio_pre_refresh":ratio_pre,"ratio_maxerr":ratio_err,
                     "termination_fraction":main["termination_fraction"],"ppo_grad_norm":float(gpn.cpu()),
                     "tail_grad_norm":float(gtn.cpu()),"grad_cosine":float(cos.cpu()),
                     "removed_over_ppo":float((torch.sqrt(removed2)/(gpn+1e-12)).cpu()),
                     "tail_descent_over_projected":float((tail_scale*gtn/(projn+1e-12)).cpu()),
                     "grad_norm_preclip":total}
                rows.append(row);print("UPDATE",args.arm,uidx,json.dumps(row),flush=True)
                if uidx==args.updates:audit(m,probe,tau,out,uidx,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx,prefit_trace)
    
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
            jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
            rep={"schema":"authority_isolated_ai_c2_paired_train_v1","arm":args.arm,"updates":args.updates,
                 "source_checkpoint":str(SOURCE_CK.relative_to(ROOT)),"prefit_steps":PREFIT_STEPS,
                 "summary":{"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
                            "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                            "max_ratio_error":float(max(r["ratio_maxerr"] for r in rows[-args.updates:])),
                            "max_termination_fraction":float(max(r["termination_fraction"] for r in rows[-args.updates:]))},
                 "rows":rows,"snapshots":snaps,"prefit_trace":prefit_trace}
            (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("FINAL",args.arm,json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    
    if True:main()

def run_authority_isolated_ai_c2_inherited_wide():
    """Run former authority_isolated_ai_c2_inherited_wide.py stage."""
    
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
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SOURCE_CK=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    SOURCE_STATE=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt"
    NARROW_DISTILLED=ROOT/"runs/authority_isolated_ai_c2_capacity_init-2026-09-25/narrow_distilled.pt"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    NENV=h1.NENV;H=h1.H;G=.99
    ORDER=h1.ORDER;PREFS=h1.PREFS
    PREFIT_SEEDS=[960000+i*173 for i in range(6)]
    PREFIT_STEPS=2500;PREFIT_BATCH=1024
    KAPPA=.05
    
    def ot(x):
        return h1.ot(x)
    
    def copy_actor_exact(dst,src_state):
        tgt=dst.state_dict()
        for k in tgt:
            if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
                tgt[k].copy_(src_state[k])
        dst.load_state_dict(tgt)
    
    def terms(raw,names):
        return {n:raw[:,i] for i,n in enumerate(names)}
    
    def trunc_h32(rt,dt):
        # Phase-local H32 targets: bootstrap accumulator is reset every 32 steps.
        out=torch.zeros_like(rt)
        for st in range(0,len(rt),32):
            en=min(st+32,len(rt));run=torch.zeros_like(rt[-1])
            for t in range(en-1,st-1,-1):
                run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    
    def collect_prefit(env,m,mgr):
        from talon_rl.rewards.objectives import normalized_objective_vector
        X=[];W=[];Y=[]
        m.eval()
        for seed in PREFIT_SEEDS:
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];rw=[];dn=[]
                with torch.no_grad():
                    for _ in range(64):
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,te,tr,_=env.step(a)
                        raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                        vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                        obs.append(cur.cpu());rw.append(torch.tensor(vec)*env.unwrapped.step_dt)
                        dn.append((te|tr).cpu());cur=ot(nxt).cuda()
                rt=torch.stack(rw);dt=torch.stack(dn).bool();yy=trunc_h32(rt,dt)
                X.append(torch.cat(obs));W.append(w.cpu().repeat(64,1));Y.append(yy.reshape(-1,4))
        return torch.cat(X).float(),torch.cat(W).float(),torch.cat(Y).float()
    
    def prefit_critic(m,X,W,Y):
        params=list(m.critic_body.parameters())+list(m.critic_head.parameters())
        opt=torch.optim.Adam(params,lr=3e-4)
        rng=np.random.default_rng(2609252201)
        m.train();trace=[]
        for step in range(1,PREFIT_STEPS+1):
            idx=rng.integers(0,len(X),size=PREFIT_BATCH)
            xo=X[idx].cuda();ww=W[idx].cuda();yy=Y[idx].cuda()
            pv=m.value_with_preference(xo,ww);loss=(pv-yy).pow(2).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%250==0:trace.append({"step":step,"loss":float(loss.detach().cpu())})
        return trace
    
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    
    def audit(m,probe,tau,out,tag,snaps,arm):
        m.eval();sens=h1.sensitivity(m,probe)
        P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(P),1)
        with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
        snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
        torch.save({"model":m.state_dict(),"update":int(tag),"arm":arm},out/f"model_{tag}.pt")
        m.train()
    
    def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update,prefit_trace):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
          "prefit_trace":prefit_trace,"torch_rng":torch.get_rng_state(),
          "cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state()},path)
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--arm",choices=["narrow","wide"],required=True)
        ap.add_argument("--seed",type=int,default=73001)
        ap.add_argument("--updates",type=int,default=10)
        ap.add_argument("--run-tag",default="clean2")
        args=ap.parse_args()
        out=ROOT/f"runs/authority_isolated_ai_c2_{args.run_tag}_{args.arm}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        state_path=out/"resume_state.pt"
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
    
            source=torch.load(SOURCE_CK,map_location="cuda",weights_only=False)["model"]
            source_resume=torch.load(SOURCE_STATE,map_location="cpu",weights_only=False)
            cls=AuthorityIsolatedActorCritic if args.arm=="narrow" else AuthorityIsolatedWideCritic
            torch.manual_seed(26092520)
            m=cls(o.shape[-1],ad).cuda()
    
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False)
                m.load_state_dict(st["model"])
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
                opt=torch.optim.Adam(actor_params,lr=1e-3);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
                prefit_trace=st["prefit_trace"]
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
                print("RESUME",args.arm,cur,flush=True)
            else:
                # Capacity-only initialization:
                # wide = inherited robust-u20 critic exact;
                # narrow = best-approximation distillation to that inherited wide critic.
                if args.arm=="wide":
                    m.load_state_dict(source)
                    prefit_trace=[{"mode":"inherited_exact"}]
                else:
                    m.load_state_dict(torch.load(NARROW_DISTILLED,map_location="cuda",weights_only=False)["model"])
                    prefit_trace=[{"mode":"distilled_to_inherited_wide"}]
                chk=m.state_dict();max_actor=0.0
                for k,v in source.items():
                    if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
                        max_actor=max(max_actor,float((chk[k]-v).abs().max().detach().cpu()))
                if max_actor>1e-7:raise RuntimeError(f"actor mismatch at capacity init {max_actor}")
                print("CAPACITY_INIT",args.arm,prefit_trace[-1],"actor_err",max_actor,flush=True)
    
                for n,p in m.named_parameters():
                    if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
                opt=torch.optim.Adam(actor_params,lr=1e-3)
                opt.load_state_dict(source_resume["optimizer"])
    
                # Rebuild matched support topology from the same actor/reset seeds for both architectures.
                cand=[]
                for k in range(h1.ANCHOR_CANDIDATES):
                    _,w=h1.pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                    units=h1.collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
                anchor_specs=[tuple(x) for x in source_resume["anchor_specs"]]
                adaptive_pools={"early":[],"late":[]}
                for _,_,units in cand:
                    for u in units:adaptive_pools[u["phase"]].append(u)
                rows=[];snaps={};cur=0
                audit(m,probe,tau,out,0,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur,prefit_trace)
    
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
    
            # Ensure critic params remain frozen after resume too.
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
    
            # This is continuation from the robust global update-20 checkpoint.
            # Local AI-C2 steps 1..10 must therefore use the *new* global update
            # schedule 21..30 rather than replaying reset/support seeds from 1..10.
            GLOBAL_START_UPDATE=20
            for uidx in range(cur+1,args.updates+1):
                gidx=GLOBAL_START_UPDATE+uidx
                labs,w=h1.pref_batch(gidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,args.seed+gidx*211,True)
                with torch.no_grad():
                    lp_pre=m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])
                    ratio_pre=float((torch.exp(lp_pre-main["old"])-1).abs().max().cpu())
                _,ws=h1.pref_batch(gidx+17,torch.device("cuda"))
                units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+gidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                anchors_now=current_anchor_units();h1.fit_expanded_current_policy(m,anchors_now,adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                lp_parts=[]
                for st in range(0,len(main["obs"]),NENV):
                    lp_parts.append(m.logp_from_pre_tanh_with_preference(main["obs"][st:st+NENV],main["w"][st:st+NENV],main["u"][st:st+NENV]))
                ratio=torch.exp(torch.cat(lp_parts)-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2);gtn=torch.sqrt(gt2)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                cos=dot/(gpn*gtn+1e-12);coeff=torch.minimum(dot/(gt2+1e-12),torch.zeros_like(dot))
                proj_parts=[];proj2=torch.zeros((),device="cuda");removed2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a;bb=torch.zeros_like(p) if b is None else b
                    corr=coeff*bb;pp=aa-corr;proj_parts.append((pp,bb,corr))
                    proj2+=(pp.detach()**2).sum();removed2+=(corr.detach()**2).sum()
                projn=torch.sqrt(proj2);tail_scale=KAPPA*projn/(gtn+1e-12)
                opt.zero_grad(set_to_none=True)
                for p,(pp,bb,corr) in zip(actor_params,proj_parts):p.grad=pp+tail_scale*bb
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                     "tail_fraction":float(tailfrac.detach().cpu()),"ratio_pre_refresh":ratio_pre,"ratio_maxerr":ratio_err,
                     "termination_fraction":main["termination_fraction"],"ppo_grad_norm":float(gpn.cpu()),
                     "tail_grad_norm":float(gtn.cpu()),"grad_cosine":float(cos.cpu()),
                     "removed_over_ppo":float((torch.sqrt(removed2)/(gpn+1e-12)).cpu()),
                     "tail_descent_over_projected":float((tail_scale*gtn/(projn+1e-12)).cpu()),
                     "grad_norm_preclip":total}
                rows.append(row);print("UPDATE",args.arm,uidx,json.dumps(row),flush=True)
                if uidx==args.updates:audit(m,probe,tau,out,uidx,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx,prefit_trace)
    
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
            jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
            rep={"schema":"authority_isolated_ai_c2_paired_train_v1","arm":args.arm,"updates":args.updates,
                 "source_checkpoint":str(SOURCE_CK.relative_to(ROOT)),"prefit_steps":PREFIT_STEPS,
                 "summary":{"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
                            "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                            "max_ratio_error":float(max(r["ratio_maxerr"] for r in rows[-args.updates:])),
                            "max_termination_fraction":float(max(r["termination_fraction"] for r in rows[-args.updates:]))},
                 "rows":rows,"snapshots":snaps,"prefit_trace":prefit_trace}
            (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("FINAL",args.arm,json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    
    if True:main()

def run_authority_isolated_ai_c2_inherited_wide_eval():
    """Run former authority_isolated_ai_c2_inherited_wide_eval.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK=ROOT/"runs/authority_isolated_ai_c2_inherited_wide_wide-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_ai_c2_inherited_wide_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S","C");PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    ROB={"suite2":840003,"suite3":840004,"held4":850101,"held5":850202,"held6":850303}
    FRESH=[9300000+i*113 for i in range(4)]
    NENV=8;H=64;G=.99
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
            for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def rollout(env,m,wv,seed,need_values=False):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool);ft=np.full(NENV,-1,int)
        with torch.no_grad():
            for t in range(H):
                if need_values:V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy().astype(bool);new=(ft<0)&dd;ft[new]=t;D.append(dd);done|=dd;cur=ot(nxt).cuda()
        q={"survival":float(1-done.mean()),"fail_count":int(done.sum()),"fail_t":ft.tolist()}
        if need_values:
            R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
            q.update({"h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
                      "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                      "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
                      "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]})
        return q
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
            rob=[];fresh=[]
            for sname,seed in ROB.items():
                for pref in ORDER:
                    q=rollout(env,m,PREFS[pref],seed,False);q.update({"suite":sname,"preference":pref});rob.append(q);print("ROB",sname,pref,q["survival"],flush=True)
            for pi,pref in enumerate(ORDER):
                for si,base in enumerate(FRESH):
                    seed=base+pi*1000;q=rollout(env,m,PREFS[pref],seed,True);q.update({"preference":pref,"suite":si});fresh.append(q);print("FRESH",pref,si,q["survival"],flush=True)
            sem=[x for x in rob if x["suite"] in ("suite2","suite3")];held=[x for x in rob if x["suite"].startswith("held")]
            h=np.array([x["h32_ev"] for x in fresh]);mc=np.array([x["mc64_ev"] for x in fresh]);hb=np.array([x["h32_bias"] for x in fresh]);mb=np.array([x["mc64_bias"] for x in fresh])
            crit={"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),"mc64_ev_mean":float(mc.mean()),"mc64_negative_fraction":float((mc<0).mean()),
                  "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),"min_survival":float(min(x["survival"] for x in fresh))}
            rep={"semantic_min_survival":float(min(x["survival"] for x in sem)),"semantic_failed_lanes":int(sum(x["fail_count"] for x in sem)),
                 "heldout_min_survival":float(min(x["survival"] for x in held)),"heldout_failed_lanes":int(sum(x["fail_count"] for x in held)),
                 "critic":crit,"fresh_rows":fresh,"robustness_rows":rob}
            (OUT/"eval.json").write_text(json.dumps(rep,indent=2)+"\n");print("SUMMARY",json.dumps(rep,indent=2)[:2500],flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_c2_offline_credit_surface():
    """Run former authority_isolated_ai_c2_offline_credit_surface.py stage."""
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    CK={
     "inherited_wide":ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt",
     "refit_wide":ROOT/"runs/authority_isolated_ai_c2_clean2_wide-2026-09-25/model_0.pt",
     "refit_narrow":ROOT/"runs/authority_isolated_ai_c2_clean2_narrow-2026-09-25/model_0.pt"}
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    OUT=ROOT/"runs/authority_isolated_ai_c2_credit_surface-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    def corr(a,b):
     a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
     return float(np.corrcoef(a,b)[0,1])
    def jacobian_tangent(m,obs,w):
     # tangent basis on simplex
     B=torch.tensor([[1.,-1,0,0],[1,0,-1,0],[1,0,0,-1]],device=obs.device)
     vals=[]
     for i in range(len(obs)):
      wi=w[i:i+1].clone().detach().requires_grad_(True)
      oi=obs[i:i+1]
      v=m.value_with_preference(oi,wi).squeeze(0)
      J=[]
      for j in range(4):
       g=torch.autograd.grad(v[j],wi,retain_graph=True)[0].squeeze(0)
       J.append(B@g)
      vals.append(torch.stack(J)) # [4,3]
     return torch.stack(vals)
    def main():
     X=np.load(PROBE)["obs"].astype(np.float32)
     obs=torch.tensor(X,device="cuda")
     models={}
     for k,p in CK.items():
      cls=AuthorityIsolatedActorCritic if k=="refit_narrow" else AuthorityIsolatedWideCritic
      m=cls(obs.shape[1],12).cuda();m.load_state_dict(torch.load(p,map_location="cuda",weights_only=False)["model"]);m.eval();models[k]=m
     # actor exactness
     src=models["inherited_wide"].state_dict();errs={}
     for k,m in models.items():
      e=0.
      for n,v in m.state_dict().items():
       if n.startswith("actor_") or n.startswith("family_") or n=="log_std":
        e=max(e,float((v-src[n]).abs().max().detach().cpu()))
      errs[k]=e
     rep={"schema":"ai_c2_offline_credit_surface_v1","actor_max_error":errs,"preferences":{}}
     ref=models["inherited_wide"]
     for lab,wv in PREFS.items():
      w=torch.tensor(wv,device="cuda").repeat(len(obs),1)
      with torch.no_grad():vr=ref.value_with_preference(obs,w).cpu().numpy()
      Jr=jacobian_tangent(ref,obs,w).detach().cpu().numpy()
      q={}
      for k in ("refit_wide","refit_narrow"):
       m=models[k]
       with torch.no_grad():v=m.value_with_preference(obs,w).cpu().numpy()
       J=jacobian_tangent(m,obs,w).detach().cpu().numpy()
       d=v-vr
       jd=J-Jr
       q[k]={
        "value_rmse":float(np.sqrt(np.mean(d*d))),
        "value_mae":float(np.mean(np.abs(d))),
        "value_corr_all":corr(vr,v),
        "value_corr_by_head":[corr(vr[:,j],v[:,j]) for j in range(4)],
        "value_mean_shift_by_head":[float(np.mean(d[:,j])) for j in range(4)],
        "tangent_jacobian_rmse":float(np.sqrt(np.mean(jd*jd))),
        "tangent_jacobian_norm_ratio":float(np.linalg.norm(J)/(np.linalg.norm(Jr)+1e-12)),
        "tangent_jacobian_cosine":float(np.sum(J*Jr)/(np.linalg.norm(J)*np.linalg.norm(Jr)+1e-12))
       }
      rep["preferences"][lab]=q
     agg={}
     for k in ("refit_wide","refit_narrow"):
      keys=["value_rmse","value_mae","value_corr_all","tangent_jacobian_rmse","tangent_jacobian_norm_ratio","tangent_jacobian_cosine"]
      agg[k]={z:float(np.mean([rep["preferences"][lab][k][z] for lab in PREFS])) for z in keys}
     rep["aggregate"]=agg
     (OUT/"credit_surface.json").write_text(json.dumps(rep,indent=2)+"\n")
     print(json.dumps(rep,indent=2))
    if True:main()

def run_authority_isolated_ai_c2_paired_train():
    """Run former authority_isolated_ai_c2_paired_train.py stage."""
    
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
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SOURCE_CK=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    SOURCE_STATE=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    NENV=h1.NENV;H=h1.H;G=.99
    ORDER=h1.ORDER;PREFS=h1.PREFS
    PREFIT_SEEDS=[960000+i*173 for i in range(6)]
    PREFIT_STEPS=2500;PREFIT_BATCH=1024
    KAPPA=.05
    
    def ot(x):
        return h1.ot(x)
    
    def copy_actor_exact(dst,src_state):
        tgt=dst.state_dict()
        for k in tgt:
            if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
                tgt[k].copy_(src_state[k])
        dst.load_state_dict(tgt)
    
    def terms(raw,names):
        return {n:raw[:,i] for i,n in enumerate(names)}
    
    def trunc_h32(rt,dt):
        # Phase-local H32 targets: bootstrap accumulator is reset every 32 steps.
        out=torch.zeros_like(rt)
        for st in range(0,len(rt),32):
            en=min(st+32,len(rt));run=torch.zeros_like(rt[-1])
            for t in range(en-1,st-1,-1):
                run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    
    def collect_prefit(env,m,mgr):
        from talon_rl.rewards.objectives import normalized_objective_vector
        X=[];W=[];Y=[]
        m.eval()
        for seed in PREFIT_SEEDS:
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];rw=[];dn=[]
                with torch.no_grad():
                    for _ in range(64):
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,te,tr,_=env.step(a)
                        raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                        vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                        obs.append(cur.cpu());rw.append(torch.tensor(vec)*env.unwrapped.step_dt)
                        dn.append((te|tr).cpu());cur=ot(nxt).cuda()
                rt=torch.stack(rw);dt=torch.stack(dn).bool();yy=trunc_h32(rt,dt)
                X.append(torch.cat(obs));W.append(w.cpu().repeat(64,1));Y.append(yy.reshape(-1,4))
        return torch.cat(X).float(),torch.cat(W).float(),torch.cat(Y).float()
    
    def prefit_critic(m,X,W,Y):
        params=list(m.critic_body.parameters())+list(m.critic_head.parameters())
        opt=torch.optim.Adam(params,lr=3e-4)
        rng=np.random.default_rng(2609252201)
        m.train();trace=[]
        for step in range(1,PREFIT_STEPS+1):
            idx=rng.integers(0,len(X),size=PREFIT_BATCH)
            xo=X[idx].cuda();ww=W[idx].cuda();yy=Y[idx].cuda()
            pv=m.value_with_preference(xo,ww);loss=(pv-yy).pow(2).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%250==0:trace.append({"step":step,"loss":float(loss.detach().cpu())})
        return trace
    
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    
    def audit(m,probe,tau,out,tag,snaps,arm):
        m.eval();sens=h1.sensitivity(m,probe)
        P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(P),1)
        with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
        snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
        torch.save({"model":m.state_dict(),"update":int(tag),"arm":arm},out/f"model_{tag}.pt")
        m.train()
    
    def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update,prefit_trace):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
          "prefit_trace":prefit_trace,"torch_rng":torch.get_rng_state(),
          "cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state()},path)
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--arm",choices=["narrow","wide"],required=True)
        ap.add_argument("--seed",type=int,default=73001)
        ap.add_argument("--updates",type=int,default=10)
        ap.add_argument("--run-tag",default="clean2")
        args=ap.parse_args()
        out=ROOT/f"runs/authority_isolated_ai_c2_{args.run_tag}_{args.arm}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        state_path=out/"resume_state.pt"
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
    
            source=torch.load(SOURCE_CK,map_location="cuda",weights_only=False)["model"]
            source_resume=torch.load(SOURCE_STATE,map_location="cpu",weights_only=False)
            cls=AuthorityIsolatedActorCritic if args.arm=="narrow" else AuthorityIsolatedWideCritic
            torch.manual_seed(26092520)
            m=cls(o.shape[-1],ad).cuda();copy_actor_exact(m,source)
    
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False)
                m.load_state_dict(st["model"])
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
                opt=torch.optim.Adam(actor_params,lr=1e-3);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
                prefit_trace=st["prefit_trace"]
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
                print("RESUME",args.arm,cur,flush=True)
            else:
                # Same frozen robust-u20 policy dataset for both critic architectures.
                X,W,Y=collect_prefit(env,m,mgr)
                print("PREFIT_DATA",args.arm,len(X),flush=True)
                prefit_trace=prefit_critic(m,X,W,Y)
                # Actor must remain exact after critic-only equilibration.
                chk=m.state_dict();max_actor=0.0
                for k,v in source.items():
                    if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
                        max_actor=max(max_actor,float((chk[k]-v).abs().max().detach().cpu()))
                if max_actor>1e-7:raise RuntimeError(f"actor changed during prefit {max_actor}")
                print("PREFIT_DONE",args.arm,prefit_trace[-1],"actor_err",max_actor,flush=True)
    
                # Freeze critic representation during actor-learning phase; linear head is refreshed analytically.
                for n,p in m.named_parameters():
                    if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
                opt=torch.optim.Adam(actor_params,lr=1e-3)
                # Load exact robust-u20 actor optimizer state; actor parameter ordering is architecture-invariant.
                opt.load_state_dict(source_resume["optimizer"])
    
                # Matched support topology; each arm builds features through its own critic representation.
                cand=[]
                for k in range(h1.ANCHOR_CANDIDATES):
                    _,w=h1.pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                    units=h1.collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
                # Exact same anchor/reset schedule for both critic architectures.
                anchor_specs=[tuple(x) for x in source_resume["anchor_specs"]]
                adaptive_pools={"early":[],"late":[]}
                for _,_,units in cand:
                    for u in units:adaptive_pools[u["phase"]].append(u)
                rows=[];snaps={};cur=0
                audit(m,probe,tau,out,0,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur,prefit_trace)
    
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
    
            # Ensure critic params remain frozen after resume too.
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
    
            # This is continuation from the robust global update-20 checkpoint.
            # Local AI-C2 steps 1..10 must therefore use the *new* global update
            # schedule 21..30 rather than replaying reset/support seeds from 1..10.
            GLOBAL_START_UPDATE=20
            for uidx in range(cur+1,args.updates+1):
                gidx=GLOBAL_START_UPDATE+uidx
                labs,w=h1.pref_batch(gidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,args.seed+gidx*211,True)
                with torch.no_grad():
                    lp_pre=m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])
                    ratio_pre=float((torch.exp(lp_pre-main["old"])-1).abs().max().cpu())
                _,ws=h1.pref_batch(gidx+17,torch.device("cuda"))
                units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+gidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                anchors_now=current_anchor_units();h1.fit_expanded_current_policy(m,anchors_now,adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                lp_parts=[]
                for st in range(0,len(main["obs"]),NENV):
                    lp_parts.append(m.logp_from_pre_tanh_with_preference(main["obs"][st:st+NENV],main["w"][st:st+NENV],main["u"][st:st+NENV]))
                ratio=torch.exp(torch.cat(lp_parts)-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2);gtn=torch.sqrt(gt2)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                cos=dot/(gpn*gtn+1e-12);coeff=torch.minimum(dot/(gt2+1e-12),torch.zeros_like(dot))
                proj_parts=[];proj2=torch.zeros((),device="cuda");removed2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a;bb=torch.zeros_like(p) if b is None else b
                    corr=coeff*bb;pp=aa-corr;proj_parts.append((pp,bb,corr))
                    proj2+=(pp.detach()**2).sum();removed2+=(corr.detach()**2).sum()
                projn=torch.sqrt(proj2);tail_scale=KAPPA*projn/(gtn+1e-12)
                opt.zero_grad(set_to_none=True)
                for p,(pp,bb,corr) in zip(actor_params,proj_parts):p.grad=pp+tail_scale*bb
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                     "tail_fraction":float(tailfrac.detach().cpu()),"ratio_pre_refresh":ratio_pre,"ratio_maxerr":ratio_err,
                     "termination_fraction":main["termination_fraction"],"ppo_grad_norm":float(gpn.cpu()),
                     "tail_grad_norm":float(gtn.cpu()),"grad_cosine":float(cos.cpu()),
                     "removed_over_ppo":float((torch.sqrt(removed2)/(gpn+1e-12)).cpu()),
                     "tail_descent_over_projected":float((tail_scale*gtn/(projn+1e-12)).cpu()),
                     "grad_norm_preclip":total}
                rows.append(row);print("UPDATE",args.arm,uidx,json.dumps(row),flush=True)
                if uidx==args.updates:audit(m,probe,tau,out,uidx,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx,prefit_trace)
    
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
            jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
            rep={"schema":"authority_isolated_ai_c2_paired_train_v1","arm":args.arm,"updates":args.updates,
                 "source_checkpoint":str(SOURCE_CK.relative_to(ROOT)),"prefit_steps":PREFIT_STEPS,
                 "summary":{"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
                            "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                            "max_ratio_error":float(max(r["ratio_maxerr"] for r in rows[-args.updates:])),
                            "max_termination_fraction":float(max(r["termination_fraction"] for r in rows[-args.updates:]))},
                 "rows":rows,"snapshots":snaps,"prefit_trace":prefit_trace}
            (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("FINAL",args.arm,json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    
    if True:main()

STAGES = {
    "authority_isolated_ai_c2_capacity_init": run_authority_isolated_ai_c2_capacity_init,
    "authority_isolated_ai_c2_continuity_control": run_authority_isolated_ai_c2_continuity_control,
    "authority_isolated_ai_c2_credit_geometry_audit": run_authority_isolated_ai_c2_credit_geometry_audit,
    "authority_isolated_ai_c2_edge_endpoint_audit": run_authority_isolated_ai_c2_edge_endpoint_audit,
    "authority_isolated_ai_c2_edge_noregression_eval": run_authority_isolated_ai_c2_edge_noregression_eval,
    "authority_isolated_ai_c2_edge_paired_train": run_authority_isolated_ai_c2_edge_paired_train,
    "authority_isolated_ai_c2_endpoint_audit": run_authority_isolated_ai_c2_endpoint_audit,
    "authority_isolated_ai_c2_endpoint_eval": run_authority_isolated_ai_c2_endpoint_eval,
    "authority_isolated_ai_c2_formal_edge_authority_audit": run_authority_isolated_ai_c2_formal_edge_authority_audit,
    "authority_isolated_ai_c2_formal_edge_endpoint_eval": run_authority_isolated_ai_c2_formal_edge_endpoint_eval,
    "authority_isolated_ai_c2_formal_edge_train": run_authority_isolated_ai_c2_formal_edge_train,
    "authority_isolated_ai_c2_inherited_wide": run_authority_isolated_ai_c2_inherited_wide,
    "authority_isolated_ai_c2_inherited_wide_eval": run_authority_isolated_ai_c2_inherited_wide_eval,
    "authority_isolated_ai_c2_offline_credit_surface": run_authority_isolated_ai_c2_offline_credit_surface,
    "authority_isolated_ai_c2_paired_train": run_authority_isolated_ai_c2_paired_train,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
