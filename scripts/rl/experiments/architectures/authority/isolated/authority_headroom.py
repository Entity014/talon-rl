"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_coordinate_headroom_train():
    """Run former authority_isolated_coordinate_headroom_train.py stage."""
    
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
    NENV=h1.NENV;H=h1.H;SNAPS=(0,10)
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w)
        excess=torch.relu(z.abs()-tau)
        loss=((excess/(tau+1e-6))**2).mean()
        frac=(z.abs()>tau).float().mean()
        return loss,frac,z
    def load_model(obs_dim,ad):
        m=AuthorityIsolatedWideCritic(obs_dim,ad).cuda()
        m.load_state_dict(torch.load(BASE,map_location="cuda",weights_only=False)["model"])
        return m
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["control","treatment"],required=True)
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--updates",type=int,default=10)
        args=ap.parse_args();lam=0.0 if args.arm=="control" else 0.01
        out=ROOT/f"runs/authority_isolated_coordinate_headroom_{args.arm}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=h1.ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=load_model(o.shape[-1],ad);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
            cand=[]
            for k in range(h1.ANCHOR_CANDIDATES):
                _,w=h1.pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=h1.collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=h1.select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
            adaptive_pools={"early":[],"late":[]}
            for _,_,units in cand:
                for u in units:adaptive_pools[u["phase"]].append(u)
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
            h1.fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
            rows=[];snaps={}
            def audit(tag):
                m.eval()
                sens=h1.sensitivity(m,probe)
                P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
                wm=torch.tensor(h1.PREFS["C"],device="cuda").repeat(len(P),1)
                with torch.no_grad():tl,tf,_=tail_metrics(m,P,wm,tau)
                snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed,"arm":args.arm,"lambda_tail":lam},out/f"model_{tag}.pt")
                m.train()
            audit(0)
            for uidx in range(1,args.updates+1):
                labs,w=h1.pref_batch(uidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                _,ws=h1.pref_batch(uidx+17,torch.device("cuda"))
                units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                h1.fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac,_=tail_metrics(m,main["obs"],main["w"],tau)
                loss=ppo+lam*tail
                opt.zero_grad(set_to_none=True);loss.backward()
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                rows.append({"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                             "tail_fraction":float(tailfrac.detach().cpu()),"total_loss":float(loss.detach().cpu()),
                             "grad_norm_preclip":total,"ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"]})
                print("UPDATE",uidx,"ppo",float(ppo.detach().cpu()),"tail",float(tail.detach().cpu()),
                      "tail_frac",float(tailfrac.detach().cpu()),"term",main["termination_fraction"],flush=True)
                if uidx in SNAPS:audit(uidx)
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
            jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
            rep={"schema":"authority_isolated_coordinate_headroom_train_v1","arm":args.arm,"lambda_tail":lam,
                 "seed":args.seed,"updates":args.updates,"tau_source":str(TAU_PATH.relative_to(ROOT)),
                 "base_checkpoint":str(BASE.relative_to(ROOT)),"rows":rows,"snapshots":snaps,
                 "summary":{"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
                            "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                            "probe_tail_loss_ratio":snaps[str(args.updates)]["probe_tail_loss"]/(snaps["0"]["probe_tail_loss"]+1e-12),
                            "probe_tail_fraction_delta":snaps[str(args.updates)]["probe_tail_fraction"]-snaps["0"]["probe_tail_fraction"]}}
            (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print(json.dumps(rep["summary"],indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_coordinate_headroom_train_chunked():
    """Run former authority_isolated_coordinate_headroom_train_chunked.py stage."""
    
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
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["control","treatment"],required=True)
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--target-updates",type=int,default=10)
        ap.add_argument("--chunk-updates",type=int,default=3);args=ap.parse_args()
        lam=0.0 if args.arm=="control" else 0.01
        out=ROOT/f"runs/authority_isolated_coordinate_headroom_{args.arm}-2026-09-25"
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
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau);loss=ppo+lam*tail
                print("AFTER_TAIL",uidx,flush=True)
                print("BEFORE_BACKWARD",uidx,flush=True)
                opt.zero_grad(set_to_none=True);loss.backward()
                print("AFTER_BACKWARD",uidx,flush=True)
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                print("AFTER_STEP",uidx,flush=True)
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                     "tail_fraction":float(tailfrac.detach().cpu()),"total_loss":float(loss.detach().cpu()),
                     "grad_norm_preclip":total,"ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"]}
                rows.append(row);print("UPDATE",uidx,json.dumps(row),flush=True)
                if uidx==args.target_updates:audit(m,probe,tau,out,uidx,snaps,args,lam)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx)
            print("CHUNK_DONE",end,flush=True)
            if end==args.target_updates:
                s0=snaps["0"]["sensitivity"];sf=snaps[str(end)]["sensitivity"]
                pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
                jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
                rep={"schema":"authority_isolated_coordinate_headroom_train_v2_chunked","arm":args.arm,"lambda_tail":lam,
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

def run_authority_isolated_coordinate_headroom_train_contiguous():
    """Run former authority_isolated_coordinate_headroom_train_contiguous.py stage."""
    
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
    NENV=h1.NENV;H=h1.H;SNAPS=(0,10)
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w)
        excess=torch.relu(z.abs()-tau)
        loss=((excess/(tau+1e-6))**2).mean()
        frac=(z.abs()>tau).float().mean()
        return loss,frac,z
    def load_model(obs_dim,ad):
        m=AuthorityIsolatedWideCritic(obs_dim,ad).cuda()
        m.load_state_dict(torch.load(BASE,map_location="cuda",weights_only=False)["model"])
        return m
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["control","treatment"],required=True)
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--updates",type=int,default=10)
        args=ap.parse_args();lam=0.0 if args.arm=="control" else 0.01
        out=ROOT/f"runs/authority_isolated_coordinate_headroom_{args.arm}_contiguous-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=h1.ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=load_model(o.shape[-1],ad);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
            cand=[]
            for k in range(h1.ANCHOR_CANDIDATES):
                _,w=h1.pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=h1.collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=h1.select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
            adaptive_pools={"early":[],"late":[]}
            for _,_,units in cand:
                for u in units:adaptive_pools[u["phase"]].append(u)
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
            h1.fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
            rows=[];snaps={}
            def audit(tag):
                m.eval()
                sens=h1.sensitivity(m,probe)
                P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
                wm=torch.tensor(h1.PREFS["C"],device="cuda").repeat(len(P),1)
                with torch.no_grad():tl,tf,_=tail_metrics(m,P,wm,tau)
                snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed,"arm":args.arm,"lambda_tail":lam},out/f"model_{tag}.pt")
                m.train()
            audit(0)
            for uidx in range(1,args.updates+1):
                labs,w=h1.pref_batch(uidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                _,ws=h1.pref_batch(uidx+17,torch.device("cuda"))
                units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                h1.fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac,_=tail_metrics(m,main["obs"],main["w"],tau)
                loss=ppo+lam*tail
                opt.zero_grad(set_to_none=True);loss.backward()
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                rows.append({"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                             "tail_fraction":float(tailfrac.detach().cpu()),"total_loss":float(loss.detach().cpu()),
                             "grad_norm_preclip":total,"ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"]})
                print("UPDATE",uidx,"ppo",float(ppo.detach().cpu()),"tail",float(tail.detach().cpu()),
                      "tail_frac",float(tailfrac.detach().cpu()),"term",main["termination_fraction"],flush=True)
                if uidx in SNAPS:audit(uidx)
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
            jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
            rep={"schema":"authority_isolated_coordinate_headroom_train_v1","arm":args.arm,"lambda_tail":lam,
                 "seed":args.seed,"updates":args.updates,"tau_source":str(TAU_PATH.relative_to(ROOT)),
                 "base_checkpoint":str(BASE.relative_to(ROOT)),"rows":rows,"snapshots":snaps,
                 "summary":{"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
                            "authority_gate":bool(pair_ret>=.9 and jac_ret>=.9),
                            "probe_tail_loss_ratio":snaps[str(args.updates)]["probe_tail_loss"]/(snaps["0"]["probe_tail_loss"]+1e-12),
                            "probe_tail_fraction_delta":snaps[str(args.updates)]["probe_tail_fraction"]-snaps["0"]["probe_tail_fraction"]}}
            (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print(json.dumps(rep["summary"],indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_coordinate_tail_repair_screen():
    """Run former authority_isolated_coordinate_tail_repair_screen.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts"/"rl")]
    OUT=ROOT/"runs/authority_isolated_coordinate_tail_repair-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    WIDE=ROOT/"runs/authority_isolated_critic_capacity-2026-09-25/wide_critic_fixed_policy.pt"
    BUDGET=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),"C":np.array([.25]*4,np.float32)}
    ORDER=("T","A","O","S","C");NENV=8;H=32;UPDATES=25;RHO=.25;SNAPS=(0,5,10,20,25)
    from rl.experiments.common.utilities.authority_isolated_h1_screen import collect_actor,collect_support64,fit_expanded_current_policy,pref_batch,select_diverse,sensitivity
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tail_stats(m,probe,tau):
        xs=[]
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                z=m._actor_mean_with_preference(probe,w).abs();xs.append(z)
        X=torch.cat(xs);ex=torch.relu(X-tau)
        return {"mean_excess":float(ex.mean().cpu()),"active_fraction":float((ex>0).float().mean().cpu()),
          "per_coordinate_mean_excess":ex.mean(0).cpu().tolist(),"per_coordinate_active_fraction":(ex>0).float().mean(0).cpu().tolist()}
    def robustness(env,m):
        rows=[];total_fail=0;minsur=1.
        for su,seed in enumerate((840003,840004),start=2):
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda();done=np.zeros(NENV,bool);ft=np.full(NENV,-1,int)
                fl=[]
                with torch.no_grad():
                    for t in range(64):
                        a=m.act_inference_with_preference(obs,w)
                        if su==3 and lab=="C" and t in (4,5,6,7):fl.append(float(a[0,0].cpu()))
                        nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool)
                        for i in range(NENV):
                            if dd[i] and not done[i]:ft[i]=t
                        done|=dd;obs=ot(nxt).cuda()
                surv=float(1-done.mean());minsur=min(minsur,surv);total_fail+=int(done.sum())
                rows.append({"suite":su,"preference":lab,"survival":surv,"fail_t":ft.tolist(),"flhip_t4_7":fl})
        return {"min_survival":minsur,"total_failed_lanes":total_fail,"rows":rows}
    def make_model(obs_dim,ad):
        from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
        m=AuthorityIsolatedWideCritic(obs_dim,ad).cuda();m.load_state_dict(torch.load(WIDE,map_location="cuda",weights_only=False)["model"]);m.train()
        for n,p in m.named_parameters():
            if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
        return m
    def build_support(env,m,mgr,seed):
        cand=[]
        for k in range(12):
            _,w=pref_batch(k,torch.device("cuda"));sd=seed+10000+k*137;units=collect_support64(env,m,w,mgr,sd);cand.append((k,sd,units))
        comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
        idx=select_diverse(comb,6);spec=[(cand[i][0],cand[i][1]) for i in idx]
        pools={"early":[],"late":[]}
        for _,_,units in cand:
            for u in units:pools[u["phase"]].append(u)
        return spec,pools
    def anchor_units(env,m,mgr,spec):
        out={"early":[],"late":[]}
        for k,sd in spec:
            _,w=pref_batch(k,torch.device("cuda"))
            for u in collect_support64(env,m,w,mgr,sd):out[u["phase"]].append(u)
        return out
    def grad_norm(gs):
        vals=[g.pow(2).sum() for g in gs if g is not None]
        return torch.sqrt(torch.stack(vals).sum()) if vals else torch.tensor(0.,device="cuda")
    def run_arm(env,obs_dim,ad,mgr,probe,tau,seed,name,rho,spec_ref=None):
        from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
        m=make_model(obs_dim,ad);params=[p for n,p in m.named_parameters() if p.requires_grad]
        opt=torch.optim.Adam(params,lr=1e-3)
        spec,pools=build_support(env,m,mgr,seed)
        if spec_ref is not None and spec!=spec_ref:raise RuntimeError("paired anchor spec mismatch")
        fit_expanded_current_policy(m,anchor_units(env,m,mgr,spec),pools)
        rows=[];snaps={}
        def audit(k):
            m.eval();snaps[str(k)]={"sensitivity":sensitivity(m,probe),"tail":tail_stats(m,probe,tau),"robustness":robustness(env,m)}
            torch.save({"model":m.state_dict(),"update":k,"arm":name},OUT/f"{name}_model_{k}.pt");m.train()
        audit(0)
        for uidx in range(1,UPDATES+1):
            _,w=pref_batch(uidx,torch.device("cuda"))
            main=collect_actor(env,m,w,mgr,seed+uidx*211,True)
            _,ws=pref_batch(uidx+17,torch.device("cuda"));units=collect_support64(env,m,ws,mgr,seed+200000+uidx*223)
            for u in units:
                ph=u["phase"];pools[ph].append(u)
                if len(pools[ph])>24:pools[ph].pop(0)
            fit_expanded_current_policy(m,anchor_units(env,m,mgr,spec),pools)
            with torch.no_grad():
                vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4);nv=m.value_with_preference(main["next_obs"],w)
                adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
            ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
            ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
            z=m._actor_mean_with_preference(main["obs"],main["w"]);tail=torch.relu(z.abs()-tau).pow(2).mean()
            gp=torch.autograd.grad(ppo,params,retain_graph=True,allow_unused=True);gt=torch.autograd.grad(tail,params,allow_unused=True)
            npg=grad_norm(gp);nt=grad_norm(gt);alpha=0. if rho==0 else min(1.,float((rho*npg/(nt+1e-12)).detach().cpu()))
            opt.zero_grad(set_to_none=True)
            for p,g1,g2 in zip(params,gp,gt):
                if g1 is None and g2 is None:continue
                p.grad=(torch.zeros_like(p) if g1 is None else g1)+(0 if g2 is None else alpha*g2)
            total=float(torch.nn.utils.clip_grad_norm_(params,1.).cpu());opt.step()
            with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
            rows.append({"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),"tail_alpha":alpha,
              "ppo_grad_norm":float(npg.detach().cpu()),"tail_grad_norm":float(nt.detach().cpu()),"total_grad_preclip":total,
              "termination_fraction":main["termination_fraction"]})
            if uidx in SNAPS:
                fit_expanded_current_policy(m,anchor_units(env,m,mgr,spec),pools);audit(uidx)
        return {"arm":name,"rho":rho,"anchor_specs":spec,"rows":rows,"snapshots":snaps}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=73001;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=73001);o=ot(o).cuda();mgr=env.unwrapped.reward_manager;ad=env.unwrapped.action_manager.total_action_dim
            probe=torch.tensor(np.load(PROBE)["obs"],device="cuda");tau=torch.tensor(json.loads(BUDGET.read_text())["tau"],device="cuda")
            control=run_arm(env,o.shape[-1],ad,mgr,probe,tau,73001,"control",0.)
            treat=run_arm(env,o.shape[-1],ad,mgr,probe,tau,73001,"treatment",RHO,control["anchor_specs"])
            c0=control["snapshots"]["0"]["sensitivity"];t0=treat["snapshots"]["0"]["sensitivity"]
            max0=max(abs(c0["pairwise_action_distance"]["mean"]-t0["pairwise_action_distance"]["mean"]),abs(c0["tangent_jacobian_fro_mean"]-t0["tangent_jacobian_fro_mean"]))
            cf=control["snapshots"]["25"];tf=treat["snapshots"]["25"];start=treat["snapshots"]["0"]
            pair_ret=tf["sensitivity"]["pairwise_action_distance"]["mean"]/(start["sensitivity"]["pairwise_action_distance"]["mean"]+1e-12)
            jac_ret=tf["sensitivity"]["tangent_jacobian_fro_mean"]/(start["sensitivity"]["tangent_jacobian_fro_mean"]+1e-12)
            gate={"paired_start_match":max0<1e-6,"pairwise_retention_ge_90":pair_ret>=.9,"jacobian_retention_ge_90":jac_ret>=.9,
              "min_survival_ge_95":tf["robustness"]["min_survival"]>=.95,
              "failed_lanes_improved":tf["robustness"]["total_failed_lanes"]<cf["robustness"]["total_failed_lanes"],
              "residual_lane_survives":next(x for x in tf["robustness"]["rows"] if x["suite"]==3 and x["preference"]=="C")["survival"]>=.95,
              "tail_excess_reduced":tf["tail"]["mean_excess"]<cf["tail"]["mean_excess"]}
            rep={"schema":"authority_isolated_coordinate_tail_repair_v1","tau":tau.cpu().tolist(),"control":control,"treatment":treat,
              "summary":{"gate":gate,"pass":all(gate.values()),"pairwise_retention":pair_ret,"jacobian_retention":jac_ret,
              "control_final_robustness":cf["robustness"],"treatment_final_robustness":tf["robustness"],
              "control_final_tail":cf["tail"],"treatment_final_tail":tf["tail"]}}
            (OUT/"authority_isolated_coordinate_tail_repair_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print(json.dumps(rep["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_gradient_budgeted_headroom_train():
    """Run former authority_isolated_gradient_budgeted_headroom_train.py stage."""
    
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
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["budgeted"],default="budgeted")
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--target-updates",type=int,default=10)
        ap.add_argument("--chunk-updates",type=int,default=3);args=ap.parse_args()
        lam=0.0;rho=0.10
        out=ROOT/"runs/authority_isolated_gradient_budgeted_headroom-2026-09-25"
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
                scale=rho*gpn/(gtn+1e-12)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                cos=dot/(gpn*gtn+1e-12)
                opt.zero_grad(set_to_none=True)
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a
                    bb=torch.zeros_like(p) if b is None else b
                    p.grad=aa+scale*bb
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                print("AFTER_STEP",uidx,flush=True)
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),
                     "tail_fraction":float(tailfrac.detach().cpu()),"rho":rho,
                     "ppo_grad_norm":float(gpn.cpu()),"tail_grad_norm":float(gtn.cpu()),
                     "tail_scale":float(scale.cpu()),"grad_cosine":float(cos.cpu()),
                     "scaled_tail_over_ppo":float((scale*gtn/(gpn+1e-12)).cpu()),
                     "grad_norm_preclip":total,"ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"]}
                rows.append(row);print("UPDATE",uidx,json.dumps(row),flush=True)
                if uidx==args.target_updates:audit(m,probe,tau,out,uidx,snaps,args,lam)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx)
            print("CHUNK_DONE",end,flush=True)
            if end==args.target_updates:
                s0=snaps["0"]["sensitivity"];sf=snaps[str(end)]["sensitivity"]
                pair_ret=sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+1e-12)
                jac_ret=sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+1e-12)
                rep={"schema":"authority_isolated_gradient_budgeted_headroom_v1","arm":args.arm,"rho":rho,
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

def run_authority_isolated_headroom_endpoint_audit():
    """Run former authority_isolated_headroom_endpoint_audit.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={"control":ROOT/"runs/authority_isolated_coordinate_headroom_control-2026-09-25/model_10.pt",
        "treatment":ROOT/"runs/authority_isolated_coordinate_headroom_treatment-2026-09-25/model_10.pt"}
    OUT=ROOT/"runs/authority_isolated_headroom_endpoint_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEEDS=(840003,840004);NENV=8;H=64
    TAU=torch.tensor(json.load(open(ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"))["tau"])
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(env):
        tm=env.unwrapped.termination_manager;out={}
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except:pass
        return out
    def rollout(env,m,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        done=np.zeros(NENV,bool);ft=np.full(NENV,-1,int);why=[[] for _ in range(NENV)]
        tf=[];fl=[]
        for t in range(H):
            with torch.no_grad():
                z=m._actor_mean_with_preference(obs,w);a=torch.tanh(z)
            tf.append(float((z.abs()>TAU.to("cuda")).float().mean().cpu()))
            if seed==840004 and lab=="C" and t in (4,5,6,7):fl.append(float(a[0,0].cpu()))
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool);rr=terms(env)
            for i in range(NENV):
                if dd[i] and not done[i]:ft[i]=t;why[i]=[n for n,v in rr.items() if v[i]]
            done|=dd;obs=ot(nxt).cuda()
        return {"suite":2 if seed==840003 else 3,"preference":lab,"survival":float(1-done.mean()),
                "fail_count":int(done.sum()),"fail_t":ft.tolist(),"reason":why,
                "tail_fraction_mean":float(np.mean(tf)),"flhip_t4_7":fl}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            rep={}
            for arm,path in CK.items():
                m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval()
                rows=[]
                for seed in SEEDS:
                    for lab in PREFS:
                        q=rollout(env,m,lab,seed);rows.append(q);print(arm,q,flush=True)
                rep[arm]={"rows":rows,"min_survival":min(x["survival"] for x in rows),
                          "failed_lanes":sum(x["fail_count"] for x in rows),
                          "mean_tail_fraction":float(np.mean([x["tail_fraction_mean"] for x in rows])),
                          "residual_flhip_t4_7":[x["flhip_t4_7"] for x in rows if x["suite"]==3 and x["preference"]=="C"][0]}
            (OUT/"endpoint_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("SUMMARY",json.dumps(rep,indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_residual_coordinate_rescue():
    """Run former authority_isolated_residual_coordinate_rescue.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_residual_coordinate_rescue-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CKPT=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt";SEED=840004;NENV=8;LANE=0;H=24
    W=[.25]*4
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tf(z,c):return float(c)*torch.tanh(z/float(c))
    def reasons(env):
        out={};tm=env.unwrapped.termination_manager
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except Exception:pass
        return out
    def rollout(env,m,indices,label):
        w=torch.tensor(W,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=SEED);obs=ot(obs).cuda()
        ff=None;trace=[]
        for t in range(H):
            with torch.no_grad():
                z=m._actor_mean_with_preference(obs,w);a20=torch.tanh(tf(z,2.0));a25=torch.tanh(tf(z,2.5));a=a20.clone()
                if indices:a[:,indices]=a25[:,indices]
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool);rr=reasons(env)
            trace.append({"t":t,"a20":a20[LANE].cpu().tolist(),"a25":a25[LANE].cpu().tolist(),"a":a[LANE].cpu().tolist(),
              "done":bool(dd[LANE]),"terms":[n for n,v in rr.items() if v[LANE]]})
            if dd[LANE] and ff is None:ff=t
            obs=ot(nxt).cuda()
        return {"label":label,"indices":indices,"first_fail":ff,"survived":ff is None,"trace":trace}
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            names=list(env.unwrapped.scene["robot"].data.joint_names)
            tests=[("none",[]),("all",list(range(12)))]
            tests += [(f"single:{names[j]}",[j]) for j in range(12)]
            tests += [("hips",[0,1,2,3]),("thighs",[4,5,6,7]),("calves",[8,9,10,11]),
                      ("FL",[0,4,8]),("FR",[1,5,9]),("RL",[2,6,10]),("RR",[3,7,11]),
                      ("front",[0,1,4,5,8,9]),("rear",[2,3,6,7,10,11])]
            rows=[]
            for label,idx in tests:
                r=rollout(env,m,idx,label);rows.append(r);print(label,"SURV",r["survived"],"FAIL",r["first_fail"],flush=True)
            rep={"schema":"authority_isolated_residual_coordinate_rescue_v1","base":"c2.0","donor":"c2.5",
              "suite":3,"preference":"C","lane":0,"joint_names":names,"rows":rows}
            (OUT/"authority_isolated_residual_coordinate_rescue_report.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "authority_isolated_coordinate_headroom_train": run_authority_isolated_coordinate_headroom_train,
    "authority_isolated_coordinate_headroom_train_chunked": run_authority_isolated_coordinate_headroom_train_chunked,
    "authority_isolated_coordinate_headroom_train_contiguous": run_authority_isolated_coordinate_headroom_train_contiguous,
    "authority_isolated_coordinate_tail_repair_screen": run_authority_isolated_coordinate_tail_repair_screen,
    "authority_isolated_gradient_budgeted_headroom_train": run_authority_isolated_gradient_budgeted_headroom_train,
    "authority_isolated_headroom_endpoint_audit": run_authority_isolated_headroom_endpoint_audit,
    "authority_isolated_residual_coordinate_rescue": run_authority_isolated_residual_coordinate_rescue,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
