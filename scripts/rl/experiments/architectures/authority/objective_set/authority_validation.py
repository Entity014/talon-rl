"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_objective_set_g1_anchor_screen():
    """Run former objective_set_g1_anchor_screen.py stage."""
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
    RUNS=[(f,s) for f in ("G1-2","G1-3") for s in (73101,73102,73103)]
    OUT=ROOT/"runs/objective_set_g1_anchor_screen-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    
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
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"];rows=[]
            for fold,seed in RUNS:
                ck=ROOT/f"runs/objective_set_{fold.lower().replace('-','_')}_seed{seed}-2026-09-25/model_30.pt"
                m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
                rr=[]
                for suite in range(4):
                    rs=840001+suite
                    for lab in ("T","A","O","S","C"):
                        q=h2.evaluate(env,m,mgr,robot,h2.PREFS[lab],rs);q.update({"suite":suite,"label":lab});rr.append(q)
                ep={};cev=[];cb=[]
                for lab in h2.ORDER:
                    j=h2.IDX[lab];pk=h2.PHYS[lab];oo=[];pp=[];sv=[]
                    for suite in range(4):
                        r=next(x for x in rr if x["suite"]==suite and x["label"]==lab);c=next(x for x in rr if x["suite"]==suite and x["label"]=="C")
                        oo.append(r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]>0)
                        pp.append(r["physical"][pk]-c["physical"][pk]<0);sv.append(r["survival"])
                    ep[lab]={"objective":float(np.mean(oo)),"physical":float(np.mean(pp)),"min_survival":float(min(sv)),
                             "pass":bool(np.mean(oo)>=.75 and np.mean(pp)>=.75 and min(sv)>=.95)}
                for x in rr:cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"];cb+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
                critic={"ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),"mean_abs_bias":float(np.mean(np.abs(cb)))}
                rec={"fold":fold,"seed":seed,"endpoint":ep,"critic":critic,"TAO_pass":all(ep[x]["pass"] for x in ("T","A","O")),
                     "critic_pass":critic["ev_mean"]>0 and critic["negative_fraction"]<=.25}
                rows.append(rec);print(fold,seed,json.dumps(rec),flush=True)
            rep={"schema":"objective_set_g1_anchor_screen_v1","rows":rows,
                 "summary":{"TAO_pass_runs":sum(x["TAO_pass"] for x in rows),"critic_pass_runs":sum(x["critic_pass"] for x in rows),
                            "all_runs":len(rows)}}
            (OUT/"anchor_screen.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_objective_set_g1_endpoint_screen_batch():
    """Run former objective_set_g1_endpoint_screen_batch.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.objective_set_g1_endpoint_screen as s
    from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic
    
    SPECS=(("G1-2",73102),("G1-2",73103),("G1-3",73101),("G1-3",73102),("G1-3",73103))
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app
        sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=s.NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=s.ot(o).cuda();mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            for fold,seed in SPECS:
                run=ROOT/f"runs/objective_set_{fold.lower().replace('-','_')}_seed{seed}-2026-09-25"
                tr=json.load(open(run/"training_report.json"))
                if not tr["leakage_free"]:raise RuntimeError(f"leakage {fold} {seed}")
                m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],12).cuda()
                m.load_state_dict(torch.load(run/"model_30.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
                hold=s.FOLDS[fold]["holdout"];sets=list(__import__("itertools").combinations(range(4),hold));results={}
                for ids in sets:
                    key="".join(s.ORDER[i] for i in ids);results[key]=s.eval_set(env,m,mgr,robot,ids)
                    print("SET",fold,seed,key,json.dumps({"req":results[key]["required_TAO_pass"],"ep":results[key]["endpoint"],"critic":results[key]["critic"]}),flush=True)
                anchor=s.eval_set(env,m,mgr,robot,(0,1,2,3))
                criteria={"leakage_free":True,
                  "heldout_TAO_endpoints":all(x["required_TAO_pass"] for x in results.values()),
                  "heldout_critic":all(x["critic"]["pass"] for x in results.values()),
                  "heldout_permutation":all(x["permutation"]["pass"] for x in results.values()),
                  "heldout_survival":all(x["min_survival"]>=.95 for x in results.values()),
                  "m4_anchor_TAO":anchor["required_TAO_pass"],"m4_anchor_critic":anchor["critic"]["pass"],"m4_anchor_survival":anchor["min_survival"]>=.95}
                rep={"schema":"objective_set_g1_endpoint_screen_v1","fold":fold,"seed":seed,"heldout_cardinality":hold,
                  "heldout_sets":results,"m4_anchor":anchor,"criteria":criteria,"endpoint_screen_pass":bool(all(criteria.values()))}
                out=run/"endpoint_screen";out.mkdir(exist_ok=True)
                (out/"endpoint_screen.json").write_text(json.dumps(rep,indent=2)+"\n")
                print("SCREEN",fold,seed,json.dumps(criteria),flush=True)
                del m;torch.cuda.empty_cache()
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_objective_set_g1_ta_seed3_confirm():
    """Run former objective_set_g1_ta_seed3_confirm.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.objective_set_g1_endpoint_screen as s
    from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic
    
    def main():
        run=ROOT/"runs/objective_set_g1_2_seed73103-2026-09-25"
        out=run/"endpoint_screen";out.mkdir(exist_ok=True)
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=s.NENV;cfg.seed=73103
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=73103);o=s.ot(o).cuda()
            m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],12).cuda()
            m.load_state_dict(torch.load(run/"model_30.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
            q=s.eval_set(env,m,env.unwrapped.reward_manager,env.unwrapped.scene["robot"],(0,1))
            rep={"schema":"objective_set_g1_ta_seed3_confirmation_v1","fold":"G1-2","seed":73103,"set":"TA","result":q,
                 "TA_required_pass":q["required_TAO_pass"]}
            (out/"ta_seed3_confirmation.json").write_text(json.dumps(rep,indent=2)+"\n")
            print(json.dumps(rep,indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_objective_set_g1_temporal_anchor_audit():
    """Run former objective_set_g1_temporal_anchor_audit.py stage."""
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
    RUNS=[(f,s) for f in ("G1-2","G1-3") for s in (73101,73102,73103)]
    SNAPS=(0,10,20,30)
    OUT=ROOT/"runs/objective_set_g1_temporal_anchor_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def eval_anchor(env,mgr,robot,m):
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
        cev=[];cb=[]
        for x in rows:cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"];cb+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
        critic={"ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),"mean_abs_bias":float(np.mean(np.abs(cb)))}
        return {"endpoint":ep,"TAO_pass":all(ep[x]["pass"] for x in ("T","A","O")),
                "critic":critic,"critic_pass":critic["ev_mean"]>0 and critic["negative_fraction"]<=.25}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=h2.NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"];rows=[]
            for fold,seed in RUNS:
                tag=fold.lower().replace("-","_");run=ROOT/f"runs/objective_set_{tag}_seed{seed}-2026-09-25"
                for u in SNAPS:
                    ck=run/f"model_{u}.pt";m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
                    q=eval_anchor(env,mgr,robot,m);q.update({"fold":fold,"seed":seed,"update":u});rows.append(q)
                    print(fold,seed,u,"TAO",q["TAO_pass"],"EV",round(q["critic"]["ev_mean"],4),"neg",round(q["critic"]["negative_fraction"],3),
                          "ep",{k:(v["objective"],v["physical"]) for k,v in q["endpoint"].items()},flush=True)
            rep={"schema":"objective_set_g1_temporal_anchor_audit_v1","rows":rows}
            (OUT/"temporal_anchor_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "objective_set_g1_anchor_screen": run_objective_set_g1_anchor_screen,
    "objective_set_g1_endpoint_screen_batch": run_objective_set_g1_endpoint_screen_batch,
    "objective_set_g1_ta_seed3_confirm": run_objective_set_g1_ta_seed3_confirm,
    "objective_set_g1_temporal_anchor_audit": run_objective_set_g1_temporal_anchor_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
