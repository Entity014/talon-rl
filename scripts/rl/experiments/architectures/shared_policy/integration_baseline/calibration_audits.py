"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_v1a_e05_calibrate():
    """Run former v1a_e05_calibrate.py stage."""
    """M0.1-only variability calibration for V1A-E0.5.
    
    This runner intentionally accepts no V1-A checkpoint argument and never
    constructs the V1-A wrapper.  It produces descriptive suite-level metrics;
    margin selection and verdict logic are explicitly out of scope.
    """
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    
    import argparse
    import json
    import sys
    import time
    from pathlib import Path
    
    import numpy as np
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--suite-count", type=int, default=8)
        parser.add_argument("--num-envs", type=int, default=64)
        parser.add_argument("--steps", type=int, default=500)
        parser.add_argument("--reset-seed-base", type=int, default=27001)
        args = parser.parse_args()
        if args.suite_count < 1 or args.num_envs < 1 or args.steps < 1:
            raise ValueError("suite-count, num-envs, and steps must be positive")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lifecycle = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
    
        def mark(event: str, extra=None, **kwargs):
            row = {"event": event, "unix": time.time()}
            if extra:
                row.update(extra)
            row.update(kwargs)
            with lifecycle.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    
        mark("RUN_STARTED", suite_count=args.suite_count, num_envs=args.num_envs)
        from isaaclab.app import AppLauncher
    
        saved_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved_argv
        mark("APP_INIT_OK")
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.experiments.common.utilities.v1a_e0_eval import load_policy, run_one
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = args.reset_seed_base
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            mark("ENV_CREATED")
            source_paths = {
                0: ROOT / "runs/m0_1_seed0_2026-09-22/model_299.pt",
                1: ROOT / "runs/m0_1_seed1_2026-09-22-rerun/model_299.pt",
                2: ROOT / "runs/m0_1_seed2_2026-09-22/model_299.pt",
            }
            models = {seed: load_policy(path, "m01") for seed, path in source_paths.items()}
            mark("M01_CHECKPOINTS_LOADED", seeds=[0, 1, 2])
            rows = []
            for suite in range(args.suite_count):
                suite_seed = args.reset_seed_base + suite * 101
                for seed in (0, 1, 2):
                    reset_seed = suite_seed + seed
                    metrics = run_one(env, models[seed], "m01", reset_seed, args.steps, mark)
                    rows.append({"suite": suite, "seed": seed, "reset_seed": reset_seed, "metrics": metrics})
                    mark("SUITE_SEED_DONE", suite=suite, seed=seed)
            result = {
                "schema": "v1a_e05_baseline_variability_v1",
                "protocol": "V1A-E0.5",
                "status": "COMPLETE_DESCRIPTIVE_ONLY",
                "suite_count": args.suite_count,
                "num_envs": args.num_envs,
                "steps": args.steps,
                "reset_seed_base": args.reset_seed_base,
                "subjects": "M0.1 terminal checkpoints only",
                "v1a_checkpoint_accessed": False,
                "margin_selected": False,
                "binary_verdict": "FORBIDDEN",
                "rows": rows,
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            mark("ARTIFACT_WRITTEN", path=str(args.output))
            mark("RUN_DONE", status="RUN_DONE")
            print(json.dumps(result, indent=2))
        except BaseException as exc:
            error = args.output.with_name(args.output.stem + ".ERROR.json")
            error.write_text(json.dumps({"schema": "v1a_e05_error_v1", "status": "ERROR", "error": str(exc)}, indent=2) + "\n")
            mark("ERROR", error=str(exc))
            raise
        finally:
            if env is not None:
                env.close()
            app.close()
    
    
    if True:
        main()

def run_v1a_e06_audit():
    """Run former v1a_e06_audit.py stage."""
    """Baseline-only audit of stock rsl_rl deployment action semantics."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    
    import argparse
    import json
    import sys
    import time
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    
    def main() -> None:
        p = argparse.ArgumentParser()
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--num-envs", type=int, default=64)
        p.add_argument("--steps", type=int, default=500)
        p.add_argument("--reset-seed-base", type=int, default=37001)
        args = p.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lifecycle = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
    
        def mark(event, **extra):
            row = {"event": event, "unix": time.time()}
            row.update(extra)
            with lifecycle.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    
        mark("RUN_STARTED")
        from isaaclab.app import AppLauncher
    
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        mark("APP_INIT_OK")
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.experiments.common.utilities.v1a_e0_eval import load_policy, obs_tensor
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = args.reset_seed_base
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            mark("ENV_CREATED")
            paths = {
                0: ROOT / "runs/m0_1_seed0_2026-09-22/model_299.pt",
                1: ROOT / "runs/m0_1_seed1_2026-09-22-rerun/model_299.pt",
                2: ROOT / "runs/m0_1_seed2_2026-09-22/model_299.pt",
            }
            models = {seed: load_policy(path, "m01") for seed, path in paths.items()}
            mark("M01_CHECKPOINTS_LOADED")
            rows = []
            for seed, model in models.items():
                reset_seed = args.reset_seed_base + seed
                obs, _ = env.reset(seed=reset_seed)
                obs = obs_tensor(obs).cuda()
                before = {k: v.detach().clone() for k, v in model.state_dict().items()}
                clip_count = 0
                action_count = 0
                raw_abs = []
                clipped_abs = []
                finite = True
                velocity_error = []
                done_any = np.zeros(args.num_envs, dtype=bool)
                with torch.no_grad():
                    for step in range(args.steps):
                        raw = model.act_inference({"policy": obs, "critic": obs})
                        action = torch.clamp(raw, -1.0, 1.0)
                        clip_count += int(((raw < -1.0) | (raw > 1.0)).sum())
                        action_count += int(raw.numel())
                        raw_abs.append(raw.abs().detach().cpu().numpy())
                        clipped_abs.append(action.abs().detach().cpu().numpy())
                        nxt, _, term, trunc, _ = env.step(action)
                        done_any |= (term | trunc).detach().cpu().numpy()
                        data = env.unwrapped.scene["robot"].data
                        command = env.unwrapped.command_manager.get_command("base_velocity")
                        velocity_error.append((data.root_lin_vel_b[:, 0] - command[:, 0]).abs().cpu().numpy())
                        obs = obs_tensor(nxt).cuda()
                        finite &= bool(torch.isfinite(raw).all() and torch.isfinite(action).all() and torch.isfinite(obs).all())
                        if step == 0:
                            mark("STEP_1_OK", seed=seed)
                after = model.state_dict()
                nonmutation = all(torch.equal(before[k], after[k]) for k in before)
                rows.append({
                    "seed": seed,
                    "reset_seed": reset_seed,
                    "checkpoint": str(paths[seed]),
                    "stock_clip_actions": 1.0,
                    "raw_action_abs_p99": float(np.percentile(np.concatenate(raw_abs), 99)),
                    "clipped_action_abs_p99": float(np.percentile(np.concatenate(clipped_abs), 99)),
                    "raw_action_max": float(np.max(np.concatenate(raw_abs))),
                    "clip_fraction": float(clip_count / action_count),
                    "survival": float(1.0 - done_any.mean()),
                    "velocity_tracking_error": float(np.mean(np.concatenate(velocity_error))),
                    "finite_state": bool(finite),
                    "evaluator_non_mutation": bool(nonmutation),
                })
                mark("SEED_DONE", seed=seed)
            result = {
                "schema": "v1a_e06_audit_v1",
                "protocol": "V1A-E0.6",
                "status": "COMPLETE_AUDIT_ONLY",
                "subjects": "M0.1 terminal checkpoints only",
                "v1a_checkpoint_accessed": False,
                "reference_path": "raw rsl_rl actor mean followed by stock clip_actions=1.0",
                "rows": rows,
                "binary_verdict": "FORBIDDEN",
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            mark("ARTIFACT_WRITTEN")
            mark("RUN_DONE", status="RUN_DONE")
            print(json.dumps(result, indent=2))
        except BaseException as exc:
            error = args.output.with_name(args.output.stem + ".ERROR.json")
            error.write_text(json.dumps({"schema": "v1a_e06_error_v1", "status": "ERROR", "error": str(exc)}, indent=2) + "\n")
            mark("ERROR", error=str(exc))
            raise
        finally:
            if env is not None:
                env.close()
            app.close()
    
    
    if True:
        main()

def run_v1a_e1r_calibrate():
    """Run former v1a_e1r_calibrate.py stage."""
    """M0.1-only corrected-path calibration; never loads V1-A."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    import argparse,json,sys,time
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    def main():
     p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--suite-count',type=int,default=8);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--steps',type=int,default=500);p.add_argument('--reset-seed-base',type=int,default=57001);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);life=a.output.with_name(a.output.stem+'.lifecycle.jsonl')
     def mark(event,**kw):
      with life.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**kw},sort_keys=True)+'\n')
     mark('RUN_STARTED');from isaaclab.app import AppLauncher;saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK');env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from rl.experiments.common.utilities.v1a_e0_eval import load_policy,obs_tensor,snapshot
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.reset_seed_base;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED')
      paths={0:ROOT/'runs/m0_1_seed0_2026-09-22/model_299.pt',1:ROOT/'runs/m0_1_seed1_2026-09-22-rerun/model_299.pt',2:ROOT/'runs/m0_1_seed2_2026-09-22/model_299.pt'};models={s:load_policy(p,'m01') for s,p in paths.items()};mark('M01_CHECKPOINTS_LOADED')
      rows=[]
      for suite in range(a.suite_count):
       for seed,model in models.items():
        rs=a.reset_seed_base+suite*101+seed;obs,_=env.reset(seed=rs);obs=obs_tensor(obs).cuda();before=snapshot(model);done_any=np.zeros(a.num_envs,bool);vx=[];tilt=[];terms=[];contacts=[];finite=True
        with torch.no_grad():
         for step in range(a.steps):
          raw=model.act_inference({'policy':obs,'critic':obs});act=torch.clamp(raw,-1.,1.);nxt,_,term,trunc,_=env.step(act);done_any|=(term|trunc).cpu().numpy();terms.append(term.cpu().numpy().astype(np.float32));d=env.unwrapped.scene['robot'].data;c=env.unwrapped.command_manager.get_command('base_velocity');vx.append((d.root_lin_vel_b[:,0]-c[:,0]).abs().cpu().numpy());q=d.root_quat_w;roll=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2));pitch=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1));tilt.append(torch.rad2deg(torch.maximum(roll.abs(),pitch.abs())).cpu().numpy());contacts.append(env.unwrapped.termination_manager.get_term('base_contact').cpu().numpy().astype(np.float32));obs=obs_tensor(nxt).cuda();finite&=bool(torch.isfinite(raw).all() and torch.isfinite(obs).all());
        after=model.state_dict();t=np.concatenate(tilt);rows.append({'suite':suite,'seed':seed,'reset_seed':rs,'metrics':{'survival':float(1-done_any.mean()),'velocity_tracking_error':float(np.mean(np.concatenate(vx))),'tilt_p95_deg':float(np.percentile(t,95)),'max_tilt_deg':float(t.max()),'termination_rate':float(np.mean(np.concatenate(terms))),'base_contact_rate':float(np.mean(np.concatenate(contacts))),'finite_state':finite,'evaluator_non_mutation':all(torch.equal(before[k],after[k]) for k in before)}});mark('SUITE_SEED_DONE',suite=suite,seed=seed)
      out={'schema':'v1a_e1r_calibration_v1','protocol':'V1A-E1R-CAL','status':'COMPLETE_DESCRIPTIVE_ONLY','subjects':'M0.1 terminal checkpoints only','stock_clip_actions':1.0,'v1a_checkpoint_accessed':False,'margin_selected':False,'binary_verdict':'FORBIDDEN','rows':rows};a.output.write_text(json.dumps(out,indent=2)+'\n');mark('ARTIFACT_WRITTEN');mark('RUN_DONE',status='RUN_DONE');print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

STAGES = {
    "v1a_e05_calibrate": run_v1a_e05_calibrate,
    "v1a_e06_audit": run_v1a_e06_audit,
    "v1a_e1r_calibrate": run_v1a_e1r_calibrate,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
