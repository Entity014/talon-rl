"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_authorize_l0b():
    """Run former authorize_l0b.py stage."""
    """Authorize L0-B only after the real-Isaac lifecycle smoke passes."""
    import hashlib, json
    from pathlib import Path
    import argparse
    
    ROOT = Path(__file__).resolve().parents[4]
    MANIFEST = ROOT / "artifacts/l0b_freeze/L0B_FREEZE.json"
    
    def digest(p):
        h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
    
    def main():
        ap = argparse.ArgumentParser(); ap.add_argument('--smoke-dir', type=Path, required=True); run = ap.parse_args().smoke_dir.resolve()
        done, metrics, ckpt = run / "RUN_DONE.json", run / "training_metrics.json", run / "checkpoint.pt"
        if not all(p.exists() for p in (done, metrics, ckpt)):
            raise RuntimeError("L0-B smoke artifacts incomplete")
        d = json.loads(done.read_text()); rows = json.loads(metrics.read_text())["metrics"]
        if d.get("status") != "RUN_DONE" or d.get("exit_code") != 0 or len(rows) != 2:
            raise RuntimeError("L0-B smoke lifecycle failed")
        for row in rows:
            if row["epochs_completed"] != 5 or row["minibatches"] != 4 or not all(float(row[k]) == float(row[k]) for k in ("learned_std", "analytic_kl", "learning_rate")):
                raise RuntimeError("L0-B smoke reference PPO semantics incomplete")
        mf = json.loads(MANIFEST.read_text()); mf["training_authorized"] = True
        mf["authorization"] = {"smoke_run": str(run.relative_to(ROOT)), "smoke_checkpoint_sha256": digest(ckpt), "smoke_metrics_sha256": digest(metrics), "checks": ["RUN_STARTED", "RUN_DONE", "learned_std", "5_epochs", "4_minibatches", "adaptive_KL", "checkpoint"]}
        MANIFEST.write_text(json.dumps(mf, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": mf["status"], "training_authorized": True, "smoke": "PASS"}, indent=2))
    
    if True: main()

def run_freeze_l0b():
    """Run former freeze_l0b.py stage."""
    """Create the L0-B formulation manifest; authorization remains off."""
    import hashlib, json
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    OUT = ROOT / "artifacts/l0b_freeze/L0B_FREEZE.json"
    
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    def main():
        files = {
            "gate0b_runner": ROOT / "scripts/rl/experiments/baselines/scalar_substrate/substrate_gate.py",
            "gate0b_evaluator": ROOT / "scripts/rl/experiments/baselines/scalar_substrate/substrate_gate.py",
            "gate0b_reward": ROOT / "talon_rl/rewards/baselines.py",
            "gate0b_wrapper": ROOT / "talon_rl/wrappers/scalar_reward_env.py",
            "gate0b_env_cfg": ROOT / "talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py",
            "gate0b_env": ROOT / "talon_rl/tasks/locomotion/a1_env/a1_env.py",
            "reference_trainer": ROOT / "scripts/rl/core/algorithms/l0_reference_ppo.py",
            "production_smoke": ROOT / "scripts/rl/experiments/baselines/reference_policy/reference_training.py",
            "full_runner": ROOT / "scripts/rl/experiments/baselines/reference_policy/reference_training.py",
            "actor_critic": ROOT / "scripts/rl/core/modules/actor_critic.py",
            "reset_states": ROOT / "artifacts/gate0b_reset_states.npz",
        }
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("missing L0-B freeze input: " + ", ".join(missing))
        manifest = {
            "schema": "l0b_freeze_v1",
            "status": "FROZEN",
            "training_authorized": False,
            "primary_experiment": "L0-B",
            "question": "reference-style PPO recipe on unchanged Gate-0B task",
            "seeds": [0, 1, 2], "num_envs": 4096, "updates": 500,
            "commands": [[0.25, 0.0, 0.0], [0.50, 0.0, 0.0], [0.75, 0.0, 0.0]],
            "command_schedule": "update index modulo 3; command held over each 24-step rollout",
            "rollout_steps": 24,
            "actor_hidden_dims": [128, 128, 128],
            "critic_hidden_dims": [128, 128, 128],
            "activation": "elu",
            "learned_std": {"initial": 1.0, "trainable": True},
            "ppo": {"learning_rate": 1e-3, "adaptive_kl": True, "desired_kl": 0.01,
                    "clip_eps": 0.2, "value_loss_coef": 1.0, "entropy_coef": 0.01,
                    "gamma": 0.99, "gae_lambda": 0.95, "epochs": 5,
                    "minibatches": 4, "max_grad_norm": 1.0, "lr_factor": 1.5,
                    "lr_bounds": [1e-5, 1e-2]},
            "task_contract": "Gate-0B env/reward/reset/command/evaluator unchanged",
            "distribution_note": "Uses repository ActorCritic tanh-squashed action/logp path; not byte-identical rsl_rl distribution.",
            "checkpoint_policy": "update_25 multiples and final update_500; optimizer, learned log_std, LR, update index",
            "sha256": {name: digest(path) for name, path in files.items()},
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"path": str(OUT), "status": manifest["status"], "training_authorized": False}, indent=2))
    
    if True: main()

def run_l0b_production_smoke():
    """Run former l0b_production_smoke.py stage."""
    """Small real-Isaac L0-B lifecycle smoke; never authorizes full training."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    import argparse, json, time, traceback
    from pathlib import Path
    import sys
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    MANIFEST = ROOT / "artifacts/l0b_freeze/L0B_FREEZE.json"
    
    def write(p, x): p.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")
    
    def main():
        ap = argparse.ArgumentParser(); ap.add_argument("--num-envs", type=int, default=4); ap.add_argument("--updates", type=int, default=2); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--run-dir", type=Path, required=True)
        a = ap.parse_args(); mf = json.loads(MANIFEST.read_text())
        if mf["schema"] != "l0b_freeze_v1" or mf["training_authorized"]:
            raise RuntimeError("unexpected L0-B manifest state")
        run = a.run_dir.resolve()
        if run.exists(): raise FileExistsError(run)
        run.mkdir(parents=True); write(run / "RUN_STARTED.json", {"status": "RUN_STARTED", "seed": a.seed, "unix": time.time()})
        app = base = None
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym
            import talon_rl.tasks.locomotion.a1_env
            from rl.experiments.common.utilities.train_b0 import nominalize
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.scalar_ppo import ScalarRolloutBuffer, scalar_gae
            from rl.core.algorithms.scalar_ppo import L0ReferencePPO, L0ReferenceConfig
            torch.manual_seed(a.seed); np.random.seed(a.seed)
            cfg = IsaacLabTalonEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = a.seed; cfg.sim.dt = .01; cfg.decimation = 1; nominalize(cfg)
            base = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped; env = B0TalonEnv(base)
            model = ActorCritic(base.obs_dim, base.obs_dim, base.action_dim, 1, [128, 128, 128]).to(base.device)
            trainer = L0ReferencePPO(model, L0ReferenceConfig()); trans = env.reset(); obs = trans["obs"]; rows=[]
            for update in range(a.updates):
                env.command = tuple(mf["commands"][update % 3]); env._command(); std = trainer.begin_update(); buf = ScalarRolloutBuffer(24, a.num_envs)
                old_means=[]
                for _ in range(24):
                    x = torch.as_tensor(obs, device=base.device, dtype=torch.float32)
                    with torch.no_grad():
                        old_means.append(model.raw_mean(x).cpu()); act, lp = model.act(x); val = model.value(x).squeeze(-1)
                    trans = env.step(act.cpu().numpy().astype(np.float32)); buf.append(obs, act.cpu().numpy(), lp.cpu().numpy(), trans["reward"], trans["done"], val.cpu().numpy()); obs = trans["obs"]
                with torch.no_grad(): fv = model.value(torch.as_tensor(obs, device=base.device, dtype=torch.float32)).squeeze(-1).cpu().numpy()
                buf.finish(fv); ar = buf.arrays(); adv = scalar_gae(ar["rewards"], np.r_[ar["values"], ar["final_value"][None]], ar["dones"], .99, .95); ret = adv + ar["values"]; fl = buf.flatten()
                x = torch.as_tensor(fl["obs"], device=base.device, dtype=torch.float32); ac = torch.as_tensor(fl["actions"], device=base.device, dtype=torch.float32); old = torch.as_tensor(fl["logp_old"], device=base.device); am = torch.cat(old_means, 0).to(base.device)
                st = trainer.update(x, ac, old, torch.as_tensor(adv.reshape(-1), device=base.device), torch.as_tensor(ret.reshape(-1), device=base.device), am, seed=a.seed); st.update({"sampled_std": std, "command": mf["commands"][update % 3], "finite": all(np.isfinite(v).all() if isinstance(v, np.ndarray) else np.isfinite(v) for v in [ar["rewards"], ar["values"]])}); rows.append(st)
            trainer.save(run / "checkpoint.pt"); write(run / "training_metrics.json", {"metrics": rows}); write(run / "RUN_DONE.json", {"status": "RUN_DONE", "exit_code": 0, "final_update": trainer.update_idx, "actor_lr": trainer.optim.param_groups[0]["lr"]})
        except BaseException as exc:
            write(run / "ERROR.json", {"status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()}); raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    
    if True: main()

def run_train_l0b():
    """Run former train_l0b.py stage."""
    """Authorized L0-B full runner: reference PPO on the unchanged Gate-0B task."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    import argparse, hashlib, json, time, traceback, sys
    from pathlib import Path
    import numpy as np
    import torch
    ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
    MANIFEST=ROOT/'artifacts/l0b_freeze/L0B_FREEZE.json'; STEPS=24; HIDDEN=[128,128,128]
    def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
    def main():
        p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--num-envs',type=int,default=4096); p.add_argument('--updates',type=int,default=500); p.add_argument('--run-dir',type=Path,required=True); a=p.parse_args()
        mf=json.loads(MANIFEST.read_text())
        if mf.get('schema')!='l0b_freeze_v1' or not mf.get('training_authorized'): raise RuntimeError('L0-B is not authorized')
        if a.seed not in mf['seeds'] or a.num_envs!=mf['num_envs'] or a.updates!=mf['updates']: raise RuntimeError('L0-B CLI differs from frozen manifest')
        if mf['rollout_steps']!=STEPS or mf['actor_hidden_dims']!=HIDDEN: raise RuntimeError('L0-B architecture/rollout mismatch')
        run=a.run_dir.resolve()
        if run.exists(): raise FileExistsError(run)
        for q in (run,run/'checkpoints'): q.mkdir(parents=True)
        write(run/'config.json',{'schema':'l0b_runner_v1','seed':a.seed,'num_envs':a.num_envs,'updates':a.updates,'commands':mf['commands'],'rollout_steps':STEPS,'hidden_dims':HIDDEN,'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}); write(run/'RUN_STARTED.json',{'status':'RUN_STARTED','seed':a.seed,'unix':time.time()})
        app=base=None
        try:
            from isaaclab.app import AppLauncher
            app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
            from rl.experiments.common.utilities.train_b0 import nominalize
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.scalar_ppo import ScalarRolloutBuffer, scalar_gae
            from rl.core.algorithms.scalar_ppo import L0ReferencePPO, L0ReferenceConfig
            torch.manual_seed(a.seed); np.random.seed(a.seed)
            cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=a.num_envs; cfg.seed=a.seed; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg)
            base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base); model=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,HIDDEN).to(base.device); trainer=L0ReferencePPO(model,L0ReferenceConfig()); trans=env.reset(); obs=trans['obs']; metrics=[]
            for update in range(a.updates):
                env.command=tuple(mf['commands'][update%3]); env._command(); sampled_std=trainer.begin_update(); buf=ScalarRolloutBuffer(STEPS,a.num_envs); means=[]
                for _ in range(STEPS):
                    x=torch.as_tensor(obs,device=base.device,dtype=torch.float32)
                    with torch.no_grad(): means.append(model.raw_mean(x).cpu()); act,lp=model.act(x); val=model.value(x).squeeze(-1)
                    tr=env.step(act.cpu().numpy().astype(np.float32)); buf.append(obs,act.cpu().numpy(),lp.cpu().numpy(),tr['reward'],tr['done'],val.cpu().numpy()); obs=tr['obs']
                with torch.no_grad(): fv=model.value(torch.as_tensor(obs,device=base.device,dtype=torch.float32)).squeeze(-1).cpu().numpy()
                buf.finish(fv); ar=buf.arrays(); adv=scalar_gae(ar['rewards'],np.r_[ar['values'],ar['final_value'][None]],ar['dones'],.99,.95); ret=adv+ar['values']; fl=buf.flatten(); xo=torch.as_tensor(fl['obs'],device=base.device,dtype=torch.float32); ac=torch.as_tensor(fl['actions'],device=base.device,dtype=torch.float32); old=torch.as_tensor(fl['logp_old'],device=base.device); old_mean=torch.cat(means,0).to(base.device)
                st=trainer.update(xo,ac,old,torch.as_tensor(adv.reshape(-1),device=base.device),torch.as_tensor(ret.reshape(-1),device=base.device),old_mean,seed=a.seed); st.update({'sampled_std':sampled_std,'command':mf['commands'][update%3],'reward_mean':float(ar['rewards'].mean())}); metrics.append(st)
                if (update+1)%25==0 or update+1==a.updates: trainer.save(run/'checkpoints'/f'update_{update+1:03d}.pt')
            trainer.save(run/'checkpoints/final.pt'); write(run/'training_metrics.json',{'metrics':metrics}); write(run/'RUN_DONE.json',{'status':'RUN_DONE','exit_code':0,'final_update':trainer.update_idx,'unix':time.time()})
        except BaseException as exc:
            write(run/'ERROR.json',{'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()}); raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    if True: main()

def run_verdict_l0b():
    """Run former verdict_l0b.py stage."""
    """Apply the prospective Gate-0B thresholds to L0-B update-500 artifacts."""
    import json
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[4]
    TH={'survival_rate':lambda x:x>=.80,'mean_abs_vx_error':lambda x:x<=.25,'tilt_p95_deg':lambda x:x<=20.,'tilt_max_deg':lambda x:x<=40.,'base_contact_rate':lambda x:x<=.10}
    def main():
        out=ROOT/'artifacts/l0b_verdict'; out.mkdir(parents=True,exist_ok=True); seeds={}; overall=True
        for seed in (0,1,2):
            p=ROOT/f'artifacts/l0b_seed{seed}_eval/evaluation.json'; d=json.loads(p.read_text()); cells={}
            for cmd,m in d['cells'].items():
                checks={k:bool(fn(float(m[k]))) for k,fn in TH.items()}; cells[cmd]={'metrics':{k:m[k] for k in TH},'checks':checks,'pass':all(checks.values())}; overall &= cells[cmd]['pass']
            seeds[str(seed)]={'checkpoint':d['checkpoint'],'cells':cells}
        verdict={'schema':'l0b_verdict_v1','formulation':'L0-B','checkpoint_policy':'update_500_only','pass':overall,'thresholds':{'survival_rate':.80,'mean_abs_vx_error':.25,'tilt_p95_deg':20.,'tilt_max_deg':40.,'base_contact_rate':.10},'seeds':seeds}
        (out/'L0B_VERDICT.json').write_text(json.dumps(verdict,indent=2,sort_keys=True)+'\n'); print(json.dumps({'pass':overall,'path':str(out/'L0B_VERDICT.json')},indent=2))
        lines=['# L0-B Gate-0B verdict','',f"Verdict: **{'PASS' if overall else 'FAIL'}**",'', 'Decision uses update-500 checkpoints only and requires every seed × command cell to pass all thresholds.', '']
        for seed,data in seeds.items():
            for cmd,cell in data['cells'].items():
                m=cell['metrics']; lines.append(f"- seed {seed}, vx={cmd}: {'PASS' if cell['pass'] else 'FAIL'} (survival={m['survival_rate']:.3f}, vx_MAE={m['mean_abs_vx_error']:.3f}, tilt_p95={m['tilt_p95_deg']:.2f}°, max_tilt={m['tilt_max_deg']:.2f}°, base_contact={m['base_contact_rate']:.3f})")
        (out/'report.md').write_text('\n'.join(lines)+'\n')
    if True: main()

STAGES = {
    "authorize_l0b": run_authorize_l0b,
    "freeze_l0b": run_freeze_l0b,
    "l0b_production_smoke": run_l0b_production_smoke,
    "train_l0b": run_train_l0b,
    "verdict_l0b": run_verdict_l0b,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
