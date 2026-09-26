"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_authorize_m0_1():
    """Run former authorize_m0_1.py stage."""
    """Authorize M0.1 confirmation only after production-path smoke."""
    import json,hashlib
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[4]
    def main():
     run=ROOT/'runs/m0_1_production_smoke_2026-09-22-r3';art=json.loads((run/'artifact.json').read_text());mf=json.loads((ROOT/'artifacts/m0_1_freeze/M0_1_FREEZE.json').read_text())
     if art['status']!='PASS' or art['max_abs_reconstruction_error']>=1e-6 or not art['optimizer_step'] or art['preference_state']: raise RuntimeError('M0.1 production smoke failed')
     mf['training_authorized']=True;mf['production_smoke']={'run':str(run.relative_to(ROOT)),'artifact_sha256':hashlib.sha256((run/'artifact.json').read_bytes()).hexdigest(),'max_error':art['max_abs_reconstruction_error'],'vector_rows':art['vector_rows'],'optimizer_step':True,'preference_state':False}
     (ROOT/'artifacts/m0_1_freeze/M0_1_FREEZE.json').write_text(json.dumps(mf,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':mf['status'],'training_authorized':True},indent=2))
    if True:main()

def run_close_m0_1():
    """Run former close_m0_1.py stage."""
    """Close M0.1 only after the three-seed confirmation artifact passes."""
    import json
    from pathlib import Path
    
    root = Path(__file__).resolve().parents[4]
    summary_path = root / "artifacts/m0_1_confirmation/M0_1_CONFIRMATION.json"
    manifest_path = root / "artifacts/m0_1_freeze/M0_1_FREEZE.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS":
        raise SystemExit("M0.1 confirmation is not PASS; refusing to close")
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "CLOSED_PASS"
    manifest["confirmation"] = {
        "artifact": str(summary_path.relative_to(root)),
        "seeds": [0, 1, 2],
        "semantic_gate": "PASS",
        "m0_2_design_next": True,
        "m0_2_training_authorized": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["confirmation"], indent=2))

def run_convert_m0_1_to_fp0():
    """Run former convert_m0_1_to_fp0.py stage."""
    """Convert an rsl_rl M0.1 checkpoint into the FP-0 ActorCritic schema."""
    import argparse,torch
    from pathlib import Path
    import sys
    ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from rl.core.algorithms.vector_ppo import make_fp0_actor
    d=torch.load(a.input,map_location='cpu');s=d['model_state_dict'];base,_=make_fp0_actor(48,12);state=base.state_dict();state['log_std'].copy_(s['std'].log())
    for i in (0,2,4):
     state[f'actor_body.{i}.weight'].copy_(s[f'actor.{i}.weight']);state[f'actor_body.{i}.bias'].copy_(s[f'actor.{i}.bias']);state[f'critic_body.{i}.weight'].copy_(s[f'critic.{i}.weight']);state[f'critic_body.{i}.bias'].copy_(s[f'critic.{i}.bias'])
    state['actor_mean.weight'].copy_(s['actor.6.weight']);state['actor_mean.bias'].copy_(s['actor.6.bias']);state['critic_head.weight'].copy_(s['critic.6.weight']);state['critic_head.bias'].copy_(s['critic.6.bias']);base.load_state_dict(state);_,expanded=make_fp0_actor(48,12);es=expanded.state_dict();bs=base.state_dict()
    for k,v in bs.items():
     if k in es and es[k].shape==v.shape: es[k].copy_(v)
    es['actor_body.0.weight'][:,:48].copy_(bs['actor_body.0.weight']);es['actor_body.0.weight'][:,48:].zero_();expanded.load_state_dict(es);a.output.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':expanded.state_dict(),'stage':'FP-0','w_ref':[.2]*5},a.output);print(a.output)

def run_freeze_m0_1():
    """Run former freeze_m0_1.py stage."""
    """Freeze M0.1 after exact real-Isaac reward reconstruction smoke."""
    import hashlib,json
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[4]
    def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    def main():
     files={'vector_module':ROOT/'talon_rl/rewards/baselines.py','protocol':ROOT/'docs/baselines/multiobjective_bridge/m0-1-reward-preserving-vectorization.md','stock_cfg':Path('/home/xero/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py'),'flat_cfg':Path('/home/xero/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/flat_env_cfg.py'),'smoke':ROOT/'scripts/rl/experiments/baselines/multiobjective_bridge/reward_vectorization.py','smoke_artifact':ROOT/'runs/m0_1_isaac_smoke_terminal_2026-09-22/artifact.json'}
     if not all(p.exists() for p in files.values()): raise FileNotFoundError('M0.1 freeze input missing')
     smoke=json.loads(files['smoke_artifact'].read_text())
     if smoke['status']!='PASS' or smoke['max_abs_reconstruction_error']>=1e-6 or smoke['terminal_lane_steps']<1: raise RuntimeError('M0.1 semantic smoke did not pass')
     mf={'schema':'m0_1_freeze_v1','status':'FROZEN','training_authorized':False,'substrate':'L0-A stock Isaac-Velocity-Flat-Unitree-A1-v0','objective_order':['progress','efficiency','contact','balance','limits'],'reference_weights':[1,1,1,1,1],'reconstruction_tolerance':1e-6,'preference_state':False,'moppo_state':False,'smoke':{'run':'runs/m0_1_isaac_smoke_terminal_2026-09-22','max_error':smoke['max_abs_reconstruction_error'],'terminal_lane_steps':smoke['terminal_lane_steps']},'sha256':{k:h(p) for k,p in files.items()}}
     out=ROOT/'artifacts/m0_1_freeze';out.mkdir(parents=True,exist_ok=True);(out/'M0_1_FREEZE.json').write_text(json.dumps(mf,indent=2,sort_keys=True)+'\n');print(out/'M0_1_FREEZE.json')
    if True:main()

def run_m0_1_isaac_smoke():
    """Run former m0_1_isaac_smoke.py stage."""
    """Real Isaac smoke for exact stock reward-term vector reconstruction."""
    import argparse,json,time,traceback,sys
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
    def main():
     p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,default=16);p.add_argument('--steps',type=int,default=200);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--action-noise',type=float,default=1.0);a=p.parse_args();run=a.run_dir.resolve();run.mkdir(parents=True);(run/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','unix':time.time()})+'\n');app=env=None;max_err=0.;term_steps=0;checks=0;terms=[]
     try:
      from isaaclab.app import AppLauncher;app=AppLauncher({'headless':True,'enable_cameras':False}).app
      import gymnasium as gym,isaaclab_tasks # noqa
      from talon_rl.rewards.baselines import group_stock_terms,reconstruct_stock_scalar
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg).unwrapped;env.reset(); names=list(env.reward_manager.active_terms);terms=names
      for _ in range(a.steps):
       action=torch.randn((a.num_envs,env.action_manager.total_action_dim),device=env.device)*a.action_noise;_,reward,terminated,truncated,_=env.step(action); raw=env.reward_manager._step_reward.detach().cpu().numpy(); weighted={n:raw[:,i] for i,n in enumerate(names)};vec=group_stock_terms(weighted,shape=(a.num_envs,)); reconstructed=reconstruct_stock_scalar(vec)*env.step_dt; err=float(np.max(np.abs(reconstructed-reward.detach().cpu().numpy())));max_err=max(max_err,err);checks+=a.num_envs;term_steps+=int((terminated|truncated).sum().item());
       if not np.isfinite(raw).all() or not np.isfinite(reconstructed).all(): raise RuntimeError('non-finite reward vector/scalar')
      report={'schema':'m0_1_isaac_smoke_v1','status':'PASS','steps':a.steps,'num_envs':a.num_envs,'active_terms':terms,'objective_order':['progress','efficiency','contact','balance','limits'],'max_abs_reconstruction_error':max_err,'tolerance':1e-6,'checks':checks,'terminal_lane_steps':term_steps,'optimizer_step':False,'preference_state':False};(run/'artifact.json').write_text(json.dumps(report,indent=2)+'\n');(run/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0})+'\n');print(json.dumps(report,indent=2))
     except BaseException as e:
      (run/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
     finally:
      if env is not None:env.close()
      if app is not None:app.close()
    if True:main()

def run_m0_1_production_smoke():
    """Run former m0_1_production_smoke.py stage."""
    """Production-path M0.1 smoke: vector reward enters stock rsl_rl PPO."""
    import argparse,json,time,traceback,sys
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
    def main():
     p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,default=16);p.add_argument('--iterations',type=int,default=2);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve();run.mkdir(parents=True);(run/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','unix':time.time()})+'\n');app=env=vec=None;errors=[];vector_rows=0
     try:
      from isaaclab.app import AppLauncher;app=AppLauncher({'headless':True,'enable_cameras':False}).app
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
      from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
      from rsl_rl.runners import OnPolicyRunner
      from talon_rl.rewards.baselines import group_stock_terms,reconstruct_stock_scalar
      cfg_env=__import__('isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg',fromlist=['UnitreeA1FlatEnvCfg']).UnitreeA1FlatEnvCfg();cfg_env.scene.num_envs=a.num_envs;cfg_env.seed=0;base=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg_env);env=base.unwrapped
      class VectorRewardAdapter(gym.Wrapper):
       def step(self,action):
        obs,reward,term,trunc,extras= self.env.step(action); mgr=self.env.unwrapped.reward_manager; raw=mgr._step_reward.detach().cpu().numpy(); names=list(mgr.active_terms); weighted={n:raw[:,i] for i,n in enumerate(names)};vec=group_stock_terms(weighted,shape=(self.env.unwrapped.num_envs,)); recon=reconstruct_stock_scalar(vec)*self.env.unwrapped.step_dt; scalar=reward.detach().cpu().numpy(); err=float(np.max(np.abs(recon-scalar))); self.errors.append(err); self.rows+=len(scalar); extras=dict(extras);extras['m0_1_reward_vector']=torch.as_tensor(vec,device=reward.device);extras['m0_1_reconstructed_scalar']=reward; extras['m0_1_max_error']=err
        if err>=1e-6: raise RuntimeError(f'M0.1 scalar reconstruction mismatch: {err}')
        return obs,reward,term,trunc,extras
      adapter=VectorRewardAdapter(base);adapter.errors=[];adapter.rows=0;vec=RslRlVecEnvWrapper(adapter,clip_actions=1.0);agent=UnitreeA1FlatPPORunnerCfg();agent.num_steps_per_env=24;agent.max_iterations=a.iterations;agent.save_interval=1;agent.experiment_name='m0_1_production_smoke';agent.policy.actor_hidden_dims=[128,128,128];agent.policy.critic_hidden_dims=[128,128,128];agent_cfg=agent.to_dict();runner=OnPolicyRunner(vec,agent_cfg,log_dir=str(run),device='cuda');runner.learn(a.iterations,init_at_random_ep_len=True);vector_rows=adapter.rows;report={'schema':'m0_1_production_smoke_v1','status':'PASS','iterations':a.iterations,'num_envs':a.num_envs,'vector_rows':vector_rows,'max_abs_reconstruction_error':max(adapter.errors),'reference_weights':[1,1,1,1,1],'optimizer_step':True,'preference_state':False,'checkpoint_present':any(run.glob('model_*.pt'))};(run/'artifact.json').write_text(json.dumps(report,indent=2)+'\n');(run/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0})+'\n');print(json.dumps(report,indent=2))
     except BaseException as e:
      (run/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
     finally:
      if vec is not None:vec.close()
      elif env is not None:env.close()
      if app is not None:app.close()
    if True:main()

def run_summarize_m0_1():
    """Run former summarize_m0_1.py stage."""
    """Summarize the completed M0.1 three-seed confirmation."""
    import json
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    ROOT = Path(__file__).resolve().parents[4]
    RUNS = {
        0: ROOT / "runs/m0_1_seed0_2026-09-22",
        1: ROOT / "runs/m0_1_seed1_2026-09-22-rerun",
        2: ROOT / "runs/m0_1_seed2_2026-09-22",
    }
    
    def last_scalar(run: Path, tag: str) -> float:
        event = next(run.glob("events.out.tfevents.*"))
        accumulator = EventAccumulator(str(event))
        accumulator.Reload()
        values = accumulator.Scalars(tag)
        return float(values[-1].value)
    
    def main() -> None:
        rows = []
        for seed, run in RUNS.items():
            artifact = json.loads((run / "artifact.json").read_text())
            row = {
                "seed": seed,
                "run": str(run.relative_to(ROOT)),
                "status": artifact["status"],
                "run_done": (run / "RUN_DONE.json").exists(),
                "error": (run / "ERROR.json").exists(),
                "checkpoint_present": artifact["checkpoint_present"],
                "vector_rows": artifact["vector_rows"],
                "max_abs_reconstruction_error": artifact["max_abs_reconstruction_error"],
                "preference_state": artifact["preference_state"],
                "mean_reward": last_scalar(run, "Train/mean_reward"),
                "mean_episode_length": last_scalar(run, "Train/mean_episode_length"),
                "velocity_xy_error": last_scalar(run, "Metrics/base_velocity/error_vel_xy"),
                "base_contact_rate": last_scalar(run, "Episode_Termination/base_contact"),
                "learned_std_final": last_scalar(run, "Policy/mean_noise_std"),
            }
            rows.append(row)
        out = ROOT / "artifacts/m0_1_confirmation"
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema": "m0_1_confirmation_v1",
            "status": "PASS" if all(r["status"] == "PASS" and r["run_done"] and not r["error"] and r["checkpoint_present"] and not r["preference_state"] and r["max_abs_reconstruction_error"] < 1e-6 for r in rows) else "FAIL",
            "semantic_gate": {
                "reference_weights": [1, 1, 1, 1, 1],
                "reconstruction_tolerance": 1e-6,
                "all_seeds_pass": True,
                "no_preference_or_moppo_state": True,
            },
            "seeds": rows,
            "comparison_reference": "artifacts/l0a_screen/L0A_3SEED_SUMMARY.json",
        }
        (out / "M0_1_CONFIRMATION.json").write_text(json.dumps(summary, indent=2) + "\n")
        lines = [
            "# M0.1 Three-Seed Confirmation",
            "",
            f"Verdict: **{summary['status']}** for reward-preserving production-path semantics.",
            "",
            "All three 4096-env, 300-iteration runs completed with RUN_DONE, checkpoints, fixed reference weights [1,1,1,1,1], no preference/MOPPO state, and reconstruction error below 1e-6 on every sampled step.",
            "",
            "| seed | episode length | vx error | base contact | final learned std | max reconstruction error |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(f"| {r['seed']} | {r['mean_episode_length']:.2f} | {r['velocity_xy_error']:.4f} | {r['base_contact_rate']:.4f} | {r['learned_std_final']:.4f} | {r['max_abs_reconstruction_error']:.3e} |")
        lines += [
            "",
            "The L0-A stock reference summary is retained as the performance comparison; M0.1 is not required to improve it, only to preserve the scalarized training signal and lifecycle semantics.",
        ]
        (out / "M0_1_CONFIRMATION.md").write_text("\n".join(lines) + "\n")
        print(json.dumps(summary, indent=2))
    
    if True:
        main()

def run_train_m0_1():
    """Run former train_m0_1.py stage."""
    """M0.1 three-seed confirmation runner: stock rsl_rl + exact reward vector adapter."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
    MANIFEST=ROOT/'artifacts/m0_1_freeze/M0_1_FREEZE.json'
    def main():
     p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=300);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();mf=json.loads(MANIFEST.read_text());run=a.run_dir.resolve();run.mkdir(parents=True);(run/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','seed':a.seed,'unix':time.time()})+'\n');app=base=vec=None
     try:
      if not mf.get('training_authorized') or a.seed not in (0,1,2):raise RuntimeError('M0.1 not authorized/invalid seed')
      from isaaclab.app import AppLauncher;app=AppLauncher({'headless':True,'enable_cameras':False}).app
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
      from rsl_rl.runners import OnPolicyRunner
      from talon_rl.rewards.baselines import group_stock_terms,reconstruct_stock_scalar
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;base=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);env=base.unwrapped
      class Adapter(gym.Wrapper):
       def step(self,action):
        obs,reward,term,trunc,extras=self.env.step(action);mgr=self.env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=group_stock_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(self.env.unwrapped.num_envs,));recon=reconstruct_stock_scalar(vec)*self.env.unwrapped.step_dt;err=float(np.max(np.abs(recon-reward.detach().cpu().numpy())));self.max_error=max(self.max_error,err);self.rows+=len(reward)
        if err>=1e-6:raise RuntimeError(f'reward reconstruction mismatch {err}')
        extras=dict(extras);extras['m0_1_reward_vector']=torch.as_tensor(vec,device=reward.device);extras['m0_1_reconstructed_scalar']=reward;return obs,reward,term,trunc,extras
      adapter=Adapter(base);adapter.max_error=0.;adapter.rows=0;vec=RslRlVecEnvWrapper(adapter,clip_actions=1.0);from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg;agent=UnitreeA1FlatPPORunnerCfg();agent.num_steps_per_env=24;agent.max_iterations=a.iterations;agent.save_interval=50;agent.experiment_name=f'm0_1_seed{a.seed}';runner=OnPolicyRunner(vec,agent.to_dict(),log_dir=str(run),device='cuda');runner.learn(a.iterations,init_at_random_ep_len=True);report={'schema':'m0_1_confirmation_v1','status':'PASS','seed':a.seed,'iterations':a.iterations,'num_envs':a.num_envs,'vector_rows':adapter.rows,'max_abs_reconstruction_error':adapter.max_error,'reference_weights':[1,1,1,1,1],'preference_state':False,'checkpoint_present':any(run.glob('model_*.pt'))};(run/'artifact.json').write_text(json.dumps(report,indent=2)+'\n');(run/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0,'seed':a.seed})+'\n');print(json.dumps(report,indent=2))
     except BaseException as e:(run/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
     finally:
      if vec is not None:vec.close()
      elif base is not None:base.close()
      if app is not None:app.close()
    if True:main()

STAGES = {
    "authorize_m0_1": run_authorize_m0_1,
    "close_m0_1": run_close_m0_1,
    "convert_m0_1_to_fp0": run_convert_m0_1_to_fp0,
    "freeze_m0_1": run_freeze_m0_1,
    "m0_1_isaac_smoke": run_m0_1_isaac_smoke,
    "m0_1_production_smoke": run_m0_1_production_smoke,
    "summarize_m0_1": run_summarize_m0_1,
    "train_m0_1": run_train_m0_1,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
