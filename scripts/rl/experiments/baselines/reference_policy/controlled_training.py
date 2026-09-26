"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_l0c1_analyze():
    """Run former l0c1_analyze.py stage."""
    """Compare L0-C1 snapshots and locate the earliest telemetry divergence."""
    import json
    from pathlib import Path
    import numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    RUNS={0:ROOT/'runs/l0c1_seed0_2026-09-22',1:ROOT/'runs/l0c1_seed1_2026-09-22',2:ROOT/'runs/l0c1_seed2_2026-09-22'}
    FIELDS=['reward_tracking','reward_posture','reward_height','reward_terminal_fall','base_contact','height','roll','pitch','vx_error','action_norm','action_saturation','lane_age']
    def main():
     updates=[10,25,50,100,200,300,400,500]; data={s:{} for s in RUNS}
     for s,run in RUNS.items():
      for u in updates:
       p=run/'snapshots'/f'update_{u:03d}.json'; d=json.loads(p.read_text()); data[s][u]=d['telemetry']
     summary=[]
     for u in updates:
      row={'update':u}
      for f in FIELDS:
       vals=np.array([float(np.mean(data[s][u][f])) if isinstance(data[s][u][f],list) else float(data[s][u][f]) for s in RUNS]); row[f]={'seed0':float(vals[0]),'seed1':float(vals[1]),'seed2':float(vals[2]),'range':float(vals.max()-vals.min())}
      summary.append(row)
     # seed2-vs-(seed0,seed1) separation, normalized by scale with epsilon.
     divergence={}
     for f in FIELDS:
      for row in summary:
       v=row[f]; base=(v['seed0']+v['seed1'])/2; scale=max(abs(base),1e-3); score=abs(v['seed2']-base)/scale
       if score>=0.5:
        divergence[f]={'first_update':row['update'],'relative_gap':float(score),'seed0':v['seed0'],'seed1':v['seed1'],'seed2':v['seed2']}; break
     report={'schema':'l0c1_analysis_v1','status':'COMPLETE_READ_ONLY','updates':updates,'fields':FIELDS,'summary':summary,'first_relative_divergence_seed2_vs_seed01':divergence,'interpretation':'This telemetry identifies when aggregate signals separate, not causality. Reward decomposition and physical event traces are now available; the earliest separation should guide the next targeted audit without changing formulation.'}
     out=ROOT/'artifacts/l0c1_analysis'; out.mkdir(parents=True,exist_ok=True); (out/'L0C1_ANALYSIS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
     lines=['# L0-C1 instrumented analysis','', 'Status: **COMPLETE — read-only**','', '## Earliest relative separation (seed2 vs mean(seed0, seed1))','']
     for f,d in divergence.items(): lines.append(f"- `{f}`: first at update {d['first_update']} (seed0={d['seed0']:.6g}, seed1={d['seed1']:.6g}, seed2={d['seed2']:.6g}, relative gap={d['relative_gap']:.3f})")
     lines += ['', 'These are descriptive telemetry divergences. No reward, PPO, environment, command, threshold, or evaluator change was made in L0-C1.']
     (out/'report.md').write_text('\n'.join(lines)+'\n'); print(out/'L0C1_ANALYSIS.json')
    if True: main()

def run_train_l0c1():
    """Run former train_l0c1.py stage."""
    """L0-C1: formulation-identical L0-B runner with telemetry only."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    MANIFEST=ROOT/'artifacts/l0b_freeze/L0B_FREEZE.json'; STEPS=24; HIDDEN=[128,128,128]; SNAP={0,10,25,50,100,200,300,400,500}
    def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
    def main():
     p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--num-envs',type=int,default=4096); p.add_argument('--updates',type=int,default=500); p.add_argument('--smoke',action='store_true'); p.add_argument('--run-dir',type=Path,required=True); a=p.parse_args(); mf=json.loads(MANIFEST.read_text())
     if not mf.get('training_authorized') or a.seed not in mf['seeds']: raise RuntimeError('L0-B manifest/seed is not authorized')
     if not a.smoke and (a.num_envs!=mf['num_envs'] or a.updates!=mf['updates']): raise RuntimeError('L0-C1 CLI differs from L0-B contract')
     run=a.run_dir.resolve()
     if run.exists(): raise FileExistsError(run)
     for q in (run,run/'checkpoints',run/'snapshots'): q.mkdir(parents=True)
     write(run/'config.json',{'schema':'l0c1_runner_v1','inherits':'l0b_freeze_v1','seed':a.seed,'num_envs':a.num_envs,'updates':a.updates,'telemetry_only':True,'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}); write(run/'RUN_STARTED.json',{'status':'RUN_STARTED','seed':a.seed,'unix':time.time()})
     app=base=None
     try:
      from isaaclab.app import AppLauncher; app=AppLauncher({'headless':True,'enable_cameras':False}).app
      import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
      from rl.experiments.common.utilities.train_b0 import nominalize
      from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
      from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
      from talon_rl.rewards.baselines import b0_reward
      from rl.core.modules.actor_critic import ActorCritic
      from rl.core.algorithms.scalar_ppo import ScalarRolloutBuffer,scalar_gae
      from rl.core.algorithms.scalar_ppo import L0ReferencePPO,L0ReferenceConfig
      torch.manual_seed(a.seed); np.random.seed(a.seed); cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=a.num_envs; cfg.seed=a.seed; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg)
      base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base); model=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,HIDDEN).to(base.device); trainer=L0ReferencePPO(model,L0ReferenceConfig()); tr=env.reset(); obs=tr['obs']; age=np.zeros(a.num_envs,np.int32); metrics=[]
      for u in range(a.updates):
       env.command=tuple(mf['commands'][u%3]); env._command(); sampled_std=trainer.begin_update(); buf=ScalarRolloutBuffer(STEPS,a.num_envs); means=[]; tel={'reward_tracking':[],'reward_posture':[],'reward_height':[],'reward_terminal_fall':[],'episode_lengths':[],'base_contact':[],'height':[],'roll':[],'pitch':[],'vx_error':[],'action_norm':[],'action_saturation':[],'lane_age':[],'command':mf['commands'][u%3]}
       for _ in range(STEPS):
        x=torch.as_tensor(obs,device=base.device,dtype=torch.float32)
        with torch.no_grad(): means.append(model.raw_mean(x).cpu()); act,lp=model.act(x); val=model.value(x).squeeze(-1)
        arr=act.cpu().numpy().astype(np.float32); envtr=env.step(arr); f=envtr['fields']; cmd=mf['commands'][u%3][0]; fall=envtr['terminal_fall']; vx=f['v_actual'][:,0]; roll=f['roll_pitch'][:,0]; pitch=f['roll_pitch'][:,1]; h=f['height']; tel['reward_tracking'].append(float(np.exp(-((vx-cmd)/.25)**2).mean())); tel['reward_posture'].append(float((-.25*(roll**2+pitch**2)).mean())); tel['reward_height'].append(float((-2*(h-env.z_nominal)**2).mean())); tel['reward_terminal_fall'].append(float((-10*fall.astype(np.float32)).mean())); tel['episode_lengths'].extend(age[envtr['done']].tolist()); tel['base_contact'].append(float(envtr['term_base_contact'].mean())); tel['height'].append(float(h.mean())); tel['roll'].append(float(roll.mean())); tel['pitch'].append(float(pitch.mean())); tel['vx_error'].append(float(np.abs(vx-cmd).mean())); tel['action_norm'].append(float(np.linalg.norm(arr,axis=1).mean())); tel['action_saturation'].append(float((np.abs(arr)>=2.99).mean())); tel['lane_age'].append(float(age.mean())); age+=1; age[envtr['done']]=0; buf.append(obs,arr,lp.cpu().numpy(),envtr['reward'],envtr['done'],val.cpu().numpy()); obs=envtr['obs']
       with torch.no_grad(): fv=model.value(torch.as_tensor(obs,device=base.device,dtype=torch.float32)).squeeze(-1).cpu().numpy()
       buf.finish(fv); ar=buf.arrays(); adv=scalar_gae(ar['rewards'],np.r_[ar['values'],ar['final_value'][None]],ar['dones'],.99,.95); ret=adv+ar['values']; fl=buf.flatten(); oldm=torch.cat(means,0).to(base.device); st=trainer.update(torch.as_tensor(fl['obs'],device=base.device,dtype=torch.float32),torch.as_tensor(fl['actions'],device=base.device,dtype=torch.float32),torch.as_tensor(fl['logp_old'],device=base.device),torch.as_tensor(adv.reshape(-1),device=base.device),torch.as_tensor(ret.reshape(-1),device=base.device),oldm,seed=a.seed); st.update({'sampled_std':sampled_std,'command':mf['commands'][u%3],'return_mean':float(ret.mean()),'adv_mean':float(adv.mean()),'adv_std':float(adv.std()),'value_error':float(np.abs(ar['values']-np.r_[ar['values'][1:],ar['final_value'][None]]).mean()),'telemetry':{k:(float(np.mean(v)) if v else 0.0) for k,v in tel.items() if k!='command'}}); metrics.append(st)
       if u+1 in SNAP or (u+1)%25==0 or u+1==a.updates: trainer.save(run/'checkpoints'/f'update_{u+1:03d}.pt')
       if u+1 in SNAP: write(run/'snapshots'/f'update_{u+1:03d}.json',{'update':u+1,'telemetry':tel,'training':st})
      trainer.save(run/'checkpoints/final.pt'); write(run/'training_metrics.json',{'metrics':metrics}); write(run/'RUN_DONE.json',{'status':'RUN_DONE','exit_code':0,'final_update':trainer.update_idx,'telemetry_only':True})
     except BaseException as e: write(run/'ERROR.json',{'status':'ERROR','error':str(e),'traceback':traceback.format_exc()}); raise
     finally:
      if base is not None: base.close()
      if app is not None: app.close()
    if True: main()

STAGES = {
    "l0c1_analyze": run_l0c1_analyze,
    "train_l0c1": run_train_l0c1,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
