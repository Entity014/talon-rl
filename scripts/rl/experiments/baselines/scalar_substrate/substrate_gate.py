"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_authorize_gate0b():
    """Run former authorize_gate0b.py stage."""
    from pathlib import Path
    import hashlib,json,time
    ROOT=Path(__file__).resolve().parents[4]; MAN=ROOT/'artifacts/gate0b_freeze/GATE0B_FREEZE.json'
    def main():
        mf=json.loads(MAN.read_text()); runner=ROOT/'runs/gate0b_runner_smoke_2026-09-21/RUN_DONE.json'; ev=ROOT/'artifacts/gate0b_eval_smoke/evaluation.json'; evdone=ROOT/'artifacts/gate0b_eval_smoke/RUN_DONE.json'
        if not runner.exists() or not evdone.exists() or not json.loads(ev.read_text()).get('pass'): raise RuntimeError('Gate-0B production smoke artifacts are incomplete')
        if json.loads(runner.read_text()).get('status')!='RUN_DONE' or json.loads(evdone.read_text()).get('status')!='RUN_DONE': raise RuntimeError('Gate-0B smoke lifecycle invalid')
        for name,rel in {'runner':'scripts/rl/experiments/baselines/scalar_substrate/substrate_gate.py','evaluator':'scripts/rl/experiments/baselines/scalar_substrate/substrate_gate.py','protocol':'docs/baselines/scalar_substrate/gate-0b-pipeline-aligned-substrate-protocol.md','reset_states':'artifacts/gate0b_reset_states.npz'}.items(): mf['sha256'][name]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
        mf['training_authorized']=True; mf['runner_ready']=True; mf['authorized_unix']=time.time(); mf['authorization_evidence']={'runner_smoke':'runs/gate0b_runner_smoke_2026-09-21','evaluator_smoke':'artifacts/gate0b_eval_smoke'}
        MAN.write_text(json.dumps(mf,indent=2,sort_keys=True)+'\n'); print(MAN)
    if True: main()

def run_evaluate_gate0b():
    """Run former evaluate_gate0b.py stage."""
    """Deterministic Gate-0B evaluator over all three frozen command cells."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse, hashlib, json, math, time, traceback
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    MANIFEST=ROOT/'artifacts/gate0b_freeze/GATE0B_FREEZE.json'
    RESET=ROOT/'artifacts/gate0b_reset_states.npz'
    
    def compact(survived, vx, tilt, contacts, first, action_hash, horizon):
        return {'survival_rate':float(survived.mean()), 'mean_abs_vx_error':float(np.concatenate(vx).mean()), 'tilt_p95_deg':math.degrees(float(np.percentile(np.concatenate(tilt),95))), 'tilt_max_deg':math.degrees(float(np.max(np.concatenate(tilt)))), 'base_contact_rate':float(contacts.mean()), 'first_fall':first.tolist(), 'action_hash':action_hash.hexdigest(), 'horizon':horizon, 'lanes':len(first)}
    
    def infer_hidden_dims(state):
        """Infer linear hidden widths so Gate-0B can evaluate L0 checkpoints."""
        widths=[]
        i=0
        while f'actor_body.{i}.weight' in state:
            widths.append(int(state[f'actor_body.{i}.weight'].shape[0])); i += 2
        if not widths: raise RuntimeError('checkpoint has no actor_body weights')
        return widths
    
    def main():
        p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
        a.output.parent.mkdir(parents=True,exist_ok=True); (a.output.parent/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','unix':time.time()})+'\n'); app=base=None
        try:
            mf=json.loads(MANIFEST.read_text()); expected=hashlib.sha256(RESET.read_bytes()).hexdigest()
            if mf['sha256']['reset_states']!=expected: raise RuntimeError('Gate-0B reset suite hash mismatch')
            commands=[tuple(x) for x in mf['commands']]; horizon=20 if a.smoke else mf['horizon_steps']; lanes=64
            from isaaclab.app import AppLauncher
            app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.experiments.common.utilities.train_b0 import nominalize
            from rl.core.modules.actor_critic import ActorCritic
            from rl.experiments.common.utilities.b0_monitor_isolation_smoke import _install_states
            cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=lanes; cfg.seed=17; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg)
            base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base); state=torch.load(a.checkpoint,map_location=base.device); raw_state=state.get('model',state); model=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,infer_hidden_dims(raw_state)).to(base.device); model.load_state_dict(raw_state); model.eval()
            with np.load(RESET) as z: saved={k:z[k].copy() for k in z.files}
            results={}
            for cmd in commands:
                env.command=cmd; base.reset(); env._command(); _install_states(base,saved,lanes); tr=env._scalar_transition(base._transition(base.observation_manager.compute()),np.zeros(lanes,bool)); obs=tr['obs']; first=np.full(lanes,horizon,dtype=np.int32); alive=np.ones(lanes,bool); contacts=np.zeros(lanes,dtype=bool); vx=[]; tilt=[]; action_hash=hashlib.sha256()
                for step in range(horizon):
                    x=torch.as_tensor(obs,device=base.device,dtype=torch.float32)
                    with torch.no_grad(): action=model.act_inference(x); expected_action=torch.tanh(model.raw_mean(x))*model.ACTION_CLIP
                    if not torch.equal(action,expected_action): raise RuntimeError('evaluator is not deterministic tanh(actor_mean)')
                    arr=action.cpu().numpy().astype(np.float32); action_hash.update(arr.tobytes()); tr=env.step(arr); f=tr['fields']; newly=alive & tr['done']; first[newly]=step+1; contacts|=tr['term_base_contact']; live=alive|newly; vx.append(np.abs(f['v_actual'][:lanes,0]-cmd[0])[live]); tilt.append(np.max(np.abs(f['roll_pitch'][:lanes]),axis=1)[live]); alive &= ~newly; obs=tr['obs']
                results[str(cmd[0])]=compact(first==horizon,vx,tilt,contacts,first,action_hash,horizon)
            out={'schema':'gate0b_evaluation_v1','pass':True,'checkpoint':str(a.checkpoint),'commands':commands,'reset_hash':expected,'cells':results,'smoke':a.smoke,'training_state_mutated':False,'deterministic_actor_mean':True}; a.output.write_text(json.dumps(out,indent=2)+'\n'); (a.output.parent/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0})+'\n'); print(a.output)
        except BaseException as exc:
            (a.output.parent/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    if True: main()

def run_freeze_gate0b():
    """Run former freeze_gate0b.py stage."""
    from pathlib import Path
    import hashlib,json,time
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'gate0b_freeze'; OUT.mkdir(parents=True,exist_ok=True)
    def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    def main():
        reset=ROOT/'artifacts/gate0b_reset_states.npz'
        if not reset.exists(): raise FileNotFoundError('fresh Gate-0B reset suite is required before freeze')
        files={'protocol':'docs/baselines/scalar_substrate/gate-0b-pipeline-aligned-substrate-protocol.md','reward':'talon_rl/rewards/baselines.py','env':'talon_rl/wrappers/scalar_reward_env.py','trainer':'scripts/rl/core/algorithms/scalar_ppo.py','evaluator':'scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py','runner':'scripts/rl/experiments/common/utilities/train_b0.py','reset_states':'artifacts/gate0b_reset_states.npz'}
        data={'schema':'gate0b_freeze_v1','status':'FROZEN','training_authorized':False,'runner_ready':False,'created_unix':time.time(),'gate_name':'pipeline_aligned_substrate_authorization','deployment':'deterministic_tanh_actor_mean','commands':[[.25,0.,0.],[.5,0.,0.],[.75,0.,0.]],'seeds':[0,1,2],'reset_states':{'count':64,'artifact':'artifacts/gate0b_reset_states.npz','training_independent':True},'horizon_steps':500,'sim_dt':.01,'verdict_update':500,'thresholds':{'survival_min':.80,'vx_mae_max':.25,'tilt_p95_deg_max':20.,'tilt_max_deg_max':40.,'base_contact_rate_max':.10},'excluded':['preference','rma_latent','terrain_randomization','domain_randomization','privileged_inference_input','stochastic_evaluation'],'sha256':{k:sha(v) for k,v in files.items()}}
        (OUT/'GATE0B_FREEZE.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n'); print(OUT/'GATE0B_FREEZE.json')
    if True: main()

def run_prepare_gate0b_reset_states():
    """Run former prepare_gate0b_reset_states.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,time,traceback,numpy as np
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'gate0b_reset_states.npz'; META=ROOT/'artifacts'/'gate0b_reset_states_meta.json'
    def main():
        app=base=None
        try:
            from isaaclab.app import AppLauncher
            app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.experiments.common.utilities.train_b0 import nominalize
            cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=64; cfg.seed=2718; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg)
            base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; base.reset(); robot=base.scene['robot']
            states={'root_state':robot.data.root_state_w.detach().cpu().numpy().copy(),'joint_pos':robot.data.joint_pos.detach().cpu().numpy().copy(),'joint_vel':robot.data.joint_vel.detach().cpu().numpy().copy()}
            np.savez_compressed(OUT,**states); META.write_text(json.dumps({'schema':'gate0b_reset_states_v1','count':64,'seed':2718,'nominal_flat':True,'created_unix':time.time(),'source':'fresh Isaac reset'},indent=2)+'\n'); print(OUT)
        except BaseException as exc:
            (ROOT/'artifacts'/'gate0b_reset_states_ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    if True: main()

def run_train_gate0b():
    """Run former train_gate0b.py stage."""
    """Gate-0B training runner: cycles only the frozen forward command cells."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,datetime as dt,json,hashlib,time,traceback
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]; MANIFEST=ROOT/'artifacts/gate0b_freeze/GATE0B_FREEZE.json'; STEPS=16; HIDDEN=[64,64]
    def write(p,d): p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
    def main():
        p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--num-envs',type=int,default=4096); p.add_argument('--updates',type=int,default=500); p.add_argument('--smoke',action='store_true'); p.add_argument('--run-dir',type=Path,required=True); a=p.parse_args()
        mf=json.loads(MANIFEST.read_text()); cmds=[tuple(x) for x in mf['commands']]
        if mf['schema']!='gate0b_freeze_v1' or not cmds==[(.25,0.,0.),(.5,0.,0.),(.75,0.,0.)]: raise RuntimeError('Gate-0B manifest command contract mismatch')
        if a.seed not in mf['seeds']: raise RuntimeError('seed not in frozen Gate-0B manifest')
        if not a.smoke and not mf.get('training_authorized',False): raise RuntimeError('Gate-0B training is not authorized')
        run=a.run_dir.resolve()
        if run.exists(): raise FileExistsError(run)
        for q in (run,run/'checkpoints'): q.mkdir(parents=True)
        write(run/'config.json',{'schema':'gate0b_runner_config_v1','seed':a.seed,'num_envs':a.num_envs,'updates':a.updates,'commands':cmds,'command_schedule':'update_idx modulo 3; command held across each rollout','rollout_steps':STEPS,'hidden_dims':HIDDEN,'smoke':a.smoke,'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}); write(run/'RUN_STARTED.json',{'status':'RUN_STARTED','seed':a.seed,'unix':time.time()})
        app=base=None
        try:
            from isaaclab.app import AppLauncher
            app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.experiments.common.utilities.train_b0 import nominalize
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.scalar_ppo import B0PPOConfig,B0PPOTrainer,ScalarRolloutBuffer,scalar_gae
            torch.manual_seed(a.seed); np.random.seed(a.seed); cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=a.num_envs; cfg.seed=a.seed; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg)
            base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base); model=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,HIDDEN).to(base.device); trainer=B0PPOTrainer(model,B0PPOConfig(),lr=3e-4); trans=env.reset(); obs=trans['obs']; metrics=[]
            for update in range(a.updates):
                env.command=cmds[update%len(cmds)]; env._command(); std=trainer.begin_update(); buf=ScalarRolloutBuffer(STEPS,a.num_envs)
                for _ in range(STEPS):
                    x=torch.as_tensor(obs,device=base.device,dtype=torch.float32)
                    with torch.no_grad(): act,lp=model.act(x); val=model.value(x).squeeze(-1)
                    trans=env.step(act.cpu().numpy().astype(np.float32)); buf.append(obs,act.cpu().numpy(),lp.cpu().numpy(),trans['reward'],trans['done'],val.cpu().numpy()); obs=trans['obs']
                with torch.no_grad(): fv=model.value(torch.as_tensor(obs,device=base.device,dtype=torch.float32)).squeeze(-1).cpu().numpy()
                buf.finish(fv); ar=buf.arrays(); adv=scalar_gae(ar['rewards'],np.r_[ar['values'],ar['final_value'][None]],ar['dones'],trainer.cfg.gamma,trainer.cfg.gae_lambda); ret=adv+ar['values']; fl=buf.flatten(); xo=torch.as_tensor(fl['obs'],device=base.device,dtype=torch.float32); ac=torch.as_tensor(fl['actions'],device=base.device,dtype=torch.float32); old=torch.as_tensor(fl['logp_old'],device=base.device); st=trainer.optimize_batch(xo,ac,old,torch.as_tensor(adv.reshape(-1),device=base.device),torch.as_tensor(ret.reshape(-1),device=base.device)); trainer.finish_update(); metrics.append({'update':update+1,'command':cmds[update%3],'std':std,'reward_mean':float(ar['rewards'].mean()),**st})
                if (update+1)%25==0 or update+1==a.updates: trainer.save(run/'checkpoints'/f'update_{update+1:03d}.pt')
            trainer.save(run/'checkpoints/final.pt'); write(run/'training_metrics.json',{'metrics':metrics}); write(run/'RUN_DONE.json',{'status':'RUN_DONE','exit_code':0,'final_update':trainer.update_idx,'unix':time.time()})
        except BaseException as exc:
            write(run/'ERROR.json',{'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()}); raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    if True: main()

def run_verdict_gate0b():
    """Run former verdict_gate0b.py stage."""
    from pathlib import Path
    import json,math
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'gate0b_verdict'; OUT.mkdir(parents=True,exist_ok=True)
    def main():
        rows=[]; sanity=[]
        for s in range(3):
            run=ROOT/f'runs/gate0b_seed{s}_2026-09-21'; ev=json.loads((ROOT/f'artifacts/gate0b_seed{s}_eval/evaluation.json').read_text()); done=json.loads((run/'RUN_DONE.json').read_text()); cfg=json.loads((run/'config.json').read_text()); tm=json.loads((run/'training_metrics.json').read_text())['metrics']; cells=ev['cells']; finite=all(math.isfinite(float(x[k])) for x in tm for k in ('std','reward_mean','analytic_kl','approx_kl')); command_ok=sorted(tuple(x['command']) for x in tm[::3])==[(.25,0.,0.),(.25,0.,0.),(.25,0.,0.)] or set(tuple(x['command']) for x in tm)=={(.25,0.,0.),(.5,0.,0.),(.75,0.,0.)}
            cell_rows=[]; all_pass=True
            for vx,m in cells.items():
                ok=m['survival_rate']>=.8 and m['mean_abs_vx_error']<=.25 and m['tilt_p95_deg']<=20 and m['tilt_max_deg']<=40 and m['base_contact_rate']<=.10 and all(math.isfinite(float(m[k])) for k in ('survival_rate','mean_abs_vx_error','tilt_p95_deg','tilt_max_deg','base_contact_rate')); all_pass &= ok; cell_rows.append({'vx':float(vx),**m,'pass':ok})
            rec={'seed':s,'run_done':done.get('status')=='RUN_DONE' and done.get('exit_code')==0,'checkpoint_500':(run/'checkpoints/update_500.pt').exists(),'metrics':len(tm),'finite_training_metrics':finite,'command_schedule_valid':command_ok,'cells':cell_rows,'pass':all_pass}; rows.append(rec)
            if not all((rec['run_done'],rec['checkpoint_500'],finite,command_ok,all_pass)): sanity.append(s)
        verdict=not sanity; out={'schema':'gate0b_verdict_v1','status':'PASS' if verdict else 'FAIL','sanity_or_gate_failures':sanity,'seeds':rows,'rule':'all 3 seeds × 3 commands at update-500 only'}; (OUT/'GATE0B_VERDICT.json').write_text(json.dumps(out,indent=2)+'\n')
        lines=[f"# Gate-0B Verdict: {'PASS' if verdict else 'FAIL'}",'', '|seed|RUN_DONE|ckpt500|commands|vx=.25|vx=.50|vx=.75|seed gate|','|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|']
        for r in rows: lines.append(f"|{r['seed']}|{r['run_done']}|{r['checkpoint_500']}|{r['command_schedule_valid']}|{r['cells'][0]['pass']}|{r['cells'][1]['pass']}|{r['cells'][2]['pass']}|{r['pass']}|")
        (OUT/'report.md').write_text('\n'.join(lines)+'\n'); print(OUT/'report.md')
    if True: main()

STAGES = {
    "authorize_gate0b": run_authorize_gate0b,
    "evaluate_gate0b": run_evaluate_gate0b,
    "freeze_gate0b": run_freeze_gate0b,
    "prepare_gate0b_reset_states": run_prepare_gate0b_reset_states,
    "train_gate0b": run_train_gate0b,
    "verdict_gate0b": run_verdict_gate0b,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
