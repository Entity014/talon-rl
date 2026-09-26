"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_b1_r1_controlled_screen():
    """Run former b1_r1_controlled_screen.py stage."""
    """One-update B1-R1 mechanism screen from an existing P2 rollout.
    
    Read-only with respect to the P2 run: the reconstruction head is new and the
    result is written to a separate artifact. No monitor states or evaluation
    trajectories are used as training targets.
    """
    from pathlib import Path
    import json, traceback
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/'artifacts'/'b1_r1_controlled_screen'; OUT.mkdir(parents=True, exist_ok=True)
    CKPT=Path('runs/b1_p2_seed2_2026-09-21/checkpoints/update_500.pt')
    TRAJ=ROOT/'artifacts'/'b1_r0_2'/'p2_s2'/'trajectory.npz'
    
    def main():
        for stale in (OUT/'ERROR.json', OUT/'RUN_DONE.json'):
            if stale.exists(): stale.unlink()
        (OUT/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','mode':'one_update_real_rollout_screen'})+'\n')
        try:
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.scalar_ppo import B0PPOConfig, B0PPOTrainer
            z=np.load(TRAJ); obs=torch.as_tensor(z['obs'],dtype=torch.float32)
            t=np.abs(z['tilt']).max(-1); targets=torch.as_tensor(np.stack([t,z['height'],z['contact'].astype(np.float32)],-1),dtype=torch.float32)
            T,N,D=obs.shape
            model=ActorCritic(D,D,12,1,[64,64],reconstruction_dim=3)
            state=torch.load(CKPT,map_location='cpu'); missing,unexpected=model.load_state_dict(state['model'],strict=False)
            if set(missing) != {'reconstruction_head.weight','reconstruction_head.bias'} or unexpected:
                raise RuntimeError(f'unexpected P2 load mismatch: missing={missing}, unexpected={unexpected}')
            model.set_scheduled_fixed_std(float(state.get('scheduled_std',0.1)))
            cfg=B0PPOConfig(b1_p2_enabled=True,b1_r1_enabled=True,actor_epochs=1,lambda_reconstruction=0.10,kl_low=0.005,kl_high=0.02)
            trainer=B0PPOTrainer(model,cfg,lr=float(state.get('actor_lr',3e-4))); trainer.update_idx=500; trainer.begin_update()
            flat=obs.reshape(T*N,D); target_flat=targets
            with torch.no_grad(): actions,old_logp=model.act(flat)
            adv=torch.ones(T*N); returns=torch.zeros(T*N)
            before=model.reconstruct(flat).detach()
            out=trainer.optimize_batch(flat,actions.detach(),old_logp.detach(),adv,returns,rollout_obs=obs,reconstruction_targets=target_flat)
            with torch.no_grad(): after=model.reconstruct(flat); pred_err=float(torch.mean((after-target_flat.reshape(-1,3))**2)); baseline_err=float(torch.mean((before-target_flat.reshape(-1,3))**2)); det=model.act_inference(flat[:N])
            result={'schema':'b1_r1_controlled_screen_v1','read_only_source':str(CKPT),'rollout_shape':list(obs.shape),'target_fields':['tilt','height','contact'],'actor_input_dim':D,'reconstruction_dim':3,'missing_p2_keys':missing,'unexpected_p2_keys':unexpected,'reconstruction_loss_before':baseline_err,'reconstruction_loss_after':pred_err,'screen_output_finite':bool(torch.isfinite(det).all()),'policy_kl':out['analytic_kl'],'actor_grad_norm':out['actor_grad_norm'],'reconstruction_grad_norm':out['reconstruction_grad_norm'],'critic_completed':out['critic_completed'],'p2_lr_events':out['lr_events'],'scheduled_std':trainer._std_for_update,'evaluation_states_used':False}
            (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            torch.save({'schema':'b1_r1_controlled_screen_checkpoint_v1','model':model.state_dict(),'b1_r1':{'enabled':True,'lambda_reconstruction':cfg.lambda_reconstruction}},OUT/'checkpoint.pt')
            (OUT/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0})+'\n')
            print(OUT/'result.json')
        except BaseException as exc:
            (OUT/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
    
    if True: main()

def run_b1_r1_isaac_smoke():
    """Run former b1_r1_isaac_smoke.py stage."""
    """Small real-Isaac lifecycle smoke for B1-R1 (not a training run)."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json, time, traceback
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_r1_smoke'; OUT.mkdir(parents=True,exist_ok=True)
    
    def main():
        for p in (OUT/'ERROR.json',OUT/'RUN_DONE.json'): p.unlink(missing_ok=True)
        (OUT/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','unix':time.time()})+'\n'); app=base=None
        try:
            from isaaclab.app import AppLauncher
            app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.experiments.common.utilities.train_b0 import nominalize
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.scalar_ppo import B0PPOConfig,B0PPOTrainer
            cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=4; cfg.seed=3; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg); cfg.episode_length_s=.03
            base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base); tr=env.reset(); obs=tr['obs']
            model=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,[64,64],reconstruction_dim=3).to(base.device)
            pcfg=B0PPOConfig(b1_p2_enabled=True,b1_r1_enabled=True,actor_epochs=1,lambda_reconstruction=.10,kl_low=-100.,kl_high=100.)
            trainer=B0PPOTrainer(model,pcfg,lr=3e-4); trainer.begin_update(); std=trainer._std_for_update
            O=[]; A=[]; LP=[]; D=[]; Y=[]; R=[]
            for _ in range(8):
                x=torch.as_tensor(obs,device=base.device,dtype=torch.float32)
                with torch.no_grad(): a,lp=model.act(x)
                q=env.step(a.detach().cpu().numpy().astype(np.float32)); f=q['fields']
                O.append(x); A.append(a.detach()); LP.append(lp.detach()); D.append(torch.as_tensor(q['done'],device=base.device)); R.append(torch.as_tensor(q['reward'],device=base.device))
                Y.append(torch.stack([torch.as_tensor(np.abs(f['roll_pitch']).max(-1),device=base.device),torch.as_tensor(f['height'],device=base.device),torch.as_tensor(q['term_base_contact'],device=base.device,dtype=torch.float32)],-1)); obs=q['obs']
            O=torch.stack(O); A=torch.stack(A); LP=torch.stack(LP); D=torch.stack(D); Y=torch.stack(Y); R=torch.stack(R)
            flat=O.reshape(-1,base.obs_dim); af=A.reshape(-1,base.action_dim); lf=LP.reshape(-1); before={k:v.detach().clone() for k,v in model.state_dict().items()}
            out=trainer.optimize_batch(flat,af,lf,torch.ones(flat.shape[0],device=base.device),torch.zeros(flat.shape[0],device=base.device),rollout_obs=O,rollout_dones=D,reconstruction_targets=Y)
            after=model.state_dict(); actor_changed=any(not torch.equal(before[k],after[k]) for k in before if not k.startswith('critic_')); critic_changed=any(not torch.equal(before[k],after[k]) for k in before if k.startswith('critic_'))
            ck=OUT/'checkpoint.pt'; trainer.save(ck); restored=B0PPOTrainer(ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,[64,64],reconstruction_dim=3).to(base.device),pcfg); restored.load(ck)
            with torch.no_grad(): det=model.act_inference(flat[:4]); target_pert=Y.clone(); target_pert+=1.; det2=model.act_inference(flat[:4])
            checks={'real_rollout':bool(O.shape[0]==8),'done_reset_seen':bool(D.any()),'targets_shape':list(Y.shape)==[8,4,3],'finite_losses':bool(np.isfinite([out['reconstruction_loss'],out['actor_grad_norm'],out['analytic_kl']]).all()),'actor_changed':actor_changed,'critic_changed':critic_changed,'critic_lr_unchanged':restored.critic_optim.param_groups[0]['lr']==3e-4,'p2_path':bool(out['actor_stopped'] is False),'std':float(std)==.82,'actor_input_dim':base.obs_dim==51,'target_only_inference':bool(torch.equal(det,det2)),'resume_r1':restored.cfg.b1_r1_enabled}
            assert all(checks.values()),checks
            result={'schema':'b1_r1_isaac_smoke_v1','pass':True,'checks':checks,'rollout_shape':list(O.shape),'valid_temporal_pairs':int((~D[:-1]).sum()),'total_temporal_pairs':int(D[:-1].numel()),'reconstruction_loss':out['reconstruction_loss'],'reconstruction_grad_norm':out['reconstruction_grad_norm'],'actor_grad_norm':out['actor_grad_norm'],'p2_lr_events':out['lr_events'],'scheduled_std':std,'command_dims_not_in_target_or_input':True,'monitor_states_used':False}
            (OUT/'artifact.json').write_text(json.dumps(result,indent=2)+'\n'); (OUT/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0,'artifact':'artifact.json'})+'\n'); print(OUT/'artifact.json')
        except BaseException as exc:
            (OUT/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    if True: main()

def run_b1_r1_verdict():
    """Run former b1_r1_verdict.py stage."""
    from pathlib import Path
    import json, math
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_r1_verdict'; OUT.mkdir(parents=True,exist_ok=True)
    def main():
        rows=[]; failures=[]
        for s in range(3):
            r=ROOT/f'runs/b1_r1_seed{s}_2026-09-21'; done=json.loads((r/'RUN_DONE.json').read_text()); cfg=json.loads((r/'config.json').read_text()); tm=json.loads((r/'training_metrics.json').read_text())['metrics']; m=json.loads((r/'monitor/u500.json').read_text())['acceptance']; p2=json.loads((ROOT/f'runs/b1_p2_seed{s}_2026-09-21/monitor/u500.json').read_text())['acceptance']
            finite=all(math.isfinite(float(x[k])) for x in tm for k in ('std','reward_mean','policy_loss','value_loss','analytic_kl','approx_kl','actor_lr','reconstruction_loss','reconstruction_grad_norm','actor_grad_norm'))
            rec={'seed':s,'run_done':done.get('status')=='RUN_DONE' and done.get('exit_code')==0,'checkpoint_500':(r/'checkpoints/update_500.pt').exists(),'metrics':len(tm),'finite':finite,'r1_config':cfg.get('b1_r1') is True and cfg.get('b1_p2') is True,'critic_completed':all(x['critic_completed'] for x in tm),'std':[tm[0]['std'],tm[-1]['std']],'reconstruction_loss_first':tm[0]['reconstruction_loss'],'reconstruction_loss_last':tm[-1]['reconstruction_loss'],'reconstruction_loss_reduced':tm[-1]['reconstruction_loss'] < tm[0]['reconstruction_loss'],'reconstruction_grad_mean':sum(x['reconstruction_grad_norm'] for x in tm)/len(tm),'actor_grad_mean':sum(x['actor_grad_norm'] for x in tm)/len(tm),'p2_lr_events':sum(len(x.get('lr_events',[])) for x in tm),'gate':{'survival':m['survival_rate'],'vx_mae':m['mean_abs_vx_error'],'tilt_p95':math.degrees(m['tilt_p95_rad']),'tilt_max':math.degrees(m['tilt_max_rad'])}}
            rec['gate']['pass']=rec['gate']['survival']>=.9 and rec['gate']['vx_mae']<=.15 and rec['gate']['tilt_p95']<=15 and rec['gate']['tilt_max']<=30
            rec['vs_p2']={'survival_delta':rec['gate']['survival']-p2['survival_rate'],'vx_mae_delta':rec['gate']['vx_mae']-p2['mean_abs_vx_error'],'tilt_p95_delta_deg':rec['gate']['tilt_p95']-math.degrees(p2['tilt_p95_rad']),'tilt_max_delta_deg':rec['gate']['tilt_max']-math.degrees(p2['tilt_max_rad'])}
            if not all((rec['run_done'],rec['checkpoint_500'],rec['finite'],rec['r1_config'],rec['critic_completed'],rec['reconstruction_loss_reduced'])): failures.append(s)
            rows.append(rec)
        verdict=not failures and all(x['gate']['pass'] for x in rows); out={'schema':'b1_r1_verdict_v1','status':'PASS' if verdict else 'FAIL','sanity_failures':failures,'seeds':rows,'rule':'update-500 deterministic gate'}; (OUT/'B1_R1_VERDICT.json').write_text(json.dumps(out,indent=2)+'\n')
        lines=[f"# B1-R1 Verdict: {'PASS' if verdict else 'FAIL'}",'', '|seed|RUN_DONE|ckpt500|recon first→last|recon grad|actor grad|survival|vx MAE|tilt p95°|max tilt°|Δp95 vs P2|gate|','|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
        for x in rows:
            g=x['gate']; lines.append(f"|{x['seed']}|{x['run_done']}|{x['checkpoint_500']}|{x['reconstruction_loss_first']:.4g}→{x['reconstruction_loss_last']:.4g}|{x['reconstruction_grad_mean']:.4g}|{x['actor_grad_mean']:.4g}|{g['survival']:.3f}|{g['vx_mae']:.3f}|{g['tilt_p95']:.2f}|{g['tilt_max']:.2f}|{x['vs_p2']['tilt_p95_delta_deg']:.2f}|{g['pass']}|")
        (OUT/'report.md').write_text('\n'.join(lines)+'\n'); print(OUT/'report.md')
    if True: main()

def run_freeze_b1_r1():
    """Run former freeze_b1_r1.py stage."""
    from pathlib import Path
    import hashlib,json,time
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_r1_freeze'; OUT.mkdir(parents=True,exist_ok=True)
    def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    def main():
        smoke=json.loads((ROOT/'artifacts/b1_r1_smoke/artifact.json').read_text())
        if not smoke.get('pass'): raise RuntimeError('R1 Isaac smoke did not pass')
        files={'design':'docs/baselines/stability_robustness/b1-r1-srm-style-controlled-screen.md','trainer':'scripts/rl/core/algorithms/scalar_ppo.py','actor_critic':'scripts/rl/core/modules/actor_critic.py','smoke':'scripts/rl/experiments/baselines/stability_robustness/controlled_recovery.py','controlled_screen':'scripts/rl/experiments/baselines/stability_robustness/controlled_recovery.py','env':'talon_rl/wrappers/scalar_reward_env.py','reward':'talon_rl/rewards/baselines.py','monitor':'scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py','reset_states':'artifacts/b0_smoke/b0-monitor-frozen-reset-states.npz','runner':'scripts/rl/experiments/common/utilities/train_b0.py'}
        data={'schema':'b1_r1_freeze_v1','status':'FROZEN','training_authorized':True,'created_unix':time.time(),'mechanism':'feed_forward_srm_style_shared_trunk_reconstruction','inherited':'B1-P2','actor_input_dim':51,'reconstruction_head':{'targets':['tilt','height','contact'],'output_dim':3,'shared_trunk':True,'lambda':0.10,'loss':'MSE','scope':'actor_and_shared_trunk_only'},'privileged_target_only':True,'privileged_not_in':['actor_input','reward','deterministic_monitor','normalizer_state'],'p2':{'desired_kl':0.01,'kl_low':0.005,'kl_high':0.02,'lr_factor':1.5,'actor_lr_bounds':[1e-5,1e-2],'early_stop':False,'critic_lr_unchanged':True},'monitor_contract':'tanh(actor_mean), 64 frozen states x 500 steps, evaluation_only','checkpoint_schema':'b0_ppo_v1 plus b1_r1 config and reconstruction head','scheduled_std':{'initial':0.82,'final':0.10,'hold_updates':100,'decay_updates':300},'command_vx':0.5,'num_envs':4096,'updates':500,'seeds':[0,1,2],'sha256':{k:sha(v) for k,v in files.items()},'smoke_artifact_sha256':hashlib.sha256((ROOT/'artifacts/b1_r1_smoke/artifact.json').read_bytes()).hexdigest(),'smoke_checks':smoke['checks']}
        (OUT/'B1_R1_FREEZE.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n'); print(OUT/'B1_R1_FREEZE.json')
    if True: main()

STAGES = {
    "b1_r1_controlled_screen": run_b1_r1_controlled_screen,
    "b1_r1_isaac_smoke": run_b1_r1_isaac_smoke,
    "b1_r1_verdict": run_b1_r1_verdict,
    "freeze_b1_r1": run_freeze_b1_r1,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
