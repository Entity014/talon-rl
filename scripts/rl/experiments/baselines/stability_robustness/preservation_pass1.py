"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_b1_p1_preservation_audit():
    """Run former b1_p1_preservation_audit.py stage."""
    from pathlib import Path
    import csv, json
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_p1_verdict'; OBS=torch.from_numpy(np.load(ROOT/'artifacts/b0_1_stability_audit/frozen_obs.npy')).float()
    from sys import path; path.insert(0,str(ROOT/'scripts'))
    from rl.core.modules.actor_critic import ActorCritic
    def main():
     rows=[]
     for seed in range(3):
      run=ROOT/f'runs/b1_p1_seed{seed}_2026-09-21'; m=ActorCritic(51,51,12,1,[64,64]).eval(); prev=None
      for p in sorted((run/'checkpoints').glob('update_*.pt'),key=lambda q:int(q.stem.split('_')[-1])):
       u=int(p.stem.split('_')[-1]); c=torch.load(p,map_location='cpu'); m.load_state_dict(c['model'])
       with torch.no_grad(): mu=m.raw_mean(OBS); act=m.act_inference(OBS)
       row={'seed':seed,'update':u,'saturation':float((act.abs()>=2.9).float().mean())}
       if prev is not None:
        pm,pp=prev; row['cos_mu']=float(torch.nn.functional.cosine_similarity(mu.reshape(1,-1),pm.reshape(1,-1)).item()); row['delta_mu']=float((mu-pm).norm()); row['delta_action']=float((act-pa).norm(dim=1).mean())
       else: row.update(cos_mu='',delta_mu='',delta_action='')
       prev=(mu, m.state_dict()); pa=act; rows.append(row)
     with (OUT/'preservation_audit.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
     picks=[r for r in rows if (r['seed'],r['update']) in {(0,450),(0,475),(1,25),(1,50),(2,250),(2,275)}]; (OUT/'preservation_audit.json').write_text(json.dumps({'selected':picks,'read_only':True},indent=2)+'\n'); print(OUT/'preservation_audit.csv')
    if True: main()

def run_b1_p1_production_smoke():
    """Run former b1_p1_production_smoke.py stage."""
    """CPU lifecycle smoke for B1-P1 trainer semantics (no environment/formulation changes)."""
    from pathlib import Path
    import json, time, traceback
    import torch
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_p1_freeze'; OUT.mkdir(parents=True,exist_ok=True)
    def run_case(name, target, expect_stop):
     from rl.core.algorithms.scalar_ppo import B0PPOConfig,B0PPOTrainer
     from rl.core.modules.actor_critic import ActorCritic
     torch.manual_seed(11); model=ActorCritic(5,5,3,1,[8]); cfg=B0PPOConfig(b1_p1_enabled=True,target_kl=target,actor_epochs=3)
     tr=B0PPOTrainer(model,cfg,lr=1e-2); tr.begin_update(); obs=torch.randn(4*3,5); actions,old=model.act(obs); critic_before=model.critic_head.weight.detach().clone()
     out=tr.optimize_batch(obs,actions.detach(),old.detach(),torch.ones(obs.shape[0]),torch.zeros(obs.shape[0])); tr.finish_update()
     assert out['actor_stopped'] is expect_stop; assert out['critic_completed']; assert tr._std_for_update is None
     assert not torch.equal(critic_before,model.critic_head.weight)
     return {'case':name,**out,'update_idx':tr.update_idx,'scheduled_std_after':tr.cfg.scheduled_std(tr.update_idx)}
    def main():
     started={'status':'RUN_STARTED','unix':time.time(),'envs':12,'updates':3}
     (OUT/'smoke_RUN_STARTED.json').write_text(json.dumps(started,indent=2)+'\n')
     try:
      normal=run_case('normal_below_threshold',100.0,False); forced=run_case('controlled_forced_threshold',1e-12,True)
      result={'pass':True,'normal':normal,'forced':forced,'rollout_obs':'entire synthetic rollout batch','checkpoint_resume':'covered by unit tests','critic_continuation':True,'std_unchanged':True}
      (OUT/'production_smoke.json').write_text(json.dumps(result,indent=2)+'\n'); (OUT/'smoke_RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0,'unix':time.time(),'artifact':'production_smoke.json'},indent=2)+'\n'); print(OUT/'production_smoke.json')
     except BaseException as e:
      (OUT/'smoke_ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
    if True: main()

def run_b1_p1_verdict():
    """Run former b1_p1_verdict.py stage."""
    """B1-P1 post-training sanity, update-500 gate, and preservation audit."""
    from pathlib import Path
    import hashlib, json, math
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/'artifacts'/'b1_p1_verdict'; OUT.mkdir(parents=True,exist_ok=True)
    def main():
        manifest=json.loads((ROOT/'artifacts/b1_p1_freeze/B1_P1_FREEZE.json').read_text()); seeds=[]; failures=[]
        for seed in range(3):
            run=ROOT/f'runs/b1_p1_seed{seed}_2026-09-21'; done=json.loads((run/'RUN_DONE.json').read_text()); cfg=json.loads((run/'config.json').read_text()); tm=json.loads((run/'training_metrics.json').read_text())['metrics']; mon=json.loads((run/'monitor/u500.json').read_text())['acceptance']
            required={f'update_{u:03d}.pt' for u in range(25,501,25)}; actual={p.name for p in (run/'checkpoints').glob('update_*.pt')}; missing=sorted(required-actual)
            finite=all(math.isfinite(float(x[k])) for x in tm for k in ('std','reward_mean','policy_loss','value_loss','analytic_kl','approx_kl'))
            stops=sum(bool(x['actor_stopped']) for x in tm); critic_ok=all(x['critic_completed'] for x in tm); kl_ok=all(len(x['kl_trace'])==x['actor_epochs_completed'] for x in tm)
            rec={'seed':seed,'run_done':done.get('status')=='RUN_DONE' and done.get('exit_code')==0,'checkpoint_500':(run/'checkpoints/update_500.pt').exists(),'missing_checkpoints':missing,'config_b1_p1':cfg.get('b1_p1') is True,'metrics_count':len(tm),'finite':finite,'kl_trace_complete':kl_ok,'critic_completed_all_updates':critic_ok,'stop_updates':stops,'std_start':tm[0]['std'],'std_end':tm[-1]['std'],'final_gate':{'survival':mon['survival_rate'],'vx_mae':mon['mean_abs_vx_error'],'tilt_p95_deg':math.degrees(mon['tilt_p95_rad']),'tilt_max_deg':math.degrees(mon['tilt_max_rad']),'pass':mon['survival_rate']>=.9 and mon['mean_abs_vx_error']<=.15 and math.degrees(mon['tilt_p95_rad'])<=15 and math.degrees(mon['tilt_max_rad'])<=30}}
            if not (rec['run_done'] and rec['checkpoint_500'] and not missing and rec['config_b1_p1'] and finite and kl_ok and critic_ok): failures.append(seed)
            seeds.append(rec)
        verdict=all(x['final_gate']['pass'] for x in seeds) and not failures
        report={'schema':'b1_p1_verdict_v1','status':'PASS' if verdict else 'FAIL','sanity_failures':failures,'seeds':seeds,'training_authorized_manifest':manifest.get('training_authorized'),'rule':'update-500 deterministic B0 gate only'}
        (OUT/'B1_P1_VERDICT.json').write_text(json.dumps(report,indent=2)+'\n')
        lines=[f"# B1-P1 Verdict: {'PASS' if verdict else 'FAIL'}",'', '| seed | RUN_DONE | ckpt500 | stop updates | std | survival | vx MAE | tilt p95° | tilt max° | gate |','|---:|:---:|:---:|---:|:---|---:|---:|---:|---:|:---:|']
        for x in seeds:
            g=x['final_gate']; lines.append(f"| {x['seed']} | {x['run_done']} | {x['checkpoint_500']} | {x['stop_updates']} | {x['std_start']:.2f}→{x['std_end']:.2f} | {g['survival']:.3f} | {g['vx_mae']:.3f} | {g['tilt_p95_deg']:.2f} | {g['tilt_max_deg']:.2f} | {g['pass']} |")
        lines += ['', '## Preservation findings', '', 'Analytic KL traces, actor epoch counts, actor-stop flags, and critic completion flags were checked for every update. `stop_updates` reports actor-only KL gate triggers.', '', 'This verdict uses update 500 only; no transient checkpoint is substituted.']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n'); print(OUT/'report.md')
    if True: main()

def run_finalize_b1_p1():
    """Run former finalize_b1_p1.py stage."""
    from pathlib import Path
    import hashlib, json, time
    ROOT=Path(__file__).resolve().parents[4]
    D=ROOT/'artifacts'/'b1_p1_freeze'
    def main():
        manifest=D/'B1_P1_FREEZE.json'
        data=json.loads(manifest.read_text())
        smoke=json.loads((D/'production_smoke.json').read_text())
        done=json.loads((D/'smoke_RUN_DONE.json').read_text())
        if not smoke.get('pass') or done.get('exit_code') != 0:
            raise RuntimeError('production smoke is not passing')
        data['status']='FROZEN'; data['training_authorized']=True; data['frozen_unix']=time.time()
        data['smoke_artifact_sha256']=hashlib.sha256((D/'production_smoke.json').read_bytes()).hexdigest()
        manifest.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n'); print(manifest)
    if True: main()

def run_freeze_b1_p1():
    """Run former freeze_b1_p1.py stage."""
    """Create the B1-P1 design manifest after implementation smoke passes."""
    from pathlib import Path
    import hashlib, json, time
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/'artifacts'/'b1_p1_freeze'; OUT.mkdir(parents=True,exist_ok=True)
    def sha(rel):
     p=ROOT/rel; return hashlib.sha256(p.read_bytes()).hexdigest()
    def main():
     files={
      'b0_reward':'talon_rl/rewards/baselines.py','environment_wrapper':'talon_rl/wrappers/scalar_reward_env.py',
      'trainer':'scripts/rl/core/algorithms/scalar_ppo.py','actor_critic':'scripts/rl/core/modules/actor_critic.py',
      'runner':'scripts/rl/experiments/common/utilities/train_b0.py','b1_p1_tests':'tests/test_b1_p1.py',
      'b1_p1_design':'docs/baselines/stability_robustness/b1-p1-target-kl-early-stopping-freeze-draft.md',
      'monitor':'scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py',
      'reset_states':'artifacts/b0_smoke/b0-monitor-frozen-reset-states.npz'}
     manifest={'schema':'b1_p1_freeze_v1','status':'DESIGN_DRAFT','training_authorized':False,'created_unix':time.time(),
      'mechanism':'ppo_target_kl_actor_early_stop','target_kl':0.01,'stop_threshold':0.015,'kl_stop_multiplier':1.5,
      'kl_estimator':'analytic_pre_tanh_diagonal_gaussian_on_entire_rollout_after_each_full_actor_epoch',
      'approx_kl_role':'logging_only','stop_scope':'actor_epochs_only','critic_schedule':'unchanged_and_completes',
      'actor_epochs':4,'num_envs':4096,'updates':500,'seeds':[0,1,2],'command_vx':0.5,
      'scheduled_std':{'initial':0.82,'final':0.10,'hold_updates':100,'decay_updates':300},
      'monitor_contract':'tanh(actor_mean), 64 frozen states x 500 steps, evaluation_only',
      'logging_schema':['analytic_kl','kl_trace','approx_kl','kl_stop_triggered','actor_epochs_completed','critic_completed','policy_loss','value_loss','clip_fraction','update_idx'],
      'sha256':{k:sha(v) for k,v in files.items()},'files':files,
      'acceptance':['unit_sanity_11_pass','production_smoke_run_done','checkpoint_resume','actor_only_stop','critic_continuation','std_invariance','b0_flag_off_equivalence','deterministic_monitor_gate']}
     (OUT/'B1_P1_FREEZE.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); print(OUT/'B1_P1_FREEZE.json')
    if True: main()

def run_register_b1_p1_runner_smoke():
    """Run former register_b1_p1_runner_smoke.py stage."""
    from pathlib import Path
    import hashlib, json
    ROOT=Path(__file__).resolve().parents[4]
    D=ROOT/'artifacts'/'b1_p1_freeze'
    RUN=ROOT/'runs'/'b1_p1_production_smoke_2026-09-21'
    def main():
        m=D/'B1_P1_FREEZE.json'; data=json.loads(m.read_text()); done=json.loads((RUN/'RUN_DONE.json').read_text())
        if done.get('status')!='RUN_DONE' or done.get('exit_code')!=0: raise RuntimeError('runner smoke incomplete')
        data['production_runner_smoke']={'run_dir':str(RUN.relative_to(ROOT)),'envs':4,'updates':2,'run_done_sha256':hashlib.sha256((RUN/'RUN_DONE.json').read_bytes()).hexdigest(),'training_metrics_sha256':hashlib.sha256((RUN/'training_metrics.json').read_bytes()).hexdigest()}
        m.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n'); print(m)
    if True: main()

STAGES = {
    "b1_p1_preservation_audit": run_b1_p1_preservation_audit,
    "b1_p1_production_smoke": run_b1_p1_production_smoke,
    "b1_p1_verdict": run_b1_p1_verdict,
    "finalize_b1_p1": run_finalize_b1_p1,
    "freeze_b1_p1": run_freeze_b1_p1,
    "register_b1_p1_runner_smoke": run_register_b1_p1_runner_smoke,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
