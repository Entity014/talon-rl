"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_b1_p2_production_smoke():
    """Run former b1_p2_production_smoke.py stage."""
    import json,time,traceback
    from pathlib import Path
    import torch
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_p2_freeze'; OUT.mkdir(parents=True,exist_ok=True)
    def case(name,low,high,expected):
     from rl.core.algorithms.scalar_ppo import B0PPOConfig,B0PPOTrainer
     from rl.core.modules.actor_critic import ActorCritic
     torch.manual_seed(9); cfg=B0PPOConfig(b1_p2_enabled=True,b1_p1_enabled=False,actor_epochs=3,kl_low=low,kl_high=high); m=ActorCritic(5,5,3,1,[8]); tr=B0PPOTrainer(m,cfg,lr=3e-4); tr.begin_update(); std=tr._std_for_update; obs=torch.randn(12,5); act,old=m.act(obs); critic_lr=tr.critic_optim.param_groups[0]['lr']; out=tr.optimize_batch(obs,act.detach(),old.detach(),torch.ones(12),torch.zeros(12));
     assert out['actor_stopped'] is False and out['actor_epochs_completed']==3 and out['critic_completed']; assert tr.critic_optim.param_groups[0]['lr']==critic_lr; assert tr._std_for_update==std; assert all(e['event']==expected for e in out['lr_events']); return {'name':name,**out,'critic_lr':critic_lr,'p1_enabled':cfg.b1_p1_enabled,'std':std}
    def main():
     (OUT/'smoke_RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','unix':time.time()},indent=2)+'\n')
     try:
      results=[case('low_kl_lr_up',100.0,200.0,'up'),case('mid_kl_lr_hold',-100.0,100.0,'hold'),case('high_kl_lr_down',-100.0,1e-12,'down')]
      from rl.core.algorithms.scalar_ppo import B0PPOConfig,B0PPOTrainer
      from rl.core.modules.actor_critic import ActorCritic
      tr=B0PPOTrainer(ActorCritic(3,3,2,1,[4]),B0PPOConfig(b1_p2_enabled=True),lr=3e-4); tr.begin_update(); p=OUT/'resume.pt'; tr.save(p); r=B0PPOTrainer(ActorCritic(3,3,2,1,[4]),B0PPOConfig(b1_p2_enabled=True)); r.load(p); assert r.actor_optim.param_groups[0]['lr']==tr.actor_optim.param_groups[0]['lr']
      (OUT/'production_smoke.json').write_text(json.dumps({'pass':True,'cases':results,'resume_actor_lr':r.actor_optim.param_groups[0]['lr'],'p1_logic_off':True},indent=2)+'\n'); (OUT/'smoke_RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0,'unix':time.time()},indent=2)+'\n'); print(OUT/'production_smoke.json')
     except BaseException as e:
      (OUT/'smoke_ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
    if True: main()

def run_b1_p2_verdict():
    """Run former b1_p2_verdict.py stage."""
    from pathlib import Path
    import json,math
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/'artifacts'/'b1_p2_verdict'; OUT.mkdir(parents=True,exist_ok=True)
    def main():
        rows=[]; failures=[]
        for s in range(3):
            r=ROOT/f'runs/b1_p2_seed{s}_2026-09-21'; done=json.loads((r/'RUN_DONE.json').read_text()); cfg=json.loads((r/'config.json').read_text()); tm=json.loads((r/'training_metrics.json').read_text())['metrics']; m=json.loads((r/'monitor/u500.json').read_text())['acceptance']; ck=(r/'checkpoints/update_500.pt').exists()
            finite=all(math.isfinite(float(x[k])) for x in tm for k in ('std','reward_mean','policy_loss','value_loss','analytic_kl','approx_kl','actor_lr')); events=[e for x in tm for e in x.get('lr_events',[])]
            rec={'seed':s,'run_done':done.get('status')=='RUN_DONE' and done.get('exit_code')==0,'checkpoint_500':ck,'metrics':len(tm),'finite':finite,'p2_config':cfg.get('b1_p2') is True,'std':[tm[0]['std'],tm[-1]['std']],'lr_initial':events[0]['lr_before'] if events else None,'lr_final':tm[-1]['actor_lr'],'lr_up':sum(e['event']=='up' for e in events),'lr_down':sum(e['event']=='down' for e in events),'lr_hold':sum(e['event']=='hold' for e in events),'actor_epochs_all':all(x['actor_epochs_completed']==4 and not x['actor_stopped'] for x in tm),'critic_completed_all':all(x['critic_completed'] for x in tm),'gate':{'survival':m['survival_rate'],'vx_mae':m['mean_abs_vx_error'],'tilt_p95':math.degrees(m['tilt_p95_rad']),'tilt_max':math.degrees(m['tilt_max_rad'])}}
            rec['gate']['pass']=rec['gate']['survival']>=.9 and rec['gate']['vx_mae']<=.15 and rec['gate']['tilt_p95']<=15 and rec['gate']['tilt_max']<=30
            if not all((rec['run_done'],rec['checkpoint_500'],rec['finite'],rec['p2_config'],rec['actor_epochs_all'],rec['critic_completed_all'])): failures.append(s)
            rows.append(rec)
        verdict=not failures and all(x['gate']['pass'] for x in rows); out={'schema':'b1_p2_verdict_v1','status':'PASS' if verdict else 'FAIL','sanity_failures':failures,'seeds':rows,'rule':'update-500 deterministic B0 gate'}; (OUT/'B1_P2_VERDICT.json').write_text(json.dumps(out,indent=2)+'\n')
        lines=[f"# B1-P2 Verdict: {'PASS' if verdict else 'FAIL'}",'', '|seed|RUN_DONE|ckpt500|LR up|LR down|LR final|survival|vx MAE|tilt p95°|max tilt°|gate|','|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
        for x in rows:
            g=x['gate']; lines.append(f"|{x['seed']}|{x['run_done']}|{x['checkpoint_500']}|{x['lr_up']}|{x['lr_down']}|{x['lr_final']:.6g}|{g['survival']:.3f}|{g['vx_mae']:.3f}|{g['tilt_p95']:.2f}|{g['tilt_max']:.2f}|{g['pass']}|")
        (OUT/'report.md').write_text('\n'.join(lines)+'\n'); print(OUT/'report.md')
    if True: main()

def run_finalize_b1_p2():
    """Run former finalize_b1_p2.py stage."""
    from pathlib import Path
    import hashlib,json,time
    ROOT=Path(__file__).resolve().parents[4]; D=ROOT/'artifacts'/'b1_p2_freeze'
    def main():
     m=D/'B1_P2_FREEZE.json'; data=json.loads(m.read_text()); smoke=json.loads((D/'production_smoke.json').read_text()); done=json.loads((D/'smoke_RUN_DONE.json').read_text())
     if not smoke.get('pass') or done.get('exit_code')!=0: raise RuntimeError('P2 smoke failed')
     data.update(status='FROZEN',training_authorized=True,frozen_unix=time.time(),smoke_artifact_sha256=hashlib.sha256((D/'production_smoke.json').read_bytes()).hexdigest())
     m.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n'); print(m)
    if True: main()

def run_freeze_b1_p2():
    """Run former freeze_b1_p2.py stage."""
    from pathlib import Path
    import hashlib,json,time
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/'artifacts'/'b1_p2_freeze'; OUT.mkdir(parents=True,exist_ok=True)
    def main():
        files={'reward':'talon_rl/rewards/baselines.py','env':'talon_rl/wrappers/scalar_reward_env.py','trainer':'scripts/rl/core/algorithms/scalar_ppo.py','runner':'scripts/rl/experiments/common/utilities/train_b0.py','design':'docs/baselines/stability_robustness/b1-p2-adaptive-kl-lr-design-draft.md','monitor':'scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py','reset_states':'artifacts/b0_smoke/b0-monitor-frozen-reset-states.npz'}
        sha={k:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in files.items()}
        data={'schema':'b1_p2_freeze_v1','status':'DESIGN_DRAFT','training_authorized':False,'created_unix':time.time(),'mechanism':'co_rl_inspired_adaptive_kl_actor_lr','not_byte_identical_co_rl':True,'desired_kl':0.01,'kl_low':0.005,'kl_high':0.02,'lr_factor':1.5,'actor_lr_bounds':[1e-5,1e-2],'actor_critic_optimizer_separation':True,'lr_timing':'after_full_epoch_for_next_epoch_or_next_update','p1_early_stop':False,'critic_lr_unchanged':True,'actor_epochs':4,'num_envs':4096,'updates':500,'seeds':[0,1,2],'command_vx':0.5,'scheduled_std':{'initial':0.82,'final':0.10,'hold_updates':100,'decay_updates':300},'monitor_contract':'tanh(actor_mean), 64 frozen states x 500 steps, evaluation_only','logging_schema':['analytic_kl','approx_kl','actor_lr_before','actor_lr_after','lr_event','ratio_mean','clip_fraction','actor_grad_norm','policy_loss','value_loss','scheduled_std','update_idx'],'sha256':sha,'files':files}
        (OUT/'B1_P2_FREEZE.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n'); print(OUT/'B1_P2_FREEZE.json')
    if True: main()

STAGES = {
    "b1_p2_production_smoke": run_b1_p2_production_smoke,
    "b1_p2_verdict": run_b1_p2_verdict,
    "finalize_b1_p2": run_finalize_b1_p2,
    "freeze_b1_p2": run_freeze_b1_p2,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
