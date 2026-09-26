"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_separated_anchor_policy_aggregate():
    """Run former separated_anchor_policy_aggregate.py stage."""
    from pathlib import Path
    import json,hashlib,numpy as np
    ROOT=Path(__file__).resolve().parents[4];RUN=ROOT/'runs/separated_anchor_policy_control-2026-09-24';SEEDS=(986001,987001,988001);AXES=('T','A','O','S')
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
     reps=[json.load(open(RUN/f'seed_{s}/evaluation_report.json')) for s in SEEDS];pairs=[]
     for d in reps:
      for a in AXES:
       x=d['axis_stats'][a]
       if x['first_pass_checkpoint'] is not None:pairs.append({'seed':d['train_seed'],'axis':a,**x})
     v=json.load(open(ROOT/'runs/prospective_slope_validation-2026-09-24/prospective_slope_validation_report.json'));shared=[]
     for sd,arr in v['states'].items():
      for a in AXES:
       ps=[x['axes'][a]['PASS'] for x in arr];gs=np.array([x['axes'][a]['G_sem'] for x in arr],float);fp=next((i for i,z in enumerate(ps) if z),None)
       if fp is not None:
        q=gs[fp:];shared.append({'seed':int(sd),'axis':a,'first_pass':fp,'max_gate_margin_forgetting':float(np.max(np.maximum.accumulate(q)-q))})
     post=sum(8-x['first_pass_checkpoint'] for x in pairs);ptf=sum(x['pass_to_fail_events'] for x in pairs);ret=float(np.median([x['retained_pass_fraction_after_first_pass'] for x in pairs]));af=float(np.median([x['max_gate_margin_forgetting'] for x in pairs]));sf=float(np.median([x['max_gate_margin_forgetting'] for x in shared]))
     acquired={str(s):sum(reps[i]['axis_stats'][a]['first_pass_checkpoint'] is not None for a in AXES) for i,s in enumerate(SEEDS)};final={str(s):reps[i]['family_stats']['final_N_pass'] for i,s in enumerate(SEEDS)};drops={str(s):reps[i]['family_stats']['max_drop_from_running_max'] for i,s in enumerate(SEEDS)}
     criteria={'acquire_ge3of4_every_seed':all(v>=3 for v in acquired.values()),'median_retention_ge0p75':ret>=.75,'final_Npass_ge3_every_seed':all(v>=3 for v in final.values()),'forgetting_reduction_ge25pct':af<=.75*sf,'pass_to_fail_rate_le0p25':ptf/max(post,1)<=.25,'no_drop_ge2_every_seed':all(v<2 for v in drops.values())}
     if criteria['acquire_ge3of4_every_seed'] and not all([criteria['median_retention_ge0p75'],criteria['final_Npass_ge3_every_seed'],criteria['pass_to_fail_rate_le0p25']]):status='SEPARATED ANCHORS STILL SEMANTICALLY UNSTABLE'
     elif sum(v<3 for v in acquired.values())>=2:status='ANCHOR OBJECTIVES THEMSELVES INSUFFICIENT'
     elif all(criteria.values()):status='SEPARATION SUPPORTS SHARED-POLICY BOTTLENECK'
     else:status='INCONCLUSIVE'
     out={'schema':'separated_anchor_policy_control_v1','status':status,'training_seeds':list(SEEDS),'acquired_axis_seed_pairs':len(pairs),'total_axis_seed_pairs':12,'acquired_per_seed':acquired,'median_retained_pass_fraction':ret,'pass_to_fail_events':ptf,'post_acquisition_transitions':post,'pass_to_fail_rate':ptf/max(post,1),'final_N_pass_per_seed':final,'max_drop_from_running_max_per_seed':drops,'anchor_median_max_gate_margin_forgetting':af,'shared_v18_median_max_gate_margin_forgetting':sf,'forgetting_ratio_anchor_over_shared':af/(sf+1e-12),'criteria':criteria,'axis_seed_stats':pairs,'shared_v18_acquired_stats':shared,'decision':{'continuous_policy_family_as_forgetting_fix_authorized':status=='SEPARATION SUPPORTS SHARED-POLICY BOTTLENECK','new_training_method_authorized':False}}
     p=RUN/'separated_anchor_policy_control_report.json';p.write_text(json.dumps(out,indent=2)+'\n');(RUN/'FINAL_SEPARATED_ANCHOR_CONTROL_PROVENANCE.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':status,'report_sha256':sha(p),'contract_sha256':sha(ROOT/'docs/contracts/preference_control/separated-anchor-policy-control-contract.md'),'verdict_sha256':sha(ROOT/'docs/verdicts/preference_control/separated-anchor-policy-control-verdict.md'),'training_script_sha256':sha(ROOT/'scripts/rl/experiments/architectures/preference_control/separated_anchor/workflow.py'),'evaluation_script_sha256':sha(ROOT/'scripts/rl/experiments/architectures/preference_control/separated_anchor/workflow.py'),'aggregate_script_sha256':sha(Path(__file__).resolve())},indent=2)+'\n');print(json.dumps(out,indent=2))
    if True:main()

def run_separated_anchor_policy_eval():
    """Run former separated_anchor_policy_eval.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    RUN=ROOT/'runs/separated_anchor_policy_control-2026-09-24';ORDER=('T','A','O','S');ALL=('T','A','O','S','C');NENV=8;STEPS=64;SUITES=4
    PREFS={'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32),'C':np.array([.25,.25,.25,.25],np.float32)}
    IDX={'T':0,'A':1,'O':2,'S':3};PHYS={'T':'tracking_error','A':'ang_vel_xy','O':'tilt_deg','S':'action_rate'}
    SCALES=json.load(open(ROOT/'runs/semantic_gate_robustness_audit-2026-09-24/semantic_gate_robustness_report.json'))['scale']
    
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
     _,x,y,_=[q[:,i] for i in range(4)];return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def gate_margin(vals):return float(np.sort(np.asarray(vals,float))[1])
    def rollout(env,m,mgr,robot,w_np,seed):
     from talon_rl.rewards.objectives import normalized_objective_vector
     w=torch.tensor(w_np,device='cuda').repeat(NENV,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device='cuda')
     with torch.no_grad():
      for _ in range(STEPS):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
       data=robot.data;cmd=env.unwrapped.command_manager.get_command('base_velocity');phys.append({'tracking_error':float(((data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()).mean()),'ang_vel_xy':float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),'tilt_deg':float(tilt_deg(data.root_quat_w).mean()),'action_rate':float(torch.linalg.vector_norm(a-prev,dim=-1).mean())});done_any|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     R=np.asarray(R);return {'objective_mean':R.mean((0,1)).tolist(),'physical':{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},'survival':float(1-done_any.mean())}
    def main():
     ap=argparse.ArgumentParser();ap.add_argument('--train-seed',type=int,required=True);args=ap.parse_args();out=RUN/f'seed_{args.train_seed}'
     from isaaclab.app import AppLauncher
     saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd');env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene['robot']
      timeline=[]
      for u in range(9):
       models={}
       for lab in ALL:
        m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(out/f'{lab}_u{u}.pt',map_location='cuda',weights_only=False)['model']);m.eval();models[lab]=m
       raw=[]
       for suite in range(SUITES):
        sd=840001+suite
        for lab in ALL:
         raw.append({'suite':suite,'policy':lab,**rollout(env,models[lab],mgr,robot,PREFS[lab],sd)})
       axes={}
       for lab in ORDER:
        j=IDX[lab];pk=PHYS[lab];om=[];pm=[];surv=[]
        for suite in range(SUITES):
         r=next(x for x in raw if x['suite']==suite and x['policy']==lab);c=next(x for x in raw if x['suite']==suite and x['policy']=='C');om.append(r['objective_mean'][j]-c['objective_mean'][j]);pm.append(c['physical'][pk]-r['physical'][pk]);surv.extend([r['survival'],c['survival']])
        zo=np.asarray(om)/SCALES[lab]['obj'];zp=np.asarray(pm)/SCALES[lab]['phys'];Gobj=gate_margin(zo);Gphys=gate_margin(zp);G=min(Gobj,Gphys);of=float(np.mean(np.asarray(om)>0));pf=float(np.mean(np.asarray(pm)>0));passed=bool(of>=.75 and pf>=.75 and min(surv)>=.95)
        axes[lab]={'obj_margins':om,'phys_margins':pm,'obj_correct_fraction':of,'phys_correct_fraction':pf,'G_obj':Gobj,'G_phys':Gphys,'G_sem':G,'pass':passed,'min_survival':float(min(surv))}
       timeline.append({'update':u,'N_pass':sum(axes[a]['pass'] for a in ORDER),'axes':axes,'raw':raw});print('EVAL',args.train_seed,'u',u,'N_pass',timeline[-1]['N_pass'],flush=True)
      stats={}
      for lab in ORDER:
       passes=[r['axes'][lab]['pass'] for r in timeline];gs=[r['axes'][lab]['G_sem'] for r in timeline];fp=next((i for i,v in enumerate(passes) if v),None);ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])));best=np.maximum.accumulate(gs);forget=[float(best[i]-gs[i]) for i in range(len(gs))]
       stats[lab]={'first_pass_checkpoint':fp,'final_pass':bool(passes[-1]),'retained_pass_fraction_after_first_pass':ret,'max_gate_margin_forgetting':float(max(forget)),'pass_to_fail_events':int(sum(passes[i] and not passes[i+1] for i in range(len(passes)-1))),'max_consecutive_pass_after_first':0 if fp is None else max([len(list(g)) for val,g in __import__('itertools').groupby(passes[fp:]) if val] or [0]),'best_G_sem':float(max(gs)),'final_G_sem':float(gs[-1])}
      npass=[r['N_pass'] for r in timeline];maxdrop=max((max(npass[:i+1])-npass[i] for i in range(len(npass))),default=0)
      report={'schema':'separated_anchor_policy_eval_v1','train_seed':args.train_seed,'timeline':timeline,'axis_stats':stats,'family_stats':{'max_N_pass':int(max(npass)),'final_N_pass':int(npass[-1]),'N_pass_timeline':npass,'max_drop_from_running_max':int(maxdrop),'repeated_winner_rotation_ge2':bool(maxdrop>=2)}}
      rp=out/'evaluation_report.json';rp.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'seed':args.train_seed,'axis_stats':stats,'family_stats':report['family_stats']},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_separated_anchor_policy_train():
    """Run former separated_anchor_policy_train.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,hashlib,json,sys
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    import rl.experiments.common.utilities.v2b_bounded_deltaa_multiseed as core
    core.LAM=0.95
    ORDER=('T','A','O','S','C');K=8
    OUT=ROOT/'runs/separated_anchor_policy_control-2026-09-24'
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
     ap=argparse.ArgumentParser();ap.add_argument('--train-seed',type=int,required=True);args=ap.parse_args();out=OUT/f'seed_{args.train_seed}';out.mkdir(parents=True,exist_ok=True)
     from isaaclab.app import AppLauncher
     saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=core.NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
      env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);o,_=env.reset(seed=0);o=core.ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      base_state=torch.load(core.CKPT,map_location='cuda',weights_only=False)['model'];models={}
      for pi,lab in enumerate(ORDER):
       m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(base_state);m.eval();models[lab]=m
       torch.save({'model':m.state_dict(),'anchor':lab,'update':0,'train_seed':args.train_seed,'gae_lambda':0.95,'fixed_preference':core.PREFS[lab].tolist()},out/f'{lab}_u0.pt')
      step_norm=core.STEP_SCALE*core.nominal_step();rows=[]
      for k in range(1,K+1):
       for pi,lab in enumerate(ORDER):
        m=models[lab];batch_seed=args.train_seed+k*313+pi*10007;w=torch.tensor(core.PREFS[lab],device='cuda').repeat(core.NENV,1)
        b=core.collect_batch(env,m,mgr,batch_seed,w);g,meta=core.mixed_gradient(m,b);d=-g/(g.norm()+core.EPS);core.apply_flat_step(m,d,step_norm)
        ck=out/f'{lab}_u{k}.pt';torch.save({'model':m.state_dict(),'anchor':lab,'update':k,'train_seed':args.train_seed,'gae_lambda':0.95,'fixed_preference':core.PREFS[lab].tolist()},ck)
        rows.append({'update':k,'anchor':lab,'batch_seed':batch_seed,'gradient_norm':float(g.norm().cpu()),'ratio_maxerr':meta['ratio_maxerr'],'step_norm':step_norm,'checkpoint':str(ck.relative_to(ROOT))})
       (out/'partial.json').write_text(json.dumps({'train_seed':args.train_seed,'gae_lambda':0.95,'rows':rows},indent=2)+'\n')
      report={'schema':'separated_anchor_policy_training_v1','train_seed':args.train_seed,'base_checkpoint':str(core.CKPT.relative_to(ROOT)),'anchors':list(ORDER),'fixed_preferences':{k:core.PREFS[k].tolist() for k in ORDER},'gae_lambda':0.95,'updates':K,'step_norm':step_norm,'rows':rows}
      rp=out/'training_report.json';rp.write_text(json.dumps(report,indent=2)+'\n');(out/'TRAINING_PROVENANCE.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','report_sha256':sha(rp),'script_sha256':sha(Path(__file__).resolve()),'base_checkpoint_sha256':sha(core.CKPT)},indent=2)+'\n')
      print(json.dumps({'train_seed':args.train_seed,'anchors':list(ORDER),'updates':K,'step_norm':step_norm},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

STAGES = {
    "separated_anchor_policy_aggregate": run_separated_anchor_policy_aggregate,
    "separated_anchor_policy_eval": run_separated_anchor_policy_eval,
    "separated_anchor_policy_train": run_separated_anchor_policy_train,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
