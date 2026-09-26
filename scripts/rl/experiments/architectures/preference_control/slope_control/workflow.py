"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_prospective_slope_control_seed():
    """Run former prospective_slope_control_seed.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,copy,hashlib,json,sys
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    import rl.experiments.common.utilities.v2b_bounded_deltaa_multiseed as core
    
    # Freeze final-reference training semantics.
    core.LAM=0.95
    K=8
    OUT=ROOT/'runs/prospective_slope_validation-2026-09-24'
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--train-seed',type=int,required=True);args=ap.parse_args()
        out=OUT/f'seed_{args.train_seed}';out.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=core.NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
            o,_=env.reset(seed=0);o=core.ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(core.CKPT,map_location='cuda',weights_only=False)['model']);m.eval()
            step_norm=core.STEP_SCALE*core.nominal_step();rows=[]
            for k in range(1,K+1):
                seed=args.train_seed+k*313
                _,w=core.mixed_w(torch.device('cuda'),k)
                b=core.collect_batch(env,m,mgr,seed,w)
                gm,meta=core.mixed_gradient(m,b)
                d=-gm/(gm.norm()+core.EPS)
                core.apply_flat_step(m,d,step_norm)
                ck=out/f'control_u{k}.pt';torch.save({'model':m.state_dict(),'arm':'control','update':k,'train_seed':args.train_seed,'gae_lambda':0.95},ck)
                rows.append({'update':k,'batch_seed':seed,'mixed_grad_norm':float(gm.norm().cpu()),'ratio_maxerr':meta['ratio_maxerr'],'step_norm':step_norm,'checkpoint':str(ck.relative_to(ROOT))})
                (out/'partial.json').write_text(json.dumps({'train_seed':args.train_seed,'gae_lambda':0.95,'rows':rows},indent=2)+'\n')
            report={'schema':'prospective_slope_control_seed_v1','train_seed':args.train_seed,'base_checkpoint':str(core.CKPT.relative_to(ROOT)),'gae_lambda':0.95,'updates':K,'step_norm':step_norm,'rows':rows}
            rp=out/'control_training_report.json';rp.write_text(json.dumps(report,indent=2)+'\n')
            (out/'TRAINING_PROVENANCE.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','report_sha256':sha(rp),'script_sha256':sha(Path(__file__).resolve()),'base_checkpoint_sha256':sha(core.CKPT)},indent=2)+'\n')
            print(json.dumps({'train_seed':args.train_seed,'gae_lambda':0.95,'updates':K,'step_norm':step_norm,'last_checkpoint':rows[-1]['checkpoint']},indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True:main()

def run_prospective_slope_measure_checkpoint():
    """Run former prospective_slope_measure_checkpoint.py stage."""
    from pathlib import Path
    import os,sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    OUT=ROOT/'runs/prospective_slope_validation-2026-09-24';CACHE=OUT/'raw_cache'
    SEEDS=(983001,984001,985001);BASE=ROOT/'runs/v2b_adam_continuous-2026-09-24/model_75.pt'
    ORDER=('T','A','O','S');IDX={'T':0,'A':1,'O':2,'S':3};PHYS={'T':'tracking_error','A':'ang_vel_xy','O':'tilt_deg','S':'action_rate'}
    PREFS={'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32),'C':np.array([.25,.25,.25,.25],np.float32)}
    NENV=8;STEPS=64;SUITES=4
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def checkpoint_model_hash(path):
        d=torch.load(path,map_location='cpu',weights_only=False)['model'];h=hashlib.sha256()
        for k in sorted(d):h.update(k.encode());h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def refs():
        out=[('base',BASE)]
        for s in SEEDS:
            for u in range(1,9): out.append((f'{s}_u{u}',OUT/f'seed_{s}/control_u{u}.pt'))
        return out
    
    def trace(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device='cuda').repeat(NENV,1); cur,_=env.reset(seed=seed); cur=obs_tensor(cur).cuda()
        obj=[];phys=[];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device='cuda')
        with torch.no_grad():
            for step in range(STEPS):
                a=m.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                ov=normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt
                obj.append(np.mean(ov,axis=0));data=robot.data;cmd=env.unwrapped.command_manager.get_command('base_velocity')
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean());wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({'tracking_error':vx+wz,'ang_vel_xy':float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),'tilt_deg':float(tilt_deg(data.root_quat_w).mean()),'action_rate':float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=obs_tensor(nxt).cuda()
                
        return {'obj':np.asarray(obj,float),'phys':{k:np.asarray([x[k] for x in phys],float) for k in phys[0]}}
    
    def main():
        shard=int(os.environ['SLOPE_SHARD_ID']);lab,p=refs()[shard];hh=checkpoint_model_hash(p);CACHE.mkdir(parents=True,exist_ok=True);cp=CACHE/f'{hh}.npz'
        if cp.exists():print('CACHE_EXISTS',cp);return
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene['robot'];m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);m.eval()
            obj=np.zeros((4,SUITES,2,STEPS,4),np.float64);phys=np.zeros((4,SUITES,2,STEPS,4),np.float64)
            for ai,a in enumerate(ORDER):
                for si in range(SUITES):
                    sd=840001+si
                    for qi,plab in enumerate(('C',a)):
                        t=trace(env,m,mgr,robot,PREFS[plab],sd);obj[ai,si,qi]=t['obj']
                        for pk,a2 in enumerate(ORDER): phys[ai,si,qi,:,pk]=t['phys'][PHYS[a2]]
            tmp=cp.with_suffix('.tmp.npz');np.savez_compressed(tmp,obj=obj,phys=phys,checkpoint_hash=np.array(hh),checkpoint_path=np.array(str(p.relative_to(ROOT))),label=np.array(lab),reset_seeds=np.arange(840001,840005));tmp.replace(cp);print('RAW_CACHE_WRITTEN',cp.name,lab)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_prospective_slope_offline_validation():
    """Run former prospective_slope_offline_validation.py stage."""
    from pathlib import Path
    import itertools,json,hashlib
    import numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    RUN=ROOT/'runs/prospective_slope_validation-2026-09-24';CACHE=RUN/'raw_cache'
    ROB=ROOT/'runs/semantic_gate_robustness_audit-2026-09-24/semantic_gate_robustness_report.json'
    SEEDS=(983001,984001,985001);AXES=('T','A','O','S');TOL=1e-10
    BASE='runs/v2b_adam_continuous-2026-09-24/model_75.pt'
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def slope(x):
        x=np.asarray(x,float);t=np.arange(len(x),dtype=float);return float(np.polyfit(t,x,1)[0])
    def gate_margin(vals):return float(np.sort(np.asarray(vals,float))[1])
    def pass_tau(zo,zp,tau):return bool(np.sum(np.asarray(zo)>tau)>=3 and np.sum(np.asarray(zp)>tau)>=3)
    def auc(y,score):
        y=np.asarray(y,int);score=np.asarray(score,float);P=np.where(y==1)[0];N=np.where(y==0)[0]
        if len(P)==0 or len(N)==0:return float('nan')
        v=0.0
        for i in P:
          for j in N:v+=1.0 if score[i]>score[j] else .5 if score[i]==score[j] else 0.0
        return float(v/(len(P)*len(N)))
    
    def main():
        scales=json.load(open(ROB))['scale']
        bypath={}
        for p in CACHE.glob('*.npz'):
            z=np.load(p,allow_pickle=False);bypath[str(z['checkpoint_path'])]=p
        def path(seed,label):
            if label=='base':return BASE
            return f'runs/prospective_slope_validation-2026-09-24/seed_{seed}/control_{label}.pt'
        states={}
        for sd in SEEDS:
          arr=[]
          for label in ['base']+[f'u{i}' for i in range(1,9)]:
            z=np.load(bypath[path(sd,label)],allow_pickle=False);axes={}
            for ai,a in enumerate(AXES):
                # q index 0=center, 1=heavy. Arrays [axis,suite,q,time,signal]
                obj_rel=z['obj'][ai,:,1,:,ai]-z['obj'][ai,:,0,:,ai]
                phys_rel=z['phys'][ai,:,0,:,ai]-z['phys'][ai,:,1,:,ai]
                obj_seq=obj_rel.mean(0);phys_seq=phys_rel.mean(0)
                obj_suite=obj_rel.mean(1);phys_suite=phys_rel.mean(1)
                zo=obj_suite/scales[a]['obj'];zp=phys_suite/scales[a]['phys']
                go=gate_margin(zo);gp=gate_margin(zp);gs=min(go,gp)
                mo=float(obj_rel.mean());mp=float(phys_rel.mean())
                so64=slope(obj_seq);sp64=slope(phys_seq);so32=slope(obj_seq[:32]);sp32=slope(phys_seq[:32])
                mean_score=min(mo/scales[a]['obj'],mp/scales[a]['phys'])
                slope64=min(so64/scales[a]['obj'],sp64/scales[a]['phys'])
                slope32=min(so32/scales[a]['obj'],sp32/scales[a]['phys'])
                axes[a]={'obj_suite_z':zo.tolist(),'phys_suite_z':zp.tolist(),'G_obj':go,'G_phys':gp,'G_sem':gs,'PASS':bool(gs>0),
                         'M_obj':mo,'M_phys':mp,'S_obj_64':so64,'S_phys_64':sp64,'S_obj_32':so32,'S_phys_32':sp32,
                         'MeanScore':mean_score,'Slope64':slope64,'Slope32':slope32}
            arr.append({'label':label,'checkpoint':path(sd,label),'axes':axes})
          states[str(sd)]=arr
        samples=[]
        for sd in SEEDS:
          arr=states[str(sd)]
          for i in range(8):
            for a in AXES:
              s=arr[i]['axes'][a];t=arr[i+1]['axes'][a]
              if not s['PASS']:continue
              target_fail=not t['PASS'];clear=min(s['G_sem'],-t['G_sem']) if target_fail else float('-inf')
              sweep_rob=target_fail and all(pass_tau(s['obj_suite_z'],s['phys_suite_z'],tau) and not pass_tau(t['obj_suite_z'],t['phys_suite_z'],tau) for tau in (-.25,0,.25))
              flips=0
              if target_fail:
                for idxs in itertools.product(range(4),repeat=4):
                  so=[s['obj_suite_z'][k] for k in idxs];sp=[s['phys_suite_z'][k] for k in idxs];to=[t['obj_suite_z'][k] for k in idxs];tp=[t['phys_suite_z'][k] for k in idxs]
                  flips += pass_tau(so,sp,0) and (not pass_tau(to,tp,0))
              pflip=flips/256 if target_fail else 0.0
              y=int(target_fail and clear>=.25 and sweep_rob and pflip>=.50)
              samples.append({'seed':sd,'from':arr[i]['label'],'to':arr[i+1]['label'],'axis':a,'Y_robust_collapse':y,
                              'target_PASS':t['PASS'],'clearance':clear,'sweep_robust':bool(sweep_rob),'bootstrap_flip_probability':pflip,
                              'source_G_sem':s['G_sem'],'MeanScore':s['MeanScore'],'Slope64':s['Slope64'],'Slope32':s['Slope32'],
                              'S_obj_64_scaled':s['S_obj_64']/scales[a]['obj'],'S_phys_64_scaled':s['S_phys_64']/scales[a]['phys'],
                              'S_obj_32_scaled':s['S_obj_32']/scales[a]['obj'],'S_phys_32_scaled':s['S_phys_32']/scales[a]['phys']})
        y=[s['Y_robust_collapse'] for s in samples]
        aucs={
          'mean':auc(y,[-s['MeanScore'] for s in samples]),
          'slope64':auc(y,[-s['Slope64'] for s in samples]),
          'slope32':auc(y,[-s['Slope32'] for s in samples]),
          'G_sem':auc(y,[-s['source_G_sem'] for s in samples]),
          'obj_slope32':auc(y,[-s['S_obj_32_scaled'] for s in samples]),
          'phys_slope32':auc(y,[-s['S_phys_32_scaled'] for s in samples]),
          'obj_slope64':auc(y,[-s['S_obj_64_scaled'] for s in samples]),
          'phys_slope64':auc(y,[-s['S_phys_64_scaled'] for s in samples]),
        }
        perseed={};comparable=[]
        for sd in SEEDS:
          q=[s for s in samples if s['seed']==sd];pos=[s for s in q if s['Y_robust_collapse']];neg=[s for s in q if not s['Y_robust_collapse']]
          rec={'eligible_n':len(q),'positive_n':len(pos),'negative_n':len(neg),'median_slope32_positive':float(np.median([s['Slope32'] for s in pos])) if pos else None,'median_slope32_negative':float(np.median([s['Slope32'] for s in neg])) if neg else None}
          rec['positive_more_negative']=bool(rec['median_slope32_positive']<rec['median_slope32_negative']) if pos and neg else None
          if pos and neg:comparable.append(rec['positive_more_negative'])
          perseed[str(sd)]=rec
        negs=[s for s in samples if s['Slope32']<0];nonnegs=[s for s in samples if s['Slope32']>=0]
        neg_rate=float(np.mean([s['Y_robust_collapse'] for s in negs])) if negs else float('nan');nonneg_rate=float(np.mean([s['Y_robust_collapse'] for s in nonnegs])) if nonnegs else float('nan')
        split_diff=neg_rate-nonneg_rate if np.isfinite(neg_rate) and np.isfinite(nonneg_rate) else float('nan')
        pos_n=int(sum(y));eligible=len(samples)
        criteria={
          'powered_eligible_ge8_positive_ge3':eligible>=8 and pos_n>=3,
          'auc_slope32_ge0p75':bool(np.isfinite(aucs['slope32']) and aucs['slope32']>=.75),
          'slope32_beats_mean_by0p10':bool(np.isfinite(aucs['slope32']) and np.isfinite(aucs['mean']) and aucs['slope32']>=aucs['mean']+.10),
          'auc_slope64_ge0p75':bool(np.isfinite(aucs['slope64']) and aucs['slope64']>=.75),
          'median_slope32_more_negative_all_comparable_seeds':bool(all(comparable)),
          'negative_slope_risk_rate_advantage_ge0p25':bool(np.isfinite(split_diff) and split_diff>=.25),
          'mean_not_equivalent_within0p05':bool(np.isfinite(aucs['slope32']) and np.isfinite(aucs['mean']) and aucs['slope32']>=aucs['mean']+.05),
        }
        if not criteria['powered_eligible_ge8_positive_ge3']:status='UNDERPOWERED / INCONCLUSIVE'
        elif all(criteria.values()):status='PROSPECTIVE SLOPE VALIDATED'
        else:status='SLOPE DIAGNOSTIC ONLY'
        axis_counts={a:{'eligible':sum(s['axis']==a for s in samples),'positive':sum(s['axis']==a and s['Y_robust_collapse'] for s in samples)} for a in AXES}
        retained=[s for s in samples if not s['Y_robust_collapse']]
        false_high=float(np.mean([s['Slope32']<0 for s in retained])) if retained else float('nan')
        rep={'schema':'prospective_slope_validation_v1','status':status,'new_training_seeds':list(SEEDS),'eligible_source_pass_n':eligible,'robust_collapse_positive_n':pos_n,
             'aucs':aucs,'per_seed':perseed,'per_axis_counts':axis_counts,'risk_split':{'negative_slope_n':len(negs),'nonnegative_slope_n':len(nonnegs),'negative_slope_collapse_rate':neg_rate,'nonnegative_slope_collapse_rate':nonneg_rate,'difference':split_diff},
             'false_high_risk_rate_retained':false_high,'criteria':criteria,'decision':{'temporal_relational_target_design_authorized':status=='PROSPECTIVE SLOPE VALIDATED','training_authorized':False},'samples':samples,'states':states}
        out=RUN/'prospective_slope_validation_report.json';out.write_text(json.dumps(rep,indent=2)+'\n')
        (RUN/'PROSPECTIVE_SLOPE_PROVENANCE.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':status,'report_sha256':sha(out),'contract_sha256':sha(ROOT/'docs/contracts/preference_control/prospective-slope-validation-contract.md'),'training_script_sha256':sha(ROOT/'scripts/rl/experiments/architectures/preference_control/slope_control/workflow.py'),'measurement_script_sha256':sha(ROOT/'scripts/rl/experiments/architectures/preference_control/slope_control/workflow.py'),'analysis_script_sha256':sha(Path(__file__).resolve()),'raw_cache_count':len(list(CACHE.glob('*.npz')))},indent=2)+'\n')
        print(json.dumps({k:rep[k] for k in ['status','eligible_source_pass_n','robust_collapse_positive_n','aucs','per_seed','per_axis_counts','risk_split','false_high_risk_rate_retained','criteria']},indent=2))
    if True:main()

STAGES = {
    "prospective_slope_control_seed": run_prospective_slope_control_seed,
    "prospective_slope_measure_checkpoint": run_prospective_slope_measure_checkpoint,
    "prospective_slope_offline_validation": run_prospective_slope_offline_validation,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
