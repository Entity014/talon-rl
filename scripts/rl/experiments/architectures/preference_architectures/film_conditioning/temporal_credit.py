"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_v2b_competence_floor_endpoint_timeline():
    """Run former v2b_competence_floor_endpoint_timeline.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    from rl.experiments.common.utilities.v2b2_semantic_eval import evaluate,obs_tensor as ot,PREFS,ORDER,IDX,PHYS,SUITES
    PILOT=ROOT/'runs/v2b_competence_floor_paired_pilot-2026-09-24'
    OUT=ROOT/'runs/v2b_competence_floor_endpoint_timeline-2026-09-24'
    STEPS=(0,5,10,15,20,25)
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def main():
     from isaaclab.app import AppLauncher
     saved=sys.argv[:];sys.argv=[sys.argv[0]]
     app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
     env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
      env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
      o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
      mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene['robot']
      result={}
      for arm in ('control','floor'):
       result[arm]={}
       for step in STEPS:
        ck=PILOT/f'{arm}_model_{step}.pt'
        m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model']);m.eval()
        rows=[]
        for suite in range(SUITES):
         seed=840001+suite
         for lab in ('T','A','O','S','C'):
          q=evaluate(env,m,mgr,robot,PREFS[lab],seed);q.update({'suite':suite,'label':lab});rows.append(q)
        endpoint={};passes={}
        for lab in ORDER:
         j=IDX[lab];pk=PHYS[lab];oo=[];pp=[];sv=[];do=[];dp=[]
         for suite in range(SUITES):
          r=next(x for x in rows if x['suite']==suite and x['label']==lab)
          c=next(x for x in rows if x['suite']==suite and x['label']=='C')
          od=r['normalized_objective_mean'][j]-c['normalized_objective_mean'][j]
          pd=r['physical'][pk]-c['physical'][pk]
          oo.append(od>0);pp.append(pd<0);sv.append(r['survival']);do.append(od);dp.append(pd)
         endpoint[lab]={'objective_correct_fraction':float(np.mean(oo)),'physical_correct_fraction':float(np.mean(pp)),
                        'mean_objective_delta_vs_center':float(np.mean(do)),'mean_physical_delta_vs_center':float(np.mean(dp)),
                        'min_survival':float(np.min(sv))}
         passes[lab]=bool(endpoint[lab]['objective_correct_fraction']>=.75 and endpoint[lab]['physical_correct_fraction']>=.75 and endpoint[lab]['min_survival']>=.95)
        result[arm][str(step)]={'endpoint':endpoint,'endpoint_pass':passes,'pass_count':int(sum(passes.values())),'pass_axes':[k for k,v in passes.items() if v]}
        print(arm,step,result[arm][str(step)]['pass_axes'],flush=True)
      # Pilot-gate timeline criteria independent of continuum final comparison.
      post=[5,10,15,20,25]
      floor_events=sum(result['floor'][str(s)]['pass_count'] for s in post)
      control_events=sum(result['control'][str(s)]['pass_count'] for s in post)
      floor_o=sum(bool(result['floor'][str(s)]['endpoint_pass']['O']) for s in post)
      max_sim=max(result['floor'][str(s)]['pass_count'] for s in post)
      timeline={'floor_total_pass_events':floor_events,'control_total_pass_events':control_events,
                'event_advantage':floor_events-control_events,'floor_O_retained_checkpoints':floor_o,
                'floor_max_simultaneous_passes':max_sim,'floor_final_pass_count':result['floor']['25']['pass_count'],
                'control_final_pass_count':result['control']['25']['pass_count']}
      report={'schema':'v2b_competence_floor_endpoint_timeline_v1','measurement_only':True,'training_updates':0,
              'steps':list(STEPS),'arms':result,'timeline':timeline}
      OUT.mkdir(parents=True,exist_ok=True);out=OUT/'endpoint_timeline_report.json';out.write_text(json.dumps(report,indent=2)+'\n')
      (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','report_sha256':sha(out),'script_sha256':sha(Path(__file__).resolve())},indent=2)+'\n')
      print(json.dumps(timeline,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_v2b_competence_floor_maxmin_audit():
    """Run former v2b_competence_floor_maxmin_audit.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    
    from rl.experiments.common.utilities.v2b_fixed_stream_multiupdate_audit import (
        ot,pref_batch,collect_actor,gae,actor_params,ORDER,groups
    )
    
    STAGES=(50,75)
    SEEDS=(960001,960002,960003,960004)
    H=32;NENV=8;LAM=.95
    OUT=ROOT/'runs/v2b_competence_floor_maxmin_audit-2026-09-24'
    CKPT_DIR=ROOT/'runs/v2b_lambda095_pilot-2026-09-23'
    GRID_N=50  # simplex resolution 0.02
    EPS=1e-12
    
    
    def flat_grad(loss,params):
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])
    
    
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>EPS else 0.0
    
    
    def raw_objective_and_mixed_losses(m,b):
        with torch.no_grad():
            vt=m.value_with_preference(b['obs'],b['w']).reshape(H,NENV,4)
            w0=b['w'].reshape(H,NENV,4)[-1]
            nv=m.value_with_preference(b['next_obs'],w0)
            A=gae(b['rt'],vt,nv,b['dt'],LAM).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b['obs'],b['w'],b['u'])
        ratio=torch.exp(logp-b['old'].detach()); clipped=ratio.clamp(.8,1.2)
        raw={}; weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clipped*A[:,j])
            raw[lab]=-po.mean()
            weighted[lab]=-(4.0*b['w'][:,j]*po).mean()
        return raw, weighted, sum(weighted.values()), float((ratio-1).abs().max().detach().cpu())
    
    
    def simplex_maxmin(gram,n=GRID_N):
        # alpha=[a,b,c,d]/n, deterministic exhaustive integer simplex.
        best=None
        for ia in range(n+1):
            for ib in range(n-ia+1):
                for ic in range(n-ia-ib+1):
                    idd=n-ia-ib-ic
                    alpha=np.array([ia,ib,ic,idd],dtype=np.float64)/n
                    vv=float(alpha@gram@alpha)
                    if vv<=1e-16: continue
                    gains=(gram@alpha)/np.sqrt(vv)
                    worst=float(np.min(gains)); mean=float(np.mean(gains))
                    # deterministic tie-break: worst, then mean, then smaller max weight.
                    key=(worst,mean,-float(np.max(alpha)))
                    if best is None or key>best[0]:
                        best=(key,alpha,gains,np.sqrt(vv))
        if best is None: raise RuntimeError('No nonzero max-min simplex direction')
        return best[1],best[2],best[3]
    
    
    def case_geometry(m,b):
        params=actor_params(m)
        raw,weighted,total,ratioerr=raw_objective_and_mixed_losses(m,b)
        # Convert loss gradients to ascent directions.
        q={lab:-flat_grad(raw[lab],params).detach() for lab in ORDER}
        d_mix=-flat_grad(total,params).detach()
        qnorm={lab:float(q[lab].norm().cpu()) for lab in ORDER}
        if min(qnorm.values())<=1e-12 or float(d_mix.norm())<=1e-12:
            raise RuntimeError(f'degenerate gradient norms raw={qnorm} mix={float(d_mix.norm())}')
        U=torch.stack([q[lab]/(q[lab].norm()+EPS) for lab in ORDER])
        gram=(U@U.T).double().cpu().numpy()
        alpha,grid_gains,_=simplex_maxmin(gram)
        at=torch.tensor(alpha,device=U.device,dtype=U.dtype)
        v=(at[:,None]*U).sum(0)
        vnorm=v.norm()
        d_floor=d_mix.norm()*v/(vnorm+EPS)
        mix_unit=d_mix/(d_mix.norm()+EPS); floor_unit=d_floor/(d_floor.norm()+EPS)
        mix_g=(U@mix_unit).detach().cpu().numpy()
        floor_g=(U@floor_unit).detach().cpu().numpy()
        pair={}
        for i,a in enumerate(ORDER):
            for j,bn in enumerate(ORDER):
                if j>i: pair[f'{a}-{bn}']=float(gram[i,j])
        return {
          'raw_objective_gradient_norms':qnorm,
          'pairwise_objective_cosine':pair,
          'mixed_norm':float(d_mix.norm().cpu()),
          'floor_norm':float(d_floor.norm().cpu()),
          'norm_ratio':float((d_floor.norm()/(d_mix.norm()+EPS)).cpu()),
          'mixed_normalized_gain':{lab:float(mix_g[i]) for i,lab in enumerate(ORDER)},
          'floor_normalized_gain':{lab:float(floor_g[i]) for i,lab in enumerate(ORDER)},
          'mixed_worst_gain':float(np.min(mix_g)),
          'floor_worst_gain':float(np.min(floor_g)),
          'worst_gain_improvement':float(np.min(floor_g)-np.min(mix_g)),
          'mixed_mean_gain':float(np.mean(mix_g)),
          'floor_mean_gain':float(np.mean(floor_g)),
          'mixed_negative_axes':int(np.sum(mix_g<0)),
          'floor_negative_axes':int(np.sum(floor_g<0)),
          'floor_all_nonnegative':bool(np.all(floor_g>=-1e-8)),
          'floor_alpha':{lab:float(alpha[i]) for i,lab in enumerate(ORDER)},
          'floor_cosine_to_mixed':cos(d_floor,d_mix),
          'ratio_max_error':ratioerr,
          'grid_gain_consistency_max_abs':float(np.max(np.abs(grid_gains-floor_g))),
        }
    
    
    def agg(rows):
        keys=('mixed_worst_gain','floor_worst_gain','worst_gain_improvement','mixed_mean_gain','floor_mean_gain',
              'mixed_negative_axes','floor_negative_axes','floor_cosine_to_mixed','norm_ratio','ratio_max_error')
        out={}
        for k in keys:
            a=np.asarray([r[k] for r in rows],float)
            out[k]={'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'max':float(a.max())}
        out['floor_all_nonnegative_fraction']=float(np.mean([r['floor_all_nonnegative'] for r in rows]))
        out['worst_gain_improved_fraction']=float(np.mean([r['floor_worst_gain']>r['mixed_worst_gain']+1e-9 for r in rows]))
        return out
    
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            rows=[];by_stage={}
            for st in STAGES:
                ck=CKPT_DIR/f'model_{st}.pt'
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model']);m.eval()
                sr=[]
                for si,seed in enumerate(SEEDS):
                    labs,w=pref_batch(st+si+1,torch.device('cuda'))
                    b=collect_actor(env,m,w,mgr,seed)
                    g=case_geometry(m,b)
                    g.update({'stage':st,'seed':seed,'preference_labels':labs})
                    sr.append(g);rows.append(g)
                by_stage[str(st)]={'aggregate':agg(sr),'cases':sr}
    
            overall=agg(rows)
            conflict=[r for r in rows if r['mixed_negative_axes']>0]
            repaired=[r for r in conflict if r['floor_negative_axes']<r['mixed_negative_axes']]
            normerr=[abs(r['norm_ratio']-1.0) for r in rows]
            improved=sum(r['floor_worst_gain']>r['mixed_worst_gain']+1e-9 for r in rows)
            stage_relevance=all(by_stage[str(st)]['aggregate']['floor_worst_gain']['mean']>
                                by_stage[str(st)]['aggregate']['mixed_worst_gain']['mean'] for st in STAGES)
            finite_nonzero=all(np.isfinite(r['floor_norm']) and r['floor_norm']>1e-12 for r in rows)
            cosvals=[r['floor_cosine_to_mixed'] for r in rows]
            criteria={
              'norm_budget_exact':float(max(normerr))<=1e-5 and float(np.mean(normerr))<=1e-5,
              'worst_improved_at_least_7_of_8':improved>=7,
              'mean_worst_gain_improvement_ge_0p05':overall['worst_gain_improvement']['mean']>=0.05,
              'negative_axis_repair_fraction_ge_0p75':(len(conflict)>0 and len(repaired)/len(conflict)>=0.75),
              'floor_direction_finite_nonzero':finite_nonzero,
              'mean_cosine_to_mixed_gt_0p10':float(np.mean(cosvals))>0.10,
              'no_cosine_below_minus_0p50':float(np.min(cosvals))>=-0.50,
              'both_checkpoints_worst_gain_improved':stage_relevance,
              'ppo_ratio_invariant':max(r['ratio_max_error'] for r in rows)<=1e-4,
            }
            passed=all(criteria.values())
            report={
              'schema':'v2b_competence_floor_maxmin_audit_v1',
              'status':'COMPETENCE-FLOOR AUDIT PASS' if passed else 'COMPETENCE-FLOOR AUDIT FAIL',
              'measurement_only':True,'optimizer_steps':0,'training_authorized':bool(passed),
              'architecture':'V2-B','gae_lambda':LAM,'stages':list(STAGES),'seeds':list(SEEDS),
              'mixed_batch':'exact training-style endpoint preference mix, 2 envs per T/A/O/S',
              'simplex_resolution':1/GRID_N,
              'semantic_context':{'u50_endpoint_pass':['O'],'u75_endpoint_pass':['A']},
              'by_stage':by_stage,'overall':overall,
              'conflict_cases':len(conflict),'conflict_repaired_cases':len(repaired),
              'criteria':criteria,
              'decision':{'short_training_pilot_authorized':bool(passed)},
            }
            OUT.mkdir(parents=True,exist_ok=True)
            out=OUT/'competence_floor_maxmin_report.json';out.write_text(json.dumps(report,indent=2)+'\n')
            prov={'status':'FROZEN_BY_HASH','decision':report['status'],
                  'report_sha256':sha(out),'script_sha256':sha(Path(__file__).resolve()),
                  'contract_sha256':sha(ROOT/'docs/contracts/preference_architectures/competence-floor-maxmin-contract.md'),
                  'checkpoint_sha256':{str(st):sha(CKPT_DIR/f'model_{st}.pt') for st in STAGES}}
            (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps(prov,indent=2)+'\n')
            compact={'status':report['status'],'criteria':criteria,'overall':overall,
                     'stage50':by_stage['50']['aggregate'],'stage75':by_stage['75']['aggregate'],
                     'conflict_cases':len(conflict),'conflict_repaired_cases':len(repaired)}
            print(json.dumps(compact,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    
    if True:main()

def run_v2b_competence_floor_paired_pilot():
    """Run former v2b_competence_floor_paired_pilot.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import copy,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    from rl.experiments.common.utilities.v2b_lambda_training_pilot import (
        ot,pref_batch,collect_actor,collect_support64,fit_expanded_current_policy,
        fresh_phase_audit,sensitivity,POOL_MAX,H,NENV,PREFS,ORDER
    )
    from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
    
    START=ROOT/'runs/v2b_lambda095_pilot-2026-09-23/model_50.pt'
    OUT=ROOT/'runs/v2b_competence_floor_paired_pilot-2026-09-24'
    BASE_SEED=73001
    STEPS=25
    SNAPS=(0,5,10,15,20,25)
    LAM=.95
    GRID_N=50
    EPS=1e-12
    ANCHOR_SPECS=[(0,83001),(2,83275),(3,83412),(4,83549),(9,84234),(11,84508)]
    OBJ_ORDER=('T','A','O','S')
    
    
    def actor_params(m):
        return [p for n,p in m.named_parameters() if n.startswith('actor_') or n=='log_std' or n.startswith('preference_embedding') or n.startswith('preference_film')]
    
    def flat_params(ps):return torch.cat([p.detach().reshape(-1) for p in ps])
    def flat_grad(loss,ps):
        gs=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(ps,gs)])
    def assign_flat_grad(ps,g):
        pos=0
        for p in ps:
            n=p.numel();p.grad=g[pos:pos+n].reshape_as(p).clone();pos+=n
        assert pos==g.numel()
    def cos(a,b):
        den=float(a.norm()*b.norm());return float(torch.dot(a,b)/den) if den>EPS else 0.0
    
    def simplex_maxmin(gram,n=GRID_N):
        best=None
        for ia in range(n+1):
          for ib in range(n-ia+1):
           for ic in range(n-ia-ib+1):
            idd=n-ia-ib-ic
            a=np.asarray([ia,ib,ic,idd],float)/n
            vv=float(a@gram@a)
            if vv<=1e-16:continue
            gains=(gram@a)/np.sqrt(vv);key=(float(gains.min()),float(gains.mean()),-float(a.max()))
            if best is None or key>best[0]:best=(key,a,gains)
        if best is None:raise RuntimeError('degenerate simplex')
        return best[1]
    
    def floor_gradient(m,b):
        ps=actor_params(m)
        with torch.no_grad():
            vt=m.value_with_preference(b['obs'],b['w']).reshape(H,NENV,4)
            w0=b['w'].reshape(H,NENV,4)[-1]
            nv=m.value_with_preference(b['next_obs'],w0)
            adv,_=vector_gae(b['rt'],vt,nv,b['dt'],lam=LAM)
            A=adv.reshape(-1,4).detach()
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(b['obs'],b['w'],b['u'])-b['old'].detach())
        clip=ratio.clamp(.8,1.2)
        raw={};weighted={}
        for j,lab in enumerate(OBJ_ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            raw[lab]=-po.mean();weighted[lab]=-(4*b['w'][:,j]*po).mean()
        mixed=sum(weighted.values())
        # ascent vectors q_i and mixed ascent
        q={lab:-flat_grad(raw[lab],ps).detach() for lab in OBJ_ORDER}
        d_mix=-flat_grad(mixed,ps).detach()
        U=torch.stack([q[x]/(q[x].norm()+EPS) for x in OBJ_ORDER])
        gram=(U@U.T).double().cpu().numpy();alpha=simplex_maxmin(gram)
        at=torch.tensor(alpha,device=U.device,dtype=U.dtype)
        v=(at[:,None]*U).sum(0);d_floor=d_mix.norm()*v/(v.norm()+EPS)
        mu=d_mix/(d_mix.norm()+EPS);fu=d_floor/(d_floor.norm()+EPS)
        mg=(U@mu).detach().cpu().numpy();fg=(U@fu).detach().cpu().numpy()
        # optimizer consumes descent gradient.
        g_floor=-d_floor
        diag={
          'mixed_worst_gain':float(mg.min()),'floor_worst_gain':float(fg.min()),
          'mixed_negative_axes':int((mg<0).sum()),'floor_negative_axes':int((fg<0).sum()),
          'floor_alpha':{lab:float(alpha[i]) for i,lab in enumerate(OBJ_ORDER)},
          'floor_cosine_to_mixed':cos(d_floor,d_mix),'norm_ratio':float(d_floor.norm()/(d_mix.norm()+EPS)),
          'mixed_loss':float(mixed.detach().cpu()),
          'ratio_maxerr':float((ratio-1).abs().max().detach().cpu())}
        return g_floor,diag
    
    def init_pool(env,m,mgr):
        pools={'early':[],'late':[]}
        # Rebuild a matched restart pool using the same 12 canonical candidate specs.
        for k in range(12):
            _,w=pref_batch(k,torch.device('cuda'));seed=BASE_SEED+10000+k*137
            for u in collect_support64(env,m,w,mgr,seed):pools[u['phase']].append(u)
        return pools
    
    def anchor_units(env,m,mgr):
        out={'early':[],'late':[]}
        for k,seed in ANCHOR_SPECS:
            _,w=pref_batch(k,torch.device('cuda'))
            for u in collect_support64(env,m,w,mgr,seed):out[u['phase']].append(u)
        return out
    
    def prep_arm(name,model,env,mgr):
        for n,p in model.named_parameters():
            if n.startswith('critic_body') or n.startswith('critic_head'):p.requires_grad_(False)
        ps=actor_params(model);opt=torch.optim.Adam(ps,lr=1e-3)
        pools=init_pool(env,model,mgr)
        fit_expanded_current_policy(model,anchor_units(env,model,mgr),pools)
        return {'name':name,'m':model,'ps':ps,'opt':opt,'pools':pools,'rows':[],'snaps':{}}
    
    def update_pool(env,arm,mgr,global_u):
        _,ws=pref_batch(global_u+17,torch.device('cuda'))
        units=collect_support64(env,arm['m'],ws,mgr,BASE_SEED+200000+global_u*223)
        for u in units:
            ph=u['phase'];arm['pools'][ph].append(u)
            if len(arm['pools'][ph])>POOL_MAX:arm['pools'][ph].pop(0)
    
    def fit_critic(env,arm,mgr):
        return fit_expanded_current_policy(arm['m'],anchor_units(env,arm['m'],mgr),arm['pools'])
    
    def save_snap(env,arm,mgr,step):
        m=arm['m'];m.eval()
        arm['snaps'][str(step)]={'sensitivity':sensitivity(m,arm['probe']),
                                 'phase_critic':fresh_phase_audit(env,m,mgr,BASE_SEED+700000+step*1000)}
        torch.save({'model':m.state_dict(),'paired_step':step,'global_update':50+step,'arm':arm['name'],'seed':BASE_SEED},OUT/f"{arm['name']}_model_{step}.pt")
        m.train()
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=BASE_SEED
            cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
            o,_=env.reset(seed=BASE_SEED);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            state=torch.load(START,map_location='cuda',weights_only=False)['model']
            c=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();c.load_state_dict(state);c.train()
            f=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();f.load_state_dict(state);f.train()
            control=prep_arm('control',c,env,mgr);floor=prep_arm('floor',f,env,mgr)
            control['probe']=o.detach().clone();floor['probe']=o.detach().clone()
            OUT.mkdir(parents=True,exist_ok=True)
            save_snap(env,control,mgr,0);save_snap(env,floor,mgr,0)
    
            for step in range(1,STEPS+1):
                global_u=50+step
                labs,w=pref_batch(global_u,torch.device('cuda'))
                for arm in (control,floor):
                    m=arm['m'];ps=arm['ps'];opt=arm['opt']
                    # Matched reset seed; policy-dependent trajectories are allowed to diverge naturally.
                    main=collect_actor(env,m,w,mgr,BASE_SEED+global_u*211,True)
                    update_pool(env,arm,mgr,global_u);selected=fit_critic(env,arm,mgr)
                    before=flat_params(ps).clone()
                    opt.zero_grad(set_to_none=True)
                    extra={}
                    if arm['name']=='control':
                        with torch.no_grad():
                            vt=m.value_with_preference(main['obs'],main['w']).reshape(H,NENV,4)
                            nv=m.value_with_preference(main['next_obs'],w)
                            adv,_=vector_gae(main['rt'],vt,nv,main['dt'],lam=LAM)
                        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main['obs'],main['w'],main['u'])-main['old'].detach())
                        rerr=float((ratio-1).abs().max().detach().cpu())
                        loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main['w'])
                        loss.backward();extra={'loss':float(loss.detach().cpu()),'ratio_maxerr':rerr}
                    else:
                        gf,extra=floor_gradient(m,main);assign_flat_grad(ps,gf)
                    preclip=float(torch.nn.utils.clip_grad_norm_(ps,1.0).detach().cpu());opt.step()
                    with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                    stepnorm=float((flat_params(ps)-before).norm().cpu())
                    post=None
                    if step in SNAPS:post=fit_critic(env,arm,mgr)
                    row={'step':step,'global_update':global_u,'labels':labs,'preclip_grad_norm':preclip,
                         'parameter_step_norm':stepnorm,'termination_fraction':main['termination_fraction'],
                         'selected':selected,'post_refresh':post}|extra
                    arm['rows'].append(row)
                if step in SNAPS:
                    save_snap(env,control,mgr,step);save_snap(env,floor,mgr,step)
    
            report={'schema':'v2b_competence_floor_paired_pilot_v1','start_checkpoint':str(START.relative_to(ROOT)),
                    'seed':BASE_SEED,'paired_steps':STEPS,'global_updates':[51,75],'gae_lambda':LAM,
                    'fresh_adam_both_arms':True,'historical_anchor_specs':ANCHOR_SPECS,
                    'arms':{a['name']:{'rows':a['rows'],'snapshots':a['snaps']} for a in (control,floor)}}
            summaries={}
            for a in (control,floor):
                rr=a['rows'];crit=a['snaps'][str(STEPS)]['phase_critic']
                summaries[a['name']]={
                  'max_ratio_error':float(max(r.get('ratio_maxerr',0) for r in rr)),
                  'last10_termination_fraction':float(np.mean([r['termination_fraction'] for r in rr[-10:]])),
                  'mean_parameter_step_norm':float(np.mean([r['parameter_step_norm'] for r in rr])),
                  'final_critic':crit,
                  'foundation_pass':bool(crit['early']['ev_mean']>0 and crit['late']['ev_mean']>0 and crit['early']['negative_fraction']<=.25 and crit['late']['negative_fraction']<=.25 and crit['combined_negative_fraction']<=.25 and np.mean([r['termination_fraction'] for r in rr[-10:]])<.5 and max(r.get('ratio_maxerr',0) for r in rr)<=1e-4)}
            fr=floor['rows']
            summaries['floor']['mean_mixed_worst_gain']=float(np.mean([r['mixed_worst_gain'] for r in fr]))
            summaries['floor']['mean_floor_worst_gain']=float(np.mean([r['floor_worst_gain'] for r in fr]))
            summaries['floor']['all_nonnegative_fraction']=float(np.mean([r['floor_negative_axes']==0 for r in fr]))
            summaries['floor']['mean_cosine_to_mixed']=float(np.mean([r['floor_cosine_to_mixed'] for r in fr]))
            summaries['floor']['mean_norm_ratio']=float(np.mean([r['norm_ratio'] for r in fr]))
            report['summary']=summaries
            out=OUT/'paired_pilot_report.json';out.write_text(json.dumps(report,indent=2)+'\n')
            (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','report_sha256':sha(out),'script_sha256':sha(Path(__file__).resolve()),'contract_sha256':sha(ROOT/'docs/contracts/preference_architectures/competence-floor-pilot-contract.md'),'start_checkpoint_sha256':sha(START)},indent=2)+'\n')
            print(json.dumps(summaries,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_competence_floor_semantic_aggregate():
    """Run former v2b_competence_floor_semantic_aggregate.py stage."""
    from pathlib import Path
    import json,hashlib
    ROOT=Path(__file__).resolve().parents[4]
    PATHDIR=ROOT/'runs/v2b_competence_floor_semantic_path-2026-09-24'
    PILOT=ROOT/'runs/v2b_competence_floor_paired_pilot-2026-09-24/paired_pilot_report.json'
    STEPS=(0,5,10,15,20,25);ARMS=('control','floor');AXES=('T','A','O','S')
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
     out={}
     for arm in ARMS:
      rows=[]
      for st in STEPS:
       p=PATHDIR/f'{arm}_{st}/endpoint_report.json';d=json.load(open(p))
       passes=[a for a in AXES if d['endpoint_pass'][a]]
       rows.append({'step':st,'pass_axes':passes,'pass_count':len(passes),'endpoint_pass':d['endpoint_pass'],'endpoint':d['endpoint']})
      post=rows[1:]
      total=sum(r['pass_count'] for r in post)
      o_ret=sum('O' in r['pass_axes'] for r in post)
      out[arm]={'rows':rows,'post_start_total_pass_events':total,'orientation_retained_count_post_start':o_ret,
                'orientation_retained_fraction_post_start':o_ret/5,'max_simultaneous_passes_post_start':max(r['pass_count'] for r in post)}
     pilot=json.load(open(PILOT))
     criteria={
      'foundation_both_arms':pilot['summary']['control']['foundation_pass'] and pilot['summary']['floor']['foundation_pass'],
      'floor_has_ge2_simultaneous_passes':out['floor']['max_simultaneous_passes_post_start']>=2,
      'floor_total_pass_events_control_plus2':out['floor']['post_start_total_pass_events']>=out['control']['post_start_total_pass_events']+2,
      'floor_orientation_retained_ge3_of5':out['floor']['orientation_retained_count_post_start']>=3,
      'final_floor_pass_count_not_lower':out['floor']['rows'][-1]['pass_count']>=out['control']['rows'][-1]['pass_count'],
     }
     rep={'schema':'v2b_competence_floor_semantic_path_v1','arms':out,'pre_continuum_criteria':criteria,
          'continuum_final_pending':True}
     outp=PATHDIR/'semantic_path_summary.json';outp.write_text(json.dumps(rep,indent=2)+'\n')
     print(json.dumps(rep,indent=2))
    if True:main()

def run_v2b_horizon_visitation_credit_audit():
    """Run former v2b_horizon_visitation_credit_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_horizon_visitation_credit_audit-2026-09-23"
    
    PREFS={
        "O":np.array([.1,.1,.7,.1],np.float32),
        "C":np.array([.25,.25,.25,.25],np.float32),
    }
    G=.99;LAM=.95;NENV=8
    HORIZONS=(8,16,32)
    PHASES=(8,20,36,52)
    SEEDS=(900001,900002,900003,900004)
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1)
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def collect_source(env,m,source_lab,seed,max_steps=64):
        # Collect visitation states under either center or O-heavy.
        w=torch.tensor(PREFS[source_lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+111000)
        states=[];actions=[];us=[]
        with torch.no_grad():
            for _ in range(max_steps):
                states.append(cur.clone())
                a,lp,u=m.act_with_preference_latent(cur,w)
                actions.append(a.clone());us.append(u.clone())
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        return {"states":states,"actions":actions,"u":us}
    
    def restore_to_phase(env,m,source_lab,seed,phase,recorded_actions):
        # Replay exact recorded actions to rebuild simulator state at chosen phase.
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for k in range(phase):
            nxt,_,_,_,_=env.step(recorded_actions[k]);cur=ot(nxt).cuda()
        return cur
    
    def continuation_mc_direction(env,m,mgr,state,w,seed,horizon):
        # Fixed start state already loaded in simulator.
        # Estimate MC score direction by rolling stochastic continuations from this state.
        from talon_rl.rewards.objectives import normalized_objective_vector
        obs0=state
        torch.manual_seed(seed)
        obs=[];u=[];mean=[];std=[];r=[];done=[]
        cur=obs0
        with torch.no_grad():
            for _ in range(horizon):
                feat=m._actor_features_v2a(cur,w)
                mu=m.actor_mean(feat);sd=m.log_std.exp().expand_as(mu)
                dist=torch.distributions.Normal(mu,sd)
                uu=dist.sample()
                a=torch.tanh(uu)*m.ACTION_CLIP
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                rr=torch.tensor(vec[:,2],device="cuda")*env.unwrapped.step_dt
                obs.append(cur);u.append(uu);mean.append(mu);std.append(sd);r.append(rr);done.append((te|tr).cuda())
                cur=ot(nxt).cuda()
    
        U=torch.stack(u);MU=torch.stack(mean);SD=torch.stack(std);R=torch.stack(r);D=torch.stack(done).bool()
        # return-to-go
        ret=torch.zeros_like(R);run=torch.zeros_like(R[-1])
        for t in range(horizon-1,-1,-1):
            run=R[t]+G*run*(~D[t]).to(R.dtype);ret[t]=run
        centered=ret-ret.mean()
        score=(U-MU)/(SD*SD)
        g=(score*centered.unsqueeze(-1)).mean((0,1))
        return g
    
    def gae_direction_from_state(env,m,mgr,state,w,seed,horizon=32):
        # Recompute GAE on continuation from same frozen state using same policy distribution.
        from talon_rl.rewards.objectives import normalized_objective_vector
        from talon_rl.models.foundations.four_objective import vector_gae
        torch.manual_seed(seed)
        cur=state
        us=[];means=[];stds=[];R=[];D=[];V=[]
        with torch.no_grad():
            for _ in range(horizon):
                V.append(m.value_with_preference(cur,w))
                feat=m._actor_features_v2a(cur,w)
                mu=m.actor_mean(feat);sd=m.log_std.exp().expand_as(mu)
                dist=torch.distributions.Normal(mu,sd);uu=dist.sample();a=torch.tanh(uu)*m.ACTION_CLIP
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());us.append(uu);means.append(mu);stds.append(sd);cur=ot(nxt).cuda()
            nv=m.value_with_preference(cur,w)
        RT=torch.stack(R);DT=torch.stack(D).bool();VT=torch.stack(V)
        adv,_=vector_gae(RT,VT,nv,DT,lam=LAM)
        U=torch.stack(us);MU=torch.stack(means);SD=torch.stack(stds)
        score=(U-MU)/(SD*SD)
        g=(score*adv[:,:,2].unsqueeze(-1)).mean((0,1))
        return g
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute(): a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute(): a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
    
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
    
            raw=[]
            for source in ("C","O"):
                for seed in SEEDS:
                    src=collect_source(env,m,source,seed,64)
                    for phase in PHASES:
                        state=restore_to_phase(env,m,source,seed,phase,src["actions"])
                        # Always evaluate O-heavy credit from the frozen state regardless of source visitation.
                        wO=torch.tensor(PREFS["O"],device="cuda").repeat(NENV,1)
                        # Separate continuation seeds but deterministic mapping across horizon/source.
                        base_seed=seed+phase*1000
                        # GAE baseline on 32-step continuation.
                        state=restore_to_phase(env,m,source,seed,phase,src["actions"])
                        ggae=gae_direction_from_state(env,m,mgr,state,wO,base_seed+77,32)
                        mc={}
                        for H in HORIZONS:
                            state=restore_to_phase(env,m,source,seed,phase,src["actions"])
                            mc[H]=continuation_mc_direction(env,m,mgr,state,wO,base_seed+H,H)
                        row={
                            "source":source,"seed":seed,"phase":phase,
                            "gae_norm":float(ggae.norm().cpu()),
                            "mc_norms":{str(H):float(mc[H].norm().cpu()) for H in HORIZONS},
                            "gae_vs_mc":{str(H):cos(ggae,mc[H]) for H in HORIZONS},
                            "mc_short_vs_medium":cos(mc[8],mc[16]),
                            "mc_short_vs_long":cos(mc[8],mc[32]),
                            "mc_medium_vs_long":cos(mc[16],mc[32]),
                        }
                        raw.append(row)
    
            # aggregate source x phase
            agg={}
            for source in ("C","O"):
                agg[source]={}
                for phase in PHASES:
                    rr=[r for r in raw if r["source"]==source and r["phase"]==phase]
                    agg[source][str(phase)]={
                        "gae_vs_mc":{str(H):{"mean":float(np.mean([r["gae_vs_mc"][str(H)] for r in rr])),
                                             "std":float(np.std([r["gae_vs_mc"][str(H)] for r in rr]))} for H in HORIZONS},
                        "mc_short_vs_medium":{"mean":float(np.mean([r["mc_short_vs_medium"] for r in rr])),
                                              "std":float(np.std([r["mc_short_vs_medium"] for r in rr]))},
                        "mc_short_vs_long":{"mean":float(np.mean([r["mc_short_vs_long"] for r in rr])),
                                            "std":float(np.std([r["mc_short_vs_long"] for r in rr]))},
                        "mc_medium_vs_long":{"mean":float(np.mean([r["mc_medium_vs_long"] for r in rr])),
                                             "std":float(np.std([r["mc_medium_vs_long"] for r in rr]))},
                        "gae_norm":{"mean":float(np.mean([r["gae_norm"] for r in rr])),
                                    "std":float(np.std([r["gae_norm"] for r in rr]))},
                        "mc_norms":{str(H):{"mean":float(np.mean([r["mc_norms"][str(H)] for r in rr])),
                                           "std":float(np.std([r["mc_norms"][str(H)] for r in rr]))} for H in HORIZONS},
                    }
    
            # source effect at matched phase/horizon
            source_effect={}
            for phase in PHASES:
                source_effect[str(phase)]={}
                for H in HORIZONS:
                    c=agg["C"][str(phase)]["gae_vs_mc"][str(H)]["mean"]
                    o=agg["O"][str(phase)]["gae_vs_mc"][str(H)]["mean"]
                    source_effect[str(phase)][str(H)]={"O_minus_C_gae_mc_cos":o-c}
    
            report={"schema":"v2b_horizon_visitation_credit_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "checkpoint":str(a.checkpoint.relative_to(ROOT)),
                    "state_sources":["C","O"],"horizons":list(HORIZONS),"phases":list(PHASES),"seeds":list(SEEDS),
                    "aggregate":agg,"source_effect":source_effect,"raw":raw}
            out=a.output_dir/"horizon_visitation_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps({"aggregate":agg,"source_effect":source_effect},indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True:main()

def run_v2b_horizon_visitation_credit_audit_crn():
    """Run former v2b_horizon_visitation_credit_audit_crn.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_horizon_visitation_credit_audit_crn-2026-09-23"
    PREFS={"O":np.array([.1,.1,.7,.1],np.float32),"C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;LAM=.95;NENV=8;MAXH=32
    HORIZONS=(8,16,32);PHASES=(8,20,36,52);SEEDS=(910001,910002,910003,910004)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a.reshape(-1),b.reshape(-1))/den) if den>1e-12 else 0.0
    
    def collect_source(env,m,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+123000)
        acts=[]
        with torch.no_grad():
            for _ in range(64):
                a,_,_=m.act_with_preference_latent(cur,w);acts.append(a.clone())
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        return acts
    
    def restore(env,seed,phase,acts):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for k in range(phase):
            nxt,_,_,_,_=env.step(acts[k]);cur=ot(nxt).cuda()
        return cur
    
    def continuation_once(env,m,mgr,state,w,noise_seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        from talon_rl.models.foundations.four_objective import vector_gae
        torch.manual_seed(noise_seed);cur=state
        U=[];MU=[];SD=[];R=[];D=[];V=[]
        with torch.no_grad():
            for _ in range(MAXH):
                V.append(m.value_with_preference(cur,w))
                feat=m._actor_features_v2a(cur,w);mu=m.actor_mean(feat);sd=m.log_std.exp().expand_as(mu)
                dist=torch.distributions.Normal(mu,sd);u=dist.sample();a=torch.tanh(u)*m.ACTION_CLIP
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());U.append(u);MU.append(mu);SD.append(sd);cur=ot(nxt).cuda()
            nv=m.value_with_preference(cur,w)
        U=torch.stack(U);MU=torch.stack(MU);SD=torch.stack(SD);R=torch.stack(R);D=torch.stack(D).bool();V=torch.stack(V)
        score=(U-MU)/(SD*SD)
        adv,_=vector_gae(R,V,nv,D,lam=LAM)
        ggae=(score*adv[:,:,2].unsqueeze(-1)).mean((0,1))
        gmc={}
        for H in HORIZONS:
            r=R[:H,:,2];d=D[:H]
            ret=torch.zeros_like(r);run=torch.zeros_like(r[-1])
            for t in range(H-1,-1,-1):
                run=r[t]+G*run*(~d[t]).to(r.dtype);ret[t]=run
            centered=ret-ret.mean()
            gmc[H]=(score[:H]*centered.unsqueeze(-1)).mean((0,1))
        return ggae,gmc
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            for src in ("C","O"):
                for seed in SEEDS:
                    acts=collect_source(env,m,src,seed)
                    for ph in PHASES:
                        st=restore(env,seed,ph,acts)
                        w=torch.tensor(PREFS["O"],device="cuda").repeat(NENV,1)
                        # CRN: same single 32-step continuation defines GAE and all MC horizons.
                        ggae,gmc=continuation_once(env,m,mgr,st,w,seed+ph*1000+777)
                        rows.append({"source":src,"seed":seed,"phase":ph,
                                     "gae_vs_mc":{str(H):cos(ggae,gmc[H]) for H in HORIZONS},
                                     "mc_8_vs_16":cos(gmc[8],gmc[16]),"mc_8_vs_32":cos(gmc[8],gmc[32]),"mc_16_vs_32":cos(gmc[16],gmc[32]),
                                     "gae_norm":float(ggae.norm().cpu()),"mc_norms":{str(H):float(gmc[H].norm().cpu()) for H in HORIZONS}})
            agg={}
            for src in ("C","O"):
                agg[src]={}
                for ph in PHASES:
                    rr=[r for r in rows if r["source"]==src and r["phase"]==ph]
                    agg[src][str(ph)]={
                        "gae_vs_mc":{str(H):{"mean":float(np.mean([r["gae_vs_mc"][str(H)] for r in rr])),"std":float(np.std([r["gae_vs_mc"][str(H)] for r in rr]))} for H in HORIZONS},
                        "mc_8_vs_16":{"mean":float(np.mean([r["mc_8_vs_16"] for r in rr])),"std":float(np.std([r["mc_8_vs_16"] for r in rr]))},
                        "mc_8_vs_32":{"mean":float(np.mean([r["mc_8_vs_32"] for r in rr])),"std":float(np.std([r["mc_8_vs_32"] for r in rr]))},
                        "mc_16_vs_32":{"mean":float(np.mean([r["mc_16_vs_32"] for r in rr])),"std":float(np.std([r["mc_16_vs_32"] for r in rr]))},
                        "gae_norm":{"mean":float(np.mean([r["gae_norm"] for r in rr])),"std":float(np.std([r["gae_norm"] for r in rr]))},
                        "mc_norms":{str(H):{"mean":float(np.mean([r["mc_norms"][str(H)] for r in rr])),"std":float(np.std([r["mc_norms"][str(H)] for r in rr]))} for H in HORIZONS}}
            srcfx={}
            for ph in PHASES:
                srcfx[str(ph)]={}
                for H in HORIZONS:
                    srcfx[str(ph)][str(H)]={"O_minus_C_gae_mc_cos":agg["O"][str(ph)]["gae_vs_mc"][str(H)]["mean"]-agg["C"][str(ph)]["gae_vs_mc"][str(H)]["mean"]}
            report={"schema":"v2b_horizon_visitation_credit_audit_crn_v2","measurement_only":True,"optimizer_steps":0,
                    "common_random_numbers":True,"nested_horizons":True,"checkpoint":str(a.checkpoint.relative_to(ROOT)),
                    "state_sources":["C","O"],"horizons":list(HORIZONS),"phases":list(PHASES),"seeds":list(SEEDS),
                    "aggregate":agg,"source_effect":srcfx,"raw":rows}
            out=a.output_dir/"horizon_visitation_crn_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps({"aggregate":agg,"source_effect":srcfx},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_lambda_horizon_audit():
    """Run former v2b_lambda_horizon_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_lambda_horizon_audit-2026-09-23"
    
    PREFS={"O":np.array([.1,.1,.7,.1],np.float32),"C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;NENV=8;MAXH=32
    LAMBDAS=(0.0,0.5,0.8,0.95,1.0)
    NSTEPS=(1,2,4,8,16,32)
    PHASES=(8,20,36,52)
    SEEDS=(920001,920002,920003,920004)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1)
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def collect_source(env,m,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+123000)
        acts=[]
        with torch.no_grad():
            for _ in range(64):
                a,_,_=m.act_with_preference_latent(cur,w);acts.append(a.clone())
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        return acts
    
    def restore(env,seed,phase,acts):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for k in range(phase):
            nxt,_,_,_,_=env.step(acts[k]);cur=ot(nxt).cuda()
        return cur
    
    def continuation(env,m,mgr,state,w,noise_seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        torch.manual_seed(noise_seed);cur=state
        U=[];MU=[];SD=[];R=[];D=[];V=[]
        with torch.no_grad():
            for _ in range(MAXH):
                V.append(m.value_with_preference(cur,w))
                feat=m._actor_features_v2a(cur,w);mu=m.actor_mean(feat);sd=m.log_std.exp().expand_as(mu)
                dist=torch.distributions.Normal(mu,sd);u=dist.sample();a=torch.tanh(u)*m.ACTION_CLIP
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());U.append(u);MU.append(mu);SD.append(sd);cur=ot(nxt).cuda()
            nextv=m.value_with_preference(cur,w)
        return torch.stack(U),torch.stack(MU),torch.stack(SD),torch.stack(R),torch.stack(D).bool(),torch.stack(V),nextv
    
    def gae_adv(reward,value,nextv,done,lam):
        last=torch.zeros_like(nextv);adv=torch.zeros_like(reward)
        for t in range(MAXH-1,-1,-1):
            boot=nextv if t==MAXH-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*lam*nt*last;adv[t]=last
        return adv
    
    def nstep_adv(reward,value,nextv,done,n):
        # bootstrapped n-step advantage at every t, truncated by MAXH boundary / termination.
        T=reward.shape[0];out=torch.zeros_like(reward)
        for t in range(T):
            ret=torch.zeros_like(value[t]);disc=torch.ones_like(value[t])
            alive=torch.ones((value.shape[1],1),device=value.device,dtype=value.dtype)
            end=min(T,t+n)
            for k in range(t,end):
                ret=ret+disc*alive*reward[k]
                nt=(~done[k]).to(value.dtype).unsqueeze(-1)
                alive=alive*nt;disc=disc*G
            if end<T:
                boot=value[end]
            else:
                boot=nextv
            ret=ret+disc*alive*boot
            out[t]=ret-value[t]
        return out
    
    def mc32(reward,done):
        r=reward[:,:,2];ret=torch.zeros_like(r);run=torch.zeros_like(r[-1])
        for t in range(MAXH-1,-1,-1):
            run=r[t]+G*run*(~done[t]).to(r.dtype);ret[t]=run
        return ret
    
    def grad_from_weight(score,weight):
        # center scalar weights globally, preserving score-function direction while reducing baseline noise
        x=weight-weight.mean()
        return (score*x.unsqueeze(-1)).mean((0,1))
    
    def snr_from_env_contrib(score,weight):
        x=weight-weight.mean()
        # aggregate time per env -> one vector contribution per env; norm(mean)/RMS deviation
        c=(score*x.unsqueeze(-1)).mean(0)  # [E,A]
        mean=c.mean(0);dev=c-mean
        rms=torch.sqrt((dev*dev).mean())
        return float(mean.norm()/(rms+1e-12))
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
    
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            for src in ("C","O"):
                for seed in SEEDS:
                    acts=collect_source(env,m,src,seed)
                    for ph in PHASES:
                        st=restore(env,seed,ph,acts)
                        w=torch.tensor(PREFS["O"],device="cuda").repeat(NENV,1)
                        U,MU,SD,R,D,V,NV=continuation(env,m,mgr,st,w,seed+ph*1000+777)
                        score=(U-MU)/(SD*SD)
                        mc=mc32(R,D);gmc=grad_from_weight(score,mc)
                        lamdata={};ndata={}
                        for lam in LAMBDAS:
                            adv=gae_adv(R,V,NV,D,lam)[:,:,2]
                            g=grad_from_weight(score,adv)
                            lamdata[str(lam)]={"cos_mc32":cos(g,gmc),"norm":float(g.norm().cpu()),
                                               "norm_ratio_mc32":float(g.norm()/(gmc.norm()+1e-12)),
                                               "snr":snr_from_env_contrib(score,adv)}
                        for n in NSTEPS:
                            adv=nstep_adv(R,V,NV,D,n)[:,:,2]
                            g=grad_from_weight(score,adv)
                            ndata[str(n)]={"cos_mc32":cos(g,gmc),"norm":float(g.norm().cpu()),
                                           "norm_ratio_mc32":float(g.norm()/(gmc.norm()+1e-12)),
                                           "snr":snr_from_env_contrib(score,adv)}
                        rows.append({"source":src,"seed":seed,"phase":ph,
                                     "mc32_norm":float(gmc.norm().cpu()),"lambda":lamdata,"nstep":ndata})
            def stats(vals):return {"mean":float(np.mean(vals)),"std":float(np.std(vals))}
            agg={}
            for src in ("C","O"):
                agg[src]={}
                for ph in PHASES:
                    rr=[r for r in rows if r["source"]==src and r["phase"]==ph]
                    agg[src][str(ph)]={
                      "mc32_norm":stats([r["mc32_norm"] for r in rr]),
                      "lambda":{str(l):{k:stats([r["lambda"][str(l)][k] for r in rr]) for k in ("cos_mc32","norm_ratio_mc32","snr")} for l in LAMBDAS},
                      "nstep":{str(n):{k:stats([r["nstep"][str(n)][k] for r in rr]) for k in ("cos_mc32","norm_ratio_mc32","snr")} for n in NSTEPS}}
            # pooled summary across all 8 source×phase cells and 4 seeds
            pooled={"lambda":{},"nstep":{}}
            for l in LAMBDAS:
                vals=[r["lambda"][str(l)] for r in rows]
                pooled["lambda"][str(l)]={k:stats([v[k] for v in vals]) for k in ("cos_mc32","norm_ratio_mc32","snr")}
            for n in NSTEPS:
                vals=[r["nstep"][str(n)] for r in rows]
                pooled["nstep"][str(n)]={k:stats([v[k] for v in vals]) for k in ("cos_mc32","norm_ratio_mc32","snr")}
            report={"schema":"v2b_lambda_horizon_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "common_random_numbers":True,"nested_32_step_continuation":True,
                    "checkpoint":str(a.checkpoint.relative_to(ROOT)),"objective":"Orientation",
                    "lambdas":list(LAMBDAS),"nsteps":list(NSTEPS),"sources":["C","O"],"phases":list(PHASES),"seeds":list(SEEDS),
                    "aggregate":agg,"pooled":pooled,"raw":rows}
            out=a.output_dir/"lambda_horizon_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps(pooled,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_lambda_horizon_control_audit():
    """Run former v2b_lambda_horizon_control_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_lambda_horizon_audit-2026-09-23"
    
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;NENV=8;MAXH=32
    LAMBDAS=(0.0,0.5,0.8,0.95,1.0)
    NSTEPS=(1,2,4,8,16,32)
    PHASES=(8,20,36,52)
    SEEDS=(920001,920002,920003,920004)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1)
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def collect_source(env,m,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+123000)
        acts=[]
        with torch.no_grad():
            for _ in range(64):
                a,_,_=m.act_with_preference_latent(cur,w);acts.append(a.clone())
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        return acts
    
    def restore(env,seed,phase,acts):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for k in range(phase):
            nxt,_,_,_,_=env.step(acts[k]);cur=ot(nxt).cuda()
        return cur
    
    def continuation(env,m,mgr,state,w,noise_seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        torch.manual_seed(noise_seed);cur=state
        U=[];MU=[];SD=[];R=[];D=[];V=[]
        with torch.no_grad():
            for _ in range(MAXH):
                V.append(m.value_with_preference(cur,w))
                feat=m._actor_features_v2a(cur,w);mu=m.actor_mean(feat);sd=m.log_std.exp().expand_as(mu)
                dist=torch.distributions.Normal(mu,sd);u=dist.sample();a=torch.tanh(u)*m.ACTION_CLIP
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());U.append(u);MU.append(mu);SD.append(sd);cur=ot(nxt).cuda()
            nextv=m.value_with_preference(cur,w)
        return torch.stack(U),torch.stack(MU),torch.stack(SD),torch.stack(R),torch.stack(D).bool(),torch.stack(V),nextv
    
    def gae_adv(reward,value,nextv,done,lam):
        last=torch.zeros_like(nextv);adv=torch.zeros_like(reward)
        for t in range(MAXH-1,-1,-1):
            boot=nextv if t==MAXH-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*lam*nt*last;adv[t]=last
        return adv
    
    def nstep_adv(reward,value,nextv,done,n):
        # bootstrapped n-step advantage at every t, truncated by MAXH boundary / termination.
        T=reward.shape[0];out=torch.zeros_like(reward)
        for t in range(T):
            ret=torch.zeros_like(value[t]);disc=torch.ones_like(value[t])
            alive=torch.ones((value.shape[1],1),device=value.device,dtype=value.dtype)
            end=min(T,t+n)
            for k in range(t,end):
                ret=ret+disc*alive*reward[k]
                nt=(~done[k]).to(value.dtype).unsqueeze(-1)
                alive=alive*nt;disc=disc*G
            if end<T:
                boot=value[end]
            else:
                boot=nextv
            ret=ret+disc*alive*boot
            out[t]=ret-value[t]
        return out
    
    def mc32(reward,done,obj_idx):
        r=reward[:,:,obj_idx];ret=torch.zeros_like(r);run=torch.zeros_like(r[-1])
        for t in range(MAXH-1,-1,-1):
            run=r[t]+G*run*(~done[t]).to(r.dtype);ret[t]=run
        return ret
    
    def grad_from_weight(score,weight):
        # center scalar weights globally, preserving score-function direction while reducing baseline noise
        x=weight-weight.mean()
        return (score*x.unsqueeze(-1)).mean((0,1))
    
    def snr_from_env_contrib(score,weight):
        x=weight-weight.mean()
        # aggregate time per env -> one vector contribution per env; norm(mean)/RMS deviation
        c=(score*x.unsqueeze(-1)).mean(0)  # [E,A]
        mean=c.mean(0);dev=c-mean
        rms=torch.sqrt((dev*dev).mean())
        return float(mean.norm()/(rms+1e-12))
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        ap.add_argument("--objective",choices=["A","S"],required=True)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
    
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            for src in ("C",a.objective):
                for seed in SEEDS:
                    acts=collect_source(env,m,src,seed)
                    for ph in PHASES:
                        st=restore(env,seed,ph,acts)
                        w=torch.tensor(PREFS[a.objective],device="cuda").repeat(NENV,1)
                        U,MU,SD,R,D,V,NV=continuation(env,m,mgr,st,w,seed+ph*1000+777)
                        score=(U-MU)/(SD*SD)
                        obj_idx={"A":1,"S":3}[a.objective]
                        mc=mc32(R,D,obj_idx);gmc=grad_from_weight(score,mc)
                        lamdata={};ndata={}
                        for lam in LAMBDAS:
                            adv=gae_adv(R,V,NV,D,lam)[:,:,obj_idx]
                            g=grad_from_weight(score,adv)
                            lamdata[str(lam)]={"cos_mc32":cos(g,gmc),"norm":float(g.norm().cpu()),
                                               "norm_ratio_mc32":float(g.norm()/(gmc.norm()+1e-12)),
                                               "snr":snr_from_env_contrib(score,adv)}
                        for n in NSTEPS:
                            adv=nstep_adv(R,V,NV,D,n)[:,:,obj_idx]
                            g=grad_from_weight(score,adv)
                            ndata[str(n)]={"cos_mc32":cos(g,gmc),"norm":float(g.norm().cpu()),
                                           "norm_ratio_mc32":float(g.norm()/(gmc.norm()+1e-12)),
                                           "snr":snr_from_env_contrib(score,adv)}
                        rows.append({"source":src,"seed":seed,"phase":ph,
                                     "mc32_norm":float(gmc.norm().cpu()),"lambda":lamdata,"nstep":ndata})
            def stats(vals):return {"mean":float(np.mean(vals)),"std":float(np.std(vals))}
            agg={}
            for src in ("C",a.objective):
                agg[src]={}
                for ph in PHASES:
                    rr=[r for r in rows if r["source"]==src and r["phase"]==ph]
                    agg[src][str(ph)]={
                      "mc32_norm":stats([r["mc32_norm"] for r in rr]),
                      "lambda":{str(l):{k:stats([r["lambda"][str(l)][k] for r in rr]) for k in ("cos_mc32","norm_ratio_mc32","snr")} for l in LAMBDAS},
                      "nstep":{str(n):{k:stats([r["nstep"][str(n)][k] for r in rr]) for k in ("cos_mc32","norm_ratio_mc32","snr")} for n in NSTEPS}}
            # pooled summary across all 8 source×phase cells and 4 seeds
            pooled={"lambda":{},"nstep":{}}
            for l in LAMBDAS:
                vals=[r["lambda"][str(l)] for r in rows]
                pooled["lambda"][str(l)]={k:stats([v[k] for v in vals]) for k in ("cos_mc32","norm_ratio_mc32","snr")}
            for n in NSTEPS:
                vals=[r["nstep"][str(n)] for r in rows]
                pooled["nstep"][str(n)]={k:stats([v[k] for v in vals]) for k in ("cos_mc32","norm_ratio_mc32","snr")}
            report={"schema":"v2b_lambda_horizon_control_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "common_random_numbers":True,"nested_32_step_continuation":True,
                    "checkpoint":str(a.checkpoint.relative_to(ROOT)),"objective":a.objective,
                    "lambdas":list(LAMBDAS),"nsteps":list(NSTEPS),"sources":["C",a.objective],"phases":list(PHASES),"seeds":list(SEEDS),
                    "aggregate":agg,"pooled":pooled,"raw":rows}
            out=a.output_dir/"lambda_horizon_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps(pooled,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_per_objective_temporal_scale_audit():
    """Run former v2b_per_objective_temporal_scale_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_lambda_horizon_audit-2026-09-23"
    
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;NENV=8;MAXH=32
    LAMBDAS=(0.0,0.5,0.8,0.9,0.95,0.98,1.0)
    NSTEPS=(1,2,4,8,16,32)
    PHASES=(8,20,36,52)
    SEEDS=(920001,920002,920003,920004)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1)
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def collect_source(env,m,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+123000)
        acts=[]
        with torch.no_grad():
            for _ in range(64):
                a,_,_=m.act_with_preference_latent(cur,w);acts.append(a.clone())
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        return acts
    
    def restore(env,seed,phase,acts):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for k in range(phase):
            nxt,_,_,_,_=env.step(acts[k]);cur=ot(nxt).cuda()
        return cur
    
    def continuation(env,m,mgr,state,w,noise_seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        torch.manual_seed(noise_seed);cur=state
        U=[];MU=[];SD=[];R=[];D=[];V=[]
        with torch.no_grad():
            for _ in range(MAXH):
                V.append(m.value_with_preference(cur,w))
                feat=m._actor_features_v2a(cur,w);mu=m.actor_mean(feat);sd=m.log_std.exp().expand_as(mu)
                dist=torch.distributions.Normal(mu,sd);u=dist.sample();a=torch.tanh(u)*m.ACTION_CLIP
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());U.append(u);MU.append(mu);SD.append(sd);cur=ot(nxt).cuda()
            nextv=m.value_with_preference(cur,w)
        return torch.stack(U),torch.stack(MU),torch.stack(SD),torch.stack(R),torch.stack(D).bool(),torch.stack(V),nextv
    
    def gae_adv(reward,value,nextv,done,lam):
        last=torch.zeros_like(nextv);adv=torch.zeros_like(reward)
        for t in range(MAXH-1,-1,-1):
            boot=nextv if t==MAXH-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*lam*nt*last;adv[t]=last
        return adv
    
    def nstep_adv(reward,value,nextv,done,n):
        # bootstrapped n-step advantage at every t, truncated by MAXH boundary / termination.
        T=reward.shape[0];out=torch.zeros_like(reward)
        for t in range(T):
            ret=torch.zeros_like(value[t]);disc=torch.ones_like(value[t])
            alive=torch.ones((value.shape[1],1),device=value.device,dtype=value.dtype)
            end=min(T,t+n)
            for k in range(t,end):
                ret=ret+disc*alive*reward[k]
                nt=(~done[k]).to(value.dtype).unsqueeze(-1)
                alive=alive*nt;disc=disc*G
            if end<T:
                boot=value[end]
            else:
                boot=nextv
            ret=ret+disc*alive*boot
            out[t]=ret-value[t]
        return out
    
    def mc32(reward,done,obj_idx):
        r=reward[:,:,obj_idx];ret=torch.zeros_like(r);run=torch.zeros_like(r[-1])
        for t in range(MAXH-1,-1,-1):
            run=r[t]+G*run*(~done[t]).to(r.dtype);ret[t]=run
        return ret
    
    def grad_from_weight(score,weight):
        # center scalar weights globally, preserving score-function direction while reducing baseline noise
        x=weight-weight.mean()
        return (score*x.unsqueeze(-1)).mean((0,1))
    
    def snr_from_env_contrib(score,weight):
        x=weight-weight.mean()
        # aggregate time per env -> one vector contribution per env; norm(mean)/RMS deviation
        c=(score*x.unsqueeze(-1)).mean(0)  # [E,A]
        mean=c.mean(0);dev=c-mean
        rms=torch.sqrt((dev*dev).mean())
        return float(mean.norm()/(rms+1e-12))
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
    
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            for objective in ("T","A","O","S"):
                obj_idx={"T":0,"A":1,"O":2,"S":3}[objective]
                for src in ("C",objective):
                    for seed in SEEDS:
                        acts=collect_source(env,m,src,seed)
                        for ph in PHASES:
                            st=restore(env,seed,ph,acts)
                            w=torch.tensor(PREFS[objective],device="cuda").repeat(NENV,1)
                            U,MU,SD,R,D,V,NV=continuation(env,m,mgr,st,w,seed+ph*1000+777)
                            score=(U-MU)/(SD*SD)
                            mc=mc32(R,D,obj_idx);gmc=grad_from_weight(score,mc)
                            lamdata={};ndata={}
                            for lam in LAMBDAS:
                                adv=gae_adv(R,V,NV,D,lam)[:,:,obj_idx]
                                g=grad_from_weight(score,adv)
                                lamdata[str(lam)]={"cos_mc32":cos(g,gmc),"norm":float(g.norm().cpu()),
                                                   "norm_ratio_mc32":float(g.norm()/(gmc.norm()+1e-12)),
                                                   "snr":snr_from_env_contrib(score,adv)}
                            for n in NSTEPS:
                                adv=nstep_adv(R,V,NV,D,n)[:,:,obj_idx]
                                g=grad_from_weight(score,adv)
                                ndata[str(n)]={"cos_mc32":cos(g,gmc),"norm":float(g.norm().cpu()),
                                               "norm_ratio_mc32":float(g.norm()/(gmc.norm()+1e-12)),
                                               "snr":snr_from_env_contrib(score,adv)}
                            rows.append({"objective":objective,"source":src,"seed":seed,"phase":ph,
                                         "mc32_norm":float(gmc.norm().cpu()),"lambda":lamdata,"nstep":ndata})
            def stats(vals):return {"mean":float(np.mean(vals)),"std":float(np.std(vals))}
            agg={};pooled={}
            for objective in ("T","A","O","S"):
                agg[objective]={};pooled[objective]={"lambda":{},"nstep":{}}
                for src in ("C",objective):
                    agg[objective][src]={}
                    for ph in PHASES:
                        rr=[r for r in rows if r["objective"]==objective and r["source"]==src and r["phase"]==ph]
                        agg[objective][src][str(ph)]={
                          "mc32_norm":stats([r["mc32_norm"] for r in rr]),
                          "lambda":{str(l):{k:stats([r["lambda"][str(l)][k] for r in rr]) for k in ("cos_mc32","norm_ratio_mc32","snr")} for l in LAMBDAS},
                          "nstep":{str(n):{k:stats([r["nstep"][str(n)][k] for r in rr]) for k in ("cos_mc32","norm_ratio_mc32","snr")} for n in NSTEPS}}
                objrows=[r for r in rows if r["objective"]==objective]
                for l in LAMBDAS:
                    vals=[r["lambda"][str(l)] for r in objrows]
                    pooled[objective]["lambda"][str(l)]={k:stats([v[k] for v in vals]) for k in ("cos_mc32","norm_ratio_mc32","snr")}
                for n in NSTEPS:
                    vals=[r["nstep"][str(n)] for r in objrows]
                    pooled[objective]["nstep"][str(n)]={k:stats([v[k] for v in vals]) for k in ("cos_mc32","norm_ratio_mc32","snr")}
            report={"schema":"v2b_per_objective_temporal_scale_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "common_random_numbers":True,"nested_32_step_continuation":True,
                    "checkpoint":str(a.checkpoint.relative_to(ROOT)),"objectives":["T","A","O","S"],
                    "lambdas":list(LAMBDAS),"nsteps":list(NSTEPS),"source_rule":"center + corresponding heavy visitation","phases":list(PHASES),"seeds":list(SEEDS),
                    "aggregate":agg,"pooled":pooled,"raw":rows}
            out=a.output_dir/"lambda_horizon_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps(pooled,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_visitation_distribution_geometry_audit():
    """Run former v2b_visitation_distribution_geometry_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99;LAM=1.0;NENV=8;SOURCE_H=64;CONT_H=32
    PHASES=(8,20,36,52)
    SEEDS=(950001,950002,950003,950004)
    SOURCE_CKPTS={
    "u50":ROOT/"runs/v2b_adam_continuous-2026-09-24/model_50.pt",
    "u75":ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"}
    EVAL_CKPTS=SOURCE_CKPTS
    OUT=ROOT/"runs/v2b_visitation_distribution_geometry_audit-2026-09-24"
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1);den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def flat_grads(loss,params):
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])
    def pairwise(grads):
        out={}
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:out[f"{a}-{b}"]=cos(grads[a],grads[b])
        return out
    def groups(m):
        return {
          "all_actor":[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")],
          "shared_body":[p for n,p in m.named_parameters() if n.startswith("actor_body")],
          "direct_input":[m.actor_body[0].weight],
          "embedding":[p for n,p in m.named_parameters() if n.startswith("preference_embedding")],
          "film":[p for n,p in m.named_parameters() if n.startswith("preference_film")],
          "actor_head":[p for n,p in m.named_parameters() if n.startswith("actor_mean")],
        }
    
    def source_rollout(env,m,pref,seed):
        w=torch.tensor(PREFS[pref],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+10101)
        acts=[]
        with torch.no_grad():
            for _ in range(SOURCE_H):
                a,_,_=m.act_with_preference_latent(cur,w)
                acts.append(a.detach().clone())
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        return acts
    
    def restore_to_phase(env,seed,acts,phase):
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        for k in range(phase):
            nxt,_,_,_,_=env.step(acts[k]);cur=ot(nxt).cuda()
        return cur
    
    def evaluator_continuation(env,m,mgr,state,pref,noise_seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[pref],device="cuda").repeat(NENV,1)
        torch.manual_seed(noise_seed)
        cur=state
        OBS=[];U=[];R=[];D=[];V=[]
        with torch.no_grad():
            for _ in range(CONT_H):
                V.append(m.value_with_preference(cur,w))
                feat=m._actor_features_v2a(cur,w)
                mu=m.actor_mean(feat);sd=m.log_std.exp().expand_as(mu)
                eps=torch.randn_like(mu)
                u=mu+sd*eps
                a=torch.tanh(u)*m.ACTION_CLIP
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                OBS.append(cur.detach().clone());U.append(u.detach().clone())
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
            NV=m.value_with_preference(cur,w)
        return {
          "obs":torch.stack(OBS),"u":torch.stack(U),"R":torch.stack(R),
          "D":torch.stack(D).bool(),"V":torch.stack(V),"NV":NV,"w_env":w}
    
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(CONT_H-1,-1,-1):
            boot=nextv if t==CONT_H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    
    def geometry(m,data,pref):
        adv=gae(data["R"],data["V"],data["NV"],data["D"]).reshape(-1,4).detach()
        obs=data["obs"].reshape(-1,data["obs"].shape[-1])
        u=data["u"].reshape(-1,data["u"].shape[-1])
        w=data["w_env"].repeat(CONT_H,1)
        # Evaluator-policy score at its own sampled actions. No PPO old-policy ratio.
        logp=m.logp_from_pre_tanh_with_preference(obs,w,u)
        losses={lab:-(logp*adv[:,j]).mean() for j,lab in enumerate(ORDER)}
        weights=PREFS[pref]
        out={}
        for gn,ps in groups(m).items():
            raw={lab:flat_grads(losses[lab],ps) for lab in ORDER}
            wg={lab:4*float(weights[j])*raw[lab] for j,lab in enumerate(ORDER)}
            combined=sum(wg.values())
            norms={lab:float(wg[lab].norm().cpu()) for lab in ORDER}
            den=sum(norms.values())+1e-12
            heavy=pref;others=sum((wg[x] for x in ORDER if x!=heavy),torch.zeros_like(combined))
            out[gn]={
              "weighted_norm":norms,
              "weighted_share":{lab:norms[lab]/den for lab in ORDER},
              "raw_pairwise_cos":pairwise(raw),
              "combined_norm":float(combined.norm().cpu()),
              "combined_cos":{lab:cos(combined,raw[lab]) for lab in ORDER},
              "combined_cos_heavy":cos(combined,raw[heavy]),
              "override_ratio":float(others.norm()/(wg[heavy].norm()+1e-12))}
        return out
    
    def visitation_summary(state):
        # Observation-level summary of the frozen source state set.
        x=state.detach()
        return {
          "obs_mean_norm":float(x.mean(0).norm().cpu()),
          "obs_std_mean":float(x.std(0).mean().cpu()),
          "base_lin_vel_mean":x[:,0:3].mean(0).cpu().tolist(),
          "base_ang_vel_mean":x[:,3:6].mean(0).cpu().tolist(),
          "projected_gravity_mean":x[:,6:9].mean(0).cpu().tolist(),
          "command_mean":x[:,9:12].mean(0).cpu().tolist(),
          "joint_pos_mean_norm":float(x[:,12:24].mean(0).norm().cpu()),
          "joint_vel_mean_norm":float(x[:,24:36].mean(0).norm().cpu()),
          "prev_action_mean_norm":float(x[:,36:48].mean(0).norm().cpu())}
    
    def aggregate(vals):
        v0=vals[0]
        if isinstance(v0,dict):return {k:aggregate([v[k] for v in vals]) for k in v0}
        if isinstance(v0,(int,float,np.floating)):
            a=np.asarray(vals,float);return {"mean":float(a.mean()),"std":float(a.std())}
        if isinstance(v0,list) and all(isinstance(x,(int,float)) for x in v0):
            a=np.asarray(vals,float);return {"mean":np.mean(a,axis=0).tolist(),"std":np.std(a,axis=0).tolist()}
        return v0
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            models={}
            for lab,p in {**SOURCE_CKPTS,**{f"eval_{k}":v for k,v in EVAL_CKPTS.items()}}.items():
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(p,map_location="cuda",weights_only=False)["model"]);m.eval()
                models[lab]=m
    
            raw=[]
            # For each evaluator policy, compare only the source visitation distribution.
            for eval_stage in ("u50","u75"):
                evaluator=models[f"eval_{eval_stage}"]
                for pref in ORDER:
                    for seed in SEEDS:
                        source_actions={src:source_rollout(env,models[src],pref,seed) for src in ("u50","u75")}
                        for phase in PHASES:
                            for src in ("u50","u75"):
                                state=restore_to_phase(env,seed,source_actions[src],phase)
                                vsummary=visitation_summary(state)
                                # Same evaluator/noise seed for u50-vs-u75 source pair.
                                data=evaluator_continuation(env,evaluator,mgr,state,pref,seed+phase*1000+333)
                                geom=geometry(evaluator,data,pref)
                                raw.append({"eval_stage":eval_stage,"pref":pref,"seed":seed,"phase":phase,
                                            "source":src,"visitation":vsummary,"geometry":geom})
    
            agg={}
            delta={}
            for ev in ("u50","u75"):
                agg[ev]={};delta[ev]={}
                for pref in ORDER:
                    agg[ev][pref]={};delta[ev][pref]={}
                    for phase in PHASES:
                        agg[ev][pref][str(phase)]={}
                        rows_by={}
                        for src in ("u50","u75"):
                            rr=[r for r in raw if r["eval_stage"]==ev and r["pref"]==pref and r["phase"]==phase and r["source"]==src]
                            rows_by[src]=rr
                            agg[ev][pref][str(phase)][src]=aggregate([{"visitation":r["visitation"],"geometry":r["geometry"]} for r in rr])
                        # paired seed delta u75-source minus u50-source
                        ds=[]
                        for seed in SEEDS:
                            a=next(r for r in rows_by["u50"] if r["seed"]==seed)
                            b=next(r for r in rows_by["u75"] if r["seed"]==seed)
                            dd={"groups":{}}
                            for gn in a["geometry"]:
                                ga=a["geometry"][gn];gb=b["geometry"][gn]
                                dd["groups"][gn]={
                                  "share_delta":{lab:gb["weighted_share"][lab]-ga["weighted_share"][lab] for lab in ORDER},
                                  "combined_cos_delta":{lab:gb["combined_cos"][lab]-ga["combined_cos"][lab] for lab in ORDER},
                                  "combined_cos_heavy_delta":gb["combined_cos_heavy"]-ga["combined_cos_heavy"],
                                  "override_ratio_delta":gb["override_ratio"]-ga["override_ratio"],
                                  "combined_norm_ratio":gb["combined_norm"]/(ga["combined_norm"]+1e-12),
                                  "pairwise_cos_delta":{k:gb["raw_pairwise_cos"][k]-ga["raw_pairwise_cos"][k] for k in ga["raw_pairwise_cos"]}}
                            ds.append(dd)
                        delta[ev][pref][str(phase)]=aggregate(ds)
    
            report={"schema":"v2b_visitation_distribution_geometry_audit_v1","measurement_only":True,
                    "optimizer_steps":0,"lambda":LAM,
                    "source_policies":{k:str(v.relative_to(ROOT)) for k,v in SOURCE_CKPTS.items()},
                    "evaluator_policies":{k:str(v.relative_to(ROOT)) for k,v in EVAL_CKPTS.items()},
                    "same_evaluator_within_source_comparison":True,"common_random_numbers":True,
                    "preferences":{k:v.tolist() for k,v in PREFS.items()},"phases":list(PHASES),"seeds":list(SEEDS),
                    "aggregate":agg,"paired_source_delta_u75_minus_u50":delta,"raw":raw}
            out=args.output_dir/"visitation_geometry_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({
                "status":"FROZEN_BY_HASH","report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),
                "source_checkpoint_sha256":{k:sha(v) for k,v in SOURCE_CKPTS.items()}},indent=2)+"\n")
            # compact stdout
            compact={}
            for ev in ("u50","u75"):
                compact[ev]={}
                for pref in ORDER:
                    compact[ev][pref]={}
                    for phase in PHASES:
                        x=delta[ev][pref][str(phase)]["groups"]["all_actor"]
                        compact[ev][pref][str(phase)]={
                          "heavy_cos_delta":x["combined_cos_heavy_delta"],
                          "share_delta":x["share_delta"],
                          "override_delta":x["override_ratio_delta"],
                          "norm_ratio":x["combined_norm_ratio"],
                          "pairwise_delta":x["pairwise_cos_delta"]}
            print(json.dumps(compact,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "v2b_competence_floor_endpoint_timeline": run_v2b_competence_floor_endpoint_timeline,
    "v2b_competence_floor_maxmin_audit": run_v2b_competence_floor_maxmin_audit,
    "v2b_competence_floor_paired_pilot": run_v2b_competence_floor_paired_pilot,
    "v2b_competence_floor_semantic_aggregate": run_v2b_competence_floor_semantic_aggregate,
    "v2b_horizon_visitation_credit_audit": run_v2b_horizon_visitation_credit_audit,
    "v2b_horizon_visitation_credit_audit_crn": run_v2b_horizon_visitation_credit_audit_crn,
    "v2b_lambda_horizon_audit": run_v2b_lambda_horizon_audit,
    "v2b_lambda_horizon_control_audit": run_v2b_lambda_horizon_control_audit,
    "v2b_per_objective_temporal_scale_audit": run_v2b_per_objective_temporal_scale_audit,
    "v2b_visitation_distribution_geometry_audit": run_v2b_visitation_distribution_geometry_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
