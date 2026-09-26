#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib,json,math,sys
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
OUT=ROOT/'runs/trajectory_information_attribution_audit-2026-09-24'
NENV=8;STEPS=64;SUITES=4;TOL=1e-10
ORDER=('T','A','O','S');IDX={'T':0,'A':1,'O':2,'S':3}
PHYS={'T':'tracking_error','A':'ang_vel_xy','O':'tilt_deg','S':'action_rate'}
PREFS={
 'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),
 'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32),
 'C':np.array([.25,.25,.25,.25],np.float32)}

PATHS={
 'lambda095':[
  ('u10',ROOT/'runs/v2b_lambda095_pilot-2026-09-23/model_10.pt'),
  ('u25',ROOT/'runs/v2b_lambda095_pilot-2026-09-23/model_25.pt'),
  ('u50',ROOT/'runs/v2b_lambda095_pilot-2026-09-23/model_50.pt'),
  ('u75',ROOT/'runs/v2b_lambda095_pilot-2026-09-23/model_75.pt')],
 'lambda100':[
  ('u10',ROOT/'runs/v2b_lambda100_pilot-2026-09-23/model_10.pt'),
  ('u25',ROOT/'runs/v2b_lambda100_pilot-2026-09-23/model_25.pt'),
  ('u50',ROOT/'runs/v2b_lambda100_pilot-2026-09-23/model_50.pt'),
  ('u75',ROOT/'runs/v2b_lambda100_pilot-2026-09-23/model_75.pt')],
 'paired_control':[(str(s),ROOT/f'runs/v2b_competence_floor_paired_pilot-2026-09-24/control_model_{s}.pt') for s in (0,5,10,15,20,25)],
 'paired_floor':[(str(s),ROOT/f'runs/v2b_competence_floor_paired_pilot-2026-09-24/floor_model_{s}.pt') for s in (0,5,10,15,20,25)],
}
SEM_REPORTS={
 'lambda095':[
  ROOT/'runs/v2b_semantic_path_095_u10-2026-09-24/semantic_report.json',ROOT/'runs/v2b_semantic_path_095_u25-2026-09-24/semantic_report.json',
  ROOT/'runs/v2b_semantic_path_095_u50-2026-09-24/semantic_report.json',ROOT/'runs/v2b_lambda095_semantic-2026-09-23/semantic_report.json'],
 'lambda100':[
  ROOT/'runs/v2b_semantic_path_100_u10-2026-09-24/semantic_report.json',ROOT/'runs/v2b_semantic_path_100_u25-2026-09-24/semantic_report.json',
  ROOT/'runs/v2b_semantic_path_100_u50-2026-09-24/semantic_report.json',ROOT/'runs/v2b_lambda100_semantic-2026-09-23/semantic_report.json'],
 'paired_control':[ROOT/f'runs/v2b_competence_floor_semantic_path-2026-09-24/control_{s}/endpoint_report.json' for s in (0,5,10,15,20,25)],
 'paired_floor':[ROOT/f'runs/v2b_competence_floor_semantic_path-2026-09-24/floor_{s}/endpoint_report.json' for s in (0,5,10,15,20)] + [ROOT/'runs/v2b_competence_floor_final_floor-2026-09-24/semantic_report.json'],
}

def obs_tensor(x):
    if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def tilt_deg(q):
    _,x,y,_=[q[:,i] for i in range(4)]
    return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def checkpoint_model_hash(path):
    d=torch.load(path,map_location='cpu',weights_only=False)['model'];h=hashlib.sha256()
    for k in sorted(d):h.update(k.encode());h.update(d[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()

def trace(env,m,mgr,robot,w_np,seed):
    from talon_rl.rewards.objectives import normalized_objective_vector
    w=torch.tensor(w_np,device='cuda').repeat(NENV,1)
    cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
    obj=[];phys=[];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device='cuda')
    with torch.no_grad():
      for _ in range(STEPS):
        a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
        raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
        ov=normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt
        obj.append(np.mean(ov,axis=0))
        data=robot.data;cmd=env.unwrapped.command_manager.get_command('base_velocity')
        vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean());wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
        phys.append({'tracking_error':vx+wz,
                     'ang_vel_xy':float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                     'tilt_deg':float(tilt_deg(data.root_quat_w).mean()),
                     'action_rate':float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
        prev=a;cur=obs_tensor(nxt).cuda()
    return {'obj':np.asarray(obj,float),'phys':{k:np.asarray([x[k] for x in phys],float) for k in phys[0]}}

def rolling_worst(x,w=16):
    return float(min(np.mean(x[i:i+w]) for i in range(len(x)-w+1)))
def slope(x):
    t=np.arange(len(x),dtype=float);return float(np.polyfit(t,np.asarray(x,float),1)[0])
def flips(x):
    s=np.sign(np.asarray(x,float));s[np.abs(x)<=TOL]=0
    nz=s[s!=0]
    return int(np.sum(nz[1:]!=nz[:-1])) if len(nz)>1 else 0
def longest_negative(x):
    best=cur=0
    for v in x:
        if v<-TOL:cur+=1;best=max(best,cur)
        else:cur=0
    return int(best)
def min_prefix_mean(x):
    x=np.asarray(x,float);return float(np.min(np.cumsum(x)/np.arange(1,len(x)+1)))

def desc_axis(h,c,axis):
    j=IDX[axis];pk=PHYS[axis]
    rh=h['obj'][:,j];rc=c['obj'][:,j];ph=h['phys'][pk];pc=c['phys'][pk]
    ao=rh-rc;ap=pc-ph
    thirds=((0,21),(21,43),(43,64))
    d={
      'aggregate.J_heavy':float(np.mean(rh)),
      'aggregate.obj_adv_mean':float(np.mean(ao)),
      'aggregate.phys_adv_mean':float(np.mean(ap)),
      'temporal.obj_adv_early':float(np.mean(ao[:21])),
      'temporal.obj_adv_mid':float(np.mean(ao[21:43])),
      'temporal.obj_adv_late':float(np.mean(ao[43:])),
      'temporal.obj_adv_late_minus_early':float(np.mean(ao[43:])-np.mean(ao[:21])),
      'temporal.obj_adv_slope':slope(ao),
      'temporal.phys_adv_early':float(np.mean(ap[:21])),
      'temporal.phys_adv_mid':float(np.mean(ap[21:43])),
      'temporal.phys_adv_late':float(np.mean(ap[43:])),
      'temporal.phys_adv_late_minus_early':float(np.mean(ap[43:])-np.mean(ap[:21])),
      'temporal.phys_adv_slope':slope(ap),
      'tail.heavy_obj_std':float(np.std(rh)),
      'tail.heavy_obj_p10':float(np.percentile(rh,10)),
      'tail.heavy_phys_std':float(np.std(ph)),
      'tail.heavy_phys_p90_bad':float(-np.percentile(ph,90)),
      'tail.heavy_phys_p95_bad':float(-np.percentile(ph,95)),
      'tail.heavy_phys_max_bad':float(-np.max(ph)),
      'tail.obj_adv_std_bad':float(-np.std(ao)),
      'tail.phys_adv_std_bad':float(-np.std(ap)),
      'tail.obj_adv_worst16':rolling_worst(ao,16),
      'tail.phys_adv_worst16':rolling_worst(ap,16),
      'persistence.obj_positive_fraction':float(np.mean(ao>TOL)),
      'persistence.obj_negative_fraction_bad':float(-np.mean(ao<-TOL)),
      'persistence.obj_flip_count_bad':float(-flips(ao)),
      'persistence.obj_longest_negative_bad':float(-longest_negative(ao)),
      'persistence.obj_final16':float(np.mean(ao[-16:])),
      'persistence.obj_min_prefix_mean':min_prefix_mean(ao),
      'persistence.phys_positive_fraction':float(np.mean(ap>TOL)),
      'persistence.phys_negative_fraction_bad':float(-np.mean(ap<-TOL)),
      'persistence.phys_flip_count_bad':float(-flips(ap)),
      'persistence.phys_longest_negative_bad':float(-longest_negative(ap)),
      'persistence.phys_final16':float(np.mean(ap[-16:])),
      'persistence.phys_min_prefix_mean':min_prefix_mean(ap),
    }
    return d

def rankdata(x):
    x=np.asarray(x,float);o=np.argsort(x,kind='mergesort');r=np.empty(len(x),float);i=0
    while i<len(x):
      j=i+1
      while j<len(x) and x[o[j]]==x[o[i]]:j+=1
      r[o[i:j]]=(i+j-1)/2+1;i=j
    return r
def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<2 or np.std(x)<1e-15 or np.std(y)<1e-15:return float('nan')
    return float(np.corrcoef(x,y)[0,1])
def spearman(x,y):return corr(rankdata(x),rankdata(y))

def semantic_state(path):
    d=json.load(open(path));return {a:{'M_obj':float(d['endpoint'][a]['mean_objective_delta_vs_center']),
                                      'M_phys':float(-d['endpoint'][a]['mean_physical_delta_vs_center']),
                                      'PASS':bool(d['endpoint_pass'][a])} for a in ORDER}

def main():
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
    env=None
    try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
      env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
      mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene['robot']
      # dedupe checkpoints by model hash
      refs=[]
      for pname,seq in PATHS.items():
        for lab,p in seq: refs.append((pname,lab,p))
      hash_to_path={};refhash={}
      for pname,lab,p in refs:
        hh=checkpoint_model_hash(p);refhash[(pname,lab)]=hh;hash_to_path.setdefault(hh,p)
      print(f'UNIQUE_CHECKPOINTS {len(hash_to_path)} / REFERENCES {len(refs)}',flush=True)
      traces_by_hash={}
      for ci,(hh,p) in enumerate(hash_to_path.items(),1):
        print(f'CHECKPOINT {ci}/{len(hash_to_path)} {p.relative_to(ROOT)}',flush=True)
        m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);m.eval()
        suites=[]
        for suite in range(SUITES):
          seed=840001+suite;tt={}
          for lab in ('T','A','O','S','C'):
            tt[lab]=trace(env,m,mgr,robot,PREFS[lab],seed)
          suites.append(tt)
        traces_by_hash[hh]=suites
      # descriptors averaged across suites per checkpoint/axis
      states={}
      for pname,seq in PATHS.items():
        sems=[semantic_state(p) for p in SEM_REPORTS[pname]]
        arr=[]
        for idx,(lab,p) in enumerate(seq):
          suites=traces_by_hash[refhash[(pname,lab)]];axes={}
          for a in ORDER:
            ds=[desc_axis(s[a],s['C'],a) for s in suites]
            keys=ds[0].keys();avg={k:float(np.mean([d[k] for d in ds])) for k in keys}
            avg.update(sems[idx][a]);axes[a]=avg
          arr.append({'label':lab,'checkpoint':str(p.relative_to(ROOT)),'axes':axes})
        states[pname]=arr
      samples=[]
      for pname,arr in states.items():
        for i in range(len(arr)-1):
          for a in ORDER:
            aa=arr[i]['axes'][a];bb=arr[i+1]['axes'][a]
            changes={k:bb[k]-aa[k] for k in aa if k not in ('PASS','M_obj','M_phys')}
            samples.append({'path':pname,'from':arr[i]['label'],'to':arr[i+1]['label'],'axis':a,
                            'dM_obj':bb['M_obj']-aa['M_obj'],'dM_phys':bb['M_phys']-aa['M_phys'],
                            'pass_from':aa['PASS'],'pass_to':bb['PASS'],'changes':changes})
      features=sorted(next(iter(samples))['changes'])
      stats={}
      for f in features:
        vals=np.array([s['changes'][f] for s in samples])
        target='dM_phys' if ('.phys' in f or 'heavy_phys' in f) else 'dM_obj'
        y=np.array([s[target] for s in samples])
        stats[f]={'target':target,'pearson':corr(vals,y),'spearman':spearman(vals,y)}
        mask=(np.abs(vals)>TOL)&(np.abs(y)>TOL)
        stats[f]['sign_agreement']=float(np.mean(np.sign(vals[mask])==np.sign(y[mask]))) if mask.any() else float('nan')
        # consistency across paths
        perpath={}
        for p in PATHS:
          ss=[s for s in samples if s['path']==p];x=[s['changes'][f] for s in ss];yy=[s[target] for s in ss]
          perpath[p]=spearman(x,yy)
        stats[f]['per_path_spearman']=perpath
        stats[f]['paths_same_sign_as_overall']=int(sum(np.isfinite(v) and np.sign(v)==np.sign(stats[f]['spearman']) for v in perpath.values()))
        peraxis={}
        for a in ORDER:
          ss=[s for s in samples if s['axis']==a];x=[s['changes'][f] for s in ss];yy=[s[target] for s in ss]
          peraxis[a]=spearman(x,yy)
        stats[f]['per_axis_spearman']=peraxis
        stats[f]['axes_same_sign_as_overall']=int(sum(np.isfinite(v) and np.sign(v)==np.sign(stats[f]['spearman']) for v in peraxis.values()))
      # family summaries
      families={}
      for fam in ('aggregate','temporal','tail','persistence'):
        fs=[f for f in features if f.startswith(fam+'.')];absrho=[abs(stats[f]['spearman']) for f in fs if np.isfinite(stats[f]['spearman'])]
        families[fam]={'n':len(fs),'median_abs_spearman':float(np.median(absrho)),
                       'max_abs_spearman':float(np.max(absrho)),
                       'count_abs_spearman_ge_0p50':int(sum(v>=.5 for v in absrho)),
                       'top_features':sorted(fs,key=lambda f:abs(stats[f]['spearman']) if np.isfinite(stats[f]['spearman']) else -1,reverse=True)[:8]}
      base=families['aggregate']['median_abs_spearman']
      for fam in ('temporal','tail','persistence'):
        families[fam]['informative']=bool(families[fam]['count_abs_spearman_ge_0p50']>=2 and families[fam]['median_abs_spearman']>=base+.10)
      candidates=[]
      for f,v in stats.items():
        if not np.isfinite(v['spearman']):continue
        if abs(v['spearman'])>=.50 and (v['paths_same_sign_as_overall']>=3 or v['axes_same_sign_as_overall']>=3):
          candidates.append(f)
      # pass->fail focused table
      pf=[]
      focus=['aggregate.J_heavy','tail.obj_adv_worst16','tail.phys_adv_worst16','persistence.obj_positive_fraction',
             'persistence.phys_positive_fraction','persistence.obj_longest_negative_bad','persistence.phys_longest_negative_bad',
             'temporal.obj_adv_late_minus_early','temporal.phys_adv_late_minus_early']
      for s in samples:
        if s['pass_from'] and not s['pass_to']:
          pf.append({'path':s['path'],'from':s['from'],'to':s['to'],'axis':s['axis'],'dM_obj':s['dM_obj'],'dM_phys':s['dM_phys'],
                     'descriptor_changes':{f:s['changes'][f] for f in focus}})
      outcome='NO SIMPLE TRAJECTORY FAMILY DOMINATES'
      inf=[f for f in ('temporal','tail','persistence') if families[f]['informative']]
      if inf: outcome='INFORMATIVE FAMILIES: '+', '.join(inf)
      rep={'schema':'trajectory_information_attribution_audit_v1','status':outcome,'measurement_only':True,'optimizer_steps':0,
           'unique_checkpoints':len(hash_to_path),'references':len(refs),'transition_count':len(samples)//4,'sample_count':len(samples),
           'states':states,'feature_stats':stats,'family_summary':families,'candidate_missing_statistics':candidates,
           'pass_to_fail_attribution':pf,'samples':samples,
           'decision':{'reward_redesign_authorized':False,'outcome':outcome}}
      OUT.mkdir(parents=True,exist_ok=True);out=OUT/'trajectory_information_attribution_report.json';out.write_text(json.dumps(rep,indent=2)+'\n')
      prov={'status':'FROZEN_BY_HASH','decision':outcome,'report_sha256':sha(out),'script_sha256':sha(Path(__file__).resolve()),
            'contract_sha256':sha(ROOT/'docs/contracts/diagnostics/trajectory-information-attribution-contract.md'),
            'checkpoint_sha256':{str(p.relative_to(ROOT)):sha(p) for _,_,p in refs}}
      (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps(prov,indent=2)+'\n')
      compact={'status':outcome,'family_summary':families,'candidate_missing_statistics':candidates,
               'top15':[(f,stats[f]['target'],stats[f]['spearman'],stats[f]['sign_agreement'],stats[f]['paths_same_sign_as_overall'],stats[f]['axes_same_sign_as_overall']) for f in sorted(features,key=lambda f:abs(stats[f]['spearman']) if np.isfinite(stats[f]['spearman']) else -1,reverse=True)[:15]],
               'pass_to_fail_attribution':pf}
      print(json.dumps(compact,indent=2),flush=True)
    finally:
      if env is not None:env.close()
      app.close()
if __name__=='__main__':main()
