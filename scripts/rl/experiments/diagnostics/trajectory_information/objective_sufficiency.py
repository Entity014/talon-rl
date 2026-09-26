#!/usr/bin/env python3
from pathlib import Path
import json, math, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'runs/trajectory_objective_sufficiency_audit-2026-09-24'
AXES=('T','A','O','S')
IDX={'T':0,'A':1,'O':2,'S':3}
TOL=1e-10

PATHS={
 'lambda095':[
  ROOT/'runs/v2b_semantic_path_095_u10-2026-09-24/semantic_report.json',
  ROOT/'runs/v2b_semantic_path_095_u25-2026-09-24/semantic_report.json',
  ROOT/'runs/v2b_semantic_path_095_u50-2026-09-24/semantic_report.json',
  ROOT/'runs/v2b_lambda095_semantic-2026-09-23/semantic_report.json'],
 'lambda100':[
  ROOT/'runs/v2b_semantic_path_100_u10-2026-09-24/semantic_report.json',
  ROOT/'runs/v2b_semantic_path_100_u25-2026-09-24/semantic_report.json',
  ROOT/'runs/v2b_semantic_path_100_u50-2026-09-24/semantic_report.json',
  ROOT/'runs/v2b_lambda100_semantic-2026-09-23/semantic_report.json'],
 'paired_control':[
  ROOT/f'runs/v2b_competence_floor_semantic_path-2026-09-24/control_{s}/endpoint_report.json' for s in (0,5,10,15,20,25)],
 'paired_floor':[
  ROOT/f'runs/v2b_competence_floor_semantic_path-2026-09-24/floor_{s}/endpoint_report.json' for s in (0,5,10,15,20)] +
  [ROOT/'runs/v2b_competence_floor_final_floor-2026-09-24/semantic_report.json'],
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rankdata(x):
    x=np.asarray(x,float); order=np.argsort(x,kind='mergesort'); ranks=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        r=(i+j-1)/2+1
        ranks[order[i:j]]=r;i=j
    return ranks

def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<2 or np.std(x)<1e-15 or np.std(y)<1e-15:return float('nan')
    return float(np.corrcoef(x,y)[0,1])

def spearman(x,y):return corr(rankdata(x),rankdata(y))

def parse_report(path):
    d=json.load(open(path))
    rows=d.get('endpoint_rows',[])
    # index by suite,label. Both report formats use same row content.
    by={(int(r['suite']),r['label']):r for r in rows}
    endpoint=d['endpoint']; epass=d['endpoint_pass']
    out={}
    for a in AXES:
        j=IDX[a]
        heavy=[]
        for suite in range(4):
            heavy.append(float(by[(suite,a)]['normalized_objective_mean'][j]))
        out[a]={
          'J':float(np.mean(heavy)),
          'M_obj':float(endpoint[a]['mean_objective_delta_vs_center']),
          'M_phys':float(-endpoint[a]['mean_physical_delta_vs_center']),
          'PASS':bool(epass[a]),
          'objective_correct_fraction':float(endpoint[a]['objective_correct_fraction']),
          'physical_correct_fraction':float(endpoint[a]['physical_correct_fraction']),
        }
    return out

def label_for_path(path):
    p=str(path)
    for tag in ('u10','u25','u50'):
        if tag in p:return tag
    if 'lambda095_semantic' in p or 'lambda100_semantic' in p:return 'u75'
    if 'control_' in p or 'floor_' in p:
        parent=Path(path).parent.name
        if '_' in parent:return parent.split('_')[-1]
    if 'final_floor' in p:return '25'
    return Path(path).parent.name

def summarize(samples):
    dj=np.array([s['dJ'] for s in samples]);do=np.array([s['dM_obj'] for s in samples]);dp=np.array([s['dM_phys'] for s in samples])
    def sign_agree(a,b):
        mask=(np.abs(a)>TOL)&(np.abs(b)>TOL)
        return float(np.mean(np.sign(a[mask])==np.sign(b[mask]))) if mask.any() else float('nan'),int(mask.sum())
    soa,n1=sign_agree(dj,do);spa,n2=sign_agree(dj,dp)
    pos=dj>TOL
    pf=[s for s in samples if s['pass_from'] and not s['pass_to']]
    fp=[s for s in samples if (not s['pass_from']) and s['pass_to']]
    return {
      'n':len(samples),
      'pearson_dJ_dM_obj':corr(dj,do),'spearman_dJ_dM_obj':spearman(dj,do),
      'pearson_dJ_dM_phys':corr(dj,dp),'spearman_dJ_dM_phys':spearman(dj,dp),
      'objective_sign_agreement':soa,'objective_sign_agreement_n':n1,
      'physical_sign_agreement':spa,'physical_sign_agreement_n':n2,
      'positive_dJ_n':int(pos.sum()),
      'positive_dJ_obj_deterioration_fraction':float(np.mean(do[pos]<-TOL)) if pos.any() else float('nan'),
      'positive_dJ_phys_deterioration_fraction':float(np.mean(dp[pos]<-TOL)) if pos.any() else float('nan'),
      'pass_to_fail_n':len(pf),
      'pass_to_fail_with_nonnegative_dJ_fraction':float(np.mean([x['dJ']>=-TOL for x in pf])) if pf else float('nan'),
      'fail_to_pass_n':len(fp),
      'fail_to_pass_with_nonpositive_dJ_fraction':float(np.mean([x['dJ']<=TOL for x in fp])) if fp else float('nan'),
    }

def main():
    samples=[];states={}
    for pname,files in PATHS.items():
        seq=[]
        for f in files:
            if not f.exists(): raise FileNotFoundError(f)
            seq.append((label_for_path(f),parse_report(f),str(f.relative_to(ROOT))))
        states[pname]=[]
        for lab,state,src in seq:
            states[pname].append({'checkpoint_label':lab,'source':src,'axes':state})
        for k in range(len(seq)-1):
            la,a,_=seq[k];lb,b,_=seq[k+1]
            for axis in AXES:
                samples.append({
                  'path':pname,'from':la,'to':lb,'axis':axis,
                  'dJ':b[axis]['J']-a[axis]['J'],
                  'dM_obj':b[axis]['M_obj']-a[axis]['M_obj'],
                  'dM_phys':b[axis]['M_phys']-a[axis]['M_phys'],
                  'pass_from':a[axis]['PASS'],'pass_to':b[axis]['PASS'],
                  'J_from':a[axis]['J'],'J_to':b[axis]['J'],
                  'M_obj_from':a[axis]['M_obj'],'M_obj_to':b[axis]['M_obj'],
                  'M_phys_from':a[axis]['M_phys'],'M_phys_to':b[axis]['M_phys'],
                })
    overall=summarize(samples)
    per_axis={a:summarize([s for s in samples if s['axis']==a]) for a in AXES}
    per_path={p:summarize([s for s in samples if s['path']==p]) for p in PATHS}
    c={
      'spearman_obj_ge_0p50':overall['spearman_dJ_dM_obj']>=.50,
      'spearman_phys_ge_0p50':overall['spearman_dJ_dM_phys']>=.50,
      'objective_sign_agreement_ge_0p65':overall['objective_sign_agreement']>=.65,
      'physical_sign_agreement_ge_0p65':overall['physical_sign_agreement']>=.65,
      'positive_dJ_obj_deterioration_le_0p20':overall['positive_dJ_obj_deterioration_fraction']<=.20,
      'positive_dJ_phys_deterioration_le_0p20':overall['positive_dJ_phys_deterioration_fraction']<=.20,
      'pass_loss_with_nonnegative_dJ_le_0p20':overall['pass_to_fail_with_nonnegative_dJ_fraction']<=.20,
    }
    passed=all(c.values())
    critical=[s for s in samples if s['pass_from'] and not s['pass_to']]
    mismatch=[s for s in samples if s['dJ']>TOL and (s['dM_obj']<-TOL or s['dM_phys']<-TOL)]
    rep={
      'schema':'trajectory_objective_sufficiency_audit_v1',
      'status':'TRAJECTORY SUFFICIENCY PASS' if passed else 'TRAJECTORY SUFFICIENCY FAIL',
      'read_only':True,'optimizer_steps':0,'new_rollouts':0,
      'paths':states,'sample_count':len(samples),'transition_count':len(samples)//4,
      'overall':overall,'per_axis':per_axis,'per_path':per_path,'criteria':c,
      'pass_to_fail_events':critical,'positive_dJ_semantic_mismatch_events':mismatch,
      'samples':samples,
      'decision':{'trajectory_level_semantic_sufficiency_validated':bool(passed),
                  'local_objective_semantics_remain_closed':True,
                  'new_method_authorized':False}
    }
    OUT.mkdir(parents=True,exist_ok=True)
    out=OUT/'trajectory_objective_sufficiency_report.json';out.write_text(json.dumps(rep,indent=2)+'\n')
    prov={'status':'FROZEN_BY_HASH','decision':rep['status'],'report_sha256':sha(out),
          'script_sha256':sha(Path(__file__).resolve()),'contract_sha256':sha(ROOT/'docs/contracts/diagnostics/trajectory-objective-sufficiency-contract.md'),
          'source_sha256':{str(f.relative_to(ROOT)):sha(f) for fs in PATHS.values() for f in fs}}
    (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(json.dumps({'status':rep['status'],'sample_count':len(samples),'transition_count':len(samples)//4,
                      'overall':overall,'per_axis':per_axis,'criteria':c,
                      'pass_to_fail_events':critical},indent=2))
if __name__=='__main__':main()
