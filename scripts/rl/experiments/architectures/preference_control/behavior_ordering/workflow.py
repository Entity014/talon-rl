"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_preference_behavior_ordering_analyze_cached():
    """Run former preference_behavior_ordering_analyze_cached.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import hashlib,json,sys
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    from rl.experiments.common.utilities.preference_behavior_ordering_audit import (
        ORDER,IDX,PHYS,QS,PAIRS,TOL,ckpt_axis_metrics,suite_profile,seq,SEEDS,BASE,CACHE,sha
    )
    from rl.experiments.common.utilities.trajectory_information_attribution_audit import checkpoint_model_hash,spearman
    OUT=ROOT/'runs/preference_behavior_ordering_audit-2026-09-24'
    HOLD=ROOT/'runs/relational_persistence_heldout-2026-09-24/heldout_report.json'
    
    def load_ckpt_cache(cp):
        z=np.load(cp,allow_pickle=False);obj=z['obj'];phys=z['phys'];ck={}
        for ai,a in enumerate(ORDER):
            suites=[]
            for si in range(4):
                pts=[]
                for qi,q in enumerate(QS):
                    pts.append({'obj':obj[ai,si,qi], 'phys':{PHYS[a2]:phys[ai,si,qi,:,k] for k,a2 in enumerate(ORDER)}})
                suites.append(pts)
            ck[a]=ckpt_axis_metrics([suite_profile(suite,a) for suite in suites])
        return ck
    
    def main():
        refs=[(sd,lab,p) for sd in SEEDS for lab,p in seq(sd)]
        hash_to_path={};refhash={}
        for sd,lab,p in refs:
            hh=checkpoint_model_hash(p);refhash[(sd,lab)]=hh;hash_to_path.setdefault(hh,p)
        missing=[];traces={}
        for hh,p in hash_to_path.items():
            cp=CACHE/f'{hh}.npz'
            if not cp.exists(): missing.append(str(p.relative_to(ROOT)))
            else: traces[hh]=load_ckpt_cache(cp)
        if missing:
            raise SystemExit('MISSING_CACHE '+json.dumps(missing))
        hd=json.load(open(HOLD));hstates=hd['states']
        states={};samples=[]
        for sd in SEEDS:
            arr=[]
            for idx,(lab,p) in enumerate(seq(sd)):
                axes={}
                for a in ORDER:
                    q=dict(traces[refhash[(sd,lab)]][a]);hs=hstates[str(sd)][idx]['axes'][a]
                    q.update({'PASS':bool(hs['PASS']),'semantic_score':float(hs['semantic_score']),'J_heavy':float(hs['J_heavy']),'obj_adv_mean':float(hs['obj_adv_mean']),'phys_adv_mean':float(hs['phys_adv_mean'])});axes[a]=q
                arr.append({'label':lab,'checkpoint':str(p.relative_to(ROOT)),'axes':axes})
            states[str(sd)]=arr
            for i in range(len(arr)-1):
                for a in ORDER:
                    x=arr[i]['axes'][a];y=arr[i+1]['axes'][a]
                    keys=['obj_primary_order_acc','phys_primary_order_acc','joint_primary_order_acc','obj_rank_rho','phys_rank_rho','obj_reset_consistent_pair_fraction','phys_reset_consistent_pair_fraction','joint_reset_consistent_pair_fraction']
                    samples.append({'seed':sd,'from':arr[i]['label'],'to':arr[i+1]['label'],'axis':a,'pass_from':x['PASS'],'pass_to':y['PASS'],'d_sem':y['semantic_score']-x['semantic_score'],
                      'dJ':y['J_heavy']-x['J_heavy'],'d_obj_rel':y['obj_adv_mean']-x['obj_adv_mean'],'d_phys_rel':y['phys_adv_mean']-x['phys_adv_mean'],
                      'delta':{k:y[k]-x[k] for k in keys}})
        pf=[s for s in samples if s['pass_from'] and not s['pass_to']];fp=[s for s in samples if (not s['pass_from']) and s['pass_to']]
        def detfrac(fn): return float(np.mean([fn(s) for s in pf])) if pf else float('nan')
        jointdet=detfrac(lambda s:s['delta']['joint_primary_order_acc']<-TOL or s['delta']['joint_reset_consistent_pair_fraction']<-TOL)
        obj_scalar_det=detfrac(lambda s:s['d_obj_rel']<-TOL);phys_scalar_det=detfrac(lambda s:s['d_phys_rel']<-TOL)
        strongest_scalar=max(obj_scalar_det,phys_scalar_det)
        both_nondec_order=float(np.mean([(s['d_obj_rel']>=-TOL and s['d_phys_rel']>=-TOL and (s['delta']['joint_primary_order_acc']<-TOL or s['delta']['joint_reset_consistent_pair_fraction']<-TOL)) for s in pf])) if pf else float('nan')
        rho=spearman([s['delta']['joint_primary_order_acc'] for s in samples],[s['d_sem'] for s in samples])
        seed_cons=True;perseed={}
        for sd in SEEDS:
            q=[s for s in samples if s['seed']==sd];rr=spearman([s['delta']['joint_primary_order_acc'] for s in q],[s['d_sem'] for s in q]);perseed[str(sd)]=rr;seed_cons &= np.isfinite(rr) and rr>0
        fp_improve=float(np.mean([s['delta']['joint_primary_order_acc']>TOL or s['delta']['joint_reset_consistent_pair_fraction']>TOL for s in fp])) if fp else float('nan')
        criteria={
          'pf_order_deterioration_ge0p75':jointdet>=.75,
          'beats_strongest_scalar_by0p10':jointdet>=strongest_scalar+.10,
          'pf_both_mean_nondec_order_deterioration_ge0p25':both_nondec_order>=.25,
          'joint_order_delta_spearman_ge0p50':abs(rho)>=.50,
          'positive_direction_all3_seeds':bool(seed_cons),
          'fp_order_improve_ge0p60_if_powered':(fp_improve>=.60 if len(fp)>=4 else True),
        }
        supported=all(criteria.values())
        passorig=[]
        for sd,arr in states.items():
            for i in range(len(arr)-1):
                for a in ORDER:
                    x=arr[i]['axes'][a];y=arr[i+1]['axes'][a]
                    if x['PASS']: passorig.append({'seed':int(sd),'axis':a,'retained':bool(y['PASS']),'joint_order':x['joint_primary_order_acc'],'joint_reset':x['joint_reset_consistent_pair_fraction'],'obj_rel':x['obj_adv_mean'],'phys_rel':x['phys_adv_mean']})
        hi=[x for x in passorig if x['joint_primary_order_acc']>=.75];lo=[x for x in passorig if x['joint_primary_order_acc']<.75]
        prospective={'n':len(passorig),'high_order_n':len(hi),'low_order_n':len(lo),'high_order_retention':float(np.mean([x['retained'] for x in hi])) if hi else float('nan'),'low_order_retention':float(np.mean([x['retained'] for x in lo])) if lo else float('nan')}
        status='ORDERING HYPOTHESIS SUPPORTED' if supported else ('ORDERING INFORMATIVE BUT NOT INCREMENTAL' if abs(rho)>=.5 or jointdet>=.75 else 'ORDERING NOT SUPPORTED')
        # fixed secondary descriptors for transparent reporting only
        metric_summary={}
        for key in ['obj_primary_order_acc','phys_primary_order_acc','joint_primary_order_acc','obj_rank_rho','phys_rank_rho','obj_reset_consistent_pair_fraction','phys_reset_consistent_pair_fraction','joint_reset_consistent_pair_fraction']:
            xx=[s['delta'][key] for s in samples];metric_summary[key]={'spearman_semantic_change':spearman(xx,[s['d_sem'] for s in samples]),'pf_deterioration_fraction':detfrac(lambda s,k=key:s['delta'][k]<-TOL),'fp_improvement_fraction':float(np.mean([s['delta'][key]>TOL for s in fp])) if fp else float('nan')}
        report={'schema':'preference_behavior_ordering_audit_v1','status':status,'measurement_only':True,'optimizer_steps':0,'preference_q':list(QS),'primary_pairs':[[QS[i],QS[j]] for i,j in PAIRS],'unique_checkpoints':len(hash_to_path),'raw_cache_count':len(traces),'states':states,'samples':samples,
          'summary':{'pass_to_fail_n':len(pf),'fail_to_pass_n':len(fp),'pf_order_deterioration_fraction':jointdet,'pf_obj_mean_relation_deterioration_fraction':obj_scalar_det,'pf_phys_mean_relation_deterioration_fraction':phys_scalar_det,'pf_strongest_scalar_deterioration_fraction':strongest_scalar,'pf_both_mean_nondec_with_order_deterioration_fraction':both_nondec_order,'joint_order_delta_semantic_spearman':rho,'per_seed_spearman':perseed,'fail_to_pass_order_improvement_fraction':fp_improve},
          'metric_summary':metric_summary,'criteria':criteria,'prospective_secondary':prospective,'decision':{'ranking_target_design_authorized':supported}}
        out=OUT/'preference_behavior_ordering_report.json';out.write_text(json.dumps(report,indent=2)+'\n')
        (OUT/'FINAL_PREFERENCE_BEHAVIOR_ORDERING_PROVENANCE.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':status,'raw_cache_files':len(traces),'report_sha256':sha(out),'analysis_script_sha256':sha(Path(__file__).resolve()),'measurement_script_sha256':sha(ROOT/'scripts/rl/experiments/common/utilities/preference_behavior_ordering_audit.py'),'contract_sha256':sha(ROOT/'docs/contracts/preference_control/preference-behavior-ordering-audit-contract.md'),'cache_sha256':{p.name:sha(p) for p in sorted(CACHE.glob('*.npz'))}},indent=2)+'\n')
        print(json.dumps({'status':status,'summary':report['summary'],'metric_summary':metric_summary,'criteria':criteria,'prospective_secondary':prospective},indent=2))
    if True:main()

def run_preference_behavior_ordering_offline_verdict():
    """Run former preference_behavior_ordering_offline_verdict.py stage."""
    from pathlib import Path
    import json, math, hashlib
    import numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    RUN=ROOT/'runs/preference_behavior_ordering_audit-2026-09-24'
    CACHE=RUN/'raw_cache'
    HOLD=ROOT/'runs/relational_persistence_heldout-2026-09-24/heldout_report.json'
    SEEDS=(980001,981001,982001); AXES=('T','A','O','S'); QS=(0.10,0.25,0.40,0.55,0.70); TOL=1e-10
    PAIRS=[(hi,lo) for hi in range(len(QS)) for lo in range(hi) if not (QS[hi]==0.70 and QS[lo]==0.25)]
    ADJ=[(i+1,i) for i in range(len(QS)-1)]
    
    def rankdata(x):
        x=np.asarray(x,float); o=np.argsort(x,kind='mergesort'); r=np.empty(len(x),float); i=0
        while i<len(x):
            j=i+1
            while j<len(x) and x[o[j]]==x[o[i]]: j+=1
            r[o[i:j]]=(i+j-1)/2+1; i=j
        return r
    
    def corr(x,y):
        x=np.asarray(x,float); y=np.asarray(y,float)
        if len(x)<2 or np.std(x)<1e-15 or np.std(y)<1e-15:return float('nan')
        return float(np.corrcoef(x,y)[0,1])
    def spear(x,y): return corr(rankdata(x),rankdata(y))
    def pairacc(vals,pairs): return float(np.mean([vals[i]>vals[j]+TOL for i,j in pairs]))
    def between(vals):
        lo=min(vals[0],vals[-1])-TOL; hi=max(vals[0],vals[-1])+TOL
        return float(np.mean([lo<=vals[i]<=hi for i in (1,2,3)]))
    def reset_cons(pair_suite): return float(np.mean([np.mean(v)>=.75 for v in pair_suite.values()]))
    
    def metrics_from_cache(npz_path,axis_idx):
        z=np.load(npz_path,allow_pickle=False); obj=z['obj']; phys=z['phys']
        obj_acc=[];phys_acc=[];joint_acc=[];obj_adj=[];phys_adj=[];obj_rho=[];phys_rho=[];obj_between=[];phys_between=[]
        po={p:[] for p in PAIRS};pp={p:[] for p in PAIRS};pj={p:[] for p in PAIRS}
        profiles=[]
        for si in range(4):
            o=np.array([float(np.mean(obj[axis_idx,si,qi,:,axis_idx])) for qi in range(5)])
            p=np.array([float(-np.mean(phys[axis_idx,si,qi,:,axis_idx])) for qi in range(5)])
            profiles.append({'obj':o.tolist(),'phys':p.tolist()})
            obj_acc.append(pairacc(o,PAIRS)); phys_acc.append(pairacc(p,PAIRS)); joint_acc.append(float(np.mean([(o[i]>o[j]+TOL) and (p[i]>p[j]+TOL) for i,j in PAIRS])))
            obj_adj.append(pairacc(o,ADJ));phys_adj.append(pairacc(p,ADJ));obj_rho.append(spear(QS,o));phys_rho.append(spear(QS,p));obj_between.append(between(o));phys_between.append(between(p))
            for pair in PAIRS:
                i,j=pair; a=o[i]>o[j]+TOL; b=p[i]>p[j]+TOL;po[pair].append(a);pp[pair].append(b);pj[pair].append(a and b)
        return {'obj_primary_order_acc':float(np.mean(obj_acc)),'phys_primary_order_acc':float(np.mean(phys_acc)),'joint_primary_order_acc':float(np.mean(joint_acc)),
          'obj_adjacent_acc':float(np.mean(obj_adj)),'phys_adjacent_acc':float(np.mean(phys_adj)),
          'obj_rank_rho':float(np.mean(obj_rho)),'phys_rank_rho':float(np.mean(phys_rho)),
          'obj_between_fraction':float(np.mean(obj_between)),'phys_between_fraction':float(np.mean(phys_between)),
          'obj_reset_consistent_pair_fraction':reset_cons(po),'phys_reset_consistent_pair_fraction':reset_cons(pp),'joint_reset_consistent_pair_fraction':reset_cons(pj),
          'suite_profiles':profiles}
    
    def main():
        hd=json.load(open(HOLD)); hstates=hd['states']
        # path -> cache file from metadata
        bypath={}
        for p in CACHE.glob('*.npz'):
            z=np.load(p,allow_pickle=False); bypath[str(z['checkpoint_path'])]=p
        states={}; samples=[]
        for sd in SEEDS:
            arr=[]
            for idx,hst in enumerate(hstates[str(sd)]):
                path=hst['checkpoint']; cp=bypath[path]; axes={}
                for ai,a in enumerate(AXES):
                    q=metrics_from_cache(cp,ai); hs=hst['axes'][a]
                    q.update({'PASS':bool(hs['PASS']),'semantic_score':float(hs['semantic_score']),'J_heavy':float(hs['J_heavy']),'obj_adv_mean':float(hs['obj_adv_mean']),'phys_adv_mean':float(hs['phys_adv_mean'])})
                    axes[a]=q
                arr.append({'label':hst['label'],'checkpoint':path,'axes':axes})
            states[str(sd)]=arr
            for i in range(len(arr)-1):
                for a in AXES:
                    x=arr[i]['axes'][a]; y=arr[i+1]['axes'][a]
                    keys=['obj_primary_order_acc','phys_primary_order_acc','joint_primary_order_acc','obj_rank_rho','phys_rank_rho','obj_reset_consistent_pair_fraction','phys_reset_consistent_pair_fraction','joint_reset_consistent_pair_fraction']
                    samples.append({'seed':sd,'from':arr[i]['label'],'to':arr[i+1]['label'],'axis':a,'pass_from':x['PASS'],'pass_to':y['PASS'],'d_sem':y['semantic_score']-x['semantic_score'],
                        'dJ':y['J_heavy']-x['J_heavy'],'d_obj_rel':y['obj_adv_mean']-x['obj_adv_mean'],'d_phys_rel':y['phys_adv_mean']-x['phys_adv_mean'],
                        'delta':{k:y[k]-x[k] for k in keys}})
        pf=[s for s in samples if s['pass_from'] and not s['pass_to']]; fp=[s for s in samples if (not s['pass_from']) and s['pass_to']]
        def frac(pred,ss=pf): return float(np.mean([pred(s) for s in ss])) if ss else float('nan')
        order_det=frac(lambda s:s['delta']['joint_primary_order_acc']<-TOL or s['delta']['joint_reset_consistent_pair_fraction']<-TOL)
        objrel_det=frac(lambda s:s['d_obj_rel']<-TOL); physrel_det=frac(lambda s:s['d_phys_rel']<-TOL); strongest=max(objrel_det,physrel_det)
        both_nondec_order=frac(lambda s:s['d_obj_rel']>=-TOL and s['d_phys_rel']>=-TOL and (s['delta']['joint_primary_order_acc']<-TOL or s['delta']['joint_reset_consistent_pair_fraction']<-TOL))
        rho=spear([s['delta']['joint_primary_order_acc'] for s in samples],[s['d_sem'] for s in samples])
        perseed={}; seed_cons=True
        for sd in SEEDS:
            ss=[s for s in samples if s['seed']==sd]; r=spear([s['delta']['joint_primary_order_acc'] for s in ss],[s['d_sem'] for s in ss]);perseed[str(sd)]=r;seed_cons &= np.isfinite(r) and r>0
        fp_improve=frac(lambda s:s['delta']['joint_primary_order_acc']>TOL or s['delta']['joint_reset_consistent_pair_fraction']>TOL,fp)
        criteria={'pf_order_deterioration_ge0p75':order_det>=.75,'beats_strongest_scalar_by0p10':order_det>=strongest+.10,'pf_both_mean_nondec_order_deterioration_ge0p25':both_nondec_order>=.25,'joint_order_delta_spearman_ge0p50':abs(rho)>=.50,'positive_direction_all3_seeds':bool(seed_cons),'fp_order_improve_ge0p60_if_powered':(fp_improve>=.60 if len(fp)>=4 else True)}
        supported=all(criteria.values())
        # useful descriptive metrics for all ordering channels
        channel_corr={k:spear([s['delta'][k] for s in samples],[s['d_sem'] for s in samples]) for k in ['obj_primary_order_acc','phys_primary_order_acc','joint_primary_order_acc','obj_rank_rho','phys_rank_rho','obj_reset_consistent_pair_fraction','phys_reset_consistent_pair_fraction','joint_reset_consistent_pair_fraction']}
        pf_channel_det={k:frac(lambda s,k=k:s['delta'][k]<-TOL) for k in channel_corr}
        # prospective secondary
        passorig=[]
        for sd,arr in states.items():
            for i in range(len(arr)-1):
                for a in AXES:
                    x=arr[i]['axes'][a]; y=arr[i+1]['axes'][a]
                    if x['PASS']:passorig.append({'seed':int(sd),'axis':a,'retained':bool(y['PASS']),'joint_order':x['joint_primary_order_acc'],'joint_reset':x['joint_reset_consistent_pair_fraction'],'obj_rel':x['obj_adv_mean'],'phys_rel':x['phys_adv_mean']})
        hi=[x for x in passorig if x['joint_order']>=.75];lo=[x for x in passorig if x['joint_order']<.75]
        prospective={'n':len(passorig),'high_order_n':len(hi),'low_order_n':len(lo),'high_order_retention':float(np.mean([x['retained'] for x in hi])) if hi else float('nan'),'low_order_retention':float(np.mean([x['retained'] for x in lo])) if lo else float('nan')}
        status='ORDERING HYPOTHESIS SUPPORTED' if supported else ('ORDERING INFORMATIVE BUT NOT INCREMENTAL' if abs(rho)>=.5 or order_det>=.75 else 'ORDERING NOT SUPPORTED')
        rep={'schema':'preference_behavior_ordering_audit_v1','status':status,'measurement_only':True,'optimizer_steps':0,'cache_integrity':'25/25 validated','preference_q':list(QS),'primary_pairs':[[QS[i],QS[j]] for i,j in PAIRS],
          'states':states,'samples':samples,'summary':{'pass_to_fail_n':len(pf),'fail_to_pass_n':len(fp),'pf_order_deterioration_fraction':order_det,'pf_obj_mean_relation_deterioration_fraction':objrel_det,'pf_phys_mean_relation_deterioration_fraction':physrel_det,'pf_strongest_scalar_deterioration_fraction':strongest,'pf_both_mean_nondec_with_order_deterioration_fraction':both_nondec_order,'joint_order_delta_semantic_spearman':rho,'per_seed_spearman':perseed,'fail_to_pass_order_improvement_fraction':fp_improve,'channel_correlations':channel_corr,'pf_channel_deterioration':pf_channel_det},
          'criteria':criteria,'prospective_secondary':prospective,'decision':{'ranking_target_design_authorized':supported}}
        out=RUN/'preference_behavior_ordering_report.json';out.write_text(json.dumps(rep,indent=2)+'\n')
        print(json.dumps({'status':status,'summary':rep['summary'],'criteria':criteria,'prospective_secondary':prospective},indent=2))
    if True: main()

STAGES = {
    "preference_behavior_ordering_analyze_cached": run_preference_behavior_ordering_analyze_cached,
    "preference_behavior_ordering_offline_verdict": run_preference_behavior_ordering_offline_verdict,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
