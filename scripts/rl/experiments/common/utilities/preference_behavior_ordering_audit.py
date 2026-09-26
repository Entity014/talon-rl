#!/usr/bin/env python3
from __future__ import annotations

# scripts/ on sys.path so the absolute rl.experiments.* imports below
# resolve when this file is run directly, as these scripts always are.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pathlib import Path
import hashlib,json,sys,os
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
from rl.experiments.common.utilities.trajectory_information_attribution_audit import obs_tensor,trace,checkpoint_model_hash,ORDER,IDX,PHYS,NENV,STEPS,SUITES,TOL,rankdata,corr,spearman
OUT=ROOT/'runs/preference_behavior_ordering_audit-2026-09-24'
CACHE=OUT/'raw_cache'
BASE=ROOT/'runs/v2b_adam_continuous-2026-09-24/model_75.pt'
HOLD=ROOT/'runs/relational_persistence_heldout-2026-09-24/heldout_report.json'
SEEDS=(980001,981001,982001)
QS=(0.10,0.25,0.40,0.55,0.70)
PAIRS=[(hi,lo) for hi in range(len(QS)) for lo in range(hi) if not (QS[hi]==0.70 and QS[lo]==0.25)]
ALLPAIRS=[(hi,lo) for hi in range(len(QS)) for lo in range(hi)]
ADJ=[(i+1,i) for i in range(len(QS)-1)]
SHARD_COUNT=int(os.environ.get('ORDERING_SHARD_COUNT','1'))
SHARD_ID=int(os.environ.get('ORDERING_SHARD_ID','0'))
MEASURE_ONLY=os.environ.get('ORDERING_MEASURE_ONLY','0')=='1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seq(seed):return [('base',BASE)]+[(f'u{u}',ROOT/f'runs/v2b_bounded_deltaa_multiseed-2026-09-24/seed_{seed}/control_u{u}.pt') for u in range(1,9)]
def pref(axis,q):
    w=np.full(4,(1-q)/3,dtype=np.float32);w[IDX[axis]]=q;return w

def suite_profile(traces,axis):
    j=IDX[axis];pk=PHYS[axis]
    obj=[float(np.mean(t['obj'][:,j])) for t in traces]
    phys=[float(-np.mean(t['phys'][pk])) for t in traces]
    return obj,phys

def pairacc(vals,pairs):return float(np.mean([vals[i]>vals[j]+TOL for i,j in pairs]))
def between(vals):
    lo=min(vals[0],vals[-1])-TOL;hi=max(vals[0],vals[-1])+TOL
    return float(np.mean([lo<=vals[i]<=hi for i in (1,2,3)]))
def ckpt_axis_metrics(suite_profiles):
    obj_acc=[];phys_acc=[];joint_acc=[];obj_adj=[];phys_adj=[];obj_rho=[];phys_rho=[];obj_between=[];phys_between=[]
    pair_suite_obj={p:[] for p in PAIRS};pair_suite_phys={p:[] for p in PAIRS};pair_suite_joint={p:[] for p in PAIRS}
    all_obj=[];all_phys=[]
    for obj,phys in suite_profiles:
        obj=np.asarray(obj);phys=np.asarray(phys)
        obj_acc.append(pairacc(obj,PAIRS));phys_acc.append(pairacc(phys,PAIRS));joint_acc.append(float(np.mean([(obj[i]>obj[j]+TOL) and (phys[i]>phys[j]+TOL) for i,j in PAIRS])))
        obj_adj.append(pairacc(obj,ADJ));phys_adj.append(pairacc(phys,ADJ));obj_rho.append(spearman(QS,obj));phys_rho.append(spearman(QS,phys));obj_between.append(between(obj));phys_between.append(between(phys))
        for p in PAIRS:
            i,j=p;oo=obj[i]>obj[j]+TOL;pp=phys[i]>phys[j]+TOL;pair_suite_obj[p].append(oo);pair_suite_phys[p].append(pp);pair_suite_joint[p].append(oo and pp)
        all_obj.append(obj);all_phys.append(phys)
    def reset_cons(d):return float(np.mean([np.mean(v)>=.75 for v in d.values()]))
    # heavy-center direct correctness retained only as control
    hc=(QS.index(.70),QS.index(.25))
    hc_obj=float(np.mean([o[hc[0]]>o[hc[1]]+TOL for o in all_obj]));hc_phys=float(np.mean([p[hc[0]]>p[hc[1]]+TOL for p in all_phys]))
    return {
      'obj_primary_order_acc':float(np.mean(obj_acc)),'phys_primary_order_acc':float(np.mean(phys_acc)),'joint_primary_order_acc':float(np.mean(joint_acc)),
      'obj_adjacent_acc':float(np.mean(obj_adj)),'phys_adjacent_acc':float(np.mean(phys_adj)),
      'obj_rank_rho':float(np.mean(obj_rho)),'phys_rank_rho':float(np.mean(phys_rho)),
      'obj_between_fraction':float(np.mean(obj_between)),'phys_between_fraction':float(np.mean(phys_between)),
      'obj_reset_consistent_pair_fraction':reset_cons(pair_suite_obj),'phys_reset_consistent_pair_fraction':reset_cons(pair_suite_phys),'joint_reset_consistent_pair_fraction':reset_cons(pair_suite_joint),
      'heavy_center_obj_correct_fraction':hc_obj,'heavy_center_phys_correct_fraction':hc_phys,
      'suite_profiles':[{'obj':o.tolist(),'phys':p.tolist()} for o,p in zip(all_obj,all_phys)]}

def main():
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
    env=None
    try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
      env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
      mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene['robot']
      refs=[(sd,lab,p) for sd in SEEDS for lab,p in seq(sd)];hash_to_path={};refhash={}
      for sd,lab,p in refs:
        hh=checkpoint_model_hash(p);refhash[(sd,lab)]=hh;hash_to_path.setdefault(hh,p)
      print(f'UNIQUE_CHECKPOINTS {len(hash_to_path)} / REFERENCES {len(refs)}',flush=True)
      traces={}
      prefmap={(a,q):pref(a,q) for a in ORDER for q in QS}
      CACHE.mkdir(parents=True,exist_ok=True)
      for ci,(hh,p) in enumerate(hash_to_path.items(),1):
        if SHARD_COUNT>1 and ((ci-1)%SHARD_COUNT)!=SHARD_ID:
          continue
        cp=CACHE/f'{hh}.npz'
        print(f'CHECKPOINT {ci}/{len(hash_to_path)} {p.relative_to(ROOT)} cache={cp.exists()}',flush=True)
        ck={}
        if cp.exists():
          z=np.load(cp,allow_pickle=False)
          obj=z['obj'];phys=z['phys']
          for ai,a in enumerate(ORDER):
            suites=[]
            for si in range(SUITES):
              pts=[]
              for qi,q in enumerate(QS):
                pts.append({'obj':obj[ai,si,qi], 'phys':{PHYS[a2]:phys[ai,si,qi,:,k] for k,a2 in enumerate(ORDER)}})
              suites.append(pts)
            ck[a]=ckpt_axis_metrics([suite_profile(suite,a) for suite in suites])
        else:
          m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);m.eval()
          obj=np.zeros((4,SUITES,len(QS),STEPS,4),dtype=np.float64);phys=np.zeros((4,SUITES,len(QS),STEPS,4),dtype=np.float64)
          for ai,a in enumerate(ORDER):
            suites=[]
            for suite in range(SUITES):
              sd=840001+suite;pts=[]
              for qi,q in enumerate(QS):
                t=trace(env,m,mgr,robot,prefmap[(a,q)],sd);pts.append(t);obj[ai,suite,qi]=t['obj']
                for pk,a2 in enumerate(ORDER): phys[ai,suite,qi,:,pk]=t['phys'][PHYS[a2]]
              suites.append(pts)
            ck[a]=ckpt_axis_metrics([suite_profile(suite,a) for suite in suites])
          tmp=cp.with_suffix('.tmp.npz');np.savez_compressed(tmp,obj=obj,phys=phys,checkpoint_hash=np.array(hh),checkpoint_path=np.array(str(p.relative_to(ROOT))),q=np.asarray(QS),reset_seeds=np.arange(840001,840001+SUITES))
          tmp.replace(cp)
          print(f'RAW_CACHE_WRITTEN {cp.name}',flush=True)
        traces[hh]=ck
      if MEASURE_ONLY:
        print(f'MEASURE_ONLY_DONE shard={SHARD_ID}/{SHARD_COUNT} cache_count={len(list(CACHE.glob("*.npz")))}',flush=True)
        return
      # attach frozen semantic targets/scalar controls
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
      # summary
      pf=[s for s in samples if s['pass_from'] and not s['pass_to']];fp=[s for s in samples if (not s['pass_from']) and s['pass_to']]
      def detfrac(fn):return float(np.mean([fn(s) for s in pf])) if pf else float('nan')
      jointdet=detfrac(lambda s:s['delta']['joint_primary_order_acc']<-TOL or s['delta']['joint_reset_consistent_pair_fraction']<-TOL)
      strongest_scalar=max(detfrac(lambda s:s['d_obj_rel']<-TOL),detfrac(lambda s:s['d_phys_rel']<-TOL))
      both_nondec_order=float(np.mean([(s['d_obj_rel']>=-TOL and s['d_phys_rel']>=-TOL and (s['delta']['joint_primary_order_acc']<-TOL or s['delta']['joint_reset_consistent_pair_fraction']<-TOL)) for s in pf])) if pf else float('nan')
      rho=spearman([s['delta']['joint_primary_order_acc'] for s in samples],[s['d_sem'] for s in samples])
      seed_cons=True;perseed={}
      for sd in SEEDS:
        q=[s for s in samples if s['seed']==sd];r=spearman([s['delta']['joint_primary_order_acc'] for s in q],[s['d_sem'] for s in q]);perseed[str(sd)]=r;seed_cons &= np.isfinite(r) and r>0
      fp_improve=float(np.mean([s['delta']['joint_primary_order_acc']>TOL or s['delta']['joint_reset_consistent_pair_fraction']>TOL for s in fp])) if fp else float('nan')
      criteria={'pf_order_deterioration_ge0p75':jointdet>=.75,'beats_strongest_scalar_by0p10':jointdet>=strongest_scalar+.10,'pf_both_mean_nondec_order_deterioration_ge0p25':both_nondec_order>=.25,'joint_order_delta_spearman_ge0p50':abs(rho)>=.50,'positive_direction_all3_seeds':bool(seed_cons),'fp_order_improve_ge0p60_if_powered':(fp_improve>=.60 if len(fp)>=4 else True)}
      supported=all(criteria.values())
      # prospective secondary
      passorig=[]
      for sd,arr in states.items():
        for i in range(len(arr)-1):
          for a in ORDER:
            x=arr[i]['axes'][a];y=arr[i+1]['axes'][a]
            if x['PASS']:passorig.append({'seed':int(sd),'axis':a,'retained':bool(y['PASS']),'joint_order':x['joint_primary_order_acc'],'joint_reset':x['joint_reset_consistent_pair_fraction'],'obj_rel':x['obj_adv_mean'],'phys_rel':x['phys_adv_mean']})
      hi=[x for x in passorig if x['joint_primary_order_acc']>=.75];lo=[x for x in passorig if x['joint_primary_order_acc']<.75]
      prospective={'n':len(passorig),'high_order_n':len(hi),'low_order_n':len(lo),'high_order_retention':float(np.mean([x['retained'] for x in hi])) if hi else float('nan'),'low_order_retention':float(np.mean([x['retained'] for x in lo])) if lo else float('nan')}
      status='ORDERING HYPOTHESIS SUPPORTED' if supported else ('ORDERING INFORMATIVE BUT NOT INCREMENTAL' if abs(rho)>=.5 or jointdet>=.75 else 'ORDERING NOT SUPPORTED')
      rep={'schema':'preference_behavior_ordering_audit_v1','status':status,'measurement_only':True,'optimizer_steps':0,'preference_q':list(QS),'primary_pairs':[[QS[i],QS[j]] for i,j in PAIRS],'unique_checkpoints':len(hash_to_path),'states':states,'samples':samples,
        'summary':{'pass_to_fail_n':len(pf),'fail_to_pass_n':len(fp),'pf_order_deterioration_fraction':jointdet,'pf_strongest_scalar_deterioration_fraction':strongest_scalar,'pf_both_mean_nondec_with_order_deterioration_fraction':both_nondec_order,'joint_order_delta_semantic_spearman':rho,'per_seed_spearman':perseed,'fail_to_pass_order_improvement_fraction':fp_improve},'criteria':criteria,'prospective_secondary':prospective,'decision':{'ranking_target_design_authorized':supported}}
      OUT.mkdir(parents=True,exist_ok=True);out=OUT/'preference_behavior_ordering_report.json';out.write_text(json.dumps(rep,indent=2)+'\n')
      (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':status,'report_sha256':sha(out),'script_sha256':sha(Path(__file__).resolve()),'contract_sha256':sha(ROOT/'docs/contracts/preference_control/preference-behavior-ordering-audit-contract.md'),'checkpoint_sha256':{str(p.relative_to(ROOT)):sha(p) for _,_,p in refs}},indent=2)+'\n')
      print(json.dumps({'status':status,'summary':rep['summary'],'criteria':criteria,'prospective_secondary':prospective},indent=2),flush=True)
    finally:
      if env is not None:env.close()
      app.close()
if __name__=='__main__':main()
