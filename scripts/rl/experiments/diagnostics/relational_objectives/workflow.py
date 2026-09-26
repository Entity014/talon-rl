"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_relational_objective_design_audit():
    """Run former relational_objective_design_audit.py stage."""
    """Does a relation-based objective track semantic correctness better than J?
    
    Offline only, no optimizer steps. Compares the change in the heavy objective
    against the change in the pure relational term, then a family of candidates
    that weight the two, at three values of eta. A candidate is authorised only if
    the pure relation beats J outright and every eta passes every criterion —
    including that a gain is rarely driven by the objective centre degrading.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import REPO, RUNS, OfflineAudit
    from rl.experiments.common.utilities.rank_stats import corr, spearman
    
    SRC = "relational_persistence_heldout-2026-09-24/heldout_report.json"
    CONTRACT = "docs/contracts/diagnostics/relational-objective-design-audit-contract.md"
    AXES = ("T", "A", "O", "S")
    ETAS = (0.25, 0.50, 1.00)
    TOL = 1e-10
    
    
    def metric(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        mask = (np.abs(x) > TOL) & (np.abs(y) > TOL)
        return {"n": len(x), "spearman": spearman(x, y), "pearson": corr(x, y),
                "sign_agreement": (float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))
                                   if mask.any() else float("nan")),
                "sign_n": int(mask.sum())}
    
    
    def frac(part, whole, default=float("nan")):
        return float(len(part) / len(whole)) if whole else default
    
    
    class RelationalObjectiveDesignAudit(OfflineAudit):
        """Whether a relational objective earns its place over the heavy objective."""
    
        run = "relational_objective_design_audit-2026-09-24"
        report = "relational_objective_design_report.json"
        schema = "relational_objective_design_audit_v1"
    
        def samples(self):
            states = json.loads((RUNS / SRC).read_text())["states"]
            out = []
            for seed, arr in states.items():
                for i in range(len(arr) - 1):
                    a, b = arr[i], arr[i + 1]
                    for axis in AXES:
                        x, y = a["axes"][axis], b["axes"][axis]
                        jh0, jh1 = float(x["J_heavy"]), float(y["J_heavy"])
                        r0, r1 = float(x["obj_adv_mean"]), float(y["obj_adv_mean"])
                        out.append({
                            "seed": int(seed), "from": a["label"], "to": b["label"], "axis": axis,
                            "dJH": jh1 - jh0, "dJC": (jh1 - r1) - (jh0 - r0), "dR": r1 - r0,
                            "d_obj_correct": float(y["objective_correct_fraction"] - x["objective_correct_fraction"]),
                            "d_phys_correct": float(y["physical_correct_fraction"] - x["physical_correct_fraction"]),
                            "d_semantic_score": float(y["semantic_score"] - x["semantic_score"]),
                            "pass_from": bool(x["PASS"]), "pass_to": bool(y["PASS"])})
            return out
    
        def eta_block(self, samples, eta, base_obj):
            vals = []
            for s in samples:
                heavy = (1 + eta) * s["dJH"]
                center = -eta * s["dJC"]
                q = dict(s)
                q.update({"dCand": heavy + center, "heavy_contribution": heavy,
                          "center_contribution": center,
                          "guardrail_ok": bool(s["dJH"] >= -TOL),
                          "candidate_positive": bool(heavy + center > TOL)})
                vals.append(q)
    
            cand_obj = metric([s["dCand"] for s in vals], [s["d_obj_correct"] for s in vals])
            cand_sem = metric([s["dCand"] for s in vals], [s["d_semantic_score"] for s in vals])
            pos = [s for s in vals if s["dCand"] > TOL]
            center_false = [s for s in pos if s["dJH"] <= TOL]
            center_dom = [s for s in pos if s["center_contribution"] > max(s["heavy_contribution"], 0.0)]
            sem_false_obj = [s for s in pos if s["d_obj_correct"] < -TOL]
            sem_false_sem = [s for s in pos if s["d_semantic_score"] < -TOL]
            pf = [s for s in vals if s["pass_from"] and not s["pass_to"]]
            pf_approved = [s for s in pf if s["dCand"] > TOL]
            pf_guard_approved = [s for s in pf if s["dCand"] > TOL and s["guardrail_ok"]]
            guard = [s for s in vals if s["guardrail_ok"]]
            guard_pos = [s for s in guard if s["dCand"] > TOL]
            guard_obj = (metric([s["dCand"] for s in guard], [s["d_obj_correct"] for s in guard])
                         if len(guard) >= 2 else None)
            guard_sem = (metric([s["dCand"] for s in guard], [s["d_semantic_score"] for s in guard])
                         if len(guard) >= 2 else None)
    
            per_seed, signs = {}, []
            for sd in sorted({s["seed"] for s in vals}):
                ss = [s for s in vals if s["seed"] == sd]
                rr = spearman([s["dCand"] for s in ss], [s["d_obj_correct"] for s in ss])
                per_seed[str(sd)] = rr
                if np.isfinite(rr):
                    signs.append(np.sign(rr))
            overall_sign = np.sign(cand_obj["spearman"]) if np.isfinite(cand_obj["spearman"]) else 0
            sign_cons = bool(len(signs) == 3 and all(x == overall_sign for x in signs))
            keep_frac = frac(guard_pos, pos, 0.0)
    
            c = {"spearman_gain_ge_0p10": cand_obj["spearman"] >= base_obj["spearman"] + .10,
                 "semantic_false_gain_obj_le_0p25": frac(sem_false_obj, pos, 1.0) <= .25,
                 "center_degradation_false_gain_le_0p10": frac(center_false, pos, 1.0) <= .10,
                 "guardrailed_pass_fail_false_approval_le_0p20": frac(pf_guard_approved, pf, 1.0) <= .20,
                 "guardrail_retains_ge_0p60_positive": keep_frac >= .60,
                 "seed_sign_consistency_3_of_3": sign_cons}
            block = {
                "eta": eta, "candidate_obj_metric": cand_obj, "candidate_sem_metric": cand_sem,
                "guardrail_obj_metric": guard_obj, "guardrail_sem_metric": guard_sem,
                "candidate_positive_n": len(pos), "guardrail_admissible_n": len(guard),
                "guardrail_positive_n": len(guard_pos),
                "guardrail_positive_retention_fraction": keep_frac,
                "center_degradation_false_gain_fraction": frac(center_false, pos),
                "center_dominated_positive_fraction": frac(center_dom, pos),
                "semantic_false_gain_obj_fraction": frac(sem_false_obj, pos),
                "semantic_false_gain_score_fraction": frac(sem_false_sem, pos),
                "pass_to_fail_n": len(pf),
                "pass_to_fail_false_approval_fraction": frac(pf_approved, pf),
                "guardrailed_pass_to_fail_false_approval_fraction": frac(pf_guard_approved, pf),
                "per_seed_spearman": per_seed, "seed_sign_consistent": sign_cons, "criteria": c,
                "center_false_gain_events": list(center_false)}
            return block, c
    
        def analyze(self):
            samples = self.samples()
            base_obj = metric([s["dJH"] for s in samples], [s["d_obj_correct"] for s in samples])
            base_sem = metric([s["dJH"] for s in samples], [s["d_semantic_score"] for s in samples])
            rel_obj = metric([s["dR"] for s in samples], [s["d_obj_correct"] for s in samples])
            rel_sem = metric([s["dR"] for s in samples], [s["d_semantic_score"] for s in samples])
    
            per_eta, criteria_by_eta = {}, {}
            for eta in ETAS:
                per_eta[str(eta)], criteria_by_eta[str(eta)] = self.eta_block(samples, eta, base_obj)
    
            rel_confirm = rel_obj["spearman"] >= base_obj["spearman"] + .15
            all_eta = all(all(c.values()) for c in criteria_by_eta.values())
            criteria = {"pure_relation_beats_J_by_0p15": rel_confirm, "all_eta_pass": all_eta}
            authorized = bool(rel_confirm and all_eta)
            status = ("AUTHORIZED" if authorized
                      else "RELATION VALID BUT CANDIDATE PATHOLOGICAL" if rel_confirm
                      else "NO BENEFIT")
            return {"schema": self.schema, "status": status, "offline_only": True,
                    "optimizer_steps": 0, "sample_count": len(samples), "eta_values": list(ETAS),
                    "baseline_J_obj_metric": base_obj, "baseline_J_sem_metric": base_sem,
                    "pure_relation_obj_metric": rel_obj, "pure_relation_sem_metric": rel_sem,
                    "per_eta": per_eta, "criteria_by_eta": criteria_by_eta, "criteria": criteria,
                    "decision": {"objective_design_branch_authorized": authorized},
                    "samples": samples}
    
        def execute(self):
            rep = super().execute()
            self.write({"status": "FROZEN_BY_HASH", "decision": rep["status"],
                        "report_sha256": self.sha(self.out / self.report),
                        "script_sha256": self.sha(Path(__file__).resolve()),
                        "contract_sha256": self.sha(REPO / CONTRACT),
                        "source_sha256": self.sha(RUNS / SRC)},
                       "PROVENANCE_MANIFEST.json")
            return rep
    
        def summarize(self, rep):
            keep = ["candidate_obj_metric", "candidate_sem_metric",
                    "guardrail_positive_retention_fraction", "center_degradation_false_gain_fraction",
                    "center_dominated_positive_fraction", "semantic_false_gain_obj_fraction",
                    "semantic_false_gain_score_fraction", "pass_to_fail_false_approval_fraction",
                    "guardrailed_pass_to_fail_false_approval_fraction", "per_seed_spearman",
                    "criteria"]
            print(json.dumps({"status": rep["status"], "criteria": rep["criteria"],
                              "baseline_J_obj_metric": rep["baseline_J_obj_metric"],
                              "pure_relation_obj_metric": rep["pure_relation_obj_metric"],
                              "per_eta": {k: {z: v[z] for z in keep}
                                          for k, v in rep["per_eta"].items()}}, indent=2))
    
    
    if True:
        RelationalObjectiveDesignAudit.main()

def run_relational_persistence_heldout_audit():
    """Run former relational_persistence_heldout_audit.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import hashlib,json,sys
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    from rl.experiments.common.utilities.trajectory_information_attribution_audit import (
        obs_tensor,trace,slope,rolling_worst,rankdata,corr,spearman,checkpoint_model_hash,
        PREFS,ORDER,IDX,PHYS,NENV,STEPS,SUITES,TOL
    )
    OUT=ROOT/'runs/relational_persistence_heldout-2026-09-24'
    BASE=ROOT/'runs/v2b_adam_continuous-2026-09-24/model_75.pt'
    SEEDS=(980001,981001,982001)
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def seq(seed):return [('base',BASE)]+[(f'u{u}',ROOT/f'runs/v2b_bounded_deltaa_multiseed-2026-09-24/seed_{seed}/control_u{u}.pt') for u in range(1,9)]
    
    def descriptors(h,c,a):
        j=IDX[a];pk=PHYS[a]
        rh=h['obj'][:,j];rc=c['obj'][:,j];ph=h['phys'][pk];pc=c['phys'][pk]
        ao=rh-rc;ap=pc-ph
        return {
          'J_heavy':float(np.mean(rh)),
          'heavy_obj_late':float(np.mean(rh[43:])),
          'heavy_phys_late':float(-np.mean(ph[43:])),
          'obj_adv_mean':float(np.mean(ao)),
          'phys_adv_mean':float(np.mean(ap)),
          'obj_adv_late':float(np.mean(ao[43:])),
          'phys_adv_late':float(np.mean(ap[43:])),
          'obj_adv_late_minus_early':float(np.mean(ao[43:])-np.mean(ao[:21])),
          'phys_adv_late_minus_early':float(np.mean(ap[43:])-np.mean(ap[:21])),
          'obj_adv_slope':slope(ao),'phys_adv_slope':slope(ap),
          'obj_adv_final16':float(np.mean(ao[-16:])),'phys_adv_final16':float(np.mean(ap[-16:])),
          'obj_adv_worst16':rolling_worst(ao,16),'phys_adv_worst16':rolling_worst(ap,16),
          'obj_adv_positive_fraction':float(np.mean(ao>TOL)),'phys_adv_positive_fraction':float(np.mean(ap>TOL)),
        }
    
    def semantic_from_suites(suites,a):
        j=IDX[a];pk=PHYS[a];oo=[];pp=[]
        for s in suites:
            h=s[a];c=s['C']
            oo.append(float(np.mean(h['obj'][:,j])-np.mean(c['obj'][:,j]))>0)
            pp.append(float(np.mean(h['phys'][pk])-np.mean(c['phys'][pk]))<0)
        of=float(np.mean(oo));pf=float(np.mean(pp));score=.5*(of+pf)
        return {'objective_correct_fraction':of,'physical_correct_fraction':pf,'semantic_score':score,
                'PASS':bool(of>=.75 and pf>=.75)}
    
    def descriptor_target(name):
        if name.startswith('phys_') or '_phys_' in name:return 'd_phys_correct'
        return 'd_obj_correct'
    
    def stats(samples,name):
        x=np.asarray([s['changes'][name] for s in samples],float);target=descriptor_target(name);y=np.asarray([s[target] for s in samples],float)
        sp=spearman(x,y);pe=corr(x,y);mask=(np.abs(x)>TOL)&(np.abs(y)>TOL)
        sign=float(np.mean(np.sign(x[mask])==np.sign(y[mask]))) if mask.any() else float('nan')
        perseed={}
        for sd in SEEDS:
            ss=[s for s in samples if s['seed']==sd];perseed[str(sd)]=spearman([s['changes'][name] for s in ss],[s[target] for s in ss])
        peraxis={}
        for a in ORDER:
            ss=[s for s in samples if s['axis']==a];peraxis[a]=spearman([s['changes'][name] for s in ss],[s[target] for s in ss])
        return {'target':target,'spearman':sp,'pearson':pe,'sign_agreement':sign,'per_seed_spearman':perseed,'per_axis_spearman':peraxis}
    
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
          for ci,(hh,p) in enumerate(hash_to_path.items(),1):
            print(f'CHECKPOINT {ci}/{len(hash_to_path)} {p.relative_to(ROOT)}',flush=True)
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);m.eval();ss=[]
            for suite in range(SUITES):
              sd=840001+suite;t={}
              for lab in ('T','A','O','S','C'):t[lab]=trace(env,m,mgr,robot,PREFS[lab],sd)
              ss.append(t)
            traces[hh]=ss
          states={};samples=[]
          for sd in SEEDS:
            arr=[]
            for lab,p in seq(sd):
              suites=traces[refhash[(sd,lab)]];axes={}
              for a in ORDER:
                dd=[descriptors(s[a],s['C'],a) for s in suites];d={k:float(np.mean([q[k] for q in dd])) for k in dd[0]};d.update(semantic_from_suites(suites,a));axes[a]=d
              arr.append({'label':lab,'checkpoint':str(p.relative_to(ROOT)),'axes':axes})
            states[str(sd)]=arr
            for i in range(len(arr)-1):
              for a in ORDER:
                x=arr[i]['axes'][a];y=arr[i+1]['axes'][a]
                changes={k:y[k]-x[k] for k in x if k not in ('objective_correct_fraction','physical_correct_fraction','semantic_score','PASS')}
                samples.append({'seed':sd,'from':arr[i]['label'],'to':arr[i+1]['label'],'axis':a,'changes':changes,
                                'd_obj_correct':y['objective_correct_fraction']-x['objective_correct_fraction'],
                                'd_phys_correct':y['physical_correct_fraction']-x['physical_correct_fraction'],
                                'd_semantic_score':y['semantic_score']-x['semantic_score'],'pass_from':x['PASS'],'pass_to':y['PASS']})
          features=list(samples[0]['changes']);fs={f:stats(samples,f) for f in features}
          groups=['late','late_minus_early','slope','final16','worst16','positive_fraction']
          group_pass={g:bool(abs(fs[f'obj_adv_{g}']['spearman'])>=.5 and abs(fs[f'phys_adv_{g}']['spearman'])>=.5) for g in groups}
          reltemp=[f'{side}_adv_{g}' for g in groups for side in ('obj','phys')]
          med_rel=float(np.median([abs(fs[f]['spearman']) for f in reltemp]));base=abs(fs['J_heavy']['spearman'])
          relation_add=bool(abs(fs['obj_adv_late']['spearman'])>=abs(fs['heavy_obj_late']['spearman'])+.10 and abs(fs['phys_adv_late']['spearman'])>=abs(fs['heavy_phys_late']['spearman'])+.10)
          temporal_add=bool(any(abs(fs[f'obj_adv_{g}']['spearman'])>=abs(fs['obj_adv_mean']['spearman'])+.10 for g in groups) and any(abs(fs[f'phys_adv_{g}']['spearman'])>=abs(fs['phys_adv_mean']['spearman'])+.10 for g in groups))
          key4=('obj_adv_late','phys_adv_late','obj_adv_late_minus_early','phys_adv_late_minus_early')
          seed_cons=True
          for f in key4:
            sg=np.sign(fs[f]['spearman']);seed_cons &= all(np.isfinite(v) and np.sign(v)==sg for v in fs[f]['per_seed_spearman'].values())
          pf=[s for s in samples if s['pass_from'] and not s['pass_to']]
          def deterior_frac(f):return float(np.mean([s['changes'][f]<-TOL for s in pf])) if pf else float('nan')
          late_both=float(np.mean([(s['changes']['obj_adv_late']<-TOL and s['changes']['phys_adv_late']<-TOL) for s in pf])) if pf else float('nan')
          jdet=deterior_frac('J_heavy')
          criteria={
           'at_least_4_of_6_relational_temporal_groups':sum(group_pass.values())>=4,
           'median_reltemp_beats_J_by_0p15':med_rel>=base+.15,
           'relational_late_beats_heavy_late_both_sides_by_0p10':relation_add,
           'temporal_beats_relational_mean_both_sides_by_0p10':temporal_add,
           'key_temporal_seed_sign_consistency_3_of_3':bool(seed_cons),
           'pass_to_fail_event_count_ge4':len(pf)>=4,
           'pass_to_fail_late_both_deterioration_ge0p75':bool(len(pf)>=4 and late_both>=.75),
           'pass_to_fail_J_sensitivity_at_least_0p15_lower':bool(len(pf)>=4 and jdet<=late_both-.15),
          }
          passed=all(criteria.values()); status='HELD-OUT VALIDATED' if passed else ('PARTIAL / INCONCLUSIVE' if any(criteria.values()) else 'NOT REPRODUCED')
          rep={'schema':'relational_persistence_heldout_v1','status':status,'measurement_only':True,'optimizer_steps':0,'primary_control_only':True,
               'training_seeds':list(SEEDS),'unique_checkpoints':len(hash_to_path),'transitions':24,'axis_transition_samples':len(samples),
               'states':states,'feature_stats':fs,'group_pass':group_pass,'median_reltemp_abs_spearman':med_rel,'baseline_J_abs_spearman':base,
               'pass_to_fail_n':len(pf),'pass_to_fail_late_both_deterioration_fraction':late_both,'pass_to_fail_J_deterioration_fraction':jdet,
               'pass_to_fail_events':pf,'criteria':criteria,'decision':{'reward_design_branch_authorized':bool(passed)},'samples':samples}
          OUT.mkdir(parents=True,exist_ok=True);out=OUT/'heldout_report.json';out.write_text(json.dumps(rep,indent=2)+'\n')
          (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':status,'report_sha256':sha(out),'script_sha256':sha(Path(__file__).resolve()),'contract_sha256':sha(ROOT/'docs/contracts/diagnostics/relational-persistence-heldout-contract.md'),'checkpoint_sha256':{str(p.relative_to(ROOT)):sha(p) for _,_,p in refs}},indent=2)+'\n')
          compact={'status':status,'criteria':criteria,'group_pass':group_pass,'median_reltemp_abs_spearman':med_rel,'baseline_J_abs_spearman':base,'pass_to_fail_n':len(pf),'pass_to_fail_late_both_deterioration_fraction':late_both,'pass_to_fail_J_deterioration_fraction':jdet,'selected':{f:fs[f] for f in ['J_heavy','heavy_obj_late','heavy_phys_late','obj_adv_mean','phys_adv_mean','obj_adv_late','phys_adv_late','obj_adv_late_minus_early','phys_adv_late_minus_early','obj_adv_slope','phys_adv_slope','obj_adv_final16','phys_adv_final16','obj_adv_worst16','phys_adv_worst16']}}
          print(json.dumps(compact,indent=2),flush=True)
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

STAGES = {
    "relational_objective_design_audit": run_relational_objective_design_audit,
    "relational_persistence_heldout_audit": run_relational_persistence_heldout_audit,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
