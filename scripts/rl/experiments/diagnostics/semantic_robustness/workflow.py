"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_semantic_gate_factorization_audit():
    """Run former semantic_gate_factorization_audit.py stage."""
    
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
        obs_tensor,trace,checkpoint_model_hash,PREFS,ORDER,IDX,PHYS,NENV,STEPS,SUITES,TOL
    )
    OUT=ROOT/'runs/semantic_gate_factorization_audit-2026-09-24'
    BASE=ROOT/'runs/v2b_adam_continuous-2026-09-24/model_75.pt'
    SEEDS=(980001,981001,982001)
    PHASES={'early':(0,21),'mid':(21,43),'late':(43,64)}
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def seq(seed): return [('base',BASE)]+[(f'u{u}',ROOT/f'runs/v2b_bounded_deltaa_multiseed-2026-09-24/seed_{seed}/control_u{u}.pt') for u in range(1,9)]
    
    def suite_axis_factors(h,c,a):
        j=IDX[a];pk=PHYS[a]
        ao=h['obj'][:,j]-c['obj'][:,j]
        ap=c['phys'][pk]-h['phys'][pk]
        obj=float(np.mean(ao)); phys=float(np.mean(ap))
        ph={}
        for name,(lo,hi) in PHASES.items():
            om=float(np.mean(ao[lo:hi]));pm=float(np.mean(ap[lo:hi]))
            ph[name]={'obj_margin':om,'phys_margin':pm,'obj_correct':bool(om>0),'phys_correct':bool(pm>0)}
        def flips(vals):
            s=[1 if x>TOL else -1 if x<-TOL else 0 for x in vals];nz=[x for x in s if x]
            return int(sum(nz[i]!=nz[i-1] for i in range(1,len(nz))))
        obj_phase=[ph[x]['obj_margin'] for x in ('early','mid','late')]
        phys_phase=[ph[x]['phys_margin'] for x in ('early','mid','late')]
        return {'obj_margin':obj,'phys_margin':phys,'obj_correct':bool(obj>0),'phys_correct':bool(phys>0),
                'joint_correct':bool(obj>0 and phys>0),'phases':ph,
                'obj_phase_positive_count':int(sum(x>0 for x in obj_phase)),
                'phys_phase_positive_count':int(sum(x>0 for x in phys_phase)),
                'obj_phase_flip_count':flips(obj_phase),'phys_phase_flip_count':flips(phys_phase),
                'obj_worst_phase_margin':float(min(obj_phase)),'phys_worst_phase_margin':float(min(phys_phase)),
                'obj_negative_phase_count':int(sum(x<0 for x in obj_phase)),
                'phys_negative_phase_count':int(sum(x<0 for x in phys_phase))}
    
    def aggregate_axis(suites,a):
        sf=[suite_axis_factors(s[a],s['C'],a) for s in suites]
        om=np.array([x['obj_margin'] for x in sf]);pm=np.array([x['phys_margin'] for x in sf])
        obj_corr=float(np.mean(om>0));phys_corr=float(np.mean(pm>0));joint=float(np.mean([(x['joint_correct']) for x in sf]))
        obj_phys_agree=float(np.mean([(x['obj_correct']==x['phys_correct']) for x in sf]))
        obj_only=float(np.mean([(x['obj_correct'] and not x['phys_correct']) for x in sf]))
        phys_only=float(np.mean([((not x['obj_correct']) and x['phys_correct']) for x in sf]))
        obj_neg_cells=sum(x['obj_negative_phase_count'] for x in sf);phys_neg_cells=sum(x['phys_negative_phase_count'] for x in sf)
        return {
          'objective_correct_fraction':obj_corr,'physical_correct_fraction':phys_corr,'joint_correct_fraction':joint,
          'PASS':bool(obj_corr>=.75 and phys_corr>=.75),
          'obj_mean_margin':float(om.mean()),'phys_mean_margin':float(pm.mean()),
          'obj_std_margin':float(om.std()),'phys_std_margin':float(pm.std()),
          'obj_min_margin':float(om.min()),'phys_min_margin':float(pm.min()),
          'obj_max_margin':float(om.max()),'phys_max_margin':float(pm.max()),
          'obj_range_margin':float(om.max()-om.min()),'phys_range_margin':float(pm.max()-pm.min()),
          'obj_sign_disagreement_count':int(min(np.sum(om>0),np.sum(om<=0))),
          'phys_sign_disagreement_count':int(min(np.sum(pm>0),np.sum(pm<=0))),
          'obj_phys_sign_agreement_fraction':obj_phys_agree,
          'obj_only_correct_fraction':obj_only,'phys_only_correct_fraction':phys_only,
          'obj_negative_suite_phase_cells':int(obj_neg_cells),'phys_negative_suite_phase_cells':int(phys_neg_cells),
          'obj_worst_suite_phase_margin':float(min(x['obj_worst_phase_margin'] for x in sf)),
          'phys_worst_suite_phase_margin':float(min(x['phys_worst_phase_margin'] for x in sf)),
          'suite_factors':sf,
        }
    
    def transition_flags(a,b):
        mean_deg=(b['obj_mean_margin']<a['obj_mean_margin']-TOL) or (b['phys_mean_margin']<a['phys_mean_margin']-TOL)
        suite_loss=(b['objective_correct_fraction']<a['objective_correct_fraction']-TOL) or (b['physical_correct_fraction']<a['physical_correct_fraction']-TOL)
        worst_suite=((a['obj_min_margin']>=-TOL and b['obj_min_margin']<-TOL) or (b['obj_min_margin']<a['obj_min_margin']-TOL) or
                     (a['phys_min_margin']>=-TOL and b['phys_min_margin']<-TOL) or (b['phys_min_margin']<a['phys_min_margin']-TOL))
        phase=(b['obj_negative_suite_phase_cells']>a['obj_negative_suite_phase_cells'] or b['phys_negative_suite_phase_cells']>a['phys_negative_suite_phase_cells'])
        disagree=b['obj_phys_sign_agreement_fraction']<a['obj_phys_sign_agreement_fraction']-TOL
        hetero=((b['obj_std_margin']>a['obj_std_margin']+TOL and b['obj_mean_margin']>=a['obj_mean_margin']-TOL) or
                (b['phys_std_margin']>a['phys_std_margin']+TOL and b['phys_mean_margin']>=a['phys_mean_margin']-TOL))
        both_mean_nondec=(b['obj_mean_margin']>=a['obj_mean_margin']-TOL and b['phys_mean_margin']>=a['phys_mean_margin']-TOL)
        one_mean_nondec=(b['obj_mean_margin']>=a['obj_mean_margin']-TOL or b['phys_mean_margin']>=a['phys_mean_margin']-TOL)
        context_flip=False
        for old,new in zip(a['suite_factors'],b['suite_factors']):
            if (old['obj_correct'] and not new['obj_correct']) or (old['phys_correct'] and not new['phys_correct']):context_flip=True
            for ph in PHASES:
                if (old['phases'][ph]['obj_correct'] and not new['phases'][ph]['obj_correct']) or (old['phases'][ph]['phys_correct'] and not new['phases'][ph]['phys_correct']):context_flip=True
        return {'mean_relation_degradation':bool(mean_deg),'suite_sign_loss':bool(suite_loss),'worst_suite_collapse':bool(worst_suite),
                'phase_specific_collapse':bool(phase),'objective_physical_disagreement_growth':bool(disagree),'heterogeneity_growth':bool(hetero),
                'both_mean_relations_non_decreasing':bool(both_mean_nondec),'at_least_one_mean_relation_non_decreasing':bool(one_mean_nondec),
                'context_sign_flip_to_negative':bool(context_flip)}
    
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
              seed=840001+suite;t={}
              for lab in ('T','A','O','S','C'):t[lab]=trace(env,m,mgr,robot,PREFS[lab],seed)
              ss.append(t)
            traces[hh]=ss
          states={};events=[];alltrans=[]
          for sd in SEEDS:
            arr=[]
            for lab,p in seq(sd):
              suites=traces[refhash[(sd,lab)]];axes={a:aggregate_axis(suites,a) for a in ORDER}
              arr.append({'label':lab,'checkpoint':str(p.relative_to(ROOT)),'axes':axes})
            states[str(sd)]=arr
            for i in range(len(arr)-1):
              for a in ORDER:
                old=arr[i]['axes'][a];new=arr[i+1]['axes'][a];fl=transition_flags(old,new)
                rec={'seed':sd,'from':arr[i]['label'],'to':arr[i+1]['label'],'axis':a,'pass_from':old['PASS'],'pass_to':new['PASS'],'flags':fl,
                     'delta':{'obj_mean_margin':new['obj_mean_margin']-old['obj_mean_margin'],'phys_mean_margin':new['phys_mean_margin']-old['phys_mean_margin'],
                              'objective_correct_fraction':new['objective_correct_fraction']-old['objective_correct_fraction'],'physical_correct_fraction':new['physical_correct_fraction']-old['physical_correct_fraction'],
                              'obj_std_margin':new['obj_std_margin']-old['obj_std_margin'],'phys_std_margin':new['phys_std_margin']-old['phys_std_margin'],
                              'obj_negative_suite_phase_cells':new['obj_negative_suite_phase_cells']-old['obj_negative_suite_phase_cells'],
                              'phys_negative_suite_phase_cells':new['phys_negative_suite_phase_cells']-old['phys_negative_suite_phase_cells']}}
                alltrans.append(rec)
                if old['PASS'] and not new['PASS']:events.append(rec)
          n=len(events)
          def frac(flag):return float(np.mean([e['flags'][flag] for e in events])) if n else float('nan')
          # heavy J deterioration baseline from previous heldout report samples matched by transition
          hd=json.load(open(ROOT/'runs/relational_persistence_heldout-2026-09-24/heldout_report.json'))
          lookup={(s['seed'],s['from'],s['to'],s['axis']):s for s in hd['samples']}
          jdet=float(np.mean([lookup[(e['seed'],e['from'],e['to'],e['axis'])]['changes']['J_heavy']<-TOL for e in events])) if n else float('nan')
          perseed={str(sd):[e for e in events if e['seed']==sd] for sd in SEEDS}
          seed_pattern=all(len(es)>0 and np.mean([x['flags']['worst_suite_collapse'] or x['flags']['phase_specific_collapse'] for x in es])>=.5 for es in perseed.values())
          any_context=frac('worst_suite_collapse')
          phase_frac=frac('phase_specific_collapse')
          context_union=float(np.mean([e['flags']['worst_suite_collapse'] or e['flags']['phase_specific_collapse'] for e in events])) if n else float('nan')
          mean_nondec=float(np.mean([e['flags']['at_least_one_mean_relation_non_decreasing'] for e in events])) if n else float('nan')
          bothmean_or_hetero=float(np.mean([e['flags']['both_mean_relations_non_decreasing'] or (e['flags']['at_least_one_mean_relation_non_decreasing'] and e['flags']['heterogeneity_growth']) for e in events])) if n else float('nan')
          factors={'pass_to_fail_n':n,'mean_relation_degradation_fraction':frac('mean_relation_degradation'),'suite_sign_loss_fraction':frac('suite_sign_loss'),
                   'worst_suite_collapse_fraction':any_context,'phase_specific_collapse_fraction':phase_frac,'worst_or_phase_collapse_fraction':context_union,
                   'objective_physical_disagreement_growth_fraction':frac('objective_physical_disagreement_growth'),'heterogeneity_growth_fraction':frac('heterogeneity_growth'),
                   'at_least_one_mean_non_decreasing_fraction':mean_nondec,'both_mean_nondec_or_one_plus_hetero_fraction':bothmean_or_hetero,
                   'context_sign_flip_fraction':frac('context_sign_flip_to_negative'),'heavy_J_deterioration_fraction':jdet,
                   'per_seed_pass_to_fail_count':{k:len(v) for k,v in perseed.items()}}
          criteria={
            'context_collapse_ge_0p75':context_union>=.75,
            'mean_nondec_events_ge_0p25':mean_nondec>=.25,
            'both_mean_nondec_or_one_plus_hetero_ge_0p20':bothmean_or_hetero>=.20,
            'context_factor_exceeds_heavy_J_deterioration':context_union>jdet,
            'pattern_present_all_3_seeds':bool(seed_pattern),
          }
          supported=all(criteria.values());status='DISTRIBUTIONAL HYPOTHESIS SUPPORTED' if supported else ('MOSTLY MEAN-DRIVEN' if factors['mean_relation_degradation_fraction']>=.8 and context_union<=jdet+.05 else 'INCONCLUSIVE')
          rep={'schema':'semantic_gate_factorization_audit_v1','status':status,'read_only':True,'optimizer_steps':0,'training_seeds':list(SEEDS),
               'unique_checkpoints':len(hash_to_path),'states':states,'pass_to_fail_events':events,'all_transitions':alltrans,'factor_summary':factors,'criteria':criteria,
               'decision':{'distributional_objective_training_authorized':False,'distributional_hypothesis_supported':supported}}
          OUT.mkdir(parents=True,exist_ok=True);out=OUT/'semantic_gate_factorization_report.json';out.write_text(json.dumps(rep,indent=2)+'\n')
          (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','decision':status,'report_sha256':sha(out),'script_sha256':sha(Path(__file__).resolve()),'contract_sha256':sha(ROOT/'docs/contracts/diagnostics/semantic-gate-factorization-contract.md'),'checkpoint_sha256':{str(p.relative_to(ROOT)):sha(p) for _,_,p in refs}},indent=2)+'\n')
          print(json.dumps({'status':status,'factor_summary':factors,'criteria':criteria,'events':events},indent=2),flush=True)
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_semantic_gate_robustness_audit():
    """Run former semantic_gate_robustness_audit.py stage."""
    """Are the semantic-gate pass-to-fail events robust to how the gate is scored?
    
    Rescales each axis by the discovery run's median margin, checks that the
    rescaled gate reproduces the recorded PASS exactly, then for every recorded
    pass-to-fail event measures boundary clearance, a threshold sweep and a
    resampling of the four suites.
    """
    import itertools
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import REPO, OfflineAudit
    
    AXES = ("T", "A", "O", "S")
    FACTORIZATION = ("semantic_gate_factorization_audit-2026-09-24/"
                     "semantic_gate_factorization_report.json")
    DISCOVERY = ("trajectory_information_attribution_audit-2026-09-24/"
                 "trajectory_information_attribution_report.json")
    CONTRACT = "docs/contracts/diagnostics/semantic-gate-robustness-audit-contract.md"
    TAUS = (-0.25, 0.0, 0.25)
    
    
    def gate_margin(vals):
        """The second-smallest margin: the gate needs three of four suites."""
        return float(np.sort(np.asarray(vals, float))[1])
    
    
    def pass_at(obj, phys, tau=0.0):
        return bool(np.sum(np.asarray(obj) > tau) >= 3 and np.sum(np.asarray(phys) > tau) >= 3)
    
    
    class SemanticGateRobustnessAudit(OfflineAudit):
        """Robustness of the semantic-gate pass-to-fail events to gate scoring."""
    
        run = "semantic_gate_robustness_audit-2026-09-24"
        report = "semantic_gate_robustness_report.json"
        schema = "semantic_gate_robustness_audit_v1"
    
        def scales(self, disc):
            """Per-axis median |advantage|, used to put both channels on one scale."""
            out = {}
            for a in AXES:
                om, pm = [], []
                for arr in disc["states"].values():
                    for st in arr:
                        q = st["axes"][a]
                        om.append(abs(float(q["aggregate.obj_adv_mean"])))
                        pm.append(abs(float(q["aggregate.phys_adv_mean"])))
                out[a] = {"obj": float(np.median(om) + 1e-8), "phys": float(np.median(pm) + 1e-8)}
            return out
    
        def rescaled_states(self, fac, scales):
            """Every state's rescaled margins, plus any disagreement with recorded PASS."""
            states, mismatch = {}, []
            for sd, arr in fac["states"].items():
                rows = []
                for st in arr:
                    axes = {}
                    for a in AXES:
                        q = st["axes"][a]
                        zo = [float(x["obj_margin"]) / scales[a]["obj"] for x in q["suite_factors"]]
                        zp = [float(x["phys_margin"]) / scales[a]["phys"] for x in q["suite_factors"]]
                        go, gp = gate_margin(zo), gate_margin(zp)
                        gs = min(go, gp)
                        if (gs > 0) != bool(q["PASS"]):
                            mismatch.append((sd, st["label"], a, gs, q["PASS"]))
                        axes[a] = {"z_obj": zo, "z_phys": zp, "G_obj": go, "G_phys": gp,
                                   "G_sem": gs, "PASS": bool(q["PASS"])}
                    rows.append({"label": st["label"], "axes": axes})
                states[sd] = rows
            return states, mismatch
    
        def event(self, e, states):
            sd, a = str(e["seed"]), e["axis"]
            arr = states[sd]
            s = next(x for x in arr if x["label"] == e["from"])["axes"][a]
            t = next(x for x in arr if x["label"] == e["to"])["axes"][a]
            clearance = min(s["G_sem"], -t["G_sem"])
    
            sweep = {str(tau): {"source_pass": pass_at(s["z_obj"], s["z_phys"], tau),
                                "target_pass": pass_at(t["z_obj"], t["z_phys"], tau)}
                     for tau in TAUS}
            sweep_rob = all(v["source_pass"] and not v["target_pass"] for v in sweep.values())
    
            flips = srcpass = tgtpass = 0
            for idxs in itertools.product(range(4), repeat=4):
                ps = pass_at([s["z_obj"][k] for k in idxs], [s["z_phys"][k] for k in idxs], 0)
                pt = pass_at([t["z_obj"][k] for k in idxs], [t["z_phys"][k] for k in idxs], 0)
                srcpass += ps
                tgtpass += pt
                flips += (ps and not pt)
            pflip, psrc, ptgt = flips / 256, srcpass / 256, tgtpass / 256
    
            if clearance < 0.10 or pflip < 0.50 or not sweep_rob:
                cls = "A"
            elif clearance >= 0.25 and sweep_rob and pflip >= 0.75:
                cls = "B"
            else:
                cls = "MIXED"
            return {"seed": e["seed"], "from": e["from"], "to": e["to"], "axis": a,
                    "G_source": s["G_sem"], "G_target": t["G_sem"], "clearance": clearance,
                    "delta_G": t["G_sem"] - s["G_sem"], "sweep": sweep,
                    "threshold_sweep_robust_flip": sweep_rob,
                    "bootstrap_source_pass_prob": psrc, "bootstrap_target_pass_prob": ptgt,
                    "bootstrap_pass_to_fail_prob": pflip, "class": cls,
                    "target_limiting_channel": "objective" if t["G_obj"] <= t["G_phys"] else "physical"}
    
        def analyze(self):
            fac = self.load(Path(FACTORIZATION).name, str(Path(FACTORIZATION).parent))
            disc = self.load(Path(DISCOVERY).name, str(Path(DISCOVERY).parent))
            scales = self.scales(disc)
            states, mismatch = self.rescaled_states(fac, scales)
            events = [self.event(e, states) for e in fac["pass_to_fail_events"]]
    
            n = len(events)
            counts = {k: sum(e["class"] == k for e in events) for k in ("A", "B", "MIXED")}
            medc = float(np.median([e["clearance"] for e in events]))
            medb = float(np.median([e["bootstrap_pass_to_fail_prob"] for e in events]))
            bseeds = {e["seed"] for e in events if e["class"] == "B"}
            robust = (counts["B"] / n >= .60 and counts["A"] / n <= .25
                      and medc >= .25 and medb >= .75 and len(bseeds) == 3)
            sens = counts["A"] / n >= .50 or medc < .10 or medb < .50
            status = ("FORGETTING ROBUST" if robust
                      else "EVALUATOR SENSITIVITY MATERIAL" if sens else "MIXED / INCONCLUSIVE")
            return {"schema": self.schema, "status": status, "read_only": True, "scale": scales,
                    "equivalence_mismatches": mismatch, "event_count": n, "class_counts": counts,
                    "class_fractions": {k: counts[k] / n for k in counts},
                    "median_boundary_clearance": medc,
                    "median_bootstrap_pass_to_fail_probability": medb,
                    "class_B_seeds": sorted(bseeds), "events": events,
                    "decision": {"training_method_authorized": False,
                                 "evaluation_contract_change_authorized": False}}
    
        def execute(self):
            rep = super().execute()
            self.write({"status": "FROZEN_BY_HASH", "decision": rep["status"],
                        "report_sha256": self.sha(self.out / self.report),
                        "script_sha256": self.sha(Path(__file__).resolve()),
                        "contract_sha256": self.sha(REPO / CONTRACT),
                        "factorization_sha256": self.sha(self.dir.parent / FACTORIZATION),
                        "discovery_sha256": self.sha(self.dir.parent / DISCOVERY)},
                       "PROVENANCE_MANIFEST.json")
            return rep
    
        def summarize(self, rep):
            print(json.dumps({k: rep[k] for k in
                              ("status",)} | {"equivalence_mismatches": len(rep["equivalence_mismatches"])}
                             | {k: rep[k] for k in ("class_counts", "class_fractions",
                                                    "median_boundary_clearance",
                                                    "median_bootstrap_pass_to_fail_probability",
                                                    "events")}, indent=2))
    
    
    if True:
        SemanticGateRobustnessAudit.main()

STAGES = {
    "semantic_gate_factorization_audit": run_semantic_gate_factorization_audit,
    "semantic_gate_robustness_audit": run_semantic_gate_robustness_audit,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
