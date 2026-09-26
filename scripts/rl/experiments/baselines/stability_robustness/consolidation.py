"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_b1_c0_consolidation():
    """Run former b1_c0_consolidation.py stage."""
    """Read-only consolidation of residual failures across P2/S1/R1."""
    from pathlib import Path
    import json, math
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_c0_consolidation'; OUT.mkdir(parents=True,exist_ok=True)
    BRANCHES={'P2':'b1_p2','S1':'b1_s1','R1':'b1_r1'}
    def main():
        rows=[]
        for branch,prefix in BRANCHES.items():
            for seed in range(3):
                run=ROOT/f'runs/{prefix}_seed{seed}_2026-09-21'
                if branch=='S1' and seed==0 and not (run/'RUN_DONE.json').exists(): run=ROOT/'runs/b1_s1_seed0_retry1_2026-09-21'
                m=json.loads((run/'monitor/u500.json').read_text())['acceptance']; tm=json.loads((run/'training_metrics.json').read_text())['metrics'];
                finite=all(math.isfinite(float(x[k])) for x in tm for k in ('analytic_kl','approx_kl'))
                grad=[x['actor_grad_norm'] for x in tm if 'actor_grad_norm' in x]
                row={'branch':branch,'seed':seed,'survival':m['survival_rate'],'vx_mae':m['mean_abs_vx_error'],'tilt_p95_deg':math.degrees(m['tilt_p95_rad']),'tilt_max_deg':math.degrees(m['tilt_max_rad']),'mean_displacement':m['mean_displacement'],'gate':m['survival_rate']>=.9 and m['mean_abs_vx_error']<=.15 and math.degrees(m['tilt_p95_rad'])<=15 and math.degrees(m['tilt_max_rad'])<=30,'finite':finite,'mean_analytic_kl':sum(x['analytic_kl'] for x in tm)/len(tm),'max_analytic_kl':max(x['analytic_kl'] for x in tm),'mean_actor_grad':sum(grad)/len(grad) if grad else None}
                if branch=='R1': row.update({'reconstruction_first':tm[0]['reconstruction_loss'],'reconstruction_last':tm[-1]['reconstruction_loss'],'reconstruction_grad_mean':sum(x['reconstruction_grad_norm'] for x in tm)/len(tm)})
                rows.append(row)
        out={'schema':'b1_c0_consolidation_v1','read_only':True,'branches':list(BRANCHES),'rows':rows,'decision':'no_common_single mechanism yet; residual failure is seed-dependent and mixed (tracking failure in seed0, large-angle robustness failure in seed2)'}
        (OUT/'B1_C0_CONSOLIDATION.json').write_text(json.dumps(out,indent=2)+'\n')
        lines=['# B1-C0 residual-failure consolidation','', '|branch|seed|survival|vx MAE|tilt p95°|max tilt°|displacement|KL mean|max KL|gate|','|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
        for x in rows: lines.append(f"|{x['branch']}|{x['seed']}|{x['survival']:.3f}|{x['vx_mae']:.3f}|{x['tilt_p95_deg']:.2f}|{x['tilt_max_deg']:.2f}|{x['mean_displacement']:.3f}|{x['mean_analytic_kl']:.4g}|{x['max_analytic_kl']:.4g}|{x['gate']}|")
        lines += ['', '## Residual signatures', '', '- Seed 0: R1 reaches the tilt gate but remains outside the tracking gate; this is not explained by a pure stability-margin failure.', '- Seed 2: R1 improves survival/tracking relative to P2 but retains a large-angle tilt failure; this is a robustness-tail failure.', '- Seed 1: R1 passes the final gate, showing that the representation intervention can be sufficient in at least one basin.', '', 'Conclusion: no single causal mechanism is established across seeds. Do not sweep coefficients or stack branches; keep B0.2/MOPPO closed pending the next formulation decision.']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n'); print(OUT/'report.md')
    if True: main()

def run_b1p_consecutive_policy_kl_audit():
    """Run former b1p_consecutive_policy_kl_audit.py stage."""
    """Read-only pre-design KL audit for consecutive B0.1 policies."""
    from pathlib import Path
    import csv, json, sys
    import numpy as np
    import torch
    ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT/'scripts'))
    from rl.core.modules.actor_critic import ActorCritic
    OUT=ROOT/'artifacts'/'b1p_pre_design_kl'; OUT.mkdir(parents=True,exist_ok=True)
    OBS=torch.from_numpy(np.load(ROOT/'artifacts/b0_1_stability_audit/frozen_obs.npy')).float()
    def main():
     rows=[]
     for seed in range(3):
      run=ROOT/f'runs/b0_1_seed{seed}_2026-09-21'; ck=sorted((run/'checkpoints').glob('update_*.pt'), key=lambda p:int(p.stem.split('_')[-1]))
      m=ActorCritic(51,51,12,1,[64,64]).eval(); prev=None
      for p in ck:
       u=int(p.stem.split('_')[-1]); c=torch.load(p,map_location='cpu'); m.load_state_dict(c['model'])
       with torch.no_grad(): mu=m.raw_mean(OBS).double()
       std=float(c['scheduled_std']);
       if prev is not None:
        pu,pm,ps=prev; qstd=std
        # D_KL(N(pm,ps^2)||N(mu,std^2)), summed over action dimensions.
        kl=(np.log(qstd/ps)+(ps*ps+(pm-mu)**2)/(2*qstd*qstd)-0.5).sum(dim=1)
        target=json.loads((run/'monitor'/f'u{u:03d}.json').read_text())['acceptance']
        rows.append({'seed':seed,'from_update':pu,'to_update':u,'std_from':ps,'std_to':qstd,'kl_mean':float(kl.mean()),'kl_p95':float(torch.quantile(kl,.95)),'kl_max':float(kl.max()),'survival_to':target['survival_rate']})
       prev=(u,mu,std)
     fields=list(rows[0]);
     with (OUT/'consecutive_kl.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
     picks=[r for r in rows if (r['seed'],r['from_update'],r['to_update']) in {(1,25,50),(0,450,475),(2,250,275)}]
     summary={'schema':'b1p_consecutive_policy_kl_v1','read_only':True,'observation_shape':list(OBS.shape),'selected':picks,'formula':'KL(N(mu_k,std_k^2)||N(mu_k+1,std_k+1^2)), summed over 12 action dims','note':'pre-tanh Gaussian; scheduled_std read from each checkpoint'}
     (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
     lines=['# B1-P consecutive-policy KL audit','','Pre-tanh Gaussian KL on the independent 64×51 frozen observation set.','', '| seed | transition | mean KL | p95 KL | max KL | std | survival at target |','|---:|:---|---:|---:|---:|:---|---:|']
     for r in picks: lines.append(f"| {r['seed']} | {r['from_update']}→{r['to_update']} | {r['kl_mean']:.4f} | {r['kl_p95']:.4f} | {r['kl_max']:.4f} | {r['std_from']:.2f}→{r['std_to']:.2f} | {r['survival_to']:.3f} |")
     lines += ['', '## Stable-transition context', '']
     for seed in range(3):
      stable=[r for r in rows if r['seed']==seed and r['survival_to'] >= 0.9]
      if stable:
       lines.append(f"- seed {seed}: {len(stable)} transitions reached survival ≥0.90; KL mean range `{min(r['kl_mean'] for r in stable):.4f}–{max(r['kl_mean'] for r in stable):.4f}`.")
      else: lines.append(f"- seed {seed}: no transition reached survival ≥0.90.")
     lines += ['', 'This is a read-only pre-design calculation. The 64 monitor states are not used as a training anchor set.']
     (OUT/'report.md').write_text('\n'.join(lines)+'\n'); print(OUT/'report.md')
    if True: main()

STAGES = {
    "b1_c0_consolidation": run_b1_c0_consolidation,
    "b1p_consecutive_policy_kl_audit": run_b1p_consecutive_policy_kl_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
