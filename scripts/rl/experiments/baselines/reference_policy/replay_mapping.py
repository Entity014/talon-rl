"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_l0c2_audit():
    """Run former l0c2_audit.py stage."""
    """What the C1 snapshots can and cannot say about lane-level geometry.
    
    Read-only, and deliberately inconclusive: the C1 snapshots reduced action and
    state arrays to batch means before serialising, so the requested lane-level
    correlation is not computable from them. This reports the quantiles that do
    survive and names exactly what is missing, rather than inventing lane geometry.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import ARTIFACTS, RUNS, OfflineAudit
    
    RUNS_BY_SEED = {0: "l0c1_seed0_2026-09-22", 1: "l0c1_seed1_2026-09-22", 2: "l0c1_seed2_2026-09-22"}
    UPDATES = (200, 300, 400, 500)
    CONCLUSION = (
        "Existing C1 snapshots cannot answer the requested lane-level geometry "
        "correlation because action/state arrays were reduced to batch means before "
        "serialization. A truthful C2 requires read-only checkpoint replay that emits "
        "lane-indexed arrays (and a stable lane-to-evaluator-state mapping); no "
        "training or formulation change is authorized.")
    
    
    def quantiles(x):
        x = np.asarray(x, float)
        return {"n": int(x.size),
                "p50": float(np.percentile(x, 50)) if x.size else None,
                "p95": float(np.percentile(x, 95)) if x.size else None,
                "p99": float(np.percentile(x, 99)) if x.size else None,
                "max": float(x.max()) if x.size else None}
    
    
    class L0C2ArtifactGranularityAudit(OfflineAudit):
        """C2 lane-geometry audit, limited by what C1 actually serialised."""
    
        root = ARTIFACTS
        run = "l0c2_audit"
        report = "L0C2_AUDIT.json"
        sort_keys = True
        schema = "l0c2_lane_geometry_audit_v1"
    
        def analyze(self):
            out = {
                "schema": self.schema,
                "status": "INCONCLUSIVE_ARTIFACT_GRANULARITY",
                "updates": UPDATES,
                "available_lane_level": ["episode_lengths for completed episodes only"],
                "aggregate_only": ["action_norm", "action_saturation", "base_contact", "height",
                                   "roll", "pitch", "vx_error", "reward terms", "lane_age"],
                "missing_for_requested_c2": ["per-lane action norm/saturation",
                                             "per-joint action values",
                                             "per-lane actor mean",
                                             "per-lane tilt/height/contact/vx trajectories",
                                             "lane identity linking training snapshots to deterministic evaluator"],
                "seeds": {}}
            for s, run in RUNS_BY_SEED.items():
                out["seeds"][str(s)] = {}
                for u in UPDATES:
                    snap = json.loads((RUNS / run / "snapshots" / f"update_{u:03d}.json").read_text())
                    x = snap["telemetry"]
                    out["seeds"][str(s)][str(u)] = {
                        "completed_episode_length_quantiles": quantiles(x["episode_lengths"]),
                        "aggregate_action_norm": x["action_norm"],
                        "aggregate_action_saturation": x["action_saturation"],
                        "aggregate_vx_error": x["vx_error"]}
            out["conclusion"] = CONCLUSION
            return out
    
        def summarize(self, report):
            lines = ["# L0-C2 lane-level geometry audit", "",
                     "Status: **INCONCLUSIVE — existing artifact granularity**", "",
                     report["conclusion"], "",
                     "Requested lane-level quantiles/covariance cannot be computed from C1 "
                     "snapshots without inventing data. Only completed-episode length quantiles "
                     "and batch aggregates are reported in the JSON artifact."]
            (self.out / "report.md").write_text("\n".join(lines) + "\n")
            print(self.out / self.report)
    
    
    if True:
        L0C2ArtifactGranularityAudit.main()

def run_l0c2j_audit():
    """Run former l0c2j_audit.py stage."""
    """Joint-wise saturation and lane-level covariance over the C2R arrays.
    
    Read-only and descriptive: per joint it reports the signed action, how often
    and how long it saturates, the gap between lanes that fell and lanes that did
    not, how saturation co-varies with tilt/height/tracking/contact, and a 50-step
    summary before each first fall. It decides nothing.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import ARTIFACTS, OfflineAudit
    
    SEEDS = (0, 1, 2)
    UPDATE = 500
    SATURATION = 2.99
    PRECURSOR_STEPS = 50
    
    
    def quantiles(x):
        x = np.asarray(x, float)
        return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
                "max": float(np.max(x))}
    
    
    def longest_streak(v):
        best = cur = 0
        for x in v:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        return best
    
    
    class L0C2JJointGeometryAudit(OfflineAudit):
        """Per-joint saturation geometry at update 500, across the three C2R seeds."""
    
        root = ARTIFACTS
        run = "l0c2j_audit"
        report = "L0C2J_AUDIT.json"
        sort_keys = True
        schema = "l0c2j_joint_geometry_audit_v1"
    
        def joint_metrics(self, a, sat, fail, j):
            duration = np.array([longest_streak(sat[:, lane, j]) for lane in range(sat.shape[1])])
            any_fail, any_surv = fail.any(), (~fail).any()
            return {
                "joint": j,
                "abs_action": quantiles(np.abs(a[:, :, j]).reshape(-1)),
                "signed_mean": float(a[:, :, j].mean()),
                "signed_p05": float(np.percentile(a[:, :, j], 5)),
                "signed_p95": float(np.percentile(a[:, :, j], 95)),
                "saturation_fraction": float(sat[:, :, j].mean()),
                "lane_max_streak_p95": float(np.percentile(duration, 95)),
                "failed_sat_fraction": float(sat[:, fail, j].mean()) if any_fail else None,
                "survived_sat_fraction": float(sat[:, ~fail, j].mean()) if any_surv else None,
                "failure_delta": (float(sat[:, fail, j].mean() - sat[:, ~fail, j].mean())
                                  if any_fail and any_surv else None)}
    
        def seed(self, path):
            z = np.load(path)
            a, done = z["action"], z["done"].astype(bool)
            sat = np.abs(a) >= SATURATION
            roll, pitch, height, vx = z["roll"], z["pitch"], z["height"], z["vx"]
            contact = z["contact"].astype(bool)
            out = {}
            for ci, cmd in enumerate(z["commands"]):
                aa, ss = a[ci], sat[ci]
                fail = done[ci].any(0)
                tilt = np.maximum(np.abs(roll[ci]), np.abs(pitch[ci]))
                joints = [self.joint_metrics(aa, ss, fail, j) for j in range(aa.shape[-1])]
                co = ss.mean(-1)
                pair = np.corrcoef(np.stack([co.reshape(-1),
                                             np.abs(roll[ci]).reshape(-1),
                                             np.abs(pitch[ci]).reshape(-1),
                                             height[ci].reshape(-1),
                                             np.abs(vx[ci] - cmd[0]).reshape(-1),
                                             contact[ci].reshape(-1)]))
                pre = []
                for lane in range(ss.shape[1]):
                    idx = np.flatnonzero(done[ci, :, lane])
                    if not idx.size:
                        continue
                    t = int(idx[0])
                    lo = max(0, t - PRECURSOR_STEPS)
                    pre.append({"lane": lane, "first_fall": t,
                                "sat_mean_last50": float(ss[lo:t, lane].mean()),
                                "tilt_start": float(np.degrees(tilt[lo, lane])),
                                "tilt_end": float(np.degrees(tilt[max(t - 1, lo), lane])),
                                "height_start": float(height[ci, lo, lane]),
                                "height_end": float(height[ci, max(t - 1, lo), lane])})
                out[str(float(cmd[0]))] = {"joint_metrics": joints,
                                           "co_saturation_correlation": pair.tolist(),
                                           "co_saturation_pattern_p95": quantiles(co.reshape(-1)),
                                           "failed_lanes": int(fail.sum()),
                                           "precursor_last50": pre}
            return out
    
        def analyze(self):
            return {"schema": self.schema, "status": "COMPLETE_READ_ONLY",
                    "checkpoint_update": UPDATE,
                    "seeds": {str(s): self.seed(ARTIFACTS / f"l0c2r_seed{s}" / f"update_{UPDATE:03d}.npz")
                              for s in SEEDS}}
    
        def summarize(self, report):
            lines = ["# L0-C2J joint-wise geometry audit", "",
                     "Status: **COMPLETE — read-only**", "",
                     "This audit uses raw C2R arrays at update 500. It reports signed per-joint "
                     "action, saturation fraction/streak, survivor/fall deltas, co-saturation "
                     "correlations, and 50-step pre-fall summaries.", ""]
            for s in SEEDS:
                lines.append(f"## seed {s}")
                for cmd, m in report["seeds"][str(s)].items():
                    ranked = sorted((x for x in m["joint_metrics"] if x["failure_delta"] is not None),
                                    key=lambda x: x["failure_delta"], reverse=True)[:4]
                    lines.append(
                        f"- vx={cmd}: failed lanes={m['failed_lanes']}, "
                        f"co-sat p95={m['co_saturation_pattern_p95']['p95']:.3f}, "
                        "top failed-minus-survivor joints="
                        + ", ".join(f"j{x['joint']}:{x['failure_delta']:.3f}" for x in ranked))
            lines += ["", "No intervention decision is made by this descriptive audit."]
            (self.out / "report.md").write_text("\n".join(lines) + "\n")
            print(self.out / self.report)
    
    
    if True:
        L0C2JJointGeometryAudit.main()

def run_l0c2r_analyze():
    """Run former l0c2r_analyze.py stage."""
    """Summarize lane-indexed C2R replay arrays and saturation/failure links."""
    import json
    from pathlib import Path
    import numpy as np
    ROOT=Path(__file__).resolve().parents[4]; SEEDS=(0,1,2); UPDATES=(200,300,400,500)
    def q(a):
     a=np.asarray(a,float); return {'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99)),'max':float(np.max(a))}
    def main():
     out={'schema':'l0c2r_geometry_summary_v1','status':'COMPLETE_READ_ONLY','seeds':{}}
     for s in SEEDS:
      out['seeds'][str(s)]={}
      for u in UPDATES:
       z=np.load(ROOT/f'artifacts/l0c2r_seed{s}/update_{u:03d}.npz'); cmdout={}
       for ci,cmd in enumerate(z['commands']):
        sat=z['saturation'][ci]; norm=z['action_norm'][ci]; tilt=np.maximum(np.abs(z['roll'][ci]),np.abs(z['pitch'][ci])); done=z['done'][ci].astype(bool); fail=done.any(axis=0); max_sat=sat.max(axis=0); max_tilt=tilt.max(axis=0)
        cmdout[str(float(cmd[0]))]={'action_norm':q(norm.reshape(-1)),'saturation':q(sat.reshape(-1)),'max_saturation_per_lane':q(max_sat),'max_tilt_deg_per_lane':q(np.degrees(max_tilt)),'done_rate':float(fail.mean()),'failure_saturation_means':{'failed':float(max_sat[fail].mean()) if fail.any() else None,'survived':float(max_sat[~fail].mean()) if (~fail).any() else None},'joint_saturation_mean':sat.mean(axis=(0,1)).tolist()}
       out['seeds'][str(s)][str(u)]=cmdout
     lines=['# L0-C2R lane geometry summary','', 'Status: **COMPLETE — read-only checkpoint replay**','', 'Quantiles are computed from raw lane-indexed arrays; `done_rate` is first-fall/base-contact occurrence over the 500-step replay.','']
     for s in SEEDS:
      lines.append(f'## seed {s}')
      for u in UPDATES:
       for cmd,m in out['seeds'][str(s)][str(u)].items(): lines.append(f"- update {u}, vx={cmd}: action p95={m['action_norm']['p95']:.3f}, saturation p95={m['saturation']['p95']:.3f}, lane max-sat p95={m['max_saturation_per_lane']['p95']:.3f}, max-tilt p95={m['max_tilt_deg_per_lane']['p95']:.2f}°, done={m['done_rate']:.3f}, failed/survived max-sat={m['failure_saturation_means']['failed']}/{m['failure_saturation_means']['survived']}")
     d=ROOT/'artifacts/l0c2r_analysis';d.mkdir(parents=True,exist_ok=True);(d/'L0C2R_SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');(d/'report.md').write_text('\n'.join(lines)+'\n');print(d/'L0C2R_SUMMARY.json')
    if True:main()

def run_l0c2r_replay():
    """Run former l0c2r_replay.py stage."""
    """Read-only lane-indexed replay of existing L0-B/C1 checkpoints."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    import argparse, hashlib, json, sys, time
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    RESET=ROOT/'artifacts/gate0b_reset_states.npz'; UPDATES=(200,300,400,500); COMMANDS=((.25,0.,0.),(.5,0.,0.),(.75,0.,0.)); LANES=64; HORIZON=500
    def infer_hidden(state):
     out=[]; i=0
     while f'actor_body.{i}.weight' in state: out.append(int(state[f'actor_body.{i}.weight'].shape[0])); i+=2
     return out
    def main():
     p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--only-update',type=int);a=p.parse_args(); out=a.run_dir.resolve(); out.mkdir(parents=True,exist_ok=True); updates=(a.only_update,) if a.only_update else UPDATES
     rng=(torch.random.get_rng_state(),np.random.get_state()); app=base=None; manifest={}
     try:
      from isaaclab.app import AppLauncher;app=AppLauncher({'headless':True,'enable_cameras':False}).app
      import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
      from rl.experiments.common.utilities.train_b0 import nominalize
      from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
      from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
      from rl.core.modules.actor_critic import ActorCritic
      from rl.experiments.common.utilities.b0_monitor_isolation_smoke import _install_states
      cfg=IsaacLabTalonEnvCfg();cfg.scene.num_envs=LANES;cfg.seed=17;cfg.sim.dt=.01;cfg.decimation=1;nominalize(cfg);base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped;env=B0TalonEnv(base)
      with np.load(RESET) as z: saved={k:z[k].copy() for k in z.files}
      for update in updates:
       ck=ROOT/f'runs/l0b_seed{a.seed}_2026-09-21/checkpoints/update_{update:03d}.pt' if a.seed<2 else ROOT/f'runs/l0b_seed2_2026-09-22/checkpoints/update_{update:03d}.pt'
       state=torch.load(ck,map_location=base.device);raw=state.get('model',state);model=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,infer_hidden(raw)).to(base.device);model.load_state_dict(raw);model.eval(); model.set_learned_std()
       arrays={k:[] for k in ('action','mean','action_norm','saturation','roll','pitch','height','vx','contact','done')}
       for cmd in COMMANDS:
        env.command=cmd;base.reset();env._command();_install_states(base,saved,LANES);tr=env._scalar_transition(base._transition(base.observation_manager.compute()),np.zeros(LANES,bool));obs=tr['obs']; rows={k:[] for k in arrays}
        for t in range(HORIZON):
         x=torch.as_tensor(obs,device=base.device,dtype=torch.float32)
         with torch.no_grad(): mean=model.raw_mean(x);act=torch.tanh(mean)*model.ACTION_CLIP
         arr=act.cpu().numpy().astype(np.float32); rows['action'].append(arr);rows['mean'].append(mean.cpu().numpy());rows['action_norm'].append(np.linalg.norm(arr,axis=1));rows['saturation'].append((np.abs(arr)>=2.99).mean(axis=1)); tr=env.step(arr);f=tr['fields'];rows['roll'].append(f['roll_pitch'][:,0]);rows['pitch'].append(f['roll_pitch'][:,1]);rows['height'].append(f['height']);rows['vx'].append(f['v_actual'][:,0]);rows['contact'].append(tr['term_base_contact']);rows['done'].append(tr['done']);obs=tr['obs']
        for k in rows: arrays[k].append(np.stack(rows[k],axis=0))
       path=out/f'update_{update:03d}.npz';np.savez_compressed(path,commands=np.asarray(COMMANDS,np.float32),env_id=np.arange(LANES),**{k:np.stack(v,axis=0) for k,v in arrays.items()});manifest[str(update)]={'checkpoint':str(ck),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':{k:list(v.shape) for k,v in arrays.items()}}
      (out/'manifest.json').write_text(json.dumps({'schema':'l0c2r_replay_v1','seed':a.seed,'frozen_reset_sha256':hashlib.sha256(RESET.read_bytes()).hexdigest(),'commands':COMMANDS,'horizon':HORIZON,'lane_mapping':'env_id 0..63 fixed','training_state_mutated':False,'checkpoints':manifest},indent=2,sort_keys=True)+'\n')
     finally:
      torch.random.set_rng_state(rng[0]);np.random.set_state(rng[1]);
      if base is not None:base.close()
      if app is not None:app.close()
    if True:main()

STAGES = {
    "l0c2_audit": run_l0c2_audit,
    "l0c2j_audit": run_l0c2j_audit,
    "l0c2r_analyze": run_l0c2r_analyze,
    "l0c2r_replay": run_l0c2r_replay,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
