"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_b1_s1_controlled_smoke():
    """Run former b1_s1_controlled_smoke.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,time,traceback
    import torch
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/'artifacts'/'b1_s1_smoke'; OUT.mkdir(parents=True,exist_ok=True)
    def main():
        (OUT/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','unix':time.time()},indent=2)+'\n'); app=base=None
        try:
            from isaaclab.app import AppLauncher
            app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.experiments.common.utilities.train_b0 import nominalize
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.scalar_ppo import B0PPOConfig,B0PPOTrainer
            cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=4; cfg.seed=3; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg); cfg.episode_length_s=.03
            base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base); trn=env.reset(); obs=trn['obs']; m=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,[64,64]).to(base.device); cfgp=B0PPOConfig(b1_p2_enabled=True,b1_s1_enabled=True,actor_epochs=2,kl_low=-100,kl_high=100); trainer=B0PPOTrainer(m,cfgp,lr=3e-4); trainer.begin_update(); std=trainer._std_for_update; os=[]; ds=[]; acts=[]; lps=[]
            for _ in range(8):
                x=torch.as_tensor(obs,device=base.device,dtype=torch.float32); a,lp=m.act(x); q=env.step(a.detach().cpu().numpy().astype('float32')); os.append(x); acts.append(a.detach()); lps.append(lp.detach()); ds.append(torch.as_tensor(q['done'],device=base.device)); obs=q['obs']
            O=torch.stack(os); D=torch.stack(ds); flat=O.reshape(-1,base.obs_dim); A=torch.stack(acts).reshape(-1,base.action_dim); LP=torch.stack(lps).reshape(-1); before={k:v.detach().clone() for k,v in m.state_dict().items()}; out=trainer.optimize_batch(flat,A,LP,torch.ones(flat.shape[0],device=base.device),torch.zeros(flat.shape[0],device=base.device),rollout_obs=O,rollout_dones=D); trainer.finish_update(); after=m.state_dict(); changed_actor=any(not torch.equal(before[k],after[k]) for k in before if not k.startswith('critic_')); changed_critic=any(not torch.equal(before[k],after[k]) for k in before if k.startswith('critic_')); valid=int((~D[:-1]).sum()); total=int(D[:-1].numel()); checks={'done':bool(D.any()),'masked':valid<total,'finite':out['spatial_loss']>=0 and out['temporal_loss']>=0,'actor_changed':changed_actor,'critic_changed':changed_critic,'critic_lr':trainer.critic_optim.param_groups[0]['lr']==3e-4,'std_finished':trainer._std_for_update is None}; assert all(checks.values()),checks
            p=OUT/'checkpoint.pt'; trainer.save(p); restored=B0PPOTrainer(ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,[64,64]).to(base.device),cfgp); restored.load(p); assert restored.cfg.b1_s1_enabled and restored.cfg.b1_p2_enabled and restored.actor_optim.param_groups[0]['lr']==trainer.actor_optim.param_groups[0]['lr']
            result={'pass':True,'done_count':int(D.sum()),'valid_pairs':valid,'total_pairs':total,'actor_epochs':out['actor_epochs_completed'],'actor_stopped':out['actor_stopped'],'spatial_loss':out['spatial_loss'],'temporal_loss':out['temporal_loss'],'actor_grad_norm':out['actor_grad_norm'],'spatial_grad_norm':out['spatial_grad_norm'],'temporal_grad_norm':out['temporal_grad_norm'],'critic_lr':trainer.critic_optim.param_groups[0]['lr'],'scheduled_std':std,'p1_enabled':cfgp.b1_p1_enabled,'resume_actor_lr':restored.actor_optim.param_groups[0]['lr'],'command_mask':[42,43,44]}
            (OUT/'artifact.json').write_text(json.dumps(result,indent=2)+'\n'); (OUT/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0,'unix':time.time(),'artifact':'artifact.json'},indent=2)+'\n'); print(OUT/'artifact.json')
        except BaseException as e:
            (OUT/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    if True: main()

def run_b1_s1_real_rollout_scale_audit():
    """Run former b1_s1_real_rollout_scale_audit.py stage."""
    """The S1 regularizer scales again, this time on a real Isaac rollout.
    
    No optimizer step is taken. The synthetic scale audit fixes the rollout; this
    one draws it from the trained seed-0 policy so the losses and gradient norms
    are measured against states the policy actually reaches. The temporal term
    only counts consecutive pairs that did not cross a termination.
    """
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.isaac_audit import ARTIFACTS, RUNS, IsaacAudit
    
    CHECKPOINT = "b1_p2_seed0_2026-09-21/checkpoints/update_500.pt"
    NUM_ENVS, STEPS = 16, 16
    COMMAND_SLICE = slice(42, 45)
    SPATIAL_COEF, TEMPORAL_COEF = .1, .05
    NOISE_SCALE, NOISE_CLIP = .02, .05
    
    
    def grad_norm(loss, params):
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((x.detach() ** 2).sum() for x in g if x is not None)))
    
    
    class B1S1RealRolloutScaleAudit(IsaacAudit):
        """S1 regularizer scale on a real rollout, with no update applied."""
    
        root = ARTIFACTS
        run = "b1_s1_scale_audit"
        report = "real_rollout_summary.json"
        num_envs = NUM_ENVS
    
        def build_env(self):
            """The Talon A1 task under the B0 wrapper, nominalised, not the flat
            velocity task the other Isaac audits use."""
            import gymnasium as gym
            import talon_rl.tasks.locomotion.a1_env  # noqa: F401  (registers the task)
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
    
            from rl.experiments.common.utilities.train_b0 import nominalize
    
            cfg = IsaacLabTalonEnvCfg()
            cfg.scene.num_envs = self.num_envs
            cfg.seed = 0
            cfg.sim.dt = .01
            cfg.decimation = 1
            nominalize(cfg)
            self.base = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
            env = B0TalonEnv(self.base)
            return env, env.reset()["obs"]
    
        def rollout(self, env, obs):
            from rl.core.modules.actor_critic import ActorCritic
    
            base = self.base
            m = ActorCritic(base.obs_dim, base.obs_dim, base.action_dim, 1, [64, 64]).to(base.device).eval()
            state = torch.load(RUNS / CHECKPOINT, map_location=base.device)
            m.load_state_dict(state["model"])
            m.set_scheduled_fixed_std(float(state["scheduled_std"]))
    
            obs_seq, act_seq, logp_seq, rew, dones = [], [], [], [], []
            for _ in range(STEPS):
                x = torch.as_tensor(obs, device=base.device, dtype=torch.float32)
                action, logp = m.act(x)
                tr = env.step(action.detach().cpu().numpy().astype(np.float32))
                obs_seq.append(x)
                act_seq.append(action.detach())
                logp_seq.append(logp.detach())
                rew.append(torch.as_tensor(tr["reward"], device=base.device))
                dones.append(torch.as_tensor(tr["done"], device=base.device))
                obs = tr["obs"]
    
            o = torch.stack(obs_seq)
            a = torch.stack(act_seq)
            lp = torch.stack(logp_seq)
            d = torch.stack(dones)
            flat = o.reshape(-1, base.obs_dim)
            actions = a.reshape(-1, base.action_dim)
            old = lp.reshape(-1)
            adv = torch.ones_like(old)
    
            mu = m.raw_mean(flat).reshape(STEPS, NUM_ENVS, -1)
            mask = torch.ones(base.obs_dim, device=base.device)
            mask[COMMAND_SLICE] = 0
            noise = torch.randn_like(o) * NOISE_SCALE
            noise.clamp_(-NOISE_CLIP, NOISE_CLIP)
            noise *= mask
            spatial = ((m.raw_mean((o + noise).reshape(-1, base.obs_dim)).reshape_as(mu) - mu) ** 2).mean()
            # only pairs that did not cross a termination are real transitions
            valid = (~d[:-1]).float()
            temporal = ((mu[1:] - mu[:-1]) ** 2).mean(dim=-1)
            temporal_loss = (temporal * valid).sum() / (valid.sum() * mu.shape[-1] + 1e-8)
    
            ppo = -(torch.exp(m.logp(flat, actions) - old) * adv).mean()
            actor = [p for n, p in m.named_parameters()
                     if n != "log_std" and not n.startswith("critic_")]
            weighted_s = SPATIAL_COEF * spatial
            weighted_t = TEMPORAL_COEF * temporal_loss
    
            result = {"rollout_shape": list(o.shape),
                      "ppo_actor_loss": float(ppo.detach()),
                      "spatial_loss": float(spatial.detach()),
                      "weighted_spatial": float(weighted_s.detach()),
                      "temporal_loss": float(temporal_loss.detach()),
                      "weighted_temporal": float(weighted_t.detach()),
                      "grad_norm_ppo": grad_norm(ppo, actor),
                      "grad_norm_spatial": grad_norm(weighted_s, actor),
                      "grad_norm_temporal": grad_norm(weighted_t, actor),
                      "valid_temporal_pairs": float(valid.sum()),
                      "temporal_pair_fraction": float(valid.mean()),
                      "mu_delta_sq_mean": float(temporal.mean()),
                      "near_fall_pairs": int(d[:-1].sum().item()),
                      "scheduled_std": float(state["scheduled_std"]),
                      "checkpoint": "seed0/update500", "no_optimizer_step": True}
            self.write(result)
            self.write({"status": "RUN_DONE", "exit_code": 0, "unix": time.time()},
                       "real_RUN_DONE.json")
            print(self.out / self.report)
            return result
    
        def execute(self):
            self.write({"status": "RUN_STARTED", "mode": "real_rollout_no_update",
                        "envs": NUM_ENVS, "steps": STEPS}, "real_RUN_STARTED.json")
            self.base = None
            try:
                return super().execute()
            except BaseException as e:
                self.write({"status": "ERROR", "error": str(e),
                            "traceback": traceback.format_exc()}, "real_ERROR.json")
                raise
            finally:
                if self.base is not None:
                    self.base.close()
    
    
    if True:
        B1S1RealRolloutScaleAudit.main()

def run_b1_s1_scale_audit():
    """Run former b1_s1_scale_audit.py stage."""
    """How large would the proposed S1 regularizers be against the PPO actor loss?
    
    Read-only: it builds each candidate loss on one synthetic rollout and reports
    its value and gradient norm beside PPO's, so the weights can be judged on
    scale. It trains nothing and tunes nothing. Command dimensions are held fixed
    in the spatial perturbation, since a regularizer must not penalise responding
    to the command.
    """
    import json
    import sys
    from pathlib import Path
    
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import ARTIFACTS, OfflineAudit
    
    SEED = 21
    OBS_DIM, ACT_DIM = 51, 12
    LANES, STEPS = 8, 4
    COMMAND_SLICE = slice(42, 45)
    SPATIAL_COEF, TEMPORAL_COEF = .1, .05
    NOISE_SCALE, NOISE_CLIP = .02, .05
    
    
    def grad_norm(loss, params):
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((x.detach() ** 2).sum() for x in g if x is not None)))
    
    
    class B1S1ScaleAudit(OfflineAudit):
        """Scale of the proposed S1 regularizers against the PPO actor loss."""
    
        root = ARTIFACTS
        run = "b1_s1_scale_audit"
        report = "summary.json"
    
        def analyze(self):
            from rl.core.modules.actor_critic import ActorCritic
    
            torch.manual_seed(SEED)
            m = ActorCritic(OBS_DIM, OBS_DIM, ACT_DIM, 1, [64, 64])
            obs = torch.randn(LANES, STEPS, OBS_DIM)
            obs[:, :, COMMAND_SLICE] = torch.tensor([.5, 0., 0.])
            flat = obs.reshape(-1, OBS_DIM)
    
            act, old = m.act(flat)
            adv = torch.randn(flat.shape[0])
            # the original drew returns here too and never used them; the draw has
            # to stay, or every later randn comes off a different RNG state
            torch.randn(flat.shape[0])
            ratio = torch.exp(m.logp(flat, act) - old)
            ppo = -(ratio * adv).mean()
    
            mu = m.raw_mean(flat).reshape(LANES, STEPS, ACT_DIM)
            mask = torch.ones(OBS_DIM)
            mask[COMMAND_SLICE] = 0
            noise = torch.randn_like(obs) * NOISE_SCALE
            noise.clamp_(-NOISE_CLIP, NOISE_CLIP)
            noise *= mask
            spatial = ((m.raw_mean((obs + noise).reshape(-1, OBS_DIM)).reshape(LANES, STEPS, ACT_DIM)
                        - mu) ** 2).mean()
            temporal = ((mu[:, 1:] - mu[:, :-1]) ** 2).mean()
    
            actor = [p for n, p in m.named_parameters()
                     if n != "log_std" and not n.startswith("critic_")]
            return {"rollout_shape": list(obs.shape),
                    "ppo_actor_loss": float(ppo.detach()),
                    "spatial_loss": float(spatial.detach()),
                    "weighted_spatial": float(SPATIAL_COEF * spatial.detach()),
                    "temporal_loss": float(temporal.detach()),
                    "weighted_temporal": float(TEMPORAL_COEF * temporal.detach()),
                    "grad_norm_ppo": grad_norm(ppo, actor),
                    "grad_norm_spatial": grad_norm(SPATIAL_COEF * spatial, actor),
                    "grad_norm_temporal": grad_norm(TEMPORAL_COEF * temporal, actor),
                    "command_mask_excluded_indices": list(range(COMMAND_SLICE.start,
                                                                COMMAND_SLICE.stop)),
                    "read_only": True}
    
        def summarize(self, report):
            print(self.out / self.report)
    
    
    if True:
        B1S1ScaleAudit.main()

def run_b1_s1_verdict():
    """Run former b1_s1_verdict.py stage."""
    from pathlib import Path
    import json,math
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_s1_verdict'; OUT.mkdir(parents=True,exist_ok=True)
    RUNS={0:ROOT/'runs/b1_s1_seed0_retry1_2026-09-21',1:ROOT/'runs/b1_s1_seed1_2026-09-21',2:ROOT/'runs/b1_s1_seed2_2026-09-21'}
    P2={0:ROOT/'runs/b1_p2_seed0_2026-09-21',1:ROOT/'runs/b1_p2_seed1_2026-09-21',2:ROOT/'runs/b1_p2_seed2_2026-09-21'}
    def main():
     rows=[]; fails=[]
     for s,r in RUNS.items():
      done=json.loads((r/'RUN_DONE.json').read_text()); cfg=json.loads((r/'config.json').read_text()); tm=json.loads((r/'training_metrics.json').read_text())['metrics']; mon=json.loads((r/'monitor/u500.json').read_text())['acceptance']; events=[e for x in tm for e in x.get('lr_events',[])]
      finite=all(math.isfinite(float(x[k])) for x in tm for k in ('std','reward_mean','policy_loss','value_loss','analytic_kl','approx_kl','actor_lr','spatial_loss','temporal_loss','actor_grad_norm','spatial_grad_norm','temporal_grad_norm'))
      rec={'seed':s,'run_done':done.get('status')=='RUN_DONE' and done.get('exit_code')==0,'checkpoint_500':(r/'checkpoints/update_500.pt').exists(),'s1_config':cfg.get('b1_s1') is True,'metrics':len(tm),'finite':finite,'critic_completed':all(x['critic_completed'] for x in tm),'p2_lr_events':len(events),'std':[tm[0]['std'],tm[-1]['std']],'s1_spatial_mean':sum(x['spatial_loss'] for x in tm)/len(tm),'s1_temporal_mean':sum(x['temporal_loss'] for x in tm)/len(tm),'gate':{'survival':mon['survival_rate'],'vx_mae':mon['mean_abs_vx_error'],'tilt_p95':math.degrees(mon['tilt_p95_rad']),'tilt_max':math.degrees(mon['tilt_max_rad'])}}
      rec['gate']['pass']=rec['gate']['survival']>=.9 and rec['gate']['vx_mae']<=.15 and rec['gate']['tilt_p95']<=15 and rec['gate']['tilt_max']<=30
      pm=json.loads((P2[s]/'monitor/u500.json').read_text())['acceptance']; rec['vs_p2']={'survival_delta':rec['gate']['survival']-pm['survival_rate'],'vx_mae_delta':rec['gate']['vx_mae']-pm['mean_abs_vx_error'],'tilt_p95_delta_deg':rec['gate']['tilt_p95']-math.degrees(pm['tilt_p95_rad']),'tilt_max_delta_deg':rec['gate']['tilt_max']-math.degrees(pm['tilt_max_rad'])}
      if not all((rec['run_done'],rec['checkpoint_500'],rec['s1_config'],rec['finite'],rec['critic_completed'])): fails.append(s)
      rows.append(rec)
     verdict=not fails and all(x['gate']['pass'] for x in rows); out={'schema':'b1_s1_verdict_v1','status':'PASS' if verdict else 'FAIL','sanity_failures':fails,'seeds':rows,'rule':'update-500 deterministic gate'}; (OUT/'B1_S1_VERDICT.json').write_text(json.dumps(out,indent=2)+'\n')
     lines=[f"# B1-S1 Verdict: {'PASS' if verdict else 'FAIL'}",'', '|seed|survival|vx MAE|tilt p95°|max tilt°|Δp95 vs P2|spatial mean|temporal mean|gate|','|---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
     for x in rows: g=x['gate']; lines.append(f"|{x['seed']}|{g['survival']:.3f}|{g['vx_mae']:.3f}|{g['tilt_p95']:.2f}|{g['tilt_max']:.2f}|{x['vs_p2']['tilt_p95_delta_deg']:.2f}|{x['s1_spatial_mean']:.4g}|{x['s1_temporal_mean']:.4g}|{g['pass']}|")
     (OUT/'report.md').write_text('\n'.join(lines)+'\n'); print(OUT/'report.md')
    if True: main()

def run_finalize_b1_s1():
    """Run former finalize_b1_s1.py stage."""
    from pathlib import Path
    import hashlib,json,time
    ROOT=Path(__file__).resolve().parents[4]; D=ROOT/'artifacts'/'b1_s1_freeze'
    def main():
     m=D/'B1_S1_FREEZE.json'; data=json.loads(m.read_text()); a=json.loads((ROOT/'artifacts/b1_s1_smoke/artifact.json').read_text()); done=json.loads((ROOT/'artifacts/b1_s1_smoke/RUN_DONE.json').read_text())
     if not a.get('pass') or done.get('exit_code')!=0: raise RuntimeError('S1 smoke failed')
     data.update(status='FROZEN',training_authorized=True,frozen_unix=time.time(),smoke_artifact_sha256=hashlib.sha256((ROOT/'artifacts/b1_s1_smoke/artifact.json').read_bytes()).hexdigest())
     m.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n'); print(m)
    if True: main()

def run_freeze_b1_s1():
    """Run former freeze_b1_s1.py stage."""
    from pathlib import Path
    import hashlib,json,time
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_s1_freeze'; OUT.mkdir(parents=True,exist_ok=True)
    def main():
     files={'trainer':'scripts/rl/core/algorithms/scalar_ppo.py','runner':'scripts/rl/experiments/common/utilities/train_b0.py','design':'docs/baselines/stability_robustness/b1-s1-acaps-stability-margin-design-draft.md','smoke':'scripts/rl/experiments/baselines/stability_robustness/reward_scaling.py','monitor':'scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py','reset_states':'artifacts/b0_smoke/b0-monitor-frozen-reset-states.npz'}
     sha={k:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in files.items()}
     data={'schema':'b1_s1_freeze_v1','status':'DESIGN_DRAFT','training_authorized':False,'created_unix':time.time(),'mechanism':'b1_p2_plus_acaps_inspired_stability_margin','p2_inherited':True,'desired_kl':0.01,'kl_low':0.005,'kl_high':0.02,'lr_factor':1.5,'actor_lr_bounds':[1e-5,1e-2],'p1_early_stop':False,'lambda_spatial':0.10,'lambda_temporal':0.05,'perturb_std':0.02,'perturb_clip':0.05,'command_mask':[42,43,44],'temporal_storage':'time_major_before_minibatch_permutation','done_mask':'cross_reset_pairs_excluded','regularizer_scope':'actor_only','num_envs':4096,'updates':500,'seeds':[0,1,2],'command_vx':0.5,'scheduled_std':{'initial':0.82,'final':0.10,'hold_updates':100,'decay_updates':300},'monitor_contract':'tanh(actor_mean), 64 frozen states x 500 steps, evaluation_only','sha256':sha}
     (OUT/'B1_S1_FREEZE.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n'); print(OUT/'B1_S1_FREEZE.json')
    if True: main()

STAGES = {
    "b1_s1_controlled_smoke": run_b1_s1_controlled_smoke,
    "b1_s1_real_rollout_scale_audit": run_b1_s1_real_rollout_scale_audit,
    "b1_s1_scale_audit": run_b1_s1_scale_audit,
    "b1_s1_verdict": run_b1_s1_verdict,
    "finalize_b1_s1": run_finalize_b1_s1,
    "freeze_b1_s1": run_freeze_b1_s1,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
