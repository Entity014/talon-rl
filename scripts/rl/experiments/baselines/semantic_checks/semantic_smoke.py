#!/usr/bin/env python3
"""Small Isaac smoke for G1 command/observation boundary semantics."""
from __future__ import annotations
import hashlib, json, os
import argparse, time
from pathlib import Path
import numpy as np
import torch

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts/g1_smoke'
OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(); parser.add_argument('--num-envs', type=int, default=4)
args = parser.parse_args()
run = OUT / f'RUN_STARTED_n{args.num_envs}.json'
run.write_text(json.dumps(dict(status='RUN_STARTED', pid=os.getpid(), unix=time.time(), num_envs=args.num_envs))+'\n')
def mark(name, **extra):
    (OUT / f'{name}_n{args.num_envs}.json').write_text(json.dumps(dict(status=name, unix=time.time(), **extra), indent=2)+'\n')
def check(condition, label, **context):
    if not condition:
        mark('ASSERT_FAIL', label=label, **context)
        raise AssertionError(label)
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
from isaaclab.app import AppLauncher
app = AppLauncher({'headless': True, 'enable_cameras': False}).app
mark('APP_OK')
import gymnasium as gym
import talon_rl.tasks.locomotion.a1_env  # noqa
from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

cfg = IsaacLabTalonEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = 20260921
cfg.sim.dt = .01
cfg.decimation = 1
cfg.sim.render_interval = 1
cfg.g1_command_exposure = True
cfg.scene.terrain.terrain_type = 'plane'
cfg.scene.terrain.terrain_generator = None
cfg.curriculum.terrain_levels = None
cfg.events.push_robot = None
env = gym.make('Isaac-Talon-A1-v0', cfg=cfg, render_mode=None).unwrapped
mark('INIT_OK')
transition = env.reset()
mark('RESET_OK')
n = env.num_envs
check(env._g1_schedule is not None, 'schedule_missing')
records=[]; episode=np.zeros(n,dtype=int); step=np.zeros(n,dtype=int)
initial_zero = env._g1_schedule.zero_hold.copy()
for _ in range(230):
    loop_step = len(records)
    if loop_step >= 199:
        mark('PRE_SWITCH_STEP', step=loop_step)
    applied = env.v_command_buf.detach().cpu().numpy().copy()
    scheduler = env._g1_schedule.command.copy()
    zero_before = env._g1_schedule.zero_hold.copy()
    check(np.allclose(applied, scheduler), 'applied_scheduler_mismatch', step=loop_step,
          applied=applied.tolist(), scheduler=scheduler.tolist())
    if loop_step >= 199: mark('COMMAND_WRITTEN', step=loop_step,
        shape=list(applied.shape), dtype=str(applied.dtype), finite=bool(np.isfinite(applied).all()))
    action = np.zeros((n, env.action_dim), dtype=np.float32)
    check(np.isfinite(action).all(), 'action_nonfinite', step=loop_step)
    if loop_step >= 199: mark('ACTOR_INFERENCE_DONE', step=loop_step,
        action_shape=list(action.shape), action_finite=bool(np.isfinite(action).all()))
    if loop_step >= 199: mark('ENV_STEP_ENTER', step=loop_step)
    transition, done = env.step(action)
    if loop_step >= 199: mark('ENV_STEP_RETURN', step=loop_step,
        done=done.tolist())
    observed = np.asarray(transition['v_command']).copy()
    event = getattr(env, '_g1_last_event', None)
    switched = event.progress_reset.copy() if event is not None else np.zeros(n,bool)
    dp_reset = getattr(env, '_g1_last_dp_reset', np.zeros(n, bool)).copy()
    check(np.isfinite(observed).all(), 'observation_nonfinite', step=loop_step)
    check(observed.shape == (n,3), 'observation_shape', step=loop_step, shape=list(observed.shape))
    if loop_step >= 199: mark('OBS_BUILT', step=loop_step,
        shape=list(observed.shape), finite=bool(np.isfinite(observed).all()))
    # transition observation is the command consumed by the next policy call.
    check(np.allclose(observed, env.v_command_buf.detach().cpu().numpy()),
          'observed_next_command_mismatch', step=loop_step,
          observed=observed.tolist(), next_command=env.v_command_buf.detach().cpu().numpy().tolist())
    records.append(dict(env_id=list(range(n)), episode_id=episode.tolist(),
        policy_step_in_episode=step.tolist(), zero_hold_flag=initial_zero.tolist(),
        hold_counter=env._g1_schedule.hold_step.tolist(),
        scheduler_command=scheduler.tolist(), observation_command=observed.tolist(),
        applied_command=applied.tolist(), command_switch_event=switched.tolist(),
        directed_progress_reset_event=dp_reset.tolist(), done=done.tolist()))
    check(np.array_equal(switched, dp_reset), 'switch_dp_reset_mismatch', step=loop_step,
          switched=switched.tolist(), dp_reset=dp_reset.tolist(), done=done.tolist(),
          hold=env._g1_schedule.hold_step.tolist())
    current_zero = zero_before
    for i in range(n):
        if current_zero[i] and step[i] < 200:
            check(np.allclose(scheduler[i], 0.0), 'zero_hold_nonzero_command', step=loop_step, lane=i,
                  command=scheduler[i].tolist(), hold=int(env._g1_schedule.hold_step[i]))
        if current_zero[i] and step[i] == 200:
            check(switched[i], 'missing_switch_at_200', step=loop_step, lane=i)
    for i in np.flatnonzero(done):
        episode[i] += 1; step[i] = 0
        # The post-step scheduler has already prepared the new episode's next
        # observation, so a fresh zero-hold lane may show counter 1 here.
        check(env._g1_schedule.hold_step[i] <= 1, 'stale_hold_after_done', step=loop_step, lane=i,
              hold=int(env._g1_schedule.hold_step[i]), done=done.tolist())
    step[~done] += 1
    if loop_step >= 199: mark('POST_STEP_METRICS_DONE', step=loop_step)
    if len(records) == 50: mark('STEP_50_OK', rows=len(records)*n)
    if len(records) == 200: mark('STEP_200_OK', rows=len(records)*n)
payload=dict(environment='plane', num_envs=n, steps=len(records), rows=len(records)*n,
             zero_hold_lanes=int(initial_zero.sum()), records=records,
             design_sha256=sha(ROOT/'docs/baselines/multiobjective_bridge/g1-command-exposure-design.md'),
             scheduler_sha256=sha(ROOT/'talon_rl/g1_command_schedule.py'),
             assertions=['finite','shape','scheduler==applied','observed==next_applied',
                         'termination_reset'], status='PASS')
(OUT/'g1-semantic-smoke.json').write_text(json.dumps(payload,indent=2)+'\n')
(OUT/'g1-semantic-smoke-config.json').write_text(json.dumps({k:v for k,v in vars(cfg).items() if not k.startswith('_')},default=str,indent=2)+'\n')
mark('ARTIFACT_WRITTEN', rows=len(records)*n)
env.close(); app.close()
print(json.dumps({k:payload[k] for k in ('num_envs','steps','rows','zero_hold_lanes','status')}))
