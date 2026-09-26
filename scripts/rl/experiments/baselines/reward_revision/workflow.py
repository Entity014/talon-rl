"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_r1_directed_progress_smoke():
    """Run former r1_directed_progress_smoke.py stage."""
    """Isaac Sim smoke check for a1_env.py's directed-progress wiring (R1,
    2026-09-20, artifacts/r1_freeze/FREEZE.md) -- the pure math is unit-tested
    in tests/test_directed_progress.py, but the plumbing (root_pos_w/yaw
    extraction, reset-hook ordering, terminal-frame double-call) only runs
    through real Isaac Lab, which test_a1_env.py can't do on this machine. Not
    a pytest -- prints per-step directed_progress so a human/this session can
    eyeball "no spike at reset or command change" before committing to a
    3-seed retrain.
    """
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    
    import argparse
    import os
    from pathlib import Path
    
    import numpy as np
    import torch
    import yaml
    
    from talon_rl.config import ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg
    from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
    from rl.experiments.common.utilities.final_locomotion_eval import _nominalize_cfg, _select
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("checkpoint", type=Path)
        parser.add_argument("--num-envs", type=int, default=4)
        parser.add_argument("--steps", type=int, default=150)
        args = parser.parse_args()
    
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
        simulation_app = app_launcher.app
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
    
        stored = yaml.safe_load((args.checkpoint.resolve().parents[1] / "config.yaml").read_text())
        obs_cfg = ObservationSpaceCfg(**_select(ObservationSpaceCfg, stored["obs"]))
        action_cfg = ActionSpaceCfg(**_select(ActionSpaceCfg, stored["action"]))
        reward_cfg = RewardVectorCfg(**_select(RewardVectorCfg, stored["reward"]))
        pref_cfg = PreferenceCfg(**_select(PreferenceCfg, stored["preference"]))
        stack_cfg = ObservationStackCfg(**_select(ObservationStackCfg, stored["stack"]))
        moppo_cfg = MOPPOConfig(**_select(MOPPOConfig, stored["moppo"]))
        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = args.num_envs; cfg.seed = 1001
        cfg.sim.dt = 0.01; cfg.decimation = 1; cfg.sim.render_interval = 1
        _nominalize_cfg(cfg)
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
        trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg,
                               stack_cfg=stack_cfg, extrinsics_cfg=ExtrinsicsCfg(), seed=1001)
        trainer.load(str(args.checkpoint)); trainer.model.eval()
        trainer.w = np.repeat(np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32), env.num_envs, axis=0)
    
        transition = env.reset()
        trainer.push_obs(transition["obs"], update_normalizer=False)
        trainer._last_extrinsics = transition.get("extrinsics")
    
        prev_dp = None
        max_step_delta = 0.0
        switch_step = args.steps // 2
        for step in range(args.steps):
            command_x = 0.5 if step < switch_step else -0.5  # forced command switch mid-run
            env.v_command_buf[:] = torch.tensor((command_x, 0.0, 0.0), device=env.device)
            action = trainer.act_inference()
            transition, done = env.step(action)
            trainer._last_extrinsics = transition.get("extrinsics")
            trainer.push_obs(transition["obs"], done_mask=done, update_normalizer=False)
            dp = transition["directed_progress"]
            if prev_dp is not None and not done.any() and step != switch_step:
                max_step_delta = max(max_step_delta, float(np.max(np.abs(dp - prev_dp))))
            flag = " <- COMMAND SWITCH" if step == switch_step else (" <- lane(s) reset" if done.any() else "")
            print(f"step={step:3d} cmd_x={command_x:+.2f} directed_progress={np.round(dp, 3)}{flag}", flush=True)
            prev_dp = dp
    
        print(f"\nmax per-step |delta| outside reset/switch steps: {max_step_delta:.4f}")
        env.close(); simulation_app.close()
    
    
    if True:
        main()

def run_r1_mismatch_audit():
    """Run former r1_mismatch_audit.py stage."""
    """Frozen-checkpoint diagnostic. Never updates policy, normalizers or rewards.
    
    600-step right-censored first-episode probe, 64 lanes, uniform preference.
    Run nominal and training environments separately so scene creation is clean.
    """
    
    import argparse
    import copy
    import json
    import os
    from pathlib import Path
    
    import numpy as np
    import torch
    import yaml
    
    from rl.final_locomotion_eval import (
        _select, _sha256, _nominalize_cfg, _set_initial_state, _terminal_frame,
        ObservationSpaceCfg, RewardVectorCfg, PreferenceCfg, ObservationStackCfg,
        ExtrinsicsCfg, MOPPOConfig, MOPPOTrainer,
    )
    
    
    def main():
        p = argparse.ArgumentParser(__doc__)
        p.add_argument('--seed', type=int, required=True)
        p.add_argument('--environment', choices=['nominal', 'training', 'terrain_only', 'physics_only'], required=True)
        p.add_argument('--output', type=Path, default=Path('artifacts/r1_mismatch_audit'))
        p.add_argument('--num-envs', type=int, default=64)
        p.add_argument('--init-only', action='store_true')
        p.add_argument('--condition', choices=['all', 'forced_mean'], default='all')
        p.add_argument('--g1', action='store_true', help='enable integrated G1 scheduler for smoke only')
        args = p.parse_args()
        args.output.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
        from isaaclab.app import AppLauncher
        app = AppLauncher({'headless': True, 'enable_cameras': False}).app
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
    
        checkpoint = Path(f'runs/phase1_r1_dt01_seed{args.seed}_2026-09-20/checkpoints/checkpoint.pt')
        stored = yaml.safe_load((checkpoint.parents[1] / 'config.yaml').read_text())
        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = 1001
        cfg.sim.dt = .01
        cfg.decimation = 1
        cfg.sim.render_interval = 1
        cfg.g1_command_exposure = args.g1
        if args.environment == 'nominal':
            _nominalize_cfg(cfg)
        else:
            # Fresh level-0 terrain distribution, held fixed for paired comparisons;
            # the final training simulator/curriculum state was not checkpointed.
            cfg.curriculum.terrain_levels = None
            if args.environment == 'terrain_only':
                # Keep generated terrain and native terrain origins; remove physical
                # randomization while retaining the same scripted command probe.
                cfg.events.randomize_payload_mass = None
                cfg.events.randomize_payload_com = None
                cfg.events.randomize_friction = None
                cfg.events.randomize_motor_power = None
                cfg.events.randomize_joint_range = None
                cfg.events.push_robot = None
            elif args.environment == 'physics_only':
                # Flat nominal plane, retain reset-time physical randomization.
                cfg.scene.terrain.terrain_type = 'plane'
                cfg.scene.terrain.terrain_generator = None
                cfg.scene.robot = cfg.scene.robot.replace()
                cfg.events.push_robot = None
                cfg.curriculum.terrain_levels = None
        env = gym.make('Isaac-Talon-A1-v0', cfg=cfg, render_mode=None).unwrapped
        if args.init_only:
            transition = env.reset()
            payload = dict(environment=args.environment, seed=args.seed,
                           num_envs=args.num_envs, checkpoint=str(checkpoint),
                           checkpoint_sha256=_sha256(checkpoint), reset_keys=sorted(transition),
                           terrain_type=cfg.scene.terrain.terrain_type,
                           has_terrain_generator=cfg.scene.terrain.terrain_generator is not None,
                           events={k: getattr(cfg.events, k) is not None for k in
                                   ('randomize_payload_mass','randomize_payload_com','randomize_friction',
                                    'randomize_motor_power','randomize_joint_range','push_robot')})
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f'init_{args.environment}_n{args.num_envs}.json').write_text(json.dumps(payload, indent=2) + '\n')
            print(json.dumps(payload), flush=True)
            env.close(); app.close(); return
        moppo = MOPPOConfig(**_select(MOPPOConfig, stored['moppo']))
        moppo.torch_compile = False  # Same model operations, no compilation overhead.
        trainer = MOPPOTrainer(env,
            ObservationSpaceCfg(**_select(ObservationSpaceCfg, stored['obs'])),
            RewardVectorCfg(**_select(RewardVectorCfg, stored['reward'])),
            PreferenceCfg(**_select(PreferenceCfg, stored['preference'])),
            moppo_cfg=moppo,
            stack_cfg=ObservationStackCfg(**_select(ObservationStackCfg, stored['stack'])),
            extrinsics_cfg=ExtrinsicsCfg(), seed=1001)
        trainer.load(str(checkpoint))
        trainer.model.eval()
        model_before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        norm_before = copy.deepcopy(trainer.obs_norm.state_dict())
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        summaries = []
        conditions = [('eval_reset', 'forced'), ('eval_reset', 'random'),
                      ('training_reset', 'random'), ('eval_reset', 'forced_fresh_obs')]
        if args.condition == 'forced_mean':
            conditions = [('eval_reset', 'forced')]
        if args.environment == 'training':
            conditions = [('training_reset', 'random'), ('training_reset', 'forced')]
        for reset, schedule in conditions:
            paired_initial = None
            modes = [('mean', 0), ('sample', 7001), ('sample', 7002)]
            if args.condition == 'forced_mean':
                modes = [('mean', 0)]
            for mode, noise_seed in modes:
                # Same randomization draws in each paired reset, independent action RNG.
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(gpu_rng)
                env.common_step_counter = 12000 if args.environment == 'training' else 0
                transition = env.reset()
                if reset == 'eval_reset':
                    _set_initial_state(env, np.random.default_rng(1001))
                    transition = env._transition(env.observation_manager.compute())
                initial_command = env.v_command_buf.cpu().numpy().copy()
                if schedule == 'forced_fresh_obs':
                    env.v_command_buf[:] = 0
                    transition = env._transition(env.observation_manager.compute())
                initial = np.concatenate([
                    env.scene['robot'].data.root_state_w.cpu().numpy(),
                    env.scene['robot'].data.joint_pos.cpu().numpy(),
                    env.scene['robot'].data.joint_vel.cpu().numpy(), initial_command], axis=1)
                if paired_initial is None:
                    paired_initial = initial.copy()
                else:
                    np.testing.assert_allclose(initial, paired_initial, atol=1e-6, rtol=0)
                trainer.push_obs(transition['obs'], update_normalizer=False)
                trainer._last_extrinsics = transition.get('extrinsics') if trainer.encoder else None
                trainer.w[:] = .25
                generator = torch.Generator(device=trainer.device).manual_seed(noise_seed)
                alive = np.ones(64, bool)
                failure = np.full(64, 600, dtype=int)
                reason = np.full(64, 'censored', dtype='<U24')
                trace = []
                for step in range(600):
                    if schedule.startswith('forced'):
                        env.v_command_buf[:] = torch.tensor(
                            [0. if step < 200 else -.25, 0., 0.], device=env.device)
                    command = env.v_command_buf.cpu().numpy().copy()
                    with torch.no_grad():
                        obs = torch.from_numpy(trainer._actor_obs()).to(trainer.device)
                        dist = trainer.model._pre_tanh_dist(obs)
                        mean = dist.mean
                        u = mean if mode == 'mean' else mean + dist.stddev * torch.randn(
                            mean.shape, device=mean.device, generator=generator)
                        action = (torch.tanh(u) * trainer.model.ACTION_CLIP).cpu().numpy()
                        if mode == 'mean':
                            np.testing.assert_allclose(action, trainer.act_inference(), atol=1e-6)
                    transition, done = env.step(action)
                    frame = _terminal_frame(transition, done)
                    finite = np.isfinite(action).all(1) & np.isfinite(frame['height'])
                    newly = alive & (done | ~finite)
                    failure[newly] = step + 1
                    for key in ['term_time_out', 'term_obstacle_reached', 'term_base_contact']:
                        reason[newly & frame[key].astype(bool)] = key
                    reason[newly & ~finite] = 'numerical'
                    if step < 50:
                        trace.append(dict(actor_mean=mean.cpu().numpy().copy(),
                            pre_tanh_action=u.cpu().numpy().copy(), action=action.copy(),
                            alive=alive.copy(), height=frame['height'].copy(),
                            roll_pitch=frame['roll_pitch'].copy(),
                            contact=frame['foot_contact_force'].copy(),
                            undesired_contact=frame['undesired_contact_count'].copy(),
                            command=command, done=done.copy()))
                    alive &= ~(done | ~finite)
                    trainer._last_extrinsics = transition.get('extrinsics') if trainer.encoder else None
                    trainer.push_obs(transition['obs'], done_mask=done, update_normalizer=False)
                    if step >= 49 and not alive.any():
                        break
                name = f'seed{args.seed}_{args.environment}_{reset}_{schedule}_{mode}_{noise_seed}'
                np.savez_compressed(args.output / f'{name}.npz',
                    initial=initial, failure_step=failure, censored=alive, reason=reason,
                    env_origins=env.scene.env_origins.cpu().numpy(),
                    log_std=trainer.model.log_std.detach().cpu().numpy(),
                    std=trainer.model.log_std.detach().exp().cpu().numpy(),
                    **{k: np.stack([t[k] for t in trace]) for k in trace[0]})
                entry = dict(name=name, training_seed=args.seed, environment=args.environment,
                    reset=reset, schedule=schedule, mode=mode, noise_seed=noise_seed,
                    lanes=64, horizon=600, failures=int((~alive).sum()),
                    before_switch=int(((failure <= 200) & ~alive).sum()),
                    survive_50=int((failure > 50).sum()), survive_200=int((failure > 200).sum()),
                    survive_600=int(alive.sum()), restricted_mean_steps=float(failure.mean()),
                    median_steps=float(np.median(failure)),
                    std_min=float(trainer.model.log_std.exp().min()),
                    std_max=float(trainer.model.log_std.exp().max()))
                summaries.append(entry)
                print(json.dumps(entry), flush=True)
                (args.output / f'seed{args.seed}_{args.environment}.json').write_text(json.dumps(
                    dict(checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint),
                         eval_seed=1001, results=summaries), indent=2) + '\n')
        for key, value in trainer.model.state_dict().items():
            assert torch.equal(value, model_before[key]), key
        for key, value in trainer.obs_norm.state_dict().items():
            np.testing.assert_array_equal(value, norm_before[key])
        print('READ_ONLY_CHECK_PASS', flush=True)
        env.close()
        app.close()
    
    
    if True:
        main()

def run_r1_offline_audit():
    """Run former r1_offline_audit.py stage."""
    """R1 offline audit: analyze the 45-cell read-only replay to decide, per
    sub-reward, retain / gate / move / disable -- no simulator needed.
    
    Reads artifacts/r1_readonly_replay_v2/*.npz (component traces from
    r1_readonly_replay.py) and writes a report + machine-readable summary.
    """
    
    import json
    from pathlib import Path
    
    import numpy as np
    
    REPLAY_DIR = Path("artifacts/r1_readonly_replay_v2")
    OUT_DIR = Path("artifacts/r1_offline_audit")
    SETTLE = 200
    COMMANDS = (-0.25, 0.0, 0.25, 0.5, 0.75)
    
    
    def load_cells():
        cells = []
        for path in sorted(REPLAY_DIR.glob("seed*_eval*_vx*.npz")):
            d = np.load(path)
            names = list(d["component_names"])
            cells.append({
                "path": path.name,
                "training_seed": int(d["training_seed"]),
                "eval_seed": int(d["eval_seed"]),
                "command_vx": float(d["command_vx"]),
                "components": d["components"],  # (2200, 64, 19)
                "names": names,
                "first_fall_step": d["first_fall_step"],
            })
        return cells
    
    
    def col(cell, name):
        return cell["components"][:, :, cell["names"].index(name)]
    
    
    def post_settle_alive_mask(cell):
        """(steps-settle, lanes) bool mask: True where the lane is alive AND
        past settle -- pre-fall K=5 zeroing already applied to progress_training_k5
        only, so for other terms we mask on first_fall_step ourselves."""
        steps, lanes = cell["components"].shape[0], cell["components"].shape[1]
        fall = cell["first_fall_step"]
        step_idx = np.arange(SETTLE, steps)[:, None]
        return step_idx < fall[None, :]
    
    
    def summarize(values, mask):
        v = values[mask]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"mean": None, "std": None, "frac_zero": None, "frac_negative": None, "n": 0}
        return {
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "frac_zero": float(np.mean(v == 0.0)), "frac_negative": float(np.mean(v < 0.0)),
            "n": int(v.size),
        }
    
    
    def by_command_sign(cells, name, transform=None):
        """Group post-settle, pre-fall values of `name` by command sign (zero / positive / negative)."""
        groups = {"zero": [], "positive": [], "negative": []}
        for cell in cells:
            vx = cell["command_vx"]
            key = "zero" if vx == 0.0 else ("positive" if vx > 0 else "negative")
            mask = post_settle_alive_mask(cell)
            vals = col(cell, name)[SETTLE:]
            if transform is not None:
                vals = transform(cell, vals)
            groups[key].append(summarize(vals, mask))
        return {k: aggregate(v) for k, v in groups.items()}
    
    
    def aggregate(summaries):
        means = [s["mean"] for s in summaries if s["mean"] is not None]
        stds = [s["std"] for s in summaries if s["std"] is not None]
        fz = [s["frac_zero"] for s in summaries if s["frac_zero"] is not None]
        fn = [s["frac_negative"] for s in summaries if s["frac_negative"] is not None]
        n = sum(s["n"] for s in summaries)
        if not means:
            return {"mean": None, "std": None, "frac_zero": None, "frac_negative": None, "n": n}
        return {
            "mean": float(np.mean(means)), "std": float(np.mean(stds)),
            "frac_zero": float(np.mean(fz)), "frac_negative": float(np.mean(fn)), "n": n,
        }
    
    
    def balance_decomposition(cells):
        terms = ("balance_tilt", "balance_alive", "balance_vz", "balance_height",
                 "balance_hip_activation", "balance_hip_symmetry", "balance_fall")
        out = {}
        for t in terms:
            summaries = []
            for cell in cells:
                mask = post_settle_alive_mask(cell)
                summaries.append(summarize(col(cell, t)[SETTLE:], mask))
            out[t] = aggregate(summaries)
        total_summaries = []
        for cell in cells:
            mask = post_settle_alive_mask(cell)
            total_summaries.append(summarize(col(cell, "balance_total")[SETTLE:], mask))
        out["balance_total"] = aggregate(total_summaries)
        # share of |total| explained by |mean of each term| (rough decomposition, not exact since signs vary per-step)
        denom = sum(abs(out[t]["mean"]) for t in terms if out[t]["mean"] is not None) or 1.0
        out["_share_of_abs_mean"] = {t: (abs(out[t]["mean"]) / denom if out[t]["mean"] is not None else None) for t in terms}
        return out
    
    
    def exact_vs_k5(cells):
        """How many post-settle, pre-fall steps does K=5 masking zero out that
        the exact mask would not, and does it change the mean reward materially."""
        diffs = []
        exact_means, k5_means = [], []
        for cell in cells:
            mask = post_settle_alive_mask(cell)
            exact = col(cell, "progress_exact_mask")[SETTLE:]
            k5 = col(cell, "progress_training_k5")[SETTLE:]
            m = mask & np.isfinite(exact) & np.isfinite(k5)
            if not m.any():
                continue
            diffs.append(float(np.mean(exact[m] != k5[m])))
            exact_means.append(float(np.mean(exact[m])))
            k5_means.append(float(np.mean(k5[m])))
        return {
            "frac_steps_differ": float(np.mean(diffs)) if diffs else None,
            "exact_mean": float(np.mean(exact_means)) if exact_means else None,
            "k5_mean": float(np.mean(k5_means)) if k5_means else None,
            "relative_reward_shrink": (
                float(1 - np.mean(k5_means) / np.mean(exact_means))
                if exact_means and np.mean(exact_means) else None
            ),
        }
    
    
    def zero_reverse_controls(cells):
        out = {}
        zero_cells = [c for c in cells if c["command_vx"] == 0.0]
        rev_cells = [c for c in cells if c["command_vx"] < 0]
        fwd_cells = [c for c in cells if c["command_vx"] > 0]
    
        def eng_stats(group):
            summaries = []
            for cell in group:
                mask = post_settle_alive_mask(cell)
                summaries.append(summarize(col(cell, "signed_engagement")[SETTLE:], mask))
            return aggregate(summaries)
    
        def dirv_stats(group):
            summaries = []
            for cell in group:
                mask = post_settle_alive_mask(cell)
                summaries.append(summarize(col(cell, "directed_velocity_component")[SETTLE:], mask))
            return aggregate(summaries)
    
        out["engagement_zero_command"] = eng_stats(zero_cells)  # expect ~1 always (hardcoded ones())
        out["engagement_forward"] = eng_stats(fwd_cells)
        out["engagement_reverse"] = eng_stats(rev_cells)
        out["directed_velocity_forward"] = dirv_stats(fwd_cells)
        out["directed_velocity_reverse"] = dirv_stats(rev_cells)
    
        # sign sanity: for reverse commands, v_actual_x should trend negative post-settle
        vx_means = []
        for cell in rev_cells:
            mask = post_settle_alive_mask(cell)
            vx_means.append(summarize(col(cell, "v_actual_x")[SETTLE:], mask)["mean"])
        out["reverse_command_v_actual_x_mean"] = float(np.mean([v for v in vx_means if v is not None]))
        vx_means_fwd = []
        for cell in fwd_cells:
            mask = post_settle_alive_mask(cell)
            vx_means_fwd.append(summarize(col(cell, "v_actual_x")[SETTLE:], mask)["mean"])
        out["forward_command_v_actual_x_mean"] = float(np.mean([v for v in vx_means_fwd if v is not None]))
        return out
    
    
    def hip_activation_analysis(cells):
        """Does hip_activation track locomotion speed (progress-like) or is it
        speed-independent (genuinely a balance term)?"""
        per_speed = {}
        for cell in cells:
            if cell["command_vx"] == 0.0:
                continue
            mask = post_settle_alive_mask(cell)
            hip = summarize(col(cell, "balance_hip_activation")[SETTLE:], mask)
            speed = abs(cell["command_vx"])
            per_speed.setdefault(speed, []).append(hip["mean"])
        return {str(k): float(np.mean([x for x in v if x is not None])) for k, v in sorted(per_speed.items())}
    
    
    def main():
        cells = load_cells()
        assert len(cells) == 45, f"expected 45 replay cells, found {len(cells)}"
    
        report = {
            "n_cells": len(cells),
            "tracking_kernel_by_command_sign": by_command_sign(cells, "tracking_kernel"),
            "airtime_by_command_sign": by_command_sign(cells, "airtime_component"),
            "balance_decomposition": balance_decomposition(cells),
            "exact_vs_k5_masking": exact_vs_k5(cells),
            "zero_reverse_controls": zero_reverse_controls(cells),
            "hip_activation_vs_speed": hip_activation_analysis(cells),
        }
    
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "audit-data.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    
    
    if True:
        main()

def run_r1_readonly_replay():
    """Run former r1_readonly_replay.py stage."""
    """Read-only R1 replay: record missing per-step reward inputs, never update."""
    
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    
    import argparse
    import json
    import os
    from dataclasses import fields
    from pathlib import Path
    
    import numpy as np
    import torch
    import yaml
    
    from talon_rl.config import ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg
    from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
    from rl.experiments.common.utilities.final_locomotion_eval import (
        COMMANDS,
        PREFERENCES,
        _nominalize_cfg,
        _run_cell,
        _select,
        _set_initial_state,
        _terminal_frame,
    )
    
    
    UNIFORM = np.array((0.25, 0.25, 0.25, 0.25), dtype=np.float32)
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("checkpoint", type=Path)
        parser.add_argument("--training-seed", type=int, required=True, choices=(0, 1, 2))
        parser.add_argument("--eval-seed", type=int, required=True, choices=(1001, 1002, 1003))
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args()
    
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
        simulation_app = app_launcher.app
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
    
        stored = yaml.safe_load((args.checkpoint.resolve().parents[1] / "config.yaml").read_text())
        obs_cfg = ObservationSpaceCfg(**_select(ObservationSpaceCfg, stored["obs"]))
        action_cfg = ActionSpaceCfg(**_select(ActionSpaceCfg, stored["action"]))
        reward_cfg = RewardVectorCfg(**_select(RewardVectorCfg, stored["reward"]))
        pref_cfg = PreferenceCfg(**_select(PreferenceCfg, stored["preference"]))
        stack_cfg = ObservationStackCfg(**_select(ObservationStackCfg, stored["stack"]))
        moppo_cfg = MOPPOConfig(**_select(MOPPOConfig, stored["moppo"]))
        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = 64; cfg.seed = args.eval_seed
        cfg.sim.dt = 0.01; cfg.decimation = 1; cfg.sim.render_interval = 1
        _nominalize_cfg(cfg)
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
        trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg,
                               stack_cfg=stack_cfg, extrinsics_cfg=ExtrinsicsCfg(), seed=args.eval_seed)
        trainer.load(str(args.checkpoint)); trainer.model.eval()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
        component_names = (
            "tracking_kernel", "airtime_component", "progress_exact_mask",
            "progress_training_k5", "signed_engagement", "directed_velocity_component",
            "balance_tilt", "balance_alive", "balance_vz", "balance_height",
            "balance_hip_activation", "balance_hip_symmetry", "balance_fall", "balance_total",
            "height", "hip_qdot_l", "hip_qdot_r", "terminal_fall", "v_actual_x",
        )
        masses = env.scene["robot"].root_physx_view.get_masses().detach().cpu().numpy().sum(axis=1)
        traced_commands = []
        # Preserve the exact 25-cell v1 process history. Isaac Lab reset does not
        # rewind every simulator/RNG stream, so launching only the target cells in
        # a fresh process does not reproduce the frozen evaluation trajectories.
        for vx_cmd in COMMANDS:
            for preference in PREFERENCES:
                if preference != "uniform":
                    _run_cell(env, trainer, reward_cfg, vx_cmd, preference, args.eval_seed, 200, 2000, masses)
                    continue
                print(f"replay vx={vx_cmd:g}", flush=True)
                transition = env.reset()
                _, initial_yaw = _set_initial_state(env, np.random.default_rng(args.eval_seed))
                transition = env._transition(env.observation_manager.compute())
                trainer.push_obs(transition["obs"], update_normalizer=False)
                trainer._last_extrinsics = transition.get("extrinsics")
                trainer.w = np.repeat(UNIFORM[None], env.num_envs, axis=0)
                total, settle, n = 2200, 200, env.num_envs
                data = np.full((total, n, len(component_names)), np.nan, dtype=np.float32)
                root_xy = np.full((total, n, 2), np.nan, dtype=np.float32)
                alive = np.ones(n, dtype=bool)
                first_fall_step = np.full(n, total, dtype=np.int32)
                norm_before = trainer.obs_norm.state_dict()
                for step in range(total):
                    command = 0.0 if step < settle else vx_cmd
                    env.v_command_buf[:] = torch.tensor((command, 0.0, 0.0), device=env.device)
                    action = trainer.act_inference()
                    transition, done = env.step(action)
                    frame = _terminal_frame(transition, done)
                    trainer._last_extrinsics = transition.get("extrinsics")
                    trainer.push_obs(transition["obs"], done_mask=done, update_normalizer=False)
                    terminal = frame.get("terminal_fall", done).astype(bool)
                    use = alive.copy()
                    first_fall_step[use & terminal] = step
                    err = frame["v_actual"][:, 0] - command
                    tracking = np.exp(-(err ** 2) / reward_cfg.progress_std ** 2)
                    air = 0.04 * frame["foot_air_time_reward"]
                    exact = (tracking + air) * (~terminal)
                    engagement = np.clip(np.sign(command) * frame["v_actual"][:, 0] / max(abs(command), 1e-6), 0, 1) if command else np.ones(n)
                    rp = frame["roll_pitch"]; tilt = -reward_cfg.balance_tilt_coef * np.sum(rp ** 2, axis=1)
                    alive_term = np.full(n, reward_cfg.alive_bonus, dtype=np.float32)
                    vz_term = -(frame["v_z"] ** 2)
                    height_term = -reward_cfg.balance_height_coef * (frame["height"] - reward_cfg.target_height) ** 2
                    hip = reward_cfg.balance_hip_activation_coef * np.minimum(np.abs(frame["hip_qdot_L"]), np.abs(frame["hip_qdot_R"]))
                    hip_sym = -reward_cfg.balance_hip_sym_coef * (frame["hip_q_L"] + frame["hip_q_R"]) ** 2
                    fall = -reward_cfg.fall_penalty * terminal.astype(np.float32)
                    values = (tracking, air, exact, exact.copy(), engagement, engagement * tracking,
                              tilt, alive_term, vz_term, height_term, hip, hip_sym, fall,
                              tilt + alive_term + vz_term + height_term + hip + hip_sym + fall,
                              frame["height"], frame["hip_qdot_L"], frame["hip_qdot_R"], terminal.astype(np.float32),
                              frame["v_actual"][:, 0])
                    for idx, value in enumerate(values): data[step, use, idx] = value[use]
                    root_xy[step, use] = frame["root_pos_w"][use, :2]
                    alive &= ~done
                # Match training's K=5 retroactive raw Progress zeroing.
                k5 = component_names.index("progress_training_k5")
                for lane, fall_step in enumerate(first_fall_step):
                    if fall_step < total:
                        data[max(0, fall_step - 4):fall_step + 1, lane, k5] = 0.0
                for key in norm_before:
                    if not np.array_equal(norm_before[key], trainer.obs_norm.state_dict()[key]):
                        raise RuntimeError(f"normalizer mutated: {key}")
                path = args.output_dir / f"seed{args.training_seed}_eval{args.eval_seed}_vx{vx_cmd:g}.npz"
                np.savez_compressed(path, components=data, root_xy=root_xy,
                                    component_names=np.array(component_names), initial_yaw=initial_yaw,
                                    first_fall_step=first_fall_step, command_vx=vx_cmd,
                                    training_seed=args.training_seed, eval_seed=args.eval_seed,
                                    checkpoint=str(args.checkpoint))
                traced_commands.append(vx_cmd)
        meta = {"training_seed": args.training_seed, "eval_seed": args.eval_seed,
                "commands": traced_commands, "cell_order": [[v, p] for v in COMMANDS for p in PREFERENCES],
                "lanes": 64, "steps": 2200,
                "read_only": True, "training_updates": trainer._t // moppo_cfg.num_steps}
        (args.output_dir / f"seed{args.training_seed}_eval{args.eval_seed}.json").write_text(json.dumps(meta, indent=2) + "\n")
        env.close(); simulation_app.close()
    
    
    if True:
        main()

def run_r1_replay_validate():
    """Run former r1_replay_validate.py stage."""
    """Validate that R1 replay traces reproduce the frozen v1 uniform cells."""
    
    
    import argparse
    import csv
    import json
    from pathlib import Path
    
    import numpy as np
    
    
    COMMANDS = (-0.25, 0.0, 0.25, 0.50, 0.75)
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--replay-dir", type=Path, default=Path("artifacts/r1_readonly_replay_v2"))
        parser.add_argument("--frozen-dir", type=Path, default=Path("artifacts/final_locomotion_eval_v1/episodes"))
        parser.add_argument("--require-complete", action="store_true")
        args = parser.parse_args()
    
        results = []
        missing = []
        for training_seed in range(3):
            for eval_seed in (1001, 1002, 1003):
                csv_path = args.frozen_dir / f"seed{training_seed}_eval{eval_seed}_episodes.csv"
                rows = list(csv.DictReader(csv_path.open()))
                for command in COMMANDS:
                    path = args.replay_dir / f"seed{training_seed}_eval{eval_seed}_vx{command:g}.npz"
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    frozen = sorted(
                        (r for r in rows if r["preference"] == "uniform" and float(r["command_vx"]) == command),
                        key=lambda r: int(r["lane"]),
                    )
                    replay = np.load(path)
                    names = list(replay["component_names"])
                    values = replay["components"]
                    vx = values[:, :, names.index("v_actual_x")]
                    xy = replay["root_xy"]
                    yaw = replay["initial_yaw"]
                    first_fall = replay["first_fall_step"]
    
                    replay_failure = np.where(first_fall < 2200, first_fall + 1, 2200)
                    frozen_failure = np.array([int(r["failure_step"]) for r in frozen])
                    valid_steps = np.array([int(r["valid_command_steps"]) for r in frozen])
                    replay_mae = np.zeros(64)
                    for lane in range(64):
                        lane_vx = vx[200:, lane]
                        lane_vx = lane_vx[np.isfinite(lane_vx)]
                        if len(lane_vx):
                            replay_mae[lane] = np.mean(np.abs(lane_vx - command))
                    start = xy[199]
                    last = np.empty_like(start)
                    for lane in range(64):
                        indices = np.flatnonzero(np.isfinite(xy[:, lane, 0]))
                        last[lane] = xy[indices[-1], lane]
                    delta = last - start
                    replay_directed = (
                        np.sign(command) * (delta[:, 0] * np.cos(yaw) + delta[:, 1] * np.sin(yaw))
                        if command else np.zeros(64)
                    )
                    # a lane that fails before step 199 never reaches the command
                    # window, leaving xy[199] NaN; frozen data scores those lanes
                    # 0 displacement, so match that instead of letting NaN poison np.max.
                    replay_directed = np.where(np.isfinite(start[:, 0]), replay_directed, 0.0)
                    frozen_mae = np.array([float(r["vx_mae"]) for r in frozen])
                    frozen_directed = np.array([float(r["directed_displacement_m"]) for r in frozen])
    
                    failure_mismatches = int(np.count_nonzero(replay_failure != frozen_failure))
                    valid_count_mismatches = int(np.count_nonzero(np.isfinite(vx[200:]).sum(axis=0) != valid_steps))
                    mae_max_error = float(np.max(np.abs(replay_mae - frozen_mae)))
                    displacement_max_error = float(np.max(np.abs(replay_directed - frozen_directed)))
    
                    k5 = values[:, :, names.index("progress_training_k5")]
                    exact = values[:, :, names.index("progress_exact_mask")]
                    expected = exact.copy()
                    for lane, fall_step in enumerate(first_fall):
                        if fall_step < 2200:
                            expected[max(0, fall_step - 4):fall_step + 1, lane] = 0.0
                    k5_mismatches = int(np.count_nonzero(~np.isclose(k5, expected, equal_nan=True)))
                    passed = (
                        failure_mismatches == 0 and valid_count_mismatches == 0
                        and mae_max_error <= 2e-5 and displacement_max_error <= 2e-6
                        and k5_mismatches == 0
                    )
                    results.append({
                        "training_seed": training_seed, "eval_seed": eval_seed, "command_vx": command,
                        "passed": passed, "failure_step_mismatches": failure_mismatches,
                        "valid_step_count_mismatches": valid_count_mismatches,
                        "vx_mae_max_abs_error": mae_max_error,
                        "directed_displacement_max_abs_error_m": displacement_max_error,
                        "k5_mask_mismatches": k5_mismatches,
                    })
    
        summary = {
            "status": "PASS" if results and all(r["passed"] for r in results) and not missing else "INCOMPLETE_OR_FAIL",
            "validated_cells": len(results), "expected_cells": 45, "missing": missing, "cells": results,
        }
        args.replay_dir.mkdir(parents=True, exist_ok=True)
        (args.replay_dir / "replay-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ("status", "validated_cells", "expected_cells")}, indent=2))
        if any(not r["passed"] for r in results) or (args.require_complete and missing):
            raise SystemExit(1)
    
    
    if True:
        main()

def run_run_r1_mismatch_audit():
    """Run former run_r1_mismatch_audit.py stage."""
    """Run the bounded 3-checkpoint audit, with separate simulator processes."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    
    root = Path(__file__).resolve().parents[4]
    output = root / 'artifacts/r1_mismatch_audit'
    output.mkdir(parents=True, exist_ok=True)
    for environment in ['nominal', 'training']:
        for seed in range(3):
            log = output / f'seed{seed}_{environment}.log'
            if log.exists() and 'READ_ONLY_CHECK_PASS' in log.read_text():
                continue
            # The initial nominal seed-0 process is run separately as the smoke check.
            smoke = Path('/tmp/r1_audit_seed0_nominal.log')
            if seed == 0 and environment == 'nominal' and smoke.exists() and 'READ_ONLY_CHECK_PASS' in smoke.read_text():
                log.write_text(smoke.read_text())
                continue
            print(f'RUN seed={seed} environment={environment}', flush=True)
            with log.open('w') as stream:
                subprocess.run([sys.executable, str(root / 'scripts/rl/experiments/baselines/reward_revision/workflow.py'), 'r1_mismatch_audit',
                                '--seed', str(seed), '--environment', environment],
                               cwd=root, env=dict(os.environ, PYTHONPATH=f'{root}:{root / "scripts"}'),
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            if 'READ_ONLY_CHECK_PASS' not in log.read_text():
                raise RuntimeError(f'Missing immutability check: {log}')
    print('ALL_AUDIT_SHARDS_COMPLETE', flush=True)

def run_run_r1_replay_shards():
    """Run former run_r1_replay_shards.py stage."""
    """Run/resume all nine read-only R1 replay shards."""
    import argparse, os, subprocess
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    OUT = ROOT / "artifacts/r1_readonly_replay_v2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--eval-seed", type=int, choices=(1001, 1002, 1003))
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'scripts'}:{ROOT / 'scripts/rl'}")
    for seed in range(3):
        if args.training_seed is not None and seed != args.training_seed: continue
        ckpt = ROOT / f"runs/phase1_hipact_dt01_seed{seed}_2026-09-20/checkpoints/checkpoint.pt"
        for eval_seed in (1001, 1002, 1003):
            if args.eval_seed is not None and eval_seed != args.eval_seed: continue
            targets = [OUT / f"seed{seed}_eval{eval_seed}_vx{vx:g}.npz" for vx in (-.25, 0, .25, .5, .75)]
            if all(path.is_file() for path in targets):
                print(f"skip seed={seed} eval={eval_seed}", flush=True); continue
            print(f"start seed={seed} eval={eval_seed}", flush=True)
            subprocess.run(["/home/xero/isaac-lab-env/bin/python", str(ROOT / "scripts/rl/experiments/baselines/reward_revision/workflow.py"), "r1_readonly_replay",
                            str(ckpt), "--training-seed", str(seed), "--eval-seed", str(eval_seed),
                            "--output-dir", str(OUT)], cwd=ROOT, env=env, check=True)
            if not all(path.is_file() for path in targets): raise RuntimeError("replay shard incomplete")
    print("all replay shards complete", flush=True)

def run_run_r1_retrain_shards():
    """Run former run_r1_retrain_shards.py stage."""
    """Run/resume the R1 3-seed retrain (artifacts/r1_freeze/FREEZE.md).
    
    Flags reconstructed from runs/phase1_hipact_dt01_seed0_2026-09-20/config.yaml
    to match the original baseline run EXACTLY except for the reward code
    itself (RewardVectorCfg's R1 field defaults now apply automatically --
    no reward-related CLI flags are passed here, which is the point: nothing
    about the ablation is a coefficient the retrain script chose, it's what
    reward.py's new defaults already are). Confirmed against that config.yaml:
    mean_reg_coef=0.01 and progress_leak_window=5 are the only two MOPPOConfig
    values that differ from train_prelim.py's own CLI defaults; every other
    MOPPOConfig field in that file already matches its dataclass default
    (weight_decay, gamma, gae_lambda, clip_eps, epochs_per_update,
    num_minibatches, num_steps, entropy_coef, diversity_lambda/alpha,
    log_std_max_anneal_*) so needs no override. num_envs=4096 confirmed from
    that run's logged total_timesteps/update (98304 = 4096*24). sim_dt=0.01
    confirmed by the run's own name ("dt01", Experiment 2E's dt causality
    ablation) and IsaacLabTalonEnvCfg's decimation=1 default (unchanged).
    """
    import argparse, os, subprocess
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parents[4]
    DATE = "2026-09-20"
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'scripts'}:{ROOT / 'scripts/rl'}")
    
    for seed in range(3):
        if args.seed is not None and seed != args.seed:
            continue
        run_name = f"phase1_g1_dt01_seed{seed}_{DATE}"
        checkpoint = ROOT / "runs" / run_name / "checkpoints" / "checkpoint.pt"
        if checkpoint.is_file():
            print(f"skip seed={seed} (checkpoint already exists at {checkpoint})", flush=True)
            continue
        print(f"start seed={seed}", flush=True)
        subprocess.run(
            [
                "/home/xero/isaac-lab-env/bin/python", "-u", str(ROOT / "scripts/rl/train_prelim.py"),
                "--env", "isaac_lab",
                "--num_envs", "4096",
                "--updates", "500",
                "--seed", str(seed),
                "--sim_dt", "0.01",
                "--torch_compile",
                "--mean_reg_coef", "0.01",
                "--progress_leak_window", "5",
                "--g1_command_exposure",
                "--logs_root", str(ROOT / "runs"),
                "--run_name", run_name,
            ],
            cwd=ROOT, env=env, check=True,
        )
        if not checkpoint.is_file():
            raise RuntimeError(f"retrain seed={seed} finished without producing {checkpoint}")
    
    print("all R1 retrain shards complete", flush=True)

def run_run_r1_single_cell():
    """Run former run_r1_single_cell.py stage."""
    """Direct one-cell runner with lifecycle markers and GPU diagnostics."""
    import json, os, subprocess, sys, time
    from pathlib import Path
    
    root = Path(__file__).resolve().parents[4]
    variant = sys.argv[1] if len(sys.argv) > 1 else 'terrain_only'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = root / 'artifacts/r1_mismatch_audit'
    prefix = out / f'single_{variant}_seed{seed}_forced_mean'
    started = prefix.with_name(prefix.name + '_RUN_STARTED.json')
    done = prefix.with_name(prefix.name + '_RUN_DONE.json')
    log = prefix.with_suffix('.log')
    cmd = [sys.executable, str(root/'scripts/rl/experiments/baselines/reward_revision/workflow.py'), 'r1_mismatch_audit',
           '--seed', str(seed), '--environment', variant, '--condition', 'forced_mean']
    payload = dict(command=cmd, variant=variant, seed=seed, condition='forced_mean',
                   started_unix=time.time(), pid=None, checkpoint_sha256='c5dfe42edfe1375dcd4293c1308327a33aafa68c20aecc601876f4f6adae88fd')
    out.mkdir(parents=True, exist_ok=True)
    started.write_text(json.dumps(payload, indent=2) + '\n')
    gpu_before = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'], text=True, capture_output=True).stdout.strip()
    with log.open('w') as stream:
        proc = subprocess.Popen(cmd, cwd=root, env=dict(os.environ, PYTHONPATH=f'{root}:{root/"scripts"}'), stdout=stream, stderr=subprocess.STDOUT)
        payload['pid'] = proc.pid
        started.write_text(json.dumps(payload, indent=2) + '\n')
        code = proc.wait()
    gpu_after = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'], text=True, capture_output=True).stdout.strip()
    payload.update(ended_unix=time.time(), exit_code=code, gpu_before=gpu_before, gpu_after=gpu_after)
    json_path = out / f'seed{seed}_{variant}.json'
    npz = list(out.glob(f'seed{seed}_{variant}_*.npz'))
    complete = code == 0 and json_path.is_file() and len(npz) >= 1 and 'READ_ONLY_CHECK_PASS' in log.read_text(errors='replace')
    payload['artifact_json'] = str(json_path)
    payload['npz_count'] = len(npz)
    payload['complete'] = complete
    if complete:
        done.write_text(json.dumps(payload, indent=2) + '\n')
    else:
        payload['failure_reason'] = 'no completion marker or incomplete artifact'
        done.write_text(json.dumps(payload, indent=2) + '\n')
        raise SystemExit(code or 2)
    print(json.dumps(payload), flush=True)

def run_summarize_r1_mismatch_audit():
    """Run former summarize_r1_mismatch_audit.py stage."""
    """Summarize saved mismatch traces, without simulator access."""
    import csv
    import hashlib
    import json
    from pathlib import Path
    
    import numpy as np
    
    ROOT = Path(__file__).resolve().parents[4]
    OUT = ROOT / 'artifacts/r1_mismatch_audit'
    
    
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def main():
        rows, replay, frozen = [], [], []
        for seed in range(3):
            for environment in ['nominal', 'training']:
                source = OUT / f'seed{seed}_{environment}.json'
                data = json.loads(source.read_text())
                assert 'READ_ONLY_CHECK_PASS' in (OUT / f'seed{seed}_{environment}.log').read_text()
                for r in data['results']:
                    z = np.load(OUT / (r['name'] + '.npz'))
                    r['termination_counts'] = dict(zip(*np.unique(z['reason'], return_counts=True)))
                    live = z['alive']
                    r['first50_saturation_fraction'] = float((np.abs(z['action'][live]) >= 2.85).mean())
                    r['first50_mean_abs_raw_actor_mean'] = float(np.abs(z['actor_mean'][live]).mean())
                    r['first50_mean_height'] = float(z['height'][live].mean())
                    r['first50_mean_max_abs_tilt'] = float(np.abs(z['roll_pitch'][live]).max(-1).mean())
                    r['first50_undesired_contact_fraction'] = float((z['undesired_contact'][live] > 0).mean())
                    r['log_std'] = z['log_std'].tolist()
                    rows.append(r)
            baseline = list(csv.DictReader((ROOT / f'artifacts/final_locomotion_eval_r1/episodes/seed{seed}_eval1001_episodes.csv').open()))
            ref = np.array([int(r['failure_step']) for r in baseline if r['preference'] == 'uniform' and float(r['command_vx']) == -.25])
            z = np.load(OUT / f'seed{seed}_nominal_eval_reset_forced_mean_0.npz')
            target = np.minimum(ref, 600)
            replay.append(dict(seed=seed, exact_lanes=int((z['failure_step'] == target).sum()),
                               max_step_difference=int(np.abs(z['failure_step'] - target).max())))
            all_rows = []
            for eval_seed in [1001, 1002, 1003]:
                all_rows += list(csv.DictReader((ROOT / f'artifacts/final_locomotion_eval_r1/episodes/seed{seed}_eval{eval_seed}_episodes.csv').open()))
            uniform = [r for r in all_rows if r['preference'] == 'uniform']
            frozen.append(dict(seed=seed, uniform_episodes=len(uniform),
                falls=sum(int(r['fell']) for r in uniform),
                settling_falls=sum(int(r['fell']) and int(r['failure_step']) <= 200 for r in uniform)))
        def convert(value):
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value))
        (OUT / 'summary.json').write_text(json.dumps(dict(results=rows, replay=replay,
            frozen_uniform=frozen), indent=2, default=convert) + '\n')
        lines = ['# R1 training-vs-evaluation mismatch audit', '',
            'Read-only checkpoint audit; no training, reward edits, coefficient tuning, or checkpoint sweep. '
            'Frozen R1 remains FAIL / INADEQUATE. This report supplements, and does not modify, the frozen report.', '',
            '## Diagnostic verdict', '',
            '**The evidence supports a seed-dependent sampling/command generalization gap. It does not justify '
            'attributing the collapse solely to alive-gate bootstrap starvation or opening R2 reward tuning now.**', '',
            '- **Seed1: strong dependence on stochastic execution under the audited eval conditions.** '
            'Mean policy: 0/64 survive 600 steps, mean failure 32.1 steps. Sampling: 19/64 and 24/64 survive '
            '600 steps with the same reset/command. This supports a practically unusable deterministic mean '
            'in this condition; it does not prove a dynamical attractor or usable 22-second locomotion.',
            '- **Seed0: command condition is the larger demonstrated factor.** Original schedule: mean and '
            'both stochastic repeats all fail by 600; sampling delays failure (RMST 69.6/74.4 vs 27.8 steps). '
            'Keeping eval initial states but using training-style commands yields 18/64 mean-policy survivors '
            'and 22/64–28/64 stochastic survivors at 600. Changing only the reset to native defaults gives '
            'similar results (20/64 and 24/64–31/64). Small pose jitter alone does not explain the collapse.',
            '- **Seed2: noise is not a universal remedy.** Original-schedule mean survives in 22/64 lanes '
            'versus 16/64 and 15/64 with sampling. With random commands, mean survival is 41/64 versus '
            '21/64 and 13/64. Making the very first observation consistent with zero command changes '
            'mean survival from 22/64 to 0/64 (mean failure 23.4 steps). This exposes severe initialization '
            'sensitivity, not grounds to silently change the frozen evaluator.',
            '- **Timing rules out the forced switch as the trigger for seed0/1 collapse.** All 960/960 '
            'uniform frozen episodes per seed fall before the 200-step settling period ends, when alive '
            'gate is fully open. The final command never reaches their first episode.',
            '- **Training-like probes do not reconstruct the reported training rollout.** Seed0 benefits '
            'from sampling with random commands (12/64 and 14/64 reach the 400-step timeout, versus 3/64 '
            'for mean). Seeds1/2 still mostly fail on fresh training terrain. This leaves simulator-state, '
            'preference-distribution, normalizer evolution, and pre-/post-final-update differences unresolved; '
            'it is not evidence that every training condition fails.', '',
            'The bounded audit contains 54 cells / 3,456 first-episode trajectories, at one reset seed and '
            'one fixed post-settling command. It is diagnostic evidence, not a new benchmark or a statistical '
            'claim over all commands/preferences. R2 remains unopened. Before proposing any revision, '
            'use these findings to specify whether the intended change addresses deterministic execution, '
            'per-episode zero-command coverage, or reward learning. Reward causality remains unproven.', '',
            '## Scope and controls', '',
            'Three final R1 checkpoints; eval reset seed 1001; 64 paired lanes; uniform preference; '
            '600-step (6 s) horizon; deterministic tanh(actor mean) and two independent Gaussian action RNG streams '
            '(7001/7002), using the training policy distribution and tanh transform. Sampling repeats share initial '
            'lanes and are not independent reset populations. Actor and observation-normalizer states were checked '
            'unchanged after every shard. No optimizer or reward-normalizer update is called.', '',
            'Nominal controls vary (a) original eval reset + original 200-step zero → fixed vx=-0.25 schedule, '
            '(b) same reset + training-style random command from reset, (c) native default reset + random command, '
            '(d) original eval conditions with command zero also inserted into the first observation. '
            'The original schedule retains its observation timing, including the one-step observation lag at the switch. '
            'All 50-step traces record raw actor mean, pre-tanh sampled action, executed action, log_std/std, '
            'alive mask, terminal-preserving height/tilt/contact, command and done.', '',
            'Training-like controls use the actual terrain generator, robot variants, domain randomization, '
            'native reset, and 400-step training timeout, with the global clock past initial stand phase. '
            'Terrain curriculum is held at its fresh level-0 distribution for pairing. These are 64 fresh lanes, '
            'not reconstruction of the 4096-lane training simulator at update 500; its state was not saved. '
            'Uniform preference and frozen normalizers isolate the requested factors and differ from ongoing training.', '',
            'The final-eval files do not contain saved full simulator initial states. The audit reconstructs the '
            'specified offset/yaw draws with the original reset function, and saves its own initial root/joint states. '
            'Paired initial states and commands agree within 1e-6. Frozen-reference first-cell failure times '
            '(capped at audit horizon) agree as follows:', '',
            '| seed | exact lanes / 64 | maximum step difference |', '|---|---:|---:|']
        lines += [f"| {r['seed']} | {r['exact_lanes']} | {r['max_step_difference']} |" for r in replay]
        lines += ['', 'Seed1 differs by one step in two lanes; exact bitwise simulator reproduction is not '
            'claimed. Seed0 and seed2 match all 64 capped lane outcomes. The audit disables torch.compile '
            'and uses eager versions of the same model operations; the source of the two one-step deviations '
            'was not isolated.', '', '## Results', '',
            'M = deterministic mean, S1/S2 = stochastic streams. RMST is mean min(first termination,600), '
            'not completed-episode mean. Training timeouts at 400 are successful horizon completions, not falls; '
            'therefore use reason counts and survival through 200 when comparing training-like rows. '
            '“Alive600” only applies to nominal rows. None of these are final 22-second success scores.', '',
            '| seed | environment | reset / command | mode | RMST steps | alive >200 | alive600 | termination reasons |',
            '|---|---|---|---|---:|---:|---:|---|']
        for r in rows:
            mode = 'M' if r['mode'] == 'mean' else ('S1' if r['noise_seed'] == 7001 else 'S2')
            reasons = ', '.join(f'{k.removeprefix("term_")}={v}' for k, v in r['termination_counts'].items())
            lines.append(f"| {r['training_seed']} | {r['environment']} | {r['reset']} / {r['schedule']} | {mode} | {r['restricted_mean_steps']:.1f} | {r['survive_200']}/64 | {r['survive_600'] if r['environment']=='nominal' else 'n/a'} | {reasons} |")
        lines += ['', '## First 50 steps under original eval conditions', '',
            'Metrics include only first-episode live steps, including terminal frames. Saturation is |action| ≥ 0.95×3.0. '
            'Raw Gaussian std is pre-tanh, not executed-action standard deviation. Trace arrays retain all joints/lanes; '
            'mask with alive to exclude automatic reset trajectories.', '',
            '| seed | mode | raw std | saturation | mean height (m) | mean max abs tilt (rad) | undesired contact fraction |',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['environment'] == 'nominal' and r['reset'] == 'eval_reset' and r['schedule'] == 'forced':
                lines.append(f"| {r['training_seed']} | {r['mode']} {r['noise_seed']} | {r['std_min']:.3f}–{r['std_max']:.3f} | {r['first50_saturation_fraction']:.3f} | {r['first50_mean_height']:.3f} | {r['first50_mean_max_abs_tilt']:.3f} | {r['first50_undesired_contact_fraction']:.3f} |")
        lines += ['', '## Frozen evidence: failure timing', '',
            '| seed | uniform episodes | falls | falls during settling (≤200) |', '|---|---:|---:|---:|']
        lines += [f"| {r['seed']} | {r['uniform_episodes']} | {r['falls']} | {r['settling_falls']} |" for r in frozen]
        lines += ['', '## Source-backed corrections to the earlier diagnosis', '',
            '- `MOPPOTrainer.update()` reports `_lane_step_count.mean()` after resetting terminated lanes to zero. '
            'This is current episode age, not mean completed episode length. TensorBoard update-500 values are '
            '134.0315 / 166.5237 / 166.7412 for seeds 0/1/2. They describe rollouts collected before the final PPO update, '
            'whereas checkpoint evaluation uses the policy after that update. A direct ratio against eval failure time is invalid.',
            '- Training stand phase is a global startup gate, checked only at resets: '
            '`common_step_counter * step_dt < stand_phase_s`. It is not 2 s of standing every episode. '
            'After startup, commands are drawn once per reset: vx∈[-0.3,1.0], vy∈[-0.3,0.3], wz∈[-0.5,0.5]. '
            'Exact all-zero command has probability zero under these continuous draws. Eval requires it for every settling period.',
            '- Native training reset returns default root/joint state at terrain origins; eval adds xy/yaw jitter '
            '(±0.05 m / ±0.05 rad), forces zero root velocity, and uses nominal joints. Training also varies terrain, '
            'leg geometry, payload 0–5 kg, COM, friction, gains and joint limits; eval nominalizes these and disables pushes. '
            'Thus fixed reset seeds are not the only distribution difference.',
            '- At cx=0, `signed_engagement` is 1 and alive gate is fully open. Early settling falls therefore cannot '
            'be caused by the alive reward being gated off on those eval steps. Reward is not used to choose inference actions; '
            'an indirect effect of R1 on learned weights remains possible. This audit does not establish or refute a training '
            'bootstrap failure causally.', '',
            'Sources: `scripts/rl/core/algorithms/moppo.py`, `scripts/rl/core/modules/actor_critic.py`, '
            '`scripts/rl/experiments/common/utilities/final_locomotion_eval.py`, `talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py`, '
            '`talon_rl/tasks/locomotion/a1_env/mdp/events.py`, `talon_rl/reward.py`.', '']
        (OUT / 'audit-report.md').write_text('\n'.join(lines))
        inputs = [*ROOT.glob('runs/phase1_r1_dt01_seed*_2026-09-20/checkpoints/checkpoint.pt'),
                  *ROOT.glob('artifacts/final_locomotion_eval_r1/**/*')]
        inputs += [ROOT / s for s in [
            'scripts/rl/experiments/baselines/reward_revision/workflow.py', 'scripts/rl/experiments/baselines/reward_revision/workflow.py',
            'scripts/rl/experiments/baselines/reward_revision/workflow.py', 'scripts/rl/experiments/common/utilities/final_locomotion_eval.py',
            'scripts/rl/core/algorithms/moppo.py', 'scripts/rl/core/modules/actor_critic.py',
            'talon_rl/reward.py', 'talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py',
            'talon_rl/tasks/locomotion/a1_env/mdp/events.py']]
        manifest = {str(p.relative_to(ROOT)): sha(p) for p in inputs if p.is_file()}
        manifest.update({str(p.relative_to(ROOT)): sha(p) for p in OUT.glob('*.npz')})
        (OUT / 'sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    
    
    if True:
        main()

STAGES = {
    "r1_directed_progress_smoke": run_r1_directed_progress_smoke,
    "r1_mismatch_audit": run_r1_mismatch_audit,
    "r1_offline_audit": run_r1_offline_audit,
    "r1_readonly_replay": run_r1_readonly_replay,
    "r1_replay_validate": run_r1_replay_validate,
    "run_r1_mismatch_audit": run_run_r1_mismatch_audit,
    "run_r1_replay_shards": run_run_r1_replay_shards,
    "run_r1_retrain_shards": run_run_r1_retrain_shards,
    "run_r1_single_cell": run_run_r1_single_cell,
    "summarize_r1_mismatch_audit": run_summarize_r1_mismatch_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
