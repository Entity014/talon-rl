"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_b0_env_smoke():
    """Run former b0_env_smoke.py stage."""
    """No-training smoke for the B0 scalar wrapper."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from isaaclab.app import AppLauncher
    
    app = AppLauncher({"headless": True, "enable_cameras": False}).app
    import gymnasium as gym
    import talon_rl.tasks.locomotion.a1_env  # registers env
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
    from rl.experiments.common.utilities.final_locomotion_eval import _nominalize_cfg
    from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
    
    cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=4; cfg.seed=0; cfg.sim.dt=.01; cfg.decimation=1; _nominalize_cfg(cfg)
    base=gym.make("Isaac-Talon-A1-v0",cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base)
    x=env.reset(); assert x["reward"].shape==(4,) and np.isfinite(x["reward"]).all(); assert np.allclose(base.v_command_buf.cpu().numpy(), .5*np.array([[1,0,0]]))
    rows=[]
    for step in range(8):
        x=env.step(np.zeros((4,base.action_dim),np.float32)); f=x["fields"]
        assert x["reward"].shape==(4,) and np.isfinite(x["reward"]).all() and np.allclose(x["command"],(.5,0,0))
        rows.append({"step":step,"vx":f["v_actual"][:,0].tolist(),"roll":f["roll_pitch"][:,0].tolist(),"pitch":f["roll_pitch"][:,1].tolist(),"height":f["height"].tolist(),"reward":x["reward"].tolist(),"done":x["done"].tolist(),"command":x["command"].tolist()})
    out=Path("artifacts/b0_smoke/b0-env-smoke.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"pass":True,"z_nominal":env.z_nominal,"rows":rows},indent=2)); print(out)
    base.close(); app.close()

def run_b0_ppo_smoke():
    """Run former b0_ppo_smoke.py stage."""
    """End-to-end B0 PPO smoke: real B0Env, scalar rollout, and fixed-std PPO."""
    
    import json
    import os
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    
    ROOT = Path(__file__).resolve().parents[4]
    OUT = ROOT / "artifacts" / "b0_smoke"
    STARTED = OUT / "b0-ppo-smoke_RUN_STARTED.json"
    ERROR = OUT / "b0-ppo-smoke_ERROR.json"
    COMPLETE = OUT / "b0-ppo-smoke_RUN_DONE.json"
    ARTIFACT = OUT / "b0-ppo-smoke.json"
    CHECKPOINT = OUT / "b0-ppo-smoke.pt"
    UPDATES, HORIZON, LANES = 2, 16, 4
    
    
    def _write(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    
    
    def _nominalize_cfg(cfg) -> None:
        from talon_rl.assets.unitree_a1.a1 import TALON_A1_CFG
        cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
        cfg.scene.robot = TALON_A1_CFG.replace()
        cfg.events.randomize_payload_mass.params["mass_distribution_params"] = (0.0, 0.0)
        cfg.events.randomize_payload_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        cfg.events.randomize_friction.params.update(static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0), restitution_range=(0.0, 0.0))
        cfg.events.randomize_motor_power.params.update(stiffness_distribution_params=(1.0, 1.0), damping_distribution_params=(1.0, 1.0))
        cfg.events.randomize_joint_range.params["scale_range"] = (1.0, 1.0)
        cfg.events.push_robot = None; cfg.curriculum.terrain_levels = None
        cfg.terminations.obstacle_reached = None; cfg.episode_length_s = 22.0; cfg.stand_phase_s = 0.0
    
    
    def _finite(*arrays) -> bool:
        return all(np.isfinite(np.asarray(value)).all() for value in arrays)
    
    
    def main() -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        for path in (ERROR, COMPLETE, ARTIFACT, CHECKPOINT): path.unlink(missing_ok=True)
        started = {"status": "RUN_STARTED", "pid": os.getpid(), "started_unix": time.time(), "updates": UPDATES, "horizon": HORIZON, "lanes": LANES}
        _write(STARTED, started)
        app = base = None
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym
            import talon_rl.tasks.locomotion.a1_env  # register after SimulationApp starts
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.core.algorithms.scalar_ppo import B0PPOConfig, B0PPOTrainer, ScalarRolloutBuffer, scalar_gae
            from rl.core.modules.actor_critic import ActorCritic
    
            torch.manual_seed(0); np.random.seed(0)
            cfg = IsaacLabTalonEnvCfg(); cfg.scene.num_envs = LANES; cfg.seed = 0; cfg.sim.dt = 0.01; cfg.decimation = 1
            _nominalize_cfg(cfg)
            base = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
            env = B0TalonEnv(base)
            transition = env.reset(); obs = transition["obs"]
            assert obs.shape == (LANES, base.obs_dim)
            ppo_cfg = B0PPOConfig(std_hold_updates=1, std_decay_updates=3)
            model = ActorCritic(base.obs_dim, base.obs_dim, base.action_dim, 1, [64, 64]).to(base.device)
            trainer = B0PPOTrainer(model, ppo_cfg)
            log_std_before = model.log_std.detach().clone()
            summaries = []
            for update in range(UPDATES):
                std = trainer.begin_update(); assert trainer._std_for_update == std
                buffer = ScalarRolloutBuffer(HORIZON, LANES)
                per_step_std = []
                for _step in range(HORIZON):
                    actor_obs = torch.as_tensor(obs, device=base.device, dtype=torch.float32)
                    with torch.no_grad():
                        action, logp_old = model.act(actor_obs)
                        value = model.value(actor_obs).squeeze(-1)
                        # Same sampled action under the same scheduled std must
                        # reproduce logp_old before any optimizer step.
                        assert torch.allclose(model.logp(actor_obs, action), logp_old, atol=2e-5)
                        per_step_std.append(float(model._pre_tanh_dist(actor_obs).scale.mean().item()))
                    transition = env.step(action.cpu().numpy().astype(np.float32))
                    reward, done = transition["reward"], transition["done"]
                    assert _finite(reward, done, value.cpu().numpy(), logp_old.cpu().numpy())
                    buffer.append(obs, action.cpu().numpy(), logp_old.cpu().numpy(), reward, done, value.cpu().numpy())
                    obs = transition["obs"]
                final_obs = torch.as_tensor(obs, device=base.device, dtype=torch.float32)
                with torch.no_grad(): final_value = model.value(final_obs).squeeze(-1).cpu().numpy()
                buffer.finish(final_value); arrays = buffer.arrays()
                assert arrays["obs"].shape == (HORIZON, LANES, base.obs_dim)
                assert arrays["actions"].shape == (HORIZON, LANES, base.action_dim)
                assert arrays["rewards"].shape == arrays["dones"].shape == arrays["values"].shape == (HORIZON, LANES)
                assert _finite(*arrays.values())
                values = np.concatenate([arrays["values"], arrays["final_value"][None]], axis=0)
                advantages = scalar_gae(arrays["rewards"], values, arrays["dones"], ppo_cfg.gamma, ppo_cfg.gae_lambda)
                returns = advantages + arrays["values"]
                assert _finite(advantages, returns)
                flat = buffer.flatten(); batch = HORIZON * LANES
                actor_obs = torch.as_tensor(flat["obs"], device=base.device, dtype=torch.float32)
                actions = torch.as_tensor(flat["actions"], device=base.device, dtype=torch.float32)
                old_logp = torch.as_tensor(flat["logp_old"], device=base.device, dtype=torch.float32)
                adv = torch.as_tensor(advantages.reshape(batch), device=base.device)
                ret = torch.as_tensor(returns.reshape(batch), device=base.device)
                policy_loss, value_loss, entropy, ratio, clip_fraction = trainer.ppo_loss(actor_obs, actor_obs, actions, old_logp, adv, ret)
                assert torch.allclose(ratio, torch.ones_like(ratio), atol=2e-5)
                before = [param.detach().clone() for param in model.parameters() if param is not model.log_std]
                trainer.optim.zero_grad(); (policy_loss + 0.5 * value_loss - 0.001 * entropy).backward(); trainer.optim.step()
                after = [param.detach() for param in model.parameters() if param is not model.log_std]
                assert any(not torch.equal(a, b) for a, b in zip(before, after)), "optimizer did not change actor/critic parameters"
                assert torch.equal(model.log_std.detach(), log_std_before), "fixed std must not update log_std"
                assert np.allclose(per_step_std, std)
                trainer.finish_update()
                summaries.append({"update": update, "std": std, "ratio_mean_before_step": float(ratio.mean()),
                                  "clip_fraction_before_step": float(clip_fraction), "policy_loss": float(policy_loss),
                                  "value_loss": float(value_loss), "rollout_shapes": {key: list(value.shape) for key, value in arrays.items()}})
    
            trainer.save(CHECKPOINT)
            restored = B0PPOTrainer(ActorCritic(base.obs_dim, base.obs_dim, base.action_dim, 1, [64, 64]).to(base.device), ppo_cfg)
            restored.load(CHECKPOINT)
            expected_next_std = ppo_cfg.scheduled_std(UPDATES)
            assert restored.update_idx == UPDATES and np.isclose(restored._std_for_update, expected_next_std)
            assert restored.model.exploration_mode == "scheduled_fixed_std"
            assert not hasattr(trainer, "preference") and not hasattr(trainer, "moppo")
            artifact = {"pass": True, "checks": {
                "scalar_rollout_shapes": True, "finite_reward_done_value_logp": True,
                "logp_old_matches_sampling_scheduled_std": True, "ratio_one_before_first_optimizer_step": True,
                "scalar_gae_and_returns": True, "optimizer_changes_actor_or_critic": True,
                "log_std_unchanged": True, "std_constant_within_update": True,
                "checkpoint_save_load": True, "update_index_and_std_resume": True,
                "no_preference_or_moppo_state": True}, "updates": summaries,
                "resume": {"update_idx": restored.update_idx, "scheduled_std": expected_next_std, "checkpoint": str(CHECKPOINT)}}
            _write(ARTIFACT, artifact)
            _write(COMPLETE, dict(started, status="RUN_DONE", ended_unix=time.time(), exit_code=0, artifact=str(ARTIFACT), complete=True))
            print(ARTIFACT)
        except BaseException as exc:
            _write(ERROR, dict(started, status="ERROR", ended_unix=time.time(), error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
            raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    
    
    if True: main()

def run_b0_terminal_smoke():
    """Run former b0_terminal_smoke.py stage."""
    """Terminal-lifecycle smoke for the B0 scalar environment wrapper."""
    
    import json
    import os
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    OUT = ROOT / "artifacts" / "b0_smoke"
    STARTED = OUT / "b0-terminal-smoke_RUN_STARTED.json"
    ERROR = OUT / "b0-terminal-smoke_ERROR.json"
    COMPLETE = OUT / "b0-terminal-smoke_RUN_DONE.json"
    ARTIFACT = OUT / "b0-terminal-smoke.json"
    
    
    def _write(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    
    
    def _row(step: int, lane: int, transition: dict) -> dict:
        return {"step": step, "lane": lane, "reward": float(transition["reward"][lane]),
                "done": bool(transition["done"][lane]),
                "terminal_fall": bool(transition["terminal_fall"][lane]),
                "term_base_contact": bool(transition["term_base_contact"][lane]),
                "height": float(transition["fields"]["height"][lane]),
                "vx": float(transition["fields"]["v_actual"][lane, 0])}
    
    
    def _nominalize_cfg(cfg) -> None:
        """Keep this no-training harness independent of PPO evaluation imports."""
        from talon_rl.assets.unitree_a1.a1 import TALON_A1_CFG
    
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.robot = TALON_A1_CFG.replace()
        cfg.events.randomize_payload_mass.params["mass_distribution_params"] = (0.0, 0.0)
        cfg.events.randomize_payload_com.params["com_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        cfg.events.randomize_friction.params.update(
            static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0), restitution_range=(0.0, 0.0))
        cfg.events.randomize_motor_power.params.update(
            stiffness_distribution_params=(1.0, 1.0), damping_distribution_params=(1.0, 1.0))
        cfg.events.randomize_joint_range.params["scale_range"] = (1.0, 1.0)
        cfg.events.push_robot = None
        cfg.curriculum.terrain_levels = None
        cfg.terminations.obstacle_reached = None
        cfg.episode_length_s = 22.0
        cfg.stand_phase_s = 0.0
    
    
    def _induce_base_contact(base) -> None:
        """Place the trunk into the plane to exercise the real contact terminal.
    
        Passive zero actions leave this nominal robot standing indefinitely, so a
        fall cannot be a deterministic smoke prerequisite.  This changes only
        the initial physics state; termination still comes from the configured
        contact sensor on the next simulated transition.
        """
        robot = base.scene["robot"]
        state = robot.data.default_root_state.clone()
        state[:, :3] += base.scene.env_origins
        state[:, 2] = base.scene.env_origins[:, 2] + 0.05
        state[:, 7:] = 0.0
        robot.write_root_pose_to_sim(state[:, :7])
        robot.write_root_velocity_to_sim(state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        base.scene.write_data_to_sim()
        base.sim.forward()
    
    
    def main() -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        for path in (ERROR, COMPLETE, ARTIFACT):
            path.unlink(missing_ok=True)
        started = {"status": "RUN_STARTED", "pid": os.getpid(), "started_unix": time.time()}
        _write(STARTED, started)
        app = base = None
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym
            import talon_rl.tasks.locomotion.a1_env  # registers the environment
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
    
            cfg = IsaacLabTalonEnvCfg(); cfg.scene.num_envs = 4; cfg.seed = 0
            cfg.sim.dt = 0.01; cfg.decimation = 1; _nominalize_cfg(cfg)
            base = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
            env = B0TalonEnv(base); env.reset(); _induce_base_contact(base)
            action = np.zeros((env.num_envs, base.action_dim), dtype=np.float32)
            terminal = None
            for step in range(100):
                transition = env.step(action)
                fall = transition["term_base_contact"] & transition["done"]
                # Only a base-contact done is a B0 fall, and only those receive
                # the terminal reward below -9.
                assert np.array_equal(transition["terminal_fall"], fall)
                assert np.all(transition["reward"][~fall] >= -9.0)
                if fall.any():
                    lane = int(np.flatnonzero(fall)[0]); terminal = _row(step, lane, transition)
                    assert terminal["reward"] < -9.0
                    assert terminal["done"] and terminal["terminal_fall"] and terminal["term_base_contact"]
                    break
            assert terminal is not None, "no terminal base-contact event observed"
    
            # SAME_STEP autoreset has already occurred. The next non-fall action
            # for this lane must not inherit the preceding fall charge.
            after_reset = None
            for step in range(terminal["step"] + 1, terminal["step"] + 101):
                transition = env.step(action); lane = terminal["lane"]
                if not transition["terminal_fall"][lane]:
                    after_reset = _row(step, lane, transition)
                    assert after_reset["reward"] >= -9.0
                    break
            assert after_reset is not None, "no non-fall transition after autoreset"
            _write(ARTIFACT, {"pass": True, "checks": {
                "base_contact_termination_observed": True,
                "terminal_reward_lt_minus_9_only_on_fall": True,
                "post_reset_transition_has_no_carried_fall_penalty": True,
                "done_terminal_flags_aligned": True},
                "terminal_transition": terminal, "post_reset_transition": after_reset})
            _write(COMPLETE, dict(started, status="RUN_DONE", ended_unix=time.time(), exit_code=0,
                                  artifact=str(ARTIFACT), complete=True))
            print(ARTIFACT)
        except BaseException as exc:
            _write(ERROR, dict(started, status="ERROR", ended_unix=time.time(),
                               error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
            raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    
    
    if True:
        main()

STAGES = {
    "b0_env_smoke": run_b0_env_smoke,
    "b0_ppo_smoke": run_b0_ppo_smoke,
    "b0_terminal_smoke": run_b0_terminal_smoke,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
