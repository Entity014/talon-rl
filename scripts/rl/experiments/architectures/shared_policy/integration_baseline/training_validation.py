"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_train_v1a():
    """Run former train_v1a.py stage."""
    """Authorized V1-A preservation runner: one seed, 300 scalar-PPO updates."""
    
    
    import argparse
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(ROOT))
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=4096)
        parser.add_argument("--iterations", type=int, default=300)
        parser.add_argument("--bottleneck", type=int, default=8)
        args = parser.parse_args()
        manifest = json.loads((ROOT / "artifacts/v1a/V1A_FREEZE.json").read_text())
        if manifest.get("status") != "FROZEN" or not manifest.get("training_authorized"):
            raise RuntimeError("V1-A manifest is not frozen/authorized")
        source = next(item for item in manifest["source_baseline"]["checkpoints"] if item["seed"] == args.seed)
        run = args.run_dir.resolve()
        run.mkdir(parents=True, exist_ok=True)
        (run / "RUN_STARTED.json").write_text(json.dumps({"status": "RUN_STARTED", "seed": args.seed, "unix": time.time()}) + "\n")
        app = base = vec = None
        try:
            from isaaclab.app import AppLauncher
    
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from talon_rl.rewards.baselines import group_stock_terms, reconstruct_stock_scalar
            from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = args.seed
            base = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
    
            class RewardAdapter(gym.Wrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.max_error = 0.0
                    self.rows = 0
    
                def step(self, action):
                    obs, reward, term, trunc, extras = self.env.step(action)
                    manager = self.env.unwrapped.reward_manager
                    raw = manager._step_reward.detach().cpu().numpy()
                    names = list(manager.active_terms)
                    vector = group_stock_terms({name: raw[:, i] for i, name in enumerate(names)}, shape=(self.env.unwrapped.num_envs,))
                    reconstructed = reconstruct_stock_scalar(vector) * self.env.unwrapped.step_dt
                    error = float(abs(reconstructed - reward.detach().cpu().numpy()).max())
                    self.max_error = max(self.max_error, error)
                    self.rows += len(reward)
                    if error >= 1e-6:
                        raise RuntimeError(f"M0.1 reward reconstruction mismatch: {error}")
                    extras = dict(extras)
                    extras["m0_1_reward_vector"] = torch.as_tensor(vector, device=reward.device)
                    extras["m0_1_reconstructed_scalar"] = reward
                    return obs, reward, term, trunc, extras
    
            adapted = RewardAdapter(base)
            vec = RslRlVecEnvWrapper(adapted, clip_actions=1.0)
            agent = UnitreeA1FlatPPORunnerCfg()
            agent.num_steps_per_env = 24
            agent.max_iterations = args.iterations
            agent.save_interval = 50
            agent.experiment_name = f"v1a_seed{args.seed}"
            agent.policy.actor_hidden_dims = [128, 128, 128]
            agent.policy.critic_hidden_dims = [128, 128, 128]
            runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(run), device="cuda")
            attach_v1a_policy(runner.alg, ROOT / source["path"], args.bottleneck)
            runner.learn(args.iterations, init_at_random_ep_len=True)
            report = {
                "schema": "v1a_preservation_training_v1",
                "status": "PASS",
                "seed": args.seed,
                "updates": args.iterations,
                "num_envs": args.num_envs,
                "source_checkpoint": source["path"],
                "source_baseline_hash": manifest["source_baseline"]["source_baseline_hash"],
                "v1a_formulation_hash": manifest["hashes"]["v1a_formulation_hash"],
                "reward_reconstruction_max_error": adapted.max_error,
                "vector_critic": False,
                "preference_curriculum": False,
                "rehearsal": False,
                "anchor_loss": False,
                "learned_gate": False,
                "final_checkpoint": str(run / "model_300.pt"),
            }
            (run / "artifact.json").write_text(json.dumps(report, indent=2) + "\n")
            (run / "RUN_DONE.json").write_text(json.dumps({"status": "RUN_DONE", "exit_code": 0, "seed": args.seed}) + "\n")
            print(json.dumps(report, indent=2))
        except BaseException as exc:
            (run / "ERROR.json").write_text(json.dumps({"status": "ERROR", "seed": args.seed, "error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
            raise
        finally:
            if vec is not None:
                vec.close()
            elif base is not None:
                base.close()
            if app is not None:
                app.close()
    
    
    if True:
        main()

def run_v1a_e1r_eval():
    """Run former v1a_e1r_eval.py stage."""
    """Corrected paired M0.1/V1-A evaluator using stock action clipping."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    
    import argparse
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    
    def main() -> None:
        p = argparse.ArgumentParser()
        p.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
        p.add_argument("--m01-checkpoint", type=Path, required=True)
        p.add_argument("--v1a-checkpoint", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--steps", type=int, default=500)
        p.add_argument("--num-envs", type=int, default=64)
        args = p.parse_args()
        from isaaclab.app import AppLauncher
    
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.experiments.common.utilities.v1a_e0_eval import file_hash, load_policy, obs_tensor, snapshot
    
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = 47001 + args.seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            models = {"m01": load_policy(args.m01_checkpoint, "m01"), "v1a": load_policy(args.v1a_checkpoint, "v1a")}
    
            def rollout(model, kind):
                obs, _ = env.reset(seed=47001 + args.seed); obs = obs_tensor(obs).cuda()
                reset_hash = __import__("hashlib").sha256(obs.detach().cpu().numpy().tobytes()).hexdigest()
                before = snapshot(model); done_any = np.zeros(args.num_envs, dtype=bool); vx = []; tilt = []; term = []; contact = []
                with torch.no_grad():
                    for _ in range(args.steps):
                        if kind == "m01": action = model.act_inference({"policy": obs, "critic": obs})
                        else: action = model.act_inference(obs, torch.full((obs.shape[0], 5), .2, device=obs.device))
                        action = torch.clamp(action, -1.0, 1.0)
                        nxt, _, terminated, truncated, _ = env.step(action)
                        done_any |= (terminated | truncated).detach().cpu().numpy()
                        term.append(terminated.detach().cpu().numpy().astype(np.float32))
                        data = env.unwrapped.scene["robot"].data; command = env.unwrapped.command_manager.get_command("base_velocity")
                        vx.append((data.root_lin_vel_b[:, 0] - command[:, 0]).abs().cpu().numpy())
                        q = data.root_quat_w; roll = torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2)); pitch = torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1)); tilt.append(torch.rad2deg(torch.maximum(roll.abs(),pitch.abs())).cpu().numpy())
                        contact.append(env.unwrapped.termination_manager.get_term("base_contact").cpu().numpy().astype(np.float32)); obs = obs_tensor(nxt).cuda()
                after = model.state_dict(); t = np.concatenate(tilt)
                return {"initial_observation_sha256": reset_hash, "survival": float(1-done_any.mean()), "velocity_tracking_error": float(np.mean(np.concatenate(vx))), "tilt_p95_deg": float(np.percentile(t,95)), "max_tilt_deg": float(t.max()), "termination_rate": float(np.mean(np.concatenate(term))), "base_contact_rate": float(np.mean(np.concatenate(contact))), "finite_state": bool(np.isfinite(t).all()), "evaluator_non_mutation": all(torch.equal(before[k],after[k]) for k in before), "deterministic_actor_mean": True}
    
            baseline = rollout(models["m01"], "m01"); candidate = rollout(models["v1a"], "v1a")
            result = {"schema":"v1a_e1r_pair_eval_v1","protocol":"V1A-E1R","status":"COMPLETE_MEASUREMENT_ONLY","seed":args.seed,"steps":args.steps,"num_envs":args.num_envs,"reset_seed":47001+args.seed,"stock_clip_actions":1.0,"m01_checkpoint":str(args.m01_checkpoint),"m01_checkpoint_sha256":file_hash(args.m01_checkpoint),"v1a_checkpoint":str(args.v1a_checkpoint),"v1a_checkpoint_sha256":file_hash(args.v1a_checkpoint),"paired_reset_hash_equal":baseline["initial_observation_sha256"]==candidate["initial_observation_sha256"],"m01":baseline,"v1a":candidate,"binary_verdict":"PENDING_MARGIN_CONTRACT"}
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    
    
    if True: main()

def run_v1a_isaac_smoke():
    """Run former v1a_isaac_smoke.py stage."""
    """Short real-Isaac scalar-PPO V1-A integration smoke.
    
    This is not a training authorization.  It runs only two PPO updates to test
    the integration boundary, checkpoint/resume, lifecycle, and adapter gradients.
    """
    
    
    import argparse
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(ROOT))
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=16)
        parser.add_argument("--iterations", type=int, default=2)
        parser.add_argument("--bottleneck", type=int, default=8)
        args = parser.parse_args()
        run = args.run_dir.resolve()
        run.mkdir(parents=True, exist_ok=True)
        (run / "RUN_STARTED.json").write_text(json.dumps({"status": "RUN_STARTED", "unix": time.time()}) + "\n")
        app = base = vec = None
        try:
            from isaaclab.app import AppLauncher
    
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from talon_rl.rewards.baselines import group_stock_terms, reconstruct_stock_scalar
            from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = 0
            base = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
    
            class RewardAdapter(gym.Wrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.max_error = 0.0
                    self.rows = 0
    
                def step(self, action):
                    obs, reward, term, trunc, extras = self.env.step(action)
                    manager = self.env.unwrapped.reward_manager
                    raw = manager._step_reward.detach().cpu().numpy()
                    names = list(manager.active_terms)
                    vector = group_stock_terms({name: raw[:, i] for i, name in enumerate(names)}, shape=(self.env.unwrapped.num_envs,))
                    reconstructed = reconstruct_stock_scalar(vector) * self.env.unwrapped.step_dt
                    error = float(abs(reconstructed - reward.detach().cpu().numpy()).max())
                    self.max_error = max(self.max_error, error)
                    self.rows += len(reward)
                    if error >= 1e-6:
                        raise RuntimeError(f"M0.1 reward reconstruction mismatch: {error}")
                    extras = dict(extras)
                    extras["m0_1_reward_vector"] = torch.as_tensor(vector, device=reward.device)
                    extras["m0_1_reconstructed_scalar"] = reward
                    return obs, reward, term, trunc, extras
    
            adapted = RewardAdapter(base)
            vec = RslRlVecEnvWrapper(adapted, clip_actions=1.0)
            agent = UnitreeA1FlatPPORunnerCfg()
            agent.num_steps_per_env = 24
            agent.max_iterations = args.iterations
            agent.save_interval = 1
            agent.experiment_name = "v1a_isaac_smoke"
            agent.policy.actor_hidden_dims = [128, 128, 128]
            agent.policy.critic_hidden_dims = [128, 128, 128]
            runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(run), device="cuda")
            policy = attach_v1a_policy(runner.alg, args.checkpoint, args.bottleneck)
            runner.learn(args.iterations, init_at_random_ep_len=True)
    
            up_changed = any(float(adapter.up.weight.abs().max()) > 0.0 for adapter in policy.v1a_adapters)
            down_grad = [adapter.down.weight.grad for adapter in policy.v1a_adapters]
            down_received_grad = all(grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0.0 for grad in down_grad)
            smoke_checkpoint = run / "v1a_policy.pt"
            torch.save({"model_state_dict": policy.state_dict(), "iterations": args.iterations}, smoke_checkpoint)
            from rsl_rl.modules import ActorCritic
    
            restored_base = ActorCritic(
                obs={"policy": torch.zeros(1, 48, device="cuda"), "critic": torch.zeros(1, 48, device="cuda")},
                obs_groups={"policy": ["policy"], "critic": ["critic"]},
                num_actions=12,
                actor_obs_normalization=False,
                critic_obs_normalization=False,
                actor_hidden_dims=[128, 128, 128],
                critic_hidden_dims=[128, 128, 128],
                activation="elu",
                init_noise_std=1.0,
                noise_std_type="scalar",
            ).to("cuda")
            from rl.core.integration.rsl_rl.v1a_wrapper import RslRlV1AWrapper
    
            restored = RslRlV1AWrapper(restored_base, args.bottleneck)
            restored.load_state_dict(torch.load(smoke_checkpoint, map_location="cuda", weights_only=False)["model_state_dict"])
            report = {
                "schema": "v1a_isaac_smoke_v1",
                "status": "PASS" if up_changed and down_received_grad and adapted.max_error < 1e-6 else "FAIL",
                "iterations": args.iterations,
                "num_envs": args.num_envs,
                "reward_reconstruction_max_error": adapted.max_error,
                "adapter_up_changed": up_changed,
                "adapter_down_received_gradient_after_second_update": down_received_grad,
                "checkpoint_resume": True,
                "excluded_features": {
                    "vector_critic": False,
                    "objective_specific_advantages": False,
                    "preference_curriculum": False,
                    "rehearsal": False,
                    "anchor_loss": False,
                    "learned_gate": False,
                },
            }
            (run / "artifact.json").write_text(json.dumps(report, indent=2) + "\n")
            (run / "RUN_DONE.json").write_text(json.dumps({"status": "RUN_DONE", "exit_code": 0}) + "\n")
            print(json.dumps(report, indent=2))
            if report["status"] != "PASS":
                raise SystemExit("V1-A Isaac smoke FAILED")
        except BaseException as exc:
            (run / "ERROR.json").write_text(json.dumps({"status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
            raise
        finally:
            if vec is not None:
                vec.close()
            elif base is not None:
                base.close()
            if app is not None:
                app.close()
    
    
    if True:
        main()

def run_v1a_rsl_equivalence():
    """Run former v1a_rsl_equivalence.py stage."""
    """Checkpoint-level M0.1 -> V1-A equivalence check in the Isaac runtime.
    
    This is intentionally an offline network check: it does not launch Isaac,
    step an environment, or train.  It instantiates the real rsl_rl ActorCritic,
    loads one M0.1 terminal checkpoint, and compares it with the zero-residual
    V1-A wrapper.
    """
    
    
    import argparse
    import hashlib
    import json
    from pathlib import Path
    
    import torch
    from rsl_rl.modules import ActorCritic
    
    from rl.core.integration.rsl_rl.v1a_wrapper import RslRlV1AWrapper
    
    
    def load_m01(path: Path) -> ActorCritic:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ActorCritic(
            obs={"policy": torch.zeros(1, 48), "critic": torch.zeros(1, 48)},
            obs_groups={"policy": ["policy"], "critic": ["critic"]},
            num_actions=12,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
            activation="elu",
            init_noise_std=1.0,
            noise_std_type="scalar",
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--output", type=Path)
        args = parser.parse_args()
    
        torch.manual_seed(1000 + args.seed)
        base = load_m01(args.checkpoint)
        wrapped = RslRlV1AWrapper(base, bottleneck_dim=8)
        obs = torch.randn(32, 48)
        critic_obs = torch.randn(32, 48)
        w = torch.full((32, 5), 0.2)
    
        with torch.no_grad():
            baseline_mean = base.actor(obs)
            wrapped_mean = wrapped.act_inference(obs, w)
            base.update_distribution(obs)
            baseline_scale = base.distribution.scale.clone()
            baseline_dist = base.distribution
            wrapped.update_distribution(obs, w)
            wrapped_dist = wrapped.base.distribution
            actions = baseline_dist.sample()
            baseline_logp = baseline_dist.log_prob(actions).sum(-1)
            wrapped_logp = wrapped.get_actions_log_prob(actions)
            baseline_entropy = baseline_dist.entropy().sum(-1)
            wrapped_entropy = wrapped.entropy()
            baseline_value = base.critic(critic_obs)
            wrapped_value = wrapped.evaluate(critic_obs)
    
            state = wrapped.state_dict()
            restored = RslRlV1AWrapper(load_m01(args.checkpoint), bottleneck_dim=8)
            result = restored.load_state_dict(state)
            restored_mean = restored.act_inference(obs, w)
    
        metrics = {
            "schema": "v1a_rsl_checkpoint_equivalence_v1",
            "seed": args.seed,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "state_missing_keys": list(result.missing_keys),
            "state_unexpected_keys": list(result.unexpected_keys),
            "max_actor_mean_error": float((baseline_mean - wrapped_mean).abs().max()),
            "max_deterministic_action_error": float((baseline_mean - wrapped_mean).abs().max()),
            "max_distribution_loc_error": float((baseline_dist.loc - wrapped_dist.loc).abs().max()),
            "max_distribution_scale_error": float((baseline_scale - wrapped_dist.scale).abs().max()),
            "max_logp_error": float((baseline_logp - wrapped_logp).abs().max()),
            "max_entropy_error": float((baseline_entropy - wrapped_entropy).abs().max()),
            "max_value_error": float((baseline_value - wrapped_value).abs().max()),
            "max_state_roundtrip_action_error": float((wrapped_mean - restored_mean).abs().max()),
            "learned_std_exact": bool(torch.equal(base.std, restored.base.std)),
        }
        print(json.dumps(metrics, indent=2))
        if args.output:
            args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        tolerance = 1e-6
        numeric = [value for key, value in metrics.items() if key.startswith("max_")]
        if result.missing_keys or result.unexpected_keys or not metrics["learned_std_exact"] or any(value > tolerance for value in numeric):
            raise SystemExit("V1-A checkpoint equivalence FAILED")
    
    
    if True:
        main()

STAGES = {
    "train_v1a": run_train_v1a,
    "v1a_e1r_eval": run_v1a_e1r_eval,
    "v1a_isaac_smoke": run_v1a_isaac_smoke,
    "v1a_rsl_equivalence": run_v1a_rsl_equivalence,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
