"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_v1b_leave_one_out():
    """Run former v1b_leave_one_out.py stage."""
    """Diagnostic leave-one-adapter-out evaluation for a V1-B screen checkpoint."""
    
    import argparse
    import hashlib
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    OBJECTIVES = ("progress", "efficiency", "contact", "balance", "limits")
    S7_OBJECTIVES = ("progress", "balance", "efficiency")
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-checkpoint", type=Path, required=True)
        parser.add_argument("--screen-checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=64)
        parser.add_argument("--steps", type=int, default=200)
        parser.add_argument("--reset-seed", type=int, default=58001)
        parser.add_argument("--objective-set", choices=("v1b_legacy5", "v1b_s7_pass3"), default="v1b_legacy5")
        args = parser.parse_args()
    
        from isaaclab.app import AppLauncher
    
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
            from talon_rl.rewards.baselines import group_v1b_s7_terms, S7_OBJECTIVE_ORDER
            from talon_rl.rewards.baselines import group_stock_terms
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = args.reset_seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            agent = UnitreeA1FlatPPORunnerCfg()
            agent.policy.actor_hidden_dims = [128, 128, 128]
            agent.policy.critic_hidden_dims = [128, 128, 128]
            agent.num_steps_per_env = 24
            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(args.output.parent), device="cuda")
            objective_names = S7_OBJECTIVE_ORDER if args.objective_set == "v1b_s7_pass3" else OBJECTIVES
            objective_count = len(objective_names)
            group_terms = group_v1b_s7_terms if args.objective_set == "v1b_s7_pass3" else group_stock_terms
            actor = attach_v1a_policy(
                runner.alg,
                args.source_checkpoint,
                bottleneck_dim=8,
                w_ref=(1.0 / objective_count,) * objective_count,
                objectives=objective_names,
            )
            state = torch.load(args.screen_checkpoint, map_location="cuda", weights_only=False)
            actor.load_state_dict(state["actor"], strict=True)
            actor.eval()
            weights = torch.full((args.num_envs, objective_count), 1.0 / objective_count, device="cuda")
            screen_actor_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    
            def rollout(disabled):
                actor.load_state_dict(screen_actor_state, strict=True)
                before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
                if disabled is not None:
                    for parameter in actor.v1a_adapters[disabled].up.parameters():
                        parameter.data.zero_()
                obs, _ = env.reset(seed=args.reset_seed)
                obs = obs_tensor(obs).cuda()
                done_any = np.zeros(args.num_envs, dtype=bool)
                velocity_error, yaw_error, tilt, termination, contact, action_abs, action_rate = [], [], [], [], [], [], []
                objective_rewards, balance_metrics, efficiency_metrics = [], [], []
                previous_action = torch.zeros((args.num_envs, 12), device="cuda")
                with torch.no_grad():
                    for _ in range(args.steps):
                        action = actor.act_inference(obs, weights)
                        action = torch.clamp(action, -1.0, 1.0)
                        current_action_rate = (action - previous_action).square().mean(dim=-1)
                        next_obs, _, terminated, truncated, _ = env.step(action)
                        done_any |= (terminated | truncated).detach().cpu().numpy()
                        data = env.unwrapped.scene["robot"].data
                        command = env.unwrapped.command_manager.get_command("base_velocity")
                        velocity_error.append((data.root_lin_vel_b[:, 0] - command[:, 0]).abs().cpu().numpy())
                        yaw_error.append((data.root_ang_vel_b[:, 2] - command[:, 2]).abs().cpu().numpy())
                        quat = data.root_quat_w
                        roll = torch.atan2(2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1 - 2 * (quat[:, 1].square() + quat[:, 2].square()))
                        pitch = torch.asin(torch.clamp(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1, 1))
                        tilt.append(torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())).cpu().numpy())
                        termination.append(terminated.detach().cpu().numpy().astype(np.float32))
                        contact.append(env.unwrapped.termination_manager.get_term("base_contact").cpu().numpy().astype(np.float32))
                        action_abs.append(action.abs().cpu().numpy())
                        action_rate.append(current_action_rate.cpu().numpy())
                        raw = env.unwrapped.reward_manager._step_reward.detach().cpu().numpy()
                        names = list(env.unwrapped.reward_manager.active_terms)
                        grouped = group_terms({name: raw[:, i] for i, name in enumerate(names)}, shape=(args.num_envs,))
                        objective_rewards.append(grouped)
                        balance_metrics.append(np.stack([
                            data.root_lin_vel_b[:, 2].abs().cpu().numpy(),
                            torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1).cpu().numpy(),
                            torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())).cpu().numpy(),
                        ], axis=1))
                        efficiency_metrics.append(np.stack([
                            data.applied_torque.square().mean(dim=-1).cpu().numpy(),
                            current_action_rate.cpu().numpy(),
                        ], axis=1))
                        previous_action = action
                        obs = obs_tensor(next_obs).cuda()
                after = actor.state_dict()
                disabled_prefix = None if disabled is None else f"v1a_adapters.{disabled}."
                non_mutated = all(
                    torch.equal(before[name], after[name])
                    for name in before
                    if disabled_prefix is None or not name.startswith(disabled_prefix)
                )
                tilt_values = np.concatenate(tilt)
                action_values = np.concatenate(action_abs)
                objective_values = np.concatenate(objective_rewards)
                balance_values = np.concatenate(balance_metrics)
                efficiency_values = np.concatenate(efficiency_metrics)
                result = {
                    "disabled_adapter": None if disabled is None else objective_names[disabled],
                    "survival": float(1.0 - done_any.mean()),
                    "velocity_tracking_error": float(np.mean(np.concatenate(velocity_error))),
                    "yaw_tracking_error": float(np.mean(np.concatenate(yaw_error))),
                    "tilt_p95_deg": float(np.percentile(tilt_values, 95)),
                    "max_tilt_deg": float(tilt_values.max()),
                    "termination_rate": float(np.mean(np.concatenate(termination))),
                    "base_contact_rate": float(np.mean(np.concatenate(contact))),
                    "action_abs_mean": float(action_values.mean()),
                    "action_saturation_rate": float((action_values >= 0.999999).mean()),
                    "objective_reward_mean": objective_values.mean(axis=0).tolist(),
                    "balance_metric_mean": balance_values.mean(axis=0).tolist(),
                    "efficiency_metric_mean": efficiency_values.mean(axis=0).tolist(),
                    "finite_state": bool(np.isfinite(tilt_values).all()),
                    "evaluator_non_mutation": non_mutated,
                }
                actor.load_state_dict(screen_actor_state, strict=True)
                return result
    
            full = rollout(None)
            ablations = [rollout(index) for index in range(objective_count)]
            metrics = ("survival", "velocity_tracking_error", "yaw_tracking_error", "tilt_p95_deg", "max_tilt_deg", "termination_rate", "base_contact_rate", "action_abs_mean", "action_saturation_rate")
            effect_matrix = {}
            for metric in metrics:
                effect_matrix[metric] = {
                    row["disabled_adapter"]: row[metric] - full[metric] for row in ablations
                }
            result = {
                "schema": "v1b_leave_one_out_v1",
                "status": "DIAGNOSTIC_COMPLETE",
                "verdict": "SPECIALIZATION_NOT_DECIDED",
                "protocol": "stock_a1_clip_actions_1.0",
                "objective_set": args.objective_set,
                "objective_order": list(objective_names),
                "steps": args.steps,
                "num_envs": args.num_envs,
                "reset_seed": args.reset_seed,
                "source_checkpoint": str(args.source_checkpoint.resolve()),
                "source_checkpoint_sha256": sha256(args.source_checkpoint),
                "screen_checkpoint": str(args.screen_checkpoint.resolve()),
                "screen_checkpoint_sha256": sha256(args.screen_checkpoint),
                "full_policy": full,
                "ablations": ablations,
                "effect_matrix_delta_ablated_minus_full": effect_matrix,
                "objective_reward_effect_matrix": {
                    row["disabled_adapter"]: (np.asarray(row["objective_reward_mean"]) - np.asarray(full["objective_reward_mean"])).tolist()
                    for row in ablations
                },
                "balance_metric_effect_matrix": {
                    row["disabled_adapter"]: (np.asarray(row["balance_metric_mean"]) - np.asarray(full["balance_metric_mean"])).tolist()
                    for row in ablations
                },
                "efficiency_metric_effect_matrix": {
                    row["disabled_adapter"]: (np.asarray(row["efficiency_metric_mean"]) - np.asarray(full["efficiency_metric_mean"])).tolist()
                    for row in ablations
                },
                "interpretation_rule": "diagonal dominance is descriptive only; no V1-B verdict from this screen",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            if env is not None:
                env.close()
            app.close()
    
    
    if True:
        main()

def run_v1b_s3_alignment_audit():
    """Run former v1b_s3_alignment_audit.py stage."""
    """Read-only V1-B S3 audit: shared geometry, objective alignment, and ablation metrics."""
    
    import argparse
    import hashlib
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    OBJECTIVES = ("progress", "efficiency", "contact", "balance", "limits")
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def correlation_matrix(values: torch.Tensor) -> list[list[float]]:
        values = values.float()
        centered = values - values.mean(dim=0, keepdim=True)
        normalized = centered / centered.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)
        return (normalized.T @ normalized).cpu().tolist()
    
    
    def cross_correlation(left: torch.Tensor, right: torch.Tensor) -> list[list[float]]:
        left = left.float()
        right = right.float()
        left = (left - left.mean(dim=0, keepdim=True)) / left.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
        right = (right - right.mean(dim=0, keepdim=True)) / right.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
        return ((left.T @ right) / left.shape[0]).cpu().tolist()
    
    
    def norm(gradient_list) -> float:
        terms = [g.detach().square().sum() for g in gradient_list if g is not None]
        return float(torch.sqrt(torch.stack(terms).sum()).item()) if terms else 0.0
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-checkpoint", type=Path, required=True)
        parser.add_argument("--s2-checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=16)
        parser.add_argument("--horizon", type=int, default=24)
        parser.add_argument("--ablation-steps", type=int, default=200)
        parser.add_argument("--reset-seed", type=int, default=59001)
        args = parser.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "RUN_STARTED.json").write_text(json.dumps({"status": "RUN_STARTED", "protocol": "V1-B-S3"}) + "\n")
        def phase(name):
            (args.output.parent / "PHASE.txt").write_text(name + "\n")
            print(name, flush=True)
        phase("S3_RUN_STARTED")
    
        from isaaclab.app import AppLauncher
    
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        phase("S3_APP_INIT_OK")
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from talon_rl.rewards.baselines import group_v1b_s1_terms, reconstruct_v1b_s1_scalar
            from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
            from rl.core.objectives.routing import SharedObjectiveCritic, normalize_objective_advantages
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = args.reset_seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            phase("S3_ENV_CREATED")
            agent = UnitreeA1FlatPPORunnerCfg()
            agent.policy.actor_hidden_dims = [128, 128, 128]
            agent.policy.critic_hidden_dims = [128, 128, 128]
            agent.num_steps_per_env = args.horizon
            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(args.output.parent), device="cuda")
            phase("S3_RUNNER_CREATED")
            actor = attach_v1a_policy(runner.alg, args.source_checkpoint, bottleneck_dim=8, w_ref=(0.2,) * 5)
            phase("S3_ACTOR_LOADED")
            state = torch.load(args.s2_checkpoint, map_location="cuda", weights_only=False)
            actor.load_state_dict(state["actor"], strict=True)
            critic = SharedObjectiveCritic(48, 128, 5).cuda()
            critic.load_state_dict(state["critic"], strict=True)
            phase("S3_CHECKPOINT_LOADED")
            actor.eval(); critic.eval()
            weights = torch.full((args.num_envs, 5), 0.2, device="cuda")
    
            # One frozen rollout supplies the real PPO batch for gradient geometry.
            obs, _ = env.reset(seed=args.reset_seed)
            obs = obs_tensor(obs).cuda()
            observations, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], []
            raw_terms = []
            physical = []
            max_reconstruction_error = 0.0
            for _ in range(args.horizon):
                with torch.no_grad():
                    actor.update_distribution({"policy": obs, "critic": obs}, weights)
                    action = actor.base.distribution.sample()
                    old_log_prob = actor.get_actions_log_prob(action)
                    value = critic(obs)
                next_obs, reward_returned, terminated, truncated, _ = env.step(torch.clamp(action, -1.0, 1.0))
                manager = env.unwrapped.reward_manager
                raw = manager._step_reward.detach().cpu().numpy()
                names = list(manager.active_terms)
                term_dict = {name: raw[:, index] for index, name in enumerate(names)}
                vector = group_v1b_s1_terms(term_dict, shape=(args.num_envs,))
                max_reconstruction_error = max(
                    max_reconstruction_error,
                    float(np.abs(reconstruct_v1b_s1_scalar(vector) * env.unwrapped.step_dt - reward_returned.detach().cpu().numpy()).max()),
                )
                raw_terms.append(vector * env.unwrapped.step_dt)
                data = env.unwrapped.scene["robot"].data
                command = env.unwrapped.command_manager.get_command("base_velocity")
                quat = data.root_quat_w
                roll = torch.atan2(2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1 - 2 * (quat[:, 1].square() + quat[:, 2].square()))
                pitch = torch.asin(torch.clamp(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1, 1))
                physical.append(torch.stack([
                    (data.root_lin_vel_b[:, 0] - command[:, 0]).abs(),
                    data.root_lin_vel_b[:, 2].abs(),
                    torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1),
                    torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())),
                    action.abs().mean(dim=-1),
                ], dim=-1).detach().cpu())
                observations.append(obs); actions.append(action); old_log_probs.append(old_log_prob)
                rewards.append(torch.as_tensor(vector, device="cuda") * env.unwrapped.step_dt)
                values.append(value); dones.append((terminated | truncated).to("cuda"))
                obs = obs_tensor(next_obs).cuda()
            phase("S3_ROLLOUT_COMPLETE")
    
            reward_tensor = torch.stack(rewards)
            value_tensor = torch.stack(values)
            done_tensor = torch.stack(dones).float()
            returns = torch.zeros_like(reward_tensor)
            running = critic(obs).detach()
            for step in reversed(range(args.horizon)):
                running = reward_tensor[step] + 0.99 * running * (1.0 - done_tensor[step].unsqueeze(-1))
                returns[step] = running
            flat_obs = torch.cat(observations)
            flat_actions = torch.cat(actions)
            flat_old = torch.cat(old_log_probs)
            flat_returns = returns.reshape(-1, 5)
            flat_values = value_tensor.reshape(-1, 5)
            raw_advantages = (flat_returns - flat_values).detach()
            advantages = normalize_objective_advantages(raw_advantages)
            actor.update_distribution({"policy": flat_obs, "critic": flat_obs}, weights.repeat(args.horizon, 1))
            ratio = torch.exp(actor.get_actions_log_prob(flat_actions) - flat_old.detach())
            losses = []
            for index in range(5):
                losses.append(-torch.minimum(ratio * advantages[:, index], torch.clamp(ratio, .8, 1.2) * advantages[:, index]).mean())
    
            adapter_groups = [list(adapter.parameters()) for adapter in actor.v1a_adapters]
            shared_parameters = list(actor._shared_actor.parameters()) + list(actor._action_head.parameters())
            adapter_grads = []
            shared_grads = []
            for loss, group in zip(losses, adapter_groups):
                adapter_grads.append(torch.autograd.grad(loss, group, retain_graph=True, allow_unused=True))
                shared_grads.append(torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True))
            adapter_norms = [norm(group) for group in adapter_grads]
            shared_norms = [norm(group) for group in shared_grads]
            shared_vectors = [torch.cat([g.detach().flatten() for g in group if g is not None]) for group in shared_grads]
            shared_cosine = torch.stack([F.cosine_similarity(left, right, dim=0) for left in shared_vectors for right in shared_vectors]).reshape(5, 5).cpu().tolist()
    
            reward_samples = reward_tensor.reshape(-1, 5).detach().cpu()
            advantage_samples = raw_advantages.cpu()
            physical_samples = torch.cat(physical, dim=0)
            alignment = cross_correlation(reward_samples, physical_samples)
    
            # Deterministic, paired leave-one-out replay with raw reward means and
            # physical metrics so reward-to-metric mismatch is visible.
            del losses, adapter_grads, shared_grads, shared_vectors, ratio
            torch.cuda.empty_cache()
            screen_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    
            def evaluate(disabled):
                actor.load_state_dict(screen_state, strict=True)
                if disabled is not None:
                    for parameter in actor.v1a_adapters[disabled].up.parameters():
                        parameter.data.zero_()
                current, _ = env.reset(seed=args.reset_seed)
                current = obs_tensor(current).cuda()
                reward_rows, metric_rows = [], []
                for _ in range(args.ablation_steps):
                    with torch.no_grad():
                        action = torch.clamp(actor.act_inference(current, weights), -1.0, 1.0)
                    nxt, _, _, _, _ = env.step(action)
                    manager = env.unwrapped.reward_manager
                    raw = manager._step_reward.detach().cpu().numpy()
                    names = list(manager.active_terms)
                    vector = group_v1b_s1_terms({name: raw[:, index] for index, name in enumerate(names)}, shape=(args.num_envs,))
                    reward_rows.append(vector * env.unwrapped.step_dt)
                    data = env.unwrapped.scene["robot"].data
                    command = env.unwrapped.command_manager.get_command("base_velocity")
                    quat = data.root_quat_w
                    roll = torch.atan2(2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1 - 2 * (quat[:, 1].square() + quat[:, 2].square()))
                    pitch = torch.asin(torch.clamp(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1, 1))
                    metric_rows.append(torch.stack([(data.root_lin_vel_b[:, 0] - command[:, 0]).abs(), data.root_lin_vel_b[:, 2].abs(), torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1), torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())), action.abs().mean(dim=-1)], dim=-1).cpu())
                    current = obs_tensor(nxt).cuda()
                result = {"disabled_adapter": None if disabled is None else OBJECTIVES[disabled], "reward_means": torch.cat(reward_rows).mean(dim=0).tolist(), "physical_metric_means": torch.cat(metric_rows).mean(dim=0).tolist()}
                actor.load_state_dict(screen_state, strict=True)
                return result
    
            if args.ablation_steps > 0:
                ablation = [evaluate(None)] + [evaluate(index) for index in range(5)]
                phase("S3_ABLATION_COMPLETE")
            else:
                ablation = []
                phase("S3_ABLATION_SKIPPED")
            full = ablation[0] if ablation else None
            effect = [{"disabled_adapter": row["disabled_adapter"], "reward_delta": (np.asarray(row["reward_means"]) - np.asarray(full["reward_means"])).tolist(), "physical_delta": (np.asarray(row["physical_metric_means"]) - np.asarray(full["physical_metric_means"])).tolist()} for row in ablation[1:]] if full else []
            result = {
                "schema": "v1b_s3_alignment_audit_v1",
                "status": "READ_ONLY_COMPLETE",
                "verdict": "AUDIT_ONLY",
                "num_envs": args.num_envs,
                "horizon": args.horizon,
                "ablation_steps": args.ablation_steps,
                "reset_seed": args.reset_seed,
                "source_checkpoint_sha256": sha256(args.source_checkpoint),
                "s2_checkpoint_sha256": sha256(args.s2_checkpoint),
                "adapter_gradient_norms": adapter_norms,
                "shared_gradient_norms": shared_norms,
                "adapter_to_shared_R": [a / (s + 1e-8) for a, s in zip(adapter_norms, shared_norms)],
                "shared_gradient_cosine": shared_cosine,
                "reward_correlation": correlation_matrix(reward_samples),
                "advantage_correlation": correlation_matrix(advantage_samples),
                "reward_to_physical_metric_correlation": alignment,
                "physical_metric_order": ["vx_error", "abs_lin_vel_z", "abs_ang_vel_xy", "tilt_deg", "mean_abs_action"],
                "objective_order": list(OBJECTIVES),
                "leave_one_out_reward_and_physical_effects": effect,
                "scalar_reconstruction_check": {"max_error": max_reconstruction_error, "note": "S1 regrouping compared against the stock environment reward on the frozen rollout"},
                "interpretation": "Read-only geometry/alignment evidence; no intervention or V1-B verdict.",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            phase("S3_ARTIFACT_WRITTEN")
            print(json.dumps(result, indent=2))
        finally:
            if env is not None:
                env.close()
            app.close()
    
    
    if True:
        main()

def run_v1b_s5_measurement_audit():
    """Run former v1b_s5_measurement_audit.py stage."""
    """Read-only V1-B S5 specialization measurement audit."""
    
    import argparse
    import hashlib
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    OBJECTIVES = ("progress", "efficiency", "contact", "balance", "limits")
    SCALES = (0.0, 0.5, 1.0, 1.5)
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def cosine_matrix(outputs):
        vectors = [x.flatten(1) for x in outputs]
        return torch.stack([
            torch.nn.functional.cosine_similarity(left, right, dim=1).mean()
            for left in vectors for right in vectors
        ]).reshape(len(vectors), len(vectors)).cpu().tolist()
    
    
    def linear_cka(left, right):
        left = left - left.mean(dim=0, keepdim=True)
        right = right - right.mean(dim=0, keepdim=True)
        gram_left = left @ left.T
        gram_right = right @ right.T
        hsic = (gram_left * gram_right).sum()
        return float((hsic / (gram_left.square().sum().sqrt() * gram_right.square().sum().sqrt()).clamp_min(1e-12)).item())
    
    
    def correlation(left, right):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.std() < 1e-12 or right.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-checkpoint", type=Path, required=True)
        parser.add_argument("--s4-checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=32)
        parser.add_argument("--steps", type=int, default=150)
        parser.add_argument("--reset-seed", type=int, default=60001)
        args = parser.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "RUN_STARTED.json").write_text(json.dumps({"status": "RUN_STARTED", "protocol": "V1-B-S5"}) + "\n")
        def phase(name):
            (args.output.parent / "PHASE.txt").write_text(name + "\n")
            print(name, flush=True)
        phase("S5_RUN_STARTED")
    
        from isaaclab.app import AppLauncher
    
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        phase("S5_APP_INIT_OK")
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from talon_rl.rewards.baselines import group_v1b_s1_terms
            from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = args.reset_seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            phase("S5_ENV_CREATED")
            agent = UnitreeA1FlatPPORunnerCfg()
            agent.policy.actor_hidden_dims = [128, 128, 128]
            agent.policy.critic_hidden_dims = [128, 128, 128]
            agent.num_steps_per_env = 24
            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(args.output.parent), device="cuda")
            phase("S5_RUNNER_CREATED")
            actor = attach_v1a_policy(runner.alg, args.source_checkpoint, bottleneck_dim=8, w_ref=(0.2,) * 5)
            phase("S5_ACTOR_LOADED")
            state = torch.load(args.s4_checkpoint, map_location="cuda", weights_only=False)
            actor.load_state_dict(state["actor"], strict=True)
            actor.eval()
            phase("S5_CHECKPOINT_LOADED")
            weights = torch.full((args.num_envs, 5), 0.2, device="cuda")
    
            obs, _ = env.reset(seed=args.reset_seed)
            obs = obs_tensor(obs).cuda()
            with torch.no_grad():
                hidden = actor._shared_actor(obs)
                residuals = [adapter(hidden) for adapter in actor.v1a_adapters]
                residual_cosine = cosine_matrix(residuals)
                residual_cka = [[linear_cka(residuals[i], residuals[j]) for j in range(5)] for i in range(5)]
                hidden_norm = torch.linalg.vector_norm(hidden, dim=-1).mean()
                residual_norms = [float(torch.linalg.vector_norm(residual, dim=-1).mean().item()) for residual in residuals]
                weighted_influence = [float((0.2 * torch.linalg.vector_norm(residual, dim=-1).mean() / hidden_norm.clamp_min(1e-12)).item()) for residual in residuals]
                action_weight_norm = float(actor._action_head.weight.norm().item())
                action_sensitivity = [0.2 * action_weight_norm] * 5
    
            def action_for_scales(current_obs, scales):
                h = actor._shared_actor(current_obs)
                delta = torch.stack([adapter(h) for adapter in actor.v1a_adapters], dim=1)
                return actor._action_head(h + (0.2 * torch.as_tensor(scales, device=h.device).view(1, 5, 1) * delta).sum(dim=1))
    
            with torch.no_grad():
                base_action = action_for_scales(obs, [1.0] * 5)
                fixed_action_sensitivity = []
                for index in range(5):
                    row = []
                    for scale in SCALES:
                        candidate = action_for_scales(obs, [scale if j == index else 1.0 for j in range(5)])
                        row.append(float(torch.linalg.vector_norm(candidate - base_action, dim=-1).mean().item()))
                    fixed_action_sensitivity.append(row)
            phase("S5_FIXED_COUNTERFACTUAL_COMPLETE")
    
            def rollout(scales, current):
                reward_rows, metric_rows = [], []
                done_any = np.zeros(args.num_envs, dtype=bool)
                with torch.no_grad():
                    for _ in range(args.steps):
                        raw_action = action_for_scales(current, scales)
                        action = torch.clamp(raw_action, -1.0, 1.0)
                        nxt, _, terminated, truncated, _ = env.step(action)
                        done_any |= (terminated | truncated).cpu().numpy()
                        manager = env.unwrapped.reward_manager
                        raw = manager._step_reward.detach().cpu().numpy()
                        names = list(manager.active_terms)
                        vector = group_v1b_s1_terms({name: raw[:, i] for i, name in enumerate(names)}, shape=(args.num_envs,))
                        reward_rows.append(vector * env.unwrapped.step_dt)
                        data = env.unwrapped.scene["robot"].data
                        command = env.unwrapped.command_manager.get_command("base_velocity")
                        quat = data.root_quat_w
                        roll = torch.atan2(2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1 - 2 * (quat[:, 1].square() + quat[:, 2].square()))
                        pitch = torch.asin(torch.clamp(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1, 1))
                        metric_rows.append(torch.stack([(data.root_lin_vel_b[:, 0] - command[:, 0]).abs(), data.root_lin_vel_b[:, 2].abs(), torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1), torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())), action.abs().mean(dim=-1)], dim=-1).cpu())
                        current = obs_tensor(nxt).cuda()
                return {
                    "scales": list(scales),
                    "survival": float(1.0 - done_any.mean()),
                    "reward_means": torch.cat(reward_rows).mean(dim=0).tolist(),
                    "physical_metric_means": torch.cat(metric_rows).mean(dim=0).tolist(),
                }, current
    
            sweep = []
            if args.steps > 0:
                sweep_obs = obs
                for index in range(5):
                    for scale in SCALES:
                        scales = [1.0] * 5
                        scales[index] = scale
                        phase(f"S5_ROLLOUT_BEGIN_{OBJECTIVES[index]}_{scale}")
                        condition_result, sweep_obs = rollout(scales, sweep_obs)
                        condition = {"adapter": OBJECTIVES[index], **condition_result}
                        sweep.append(condition)
                        (args.output.parent / f"condition_{index}_{str(scale).replace('.', 'p')}.json").write_text(json.dumps(condition, indent=2) + "\n")
                        phase(f"S5_ROLLOUT_DONE_{OBJECTIVES[index]}_{scale}")
                phase("S5_ROLLOUT_SWEEP_COMPLETE")
            else:
                phase("S5_ROLLOUT_SWEEP_SKIPPED")
    
            reward_response = {objective: {} for objective in OBJECTIVES}
            physical_response = {objective: {} for objective in OBJECTIVES}
            for row in sweep:
                index = OBJECTIVES.index(row["adapter"])
                scale = str(row["scales"][index])
                reward_response[row["adapter"]][scale] = row["reward_means"]
                physical_response[row["adapter"]][scale] = row["physical_metric_means"]
    
            # A compact monotonicity diagnostic: correlation of each adapter's
            # scale with each objective reward across its four counterfactuals.
            monotonicity = {}
            if sweep:
                for objective in OBJECTIVES:
                    scales = np.asarray(SCALES)
                    monotonicity[objective] = {
                        target: correlation(scales, [reward_response[objective][str(scale)][target_index] for scale in SCALES])
                        for target, target_index in zip(OBJECTIVES, range(5))
                    }
    
            result = {
                "schema": "v1b_s5_measurement_audit_v1",
                "status": "READ_ONLY_COMPLETE",
                "verdict": "AUDIT_ONLY",
                "num_envs": args.num_envs,
                "steps": args.steps,
                "reset_seed": args.reset_seed,
                "source_checkpoint_sha256": sha256(args.source_checkpoint),
                "s4_checkpoint_sha256": sha256(args.s4_checkpoint),
                "objective_order": list(OBJECTIVES),
                "scales": list(SCALES),
                "adapter_residual_norms": residual_norms,
                "weighted_residual_influence_C": weighted_influence,
                "adapter_output_cosine": residual_cosine,
                "adapter_output_linear_cka": residual_cka,
                "action_head_weight_frobenius_norm": action_weight_norm,
                "action_sensitivity_per_unit_weight": action_sensitivity,
                "fixed_batch_action_delta_norm_by_adapter_and_scale": fixed_action_sensitivity,
                "reward_response_by_adapter_scale": reward_response,
                "physical_response_by_adapter_scale": physical_response,
                "scale_reward_response_correlation": monotonicity,
                "interpretation": "Read-only continuous counterfactual evidence; no V1-B verdict or training authorization.",
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            (args.output.parent / "RUN_DONE.json").write_text(json.dumps({"status": "READ_ONLY_COMPLETE"}) + "\n")
            phase("S5_ARTIFACT_WRITTEN")
            print(json.dumps(result, indent=2))
        finally:
            if env is not None:
                env.close()
            app.close()
    
    
    if True:
        main()

def run_v1b_s5r_reward_sweep():
    """Run former v1b_s5r_reward_sweep.py stage."""
    """V1-B S5R counterfactual adapter-scale sweep using stock actor inference."""
    
    import argparse
    import hashlib
    import json
    import sys
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    OBJECTIVES = ("progress", "efficiency", "contact", "balance", "limits")
    SCALES = (0.0, 0.5, 1.0, 1.5)
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def sha256(path: Path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--source-checkpoint", type=Path, required=True)
        p.add_argument("--s4-checkpoint", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--num-envs", type=int, default=32)
        p.add_argument("--steps", type=int, default=150)
        p.add_argument("--reset-seed", type=int, default=60001)
        args = p.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "RUN_STARTED.json").write_text(json.dumps({"status": "RUN_STARTED", "protocol": "V1-B-S5R"}) + "\n")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    
        from isaaclab.app import AppLauncher
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        (args.output.parent / "APP_INIT_OK").write_text("ok\n")
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from talon_rl.rewards.baselines import group_v1b_s1_terms
            from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
    
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = args.reset_seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            (args.output.parent / "ENV_CREATED").write_text("ok\n")
            agent = UnitreeA1FlatPPORunnerCfg(); agent.policy.actor_hidden_dims = [128, 128, 128]; agent.policy.critic_hidden_dims = [128, 128, 128]; agent.num_steps_per_env = 24
            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(args.output.parent), device="cuda")
            (args.output.parent / "RUNNER_CREATED").write_text("ok\n")
            actor = attach_v1a_policy(runner.alg, args.source_checkpoint, bottleneck_dim=8, w_ref=(0.2,) * 5)
            state = torch.load(args.s4_checkpoint, map_location="cuda", weights_only=False)
            actor.load_state_dict(state["actor"], strict=True); actor.eval()
            (args.output.parent / "CHECKPOINT_LOADED").write_text("ok\n")
            reference = {name: value.detach().clone() for name, value in actor.state_dict().items()}
            weights = torch.full((args.num_envs, 5), 0.2, device="cuda")
            rows = []
            for adapter_index, objective in enumerate(OBJECTIVES):
                for scale in SCALES:
                    condition_id = f"{adapter_index}:{objective}:{scale}"
                    (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "CONDITION_START", "condition": condition_id}) + "\n")
                    actor.load_state_dict(reference, strict=True)
                    for parameter in actor.v1a_adapters[adapter_index].up.parameters():
                        parameter.data.mul_(scale)
                    try:
                        obs, _ = env.reset(seed=args.reset_seed)
                        (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "RESET_OK", "condition": condition_id}) + "\n")
                        obs = obs_tensor(obs).cuda()
                    except BaseException as error:
                        (args.output.parent / "ERROR.json").write_text(json.dumps({"status": "ABORTED", "phase": "RESET", "condition": condition_id, "error": str(error), "traceback": traceback.format_exc()}, indent=2) + "\n")
                        raise
                    reward_rows, vx, tilt, action_abs = [], [], [], []
                    done_any = np.zeros(args.num_envs, dtype=bool)
                    with torch.no_grad():
                        for _ in range(args.steps):
                            try:
                                (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "ACTOR_STEP_START", "condition": condition_id, "step": len(reward_rows) + 1}) + "\n")
                                action = torch.clamp(actor.act_inference(obs, weights), -1.0, 1.0)
                                (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "ACTION_OK", "condition": condition_id, "step": len(reward_rows) + 1, "finite": bool(torch.isfinite(action).all()), "min": float(action.min().item()), "max": float(action.max().item())}) + "\n")
                                nxt, _, terminated, truncated, _ = env.step(action)
                                (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "STEP_OK", "condition": condition_id, "step": len(reward_rows) + 1}) + "\n")
                            except BaseException as error:
                                (args.output.parent / "ERROR.json").write_text(json.dumps({"status": "ABORTED", "phase": "STEP", "condition": condition_id, "step": len(reward_rows) + 1, "error": str(error), "traceback": traceback.format_exc()}, indent=2) + "\n")
                                raise
                            done_any |= (terminated | truncated).cpu().numpy()
                            manager = env.unwrapped.reward_manager; raw = manager._step_reward.detach().cpu().numpy(); names = list(manager.active_terms)
                            (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "REWARD_RAW_OK", "condition": condition_id, "step": len(reward_rows) + 1}) + "\n")
                            vector = group_v1b_s1_terms({name: raw[:, i] for i, name in enumerate(names)}, shape=(args.num_envs,))
                            (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "REWARD_GROUP_OK", "condition": condition_id, "step": len(reward_rows) + 1}) + "\n")
                            reward_rows.append(vector * env.unwrapped.step_dt)
                            (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "REWARD_APPEND_OK", "condition": condition_id, "step": len(reward_rows)}) + "\n")
                            data = env.unwrapped.scene["robot"].data; command = env.unwrapped.command_manager.get_command("base_velocity")
                            quat = data.root_quat_w
                            roll = torch.atan2(2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1 - 2 * (quat[:, 1].square() + quat[:, 2].square()))
                            pitch = torch.asin(torch.clamp(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1, 1))
                            vx.append((data.root_lin_vel_b[:, 0] - command[:, 0]).abs().cpu().numpy())
                            tilt.append(torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())).cpu().numpy())
                            action_abs.append(action.abs().mean(dim=-1).cpu().numpy())
                            if len(reward_rows) < args.steps:
                                obs = obs_tensor(nxt).cuda()
                    rows.append({"adapter": objective, "scale": scale, "survival": float(1 - done_any.mean()), "reward_means": np.stack(reward_rows).mean(axis=(0, 1)).tolist(), "vx_error": float(np.concatenate(vx).mean()), "tilt_p95_deg": float(np.percentile(np.concatenate(tilt), 95)), "max_tilt_deg": float(np.concatenate(tilt).max()), "mean_abs_action": float(np.concatenate(action_abs).mean())})
                    (args.output.parent / "LAST_CONDITION").write_text(f"{objective}:{scale}\n")
                    (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "CONDITION_DONE", "condition": condition_id}) + "\n")
            actor.load_state_dict(reference, strict=True)
            result = {"schema": "v1b_s5r_reward_sweep_v1", "status": "READ_ONLY_COMPLETE", "verdict": "SEMANTIC_SPECIALIZATION_UNRESOLVED", "num_envs": args.num_envs, "steps": args.steps, "reset_seed": args.reset_seed, "scales": list(SCALES), "objective_order": list(OBJECTIVES), "source_checkpoint_sha256": sha256(args.source_checkpoint), "s4_checkpoint_sha256": sha256(args.s4_checkpoint), "rows": rows, "interpretation": "Continuous adapter-scale counterfactual using inherited stock rsl_rl deterministic actor path; no training or verdict authorization."}
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "MATRIX_WRITTEN", "conditions": len(rows)}) + "\n")
            (args.output.parent / "LIFECYCLE.jsonl").open("a").write(json.dumps({"event": "RUN_DONE", "status": "READ_ONLY_COMPLETE"}) + "\n")
            (args.output.parent / "RUN_DONE.json").write_text(json.dumps({"status": "READ_ONLY_COMPLETE", "conditions": len(rows)}) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    
    
    if True: main()

def run_v1b_s7_impact_identifiability_audit():
    """Run former v1b_s7_impact_identifiability_audit.py stage."""
    """V1-B S7 impact-identifiability audit; baseline-only and read-only."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    
    import argparse
    import atexit
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    from rl.experiments.common.utilities.v1b_s6_objective_decomposition_audit import corr_matrix, obs_tensor, safe_corr, sha256, snapshot
    
    
    def main() -> None:
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--num-envs", type=int, default=64)
        p.add_argument("--steps", type=int, default=500)
        p.add_argument("--reset-seed", type=int, default=62001)
        args = p.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lifecycle = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
        error_path = args.output.with_name(args.output.stem + ".ERROR.json")
        state = {"written": False, "failed": False}
    
        def mark(event, **extra):
            with lifecycle.open("a") as f:
                f.write(json.dumps({"event": event, "unix": time.time(), **extra}, sort_keys=True) + "\n")
                f.flush()
    
        def fail(reason, exc=None):
            state["failed"] = True
            payload = {"schema": "v1b_s7_error_v1", "status": "ERROR", "reason": reason}
            if exc is not None:
                payload.update(error=str(exc), traceback=traceback.format_exc())
            error_path.write_text(json.dumps(payload, indent=2) + "\n")
            mark("ERROR", reason=reason, error=str(exc) if exc else None)
    
        def guard():
            if not state["written"] and not state["failed"]:
                fail("PROCESS_EXIT_BEFORE_ARTIFACT")
    
        atexit.register(guard)
        mark("RUN_STARTED", protocol="V1-B-S7", output=str(args.output))
        if not args.checkpoint.exists():
            fail("CHECKPOINT_MISSING", FileNotFoundError(args.checkpoint))
            raise FileNotFoundError(args.checkpoint)
    
        from isaaclab.app import AppLauncher
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        mark("APP_INIT_OK")
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.experiments.common.utilities.v1a_e0_eval import load_policy
    
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = args.reset_seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            mark("ENV_CREATED", num_envs=args.num_envs, reset_seed=args.reset_seed)
            model = load_policy(args.checkpoint, "m01"); model.eval(); before = snapshot(model)
            mark("CKPT_LOADED", checkpoint_sha256=sha256(args.checkpoint))
            obs, _ = env.reset(seed=args.reset_seed); obs = obs_tensor(obs).cuda(); mark("RESET_OK")
    
            term_names = None; term_rows = []; impact_rows = []; balance_rows = []
            returned = []; reconstructed = []; finite = True
            contact_sensor = env.unwrapped.scene["contact_forces"]
            sensor_data = contact_sensor.data
            foot_ids = list(range(sensor_data.net_forces_w.shape[1]))
            previous_contact = torch.zeros((args.num_envs, len(foot_ids)), dtype=torch.bool, device="cuda")
    
            with torch.no_grad():
                for step in range(args.steps):
                    action = torch.clamp(model.act_inference({"policy": obs, "critic": obs}), -1.0, 1.0)
                    nxt, reward, terminated, truncated, _ = env.step(action)
                    if step == 0: mark("STEP_1_OK")
                    manager = env.unwrapped.reward_manager
                    names = list(manager.active_terms)
                    if term_names is None: term_names = names; mark("REWARD_TERMS_DISCOVERED", terms=names)
                    raw = manager._step_reward.detach().cpu().numpy().astype(np.float64)
                    rew = reward.detach().cpu().numpy().astype(np.float64)
                    term_rows.append(raw); returned.append(rew); reconstructed.append(raw.sum(axis=1) * float(env.unwrapped.step_dt))
    
                    forces = sensor_data.net_forces_w[:, foot_ids, :]
                    force_norm = torch.linalg.vector_norm(forces, dim=-1)
                    contact = force_norm > float(contact_sensor.cfg.force_threshold)
                    first_contact = contact & ~previous_contact
                    peak_force = force_norm.max(dim=-1).values
                    total_force = force_norm.sum(dim=-1)
                    impulse_proxy = total_force * float(env.unwrapped.step_dt)
                    touchdown_vz = torch.where(first_contact.any(dim=-1), env.unwrapped.scene["robot"].data.root_lin_vel_b[:, 2].abs(), torch.zeros_like(peak_force))
                    impact_rows.append(torch.stack([peak_force, total_force, impulse_proxy, touchdown_vz], dim=1).cpu().numpy())
    
                    data = env.unwrapped.scene["robot"].data
                    quat = data.root_quat_w
                    roll = torch.atan2(2*(quat[:,0]*quat[:,1]+quat[:,2]*quat[:,3]), 1-2*(quat[:,1].square()+quat[:,2].square()))
                    pitch = torch.asin(torch.clamp(2*(quat[:,0]*quat[:,2]-quat[:,3]*quat[:,1]), -1, 1))
                    balance_rows.append(torch.stack([data.root_lin_vel_b[:,2].abs(), torch.linalg.vector_norm(data.root_ang_vel_b[:,:2], dim=-1), torch.rad2deg(torch.maximum(roll.abs(), pitch.abs()))], dim=1).cpu().numpy())
                    previous_contact = contact
                    finite &= bool(torch.isfinite(obs).all() and torch.isfinite(action).all() and torch.isfinite(reward).all())
                    obs = obs_tensor(nxt).cuda()
                    if (step + 1) % 100 == 0: mark("STEP_PROGRESS", step=step + 1)
    
            terms = np.concatenate(term_rows); impact = np.concatenate(impact_rows); balance = np.concatenate(balance_rows)
            impact_names = ["peak_foot_force", "total_foot_force", "foot_impulse_proxy", "touchdown_abs_vz"]
            balance_names = ["abs_lin_vel_z", "abs_ang_vel_xy", "tilt_deg"]
            term_impact = np.array([[safe_corr(terms[:, i], impact[:, j]) for j in range(impact.shape[1])] for i in range(terms.shape[1])])
            term_balance = np.array([[safe_corr(terms[:, i], balance[:, j]) for j in range(balance.shape[1])] for i in range(terms.shape[1])])
            after = snapshot(model)
            recon = np.abs(np.concatenate(returned) - np.concatenate(reconstructed))
            stats = {name: {"mean": float(terms[:, i].mean()), "std": float(terms[:, i].std()), "nonzero_fraction": float(np.mean(np.abs(terms[:, i]) > 0))} for i, name in enumerate(term_names)}
            impact_stats = {name: {"mean": float(impact[:, i].mean()), "std": float(impact[:, i].std()), "nonzero_fraction": float(np.mean(np.abs(impact[:, i]) > 0))} for i, name in enumerate(impact_names)}
            result = {
                "schema": "v1b_s7_impact_identifiability_audit_v1", "protocol": "V1-B-S7", "status": "READ_ONLY_COMPLETE",
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "num_envs": args.num_envs, "steps": args.steps, "reset_seed": args.reset_seed, "stock_clip_actions": 1.0,
                "reward_term_order": term_names, "impact_metric_order": impact_names, "balance_metric_order": balance_names,
                "term_statistics": stats, "impact_metric_statistics": impact_stats, "term_impact_pearson": term_impact.tolist(), "term_balance_pearson": term_balance.tolist(),
                "scalar_reward_reconstruction": {"max_abs_error": float(recon.max()), "mean_abs_error": float(recon.mean()), "pass": bool(recon.max() <= 1e-5)},
                "integrity": {"finite_state": finite, "evaluator_non_mutation": all(torch.equal(before[k], after[k]) for k in before)},
                "decision": "IMPACT_CANDIDATE_EVIDENCE_ONLY", "interpretation": "Impact is admissible only if a stock reward term is non-degenerate, aligned with impact metrics, and not redundant with balance; no reward was added or regrouped."
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n"); state["written"] = True; mark("ARTIFACT_WRITTEN", path=str(args.output)); mark("RUN_DONE", status="READ_ONLY_COMPLETE")
            print(json.dumps(result, indent=2))
        except BaseException as exc:
            fail("EVALUATION_EXCEPTION", exc); raise
        finally:
            if env is not None: env.close()
            app.close()
    
    
    if True: main()

STAGES = {
    "v1b_leave_one_out": run_v1b_leave_one_out,
    "v1b_s3_alignment_audit": run_v1b_s3_alignment_audit,
    "v1b_s5_measurement_audit": run_v1b_s5_measurement_audit,
    "v1b_s5r_reward_sweep": run_v1b_s5r_reward_sweep,
    "v1b_s7_impact_identifiability_audit": run_v1b_s7_impact_identifiability_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
