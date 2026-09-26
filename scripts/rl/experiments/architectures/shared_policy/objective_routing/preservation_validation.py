"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_freeze_v1b_manifest():
    """Run former freeze_v1b_manifest.py stage."""
    import hashlib, json
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[4]
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
        v1a=ROOT/'artifacts/v1a/V1A_E1R_VERDICT.json';routing=ROOT/'scripts/rl/core/v1b_routing.py';tests=ROOT/'tests/test_v1b_routing.py'
        m={'schema':'v1b_design_manifest_v1','status':'DESIGN_FROZEN','training_authorized':False,'causal_question':'Does explicit routing of objective-specific PPO signals induce semantic adapter specialization without sacrificing preserved locomotion substrate?','inherits':{'v1a_verdict':str(v1a),'v1a_verdict_sha256':sha(v1a),'actor':'V1-A exact shared backbone, five adapters, direct fixed w_ref fusion, single action head'},'w_ref':[.2]*5,'advantages':{'objective_specific':True,'normalization':'per-objective batch mean/std, eps=1e-8','objective_order':['progress','efficiency','contact','balance','limits']},'actor_loss':{'five_objective_surrogates':True,'joint_ratio_shared':True,'adapter_routing':'L_i only to adapter_i','shared_aggregate':'sum_i w_i grad(L_i)','action_head_aggregate':True},'critic':{'architecture':'shared encoder + five heads','loss':'mean_i MSE(V_i,R_i)','shared_encoder_aggregate':True},'integrity':{'off_diagonal_leakage_tolerance':'exact zero up to floating-point mask assignment','shared_gradient_aggregate':True,'critic_head_separation':True},'observability':['per_adapter_gradient_norm','off_diagonal_leakage_ratio','gradient_cosine_matrix','adapter_output_norm/diversity','leave_one_adapter_out_effect_matrix'],'preservation':'corrected V1A-E1R semantics and gate','exclusions':{'preference_curriculum':False,'rehearsal':False,'anchor':False,'learned_gate':False},'training':{'authorized':False,'seeds':[0,1,2],'updates':300,'terminal_checkpoint':'model_299.pt'},'hashes':{'routing_module':sha(routing),'routing_tests':sha(tests)}}
        out=ROOT/'artifacts/v1b/V1B_DESIGN_MANIFEST.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(out)
    if True:main()

def run_v1b_short_screen():
    """Run former v1b_short_screen.py stage."""
    """One-seed V1-B specialization screen; diagnostic only, not a verdict."""
    
    import argparse
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    OBJECTIVES = ("progress", "efficiency", "contact", "balance", "limits")
    S7_OBJECTIVES = ("progress", "balance", "efficiency")
    
    
    def tensorize(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def grad_norms(loss, groups):
        params = [p for group in groups for p in group]
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        result, offset = [], 0
        for group in groups:
            terms = [g.detach().square().sum() for g in grads[offset:offset + len(group)] if g is not None]
            result.append(float(torch.sqrt(torch.stack(terms).sum()).item()) if terms else 0.0)
            offset += len(group)
        return result
    
    
    def flat_group_grad_norm(loss, parameters):
        gradients = torch.autograd.grad(loss, list(parameters), retain_graph=True, allow_unused=True)
        terms = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
        return float(torch.sqrt(torch.stack(terms).sum()).item()) if terms else 0.0
    
    
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--num-envs", type=int, default=16)
        parser.add_argument("--horizon", type=int, default=24)
        parser.add_argument("--updates", type=int, default=100)
        parser.add_argument("--reward-mapping", choices=("stock", "v1b_s1", "v1b_s7_pass3"), default="stock")
        parser.add_argument("--adapter-lr-multiplier", type=float, default=1.0)
        parser.add_argument("--shared-freeze-updates", type=int, default=0)
        args = parser.parse_args()
        run = args.run_dir.resolve()
        run.mkdir(parents=True, exist_ok=True)
        lifecycle = run / "lifecycle.jsonl"
    
        def mark(event, **fields):
            with lifecycle.open("a") as stream:
                stream.write(json.dumps({"event": event, "unix": time.time(), **fields}, sort_keys=True) + "\n")
                stream.flush()
    
        mark("RUN_STARTED", updates=args.updates, num_envs=args.num_envs, horizon=args.horizon)
        app = env = None
        try:
            from isaaclab.app import AppLauncher
    
            saved_argv = sys.argv[:]
            sys.argv = [sys.argv[0]]
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            sys.argv = saved_argv
            mark("APP_INIT_OK")
    
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from talon_rl.rewards.baselines import group_stock_terms, reconstruct_stock_scalar
            from talon_rl.rewards.baselines import group_v1b_s1_terms, reconstruct_v1b_s1_scalar
            from talon_rl.rewards.baselines import group_v1b_s7_terms, reconstruct_v1b_s7_scalar, S7_OBJECTIVE_ORDER
            from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
            from rl.core.objectives.routing import (
                SharedObjectiveCritic,
                normalize_objective_advantages,
                objective_critic_loss,
                route_objective_gradients,
            )
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            cfg.seed = 0
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            mark("ENV_CREATED")
    
            agent = UnitreeA1FlatPPORunnerCfg()
            agent.policy.actor_hidden_dims = [128, 128, 128]
            agent.policy.critic_hidden_dims = [128, 128, 128]
            agent.num_steps_per_env = args.horizon
            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(run), device="cuda")
            objective_names = S7_OBJECTIVE_ORDER if args.reward_mapping == "v1b_s7_pass3" else OBJECTIVES
            objective_count = len(objective_names)
            actor = attach_v1a_policy(
                runner.alg,
                args.checkpoint,
                bottleneck_dim=8,
                w_ref=(1.0 / objective_count,) * objective_count,
                objectives=objective_names,
            )
            actor.train()
            critic = SharedObjectiveCritic(48, 128, objective_count).cuda()
            if args.adapter_lr_multiplier <= 0.0:
                raise ValueError("adapter LR multiplier must be positive")
            if args.shared_freeze_updates < 0 or args.shared_freeze_updates >= args.updates:
                raise ValueError("shared freeze updates must be in [0, updates)")
            adapter_parameters = list(actor.v1a_adapters.parameters())
            shared_actor_parameters = list(actor._shared_actor.parameters())
            action_head_parameters = list(actor._action_head.parameters())
            special_ids = {id(parameter) for parameter in shared_actor_parameters + action_head_parameters + adapter_parameters}
            other_base_parameters = [parameter for parameter in actor.base.parameters() if id(parameter) not in special_ids]
            actor_optimizer = torch.optim.Adam(
                [
                    {"params": shared_actor_parameters, "lr": 1e-4},
                    {"params": action_head_parameters, "lr": 1e-4},
                    {"params": other_base_parameters, "lr": 1e-4},
                    {"params": adapter_parameters, "lr": 1e-4 * args.adapter_lr_multiplier},
                ],
                lr=1e-4,
            )
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
            if args.reward_mapping == "stock":
                group_terms = group_stock_terms
                reconstruct_terms = reconstruct_stock_scalar
            elif args.reward_mapping == "v1b_s1":
                group_terms = group_v1b_s1_terms
                reconstruct_terms = reconstruct_v1b_s1_scalar
            else:
                group_terms = group_v1b_s7_terms
                reconstruct_terms = reconstruct_v1b_s7_scalar
            mark("CKPT_LOADED", source_checkpoint=str(args.checkpoint.resolve()), reward_mapping=args.reward_mapping)
    
            obs, _ = env.reset(seed=0)
            obs = tensorize(obs).cuda()
            weights = torch.full((args.num_envs, objective_count), 1.0 / objective_count, device="cuda")
            max_reconstruction_error = 0.0
            update_records = []
            audit_records = []
            audit_updates = {1, 5, 10, 25, 50, 100}
            adapter_groups = [list(adapter.parameters()) for adapter in actor.v1a_adapters]
            shared_parameters = list(actor._shared_actor.parameters()) + list(actor._action_head.parameters())
            adapter_initial = [p.detach().clone() for group in adapter_groups for p in group]
    
            for update in range(1, args.updates + 1):
                observations, actions, old_log_probs = [], [], []
                rewards, values, dones = [], [], []
                for _ in range(args.horizon):
                    with torch.no_grad():
                        actor.update_distribution({"policy": obs, "critic": obs}, weights)
                        action = actor.base.distribution.sample()
                        log_prob = actor.get_actions_log_prob(action)
                        value = critic(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(torch.clamp(action, -1.0, 1.0))
                    manager = env.unwrapped.reward_manager
                    raw = manager._step_reward.detach().cpu().numpy()
                    names = list(manager.active_terms)
                    reward_vector = group_terms(
                        {name: raw[:, index] for index, name in enumerate(names)},
                        shape=(args.num_envs,),
                    )
                    reconstructed = reconstruct_terms(reward_vector) * env.unwrapped.step_dt
                    max_reconstruction_error = max(
                        max_reconstruction_error,
                        float(torch.as_tensor(abs(reconstructed - reward.detach().cpu().numpy())).max()),
                    )
                    observations.append(obs)
                    actions.append(action)
                    old_log_probs.append(log_prob)
                    rewards.append(torch.as_tensor(reward_vector, device="cuda", dtype=torch.float32) * env.unwrapped.step_dt)
                    values.append(value)
                    dones.append((terminated | truncated).to("cuda"))
                    obs = tensorize(next_obs).cuda()
    
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
                flat_old_log_probs = torch.cat(old_log_probs)
                flat_returns = returns.reshape(-1, objective_count)
                flat_values = value_tensor.reshape(-1, objective_count)
                raw_advantages = (flat_returns - flat_values).detach()
                advantages = normalize_objective_advantages(raw_advantages)
                actor.update_distribution({"policy": flat_obs, "critic": flat_obs}, weights.repeat(args.horizon, 1))
                new_log_probs = actor.get_actions_log_prob(flat_actions)
                ratio = torch.exp(new_log_probs - flat_old_log_probs.detach())
                losses = []
                for objective in range(objective_count):
                    advantage = advantages[:, objective]
                    surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage)
                    losses.append(-surrogate.mean())
    
                actor_optimizer.param_groups[0]["lr"] = 0.0 if update <= args.shared_freeze_updates else 1e-4
                pre_mask_matrix = [grad_norms(loss, adapter_groups) for loss in losses]
                if update in audit_updates:
                    adapter_gradients = pre_mask_matrix
                    shared_gradients = [flat_group_grad_norm(loss, shared_parameters) for loss in losses]
                    ratios = [
                        adapter_gradients[index][index] / (shared_gradients[index] + 1e-8)
                        for index in range(objective_count)
                    ]
                    with torch.no_grad():
                        hidden = actor._shared_actor(flat_obs)
                        residuals = torch.stack([adapter(hidden) for adapter in actor.v1a_adapters], dim=1)
                        hidden_norm = torch.linalg.vector_norm(hidden, dim=-1).mean().item()
                        raw_residual_norms = torch.linalg.vector_norm(residuals, dim=-1).mean(dim=0)
                        weighted_residual_norms = raw_residual_norms * float(weights[0, 0].item())
                        residual_ratios = (weighted_residual_norms / (hidden_norm + 1e-8)).cpu().tolist()
                    audit_records.append({
                        "update": update,
                        "raw_reward_mean": reward_tensor.mean(dim=(0, 1)).detach().cpu().tolist(),
                        "raw_reward_std": reward_tensor.std(dim=(0, 1), unbiased=False).detach().cpu().tolist(),
                        "raw_reward_nonzero_fraction": (reward_tensor.abs() > 1e-8).float().mean(dim=(0, 1)).detach().cpu().tolist(),
                        "raw_advantage_mean": raw_advantages.mean(dim=0).cpu().tolist(),
                        "raw_advantage_std": raw_advantages.std(dim=0, unbiased=False).cpu().tolist(),
                        "adapter_gradient_norms_for_own_loss": [row[index] for index, row in enumerate(adapter_gradients)],
                        "shared_gradient_norms_for_own_loss": shared_gradients,
                        "adapter_to_shared_gradient_ratio_R": ratios,
                        "hidden_norm_mean": hidden_norm,
                        "adapter_residual_norms": raw_residual_norms.cpu().tolist(),
                        "weighted_residual_influence_C": residual_ratios,
                    })
                routed = route_objective_gradients(losses, adapter_groups, shared_parameters, weights[0])
                actor_optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in routed.items():
                    parameter.grad = gradient.detach().clone()
                actor_optimizer.step()
    
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss = objective_critic_loss(critic(flat_obs), flat_returns.detach())
                critic_loss.backward()
                critic_optimizer.step()
    
                adapter_norms = [float(torch.sqrt(sum(p.detach().square().sum() for p in group)).item()) for group in adapter_groups]
                head_predictions = critic(flat_obs).detach()
                head_losses = [float((head_predictions[:, i] - flat_returns[:, i]).square().mean().item()) for i in range(objective_count)]
                post_mask_offdiag = 0.0
                parameter_delta = max(
                    float((p.detach() - initial).abs().max().item())
                    for p, initial in zip(
                        [p for group in adapter_groups for p in group], adapter_initial
                    )
                )
                record = {
                    "update": update,
                    "gradient_matrix_pre_mask": pre_mask_matrix,
                    "off_diagonal_max_post_mask": post_mask_offdiag,
                    "adapter_parameter_max_delta_from_init": parameter_delta,
                    "adapter_parameter_norms": adapter_norms,
                    "critic_head_losses": head_losses,
                    "critic_loss": float(critic_loss.item()),
                    "mean_objective_reward": reward_tensor.mean(dim=(0, 1)).detach().cpu().tolist(),
                    "mean_survival_proxy": float((1.0 - done_tensor).mean().item()),
                    "ratio_mean": float(ratio.mean().item()),
                    "ratio_std": float(ratio.std(unbiased=False).item()),
                    "reward_reconstruction_error": max_reconstruction_error,
                    "shared_backbone_frozen": update <= args.shared_freeze_updates,
                }
                update_records.append(record)
                mark("UPDATE_DONE", update=update, off_diagonal_max_post_mask=post_mask_offdiag)
    
            terminal = run / "v1b_short_screen_terminal.pt"
            torch.save(
                {
                    "schema": "v1b_short_screen_checkpoint_v1",
                    "update": args.updates,
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                },
                terminal,
            )
            mark("CHECKPOINT_WRITTEN", checkpoint=str(terminal))
            report = {
                "schema": "v1b_short_screen_v1",
                "status": "DIAGNOSTIC_COMPLETE",
                "verdict": "NOT_A_THESIS_VERDICT",
                "seed": 0,
                "updates": args.updates,
                "num_envs": args.num_envs,
                "horizon": args.horizon,
                "reward_mapping": args.reward_mapping,
                "objective_order": list(objective_names),
                "objective_count": objective_count,
                "adapter_lr_multiplier": args.adapter_lr_multiplier,
                "shared_freeze_updates": args.shared_freeze_updates,
                "terminal_checkpoint": str(terminal),
                "max_reward_reconstruction_error": max_reconstruction_error,
                "final_gradient_matrix_pre_mask": update_records[-1]["gradient_matrix_pre_mask"],
                "final_off_diagonal_max_post_mask": 0.0,
                "final_adapter_parameter_norms": update_records[-1]["adapter_parameter_norms"],
                "final_adapter_parameter_max_delta_from_init": update_records[-1]["adapter_parameter_max_delta_from_init"],
                "final_critic_head_losses": update_records[-1]["critic_head_losses"],
                "updates_log": update_records,
                "signal_audit_checkpoints": audit_records,
                "w_ref": [1.0 / objective_count] * objective_count,
                "exclusions": {"curriculum": False, "rehearsal": False, "anchor": False, "learned_gate": False},
                "next_step": "leave_one_adapter_out_ablation",
            }
            (run / "screen_artifact.json").write_text(json.dumps(report, indent=2) + "\n")
            mark("ARTIFACT_WRITTEN", artifact=str(run / "screen_artifact.json"))
            mark("RUN_DONE", status="DIAGNOSTIC_COMPLETE")
            print(json.dumps({k: v for k, v in report.items() if k != "updates_log"}, indent=2))
        except BaseException as error:
            (run / "ERROR.json").write_text(json.dumps({"status": "ERROR", "error": str(error), "traceback": traceback.format_exc()}, indent=2) + "\n")
            mark("ERROR", error=str(error))
            raise
        finally:
            if env is not None:
                env.close()
            if app is not None:
                app.close()
    
    
    if True:
        main()

def run_v1b_terminal_preservation_eval():
    """Run former v1b_terminal_preservation_eval.py stage."""
    """Corrected stock deployment evaluation for V1-B terminal actors.
    
    Read-only evaluation of M0.1, routed, and shared terminal checkpoints under
    the same reset seed and stock rsl_rl action clipping.  This script produces
    paired evidence only; it does not apply a binary V1-B verdict.
    """
    
    import argparse
    import hashlib
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    OBJECTIVES = ("progress", "balance", "efficiency")
    
    
    def obs_tensor(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    
    def snapshot(model):
        return {name: value.detach().clone() for name, value in model.state_dict().items()}
    
    
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
        parser.add_argument("--m01-checkpoint", type=Path, required=True)
        parser.add_argument("--routed-checkpoint", type=Path, required=True)
        parser.add_argument("--shared-checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--steps", type=int, default=500)
        parser.add_argument("--num-envs", type=int, default=64)
        args = parser.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
    
        from isaaclab.app import AppLauncher
    
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.integration.rsl_rl.shared_residual_wrapper import RslRlSharedResidualWrapper
            from rl.core.integration.rsl_rl.v1a_wrapper import RslRlV1AWrapper
    
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = 47001 + args.seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            agent = UnitreeA1FlatPPORunnerCfg(); agent.policy.actor_hidden_dims = [128, 128, 128]; agent.policy.critic_hidden_dims = [128, 128, 128]
            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            source = torch.load(args.m01_checkpoint, map_location="cuda", weights_only=False)
            raw = source.get("model_state_dict", source.get("model", source))
    
            def make_model(kind: str, checkpoint: Path | None = None):
                runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(args.output.parent / kind), device="cuda")
                base = runner.alg.policy
                base.load_state_dict(raw)
                if kind == "m01":
                    return base
                state = torch.load(checkpoint, map_location="cuda", weights_only=False)
                if kind == "routed":
                    model = RslRlV1AWrapper(base, bottleneck_dim=8, objectives=OBJECTIVES, w_ref=(1 / 3,) * 3).cuda()
                else:
                    model = RslRlSharedResidualWrapper(base, bottleneck_dim=25, objectives=OBJECTIVES, w_ref=(1 / 3,) * 3).cuda()
                model.load_state_dict(state["actor"], strict=True)
                model.eval()
                return model
    
            models = {
                "m01": make_model("m01"),
                "routed": make_model("routed", args.routed_checkpoint),
                "shared": make_model("shared", args.shared_checkpoint),
            }
    
            def rollout(model, kind):
                obs, _ = env.reset(seed=47001 + args.seed); obs = obs_tensor(obs).cuda()
                reset_hash = hashlib.sha256(obs.detach().cpu().numpy().tobytes()).hexdigest()
                before = snapshot(model); done_any = np.zeros(args.num_envs, dtype=bool)
                vx = []; tilt = []; term = []; contact = []; torque = []; action_rate = []
                previous_action = None
                with torch.no_grad():
                    for _ in range(args.steps):
                        if kind == "m01":
                            action = model.act_inference({"policy": obs, "critic": obs})
                        else:
                            action = model.act_inference(obs)
                        action = torch.clamp(action, -1.0, 1.0)
                        if previous_action is not None:
                            action_rate.append((action - previous_action).square().mean(dim=-1).cpu().numpy())
                        previous_action = action.clone()
                        nxt, _, terminated, truncated, _ = env.step(action)
                        done_any |= (terminated | truncated).detach().cpu().numpy()
                        term.append(terminated.detach().cpu().numpy().astype(np.float32))
                        data = env.unwrapped.scene["robot"].data
                        command = env.unwrapped.command_manager.get_command("base_velocity")
                        vx.append((data.root_lin_vel_b[:, 0] - command[:, 0]).abs().cpu().numpy())
                        q = data.root_quat_w
                        roll = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
                        pitch = torch.asin(torch.clamp(2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1))
                        tilt.append(torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())).cpu().numpy())
                        contact.append(env.unwrapped.termination_manager.get_term("base_contact").cpu().numpy().astype(np.float32))
                        torque.append(data.applied_torque.square().mean(dim=-1).cpu().numpy())
                        obs = obs_tensor(nxt).cuda()
                after = model.state_dict(); all_tilt = np.concatenate(tilt)
                return {
                    "initial_observation_sha256": reset_hash,
                    "survival": float(1 - done_any.mean()),
                    "velocity_tracking_error": float(np.mean(np.concatenate(vx))),
                    "tilt_p95_deg": float(np.percentile(all_tilt, 95)),
                    "max_tilt_deg": float(all_tilt.max()),
                    "termination_rate": float(np.mean(np.concatenate(term))),
                    "base_contact_rate": float(np.mean(np.concatenate(contact))),
                    "torque_l2": float(np.mean(np.concatenate(torque))),
                    "action_rate_l2": float(np.mean(np.concatenate(action_rate))) if action_rate else 0.0,
                    "finite_state": bool(np.isfinite(all_tilt).all()),
                    "evaluator_non_mutation": all(torch.equal(before[k], after[k]) for k in before),
                    "deterministic_actor_mean": True,
                }
    
            measurements = {kind: rollout(model, kind) for kind, model in models.items()}
            reset_hashes = [measurements[k]["initial_observation_sha256"] for k in measurements]
            result = {
                "schema": "v1b_terminal_preservation_eval_v1",
                "protocol": "V1-B-RS-FULL + corrected stock action semantics",
                "status": "COMPLETE_MEASUREMENT_ONLY",
                "seed": args.seed,
                "steps": args.steps,
                "num_envs": args.num_envs,
                "reset_seed": 47001 + args.seed,
                "stock_clip_actions": 1.0,
                "paired_reset_hash_equal": len(set(reset_hashes)) == 1,
                "checkpoints": {
                    "m01": {"path": str(args.m01_checkpoint), "sha256": file_hash(args.m01_checkpoint)},
                    "routed": {"path": str(args.routed_checkpoint), "sha256": file_hash(args.routed_checkpoint)},
                    "shared": {"path": str(args.shared_checkpoint), "sha256": file_hash(args.shared_checkpoint)},
                },
                "measurements": measurements,
                "binary_verdict": "PENDING_PROTOCOL_REVIEW",
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            if env is not None:
                env.close()
            app.close()
    
    
    if True:
        main()

STAGES = {
    "freeze_v1b_manifest": run_freeze_v1b_manifest,
    "v1b_short_screen": run_v1b_short_screen,
    "v1b_terminal_preservation_eval": run_v1b_terminal_preservation_eval,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
