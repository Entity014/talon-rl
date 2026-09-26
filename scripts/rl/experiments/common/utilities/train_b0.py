#!/usr/bin/env python3
"""Frozen B0.1 scalar-PPO training entrypoint; intentionally no MOPPO path."""
from __future__ import annotations

import argparse, datetime as dt, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "artifacts" / "b0_1_freeze" / "FREEZE.json"
B1_FREEZE = ROOT / "artifacts" / "b1_p1_freeze" / "B1_P1_FREEZE.json"
B2_FREEZE = ROOT / "artifacts" / "b1_p2_freeze" / "B1_P2_FREEZE.json"
B1S1_FREEZE = ROOT / "artifacts" / "b1_s1_freeze" / "B1_S1_FREEZE.json"
B1R1_FREEZE = ROOT / "artifacts" / "b1_r1_freeze" / "B1_R1_FREEZE.json"
ROLL_OUT_STEPS = 16                 # validated B0 smoke collection length
HIDDEN_DIMS = [64, 64]              # validated B0 smoke architecture


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path: Path, data: dict) -> None: path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def nominalize(cfg) -> None:
    from talon_rl.assets.unitree_a1.a1 import TALON_A1_CFG
    cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None; cfg.scene.robot = TALON_A1_CFG.replace()
    cfg.events.randomize_payload_mass.params["mass_distribution_params"] = (0., 0.)
    cfg.events.randomize_payload_com.params["com_range"] = {"x": (0., 0.), "y": (0., 0.), "z": (0., 0.)}
    cfg.events.randomize_friction.params.update(static_friction_range=(1., 1.), dynamic_friction_range=(1., 1.), restitution_range=(0., 0.))
    cfg.events.randomize_motor_power.params.update(stiffness_distribution_params=(1., 1.), damping_distribution_params=(1., 1.))
    cfg.events.randomize_joint_range.params["scale_range"] = (1., 1.); cfg.events.push_robot = None
    cfg.curriculum.terrain_levels = None; cfg.terminations.obstacle_reached = None; cfg.episode_length_s = 22.; cfg.stand_phase_s = 0.


def verify(manifest: dict, args) -> None:
    if manifest.get("status") != "FROZEN" or not manifest.get("training_authorized"): raise RuntimeError("B0.1 is not an authorized frozen formulation")
    files = {"reward": "talon_rl/rewards/baselines.py", "b0_reward": "talon_rl/rewards/baselines.py", "trainer": "scripts/rl/core/algorithms/scalar_ppo.py", "environment_wrapper": "talon_rl/wrappers/scalar_reward_env.py", "monitor": "scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py", "train_runner": "scripts/rl/experiments/common/utilities/train_b0.py", "runner": "scripts/rl/experiments/common/utilities/train_b0.py"}
    for name, rel in files.items():
        if name in manifest["sha256"] and manifest["sha256"].get(name) != digest(ROOT / rel): raise RuntimeError(f"freeze mismatch: {name}")
    for name, rel in {"frozen_reset_states": "artifacts/b0_smoke/b0-monitor-frozen-reset-states.npz", "reset_states": "artifacts/b0_smoke/b0-monitor-frozen-reset-states.npz"}.items():
        if name in manifest["sha256"] and manifest["sha256"].get(name) != digest(ROOT / rel): raise RuntimeError(f"freeze mismatch: {name}")
    if args.b1_p1 and manifest.get("schema") != "b1_p1_freeze_v1": raise RuntimeError("B1-P1 requires B1 manifest")
    if args.b1_r1 and manifest.get("schema") != "b1_r1_freeze_v1": raise RuntimeError("B1-R1 requires R1 manifest")
    if args.b1_p2 and not args.b1_r1 and manifest.get("schema") not in ("b1_p2_freeze_v1", "b1_s1_freeze_v1"): raise RuntimeError("B1-P2 requires P2/S1 manifest")
    if args.b1_p2 and (manifest.get("desired_kl") != 0.01 or manifest.get("kl_low") != 0.005 or manifest.get("kl_high") != 0.02): raise RuntimeError("B1-P2 KL config differs from freeze")
    if manifest.get("schema") in ("b1_p1_freeze_v1", "b1_p2_freeze_v1", "b1_s1_freeze_v1", "b1_r1_freeze_v1"):
        if args.seed not in manifest["seeds"] or args.command_vx != manifest["command_vx"]: raise RuntimeError("B1 seed or command differs from freeze")
        if args.b1_p1 and (manifest["target_kl"] != 0.01 or manifest["stop_threshold"] != 0.015): raise RuntimeError("B1-P1 KL config differs from freeze")
        if not args.smoke and (args.num_envs != manifest["num_envs"] or args.updates != manifest["updates"]): raise RuntimeError("B1 production num_envs/updates differ from freeze")
        return
    cfg = manifest["effective_config"]
    if args.seed not in cfg["seeds"] or args.command_vx != cfg["command"][0]: raise RuntimeError("seed or command differs from freeze")
    if cfg.get("rollout_steps") != ROLL_OUT_STEPS or cfg.get("hidden_dims") != HIDDEN_DIMS: raise RuntimeError("runner architecture or rollout length differs from freeze")
    if not args.smoke and (args.num_envs != cfg["num_envs"] or args.updates != cfg["updates"]): raise RuntimeError("production num_envs/updates differ from freeze")


def monitor(run: Path, model: torch.nn.Module, update: int) -> dict:
    model_path, result = run / f"monitor_model_u{update:03d}.pt", run / "monitor" / f"u{update:03d}.json"
    torch.save(model.state_dict(), model_path); result.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/rl/experiments/common/utilities/b0_monitor_isolation_smoke.py"), "--worker", str(model_path), str(result), "--reuse-frozen"]
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'scripts'}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return json.loads(result.read_text())


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True); p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--updates", type=int, default=500); p.add_argument("--command-vx", type=float, default=.5); p.add_argument("--smoke", action="store_true")
    p.add_argument("--b1-p1", action="store_true", help="enable target-KL actor-only early stopping (closed B1-P1)")
    p.add_argument("--b1-p2", action="store_true", help="enable co_rl-inspired adaptive-KL actor LR (B1-P2 draft)")
    p.add_argument("--b1-s1", action="store_true", help="enable ACAPS-style stability regularization (B1-S1 draft)")
    p.add_argument("--b1-r1", action="store_true", help="enable shared-trunk privileged reconstruction (B1-R1)")
    p.add_argument("--run-dir", type=Path); args = p.parse_args()
    manifest_path = B1R1_FREEZE if args.b1_r1 else (B1_FREEZE if args.b1_p1 else (B1S1_FREEZE if args.b1_s1 else (B2_FREEZE if args.b1_p2 else FREEZE)))
    manifest = json.loads(manifest_path.read_text()); verify(manifest, args)
    suffix = "smoke" if args.smoke else dt.date.today().isoformat(); run = args.run_dir or ROOT / "runs" / f"b0_1_seed{args.seed}_{suffix}"
    run = run.resolve()
    if run.exists(): raise FileExistsError(f"refusing to reuse run directory: {run}")
    for sub in (run, run / "checkpoints", run / "monitor"): sub.mkdir(parents=True)
    write(run / "config.json", {"seed": args.seed, "num_envs": args.num_envs, "updates": args.updates, "command": [args.command_vx, 0., 0.], "rollout_steps": ROLL_OUT_STEPS, "hidden_dims": HIDDEN_DIMS, "smoke": args.smoke, "b1_p1": args.b1_p1, "b1_p2": args.b1_p2 or args.b1_s1 or args.b1_r1, "b1_s1": args.b1_s1, "b1_r1": args.b1_r1, "target_kl": 0.01 if args.b1_p1 else None, "kl_stop_threshold": 0.015 if args.b1_p1 else None})
    (run / "freeze_manifest_copy.json").write_bytes(FREEZE.read_bytes()); write(run / "RUN_STARTED.json", {"status": "RUN_STARTED", "unix": time.time(), "seed": args.seed})
    app = base = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env
        from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
        from rl.core.algorithms.scalar_ppo import B0PPOConfig, B0PPOTrainer, ScalarRolloutBuffer, scalar_gae
        from rl.core.modules.actor_critic import ActorCritic
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        cfg = IsaacLabTalonEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = args.seed; cfg.sim.dt = .01; cfg.decimation = 1; nominalize(cfg)
        base = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped; env = B0TalonEnv(base); trans = env.reset(); obs = trans["obs"]
        model = ActorCritic(base.obs_dim, base.obs_dim, base.action_dim, 1, HIDDEN_DIMS, reconstruction_dim=3 if args.b1_r1 else 0).to(base.device)
        if args.b1_p1 and (args.b1_p2 or args.b1_s1 or args.b1_r1): raise ValueError("B1-P1 is mutually exclusive with P2/S1/R1")
        if args.b1_s1 and not args.b1_p2: raise ValueError("B1-S1 requires P2 adaptive LR")
        trainer_cfg = B0PPOConfig(b1_p1_enabled=args.b1_p1, b1_p2_enabled=(args.b1_p2 or args.b1_s1 or args.b1_r1), b1_s1_enabled=args.b1_s1, b1_r1_enabled=args.b1_r1, actor_epochs=4 if (args.b1_p1 or args.b1_p2 or args.b1_s1 or args.b1_r1) else 1)
        trainer = B0PPOTrainer(model, trainer_cfg)
        metrics, monitor_metrics = [], [{"update": 0, **monitor(run, model, 0)}]
        for update in range(args.updates):
            std = trainer.begin_update(); buf = ScalarRolloutBuffer(ROLL_OUT_STEPS, args.num_envs)
            rollout_targets = []
            for _ in range(ROLL_OUT_STEPS):
                x = torch.as_tensor(obs, device=base.device, dtype=torch.float32)
                with torch.no_grad(): action, logp = model.act(x); value = model.value(x).squeeze(-1)
                trans = env.step(action.cpu().numpy().astype(np.float32)); buf.append(obs, action.cpu().numpy(), logp.cpu().numpy(), trans["reward"], trans["done"], value.cpu().numpy())
                if args.b1_r1:
                    fields=trans["fields"]; rollout_targets.append(np.stack([np.abs(fields["roll_pitch"]).max(-1), fields["height"], trans["term_base_contact"].astype(np.float32)],-1))
                obs = trans["obs"]
            with torch.no_grad(): final = model.value(torch.as_tensor(obs, device=base.device, dtype=torch.float32)).squeeze(-1).cpu().numpy()
            buf.finish(final); a = buf.arrays(); adv = scalar_gae(a["rewards"], np.r_[a["values"], a["final_value"][None]], a["dones"], trainer.cfg.gamma, trainer.cfg.gae_lambda); ret = adv + a["values"]
            flat = buf.flatten(); x = torch.as_tensor(flat["obs"], device=base.device, dtype=torch.float32); act = torch.as_tensor(flat["actions"], device=base.device, dtype=torch.float32)
            loss = trainer.ppo_loss(x, x, act, torch.as_tensor(flat["logp_old"], device=base.device), torch.as_tensor(adv.reshape(-1), device=base.device), torch.as_tensor(ret.reshape(-1), device=base.device))
            update_stats = trainer.optimize_batch(x, act, torch.as_tensor(flat["logp_old"], device=base.device), torch.as_tensor(adv.reshape(-1), device=base.device), torch.as_tensor(ret.reshape(-1), device=base.device), rollout_obs=torch.as_tensor(a["obs"], device=base.device, dtype=torch.float32), rollout_dones=torch.as_tensor(a["dones"], device=base.device, dtype=torch.bool), reconstruction_targets=(torch.as_tensor(np.stack(rollout_targets), device=base.device, dtype=torch.float32) if args.b1_r1 else None))
            trainer.finish_update()
            metrics.append({"update": update + 1, "std": std, "reward_mean": float(a["rewards"].mean()), **update_stats})
            if (update + 1) % 25 == 0 or update + 1 == args.updates:
                trainer.save(run / "checkpoints" / f"update_{update + 1:03d}.pt")
            if (update + 1) % 25 == 0:
                report = monitor(run, model, update + 1); monitor_metrics.append({"update": update + 1, **report})
                if update + 1 == 250:
                    write(run / "early_review_input.json", {"seed": args.seed, "update": 250,
                          "survival_at_0": monitor_metrics[0]["acceptance"]["survival_rate"],
                          "survival_at_250": report["acceptance"]["survival_rate"],
                          "rule": "aggregate all three seeds before deciding"})
        write(run / "training_metrics.json", {"metrics": metrics, "monitor": monitor_metrics})
        trainer.save(run / "checkpoints" / "final.pt")
        write(run / "RUN_DONE.json", {"status": "RUN_DONE", "unix": time.time(), "exit_code": 0, "final_update": trainer.update_idx})
    except BaseException as exc:
        write(run / "ERROR.json", {"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}); raise
    finally:
        if base is not None: base.close()
        if app is not None: app.close()


if __name__ == "__main__": main()
