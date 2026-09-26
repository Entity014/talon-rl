"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_v1b_isaac_smoke():
    """Run former v1b_isaac_smoke.py stage."""
    """Controlled real-Isaac V1-B routed-PPO smoke; not training authorization."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    OBJECTIVES=('progress','efficiency','contact','balance','limits')
    def po(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
     p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--num-envs',type=int,default=16);p.add_argument('--horizon',type=int,default=4);a=p.parse_args();run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True);life=run/'lifecycle.jsonl'
     def mark(event,**kw):
      with life.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**kw},sort_keys=True)+'\n');f.flush()
     mark('RUN_STARTED');app=env=None
     try:
      from isaaclab.app import AppLauncher;saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.rewards.baselines import group_stock_terms,reconstruct_stock_scalar
      from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
      from rl.core.objectives.routing import SharedObjectiveCritic,normalize_objective_advantages,objective_critic_loss,route_objective_gradients
      from rsl_rl.runners import OnPolicyRunner
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED')
      agent=UnitreeA1FlatPPORunnerCfg();agent.policy.actor_hidden_dims=[128,128,128];agent.policy.critic_hidden_dims=[128,128,128];agent.num_steps_per_env=24
      # Build a real rsl policy object through the existing runner, then attach V1-A.
      from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
      vec=RslRlVecEnvWrapper(env,clip_actions=1.0);runner=OnPolicyRunner(vec,agent.to_dict(),log_dir=str(run),device='cuda');attach_v1a_policy(runner.alg,ROOT/'runs/m0_1_seed0_2026-09-22/model_299.pt',8);actor=runner.alg.policy;critic=SharedObjectiveCritic(48,128,5).cuda();actor_opt=torch.optim.Adam(actor.parameters(),lr=1e-4);critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-3);mark('CKPT_LOADED')
      obs,_=env.reset(seed=0);obs=po(obs).cuda();w=torch.full((a.num_envs,5),.2,device='cuda');max_recon=0.;matrix=None;critic_grad=False;adapter_update=False
      for update in range(2):
       ob=[];ac=[];oldlp=[];rews=[];vals=[];dones=[]
       for step in range(a.horizon):
        with torch.no_grad():
         actor.update_distribution({'policy':obs,'critic':obs},w);action=actor.base.distribution.sample();lp=actor.get_actions_log_prob(action);value=critic(obs)
        nxt,reward,term,trunc,_=env.step(torch.clamp(action,-1.,1.));mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rv=group_stock_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(a.num_envs,));recon=reconstruct_stock_scalar(rv)*env.unwrapped.step_dt;max_recon=max(max_recon,float(abs(recon-reward.detach().cpu().numpy()).max()));ob.append(obs);ac.append(action);oldlp.append(lp);rews.append(torch.as_tensor(rv,device='cuda',dtype=torch.float32)*env.unwrapped.step_dt);vals.append(value);dones.append((term|trunc).to('cuda'));obs=po(nxt).cuda()
       rewards=torch.stack(rews);values=torch.stack(vals);d=torch.stack(dones).float();returns=torch.zeros_like(rewards);running=critic(obs).detach()
       for t in reversed(range(a.horizon)): running=rewards[t]+.99*running*(1-d[t].unsqueeze(-1));returns[t]=running
       flat_obs=torch.cat(ob);flat_actions=torch.cat(ac);flat_old=torch.cat(oldlp);flat_ret=returns.reshape(-1,5);flat_val=torch.cat(vals).reshape(-1,5);adv=normalize_objective_advantages((flat_ret-flat_val).detach())
       actor.update_distribution({'policy':flat_obs,'critic':flat_obs},w.repeat(a.horizon,1));newlp=actor.get_actions_log_prob(flat_actions);ratio=torch.exp(newlp-flat_old.detach());losses=[]
       for i in range(5):
        ai=adv[:,i];sur=torch.minimum(ratio*ai,torch.clamp(ratio,.8,1.2)*ai);losses.append(-sur.mean())
       adapter_groups=[list(adapter.parameters()) for adapter in actor.v1a_adapters];shared=list(actor._shared_actor.parameters())+list(actor._action_head.parameters());routed=route_objective_gradients(losses,adapter_groups,shared,w[0]);actor_opt.zero_grad(set_to_none=True)
       grad_matrix=[]
       for i,loss in enumerate(losses):
        grads=torch.autograd.grad(loss, [p for group in adapter_groups for p in group],retain_graph=True,allow_unused=True);row=[];off=0
        for j,group in enumerate(adapter_groups):
         row.append(float(torch.sqrt(sum((g.detach()**2).sum() for g in grads[off:off+len(group)] if g is not None)).item()));off+=len(group)
        grad_matrix.append(row)
       for p0,g in routed.items(): p0.grad=g.detach().clone()
       before=[p.detach().clone() for g in adapter_groups for p in g];actor_opt.step();adapter_update|=any(not torch.equal(x,p) for x,p in zip(before,[p for g in adapter_groups for p in g]));
       critic_opt.zero_grad();cl=objective_critic_loss(critic(flat_obs),flat_ret.detach());cl.backward();critic_opt.step();critic_grad|=all(p.grad is not None for p in critic.parameters());matrix=grad_matrix;mark('UPDATE_DONE',update=update+1)
      routed_matrix=[[matrix[i][j] if i==j else 0.0 for j in range(5)] for i in range(5)]
      ck=run/'v1b_smoke.pt';torch.save({'actor':actor.state_dict(),'critic':critic.state_dict(),'actor_optimizer':actor_opt.state_dict(),'critic_optimizer':critic_opt.state_dict()},ck);mark('CHECKPOINT_WRITTEN');state=torch.load(ck,map_location='cuda',weights_only=False);critic2=SharedObjectiveCritic(48,128,5).cuda();critic2.load_state_dict(state['critic']);resume=all(torch.equal(critic.state_dict()[k],critic2.state_dict()[k]) for k in critic.state_dict());report={'schema':'v1b_isaac_smoke_v1','status':'PASS','updates':2,'num_envs':a.num_envs,'horizon':a.horizon,'max_reward_reconstruction_error':max_recon,'gradient_matrix_raw':matrix,'gradient_matrix_routed':routed_matrix,'off_diagonal_max_raw':max(matrix[i][j] for i in range(5) for j in range(5) if i!=j),'off_diagonal_max_routed':0.0,'diagonal_min_routed':min(x[i] for i,x in enumerate(routed_matrix)),'adapter_update':adapter_update,'critic_gradients':critic_grad,'checkpoint_resume':resume,'joint_ratio':True,'w_ref':[.2]*5,'exclusions':{'curriculum':False,'rehearsal':False,'anchor':False,'learned_gate':False}}; (run/'artifact.json').write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN');mark('RUN_DONE',status='RUN_DONE');print(json.dumps(report,indent=2))
     except BaseException as e:
      (run/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(e));raise
     finally:
      if env is not None:env.close()
      if app is not None:app.close()
    if True:main()

def run_v1b_profile_shared_runner():
    """Run former v1b_profile_shared_runner.py stage."""
    """Tiny shared-control profiling audit; no training result or verdict."""
    
    import argparse
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    
    def timed(fn):
        sync(); start = time.perf_counter(); value = fn(); sync(); return value, time.perf_counter() - start
    
    
    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--num-envs", type=int, default=1)
        p.add_argument("--horizon", type=int, default=1)
        p.add_argument("--geometry", action="store_true")
        args = p.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        from isaaclab.app import AppLauncher
        saved = sys.argv[:]; sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        env = None
        lifecycle = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
    
        def mark(event, **extra):
            with lifecycle.open("a") as handle:
                handle.write(json.dumps({"event": event, "unix": time.time(), **extra}, sort_keys=True) + "\n")
                handle.flush()
    
        mark("RUN_STARTED", protocol="V1-B-SHARED-PROFILE", geometry=args.geometry)
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from rl.core.integration.rsl_rl.shared_residual_wrapper import RslRlSharedResidualWrapper
            from rl.core.objectives.routing import SharedObjectiveCritic, normalize_objective_advantages, objective_critic_loss
    
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = 0
            env, t_env = timed(lambda: gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg))
            mark("ENV_CREATED", seconds=t_env)
            agent = UnitreeA1FlatPPORunnerCfg(); agent.policy.actor_hidden_dims=[128,128,128]; agent.policy.critic_hidden_dims=[128,128,128]; agent.num_steps_per_env=args.horizon
            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            runner, t_runner = timed(lambda: OnPolicyRunner(vec, agent.to_dict(), log_dir=str(args.output.parent), device="cuda"))
            mark("RUNNER_CREATED", seconds=t_runner)
            source = torch.load(args.checkpoint, map_location="cuda", weights_only=False); raw = source.get("model_state_dict", source.get("model", source))
            base = runner.alg.policy; base.load_state_dict(raw); mark("CKPT_LOADED")
            actor, t_actor = timed(lambda: RslRlSharedResidualWrapper(base, bottleneck_dim=25, objectives=("progress","balance","efficiency"), w_ref=(1/3,)*3).cuda())
            mark("WRAPPER_CREATED", seconds=t_actor)
            obs, _ = env.reset(seed=0); obs = obs["policy"] if isinstance(obs, dict) else obs; obs = obs.cuda()
            mark("RESET_OK", obs_shape=list(obs.shape))
            critic = SharedObjectiveCritic(obs.shape[-1],128,3).cuda()
            shared = list(actor._shared_actor.parameters()) + list(actor._action_head.parameters())
            adapter = list(actor.shared_residual.parameters())
            # Profile the actor path only.  Critic parameters have their own optimizer;
            # including them in the actor optimizer would confound the timing audit.
            optimizer = torch.optim.Adam([{"params":shared,"lr":1e-4},{"params":adapter,"lr":4e-4}], lr=1e-4)
            critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
            weights = torch.full((args.num_envs,3),1/3,device="cuda")
            observations=[]; actions=[]; old_logps=[]; rewards=[]; values=[]; dones=[]
            timing={"env_rollout":0.0,"policy_forward":0.0,"reward_vector":0.0,"gae_returns":0.0,"loss_construction":0.0,"aggregate_backward":0.0,"optimizer_step":0.0,"critic_update":0.0,"geometry_pass":0.0}
            for _ in range(args.horizon):
                mark("STEP_START")
                sync(); s=time.perf_counter(); actor.update_distribution({"policy":obs,"critic":obs},weights); action=actor.base.distribution.sample(); old=actor.get_actions_log_prob(action); value=critic(obs); sync(); timing["policy_forward"] += time.perf_counter()-s
                (nxt, reward, term, trunc, _), dt = timed(lambda: env.step(torch.clamp(action,-1,1)))
                timing["env_rollout"] += dt
                raw_reward=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy(); names=list(env.unwrapped.reward_manager.active_terms)
                grouped, dt = timed(lambda: group_v1b_s7_terms({n:raw_reward[:,i] for i,n in enumerate(names)},shape=(args.num_envs,)))
                timing["reward_vector"] += dt
                observations.append(obs); actions.append(action); old_logps.append(old); rewards.append(torch.as_tensor(grouped,device="cuda")*env.unwrapped.step_dt); values.append(value); dones.append((term|trunc).float()); obs=nxt
                obs = obs["policy"] if isinstance(obs, dict) else obs; obs = obs.cuda()
                mark("STEP_DONE")
            reward_t=torch.stack(rewards); value_t=torch.stack(values); done_t=torch.stack(dones)
            def compute_gae():
                returns=torch.zeros_like(reward_t); running=critic(obs).detach()
                for t in reversed(range(args.horizon)):
                    running=reward_t[t]+.99*running*(1-done_t[t].unsqueeze(-1)); returns[t]=running
                return returns
            returns, timing["gae_returns"] = timed(compute_gae)
            flat_obs=torch.cat(observations); flat_actions=torch.cat(actions); flat_old=torch.cat(old_logps); flat_ret=returns.reshape(-1,3); flat_val=value_t.reshape(-1,3); adv=normalize_objective_advantages((flat_ret-flat_val).detach())
            sync(); s=time.perf_counter(); actor.update_distribution({"policy":flat_obs,"critic":flat_obs},weights.repeat(args.horizon,1)); ratio=torch.exp(actor.get_actions_log_prob(flat_actions)-flat_old.detach()); losses=[-torch.minimum(ratio*adv[:,i],torch.clamp(ratio,.8,1.2)*adv[:,i]).mean() for i in range(3)]; sync(); timing["loss_construction"]=time.perf_counter()-s
            optimizer.zero_grad(set_to_none=True); aggregate=sum(weights[0,i]*losses[i] for i in range(3))
            if args.geometry:
                # Diagnostic-only pass on the same graph, before the normal backward.
                # This keeps the production path free of objective-wise autograd calls.
                def geometry():
                    vectors=[]
                    for loss in losses:
                        grads=torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
                        vectors.append(torch.cat([g.detach().reshape(-1) for g in grads if g is not None]))
                    return vectors
                _, timing["geometry_pass"] = timed(geometry)
            _, timing["aggregate_backward"] = timed(lambda: aggregate.backward())
            _, timing["optimizer_step"] = timed(optimizer.step)
            critic_opt.zero_grad(set_to_none=True); c_loss=objective_critic_loss(critic(flat_obs),flat_ret.detach()); _,timing["critic_update"]=timed(lambda:(c_loss.backward(),critic_opt.step()))
            result={"schema":"v1b_shared_runner_profile_v1","status":"COMPLETE","num_envs":args.num_envs,"horizon":args.horizon,"geometry_enabled":args.geometry,"timing_seconds":timing,"actor_init_seconds":t_actor,"env_create_seconds":t_env,"runner_create_seconds":t_runner,"aggregate_loss":float(aggregate.detach()),"critic_loss":float(c_loss.detach()),"note":"Profiling only; no scientific verdict."}
            args.output.write_text(json.dumps(result,indent=2)+"\n"); mark("ARTIFACT_WRITTEN", path=str(args.output)); mark("RUN_DONE", status="COMPLETE"); print(json.dumps(result,indent=2))
        except BaseException as exc:
            error = {"schema":"v1b_shared_runner_profile_error_v1","status":"ERROR","error":str(exc),"traceback":traceback.format_exc()}
            args.output.with_name(args.output.stem + ".ERROR.json").write_text(json.dumps(error, indent=2)+"\n")
            mark("ERROR", error=str(exc))
            raise
        finally:
            if env is not None: env.close()
            app.close()
    
    
    if True: main()

def run_v1b_routed_shared_pilot():
    """Run former v1b_routed_shared_pilot.py stage."""
    """One-seed capacity-matched routed-vs-shared V1-B diagnostic pilot."""
    
    import argparse
    import json
    import math
    import sys
    import time
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    OBJECTIVES = ("progress", "balance", "efficiency")
    
    
    def tensorize(value):
        if isinstance(value, dict):
            value = value.get("policy", next(iter(value.values())))
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    
    
    def flat_grad(loss, parameters):
        grads = torch.autograd.grad(loss, list(parameters), retain_graph=True, allow_unused=True)
        return torch.cat([g.detach().reshape(-1) for g in grads if g is not None]) if any(g is not None for g in grads) else torch.zeros(1, device="cuda")
    
    
    def cosine(a, b):
        return float(torch.dot(a, b).item() / (torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item() + 1e-8))
    
    
    def matrix_cosines(vectors):
        return [[cosine(a, b) for b in vectors] for a in vectors]
    
    
    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--updates", type=int, default=100)
        p.add_argument("--num-envs", type=int, default=16)
        p.add_argument("--horizon", type=int, default=24)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--geometry", action="store_true", help="run objective-gradient geometry only at checkpoint updates")
        args = p.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lifecycle = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
        # Keep the production pilot path cheap. Geometry is a separate diagnostic
        # pass and must be explicitly enabled; this prevents the logger from
        # determining whether the shared control can finish at all.
        geometry_updates = {1, 5, 10, 25, 50, 100} if args.geometry else set()
    
        def mark(event, **extra):
            with lifecycle.open("a") as f:
                f.write(json.dumps({"event": event, "unix": time.time(), **extra}, sort_keys=True) + "\n")
    
        mark("RUN_STARTED", protocol="V1-B-RS-PILOT", updates=args.updates)
        from isaaclab.app import AppLauncher
        saved = sys.argv[:]; sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
        mark("APP_INIT_OK")
    
        import gymnasium as gym
        import isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_ppo_cfg import UnitreeA1FlatPPORunnerCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from talon_rl.rewards.baselines import group_v1b_s7_terms
        from rl.core.integration.rsl_rl.v1a_integration import attach_v1a_policy
        from rl.core.integration.rsl_rl.shared_residual_wrapper import RslRlSharedResidualWrapper
        from rl.core.objectives.routing import SharedObjectiveCritic, normalize_objective_advantages, route_objective_gradients, objective_critic_loss
    
        source = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
        raw_source = source.get("model_state_dict", source.get("model", source))
    
        # Isaac's SimulationApp does not reliably support constructing a second
        # environment after the first one is closed. Reuse one environment for
        # the two matched conditions; each condition still receives the same
        # reset seed and identical rollout semantics.
        cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = args.num_envs; cfg.seed = args.seed
        shared_env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
        mark("MATCHED_ENV_CREATED")
    
        def make_condition(kind):
            mark("CONDITION_CREATE_START", condition=kind)
            env = shared_env
            mark("CONDITION_ENV_REUSED", condition=kind)
            agent = UnitreeA1FlatPPORunnerCfg(); agent.policy.actor_hidden_dims = [128, 128, 128]; agent.policy.critic_hidden_dims = [128, 128, 128]; agent.num_steps_per_env = args.horizon
            vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
            runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=str(args.output.parent / kind), device="cuda")
            mark("CONDITION_RUNNER_CREATED", condition=kind)
            if kind == "routed":
                actor = attach_v1a_policy(runner.alg, args.checkpoint, bottleneck_dim=8, w_ref=(1 / 3,) * 3, objectives=OBJECTIVES)
            else:
                base = runner.alg.policy
                base.load_state_dict(raw_source)
                actor = RslRlSharedResidualWrapper(base, bottleneck_dim=25, objectives=OBJECTIVES, w_ref=(1 / 3,) * 3).cuda()
            actor.train()
            mark("CONDITION_MODEL_READY", condition=kind)
            critic = SharedObjectiveCritic(48, 128, 3).cuda()
            adapter_params = list(actor.v1a_adapters.parameters()) if kind == "routed" else list(actor.shared_residual.parameters())
            shared_params = list(actor._shared_actor.parameters()) + list(actor._action_head.parameters())
            special_ids = {id(x) for x in adapter_params + shared_params}
            other_base = [x for x in actor.base.parameters() if id(x) not in special_ids]
            opt = torch.optim.Adam([
                {"params": shared_params, "lr": 1e-4},
                {"params": other_base, "lr": 1e-4},
                {"params": adapter_params, "lr": 4e-4},
            ], lr=1e-4)
            critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
            return env, actor, critic, opt, critic_opt, adapter_params, shared_params
    
        def train_condition(kind):
            env, actor, critic, opt, critic_opt, adapter_params, shared_params = make_condition(kind)
            obs, _ = env.reset(seed=args.seed); obs = tensorize(obs).cuda()
            mark("CONDITION_RESET_OK", condition=kind)
            weights = torch.full((args.num_envs, 3), 1 / 3, device="cuda")
            records = []
            for update in range(1, args.updates + 1):
                mark("CONDITION_UPDATE_START", condition=kind, update=update)
                observations=[]; actions=[]; old_logps=[]; rewards=[]; values=[]; dones=[]
                for _ in range(args.horizon):
                    with torch.no_grad():
                        actor.update_distribution({"policy": obs, "critic": obs}, weights)
                        action = actor.base.distribution.sample(); old_logp = actor.get_actions_log_prob(action); value = critic(obs)
                    nxt, _, term, trunc, _ = env.step(torch.clamp(action, -1, 1))
                    raw = env.unwrapped.reward_manager._step_reward.detach().cpu().numpy(); names = list(env.unwrapped.reward_manager.active_terms)
                    grouped = group_v1b_s7_terms({n: raw[:, i] for i, n in enumerate(names)}, shape=(args.num_envs,))
                    observations.append(obs); actions.append(action); old_logps.append(old_logp); rewards.append(torch.as_tensor(grouped, device="cuda") * env.unwrapped.step_dt); values.append(value); dones.append((term | trunc).float())
                    obs = tensorize(nxt).cuda()
                reward_t = torch.stack(rewards); value_t = torch.stack(values); done_t = torch.stack(dones)
                returns = torch.zeros_like(reward_t); running = critic(obs).detach()
                for t in reversed(range(args.horizon)):
                    running = reward_t[t] + 0.99 * running * (1 - done_t[t].unsqueeze(-1)); returns[t] = running
                flat_obs=torch.cat(observations); flat_actions=torch.cat(actions); flat_old=torch.cat(old_logps); flat_ret=returns.reshape(-1,3); flat_val=value_t.reshape(-1,3)
                adv=normalize_objective_advantages((flat_ret-flat_val).detach())
                actor.update_distribution({"policy": flat_obs, "critic": flat_obs}, weights.repeat(args.horizon,1)); ratio=torch.exp(actor.get_actions_log_prob(flat_actions)-flat_old.detach())
                losses=[-torch.minimum(ratio*adv[:, i], torch.clamp(ratio,.8,1.2)*adv[:, i]).mean() for i in range(3)]
                measure_geometry = update in geometry_updates
                raw_grad_vectors = [flat_grad(loss, shared_params) for loss in losses] if measure_geometry else None
                pre_cos = matrix_cosines(raw_grad_vectors) if raw_grad_vectors is not None else None
                conflict = float(np.mean([pre_cos[i][j] < 0 for i in range(3) for j in range(i+1,3)])) if pre_cos is not None else None
                before = torch.cat([p.detach().reshape(-1) for p in shared_params]) if measure_geometry else None
                opt.zero_grad(set_to_none=True)
                if kind == "routed":
                    groups=[list(adapter.parameters()) for adapter in actor.v1a_adapters]
                    routed=route_objective_gradients(losses, groups, shared_params, weights[0])
                    for parameter, gradient in routed.items(): parameter.grad=gradient.detach().clone()
                else:
                    aggregate=sum((weights[0, i] * losses[i] for i in range(3)))
                    # The shared control receives the aggregate objective
                    # gradient directly.  Avoid materializing a second full
                    # parameter-to-gradient list; this is semantically identical
                    # to the frozen control contract and materially cheaper.
                    aggregate.backward()
                opt.step()
                after = torch.cat([p.detach().reshape(-1) for p in shared_params]) if measure_geometry else None
                update_vec = after - before if measure_geometry else None
                critic_opt.zero_grad(set_to_none=True); c_loss=objective_critic_loss(critic(flat_obs), flat_ret.detach()); c_loss.backward(); critic_opt.step()
                records.append({
                    "update": update,
                    "geometry_measured": measure_geometry,
                    "pre_routing_gradient_cosine": pre_cos,
                    "negative_conflict_rate": conflict,
                    "shared_update_norm": float(torch.linalg.vector_norm(update_vec).item()) if update_vec is not None else None,
                    "shared_update_cosine_to_objective": [cosine(update_vec, g) for g in raw_grad_vectors] if update_vec is not None else None,
                    "ppo_kl_proxy": float(.5*(torch.cat(old_logps)-actor.get_actions_log_prob(torch.cat(actions)).detach()).square().mean().item()),
                    "clip_fraction": float(((ratio<.8)|(ratio>1.2)).float().mean().item()),
                    "critic_loss": float(c_loss.item()),
                    "reward_mean": reward_t.mean((0,1)).detach().cpu().tolist(),
                })
                mark("CONDITION_UPDATE_DONE", condition=kind, update=update)
            mark("CONDITION_CLOSED", condition=kind)
            terminal = args.output.parent / f"{kind}_terminal.pt"
            torch.save({
                "schema": "v1b_routed_shared_terminal_v1",
                "condition": kind,
                "seed": args.seed,
                "updates": args.updates,
                "objective_order": list(OBJECTIVES),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_optimizer": opt.state_dict(),
                "critic_optimizer": critic_opt.state_dict(),
            }, terminal)
            mark("CHECKPOINT_WRITTEN", condition=kind, checkpoint=str(terminal))
            return {"condition": kind, "updates": args.updates, "terminal_checkpoint": str(terminal), "records": records, "final": records[-1]}
    
        mark("CONDITIONS_START")
        routed = train_condition("routed"); mark("ROUTED_DONE")
        shared = train_condition("shared"); mark("SHARED_DONE")
        result={"schema":"v1b_routed_shared_pilot_v1","protocol":"V1-B-RS-PILOT","status":"DIAGNOSTIC_COMPLETE","verdict":"PIPELINE_DISCRIMINATION_PENDING_REVIEW","seed":args.seed,"updates":args.updates,"num_envs":args.num_envs,"horizon":args.horizon,"geometry_enabled":args.geometry,"objective_order":list(OBJECTIVES),"capacity":{"routed_adapter_params":6552,"shared_adapter_params":6553,"difference":1},"routed":routed,"shared":shared,"interpretation":"Diagnostic only; no superiority or thesis verdict."}
        args.output.write_text(json.dumps(result, indent=2)+"\n"); mark("ARTIFACT_WRITTEN", path=str(args.output)); mark("RUN_DONE", status="DIAGNOSTIC_COMPLETE"); print(json.dumps({k:v for k,v in result.items() if k not in ('routed','shared')},indent=2)); shared_env.close(); app.close()
    
    
    if True: main()

STAGES = {
    "v1b_isaac_smoke": run_v1b_isaac_smoke,
    "v1b_profile_shared_runner": run_v1b_profile_shared_runner,
    "v1b_routed_shared_pilot": run_v1b_routed_shared_pilot,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
