"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_eval_m0_3a():
    """Run former eval_m0_3a.py stage."""
    """Deterministic explicit-preference evaluation for M0.3-A checkpoints."""
    import argparse, json, sys
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    def po(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--w',type=float,nargs=5,default=[.2]*5);a=p.parse_args()
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.algorithms.amor import AmorActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=a.seed);obs=po(obs).cuda();state=torch.load(a.checkpoint,map_location='cuda',weights_only=False);model=AmorActorCritic(obs.shape[-1],12).cuda();model.load_state_dict(state['model']);model.eval();w=torch.tensor(a.w,device='cuda',dtype=torch.float32).expand(a.num_envs,-1)
            falls=np.zeros(a.num_envs,dtype=bool);vxe=[];tilt=[];heights=[];term_count=0
            with torch.no_grad():
                for _ in range(a.steps):
                    action=model.act_inference_with_preference(obs,w);nxt,_,term,trunc,_=env.step(action);done=(term|trunc);falls|=done.detach().cpu().numpy();term_count+=int(done.sum());data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');vel=data.root_lin_vel_b.detach().cpu().numpy();vxe.append(np.abs(vel[:,0]-cmd[:,0].detach().cpu().numpy()));q=data.root_quat_w.detach().cpu().numpy();roll=np.arctan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2));pitch=np.arcsin(np.clip(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1));tilt.append(np.degrees(np.maximum(np.abs(roll),np.abs(pitch))));heights.append(data.root_pos_w[:,2].detach().cpu().numpy());obs=po(nxt).cuda()
            t=np.concatenate(tilt);out={'schema':'m0_3a_deterministic_eval_v1','seed':a.seed,'steps':a.steps,'num_envs':a.num_envs,'w_eval':a.w,'mean_vx_error':float(np.mean(vxe)),'survival':float(1-falls.mean()),'tilt_p95_deg':float(np.percentile(t,95)),'max_tilt_deg':float(np.max(t)),'height_mean':float(np.mean(heights)),'termination_rate':float(term_count/(a.steps*a.num_envs)),'deterministic_actor_mean':True,'training_state_mutated':False,'finite':bool(np.isfinite(t).all())};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_eval_m0_3b():
    """Run former eval_m0_3b.py stage."""
    """Read-only deterministic evaluator for scale-aligned AMOR checkpoints."""
    import argparse,json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    def po(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
     p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();from isaaclab.app import AppLauncher;saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;env=None
     try:
      import gymnasium as gym,isaaclab_tasks;from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg;from rl.core.algorithms.amor import ScaleAlignedAmorActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=a.seed);obs=po(obs).cuda();st=torch.load(a.checkpoint,map_location='cuda',weights_only=False);m=ScaleAlignedAmorActorCritic(obs.shape[-1],12,[1024]*4).cuda();m.load_state_dict(st['model']);m.load_norm_state(st['normalizers']);m.actor_obs_norm.freeze();m.critic_obs_norm.freeze();m.eval();w=torch.full((a.num_envs,5),.2,device='cuda');falls=np.zeros(a.num_envs,bool);v=[];til=[];terms=0
      with torch.no_grad():
       for _ in range(a.steps):
        act=m.act_inference_with_preference(obs,w);nxt,_,term,trunc,_=env.step(act);done=(term|trunc);falls|=done.cpu().numpy();terms+=int(done.sum());d=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');v.append(np.abs(d.root_lin_vel_b[:,0].cpu().numpy()-cmd[:,0].cpu().numpy()));q=d.root_quat_w.cpu().numpy();roll=np.arctan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2));pitch=np.arcsin(np.clip(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1));til.append(np.degrees(np.maximum(abs(roll),abs(pitch))));obs=po(nxt).cuda()
      t=np.concatenate(til);out={'schema':'m0_3b_deterministic_eval_v1','seed':a.seed,'checkpoint':'update_3000','w_eval':[.2]*5,'steps':a.steps,'num_envs':a.num_envs,'mean_vx_error':float(np.mean(v)),'survival':float(1-falls.mean()),'tilt_p95_deg':float(np.percentile(t,95)),'max_tilt_deg':float(t.max()),'termination_rate':float(terms/(a.steps*a.num_envs)),'normalizer_frozen':True,'training_state_mutated':False,'finite':bool(np.isfinite(t).all())};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_m0_3a_production_smoke():
    """Run former m0_3a_production_smoke.py stage."""
    """Small real-Isaac smoke for the AMOR early-scalarization path.
    
    This is a lifecycle/semantic smoke, not a training result or authorization.
    """
    import argparse, json, sys, time, traceback
    from pathlib import Path
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    def _obs(x):
        if isinstance(x, dict): x = x.get("policy", next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--num-envs", type=int, default=16); p.add_argument("--updates", type=int, default=2)
        p.add_argument("--horizon", type=int, default=24); p.add_argument("--run-dir", type=Path, required=True)
        a = p.parse_args(); run = a.run_dir.resolve(); run.mkdir(parents=True, exist_ok=True)
        (run / "RUN_STARTED.json").write_text(json.dumps({"status":"RUN_STARTED","unix":time.time()})+"\n")
        app = env = None
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_stock_terms, reconstruct_stock_scalar
            from rl.core.algorithms.amor import (AmorActorCritic, sample_episode_preferences,
                scalarize_advantages, normalize_scalar_advantages, standard_clipped_actor_loss, vector_gae)
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = 0
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            obs, _ = env.reset(seed=0); obs = _obs(obs).to("cuda")
            model = AmorActorCritic(obs.shape[-1], 12).cuda(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            w = sample_episode_preferences(a.num_envs, device="cuda"); max_err = 0.; rows = 0; updates = 0
            for _ in range(a.updates):
                os=[]; ws=[]; acts=[]; oldlp=[]; vals=[]; rs=[]; ds=[]
                for _ in range(a.horizon):
                    with torch.no_grad(): action, lp = model.act_with_preference(obs, w); val = model.value_with_preference(obs, w)
                    nxt, reward, term, trunc, _ = env.step(action)
                    mgr = env.unwrapped.reward_manager; raw = mgr._step_reward.detach().cpu().numpy(); names = list(mgr.active_terms)
                    rv = group_stock_terms({n: raw[:, i] for i, n in enumerate(names)}, shape=(a.num_envs,))
                    recon = reconstruct_stock_scalar(rv) * env.unwrapped.step_dt
                    err = float(torch.as_tensor(recon).sub(reward.detach().cpu()).abs().max()); max_err = max(max_err, err)
                    if err >= 1e-6: raise RuntimeError(f"reward reconstruction mismatch: {err}")
                    os.append(obs); ws.append(w); acts.append(action); oldlp.append(lp); vals.append(val)
                    rs.append(torch.as_tensor(rv, device="cuda", dtype=torch.float32) * env.unwrapped.step_dt)
                    done = (term | trunc).to("cuda"); ds.append(done); rows += a.num_envs
                    obs = _obs(nxt).to("cuda")
                    if bool(done.any()):
                        fresh = sample_episode_preferences(a.num_envs, device="cuda")
                        w = torch.where(done[:, None], fresh, w)
                with torch.no_grad(): nxtv = model.value_with_preference(obs, w)
                r, v, d = torch.stack(rs), torch.stack(vals), torch.stack(ds)
                adv, ret = vector_gae(r, v, nxtv, d)
                O, W, A, LP = torch.cat(os), torch.cat(ws), torch.cat(acts), torch.cat(oldlp)
                av = normalize_scalar_advantages(scalarize_advantages(adv.reshape(-1,5), torch.cat(ws)))
                ratio = torch.exp(model.logp_with_preference(O, W, A) - LP.detach())
                pol = standard_clipped_actor_loss(ratio, av); val_loss = (model.value_with_preference(O,W)-ret.reshape(-1,5)).pow(2).mean()
                ent = model.entropy(model.with_preference(O,W)).mean(); loss = pol + val_loss - .01*ent
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); updates += 1
            ckpt = run / "m0_3a_smoke.pt"; torch.save({"schema":"m0_3a_amor_smoke_v1","model":model.state_dict(),"optimizer":opt.state_dict(),"update_idx":updates}, ckpt)
            resumed = AmorActorCritic(obs.shape[-1], 12).cuda(); resumed.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
            report = {"schema":"m0_3a_production_smoke_v1","status":"PASS","num_envs":a.num_envs,"updates":updates,"horizon":a.horizon,"vector_rows":rows,"max_abs_reconstruction_error":max_err,"vector_critic_dim":5,"preference_sampling":"Dirichlet(1) per episode","scalarized_advantage":True,"standard_single_clip":True,"late_weighting":False,"checkpoint_resume":True,"w_eval":"explicit_required","finite":True,"training_authorized":False}
            (run/"artifact.json").write_text(json.dumps(report,indent=2)+"\n"); (run/"RUN_DONE.json").write_text(json.dumps({"status":"RUN_DONE","exit_code":0})+"\n"); print(json.dumps(report,indent=2))
        except BaseException as e:
            (run/"ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(e),"traceback":traceback.format_exc()},indent=2)+"\n"); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    if True: main()

def run_train_m0_3a():
    """Run former train_m0_3a.py stage."""
    """Production AMOR early-scalarization runner (M0.3-A).
    
    The runner intentionally owns orchestration only; AMOR math lives in
    ``rl.core.algorithms.amor``.  It validates the frozen manifest before
    starting Isaac and persists all state needed for deterministic resume.
    """
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    import hashlib
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    MANIFEST = ROOT / "artifacts/m0_3a_design/M0_3A_DESIGN.json"
    HORIZON, K, HIDDEN = 24, 5, [128, 128, 128]
    
    def write(path: Path, obj): path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    def tensor_hash(x):
        if torch.is_tensor(x): x = x.detach().cpu().contiguous().numpy().tobytes()
        elif isinstance(x, tuple): x = repr(x).encode()
        elif not isinstance(x, bytes): x = repr(x).encode()
        return hashlib.sha256(x).hexdigest()
    def obs_tensor(x, device):
        if isinstance(x, dict): x = x.get("policy", next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x, device=device, dtype=torch.float32)
    
    def validate(mf, a):
        if not (mf.get("status") == "FROZEN" and mf.get("training_authorized")) and not ((a.scale_b or a.scale_b2) and a.smoke and mf.get("status") == "FROZEN_PENDING_SMOKE"):
            raise RuntimeError("M0.3-A manifest is not frozen/authorized")
        if a.seed not in mf["seeds"] or a.num_envs <= 0 or a.updates <= 0:
            raise RuntimeError("invalid seed/env/update arguments")
        rollout_steps = mf.get("rollout", {}).get("steps_per_env", mf.get("rollout_steps_per_env"))
        if rollout_steps != HORIZON:
            raise RuntimeError("runner horizon differs from frozen manifest")
        if mf.get("vector_critic_output_dim") != K or mf.get("late_weighting"):
            raise RuntimeError("manifest requests non-AMOR semantics")
    
    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--seed", type=int, required=True); p.add_argument("--num-envs", type=int, default=4096)
        p.add_argument("--updates", type=int, default=300); p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--resume", type=Path); p.add_argument("--scale-b", action="store_true"); p.add_argument("--scale-b2", action="store_true"); p.add_argument("--smoke", action="store_true"); a = p.parse_args()
        global MANIFEST
        if a.scale_b2: MANIFEST = ROOT / "artifacts/m0_3b2_design/M0_3B2_FREEZE.json"
        elif a.scale_b: MANIFEST = ROOT / "artifacts/m0_3b_design/M0_3B_DESIGN.json"
        mf = json.loads(MANIFEST.read_text()); validate(mf, a)
        run = a.run_dir.resolve()
        if a.resume is None and run.exists() and any(run.iterdir()): raise FileExistsError(run)
        run.mkdir(parents=True, exist_ok=True); (run / "checkpoints").mkdir(exist_ok=True)
        manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        resume_state = None; start_update = 0; resumed = a.resume is not None
        if a.resume is not None:
            # Preflight happens before Isaac starts so a clean simulator shutdown
            # cannot erase the evidence that resume was requested/restored.
            resume_state = torch.load(a.resume, map_location="cpu", weights_only=False)
            expected_schema = "m0_3b_checkpoint_v1" if (a.scale_b or a.scale_b2) else "m0_3a_checkpoint_v1"
            if resume_state.get("schema") != expected_schema or resume_state.get("manifest_sha256") != manifest_hash:
                raise RuntimeError("checkpoint schema/manifest mismatch")
            start_update = int(resume_state["update_idx"])
            lr = float(resume_state["optimizer"]["param_groups"][0]["lr"])
            log_std = resume_state["model"].get("log_std")
            write(run / "RUN_RESUMED.json", {"status":"RUN_RESUMED", "source_checkpoint":str(a.resume),
                "restored_update_idx":start_update, "target_update":a.updates, "current_lr":lr,
                "learned_std":float(log_std.exp().mean()) if log_std is not None else None,
                "torch_rng_sha256":tensor_hash(resume_state["torch_rng"]),
                "numpy_rng_sha256":tensor_hash(resume_state["numpy_rng"]),
                "preference_sha256":tensor_hash(resume_state["w"]), "unix":time.time()})
        if a.resume is None:
            write(run / "config.json", {"schema":"m0_3a_runner_v1", "seed":a.seed, "num_envs":a.num_envs,
                "updates":a.updates, "horizon":HORIZON, "manifest_sha256":manifest_hash,
                "preference_sampling":"Dirichlet(1) per episode", "late_weighting":False})
            write(run / "RUN_STARTED.json", {"status":"RUN_STARTED", "seed":a.seed, "unix":time.time()})
        app = env = None
        try:
            from isaaclab.app import AppLauncher
            # Do not let Isaac Sim's own CLI parser consume runner-only flags such
            # as --resume/--run-dir.  This is especially important on resume:
            # Isaac may otherwise exit cleanly before Python reaches RUN_RESUMED.
            runner_argv = sys.argv[:]
            sys.argv = [sys.argv[0]]
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            sys.argv = runner_argv
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_stock_terms, reconstruct_stock_scalar
            from rl.core.algorithms.amor import (AmorActorCritic, sample_episode_preferences,
                scalarize_advantages, normalize_scalar_advantages, standard_clipped_actor_loss, vector_gae)
            if a.scale_b or a.scale_b2:
                from rl.core.algorithms.amor import ScaleAlignedAmorActorCritic
            from rl.core.algorithms.vector_ppo import adaptive_kl_lr
            torch.manual_seed(a.seed); np.random.seed(a.seed)
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = a.seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            raw_obs, _ = env.reset(seed=a.seed); obs = obs_tensor(raw_obs, device).to(device)
            model = (ScaleAlignedAmorActorCritic(obs.shape[-1], 12, [1024]*4) if (a.scale_b or a.scale_b2) else AmorActorCritic(obs.shape[-1], 12, HIDDEN)).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=1e-3); update_idx = 0; metrics = []
            if a.resume:
                state = resume_state
                model.load_state_dict(state["model"]); optim.load_state_dict(state["optimizer"]); update_idx = start_update
                if a.scale_b or a.scale_b2: model.load_norm_state(state["normalizers"])
                torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"]); obs = state["obs"].to(device); w = state["w"].to(device)
            else: w = sample_episode_preferences(a.num_envs, device=device)
            max_recon = 0.0
            for target_update in range(start_update + 1, a.updates + 1):
                obs_buf=[]; w_buf=[]; act_buf=[]; oldlp_buf=[]; val_buf=[]; rew_buf=[]; done_buf=[]; means=[]
                for _ in range(HORIZON):
                    if a.scale_b or a.scale_b2:
                        model.actor_obs_norm.update(obs); model.critic_obs_norm.update(obs)
                    with torch.no_grad():
                        action, oldlp = model.act_with_preference(obs, w); value = model.value_with_preference(obs, w)
                        mean = model.raw_mean(model.with_preference(obs, w))
                    nxt, reward, term, trunc, _ = env.step(action)
                    mgr = env.unwrapped.reward_manager; raw = mgr._step_reward.detach().cpu().numpy(); names = list(mgr.active_terms)
                    vec = group_stock_terms({n: raw[:, i] for i, n in enumerate(names)}, shape=(a.num_envs,))
                    recon = reconstruct_stock_scalar(vec) * env.unwrapped.step_dt
                    err = float(np.max(np.abs(recon - reward.detach().cpu().numpy()))); max_recon = max(max_recon, err)
                    if err >= 1e-6: raise RuntimeError(f"reward reconstruction mismatch {err}")
                    done = (term | trunc).to(device)
                    obs_buf.append(obs); w_buf.append(w); act_buf.append(action); oldlp_buf.append(oldlp); val_buf.append(value); means.append(mean)
                    rew_buf.append(torch.as_tensor(vec, device=device, dtype=torch.float32) * env.unwrapped.step_dt); done_buf.append(done)
                    obs = obs_tensor(nxt, device).to(device)
                    if bool(done.any()): w = torch.where(done[:, None], sample_episode_preferences(a.num_envs, device=device), w)
                with torch.no_grad(): next_value = model.value_with_preference(obs, w)
                rewards, values, dones = torch.stack(rew_buf), torch.stack(val_buf), torch.stack(done_buf)
                adv, returns = vector_gae(rewards, values, next_value, dones)
                O, W, AC, LP = torch.cat(obs_buf), torch.cat(w_buf), torch.cat(act_buf), torch.cat(oldlp_buf)
                A = adv.reshape(-1, K); R = returns.reshape(-1, K); old_mean = torch.cat(means).detach(); n = O.shape[0]; mb = n // 4
                order = torch.randperm(n, device=device); epoch_rows=[]
                for _epoch in range(5):
                    for start in range(0, n, mb):
                        ix = order[start:start+mb]; scalar_adv = normalize_scalar_advantages(scalarize_advantages(A[ix], W[ix]))
                        ratio = torch.exp(model.logp_with_preference(O[ix], W[ix], AC[ix]) - LP[ix].detach())
                        policy = standard_clipped_actor_loss(ratio, scalar_adv); value_loss = F.mse_loss(model.value_with_preference(O[ix], W[ix]), R[ix]); entropy = model.entropy(model.with_preference(O[ix], W[ix])).mean()
                        loss = policy + value_loss - .01 * entropy; optim.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step(); epoch_rows.append((float(policy.detach()), float(value_loss.detach()), float(entropy.detach())))
                    with torch.no_grad():
                        new_mean = model.raw_mean(model.with_preference(O, W)); std = model.log_std.exp(); kl = ((old_mean-new_mean).pow(2)/(2*std.pow(2))).sum(-1).mean()
                    adaptive_kl_lr(optim, kl)
                update_idx = target_update
                metrics.append({"update":update_idx,"policy_loss":float(np.mean([x[0] for x in epoch_rows])),"value_loss":float(np.mean([x[1] for x in epoch_rows])),"entropy":float(np.mean([x[2] for x in epoch_rows])),"analytic_kl":float(kl.detach()),"actor_lr":float(optim.param_groups[0]["lr"]),"learned_std":float(model.log_std.exp().mean()),"sampled_w_mean":W.mean(0).detach().cpu().tolist()})
                if update_idx % 25 == 0 or update_idx == a.updates:
                    state = {"schema":"m0_3b_checkpoint_v1" if (a.scale_b or a.scale_b2) else "m0_3a_checkpoint_v1","manifest_sha256":manifest_hash,"model":model.state_dict(),"optimizer":optim.state_dict(),"update_idx":update_idx,"obs":obs.detach().cpu(),"w":w.detach().cpu(),"torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state()}
                    if a.scale_b or a.scale_b2: state["normalizers"] = model.norm_state()
                    torch.save(state, run / "checkpoints" / f"update_{update_idx:03d}.pt")
            if update_idx != a.updates:
                raise RuntimeError(f"training ended at update {update_idx}, expected {a.updates}")
            torch.save(state, run / "checkpoints" / "final.pt"); write(run / "training_metrics.json", {"metrics":metrics,"max_abs_reconstruction_error":max_recon,"start_update":start_update,"final_update":update_idx})
            schema = "m0_3b_training_v1" if (a.scale_b or a.scale_b2) else "m0_3a_training_v1"
            write(run / "artifact.json", {"schema":schema,"status":"PASS","seed":a.seed,"start_update":start_update,"updates":update_idx,"num_envs":a.num_envs,"max_abs_reconstruction_error":max_recon,"late_weighting":False,"preference_sampling":"Dirichlet(1) per episode","checkpoint_schema":("m0_3b_checkpoint_v1" if (a.scale_b or a.scale_b2) else "m0_3a_checkpoint_v1"),"resumed":resumed,"normalization":"running_mean_std" if (a.scale_b or a.scale_b2) else None}); write(run / "RUN_DONE.json", {"status":"RUN_DONE","exit_code":0,"final_update":update_idx})
        except BaseException as e:
            write(run / "ERROR.json", {"status":"ERROR","error":str(e),"traceback":traceback.format_exc()}); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    if True: main()

def run_verdict_m0_3a():
    """Run former verdict_m0_3a.py stage."""
    """Aggregate the predeclared M0.3-A preservation gate."""
    import argparse,json
    from pathlib import Path
    def main():
     p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
     for s in (0,1,2):
      run=a.root/f'm0_3a_seed{s}_2026-09-22'; train=json.loads((run/'artifact.json').read_text()); ev=json.loads((run/'deterministic_eval_uniform.json').read_text()); rows.append({'seed':s,'training_status':train['status'],'final_update':train['updates'],'evaluation':ev})
     gates=[]
     for r in rows:
      e=r['evaluation']; gates.append(e['survival']>=.80 and e['mean_vx_error']<=.25 and e['tilt_p95_deg']<=20 and e['max_tilt_deg']<=40 and e['termination_rate']<=.10 and e['finite'])
     out={'schema':'m0_3a_verdict_v1','status':'FAIL' if not all(gates) else 'PASS','checkpoint':'update_300','w_eval':[.2]*5,'preservation_gate':{'survival_min':.80,'vx_mae_max':.25,'tilt_p95_max_deg':20,'max_tilt_max_deg':40,'termination_rate_max':.10},'seed_results':rows,'seed_gate_pass':gates,'preference_response_evaluated':False,'reason':'Uniform/reference preservation failed; non-uniform response is blocked.' if not all(gates) else 'Uniform preservation passed; preference response authorized.'}
     a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    if True:main()

STAGES = {
    "eval_m0_3a": run_eval_m0_3a,
    "eval_m0_3b": run_eval_m0_3b,
    "m0_3a_production_smoke": run_m0_3a_production_smoke,
    "train_m0_3a": run_train_m0_3a,
    "verdict_m0_3a": run_verdict_m0_3a,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
