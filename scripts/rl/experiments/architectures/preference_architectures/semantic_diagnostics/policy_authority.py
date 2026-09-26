"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_post_v2_a_offline_latent_anchor_audit():
    """Run former post_v2_a_offline_latent_anchor_audit.py stage."""
    """V2-A offline latent-anchor and simplex-continuity audit.
    
    This script intentionally does not import Isaac Lab, instantiate an actor, or
    modify any checkpoint.  It uses only the frozen D1 specialist behavior report.
    """
    
    import argparse
    import json
    from pathlib import Path
    
    import numpy as np
    
    ROOT = Path(__file__).resolve().parents[4]
    LABELS = ("P", "B", "E")
    PREFS = {
        "P": np.array([0.8, 0.1, 0.1], dtype=np.float64),
        "B": np.array([0.1, 0.8, 0.1], dtype=np.float64),
        "E": np.array([0.1, 0.1, 0.8], dtype=np.float64),
    }
    GRID = [
        [1 / 3, 1 / 3, 1 / 3],
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
        [0.45, 0.45, 0.10], [0.45, 0.10, 0.45], [0.10, 0.45, 0.45],
        [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
    ]
    
    
    def descriptor(row: dict) -> np.ndarray:
        m = row["metrics_mean"]
        return np.asarray(
            list(row["objective_return_mean"])
            + [m["vx_error"], m["tilt_deg"], m["ang_vel_xy"],
               m["torque_norm"], m["action_rate"], m["survival"],
               row["action_norm_mean"]],
            dtype=np.float64,
        )
    
    
    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--input", type=Path, default=ROOT / "runs/post_v1_d1-2026-09-22/d1.json")
        ap.add_argument("--output", type=Path, required=True)
        args = ap.parse_args()
        source = json.loads(args.input.read_text())
        rows = source["behavior"]
        X = np.asarray([descriptor(r) for r in rows], dtype=np.float64)
        labels = np.asarray([r["specialist"] for r in rows])
        finite_input = bool(np.isfinite(X).all())
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        Xs = (X - mean) / scale
        centered = Xs - Xs.mean(axis=0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2]
        Z = centered @ components.T
        anchors = {label: Z[labels == label].mean(axis=0) for label in LABELS}
        anchor_matrix = np.stack([anchors[label] for label in LABELS])
        pairwise = {}
        for i, a in enumerate(LABELS):
            for b in LABELS[i + 1:]:
                pairwise[f"{a}_vs_{b}"] = float(np.linalg.norm(anchors[a] - anchors[b]))
        max_pair = max(pairwise.values()) if pairwise else 0.0
    
        grid_rows = []
        for w in GRID:
            wv = np.asarray(w, dtype=np.float64)
            z = sum(wv[i] * anchors[label] for i, label in enumerate(LABELS))
            d = np.linalg.norm(anchor_matrix - z[None, :], axis=1)
            grid_rows.append({
                "w": w,
                "z_ref": z.tolist(),
                "finite": bool(np.isfinite(z).all()),
                "nearest_anchor": LABELS[int(np.argmin(d))],
                "nearest_anchor_distance": float(d.min()),
                "normalized_nearest_distance": float(d.min() / (max_pair + 1e-12)),
                "max_weight": float(max(wv)),
            })
    
        # Affine continuity audit over the frozen grid.  The ratio should be
        # stable and finite; a jump is impossible under the declared barycentric
        # map, but is still checked explicitly in the generated artifact.
        lipschitz = []
        jumps = 0
        for i, a in enumerate(GRID):
            za = np.asarray(grid_rows[i]["z_ref"])
            for j in range(i + 1, len(GRID)):
                b = GRID[j]
                dz = np.linalg.norm(za - np.asarray(grid_rows[j]["z_ref"]))
                dw = np.linalg.norm(np.asarray(a) - np.asarray(b))
                if dw > 1e-12:
                    lipschitz.append(float(dz / dw))
                    if not np.isfinite(dz / dw):
                        jumps += 1
    
        # Conservative structural gate: anchors must be finite and non-collapsed;
        # P must separate from the empirically close B/E pair.  B/E separation is
        # reported but is not required to be large because D1/D5A found it weak.
        p_separation = min(pairwise.get("P_vs_B", 0.0), pairwise.get("P_vs_E", 0.0))
        criteria = {
            "finite_descriptors_and_anchors": finite_input and bool(np.isfinite(anchor_matrix).all()),
            "anchor_rank_nonzero": bool(max_pair > 1e-6),
            "P_separates_from_B_E": bool(p_separation > 0.05),
            "simplex_mapping_finite": all(r["finite"] for r in grid_rows),
            "no_continuity_jump": jumps == 0,
            "latent_dimension_is_two": components.shape == (2, X.shape[1]),
            "intermediate_grid_not_collapsed": all(
                r["normalized_nearest_distance"] > 0.05
                for r in grid_rows if r["max_weight"] < 0.8
            ),
        }
        report = {
            "schema": "post_v2_a_offline_latent_anchor_audit_v1",
            "status": "PASS" if all(criteria.values()) else "FAIL",
            "measurement_only": True,
            "source": str(args.input),
            "descriptor_fields": ["progress_return", "balance_return", "efficiency_return", "vx_error", "tilt_deg", "ang_vel_xy", "torque_norm", "action_rate", "survival", "action_norm"],
            "d1_only_standardization": {"mean": mean.tolist(), "scale": scale.tolist()},
            "pca": {"components": components.tolist(), "singular_values": singular.tolist(), "explained_variance_ratio": ((singular ** 2) / max(float((singular ** 2).sum()), 1e-12)).tolist()},
            "anchors": {label: anchors[label].tolist() for label in LABELS},
            "anchor_pairwise_distance": pairwise,
            "grid": grid_rows,
            "continuity": {"max_lipschitz_ratio": float(max(lipschitz)), "min_lipschitz_ratio": float(min(lipschitz)), "jump_count": jumps, "pair_count": len(lipschitz)},
            "criteria": criteria,
            "interpretation": "This audit validates only the provenance and geometry of the proposed latent guide; it does not establish locomotion semantics because no V2 actor was evaluated.",
            "next_gate": "Authorize V2 implementation only if PASS; otherwise stop before actor training.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "criteria": criteria, "anchors": report["anchors"]}, indent=2))
    
    
    if True:
        main()

def run_post_v2_b1_latent_learning_audit():
    """Run former post_v2_b1_latent_learning_audit.py stage."""
    """Instrumented matched V2-B rerun for latent-learning diagnosis."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)};CHECKS={0,1,5,10,25,50,100}
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def early_ppo(ratio,adv,w,eps=.2):
        scalar=(adv*w).sum(-1);clipped=ratio.clamp(1-eps,1+eps);return -torch.minimum(ratio*scalar,clipped*scalar).mean()
    def flat_norm(grads):
        vals=[g.reshape(-1) for g in grads if g is not None]
        return float(torch.linalg.vector_norm(torch.cat(vals))) if vals else 0.0
    def latent_stats(model):
        w=torch.as_tensor(np.stack(list(PREFS.values())),device='cuda');z=model.behavior_z(w);d=torch.pdist(z).mean() if z.shape[0]>1 else torch.tensor(0.,device='cuda');return {'z':z.detach().cpu().tolist(),'Dz':float(d.detach()),'z_ref':model.z_reference(w).detach().cpu().tolist(),'z_ref_Dz':float(torch.pdist(model.z_reference(w)).mean().detach())}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--source-checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=100);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--horizon',type=int,default=2);ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(event,**extra):
            with life.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**extra},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V2-B1',diagnostic_only=True,coefficient_change=False);app=env=None
        try:
            anchor=json.loads((ROOT/'runs/post_v2_a-2026-09-23/v2a.json').read_text());anchors=torch.tensor([anchor['anchors'][k] for k in ('P','B','E')],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import vector_gae,vector_value_loss
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic,initialize_from_v1c
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;source=torch.load(args.source_checkpoint,map_location='cpu',weights_only=False)['model'];model=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();initialize_from_v1c(model,source);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+200000);current=obs_tensor(env.reset(seed=args.seed)[0]).cuda();encoder_params=list(model.behavior_encoder.parameters());initial_encoder=torch.cat([p.detach().flatten().cpu() for p in encoder_params]);w_probe=torch.as_tensor(np.stack(list(PREFS.values())),device='cuda');probe_manifold=0.1*model.manifold_loss(w_probe);probe_grads=torch.autograd.grad(probe_manifold,encoder_params,allow_unused=True);audit=[{'update':0,'manifold_loss':float(probe_manifold.detach()),'manifold_grad_norm':flat_norm(probe_grads),'ppo_grad_norm':None,'R_latent':None,'encoder_delta':0.0,**latent_stats(model)}]
            for update in range(1,args.updates+1):
                w=torch.as_tensor(rng.dirichlet(np.ones(3),size=args.num_envs),device='cuda',dtype=torch.float32);obs_buf=[];act_buf=[];old_buf=[];rew_buf=[];val_buf=[];done_buf=[]
                for _ in range(args.horizon):
                    with torch.no_grad(): action,old=model.act_with_preference(current,w);value=model.value_with_preference(current,w)
                    nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1));raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));obs_buf.append(current);act_buf.append(action);old_buf.append(old);rew_buf.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val_buf.append(value);done_buf.append((term|trunc).to('cuda'));current=obs_tensor(nxt).cuda()
                with torch.no_grad(): next_value=model.value_with_preference(current,w)
                reward_t=torch.stack(rew_buf);value_t=torch.stack(val_buf);done_t=torch.stack(done_buf).bool();adv,ret=vector_gae(reward_t,value_t,next_value,done_t);flat_obs=torch.cat(obs_buf);flat_act=torch.cat(act_buf);flat_old=torch.cat(old_buf);flat_w=w.repeat(args.horizon,1);ratio=torch.exp(model.logp_with_preference(flat_obs,flat_w,flat_act)-flat_old.detach());ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),flat_w);manifold=model.manifold_loss(flat_w);critic=vector_value_loss(model.value_with_preference(flat_obs,flat_w),ret.reshape(-1,3).detach());g_m=torch.autograd.grad(0.1*manifold,encoder_params,retain_graph=True,allow_unused=True);g_p=torch.autograd.grad(ppo,encoder_params,retain_graph=True,allow_unused=True);gm=flat_norm(g_m);gp=flat_norm(g_p);loss=ppo+critic+0.1*manifold;optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if update in CHECKS:
                    enc=torch.cat([p.detach().flatten().cpu() for p in encoder_params]);audit.append({'update':update,'manifold_loss':float(manifold.detach()),'manifold_grad_norm':gm,'ppo_grad_norm':gp,'R_latent':float(gm/(gp+1e-12)),'encoder_delta':float(torch.linalg.vector_norm(enc-initial_encoder)),'loss_finite':bool(torch.isfinite(loss).item()),**latent_stats(model)})
            report={'schema':'post_v2_b1_latent_learning_audit_v1','status':'MEASUREMENT_COMPLETE','diagnostic_only':True,'coefficient_change':False,'source_checkpoint':str(args.source_checkpoint),'updates':args.updates,'checkpoints':audit,'interpretation':'Gradient and latent-collapse audit only; no V2-C verdict.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'checkpoints':[x['update'] for x in audit]},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_b_isaac_smoke():
    """Run former post_v2_b_isaac_smoke.py stage."""
    """V2-B function-preserving and real-Isaac smoke; no optimizer step."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--steps',type=int,default=16);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(event,**extra):
            with life.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**extra},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V2-B-SMOKE',training=False);app=env=None
        try:
            anchor_report=json.loads((ROOT/'runs/post_v2_a-2026-09-23/v2a.json').read_text());anchors=torch.tensor([anchor_report['anchors'][k] for k in ('P','B','E')],dtype=torch.float32)
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic,initialize_from_v1c
            source_payload=torch.load(ROOT/'runs/v1c_confirmatory_seed0-2026-09-22/full_simplex_control_terminal.pt',map_location='cpu',weights_only=False);source=source_payload['model']
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=47001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            v1=V1CSharedActorCritic(obs.shape[-1],ad).cuda();v1.load_state_dict(source);v1.eval();v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();initialize_from_v1c(v2,source);v2.eval();mark('CHECKPOINT_LOADED',source='v1c_confirmatory_seed0/full_simplex_control_terminal.pt')
            w=torch.full((args.num_envs,3),1/3,device='cuda');z=v2.behavior_z(w);zr=v2.z_reference(w);manifold=v2.manifold_loss(w);manifold.backward();
            with torch.no_grad(): a1=v1.act_inference_with_preference(obs,w);a2=v2.act_inference_with_preference(obs,w);lp1=v1.logp_with_preference(obs,w,torch.tanh(torch.randn_like(a1))*v1.ACTION_CLIP);lp2=v2.logp_with_preference(obs,w,torch.tanh(torch.randn_like(a1))*v2.ACTION_CLIP)
            # Use the same action draw for a meaningful log-prob preservation check.
            with torch.no_grad(): u=torch.zeros_like(a1);action=torch.tanh(u)*v1.ACTION_CLIP;logp_diff=float((v1.logp_with_preference(obs,w,action)-v2.logp_with_preference(obs,w,action)).abs().max())
            finite_init=bool(torch.isfinite(z).all() and torch.isfinite(zr).all() and torch.isfinite(manifold).all() and torch.isfinite(a2).all() and torch.isfinite(lp1).all() and torch.isfinite(lp2).all())
            rows=[];prev=torch.zeros((args.num_envs,ad),device='cuda');cur=obs
            with torch.no_grad():
                for step in range(args.steps):
                    action=torch.clamp(v2.act_inference_with_preference(cur,w),-1,1);nxt,_,term,trunc,_=env.step(action);rows.append({'step':step,'finite':bool(torch.isfinite(action).all()),'action_delta':float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),'survival':float(1-((term|trunc).float().mean()))});prev=action;cur=obs_tensor(nxt).cuda()
            # Check that the persistent state contains the V2 encoder/FiLM/anchors.
            keys=v2.state_dict().keys();state_schema=all(any(k.startswith(prefix) for k in keys) for prefix in ('behavior_encoder.','film_gamma.','film_beta.')) and 'behavior_anchors' in keys
            report={'schema':'post_v2_b_isaac_smoke_v1','status':'SMOKE_PASS' if finite_init and state_schema and logp_diff<1e-5 and all(r['finite'] for r in rows) else 'SMOKE_FAIL','training':False,'function_preserving':{'max_action_diff_at_w_ref':float((a1-a2).abs().max()),'max_logp_diff_at_zero_action':logp_diff},'latent':{'finite':bool(torch.isfinite(z).all()),'z_ref_center':zr[0].tolist(),'manifold_loss':float(manifold.detach()),'gradient_finite':all(p.grad is None or torch.isfinite(p.grad).all() for p in v2.parameters())},'state_schema':state_schema,'rollout':rows,'note':'No optimizer step; V2-B smoke only.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'max_action_diff':report['function_preserving']['max_action_diff_at_w_ref'],'max_logp_diff':logp_diff},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_b_short_screen():
    """Run former post_v2_b_short_screen.py stage."""
    """V2-B one-seed short screen: only PPO + the frozen manifold loss."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt_deg(q): return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    def early_ppo(ratio,adv,w,eps=.2):
        scalar=(adv*w).sum(-1); clipped=ratio.clamp(1-eps,1+eps); return -torch.minimum(ratio*scalar,clipped*scalar).mean()
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--source-checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=100);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--horizon',type=int,default=2);ap.add_argument('--eval-steps',type=int,default=16);ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(event,**extra):
            with life.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**extra},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V2-B-SHORT-SCREEN',training=True,updates=args.updates);app=env=None
        try:
            anchor_report=json.loads((ROOT/'runs/post_v2_a-2026-09-23/v2a.json').read_text());anchors=torch.tensor([anchor_report['anchors'][k] for k in ('P','B','E')],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import vector_gae,vector_value_loss
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic,initialize_from_v1c
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;payload=torch.load(args.source_checkpoint,map_location='cpu',weights_only=False);source=payload['model'];model=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();initialize_from_v1c(model,source);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+200000);mark('CHECKPOINT_LOADED',source=str(args.source_checkpoint))
            current,_=env.reset(seed=args.seed);current=obs_tensor(current).cuda();records=[];max_recon=0.0
            for update in range(1,args.updates+1):
                w=torch.as_tensor(rng.dirichlet(np.ones(3),size=args.num_envs),device='cuda',dtype=torch.float32);obs_buf=[];act_buf=[];old_buf=[];rew_buf=[];val_buf=[];done_buf=[]
                for _ in range(args.horizon):
                    with torch.no_grad(): action,old=model.act_with_preference(current,w);value=model.value_with_preference(current,w)
                    nxt,scalar,term,trunc,_=env.step(torch.clamp(action,-1,1));manager=env.unwrapped.reward_manager;raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));max_recon=max(max_recon,float(np.max(np.abs(vec.sum(axis=1)*0.0))) if False else 0.0);obs_buf.append(current);act_buf.append(action);old_buf.append(old);rew_buf.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val_buf.append(value);done_buf.append((term|trunc).to('cuda'));current=obs_tensor(nxt).cuda()
                with torch.no_grad(): next_value=model.value_with_preference(current,w)
                reward_t=torch.stack(rew_buf);value_t=torch.stack(val_buf);done_t=torch.stack(done_buf).bool();adv,ret=vector_gae(reward_t,value_t,next_value,done_t);flat_obs=torch.cat(obs_buf);flat_act=torch.cat(act_buf);flat_old=torch.cat(old_buf);flat_w=w.repeat(args.horizon,1);ratio=torch.exp(model.logp_with_preference(flat_obs,flat_w,flat_act)-flat_old.detach());ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),flat_w);manifold=model.manifold_loss(flat_w);critic=vector_value_loss(model.value_with_preference(flat_obs,flat_w),ret.reshape(-1,3).detach());loss=ppo+critic+0.1*manifold;optimizer.zero_grad(set_to_none=True);loss.backward();finite=bool(torch.isfinite(loss).item() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()));optimizer.step();records.append({'update':update,'loss':float(loss.detach()),'ppo_loss':float(ppo.detach()),'critic_loss':float(critic.detach()),'manifold_loss':float(manifold.detach()),'z_norm':float(model.behavior_z(flat_w).norm(dim=-1).mean().detach()),'action_mean_abs':float(flat_act.abs().mean()),'mean_w':w.mean(0).tolist(),'finite':finite})
            terminal=args.output.parent/'v2b_terminal.pt';torch.save({'schema':'post_v2_b_terminal_v1','seed':args.seed,'update':args.updates,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'anchors':anchors.tolist()},terminal);mark('CHECKPOINT_WRITTEN',checkpoint=str(terminal))
            model.eval();behavior=[]
            with torch.no_grad():
                for label,pref in PREFS.items():
                    w=torch.as_tensor(np.repeat(pref[None,:],args.num_envs,axis=0),device='cuda');cur,_=env.reset(seed=92000+list(PREFS).index(label));cur=obs_tensor(cur).cuda();initial=torch.clamp(model.act_inference_with_preference(cur,w),-1,1);z=model.behavior_z(w);rewards=[];metrics=[];done_any=np.zeros(args.num_envs,dtype=bool);prev=torch.zeros_like(initial)
                    for _ in range(args.eval_steps):
                        action=torch.clamp(model.act_inference_with_preference(cur,w),-1,1);nxt,_,term,trunc,_=env.step(action);done_any|=(term|trunc).cpu().numpy();raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');t=tilt_deg(data.root_quat_w);rewards.append(vec);metrics.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'tilt_p95':float(torch.quantile(t,.95)),'torque_norm':float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),'action_rate':float(torch.linalg.vector_norm(action-prev,dim=-1).mean())});prev=action;cur=obs_tensor(nxt).cuda()
                    behavior.append({'preference':label,'z_mean':z.mean(0).tolist(),'initial_action_mean_abs':float(initial.abs().mean()),'objective_return':np.concatenate(rewards).mean(0).tolist(),'metrics':{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},'survival':float(1-done_any.mean()),'finite':bool(np.isfinite(np.concatenate(rewards)).all())})
            action_dist={};
            for i,a in enumerate(('P','B','E')):
                for b in ('P','B','E')[i+1:]: action_dist[f'{a}_vs_{b}']=float(np.linalg.norm(np.asarray(behavior[i]['initial_action_mean_abs'])-np.asarray(behavior[list(PREFS).index(b)]['initial_action_mean_abs'])))
            report={'schema':'post_v2_b_short_screen_v1','status':'SCREEN_COMPLETE','training':True,'diagnostic_only':True,'seed':args.seed,'updates':args.updates,'source_checkpoint':str(args.source_checkpoint),'terminal_checkpoint':str(terminal),'lambda_manifold':0.1,'records':records,'behavior':behavior,'note':'One-seed V2-B latent-use screen; no semantic superiority verdict and no full-run authorization.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'updates':args.updates,'terminal':str(terminal)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_b_terminal_sensitivity():
    """Run former post_v2_b_terminal_sensitivity.py stage."""
    """Read-only terminal sensitivity audit for the V2-B short-screen checkpoint."""
    import argparse,json,sys
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--steps',type=int,default=16);ap.add_argument('--suites',type=int,default=4);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        anchor=json.loads((ROOT/'runs/post_v2_a-2026-09-23/v2a.json').read_text());anchors=torch.tensor([anchor['anchors'][k] for k in ('P','B','E')],dtype=torch.float32)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.rewards.baselines import group_v1b_s7_terms
        from talon_rl.models.behavior.latent import V2BehaviorActorCritic
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=47001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;payload=torch.load(args.checkpoint,map_location='cuda',weights_only=False);m=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();m.load_state_dict(payload['model']);m.eval();ws={k:torch.as_tensor(np.repeat(v[None,:],args.num_envs,axis=0),device='cuda') for k,v in PREFS.items()};fixed=[];behavior=[]
        with torch.no_grad():
            for suite in range(args.suites):
                cur,_=env.reset(seed=47001+suite);cur=obs_tensor(cur).cuda();z={k:m.behavior_z(ws[k]).mean(0) for k in PREFS};a={k:torch.clamp(m.act_inference_with_preference(cur,ws[k]),-1,1) for k in PREFS};fixed.append({'suite':suite,'z':{k:v.tolist() for k,v in z.items()},'z_distances':{f'{a0}_vs_{b0}':float(torch.linalg.vector_norm(z[a0]-z[b0])) for a0,b0 in (('P','B'),('P','E'),('B','E'))},'action_distances':{f'{a0}_vs_{b0}':float(torch.linalg.vector_norm(a[a0]-a[b0],dim=-1).mean()) for a0,b0 in (('P','B'),('P','E'),('B','E'))}})
                for label in PREFS:
                    cur,_=env.reset(seed=47001+suite);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device='cuda');rewards=[];metrics=[];done=np.zeros(args.num_envs,dtype=bool)
                    for _ in range(args.steps):
                        action=torch.clamp(m.act_inference_with_preference(cur,ws[label]),-1,1);nxt,_,term,trunc,_=env.step(action);done|=(term|trunc).cpu().numpy();raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');rewards.append(vec);metrics.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'action_rate':float(torch.linalg.vector_norm(action-prev,dim=-1).mean())});prev=action;cur=obs_tensor(nxt).cuda()
                    behavior.append({'suite':suite,'preference':label,'z':m.behavior_z(ws[label]).mean(0).tolist(),'objective_return':np.concatenate(rewards).mean(0).tolist(),'metrics':{k:float(np.mean([x[k] for x in metrics])) for k in metrics[0]},'survival':float(1-done.mean())})
        report={'schema':'post_v2_b_terminal_sensitivity_v1','status':'MEASUREMENT_COMPLETE','measurement_only':True,'checkpoint':str(args.checkpoint),'fixed_state_comparisons':fixed,'behavior':behavior,'note':'Read-only terminal sensitivity; no parameters updated.'};args.output.write_text(json.dumps(report,indent=2)+'\n');env.close();app.close();print(json.dumps({'status':report['status'],'fixed_rows':len(fixed),'behavior_rows':len(behavior)},indent=2))
    if True:main()

def run_post_v2_c1_action_behavior_causal_audit():
    """Run former post_v2_c1_action_behavior_causal_audit.py stage."""
    """V2-C1 read-only action-to-behavior causal audit."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    import torch.nn.functional as F
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8],
           "PB_30":[.6,.3,.1],"PB_45":[.45,.45,.1],"PB_60":[.3,.6,.1],
           "PE_30":[.6,.1,.3],"PE_45":[.45,.1,.45],"PE_60":[.3,.1,.6]}
    TARGETS={"B":"P_to_B","E":"P_to_E"}; SCALES=[0.0,0.5,1.0,2.0]
    CODE_COMMIT="923faa1f74dd7c7f3aceebc00afbbed1a0b2d0d9"
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt_deg(q): return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def safe_cos(a,b):
        na=torch.linalg.vector_norm(a,dim=-1);nb=torch.linalg.vector_norm(b,dim=-1)
        mask=(na>1e-9)&(nb>1e-9)
        c=torch.zeros_like(na)
        if mask.any(): c[mask]=F.cosine_similarity(a[mask],b[mask],dim=-1)
        return c,mask
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--steps",type=int,default=32);ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint);mark("RUN_STARTED",protocol="V2-C1",measurement_only=True,checkpoint_sha256=cksha)
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json";anchor=json.loads(anchor_path.read_text())
            anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=67001);obs=obs_tensor(obs).cuda()
            ad=env.unwrapped.action_manager.total_action_dim;manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False);v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();v2.load_state_dict(payload["model"]);v2.eval()
            specs={}
            for lab in ("P","B","E"):
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
            spw={k:ws[k] for k in ("P","B","E")}
            # recover action/joint names if available
            action_meta={"total_action_dim":ad,"active_terms":list(getattr(env.unwrapped.action_manager,"active_terms",[]))}
            try: action_meta["joint_names"]=list(robot.data.joint_names)
            except Exception: action_meta["joint_names"]=[f"action_{i}" for i in range(ad)]
            # same-state action geometry on reset states
            same=[]
            for suite in range(args.suites):
                cur,_=env.reset(seed=67001+suite);cur=obs_tensor(cur).cuda()
                with torch.no_grad():
                    av={k:torch.clamp(v2.act_inference_with_preference(cur,ws[k]),-1,1) for k in PREFS}
                    asp={k:torch.clamp(specs[k].act_inference_with_preference(cur,spw[k]),-1,1) for k in specs}
                    base=av["P"]; row={"suite":suite,"pairs":{},"interpolation":{}}
                    for tgt in ("B","E"):
                        dv=av[tgt]-base;ds=asp[tgt]-asp["P"];cos,mask=safe_cos(dv,ds)
                        per=torch.mean(torch.abs(dv),dim=0)
                        row["pairs"][f"P_to_{tgt}"]={
                          "v2_delta_l2_mean":float(torch.linalg.vector_norm(dv,dim=-1).mean()),
                          "v2_delta_abs_mean":float(dv.abs().mean()),
                          "base_action_l2_mean":float(torch.linalg.vector_norm(base,dim=-1).mean()),
                          "delta_over_base_mean":float((torch.linalg.vector_norm(dv,dim=-1)/(torch.linalg.vector_norm(base,dim=-1)+1e-9)).mean()),
                          "specialist_delta_l2_mean":float(torch.linalg.vector_norm(ds,dim=-1).mean()),
                          "v2_over_specialist_delta_mean":float((torch.linalg.vector_norm(dv,dim=-1)/(torch.linalg.vector_norm(ds,dim=-1)+1e-9)).mean()),
                          "cosine_to_specialist_mean":float(cos[mask].mean()) if mask.any() else 0.0,
                          "cosine_to_specialist_positive_fraction":float((cos[mask]>0).float().mean()) if mask.any() else 0.0,
                          "per_action_abs_delta":per.cpu().tolist(),
                        }
                    for lab in ("PB_30","PB_45","PB_60","PE_30","PE_45","PE_60"):
                        d=av[lab]-base
                        row["interpolation"][lab]={"delta_l2_mean":float(torch.linalg.vector_norm(d,dim=-1).mean()),"delta_abs_mean":float(d.abs().mean())}
                    same.append(row)
            # closed-loop scaled direction rollouts, direction recomputed at each visited state
            scaled=[]
            for target,axis in TARGETS.items():
                for suite in range(args.suites):
                    seed=68001+suite
                    for scale in SCALES:
                        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device="cuda");vecs=[];mets=[];done=np.zeros(args.num_envs,dtype=bool);deltas=[]
                        with torch.no_grad():
                            for _ in range(args.steps):
                                ap_=torch.clamp(v2.act_inference_with_preference(cur,ws["P"]),-1,1)
                                at_=torch.clamp(v2.act_inference_with_preference(cur,ws[target]),-1,1)
                                delta=at_-ap_;action=torch.clamp(ap_+scale*delta,-1,1);deltas.append(float(torch.linalg.vector_norm(delta,dim=-1).mean()))
                                nxt,_,term,trunc,_=env.step(action);done|=(term|trunc).cpu().numpy()
                                raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity");tilt=tilt_deg(data.root_quat_w)
                                vecs.append(vec.mean(0));mets.append({
                                  "vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),
                                  "tilt_p95":float(torch.quantile(tilt,.95)),
                                  "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                                  "torque_norm":float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),
                                  "action_rate":float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),
                                });prev=action;cur=obs_tensor(nxt).cuda()
                        scaled.append({"axis":axis,"target":target,"suite":suite,"scale":scale,"mean_raw_direction_l2":float(np.mean(deltas)),
                          "objective_return":np.mean(vecs,axis=0).tolist(),"metrics":{k:float(np.mean([m[k] for m in mets])) for k in mets[0]},
                          "survival":float(1-done.mean())})
            # summarize scale response deltas relative to scale 0, per suite then average
            scale_summary={}
            for axis in TARGETS.values():
                scale_summary[axis]={}
                for s in SCALES:
                    rs=[r for r in scaled if r["axis"]==axis and r["scale"]==s]
                    scale_summary[axis][str(s)]={
                      "objective_return_mean":np.mean([r["objective_return"] for r in rs],axis=0).tolist(),
                      "metrics_mean":{k:float(np.mean([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]},
                      "survival_mean":float(np.mean([r["survival"] for r in rs])),
                      "mean_raw_direction_l2":float(np.mean([r["mean_raw_direction_l2"] for r in rs])),
                    }
            report={"schema":"post_v2_c1_action_behavior_causal_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                            "v2a_anchor_sha256":sha(anchor_path),
                            "specialist_checkpoints":{k:{"path":f"runs/post_v1_d1-2026-09-22/specialist_{k}_terminal.pt","sha256":sha(ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{k}_terminal.pt")} for k in specs}},
              "action_meta":action_meta,"scales":SCALES,"same_state_geometry":same,"scaled_rollouts":scaled,"scale_summary":scale_summary,
              "note":"Read-only. Scaled action is a_P + scale*(a_target-a_P), recomputed each timestep; no parameters updated."}
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"same_state_rows":len(same),"scaled_rows":len(scaled)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    if True: main()

def run_post_v2_c2_projection_audit():
    """Run former post_v2_c2_projection_audit.py stage."""
    """V2-C2 read-only semantic direction projection/orthogonal causal audit."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    AXES={"P_to_B":"B","P_to_E":"E"}
    CONDITIONS=("baseline","v2","parallel","orthogonal","specialist")
    SCALES=(0.5,1.0,2.0)
    EPS=1e-8
    CODE_COMMIT="923faa1f74dd7c7f3aceebc00afbbed1a0b2d0d9"
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def tilt_deg(q):
        return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def unit(v):
        n=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        return v/(n+EPS), n.squeeze(-1)
    
    def decompose(dv, ds):
        dspec, ns = unit(ds)
        coeff=(dv*dspec).sum(-1,keepdim=True)
        par=coeff*dspec
        perp=dv-par
        u_v,nv=unit(dv); u_par,npar=unit(par); u_perp,nperp=unit(perp)
        return {
          "u_v2":u_v,"u_parallel":u_par,"u_perp":u_perp,"u_spec":dspec,
          "n_v2":nv,"n_spec":ns,"n_parallel":npar,"n_perp":nperp,
          "parallel_valid":npar>EPS,"perp_valid":nperp>EPS,"spec_valid":ns>EPS,
          "projection_coeff":coeff.squeeze(-1),
        }
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--steps",type=int,default=32)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint)
        mark("RUN_STARTED",protocol="V2-C2",measurement_only=True,checkpoint_sha256=cksha,scales=list(SCALES))
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text())
            anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=77001); obs=obs_tensor(obs).cuda()
            ad=env.unwrapped.action_manager.total_action_dim
            manager=env.unwrapped.reward_manager; robot=env.unwrapped.scene["robot"]
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();v2.load_state_dict(payload["model"]);v2.eval()
            specs={}
            spec_paths={}
            for lab in PREFS:
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt"
                pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval()
                specs[lab]=m; spec_paths[lab]=pp
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
            try: joint_names=list(robot.data.joint_names)
            except Exception: joint_names=[f"action_{i}" for i in range(ad)]
    
            geometry=[]
            for axis,target in AXES.items():
                for suite in range(args.suites):
                    cur,_=env.reset(seed=77001+suite);cur=obs_tensor(cur).cuda()
                    with torch.no_grad():
                        aP=torch.clamp(v2.act_inference_with_preference(cur,ws["P"]),-1,1)
                        aT=torch.clamp(v2.act_inference_with_preference(cur,ws[target]),-1,1)
                        sP=torch.clamp(specs["P"].act_inference_with_preference(cur,ws["P"]),-1,1)
                        sT=torch.clamp(specs[target].act_inference_with_preference(cur,ws[target]),-1,1)
                        dv=aT-aP; ds=sT-sP; dc=decompose(dv,ds)
                        par_energy=(dc["n_parallel"]**2)/(dc["n_v2"]**2+EPS)
                        perp_energy=(dc["n_perp"]**2)/(dc["n_v2"]**2+EPS)
                        geometry.append({
                          "axis":axis,"suite":suite,
                          "v2_norm_mean":float(dc["n_v2"].mean()),"spec_norm_mean":float(dc["n_spec"].mean()),
                          "parallel_norm_mean":float(dc["n_parallel"].mean()),"perp_norm_mean":float(dc["n_perp"].mean()),
                          "parallel_energy_fraction_mean":float(par_energy.mean()),"perp_energy_fraction_mean":float(perp_energy.mean()),
                          "projection_coeff_mean":float(dc["projection_coeff"].mean()),
                          "parallel_valid_fraction":float(dc["parallel_valid"].float().mean()),
                          "perp_valid_fraction":float(dc["perp_valid"].float().mean()),
                          "per_action_abs_v2_delta":dv.abs().mean(0).cpu().tolist(),
                          "per_action_abs_parallel":(dc["u_parallel"]*dc["n_v2"].unsqueeze(-1)).abs().mean(0).cpu().tolist(),
                          "per_action_abs_perp":(dc["u_perp"]*dc["n_v2"].unsqueeze(-1)).abs().mean(0).cpu().tolist(),
                          "per_action_abs_spec":(dc["u_spec"]*dc["n_v2"].unsqueeze(-1)).abs().mean(0).cpu().tolist(),
                        })
    
            rows=[]
            for axis,target in AXES.items():
                for suite in range(args.suites):
                    seed=78001+suite
                    for scale in SCALES:
                        for condition in CONDITIONS:
                            cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                            prev=torch.zeros((args.num_envs,ad),device="cuda");vecs=[];mets=[];done=np.zeros(args.num_envs,dtype=bool)
                            mags=[];valids=[];clip_fracs=[]
                            with torch.no_grad():
                                for _ in range(args.steps):
                                    aP=torch.clamp(v2.act_inference_with_preference(cur,ws["P"]),-1,1)
                                    aT=torch.clamp(v2.act_inference_with_preference(cur,ws[target]),-1,1)
                                    sP=torch.clamp(specs["P"].act_inference_with_preference(cur,ws["P"]),-1,1)
                                    sT=torch.clamp(specs[target].act_inference_with_preference(cur,ws[target]),-1,1)
                                    dv=aT-aP; ds=sT-sP; dc=decompose(dv,ds)
                                    m=dc["n_v2"]*scale
                                    if condition=="baseline":
                                        raw_action=aP; valid=torch.ones_like(m,dtype=torch.bool)
                                    elif condition=="v2":
                                        raw_action=aP+m.unsqueeze(-1)*dc["u_v2"]; valid=dc["n_v2"]>EPS
                                    elif condition=="parallel":
                                        raw_action=aP+m.unsqueeze(-1)*dc["u_parallel"]; valid=dc["parallel_valid"]
                                    elif condition=="orthogonal":
                                        raw_action=aP+m.unsqueeze(-1)*dc["u_perp"]; valid=dc["perp_valid"]
                                    elif condition=="specialist":
                                        raw_action=aP+m.unsqueeze(-1)*dc["u_spec"]; valid=dc["spec_valid"]
                                    action=torch.clamp(raw_action,-1,1)
                                    clip_fracs.append(float((raw_action.abs()>1).float().mean()))
                                    mags.append(float(m.mean()));valids.append(float(valid.float().mean()))
                                    nxt,_,term,trunc,_=env.step(action);done|=(term|trunc).cpu().numpy()
                                    raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                                    vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                                    data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity");tilt=tilt_deg(data.root_quat_w)
                                    vecs.append(vec.mean(0));mets.append({
                                      "vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),
                                      "tilt_p95":float(torch.quantile(tilt,.95)),
                                      "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                                      "torque_norm":float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),
                                      "action_rate":float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),
                                    })
                                    prev=action;cur=obs_tensor(nxt).cuda()
                            rows.append({
                              "axis":axis,"target":target,"suite":suite,"scale":scale,"condition":condition,
                              "mean_intervention_norm":float(np.mean(mags)),"valid_fraction":float(np.mean(valids)),
                              "clip_fraction":float(np.mean(clip_fracs)),
                              "objective_return":np.mean(vecs,axis=0).tolist(),
                              "metrics":{k:float(np.mean([m[k] for m in mets])) for k in mets[0]},
                              "survival":float(1-done.mean()),
                            })
    
            # aggregate conditions per axis/scale
            summary={}
            for axis in AXES:
                summary[axis]={}
                for scale in SCALES:
                    summary[axis][str(scale)]={}
                    for condition in CONDITIONS:
                        rs=[r for r in rows if r["axis"]==axis and r["scale"]==scale and r["condition"]==condition]
                        summary[axis][str(scale)][condition]={
                          "objective_return_mean":np.mean([r["objective_return"] for r in rs],axis=0).tolist(),
                          "metrics_mean":{k:float(np.mean([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]},
                          "survival_mean":float(np.mean([r["survival"] for r in rs])),
                          "valid_fraction_mean":float(np.mean([r["valid_fraction"] for r in rs])),
                          "clip_fraction_mean":float(np.mean([r["clip_fraction"] for r in rs])),
                          "intervention_norm_mean":float(np.mean([r["mean_intervention_norm"] for r in rs])),
                        }
            report={
              "schema":"post_v2_c2_projection_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                            "v2a_anchor_sha256":sha(anchor_path),
                            "specialist_checkpoints":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
              "definition":{"baseline":"a_P","v2":"a_P + scale*||dv2||*unit(dv2)",
                            "parallel":"a_P + scale*||dv2||*unit(proj_dspec(dv2))",
                            "orthogonal":"a_P + scale*||dv2||*unit(dv2-proj_dspec(dv2))",
                            "specialist":"a_P + scale*||dv2||*unit(dspec)"},
              "scales":list(SCALES),"conditions":list(CONDITIONS),"joint_names":joint_names,
              "geometry":geometry,"rows":rows,"summary":summary,
              "note":"All non-baseline interventions use matched state-wise magnitude based on original V2 delta norm. Directions are recomputed at every visited state. No parameters updated."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"geometry_rows":len(geometry),"rollout_rows":len(rows)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_c3_trajectory_divergence_audit():
    """Run former post_v2_c3_trajectory_divergence_audit.py stage."""
    """V2-C3 read-only short-horizon closed-loop semantic divergence audit."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    HORIZONS=(4,8,16)
    CODE_COMMIT="923faa1f74dd7c7f3aceebc00afbbed1a0b2d0d9"
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def tilt_deg(q):
        return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def vecnorm(x): return torch.linalg.vector_norm(x,dim=-1)
    
    def first_divergence(values, threshold):
        for i,v in enumerate(values):
            if v > threshold: return i
        return None
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint)
        mark("RUN_STARTED",protocol="V2-C3",measurement_only=True,checkpoint_sha256=cksha,horizons=list(HORIZONS))
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text());anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=87001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
    
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();v2.load_state_dict(payload["model"]);v2.eval()
            specs={}
            spec_paths={}
            for lab in PREFS:
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
    
            families={
              "D1_P":("spec","P"),"D1_B":("spec","B"),"D1_E":("spec","E"),
              "V2_P":("v2","P"),"V2_B":("v2","B"),"V2_E":("v2","E"),
            }
            maxH=max(HORIZONS); trajectories=[]
            for suite in range(args.suites):
                seed=87001+suite
                for fam,(kind,label) in families.items():
                    cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                    prev=torch.zeros((args.num_envs,ad),device="cuda");cum_obj=np.zeros(3,dtype=np.float64);steps=[]
                    with torch.no_grad():
                        for t in range(maxH):
                            model=specs[label] if kind=="spec" else v2
                            raw_action=model.act_inference_with_preference(cur,ws[label])
                            sat_frac=float((raw_action.abs()>1.0).float().mean())
                            action=torch.clamp(raw_action,-1,1)
                            nxt,_,term,trunc,_=env.step(action)
                            raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                            vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                            obj_mean=vec.mean(0);cum_obj+=obj_mean
                            data=robot.data
                            # contact proxy: binary net contact force norm at feet if available
                            contact=None
                            try:
                                sensor=env.unwrapped.scene.sensors.get("contact_forces",None)
                                if sensor is not None:
                                    forces=sensor.data.net_forces_w
                                    contact=(torch.linalg.vector_norm(forces,dim=-1)>1.0).float()
                            except Exception:
                                contact=None
                            step={
                              "t":t,
                              "action_mean":action.mean(0).cpu().tolist(),
                              "action_l2_mean":float(vecnorm(action).mean()),
                              "action_rate_mean":float(vecnorm(action-prev).mean()),
                              "saturation_fraction":sat_frac,
                              "joint_pos_mean":data.joint_pos.mean(0).cpu().tolist(),
                              "joint_vel_mean":data.joint_vel.mean(0).cpu().tolist(),
                              "base_lin_vel_mean":data.root_lin_vel_b.mean(0).cpu().tolist(),
                              "base_ang_vel_mean":data.root_ang_vel_b.mean(0).cpu().tolist(),
                              "tilt_mean":float(tilt_deg(data.root_quat_w).mean()),
                              "torque_mean":data.applied_torque.mean(0).cpu().tolist(),
                              "torque_norm_mean":float(vecnorm(data.applied_torque).mean()),
                              "objective_step_mean":obj_mean.tolist(),
                              "objective_cumulative":cum_obj.tolist(),
                              "termination_fraction":float((term|trunc).float().mean()),
                            }
                            if contact is not None:
                                step["contact_pattern_mean"]=contact.mean(0).cpu().tolist()
                                step["contact_fraction_mean"]=float(contact.mean())
                            steps.append(step);prev=action;cur=obs_tensor(nxt).cuda()
                    trajectories.append({"suite":suite,"seed":seed,"family":fam,"kind":kind,"preference":label,"steps":steps})
    
            # pairwise divergence over time
            pairs=[
              ("D1_P","D1_B","D1_P_vs_B"),("D1_P","D1_E","D1_P_vs_E"),
              ("V2_P","V2_B","V2_P_vs_B"),("V2_P","V2_E","V2_P_vs_E"),
            ]
            div_rows=[]
            def arr(step,key):
                return np.asarray(step[key],dtype=float)
            for suite in range(args.suites):
                for a,b,name in pairs:
                    ta=next(x for x in trajectories if x["suite"]==suite and x["family"]==a)["steps"]
                    tb=next(x for x in trajectories if x["suite"]==suite and x["family"]==b)["steps"]
                    for H in HORIZONS:
                        ts=[]
                        for t in range(H):
                            xa,xb=ta[t],tb[t]
                            row={
                              "t":t,
                              "action":float(np.linalg.norm(arr(xa,"action_mean")-arr(xb,"action_mean"))),
                              "joint_pos":float(np.linalg.norm(arr(xa,"joint_pos_mean")-arr(xb,"joint_pos_mean"))),
                              "joint_vel":float(np.linalg.norm(arr(xa,"joint_vel_mean")-arr(xb,"joint_vel_mean"))),
                              "base_lin_vel":float(np.linalg.norm(arr(xa,"base_lin_vel_mean")-arr(xb,"base_lin_vel_mean"))),
                              "base_ang_vel":float(np.linalg.norm(arr(xa,"base_ang_vel_mean")-arr(xb,"base_ang_vel_mean"))),
                              "tilt":abs(float(xa["tilt_mean"]-xb["tilt_mean"])),
                              "torque":float(np.linalg.norm(arr(xa,"torque_mean")-arr(xb,"torque_mean"))),
                              "objective_cumulative":float(np.linalg.norm(arr(xa,"objective_cumulative")-arr(xb,"objective_cumulative"))),
                              "saturation_gap":abs(float(xa["saturation_fraction"]-xb["saturation_fraction"])),
                            }
                            if "contact_pattern_mean" in xa and "contact_pattern_mean" in xb:
                                row["contact"]=float(np.linalg.norm(arr(xa,"contact_pattern_mean")-arr(xb,"contact_pattern_mean")))
                            ts.append(row)
                        div_rows.append({"suite":suite,"pair":name,"horizon":H,"timeseries":ts})
    
            # thresholds derived from fixed numerical floors + relative scale for interpretability
            thresholds={"action":1e-3,"joint_pos":1e-3,"joint_vel":1e-3,"base_lin_vel":1e-3,"base_ang_vel":1e-3,"tilt":0.05,"torque":1e-2,"objective_cumulative":1e-3,"contact":1e-3,"saturation_gap":1e-3}
            summaries={}
            for _,_,name in pairs:
                summaries[name]={}
                for H in HORIZONS:
                    rs=[r for r in div_rows if r["pair"]==name and r["horizon"]==H]
                    metrics=list(rs[0]["timeseries"][0].keys());metrics.remove("t")
                    summaries[name][str(H)]={}
                    for m in metrics:
                        vals=np.asarray([[x["timeseries"][t][m] for t in range(H)] for x in rs],dtype=float)
                        mean_t=vals.mean(0)
                        firsts=[first_divergence([x["timeseries"][t][m] for t in range(H)],thresholds.get(m,1e-3)) for x in rs]
                        finite_first=[x for x in firsts if x is not None]
                        summaries[name][str(H)][m]={
                          "mean_timeseries":mean_t.tolist(),
                          "terminal_mean":float(mean_t[-1]),
                          "first_divergence_by_suite":firsts,
                          "first_divergence_mean":float(np.mean(finite_first)) if finite_first else None,
                          "diverged_fraction":float(len(finite_first)/len(firsts)),
                          "threshold":thresholds.get(m,1e-3),
                        }
            report={
              "schema":"post_v2_c3_trajectory_divergence_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                            "v2a_anchor_sha256":sha(anchor_path),
                            "specialist_checkpoints":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
              "horizons":list(HORIZONS),"thresholds":thresholds,"families":families,
              "trajectories":trajectories,"divergence":div_rows,"summaries":summaries,
              "note":"Matched initial states, deterministic policies, closed-loop rollout. No parameters updated."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"trajectory_rows":len(trajectories),"divergence_rows":len(div_rows)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_c4_saturation_authority_audit():
    """Run former post_v2_c4_saturation_authority_audit.py stage."""
    """V2-C4 read-only saturation-aware trajectory authority audit."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    HORIZONS=(4,8,16)
    CODE_COMMIT="923faa1f74dd7c7f3aceebc00afbbed1a0b2d0d9"
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def tilt_deg(q):
        return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def vecnorm(x): return torch.linalg.vector_norm(x,dim=-1)
    
    def safe_cos(a,b,eps=1e-12):
        na=vecnorm(a); nb=vecnorm(b); mask=(na>eps)&(nb>eps)
        out=torch.zeros_like(na)
        if mask.any(): out[mask]=torch.nn.functional.cosine_similarity(a[mask],b[mask],dim=-1)
        return out,mask
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint)
        mark("RUN_STARTED",protocol="V2-C4",measurement_only=True,checkpoint_sha256=cksha,horizons=list(HORIZONS))
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text()); anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=97001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
    
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();v2.load_state_dict(payload["model"]);v2.eval()
            specs={};spec_paths={}
            for lab in PREFS:
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
            try: joint_names=list(robot.data.joint_names)
            except Exception: joint_names=[f"action_{i}" for i in range(ad)]
    
            families={
              "D1_P":("spec","P"),"D1_B":("spec","B"),"D1_E":("spec","E"),
              "V2_P":("v2","P"),"V2_B":("v2","B"),"V2_E":("v2","E"),
            }
            maxH=max(HORIZONS);trajs=[]
            for suite in range(args.suites):
                seed=97001+suite
                for fam,(kind,label) in families.items():
                    cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device="cuda")
                    cum=np.zeros(3,dtype=np.float64);steps=[]
                    with torch.no_grad():
                        for t in range(maxH):
                            model=specs[label] if kind=="spec" else v2
                            raw_action=model.act_inference_with_preference(cur,ws[label])
                            action=torch.clamp(raw_action,-1,1)
                            nxt,_,term,trunc,_=env.step(action)
                            raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                            vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                            obj=vec.mean(0);cum+=obj
                            data=robot.data
                            contact=None
                            try:
                                sensor=env.unwrapped.scene.sensors.get("contact_forces",None)
                                if sensor is not None:
                                    forces=sensor.data.net_forces_w
                                    contact=(torch.linalg.vector_norm(forces,dim=-1)>1.0).float()
                            except Exception:
                                contact=None
                            step={
                              "t":t,
                              "raw_action_mean":raw_action.mean(0).cpu().tolist(),
                              "post_action_mean":action.mean(0).cpu().tolist(),
                              "raw_saturation_fraction":float((raw_action.abs()>1).float().mean()),
                              "post_action_l2_mean":float(vecnorm(action).mean()),
                              "joint_pos_mean":data.joint_pos.mean(0).cpu().tolist(),
                              "joint_vel_mean":data.joint_vel.mean(0).cpu().tolist(),
                              "base_lin_vel_mean":data.root_lin_vel_b.mean(0).cpu().tolist(),
                              "base_ang_vel_mean":data.root_ang_vel_b.mean(0).cpu().tolist(),
                              "tilt_mean":float(tilt_deg(data.root_quat_w).mean()),
                              "torque_mean":data.applied_torque.mean(0).cpu().tolist(),
                              "objective_cumulative":cum.tolist(),
                              "termination_fraction":float((term|trunc).float().mean()),
                            }
                            if contact is not None:
                                step["contact_pattern_mean"]=contact.mean(0).cpu().tolist()
                            steps.append(step);prev=action;cur=obs_tensor(nxt).cuda()
                    trajs.append({"suite":suite,"family":fam,"kind":kind,"preference":label,"steps":steps})
    
            pairs=[("D1_P","D1_B","D1_P_vs_B"),("D1_P","D1_E","D1_P_vs_E"),("V2_P","V2_B","V2_P_vs_B"),("V2_P","V2_E","V2_P_vs_E")]
            pair_rows=[]
            for suite in range(args.suites):
                for a,b,name in pairs:
                    ta=next(x for x in trajs if x["suite"]==suite and x["family"]==a)["steps"]
                    tb=next(x for x in trajs if x["suite"]==suite and x["family"]==b)["steps"]
                    ts=[]
                    for t in range(maxH):
                        ra=torch.tensor(ta[t]["raw_action_mean"],dtype=torch.float32)
                        rb=torch.tensor(tb[t]["raw_action_mean"],dtype=torch.float32)
                        pa=torch.tensor(ta[t]["post_action_mean"],dtype=torch.float32)
                        pb=torch.tensor(tb[t]["post_action_mean"],dtype=torch.float32)
                        dpre=rb-ra; dpost=pb-pa; c,mask=safe_cos(dpre.unsqueeze(0),dpost.unsqueeze(0))
                        same_clip=((ra>1)&(rb>1))|((ra<-1)&(rb<-1))
                        raw_diff=(dpre.abs()>1e-8)
                        post_zero=(dpost.abs()<=1e-8)
                        lost=raw_diff & post_zero
                        jloss=(dpre.abs()-dpost.abs()).clamp(min=0)
                        xa,xb=ta[t],tb[t]
                        row={
                          "t":t,
                          "pre_action_delta":float(torch.linalg.vector_norm(dpre)),
                          "post_action_delta":float(torch.linalg.vector_norm(dpost)),
                          "G_clip":float(torch.linalg.vector_norm(dpost)/(torch.linalg.vector_norm(dpre)+1e-12)),
                          "pre_post_cosine":float(c[0]) if mask[0] else 0.0,
                          "same_clip_value_fraction":float(same_clip.float().mean()),
                          "raw_diff_post_zero_fraction":float(lost.float().mean()),
                          "per_joint_clip_loss":jloss.tolist(),
                          "joint_pos_delta":float(np.linalg.norm(np.asarray(xa["joint_pos_mean"])-np.asarray(xb["joint_pos_mean"]))),
                          "joint_vel_delta":float(np.linalg.norm(np.asarray(xa["joint_vel_mean"])-np.asarray(xb["joint_vel_mean"]))),
                          "base_lin_vel_delta":float(np.linalg.norm(np.asarray(xa["base_lin_vel_mean"])-np.asarray(xb["base_lin_vel_mean"]))),
                          "base_ang_vel_delta":float(np.linalg.norm(np.asarray(xa["base_ang_vel_mean"])-np.asarray(xb["base_ang_vel_mean"]))),
                          "tilt_delta":abs(float(xa["tilt_mean"]-xb["tilt_mean"])),
                          "torque_delta":float(np.linalg.norm(np.asarray(xa["torque_mean"])-np.asarray(xb["torque_mean"]))),
                          "objective_delta":float(np.linalg.norm(np.asarray(xa["objective_cumulative"])-np.asarray(xb["objective_cumulative"]))),
                        }
                        if "contact_pattern_mean" in xa and "contact_pattern_mean" in xb:
                            row["contact_delta"]=float(np.linalg.norm(np.asarray(xa["contact_pattern_mean"])-np.asarray(xb["contact_pattern_mean"])))
                        ts.append(row)
                    pair_rows.append({"suite":suite,"pair":name,"timeseries":ts})
    
            summaries={}
            for _,_,name in pairs:
                summaries[name]={}
                rs=[r for r in pair_rows if r["pair"]==name]
                for H in HORIZONS:
                    pre_sums=[];post_sums=[];gclips=[];cos=[];same=[];lost=[]
                    dyn={"joint_pos":[],"joint_vel":[],"base_lin_vel":[],"base_ang_vel":[],"tilt":[],"torque":[],"objective":[],"contact":[]}
                    per_joint=[]
                    for r in rs:
                        ts=r["timeseries"][:H]
                        pre_sum=sum(x["pre_action_delta"] for x in ts);post_sum=sum(x["post_action_delta"] for x in ts)
                        pre_sums.append(pre_sum);post_sums.append(post_sum)
                        gclips.append(post_sum/(pre_sum+1e-12))
                        cos.append(float(np.mean([x["pre_post_cosine"] for x in ts])))
                        same.append(float(np.mean([x["same_clip_value_fraction"] for x in ts])))
                        lost.append(float(np.mean([x["raw_diff_post_zero_fraction"] for x in ts])))
                        per_joint.append(np.mean([x["per_joint_clip_loss"] for x in ts],axis=0))
                        last=ts[-1]
                        dyn["joint_pos"].append(last["joint_pos_delta"]/(post_sum+1e-12))
                        dyn["joint_vel"].append(last["joint_vel_delta"]/(post_sum+1e-12))
                        dyn["base_lin_vel"].append(last["base_lin_vel_delta"]/(post_sum+1e-12))
                        dyn["base_ang_vel"].append(last["base_ang_vel_delta"]/(post_sum+1e-12))
                        dyn["tilt"].append(last["tilt_delta"]/(post_sum+1e-12))
                        dyn["torque"].append(last["torque_delta"]/(post_sum+1e-12))
                        dyn["objective"].append(last["objective_delta"]/(post_sum+1e-12))
                        if "contact_delta" in last:dyn["contact"].append(last["contact_delta"]/(post_sum+1e-12))
                    pj=np.mean(per_joint,axis=0)
                    summaries[name][str(H)]={
                      "G_clip_sum_mean":float(np.mean(gclips)),
                      "pre_post_cosine_mean":float(np.mean(cos)),
                      "same_clip_value_fraction_mean":float(np.mean(same)),
                      "raw_diff_post_zero_fraction_mean":float(np.mean(lost)),
                      "pre_action_delta_sum_mean":float(np.mean(pre_sums)),
                      "post_action_delta_sum_mean":float(np.mean(post_sums)),
                      "per_joint_clip_loss_mean":pj.tolist(),
                      "G_dyn":{k:(float(np.mean(v)) if len(v) else None) for k,v in dyn.items()},
                    }
            report={
              "schema":"post_v2_c4_saturation_authority_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                            "v2a_anchor_sha256":sha(anchor_path),
                            "specialist_checkpoints":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
              "horizons":list(HORIZONS),"joint_names":joint_names,"trajectories":trajs,"pairs":pair_rows,"summaries":summaries,
              "definitions":{"G_clip":"sum_t ||delta a_post|| / (sum_t ||delta a_pre|| + eps)",
                             "G_dyn":"terminal signal divergence / (sum_t ||delta a_post|| + eps)"},
              "note":"Matched initial states, deterministic policies, explicit pre/post clip tracing. No parameters updated."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"pairs":list(summaries)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_c5_raw_output_authority_audit():
    """Run former post_v2_c5_raw_output_authority_audit.py stage."""
    """V2-C5 read-only raw-output authority audit."""
    import argparse, hashlib, json, math, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    PAIRS=(("P","B","P_to_B"),("P","E","P_to_E"))
    CODE_COMMIT="923faa1f74dd7c7f3aceebc00afbbed1a0b2d0d9"
    EPS=1e-12
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def l2(x): return torch.linalg.vector_norm(x,dim=-1)
    
    def ratio(num,den): return num/(den+EPS)
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint);mark("RUN_STARTED",protocol="V2-C5",measurement_only=True,checkpoint_sha256=cksha)
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text());anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=107001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            robot=env.unwrapped.scene["robot"]
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();v2.load_state_dict(payload["model"]);v2.eval()
            specs={};spec_paths={}
            for lab in PREFS:
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
            try: joint_names=list(robot.data.joint_names)
            except Exception: joint_names=[f"action_{i}" for i in range(ad)]
            ext_mu_thresh=math.atanh(1.0/float(v2.ACTION_CLIP))
    
            rows=[]
            for suite in range(args.suites):
                cur,_=env.reset(seed=107001+suite);cur=obs_tensor(cur).cuda()
                with torch.no_grad():
                    v2cache={}
                    for lab in PREFS:
                        z=v2.behavior_z(ws[lab])
                        h0=torch.nn.functional.elu(v2.actor_pre(cur))
                        gamma=v2.film_gamma(z); beta=v2.film_beta(z)
                        scale=1.0+0.1*torch.tanh(gamma); shift=0.1*torch.tanh(beta)
                        hf=scale*h0+shift
                        hr=v2.actor_rest(hf)
                        mu=v2.actor_mean(hr)
                        atan=torch.tanh(mu)*v2.ACTION_CLIP
                        aclip=torch.clamp(atan,-1,1)
                        deriv=v2.ACTION_CLIP*(1-torch.tanh(mu).pow(2))
                        v2cache[lab]={"z":z,"h0":h0,"gamma":gamma,"beta":beta,"scale":scale,"shift":shift,"hf":hf,"hr":hr,"mu":mu,"atan":atan,"aclip":aclip,"deriv":deriv}
                    speccache={}
                    for lab in PREFS:
                        x=torch.cat((cur,ws[lab]),dim=-1)
                        body=specs[lab].actor_body(x)
                        mu=specs[lab].actor_mean(body)
                        atan=torch.tanh(mu)*specs[lab].ACTION_CLIP
                        aclip=torch.clamp(atan,-1,1)
                        deriv=specs[lab].ACTION_CLIP*(1-torch.tanh(mu).pow(2))
                        speccache[lab]={"body":body,"mu":mu,"atan":atan,"aclip":aclip,"deriv":deriv}
                    for p,t,name in PAIRS:
                        P=v2cache[p];T=v2cache[t]
                        dz=T["z"]-P["z"]
                        dgamma=T["gamma"]-P["gamma"]; dbeta=T["beta"]-P["beta"]
                        dscale=T["scale"]-P["scale"]; dshift=T["shift"]-P["shift"]
                        dhf=T["hf"]-P["hf"]; dhr=T["hr"]-P["hr"]; dmu=T["mu"]-P["mu"]; datan=T["atan"]-P["atan"]; dclip=T["aclip"]-P["aclip"]
                        SP=speccache[p];ST=speccache[t]; sdmu=ST["mu"]-SP["mu"]; sdatan=ST["atan"]-SP["atan"]; sdclip=ST["aclip"]-SP["aclip"]
                        mu_base=l2(P["mu"]); hfbase=l2(P["hf"])
                        per_mu=dmu.abs().mean(0); per_tanh=datan.abs().mean(0); per_clip=dclip.abs().mean(0)
                        ext_sat=(P["mu"].abs()>ext_mu_thresh)
                        target_ext_sat=(T["mu"].abs()>ext_mu_thresh)
                        both_same_side=((P["mu"]>ext_mu_thresh)&(T["mu"]>ext_mu_thresh))|((P["mu"]<-ext_mu_thresh)&(T["mu"]<-ext_mu_thresh))
                        tanh_compression=ratio(l2(datan),l2(dmu))
                        clip_retention=ratio(l2(dclip),l2(datan))
                        spec_mu_ratio=ratio(l2(dmu),l2(sdmu))
                        rows.append({
                          "suite":suite,"pair":name,
                          "z_delta_l2_mean":float(l2(dz).mean()),
                          "gamma_delta_l2_mean":float(l2(dgamma).mean()),"beta_delta_l2_mean":float(l2(dbeta).mean()),
                          "film_scale_delta_l2_mean":float(l2(dscale).mean()),"film_shift_delta_l2_mean":float(l2(dshift).mean()),
                          "h_postfilm_delta_l2_mean":float(l2(dhf).mean()),"h_rest_delta_l2_mean":float(l2(dhr).mean()),
                          "mu_delta_l2_mean":float(l2(dmu).mean()),"post_tanh_delta_l2_mean":float(l2(datan).mean()),"post_clip_delta_l2_mean":float(l2(dclip).mean()),
                          "G_FiLM":float(ratio(l2(dhf),l2(dz)).mean()),
                          "G_rest_head":float(ratio(l2(dmu),l2(dhf)).mean()),
                          "G_tanh":float(tanh_compression.mean()),
                          "G_external_clip":float(clip_retention.mean()),
                          "mu_delta_over_muP_mean":float(ratio(l2(dmu),mu_base).mean()),
                          "h_delta_over_hP_mean":float(ratio(l2(dhf),hfbase).mean()),
                          "muP_l2_mean":float(mu_base.mean()),"muT_l2_mean":float(l2(T["mu"]).mean()),
                          "tanh_derivative_P_mean":float(P["deriv"].mean()),"tanh_derivative_T_mean":float(T["deriv"].mean()),
                          "external_saturated_P_fraction":float(ext_sat.float().mean()),"external_saturated_T_fraction":float(target_ext_sat.float().mean()),
                          "both_same_external_saturation_side_fraction":float(both_same_side.float().mean()),
                          "D1_mu_delta_l2_mean":float(l2(sdmu).mean()),"D1_post_tanh_delta_l2_mean":float(l2(sdatan).mean()),"D1_post_clip_delta_l2_mean":float(l2(sdclip).mean()),
                          "V2_to_D1_mu_delta_ratio_mean":float(spec_mu_ratio.mean()),
                          "per_joint_mu_abs_delta":per_mu.cpu().tolist(),
                          "per_joint_post_tanh_abs_delta":per_tanh.cpu().tolist(),
                          "per_joint_post_clip_abs_delta":per_clip.cpu().tolist(),
                          "per_joint_muP_abs_mean":P["mu"].abs().mean(0).cpu().tolist(),
                          "per_joint_tanh_derivative_P_mean":P["deriv"].mean(0).cpu().tolist(),
                          "per_joint_external_sat_P_fraction":ext_sat.float().mean(0).cpu().tolist(),
                        })
    
            summary={}
            for _,_,name in PAIRS:
                rs=[r for r in rows if r["pair"]==name]
                keys=[k for k,v in rs[0].items() if isinstance(v,(float,int)) and k!="suite"]
                summary[name]={k:float(np.mean([r[k] for r in rs])) for k in keys}
                # joint attribution
                for key in ["per_joint_mu_abs_delta","per_joint_post_tanh_abs_delta","per_joint_post_clip_abs_delta","per_joint_muP_abs_mean","per_joint_tanh_derivative_P_mean","per_joint_external_sat_P_fraction"]:
                    summary[name][key]=np.mean([r[key] for r in rs],axis=0).tolist()
                mu=np.asarray(summary[name]["per_joint_mu_abs_delta"])
                shares=mu/(mu.sum()+EPS);order=np.argsort(-shares)
                summary[name]["top_mu_delta_joints"]=[{"joint":joint_names[i],"share":float(shares[i]),"mu_abs_delta":float(mu[i]),
                  "muP_abs_mean":float(summary[name]["per_joint_muP_abs_mean"][i]),
                  "tanh_derivative_P":float(summary[name]["per_joint_tanh_derivative_P_mean"][i]),
                  "external_sat_P_fraction":float(summary[name]["per_joint_external_sat_P_fraction"][i])} for i in order[:6]]
            report={
              "schema":"post_v2_c5_raw_output_authority_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                            "v2a_anchor_sha256":sha(anchor_path),
                            "specialist_checkpoints":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
              "external_clip_mu_threshold":ext_mu_thresh,"action_clip_internal":float(v2.ACTION_CLIP),"joint_names":joint_names,
              "rows":rows,"summary":summary,
              "definitions":{"G_FiLM":"||delta h_postfilm||/(||delta z||+eps)",
                             "G_rest_head":"||delta mu||/(||delta h_postfilm||+eps)",
                             "G_tanh":"||delta post_tanh action||/(||delta mu||+eps)",
                             "G_external_clip":"||delta post_external_clip||/(||delta post_tanh action||+eps)"},
              "note":"Same-state deterministic layer decomposition. No parameters updated."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"pairs":list(summary)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_c_semantic_response_audit():
    """Run former post_v2_c_semantic_response_audit.py stage."""
    """V2-C read-only semantic preference-response audit on frozen V2-B checkpoint."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CODE_COMMIT="923faa1f74dd7c7f3aceebc00afbbed1a0b2d0d9"
    GRID={
     "P":[.8,.1,.1],"PB_30":[.6,.3,.1],"PB_45":[.45,.45,.1],"PB_60":[.3,.6,.1],"B":[.1,.8,.1],
     "PE_30":[.6,.1,.3],"PE_45":[.45,.1,.45],"PE_60":[.3,.1,.6],"E":[.1,.1,.8],
     "C":[1/3,1/3,1/3],
    }
    LINES={"P_to_B":["P","PB_30","PB_45","PB_60","B"],"P_to_E":["P","PE_30","PE_45","PE_60","E"]}
    EXPECTED={
     "P_to_B":{
       "objective_progress":"decrease","objective_balance":"increase",
       "vx_error":"increase","tilt_p95":"decrease","ang_vel_xy":"decrease",
     },
     "P_to_E":{
       "objective_progress":"decrease","objective_efficiency":"increase",
       "vx_error":"increase","torque_norm":"decrease","action_rate":"decrease",
     },
    }
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt_deg(q): return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    def directional_fraction(vals,direction,tol=1e-8):
        d=np.diff(np.asarray(vals,dtype=float))
        if direction=="increase": return float(np.mean(d>=-tol))
        if direction=="decrease": return float(np.mean(d<=tol))
        raise ValueError(direction)
    def endpoint_between_fraction(vals,tol=1e-8):
        v=np.asarray(vals,dtype=float); lo=min(v[0],v[-1])-tol; hi=max(v[0],v[-1])+tol
        return float(np.mean((v[1:-1]>=lo)&(v[1:-1]<=hi)))
    def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--steps",type=int,default=32)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        audit_sha=sha256(__file__); ck_sha=sha256(args.checkpoint)
        mark("RUN_STARTED",protocol="V2-C",measurement_only=True,code_commit=CODE_COMMIT,checkpoint_sha256=ck_sha,audit_script_sha256=audit_sha)
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text()); anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=47001); obs=obs_tensor(obs).cuda(); ad=env.unwrapped.action_manager.total_action_dim
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            model=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda(); model.load_state_dict(payload["model"]); model.eval()
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in GRID.items()}
            rows=[]
            manager=env.unwrapped.reward_manager; robot=env.unwrapped.scene["robot"]
            with torch.no_grad():
                for suite in range(args.suites):
                    seed=57001+suite
                    for label in GRID:
                        cur,_=env.reset(seed=seed); cur=obs_tensor(cur).cuda(); prev=torch.zeros((args.num_envs,ad),device="cuda")
                        vecs=[]; mets=[]; done_any=np.zeros(args.num_envs,dtype=bool); finite=True
                        z=model.behavior_z(ws[label]).mean(0)
                        for _ in range(args.steps):
                            action=torch.clamp(model.act_inference_with_preference(cur,ws[label]),-1,1)
                            finite=finite and bool(torch.isfinite(action).all())
                            nxt,_,term,trunc,_=env.step(action); done_any|=(term|trunc).cpu().numpy()
                            raw=manager._step_reward.detach().cpu().numpy(); names=list(manager.active_terms)
                            vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                            data=robot.data; cmd=env.unwrapped.command_manager.get_command("base_velocity"); tilt=tilt_deg(data.root_quat_w)
                            vecs.append(vec.mean(0))
                            mets.append({
                              "vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),
                              "tilt_p95":float(torch.quantile(tilt,.95)),
                              "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                              "torque_norm":float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),
                              "action_rate":float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),
                            })
                            prev=action; cur=obs_tensor(nxt).cuda()
                        rows.append({
                          "suite":suite,"seed":seed,"preference":label,"w":GRID[label],"z":z.cpu().tolist(),
                          "objective_return":np.mean(vecs,axis=0).tolist(),
                          "metrics":{k:float(np.mean([m[k] for m in mets])) for k in mets[0]},
                          "survival":float(1-done_any.mean()),"finite":finite,
                        })
            profiles={}
            metric_keys=["objective_progress","objective_balance","objective_efficiency","vx_error","tilt_p95","ang_vel_xy","torque_norm","action_rate","survival"]
            for axis,labels in LINES.items():
                profiles[axis]={}
                for key in metric_keys:
                    suites=[]
                    for suite in range(args.suites):
                        rs=[next(r for r in rows if r["suite"]==suite and r["preference"]==lab) for lab in labels]
                        if key.startswith("objective_"):
                            idx={"objective_progress":0,"objective_balance":1,"objective_efficiency":2}[key]
                            vals=[r["objective_return"][idx] for r in rs]
                        else: vals=[r[key] if key=="survival" else r["metrics"][key] for r in rs]
                        item={"suite":suite,"values":vals,"endpoint_between_fraction":endpoint_between_fraction(vals)}
                        if key in EXPECTED[axis]:
                            item["expected_direction"]=EXPECTED[axis][key]
                            item["monotonicity_fraction"]=directional_fraction(vals,EXPECTED[axis][key])
                        suites.append(item)
                    profiles[axis][key]={
                      "suite_profiles":suites,
                      "mean_endpoint_between_fraction":float(np.mean([s["endpoint_between_fraction"] for s in suites])),
                    }
                    directed=[s.get("monotonicity_fraction") for s in suites if "monotonicity_fraction" in s]
                    if directed: profiles[axis][key]["mean_monotonicity_fraction"]=float(np.mean(directed))
            endpoint_summary={}
            for label in ("P","B","E","C"):
                rs=[r for r in rows if r["preference"]==label]
                endpoint_summary[label]={
                  "objective_return_mean":np.mean([r["objective_return"] for r in rs],axis=0).tolist(),
                  "metrics_mean":{k:float(np.mean([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]},
                  "survival_mean":float(np.mean([r["survival"] for r in rs])),
                }
            report={
              "schema":"post_v2_c_semantic_response_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":ck_sha,
                            "audit_script_sha256":audit_sha,"v2a_anchor_sha256":sha256(anchor_path)},
              "grid":GRID,"lines":LINES,"expected_directions":EXPECTED,"suites":args.suites,"steps":args.steps,
              "rows":rows,"profiles":profiles,"endpoint_summary":endpoint_summary,
              "note":"No training or parameter updates. Semantic directions were fixed in code before measurement."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n"); mark("ARTIFACT_WRITTEN",path=str(args.output)); mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"rows":len(rows),"checkpoint_sha256":ck_sha},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n")
            mark("ERROR",error=str(exc)); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    if True: main()

def run_post_v2_r1_authority_audit():
    """Run former post_v2_r1_authority_audit.py stage."""
    """V2-R1 read-only raw-output authority audit for fixed FiLM alpha=0.5."""
    import argparse, hashlib, json, math, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    PAIRS=(("P","B","P_to_B"),("P","E","P_to_E"))
    CODE_COMMIT="f77af91932b1651b531cb3f64cb3c1852ab800cb"
    EPS=1e-12
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def l2(x): return torch.linalg.vector_norm(x,dim=-1)
    
    def ratio(num,den): return num/(den+EPS)
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint);mark("RUN_STARTED",protocol="V2-R1-AUTHORITY",measurement_only=True,checkpoint_sha256=cksha,film_alpha=0.5)
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text());anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=107001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            robot=env.unwrapped.scene["robot"]
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda();v2.load_state_dict(payload["model"]);v2.eval()
            specs={};spec_paths={}
            for lab in PREFS:
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
            try: joint_names=list(robot.data.joint_names)
            except Exception: joint_names=[f"action_{i}" for i in range(ad)]
            ext_mu_thresh=math.atanh(1.0/float(v2.ACTION_CLIP))
    
            rows=[]
            for suite in range(args.suites):
                cur,_=env.reset(seed=107001+suite);cur=obs_tensor(cur).cuda()
                with torch.no_grad():
                    v2cache={}
                    for lab in PREFS:
                        z=v2.behavior_z(ws[lab])
                        h0=torch.nn.functional.elu(v2.actor_pre(cur))
                        gamma=v2.film_gamma(z); beta=v2.film_beta(z)
                        scale=1.0+v2.film_alpha*torch.tanh(gamma); shift=v2.film_alpha*torch.tanh(beta)
                        hf=scale*h0+shift
                        hr=v2.actor_rest(hf)
                        mu=v2.actor_mean(hr)
                        atan=torch.tanh(mu)*v2.ACTION_CLIP
                        aclip=torch.clamp(atan,-1,1)
                        deriv=v2.ACTION_CLIP*(1-torch.tanh(mu).pow(2))
                        v2cache[lab]={"z":z,"h0":h0,"gamma":gamma,"beta":beta,"scale":scale,"shift":shift,"hf":hf,"hr":hr,"mu":mu,"atan":atan,"aclip":aclip,"deriv":deriv}
                    speccache={}
                    for lab in PREFS:
                        x=torch.cat((cur,ws[lab]),dim=-1)
                        body=specs[lab].actor_body(x)
                        mu=specs[lab].actor_mean(body)
                        atan=torch.tanh(mu)*specs[lab].ACTION_CLIP
                        aclip=torch.clamp(atan,-1,1)
                        deriv=specs[lab].ACTION_CLIP*(1-torch.tanh(mu).pow(2))
                        speccache[lab]={"body":body,"mu":mu,"atan":atan,"aclip":aclip,"deriv":deriv}
                    for p,t,name in PAIRS:
                        P=v2cache[p];T=v2cache[t]
                        dz=T["z"]-P["z"]
                        dgamma=T["gamma"]-P["gamma"]; dbeta=T["beta"]-P["beta"]
                        dscale=T["scale"]-P["scale"]; dshift=T["shift"]-P["shift"]
                        dhf=T["hf"]-P["hf"]; dhr=T["hr"]-P["hr"]; dmu=T["mu"]-P["mu"]; datan=T["atan"]-P["atan"]; dclip=T["aclip"]-P["aclip"]
                        SP=speccache[p];ST=speccache[t]; sdmu=ST["mu"]-SP["mu"]; sdatan=ST["atan"]-SP["atan"]; sdclip=ST["aclip"]-SP["aclip"]
                        mu_base=l2(P["mu"]); hfbase=l2(P["hf"])
                        per_mu=dmu.abs().mean(0); per_tanh=datan.abs().mean(0); per_clip=dclip.abs().mean(0)
                        ext_sat=(P["mu"].abs()>ext_mu_thresh)
                        target_ext_sat=(T["mu"].abs()>ext_mu_thresh)
                        both_same_side=((P["mu"]>ext_mu_thresh)&(T["mu"]>ext_mu_thresh))|((P["mu"]<-ext_mu_thresh)&(T["mu"]<-ext_mu_thresh))
                        tanh_compression=ratio(l2(datan),l2(dmu))
                        clip_retention=ratio(l2(dclip),l2(datan))
                        spec_mu_ratio=ratio(l2(dmu),l2(sdmu))
                        rows.append({
                          "suite":suite,"pair":name,
                          "z_delta_l2_mean":float(l2(dz).mean()),
                          "gamma_delta_l2_mean":float(l2(dgamma).mean()),"beta_delta_l2_mean":float(l2(dbeta).mean()),
                          "film_scale_delta_l2_mean":float(l2(dscale).mean()),"film_shift_delta_l2_mean":float(l2(dshift).mean()),
                          "h_postfilm_delta_l2_mean":float(l2(dhf).mean()),"h_rest_delta_l2_mean":float(l2(dhr).mean()),
                          "mu_delta_l2_mean":float(l2(dmu).mean()),"post_tanh_delta_l2_mean":float(l2(datan).mean()),"post_clip_delta_l2_mean":float(l2(dclip).mean()),
                          "G_FiLM":float(ratio(l2(dhf),l2(dz)).mean()),
                          "G_rest_head":float(ratio(l2(dmu),l2(dhf)).mean()),
                          "G_tanh":float(tanh_compression.mean()),
                          "G_external_clip":float(clip_retention.mean()),
                          "mu_delta_over_muP_mean":float(ratio(l2(dmu),mu_base).mean()),
                          "h_delta_over_hP_mean":float(ratio(l2(dhf),hfbase).mean()),
                          "muP_l2_mean":float(mu_base.mean()),"muT_l2_mean":float(l2(T["mu"]).mean()),
                          "tanh_derivative_P_mean":float(P["deriv"].mean()),"tanh_derivative_T_mean":float(T["deriv"].mean()),
                          "external_saturated_P_fraction":float(ext_sat.float().mean()),"external_saturated_T_fraction":float(target_ext_sat.float().mean()),
                          "both_same_external_saturation_side_fraction":float(both_same_side.float().mean()),
                          "D1_mu_delta_l2_mean":float(l2(sdmu).mean()),"D1_post_tanh_delta_l2_mean":float(l2(sdatan).mean()),"D1_post_clip_delta_l2_mean":float(l2(sdclip).mean()),
                          "V2_to_D1_mu_delta_ratio_mean":float(spec_mu_ratio.mean()),
                          "per_joint_mu_abs_delta":per_mu.cpu().tolist(),
                          "per_joint_post_tanh_abs_delta":per_tanh.cpu().tolist(),
                          "per_joint_post_clip_abs_delta":per_clip.cpu().tolist(),
                          "per_joint_muP_abs_mean":P["mu"].abs().mean(0).cpu().tolist(),
                          "per_joint_tanh_derivative_P_mean":P["deriv"].mean(0).cpu().tolist(),
                          "per_joint_external_sat_P_fraction":ext_sat.float().mean(0).cpu().tolist(),
                        })
    
            summary={}
            for _,_,name in PAIRS:
                rs=[r for r in rows if r["pair"]==name]
                keys=[k for k,v in rs[0].items() if isinstance(v,(float,int)) and k!="suite"]
                summary[name]={k:float(np.mean([r[k] for r in rs])) for k in keys}
                # joint attribution
                for key in ["per_joint_mu_abs_delta","per_joint_post_tanh_abs_delta","per_joint_post_clip_abs_delta","per_joint_muP_abs_mean","per_joint_tanh_derivative_P_mean","per_joint_external_sat_P_fraction"]:
                    summary[name][key]=np.mean([r[key] for r in rs],axis=0).tolist()
                mu=np.asarray(summary[name]["per_joint_mu_abs_delta"])
                shares=mu/(mu.sum()+EPS);order=np.argsort(-shares)
                summary[name]["top_mu_delta_joints"]=[{"joint":joint_names[i],"share":float(shares[i]),"mu_abs_delta":float(mu[i]),
                  "muP_abs_mean":float(summary[name]["per_joint_muP_abs_mean"][i]),
                  "tanh_derivative_P":float(summary[name]["per_joint_tanh_derivative_P_mean"][i]),
                  "external_sat_P_fraction":float(summary[name]["per_joint_external_sat_P_fraction"][i])} for i in order[:6]]
            report={
              "schema":"post_v2_r1_raw_output_authority_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,"film_alpha":0.5,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                            "v2a_anchor_sha256":sha(anchor_path),
                            "specialist_checkpoints":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
              "external_clip_mu_threshold":ext_mu_thresh,"action_clip_internal":float(v2.ACTION_CLIP),"joint_names":joint_names,
              "rows":rows,"summary":summary,
              "definitions":{"G_FiLM":"||delta h_postfilm||/(||delta z||+eps)",
                             "G_rest_head":"||delta mu||/(||delta h_postfilm||+eps)",
                             "G_tanh":"||delta post_tanh action||/(||delta mu||+eps)",
                             "G_external_clip":"||delta post_external_clip||/(||delta post_tanh action||+eps)"},
              "note":"Same-state deterministic layer decomposition for V2-R1 alpha=0.5. No parameters updated."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"pairs":list(summary)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_r1_authority_compare():
    """Run former post_v2_r1_authority_compare.py stage."""
    """Read-only V2 control vs V2-R1 authority comparison on matched states."""
    import argparse,hashlib,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    PAIRS=(("P","B","P_to_B"),("P","E","P_to_E"))
    EPS=1e-12
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def l2(x):return torch.linalg.vector_norm(x,dim=-1)
    def decomp(model,obs,w,alpha):
        z=model.behavior_z(w)
        h0=torch.nn.functional.elu(model.actor_pre(obs))
        gamma=model.film_gamma(z);beta=model.film_beta(z)
        hf=(1+alpha*torch.tanh(gamma))*h0+alpha*torch.tanh(beta)
        hr=model.actor_rest(hf);mu=model.actor_mean(hr)
        atan=torch.tanh(mu)*model.ACTION_CLIP;aclip=torch.clamp(atan,-1,1)
        return {"z":z,"h0":h0,"hf":hf,"hr":hr,"mu":mu,"atan":atan,"aclip":aclip}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--control",type=Path,required=True);ap.add_argument("--treatment",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json";anchor=json.loads(anchor_path.read_text());anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
            from talon_rl.models.behavior.high_authority import V2R1BehaviorActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=117001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            pc=torch.load(args.control,map_location="cuda",weights_only=False);pt=torch.load(args.treatment,map_location="cuda",weights_only=False)
            mc=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();mc.load_state_dict(pc["model"]);mc.eval()
            mt=V2R1BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();mt.load_state_dict(pt["model"]);mt.eval()
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
            rows=[]
            with torch.no_grad():
                for suite in range(args.suites):
                    cur,_=env.reset(seed=117001+suite);cur=obs_tensor(cur).cuda()
                    cc={k:decomp(mc,cur,ws[k],0.1) for k in PREFS};tt={k:decomp(mt,cur,ws[k],0.5) for k in PREFS}
                    for p,t,name in PAIRS:
                        row={"suite":suite,"pair":name}
                        for label,cache in [("control",cc),("treatment",tt)]:
                            P=cache[p];T=cache[t]
                            dz=T["z"]-P["z"];dh=T["hf"]-P["hf"];dmu=T["mu"]-P["mu"];dat=T["atan"]-P["atan"];dcl=T["aclip"]-P["aclip"]
                            row[label]={
                              "z_delta":float(l2(dz).mean()),
                              "h_delta":float(l2(dh).mean()),
                              "h_delta_over_hP":float((l2(dh)/(l2(P["hf"])+EPS)).mean()),
                              "mu_delta":float(l2(dmu).mean()),
                              "mu_delta_over_muP":float((l2(dmu)/(l2(P["mu"])+EPS)).mean()),
                              "post_tanh_delta":float(l2(dat).mean()),
                              "post_clip_delta":float(l2(dcl).mean()),
                            }
                        rows.append(row)
            summary={}
            for _,_,name in PAIRS:
                rs=[r for r in rows if r["pair"]==name];summary[name]={}
                for label in ("control","treatment"):
                    ks=rs[0][label].keys();summary[name][label]={k:float(np.mean([r[label][k] for r in rs])) for k in ks}
                summary[name]["gain_treatment_over_control"]={k:summary[name]["treatment"][k]/(summary[name]["control"][k]+EPS) for k in summary[name]["control"]}
            report={"schema":"v2_r1_authority_compare_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"control":str(args.control),"control_sha256":sha(args.control),"treatment":str(args.treatment),"treatment_sha256":sha(args.treatment),"anchor_sha256":sha(anchor_path)},
              "rows":rows,"summary":summary,
              "note":"Matched-state post-training authority comparison. Control alpha=0.1, treatment alpha=0.5; no parameters updated."}
            args.output.write_text(json.dumps(report,indent=2)+"\\n");print(json.dumps(report["summary"],indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\\n");raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_r1_isaac_smoke():
    """Run former post_v2_r1_isaac_smoke.py stage."""
    """V2-R1 real-Isaac function-preserving smoke; no optimizer step."""
    import argparse,json,sys,time,traceback,hashlib
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    CODE_COMMIT="f76aa7b21eb125bd745ca4330ddcd5c1f8bd780b"
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--steps',type=int,default=16)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(event,**extra):
            with life.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**extra},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='V2-R1-SMOKE',training=False,code_commit=CODE_COMMIT)
        app=env=None
        try:
            anchor_path=ROOT/'runs/post_v2_a-2026-09-23/v2a.json'
            anchor=json.loads(anchor_path.read_text());anchors=torch.tensor([anchor['anchors'][k] for k in ('P','B','E')],dtype=torch.float32)
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic,initialize_from_v1c
            from talon_rl.models.behavior.high_authority import V2R1BehaviorActorCritic,FILM_AUTHORITY_R1
            source_path=ROOT/'runs/v1c_confirmatory_seed0-2026-09-22/full_simplex_control_terminal.pt'
            source_payload=torch.load(source_path,map_location='cpu',weights_only=False);source=source_payload['model']
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=47001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            v1=V1CSharedActorCritic(obs.shape[-1],ad).cuda();v1.load_state_dict(source);v1.eval()
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();initialize_from_v1c(v2,source);v2.eval()
            r1=V2R1BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors).cuda();initialize_from_v1c(r1,source);r1.eval()
            prefs={k:torch.tensor(v,dtype=torch.float32,device='cuda').repeat(args.num_envs,1) for k,v in {'P':[.8,.1,.1],'B':[.1,.8,.1],'E':[.1,.1,.8],'C':[1/3,1/3,1/3]}.items()}
            diffs={}
            with torch.no_grad():
                for k,w in prefs.items():
                    a2=v2.act_inference_with_preference(obs,w);ar=r1.act_inference_with_preference(obs,w)
                    val2=v2.value_with_preference(obs,w);valr=r1.value_with_preference(obs,w)
                    zero=torch.zeros_like(ar)
                    lp2=v2.logp_with_preference(obs,w,zero);lpr=r1.logp_with_preference(obs,w,zero)
                    diffs[k]={'action_max_abs':float((a2-ar).abs().max()),'value_max_abs':float((val2-valr).abs().max()),'logp_max_abs':float((lp2-lpr).abs().max())}
            state_schema=set(v2.state_dict())==set(r1.state_dict())
            rows=[];cur=obs
            with torch.no_grad():
                w=prefs['C']
                for step in range(args.steps):
                    a=torch.clamp(r1.act_inference_with_preference(cur,w),-1,1);nxt,_,term,trunc,_=env.step(a)
                    rows.append({'step':step,'finite':bool(torch.isfinite(a).all()),'survival':float(1-((term|trunc).float().mean()))});cur=obs_tensor(nxt).cuda()
            maxdiff=max(max(x.values()) for x in diffs.values())
            ok=FILM_AUTHORITY_R1==0.5 and state_schema and maxdiff<1e-6 and all(x['finite'] for x in rows)
            report={'schema':'v2_r1_isaac_smoke_v1','status':'SMOKE_PASS' if ok else 'SMOKE_FAIL','training':False,
              'provenance':{'code_commit':CODE_COMMIT,'source_checkpoint':str(source_path.relative_to(ROOT)),'source_checkpoint_sha256':sha(source_path),'v2a_anchor_sha256':sha(anchor_path)},
              'film_authority':FILM_AUTHORITY_R1,'state_schema_identical_to_v2':state_schema,'function_preserving_vs_v2_at_init':diffs,
              'rollout':rows,'note':'No optimizer step. R1 differs only by fixed FiLM authority, which is identity-inactive at initialization.'}
            args.output.write_text(json.dumps(report,indent=2)+'\n');mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'max_diff':maxdiff,'film_authority':FILM_AUTHORITY_R1},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_r1_semantic_response_audit():
    """Run former post_v2_r1_semantic_response_audit.py stage."""
    """V2-R1 read-only semantic preference-response audit for fixed FiLM alpha=0.5."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CODE_COMMIT="f77af91932b1651b531cb3f64cb3c1852ab800cb"
    GRID={
     "P":[.8,.1,.1],"PB_30":[.6,.3,.1],"PB_45":[.45,.45,.1],"PB_60":[.3,.6,.1],"B":[.1,.8,.1],
     "PE_30":[.6,.1,.3],"PE_45":[.45,.1,.45],"PE_60":[.3,.1,.6],"E":[.1,.1,.8],
     "C":[1/3,1/3,1/3],
    }
    LINES={"P_to_B":["P","PB_30","PB_45","PB_60","B"],"P_to_E":["P","PE_30","PE_45","PE_60","E"]}
    EXPECTED={
     "P_to_B":{
       "objective_progress":"decrease","objective_balance":"increase",
       "vx_error":"increase","tilt_p95":"decrease","ang_vel_xy":"decrease",
     },
     "P_to_E":{
       "objective_progress":"decrease","objective_efficiency":"increase",
       "vx_error":"increase","torque_norm":"decrease","action_rate":"decrease",
     },
    }
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt_deg(q): return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    def directional_fraction(vals,direction,tol=1e-8):
        d=np.diff(np.asarray(vals,dtype=float))
        if direction=="increase": return float(np.mean(d>=-tol))
        if direction=="decrease": return float(np.mean(d<=tol))
        raise ValueError(direction)
    def endpoint_between_fraction(vals,tol=1e-8):
        v=np.asarray(vals,dtype=float); lo=min(v[0],v[-1])-tol; hi=max(v[0],v[-1])+tol
        return float(np.mean((v[1:-1]>=lo)&(v[1:-1]<=hi)))
    def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--steps",type=int,default=32)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        audit_sha=sha256(__file__); ck_sha=sha256(args.checkpoint)
        mark("RUN_STARTED",protocol="V2-R1-SEMANTIC",measurement_only=True,film_alpha=0.5,code_commit=CODE_COMMIT,checkpoint_sha256=ck_sha,audit_script_sha256=audit_sha)
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text()); anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=47001); obs=obs_tensor(obs).cuda(); ad=env.unwrapped.action_manager.total_action_dim
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            model=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda(); model.load_state_dict(payload["model"]); model.eval()
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in GRID.items()}
            rows=[]
            manager=env.unwrapped.reward_manager; robot=env.unwrapped.scene["robot"]
            with torch.no_grad():
                for suite in range(args.suites):
                    seed=57001+suite
                    for label in GRID:
                        cur,_=env.reset(seed=seed); cur=obs_tensor(cur).cuda(); prev=torch.zeros((args.num_envs,ad),device="cuda")
                        vecs=[]; mets=[]; done_any=np.zeros(args.num_envs,dtype=bool); finite=True
                        z=model.behavior_z(ws[label]).mean(0)
                        for _ in range(args.steps):
                            action=torch.clamp(model.act_inference_with_preference(cur,ws[label]),-1,1)
                            finite=finite and bool(torch.isfinite(action).all())
                            nxt,_,term,trunc,_=env.step(action); done_any|=(term|trunc).cpu().numpy()
                            raw=manager._step_reward.detach().cpu().numpy(); names=list(manager.active_terms)
                            vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                            data=robot.data; cmd=env.unwrapped.command_manager.get_command("base_velocity"); tilt=tilt_deg(data.root_quat_w)
                            vecs.append(vec.mean(0))
                            mets.append({
                              "vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),
                              "tilt_p95":float(torch.quantile(tilt,.95)),
                              "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                              "torque_norm":float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),
                              "action_rate":float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),
                            })
                            prev=action; cur=obs_tensor(nxt).cuda()
                        rows.append({
                          "suite":suite,"seed":seed,"preference":label,"w":GRID[label],"z":z.cpu().tolist(),
                          "objective_return":np.mean(vecs,axis=0).tolist(),
                          "metrics":{k:float(np.mean([m[k] for m in mets])) for k in mets[0]},
                          "survival":float(1-done_any.mean()),"finite":finite,
                        })
            profiles={}
            metric_keys=["objective_progress","objective_balance","objective_efficiency","vx_error","tilt_p95","ang_vel_xy","torque_norm","action_rate","survival"]
            for axis,labels in LINES.items():
                profiles[axis]={}
                for key in metric_keys:
                    suites=[]
                    for suite in range(args.suites):
                        rs=[next(r for r in rows if r["suite"]==suite and r["preference"]==lab) for lab in labels]
                        if key.startswith("objective_"):
                            idx={"objective_progress":0,"objective_balance":1,"objective_efficiency":2}[key]
                            vals=[r["objective_return"][idx] for r in rs]
                        else: vals=[r[key] if key=="survival" else r["metrics"][key] for r in rs]
                        item={"suite":suite,"values":vals,"endpoint_between_fraction":endpoint_between_fraction(vals)}
                        if key in EXPECTED[axis]:
                            item["expected_direction"]=EXPECTED[axis][key]
                            item["monotonicity_fraction"]=directional_fraction(vals,EXPECTED[axis][key])
                        suites.append(item)
                    profiles[axis][key]={
                      "suite_profiles":suites,
                      "mean_endpoint_between_fraction":float(np.mean([s["endpoint_between_fraction"] for s in suites])),
                    }
                    directed=[s.get("monotonicity_fraction") for s in suites if "monotonicity_fraction" in s]
                    if directed: profiles[axis][key]["mean_monotonicity_fraction"]=float(np.mean(directed))
            endpoint_summary={}
            for label in ("P","B","E","C"):
                rs=[r for r in rows if r["preference"]==label]
                endpoint_summary[label]={
                  "objective_return_mean":np.mean([r["objective_return"] for r in rs],axis=0).tolist(),
                  "metrics_mean":{k:float(np.mean([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]},
                  "survival_mean":float(np.mean([r["survival"] for r in rs])),
                }
            report={
              "schema":"post_v2_r1_semantic_response_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,"film_alpha":0.5,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":ck_sha,
                            "audit_script_sha256":audit_sha,"v2a_anchor_sha256":sha256(anchor_path)},
              "grid":GRID,"lines":LINES,"expected_directions":EXPECTED,"suites":args.suites,"steps":args.steps,
              "rows":rows,"profiles":profiles,"endpoint_summary":endpoint_summary,
              "note":"No training or parameter updates. R1 fixed FiLM alpha=0.5; semantic directions were fixed before measurement."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n"); mark("ARTIFACT_WRITTEN",path=str(args.output)); mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"rows":len(rows),"checkpoint_sha256":ck_sha},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n")
            mark("ERROR",error=str(exc)); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    if True: main()

def run_post_v2_r1_short_authority_screen():
    """Run former post_v2_r1_short_authority_screen.py stage."""
    """V2-R1 one-seed short authority screen: V2 with fixed FiLM alpha=0.5 only."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt_deg(q): return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    def early_ppo(ratio,adv,w,eps=.2):
        scalar=(adv*w).sum(-1); clipped=ratio.clamp(1-eps,1+eps); return -torch.minimum(ratio*scalar,clipped*scalar).mean()
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--source-checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=100);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--horizon',type=int,default=2);ap.add_argument('--eval-steps',type=int,default=16);ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(event,**extra):
            with life.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**extra},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V2-R1-SHORT-AUTHORITY-SCREEN',training=True,updates=args.updates,film_alpha=0.5);app=env=None
        try:
            anchor_report=json.loads((ROOT/'runs/post_v2_a-2026-09-23/v2a.json').read_text());anchors=torch.tensor([anchor_report['anchors'][k] for k in ('P','B','E')],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import vector_gae,vector_value_loss
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic,initialize_from_v1c
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;payload=torch.load(args.source_checkpoint,map_location='cpu',weights_only=False);source=payload['model'];model=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda();initialize_from_v1c(model,source);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+200000);mark('CHECKPOINT_LOADED',source=str(args.source_checkpoint))
            current,_=env.reset(seed=args.seed);current=obs_tensor(current).cuda();records=[];max_recon=0.0
            for update in range(1,args.updates+1):
                w=torch.as_tensor(rng.dirichlet(np.ones(3),size=args.num_envs),device='cuda',dtype=torch.float32);obs_buf=[];act_buf=[];old_buf=[];rew_buf=[];val_buf=[];done_buf=[]
                for _ in range(args.horizon):
                    with torch.no_grad(): action,old=model.act_with_preference(current,w);value=model.value_with_preference(current,w)
                    nxt,scalar,term,trunc,_=env.step(torch.clamp(action,-1,1));manager=env.unwrapped.reward_manager;raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));max_recon=max(max_recon,float(np.max(np.abs(vec.sum(axis=1)*0.0))) if False else 0.0);obs_buf.append(current);act_buf.append(action);old_buf.append(old);rew_buf.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val_buf.append(value);done_buf.append((term|trunc).to('cuda'));current=obs_tensor(nxt).cuda()
                with torch.no_grad(): next_value=model.value_with_preference(current,w)
                reward_t=torch.stack(rew_buf);value_t=torch.stack(val_buf);done_t=torch.stack(done_buf).bool();adv,ret=vector_gae(reward_t,value_t,next_value,done_t);flat_obs=torch.cat(obs_buf);flat_act=torch.cat(act_buf);flat_old=torch.cat(old_buf);flat_w=w.repeat(args.horizon,1);ratio=torch.exp(model.logp_with_preference(flat_obs,flat_w,flat_act)-flat_old.detach());ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),flat_w);manifold=model.manifold_loss(flat_w);critic=vector_value_loss(model.value_with_preference(flat_obs,flat_w),ret.reshape(-1,3).detach());loss=ppo+critic+0.1*manifold;optimizer.zero_grad(set_to_none=True);loss.backward();finite=bool(torch.isfinite(loss).item() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()));optimizer.step();records.append({'update':update,'loss':float(loss.detach()),'ppo_loss':float(ppo.detach()),'critic_loss':float(critic.detach()),'manifold_loss':float(manifold.detach()),'z_norm':float(model.behavior_z(flat_w).norm(dim=-1).mean().detach()),'action_mean_abs':float(flat_act.abs().mean()),'mean_w':w.mean(0).tolist(),'finite':finite})
            terminal=args.output.parent/'v2r1_terminal.pt';torch.save({'schema':'post_v2_r1_terminal_v1','seed':args.seed,'update':args.updates,'film_alpha':0.5,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'anchors':anchors.tolist()},terminal);mark('CHECKPOINT_WRITTEN',checkpoint=str(terminal))
            model.eval();behavior=[]
            with torch.no_grad():
                for label,pref in PREFS.items():
                    w=torch.as_tensor(np.repeat(pref[None,:],args.num_envs,axis=0),device='cuda');cur,_=env.reset(seed=92000+list(PREFS).index(label));cur=obs_tensor(cur).cuda();initial=torch.clamp(model.act_inference_with_preference(cur,w),-1,1);z=model.behavior_z(w);rewards=[];metrics=[];done_any=np.zeros(args.num_envs,dtype=bool);prev=torch.zeros_like(initial)
                    for _ in range(args.eval_steps):
                        action=torch.clamp(model.act_inference_with_preference(cur,w),-1,1);nxt,_,term,trunc,_=env.step(action);done_any|=(term|trunc).cpu().numpy();raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');t=tilt_deg(data.root_quat_w);rewards.append(vec);metrics.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'tilt_p95':float(torch.quantile(t,.95)),'torque_norm':float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),'action_rate':float(torch.linalg.vector_norm(action-prev,dim=-1).mean())});prev=action;cur=obs_tensor(nxt).cuda()
                    behavior.append({'preference':label,'z_mean':z.mean(0).tolist(),'initial_action_mean_abs':float(initial.abs().mean()),'objective_return':np.concatenate(rewards).mean(0).tolist(),'metrics':{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},'survival':float(1-done_any.mean()),'finite':bool(np.isfinite(np.concatenate(rewards)).all())})
            action_dist={};
            for i,a in enumerate(('P','B','E')):
                for b in ('P','B','E')[i+1:]: action_dist[f'{a}_vs_{b}']=float(np.linalg.norm(np.asarray(behavior[i]['initial_action_mean_abs'])-np.asarray(behavior[list(PREFS).index(b)]['initial_action_mean_abs'])))
            report={'schema':'post_v2_r1_short_authority_screen_v1','status':'SCREEN_COMPLETE','training':True,'diagnostic_only':True,'seed':args.seed,'updates':args.updates,'film_alpha':0.5,'source_checkpoint':str(args.source_checkpoint),'terminal_checkpoint':str(terminal),'lambda_manifold':0.1,'records':records,'behavior':behavior,'note':'One-seed V2-R1 authority screen; the only treatment change is fixed FiLM alpha 0.1 -> 0.5. No semantic-pass verdict and no full-run authorization.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'updates':args.updates,'terminal':str(terminal)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_r21_shared_residual_geometry_audit():
    """Run former post_v2_r21_shared_residual_geometry_audit.py stage."""
    """V2-R2.1 read-only shared/residual hidden-geometry audit."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    import torch.nn.functional as F
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    CODE_COMMIT="f77af91932b1651b531cb3f64cb3c1852ab800cb"
    EPS=1e-12
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def d1_hidden(m,obs,w):
        x=torch.cat((obs,w),dim=-1)
        h1=F.elu(m.actor_body[0](x))
        h2=F.elu(m.actor_body[2](h1))
        h3=F.elu(m.actor_body[4](h2))
        return h1,h3
    
    def v2_hidden(m,obs,w):
        z=m.behavior_z(w)
        h0=F.elu(m.actor_pre(obs))
        h1=(1+m.film_alpha*torch.tanh(m.film_gamma(z)))*h0 + m.film_alpha*torch.tanh(m.film_beta(z))
        h3=m.actor_rest(h1)
        return h1,h3
    
    def normalize(v):
        return v/(torch.linalg.vector_norm(v,dim=-1,keepdim=True)+EPS)
    
    def corr(a,b):
        a=np.asarray(a,float); b=np.asarray(b,float)
        if a.size<2 or np.std(a)<1e-12 or np.std(b)<1e-12: return None
        return float(np.corrcoef(a,b)[0,1])
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--steps",type=int,default=16)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint); mark("RUN_STARTED",protocol="V2-R2.1",measurement_only=True,checkpoint_sha256=cksha)
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text()); anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.rewards.baselines import group_v1b_s7_terms
    
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=127001); obs=obs_tensor(obs).cuda(); ad=env.unwrapped.action_manager.total_action_dim
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda(); v2.load_state_dict(payload["model"]); v2.eval()
            specs={}
            for lab in PREFS:
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt"; pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda(); m.load_state_dict(pl["model"]); m.eval(); specs[lab]=m
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
    
            rows=[]
            for suite in range(args.suites):
                cur,_=env.reset(seed=127001+suite); cur=obs_tensor(cur).cuda()
                for t in range(args.steps):
                    with torch.no_grad():
                        dh={k:d1_hidden(specs[k],cur,ws[k]) for k in PREFS}
                        vh={k:v2_hidden(v2,cur,ws[k]) for k in PREFS}
                        raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy()
                        names=list(env.unwrapped.reward_manager.active_terms)
                        vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                        obj=vec.mean(0)
                        data=env.unwrapped.scene["robot"].data
                        cmd=env.unwrapped.command_manager.get_command("base_velocity")
                        bal_proxy=float(-(data.root_ang_vel_b[:,:2].pow(2).sum(-1)).mean())
                        eff_proxy=float(-(torch.linalg.vector_norm(data.applied_torque,dim=-1)).mean())
                        for layer,idx in (("h1",0),("h3",1)):
                            dPB=dh["B"][idx]-dh["P"][idx]
                            dPE=dh["E"][idx]-dh["P"][idx]
                            dBE=dh["E"][idx]-dh["B"][idx]
                            shared=normalize((dPB+dPE)/2.0)
                            be_orth=dBE-(dBE*shared).sum(-1,keepdim=True)*shared
                            be=normalize(be_orth)
                            for fam,h in (("D1",dh),("V2",vh)):
                                dPBf=h["B"][idx]-h["P"][idx]
                                dPEf=h["E"][idx]-h["P"][idx]
                                dBEf=h["E"][idx]-h["B"][idx]
                                for name,d in (("PB",dPBf),("PE",dPEf),("BE",dBEf)):
                                    cs=(d*shared).sum(-1)
                                    cb=(d*be).sum(-1)
                                    rows.append({
                                      "suite":suite,"t":t,"layer":layer,"family":fam,"delta":name,
                                      "c_shared_mean":float(cs.mean()),"c_BE_mean":float(cb.mean()),
                                      "c_shared_sign_fraction_pos":float((cs>0).float().mean()),
                                      "c_BE_sign_fraction_pos":float((cb>0).float().mean()),
                                      "shared_norm_mean":float(torch.linalg.vector_norm((dPB+dPE)/2.0,dim=-1).mean()),
                                      "BE_residual_norm_mean":float(torch.linalg.vector_norm(be_orth,dim=-1).mean()),
                                      "objective_progress":float(obj[0]),"objective_balance":float(obj[1]),"objective_efficiency":float(obj[2]),
                                      "balance_proxy":bal_proxy,"efficiency_proxy":eff_proxy,
                                    })
                        action=v2.act_inference_with_preference(cur,ws["P"])
                        nxt,_,_,_,_=env.step(torch.clamp(action,-1,1)); cur=obs_tensor(nxt).cuda()
    
            summary={}
            for layer in ("h1","h3"):
                summary[layer]={}
                for fam in ("D1","V2"):
                    summary[layer][fam]={}
                    for delta in ("PB","PE","BE"):
                        rs=[r for r in rows if r["layer"]==layer and r["family"]==fam and r["delta"]==delta]
                        cs=[r["c_shared_mean"] for r in rs]; cb=[r["c_BE_mean"] for r in rs]
                        suite_shared=[]; suite_be=[]
                        for s in range(args.suites):
                            sr=[r for r in rs if r["suite"]==s]
                            suite_shared.append(float(np.mean([r["c_shared_mean"] for r in sr])))
                            suite_be.append(float(np.mean([r["c_BE_mean"] for r in sr])))
                        sign_shared=[np.sign(r["c_shared_mean"]) for r in rs if abs(r["c_shared_mean"])>1e-9]
                        sign_be=[np.sign(r["c_BE_mean"]) for r in rs if abs(r["c_BE_mean"])>1e-9]
                        summary[layer][fam][delta]={
                          "c_shared_mean":float(np.mean(cs)),"c_shared_std":float(np.std(cs)),
                          "c_BE_mean":float(np.mean(cb)),"c_BE_std":float(np.std(cb)),
                          "c_shared_temporal_sign_consistency":float(max(sign_shared.count(1),sign_shared.count(-1))/len(sign_shared)) if sign_shared else None,
                          "c_BE_temporal_sign_consistency":float(max(sign_be.count(1),sign_be.count(-1))/len(sign_be)) if sign_be else None,
                          "c_shared_cross_suite_cv":float(np.std(suite_shared)/(abs(np.mean(suite_shared))+EPS)),
                          "c_BE_cross_suite_cv":float(np.std(suite_be)/(abs(np.mean(suite_be))+EPS)),
                          "corr_cshared_balance_obj":corr(cs,[r["objective_balance"] for r in rs]),
                          "corr_cBE_balance_obj":corr(cb,[r["objective_balance"] for r in rs]),
                          "corr_cshared_eff_obj":corr(cs,[r["objective_efficiency"] for r in rs]),
                          "corr_cBE_eff_obj":corr(cb,[r["objective_efficiency"] for r in rs]),
                          "corr_cBE_balance_proxy":corr(cb,[r["balance_proxy"] for r in rs]),
                          "corr_cBE_eff_proxy":corr(cb,[r["efficiency_proxy"] for r in rs]),
                        }
            # D1 basis stability itself
            basis_stats={}
            for layer in ("h1","h3"):
                rs=[r for r in rows if r["layer"]==layer and r["family"]=="D1" and r["delta"]=="PB"]
                basis_stats[layer]={
                  "shared_norm_mean":float(np.mean([r["shared_norm_mean"] for r in rs])),
                  "BE_residual_norm_mean":float(np.mean([r["BE_residual_norm_mean"] for r in rs])),
                  "residual_over_shared":float(np.mean([r["BE_residual_norm_mean"]/(r["shared_norm_mean"]+EPS) for r in rs]))
                }
            report={
              "schema":"post_v2_r21_shared_residual_geometry_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha},
              "basis_definition":{"shared":"normalize((dPB+dPE)/2)","BE":"normalize(dBE-proj_shared(dBE))"},
              "steps":args.steps,"suites":args.suites,"summary":summary,"basis_stats":basis_stats,
              "note":"Basis built pointwise from D1 hidden differences on matched V2-P trajectory states; coefficients measured for D1 and V2. No training."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n"); mark("ARTIFACT_WRITTEN",path=str(args.output)); mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"basis_stats":basis_stats},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    
    if True: main()

def run_post_v2_r22_coefficient_gradient_audit():
    """Run former post_v2_r22_coefficient_gradient_audit.py stage."""
    """V2-R2.2 diagnostic-only coefficient gradient orientation audit."""
    import argparse,hashlib,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    from torch.distributions import Normal
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":np.array([.8,.1,.1],np.float32),"B":np.array([.1,.8,.1],np.float32),"E":np.array([.1,.1,.8],np.float32)}
    SNAPS=(0,1,5,10); EPS=1e-12
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def early_ppo(ratio,adv,w,eps=.2):
        scalar=(adv*w).sum(-1);clipped=ratio.clamp(1-eps,1+eps)
        return -torch.minimum(ratio*scalar,clipped*scalar).mean()
    def discounted_returns(rew,done,gamma=.99):
        out=torch.zeros_like(rew);running=torch.zeros_like(rew[0])
        for t in range(rew.shape[0]-1,-1,-1):
            running=rew[t]+gamma*running*(~done[t]).unsqueeze(-1)
            out[t]=running
        return out
    def logp_with_coeff(model,obs,coeff,action):
        h=torch.nn.functional.elu(model.actor_pre(obs))
        h=h+model.projection_alpha*(coeff@model.semantic_basis)
        h=model.actor_rest(h);mean=model.actor_mean(h)
        std=(model.log_std if model.exploration_mode=="learned" else model.scheduled_log_std).exp()
        dist=Normal(mean,std)
        u=torch.atanh((action/model.ACTION_CLIP).clamp(-1+model._ATANH_EPS,1-model._ATANH_EPS))
        return model._squash(dist,u)[1]
    def sign(x,tol=1e-10):return 0 if abs(x)<=tol else (1 if x>0 else -1)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--source-checkpoint",type=Path,required=True);ap.add_argument("--basis",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True);ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--horizon",type=int,default=2);ap.add_argument("--probe-horizon",type=int,default=8);ap.add_argument("--suites",type=int,default=4);ap.add_argument("--seed",type=int,default=0)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        mark("RUN_STARTED",protocol="V2-R2.2",diagnostic_only=True,snapshots=list(SNAPS));app=env=None
        try:
            anchors_j=json.loads((ROOT/"runs/post_v2_a-2026-09-23/v2a.json").read_text());anchors=torch.tensor([anchors_j["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            r21=json.loads((ROOT/"runs/post_v2_r21_shared_residual-2026-09-23/audit.json").read_text())
            d1_target={"PB":[1,-1],"PE":[1,1],"BE":[0,1]}
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import vector_gae,vector_value_loss
            from talon_rl.models.behavior.projected import V2R2ProjectedActorCritic,initialize_from_v1c
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            source=torch.load(args.source_checkpoint,map_location="cpu",weights_only=False)["model"]
            bp=torch.load(args.basis,map_location="cuda",weights_only=False);basis=torch.stack([bp["b_shared"],bp["b_BE"]],0).cuda()
            model=V2R2ProjectedActorCritic(obs.shape[-1],ad,basis,anchors=anchors,projection_alpha=1.0).cuda();initialize_from_v1c(model,source)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+300000)
            current,_=env.reset(seed=args.seed);current=obs_tensor(current).cuda()
            rows=[]
    
            def probe(snapshot_update):
                model.eval(); pref_results={}
                for pi,(label,pv) in enumerate(PREFS.items()):
                    suite_rows=[]
                    for suite in range(args.suites):
                        w=torch.as_tensor(np.repeat(pv[None,:],args.num_envs,0),device="cuda")
                        cur,_=env.reset(seed=150000+snapshot_update*100+pi*10+suite);cur=obs_tensor(cur).cuda()
                        obs_b=[];act_b=[];old_b=[];rew_b=[];val_b=[];done_b=[]
                        for _ in range(args.probe_horizon):
                            with torch.no_grad(): action,old=model.act_with_preference(cur,w);value=model.value_with_preference(cur,w)
                            nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1))
                            mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                            vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                            obs_b.append(cur);act_b.append(action);old_b.append(old);rew_b.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);val_b.append(value);done_b.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
                        with torch.no_grad():nv=model.value_with_preference(cur,w)
                        rew=torch.stack(rew_b);vals=torch.stack(val_b);done=torch.stack(done_b).bool()
                        gae,_=vector_gae(rew,vals,nv,done);rtg=discounted_returns(rew,done)
                        fo=torch.cat(obs_b);fa=torch.cat(act_b);fold=torch.cat(old_b).detach();fw=w.repeat(args.probe_horizon,1)
                        coeff0=model.semantic_coefficients(fw).detach().requires_grad_(True)
                        lp=logp_with_coeff(model,fo,coeff0,fa);ratio=torch.exp(lp-fold)
                        def grad_for(adv):
                            loss=early_ppo(ratio,adv.reshape(-1,3).detach(),fw)
                            g=torch.autograd.grad(loss,coeff0,retain_graph=True)[0]
                            return (-g).mean(0)
                        g_gae=grad_for(gae);g_rtg=grad_for(rtg)
                        obj=[]
                        for j in range(3):
                            mask=torch.zeros_like(gae);mask[:,:,j]=gae[:,:,j]
                            obj.append(grad_for(mask))
                        suite_rows.append({
                          "suite":suite,"coeff":model.semantic_coefficients(w)[0].detach().cpu().tolist(),
                          "update_dir_gae":g_gae.detach().cpu().tolist(),"update_dir_rtg":g_rtg.detach().cpu().tolist(),
                          "objective_update_dirs":[x.detach().cpu().tolist() for x in obj],
                          "adv_mean":gae.mean((0,1)).detach().cpu().tolist(),"rtg_mean":rtg.mean((0,1)).detach().cpu().tolist()
                        })
                    pref_results[label]=suite_rows
                # Aggregate preference pressure then implied pairwise orientation
                agg={}
                for label,rs in pref_results.items():
                    for mode in ("update_dir_gae","update_dir_rtg"):
                        agg.setdefault(label,{})[mode]=np.mean([r[mode] for r in rs],axis=0).tolist()
                    agg[label]["objective_update_dirs"]=np.mean([r["objective_update_dirs"] for r in rs],axis=0).tolist()
                pair={}
                for a,b,name in (("P","B","PB"),("P","E","PE"),("B","E","BE")):
                    pair[name]={}
                    for mode in ("update_dir_gae","update_dir_rtg"):
                        v=np.asarray(agg[b][mode])-np.asarray(agg[a][mode]);target=np.asarray(d1_target[name],float)
                        cos=float(v@target/(np.linalg.norm(v)*np.linalg.norm(target)+EPS))
                        pair[name][mode]={"vector":v.tolist(),"sign":[sign(x) for x in v],"target_sign":d1_target[name],"cosine_to_D1_target":cos}
                return {"update":snapshot_update,"preferences":pref_results,"aggregate":agg,"pairwise_pressure":pair}
            rows.append(probe(0));model.train()
    
            for update in range(1,max(SNAPS)+1):
                w=torch.as_tensor(rng.dirichlet(np.ones(3),size=args.num_envs),device="cuda",dtype=torch.float32)
                ob=[];ac=[];ol=[];rw=[];va=[];dn=[]
                for _ in range(args.horizon):
                    with torch.no_grad():action,old=model.act_with_preference(current,w);value=model.value_with_preference(current,w)
                    nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1));mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                    ob.append(current);ac.append(action);ol.append(old);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);va.append(value);dn.append((term|trunc).cuda());current=obs_tensor(nxt).cuda()
                with torch.no_grad():nv=model.value_with_preference(current,w)
                rew=torch.stack(rw);vals=torch.stack(va);done=torch.stack(dn).bool();adv,ret=vector_gae(rew,vals,nv,done)
                fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(ol);fw=w.repeat(args.horizon,1)
                ratio=torch.exp(model.logp_with_preference(fo,fw,fa)-fold.detach());ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),fw)
                manifold=model.manifold_loss(fw);critic=vector_value_loss(model.value_with_preference(fo,fw),ret.reshape(-1,3).detach());loss=ppo+critic+0.1*manifold
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if update in SNAPS:
                    rows.append(probe(update));model.train();mark("SNAPSHOT",update=update)
    
            report={"schema":"post_v2_r22_coefficient_gradient_audit_v1","status":"MEASUREMENT_COMPLETE","diagnostic_only":True,
              "source_checkpoint":str(args.source_checkpoint),"basis":str(args.basis),"basis_sha256":sha(args.basis),
              "snapshots":rows,"d1_target_signs":d1_target,
              "definitions":{"update_direction":"-dL_PPO/d coefficient output","pairwise_pressure":"update_direction(target)-update_direction(P/reference)",
                             "rtg_comparator":"discounted vector reward-to-go without critic baseline"},
              "note":"Training replay only reconstructs frozen R2 early updates; no architecture/loss/optimizer changes."}
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE")
            print(json.dumps({"status":report["status"],"snapshots":[x["update"] for x in rows]},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_r23a_coeff_guidance_gradient_audit():
    """Run former post_v2_r23a_coeff_guidance_gradient_audit.py stage."""
    """V2-R2.3-A read-only semantic coefficient-guidance gradient compatibility audit."""
    import argparse,hashlib,json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    from torch.distributions import Normal
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":np.array([.8,.1,.1],np.float32),"B":np.array([.1,.8,.1],np.float32),"E":np.array([.1,.1,.8],np.float32)}
    SNAPS=(0,1,5,10)
    LAMBDAS=(0.0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0)
    EPS=1e-12
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def early_ppo(ratio,adv,w,eps=.2):
        scalar=(adv*w).sum(-1);clipped=ratio.clamp(1-eps,1+eps)
        return -torch.minimum(ratio*scalar,clipped*scalar).mean()
    def sign(v,tol=1e-12):return 0 if abs(v)<=tol else (1 if v>0 else -1)
    def flat_grads(grads):
        xs=[]
        for g in grads:
            if g is not None: xs.append(g.reshape(-1))
        return torch.cat(xs) if xs else torch.zeros(1,device="cuda")
    def target_coeff(w,verts):
        return w@verts
    
    def logp_with_coeff(model,obs,coeff,action):
        h=torch.nn.functional.elu(model.actor_pre(obs))
        h=h+model.projection_alpha*(coeff@model.semantic_basis)
        h=model.actor_rest(h);mean=model.actor_mean(h)
        std=(model.log_std if model.exploration_mode=="learned" else model.scheduled_log_std).exp()
        dist=Normal(mean,std)
        u=torch.atanh((action/model.ACTION_CLIP).clamp(-1+model._ATANH_EPS,1-model._ATANH_EPS))
        return model._squash(dist,u)[1]
    def pair_ok(v,target):
        if target[0]!=0 and sign(v[0])!=target[0]: return False
        if target[1]!=0 and sign(v[1])!=target[1]: return False
        return True
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--source-checkpoint",type=Path,required=True);ap.add_argument("--basis",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True);ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--horizon",type=int,default=2);ap.add_argument("--probe-horizon",type=int,default=8);ap.add_argument("--suites",type=int,default=4);ap.add_argument("--seed",type=int,default=0)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        anchors_j=json.loads((ROOT/"runs/post_v2_a-2026-09-23/v2a.json").read_text());anchors=torch.tensor([anchors_j["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
        r21=json.loads((ROOT/"runs/post_v2_r21_shared_residual-2026-09-23/audit.json").read_text())
        d1=r21["summary"]["h1"]["D1"]
        cP=np.array([0.0,0.0],np.float32)
        cB=np.array([d1["PB"]["c_shared_mean"],d1["PB"]["c_BE_mean"]],np.float32)
        cE=np.array([d1["PE"]["c_shared_mean"],d1["PE"]["c_BE_mean"]],np.float32)
        verts_cpu=np.stack([cP,cB,cE],0)
        targets={"PB":[1,-1],"PE":[1,1],"BE":[0,1]}
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import vector_gae,vector_value_loss
            from talon_rl.models.behavior.projected import V2R2ProjectedActorCritic,initialize_from_v1c
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            source=torch.load(args.source_checkpoint,map_location="cpu",weights_only=False)["model"]
            bp=torch.load(args.basis,map_location="cuda",weights_only=False);basis=torch.stack([bp["b_shared"],bp["b_BE"]],0).cuda()
            verts=torch.tensor(verts_cpu,device="cuda")
            model=V2R2ProjectedActorCritic(obs.shape[-1],ad,basis,anchors=anchors,projection_alpha=1.0).cuda();initialize_from_v1c(model,source)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+300000)
            current,_=env.reset(seed=args.seed);current=obs_tensor(current).cuda()
            rows=[]
            coeff_params=list(model.coeff_net.parameters())
    
            def probe(update):
                model.eval(); pref={}
                for pi,(label,pv) in enumerate(PREFS.items()):
                    suites=[]
                    for suite in range(args.suites):
                        w=torch.as_tensor(np.repeat(pv[None,:],args.num_envs,0),device="cuda")
                        cur,_=env.reset(seed=160000+update*100+pi*10+suite);cur=obs_tensor(cur).cuda()
                        ob=[];ac=[];ol=[];rw=[];va=[];dn=[]
                        for _ in range(args.probe_horizon):
                            with torch.no_grad():action,old=model.act_with_preference(cur,w);value=model.value_with_preference(cur,w)
                            nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1));mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                            vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                            ob.append(cur);ac.append(action);ol.append(old);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);va.append(value);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
                        with torch.no_grad():nv=model.value_with_preference(cur,w)
                        rew=torch.stack(rw);vals=torch.stack(va);done=torch.stack(dn).bool();adv,_=vector_gae(rew,vals,nv,done)
                        fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(ol).detach();fw=w.repeat(args.probe_horizon,1)
                        # Direct coefficient-output pressure.
                        coeff=model.semantic_coefficients(fw).detach().requires_grad_(True)
                        lp=logp_with_coeff(model,fo,coeff,fa);ratio=torch.exp(lp-fold);ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),fw)
                        cstar=target_coeff(fw,verts)
                        lc=((coeff-cstar)**2).mean()
                        gp_out=torch.autograd.grad(ppo,coeff,retain_graph=True,allow_unused=False)[0]
                        gc_out=torch.autograd.grad(lc,coeff,retain_graph=True,allow_unused=False)[0]
                        # Parameter-space compatibility: recompute losses through coeff_net parameters.
                        coeff_param=model.semantic_coefficients(fw)
                        lp_param=logp_with_coeff(model,fo,coeff_param,fa)
                        ppo_param=early_ppo(torch.exp(lp_param-fold),adv.reshape(-1,3).detach(),fw)
                        lc_param=((coeff_param-cstar)**2).mean()
                        gp=flat_grads(torch.autograd.grad(ppo_param,coeff_params,retain_graph=True,allow_unused=True))
                        gc=flat_grads(torch.autograd.grad(lc_param,coeff_params,retain_graph=True,allow_unused=True))
                        npp=float(gp.norm()); nc=float(gc.norm()); cos=float((gp@gc)/(gp.norm()*gc.norm()+EPS))
                        suites.append({
                          "suite":suite,"coeff_mean":coeff.detach().mean(0).cpu().tolist(),"target_mean":cstar.detach().mean(0).cpu().tolist(),
                          "ppo_update_dir_out":(-gp_out).mean(0).detach().cpu().tolist(),
                          "coeff_update_dir_out":(-gc_out).mean(0).detach().cpu().tolist(),
                          "gppo_param_norm":npp,"gcoeff_param_norm":nc,"param_grad_cosine":cos,
                          "raw_ratio_gcoeff_over_gppo":nc/(npp+EPS)
                        })
                    pref[label]=suites
                # candidate lambda combined output pressure
                cand={}
                for lam in LAMBDAS:
                    ag={}
                    for label,rs in pref.items():
                        up=np.mean([np.array(r["ppo_update_dir_out"])+lam*np.array(r["coeff_update_dir_out"]) for r in rs],axis=0)
                        ag[label]=up
                    pairs={}
                    allok=True
                    for a,b,name in (("P","B","PB"),("P","E","PE"),("B","E","BE")):
                        v=ag[b]-ag[a];ok=pair_ok(v,targets[name]);allok=allok and ok
                        suites_ok=[]
                        for i in range(args.suites):
                            va=np.array(pref[a][i]["ppo_update_dir_out"])+lam*np.array(pref[a][i]["coeff_update_dir_out"])
                            vb=np.array(pref[b][i]["ppo_update_dir_out"])+lam*np.array(pref[b][i]["coeff_update_dir_out"])
                            suites_ok.append(pair_ok(vb-va,targets[name]))
                        pairs[name]={"vector":v.tolist(),"sign":[sign(x) for x in v],"target":targets[name],"aggregate_ok":ok,"suite_match_fraction":float(np.mean(suites_ok))}
                    # parameter weighted ratio averaged over pref/suites
                    wr=np.mean([lam*r["raw_ratio_gcoeff_over_gppo"] for rs in pref.values() for r in rs])
                    cand[str(lam)]={"pairs":pairs,"all_aggregate_ok":allok,"mean_lambda_gcoeff_over_gppo":float(wr)}
                return {"update":update,"preferences":pref,"lambda_candidates":cand}
            rows.append(probe(0));model.train()
    
            for update in range(1,max(SNAPS)+1):
                w=torch.as_tensor(rng.dirichlet(np.ones(3),size=args.num_envs),device="cuda",dtype=torch.float32)
                ob=[];ac=[];ol=[];rw=[];va=[];dn=[]
                for _ in range(args.horizon):
                    with torch.no_grad():action,old=model.act_with_preference(current,w);value=model.value_with_preference(current,w)
                    nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1));mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                    ob.append(current);ac.append(action);ol.append(old);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);va.append(value);dn.append((term|trunc).cuda());current=obs_tensor(nxt).cuda()
                with torch.no_grad():nv=model.value_with_preference(current,w)
                rew=torch.stack(rw);vals=torch.stack(va);done=torch.stack(dn).bool();adv,ret=vector_gae(rew,vals,nv,done)
                fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(ol);fw=w.repeat(args.horizon,1)
                ratio=torch.exp(model.logp_with_preference(fo,fw,fa)-fold.detach());ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),fw)
                manifold=model.manifold_loss(fw);critic=vector_value_loss(model.value_with_preference(fo,fw),ret.reshape(-1,3).detach());loss=ppo+critic+0.1*manifold
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if update in SNAPS:rows.append(probe(update));model.train()
    
            # smallest lambda satisfying all aggregate signs at all audited snapshots
            effective=[]
            for lam in LAMBDAS:
                key=str(lam)
                if all(s["lambda_candidates"][key]["all_aggregate_ok"] for s in rows):
                    effective.append(lam)
            min_eff=effective[0] if effective else None
            report={
              "schema":"post_v2_r23a_coeff_guidance_gradient_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "source_checkpoint":str(args.source_checkpoint),"basis":str(args.basis),"basis_sha256":sha(args.basis),
              "coefficient_vertices":{"P":cP.tolist(),"B":cB.tolist(),"E":cE.tolist()},
              "target_definition":"c*(w)=w_P cP + w_B cB + w_E cE",
              "lambda_candidates":list(LAMBDAS),"snapshots":rows,
              "minimum_all_snapshot_aggregate_orientation_lambda":min_eff,
              "acceptance_rule":"minimum lambda whose combined output pressure has correct aggregate D1 signs for PB/PE/BE at updates 0/1/5/10",
              "note":"Read-only gradient algebra. R2 replay uses original PPO training only; coefficient guidance is never applied to optimizer."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({"status":report["status"],"minimum_lambda":min_eff},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");raise
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_post_v2_r23c_normalized_guidance_audit():
    """Run former post_v2_r23c_normalized_guidance_audit.py stage."""
    """How much normalised guidance is needed to orient the pairwise semantics?
    
    Read-only algebra over the R2.3-A coefficient pressures: for each budget rho,
    add a guidance vector of norm rho * ||PPO pressure|| and check whether the
    three preference-pair differences then point the way they should. Reports the
    smallest rho that works at every snapshot, and what the combination costs in
    cosine against PPO's own direction. No optimizer or model update.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import REPO, OfflineAudit
    
    RHO = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    EPS = 1e-12
    TARGETS = {"PB": np.array([1., -1.]), "PE": np.array([1., 1.]), "BE": np.array([0., 1.])}
    PAIRS = (("P", "B", "PB"), ("P", "E", "PE"), ("B", "E", "BE"))
    
    
    def sign(x, tol=1e-12):
        return 0 if abs(x) <= tol else (1 if x > 0 else -1)
    
    
    def pair_ok(v, t):
        """A zero in the target means that component is unconstrained."""
        if t[0] and sign(v[0]) != int(t[0]):
            return False
        if t[1] and sign(v[1]) != int(t[1]):
            return False
        return True
    
    
    def cosine(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
    
    
    class NormalizedGuidanceAudit(OfflineAudit):
        """Smallest guidance budget that orients every preference pair."""
    
        run = "post_v2_r23c_normalized_guidance-2026-09-23"
        report = "audit.json"
        schema = "post_v2_r23c_normalized_guidance_audit_v1"
    
        def __init__(self, out=None, source=None):
            super().__init__(out)
            self.source = Path(source) if source else self.dir.parent / "post_v2_r23a_coeff_guidance-2026-09-23" / "audit.json"
    
        def rho_row(self, snapshot, rho):
            pref_comb, pref_cos = {}, {}
            for pref, rs in snapshot["preferences"].items():
                comb, cos = [], []
                for r in rs:
                    p = np.asarray(r["ppo_update_dir_out"], float)
                    g = np.asarray(r["coeff_update_dir_out"], float)
                    gn = np.linalg.norm(g)
                    guide = np.zeros_like(g) if gn < EPS else rho * np.linalg.norm(p) * g / (gn + EPS)
                    total = p + guide
                    comb.append(total)
                    cos.append(cosine(total, p))
                pref_comb[pref], pref_cos[pref] = comb, cos
    
            pairs, all_agg, all_suite = {}, True, True
            for a, b, name in PAIRS:
                arr = np.asarray(pref_comb[b]) - np.asarray(pref_comb[a])
                mean, target = arr.mean(0), TARGETS[name]
                suite_ok = [pair_ok(v, target) for v in arr]
                agg_ok = pair_ok(mean, target)
                all_agg &= agg_ok
                all_suite &= all(suite_ok)
                pairs[name] = {"mean_vector": mean.tolist(),
                               "sign": [sign(x) for x in mean],
                               "target_sign": [int(x) for x in target],
                               "aggregate_ok": bool(agg_ok),
                               "suite_match_fraction": float(np.mean(suite_ok)),
                               "mean_cosine_to_target": float(np.mean([cosine(v, target) for v in arr]))}
            all_cos = [x for pref in pref_cos.values() for x in pref]
            return {"pairs": pairs, "all_aggregate_ok": bool(all_agg), "all_suite_ok": bool(all_suite),
                    "mean_combined_vs_ppo_cosine": float(np.mean(all_cos)),
                    "min_combined_vs_ppo_cosine": float(np.min(all_cos)),
                    "guide_norm_budget_ratio": rho}
    
        def analyze(self):
            d = json.loads(self.source.read_text())
            snaps = [{"update": s["update"],
                      "rho_candidates": {str(rho): self.rho_row(s, rho) for rho in RHO}}
                     for s in d["snapshots"]]
            first = lambda key: next(  # noqa: E731
                (rho for rho in RHO if all(s["rho_candidates"][str(rho)][key] for s in snaps)), None)
            return {
                "schema": self.schema,
                "status": "MEASUREMENT_COMPLETE", "measurement_only": True,
                # recorded repo-relative, as the original invocation did
                "source_r23a_audit": str(self.source.relative_to(REPO)
                                         if self.source.is_absolute() else self.source),
                "source_r23a_sha256": self.sha(self.source),
                "rho_candidates": list(RHO),
                "snapshots": snaps,
                "minimum_all_snapshot_aggregate_orientation_rho": first("all_aggregate_ok"),
                "minimum_all_snapshot_all_suite_orientation_rho": first("all_suite_ok"),
                "acceptance_rule": {
                    "orientation": "PB/PE/BE aggregate signs correct at updates 0/1/5/10",
                    "budget": "guide norm fixed to rho * ||PPO output-pressure|| per preference/suite",
                    "ppo_preservation": "report combined-vs-PPO cosine; no training authorization in this audit"},
                "note": "Pure read-only algebra on R2.3-A coefficient-output pressures. No optimizer/model updates."}
    
        def summarize(self, rep):
            print(json.dumps({"status": rep["status"],
                              "min_aggregate_rho": rep["minimum_all_snapshot_aggregate_orientation_rho"],
                              "min_all_suite_rho": rep["minimum_all_snapshot_all_suite_orientation_rho"]},
                             indent=2))
    
        @classmethod
        def main(cls):
            args = cls.parse_args((("--r23a-audit",), {"help": "the R2.3-A audit to read"}))
            cls(args.out, getattr(args, "r23a_audit", None)).execute()
    
    
    if True:
        NormalizedGuidanceAudit.main()

def run_post_v2_r23d_min_norm_constraint_audit():
    """Run former post_v2_r23d_min_norm_constraint_audit.py stage."""
    """V2-R2.3-D read-only minimum-norm pairwise semantic orientation correction audit."""
    import argparse,hashlib,json
    from pathlib import Path
    import numpy as np
    from scipy.optimize import minimize
    
    ROOT=Path(__file__).resolve().parents[4]
    EPS=1e-12
    MARGIN=1e-6
    PAIRS=(("P","B","PB"),("P","E","PE"),("B","E","BE"))
    TARGETS={"PB":np.array([1.,-1.]),"PE":np.array([1.,1.]),"BE":np.array([0.,1.])}
    IDX={"P":0,"B":2,"E":4}
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def cosine(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+EPS))
    def pair_vec(flat,a,b):
        ia,ib=IDX[a],IDX[b]
        return flat[ib:ib+2]-flat[ia:ia+2]
    
    def solve_min_correction(u0):
        # x is correction to flattened [P(2), B(2), E(2)]
        cons=[]
        for a,b,name in PAIRS:
            t=TARGETS[name]
            for j in range(2):
                if t[j]==0: continue
                ia,ib=IDX[a]+j,IDX[b]+j
                s=float(t[j])
                # require s * ((u_b + x_b) - (u_a + x_a)) >= MARGIN
                def fun(x, ia=ia, ib=ib, s=s):
                    return s*((u0[ib]+x[ib])-(u0[ia]+x[ia]))-MARGIN
                cons.append({"type":"ineq","fun":fun})
        res=minimize(lambda x:0.5*float(x@x),np.zeros_like(u0),jac=lambda x:x,
                     constraints=cons,method="SLSQP",
                     options={"ftol":1e-12,"maxiter":500,"disp":False})
        return res
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--r23a-audit",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        src=json.loads(args.r23a_audit.read_text())
        rows=[]
        for snap in src["snapshots"]:
            update=snap["update"]
            # suite-level solves
            for suite in range(4):
                pref={}
                for p in ("P","B","E"):
                    pref[p]=np.asarray(snap["preferences"][p][suite]["ppo_update_dir_out"],float)
                u0=np.concatenate([pref["P"],pref["B"],pref["E"]])
                res=solve_min_correction(u0)
                corr=res.x if res.success else np.full_like(u0,np.nan)
                uc=u0+corr if res.success else np.full_like(u0,np.nan)
                pair_info={}
                for a,b,name in PAIRS:
                    pv=pair_vec(uc,a,b);t=TARGETS[name]
                    margins=[]
                    ok=True
                    for j in range(2):
                        if t[j]==0: continue
                        m=float(t[j]*pv[j]);margins.append(m);ok &= m>=MARGIN-1e-9
                    pair_info[name]={
                      "corrected_vector":pv.tolist(),
                      "target_sign":[int(x) for x in t],
                      "min_signed_margin":min(margins) if margins else None,
                      "constraint_ok":bool(ok),
                    }
                ppo_norm=float(np.linalg.norm(u0));corr_norm=float(np.linalg.norm(corr)) if res.success else None
                shared_corr=np.linalg.norm(corr[[0,2,4]]) if res.success else None
                be_corr=np.linalg.norm(corr[[1,3,5]]) if res.success else None
                rows.append({
                  "update":update,"suite":suite,"feasible":bool(res.success),
                  "solver_status":res.message,
                  "ppo_pressure_flat":u0.tolist(),
                  "correction_flat":corr.tolist() if res.success else None,
                  "corrected_pressure_flat":uc.tolist() if res.success else None,
                  "ppo_norm":ppo_norm,"correction_norm":corr_norm,
                  "correction_ratio":None if not res.success else corr_norm/(ppo_norm+EPS),
                  "corrected_vs_ppo_cosine":None if not res.success else cosine(uc,u0),
                  "shared_correction_norm":shared_corr,"BE_correction_norm":be_corr,
                  "BE_fraction_of_correction":None if not res.success else be_corr/(corr_norm+EPS),
                  "pairs":pair_info
                })
            # aggregate-mean solve too
            pref={}
            for p in ("P","B","E"):
                pref[p]=np.mean([np.asarray(r["ppo_update_dir_out"],float) for r in snap["preferences"][p]],axis=0)
            u0=np.concatenate([pref["P"],pref["B"],pref["E"]])
            res=solve_min_correction(u0);corr=res.x if res.success else np.full_like(u0,np.nan);uc=u0+corr if res.success else np.full_like(u0,np.nan)
            pair_info={}
            for a,b,name in PAIRS:
                pv=pair_vec(uc,a,b);t=TARGETS[name];margins=[];ok=True
                for j in range(2):
                    if t[j]==0: continue
                    m=float(t[j]*pv[j]);margins.append(m);ok &= m>=MARGIN-1e-9
                pair_info[name]={"corrected_vector":pv.tolist(),"target_sign":[int(x) for x in t],
                                 "min_signed_margin":min(margins) if margins else None,"constraint_ok":bool(ok)}
            ppo_norm=float(np.linalg.norm(u0));corr_norm=float(np.linalg.norm(corr)) if res.success else None
            rows.append({
              "update":update,"suite":"aggregate","feasible":bool(res.success),"solver_status":res.message,
              "ppo_pressure_flat":u0.tolist(),"correction_flat":corr.tolist() if res.success else None,
              "corrected_pressure_flat":uc.tolist() if res.success else None,
              "ppo_norm":ppo_norm,"correction_norm":corr_norm,
              "correction_ratio":None if not res.success else corr_norm/(ppo_norm+EPS),
              "corrected_vs_ppo_cosine":None if not res.success else cosine(uc,u0),
              "shared_correction_norm":None if not res.success else float(np.linalg.norm(corr[[0,2,4]])),
              "BE_correction_norm":None if not res.success else float(np.linalg.norm(corr[[1,3,5]])),
              "BE_fraction_of_correction":None if not res.success else float(np.linalg.norm(corr[[1,3,5]])/(corr_norm+EPS)),
              "pairs":pair_info
            })
    
        summary={}
        for update in sorted({r["update"] for r in rows}):
            sr=[r for r in rows if r["update"]==update and r["suite"]!="aggregate"]
            ar=next(r for r in rows if r["update"]==update and r["suite"]=="aggregate")
            summary[str(update)]={
              "suite_feasibility_fraction":float(np.mean([r["feasible"] for r in sr])),
              "suite_all_constraints_fraction":float(np.mean([r["feasible"] and all(x["constraint_ok"] for x in r["pairs"].values()) for r in sr])),
              "correction_ratio_mean":float(np.mean([r["correction_ratio"] for r in sr])),
              "correction_ratio_max":float(np.max([r["correction_ratio"] for r in sr])),
              "corrected_vs_ppo_cosine_mean":float(np.mean([r["corrected_vs_ppo_cosine"] for r in sr])),
              "corrected_vs_ppo_cosine_min":float(np.min([r["corrected_vs_ppo_cosine"] for r in sr])),
              "BE_fraction_of_correction_mean":float(np.mean([r["BE_fraction_of_correction"] for r in sr])),
              "aggregate":ar
            }
        report={
          "schema":"post_v2_r23d_min_norm_constraint_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
          "source_r23a_audit":str(args.r23a_audit),"source_r23a_sha256":sha(args.r23a_audit),
          "margin":MARGIN,
          "optimization":"min 0.5||delta u||^2 subject to D1 PB/PE/BE coefficient half-space constraints",
          "rows":rows,"summary":summary,
          "note":"Read-only coefficient-output pressure correction. No optimizer/model updates."
        }
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"status":report["status"],"updates":list(summary)},indent=2))
    if True:main()

def run_post_v2_r2_freeze_basis():
    """Run former post_v2_r2_freeze_basis.py stage."""
    """Freeze provenance-backed R2 h1 shared/residual basis from D1 specialists."""
    import argparse, hashlib, json, sys
    from pathlib import Path
    import numpy as np, torch
    import torch.nn.functional as F
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    CODE_COMMIT="f77af91932b1651b531cb3f64cb3c1852ab800cb"
    EPS=1e-12
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def d1_h1(m,obs,w):
        return F.elu(m.actor_body[0](torch.cat((obs,w),dim=-1)))
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--steps",type=int,default=16);ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json";anchor=json.loads(anchor_path.read_text())
        anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        import gymnasium as gym, isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
        from talon_rl.models.behavior.latent import V2BehaviorActorCritic
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        obs,_=env.reset(seed=127001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
        p=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
        v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda();v2.load_state_dict(p["model"]);v2.eval()
        specs={};spec_paths={}
        for lab in PREFS:
            pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
            m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
        ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
        shared_samples=[];be_samples=[]
        with torch.no_grad():
            for suite in range(args.suites):
                cur,_=env.reset(seed=127001+suite);cur=obs_tensor(cur).cuda()
                for _ in range(args.steps):
                    h={k:d1_h1(specs[k],cur,ws[k]) for k in PREFS}
                    dPB=h["B"]-h["P"];dPE=h["E"]-h["P"];dBE=h["E"]-h["B"]
                    shared=(dPB+dPE)/2.0
                    shared_samples.append(shared.cpu())
                    # pointwise residual relative to pointwise shared direction
                    u=shared/(torch.linalg.vector_norm(shared,dim=-1,keepdim=True)+EPS)
                    be=dBE-(dBE*u).sum(-1,keepdim=True)*u
                    be_samples.append(be.cpu())
                    action=v2.act_inference_with_preference(cur,ws["P"])
                    nxt,_,_,_,_=env.step(torch.clamp(action,-1,1));cur=obs_tensor(nxt).cuda()
        S=torch.cat(shared_samples,0);R=torch.cat(be_samples,0)
        b_shared=S.mean(0);b_shared=b_shared/(b_shared.norm()+EPS)
        # Freeze residual by global mean, then re-orthogonalize to frozen shared basis.
        b_be=R.mean(0);b_be=b_be-(b_be@b_shared)*b_shared;b_be=b_be/(b_be.norm()+EPS)
        dot=float(b_shared@b_be)
        tensor_path=args.output.with_suffix(".pt")
        torch.save({"b_shared":b_shared,"b_BE":b_be},tensor_path)
        report={
          "schema":"v2_r2_frozen_basis_v1","status":"FROZEN","layer":"h1_post_elu_pre_modulation",
          "protocol":{"state_source":"V2-R1 P-policy closed-loop","reset_seed_base":127001,"suites":args.suites,"steps":args.steps,"num_envs":args.num_envs,
                      "shared":"normalize(mean_samples((dPB+dPE)/2))",
                      "BE":"normalize(orthogonalize(mean_samples(pointwise_orthogonalized_dBE), b_shared))"},
          "provenance":{"code_commit":CODE_COMMIT,"r1_checkpoint":str(args.checkpoint),"r1_checkpoint_sha256":sha(args.checkpoint),
                        "v2a_anchor_sha256":sha(anchor_path),
                        "specialists":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
          "basis":{"dimension":int(b_shared.numel()),"dot_shared_BE":dot,"shared_norm":float(b_shared.norm()),"BE_norm":float(b_be.norm()),
                   "tensor_file":str(tensor_path),"tensor_sha256":sha(tensor_path)}
        }
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report["basis"],indent=2))
        env.close();app.close()
    if True:main()

def run_post_v2_r2_semantic_geometry_audit():
    """Run former post_v2_r2_semantic_geometry_audit.py stage."""
    """V2-R2 read-only semantic hidden-geometry audit."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    import torch.nn.functional as F
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    PAIRS=(("P","B","P_to_B"),("P","E","P_to_E"),("B","E","B_to_E"))
    CODE_COMMIT="f77af91932b1651b531cb3f64cb3c1852ab800cb"
    EPS=1e-12
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def unit_cos(a,b):
        na=torch.linalg.vector_norm(a,dim=-1); nb=torch.linalg.vector_norm(b,dim=-1)
        mask=(na>1e-9)&(nb>1e-9)
        c=torch.zeros_like(na)
        if mask.any(): c[mask]=F.cosine_similarity(a[mask],b[mask],dim=-1)
        return c,mask
    
    def d1_hidden(m,obs,w):
        x=torch.cat((obs,w),dim=-1)
        h1=F.elu(m.actor_body[0](x))
        h2=F.elu(m.actor_body[2](h1))
        h3=F.elu(m.actor_body[4](h2))
        return h1,h3
    
    def v2_hidden(m,obs,w):
        z=m.behavior_z(w)
        h0=F.elu(m.actor_pre(obs))
        h1=(1+m.film_alpha*torch.tanh(m.film_gamma(z)))*h0 + m.film_alpha*torch.tanh(m.film_beta(z))
        h3=m.actor_rest(h1)
        return h1,h3
    
    def svd_stats(x,maxk=(1,2,4,8,16)):
        x=x - x.mean(0,keepdim=True)
        if x.shape[0] < 2: return {}
        _,s,vh=torch.linalg.svd(x,full_matrices=False)
        e=s.pow(2); total=float(e.sum()+EPS); cum=torch.cumsum(e,0)/(e.sum()+EPS)
        out={"singular_values":s.cpu().tolist(),"explained":{}}
        for k in maxk:
            kk=min(k,len(s)); out["explained"][str(k)]=float(cum[kk-1])
        out["basis"]=vh
        return out
    
    def projection_energy(x,basis,k):
        k=min(k,basis.shape[0]); b=basis[:k]
        proj=(x@b.T)@b
        return float((proj.pow(2).sum(-1)/(x.pow(2).sum(-1)+EPS)).mean())
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--steps",type=int,default=16)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint);mark("RUN_STARTED",protocol="V2-R2-SEMANTIC-GEOMETRY",measurement_only=True,checkpoint_sha256=cksha)
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text());anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=117001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda();v2.load_state_dict(payload["model"]);v2.eval()
            specs={};spec_paths={}
            for lab in PREFS:
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
    
            samples=[]
            # two baseline state distributions: D1-P closed loop and V2-P closed loop
            for source in ("D1_P","V2_P"):
                for suite in range(args.suites):
                    cur,_=env.reset(seed=117001+suite);cur=obs_tensor(cur).cuda()
                    for t in range(args.steps):
                        with torch.no_grad():
                            dh={};vh={}
                            for lab in PREFS:
                                dh[lab]=d1_hidden(specs[lab],cur,ws[lab])
                                vh[lab]=v2_hidden(v2,cur,ws[lab])
                            for p,q,name in PAIRS:
                                row={"source":source,"suite":suite,"t":t,"pair":name}
                                for layer,idx in (("h1",0),("h3",1)):
                                    dd=dh[q][idx]-dh[p][idx];vd=vh[q][idx]-vh[p][idx]
                                    c,mask=unit_cos(vd,dd)
                                    row[layer]={
                                      "d1_norm_mean":float(torch.linalg.vector_norm(dd,dim=-1).mean()),
                                      "v2_norm_mean":float(torch.linalg.vector_norm(vd,dim=-1).mean()),
                                      "cosine_mean":float(c[mask].mean()) if mask.any() else 0.0,
                                      "positive_fraction":float((c[mask]>0).float().mean()) if mask.any() else 0.0,
                                      "d1_vectors":dd.cpu().tolist(),"v2_vectors":vd.cpu().tolist(),
                                    }
                                samples.append(row)
                            action = specs["P"].act_inference_with_preference(cur,ws["P"]) if source=="D1_P" else v2.act_inference_with_preference(cur,ws["P"])
                            nxt,_,_,_,_=env.step(torch.clamp(action,-1,1));cur=obs_tensor(nxt).cuda()
    
            summary={}
            for source in ("D1_P","V2_P"):
                summary[source]={}
                for pair in ("P_to_B","P_to_E","B_to_E"):
                    rs=[r for r in samples if r["source"]==source and r["pair"]==pair]
                    summary[source][pair]={}
                    for layer in ("h1","h3"):
                        d1=torch.tensor(np.concatenate([r[layer]["d1_vectors"] for r in rs],axis=0),dtype=torch.float32)
                        vv=torch.tensor(np.concatenate([r[layer]["v2_vectors"] for r in rs],axis=0),dtype=torch.float32)
                        st=svd_stats(d1);basis=st.pop("basis")
                        # cross-suite mean-direction consistency
                        means=[]
                        for suite in range(args.suites):
                            sr=[r for r in rs if r["suite"]==suite]
                            x=torch.tensor(np.concatenate([r[layer]["d1_vectors"] for r in sr],axis=0),dtype=torch.float32)
                            means.append(x.mean(0))
                        paircos=[]
                        for i in range(len(means)):
                            for j in range(i+1,len(means)):
                                c,_=unit_cos(means[i].unsqueeze(0),means[j].unsqueeze(0));paircos.append(float(c[0]))
                        summary[source][pair][layer]={
                          "pointwise_cosine_mean":float(np.mean([r[layer]["cosine_mean"] for r in rs])),
                          "pointwise_positive_fraction":float(np.mean([r[layer]["positive_fraction"] for r in rs])),
                          "d1_norm_mean":float(torch.linalg.vector_norm(d1,dim=-1).mean()),
                          "v2_norm_mean":float(torch.linalg.vector_norm(vv,dim=-1).mean()),
                          "v2_over_d1_norm_mean":float((torch.linalg.vector_norm(vv,dim=-1)/(torch.linalg.vector_norm(d1,dim=-1)+EPS)).mean()),
                          "d1_svd_explained":st["explained"],
                          "v2_projection_energy_in_d1_basis":{
                            str(k):projection_energy(vv,basis,k) for k in (1,2,4,8,16)
                          },
                          "d1_cross_suite_mean_direction_cosine":float(np.mean(paircos)),
                        }
            # axis-angle relation inside D1 and V2 at each layer/source
            axis_relation={}
            for source in ("D1_P","V2_P"):
                axis_relation[source]={}
                for layer in ("h1","h3"):
                    pbr=[r for r in samples if r["source"]==source and r["pair"]=="P_to_B"]
                    per=[r for r in samples if r["source"]==source and r["pair"]=="P_to_E"]
                    d1pb=torch.tensor(np.concatenate([r[layer]["d1_vectors"] for r in pbr],0),dtype=torch.float32).mean(0)
                    d1pe=torch.tensor(np.concatenate([r[layer]["d1_vectors"] for r in per],0),dtype=torch.float32).mean(0)
                    vpb=torch.tensor(np.concatenate([r[layer]["v2_vectors"] for r in pbr],0),dtype=torch.float32).mean(0)
                    vpe=torch.tensor(np.concatenate([r[layer]["v2_vectors"] for r in per],0),dtype=torch.float32).mean(0)
                    cd,_=unit_cos(d1pb[None],d1pe[None]);cv,_=unit_cos(vpb[None],vpe[None])
                    axis_relation[source][layer]={"D1_PB_vs_PE_cosine":float(cd[0]),"V2_PB_vs_PE_cosine":float(cv[0])}
    
            report={
              "schema":"post_v2_r2_semantic_geometry_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                            "v2a_anchor_sha256":sha(anchor_path),
                            "specialist_checkpoints":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
              "steps":args.steps,"suites":args.suites,"state_sources":["D1_P","V2_P"],
              "summary":summary,"axis_relation":axis_relation,
              "note":"Hidden geometry only. D1 h1 is first ELU hidden, V2 h1 is post-FiLM hidden; h3 is final actor hidden. States come from matched D1-P and V2-P closed-loop trajectories. No parameters updated."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"sources":list(summary)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    
    if True:main()

def run_post_v2_r2_short_coefficient_screen():
    """Run former post_v2_r2_short_coefficient_screen.py stage."""
    """V2-R2 one-seed short coefficient/authority screen. No semantic verdict."""
    import argparse,hashlib,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":np.array([.8,.1,.1],np.float32),"B":np.array([.1,.8,.1],np.float32),"E":np.array([.1,.1,.8],np.float32)}
    CHECKPOINT_UPDATES=(0,1,5,10,25,50,100)
    CODE_COMMIT="3d0b4dd2db9e0a140bbe3bc37f958a3f52be0dbe"
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt_deg(q):return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    def early_ppo(ratio,adv,w,eps=.2):
        scalar=(adv*w).sum(-1);clipped=ratio.clamp(1-eps,1+eps)
        return -torch.minimum(ratio*scalar,clipped*scalar).mean()
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--source-checkpoint",type=Path,required=True)
        ap.add_argument("--basis",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--updates",type=int,default=100)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--horizon",type=int,default=2)
        ap.add_argument("--eval-steps",type=int,default=16)
        ap.add_argument("--seed",type=int,default=0)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        basis_sha=sha(args.basis)
        mark("RUN_STARTED",protocol="V2-R2-SHORT-COEFFICIENT-SCREEN",training=True,updates=args.updates,basis_sha256=basis_sha)
        app=env=None
        try:
            anchor_report=json.loads((ROOT/"runs/post_v2_a-2026-09-23/v2a.json").read_text())
            anchors=torch.tensor([anchor_report["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            r21=json.loads((ROOT/"runs/post_v2_r21_shared_residual-2026-09-23/audit.json").read_text())
            d1_ref={
              "h1_shared_PB":r21["summary"]["h1"]["D1"]["PB"]["c_shared_mean"],
              "h1_shared_PE":r21["summary"]["h1"]["D1"]["PE"]["c_shared_mean"],
              "h1_BE_PB":r21["summary"]["h1"]["D1"]["PB"]["c_BE_mean"],
              "h1_BE_PE":r21["summary"]["h1"]["D1"]["PE"]["c_BE_mean"],
              "h1_BE_BE":r21["summary"]["h1"]["D1"]["BE"]["c_BE_mean"],
            }
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]]
            app=AppLauncher({"headless":True,"enable_cameras":False}).app
            sys.argv=saved;mark("APP_INIT_OK")
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import vector_gae,vector_value_loss
            from talon_rl.models.behavior.projected import V2R2ProjectedActorCritic,initialize_from_v1c
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);mark("ENV_CREATED")
            obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            source_payload=torch.load(args.source_checkpoint,map_location="cpu",weights_only=False);source=source_payload["model"]
            bp=torch.load(args.basis,map_location="cuda",weights_only=False)
            basis=torch.stack([bp["b_shared"],bp["b_BE"]],0).cuda()
            model=V2R2ProjectedActorCritic(obs.shape[-1],ad,basis,anchors=anchors,projection_alpha=1.0).cuda()
            initialize_from_v1c(model,source)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+300000)
            mark("CHECKPOINT_LOADED",source=str(args.source_checkpoint))
            current,_=env.reset(seed=args.seed);current=obs_tensor(current).cuda()
            train_records=[];snapshots=[]
    
            def evaluate_snapshot(update:int):
                model.eval()
                with torch.no_grad():
                    w={k:torch.as_tensor(np.repeat(v[None,:],args.num_envs,axis=0),device="cuda") for k,v in PREFS.items()}
                    coeff={k:model.semantic_coefficients(x)[0].cpu().tolist() for k,x in w.items()}
                    delta={k:model.projected_delta(x)[0].cpu().tolist() for k,x in w.items()}
                    cd={
                      "PB":[coeff["B"][0]-coeff["P"][0],coeff["B"][1]-coeff["P"][1]],
                      "PE":[coeff["E"][0]-coeff["P"][0],coeff["E"][1]-coeff["P"][1]],
                      "BE":[coeff["E"][0]-coeff["B"][0],coeff["E"][1]-coeff["B"][1]],
                    }
                    pd={
                      "PB":float(torch.linalg.vector_norm(model.projected_delta(w["B"])-model.projected_delta(w["P"]),dim=-1).mean()),
                      "PE":float(torch.linalg.vector_norm(model.projected_delta(w["E"])-model.projected_delta(w["P"]),dim=-1).mean()),
                      "BE":float(torch.linalg.vector_norm(model.projected_delta(w["E"])-model.projected_delta(w["B"]),dim=-1).mean()),
                    }
                    # fixed-state action distance on one matched reset state
                    cur,_=env.reset(seed=93000);cur=obs_tensor(cur).cuda()
                    acts={k:torch.clamp(model.act_inference_with_preference(cur,w[k]),-1,1) for k in PREFS}
                    adist={
                      "PB":float(torch.linalg.vector_norm(acts["B"]-acts["P"],dim=-1).mean()),
                      "PE":float(torch.linalg.vector_norm(acts["E"]-acts["P"],dim=-1).mean()),
                      "BE":float(torch.linalg.vector_norm(acts["E"]-acts["B"],dim=-1).mean()),
                    }
                    behavior=[]
                    for idx,(label,pref) in enumerate(PREFS.items()):
                        ww=w[label];cur,_=env.reset(seed=94000+idx);cur=obs_tensor(cur).cuda()
                        done_any=np.zeros(args.num_envs,bool);tilts=[];finite=True
                        for _ in range(args.eval_steps):
                            action=torch.clamp(model.act_inference_with_preference(cur,ww),-1,1)
                            finite=finite and bool(torch.isfinite(action).all())
                            nxt,_,term,trunc,_=env.step(action);done_any|=(term|trunc).cpu().numpy()
                            tilts.append(float(torch.quantile(tilt_deg(env.unwrapped.scene["robot"].data.root_quat_w),.95)))
                            cur=obs_tensor(nxt).cuda()
                        behavior.append({"preference":label,"survival":float(1-done_any.mean()),"tilt_p95_mean":float(np.mean(tilts)),"finite":finite})
                    snap={
                      "update":update,"coefficients":coeff,"coefficient_deltas":cd,
                      "projected_delta_norm":pd,"fixed_state_action_distance":adist,
                      "behavior":behavior,
                      "basis_sha256_runtime":sha(args.basis),
                      "basis_unchanged":sha(args.basis)==basis_sha,
                      "d1_reference":d1_ref,
                    }
                model.train();return snap
    
            snapshots.append(evaluate_snapshot(0))
            for update in range(1,args.updates+1):
                w=torch.as_tensor(rng.dirichlet(np.ones(3),size=args.num_envs),device="cuda",dtype=torch.float32)
                obs_buf=[];act_buf=[];old_buf=[];rew_buf=[];val_buf=[];done_buf=[]
                for _ in range(args.horizon):
                    with torch.no_grad():action,old=model.act_with_preference(current,w);value=model.value_with_preference(current,w)
                    nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1))
                    manager=env.unwrapped.reward_manager;raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                    vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                    obs_buf.append(current);act_buf.append(action);old_buf.append(old);rew_buf.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);val_buf.append(value);done_buf.append((term|trunc).to("cuda"));current=obs_tensor(nxt).cuda()
                with torch.no_grad():next_value=model.value_with_preference(current,w)
                reward_t=torch.stack(rew_buf);value_t=torch.stack(val_buf);done_t=torch.stack(done_buf).bool()
                adv,ret=vector_gae(reward_t,value_t,next_value,done_t)
                flat_obs=torch.cat(obs_buf);flat_act=torch.cat(act_buf);flat_old=torch.cat(old_buf);flat_w=w.repeat(args.horizon,1)
                ratio=torch.exp(model.logp_with_preference(flat_obs,flat_w,flat_act)-flat_old.detach())
                ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),flat_w)
                manifold=model.manifold_loss(flat_w)
                critic=vector_value_loss(model.value_with_preference(flat_obs,flat_w),ret.reshape(-1,3).detach())
                loss=ppo+critic+0.1*manifold
                optimizer.zero_grad(set_to_none=True);loss.backward()
                finite=bool(torch.isfinite(loss).item() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                optimizer.step()
                train_records.append({"update":update,"loss":float(loss.detach()),"ppo_loss":float(ppo.detach()),"critic_loss":float(critic.detach()),"manifold_loss":float(manifold.detach()),"finite":finite})
                if update in CHECKPOINT_UPDATES:
                    snapshots.append(evaluate_snapshot(update))
                    mark("SNAPSHOT",update=update)
    
            terminal=args.output.parent/"v2r2_terminal.pt"
            torch.save({"schema":"post_v2_r2_terminal_v1","seed":args.seed,"update":args.updates,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"anchors":anchors.tolist(),"basis_sha256":basis_sha,"projection_alpha":1.0},terminal)
            mark("CHECKPOINT_WRITTEN",checkpoint=str(terminal))
            final=snapshots[-1]
            sign_gate={
              "PB_shared_positive":final["coefficient_deltas"]["PB"][0]>0,
              "PB_BE_negative":final["coefficient_deltas"]["PB"][1]<0,
              "PE_shared_positive":final["coefficient_deltas"]["PE"][0]>0,
              "PE_BE_positive":final["coefficient_deltas"]["PE"][1]>0,
              "BE_BE_positive":final["coefficient_deltas"]["BE"][1]>0,
            }
            report={
              "schema":"post_v2_r2_short_coefficient_screen_v1","status":"SCREEN_COMPLETE","training":True,"diagnostic_only":True,
              "semantic_verdict":False,"seed":args.seed,"updates":args.updates,
              "source_checkpoint":str(args.source_checkpoint),"terminal_checkpoint":str(terminal),
              "basis":str(args.basis),"basis_sha256":basis_sha,"code_commit":CODE_COMMIT,
              "lambda_manifold":0.1,"projection_alpha":1.0,
              "checkpoints":list(CHECKPOINT_UPDATES),"train_records":train_records,"snapshots":snapshots,
              "final_sign_gate":sign_gate,"final_sign_gate_all":all(sign_gate.values()),
              "note":"R2 short coefficient/authority screen only. No semantic or multi-seed authorization."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"final_sign_gate":sign_gate,"terminal":str(terminal)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_t0_c3_rollout32():
    """Run former post_v2_t0_c3_rollout32.py stage."""
    """V2-C3 read-only short-horizon closed-loop semantic divergence audit."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
    HORIZONS=(8,16,32)
    CODE_COMMIT="f77af91932b1651b531cb3f64cb3c1852ab800cb"
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def tilt_deg(q):
        return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def vecnorm(x): return torch.linalg.vector_norm(x,dim=-1)
    
    def first_divergence(values, threshold):
        for i,v in enumerate(values):
            if v > threshold: return i
        return None
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--checkpoint",type=Path,required=True)
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--suites",type=int,default=4)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        cksha=sha(args.checkpoint)
        mark("RUN_STARTED",protocol="V2-T0-C3-ROLLOUT32",measurement_only=True,checkpoint_sha256=cksha,horizons=list(HORIZONS))
        app=env=None
        try:
            anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
            anchor=json.loads(anchor_path.read_text());anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.behavior.latent import V2BehaviorActorCritic
    
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=87001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
    
            payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
            v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda();v2.load_state_dict(payload["model"]);v2.eval()
            specs={}
            spec_paths={}
            for lab in PREFS:
                pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
            ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
    
            families={
              "D1_P":("spec","P"),"D1_B":("spec","B"),"D1_E":("spec","E"),
              "V2_P":("v2","P"),"V2_B":("v2","B"),"V2_E":("v2","E"),
            }
            maxH=max(HORIZONS); trajectories=[]
            for suite in range(args.suites):
                seed=87001+suite
                for fam,(kind,label) in families.items():
                    cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                    prev=torch.zeros((args.num_envs,ad),device="cuda");cum_obj=np.zeros(3,dtype=np.float64);steps=[]
                    with torch.no_grad():
                        for t in range(maxH):
                            model=specs[label] if kind=="spec" else v2
                            raw_action=model.act_inference_with_preference(cur,ws[label])
                            sat_frac=float((raw_action.abs()>1.0).float().mean())
                            action=torch.clamp(raw_action,-1,1)
                            nxt,_,term,trunc,_=env.step(action)
                            raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                            vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                            obj_mean=vec.mean(0);cum_obj+=obj_mean
                            data=robot.data
                            # contact proxy: binary net contact force norm at feet if available
                            contact=None
                            try:
                                sensor=env.unwrapped.scene.sensors.get("contact_forces",None)
                                if sensor is not None:
                                    forces=sensor.data.net_forces_w
                                    contact=(torch.linalg.vector_norm(forces,dim=-1)>1.0).float()
                            except Exception:
                                contact=None
                            step={
                              "t":t,
                              "action_mean":action.mean(0).cpu().tolist(),
                              "action_l2_mean":float(vecnorm(action).mean()),
                              "action_rate_mean":float(vecnorm(action-prev).mean()),
                              "saturation_fraction":sat_frac,
                              "joint_pos_mean":data.joint_pos.mean(0).cpu().tolist(),
                              "joint_vel_mean":data.joint_vel.mean(0).cpu().tolist(),
                              "base_lin_vel_mean":data.root_lin_vel_b.mean(0).cpu().tolist(),
                              "base_ang_vel_mean":data.root_ang_vel_b.mean(0).cpu().tolist(),
                              "tilt_mean":float(tilt_deg(data.root_quat_w).mean()),
                              "torque_mean":data.applied_torque.mean(0).cpu().tolist(),
                              "torque_norm_mean":float(vecnorm(data.applied_torque).mean()),
                              "objective_step_mean":obj_mean.tolist(),
                              "objective_cumulative":cum_obj.tolist(),
                              "termination_fraction":float((term|trunc).float().mean()),
                            }
                            if contact is not None:
                                step["contact_pattern_mean"]=contact.mean(0).cpu().tolist()
                                step["contact_fraction_mean"]=float(contact.mean())
                            steps.append(step);prev=action;cur=obs_tensor(nxt).cuda()
                    trajectories.append({"suite":suite,"seed":seed,"family":fam,"kind":kind,"preference":label,"steps":steps})
    
            # pairwise divergence over time
            pairs=[
              ("D1_P","D1_B","D1_P_vs_B"),("D1_P","D1_E","D1_P_vs_E"),
              ("V2_P","V2_B","V2_P_vs_B"),("V2_P","V2_E","V2_P_vs_E"),
            ]
            div_rows=[]
            def arr(step,key):
                return np.asarray(step[key],dtype=float)
            for suite in range(args.suites):
                for a,b,name in pairs:
                    ta=next(x for x in trajectories if x["suite"]==suite and x["family"]==a)["steps"]
                    tb=next(x for x in trajectories if x["suite"]==suite and x["family"]==b)["steps"]
                    for H in HORIZONS:
                        ts=[]
                        for t in range(H):
                            xa,xb=ta[t],tb[t]
                            row={
                              "t":t,
                              "action":float(np.linalg.norm(arr(xa,"action_mean")-arr(xb,"action_mean"))),
                              "joint_pos":float(np.linalg.norm(arr(xa,"joint_pos_mean")-arr(xb,"joint_pos_mean"))),
                              "joint_vel":float(np.linalg.norm(arr(xa,"joint_vel_mean")-arr(xb,"joint_vel_mean"))),
                              "base_lin_vel":float(np.linalg.norm(arr(xa,"base_lin_vel_mean")-arr(xb,"base_lin_vel_mean"))),
                              "base_ang_vel":float(np.linalg.norm(arr(xa,"base_ang_vel_mean")-arr(xb,"base_ang_vel_mean"))),
                              "tilt":abs(float(xa["tilt_mean"]-xb["tilt_mean"])),
                              "torque":float(np.linalg.norm(arr(xa,"torque_mean")-arr(xb,"torque_mean"))),
                              "objective_cumulative":float(np.linalg.norm(arr(xa,"objective_cumulative")-arr(xb,"objective_cumulative"))),
                              "saturation_gap":abs(float(xa["saturation_fraction"]-xb["saturation_fraction"])),
                            }
                            if "contact_pattern_mean" in xa and "contact_pattern_mean" in xb:
                                row["contact"]=float(np.linalg.norm(arr(xa,"contact_pattern_mean")-arr(xb,"contact_pattern_mean")))
                            ts.append(row)
                        div_rows.append({"suite":suite,"pair":name,"horizon":H,"timeseries":ts})
    
            # thresholds derived from fixed numerical floors + relative scale for interpretability
            thresholds={"action":1e-3,"joint_pos":1e-3,"joint_vel":1e-3,"base_lin_vel":1e-3,"base_ang_vel":1e-3,"tilt":0.05,"torque":1e-2,"objective_cumulative":1e-3,"contact":1e-3,"saturation_gap":1e-3}
            summaries={}
            for _,_,name in pairs:
                summaries[name]={}
                for H in HORIZONS:
                    rs=[r for r in div_rows if r["pair"]==name and r["horizon"]==H]
                    metrics=list(rs[0]["timeseries"][0].keys());metrics.remove("t")
                    summaries[name][str(H)]={}
                    for m in metrics:
                        vals=np.asarray([[x["timeseries"][t][m] for t in range(H)] for x in rs],dtype=float)
                        mean_t=vals.mean(0)
                        firsts=[first_divergence([x["timeseries"][t][m] for t in range(H)],thresholds.get(m,1e-3)) for x in rs]
                        finite_first=[x for x in firsts if x is not None]
                        summaries[name][str(H)][m]={
                          "mean_timeseries":mean_t.tolist(),
                          "terminal_mean":float(mean_t[-1]),
                          "first_divergence_by_suite":firsts,
                          "first_divergence_mean":float(np.mean(finite_first)) if finite_first else None,
                          "diverged_fraction":float(len(finite_first)/len(firsts)),
                          "threshold":thresholds.get(m,1e-3),
                        }
            report={
              "schema":"post_v2_t0_c3_rollout32_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                            "v2a_anchor_sha256":sha(anchor_path),
                            "specialist_checkpoints":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
              "horizons":list(HORIZONS),"thresholds":thresholds,"families":families,
              "trajectories":trajectories,"divergence":div_rows,"summaries":summaries,
              "note":"Matched initial states, deterministic policies, closed-loop rollout. No parameters updated."
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
            print(json.dumps({"status":report["status"],"trajectory_rows":len(trajectories),"divergence_rows":len(div_rows)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_t0_offline_crossrank():
    """Run former post_v2_t0_offline_crossrank.py stage."""
    import json, numpy as np
    from pathlib import Path
    p=Path("runs/post_v2_t0_trajectory_credit-2026-09-23/audit.json")
    d=json.loads(p.read_text())
    weights={"P":np.array([.8,.1,.1]),"B":np.array([.1,.8,.1]),"E":np.array([.1,.1,.8])}
    cross={}
    for kind in ("D1","V2"):
        cross[kind]={}
        for H in ("8","16","32"):
            rows=d["summary"][kind][H]["suite_rows"]
            per=[]
            for r in rows:
                mat=np.asarray(r["objective_matrix"],float)
                wins={}; vals={}; diag={}
                for i,lab in enumerate(("P","B","E")):
                    sc=mat@weights[lab]
                    vals[lab]=sc.tolist()
                    win=int(np.argmax(sc))
                    wins[lab]=win
                    diag[lab]=bool(win==i)
                per.append({"suite":r["suite"],"scalarized_values_by_target_pref":vals,"winners_by_target_pref":wins,"diagonal":diag})
            cross[kind][H]={
                "diagonal_fraction":{lab:float(np.mean([x["diagonal"][lab] for x in per])) for lab in ("P","B","E")},
                "all_three_fraction":float(np.mean([all(x["diagonal"].values()) for x in per])),
                "suite_rows":per
            }
    d["cross_preference_scalarized_ranking"]=cross
    p.write_text(json.dumps(d,indent=2)+"\n")
    print(json.dumps(cross,indent=2))

def run_post_v2_t1_specialist_reward_semantics_audit():
    """Run former post_v2_t1_specialist_reward_semantics_audit.py stage."""
    import json, hashlib
    from pathlib import Path
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    D1=ROOT/"runs/post_v1_d1-2026-09-22/d1.json"
    AGG=ROOT/"runs/post_v1_d1-2026-09-22/aggregate.json"
    T0=ROOT/"runs/post_v2_t0_trajectory_credit-2026-09-23/audit.json"
    OUT=ROOT/"runs/post_v2_t1_specialist_semantics-2026-09-23"
    OUT.mkdir(parents=True,exist_ok=True)
    PREFS={"P":np.array([.8,.1,.1]),"B":np.array([.1,.8,.1]),"E":np.array([.1,.1,.8])}
    LABS=["P","B","E"]
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    d1=json.loads(D1.read_text())
    agg=json.loads(AGG.read_text())
    t0=json.loads(T0.read_text())
    
    # terminal objective reward-rate vectors from original D1 aggregate
    terminal=np.stack([np.array(agg["summary"][lab]["objective_return_mean"],float) for lab in LABS],0)
    cross=np.zeros((3,3))
    contrib={}
    for i,plab in enumerate(LABS):
        contrib[plab]={}
        for j,wlab in enumerate(LABS):
            w=PREFS[wlab]
            weighted=terminal[i]*w
            cross[i,j]=weighted.sum()
            denom=np.sum(np.abs(weighted))+1e-12
            contrib[plab][wlab]={
                "weighted_terms":weighted.tolist(),
                "scalarized_reward_rate":float(weighted.sum()),
                "absolute_contribution_share":(np.abs(weighted)/denom).tolist(),
            }
    
    # training-time grouped reward vectors (already multiplied by step_dt in training)
    training={}
    for lab in LABS:
        rec=d1["records"][lab]
        arr=np.array([r["reward_mean"] for r in rec],float)
        w=PREFS[lab]
        windows={}
        for name,sl in {
            "early_1_25":slice(0,25),"mid_126_175":slice(125,175),"late_276_300":slice(275,300)
        }.items():
            x=arr[sl].mean(0); wc=x*w
            windows[name]={
              "objective_stepdt_mean":x.tolist(),
              "weighted_contributions":wc.tolist(),
              "scalarized_stepdt_mean":float(wc.sum()),
              "absolute_contribution_share":(np.abs(wc)/(np.abs(wc).sum()+1e-12)).tolist()
            }
        training[lab]=windows
    
    # Physical semantic rankings from D1 aggregate
    metrics={lab:agg["summary"][lab]["metrics_mean"] for lab in LABS}
    physical={
     "progress_vx_error_best":min(LABS,key=lambda x:metrics[x]["vx_error"]),
     "balance_tilt_best":min(LABS,key=lambda x:metrics[x]["tilt_deg"]),
     "balance_ang_vel_best":min(LABS,key=lambda x:metrics[x]["ang_vel_xy"]),
     "efficiency_torque_best":min(LABS,key=lambda x:metrics[x]["torque_norm"]),
     "efficiency_action_rate_best":min(LABS,key=lambda x:metrics[x]["action_rate"]),
    }
    
    # Cross-eval winners under each evaluation preference.
    cross_winners={LABS[j]:LABS[int(np.argmax(cross[:,j]))] for j in range(3)}
    cross_margin={}
    for j,wlab in enumerate(LABS):
        order=np.argsort(-cross[:,j]); cross_margin[wlab]={
          "winner":LABS[int(order[0])],"runner_up":LABS[int(order[1])],
          "margin":float(cross[order[0],j]-cross[order[1],j])
        }
    
    # Reward-scale diagnostics across policies at terminal
    obj_ranges=terminal.max(0)-terminal.min(0)
    obj_abs_mean=np.abs(terminal).mean(0)
    scale={
     "terminal_objective_abs_mean":obj_abs_mean.tolist(),
     "terminal_objective_policy_range":obj_ranges.tolist(),
     "range_over_abs_mean":(obj_ranges/(obj_abs_mean+1e-12)).tolist()
    }
    
    report={
     "schema":"v2_t1_specialist_reward_semantics_audit_v1",
     "status":"MEASUREMENT_COMPLETE","measurement_only":True,
     "provenance":{"d1_sha256":sha(D1),"aggregate_sha256":sha(AGG),"t0_sha256":sha(T0)},
     "objective_order":["progress","balance","efficiency"],
     "fixed_preferences":{k:v.tolist() for k,v in PREFS.items()},
     "metric_semantics_note":"D1 aggregate objective_return_mean is a terminal mean weighted reward-rate per env-step, not an episodic return; training reward_mean is the same grouped reward multiplied by step_dt before GAE.",
     "terminal_objective_reward_rate_matrix":{"rows_policy":LABS,"columns_objective":["P","B","E"],"values":terminal.tolist()},
     "cross_evaluation_scalarized_matrix":{"rows_policy":LABS,"columns_eval_preference":LABS,"values":cross.tolist(),"winners":cross_winners,"margins":cross_margin},
     "terminal_scalarized_contributions":contrib,
     "training_time":training,
     "physical_metrics":metrics,
     "physical_best":physical,
     "reward_scale":scale,
     "t0_summary":t0["summary"]["D1"],
    }
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
     "cross_matrix":cross.tolist(),
     "winners":cross_winners,
     "physical_best":physical,
     "reward_scale":scale,
     "late_training":{k:v["late_276_300"] for k,v in training.items()}
    },indent=2))

def run_post_v2_t3a2_atomic_coherence_audit():
    """Run former post_v2_t3a2_atomic_coherence_audit.py stage."""
    """T3-A2 read-only atomic semantic-coordinate/coherence audit."""
    import json, hashlib
    from pathlib import Path
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    T3=ROOT/"runs/post_v2_t3a_regrouping-2026-09-23/audit.json"
    S6=[ROOT/f"runs/v1b_s6_objective_decomposition_seed{s}-2026-09-22/audit.json" for s in (0,1,2)]
    OUT=ROOT/"runs/post_v2_t3a2_atomic_coherence-2026-09-23"; OUT.mkdir(parents=True,exist_ok=True)
    LABS=("P","B","E")
    ATOMIC={
     "velocity_tracking":["track_lin_vel_xy_exp","track_ang_vel_z_exp"],
     "vertical_stability":["lin_vel_z_l2"],
     "angular_stability":["ang_vel_xy_l2"],
     "orientation_stability":["flat_orientation_l2"],
     "effort":["dof_torques_l2"],
     "joint_smoothness":["dof_acc_l2"],
     "control_smoothness":["action_rate_l2"],
     "gait_contact":["feet_air_time"],
    }
    PHYS_TARGET={
     "velocity_tracking":["vx_error","wz_error"],
     "vertical_stability":["abs_lin_vel_z"],
     "angular_stability":["abs_ang_vel_xy"],
     "orientation_stability":["tilt_deg"],
     "effort":["torque_l2"],
     "joint_smoothness":[],
     "control_smoothness":["action_rate_l2"],
     "gait_contact":[],
    }
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def corr(x,y):
        x=np.asarray(x,float);y=np.asarray(y,float)
        if x.std()==0 or y.std()==0:return 0.0
        return float(np.corrcoef(x,y)[0,1])
    
    t3=json.loads(T3.read_text())
    s6=[json.loads(p.read_text()) for p in S6]
    
    # matched-policy atomic statistics from T3-A term means/std/nonzero
    stats={}
    for name,terms in ATOMIC.items():
        policy_means={lab:[] for lab in LABS}
        stds=[]; nonzeros=[]
        for r in t3["suite_policy_rows"]:
            vals=[r["term_mean"][t] for t in terms]
            policy_means[r["policy"]].append(float(sum(vals)))
            if len(terms)==1:
                stds.append(r["term_std"][terms[0]]); nonzeros.append(r["term_nonzero"][terms[0]])
            else:
                # Exact std/nonzero for velocity_tracking is available from T3-A group stats.
                stds.append(r["group_std"]["progress"]); nonzeros.append(r["group_nonzero"]["progress"])
        avg={lab:float(np.mean(v)) for lab,v in policy_means.items()}
        scale=float(np.mean(np.abs([x for v in policy_means.values() for x in v])))
        prange=float(max(avg.values())-min(avg.values()))
        winners=[]
        for suite in range(4):
            vals={lab:float(sum(next(r for r in t3["suite_policy_rows"] if r["suite"]==suite and r["policy"]==lab)["term_mean"][t] for t in terms)) for lab in LABS}
            winners.append(max(vals,key=vals.get))
        stats[name]={
          "terms":terms,"policy_mean":avg,"range_over_scale":prange/(scale+1e-12),
          "abs_mean_scale":scale,"mean_std":float(np.mean(stds)),"mean_nonzero_fraction":float(np.mean(nonzeros)),
          "winner_fraction":{lab:float(np.mean([w==lab for w in winners])) for lab in LABS}
        }
    
    # multi-seed atomic physical alignment and evidence graph.
    # velocity tracking uses the grouped progress alignment already computed by T3-A matched audit;
    # singleton atomics use S6 per-term correlations across seeds.
    phys={}
    for name,terms in ATOMIC.items():
        phys[name]={}
        if name=="velocity_tracking":
            phys[name]={"vx_error":t3["mean_group_physical_corr"]["progress"]["vx_error"],
                        "wz_error":t3["mean_group_physical_corr"]["progress"]["wz_error"]}
        elif len(terms)==1:
            t=terms[0]
            for metric in s6[0]["physical_metric_order"]:
                vals=[]
                for d in s6:
                    ti=d["reward_term_order"].index(t); pi=d["physical_metric_order"].index(metric)
                    vals.append(d["term_physical_pearson"][ti][pi])
                phys[name][metric]={"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals))}
    # semantic target alignment summary
    alignment={}
    for name,targets in PHYS_TARGET.items():
        if name=="velocity_tracking":
            vals=[abs(phys[name][m]) for m in targets]
        else:
            vals=[abs(phys[name][m]["mean"]) for m in targets] if targets else []
        alignment[name]={"target_metrics":targets,"mean_abs_target_alignment":float(np.mean(vals)) if vals else None}
    
    # Pairwise atomic correlations from S6 multi-seed.
    # velocity_tracking correlations cannot be reconstructed exactly from term-only correlation matrices,
    # so use T3-A progress group correlation where the counterpart is an existing T3-A group only;
    # all singleton-singleton pairs are exact S6 term correlations.
    names=list(ATOMIC)
    graph=[]
    for i,a in enumerate(names):
      for b in names[i+1:]:
        vals=[]; source=None
        if len(ATOMIC[a])==1 and len(ATOMIC[b])==1:
          ta,tb=ATOMIC[a][0],ATOMIC[b][0]
          for d in s6:
            ia=d["reward_term_order"].index(ta); ib=d["reward_term_order"].index(tb)
            vals.append(d["term_term_pearson"][ia][ib])
          source="S6 multi-seed term correlation"
        elif a=="velocity_tracking" or b=="velocity_tracking":
          other=b if a=="velocity_tracking" else a
          groupmap={"vertical_stability":"stability","angular_stability":"stability","orientation_stability":"stability",
                    "effort":"effort","joint_smoothness":"smoothness","control_smoothness":"smoothness","gait_contact":"gait_contact_aux"}
          # only coarse parent-group relation is available for velocity tracking.
          gi=t3["core_group_order"].index("progress"); gj=t3["core_group_order"].index(groupmap[other])
          vals=[t3["mean_group_pairwise_corr"][gi][gj]]
          source="T3-A coarse progress-to-parent-group correlation"
        graph.append({"a":a,"b":b,"corr_mean":float(np.mean(vals)),"abs_corr_mean":float(np.mean(np.abs(vals))),
                      "corr_min":float(np.min(vals)),"corr_max":float(np.max(vals)),"source":source})
    
    # recombination candidates require both semantic affinity and correlation.
    # Use a conservative evidence label rather than hard threshold as a design decision.
    for e in graph:
        a,b=e["a"],e["b"]
        same_domain=((a in {"vertical_stability","angular_stability","orientation_stability"} and b in {"vertical_stability","angular_stability","orientation_stability"})
                     or (a in {"joint_smoothness","control_smoothness"} and b in {"joint_smoothness","control_smoothness"}))
        if same_domain and e["abs_corr_mean"]>=0.5:
            e["recombine_evidence"]="STRONG"
        elif same_domain and e["abs_corr_mean"]>=0.3:
            e["recombine_evidence"]="MODERATE"
        elif same_domain:
            e["recombine_evidence"]="WEAK"
        else:
            e["recombine_evidence"]="NOT_PROPOSED"
    
    report={
     "schema":"v2_t3a2_atomic_coherence_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
     "atomic_coordinates":ATOMIC,
     "provenance":{"t3a":str(T3.relative_to(ROOT)),"t3a_sha256":sha(T3),
                   "s6":[{"path":str(p.relative_to(ROOT)),"sha256":sha(p)} for p in S6]},
     "matched_policy_atomic_stats":stats,
     "physical_alignment":phys,
     "target_alignment_summary":alignment,
     "atomic_evidence_graph":graph,
     "limitations":[
      "Velocity-tracking cross-correlations to singleton atomics are only available through T3-A parent-group correlations; no raw matched sample matrix was persisted for exact atomic progress correlations.",
      "D1 policies were trained on the old objective buckets, so current matched policy separation is sensitivity evidence, not proof of optimizability of the new atomic coordinate.",
      "No scaling or normalization is applied in T3-A2."
     ]
    }
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"stats":stats,"alignment":alignment,
                      "same_domain_edges":[e for e in graph if e["recombine_evidence"]!="NOT_PROPOSED"]},indent=2))

def run_post_v2_t3a3_objective_set_selection():
    """Run former post_v2_t3a3_objective_set_selection.py stage."""
    """T3-A3 read-only objective-set selection/compression review."""
    import json, hashlib
    from pathlib import Path
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    T3=ROOT/"runs/post_v2_t3a_regrouping-2026-09-23/audit.json"
    A2=ROOT/"runs/post_v2_t3a2_atomic_coherence-2026-09-23/audit.json"
    OUT=ROOT/"runs/post_v2_t3a3_objective_selection-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    LABS=("P","B","E")
    ATOMIC={
     "velocity_tracking":["track_lin_vel_xy_exp","track_ang_vel_z_exp"],
     "vertical_stability":["lin_vel_z_l2"],
     "angular_stability":["ang_vel_xy_l2"],
     "orientation_stability":["flat_orientation_l2"],
     "effort":["dof_torques_l2"],
     "joint_smoothness":["dof_acc_l2"],
     "control_smoothness":["action_rate_l2"],
     "gait_contact":["feet_air_time"],
    }
    PRIMARY_CANDIDATES=[k for k in ATOMIC if k!="gait_contact"]
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def pear(x,y):
        x=np.asarray(x,float);y=np.asarray(y,float)
        if x.std()<1e-12 or y.std()<1e-12:return 0.0
        return float(np.corrcoef(x,y)[0,1])
    def rankdata(x):
        # simple average-rank-free ordering sufficient here because ties are negligible
        order=np.argsort(np.asarray(x))
        ranks=np.empty(len(order),float);ranks[order]=np.arange(len(order),dtype=float)
        return ranks
    def spearman(x,y):return pear(rankdata(x),rankdata(y))
    def nondominated(vals):
        # maximize every reward coordinate
        n=len(vals); keep=[]
        for i in range(n):
            dominated=False
            for j in range(n):
                if i==j:continue
                if np.all(vals[j]>=vals[i]-1e-12) and np.any(vals[j]>vals[i]+1e-12):
                    dominated=True;break
            if not dominated:keep.append(i)
        return keep
    
    t3=json.loads(T3.read_text());a2=json.loads(A2.read_text())
    
    # Build matched 12-point atomic reward matrix from suite-policy means.
    points=[]; X=[]
    for r in t3["suite_policy_rows"]:
        vals=[]
        for name,terms in ATOMIC.items():
            vals.append(float(sum(r["term_mean"][t] for t in terms)))
        points.append({"suite":r["suite"],"policy":r["policy"]})
        X.append(vals)
    X=np.asarray(X,float); names=list(ATOMIC)
    name_to_idx={n:i for i,n in enumerate(names)}
    
    # Pairwise policy-induced redundancy. Center within each matched reset suite to remove environment/reset effects.
    Xc=X.copy()
    for suite in range(4):
        inds=[k for k,p in enumerate(points) if p["suite"]==suite]
        Xc[inds]=X[inds]-X[inds].mean(0,keepdims=True)
    corr={}
    for i,a in enumerate(names):
        for j,b in enumerate(names[i+1:],i+1):
            corr[f"{a}|{b}"]={
              "within_suite_centered_pearson":pear(Xc[:,i],Xc[:,j]),
              "within_suite_centered_spearman":spearman(Xc[:,i],Xc[:,j]),
              "raw_pearson_for_context":pear(X[:,i],X[:,j]),
            }
    
    # Pareto-front uniqueness within each matched suite, then aggregate.
    Pidx=[name_to_idx[n] for n in PRIMARY_CANDIDATES]
    pareto={}
    for n in PRIMARY_CANDIDATES:
        cols=[name_to_idx[x] for x in PRIMARY_CANDIDATES if x!=n]
        suite_rows=[]
        for suite in range(4):
            inds=[k for k,p in enumerate(points) if p["suite"]==suite]
            full_local=nondominated(X[inds][:,Pidx])
            drop_local=nondominated(X[inds][:,cols])
            sf=set(full_local);sd=set(drop_local)
            suite_rows.append({
              "suite":suite,"full_front_size":len(sf),"front_without_size":len(sd),
              "front_jaccard":len(sf&sd)/max(1,len(sf|sd)),"changes_front":bool(sf!=sd)
            })
        pareto[n]={
          "suite_change_fraction":float(np.mean([r["changes_front"] for r in suite_rows])),
          "mean_front_jaccard":float(np.mean([r["front_jaccard"] for r in suite_rows])),
          "suite_rows":suite_rows,
        }
    
    # Unique winner/order sensitivity across matched suites.
    winner_uniqueness={}
    for n in PRIMARY_CANDIDATES:
        i=name_to_idx[n]
        winners=[]
        for suite in range(4):
            inds=[k for k,p in enumerate(points) if p["suite"]==suite]
            best=max(inds,key=lambda k:X[k,i])
            winners.append(points[best]["policy"])
        winner_uniqueness[n]={"winners_by_suite":winners,"num_unique_winner_policies":len(set(winners))}
    
    # Role evidence from A2.
    stats=a2["matched_policy_atomic_stats"];align=a2["target_alignment_summary"]
    # conservative evidence categories, not a numeric score
    role={}
    for n in PRIMARY_CANDIDATES:
        sep=stats[n]["range_over_scale"]
        phys=align[n]["mean_abs_target_alignment"]
        pr=[abs(v["within_suite_centered_pearson"]) for k,v in corr.items() if n in k.split("|")]
        maxcorr=max(pr) if pr else 0.0
        role[n]={
          "task_relevance":"HIGH" if n in {"velocity_tracking","vertical_stability","angular_stability","orientation_stability","effort","control_smoothness"} else "MEDIUM",
          "physical_alignment":phys,
          "controllability_proxy_range_over_scale":sep,
          "controllability_proxy":"MODERATE" if sep>=0.04 else ("WEAK" if sep<0.02 else "LOW_MODERATE"),
          "max_abs_policy_level_corr":maxcorr,
          "pareto_suite_change_fraction_when_removed":pareto[n]["suite_change_fraction"],
          "pareto_mean_front_jaccard_without":pareto[n]["mean_front_jaccard"],
          "winner_diversity":winner_uniqueness[n],
        }
    
    # Compression logic based on current evidence:
    # - velocity tracking essential task objective.
    # - choose angular + orientation as stability axes; vertical retained as constraint candidate because P wins all suites and it is coupled to joint_smoothness.
    # - effort primary despite weak current sensitivity because semantics/physics are clean and deployment relevance high.
    # - control smoothness primary only if sufficiently distinct from effort and has measurable sensitivity.
    # - joint smoothness auxiliary until direct physical proxy validation.
    # - gait auxiliary.
    recommended={
     "primary_morl_objectives":[
       "velocity_tracking",
       "angular_stability",
       "orientation_stability",
       "effort",
       "control_smoothness"
     ],
     "constraints_or_safety_metrics":[
       "vertical_stability"
     ],
     "auxiliary_metrics":[
       "joint_smoothness",
       "gait_contact"
     ]
    }
    
    report={
     "schema":"v2_t3a3_objective_set_selection_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
     "provenance":{"t3a_sha256":sha(T3),"t3a2_sha256":sha(A2)},
     "points":points,"atomic_order":names,
     "policy_level_correlation":corr,
     "pareto_redundancy":pareto,
     "winner_uniqueness":winner_uniqueness,
     "role_evidence":role,
     "recommended_set":recommended,
     "rationale":{
       "velocity_tracking":"Keep as mandatory task-performance axis.",
       "angular_stability":"Keep as MORL axis: strong physical alignment, highest current matched separation among stability atomics, and distinct from orientation.",
       "orientation_stability":"Keep as MORL axis: strong tilt alignment and weak correlation with angular stability, so it contributes a different physical trade-off.",
       "vertical_stability":"Prefer constraint/safety role for now: physically valid but current old-bucket policies show P best in all matched suites; it is moderately coupled to joint_smoothness and does not yet demonstrate an independent desirable preference axis.",
       "effort":"Keep as MORL axis despite weak current separation because physical semantics are exceptionally clean and deployment relevance is direct; controllability must be revalidated after scaling/retraining.",
       "control_smoothness":"Keep as MORL axis provisionally: measurable physical meaning and nontrivial policy separation; distinct from effort and joint_smoothness.",
       "joint_smoothness":"Auxiliary until a direct joint-acceleration physical proxy and controllability evidence exist.",
       "gait_contact":"Auxiliary: distinct and sensitive but not justified as thesis-level preference axis."
     },
     "limitations":[
       "Pareto analysis uses only 12 matched points from three old-bucket D1 policies across four suites; it is a compression diagnostic, not a full reachable-set Pareto characterization.",
       "Current controllability proxies come from policies trained on old objectives, so weak separation does not prove an atomic objective is uncontrollable.",
       "No scaling/normalization or retraining is performed here."
     ]
    }
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"recommended_set":recommended,"pareto":pareto,"role_evidence":role},indent=2))

def run_post_v2_t3a4_controllability_audit():
    """Run former post_v2_t3a4_controllability_audit.py stage."""
    """T3-A4 read-only controllability / reachable-set audit."""
    import json, hashlib
    from pathlib import Path
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    T3=ROOT/"runs/post_v2_t3a_regrouping-2026-09-23/audit.json"
    A3=ROOT/"runs/post_v2_t3a3_objective_selection-2026-09-23/audit.json"
    OUT=ROOT/"runs/post_v2_t3a4_controllability-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    CANDS=["angular_stability","orientation_stability","effort","control_smoothness"]
    ATOMIC={
     "velocity_tracking":["track_lin_vel_xy_exp","track_ang_vel_z_exp"],
     "angular_stability":["ang_vel_xy_l2"],
     "orientation_stability":["flat_orientation_l2"],
     "effort":["dof_torques_l2"],
     "control_smoothness":["action_rate_l2"],
    }
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def fit_residual(y,x):
        x=np.asarray(x,float);y=np.asarray(y,float)
        A=np.stack([np.ones_like(x),x],1)
        beta=np.linalg.lstsq(A,y,rcond=None)[0]
        pred=A@beta
        return y-pred,beta
    def slope(a,b):
        da=b[0]-a[0]; db=b[1]-a[1]
        return None if abs(da)<1e-12 else float(db/da)
    
    t3=json.loads(T3.read_text());a3=json.loads(A3.read_text())
    
    # 12 matched points, all reward coordinates "higher is better".
    rows=[]
    for r in t3["suite_policy_rows"]:
        x={n:float(sum(r["term_mean"][t] for t in terms)) for n,terms in ATOMIC.items()}
        rows.append({"suite":r["suite"],"policy":r["policy"],**x})
    
    # Normalize each coordinate by global abs-mean scale for dimensionless slopes/widths.
    scale={n:float(np.mean(np.abs([r[n] for r in rows])))+1e-12 for n in ATOMIC}
    Z=[]
    for r in rows:
        Z.append({"suite":r["suite"],"policy":r["policy"],**{n:r[n]/scale[n] for n in ATOMIC}})
    
    result={}
    track=np.array([r["velocity_tracking"] for r in Z],float)
    for cand in CANDS:
        y=np.array([r[cand] for r in Z],float)
    
        # Conditional sensitivity: remove linear dependence on tracking globally and within-suite centered.
        resid,beta=fit_residual(y,track)
        raw_std=float(np.std(y)); resid_std=float(np.std(resid))
        residual_fraction=resid_std/(raw_std+1e-12)
    
        # Within-suite centered residual variation.
        yc=[];tc=[]
        for s in range(4):
            rs=[r for r in Z if r["suite"]==s]
            yy=np.array([r[cand] for r in rs]); tt=np.array([r["velocity_tracking"] for r in rs])
            yc.extend((yy-yy.mean()).tolist());tc.extend((tt-tt.mean()).tolist())
        yc=np.array(yc);tc=np.array(tc)
        cresid,cbeta=fit_residual(yc,tc)
        centered_residual_fraction=float(np.std(cresid)/(np.std(yc)+1e-12))
    
        # Pairwise finite differences inside each matched suite.
        pair_rows=[]
        labs=("P","B","E")
        for s in range(4):
            rs={r["policy"]:r for r in Z if r["suite"]==s}
            for i,a in enumerate(labs):
                for b in labs[i+1:]:
                    dt=rs[b]["velocity_tracking"]-rs[a]["velocity_tracking"]
                    dc=rs[b][cand]-rs[a][cand]
                    pair_rows.append({"suite":s,"a":a,"b":b,"delta_tracking":float(dt),"delta_candidate":float(dc),
                                      "abs_tradeoff_ratio":float(abs(dc)/(abs(dt)+1e-12)),
                                      "candidate_improves":bool(dc>0),"tracking_improves":bool(dt>0)})
    
        # Reachable width under tracking tolerances relative to best tracking policy within suite.
        tol_results={}
        for tol in (0.01,0.025,0.05,0.10):
            widths=[];counts=[];improve_exists=[]
            for s in range(4):
                rs=[r for r in Z if r["suite"]==s]
                best_track=max(r["velocity_tracking"] for r in rs)
                eligible=[r for r in rs if r["velocity_tracking"]>=best_track-tol]
                vals=[r[cand] for r in eligible]
                widths.append(float(max(vals)-min(vals)) if len(vals)>=2 else 0.0)
                counts.append(len(vals))
                bestcand=max(r[cand] for r in rs)
                improve_exists.append(bool(any(r[cand]>=bestcand-1e-12 for r in eligible)))
            tol_results[str(tol)]={
              "mean_width":float(np.mean(widths)),
              "max_width":float(np.max(widths)),
              "mean_eligible_count":float(np.mean(counts)),
              "fraction_suites_best_candidate_reachable_within_tracking_tol":float(np.mean(improve_exists)),
            }
    
        # Local Pareto availability: candidate can improve versus another policy while tracking loss <= tol.
        local={}
        for tol in (0.01,0.025,0.05,0.10):
            successes=0;total=0
            for pr in pair_rows:
                # either direction; candidate-improving move with tracking loss no worse than tol
                for dt,dc in ((pr["delta_tracking"],pr["delta_candidate"]),(-pr["delta_tracking"],-pr["delta_candidate"])):
                    if dc>0:
                        total+=1
                        if dt>=-tol:successes+=1
            local[str(tol)]={"fraction_candidate_improvements_with_tracking_loss_within_tol":float(successes/max(1,total)),
                             "num_improvement_moves":total}
    
        result[cand]={
          "conditional_sensitivity":{
            "global_tracking_beta":beta.tolist(),
            "residual_std_fraction_of_raw":float(residual_fraction),
            "within_suite_tracking_beta":cbeta.tolist(),
            "within_suite_residual_std_fraction_of_raw":centered_residual_fraction,
          },
          "finite_difference_pairs":pair_rows,
          "tracking_tolerance_width":tol_results,
          "local_tradeoff_availability":local,
          "current_policy_range_over_scale":a3["role_evidence"][cand]["controllability_proxy_range_over_scale"],
          "policy_level_max_abs_corr":a3["role_evidence"][cand]["max_abs_policy_level_corr"],
        }
    
    # Conservative role transition rules.
    roles={}
    for cand,x in result.items():
        rf=x["conditional_sensitivity"]["within_suite_residual_std_fraction_of_raw"]
        sep=x["current_policy_range_over_scale"]
        w5=x["tracking_tolerance_width"]["0.05"]["mean_width"]
        avail=x["local_tradeoff_availability"]["0.05"]["fraction_candidate_improvements_with_tracking_loss_within_tol"]
        if rf>=0.5 and sep>=0.03 and (w5>=0.01 or avail>=0.5):
            role="KEEP_AS_AXIS"
        elif rf<0.25 and x["policy_level_max_abs_corr"]>=0.9:
            role="INSUFFICIENT_EVIDENCE"
        elif cand=="effort" and sep<0.02:
            role="INSUFFICIENT_EVIDENCE"
        elif cand=="control_smoothness" and rf<0.4:
            role="MOVE_TO_REGULARIZER"
        else:
            role="INSUFFICIENT_EVIDENCE"
        roles[cand]=role
    
    report={
     "schema":"v2_t3a4_controllability_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
     "provenance":{"t3a_sha256":sha(T3),"t3a3_sha256":sha(A3)},
     "normalization":"Each reward coordinate divided by its global absolute mean scale; no reward/training changes.",
     "tracking_tolerances":[0.01,0.025,0.05,0.10],
     "candidate_results":result,
     "role_transition":roles,
     "limitations":[
       "Reachable set contains only three old-bucket D1 policies per suite, so this audit can establish local evidence but cannot prove global controllability.",
       "Linear residualization tests conditional variation, not causal intervention.",
       "No retraining, reward scaling, architecture changes, or policy interpolation are performed."
     ]
    }
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"roles":roles,"summary":{k:{
     "resid_frac":v["conditional_sensitivity"]["within_suite_residual_std_fraction_of_raw"],
     "sep":v["current_policy_range_over_scale"],
     "width05":v["tracking_tolerance_width"]["0.05"]["mean_width"],
     "avail05":v["local_tradeoff_availability"]["0.05"]["fraction_candidate_improvements_with_tracking_loss_within_tol"]
    } for k,v in result.items()}},indent=2))

def run_post_v2_t3a5_effort_controllability_screen():
    """Run former post_v2_t3a5_effort_controllability_screen.py stage."""
    """T3-A5 targeted effort controllability screen.
    
    Single-variable exploratory screen:
      reward vector = [velocity_tracking, effort, 0]
      fixed preference encodes relative emphasis beta as [1, beta, 0]/(1+beta)
    Everything else follows the frozen V1C PPO path.
    """
    import argparse,json,sys,time,traceback,hashlib
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    BETAS=(0.0,0.5,1.0,2.0)
    TRACK_TERMS=("track_lin_vel_xy_exp","track_ang_vel_z_exp")
    EFFORT_TERM="dof_torques_l2"
    ACCEPT={"effort_improvement_fraction":0.03,"tracking_degradation_fraction":0.025,"required_suites":3}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def pref(beta,n,device):
        if beta==0:return torch.tensor([1.,0.,0.],device=device).repeat(n,1)
        v=torch.tensor([1.,beta,0.],device=device);v=v/v.sum();return v.repeat(n,1)
    def vec_from_raw(raw,names):
        idx={n:i for i,n in enumerate(names)}
        progress=raw[:,idx[TRACK_TERMS[0]]]+raw[:,idx[TRACK_TERMS[1]]]
        effort=raw[:,idx[EFFORT_TERM]]
        zero=np.zeros_like(progress)
        return np.stack([progress,effort,zero],axis=1).astype(np.float32)
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--updates",type=int,default=100)
        ap.add_argument("--horizon",type=int,default=2)
        ap.add_argument("--num-envs",type=int,default=8)
        ap.add_argument("--eval-steps",type=int,default=64)
        ap.add_argument("--suites",type=int,default=4)
        ap.add_argument("--seed",type=int,default=0)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**kw):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**kw},sort_keys=True)+"\n")
        mark("RUN_STARTED",protocol="T3-A5-EFFORT",betas=BETAS,acceptance=ACCEPT,updates=args.updates)
        app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);mark("ENV_CREATED")
            obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            models={};records={};checkpoints={}
            for bi,beta in enumerate(BETAS):
                torch.manual_seed(21000+args.seed*100+bi);np.random.seed(21000+args.seed*100+bi)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu")
                opt=torch.optim.Adam(m.parameters(),lr=1e-3);w=pref(beta,args.num_envs,"cuda")
                cur,_=env.reset(seed=200001+args.seed*1000);cur=obs_tensor(cur).cuda();rec=[]
                mark("BETA_START",beta=beta,w=w[0].tolist())
                for update in range(1,args.updates+1):
                    ob=[];ac=[];old=[];rw=[];val=[];dn=[]
                    for _ in range(args.horizon):
                        with torch.no_grad():a,lp=m.act_with_preference(cur,w);v=m.value_with_preference(cur,w)
                        nxt,_,term,trunc,_=env.step(torch.clamp(a,-1,1))
                        raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vv=vec_from_raw(raw,names)
                        ob.append(cur);ac.append(a);old.append(lp);rw.append(torch.as_tensor(vv,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
                    with torch.no_grad():nv=m.value_with_preference(cur,w)
                    rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt)
                    fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(old);fw=w.repeat(args.horizon,1)
                    ratio=torch.exp(m.logp_with_preference(fo,fw,fa)-fold.detach())
                    al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,3).detach(),fw)
                    cl=vector_value_loss(m.value_with_preference(fo,fw),ret.reshape(-1,3).detach())
                    loss=al+cl;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    if update in (1,5,10,25,50,100):
                        rec.append({"update":update,"loss":float(loss.detach()),"reward_vec_stepdt_mean":rt.mean((0,1)).detach().cpu().tolist()})
                cp=args.output.parent/f"beta_{str(beta).replace('.','p')}_terminal.pt"
                torch.save({"schema":"t3_a5_effort_terminal_v1","beta":beta,"preference":w[0].tolist(),"update":args.updates,"model":m.state_dict()},cp)
                models[beta]=m.eval();records[str(beta)]=rec;checkpoints[str(beta)]={"path":str(cp),"sha256":sha(cp)}
                mark("BETA_DONE",beta=beta,checkpoint=str(cp))
            # Matched-state evaluation.
            evalrows=[]
            for suite in range(args.suites):
                seed=210001+suite
                for beta in BETAS:
                    m=models[beta];w=pref(beta,args.num_envs,"cuda");cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                    tr=[];eff=[];tor=[];vxerr=[];wzerr=[];done=np.zeros(args.num_envs,bool)
                    with torch.no_grad():
                        for _ in range(args.eval_steps):
                            a=torch.clamp(m.act_inference_with_preference(cur,w),-1,1)
                            nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);ii={n:i for i,n in enumerate(names)}
                            progress=raw[:,ii[TRACK_TERMS[0]]]+raw[:,ii[TRACK_TERMS[1]]]
                            effort=raw[:,ii[EFFORT_TERM]]
                            data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                            tr.append(float(progress.mean()));eff.append(float(effort.mean()))
                            tor.append(float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()))
                            vxerr.append(float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()))
                            wzerr.append(float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()))
                            done|=(term|trunc).cpu().numpy();cur=obs_tensor(nxt).cuda()
                    evalrows.append({"suite":suite,"beta":beta,"progress_reward":float(np.mean(tr)),"effort_reward":float(np.mean(eff)),
                                     "torque_norm":float(np.mean(tor)),"vx_error":float(np.mean(vxerr)),"wz_error":float(np.mean(wzerr)),
                                     "survival":float(1-done.mean())})
            # Compare to beta=0 per matched suite. Higher progress reward is better, less torque is better.
            comparisons={}
            for beta in BETAS[1:]:
                rows=[]
                for s in range(args.suites):
                    b=next(r for r in evalrows if r["suite"]==s and r["beta"]==0.0)
                    x=next(r for r in evalrows if r["suite"]==s and r["beta"]==beta)
                    effort_imp=(b["torque_norm"]-x["torque_norm"])/(abs(b["torque_norm"])+1e-12)
                    track_deg=(b["progress_reward"]-x["progress_reward"])/(abs(b["progress_reward"])+1e-12)
                    passes=effort_imp>=ACCEPT["effort_improvement_fraction"] and track_deg<=ACCEPT["tracking_degradation_fraction"]
                    rows.append({"suite":s,"effort_improvement_fraction":float(effort_imp),"tracking_degradation_fraction":float(track_deg),"pass":bool(passes),
                                 "baseline":b,"candidate":x})
                comparisons[str(beta)]={
                  "suite_rows":rows,
                  "pass_suites":int(sum(r["pass"] for r in rows)),
                  "mean_effort_improvement_fraction":float(np.mean([r["effort_improvement_fraction"] for r in rows])),
                  "mean_tracking_degradation_fraction":float(np.mean([r["tracking_degradation_fraction"] for r in rows])),
                  "acceptance_pass":bool(sum(r["pass"] for r in rows)>=ACCEPT["required_suites"] and np.mean([r["effort_improvement_fraction"] for r in rows])>0)
                }
            accepted=[float(k) for k,v in comparisons.items() if v["acceptance_pass"]]
            report={"schema":"t3_a5_effort_controllability_screen_v1","status":"SCREEN_COMPLETE","exploratory_training":True,
                    "seed":args.seed,"betas":list(BETAS),"acceptance":ACCEPT,"source_checkpoint":str(args.checkpoint),"source_checkpoint_sha256":sha(args.checkpoint),
                    "records":records,"checkpoints":checkpoints,"evaluation":evalrows,"comparisons":comparisons,
                    "accepted_betas":accepted,"screen_verdict":"EFFORT_CONTROLLABILITY_SUPPORTED" if accepted else "EFFORT_CONTROLLABILITY_NOT_SUPPORTED_IN_SCREEN",
                    "note":"Targeted one-seed effort controllability screen only; no final specialist or MORL training."}
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",verdict=report["screen_verdict"])
            print(json.dumps({"verdict":report["screen_verdict"],"accepted_betas":accepted,"comparisons":comparisons},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v2_t3a_regrouping_audit():
    """Run former post_v2_t3a_regrouping_audit.py stage."""
    """T3-A read-only candidate reward regrouping/separability audit on matched D1 specialists."""
    import argparse,atexit,hashlib,json,sys,time,traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    LABS=("P","B","E")
    GROUPS={
     "progress":["track_lin_vel_xy_exp","track_ang_vel_z_exp"],
     "stability":["lin_vel_z_l2","ang_vel_xy_l2","flat_orientation_l2"],
     "effort":["dof_torques_l2"],
     "smoothness":["dof_acc_l2","action_rate_l2"],
     "gait_contact_aux":["feet_air_time"],
     "unresolved_zero":["dof_pos_limits"],
    }
    CORE_GROUPS=("progress","stability","effort","smoothness","gait_contact_aux")
    PHYS=("vx_error","wz_error","abs_lin_vel_z","abs_ang_vel_xy","tilt_deg","torque_l2","action_rate_l2")
    
    def obs_tensor(v):
        if isinstance(v,dict):v=v.get("policy",next(iter(v.values())))
        return v if torch.is_tensor(v) else torch.as_tensor(v)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def safe_corr(x,y):
        x=np.asarray(x,float);y=np.asarray(y,float)
        if x.std()==0 or y.std()==0:return 0.0
        return float(np.corrcoef(x,y)[0,1])
    def corrmat(x):
        x=np.asarray(x,float);n=x.shape[1];o=np.zeros((n,n))
        for i in range(n):
          for j in range(n):o[i,j]=safe_corr(x[:,i],x[:,j]) if i!=j else 1.0
        return o
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--num-envs",type=int,default=32)
        ap.add_argument("--steps",type=int,default=250)
        ap.add_argument("--suites",type=int,default=4)
        ap.add_argument("--seed-base",type=int,default=188001)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl");err=args.output.with_name(args.output.stem+".ERROR.json")
        state={"written":False,"failed":False}
        def mark(event,**kw):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**kw},sort_keys=True)+"\n")
        def fail(reason,exc=None):
            state["failed"]=True;err.write_text(json.dumps({"status":"ERROR","reason":reason,"error":str(exc) if exc else None,"traceback":traceback.format_exc() if exc else None},indent=2)+"\n");mark("ERROR",reason=reason)
        def guard():
            if not state["written"] and not state["failed"]:fail("PROCESS_EXIT_BEFORE_ARTIFACT")
        atexit.register(guard);mark("RUN_STARTED",protocol="V2-T3-A",groups=GROUPS)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed_base
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);mark("ENV_CREATED")
            obs,_=env.reset(seed=args.seed_base);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            policies={};ckpts={}
            for lab in LABS:
                p=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(p,map_location="cuda",weights_only=False)
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();policies[lab]=m;ckpts[lab]=p
            wp={lab:torch.tensor({"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}[lab],device="cuda").repeat(args.num_envs,1) for lab in LABS}
            suite_rows=[]; term_names=None
            for suite in range(args.suites):
                seed=args.seed_base+suite
                for lab in LABS:
                    cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                    terms=[];phys=[];returned=[];recon=[];done_any=np.zeros(args.num_envs,bool);finite=True
                    with torch.no_grad():
                        for step in range(args.steps):
                            action=torch.clamp(policies[lab].act_inference_with_preference(cur,wp[lab]),-1,1)
                            prev=env.unwrapped.action_manager.prev_action.clone()
                            ar=(action-prev).square().mean(-1)
                            nxt,reward,term,trunc,_=env.step(action)
                            mgr=env.unwrapped.reward_manager;names=list(mgr.active_terms)
                            if term_names is None:term_names=names
                            elif names!=term_names:raise RuntimeError("reward term order changed")
                            raw=mgr._step_reward.detach().cpu().numpy().astype(np.float64)
                            terms.append(raw);returned.append(reward.detach().cpu().numpy().astype(np.float64));recon.append(raw.sum(1)*float(env.unwrapped.step_dt))
                            data=env.unwrapped.scene["robot"].data;cmd=env.unwrapped.command_manager.get_command("base_velocity");q=data.root_quat_w
                            roll=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
                            pitch=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
                            phys.append(np.stack([
                              (data.root_lin_vel_b[:,0]-cmd[:,0]).abs().cpu().numpy(),
                              (data.root_ang_vel_b[:,2]-cmd[:,2]).abs().cpu().numpy(),
                              data.root_lin_vel_b[:,2].abs().cpu().numpy(),
                              torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).cpu().numpy(),
                              torch.rad2deg(torch.maximum(roll.abs(),pitch.abs())).cpu().numpy(),
                              torch.linalg.vector_norm(data.applied_torque,dim=-1).cpu().numpy(),
                              ar.cpu().numpy(),
                            ],1))
                            done_any|=(term|trunc).cpu().numpy();finite &= bool(torch.isfinite(action).all() and torch.isfinite(reward).all());cur=obs_tensor(nxt).cuda()
                    T=np.concatenate(terms);P=np.concatenate(phys);ti={n:i for i,n in enumerate(term_names)}
                    G=np.stack([T[:,[ti[t] for t in GROUPS[g]]].sum(1) for g in CORE_GROUPS],1)
                    suite_rows.append({
                      "suite":suite,"policy":lab,
                      "term_mean":{n:float(T[:,i].mean()) for i,n in enumerate(term_names)},
                      "term_std":{n:float(T[:,i].std()) for i,n in enumerate(term_names)},
                      "term_nonzero":{n:float(np.mean(np.abs(T[:,i])>0)) for i,n in enumerate(term_names)},
                      "group_mean":{g:float(G[:,i].mean()) for i,g in enumerate(CORE_GROUPS)},
                      "group_std":{g:float(G[:,i].std()) for i,g in enumerate(CORE_GROUPS)},
                      "group_nonzero":{g:float(np.mean(np.abs(G[:,i])>0)) for i,g in enumerate(CORE_GROUPS)},
                      "group_corr":corrmat(G).tolist(),
                      "group_physical_corr":{g:{m:safe_corr(G[:,i],P[:,j]) for j,m in enumerate(PHYS)} for i,g in enumerate(CORE_GROUPS)},
                      "within_group_term_corr":{
                        g:(corrmat(T[:,[ti[t] for t in GROUPS[g]]]).tolist() if len(GROUPS[g])>1 else [[1.0]]) for g in CORE_GROUPS
                      },
                      "physical_mean":{m:float(P[:,j].mean()) for j,m in enumerate(PHYS)},
                      "survival":float(1-done_any.mean()),"finite":finite,
                      "scalar_reconstruction_max_abs_error":float(np.max(np.abs(np.concatenate(returned)-np.concatenate(recon))))
                    })
                    mark("POLICY_SUITE_DONE",suite=suite,policy=lab)
            # matched-policy separability and winner fractions
            sep={};winner_fraction={};group_scale={}
            for g in CORE_GROUPS:
                means={lab:[] for lab in LABS}
                for r in suite_rows:means[r["policy"]].append(r["group_mean"][g])
                allvals=np.array([x for lab in LABS for x in means[lab]],float)
                scale=float(np.mean(np.abs(allvals)));policy_avg={lab:float(np.mean(v)) for lab,v in means.items()}
                prange=max(policy_avg.values())-min(policy_avg.values())
                winners=[]
                for s in range(args.suites):
                    vals={lab:next(r for r in suite_rows if r["suite"]==s and r["policy"]==lab)["group_mean"][g] for lab in LABS}
                    winners.append(max(vals,key=vals.get))  # higher reward is better
                sep[g]={"policy_mean":policy_avg,"policy_range":float(prange),"abs_mean_scale":scale,"range_over_scale":float(prange/(scale+1e-12))}
                winner_fraction[g]={lab:float(np.mean([w==lab for w in winners])) for lab in LABS}
                group_scale[g]={"global_abs_mean":scale,"global_std_of_suite_policy_means":float(np.std(allvals))}
            # aggregate correlation / coherence over matched rows
            group_corr_mean=np.mean([np.array(r["group_corr"]) for r in suite_rows],axis=0)
            within={}
            for g in CORE_GROUPS:
                mats=[np.array(r["within_group_term_corr"][g]) for r in suite_rows]
                M=np.mean(mats,axis=0)
                if M.shape[0]>1:
                    vals=[M[i,j] for i in range(M.shape[0]) for j in range(i+1,M.shape[1])]
                    within[g]={"mean_pairwise_corr":float(np.mean(vals)),"mean_abs_pairwise_corr":float(np.mean(np.abs(vals))),"matrix":M.tolist()}
                else:within[g]={"mean_pairwise_corr":1.0,"mean_abs_pairwise_corr":1.0,"matrix":M.tolist()}
            # physical alignment averaged across policy/suite
            align={g:{m:float(np.mean([r["group_physical_corr"][g][m] for r in suite_rows])) for m in PHYS} for g in CORE_GROUPS}
            result={
              "schema":"v2_t3a_candidate_regrouping_audit_v1","status":"READ_ONLY_COMPLETE","measurement_only":True,
              "candidate_groups":GROUPS,"core_group_order":list(CORE_GROUPS),
              "protocol":{"matched_reset":True,"num_envs":args.num_envs,"steps":args.steps,"suites":args.suites,"seed_base":args.seed_base},
              "provenance":{"specialists":{lab:{"path":str(ckpts[lab].relative_to(ROOT)),"sha256":sha(ckpts[lab])} for lab in LABS}},
              "scalar_reconstruction":{"max_abs_error":float(max(r["scalar_reconstruction_max_abs_error"] for r in suite_rows)),"pass":bool(max(r["scalar_reconstruction_max_abs_error"] for r in suite_rows)<=1e-5)},
              "policy_separation":sep,"winner_fraction":winner_fraction,"group_scale":group_scale,
              "mean_group_pairwise_corr":group_corr_mean.tolist(),"within_group_coherence":within,
              "mean_group_physical_corr":align,"suite_policy_rows":suite_rows,
              "integrity":{"all_finite":all(r["finite"] for r in suite_rows),"min_survival":float(min(r["survival"] for r in suite_rows))}
            }
            args.output.write_text(json.dumps(result,indent=2)+"\n");state["written"]=True;mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE")
            print(json.dumps({"status":result["status"],"separation":sep,"winners":winner_fraction,"within":within,"alignment":align},indent=2))
        except BaseException as exc:
            fail("EVALUATION_EXCEPTION",exc);raise
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_post_v2_t3b_scaling_audit():
    """Run former post_v2_t3b_scaling_audit.py stage."""
    """T3-B read-only scaling/normalization audit for frozen 4D MORL vector."""
    import argparse,hashlib,json,sys,time,traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OBJ=("velocity_tracking","angular_stability","orientation_stability","control_smoothness")
    TERMS={
     "velocity_tracking":("track_lin_vel_xy_exp","track_ang_vel_z_exp"),
     "angular_stability":("ang_vel_xy_l2",),
     "orientation_stability":("flat_orientation_l2",),
     "control_smoothness":("action_rate_l2",),
    }
    PHYS_REF={
     # reward-space magnitude at a documented unit/reference condition.
     # tracking: max of frozen positive terms 1.5+0.75.
     # angular: |w_xy|^2 = 1 -> 0.05.
     # orientation: projected-gravity xy norm^2 = 1 -> 2.5.
     # control: summed squared action change = 1 -> 0.01.
     "velocity_tracking":2.25,
     "angular_stability":0.05,
     "orientation_stability":2.5,
     "control_smoothness":0.01,
    }
    EPS=1e-8
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def objective_vec(raw,names):
        idx={n:i for i,n in enumerate(names)}
        cols=[]
        for o in OBJ:
            cols.append(sum(raw[:,idx[t]] for t in TERMS[o]))
        return np.stack(cols,1).astype(np.float32)
    def discounted_rtg(rew,done,gamma=.99):
        # [T,N,O]
        out=np.zeros_like(rew,dtype=np.float64);running=np.zeros_like(rew[0],dtype=np.float64)
        for t in range(len(rew)-1,-1,-1):
            running=rew[t]+gamma*running*(~done[t])[:,None]
            out[t]=running
        return out
    def flat_grad(grads):
        xs=[g.reshape(-1) for g in grads if g is not None]
        return torch.cat(xs) if xs else torch.zeros(1,device="cuda")
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--num-envs",type=int,default=16)
        ap.add_argument("--steps",type=int,default=192)
        ap.add_argument("--reset-seeds",type=int,nargs="+",default=[230001,230101,230201])
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic,initialize_from_rsl_m01
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.reset_seeds[0]
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            obs,_=env.reset(seed=args.reset_seeds[0]);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            model=V1CSharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(model,args.checkpoint,device="cpu");model.eval()
            w=torch.tensor([1.,0.,0.],device="cuda").repeat(args.num_envs,1)
            all_reward=[]; seed_rows=[]; grad_records=[]
            actor_params=[p for n,p in model.named_parameters() if n.startswith("actor_") or n=="log_std"]
            for si,seed in enumerate(args.reset_seeds):
                cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                obs_b=[];act_b=[];rew_b=[];done_b=[]
                for _ in range(args.steps):
                    with torch.no_grad():a,_=model.act_with_preference(cur,w)
                    nxt,_,term,trunc,_=env.step(torch.clamp(a,-1,1))
                    mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy().astype(np.float64);names=list(mgr.active_terms)
                    vec=objective_vec(raw,names)
                    obs_b.append(cur.detach().cpu());act_b.append(a.detach().cpu());rew_b.append(vec);done_b.append((term|trunc).cpu().numpy().astype(bool))
                    cur=obs_tensor(nxt).cuda()
                R=np.asarray(rew_b);D=np.asarray(done_b);all_reward.append(R.reshape(-1,4))
                seed_rows.append({"seed":seed,"mean":R.mean((0,1)).tolist(),"std":R.std((0,1)).tolist(),
                                  "nonzero_fraction":(np.abs(R)>0).mean((0,1)).tolist()})
                # Keep rollout for later gradient audit.
                grad_records.append((torch.cat(obs_b).cuda(),torch.cat(act_b).cuda(),R,D))
            A=np.concatenate(all_reward,0)
            # distribution statistics
            q=np.quantile(A,[.01,.05,.25,.5,.75,.95,.99],axis=0)
            med=np.median(A,0);mad=np.median(np.abs(A-med),0)
            scales={
              "raw":np.ones(4),
              "std":A.std(0),
              "iqr_sigma":(np.quantile(A,.75,axis=0)-np.quantile(A,.25,axis=0))/1.349,
              "mad_sigma":1.4826*mad,
              "abs_mean":np.mean(np.abs(A),0),
              "physical_reference":np.array([PHYS_REF[o] for o in OBJ],float),
            }
            scales={k:np.maximum(v,EPS) for k,v in scales.items()}
            dist={
              "mean":A.mean(0).tolist(),"std":A.std(0).tolist(),"abs_mean":np.mean(np.abs(A),0).tolist(),
              "nonzero_fraction":(np.abs(A)>0).mean(0).tolist(),
              "quantiles":{str(p):q[i].tolist() for i,p in enumerate([.01,.05,.25,.5,.75,.95,.99])},
              "mad":mad.tolist(),"iqr":(np.quantile(A,.75,axis=0)-np.quantile(A,.25,axis=0)).tolist(),
              "seed_rows":seed_rows,
            }
            candidates={}
            for cname,scale in scales.items():
                norm=A/scale
                # uniform scalar contribution in reward space
                abs_contrib=np.mean(np.abs(0.25*norm),0)
                reward_share=abs_contrib/(abs_contrib.sum()+1e-12)
                seed_reward_shares=[]
                grad_norms_seed=[]
                adv_std_seed=[]
                grad_cos_seed=[]
                for rec_i,(fo,fa,R,D) in enumerate(grad_records):
                    rtg=discounted_rtg(R/scale,D)
                    # advantage proxy: center RTG per objective over the sampled rollout.
                    adv=rtg.reshape(-1,4);adv=adv-adv.mean(0,keepdims=True)
                    adv_std_seed.append(adv.std(0).tolist())
                    logp=model.logp_with_preference(fo,w.repeat(args.steps,1),fa)
                    gvec=[]
                    for j in range(4):
                        aj=torch.as_tensor(adv[:,j],device="cuda",dtype=logp.dtype)
                        loss=-(logp*aj.detach()).mean()
                        g=flat_grad(torch.autograd.grad(loss,actor_params,retain_graph=True,allow_unused=True))
                        gvec.append(g)
                    norms=np.array([float(x.norm()) for x in gvec])
                    grad_norms_seed.append(norms.tolist())
                    # pairwise gradient cosine
                    C=np.eye(4)
                    for i in range(4):
                        for j in range(i+1,4):
                            C[i,j]=C[j,i]=float((gvec[i]@gvec[j])/(gvec[i].norm()*gvec[j].norm()+1e-12))
                    grad_cos_seed.append(C.tolist())
                    rr=R.reshape(-1,4)/scale
                    ac=np.mean(np.abs(.25*rr),0);seed_reward_shares.append((ac/(ac.sum()+1e-12)).tolist())
                G=np.asarray(grad_norms_seed)
                gmean=G.mean(0);gshare=gmean/(gmean.sum()+1e-12)
                Gshare=G/(G.sum(1,keepdims=True)+1e-12)
                candidates[cname]={
                  "divisor":scale.tolist(),
                  "reward_abs_contribution_share":reward_share.tolist(),
                  "reward_dominance_max_share":float(reward_share.max()),
                  "reward_share_seed_std":np.std(np.asarray(seed_reward_shares),axis=0).tolist(),
                  "advantage_proxy_std_mean":np.mean(np.asarray(adv_std_seed),axis=0).tolist(),
                  "actor_gradient_norm_mean":gmean.tolist(),
                  "actor_gradient_share":gshare.tolist(),
                  "actor_gradient_share_seed_std":Gshare.std(0).tolist(),
                  "actor_gradient_norm_seed_cv":(G.std(0)/(G.mean(0)+1e-12)).tolist(),
                  "actor_gradient_dominance_max_share":float(gshare.max()),
                  "actor_gradient_max_min_ratio":float(gmean.max()/(gmean.min()+1e-12)),
                  "actor_gradient_pairwise_cosine_mean":np.mean(np.asarray(grad_cos_seed),axis=0).tolist(),
                }
            report={
              "schema":"v2_t3b_scaling_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
              "objective_order":list(OBJ),"sign_convention":"All four are weighted rewards with higher=better; three penalties are negative-valued and improve toward zero.",
              "source_checkpoint":str(args.checkpoint),"source_checkpoint_sha256":sha(args.checkpoint),
              "protocol":{"num_envs":args.num_envs,"steps":args.steps,"reset_seeds":args.reset_seeds,"stochastic_policy_actions":True},
              "raw_distribution":dist,"normalization_candidates":candidates,
              "physical_reference_definition":PHYS_REF,
              "advantage_note":"Advantage scale uses centered discounted return-to-go as a read-only policy-gradient proxy; no critic training or optimizer step is performed.",
              "effort_excluded":True,
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({"distribution":dist,"candidate_summary":{k:{
              "divisor":v["divisor"],"reward_share":v["reward_abs_contribution_share"],"grad_share":v["actor_gradient_share"],
              "grad_ratio":v["actor_gradient_max_min_ratio"],"reward_dom":v["reward_dominance_max_share"],"grad_dom":v["actor_gradient_dominance_max_share"]
            } for k,v in candidates.items()}},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");raise
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_post_v2_t4_init_smoke():
    """Run former post_v2_t4_init_smoke.py stage."""
    import argparse,json,torch,sys
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
    
    def source_actor(obs,state):
        x=obs
        for i in (0,2,4):
            x=torch.nn.functional.elu(torch.nn.functional.linear(x,state[f"actor.{i}.weight"],state[f"actor.{i}.bias"]))
        return torch.tanh(torch.nn.functional.linear(x,state["actor.6.weight"],state["actor.6.bias"]))*T4SharedActorCritic.ACTION_CLIP
    def source_value(obs,state):
        x=obs
        for i in (0,2,4):
            x=torch.nn.functional.elu(torch.nn.functional.linear(x,state[f"critic.{i}.weight"],state[f"critic.{i}.bias"]))
        return torch.nn.functional.linear(x,state["critic.6.weight"],state["critic.6.bias"]).squeeze(-1)
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"));ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
        p=torch.load(args.checkpoint,map_location="cpu",weights_only=False);s=p.get("model_state_dict",p.get("model",p))
        torch.manual_seed(44);obs=torch.randn(64,48);m=T4SharedActorCritic(48,12);initialize_from_rsl_m01(m,args.checkpoint)
        src_mu=source_actor(obs,s);src_v=source_value(obs,s)
        prefs=torch.tensor([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7],[.25,.25,.25,.25]])
        rows=[]
        for w0 in prefs:
            w=w0.repeat(len(obs),1)
            mu=m.act_inference_with_preference(obs,w);v=m.value_with_preference(obs,w)
            rows.append({"w":w0.tolist(),"action_max_abs_diff":float((mu-src_mu).abs().max()),"value_max_abs_diff":float((v-src_v[:,None]).abs().max())})
        logstd_diff=float((m.log_std-s["std"].log()).abs().max())
        report={"schema":"t4_init_smoke_v1","rows":rows,"logstd_max_abs_diff":logstd_diff,
                "pass":all(r["action_max_abs_diff"]<=1e-6 and r["value_max_abs_diff"]<=1e-5 for r in rows) and logstd_diff<=1e-7}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
    if True:main()

def run_post_v2_t4_specialists():
    """Run former post_v2_t4_specialists.py stage."""
    """T4: generate four fixed-preference specialists on frozen normalized 4D objectives and validate endpoints."""
    import argparse,hashlib,json,sys,time,traceback
    from pathlib import Path
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
    }
    OBJ_NAMES=("velocity_tracking","angular_stability","orientation_stability","control_smoothness")
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def weighted_terms(raw,names):
        return {n:raw[:,i] for i,n in enumerate(names)}
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--updates",type=int,default=300);ap.add_argument("--horizon",type=int,default=2)
        ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--eval-steps",type=int,default=64)
        ap.add_argument("--reset-suites",type=int,default=4);ap.add_argument("--seed",type=int,default=0)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--critic-head-init",choices=("scalar","zero"),default="scalar")
        ap.add_argument("--actor-lr",type=float,default=1e-3)
        ap.add_argument("--critic-lr",type=float,default=1e-3)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event,**extra):
            with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        mark("RUN_STARTED",protocol="T4",updates=args.updates,prefs={k:v.tolist() for k,v in PREFS.items()})
        app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
            from talon_rl.rewards.objectives import normalized_objective_vector,raw_objective_vector,NORMALIZATION_DIVISORS
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);mark("ENV_CREATED")
            obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            models={};records={};checkpoints={}
            for idx,label in enumerate(ORDER):
                w_np=PREFS[label];torch.manual_seed(31000+idx);np.random.seed(31000+idx)
                m=T4SharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init=args.critic_head_init)
                actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
                critic_params=[p for n,p in m.named_parameters() if n.startswith("critic_")]
                opt=torch.optim.Adam([{"params":actor_params,"lr":args.actor_lr},{"params":critic_params,"lr":args.critic_lr}])
                w=torch.as_tensor(np.repeat(w_np[None,:],args.num_envs,axis=0),device="cuda")
                cur,_=env.reset(seed=310001+idx*1000);cur=obs_tensor(cur).cuda();rec=[]
                mark("SPECIALIST_START",specialist=label,w=w_np.tolist())
                for update in range(1,args.updates+1):
                    ob=[];ac=[];pre=[];old=[];rw=[];val=[];dn=[]
                    for _ in range(args.horizon):
                        with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
                        # T5 repair: the exact policy action is already in [-1,1] and is applied unchanged.
                        nxt,_,term,trunc,_=env.step(a)
                        raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                        vec=normalized_objective_vector(weighted_terms(raw,names),shape=(args.num_envs,))
                        ob.append(cur);ac.append(a);pre.append(u);old.append(lp);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda())
                        cur=obs_tensor(nxt).cuda()
                    with torch.no_grad():nv=m.value_with_preference(cur,w)
                    rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt)
                    fo=torch.cat(ob);fa=torch.cat(ac);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1)
                    ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
                    ratio_maxerr=float((ratio-1).abs().max().detach())
                    if ratio_maxerr>1e-4 or not torch.isfinite(ratio).all():
                        raise RuntimeError(f"PPO pre-update ratio invariant failed: max|r-1|={ratio_maxerr}")
                    al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                    cl=vector_value_loss(m.value_with_preference(fo,fw),ret.reshape(-1,4).detach())
                    loss=al+cl;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    if update in (1,5,10,25,50,100,200,300):
                        rec.append({"update":update,"loss":float(loss.detach()),"normalized_reward_stepdt_mean":rt.mean((0,1)).detach().cpu().tolist()})
                cp=args.output.parent/f"specialist_{label}_terminal.pt"
                torch.save({"schema":"t4_specialist_terminal_v1","specialist":label,"objective_order":OBJ_NAMES,"w":w_np.tolist(),"update":args.updates,"normalization_divisors":NORMALIZATION_DIVISORS.tolist(),"model":m.state_dict()},cp)
                models[label]=m.eval();records[label]=rec;checkpoints[label]={"path":str(cp),"sha256":sha(cp)};mark("SPECIALIST_DONE",specialist=label)
            # Paired deterministic endpoint evaluation. Each policy acts using its own fixed training preference.
            rows=[]
            for suite in range(args.reset_suites):
                reset_seed=320001+suite
                for label in ORDER:
                    m=models[label];w_np=PREFS[label];w=torch.as_tensor(np.repeat(w_np[None,:],args.num_envs,axis=0),device="cuda")
                    cur,_=env.reset(seed=reset_seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device="cuda")
                    raw_obj=[];norm_obj=[];metrics=[];done_any=np.zeros(args.num_envs,bool)
                    with torch.no_grad():
                        for _ in range(args.eval_steps):
                            a=m.act_inference_with_preference(cur,w)
                            nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                            rt=raw_objective_vector(weighted_terms(raw,names),shape=(args.num_envs,))
                            nt=normalized_objective_vector(weighted_terms(raw,names),shape=(args.num_envs,))
                            data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                            metrics.append({
                              "vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),
                              "wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),
                              "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                              "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                              "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
                              "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean()),
                              "torque_norm":float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),
                            })
                            raw_obj.append(rt.mean(0));norm_obj.append(nt.mean(0));done_any|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
                    norm=np.asarray(norm_obj).mean(0);rawm=np.asarray(raw_obj).mean(0);mm={k:float(np.mean([x[k] for x in metrics])) for k in metrics[0]}
                    scalarized={ev:float(norm@PREFS[ev]) for ev in ORDER}
                    rows.append({"suite":suite,"policy":label,"train_w":w_np.tolist(),"raw_objective_mean":rawm.tolist(),"normalized_objective_mean":norm.tolist(),
                                 "scalarized_by_eval_preference":scalarized,"physical":mm,"survival":float(1-done_any.mean())})
            # Cross-eval matrix and endpoint gates.
            matrix={ev:{pol:float(np.mean([r["scalarized_by_eval_preference"][ev] for r in rows if r["policy"]==pol])) for pol in ORDER} for ev in ORDER}
            objective_winner_fraction={};physical_winner_fraction={};diag_scalar_winner_fraction={}
            phys_key={"T":None,"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
            for i,label in enumerate(ORDER):
                ow=[];pw=[];sw=[]
                for suite in range(args.reset_suites):
                    rr=[r for r in rows if r["suite"]==suite]
                    ow.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==label)
                    if label=="T":
                        # tracking physical score: lower normalized sum of vx/wz absolute errors (equal weight in validation only)
                        pw.append(min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==label)
                    else:
                        k=phys_key[label];pw.append(min(rr,key=lambda r:r["physical"][k])["policy"]==label)
                    sw.append(max(rr,key=lambda r:r["scalarized_by_eval_preference"][label])["policy"]==label)
                objective_winner_fraction[label]=float(np.mean(ow));physical_winner_fraction[label]=float(np.mean(pw));diag_scalar_winner_fraction[label]=float(np.mean(sw))
            # Safety/constraint monitors.
            min_survival=min(r["survival"] for r in rows)
            vertical_by_policy={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in rows if r["policy"]==p])) for p in ORDER}
            track_vertical=vertical_by_policy["T"]
            vertical_ratio={p:vertical_by_policy[p]/(track_vertical+1e-12) for p in ORDER}
            endpoint_pass={p:bool(objective_winner_fraction[p]>=.75 and physical_winner_fraction[p]>=.75 and diag_scalar_winner_fraction[p]>=.75) for p in ORDER}
            safety_pass=bool(min_survival>=.95 and max(vertical_ratio.values())<=2.0)
            report={
              "schema":"t4_specialist_generation_validation_v1","status":"TRAINING_AND_VALIDATION_COMPLETE",
              "updates":args.updates,"critic_head_init":args.critic_head_init,"actor_lr":args.actor_lr,"critic_lr":args.critic_lr,"objective_order":OBJ_NAMES,"fixed_preferences":{k:v.tolist() for k,v in PREFS.items()},
              "normalization_divisors":NORMALIZATION_DIVISORS.tolist(),"checkpoints":checkpoints,"training_records":records,
              "validation_protocol":{"paired_reset":True,"reset_seed_base":320001,"suites":args.reset_suites,"eval_steps":args.eval_steps,
                                     "deterministic_actor":True,"clip_actions":1.0,"policy_acts_with_training_preference":True},
              "rows":rows,"cross_evaluation_matrix":matrix,
              "gates":{"objective_winner_fraction":objective_winner_fraction,"physical_winner_fraction":physical_winner_fraction,
                       "diagonal_scalarized_winner_fraction":diag_scalar_winner_fraction,"endpoint_pass":endpoint_pass,
                       "min_survival":min_survival,"vertical_ratio_to_tracking_policy":vertical_ratio,"safety_pass":safety_pass,
                       "overall_semantic_endpoint_pass":bool(all(endpoint_pass.values()) and safety_pass)},
              "notes":{"effort":"monitor only; excluded from 4D preference vector","vertical_stability":"constraint monitor only; excluded from 4D preference vector"}
            }
            args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",overall_pass=report["gates"]["overall_semantic_endpoint_pass"])
            print(json.dumps({"gates":report["gates"],"matrix":matrix},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

STAGES = {
    "post_v2_a_offline_latent_anchor_audit": run_post_v2_a_offline_latent_anchor_audit,
    "post_v2_b1_latent_learning_audit": run_post_v2_b1_latent_learning_audit,
    "post_v2_b_isaac_smoke": run_post_v2_b_isaac_smoke,
    "post_v2_b_short_screen": run_post_v2_b_short_screen,
    "post_v2_b_terminal_sensitivity": run_post_v2_b_terminal_sensitivity,
    "post_v2_c1_action_behavior_causal_audit": run_post_v2_c1_action_behavior_causal_audit,
    "post_v2_c2_projection_audit": run_post_v2_c2_projection_audit,
    "post_v2_c3_trajectory_divergence_audit": run_post_v2_c3_trajectory_divergence_audit,
    "post_v2_c4_saturation_authority_audit": run_post_v2_c4_saturation_authority_audit,
    "post_v2_c5_raw_output_authority_audit": run_post_v2_c5_raw_output_authority_audit,
    "post_v2_c_semantic_response_audit": run_post_v2_c_semantic_response_audit,
    "post_v2_r1_authority_audit": run_post_v2_r1_authority_audit,
    "post_v2_r1_authority_compare": run_post_v2_r1_authority_compare,
    "post_v2_r1_isaac_smoke": run_post_v2_r1_isaac_smoke,
    "post_v2_r1_semantic_response_audit": run_post_v2_r1_semantic_response_audit,
    "post_v2_r1_short_authority_screen": run_post_v2_r1_short_authority_screen,
    "post_v2_r21_shared_residual_geometry_audit": run_post_v2_r21_shared_residual_geometry_audit,
    "post_v2_r22_coefficient_gradient_audit": run_post_v2_r22_coefficient_gradient_audit,
    "post_v2_r23a_coeff_guidance_gradient_audit": run_post_v2_r23a_coeff_guidance_gradient_audit,
    "post_v2_r23c_normalized_guidance_audit": run_post_v2_r23c_normalized_guidance_audit,
    "post_v2_r23d_min_norm_constraint_audit": run_post_v2_r23d_min_norm_constraint_audit,
    "post_v2_r2_freeze_basis": run_post_v2_r2_freeze_basis,
    "post_v2_r2_semantic_geometry_audit": run_post_v2_r2_semantic_geometry_audit,
    "post_v2_r2_short_coefficient_screen": run_post_v2_r2_short_coefficient_screen,
    "post_v2_t0_c3_rollout32": run_post_v2_t0_c3_rollout32,
    "post_v2_t0_offline_crossrank": run_post_v2_t0_offline_crossrank,
    "post_v2_t1_specialist_reward_semantics_audit": run_post_v2_t1_specialist_reward_semantics_audit,
    "post_v2_t3a2_atomic_coherence_audit": run_post_v2_t3a2_atomic_coherence_audit,
    "post_v2_t3a3_objective_set_selection": run_post_v2_t3a3_objective_set_selection,
    "post_v2_t3a4_controllability_audit": run_post_v2_t3a4_controllability_audit,
    "post_v2_t3a5_effort_controllability_screen": run_post_v2_t3a5_effort_controllability_screen,
    "post_v2_t3a_regrouping_audit": run_post_v2_t3a_regrouping_audit,
    "post_v2_t3b_scaling_audit": run_post_v2_t3b_scaling_audit,
    "post_v2_t4_init_smoke": run_post_v2_t4_init_smoke,
    "post_v2_t4_specialists": run_post_v2_t4_specialists,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
