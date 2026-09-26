"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_post_v1_d5a_specialist_compatibility_audit():
    """Run former post_v1_d5a_specialist_compatibility_audit.py stage."""
    """Post-V1 D5-A: read-only compatibility audit for fixed-preference specialists.
    
    Checks whether D1 specialist actions are state-consistent teacher targets and
    whether simple action interpolation remains finite and locomotion-valid.
    No parameters are updated.
    """
    
    import argparse
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    PREFS = {
        "P": np.array([.8, .1, .1], dtype=np.float32),
        "B": np.array([.1, .8, .1], dtype=np.float32),
        "E": np.array([.1, .1, .8], dtype=np.float32),
    }
    PAIRS = [("P", "B"), ("P", "E"), ("B", "E")]
    
    
    def obs_tensor(x):
        if isinstance(x, dict):
            x = x.get("policy", next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    
    def tilt_deg(q):
        _, x, y, _ = [q[:, i] for i in range(4)]
        return torch.rad2deg(torch.acos((1 - 2 * (x * x + y * y)).clamp(-1, 1)))
    
    
    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--output", type=Path, required=True)
        ap.add_argument("--num-envs", type=int, default=8)
        ap.add_argument("--steps", type=int, default=32)
        ap.add_argument("--reset-suites", type=int, default=4)
        args = ap.parse_args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        life = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
    
        def mark(event, **extra):
            with life.open("a") as f:
                f.write(json.dumps({"event": event, "unix": time.time(), **extra}, sort_keys=True) + "\n")
    
        mark("RUN_STARTED", protocol="POST-V1-D5A", measurement_only=True)
        app = env = None
        try:
            from isaaclab.app import AppLauncher
    
            saved = sys.argv[:]
            sys.argv = [sys.argv[0]]
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            sys.argv = saved
            mark("APP_INIT_OK")
    
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = args.num_envs
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            mark("ENV_CREATED")
            obs, _ = env.reset(seed=47001)
            obs = obs_tensor(obs).cuda()
            action_dim = env.unwrapped.action_manager.total_action_dim
            manager = env.unwrapped.reward_manager
            robot = env.unwrapped.scene["robot"]
    
            models = {}
            for label in PREFS:
                path = ROOT / f"runs/post_v1_d1-2026-09-22/specialist_{label}_terminal.pt"
                payload = torch.load(path, map_location="cuda", weights_only=False)
                model = V1CSharedActorCritic(obs.shape[-1], action_dim).cuda()
                model.load_state_dict(payload["model"])
                model.eval()
                models[label] = model
            mark("CHECKPOINTS_LOADED", count=len(models))
    
            w = {
                k: torch.as_tensor(np.repeat(v[None, :], args.num_envs, axis=0), device="cuda")
                for k, v in PREFS.items()
            }
            condition_names = ["P", "B", "E", "P_B_mid", "P_E_mid", "B_E_mid", "center_mean"]
            condition_weights = {
                "P": {"P": 1.0}, "B": {"B": 1.0}, "E": {"E": 1.0},
                "P_B_mid": {"P": .5, "B": .5},
                "P_E_mid": {"P": .5, "E": .5},
                "B_E_mid": {"B": .5, "E": .5},
                "center_mean": {"P": 1 / 3, "B": 1 / 3, "E": 1 / 3},
            }
    
            target_rows = []
            interpolation_rows = []
            for suite in range(args.reset_suites):
                cur, _ = env.reset(seed=47001 + suite)
                cur = obs_tensor(cur).cuda()
                with torch.no_grad():
                    acts = {k: torch.clamp(models[k].act_inference_with_preference(cur, w[k]), -1, 1) for k in models}
                    pair_stats = {}
                    for a, b in PAIRS:
                        delta = acts[a] - acts[b]
                        pair_stats[f"{a}_vs_{b}"] = {
                            "mean_l2": float(torch.linalg.vector_norm(delta, dim=-1).mean()),
                            "mean_abs": float(delta.abs().mean()),
                            "per_dof_std_mean": float(delta.std(dim=0).mean()),
                            "cosine_mean": float(torch.nn.functional.cosine_similarity(acts[a], acts[b], dim=-1).mean()),
                        }
                    stacked = torch.stack([acts[k] for k in PREFS], dim=0)
                    target_rows.append({
                        "reset_suite": suite,
                        "specialist_action_mean_abs": {k: float(acts[k].abs().mean()) for k in acts},
                        "per_state_target_variance_mean": float(stacked.var(dim=0, unbiased=False).mean()),
                        "per_state_target_variance_p95": float(torch.quantile(stacked.var(dim=0, unbiased=False).mean(dim=-1), .95)),
                        "pairwise": pair_stats,
                    })
    
                for condition in condition_names:
                    cur, _ = env.reset(seed=47001 + suite)
                    cur = obs_tensor(cur).cuda()
                    metrics = []
                    finite = True
                    previous_action = torch.zeros((args.num_envs, action_dim), device="cuda")
                    with torch.no_grad():
                        for _ in range(args.steps):
                            specialist_actions = {
                                k: torch.clamp(models[k].act_inference_with_preference(cur, w[k]), -1, 1)
                                for k in models
                            }
                            action = sum(weight * specialist_actions[k] for k, weight in condition_weights[condition].items())
                            finite = finite and bool(torch.isfinite(action).all())
                            nxt, _, term, trunc, _ = env.step(action)
                            raw = manager._step_reward.detach().cpu().numpy()
                            names = list(manager.active_terms)
                            vec = group_v1b_s7_terms({n: raw[:, i] for i, n in enumerate(names)}, shape=(args.num_envs,))
                            data = robot.data
                            cmd = env.unwrapped.command_manager.get_command("base_velocity")
                            metrics.append({
                                "objective_return": vec.mean(0).tolist(),
                                "vx_error": float((data.root_lin_vel_b[:, 0] - cmd[:, 0]).abs().mean()),
                                "tilt_deg": float(tilt_deg(data.root_quat_w).mean()),
                                "ang_vel_xy": float(torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1).mean()),
                                "torque_norm": float(torch.linalg.vector_norm(data.applied_torque, dim=-1).mean()),
                                "action_rate": float(torch.linalg.vector_norm(action - previous_action, dim=-1).mean()),
                                "termination_rate": float((term | trunc).float().mean()),
                            })
                            previous_action = action
                            cur = obs_tensor(nxt).cuda()
                    interpolation_rows.append({
                        "reset_suite": suite,
                        "condition": condition,
                        "finite": finite,
                        "metrics_mean": {
                            k: (np.asarray([m[k] for m in metrics]).mean(0).tolist() if k == "objective_return" else float(np.mean([m[k] for m in metrics])))
                            for k in metrics[0]
                        },
                    })
    
            report = {
                "schema": "post_v1_d5a_specialist_compatibility_audit_v1",
                "status": "MEASUREMENT_COMPLETE",
                "measurement_only": True,
                "d1_checkpoints": {k: f"runs/post_v1_d1-2026-09-22/specialist_{k}_terminal.pt" for k in PREFS},
                "reset_suites": args.reset_suites,
                "steps": args.steps,
                "target_consistency": target_rows,
                "interpolation_conditions": interpolation_rows,
                "condition_weights": condition_weights,
                "note": "D5-A audits state-matched specialist target consistency, objective response, and interpolation validity; no parameters are updated.",
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            mark("ARTIFACT_WRITTEN", path=str(args.output))
            mark("RUN_DONE", status=report["status"])
            print(json.dumps({"status": report["status"], "target_rows": len(target_rows), "interpolation_rows": len(interpolation_rows)}, indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem + ".ERROR.json").write_text(json.dumps({"status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
            mark("ERROR", error=str(exc))
            raise
        finally:
            if env is not None:
                env.close()
            if app is not None:
                app.close()
    
    
    if True:
        main()

def run_post_v1_d5b_alignment_pilot():
    """Run former post_v1_d5b_alignment_pilot.py stage."""
    """Post-V1 D5-B: compatibility-aware specialist-alignment pilot.
    
    Adds only 0.1 * L_align to the frozen D4-B objective. Specialist teachers are
    loaded from D1, clipped deterministic means, frozen, and evaluated no-grad.
    The only verdict checkpoint is update 100.
    """
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def flat(gs,ps): return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs,ps)])
    
    def align_components(teachers,obs,w):
        with torch.no_grad():
            fixed_w={k:teachers['_w'][k][:1].repeat(obs.shape[0],1) for k in PREFS}
            fixed={k:torch.clamp(teachers[k].act_inference_with_preference(obs,fixed_w[k]),-1,1) for k in PREFS}
            be=(w[:,1:2]*fixed['B']+w[:,2:3]*fixed['E'])/(w[:,1:2]+w[:,2:3]+1e-8)
        return fixed['P'],be
    
    def measure_alignment(student,teachers,obs):
        with torch.no_grad():
            fixed_w={k:teachers['_w'][k][:1].repeat(obs.shape[0],1) for k in PREFS}
            acts={k:torch.clamp(student.act_inference_with_preference(obs,fixed_w[k]),-1,1) for k in PREFS}
            refs={k:torch.clamp(teachers[k].act_inference_with_preference(obs,fixed_w[k]),-1,1) for k in PREFS}
            own={k:float(torch.linalg.vector_norm(acts[k]-refs[k],dim=-1).mean()) for k in PREFS}
            other={k:float(torch.stack([torch.linalg.vector_norm(acts[k]-refs[j],dim=-1) for j in PREFS if j!=k]).mean()) for k in PREFS}
            cos={}
            for a,b in [('P','B'),('P','E'),('B','E')]:
                cos[f'{a}_vs_{b}']=float(torch.nn.functional.cosine_similarity(acts[a]-acts[b],refs[a]-refs[b],dim=-1).mean())
        return {'own_distance':own,'other_distance_mean':other,'pairwise_difference_cosine':cos}
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=100);ap.add_argument('--horizon',type=int,default=2);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--eval-steps',type=int,default=32);ap.add_argument('--reset-suites',type=int,default=4);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V1-D5B',lambda_aux=1.0,lambda_align=0.1,verdict_update=100);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.preferences.v1c_curriculum import load_manifest,sample_preferences
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
            from talon_rl.models.auxiliary.preference_head import D4BAuxActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            source=torch.load(ROOT/'runs/post_v1_d4b-2026-09-22/d4b_terminal.pt',map_location='cuda',weights_only=False)['model'];student=D4BAuxActorCritic(obs.shape[-1],ad).cuda();student.load_state_dict(source);student.train()
            teachers={'_w':{k:torch.as_tensor(np.repeat(v[None,:],args.num_envs,0),device='cuda') for k,v in PREFS.items()}}
            for k in PREFS:
                path=ROOT/f'runs/post_v1_d1-2026-09-22/specialist_{k}_terminal.pt';m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval();teachers[k]=m
            for m in teachers.values():
                if isinstance(m,torch.nn.Module):
                    for p in m.parameters(): p.requires_grad_(False)
            mark('CHECKPOINTS_LOADED',student='D4B',teachers=3)
            opt=torch.optim.Adam(student.parameters(),lr=1e-3);manifest=load_manifest();rng=np.random.default_rng(100000);manager=env.unwrapped.reward_manager;cur,_=env.reset(seed=0);cur=obs_tensor(cur).cuda();params=list(student.actor_body.parameters())+list(student.actor_mean.parameters())+list(student.preference_head.parameters());records=[];checks={}
            for update in range(1,args.updates+1):
                w_np,_=sample_preferences(rng,update,args.num_envs,'full_simplex_control',manifest);w=torch.as_tensor(w_np,device='cuda');ob=[];ac=[];old=[];rew=[];val=[];done=[]
                for _ in range(args.horizon):
                    with torch.no_grad(): a,lp=student.act_with_preference(cur,w);v=student.value_with_preference(cur,w)
                    a=torch.clamp(a,-1,1);nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));ob.append(cur);ac.append(a);old.append(lp);rew.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val.append(v);done.append((term|trunc).to('cuda'));cur=obs_tensor(nxt).cuda()
                with torch.no_grad(): nv=student.value_with_preference(cur,w)
                rt=torch.stack(rew);vt=torch.stack(val);dt=torch.stack(done).bool();adv,ret=vector_gae(rt,vt,nv,dt);fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(old);fw=w.repeat(args.horizon,1);ratio=torch.exp(student.logp_with_preference(fo,fw,fa)-fold.detach());ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,3).detach(),fw);aux=student.auxiliary_preference_loss(fo,fw);tp,tbe=align_components(teachers,fo,fw);mu=torch.clamp(student.act_inference_with_preference(fo,fw),-1,1);mse_p=(mu-tp).pow(2).mean(-1,keepdim=True);mse_be=(mu-tbe).pow(2).mean(-1,keepdim=True);align=(fw[:,0:1]*mse_p+fw[:,1:].sum(-1,keepdim=True)*mse_be).mean();critic=vector_value_loss(student.value_with_preference(fo,fw),ret.reshape(-1,3).detach());loss=ppo+aux+0.1*align;gg=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(align,params,retain_graph=True,allow_unused=True);gaux=flat(gg,params);galign=flat(ga,params);ratio_a=float(torch.linalg.vector_norm(galign)/(torch.linalg.vector_norm(gaux)+1e-12));opt.zero_grad(set_to_none=True);(loss+critic).backward();opt.step();records.append({'update':update,'ppo_loss':float(ppo.detach()),'aux_loss':float(aux.detach()),'align_loss':float(align.detach()),'critic_loss':float(critic.detach()),'align_to_aux_grad_ratio':ratio_a})
                if update in (25,50,75,100): student.eval();checks[str(update)]={'alignment':measure_alignment(student,teachers,cur)};student.train()
            student.eval();behavior=[]
            for label,v in PREFS.items():
                fixed_w=torch.as_tensor(np.repeat(v[None,:],args.num_envs,0),device='cuda')
                for suite in range(args.reset_suites):
                    cur,_=env.reset(seed=47001+suite);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device='cuda');vecs=[];metrics=[]
                    with torch.no_grad():
                        for _ in range(args.eval_steps):
                            action=torch.clamp(student.act_inference_with_preference(cur,fixed_w),-1,1);nxt,_,term,trunc,_=env.step(action);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');metrics.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'tilt_deg':float(torch.rad2deg(torch.acos((1-2*(data.root_quat_w[:,1]**2+data.root_quat_w[:,2]**2)).clamp(-1,1))).mean()),'ang_vel_xy':float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),'torque_norm':float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),'action_rate':float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),'survival':float(1-((term|trunc).float().mean()))});vecs.append(vec.mean(0));prev=action;cur=obs_tensor(nxt).cuda()
                    behavior.append({'preference':label,'reset_suite':suite,'objective_return_mean':np.asarray(vecs).mean(0).tolist(),'metrics_mean':{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}})
            terminal=args.output.parent/'d5b_terminal.pt';torch.save({'schema':'post_v1_d5b_terminal_v1','update':args.updates,'lambda_align':0.1,'model':student.state_dict(),'optimizer':opt.state_dict()},terminal);report={'schema':'post_v1_d5b_alignment_pilot_v1','status':'PILOT_COMPLETE','measurement_only_after_training':False,'lambda_aux':1.0,'lambda_align':0.1,'verdict_checkpoint_update':100,'source_checkpoint':'runs/post_v1_d4b-2026-09-22/d4b_terminal.pt','teacher_checkpoints':{k:f'runs/post_v1_d1-2026-09-22/specialist_{k}_terminal.pt' for k in PREFS},'records':records,'alignment_checkpoints':checks,'behavior':behavior,'terminal_checkpoint':str(terminal),'note':'One-seed pilot; chain-based verdict only. Intermediate checkpoints are diagnostic and update 100 is the sole verdict checkpoint.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'verdict_update':100,'checkpoints':list(checks),'behavior_rows':len(behavior)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True: main()

STAGES = {
    "post_v1_d5a_specialist_compatibility_audit": run_post_v1_d5a_specialist_compatibility_audit,
    "post_v1_d5b_alignment_pilot": run_post_v1_d5b_alignment_pilot,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
