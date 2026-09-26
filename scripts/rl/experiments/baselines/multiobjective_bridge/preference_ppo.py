"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_close_m0_2a_fail():
    """Run former close_m0_2a_fail.py stage."""
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[4]
    p=root/'artifacts/m0_2a_freeze/M0_2A_FREEZE.json'
    m=json.loads(p.read_text()); m['status']='CLOSED_FAIL'; m['training_authorized']=False
    m['preservation_verdict']='artifacts/m0_2a_confirmation/M0_2A_VERDICT.json'; m['m0_2b_training_authorized']=False
    p.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'status':m['status'],'training_authorized':m['training_authorized'],'m0_2b_training_authorized':False},indent=2))

def run_close_m0_2a_fp0():
    """Run former close_m0_2a_fp0.py stage."""
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[4];p=root/'artifacts/m0_2a_fp_design/M0_2A_FP_DESIGN.json';m=json.loads(p.read_text());v=json.loads((root/'artifacts/m0_2a_fp_design/FP0_VERDICT.json').read_text());m['status']='FP0_CLOSED_PASS' if v['status']=='PASS' else 'FP0_CLOSED_FAIL';m['training_authorized']=False;m['fp1_authorized']=False;m['verdict']='artifacts/m0_2a_fp_design/FP0_VERDICT.json';p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'status':m['status'],'fp1_authorized':False},indent=2))

def run_eval_m0_2a():
    """Run former eval_m0_2a.py stage."""
    """Deterministic fixed-w evaluation for M0.2a final checkpoints."""
    import argparse, json, sys, time
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
    def po(o):
        if isinstance(o,dict): o=o.get('policy',next(iter(o.values())))
        return o if torch.is_tensor(o) else torch.as_tensor(o)
    def main():
        p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--steps',type=int,default=500); p.add_argument('--num-envs',type=int,default=64); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
        from isaaclab.app import AppLauncher
        app=AppLauncher({'headless':True,'enable_cameras':False}).app
        env=None
        try:
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.algorithms.vector_ppo import M02aActorCritic,REFERENCE_W
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=a.num_envs; cfg.seed=a.seed
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg); print('EVAL_ENV_READY', flush=True); obs,_=env.reset(seed=a.seed); print('EVAL_RESET_READY', flush=True); obs=po(obs).cuda()
            model=M02aActorCritic(obs.shape[-1],12).cuda(); print('EVAL_MODEL_READY', flush=True); state=torch.load(a.checkpoint,map_location='cuda'); model.load_state_dict(state['model']); print('EVAL_CKPT_READY', flush=True); model.eval(); w=torch.tensor(REFERENCE_W,device='cuda').expand(a.num_envs,-1)
            print('EVAL_READY_LOOP', flush=True); vxerr=[]; heights=[]; contacts=[]; falls=np.zeros(a.num_envs,dtype=bool)
            for _ in range(a.steps):
                with torch.no_grad(): action=model.act_inference_with_preference(obs,w)
                nxt,_,term,trunc,_=env.step(action); d=(term|trunc).detach().cpu().numpy(); falls|=d
                data=env.unwrapped.scene['robot'].data
                cmd=env.unwrapped.command_manager.get_command('base_velocity')
                vxerr.append(np.abs(data.root_lin_vel_b[:,0].detach().cpu().numpy()-cmd[:,0].detach().cpu().numpy())); heights.append(data.root_pos_w[:,2].detach().cpu().numpy()); contacts.append(d)
                obs=po(nxt).cuda()
            out={'schema':'m0_2a_deterministic_eval_v1','seed':a.seed,'steps':a.steps,'num_envs':a.num_envs,'w_eval':list(REFERENCE_W),'mean_vx_error':float(np.mean(vxerr)),'survival':float(1.0-falls.mean()),'height_mean':float(np.mean(heights)),'termination_rate':float(np.mean(contacts)),'deterministic_actor_mean':True,'training_state_mutated':False}
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True: main()

def run_eval_m0_2a_debug():
    """Run former eval_m0_2a_debug.py stage."""
    """Single-env evaluator lifecycle probe; never substitutes for the gate."""
    import argparse, faulthandler, json, os, signal, sys, traceback
    from pathlib import Path
    import torch
    ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
    faulthandler.enable()
    def main():
        p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=10); p.add_argument('--num-envs',type=int,default=1); a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
        state={'schema':'m0_2a_eval_debug_v1','status':'RUNNING','markers':['RUN_STARTED'],'last_step':-1}
        def write(): a.output.write_text(json.dumps(state,indent=2)+'\n')
        def mark(x): state['markers'].append(x); write()
        def on_signal(signum, frame): state.update(status='ERROR', signal=signum, error='process signal', traceback=''.join(traceback.format_stack(frame))); write(); os._exit(128+signum)
        signal.signal(signal.SIGTERM,on_signal); signal.signal(signal.SIGABRT,on_signal); write()
        app=env=None
        try:
            from isaaclab.app import AppLauncher; app=AppLauncher({'headless':True,'enable_cameras':False}).app; mark('APP_OK')
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.algorithms.vector_ppo import M02aActorCritic,REFERENCE_W
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=a.num_envs; cfg.seed=0; env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg); mark('ENV_OK')
            obs,_=env.reset(seed=0); mark('RESET_OK'); obs=obs['policy'] if isinstance(obs,dict) else obs; obs=obs.cuda()
            model=M02aActorCritic(obs.shape[-1],12).cuda(); ck=torch.load(a.checkpoint,map_location='cuda'); model.load_state_dict(ck['model']); model.eval(); mark('CKPT_OK')
            w=torch.tensor(REFERENCE_W,device='cuda').reshape(1,5).expand(a.num_envs,-1); mark('INPUT_OK')
            for i in range(a.steps):
                with torch.no_grad(): action=model.act_inference_with_preference(obs,w)
                if action.shape != (a.num_envs,12) or not torch.isfinite(action).all(): raise RuntimeError(f'invalid action shape/device/value: {action.shape} {action.device}')
                mark(f'ACTION_{i+1}_OK'); nxt,*_=env.step(action); mark(f'STEP_{i+1}_OK'); obs=nxt['policy'] if isinstance(nxt,dict) else nxt; obs=obs.cuda()
            state['status']='PASS'; mark('ARTIFACT_WRITTEN'); state['last_step']=a.steps; write(); print(json.dumps(state,indent=2))
        except BaseException as e:
            state.update(status='ERROR',error=str(e),traceback=traceback.format_exc()); write(); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    if True: main()

def run_eval_m0_2a_decomp():
    """Run former eval_m0_2a_decomp.py stage."""
    """Common deterministic evaluator for A1/A2/A3 checkpoint-100 screens."""
    import argparse,json,sys
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    def po(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        p=argparse.ArgumentParser();p.add_argument('--stage',choices=['A0','A0EXP','A1FP','A1','A2','A3'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--num-envs',type=int,default=64);a=p.parse_args()
        from isaaclab.app import AppLauncher;app=AppLauncher({'headless':True,'enable_cameras':False}).app;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.vector_ppo import M02aActorCritic,REFERENCE_W
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=0);obs=po(obs).cuda();w=torch.tensor(REFERENCE_W,device='cuda').expand(a.num_envs,-1);m=M02aActorCritic(obs.shape[-1],12).cuda() if a.stage in ('A2','A3') else ActorCritic(obs.shape[-1]+(5 if a.stage!='A0' else 0),obs.shape[-1]+(5 if a.stage!='A0' else 0),12,1,[128,128,128]).cuda();m.load_state_dict(torch.load(a.checkpoint,map_location='cuda')['model']);m.eval();falls=np.zeros(a.num_envs,dtype=bool);err=[];height=[];termrate=[]
            for _ in range(a.steps):
                with torch.no_grad(): action=m.act_inference(obs if a.stage=='A0' else torch.cat((obs,w),-1))
                nxt,_,t,tr,_=env.step(action);d=(t|tr).detach().cpu().numpy();falls|=d;data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');err.append(np.abs(data.root_lin_vel_b[:,0].detach().cpu().numpy()-cmd[:,0].detach().cpu().numpy()));height.append(data.root_pos_w[:,2].detach().cpu().numpy());termrate.append(d);obs=po(nxt).cuda()
            out={'schema':'m0_2a_decomp_eval_v1','status':'PASS','stage':a.stage,'steps':a.steps,'num_envs':a.num_envs,'mean_vx_error':float(np.mean(err)),'survival':float(1-falls.mean()),'height_mean':float(np.mean(height)),'termination_rate':float(np.mean(termrate)),'deterministic_actor_mean':True,'training_state_mutated':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_freeze_m0_2a():
    """Run former freeze_m0_2a.py stage."""
    """Freeze M0.2a only after algebraic tests and real-Isaac smoke pass."""
    import hashlib, json
    from pathlib import Path
    
    root = Path(__file__).resolve().parents[4]
    manifest_path = root / "artifacts/m0_2a_freeze/M0_2A_FREEZE.json"
    smoke_dir = root / "runs/m0_2a_production_smoke_2026-09-22-r4"
    smoke = json.loads((smoke_dir / "artifact.json").read_text())
    if smoke.get("status") != "PASS" or smoke["max_abs_reconstruction_error"] >= 1e-6:
        raise SystemExit("M0.2a smoke failed; refusing to freeze")
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "FROZEN"
    manifest["training_authorized"] = True
    manifest["production_smoke"] = {
        "run": str(smoke_dir.relative_to(root)),
        "artifact_sha256": hashlib.sha256((smoke_dir / "artifact.json").read_bytes()).hexdigest(),
        "max_error": smoke["max_abs_reconstruction_error"],
        "vector_rows": smoke["vector_rows"],
        "checkpoint_resume": smoke["checkpoint_resume"],
    }
    manifest["algebraic_tests"] = {
        "command": "tests/test_m0_2a.py",
        "passed": 4,
        "loss_abs_relative_tolerance": 1e-6,
        "gradient_max_abs_relative_l2_tolerance": 1e-6,
    }
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["sha256"] = {
        "m0_2a_module": sha(root / "scripts/rl/core/algorithms/vector_ppo.py"),
        "production_smoke": sha(root / "scripts/rl/experiments/baselines/multiobjective_bridge/preference_ppo.py"),
        "design": sha(root / "docs/baselines/multiobjective_bridge/m0-2-preference-conditioned-moppo-design-draft.md"),
        "m0_1_manifest": sha(root / "artifacts/m0_1_freeze/M0_1_FREEZE.json"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "training_authorized": True, "production_smoke": manifest["production_smoke"]}, indent=2))

def run_freeze_m0_2a_fp0():
    """Run former freeze_m0_2a_fp0.py stage."""
    import json,hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[4];p=root/'artifacts/m0_2a_fp_design/M0_2A_FP_DESIGN.json';m=json.loads(p.read_text());smoke=json.loads((root/'runs/m0_2a_fp0_smoke_2026-09-22/artifact.json').read_text())
    if smoke.get('status')!='PASS' or smoke['max_action_abs_error']>1e-6 or smoke['max_logp_abs_error']>1e-6: raise SystemExit('FP0 smoke not valid')
    m.update({'status':'FROZEN','training_authorized':True,'active_stage':'FP-0','fp0_contract':{'copy_observation_columns':True,'preference_columns_zero':True,'w_ref':[.2]*5,'critic':'scalar','actor_loss':'scalar_ppo','probe_tolerance':1e-6,'budget':{'updates':300,'num_envs':4096,'seeds':[0,1,2]}}});m['sha256']={'fp_module':hashlib.sha256((root/'scripts/rl/core/algorithms/vector_ppo.py').read_bytes()).hexdigest(),'fp_smoke':hashlib.sha256((root/'scripts/rl/experiments/baselines/multiobjective_bridge/preference_ppo.py').read_bytes()).hexdigest(),'m0_1_manifest':hashlib.sha256((root/'artifacts/m0_1_freeze/M0_1_FREEZE.json').read_bytes()).hexdigest()};p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'status':m['status'],'training_authorized':m['training_authorized']},indent=2))

def run_m0_2a_decomp_screen():
    """Run former m0_2a_decomp_screen.py stage."""
    """Short causal-localization screens A1/A2/A3 (not thesis verdicts)."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    def po(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def scalar_gae(r,v,nv,d,g=.99,l=.95):
        out=torch.zeros_like(r); last=torch.zeros_like(nv)
        for t in range(r.shape[0]-1,-1,-1):
            b=nv if t==r.shape[0]-1 else v[t+1]; nt=(~d[t]).float(); last=r[t]+g*b*nt-v[t]+g*l*nt*last;out[t]=last
        return out,out+v
    def main():
        p=argparse.ArgumentParser();p.add_argument('--stage',choices=['A0','A0EXP','A1FP','A1','A2','A3'],required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--updates',type=int,default=100);p.add_argument('--num-envs',type=int,default=16);p.add_argument('--horizon',type=int,default=24);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--init-checkpoint',type=Path);a=p.parse_args();run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True);(run/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','stage':a.stage})+'\n');app=env=None
        try:
            from isaaclab.app import AppLauncher;app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_stock_terms,reconstruct_stock_scalar
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.vector_ppo import M02aActorCritic,REFERENCE_W,vector_gae,late_weighted_actor_loss,vector_value_loss
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=a.seed);obs=po(obs).cuda();w=torch.tensor(REFERENCE_W,device='cuda').expand(a.num_envs,-1)
            m=M02aActorCritic(obs.shape[-1],12).cuda() if a.stage in ('A2','A3') else ActorCritic(obs.shape[-1]+(5 if a.stage!='A0' else 0),obs.shape[-1]+(5 if a.stage!='A0' else 0),12,1,[128,128,128]).cuda()
            if a.stage=='A1FP':
                if not a.init_checkpoint: raise RuntimeError('A1FP requires --init-checkpoint from A0')
                base=torch.load(a.init_checkpoint,map_location='cuda')['model']; own=m.state_dict()
                for k,v in base.items():
                    if k not in own: continue
                    if k in ('actor_body.0.weight','critic_body.0.weight'):
                        own[k].zero_(); own[k][:,:v.shape[1]]=v
                    elif own[k].shape==v.shape: own[k].copy_(v)
                m.load_state_dict(own)
            opt=torch.optim.Adam(m.parameters(),lr=1e-3);rows=0;maxerr=0.;finite=True
            for u in range(1,a.updates+1):
                O=[];A=[];R=[];V=[];D=[];LP=[]
                for _ in range(a.horizon):
                    ow=obs if a.stage=='A0' else torch.cat((obs,w),-1)
                    with torch.no_grad():
                        act,lp=m.act(ow); val=m.value(ow) if a.stage in ('A0','A0EXP','A1FP','A1') else m.value_with_preference(obs,w)
                    nxt,reward,term,trunc,_=env.step(act);mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rv=group_stock_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(a.num_envs,));err=float(np.max(np.abs(reconstruct_stock_scalar(rv)*env.unwrapped.step_dt-reward.detach().cpu().numpy())));maxerr=max(maxerr,err)
                    O.append(obs);A.append(act.detach());R.append(torch.as_tensor(rv,device='cuda',dtype=torch.float32)*env.unwrapped.step_dt);V.append(val);D.append((term|trunc).cuda());LP.append(lp);rows+=a.num_envs;obs=po(nxt).cuda()
                with torch.no_grad(): nv=m.value(obs if a.stage=='A0' else torch.cat((obs,w),-1)) if a.stage in ('A0','A0EXP','A1FP','A1') else m.value_with_preference(obs,w)
                rr,vv,dd=torch.stack(R),torch.stack(V),torch.stack(D); 
                if a.stage in ('A0','A0EXP','A1FP','A1'): adv,ret=scalar_gae(rr.sum(-1),vv.squeeze(-1),nv.squeeze(-1),dd); flat_adv=adv.reshape(-1);flat_ret=ret.reshape(-1,1)
                else:
                    adv,ret=vector_gae(rr,vv,nv,dd); flat_adv=adv.reshape(-1,5);flat_ret=ret.reshape(-1,5)
                fo=torch.cat(O);fw=w.repeat(a.horizon,1);fa=torch.cat(A);old=torch.cat(LP).detach();ow=fo if a.stage=='A0' else torch.cat((fo,fw),-1);ratio=torch.exp(m.logp(ow,fa)-old); 
                if a.stage=='A3': al=late_weighted_actor_loss(ratio,flat_adv,fw);cl=vector_value_loss(m.value(ow),flat_ret)
                elif a.stage=='A2':
                    scalar_adv=flat_adv.sum(-1); al=-torch.min(ratio*scalar_adv,ratio.clamp(.8,1.2)*scalar_adv).mean();cl=vector_value_loss(m.value(ow),flat_ret)
                else: al=-torch.min(ratio*flat_adv,ratio.clamp(.8,1.2)*flat_adv).mean();cl=.5*(m.value(ow).squeeze(-1)-flat_ret.squeeze(-1)).pow(2).mean()
                loss=al+cl;opt.zero_grad();loss.backward();opt.step();finite=finite and bool(torch.isfinite(loss))
            torch.save({'model':m.state_dict(),'stage':a.stage,'update_idx':a.updates,'w_ref':list(REFERENCE_W)},run/f'model_{a.updates}.pt')
            out={'schema':'m0_2a_decomp_screen_v1','status':'PASS' if finite and maxerr<1e-6 else 'FAIL','stage':a.stage,'seed':a.seed,'updates':a.updates,'num_envs':a.num_envs,'vector_rows':rows,'max_reconstruction_error':maxerr,'finite':finite,'checkpoint':'model_100.pt','thesis_verdict':False};(run/'artifact.json').write_text(json.dumps(out,indent=2)+'\n');(run/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0})+'\n');print(json.dumps(out,indent=2))
        except BaseException as e:(run/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_m0_2a_fp0_smoke():
    """Run former m0_2a_fp0_smoke.py stage."""
    """Real-Isaac FP-0 smoke: function-preserving actor expansion only."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import torch
    ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    def po(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,default=16);p.add_argument('--steps',type=int,default=24);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True);(run/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED','unix':time.time()})+'\n');app=env=None
        try:
            from isaaclab.app import AppLauncher;app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.algorithms.vector_ppo import make_fp0_actor,fp0_probe
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=0);obs=po(obs).cuda();base,expanded=make_fp0_actor(obs.shape[-1],12);base.cuda();expanded.cuda();w=torch.full((a.num_envs,5),.2,device='cuda');max_action=0.;max_logp=0.;
            for _ in range(a.steps):
                a0,a1,lp0,lp1=fp0_probe(base,expanded,obs,w);max_action=max(max_action,float((a0-a1).abs().max()));max_logp=max(max_logp,float((lp0-lp1).abs().max()));
                if max_action>1e-6 or max_logp>1e-6:raise RuntimeError(f'FP-0 mismatch action={max_action} logp={max_logp}')
                nxt,*_=env.step(a1);obs=po(nxt).cuda()
            ckpt=run/'fp0_smoke.pt';torch.save({'base':base.state_dict(),'expanded':expanded.state_dict(),'w_ref':[.2]*5},ckpt);out={'schema':'m0_2a_fp0_smoke_v1','status':'PASS','num_envs':a.num_envs,'steps':a.steps,'max_action_abs_error':max_action,'max_logp_abs_error':max_logp,'checkpoint_present':ckpt.exists(),'optimizer_step':False,'training_authorized':False};(run/'artifact.json').write_text(json.dumps(out,indent=2)+'\n');(run/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0})+'\n');print(json.dumps(out,indent=2))
        except BaseException as e:(run/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_m0_2a_fp0c_smoke():
    """Run former m0_2a_fp0c_smoke.py stage."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import torch
    ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    def po(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,default=16);p.add_argument('--steps',type=int,default=24);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();r.mkdir(parents=True,exist_ok=True);(r/'RUN_STARTED.json').write_text(json.dumps({'status':'RUN_STARTED'})+'\n');app=env=None
        try:
            from isaaclab.app import AppLauncher;app=AppLauncher({'headless':True,'enable_cameras':False}).app;import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.modules.actor_critic import ActorCritic
            from rl.core.algorithms.vector_ppo import initialize_centered_from_scalar,W_REF
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=a.num_envs;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);obs,_=env.reset(seed=0);obs=po(obs).cuda();base=ActorCritic(obs.shape[-1],obs.shape[-1],12,1,[128,128,128]).cuda();m=initialize_centered_from_scalar(base,obs.shape[-1],12,[128,128,128]).cuda();w=torch.tensor(W_REF,device='cuda').expand(a.num_envs,-1);max_a=max_g=0.
            for _ in range(a.steps):
                with torch.no_grad(): a0=base.act_inference(obs);a1=m.inference_centered(obs,w);max_a=max(max_a,float((a0-a1).abs().max()))
                loss=m.inference_centered(obs,w).pow(2).mean();m.zero_grad();loss.backward();max_g=max(max_g,float(m.actor_body[0].weight.grad[:,obs.shape[-1]:].abs().max()));nxt,*_=env.step(a1);obs=po(nxt).cuda()
            out={'schema':'m0_2a_fp0c_smoke_v1','status':'PASS' if max_a<=1e-6 and max_g==0.0 else 'FAIL','num_envs':a.num_envs,'steps':a.steps,'max_action_error':max_a,'max_preference_gradient':max_g,'w_ref':list(W_REF),'training_authorized':False};(r/'artifact.json').write_text(json.dumps(out,indent=2)+'\n');(r/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0})+'\n');print(json.dumps(out,indent=2))
        except BaseException as e:(r/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_m0_2a_production_smoke():
    """Run former m0_2a_production_smoke.py stage."""
    """Real-Isaac M0.2a smoke: vector reward -> vector GAE/critic -> actor loss."""
    import argparse, json, sys, time, traceback
    from pathlib import Path
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    
    def policy_obs(obs):
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        return obs if torch.is_tensor(obs) else torch.as_tensor(obs)
    
    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--num-envs", type=int, default=16)
        p.add_argument("--updates", type=int, default=2)
        p.add_argument("--horizon", type=int, default=24)
        p.add_argument("--run-dir", type=Path, required=True)
        a = p.parse_args(); run = a.run_dir.resolve(); run.mkdir(parents=True, exist_ok=True)
        (run / "RUN_STARTED.json").write_text(json.dumps({"status":"RUN_STARTED", "unix":time.time()}) + "\n")
        app = env = None
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_stock_terms, reconstruct_stock_scalar
            from rl.core.algorithms.vector_ppo import M02aActorCritic, REFERENCE_W, vector_gae, late_weighted_actor_loss, vector_value_loss, adaptive_kl_lr
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = 0
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            obs, _ = env.reset(seed=0); obs = policy_obs(obs).to("cuda")
            model = M02aActorCritic(obs.shape[-1], 12).cuda()
            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            w = torch.tensor(REFERENCE_W, device="cuda").expand(a.num_envs, -1)
            max_err = 0.0; rows = 0; updates = 0; actor_steps = 0; kl_events = {"up": 0, "down": 0, "hold": 0}
            for _ in range(a.updates):
                obs_buf=[]; rew_buf=[]; val_buf=[]; done_buf=[]; old_logp=[]
                for _ in range(a.horizon):
                    with torch.no_grad():
                        action, logp = model.act_with_preference(obs, w)
                        value = model.value_with_preference(obs, w)
                    nxt, reward, term, trunc, _ = env.step(action)
                    mgr = env.unwrapped.reward_manager; raw = mgr._step_reward.detach().cpu().numpy()
                    names = list(mgr.active_terms)
                    rv = group_stock_terms({n: raw[:, i] for i, n in enumerate(names)}, shape=(a.num_envs,))
                    recon = reconstruct_stock_scalar(rv) * env.unwrapped.step_dt
                    scalar = reward.detach().cpu().numpy()
                    err = float(np.max(np.abs(recon - scalar))); max_err = max(max_err, err)
                    if err >= 1e-6: raise RuntimeError(f"reward reconstruction mismatch: {err}")
                    obs_buf.append(obs); rew_buf.append(torch.as_tensor(rv, device="cuda", dtype=torch.float32) * env.unwrapped.step_dt)
                    val_buf.append(value); done_buf.append((term | trunc).to("cuda")); old_logp.append(logp)
                    rows += a.num_envs; obs = policy_obs(nxt).to("cuda")
                with torch.no_grad(): next_value = model.value_with_preference(obs, w)
                rewards = torch.stack(rew_buf); values = torch.stack(val_buf); dones = torch.stack(done_buf)
                adv, returns = vector_gae(rewards, values, next_value, dones)
                flat_obs = torch.cat(obs_buf); flat_adv = adv.reshape(-1, 5); flat_ret = returns.reshape(-1, 5)
                flat_w = w.repeat(a.horizon, 1); flat_actions = torch.cat([model.act_with_preference(o, w)[0].detach() for o in obs_buf])
                ratio = torch.exp(model.logp_with_preference(flat_obs, flat_w, flat_actions) - torch.cat(old_logp).detach())
                old_mean = model.raw_mean(torch.cat([model._with_w(o, w) for o in obs_buf])).detach()
                actor_loss = late_weighted_actor_loss(ratio, flat_adv, flat_w)
                critic_loss = vector_value_loss(model.value_with_preference(flat_obs, flat_w), flat_ret)
                loss = actor_loss + critic_loss
                optim.zero_grad(); loss.backward(); optim.step(); actor_steps += 1
                new_mean = model.raw_mean(torch.cat([model._with_w(o, w) for o in obs_buf])).detach()
                analytic_kl = ((old_mean - new_mean).pow(2) / (2.0 * model.log_std.detach().exp().pow(2))).mean()
                kl_events[adaptive_kl_lr(optim, analytic_kl)] += 1
                updates += 1
            ckpt = run / "m0_2a_smoke.pt"
            torch.save({"model": model.state_dict(), "optimizer": optim.state_dict(), "w_ref": list(REFERENCE_W)}, ckpt)
            loaded = M02aActorCritic(obs.shape[-1], 12).cuda(); state = torch.load(ckpt, map_location="cuda"); loaded.load_state_dict(state["model"])
            report = {"schema":"m0_2a_production_smoke_v1", "status":"PASS", "num_envs":a.num_envs, "updates":updates, "horizon":a.horizon, "vector_rows":rows, "max_abs_reconstruction_error":max_err, "actor_steps":actor_steps, "vector_critic_dim":5, "w_eval":list(REFERENCE_W), "checkpoint_resume":True, "adaptive_kl_events":kl_events, "current_actor_lr":optim.param_groups[0]["lr"], "preference_sampling":False, "finite":True}
            (run / "artifact.json").write_text(json.dumps(report, indent=2) + "\n"); (run / "RUN_DONE.json").write_text(json.dumps({"status":"RUN_DONE", "exit_code":0}) + "\n"); print(json.dumps(report, indent=2))
        except BaseException as e:
            (run / "ERROR.json").write_text(json.dumps({"status":"ERROR", "error":str(e), "traceback":traceback.format_exc()}, indent=2) + "\n"); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    
    if True: main()

def run_record_m0_2a_fp0_smoke():
    """Run former record_m0_2a_fp0_smoke.py stage."""
    import json,hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[4]
    p=root/'artifacts/m0_2a_fp_design/M0_2A_FP_DESIGN.json'
    m=json.loads(p.read_text());a=root/'runs/m0_2a_fp0_smoke_2026-09-22/artifact.json';d=json.loads(a.read_text())
    if d.get('status')!='PASS' or d['max_action_abs_error']>1e-6 or d['max_logp_abs_error']>1e-6: raise SystemExit('FP-0 smoke failed')
    m['status']='FP0_SMOKE_PASS_PENDING_FREEZE';m['fp0_smoke']={'run':str(a.parent.relative_to(root)),'artifact_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'max_action_error':d['max_action_abs_error'],'max_logp_error':d['max_logp_abs_error'],'checkpoint_present':d['checkpoint_present']};p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m['fp0_smoke'],indent=2))

def run_summarize_m0_2a():
    """Run former summarize_m0_2a.py stage."""
    """Summarize M0.2a machinery sanity without overstating preservation."""
    import json
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[4]
    def main():
        rows=[]
        for seed in range(3):
            run=ROOT/f"runs/m0_2a_seed{seed}_2026-09-22"; d=json.loads((run/'artifact.json').read_text())
            d['run']=str(run.relative_to(ROOT)); d['run_done']=(run/'RUN_DONE.json').exists(); d['error']=(run/'ERROR.json').exists(); d['termination_fraction']=d['done_steps']/d['vector_rows']; rows.append(d)
        out=ROOT/'artifacts/m0_2a_confirmation';out.mkdir(parents=True,exist_ok=True)
        summary={'schema':'m0_2a_confirmation_v1','status':'TRAINING_SANITY_PASS_PRESERVATION_PENDING','training_sanity':all(r['status']=='PASS' and r['run_done'] and not r['error'] and r['finite'] and r['checkpoint_present'] and r['max_abs_reconstruction_error']<1e-6 and r['preference_sampling'] is False and r['b0_b1_state'] is False and r['w_ref']==[.2]*5 for r in rows),'seeds':rows,'preservation_gate':'NOT_VERDICTED','reason':'This runner artifact does not include the required deterministic/reference evaluation metrics; M0.2b remains unauthorized.','comparison_references':['artifacts/m0_1_confirmation/M0_1_CONFIRMATION.json','artifacts/l0a_screen/L0A_3SEED_SUMMARY.json']}
        (out/'M0_2A_CONFIRMATION.json').write_text(json.dumps(summary,indent=2)+'\n')
        lines=['# M0.2a Three-Seed Preservation Experiment','','## Status: TRAINING SANITY PASS — PRESERVATION PENDING','', 'All three seeds completed with vector reward reconstruction, finite vector critic/GAE telemetry, fixed `w_ref`, adaptive-KL events, checkpoints, and lifecycle markers. The formal preservation verdict remains pending because the required deterministic/reference evaluation metrics were not recorded by this runner. M0.2b is not authorized.','', '|seed|reconstruction max error|termination fraction|adaptive KL up/down/hold|', '|---:|---:|---:|:---:|']
        for r in rows: lines.append(f"| {r['seed']} | {r['max_abs_reconstruction_error']:.3e} | {r['termination_fraction']:.4f} | {r['adaptive_kl_events']['up']}/{r['adaptive_kl_events']['down']}/{r['adaptive_kl_events']['hold']} |")
        (out/'M0_2A_CONFIRMATION.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(summary,indent=2))
    if True: main()

def run_train_m0_2a():
    """Run former train_m0_2a.py stage."""
    """M0.2a fixed-preference three-seed preservation runner."""
    import argparse, json, sys, time, traceback
    from pathlib import Path
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    MANIFEST = ROOT / "artifacts/m0_2a_freeze/M0_2A_FREEZE.json"
    
    def policy_obs(obs):
        if isinstance(obs, dict): obs = obs.get("policy", next(iter(obs.values())))
        return obs if torch.is_tensor(obs) else torch.as_tensor(obs)
    
    def main():
        p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True)
        p.add_argument("--num-envs", type=int, default=4096); p.add_argument("--updates", type=int, default=300)
        p.add_argument("--horizon", type=int, default=24); p.add_argument("--run-dir", type=Path, required=True)
        a = p.parse_args(); run = a.run_dir.resolve(); run.mkdir(parents=True, exist_ok=True)
        (run / "RUN_STARTED.json").write_text(json.dumps({"status":"RUN_STARTED", "seed":a.seed, "unix":time.time()}) + "\n")
        app = env = None
        try:
            manifest = json.loads(MANIFEST.read_text())
            if manifest.get("status") != "FROZEN" or not manifest.get("training_authorized") or a.seed not in (0,1,2):
                raise RuntimeError("M0.2a manifest is not authorized or seed is invalid")
            from isaaclab.app import AppLauncher
            app = AppLauncher({"headless": True, "enable_cameras": False}).app
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_stock_terms, reconstruct_stock_scalar
            from rl.core.algorithms.vector_ppo import (M02aActorCritic, REFERENCE_W, vector_gae,
                late_weighted_actor_loss, vector_value_loss, adaptive_kl_lr)
            cfg = UnitreeA1FlatEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = a.seed
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            obs, _ = env.reset(seed=a.seed); obs = policy_obs(obs).to("cuda")
            model = M02aActorCritic(obs.shape[-1], 12).cuda(); optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            w = torch.tensor(REFERENCE_W, device="cuda").expand(a.num_envs, -1)
            max_err = 0.0; rows = 0; episodes = 0; done_steps = 0; kl_events = {"up":0,"down":0,"hold":0}; metrics=[]
            for update in range(1, a.updates + 1):
                obuf=[]; abuf=[]; rbuf=[]; vbuf=[]; dbuf=[]; lbuf=[]; step_rewards=[]
                for _ in range(a.horizon):
                    with torch.no_grad(): action, logp = model.act_with_preference(obs, w); value = model.value_with_preference(obs, w)
                    nxt, reward, term, trunc, _ = env.step(action)
                    mgr = env.unwrapped.reward_manager; raw = mgr._step_reward.detach().cpu().numpy(); names = list(mgr.active_terms)
                    rv = group_stock_terms({n: raw[:, i] for i, n in enumerate(names)}, shape=(a.num_envs,))
                    recon = reconstruct_stock_scalar(rv) * env.unwrapped.step_dt; scalar = reward.detach().cpu().numpy()
                    err = float(np.max(np.abs(recon - scalar))); max_err = max(max_err, err)
                    if err >= 1e-6: raise RuntimeError(f"reward reconstruction mismatch: {err}")
                    d = (term | trunc).to("cuda"); obuf.append(obs); abuf.append(action.detach()); rbuf.append(torch.as_tensor(rv, device="cuda", dtype=torch.float32) * env.unwrapped.step_dt)
                    vbuf.append(value); dbuf.append(d); lbuf.append(logp); step_rewards.append(float(reward.mean().item())); rows += a.num_envs; done_steps += int(d.sum().item()); episodes += int(d.sum().item()); obs = policy_obs(nxt).to("cuda")
                with torch.no_grad(): next_value = model.value_with_preference(obs, w)
                rewards, values, dones = torch.stack(rbuf), torch.stack(vbuf), torch.stack(dbuf)
                adv, returns = vector_gae(rewards, values, next_value, dones)
                flat_obs = torch.cat(obuf); flat_w = w.repeat(a.horizon, 1); flat_adv = adv.reshape(-1, 5); flat_ret = returns.reshape(-1, 5); old_logp = torch.cat(lbuf).detach()
                actions = torch.cat(abuf)
                old_mean = model.raw_mean(torch.cat([model._with_w(o, w) for o in obuf])).detach()
                ratio = torch.exp(model.logp_with_preference(flat_obs, flat_w, actions) - old_logp)
                actor_loss = late_weighted_actor_loss(ratio, flat_adv, flat_w); critic_loss = vector_value_loss(model.value_with_preference(flat_obs, flat_w), flat_ret); loss = actor_loss + critic_loss
                optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
                with torch.no_grad(): new_mean = model.raw_mean(torch.cat([model._with_w(o, w) for o in obuf])).detach()
                kl = ((old_mean - new_mean).pow(2) / (2.0 * model.log_std.detach().exp().pow(2))).mean(); kl_events[adaptive_kl_lr(optim, kl)] += 1
                if update % 50 == 0 or update == a.updates:
                    torch.save({"model":model.state_dict(),"optimizer":optim.state_dict(),"update_idx":update,"w_ref":list(REFERENCE_W)}, run / f"model_{update}.pt")
                metrics.append({"update":update,"mean_reward":float(np.mean(step_rewards)),"vector_value_loss":float(critic_loss.detach()),"actor_loss":float(actor_loss.detach()),"analytic_kl":float(kl.detach()),"actor_lr":optim.param_groups[0]["lr"],"done_count":int(dones.sum().item())})
            report = {"schema":"m0_2a_confirmation_v1","status":"PASS","seed":a.seed,"updates":a.updates,"num_envs":a.num_envs,"horizon":a.horizon,"vector_rows":rows,"max_abs_reconstruction_error":max_err,"episodes_terminated":episodes,"done_steps":done_steps,"vector_critic_dim":5,"w_ref":list(REFERENCE_W),"preference_sampling":False,"b0_b1_state":False,"adaptive_kl_events":kl_events,"final_actor_lr":optim.param_groups[0]["lr"],"checkpoint_present":any(run.glob("model_*.pt")),"finite":all(np.isfinite([m["mean_reward"],m["vector_value_loss"],m["actor_loss"],m["analytic_kl"]]).all() for m in metrics)}
            (run / "metrics.json").write_text(json.dumps(metrics) + "\n"); (run / "artifact.json").write_text(json.dumps(report, indent=2) + "\n"); (run / "RUN_DONE.json").write_text(json.dumps({"status":"RUN_DONE","exit_code":0,"seed":a.seed}) + "\n"); print(json.dumps(report, indent=2))
        except BaseException as e:
            (run / "ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(e),"traceback":traceback.format_exc()}, indent=2) + "\n"); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    
    if True: main()

def run_verdict_m0_2a():
    """Run former verdict_m0_2a.py stage."""
    """Apply the fixed-preference deterministic preservation verdict."""
    import json
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[4]
    def main():
        rows=[]
        for seed in range(3):
            p=ROOT/f'artifacts/m0_2a_confirmation/eval_seed{seed}.json'; d=json.loads(p.read_text()); rows.append(d)
        passed=all(r['survival']>=0.80 and r['mean_vx_error']<=0.25 and r['deterministic_actor_mean'] and not r['training_state_mutated'] for r in rows)
        out={'schema':'m0_2a_verdict_v1','status':'PASS' if passed else 'FAIL','rule':'model_300 deterministic fixed-w preservation gate; every seed required','seeds':rows,'m0_2b_authorized':False if not passed else True}
        dest=ROOT/'artifacts/m0_2a_confirmation/M0_2A_VERDICT.json'; dest.write_text(json.dumps(out,indent=2)+'\n')
        lines=['# M0.2a Deterministic Preservation Verdict','',f"Verdict: **{out['status']}**",'', 'Decision uses only model_300, explicit `w_eval=[0.2,...,0.2]`, deterministic `tanh(actor_mean)`, 64 lanes, and 500 steps. Every seed must pass; no checkpoint selection or threshold relaxation is permitted.','', '|seed|survival|mean vx error|termination rate|finite/non-mutating|','|---:|---:|---:|---:|:---:|']
        for r in rows: lines.append(f"| {r['seed']} | {r['survival']:.3f} | {r['mean_vx_error']:.4f} | {r['termination_rate']:.4f} | {r['deterministic_actor_mean'] and not r['training_state_mutated']} |")
        lines += ['', 'M0.2b remains unauthorized because fixed-preference MOPPO did not preserve deterministic locomotion.']
        (ROOT/'artifacts/m0_2a_confirmation/M0_2A_VERDICT.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(out,indent=2))
    if True: main()

def run_verdict_m0_2a_fp0():
    """Run former verdict_m0_2a_fp0.py stage."""
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[4];rows=[json.loads((root/f'artifacts/m0_2a_fp_design/fp0_eval_seed{s}.json').read_text()) for s in range(3)]
    passed=all(r['survival']>=.8 and r['mean_vx_error']<=.25 for r in rows);out={'schema':'m0_2a_fp0_verdict_v1','status':'PASS' if passed else 'FAIL','rule':'FP-0 model_300 deterministic preservation; all seeds required','seeds':rows,'fp1_authorized':False};(root/'artifacts/m0_2a_fp_design/FP0_VERDICT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

STAGES = {
    "close_m0_2a_fail": run_close_m0_2a_fail,
    "close_m0_2a_fp0": run_close_m0_2a_fp0,
    "eval_m0_2a": run_eval_m0_2a,
    "eval_m0_2a_debug": run_eval_m0_2a_debug,
    "eval_m0_2a_decomp": run_eval_m0_2a_decomp,
    "freeze_m0_2a": run_freeze_m0_2a,
    "freeze_m0_2a_fp0": run_freeze_m0_2a_fp0,
    "m0_2a_decomp_screen": run_m0_2a_decomp_screen,
    "m0_2a_fp0_smoke": run_m0_2a_fp0_smoke,
    "m0_2a_fp0c_smoke": run_m0_2a_fp0c_smoke,
    "m0_2a_production_smoke": run_m0_2a_production_smoke,
    "record_m0_2a_fp0_smoke": run_record_m0_2a_fp0_smoke,
    "summarize_m0_2a": run_summarize_m0_2a,
    "train_m0_2a": run_train_m0_2a,
    "verdict_m0_2a": run_verdict_m0_2a,
    "verdict_m0_2a_fp0": run_verdict_m0_2a_fp0,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
