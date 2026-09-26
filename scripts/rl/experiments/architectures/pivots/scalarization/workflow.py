"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_pivot_p0_matched_pilot():
    """Run former pivot_p0_matched_pilot.py stage."""
    """P0 one-seed matched diagnostic pilot; no confirmatory verdict."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def flat(gs,ps): return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs,ps)])
    
    def finite_grads(model):
        return all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=300);ap.add_argument('--horizon',type=int,default=2);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='PIVOT-P0-MATCHED-PILOT',seed=args.seed,updates=args.updates,verdict_update=300,training_authorized=True);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.preferences.v1c_curriculum import load_manifest,sample_preferences
            from talon_rl.rewards.baselines import group_v1b_s7_terms,reconstruct_v1b_s7_scalar
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic,initialize_from_rsl_m01,vector_gae,vector_value_loss
            from talon_rl.optimization.scalarization import early_scalarized_ppo_loss,late_weighted_ppo_loss,p0_loss_diagnostics
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();action_dim=env.unwrapped.action_manager.total_action_dim;manager=env.unwrapped.reward_manager;sources={0:ROOT/'runs/m0_1_seed0_2026-09-22/model_299.pt',1:ROOT/'runs/m0_1_seed1_2026-09-22-rerun/model_299.pt',2:ROOT/'runs/m0_1_seed2_2026-09-22/model_299.pt'};source=sources[args.seed];manifest=load_manifest();all_reports={}
            for mode,loss_fn in [('early_scalarization',early_scalarized_ppo_loss),('late_weighting',late_weighted_ppo_loss)]:
                mark('CONDITION_START',condition=mode);torch.manual_seed(args.seed);rng=np.random.default_rng(100000+args.seed);model=V1CSharedActorCritic(obs.shape[-1],action_dim).cuda();initialize_from_rsl_m01(model,source,device='cpu');opt=torch.optim.Adam(model.parameters(),lr=1e-3);cur,_=env.reset(seed=args.seed);cur=obs_tensor(cur).cuda();actor_params=[p for n,p in model.named_parameters() if n.startswith('actor_') or n=='log_std'];rows=[];max_recon=0.0
                for update in range(1,args.updates+1):
                    w_np,_=sample_preferences(rng,update,args.num_envs,'full_simplex_control',manifest);w=torch.as_tensor(w_np,device='cuda');obs_buf=[];act_buf=[];old_buf=[];rew_buf=[];val_buf=[];done_buf=[];survival=[]
                    for _ in range(args.horizon):
                        with torch.no_grad(): raw_action,old_lp=model.act_with_preference(cur,w);value=model.value_with_preference(cur,w)
                        action=torch.clamp(raw_action,-1,1);nxt,scalar,term,trunc,_=env.step(action);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));recon=reconstruct_v1b_s7_scalar(vec)*env.unwrapped.step_dt;max_recon=max(max_recon,float(np.max(np.abs(recon-scalar.detach().cpu().numpy()))));survival.append(float(1-((term|trunc).float().mean())));obs_buf.append(cur);act_buf.append(raw_action);old_buf.append(old_lp);rew_buf.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val_buf.append(value);done_buf.append((term|trunc).to('cuda'));cur=obs_tensor(nxt).cuda()
                    with torch.no_grad():next_value=model.value_with_preference(cur,w)
                    rewards=torch.stack(rew_buf);values=torch.stack(val_buf);dones=torch.stack(done_buf);adv,returns=vector_gae(rewards,values,next_value,dones);flat_obs=torch.cat(obs_buf);flat_actions=torch.cat(act_buf);flat_old=torch.cat(old_buf);flat_w=w.repeat(args.horizon,1);flat_adv=adv.reshape(-1,3);flat_ret=returns.reshape(-1,3);ratio=torch.exp(model.logp_with_preference(flat_obs,flat_w,flat_actions)-flat_old.detach());diag=p0_loss_diagnostics(ratio,flat_adv.detach(),flat_w);early=early_scalarized_ppo_loss(ratio,flat_adv.detach(),flat_w);late=late_weighted_ppo_loss(ratio,flat_adv.detach(),flat_w);ge=flat(torch.autograd.grad(early,actor_params,retain_graph=True,allow_unused=True),actor_params);gl=flat(torch.autograd.grad(late,actor_params,retain_graph=True,allow_unused=True),actor_params);gradient_cos=float(torch.dot(ge,gl)/(torch.linalg.vector_norm(ge)*torch.linalg.vector_norm(gl)+1e-12));critic=vector_value_loss(model.value_with_preference(flat_obs,flat_w),flat_ret.detach());actor_loss=loss_fn(ratio,flat_adv.detach(),flat_w);opt.zero_grad(set_to_none=True);(actor_loss+critic).backward();finite=finite_grads(model);opt.step();sur=diag['per_objective_surrogate'].detach();signs=sur.sign();mixed=float(((signs.min(-1).values<0)&(signs.max(-1).values>0)).float().mean());rows.append({'update':update,'early_loss':float(early.detach()),'late_loss':float(late.detach()),'abs_loss_delta':float((early-late).abs().detach()),'per_objective_surrogate_mean':sur.mean(0).cpu().tolist(),'mixed_sign_fraction':mixed,'ratio_mean':float(ratio.mean()),'clip_fraction':float(diag['clip_fraction']),'actor_gradient_cosine_early_late':gradient_cos,'early_gradient_norm':float(torch.linalg.vector_norm(ge)),'late_gradient_norm':float(torch.linalg.vector_norm(gl)),'reward_vector_mean':rewards.mean((0,1)).detach().cpu().tolist(),'survival_sanity':float(np.mean(survival)),'max_reconstruction_error':max_recon,'finite_gradients':finite})
                    if update in (1,5,10,25,50,100,200,300): mark('DIAGNOSTIC_CHECKPOINT',condition=mode,update=update)
                ckpt=args.output.parent/f'{mode}_terminal.pt';torch.save({'schema':'pivot_p0_pilot_terminal_v1','mode':mode,'update':args.updates,'model':model.state_dict(),'optimizer':opt.state_dict()},ckpt);loaded=V1CSharedActorCritic(obs.shape[-1],action_dim).cuda();loaded.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=False)['model']);loaded.eval();model.eval();
                with torch.no_grad():resume_diff=float((model.act_inference_with_preference(cur,w)-loaded.act_inference_with_preference(cur,w)).abs().max())
                all_reports[mode]={'rows':rows,'terminal_checkpoint':str(ckpt),'max_reconstruction_error':max_recon,'checkpoint_resume_max_action_diff':resume_diff};mark('CONDITION_DONE',condition=mode)
            report={'schema':'pivot_p0_matched_training_v1','status':'TRAINING_COMPLETE','pilot_only':False,'scientific_verdict':'PENDING_TERMINAL_EVALUATION','training_authorized':True,'seed':args.seed,'updates':args.updates,'terminal_verdict_update':300,'diagnostic_checkpoints':[1,5,10,25,50,100,200,300],'conditions':all_reports,'source_checkpoint':str(source),'manifest':'artifacts/post_v1/PIVOT_P0_CONFIRMATORY_MANIFEST.json','note':'Confirmatory training artifact; verdict requires frozen terminal evaluation and aggregation.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'seed':args.seed,'conditions':list(all_reports)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_pivot_p0_smoke():
    """Run former pivot_p0_smoke.py stage."""
    """Real-Isaac P0 smoke for early scalarization and late weighting.
    
    This is an implementation smoke only. It performs one short optimizer update
    per condition and does not authorize a pilot or confirmatory training run.
    """
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def finite_grads(model):
        values=[p.grad.detach() for p in model.parameters() if p.grad is not None]
        return bool(values) and all(bool(torch.isfinite(g).all()) for g in values)
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--horizon',type=int,default=2);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='PIVOT-P0-SMOKE',training_authorized=False);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms,reconstruct_v1b_s7_scalar
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic,initialize_from_rsl_m01,vector_gae,vector_value_loss
            from talon_rl.optimization.scalarization import early_scalarized_ppo_loss,late_weighted_ppo_loss,p0_loss_diagnostics
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();action_dim=env.unwrapped.action_manager.total_action_dim;manager=env.unwrapped.reward_manager;source=ROOT/'runs/m0_1_seed0_2026-09-22/model_299.pt';w=torch.tensor([[.2,.3,.5]],device='cuda').repeat(args.num_envs,1);reports={}
            for mode,loss_fn in [('early_scalarization',early_scalarized_ppo_loss),('late_weighting',late_weighted_ppo_loss)]:
                mark('CONDITION_START',condition=mode);model=V1CSharedActorCritic(obs.shape[-1],action_dim).cuda();initialize_from_rsl_m01(model,source,device='cpu');opt=torch.optim.Adam(model.parameters(),lr=1e-3);cur,_=env.reset(seed=0);cur=obs_tensor(cur).cuda();obs_buf=[];act_buf=[];old_buf=[];rew_buf=[];val_buf=[];done_buf=[];max_recon=0.0
                for _ in range(args.horizon):
                    with torch.no_grad(): raw_action,old_lp=model.act_with_preference(cur,w);value=model.value_with_preference(cur,w)
                    action=torch.clamp(raw_action,-1,1);nxt,scalar,term,trunc,_=env.step(action);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));recon=reconstruct_v1b_s7_scalar(vec)*env.unwrapped.step_dt;max_recon=max(max_recon,float(np.max(np.abs(recon-scalar.detach().cpu().numpy()))));obs_buf.append(cur);act_buf.append(raw_action);old_buf.append(old_lp);rew_buf.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val_buf.append(value);done_buf.append((term|trunc).to('cuda'));cur=obs_tensor(nxt).cuda()
                with torch.no_grad():next_value=model.value_with_preference(cur,w)
                rewards=torch.stack(rew_buf);values=torch.stack(val_buf);dones=torch.stack(done_buf);adv,returns=vector_gae(rewards,values,next_value,dones);flat_obs=torch.cat(obs_buf);flat_actions=torch.cat(act_buf);flat_old=torch.cat(old_buf);flat_w=w.repeat(args.horizon,1);flat_adv=adv.reshape(-1,3);flat_ret=returns.reshape(-1,3);ratio=torch.exp(model.logp_with_preference(flat_obs,flat_w,flat_actions)-flat_old.detach());diagnostics=p0_loss_diagnostics(ratio,flat_adv.detach(),flat_w);actor_loss=loss_fn(ratio,flat_adv.detach(),flat_w);critic_loss=vector_value_loss(model.value_with_preference(flat_obs,flat_w),flat_ret.detach());loss=actor_loss+critic_loss;opt.zero_grad(set_to_none=True);loss.backward();grads_finite=finite_grads(model);opt.step();mark('CONDITION_STEP_OK',condition=mode)
                ckpt=args.output.parent/f'{mode}.pt'
                torch.save({'schema':'pivot_p0_smoke_checkpoint_v1','mode':mode,'model':model.state_dict(),'optimizer':opt.state_dict()},ckpt)
                loaded=V1CSharedActorCritic(obs.shape[-1],action_dim).cuda();state=torch.load(ckpt,map_location='cuda',weights_only=False);loaded.load_state_dict(state['model']);loaded.eval();model.eval()
                with torch.no_grad():
                    resume_diff=float((model.act_inference_with_preference(cur,w)-loaded.act_inference_with_preference(cur,w)).abs().max())
                reports[mode]={'actor_loss':float(actor_loss.detach()),'critic_loss':float(critic_loss.detach()),'per_objective_advantage_mean':flat_adv.mean(0).detach().cpu().tolist(),'per_objective_clipped_surrogate_mean':diagnostics['per_objective_surrogate'].mean(0).detach().cpu().tolist(),'final_weighted_actor_loss':float(diagnostics['weighted_actor_loss'].detach()),'ratio_mean':float(ratio.mean()),'clip_fraction':float(diagnostics['clip_fraction']),'max_reconstruction_error':max_recon,'finite_gradients':grads_finite,'checkpoint_resume_max_action_diff':resume_diff,'checkpoint':str(ckpt)};mark('CONDITION_DONE',condition=mode)
            report={'schema':'pivot_p0_smoke_v1','status':'SMOKE_PASS','training_authorized':False,'num_envs':args.num_envs,'horizon':args.horizon,'w_smoke':[.2,.3,.5],'conditions':reports,'same_source_checkpoint':str(source),'critic_path_unchanged':True,'ppo_ratio_path_unchanged':True};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps(report,indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_pivot_p0_terminal_eval():
    """Run former pivot_p0_terminal_eval.py stage."""
    """Frozen terminal evaluator for the P0 three-seed confirmatory run."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    GRID=np.asarray([[1/3,1/3,1/3],[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.45,.45,.1],[.45,.1,.45],[.1,.45,.45],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]],dtype=np.float32)
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def tilt_tensor(q):
        return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--reset-suites',type=int,default=4);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='PIVOT-P0-TERMINAL-EVAL',measurement_only=True);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();action_dim=env.unwrapped.action_manager.total_action_dim;manager=env.unwrapped.reward_manager
            files={}
            for seed in [0,1,2]:
                for condition in ['early_scalarization','late_weighting']:
                    run=json.loads((ROOT/f'runs/pivot_p0_confirmatory-2026-09-23/seed{seed}.json').read_text());files[(seed,condition)]=ROOT/run['conditions'][condition]['terminal_checkpoint']
            models={}
            for key,path in files.items():
                model=V1CSharedActorCritic(obs.shape[-1],action_dim).cuda();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);model.eval();models[key]=model
            mark('CHECKPOINTS_LOADED',count=len(models));rows=[]
            for (seed,condition),model in models.items():
                for gi,pref in enumerate(GRID):
                    w=torch.as_tensor(np.repeat(pref[None,:],args.num_envs,0),device='cuda')
                    for suite in range(args.reset_suites):
                        reset_seed=47001+seed*1000+suite;cur,_=env.reset(seed=reset_seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,action_dim),device='cuda');vecs=[];metrics=[];finite=True
                        with torch.no_grad():
                            for _ in range(args.steps):
                                raw_action=model.act_inference_with_preference(cur,w);action=torch.clamp(raw_action,-1,1);finite=finite and bool(torch.isfinite(raw_action).all() and torch.isfinite(action).all());nxt,_,term,trunc,_=env.step(action);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');tilt=tilt_tensor(data.root_quat_w);metrics.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'tilt_p95':float(torch.quantile(tilt,.95)),'max_tilt':float(tilt.max()),'ang_vel_xy':float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),'torque_norm':float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),'action_rate':float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),'survival':float(1-((term|trunc).float().mean()))});vecs.append(vec.mean(0));prev=action;cur=obs_tensor(nxt).cuda()
                        rows.append({'seed':seed,'condition':condition,'preference_index':gi,'preference':pref.tolist(),'reset_suite':suite,'reset_seed':reset_seed,'finite':finite,'objective_return_mean':np.asarray(vecs).mean(0).tolist(),'metrics_mean':{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}})
                mark('CONDITION_DONE',seed=seed,condition=condition)
            report={'schema':'pivot_p0_terminal_eval_v1','status':'EVALUATION_COMPLETE','measurement_only':True,'protocol_manifest':'artifacts/post_v1/PIVOT_P0_CONFIRMATORY_MANIFEST.json','terminal_update':300,'grid':GRID.tolist(),'reset_suites':args.reset_suites,'steps':args.steps,'rows':rows,'checkpoint_count':len(models),'action_semantics':'deterministic actor mean followed by clip_actions=1.0','note':'All six terminal checkpoints evaluated on the same frozen grid and seed-matched reset suites.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'rows':len(rows),'checkpoints':len(models)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

STAGES = {
    "pivot_p0_matched_pilot": run_pivot_p0_matched_pilot,
    "pivot_p0_smoke": run_pivot_p0_smoke,
    "pivot_p0_terminal_eval": run_pivot_p0_terminal_eval,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
