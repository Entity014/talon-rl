"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_pivot_p1_pilot():
    """Run former pivot_p1_pilot.py stage."""
    """One-seed P1 pilot: scalar versus objective-separated value/GAE paths."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def flat(gs,ps): return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs,ps)])
    def finite_grads(m): return all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=100);ap.add_argument('--horizon',type=int,default=2);ap.add_argument('--num-envs',type=int,default=8);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='PIVOT-P1-PILOT',seed=0,updates=args.updates,verdict_update=100,training_authorized=True);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.preferences.v1c_curriculum import load_manifest,sample_preferences
            from talon_rl.rewards.baselines import group_v1b_s7_terms,reconstruct_v1b_s7_scalar
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic,initialize_from_rsl_m01,vector_gae
            from talon_rl.optimization.scalar_critic import P1ScalarCriticActorCritic,initialize_p1_scalar_from_rsl
            from talon_rl.optimization.scalar_critic import scalar_gae,normalize_final_advantage,scalar_ppo_loss
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;manager=env.unwrapped.reward_manager;source=ROOT/'runs/m0_1_seed0_2026-09-22/model_299.pt';manifest=load_manifest();all_reports={}
            for mode in ['scalar_control','vector_treatment']:
                mark('CONDITION_START',condition=mode);torch.manual_seed(0);rng=np.random.default_rng(100000);model=(P1ScalarCriticActorCritic(obs.shape[-1],ad).cuda() if mode=='scalar_control' else V1CSharedActorCritic(obs.shape[-1],ad).cuda());(initialize_p1_scalar_from_rsl(model,source) if mode=='scalar_control' else initialize_from_rsl_m01(model,source,device='cpu'));opt=torch.optim.Adam(model.parameters(),lr=1e-3);cur,_=env.reset(seed=0);cur=obs_tensor(cur).cuda();params=[p for n,p in model.named_parameters() if n.startswith('actor_') or n=='log_std'];rows=[];max_recon=0.0
                for update in range(1,args.updates+1):
                    w_np,_=sample_preferences(rng,update,args.num_envs,'full_simplex_control',manifest);w=torch.as_tensor(w_np,device='cuda');ob=[];act=[];old=[];rew=[];val=[];done=[];surv=[]
                    for _ in range(args.horizon):
                        with torch.no_grad():raw_action,old_lp=model.act_with_preference(cur,w);value=model.value_with_preference(cur,w)
                        action=torch.clamp(raw_action,-1,1);nxt,scalar,term,trunc,_=env.step(action);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));max_recon=max(max_recon,float(np.max(np.abs(reconstruct_v1b_s7_scalar(vec)*env.unwrapped.step_dt-scalar.detach().cpu().numpy()))));surv.append(float(1-((term|trunc).float().mean())));ob.append(cur);act.append(raw_action);old.append(old_lp);rew.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val.append(value);done.append((term|trunc).to('cuda'));cur=obs_tensor(nxt).cuda()
                    with torch.no_grad():nv=model.value_with_preference(cur,w)
                    rewards=torch.stack(rew);values=torch.stack(val);dones=torch.stack(done);fo=torch.cat(ob);fa=torch.cat(act);fold=torch.cat(old);fw=w.repeat(args.horizon,1);ratio=torch.exp(model.logp_with_preference(fo,fw,fa)-fold.detach());
                    if mode=='scalar_control':
                        scalar_r=(rewards*fw.reshape(args.horizon,args.num_envs,3)).sum(-1);raw_a,returns=scalar_gae(scalar_r,values.squeeze(-1),nv.squeeze(-1),dones);aw=normalize_final_advantage(raw_a);critic=(values.squeeze(-1)-returns).pow(2).mean();corr=1.0;diff_mean=0.0;diff_max=0.0;grad_cos=None;grad_a=None;grad_w=None
                    else:
                        raw_vec,returns=vector_gae(rewards,values,nv,dones);raw_aw=(raw_vec*fw.reshape(args.horizon,args.num_envs,3)).sum(-1);scalar_v=(values*fw.reshape(args.horizon,args.num_envs,3)).sum(-1);scalar_n=(nv*w).sum(-1);scalar_r=(rewards*fw.reshape(args.horizon,args.num_envs,3)).sum(-1);raw_scalar,_=scalar_gae(scalar_r,scalar_v,scalar_n,dones);aw=normalize_final_advantage(raw_aw);a_scalar=normalize_final_advantage(raw_scalar);critic=(values-returns).pow(2).mean();x=raw_scalar.reshape(-1);y=raw_aw.reshape(-1);corr=float(torch.corrcoef(torch.stack([x,y]))[0,1]) if x.std()>1e-8 and y.std()>1e-8 else 1.0;diff_mean=float((x-y).abs().mean());diff_max=float((x-y).abs().max());g1=flat(torch.autograd.grad(scalar_ppo_loss(ratio,a_scalar.reshape(-1)),params,retain_graph=True,allow_unused=True),params);g2=flat(torch.autograd.grad(scalar_ppo_loss(ratio,aw.reshape(-1)),params,retain_graph=True,allow_unused=True),params);grad_cos=float(torch.dot(g1,g2)/(torch.linalg.vector_norm(g1)*torch.linalg.vector_norm(g2)+1e-12));grad_a=float(torch.linalg.vector_norm(g1));grad_w=float(torch.linalg.vector_norm(g2))
                    actor=scalar_ppo_loss(ratio,aw.reshape(-1));new_lp=model.logp_with_preference(fo,fw,fa);approx_kl=float((fold.detach()-new_lp.detach()).mean());clip_fraction=float(((ratio-ratio.clamp(.8,1.2)).abs()>0).float().mean());opt.zero_grad(set_to_none=True);(actor+critic).backward();finite=finite_grads(model);opt.step();rows.append({'update':update,'adv_correlation':corr,'adv_abs_diff_mean':diff_mean,'adv_abs_diff_max':diff_max,'actor_gradient_cosine_scalar_vs_decomposed':grad_cos,'scalar_gradient_norm':grad_a,'decomposed_gradient_norm':grad_w,'approx_kl':approx_kl,'clip_fraction':clip_fraction,'critic_loss':float(critic.detach()),'actor_loss':float(actor.detach()),'reward_vector_mean':rewards.mean((0,1)).detach().cpu().tolist(),'survival_sanity':float(np.mean(surv)),'max_reconstruction_error':max_recon,'finite_gradients':finite});
                    if update in (25,50,75,100):mark('DIAGNOSTIC_CHECKPOINT',condition=mode,update=update)
                ckpt=args.output.parent/f'{mode}_terminal.pt';torch.save({'schema':'pivot_p1_pilot_terminal_v1','mode':mode,'update':args.updates,'model':model.state_dict(),'optimizer':opt.state_dict()},ckpt);all_reports[mode]={'rows':rows,'terminal_checkpoint':str(ckpt),'max_reconstruction_error':max_recon,'checkpoint_resume_required':True};mark('CONDITION_DONE',condition=mode)
            report={'schema':'pivot_p1_pilot_v1','status':'PILOT_COMPLETE','pilot_only':True,'scientific_verdict':'PENDING','training_authorized':True,'seed':0,'updates':args.updates,'verdict_update':100,'diagnostic_checkpoints':[25,50,75,100],'conditions':all_reports,'manifest':'artifacts/post_v1/PIVOT_P1_VALUE_ADVANTAGE_MANIFEST.json','note':'One-seed diagnostic pilot; no superiority claim.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'conditions':list(all_reports)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_pivot_p1_smoke():
    """Run former pivot_p1_smoke.py stage."""
    """Real-Isaac P1 scalar-vs-vector value/advantage smoke."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def finite_grads(m): return all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--horizon',type=int,default=2);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='PIVOT-P1-SMOKE',training_authorized=False);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms,reconstruct_v1b_s7_scalar
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic,initialize_from_rsl_m01,vector_gae
            from talon_rl.optimization.scalar_critic import P1ScalarCriticActorCritic,initialize_p1_scalar_from_rsl
            from talon_rl.optimization.scalar_critic import scalar_gae,normalize_final_advantage,scalar_ppo_loss
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;manager=env.unwrapped.reward_manager;source=ROOT/'runs/m0_1_seed0_2026-09-22/model_299.pt';w=torch.tensor([[.2,.3,.5]],device='cuda').repeat(args.num_envs,1);reports={}
            for mode in ['scalar_control','vector_treatment']:
                mark('CONDITION_START',condition=mode);model=(P1ScalarCriticActorCritic(obs.shape[-1],ad).cuda() if mode=='scalar_control' else V1CSharedActorCritic(obs.shape[-1],ad).cuda());(initialize_p1_scalar_from_rsl(model,source) if mode=='scalar_control' else initialize_from_rsl_m01(model,source,device='cpu'));opt=torch.optim.Adam(model.parameters(),lr=1e-3);cur,_=env.reset(seed=0);cur=obs_tensor(cur).cuda();ob=[];ac=[];old=[];rv=[];vals=[];done=[];max_recon=0.0
                for _ in range(args.horizon):
                    with torch.no_grad():raw_action,lp=model.act_with_preference(cur,w);value=model.value_with_preference(cur,w)
                    action=torch.clamp(raw_action,-1,1);nxt,scalar,term,trunc,_=env.step(action);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));max_recon=max(max_recon,float(np.max(np.abs(reconstruct_v1b_s7_scalar(vec)*env.unwrapped.step_dt-scalar.detach().cpu().numpy()))));ob.append(cur);ac.append(raw_action);old.append(lp);rv.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);vals.append(value);done.append((term|trunc).to('cuda'));cur=obs_tensor(nxt).cuda()
                with torch.no_grad():nv=model.value_with_preference(cur,w)
                rewards=torch.stack(rv);values=torch.stack(vals);dones=torch.stack(done);fo=torch.cat(ob);fw=w.repeat(args.horizon,1);fa=torch.cat(ac);ratio=torch.exp(model.logp_with_preference(fo,fw,fa)-torch.cat(old).detach())
                if mode=='scalar_control':
                    scalar_reward=(rewards*fw.reshape(args.horizon,args.num_envs,3)).sum(-1);scalar_value=values.squeeze(-1);scalar_next=nv.squeeze(-1);raw_adv,returns=scalar_gae(scalar_reward,scalar_value,scalar_next,dones);adv=normalize_final_advantage(raw_adv);critic=(scalar_value-returns).pow(2).mean();
                else:
                    raw_vec,returns=vector_gae(rewards,values,nv,dones);raw_adv=(raw_vec*fw.reshape(args.horizon,args.num_envs,3)).sum(-1);adv=normalize_final_advantage(raw_adv);critic=(values-returns).pow(2).mean()
                actor=scalar_ppo_loss(ratio,adv.reshape(-1));loss=actor+critic;opt.zero_grad(set_to_none=True);loss.backward();finite=finite_grads(model);opt.step();ckpt=args.output.parent/f'{mode}.pt';torch.save({'schema':'pivot_p1_smoke_checkpoint_v1','mode':mode,'model':model.state_dict(),'optimizer':opt.state_dict()},ckpt);reports[mode]={'actor_loss':float(actor.detach()),'critic_loss':float(critic.detach()),'final_adv_mean':float(adv.mean()),'final_adv_std':float(adv.std(unbiased=False)),'ratio_mean':float(ratio.mean()),'clip_fraction':float(((ratio-ratio.clamp(.8,1.2)).abs()>0).float().mean()),'max_reconstruction_error':max_recon,'finite_gradients':finite,'checkpoint':str(ckpt)};mark('CONDITION_DONE',condition=mode)
            report={'schema':'pivot_p1_smoke_v1','status':'SMOKE_PASS','training_authorized':False,'conditions':reports,'same_source_checkpoint':str(source),'note':'P1 smoke validates scalar versus objective-separated value/GAE paths; no pilot verdict.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps(report,indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

STAGES = {
    "pivot_p1_pilot": run_pivot_p1_pilot,
    "pivot_p1_smoke": run_pivot_p1_smoke,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
