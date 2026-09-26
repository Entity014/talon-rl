"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_post_p1_q0_policy_family_audit():
    """Run former post_p1_q0_policy_family_audit.py stage."""
    """Q0 read-only policy-family and trajectory-feasibility audit."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt(q):return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    def interp(states,weights):
        out={}
        for k in states[0]:
            vals=[s[k] for s in states]
            out[k]=sum(float(w)*v for w,v in zip(weights,vals)) if torch.is_floating_point(vals[0]) else vals[0]
        return out
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--reset-suites',type=int,default=4);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-P1-Q0',measurement_only=True);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;manager=env.unwrapped.reward_manager
            labels=list(PREFS);states=[];base_models={}
            for label in labels:
                path=ROOT/f'runs/post_v1_d1-2026-09-22/specialist_{label}_terminal.pt';payload=torch.load(path,map_location='cuda',weights_only=False);m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(payload['model']);m.eval();base_models[label]=m;states.append({k:v.detach().clone() for k,v in m.state_dict().items()})
            mark('CHECKPOINTS_LOADED',count=3)
            combos={'P':([1,0,0],labels[0]),'B':([0,1,0],labels[1]),'E':([0,0,1],labels[2]),'P_B_mid':([.5,.5,0],None),'P_E_mid':([.5,0,.5],None),'B_E_mid':([0,.5,.5],None),'center_mean':([1/3,1/3,1/3],None)}
            models={}
            for name,(weights,_) in combos.items():
                m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(interp(states,weights));m.eval();models[name]=m
            param_rows=[];latent_records=[]
            for name,m in models.items():
                # Parameter interpolation is evaluated at the common reference
                # preference.  For the latent/trajectory audit, the three original
                # specialists are also evaluated at their own fixed preferences.
                pref_center=torch.full((args.num_envs,3),1/3,device='cuda')
                pref_own=torch.as_tensor(np.repeat(PREFS[name][None,:],args.num_envs,axis=0),device='cuda') if name in PREFS else pref_center
                for suite in range(args.reset_suites):
                    cur,_=env.reset(seed=47001+suite);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device='cuda');vecs=[];ms=[];actions=[];finite=True
                    with torch.no_grad():
                        for _ in range(args.steps):
                            action=torch.clamp(m.act_inference_with_preference(cur,pref_own),-1,1)
                            finite=finite and bool(torch.isfinite(action).all())
                            nxt,_,term,trunc,_=env.step(action)
                            raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                            vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                            data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');t=tilt(data.root_quat_w)
                            ms.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'tilt_p95':float(torch.quantile(t,.95)),'torque_norm':float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),'action_rate':float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),'survival':float(1-((term|trunc).float().mean()))})
                            vecs.append(vec.mean(0));actions.append(action.detach().cpu().numpy());prev=action;cur=obs_tensor(nxt).cuda()
                    metrics={k:float(np.mean([x[k] for x in ms])) for k in ms[0]};param_rows.append({'policy':name,'weights':combos[name][0],'reset_suite':suite,'finite':finite,'objective_return_mean':np.asarray(vecs).mean(0).tolist(),'metrics_mean':metrics});arr=np.concatenate(actions,axis=0);feature=np.concatenate([arr.mean(0),arr.std(0),np.asarray(vecs).mean(0),np.asarray(list(metrics.values()))]).astype(np.float64)
                    if name in labels:
                        latent_records.append({'policy':name,'reset_suite':suite,'feature':feature.tolist()})
            # Specialist trajectory separability: standardised feature PCA and
            # leave-one-suite-out nearest-centroid accuracy.  This avoids the
            # invalid in-sample accuracy of using each point to define its own
            # centroid and keeps the audit tied to the three D1 endpoints.
            raw_latent=np.asarray([r['feature'] for r in latent_records],dtype=np.float64)
            X=(raw_latent-raw_latent.mean(0))/(raw_latent.std(0)+1e-8)
            _,s,_=np.linalg.svd(X-X.mean(0),full_matrices=False);den=float((s*s).sum());pca_var=((s*s/den) if den else np.zeros_like(s)).tolist()
            nearest=[]
            for held_out in range(args.reset_suites):
                train=[r for r in latent_records if r['reset_suite']!=held_out]
                test=[r for r in latent_records if r['reset_suite']==held_out]
                cent={label:np.mean([np.asarray(r['feature']) for r in train if r['policy']==label],axis=0) for label in labels}
                for r in test:
                    x=np.asarray(r['feature']);pred=min(labels,key=lambda k:float(np.linalg.norm((x-raw_latent.mean(0))/(raw_latent.std(0)+1e-8)-(cent[k]-raw_latent.mean(0))/(raw_latent.std(0)+1e-8))))
                    nearest.append({'held_out_suite':held_out,'policy':r['policy'],'predicted':pred,'correct':pred==r['policy']})
            report={'schema':'post_p1_q0_policy_family_audit_v2','status':'MEASUREMENT_COMPLETE','measurement_only':True,'d1_checkpoints':{k:f'runs/post_v1_d1-2026-09-22/specialist_{k}_terminal.pt' for k in labels},'parameter_interpolation':param_rows,'trajectory_latent':{'sample_count':len(latent_records),'pca_explained_variance':pca_var,'leave_one_suite_out_nearest_centroid_accuracy':float(np.mean([r['correct'] for r in nearest])) if nearest else None,'classification_rows':nearest,'feature_definition':'own-preference specialist action mean/std, objective return mean, vx error, tilt p95, torque norm, action rate, survival'},'interpolation_policy_names':list(combos),'reset_suites':args.reset_suites,'steps':args.steps,'note':'Read-only feasibility audit; no parameters updated and no pivot selected automatically.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'rows':len(param_rows),'pca_variance':pca_var[:3]},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_p1_q1_semantic_interpolation_audit():
    """Run former post_p1_q1_semantic_interpolation_audit.py stage."""
    """Q1 read-only action-level semantic interpolation audit."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    AXES={'P_to_B':'B','P_to_E':'E'}; ALPHAS=[0.0,.25,.5,.75,1.0]
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def tilt_deg(q): return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
    def alpha_between(values,tol=1e-8):
        vals=np.asarray(values,dtype=float); lo=min(vals[0],vals[-1])-tol; hi=max(vals[0],vals[-1])+tol
        return float(np.mean([(lo<=x<=hi) for x in vals[1:-1]]))
    def abs_monotone(values,tol=1e-8):
        d=np.diff(np.asarray(values,dtype=float)); d[np.abs(d)<=tol]=0; nz=d[d!=0]
        return float(np.all(nz>=0) or np.all(nz<=0)) if len(nz) else 1.0
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--reset-suites',type=int,default=4);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(event,**extra):
            with life.open('a') as f:f.write(json.dumps({'event':event,'unix':time.time(),**extra},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-P1-Q1',measurement_only=True,interpolation_mode='action_level');app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=47001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene['robot']
            models={}
            for label in PREFS:
                payload=torch.load(ROOT/f'runs/post_v1_d1-2026-09-22/specialist_{label}_terminal.pt',map_location='cuda',weights_only=False);m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(payload['model']);m.eval();models[label]=m
            mark('CHECKPOINTS_LOADED',count=3);weights={k:torch.as_tensor(np.repeat(v[None,:],args.num_envs,axis=0),device='cuda') for k,v in PREFS.items()};rows=[]
            for axis,target in AXES.items():
                for suite in range(args.reset_suites):
                    for alpha in ALPHAS:
                        cur,_=env.reset(seed=47001+suite);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device='cuda');vecs=[];metrics=[];finite=True
                        with torch.no_grad():
                            for _ in range(args.steps):
                                ap=torch.clamp(models['P'].act_inference_with_preference(cur,weights['P']),-1,1);at=torch.clamp(models[target].act_inference_with_preference(cur,weights[target]),-1,1);action=torch.clamp((1-alpha)*ap+alpha*at,-1,1);finite=finite and bool(torch.isfinite(action).all());nxt,_,term,trunc,_=env.step(action);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));data=robot.data;cmd=env.unwrapped.command_manager.get_command('base_velocity');tilt=tilt_deg(data.root_quat_w);vecs.append(vec.mean(0));metrics.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'tilt_p95':float(torch.quantile(tilt,.95)),'ang_vel_xy':float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),'torque_norm':float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),'action_rate':float(torch.linalg.vector_norm(action-prev,dim=-1).mean()),'survival':float(1-((term|trunc).float().mean())),'termination_rate':float((term|trunc).float().mean())});prev=action;cur=obs_tensor(nxt).cuda()
                        rows.append({'axis':axis,'target':target,'reset_suite':suite,'alpha':alpha,'finite':finite,'objective_return':np.mean(vecs,axis=0).tolist(),'metrics':{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}})
            profiles={}
            for axis in AXES:
                profiles[axis]={}
                for metric in ['vx_error','tilt_p95','ang_vel_xy','torque_norm','action_rate','survival','termination_rate']:
                    profiles[axis][metric]={'suite_profiles':[]}
                    for suite in range(args.reset_suites):
                        rs=sorted([r for r in rows if r['axis']==axis and r['reset_suite']==suite],key=lambda r:r['alpha']);vals=[r['metrics'][metric] for r in rs];profiles[axis][metric]['suite_profiles'].append({'values':vals,'endpoint_between_fraction':alpha_between(vals),'absolute_monotonicity':abs_monotone(vals)})
                for j,obj in enumerate(['progress','balance','efficiency']):
                    key=f'objective_{obj}';profiles[axis][key]={'suite_profiles':[]}
                    for suite in range(args.reset_suites):
                        rs=sorted([r for r in rows if r['axis']==axis and r['reset_suite']==suite],key=lambda r:r['alpha']);vals=[r['objective_return'][j] for r in rs];profiles[axis][key]['suite_profiles'].append({'values':vals,'endpoint_between_fraction':alpha_between(vals),'absolute_monotonicity':abs_monotone(vals)})
            report={'schema':'post_p1_q1_semantic_interpolation_audit_v1','status':'MEASUREMENT_COMPLETE','measurement_only':True,'manifest':'artifacts/post_v1/Q1_SEMANTIC_INTERPOLATION_MANIFEST.json','rows':rows,'profiles':profiles,'alphas':ALPHAS,'axes':AXES,'reset_suites':args.reset_suites,'steps':args.steps,'note':'Action-level interpolation only; no parameters updated and no pivot selected automatically.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'rows':len(rows),'axes':list(AXES)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    if True: main()

STAGES = {
    "post_p1_q0_policy_family_audit": run_post_p1_q0_policy_family_audit,
    "post_p1_q1_semantic_interpolation_audit": run_post_p1_q1_semantic_interpolation_audit,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
