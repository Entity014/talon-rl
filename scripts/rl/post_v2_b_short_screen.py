#!/usr/bin/env python3
"""V2-B one-seed short screen: only PPO + the frozen manifold loss."""
from __future__ import annotations
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
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
        from talon_rl.v1b_s7_reward_vector import group_v1b_s7_terms
        from talon_rl.v1c_actor_critic import vector_gae,vector_value_loss
        from talon_rl.v2_behavior_actor_critic import V2BehaviorActorCritic,initialize_from_v1c
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
if __name__=='__main__':main()
