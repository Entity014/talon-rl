#!/usr/bin/env python3
"""Instrumented matched V2-B rerun for latent-learning diagnosis."""
from __future__ import annotations
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
        from talon_rl.v1b_s7_reward_vector import group_v1b_s7_terms
        from talon_rl.v1c_actor_critic import vector_gae,vector_value_loss
        from talon_rl.v2_behavior_actor_critic import V2BehaviorActorCritic,initialize_from_v1c
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
if __name__=='__main__':main()
