#!/usr/bin/env python3
"""Read-only terminal sensitivity audit for the V2-B short-screen checkpoint."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
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
    from talon_rl.v1b_s7_reward_vector import group_v1b_s7_terms
    from talon_rl.v2_behavior_actor_critic import V2BehaviorActorCritic
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
if __name__=='__main__':main()
