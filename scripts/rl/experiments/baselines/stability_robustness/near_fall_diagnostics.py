"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_b1_r0_1_near_fall():
    """Run former b1_r0_1_near_fall.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,math,traceback
    import numpy as np, torch
    import os
    ROOT=Path(__file__).resolve().parents[4]
    OUT=Path(os.environ.get('R0_OUT',str(ROOT/'artifacts'/'b1_r0_1'))); OUT.mkdir(parents=True,exist_ok=True)
    def main():
        app=base=None
        try:
            from isaaclab.app import AppLauncher
            app=AppLauncher({'headless':True,'enable_cameras':False}).app
            import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
            from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
            from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
            from rl.experiments.common.utilities.train_b0 import nominalize
            from rl.core.modules.actor_critic import ActorCritic
            cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=16; cfg.seed=17; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg); base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base); tr=env.reset(); obs=tr['obs']; m=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,[64,64]).to(base.device).eval(); c=torch.load(Path(os.environ.get('R0_CKPT',str(ROOT/'runs/b1_p2_seed2_2026-09-21/checkpoints/update_500.pt'))),map_location=base.device); m.load_state_dict(c['model']); m.set_scheduled_fixed_std(float(c['scheduled_std']))
            O=[]; A=[]; D=[]; T=[]; H=[]; C=[]
            for _ in range(500):
                x=torch.as_tensor(obs,device=base.device,dtype=torch.float32)
                with torch.no_grad(): a=m.act_inference(x)
                q=env.step(a.cpu().numpy().astype(np.float32)); O.append(np.asarray(obs)); A.append(a.cpu().numpy()); D.append(q['done'].copy()); f=q['fields']; T.append(np.asarray(f['roll_pitch']).copy()); H.append(np.asarray(f['height']).copy()); C.append(np.asarray(q['term_base_contact']).copy()); obs=q['obs']
            base.close(); O=np.stack(O); A=np.stack(A); D=np.stack(D); T=np.stack(T); H=np.stack(H); C=np.stack(C); np.savez_compressed(OUT/'trajectory.npz',obs=O,actions=A,done=D,tilt=T,height=H,contact=C); tilt=np.abs(T).max(-1); near=(tilt>math.radians(15))|(H<0.15)|D|C; idx=np.where(near.any(1))[0]; pairs=[]
            for i in idx:
                for j in idx:
                    if j<=i or abs(int(i)-int(j))<2: continue
                    for lane in range(16):
                        if D[i,lane] or D[j,lane] or np.any(D[max(0,i-4):i,lane]) or np.any(D[max(0,j-4):j,lane]): continue
                        diff=O[i,lane].copy()-O[j,lane].copy(); diff[42:45]=0; dist=float(np.linalg.norm(diff)/math.sqrt(48))
                        if dist<0.05: pairs.append({'i':int(i),'j':int(j),'lane':lane,'obs_dist':dist,'action_dist':float(np.linalg.norm(A[i,lane]-A[j,lane])),'history_action_dist':float(np.linalg.norm(A[max(0,i-4):i,lane]-A[max(0,j-4):j,lane])),'tilt_i':float(tilt[i,lane]),'tilt_j':float(tilt[j,lane]),'height_i':float(H[i,lane]),'height_j':float(H[j,lane]),'future_done_i':bool(D[min(i+10,499),lane]),'future_done_j':bool(D[min(j+10,499),lane])})
            result={'schema':'b1_r0_1_near_fall_v1','read_only':True,'checkpoint':'P2 seed2 update500','rollout_shape':list(O.shape),'near_event_steps':int(len(idx)),'done_count':int(D.sum()),'near_pairs':len(pairs),'pairs':pairs[:500],'event_definition':'tilt>15deg or height<0.15 or base contact/done','command_dims_excluded':[42,43,44]}; (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); (OUT/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0},indent=2)+'\n'); print(OUT/'summary.json')
        except BaseException as e:
            (OUT/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
        finally:
            if base is not None: base.close()
            if app is not None: app.close()
    if True: main()

def run_b1_r0_2_analyze():
    """Run former b1_r0_2_analyze.py stage."""
    from pathlib import Path
    import json,numpy as np
    ROOT=Path(__file__).resolve().parents[4]; BASE=ROOT/'artifacts'/'b1_r0_2'; OUT=BASE/'analysis.json'
    GROUPS=['p2_s0','p2_s1','p2_s2','s1_s0','s1_s1','s1_s2']
    def auc(y,s):
        order=np.argsort(s); ranks=np.empty_like(order); ranks[order]=np.arange(len(s)); pos=y==1; n1=pos.sum(); n0=len(y)-n1
        return float((ranks[pos].sum()-n1*(n1-1)/2)/(n1*n0)) if n1 and n0 else float('nan')
    def auprc(y,s):
        pos=float(np.sum(y))
        if pos == 0 or pos == len(y): return float('nan')
        order=np.argsort(-s); yy=y[order]; tp=np.cumsum(yy)
        precision=tp/(np.arange(len(y))+1)
        return float(np.mean(precision[yy==1]))
    def rows_for(z):
        o,d,t,h,c=z['obs'],z['done'],np.abs(z['tilt']).max(-1),z['height'],z['contact']; T,N,D=o.shape; out=[]
        ev=(t>np.deg2rad(15))|(h<.15)|d|c
        for k in range(4,T-10):
            if d[k-3:k].any(): continue
            y=ev[k+1:k+11].any(0).astype(float); cont=np.stack([t[k+10],h[k+10]],-1); cur=o[k]; hist=o[k-3:k+1].transpose(1,0,2).reshape(N,-1); priv=np.stack([t[k],h[k],c[k]],-1)
            out.append((cur,hist,priv,y,cont))
        return out
    def main():
        data={g:rows_for(np.load(BASE/g/'trajectory.npz')) for g in GROUPS}
        result=[]
        for family in ['p2','s1','pooled']:
            allowed=[g for g in GROUPS if family=='pooled' or g.startswith(family+'_')]
            for held in range(3):
                train=[x for g,v in data.items() if g in allowed and int(g[-1])!=held for x in v]
                test=[x for g,v in data.items() if g in allowed and int(g[-1])==held for x in v]
                if not train or not test:
                    continue
                for name,k in [('current',0),('history',1),('privileged',2)]:
                    Xtr=np.concatenate([x[k] for x in train],0); Xte=np.concatenate([x[k] for x in test],0)
                    ytr=np.concatenate([x[3] for x in train]); yte=np.concatenate([x[3] for x in test])
                    ctr=np.concatenate([x[4] for x in train]); cte=np.concatenate([x[4] for x in test])
                    A=np.c_[np.ones(len(Xtr)),Xtr]; B=np.c_[np.ones(len(Xte)),Xte]
                    w=np.linalg.solve(A.T@A+1e-2*np.eye(A.shape[1]),A.T@ytr)
                    wc=np.linalg.solve(A.T@A+1e-2*np.eye(A.shape[1]),A.T@ctr)
                    score=B@w; pred=B@wc
                    result.append({'family':family,'heldout_seed':held,'features':name,'event_rate':float(yte.mean()),'auroc':auc(yte,score),'auprc':auprc(yte,score),'event_brier':float(np.mean((score-yte)**2)),'future_tilt_mae':float(np.mean(np.abs(pred[:,0]-cte[:,0]))),'future_height_mae':float(np.mean(np.abs(pred[:,1]-cte[:,1]))),'test_rows':len(yte)})
        out={'schema':'b1_r0_2_information_value_v1','read_only':True,'rows':result,'note':'equal-size linear ridge probes; no RL update'}
        OUT.write_text(json.dumps(out,indent=2)+'\n'); print(OUT)
    if True: main()

def run_b1_r0_diagnostic():
    """Run former b1_r0_diagnostic.py stage."""
    """Read-only observability/memory diagnostic from P2/S1 final checkpoints."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json, math, traceback
    import os
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'artifacts'/'b1_r0_diagnostic'; OUT.mkdir(parents=True,exist_ok=True)
    MODELS={'p2':ROOT/'runs/b1_p2_seed0_2026-09-21/checkpoints/update_500.pt','s1':ROOT/'runs/b1_s1_seed0_retry1_2026-09-21/checkpoints/update_500.pt'}
    def collect(label,path):
     from isaaclab.app import AppLauncher
     import gymnasium as gym, talon_rl.tasks.locomotion.a1_env
     from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
     from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
     from rl.experiments.common.utilities.train_b0 import nominalize
     from rl.core.modules.actor_critic import ActorCritic
     cfg=IsaacLabTalonEnvCfg(); cfg.scene.num_envs=16; cfg.seed=17; cfg.sim.dt=.01; cfg.decimation=1; nominalize(cfg)
     base=gym.make('Isaac-Talon-A1-v0',cfg=cfg,render_mode=None).unwrapped; env=B0TalonEnv(base); trans=env.reset(); obs=trans['obs']; m=ActorCritic(base.obs_dim,base.obs_dim,base.action_dim,1,[64,64]).to(base.device).eval(); c=torch.load(path,map_location=base.device); m.load_state_dict(c['model']); m.set_scheduled_fixed_std(float(c['scheduled_std']))
     O=[]; A=[]; D=[]; T=[]; H=[]; C=[]
     for _ in range(48):
      x=torch.as_tensor(obs,device=base.device,dtype=torch.float32)
      with torch.no_grad(): a=m.act_inference(x)
      q=env.step(a.cpu().numpy().astype(np.float32)); O.append(np.asarray(obs)); A.append(a.cpu().numpy()); D.append(q['done'].copy()); f=q['fields']; T.append(np.asarray(f['roll_pitch']).copy()); H.append(np.asarray(f['height']).copy()); C.append(np.asarray(q['term_base_contact']).copy()); obs=q['obs']
     base.close(); return {'obs':np.stack(O),'actions':np.stack(A),'done':np.stack(D),'tilt':np.stack(T),'height':np.stack(H),'contact':np.stack(C)}
    def pairs(x):
     o=x['obs']; a=x['actions']; d=x['done']; t=x['tilt']; h=x['height']; valid=[]
     for i in range(o.shape[0]):
      for j in range(i+1,o.shape[0]):
       for lane in range(o.shape[1]):
        if d[i,lane] or d[j,lane] or np.any(d[max(0,i-4):i,lane]) or np.any(d[max(0,j-4):j,lane]): continue
        delta=o[i,lane].copy()-o[j,lane].copy(); delta[42:45]=0
        dist=float(np.linalg.norm(delta)/math.sqrt(48))
        if dist<0.05:
         ai,aj=a[i,lane],a[j,lane]
         outcome=float(np.abs(t[min(i+4,t.shape[0]-1),lane]-t[min(j+4,t.shape[0]-1),lane]).mean())
         valid.append({'i':i,'j':j,'lane':lane,'obs_dist':dist,'action_dist':float(np.linalg.norm(ai-aj)),'history_action_dist':float(np.linalg.norm(a[max(0,i-4):i,lane]-a[max(0,j-4):j,lane])),'future_tilt_delta':outcome})
     return valid
    def main():
     app=None
     try:
      from isaaclab.app import AppLauncher
      app=AppLauncher({'headless':True,'enable_cameras':False}).app; data={}
      selected=os.environ.get('R0_LABEL')
      for label,path in MODELS.items():
       if selected and label!=selected: continue
       data[label]=collect(label,path)
      summary={'schema':'b1_r0_observability_v1','read_only':True,'models':{},'observation_layout':{'joint_pos':'0:12','joint_vel':'12:24','roll_pitch':'24:26','foot_contact':'26:30','last_action':'30:42','command':'42:45','base_ang_vel':'45:48','projected_gravity':'48:51'},'decision_rule':'history dependence -> LSTM; missing cues -> SRM; no signal -> stop expansion'}
      for label,x in data.items():
       ps=pairs(x); summary['models'][label]={'rollout_shape':list(x['obs'].shape),'done_count':int(x['done'].sum()),'near_pairs':len(ps),'pairs':ps[:200],'mean_action_norm':float(np.linalg.norm(x['actions'],axis=-1).mean()),'tilt_mean':float(np.abs(x['tilt']).mean()),'height_mean':float(np.abs(x['height']).mean())}
      suffix=selected or 'all'; (OUT/f'summary_{suffix}.json').write_text(json.dumps(summary,indent=2)+'\n'); (OUT/'RUN_DONE.json').write_text(json.dumps({'status':'RUN_DONE','exit_code':0,'label':suffix},indent=2)+'\n'); print(OUT/f'summary_{suffix}.json')
     except BaseException as e:
      (OUT/'ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n'); raise
     finally:
      if app is not None: app.close()
    if True: main()

STAGES = {
    "b1_r0_1_near_fall": run_b1_r0_1_near_fall,
    "b1_r0_2_analyze": run_b1_r0_2_analyze,
    "b1_r0_diagnostic": run_b1_r0_diagnostic,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
