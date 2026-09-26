"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_post_v1_d4a_audit():
    """Run former post_v1_d4a_audit.py stage."""
    """Post-V1 D4-A: read-only preference recoverability and gradient audit."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np, torch
    from torch import nn
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def flat_grads(grads,params):
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(grads,params)])
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--reset-suites',type=int,default=4);ap.add_argument('--seed',type=int,default=47001);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V1-D4A',measurement_only=True);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            from talon_rl.models.conditioning.feature_film import D3FiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED')
            obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            specs={'v1c_control':(ROOT/'runs/v1c_confirmatory_seed0-2026-09-22/full_simplex_control_terminal.pt',V1CSharedActorCritic),'d3_film':(ROOT/'runs/post_v1_d3-2026-09-22/d3_film_terminal.pt',D3FiLMActorCritic)};models={}
            for name,(path,cls) in specs.items():
                s=torch.load(path,map_location='cuda',weights_only=False)['model'];m=cls(obs.shape[-1],ad).cuda();m.load_state_dict(s);m.eval();models[name]=m
            mark('CHECKPOINTS_LOADED',count=len(models))
            # Common observation batch: same reset suites and deterministic P-driven path.
            ref=models['v1c_control'];obs_batches=[]
            with torch.no_grad():
                for suite in range(args.reset_suites):
                    cur,_=env.reset(seed=args.seed+suite);cur=obs_tensor(cur).cuda()
                    for _ in range(args.steps):
                        obs_batches.append(cur.detach().cpu());w=torch.as_tensor(np.repeat(PREFS['P'][None,:],args.num_envs,axis=0),device='cuda');a=torch.clamp(ref.act_inference_with_preference(cur,w),-1,1);nxt,*_=env.step(a);cur=obs_tensor(nxt).cuda()
            observations=torch.cat(obs_batches,0).cuda();n=observations.shape[0]
            rows=[]
            for model_name,m in models.items():
                def hidden_for(o,w):
                    if model_name=='v1c_control':
                        x=m._with_w(o,w);hs=[];h=x
                        for layer in m.actor_body:
                            h=layer(h)
                            if isinstance(layer,nn.ELU):hs.append(h)
                        return hs
                    h=m.actor_pre(m._with_w(o,w));hs=[torch.nn.functional.elu(h)];h=m.film_gamma(w)*hs[-1]+m.film_beta(w);h=m.actor_rest(h);hs.append(h);return hs
                probe_results={};sens=[]
                layer_count = 3 if model_name == 'v1c_control' else 2
                for li in range(layer_count):
                    feats=[];labels=[]
                    for label,w_np in PREFS.items():
                        w=torch.as_tensor(np.repeat(w_np[None,:],n,axis=0),device='cuda');feats.append(hidden_for(observations,w)[li].detach());labels.append(w)
                    X=torch.cat(feats);Y=torch.cat(labels);perm=torch.randperm(X.shape[0],device='cuda');split=int(.8*len(perm));tr,te=perm[:split],perm[split:];mu=X[tr].mean(0,keepdim=True);sd=X[tr].std(0,keepdim=True).clamp_min(1e-5);Xn=(X-mu)/sd;probe=nn.Linear(X.shape[1],3).cuda();opt=torch.optim.Adam(probe.parameters(),lr=.02)
                    for _ in range(300):
                        loss=(probe(Xn[tr])-Y[tr]).pow(2).mean();opt.zero_grad();loss.backward();opt.step()
                    with torch.no_grad():pred=probe(Xn[te]);mse=float((pred-Y[te]).pow(2).mean());base=float((Y[te]-Y[tr].mean(0)).pow(2).mean());r2=1-mse/(base+1e-12);acc=float((pred.argmax(-1)==Y[te].argmax(-1)).float().mean())
                    params=[p for p in m.parameters() if p.requires_grad];probe_obs=observations[:min(64,n)];probe_w=torch.as_tensor(np.repeat(PREFS['P'][None,:],min(64,n),axis=0),device='cuda');aux=(probe((hidden_for(probe_obs,probe_w)[li]-mu)/sd)-probe_w).pow(2).mean();ag=flat_grads(torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True),params)
                    with torch.no_grad(): sample,old=m.act_with_preference(probe_obs,probe_w)
                    ratio=torch.exp(m.logp_with_preference(probe_obs,probe_w,sample)-old.detach());ppo_loss=-(ratio*torch.randn_like(ratio)).mean();pg=flat_grads(torch.autograd.grad(ppo_loss,params,retain_graph=True,allow_unused=True),params);cos=float(torch.dot(ag,pg)/(torch.linalg.vector_norm(ag)*torch.linalg.vector_norm(pg)+1e-12));probe_results[f'layer_{li}']={'probe_mse':mse,'probe_r2':float(r2),'probe_accuracy':acc,'aux_grad_norm':float(torch.linalg.vector_norm(ag)),'ppo_proxy_grad_norm':float(torch.linalg.vector_norm(pg)),'aux_ppo_cosine':cos}
                    vals=[]
                    for k in range(min(16,n)):
                        oi=observations[k:k+1].detach();wi=torch.tensor(PREFS['P'],device='cuda',requires_grad=True).reshape(1,3);fn=lambda q:hidden_for(oi,q)[li].reshape(-1);vals.append(float(torch.linalg.norm(torch.autograd.functional.jacobian(fn,wi)).detach()))
                    sens.append({'layer':li,'dw_norm_mean':float(np.mean(vals)),'dw_norm_sd':float(np.std(vals,ddof=1))})
                # Action and PPO-gradient proxy.
                w=torch.as_tensor(np.repeat(PREFS['P'][None,:],n,axis=0),device='cuda');with_grad=observations[:64].detach();w64=w[:64];a=torch.clamp(m.act_inference_with_preference(with_grad,w64),-1,1);action_j=[]
                for k in range(min(16,n)):
                    oi=observations[k:k+1].detach();wi=torch.tensor(PREFS['P'],device='cuda',requires_grad=True).reshape(1,3);action_j.append(float(torch.linalg.norm(torch.autograd.functional.jacobian(lambda q:torch.clamp(m.act_inference_with_preference(oi,q),-1,1).reshape(-1),wi)).detach()))
                compat=[{'layer':layer_name,'aux_grad_norm':info['aux_grad_norm'],'ppo_proxy_grad_norm':info['ppo_proxy_grad_norm'],'aux_ppo_cosine':info['aux_ppo_cosine']} for layer_name,info in probe_results.items()]
                rows.append({'model':model_name,'probe_results':probe_results,'layer_sensitivity':sens,'action_jacobian_norm_mean':float(np.mean(action_j)),'action_jacobian_norm_sd':float(np.std(action_j,ddof=1)),'gradient_compatibility':compat})
            report={'schema':'post_v1_d4a_audit_v1','status':'MEASUREMENT_COMPLETE','measurement_only':True,'training':False,'preferences':{k:v.tolist() for k,v in PREFS.items()},'models':list(models),'observation_count':int(n),'rows':rows,'note':'Probe heads are auxiliary read-only measurements; no policy optimizer step was performed. PPO gradient is a diagnostic proxy only.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'models':list(models),'observations':n},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v1_d4b_screen():
    """Run former post_v1_d4b_screen.py stage."""
    """Post-V1 D4-B one-seed 100-update auxiliary-loss screen."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def flat(gs,ps):return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs,ps)])
    def measure(m,obs):
        with torch.no_grad():
            hs={};acts={};pred={}
            for k,v in PREFS.items():
                w=torch.as_tensor(np.repeat(v[None,:],len(obs),0),device='cuda');hs[k]=m.actor_hidden_first(obs,w);acts[k]=torch.clamp(m.act_inference_with_preference(obs,w),-1,1);pred[k]=m.predict_preference(obs,w)
            ap={f'{a}_vs_{b}':float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean()) for a,b in [('P','B'),('P','E'),('B','E')]};hp={f'{a}_vs_{b}':float(torch.linalg.vector_norm(hs[a]-hs[b],dim=-1).mean()) for a,b in [('P','B'),('P','E'),('B','E')]};pmse=float(np.mean([((pred[k]-torch.as_tensor(np.repeat(v[None,:],len(obs),0),device='cuda'))**2).mean().item() for k,v in PREFS.items()]))
        jac=[]
        for k,v in PREFS.items():
            vals=[]
            for i in range(min(16,len(obs))):
                oi=obs[i:i+1].detach();wi=torch.tensor(v,device='cuda',requires_grad=True).reshape(1,3);vals.append(float(torch.linalg.norm(torch.autograd.functional.jacobian(lambda q:torch.clamp(m.act_inference_with_preference(oi,q),-1,1).reshape(-1),wi)).detach()))
            jac.append({'preference':k,'mean':float(np.mean(vals))})
        return {'action_pair_distance':ap,'hidden_pair_distance':hp,'prediction_mse':pmse,'action_w_jacobian':jac}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=100);ap.add_argument('--horizon',type=int,default=2);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--eval-steps',type=int,default=32);ap.add_argument('--reset-suites',type=int,default=4);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V1-D4B',lambda_aux=1.0,updates=args.updates);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.preferences.v1c_curriculum import load_manifest,sample_preferences
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import vector_gae,scalarized_late_weighted_ppo,vector_value_loss
            from talon_rl.models.auxiliary.preference_head import D4BAuxActorCritic,initialize_from_v1c
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;src=torch.load(ROOT/'runs/v1c_confirmatory_seed0-2026-09-22/full_simplex_control_terminal.pt',map_location='cpu',weights_only=False)['model'];m=D4BAuxActorCritic(obs.shape[-1],ad).cuda();initialize_from_v1c(m,src);opt=torch.optim.Adam(m.parameters(),lr=1e-3);manifest=load_manifest();rng=np.random.default_rng(100000);manager=env.unwrapped.reward_manager;checkpoints={'0':measure(m,obs)};records=[];cur,_=env.reset(seed=0);cur=obs_tensor(cur).cuda();actor_params=list(m.actor_body.parameters())+list(m.actor_mean.parameters())
            for update in range(1,args.updates+1):
                w_np,_=sample_preferences(rng,update,args.num_envs,'full_simplex_control',manifest);w=torch.as_tensor(w_np,device='cuda');ob=[];ac=[];old=[];rew=[];val=[];done=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp=m.act_with_preference(cur,w);v=m.value_with_preference(cur,w)
                    a=torch.clamp(a,-1,1);nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));ob.append(cur);ac.append(a);old.append(lp);rew.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val.append(v);done.append((term|trunc).to('cuda'));cur=obs_tensor(nxt).cuda()
                with torch.no_grad():nv=m.value_with_preference(cur,w)
                rt=torch.stack(rew);vt=torch.stack(val);dt=torch.stack(done).bool();adv,ret=vector_gae(rt,vt,nv,dt);fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(old);fw=w.repeat(args.horizon,1);ratio=torch.exp(m.logp_with_preference(fo,fw,fa)-fold.detach());ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,3).detach(),fw);aux=m.auxiliary_preference_loss(fo,fw);gg=torch.autograd.grad(aux,actor_params,retain_graph=True,allow_unused=True);ppo_proxy=-m.logp_with_preference(fo,fw,fa).mean();gp=torch.autograd.grad(ppo_proxy,actor_params,retain_graph=True,allow_unused=True);gaux=flat(gg,actor_params);gppo=flat(gp,actor_params);ratio_g=float(torch.linalg.vector_norm(gaux)/(torch.linalg.vector_norm(gppo)+1e-12));cos=float(torch.dot(gaux,gppo)/(torch.linalg.vector_norm(gaux)*torch.linalg.vector_norm(gppo)+1e-12));critic=vector_value_loss(m.value_with_preference(fo,fw),ret.reshape(-1,3).detach());loss=ppo+critic+aux;opt.zero_grad(set_to_none=True);loss.backward();opt.step();records.append({'update':update,'ppo_loss':float(ppo.detach()),'aux_loss':float(aux.detach()),'critic_loss':float(critic.detach()),'aux_to_ppo_grad_ratio':ratio_g,'aux_ppo_grad_cosine':cos,'ppo_gradient_proxy':'negative mean log-prob; deterministic diagnostic only'})
                if update in (25,50,100):m.eval();checkpoints[str(update)]=measure(m,cur);m.train()
            behavior=[];m.eval()
            for k,v in PREFS.items():
                w=torch.as_tensor(np.repeat(v[None,:],args.num_envs,0),device='cuda')
                for suite in range(args.reset_suites):
                    cur,_=env.reset(seed=47001+suite);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device='cuda');vs=[];ms=[]
                    with torch.no_grad():
                        for _ in range(args.eval_steps):
                            a=torch.clamp(m.act_inference_with_preference(cur,w),-1,1);nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');ms.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'tilt_deg':float(torch.rad2deg(torch.acos((1-2*(data.root_quat_w[:,1]**2+data.root_quat_w[:,2]**2)).clamp(-1,1))).mean()),'torque_norm':float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),'action_rate':float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),'survival':float(1-((term|trunc).float().mean()))});vs.append(vec.mean(0));prev=a;cur=obs_tensor(nxt).cuda()
                    behavior.append({'preference':k,'reset_suite':suite,'objective_return_mean':np.asarray(vs).mean(0).tolist(),'metrics_mean':{x:float(np.mean([z[x] for z in ms])) for x in ms[0]}})
            terminal=args.output.parent/'d4b_terminal.pt';torch.save({'schema':'post_v1_d4b_terminal_v1','update':args.updates,'model':m.state_dict(),'optimizer':opt.state_dict()},terminal);report={'schema':'post_v1_d4b_screen_v1','status':'SCREEN_COMPLETE','training_authorized':False,'lambda_aux':1.0,'source_checkpoint':'runs/v1c_confirmatory_seed0-2026-09-22/full_simplex_control_terminal.pt','records':records,'sensitivity_checkpoints':checkpoints,'behavior':behavior,'terminal_checkpoint':str(terminal),'note':'One-seed diagnostic only; no full D4-B authorization.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'checkpoints':list(checkpoints),'behavior_rows':len(behavior)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v1_d4c_alignment_audit():
    """Run former post_v1_d4c_alignment_audit.py stage."""
    """Do D4-B's preference-dependent actions point where the D1 specialists do?
    
    Measurement only, no parameters updated. For each preference it compares
    D4-B's action against that preference's specialist, and checks the difference
    between any two preferences matches the difference between the corresponding
    specialists in size and direction.
    """
    import json
    import sys
    import time
    import traceback
    from pathlib import Path
    
    import numpy as np
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.isaac_audit import RUNS, IsaacAudit, obs_tensor
    
    PREFS = {"P": np.array([.8, .1, .1], dtype=np.float32),
             "B": np.array([.1, .8, .1], dtype=np.float32),
             "E": np.array([.1, .1, .8], dtype=np.float32)}
    PAIRS = [("P", "B"), ("P", "E"), ("B", "E")]
    D4B = "post_v1_d4b-2026-09-22/d4b_terminal.pt"
    D1 = "post_v1_d1-2026-09-22/specialist_{}_terminal.pt"
    BASE_SEED = 47001
    
    
    class D4CAlignmentAudit(IsaacAudit):
        """D4-B action alignment against the fixed-preference D1 specialists."""
    
        run = "post_v1_d4c-2026-09-22"
        report = "d4c.json"
        schema = "post_v1_d4c_alignment_audit_v1"
        reset_seed = BASE_SEED
        set_cfg_seed = False
        set_usd_path = False
        steps = 32
        reset_suites = 4
    
        def mark(self, event, **kw):
            path = self.out / "d4c.lifecycle.jsonl"
            with path.open("a") as f:
                f.write(json.dumps({"event": event, "unix": time.time(), **kw}, sort_keys=True) + "\n")
    
        def pref_tensor(self, label):
            return torch.as_tensor(np.repeat(PREFS[label][None, :], self.num_envs, 0), device="cuda")
    
        def suite(self, env, d4, spec, suite):
            cur, _ = env.reset(seed=BASE_SEED + suite)
            cur = obs_tensor(cur).cuda()
            dist = {f"{a}_vs_{b}": [] for a, b in PAIRS}
            cos = {f"{a}_vs_{b}": [] for a, b in PAIRS}
            align, dof = [], []
            with torch.no_grad():
                for _ in range(self.steps):
                    d4a = {k: torch.clamp(d4.act_inference_with_preference(cur, self.pref_tensor(k)), -1, 1)
                           for k in PREFS}
                    sa = {k: torch.clamp(spec[k].act_inference_with_preference(cur, self.pref_tensor(k)), -1, 1)
                          for k in PREFS}
                    for a in PREFS:
                        own = torch.linalg.vector_norm(d4a[a] - sa[a], dim=-1)
                        others = torch.stack([torch.linalg.vector_norm(d4a[a] - sa[b], dim=-1)
                                              for b in PREFS if b != a], 0)
                        align.append({"preference": a, "own_distance": float(own.mean()),
                                      "other_distance_mean": float(others.mean()),
                                      "own_is_nearest": bool((own < others.min(0).values).float().mean())})
                        dof.append({"preference": a,
                                    "own_per_dof": (d4a[a] - sa[a]).abs().mean(0).detach().cpu().tolist()})
                    for a, b in PAIRS:
                        u, v = d4a[a] - d4a[b], sa[a] - sa[b]
                        dist[f"{a}_vs_{b}"].append(float(torch.linalg.vector_norm(u - v, dim=-1).mean()))
                        cos[f"{a}_vs_{b}"].append(
                            float(torch.nn.functional.cosine_similarity(u, v, dim=-1).mean()))
                    nxt, *_ = env.step(d4a["P"])
                    cur = obs_tensor(nxt).cuda()
            return {"reset_suite": suite,
                    "pairwise_difference_distance": {k: float(np.mean(v)) for k, v in dist.items()},
                    "pairwise_difference_cosine": {k: float(np.mean(v)) for k, v in cos.items()},
                    "preference_alignment": align, "per_dof": dof}
    
        def rollout(self, env, obs):
            from talon_rl.models.auxiliary.preference_head import D4BAuxActorCritic
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
    
            self.mark("APP_INIT_OK")
            self.mark("ENV_CREATED")
            ad = env.unwrapped.action_manager.total_action_dim
            d4 = D4BAuxActorCritic(obs.shape[-1], ad).cuda()
            d4.load_state_dict(torch.load(RUNS / D4B, map_location="cuda",
                                          weights_only=False)["model"])
            d4.eval()
            spec = {}
            for label in PREFS:
                m = V1CSharedActorCritic(obs.shape[-1], ad).cuda()
                m.load_state_dict(torch.load(RUNS / D1.format(label), map_location="cuda",
                                             weights_only=False)["model"])
                m.eval()
                spec[label] = m
            self.mark("CHECKPOINTS_LOADED", count=4)
    
            rows = [self.suite(env, d4, spec, s) for s in range(self.reset_suites)]
            report = {"schema": self.schema, "status": "MEASUREMENT_COMPLETE",
                      "measurement_only": True, "d4b_checkpoint": f"runs/{D4B}",
                      "d1_checkpoints": {k: f"runs/{D1.format(k)}" for k in PREFS},
                      "reset_suites": self.reset_suites, "steps": self.steps, "rows": rows,
                      "note": "D4-C compares D4-B preference-dependent action changes against "
                              "fixed-preference specialist action references; no parameters are updated."}
            self.write(report)
            self.mark("ARTIFACT_WRITTEN", path=str(self.out / self.report))
            self.mark("RUN_DONE", status=report["status"])
            print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))
            return report
    
        def execute(self):
            self.mark("RUN_STARTED", protocol="POST-V1-D4C", measurement_only=True)
            try:
                return super().execute()
            except BaseException as exc:
                self.write({"status": "ERROR", "error": str(exc),
                            "traceback": traceback.format_exc()}, "d4c.ERROR.json")
                self.mark("ERROR", error=str(exc))
                raise
    
    
    if True:
        D4CAlignmentAudit.main()

STAGES = {
    "post_v1_d4a_audit": run_post_v1_d4a_audit,
    "post_v1_d4b_screen": run_post_v1_d4b_screen,
    "post_v1_d4c_alignment_audit": run_post_v1_d4c_alignment_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
