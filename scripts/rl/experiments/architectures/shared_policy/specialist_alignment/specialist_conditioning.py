"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_post_v1_d1_aggregate():
    """Run former post_v1_d1_aggregate.py stage."""
    """Aggregate the frozen D1 fixed-specialist endpoint audit."""
    import argparse, json
    from pathlib import Path
    import numpy as np
    
    def main():
        ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
        d=json.loads(args.input.read_text()); labels=['P','B','E']; rows=d['behavior'];
        summary={}
        for label in labels:
            rr=[r for r in rows if r['specialist']==label]
            summary[label]={
                'w':rr[0]['w'],
                'objective_return_mean':np.mean([r['objective_return_mean'] for r in rr],axis=0).tolist(),
                'objective_return_reset_sd':np.std([r['objective_return_mean'] for r in rr],axis=0,ddof=1).tolist(),
                'metrics_mean':{k:float(np.mean([r['metrics_mean'][k] for r in rr])) for k in rr[0]['metrics_mean']},
                'metrics_reset_sd':{k:float(np.std([r['metrics_mean'][k] for r in rr],ddof=1)) for k in rr[0]['metrics_mean']},
                'survival_mean':float(np.mean([r['metrics_mean']['survival'] for r in rr])),
            }
        pairs={}
        for a,b in [('P','B'),('P','E'),('B','E')]:
            pairs[f'{a}_vs_{b}']={
                'action_distance_mean':float(np.mean([r['distances'][f'{a}_vs_{b}'] for r in d['action_distances']])),
                'action_distance_sd':float(np.std([r['distances'][f'{a}_vs_{b}'] for r in d['action_distances']],ddof=1)),
                'objective_return_delta':(np.asarray(summary[a]['objective_return_mean'])-np.asarray(summary[b]['objective_return_mean'])).tolist(),
                'vx_error_delta':summary[a]['metrics_mean']['vx_error']-summary[b]['metrics_mean']['vx_error'],
                'tilt_delta':summary[a]['metrics_mean']['tilt_deg']-summary[b]['metrics_mean']['tilt_deg'],
                'torque_delta':summary[a]['metrics_mean']['torque_norm']-summary[b]['metrics_mean']['torque_norm'],
                'action_rate_delta':summary[a]['metrics_mean']['action_rate']-summary[b]['metrics_mean']['action_rate'],
            }
        classification='D1-A' if max(v['action_distance_mean'] for v in pairs.values()) > 1e-2 and any(np.linalg.norm(v['objective_return_delta']) > 1e-2 for v in pairs.values()) else 'D1-B'
        result={'schema':'post_v1_d1_aggregate_v1','status':'CLASSIFIED','classification':classification,'measurement_only_after_training':True,'input':str(args.input),'no_curriculum':d['no_curriculum'],'no_adapters_or_routing':d['no_adapters_or_routing'],'terminal_only':True,'summary':summary,'pairwise':pairs,'decision_rule':'D1-A requires specialist action separation above reset-scale magnitude together with objective-return separation; D1-B otherwise; D1-C is reserved for selective pair separation after multi-specialist replication.','interpretation':'Fixed-preference specialists produce clearly distinct action and objective-return endpoints, supporting a preference-conditioned shared-policy/optimization bottleneck rather than an immediately unidentifiable objective set. D1-A is diagnostic evidence, not a V1 scientific superiority verdict.'}
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'classification':classification},indent=2))
    if True: main()

def run_post_v1_d1_specialists():
    """Run former post_v1_d1_specialists.py stage."""
    """Post-V1 D1: train three fixed-preference specialists, then evaluate terminals."""
    
    import argparse, json, sys, time, traceback
    from pathlib import Path
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    SPECIALISTS = {"P": np.array([.8, .1, .1], dtype=np.float32), "B": np.array([.1, .8, .1], dtype=np.float32), "E": np.array([.1, .1, .8], dtype=np.float32)}
    
    def obs_tensor(x):
        if isinstance(x, dict): x = x.get("policy", next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def tilt_deg(q):
        _, x, y, _ = [q[:, i] for i in range(4)]
        return torch.rad2deg(torch.acos((1 - 2 * (x*x + y*y)).clamp(-1, 1)))
    
    def main():
        ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--updates", type=int, default=300); ap.add_argument("--horizon", type=int, default=2); ap.add_argument("--num-envs", type=int, default=8); ap.add_argument("--eval-steps", type=int, default=32); ap.add_argument("--reset-suites", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--checkpoint", type=Path, default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt")); args = ap.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
        life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
        def mark(event, **extra):
            with life.open("a") as f: f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
        mark("RUN_STARTED", protocol="POST-V1-D1", updates=args.updates, training_authorized=True)
        app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:]; sys.argv=[sys.argv[0]]; app=AppLauncher({"headless":True,"enable_cameras":False}).app; sys.argv=saved; mark("APP_INIT_OK")
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic, initialize_from_rsl_m01, vector_gae, scalarized_late_weighted_ppo, vector_value_loss
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=args.seed; env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg); mark("ENV_CREATED")
            obs,_=env.reset(seed=args.seed); obs=obs_tensor(obs).cuda(); action_dim=env.unwrapped.action_manager.total_action_dim; manager=env.unwrapped.reward_manager; robot=env.unwrapped.scene["robot"]
            models={}; terminal={}; records={}
            for idx,(label,w_np) in enumerate(SPECIALISTS.items()):
                torch.manual_seed(1000+idx); np.random.seed(1000+idx); model=V1CSharedActorCritic(obs.shape[-1],action_dim).cuda(); initialize_from_rsl_m01(model,args.checkpoint,device="cpu"); opt=torch.optim.Adam(model.parameters(),lr=1e-3); w=torch.as_tensor(np.repeat(w_np[None,:],args.num_envs,axis=0),device="cuda"); current,_=env.reset(seed=47001+idx*1000); current=obs_tensor(current).cuda(); rec=[]
                mark("SPECIALIST_START", specialist=label, w=w_np.tolist())
                for update in range(1,args.updates+1):
                    obs_buf=[]; act_buf=[]; old_buf=[]; rew_buf=[]; val_buf=[]; done_buf=[]
                    for _ in range(args.horizon):
                        with torch.no_grad(): action,old=model.act_with_preference(current,w); value=model.value_with_preference(current,w)
                        action=torch.clamp(action,-1,1); nxt,scalar,term,trunc,_=env.step(action); raw=manager._step_reward.detach().cpu().numpy(); names=list(manager.active_terms); vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,)); obs_buf.append(current); act_buf.append(action); old_buf.append(old); rew_buf.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt); val_buf.append(value); done_buf.append((term|trunc).to("cuda")); current=obs_tensor(nxt).cuda()
                    with torch.no_grad(): nxtval=model.value_with_preference(current,w)
                    rt=torch.stack(rew_buf); vt=torch.stack(val_buf); dt=torch.stack(done_buf).bool(); adv,ret=vector_gae(rt,vt,nxtval,dt); fo=torch.cat(obs_buf); fa=torch.cat(act_buf); fold=torch.cat(old_buf); fw=w.repeat(args.horizon,1); ratio=torch.exp(model.logp_with_preference(fo,fw,fa)-fold.detach()); al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,3).detach(),fw); cl=vector_value_loss(model.value_with_preference(fo,fw),ret.reshape(-1,3).detach()); loss=al+cl; opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); rec.append({"update":update,"loss":float(loss.detach()),"reward_mean":rt.mean((0,1)).detach().cpu().tolist()})
                path=args.output.parent/f"specialist_{label}_terminal.pt"; torch.save({"schema":"post_v1_d1_specialist_terminal_v1","specialist":label,"w":w_np.tolist(),"update":args.updates,"model":model.state_dict()},path); models[label]=model.eval(); terminal[label]=str(path); records[label]=rec; mark("SPECIALIST_DONE",specialist=label,checkpoint=str(path))
            behavior=[]; actions=[]
            for idx,(label,w_np) in enumerate(SPECIALISTS.items()):
                model=models[label]; w=torch.as_tensor(np.repeat(w_np[None,:],args.num_envs,axis=0),device="cuda")
                for suite in range(args.reset_suites):
                    cur,_=env.reset(seed=47001+idx*1000+suite); cur=obs_tensor(cur).cuda(); prev=torch.zeros((args.num_envs,action_dim),device="cuda"); vs=[]; mets=[]; actnorm=[]
                    with torch.no_grad():
                        for _ in range(args.eval_steps):
                            a=torch.clamp(model.act_inference_with_preference(cur,w),-1,1); nxt,_,term,trunc,_=env.step(a); raw=manager._step_reward.detach().cpu().numpy(); names=list(manager.active_terms); vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,)); data=robot.data; cmd=env.unwrapped.command_manager.get_command("base_velocity"); mets.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"tilt_deg":float(tilt_deg(data.root_quat_w).mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"torque_norm":float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"survival":float(1-((term|trunc).float().mean()))}); vs.append(vec.mean(0)); actnorm.append(float(torch.linalg.vector_norm(a,dim=-1).mean())); prev=a; cur=obs_tensor(nxt).cuda()
                    behavior.append({"specialist":label,"w":w_np.tolist(),"reset_suite":suite,"objective_return_mean":np.asarray(vs).mean(0).tolist(),"metrics_mean":{k:float(np.mean([m[k] for m in mets])) for k in mets[0]},"action_norm_mean":float(np.mean(actnorm))})
            for suite in range(args.reset_suites):
                cur,_=env.reset(seed=47001+suite); cur=obs_tensor(cur).cuda(); ws={k:torch.as_tensor(np.repeat(v[None,:],args.num_envs,axis=0),device="cuda") for k,v in SPECIALISTS.items()}; ds={"P_vs_B":[],"P_vs_E":[],"B_vs_E":[]}
                with torch.no_grad():
                    for _ in range(args.eval_steps):
                        aa={k:torch.clamp(models[k].act_inference_with_preference(cur,ws[k]),-1,1) for k in SPECIALISTS}; ds["P_vs_B"].append(float(torch.linalg.vector_norm(aa["P"]-aa["B"],dim=-1).mean())); ds["P_vs_E"].append(float(torch.linalg.vector_norm(aa["P"]-aa["E"],dim=-1).mean())); ds["B_vs_E"].append(float(torch.linalg.vector_norm(aa["B"]-aa["E"],dim=-1).mean())); nxt,*_=env.step(aa["P"]); cur=obs_tensor(nxt).cuda()
                actions.append({"reset_suite":suite,"distances":{k:float(np.mean(v)) for k,v in ds.items()}})
            report={"schema":"post_v1_d1_specialist_run_v1","status":"TRAINING_AND_TERMINAL_EVALUATION_COMPLETE","manifest":"artifacts/post_v1/D1_FIXED_SPECIALIST_MANIFEST.json","seed":args.seed,"updates":args.updates,"specialists":list(SPECIALISTS),"terminal_checkpoints":terminal,"records":records,"behavior":behavior,"action_distances":actions,"no_curriculum":True,"no_adapters_or_routing":True,"evaluator":"deterministic actor mean + clip_actions=1.0","note":"D1 measurement artifact; verdict is produced by separate aggregation."}; args.output.write_text(json.dumps(report,indent=2)+"\n"); mark("ARTIFACT_WRITTEN",path=str(args.output)); mark("RUN_DONE",status=report["status"]); print(json.dumps({"status":report["status"],"specialists":list(SPECIALISTS),"behavior_rows":len(behavior),"action_rows":len(actions)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n"); mark("ERROR",error=str(exc)); raise
        finally:
            if env is not None: env.close()
            if app is not None: app.close()
    
    if True: main()

def run_post_v1_d2_conditioning_audit():
    """Run former post_v1_d2_conditioning_audit.py stage."""
    """Post-V1 D2: read-only layer-wise preference-conditioning audit."""
    import argparse, hashlib, json, sys, time, traceback
    from pathlib import Path
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    PAIRS=(('P','B'),('P','E'),('B','E'))
    
    def hash_model(m):
        return hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def main():
        ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--num-envs',type=int,default=8); ap.add_argument('--seed',type=int,default=47001); args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
        life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V1-D2',measurement_only=True); app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym, isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.three_objective import V1CSharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED')
            obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();action_dim=env.unwrapped.action_manager.total_action_dim
            models={};before={}
            for seed in range(3):
                for condition in ('curriculum','full_simplex_control'):
                    path=ROOT/'runs'/f'v1c_confirmatory_seed{seed}-2026-09-22'/f'{condition}_terminal.pt';payload=torch.load(path,map_location='cuda',weights_only=False);m=V1CSharedActorCritic(obs.shape[-1],action_dim).cuda();m.load_state_dict(payload['model']);m.eval();models[(seed,condition)]=m;before[f'{seed}:{condition}']=hash_model(m)
            mark('CHECKPOINTS_LOADED',count=len(models))
            common=[]
            for (seed,condition),m in models.items():
                pref_outputs={}; pref_hidden={}
                for label,w_np in PREFS.items():
                    w=torch.as_tensor(np.repeat(w_np[None,:],args.num_envs,axis=0),device='cuda')
                    x=torch.cat((obs,w),dim=-1)
                    hs=[];h=x
                    for layer in m.actor_body:
                        h=layer(h)
                        if isinstance(layer,torch.nn.ELU): hs.append(h.detach())
                    pref_hidden[label]=hs
                    pref_outputs[label]=torch.clamp(m.act_inference_with_preference(obs,w),-1,1).detach()
                hidden_pair={}
                for a,b in PAIRS:
                    hidden_pair[f'{a}_vs_{b}']=[float(torch.linalg.vector_norm(pref_hidden[a][i]-pref_hidden[b][i],dim=-1).mean().item()) for i in range(3)]
                action_pair={f'{a}_vs_{b}':float(torch.linalg.vector_norm(pref_outputs[a]-pref_outputs[b],dim=-1).mean().item()) for a,b in PAIRS}
                jac=[]; critic_jac=[]
                for label,w_np in PREFS.items():
                    vals=[]
                    for n in range(args.num_envs):
                        oi=obs[n:n+1].detach(); wi=torch.tensor(w_np,device='cuda',requires_grad=True).reshape(1,3)
                        def fn(q): return torch.clamp(m.act_inference_with_preference(oi,q),-1,1).reshape(-1)
                        j=torch.autograd.functional.jacobian(fn,wi,create_graph=False).detach().cpu().numpy();vals.append(float(np.linalg.norm(j)))
                    jac.append({'preference':label,'action_w_jacobian_norm_mean':float(np.mean(vals)),'action_w_jacobian_norm_sd':float(np.std(vals,ddof=1))})
                    vals=[]
                    for n in range(args.num_envs):
                        oi=obs[n:n+1].detach(); wi=torch.tensor(w_np,device='cuda',requires_grad=True).reshape(1,3)
                        def value_fn(q): return m.value_with_preference(oi,q).reshape(-1)
                        j=torch.autograd.functional.jacobian(value_fn,wi,create_graph=False).detach().cpu().numpy(); vals.append(float(np.linalg.norm(j)))
                    critic_jac.append({'preference':label,'value_w_jacobian_norm_mean':float(np.mean(vals)),'value_w_jacobian_norm_sd':float(np.std(vals,ddof=1))})
                aw=m.actor_body[0].weight.detach();cw=m.critic_body[0].weight.detach();pd=float(torch.linalg.vector_norm(aw[:,m.physical_obs_dim:]).item());pp=float(torch.linalg.vector_norm(aw[:,:m.physical_obs_dim]).item());cd=float(torch.linalg.vector_norm(cw[:,m.physical_obs_dim:]).item());cp=float(torch.linalg.vector_norm(cw[:,:m.physical_obs_dim]).item())
                common.append({'seed':seed,'condition':condition,'actor_preference_column_norm':pd,'actor_physical_column_norm':pp,'actor_preference_to_physical_ratio':pd/(pp+1e-12),'critic_preference_column_norm':cd,'critic_physical_column_norm':cp,'critic_preference_to_physical_ratio':cd/(cp+1e-12),'hidden_pair_distances':hidden_pair,'action_pair_distances':action_pair,'action_w_jacobian':jac,'critic_w_jacobian':critic_jac})
            after={f'{k[0]}:{k[1]}':hash_model(m) for k,m in models.items()};report={'schema':'post_v1_d2_conditioning_audit_v1','status':'MEASUREMENT_COMPLETE','measurement_only':True,'training':False,'preferences':{k:v.tolist() for k,v in PREFS.items()},'conditions':['curriculum','full_simplex_control'],'seeds':[0,1,2],'observation_seed':args.seed,'layerwise_hidden_layers':3,'rows':common,'model_hashes_before':before,'model_hashes_after':after,'no_mutation':before==after,'note':'D2 measures where preference sensitivity is lost; no parameters are updated.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status'],no_mutation=report['no_mutation']);print(json.dumps({'status':report['status'],'rows':len(common),'no_mutation':report['no_mutation']},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

def run_post_v1_d3_screen():
    """Run former post_v1_d3_screen.py stage."""
    """Post-V1 D3 one-seed, 100-update FiLM diagnostic screen."""
    import argparse,json,sys,time,traceback
    from pathlib import Path
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    PREFS={'P':np.array([.8,.1,.1],dtype=np.float32),'B':np.array([.1,.8,.1],dtype=np.float32),'E':np.array([.1,.1,.8],dtype=np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)];return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def audit(model,obs):
        out={}; hs={}; acts={}
        for label,w_np in PREFS.items():
            w=torch.as_tensor(np.repeat(w_np[None,:],obs.shape[0],axis=0),device='cuda');x=torch.cat((obs,w),-1);h=model.actor_pre(x);h=torch.nn.functional.elu(h);hs[label]=[h.detach()];h=model.film_gamma(w)*h+model.film_beta(w);h=model.actor_rest(h);hs[label].append(h.detach());acts[label]=torch.clamp(model.act_inference_with_preference(obs,w),-1,1).detach()
        out['action_pair_distance']={f'{a}_vs_{b}':float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean()) for a,b in [('P','B'),('P','E'),('B','E')]}
        out['hidden_pair_distance']={f'{a}_vs_{b}':[float(torch.linalg.vector_norm(hs[a][i]-hs[b][i],dim=-1).mean()) for i in range(2)] for a,b in [('P','B'),('P','E'),('B','E')]}
        out['film_gamma_pair_distance']={f'{a}_vs_{b}':float(torch.linalg.vector_norm(model.film_gamma(torch.as_tensor(PREFS[a],device='cuda'))-model.film_gamma(torch.as_tensor(PREFS[b],device='cuda')))) for a,b in [('P','B'),('P','E'),('B','E')]}
        out['film_beta_pair_distance']={f'{a}_vs_{b}':float(torch.linalg.vector_norm(model.film_beta(torch.as_tensor(PREFS[a],device='cuda'))-model.film_beta(torch.as_tensor(PREFS[b],device='cuda')))) for a,b in [('P','B'),('P','E'),('B','E')]}
        jac=[]
        for label,w_np in PREFS.items():
            vals=[]
            for n in range(obs.shape[0]):
                oi=obs[n:n+1].detach();wi=torch.tensor(w_np,device='cuda',requires_grad=True).reshape(1,3)
                fn=lambda q: torch.clamp(model.act_inference_with_preference(oi,q),-1,1).reshape(-1)
                vals.append(float(np.linalg.norm(torch.autograd.functional.jacobian(fn,wi).detach().cpu().numpy())))
            jac.append({'preference':label,'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1))})
        out['action_w_jacobian']=jac;out['preference_column_norm']=float(torch.linalg.vector_norm(model.actor_pre.weight[:,model.physical_obs_dim:]).detach());out['film_gamma_norm']=float(torch.linalg.vector_norm(model.film_gamma.weight).detach());out['film_beta_norm']=float(torch.linalg.vector_norm(model.film_beta.weight).detach());return out
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=100);ap.add_argument('--horizon',type=int,default=2);ap.add_argument('--num-envs',type=int,default=8);ap.add_argument('--eval-steps',type=int,default=32);ap.add_argument('--reset-suites',type=int,default=4);ap.add_argument('--seed',type=int,default=0);args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+'.lifecycle.jsonl')
        def mark(e,**x):
            with life.open('a') as f:f.write(json.dumps({'event':e,'unix':time.time(),**x},sort_keys=True)+'\n')
        mark('RUN_STARTED',protocol='POST-V1-D3-SCREEN',updates=args.updates,training=True);app=env=None
        try:
            from isaaclab.app import AppLauncher
            saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved;mark('APP_INIT_OK')
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from rl.core.preferences.v1c_curriculum import load_manifest,sample_preferences
            from talon_rl.rewards.baselines import group_v1b_s7_terms
            from talon_rl.models.foundations.three_objective import vector_gae,scalarized_late_weighted_ppo,vector_value_loss
            from talon_rl.models.conditioning.feature_film import D3FiLMActorCritic,initialize_from_v1c
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);mark('ENV_CREATED');obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
            source=torch.load(ROOT/'runs/v1c_confirmatory_seed0-2026-09-22/full_simplex_control_terminal.pt',map_location='cpu',weights_only=False)['model'];model=D3FiLMActorCritic(obs.shape[-1],ad).cuda();initialize_from_v1c(model,source);model.train();opt=torch.optim.Adam(model.parameters(),lr=1e-3);manifest=load_manifest();rng=np.random.default_rng(args.seed+100000);manager=env.unwrapped.reward_manager;checkpoints={'0':audit(model,obs)};records=[];cur,_=env.reset(seed=args.seed);cur=obs_tensor(cur).cuda();
            for update in range(1,args.updates+1):
                w_np,_=sample_preferences(rng,update,args.num_envs,'full_simplex_control',manifest);w=torch.as_tensor(w_np,device='cuda');ob=[];ac=[];old=[];rew=[];val=[];done=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp=model.act_with_preference(cur,w);v=model.value_with_preference(cur,w)
                    a=torch.clamp(a,-1,1);nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));ob.append(cur);ac.append(a);old.append(lp);rew.append(torch.as_tensor(vec,device='cuda')*env.unwrapped.step_dt);val.append(v);done.append((term|trunc).to('cuda'));cur=obs_tensor(nxt).cuda()
                with torch.no_grad():nv=model.value_with_preference(cur,w)
                rt=torch.stack(rew);vt=torch.stack(val);dt=torch.stack(done).bool();adv,ret=vector_gae(rt,vt,nv,dt);fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(old);fw=w.repeat(args.horizon,1);ratio=torch.exp(model.logp_with_preference(fo,fw,fa)-fold.detach());al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,3).detach(),fw);cl=vector_value_loss(model.value_with_preference(fo,fw),ret.reshape(-1,3).detach());loss=al+cl;opt.zero_grad(set_to_none=True);loss.backward();opt.step();records.append({'update':update,'loss':float(loss.detach()),'reward_mean':rt.mean((0,1)).detach().cpu().tolist()})
                if update in (25,50,100): model.eval();checkpoints[str(update)]=audit(model,cur);model.train()
            model.eval();behavior=[]
            for label,w_np in PREFS.items():
                w=torch.as_tensor(np.repeat(w_np[None,:],args.num_envs,axis=0),device='cuda')
                for suite in range(args.reset_suites):
                    cur,_=env.reset(seed=47001+suite);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device='cuda');vecs=[];ms=[]
                    with torch.no_grad():
                        for _ in range(args.eval_steps):
                            a=torch.clamp(model.act_inference_with_preference(cur,w),-1,1);nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,));data=env.unwrapped.scene['robot'].data;cmd=env.unwrapped.command_manager.get_command('base_velocity');ms.append({'vx_error':float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),'tilt_deg':float(tilt(data.root_quat_w).mean()),'ang_vel_xy':float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),'torque_norm':float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),'action_rate':float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),'survival':float(1-((term|trunc).float().mean()))});vecs.append(vec.mean(0));prev=a;cur=obs_tensor(nxt).cuda()
                    behavior.append({'preference':label,'reset_suite':suite,'objective_return_mean':np.asarray(vecs).mean(0).tolist(),'metrics_mean':{k:float(np.mean([m[k] for m in ms])) for k in ms[0]}})
            terminal=args.output.parent/'d3_film_terminal.pt';torch.save({'schema':'post_v1_d3_film_terminal_v1','update':args.updates,'model':model.state_dict(),'optimizer':opt.state_dict()},terminal);report={'schema':'post_v1_d3_screen_v1','status':'SCREEN_COMPLETE','training_authorized':False,'source_checkpoint':'runs/v1c_confirmatory_seed0-2026-09-22/full_simplex_control_terminal.pt','updates':args.updates,'records':records,'sensitivity_checkpoints':checkpoints,'behavior':behavior,'terminal_checkpoint':str(terminal),'no_new_mechanisms':True,'note':'Diagnostic screen only; no full D3 authorization.'};args.output.write_text(json.dumps(report,indent=2)+'\n');mark('ARTIFACT_WRITTEN',path=str(args.output));mark('RUN_DONE',status=report['status']);print(json.dumps({'status':report['status'],'checkpoints':list(checkpoints),'behavior_rows':len(behavior)},indent=2))
        except BaseException as exc:
            args.output.with_name(args.output.stem+'.ERROR.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');mark('ERROR',error=str(exc));raise
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if True:main()

STAGES = {
    "post_v1_d1_aggregate": run_post_v1_d1_aggregate,
    "post_v1_d1_specialists": run_post_v1_d1_specialists,
    "post_v1_d2_conditioning_audit": run_post_v1_d2_conditioning_audit,
    "post_v1_d3_screen": run_post_v1_d3_screen,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
