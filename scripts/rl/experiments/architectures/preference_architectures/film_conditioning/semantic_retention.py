"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_v2b_endpoint_semantic_eval():
    """Run former v2b_endpoint_semantic_eval.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/rl')]
    from rl.experiments.common.utilities.v2b2_semantic_eval import obs_tensor,evaluate,PREFS,ORDER,IDX,PHYS,SUITES,NENV,STEPS
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
        a=ap.parse_args();
        if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
            o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene['robot']
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(a.checkpoint,map_location='cuda',weights_only=False)['model']);m.eval()
            rows=[]
            for suite in range(SUITES):
                seed=840001+suite
                for lab in ('T','A','O','S','C'):
                    q=evaluate(env,m,mgr,robot,PREFS[lab],seed);q.update({'suite':suite,'label':lab});rows.append(q)
            endpoint={};endpoint_pass={}
            for lab in ORDER:
                j=IDX[lab];pk=PHYS[lab];obj_ok=[];phys_ok=[];surv=[];doall=[];dpall=[]
                for suite in range(SUITES):
                    r=next(x for x in rows if x['suite']==suite and x['label']==lab);c=next(x for x in rows if x['suite']==suite and x['label']=='C')
                    do=r['normalized_objective_mean'][j]-c['normalized_objective_mean'][j];dp=r['physical'][pk]-c['physical'][pk]
                    obj_ok.append(do>0);phys_ok.append(dp<0);surv.append(r['survival']);doall.append(do);dpall.append(dp)
                endpoint[lab]={'objective_correct_fraction':float(np.mean(obj_ok)),'physical_correct_fraction':float(np.mean(phys_ok)),
                               'mean_objective_delta_vs_center':float(np.mean(doall)),'mean_physical_delta_vs_center':float(np.mean(dpall)),'min_survival':float(np.min(surv))}
                endpoint_pass[lab]=bool(endpoint[lab]['objective_correct_fraction']>=.75 and endpoint[lab]['physical_correct_fraction']>=.75 and endpoint[lab]['min_survival']>=.95)
            rep={'schema':'v2b_endpoint_semantic_eval_v1','measurement_only':True,'training_updates':0,'checkpoint':str(a.checkpoint.relative_to(ROOT)),
                 'endpoint':endpoint,'endpoint_pass':endpoint_pass,'pass_count':int(sum(endpoint_pass.values())),'endpoint_rows':rows}
            out=a.output_dir/'endpoint_report.json';out.write_text(json.dumps(rep,indent=2)+'\n')
            (a.output_dir/'PROVENANCE_MANIFEST.json').write_text(json.dumps({'status':'FROZEN_BY_HASH','report_sha256':sha(out),'checkpoint_sha256':sha(a.checkpoint),'script_sha256':sha(Path(__file__).resolve())},indent=2)+'\n')
            print(json.dumps({'checkpoint':rep['checkpoint'],'endpoint_pass':endpoint_pass,'pass_count':rep['pass_count'],'endpoint':endpoint},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_l1_der_diverse_replay_gate():
    """Run former v2b_l1_der_diverse_replay_gate.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_l1_der_diverse_replay_gate-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    DER_CAP=32;DER_STATES_PER_TRAJ=8
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REF_PHASES=(0,8,16,24)
    WINDOW=8
    RHO=0.25
    BETA0=2.497041993384243
    KAPPA=0.50
    EPS=1e-12
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps,retain=False):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=retain,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def apply_flat_step(m,direction,step_norm):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.add_(direction[off:off+n].view_as(p)*step_norm);off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w_env.repeat(H,1),"w_env":w_env}
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    def mixed_w(device,k):
        labs=[ORDER[(k+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        vx=(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()
        wz=(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()
        return {
            "tracking_error":vx+wz,
            "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
            "tilt_deg":tilt_deg(data.root_quat_w),
            "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)
        }
    
    def collect_semantic_rollout(env,m,mgr,robot,seed,lab):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+1234+(0 if lab=="C" else IDX.get(lab,0))*17)
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone())
                obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev)
                phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone())
                prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done),"w":w}
    
    def margin_windows(h,c,lab):
        j=IDX[lab];pk=PHYS[lab]
        out=[]
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            # each env is one matched sample; positive means heavy preference is semantically better.
            obj_h=h["obj"][sl,:,j].mean(0);obj_c=c["obj"][sl,:,j].mean(0)
            ph_h=h["phys"][pk][sl].mean(0);ph_c=c["phys"][pk][sl].mean(0)
            surv_h=(~h["done"][sl]).float().mean(0);surv_c=(~c["done"][sl]).float().mean(0)
            out.append({"phase":phase,
                        "obj_margin":(obj_h-obj_c).detach(),
                        "phys_margin":(ph_c-ph_h).detach(),
                        "survival":torch.minimum(surv_h,surv_c).detach()})
        return out
    
    def capture_outcome_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect_semantic_rollout(env,m,mgr,robot,seed,lab)
            c=collect_semantic_rollout(env,m,mgr,robot,seed,"C")
            wins=margin_windows(h,c,lab)
            for w in wins:
                for e in range(NENV):
                    entries.append({
                        "seed":seed,"phase":w["phase"],"env":e,
                        "obj_ref":float(w["obj_margin"][e].cpu()),
                        "phys_ref":float(w["phys_margin"][e].cpu()),
                        "survival_ref":float(w["survival"][e].cpu())})
        return entries
    
    def semantic_outcome_rehearsal_gradient(env,m,mgr,robot,refs):
        nps=actor_named_params(m)
        if not refs:
            z=torch.zeros(sum(p.numel() for _,p in nps),device="cuda")
            return z,{"active_windows":0,"total_windows":0,"mean_obj_deficit":0.0,"mean_phys_deficit":0.0}
        losses=[];objdefs=[];phydefs=[];active=0;total=0
        for lab,entries in refs.items():
            byseed={}
            for x in entries: byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h=collect_semantic_rollout(env,m,mgr,robot,seed,lab)
                c=collect_semantic_rollout(env,m,mgr,robot,seed,"C")
                wins={w["phase"]:w for w in margin_windows(h,c,lab)}
                # differentiable current-policy log-probs for sampled actions.
                oh=h["obs"].reshape(H*NENV,-1); uh=h["u"].reshape(H*NENV,-1)
                oc=c["obs"].reshape(H*NENV,-1); uc=c["u"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lph=m.logp_from_pre_tanh_with_preference(oh,wh,uh).reshape(H,NENV)
                lpc=m.logp_from_pre_tanh_with_preference(oc,wc,uc).reshape(H,NENV)
                for x in ents:
                    total+=1;phase=x["phase"];e=x["env"];cur=wins[phase]
                    mobj=float(cur["obj_margin"][e].cpu()); mphys=float(cur["phys_margin"][e].cpu())
                    obj_valid=x["obj_ref"]>1e-8
                    phys_valid=x["phys_ref"]>1e-8
                    floor_obj=KAPPA*x["obj_ref"] if obj_valid else None
                    floor_phys=KAPPA*x["phys_ref"] if phys_valid else None
                    dobj=max(0.0,floor_obj-mobj) if obj_valid else 0.0
                    dphys=max(0.0,floor_phys-mphys) if phys_valid else 0.0
                    if dobj<=0 and dphys<=0: continue
                    active+=1;objdefs.append(dobj);phydefs.append(dphys)
                    sl=slice(phase,phase+WINDOW)
                    # score-function surrogate for increasing heavy-center semantic margin.
                    # Objective return higher is better; physical metric lower is better.
                    obj_h=float(h["obj"][sl,e,IDX[lab]].mean().cpu())
                    obj_c=float(c["obj"][sl,e,IDX[lab]].mean().cpu())
                    ph_h=float(h["phys"][PHYS[lab]][sl,e].mean().cpu())
                    ph_c=float(c["phys"][PHYS[lab]][sl,e].mean().cpu())
                    logh=lph[sl,e].mean(); logc=lpc[sl,e].mean()
                    sample_loss=torch.tensor(0.0,device="cuda")
                    if dobj>0:
                        sev=dobj/(abs(floor_obj)+1e-6)
                        sample_loss=sample_loss+sev*(-obj_h*logh + obj_c*logc)
                    if dphys>0:
                        sev=dphys/(abs(floor_phys)+1e-6)
                        # maximize phys margin = M_center - M_heavy
                        sample_loss=sample_loss+sev*(-ph_c*logc + ph_h*logh)
                    losses.append(sample_loss)
        if not losses:
            z=torch.zeros(sum(p.numel() for _,p in nps),device="cuda")
            return z,{"active_windows":0,"total_windows":total,"mean_obj_deficit":0.0,"mean_phys_deficit":0.0}
        loss=sum(losses)/len(losses)
        g=flat_grad(loss,nps).detach()
        return g,{"active_windows":active,"total_windows":total,
                  "active_fraction":active/max(total,1),
                  "mean_obj_deficit":float(np.mean(objdefs)) if objdefs else 0.0,
                  "mean_phys_deficit":float(np.mean(phydefs)) if phydefs else 0.0,
                  "loss":float(loss.detach().cpu())}
    
    def action_response_reference(env,m,lab):
        states=[];meta=[]
        for seed in REF_SEEDS[lab]:
            for plab in (lab,"C"):
                w=torch.tensor(PREFS[plab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
                with torch.no_grad():
                    for t in range(H):
                        if t in REF_PHASES:
                            states.append(cur.detach().clone());meta.extend([{"seed":seed,"phase":t,"source_pref":plab} for _ in range(NENV)])
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        s=torch.cat(states,0);wi=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1);wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(s),1)
        with torch.no_grad():
            delta=(m.act_inference_with_preference(s,wi)-m.act_inference_with_preference(s,wc)).detach().clone()
        scale=float(torch.sqrt(torch.mean(delta*delta)).cpu())
        return {"states":s,"wi":wi,"wc":wc,"delta_ref":delta,"scale":max(scale,1e-4),"n_states":len(s)}
    def action_rehearsal_gradient(m,refs):
        nps=actor_named_params(m)
        if not refs: return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"loss":0.0}
        ls=[]
        for r in refs.values():
            d=m.act_inference_with_preference(r["states"],r["wi"])-m.act_inference_with_preference(r["states"],r["wc"])
            ls.append(torch.mean((d-r["delta_ref"])**2)/(r["scale"]**2+1e-12))
        loss=sum(ls)/len(ls);g=flat_grad(loss,nps).detach()
        return g,{"loss":float(loss.detach().cpu())}
    
    def crowding_distance(signatures):
        X=np.asarray(signatures,np.float64)
        n,d=X.shape
        if n<=2:return np.full(n,np.inf)
        cd=np.zeros(n,np.float64)
        for j in range(d):
            order=np.argsort(X[:,j]);lo=X[order[0],j];hi=X[order[-1],j]
            cd[order[0]]=np.inf;cd[order[-1]]=np.inf
            den=hi-lo
            if den<=1e-12:continue
            for k in range(1,n-1):
                if np.isinf(cd[order[k]]):continue
                cd[order[k]]+=(X[order[k+1],j]-X[order[k-1],j])/den
        return cd
    
    def prune_diverse(units,cap=DER_CAP):
        units=list(units)
        while len(units)>cap:
            sig=np.stack([u["signature"] for u in units])
            cd=crowding_distance(sig)
            finite=np.where(np.isfinite(cd))[0]
            if len(finite)==0:units=units[:cap];break
            drop=int(finite[np.argmin(cd[finite])]);units.pop(drop)
        return units
    
    def collect_der_candidates(env,m,mgr,seed,lab):
        from talon_rl.rewards.objectives import normalized_objective_vector
        out=[];idx=np.linspace(0,H-1,DER_STATES_PER_TRAJ,dtype=int)
        for plab in (lab,"C"):
            w=torch.tensor(PREFS[plab],device="cuda").repeat(NENV,1)
            cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];rews=[]
            with torch.no_grad():
                for _ in range(H):
                    a=m.act_inference_with_preference(cur,w)
                    nxt,_,_,_,_=env.step(a)
                    raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt
                    obs.append(cur.detach().clone());rews.append(vec);cur=ot(nxt).cuda()
            O=torch.stack(obs);R=np.asarray(rews)
            disc=(G**np.arange(H))[:,None,None]
            returns=(R*disc).sum(0)
            for e in range(NENV):
                states=O[idx,e].detach().clone()
                wi=torch.tensor(PREFS[lab],device="cuda").repeat(len(states),1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(states),1)
                with torch.no_grad():delta=(m.act_inference_with_preference(states,wi)-m.act_inference_with_preference(states,wc)).detach().clone()
                sig=np.r_[returns[e],PREFS[plab]].astype(np.float64)
                out.append({"states":states,"delta_ref":delta,"signature":sig,"source_pref":plab,"seed":int(seed),"env":int(e)})
        return out
    
    def der_update_memory(env,m,mgr,memory,lab,seeds):
        units=list(memory.get(lab,[]))
        for seed in seeds:units.extend(collect_der_candidates(env,m,mgr,seed,lab))
        memory[lab]=prune_diverse(units,DER_CAP)
    
    def der_rehearsal_gradient(m,memory):
        nps=actor_named_params(m)
        if not memory:return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"loss":0.0,"units":0,"states":0}
        losses=[];units_n=0;states_n=0
        for lab,units in memory.items():
            if not units:continue
            refs=torch.cat([u["delta_ref"] for u in units],0)
            scale=float(torch.sqrt(torch.mean(refs*refs)).cpu());scale=max(scale,1e-4)
            ls=[]
            for u in units:
                s=u["states"];wi=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1);wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(s),1)
                d=m.act_inference_with_preference(s,wi)-m.act_inference_with_preference(s,wc)
                ls.append(torch.mean((d-u["delta_ref"])**2)/(scale**2+1e-12));units_n+=1;states_n+=len(s)
            losses.append(sum(ls)/len(ls))
        if not losses:return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"loss":0.0,"units":0,"states":0}
        loss=sum(losses)/len(losses);g=flat_grad(loss,nps).detach()
        return g,{"loss":float(loss.detach().cpu()),"units":units_n,"states":states_n,
                  "per_axis_units":{lab:len(units) for lab,units in memory.items()}}
    
    def mixed_gradient(m,b):
        weighted,ratio=objective_losses(m,b);loss=sum(weighted.values())
        g=flat_grad(loss,actor_named_params(m)).detach()
        return g,{"mixed_loss":float(loss.detach().cpu()),"ratio_maxerr":float((ratio-1).abs().max().detach().cpu())}
    def combine_bounded(gm,gr):
        gmnorm=float(gm.norm().cpu());grnorm=float(gr.norm().cpu())
        alpha=min(BETA0,RHO*gmnorm/(grnorm+EPS)) if grnorm>EPS else 0.0
        gt=gm+alpha*gr
        d=-gt/(gt.norm()+EPS);dm=-gm/(gm.norm()+EPS)
        return d,{"alpha_t":float(alpha),"mixed_grad_norm":gmnorm,"rehearsal_grad_norm":grnorm,
                  "raw_rehearsal_to_mixed_grad_ratio":grnorm/(gmnorm+EPS),
                  "weighted_rehearsal_to_mixed_grad_ratio":float((alpha*gr).norm().cpu()/(gmnorm+EPS)),
                  "cos_mixed_total":cos(dm,d)}
    
    def rollout(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1);cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(FULL_STEPS):
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                done_any|=(te|tr).cpu().numpy()
                pv=phys_vector(robot,env,a,prev);phys.append({k:float(v.mean().cpu()) for k,v in pv.items()})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    def endpoint_eval(env,m,mgr,robot,eval_seed_base=840001):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=eval_seed_base+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab);c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,"semantic_score":0.5*(of+pf),
                     "pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),"min_survival":float(min(surv))}
        return ep
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1);acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i: vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),"final_score":float(scores[-1]),
                      "first_pass_checkpoint":fp,"retained_pass_fraction_after_first_pass":ret}
        return out
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output-dir",type=Path,default=OUT)
        ap.add_argument("--train-seed",type=int,default=BASE_SEED)
        ap.add_argument("--eval-seed-base",type=int,default=840001)
        args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init=copy.deepcopy(base.state_dict());step_norm=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1);probe=collect_batch(env,base,mgr,args.train_seed,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot,args.eval_seed_base)
            arms=("control","action_response","der_diverse")
            models={};rows={}
            act_refs={};der_memory={}
            activation={"action_response":{lab:None for lab in ORDER},"der_diverse":{lab:None for lab in ORDER}}
            for arm in arms:
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval();models[arm]=m;rows[arm]=[]
            for lab in ORDER:
                if base_ep[lab]["pass"]:
                    act_refs[lab]=action_response_reference(env,models["action_response"],lab)
                    activation["action_response"][lab]=0
                    der_update_memory(env,models["der_diverse"],mgr,der_memory,lab,REF_SEEDS[lab])
                    activation["der_diverse"][lab]=0
            prev={a:None for a in arms}
            for k in range(1,K+1):
                seed=args.train_seed+k*313
                for arm in arms:
                    m=models[arm];_,w=mixed_w(torch.device("cuda"),k);b=collect_batch(env,m,mgr,seed,w)
                    gm,mmeta=mixed_gradient(m,b)
                    if arm=="control":
                        gr=torch.zeros_like(gm);rmeta={"type":"none"}
                    elif arm=="action_response":
                        gr,rmeta=action_rehearsal_gradient(m,act_refs);rmeta["type"]="action_response_static"
                    else:
                        gr,rmeta=der_rehearsal_gradient(m,der_memory);rmeta["type"]="der_diverse_deltaa"
                    d,bmeta=combine_bounded(gm,gr)
                    cp=None if prev[arm] is None else cos(d,prev[arm]);prev[arm]=d.detach().clone()
                    apply_flat_step(m,d,step_norm)
                    ep=endpoint_eval(env,m,mgr,robot,args.eval_seed_base)
                    new=[]
                    if arm=="action_response":
                        for lab in ORDER:
                            if ep[lab]["pass"] and lab not in act_refs:
                                act_refs[lab]=action_response_reference(env,m,lab)
                                activation["action_response"][lab]=k;new.append(lab)
                        active_before=sorted(set(act_refs)-set(new))
                    elif arm=="der_diverse":
                        for lab in ORDER:
                            if ep[lab]["pass"]:
                                if lab not in der_memory:
                                    der_update_memory(env,m,mgr,der_memory,lab,REF_SEEDS[lab]);activation["der_diverse"][lab]=k;new.append(lab)
                                else:
                                    der_update_memory(env,m,mgr,der_memory,lab,[seed+50000+IDX[lab]*101])
                        active_before=sorted(set(der_memory)-set(new))
                    else:
                        active_before=[]
                    rows[arm].append({"update":k,"active_refs_before_update":active_before,
                        "new_axes_activated":new,"step_norm":step_norm,"mixed_meta":mmeta,"rehearsal_meta":rmeta,"budget_meta":bmeta,
                        "update_cos_prev":cp,"preference_separation":preference_separation(m,probe),"endpoint":ep,
                        "semantic_scores":{lab:ep[lab]["semantic_score"] for lab in ORDER}})
                    torch.save({"model":m.state_dict(),"arm":arm,"update":k},args.output_dir/f"{arm}_u{k}.pt")
                (args.output_dir/"l1_der_partial.json").write_text(json.dumps({"activation":activation,"active":{"action_response":sorted(act_refs),"der_diverse":sorted(der_memory)},"der_memory_units":{lab:len(v) for lab,v in der_memory.items()},"arms":rows},indent=2)+"\n")
            report={"schema":"v2b_l1_der_diverse_replay_gate_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "train_seed":args.train_seed,"eval_seed_base":args.eval_seed_base,
                    "rho":RHO,"beta_cap":BETA0,"reference_phases":list(REF_PHASES),
                    "reference_seeds":REF_SEEDS,"step_norm":step_norm,"updates":K,"baseline_endpoint":base_ep,"activation":activation,
                    "arms":rows,"retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            for arm in arms:
                rr=rows[arm]
                report.setdefault("arm_summary",{})[arm]={
                    "pass_events":int(sum(int(x["endpoint"][lab]["pass"]) for x in rr for lab in ORDER)),
                    "semantic_score_mean":float(np.mean([x["semantic_scores"][lab] for x in rr for lab in ORDER])),
                    "preference_separation_mean":float(np.mean([x["preference_separation"] for x in rr])),
                    "mean_cos_mixed_total":float(np.mean([x["budget_meta"]["cos_mixed_total"] for x in rr])),
                    "mean_weighted_rehearsal_ratio":float(np.mean([x["budget_meta"]["weighted_rehearsal_to_mixed_grad_ratio"] for x in rr])),
                }
            out=args.output_dir/"l1_der_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),
              "script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"]},indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True: main()

def run_v2b_l2_policy_consolidation_gate():
    """Run former v2b_l2_policy_consolidation_gate.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_l2_policy_consolidation_gate-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REF_PHASES=(0,8,16,24)
    WINDOW=8
    RHO=0.25
    BETA0=2.497041993384243
    KAPPA=0.50
    EPS=1e-12
    PC_PERIODS=(1,2,4)
    PC_WEIGHTS=(1.0,2.0,4.0)
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps,retain=False):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=retain,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def apply_flat_step(m,direction,step_norm):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.add_(direction[off:off+n].view_as(p)*step_norm);off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w_env.repeat(H,1),"w_env":w_env}
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    def mixed_w(device,k):
        labs=[ORDER[(k+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        vx=(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()
        wz=(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()
        return {
            "tracking_error":vx+wz,
            "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
            "tilt_deg":tilt_deg(data.root_quat_w),
            "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)
        }
    
    def collect_semantic_rollout(env,m,mgr,robot,seed,lab):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+1234+(0 if lab=="C" else IDX.get(lab,0))*17)
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone())
                obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev)
                phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone())
                prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done),"w":w}
    
    def margin_windows(h,c,lab):
        j=IDX[lab];pk=PHYS[lab]
        out=[]
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            # each env is one matched sample; positive means heavy preference is semantically better.
            obj_h=h["obj"][sl,:,j].mean(0);obj_c=c["obj"][sl,:,j].mean(0)
            ph_h=h["phys"][pk][sl].mean(0);ph_c=c["phys"][pk][sl].mean(0)
            surv_h=(~h["done"][sl]).float().mean(0);surv_c=(~c["done"][sl]).float().mean(0)
            out.append({"phase":phase,
                        "obj_margin":(obj_h-obj_c).detach(),
                        "phys_margin":(ph_c-ph_h).detach(),
                        "survival":torch.minimum(surv_h,surv_c).detach()})
        return out
    
    def capture_outcome_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect_semantic_rollout(env,m,mgr,robot,seed,lab)
            c=collect_semantic_rollout(env,m,mgr,robot,seed,"C")
            wins=margin_windows(h,c,lab)
            for w in wins:
                for e in range(NENV):
                    entries.append({
                        "seed":seed,"phase":w["phase"],"env":e,
                        "obj_ref":float(w["obj_margin"][e].cpu()),
                        "phys_ref":float(w["phys_margin"][e].cpu()),
                        "survival_ref":float(w["survival"][e].cpu())})
        return entries
    
    def semantic_outcome_rehearsal_gradient(env,m,mgr,robot,refs):
        nps=actor_named_params(m)
        if not refs:
            z=torch.zeros(sum(p.numel() for _,p in nps),device="cuda")
            return z,{"active_windows":0,"total_windows":0,"mean_obj_deficit":0.0,"mean_phys_deficit":0.0}
        losses=[];objdefs=[];phydefs=[];active=0;total=0
        for lab,entries in refs.items():
            byseed={}
            for x in entries: byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h=collect_semantic_rollout(env,m,mgr,robot,seed,lab)
                c=collect_semantic_rollout(env,m,mgr,robot,seed,"C")
                wins={w["phase"]:w for w in margin_windows(h,c,lab)}
                # differentiable current-policy log-probs for sampled actions.
                oh=h["obs"].reshape(H*NENV,-1); uh=h["u"].reshape(H*NENV,-1)
                oc=c["obs"].reshape(H*NENV,-1); uc=c["u"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lph=m.logp_from_pre_tanh_with_preference(oh,wh,uh).reshape(H,NENV)
                lpc=m.logp_from_pre_tanh_with_preference(oc,wc,uc).reshape(H,NENV)
                for x in ents:
                    total+=1;phase=x["phase"];e=x["env"];cur=wins[phase]
                    mobj=float(cur["obj_margin"][e].cpu()); mphys=float(cur["phys_margin"][e].cpu())
                    obj_valid=x["obj_ref"]>1e-8
                    phys_valid=x["phys_ref"]>1e-8
                    floor_obj=KAPPA*x["obj_ref"] if obj_valid else None
                    floor_phys=KAPPA*x["phys_ref"] if phys_valid else None
                    dobj=max(0.0,floor_obj-mobj) if obj_valid else 0.0
                    dphys=max(0.0,floor_phys-mphys) if phys_valid else 0.0
                    if dobj<=0 and dphys<=0: continue
                    active+=1;objdefs.append(dobj);phydefs.append(dphys)
                    sl=slice(phase,phase+WINDOW)
                    # score-function surrogate for increasing heavy-center semantic margin.
                    # Objective return higher is better; physical metric lower is better.
                    obj_h=float(h["obj"][sl,e,IDX[lab]].mean().cpu())
                    obj_c=float(c["obj"][sl,e,IDX[lab]].mean().cpu())
                    ph_h=float(h["phys"][PHYS[lab]][sl,e].mean().cpu())
                    ph_c=float(c["phys"][PHYS[lab]][sl,e].mean().cpu())
                    logh=lph[sl,e].mean(); logc=lpc[sl,e].mean()
                    sample_loss=torch.tensor(0.0,device="cuda")
                    if dobj>0:
                        sev=dobj/(abs(floor_obj)+1e-6)
                        sample_loss=sample_loss+sev*(-obj_h*logh + obj_c*logc)
                    if dphys>0:
                        sev=dphys/(abs(floor_phys)+1e-6)
                        # maximize phys margin = M_center - M_heavy
                        sample_loss=sample_loss+sev*(-ph_c*logc + ph_h*logh)
                    losses.append(sample_loss)
        if not losses:
            z=torch.zeros(sum(p.numel() for _,p in nps),device="cuda")
            return z,{"active_windows":0,"total_windows":total,"mean_obj_deficit":0.0,"mean_phys_deficit":0.0}
        loss=sum(losses)/len(losses)
        g=flat_grad(loss,nps).detach()
        return g,{"active_windows":active,"total_windows":total,
                  "active_fraction":active/max(total,1),
                  "mean_obj_deficit":float(np.mean(objdefs)) if objdefs else 0.0,
                  "mean_phys_deficit":float(np.mean(phydefs)) if phydefs else 0.0,
                  "loss":float(loss.detach().cpu())}
    
    def action_response_reference(env,m,lab):
        states=[];meta=[]
        for seed in REF_SEEDS[lab]:
            for plab in (lab,"C"):
                w=torch.tensor(PREFS[plab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
                with torch.no_grad():
                    for t in range(H):
                        if t in REF_PHASES:
                            states.append(cur.detach().clone());meta.extend([{"seed":seed,"phase":t,"source_pref":plab} for _ in range(NENV)])
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        s=torch.cat(states,0);wi=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1);wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(s),1)
        with torch.no_grad():
            delta=(m.act_inference_with_preference(s,wi)-m.act_inference_with_preference(s,wc)).detach().clone()
        scale=float(torch.sqrt(torch.mean(delta*delta)).cpu())
        return {"states":s,"wi":wi,"wc":wc,"delta_ref":delta,"scale":max(scale,1e-4),"n_states":len(s)}
    def action_rehearsal_gradient(m,refs):
        nps=actor_named_params(m)
        if not refs: return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"loss":0.0}
        ls=[]
        for r in refs.values():
            d=m.act_inference_with_preference(r["states"],r["wi"])-m.act_inference_with_preference(r["states"],r["wc"])
            ls.append(torch.mean((d-r["delta_ref"])**2)/(r["scale"]**2+1e-12))
        loss=sum(ls)/len(ls);g=flat_grad(loss,nps).detach()
        return g,{"loss":float(loss.detach().cpu())}
    
    def gaussian_kl_current_to_teacher(cur_loc,cur_logstd,tea_loc,tea_logstd):
        cs=torch.exp(cur_logstd);ts=torch.exp(tea_logstd)
        v=(tea_logstd-cur_logstd)+(cs*cs+(cur_loc-tea_loc)**2)/(2*ts*ts)-0.5
        return v.sum(-1).mean()
    
    def pc_gradient(m,b,teachers):
        nps=actor_named_params(m)
        if not teachers:
            return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"loss":0.0,"per_teacher":[]}
        obs=b["obs"];w=b["w"]
        cur=m._pre_tanh_dist_with_preference(obs,w);cur_loc=cur.loc;cur_logstd=m.log_std.view(1,-1).expand_as(cur_loc)
        vals=[];meta=[];den=float(sum(PC_WEIGHTS))
        for weight,tm,period in zip(PC_WEIGHTS,teachers,PC_PERIODS):
            with torch.no_grad():
                td=tm._pre_tanh_dist_with_preference(obs,w);tl=td.loc.detach();tls=tm.log_std.detach().view(1,-1).expand_as(tl)
            kl=gaussian_kl_current_to_teacher(cur_loc,cur_logstd,tl,tls)
            vals.append((weight/den)*kl);meta.append({"period":period,"weight":weight,"kl":float(kl.detach().cpu())})
        loss=sum(vals);g=flat_grad(loss,nps).detach()
        return g,{"loss":float(loss.detach().cpu()),"per_teacher":meta}
    
    def refresh_pc_teachers(student,teachers,k):
        refreshed=[]
        for period,tm in zip(PC_PERIODS,teachers):
            if k%period==0:
                tm.load_state_dict(copy.deepcopy(student.state_dict()));tm.eval();refreshed.append(period)
        return refreshed
    
    def mixed_gradient(m,b):
        weighted,ratio=objective_losses(m,b);loss=sum(weighted.values())
        g=flat_grad(loss,actor_named_params(m)).detach()
        return g,{"mixed_loss":float(loss.detach().cpu()),"ratio_maxerr":float((ratio-1).abs().max().detach().cpu())}
    def combine_bounded(gm,gr):
        gmnorm=float(gm.norm().cpu());grnorm=float(gr.norm().cpu())
        alpha=min(BETA0,RHO*gmnorm/(grnorm+EPS)) if grnorm>EPS else 0.0
        gt=gm+alpha*gr
        d=-gt/(gt.norm()+EPS);dm=-gm/(gm.norm()+EPS)
        return d,{"alpha_t":float(alpha),"mixed_grad_norm":gmnorm,"rehearsal_grad_norm":grnorm,
                  "raw_rehearsal_to_mixed_grad_ratio":grnorm/(gmnorm+EPS),
                  "weighted_rehearsal_to_mixed_grad_ratio":float((alpha*gr).norm().cpu()/(gmnorm+EPS)),
                  "cos_mixed_total":cos(dm,d)}
    
    def rollout(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1);cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(FULL_STEPS):
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                done_any|=(te|tr).cpu().numpy()
                pv=phys_vector(robot,env,a,prev);phys.append({k:float(v.mean().cpu()) for k,v in pv.items()})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    def endpoint_eval(env,m,mgr,robot,eval_seed_base=840001):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=eval_seed_base+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab);c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,"semantic_score":0.5*(of+pf),
                     "pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),"min_survival":float(min(surv))}
        return ep
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1);acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i: vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),"final_score":float(scores[-1]),
                      "first_pass_checkpoint":fp,"retained_pass_fraction_after_first_pass":ret}
        return out
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output-dir",type=Path,default=OUT)
        ap.add_argument("--train-seed",type=int,default=BASE_SEED)
        ap.add_argument("--eval-seed-base",type=int,default=840001)
        args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init=copy.deepcopy(base.state_dict());step_norm=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1);probe=collect_batch(env,base,mgr,args.train_seed,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot,args.eval_seed_base)
            arms=("control","action_response","policy_consolidation")
            models={};rows={}
            act_refs={}
            activation={"action_response":{lab:None for lab in ORDER}}
            pc_teachers=[]
            for arm in arms:
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval();models[arm]=m;rows[arm]=[]
            for _ in PC_PERIODS:
                tm=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();tm.load_state_dict(init);tm.eval()
                for p in tm.parameters():p.requires_grad_(False)
                pc_teachers.append(tm)
            for lab in ORDER:
                if base_ep[lab]["pass"]:
                    act_refs[lab]=action_response_reference(env,models["action_response"],lab)
                    activation["action_response"][lab]=0
            prev={a:None for a in arms}
            for k in range(1,K+1):
                seed=args.train_seed+k*313
                for arm in arms:
                    m=models[arm];_,w=mixed_w(torch.device("cuda"),k);b=collect_batch(env,m,mgr,seed,w)
                    gm,mmeta=mixed_gradient(m,b)
                    if arm=="control":
                        gr=torch.zeros_like(gm);rmeta={"type":"none"}
                    elif arm=="action_response":
                        gr,rmeta=action_rehearsal_gradient(m,act_refs);rmeta["type"]="action_response_static"
                    else:
                        gr,rmeta=pc_gradient(m,b,pc_teachers);rmeta["type"]="policy_consolidation"
                    d,bmeta=combine_bounded(gm,gr)
                    cp=None if prev[arm] is None else cos(d,prev[arm]);prev[arm]=d.detach().clone()
                    apply_flat_step(m,d,step_norm)
                    refreshed=refresh_pc_teachers(m,pc_teachers,k) if arm=="policy_consolidation" else []
                    ep=endpoint_eval(env,m,mgr,robot,args.eval_seed_base)
                    new=[]
                    if arm=="action_response":
                        for lab in ORDER:
                            if ep[lab]["pass"] and lab not in act_refs:
                                act_refs[lab]=action_response_reference(env,m,lab)
                                activation["action_response"][lab]=k;new.append(lab)
                        active_before=sorted(set(act_refs)-set(new))
                    else:
                        active_before=[]
                    rows[arm].append({"update":k,"active_refs_before_update":active_before,
                        "new_axes_activated":new,"step_norm":step_norm,"mixed_meta":mmeta,"rehearsal_meta":rmeta,"budget_meta":bmeta,
                        "update_cos_prev":cp,"preference_separation":preference_separation(m,probe),"endpoint":ep,
                        "pc_refreshed_periods":refreshed,
                        "semantic_scores":{lab:ep[lab]["semantic_score"] for lab in ORDER}})
                    torch.save({"model":m.state_dict(),"arm":arm,"update":k},args.output_dir/f"{arm}_u{k}.pt")
                (args.output_dir/"l2_pc_partial.json").write_text(json.dumps({"activation":activation,"active":{"action_response":sorted(act_refs)},"arms":rows},indent=2)+"\n")
            report={"schema":"v2b_l2_policy_consolidation_gate_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "train_seed":args.train_seed,"eval_seed_base":args.eval_seed_base,
                    "rho":RHO,"beta_cap":BETA0,"pc_periods":list(PC_PERIODS),"pc_weights":list(PC_WEIGHTS),"reference_phases":list(REF_PHASES),
                    "reference_seeds":REF_SEEDS,"step_norm":step_norm,"updates":K,"baseline_endpoint":base_ep,"activation":activation,
                    "arms":rows,"retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            for arm in arms:
                rr=rows[arm]
                report.setdefault("arm_summary",{})[arm]={
                    "pass_events":int(sum(int(x["endpoint"][lab]["pass"]) for x in rr for lab in ORDER)),
                    "semantic_score_mean":float(np.mean([x["semantic_scores"][lab] for x in rr for lab in ORDER])),
                    "preference_separation_mean":float(np.mean([x["preference_separation"] for x in rr])),
                    "mean_cos_mixed_total":float(np.mean([x["budget_meta"]["cos_mixed_total"] for x in rr])),
                    "mean_weighted_rehearsal_ratio":float(np.mean([x["budget_meta"]["weighted_rehearsal_to_mixed_grad_ratio"] for x in rr])),
                }
            out=args.output_dir/"l2_pc_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),
              "script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"]},indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True: main()

def run_v2b_o_trajectory_realization_audit():
    """Run former v2b_o_trajectory_realization_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_o_trajectory_realization_audit-2026-09-23"
    PREFS={
    "O":np.array([.1,.1,.7,.1],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32),
    }
    G=.99;LAM=.95;H=64;NENV=8
    SUITES=4
    RESET_SEEDS=[880001+i for i in range(SUITES)]
    NOISE_SEEDS=[990001+i for i in range(SUITES)]
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    
    def rtg(x,done):
        out=torch.zeros_like(x);run=torch.zeros_like(x[-1])
        for t in range(len(x)-1,-1,-1):
            run=x[t]+G*run*(~done[t]).to(x.dtype)
            out[t]=run
        return out
    
    def collect(env,m,mgr,robot,w_np,reset_seed,noise_seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        from talon_rl.models.foundations.four_objective import vector_gae
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=reset_seed);cur=ot(cur).cuda()
        torch.manual_seed(noise_seed)
    
        obs=[];u=[];old=[];R=[];D=[];vals=[]
        logs={k:[] for k in (
            "tilt_deg","ang_vel_xy","abs_vz","action_norm","action_rate",
            "gamma_norm","beta_norm","o_reward","joint_pos_norm","joint_vel_norm")}
        actions=[];means=[];stds=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
    
        with torch.no_grad():
            for t in range(H):
                actor_in=m._with_w(cur,w)
                h=m.actor_body(actor_in)
                gb=m.preference_film(w);gamma,beta=torch.chunk(gb,2,dim=-1)
                hfilm=(1+gamma)*h+beta
                emb=m.preference_embedding(w)
                mean=m.actor_mean(torch.cat((hfilm,emb),dim=-1))
                std=m.log_std.exp().expand_as(mean)
                dist=torch.distributions.Normal(mean,std)
                uu=dist.sample()
                a=torch.tanh(uu)*m.ACTION_CLIP
                lp=(dist.log_prob(uu)-m._log_det_jacobian(uu)).sum(-1)
    
                vals.append(m.value_with_preference(cur,w))
                data=robot.data
                logs["tilt_deg"].append(tilt_deg(data.root_quat_w).clone())
                logs["ang_vel_xy"].append(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).clone())
                logs["abs_vz"].append(data.root_lin_vel_b[:,2].abs().clone())
                logs["action_norm"].append(torch.linalg.vector_norm(a,dim=-1))
                logs["action_rate"].append(torch.linalg.vector_norm(a-prev,dim=-1))
                logs["gamma_norm"].append(torch.linalg.vector_norm(gamma,dim=-1))
                logs["beta_norm"].append(torch.linalg.vector_norm(beta,dim=-1))
                logs["joint_pos_norm"].append(torch.linalg.vector_norm(data.joint_pos,dim=-1).clone())
                logs["joint_vel_norm"].append(torch.linalg.vector_norm(data.joint_vel,dim=-1).clone())
    
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                rr=torch.tensor(vec,device="cuda")*env.unwrapped.step_dt
                logs["o_reward"].append(rr[:,2].clone())
    
                obs.append(cur);u.append(uu);old.append(lp);R.append(rr);D.append((te|tr).cuda())
                actions.append(a);means.append(mean);stds.append(std);prev=a;cur=ot(nxt).cuda()
    
            nextv=m.value_with_preference(cur,w)
    
        obsT=torch.stack(obs);uT=torch.stack(u);oldT=torch.stack(old);RT=torch.stack(R);DT=torch.stack(D).bool()
        VT=torch.stack(vals)
        adv,_=vector_gae(RT,VT,nextv,DT,lam=LAM)
        mc=rtg(RT[:,:,2],DT)
        # score wrt Gaussian mean for fixed pre-tanh sample; tanh Jacobian has no mean term.
        meanT=torch.stack(means);stdT=torch.stack(stds)
        score_mean=(uT-meanT)/(stdT*stdT)
        gae_score=score_mean*adv[:,:,2].unsqueeze(-1)
        mc_centered=mc-mc.mean()
        mc_score=score_mean*mc_centered.unsqueeze(-1)
    
        # per-time aggregate direction and alignment between GAE and MC score contributions.
        gae_vec=gae_score.mean(1);mc_vec=mc_score.mean(1)
        cos_time=[]
        for t in range(H):
            a=gae_vec[t];b=mc_vec[t]
            den=float(a.norm()*b.norm())
            cos_time.append(float(torch.dot(a,b)/den) if den>1e-12 else 0.0)
    
        out={k:torch.stack(v).cpu().numpy() for k,v in logs.items()}
        out.update({
          "action":torch.stack(actions).cpu().numpy(),
          "gae_o":adv[:,:,2].cpu().numpy(),
          "mc_o":mc.cpu().numpy(),
          "gae_score_norm":torch.linalg.vector_norm(gae_score,dim=-1).cpu().numpy(),
          "mc_score_norm":torch.linalg.vector_norm(mc_score,dim=-1).cpu().numpy(),
          "gae_mc_score_cos_time":np.asarray(cos_time,np.float32),
          "done":DT.cpu().numpy(),
        })
        return out
    
    def paired_stats(O,C,key):
        # Difference O-heavy - center over suites/envs, preserving time.
        od=np.concatenate([x[key] for x in O],axis=1) if O[0][key].ndim>=2 else np.stack([x[key] for x in O])
        cd=np.concatenate([x[key] for x in C],axis=1) if C[0][key].ndim>=2 else np.stack([x[key] for x in C])
        if od.ndim==3: # vector actions [T, ensemble, D]
            diff=od-cd
            mag=np.linalg.norm(diff,axis=-1)
            return {"mean_action_delta_norm":mag.mean(1).tolist(),
                    "p50_action_delta_norm":np.median(mag,axis=1).tolist()}
        diff=od-cd
        return {"mean_delta":diff.mean(1).tolist(),
                "sem_delta":(diff.std(1,ddof=1)/np.sqrt(diff.shape[1])).tolist(),
                "o_mean":od.mean(1).tolist(),"c_mean":cd.mean(1).tolist()}
    
    def first_sustained(mean_delta,sem_delta,direction,run=3,z=1.0):
        # direction: "positive" or "negative"; require 3 consecutive time points beyond z*SEM.
        md=np.asarray(mean_delta);se=np.asarray(sem_delta)
        good=(md>z*se) if direction=="positive" else (md<-z*se)
        for t in range(len(good)-run+1):
            if np.all(good[t:t+run]):return int(t)
        return None
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
    
            OO=[];CC=[]
            for rs,ns in zip(RESET_SEEDS,NOISE_SEEDS):
                OO.append(collect(env,m,mgr,robot,PREFS["O"],rs,ns))
                CC.append(collect(env,m,mgr,robot,PREFS["C"],rs,ns))
    
            scalar_keys=["tilt_deg","ang_vel_xy","abs_vz","action_norm","action_rate","gamma_norm","beta_norm",
                         "o_reward","joint_pos_norm","joint_vel_norm","gae_o","mc_o","gae_score_norm","mc_score_norm"]
            stats={k:paired_stats(OO,CC,k) for k in scalar_keys}
            stats["action"]=paired_stats(OO,CC,"action")
    
            first={
              "action_change":first_sustained(stats["action_norm"]["mean_delta"],stats["action_norm"]["sem_delta"],"positive"),
              "o_reward_better":first_sustained(stats["o_reward"]["mean_delta"],stats["o_reward"]["sem_delta"],"positive"),
              "tilt_better":first_sustained(stats["tilt_deg"]["mean_delta"],stats["tilt_deg"]["sem_delta"],"negative"),
              "tilt_worse":first_sustained(stats["tilt_deg"]["mean_delta"],stats["tilt_deg"]["sem_delta"],"positive"),
              "ang_vel_xy_better":first_sustained(stats["ang_vel_xy"]["mean_delta"],stats["ang_vel_xy"]["sem_delta"],"negative"),
              "ang_vel_xy_worse":first_sustained(stats["ang_vel_xy"]["mean_delta"],stats["ang_vel_xy"]["sem_delta"],"positive"),
              "vz_worse":first_sustained(stats["abs_vz"]["mean_delta"],stats["abs_vz"]["sem_delta"],"positive"),
            }
    
            # GAE-vs-MC score alignment time profile by preference.
            def cos_profile(rows):
                return np.mean(np.stack([x["gae_mc_score_cos_time"] for x in rows]),axis=0)
            score_cos={"O":cos_profile(OO).tolist(),"C":cos_profile(CC).tolist()}
    
            # 8-step windows make the causal sequence easier to inspect.
            windows=[]
            for st in range(0,H,8):
                en=min(st+8,H)
                row={"start":st,"end":en-1}
                for k in ("tilt_deg","ang_vel_xy","abs_vz","o_reward","action_rate","action_norm","gae_o","mc_o","gae_score_norm"):
                    row[k+"_delta_mean"]=float(np.mean(stats[k]["mean_delta"][st:en]))
                row["gae_mc_score_cos_O"]=float(np.mean(score_cos["O"][st:en]))
                windows.append(row)
    
            report={"schema":"v2b_o_trajectory_realization_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "checkpoint":str(a.checkpoint.relative_to(ROOT)),
                    "matched_design":{"suites":SUITES,"envs_per_suite":NENV,"steps":H,
                                      "reset_seeds":RESET_SEEDS,"policy_noise_seeds":NOISE_SEEDS,
                                      "same_noise_seed_O_vs_center":True},
                    "first_sustained_divergence":first,"windows":windows,
                    "paired_time_series":stats,"score_alignment_time":score_cos}
            out=a.output_dir/"o_trajectory_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH",
                "report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps({"first_sustained_divergence":first,"windows":windows},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_reference_gradient_retention_gate():
    """Run former v2b_reference_gradient_retention_gate.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_reference_gradient_retention_gate-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    REF_SEEDS={"T":991001,"A":992001,"O":993001,"S":994001}
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def apply_flat_step(m,direction,step_norm):
        nps=actor_named_params(m);off=0
        with torch.no_grad():
            for _,p in nps:
                n=p.numel();p.add_(direction[off:off+n].view_as(p)*step_norm);off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w_env.repeat(H,1),"w_env":w_env}
    
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    
    def mixed_w(device,update_idx):
        labs=[ORDER[(update_idx+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def mixed_direction(m,b):
        nps=actor_named_params(m);weighted,ratio=objective_losses(m,b)
        loss=sum(weighted.values());g=flat_grad(loss,nps).detach()
        d=-g/(g.norm()+1e-12)
        return d,float(g.norm().cpu()),float((ratio-1).abs().max().detach().cpu())
    
    def reference_direction(env,m,mgr,lab):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        b=collect_batch(env,m,mgr,REF_SEEDS[lab],w)
        nps=actor_named_params(m);weighted,ratio=objective_losses(m,b)
        g=flat_grad(weighted[lab],nps).detach()
        d=-g/(g.norm()+1e-12)
        return d,float(g.norm().cpu()),float((ratio-1).abs().max().detach().cpu())
    
    def project_retention(d,refs,max_cycles=16,tol=1e-8):
        if not refs:
            return d,{"projected":False,"pre_dot":{},"post_dot":{},"cycles":0,"cos_raw_projected":1.0}
        x=d.clone()
        pre={lab:float(torch.dot(x,r).cpu()) for lab,r in refs.items()}
        projected=any(v<0 for v in pre.values())
        cycles=0
        if projected:
            for cyc in range(max_cycles):
                changed=False
                for lab in ORDER:
                    if lab not in refs:continue
                    r=refs[lab]
                    dot=torch.dot(x,r)
                    if float(dot)<-tol:
                        x=x-dot*r
                        changed=True
                cycles=cyc+1
                if not changed:break
        post={lab:float(torch.dot(x,r).cpu()) for lab,r in refs.items()}
        if float(x.norm())<1e-12:
            x=d.clone()
        else:
            x=x/x.norm()
        return x,{"projected":projected,"pre_dot":pre,"post_dot":post,"cycles":cycles,
                  "cos_raw_projected":cos(d,x)}
    
    def rollout(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(FULL_STEPS):
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();done_any|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    
    def endpoint_eval(env,m,mgr,robot):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];dobj=[];dphy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                do=r["objective_mean"][j]-c["objective_mean"][j];dp=r["physical"][pk]-c["physical"][pk]
                obj.append(do>0);phy.append(dp<0);dobj.append(do);dphy.append(dp);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,
                     "semantic_score":0.5*(of+pf),
                     "pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),
                     "mean_objective_delta_vs_center":float(np.mean(dobj)),
                     "mean_physical_delta_vs_center":float(np.mean(dphy)),
                     "min_survival":float(min(surv))}
        return ep
    
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[];pair={}
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:
                    v=float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu())
                    pair[f"{a}-{b}"]=v;vals.append(v)
        return {"pairwise":pair,"mean_pairwise":float(np.mean(vals))}
    def displacement_from_init(m,init_state):
        return float(torch.cat([(p.detach()-init_state[n].to(p.device)).reshape(-1) for n,p in actor_named_params(m)]).norm().cpu())
    
    def retention_stats(baseline_ep,rows):
        stats={}
        for lab in ORDER:
            scores=[baseline_ep[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[baseline_ep[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            running_best=np.maximum.accumulate(scores)
            forgetting=[float(running_best[i]-scores[i]) for i in range(len(scores))]
            first_pass=next((i for i,v in enumerate(passes) if v),None)
            if first_pass is None or first_pass==len(passes)-1:
                retained_after=None if first_pass is None else 1.0
            else:
                retained_after=float(np.mean(passes[first_pass+1:]))
            stats[lab]={"scores":scores,"passes":passes,
                        "max_forgetting":float(max(forgetting)),
                        "final_forgetting_from_best":float(running_best[-1]-scores[-1]),
                        "first_pass_checkpoint":first_pass,
                        "retained_pass_fraction_after_first_pass":retained_after,
                        "best_score":float(max(scores)),"final_score":float(scores[-1])}
        return stats
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init_state=copy.deepcopy(base.state_dict())
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
            pb=collect_batch(env,base,mgr,BASE_SEED,wb);probe=pb["obs"][:64].detach().clone()
            step_norm=STEP_SCALE*nominal_step()
            baseline_ep=endpoint_eval(env,base,mgr,robot)
            baseline_sep=preference_separation(base,probe)
    
            models={};rows={};prev_dirs={}
            for arm in ("control","retention"):
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init_state);m.eval()
                models[arm]=m;rows[arm]=[];prev_dirs[arm]=None
            active={lab for lab in ORDER if baseline_ep[lab]["pass"]}
            activation={lab:(0 if lab in active else None) for lab in ORDER}
    
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                for arm,m in models.items():
                    labs,w=mixed_w(torch.device("cuda"),k)
                    b=collect_batch(env,m,mgr,seed,w)
                    raw_d,gnorm,ratioerr=mixed_direction(m,b)
                    ref_meta={};proj_meta={"projected":False,"pre_dot":{},"post_dot":{},"cycles":0,"cos_raw_projected":1.0}
                    d=raw_d
                    active_before=sorted(active) if arm=="retention" else []
                    if arm=="retention" and active:
                        refs={}
                        for lab in ORDER:
                            if lab in active:
                                rd,rgn,rratio=reference_direction(env,m,mgr,lab)
                                refs[lab]=rd
                                ref_meta[lab]={"grad_norm":rgn,"ratio_maxerr":rratio}
                        d,proj_meta=project_retention(raw_d,refs)
                    update_cos=None if prev_dirs[arm] is None else cos(d,prev_dirs[arm])
                    apply_flat_step(m,d,step_norm);prev_dirs[arm]=d.clone()
                    ep=endpoint_eval(env,m,mgr,robot)
                    if arm=="retention":
                        for lab in ORDER:
                            if ep[lab]["pass"] and lab not in active:
                                active.add(lab);activation[lab]=k
                    sep=preference_separation(m,probe)
                    row={"update":k,"batch_labels":labs,"step_norm":step_norm,
                         "raw_grad_norm":gnorm,"ratio_maxerr":ratioerr,
                         "raw_to_applied_cosine":cos(raw_d,d),
                         "update_cos_prev":update_cos,
                         "active_references_before_update":active_before,
                         "reference_meta":ref_meta,"projection":proj_meta,
                         "parameter_displacement_from_theta0":displacement_from_init(m,init_state),
                         "preference_separation":sep,"endpoint":ep,
                         "semantic_scores":{lab:ep[lab]["semantic_score"] for lab in ORDER}}
                    rows[arm].append(row)
                    torch.save({"model":m.state_dict(),"arm":arm,"update":k,"step_norm":step_norm},args.output_dir/f"{arm}_u{k}.pt")
                partial={"schema":"v2b_reference_gradient_retention_gate_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
                         "step_norm":step_norm,"baseline_endpoint":baseline_ep,"activation":activation,
                         "active_references_after_update":sorted(active),"arms":rows}
                (args.output_dir/"retention_gate_partial.json").write_text(json.dumps(partial,indent=2)+"\n")
    
            report={"schema":"v2b_reference_gradient_retention_gate_v1","measurement_only":True,
                    "base_checkpoint":str(CKPT.relative_to(ROOT)),"lambda":LAM,"updates":K,
                    "step_scale":STEP_SCALE,"step_norm":step_norm,"nominal_step_norm":nominal_step(),
                    "fresh_on_policy_mixed_batch_each_update":True,
                    "fixed_reference_reset_seeds":REF_SEEDS,
                    "projection_rule":"cyclic half-space projection requiring applied descent dot current-policy retained-axis descent >= 0; then renormalize to identical step norm",
                    "baseline_endpoint":baseline_ep,"baseline_preference_separation":baseline_sep,
                    "activation":activation,"arms":rows,
                    "retention_stats":{arm:retention_stats(baseline_ep,rr) for arm,rr in rows.items()}}
            out=args.output_dir/"reference_gradient_retention_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({
              "status":"FROZEN_BY_HASH","report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),
              "base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            compact={"activation":activation}
            for arm,rr in rows.items():
                compact[arm]=[{"u":r["update"],"active":r["active_references_before_update"],
                               "proj":r["projection"]["projected"],"raw_proj_cos":r["raw_to_applied_cosine"],
                               "scores":r["semantic_scores"],"passes":{lab:r["endpoint"][lab]["pass"] for lab in ORDER}}
                              for r in rr]
            compact["retention_stats"]=report["retention_stats"]
            print(json.dumps(compact,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_semantic_credit_proxy_audit():
    """Run former v2b_semantic_credit_proxy_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_semantic_credit_proxy_audit-2026-09-23"
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    }
    OBJ_INDEX={"T":0,"A":1,"O":2,"S":3}
    G=.99;LAM=.95;H=32;NENV=8
    SEEDS=[870001,870002,870003,870004]
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def rtg(x,done):
        # x [T,E], done [T,E]
        out=torch.zeros_like(x);run=torch.zeros_like(x[-1])
        for t in range(len(x)-1,-1,-1):
            run=x[t]+G*run*(~done[t]).to(x.dtype)
            out[t]=run
        return out
    def flat_grad(loss,params):
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])
    def cos(a,b):
        if float(a.norm())<1e-12 or float(b.norm())<1e-12:return 0.0
        return float(torch.dot(a,b)/(a.norm()*b.norm()))
    
    def collect(env,m,w,mgr,seed,lab):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];u=[];old=[];rw=[];dn=[];proxy=[]
        with torch.no_grad():
            for _ in range(H):
                # physical proxy before step uses current state + sampled action.
                a,lp,uu=m.act_with_preference_latent(cur,w)
                if lab=="T":
                    # higher = better velocity-command tracking
                    lin=cur[:,0:3];ang=cur[:,3:6];cmd=cur[:,9:12]
                    pr=-((lin[:,0]-cmd[:,0])**2 + (ang[:,2]-cmd[:,2])**2)
                elif lab=="A":
                    pr=-(cur[:,3:5]**2).sum(-1)  # lower roll/pitch angular velocity
                elif lab=="O":
                    gxy=cur[:,6:8]
                    pr=-(gxy*gxy).sum(-1)  # higher = flatter
                else:
                    prev=cur[:,-12:]
                    pr=-((a-prev)**2).mean(-1) # higher = smoother
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());proxy.append(pr)
                cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),"rt":torch.stack(rw),
                "dt":torch.stack(dn).bool(),"proxy":torch.stack(proxy),"next_obs":cur,"w":w.repeat(H,1)}
    
    def audit_case(m,b,lab):
        from talon_rl.models.foundations.four_objective import vector_gae
        j=OBJ_INDEX[lab];w=b["w"]
        with torch.no_grad():
            vt=m.value_with_preference(b["obs"],w).reshape(H,NENV,4)
            nv=m.value_with_preference(b["next_obs"],torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1))
            adv,_=vector_gae(b["rt"],vt,nv,b["dt"],lam=LAM)
            gae=adv[:,:,j].reshape(-1)
            mc=rtg(b["rt"][:,:,j],b["dt"]).reshape(-1)
            proxy=rtg(b["proxy"],b["dt"]).reshape(-1)
            # Center weights to reduce finite-sample baseline noise; direction remains score-function diagnostic.
            mc=mc-mc.mean();proxy=proxy-proxy.mean();gae=gae-gae.mean()
    
        obs=b["obs"]
        dist=m._pre_tanh_dist_with_preference(obs,w)
        logp=(dist.log_prob(b["u"])-m._log_det_jacobian(b["u"])).sum(-1)
        losses={
          "gae_objective":-(logp*gae.detach()).mean(),
          "mc_objective":-(logp*mc.detach()).mean(),
          "physical_proxy":-(logp*proxy.detach()).mean(),
        }
        groups={
          "all_actor":[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")],
          "film":[p for n,p in m.named_parameters() if n.startswith("preference_film")],
          "shared_body":[p for n,p in m.named_parameters() if n.startswith("actor_body")],
        }
        out={}
        for gn,ps in groups.items():
            gs={k:flat_grad(v,ps) for k,v in losses.items()}
            out[gn]={
              "norms":{k:float(g.norm().cpu()) for k,g in gs.items()},
              "cos_gae_vs_mc_objective":cos(gs["gae_objective"],gs["mc_objective"]),
              "cos_gae_vs_physical_proxy":cos(gs["gae_objective"],gs["physical_proxy"]),
              "cos_mc_objective_vs_physical_proxy":cos(gs["mc_objective"],gs["physical_proxy"]),
            }
        return out
    
    def aggregate(rows):
        out={}
        for gn in rows[0]:
            out[gn]={}
            for k in rows[0][gn]:
                if isinstance(rows[0][gn][k],dict):
                    out[gn][k]={kk:{"mean":float(np.mean([r[gn][k][kk] for r in rows])),
                                      "std":float(np.std([r[gn][k][kk] for r in rows]))} for kk in rows[0][gn][k]}
                else:
                    vals=[r[gn][k] for r in rows]
                    out[gn][k]={"mean":float(np.mean(vals)),"std":float(np.std(vals))}
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
            result={}
            for lab in ("T","A","O","S"):
                rows=[]
                for seed in SEEDS:
                    w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
                    rows.append(audit_case(m,collect(env,m,w,mgr,seed,lab),lab))
                result[lab]=aggregate(rows)
            report={"schema":"v2b_semantic_credit_proxy_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "checkpoint":str(a.checkpoint.relative_to(ROOT)),"seeds":SEEDS,"result":result}
            out=a.output_dir/"semantic_credit_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps(result,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_trajectory_semantic_retention_gate():
    """Run former v2b_trajectory_semantic_retention_gate.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_trajectory_semantic_retention_gate-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    FULL_STEPS=64;SHORT_STEPS=32;ENDPOINT_SUITES=4
    BASE_SEED=980001
    TRAJ_SEEDS={"T":[901001,901101],"A":[902001,902101],"O":[903001,903101],"S":[904001,904101]}
    RETAIN_FRAC=0.50
    SURV_MIN=0.95
    ROTATE_ALPHAS=(0.25,0.5,0.75,1.0)
    MIN_STEP_RATIO=1/128
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flatten_params(m):return torch.cat([p.detach().reshape(-1) for _,p in actor_named_params(m)])
    def set_flat_params(m,vec):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.copy_(vec[off:off+n].view_as(p));off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=False,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),"rt":torch.stack(R),
                "dt":torch.stack(D).bool(),"next_obs":cur,"w":w_env.repeat(H,1),"w_env":w_env}
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j]);weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    def mixed_w(device,k):
        labs=[ORDER[(k+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    def mixed_direction(m,b):
        weighted,ratio=objective_losses(m,b);g=flat_grad(sum(weighted.values()),actor_named_params(m)).detach()
        return -g/(g.norm()+1e-12),float(g.norm().cpu()),float((ratio-1).abs().max().detach().cpu())
    def axis_direction(env,m,mgr,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        b=collect_batch(env,m,mgr,seed,w);weighted,_=objective_losses(m,b)
        g=flat_grad(weighted[lab],actor_named_params(m)).detach()
        return -g/(g.norm()+1e-12)
    
    def rollout(env,m,mgr,robot,w_np,seed,steps):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(steps):
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                done_any|=(te|tr).cpu().numpy()
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean());wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    
    def short_semantic(env,m,mgr,robot,lab):
        j=IDX[lab];pk=PHYS[lab];om=[];pm=[];sv=[]
        for seed in TRAJ_SEEDS[lab]:
            h=rollout(env,m,mgr,robot,PREFS[lab],seed,SHORT_STEPS)
            c=rollout(env,m,mgr,robot,PREFS["C"],seed,SHORT_STEPS)
            om.append(h["objective_mean"][j]-c["objective_mean"][j])
            pm.append(c["physical"][pk]-h["physical"][pk])
            sv.append(min(h["survival"],c["survival"]))
        return {"objective_margin":float(np.mean(om)),"physical_margin":float(np.mean(pm)),
                "survival":float(min(sv))}
    
    def freeze_reference(env,m,mgr,robot,lab):
        x=short_semantic(env,m,mgr,robot,lab)
        return {"objective_margin_ref":x["objective_margin"],
                "physical_margin_ref":x["physical_margin"],
                "objective_floor":RETAIN_FRAC*max(x["objective_margin"],0.0),
                "physical_floor":RETAIN_FRAC*max(x["physical_margin"],0.0),
                "survival_floor":SURV_MIN}
    
    def check_refs(env,m,mgr,robot,refs):
        out={};ok=True
        for lab,r in refs.items():
            x=short_semantic(env,m,mgr,robot,lab)
            good=(x["objective_margin"]>=r["objective_floor"] and
                  x["physical_margin"]>=r["physical_floor"] and
                  x["survival"]>=r["survival_floor"])
            x["pass_constraint"]=bool(good);out[lab]=x;ok=ok and good
        return ok,out
    
    def finite_traj_step(env,m,mgr,robot,raw_d,nom_step,refs):
        theta0=flatten_params(m)
        # full raw candidate
        set_flat_params(m,theta0+nom_step*raw_d)
        ok_raw,raw_sem=check_refs(env,m,mgr,robot,refs)
        if not refs or ok_raw:
            return raw_d,nom_step,{"corrected":False,"backtracked":False,"raw_semantics":raw_sem,
                                  "final_semantics":raw_sem,"raw_to_applied_cosine":1.0,"step_ratio":1.0,
                                  "recovery_axes":[]}
        # restore current policy and build recovery direction from currently violated axes.
        set_flat_params(m,theta0)
        violated=[lab for lab,x in raw_sem.items() if not x["pass_constraint"]]
        recover=[]
        for lab in violated:
            recover.append(axis_direction(env,m,mgr,lab,TRAJ_SEEDS[lab][0]))
        candidates=[]
        if recover:
            r=sum(recover);r=r/(r.norm()+1e-12)
            for a in ROTATE_ALPHAS:
                d=(1-a)*raw_d+a*r;d=d/(d.norm()+1e-12)
                candidates.append((a,d))
        # Try rotated full-norm candidates; select first feasible, else best constraint score.
        best=None
        def score(sem):
            vals=[]
            for lab,x in sem.items():
                rr=refs[lab]
                a=x["objective_margin"]/(rr["objective_floor"]+1e-8) if rr["objective_floor"]>0 else 1.0
                b=x["physical_margin"]/(rr["physical_floor"]+1e-8) if rr["physical_floor"]>0 else 1.0
                c=x["survival"]/rr["survival_floor"]
                vals.append(min(a,b,c))
            return min(vals) if vals else 1.0
        for a,d in candidates:
            set_flat_params(m,theta0+nom_step*d)
            ok,sem=check_refs(env,m,mgr,robot,refs);sc=score(sem)
            if best is None or sc>best[0]:best=(sc,a,d,sem)
            if ok:
                return d,nom_step,{"corrected":True,"backtracked":False,"raw_semantics":raw_sem,
                    "final_semantics":sem,"raw_to_applied_cosine":cos(raw_d,d),"step_ratio":1.0,
                    "recovery_axes":violated,"rotate_alpha":a}
        # Backtrack along best rotated direction (or raw if no recovery).
        if best is None:
            d=raw_d;sem0=raw_sem
        else:
            _,aa,d,sem0=best
        actual=nom_step/2
        while actual>=nom_step*MIN_STEP_RATIO:
            set_flat_params(m,theta0+actual*d)
            ok,sem=check_refs(env,m,mgr,robot,refs)
            if ok:
                return d,actual,{"corrected":True,"backtracked":True,"raw_semantics":raw_sem,
                    "final_semantics":sem,"raw_to_applied_cosine":cos(raw_d,d),
                    "step_ratio":actual/(nom_step+1e-12),"recovery_axes":violated}
            actual*=0.5
        # Last resort minimum step, report feasibility honestly.
        actual=nom_step*MIN_STEP_RATIO
        set_flat_params(m,theta0+actual*d)
        ok,sem=check_refs(env,m,mgr,robot,refs)
        return d,actual,{"corrected":True,"backtracked":True,"raw_semantics":raw_sem,
            "final_semantics":sem,"raw_to_applied_cosine":cos(raw_d,d),
            "step_ratio":actual/(nom_step+1e-12),"recovery_axes":violated,
            "feasible_at_min_step":bool(ok)}
    
    def endpoint_eval(env,m,mgr,robot):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed,FULL_STEPS);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,
                     "semantic_score":0.5*(of+pf),"pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),
                     "min_survival":float(min(surv))}
        return ep
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),
                      "final_score":float(scores[-1]),"first_pass_checkpoint":fp,
                      "retained_pass_fraction_after_first_pass":ret}
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init=copy.deepcopy(base.state_dict());nom_step=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
            probe=collect_batch(env,base,mgr,BASE_SEED,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot)
            models={};rows={};refs={};activation={}
            for arm in ("control","trajectory_retention"):
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval();models[arm]=m;rows[arm]=[]
            for lab in ORDER:
                activation[lab]=None
                if base_ep[lab]["pass"]:
                    refs[lab]=freeze_reference(env,models["trajectory_retention"],mgr,robot,lab);activation[lab]=0
            prev_dirs={a:None for a in models}
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                mt=models["trajectory_retention"]
                _,w=mixed_w(torch.device("cuda"),k);bt=collect_batch(env,mt,mgr,seed,w)
                raw_t,gn_t,ratio_t=mixed_direction(mt,bt)
                active_before=sorted(refs)
                applied_t,actual_step,meta=finite_traj_step(env,mt,mgr,robot,raw_t,nom_step,refs)
                cosprev_t=None if prev_dirs["trajectory_retention"] is None else cos(applied_t,prev_dirs["trajectory_retention"])
                prev_dirs["trajectory_retention"]=applied_t.detach().clone()
                ep_t=endpoint_eval(env,mt,mgr,robot)
                for lab in ORDER:
                    if ep_t[lab]["pass"] and lab not in refs:
                        refs[lab]=freeze_reference(env,mt,mgr,robot,lab);activation[lab]=k
                rows["trajectory_retention"].append({"update":k,"active_refs_before_update":active_before,
                    "raw_grad_norm":gn_t,"ratio_maxerr":ratio_t,"retention":meta,
                    "effective_step_norm":actual_step,"update_cos_prev":cosprev_t,
                    "parameter_displacement":float(torch.linalg.vector_norm(flatten_params(mt)-torch.cat([init[n].reshape(-1).cuda() for n,_ in actor_named_params(mt)])).cpu()),
                    "preference_separation":preference_separation(mt,probe),"endpoint":ep_t,
                    "semantic_scores":{lab:ep_t[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mt.state_dict(),"arm":"trajectory_retention","update":k,"effective_step_norm":actual_step},args.output_dir/f"trajectory_retention_u{k}.pt")
                mc=models["control"];_,wc=mixed_w(torch.device("cuda"),k);bc=collect_batch(env,mc,mgr,seed,wc)
                raw_c,gn_c,ratio_c=mixed_direction(mc,bc)
                theta=flatten_params(mc);set_flat_params(mc,theta+actual_step*raw_c)
                cosprev_c=None if prev_dirs["control"] is None else cos(raw_c,prev_dirs["control"]);prev_dirs["control"]=raw_c.detach().clone()
                ep_c=endpoint_eval(env,mc,mgr,robot)
                rows["control"].append({"update":k,"active_refs_before_update":[],
                    "raw_grad_norm":gn_c,"ratio_maxerr":ratio_c,
                    "retention":{"corrected":False,"backtracked":False,"raw_to_applied_cosine":1.0,"step_ratio":actual_step/(nom_step+1e-12)},
                    "effective_step_norm":actual_step,"update_cos_prev":cosprev_c,
                    "parameter_displacement":float(torch.linalg.vector_norm(flatten_params(mc)-torch.cat([init[n].reshape(-1).cuda() for n,_ in actor_named_params(mc)])).cpu()),
                    "preference_separation":preference_separation(mc,probe),"endpoint":ep_c,
                    "semantic_scores":{lab:ep_c[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mc.state_dict(),"arm":"control","update":k,"effective_step_norm":actual_step},args.output_dir/f"control_u{k}.pt")
                (args.output_dir/"trajectory_retention_partial.json").write_text(json.dumps({"activation":activation,"active":sorted(refs),"refs":refs,"arms":rows},indent=2)+"\n")
            report={"schema":"v2b_trajectory_semantic_retention_gate_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "retain_fraction":RETAIN_FRAC,"short_steps":SHORT_STEPS,"trajectory_seeds":TRAJ_SEEDS,
                    "nominal_step_norm":nom_step,"updates":K,"baseline_endpoint":base_ep,"activation":activation,
                    "refs":refs,"arms":rows,"retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            out=args.output_dir/"trajectory_semantic_retention_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH",
              "report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "v2b_endpoint_semantic_eval": run_v2b_endpoint_semantic_eval,
    "v2b_l1_der_diverse_replay_gate": run_v2b_l1_der_diverse_replay_gate,
    "v2b_l2_policy_consolidation_gate": run_v2b_l2_policy_consolidation_gate,
    "v2b_o_trajectory_realization_audit": run_v2b_o_trajectory_realization_audit,
    "v2b_reference_gradient_retention_gate": run_v2b_reference_gradient_retention_gate,
    "v2b_semantic_credit_proxy_audit": run_v2b_semantic_credit_proxy_audit,
    "v2b_trajectory_semantic_retention_gate": run_v2b_trajectory_semantic_retention_gate,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
