"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_v2b_finite_gradient_behavior_realization_audit():
    """Run former v2b_finite_gradient_behavior_realization_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_finite_gradient_behavior_realization-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8
    SCALES=(0.0,0.25,0.5,1.0,2.0)
    BATCH_SEEDS=(960001,960002,960003,960004)
    ENDPOINT_SUITES=4
    SHORT_STEPS=16;FULL_STEPS=64
    
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
    def apply_flat_step(m, direction, step_norm):
        nps=actor_named_params(m);off=0
        with torch.no_grad():
            for _,p in nps:
                n=p.numel()
                p.add_(direction[off:off+n].view_as(p)*step_norm)
                off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    
    def collect_batch(env,m,mgr,seed,pref="A"):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[pref],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w.repeat(H,1),"w_env":w}
    
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    
    def losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        L={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            L[lab]=-po.mean()
        weights=PREFS["A"]
        combined=sum(4*float(weights[j])*L[lab] for j,lab in enumerate(ORDER))
        Aheavy=4*float(weights[1])*L["A"]
        return L,Aheavy,combined,ratio
    
    def local_metrics(m,b):
        L,LA,LC,ratio=losses(m,b)
        return {"A_weighted_loss":float(LA.detach().cpu()),
                "combined_loss":float(LC.detach().cpu()),
                "raw_losses":{k:float(v.detach().cpu()) for k,v in L.items()},
                "ratio_maxerr":float((ratio-1).abs().max().detach().cpu())}
    
    def deterministic_actions(m,probe):
        out={}
        with torch.no_grad():
            for lab in ("A","C","T","O","S"):
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                out[lab]=m.act_inference_with_preference(probe,w)
        return out
    
    def fixed_action_metrics(base_actions,m,probe):
        acts=deterministic_actions(m,probe)
        return {lab:{
          "delta_vs_base":float(torch.linalg.vector_norm(acts[lab]-base_actions[lab],dim=-1).mean().cpu()),
          "mean_action_norm":float(torch.linalg.vector_norm(acts[lab],dim=-1).mean().cpu())}
          for lab in acts}
    
    def rollout(env,m,mgr,robot,w_np,seed,steps):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(steps):
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
    
    def endpoint_suite(env,m,mgr,robot):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed,FULL_STEPS)
                q.update({"suite":suite,"label":lab});rows.append(q)
        endpoint={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj_ok=[];phys_ok=[];dobj=[];dphys=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                do=r["objective_mean"][j]-c["objective_mean"][j]
                dp=r["physical"][pk]-c["physical"][pk]
                obj_ok.append(do>0);phys_ok.append(dp<0);dobj.append(do);dphys.append(dp);surv.append(r["survival"])
            endpoint[lab]={"objective_correct_fraction":float(np.mean(obj_ok)),
                           "physical_correct_fraction":float(np.mean(phys_ok)),
                           "mean_objective_delta_vs_center":float(np.mean(dobj)),
                           "mean_physical_delta_vs_center":float(np.mean(dphys)),
                           "min_survival":float(np.min(surv))}
        return endpoint,rows
    
    def mean_nominal_step():
        d=json.load(open(OPTLOG))
        vals=[r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]
        return float(np.mean(vals))
    
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
            # Four A-heavy batches -> average gradient direction for robustness.
            nps=actor_named_params(base);gAs=[];gCs=[];batches=[]
            for seed in BATCH_SEEDS:
                b=collect_batch(env,base,mgr,seed,"A");batches.append(b)
                L,LA,LC,_=losses(base,b)
                gAs.append(flat_grad(LA,nps).detach())
                gCs.append(flat_grad(LC,nps).detach())
            gA=torch.stack(gAs).mean(0);gC=torch.stack(gCs).mean(0)
            # descent directions
            dA=-gA/(gA.norm()+1e-12);dC=-gC/(gC.norm()+1e-12)
            nominal=mean_nominal_step()
            probe=batches[0]["obs"][:64].detach().clone()
            base_actions=deterministic_actions(base,probe)
            directions={"A_only":dA,"combined":dC}
            results={}
            for dname,dvec in directions.items():
                results[dname]={}
                for scale in SCALES:
                    m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init_state);m.eval()
                    step=nominal*scale
                    if scale>0:apply_flat_step(m,dvec,step)
                    locals_=[local_metrics(m,b) for b in batches]
                    fixed=fixed_action_metrics(base_actions,m,probe)
                    shorts=[rollout(env,m,mgr,robot,PREFS["A"],970001+s,FULL_STEPS if False else SHORT_STEPS) for s in range(4)]
                    endpoint,endpoint_rows=endpoint_suite(env,m,mgr,robot)
                    key=str(scale)
                    ck=args.output_dir/f"{dname}_scale_{key.replace('.','p')}.pt"
                    torch.save({"model":m.state_dict(),"direction":dname,"scale":scale,"nominal_step":nominal},ck)
                    results[dname][key]={
                      "step_norm":step,
                      "local":{"A_weighted_loss_mean":float(np.mean([x["A_weighted_loss"] for x in locals_])),
                               "combined_loss_mean":float(np.mean([x["combined_loss"] for x in locals_])),
                               "ratio_maxerr_mean":float(np.mean([x["ratio_maxerr"] for x in locals_]))},
                      "fixed_action":fixed,
                      "short_A":{"objective_mean":np.mean([x["objective_mean"] for x in shorts],axis=0).tolist(),
                                 "ang_vel_xy_mean":float(np.mean([x["physical"]["ang_vel_xy"] for x in shorts])),
                                 "survival_min":float(np.min([x["survival"] for x in shorts]))},
                      "endpoint":endpoint,
                      "checkpoint":str(ck.relative_to(ROOT))}
            report={"schema":"v2b_finite_gradient_behavior_realization_v1","measurement_only":True,
                    "base_checkpoint":str(CKPT.relative_to(ROOT)),"lambda":LAM,
                    "nominal_actor_step_norm_u51_u75_mean":nominal,
                    "scales":list(SCALES),"gradient_batch_seeds":list(BATCH_SEEDS),
                    "gradient_cos_A_vs_combined":float(torch.dot(dA,dC)/(dA.norm()*dC.norm()+1e-12)),
                    "gradient_norm_A":float(gA.norm().cpu()),"gradient_norm_combined":float(gC.norm().cpu()),
                    "results":results}
            out=args.output_dir/"finite_gradient_behavior_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({
              "status":"FROZEN_BY_HASH","report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),
              "base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps(report,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_objective_interference_audit():
    """Run former v2b_objective_interference_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_objective_interference_audit-2026-09-23"
    ORDER=("T","A","O","S")
    HEADS=("Tracking","Angular","Orientation","Smoothness")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    }
    G=.99;LAM=.95;H=32;NENV=8
    SUITES=4
    SEEDS=[860001,860002,860003,860004]
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    
    def collect(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,u=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);pre.append(u);old.append(lp)
                rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {
            "obs":torch.cat(ob),"w":w.repeat(H,1),"u":torch.cat(pre),"old":torch.cat(old),
            "rt":torch.stack(rw),"dt":torch.stack(dn).bool(),"next_obs":cur
        }
    
    def flat_grads(loss, params):
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        chunks=[]
        for p,g in zip(params,gs):
            chunks.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
        return torch.cat(chunks)
    
    def cos(a,b):
        na=float(a.norm());nb=float(b.norm())
        if na<1e-12 or nb<1e-12:return 0.0
        return float(torch.dot(a,b)/(a.norm()*b.norm()))
    
    def pairwise_cos(grads):
        out={}
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j<=i:continue
                out[f"{a}-{b}"]=cos(grads[a],grads[b])
        return out
    
    def summary_vec(g):
        return {"norm":float(g.norm().detach().cpu())}
    
    def compute_case(m,batch,pref_lab):
        from talon_rl.models.foundations.four_objective import vector_gae
        w=batch["w"]
        with torch.no_grad():
            vt=m.value_with_preference(batch["obs"],w).reshape(H,NENV,4)
            nv=m.value_with_preference(batch["next_obs"],torch.tensor(PREFS[pref_lab],device="cuda").repeat(NENV,1))
            adv,_=vector_gae(batch["rt"],vt,nv,batch["dt"],lam=LAM)
            A=adv.reshape(-1,4).detach()
    
        # Rebuild differentiable policy path from fixed obs/u.
        obs=batch["obs"]
        actor_in=m._with_w(obs,w)
        h=m.actor_body(actor_in)
        gb=m.preference_film(w);gamma,beta=torch.chunk(gb,2,dim=-1)
        hfilm=(1+gamma)*h+beta
        emb=m.preference_embedding(w)
        feat=torch.cat((hfilm,emb),dim=-1)
        mean=m.actor_mean(feat)
        std=m.log_std.exp().expand_as(mean)
        dist=torch.distributions.Normal(mean,std)
        logp=(dist.log_prob(batch["u"])-m._log_det_jacobian(batch["u"])).sum(-1)
        ratio=torch.exp(logp-batch["old"].detach())
        clipped=ratio.clamp(.8,1.2)
    
        # Per-objective PPO objective terms; at frozen checkpoint ratio≈1.
        losses={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clipped*A[:,j])
            losses[lab]=-po.mean()
    
        weighted=4*sum(float(PREFS[pref_lab][j])*losses[lab] for j,lab in enumerate(ORDER))
    
        groups={
          "all_actor":[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")],
          "shared_body":[p for n,p in m.named_parameters() if n.startswith("actor_body")],
          "direct_input":[m.actor_body[0].weight],
          "embedding":[p for n,p in m.named_parameters() if n.startswith("preference_embedding")],
          "film":[p for n,p in m.named_parameters() if n.startswith("preference_film")],
          "actor_head":[p for n,p in m.named_parameters() if n.startswith("actor_mean")],
        }
    
        out={"parameter_groups":{},"representation":{}}
        for gname,params in groups.items():
            og={lab:flat_grads(losses[lab],params) for lab in ORDER}
            cg=flat_grads(weighted,params)
            norms={lab:float(og[lab].norm().cpu()) for lab in ORDER}
            denom=sum(norms.values())+1e-12
            weighted_components={lab:4*float(PREFS[pref_lab][j])*og[lab] for j,lab in enumerate(ORDER)}
            heavy=pref_lab
            heavy_comp=weighted_components[heavy]
            others=sum((weighted_components[x] for x in ORDER if x!=heavy),torch.zeros_like(heavy_comp))
            out["parameter_groups"][gname]={
                "objective_norms":norms,
                "objective_norm_share":{lab:norms[lab]/denom for lab in ORDER},
                "pairwise_cosine":pairwise_cos(og),
                "combined_norm":float(cg.norm().cpu()),
                "combined_cosine_to_objectives":{lab:cos(cg,og[lab]) for lab in ORDER},
                "heavy_objective":heavy,
                "heavy_weighted_component_norm":float(heavy_comp.norm().cpu()),
                "other_weighted_components_norm":float(others.norm().cpu()),
                "combined_cosine_to_heavy":cos(cg,og[heavy]),
                "heavy_vs_others_cosine":cos(heavy_comp,others),
                "override_ratio_other_over_heavy":float(others.norm()/(heavy_comp.norm()+1e-12)),
            }
    
        # Representation-level gradients isolate where conflict appears.
        reps={"h_pre_film":h,"gamma":gamma,"beta":beta,"h_post_film":hfilm,"embedding":emb,"pre_tanh_mean":mean}
        for rname,t in reps.items():
            og={}
            for lab in ORDER:
                g=torch.autograd.grad(losses[lab],t,retain_graph=True,allow_unused=True)[0]
                if g is None:g=torch.zeros_like(t)
                og[lab]=g.reshape(-1)
            cg=torch.autograd.grad(weighted,t,retain_graph=True,allow_unused=True)[0]
            if cg is None:cg=torch.zeros_like(t)
            cg=cg.reshape(-1)
            weighted_components={lab:4*float(PREFS[pref_lab][j])*og[lab] for j,lab in enumerate(ORDER)}
            heavy_comp=weighted_components[pref_lab]
            others=sum((weighted_components[x] for x in ORDER if x!=pref_lab),torch.zeros_like(heavy_comp))
            out["representation"][rname]={
                "objective_norms":{lab:float(og[lab].norm().cpu()) for lab in ORDER},
                "pairwise_cosine":pairwise_cos(og),
                "combined_norm":float(cg.norm().cpu()),
                "combined_cosine_to_objectives":{lab:cos(cg,og[lab]) for lab in ORDER},
                "combined_cosine_to_heavy":cos(cg,og[pref_lab]),
                "heavy_vs_others_cosine":cos(heavy_comp,others),
                "override_ratio_other_over_heavy":float(others.norm()/(heavy_comp.norm()+1e-12)),
            }
        return out
    
    def aggregate(cases):
        # recursive average for numeric leaves across suites.
        def rec(vals):
            v0=vals[0]
            if isinstance(v0,dict):
                return {k:rec([v[k] for v in vals]) for k in v0}
            if isinstance(v0,(int,float,np.floating)):
                a=np.asarray(vals,float)
                return {"mean":float(a.mean()),"std":float(a.std(ddof=0))}
            return v0
        return rec(cases)
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        args=ap.parse_args()
        if not args.checkpoint.is_absolute():args.checkpoint=(ROOT/args.checkpoint).resolve()
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
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(args.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
    
            raw={lab:[] for lab in ORDER}
            for si,seed in enumerate(SEEDS):
                for lab in ORDER:
                    w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
                    batch=collect(env,m,w,mgr,seed)
                    raw[lab].append(compute_case(m,batch,lab))
    
            agg={lab:aggregate(raw[lab]) for lab in ORDER}
    
            # compact diagnostic table for dominant all-actor + representation conflict.
            compact={}
            for lab in ORDER:
                pg=agg[lab]["parameter_groups"]["all_actor"]
                rg=agg[lab]["representation"]["h_post_film"]
                fg=agg[lab]["parameter_groups"]["film"]
                compact[lab]={
                  "all_actor_combined_cos_to_heavy":pg["combined_cosine_to_heavy"],
                  "all_actor_override_ratio":pg["override_ratio_other_over_heavy"],
                  "all_actor_heavy_vs_others_cos":pg["heavy_vs_others_cosine"],
                  "post_film_combined_cos_to_heavy":rg["combined_cosine_to_heavy"],
                  "post_film_override_ratio":rg["override_ratio_other_over_heavy"],
                  "film_combined_cos_to_heavy":fg["combined_cosine_to_heavy"],
                  "film_override_ratio":fg["override_ratio_other_over_heavy"],
                  "all_actor_objective_norms":pg["objective_norms"],
                  "all_actor_pairwise_cosine":pg["pairwise_cosine"],
                }
    
            report={"schema":"v2b_objective_interference_audit_v1","measurement_only":True,"optimizer_steps":0,
                    "checkpoint":str(args.checkpoint.relative_to(ROOT)),"seeds":SEEDS,"preferences":{k:v.tolist() for k,v in PREFS.items()},
                    "aggregate":agg,"compact":compact}
            out=args.output_dir/"interference_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH",
              "report_sha256":sha(out),"checkpoint_sha256":sha(args.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps(compact,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_state_fixed_phase_credit_audit():
    """Run former v2b_state_fixed_phase_credit_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
    OUT=ROOT/"runs/v2b_state_fixed_phase_credit_audit-2026-09-23"
    PREF=np.array([.1,.1,.7,.1],np.float32)
    PHASES=(8,20,36,52)
    H=64;NENV=8;G=.99;LAM=.95
    SEEDS=(890001,890002,890003,890004)
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    
    def rtg(x,done):
        out=torch.zeros_like(x);run=torch.zeros_like(x[-1])
        for t in range(len(x)-1,-1,-1):
            run=x[t]+G*run*(~done[t]).to(x.dtype);out[t]=run
        return out
    
    def cos(a,b):
        a=a.reshape(-1);b=b.reshape(-1)
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def collect(env,m,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        from talon_rl.models.foundations.four_objective import vector_gae
        w=torch.tensor(PREF,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+100000)
        obs=[];u=[];old=[];R=[];D=[];V=[]
        with torch.no_grad():
            for _ in range(H):
                obs.append(cur);V.append(m.value_with_preference(cur,w))
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());u.append(uu);old.append(lp);cur=ot(nxt).cuda()
            nv=m.value_with_preference(cur,w)
        O=torch.stack(obs);U=torch.stack(u);R=torch.stack(R);D=torch.stack(D).bool();V=torch.stack(V)
        adv,_=vector_gae(R,V,nv,D,lam=LAM)
        mc=rtg(R[:,:,2],D)
        return {"obs":O,"u":U,"R":R,"D":D,"advO":adv[:,:,2],"mcO":mc,"w":w}
    
    def local_action_grad(env,m,mgr,state,w,h):
        # Deterministic mean action, one-step orientation reward finite-difference in applied action coordinates.
        # This is state-fixed and simulator-based; central FD avoids autograd through simulator.
        # clone state using simulator reset-to-state helper unavailable generically, so use reward-model proxy:
        # Orientation objective is flat_orientation_l2; local action gradient through one-step dynamics is estimated
        # by short replay from captured simulator state via articulation state snapshot/restore.
        robot=env.unwrapped.scene["robot"]
        # capture simulator state
        root=robot.data.root_state_w.clone()
        jp=robot.data.joint_pos.clone();jv=robot.data.joint_vel.clone()
        cur=state
        with torch.no_grad():
            mean=m.actor_mean(m._actor_features_v2a(cur,w))
            a0=torch.tanh(mean)*m.ACTION_CLIP
        grads=torch.zeros_like(a0)
        def restore():
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(jp,jv)
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step()
            env.unwrapped.scene.update(env.unwrapped.physics_dt)
        # Evaluate immediate next-step normalized Orientation objective under +/- action.
        from talon_rl.rewards.objectives import normalized_objective_vector
        for j in range(a0.shape[1]):
            vals=[]
            for sgn in (+1,-1):
                restore()
                aa=a0.clone();aa[:,j]+=sgn*h
                env.step(aa)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                vals.append(torch.tensor(vec[:,2],device="cuda"))
            grads[:,j]=(vals[0]-vals[1])/(2*h)
        restore()
        # Map applied-action gradient into pre-tanh mean coordinates:
        # a = ACTION_CLIP * tanh(mu), so d r / d mu = ACTION_CLIP*(1-tanh(mu)^2) * d r / d a.
        jac=m.ACTION_CLIP*(1.0-torch.tanh(mean)**2)
        local_mu=jac*grads
        return local_mu,a0
    
    def score_action_directions(m,b,t):
        # Map MC and GAE score gradients into pre-tanh/action-mean coordinates at fixed states.
        obs=b["obs"][t];u=b["u"][t];w=b["w"]
        with torch.no_grad():
            mean=m.actor_mean(m._actor_features_v2a(obs,w))
            std=m.log_std.exp().expand_as(mean)
            score_mean=(u-mean)/(std*std)
            gae=b["advO"][t]
            mc=b["mcO"][t]-b["mcO"][t].mean()
            g_gae=score_mean*gae.unsqueeze(-1)
            g_mc=score_mean*mc.unsqueeze(-1)
        return g_gae,g_mc
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
        a=ap.parse_args()
        if not a.checkpoint.is_absolute(): a.checkpoint=(ROOT/a.checkpoint).resolve()
        if not a.output_dir.is_absolute(): a.output_dir=(ROOT/a.output_dir).resolve()
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
    
            rows={str(t):[] for t in PHASES}
            for seed in SEEDS:
                b=collect(env,m,mgr,seed)
                w=b["w"]
                for t in PHASES:
                    # Reconstruct env at same seed and advance deterministically with recorded sampled actions to phase t,
                    # so simulator state matches captured rollout before local FD.
                    env.reset(seed=seed);cur=ot(env.unwrapped.observation_manager.compute()["policy"]).cuda()
                    for k in range(t):
                        aa=torch.tanh(b["u"][k])*m.ACTION_CLIP
                        nxt,_,_,_,_=env.step(aa);cur=ot(nxt).cuda()
                    # state should now correspond to b.obs[t]
                    ggae,gmc=score_action_directions(m,b,t)
                    hs=(0.04,0.02,0.01,0.005)
                    locals={}
                    for hh in hs:
                        locals[str(hh)],_=local_action_grad(env,m,mgr,b["obs"][t],w,hh)
                    base=locals["0.01"]
                    row={
                        "mc_vs_gae":cos(gmc,ggae),
                        "mc_norm":float(gmc.norm().cpu()),
                        "gae_norm":float(ggae.norm().cpu()),
                    }
                    for hh in hs:
                        key=str(hh);lv=locals[key]
                        row[f"local_vs_mc_h{key}"]=cos(lv,gmc)
                        row[f"local_vs_gae_h{key}"]=cos(lv,ggae)
                        row[f"local_norm_h{key}"]=float(lv.norm().cpu())
                    for h1,h2 in zip(hs[:-1],hs[1:]):
                        row[f"local_cos_h{h1}_h{h2}"]=cos(locals[str(h1)],locals[str(h2)])
                    rows[str(t)].append(row)
    
            summary={}
            for t in PHASES:
                rr=rows[str(t)]
                summary[str(t)]={k:{"mean":float(np.mean([x[k] for x in rr])),
                                     "std":float(np.std([x[k] for x in rr]))}
                                 for k in rr[0]}
            report={"schema":"v2b_state_fixed_phase_credit_audit_fd_sweep_v2","measurement_only":True,"optimizer_steps":0,
                    "checkpoint":str(a.checkpoint.relative_to(ROOT)),"phases":list(PHASES),"seeds":list(SEEDS),
                    "summary":summary,"raw":rows,
                    "note":"local gradient is one-step simulator finite-difference in applied-action coordinates, then mapped with the exact tanh Jacobian into pre-tanh mean coordinates; MC/GAE score directions are in the same pre-tanh mean coordinates."}
            out=a.output_dir/"phase_credit_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
            print(json.dumps(summary,indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True:main()

STAGES = {
    "v2b_finite_gradient_behavior_realization_audit": run_v2b_finite_gradient_behavior_realization_audit,
    "v2b_objective_interference_audit": run_v2b_objective_interference_audit,
    "v2b_state_fixed_phase_credit_audit": run_v2b_state_fixed_phase_credit_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
