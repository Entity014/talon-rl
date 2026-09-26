"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_rv1_c_critic_foundation_revalidation():
    """Run former rv1_c_critic_foundation_revalidation.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/rv1_b_preference_sensitivity-2026-09-23"
    OUT_DEFAULT=ROOT/"runs/rv1_c_critic_revalidation-2026-09-23"
    SNAPS=(50,75);ORDER=("T","A","O","S","C");G=.99;NENV=8;H=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
        out=np.zeros_like(R)
        if seg is None:
            run=np.zeros_like(R[0])
            for t in range(len(R)-1,-1,-1):
                run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):
                    run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def eval_roll(env,m,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,12),device="cuda");done=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
        return {"obj":R.mean((0,1)).tolist(),"phys":pm,"survival":float(1-done.mean()),
                "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
                "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
                "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def sha(p):
        h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);a=ap.parse_args()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            report={"schema":"rv1_c_critic_foundation_revalidation_v1","measurement_only":True,"snapshots":{}}
            for snap in SNAPS:
                m=T4SharedActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(RUN/f"model_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
                rows=[]
                # C23-style representative reset-diverse audit: 4 independent suites per preference.
                for li,lab in enumerate(ORDER):
                    for suite in range(4):
                        seed=9100000+snap*10000+li*1000+suite*113
                        q=eval_roll(env,m,PREFS[lab],seed);q.update({"preference":lab,"suite":suite});rows.append(q)
                report["snapshots"][str(snap)]={"rows":rows}
                # Aggregate per head and preference.
                by_pref={};by_head=[]
                for lab in ORDER:
                    rr=[x for x in rows if x["preference"]==lab]
                    h=np.array([x["h32_ev"] for x in rr]);mc=np.array([x["mc64_ev"] for x in rr])
                    hb=np.array([x["h32_bias"] for x in rr]);mb=np.array([x["mc64_bias"] for x in rr])
                    by_pref[lab]={"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
                                  "mc64_ev_mean":float(mc.mean()),"mc64_negative_fraction":float((mc<0).mean()),
                                  "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
                                  "min_survival":float(min(x["survival"] for x in rr))}
                for j in range(4):
                    hv=[x["h32_ev"][j] for x in rows];mv=[x["mc64_ev"][j] for x in rows]
                    hb=[x["h32_bias"][j] for x in rows];mb=[x["mc64_bias"][j] for x in rows]
                    by_head.append({"head":j,"h32_ev_mean":float(np.mean(hv)),"h32_negative_fraction":float(np.mean(np.array(hv)<0)),
                                    "mc64_ev_mean":float(np.mean(mv)),"mc64_negative_fraction":float(np.mean(np.array(mv)<0)),
                                    "h32_mean_abs_bias":float(np.mean(np.abs(hb))),"mc64_mean_abs_bias":float(np.mean(np.abs(mb)))})
                hall=np.array([z for x in rows for z in x["h32_ev"]]);mall=np.array([z for x in rows for z in x["mc64_ev"]])
                hbias=np.array([z for x in rows for z in x["h32_bias"]]);mbias=np.array([z for x in rows for z in x["mc64_bias"]])
                agg={"h32_ev_mean":float(hall.mean()),"h32_negative_fraction":float((hall<0).mean()),
                     "mc64_ev_mean":float(mall.mean()),"mc64_negative_fraction":float((mall<0).mean()),
                     "h32_mean_abs_bias":float(np.abs(hbias).mean()),"mc64_mean_abs_bias":float(np.abs(mbias).mean()),
                     "min_survival":float(min(x["survival"] for x in rows))}
                report["snapshots"][str(snap)].update({"aggregate":agg,"by_preference":by_pref,"by_head":by_head})
            # Critic-independent behavior comparison using matched resets on u75 only.
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(RUN/"model_75.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
            behavior=[]
            for suite in range(4):
                seed=9200000+suite
                for lab in ORDER:
                    q=eval_roll(env,m,PREFS[lab],seed)
                    behavior.append({"suite":suite,"preference":lab,"obj":q["obj"],"phys":q["phys"],"survival":q["survival"]})
            center=[x for x in behavior if x["preference"]=="C"]
            sem={}
            pkeys={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
            for j,lab in enumerate(("T","A","O","S")):
                ook=[];pok=[]
                for suite in range(4):
                    r=next(x for x in behavior if x["suite"]==suite and x["preference"]==lab)
                    c=next(x for x in behavior if x["suite"]==suite and x["preference"]=="C")
                    ook.append(r["obj"][j]>c["obj"][j]);pok.append(r["phys"][pkeys[lab]]<c["phys"][pkeys[lab]])
                sem[lab]={"objective_correct_fraction":float(np.mean(ook)),"physical_correct_fraction":float(np.mean(pok))}
            report["critic_independent_behavior"]={"endpoint_vs_center":sem,
                "all_axes_pass_075":bool(all(v["objective_correct_fraction"]>=.75 and v["physical_correct_fraction"]>=.75 for v in sem.values())),
                "min_survival":float(min(x["survival"] for x in behavior))}
            # Revalidation gate uses the same spirit as repaired critic contract: positive H32/MC64 EV, bounded negatives, low bias, survival.
            a75=report["snapshots"]["75"]["aggregate"]
            critic_pass=bool(a75["h32_ev_mean"]>0 and a75["mc64_ev_mean"]>0 and
                             a75["h32_negative_fraction"]<=.25 and a75["mc64_negative_fraction"]<=.25 and
                             a75["min_survival"]>=.95)
            behavior_fail=not report["critic_independent_behavior"]["all_axes_pass_075"]
            report["decision"]={"critic_revalidation_pass":critic_pass,
                                "critic_independent_semantic_fail":behavior_fail,
                                "v2_authorized":bool(critic_pass and behavior_fail),
                                "rv1_architecture_conclusion_blocked":bool(not critic_pass)}
            report["status"]="PASS_CLEAN_V1_FAIL" if critic_pass and behavior_fail else ("CRITIC_REVALIDATION_FAIL" if not critic_pass else "SEMANTICS_RECOVERED")
            out=a.output_dir/"revalidation_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            prov={"status":"FROZEN_BY_HASH","gate":"RV1-C critic revalidation","verdict":report["status"],
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},
                               "runs/rv1_b_preference_sensitivity-2026-09-23/model_50.pt":{"sha256":sha(RUN/"model_50.pt")},
                               "runs/rv1_b_preference_sensitivity-2026-09-23/model_75.pt":{"sha256":sha(RUN/"model_75.pt")},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())}}}
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
            print(json.dumps({"status":report["status"],"u50":report["snapshots"]["50"]["aggregate"],
                              "u75":a75,"behavior":report["critic_independent_behavior"],"decision":report["decision"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_critic_persistence_repair():
    """Run former rv1_critic_persistence_repair.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    OUT_DEFAULT=ROOT/"runs/rv1_critic_persistence_repair-2026-09-23"
    ORDER=("T","A","O","S","C")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
     "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;NENV=8;SUPPORT=12;ANCHOR_K=6;ADAPT_K=6;POOL_MAX=24
    SNAPS=(0,10,25,50,75)
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
        return sol[:-1].T,sol[-1]
    
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        d0=np.mean(X*X,axis=1);sel=[int(np.argmax(d0))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        arr=np.stack([PREFS[x] for x in labs])
        return labs,torch.tensor(arr,device=device)
    
    def collect(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=obs_tensor(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
        fo=torch.cat(ob);fw=w.repeat(H,1)
        with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
        C=np.concatenate(cmd,0)
        summary=np.r_[C.mean(0),C.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy()]
        out={"obs":fo,"w":fw,"Y":Y,"F":F,"rt":rt,"dt":dt,"next_obs":cur,
             "summary":summary.astype(np.float32),"termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def fit_head_anchor_adaptive(m,anchors,anchor_summaries,pool,summaries):
        aidx=select_diverse(anchor_summaries,ANCHOR_K)
        ridx=select_diverse(summaries,ADAPT_K)
        support=[anchors[i] for i in aidx]+[pool[i] for i in ridx]
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return aidx,ridx
    
    def pairwise_dist(actions):
        labs=list(actions);vals=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:
                vals.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals))}
    def preference_sensitivity(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        pair=pairwise_dist(acts)
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w)
        rows=[]
        for j in range(a.shape[1]):
            g=torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0]
            rows.append(g)
        jac=torch.stack(rows,dim=1)
        first=m.actor_body[0].weight[:,m.physical_obs_dim:]
        return {"pairwise_action_distance":pair,
                "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
                "preference_weight_norm":float(first.norm().detach().cpu())}
    
    def fresh_critic(env,m,mgr,seed):
        hs=[];bias=[];surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            q=collect(env,m,w,mgr,seed+qi*101,False)
            with torch.no_grad():V=m.value_with_preference(q["obs"],q["w"])
            Y=q["Y"]
            hs.extend(ev(Y[:,j].cpu(),V[:,j].cpu()) for j in range(4))
            bias.extend(float((V[:,j]-Y[:,j]).mean().cpu()) for j in range(4))
            surv.append(1-q["termination_fraction"])
        return {"h32_ev_mean":float(np.mean(hs)),"h32_negative_fraction":float(np.mean(np.array(hs)<0)),
                "mean_abs_bias":float(np.mean(np.abs(bias))),"min_survival":float(np.min(surv))}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75)
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);ap.add_argument("--seed",type=int,default=73001)
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
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=obs_tensor(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(INIT,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
            anchors=[];anchor_summaries=[];pool=[];summaries=[]
            # Frozen anchor bank: collected once from initialization and never evicted.
            # Use 12 candidates then retain 6 by diversity at every fit.
            for k in range(SUPPORT):
                _,wk=pref_batch(k,torch.device("cuda"))
                q=collect(env,m,wk,mgr,args.seed+10000+k*137,False)
                anchors.append((q["F"].cpu().numpy(),q["Y"].cpu().numpy()));anchor_summaries.append(q["summary"])
            # Seed adaptive bank with the same initial supports so update 0 is well-defined.
            pool=list(anchors);summaries=list(anchor_summaries)
            fit_head_anchor_adaptive(m,anchors,anchor_summaries,pool,summaries)
            rows=[];snaps={}
            def audit(tag):
                m.eval();sens=preference_sensitivity(m,probe)
                fresh=fresh_critic(env,m,mgr,args.seed+500000+int(tag)*1000)
                snaps[str(tag)]={"sensitivity":sens,"fresh_critic":fresh}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt")
                m.train()
            audit(0)
            for uidx in range(1,args.updates+1):
                labs,w=pref_batch(uidx,torch.device("cuda"))
                mainb=collect(env,m,w,mgr,args.seed+uidx*211,True)
                # Add one independent reset support candidate and keep a bounded pool.
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                q=collect(env,m,ws,mgr,args.seed+200000+uidx*223,False)
                pool.append((q["F"].cpu().numpy(),q["Y"].cpu().numpy()));summaries.append(q["summary"])
                if len(pool)>POOL_MAX:pool.pop(0);summaries.pop(0)
                anchor_idx,adaptive_idx=fit_head_anchor_adaptive(m,anchors,anchor_summaries,pool,summaries)
                with torch.no_grad():
                    vt=m.value_with_preference(mainb["obs"],mainb["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(mainb["next_obs"],w)
                    adv,_=vector_gae(mainb["rt"],vt,nv,mainb["dt"],lam=.95)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(mainb["obs"],mainb["w"],mainb["u"])-mainb["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"PPO ratio invariant failed: {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),mainb["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pref_grad=m.actor_body[0].weight.grad[:,m.physical_obs_dim:]
                pref_grad_norm=float(pref_grad.norm().detach().cpu())
                total_grad=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu())
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                rows.append({"update":uidx,"preferences":labs,"loss":float(loss.detach().cpu()),
                             "ratio_maxerr":ratio_err,"preference_input_grad_norm":pref_grad_norm,
                             "actor_grad_norm_preclip":total_grad,"termination_fraction":mainb["termination_fraction"],
                              "anchor_support_indices":anchor_idx,
                             "adaptive_support_indices":adaptive_idx})
                if uidx in SNAPS:audit(uidx)
            report={"schema":"rv1_critic_persistence_repair_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"critic persistence repair: 6 frozen anchor + 6 recent adaptive supports","objective_semantics_gate":False,
                    "rows":rows,"snapshots":snaps}
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            fr=snaps[str(args.updates)]["fresh_critic"]
            grad=np.asarray([r["preference_input_grad_norm"] for r in rows])
            term=np.asarray([r["termination_fraction"] for r in rows])
            ratio=np.asarray([r["ratio_maxerr"] for r in rows])
            # Screen criteria intentionally test use of w + foundation preservation only.
            criteria={
                "preference_weights_moved":sf["preference_weight_norm"]>1e-5,
                "fixed_state_action_separation":sf["pairwise_action_distance"]["mean"]>1e-3,
                "preference_jacobian_nonzero":sf["jacobian_fro_mean"]>1e-3,
                "preference_gradient_observed":float(np.median(grad))>1e-6,
                "ppo_ratio_invariant":float(np.max(ratio))<=1e-4,
                "locomotion_not_collapsed":float(np.mean(term[-10:]))<0.5,
                "fresh_critic_not_collapsed":fr["h32_ev_mean"]>-0.5 and fr["h32_negative_fraction"]<0.75,
            }
            passed=all(criteria.values())
            report["summary"]={"status":"RV1-B PASS" if passed else "RV1-B FAIL","criteria":criteria,
                "initial_sensitivity":s0,"final_sensitivity":sf,"final_fresh_critic":fr,
                "median_preference_grad_norm":float(np.median(grad)),
                "last10_termination_fraction":float(np.mean(term[-10:])),
                "max_ratio_error":float(np.max(ratio)),
                "semantic_gate_authorized":False}
            out=args.output_dir/"rv1_b_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_critic_persistence_repair_v2():
    """Run former rv1_critic_persistence_repair_v2.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    OUT_DEFAULT=ROOT/"runs/rv1_critic_persistence_repair_v2-2026-09-23"
    ORDER=("T","A","O","S","C")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
     "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;NENV=8;SUPPORT=12;ANCHOR_K=6;ADAPT_K=6;POOL_MAX=24
    SNAPS=(0,10,25,50,75)
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
        return sol[:-1].T,sol[-1]
    
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        d0=np.mean(X*X,axis=1);sel=[int(np.argmax(d0))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        arr=np.stack([PREFS[x] for x in labs])
        return labs,torch.tensor(arr,device=device)
    
    def collect(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=obs_tensor(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
        fo=torch.cat(ob);fw=w.repeat(H,1)
        with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
        C=np.concatenate(cmd,0)
        summary=np.r_[C.mean(0),C.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy()]
        out={"obs":fo,"w":fw,"Y":Y,"F":F,"rt":rt,"dt":dt,"next_obs":cur,
             "summary":summary.astype(np.float32),"termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def fit_head_anchor_adaptive(m,anchors,anchor_summaries,pool,summaries):
        aidx=select_diverse(anchor_summaries,ANCHOR_K)
        ridx=select_diverse(summaries,ADAPT_K)
        support=[anchors[i] for i in aidx]+[pool[i] for i in ridx]
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return aidx,ridx
    
    def pairwise_dist(actions):
        labs=list(actions);vals=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:
                vals.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals))}
    def preference_sensitivity(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        pair=pairwise_dist(acts)
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w)
        rows=[]
        for j in range(a.shape[1]):
            g=torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0]
            rows.append(g)
        jac=torch.stack(rows,dim=1)
        first=m.actor_body[0].weight[:,m.physical_obs_dim:]
        return {"pairwise_action_distance":pair,
                "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
                "preference_weight_norm":float(first.norm().detach().cpu())}
    
    def fresh_critic(env,m,mgr,seed):
        hs=[];bias=[];surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            q=collect(env,m,w,mgr,seed+qi*101,False)
            with torch.no_grad():V=m.value_with_preference(q["obs"],q["w"])
            Y=q["Y"]
            hs.extend(ev(Y[:,j].cpu(),V[:,j].cpu()) for j in range(4))
            bias.extend(float((V[:,j]-Y[:,j]).mean().cpu()) for j in range(4))
            surv.append(1-q["termination_fraction"])
        return {"h32_ev_mean":float(np.mean(hs)),"h32_negative_fraction":float(np.mean(np.array(hs)<0)),
                "mean_abs_bias":float(np.mean(np.abs(bias))),"min_survival":float(np.min(surv))}
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75)
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);ap.add_argument("--seed",type=int,default=73001)
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
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=obs_tensor(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(INIT,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
            anchor_candidates=[];anchor_candidate_summaries=[];pool=[];summaries=[]
            # Freeze anchor DISTRIBUTION, not old-policy targets:
            # choose 6 reset/preference seed specs once from 12 initialization candidates.
            for k in range(SUPPORT):
                _,wk=pref_batch(k,torch.device("cuda"))
                q=collect(env,m,wk,mgr,args.seed+10000+k*137,False)
                anchor_candidates.append((q["F"].cpu().numpy(),q["Y"].cpu().numpy()))
                anchor_candidate_summaries.append(q["summary"])
            anchor_spec_idx=select_diverse(anchor_candidate_summaries,ANCHOR_K)
            anchor_specs=[(k,args.seed+10000+k*137) for k in anchor_spec_idx]
            # Seed adaptive recent pool with the initial candidates.
            pool=list(anchor_candidates);summaries=list(anchor_candidate_summaries)
            init_anchors=[anchor_candidates[k] for k in anchor_spec_idx]
            init_anchor_summaries=[anchor_candidate_summaries[k] for k in anchor_spec_idx]
            fit_head_anchor_adaptive(m,init_anchors,init_anchor_summaries,pool,summaries)
            rows=[];snaps={}
            def audit(tag):
                m.eval();sens=preference_sensitivity(m,probe)
                fresh=fresh_critic(env,m,mgr,args.seed+500000+int(tag)*1000)
                snaps[str(tag)]={"sensitivity":sens,"fresh_critic":fresh}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt")
                m.train()
            audit(0)
            for uidx in range(1,args.updates+1):
                labs,w=pref_batch(uidx,torch.device("cuda"))
                mainb=collect(env,m,w,mgr,args.seed+uidx*211,True)
                # Add one independent reset support candidate and keep a bounded pool.
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                q=collect(env,m,ws,mgr,args.seed+200000+uidx*223,False)
                pool.append((q["F"].cpu().numpy(),q["Y"].cpu().numpy()));summaries.append(q["summary"])
                if len(pool)>POOL_MAX:pool.pop(0);summaries.pop(0)
                # Recollect frozen anchor seed specs with CURRENT policy so targets stay on-policy
                # while state/reset/command coverage remains fixed.
                current_anchors=[];current_anchor_summaries=[]
                for k,aseed in anchor_specs:
                    _,wa=pref_batch(k,torch.device("cuda"))
                    qa=collect(env,m,wa,mgr,aseed,False)
                    current_anchors.append((qa["F"].cpu().numpy(),qa["Y"].cpu().numpy()))
                    current_anchor_summaries.append(qa["summary"])
                anchor_idx,adaptive_idx=fit_head_anchor_adaptive(m,current_anchors,current_anchor_summaries,pool,summaries)
                with torch.no_grad():
                    vt=m.value_with_preference(mainb["obs"],mainb["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(mainb["next_obs"],w)
                    adv,_=vector_gae(mainb["rt"],vt,nv,mainb["dt"],lam=.95)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(mainb["obs"],mainb["w"],mainb["u"])-mainb["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"PPO ratio invariant failed: {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),mainb["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pref_grad=m.actor_body[0].weight.grad[:,m.physical_obs_dim:]
                pref_grad_norm=float(pref_grad.norm().detach().cpu())
                total_grad=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu())
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                rows.append({"update":uidx,"preferences":labs,"loss":float(loss.detach().cpu()),
                             "ratio_maxerr":ratio_err,"preference_input_grad_norm":pref_grad_norm,
                             "actor_grad_norm_preclip":total_grad,"termination_fraction":mainb["termination_fraction"],
                              "anchor_seed_spec_indices":anchor_spec_idx,"anchor_local_indices":anchor_idx,
                             "adaptive_support_indices":adaptive_idx})
                if uidx in SNAPS:audit(uidx)
            report={"schema":"rv1_critic_persistence_repair_v2","seed":args.seed,"updates":args.updates,
                    "training_scope":"critic persistence repair v2: 6 frozen anchor seed specs recollected current-policy + 6 recent adaptive supports","objective_semantics_gate":False,
                    "rows":rows,"snapshots":snaps}
            s0=snaps["0"]["sensitivity"];sf=snaps[str(args.updates)]["sensitivity"]
            fr=snaps[str(args.updates)]["fresh_critic"]
            grad=np.asarray([r["preference_input_grad_norm"] for r in rows])
            term=np.asarray([r["termination_fraction"] for r in rows])
            ratio=np.asarray([r["ratio_maxerr"] for r in rows])
            # Screen criteria intentionally test use of w + foundation preservation only.
            criteria={
                "preference_weights_moved":sf["preference_weight_norm"]>1e-5,
                "fixed_state_action_separation":sf["pairwise_action_distance"]["mean"]>1e-3,
                "preference_jacobian_nonzero":sf["jacobian_fro_mean"]>1e-3,
                "preference_gradient_observed":float(np.median(grad))>1e-6,
                "ppo_ratio_invariant":float(np.max(ratio))<=1e-4,
                "locomotion_not_collapsed":float(np.mean(term[-10:]))<0.5,
                "fresh_critic_not_collapsed":fr["h32_ev_mean"]>-0.5 and fr["h32_negative_fraction"]<0.75,
            }
            passed=all(criteria.values())
            report["summary"]={"status":"RV1-B PASS" if passed else "RV1-B FAIL","criteria":criteria,
                "initial_sensitivity":s0,"final_sensitivity":sf,"final_fresh_critic":fr,
                "median_preference_grad_norm":float(np.median(grad)),
                "last10_termination_fraction":float(np.mean(term[-10:])),
                "max_ratio_error":float(np.max(ratio)),
                "semantic_gate_authorized":False}
            out=args.output_dir/"rv1_b_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_critic_persistence_revalidation():
    """Run former rv1_critic_persistence_revalidation.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/rv1_critic_persistence_repair-2026-09-23"
    OUT_DEFAULT=ROOT/"runs/rv1_critic_persistence_revalidation-2026-09-23"
    SNAPS=(50,75);ORDER=("T","A","O","S","C");G=.99;NENV=8;H=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
        out=np.zeros_like(R)
        if seg is None:
            run=np.zeros_like(R[0])
            for t in range(len(R)-1,-1,-1):
                run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):
                    run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def eval_roll(env,m,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,12),device="cuda");done=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
        return {"obj":R.mean((0,1)).tolist(),"phys":pm,"survival":float(1-done.mean()),
                "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
                "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
                "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def sha(p):
        h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);a=ap.parse_args()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            report={"schema":"rv1_critic_persistence_revalidation_v1","measurement_only":True,"snapshots":{}}
            for snap in SNAPS:
                m=T4SharedActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(RUN/f"model_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
                rows=[]
                # C23-style representative reset-diverse audit: 4 independent suites per preference.
                for li,lab in enumerate(ORDER):
                    for suite in range(4):
                        seed=9100000+snap*10000+li*1000+suite*113
                        q=eval_roll(env,m,PREFS[lab],seed);q.update({"preference":lab,"suite":suite});rows.append(q)
                report["snapshots"][str(snap)]={"rows":rows}
                # Aggregate per head and preference.
                by_pref={};by_head=[]
                for lab in ORDER:
                    rr=[x for x in rows if x["preference"]==lab]
                    h=np.array([x["h32_ev"] for x in rr]);mc=np.array([x["mc64_ev"] for x in rr])
                    hb=np.array([x["h32_bias"] for x in rr]);mb=np.array([x["mc64_bias"] for x in rr])
                    by_pref[lab]={"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
                                  "mc64_ev_mean":float(mc.mean()),"mc64_negative_fraction":float((mc<0).mean()),
                                  "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
                                  "min_survival":float(min(x["survival"] for x in rr))}
                for j in range(4):
                    hv=[x["h32_ev"][j] for x in rows];mv=[x["mc64_ev"][j] for x in rows]
                    hb=[x["h32_bias"][j] for x in rows];mb=[x["mc64_bias"][j] for x in rows]
                    by_head.append({"head":j,"h32_ev_mean":float(np.mean(hv)),"h32_negative_fraction":float(np.mean(np.array(hv)<0)),
                                    "mc64_ev_mean":float(np.mean(mv)),"mc64_negative_fraction":float(np.mean(np.array(mv)<0)),
                                    "h32_mean_abs_bias":float(np.mean(np.abs(hb))),"mc64_mean_abs_bias":float(np.mean(np.abs(mb)))})
                hall=np.array([z for x in rows for z in x["h32_ev"]]);mall=np.array([z for x in rows for z in x["mc64_ev"]])
                hbias=np.array([z for x in rows for z in x["h32_bias"]]);mbias=np.array([z for x in rows for z in x["mc64_bias"]])
                agg={"h32_ev_mean":float(hall.mean()),"h32_negative_fraction":float((hall<0).mean()),
                     "mc64_ev_mean":float(mall.mean()),"mc64_negative_fraction":float((mall<0).mean()),
                     "h32_mean_abs_bias":float(np.abs(hbias).mean()),"mc64_mean_abs_bias":float(np.abs(mbias).mean()),
                     "min_survival":float(min(x["survival"] for x in rows))}
                report["snapshots"][str(snap)].update({"aggregate":agg,"by_preference":by_pref,"by_head":by_head})
            # Critic-independent behavior comparison using matched resets on u75 only.
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(RUN/"model_75.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
            behavior=[]
            for suite in range(4):
                seed=9200000+suite
                for lab in ORDER:
                    q=eval_roll(env,m,PREFS[lab],seed)
                    behavior.append({"suite":suite,"preference":lab,"obj":q["obj"],"phys":q["phys"],"survival":q["survival"]})
            center=[x for x in behavior if x["preference"]=="C"]
            sem={}
            pkeys={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
            for j,lab in enumerate(("T","A","O","S")):
                ook=[];pok=[]
                for suite in range(4):
                    r=next(x for x in behavior if x["suite"]==suite and x["preference"]==lab)
                    c=next(x for x in behavior if x["suite"]==suite and x["preference"]=="C")
                    ook.append(r["obj"][j]>c["obj"][j]);pok.append(r["phys"][pkeys[lab]]<c["phys"][pkeys[lab]])
                sem[lab]={"objective_correct_fraction":float(np.mean(ook)),"physical_correct_fraction":float(np.mean(pok))}
            report["critic_independent_behavior"]={"endpoint_vs_center":sem,
                "all_axes_pass_075":bool(all(v["objective_correct_fraction"]>=.75 and v["physical_correct_fraction"]>=.75 for v in sem.values())),
                "min_survival":float(min(x["survival"] for x in behavior))}
            # Revalidation gate uses the same spirit as repaired critic contract: positive H32/MC64 EV, bounded negatives, low bias, survival.
            a75=report["snapshots"]["75"]["aggregate"]
            critic_pass=bool(a75["h32_ev_mean"]>0 and a75["mc64_ev_mean"]>0 and
                             a75["h32_negative_fraction"]<=.25 and a75["mc64_negative_fraction"]<=.25 and
                             a75["min_survival"]>=.95)
            behavior_fail=not report["critic_independent_behavior"]["all_axes_pass_075"]
            report["decision"]={"critic_revalidation_pass":critic_pass,
                                "critic_independent_semantic_fail":behavior_fail,
                                "v2_authorized":bool(critic_pass and behavior_fail),
                                "rv1_architecture_conclusion_blocked":bool(not critic_pass)}
            report["status"]="PASS_CLEAN_V1_FAIL" if critic_pass and behavior_fail else ("CRITIC_REVALIDATION_FAIL" if not critic_pass else "SEMANTICS_RECOVERED")
            out=a.output_dir/"revalidation_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            prov={"status":"FROZEN_BY_HASH","gate":"RV1-C critic revalidation","verdict":report["status"],
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},
                               "runs/rv1_critic_persistence_repair-2026-09-23/model_50.pt":{"sha256":sha(RUN/"model_50.pt")},
                               "runs/rv1_critic_persistence_repair-2026-09-23/model_75.pt":{"sha256":sha(RUN/"model_75.pt")},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())}}}
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
            print(json.dumps({"status":report["status"],"u50":report["snapshots"]["50"]["aggregate"],
                              "u75":a75,"behavior":report["critic_independent_behavior"],"decision":report["decision"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_critic_persistence_revalidation_v2():
    """Run former rv1_critic_persistence_revalidation_v2.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/rv1_critic_persistence_repair_v2-2026-09-23"
    OUT_DEFAULT=ROOT/"runs/rv1_critic_persistence_revalidation_v2-2026-09-23"
    SNAPS=(50,75);ORDER=("T","A","O","S","C");G=.99;NENV=8;H=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
        out=np.zeros_like(R)
        if seg is None:
            run=np.zeros_like(R[0])
            for t in range(len(R)-1,-1,-1):
                run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):
                    run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def eval_roll(env,m,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,12),device="cuda");done=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
        return {"obj":R.mean((0,1)).tolist(),"phys":pm,"survival":float(1-done.mean()),
                "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
                "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
                "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def sha(p):
        h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);a=ap.parse_args()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            report={"schema":"rv1_critic_persistence_revalidation_v2","measurement_only":True,"snapshots":{}}
            for snap in SNAPS:
                m=T4SharedActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(RUN/f"model_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
                rows=[]
                # C23-style representative reset-diverse audit: 4 independent suites per preference.
                for li,lab in enumerate(ORDER):
                    for suite in range(4):
                        seed=9100000+snap*10000+li*1000+suite*113
                        q=eval_roll(env,m,PREFS[lab],seed);q.update({"preference":lab,"suite":suite});rows.append(q)
                report["snapshots"][str(snap)]={"rows":rows}
                # Aggregate per head and preference.
                by_pref={};by_head=[]
                for lab in ORDER:
                    rr=[x for x in rows if x["preference"]==lab]
                    h=np.array([x["h32_ev"] for x in rr]);mc=np.array([x["mc64_ev"] for x in rr])
                    hb=np.array([x["h32_bias"] for x in rr]);mb=np.array([x["mc64_bias"] for x in rr])
                    by_pref[lab]={"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
                                  "mc64_ev_mean":float(mc.mean()),"mc64_negative_fraction":float((mc<0).mean()),
                                  "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
                                  "min_survival":float(min(x["survival"] for x in rr))}
                for j in range(4):
                    hv=[x["h32_ev"][j] for x in rows];mv=[x["mc64_ev"][j] for x in rows]
                    hb=[x["h32_bias"][j] for x in rows];mb=[x["mc64_bias"][j] for x in rows]
                    by_head.append({"head":j,"h32_ev_mean":float(np.mean(hv)),"h32_negative_fraction":float(np.mean(np.array(hv)<0)),
                                    "mc64_ev_mean":float(np.mean(mv)),"mc64_negative_fraction":float(np.mean(np.array(mv)<0)),
                                    "h32_mean_abs_bias":float(np.mean(np.abs(hb))),"mc64_mean_abs_bias":float(np.mean(np.abs(mb)))})
                hall=np.array([z for x in rows for z in x["h32_ev"]]);mall=np.array([z for x in rows for z in x["mc64_ev"]])
                hbias=np.array([z for x in rows for z in x["h32_bias"]]);mbias=np.array([z for x in rows for z in x["mc64_bias"]])
                agg={"h32_ev_mean":float(hall.mean()),"h32_negative_fraction":float((hall<0).mean()),
                     "mc64_ev_mean":float(mall.mean()),"mc64_negative_fraction":float((mall<0).mean()),
                     "h32_mean_abs_bias":float(np.abs(hbias).mean()),"mc64_mean_abs_bias":float(np.abs(mbias).mean()),
                     "min_survival":float(min(x["survival"] for x in rows))}
                report["snapshots"][str(snap)].update({"aggregate":agg,"by_preference":by_pref,"by_head":by_head})
            # Critic-independent behavior comparison using matched resets on u75 only.
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(RUN/"model_75.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
            behavior=[]
            for suite in range(4):
                seed=9200000+suite
                for lab in ORDER:
                    q=eval_roll(env,m,PREFS[lab],seed)
                    behavior.append({"suite":suite,"preference":lab,"obj":q["obj"],"phys":q["phys"],"survival":q["survival"]})
            center=[x for x in behavior if x["preference"]=="C"]
            sem={}
            pkeys={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
            for j,lab in enumerate(("T","A","O","S")):
                ook=[];pok=[]
                for suite in range(4):
                    r=next(x for x in behavior if x["suite"]==suite and x["preference"]==lab)
                    c=next(x for x in behavior if x["suite"]==suite and x["preference"]=="C")
                    ook.append(r["obj"][j]>c["obj"][j]);pok.append(r["phys"][pkeys[lab]]<c["phys"][pkeys[lab]])
                sem[lab]={"objective_correct_fraction":float(np.mean(ook)),"physical_correct_fraction":float(np.mean(pok))}
            report["critic_independent_behavior"]={"endpoint_vs_center":sem,
                "all_axes_pass_075":bool(all(v["objective_correct_fraction"]>=.75 and v["physical_correct_fraction"]>=.75 for v in sem.values())),
                "min_survival":float(min(x["survival"] for x in behavior))}
            # Revalidation gate uses the same spirit as repaired critic contract: positive H32/MC64 EV, bounded negatives, low bias, survival.
            a75=report["snapshots"]["75"]["aggregate"]
            critic_pass=bool(a75["h32_ev_mean"]>0 and a75["mc64_ev_mean"]>0 and
                             a75["h32_negative_fraction"]<=.25 and a75["mc64_negative_fraction"]<=.25 and
                             a75["min_survival"]>=.95)
            behavior_fail=not report["critic_independent_behavior"]["all_axes_pass_075"]
            report["decision"]={"critic_revalidation_pass":critic_pass,
                                "critic_independent_semantic_fail":behavior_fail,
                                "v2_authorized":bool(critic_pass and behavior_fail),
                                "rv1_architecture_conclusion_blocked":bool(not critic_pass)}
            report["status"]="PASS_CLEAN_V1_FAIL" if critic_pass and behavior_fail else ("CRITIC_REVALIDATION_FAIL" if not critic_pass else "SEMANTICS_RECOVERED")
            out=a.output_dir/"revalidation_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            prov={"status":"FROZEN_BY_HASH","gate":"RV1-C critic revalidation","verdict":report["status"],
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},
                               "runs/rv1_critic_persistence_repair_v2-2026-09-23/model_50.pt":{"sha256":sha(RUN/"model_50.pt")},
                               "runs/rv1_critic_persistence_repair_v2-2026-09-23/model_75.pt":{"sha256":sha(RUN/"model_75.pt")},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())}}}
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
            print(json.dumps({"status":report["status"],"u50":report["snapshots"]["50"]["aggregate"],
                              "u75":a75,"behavior":report["critic_independent_behavior"],"decision":report["decision"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_critic_phase_balanced_repair():
    """Run former rv1_critic_phase_balanced_repair.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    OUT_DEFAULT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23"
    ORDER=("T","A","O","S","C")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;SUP_H=64;NENV=8;POOL_MAX=24
    ANCHOR_CANDIDATES=12;ANCHOR_PHASE_K=3;ADAPT_PHASE_K=3
    SNAPS=(0,10,25,50,75)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        sel=[int(np.argmax(np.mean(X*X,axis=1)))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_actor(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        with torch.no_grad():
            for _ in range(H):
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool()
        out={"obs":torch.cat(ob),"w":w.repeat(H,1),"rt":rt,"dt":dt,"next_obs":cur,
             "termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def collect_support64(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();obs=torch.stack(ob) # [64,E,D]
        C=np.stack(cmd) # [64,E,3]
        phases=[]
        for pi,(st,en) in enumerate(((0,32),(32,64))):
            Y=trunc(rt[st:en],dt[st:en]).reshape(-1,4).detach()
            po=obs[st:en].reshape(-1,obs.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).detach()
            cc=C[st:en].reshape(-1,C.shape[-1])
            summary=np.r_[cc.mean(0),cc.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),float(pi)]
            phases.append({"F":F.cpu().numpy(),"Y":Y.cpu().numpy(),"summary":summary.astype(np.float32),
                           "phase":"early" if pi==0 else "late"})
        return phases
    
    def fit_phase_balanced(m,anchor_units,adaptive_pools,adaptive_summaries):
        support=[];sel={}
        for phase in ("early","late"):
            au=anchor_units[phase]; asum=[x["summary"] for x in au]
            ai=select_diverse(asum,ANCHOR_PHASE_K)
            ri=select_diverse(adaptive_summaries[phase],ADAPT_PHASE_K)
            support += [(au[i]["F"],au[i]["Y"]) for i in ai]
            support += [(adaptive_pools[phase][i]["F"],adaptive_pools[phase][i]["Y"]) for i in ri]
            sel[phase]={"anchor":ai,"adaptive":ri}
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return sel
    
    def pairwise_dist(actions):
        labs=list(actions);v=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:v.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def sensitivity(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1);acts[lab]=m.act_inference_with_preference(probe,w)
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w);rows=[]
        for j in range(a.shape[1]):rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
        return {"pairwise_action_distance":pairwise_dist(acts),
                "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
                "preference_weight_norm":float(m.actor_body[0].weight[:,m.physical_obs_dim:].norm().detach().cpu())}
    
    def fresh_phase_audit(env,m,mgr,seed):
        phase_ev={"early":[],"late":[]};phase_bias={"early":[],"late":[]};surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            units=collect_support64(env,m,w,mgr,seed+qi*101)
            for unit in units:
                with torch.no_grad():
                    # Features already current-policy and head is linear.
                    F=torch.tensor(unit["F"],device="cuda",dtype=m.critic_head.weight.dtype)
                    V=m.critic_head(F).cpu().numpy()
                Y=unit["Y"];ph=unit["phase"]
                phase_ev[ph].extend(ev(Y[:,j],V[:,j]) for j in range(4))
                phase_bias[ph].extend(float(np.mean(V[:,j]-Y[:,j])) for j in range(4))
            surv.append(1.0) # deterministic support rollouts are diagnostic; detailed terminations in independent revalidation
        out={}
        for ph in ("early","late"):
            a=np.asarray(phase_ev[ph]);b=np.asarray(phase_bias[ph])
            out[ph]={"ev_mean":float(a.mean()),"negative_fraction":float((a<0).mean()),"mean_abs_bias":float(np.abs(b).mean())}
        out["combined_negative_fraction"]=float(np.mean(np.r_[np.asarray(phase_ev["early"])<0,np.asarray(phase_ev["late"])<0]))
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001)
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(INIT,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            # Freeze anchor seed/preference specs once.
            cand=[]
            for k in range(ANCHOR_CANDIDATES):
                _,w=pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            # Select 6 seed specs on combined early+late summary geometry.
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
    
            adaptive_pools={"early":[],"late":[]};adaptive_summaries={"early":[],"late":[]}
            # initialize adaptive pools from all initialization candidates
            for _,_,units in cand:
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
    
            def current_anchor_units():
                out={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=pref_batch(k,torch.device("cuda"))
                    for u in collect_support64(env,m,w,mgr,seed):
                        out[u["phase"]].append(u)
                return out
            fit_phase_balanced(m,current_anchor_units(),adaptive_pools,adaptive_summaries)
    
            rows=[];snaps={}
            def audit(tag):
                m.eval();snaps[str(tag)]={"sensitivity":sensitivity(m,probe),
                    "phase_critic":fresh_phase_audit(env,m,mgr,args.seed+500000+int(tag)*1000)}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt");m.train()
            audit(0)
    
            for uidx in range(1,args.updates+1):
                labs,w=pref_batch(uidx,torch.device("cuda"))
                main=collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                # one recent adaptive 64-step support candidate contributes one early + one late unit
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                units=collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
                    if len(adaptive_pools[ph])>POOL_MAX:
                        adaptive_pools[ph].pop(0);adaptive_summaries[ph].pop(0)
                selected=fit_phase_balanced(m,current_anchor_units(),adaptive_pools,adaptive_summaries)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pg=float(m.actor_body[0].weight.grad[:,m.physical_obs_dim:].norm().detach().cpu())
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                             "preference_input_grad_norm":pg,"actor_grad_norm_preclip":total,
                             "termination_fraction":main["termination_fraction"],"selected":selected})
                if uidx in SNAPS:audit(uidx)
    
            report={"schema":"rv1_critic_phase_balanced_repair_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"critic-only support construction change: 6 early + 6 late; each phase 3 current-policy frozen-anchor specs + 3 recent adaptive",
                    "actor_ppo_objectives_changed":False,"anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps}
            final=snaps[str(args.updates)];term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            crit=final["phase_critic"];sens=final["sensitivity"]
            criteria={"early_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
                      "late_repaired":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
                      "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
                      "actor_sensitivity_preserved":sens["pairwise_action_distance"]["mean"]>1e-3 and sens["jacobian_fro_mean"]>1e-3,
                      "ppo_ratio_invariant":float(ratio.max())<=1e-4,
                      "survival_preserved":float(term[-10:].mean())<.5}
            report["summary"]={"status":"PHASE-PERSISTENCE INTERNAL PASS" if all(criteria.values()) else "PHASE-PERSISTENCE INTERNAL FAIL",
                               "criteria":criteria,"final":final,"last10_termination_fraction":float(term[-10:].mean()),
                               "max_ratio_error":float(ratio.max())}
            (args.output_dir/"phase_repair_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_critic_phase_balanced_revalidation():
    """Run former rv1_critic_phase_balanced_revalidation.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23"
    OUT_DEFAULT=ROOT/"runs/rv1_critic_phase_balanced_revalidation-2026-09-23"
    SNAPS=(50,75);ORDER=("T","A","O","S","C");G=.99;NENV=8;H=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
        out=np.zeros_like(R)
        if seg is None:
            run=np.zeros_like(R[0])
            for t in range(len(R)-1,-1,-1):
                run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):
                    run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def eval_roll(env,m,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,12),device="cuda");done=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
        return {"obj":R.mean((0,1)).tolist(),"phys":pm,"survival":float(1-done.mean()),
                "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
                "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
                "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def sha(p):
        h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);a=ap.parse_args()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            report={"schema":"rv1_critic_phase_balanced_revalidation_v1","measurement_only":True,"snapshots":{}}
            for snap in SNAPS:
                m=T4SharedActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(RUN/f"model_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
                rows=[]
                # C23-style representative reset-diverse audit: 4 independent suites per preference.
                for li,lab in enumerate(ORDER):
                    for suite in range(4):
                        seed=9100000+snap*10000+li*1000+suite*113
                        q=eval_roll(env,m,PREFS[lab],seed);q.update({"preference":lab,"suite":suite});rows.append(q)
                report["snapshots"][str(snap)]={"rows":rows}
                # Aggregate per head and preference.
                by_pref={};by_head=[]
                for lab in ORDER:
                    rr=[x for x in rows if x["preference"]==lab]
                    h=np.array([x["h32_ev"] for x in rr]);mc=np.array([x["mc64_ev"] for x in rr])
                    hb=np.array([x["h32_bias"] for x in rr]);mb=np.array([x["mc64_bias"] for x in rr])
                    by_pref[lab]={"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
                                  "mc64_ev_mean":float(mc.mean()),"mc64_negative_fraction":float((mc<0).mean()),
                                  "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
                                  "min_survival":float(min(x["survival"] for x in rr))}
                for j in range(4):
                    hv=[x["h32_ev"][j] for x in rows];mv=[x["mc64_ev"][j] for x in rows]
                    hb=[x["h32_bias"][j] for x in rows];mb=[x["mc64_bias"][j] for x in rows]
                    by_head.append({"head":j,"h32_ev_mean":float(np.mean(hv)),"h32_negative_fraction":float(np.mean(np.array(hv)<0)),
                                    "mc64_ev_mean":float(np.mean(mv)),"mc64_negative_fraction":float(np.mean(np.array(mv)<0)),
                                    "h32_mean_abs_bias":float(np.mean(np.abs(hb))),"mc64_mean_abs_bias":float(np.mean(np.abs(mb)))})
                hall=np.array([z for x in rows for z in x["h32_ev"]]);mall=np.array([z for x in rows for z in x["mc64_ev"]])
                hbias=np.array([z for x in rows for z in x["h32_bias"]]);mbias=np.array([z for x in rows for z in x["mc64_bias"]])
                agg={"h32_ev_mean":float(hall.mean()),"h32_negative_fraction":float((hall<0).mean()),
                     "mc64_ev_mean":float(mall.mean()),"mc64_negative_fraction":float((mall<0).mean()),
                     "h32_mean_abs_bias":float(np.abs(hbias).mean()),"mc64_mean_abs_bias":float(np.abs(mbias).mean()),
                     "min_survival":float(min(x["survival"] for x in rows))}
                report["snapshots"][str(snap)].update({"aggregate":agg,"by_preference":by_pref,"by_head":by_head})
            # Critic-independent behavior comparison using matched resets on u75 only.
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(RUN/"model_75.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
            behavior=[]
            for suite in range(4):
                seed=9200000+suite
                for lab in ORDER:
                    q=eval_roll(env,m,PREFS[lab],seed)
                    behavior.append({"suite":suite,"preference":lab,"obj":q["obj"],"phys":q["phys"],"survival":q["survival"]})
            center=[x for x in behavior if x["preference"]=="C"]
            sem={}
            pkeys={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
            for j,lab in enumerate(("T","A","O","S")):
                ook=[];pok=[]
                for suite in range(4):
                    r=next(x for x in behavior if x["suite"]==suite and x["preference"]==lab)
                    c=next(x for x in behavior if x["suite"]==suite and x["preference"]=="C")
                    ook.append(r["obj"][j]>c["obj"][j]);pok.append(r["phys"][pkeys[lab]]<c["phys"][pkeys[lab]])
                sem[lab]={"objective_correct_fraction":float(np.mean(ook)),"physical_correct_fraction":float(np.mean(pok))}
            report["critic_independent_behavior"]={"endpoint_vs_center":sem,
                "all_axes_pass_075":bool(all(v["objective_correct_fraction"]>=.75 and v["physical_correct_fraction"]>=.75 for v in sem.values())),
                "min_survival":float(min(x["survival"] for x in behavior))}
            # Revalidation gate uses the same spirit as repaired critic contract: positive H32/MC64 EV, bounded negatives, low bias, survival.
            a75=report["snapshots"]["75"]["aggregate"]
            critic_pass=bool(a75["h32_ev_mean"]>0 and a75["mc64_ev_mean"]>0 and
                             a75["h32_negative_fraction"]<=.25 and a75["mc64_negative_fraction"]<=.25 and
                             a75["min_survival"]>=.95)
            behavior_fail=not report["critic_independent_behavior"]["all_axes_pass_075"]
            report["decision"]={"critic_revalidation_pass":critic_pass,
                                "critic_independent_semantic_fail":behavior_fail,
                                "v2_authorized":bool(critic_pass and behavior_fail),
                                "rv1_architecture_conclusion_blocked":bool(not critic_pass)}
            report["status"]="PASS_CLEAN_V1_FAIL" if critic_pass and behavior_fail else ("CRITIC_REVALIDATION_FAIL" if not critic_pass else "SEMANTICS_RECOVERED")
            out=a.output_dir/"revalidation_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            prov={"status":"FROZEN_BY_HASH","gate":"RV1-C critic revalidation","verdict":report["status"],
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},
                               "runs/rv1_critic_phase_balanced_repair-2026-09-23/model_50.pt":{"sha256":sha(RUN/"model_50.pt")},
                               "runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt":{"sha256":sha(RUN/"model_75.pt")},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())}}}
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
            print(json.dumps({"status":report["status"],"u50":report["snapshots"]["50"]["aggregate"],
                              "u75":a75,"behavior":report["critic_independent_behavior"],"decision":report["decision"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_head_fitting_support_weighting_audit():
    """Run former rv1_head_fitting_support_weighting_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
    REP=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/phase_repair_report.json"
    OUT=ROOT/"runs/rv1_head_fitting_support_weighting-2026-09-23"
    NENV=8;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    ORDER=("T","A","O","S","C");ALPHAS=(0,.25,.5,.75,1.0);PATHS=(("T","A"),("T","O"),("T","S"))
    HEADS=("Tracking","Angular","Orientation","Smoothness")
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    def interp(a,b,t):return (1-t)*PREFS[a]+t*PREFS[b]
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    
    def collect64(env,m,w_np,seed,label,source,update=None):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager
        w=torch.tensor(w_np,device="cuda") if np.asarray(w_np).ndim==2 else torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];R=[];D=[];cmd=[]
        with torch.no_grad():
            for _ in range(64):
                obs.append(cur);cmd.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector({n:raw[:,i] for i,n in enumerate(names)},shape=(NENV,))
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);D.append((te|tr).cuda());cur=ot(nxt).cuda()
        O=torch.stack(obs);R=torch.stack(R);D=torch.stack(D).bool();C=np.asarray(cmd)
        units=[]
        for phase,(st,en) in (("early",(0,32)),("late",(32,64))):
            po=O[st:en].reshape(-1,O.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).cpu().numpy()
            Y=trunc(R[st:en],D[st:en]).reshape(-1,4).cpu().numpy()
            CC=C[st:en].reshape(-1,3);WW=pw.cpu().numpy()
            n=len(F)
            units.append({"F":F,"Y":Y,"cmd":CC,"w":WW,
                          "phase":phase,"source":source,"label":label,"update":update,
                          "unit_label":np.array([f"{source}:{label}:{phase}"]*n,object)})
        return units
    
    def wridge(F,Y,weights=None,l2=1.0):
        A=np.c_[F,np.ones(len(F))]
        if weights is None:weights=np.ones(len(F))
        sw=np.sqrt(np.asarray(weights,float).clip(1e-12))[:,None]
        Aw=A*sw;Yw=Y*sw
        I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(Aw.T@Aw+l2*I,Aw.T@Yw)
        return sol[:-1],sol[-1]
    
    def predict(F,fit):W,b=fit;return F@W+b
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def metrics(Y,P,mask=None):
        if mask is not None:Y=Y[mask];P=P[mask]
        return {"ev":[ev(Y[:,j],P[:,j]) for j in range(4)],
                "mae":[float(np.mean(np.abs(Y[:,j]-P[:,j]))) for j in range(4)]}
    
    def geometry(F,weights=None,l2=1.0):
        A=np.c_[F,np.ones(len(F))]
        if weights is None:weights=np.ones(len(F))
        sw=np.sqrt(np.asarray(weights,float).clip(1e-12))[:,None];Aw=A*sw
        G=Aw.T@Aw
        # condition number of regularized normal matrix
        I=np.eye(A.shape[1]);I[-1,-1]=0;M=G+l2*I
        cond=float(np.linalg.cond(M))
        inv=np.linalg.pinv(M)
        # weighted ridge leverage diag of Aw M^-1 Aw^T
        lev=np.einsum("ij,jk,ik->i",Aw,inv,Aw)
        return {"condition_number":cond,"leverage_mean":float(lev.mean()),
                "leverage_p95":float(np.quantile(lev,.95)),"leverage_max":float(lev.max())}
    
    def nearest_dist(SF,QF):
        mu=SF.mean(0);sd=SF.std(0)+1e-8;SZ=(SF-mu)/sd;QZ=(QF-mu)/sd
        St=torch.tensor(SZ,dtype=torch.float32,device="cuda");vals=[]
        for i in range(0,len(QZ),512):
            Qt=torch.tensor(QZ[i:i+512],dtype=torch.float32,device="cuda")
            vals.append((torch.cdist(Qt,St).min(1).values/np.sqrt(SF.shape[1])).cpu().numpy())
        return np.concatenate(vals)
    
    def command_bin_weights(C,bins=4):
        C=np.asarray(C,float)
        # quantile bin each dimension then inverse-frequency joint bin
        ids=[]
        for j in range(C.shape[1]):
            edges=np.unique(np.quantile(C[:,j],np.linspace(0,1,bins+1)))
            if len(edges)<=2: ids.append(np.zeros(len(C),int))
            else: ids.append(np.clip(np.digitize(C[:,j],edges[1:-1]),0,len(edges)-2))
        key=np.stack(ids,1)
        _,inv,cnt=np.unique(key,axis=0,return_inverse=True,return_counts=True)
        w=1.0/cnt[inv];return w/w.mean()
    
    def source_phase_weights(meta_source,meta_phase):
        s=np.asarray(meta_source,object);p=np.asarray(meta_phase,object);w=np.ones(len(s))
        # four strata: anchor/adaptive x early/late, equal total weight
        strata=[("anchor","early"),("anchor","late"),("adaptive","early"),("adaptive","late")]
        for st in strata:
            m=(s==st[0])&(p==st[1])
            if m.any():w[m]=1.0/m.sum()
        return w/w.mean()
    
    def semantic_proximity_weights(F,semanticF):
        # oracle diagnostic: weight support by closeness to actual semantic eval distribution.
        mu=F.mean(0);sd=F.std(0)+1e-8;SZ=(F-mu)/sd;QZ=(semanticF-mu)/sd
        Qt=torch.tensor(QZ,dtype=torch.float32,device="cuda");St=torch.tensor(SZ,dtype=torch.float32,device="cuda")
        d=[]
        for i in range(0,len(SZ),512):
            Sx=St[i:i+512];d.append((torch.cdist(Sx,Qt).min(1).values/np.sqrt(F.shape[1])).cpu().numpy())
        d=np.concatenate(d)
        scale=np.quantile(d,.5)+1e-8
        w=np.exp(-d/scale);return w/w.mean(),d
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);a=ap.parse_args()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            tr=json.load(open(REP));row75=tr["rows"][-1];sel=row75["selected"];anchor_specs=tr["anchor_specs"]
    
            # Reconstruct expanded real support at u75 with current policy:
            # all 6 frozen anchor specs + all recent adaptive updates 52..75, both phases.
            anchor_units=[]
            for k,seed in anchor_specs:
                anchor_units += collect64(env,m,pref_batch(k,torch.device("cuda")).cpu().numpy(),seed,f"a{k}","anchor")
            adaptive_units=[]
            for u in range(52,76):
                seed=73001+200000+u*223
                adaptive_units += collect64(env,m,pref_batch(u+17,torch.device("cuda")).cpu().numpy(),seed,f"u{u}","adaptive",u)
            expanded=anchor_units+adaptive_units
    
            # selected 12 support units according to update75 indices
            selected=[]
            for phase in ("early","late"):
                au=[x for x in anchor_units if x["phase"]==phase]
                adu=[x for x in adaptive_units if x["phase"]==phase]
                for i in sel[phase]["anchor"]:selected.append(au[i])
                for i in sel[phase]["adaptive"]:selected.append(adu[i])
    
            # semantic real evaluation set, second-H32 only
            sem=[]
            for suite in range(4):
                seed=840001+suite
                for lab in ORDER:
                    sem += [x for x in collect64(env,m,PREFS[lab],seed,f"ep_{lab}_{suite}","semantic") if x["phase"]=="late"]
            for pi,(x,y) in enumerate(PATHS):
                for suite in range(4):
                    seed=850001+pi*1000+suite
                    for al in ALPHAS:
                        sem += [z for z in collect64(env,m,interp(x,y,al),seed,f"{x}{y}_{al}_{suite}","semantic") if z["phase"]=="late"]
    
            def cat(units,key):return np.concatenate([u[key] for u in units],axis=0)
            def meta(units,key):
                return np.concatenate([np.array([u[key]]*len(u["F"]),object) for u in units])
    
            SFs,SYs,SCs=cat(selected,"F"),cat(selected,"Y"),cat(selected,"cmd")
            SFe,SYe,SCe=cat(expanded,"F"),cat(expanded,"Y"),cat(expanded,"cmd")
            src=meta(expanded,"source");ph=meta(expanded,"phase")
            QF,QY=cat(sem,"F"),cat(sem,"Y")
            with torch.no_grad():Qcur=m.critic_head(torch.tensor(QF,dtype=torch.float32,device="cuda")).cpu().numpy()
    
            # Close/far defined relative to expanded support, fixed for all candidate fits.
            fd=nearest_dist(SFe,QF);close=fd<=np.quantile(fd,.25);far=fd>=np.quantile(fd,.75)
    
            fits={}
            fits["current_head"]=None
            fits["selected12_uniform"]=wridge(SFs,SYs,None,1.0)
            fits["expanded_uniform"]=wridge(SFe,SYe,None,1.0)
            wsp=source_phase_weights(src,ph)
            fits["expanded_source_phase_balanced"]=wridge(SFe,SYe,wsp,1.0)
            wcmd=command_bin_weights(SCe,4)
            fits["expanded_command_balanced"]=wridge(SFe,SYe,wcmd,1.0)
            woracle,oracle_d=semantic_proximity_weights(SFe,QF)
            fits["expanded_semantic_proximity_ORACLE"]=wridge(SFe,SYe,woracle,1.0)
    
            result={"schema":"rv1_head_fitting_support_weighting_v1","measurement_only":True,
                    "support_counts":{"selected_units":len(selected),"expanded_units":len(expanded),
                                      "selected_samples":len(SFs),"expanded_samples":len(SFe),"semantic_samples":len(QF)},
                    "selected_indices":sel,"geometry":{},"fits":{},"support_fit":{}}
    
            # current and candidate eval
            candidates={"current_head":Qcur}
            for name,fit in fits.items():
                if fit is not None:candidates[name]=predict(QF,fit)
            for name,P in candidates.items():
                result["fits"][name]={"all":metrics(QY,P),"close":metrics(QY,P,close),"far":metrics(QY,P,far)}
    
            # support geometry + support residuals for each actual fit dataset/weighting
            specs={
                "selected12_uniform":(SFs,SYs,np.ones(len(SFs))),
                "expanded_uniform":(SFe,SYe,np.ones(len(SFe))),
                "expanded_source_phase_balanced":(SFe,SYe,wsp),
                "expanded_command_balanced":(SFe,SYe,wcmd),
                "expanded_semantic_proximity_ORACLE":(SFe,SYe,woracle)}
            for name,(F,Y,w) in specs.items():
                result["geometry"][name]=geometry(F,w,1.0)
                P=predict(F,fits[name])
                # weighted and unweighted residual summaries
                result["support_fit"][name]={
                    "unweighted_mae":[float(np.mean(np.abs(P[:,j]-Y[:,j]))) for j in range(4)],
                    "weighted_mae":[float(np.average(np.abs(P[:,j]-Y[:,j]),weights=w)) for j in range(4)]}
    
            # Current head residual on selected and expanded support.
            with torch.no_grad():
                Ps=m.critic_head(torch.tensor(SFs,dtype=torch.float32,device="cuda")).cpu().numpy()
                Pe=m.critic_head(torch.tensor(SFe,dtype=torch.float32,device="cuda")).cpu().numpy()
            result["support_fit"]["current_head_selected"]={"unweighted_mae":[float(np.mean(np.abs(Ps[:,j]-SYs[:,j]))) for j in range(4)]}
            result["support_fit"]["current_head_expanded"]={"unweighted_mae":[float(np.mean(np.abs(Pe[:,j]-SYe[:,j]))) for j in range(4)]}
    
            # Head-specific attribution of improvement on close semantic set.
            summary=[]
            base=result["fits"]["current_head"]["close"]["mae"]
            for j,h in enumerate(HEADS):
                row={"head":h,"current_close_mae":base[j]}
                for name in fits:
                    if name=="current_head":continue
                    v=result["fits"][name]["close"]["mae"][j]
                    row[name+"_close_mae"]=v
                    row[name+"_gain_frac"]=float((base[j]-v)/(base[j]+1e-12))
                summary.append(row)
            result["summary"]={"close_set_head_fitting":summary,
                "oracle_support_distance":{"mean":float(oracle_d.mean()),"p50":float(np.quantile(oracle_d,.5)),"p95":float(np.quantile(oracle_d,.95))}}
            out=a.output_dir/"head_fitting_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":hashlib.sha256(out.read_bytes()).hexdigest()},indent=2)+"\n")
            print(json.dumps(result["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_rv1_late_phase_support_representativeness_audit():
    """Run former rv1_late_phase_support_representativeness_audit.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
    TRAIN_REPORT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/phase_repair_report.json"
    OUT_DEFAULT=ROOT/"runs/rv1_late_phase_support_representativeness-2026-09-23"
    NENV=8;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    ORDER=("T","A","O","S","C")
    ALPHAS=(0.0,.25,.5,.75,1.0);PATHS=(("T","A"),("T","O"),("T","S"))
    OBJ_NAMES=("Tracking","Angular","Orientation","Smoothness")
    CMD_PREF_NAMES=("cmd_vx","cmd_vy","cmd_wz","w_T","w_A","w_O","w_S")
    PHYS_NAMES=("tilt_deg","ang_vel_xy","abs_vz","joint_pos_rms","joint_vel_rms","root_vx","root_vy")
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    def interp(a,b,x):return (1-x)*PREFS[a]+x*PREFS[b]
    
    def collect64(env,m,w_np,seed,label):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        w=torch.tensor(w_np,device="cuda") if np.asarray(w_np).ndim==2 else torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        obs=[];R=[];D=[];cmd=[];phys=[]
        with torch.no_grad():
            for _ in range(64):
                data=robot.data;c=env.unwrapped.command_manager.get_command("base_velocity")
                cmd.append(c.detach().cpu().numpy())
                phys.append(np.stack([
                    tilt(data.root_quat_w).cpu().numpy(),
                    torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).cpu().numpy(),
                    data.root_lin_vel_b[:,2].abs().cpu().numpy(),
                    torch.sqrt(torch.mean(data.joint_pos**2,dim=1)).cpu().numpy(),
                    torch.sqrt(torch.mean(data.joint_vel**2,dim=1)).cpu().numpy(),
                    data.root_lin_vel_b[:,0].cpu().numpy(),
                    data.root_lin_vel_b[:,1].cpu().numpy()],axis=1))
                obs.append(cur)
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(torch.tensor(normalized_objective_vector(terms(raw,names),shape=(NENV,)),device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        obs=torch.stack(obs);R=torch.stack(R);D=torch.stack(D).bool()
        st,en=32,64
        po=obs[st:en].reshape(-1,obs.shape[-1]);pw=w.repeat(en-st,1)
        with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).cpu().numpy()
        Y=trunc(R[st:en],D[st:en]).reshape(-1,4).cpu().numpy()
        C=np.asarray(cmd)[st:en].reshape(-1,3)
        W=np.repeat(w.cpu().numpy()[None,:,:],en-st,axis=0).reshape(-1,4)
        P=np.asarray(phys)[st:en].reshape(-1,len(PHYS_NAMES))
        return {"feature":F,"cmdpref":np.c_[C,W],"physical":P,"target":Y,
                "label":np.array([label]*len(F),object)}
    
    def z_nn_audit(S,Q,names,max_q=12000):
        S=np.asarray(S,float);Q=np.asarray(Q,float)
        mu=S.mean(0);sd=S.std(0)+1e-8
        Sz=(S-mu)/sd;Qz=(Q-mu)/sd
        # Cap query count deterministically for cost only.
        if len(Qz)>max_q:
            idx=np.linspace(0,len(Qz)-1,max_q,dtype=int);Qz=Qz[idx];Qraw=Q[idx]
        else:Qraw=Q
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        St=torch.tensor(Sz,dtype=torch.float32,device=dev)
        # leave-one-out support NN
        dss=torch.cdist(St,St)
        dss.fill_diagonal_(float("inf"))
        base=dss.min(dim=1).values.cpu().numpy()/np.sqrt(S.shape[1])
        qs=[]
        for i in range(0,len(Qz),512):
            Qt=torch.tensor(Qz[i:i+512],dtype=torch.float32,device=dev)
            qs.append((torch.cdist(Qt,St).min(dim=1).values/np.sqrt(S.shape[1])).cpu().numpy())
        qdist=np.concatenate(qs)
        thr=float(np.quantile(base,.95))
        mean_shift=(Q.mean(0)-mu)/sd
        # std ratio and range outside fraction by dimension
        qsd=Q.std(0);ratio=qsd/sd
        top=np.argsort(np.abs(mean_shift))[::-1][:min(10,len(names))]
        return {"support_loo_nn":{"mean":float(base.mean()),"p50":float(np.quantile(base,.5)),"p95":thr},
                "semantic_to_support_nn":{"mean":float(qdist.mean()),"p50":float(np.quantile(qdist,.5)),
                                         "p95":float(np.quantile(qdist,.95)),
                                         "out_of_support_fraction_gt_support_p95":float(np.mean(qdist>thr))},
                "top_mean_shifts":[{"dim":str(names[i]),"z_mean_shift":float(mean_shift[i]),"std_ratio":float(ratio[i])} for i in top]}
    
    def target_audit(S,Q):
        S=np.asarray(S,float);Q=np.asarray(Q,float);out=[]
        for j,n in enumerate(OBJ_NAMES):
            sm,ss=S[:,j].mean(),S[:,j].std()+1e-12;qm,qs=Q[:,j].mean(),Q[:,j].std()
            # empirical quantile exceedance vs support 1/99 percentiles
            lo,hi=np.quantile(S[:,j],[.01,.99])
            out.append({"objective":n,"support_mean":float(sm),"semantic_mean":float(qm),
                        "mean_shift_in_support_sd":float((qm-sm)/ss),"support_std":float(ss),"semantic_std":float(qs),
                        "semantic_outside_support_1_99_fraction":float(np.mean((Q[:,j]<lo)|(Q[:,j]>hi)))})
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);a=ap.parse_args()
        if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            m=T4SharedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
    
            tr=json.load(open(TRAIN_REPORT));row75=tr["rows"][-1]
            anchor_specs=tr["anchor_specs"];late_sel=row75["selected"]["late"]
            support=[]
            # Exact selected anchor seed specs, recollected with current policy.
            for local in late_sel["anchor"]:
                k,seed=anchor_specs[local];w=pref_batch(k,torch.device("cuda")).cpu().numpy()
                support.append(collect64(env,m,w,seed,f"anchor{k}"))
            # At update75, adaptive pool indices 0..23 correspond updates 52..75.
            for idx in late_sel["adaptive"]:
                u=52+idx;seed=73001+200000+u*223
                w=pref_batch(u+17,torch.device("cuda")).cpu().numpy()
                support.append(collect64(env,m,w,seed,f"adaptive_u{u}"))
    
            semantic_endpoint=[];semantic_cont=[]
            for suite in range(4):
                seed=840001+suite
                for lab in ORDER:
                    semantic_endpoint.append(collect64(env,m,PREFS[lab],seed,f"endpoint_{lab}_s{suite}"))
            for pi,(x,y) in enumerate(PATHS):
                for suite in range(4):
                    seed=850001+pi*1000+suite
                    for alpha in ALPHAS:
                        semantic_cont.append(collect64(env,m,interp(x,y,alpha),seed,f"{x}-{y}_{alpha}_s{suite}"))
    
            def cat(rows,key):return np.concatenate([r[key] for r in rows],axis=0)
            result={"schema":"rv1_late_phase_support_representativeness_v1","measurement_only":True,
                    "support_reconstruction":{"note":"Exact u75 support seed/preference specifications reconstructed and recollected with model_75; historical adaptive tensors were not serialized.",
                                              "late_selected":late_sel,"anchor_specs":anchor_specs,
                                              "adaptive_index_to_update":{str(i):52+i for i in late_sel["adaptive"]}},
                    "counts":{"support_states":int(len(cat(support,"feature"))),
                              "semantic_endpoint_states":int(len(cat(semantic_endpoint,"feature"))),
                              "semantic_continuum_states":int(len(cat(semantic_cont,"feature")))}}
            for name,rows in (("endpoint",semantic_endpoint),("continuum",semantic_cont),("all",semantic_endpoint+semantic_cont)):
                result.setdefault("comparisons",{})[name]={
                    "feature_space":z_nn_audit(cat(support,"feature"),cat(rows,"feature"),[f"critic_feature_{i}" for i in range(cat(support,"feature").shape[1])]),
                    "command_preference":z_nn_audit(cat(support,"cmdpref"),cat(rows,"cmdpref"),CMD_PREF_NAMES),
                    "physical_state":z_nn_audit(cat(support,"physical"),cat(rows,"physical"),PHYS_NAMES),
                    "target_return":target_audit(cat(support,"target"),cat(rows,"target"))}
            # Evidence rule: representativeness concern if semantic OOS fraction is high in feature/physical/cmdpref
            allc=result["comparisons"]["all"]
            oos={k:allc[k]["semantic_to_support_nn"]["out_of_support_fraction_gt_support_p95"] for k in ("feature_space","command_preference","physical_state")}
            result["summary"]={"oos_fraction":oos,
                "representativeness_gap_supported":bool(any(v>.25 for v in oos.values())),
                "interpretation":"support-coverage mismatch" if any(v>.25 for v in oos.values()) else "coverage appears adequate; investigate mapping/generalization"}
            out=a.output_dir/"representativeness_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
            h=hashlib.sha256(out.read_bytes()).hexdigest()
            (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":h,
                "checkpoint":str(CKPT.relative_to(ROOT))},indent=2)+"\n")
            print(json.dumps({"summary":result["summary"],"counts":result["counts"],
                              "all_feature":allc["feature_space"]["semantic_to_support_nn"],
                              "all_cmdpref":allc["command_preference"]["semantic_to_support_nn"],
                              "all_physical":allc["physical_state"]["semantic_to_support_nn"],
                              "target":allc["target_return"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "rv1_c_critic_foundation_revalidation": run_rv1_c_critic_foundation_revalidation,
    "rv1_critic_persistence_repair": run_rv1_critic_persistence_repair,
    "rv1_critic_persistence_repair_v2": run_rv1_critic_persistence_repair_v2,
    "rv1_critic_persistence_revalidation": run_rv1_critic_persistence_revalidation,
    "rv1_critic_persistence_revalidation_v2": run_rv1_critic_persistence_revalidation_v2,
    "rv1_critic_phase_balanced_repair": run_rv1_critic_phase_balanced_repair,
    "rv1_critic_phase_balanced_revalidation": run_rv1_critic_phase_balanced_revalidation,
    "rv1_head_fitting_support_weighting_audit": run_rv1_head_fitting_support_weighting_audit,
    "rv1_late_phase_support_representativeness_audit": run_rv1_late_phase_support_representativeness_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
