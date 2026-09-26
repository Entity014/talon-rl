#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,json,sys,hashlib,copy
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
OUT=ROOT/"runs/v2b_bounded_deltaa_multiseed-2026-09-24"
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
        arms=("control","action_response")
        models={};rows={}
        act_refs={}
        activation={"action_response":{lab:None for lab in ORDER}}
        for arm in arms:
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval();models[arm]=m;rows[arm]=[]
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
                else:
                    gr,rmeta=action_rehearsal_gradient(m,act_refs);rmeta["type"]="action_response"
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
                else:
                    active_before=[]
                rows[arm].append({"update":k,"active_refs_before_update":active_before,
                    "new_axes_activated":new,"step_norm":step_norm,"mixed_meta":mmeta,"rehearsal_meta":rmeta,"budget_meta":bmeta,
                    "update_cos_prev":cp,"preference_separation":preference_separation(m,probe),"endpoint":ep,
                    "semantic_scores":{lab:ep[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":m.state_dict(),"arm":arm,"update":k},args.output_dir/f"{arm}_u{k}.pt")
            (args.output_dir/"bounded_deltaa_partial.json").write_text(json.dumps({"activation":activation,"active":{"action_response":sorted(act_refs)},"arms":rows},indent=2)+"\n")
        report={"schema":"v2b_bounded_deltaa_multiseed_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
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
        out=args.output_dir/"bounded_deltaa_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),
          "script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
        print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"]},indent=2))
    finally:
        if env is not None: env.close()
        app.close()
if __name__=="__main__": main()
