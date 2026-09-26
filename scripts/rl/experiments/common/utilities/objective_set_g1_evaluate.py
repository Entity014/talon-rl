#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,itertools,json,sys,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic,canonical_tokens
from talon_rl.rewards.objectives import normalized_objective_vector
PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
G0=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
CONTRACT=ROOT/"docs/contracts/objective_set/objective-set-g1-variable-cardinality-contract.md"
NENV=8;STEPS=64;SUITES=4;G=.99
ORDER=("T","A","O","S");IDX={x:i for i,x in enumerate(ORDER)}
PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
FOLDS={"G1-2":{"seen":(3,4),"holdout":2},"G1-3":{"seen":(2,4),"holdout":3}}
ALPHAS=(0.,.25,.5,.75,1.)
AUTH_THR=.75;TOL=1e-6

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def tilt_deg(q):
    _,x,y,_=[q[:,i] for i in range(4)]
    return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def segret(R,D,st,en):
    out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
    for t in range(en-1,st-1,-1):
        run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
    return out
def mono(vals):
    vals=np.asarray(vals,float);d=np.diff(vals);target=np.sign(vals[-1]-vals[0])
    return 0.0 if target==0 else float(np.mean(np.sign(d)==target))
def between(vals):
    vals=np.asarray(vals,float);lo=min(vals[0],vals[-1]);hi=max(vals[0],vals[-1])
    return float(np.mean((vals[1:-1]>=lo-1e-9)&(vals[1:-1]<=hi+1e-9)))
def make_set(ids,w,device):
    ids=torch.tensor(ids,device=device,dtype=torch.long)
    tok=canonical_tokens(device=device)[ids].unsqueeze(0).repeat(NENV,1,1)
    ww=torch.tensor(w,device=device,dtype=torch.float32).unsqueeze(0).repeat(NENV,1)
    return tok,ww
def center_w(m):return np.full(m,1/m,np.float32)
def heavy_w(m,h):
    w=np.full(m,.30/(m-1),np.float32);w[h]=.70;return w
def interp(a,b,alpha):return ((1-alpha)*a+alpha*b).astype(np.float32)

def evaluate(env,mgr,robot,model,ids,w_np,seed):
    tok,w=make_set(ids,w_np,torch.device("cuda"));ids_t=torch.tensor(ids,device="cuda").repeat(NENV,1)
    cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
    R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda");done_any=np.zeros(NENV,bool)
    with torch.no_grad():
        for _ in range(STEPS):
            V.append(model.query_values_from_set(cur,tok,w,tok).cpu().numpy())
            a=model.act_inference_from_set(cur,tok,w)
            nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            full=normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt
            R.append(full[:,ids])
            dd=(te|tr).cpu().numpy();D.append(dd);done_any|=dd
            data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean());wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
            phys.append({"tracking_error":vx+wz,"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                         "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
            prev=a;cur=ot(nxt).cuda()
    R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H0=segret(R,D,0,32);H1=segret(R,D,32,64)
    p={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
    return {"active_objective_mean":R.mean((0,1)).tolist(),"physical":p,"survival":float(1-done_any.mean()),
      "critic":{"ev":[ev(H0[:,:,j],V[:32,:,j]) for j in range(len(ids))]+[ev(H1[:,:,j],V[32:,:,j]) for j in range(len(ids))],
                "bias":[float(np.mean(V[:32,:,j]-H0[:,:,j])) for j in range(len(ids))]+[float(np.mean(V[32:,:,j]-H1[:,:,j])) for j in range(len(ids))]}}
def authority(model,g0,probe,ids):
    dev=probe.device;m=len(ids);base=canonical_tokens(device=dev)[torch.tensor(ids,device=dev)]
    tok=base.unsqueeze(0).repeat(len(probe),1,1)
    prefs=[center_w(m)]+[heavy_w(m,h) for h in range(m)]
    def acts(mod):
        out=[]
        with torch.no_grad():
            for wv in prefs:
                w=torch.tensor(wv,device=dev).repeat(len(probe),1)
                out.append(mod.act_inference_from_set(probe,tok,w))
        vals=[]
        for i in range(len(out)):
            for j in range(i+1,len(out)):
                vals.append(float(torch.linalg.vector_norm(out[i]-out[j],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    pair=acts(model);pair0=acts(g0)
    wc=torch.tensor(center_w(m),device=dev)
    D=torch.eye(m,device=dev)[:,:m-1]-torch.eye(m,device=dev)[:,[-1]]
    Q,_=torch.linalg.qr(D,mode="reduced")
    from torch.func import jacrev,vmap
    def jnorm(mod):
        def f(o,w):
            tt=base.unsqueeze(0);ww=w.unsqueeze(0)
            return mod.act_inference_from_set(o.unsqueeze(0),tt,ww).squeeze(0)
        J=vmap(jacrev(f,argnums=1),in_dims=(0,None))(probe,wc)@Q
        return float(torch.linalg.matrix_norm(J,ord="fro",dim=(1,2)).mean().detach().cpu())
    j=jnorm(model);j0=jnorm(g0)
    return {"pairwise":pair,"pairwise_g0":pair0,"pairwise_retention":pair/(pair0+1e-12),
            "tangent":j,"tangent_g0":j0,"tangent_retention":j/(j0+1e-12)}

def permutation_drift(model,probe,ids):
    m=len(ids);base=canonical_tokens(device=probe.device)[torch.tensor(ids,device=probe.device)]
    tok=base.unsqueeze(0).repeat(len(probe),1,1);w=torch.tensor(center_w(m),device=probe.device).repeat(len(probe),1)
    with torch.no_grad():
        ar=model.act_inference_from_set(probe,tok,w);vr=model.query_values_from_set(probe,tok,w,tok)
    da=dv=0.0
    for p in itertools.permutations(range(m)):
        q=torch.tensor(p,device=probe.device)
        with torch.no_grad():
            a=model.act_inference_from_set(probe,tok[:,q],w[:,q]);v=model.query_values_from_set(probe,tok[:,q],w[:,q],tok)
        da=max(da,float((a-ar).abs().max().cpu()));dv=max(dv,float((v-vr).abs().max().cpu()))
    return {"action":da,"value":dv}
def eval_set(env,mgr,robot,model,g0,probe,ids,set_index,role):
    m=len(ids);labs=[ORDER[i] for i in ids];rows=[];cw=center_w(m)
    for suite in range(SUITES):
        seed=860001+set_index*100+suite
        c=evaluate(env,mgr,robot,model,ids,cw,seed);c.update({"suite":suite,"label":"C"});rows.append(c)
        for h,lab in enumerate(labs):
            q=evaluate(env,mgr,robot,model,ids,heavy_w(m,h),seed);q.update({"suite":suite,"label":lab});rows.append(q)
    endpoints={}
    for h,lab in enumerate(labs):
        oo=[];pp=[];ss=[];do=[];dp=[]
        for suite in range(SUITES):
            r=next(x for x in rows if x["suite"]==suite and x["label"]==lab);c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
            x=r["active_objective_mean"][h]-c["active_objective_mean"][h];y=r["physical"][PHYS[lab]]-c["physical"][PHYS[lab]]
            oo.append(x>0);pp.append(y<0);ss.append(r["survival"]);do.append(x);dp.append(y)
        endpoints[lab]={"objective_correct_fraction":float(np.mean(oo)),"physical_correct_fraction":float(np.mean(pp)),
                        "mean_objective_delta":float(np.mean(do)),"mean_physical_delta":float(np.mean(dp)),"min_survival":float(np.min(ss)),
                        "pass":bool(np.mean(oo)>=.75 and np.mean(pp)>=.75 and np.min(ss)>=.95)}
    cont=[]
    for pi,(ha,hb) in enumerate(itertools.combinations(range(m),2)):
        wa=heavy_w(m,ha);wb=heavy_w(m,hb)
        for suite in range(SUITES):
            rr=[]
            seed=870001+set_index*1000+pi*10+suite
            for alpha in ALPHAS:rr.append(evaluate(env,mgr,robot,model,ids,interp(wa,wb,alpha),seed))
            for h in (ha,hb):
                lab=labs[h];ov=[x["active_objective_mean"][h] for x in rr];pv=[x["physical"][PHYS[lab]] for x in rr]
                cont.append({"axis":lab,"objective_monotonic":mono(ov),"objective_between":between(ov),"physical_monotonic":mono(pv),"physical_between":between(pv)})
    mono_f=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont])) if cont else 1.0
    between_f=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont])) if cont else 1.0
    center_between=[]
    for suite in range(SUITES):
        hs=[next(x for x in rows if x["suite"]==suite and x["label"]==lab) for lab in labs];cc=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
        for h in range(m):
            vals=[x["active_objective_mean"][h] for x in hs];cv=cc["active_objective_mean"][h]
            center_between.append(min(vals)-1e-9<=cv<=max(vals)+1e-9)
    cev=[];cb=[];surv=[]
    for x in rows:cev+=x["critic"]["ev"];cb+=x["critic"]["bias"];surv.append(x["survival"])
    critic={"ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),"mean_abs_bias":float(np.mean(np.abs(cb)))}
    auth=authority(model,g0,probe,ids);perm=permutation_drift(model,probe,ids)
    rng=np.random.default_rng(20260925+set_index);interiors=[]
    for k in range(3):
        wv=rng.dirichlet(np.ones(m)).astype(np.float32);qs=[]
        for suite in range(SUITES):
            qs.append(evaluate(env,mgr,robot,model,ids,wv,880001+set_index*100+k*10+suite))
        interiors.append({"weights":wv.tolist(),"min_survival":float(min(q["survival"] for q in qs))})
    required=[lab for lab in labs if lab in ("T","A","O")]
    criteria={"required_semantics":all(endpoints[x]["pass"] for x in required),
      "center_compromise":float(np.mean(center_between))>=.75,
      "continuum_monotonicity":mono_f>=.65,"continuum_between":between_f>=.65,
      "critic_valid":critic["ev_mean"]>0 and critic["negative_fraction"]<=.25,
      "endpoint_survival":min(surv)>=.95,
      "authority_pairwise":auth["pairwise_retention"]>=AUTH_THR,
      "authority_tangent":auth["tangent_retention"]>=AUTH_THR,
      "permutation":perm["action"]<=TOL and perm["value"]<=TOL}
    return {"ids":list(ids),"labels":labs,"cardinality":m,"role":role,"endpoint":endpoints,
      "center_compromise_fraction":float(np.mean(center_between)),"continuum":{"monotonicity":mono_f,"between":between_f},
      "critic":critic,"authority":auth,"permutation_drift":perm,"interiors":interiors,
      "min_endpoint_survival":float(min(surv)),"criteria":criteria,"pass":bool(all(criteria.values()))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--fold",choices=tuple(FOLDS),required=True);ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args();run=ROOT/f"runs/objective_set_{args.fold.lower().replace('-','_')}_seed{args.seed}-2026-09-25";ck=run/"model_30.pt"
    out=run/"g1_evaluation.json"
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
        mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"];probe=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        model=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();model.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);model.eval()
        g0=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();g0.load_state_dict(torch.load(G0,map_location="cuda",weights_only=False)["model"]);g0.eval()
        sets=[];si=0
        for m in (2,3,4):
            role="heldout" if m==FOLDS[args.fold]["holdout"] else ("seen" if m in FOLDS[args.fold]["seen"] else "other")
            for ids in itertools.combinations(range(4),m):
                sets.append(eval_set(env,mgr,robot,model,g0,probe,ids,si,role));print("SET",args.fold,args.seed,ids,role,sets[-1]["pass"],flush=True);si+=1
        held=[x for x in sets if x["role"]=="heldout"];seen=[x for x in sets if x["role"]=="seen"];anchor=next(x for x in sets if x["cardinality"]==4)
        held_pass=all(x["pass"] for x in held);anchor_pass=anchor["criteria"]["required_semantics"] and anchor["criteria"]["critic_valid"] and anchor["criteria"]["endpoint_survival"]
        rep={"schema":"objective_set_g1_evaluation_v1","fold":args.fold,"seed":args.seed,"checkpoint":str(ck.relative_to(ROOT)),
          "sets":sets,"summary":{"heldout_set_count":len(held),"heldout_pass_count":sum(x["pass"] for x in held),"seen_pass_count":sum(x["pass"] for x in seen),
          "seen_set_count":len(seen),"heldout_cardinality_pass":held_pass,"full_set_anchor_pass":bool(anchor_pass),
          "fold_seed_pass":bool(held_pass and anchor_pass)}}
        out.write_text(json.dumps(rep,indent=2)+"\n")
        print("FINAL",json.dumps(rep["summary"]),flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
