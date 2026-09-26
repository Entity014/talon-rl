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
ORDER=("T","A","O","S");IDX={x:i for i,x in enumerate(ORDER)}
PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
NENV=8;STEPS=64;SUITES=4;G=.99
FOLDS={"G1-2":{"holdout":2},"G1-3":{"holdout":3}}

def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def tilt_deg(q):
    _,x,y,_=[q[:,i] for i in range(4)]
    return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def segret(R,D,st,en):
    out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
    for t in range(en-1,st-1,-1):
        run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
    return out
def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def active_spec(ids):
    m=len(ids);center=np.full(m,1/m,np.float32);prefs={"C":center}
    for k,i in enumerate(ids):
        w=np.full(m,.30/(m-1),np.float32);w[k]=.70;prefs[ORDER[i]]=w
    return prefs
def evaluate(env,m,mgr,robot,ids,w_np,seed):
    dev=torch.device("cuda");ids_t=torch.tensor(ids,device=dev,dtype=torch.long)
    tok=canonical_tokens(device=dev)[ids_t].unsqueeze(0).repeat(NENV,1,1)
    w=torch.tensor(w_np,device=dev).repeat(NENV,1)
    cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
    R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,12),device=dev);done_any=np.zeros(NENV,bool)
    with torch.no_grad():
        for _ in range(STEPS):
            V.append(m.query_values_from_set(cur,tok,w,tok).cpu().numpy())
            a=m.act_inference_from_set(cur,tok,w)
            nxt,_,te,tr,_=env.step(a)
            full=normalized_objective_vector(terms(mgr._step_reward.detach().cpu().numpy(),list(mgr.active_terms)),shape=(NENV,))
            R.append(full[:,ids])
            dd=(te|tr).cpu().numpy();D.append(dd);done_any|=dd
            data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            vx=(data.root_lin_vel_b[:,0]-cmd[:,0]).abs();wz=(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()
            phys.append({"tracking_error":float((vx+wz).mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
              "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
            prev=a;cur=ot(nxt).cuda()
    R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H0=segret(R,D,0,32);H1=segret(R,D,32,64)
    return {"normalized_objective_mean":R.mean((0,1)).tolist(),"physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
      "survival":float(1-done_any.mean()),"critic":{"first_h32_ev":[ev(H0[:,:,j],V[:32,:,j]) for j in range(len(ids))],
      "second_h32_ev":[ev(H1[:,:,j],V[32:,:,j]) for j in range(len(ids))]}}
def authority(model,ids):
    P=torch.tensor(np.load(PROBE)["obs"],device="cuda",dtype=torch.float32)
    ids_t=torch.tensor(ids,device="cuda");tok=canonical_tokens(device=P.device)[ids_t].unsqueeze(0).repeat(len(P),1,1)
    acts=[];prefs=active_spec(ids)
    with torch.no_grad():
        for _,wv in prefs.items():
            w=torch.tensor(wv,device=P.device).repeat(len(P),1)
            acts.append(model.act_inference_from_set(P,tok,w))
    ds=[]
    for i in range(len(acts)):
        for j in range(i+1,len(acts)):
            ds.append(float(torch.linalg.vector_norm(acts[i]-acts[j],dim=-1).mean().cpu()))
    return {"mean_pairwise_action_distance":float(np.mean(ds)),"min_pairwise_action_distance":float(np.min(ds))}

def perm_gate(model,ids):
    P=torch.tensor(np.load(PROBE)["obs"][:64],device="cuda",dtype=torch.float32)
    ids_t=torch.tensor(ids,device=P.device);base=canonical_tokens(device=P.device)[ids_t]
    prefs=active_spec(ids);w0=torch.tensor(prefs["C"],device=P.device).repeat(len(P),1)
    tok0=base.unsqueeze(0).repeat(len(P),1,1)
    with torch.no_grad():ar=model.act_inference_from_set(P,tok0,w0);vr=model.query_values_from_set(P,tok0,w0,tok0)
    ma=mv=0.
    for p in itertools.permutations(range(len(ids))):
        q=torch.tensor(p,device=P.device);tp=tok0[:,q];wp=w0[:,q]
        with torch.no_grad():a=model.act_inference_from_set(P,tp,wp);v=model.query_values_from_set(P,tp,wp,tok0)
        ma=max(ma,float((a-ar).abs().max().cpu()));mv=max(mv,float((v-vr).abs().max().cpu()))
    return {"action_max_drift":ma,"value_max_drift":mv,"pass":ma<=1e-6 and mv<=1e-6}
def eval_set(env,m,mgr,robot,ids):
    prefs=active_spec(ids);rows=[]
    for suite in range(SUITES):
        seed=860001+suite
        for lab,w in prefs.items():
            q=evaluate(env,m,mgr,robot,ids,w,seed);q.update({"suite":suite,"label":lab});rows.append(q)
    ep={};active_labels=[ORDER[i] for i in ids]
    for pos,lab in enumerate(active_labels):
        oo=[];pp=[];sv=[];do=[];dp=[]
        for suite in range(SUITES):
            r=next(x for x in rows if x["suite"]==suite and x["label"]==lab);c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
            x=r["normalized_objective_mean"][pos]-c["normalized_objective_mean"][pos];y=r["physical"][PHYS[lab]]-c["physical"][PHYS[lab]]
            oo.append(x>0);pp.append(y<0);sv.append(r["survival"]);do.append(x);dp.append(y)
        ep[lab]={"objective_correct_fraction":float(np.mean(oo)),"physical_correct_fraction":float(np.mean(pp)),
          "mean_objective_delta_vs_center":float(np.mean(do)),"mean_physical_delta_vs_center":float(np.mean(dp)),"min_survival":float(np.min(sv)),
          "pass":bool(np.mean(oo)>=.75 and np.mean(pp)>=.75 and np.min(sv)>=.95)}
    between=[]
    for suite in range(SUITES):
        hs=[next(x for x in rows if x["suite"]==suite and x["label"]==lab) for lab in active_labels]
        c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
        for j in range(len(ids)):
            vals=[x["normalized_objective_mean"][j] for x in hs];cv=c["normalized_objective_mean"][j]
            between.append(min(vals)-1e-9<=cv<=max(vals)+1e-9)
    cev=[]
    for x in rows:cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"]
    crit={"h32_ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),
          "pass":bool(np.mean(cev)>0 and np.mean(np.asarray(cev)<0)<=.25)}
    req=[lab for lab in active_labels if lab!="S"]
    return {"ids":list(ids),"labels":active_labels,"endpoint":ep,"required_TAO_pass":bool(all(ep[x]["pass"] for x in req)),
      "center_compromise_fraction":float(np.mean(between)),"center_pass":bool(np.mean(between)>=.75),"critic":crit,
      "min_survival":float(min(x["survival"] for x in rows)),"authority":authority(m,ids),"permutation":perm_gate(m,ids)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--fold",choices=tuple(FOLDS),required=True);ap.add_argument("--seed",type=int,required=True);args=ap.parse_args()
    run=ROOT/f"runs/objective_set_{args.fold.lower().replace('-','_')}_seed{args.seed}-2026-09-25";ck=run/"model_30.pt"
    out=run/"endpoint_screen";out.mkdir(exist_ok=True)
    tr=json.load(open(run/"training_report.json"))
    if not tr["leakage_free"]:raise RuntimeError("training leakage report is not clean")
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=args.seed);o=ot(o).cuda();mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(ck,map_location="cuda",weights_only=False)["model"]);m.eval()
        hold=FOLDS[args.fold]["holdout"];sets=list(itertools.combinations(range(4),hold));results={}
        for ids in sets:
            key="".join(ORDER[i] for i in ids);results[key]=eval_set(env,m,mgr,robot,ids);print("SET",args.fold,args.seed,key,json.dumps({"req":results[key]["required_TAO_pass"],"ep":results[key]["endpoint"],"critic":results[key]["critic"]}),flush=True)
        anchor=eval_set(env,m,mgr,robot,(0,1,2,3))
        held_req=all(x["required_TAO_pass"] for x in results.values());held_crit=all(x["critic"]["pass"] for x in results.values())
        held_perm=all(x["permutation"]["pass"] for x in results.values());held_surv=all(x["min_survival"]>=.95 for x in results.values())
        anchor_req=anchor["required_TAO_pass"];anchor_crit=anchor["critic"]["pass"];anchor_surv=anchor["min_survival"]>=.95
        criteria={"leakage_free":True,"heldout_TAO_endpoints":held_req,"heldout_critic":held_crit,"heldout_permutation":held_perm,
          "heldout_survival":held_surv,"m4_anchor_TAO":anchor_req,"m4_anchor_critic":anchor_crit,"m4_anchor_survival":anchor_surv}
        rep={"schema":"objective_set_g1_endpoint_screen_v1","fold":args.fold,"seed":args.seed,"heldout_cardinality":hold,
          "heldout_sets":results,"m4_anchor":anchor,"criteria":criteria,"endpoint_screen_pass":bool(all(criteria.values()))}
        (out/"endpoint_screen.json").write_text(json.dumps(rep,indent=2)+"\n");print("SCREEN",args.fold,args.seed,json.dumps(criteria),flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
