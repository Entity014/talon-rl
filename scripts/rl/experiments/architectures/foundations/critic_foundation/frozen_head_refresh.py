#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
CKPT=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/model_75.pt"
REP=ROOT/"runs/rv1_critic_phase_balanced_repair-2026-09-23/phase_repair_report.json"
OUT=ROOT/"runs/rv1_actor_frozen_head_refresh-2026-09-23"
NENV=8;G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
"C":np.array([.25,.25,.25,.25],np.float32)}
ORDER=("T","A","O","S","C")
HEADS=("Tracking","Angular","Orientation","Smoothness")
def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def pref_batch(update,device):
    labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
    return torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
def trunc(rt,dt):
    out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
    for t in range(len(rt)-1,-1,-1):
        run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
    return out
def ridge(F,Y,l2=1.0):
    A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
    sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
    return sol[:-1],sol[-1]
def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def collect64(env,m,w_np,seed,label):
    from talon_rl.rewards.objectives import normalized_objective_vector
    mgr=env.unwrapped.reward_manager
    w=torch.tensor(w_np,device="cuda") if np.asarray(w_np).ndim==2 else torch.tensor(w_np,device="cuda").repeat(NENV,1)
    cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];R=[];D=[]
    with torch.no_grad():
        for _ in range(64):
            obs.append(cur)
            a=m.act_inference_with_preference(cur,w)
            nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            vec=normalized_objective_vector({n:raw[:,i] for i,n in enumerate(names)},shape=(NENV,))
            R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);D.append((te|tr).cuda())
            cur=ot(nxt).cuda()
    O=torch.stack(obs);R=torch.stack(R);D=torch.stack(D).bool()
    out=[]
    for phase,(st,en) in (("first",(0,32)),("second",(32,64))):
        po=O[st:en].reshape(-1,O.shape[-1]);pw=w.repeat(en-st,1)
        with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).cpu().numpy()
        Y=trunc(R[st:en],D[st:en]).reshape(-1,4).cpu().numpy()
        out.append({"phase":phase,"F":F,"Y":Y,"label":label})
    return out
def fit_units(units):
    F=np.concatenate([u["F"] for u in units]);Y=np.concatenate([u["Y"] for u in units])
    return ridge(F,Y,1.0),{"samples":len(F),"support_mae":[float(np.mean(np.abs((F@ridge(F,Y,1.0)[0]+ridge(F,Y,1.0)[1])[:,j]-Y[:,j]))) for j in range(4)]}
def pred(F,fit): return F@fit[0]+fit[1]
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
        m=T4SharedActorCritic(o.shape[-1],ad).cuda()
        m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
        tr=json.load(open(REP));row75=tr["rows"][-1];sel=row75["selected"];anchor_specs=tr["anchor_specs"]

        # Recollect all current-policy anchor/adaptive support.
        anchors=[]
        for k,seed in anchor_specs:
            anchors += [dict(u,source="anchor",spec=k) for u in collect64(env,m,pref_batch(k,torch.device("cuda")).cpu().numpy(),seed,f"a{k}")]
        adapt=[]
        for u in range(52,76):
            seed=73001+200000+u*223
            adapt += [dict(z,source="adaptive",update=u) for z in collect64(env,m,pref_batch(u+17,torch.device("cuda")).cpu().numpy(),seed,f"u{u}")]
        selected=[]
        for phase_key,phase in (("early","first"),("late","second")):
            au=[x for x in anchors if x["phase"]==phase]
            adu=[x for x in adapt if x["phase"]==phase]
            for i in sel[phase_key]["anchor"]:selected.append(au[i])
            for i in sel[phase_key]["adaptive"]:selected.append(adu[i])
        expanded=anchors+adapt

        fitA,metaA=fit_units(selected)
        fitB,metaB=fit_units(expanded)

        # Exact matched RV1-C endpoint suites, same seeds/preference set.
        eval_units=[]
        for suite in range(4):
            seed=840001+suite
            for lab in ORDER:
                eval_units += [dict(x,suite=suite,preference=lab) for x in collect64(env,m,PREFS[lab],seed,f"{lab}_s{suite}")]

        def evaluate_head(name,fit):
            by_phase={};all_ev=[];all_bias=[]
            for phase in ("first","second"):
                rows=[x for x in eval_units if x["phase"]==phase]
                per_head=[];per_pref={}
                for j,h in enumerate(HEADS):
                    vals=[];bias=[]
                    for r in rows:
                        P=pred(r["F"],fit)
                        vals.append(ev(r["Y"][:,j],P[:,j]))
                        bias.append(float(np.mean(P[:,j]-r["Y"][:,j])))
                    per_head.append({"head":h,"ev_mean":float(np.mean(vals)),
                                     "negative_fraction":float(np.mean(np.asarray(vals)<0)),
                                     "mean_abs_bias":float(np.mean(np.abs(bias)))})
                    all_ev += vals;all_bias += bias
                for lab in ORDER:
                    rr=[x for x in rows if x["preference"]==lab]
                    vals=[]
                    for r in rr:
                        P=pred(r["F"],fit)
                        vals += [ev(r["Y"][:,j],P[:,j]) for j in range(4)]
                    per_pref[lab]={"ev_mean":float(np.mean(vals)),
                                   "negative_fraction":float(np.mean(np.asarray(vals)<0))}
                by_phase[phase]={"by_head":per_head,"by_preference":per_pref,
                                 "aggregate_ev_mean":float(np.mean([x["ev_mean"] for x in per_head])),
                                 "aggregate_negative_fraction":float(np.mean([x["negative_fraction"] for x in per_head])),
                                 "aggregate_mean_abs_bias":float(np.mean([x["mean_abs_bias"] for x in per_head]))}
            return {"name":name,"phases":by_phase,
                    "overall_ev_mean":float(np.mean(all_ev)),
                    "overall_negative_fraction":float(np.mean(np.asarray(all_ev)<0)),
                    "overall_mean_abs_bias":float(np.mean(np.abs(all_bias)))}

        # Current checkpoint head as baseline, represented as numpy fit.
        Wc=m.critic_head.weight.detach().cpu().numpy().T
        bc=m.critic_head.bias.detach().cpu().numpy()
        baseline=evaluate_head("current_checkpoint_head",(Wc,bc))
        A=evaluate_head("A_selected12_current_policy",fitA)
        B=evaluate_head("B_expanded_current_policy",fitB)

        # Predeclared late gate.
        def gate(x):
            first=x["phases"]["first"];second=x["phases"]["second"]
            bh={z["head"]:z for z in second["by_head"]}
            return {
                "first_preserved": first["aggregate_ev_mean"]>0 and first["aggregate_negative_fraction"]<=.25,
                "second_tracking_ev_positive": bh["Tracking"]["ev_mean"]>0,
                "second_orientation_ev_positive": bh["Orientation"]["ev_mean"]>0,
                "second_negative_fraction": second["aggregate_negative_fraction"]<=.25,
                "second_bias_bounded": second["aggregate_mean_abs_bias"]<=.10,
            }
        gates={}
        for x in (baseline,A,B):
            g=gate(x);g["pass"]=all(g.values());gates[x["name"]]=g

        # Interpretation branch.
        if gates[A["name"]]["pass"]:
            verdict="A_PASS_FRESHNESS_LAG_SUFFICIENT"
        elif gates[B["name"]]["pass"]:
            verdict="B_PASS_SUPPORT_BREADTH_PRIMARY"
        else:
            verdict="A_B_FAIL_TRUE_SUPPORT_DISTRIBUTION_GAP_REMAINS"

        result={"schema":"rv1_actor_frozen_head_refresh_audit_v1","measurement_only":True,
                "actor_frozen":True,"critic_body_frozen":True,"optimizer_steps":0,
                "support":{"selected12":metaA,"expanded":metaB,"selected_indices":sel},
                "baseline":baseline,"A_selected12":A,"B_expanded":B,
                "gates":gates,"verdict":verdict}
        out=a.output_dir/"head_refresh_report.json";out.write_text(json.dumps(result,indent=2)+"\n")
        (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({
            "status":"FROZEN_BY_HASH","report_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
            "checkpoint_sha256":hashlib.sha256(CKPT.read_bytes()).hexdigest()},indent=2)+"\n")
        print(json.dumps({"verdict":verdict,"gates":gates,
                          "second_phase":{"baseline":baseline["phases"]["second"],
                                          "A":A["phases"]["second"],"B":B["phases"]["second"]}},indent=2))
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
