#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
CKPT=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
OUT_DEFAULT=ROOT/"runs/authority_isolated_h2a_u30_semantic_validity-2026-09-25"
NENV=8;STEPS=64;SUITES=4;G=.99
ORDER=("T","A","O","S")
PREFS={
 "T":np.array([.7,.1,.1,.1],np.float32),
 "A":np.array([.1,.7,.1,.1],np.float32),
 "O":np.array([.1,.1,.7,.1],np.float32),
 "S":np.array([.1,.1,.1,.7],np.float32),
 "C":np.array([.25,.25,.25,.25],np.float32)}
IDX={"T":0,"A":1,"O":2,"S":3}
PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
ALPHAS=(0.0,.25,.5,.75,1.0)
PATHS=(("T","A"),("T","O"),("T","S"),("A","O"),("A","S"),("O","S"))

def obs_tensor(x):
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

def evaluate(env,m,mgr,robot,w_np,seed):
    from talon_rl.rewards.objectives import normalized_objective_vector
    w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
    cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
    R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
    done_any=np.zeros(NENV,bool)
    with torch.no_grad():
        for _ in range(STEPS):
            V.append(m.value_with_preference(cur,w).cpu().numpy())
            a=m.act_inference_with_preference(cur,w)
            nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
            dd=(te|tr).cpu().numpy();D.append(dd);done_any|=dd
            data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
            wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
            phys.append({"vx_error":vx,"wz_error":wz,"tracking_error":vx+wz,
                         "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                         "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                         "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
                         "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
            prev=a;cur=obs_tensor(nxt).cuda()
    R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V)
    H0=segret(R,D,0,32);H1=segret(R,D,32,64)
    pmean={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
    obj=R.mean((0,1))
    return {"normalized_objective_mean":obj.tolist(),"physical":pmean,
            "survival":float(1-done_any.mean()),
            "critic":{"first_h32_ev":[ev(H0[:,:,j],V[:32,:,j]) for j in range(4)],
                      "second_h32_ev":[ev(H1[:,:,j],V[32:,:,j]) for j in range(4)],
                      "first_h32_bias":[float(np.mean(V[:32,:,j]-H0[:,:,j])) for j in range(4)],
                      "second_h32_bias":[float(np.mean(V[32:,:,j]-H1[:,:,j])) for j in range(4)]}}

def interp(a,b,alpha):return (1-alpha)*PREFS[a]+alpha*PREFS[b]

def monotonic_fraction(vals):
    vals=np.asarray(vals,float);d=np.diff(vals);target=np.sign(vals[-1]-vals[0])
    if target==0:return 0.0
    return float(np.mean(np.sign(d)==target))

def between_fraction(vals):
    vals=np.asarray(vals,float);lo=min(vals[0],vals[-1]);hi=max(vals[0],vals[-1])
    return float(np.mean((vals[1:-1]>=lo-1e-9)&(vals[1:-1]<=hi+1e-9)))
def sha(p):
    h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()

def main():
    class Args: pass
    args=Args();args.checkpoint=CKPT;args.output_dir=OUT_DEFAULT
    args.output_dir.mkdir(parents=True,exist_ok=True)
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]]
    app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
        mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        m=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda()
        m.load_state_dict(torch.load(args.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
        rows=[]
        # Matched endpoints + center: exact same reset seed across every preference in a suite.
        for suite in range(SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=evaluate(env,m,mgr,robot,PREFS[lab],seed)
                q.update({"suite":suite,"kind":"endpoint","label":lab,"w":PREFS[lab].tolist()});rows.append(q)
        # Continuum paths, again matched by suite and path.
        continuum=[]
        for pi,(a,b) in enumerate(PATHS):
            for suite in range(SUITES):
                seed=850001+pi*1000+suite
                vals=[]
                for alpha in ALPHAS:
                    w=interp(a,b,alpha)
                    q=evaluate(env,m,mgr,robot,w,seed)
                    q.update({"path":f"{a}-{b}","a":a,"b":b,"suite":suite,"alpha":alpha,"w":w.tolist()})
                    vals.append(q);continuum.append(q)
        # Endpoint directional correctness vs matched center.
        endpoint={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj_ok=[];phys_ok=[];surv=[]
            del_obj=[];del_phys=[]
            for suite in range(SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                do=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                dp=r["physical"][pk]-c["physical"][pk]
                # Objectives are higher-is-better; physical proxies are lower-is-better.
                obj_ok.append(do>0);phys_ok.append(dp<0);surv.append(r["survival"])
                del_obj.append(do);del_phys.append(dp)
            endpoint[lab]={"objective_correct_fraction":float(np.mean(obj_ok)),
                           "physical_correct_fraction":float(np.mean(phys_ok)),
                           "mean_objective_delta_vs_center":float(np.mean(del_obj)),
                           "mean_physical_delta_vs_center":float(np.mean(del_phys)),
                           "min_survival":float(np.min(surv))}
        cont_stats=[]
        for a,b in PATHS:
            for suite in range(SUITES):
                rr=sorted([x for x in continuum if x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                for lab in (a,b):
                    j=IDX[lab];pk=PHYS[lab]
                    ov=[x["normalized_objective_mean"][j] for x in rr]
                    pv=[x["physical"][pk] for x in rr]
                    cont_stats.append({"path":f"{a}-{b}","suite":suite,"axis":lab,
                        "objective_monotonic":monotonic_fraction(ov),
                        "objective_between":between_fraction(ov),
                        # lower physical is better, but monotonic_fraction only tests coherence between endpoints.
                        "physical_monotonic":monotonic_fraction(pv),
                        "physical_between":between_fraction(pv)})
        mono=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont_stats]))
        between=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont_stats]))
        # Cross-axis trade-off/collateral summary.
        centers=[x for x in rows if x["label"]=="C"]
        center_track=float(np.mean([x["physical"]["tracking_error"] for x in centers]))
        # Center compromise: for each objective axis and suite, center should lie within the
        # envelope spanned by the four heavy policies rather than becoming an out-of-family extreme.
        center_between=[]
        center_rank=[]
        for suite in range(SUITES):
            hs=[next(x for x in rows if x["suite"]==suite and x["label"]==lab) for lab in ORDER]
            cc=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
            for j in range(4):
                vals=[x["normalized_objective_mean"][j] for x in hs];cv=cc["normalized_objective_mean"][j]
                center_between.append(min(vals)-1e-9 <= cv <= max(vals)+1e-9)
                center_rank.append(float(sum(v<=cv for v in vals))/len(vals))
        center_compromise={"between_heavy_envelope_fraction":float(np.mean(center_between)),
                           "mean_fraction_heavies_below_center":float(np.mean(center_rank))}
        max_track_ratio=0.0
        for lab in ORDER:
            rr=[x for x in rows if x["label"]==lab]
            tr=float(np.mean([x["physical"]["tracking_error"] for x in rr]))
            max_track_ratio=max(max_track_ratio,tr/(center_track+1e-12))
        all_surv=[x["survival"] for x in rows]+[x["survival"] for x in continuum]
        # Fresh critic aggregate over all endpoint/center matched evaluations.
        cev=[];cbias=[]
        for x in rows:
            cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"]
            cbias+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
        critic={"h32_ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),
                "mean_abs_bias":float(np.mean(np.abs(cbias)))}
        endpoint_pass={lab:(endpoint[lab]["objective_correct_fraction"]>=.75 and
                            endpoint[lab]["physical_correct_fraction"]>=.75 and
                            endpoint[lab]["min_survival"]>=.95) for lab in ORDER}
        criteria={
            "all_endpoint_axes_correct":bool(all(endpoint_pass.values())),
            "continuum_monotonicity":mono>=.65,
            "continuum_endpoint_between":between>=.65,
            "survival_preserved":float(np.min(all_surv))>=.95,
            "tracking_collateral_bounded":max_track_ratio<=2.0,
            "center_compromise":center_compromise["between_heavy_envelope_fraction"]>=.75,
            "fresh_critic_not_collapsed":critic["h32_ev_mean"]>0.0 and critic["negative_fraction"]<=.25,
        }
        passed=all(criteria.values())
        report={"schema":"authority_isolated_h2a_u30_semantic_validity_v1","status":"H2a SEMANTIC VALIDITY PASS" if passed else "H2a SEMANTIC VALIDITY FAIL",
            "measurement_only":True,"training_updates":0,"checkpoint":str(args.checkpoint.relative_to(ROOT)),
            "protocol":{"matched_reset":True,"suites":SUITES,"steps":STEPS,"alphas":list(ALPHAS),
                        "paths":[f"{a}-{b}" for a,b in PATHS],"thresholds":{"endpoint_fraction":.75,"continuum_fraction":.65,
                        "min_survival":.95,"max_tracking_ratio_to_center":2.0}},
            "endpoint":endpoint,"endpoint_pass":endpoint_pass,"continuum_summary":{"monotonicity_fraction":mono,
                        "endpoint_between_fraction":between,"details":cont_stats},
            "center_compromise":center_compromise,
            "collateral":{"center_tracking_error":center_track,"max_tracking_ratio_to_center":max_track_ratio},
            "critic":critic,"min_survival_all":float(np.min(all_surv)),"criteria":criteria,
            "decision":{"semantic_pass":bool(passed),"architecture":"authority_isolated_wide","hypothesis":"H2a_validity_at_validated_u30"},
            "endpoint_rows":rows,"continuum_rows":continuum}
        out=args.output_dir/"semantic_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:report[k] for k in ("status","endpoint","endpoint_pass","continuum_summary","collateral","critic","min_survival_all","criteria","decision") if k in report and k!="continuum_summary"} | {"continuum_summary":{"monotonicity_fraction":mono,"endpoint_between_fraction":between}},indent=2))
        prov={"status":"FROZEN_BY_HASH","gate":"H2A-U30-SEMANTIC-VALIDITY","verdict":report["status"],"measurement_only":True,
              "architecture":"authority_isolated_wide",
              "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},
                           str(args.checkpoint.relative_to(ROOT)):{"sha256":sha(args.checkpoint)},
                           str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())}}}
        (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
