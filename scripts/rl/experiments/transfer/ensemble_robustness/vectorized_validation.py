#!/usr/bin/env python3
from __future__ import annotations

# scripts/ on sys.path so the absolute rl.experiments.* imports below
# resolve when this file is run directly, as these scripts always are.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pathlib import Path
import hashlib,json,sys,math
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
import rl.experiments.common.utilities.authority_isolated_h1_screen as h1

CKPT=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
MAN=ROOT/"artifacts/phase5_e0_plant_ensemble_manifest.json"
OUT=ROOT/"runs/phase5_e1_vectorized_nominal_probe"
OUT.mkdir(parents=True,exist_ok=True)

LANES=8;STEPS=64;SUITES=4;G=.99
ORDER=("T","A","O","S")
PREFS={k:np.array(v,np.float32) for k,v in {
"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],
"S":[.1,.1,.1,.7],"C":[.25,.25,.25,.25]}.items()}
IDX={"T":0,"A":1,"O":2,"S":3}
PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
ALPHAS=(0.,.25,.5,.75,1.)
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

def mono(vals):
    vals=np.asarray(vals,float);d=np.diff(vals);target=np.sign(vals[-1]-vals[0])
    return 0.0 if target==0 else float(np.mean(np.sign(d)==target))

def between(vals):
    vals=np.asarray(vals,float);lo=min(vals[0],vals[-1]);hi=max(vals[0],vals[-1])
    return float(np.mean((vals[1:-1]>=lo-1e-9)&(vals[1:-1]<=hi+1e-9)))
def build_bank(man):
    return [{"id":"nominal","group":"nominal","mass_delta_kg":0.0,"passive_blend":0.0,"contact_blend":0.0}]

def apply_matched_reset(env,robot,bank,seed):
    cur,_=env.reset(seed=seed);u=env.unwrapped;dev=u.device
    nplant=len(bank);N=nplant*LANES
    # Replicate first eight physical lanes across plant groups.
    rp=(robot.data.root_link_pos_w[:LANES]-u.scene.env_origins[:LANES]).detach()
    rq=robot.data.root_link_quat_w[:LANES].detach()
    rv=torch.cat([robot.data.root_link_lin_vel_w[:LANES],robot.data.root_link_ang_vel_w[:LANES]],dim=1).detach()
    q=robot.data.joint_pos[:LANES].detach();qd=robot.data.joint_vel[:LANES].detach()
    rp=rp.repeat(nplant,1)+u.scene.env_origins
    robot.write_root_pose_to_sim(torch.cat([rp,rq.repeat(nplant,1)],dim=1))
    robot.write_root_velocity_to_sim(rv.repeat(nplant,1))
    robot.write_joint_state_to_sim(q.repeat(nplant,1),qd.repeat(nplant,1))
    term=u.command_manager.get_term("base_velocity")
    cmd=u.command_manager.get_command("base_velocity")[:LANES].detach()
    term.vel_command_b[:]=cmd.repeat(nplant,1)
    term.is_standing_env[:]=False
    if hasattr(term,"is_heading_env"):term.is_heading_env[:]=False
    # Override all plant parameters deterministically after reset.
    ids=torch.arange(N,dtype=torch.int32,device="cpu")
    bodies=list(robot.data.body_names);trunk=bodies.index("trunk")
    joints=list(robot.data.joint_names)
    hips=[i for i,n in enumerate(joints) if "_hip_joint" in n]
    flex=[i for i,n in enumerate(joints) if "_thigh_joint" in n or "_calf_joint" in n]
    masses=robot.root_physx_view.get_masses();inertias=robot.root_physx_view.get_inertias()
    base_mass=float(robot.data.default_mass[0,trunk].cpu())
    damp=robot.root_physx_view.get_dof_dampings();arm=robot.root_physx_view.get_dof_armatures()
    mats=robot.root_physx_view.get_material_properties()
    for pi,p in enumerate(bank):
        sl=slice(pi*LANES,(pi+1)*LANES)
        nm=base_mass+p["mass_delta_kg"]
        masses[sl,trunk]=nm
        ratio=nm/base_mass
        inertias[sl,trunk]=robot.data.default_inertia[sl,trunk].cpu()*ratio
        damp[sl]=0.0;arm[sl]=0.0
        damp[sl,hips]=p["passive_blend"]*1.0
        damp[sl,flex]=p["passive_blend"]*2.0
        arm[sl,:]=p["passive_blend"]*0.01
        mats[sl,:,0]=0.8;mats[sl,:,1]=0.6+0.2*p["contact_blend"];mats[sl,:,2]=0.0
    robot.root_physx_view.set_masses(masses,ids)
    robot.root_physx_view.set_inertias(inertias,ids)
    robot.root_physx_view.set_dof_dampings(damp,indices=ids)
    robot.root_physx_view.set_dof_armatures(arm,indices=ids)
    robot.root_physx_view.set_material_properties(mats,ids)
    u.action_manager.reset();u.scene.write_data_to_sim();u.sim.forward()
    return obs_tensor(u.observation_manager.compute(update_history=True)).cuda()
def evaluate(env,m,mgr,robot,bank,w_np,seed,collect_authority=False):
    from talon_rl.rewards.objectives import normalized_objective_vector
    nplant=len(bank);N=nplant*LANES
    w=torch.tensor(w_np,device="cuda").repeat(N,1)
    cur=apply_matched_reset(env,robot,bank,seed)
    R=[];D=[];V=[];prev=torch.zeros((N,12),device="cuda")
    done_any=np.zeros(N,bool);nonfinite=np.zeros(N,bool);ctrlbad=np.zeros(N,bool)
    min_h=np.full(N,np.inf);max_t=np.zeros(N);sat_num=np.zeros(N);sat_den=np.zeros(N)
    phys={k:[] for k in ("tracking_error","ang_vel_xy","tilt_deg","action_rate")}
    auth=[[] for _ in bank]
    with torch.no_grad():
        for _ in range(STEPS):
            if collect_authority:
                cc=cur.detach().cpu()
                for pi in range(nplant):auth[pi].append(cc[pi*LANES].numpy())
            V.append(m.value_with_preference(cur,w).cpu().numpy())
            a=m.act_inference_with_preference(cur,w)
            aa=a.detach().cpu().numpy()
            sat_num+=(np.abs(aa)>=.95).sum(1);sat_den+=aa.shape[1]
            nonfinite|=(~np.isfinite(aa)).any(1);ctrlbad|=(np.abs(aa)>1.000001).any(1)
            nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            R.append(normalized_objective_vector(terms(raw,names),shape=(N,))*env.unwrapped.step_dt)
            dd=(te|tr).cpu().numpy();D.append(dd);done_any|=dd
            data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
            vx=(data.root_lin_vel_b[:,0]-cmd[:,0]).abs();wz=(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()
            tt=tilt_deg(data.root_quat_w)
            phys["tracking_error"].append((vx+wz).detach().cpu().numpy())
            phys["ang_vel_xy"].append(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).cpu().numpy())
            phys["tilt_deg"].append(tt.cpu().numpy())
            phys["action_rate"].append(torch.linalg.vector_norm(a-prev,dim=-1).cpu().numpy())
            h=data.root_link_pos_w[:,2].detach().cpu().numpy();ta=tt.cpu().numpy()
            min_h=np.minimum(min_h,h);max_t=np.maximum(max_t,ta)
            nonfinite|=(~np.isfinite(h))|(~np.isfinite(data.joint_pos.detach().cpu().numpy()).all(1))
            prev=a;cur=obs_tensor(nxt).cuda()
    R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V)
    pp={k:np.asarray(v) for k,v in phys.items()}
    H0=segret(R,D,0,32);H1=segret(R,D,32,64)
    out=[]
    for pi,p in enumerate(bank):
        sl=slice(pi*LANES,(pi+1)*LANES)
        pobj=R[:,sl].mean((0,1))
        pphys={k:float(v[:,sl].mean()) for k,v in pp.items()}
        cev=[];cbias=[]
        for j in range(4):
            cev += [ev(H0[:,sl,j],V[:32,sl,j]),ev(H1[:,sl,j],V[32:,sl,j])]
            cbias += [float(np.mean(V[:32,sl,j]-H0[:,sl,j])),float(np.mean(V[32:,sl,j]-H1[:,sl,j]))]
        out.append({
          "plant_id":p["id"],"group":p["group"],"normalized_objective_mean":pobj.tolist(),
          "physical":pphys,"survival":float(1-done_any[sl].mean()),
          "min_height":float(min_h[sl].min()),"max_tilt_deg":float(max_t[sl].max()),
          "action_saturation_fraction":float(sat_num[sl].sum()/max(1,sat_den[sl].sum())),
          "nonfinite_fraction":float(nonfinite[sl].mean()),"ctrl_violation_fraction":float(ctrlbad[sl].mean()),
          "critic":{"h32_ev":cev,"h32_bias":cbias}})
    aprobe=[np.asarray(x,np.float32) for x in auth] if collect_authority else None
    return out,aprobe
def interp(a,b,alpha):return (1-alpha)*PREFS[a]+alpha*PREFS[b]

def stats(v):
    x=np.asarray(v,float)
    return {"mean":float(np.mean(x)),"median":float(np.median(x)),"p10":float(np.quantile(x,.1)),
            "worst_min":float(np.min(x)),"worst_max":float(np.max(x))}

def main():
    man=json.load(open(MAN));bank=build_bank(man);nplant=len(bank);N=nplant*LANES
    (OUT/"plant_bank.json").write_text(json.dumps(bank,indent=2)+"\n")
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]]
    app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=N;cfg.seed=0
        cfg.events.add_base_mass=None;cfg.observations.policy.enable_corruption=False
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        o,_=env.reset(seed=0);o=obs_tensor(o).cuda();mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        m=AuthorityIsolatedWideCritic(o.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda()
        m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
        rows=[];auth_obs=[[] for _ in bank]
        for suite in range(SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                rr,ap=evaluate(env,m,mgr,robot,bank,PREFS[lab],seed,collect_authority=(lab=="C"))
                for x in rr:x.update({"suite":suite,"kind":"endpoint","label":lab});rows.append(x)
                if ap is not None:
                    for pi,x in enumerate(ap):auth_obs[pi].append(x)
            print("ENDPOINT_SUITE",suite,flush=True)
        continuum=[]
        for pi,(a,b) in enumerate(PATHS):
            for suite in range(SUITES):
                seed=850001+pi*1000+suite
                for alpha in ALPHAS:
                    rr,_=evaluate(env,m,mgr,robot,bank,interp(a,b,alpha),seed)
                    for x in rr:x.update({"path":f"{a}-{b}","a":a,"b":b,"suite":suite,"alpha":alpha});continuum.append(x)
                print("CONT",a,b,suite,flush=True)

        # Same-state preference authority on center-state distributions.
        authority={}
        for pi,p in enumerate(bank):
            probe=torch.tensor(np.concatenate(auth_obs[pi],axis=0),device="cuda")
            ss=h1.sensitivity(m,probe)
            authority[p["id"]]=ss
            print("AUTH",p["id"],flush=True)
        nom=authority["nominal"]
        for pid,a in authority.items():
            a["pairwise_retention_vs_nominal"]=a["pairwise_action_distance"]["mean"]/(nom["pairwise_action_distance"]["mean"]+1e-12)
            a["tangent_retention_vs_nominal"]=a["tangent_jacobian_fro_mean"]/(nom["tangent_jacobian_fro_mean"]+1e-12)

        plants={}
        for p in bank:
            pid=p["id"];endpoint={};center_between=[];cont_stats=[]
            for lab in ORDER:
                j=IDX[lab];pk=PHYS[lab];oo=[];po=[];surv=[];do=[];dp=[]
                for suite in range(SUITES):
                    r=next(x for x in rows if x["plant_id"]==pid and x["suite"]==suite and x["label"]==lab)
                    c=next(x for x in rows if x["plant_id"]==pid and x["suite"]==suite and x["label"]=="C")
                    d0=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                    d1=r["physical"][pk]-c["physical"][pk]
                    oo.append(d0>0);po.append(d1<0);surv.append(r["survival"]);do.append(d0);dp.append(d1)
                endpoint[lab]={"objective_correct_fraction":float(np.mean(oo)),"physical_correct_fraction":float(np.mean(po)),
                  "mean_objective_delta_vs_center":float(np.mean(do)),"mean_physical_delta_vs_center":float(np.mean(dp)),
                  "min_survival":float(np.min(surv))}
            for suite in range(SUITES):
                hs=[next(x for x in rows if x["plant_id"]==pid and x["suite"]==suite and x["label"]==lab) for lab in ORDER]
                cc=next(x for x in rows if x["plant_id"]==pid and x["suite"]==suite and x["label"]=="C")
                for j in range(4):
                    vals=[x["normalized_objective_mean"][j] for x in hs];cv=cc["normalized_objective_mean"][j]
                    center_between.append(min(vals)-1e-9<=cv<=max(vals)+1e-9)
            for a,b in PATHS:
                for suite in range(SUITES):
                    rr=sorted([x for x in continuum if x["plant_id"]==pid and x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                    for lab in (a,b):
                        j=IDX[lab];pk=PHYS[lab]
                        cont_stats.append({"path":f"{a}-{b}","suite":suite,"axis":lab,
                          "objective_monotonic":mono([x["normalized_objective_mean"][j] for x in rr]),
                          "objective_between":between([x["normalized_objective_mean"][j] for x in rr]),
                          "physical_monotonic":mono([x["physical"][pk] for x in rr]),
                          "physical_between":between([x["physical"][pk] for x in rr])})
            mono_all=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont_stats]))
            bet_all=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont_stats]))
            er=[x for x in rows if x["plant_id"]==pid];cr=[x for x in continuum if x["plant_id"]==pid]
            cev=[];cbias=[]
            for x in er:
                cev+=x["critic"]["h32_ev"];cbias+=x["critic"]["h32_bias"]
            critic={"h32_ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),
                    "mean_abs_bias":float(np.mean(np.abs(cbias)))}
            eng={"min_survival":float(min(x["survival"] for x in er+cr)),
                 "min_height":float(min(x["min_height"] for x in er+cr)),
                 "max_tilt_deg":float(max(x["max_tilt_deg"] for x in er+cr)),
                 "max_action_saturation_fraction":float(max(x["action_saturation_fraction"] for x in er+cr)),
                 "max_nonfinite_fraction":float(max(x["nonfinite_fraction"] for x in er+cr)),
                 "max_ctrl_violation_fraction":float(max(x["ctrl_violation_fraction"] for x in er+cr))}
            endpoint_pass={lab:bool(endpoint[lab]["objective_correct_fraction"]>=.75 and
                                    endpoint[lab]["physical_correct_fraction"]>=.75 and
                                    endpoint[lab]["min_survival"]>=.95) for lab in ORDER}
            plants[pid]={"plant":p,"endpoint":endpoint,"endpoint_pass":endpoint_pass,
              "center_compromise":float(np.mean(center_between)),"continuum_monotonicity":mono_all,
              "continuum_endpoint_between":bet_all,"critic":critic,"engineering":eng,"authority":authority[pid]}

        # Aggregate by bank group.
        aggregates={}
        for grp in ("nominal",):
            ids=[p["id"] for p in bank if p["group"]==grp]
            ag={"n":len(ids),"objective_pass_fraction":{lab:float(np.mean([plants[i]["endpoint_pass"][lab] for i in ids])) for lab in ORDER},
                "survival_gate_fraction":float(np.mean([plants[i]["engineering"]["min_survival"]>=.95 for i in ids])),
                "critic_valid_fraction":float(np.mean([plants[i]["critic"]["h32_ev_mean"]>0 and plants[i]["critic"]["negative_fraction"]<=.25 for i in ids])),
                "authority_pairwise_retention_fraction":float(np.mean([plants[i]["authority"]["pairwise_retention_vs_nominal"]>=.90 for i in ids])),
                "authority_tangent_retention_fraction":float(np.mean([plants[i]["authority"]["tangent_retention_vs_nominal"]>=.90 for i in ids])),
                "center_pass_fraction":float(np.mean([plants[i]["center_compromise"]>=.75 for i in ids])),
                "continuum_monotonic_pass_fraction":float(np.mean([plants[i]["continuum_monotonicity"]>=.65 for i in ids])),
                "continuum_between_pass_fraction":float(np.mean([plants[i]["continuum_endpoint_between"]>=.65 for i in ids]))}
            for key in ("min_survival","min_height","max_tilt_deg","max_action_saturation_fraction"):
                ag[key]=stats([plants[i]["engineering"][key] for i in ids])
            aggregates[grp]=ag
        # Descriptive plant-sensitivity correlations. No selection/tuning use.
        sens={}
        ids=[p["id"] for p in bank if p["group"]!="nominal"]
        lat={k:np.array([plants[i]["plant"][k] for i in ids],float) for k in ("mass_delta_kg","passive_blend","contact_blend")}
        metrics={"survival":np.array([plants[i]["engineering"]["min_survival"] for i in ids]),
                 "continuum_monotonicity":np.array([plants[i]["continuum_monotonicity"] for i in ids]),
                 "center_compromise":np.array([plants[i]["center_compromise"] for i in ids]),
                 "pairwise_authority":np.array([plants[i]["authority"]["pairwise_retention_vs_nominal"] for i in ids])}
        for lab in ORDER:
            metrics[f"{lab}_objective_margin"]=np.array([plants[i]["endpoint"][lab]["mean_objective_delta_vs_center"] for i in ids])
            metrics[f"{lab}_physical_margin"]=np.array([plants[i]["endpoint"][lab]["mean_physical_delta_vs_center"] for i in ids])
        try:
            from scipy.stats import spearmanr
            for lk,x in lat.items():
                sens[lk]={mk:{"rho":float(spearmanr(x,y).statistic),"p":float(spearmanr(x,y).pvalue)} for mk,y in metrics.items()}
        except Exception as e:sens={"error":str(e)}

        report={"schema":"phase5_e1_validity_envelope_v1","measurement_only":True,"training_updates":0,
          "checkpoint":str(CKPT.relative_to(ROOT)),"manifest_sha256":hashlib.sha256(MAN.read_bytes()).hexdigest(),
          "plant_bank":bank,"plants":plants,"aggregates":aggregates,"plant_sensitivity":sens}
        (OUT/"e1_report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"aggregates":aggregates,"plant_sensitivity":sens},indent=2),flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
