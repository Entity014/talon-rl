#!/usr/bin/env python3
from pathlib import Path
import json,sys,torch,numpy as np
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
U50=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt"
REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
OUT=ROOT/"runs/authority_isolated_classB_invariant_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
CASES={"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25,.25,.25,.25]}
SEED=840004;NENV=8;H=20;WIN=(6,10)
LEG_IDX=[[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def contact(env):
    try:
        f=env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w
        if f.ndim==4:f=f[:,-1]
        c=(torch.linalg.vector_norm(f,dim=-1)>1.0).float()
        return c.mean(-1),c
    except:
        z=torch.zeros(NENV,device="cuda");return z,z[:,None]
def safe_cos(a,b):
    return (a*b).sum(-1)/(torch.linalg.vector_norm(a,dim=-1)*torch.linalg.vector_norm(b,dim=-1)+1e-8)
def features(env,obs,a,prev_contact):
    d=env.unwrapped.scene["robot"].data
    pg=obs[:,6:9];ang=d.root_ang_vel_b;lin=d.root_lin_vel_b;jv=d.joint_vel
    hip=a[:,:4];th=a[:,4:8];calf=a[:,8:12]
    leg_a=torch.stack([torch.linalg.vector_norm(a[:,ix],dim=1) for ix in LEG_IDX],dim=1)
    leg_v=torch.stack([torch.linalg.vector_norm(jv[:,ix],dim=1) for ix in LEG_IDX],dim=1)
    cf,cm=contact(env);dc=(cf-prev_contact).abs()
    hip_lr=(hip[:,0]+hip[:,2])-(hip[:,1]+hip[:,3])
    hip_fr=(hip[:,0]+hip[:,1])-(hip[:,2]+hip[:,3])
    hip_diag=(hip[:,0]+hip[:,3])-(hip[:,1]+hip[:,2])
    act_lr=(leg_a[:,0]+leg_a[:,2])-(leg_a[:,1]+leg_a[:,3])
    act_diag=(leg_a[:,0]+leg_a[:,3])-(leg_a[:,1]+leg_a[:,2])
    vel_lr=(leg_v[:,0]+leg_v[:,2])-(leg_v[:,1]+leg_v[:,3])
    vel_diag=(leg_v[:,0]+leg_v[:,3])-(leg_v[:,1]+leg_v[:,2])
    F={
      "height":d.root_pos_w[:,2],"pg_xy":torch.linalg.vector_norm(pg[:,:2],dim=1),"pg_x":pg[:,0],"pg_y":pg[:,1],
      "ang_xy":torch.linalg.vector_norm(ang[:,:2],dim=1),"ang_x":ang[:,0],"ang_y":ang[:,1],"ang_z":ang[:,2],
      "lin_xy":torch.linalg.vector_norm(lin[:,:2],dim=1),"lin_x":lin[:,0],"lin_y":lin[:,1],"lin_z":lin[:,2],
      "joint_vel_norm":torch.linalg.vector_norm(jv,dim=1),"action_norm":torch.linalg.vector_norm(a,dim=1),
      "contact_frac":cf,"contact_change":dc,
      "hip_lr":hip_lr,"hip_frontrear":hip_fr,"hip_diag":hip_diag,"hip_std":hip.std(dim=1),
      "leg_action_lr":act_lr,"leg_action_diag":act_diag,"leg_action_std":leg_a.std(dim=1),
      "leg_vel_lr":vel_lr,"leg_vel_diag":vel_diag,"leg_vel_std":leg_v.std(dim=1),
      "action_vel_cos":safe_cos(a,jv),"hip_vel_cos":safe_cos(hip,jv[:,:4]),
      "roll_hip_coupling":ang[:,0]*hip_lr,"pitch_hip_coupling":ang[:,1]*hip_fr,
      "gravity_hip_coupling":pg[:,1]*hip_lr+pg[:,0]*hip_fr,
      "ang_support_energy":ang[:,0]*hip_lr+ang[:,1]*hip_fr,
      "support_velocity_coupling":hip_lr*vel_lr+hip_diag*vel_diag}
    return {k:v.detach().cpu().numpy() for k,v in F.items()},cf
def rollout(env,base,donor,wv,rescue=False):
    w=torch.tensor(wv,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=SEED);obs=ot(obs).cuda()
    rows=[];pc=torch.zeros(NENV,device="cuda")
    for t in range(H):
        with torch.no_grad():
            a=base.act_inference_with_preference(obs,w)
            if rescue and WIN[0]<=t<=WIN[1]:
                ad=donor.act_inference_with_preference(obs,w);a=ad
        F,pc2=features(env,obs,a,pc);rows.append({"t":t,"features":{k:v.tolist() for k,v in F.items()}})
        nxt,_,_,_,_=env.step(a);obs=ot(nxt).cuda();pc=pc2
    return rows
def summarize(raw):
    prefs=list(CASES);names=list(next(iter(raw.values()))["repair"][0]["features"])
    rep={}
    for f in names:
        rec=[]
        for pref in prefs:
            R=raw[pref]["repair"];Q=raw[pref]["rescue"];U=raw[pref]["u50"]
            for t in range(4,14):
                rv=np.array(R[t]["features"][f]);qv=np.array(Q[t]["features"][f]);uv=np.array(U[t]["features"][f])
                ctrl=rv[1:];mu=ctrl.mean();sd=ctrl.std()+1e-4
                rec.append({"pref":pref,"t":t,"fail":float(rv[0]),"rescue":float(qv[0]),"u50":float(uv[0]),
                 "ctrl_mean":float(mu),"ctrl_std":float(sd),"zf":float((rv[0]-mu)/sd),"zr":float((qv[0]-mu)/sd),"zu":float((uv[0]-mu)/sd)})
        mid=[x for x in rec if 6<=x["t"]<=10];late=[x for x in rec if 9<=x["t"]<=13]
        signs=[]
        for pref in prefs:
            q=[x["fail"]-x["ctrl_mean"] for x in mid if x["pref"]==pref]
            signs.append(np.sign(np.mean(q)))
        rep[f]={"records":rec,
          "mid_fail_abs_z":float(np.mean([abs(x["zf"]) for x in mid])),
          "mid_rescue_abs_z":float(np.mean([abs(x["zr"]) for x in mid])),
          "mid_u50_abs_z":float(np.mean([abs(x["zu"]) for x in mid])),
          "mid_margin":float(np.mean([abs(x["zf"])-abs(x["zr"]) for x in mid])),
          "late_margin":float(np.mean([abs(x["zf"])-abs(x["zr"]) for x in late])),
          "direction_consistency":bool(abs(sum(signs))==3),
          "direction_signs":[int(x) for x in signs]}
    return rep
def main():
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
        u50=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u50.load_state_dict(torch.load(U50,map_location="cuda",weights_only=False)["model"]);u50.eval()
        rep=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();rep.load_state_dict(torch.load(REP,map_location="cuda",weights_only=False)["model"]);rep.eval()
        raw={}
        for pref,wv in CASES.items():
            raw[pref]={"repair":rollout(env,rep,u50,wv,False),"rescue":rollout(env,rep,u50,wv,True),"u50":rollout(env,u50,u50,wv,False)}
            print("DONE",pref,flush=True)
        score=summarize(raw)
        rank=sorted(score,key=lambda k:score[k]["mid_margin"],reverse=True)
        out={"schema":"classB_invariant_audit_v1","rescue":"whole-u50 donor t6-10","raw":raw,"features":score,"rank_mid_margin":rank}
        (OUT/"classB_invariant_audit.json").write_text(json.dumps(out,indent=2)+"\n")
        for k in rank[:15]:
            q=score[k];print(k,round(q["mid_margin"],3),round(q["mid_fail_abs_z"],3),round(q["mid_rescue_abs_z"],3),q["direction_signs"],flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
