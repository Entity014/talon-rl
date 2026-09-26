"""Consolidated experiment stages. Use the first CLI argument to select a former script stage."""
from __future__ import annotations
import argparse

def run_phase4_r0_capture_source_obs():
    """Run former phase4_r0_capture_source_obs.py stage."""
    from pathlib import Path
    import json,sys
    import torch
    ROOT=Path(__file__).resolve().parents[4]
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8
        cfg.observations.policy.enable_corruption=False
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        seeds=(840001,840002,840003,840004)
        all_obs=[];meta=[]
        for si,seed in enumerate(seeds):
            obs,_=env.reset(seed=seed)
            x=obs["policy"] if isinstance(obs,dict) else obs
            cmd=env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().tolist()
            xs=x.detach().cpu().tolist()
            for e,row in enumerate(xs):
                all_obs.append(row);meta.append({"suite":si,"seed":seed,"env":e,"command":cmd[e]})
        rep={"schema":"phase4_r0_source_obs_v1","n":len(all_obs),"obs":all_obs,"meta":meta}
        out=ROOT/"runs/phase4_controller_robustness";out.mkdir(parents=True,exist_ok=True)
        (out/"source_obs.json").write_text(json.dumps(rep,indent=2)+"\n")
        print("saved",len(all_obs),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

def run_phase4_r0_actor_sensitivity():
    """Run former phase4_r0_actor_sensitivity.py stage."""
    from pathlib import Path
    import json,sys
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import Phase1StandaloneActor
    
    SRC=ROOT/"runs/phase4_controller_robustness/source_obs.json"
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase4_controller_robustness";OUT.mkdir(parents=True,exist_ok=True)
    
    PREFS={"T":[.7,.1,.1,.1],"C":[.25,.25,.25,.25]}
    BLOCKS={
     "base_lin_vel":(0,3,.01),
     "base_ang_vel":(3,6,.01),
     "projected_gravity":(6,9,.005),
     "joint_pos":(12,24,.005),
     "joint_vel":(24,36,.05),
     "prev_action":(36,48,.01),
    }
    
    def main():
        src=json.load(open(SRC));obs=np.asarray(src["obs"],np.float32)
        bundle=torch.load(ART,map_location="cuda",weights_only=False);state=bundle.get("actor_state",bundle)
        model=Phase1StandaloneActor().cuda();model.load_state_dict(state);model.eval()
        rows=[]
        for label,w in PREFS.items():
          wt=torch.tensor([w],device="cuda",dtype=torch.float32)
          for i,x in enumerate(obs):
            J=np.zeros((12,48),np.float64)
            for b,(lo,hi,eps) in BLOCKS.items():
              for j in range(lo,hi):
                xp=x.copy();xm=x.copy();xp[j]+=eps;xm[j]-=eps
                with torch.inference_mode():
                  ap=model(torch.from_numpy(xp[None]).cuda(),wt).cpu().numpy()[0]
                  am=model(torch.from_numpy(xm[None]).cuda(),wt).cpu().numpy()[0]
                J[:,j]=(ap-am)/(2*eps)
            active=np.concatenate([np.arange(lo,hi) for lo,hi,_ in BLOCKS.values()])
            s=np.linalg.svd(J[:,active],compute_uv=False)
            blocks={}
            for b,(lo,hi,eps) in BLOCKS.items():
                B=J[:,lo:hi]
                blocks[b]={
                  "fro":float(np.linalg.norm(B,"fro")),
                  "spectral":float(np.linalg.svd(B,compute_uv=False)[0]),
                  "mean_col_norm":float(np.mean(np.linalg.norm(B,axis=0))),
                  "max_col_norm":float(np.max(np.linalg.norm(B,axis=0))),
                }
            rows.append({"label":label,"state":i,"suite":src["meta"][i]["suite"],
                         "spectral_norm":float(s[0]),"fro_norm":float(np.linalg.norm(J[:,active],"fro")),
                         "blocks":blocks})
            if (i+1)%8==0: print("DONE",label,i+1,flush=True)
        summary={}
        for label in PREFS:
          rr=[r for r in rows if r["label"]==label]
          summary[label]={
            "spectral_mean":float(np.mean([r["spectral_norm"] for r in rr])),
            "spectral_median":float(np.median([r["spectral_norm"] for r in rr])),
            "spectral_p90":float(np.quantile([r["spectral_norm"] for r in rr],.9)),
            "spectral_max":float(np.max([r["spectral_norm"] for r in rr])),
            "block_spectral_mean":{b:float(np.mean([r["blocks"][b]["spectral"] for r in rr])) for b in BLOCKS},
            "block_maxcol_mean":{b:float(np.mean([r["blocks"][b]["max_col_norm"] for r in rr])) for b in BLOCKS},
          }
        # paired preference sensitivity difference on same states
        paired=[]
        for i in range(len(obs)):
          t=next(r for r in rows if r["label"]=="T" and r["state"]==i)
          c=next(r for r in rows if r["label"]=="C" and r["state"]==i)
          paired.append(t["spectral_norm"]-c["spectral_norm"])
        rep={"schema":"phase4_r0_actor_sensitivity_v1","n_states":len(obs),"blocks":BLOCKS,
             "summary":summary,"paired_T_minus_C_spectral_mean":float(np.mean(paired)),
             "paired_T_gt_C_fraction":float(np.mean(np.asarray(paired)>0)),"rows":rows}
        (OUT/"r0_actor_sensitivity.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps({k:v for k,v in rep.items() if k!="rows"},indent=2),flush=True)
    if True:main()

STAGES = {
    "phase4_r0_capture_source_obs": run_phase4_r0_capture_source_obs,
    "phase4_r0_actor_sensitivity": run_phase4_r0_actor_sensitivity,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
