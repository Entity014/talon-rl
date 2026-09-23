#!/usr/bin/env python3
"""T4: generate four fixed-preference specialists on frozen normalized 4D objectives and validate endpoints."""
from __future__ import annotations
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ORDER=("T","A","O","S")
PREFS={
 "T":np.array([.7,.1,.1,.1],np.float32),
 "A":np.array([.1,.7,.1,.1],np.float32),
 "O":np.array([.1,.1,.7,.1],np.float32),
 "S":np.array([.1,.1,.1,.7],np.float32),
}
OBJ_NAMES=("velocity_tracking","angular_stability","orientation_stability","control_smoothness")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def tilt_deg(q):
    _,x,y,_=[q[:,i] for i in range(4)]
    return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def weighted_terms(raw,names):
    return {n:raw[:,i] for i,n in enumerate(names)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--updates",type=int,default=300);ap.add_argument("--horizon",type=int,default=2)
    ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--eval-steps",type=int,default=64)
    ap.add_argument("--reset-suites",type=int,default=4);ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
    ap.add_argument("--critic-head-init",choices=("scalar","zero"),default="scalar")
    ap.add_argument("--actor-lr",type=float,default=1e-3)
    ap.add_argument("--critic-lr",type=float,default=1e-3)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
    def mark(event,**extra):
        with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
    mark("RUN_STARTED",protocol="T4",updates=args.updates,prefs={k:v.tolist() for k,v in PREFS.items()})
    app=env=None
    try:
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
        from talon_rl.t3b_objectives import normalized_objective_vector,raw_objective_vector,NORMALIZATION_DIVISORS
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);mark("ENV_CREATED")
        obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
        manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        models={};records={};checkpoints={}
        for idx,label in enumerate(ORDER):
            w_np=PREFS[label];torch.manual_seed(31000+idx);np.random.seed(31000+idx)
            m=T4SharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init=args.critic_head_init)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            critic_params=[p for n,p in m.named_parameters() if n.startswith("critic_")]
            opt=torch.optim.Adam([{"params":actor_params,"lr":args.actor_lr},{"params":critic_params,"lr":args.critic_lr}])
            w=torch.as_tensor(np.repeat(w_np[None,:],args.num_envs,axis=0),device="cuda")
            cur,_=env.reset(seed=310001+idx*1000);cur=obs_tensor(cur).cuda();rec=[]
            mark("SPECIALIST_START",specialist=label,w=w_np.tolist())
            for update in range(1,args.updates+1):
                ob=[];ac=[];pre=[];old=[];rw=[];val=[];dn=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
                    # T5 repair: the exact policy action is already in [-1,1] and is applied unchanged.
                    nxt,_,term,trunc,_=env.step(a)
                    raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                    vec=normalized_objective_vector(weighted_terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);ac.append(a);pre.append(u);old.append(lp);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda())
                    cur=obs_tensor(nxt).cuda()
                with torch.no_grad():nv=m.value_with_preference(cur,w)
                rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt)
                fo=torch.cat(ob);fa=torch.cat(ac);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
                ratio_maxerr=float((ratio-1).abs().max().detach())
                if ratio_maxerr>1e-4 or not torch.isfinite(ratio).all():
                    raise RuntimeError(f"PPO pre-update ratio invariant failed: max|r-1|={ratio_maxerr}")
                al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                cl=vector_value_loss(m.value_with_preference(fo,fw),ret.reshape(-1,4).detach())
                loss=al+cl;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if update in (1,5,10,25,50,100,200,300):
                    rec.append({"update":update,"loss":float(loss.detach()),"normalized_reward_stepdt_mean":rt.mean((0,1)).detach().cpu().tolist()})
            cp=args.output.parent/f"specialist_{label}_terminal.pt"
            torch.save({"schema":"t4_specialist_terminal_v1","specialist":label,"objective_order":OBJ_NAMES,"w":w_np.tolist(),"update":args.updates,"normalization_divisors":NORMALIZATION_DIVISORS.tolist(),"model":m.state_dict()},cp)
            models[label]=m.eval();records[label]=rec;checkpoints[label]={"path":str(cp),"sha256":sha(cp)};mark("SPECIALIST_DONE",specialist=label)
        # Paired deterministic endpoint evaluation. Each policy acts using its own fixed training preference.
        rows=[]
        for suite in range(args.reset_suites):
            reset_seed=320001+suite
            for label in ORDER:
                m=models[label];w_np=PREFS[label];w=torch.as_tensor(np.repeat(w_np[None,:],args.num_envs,axis=0),device="cuda")
                cur,_=env.reset(seed=reset_seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((args.num_envs,ad),device="cuda")
                raw_obj=[];norm_obj=[];metrics=[];done_any=np.zeros(args.num_envs,bool)
                with torch.no_grad():
                    for _ in range(args.eval_steps):
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                        rt=raw_objective_vector(weighted_terms(raw,names),shape=(args.num_envs,))
                        nt=normalized_objective_vector(weighted_terms(raw,names),shape=(args.num_envs,))
                        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                        metrics.append({
                          "vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),
                          "wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),
                          "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                          "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                          "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
                          "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean()),
                          "torque_norm":float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()),
                        })
                        raw_obj.append(rt.mean(0));norm_obj.append(nt.mean(0));done_any|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
                norm=np.asarray(norm_obj).mean(0);rawm=np.asarray(raw_obj).mean(0);mm={k:float(np.mean([x[k] for x in metrics])) for k in metrics[0]}
                scalarized={ev:float(norm@PREFS[ev]) for ev in ORDER}
                rows.append({"suite":suite,"policy":label,"train_w":w_np.tolist(),"raw_objective_mean":rawm.tolist(),"normalized_objective_mean":norm.tolist(),
                             "scalarized_by_eval_preference":scalarized,"physical":mm,"survival":float(1-done_any.mean())})
        # Cross-eval matrix and endpoint gates.
        matrix={ev:{pol:float(np.mean([r["scalarized_by_eval_preference"][ev] for r in rows if r["policy"]==pol])) for pol in ORDER} for ev in ORDER}
        objective_winner_fraction={};physical_winner_fraction={};diag_scalar_winner_fraction={}
        phys_key={"T":None,"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
        for i,label in enumerate(ORDER):
            ow=[];pw=[];sw=[]
            for suite in range(args.reset_suites):
                rr=[r for r in rows if r["suite"]==suite]
                ow.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==label)
                if label=="T":
                    # tracking physical score: lower normalized sum of vx/wz absolute errors (equal weight in validation only)
                    pw.append(min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==label)
                else:
                    k=phys_key[label];pw.append(min(rr,key=lambda r:r["physical"][k])["policy"]==label)
                sw.append(max(rr,key=lambda r:r["scalarized_by_eval_preference"][label])["policy"]==label)
            objective_winner_fraction[label]=float(np.mean(ow));physical_winner_fraction[label]=float(np.mean(pw));diag_scalar_winner_fraction[label]=float(np.mean(sw))
        # Safety/constraint monitors.
        min_survival=min(r["survival"] for r in rows)
        vertical_by_policy={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in rows if r["policy"]==p])) for p in ORDER}
        track_vertical=vertical_by_policy["T"]
        vertical_ratio={p:vertical_by_policy[p]/(track_vertical+1e-12) for p in ORDER}
        endpoint_pass={p:bool(objective_winner_fraction[p]>=.75 and physical_winner_fraction[p]>=.75 and diag_scalar_winner_fraction[p]>=.75) for p in ORDER}
        safety_pass=bool(min_survival>=.95 and max(vertical_ratio.values())<=2.0)
        report={
          "schema":"t4_specialist_generation_validation_v1","status":"TRAINING_AND_VALIDATION_COMPLETE",
          "updates":args.updates,"critic_head_init":args.critic_head_init,"actor_lr":args.actor_lr,"critic_lr":args.critic_lr,"objective_order":OBJ_NAMES,"fixed_preferences":{k:v.tolist() for k,v in PREFS.items()},
          "normalization_divisors":NORMALIZATION_DIVISORS.tolist(),"checkpoints":checkpoints,"training_records":records,
          "validation_protocol":{"paired_reset":True,"reset_seed_base":320001,"suites":args.reset_suites,"eval_steps":args.eval_steps,
                                 "deterministic_actor":True,"clip_actions":1.0,"policy_acts_with_training_preference":True},
          "rows":rows,"cross_evaluation_matrix":matrix,
          "gates":{"objective_winner_fraction":objective_winner_fraction,"physical_winner_fraction":physical_winner_fraction,
                   "diagonal_scalarized_winner_fraction":diag_scalar_winner_fraction,"endpoint_pass":endpoint_pass,
                   "min_survival":min_survival,"vertical_ratio_to_tracking_policy":vertical_ratio,"safety_pass":safety_pass,
                   "overall_semantic_endpoint_pass":bool(all(endpoint_pass.values()) and safety_pass)},
          "notes":{"effort":"monitor only; excluded from 4D preference vector","vertical_stability":"constraint monitor only; excluded from 4D preference vector"}
        }
        args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",overall_pass=report["gates"]["overall_semantic_endpoint_pass"])
        print(json.dumps({"gates":report["gates"],"matrix":matrix},indent=2))
    except BaseException as exc:
        args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()
if __name__=="__main__":main()
