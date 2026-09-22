#!/usr/bin/env python3
"""T3-A5 targeted effort controllability screen.

Single-variable exploratory screen:
  reward vector = [velocity_tracking, effort, 0]
  fixed preference encodes relative emphasis beta as [1, beta, 0]/(1+beta)
Everything else follows the frozen V1C PPO path.
"""
from __future__ import annotations
import argparse,json,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
BETAS=(0.0,0.5,1.0,2.0)
TRACK_TERMS=("track_lin_vel_xy_exp","track_ang_vel_z_exp")
EFFORT_TERM="dof_torques_l2"
ACCEPT={"effort_improvement_fraction":0.03,"tracking_degradation_fraction":0.025,"required_suites":3}
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pref(beta,n,device):
    if beta==0:return torch.tensor([1.,0.,0.],device=device).repeat(n,1)
    v=torch.tensor([1.,beta,0.],device=device);v=v/v.sum();return v.repeat(n,1)
def vec_from_raw(raw,names):
    idx={n:i for i,n in enumerate(names)}
    progress=raw[:,idx[TRACK_TERMS[0]]]+raw[:,idx[TRACK_TERMS[1]]]
    effort=raw[:,idx[EFFORT_TERM]]
    zero=np.zeros_like(progress)
    return np.stack([progress,effort,zero],axis=1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--updates",type=int,default=100)
    ap.add_argument("--horizon",type=int,default=2)
    ap.add_argument("--num-envs",type=int,default=8)
    ap.add_argument("--eval-steps",type=int,default=64)
    ap.add_argument("--suites",type=int,default=4)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
    def mark(event,**kw):
        with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**kw},sort_keys=True)+"\n")
    mark("RUN_STARTED",protocol="T3-A5-EFFORT",betas=BETAS,acceptance=ACCEPT,updates=args.updates)
    app=env=None
    try:
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.v1c_actor_critic import V1CSharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo,vector_value_loss
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);mark("ENV_CREATED")
        obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
        manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
        models={};records={};checkpoints={}
        for bi,beta in enumerate(BETAS):
            torch.manual_seed(21000+args.seed*100+bi);np.random.seed(21000+args.seed*100+bi)
            m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu")
            opt=torch.optim.Adam(m.parameters(),lr=1e-3);w=pref(beta,args.num_envs,"cuda")
            cur,_=env.reset(seed=200001+args.seed*1000);cur=obs_tensor(cur).cuda();rec=[]
            mark("BETA_START",beta=beta,w=w[0].tolist())
            for update in range(1,args.updates+1):
                ob=[];ac=[];old=[];rw=[];val=[];dn=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp=m.act_with_preference(cur,w);v=m.value_with_preference(cur,w)
                    nxt,_,term,trunc,_=env.step(torch.clamp(a,-1,1))
                    raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);vv=vec_from_raw(raw,names)
                    ob.append(cur);ac.append(a);old.append(lp);rw.append(torch.as_tensor(vv,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
                with torch.no_grad():nv=m.value_with_preference(cur,w)
                rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt)
                fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(old);fw=w.repeat(args.horizon,1)
                ratio=torch.exp(m.logp_with_preference(fo,fw,fa)-fold.detach())
                al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,3).detach(),fw)
                cl=vector_value_loss(m.value_with_preference(fo,fw),ret.reshape(-1,3).detach())
                loss=al+cl;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if update in (1,5,10,25,50,100):
                    rec.append({"update":update,"loss":float(loss.detach()),"reward_vec_stepdt_mean":rt.mean((0,1)).detach().cpu().tolist()})
            cp=args.output.parent/f"beta_{str(beta).replace('.','p')}_terminal.pt"
            torch.save({"schema":"t3_a5_effort_terminal_v1","beta":beta,"preference":w[0].tolist(),"update":args.updates,"model":m.state_dict()},cp)
            models[beta]=m.eval();records[str(beta)]=rec;checkpoints[str(beta)]={"path":str(cp),"sha256":sha(cp)}
            mark("BETA_DONE",beta=beta,checkpoint=str(cp))
        # Matched-state evaluation.
        evalrows=[]
        for suite in range(args.suites):
            seed=210001+suite
            for beta in BETAS:
                m=models[beta];w=pref(beta,args.num_envs,"cuda");cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                tr=[];eff=[];tor=[];vxerr=[];wzerr=[];done=np.zeros(args.num_envs,bool)
                with torch.no_grad():
                    for _ in range(args.eval_steps):
                        a=torch.clamp(m.act_inference_with_preference(cur,w),-1,1)
                        nxt,_,term,trunc,_=env.step(a);raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms);ii={n:i for i,n in enumerate(names)}
                        progress=raw[:,ii[TRACK_TERMS[0]]]+raw[:,ii[TRACK_TERMS[1]]]
                        effort=raw[:,ii[EFFORT_TERM]]
                        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                        tr.append(float(progress.mean()));eff.append(float(effort.mean()))
                        tor.append(float(torch.linalg.vector_norm(data.applied_torque,dim=-1).mean()))
                        vxerr.append(float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()))
                        wzerr.append(float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()))
                        done|=(term|trunc).cpu().numpy();cur=obs_tensor(nxt).cuda()
                evalrows.append({"suite":suite,"beta":beta,"progress_reward":float(np.mean(tr)),"effort_reward":float(np.mean(eff)),
                                 "torque_norm":float(np.mean(tor)),"vx_error":float(np.mean(vxerr)),"wz_error":float(np.mean(wzerr)),
                                 "survival":float(1-done.mean())})
        # Compare to beta=0 per matched suite. Higher progress reward is better, less torque is better.
        comparisons={}
        for beta in BETAS[1:]:
            rows=[]
            for s in range(args.suites):
                b=next(r for r in evalrows if r["suite"]==s and r["beta"]==0.0)
                x=next(r for r in evalrows if r["suite"]==s and r["beta"]==beta)
                effort_imp=(b["torque_norm"]-x["torque_norm"])/(abs(b["torque_norm"])+1e-12)
                track_deg=(b["progress_reward"]-x["progress_reward"])/(abs(b["progress_reward"])+1e-12)
                passes=effort_imp>=ACCEPT["effort_improvement_fraction"] and track_deg<=ACCEPT["tracking_degradation_fraction"]
                rows.append({"suite":s,"effort_improvement_fraction":float(effort_imp),"tracking_degradation_fraction":float(track_deg),"pass":bool(passes),
                             "baseline":b,"candidate":x})
            comparisons[str(beta)]={
              "suite_rows":rows,
              "pass_suites":int(sum(r["pass"] for r in rows)),
              "mean_effort_improvement_fraction":float(np.mean([r["effort_improvement_fraction"] for r in rows])),
              "mean_tracking_degradation_fraction":float(np.mean([r["tracking_degradation_fraction"] for r in rows])),
              "acceptance_pass":bool(sum(r["pass"] for r in rows)>=ACCEPT["required_suites"] and np.mean([r["effort_improvement_fraction"] for r in rows])>0)
            }
        accepted=[float(k) for k,v in comparisons.items() if v["acceptance_pass"]]
        report={"schema":"t3_a5_effort_controllability_screen_v1","status":"SCREEN_COMPLETE","exploratory_training":True,
                "seed":args.seed,"betas":list(BETAS),"acceptance":ACCEPT,"source_checkpoint":str(args.checkpoint),"source_checkpoint_sha256":sha(args.checkpoint),
                "records":records,"checkpoints":checkpoints,"evaluation":evalrows,"comparisons":comparisons,
                "accepted_betas":accepted,"screen_verdict":"EFFORT_CONTROLLABILITY_SUPPORTED" if accepted else "EFFORT_CONTROLLABILITY_NOT_SUPPORTED_IN_SCREEN",
                "note":"Targeted one-seed effort controllability screen only; no final specialist or MORL training."}
        args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",verdict=report["screen_verdict"])
        print(json.dumps({"verdict":report["screen_verdict"],"accepted_betas":accepted,"comparisons":comparisons},indent=2))
    except BaseException as exc:
        args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()
if __name__=="__main__":main()
