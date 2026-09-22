#!/usr/bin/env python3
"""V2-R2 one-seed short coefficient/authority screen. No semantic verdict."""
from __future__ import annotations
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
PREFS={"P":np.array([.8,.1,.1],np.float32),"B":np.array([.1,.8,.1],np.float32),"E":np.array([.1,.1,.8],np.float32)}
CHECKPOINT_UPDATES=(0,1,5,10,25,50,100)
CODE_COMMIT="3d0b4dd2db9e0a140bbe3bc37f958a3f52be0dbe"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def tilt_deg(q):return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))
def early_ppo(ratio,adv,w,eps=.2):
    scalar=(adv*w).sum(-1);clipped=ratio.clamp(1-eps,1+eps)
    return -torch.minimum(ratio*scalar,clipped*scalar).mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-checkpoint",type=Path,required=True)
    ap.add_argument("--basis",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--updates",type=int,default=100)
    ap.add_argument("--num-envs",type=int,default=8)
    ap.add_argument("--horizon",type=int,default=2)
    ap.add_argument("--eval-steps",type=int,default=16)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
    def mark(event,**extra):
        with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
    basis_sha=sha(args.basis)
    mark("RUN_STARTED",protocol="V2-R2-SHORT-COEFFICIENT-SCREEN",training=True,updates=args.updates,basis_sha256=basis_sha)
    app=env=None
    try:
        anchor_report=json.loads((ROOT/"runs/post_v2_a-2026-09-23/v2a.json").read_text())
        anchors=torch.tensor([anchor_report["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
        r21=json.loads((ROOT/"runs/post_v2_r21_shared_residual-2026-09-23/audit.json").read_text())
        d1_ref={
          "h1_shared_PB":r21["summary"]["h1"]["D1"]["PB"]["c_shared_mean"],
          "h1_shared_PE":r21["summary"]["h1"]["D1"]["PE"]["c_shared_mean"],
          "h1_BE_PB":r21["summary"]["h1"]["D1"]["PB"]["c_BE_mean"],
          "h1_BE_PE":r21["summary"]["h1"]["D1"]["PE"]["c_BE_mean"],
          "h1_BE_BE":r21["summary"]["h1"]["D1"]["BE"]["c_BE_mean"],
        }
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app
        sys.argv=saved;mark("APP_INIT_OK")
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.v1b_s7_reward_vector import group_v1b_s7_terms
        from talon_rl.v1c_actor_critic import vector_gae,vector_value_loss
        from talon_rl.v2_r2_projected_actor_critic import V2R2ProjectedActorCritic,initialize_from_v1c

        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);mark("ENV_CREATED")
        obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
        source_payload=torch.load(args.source_checkpoint,map_location="cpu",weights_only=False);source=source_payload["model"]
        bp=torch.load(args.basis,map_location="cuda",weights_only=False)
        basis=torch.stack([bp["b_shared"],bp["b_BE"]],0).cuda()
        model=V2R2ProjectedActorCritic(obs.shape[-1],ad,basis,anchors=anchors,projection_alpha=1.0).cuda()
        initialize_from_v1c(model,source)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+300000)
        mark("CHECKPOINT_LOADED",source=str(args.source_checkpoint))
        current,_=env.reset(seed=args.seed);current=obs_tensor(current).cuda()
        train_records=[];snapshots=[]

        def evaluate_snapshot(update:int):
            model.eval()
            with torch.no_grad():
                w={k:torch.as_tensor(np.repeat(v[None,:],args.num_envs,axis=0),device="cuda") for k,v in PREFS.items()}
                coeff={k:model.semantic_coefficients(x)[0].cpu().tolist() for k,x in w.items()}
                delta={k:model.projected_delta(x)[0].cpu().tolist() for k,x in w.items()}
                cd={
                  "PB":[coeff["B"][0]-coeff["P"][0],coeff["B"][1]-coeff["P"][1]],
                  "PE":[coeff["E"][0]-coeff["P"][0],coeff["E"][1]-coeff["P"][1]],
                  "BE":[coeff["E"][0]-coeff["B"][0],coeff["E"][1]-coeff["B"][1]],
                }
                pd={
                  "PB":float(torch.linalg.vector_norm(model.projected_delta(w["B"])-model.projected_delta(w["P"]),dim=-1).mean()),
                  "PE":float(torch.linalg.vector_norm(model.projected_delta(w["E"])-model.projected_delta(w["P"]),dim=-1).mean()),
                  "BE":float(torch.linalg.vector_norm(model.projected_delta(w["E"])-model.projected_delta(w["B"]),dim=-1).mean()),
                }
                # fixed-state action distance on one matched reset state
                cur,_=env.reset(seed=93000);cur=obs_tensor(cur).cuda()
                acts={k:torch.clamp(model.act_inference_with_preference(cur,w[k]),-1,1) for k in PREFS}
                adist={
                  "PB":float(torch.linalg.vector_norm(acts["B"]-acts["P"],dim=-1).mean()),
                  "PE":float(torch.linalg.vector_norm(acts["E"]-acts["P"],dim=-1).mean()),
                  "BE":float(torch.linalg.vector_norm(acts["E"]-acts["B"],dim=-1).mean()),
                }
                behavior=[]
                for idx,(label,pref) in enumerate(PREFS.items()):
                    ww=w[label];cur,_=env.reset(seed=94000+idx);cur=obs_tensor(cur).cuda()
                    done_any=np.zeros(args.num_envs,bool);tilts=[];finite=True
                    for _ in range(args.eval_steps):
                        action=torch.clamp(model.act_inference_with_preference(cur,ww),-1,1)
                        finite=finite and bool(torch.isfinite(action).all())
                        nxt,_,term,trunc,_=env.step(action);done_any|=(term|trunc).cpu().numpy()
                        tilts.append(float(torch.quantile(tilt_deg(env.unwrapped.scene["robot"].data.root_quat_w),.95)))
                        cur=obs_tensor(nxt).cuda()
                    behavior.append({"preference":label,"survival":float(1-done_any.mean()),"tilt_p95_mean":float(np.mean(tilts)),"finite":finite})
                snap={
                  "update":update,"coefficients":coeff,"coefficient_deltas":cd,
                  "projected_delta_norm":pd,"fixed_state_action_distance":adist,
                  "behavior":behavior,
                  "basis_sha256_runtime":sha(args.basis),
                  "basis_unchanged":sha(args.basis)==basis_sha,
                  "d1_reference":d1_ref,
                }
            model.train();return snap

        snapshots.append(evaluate_snapshot(0))
        for update in range(1,args.updates+1):
            w=torch.as_tensor(rng.dirichlet(np.ones(3),size=args.num_envs),device="cuda",dtype=torch.float32)
            obs_buf=[];act_buf=[];old_buf=[];rew_buf=[];val_buf=[];done_buf=[]
            for _ in range(args.horizon):
                with torch.no_grad():action,old=model.act_with_preference(current,w);value=model.value_with_preference(current,w)
                nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1))
                manager=env.unwrapped.reward_manager;raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                obs_buf.append(current);act_buf.append(action);old_buf.append(old);rew_buf.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);val_buf.append(value);done_buf.append((term|trunc).to("cuda"));current=obs_tensor(nxt).cuda()
            with torch.no_grad():next_value=model.value_with_preference(current,w)
            reward_t=torch.stack(rew_buf);value_t=torch.stack(val_buf);done_t=torch.stack(done_buf).bool()
            adv,ret=vector_gae(reward_t,value_t,next_value,done_t)
            flat_obs=torch.cat(obs_buf);flat_act=torch.cat(act_buf);flat_old=torch.cat(old_buf);flat_w=w.repeat(args.horizon,1)
            ratio=torch.exp(model.logp_with_preference(flat_obs,flat_w,flat_act)-flat_old.detach())
            ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),flat_w)
            manifold=model.manifold_loss(flat_w)
            critic=vector_value_loss(model.value_with_preference(flat_obs,flat_w),ret.reshape(-1,3).detach())
            loss=ppo+critic+0.1*manifold
            optimizer.zero_grad(set_to_none=True);loss.backward()
            finite=bool(torch.isfinite(loss).item() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            train_records.append({"update":update,"loss":float(loss.detach()),"ppo_loss":float(ppo.detach()),"critic_loss":float(critic.detach()),"manifold_loss":float(manifold.detach()),"finite":finite})
            if update in CHECKPOINT_UPDATES:
                snapshots.append(evaluate_snapshot(update))
                mark("SNAPSHOT",update=update)

        terminal=args.output.parent/"v2r2_terminal.pt"
        torch.save({"schema":"post_v2_r2_terminal_v1","seed":args.seed,"update":args.updates,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"anchors":anchors.tolist(),"basis_sha256":basis_sha,"projection_alpha":1.0},terminal)
        mark("CHECKPOINT_WRITTEN",checkpoint=str(terminal))
        final=snapshots[-1]
        sign_gate={
          "PB_shared_positive":final["coefficient_deltas"]["PB"][0]>0,
          "PB_BE_negative":final["coefficient_deltas"]["PB"][1]<0,
          "PE_shared_positive":final["coefficient_deltas"]["PE"][0]>0,
          "PE_BE_positive":final["coefficient_deltas"]["PE"][1]>0,
          "BE_BE_positive":final["coefficient_deltas"]["BE"][1]>0,
        }
        report={
          "schema":"post_v2_r2_short_coefficient_screen_v1","status":"SCREEN_COMPLETE","training":True,"diagnostic_only":True,
          "semantic_verdict":False,"seed":args.seed,"updates":args.updates,
          "source_checkpoint":str(args.source_checkpoint),"terminal_checkpoint":str(terminal),
          "basis":str(args.basis),"basis_sha256":basis_sha,"code_commit":CODE_COMMIT,
          "lambda_manifold":0.1,"projection_alpha":1.0,
          "checkpoints":list(CHECKPOINT_UPDATES),"train_records":train_records,"snapshots":snapshots,
          "final_sign_gate":sign_gate,"final_sign_gate_all":all(sign_gate.values()),
          "note":"R2 short coefficient/authority screen only. No semantic or multi-seed authorization."
        }
        args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
        print(json.dumps({"status":report["status"],"final_sign_gate":sign_gate,"terminal":str(terminal)},indent=2))
    except BaseException as exc:
        args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()
if __name__=="__main__":main()
