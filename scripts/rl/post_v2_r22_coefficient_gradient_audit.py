#!/usr/bin/env python3
"""V2-R2.2 diagnostic-only coefficient gradient orientation audit."""
from __future__ import annotations
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np,torch
from torch.distributions import Normal

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
PREFS={"P":np.array([.8,.1,.1],np.float32),"B":np.array([.1,.8,.1],np.float32),"E":np.array([.1,.1,.8],np.float32)}
SNAPS=(0,1,5,10); EPS=1e-12
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def early_ppo(ratio,adv,w,eps=.2):
    scalar=(adv*w).sum(-1);clipped=ratio.clamp(1-eps,1+eps)
    return -torch.minimum(ratio*scalar,clipped*scalar).mean()
def discounted_returns(rew,done,gamma=.99):
    out=torch.zeros_like(rew);running=torch.zeros_like(rew[0])
    for t in range(rew.shape[0]-1,-1,-1):
        running=rew[t]+gamma*running*(~done[t]).unsqueeze(-1)
        out[t]=running
    return out
def logp_with_coeff(model,obs,coeff,action):
    h=torch.nn.functional.elu(model.actor_pre(obs))
    h=h+model.projection_alpha*(coeff@model.semantic_basis)
    h=model.actor_rest(h);mean=model.actor_mean(h)
    std=(model.log_std if model.exploration_mode=="learned" else model.scheduled_log_std).exp()
    dist=Normal(mean,std)
    u=torch.atanh((action/model.ACTION_CLIP).clamp(-1+model._ATANH_EPS,1-model._ATANH_EPS))
    return model._squash(dist,u)[1]
def sign(x,tol=1e-10):return 0 if abs(x)<=tol else (1 if x>0 else -1)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-checkpoint",type=Path,required=True);ap.add_argument("--basis",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--num-envs",type=int,default=8)
    ap.add_argument("--horizon",type=int,default=2);ap.add_argument("--probe-horizon",type=int,default=8);ap.add_argument("--suites",type=int,default=4);ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
    def mark(event,**extra):
        with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
    mark("RUN_STARTED",protocol="V2-R2.2",diagnostic_only=True,snapshots=list(SNAPS));app=env=None
    try:
        anchors_j=json.loads((ROOT/"runs/post_v2_a-2026-09-23/v2a.json").read_text());anchors=torch.tensor([anchors_j["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
        r21=json.loads((ROOT/"runs/post_v2_r21_shared_residual-2026-09-23/audit.json").read_text())
        d1_target={"PB":[1,-1],"PE":[1,1],"BE":[0,1]}
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.v1b_s7_reward_vector import group_v1b_s7_terms
        from talon_rl.v1c_actor_critic import vector_gae,vector_value_loss
        from talon_rl.v2_r2_projected_actor_critic import V2R2ProjectedActorCritic,initialize_from_v1c
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        obs,_=env.reset(seed=args.seed);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
        source=torch.load(args.source_checkpoint,map_location="cpu",weights_only=False)["model"]
        bp=torch.load(args.basis,map_location="cuda",weights_only=False);basis=torch.stack([bp["b_shared"],bp["b_BE"]],0).cuda()
        model=V2R2ProjectedActorCritic(obs.shape[-1],ad,basis,anchors=anchors,projection_alpha=1.0).cuda();initialize_from_v1c(model,source)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+300000)
        current,_=env.reset(seed=args.seed);current=obs_tensor(current).cuda()
        rows=[]

        def probe(snapshot_update):
            model.eval(); pref_results={}
            for pi,(label,pv) in enumerate(PREFS.items()):
                suite_rows=[]
                for suite in range(args.suites):
                    w=torch.as_tensor(np.repeat(pv[None,:],args.num_envs,0),device="cuda")
                    cur,_=env.reset(seed=150000+snapshot_update*100+pi*10+suite);cur=obs_tensor(cur).cuda()
                    obs_b=[];act_b=[];old_b=[];rew_b=[];val_b=[];done_b=[]
                    for _ in range(args.probe_horizon):
                        with torch.no_grad(): action,old=model.act_with_preference(cur,w);value=model.value_with_preference(cur,w)
                        nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1))
                        mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                        vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                        obs_b.append(cur);act_b.append(action);old_b.append(old);rew_b.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);val_b.append(value);done_b.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
                    with torch.no_grad():nv=model.value_with_preference(cur,w)
                    rew=torch.stack(rew_b);vals=torch.stack(val_b);done=torch.stack(done_b).bool()
                    gae,_=vector_gae(rew,vals,nv,done);rtg=discounted_returns(rew,done)
                    fo=torch.cat(obs_b);fa=torch.cat(act_b);fold=torch.cat(old_b).detach();fw=w.repeat(args.probe_horizon,1)
                    coeff0=model.semantic_coefficients(fw).detach().requires_grad_(True)
                    lp=logp_with_coeff(model,fo,coeff0,fa);ratio=torch.exp(lp-fold)
                    def grad_for(adv):
                        loss=early_ppo(ratio,adv.reshape(-1,3).detach(),fw)
                        g=torch.autograd.grad(loss,coeff0,retain_graph=True)[0]
                        return (-g).mean(0)
                    g_gae=grad_for(gae);g_rtg=grad_for(rtg)
                    obj=[]
                    for j in range(3):
                        mask=torch.zeros_like(gae);mask[:,:,j]=gae[:,:,j]
                        obj.append(grad_for(mask))
                    suite_rows.append({
                      "suite":suite,"coeff":model.semantic_coefficients(w)[0].detach().cpu().tolist(),
                      "update_dir_gae":g_gae.detach().cpu().tolist(),"update_dir_rtg":g_rtg.detach().cpu().tolist(),
                      "objective_update_dirs":[x.detach().cpu().tolist() for x in obj],
                      "adv_mean":gae.mean((0,1)).detach().cpu().tolist(),"rtg_mean":rtg.mean((0,1)).detach().cpu().tolist()
                    })
                pref_results[label]=suite_rows
            # Aggregate preference pressure then implied pairwise orientation
            agg={}
            for label,rs in pref_results.items():
                for mode in ("update_dir_gae","update_dir_rtg"):
                    agg.setdefault(label,{})[mode]=np.mean([r[mode] for r in rs],axis=0).tolist()
                agg[label]["objective_update_dirs"]=np.mean([r["objective_update_dirs"] for r in rs],axis=0).tolist()
            pair={}
            for a,b,name in (("P","B","PB"),("P","E","PE"),("B","E","BE")):
                pair[name]={}
                for mode in ("update_dir_gae","update_dir_rtg"):
                    v=np.asarray(agg[b][mode])-np.asarray(agg[a][mode]);target=np.asarray(d1_target[name],float)
                    cos=float(v@target/(np.linalg.norm(v)*np.linalg.norm(target)+EPS))
                    pair[name][mode]={"vector":v.tolist(),"sign":[sign(x) for x in v],"target_sign":d1_target[name],"cosine_to_D1_target":cos}
            return {"update":snapshot_update,"preferences":pref_results,"aggregate":agg,"pairwise_pressure":pair}
        rows.append(probe(0));model.train()

        for update in range(1,max(SNAPS)+1):
            w=torch.as_tensor(rng.dirichlet(np.ones(3),size=args.num_envs),device="cuda",dtype=torch.float32)
            ob=[];ac=[];ol=[];rw=[];va=[];dn=[]
            for _ in range(args.horizon):
                with torch.no_grad():action,old=model.act_with_preference(current,w);value=model.value_with_preference(current,w)
                nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1));mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                ob.append(current);ac.append(action);ol.append(old);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);va.append(value);dn.append((term|trunc).cuda());current=obs_tensor(nxt).cuda()
            with torch.no_grad():nv=model.value_with_preference(current,w)
            rew=torch.stack(rw);vals=torch.stack(va);done=torch.stack(dn).bool();adv,ret=vector_gae(rew,vals,nv,done)
            fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(ol);fw=w.repeat(args.horizon,1)
            ratio=torch.exp(model.logp_with_preference(fo,fw,fa)-fold.detach());ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),fw)
            manifold=model.manifold_loss(fw);critic=vector_value_loss(model.value_with_preference(fo,fw),ret.reshape(-1,3).detach());loss=ppo+critic+0.1*manifold
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if update in SNAPS:
                rows.append(probe(update));model.train();mark("SNAPSHOT",update=update)

        report={"schema":"post_v2_r22_coefficient_gradient_audit_v1","status":"MEASUREMENT_COMPLETE","diagnostic_only":True,
          "source_checkpoint":str(args.source_checkpoint),"basis":str(args.basis),"basis_sha256":sha(args.basis),
          "snapshots":rows,"d1_target_signs":d1_target,
          "definitions":{"update_direction":"-dL_PPO/d coefficient output","pairwise_pressure":"update_direction(target)-update_direction(P/reference)",
                         "rtg_comparator":"discounted vector reward-to-go without critic baseline"},
          "note":"Training replay only reconstructs frozen R2 early updates; no architecture/loss/optimizer changes."}
        args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE")
        print(json.dumps({"status":report["status"],"snapshots":[x["update"] for x in rows]},indent=2))
    except BaseException as exc:
        args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()
if __name__=="__main__":main()
