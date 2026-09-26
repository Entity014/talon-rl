#!/usr/bin/env python3
"""V2-R2.3-A read-only semantic coefficient-guidance gradient compatibility audit."""
from __future__ import annotations
import argparse,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np,torch
from torch.distributions import Normal

ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
PREFS={"P":np.array([.8,.1,.1],np.float32),"B":np.array([.1,.8,.1],np.float32),"E":np.array([.1,.1,.8],np.float32)}
SNAPS=(0,1,5,10)
LAMBDAS=(0.0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0)
EPS=1e-12
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def early_ppo(ratio,adv,w,eps=.2):
    scalar=(adv*w).sum(-1);clipped=ratio.clamp(1-eps,1+eps)
    return -torch.minimum(ratio*scalar,clipped*scalar).mean()
def sign(v,tol=1e-12):return 0 if abs(v)<=tol else (1 if v>0 else -1)
def flat_grads(grads):
    xs=[]
    for g in grads:
        if g is not None: xs.append(g.reshape(-1))
    return torch.cat(xs) if xs else torch.zeros(1,device="cuda")
def target_coeff(w,verts):
    return w@verts

def logp_with_coeff(model,obs,coeff,action):
    h=torch.nn.functional.elu(model.actor_pre(obs))
    h=h+model.projection_alpha*(coeff@model.semantic_basis)
    h=model.actor_rest(h);mean=model.actor_mean(h)
    std=(model.log_std if model.exploration_mode=="learned" else model.scheduled_log_std).exp()
    dist=Normal(mean,std)
    u=torch.atanh((action/model.ACTION_CLIP).clamp(-1+model._ATANH_EPS,1-model._ATANH_EPS))
    return model._squash(dist,u)[1]
def pair_ok(v,target):
    if target[0]!=0 and sign(v[0])!=target[0]: return False
    if target[1]!=0 and sign(v[1])!=target[1]: return False
    return True
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-checkpoint",type=Path,required=True);ap.add_argument("--basis",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--num-envs",type=int,default=8)
    ap.add_argument("--horizon",type=int,default=2);ap.add_argument("--probe-horizon",type=int,default=8);ap.add_argument("--suites",type=int,default=4);ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    anchors_j=json.loads((ROOT/"runs/post_v2_a-2026-09-23/v2a.json").read_text());anchors=torch.tensor([anchors_j["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
    r21=json.loads((ROOT/"runs/post_v2_r21_shared_residual-2026-09-23/audit.json").read_text())
    d1=r21["summary"]["h1"]["D1"]
    cP=np.array([0.0,0.0],np.float32)
    cB=np.array([d1["PB"]["c_shared_mean"],d1["PB"]["c_BE_mean"]],np.float32)
    cE=np.array([d1["PE"]["c_shared_mean"],d1["PE"]["c_BE_mean"]],np.float32)
    verts_cpu=np.stack([cP,cB,cE],0)
    targets={"PB":[1,-1],"PE":[1,1],"BE":[0,1]}
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    env=None
    try:
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
        verts=torch.tensor(verts_cpu,device="cuda")
        model=V2R2ProjectedActorCritic(obs.shape[-1],ad,basis,anchors=anchors,projection_alpha=1.0).cuda();initialize_from_v1c(model,source)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(args.seed+300000)
        current,_=env.reset(seed=args.seed);current=obs_tensor(current).cuda()
        rows=[]
        coeff_params=list(model.coeff_net.parameters())

        def probe(update):
            model.eval(); pref={}
            for pi,(label,pv) in enumerate(PREFS.items()):
                suites=[]
                for suite in range(args.suites):
                    w=torch.as_tensor(np.repeat(pv[None,:],args.num_envs,0),device="cuda")
                    cur,_=env.reset(seed=160000+update*100+pi*10+suite);cur=obs_tensor(cur).cuda()
                    ob=[];ac=[];ol=[];rw=[];va=[];dn=[]
                    for _ in range(args.probe_horizon):
                        with torch.no_grad():action,old=model.act_with_preference(cur,w);value=model.value_with_preference(cur,w)
                        nxt,_,term,trunc,_=env.step(torch.clamp(action,-1,1));mgr=env.unwrapped.reward_manager;raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                        vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                        ob.append(cur);ac.append(action);ol.append(old);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);va.append(value);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
                    with torch.no_grad():nv=model.value_with_preference(cur,w)
                    rew=torch.stack(rw);vals=torch.stack(va);done=torch.stack(dn).bool();adv,_=vector_gae(rew,vals,nv,done)
                    fo=torch.cat(ob);fa=torch.cat(ac);fold=torch.cat(ol).detach();fw=w.repeat(args.probe_horizon,1)
                    # Direct coefficient-output pressure.
                    coeff=model.semantic_coefficients(fw).detach().requires_grad_(True)
                    lp=logp_with_coeff(model,fo,coeff,fa);ratio=torch.exp(lp-fold);ppo=early_ppo(ratio,adv.reshape(-1,3).detach(),fw)
                    cstar=target_coeff(fw,verts)
                    lc=((coeff-cstar)**2).mean()
                    gp_out=torch.autograd.grad(ppo,coeff,retain_graph=True,allow_unused=False)[0]
                    gc_out=torch.autograd.grad(lc,coeff,retain_graph=True,allow_unused=False)[0]
                    # Parameter-space compatibility: recompute losses through coeff_net parameters.
                    coeff_param=model.semantic_coefficients(fw)
                    lp_param=logp_with_coeff(model,fo,coeff_param,fa)
                    ppo_param=early_ppo(torch.exp(lp_param-fold),adv.reshape(-1,3).detach(),fw)
                    lc_param=((coeff_param-cstar)**2).mean()
                    gp=flat_grads(torch.autograd.grad(ppo_param,coeff_params,retain_graph=True,allow_unused=True))
                    gc=flat_grads(torch.autograd.grad(lc_param,coeff_params,retain_graph=True,allow_unused=True))
                    npp=float(gp.norm()); nc=float(gc.norm()); cos=float((gp@gc)/(gp.norm()*gc.norm()+EPS))
                    suites.append({
                      "suite":suite,"coeff_mean":coeff.detach().mean(0).cpu().tolist(),"target_mean":cstar.detach().mean(0).cpu().tolist(),
                      "ppo_update_dir_out":(-gp_out).mean(0).detach().cpu().tolist(),
                      "coeff_update_dir_out":(-gc_out).mean(0).detach().cpu().tolist(),
                      "gppo_param_norm":npp,"gcoeff_param_norm":nc,"param_grad_cosine":cos,
                      "raw_ratio_gcoeff_over_gppo":nc/(npp+EPS)
                    })
                pref[label]=suites
            # candidate lambda combined output pressure
            cand={}
            for lam in LAMBDAS:
                ag={}
                for label,rs in pref.items():
                    up=np.mean([np.array(r["ppo_update_dir_out"])+lam*np.array(r["coeff_update_dir_out"]) for r in rs],axis=0)
                    ag[label]=up
                pairs={}
                allok=True
                for a,b,name in (("P","B","PB"),("P","E","PE"),("B","E","BE")):
                    v=ag[b]-ag[a];ok=pair_ok(v,targets[name]);allok=allok and ok
                    suites_ok=[]
                    for i in range(args.suites):
                        va=np.array(pref[a][i]["ppo_update_dir_out"])+lam*np.array(pref[a][i]["coeff_update_dir_out"])
                        vb=np.array(pref[b][i]["ppo_update_dir_out"])+lam*np.array(pref[b][i]["coeff_update_dir_out"])
                        suites_ok.append(pair_ok(vb-va,targets[name]))
                    pairs[name]={"vector":v.tolist(),"sign":[sign(x) for x in v],"target":targets[name],"aggregate_ok":ok,"suite_match_fraction":float(np.mean(suites_ok))}
                # parameter weighted ratio averaged over pref/suites
                wr=np.mean([lam*r["raw_ratio_gcoeff_over_gppo"] for rs in pref.values() for r in rs])
                cand[str(lam)]={"pairs":pairs,"all_aggregate_ok":allok,"mean_lambda_gcoeff_over_gppo":float(wr)}
            return {"update":update,"preferences":pref,"lambda_candidates":cand}
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
            if update in SNAPS:rows.append(probe(update));model.train()

        # smallest lambda satisfying all aggregate signs at all audited snapshots
        effective=[]
        for lam in LAMBDAS:
            key=str(lam)
            if all(s["lambda_candidates"][key]["all_aggregate_ok"] for s in rows):
                effective.append(lam)
        min_eff=effective[0] if effective else None
        report={
          "schema":"post_v2_r23a_coeff_guidance_gradient_audit_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
          "source_checkpoint":str(args.source_checkpoint),"basis":str(args.basis),"basis_sha256":sha(args.basis),
          "coefficient_vertices":{"P":cP.tolist(),"B":cB.tolist(),"E":cE.tolist()},
          "target_definition":"c*(w)=w_P cP + w_B cB + w_E cE",
          "lambda_candidates":list(LAMBDAS),"snapshots":rows,
          "minimum_all_snapshot_aggregate_orientation_lambda":min_eff,
          "acceptance_rule":"minimum lambda whose combined output pressure has correct aggregate D1 signs for PB/PE/BE at updates 0/1/5/10",
          "note":"Read-only gradient algebra. R2 replay uses original PPO training only; coefficient guidance is never applied to optimizer."
        }
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"status":report["status"],"minimum_lambda":min_eff},indent=2))
    except BaseException as exc:
        args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");raise
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
