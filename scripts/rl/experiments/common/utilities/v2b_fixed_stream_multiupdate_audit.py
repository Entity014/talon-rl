#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
CKPT=ROOT/"runs/v2b1_film_authority-2026-09-23/model_75.pt"
OUT=ROOT/"runs/v2b_fixed_stream_multiupdate_audit-2026-09-24"
ORDER=("T","A","O","S")
PREFS={
"T":np.array([.7,.1,.1,.1],np.float32),
"A":np.array([.1,.7,.1,.1],np.float32),
"O":np.array([.1,.1,.7,.1],np.float32),
"S":np.array([.1,.1,.1,.7],np.float32)}
LAMS=(0.95,1.0)
G=.99;H=32;NENV=8;K=16
BASE_SEED=73001

def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def pref_batch(update,device):
    labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
    return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
def cos(a,b):
    den=float(a.norm()*b.norm())
    return float(torch.dot(a.reshape(-1),b.reshape(-1))/den) if den>1e-12 else 0.0
def flat_grads(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])
def pairwise(grads):
    out={}
    for i,a in enumerate(ORDER):
        for j,b in enumerate(ORDER):
            if j>i:out[f"{a}-{b}"]=cos(grads[a],grads[b])
    return out

def collect_actor(env,m,w,mgr,seed):
    from talon_rl.rewards.objectives import normalized_objective_vector
    cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
    ob=[];u=[];old=[];R=[];D=[]
    with torch.no_grad():
        for _ in range(H):
            a,lp,uu=m.act_with_preference_latent(cur,w)
            nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
            ob.append(cur);u.append(uu);old.append(lp);R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
            D.append((te|tr).cuda());cur=ot(nxt).cuda()
    return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
            "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
            "w":w.repeat(H,1)}

def gae(reward,value,nextv,done,lam):
    adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
    for t in range(H-1,-1,-1):
        boot=nextv if t==H-1 else value[t+1]
        nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
        delta=reward[t]+G*boot*nt-value[t]
        last=delta+G*lam*nt*last;adv[t]=last
    return adv

def actor_params(m):
    return [p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]

def groups(m):
    return {
      "all_actor":actor_params(m),
      "shared_body":[p for n,p in m.named_parameters() if n.startswith("actor_body")],
      "direct_input":[m.actor_body[0].weight],
      "embedding":[p for n,p in m.named_parameters() if n.startswith("preference_embedding")],
      "film":[p for n,p in m.named_parameters() if n.startswith("preference_film")],
      "actor_head":[p for n,p in m.named_parameters() if n.startswith("actor_mean")],
    }

def losses_on_batch(m,b,lam):
    with torch.no_grad():
        vt=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
        w0=b["w"].reshape(H,NENV,4)[-1]
        nv=m.value_with_preference(b["next_obs"],w0)
        A=gae(b["rt"],vt,nv,b["dt"],lam).reshape(-1,4).detach()
    logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
    ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
    comp={}
    for j,lab in enumerate(ORDER):
        po=torch.minimum(ratio*A[:,j],clip*A[:,j])
        comp[lab]=-(4.0*b["w"][:,j]*po).mean()
    return comp,sum(comp.values()),ratio

def geometry(m,b,lam):
    comp,total,ratio=losses_on_batch(m,b,lam)
    out={}
    for gn,ps in groups(m).items():
        g={lab:flat_grads(comp[lab],ps) for lab in ORDER}
        cg=sum(g.values())
        norms={lab:float(g[lab].norm().cpu()) for lab in ORDER};den=sum(norms.values())+1e-12
        out[gn]={
          "norm":norms,"share":{lab:norms[lab]/den for lab in ORDER},
          "pairwise_cos":pairwise(g),"combined_norm":float(cg.norm().cpu()),
          "combined_cos":{lab:cos(cg,g[lab]) for lab in ORDER}}
    return out,float((ratio-1).abs().max().detach().cpu())

def probe_metrics(m,probe):
    acts={}
    with torch.no_grad():
        for lab in ORDER:
            w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
            acts[lab]=m.act_inference_with_preference(probe,w)
    ds=[]
    for i,a in enumerate(ORDER):
        for j,b in enumerate(ORDER):
            if j>i:ds.append(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean())
    w=torch.tensor([[.25,.25,.25,.25]],device="cuda").repeat(len(probe),1).requires_grad_(True)
    aa=m.act_inference_with_preference(probe,w);rows=[]
    for j in range(aa.shape[1]):
        rows.append(torch.autograd.grad(aa[:,j].mean(),w,retain_graph=True)[0])
    jac=torch.stack(rows,dim=1)
    return {"pairwise_action_distance":float(torch.stack(ds).mean().detach().cpu()),
            "preference_jacobian_fro":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--output-dir",type=Path,default=OUT)
    a=ap.parse_args()
    if not a.checkpoint.is_absolute():a.checkpoint=(ROOT/a.checkpoint).resolve()
    if not a.output_dir.is_absolute():a.output_dir=(ROOT/a.output_dir).resolve()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]]
    app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
        base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
        base.load_state_dict(torch.load(a.checkpoint,map_location="cuda",weights_only=False)["model"]);base.eval()

        stream=[];probe=None
        for u in range(1,K+1):
            _,w=pref_batch(u,torch.device("cuda"))
            b=collect_actor(env,base,w,mgr,BASE_SEED+u*211);stream.append(b)
            if probe is None:probe=b["obs"][:64].detach().clone()

        models={};opts={}
        init=base.state_dict()
        for lam in LAMS:
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.train()
            models[lam]=m;opts[lam]=torch.optim.Adam(actor_params(m),lr=1e-3)

        rows={str(l):[] for l in LAMS}
        checkpoints=(0,1,2,4,8,16)
        for lam in LAMS:
            geom,ratioerr=geometry(models[lam],stream[0],lam)
            rows[str(lam)].append({"step":0,"geometry":geom,"ratio_error":ratioerr,"probe":probe_metrics(models[lam],probe)})
        for step,b in enumerate(stream,1):
            for lam in LAMS:
                m=models[lam];opt=opts[lam]
                comp,loss,ratio=losses_on_batch(m,b,lam)
                opt.zero_grad(set_to_none=True);loss.backward()
                preclip=float(torch.nn.utils.clip_grad_norm_(actor_params(m),1.0).detach().cpu())
                opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                if step in checkpoints:
                    nextb=stream[min(step,K-1)]
                    geom,ratioerr=geometry(m,nextb,lam)
                    rows[str(lam)].append({"step":step,"loss":float(loss.detach().cpu()),"preclip_grad_norm":preclip,
                                          "geometry":geom,"ratio_error":ratioerr,"probe":probe_metrics(m,probe)})

        paired=[]
        for idx in range(len(rows["0.95"])):
            a95=rows["0.95"][idx];a1=rows["1.0"][idx];assert a95["step"]==a1["step"]
            d={"step":a95["step"],"probe":{
                "pairwise_action_distance_delta":a1["probe"]["pairwise_action_distance"]-a95["probe"]["pairwise_action_distance"],
                "preference_jacobian_delta":a1["probe"]["preference_jacobian_fro"]-a95["probe"]["preference_jacobian_fro"]},"groups":{}}
            for gn in groups(models[0.95]):
                g95=a95["geometry"][gn];g1=a1["geometry"][gn]
                d["groups"][gn]={
                    "share_delta":{lab:g1["share"][lab]-g95["share"][lab] for lab in ORDER},
                    "combined_cos_delta":{lab:g1["combined_cos"][lab]-g95["combined_cos"][lab] for lab in ORDER},
                    "pairwise_cos_delta":{k:g1["pairwise_cos"][k]-g95["pairwise_cos"][k] for k in g95["pairwise_cos"]},
                    "combined_norm_ratio":g1["combined_norm"]/(g95["combined_norm"]+1e-12)}
            paired.append(d)

        report={"schema":"v2b_fixed_stream_multiupdate_audit_v1","measurement_only":True,
                "environment_recollection_during_updates":False,"optimizer_steps_per_arm":K,
                "checkpoint":str(a.checkpoint.relative_to(ROOT)),"lambdas":[0.95,1.0],
                "optimizer":"Adam lr=1e-3, grad clip=1.0","frozen_stream_updates":K,
                "rows":rows,"paired":paired}
        out=a.output_dir/"fixed_stream_multiupdate_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (a.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),"checkpoint_sha256":sha(a.checkpoint),"script_sha256":sha(Path(__file__).resolve())},indent=2)+"\n")
        print(json.dumps(paired,indent=2))
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
