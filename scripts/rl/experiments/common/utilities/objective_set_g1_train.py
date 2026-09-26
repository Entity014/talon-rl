#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,itertools,json,sys,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic,canonical_tokens
from talon_rl.rewards.objectives import normalized_objective_vector
INIT=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
CONTRACT=ROOT/"docs/contracts/objective_set/objective-set-g1-variable-cardinality-contract.md"
TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
NENV=8;H=32;GAMMA=.99;LAM=.95;CLIP=.20;RHO=.25;BETA0=2.497041993384243;EDGE_GAMMA=.90;KAPPA=.05;EPS=1e-12
SNAPS=(0,10,20,30);ORDER=("T","A","O","S");OBJ_IDX={x:i for i,x in enumerate(ORDER)}
FOLDS={"G1-2":{"train":(3,4),"holdout":2},"G1-3":{"train":(2,4),"holdout":3}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ot(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def gae(r,v,nv,d):
    adv=torch.zeros_like(r);last=torch.zeros_like(nv)
    for t in range(H-1,-1,-1):
        boot=nv if t==H-1 else v[t+1];nt=(~d[t]).to(r.dtype).unsqueeze(-1)
        delta=r[t]+GAMMA*boot*nt-v[t];last=delta+GAMMA*LAM*nt*last;adv[t]=last
    return adv,adv+v
def sample_sets(fold,seed,update,device):
    rng=np.random.default_rng(seed*100003+update*9973)
    allowed=FOLDS[fold]["train"];m=int(rng.choice(allowed))
    combos=list(itertools.combinations(range(4),m));ids=[];ws=[];modes=[]
    for _ in range(NENV):
        sub=np.array(combos[int(rng.integers(len(combos)))],np.int64);ids.append(sub)
        q=float(rng.random())
        if q<.20:w=np.full(m,1/m,np.float32);mode="center"
        elif q<.60:
            h=int(rng.integers(m));w=np.full(m,.30/(m-1),np.float32);w[h]=.70;mode=f"heavy{h}"
        else:w=rng.dirichlet(np.ones(m)).astype(np.float32);mode="interior"
        ws.append(w);modes.append(mode)
    ids=torch.tensor(np.stack(ids),device=device,dtype=torch.long)
    w=torch.tensor(np.stack(ws),device=device,dtype=torch.float32)
    tok=canonical_tokens(device=device)[ids]
    assert ids.shape[1] in allowed and ids.shape[1]!=FOLDS[fold]["holdout"]
    return m,ids,tok,w,modes

def collect(env,mgr,model,fold,seed,update):
    card,ids,tok,w,modes=sample_sets(fold,seed,update,torch.device("cuda"))
    obs,_=env.reset(seed=seed+update*211);obs=ot(obs).cuda()
    O=[];U=[];LP=[];R=[];D=[];V=[]
    with torch.no_grad():
        for _ in range(H):
            v=model.query_values_from_set(obs,tok,w,tok)
            a,lp,u=model.act_with_set_latent(obs,tok,w)
            nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            full=torch.tensor(normalized_objective_vector(terms(raw,names),shape=(NENV,)),device="cuda")*env.unwrapped.step_dt
            active=torch.gather(full,1,ids)
            O.append(obs);U.append(u);LP.append(lp);R.append(active);D.append((te|tr).cuda());V.append(v)
            obs=ot(nxt).cuda()
        nv=model.query_values_from_set(obs,tok,w,tok)
    return {"card":card,"ids":ids,"tok":tok,"w":w,"modes":modes,"obs":torch.stack(O),"u":torch.stack(U),
      "oldlp":torch.stack(LP),"r":torch.stack(R),"d":torch.stack(D).bool(),"v":torch.stack(V),"nextv":nv,
      "termination_fraction":float(torch.stack(D).any(0).float().mean().cpu())}
def tail_metrics(model,obs,tok,w,tau):
    z=model.pre_tanh_mean_from_set(obs,tok,w);ex=torch.relu(z.abs()-tau)
    return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()

def endpoint_weights(ids):
    B,M=ids.shape;dev=ids.device
    center=torch.full((B,M),1.0/M,device=dev)
    heavy=[]
    for h in range(M):
        w=torch.full((B,M),.30/(M-1),device=dev);w[:,h]=.70;heavy.append(w)
    return [center]+heavy

def edge_loss(model,ref,obs,ids):
    # obs/ids are one state per lane or a matched repeated batch; cardinality is training-supported only.
    tok=canonical_tokens(device=obs.device)[ids];prefs=endpoint_weights(ids);acts=[];refs=[]
    for w in prefs:
        acts.append(model.act_inference_from_set(obs,tok,w))
        with torch.no_grad():refs.append(ref.act_inference_from_set(obs,tok,w))
    terms=[]
    for i in range(len(prefs)):
        for j in range(i+1,len(prefs)):
            d=torch.linalg.vector_norm(acts[i]-acts[j],dim=-1)
            dr=torch.linalg.vector_norm(refs[i]-refs[j],dim=-1)
            terms.append(torch.relu(EDGE_GAMMA*dr-d).pow(2))
    return torch.stack(terms,1).mean()

def grad_norm(gs):
    return torch.sqrt(sum((g.detach()**2).sum() for g in gs if g is not None)+EPS)

def audit_snapshot(model,out,u,fold,seed):
    torch.save({"model":model.state_dict(),"update":u,"fold":fold,"seed":seed},out/f"model_{u}.pt")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--fold",choices=tuple(FOLDS),required=True);ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--updates",type=int,default=30);args=ap.parse_args()
    if args.updates!=30:raise ValueError("G1 frozen budget is 30 updates")
    out=ROOT/f"runs/objective_set_{args.fold.lower().replace('-','_')}_seed{args.seed}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=args.seed);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
        base=torch.load(INIT,map_location="cuda",weights_only=False)["model"]
        model=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();model.load_state_dict(base);model.train()
        ref=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();ref.load_state_dict(base);ref.eval()
        for p in ref.parameters():p.requires_grad_(False)
        actor_names=lambda n:n.startswith("actor_") or n.startswith("family_") or n=="log_std"
        critic_names=lambda n:n.startswith("critic_body") or n.startswith("value_basis")
        actor_params=[p for n,p in model.named_parameters() if actor_names(n)]
        critic_params=[p for n,p in model.named_parameters() if critic_names(n)]
        aopt=torch.optim.Adam(actor_params,lr=1e-3);copt=torch.optim.Adam(critic_params,lr=1e-3)
        rows=[];audit_snapshot(model,out,0,args.fold,args.seed)
        for uidx in range(1,args.updates+1):
            b=collect(env,mgr,model,args.fold,args.seed,uidx);card=b["card"]
            assert card in FOLDS[args.fold]["train"] and card!=FOLDS[args.fold]["holdout"]
            adv,ret=gae(b["r"],b["v"],b["nextv"],b["d"])
            B=H*NENV;M=card
            obs=b["obs"].reshape(B,-1);u=b["u"].reshape(B,-1);old=b["oldlp"].reshape(B)
            ids=b["ids"].repeat(H,1);tok=b["tok"].repeat(H,1,1);w=b["w"].repeat(H,1)
            lp=model.logp_from_pre_tanh_from_set(obs,tok,w,u);ratio=torch.exp(lp-old.detach())
            ratio_err=float((ratio-1).abs().max().detach().cpu())
            if ratio_err>1e-4:raise RuntimeError(f"PPO ratio invariant {ratio_err}")
            aa=adv.reshape(B,M).detach();cl=ratio.clamp(1-CLIP,1+CLIP)
            po=torch.minimum(ratio[:,None]*aa,cl[:,None]*aa)
            ppo=-(M*(w*po).sum(-1)).mean()
            tail,tailfrac=tail_metrics(model,obs,tok,w,tau)
            # Retention uses the current rollout's exact active sets only.
            eobs=b["obs"][0];eids=b["ids"]
            retain=edge_loss(model,ref,eobs,eids)
            gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
            gt=torch.autograd.grad(tail,actor_params,retain_graph=True,allow_unused=True)
            gr=torch.autograd.grad(retain,actor_params,allow_unused=True)
            gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
            dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
            coeff=torch.minimum(dot/(gt2+EPS),torch.zeros((),device="cuda"))
            baseg=[];proj2=torch.zeros((),device="cuda")
            for p,a,t in zip(actor_params,gp,gt):
                aa0=torch.zeros_like(p) if a is None else a;tt=torch.zeros_like(p) if t is None else t
                q=aa0-coeff*tt;baseg.append([q,tt]);proj2+=(q.detach()**2).sum()
            projn=torch.sqrt(proj2+EPS);gtn=grad_norm(gt);scale=KAPPA*projn/(gtn+EPS)
            gbase=[q+scale*t for q,t in baseg];gbn=grad_norm(gbase);grn=grad_norm(gr)
            alpha=min(BETA0,RHO*float(gbn.cpu())/(float(grn.cpu())+EPS))
            aopt.zero_grad(set_to_none=True)
            for p,g,r in zip(actor_params,gbase,gr):
                p.grad=g+(alpha*r if r is not None else 0)
            agrad=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).cpu());aopt.step()
            with torch.no_grad():model.log_std.clamp_(-1.6,0.0)
            # Critic is trained only through active token queries.
            cv=model.query_values_from_set(obs.detach(),tok.detach(),w.detach(),tok.detach())
            vl=(cv-ret.reshape(B,M).detach()).pow(2).mean()
            copt.zero_grad(set_to_none=True);vl.backward();cgrad=float(torch.nn.utils.clip_grad_norm_(critic_params,1.0).cpu());copt.step()
            queried=sorted(set(int(x) for x in b["ids"].detach().cpu().reshape(-1).tolist()))
            rec={"update":uidx,"cardinality":card,"allowed_cardinalities":list(FOLDS[args.fold]["train"]),
              "heldout_cardinality":FOLDS[args.fold]["holdout"],"active_ids":b["ids"].detach().cpu().tolist(),
              "weights":b["w"].detach().cpu().tolist(),"modes":b["modes"],"critic_queried_objective_ids":queried,
              "retention_cardinality":card,"ppo_loss":float(ppo.detach().cpu()),"critic_loss":float(vl.detach().cpu()),
              "tail_loss":float(tail.detach().cpu()),"tail_fraction":float(tailfrac.detach().cpu()),
              "retain_loss":float(retain.detach().cpu()),"retain_alpha":float(alpha),
              "weighted_retain_over_base":float(alpha*float(grn.cpu())/(float(gbn.cpu())+EPS)),
              "ratio_maxerr":ratio_err,"termination_fraction":b["termination_fraction"],"actor_grad_preclip":agrad,"critic_grad_preclip":cgrad,
              "leakage_assertion":bool(card in FOLDS[args.fold]["train"] and card!=FOLDS[args.fold]["holdout"])}
            rows.append(rec);print("UPDATE",args.fold,args.seed,uidx,json.dumps({k:rec[k] for k in ("cardinality","ppo_loss","critic_loss","retain_loss","retain_alpha","termination_fraction")}),flush=True)
            if uidx in SNAPS:audit_snapshot(model,out,uidx,args.fold,args.seed)
        leakage=all(r["leakage_assertion"] and r["cardinality"]!=FOLDS[args.fold]["holdout"] for r in rows)
        rep={"schema":"objective_set_g1_train_v1","fold":args.fold,"seed":args.seed,"updates":args.updates,
          "train_cardinalities":list(FOLDS[args.fold]["train"]),"heldout_cardinality":FOLDS[args.fold]["holdout"],
          "leakage_free":leakage,"rows":rows,
          "summary":{"max_ratio_error":max(r["ratio_maxerr"] for r in rows),"max_termination_fraction":max(r["termination_fraction"] for r in rows),
          "mean_retain_budget":float(np.mean([r["weighted_retain_over_base"] for r in rows]))}}
        rp=out/"training_report.json";rp.write_text(json.dumps(rep,indent=2)+"\n")
        (out/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","fold":args.fold,"seed":args.seed,
          "contract_sha256":sha(CONTRACT),"init_sha256":sha(INIT),"script_sha256":sha(Path(__file__).resolve()),
          "model_source_sha256":sha(ROOT/"talon_rl/objective_set_actor_critic.py"),"report_sha256":sha(rp)},indent=2)+"\n")
        print("FINAL",json.dumps(rep["summary"]),flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
