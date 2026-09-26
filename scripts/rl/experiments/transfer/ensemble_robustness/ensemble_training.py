from __future__ import annotations

# scripts/ on sys.path so the absolute rl.experiments.* imports below
# resolve when this file is run directly, as these scripts always are.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pathlib import Path
import argparse,json,sys
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
from talon_rl.wrappers.plant_ensemble import Phase5PlantEnsembleWrapper

E0_MANIFEST=ROOT/"artifacts/phase5_e0_plant_ensemble_manifest.json"

SOURCE_STATE=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt"
SOURCE_MODEL=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
SUPPORT=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
NENV=h1.NENV;H=h1.H;KAPPA=.05;RHO=.25;BETA0=2.497041993384243;EPS=1e-12
HEAVY=("T","A","O","S")

def tail_metrics(m,obs,w,tau):
    z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
    return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()

def audit(m,probe,tau,out,tag,snaps,arm):
    m.eval();sens=h1.sensitivity(m,probe)
    P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
    wm=torch.tensor(h1.PREFS["C"],device="cuda").repeat(len(P),1)
    with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
    snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
    torch.save({"model":m.state_dict(),"update":int(tag),"arm":arm},out/f"model_{tag}.pt");m.train()

def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update):
    torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
      "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
      "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state()},path)

def balanced_indices(origin,phase,update):
    rng=np.random.default_rng(2609252401+update);ids=[]
    for oi in range(5):
        for pi in range(3):
            pool=np.flatnonzero((origin==oi)&(phase==pi))
            ids.extend(rng.choice(pool,size=8,replace=False).tolist())
    return np.asarray(ids,np.int64)

ALL_PREFS=("T","A","O","S","C")
EDGE_PAIRS=tuple((ALL_PREFS[i],ALL_PREFS[j]) for i in range(len(ALL_PREFS)) for j in range(i+1,len(ALL_PREFS)))
GAMMA=.90

def edge_floor_loss(m,ref,obs):
    n=len(obs)
    acts={};refs={}
    for lab in ALL_PREFS:
        w=torch.tensor(h1.PREFS[lab],device="cuda").repeat(n,1)
        acts[lab]=m.act_inference_with_preference(obs,w)
        with torch.no_grad():
            refs[lab]=ref.act_inference_with_preference(obs,w)
    terms=[]
    for i,j in EDGE_PAIRS:
        d=torch.linalg.vector_norm(acts[i]-acts[j],dim=1)
        dr=torch.linalg.vector_norm(refs[i]-refs[j],dim=1)
        terms.append(torch.relu(GAMMA*dr-d).pow(2))
    return torch.stack(terms,dim=1).mean()

def grad_norm(gs):
    return torch.sqrt(sum((g.detach()**2).sum() for g in gs if g is not None)+EPS)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["control","retain"],default="retain")
    ap.add_argument("--plant-arm",choices=["control","ensemble"],required=True)
    ap.add_argument("--seed",type=int,default=75001);ap.add_argument("--target-updates",type=int,default=30)
    args=ap.parse_args()
    if args.arm != "retain": raise RuntimeError("Phase5 E2 freezes simplex-edge retention ON")
    out=ROOT/f"runs/phase5_e2_{args.plant_arm}_seed{args.seed}";out.mkdir(parents=True,exist_ok=True)
    state_path=out/"resume_state.pt";tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
    sd=np.load(SUPPORT);ref_obs=torch.tensor(sd["obs"],device="cuda");origin=sd["origin"];phase=sd["phase"]
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
        if args.plant_arm=="ensemble": cfg.events.add_base_mass=None
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        if args.plant_arm=="ensemble":
            env=Phase5PlantEnsembleWrapper(env,E0_MANIFEST,out/"realized_plant_tuples.jsonl")
        o,_=env.reset(seed=args.seed);o=h1.ot(o).cuda();probe=o.detach().clone()
        ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
        m=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda()
        ref=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();ref.load_state_dict(torch.load(SOURCE_MODEL,map_location="cuda",weights_only=False)["model"]);ref.eval()
        for p in ref.parameters():p.requires_grad_(False)
        for n,p in m.named_parameters():
            if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
        actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
        opt=torch.optim.Adam(actor_params,lr=1e-3)

        if state_path.exists():
            st=torch.load(state_path,map_location="cpu",weights_only=False);m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
            adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
            torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
        else:
            st=torch.load(SOURCE_STATE,map_location="cpu",weights_only=False);m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
            adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=[];snaps={"20":st["snaps"]["20"]};cur=20
            audit(m,probe,tau,out,20,snaps,args.arm);save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur)

        def current_anchor_units():
            q={"early":[],"late":[]}
            for k,seed in anchor_specs:
                _,w=h1.pref_batch(k,torch.device("cuda"))
                for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
            return q

        for uidx in range(cur+1,args.target_updates+1):
            _,w=h1.pref_batch(uidx,torch.device("cuda"))
            main=h1.collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
            _,ws=h1.pref_batch(uidx+17,torch.device("cuda"));units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
            for u in units:
                ph=u["phase"];adaptive_pools[ph].append(u)
                if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
            h1.fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
            with torch.no_grad():
                vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4);nv=m.value_with_preference(main["next_obs"],w)
                adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
            lp=[]
            for stx in range(0,len(main["obs"]),NENV):
                lp.append(m.logp_from_pre_tanh_with_preference(main["obs"][stx:stx+NENV],main["w"][stx:stx+NENV],main["u"][stx:stx+NENV]))
            ratio=torch.exp(torch.cat(lp)-main["old"].detach());ratio_err=float((ratio-1).abs().max().detach().cpu())
            if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
            ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
            tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
            gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
            gt=torch.autograd.grad(tail,actor_params,retain_graph=(args.arm=="retain"),allow_unused=True)
            gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
            gpn=torch.sqrt(gp2+EPS);gtn=torch.sqrt(gt2+EPS)
            dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
            coeff=torch.minimum(dot/(gt2+EPS),torch.zeros_like(dot));base=[];proj2=torch.zeros((),device="cuda")
            for p,a,b in zip(actor_params,gp,gt):
                aa=torch.zeros_like(p) if a is None else a;bb=torch.zeros_like(p) if b is None else b
                pp=aa-coeff*bb;base.append([pp,bb]);proj2+=(pp.detach()**2).sum()
            projn=torch.sqrt(proj2+EPS);tail_scale=KAPPA*projn/(gtn+EPS)
            gbase=[pp+tail_scale*bb for pp,bb in base];gbn=grad_norm(gbase)

            retain=torch.zeros((),device="cuda");gr=[None]*len(actor_params);alpha=0.;grn=torch.zeros((),device="cuda")
            if args.arm=="retain":
                ids=balanced_indices(origin,phase,uidx);retain=edge_floor_loss(m,ref,ref_obs[ids])
                gr=torch.autograd.grad(retain,actor_params,allow_unused=True);grn=grad_norm(gr)
                alpha=min(BETA0,RHO*float(gbn.detach().cpu())/(float(grn.detach().cpu())+EPS))
            opt.zero_grad(set_to_none=True)
            for i,p in enumerate(actor_params):
                gg=gbase[i]
                if args.arm=="retain" and gr[i] is not None:gg=gg+alpha*gr[i]
                p.grad=gg
            preclip=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
            with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
            row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),"tail_fraction":float(tailfrac.detach().cpu()),
                 "retain_loss":float(retain.detach().cpu()),"retain_alpha":float(alpha),"base_grad_norm":float(gbn.cpu()),"retain_grad_norm":float(grn.cpu()),
                 "weighted_retain_over_base":float(alpha*float(grn.cpu())/(float(gbn.cpu())+EPS)) if args.arm=="retain" else 0.,
                 "ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"],"grad_norm_preclip":preclip}
            rows.append(row);print("UPDATE",args.arm,uidx,json.dumps(row),flush=True)
            if uidx==args.target_updates:audit(m,probe,tau,out,uidx,snaps,args.arm)
            save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx)

        s0=snaps["20"]["sensitivity"];sf=snaps[str(args.target_updates)]["sensitivity"]
        rep={"schema":"phase5_e2_training_v1","arm":args.arm,"plant_arm":args.plant_arm,"seed":args.seed,"rho":RHO,"beta0":BETA0,"gamma":GAMMA,"updates":args.target_updates,
             "summary":{"fixed_probe_pairwise_retention":sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+EPS),
                        "fixed_probe_tangent_retention":sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+EPS),
                        "max_ratio_error":max(r["ratio_maxerr"] for r in rows),"max_termination_fraction":max(r["termination_fraction"] for r in rows),
                        "mean_weighted_retain_over_base":float(np.mean([r["weighted_retain_over_base"] for r in rows]))},
             "rows":rows,"snapshots":snaps}
        (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",args.arm,json.dumps(rep["summary"]),flush=True)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
