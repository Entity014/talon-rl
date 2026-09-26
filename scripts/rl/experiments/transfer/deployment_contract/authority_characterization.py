"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_phase1_d1_characterize():
    """Run former phase1_d1_characterize.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import itertools,json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h2a_u30_semantic_validity as h2
    import rl.experiments.common.utilities.authority_isolated_simplex_edge_noregression_eval as nr
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SEEDS=(73001,73002,73003)
    ORDER=("T","A","O","S","C")
    HEAVY=("T","A","O","S")
    PAIRS=tuple(itertools.combinations(ORDER,2))
    SUPPORT=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    REF20=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    OUT=ROOT/"runs/phase1_d1_characterization";OUT.mkdir(parents=True,exist_ok=True)
    HELD={"held4":850101,"held5":850202,"held6":850303}
    FRESH=[9700000+i*113 for i in range(4)]
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def stats(xs):
        a=np.asarray(xs,float)
        return {"n":len(a),"mean":float(a.mean()),"std":float(a.std(ddof=1)) if len(a)>1 else 0.0,
                "median":float(np.median(a)),"min":float(a.min()),"max":float(a.max())}
    def edge_geometry(model,ref):
        sd=np.load(SUPPORT);obs=torch.tensor(sd["obs"],device="cuda")
        acts={};refs={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(h2.PREFS[lab],device="cuda").repeat(len(obs),1)
                acts[lab]=model.act_inference_with_preference(obs,w)
                refs[lab]=ref.act_inference_with_preference(obs,w)
        edge={}
        for a,b in PAIRS:
            d=torch.linalg.vector_norm(acts[a]-acts[b],dim=-1)
            r=torch.linalg.vector_norm(refs[a]-refs[b],dim=-1)
            ratio=float(((d*d).mean()/((r*r).mean()+1e-12)).cpu())
            edge[f"{a}-{b}"]={"energy_retention":ratio,"mean_distance":float(d.mean().cpu()),"ref_mean_distance":float(r.mean().cpu())}
        hh=[edge[f"{a}-{b}"]["energy_retention"] for a,b in itertools.combinations(HEAVY,2)]
        hc=[edge[f"{a}-C"]["energy_retention"] for a in HEAVY]
        return {"edges":edge,"heavy_heavy_mean":float(np.mean(hh)),"heavy_center_mean":float(np.mean(hc)),
                "minimum":float(min(x["energy_retention"] for x in edge.values())),
                "all_ge_0_90":bool(min(x["energy_retention"] for x in edge.values())>=.90)}
    
    def endpoint_semantics(env,mgr,robot,model):
        rows=[]
        for suite in range(h2.SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=h2.evaluate(env,model,mgr,robot,h2.PREFS[lab],seed)
                q.update({"suite":suite,"label":lab});rows.append(q)
        out={}
        for lab in h2.ORDER:
            j=h2.IDX[lab];pk=h2.PHYS[lab];oo=[];pp=[];sv=[];do=[];dp=[]
            for suite in range(h2.SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                x=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                y=r["physical"][pk]-c["physical"][pk]
                oo.append(x>0);pp.append(y<0);sv.append(r["survival"]);do.append(x);dp.append(y)
            out[lab]={"objective_correct_fraction":float(np.mean(oo)),"physical_correct_fraction":float(np.mean(pp)),
                      "mean_objective_delta_vs_center":float(np.mean(do)),"mean_physical_delta_vs_center":float(np.mean(dp)),
                      "min_survival":float(np.min(sv)),
                      "pass":bool(np.mean(oo)>=.75 and np.mean(pp)>=.75 and np.min(sv)>=.95)}
        return rows,out
    def continuum(env,mgr,robot,model):
        rows=[]
        for pi,(a,b) in enumerate(h2.PATHS):
            for suite in range(h2.SUITES):
                seed=850001+pi*1000+suite
                for alpha in h2.ALPHAS:
                    w=h2.interp(a,b,alpha);q=h2.evaluate(env,model,mgr,robot,w,seed)
                    q.update({"path":f"{a}-{b}","a":a,"b":b,"suite":suite,"alpha":alpha});rows.append(q)
        cells=[]
        for a,b in h2.PATHS:
            for suite in range(h2.SUITES):
                rr=sorted([x for x in rows if x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                for lab in (a,b):
                    j=h2.IDX[lab];pk=h2.PHYS[lab];ov=[x["normalized_objective_mean"][j] for x in rr];pv=[x["physical"][pk] for x in rr]
                    cells.append({"path":f"{a}-{b}","suite":suite,"axis":lab,
                      "objective_monotonic":h2.monotonic_fraction(ov),"objective_between":h2.between_fraction(ov),
                      "physical_monotonic":h2.monotonic_fraction(pv),"physical_between":h2.between_fraction(pv)})
        mono=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cells]))
        between=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cells]))
        s_cells=[x for x in cells if "S" in x["path"]];ns=[x for x in cells if "S" not in x["path"]]
        def sm(q):return {"monotonicity":float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in q])),
                          "between":float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in q]))}
        min_surv=float(min(x["survival"] for x in rows))
        return rows,{"monotonicity":mono,"between":between,"S_paths":sm(s_cells),"nonS_paths":sm(ns),"min_survival":min_surv}
    
    def center_compromise(endpoint_rows):
        vals=[]
        for suite in range(h2.SUITES):
            hs=[next(x for x in endpoint_rows if x["suite"]==suite and x["label"]==lab) for lab in h2.ORDER]
            c=next(x for x in endpoint_rows if x["suite"]==suite and x["label"]=="C")
            for j in range(4):
                v=[x["normalized_objective_mean"][j] for x in hs]
                vals.append(min(v)-1e-9<=c["normalized_objective_mean"][j]<=max(v)+1e-9)
        return float(np.mean(vals))
    def engineering(env,model):
        held=[];fresh=[]
        for suite,seed in HELD.items():
            for lab in ORDER:
                q=nr.rollout(env,model,h2.PREFS[lab],seed);q.update({"suite":suite,"preference":lab});held.append(q)
        for li,lab in enumerate(ORDER):
            for si,base in enumerate(FRESH):
                seed=base+li*1000
                q=nr.rollout(env,model,h2.PREFS[lab],seed);q.update({"suite":si,"preference":lab,"seed":seed});fresh.append(q)
        hs=nr.summarize(held);fs=nr.summarize(fresh)
        gate=bool(fs["h32_ev_mean"]>0 and fs["mc64_ev_mean"]>0 and fs["h32_negative_fraction"]<=.25 and
                  fs["mc64_negative_fraction"]<=.25 and fs["min_survival"]>=.95 and hs["min_survival"]>=.95)
        return {"heldout":hs,"fresh":fs,"critic_robustness_gate":gate,"heldout_rows":held,"fresh_rows":fresh}
    
    def seed_report(env,mgr,robot,ref,seed):
        run=ROOT/f"runs/phase1_d1_seed_{seed}"
        model=AuthorityIsolatedWideCritic(48,12).cuda()
        model.load_state_dict(torch.load(run/"model_30.pt",map_location="cuda",weights_only=False)["model"]);model.eval()
        train=json.load(open(run/"training_report.json"))
        eprows,ep=endpoint_semantics(env,mgr,robot,model)
        crows,cont=continuum(env,mgr,robot,model)
        center=center_compromise(eprows);edge=edge_geometry(model,ref);eng=engineering(env,model)
        auth={"pairwise_retention":train["summary"]["fixed_probe_pairwise_retention"],
              "tangent_retention":train["summary"]["fixed_probe_tangent_retention"],
              "all_edge_min_retention":edge["minimum"],"all_edge_gate":edge["all_ge_0_90"],
              "pass":bool(train["summary"]["fixed_probe_pairwise_retention"]>=.90 and train["summary"]["fixed_probe_tangent_retention"]>=.90 and edge["all_ge_0_90"])}
        critic_ok=eng["critic_robustness_gate"]
        if ep["S"]["pass"]:scat="S-valid"
        elif ep["S"]["min_survival"]>=.95 and critic_ok:scat="S-semantic-inconsistent"
        else:scat="S-engineering-confounded"
        return {"seed":seed,"authority":auth,"edge_geometry":edge,"endpoint":ep,
                "center_compromise_fraction":center,"continuum":cont,"engineering":eng,
                "ppo_ratio_maxerr":train["summary"]["max_ratio_error"],
                "training_max_termination_fraction":train["summary"]["max_termination_fraction"],
                "S_category":scat}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=h2.NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            ref=AuthorityIsolatedWideCritic(o.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda()
            ref.load_state_dict(torch.load(REF20,map_location="cuda",weights_only=False)["model"]);ref.eval()
            reports=[]
            for seed in SEEDS:
                q=seed_report(env,mgr,robot,ref,seed);reports.append(q)
                (ROOT/f"runs/phase1_d1_seed_{seed}/characterization.json").write_text(json.dumps(q,indent=2)+"\n")
                print("SEED",seed,"AUTH",q["authority"]["pass"],"EP",{k:v["pass"] for k,v in q["endpoint"].items()},
                      "S",q["S_category"],"CRIT",q["engineering"]["critic_robustness_gate"],flush=True)
            props={}
            for name,fn in {
              "authority":lambda q:q["authority"]["pass"],
              "T_semantics":lambda q:q["endpoint"]["T"]["pass"],
              "A_semantics":lambda q:q["endpoint"]["A"]["pass"],
              "O_semantics":lambda q:q["endpoint"]["O"]["pass"],
              "critic_robustness":lambda q:q["engineering"]["critic_robustness_gate"],
              "heldout_robustness":lambda q:q["engineering"]["heldout"]["min_survival"]>=.95}.items():
                n=sum(bool(fn(q)) for q in reports)
                props[name]={"pass_count":n,"label":"strongly reproducible" if n==3 else ("partially reproducible / seed-sensitive" if n==2 else "not robustly reproducible")}
            continuous={}
            extract={
              "pairwise_retention":lambda q:q["authority"]["pairwise_retention"],
              "tangent_retention":lambda q:q["authority"]["tangent_retention"],
              "minimum_edge_retention":lambda q:q["authority"]["all_edge_min_retention"],
              "continuum_monotonicity":lambda q:q["continuum"]["monotonicity"],
              "continuum_between":lambda q:q["continuum"]["between"],
              "center_compromise":lambda q:q["center_compromise_fraction"],
              "fresh_h32_ev":lambda q:q["engineering"]["fresh"]["h32_ev_mean"],
              "fresh_mc64_ev":lambda q:q["engineering"]["fresh"]["mc64_ev_mean"] }
            for k,fn in extract.items():continuous[k]=stats([fn(q) for q in reports])
            agg={"schema":"phase1_d1_multiseed_characterization_v1","seeds":reports,"property_reproducibility":props,
                 "S_categories":{str(q["seed"]):q["S_category"] for q in reports},"continuous_summary":continuous}
            (OUT/"multiseed_characterization.json").write_text(json.dumps(agg,indent=2)+"\n")
            print("FINAL",json.dumps({"properties":props,"S":agg["S_categories"],"continuous":continuous},indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_phase1_d1_formal_authority():
    """Run former phase1_d1_formal_authority.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import itertools,json,sys
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SUP=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    REF=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    OUT=ROOT/"runs/phase1_d1_characterization/formal_authority_multiseed.json"
    
    def load(path):
        s=torch.load(path,map_location="cuda",weights_only=False)["model"]
        m=AuthorityIsolatedWideCritic(48,12).cuda();m.load_state_dict(s);m.eval();return m
    
    def actions(m,x):
        d={}
        with torch.no_grad():
            for lab in h1.ORDER:
                w=torch.tensor(h1.PREFS[lab],device="cuda").repeat(len(x),1)
                d[lab]=m.act_inference_with_preference(x,w)
        return d
    def main():
        d=np.load(SUP);X=torch.tensor(d["obs"],device="cuda");origin=d["origin"];phase=d["phase"]
        rng=np.random.default_rng(2609252701);ids=[]
        for oi in range(5):
            for pi in range(3):
                z=np.flatnonzero((origin==oi)&(phase==pi))
                ids.extend(rng.choice(z,size=25,replace=False).tolist())
        xx=X[np.asarray(ids)]
        u20=load(REF);base=h1.sensitivity(u20,xx);A0=actions(u20,X)
        rep={}
        for seed in (73001,73002,73003):
            m=load(ROOT/f"runs/phase1_d1_seed_{seed}/model_30.pt")
            q=h1.sensitivity(m,xx);A=actions(m,X)
            pair=q["pairwise_action_distance"]["mean"]/(base["pairwise_action_distance"]["mean"]+1e-12)
            tan=q["tangent_jacobian_fro_mean"]/(base["tangent_jacobian_fro_mean"]+1e-12)
            edges={}
            for i,j in itertools.combinations(h1.ORDER,2):
                rr=torch.linalg.vector_norm(A0[i]-A0[j],dim=1)
                qq=torch.linalg.vector_norm(A[i]-A[j],dim=1)
                edges[f"{i}-{j}"]=float(torch.sqrt((qq.pow(2).sum()+1e-12)/(rr.pow(2).sum()+1e-12)).cpu())
            mn=min(edges,key=edges.get)
            rep[str(seed)]={"pairwise_retention":pair,"tangent_retention":tan,
                "min_edge":edges[mn],"min_edge_name":mn,
                "authority_gate":bool(pair>=.9 and tan>=.9),
                "edge_gate":bool(edges[mn]>=.9),"edges":edges}
            print(seed,json.dumps({k:v for k,v in rep[str(seed)].items() if k!="edges"}),flush=True)
        OUT.write_text(json.dumps(rep,indent=2)+"\n")
    
    if True:main()

def run_phase1_d1_isolated_runner():
    """Run former phase1_d1_isolated_runner.py stage."""
    from pathlib import Path
    import argparse,hashlib,json,os,platform,shutil,subprocess,sys,tempfile
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    CANON_SCRIPT=ROOT/"scripts/rl/experiments/architectures/authority/isolated/authority_diagnostics.py"
    CANON_SHA="a23498c37ac7d398b0a35eaea47a4f3953e265e2a0f55f362883d7f469de410b"
    SOURCE_STATE=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt"
    SOURCE_MODEL=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    SOURCE_STATE_SHA=None
    SOURCE_MODEL_SHA=None
    SOURCE_DIRS=(
        "authority_isolated_coverage2_control-2026-09-25",
        "authority_isolated_u20_authority_support-2026-09-25",
        "authority_isolated_tail_budget_calibration-2026-09-25",
        "update_functional_effect_audit-2026-09-24",
    )
    ALLOWED_SEEDS=(73001,73002,73003)
    CANON_OUT=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25"
    
    def sha(p:Path)->str:
        h=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(1<<20),b""):h.update(b)
        return h.hexdigest()
    def git_commit():
        return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    
    def assert_sources():
        got=sha(CANON_SCRIPT)
        if got!=CANON_SHA:
            raise RuntimeError(f"canonical script hash mismatch: {got}")
        if not SOURCE_STATE.exists() or not SOURCE_MODEL.exists():
            raise FileNotFoundError("validated u20 source missing")
        if not CANON_OUT.exists():
            raise FileNotFoundError("canonical u30 artifact directory missing")
        return {"canonical_script_sha256":got,
                "source_state_sha256":sha(SOURCE_STATE),
                "source_model_sha256":sha(SOURCE_MODEL)}
    
    def make_overlay(seed:int,outdir:Path):
        base=ROOT/".d1_overlay"/f"seed_{seed}"
        if base.exists():shutil.rmtree(base)
        base.mkdir(parents=True)
        shutil.copytree(ROOT/"scripts",base/"scripts")
        copied=base/"scripts/rl/experiments/architectures/authority/isolated/authority_diagnostics.py"
        if sha(copied)!=CANON_SHA:raise RuntimeError("copied canonical script differs")
        os.symlink(ROOT/"talon_rl",base/"talon_rl",target_is_directory=True)
        runs=base/"runs";runs.mkdir()
        for name in SOURCE_DIRS:
            src=ROOT/"runs"/name
            if not src.exists():raise FileNotFoundError(src)
            os.symlink(src,runs/name,target_is_directory=True)
        expected=runs/"authority_isolated_simplex_edge_retain-2026-09-25"
        os.symlink(outdir,expected,target_is_directory=True)
        return base,copied
    def write_config(outdir,seed,mode,src_hashes,overlay):
        cfg={
          "schema":"phase1_d1_isolated_runner_v1",
          "seed":seed,"mode":mode,
          "start_update":20,
          "target_update":21 if mode=="smoke" else 30,
          "canonical_training_script":str(CANON_SCRIPT.relative_to(ROOT)),
          "canonical_script_sha256":src_hashes["canonical_script_sha256"],
          "source_state":str(SOURCE_STATE.relative_to(ROOT)),
          "source_state_sha256":src_hashes["source_state_sha256"],
          "source_model":str(SOURCE_MODEL.relative_to(ROOT)),
          "source_model_sha256":src_hashes["source_model_sha256"],
          "git_commit":git_commit(),
          "python":sys.version.split()[0],
          "torch":torch.__version__,
          "cuda":torch.version.cuda,
          "platform":platform.platform(),
          "overlay":str(overlay),
          "canonical_output_path":str(CANON_OUT),
          "canonical_output_sha256_before":sha(CANON_OUT/"model_30.pt"),
        }
        (outdir/"run_config.json").write_text(json.dumps(cfg,indent=2)+"\\n")
        return cfg
    
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--seed",type=int,choices=ALLOWED_SEEDS,required=True)
        ap.add_argument("--mode",choices=("smoke","full"),default="full")
        ap.add_argument("--output-dir")
        args=ap.parse_args()
        src_hashes=assert_sources()
        name=args.output_dir or (f"phase1_d1_smoke_seed_{args.seed}" if args.mode=="smoke" else f"phase1_d1_seed_{args.seed}")
        outdir=ROOT/"runs"/name
        if outdir.resolve()==CANON_OUT.resolve():raise RuntimeError("refusing canonical output")
        outdir.mkdir(parents=True,exist_ok=True)
        if any(outdir.iterdir()):
            raise RuntimeError(f"output directory must be empty for isolated D1 run: {outdir}")
        overlay,copied=make_overlay(args.seed,outdir)
        cfg=write_config(outdir,args.seed,args.mode,src_hashes,overlay)
        target=21 if args.mode=="smoke" else 30
        cmd=[sys.executable,str(copied),"--arm","retain","--seed",str(args.seed),"--target-updates",str(target)]
        env=os.environ.copy()
        env["PYTHONPATH"]=f"{overlay}:{overlay/'scripts'}"
        proc=subprocess.run(cmd,cwd=overlay,env=env)
        cfg["returncode"]=proc.returncode
        cfg["canonical_output_sha256_after"]=sha(CANON_OUT/"model_30.pt")
        cfg["canonical_unchanged"]=cfg["canonical_output_sha256_after"]==cfg["canonical_output_sha256_before"]
        cfg["expected_model_exists"]=(outdir/f"model_{target}.pt").exists()
        cfg["resume_state_exists"]=(outdir/"resume_state.pt").exists()
        cfg["training_report_exists"]=(outdir/"training_report.json").exists()
        cfg["status"]="COMPLETE" if proc.returncode==0 and cfg["canonical_unchanged"] and cfg["expected_model_exists"] else "INVALID_FAILED_RUN"
        (outdir/"run_config.json").write_text(json.dumps(cfg,indent=2)+"\\n")
        if not cfg["canonical_unchanged"]:
            raise RuntimeError("canonical artifact changed during D1 run")
        if proc.returncode!=0:
            raise SystemExit(proc.returncode)
        if not cfg["expected_model_exists"]:
            raise RuntimeError("expected fixed endpoint missing")
        print(json.dumps({k:cfg[k] for k in ("seed","mode","status","canonical_unchanged","expected_model_exists")},indent=2))
    
    if True:main()

STAGES = {
    "phase1_d1_characterize": run_phase1_d1_characterize,
    "phase1_d1_formal_authority": run_phase1_d1_formal_authority,
    "phase1_d1_isolated_runner": run_phase1_d1_isolated_runner,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
