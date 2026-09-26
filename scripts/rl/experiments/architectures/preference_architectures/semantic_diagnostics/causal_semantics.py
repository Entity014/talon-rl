"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def _exec_stage_preamble(stage, namespace):
    """Execute only the pre-main definitions of another consolidated stage."""
    import ast, inspect, textwrap
    fn = globals()["run_" + stage]
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = tree.body[0]
    body = []
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "main":
            break
        if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant) and isinstance(stmt.value.value, str):
            continue
        body.append(stmt)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, __file__, "exec"), namespace)

def run_post_v2_t5_c20_collect_pool():
    """Run former post_v2_t5_c20_collect_pool.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/post_v2_t5_c20_coverage-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S");G=.99;H=32;NTRAIN=16;NTEST=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def mc(R,D):
     out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=16;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      meta={"schema":"c20_pool_v1","horizon":H,"ntrain":NTRAIN,"ntest":NTEST,"num_envs":16,"specialists":{}}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(od,ad).cuda();initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero");m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(16,1)
       arrays={};rows=[]
       for split,nroll,base_seed in (("train",NTRAIN,710000+bi*10000),("test",NTEST,910000+bi*10000)):
        splitrows=[]
        for r in range(nroll):
         cur,_=env.reset(seed=base_seed+r*97);cur=obs_tensor(cur).cuda();obs=[];R=[];D=[];cmds=[]
         with torch.no_grad():
          for _ in range(H):
           obs.append(cur);cmds.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
           a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           vec=normalized_objective_vector(terms(raw,names),shape=(16,))
           R.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);D.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
         X=torch.cat(obs);Wp=w.repeat(H,1);Y=mc(torch.stack(R),torch.stack(D).bool()).reshape(-1,4)
         with torch.no_grad():F=m.critic_body(m._with_w(X,Wp))
         C=np.concatenate(cmds,0)
         arrays[f"{split}_F{r}"]=F.cpu().numpy();arrays[f"{split}_Y{r}"]=Y.cpu().numpy()
         summary=np.r_[C.mean(0),C.std(0),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),Y.mean(0).cpu().numpy(),Y.std(0).cpu().numpy()]
         arrays[f"{split}_S{r}"]=summary.astype(np.float32)
         splitrows.append({"rollout":r,"seed":base_seed+r*97,"command_mean":C.mean(0).tolist(),"command_std":C.std(0).tolist(),
                           "feature_mean_norm":float(F.mean(0).norm()),"feature_std_norm":float(F.std(0).norm()),
                           "target_mean":Y.mean(0).cpu().tolist(),"target_std":Y.std(0).cpu().tolist()})
        rows.append((split,splitrows))
       np.savez_compressed(OUT/f"{lab}_pool.npz",**arrays)
       meta["specialists"][lab]={k:v for k,v in rows}
       print(lab,"saved",flush=True)
      (OUT/"pool_meta.json").write_text(json.dumps(meta,indent=2)+"\n")
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c20_coverage_analyze():
    """Run former post_v2_t5_c20_coverage_analyze.py stage."""
    from pathlib import Path
    import json,numpy as np
    ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/"runs/post_v2_t5_c20_coverage-2026-09-23"
    ORDER=("T","A","O","S");RIDGE=1.0
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def fit(F,Y):
     A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0;sol=np.linalg.solve(A.T@A+RIDGE*I,A.T@Y);return sol[:-1].T,sol[-1]
    def evaluate(W,b,items):
     es=[];ms=[]
     for F,Y in items:
      P=F@W.T+b
      es += [ev(Y[:,j],P[:,j]) for j in range(4)]
      ms += [float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]
     return {"ev_mean":float(np.mean(es)),"ev_negative_fraction":float(np.mean(np.array(es)<0)),"mse_mean":float(np.mean(ms)),
             "ev_by_head":[float(np.mean(es[j::4])) for j in range(4)]}
    def select_diverse(S,k):
     # use command+feature mean/std only, excluding final 8 target summary coordinates
     X=S[:,:-8].astype(float)
     X=(X-X.mean(0))/(X.std(0)+1e-6)
     # deterministic farthest point: start farthest from centroid
     d0=np.sum(X*X,1);sel=[int(np.argmax(d0))]
     mind=np.sum((X-X[sel[0]])**2,1)
     while len(sel)<k:
      mind[sel]=-1
      q=int(np.argmax(mind));sel.append(q)
      mind=np.minimum(mind,np.sum((X-X[q])**2,1))
     return sorted(sel)
    def summary_distance(trainS,testS):
     # report standardized nearest-summary distance; command-only and feature-only
     out={}
     for name,sl in [("command",slice(0,6)),("feature",slice(6,-8)),("target",slice(-8,None))]:
      A=trainS[:,sl].astype(float);B=testS[:,sl].astype(float);mu=A.mean(0);sd=A.std(0)+1e-6;Az=(A-mu)/sd;Bz=(B-mu)/sd
      ds=[]
      for b in Bz: ds.append(float(np.sqrt(np.min(np.mean((Az-b)**2,axis=1)))))
      out[name+"_nn_std_distance_mean"]=float(np.mean(ds))
      out[name+"_nn_std_distance_max"]=float(np.max(ds))
     return out
    report={"schema":"c20_coverage_audit_v1","specialists":{}}
    for lab in ORDER:
     z=np.load(OUT/f"{lab}_pool.npz")
     train=[(z[f"train_F{i}"].astype(float),z[f"train_Y{i}"].astype(float)) for i in range(16)]
     test=[(z[f"test_F{i}"].astype(float),z[f"test_Y{i}"].astype(float)) for i in range(8)]
     S=np.stack([z[f"train_S{i}"] for i in range(16)]);St=np.stack([z[f"test_S{i}"] for i in range(8)])
     supports={
      "recent3":list(range(13,16)),
      "recent6":list(range(10,16)),
      "recent12":list(range(4,16)),
      "all16":list(range(16)),
      "diverse6":select_diverse(S,6),
      "diverse12":select_diverse(S,12)
     }
     q={}
     for name,idx in supports.items():
      F=np.concatenate([train[i][0] for i in idx]);Y=np.concatenate([train[i][1] for i in idx]);W,b=fit(F,Y)
      q[name]={"indices":idx,"train":evaluate(W,b,[train[i] for i in idx]),"fresh":evaluate(W,b,test),
               "weight_norm":float(np.linalg.norm(W)),"bias_norm":float(np.linalg.norm(b)),
               "coverage":summary_distance(S[idx],St)}
     # rolling solution drift for temporal windows
     for win in (3,6,12):
      sols=[]
      for end in range(win,17):
       idx=list(range(end-win,end));F=np.concatenate([train[i][0] for i in idx]);Y=np.concatenate([train[i][1] for i in idx]);W,b=fit(F,Y);sols.append(np.r_[W.reshape(-1),b])
      dr=[np.linalg.norm(sols[i]-sols[i-1]) for i in range(1,len(sols))]
      q[f"recent{win}"]["rolling_solution_drift_mean"]=float(np.mean(dr)) if dr else 0.0
      q[f"recent{win}"]["rolling_solution_drift_p90"]=float(np.quantile(dr,.9)) if dr else 0.0
     report["specialists"][lab]=q
    agg={}
    for name in ("recent3","recent6","recent12","all16","diverse6","diverse12"):
     fresh=[];train_ev=[];wn=[];cmd=[];feat=[];targ=[];by=[[] for _ in range(4)]
     drift=[]
     for lab in ORDER:
      q=report["specialists"][lab][name];fresh.append(q["fresh"]["ev_mean"]);train_ev.append(q["train"]["ev_mean"]);wn.append(q["weight_norm"])
      cmd.append(q["coverage"]["command_nn_std_distance_mean"]);feat.append(q["coverage"]["feature_nn_std_distance_mean"]);targ.append(q["coverage"]["target_nn_std_distance_mean"])
      for j,x in enumerate(q["fresh"]["ev_by_head"]):by[j].append(x)
      if "rolling_solution_drift_mean" in q:drift.append(q["rolling_solution_drift_mean"])
     agg[name]={"train_ev_mean":float(np.mean(train_ev)),"fresh_ev_mean":float(np.mean(fresh)),
                "fresh_ev_by_head":[float(np.mean(x)) for x in by],
                "weight_norm_mean":float(np.mean(wn)),
                "command_nn_distance_mean":float(np.mean(cmd)),"feature_nn_distance_mean":float(np.mean(feat)),"target_nn_distance_mean":float(np.mean(targ))}
     if drift:agg[name]["rolling_solution_drift_mean"]=float(np.mean(drift))
    report["aggregate"]=agg
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(agg,indent=2))

def run_post_v2_t5_c20_mc64_eval():
    """Run former post_v2_t5_c20_mc64_eval.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    OUT=ROOT/'runs/post_v2_t5_c20_coverage-2026-09-23';ORDER=('T','A','O','S');SUP=('recent3','recent6','recent12','all16','diverse12');G=.99
    PREFS={'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D):
     out=np.zeros_like(R);run=np.zeros_like(R[0])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({'headless':True,'enable_cameras':False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=16;cfg.seed=0;env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      sol=np.load(OUT/'head_solutions.npz');res={}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(od,ad).cuda();initialize_from_rsl_m01(m,ROOT/'runs/m0_1_seed0_2026-09-22/model_299.pt',device='cpu',critic_head_init='zero');m.eval();w=torch.tensor(PREFS[lab],device='cuda').repeat(16,1)
       # collect one fresh 64 rollout and body features/policy actions once
       cur,_=env.reset(seed=990000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];F=[]
       with torch.no_grad():
        for _ in range(64):
         F.append(m.critic_body(m._with_w(cur,w)).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         R.append(normalized_objective_vector(terms(raw,names),shape=(16,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
       R=np.asarray(R);D=np.asarray(D,bool);F=np.asarray(F);MC=ret(R,D);q={}
       for s in SUP:
        W=sol[f'{lab}_{s}_W'];b=sol[f'{lab}_{s}_b'];P=np.einsum('ted,hd->teh',F,W)+b
        q[s]={'mc64_ev':[ev(MC[:,:,j],P[:,:,j]) for j in range(4)],'mc64_bias':[float(np.mean(P[:,:,j]-MC[:,:,j])) for j in range(4)]}
       res[lab]=q
      agg={}
      for s in SUP:
       e=[];b=[]
       for lab in ORDER:e+=res[lab][s]['mc64_ev'];b+=res[lab][s]['mc64_bias']
       agg[s]={'mc64_ev_mean':float(np.mean(e)),'mc64_negative_fraction':float(np.mean(np.array(e)<0)),'mc64_mean_abs_bias':float(np.mean(np.abs(b)))}
      out={'schema':'c20_mc64_eval_v1','specialists':res,'aggregate':agg};(OUT/'mc64_eval.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c21_replay_audit():
    """Run former post_v2_t5_c21_replay_audit.py stage."""
    """Value replay for C21 recent3 versus recent12 support."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.replay_audit import ReplayAudit
    
    
    class C21ReplayAudit(ReplayAudit):
        """Value replay for C21 recent3 versus recent12 support."""
    
        arms = {"recent3": "post_v2_t5_c21_recent3-2026-09-23",
                "recent12": "post_v2_t5_c21_recent12-2026-09-23"}
        schema = "t5_c21_replay_audit_v1"
        seed_base = 1210000
        run = "post_v2_t5_c21_recent12-2026-09-23"
    
    
    if True:
        C21ReplayAudit.main()

def run_post_v2_t5_c21_support_width_train():
    """Run former post_v2_t5_c21_support_width_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    from collections import deque
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2):
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--actor-update",choices=("on","off"),required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--recent-window",type=int,default=3)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd");env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(51000+idx);np.random.seed(51000+idx)
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            for n,p in m.named_parameters():
                if n.startswith("critic_body"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
            w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            recent=deque(maxlen=args.recent_window)
            cur,_=env.reset(seed=510001+idx*1000);cur=obs_tensor(cur).cuda()
            logs=[]
            def save_snap(tag):
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"actor_update":args.actor_update},
                           args.output.parent/f"{label}_snap_{tag}.pt")
            save_snap(0)
            prev_F=None;prev_Y=None;prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
            prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params])
            for update in range(1,args.updates+1):
                ob=[];pre=[];old=[];rw=[];dn=[];acts=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
                    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());acts.append(a)
                    cur=obs_tensor(nxt).cuda()
                fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1);fa=torch.cat(acts)
                rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
                recent.append((F.detach().cpu(),Y.detach().cpu()))
                with torch.no_grad():pre=m.critic_head(F);pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
                FF=torch.cat([x[0] for x in recent],0).cuda();YY=torch.cat([x[1] for x in recent],0).cuda()
                Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit);post=m.critic_head(F)
                post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
                # drift metrics before optional actor update
                Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
                feat_mean=F.mean(0);feat_std=F.std(0);tar_mean=Y.mean(0);tar_std=Y.std(0);act_mean=fa.mean(0);act_std=fa.std(0)
                row={"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"buffer_len":len(recent),
                     "feature_mean_norm":float(feat_mean.norm()),"feature_std_norm":float(feat_std.norm()),
                     "target_mean":tar_mean.cpu().tolist(),"target_std":tar_std.cpu().tolist(),
                     "action_mean_norm":float(act_mean.norm()),"action_std_norm":float(act_std.norm()),
                     "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                     "head_bias_drift":float((bnow-prev_b).norm())}
                if prev_F is not None:
                    row["feature_mean_drift"]=float((feat_mean-prev_F[0]).norm())
                    row["feature_std_drift"]=float((feat_std-prev_F[1]).norm())
                    row["target_mean_drift"]=float((tar_mean-prev_Y[0]).norm())
                    row["target_std_drift"]=float((tar_std-prev_Y[1]).norm())
                # optional actor update from refreshed critic
                if args.actor_update=="on":
                    with torch.no_grad():
                        vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4);nv=m.value_with_preference(cur,w)
                        adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
                    ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
                    if float((ratio-1).abs().max())>1e-4:raise RuntimeError("ratio invariant")
                    al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                    actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
                actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
                row["actor_param_drift"]=float((actor_now-prev_actor).norm())
                prev_actor=actor_now.clone();prev_W=Wnow.clone();prev_b=bnow.clone();prev_F=(feat_mean.clone(),feat_std.clone());prev_Y=(tar_mean.clone(),tar_std.clone())
                logs.append(row);save_snap(update)
            all_logs[label]=logs
          out={"schema":"t5_c19_actor_coupling_train_v1","status":"COMPLETE","actor_update":args.actor_update,"recent_window":args.recent_window,
               "ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,"specialist_logs":all_logs}
          args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","actor_update":args.actor_update},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c21_warm_replay_audit():
    """Run former post_v2_t5_c21_warm_replay_audit.py stage."""
    """Value replay for C21 recent3 versus warm-started recent12."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.replay_audit import ReplayAudit
    
    
    class C21WarmReplayAudit(ReplayAudit):
        """Value replay for C21 recent3 versus warm-started recent12."""
    
        arms = {"recent3": "post_v2_t5_c21_recent3-2026-09-23",
                "recent12warm": "post_v2_t5_c21_recent12warm-2026-09-23"}
        schema = "t5_c21_replay_audit_v1"
        seed_base = 1210000
        run = "post_v2_t5_c21_recent12warm-2026-09-23"
    
    
    if True:
        C21WarmReplayAudit.main()

def run_post_v2_t5_c21_warm_support_train():
    """Run former post_v2_t5_c21_warm_support_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    from collections import deque
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2):
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--actor-update",choices=("on","off"),required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--recent-window",type=int,default=3)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd");env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(51000+idx);np.random.seed(51000+idx)
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            for n,p in m.named_parameters():
                if n.startswith("critic_body"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
            w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            recent=deque(maxlen=args.recent_window)
            cur,_=env.reset(seed=510001+idx*1000);cur=obs_tensor(cur).cuda()
            logs=[]
            def save_snap(tag):
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"actor_update":args.actor_update},
                           args.output.parent/f"{label}_snap_{tag}.pt")
            save_snap(0)
            prev_F=None;prev_Y=None;prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
            prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params])
            for update in range(1,args.updates+1):
                ob=[];pre=[];old=[];rw=[];dn=[];acts=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
                    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());acts.append(a)
                    cur=obs_tensor(nxt).cuda()
                fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1);fa=torch.cat(acts)
                rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
                recent.append((F.detach().cpu(),Y.detach().cpu()))
                with torch.no_grad():pre=m.critic_head(F);pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
                if len(recent) >= args.recent_window:
                    FF=torch.cat([x[0] for x in recent],0).cuda();YY=torch.cat([x[1] for x in recent],0).cuda()
                    Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                    with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit);post=m.critic_head(F)
                else:
                    with torch.no_grad():post=m.critic_head(F)
                post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
                # drift metrics before optional actor update
                Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
                feat_mean=F.mean(0);feat_std=F.std(0);tar_mean=Y.mean(0);tar_std=Y.std(0);act_mean=fa.mean(0);act_std=fa.std(0)
                row={"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"buffer_len":len(recent),
                     "feature_mean_norm":float(feat_mean.norm()),"feature_std_norm":float(feat_std.norm()),
                     "target_mean":tar_mean.cpu().tolist(),"target_std":tar_std.cpu().tolist(),
                     "action_mean_norm":float(act_mean.norm()),"action_std_norm":float(act_std.norm()),
                     "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                     "head_bias_drift":float((bnow-prev_b).norm())}
                if prev_F is not None:
                    row["feature_mean_drift"]=float((feat_mean-prev_F[0]).norm())
                    row["feature_std_drift"]=float((feat_std-prev_F[1]).norm())
                    row["target_mean_drift"]=float((tar_mean-prev_Y[0]).norm())
                    row["target_std_drift"]=float((tar_std-prev_Y[1]).norm())
                # optional actor update from refreshed critic
                if args.actor_update=="on":
                    with torch.no_grad():
                        vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4);nv=m.value_with_preference(cur,w)
                        adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
                    ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
                    if float((ratio-1).abs().max())>1e-4:raise RuntimeError("ratio invariant")
                    al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                    actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
                actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
                row["actor_param_drift"]=float((actor_now-prev_actor).norm())
                prev_actor=actor_now.clone();prev_W=Wnow.clone();prev_b=bnow.clone();prev_F=(feat_mean.clone(),feat_std.clone());prev_Y=(tar_mean.clone(),tar_std.clone())
                logs.append(row);save_snap(update)
            all_logs[label]=logs
          out={"schema":"t5_c19_actor_coupling_train_v1","status":"COMPLETE","actor_update":args.actor_update,"recent_window":args.recent_window,
               "ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,"specialist_logs":all_logs}
          args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","actor_update":args.actor_update},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c22_diverse_support_train():
    """Run former post_v2_t5_c22_diverse_support_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2):
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64)
        X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        d0=np.mean(X*X,axis=1);sel=[int(np.argmax(d0))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1.0
            q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--support-size",type=int,default=12)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
          cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
          env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(51000+idx);np.random.seed(51000+idx)
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            for p in m.parameters():p.requires_grad_(False)
            w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            pool=[];summaries=[];cur,_=env.reset(seed=510001+idx*1000);cur=obs_tensor(cur).cuda();logs=[]
            def save_snap(tag):
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"support_mode":"diverse12"},
                           args.output.parent/f"{label}_snap_{tag}.pt")
            save_snap(0)
            prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
            for update in range(1,args.updates+1):
                ob=[];rw=[];dn=[];cmds=[]
                for _ in range(args.horizon):
                    with torch.no_grad():
                        cmds.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
                        a=m.act_inference_with_preference(cur,w)
                    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
                fo=torch.cat(ob);fw=w.repeat(args.horizon,1);rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach();pre=m.critic_head(F)
                pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
                C=np.concatenate(cmds,0);summary=np.r_[C.mean(0),C.std(0),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy()]
                pool.append((F.detach().cpu(),Y.detach().cpu()));summaries.append(summary.astype(np.float32))
                selected=[]
                if len(pool)>=args.support_size:
                    selected=select_diverse(summaries,args.support_size)
                    FF=torch.cat([pool[i][0] for i in selected],0).cuda();YY=torch.cat([pool[i][1] for i in selected],0).cuda()
                    Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                    with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit)
                with torch.no_grad():post=m.critic_head(F)
                post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
                Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
                # coverage of chosen support summaries relative to all seen summaries
                cov=None
                if selected:
                    X=np.asarray(summaries,float);A=X[selected];mu=A.mean(0);sd=A.std(0)+1e-6;Az=(A-mu)/sd;Xz=(X-mu)/sd
                    nn=[]
                    for x in Xz:nn.append(float(np.sqrt(np.min(np.mean((Az-x)**2,axis=1)))))
                    cov=float(np.mean(nn))
                logs.append({"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"pool_size":len(pool),"support_size":len(selected),
                             "selected_indices":selected,"support_age_span":(int(selected[-1]-selected[0]+1) if selected else 0),
                             "support_summary_nn_distance_to_seen":cov,"rollout_summary":summary.tolist(),
                             "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                             "head_bias_drift":float((bnow-prev_b).norm())})
                prev_W=Wnow.clone();prev_b=bnow.clone();save_snap(update)
            all_logs[label]=logs
          out={"schema":"t5_c22_diverse_support_train_v1","status":"COMPLETE","actor_frozen":True,"critic_body_frozen":True,
               "support_mode":"diverse","support_size":args.support_size,"ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,
               "selection_summary":"command mean/std + critic feature mean/std only; target excluded","specialist_logs":all_logs}
          args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","support_mode":"diverse","support_size":args.support_size},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c22_freeze():
    """Run former post_v2_t5_c22_freeze.py stage."""
    """Freeze C22: support-topology diversity."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.experiment_io.freeze import RUNS, Freeze
    
    CONTROL = "post_v2_t5_c21_recent12warm-2026-09-23"
    
    
    class C22Freeze(Freeze):
        """Freeze C22: support-topology diversity."""
    
        run = "post_v2_t5_c22_diverse12-2026-09-23"
        schema = "t5_c22_support_topology_diversity_synthesis_v1"
        status = "C22_CLOSED_DIVERSITY_STRONGLY_HELPFUL_BUT_CURRENT_STREAM_DIVERSE12_NOT_SUFFICIENT"
        artifacts = ("audit.json", "replay_audit.json", "terminal_multisuite.json", "synthesis.json")
    
        def body(self):
            rep = self.load("replay_audit.json")
            mul = self.load("terminal_multisuite.json")
            return {
                "aggregate": {
                    "coverage": {
                        "consecutive_support_to_seen_distance_mean": 2.1288670668,
                        "diverse_support_to_seen_distance_mean": 0.0608117817,
                        "diverse_over_consecutive_ratio": 0.028565326,
                        "diverse_support_age_span_mean": 18.42857143,
                    },
                    "head_solution_drift": {"consecutive12": 1.027003399, "diverse12": 0.625231252},
                    "replay_u25": {
                        "consecutive12": rep["aggregate"]["consecutive12"][25],
                        "diverse12": rep["aggregate"]["diverse12"][25],
                    },
                    "terminal_multisuite": mul["aggregate"],
                },
                "decision": {
                    "C22": "CLOSED — diversity hypothesis SUPPORTED; sufficiency gate FAIL",
                    "support_diversity": "STRONGLY SUPPORTED",
                    "diverse12_within_single_stream": "INSUFFICIENT as final repair",
                    "actor_updates": "REMAIN OFF",
                    "full_T4": "BLOCKED",
                    "V2": "OFF",
                    "next": "C23 true reset-diverse reference-support pilot with exactly 12 independently reset/command-seeded supports; actor/body frozen, ridge lambda=1, H32 target. Primary gate: multi-suite fresh H32 EV non-negative, especially Orientation.",
                },
                "provenance": {
                    "control_audit_sha256": self.sha(RUNS / CONTROL / "audit.json"),
                    "diverse_audit_sha256": self.sha(self.dir / "audit.json"),
                    "replay_sha256": self.sha(self.dir / "replay_audit.json"),
                    "multisuite_sha256": self.sha(self.dir / "terminal_multisuite.json"),
                },
            }
    
        def manifest_extra(self):
            return {"paired_control": {"audit.json": {"sha256": self.sha(RUNS / CONTROL / "audit.json")}}}
    
        def summary(self, syn):
            return {"status": syn["status"],
                    "multisuite": syn["aggregate"]["terminal_multisuite"],
                    "next": syn["decision"]["next"]}
    
    
    if True:
        C22Freeze.main()

def run_post_v2_t5_c22_replay_audit():
    """Run former post_v2_t5_c22_replay_audit.py stage."""
    """Value replay for C22 consecutive versus diverse support."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.replay_audit import ReplayAudit
    
    
    class C22ReplayAudit(ReplayAudit):
        """Value replay for C22 consecutive versus diverse support."""
    
        arms = {"consecutive12": "post_v2_t5_c21_recent12warm-2026-09-23",
                "diverse12": "post_v2_t5_c22_diverse12-2026-09-23"}
        schema = "t5_c22_replay_audit_v1"
        seed_base = 1310000
        run = "post_v2_t5_c22_diverse12-2026-09-23"
    
    
    if True:
        C22ReplayAudit.main()

def run_post_v2_t5_c22_reset_replay_audit():
    """Run former post_v2_t5_c22_reset_replay_audit.py stage."""
    """Value replay for C22 consecutive versus reset-seeded support."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.replay_audit import ReplayAudit
    
    
    class C22ResetReplayAudit(ReplayAudit):
        """Value replay for C22 consecutive versus reset-seeded support."""
    
        arms = {"consecutive12": "post_v2_t5_c21_recent12warm-2026-09-23",
                "reset12": "post_v2_t5_c22_reset12-2026-09-23"}
        schema = "t5_c21_replay_audit_v1"
        seed_base = 1210000
        run = "post_v2_t5_c22_reset12-2026-09-23"
    
    
    if True:
        C22ResetReplayAudit.main()

def run_post_v2_t5_c22_reset_support_train():
    """Run former post_v2_t5_c22_reset_support_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    from collections import deque
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2):
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--actor-update",choices=("on","off"),required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--recent-window",type=int,default=3)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd");env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(51000+idx);np.random.seed(51000+idx)
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            for n,p in m.named_parameters():
                if n.startswith("critic_body"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
            w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            recent=deque(maxlen=args.recent_window)
            cur=None
            logs=[]
            def save_snap(tag):
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"actor_update":args.actor_update},
                           args.output.parent/f"{label}_snap_{tag}.pt")
            save_snap(0)
            prev_F=None;prev_Y=None;prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
            prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params])
            for update in range(1,args.updates+1):
                # independent reset/command support rollout each update
                cur,_=env.reset(seed=1510000+idx*10000+update*97);cur=obs_tensor(cur).cuda()
                ob=[];pre=[];old=[];rw=[];dn=[];acts=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
                    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());acts.append(a)
                    cur=obs_tensor(nxt).cuda()
                fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1);fa=torch.cat(acts)
                rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
                recent.append((F.detach().cpu(),Y.detach().cpu()))
                with torch.no_grad():pre=m.critic_head(F);pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
                if len(recent) >= args.recent_window:
                    FF=torch.cat([x[0] for x in recent],0).cuda();YY=torch.cat([x[1] for x in recent],0).cuda()
                    Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                    with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit);post=m.critic_head(F)
                else:
                    with torch.no_grad():post=m.critic_head(F)
                post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
                # drift metrics before optional actor update
                Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
                feat_mean=F.mean(0);feat_std=F.std(0);tar_mean=Y.mean(0);tar_std=Y.std(0);act_mean=fa.mean(0);act_std=fa.std(0)
                row={"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"buffer_len":len(recent),
                     "feature_mean_norm":float(feat_mean.norm()),"feature_std_norm":float(feat_std.norm()),
                     "target_mean":tar_mean.cpu().tolist(),"target_std":tar_std.cpu().tolist(),
                     "action_mean_norm":float(act_mean.norm()),"action_std_norm":float(act_std.norm()),
                     "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                     "head_bias_drift":float((bnow-prev_b).norm())}
                if prev_F is not None:
                    row["feature_mean_drift"]=float((feat_mean-prev_F[0]).norm())
                    row["feature_std_drift"]=float((feat_std-prev_F[1]).norm())
                    row["target_mean_drift"]=float((tar_mean-prev_Y[0]).norm())
                    row["target_std_drift"]=float((tar_std-prev_Y[1]).norm())
                # optional actor update from refreshed critic
                if args.actor_update=="on":
                    with torch.no_grad():
                        vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4);nv=m.value_with_preference(cur,w)
                        adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
                    ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
                    if float((ratio-1).abs().max())>1e-4:raise RuntimeError("ratio invariant")
                    al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                    actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
                actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
                row["actor_param_drift"]=float((actor_now-prev_actor).norm())
                prev_actor=actor_now.clone();prev_W=Wnow.clone();prev_b=bnow.clone();prev_F=(feat_mean.clone(),feat_std.clone());prev_Y=(tar_mean.clone(),tar_std.clone())
                logs.append(row);save_snap(update)
            all_logs[label]=logs
          out={"schema":"t5_c22_reset_support_train_v1","status":"COMPLETE","actor_update":args.actor_update,"support_topology":"independent_reset","recent_window":args.recent_window,
               "ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,"specialist_logs":all_logs}
          args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","actor_update":args.actor_update},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c22_reset_terminal_multisuite():
    """Run former post_v2_t5_c22_reset_terminal_multisuite.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"consecutive12":ROOT/"runs/post_v2_t5_c21_recent12warm-2026-09-23","reset12":ROOT/"runs/post_v2_t5_c22_reset12-2026-09-23"}
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"t5_c22_terminal_multisuite_v1","arms":{}}
      for arm,run in RUNS.items():
       ar={}
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);suites=[]
        for si in range(4):
         seed=1410000+bi*1000+si*97;cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];F=[];C=[]
         with torch.no_grad():
          for t in range(64):
           feat=m.critic_body(m._with_w(cur,w));V.append(m.critic_head(feat).cpu().numpy())
           if t<32:F.append(feat.cpu().numpy());C.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
           a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None);FF=np.concatenate(F,0);CC=np.concatenate(C,0)
         # first H32 segment target summary to match training H32 segments
         Y0=H32[:32].reshape(-1,4)
         suites.append({"suite":si,"seed":seed,"h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
           "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean()),
           "fresh_summary":np.r_[CC.mean(0),CC.std(0),FF.mean(0),FF.std(0)].tolist(),
           "target_summary":np.r_[Y0.mean(0),Y0.std(0)].tolist()})
        ar[lab]=suites
       out["arms"][arm]=ar
      agg={}
      for arm in RUNS:
       e=[];mc=[];b=[];sv=[];by=[[] for _ in range(4)]
       for lab in ORDER:
        for s in out["arms"][arm][lab]:
         e+=s["h32_ev"];mc+=s["mc64_ev"];b+=s["h32_bias"];sv.append(s["survival"])
         for j,x in enumerate(s["h32_ev"]):by[j].append(x)
       agg[arm]={"h32_ev_mean":float(np.mean(e)),"h32_ev_std":float(np.std(e)),"h32_negative_fraction":float(np.mean(np.array(e)<0)),
        "h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(mc)),"mc64_negative_fraction":float(np.mean(np.array(mc)<0)),
        "h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg
      p=RUNS["reset12"]/"terminal_multisuite.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c22_terminal_multisuite():
    """Run former post_v2_t5_c22_terminal_multisuite.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"consecutive12":ROOT/"runs/post_v2_t5_c21_recent12warm-2026-09-23","diverse12":ROOT/"runs/post_v2_t5_c22_diverse12-2026-09-23"}
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c22_terminal_multisuite_v1","arms":{}}
      for arm,run in RUNS.items():
       rows=[]
       for suite in range(3):
        for bi,lab in enumerate(ORDER):
         m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
         w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);cur,_=env.reset(seed=1410000+suite*10000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H=ret(R,D,32);M=ret(R,D,None)
         rows.append({"suite":suite,"policy":lab,"h32_ev":[ev(H[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(M[:,:,j],V[:,:,j]) for j in range(4)],
                      "h32_bias":[float(np.mean(V[:,:,j]-H[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean())})
       out["arms"][arm]=rows
      agg={}
      for arm,rows in out["arms"].items():
       e=[];m=[];b=[];sv=[];by=[[] for _ in range(4)]
       for r in rows:
        e+=r["h32_ev"];m+=r["mc64_ev"];b+=r["h32_bias"];sv.append(r["survival"])
        for j,x in enumerate(r["h32_ev"]):by[j].append(x)
       agg[arm]={"h32_ev_mean":float(np.mean(e)),"h32_ev_std":float(np.std(e)),"h32_negative_fraction":float(np.mean(np.array(e)<0)),
                 "h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(m)),"mc64_negative_fraction":float(np.mean(np.array(m)<0)),
                 "h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg
      p=RUNS["diverse12"]/"terminal_multisuite.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c23_freeze():
    """Run former post_v2_t5_c23_freeze.py stage."""
    """Freeze C23: the reset-diverse support principle."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.experiment_io.freeze import Freeze
    
    PILOT = "scripts/rl/experiments/architectures/preference_architectures/post_v2/causal_semantics.py"
    
    
    class C23Freeze(Freeze):
        """Freeze C23: the reset-diverse support principle."""
    
        run = "post_v2_t5_c23_reset_diverse-2026-09-23"
        schema = "t5_c23_reset_diverse_synthesis_v1"
        status = "C23_CLOSED_RESET_DIVERSE_SUPPORT_PRINCIPLE_PASS_ALL_HEAD_ROBUSTNESS_PARTIAL"
        artifacts = ("audit.json", "synthesis.json")
    
        def body(self):
            return {
                "aggregate": self.load("audit.json")["aggregate"],
                "findings": {
                    "primary": "At identical support count, ridge lambda, frozen actor/body and H32 target, independent reset/seeded support changes fresh H32 EV from -1.211 to +0.196 and Orientation from -1.142 to +0.236.",
                    "mc64": "Fresh MC64 EV changes from -0.249 to +0.464; reset-diverse has 0% negative MC64 head evaluations and survival 1.0.",
                    "round_consistency": "Reset-diverse H32 mean remains positive in all four rounds (+0.233,+0.159,+0.141,+0.252); Orientation is also positive in all four rounds.",
                    "mechanism": "Head-solution drift is not materially lower (2.239 vs 2.184 excluding first round). The repair therefore comes primarily from representative support/generalization, not merely a more stationary optimum.",
                    "residual": "Tracking remains the weak head: mean H32 EV about -0.092 with high negative fraction, while Angular, Orientation and Smoothness are positive. Thus the support principle passes but all-head robustness is not yet complete.",
                },
                "decision": {
                    "C23": "CLOSED — reset-diverse support principle PASS; all-head robustness PARTIAL",
                    "representative_reset_command_state_support": "SUPPORTED",
                    "separate_critics": "NOT JUSTIFIED",
                    "body_anchor": "NOT JUSTIFIED",
                    "actor_updates": "STILL OFF for one more gate",
                    "full_T4": "BLOCKED",
                    "V2": "OFF",
                    "next": "C24 Tracking-head residual audit under the reset-diverse repair. Keep the C23 support mechanism fixed and diagnose why Tracking H32 EV remains slightly negative while A/O/S generalize. First determine whether this is target variance/horizon mismatch or a head-specific feature-fit issue before re-enabling actor updates.",
                },
                "provenance": {
                    "audit_sha256": self.sha(self.dir / "audit.json"),
                    "pilot_script_sha256": self.sha(PILOT),
                },
            }
    
        def manifest_extra(self):
            return {"sources": {"pilot": {"sha256": self.sha(PILOT)}}}
    
        def summary(self, syn):
            return {"status": syn["status"], "aggregate": syn["aggregate"], "next": syn["decision"]["next"]}
    
    
    if True:
        C23Freeze.main()

def run_post_v2_t5_c23_paired_terminal_multisuite():
    """Run former post_v2_t5_c23_paired_terminal_multisuite.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"reset12_recent":ROOT/"runs/post_v2_t5_c23_reset_recent12-2026-09-23","reset12_diverse":ROOT/"runs/post_v2_t5_c23_reset_diverse12-2026-09-23"}
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"t5_c22_terminal_multisuite_v1","arms":{}}
      for arm,run in RUNS.items():
       ar={}
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);suites=[]
        for si in range(4):
         seed=1410000+bi*1000+si*97;cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];F=[];C=[]
         with torch.no_grad():
          for t in range(64):
           feat=m.critic_body(m._with_w(cur,w));V.append(m.critic_head(feat).cpu().numpy())
           if t<32:F.append(feat.cpu().numpy());C.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
           a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None);FF=np.concatenate(F,0);CC=np.concatenate(C,0)
         # first H32 segment target summary to match training H32 segments
         Y0=H32[:32].reshape(-1,4)
         suites.append({"suite":si,"seed":seed,"h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
           "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean()),
           "fresh_summary":np.r_[CC.mean(0),CC.std(0),FF.mean(0),FF.std(0)].tolist(),
           "target_summary":np.r_[Y0.mean(0),Y0.std(0)].tolist()})
        ar[lab]=suites
       out["arms"][arm]=ar
      agg={}
      for arm in RUNS:
       e=[];mc=[];b=[];sv=[];by=[[] for _ in range(4)]
       for lab in ORDER:
        for s in out["arms"][arm][lab]:
         e+=s["h32_ev"];mc+=s["mc64_ev"];b+=s["h32_bias"];sv.append(s["survival"])
         for j,x in enumerate(s["h32_ev"]):by[j].append(x)
       agg[arm]={"h32_ev_mean":float(np.mean(e)),"h32_ev_std":float(np.std(e)),"h32_negative_fraction":float(np.mean(np.array(e)<0)),
        "h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(mc)),"mc64_negative_fraction":float(np.mean(np.array(mc)<0)),
        "h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg
      p=RUNS["reset12_diverse"]/"terminal_multisuite.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c23_phase_split_audit():
    """Run former post_v2_t5_c23_phase_split_audit.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"recent":ROOT/"runs/post_v2_t5_c23_reset_recent12-2026-09-23","diverse":ROOT/"runs/post_v2_t5_c23_reset_diverse12-2026-09-23"}
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def segret(R,D,st,en):
     out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
     for t in range(en-1,st-1,-1):
      run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c23_phase_split_v1","arms":{}}
      for arm,run in RUNS.items():
       ar={}
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);suites=[]
        for si in range(4):
         cur,_=env.reset(seed=1710000+bi*1000+si*97);cur=ot(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=ot(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);Y0=segret(R,D,0,32);Y1=segret(R,D,32,64)
         suites.append({"suite":si,"first_ev":[ev(Y0[:,:,j],V[:32,:,j]) for j in range(4)],"second_ev":[ev(Y1[:,:,j],V[32:,:,j]) for j in range(4)]})
        ar[lab]=suites
       out["arms"][arm]=ar
      agg={}
      for arm in RUNS:
       f=[[] for _ in range(4)];s=[[] for _ in range(4)]
       for lab in ORDER:
        for q in out["arms"][arm][lab]:
         for j in range(4):f[j].append(q["first_ev"][j]);s[j].append(q["second_ev"][j])
       agg[arm]={"first_h32_mean":float(np.mean(f)),"second_h32_mean":float(np.mean(s)),
                 "first_by_head":[float(np.mean(x)) for x in f],"second_by_head":[float(np.mean(x)) for x in s],
                 "first_negative_fraction":float(np.mean(np.array(f)<0)),"second_negative_fraction":float(np.mean(np.array(s)<0))}
      out["aggregate"]=agg
      p=RUNS["diverse"]/"phase_split.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c23_replay_audit():
    """Run former post_v2_t5_c23_replay_audit.py stage."""
    """Value replay for C23 reset-recent versus reset-diverse support."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.replay_audit import ReplayAudit
    
    
    class C23ReplayAudit(ReplayAudit):
        """Value replay for C23 reset-recent versus reset-diverse support."""
    
        arms = {"reset12_recent": "post_v2_t5_c22_reset12-2026-09-23",
                "reset12_diverse": "post_v2_t5_c23_reset_diverse12-2026-09-23"}
        schema = "t5_c21_replay_audit_v1"
        seed_base = 1210000
        run = "post_v2_t5_c23_reset_diverse12-2026-09-23"
    
    
    if True:
        C23ReplayAudit.main()

def run_post_v2_t5_c23_reset_diverse_pilot():
    """Run former post_v2_t5_c23_reset_diverse_pilot.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/post_v2_t5_c23_reset_diverse-2026-09-23"
    OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S");G=.99;H=32;NENV=8;NROUND=4;NSUP=12
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def mc(rt,dt):
     out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
     for t in range(len(rt)-1,-1,-1):
      run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
     return out
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
     A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
     sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
     return sol[:-1].T,sol[-1]
    def collect_segment(env,m,w,mgr,cur=None,seed=None):
     if seed is not None:
      cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
     ob=[];rw=[];dn=[];cmds=[]
     with torch.no_grad():
      for _ in range(H):
       cmds.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
       a=m.act_inference_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       from talon_rl.rewards.objectives import normalized_objective_vector
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
       dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
     fo=torch.cat(ob);fw=w.repeat(H,1);Y=mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
     with torch.no_grad():F=m.critic_body(m._with_w(fo,fw))
     C=np.concatenate(cmds,0)
     S=np.r_[C.mean(0),C.std(0),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),Y.mean(0).cpu().numpy(),Y.std(0).cpu().numpy()]
     return F.cpu().numpy(),Y.cpu().numpy(),S.astype(np.float32),cur
    def eval_head(env,m,w,mgr,W,b,seed,horizon=64):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];D=[];F=[]
     from talon_rl.rewards.objectives import normalized_objective_vector
     with torch.no_grad():
      for _ in range(horizon):
       f=m.critic_body(m._with_w(cur,w));F.append(f.cpu().numpy())
       a=m.act_inference_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
       D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
     R=np.asarray(R);D=np.asarray(D,bool);F=np.asarray(F)
     P=np.einsum("ted,hd->teh",F,W)+b
     def ret(seg=None):
      out=np.zeros_like(R)
      if seg is None:
       run=np.zeros_like(R[0])
       for t in range(len(R)-1,-1,-1):
        run=R[t]+G*run*(~D[t])[:,None];out[t]=run
      else:
       for st in range(0,len(R),seg):
        en=min(st+seg,len(R));run=np.zeros_like(R[0])
        for t in range(en-1,st-1,-1):
         run=R[t]+G*run*(~D[t])[:,None];out[t]=run
      return out
     H32=ret(32);MC64=ret(None)
     return {"h32_ev":[ev(H32[:,:,j],P[:,:,j]) for j in range(4)],
             "mc64_ev":[ev(MC64[:,:,j],P[:,:,j]) for j in range(4)],
             "h32_bias":[float(np.mean(P[:,:,j]-H32[:,:,j])) for j in range(4)],
             "survival":float(1-D.any(0).mean())}
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
      mgr=env.unwrapped.reward_manager;report={"schema":"c23_reset_diverse_v1","rounds":{}}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(o.shape[-1],ad).cuda()
       initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero")
       m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       prev={"consecutive":None,"reset_diverse":None};rows=[]
       stream,_=env.reset(seed=1510000+bi*10000);stream=obs_tensor(stream).cuda()
       for rnd in range(NROUND):
        arms={}
        for mode in ("consecutive","reset_diverse"):
         Fs=[];Ys=[];Ss=[]
         if mode=="consecutive":
          cur=stream
          for k in range(NSUP):
           F,Y,S,cur=collect_segment(env,m,w,mgr,cur=cur)
           Fs.append(F);Ys.append(Y);Ss.append(S)
          stream=cur
         else:
          for k in range(NSUP):
           seed=1610000+bi*100000+rnd*1000+k*37
           F,Y,S,_=collect_segment(env,m,w,mgr,seed=seed)
           Fs.append(F);Ys.append(Y);Ss.append(S)
         F=np.concatenate(Fs);Y=np.concatenate(Ys);W,b=ridge(F,Y,1.0)
         sol=np.r_[W.reshape(-1),b]
         drift=0.0 if prev[mode] is None else float(np.linalg.norm(sol-prev[mode]))
         prev[mode]=sol
         suites=[]
         for s in range(3):
          seed=1710000+bi*100000+rnd*10000+s*503
          suites.append(eval_head(env,m,w,mgr,W,b,seed))
         arms[mode]={"solution_drift":drift,"support_summary_std_mean":float(np.std(np.stack(Ss),axis=0).mean()),"suites":suites}
        rows.append({"round":rnd,"arms":arms})
       report["rounds"][lab]=rows
      (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
      agg={}
      for mode in ("consecutive","reset_diverse"):
       h=[];mc=[];bias=[];sv=[];dr=[];orient=[]
       for lab in ORDER:
        for row in report["rounds"][lab]:
         dr.append(row["arms"][mode]["solution_drift"])
         for q in row["arms"][mode]["suites"]:
          h+=q["h32_ev"];mc+=q["mc64_ev"];bias+=q["h32_bias"];sv.append(q["survival"]);orient.append(q["h32_ev"][2])
       agg[mode]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),
                  "orientation_h32_ev_mean":float(np.mean(orient)),"mc64_ev_mean":float(np.mean(mc)),
                  "mc64_negative_fraction":float(np.mean(np.array(mc)<0)),"h32_mean_abs_bias":float(np.mean(np.abs(bias))),
                  "solution_drift_mean_excluding_first":float(np.mean([x for x in dr if x>0])),"min_survival":float(np.min(sv))}
      report["aggregate"]=agg;(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
      print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c23_reset_diverse_train():
    """Run former post_v2_t5_c23_reset_diverse_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2):
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64)
        X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        d0=np.mean(X*X,axis=1);sel=[int(np.argmax(d0))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1.0
            q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--support-size",type=int,default=12)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
          cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
          env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(51000+idx);np.random.seed(51000+idx)
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            for p in m.parameters():p.requires_grad_(False)
            w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            pool=[];summaries=[];cur=None;logs=[]
            def save_snap(tag):
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"support_mode":"diverse12"},
                           args.output.parent/f"{label}_snap_{tag}.pt")
            save_snap(0)
            prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
            for update in range(1,args.updates+1):
                cur,_=env.reset(seed=1610000+idx*10000+update*97);cur=obs_tensor(cur).cuda()
                ob=[];rw=[];dn=[];cmds=[]
                for _ in range(args.horizon):
                    with torch.no_grad():
                        cmds.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
                        a=m.act_inference_with_preference(cur,w)
                    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
                fo=torch.cat(ob);fw=w.repeat(args.horizon,1);rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach();pre=m.critic_head(F)
                pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
                C=np.concatenate(cmds,0);summary=np.r_[C.mean(0),C.std(0),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy()]
                pool.append((F.detach().cpu(),Y.detach().cpu()));summaries.append(summary.astype(np.float32))
                selected=[]
                if len(pool)>=args.support_size:
                    selected=select_diverse(summaries,args.support_size)
                    FF=torch.cat([pool[i][0] for i in selected],0).cuda();YY=torch.cat([pool[i][1] for i in selected],0).cuda()
                    Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                    with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit)
                with torch.no_grad():post=m.critic_head(F)
                post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
                Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
                # coverage of chosen support summaries relative to all seen summaries
                cov=None
                if selected:
                    X=np.asarray(summaries,float);A=X[selected];mu=A.mean(0);sd=A.std(0)+1e-6;Az=(A-mu)/sd;Xz=(X-mu)/sd
                    nn=[]
                    for x in Xz:nn.append(float(np.sqrt(np.min(np.mean((Az-x)**2,axis=1)))))
                    cov=float(np.mean(nn))
                logs.append({"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"pool_size":len(pool),"support_size":len(selected),
                             "selected_indices":selected,"support_age_span":(int(selected[-1]-selected[0]+1) if selected else 0),
                             "support_summary_nn_distance_to_seen":cov,"rollout_summary":summary.tolist(),
                             "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                             "head_bias_drift":float((bnow-prev_b).norm())})
                prev_W=Wnow.clone();prev_b=bnow.clone();save_snap(update)
            all_logs[label]=logs
          out={"schema":"t5_c23_reset_diverse_train_v1","status":"COMPLETE","actor_frozen":True,"critic_body_frozen":True,
               "support_mode":"independent_reset_diverse","support_size":args.support_size,"ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,
               "selection_summary":"command mean/std + critic feature mean/std only; target excluded","specialist_logs":all_logs}
          args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","support_mode":"independent_reset_diverse","support_size":args.support_size},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c23_reset_recent_paired_train():
    """Run former post_v2_t5_c23_reset_recent_paired_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    from collections import deque
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2):
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--actor-update",choices=("on","off"),required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--recent-window",type=int,default=3)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd");env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(51000+idx);np.random.seed(51000+idx)
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            for n,p in m.named_parameters():
                if n.startswith("critic_body"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
            w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            recent=deque(maxlen=args.recent_window)
            cur=None
            logs=[]
            def save_snap(tag):
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"actor_update":args.actor_update},
                           args.output.parent/f"{label}_snap_{tag}.pt")
            save_snap(0)
            prev_F=None;prev_Y=None;prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
            prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params])
            for update in range(1,args.updates+1):
                # independent reset/command support rollout each update
                cur,_=env.reset(seed=1610000+idx*10000+update*97);cur=obs_tensor(cur).cuda()
                ob=[];pre=[];old=[];rw=[];dn=[];acts=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
                    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());acts.append(a)
                    cur=obs_tensor(nxt).cuda()
                fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1);fa=torch.cat(acts)
                rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
                recent.append((F.detach().cpu(),Y.detach().cpu()))
                with torch.no_grad():pre=m.critic_head(F);pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
                if len(recent) >= args.recent_window:
                    FF=torch.cat([x[0] for x in recent],0).cuda();YY=torch.cat([x[1] for x in recent],0).cuda()
                    Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                    with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit);post=m.critic_head(F)
                else:
                    with torch.no_grad():post=m.critic_head(F)
                post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
                # drift metrics before optional actor update
                Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
                feat_mean=F.mean(0);feat_std=F.std(0);tar_mean=Y.mean(0);tar_std=Y.std(0);act_mean=fa.mean(0);act_std=fa.std(0)
                row={"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"buffer_len":len(recent),
                     "feature_mean_norm":float(feat_mean.norm()),"feature_std_norm":float(feat_std.norm()),
                     "target_mean":tar_mean.cpu().tolist(),"target_std":tar_std.cpu().tolist(),
                     "action_mean_norm":float(act_mean.norm()),"action_std_norm":float(act_std.norm()),
                     "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                     "head_bias_drift":float((bnow-prev_b).norm())}
                if prev_F is not None:
                    row["feature_mean_drift"]=float((feat_mean-prev_F[0]).norm())
                    row["feature_std_drift"]=float((feat_std-prev_F[1]).norm())
                    row["target_mean_drift"]=float((tar_mean-prev_Y[0]).norm())
                    row["target_std_drift"]=float((tar_std-prev_Y[1]).norm())
                # optional actor update from refreshed critic
                if args.actor_update=="on":
                    with torch.no_grad():
                        vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4);nv=m.value_with_preference(cur,w)
                        adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
                    ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
                    if float((ratio-1).abs().max())>1e-4:raise RuntimeError("ratio invariant")
                    al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                    actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
                actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
                row["actor_param_drift"]=float((actor_now-prev_actor).norm())
                prev_actor=actor_now.clone();prev_W=Wnow.clone();prev_b=bnow.clone();prev_F=(feat_mean.clone(),feat_std.clone());prev_Y=(tar_mean.clone(),tar_std.clone())
                logs.append(row);save_snap(update)
            all_logs[label]=logs
          out={"schema":"t5_c23_reset_recent_paired_train_v1","status":"COMPLETE","actor_update":args.actor_update,"support_topology":"independent_reset","recent_window":args.recent_window,
               "ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,"specialist_logs":all_logs}
          args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","actor_update":args.actor_update},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c23_terminal_multisuite():
    """Run former post_v2_t5_c23_terminal_multisuite.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"reset12_recent":ROOT/"runs/post_v2_t5_c22_reset12-2026-09-23","reset12_diverse":ROOT/"runs/post_v2_t5_c23_reset_diverse12-2026-09-23"}
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"t5_c22_terminal_multisuite_v1","arms":{}}
      for arm,run in RUNS.items():
       ar={}
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);suites=[]
        for si in range(4):
         seed=1410000+bi*1000+si*97;cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];F=[];C=[]
         with torch.no_grad():
          for t in range(64):
           feat=m.critic_body(m._with_w(cur,w));V.append(m.critic_head(feat).cpu().numpy())
           if t<32:F.append(feat.cpu().numpy());C.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
           a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None);FF=np.concatenate(F,0);CC=np.concatenate(C,0)
         # first H32 segment target summary to match training H32 segments
         Y0=H32[:32].reshape(-1,4)
         suites.append({"suite":si,"seed":seed,"h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
           "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean()),
           "fresh_summary":np.r_[CC.mean(0),CC.std(0),FF.mean(0),FF.std(0)].tolist(),
           "target_summary":np.r_[Y0.mean(0),Y0.std(0)].tolist()})
        ar[lab]=suites
       out["arms"][arm]=ar
      agg={}
      for arm in RUNS:
       e=[];mc=[];b=[];sv=[];by=[[] for _ in range(4)]
       for lab in ORDER:
        for s in out["arms"][arm][lab]:
         e+=s["h32_ev"];mc+=s["mc64_ev"];b+=s["h32_bias"];sv.append(s["survival"])
         for j,x in enumerate(s["h32_ev"]):by[j].append(x)
       agg[arm]={"h32_ev_mean":float(np.mean(e)),"h32_ev_std":float(np.std(e)),"h32_negative_fraction":float(np.mean(np.array(e)<0)),
        "h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(mc)),"mc64_negative_fraction":float(np.mean(np.array(mc)<0)),
        "h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg
      p=RUNS["reset12_diverse"]/"terminal_multisuite.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c24_freeze():
    """Run former post_v2_t5_c24_freeze.py stage."""
    """Freeze C24: the Tracking-head residual under the reset-diverse repair."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.experiment_io.freeze import Freeze
    
    SRC = "scripts/rl/experiments/architectures/preference_architectures/post_v2"
    
    
    class C24Freeze(Freeze):
        """Freeze C24: the Tracking-head residual under the reset-diverse repair."""
    
        run = "post_v2_t5_c24_tracking_residual-2026-09-23"
        schema = "t5_c24_tracking_residual_synthesis_v1"
        status = "C24_CLOSED_TRACKING_RESIDUAL_NONSTRUCTURAL_SHORT_HORIZON_TARGET_GEOMETRY"
        artifacts = ("audit.json", "reset_breakdown.json", "synthesis.json")
    
        def body(self):
            a = self.load("audit.json")
            r = self.load("reset_breakdown.json")
            t = a["aggregate"]["target_stats"]["Tracking"]
            f = a["aggregate"]["fit_summary"]["Tracking"]
            return {
                "evidence": {
                    "tracking_h32_mc64_corr": t["h32_mc64_corr"],
                    "tracking_h32_var": t["h32_var"],
                    "tracking_mc64_var": t["mc64_var"],
                    "tracking_h32_ridge1_ev": f["h32_ridge_sweep"]["1.0"]["h32_ev_mean"],
                    "tracking_h32_best_sweep_ev": max(v["h32_ev_mean"] for v in f["h32_ridge_sweep"].values()),
                    "tracking_mc64_ridge1_ev": f["mc64_ridge1_ev_mean"],
                    "reset_breakdown": r["aggregate"],
                },
                "decision": {
                    "C24": "CLOSED",
                    "shared_body_tracking_capacity": "SUFFICIENT but H32 predictability weak",
                    "ridge_tuning": "NOT JUSTIFIED",
                    "tracking_h32_residual": "NONSTRUCTURAL / horizon-target-sensitive",
                    "critic_repair_principle": "RETAIN C23 reset-diverse support",
                    "actor_updates": "AUTHORIZED for next pilot only",
                    "full_T4": "STILL BLOCKED",
                    "V2": "OFF",
                    "next": "C25 actor-updating reset-diverse critic-support pilot. Keep C23 representative reset/command/state support, shared frozen critic body, ridge lambda=1, H32 target and PPO semantics fixed; re-enable actor updates only. Primary gate: A/O semantic gradients and fresh value generalization must remain stable under policy shift.",
                },
            }
    
        def manifest_extra(self):
            return {"sources": {
                "main": {"sha256": self.sha(f"{SRC}/post_v2_t5_c24_tracking_residual_audit.py")},
                "reset": {"sha256": self.sha(f"{SRC}/post_v2_t5_c24_reset_breakdown.py")},
            }}
    
        def summary(self, syn):
            return {"status": syn["status"], "evidence": syn["evidence"], "next": syn["decision"]["next"]}
    
    
    if True:
        C24Freeze.main()

def run_post_v2_t5_c24_phase_balanced_train():
    """Run former post_v2_t5_c24_phase_balanced_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2):
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64)
        X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        d0=np.mean(X*X,axis=1);sel=[int(np.argmax(d0))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1.0
            q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--support-size",type=int,default=12)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
          cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
          env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(51000+idx);np.random.seed(51000+idx)
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            for p in m.parameters():p.requires_grad_(False)
            w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            pool=[];summaries=[];cur=None;logs=[]
            def save_snap(tag):
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"support_mode":"diverse12"},
                           args.output.parent/f"{label}_snap_{tag}.pt")
            save_snap(0)
            prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
            for update in range(1,args.updates+1):
                cur,_=env.reset(seed=1610000+idx*10000+update*97);cur=obs_tensor(cur).cuda()
                ob=[];rw=[];dn=[];cmds=[]
                total_steps=args.horizon*2
                for _ in range(total_steps):
                    with torch.no_grad():
                        cmds.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
                        a=m.act_inference_with_preference(cur,w)
                    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
                phase_start=0 if (update % 2 == 1) else args.horizon
                phase_end=phase_start+args.horizon
                phase_ob=ob[phase_start:phase_end];phase_rw=rw[phase_start:phase_end];phase_dn=dn[phase_start:phase_end];phase_cmds=cmds[phase_start:phase_end]
                fo=torch.cat(phase_ob);fw=w.repeat(args.horizon,1);rt=torch.stack(phase_rw);dt=torch.stack(phase_dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach();pre=m.critic_head(F)
                pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
                C=np.concatenate(phase_cmds,0);summary=np.r_[C.mean(0),C.std(0),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy()]
                pool.append((F.detach().cpu(),Y.detach().cpu()));summaries.append(summary.astype(np.float32))
                selected=[]
                if len(pool)>=args.support_size:
                    selected=select_diverse(summaries,args.support_size)
                    FF=torch.cat([pool[i][0] for i in selected],0).cuda();YY=torch.cat([pool[i][1] for i in selected],0).cuda()
                    Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                    with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit)
                with torch.no_grad():post=m.critic_head(F)
                post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
                Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
                # coverage of chosen support summaries relative to all seen summaries
                cov=None
                if selected:
                    X=np.asarray(summaries,float);A=X[selected];mu=A.mean(0);sd=A.std(0)+1e-6;Az=(A-mu)/sd;Xz=(X-mu)/sd
                    nn=[]
                    for x in Xz:nn.append(float(np.sqrt(np.min(np.mean((Az-x)**2,axis=1)))))
                    cov=float(np.mean(nn))
                logs.append({"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"pool_size":len(pool),"support_size":len(selected),
                             "selected_indices":selected,"support_age_span":(int(selected[-1]-selected[0]+1) if selected else 0),
                             "support_summary_nn_distance_to_seen":cov,"rollout_summary":summary.tolist(),"phase_start":phase_start,
                             "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                             "head_bias_drift":float((bnow-prev_b).norm())})
                prev_W=Wnow.clone();prev_b=bnow.clone();save_snap(update)
            all_logs[label]=logs
          out={"schema":"t5_c24_phase_balanced_train_v1","status":"COMPLETE","actor_frozen":True,"critic_body_frozen":True,
               "support_mode":"independent_reset_phase_balanced_diverse","support_size":args.support_size,"ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,
               "selection_summary":"command mean/std + critic feature mean/std only; target excluded","specialist_logs":all_logs}
          args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","support_mode":"independent_reset_phase_balanced_diverse","support_size":args.support_size},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c24_phase_eval():
    """Run former post_v2_t5_c24_phase_eval.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"reset_diverse":ROOT/"runs/post_v2_t5_c23_reset_diverse12-2026-09-23","phase_balanced":ROOT/"runs/post_v2_t5_c24_phase_balanced12-2026-09-23"}
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def segret(R,D,st,en):
     out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
     for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c24_phase_eval_v1","arms":{}}
      for arm,run in RUNS.items():
       ar={}
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);suites=[]
        for si in range(4):
         cur,_=env.reset(seed=1810000+bi*1000+si*97);cur=ot(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=ot(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V)
         Y0=segret(R,D,0,32);Y1=segret(R,D,32,64);MC=segret(R,D,0,64)
         suites.append({"suite":si,"first_ev":[ev(Y0[:,:,j],V[:32,:,j]) for j in range(4)],
                        "second_ev":[ev(Y1[:,:,j],V[32:,:,j]) for j in range(4)],
                        "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                        "survival":float(1-D.any(0).mean())})
        ar[lab]=suites
       out["arms"][arm]=ar
      agg={}
      for arm in RUNS:
       f=[[] for _ in range(4)];s=[[] for _ in range(4)];mc=[[] for _ in range(4)];sv=[]
       for lab in ORDER:
        for q in out["arms"][arm][lab]:
         sv.append(q["survival"])
         for j in range(4):f[j].append(q["first_ev"][j]);s[j].append(q["second_ev"][j]);mc[j].append(q["mc64_ev"][j])
       both=[v for arr in f+s for v in arr]
       agg[arm]={"first_h32_mean":float(np.mean(f)),"second_h32_mean":float(np.mean(s)),"combined_h32_mean":float(np.mean(both)),
                 "first_by_head":[float(np.mean(x)) for x in f],"second_by_head":[float(np.mean(x)) for x in s],
                 "first_negative_fraction":float(np.mean(np.array(f)<0)),"second_negative_fraction":float(np.mean(np.array(s)<0)),
                 "combined_negative_fraction":float(np.mean(np.array(both)<0)),
                 "mc64_mean":float(np.mean(mc)),"mc64_by_head":[float(np.mean(x)) for x in mc],
                 "mc64_negative_fraction":float(np.mean(np.array(mc)<0)),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg
      p=RUNS["phase_balanced"]/"phase_eval.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c24_reset_breakdown():
    """Run former post_v2_t5_c24_reset_breakdown.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/post_v2_t5_c24_tracking_residual-2026-09-23"
    ORDER=("T","A","O","S");G=.99;NENV=8;H=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def returns(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def ev(y,p):
     return float(1-np.var(np.asarray(y).reshape(-1)-np.asarray(p).reshape(-1))/(np.var(np.asarray(y).reshape(-1))+1e-12))
    def ridge(F,Y):
     A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
     sol=np.linalg.solve(A.T@A+I,A.T@Y);return sol[:-1],sol[-1]
    def collect(env,m,w,mgr,seed):
     from talon_rl.rewards.objectives import normalized_objective_vector
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();F=[];R=[];D=[];C=[]
     with torch.no_grad():
      for _ in range(H):
       F.append(m.critic_body(m._with_w(cur,w)).cpu().numpy())
       C.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
       D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
     R=np.asarray(R);D=np.asarray(D,bool)
     return np.asarray(F),returns(R,D,32)[:,:,0],returns(R,D,None)[:,:,0],np.asarray(C)
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda()
      ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;out={}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(o.shape[-1],ad).cuda()
       initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero");m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       tr=[collect(env,m,w,mgr,2710000+bi*100000+i*97) for i in range(12)]
       F=np.concatenate([x[0].reshape(-1,x[0].shape[-1]) for x in tr])
       Y=np.concatenate([x[1].reshape(-1) for x in tr]);W,b=ridge(F,Y)
       rows=[]
       for i in range(6):
        seed=2910000+bi*100000+i*103;f,h32,mc64,c=collect(env,m,w,mgr,seed)
        p=f.reshape(-1,f.shape[-1])@W+b;p=p.reshape(h32.shape)
        cmd=np.linalg.norm(c[...,:2],axis=-1)
        rows.append({"seed":seed,"h32_ev":ev(h32,p),"h32_mc64_corr":float(np.corrcoef(h32.reshape(-1),mc64.reshape(-1))[0,1]),
          "command_speed_mean":float(cmd.mean()),"command_speed_std":float(cmd.std()),"h32_target_std":float(h32.std())})
       out[lab]=rows
      agg=[x for lab in ORDER for x in out[lab]]
      report={"schema":"c24_reset_breakdown_v1","specialists":out,
        "aggregate":{"h32_ev_mean":float(np.mean([x["h32_ev"] for x in agg])),"h32_ev_min":float(np.min([x["h32_ev"] for x in agg])),
          "negative_fraction":float(np.mean(np.array([x["h32_ev"] for x in agg])<0)),
          "ev_command_corr":float(np.corrcoef([x["h32_ev"] for x in agg],[x["command_speed_mean"] for x in agg])[0,1]),
          "ev_targetstd_corr":float(np.corrcoef([x["h32_ev"] for x in agg],[x["h32_target_std"] for x in agg])[0,1])}}
      (OUT/"reset_breakdown.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report["aggregate"],indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c24_tracking_residual_audit():
    """Run former post_v2_t5_c24_tracking_residual_audit.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/post_v2_t5_c24_tracking_residual-2026-09-23"
    OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S");G=.99;NENV=8;H=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    LAMBDAS=(0.0,1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0)
    def obs_tensor(x):
     if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):
       run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):
        run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2):
     A=np.c_[F,np.ones(len(F))]
     if l2==0:
      sol=np.linalg.lstsq(A,Y,rcond=None)[0]
     else:
      I=np.eye(A.shape[1]);I[-1,-1]=0
      sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
     return sol[:-1].T,sol[-1]
    def collect(env,m,w,mgr,seed):
     from talon_rl.rewards.objectives import normalized_objective_vector
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
     Fs=[];Rs=[];Ds=[];Cs=[]
     with torch.no_grad():
      for _ in range(H):
       Fs.append(m.critic_body(m._with_w(cur,w)).cpu().numpy())
       Cs.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
       a=m.act_inference_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       Rs.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
       Ds.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
     R=np.asarray(Rs);D=np.asarray(Ds,bool);F=np.asarray(Fs);C=np.asarray(Cs)
     return {"F":F,"C":C,"H32":ret(R,D,32),"MC64":ret(R,D,None),"D":D}
    def flatten(ds,key):
     return np.concatenate([x[key].reshape(-1,x[key].shape[-1]) for x in ds],axis=0)
    def corr(a,b):
     a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
     return float(np.corrcoef(a,b)[0,1])
    def command_bins(C):
     speed=np.linalg.norm(C[...,:2],axis=-1).reshape(-1)
     q=np.quantile(speed,[0,.25,.5,.75,1.0])
     return speed,q
    def eval_by_bins(Y,P,C):
     speed,q=command_bins(C);rows=[]
     for i in range(4):
      lo,hi=q[i],q[i+1]
      mask=(speed>=lo)&(speed<=hi if i==3 else speed<hi)
      rows.append({"bin":i,"lo":float(lo),"hi":float(hi),"n":int(mask.sum()),
                   "ev":ev(Y.reshape(-1)[mask],P.reshape(-1)[mask]),
                   "mse":float(np.mean((Y.reshape(-1)[mask]-P.reshape(-1)[mask])**2)),
                   "bias":float(np.mean(P.reshape(-1)[mask]-Y.reshape(-1)[mask]))})
     return rows
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
      mgr=env.unwrapped.reward_manager;report={"schema":"c24_tracking_residual_v1","specialists":{}}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(o.shape[-1],ad).cuda()
       initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero")
       m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       train=[collect(env,m,w,mgr,2110000+bi*100000+i*97) for i in range(16)]
       test=[collect(env,m,w,mgr,2310000+bi*100000+i*101) for i in range(8)]
       Ftr=flatten(train,"F");Fte=flatten(test,"F")
       Cte=flatten(test,"C");Htr=flatten(train,"H32");Hte=flatten(test,"H32")
       Mtr=flatten(train,"MC64");Mte=flatten(test,"MC64")
       q={"target_stats":{},"fits":{}}
       for j,name in enumerate(("Tracking","Angular","Orientation","Smoothness")):
        q["target_stats"][name]={
          "h32_mean":float(Hte[:,j].mean()),"h32_std":float(Hte[:,j].std()),"h32_var":float(Hte[:,j].var()),
          "mc64_mean":float(Mte[:,j].mean()),"mc64_std":float(Mte[:,j].std()),"mc64_var":float(Mte[:,j].var()),
          "h32_mc64_corr":corr(Hte[:,j],Mte[:,j])}
        sweeps={}
        for lam in LAMBDAS:
         W,b=ridge(Ftr,Htr[:,j:j+1],lam);P=Fte@W.T+b
         sweeps[str(lam)]={"h32_ev":ev(Hte[:,j],P[:,0]),"h32_mse":float(np.mean((Hte[:,j]-P[:,0])**2)),
           "h32_bias":float(np.mean(P[:,0]-Hte[:,j])),"weight_norm":float(np.linalg.norm(W))}
        Wm,bm=ridge(Ftr,Mtr[:,j:j+1],1.0);Pm=Fte@Wm.T+bm
        q["fits"][name]={"h32_ridge_sweep":sweeps,"mc64_ridge1_ev":ev(Mte[:,j],Pm[:,0]),
          "mc64_ridge1_mse":float(np.mean((Mte[:,j]-Pm[:,0])**2))}
        if name=="Tracking":
         W,b=ridge(Ftr,Htr[:,j:j+1],1.0);P=Fte@W.T+b
         q["tracking_command_bins"]=eval_by_bins(Hte[:,j],P[:,0],Cte)
       report["specialists"][lab]=q
      # aggregate across specialist policies
      agg={"target_stats":{},"fit_summary":{}}
      for name in ("Tracking","Angular","Orientation","Smoothness"):
       ts=[report["specialists"][lab]["target_stats"][name] for lab in ORDER]
       agg["target_stats"][name]={k:float(np.mean([x[k] for x in ts])) for k in ts[0]}
       fs={}
       for lam in LAMBDAS:
        vals=[report["specialists"][lab]["fits"][name]["h32_ridge_sweep"][str(lam)]["h32_ev"] for lab in ORDER]
        fs[str(lam)]={"h32_ev_mean":float(np.mean(vals)),"h32_ev_min":float(np.min(vals))}
       mc=[report["specialists"][lab]["fits"][name]["mc64_ridge1_ev"] for lab in ORDER]
       agg["fit_summary"][name]={"h32_ridge_sweep":fs,"mc64_ridge1_ev_mean":float(np.mean(mc))}
      report["aggregate"]=agg
      (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
      print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_actor_updating_reset_support():
    """Run former post_v2_t5_c25_actor_updating_reset_support.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S");G=.99;H=32;NENV=8;NSUP=12
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
     out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
     for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
     return out
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
     A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
     sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
     return sol[:-1].T,sol[-1]
    def flat_grad(loss,params,retain=True):
     gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
     return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,params)])
    def collect(env,m,w,mgr,seed,need_latent=False):
     from talon_rl.rewards.objectives import normalized_objective_vector
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
     ob=[];pre=[];old=[];rw=[];dn=[]
     with torch.no_grad():
      for _ in range(H):
       if need_latent:a,lp,u=m.act_with_preference_latent(cur,w)
       else:a=m.act_inference_with_preference(cur,w);lp=u=None
       nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda())
       if need_latent:pre.append(u);old.append(lp)
       cur=obs_tensor(nxt).cuda()
     rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
     fo=torch.cat(ob);fw=w.repeat(H,1)
     with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
     out={"obs":fo,"w":fw,"Y":Y,"F":F,"rt":rt,"dt":dt,"next_obs":cur}
     if need_latent:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
     return out
    def cosine_matrix(gs):
     M=[]
     for a in gs:
      row=[]
      for b in gs:row.append(float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12)))
      M.append(row)
     return M
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda()
      ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      report={"schema":"c25_actor_updating_reset_support_v1","updates":10,"support_size":NSUP,"actor_lr":1e-3,"ridge_lambda":1.0,"specialists":{}}
      for bi,lab in enumerate(ORDER):
       torch.manual_seed(61000+bi);np.random.seed(61000+bi)
       m=T4SharedActorCritic(o.shape[-1],ad).cuda()
       initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero")
       for n,p in m.named_parameters():
        if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
       actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
       opt=torch.optim.Adam(actor_params,lr=1e-3);w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       rows=[];torch.save({"model":m.state_dict(),"snapshot":0,"specialist":lab},OUT/f"{lab}_snap_0.pt")
       prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params]);prev_sol=None
       for uidx in range(1,11):
        mainb=collect(env,m,w,mgr,1810000+bi*100000+uidx*131,need_latent=True)
        Fs=[];Ys=[]
        for k in range(NSUP):
         q=collect(env,m,w,mgr,1910000+bi*100000+uidx*1000+k*37,need_latent=False)
         Fs.append(q["F"].cpu().numpy());Ys.append(q["Y"].cpu().numpy())
        F=np.concatenate(Fs);Y=np.concatenate(Ys);W,b=ridge(F,Y,1.0)
        with torch.no_grad():
         m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
         m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
         vt=m.value_with_preference(mainb["obs"],mainb["w"]).reshape(H,NENV,4)
         nv=m.value_with_preference(mainb["next_obs"],w)
         adv,_=vector_gae(mainb["rt"],vt,nv,mainb["dt"],lam=.95)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(mainb["obs"],mainb["w"],mainb["u"])-mainb["old"].detach())
        ratio_err=float((ratio-1).abs().max())
        if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
        av=adv.reshape(-1,4).detach()
        obj_grads=[]
        for j in range(4):
         li=-(ratio*av[:,j]).mean();obj_grads.append(flat_grad(li,actor_params,retain=True).detach())
        cm=cosine_matrix(obj_grads)
        loss=scalarized_late_weighted_ppo(ratio,av,mainb["w"])
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor_params,1.0);opt.step()
        actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
        sol=np.r_[W.reshape(-1),b];sdr=0.0 if prev_sol is None else float(np.linalg.norm(sol-prev_sol));prev_sol=sol
        row={"update":uidx,"ratio_maxerr":ratio_err,"actor_loss":float(loss.detach()),"actor_param_drift":float((actor_now-prev_actor).norm()),
             "objective_grad_norm":[float(g.norm()) for g in obj_grads],"objective_grad_cosine":cm,"head_solution_drift":sdr}
        prev_actor=actor_now.clone();rows.append(row)
        torch.save({"model":m.state_dict(),"snapshot":uidx,"specialist":lab},OUT/f"{lab}_snap_{uidx}.pt")
       report["specialists"][lab]=rows
      (OUT/"train.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","updates":10},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_actor_updating_reset_support_25():
    """Run former post_v2_t5_c25_actor_updating_reset_support_25.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S");G=.99;H=32;NENV=8;NSUP=12
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
     out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
     for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
     return out
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
     A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
     sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
     return sol[:-1].T,sol[-1]
    def flat_grad(loss,params,retain=True):
     gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
     return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,params)])
    def collect(env,m,w,mgr,seed,need_latent=False):
     from talon_rl.rewards.objectives import normalized_objective_vector
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
     ob=[];pre=[];old=[];rw=[];dn=[]
     with torch.no_grad():
      for _ in range(H):
       if need_latent:a,lp,u=m.act_with_preference_latent(cur,w)
       else:a=m.act_inference_with_preference(cur,w);lp=u=None
       nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda())
       if need_latent:pre.append(u);old.append(lp)
       cur=obs_tensor(nxt).cuda()
     rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
     fo=torch.cat(ob);fw=w.repeat(H,1)
     with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
     out={"obs":fo,"w":fw,"Y":Y,"F":F,"rt":rt,"dt":dt,"next_obs":cur}
     if need_latent:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
     return out
    def cosine_matrix(gs):
     M=[]
     for a in gs:
      row=[]
      for b in gs:row.append(float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12)))
      M.append(row)
     return M
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda()
      ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      report={"schema":"c25_actor_updating_reset_support_v1","updates":25,"support_size":NSUP,"actor_lr":1e-3,"ridge_lambda":1.0,"specialists":{}}
      for bi,lab in enumerate(ORDER):
       torch.manual_seed(61000+bi);np.random.seed(61000+bi)
       m=T4SharedActorCritic(o.shape[-1],ad).cuda()
       initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero")
       for n,p in m.named_parameters():
        if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
       actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
       opt=torch.optim.Adam(actor_params,lr=1e-3);w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       rows=[];torch.save({"model":m.state_dict(),"snapshot":0,"specialist":lab},OUT/f"{lab}_snap_0.pt")
       prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params]);prev_sol=None
       for uidx in range(1,26):
        mainb=collect(env,m,w,mgr,1810000+bi*100000+uidx*131,need_latent=True)
        Fs=[];Ys=[]
        for k in range(NSUP):
         q=collect(env,m,w,mgr,1910000+bi*100000+uidx*1000+k*37,need_latent=False)
         Fs.append(q["F"].cpu().numpy());Ys.append(q["Y"].cpu().numpy())
        F=np.concatenate(Fs);Y=np.concatenate(Ys);W,b=ridge(F,Y,1.0)
        with torch.no_grad():
         m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
         m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
         vt=m.value_with_preference(mainb["obs"],mainb["w"]).reshape(H,NENV,4)
         nv=m.value_with_preference(mainb["next_obs"],w)
         adv,_=vector_gae(mainb["rt"],vt,nv,mainb["dt"],lam=.95)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(mainb["obs"],mainb["w"],mainb["u"])-mainb["old"].detach())
        ratio_err=float((ratio-1).abs().max())
        if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
        av=adv.reshape(-1,4).detach()
        obj_grads=[]
        for j in range(4):
         li=-(ratio*av[:,j]).mean();obj_grads.append(flat_grad(li,actor_params,retain=True).detach())
        cm=cosine_matrix(obj_grads)
        loss=scalarized_late_weighted_ppo(ratio,av,mainb["w"])
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor_params,1.0);opt.step()
        actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
        sol=np.r_[W.reshape(-1),b];sdr=0.0 if prev_sol is None else float(np.linalg.norm(sol-prev_sol));prev_sol=sol
        row={"update":uidx,"ratio_maxerr":ratio_err,"actor_loss":float(loss.detach()),"actor_param_drift":float((actor_now-prev_actor).norm()),
             "objective_grad_norm":[float(g.norm()) for g in obj_grads],"objective_grad_cosine":cm,"head_solution_drift":sdr}
        prev_actor=actor_now.clone();rows.append(row)
        torch.save({"model":m.state_dict(),"snapshot":uidx,"specialist":lab},OUT/f"{lab}_snap_{uidx}.pt")
       report["specialists"][lab]=rows
      (OUT/"train.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","updates":25},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_angular_credit_confirm():
    """Run former post_v2_t5_c25_angular_credit_confirm.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23"
    _exec_stage_preamble("post_v2_t5_c25_eval", globals())
    SCALES=(0.0,0.5,1.0,2.0,4.0)
    def rollout_full(env,m,w,mgr,seed):
     from talon_rl.rewards.objectives import normalized_objective_vector
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];ang=[];obj=[];done=np.zeros(NENV,bool)
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,));obj.append(vec[:,1].mean())
       ang.append(float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[:,:2],dim=-1).mean()))
       done|=(te|tr).cpu().numpy();cur=obs_tensor(nxt).cuda()
     return {"ang_vel_xy":float(np.mean(ang)),"angular_objective":float(np.mean(obj)),"survival":float(1-done.mean())}
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/"A_snap_10.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
      gs=[grad_perturb(env,m,w,mgr,1,3010000+i*211) for i in range(4)]
      cos=[]
      for i in range(4):
       for j in range(i+1,4):cos.append(float(torch.dot(gs[i],gs[j])/(gs[i].norm()*gs[j].norm()+1e-12)))
      g=torch.stack(gs).mean(0);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base]);unit=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
      models={}
      for sc in SCALES:
       pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/"A_snap_10.pt",map_location="cuda",weights_only=False)["model"])
       paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")]
       if sc>0:setflat(paps,[p.detach().clone() for p in paps],unit*sc)
       pm.eval();models[sc]=pm
      suites=[]
      for ss in range(8):
       seed=3110000+ss*149;row={"suite":ss}
       for sc in SCALES:row[str(sc)]=rollout_full(env,models[sc],w,mgr,seed)
       suites.append(row)
      summary={}
      for sc in SCALES:
       if sc==0:continue
       da=[];dr=[]
       for row in suites:
        da.append(row[str(sc)]["ang_vel_xy"]-row["0.0"]["ang_vel_xy"])
        dr.append(row[str(sc)]["angular_objective"]-row["0.0"]["angular_objective"])
       summary[str(sc)]={"delta_ang_mean":float(np.mean(da)),"delta_ang_median":float(np.median(da)),"physical_correct_fraction":float(np.mean(np.array(da)<0)),
        "delta_objective_mean":float(np.mean(dr)),"objective_correct_fraction":float(np.mean(np.array(dr)>0))}
      out={"gradient_batch_pairwise_cosine":cos,"gradient_batch_cosine_mean":float(np.mean(cos)),"mean_grad_norm":float(g.norm()),"summary":summary,"suites":suites}
      (RUN/"angular_credit_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_angular_u0_control():
    """Run former post_v2_t5_c25_angular_u0_control.py stage."""
    from pathlib import Path
    p=Path(__file__).with_name("post_v2_t5_c25_angular_credit_confirm.py")
    s=p.read_text()
    s=s.replace('A_snap_10.pt','A_snap_0.pt')
    s=s.replace('angular_credit_confirm.json','angular_u0_control.json')
    s=s.replace('3010000+i*211','3210000+i*211')
    s=s.replace('3110000+ss*149','3310000+ss*149')
    exec(compile(s,str(p),'exec'))

def run_post_v2_t5_c25_causal_semantics():
    """Run former post_v2_t5_c25_causal_semantics.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23"
    ORDER=("T","A","O","S");IDX={"A":1,"O":2};G=.99;H=(1,2,4,8,16,32);NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
     o=0
     with torch.no_grad():
      for p,b in zip(ps,base):
       n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
     rows=[];done=np.zeros(NENV,bool)
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
       rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
       done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     return {str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H},float(1-done.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_causal_semantics_v1","snapshots":{}}
      for snap in (5,10):
       srows={"perturbations":[],"smoothness_baseline":[]}
       for lab in ("A","O"):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);j=IDX[lab]
        cur,_=env.reset(seed=2510000+snap*10000+ORDER.index(lab)*1000);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
        for _ in range(16):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
        loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach();base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,[p.detach().clone() for p in paps],delta);pm.eval()
        metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
        for ss in range(3):
         seed=2520000+snap*10000+ORDER.index(lab)*1000+ss;b,bs=rollout_metric(env,m,w,seed);p,ps=rollout_metric(env,pm,w,seed);suites.append({"baseline":b,"perturbed":p,"baseline_survival":bs,"perturbed_survival":ps})
        srows["perturbations"].append({"branch":lab,"grad_norm":float(g.norm()),"metric":metric,"suites":suites})
       sm=T4SharedActorCritic(od,ad).cuda();sm.load_state_dict(torch.load(RUN/f"S_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);sm.eval();sw=torch.tensor(PREFS["S"],device="cuda").repeat(NENV,1)
       for ss in range(6):
        seed=2530000+snap*10000+ss;met,surv=rollout_metric(env,sm,sw,seed);srows["smoothness_baseline"].append({"suite":ss,"metrics":met,"survival":surv})
       out["snapshots"][str(snap)]=srows
      (RUN/"causal_semantics.json").write_text(json.dumps(out,indent=2)+"\n")
      summary={}
      for snap,srows in out["snapshots"].items():
       q={}
       for row in srows["perturbations"]:
        metric=row["metric"];q[row["branch"]]={}
        for h in ("1","2","4","8","16","32"):
         ds=[x["perturbed"][h][metric]-x["baseline"][h][metric] for x in row["suites"]]
         q[row["branch"]][h]={"mean_delta":float(np.mean(ds)),"improve_fraction":float(np.mean(np.array(ds)<0))}
       q["S_survival"]=[x["survival"] for x in srows["smoothness_baseline"]]
       q["S_action_rate32"]=[x["metrics"]["32"]["action_rate"] for x in srows["smoothness_baseline"]]
       q["S_ang_vel32"]=[x["metrics"]["32"]["ang_vel_xy"] for x in srows["smoothness_baseline"]]
       q["S_tilt32"]=[x["metrics"]["32"]["tilt_deg"] for x in srows["smoothness_baseline"]]
       summary[snap]=q
      print(json.dumps(summary,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_causal_u10_confirm():
    """Run former post_v2_t5_c25_causal_u10_confirm.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23";NENV=8;G=.99
    PREFS={"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)];return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def setflat(ps,delta):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(delta[o:o+n].view_as(p));o+=n
    def rollout(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];rows=[]
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a);d=robot.data
       rows.append((float(torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1).mean()),float(tilt(d.root_quat_w).mean())));cur=obs_tensor(nxt).cuda()
     return rows
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={}
      for lab,j,mi in (("A",1,0),("O",2,1)):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_10.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       cur,_=env.reset(seed=2610000+j*1000);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
       for _ in range(16):
        with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
        nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
       with torch.no_grad():nv=m.value_with_preference(cur,w)
       adv,_=vector_gae(torch.stack(rw),torch.stack(val),nv,torch.stack(dn).bool(),lam=.95);fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1)
       ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach());aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
       loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach();basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
       pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_10.pt",map_location="cuda",weights_only=False)["model"]);setflat([p for n,p in pm.named_parameters() if n.startswith("actor_")],delta);pm.eval()
       ds={h:[] for h in (1,2,4,8,16,32)}
       for ss in range(10):
        seed=2620000+j*1000+ss;b=rollout(env,m,w,seed);p=rollout(env,pm,w,seed)
        for h in ds: ds[h].append(float(np.mean([x[mi] for x in p[:h]])-np.mean([x[mi] for x in b[:h]])))
       out[lab]={str(h):{"mean_delta":float(np.mean(v)),"median_delta":float(np.median(v)),"improve_fraction":float(np.mean(np.array(v)<0))} for h,v in ds.items()}
      (RUN/"causal_u10_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_causal_u25_confirm():
    """Run former post_v2_t5_c25_causal_u25_confirm.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";NENV=8;G=.99
    PREFS={"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)];return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def setflat(ps,delta):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(delta[o:o+n].view_as(p));o+=n
    def rollout(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];rows=[]
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a);d=robot.data
       rows.append((float(torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1).mean()),float(tilt(d.root_quat_w).mean())));cur=obs_tensor(nxt).cuda()
     return rows
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={}
      for lab,j,mi in (("A",1,0),("O",2,1)):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       cur,_=env.reset(seed=2610000+j*1000);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
       for _ in range(16):
        with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
        nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
       with torch.no_grad():nv=m.value_with_preference(cur,w)
       adv,_=vector_gae(torch.stack(rw),torch.stack(val),nv,torch.stack(dn).bool(),lam=.95);fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1)
       ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach());aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
       loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach();basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
       pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);setflat([p for n,p in pm.named_parameters() if n.startswith("actor_")],delta);pm.eval()
       ds={h:[] for h in (1,2,4,8,16,32)}
       for ss in range(10):
        seed=2620000+j*1000+ss;b=rollout(env,m,w,seed);p=rollout(env,pm,w,seed)
        for h in ds: ds[h].append(float(np.mean([x[mi] for x in p[:h]])-np.mean([x[mi] for x in b[:h]])))
       out[lab]={str(h):{"mean_delta":float(np.mean(v)),"median_delta":float(np.median(v)),"improve_fraction":float(np.mean(np.array(v)<0))} for h,v in ds.items()}
      (RUN/"causal_u25_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_eval():
    """Run former post_v2_t5_c25_eval.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23"
    ORDER=("T","A","O","S");IDX={"T":0,"A":1,"O":2,"S":3};G=.99;NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    SNAPS=(0,1,5,10)
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def flat(gs,ps):
     return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
     o=0
     with torch.no_grad():
      for p,b in zip(ps,base):
       n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def value_eval(env,m,w,mgr,seed):
     from talon_rl.rewards.objectives import normalized_objective_vector
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
     with torch.no_grad():
      for _ in range(64):
       V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
       D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
     R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H=ret(R,D,32);M=ret(R,D,None)
     return {"h32_ev":[ev(H[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(M[:,:,j],V[:,:,j]) for j in range(4)],
      "h32_bias":[float(np.mean(V[:,:,j]-H[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean())}
    def phys_roll(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];rows=[];done=np.zeros(NENV,bool)
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
       rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                    "tilt_deg":float(tilt(data.root_quat_w).mean())})
       done|=(te|tr).cpu().numpy();cur=obs_tensor(nxt).cuda()
     return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]},float(1-done.mean())
    def grad_perturb(env,m,w,mgr,j,seed):
     from talon_rl.models.foundations.four_objective import vector_gae
     from talon_rl.rewards.objectives import normalized_objective_vector
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
     for _ in range(16):
      with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
      nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
      ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(normalized_objective_vector(terms(raw,names),shape=(NENV,)),device="cuda")*env.unwrapped.step_dt)
      val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
     with torch.no_grad():nv=m.value_with_preference(cur,w)
     adv,_=vector_gae(torch.stack(rw),torch.stack(val),nv,torch.stack(dn).bool(),lam=.95)
     fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1)
     ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
     aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
     loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
     return g
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda()
      od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_eval_v1","value":{},"perturbations":[]}
      for snap in SNAPS:
       vals=[]
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        for ss in range(3):
         q=value_eval(env,m,w,mgr,2510000+snap*10000+bi*1000+ss*113);q.update({"policy":lab,"suite":ss});vals.append(q)
       out["value"][str(snap)]=vals
      for snap in (1,5,10):
       for lab in ("A","O"):
        bi=ORDER.index(lab);j=IDX[lab]
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);g=grad_perturb(env,m,w,mgr,j,2610000+snap*10000+bi*1000)
        aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base])
        delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"])
        paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,[p.detach().clone() for p in paps],delta);pm.eval()
        suites=[];metric="ang_vel_xy" if lab=="A" else "tilt_deg"
        for ss in range(3):
         seed=2710000+snap*10000+bi*1000+ss*127;b,sb=phys_roll(env,m,w,seed);p,sp=phys_roll(env,pm,w,seed)
         suites.append({"suite":ss,"baseline":b,"perturbed":p,"delta_metric":p[metric]-b[metric],"survival_base":sb,"survival_perturbed":sp})
        out["perturbations"].append({"snapshot":snap,"branch":lab,"metric":metric,"grad_norm":float(g.norm()),"suites":suites})
      agg={}
      for snap in SNAPS:
       vals=out["value"][str(snap)];h=sum([x["h32_ev"] for x in vals],[]);mc=sum([x["mc64_ev"] for x in vals],[]);b=sum([x["h32_bias"] for x in vals],[])
       by=[[] for _ in range(4)]
       for x in vals:
        for j,z in enumerate(x["h32_ev"]):by[j].append(z)
       agg[str(snap)]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),
        "h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(mc)),"mc64_negative_fraction":float(np.mean(np.array(mc)<0)),
        "h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(min(x["survival"] for x in vals))}
      out["aggregate"]=agg;(RUN/"eval.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({"value":agg,"perturbations":out["perturbations"]},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_freeze():
    """Run former post_v2_t5_c25_freeze.py stage."""
    """Freeze C25: the critic repair under actor learning."""
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.experiment_io.freeze import Freeze
    
    SRC = "scripts/rl/experiments/architectures/preference_architectures/post_v2"
    SOURCES = (f"{SRC}/post_v2_t5_c25_actor_updating_reset_support_25.py",
               f"{SRC}/post_v2_t5_c25_replay_audit_25.py",
               f"{SRC}/post_v2_t5_c25_causal_u25_confirm.py")
    
    
    class C25Freeze(Freeze):
        """Freeze C25: the critic repair under actor learning."""
    
        run = "post_v2_t5_c25_actor_updating25-2026-09-23"
        schema = "t5_c25_actor_updating_reset_support_synthesis_v1"
        status = "C25_CLOSED_CRITIC_REPAIR_SURVIVES_ACTOR_LEARNING_BUT_SEMANTIC_CAUSAL_CREDIT_DRIFTS"
        artifacts = ("train.json", "replay_audit.json", "causal_u25_confirm.json", "synthesis.json")
    
        def grad_stats(self, train, u):
            """Off-diagonal gradient cosines and drift, pooled over the four heads."""
            cs, adr, hdr, gn = [], [], [], []
            for lab in ("T", "A", "O", "S"):
                x = train["specialists"][lab][u - 1]
                m = np.array(x["objective_grad_cosine"])
                cs += m[np.triu_indices(4, 1)].tolist()
                adr.append(x["actor_param_drift"])
                hdr.append(x["head_solution_drift"])
                gn += x["objective_grad_norm"]
            return {"offdiag_cos_mean": float(np.mean(cs)),
                    "offdiag_cos_min": float(np.min(cs)),
                    "offdiag_cos_max": float(np.max(cs)),
                    "actor_param_drift_mean": float(np.mean(adr)),
                    "head_solution_drift_mean": float(np.mean(hdr)),
                    "grad_norm_mean": float(np.mean(gn))}
    
        def body(self):
            train = self.load("train.json")
            return {
                "evidence": {
                    "fresh_replay": self.load("replay_audit.json")["aggregate"],
                    "u25_gradient_geometry": self.grad_stats(train, 25),
                    "u10_gradient_geometry": self.grad_stats(train, 10),
                    "u25_causal": self.load("causal_u25_confirm.json"),
                },
                "findings": {
                    "critic": "Reset-diverse critic repair remains effective under actor learning through u25: fresh H32 EV ~0.189, MC64 EV ~0.416, Orientation H32 ~0.082, H32 mean |bias| ~0.061, MC64 negative fraction 0%.",
                    "tracking": "Tracking H32 is near neutral at u25 (~-0.002), consistent with C24's nonstructural short-horizon residual diagnosis.",
                    "gradient_separability": "Objective gradients remain distinct through u25; mean off-diagonal cosine stays near zero and PPO ratio invariance remains ~1e-5.",
                    "semantic_credit": "Physical causal semantics do not persist. At u25 Angular perturbation is wrong-sign on average for H1-H16 and Orientation is correct locally H1-H4 but wrong-sign for H8-H32.",
                    "safety": "One Smoothness fresh suite shows survival 0.875 at u10/u25, while other suites and dedicated additional Smoothness probes survive at 1.0. This is a warning but not a global collapse.",
                    "interpretation": "The critic-side foundation survives the coupled loop. The remaining blocker has moved downstream: distinct, numerically valid objective gradients cease to map reliably to intended closed-loop physical semantics as the actor policy evolves.",
                },
                "decision": {
                    "C25": "CLOSED — critic coupled-loop PASS / semantic-credit persistence FAIL",
                    "critic_repair_contract": "RETAIN",
                    "actor_distribution_shift_as_value_failure": "REJECTED under reset-diverse support",
                    "full_T4": "BLOCKED",
                    "V2": "OFF",
                    "next": "C26 actor-policy semantic-drift audit, diagnostic-only. Track A/O objective-gradient causal response across checkpoints u0/u1/u5/u10/u25 on matched states and horizons, while measuring state visitation/contact/action saturation changes. Determine when and why a still-distinct objective gradient loses physical meaning before changing reward, PPO, or critic again.",
                },
                "provenance": {
                    "train_sha256": self.sha(self.dir / "train.json"),
                    "replay_sha256": self.sha(self.dir / "replay_audit.json"),
                    "causal_u25_sha256": self.sha(self.dir / "causal_u25_confirm.json"),
                    "train_script_sha256": self.sha(SOURCES[0]),
                    "replay_script_sha256": self.sha(SOURCES[1]),
                    "causal_script_sha256": self.sha(SOURCES[2]),
                },
            }
    
        def manifest_extra(self):
            return {"sources": {s: {"sha256": self.sha(s)} for s in SOURCES}}
    
        def summary(self, syn):
            return {"status": syn["status"],
                    "u25": syn["evidence"]["fresh_replay"]["25"],
                    "grad": syn["evidence"]["u25_gradient_geometry"],
                    "next": syn["decision"]["next"]}
    
    
    if True:
        C25Freeze.main()

def run_post_v2_t5_c25_mid_causal_semantics():
    """Run former post_v2_t5_c25_mid_causal_semantics.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    ORDER=("T","A","O","S");IDX={"A":1,"O":2};G=.99;H=(1,2,4,8,16,32);NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
     o=0
     with torch.no_grad():
      for p,b in zip(ps,base):
       n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
     rows=[];done=np.zeros(NENV,bool)
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
       rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
       done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     return {str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H},float(1-done.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_causal_semantics_v1","snapshots":{}}
      for snap in (15,20):
       srows={"perturbations":[],"smoothness_baseline":[]}
       for lab in ("A","O"):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);j=IDX[lab]
        cur,_=env.reset(seed=2510000+snap*10000+ORDER.index(lab)*1000);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
        for _ in range(16):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
        loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach();base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,[p.detach().clone() for p in paps],delta);pm.eval()
        metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
        for ss in range(3):
         seed=2520000+snap*10000+ORDER.index(lab)*1000+ss;b,bs=rollout_metric(env,m,w,seed);p,ps=rollout_metric(env,pm,w,seed);suites.append({"baseline":b,"perturbed":p,"baseline_survival":bs,"perturbed_survival":ps})
        srows["perturbations"].append({"branch":lab,"grad_norm":float(g.norm()),"metric":metric,"suites":suites})
       sm=T4SharedActorCritic(od,ad).cuda();sm.load_state_dict(torch.load(RUN/f"S_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);sm.eval();sw=torch.tensor(PREFS["S"],device="cuda").repeat(NENV,1)
       for ss in range(6):
        seed=2530000+snap*10000+ss;met,surv=rollout_metric(env,sm,sw,seed);srows["smoothness_baseline"].append({"suite":ss,"metrics":met,"survival":surv})
       out["snapshots"][str(snap)]=srows
      (RUN/"mid_causal_semantics.json").write_text(json.dumps(out,indent=2)+"\n")
      summary={}
      for snap,srows in out["snapshots"].items():
       q={}
       for row in srows["perturbations"]:
        metric=row["metric"];q[row["branch"]]={}
        for h in ("1","2","4","8","16","32"):
         ds=[x["perturbed"][h][metric]-x["baseline"][h][metric] for x in row["suites"]]
         q[row["branch"]][h]={"mean_delta":float(np.mean(ds)),"improve_fraction":float(np.mean(np.array(ds)<0))}
       q["S_survival"]=[x["survival"] for x in srows["smoothness_baseline"]]
       q["S_action_rate32"]=[x["metrics"]["32"]["action_rate"] for x in srows["smoothness_baseline"]]
       q["S_ang_vel32"]=[x["metrics"]["32"]["ang_vel_xy"] for x in srows["smoothness_baseline"]]
       q["S_tilt32"]=[x["metrics"]["32"]["tilt_deg"] for x in srows["smoothness_baseline"]]
       summary[snap]=q
      print(json.dumps(summary,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_phase_eval():
    """Run former post_v2_t5_c25_phase_eval.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"phase12":ROOT/"runs/post_v2_t5_c24_phase_balanced12-2026-09-23","phase24":ROOT/"runs/post_v2_t5_c25_phase_balanced24-2026-09-23"}
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def segret(R,D,st,en):
     out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
     for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c24_phase_eval_v1","arms":{}}
      for arm,run in RUNS.items():
       ar={}
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);suites=[]
        for si in range(4):
         cur,_=env.reset(seed=1810000+bi*1000+si*97);cur=ot(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=ot(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V)
         Y0=segret(R,D,0,32);Y1=segret(R,D,32,64);MC=segret(R,D,0,64)
         suites.append({"suite":si,"first_ev":[ev(Y0[:,:,j],V[:32,:,j]) for j in range(4)],
                        "second_ev":[ev(Y1[:,:,j],V[32:,:,j]) for j in range(4)],
                        "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                        "survival":float(1-D.any(0).mean())})
        ar[lab]=suites
       out["arms"][arm]=ar
      agg={}
      for arm in RUNS:
       f=[[] for _ in range(4)];s=[[] for _ in range(4)];mc=[[] for _ in range(4)];sv=[]
       for lab in ORDER:
        for q in out["arms"][arm][lab]:
         sv.append(q["survival"])
         for j in range(4):f[j].append(q["first_ev"][j]);s[j].append(q["second_ev"][j]);mc[j].append(q["mc64_ev"][j])
       both=[v for arr in f+s for v in arr]
       agg[arm]={"first_h32_mean":float(np.mean(f)),"second_h32_mean":float(np.mean(s)),"combined_h32_mean":float(np.mean(both)),
                 "first_by_head":[float(np.mean(x)) for x in f],"second_by_head":[float(np.mean(x)) for x in s],
                 "first_negative_fraction":float(np.mean(np.array(f)<0)),"second_negative_fraction":float(np.mean(np.array(s)<0)),
                 "combined_negative_fraction":float(np.mean(np.array(both)<0)),
                 "mc64_mean":float(np.mean(mc)),"mc64_by_head":[float(np.mean(x)) for x in mc],
                 "mc64_negative_fraction":float(np.mean(np.array(mc)<0)),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg
      p=RUNS["phase24"]/"phase_eval.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_replay25():
    """Run former post_v2_t5_c25_replay25.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    ORDER=("T","A","O","S");SNAPS=(0,1,5,10,25);G=.99;NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_replay_v1","snapshots":list(SNAPS),"specialists":{}}
      for bi,lab in enumerate(ORDER):
       rows=[];w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       for snap in SNAPS:
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        suite_rows=[]
        for ss in range(3):
         cur,_=env.reset(seed=2410000+bi*10000+ss*503);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H=ret(R,D,32);M=ret(R,D,None)
         suite_rows.append({"h32_ev":[ev(H[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(M[:,:,j],V[:,:,j]) for j in range(4)],"h32_bias":[float(np.mean(V[:,:,j]-H[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean())})
        rows.append({"snapshot":snap,"suites":suite_rows})
       out["specialists"][lab]=rows
      agg={}
      for snap in SNAPS:
       h=[];m=[];b=[];sv=[];by=[[] for _ in range(4)]
       for lab in ORDER:
        row=next(r for r in out["specialists"][lab] if r["snapshot"]==snap)
        for q in row["suites"]:
         h+=q["h32_ev"];m+=q["mc64_ev"];b+=q["h32_bias"];sv.append(q["survival"])
         for j,x in enumerate(q["h32_ev"]):by[j].append(x)
       agg[str(snap)]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),"h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(m)),"mc64_negative_fraction":float(np.mean(np.array(m)<0)),"h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg;(RUN/"replay_audit25.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_replay25_audit():
    """Run former post_v2_t5_c25_replay25_audit.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    ORDER=("T","A","O","S");SNAPS=(0,1,5,10,25);G=.99;NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_replay_v1","snapshots":list(SNAPS),"specialists":{}}
      for bi,lab in enumerate(ORDER):
       rows=[];w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       for snap in SNAPS:
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        suite_rows=[]
        for ss in range(3):
         cur,_=env.reset(seed=2410000+bi*10000+ss*503);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H=ret(R,D,32);M=ret(R,D,None)
         suite_rows.append({"h32_ev":[ev(H[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(M[:,:,j],V[:,:,j]) for j in range(4)],"h32_bias":[float(np.mean(V[:,:,j]-H[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean())})
        rows.append({"snapshot":snap,"suites":suite_rows})
       out["specialists"][lab]=rows
      agg={}
      for snap in SNAPS:
       h=[];m=[];b=[];sv=[];by=[[] for _ in range(4)]
       for lab in ORDER:
        row=next(r for r in out["specialists"][lab] if r["snapshot"]==snap)
        for q in row["suites"]:
         h+=q["h32_ev"];m+=q["mc64_ev"];b+=q["h32_bias"];sv.append(q["survival"])
         for j,x in enumerate(q["h32_ev"]):by[j].append(x)
       agg[str(snap)]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),"h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(m)),"mc64_negative_fraction":float(np.mean(np.array(m)<0)),"h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg;(RUN/"replay_audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_replay_audit():
    """Run former post_v2_t5_c25_replay_audit.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23"
    ORDER=("T","A","O","S");SNAPS=(0,1,5,10);G=.99;NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_replay_v1","snapshots":list(SNAPS),"specialists":{}}
      for bi,lab in enumerate(ORDER):
       rows=[];w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       for snap in SNAPS:
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        suite_rows=[]
        for ss in range(3):
         cur,_=env.reset(seed=2410000+bi*10000+ss*503);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H=ret(R,D,32);M=ret(R,D,None)
         suite_rows.append({"h32_ev":[ev(H[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(M[:,:,j],V[:,:,j]) for j in range(4)],"h32_bias":[float(np.mean(V[:,:,j]-H[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean())})
        rows.append({"snapshot":snap,"suites":suite_rows})
       out["specialists"][lab]=rows
      agg={}
      for snap in SNAPS:
       h=[];m=[];b=[];sv=[];by=[[] for _ in range(4)]
       for lab in ORDER:
        row=next(r for r in out["specialists"][lab] if r["snapshot"]==snap)
        for q in row["suites"]:
         h+=q["h32_ev"];m+=q["mc64_ev"];b+=q["h32_bias"];sv.append(q["survival"])
         for j,x in enumerate(q["h32_ev"]):by[j].append(x)
       agg[str(snap)]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),"h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(m)),"mc64_negative_fraction":float(np.mean(np.array(m)<0)),"h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg;(RUN/"replay_audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_replay_audit_25():
    """Run former post_v2_t5_c25_replay_audit_25.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    ORDER=("T","A","O","S");SNAPS=(0,1,5,10,25);G=.99;NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
     out=np.zeros_like(R)
     if seg is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_replay_v1","snapshots":list(SNAPS),"specialists":{}}
      for bi,lab in enumerate(ORDER):
       rows=[];w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
       for snap in SNAPS:
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        suite_rows=[]
        for ss in range(3):
         cur,_=env.reset(seed=2410000+bi*10000+ss*503);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H=ret(R,D,32);M=ret(R,D,None)
         suite_rows.append({"h32_ev":[ev(H[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(M[:,:,j],V[:,:,j]) for j in range(4)],"h32_bias":[float(np.mean(V[:,:,j]-H[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean())})
        rows.append({"snapshot":snap,"suites":suite_rows})
       out["specialists"][lab]=rows
      agg={}
      for snap in SNAPS:
       h=[];m=[];b=[];sv=[];by=[[] for _ in range(4)]
       for lab in ORDER:
        row=next(r for r in out["specialists"][lab] if r["snapshot"]==snap)
        for q in row["suites"]:
         h+=q["h32_ev"];m+=q["mc64_ev"];b+=q["h32_bias"];sv.append(q["survival"])
         for j,x in enumerate(q["h32_ev"]):by[j].append(x)
       agg[str(snap)]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),"h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(m)),"mc64_negative_fraction":float(np.mean(np.array(m)<0)),"h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
      out["aggregate"]=agg;(RUN/"replay_audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_semantic_perturb():
    """Run former post_v2_t5_c25_semantic_perturb.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23"
    ORDER=("T","A","O","S");IDX={"T":0,"A":1,"O":2,"S":3};G=.99;H=16;NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
     o=0
     with torch.no_grad():
      for p,b in zip(ps,base):
       n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout(env,m,w,seed):
     robot=env.unwrapped.scene["robot"];cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
     rows=[];done=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
       rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                    "tilt_deg":float(tilt(data.root_quat_w).mean()),
                    "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
       done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     return rows,float(1-done.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_semantic_perturb_v1","rows":[]}
      for snap in (5,10):
       for lab in ("A","O"):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);j=IDX[lab]
        cur,_=env.reset(seed=2510000+snap*10000+j*1000);cur=obs_tensor(cur).cuda()
        ob=[];pre=[];old=[];rw=[];val=[];dn=[]
        for _ in range(H):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(H,1)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
        loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean()
        g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
        basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);pm.eval()
        paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta)
        metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
        for ss in range(3):
         seed=2610000+snap*10000+j*1000+ss
         b,bs=rollout(env,m,w,seed);p,ps=rollout(env,pm,w,seed)
         q={}
         for h in (1,2,4,8,16,32):
          q[str(h)]={"baseline":float(np.mean([x[metric] for x in b[:h]])),
                     "perturbed":float(np.mean([x[metric] for x in p[:h]]))}
         suites.append({"suite":ss,"metric":metric,"horizons":q,"baseline_survival":bs,"perturbed_survival":ps})
        out["rows"].append({"snapshot":snap,"branch":lab,"grad_norm":float(g.norm()),"suites":suites})
      (RUN/"semantic_perturb.json").write_text(json.dumps(out,indent=2)+"\n")
      for row in out["rows"]:
       print("\n",row["branch"],"u",row["snapshot"])
       for h in ("1","2","4","8","16","32"):
        ds=[s["horizons"][h]["perturbed"]-s["horizons"][h]["baseline"] for s in row["suites"]]
        print(h,"delta",round(float(np.mean(ds)),5),"improve",round(float(np.mean(np.array(ds)<0)),2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_semantic_u25():
    """Run former post_v2_t5_c25_semantic_u25.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    ORDER=("T","A","O","S");IDX={"T":0,"A":1,"O":2,"S":3};G=.99;H=16;NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
     o=0
     with torch.no_grad():
      for p,b in zip(ps,base):
       n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout(env,m,w,seed):
     robot=env.unwrapped.scene["robot"];cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
     rows=[];done=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
       rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                    "tilt_deg":float(tilt(data.root_quat_w).mean()),
                    "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
       done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     return rows,float(1-done.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_semantic_perturb_v1","rows":[]}
      for snap in (25,):
       for lab in ("A","O"):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);j=IDX[lab]
        cur,_=env.reset(seed=2510000+snap*10000+j*1000);cur=obs_tensor(cur).cuda()
        ob=[];pre=[];old=[];rw=[];val=[];dn=[]
        for _ in range(H):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(H,1)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
        loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean()
        g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
        basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);pm.eval()
        paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta)
        metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
        for ss in range(3):
         seed=2610000+snap*10000+j*1000+ss
         b,bs=rollout(env,m,w,seed);p,ps=rollout(env,pm,w,seed)
         q={}
         for h in (1,2,4,8,16,32):
          q[str(h)]={"baseline":float(np.mean([x[metric] for x in b[:h]])),
                     "perturbed":float(np.mean([x[metric] for x in p[:h]]))}
         suites.append({"suite":ss,"metric":metric,"horizons":q,"baseline_survival":bs,"perturbed_survival":ps})
        out["rows"].append({"snapshot":snap,"branch":lab,"grad_norm":float(g.norm()),"suites":suites})
      (RUN/"semantic_u25.json").write_text(json.dumps(out,indent=2)+"\n")
      for row in out["rows"]:
       print("\n",row["branch"],"u",row["snapshot"])
       for h in ("1","2","4","8","16","32"):
        ds=[s["horizons"][h]["perturbed"]-s["horizons"][h]["baseline"] for s in row["suites"]]
        print(h,"delta",round(float(np.mean(ds)),5),"improve",round(float(np.mean(np.array(ds)<0)),2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_tracking_confirm():
    """Run former post_v2_t5_c25_tracking_confirm.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_phase_balanced24-2026-09-23";ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def segret(R,D,st,en):
     out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
     for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      vals=[];rows=[]
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1)
       for si in range(8):
        cur,_=env.reset(seed=1910000+bi*1000+si*97);cur=ot(cur).cuda();R=[];D=[];V=[]
        with torch.no_grad():
         for _ in range(64):
          V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
          nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
          R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);Y1=segret(R,D,32,64)
        e=[ev(Y1[:,:,j],V[32:,:,j]) for j in range(4)]
        vals.append(e[0]);rows.append({"specialist":lab,"suite":si,"second_ev":e})
      a=np.array(vals,float);out={"schema":"c25_tracking_confirm_v1","n":len(a),"tracking_second_mean":float(a.mean()),"std":float(a.std(ddof=1)),
        "sem":float(a.std(ddof=1)/np.sqrt(len(a))),"ci95_normal":[float(a.mean()-1.96*a.std(ddof=1)/np.sqrt(len(a))),float(a.mean()+1.96*a.std(ddof=1)/np.sqrt(len(a)))],
        "negative_fraction":float(np.mean(a<0)),"rows":rows}
      (RUN/"tracking_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_u10_perturb_confirm():
    """Run former post_v2_t5_c25_u10_perturb_confirm.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23"
    _exec_stage_preamble("post_v2_t5_c25_eval", globals())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={}
      for lab in ("A","O"):
       bi=ORDER.index(lab);j=IDX[lab];m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_10.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);g=grad_perturb(env,m,w,mgr,j,2810000+bi*1000)
       aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
       pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_10.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,[p.detach().clone() for p in paps],delta);pm.eval();metric="ang_vel_xy" if lab=="A" else "tilt_deg";rows=[]
       for ss in range(8):
        seed=2910000+bi*1000+ss*137;b,sb=phys_roll(env,m,w,seed);p,sp=phys_roll(env,pm,w,seed);rows.append({"suite":ss,"delta":p[metric]-b[metric],"base_survival":sb,"pert_survival":sp})
       ds=[x["delta"] for x in rows];out[lab]={"metric":metric,"mean_delta":float(np.mean(ds)),"median_delta":float(np.median(ds)),"correct_fraction":float(np.mean(np.array(ds)<0)),"rows":rows}
      (RUN/"u10_perturb_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_u10_safety_confirm():
    """Run former post_v2_t5_c25_u10_safety_confirm.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23";ORDER=("T","A","O","S");NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      out={}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_10.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);vals=[]
       for ss in range(8):
        cur,_=env.reset(seed=2810000+bi*10000+ss*503);cur=obs_tensor(cur).cuda();done=np.zeros(NENV,bool)
        with torch.no_grad():
         for _ in range(64):
          a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);done|=(te|tr).cpu().numpy();cur=obs_tensor(nxt).cuda()
        vals.append(float(1-done.mean()))
       out[lab]={"survival_by_suite":vals,"mean":float(np.mean(vals)),"min":float(np.min(vals)),"failure_suite_fraction":float(np.mean(np.array(vals)<1))}
      (RUN/"u10_safety_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_u25_causal_semantics():
    """Run former post_v2_t5_c25_u25_causal_semantics.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    ORDER=("T","A","O","S");IDX={"A":1,"O":2};G=.99;H=(1,2,4,8,16,32);NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
     o=0
     with torch.no_grad():
      for p,b in zip(ps,base):
       n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
     rows=[];done=np.zeros(NENV,bool)
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
       rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
       done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     return {str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H},float(1-done.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c25_causal_semantics_v1","snapshots":{}}
      for snap in (25,):
       srows={"perturbations":[],"smoothness_baseline":[]}
       for lab in ("A","O"):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);j=IDX[lab]
        cur,_=env.reset(seed=2510000+snap*10000+ORDER.index(lab)*1000);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
        for _ in range(16):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
        loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach();base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,[p.detach().clone() for p in paps],delta);pm.eval()
        metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
        for ss in range(3):
         seed=2520000+snap*10000+ORDER.index(lab)*1000+ss;b,bs=rollout_metric(env,m,w,seed);p,ps=rollout_metric(env,pm,w,seed);suites.append({"baseline":b,"perturbed":p,"baseline_survival":bs,"perturbed_survival":ps})
        srows["perturbations"].append({"branch":lab,"grad_norm":float(g.norm()),"metric":metric,"suites":suites})
       sm=T4SharedActorCritic(od,ad).cuda();sm.load_state_dict(torch.load(RUN/f"S_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);sm.eval();sw=torch.tensor(PREFS["S"],device="cuda").repeat(NENV,1)
       for ss in range(6):
        seed=2530000+snap*10000+ss;met,surv=rollout_metric(env,sm,sw,seed);srows["smoothness_baseline"].append({"suite":ss,"metrics":met,"survival":surv})
       out["snapshots"][str(snap)]=srows
      (RUN/"u25_causal_semantics.json").write_text(json.dumps(out,indent=2)+"\n")
      summary={}
      for snap,srows in out["snapshots"].items():
       q={}
       for row in srows["perturbations"]:
        metric=row["metric"];q[row["branch"]]={}
        for h in ("1","2","4","8","16","32"):
         ds=[x["perturbed"][h][metric]-x["baseline"][h][metric] for x in row["suites"]]
         q[row["branch"]][h]={"mean_delta":float(np.mean(ds)),"improve_fraction":float(np.mean(np.array(ds)<0))}
       q["S_survival"]=[x["survival"] for x in srows["smoothness_baseline"]]
       q["S_action_rate32"]=[x["metrics"]["32"]["action_rate"] for x in srows["smoothness_baseline"]]
       q["S_ang_vel32"]=[x["metrics"]["32"]["ang_vel_xy"] for x in srows["smoothness_baseline"]]
       q["S_tilt32"]=[x["metrics"]["32"]["tilt_deg"] for x in srows["smoothness_baseline"]]
       summary[snap]=q
      print(json.dumps(summary,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c25_u25_safety_confirm():
    """Run former post_v2_t5_c25_u25_safety_confirm.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";ORDER=("T","A","O","S");NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      out={}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);vals=[]
       for ss in range(8):
        cur,_=env.reset(seed=2810000+bi*10000+ss*503);cur=obs_tensor(cur).cuda();done=np.zeros(NENV,bool)
        with torch.no_grad():
         for _ in range(64):
          a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);done|=(te|tr).cpu().numpy();cur=obs_tensor(nxt).cuda()
        vals.append(float(1-done.mean()))
       out[lab]={"survival_by_suite":vals,"mean":float(np.mean(vals)),"min":float(np.min(vals)),"failure_suite_fraction":float(np.mean(np.array(vals)<1))}
      (RUN/"u25_safety_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c26_actor_coupled_train():
    """Run former post_v2_t5_c26_actor_coupled_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
     out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
     for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
     return out
    def ridge_fit(F,Y,l2):
     Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy();A=np.c_[Fc,np.ones(len(Fc))]
     I=np.eye(A.shape[1]);I[-1,-1]=0;sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
     return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def select_diverse(S,k):
     X=np.asarray(S,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
     if len(X)<=k:return list(range(len(X)))
     sel=[int(np.argmax(np.mean(X*X,axis=1)))];mind=np.mean((X-X[sel[0]])**2,axis=1)
     while len(sel)<k:
      mind[sel]=-1;q=int(np.argmax(mind));sel.append(q);mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
     return sorted(sel)
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--coupling-updates",type=int,default=10)
     ap.add_argument("--support-size",type=int,default=24);ap.add_argument("--horizon",type=int,default=32);ap.add_argument("--num-envs",type=int,default=8)
     ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--actor-lr",type=float,default=1e-3)
     ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
     args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
     from isaaclab.app import AppLauncher
     saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
     env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      all_logs={}
      for bi,lab in enumerate(ORDER):
       torch.manual_seed(61000+bi);np.random.seed(61000+bi)
       m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
       for p in m.parameters():p.requires_grad_(False)
       actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
       for p in actor_params:p.requires_grad_(True)
       opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
       w=torch.tensor(np.repeat(PREFS[lab][None,:],args.num_envs,axis=0),device="cuda")
       pool=[];summ=[];candidate_count=0
       def collect(seed:int,need_actor:bool):
        nonlocal candidate_count
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();obs=[];pre=[];old=[];rw=[];dn=[];cmd=[];allobs=[];next32=None
        for t in range(args.horizon*2):
         allobs.append(cur)
         with torch.no_grad():
          if need_actor and t<args.horizon:
           a,lp,u=m.act_with_preference_latent(cur,w);obs.append(cur);pre.append(u);old.append(lp)
          else:
           a=m.act_inference_with_preference(cur,w)
          cmd.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         rw.append(torch.tensor(normalized_objective_vector(terms(raw,names),shape=(args.num_envs,)),device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda())
         cur=ot(nxt).cuda()
         if t==args.horizon-1:next32=cur.detach()
        phase=0 if candidate_count%2==0 else args.horizon;candidate_count+=1;pe=phase+args.horizon
        # F and Y are taken from the exact same trajectory/action path.
        fo=torch.cat(allobs[phase:pe]);fw=w.repeat(args.horizon,1);rt=torch.stack(rw[phase:pe]);dt=torch.stack(dn[phase:pe]).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
        with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
        C=np.concatenate(cmd[phase:pe],0);summary=np.r_[C.mean(0),C.std(0),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy()]
        actor_batch=None
        if need_actor:
         actor_batch=(torch.cat(obs),torch.cat(pre),torch.cat(old),torch.stack(rw[:args.horizon]),torch.stack(dn[:args.horizon]).bool(),next32)
        return F.detach().cpu(),Y.detach().cpu(),summary.astype(np.float32),phase,actor_batch
       # prefill repaired critic support
       for j in range(args.support_size):
        F,Y,S,phase,_=collect(2010000+bi*10000+j*97,False);pool.append((F,Y));summ.append(S)
       idx=select_diverse(summ,args.support_size);FF=torch.cat([pool[i][0] for i in idx]).cuda();YY=torch.cat([pool[i][1] for i in idx]).cuda();W,b=ridge_fit(FF,YY,args.ridge_lambda)
       with torch.no_grad():m.critic_head.weight.copy_(W);m.critic_head.bias.copy_(b)
       def save(tag):torch.save({"model":m.state_dict(),"specialist":lab,"snapshot":tag,"stage":"actor_coupled"},args.output.parent/f"{lab}_snap_{tag}.pt")
       save(0);logs=[];prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params]);prevW=W.clone()
       for u in range(1,args.coupling_updates+1):
        F,Y,S,phase,batch=collect(2110000+bi*10000+u*97,True);pool.append((F,Y));summ.append(S)
        idx=select_diverse(summ,args.support_size);FF=torch.cat([pool[i][0] for i in idx]).cuda();YY=torch.cat([pool[i][1] for i in idx]).cuda();W,b=ridge_fit(FF,YY,args.ridge_lambda)
        with torch.no_grad():m.critic_head.weight.copy_(W);m.critic_head.bias.copy_(b)
        fo,fu,fold,rt,dt,next32=batch;fw=w.repeat(args.horizon,1)
        with torch.no_grad():
         vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4);nv=m.value_with_preference(next32,w);adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach());rmax=float((ratio-1).abs().max())
        if rmax>1e-4:raise RuntimeError(f"ratio invariant {rmax}")
        loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
        logs.append({"update":u,"phase_start":phase,"actor_loss":float(loss.detach()),"ratio_max_abs_err":rmax,
          "actor_step_norm":float((actor_now-prev_actor).norm()),"head_solution_drift":float((W-prevW).norm()),
          "support_age_span":int(idx[-1]-idx[0]+1),"selected_indices":idx})
        prev_actor=actor_now.clone();prevW=W.clone();save(u)
       all_logs[lab]=logs
      out={"schema":"t5_c26_actor_coupled_v1","status":"COMPLETE","prefill_support":args.support_size,"coupling_updates":args.coupling_updates,
           "support_mode":"independent_reset_phase_balanced_diverse","ridge_lambda":args.ridge_lambda,"actor_lr":args.actor_lr,"specialist_logs":all_logs}
      args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","coupling_updates":args.coupling_updates},indent=2))
     except BaseException as e:
      args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c26_angular_consensus():
    """Run former post_v2_t5_c26_angular_consensus.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c26_angular_credit-2026-09-23"
    _exec_stage_preamble("post_v2_t5_c25_eval", globals())
    SCALES=(0.0,1.0,2.0,4.0)
    def rollout_full(env,m,w,mgr,seed):
     from talon_rl.rewards.objectives import normalized_objective_vector
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];ang=[];obj=[];done=np.zeros(NENV,bool)
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,));obj.append(vec[:,1].mean())
       ang.append(float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[:,:2],dim=-1).mean()))
       done|=(te|tr).cpu().numpy();cur=obs_tensor(nxt).cuda()
     return {"ang_vel_xy":float(np.mean(ang)),"angular_objective":float(np.mean(obj)),"survival":float(1-done.mean())}
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c26_angular_consensus_v1","snapshots":{}}
      for snap in (10,25):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
       gs=[grad_perturb(env,m,w,mgr,1,3210000+snap*10000+i*211) for i in range(8)]
       cos=[]
       for i in range(8):
        for j in range(i+1,8):cos.append(float(torch.dot(gs[i],gs[j])/(gs[i].norm()*gs[j].norm()+1e-12)))
       g=torch.stack(gs).mean(0);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base]);unit=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
       models={}
       for sc in SCALES:
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"])
        paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")]
        if sc>0:setflat(paps,[p.detach().clone() for p in paps],unit*sc)
        pm.eval();models[sc]=pm
       suites=[]
       for ss in range(8):
        seed=3310000+snap*10000+ss*149;row={"suite":ss}
        for sc in SCALES:row[str(sc)]=rollout_full(env,models[sc],w,mgr,seed)
        suites.append(row)
       summary={}
       for sc in SCALES:
        if sc==0:continue
        da=[];dr=[]
        for row in suites:
         da.append(row[str(sc)]["ang_vel_xy"]-row["0.0"]["ang_vel_xy"]);dr.append(row[str(sc)]["angular_objective"]-row["0.0"]["angular_objective"])
        summary[str(sc)]={"delta_ang_mean":float(np.mean(da)),"delta_ang_median":float(np.median(da)),"physical_correct_fraction":float(np.mean(np.array(da)<0)),
          "delta_objective_mean":float(np.mean(dr)),"objective_correct_fraction":float(np.mean(np.array(dr)>0))}
       out["snapshots"][str(snap)]={"gradient_batch_pairwise_cosine_mean":float(np.mean(cos)),"gradient_batch_pairwise_cosine_std":float(np.std(cos)),
        "gradient_negative_pair_fraction":float(np.mean(np.array(cos)<0)),"mean_gradient_norm":float(g.norm()),"summary":summary}
      (OUT/"consensus.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c26_angular_credit_stability():
    """Run former post_v2_t5_c26_angular_credit_stability.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c26_angular_credit-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    SNAPS=(0,5,10,25);G=.99;H=32;NENV=8;J=1
    WREF=np.array([.1,.7,.1,.1],np.float32)
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def corr(a,b):
     a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
     return float(np.corrcoef(a,b)[0,1])
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      report={"schema":"c26_angular_credit_stability_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       w=torch.tensor(WREF,device="cuda").repeat(NENV,1);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
       batchrows=[];grads=[]
       for bi in range(8):
        cur,_=env.reset(seed=3110000+snap*10000+bi*211);cur=obs_tensor(cur).cuda()
        ob=[];pre=[];old=[];rw=[];val=[];dn=[]
        for _ in range(H):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,ret=vector_gae(rt,vt,nv,dt,lam=.95)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(H,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        av=adv.reshape(-1,4)[:,J].detach();loss=-(ratio*av).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach();grads.append(g)
        target=ret.reshape(-1,4)[:,J].detach().cpu().numpy();pred=vt.reshape(-1,4)[:,J].detach().cpu().numpy();rew=rt.reshape(-1,4)[:,J].detach().cpu().numpy()
        batchrows.append({"batch":bi,"adv_mean":float(av.mean()),"adv_std":float(av.std()),"adv_rms":float(torch.sqrt(torch.mean(av*av))),
          "adv_absmean":float(av.abs().mean()),"adv_mean_over_std":float(abs(av.mean())/(av.std()+1e-12)),
          "grad_norm":float(g.norm()),"value_ev":float(1-np.var(target-pred)/(np.var(target)+1e-12)),
          "reward_adv_corr":corr(rew,av.cpu().numpy()),"return_adv_corr":corr(target,av.cpu().numpy())})
       cos=[]
       for i in range(len(grads)):
        for j in range(i+1,len(grads)):cos.append(float(torch.dot(grads[i],grads[j])/(grads[i].norm()*grads[j].norm()+1e-12)))
       report["snapshots"][str(snap)]={"batches":batchrows,"summary":{
        "adv_std_mean":float(np.mean([x["adv_std"] for x in batchrows])),
        "adv_rms_mean":float(np.mean([x["adv_rms"] for x in batchrows])),
        "adv_snr_mean":float(np.mean([x["adv_mean_over_std"] for x in batchrows])),
        "grad_norm_mean":float(np.mean([x["grad_norm"] for x in batchrows])),
        "grad_norm_cv":float(np.std([x["grad_norm"] for x in batchrows])/(np.mean([x["grad_norm"] for x in batchrows])+1e-12)),
        "gradient_pairwise_cos_mean":float(np.mean(cos)),"gradient_pairwise_cos_std":float(np.std(cos)),
        "gradient_pairwise_cos_negative_fraction":float(np.mean(np.array(cos)<0)),
        "value_ev_mean":float(np.mean([x["value_ev"] for x in batchrows])),
        "reward_adv_corr_mean":float(np.mean([x["reward_adv_corr"] for x in batchrows])),
        "return_adv_corr_mean":float(np.mean([x["return_adv_corr"] for x in batchrows]))}}
      (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
      print(json.dumps({k:v["summary"] for k,v in report["snapshots"].items()},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c26_conditional_decomp():
    """Run former post_v2_t5_c26_conditional_decomp.py stage."""
    from pathlib import Path
    import sys,json,math,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c26_angular_credit-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    SNAPS=(0,10,25);G=.99;H=32;NENV=8;NB=8;J=1
    WREF=np.array([.1,.7,.1,.1],np.float32)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps): return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cosine(a,b): return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def bentropy(vals):
     vals=np.asarray(vals); p=float(np.mean(vals>0))
     if p<=0 or p>=1:return 0.0
     return float(-(p*math.log2(p)+(1-p)*math.log2(1-p)))
    def tilt_deg(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def bins(ctx):
     # returns dict name->boolean mask, all on device
     vx=ctx["cmd_vx"]; yaw=ctx["cmd_yaw"]; wx=ctx["wx"]; wy=ctx["wy"]; amag=ctx["amag"]; tilt=ctx["tilt"]
     return {
      "cmd_vx_neg":vx<-.15,"cmd_vx_mid":vx.abs()<=.15,"cmd_vx_pos":vx>.15,
      "cmd_yaw_neg":yaw<-.15,"cmd_yaw_mid":yaw.abs()<=.15,"cmd_yaw_pos":yaw>.15,
      "wx_neg":wx<0,"wx_pos":wx>=0,"wy_neg":wy<0,"wy_pos":wy>=0,
      "angmag_low":amag<1.0,"angmag_high":amag>=1.0,
      "tilt_low":tilt<5.0,"tilt_high":tilt>=5.0,
     }
    def pair_stats(gs):
     if len(gs)<2:return {"n_grad":len(gs),"cos_mean":None,"cos_std":None,"negative_fraction":None}
     c=[cosine(gs[i],gs[j]) for i in range(len(gs)) for j in range(i+1,len(gs))]
     return {"n_grad":len(gs),"cos_mean":float(np.mean(c)),"cos_std":float(np.std(c)),"negative_fraction":float(np.mean(np.array(c)<0))}
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      report={"schema":"c26_conditional_decomposition_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
       bin_grads={};bin_advs={};bin_counts={};reset_grads=[];reset_adv=[]
       for bi in range(NB):
        cur,_=env.reset(seed=3410000+snap*10000+bi*211);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];val=[];dn=[];cx={k:[] for k in ("cmd_vx","cmd_yaw","wx","wy","amag","tilt")}
        for _ in range(H):
         data=robot.data
         try:cmd=env.unwrapped.command_manager.get_command("base_velocity")
         except Exception:
          cmd=getattr(env.unwrapped,"v_command_buf",torch.zeros((NENV,3),device="cuda"))
         av=data.root_ang_vel_b
         cx["cmd_vx"].append(cmd[:,0].detach());cx["cmd_yaw"].append(cmd[:,2].detach())
         cx["wx"].append(av[:,0].detach());cx["wy"].append(av[:,1].detach());cx["amag"].append(torch.linalg.vector_norm(av[:,:2],dim=-1).detach());cx["tilt"].append(tilt_deg(data.root_quat_w).detach())
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(H,1)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach());av=adv.reshape(-1,4)[:,J].detach()
        g=flat(torch.autograd.grad(-(ratio*av).mean(),aps,retain_graph=True,allow_unused=True),aps).detach();reset_grads.append(g);reset_adv.extend(av.cpu().tolist())
        ctx={k:torch.stack(v).reshape(-1) for k,v in cx.items()}
        for name,mask in bins(ctx).items():
         n=int(mask.sum())
         if n<16:continue
         gb=flat(torch.autograd.grad(-(ratio[mask]*av[mask]).mean(),aps,retain_graph=True,allow_unused=True),aps).detach()
         bin_grads.setdefault(name,[]).append(gb);bin_advs.setdefault(name,[]).extend(av[mask].cpu().tolist());bin_counts[name]=bin_counts.get(name,0)+n
       bsum={}
       for name,gs in sorted(bin_grads.items()):
        st=pair_stats(gs);vals=bin_advs[name];st.update({"samples":bin_counts[name],"grad_norm_mean":float(np.mean([float(g.norm()) for g in gs])),"adv_positive_fraction":float(np.mean(np.array(vals)>0)),"adv_sign_entropy":bentropy(vals)})
        bsum[name]=st
       # between-bin cosine for complementary context pairs, using mean directions
       pairs=[("cmd_vx_neg","cmd_vx_pos"),("cmd_yaw_neg","cmd_yaw_pos"),("wx_neg","wx_pos"),("wy_neg","wy_pos"),("angmag_low","angmag_high"),("tilt_low","tilt_high")]
       between={}
       for a,b in pairs:
        if a in bin_grads and b in bin_grads:
         ga=torch.stack(bin_grads[a]).mean(0);gb=torch.stack(bin_grads[b]).mean(0);between[f"{a}__vs__{b}"]=cosine(ga,gb)
       report["snapshots"][str(snap)]={"global_reset_conditioned":{**pair_stats(reset_grads),"grad_norm_mean":float(np.mean([float(g.norm()) for g in reset_grads])),"adv_positive_fraction":float(np.mean(np.array(reset_adv)>0)),"adv_sign_entropy":bentropy(reset_adv)},"bins":bsum,"between_bin_cosine":between}
      (OUT/"conditional_decomp.json").write_text(json.dumps(report,indent=2)+"\n")
      # compact classification hints
      for snap,x in report["snapshots"].items():
       good=[(k,v["cos_mean"]) for k,v in x["bins"].items() if v["cos_mean"] is not None]
       print("\nSNAP",snap,"global",x["global_reset_conditioned"])
       for k,v in good: print(k,"cos",round(v,3),"neg",round(x["bins"][k]["negative_fraction"],3),"Hsign",round(x["bins"][k]["adv_sign_entropy"],3),"n",x["bins"][k]["samples"])
       print("between",x["between_bin_cosine"])
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c26_freeze():
    """Run former post_v2_t5_c26_freeze.py stage."""
    """Freeze C26: angular-credit stability under actor learning."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.experiment_io.freeze import RUNS, Freeze
    
    C25 = "post_v2_t5_c25_actor_updating25-2026-09-23"
    
    
    class C26Freeze(Freeze):
        """Freeze C26: angular-credit stability under actor learning."""
    
        run = "post_v2_t5_c26_angular_credit-2026-09-23"
        schema = "c26_angular_credit_stability_synthesis_v1"
        status = "C26_CLOSED_ANGULAR_TRANSIENT_LOW_CONSENSUS_NOT_CRITIC_FAILURE"
        artifacts = ("audit.json", "consensus.json", "synthesis.json")
    
        def body(self):
            rep = self.load("replay_audit25.json", C25)
            aud = self.load("audit.json")
            return {
                "evidence": {
                    "value_u25": rep["aggregate"]["25"],
                    "safety_u25": self.load("u25_safety_confirm.json", C25),
                    "semantic_u25": self.load("semantic_u25.json", C25),
                    "angular_stability": {k: v["summary"] for k, v in aud["snapshots"].items()},
                    "consensus": self.load("consensus.json")["snapshots"],
                },
                "decision": {
                    "C25_durability": "critic repair PASS through u25",
                    "angular_credit": "TRANSIENT instability; recovered by u25",
                    "critic_failure": "REJECTED",
                    "smoothness_safety": "REMAINING blocker",
                    "full_T4": "BLOCKED",
                    "V2": "OFF",
                    "next": "C27 Smoothness safety residual audit at u10/u25: isolate failed reset suite physical trajectory and determine whether safety loss is branch-specific actor behavior or evaluation noise; no critic changes.",
                },
            }
    
        def manifest_extra(self):
            return {"c25_refs": {
                "replay25": {"sha256": self.sha(RUNS / C25 / "replay_audit25.json")},
                "safety25": {"sha256": self.sha(RUNS / C25 / "u25_safety_confirm.json")},
                "semantic25": {"sha256": self.sha(RUNS / C25 / "semantic_u25.json")},
            }}
    
        def summary(self, syn):
            return {"status": syn["status"], "next": syn["decision"]["next"]}
    
    
    if True:
        C26Freeze.main()

def run_post_v2_t5_c26_fresh_eval():
    """Run former post_v2_t5_c26_fresh_eval.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c26_actor_coupled-2026-09-23";ORDER=("T","A","O","S");SNAPS=(0,1,5,10);G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,st,en):
     out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
     for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"c26_fresh_eval_v1","snapshots":list(SNAPS),"data":{}}
      for snap in SNAPS:
       rows=[]
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1)
        for si in range(4):
         cur,_=env.reset(seed=2210000+bi*1000+si*97);cur=ot(cur).cuda();R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=ot(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);Y0=ret(R,D,0,32);Y1=ret(R,D,32,64);MC=ret(R,D,0,64)
         rows.append({"specialist":lab,"suite":si,"first":[ev(Y0[:,:,j],V[:32,:,j]) for j in range(4)],
                      "second":[ev(Y1[:,:,j],V[32:,:,j]) for j in range(4)],"mc":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
                      "survival":float(1-D.any(0).mean())})
       f=np.array([x["first"] for x in rows]);s=np.array([x["second"] for x in rows]);mc=np.array([x["mc"] for x in rows])
       out["data"][str(snap)]={"first_mean":float(f.mean()),"second_mean":float(s.mean()),"combined_mean":float(np.r_[f.ravel(),s.ravel()].mean()),
          "first_by_head":f.mean(0).tolist(),"second_by_head":s.mean(0).tolist(),"first_neg":float((f<0).mean()),"second_neg":float((s<0).mean()),
          "mc_mean":float(mc.mean()),"mc_by_head":mc.mean(0).tolist(),"mc_neg":float((mc<0).mean()),"min_survival":float(min(x["survival"] for x in rows))}
      (RUN/"fresh_eval.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["data"],indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c26_local_bin_causal():
    """Run former post_v2_t5_c26_local_bin_causal.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c26_angular_credit-2026-09-23"
    SNAPS=(10,25);H=16;NENV=8;J=1;WREF=np.array([.1,.7,.1,.1],np.float32)
    BINS=("wx_neg","wx_pos","wy_neg","wy_pos","angmag_low","angmag_high")
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def masks(wx,wy,am):return {"wx_neg":wx<0,"wx_pos":wx>=0,"wy_neg":wy<0,"wy_pos":wy>=0,"angmag_low":am<1.0,"angmag_high":am>=1.0}
    def setflat(ps,delta):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(delta[o:o+n].view_as(p));o+=n
    def collect_grad(env,m,w,mgr,seed):
     from talon_rl.models.foundations.four_objective import vector_gae
     from talon_rl.rewards.objectives import normalized_objective_vector
     robot=env.unwrapped.scene["robot"];cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
     ob=[];pre=[];old=[];rw=[];val=[];dn=[];ctx={k:[] for k in BINS}
     for _ in range(H):
      av=robot.data.root_ang_vel_b;mm=masks(av[:,0],av[:,1],torch.linalg.vector_norm(av[:,:2],dim=-1))
      for k in BINS:ctx[k].append(mm[k].detach())
      with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
      nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
      vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
      ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=ot(nxt).cuda()
     with torch.no_grad():nv=m.value_with_preference(cur,w)
     rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
     fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(H,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
     aa=adv.reshape(-1,4)[:,J].detach();aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
     out={}
     for k in BINS:
      mask=torch.stack(ctx[k]).reshape(-1)
      if int(mask.sum())<12:continue
      g=flat(torch.autograd.grad(-(ratio[mask]*aa[mask]).mean(),aps,retain_graph=True,allow_unused=True),aps).detach()
      out[k]=g
     return out
    def rollout(env,m,w,mgr,seed):
     from talon_rl.rewards.objectives import normalized_objective_vector
     robot=env.unwrapped.scene["robot"];cur,_=env.reset(seed=seed);cur=ot(cur).cuda();rows=[]
     with torch.no_grad():
      for _ in range(24):
       av=robot.data.root_ang_vel_b;mm=masks(av[:,0],av[:,1],torch.linalg.vector_norm(av[:,:2],dim=-1))
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       rows.append({"obj":vec[:,1].copy(),"ang":torch.linalg.vector_norm(robot.data.root_ang_vel_b[:,:2],dim=-1).cpu().numpy().copy(),"mask":{k:mm[k].cpu().numpy().copy() for k in BINS}})
       cur=ot(nxt).cuda()
     return rows
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      report={"schema":"c26_local_bin_causal_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
       gs={k:[] for k in BINS}
       for bi in range(8):
        q=collect_grad(env,m,w,mgr,3710000+snap*10000+bi*211)
        for k,g in q.items():gs[k].append(g)
       basevec=torch.cat([p.detach().reshape(-1) for n,p in m.named_parameters() if n.startswith("actor_")])
       models={"base":m}
       for k in BINS:
        if not gs[k]:continue
        g=torch.stack(gs[k]).mean(0);pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);pm.eval()
        delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12));setflat([p for n,p in pm.named_parameters() if n.startswith("actor_")],delta);models[k]=pm
       sums={}
       for k in BINS:
        dob=[];dang=[];n=0
        if k not in models:continue
        for ss in range(6):
         seed=3810000+snap*10000+ss*149;b=rollout(env,models["base"],w,mgr,seed);p=rollout(env,models[k],w,mgr,seed)
         for t in range(len(b)):
          mask=b[t]["mask"][k]
          if mask.any():
           dob.extend((p[t]["obj"][mask]-b[t]["obj"][mask]).tolist());dang.extend((p[t]["ang"][mask]-b[t]["ang"][mask]).tolist());n+=int(mask.sum())
        sums[k]={"n":n,"delta_objective_mean":float(np.mean(dob)),"objective_correct_fraction":float(np.mean(np.array(dob)>0)),"delta_ang_mean":float(np.mean(dang)),"physical_correct_fraction":float(np.mean(np.array(dang)<0))}
       report["snapshots"][str(snap)]=sums
      (OUT/"local_bin_causal.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c27_freeze():
    """Run former post_v2_t5_c27_freeze.py stage."""
    """Freeze C27: the smoothness-branch trajectory safety risk."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.experiment_io.freeze import Freeze
    
    C25 = "post_v2_t5_c25_actor_updating25-2026-09-23"
    
    
    class C27Freeze(Freeze):
        """Freeze C27: the smoothness-branch trajectory safety risk."""
    
        run = "post_v2_t5_c27_smoothness_safety-2026-09-23"
        schema = "c27_smoothness_safety_synthesis_v1"
        status = "C27_CLOSED_SMOOTHNESS_BRANCH_TRAJECTORY_SAFETY_RISK_LOCAL_GRADIENT_NOT_SUFFICIENT"
        artifacts = ("audit.json", "counterfactual.json", "synthesis.json")
    
        def body(self):
            a = self.load("audit.json")
            return {
                "evidence": {
                    "u25_safety": self.load("u25_safety_confirm.json", C25),
                    "failed_seed_trace": a["snapshots"]["25"],
                    "counterfactual": self.load("counterfactual.json"),
                },
                "decision": {
                    "C27": "CLOSED",
                    "smoothness_safety_failure": "REAL / branch-specific / trajectory-level",
                    "critic_failure": "REJECTED",
                    "local_smoothness_gradient_as_root_cause": "NOT SUPPORTED",
                    "full_T4": "BLOCKED",
                    "V2": "OFF",
                    "next": "C28 Smoothness basin/safety audit: characterize S-heavy policy state visitation and recovery margin on the failing reset versus survivors, then decide whether smoothness should remain a free MORL axis or require a safety constraint/regularizer. No critic changes.",
                },
            }
    
        def manifest_extra(self):
            from rl.core.experiment_io.freeze import RUNS
            return {"c25_refs": {"u25_safety": {"sha256": self.sha(RUNS / C25 / "u25_safety_confirm.json")}}}
    
        def summary(self, syn):
            return {"status": syn["status"], "next": syn["decision"]["next"]}
    
    
    if True:
        C27Freeze.main()

def run_post_v2_t5_c27_smoothness_counterfactual():
    """Run former post_v2_t5_c27_smoothness_counterfactual.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c27_smoothness_safety-2026-09-23"
    _exec_stage_preamble("post_v2_t5_c25_eval", globals())
    SEEDS=[2840503]
    def rollout(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];done=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
     ar=[];ti=[];track=[]
     with torch.no_grad():
      for _ in range(64):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
       done|=(te|tr).cpu().numpy();ar.append(float(torch.linalg.vector_norm(a-prev,dim=-1).mean()));ti.append(float(tilt(data.root_quat_w).mean()))
       track.append(float(((data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()).mean()))
       prev=a;cur=obs_tensor(nxt).cuda()
     return {"survival":float(1-done.mean()),"action_rate":float(np.mean(ar)),"tilt_deg":float(np.mean(ti)),"tracking_error":float(np.mean(track))}
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/"S_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS["S"],device="cuda").repeat(NENV,1)
      gs=[grad_perturb(env,m,w,mgr,3,3510000+i*211) for i in range(8)]
      g=torch.stack(gs).mean(0);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base])
      unit=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
      models={"baseline":m}
      for name,scale in (("toward_smooth",1.0),("against_smooth", -1.0),("against_smooth_2x",-2.0)):
       pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/"S_snap_25.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")]
       setflat(paps,[p.detach().clone() for p in paps],unit*scale);pm.eval();models[name]=pm
      out={"schema":"c27_smoothness_counterfactual_v1","grad_norm":float(g.norm()),"rows":[]}
      for seed in SEEDS:
       row={"seed":seed}
       for name,pm in models.items():row[name]=rollout(env,pm,w,seed)
       out["rows"].append(row)
      (OUT/"counterfactual.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c27_smoothness_safety_audit():
    """Run former post_v2_t5_c27_smoothness_safety_audit.py stage."""
    """Why does the smoothness-heavy policy lose a lane, and does it get worse?
    
    Rolls the S-heavy branch at four snapshots on the reset that fails, recording
    action rate, action norm, angular velocity, tilt, vertical velocity and
    tracking error per step. Each is summarised over the lanes that fell and the
    lanes that did not, so a physical signature of the failure can be separated
    from the population average.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.isaac_audit import RUNS, IsaacAudit, obs_tensor
    
    SRC = "post_v2_t5_c25_actor_updating25-2026-09-23"
    SNAPS = (0, 5, 10, 25)
    SEED = 2840503
    PREFERENCE = np.array([.1, .1, .1, .7], np.float32)
    HORIZON = 64
    METRIC_KEYS = ("action_rate", "action_norm", "ang_vel_xy", "tilt_deg",
                   "abs_lin_vel_z", "tracking_abs_error")
    
    
    def tilt(q):
        """Angle between the body's up axis and the world's, in degrees."""
        _, x, y, _ = [q[:, i] for i in range(4)]
        return torch.rad2deg(torch.acos((1 - 2 * (x * x + y * y)).clamp(-1, 1)))
    
    
    class SmoothnessSafetyAudit(IsaacAudit):
        """Physical signature of the smoothness-heavy branch's lane failure."""
    
        run = "post_v2_t5_c27_smoothness_safety-2026-09-23"
        report = "audit.json"
        schema = "c27_smoothness_safety_v1"
    
        def snapshot(self, env, robot, od, ad, snap):
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
    
            n = self.num_envs
            m = T4SharedActorCritic(od, ad).cuda()
            m.load_state_dict(torch.load(RUNS / SRC / f"S_snap_{snap}.pt", map_location="cuda",
                                         weights_only=False)["model"])
            m.eval()
            w = torch.tensor(PREFERENCE, device="cuda").repeat(n, 1)
            cur, _ = env.reset(seed=SEED)
            cur = obs_tensor(cur).cuda()
            prev = torch.zeros((n, ad), device="cuda")
            first_done = [None] * n
            rows = []
            with torch.no_grad():
                for t in range(HORIZON):
                    a = m.act_inference_with_preference(cur, w)
                    nxt, _, te, tr, _ = env.step(a)
                    done = (te | tr).cpu().numpy()
                    data = robot.data
                    cmd = env.unwrapped.command_manager.get_command("base_velocity")
                    rows.append({
                        "t": t,
                        "action_rate": torch.linalg.vector_norm(a - prev, dim=-1).cpu().tolist(),
                        "action_norm": torch.linalg.vector_norm(a, dim=-1).cpu().tolist(),
                        "ang_vel_xy": torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1).cpu().tolist(),
                        "tilt_deg": tilt(data.root_quat_w).cpu().tolist(),
                        "abs_lin_vel_z": data.root_lin_vel_b[:, 2].abs().cpu().tolist(),
                        "tracking_abs_error": ((data.root_lin_vel_b[:, 0] - cmd[:, 0]).abs()
                                               + (data.root_ang_vel_b[:, 2] - cmd[:, 2]).abs()).cpu().tolist(),
                        "done": done.astype(int).tolist()})
                    for i, d in enumerate(done):
                        if d and first_done[i] is None:
                            first_done[i] = t
                    prev = a
                    cur = obs_tensor(nxt).cuda()
    
            failed = [i for i, x in enumerate(first_done) if x is not None]
            survivors = [i for i in range(n) if i not in failed]
            metrics = {}
            for key in METRIC_KEYS:
                arr = np.array([r[key] for r in rows], float)
                metrics[key] = {"all_mean": float(arr.mean()),
                                "failed_env_mean": float(arr[:, failed].mean()) if failed else None,
                                "survivor_mean": float(arr[:, survivors].mean()) if survivors else None}
            return {"first_done_step": first_done, "failed_envs": failed,
                    "survival": float(1 - len(failed) / n), "metrics": metrics, "rows": rows}
    
        def rollout(self, env, obs):
            robot = env.unwrapped.scene["robot"]
            od = obs.shape[-1]
            ad = env.unwrapped.action_manager.total_action_dim
            out = {"schema": self.schema, "seed": SEED, "snapshots": {}}
            for snap in SNAPS:
                out["snapshots"][str(snap)] = self.snapshot(env, robot, od, ad, snap)
            self.write(out)
            for s, x in out["snapshots"].items():
                print("\n", s, "survival", x["survival"], "failed", x["failed_envs"],
                      "done", x["first_done_step"])
                for k, v in x["metrics"].items():
                    print(k, round(v["all_mean"], 4), v["failed_env_mean"], v["survivor_mean"])
            return out
    
    
    if True:
        SmoothnessSafetyAudit.main()

def run_post_v2_t5_c28_temporal_credit():
    """Run former post_v2_t5_c28_temporal_credit.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c28_angular_temporal_credit-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    SNAPS=(0,10,25); HORIZONS=(1,2,4,8,16,32); NENV=8; NB=8; G=.99; J=1
    WREF=np.array([.1,.7,.1,.1],np.float32)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def pair_stats(gs):
     c=[cos(gs[i],gs[j]) for i in range(len(gs)) for j in range(i+1,len(gs))]
     return {"cos_mean":float(np.mean(c)),"cos_std":float(np.std(c)),"negative_fraction":float(np.mean(np.array(c)<0))}
    def trunc_return(rt,dt,h):
     out=torch.zeros_like(rt); T=len(rt)
     for st in range(T):
      run=torch.zeros_like(rt[0]); disc=1.0
      for t in range(st,min(T,st+h)):
       run += disc*rt[t]
       alive=(~dt[t]).unsqueeze(-1)
       run=run*alive + run*(~alive)
       disc*=G
       if bool(dt[t].all()): break
      out[st]=run
     return out
    def rollout_metrics(env,m,w,mgr,seed,steps=32):
     from talon_rl.rewards.objectives import normalized_objective_vector
     robot=env.unwrapped.scene["robot"];cur,_=env.reset(seed=seed);cur=ot(cur).cuda();rows=[]
     with torch.no_grad():
      for _ in range(steps):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       rows.append({"obj":vec[:,1].copy(),"ang":torch.linalg.vector_norm(robot.data.root_ang_vel_b[:,:2],dim=-1).cpu().numpy().copy()})
       cur=ot(nxt).cuda()
     return rows
    def set_delta(pm,delta):
     ps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(delta[o:o+n].view_as(p));o+=n
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      report={"schema":"c28_angular_temporal_credit_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
       grads={h:[] for h in HORIZONS}; imm_sign={h:[] for h in HORIZONS}
       for bi in range(NB):
        cur,_=env.reset(seed=4210000+snap*10000+bi*211);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        for _ in range(32):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(32,1)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        for h in HORIZONS:
         ret=trunc_return(rt,dt,h).reshape(-1,4)[:,J].detach()
         g=flat(torch.autograd.grad(-(ratio*ret).mean(),aps,retain_graph=True,allow_unused=True),aps).detach();grads[h].append(g)
         imm_sign[h].extend((ret>0).cpu().numpy().tolist())
       snapout={"horizons":{}}
       basevec=torch.cat([p.detach().reshape(-1) for p in aps])
       for h in HORIZONS:
        st=pair_stats(grads[h]);st["grad_norm_mean"]=float(np.mean([float(g.norm()) for g in grads[h]]));st["positive_credit_fraction"]=float(np.mean(imm_sign[h]))
        g=torch.stack(grads[h]).mean(0);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);pm.eval();set_delta(pm,delta)
        d_ang=[];d_obj=[]
        for ss in range(6):
         seed=4310000+snap*10000+ss*149;b=rollout_metrics(env,m,w,mgr,seed,32);p=rollout_metrics(env,pm,w,mgr,seed,32)
         for hh in HORIZONS:
          if hh!=h:continue
          ba=np.mean([x["ang"].mean() for x in b[:hh]]);pa=np.mean([x["ang"].mean() for x in p[:hh]])
          bo=np.mean([x["obj"].mean() for x in b[:hh]]);po=np.mean([x["obj"].mean() for x in p[:hh]])
          d_ang.append(pa-ba);d_obj.append(po-bo)
        st.update({"delta_ang_mean":float(np.mean(d_ang)),"physical_correct_fraction":float(np.mean(np.array(d_ang)<0)),"delta_objective_mean":float(np.mean(d_obj)),"objective_correct_fraction":float(np.mean(np.array(d_obj)>0))})
        snapout["horizons"][str(h)]=st
       report["snapshots"][str(snap)]=snapout
      (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c29_aggregation_temporal():
    """Run former post_v2_t5_c29_aggregation_temporal.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c29_angular_aggregation-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    SNAPS=(10,25); NENV=8; NB=8; G=.99; J=1; HMAX=8
    WREF=np.array([.1,.7,.1,.1],np.float32)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def ret_h(rt,dt,h):
     T=len(rt);out=torch.zeros_like(rt)
     for st in range(T):
      run=torch.zeros_like(rt[0]);disc=1.0
      for t in range(st,min(T,st+h)):
       run+=disc*rt[t];disc*=G
      out[st]=run
     return out
    def aggregate(gs,mode):
     X=torch.stack(gs)
     if mode=="mean":return X.mean(0)
     if mode=="median":return X.median(0).values
     if mode=="trimmed":
      n=X.shape[0];k=max(1,int(round(.125*n)))
      vals,_=torch.sort(X,dim=0);return vals[k:n-k].mean(0) if n-2*k>0 else vals.mean(0)
     if mode=="sign":
      s=torch.sign(X);vote=torch.sign(s.sum(0));mag=X.abs().median(0).values;return vote*mag
     if mode=="pcgrad":
      # symmetric deterministic projection pass
      Y=X.clone()
      for i in range(len(gs)):
       gi=Y[i]
       for j in range(len(gs)):
        if i==j:continue
        gj=Y[j];dot=torch.dot(gi,gj)
        if dot<0: gi=gi-dot/(torch.dot(gj,gj)+1e-12)*gj
       Y[i]=gi
      return Y.mean(0)
     raise ValueError(mode)
    def rollout(env,m,w,mgr,seed,steps=8):
     from talon_rl.rewards.objectives import normalized_objective_vector
     robot=env.unwrapped.scene["robot"];cur,_=env.reset(seed=seed);cur=ot(cur).cuda();rows=[]
     with torch.no_grad():
      for _ in range(steps):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       rows.append({"ang":float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[:,:2],dim=-1).mean()),"obj":float(vec[:,1].mean())})
       cur=ot(nxt).cuda()
     return rows
    def pert_model(cls,od,ad,snap,delta):
     pm=cls(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);pm.eval()
     ps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(delta[o:o+n].view_as(p));o+=n
     return pm
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      report={"schema":"c29_angular_aggregation_temporal_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
       gh={h:[] for h in (1,2,3,4,8)}
       for bi in range(NB):
        cur,_=env.reset(seed=4610000+snap*10000+bi*211);cur=ot(cur).cuda();ob=[];pre=[];old=[];rw=[];dn=[]
        for _ in range(HMAX):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(HMAX,1)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        for h in gh:
         rr=ret_h(rt,dt,h).reshape(-1,4)[:,J].detach()
         gh[h].append(flat(torch.autograd.grad(-(ratio*rr).mean(),aps,retain_graph=True,allow_unused=True),aps).detach())
       ref=aggregate(gh[2],"mean")
       snapout={"aggregation":{},"temporal":{}}
       basevec=torch.cat([p.detach().reshape(-1) for p in aps])
       for h in (1,2,4,8):
        snapout["aggregation"][str(h)]={}
        for mode in ("mean","trimmed","median","sign","pcgrad"):
         g=aggregate(gh[h],mode);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12));pm=pert_model(T4SharedActorCritic,od,ad,snap,delta)
         da=[];do=[]
         for ss in range(6):
          seed=4710000+snap*10000+ss*149;b=rollout(env,m,w,mgr,seed,h);p=rollout(env,pm,w,mgr,seed,h)
          da.append(np.mean([x["ang"] for x in p])-np.mean([x["ang"] for x in b]));do.append(np.mean([x["obj"] for x in p])-np.mean([x["obj"] for x in b]))
         agrees=[cosine(g,x)>0 for x in gh[h]]
         snapout["aggregation"][str(h)][mode]={
           "norm":float(g.norm()),"retained_vs_mean":float(g.norm()/(aggregate(gh[h],"mean").norm()+1e-12)),
           "cos_to_H2_ref":cosine(g,ref),"contributor_agree_fraction":float(np.mean(agrees)),
           "delta_ang_mean":float(np.mean(da)),"physical_correct_fraction":float(np.mean(np.array(da)<0)),
           "delta_objective_mean":float(np.mean(do)),"objective_correct_fraction":float(np.mean(np.array(do)>0))}
       # temporal increments: H1, H2-H1, H3-H2, H4-H3, H8-H4
       means={h:aggregate(gh[h],"mean") for h in gh}
       comps={"H1":means[1],"H2-H1":means[2]-means[1],"H3-H2":means[3]-means[2],"H4-H3":means[4]-means[3],"H8-H4":means[8]-means[4]}
       for name,g in comps.items():
        snapout["temporal"][name]={"norm":float(g.norm()),"cos_to_H2_ref":cosine(g,ref),"cos_to_H1":cosine(g,means[1])}
       report["snapshots"][str(snap)]=snapout
      (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

STAGES = {
    "post_v2_t5_c20_collect_pool": run_post_v2_t5_c20_collect_pool,
    "post_v2_t5_c20_coverage_analyze": run_post_v2_t5_c20_coverage_analyze,
    "post_v2_t5_c20_mc64_eval": run_post_v2_t5_c20_mc64_eval,
    "post_v2_t5_c21_replay_audit": run_post_v2_t5_c21_replay_audit,
    "post_v2_t5_c21_support_width_train": run_post_v2_t5_c21_support_width_train,
    "post_v2_t5_c21_warm_replay_audit": run_post_v2_t5_c21_warm_replay_audit,
    "post_v2_t5_c21_warm_support_train": run_post_v2_t5_c21_warm_support_train,
    "post_v2_t5_c22_diverse_support_train": run_post_v2_t5_c22_diverse_support_train,
    "post_v2_t5_c22_freeze": run_post_v2_t5_c22_freeze,
    "post_v2_t5_c22_replay_audit": run_post_v2_t5_c22_replay_audit,
    "post_v2_t5_c22_reset_replay_audit": run_post_v2_t5_c22_reset_replay_audit,
    "post_v2_t5_c22_reset_support_train": run_post_v2_t5_c22_reset_support_train,
    "post_v2_t5_c22_reset_terminal_multisuite": run_post_v2_t5_c22_reset_terminal_multisuite,
    "post_v2_t5_c22_terminal_multisuite": run_post_v2_t5_c22_terminal_multisuite,
    "post_v2_t5_c23_freeze": run_post_v2_t5_c23_freeze,
    "post_v2_t5_c23_paired_terminal_multisuite": run_post_v2_t5_c23_paired_terminal_multisuite,
    "post_v2_t5_c23_phase_split_audit": run_post_v2_t5_c23_phase_split_audit,
    "post_v2_t5_c23_replay_audit": run_post_v2_t5_c23_replay_audit,
    "post_v2_t5_c23_reset_diverse_pilot": run_post_v2_t5_c23_reset_diverse_pilot,
    "post_v2_t5_c23_reset_diverse_train": run_post_v2_t5_c23_reset_diverse_train,
    "post_v2_t5_c23_reset_recent_paired_train": run_post_v2_t5_c23_reset_recent_paired_train,
    "post_v2_t5_c23_terminal_multisuite": run_post_v2_t5_c23_terminal_multisuite,
    "post_v2_t5_c24_freeze": run_post_v2_t5_c24_freeze,
    "post_v2_t5_c24_phase_balanced_train": run_post_v2_t5_c24_phase_balanced_train,
    "post_v2_t5_c24_phase_eval": run_post_v2_t5_c24_phase_eval,
    "post_v2_t5_c24_reset_breakdown": run_post_v2_t5_c24_reset_breakdown,
    "post_v2_t5_c24_tracking_residual_audit": run_post_v2_t5_c24_tracking_residual_audit,
    "post_v2_t5_c25_actor_updating_reset_support": run_post_v2_t5_c25_actor_updating_reset_support,
    "post_v2_t5_c25_actor_updating_reset_support_25": run_post_v2_t5_c25_actor_updating_reset_support_25,
    "post_v2_t5_c25_angular_credit_confirm": run_post_v2_t5_c25_angular_credit_confirm,
    "post_v2_t5_c25_angular_u0_control": run_post_v2_t5_c25_angular_u0_control,
    "post_v2_t5_c25_causal_semantics": run_post_v2_t5_c25_causal_semantics,
    "post_v2_t5_c25_causal_u10_confirm": run_post_v2_t5_c25_causal_u10_confirm,
    "post_v2_t5_c25_causal_u25_confirm": run_post_v2_t5_c25_causal_u25_confirm,
    "post_v2_t5_c25_eval": run_post_v2_t5_c25_eval,
    "post_v2_t5_c25_freeze": run_post_v2_t5_c25_freeze,
    "post_v2_t5_c25_mid_causal_semantics": run_post_v2_t5_c25_mid_causal_semantics,
    "post_v2_t5_c25_phase_eval": run_post_v2_t5_c25_phase_eval,
    "post_v2_t5_c25_replay25": run_post_v2_t5_c25_replay25,
    "post_v2_t5_c25_replay25_audit": run_post_v2_t5_c25_replay25_audit,
    "post_v2_t5_c25_replay_audit": run_post_v2_t5_c25_replay_audit,
    "post_v2_t5_c25_replay_audit_25": run_post_v2_t5_c25_replay_audit_25,
    "post_v2_t5_c25_semantic_perturb": run_post_v2_t5_c25_semantic_perturb,
    "post_v2_t5_c25_semantic_u25": run_post_v2_t5_c25_semantic_u25,
    "post_v2_t5_c25_tracking_confirm": run_post_v2_t5_c25_tracking_confirm,
    "post_v2_t5_c25_u10_perturb_confirm": run_post_v2_t5_c25_u10_perturb_confirm,
    "post_v2_t5_c25_u10_safety_confirm": run_post_v2_t5_c25_u10_safety_confirm,
    "post_v2_t5_c25_u25_causal_semantics": run_post_v2_t5_c25_u25_causal_semantics,
    "post_v2_t5_c25_u25_safety_confirm": run_post_v2_t5_c25_u25_safety_confirm,
    "post_v2_t5_c26_actor_coupled_train": run_post_v2_t5_c26_actor_coupled_train,
    "post_v2_t5_c26_angular_consensus": run_post_v2_t5_c26_angular_consensus,
    "post_v2_t5_c26_angular_credit_stability": run_post_v2_t5_c26_angular_credit_stability,
    "post_v2_t5_c26_conditional_decomp": run_post_v2_t5_c26_conditional_decomp,
    "post_v2_t5_c26_freeze": run_post_v2_t5_c26_freeze,
    "post_v2_t5_c26_fresh_eval": run_post_v2_t5_c26_fresh_eval,
    "post_v2_t5_c26_local_bin_causal": run_post_v2_t5_c26_local_bin_causal,
    "post_v2_t5_c27_freeze": run_post_v2_t5_c27_freeze,
    "post_v2_t5_c27_smoothness_counterfactual": run_post_v2_t5_c27_smoothness_counterfactual,
    "post_v2_t5_c27_smoothness_safety_audit": run_post_v2_t5_c27_smoothness_safety_audit,
    "post_v2_t5_c28_temporal_credit": run_post_v2_t5_c28_temporal_credit,
    "post_v2_t5_c29_aggregation_temporal": run_post_v2_t5_c29_aggregation_temporal,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
