from pathlib import Path
import json,sys,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
CK={
"edge":ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"}
OUT=ROOT/"runs/authority_isolated_simplex_edge_noregression_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
ORDER=("T","A","O","S","C");PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
SEM={"suite2":840003,"suite3":840004};HELD={"suite4":850101,"suite5":850202,"suite6":850303}
NENV=8;H=64;G=.99
def ot(x):
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
        for t in range(len(R)-1,-1,-1):
            run=R[t]+G*run*(~D[t])[:,None];out[t]=run
    else:
        for st in range(0,len(R),seg):
            en=min(st+seg,len(R));run=np.zeros_like(R[0])
            for t in range(en-1,st-1,-1):
                run=R[t]+G*run*(~D[t])[:,None];out[t]=run
    return out
def rollout(env,m,wv,seed):
    from talon_rl.rewards.objectives import normalized_objective_vector
    mgr=env.unwrapped.reward_manager;w=torch.tensor(wv,device="cuda").repeat(NENV,1)
    cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool);reasons=[[] for _ in range(NENV)]
    with torch.no_grad():
        for t in range(H):
            V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();done|=dd
            tm=env.unwrapped.termination_manager
            for i in range(NENV):
                if dd[i]:
                    reasons[i]=[n for n in tm.active_terms if bool(tm.get_term(n)[i].detach().cpu())]
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
            D.append(dd);cur=ot(nxt).cuda()
    R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
    return {"survival":float(1-done.mean()),"fail_count":int(done.sum()),"reasons":reasons,
      "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
      "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
      "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
      "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
def summarize(rows):
    h=np.array([r["h32_ev"] for r in rows]);m=np.array([r["mc64_ev"] for r in rows])
    hb=np.array([r["h32_bias"] for r in rows]);mb=np.array([r["mc64_bias"] for r in rows])
    return {"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
      "mc64_ev_mean":float(m.mean()),"mc64_negative_fraction":float((m<0).mean()),
      "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
      "min_survival":float(min(r["survival"] for r in rows)),"failed_lanes":int(sum(r["fail_count"] for r in rows))}
def main():
    from isaaclab.app import AppLauncher
    sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
        rep={"schema":"authority_isolated_ai_c2_endpoint_eval_v1","models":{}}
        for arm,path in CK.items():
            cls=AuthorityIsolatedActorCritic if arm=="narrow" else AuthorityIsolatedWideCritic
            m=cls(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval()
            sem=[];held=[];fresh=[]
            for suite,seed in SEM.items():
                for lab in ORDER:
                    q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});sem.append(q)
            for suite,seed in HELD.items():
                for lab in ORDER:
                    q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});held.append(q)
            for li,lab in enumerate(ORDER):
                for si in range(4):
                    seed=9700000+li*1000+si*113
                    q=rollout(env,m,PREFS[lab],seed);q.update({"suite":si,"preference":lab,"seed":seed});fresh.append(q)
            fs=summarize(fresh);ss=summarize(sem);hs=summarize(held)
            critic_gate=bool(fs["h32_ev_mean"]>0 and fs["mc64_ev_mean"]>0 and fs["h32_negative_fraction"]<=.25 and fs["mc64_negative_fraction"]<=.25 and fs["min_survival"]>=.95)
            rep["models"][arm]={"semantic":ss,"heldout":hs,"fresh_critic":fs,"critic_gate":critic_gate,
                                "semantic_rows":sem,"heldout_rows":held,"fresh_rows":fresh}
            print("SUMMARY",arm,json.dumps(rep["models"][arm],default=str)[:2000],flush=True)
        (OUT/"endpoint_eval.json").write_text(json.dumps(rep,indent=2)+"\n")
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
