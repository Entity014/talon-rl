#!/usr/bin/env python3
from __future__ import annotations
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
      from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01
      from talon_rl.t3b_objectives import normalized_objective_vector
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
if __name__=="__main__":main()
