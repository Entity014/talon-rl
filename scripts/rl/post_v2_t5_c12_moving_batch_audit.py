#!/usr/bin/env python3
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c12_moving_batch-2026-09-23"
ORDER=("T","A","O","S");NB=6;H=32;G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def trunc_mc(R,D):
    out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
    for t in range(len(R)-1,-1,-1):
        run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
    return out
def ev(y,p):
    y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
    return float(1-np.var(y-p)/(np.var(y)+1e-12))
def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
def cos(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def solve_head(model,X,W,Y):
    with torch.no_grad():
        F=model.critic_body(model._with_w(X,W));A=torch.cat([F,torch.ones((len(F),1),device=F.device)],1)
        sol=torch.linalg.lstsq(A,Y).solution
    return sol[:-1,:].T.contiguous(),sol[-1,:].contiguous()
def set_head(model,W,b):
    with torch.no_grad():model.critic_head.weight.copy_(W);model.critic_head.bias.copy_(b)
def eval_mse_ev(model,X,W,Y):
    with torch.no_grad():P=model.value_with_preference(X,W);m=((P-Y)**2).mean(0).cpu().numpy();e=[ev(Y[:,j].cpu(),P[:,j].cpu()) for j in range(4)]
    return m,e
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  from talon_rl.t3b_objectives import normalized_objective_vector
  OUT.mkdir(parents=True,exist_ok=True)
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  report={"schema":"t5_c12_moving_batch_v1","checkpoint_snapshot":25,"batch_horizon":H,"num_batches":NB,"specialists":{}}
  for bi,lab in enumerate(ORDER):
    cp=RUN/f"{lab}_snap_25.pt";base=T4SharedActorCritic(od,ad).cuda();base.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);base.eval()
    w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
    # one continuous reset seed, then consecutive batches with no manual resets between batches
    cur,_=env.reset(seed=970000+bi*1000);cur=obs_tensor(cur).cuda()
    batches=[]
    for k in range(NB):
      obs=[];rw=[];dn=[]
      with torch.no_grad():
        for _ in range(H):
          obs.append(cur);a=base.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
          raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
          rw.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
      X=torch.cat(obs);W=w.repeat(H,1);R=torch.stack(rw);D=torch.stack(dn).bool();Y=trunc_mc(R,D).reshape(-1,4).detach()
      batches.append((X,W,Y))
    # gradients + stats + optimal heads
    critic_params=[p for n,p in base.named_parameters() if n.startswith("critic_")]
    grads=[];stats=[];heads=[]
    for k,(X,W,Y) in enumerate(batches):
      pred=base.value_with_preference(X,W);loss=((pred-Y)**2).mean()
      gs=torch.autograd.grad(loss,critic_params,retain_graph=False,allow_unused=True);g=flat(gs,critic_params).detach();grads.append(g)
      Wh,bh=solve_head(base,X,W,Y);heads.append((Wh,bh))
      obs_np=X.detach().cpu().numpy();stats.append({
        "batch":k,
        "target_mean":Y.mean(0).cpu().tolist(),"target_std":Y.std(0,unbiased=False).cpu().tolist(),
        "obs_mean":obs_np.mean(0).tolist(),"obs_std":obs_np.std(0).tolist(),
        "grad_norm":float(g.norm())
      })
    pairs=[]
    for k in range(NB-1):
      Wh,bh=heads[k];Wn,bn=heads[k+1]
      hd=float(torch.sqrt((Wh-Wn).pow(2).sum()+(bh-bn).pow(2).sum()))
      pairs.append({
        "pair":f"{k}->{k+1}",
        "gradient_cosine":cos(grads[k],grads[k+1]),
        "gradient_norm_ratio_next_over_current":float(grads[k+1].norm()/(grads[k].norm()+1e-12)),
        "optimal_head_l2_drift":hd,
        "target_mean_shift_l2":float(np.linalg.norm(np.array(stats[k+1]["target_mean"])-np.array(stats[k]["target_mean"]))),
        "target_std_shift_l2":float(np.linalg.norm(np.array(stats[k+1]["target_std"])-np.array(stats[k]["target_std"]))),
        "obs_mean_shift_l2":float(np.linalg.norm(np.array(stats[k+1]["obs_mean"])-np.array(stats[k]["obs_mean"]))),
        "obs_std_shift_l2":float(np.linalg.norm(np.array(stats[k+1]["obs_std"])-np.array(stats[k]["obs_std"])))
      })
    # transfer: fit 10 Adam steps on B_t, evaluate B_t and B_t+1 before/after
    transfer=[]
    for k in range(NB-1):
      Xt,Wt,Yt=batches[k];Xn,Wn,Yn=batches[k+1]
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);ps=[p for n,p in m.named_parameters() if n.startswith("critic_")]
      before_t=eval_mse_ev(m,Xt,Wt,Yt);before_n=eval_mse_ev(m,Xn,Wn,Yn)
      opt=torch.optim.Adam(ps,lr=1e-4)
      for _ in range(10):
        pred=m.value_with_preference(Xt,Wt);loss=((pred-Yt)**2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
      after_t=eval_mse_ev(m,Xt,Wt,Yt);after_n=eval_mse_ev(m,Xn,Wn,Yn)
      transfer.append({
        "pair":f"{k}->{k+1}",
        "current_mse_before":before_t[0].tolist(),"current_mse_after":after_t[0].tolist(),
        "next_mse_before":before_n[0].tolist(),"next_mse_after":after_n[0].tolist(),
        "current_ev_before":before_t[1],"current_ev_after":after_t[1],
        "next_ev_before":before_n[1],"next_ev_after":after_n[1],
        "current_mse_change_fraction":((before_t[0]-after_t[0])/(before_t[0]+1e-12)).tolist(),
        "next_mse_change_fraction":((before_n[0]-after_n[0])/(before_n[0]+1e-12)).tolist(),
        "next_ev_delta":(np.array(after_n[1])-np.array(before_n[1])).tolist()
      })
    report["specialists"][lab]={"batch_stats":stats,"pairwise":pairs,"transfer":transfer}
  # aggregate
  pcs=[];hds=[];tmean=[];omean=[];transfer_next=[];transfer_cur=[];transfer_next_evd=[]
  for lab in ORDER:
    for p in report["specialists"][lab]["pairwise"]:
      pcs.append(p["gradient_cosine"]);hds.append(p["optimal_head_l2_drift"]);tmean.append(p["target_mean_shift_l2"]);omean.append(p["obs_mean_shift_l2"])
    for t in report["specialists"][lab]["transfer"]:
      transfer_cur+=t["current_mse_change_fraction"];transfer_next+=t["next_mse_change_fraction"];transfer_next_evd+=t["next_ev_delta"]
  report["aggregate"]={
    "gradient_cosine_mean":float(np.mean(pcs)),"gradient_cosine_median":float(np.median(pcs)),
    "gradient_cosine_negative_fraction":float(np.mean(np.array(pcs)<0)),
    "gradient_cosine_lt_0p25_fraction":float(np.mean(np.array(pcs)<.25)),
    "optimal_head_l2_drift_mean":float(np.mean(hds)),
    "target_mean_shift_l2_mean":float(np.mean(tmean)),
    "obs_mean_shift_l2_mean":float(np.mean(omean)),
    "fit_current_mse_improvement_mean":float(np.mean(transfer_cur)),
    "transfer_next_mse_improvement_mean":float(np.mean(transfer_next)),
    "transfer_next_mse_worsen_fraction":float(np.mean(np.array(transfer_next)<0)),
    "transfer_next_ev_delta_mean":float(np.mean(transfer_next_evd)),
    "transfer_next_ev_worsen_fraction":float(np.mean(np.array(transfer_next_evd)<0))
  }
  (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps(report["aggregate"],indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
