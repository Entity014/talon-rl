#!/usr/bin/env python3
from pathlib import Path
import sys,json,copy,hashlib,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23"
OUT=ROOT/"runs/post_v2_t5_c11_loss_geometry-2026-09-23"
ORDER=("T","A","O","S");SNAPS=(10,25);G=.99;EPS=1e-6
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
def flat_grads(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
def pvec(ps):return torch.cat([p.detach().reshape(-1) for p in ps])
def cos(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
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
  report={"schema":"t5_c11_critic_loss_geometry_audit_v1","run_source":str(RUN),"snapshots":SNAPS,"specialists":{}}
  for bi,lab in enumerate(ORDER):
   report["specialists"][lab]={}
   w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
   for snap in SNAPS:
    cp=RUN/f"{lab}_snap_{snap}.pt"
    model=T4SharedActorCritic(od,ad).cuda();model.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);model.eval()
    cur,_=env.reset(seed=950000+bi*1000+snap);cur=obs_tensor(cur).cuda();OBS=[];RW=[];DN=[]
    with torch.no_grad():
     for _ in range(32):
      OBS.append(cur);a=model.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
      raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
      RW.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);DN.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
    X=torch.cat(OBS);W=w.repeat(32,1);R=torch.stack(RW);D=torch.stack(DN).bool();Y=trunc_mc(R,D).reshape(-1,4).detach()
    with torch.no_grad():P=model.value_with_preference(X,W)
    mu=Y.mean(0);std=Y.std(0,unbiased=False).clamp_min(EPS);var=std.pow(2);rng=Y.max(0).values-Y.min(0).values
    err=(P-Y).pow(2);head_loss=err.mean(0);raw_share=head_loss/head_loss.sum()
    critic_params=[p for n,p in model.named_parameters() if n.startswith("critic_")]
    body_params=[p for n,p in model.named_parameters() if n.startswith("critic_body")]
    head_params=[p for n,p in model.named_parameters() if n.startswith("critic_head")]
    per_head_grad=[]
    for j in range(4):
     gs=torch.autograd.grad((model.value_with_preference(X,W)[:,j]-Y[:,j]).pow(2).mean(),critic_params,retain_graph=False,allow_unused=True)
     gb=torch.autograd.grad((model.value_with_preference(X,W)[:,j]-Y[:,j]).pow(2).mean(),body_params,retain_graph=False,allow_unused=True)
     gh=torch.autograd.grad((model.value_with_preference(X,W)[:,j]-Y[:,j]).pow(2).mean(),head_params,retain_graph=False,allow_unused=True)
     per_head_grad.append({"all":float(flat_grads(gs,critic_params).norm()),"body":float(flat_grads(gb,body_params).norm()),"head":float(flat_grads(gh,head_params).norm())})
    modes={}
    updates={}
    for mode in ("raw","variance_norm","standardized"):
     m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.train()
     ps=[p for n,p in m.named_parameters() if n.startswith("critic_")]
     before=pvec(ps).clone()
     opt=torch.optim.Adam(ps,lr=1e-4)
     pred=m.value_with_preference(X,W)
     if mode=="raw":
      per=(pred-Y).pow(2).mean(0);loss=per.mean()
     elif mode=="variance_norm":
      per=((pred-Y).pow(2)/(var+EPS)).mean(0);loss=per.mean()
     else:
      # algebraically equivalent to variance-normalized regression when both pred/target share affine transform
      per=(((pred-mu)/(std+EPS)-(Y-mu)/(std+EPS)).pow(2)).mean(0);loss=per.mean()
     opt.zero_grad(set_to_none=True);loss.backward();opt.step()
     after=pvec(ps).clone();updates[mode]=after-before
     step_by_name={}
     with torch.no_grad():
      for n,p in m.named_parameters():
       if n.startswith("critic_"):
        p0=torch.load(cp,map_location="cuda",weights_only=False)["model"][n].to(p.device)
        step_by_name[n]=float((p-p0).norm())
     with torch.no_grad():
      pred2=m.value_with_preference(X,W);pre=((P-Y).pow(2).mean(0)).cpu().numpy();post=((pred2-Y).pow(2).mean(0)).cpu().numpy()
      ev0=[ev(Y[:,j].cpu(),P[:,j].cpu()) for j in range(4)];ev1=[ev(Y[:,j].cpu(),pred2[:,j].cpu()) for j in range(4)]
     modes[mode]={
      "optimization_loss":float(loss.detach()),"optimization_per_head":per.detach().cpu().tolist(),
      "parameter_step_norm":float((after-before).norm()),"relative_step_norm":float((after-before).norm()/(before.norm()+1e-12)),
      "step_by_parameter":step_by_name,
      "raw_mse_pre":pre.tolist(),"raw_mse_post":post.tolist(),
      "raw_mse_reduction_fraction":((pre-post)/(pre+1e-12)).tolist(),
      "ev_pre":ev0,"ev_post":ev1,"ev_delta":(np.array(ev1)-np.array(ev0)).tolist(),
      "heads_mse_improved":int(np.sum(post<pre)),"heads_ev_improved":int(np.sum(np.array(ev1)>np.array(ev0)))
     }
    modes["variance_norm"]["update_cosine_vs_raw"]=cos(updates["variance_norm"],updates["raw"])
    modes["standardized"]["update_cosine_vs_variance_norm"]=cos(updates["standardized"],updates["variance_norm"])
    # fixed-batch 10-step convergence for the two genuinely distinct objectives
    for mode in ("raw","variance_norm"):
     m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.train()
     ps=[p for n,p in m.named_parameters() if n.startswith("critic_")];opt=torch.optim.Adam(ps,lr=1e-4);trace=[]
     for kk in range(10):
      pred=m.value_with_preference(X,W)
      if mode=="raw":loss=(pred-Y).pow(2).mean()
      else:loss=((pred-Y).pow(2)/(var+EPS)).mean()
      opt.zero_grad(set_to_none=True);loss.backward();opt.step()
      with torch.no_grad():
       pred2=m.value_with_preference(X,W);mse=((pred2-Y).pow(2).mean(0)).cpu().numpy();evs=[ev(Y[:,j].cpu(),pred2[:,j].cpu()) for j in range(4)]
      trace.append({"step":kk+1,"raw_mse":mse.tolist(),"ev":evs})
     modes[mode]["fixed_batch_10step_trace"]=trace
    report["specialists"][lab][str(snap)]={
     "target_stats":{"mean":mu.cpu().tolist(),"std":std.cpu().tolist(),"variance":var.cpu().tolist(),"range":rng.cpu().tolist(),
                     "nonzero_fraction":[float((Y[:,j].abs()>1e-12).float().mean()) for j in range(4)]},
     "raw_loss":{"per_head":head_loss.cpu().tolist(),"share":raw_share.cpu().tolist(),"total_mean":float(head_loss.mean())},
     "per_head_gradient_norm":per_head_grad,
     "counterfactual":modes
    }
  # aggregate
  agg={}
  for snap in map(str,SNAPS):
   agg[snap]={}
   for mode in ("raw","variance_norm","standardized"):
    mse=[];evd=[];hmi=[];hei=[];steps=[]
    for lab in ORDER:
     x=report["specialists"][lab][snap]["counterfactual"][mode]
     mse+=x["raw_mse_reduction_fraction"];evd+=x["ev_delta"];hmi.append(x["heads_mse_improved"]);hei.append(x["heads_ev_improved"]);steps.append(x["relative_step_norm"])
    agg[snap][mode]={"mean_mse_reduction_fraction":float(np.mean(mse)),"mean_ev_delta":float(np.mean(evd)),
                     "head_mse_improve_fraction":float(np.sum(hmi)/(4*len(ORDER))),"head_ev_improve_fraction":float(np.sum(hei)/(4*len(ORDER))),
                     "mean_relative_step_norm":float(np.mean(steps))}
   cs=[];eq=[]
   for lab in ORDER:
    x=report["specialists"][lab][snap]["counterfactual"]
    cs.append(x["variance_norm"]["update_cosine_vs_raw"]);eq.append(x["standardized"]["update_cosine_vs_variance_norm"])
   agg[snap]["update_geometry"]={"variance_norm_vs_raw_cosine_mean":float(np.mean(cs)),"standardized_vs_variance_norm_cosine_mean":float(np.mean(eq))}
  report["aggregate"]=agg
  out=OUT/"audit.json";out.write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps({"aggregate":agg},indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
