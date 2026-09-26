#!/usr/bin/env python3
"""Freeze provenance-backed R2 h1 shared/residual basis from D1 specialists."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[4]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
CODE_COMMIT="f77af91932b1651b531cb3f64cb3c1852ab800cb"
EPS=1e-12
def obs_tensor(x):
    if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def d1_h1(m,obs,w):
    return F.elu(m.actor_body[0](torch.cat((obs,w),dim=-1)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--steps",type=int,default=16);ap.add_argument("--suites",type=int,default=4)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json";anchor=json.loads(anchor_path.read_text())
    anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    import gymnasium as gym, isaaclab_tasks
    from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
    from talon_rl.v1c_actor_critic import V1CSharedActorCritic
    from talon_rl.v2_behavior_actor_critic import V2BehaviorActorCritic
    cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
    env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
    obs,_=env.reset(seed=127001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
    p=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
    v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda();v2.load_state_dict(p["model"]);v2.eval()
    specs={};spec_paths={}
    for lab in PREFS:
        pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
        m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
    ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}
    shared_samples=[];be_samples=[]
    with torch.no_grad():
        for suite in range(args.suites):
            cur,_=env.reset(seed=127001+suite);cur=obs_tensor(cur).cuda()
            for _ in range(args.steps):
                h={k:d1_h1(specs[k],cur,ws[k]) for k in PREFS}
                dPB=h["B"]-h["P"];dPE=h["E"]-h["P"];dBE=h["E"]-h["B"]
                shared=(dPB+dPE)/2.0
                shared_samples.append(shared.cpu())
                # pointwise residual relative to pointwise shared direction
                u=shared/(torch.linalg.vector_norm(shared,dim=-1,keepdim=True)+EPS)
                be=dBE-(dBE*u).sum(-1,keepdim=True)*u
                be_samples.append(be.cpu())
                action=v2.act_inference_with_preference(cur,ws["P"])
                nxt,_,_,_,_=env.step(torch.clamp(action,-1,1));cur=obs_tensor(nxt).cuda()
    S=torch.cat(shared_samples,0);R=torch.cat(be_samples,0)
    b_shared=S.mean(0);b_shared=b_shared/(b_shared.norm()+EPS)
    # Freeze residual by global mean, then re-orthogonalize to frozen shared basis.
    b_be=R.mean(0);b_be=b_be-(b_be@b_shared)*b_shared;b_be=b_be/(b_be.norm()+EPS)
    dot=float(b_shared@b_be)
    tensor_path=args.output.with_suffix(".pt")
    torch.save({"b_shared":b_shared,"b_BE":b_be},tensor_path)
    report={
      "schema":"v2_r2_frozen_basis_v1","status":"FROZEN","layer":"h1_post_elu_pre_modulation",
      "protocol":{"state_source":"V2-R1 P-policy closed-loop","reset_seed_base":127001,"suites":args.suites,"steps":args.steps,"num_envs":args.num_envs,
                  "shared":"normalize(mean_samples((dPB+dPE)/2))",
                  "BE":"normalize(orthogonalize(mean_samples(pointwise_orthogonalized_dBE), b_shared))"},
      "provenance":{"code_commit":CODE_COMMIT,"r1_checkpoint":str(args.checkpoint),"r1_checkpoint_sha256":sha(args.checkpoint),
                    "v2a_anchor_sha256":sha(anchor_path),
                    "specialists":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
      "basis":{"dimension":int(b_shared.numel()),"dot_shared_BE":dot,"shared_norm":float(b_shared.norm()),"BE_norm":float(b_be.norm()),
               "tensor_file":str(tensor_path),"tensor_sha256":sha(tensor_path)}
    }
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["basis"],indent=2))
    env.close();app.close()
if __name__=="__main__":main()
