"""Consolidated experiment workflow.

Generated from the former one-script-per-stage layout. Use a stage name as the
first CLI argument to run the corresponding experiment.
"""
from __future__ import annotations
import argparse

def run_v2c0_function_preserving_gate():
    """Run former v2c0_function_preserving_gate.py stage."""
    
    from pathlib import Path
    import argparse, hashlib, json, sys
    
    import numpy as np
    import torch
    
    ROOT = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    
    V2B_INIT = ROOT / "runs/v2b0_function_preserving_gate-2026-09-23/v2b0_init.pt"
    OUT = ROOT / "runs/v2c0_function_preserving_gate-2026-09-24"
    
    WREF = np.asarray([0.25, 0.25, 0.25, 0.25], np.float32)
    PREFS = np.asarray(
        [
            [0.70, 0.10, 0.10, 0.10],
            [0.10, 0.70, 0.10, 0.10],
            [0.10, 0.10, 0.70, 0.10],
            [0.10, 0.10, 0.10, 0.70],
            [0.25, 0.25, 0.25, 0.25],
        ],
        np.float32,
    )
    NENV = 8
    STEPS = 64
    TOL = 1e-6
    
    
    def ot(x):
        if isinstance(x, dict):
            x = x.get("policy", next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    
    def ma(x):
        return float(torch.max(torch.abs(x)).detach().cpu()) if x.numel() else 0.0
    
    
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    
    
    def state_hash(model):
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    
    
    def scalarization_check():
        from talon_rl.rewards.objectives import (
            OBJECTIVE_ORDER,
            NORMALIZATION_DIVISORS,
            normalize_objectives,
            scalarize,
        )
    
        raw = np.asarray(
            [[1.7194554805755615, -0.15590913593769073, -0.01563369482755661, -0.08311229199171066]],
            np.float32,
        )
        norm = normalize_objectives(raw)
        exp = np.asarray([[1.0, -1.0, -1.0, -1.0]], np.float32)
        got = float(scalarize(norm, WREF)[0])
        ref = float((exp * WREF).sum())
        err = float(np.max(np.abs(norm - exp)))
        return {
            "objective_order": list(OBJECTIVE_ORDER),
            "normalization_divisors": NORMALIZATION_DIVISORS.tolist(),
            "normalized_probe_max_abs_error": err,
            "scalarization_abs_error": abs(got - ref),
            "pass": bool(err <= 1e-7 and abs(got - ref) <= 1e-7),
        }
    
    
    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--output-dir", type=Path, default=OUT)
        ap.add_argument("--seed", type=int, default=424242)
        args = ap.parse_args()
    
        if not args.output_dir.is_absolute():
            args.output_dir = (ROOT / args.output_dir).resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
        from isaaclab.app import AppLauncher
    
        saved = sys.argv[:]
        sys.argv = [sys.argv[0]]
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        sys.argv = saved
    
        env = None
        try:
            import gymnasium as gym
            import isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
    
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            from talon_rl.models.conditioning.residual_experts_experts import (
                V2CPreferenceGatedResidualActorCritic,
                initialize_from_v2b,
            )
    
            cfg = UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs = NENV
            cfg.seed = args.seed
            cfg.scene.robot.spawn.usd_path = str(ROOT / "talon_rl/assets/data/Robots/unitree_a1/a1.usd")
    
            env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
            obs, _ = env.reset(seed=args.seed)
            obs = ot(obs).cuda()
            action_dim = env.unwrapped.action_manager.total_action_dim
    
            state = torch.load(V2B_INIT, map_location="cuda", weights_only=False)
    
            ref = V2BSingleSiteFiLMActorCritic(obs.shape[-1], action_dim).cuda()
            ref.load_state_dict(state["model"])
            ref.eval()
    
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            v2c = V2CPreferenceGatedResidualActorCritic(obs.shape[-1], action_dim).cuda()
            initialize_from_v2b(v2c, state)
            v2c.eval()
    
            w = torch.tensor(WREF, device="cuda").repeat(NENV, 1)
    
            with torch.no_grad():
                ref_mean = ref.actor_mean(ref._actor_features_v2a(obs, w))
                c_mean = v2c.actor_mean(v2c._actor_features_v2a(obs, w))
                ref_action = ref.act_inference_with_preference(obs, w)
                c_action = v2c.act_inference_with_preference(obs, w)
                ref_value = ref.value_with_preference(obs, w)
                c_value = v2c.value_with_preference(obs, w)
    
                route = v2c.routing_weights(w)
                hfilm = v2c._film_hidden(obs, w)
                expert_outputs = torch.stack(
                    [expert(hfilm) for expert in v2c.residual_experts], dim=1
                )
                routed_residual = torch.sum(route.unsqueeze(-1) * expert_outputs, dim=1)
    
            checks = {}
    
            checks["deterministic_action_identity"] = {
                "max_abs_diff": ma(c_action - ref_action),
                "pass": ma(c_action - ref_action) <= TOL,
            }
            checks["pre_tanh_mean_identity"] = {
                "max_abs_diff": ma(c_mean - ref_mean),
                "pass": ma(c_mean - ref_mean) <= TOL,
            }
            checks["critic_identity"] = {
                "max_abs_diff": ma(c_value - ref_value),
                "pass": ma(c_value - ref_value) <= TOL,
            }
    
            uniform = torch.full_like(route, 1.0 / v2c.NUM_EXPERTS)
            checks["router_uniform_identity_init"] = {
                "max_abs_diff_from_uniform": ma(route - uniform),
                "router_weight_norm": float(v2c.preference_router.weight.norm().detach().cpu()),
                "router_bias_norm": float(v2c.preference_router.bias.norm().detach().cpu()),
                "pass": (
                    ma(route - uniform) == 0.0
                    and float(v2c.preference_router.weight.norm().detach().cpu()) == 0.0
                    and float(v2c.preference_router.bias.norm().detach().cpu()) == 0.0
                ),
            }
    
            expert_fc2_norms = []
            expert_output_max = []
            for expert in v2c.residual_experts:
                expert_fc2_norms.append(
                    {
                        "weight": float(expert.fc2.weight.norm().detach().cpu()),
                        "bias": float(expert.fc2.bias.norm().detach().cpu()),
                    }
                )
            for i in range(v2c.NUM_EXPERTS):
                expert_output_max.append(ma(expert_outputs[:, i]))
            checks["expert_zero_residual_init"] = {
                "fc2_norms": expert_fc2_norms,
                "expert_output_max_abs": expert_output_max,
                "routed_residual_max_abs": ma(routed_residual),
                "pass": (
                    all(x["weight"] == 0.0 and x["bias"] == 0.0 for x in expert_fc2_norms)
                    and max(expert_output_max) == 0.0
                    and ma(routed_residual) == 0.0
                ),
            }
    
            torch.manual_seed(args.seed + 1)
            with torch.no_grad():
                dist = ref._pre_tanh_dist_with_preference(obs, w)
                latent = dist.sample()
                ref_logp = ref.logp_from_pre_tanh_with_preference(obs, w, latent)
                c_logp = v2c.logp_from_pre_tanh_with_preference(obs, w, latent)
    
            checks["same_latent_logprob_identity"] = {
                "max_abs_diff": ma(c_logp - ref_logp),
                "pass": ma(c_logp - ref_logp) <= TOL,
            }
    
            with torch.no_grad():
                action, old_logp, latent2 = v2c.act_with_preference_latent(obs, w)
                new_logp = v2c.logp_from_pre_tanh_with_preference(obs, w, latent2)
            ratio = torch.exp(new_logp - old_logp)
            checks["ppo_ratio_invariant"] = {
                "max_abs_ratio_minus_1": ma(ratio - 1),
                "pass": ma(ratio - 1) <= 1e-6,
            }
    
            _ = env.reset(seed=args.seed + 7)
            env.step(action)
            raw = env.unwrapped.action_manager._terms["joint_pos"].raw_actions.detach()
            checks["env_raw_action_identity"] = {
                "max_abs_diff": ma(raw - action),
                "pass": ma(raw - action) <= TOL,
            }
    
            # Verify exact V2-B tensor inheritance.
            rs = ref.state_dict()
            cs = v2c.state_dict()
            max_existing = 0.0
            exact_existing = True
            for key, value in rs.items():
                if key in cs and cs[key].shape == value.shape:
                    diff = float((cs[key] - value).abs().max().cpu())
                    max_existing = max(max_existing, diff)
                    exact_existing = exact_existing and diff == 0.0
    
            checks["v2b_parameter_identity"] = {
                "max_abs_diff": max_existing,
                "pass": exact_existing,
            }
    
            ref_names = set(dict(ref.named_parameters()))
            c_names = set(dict(v2c.named_parameters()))
            new_names = sorted(c_names - ref_names)
            expected_new = {
                "preference_router.weight",
                "preference_router.bias",
            }
            for i in range(v2c.NUM_EXPERTS):
                expected_new.update(
                    {
                        f"residual_experts.{i}.fc1.weight",
                        f"residual_experts.{i}.fc1.bias",
                        f"residual_experts.{i}.fc2.weight",
                        f"residual_experts.{i}.fc2.bias",
                    }
                )
            checks["treatment_isolation"] = {
                "new_parameter_names": new_names,
                "expected_new_parameter_names": sorted(expected_new),
                "pass": set(new_names) == expected_new,
            }
    
            # Preference router is deterministic, normalized and finite on all standard probes.
            wp = torch.tensor(PREFS, device="cuda")
            with torch.no_grad():
                routes = v2c.routing_weights(wp)
            checks["router_simplex_contract"] = {
                "row_sum_max_abs_error": ma(routes.sum(-1) - 1.0),
                "min_weight": float(routes.min().cpu()),
                "max_weight": float(routes.max().cpu()),
                "all_finite": bool(torch.isfinite(routes).all()),
                "pass": (
                    ma(routes.sum(-1) - 1.0) <= TOL
                    and float(routes.min().cpu()) >= 0.0
                    and bool(torch.isfinite(routes).all())
                ),
            }
    
            # Checkpoint round trip.
            checkpoint = args.output_dir / "v2c0_init.pt"
            before = state_hash(v2c)
            torch.save(
                {
                    "model": v2c.state_dict(),
                    "reference": "v2b0_init",
                    "training_steps": 0,
                    "architecture": "V2-B + four preference-gated zero-output residual experts",
                },
                checkpoint,
            )
    
            restored = V2CPreferenceGatedResidualActorCritic(obs.shape[-1], action_dim).cuda()
            restored.load_state_dict(
                torch.load(checkpoint, map_location="cuda", weights_only=False)["model"]
            )
            restored.eval()
            with torch.no_grad():
                restored_action = restored.act_inference_with_preference(obs, w)
                restored_value = restored.value_with_preference(obs, w)
    
            checks["checkpoint_roundtrip"] = {
                "action_max_abs_diff": ma(restored_action - c_action),
                "value_max_abs_diff": ma(restored_value - c_value),
                "state_hash_equal": before == state_hash(restored),
                "pass": (
                    ma(restored_action - c_action) <= TOL
                    and ma(restored_value - c_value) <= TOL
                    and before == state_hash(restored)
                ),
            }
    
            checks["objective_contract"] = scalarization_check()
            checks["all_finite"] = {
                "pass": bool(
                    torch.isfinite(c_action).all()
                    and torch.isfinite(c_value).all()
                    and torch.isfinite(old_logp).all()
                )
            }
    
            # Paired no-update environment smoke against exact V2-B reference.
            cur, _ = env.reset(seed=args.seed + 99)
            cur = ot(cur).cuda()
            max_action_diff = 0.0
            terminations = 0
            finite = True
            for _ in range(STEPS):
                with torch.no_grad():
                    ca = v2c.act_inference_with_preference(cur, w)
                    ra = ref.act_inference_with_preference(cur, w)
                max_action_diff = max(max_action_diff, ma(ca - ra))
                finite = finite and bool(torch.isfinite(ca).all())
                nxt, _, te, tr, _ = env.step(ca)
                terminations += int((te | tr).sum().item())
                cur = ot(nxt).cuda()
    
            checks["no_update_smoke"] = {
                "steps": STEPS,
                "num_envs": NENV,
                "max_action_diff_vs_v2b": max_action_diff,
                "termination_events": terminations,
                "all_finite": finite,
                "pass": max_action_diff <= TOL and finite,
            }
    
            passed = all(v.get("pass", False) for v in checks.values())
    
            report = {
                "schema": "v2c0_function_preserving_gate_v1",
                "status": "V2-C0 PASS" if passed else "V2-C0 FAIL",
                "training_enabled": False,
                "optimizer_steps": 0,
                "reference_checkpoint": str(V2B_INIT.relative_to(ROOT)),
                "reference_definition": "frozen V2-B0 function-preserving initialization",
                "treatment": (
                    "retain complete V2-B actor/critic; add four 128->32->128 private "
                    "residual experts after post-FiLM hidden state and a deterministic "
                    "preference-only softmax router; expert output layers and router logits "
                    "zero-initialized"
                ),
                "checks": checks,
                "authorization": {
                    "v2c1_authorized": bool(passed),
                    "training_in_this_gate": False,
                },
                "checkpoint": {
                    "path": str(checkpoint.relative_to(ROOT)),
                    "sha256": sha(checkpoint),
                },
                "provenance": {
                    "v2b_init_sha256": sha(V2B_INIT),
                    "v2c_module_sha256": sha(ROOT / "talon_rl/models/conditioning/residual_experts.py"),
                    "gate_script_sha256": sha(Path(__file__).resolve()),
                    "contract_sha256": sha(ROOT / "docs/contracts/preference_architectures/v2c-contract.md"),
                },
            }
    
            out = args.output_dir / "v2c0_report.json"
            out.write_text(json.dumps(report, indent=2) + "\n")
            (args.output_dir / "PROVENANCE_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "status": "FROZEN_BY_HASH",
                        "report_sha256": sha(out),
                        "checkpoint_sha256": sha(checkpoint),
                    },
                    indent=2,
                )
                + "\n"
            )
    
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "authorization": report["authorization"],
                        "checks": checks,
                    },
                    indent=2,
                )
            )
        finally:
            if env is not None:
                env.close()
            app.close()
    
    
    if True:
        main()

def run_v2c1_modular_authority_screen():
    """Run former v2c1_modular_authority_screen.py stage."""
    from pathlib import Path
    import argparse,json,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RV1_INIT=ROOT/"runs/rv1_a_implementation_gate-2026-09-23/rv1_a_init.pt"
    V2A_INIT=ROOT/"runs/v2a0_function_preserving_gate-2026-09-23/v2a0_init.pt"
    V2C_INIT=ROOT/"runs/v2c0_function_preserving_gate-2026-09-24/v2c0_init.pt"
    OUT_DEFAULT=ROOT/"runs/v2c1_modular_authority-2026-09-24"
    ORDER=("T","A","O","S","C")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    G=.99;H=32;SUP_H=64;NENV=8;POOL_MAX=24
    ANCHOR_CANDIDATES=12;ANCHOR_PHASE_K=3;ADAPT_PHASE_K=3
    SNAPS=(0,10,25,50,75)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2=1.0):
        A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
    def select_diverse(summaries,k):
        X=np.asarray(summaries,np.float64);X=(X-X.mean(0))/(X.std(0)+1e-6)
        if len(X)<=k:return list(range(len(X)))
        sel=[int(np.argmax(np.mean(X*X,axis=1)))]
        mind=np.mean((X-X[sel[0]])**2,axis=1)
        while len(sel)<k:
            mind[sel]=-1;q=int(np.argmax(mind));sel.append(q)
            mind=np.minimum(mind,np.mean((X-X[q])**2,axis=1))
        return sorted(sel)
    def pref_batch(update,device):
        labs=[ORDER[(update+i)%len(ORDER)] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_actor(env,m,w,mgr,seed,stochastic):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];pre=[];old=[];rw=[];dn=[]
        with torch.no_grad():
            for _ in range(H):
                if stochastic:a,lp,u=m.act_with_preference_latent(cur,w)
                else:a=m.act_inference_with_preference(cur,w);lp=u=None
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda())
                if stochastic:pre.append(u);old.append(lp)
                cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool()
        out={"obs":torch.cat(ob),"w":w.repeat(H,1),"rt":rt,"dt":dt,"next_obs":cur,
             "termination_fraction":float(dt.any(0).float().mean().cpu())}
        if stochastic:out["u"]=torch.cat(pre);out["old"]=torch.cat(old)
        return out
    
    def collect_support64(env,m,w,mgr,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        ob=[];rw=[];dn=[];cmd=[]
        with torch.no_grad():
            for _ in range(SUP_H):
                cmd.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                dn.append((te|tr).cuda());cur=ot(nxt).cuda()
        rt=torch.stack(rw);dt=torch.stack(dn).bool();obs=torch.stack(ob) # [64,E,D]
        C=np.stack(cmd) # [64,E,3]
        phases=[]
        for pi,(st,en) in enumerate(((0,32),(32,64))):
            Y=trunc(rt[st:en],dt[st:en]).reshape(-1,4).detach()
            po=obs[st:en].reshape(-1,obs.shape[-1]);pw=w.repeat(en-st,1)
            with torch.no_grad():F=m.critic_body(m._with_w(po,pw)).detach()
            cc=C[st:en].reshape(-1,C.shape[-1])
            summary=np.r_[cc.mean(0),cc.std(0),w.mean(0).cpu().numpy(),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),float(pi)]
            phases.append({"F":F.cpu().numpy(),"Y":Y.cpu().numpy(),"summary":summary.astype(np.float32),
                           "phase":"early" if pi==0 else "late"})
        return phases
    
    def fit_expanded_current_policy(m,anchor_units,adaptive_pools):
        support=[]
        counts={}
        for phase in ("early","late"):
            au=anchor_units[phase]
            ad=adaptive_pools[phase]
            support += [(u["F"],u["Y"]) for u in au]
            support += [(u["F"],u["Y"]) for u in ad]
            counts[phase]={"anchor_units":len(au),"adaptive_units":len(ad)}
        F=np.concatenate([x[0] for x in support]);Y=np.concatenate([x[1] for x in support])
        W,b=ridge(F,Y,1.0)
        with torch.no_grad():
            m.critic_head.weight.copy_(torch.tensor(W,dtype=m.critic_head.weight.dtype,device="cuda"))
            m.critic_head.bias.copy_(torch.tensor(b,dtype=m.critic_head.bias.dtype,device="cuda"))
        return counts
    
    def pairwise_dist(actions):
        labs=list(actions);v=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:v.append(float(torch.linalg.vector_norm(actions[a]-actions[b],dim=1).mean().cpu()))
        return {"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v))}
    def routing_pairwise(routes):
        labs=list(routes);vals=[]
        for i,a in enumerate(labs):
            for b in labs[i+1:]:
                vals.append(float(torch.linalg.vector_norm(routes[a]-routes[b],dim=-1).mean().detach().cpu()))
        return {"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals))}
    
    def sensitivity(m,probe):
        total={};masked={};routes={};expert_norms={};single_mask_effects={};residuals={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                h=m._film_hidden(probe,w);e=m.preference_embedding(w);g=m.routing_weights(w)
                outs=torch.stack([ex(h) for ex in m.residual_experts],dim=1)
                r=(g.unsqueeze(-1)*outs).sum(1)
                total[lab]=torch.tanh(m.actor_mean(torch.cat((h+r,e),dim=-1)))*m.ACTION_CLIP
                masked[lab]=torch.tanh(m.actor_mean(torch.cat((h,e),dim=-1)))*m.ACTION_CLIP
                routes[lab]=g
                residuals[lab]=r
                expert_norms[lab]=[float(torch.linalg.vector_norm(outs[:,j],dim=-1).mean().cpu()) for j in range(m.NUM_EXPERTS)]
                effects=[]
                for j in range(m.NUM_EXPERTS):
                    rj=r-g[:,j:j+1]*outs[:,j]
                    aj=torch.tanh(m.actor_mean(torch.cat((h+rj,e),dim=-1)))*m.ACTION_CLIP
                    effects.append(float(torch.linalg.vector_norm(total[lab]-aj,dim=-1).mean().cpu()))
                single_mask_effects[lab]=effects
    
        pair_total=pairwise_dist(total);pair_masked=pairwise_dist(masked)
        authority=float(np.mean([torch.linalg.vector_norm(total[k]-masked[k],dim=1).mean().cpu() for k in ORDER]))
        route_mean={k:routes[k].mean(0).detach().cpu().tolist() for k in ORDER}
        route_entropy={k:float((-(routes[k]*torch.log(routes[k]+1e-12)).sum(-1)).mean().cpu()) for k in ORDER}
        rdist=routing_pairwise({k:routes[k] for k in ("T","A","O","S")})
        uniform_dev=max(float(torch.max(torch.abs(routes[k]-0.25)).cpu()) for k in ("T","A","O","S"))
    
        # Expert-output diversity across experts and heavy preferences.
        div=[]
        with torch.no_grad():
            for lab in ("T","A","O","S"):
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                h=m._film_hidden(probe,w)
                outs=[ex(h) for ex in m.residual_experts]
                for i in range(m.NUM_EXPERTS):
                    for j in range(i+1,m.NUM_EXPERTS):
                        div.append(float(torch.linalg.vector_norm(outs[i]-outs[j],dim=-1).mean().cpu()))
        expert_output_diversity={"mean":float(np.mean(div)),"min":float(np.min(div)),"max":float(np.max(div))}
    
        # Total da/dw.
        w=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        a=m.act_inference_with_preference(probe,w);rows=[]
        for j in range(a.shape[1]):rows.append(torch.autograd.grad(a[:,j].mean(),w,retain_graph=True)[0])
        jac=torch.stack(rows,dim=1)
    
        # Modular-path-masked da/dw = exact current V2-B path.
        wm=torch.tensor(PREFS["C"],device="cuda").repeat(len(probe),1).requires_grad_(True)
        hm=m._film_hidden(probe,wm);em=m.preference_embedding(wm)
        am=torch.tanh(m.actor_mean(torch.cat((hm,em),dim=-1)))*m.ACTION_CLIP;mrows=[]
        for j in range(am.shape[1]):mrows.append(torch.autograd.grad(am[:,j].mean(),wm,retain_graph=True)[0])
        mjac=torch.stack(mrows,dim=1)
    
        mask_mat=np.array([single_mask_effects[k] for k in ("T","A","O","S")],float)
        norm_mat=np.array([expert_norms[k] for k in ("T","A","O","S")],float)
        return {
            "pairwise_action_distance":pair_total,
            "modular_masked_pairwise_action_distance":pair_masked,
            "jacobian_fro_mean":float(torch.linalg.matrix_norm(jac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "modular_masked_jacobian_fro_mean":float(torch.linalg.matrix_norm(mjac,ord="fro",dim=(1,2)).mean().detach().cpu()),
            "modular_action_authority_mean":authority,
            "routing_matrix":route_mean,
            "routing_entropy":route_entropy,
            "routing_pairwise_distance_heavy":rdist,
            "routing_max_abs_from_uniform":uniform_dev,
            "expert_residual_norm_matrix":{"rows":["T","A","O","S"],"values":norm_mat.tolist()},
            "expert_residual_norm_max":float(norm_mat.max()),
            "expert_residual_norm_cv":float(norm_mat.std()/(norm_mat.mean()+1e-12)),
            "expert_output_diversity":expert_output_diversity,
            "single_expert_mask_effect_matrix":{"rows":["T","A","O","S"],"values":mask_mat.tolist()},
            "single_expert_mask_pref_std_mean":float(mask_mat.std(axis=0).mean()),
            "router_weight_norm":float(m.preference_router.weight.norm().detach().cpu()),
            "router_bias_norm":float(m.preference_router.bias.norm().detach().cpu()),
        }
    
    def expert_gradient_specialization(env,m,mgr,seed):
        from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
        matrix=[];router=[];ratio_err=[]
        for li,lab in enumerate(("T","A","O","S")):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            b=collect_actor(env,m,w,mgr,seed+li*701,True)
            with torch.no_grad():
                vt=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
                nv=m.value_with_preference(b["next_obs"],w)
                adv,_=vector_gae(b["rt"],vt,nv,b["dt"],lam=.95)
            ratio=torch.exp(m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])-b["old"].detach())
            ratio_err.append(float((ratio-1).abs().max().detach().cpu()))
            loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),b["w"])
            row=[]
            for ex in m.residual_experts:
                ps=list(ex.parameters());gs=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
                norm=torch.sqrt(sum((g.detach()**2).sum() for g in gs if g is not None)) if any(g is not None for g in gs) else torch.tensor(0.,device="cuda")
                row.append(float(norm.cpu()))
            rps=list(m.preference_router.parameters());rgs=torch.autograd.grad(loss,rps,retain_graph=False,allow_unused=True)
            rn=torch.sqrt(sum((g.detach()**2).sum() for g in rgs if g is not None)) if any(g is not None for g in rgs) else torch.tensor(0.,device="cuda")
            router.append(float(rn.cpu()));matrix.append(row)
        M=np.asarray(matrix,float);shares=M/(M.sum(1,keepdims=True)+1e-12)
        pd=[]
        for i in range(4):
            for j in range(i+1,4):pd.append(float(np.linalg.norm(shares[i]-shares[j])))
        return {"rows":["T","A","O","S"],"expert_gradient_norms":M.tolist(),"expert_gradient_shares":shares.tolist(),
                "gradient_share_pairwise_distance":{"mean":float(np.mean(pd)),"min":float(np.min(pd)),"max":float(np.max(pd))},
                "router_gradient_norms":router,"max_ratio_error":float(max(ratio_err))}
    
    def fresh_phase_audit(env,m,mgr,seed):
        phase_ev={"early":[],"late":[]};phase_bias={"early":[],"late":[]};surv=[]
        for qi,lab in enumerate(ORDER):
            w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
            units=collect_support64(env,m,w,mgr,seed+qi*101)
            for unit in units:
                with torch.no_grad():
                    # Features already current-policy and head is linear.
                    F=torch.tensor(unit["F"],device="cuda",dtype=m.critic_head.weight.dtype)
                    V=m.critic_head(F).cpu().numpy()
                Y=unit["Y"];ph=unit["phase"]
                phase_ev[ph].extend(ev(Y[:,j],V[:,j]) for j in range(4))
                phase_bias[ph].extend(float(np.mean(V[:,j]-Y[:,j])) for j in range(4))
            surv.append(1.0) # deterministic support rollouts are diagnostic; detailed terminations in independent revalidation
        out={}
        for ph in ("early","late"):
            a=np.asarray(phase_ev[ph]);b=np.asarray(phase_bias[ph])
            out[ph]={"ev_mean":float(a.mean()),"negative_fraction":float((a<0).mean()),"mean_abs_bias":float(np.abs(b).mean())}
        out["combined_negative_fraction"]=float(np.mean(np.r_[np.asarray(phase_ev["early"])<0,np.asarray(phase_ev["late"])<0]))
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--updates",type=int,default=75);ap.add_argument("--seed",type=int,default=73001)
        ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae,scalarized_late_weighted_ppo
            from talon_rl.models.conditioning.residual_experts_experts import V2CPreferenceGatedResidualActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=args.seed);o=ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=V2CPreferenceGatedResidualActorCritic(o.shape[-1],ad).cuda()
            init_path=V2C_INIT
            m.load_state_dict(torch.load(init_path,map_location="cuda",weights_only=False)["model"]);m.train()
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film") or n.startswith("preference_router") or n.startswith("residual_experts")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            # Freeze anchor seed/preference specs once.
            cand=[]
            for k in range(ANCHOR_CANDIDATES):
                _,w=pref_batch(k,torch.device("cuda"));seed=args.seed+10000+k*137
                units=collect_support64(env,m,w,mgr,seed);cand.append((k,seed,units))
            # Select 6 seed specs on combined early+late summary geometry.
            comb=[np.r_[x[2][0]["summary"],x[2][1]["summary"]] for x in cand]
            spec_idx=select_diverse(comb,6);anchor_specs=[(cand[i][0],cand[i][1]) for i in spec_idx]
    
            adaptive_pools={"early":[],"late":[]};adaptive_summaries={"early":[],"late":[]}
            # initialize adaptive pools from all initialization candidates
            for _,_,units in cand:
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
    
            def current_anchor_units():
                out={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=pref_batch(k,torch.device("cuda"))
                    for u in collect_support64(env,m,w,mgr,seed):
                        out[u["phase"]].append(u)
                return out
            fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
    
            rows=[];snaps={}
            def audit(tag):
                m.eval();snaps[str(tag)]={"sensitivity":sensitivity(m,probe),
                    "phase_critic":fresh_phase_audit(env,m,mgr,args.seed+500000+int(tag)*1000)}
                torch.save({"model":m.state_dict(),"update":int(tag),"seed":args.seed},args.output_dir/f"model_{tag}.pt");m.train()
            audit(0)
    
            for uidx in range(1,args.updates+1):
                labs,w=pref_batch(uidx,torch.device("cuda"))
                main=collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                # one recent adaptive 64-step support candidate contributes one early + one late unit
                _,ws=pref_batch(uidx+17,torch.device("cuda"))
                units=collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u);adaptive_summaries[ph].append(u["summary"])
                    if len(adaptive_pools[ph])>POOL_MAX:
                        adaptive_pools[ph].pop(0);adaptive_summaries[ph].pop(0)
                selected=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                ratio=torch.exp(m.logp_from_pre_tanh_with_preference(main["obs"],main["w"],main["u"])-main["old"].detach())
                ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                loss=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                opt.zero_grad(set_to_none=True);loss.backward()
                pg=float(m.actor_body[0].weight.grad[:,m.physical_obs_dim:].norm().detach().cpu())
                rwgrad=float(m.preference_router.weight.grad.norm().detach().cpu()) if m.preference_router.weight.grad is not None else 0.0
                expert_grads=[]
                for ex in m.residual_experts:
                    vals=[p.grad.detach().pow(2).sum() for p in ex.parameters() if p.grad is not None]
                    expert_grads.append(float(torch.sqrt(sum(vals)).cpu()) if vals else 0.0)
                total=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                post_refresh=None
                # Foundation V2: refresh after actor step only when an evaluation checkpoint is saved.
                if uidx in SNAPS:
                    post_refresh=fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                rows.append({"update":uidx,"loss":float(loss.detach().cpu()),"ratio_maxerr":ratio_err,
                             "preference_input_grad_norm":pg,"router_weight_grad_norm":rwgrad,"expert_grad_norms":expert_grads,"actor_grad_norm_preclip":total,
                             "termination_fraction":main["termination_fraction"],"selected":selected,"post_actor_refresh":post_refresh})
                if uidx in SNAPS:audit(uidx)
    
            final=snaps[str(args.updates)];initial=snaps["0"]
            term=np.array([r["termination_fraction"] for r in rows]);ratio=np.array([r["ratio_maxerr"] for r in rows])
            rwg=np.array([r["router_weight_grad_norm"] for r in rows])
            eg=np.asarray([r["expert_grad_norms"] for r in rows],float)
            crit=final["phase_critic"];sens=final["sensitivity"]
            grad_spec=expert_gradient_specialization(env,m,mgr,args.seed+900000)
    
            pair_total=sens["pairwise_action_distance"]["mean"]
            pair_mask=sens["modular_masked_pairwise_action_distance"]["mean"]
            jac_total=sens["jacobian_fro_mean"]
            jac_mask=sens["modular_masked_jacobian_fro_mean"]
            mask_reduces=(pair_total>pair_mask+1e-4) or (jac_total>jac_mask+1e-4)
    
            criteria={
              "routing_pairwise_separation":sens["routing_pairwise_distance_heavy"]["mean"]>=0.02,
              "routing_nonuniform":sens["routing_max_abs_from_uniform"]>=0.02,
              "expert_residual_nonzero":sens["expert_residual_norm_max"]>1e-3,
              "expert_output_diverse":sens["expert_output_diversity"]["mean"]>1e-3,
              "expert_gradient_specialization":grad_spec["gradient_share_pairwise_distance"]["mean"]>=0.02,
              "router_gradient_observed":float(np.max(rwg))>1e-7,
              "modular_action_authority_nonzero":sens["modular_action_authority_mean"]>1e-4,
              "modular_mask_reduces_preference_authority":bool(mask_reduces),
              "single_expert_mask_preference_specific":sens["single_expert_mask_pref_std_mean"]>1e-5,
              "early_critic_preserved":crit["early"]["ev_mean"]>0 and crit["early"]["negative_fraction"]<=.25,
              "late_critic_preserved":crit["late"]["ev_mean"]>0 and crit["late"]["negative_fraction"]<=.25,
              "combined_negative_fraction":crit["combined_negative_fraction"]<=.25,
              "ppo_ratio_invariant":float(ratio.max())<=1e-4 and grad_spec["max_ratio_error"]<=1e-4,
              "survival_preserved":float(term[-10:].mean())<.5}
            passed=all(criteria.values())
    
            report={"schema":"v2c1_modular_authority_screen_v1","seed":args.seed,"updates":args.updates,
                    "training_scope":"V2-C1 modular authority/specialization only; semantics blocked",
                    "actor_ppo_objectives_changed":False,"architecture":"v2c",
                    "reference_checkpoint":str(V2C_INIT.relative_to(ROOT)),
                    "anchor_specs":anchor_specs,"rows":rows,"snapshots":snaps,
                    "expert_gradient_specialization":grad_spec}
            report["summary"]={
              "status":"V2-C1 PASS" if passed else "V2-C1 FAIL",
              "criteria":criteria,
              "initial":initial,
              "final":final,
              "max_router_weight_grad_norm":float(np.max(rwg)),
              "median_router_weight_grad_norm":float(np.median(rwg)),
              "max_expert_grad_norm_per_expert":eg.max(axis=0).tolist(),
              "median_expert_grad_norm_per_expert":np.median(eg,axis=0).tolist(),
              "last10_termination_fraction":float(term[-10:].mean()),
              "max_ratio_error":float(max(ratio.max(),grad_spec["max_ratio_error"])),
              "v2c2_authorized":bool(passed),
              "semantic_judgement_performed":False}
            (args.output_dir/"v2c1_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report["summary"],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "v2c0_function_preserving_gate": run_v2c0_function_preserving_gate,
    "v2c1_modular_authority_screen": run_v2c1_modular_authority_screen,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()

if __name__ == '__main__':
    main()
