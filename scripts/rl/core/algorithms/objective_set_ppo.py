"""Objective-set PPO: the V3 objective-set MORL loss inside the M0 (stock rsl_rl) PPO shell.

Used by the V4-C TeacherV4 trainer. Everything here is pure torch so the
C0b gates in tests/core/algorithms/test_objective_set_ppo.py can check it
without Isaac. Frozen contract:
docs/verdicts/teacher_v4/teacher-v4-c0-rollout-batch-audit.md.

What matches rsl_rl (M0): GAE form and time-out bootstrap, one permutation
per update reused across epochs, clipped surrogate, clipped value loss,
Gaussian entropy, the adaptive-KL LR rule applied per minibatch before the
step, max grad norm, all constants.

Declared differences (the V4 treatment or forced by it):
- actor loss is V3's M * sum_i w_i * clipped-surrogate(A_i) over the active set;
- critic regresses one value per active objective (query critic);
- advantages are centered per objective and scaled by the std of the
  scalarized advantage sum_i w_i A_i, the MORL analogue of rsl_rl's
  full-batch normalization that keeps the divisor-calibrated ratios between
  objectives (per-objective standardization would override them);
- actor and critic have separate Adam optimizers and separate grad-norm
  clipping, so the value loss cannot scale actor steps (authority
  isolation); both follow the same adaptive LR;
- tanh-squashed Gaussian (V3 lineage) with log-std; entropy is the
  pre-tanh Gaussian's. No log-std clamp: it is logged and gated instead.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import Tensor

NUM_OBJECTIVES = 4


@dataclass(frozen=True)
class PPOConfig:
    num_steps: int = 24
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.01
    lr: float = 1e-3
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


# ----- objective sets -----

def sample_objective_sets(n: int, cardinalities: tuple[int, ...], gen: torch.Generator, device=None, num_objectives: int = NUM_OBJECTIVES,
                          required: int | None = None):
    """Per env: cardinality uniform over `cardinalities`, subset uniform over
    subsets of that size, weights by V3's modes (20% center, 40% one objective
    at 0.70, 40% Dirichlet(1)). Returns (w [n,4], mask [n,4] bool); inactive
    objectives have weight exactly 0, so ids can stay arange(K) (TeacherV4
    treats zero weights as padding). `required` (F8 task-anchored support)
    restricts subsets to those containing that objective; None leaves the
    draws exactly as before."""
    w = torch.zeros(n, num_objectives)
    for b in range(n):
        m = cardinalities[int(torch.randint(len(cardinalities), (1,), generator=gen))]
        combos = list(itertools.combinations(range(num_objectives), m))
        if required is not None:
            combos = [c for c in combos if required in c]
        sub = list(combos[int(torch.randint(len(combos), (1,), generator=gen))])
        q = float(torch.rand((), generator=gen))
        if m == 1 or q < 0.2:
            x = torch.full((m,), 1.0 / m)
        elif q < 0.6:
            x = torch.full((m,), 0.30 / (m - 1))
            x[int(torch.randint(m, (1,), generator=gen))] = 0.70
        else:
            x = -torch.rand(m, generator=gen).log()
            x = x / x.sum()
        w[b, sub] = x
    return w.to(device), (w > 0).to(device)


# ----- returns -----

def gae(rewards: Tensor, values: Tensor, last_values: Tensor, dones: Tensor, gamma: float, lam: float):
    """rsl_rl RolloutStorage.compute_returns, per objective.
    rewards/values [T,N,K], last_values [N,K], dones [T,N] -> (returns, advantages) [T,N,K]."""
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    adv = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        nv = last_values if t == T - 1 else values[t + 1]
        nt = (1.0 - dones[t].float()).unsqueeze(-1)
        delta = rewards[t] + nt * gamma * nv - values[t]
        adv = delta + nt * gamma * lam * adv
        returns[t] = adv + values[t]
    return returns, returns - values


def normalize_advantages(adv: Tensor, w: Tensor, mask: Tensor) -> Tensor:
    """adv, w, mask [B,K]. Center each objective over its active samples; one
    shared scale, the std of sum_i w_i A_i."""
    m = mask.float()
    mean = (adv * m).sum(0) / m.sum(0).clamp_min(1.0)
    scale = (w * adv).sum(-1).std() + 1e-8
    return (adv - mean) * m / scale


# ----- losses -----

def actor_surrogate(logp: Tensor, old_logp: Tensor, adv: Tensor, w: Tensor, mask: Tensor, clip: float):
    """V3's M * sum_i w_i min(r A_i, clip(r) A_i), as a loss (negated mean). Returns (loss, ratio)."""
    ratio = torch.exp(logp - old_logp)
    r = ratio.unsqueeze(-1)
    po = torch.minimum(r * adv, r.clamp(1 - clip, 1 + clip) * adv)
    card = mask.float().sum(-1)
    return -(card * (w * mask.float() * po).sum(-1)).mean(), ratio


def actor_stream_surrogates(logp: Tensor, old_logp: Tensor, adv: Tensor, w: Tensor, mask: Tensor, clip: float) -> Tensor:
    """Per-objective terms of the existing scalarized surrogate, before summing."""
    ratio = torch.exp(logp - old_logp)
    r = ratio.unsqueeze(-1)
    po = torch.minimum(r * adv, r.clamp(1 - clip, 1 + clip) * adv)
    return -(mask.float().sum(-1).unsqueeze(-1) * w * mask.float() * po).mean(0)


def value_loss(values: Tensor, old_values: Tensor, returns: Tensor, mask: Tensor, clip: float) -> Tensor:
    """rsl_rl clipped value loss per objective, averaged over active entries."""
    clipped = old_values + (values - old_values).clamp(-clip, clip)
    per = torch.maximum((values - returns).pow(2), (clipped - returns).pow(2))
    m = mask.float()
    return (per * m).sum() / m.sum()


def gaussian_kl(mu_old: Tensor, sig_old: Tensor, mu: Tensor, sig: Tensor) -> Tensor:
    """rsl_rl's KL(old || new) expression, summed over action dims, per sample."""
    return torch.sum(torch.log(sig / sig_old + 1.0e-5) + (sig_old.square() + (mu_old - mu).square()) / (2.0 * sig.square()) - 0.5, dim=-1)


def adapt_lr(lr: float, kl_mean: float, desired_kl: float) -> float:
    """rsl_rl's adaptive schedule, verbatim."""
    if kl_mean > desired_kl * 2.0:
        return max(1e-5, lr / 1.5)
    if kl_mean < desired_kl / 2.0 and kl_mean > 0.0:
        return min(1e-2, lr * 1.5)
    return lr


def compose_actor_gradients(task_grads, non_task_grads, max_norm: float, mode: str):
    """FC-G treatment on one minibatch; None denotes an unused parameter."""
    if mode not in ("GLOBAL", "SPLIT", "SPLIT-NORM-MATCHED"):
        raise ValueError(f"unknown gradient composition: {mode}")
    task = [torch.zeros_like(n) if t is None else t for t, n in zip(task_grads, non_task_grads)]
    non = [torch.zeros_like(t) if n is None else n for t, n in zip(task_grads, non_task_grads)]
    # The caller supplies dense gradients; the lists must cover the same actor parameters.
    nt = torch.stack([torch.sum(x.double().square()) for x in task]).sum().sqrt()
    nn = torch.stack([torch.sum(x.double().square()) for x in non]).sum().sqrt()
    total = [t + n for t, n in zip(task, non)]
    n0 = torch.stack([torch.sum(x.double().square()) for x in total]).sum().sqrt()
    coef = lambda n: min(1.0, max_norm / (float(n) + 1e-6))
    ct, cn, c0 = coef(nt), coef(nn), coef(n0)
    g0 = [c0 * x for x in total]
    g1 = [ct * t + cn * n for t, n in zip(task, non)]
    n1 = torch.stack([torch.sum(x.double().square()) for x in g1]).sum().sqrt()
    scale = float(torch.stack([torch.sum(x.double().square()) for x in g0]).sum().sqrt() / n1) if float(n1) > 0 else 0.0
    g2 = [scale * x for x in g1]
    dot = sum(torch.sum(a.double() * b.double()) for a, b in zip(g0, g2))
    n2 = torch.stack([torch.sum(x.double().square()) for x in g2]).sum().sqrt()
    cos = float(dot / (torch.stack([torch.sum(x.double().square()) for x in g0]).sum().sqrt() * n2)) if float(n0) > 0 and float(n2) > 0 else (1.0 if float(n0) == 0 and float(n2) == 0 else 0.0)
    selected = {"GLOBAL": g0, "SPLIT": g1, "SPLIT-NORM-MATCHED": g2}[mode]
    metrics = {"fcg_task_norm": float(nt), "fcg_non_task_norm": float(nn), "fcg_global_norm": float(n0),
               "fcg_split_norm": float(n1), "fcg_selected_norm": float(torch.stack([torch.sum(x.double().square()) for x in selected]).sum().sqrt()),
               "fcg_c_task": ct, "fcg_c_non_task": cn, "fcg_c_global": c0,
               "fcg_c_ratio": ct / cn, "fcg_cos_global_normmatched": max(-1.0, min(1.0, cos)),
               "fcg_coef_diff_frac": float(abs(ct / cn - 1) > 0.05),
               "fcg_global_suppresses_non_task_frac": float(c0 < cn - 1e-6)}
    return selected, metrics


# ----- update -----

def update(model, actor_opt, critic_opt, batch: dict, cfg: PPOConfig, lr: float, gen: torch.Generator | None = None,
           task_allowed_param_ids: set[int] | None = None, gradient_composition: str | None = None,
           gradient_audit_sink: list[dict] | None = None) -> tuple[float, dict]:
    """One PPO update over a flattened rollout. batch keys, each [B,...]:
    obs, env, ids, w, mask, u, old_logp, old_mu, old_sigma, old_values,
    returns, adv (already normalized). Returns (new lr, stats).
    Optional (FB fixed-task formulation): loss_w / loss_mask weight the
    per-stream surrogate and value loss, and query_ids chooses the critic
    streams, separately from the conditioning set (ids, w). Absent, they
    default to (w, mask, ids), so the original path is unchanged."""
    B = batch["obs"].shape[0]
    mb = B // cfg.minibatches
    perm = torch.randperm(cfg.minibatches * mb, generator=gen, device="cpu").to(batch["obs"].device)
    actor_params, critic_params = model.actor_parameters(), model.critic_parameters()
    if gradient_composition is not None:
        if gradient_composition not in ("GLOBAL", "SPLIT", "SPLIT-NORM-MATCHED") or task_allowed_param_ids is not None:
            raise ValueError("FC-G composition requires a valid arm and no FC-F task route")
    stats = {k: 0.0 for k in ("surrogate", "value", "entropy", "kl", "clip_frac")}
    if gradient_composition is not None:
        stats.update({k: 0.0 for k in ("fcg_task_norm", "fcg_non_task_norm", "fcg_global_norm", "fcg_split_norm",
                                      "fcg_selected_norm", "fcg_c_task", "fcg_c_non_task", "fcg_c_global",
                                      "fcg_c_ratio", "fcg_cos_global_normmatched", "fcg_coef_diff_frac",
                                      "fcg_global_suppresses_non_task_frac", "fcg_actor_step_norm",
                                      "fcg_stream_sum_max_error")})
    if task_allowed_param_ids is not None:
        actor_ids = {id(p) for p in actor_params}
        if not task_allowed_param_ids <= actor_ids:
            raise ValueError("task route names a parameter outside the actor")
        allowed = [p for p in actor_params if id(p) in task_allowed_param_ids]
        blocked = len(allowed) < len(actor_params)
        stats.update(actor_grad_norm=0.0, actor_clip_coef=0.0, actor_clip_min=1.0, actor_clip_fraction=0.0)
    n = 0
    for epoch in range(cfg.epochs):
        for i in range(cfg.minibatches):
            idx = perm[i * mb:(i + 1) * mb]
            g = {k: v[idx] for k, v in batch.items()}
            dist = model._dist(g["obs"], g["env"], g["ids"], g["w"])
            logp = (dist.log_prob(g["u"]) - model._log_det_jacobian(g["u"])).sum(-1)
            with torch.no_grad():
                kl = gaussian_kl(g["old_mu"], g["old_sigma"], dist.loc, dist.scale).mean()
                lr = adapt_lr(lr, float(kl), cfg.desired_kl)
                for opt in (actor_opt, critic_opt):
                    for pg in opt.param_groups:
                        pg["lr"] = lr
            lw, lm = g.get("loss_w", g["w"]), g.get("loss_mask", g["mask"])
            surr, ratio = actor_surrogate(logp, g["old_logp"], g["adv"], lw, lm, cfg.clip)
            ent = dist.entropy().sum(-1).mean()
            v = model.query_values(g["obs"], g["env"], g["ids"], g["w"], g.get("query_ids", g["ids"]))
            vl = value_loss(v, g["old_values"], g["returns"], lm, cfg.clip)
            loss = surr + cfg.value_coef * vl - cfg.entropy_coef * ent
            actor_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            if gradient_composition is not None:
                stream_losses = actor_stream_surrogates(logp, g["old_logp"], g["adv"], lw, lm, cfg.clip)
                tg = torch.autograd.grad(stream_losses[0], actor_params, retain_graph=True, allow_unused=True)
                ng = torch.autograd.grad(stream_losses[1:].sum() - cfg.entropy_coef * ent,
                                         actor_params, retain_graph=True, allow_unused=True)
                dense_t = [torch.zeros_like(p) if x is None else x for p, x in zip(actor_params, tg)]
                dense_n = [torch.zeros_like(p) if x is None else x for p, x in zip(actor_params, ng)]
                selected, audit = compose_actor_gradients(dense_t, dense_n, cfg.max_grad_norm, gradient_composition)
                if gradient_audit_sink is not None:
                    gradient_audit_sink.append({"epoch": epoch, "minibatch": i, "arm": gradient_composition,
                                                **audit, "lr": lr, "kl": float(kl)})
                for k, v_ in audit.items():
                    stats[k] += v_
                if gradient_composition == "GLOBAL":
                    loss.backward()  # exact legacy path; decomposed gradients are audit only
                    error = max(float((p.grad.detach() - t - n).abs().max()) for p, t, n in zip(actor_params, dense_t, dense_n))
                    stats["fcg_stream_sum_max_error"] = max(stats["fcg_stream_sum_max_error"], error)
                    if error > 1e-5:
                        raise RuntimeError(f"FC-G stream sum differs from legacy actor gradient: {error}")
                else:
                    (cfg.value_coef * vl).backward()
                    for p, grad in zip(actor_params, selected):
                        p.grad = grad
            elif task_allowed_param_ids is not None and blocked and bool(torch.any(lw[:, 0] * g["adv"][:, 0])):
                stream_losses = actor_stream_surrogates(logp, g["old_logp"], g["adv"], lw, lm, cfg.clip)
                task_grads = torch.autograd.grad(stream_losses[0], allowed, retain_graph=True, allow_unused=True)
                (stream_losses[1:].sum() + cfg.value_coef * vl - cfg.entropy_coef * ent).backward()
                for p, task_grad in zip(allowed, task_grads):
                    if task_grad is not None:
                        if p.grad is None:
                            p.grad = task_grad
                        else:
                            p.grad.add_(task_grad)
            else:
                loss.backward()  # FULL route: byte-for-byte legacy backward path
            if gradient_composition is not None:
                before = [p.detach().clone() for p in actor_params]
            actor_norm = torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm) if gradient_composition in (None, "GLOBAL") else torch.tensor(audit["fcg_selected_norm"])
            torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
            if task_allowed_param_ids is not None:
                norm = float(actor_norm)
                coef = min(1.0, cfg.max_grad_norm / (norm + 1e-6))
                stats["actor_grad_norm"] += norm
                stats["actor_clip_coef"] += coef
                stats["actor_clip_min"] = min(stats["actor_clip_min"], coef)
                stats["actor_clip_fraction"] += float(coef < 1.0)
            actor_opt.step()
            if gradient_composition is not None:
                stats["fcg_actor_step_norm"] += float(torch.stack([torch.sum((p.detach().double() - b.double()).square()) for p, b in zip(actor_params, before)]).sum().sqrt())
            critic_opt.step()
            stats["surrogate"] += float(surr); stats["value"] += float(vl); stats["entropy"] += float(ent)
            stats["kl"] += float(kl); stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
            n += 1
    return lr, {k: v if k in ("actor_clip_min", "fcg_stream_sum_max_error") else v / n for k, v in stats.items()}
