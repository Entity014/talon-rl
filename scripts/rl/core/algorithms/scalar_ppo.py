"""Scalar PPO algorithm family: deterministic baseline and reference-style variants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class B0PPOConfig:
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    std_initial: float = 0.82
    std_final: float = 0.10
    std_hold_updates: int = 100
    std_decay_updates: int = 300
    b1_p1_enabled: bool = False
    target_kl: float = 0.01
    kl_stop_multiplier: float = 1.5
    actor_epochs: int = 1
    b1_p2_enabled: bool = False
    desired_kl: float = 0.01
    kl_low: float = 0.005
    kl_high: float = 0.02
    lr_factor: float = 1.5
    actor_lr_min: float = 1e-5
    actor_lr_max: float = 1e-2
    b1_s1_enabled: bool = False
    lambda_spatial: float = 0.10
    lambda_temporal: float = 0.05
    perturb_std: float = 0.02
    perturb_clip: float = 0.05
    b1_r1_enabled: bool = False
    lambda_reconstruction: float = 0.10

    def scheduled_std(self, update_idx: int) -> float:
        if update_idx < 0:
            raise ValueError("update_idx must be nonnegative")
        progress = min(max((update_idx - self.std_hold_updates) / self.std_decay_updates, 0.0), 1.0)
        return self.std_initial + (self.std_final - self.std_initial) * progress


def analytic_gaussian_kl(old_mean: torch.Tensor, new_mean: torch.Tensor, std: float) -> torch.Tensor:
    """Exact pre-tanh diagonal-Gaussian KL, reduced over action dimensions."""
    if std <= 0 or old_mean.shape != new_mean.shape:
        raise ValueError("positive std and matching mean shapes required")
    return ((old_mean - new_mean).pow(2) / (2.0 * std * std)).sum(dim=-1)


def scalar_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
               gamma: float, gae_lambda: float) -> np.ndarray:
    """Scalar GAE: rewards `(T,N)`, values `(T+1,N)`, dones `(T,N)`."""
    if rewards.ndim != 2 or values.shape != (rewards.shape[0] + 1, rewards.shape[1]) or dones.shape != rewards.shape:
        raise ValueError("expected rewards/dones (T,N) and values (T+1,N)")
    advantage = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        mask = 1.0 - dones[step].astype(np.float32)
        delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
        carry = delta + gamma * gae_lambda * mask * carry
        advantage[step] = carry
    return advantage


class B0PPOTrainer:
    """Algorithm-only scalar PPO core; environment collection stays external."""
    def __init__(self, model, cfg: B0PPOConfig | None = None, *, lr: float = 3e-4):
        self.model, self.cfg, self.update_idx = model, cfg or B0PPOConfig(), 0
        params = [p for name, p in model.named_parameters() if name != "log_std"]
        self.optim = torch.optim.Adam(params, lr=lr)
        self.actor_optim = self.critic_optim = None
        if self.cfg.b1_p2_enabled:
            actor_params = [p for n,p in model.named_parameters() if n != "log_std" and not n.startswith("critic_")]
            critic_params = [p for n,p in model.named_parameters() if n.startswith("critic_")]
            self.actor_optim = torch.optim.Adam(actor_params, lr=lr)
            self.critic_optim = torch.optim.Adam(critic_params, lr=lr)
        self._std_for_update: float | None = None

    def begin_update(self) -> float:
        """Set fixed policy std before collecting this update's rollout."""
        std = self.cfg.scheduled_std(self.update_idx)
        self.model.set_scheduled_fixed_std(std)
        self._std_for_update = std
        return std

    def ppo_loss(self, actor_obs, critic_obs, actions, old_logp, advantages, returns):
        if self._std_for_update is None:
            raise RuntimeError("begin_update() must precede rollout/update")
        new_logp = self.model.logp(actor_obs, actions)
        ratio = torch.exp(new_logp - old_logp)
        adv = advantages
        unclipped, clipped = ratio * adv, ratio.clamp(1-self.cfg.clip_eps, 1+self.cfg.clip_eps) * adv
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        values = self.model.value(critic_obs).squeeze(-1)
        value_loss = F.mse_loss(values, returns)
        entropy = self.model.entropy(actor_obs).mean()
        clip_fraction = ((ratio - 1).abs() > self.cfg.clip_eps).float().mean()
        return policy_loss, value_loss, entropy, ratio, clip_fraction

    def finish_update(self) -> None:
        self.update_idx += 1
        self._std_for_update = None

    def optimize_batch(self, actor_obs, actions, old_logp, advantages, returns, *, rollout_obs=None, rollout_dones=None, reconstruction_targets=None) -> dict:
        """Run full-batch PPO epochs, with optional B1-P1 actor-only KL stopping."""
        if self._std_for_update is None:
            raise RuntimeError("begin_update() must precede optimize_batch")
        if self.cfg.b1_p2_enabled:
            return self._optimize_p2(actor_obs, actions, old_logp, advantages, returns, rollout_obs, rollout_dones, reconstruction_targets)
        epochs = self.cfg.actor_epochs if self.cfg.b1_p1_enabled else 1
        with torch.no_grad():
            old_mean = self.model.raw_mean(actor_obs).detach()
        kl_trace=[]; actor_stopped=False; last=None
        for epoch in range(epochs):
            last = self.ppo_loss(actor_obs, actor_obs, actions, old_logp, advantages, returns)
            self.optim.zero_grad()
            loss = (0.0 if actor_stopped else last[0]) + 0.5 * last[1] - 0.001 * last[2]
            loss.backward(); self.optim.step()
            with torch.no_grad():
                kl = analytic_gaussian_kl(old_mean, self.model.raw_mean(actor_obs), self._std_for_update).mean()
            kl_trace.append(float(kl))
            if self.cfg.b1_p1_enabled and kl >= self.cfg.target_kl * self.cfg.kl_stop_multiplier:
                actor_stopped = True
                break
        return {"policy_loss": float(last[0]), "value_loss": float(last[1]), "entropy": float(last[2]),
                "approx_kl": float((last[3].detach()-1.0).mean().abs()), "clip_fraction": float(last[4]),
                "analytic_kl": kl_trace[-1], "kl_trace": kl_trace, "actor_epochs_completed": len(kl_trace),
                "actor_stopped": actor_stopped, "critic_completed": True}

    def _optimize_p2(self, actor_obs, actions, old_logp, advantages, returns, rollout_obs=None, rollout_dones=None, reconstruction_targets=None) -> dict:
        with torch.no_grad(): old_mean = self.model.raw_mean(actor_obs).detach()
        traces=[]; lr_events=[]; last=None
        for epoch in range(self.cfg.actor_epochs):
            last=self.ppo_loss(actor_obs, actor_obs, actions, old_logp, advantages, returns)
            spatial=temporal=recon_loss=actor_loss=last[0]
            if self.cfg.b1_r1_enabled:
                if rollout_obs is None or reconstruction_targets is None:
                    raise ValueError("R1 requires rollout observations and privileged reconstruction targets")
                rshape=rollout_obs.shape
                pred=self.model.reconstruct(rollout_obs.reshape(-1,rshape[-1]))
                target=reconstruction_targets.reshape_as(pred)
                recon_loss=F.mse_loss(pred,target)
                actor_loss=actor_loss+self.cfg.lambda_reconstruction*recon_loss
            if self.cfg.b1_s1_enabled:
                if rollout_obs is None or rollout_dones is None: raise ValueError("S1 requires time-major rollout_obs and rollout_dones")
                shape=rollout_obs.shape; ro=rollout_obs.reshape(-1,shape[-1]); mu=self.model.raw_mean(ro).reshape(shape[0],shape[1],-1)
                mask=torch.ones(shape[-1],device=ro.device); mask[42:45]=0
                noise=torch.randn_like(rollout_obs)*self.cfg.perturb_std; noise.clamp_(-self.cfg.perturb_clip,self.cfg.perturb_clip); noise*=mask
                spatial=((self.model.raw_mean((rollout_obs+noise).reshape(-1,shape[-1])).reshape_as(mu)-mu)**2).mean()
                valid=(~rollout_dones[:-1]).float(); temporal_raw=((mu[1:]-mu[:-1])**2).mean(dim=-1); temporal=(temporal_raw*valid).sum()/(valid.sum()*mu.shape[-1]+1e-8)
                actor_loss=actor_loss+self.cfg.lambda_spatial*spatial+self.cfg.lambda_temporal*temporal
            spatial_grad=temporal_grad=recon_grad=0.0
            if self.cfg.b1_s1_enabled:
                sg=torch.autograd.grad(self.cfg.lambda_spatial*spatial,self.actor_optim.param_groups[0]['params'],retain_graph=True,allow_unused=True); spatial_grad=float(torch.sqrt(sum((g.detach()**2).sum() for g in sg if g is not None)))
                tg=torch.autograd.grad(self.cfg.lambda_temporal*temporal,self.actor_optim.param_groups[0]['params'],retain_graph=True,allow_unused=True); temporal_grad=float(torch.sqrt(sum((g.detach()**2).sum() for g in tg if g is not None)))
            if self.cfg.b1_r1_enabled:
                rg=torch.autograd.grad(self.cfg.lambda_reconstruction*recon_loss,self.actor_optim.param_groups[0]['params'],retain_graph=True,allow_unused=True); recon_grad=float(torch.sqrt(sum((g.detach()**2).sum() for g in rg if g is not None)))
            self.actor_optim.zero_grad(); (actor_loss - 0.001*last[2]).backward(); actor_grad=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in self.actor_optim.param_groups[0]['params'] if p.grad is not None))); self.actor_optim.step()
            self.critic_optim.zero_grad(); (0.5*last[1]).backward(); self.critic_optim.step()
            with torch.no_grad(): kl=float(analytic_gaussian_kl(old_mean,self.model.raw_mean(actor_obs),self._std_for_update).mean())
            traces.append(kl); before=self.actor_optim.param_groups[0]['lr']; after=before; event='hold'
            if kl > self.cfg.kl_high: after=max(self.cfg.actor_lr_min,before/self.cfg.lr_factor); event='down'
            elif kl < self.cfg.kl_low: after=min(self.cfg.actor_lr_max,before*self.cfg.lr_factor); event='up'
            for group in self.actor_optim.param_groups: group['lr']=after
            lr_events.append({'epoch':epoch+1,'lr_before':before,'lr_after':after,'event':event})
        return {"policy_loss":float(last[0]),"value_loss":float(last[1]),"entropy":float(last[2]),
                "approx_kl":float((last[3].detach()-1).mean().abs()),"clip_fraction":float(last[4]),
                "analytic_kl":traces[-1],"kl_trace":traces,"lr_events":lr_events,
                "actor_lr":self.actor_optim.param_groups[0]['lr'],"actor_epochs_completed":len(traces),
                "spatial_loss":float(spatial.detach()) if self.cfg.b1_s1_enabled else 0.0,"temporal_loss":float(temporal.detach()) if self.cfg.b1_s1_enabled else 0.0,"reconstruction_loss":float(recon_loss.detach()) if self.cfg.b1_r1_enabled else 0.0,"actor_grad_norm":actor_grad,"spatial_grad_norm":spatial_grad,"temporal_grad_norm":temporal_grad,"reconstruction_grad_norm":recon_grad,
                "actor_stopped":False,"critic_completed":True}

    def save(self, path: str | Path) -> None:
        payload={"schema":"b0_ppo_v1", "model":self.model.state_dict(), "optim":self.optim.state_dict(),
                    "update_idx":self.update_idx, "exploration_mode":self.model.exploration_mode,
                    "scheduled_std":self.cfg.scheduled_std(self.update_idx), "b1_p1": {"enabled": self.cfg.b1_p1_enabled,
                    "target_kl": self.cfg.target_kl, "kl_stop_multiplier": self.cfg.kl_stop_multiplier,
                    "actor_epochs": self.cfg.actor_epochs}, "b1_p2": {"enabled": self.cfg.b1_p2_enabled}}
        if self.cfg.b1_p2_enabled: payload["actor_optim"],payload["critic_optim"],payload["actor_lr"]=self.actor_optim.state_dict(),self.critic_optim.state_dict(),self.actor_optim.param_groups[0]['lr']
        payload["b1_s1"]={"enabled":self.cfg.b1_s1_enabled,"lambda_spatial":self.cfg.lambda_spatial,"lambda_temporal":self.cfg.lambda_temporal}
        payload["b1_r1"]={"enabled":self.cfg.b1_r1_enabled,"lambda_reconstruction":self.cfg.lambda_reconstruction}
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        state=torch.load(path, map_location=next(self.model.parameters()).device)
        if state.get("schema") != "b0_ppo_v1": raise ValueError("not a B0 PPO checkpoint")
        self.model.load_state_dict(state["model"]); self.optim.load_state_dict(state["optim"]); self.update_idx=int(state["update_idx"])
        saved_b1 = state.get("b1_p1", {"enabled": False})
        if bool(saved_b1.get("enabled", False)) != self.cfg.b1_p1_enabled:
            raise ValueError("checkpoint B1-P1 setting disagrees with trainer config")
        if bool(state.get("b1_p2", {}).get("enabled", False)) != self.cfg.b1_p2_enabled: raise ValueError("checkpoint B1-P2 setting disagrees with trainer config")
        if bool(state.get("b1_s1", {}).get("enabled", False)) != self.cfg.b1_s1_enabled: raise ValueError("checkpoint B1-S1 setting disagrees with trainer config")
        if bool(state.get("b1_r1", {}).get("enabled", False)) != self.cfg.b1_r1_enabled: raise ValueError("checkpoint B1-R1 setting disagrees with trainer config")
        if self.cfg.b1_p2_enabled:
            self.actor_optim.load_state_dict(state["actor_optim"]); self.critic_optim.load_state_dict(state["critic_optim"])
            for group in self.actor_optim.param_groups: group['lr']=float(state['actor_lr'])
        expected=self.cfg.scheduled_std(self.update_idx)
        if not np.isclose(state["scheduled_std"], expected): raise ValueError("checkpoint scheduled std disagrees with update index")
        self.model.set_scheduled_fixed_std(expected); self._std_for_update=expected


class ScalarRolloutBuffer:
    """Minimal scalar PPO rollout buffer; no objective axis exists."""
    def __init__(self, horizon: int, lanes: int):
        self.horizon, self.lanes = horizon, lanes
        self.clear()

    def clear(self):
        self.obs=[]; self.actions=[]; self.logp_old=[]; self.rewards=[]; self.dones=[]; self.values=[]; self.final_value=None

    def append(self, obs, actions, logp_old, rewards, dones, values):
        n=self.lanes
        if len(self.obs) >= self.horizon: raise RuntimeError("rollout buffer full")
        if any(np.asarray(x).shape[0] != n for x in (obs, actions, logp_old, rewards, dones, values)):
            raise ValueError("lane dimension mismatch")
        if np.asarray(rewards).ndim != 1 or np.asarray(dones).ndim != 1 or np.asarray(values).ndim != 1:
            raise ValueError("scalar reward/done/value must be (N,)")
        self.obs.append(np.asarray(obs)); self.actions.append(np.asarray(actions)); self.logp_old.append(np.asarray(logp_old)); self.rewards.append(np.asarray(rewards)); self.dones.append(np.asarray(dones)); self.values.append(np.asarray(values))

    def finish(self, final_value):
        value=np.asarray(final_value)
        if len(self.obs) != self.horizon or value.shape != (self.lanes,): raise ValueError("incomplete rollout or invalid final_value")
        self.final_value=value

    def arrays(self):
        if self.final_value is None: raise RuntimeError("finish() required")
        return {"obs":np.stack(self.obs),"actions":np.stack(self.actions),"logp_old":np.stack(self.logp_old),"rewards":np.stack(self.rewards),"dones":np.stack(self.dones),"values":np.stack(self.values),"final_value":self.final_value}

    def flatten(self):
        a=self.arrays(); t,n=self.horizon,self.lanes
        return {k:(v.reshape((t*n,)+v.shape[2:]) if k != "final_value" else v) for k,v in a.items()}


# Reference-style scalar PPO variant.
@dataclass(frozen=True)
class L0ReferenceConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 1e-3
    desired_kl: float = 0.01
    lr_factor: float = 1.5
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    max_grad_norm: float = 1.0


class L0ReferencePPO:
    def __init__(self, model, cfg: L0ReferenceConfig | None = None):
        self.model = model
        self.cfg = cfg or L0ReferenceConfig()
        self.optim = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        self.update_idx = 0

    def begin_update(self) -> float:
        self.model.set_learned_std()
        return float(self.model.log_std.exp().mean().detach().cpu())

    def update(self, obs, actions, old_logp, advantages, returns, old_mean, *, seed=0) -> dict:
        n = obs.shape[0]
        if n % self.cfg.minibatches:
            raise ValueError("rollout batch must divide evenly into minibatches")
        gen = torch.Generator(device=obs.device).manual_seed(int(seed) + self.update_idx)
        order = torch.randperm(n, generator=gen, device=obs.device)
        mb = n // self.cfg.minibatches
        policy_losses, value_losses, entropies, clips = [], [], [], []
        kl_trace = []
        lr_events = []
        for epoch in range(self.cfg.epochs):
            for start in range(0, n, mb):
                idx = order[start:start + mb]
                new_logp = self.model.logp(obs[idx], actions[idx])
                ratio = torch.exp(new_logp - old_logp[idx])
                adv = advantages[idx]
                p1 = ratio * adv
                p2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
                policy = -torch.minimum(p1, p2).mean()
                value = F.mse_loss(self.model.value(obs[idx]).squeeze(-1), returns[idx])
                entropy = self.model.entropy(obs[idx]).mean()
                loss = policy + self.cfg.value_loss_coef * value - self.cfg.entropy_coef * entropy
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optim.step()
                policy_losses.append(float(policy.detach()))
                value_losses.append(float(value.detach()))
                entropies.append(float(entropy.detach()))
                clips.append(float(((ratio - 1).abs() > self.cfg.clip_eps).float().mean().detach()))
            with torch.no_grad():
                new_mean = self.model.raw_mean(obs)
                # Same-std pre-tanh Gaussian KL; std is learned but held
                # constant while measuring this update's drift.
                std = self.model.log_std.exp()
                kl = (((old_mean - new_mean).pow(2) / (2 * std.pow(2))).sum(-1)).mean()
            kl_value = float(kl.detach())
            kl_trace.append(kl_value)
            before = float(self.optim.param_groups[0]["lr"])
            after, event = before, "hold"
            if kl_value > self.cfg.desired_kl * 2:
                after, event = max(self.cfg.lr_min, before / self.cfg.lr_factor), "down"
            elif kl_value < self.cfg.desired_kl / 2 and kl_value > 0:
                after, event = min(self.cfg.lr_max, before * self.cfg.lr_factor), "up"
            for group in self.optim.param_groups:
                group["lr"] = after
            lr_events.append({"epoch": epoch + 1, "lr_before": before, "lr_after": after, "event": event})
        self.update_idx += 1
        return {
            "update": self.update_idx,
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clips)),
            "analytic_kl": kl_trace[-1],
            "kl_trace": kl_trace,
            "lr_events": lr_events,
            "learning_rate": float(self.optim.param_groups[0]["lr"]),
            "learned_std": float(self.model.log_std.exp().mean().detach().cpu()),
            "epochs_completed": self.cfg.epochs,
            "minibatches": self.cfg.minibatches,
        }

    def save(self, path):
        torch.save({"schema": "l0_reference_ppo_v1", "model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(), "update_idx": self.update_idx,
                    "learning_rate": self.optim.param_groups[0]["lr"]}, path)

    def load(self, path):
        state = torch.load(path, map_location=next(self.model.parameters()).device)
        if state.get("schema") != "l0_reference_ppo_v1":
            raise ValueError("not an L0 reference PPO checkpoint")
        self.model.load_state_dict(state["model"])
        self.optim.load_state_dict(state["optimizer"])
        self.update_idx = int(state["update_idx"])
        for group in self.optim.param_groups:
            group["lr"] = float(state["learning_rate"])
