#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 64
    norm_adv: bool = True


def compute_gae_for_trajectory(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE. rewards/values/dones are [T]. dones=True means terminal after this step."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    lastgaelam = 0.0
    next_value = torch.tensor(float(last_value), dtype=torch.float32, device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t].float()
            nv = next_value
        else:
            next_nonterminal = 1.0 - dones[t].float()
            nv = values[t + 1]
        delta = rewards[t] + float(gamma) * nv * next_nonterminal - values[t]
        lastgaelam = delta + float(gamma) * float(gae_lambda) * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def load_trajectory_files(paths: list[str | Path]) -> dict[str, torch.Tensor]:
    """Load worker trajectory .pt files and concatenate them into one PPO batch."""
    chunks: dict[str, list[torch.Tensor]] = {}
    stats: list[dict[str, Any]] = []
    for p in paths:
        obj = torch.load(p, map_location="cpu")
        trajs = obj["trajectories"] if isinstance(obj, dict) and "trajectories" in obj else obj
        if isinstance(obj, dict) and "stats" in obj:
            stats.extend(obj["stats"])
        for traj in trajs:
            rewards = traj["rewards"].float()
            values = traj["values"].float()
            dones = traj["dones"].bool()
            adv, ret = compute_gae_for_trajectory(
                rewards,
                values,
                dones,
                gamma=float(obj.get("gamma", 0.99)) if isinstance(obj, dict) else 0.99,
                gae_lambda=float(obj.get("gae_lambda", 0.95)) if isinstance(obj, dict) else 0.95,
            )
            traj["advantages"] = adv
            traj["returns"] = ret
            for k, v in traj.items():
                if not torch.is_tensor(v):
                    continue
                chunks.setdefault(k, []).append(v.cpu())
    batch = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    batch["_num_steps"] = torch.tensor([batch["actions"].shape[0]], dtype=torch.long)
    if stats:
        # Put a compact stats tensor placeholder; detailed stats should be read from jsonl logs.
        batch["_num_episodes"] = torch.tensor([len(stats)], dtype=torch.long)
    return batch


def ppo_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    cfg: PPOConfig,
    amp: bool = False,
) -> dict[str, float]:
    actions = batch["actions"].long()
    old_logprobs = batch["logprobs"].float()
    advantages = batch["advantages"].float()
    returns = batch["returns"].float()
    old_values = batch["values"].float()

    n = actions.shape[0]
    if n == 0:
        raise RuntimeError("empty PPO batch")
    if cfg.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    idx = torch.arange(n)
    stats = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
    }
    num_updates = 0
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp))

    # Keep large inputs on CPU and move minibatches only.
    for _epoch in range(int(cfg.update_epochs)):
        perm = idx[torch.randperm(n)]
        for start in range(0, n, int(cfg.minibatch_size)):
            mb = perm[start : start + int(cfg.minibatch_size)]
            mb_prefix = batch["prefix_hidden"][mb].to(device, non_blocking=True)
            mb_mask = batch["prefix_pad_mask"][mb].to(device, non_blocking=True)
            mb_state = batch["state"][mb].to(device, non_blocking=True)
            mb_cur = batch["current_vlm_precision_id"][mb].to(device, non_blocking=True)
            mb_actions = actions[mb].to(device, non_blocking=True)
            mb_old_logprobs = old_logprobs[mb].to(device, non_blocking=True)
            mb_advantages = advantages[mb].to(device, non_blocking=True)
            mb_returns = returns[mb].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(amp)):
                _, new_logprob, entropy, new_value, _ = model.get_action_and_value(
                    prefix_hidden=mb_prefix,
                    prefix_pad_mask=mb_mask,
                    state=mb_state,
                    current_vlm_precision_id=mb_cur,
                    action=mb_actions,
                )
                logratio = new_logprob - mb_old_logprobs
                ratio = logratio.exp()
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * F.mse_loss(new_value.float(), mb_returns.float())
                ent = entropy.mean()
                loss = pg_loss + float(cfg.vf_coef) * v_loss - float(cfg.ent_coef) * ent

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
            stats["loss"] += float(loss.detach().item())
            stats["policy_loss"] += float(pg_loss.detach().item())
            stats["value_loss"] += float(v_loss.detach().item())
            stats["entropy"] += float(ent.detach().item())
            stats["approx_kl"] += float(approx_kl)
            stats["clipfrac"] += float(clipfrac)
            num_updates += 1

    for k in list(stats.keys()):
        stats[k] /= max(1, num_updates)
    return stats
