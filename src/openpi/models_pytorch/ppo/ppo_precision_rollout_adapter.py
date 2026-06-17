#!/usr/bin/env python3
"""
Rollout adapter for Success-First Budgeted PPO Precision Selector.

This file is the only place that must be connected to your existing LIBERO/OpenPI runner.
The PPO trainer/model/algorithm are complete and environment-agnostic; this adapter should call
whatever code you currently use in experiments/mode*_*.py to:
  1) reset a LIBERO episode / restore a task+episode,
  2) run frozen QuantVLA/OpenPI one chunk at a time,
  3) set current action precision and next VLM precision according to the selector,
  4) return trajectory tensors for PPO.

Expected trajectory keys per episode:
  prefix_hidden: [T, L, D] float16/float32
  prefix_pad_mask: [T, L] bool
  state: [T, state_dim] float32
  current_vlm_precision_id: [T] long
  actions: [T] long  # 0..8 precision decision id
  logprobs: [T] float32
  values: [T] float32
  rewards: [T] float32
  dones: [T] bool

Stats dict should include:
  success, total_steps, total_chunks, total_precision_cost, episode_return,
  precision_hist, task_id, episode_idx.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import random

import torch

from ppo_precision_model import (
    ActorCriticPrecisionSelector,
    BITS_TO_LABEL,
    decision_id_to_bits,
    precision_decision_cost_tensor,
    PrecisionCostConfig,
)


def _stack_trajectory(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not rows:
        raise RuntimeError("empty episode trajectory")
    out: dict[str, torch.Tensor] = {}
    keys = rows[0].keys()
    for k in keys:
        vals = [r[k] for r in rows]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
    return out


def success_first_episode_reward(
    *,
    success: bool,
    total_precision_cost: float,
    total_steps: int,
    max_precision_cost: float,
    max_steps: int,
    success_reward: float = 1000.0,
    failure_penalty: float = 1000.0,
    lambda_cost: float = 1.0,
    lambda_step: float = 0.1,
) -> float:
    """Success dominates. Failed episodes receive no cost credit."""
    if not bool(success):
        return -float(failure_penalty)
    norm_cost = float(total_precision_cost) / max(float(max_precision_cost), 1.0)
    norm_steps = float(total_steps) / max(float(max_steps), 1.0)
    return float(success_reward) - float(lambda_cost) * norm_cost - float(lambda_step) * norm_steps


class RolloutHooks:
    """
    Replace this class with calls to your existing runner.

    The current methods are deliberately explicit so the integration point is clear.
    """

    def __init__(self, *, args: Any, device: torch.device):
        self.args = args
        self.device = device

    def sample_task_episode(self, worker_id: int, local_episode_id: int) -> tuple[int, int]:
        """Return (task_id, episode_idx). Replace with your task scheduler."""
        # Default round-robin placeholder.
        task_start = int(getattr(self.args, "task_start", 0))
        task_end = int(getattr(self.args, "task_end", 9))
        task_id = task_start + ((worker_id + local_episode_id) % (task_end - task_start + 1))
        episode_idx = int(getattr(self.args, "episode_start", 0)) + local_episode_id
        return task_id, episode_idx

    def reset_episode(self, task_id: int, episode_idx: int) -> Any:
        """Create/reset your LIBERO/OpenPI env. Must return an opaque env handle."""
        raise NotImplementedError(
            "Connect reset_episode() to your existing LIBERO/OpenPI client. "
            "Copy the reset/setup logic from your current mode client here."
        )

    def get_selector_observation(self, env: Any, current_vlm_bits: int) -> dict[str, torch.Tensor]:
        """
        Return selector inputs before deciding precision:
          prefix_hidden [L,D], prefix_pad_mask [L], state [state_dim].
        This usually means: run/obtain the current VLM prefix hidden under current_vlm_bits.
        """
        raise NotImplementedError("Connect get_selector_observation() to your policy forward hooks.")

    def step_with_precision(
        self,
        env: Any,
        *,
        action_bits: int,
        next_vlm_bits: int,
    ) -> dict[str, Any]:
        """
        Execute one action chunk with current action_bits. Store next_vlm_bits in env state.
        Return dict with at least:
          done: bool
          success: bool
          total_steps: int
          optional progress_delta: float
        """
        raise NotImplementedError("Connect step_with_precision() to your action chunk execution code.")

    def close(self, env: Any) -> None:
        pass


def run_one_episode(
    *,
    model: ActorCriticPrecisionSelector,
    hooks: RolloutHooks,
    task_id: int,
    episode_idx: int,
    device: torch.device,
    args: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model.eval()
    env = hooks.reset_episode(task_id, episode_idx)
    rows: list[dict[str, torch.Tensor]] = []
    current_vlm_bits = int(getattr(args, "initial_vlm_bits", 16))
    total_precision_cost = 0.0
    total_steps = 0
    total_chunks = 0
    success = False
    decision_costs = precision_decision_cost_tensor(
        device=device,
        config=PrecisionCostConfig(
            action_cost=tuple(float(x) for x in getattr(args, "action_cost_values", [1.0, 2.0, 4.0])),
            vlm_cost=tuple(float(x) for x in getattr(args, "vlm_cost_values", [1.0, 2.0, 4.0])),
            action_weight=float(getattr(args, "action_cost_weight", 1.0)),
            vlm_weight=float(getattr(args, "vlm_cost_weight", 1.0)),
        ),
    )
    precision_hist = [0 for _ in range(9)]
    max_chunks = int(getattr(args, "max_chunks", 200))

    try:
        for _chunk in range(max_chunks):
            obs = hooks.get_selector_observation(env, current_vlm_bits=current_vlm_bits)
            prefix_hidden = obs["prefix_hidden"].to(device=device)
            prefix_pad_mask = obs["prefix_pad_mask"].to(device=device)
            state = obs["state"].to(device=device)
            if prefix_hidden.ndim == 2:
                prefix_hidden_b = prefix_hidden.unsqueeze(0)
            else:
                prefix_hidden_b = prefix_hidden
            if prefix_pad_mask.ndim == 1:
                prefix_pad_mask_b = prefix_pad_mask.unsqueeze(0)
            else:
                prefix_pad_mask_b = prefix_pad_mask
            if state.ndim == 1:
                state_b = state.unsqueeze(0)
            else:
                state_b = state
            cur_id = torch.tensor([BITS_TO_LABEL[int(current_vlm_bits)]], device=device, dtype=torch.long)
            with torch.no_grad():
                action_id, logprob, _entropy, value, _logits = model.get_action_and_value(
                    prefix_hidden=prefix_hidden_b,
                    prefix_pad_mask=prefix_pad_mask_b,
                    state=state_b,
                    current_vlm_precision_id=cur_id,
                    deterministic=bool(getattr(args, "deterministic_rollout", False)),
                    temperature=float(getattr(args, "rollout_temperature", 1.0)),
                )
            did = int(action_id.item())
            action_bits, next_vlm_bits = decision_id_to_bits(did)
            precision_hist[did] += 1
            cost = float(decision_costs[did].item())
            total_precision_cost += cost

            step_info = hooks.step_with_precision(env, action_bits=action_bits, next_vlm_bits=next_vlm_bits)
            done = bool(step_info.get("done", False))
            success = bool(step_info.get("success", False))
            total_steps = int(step_info.get("total_steps", total_steps))
            total_chunks += 1

            # Dense progress is optional. Terminal success-first return is filled below.
            reward = float(step_info.get("progress_delta", 0.0)) * float(getattr(args, "progress_weight", 0.0))
            rows.append(
                {
                    "prefix_hidden": prefix_hidden.detach().cpu().to(torch.float16),
                    "prefix_pad_mask": prefix_pad_mask.detach().cpu().bool(),
                    "state": state.detach().cpu().float(),
                    "current_vlm_precision_id": torch.tensor(BITS_TO_LABEL[int(current_vlm_bits)], dtype=torch.long),
                    "actions": torch.tensor(did, dtype=torch.long),
                    "logprobs": logprob.detach().cpu().float().squeeze(0),
                    "values": value.detach().cpu().float().squeeze(0),
                    "rewards": torch.tensor(reward, dtype=torch.float32),
                    "dones": torch.tensor(done, dtype=torch.bool),
                }
            )
            current_vlm_bits = int(next_vlm_bits)
            if done:
                break
    finally:
        hooks.close(env)

    if total_steps <= 0:
        total_steps = total_chunks
    max_precision_cost_arg = float(getattr(args, "max_precision_cost", 0.0))
    max_precision_cost = max_precision_cost_arg if max_precision_cost_arg > 0 else 8.0 * max(1, total_chunks)
    episode_return = success_first_episode_reward(
        success=success,
        total_precision_cost=total_precision_cost,
        total_steps=total_steps,
        max_precision_cost=max_precision_cost,
        max_steps=int(getattr(args, "max_steps", 600)),
        success_reward=float(getattr(args, "success_reward", 1000.0)),
        failure_penalty=float(getattr(args, "failure_penalty", 1000.0)),
        lambda_cost=float(getattr(args, "lambda_cost", 1.0)),
        lambda_step=float(getattr(args, "lambda_step", 0.1)),
    )
    # Success-first terminal reward: failed trajectories get no cost credit.
    if rows:
        rows[-1]["rewards"] = rows[-1]["rewards"] + torch.tensor(float(episode_return), dtype=torch.float32)
        rows[-1]["dones"] = torch.tensor(True, dtype=torch.bool)
    traj = _stack_trajectory(rows)
    stats = {
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "success": bool(success),
        "total_steps": int(total_steps),
        "total_chunks": int(total_chunks),
        "total_precision_cost": float(total_precision_cost),
        "episode_return": float(episode_return),
        "precision_hist": precision_hist,
    }
    return traj, stats


def collect_rollouts(
    *,
    model: ActorCriticPrecisionSelector,
    args: Any,
    device: torch.device,
    worker_id: int,
    num_episodes: int,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    hooks = RolloutHooks(args=args, device=device)
    trajectories = []
    stats = []
    for ep in range(int(num_episodes)):
        task_id, episode_idx = hooks.sample_task_episode(worker_id, ep)
        traj, st = run_one_episode(
            model=model,
            hooks=hooks,
            task_id=task_id,
            episode_idx=episode_idx,
            device=device,
            args=args,
        )
        trajectories.append(traj)
        stats.append(st)
    return trajectories, stats
