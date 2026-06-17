#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

BITS = (4, 8, 16)
BITS_TO_LABEL = {4: 0, 8: 1, 16: 2}
LABEL_TO_BITS = {0: 4, 1: 8, 2: 16}

# 9-way meta-action: current action precision, next chunk VLM precision.
PRECISION_DECISIONS: tuple[tuple[int, int], ...] = (
    (4, 4), (4, 8), (4, 16),
    (8, 4), (8, 8), (8, 16),
    (16, 4), (16, 8), (16, 16),
)
DECISION_TO_ID = {d: i for i, d in enumerate(PRECISION_DECISIONS)}
ID_TO_DECISION = {i: d for i, d in enumerate(PRECISION_DECISIONS)}


def decision_id_to_bits(decision_id: int) -> tuple[int, int]:
    return ID_TO_DECISION[int(decision_id)]


def bits_to_decision_id(action_bits: int, next_vlm_bits: int) -> int:
    return DECISION_TO_ID[(int(action_bits), int(next_vlm_bits))]


@dataclass(frozen=True)
class PrecisionCostConfig:
    action_cost: tuple[float, float, float] = (1.0, 2.0, 4.0)
    vlm_cost: tuple[float, float, float] = (1.0, 2.0, 4.0)
    action_weight: float = 1.0
    vlm_weight: float = 1.0


def precision_decision_cost_tensor(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    config: PrecisionCostConfig | None = None,
) -> torch.Tensor:
    """Return [9] cost for each precision decision."""
    cfg = config or PrecisionCostConfig()
    action_cost = torch.tensor(cfg.action_cost, device=device, dtype=dtype)
    vlm_cost = torch.tensor(cfg.vlm_cost, device=device, dtype=dtype)
    costs = []
    for action_bits, next_vlm_bits in PRECISION_DECISIONS:
        ai = BITS_TO_LABEL[int(action_bits)]
        vi = BITS_TO_LABEL[int(next_vlm_bits)]
        costs.append(cfg.action_weight * action_cost[ai] + cfg.vlm_weight * vlm_cost[vi])
    return torch.stack(costs, dim=0)


class ActorCriticPrecisionSelector(nn.Module):
    """
    PPO actor-critic selector for success-first precision scheduling.

    Input for chunk k:
      prefix_hidden: [B, T, hidden_dim]
      prefix_pad_mask: [B, T], True = valid token
      state: [B, state_dim]
      current_vlm_precision_id: [B], 0/1/2 for current VLM A4/A8/A16

    Output:
      policy_logits: [B, 9] over precision pair decisions
      value: [B] critic value V(s)
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        state_dim: int,
        selector_dim: int = 256,
        num_queries: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        selector_dim = int(selector_dim)
        num_heads = int(num_heads)
        num_queries = int(num_queries)
        if selector_dim % num_heads != 0:
            raise ValueError(f"selector_dim={selector_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = hidden_dim
        self.state_dim = int(state_dim)
        self.selector_dim = selector_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.num_actions = len(PRECISION_DECISIONS)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, selector_dim),
            nn.GELU(),
            nn.LayerNorm(selector_dim),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, selector_dim),
            nn.GELU(),
            nn.LayerNorm(selector_dim),
            nn.Linear(selector_dim, selector_dim),
            nn.GELU(),
            nn.LayerNorm(selector_dim),
        )
        self.precision_embedding = nn.Embedding(3, selector_dim)
        self.base_queries = nn.Parameter(torch.randn(num_queries, selector_dim) * 0.02)
        self.condition_to_query = nn.Sequential(
            nn.Linear(selector_dim * 2, selector_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(selector_dim * 2, selector_dim * num_queries),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=selector_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_self_attn = nn.MultiheadAttention(
            embed_dim=selector_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm1 = nn.LayerNorm(selector_dim)
        self.query_norm2 = nn.LayerNorm(selector_dim)
        self.query_norm3 = nn.LayerNorm(selector_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(selector_dim, selector_dim * int(mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(selector_dim * int(mlp_ratio), selector_dim),
            nn.Dropout(dropout),
        )

        feature_dim = selector_dim * num_queries + selector_dim + selector_dim
        head_dim = max(256, selector_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(head_dim),
        )
        self.policy_head = nn.Linear(head_dim, self.num_actions)
        self.value_head = nn.Linear(head_dim, 1)

    def encode(
        self,
        *,
        prefix_hidden: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        state: torch.Tensor,
        current_vlm_precision_id: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = prefix_hidden.shape
        if D != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {D}")
        x = self.input_norm(prefix_hidden.float())
        x = self.input_proj(x)
        mask = prefix_pad_mask.bool()

        state_feat = self.state_encoder(state.float())
        precision_feat = self.precision_embedding(current_vlm_precision_id.long())
        cond = torch.cat([state_feat, precision_feat], dim=-1)

        base_q = self.base_queries.unsqueeze(0).expand(B, -1, -1)
        cond_q = self.condition_to_query(cond).view(B, self.num_queries, self.selector_dim)
        q = base_q + cond_q

        key_padding_mask = ~mask
        attended, _ = self.cross_attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = self.query_norm1(q + attended)
        self_attended, _ = self.query_self_attn(query=q, key=q, value=q, need_weights=False)
        q = self.query_norm2(q + self_attended)
        q = self.query_norm3(q + self.query_ffn(q))

        feature = torch.cat([q.flatten(start_dim=1), state_feat, precision_feat], dim=-1)
        return self.shared_mlp(feature)

    def forward(
        self,
        *,
        prefix_hidden: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        state: torch.Tensor,
        current_vlm_precision_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.encode(
            prefix_hidden=prefix_hidden,
            prefix_pad_mask=prefix_pad_mask,
            state=state,
            current_vlm_precision_id=current_vlm_precision_id,
        )
        logits = self.policy_head(shared)
        value = self.value_head(shared).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        *,
        prefix_hidden: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        state: torch.Tensor,
        current_vlm_precision_id: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(
            prefix_hidden=prefix_hidden,
            prefix_pad_mask=prefix_pad_mask,
            state=state,
            current_vlm_precision_id=current_vlm_precision_id,
        )
        if temperature != 1.0:
            logits_for_dist = logits / max(float(temperature), 1e-6)
        else:
            logits_for_dist = logits
        dist = torch.distributions.Categorical(logits=logits_for_dist)
        if action is None:
            if deterministic:
                action = logits_for_dist.argmax(dim=-1)
            else:
                action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value, logits

    @torch.no_grad()
    def infer_bits(
        self,
        *,
        prefix_hidden: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        state: torch.Tensor,
        current_vlm_a_bits: int | torch.Tensor,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[int, int, int]:
        if prefix_hidden.ndim == 2:
            prefix_hidden = prefix_hidden.unsqueeze(0)
        if prefix_pad_mask.ndim == 1:
            prefix_pad_mask = prefix_pad_mask.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if isinstance(current_vlm_a_bits, torch.Tensor):
            cur = current_vlm_a_bits.to(device=prefix_hidden.device, dtype=torch.long)
            if cur.ndim == 0:
                cur = cur[None]
        else:
            cur = torch.tensor([BITS_TO_LABEL[int(current_vlm_a_bits)]], device=prefix_hidden.device)
        action_id, _, _, _, _ = self.get_action_and_value(
            prefix_hidden=prefix_hidden,
            prefix_pad_mask=prefix_pad_mask,
            state=state,
            current_vlm_precision_id=cur,
            deterministic=deterministic,
            temperature=temperature,
        )
        did = int(action_id[0].item())
        action_bits, next_vlm_bits = decision_id_to_bits(did)
        return action_bits, next_vlm_bits, did

    @torch.no_grad()
    def init_policy_bias_to_decision(self, decision_id: int = 8, boost: float = 6.0) -> None:
        """Safe start: strongly prefer (A16, A16) by default."""
        self.policy_head.weight.zero_()
        self.policy_head.bias.zero_()
        self.policy_head.bias[int(decision_id)] = float(boost)
        self.value_head.bias.zero_()


def build_model_from_meta(
    *,
    hidden_dim: int,
    state_dim: int,
    selector_dim: int = 256,
    num_queries: int = 4,
    num_heads: int = 4,
    dropout: float = 0.2,
    mlp_ratio: int = 4,
) -> ActorCriticPrecisionSelector:
    return ActorCriticPrecisionSelector(
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        selector_dim=selector_dim,
        num_queries=num_queries,
        num_heads=num_heads,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
    )
