#!/usr/bin/env python3
from __future__ import annotations

"""
9-pair outcome-surface model for the QuantVLA precision selector.

Semantics
---------
One training row corresponds to one current-VLM feature branch:

    input  = selector_context_tokens(current_vlm_bits)
             + selector_context_mask
             + server_state
             + current_vlm_label

The model predicts an outcome surface over the selector's *outputs*:

    axis 0 = candidate action bits:   A4, A8, A16
    axis 1 = candidate next VLM bits: A4, A8, A16

So each head returns [B, 3, 3].  current_vlm_bits is an input condition,
not an output candidate axis.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class OutcomeSurfaceConfig:
    context_dim: int = 2048
    state_dim: int = 32
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.1
    mlp_ratio: int = 4


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OutcomeSurfaceModel(nn.Module):
    """Attention reader + state/current-VLM encoder + 9-pair outcome heads.

    Inputs:
      tokens            [B, T, context_dim]
      token_mask        [B, T] bool, True for valid tokens
      state             [B, state_dim]
      current_vlm_label [B], 0=A4, 1=A8, 2=A16, other=unknown

    Outputs:
      utility           [B, 3, 3]
      success_logit     [B, 3, 3]
      cost              [B, 3, 3]
      steps             [B, 3, 3]
      base_value        [B]
      attn_weights      [B, T]
    """

    def __init__(self, config: OutcomeSurfaceConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = OutcomeSurfaceConfig(**kwargs)
        self.config = config
        d = int(config.d_model)
        h = int(config.mlp_ratio) * d
        drop = float(config.dropout)

        self.context_proj = nn.Sequential(
            nn.Linear(int(config.context_dim), d),
            nn.LayerNorm(d),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(int(config.state_dim), d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        # 0=A4, 1=A8, 2=A16, 3=unknown/invalid.
        self.current_vlm_emb = nn.Embedding(4, d)

        self.query_mlp = MLP(2 * d, h, d, dropout=drop)
        self.learned_query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.learned_query, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=int(config.n_heads),
            dropout=drop,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(d)

        self.base_fusion = nn.Sequential(
            nn.Linear(3 * d, h),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(h, d),
            nn.LayerNorm(d),
        )

        # Output-pair condition: candidate action bits × candidate next VLM bits.
        self.action_emb = nn.Embedding(3, d)
        self.next_vlm_emb = nn.Embedding(3, d)
        pair_action = torch.arange(3, dtype=torch.long).repeat_interleave(3)
        pair_next = torch.arange(3, dtype=torch.long).repeat(3)
        self.register_buffer("pair_action_label", pair_action, persistent=False)
        self.register_buffer("pair_next_vlm_label", pair_next, persistent=False)

        self.pair_fusion = nn.Sequential(
            nn.Linear(3 * d, h),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(h, d),
            nn.LayerNorm(d),
        )
        self.utility_head = nn.Linear(d, 1)
        self.success_head = nn.Linear(d, 1)
        self.cost_head = nn.Linear(d, 1)
        self.steps_head = nn.Linear(d, 1)
        self.base_value_head = nn.Linear(d, 1)

    @staticmethod
    def _safe_label(label: torch.Tensor) -> torch.Tensor:
        label = label.long()
        return torch.where((label >= 0) & (label < 3), label, torch.full_like(label, 3))

    def encode_base(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        state: torch.Tensor,
        current_vlm_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.float()
        token_mask = token_mask.bool()
        state = state.float()
        current_vlm_label = self._safe_label(current_vlm_label)

        kv = self.context_proj(tokens)
        state_emb = self.state_encoder(state)
        cur_emb = self.current_vlm_emb(current_vlm_label)

        q = self.query_mlp(torch.cat([state_emb, cur_emb], dim=-1)).unsqueeze(1)
        q = q + self.learned_query.to(device=q.device, dtype=q.dtype)

        key_padding_mask = ~token_mask
        all_pad = key_padding_mask.all(dim=1)
        if bool(all_pad.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False

        context, attn_weights = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        context = self.context_norm(context.squeeze(1))
        base = self.base_fusion(torch.cat([context, state_emb, cur_emb], dim=-1))
        return base, attn_weights.squeeze(1)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        state: torch.Tensor,
        current_vlm_label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base, attn_weights = self.encode_base(tokens, token_mask, state, current_vlm_label)
        b, d = base.shape

        pair_action_emb = self.action_emb(self.pair_action_label)      # [9, d]
        pair_next_emb = self.next_vlm_emb(self.pair_next_vlm_label)    # [9, d]
        pair_action_emb = pair_action_emb.unsqueeze(0).expand(b, -1, -1)
        pair_next_emb = pair_next_emb.unsqueeze(0).expand(b, -1, -1)
        base_rep = base.unsqueeze(1).expand(-1, 9, -1)

        pair_feat = self.pair_fusion(torch.cat([base_rep, pair_action_emb, pair_next_emb], dim=-1))

        utility = self.utility_head(pair_feat).squeeze(-1).reshape(b, 3, 3)
        success_logit = self.success_head(pair_feat).squeeze(-1).reshape(b, 3, 3)
        cost = self.cost_head(pair_feat).squeeze(-1).reshape(b, 3, 3)
        steps = self.steps_head(pair_feat).squeeze(-1).reshape(b, 3, 3)
        base_value = self.base_value_head(base).squeeze(-1)

        return {
            "utility": utility,
            "success_logit": success_logit,
            "cost": cost,
            "steps": steps,
            "base_value": base_value,
            "base_feature": base,
            "attn_weights": attn_weights,
        }


def build_model_from_cache(
    *,
    context_dim: int,
    state_dim: int,
    d_model: int = 256,
    n_heads: int = 4,
    dropout: float = 0.1,
) -> OutcomeSurfaceModel:
    return OutcomeSurfaceModel(
        OutcomeSurfaceConfig(
            context_dim=int(context_dim),
            state_dim=int(state_dim),
            d_model=int(d_model),
            n_heads=int(n_heads),
            dropout=float(dropout),
        )
    )
