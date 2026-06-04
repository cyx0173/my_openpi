#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class StateConditionedQueryDualHeadSelector(nn.Module):
    """
    Lightweight state-conditioned query-attention dual-head precision selector.

    Difference from the first smoke-test version:
      - prefix hidden is first projected from hidden_dim, e.g. 2048, to selector_dim, e.g. 512.
      - attention and MLP run in selector_dim instead of 2048.
      - checkpoint / optimizer state / GPU memory are much smaller.

    Input:
      prefix_hidden: [B, T, hidden_dim]
      prefix_pad_mask: [B, T], True = valid token
      state: [B, state_dim]
      current_vlm_precision_id: [B], 0/1/2

    Output:
      action_logits: [B, 3]
      next_vlm_logits: [B, 3]
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        state_dim: int,
        num_classes: int = 3,
        selector_dim: int = 512,
        num_queries: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = int(hidden_dim)
        selector_dim = int(selector_dim)

        if selector_dim % int(num_heads) != 0:
            raise ValueError(
                f"selector_dim={selector_dim} must be divisible by num_heads={num_heads}"
            )

        self.hidden_dim = hidden_dim
        self.selector_dim = selector_dim
        self.state_dim = int(state_dim)
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)

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

        self.precision_embedding = nn.Embedding(num_classes, selector_dim)

        self.base_queries = nn.Parameter(
            torch.randn(num_queries, selector_dim) * 0.02
        )

        # Conditioning is now cheap because it operates in selector_dim.
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
            nn.Linear(selector_dim, selector_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(selector_dim * mlp_ratio, selector_dim),
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

        self.action_head = nn.Linear(head_dim, num_classes)
        self.next_vlm_head = nn.Linear(head_dim, num_classes)

    def forward(
        self,
        *,
        prefix_hidden: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        state: torch.Tensor,
        current_vlm_precision_id: torch.Tensor,
        return_attn: bool = False,
    ):
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

        # PyTorch key_padding_mask: True means "ignore".
        key_padding_mask = ~mask

        attended, attn_weights = self.cross_attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        q = self.query_norm1(q + attended)

        self_attended, _ = self.query_self_attn(
            query=q,
            key=q,
            value=q,
            need_weights=False,
        )
        q = self.query_norm2(q + self_attended)

        q = self.query_norm3(q + self.query_ffn(q))

        q_flat = q.flatten(start_dim=1)
        feature = torch.cat([q_flat, state_feat, precision_feat], dim=-1)
        shared = self.shared_mlp(feature)

        action_logits = self.action_head(shared)
        next_vlm_logits = self.next_vlm_head(shared)

        if return_attn:
            return action_logits, next_vlm_logits, attn_weights
        return action_logits, next_vlm_logits


def precision_selector_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weight: torch.Tensor | None = None,
    lambda_under: float = 0.5,
    lambda_cost: float = 0.02,
    cost_values: tuple[float, float, float] = (1.0, 2.0, 4.0),
    ignore_index: int = -1,
) -> dict[str, torch.Tensor]:
    """
    Cost-sensitive ordinal loss for precision prediction.

    labels:
      0 = w4a4
      1 = w4a8
      2 = w4a16
      -1 ignored, used for final chunk next-vlm label.
    """
    valid = labels.ne(ignore_index)
    if not torch.any(valid):
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "ce": zero.detach(),
            "under_risk": zero.detach(),
            "expected_cost": zero.detach(),
        }

    logits_v = logits[valid]
    labels_v = labels[valid].long()

    weight_v = class_weight.to(logits.device) if class_weight is not None else None
    ce = F.cross_entropy(logits_v, labels_v, weight=weight_v)

    prob = torch.softmax(logits_v, dim=-1)

    penalty = torch.tensor(
        [
            # pred A4, pred A8, pred A16
            [0.0, 0.2, 0.5],  # label A4
            [1.0, 0.0, 0.2],  # label A8
            [2.0, 1.0, 0.0],  # label A16
        ],
        dtype=logits.dtype,
        device=logits.device,
    )
    under_risk = (prob * penalty[labels_v]).sum(dim=-1).mean()

    cost = torch.tensor(cost_values, dtype=logits.dtype, device=logits.device)
    expected_cost = (prob * cost[None, :]).sum(dim=-1).mean()

    loss = ce + float(lambda_under) * under_risk + float(lambda_cost) * expected_cost

    return {
        "loss": loss,
        "ce": ce.detach(),
        "under_risk": under_risk.detach(),
        "expected_cost": expected_cost.detach(),
    }


class PrecisionMetricAccumulator:
    """Accumulate confusion over the whole epoch, then compute metrics once.

    This avoids the smoke-test nan issue caused by averaging per-batch macro-F1
    when some classes are absent in tiny batches.
    """

    def __init__(self, num_classes: int = 3, ignore_index: int = -1):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.num = 0
        self.under = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        valid = labels.ne(self.ignore_index)
        if not torch.any(valid):
            return

        pred = logits[valid].argmax(dim=-1).detach().cpu()
        y = labels[valid].long().detach().cpu()

        self.num += int(y.numel())
        self.under += int((pred < y).sum().item())

        for t, p in zip(y.tolist(), pred.tolist()):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion[t, p] += 1

    def compute(self) -> dict[str, object]:
        conf = self.confusion
        num = int(self.num)
        if num <= 0:
            return {
                "num": 0,
                "acc": 0.0,
                "macro_f1": 0.0,
                "under_rate": 0.0,
                "confusion": conf.tolist(),
                "recall": [0.0 for _ in range(self.num_classes)],
                "precision": [0.0 for _ in range(self.num_classes)],
                "f1": [0.0 for _ in range(self.num_classes)],
            }

        acc = float(conf.diag().sum().item() / max(1, num))
        under_rate = float(self.under / max(1, num))

        recalls = []
        precisions = []
        f1s = []
        active_f1s = []

        for c in range(self.num_classes):
            tp = conf[c, c].item()
            support = conf[c, :].sum().item()
            pred_count = conf[:, c].sum().item()

            recall = tp / support if support > 0 else 0.0
            precision = tp / pred_count if pred_count > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            recalls.append(float(recall))
            precisions.append(float(precision))
            f1s.append(float(f1))

            # Macro-F1 over classes that exist in labels for this split.
            if support > 0:
                active_f1s.append(float(f1))

        macro_f1 = float(sum(active_f1s) / len(active_f1s)) if active_f1s else 0.0

        return {
            "num": num,
            "acc": acc,
            "macro_f1": macro_f1,
            "under_rate": under_rate,
            "confusion": conf.tolist(),
            "recall": recalls,
            "precision": precisions,
            "f1": f1s,
        }
