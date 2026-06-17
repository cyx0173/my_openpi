#!/usr/bin/env python3
from __future__ import annotations

import torch

from openpi.models_pytorch.quant_selector.selector.model import (
    StateConditionedQueryDualHeadSelector,
    precision_selector_loss,
)


DEVICE = "cuda"

HIDDEN_DIM = 2048
STATE_DIM = 10
SELECTOR_DIM = 512
NUM_QUERIES = 8
NUM_HEADS = 8
MLP_RATIO = 4

SEQ_LEN = 64

B_INFER = 1
B_TRAIN = 32

WARMUP = 20
ITERS = 100

LR = 1e-4
NEXT_LOSS_WEIGHT = 0.5


def sync():
    torch.cuda.synchronize()


def mem_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def make_batch(batch_size: int):
    return {
        "prefix_hidden": torch.randn(batch_size, SEQ_LEN, HIDDEN_DIM, device=DEVICE),
        "prefix_pad_mask": torch.ones(batch_size, SEQ_LEN, dtype=torch.bool, device=DEVICE),
        "state": torch.randn(batch_size, STATE_DIM, device=DEVICE),
        "current_vlm_precision_id": torch.randint(0, 3, (batch_size,), device=DEVICE),
        "action_label": torch.randint(0, 3, (batch_size,), device=DEVICE),
        "next_vlm_label": torch.randint(0, 3, (batch_size,), device=DEVICE),
    }


def forward(model, batch):
    return model(
        prefix_hidden=batch["prefix_hidden"],
        prefix_pad_mask=batch["prefix_pad_mask"],
        state=batch["state"],
        current_vlm_precision_id=batch["current_vlm_precision_id"],
    )


@torch.no_grad()
def bench_infer(model):
    model.eval()
    batch = make_batch(B_INFER)

    for _ in range(WARMUP):
        forward(model, batch)
    sync()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        forward(model, batch)
    end.record()
    sync()

    print(f"[infer] B={B_INFER} T={SEQ_LEN}")
    print(f"  ms/iter: {start.elapsed_time(end) / ITERS:.4f}")
    print(f"  peak MB: {mem_mb():.2f}")


def bench_train(model):
    model.train()
    batch = make_batch(B_TRAIN)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for _ in range(WARMUP):
        optim.zero_grad(set_to_none=True)
        action_logits, next_logits = forward(model, batch)
        loss = (
            precision_selector_loss(action_logits, batch["action_label"])["loss"]
            + NEXT_LOSS_WEIGHT
            * precision_selector_loss(next_logits, batch["next_vlm_label"])["loss"]
        )
        loss.backward()
        optim.step()
    sync()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        optim.zero_grad(set_to_none=True)
        action_logits, next_logits = forward(model, batch)
        loss = (
            precision_selector_loss(action_logits, batch["action_label"])["loss"]
            + NEXT_LOSS_WEIGHT
            * precision_selector_loss(next_logits, batch["next_vlm_label"])["loss"]
        )
        loss.backward()
        optim.step()
    end.record()
    sync()

    print(f"[train] B={B_TRAIN} T={SEQ_LEN}")
    print(f"  ms/iter: {start.elapsed_time(end) / ITERS:.4f}")
    print(f"  peak MB: {mem_mb():.2f}")
    print(f"  loss: {float(loss.detach().cpu()):.4f}")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = StateConditionedQueryDualHeadSelector(
        hidden_dim=HIDDEN_DIM,
        state_dim=STATE_DIM,
        selector_dim=SELECTOR_DIM,
        num_classes=3,
        num_queries=NUM_QUERIES,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=0.1,
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"params: {params:,}")
    print(f"param MB: {param_mb:.2f}")
    print(f"hidden_dim={HIDDEN_DIM}, selector_dim={SELECTOR_DIM}, queries={NUM_QUERIES}")
    print("=" * 80)

    bench_infer(model)
    bench_train(model)


if __name__ == "__main__":
    main()