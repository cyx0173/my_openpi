#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_load import (
    SelectorDataset,
    selector_collate,
    split_cases_from_task_json,
    PRECISIONS,
)
from model import (
    StateConditionedQueryDualHeadSelector,
    precision_selector_loss,
    PrecisionMetricAccumulator,
)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def safe_class_weights(counts: np.ndarray) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)

    if present.any():
        total = counts[present].sum()
        weights[present] = total / (present.sum() * counts[present])
        weights[present] = weights[present] / max(weights[present].mean(), 1e-12)

    return torch.tensor(weights, dtype=torch.float32)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for k in (
        "prefix_hidden",
        "prefix_pad_mask",
        "state",
        "current_vlm_precision_id",
        "action_label",
        "next_vlm_label",
    ):
        out[k] = batch[k].to(device, non_blocking=True)
    return out


def weighted_mean(xs: list[float], ns: list[int]) -> float:
    total = sum(ns)
    if total <= 0:
        return 0.0
    return float(sum(x * n for x, n in zip(xs, ns)) / total)


def finite_score(x: float) -> float:
    if not math.isfinite(float(x)):
        return -1.0
    return float(x)


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    action_weight: torch.Tensor | None,
    next_weight: torch.Tensor | None,
    lambda_next: float,
    lambda_under: float,
    lambda_cost: float,
    cost_values: tuple[float, float, float],
    grad_clip: float,
    amp: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    loss_values = []
    action_loss_values = []
    next_loss_values = []
    nums = []

    action_accum = PrecisionMetricAccumulator(num_classes=3, ignore_index=-1)
    next_accum = PrecisionMetricAccumulator(num_classes=3, ignore_index=-1)

    for batch in loader:
        batch = to_device(batch, device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            action_logits, next_logits = model(
                prefix_hidden=batch["prefix_hidden"],
                prefix_pad_mask=batch["prefix_pad_mask"],
                state=batch["state"],
                current_vlm_precision_id=batch["current_vlm_precision_id"],
            )

            action_parts = precision_selector_loss(
                action_logits,
                batch["action_label"],
                class_weight=action_weight,
                lambda_under=lambda_under,
                lambda_cost=lambda_cost,
                cost_values=cost_values,
            )
            next_parts = precision_selector_loss(
                next_logits,
                batch["next_vlm_label"],
                class_weight=next_weight,
                lambda_under=lambda_under,
                lambda_cost=lambda_cost,
                cost_values=cost_values,
                ignore_index=-1,
            )
            loss = action_parts["loss"] + float(lambda_next) * next_parts["loss"]

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        n = int(batch["action_label"].numel())
        nums.append(n)
        loss_values.append(float(loss.detach().cpu()))
        action_loss_values.append(float(action_parts["loss"].detach().cpu()))
        next_loss_values.append(float(next_parts["loss"].detach().cpu()))

        action_accum.update(action_logits.detach(), batch["action_label"])
        next_accum.update(next_logits.detach(), batch["next_vlm_label"])

    return {
        "loss": weighted_mean(loss_values, nums),
        "action_loss": weighted_mean(action_loss_values, nums),
        "next_loss": weighted_mean(next_loss_values, nums),
        "action": action_accum.compute(),
        "next_vlm": next_accum.compute(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", default="/home/chengyuxuan/openpi/experiments/selector_dataset/task_schedule.json")
    ap.add_argument("--root", default="/home/chengyuxuan/openpi/experiments/selector_dataset")
    ap.add_argument("--out-dir", default="/home/chengyuxuan/openpi/experiments/selector_dataset/selector_runs/query_dual_head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--shard-cache-root", default=None)

    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--selector-dim", type=int, default=512)
    ap.add_argument("--num-queries", type=int, default=8)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mlp-ratio", type=int, default=4)

    ap.add_argument("--lambda-next", type=float, default=1.0)
    ap.add_argument("--lambda-under", type=float, default=0.5)
    ap.add_argument("--lambda-cost", type=float, default=0.02)
    ap.add_argument("--cost-values", default="1,2,4")

    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--max-chunks-per-case", type=int, default=None)
    ap.add_argument("--require-all-precisions", action="store_true")

    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument(
        "--save-optimizer",
        action="store_true",
        help="Save optimizer state in checkpoints. Off by default to keep checkpoints small.",
    )

    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "args.json", vars(args))

    split = split_cases_from_task_json(
        args.task_json,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    save_json(out_dir / "case_split.json", split)

    train_ds = SelectorDataset(
        task_json=args.task_json,
        root=args.root,
        cache_root=args.cache_root,
        shard_cache_root=args.shard_cache_root,
        case_ids=split["train"],
        require_all_precisions=args.require_all_precisions,
        max_cases=args.max_cases,
        max_chunks_per_case=args.max_chunks_per_case,
    )
    val_ds = SelectorDataset(
        task_json=args.task_json,
        root=args.root,
        cache_root=args.cache_root,
        shard_cache_root=args.shard_cache_root,
        case_ids=split["val"],
        require_all_precisions=args.require_all_precisions,
        max_cases=args.max_cases,
        max_chunks_per_case=args.max_chunks_per_case,
    ) if split["val"] else None
    test_ds = SelectorDataset(
        task_json=args.task_json,
        root=args.root,
        cache_root=args.cache_root,
        shard_cache_root=args.shard_cache_root,
        case_ids=split["test"],
        require_all_precisions=args.require_all_precisions,
        max_cases=args.max_cases,
        max_chunks_per_case=args.max_chunks_per_case,
    ) if split["test"] else None

    first = train_ds[0]
    hidden_dim = int(first["prefix_hidden"].shape[1])
    state_dim = int(first["state"].numel())

    dataset_summary = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "test_samples": len(test_ds) if test_ds else 0,
        "hidden_dim": hidden_dim,
        "state_dim": state_dim,
        "selector_dim": int(args.selector_dim),
        "precisions": list(PRECISIONS),
        "train_action_label_counts": train_ds.action_label_counts.tolist(),
        "train_next_vlm_label_counts": train_ds.next_vlm_label_counts.tolist(),
        "train_missing_count": len(train_ds.missing),
        "cache_root": args.cache_root,
        "shard_cache_root": args.shard_cache_root,
    }
    save_json(out_dir / "dataset_summary.json", dataset_summary)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False), flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=selector_collate,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=selector_collate,
        persistent_workers=args.num_workers > 0,
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=selector_collate,
        persistent_workers=args.num_workers > 0,
    ) if test_ds else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)

    model = StateConditionedQueryDualHeadSelector(
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        selector_dim=args.selector_dim,
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    dataset_summary["num_params"] = int(num_params)
    save_json(out_dir / "dataset_summary.json", dataset_summary)
    print(f"[MODEL] num_params={num_params/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    action_weight = safe_class_weights(train_ds.action_label_counts).to(device)
    next_weight = safe_class_weights(train_ds.next_vlm_label_counts).to(device)

    cost_values = tuple(float(x) for x in args.cost_values.split(","))
    if len(cost_values) != 3:
        raise ValueError("--cost-values must have three numbers, e.g. 1,2,4")

    best_score = -1.0
    history = []

    def make_ckpt(epoch: int) -> dict[str, Any]:
        ckpt = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "hidden_dim": hidden_dim,
            "state_dim": state_dim,
            "selector_dim": int(args.selector_dim),
            "dataset_summary": dataset_summary,
        }
        if args.save_optimizer:
            ckpt["optimizer_state"] = optimizer.state_dict()
        return ckpt

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            action_weight=action_weight,
            next_weight=next_weight,
            lambda_next=args.lambda_next,
            lambda_under=args.lambda_under,
            lambda_cost=args.lambda_cost,
            cost_values=cost_values,
            grad_clip=args.grad_clip,
            amp=args.amp,
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                action_weight=action_weight,
                next_weight=next_weight,
                lambda_next=args.lambda_next,
                lambda_under=args.lambda_under,
                lambda_cost=args.lambda_cost,
                cost_values=cost_values,
                grad_clip=args.grad_clip,
                amp=args.amp,
            )

        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "elapsed_s": elapsed,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        save_json(out_dir / "history.json", history)

        if val_metrics is not None:
            score = (
                0.5 * finite_score(float(val_metrics["action"]["macro_f1"]))
                + 0.5 * finite_score(float(val_metrics["next_vlm"]["macro_f1"]))
            )
        else:
            score = (
                0.5 * finite_score(float(train_metrics["action"]["macro_f1"]))
                + 0.5 * finite_score(float(train_metrics["next_vlm"]["macro_f1"]))
            )

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_action_acc={train_metrics['action']['acc']:.4f} "
            f"train_next_acc={train_metrics['next_vlm']['acc']:.4f} "
            f"train_action_under={train_metrics['action']['under_rate']:.4f} "
            f"train_next_under={train_metrics['next_vlm']['under_rate']:.4f} "
            + (
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_action_acc={val_metrics['action']['acc']:.4f} "
                f"val_next_acc={val_metrics['next_vlm']['acc']:.4f} "
                f"val_action_under={val_metrics['action']['under_rate']:.4f} "
                f"val_next_under={val_metrics['next_vlm']['under_rate']:.4f} "
                if val_metrics is not None else ""
            )
            + f"score={score:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )

        if epoch == 1 or score > best_score:
            best_score = score
            torch.save(make_ckpt(epoch), out_dir / "best.pt")
            save_json(out_dir / "best_metrics.json", row)

        if epoch == 1 or epoch % max(1, int(args.save_every)) == 0 or epoch == args.epochs:
            torch.save(make_ckpt(epoch), out_dir / "latest.pt")

    if test_loader is not None:
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            device=device,
            action_weight=action_weight,
            next_weight=next_weight,
            lambda_next=args.lambda_next,
            lambda_under=args.lambda_under,
            lambda_cost=args.lambda_cost,
            cost_values=cost_values,
            grad_clip=args.grad_clip,
            amp=args.amp,
        )
        save_json(out_dir / "test_metrics.json", test_metrics)
        print("[TEST]", json.dumps(test_metrics, indent=2, ensure_ascii=False), flush=True)

    print(f"[DONE] saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
'''
python src/openpi/models_pytorch/selector/train.py \
  --task-json /share/chengyuxuan-local/openpi/recovery_data_selector/task_schedule.json \
  --root /share/chengyuxuan-local/openpi/recovery_data_selector \
  --shard-cache-root /share/chengyuxuan-local/openpi/recovery_data_selector_shard_fp16 \
  --out-dir /home/chengyuxuan/openpi/experiments/selector/query_dual_head_v2_shard_bs128 \
  --batch-size 128 \
  --epochs 50 \
  --lr 1e-4 \
  --num-workers 8 \
  --selector-dim 512 \
  --require-all-precisions \
  --save-every 5 \
  --amp
'''