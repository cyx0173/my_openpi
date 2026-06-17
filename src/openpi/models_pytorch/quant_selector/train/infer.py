#!/usr/bin/env python3
"""
Offline inference test for the cross-attention precision selector.

This script loads a trained selector checkpoint from train_selector.py,
runs constrained inference on clean_v8a8_labels.jsonl, and prints:
  - static baselines
  - selector success/cost under eps
  - selected pair distribution
  - key-chunk behavior
  - optional per-chunk JSONL predictions

Run from the same directory as train_selector.py:
  python infer_selector.py --labels .../clean_v8a8_labels.jsonl --ckpt .../selector_demo.pt --split_by_episode --eps 0.1
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_selector import (
    PAIRS,
    A4_IDX,
    BASELINE_IDX,
    CrossAttnSelector,
    SelectorDataset,
    collate,
    constrained_select,
    load_or_build_items,
)


def split_rows(rows, split_by_episode: bool, val_frac: float, seed: int, split: str):
    if not split_by_episode:
        return rows, rows, rows

    rng = random.Random(seed)
    cases = sorted({r["case_id"] for r in rows})
    rng.shuffle(cases)
    n_val = max(1, int(len(cases) * val_frac))
    val_cases = set(cases[:n_val])
    train_rows = [r for r in rows if r["case_id"] not in val_cases]
    val_rows = [r for r in rows if r["case_id"] in val_cases]

    if split == "train":
        eval_rows = train_rows
    elif split == "val":
        eval_rows = val_rows
    elif split == "all":
        eval_rows = rows
    else:
        raise ValueError(f"Unknown split: {split}")
    return train_rows, val_rows, eval_rows


def pair_to_bits(pair: str):
    v, a = pair.split("_")
    return int(v[1:]), int(a[1:])


@torch.no_grad()
def run_inference(model, items, rows, device, dcost_std: float, eps: float, batch_size: int):
    ds = SelectorDataset(items)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate, shuffle=False, num_workers=0)

    pred_succ_parts = []
    pred_cost_parts = []
    true_succ_parts = []
    true_dcost_parts = []
    key_parts = []
    base_cost_parts = []

    model.eval()
    for batch in loader:
        ctx = batch["ctx"].to(device)
        mask = batch["mask"].to(device)
        state = batch["state"].to(device)
        cvb = batch["current_vlm_bits"]
        pred = model(ctx, mask, state, cvb)
        pred_succ_parts.append(pred["d_success"].cpu())
        pred_cost_parts.append((pred["d_cost"].cpu() * dcost_std))
        true_succ_parts.append(batch["success_vec"].cpu())
        true_dcost_parts.append(batch["d_cost"].cpu())
        key_parts.append(batch["is_key"].cpu())
        base_cost_parts.append(batch["baseline_cost"].cpu())

    pred_succ = torch.cat(pred_succ_parts, dim=0)
    pred_cost = torch.cat(pred_cost_parts, dim=0)
    true_succ = torch.cat(true_succ_parts, dim=0)
    true_dcost = torch.cat(true_dcost_parts, dim=0)
    is_key = torch.cat(key_parts, dim=0).bool()
    base_cost = torch.cat(base_cost_parts, dim=0)
    abs_cost = base_cost[:, None] + true_dcost

    choice = constrained_select(pred_succ, pred_cost, eps)
    n = len(choice)
    idx = torch.arange(n)
    chosen_succ = true_succ[idx, choice]
    chosen_cost = abs_cost[idx, choice]

    # Oracle under true outcomes: choose cheapest truly successful pair.
    oracle_cost = abs_cost.clone()
    oracle_cost[true_succ < 0.5] = float("inf")
    oracle_choice = oracle_cost.argmin(dim=1)
    empty = torch.isinf(oracle_cost).all(dim=1)
    if empty.any():
        oracle_choice[empty] = BASELINE_IDX
    oracle_succ = true_succ[idx, oracle_choice]
    oracle_chosen_cost = abs_cost[idx, oracle_choice]

    a4_succ = true_succ[:, A4_IDX].mean().item()
    a4_cost = abs_cost[:, A4_IDX].mean().item()
    base_succ = true_succ[:, BASELINE_IDX].mean().item()
    base_cost_mean = abs_cost[:, BASELINE_IDX].mean().item()

    print("\n" + "=" * 72)
    print(f"Offline inference: N={n}, eps={eps}, key_chunks={int(is_key.sum().item())}")
    print("=" * 72)
    print("Static baselines:")
    print(f"  always v4_a4     success={a4_succ:.3f}  cost={a4_cost:.1f}")
    print(f"  always v8_a8     success={base_succ:.3f}  cost={base_cost_mean:.1f}")
    print("Selector:")
    print(f"  success={chosen_succ.mean().item():.3f}  cost={chosen_cost.mean().item():.1f}")
    if is_key.any():
        print(f"  key_chunk_success={chosen_succ[is_key].mean().item():.3f}  "
              f"key_chunk_count={int(is_key.sum().item())}")
    print(f"  a4_choice_ratio={(choice == A4_IDX).float().mean().item():.3f}")
    print("Oracle under labels:")
    print(f"  success={oracle_succ.mean().item():.3f}  cost={oracle_chosen_cost.mean().item():.1f}")

    print("\nSelected pair distribution:")
    dist = Counter(PAIRS[int(i)] for i in choice.tolist())
    for p in PAIRS:
        c = dist.get(p, 0)
        print(f"  {p:8s} {c:5d}  {c / n:7.3%}")

    if is_key.any():
        print("\nKey-chunk selected pair distribution:")
        key_choices = choice[is_key]
        key_dist = Counter(PAIRS[int(i)] for i in key_choices.tolist())
        for p in PAIRS:
            c = key_dist.get(p, 0)
            print(f"  {p:8s} {c:5d}  {c / max(1, len(key_choices)):7.3%}")

    # Detailed first few key chunks.
    print("\nFirst key-chunk details:")
    key_indices = torch.nonzero(is_key).flatten().tolist()[:20]
    if not key_indices:
        print("  <no key chunks in this split>")
    for i in key_indices:
        row = rows[i]
        ch = int(choice[i])
        oc = int(oracle_choice[i])
        print(
            f"  {row['case_id']} chunk={row['chunk_idx']:>3} "
            f"selected={PAIRS[ch]:8s} true_succ={int(true_succ[i, ch].item())} "
            f"true_cost={abs_cost[i, ch].item():.1f} "
            f"oracle={PAIRS[oc]:8s} oracle_cost={abs_cost[i, oc].item():.1f}"
        )

    return {
        "choice": choice,
        "oracle_choice": oracle_choice,
        "pred_succ": pred_succ,
        "pred_cost": pred_cost,
        "true_succ": true_succ,
        "abs_cost": abs_cost,
        "is_key": is_key,
    }


def save_predictions(path, rows, result, eps):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    choice = result["choice"]
    oracle_choice = result["oracle_choice"]
    pred_succ = result["pred_succ"]
    pred_cost = result["pred_cost"]
    true_succ = result["true_succ"]
    abs_cost = result["abs_cost"]
    is_key = result["is_key"]

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            ch = int(choice[i])
            oc = int(oracle_choice[i])
            selected_pair = PAIRS[ch]
            sv, sa = pair_to_bits(selected_pair)
            rec = {
                "case_id": row["case_id"],
                "task_id": row.get("task_id"),
                "chunk_idx": row["chunk_idx"],
                "current_vlm_a_bits": row.get("current_vlm_a_bits"),
                "eps": eps,
                "selected_pair": selected_pair,
                "selected_vlm_a_bits": sv,
                "selected_action_a_bits": sa,
                "selected_true_success": float(true_succ[i, ch].item()),
                "selected_true_cost": float(abs_cost[i, ch].item()),
                "oracle_pair": PAIRS[oc],
                "oracle_true_cost": float(abs_cost[i, oc].item()),
                "is_key_chunk": bool(is_key[i].item()),
                "pred_d_success": [float(x) for x in pred_succ[i].tolist()],
                "pred_d_cost": [float(x) for x in pred_cost[i].tolist()],
                "true_success_vec": [float(x) for x in true_succ[i].tolist()],
                "true_abs_cost": [float(x) for x in abs_cost[i].tolist()],
                "pair_order": PAIRS,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved predictions: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpt", default="/home/chengyuxuan/openpi/experiments/selector_dataset/selector_demo.pt")
    ap.add_argument("--cache_dir", default="/home/chengyuxuan/openpi/experiments/selector_dataset/selcache")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--split_by_episode", action="store_true")
    ap.add_argument("--split", choices=["train", "val", "all"], default="val")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--save_jsonl", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    rows_all = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
    train_rows, val_rows, eval_rows = split_rows(
        rows_all, args.split_by_episode, args.val_frac, args.seed, args.split
    )
    if args.split_by_episode:
        print(f"split_by_episode: train={len(train_rows)} val={len(val_rows)} eval={len(eval_rows)} split={args.split}")
    else:
        print(f"all-data eval: N={len(eval_rows)}")

    # Cache is keyed by eval rows; existing train/val caches will be reused if row set and max_tokens match.
    items = load_or_build_items(eval_rows, args.cache_dir, args.max_tokens, tag=f"infer_{args.split}")

    ckpt = torch.load(args.ckpt, map_location=device)
    dcost_std = float(ckpt.get("dcost_std", 1.0))
    model = CrossAttnSelector().to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded ckpt: {args.ckpt}")
    print(f"dcost_std={dcost_std:.3f}")

    result = run_inference(model, items, eval_rows, device, dcost_std, args.eps, args.batch_size)

    if args.save_jsonl:
        save_predictions(args.save_jsonl, eval_rows, result, args.eps)


if __name__ == "__main__":
    main()
