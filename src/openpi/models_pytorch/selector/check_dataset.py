#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_load import SelectorDataset, PRECISIONS


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--max-chunks-per-case", type=int, default=None)
    ap.add_argument("--require-all-precisions", action="store_true")
    args = ap.parse_args()

    ds = SelectorDataset(
        task_json=args.task_json,
        root=args.root,
        max_cases=args.max_cases,
        max_chunks_per_case=args.max_chunks_per_case,
        require_all_precisions=args.require_all_precisions,
    )

    first = ds[0]
    summary = {
        "num_samples": len(ds),
        "num_cases_in_task_json": len(ds.items),
        "hidden_shape_first": list(first["prefix_hidden"].shape),
        "state_dim_first": int(first["state"].numel()),
        "precisions": list(PRECISIONS),
        "action_label_counts": ds.action_label_counts.tolist(),
        "next_vlm_label_counts": ds.next_vlm_label_counts.tolist(),
        "missing_count": len(ds.missing),
        "missing_examples": ds.missing[:20],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if args.out:
        save_json(args.out, summary)


if __name__ == "__main__":
    main()
