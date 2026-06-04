#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


PRECISIONS = ("w4a4", "w4a8", "w4a16")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def case_sort_key(case_id: str) -> tuple[int, int, str]:
    m = re.search(r"task(\d+)_ep(\d+)", str(case_id))
    if m is None:
        return (999999, 999999, str(case_id))
    return (int(m.group(1)), int(m.group(2)), str(case_id))


def parse_chunk_idx_from_state_name(name: str) -> int:
    m = re.fullmatch(r"chunk(\d+)_state\.npz", name)
    if m is None:
        raise ValueError(f"Bad state file name: {name}")
    return int(m.group(1))


def make_key(case_id: str, chunk_idx: int, precision: str) -> str:
    return f"{case_id}|{int(chunk_idx):04d}|{precision}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--valid-cache-root", required=True, help="small-file valid fp16 cache root")
    ap.add_argument("--shard-cache-root", required=True, help="output shard cache root")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--max-chunks-per-case", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    valid_cache_root = Path(args.valid_cache_root)
    shard_root = Path(args.shard_cache_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    hidden_path = shard_root / "hidden_valid_fp16.npy"
    length_path = shard_root / "valid_lengths.npy"
    index_path = shard_root / "shard_index.json"
    meta_path = shard_root / "shard_meta.json"

    if hidden_path.exists() and not args.overwrite:
        raise FileExistsError(f"{hidden_path} exists. Pass --overwrite to rebuild.")

    tasks = load_json(args.task_json)
    if isinstance(tasks, dict):
        tasks = tasks.get("jobs") or tasks.get("tasks") or tasks.get("data") or []
    if not isinstance(tasks, list):
        raise ValueError("task-json must be a list or dict with jobs/tasks/data")

    case_ids = sorted(
        {str(x["case_id"]) for x in tasks if "case_id" in x},
        key=case_sort_key,
    )
    if args.max_cases is not None:
        case_ids = case_ids[: int(args.max_cases)]

    rows = []
    max_t = 0
    hidden_dim = None

    t0 = time.time()
    print("[SCAN] collecting rows and shapes...", flush=True)

    for case_id in case_ids:
        state_dir = root / "cases" / case_id / "state"
        if not state_dir.exists():
            print(f"[WARN] missing state dir: {state_dir}", flush=True)
            continue

        state_paths = sorted(
            state_dir.glob("chunk*_state.npz"),
            key=lambda p: parse_chunk_idx_from_state_name(p.name),
        )
        if args.max_chunks_per_case is not None:
            state_paths = state_paths[: int(args.max_chunks_per_case)]

        for st in state_paths:
            k = parse_chunk_idx_from_state_name(st.name)
            for precision in PRECISIONS:
                src = valid_cache_root / "cases" / case_id / "mid_valid" / precision / f"chunk{k:04d}.npy"
                if not src.exists():
                    raise FileNotFoundError(f"missing valid cache: {src}")

                arr = np.load(src, mmap_mode="r")
                if arr.ndim != 2:
                    raise ValueError(f"bad hidden shape {arr.shape}: {src}")
                if arr.dtype != np.float16:
                    raise ValueError(f"expected fp16, got {arr.dtype}: {src}")

                if hidden_dim is None:
                    hidden_dim = int(arr.shape[1])
                elif hidden_dim != int(arr.shape[1]):
                    raise ValueError(
                        f"hidden_dim mismatch: {hidden_dim} vs {arr.shape[1]} at {src}"
                    )

                max_t = max(max_t, int(arr.shape[0]))
                rows.append(
                    {
                        "row": len(rows),
                        "key": make_key(case_id, k, precision),
                        "case_id": case_id,
                        "chunk_idx": int(k),
                        "precision": precision,
                        "src": str(src),
                        "valid_len": int(arr.shape[0]),
                    }
                )

    if hidden_dim is None or not rows:
        raise RuntimeError("No rows found")

    n = len(rows)
    print(
        json.dumps(
            {
                "num_rows": n,
                "max_t": max_t,
                "hidden_dim": hidden_dim,
                "estimated_gb": n * max_t * hidden_dim * 2 / (1024**3),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    print("[BUILD] creating memmap npy shard...", flush=True)
    hidden = np.lib.format.open_memmap(
        hidden_path,
        mode="w+",
        dtype=np.float16,
        shape=(n, max_t, hidden_dim),
    )
    lengths = np.zeros((n,), dtype=np.int32)

    for i, r in enumerate(rows):
        src = Path(r["src"])
        arr = np.load(src, mmap_mode="r")
        valid_len = int(arr.shape[0])
        hidden[i, :valid_len, :] = arr
        if valid_len < max_t:
            hidden[i, valid_len:, :] = 0
        lengths[i] = valid_len

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            print(
                f"[BUILD] {i+1}/{n} elapsed={elapsed:.1f}s speed={(i+1)/max(elapsed,1e-6):.1f} rows/s",
                flush=True,
            )

    hidden.flush()
    np.save(length_path, lengths)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    meta = {
        "task_json": str(args.task_json),
        "root": str(root),
        "valid_cache_root": str(valid_cache_root),
        "shard_cache_root": str(shard_root),
        "hidden_path": str(hidden_path),
        "length_path": str(length_path),
        "index_path": str(index_path),
        "num_rows": n,
        "num_cases": len(case_ids),
        "max_t": int(max_t),
        "hidden_dim": int(hidden_dim),
        "dtype": "float16",
        "elapsed_s": time.time() - t0,
        "valid_tokens": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "p50": float(np.percentile(lengths, 50)),
            "p95": float(np.percentile(lengths, 95)),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
