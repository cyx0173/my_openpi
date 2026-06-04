#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    np.save(tmp, arr)
    # np.save adds .npy if the path does not end in .npy.
    actual_tmp = tmp if tmp.exists() else Path(str(tmp) + ".npy")
    actual_tmp.replace(path)


def build_one(
    *,
    root: Path,
    cache_root: Path,
    case_id: str,
    chunk_idx: int,
    precision: str,
    overwrite: bool,
) -> dict[str, Any]:
    src = root / "cases" / case_id / "mid" / precision / f"chunk{chunk_idx:04d}.npz"
    dst = cache_root / "cases" / case_id / "mid_valid" / precision / f"chunk{chunk_idx:04d}.npy"

    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return {
            "ok": True,
            "skipped": True,
            "case_id": case_id,
            "chunk_idx": chunk_idx,
            "precision": precision,
            "dst": str(dst),
        }

    if not src.exists():
        return {
            "ok": False,
            "error": f"missing source: {src}",
            "case_id": case_id,
            "chunk_idx": chunk_idx,
            "precision": precision,
        }

    try:
        md = np.load(src, allow_pickle=False)
        hidden = np.asarray(md["vlm_prefix_last_hidden"])
        mask = np.asarray(md["prefix_pad_mask"]).astype(np.bool_)

        if hidden.ndim != 2:
            raise ValueError(f"hidden must be [T,D], got {hidden.shape}")
        if mask.ndim != 1:
            raise ValueError(f"mask must be [T], got {mask.shape}")
        if hidden.shape[0] != mask.shape[0]:
            raise ValueError(f"hidden/mask mismatch: {hidden.shape}, {mask.shape}")

        valid_hidden = hidden[mask]

        # Keep original effective precision. Original saved mid is already fp16.
        # This does not add extra quantization.
        if valid_hidden.dtype != np.float16:
            valid_hidden = valid_hidden.astype(np.float16)

        atomic_save_npy(dst, valid_hidden)

        return {
            "ok": True,
            "skipped": False,
            "case_id": case_id,
            "chunk_idx": chunk_idx,
            "precision": precision,
            "src": str(src),
            "dst": str(dst),
            "orig_shape": list(hidden.shape),
            "valid_shape": list(valid_hidden.shape),
            "valid_tokens": int(valid_hidden.shape[0]),
            "hidden_dim": int(valid_hidden.shape[1]),
            "dtype": str(valid_hidden.dtype),
        }

    except Exception as e:
        return {
            "ok": False,
            "error": repr(e),
            "case_id": case_id,
            "chunk_idx": chunk_idx,
            "precision": precision,
            "src": str(src),
            "dst": str(dst),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--max-chunks-per-case", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

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

    jobs: list[tuple[str, int, str]] = []
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
                jobs.append((case_id, k, precision))

    print(
        json.dumps(
            {
                "root": str(root),
                "cache_root": str(cache_root),
                "num_cases": len(case_ids),
                "num_cache_files": len(jobs),
                "workers": int(args.workers),
                "overwrite": bool(args.overwrite),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    t0 = time.time()
    ok = 0
    skipped = 0
    fail = 0
    valid_tokens = []
    errors = []

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [
            ex.submit(
                build_one,
                root=root,
                cache_root=cache_root,
                case_id=case_id,
                chunk_idx=k,
                precision=p,
                overwrite=bool(args.overwrite),
            )
            for case_id, k, p in jobs
        ]

        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r.get("ok"):
                ok += 1
                if r.get("skipped"):
                    skipped += 1
                if "valid_tokens" in r:
                    valid_tokens.append(int(r["valid_tokens"]))
            else:
                fail += 1
                errors.append(r)

            if i % 500 == 0 or i == len(futs):
                elapsed = time.time() - t0
                speed = i / max(elapsed, 1e-6)
                print(
                    f"[CACHE] {i}/{len(futs)} "
                    f"ok={ok} skipped={skipped} fail={fail} "
                    f"speed={speed:.1f} files/s elapsed={elapsed:.1f}s",
                    flush=True,
                )

    meta = {
        "root": str(root),
        "cache_root": str(cache_root),
        "num_cases": len(case_ids),
        "num_cache_files": len(jobs),
        "ok": ok,
        "skipped": skipped,
        "fail": fail,
        "elapsed_s": time.time() - t0,
        "valid_tokens": {
            "count": len(valid_tokens),
            "min": min(valid_tokens) if valid_tokens else None,
            "max": max(valid_tokens) if valid_tokens else None,
            "mean": float(np.mean(valid_tokens)) if valid_tokens else None,
            "p50": float(np.percentile(valid_tokens, 50)) if valid_tokens else None,
            "p95": float(np.percentile(valid_tokens, 95)) if valid_tokens else None,
        },
        "errors": errors[:50],
    }

    with open(cache_root / "cache_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)

    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()