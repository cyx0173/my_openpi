#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


PRECISIONS = ("w4a4", "w4a8", "w4a16")
PRECISION_TO_ID = {"w4a4": 0, "w4a8": 1, "w4a16": 2}
ID_TO_PRECISION = {v: k for k, v in PRECISION_TO_ID.items()}


def normalize_precision(x: str) -> str:
    x = str(x).strip().lower()
    if x in ("a4", "w4a4"):
        return "w4a4"
    if x in ("a8", "w4a8"):
        return "w4a8"
    if x in ("a16", "w4a16"):
        return "w4a16"
    raise ValueError(f"Unknown precision: {x}")


def precision_to_id(x: str) -> int:
    return PRECISION_TO_ID[normalize_precision(x)]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_chunk_idx_from_state_name(name: str) -> int:
    m = re.fullmatch(r"chunk(\d+)_state\.npz", name)
    if m is None:
        raise ValueError(f"Bad state file name: {name}")
    return int(m.group(1))


def case_sort_key(case_id: str) -> tuple[int, int, str]:
    m = re.search(r"task(\d+)_ep(\d+)", str(case_id))
    if m is None:
        return (999999, 999999, str(case_id))
    return (int(m.group(1)), int(m.group(2)), str(case_id))


def make_shard_key(case_id: str, chunk_idx: int, precision: str) -> str:
    return f"{case_id}|{int(chunk_idx):04d}|{normalize_precision(precision)}"


@dataclass(frozen=True)
class SelectorSample:
    case_id: str
    chunk_idx: int
    current_vlm_precision: str
    current_vlm_precision_id: int
    mid_path: Path
    state_path: Path
    action_label_id: int
    next_vlm_label_id: int


class SelectorDataset(Dataset):
    """
    Dataset for the dual-head look-again precision selector.

    Hidden loading priority:
      1. shard_cache_root: one big hidden_valid_fp16.npy memmap + index
      2. cache_root: many small mid_valid/<precision>/chunkXXXX.npy files
      3. original compressed npz
    """

    def __init__(
        self,
        *,
        task_json: str | Path,
        root: str | Path = "/home/chengyuxuan/openpi/experiments/selector_dataset",
        cache_root: str | Path | None = None,
        shard_cache_root: str | Path | None = None,
        case_ids: Iterable[str] | None = None,
        mid_dir_name: str = "mid",
        require_all_precisions: bool = False,
        max_cases: int | None = None,
        max_chunks_per_case: int | None = None,
    ):
        super().__init__()

        self.task_json = Path(task_json)
        self.root = Path(root)
        self.mid_dir_name = str(mid_dir_name)
        self.cache_root = Path(cache_root) if cache_root else None
        self.shard_cache_root = Path(shard_cache_root) if shard_cache_root else None

        self._shard_hidden = None
        self._shard_key_to_index: dict[str, int] | None = None
        self._shard_lengths = None

        if self.shard_cache_root is not None:
            self._load_shard_metadata()

        raw_items = load_json(self.task_json)
        if not isinstance(raw_items, list):
            raise ValueError(f"{self.task_json} must contain a list of task items")

        allowed = set(case_ids) if case_ids is not None else None

        items: list[dict[str, Any]] = []
        for x in raw_items:
            if "case_id" not in x:
                continue
            cid = str(x["case_id"])
            if allowed is not None and cid not in allowed:
                continue
            items.append(dict(x))

        items.sort(key=lambda x: case_sort_key(str(x["case_id"])))
        if max_cases is not None:
            items = items[: int(max_cases)]

        self.items = items
        self.samples: list[SelectorSample] = []
        self.action_label_counts = np.zeros(3, dtype=np.int64)
        self.next_vlm_label_counts = np.zeros(3, dtype=np.int64)
        self.missing: list[str] = []

        self._build_samples(
            require_all_precisions=bool(require_all_precisions),
            max_chunks_per_case=max_chunks_per_case,
        )

        if not self.samples:
            raise RuntimeError(
                "No selector samples found. Check task_json/root/cache layout."
            )

    def _load_shard_metadata(self) -> None:
        assert self.shard_cache_root is not None
        meta_path = self.shard_cache_root / "shard_meta.json"
        index_path = self.shard_cache_root / "shard_index.json"
        length_path = self.shard_cache_root / "valid_lengths.npy"
        hidden_path = self.shard_cache_root / "hidden_valid_fp16.npy"

        for p in (meta_path, index_path, length_path, hidden_path):
            if not p.exists():
                raise FileNotFoundError(f"missing shard file: {p}")

        rows = load_json(index_path)
        self._shard_key_to_index = {str(x["key"]): int(x["row"]) for x in rows}
        self._shard_lengths = np.load(length_path, mmap_mode="r")

    def _get_shard_hidden(self):
        if self._shard_hidden is None:
            assert self.shard_cache_root is not None
            self._shard_hidden = np.load(
                self.shard_cache_root / "hidden_valid_fp16.npy",
                mmap_mode="r",
            )
        return self._shard_hidden

    def _read_state_labels(self, state_path: Path) -> tuple[int, int]:
        st = np.load(state_path, allow_pickle=False)

        if "action_label_id" in st:
            action_label = int(st["action_label_id"])
        elif "label_id" in st:
            action_label = int(st["label_id"])
        else:
            raise KeyError(f"{state_path} has no action_label_id / label_id")

        if "next_vlm_label_id" in st:
            next_label = int(st["next_vlm_label_id"])
        elif "next_label_id" in st:
            next_label = int(st["next_label_id"])
        else:
            next_label = -1

        if action_label not in (0, 1, 2):
            raise ValueError(f"Bad action_label={action_label} in {state_path}")
        if next_label not in (-1, 0, 1, 2):
            raise ValueError(f"Bad next_vlm_label={next_label} in {state_path}")

        return action_label, next_label

    def _build_samples(
        self,
        *,
        require_all_precisions: bool,
        max_chunks_per_case: int | None,
    ) -> None:
        for item in self.items:
            case_id = str(item["case_id"])
            case_root = self.root / "cases" / case_id
            state_dir = case_root / "state"
            mid_root = case_root / self.mid_dir_name

            if not state_dir.exists():
                self.missing.append(f"missing state_dir: {state_dir}")
                continue

            if self.shard_cache_root is None and not mid_root.exists():
                self.missing.append(f"missing mid_root: {mid_root}")
                continue

            state_paths = sorted(
                state_dir.glob("chunk*_state.npz"),
                key=lambda p: parse_chunk_idx_from_state_name(p.name),
            )
            if max_chunks_per_case is not None:
                state_paths = state_paths[: int(max_chunks_per_case)]

            for state_path in state_paths:
                chunk_idx = parse_chunk_idx_from_state_name(state_path.name)
                action_label, next_label = self._read_state_labels(state_path)

                mid_paths = {
                    p: mid_root / p / f"chunk{chunk_idx:04d}.npz"
                    for p in PRECISIONS
                }

                if self.shard_cache_root is not None:
                    assert self._shard_key_to_index is not None
                    missing_keys = [
                        make_shard_key(case_id, chunk_idx, p)
                        for p in PRECISIONS
                        if make_shard_key(case_id, chunk_idx, p) not in self._shard_key_to_index
                    ]
                    if missing_keys:
                        self.missing.extend([f"missing shard key: {x}" for x in missing_keys])
                        if require_all_precisions:
                            continue
                else:
                    missing_for_chunk = [str(path) for path in mid_paths.values() if not path.exists()]
                    if require_all_precisions and missing_for_chunk:
                        self.missing.extend(missing_for_chunk)
                        continue

                for p in PRECISIONS:
                    mid_path = mid_paths[p]
                    if self.shard_cache_root is None and not mid_path.exists():
                        self.missing.append(str(mid_path))
                        continue

                    self.samples.append(
                        SelectorSample(
                            case_id=case_id,
                            chunk_idx=chunk_idx,
                            current_vlm_precision=p,
                            current_vlm_precision_id=PRECISION_TO_ID[p],
                            mid_path=mid_path,
                            state_path=state_path,
                            action_label_id=action_label,
                            next_vlm_label_id=next_label,
                        )
                    )

                    self.action_label_counts[action_label] += 1
                    if next_label >= 0:
                        self.next_vlm_label_counts[next_label] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def _load_hidden_from_shard(self, s: SelectorSample) -> tuple[np.ndarray, np.ndarray]:
        assert self._shard_key_to_index is not None
        assert self._shard_lengths is not None
        key = make_shard_key(s.case_id, s.chunk_idx, s.current_vlm_precision)
        row = int(self._shard_key_to_index[key])
        valid_len = int(self._shard_lengths[row])
        hidden_mem = self._get_shard_hidden()
        hidden = np.asarray(hidden_mem[row, :valid_len, :], dtype=np.float16)
        mask = np.ones((valid_len,), dtype=np.bool_)
        return hidden, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[int(idx)]

        st = np.load(s.state_path, allow_pickle=False)

        if self.shard_cache_root is not None:
            hidden, mask = self._load_hidden_from_shard(s)
        elif self.cache_root is not None:
            cache_path = (
                self.cache_root
                / "cases"
                / s.case_id
                / "mid_valid"
                / s.current_vlm_precision
                / f"chunk{s.chunk_idx:04d}.npy"
            )
            if not cache_path.exists():
                raise FileNotFoundError(f"missing cache: {cache_path}")

            hidden = np.load(cache_path, mmap_mode=None)
            hidden = np.asarray(hidden, dtype=np.float16)
            mask = np.ones((hidden.shape[0],), dtype=np.bool_)
        else:
            md = np.load(s.mid_path, allow_pickle=False)
            hidden = np.asarray(md["vlm_prefix_last_hidden"], dtype=np.float16)
            mask = np.asarray(md["prefix_pad_mask"]).astype(np.bool_)
            hidden = hidden[mask]
            mask = np.ones((hidden.shape[0],), dtype=np.bool_)

        if hidden.ndim != 2:
            raise ValueError(f"hidden must be [T,D], got {hidden.shape} from {s.mid_path}")
        if mask.ndim != 1:
            raise ValueError(f"mask must be [T], got {mask.shape} from {s.mid_path}")
        if hidden.shape[0] != mask.shape[0]:
            raise ValueError(
                f"hidden/mask mismatch: {hidden.shape}, {mask.shape}, file={s.mid_path}"
            )

        state = np.asarray(st["observation_state"], dtype=np.float32)
        if state.ndim != 1:
            state = state.reshape(-1).astype(np.float32)

        return {
            "prefix_hidden": torch.from_numpy(hidden),
            "prefix_pad_mask": torch.from_numpy(mask),
            "state": torch.from_numpy(state),
            "current_vlm_precision_id": torch.tensor(
                s.current_vlm_precision_id, dtype=torch.long
            ),
            "action_label": torch.tensor(s.action_label_id, dtype=torch.long),
            "next_vlm_label": torch.tensor(s.next_vlm_label_id, dtype=torch.long),
            "case_id": s.case_id,
            "chunk_idx": s.chunk_idx,
            "current_vlm_precision": s.current_vlm_precision,
            "mid_path": str(s.mid_path),
            "state_path": str(s.state_path),
        }


def selector_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch")

    max_t = max(int(x["prefix_hidden"].shape[0]) for x in batch)
    hidden_dim = int(batch[0]["prefix_hidden"].shape[1])
    bsz = len(batch)

    hidden_dtype = batch[0]["prefix_hidden"].dtype
    prefix_hidden = torch.zeros(bsz, max_t, hidden_dim, dtype=hidden_dtype)
    prefix_pad_mask = torch.zeros(bsz, max_t, dtype=torch.bool)

    states = []
    current_ids = []
    action_labels = []
    next_labels = []
    case_ids = []
    chunk_idxs = []
    current_precisions = []
    mid_paths = []
    state_paths = []

    for i, x in enumerate(batch):
        h = x["prefix_hidden"]
        m = x["prefix_pad_mask"].bool()
        t = int(h.shape[0])

        prefix_hidden[i, :t] = h
        prefix_pad_mask[i, :t] = m

        states.append(x["state"].float())
        current_ids.append(x["current_vlm_precision_id"])
        action_labels.append(x["action_label"])
        next_labels.append(x["next_vlm_label"])

        case_ids.append(x["case_id"])
        chunk_idxs.append(int(x["chunk_idx"]))
        current_precisions.append(x["current_vlm_precision"])
        mid_paths.append(x["mid_path"])
        state_paths.append(x["state_path"])

    return {
        "prefix_hidden": prefix_hidden,
        "prefix_pad_mask": prefix_pad_mask,
        "state": torch.stack(states, dim=0),
        "current_vlm_precision_id": torch.stack(current_ids, dim=0),
        "action_label": torch.stack(action_labels, dim=0),
        "next_vlm_label": torch.stack(next_labels, dim=0),
        "case_id": case_ids,
        "chunk_idx": chunk_idxs,
        "current_vlm_precision": current_precisions,
        "mid_path": mid_paths,
        "state_path": state_paths,
    }


def split_cases_from_task_json(
    task_json: str | Path,
    *,
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[str]]:
    import random

    items = load_json(task_json)
    case_ids = sorted({str(x["case_id"]) for x in items if "case_id" in x}, key=case_sort_key)

    if not case_ids:
        raise RuntimeError(f"No case_id found in {task_json}")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1")

    rng = random.Random(int(seed))
    rng.shuffle(case_ids)

    n = len(case_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train = case_ids[:n_train]
    val = case_ids[n_train:n_train + n_val]
    test = case_ids[n_train + n_val:]

    if n >= 3:
        if not val:
            val = train[-1:]
            train = train[:-1]
        if not test:
            test = val[-1:]
            val = val[:-1]

    return {"train": train, "val": val, "test": test}


def make_class_weights(counts: np.ndarray, *, eps: float = 1e-6) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, eps)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
