from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


DEFAULT_PACK_DIR = os.environ.get("OPENPI_DUQUANT_PACKDIR", "./duquant_packed")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name.replace("..", ".")


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


class _DuQuantProfiler:
    def __init__(self) -> None:
        self.enabled = os.environ.get("OPENPI_DUQUANT_PROFILE", "0") not in ("0", "false", "False")
        self.sync_cuda = os.environ.get("OPENPI_DUQUANT_PROFILE_SYNC", "1") not in ("0", "false", "False")
        self._stats = defaultdict(lambda: {"time": 0.0, "count": 0.0, "elements": 0.0, "bytes": 0.0})
        if self.enabled:
            import atexit
            atexit.register(self.report)

    def record(self, label: str, tensor: torch.Tensor, scale: torch.Tensor, bits: int, fn):
        if not self.enabled:
            return fn()

        devices = []
        if self.sync_cuda:
            if tensor.is_cuda:
                devices.append(tensor.device)
            if isinstance(scale, torch.Tensor) and scale.is_cuda:
                devices.append(scale.device)

        for d in devices:
            torch.cuda.synchronize(d)
        start = time.perf_counter()
        out = fn()
        for d in devices:
            torch.cuda.synchronize(d)
        elapsed = time.perf_counter() - start

        st = self._stats[label]
        st["time"] += elapsed
        st["count"] += 1
        st["elements"] += tensor.numel()
        st["bytes"] += tensor.numel() * tensor.element_size()
        if isinstance(scale, torch.Tensor):
            st["bytes"] += scale.numel() * scale.element_size()
        st["bits"] = float(bits)
        return out

    def report(self) -> None:
        if not self.enabled:
            return
        print("=" * 100)
        print("[OPENPI-DUQUANT][PROFILE] fake quantization summary")
        for label, st in sorted(self._stats.items()):
            calls = int(st["count"])
            total_ms = st["time"] * 1000
            avg_ms = total_ms / max(calls, 1)
            print(f"{label:<28} calls={calls:<8d} total_ms={total_ms:10.2f} avg_ms={avg_ms:8.3f}")
        print("=" * 100)


_DUQUANT_PROFILER = _DuQuantProfiler()


def fake_quantize_sym(
    x: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    *,
    label: Optional[str] = None,
) -> torch.Tensor:
    """
    Symmetric fake quant:
        x -> round(x / scale) -> clamp -> dequantize back to float
    """
    if bits <= 0 or bits >= 16:
        return x

    def _impl() -> torch.Tensor:
        max_q = qmax(bits)
        scale_safe = torch.clamp(scale.to(device=x.device, dtype=x.dtype), min=1e-8)
        q = torch.round(x / scale_safe)
        q = torch.clamp(q, -max_q - 1, max_q)
        return q * scale_safe

    return _DUQUANT_PROFILER.record(label or "fake_quantize_sym", x, scale, bits, _impl)


@dataclass
class PackResult:
    R_in_blocks: Optional[Dict[int, np.ndarray]]
    perm: Optional[np.ndarray]
    R_out_blocks: Optional[Dict[int, np.ndarray]]
    weight_scale: np.ndarray
    meta: Dict[str, Any]


class PercentileCalibrator:
    """
    Collects per-channel activation percentile over several batches.
    """

    def __init__(self, percentile: float = 99.9, max_batches: int = 32) -> None:
        self.percentile = percentile
        self.max_batches = max_batches
        self._seen = 0
        self._p_running: Optional[torch.Tensor] = None

    def observe(self, x: torch.Tensor) -> None:
        if self._seen >= self.max_batches:
            return

        x_abs = torch.abs(x.detach().to(torch.float32))
        c = x_abs.shape[-1]
        x2d = x_abs.reshape(-1, c)
        q = torch.quantile(x2d, self.percentile / 100.0, dim=0)
        q = torch.clamp(q, min=1e-6).cpu()

        if self._p_running is None:
            self._p_running = q
        else:
            self._p_running = torch.maximum(self._p_running, q)

        self._seen += 1

    def is_full(self) -> bool:
        return self._seen >= self.max_batches

    def finalize(self) -> torch.Tensor:
        if self._p_running is None:
            return torch.tensor([1.0], dtype=torch.float32)
        return self._p_running


def _block_count(dim: int, block_size: int) -> int:
    return (dim + block_size - 1) // block_size


def compute_block_rotation(W_block: np.ndarray) -> np.ndarray:
    """
    Compute an orthonormal rotation matrix for one input/output block.
    """
    x = W_block.T
    b = x.shape[0]
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_t = torch.from_numpy(x.astype(np.float32)).to(dev)
        u, _, _ = torch.linalg.svd(x_t, full_matrices=True)
        u_np = u.cpu().numpy().astype(np.float64)
    except Exception:
        try:
            u_np, _, _ = np.linalg.svd(x.astype(np.float64, copy=False), full_matrices=True)
        except np.linalg.LinAlgError:
            u_np = np.eye(b, dtype=np.float64)

    if u_np.shape[1] < b:
        pad = np.zeros((b, b - u_np.shape[1]), dtype=u_np.dtype)
        u_np = np.concatenate([u_np, pad], axis=1)
    return u_np[:, :b]


def zigzag_permutation(energy: np.ndarray) -> np.ndarray:
    order = np.argsort(-energy)
    left, right = 0, len(order) - 1
    perm = []
    toggle = True
    while left <= right:
        if toggle:
            perm.append(order[left])
            left += 1
        else:
            perm.append(order[right])
            right -= 1
        toggle = not toggle
    return np.array(perm, dtype=np.int64)


def pack_weight(
    W: torch.Tensor,
    *,
    block_size: int = 64,
    block_out_size: Optional[int] = None,
    enable_permute: bool = True,
    lambda_smooth: float = 0.15,
) -> PackResult:
    """
    DuQuant-style preprocessing:
      1. input-channel energy smoothing
      2. zigzag permutation
      3. input block rotation
      4. output row block rotation
    """
    W_np = W.detach().to(dtype=torch.float32, device="cpu").numpy()
    out_features, in_features = W_np.shape

    channel_energy = np.mean(W_np ** 2, axis=0)
    if lambda_smooth > 0:
        mean_e = float(channel_energy.mean())
        channel_energy = (1.0 - lambda_smooth) * channel_energy + lambda_smooth * mean_e

    perm = zigzag_permutation(channel_energy) if enable_permute else None

    R_in_blocks: Dict[int, np.ndarray] = {}
    n_in_blocks = _block_count(in_features, block_size)
    for b in range(n_in_blocks):
        s = b * block_size
        e = min((b + 1) * block_size, in_features)
        cols = np.arange(s, e)
        if perm is not None:
            cols = perm[cols]
        W_block = W_np[:, cols]
        R_in_blocks[b] = compute_block_rotation(W_block)

    if block_out_size is None:
        block_out_size = block_size

    R_out_blocks: Dict[int, np.ndarray] = {}
    n_out_blocks = _block_count(out_features, block_out_size)
    for b in range(n_out_blocks):
        s = b * block_out_size
        e = min((b + 1) * block_out_size, out_features)
        W_rows = W_np[s:e, :]
        R_out_blocks[b] = compute_block_rotation(W_rows.T)

    max_abs = np.maximum(np.max(np.abs(W_np), axis=1), 1e-8)
    weight_scale = (max_abs / qmax(4)).astype(np.float32)

    meta = {
        "in_features": int(in_features),
        "out_features": int(out_features),
        "block_size": int(block_size),
        "block_out_size": int(block_out_size),
        "enable_permute": bool(enable_permute),
        "lambda_smooth": float(lambda_smooth),
    }

    return PackResult(
        R_in_blocks=R_in_blocks or None,
        perm=perm,
        R_out_blocks=R_out_blocks or None,
        weight_scale=weight_scale,
        meta=meta,
    )


def _pack_path(layer_name: str, pack_dir: Optional[str]) -> str:
    path = pack_dir or DEFAULT_PACK_DIR
    _ensure_dir(path)
    return os.path.join(path, f"{sanitize_name(layer_name)}.npz")


def save_pack(layer_name: str, pack: PackResult, pack_dir: Optional[str]) -> None:
    path = _pack_path(layer_name, pack_dir)
    data: Dict[str, Any] = {
        "weight_scale": pack.weight_scale.astype(np.float32),
        "meta": json.dumps(pack.meta),
    }
    if pack.perm is not None:
        data["perm"] = pack.perm.astype(np.int64)
    if pack.R_in_blocks:
        data["R_in_blocks"] = np.array(sorted(pack.R_in_blocks.keys()), dtype=np.int64)
        for b, R in pack.R_in_blocks.items():
            data[f"Rin_{b}"] = R.astype(np.float32)
    if pack.R_out_blocks:
        data["R_out_blocks"] = np.array(sorted(pack.R_out_blocks.keys()), dtype=np.int64)
        for b, R in pack.R_out_blocks.items():
            data[f"Rout_{b}"] = R.astype(np.float32)
    np.savez(path, **data)


def load_pack(layer_name: str, pack_dir: Optional[str]) -> Optional[PackResult]:
    path = _pack_path(layer_name, pack_dir)
    if not os.path.exists(path):
        return None

    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(f["meta"].tolist()) if "meta" in f.files else {}
        perm = f["perm"] if "perm" in f.files else None

        R_in_blocks = None
        if "R_in_blocks" in f.files:
            R_in_blocks = {}
            for b in f["R_in_blocks"]:
                R_in_blocks[int(b)] = f[f"Rin_{int(b)}"]

        R_out_blocks = None
        if "R_out_blocks" in f.files:
            R_out_blocks = {}
            for b in f["R_out_blocks"]:
                R_out_blocks[int(b)] = f[f"Rout_{int(b)}"]

        return PackResult(
            R_in_blocks=R_in_blocks,
            perm=perm,
            R_out_blocks=R_out_blocks,
            weight_scale=f["weight_scale"],
            meta=meta,
        )


def compute_mse_scales(W: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0 or bits >= 16:
        return torch.ones(W.shape[0], device=W.device, dtype=W.dtype)

    max_abs = torch.amax(torch.abs(W), dim=1).clamp_min(1e-8)
    base = max_abs / qmax(bits)
    alphas = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5], device=W.device, dtype=W.dtype)
    candidates = base[:, None] / alphas[None, :]

    W_row = W[:, None, :]
    S = candidates[:, :, None]
    q = torch.round(W_row / S).clamp(-qmax(bits) - 1, qmax(bits))
    rec = q * S
    mse = torch.mean((rec - W_row) ** 2, dim=2)
    idx = torch.argmin(mse, dim=1)
    return candidates[torch.arange(W.shape[0], device=W.device), idx]


def apply_input_transform_optimized(
    x: torch.Tensor,
    pack: PackResult,
    perm_cache: Optional[torch.Tensor],
    R_in_cache: Dict[int, torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    if perm_cache is None and not R_in_cache:
        return x

    in_features = x.shape[-1]

    if perm_cache is not None:
        x = x.index_select(dim=-1, index=perm_cache.to(x.device))

    if R_in_cache:
        orig_shape = x.shape
        x_view = x.reshape(-1, in_features)
        x_out = x_view.clone()
        n_blocks = _block_count(in_features, block_size)

        for b in range(n_blocks):
            if b not in R_in_cache:
                continue
            s = b * block_size
            e = min((b + 1) * block_size, in_features)
            R = R_in_cache[b][: e - s, : e - s].to(dtype=x_view.dtype, device=x_view.device)
            x_out[:, s:e] = x_view[:, s:e] @ R

        x = x_out.reshape(orig_shape)

    return x


def transform_weight_for_forward_optimized(
    W: torch.Tensor,
    pack: PackResult,
    *,
    weight_bits: int,
    apply_row_rot: bool,
    perm_cache: Optional[torch.Tensor],
    R_in_cache: Dict[int, torch.Tensor],
    R_out_cache: Dict[int, torch.Tensor],
    block_size: int,
    block_out_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if perm_cache is not None:
        W_t = W.index_select(dim=1, index=perm_cache.to(W.device)).clone()
    else:
        W_t = W.clone()

    in_features = W_t.shape[1]

    if R_in_cache:
        n_blocks = _block_count(in_features, block_size)
        for b in range(n_blocks):
            if b not in R_in_cache:
                continue
            s = b * block_size
            e = min((b + 1) * block_size, in_features)
            R = R_in_cache[b][: e - s, : e - s].to(dtype=W_t.dtype, device=W_t.device)
            W_t[:, s:e] = W_t[:, s:e] @ R

    if apply_row_rot and R_out_cache:
        out_features = W_t.shape[0]
        n_blocks = _block_count(out_features, block_out_size)
        for b in range(n_blocks):
            if b not in R_out_cache:
                continue
            s = b * block_out_size
            e = min((b + 1) * block_out_size, out_features)
            R = R_out_cache[b][: e - s, : e - s].to(dtype=W_t.dtype, device=W_t.device)
            W_t[s:e, :] = R @ W_t[s:e, :]

    scales = compute_mse_scales(W_t, weight_bits)
    return W_t, scales


def apply_output_restore_optimized(
    y: torch.Tensor,
    pack: PackResult,
    R_out_cache: Dict[int, torch.Tensor],
    block_out_size: int,
) -> torch.Tensor:
    if not R_out_cache:
        return y

    out_features = y.shape[-1]
    orig_shape = y.shape
    y_view = y.reshape(-1, out_features)
    y_out = y_view.clone()

    n_blocks = _block_count(out_features, block_out_size)
    for b in range(n_blocks):
        if b not in R_out_cache:
            continue
        s = b * block_out_size
        e = min((b + 1) * block_out_size, out_features)
        R = R_out_cache[b][: e - s, : e - s].to(dtype=y_view.dtype, device=y_view.device)
        y_out[:, s:e] = y_view[:, s:e] @ R

    return y_out.reshape(orig_shape)


def apply_bias_row_rot_optimized(
    bias: torch.Tensor,
    pack: PackResult,
    R_out_cache: Dict[int, torch.Tensor],
    block_out_size: int,
) -> torch.Tensor:
    if not R_out_cache:
        return bias

    out_features = bias.shape[-1]
    bias_out = bias.clone()
    n_blocks = _block_count(out_features, block_out_size)

    for b in range(n_blocks):
        if b not in R_out_cache:
            continue
        s = b * block_out_size
        e = min((b + 1) * block_out_size, out_features)
        R = R_out_cache[b][: e - s, : e - s].to(dtype=bias.dtype, device=bias.device)
        bias_out[s:e] = R @ bias[s:e]

    return bias_out