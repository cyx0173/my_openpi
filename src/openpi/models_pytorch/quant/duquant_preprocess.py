from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

DEFAULT_PACK_DIR = os.environ.get("OPENPI_DUQUANT_PACKDIR", "./duquant_packed")


def qmax(bits: int) -> int:
    return (1 << (int(bits) - 1)) - 1


def _block_count(dim: int, block_size: int) -> int:
    return (int(dim) + int(block_size) - 1) // int(block_size)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))
    return name.replace("..", ".")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pack_path(layer_name: str, pack_dir: Optional[str]) -> str:
    path = pack_dir or DEFAULT_PACK_DIR
    _ensure_dir(path)
    return os.path.join(path, f"{sanitize_name(layer_name)}.npz")


@dataclass
class PackResult:
    """Reference DuQuant transform package for one Linear layer.

    This file deliberately contains no packed INT4 kernel state.  It stores only
    the mathematically lossless pre-quantization transforms:

      x' = x[:, perm] @ R_in
      W' = W[:, perm] @ R_in
      optional row rotation W'' = R_out @ W', followed by y @ R_out restore

    The quantized Linear module then fake-quantizes W'' and x' in a slow but
    easy-to-verify reference path.
    """

    R_in_blocks: Optional[Dict[int, np.ndarray]]
    perm: Optional[np.ndarray]
    R_out_blocks: Optional[Dict[int, np.ndarray]]
    meta: Dict[str, Any]


def fake_quantize_sym(
    x: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    *,
    label: Optional[str] = None,
) -> torch.Tensor:
    """Symmetric fake quantization, always using FP32 arithmetic internally."""
    del label
    bits = int(bits)
    if bits <= 0 or bits >= 16:
        return x

    max_q = qmax(bits)
    x_fp32 = x.to(torch.float32)
    s = torch.clamp(scale.to(device=x.device, dtype=torch.float32), min=1e-12)
    q = torch.round(x_fp32 / s)
    q = torch.clamp(q, -max_q - 1, max_q)
    return (q * s).to(dtype=x.dtype)


def fake_quantize_activation(
    x: torch.Tensor,
    bits: int,
    *,
    lac: float = 1.0,
    group_size: int = 0,
) -> torch.Tensor:
    """DuQuant-style runtime activation fake quantization after transform.

    DuQuant still uses dynamic activation quantization during forward; the
    critical calibration-dependent part is the preceding permutation/rotation
    and smoothing/clipping configuration.  This function supports per-token and
    per-token-group scales with LAC-style clipping.
    """
    bits = int(bits)
    if bits <= 0 or bits >= 16:
        return x

    if not (0.0 < float(lac) <= 1.0):
        raise ValueError(f"lac/activation clip ratio must be in (0, 1], got {lac}")

    max_q = qmax(bits)
    x_fp32 = x.to(torch.float32)

    if int(group_size) <= 0:
        p = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
        clip = torch.clamp(p * float(lac), min=1e-12)
        q = torch.round(torch.clamp(x_fp32, -clip, clip) / (clip / max_q))
        q = torch.clamp(q, -max_q - 1, max_q)
        return (q * (clip / max_q)).to(dtype=x.dtype)

    g = int(group_size)
    c = int(x_fp32.shape[-1])
    pad = (g - c % g) % g
    if pad:
        x_pad = torch.nn.functional.pad(x_fp32, (0, pad))
    else:
        x_pad = x_fp32

    new_c = int(x_pad.shape[-1])
    xg = x_pad.reshape(*x_pad.shape[:-1], new_c // g, g)
    p = torch.amax(torch.abs(xg), dim=-1, keepdim=True)
    clip = torch.clamp(p * float(lac), min=1e-12)
    scale = clip / max_q
    q = torch.round(torch.clamp(xg, -clip, clip) / scale)
    q = torch.clamp(q, -max_q - 1, max_q)
    y = (q * scale).reshape(*x_pad.shape)
    if pad:
        y = y[..., :c]
    return y.to(dtype=x.dtype)


def compute_mse_scales(W: torch.Tensor, bits: int, *, swc: float = 1.0) -> torch.Tensor:
    """Per-output-channel symmetric quant scales with a small MSE grid search."""
    bits = int(bits)
    if bits <= 0 or bits >= 16:
        return torch.ones(W.shape[0], device=W.device, dtype=W.dtype)

    if not (0.0 < float(swc) <= 1.0):
        raise ValueError(f"swc/weight clip ratio must be in (0, 1], got {swc}")

    max_q = qmax(bits)
    W32 = W.to(torch.float32)
    max_abs = torch.amax(torch.abs(W32), dim=1).clamp_min(1e-8) * float(swc)
    base = max_abs / max_q

    # This is intentionally small and deterministic; it is a reference path.
    multipliers = torch.tensor([0.50, 0.75, 1.00, 1.25, 1.50], device=W.device, dtype=torch.float32)
    candidates = base[:, None] * multipliers[None, :]
    W_row = W32[:, None, :]
    S = candidates[:, :, None].clamp_min(1e-12)
    q = torch.round(W_row / S).clamp(-max_q - 1, max_q)
    rec = q * S
    mse = torch.mean((rec - W_row) ** 2, dim=2)
    idx = torch.argmin(mse, dim=1)
    return candidates[torch.arange(W.shape[0], device=W.device), idx].to(dtype=W.dtype)


def _is_power_of_two(x: int) -> bool:
    x = int(x)
    return x > 0 and (x & (x - 1)) == 0


def _hadamard(n: int) -> np.ndarray:
    """Normalized Sylvester Hadamard matrix for power-of-two n."""
    if not _is_power_of_two(n):
        raise ValueError(f"Hadamard size must be power of two, got {n}")
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(float(n))


def _deterministic_orthogonal(n: int, *, seed: int) -> np.ndarray:
    if n <= 1:
        return np.eye(n, dtype=np.float64)
    if _is_power_of_two(n):
        return _hadamard(n)
    rng = np.random.default_rng(int(seed))
    A = rng.standard_normal((n, n)).astype(np.float64)
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    return Q * signs[None, :]


def zigzag_permutation(score: np.ndarray) -> np.ndarray:
    """Spread high-outlier channels across the hidden dimension."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.argsort(-score)
    out = np.empty_like(order)
    left, right = 0, len(order) - 1
    for i, idx in enumerate(order):
        if i % 2 == 0:
            out[left] = idx
            left += 1
        else:
            out[right] = idx
            right -= 1
    return out.astype(np.int64)


def _hash_array_head(x: Optional[np.ndarray]) -> str:
    if x is None:
        return "none"
    import hashlib

    a = np.asarray(x, dtype=np.float32).reshape(-1)
    head = a[: min(a.size, 4096)]
    return hashlib.sha1(head.tobytes()).hexdigest()[:16]


def _calib_score(
    *,
    in_features: int,
    act_absmax: Optional[torch.Tensor | np.ndarray],
    act_std: Optional[torch.Tensor | np.ndarray] = None,
    require_calib: bool,
) -> np.ndarray:
    if act_absmax is None:
        if require_calib:
            raise RuntimeError(
                "Missing DuQuant calibration for this layer. Run activation calibration first "
                "and set OPENPI_DUQUANT_CALIB_PATH to the generated .pt file."
            )
        return np.ones((in_features,), dtype=np.float64)

    if isinstance(act_absmax, torch.Tensor):
        absmax = act_absmax.detach().to(torch.float32).cpu().numpy()
    else:
        absmax = np.asarray(act_absmax, dtype=np.float32)
    absmax = absmax.reshape(-1).astype(np.float64)
    if absmax.size != in_features:
        raise ValueError(f"activation stat size mismatch: expected {in_features}, got {absmax.size}")

    score = absmax.copy()
    if act_std is not None:
        if isinstance(act_std, torch.Tensor):
            std = act_std.detach().to(torch.float32).cpu().numpy()
        else:
            std = np.asarray(act_std, dtype=np.float32)
        std = std.reshape(-1).astype(np.float64)
        if std.size == in_features:
            # Blend massive-outlier prior and normal dynamic spread.
            score = 0.75 * score + 0.25 * std

    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    return np.maximum(score, 1e-12)


def pack_weight(
    W: torch.Tensor,
    *,
    layer_name: str,
    act_absmax: Optional[torch.Tensor | np.ndarray],
    act_std: Optional[torch.Tensor | np.ndarray] = None,
    block_size: int = 128,
    block_out_size: Optional[int] = None,
    enable_permute: bool = True,
    require_calib: bool = True,
    alpha: float = 0.6,
    permutation_times: int = 1,
    row_rot_mode: str = "restore",
) -> PackResult:
    """Build a calibration-driven DuQuant reference transform.

    Important difference from the old local code: the permutation is driven by
    activation calibration statistics rather than by weight-only energy.
    """
    del alpha  # Reserved for future SmoothQuant-style folding at norm/linear boundaries.

    W_np = W.detach().to(dtype=torch.float32, device="cpu").numpy()
    out_features, in_features = W_np.shape
    block_size = int(block_size)
    if block_out_size is None:
        block_out_size = block_size
    block_out_size = int(block_out_size)

    score = _calib_score(
        in_features=in_features,
        act_absmax=act_absmax,
        act_std=act_std,
        require_calib=require_calib,
    )

    perm: Optional[np.ndarray]
    if enable_permute:
        perm = np.arange(in_features, dtype=np.int64)
        cur_score = score.copy()
        for _ in range(max(int(permutation_times), 1)):
            p = zigzag_permutation(cur_score)
            perm = perm[p]
            cur_score = cur_score[p]
    else:
        perm = None

    # Input rotations: outlier-aware by first spreading channels via calibrated
    # zigzag permutation, then using full-mixing orthogonal blocks.
    n_in_blocks = _block_count(in_features, block_size)
    R_in_blocks: Dict[int, np.ndarray] = {}
    for b in range(n_in_blocks):
        s = b * block_size
        e = min((b + 1) * block_size, in_features)
        width = e - s
        R_in_blocks[b] = _deterministic_orthogonal(width, seed=17_003 + b).astype(np.float32)

    R_out_blocks: Optional[Dict[int, np.ndarray]] = None
    if str(row_rot_mode).lower() == "restore":
        n_out_blocks = _block_count(out_features, block_out_size)
        R_out_blocks = {}
        for b in range(n_out_blocks):
            s = b * block_out_size
            e = min((b + 1) * block_out_size, out_features)
            width = e - s
            R_out_blocks[b] = _deterministic_orthogonal(width, seed=29_011 + b).astype(np.float32)

    meta = {
        "format": "openpi_duquant_reference_v1",
        "layer_name": str(layer_name),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "block_size": int(block_size),
        "block_out_size": int(block_out_size),
        "enable_permute": bool(enable_permute),
        "require_calib": bool(require_calib),
        "permutation_times": int(permutation_times),
        "row_rot_mode": str(row_rot_mode),
        "calib_score_hash": _hash_array_head(score),
    }
    return PackResult(R_in_blocks=R_in_blocks, perm=perm, R_out_blocks=R_out_blocks, meta=meta)


def save_pack(layer_name: str, pack: PackResult, pack_dir: Optional[str]) -> None:
    path = _pack_path(layer_name, pack_dir)
    data: Dict[str, Any] = {"meta": json.dumps(pack.meta, sort_keys=True)}
    if pack.perm is not None:
        data["perm"] = pack.perm.astype(np.int64)
    if pack.R_in_blocks:
        data["R_in_blocks"] = np.array(sorted(pack.R_in_blocks.keys()), dtype=np.int64)
        for b, R in pack.R_in_blocks.items():
            data[f"Rin_{int(b)}"] = R.astype(np.float32)
    if pack.R_out_blocks:
        data["R_out_blocks"] = np.array(sorted(pack.R_out_blocks.keys()), dtype=np.int64)
        for b, R in pack.R_out_blocks.items():
            data[f"Rout_{int(b)}"] = R.astype(np.float32)
    np.savez(path, **data)


def _meta_matches(saved: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    keys = [
        "format",
        "layer_name",
        "in_features",
        "out_features",
        "block_size",
        "block_out_size",
        "enable_permute",
        "require_calib",
        "permutation_times",
        "row_rot_mode",
        "calib_score_hash",
    ]
    return all(saved.get(k) == expected.get(k) for k in keys)


def load_pack(layer_name: str, pack_dir: Optional[str], *, expected_meta: Optional[Dict[str, Any]] = None) -> Optional[PackResult]:
    path = _pack_path(layer_name, pack_dir)
    if not os.path.exists(path):
        return None

    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(f["meta"].tolist()) if "meta" in f.files else {}
        if expected_meta is not None and not _meta_matches(meta, expected_meta):
            return None
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

    return PackResult(R_in_blocks=R_in_blocks, perm=perm, R_out_blocks=R_out_blocks, meta=meta)


def apply_input_transform_optimized(
    x: torch.Tensor,
    pack: PackResult,
    perm_cache: Optional[torch.Tensor],
    R_in_cache: Dict[int, torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    if perm_cache is None and not R_in_cache:
        return x

    in_features = int(x.shape[-1])
    if perm_cache is not None:
        x = x.index_select(dim=-1, index=perm_cache.to(device=x.device))

    if not R_in_cache:
        return x

    orig_shape = x.shape
    x_view = x.reshape(-1, in_features)
    x_out = x_view.clone()
    n_blocks = _block_count(in_features, block_size)
    for b in range(n_blocks):
        if b not in R_in_cache:
            continue
        s = b * int(block_size)
        e = min((b + 1) * int(block_size), in_features)
        R = R_in_cache[b][: e - s, : e - s].to(dtype=x_view.dtype, device=x_view.device)
        x_out[:, s:e] = x_view[:, s:e] @ R
    return x_out.reshape(orig_shape)


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
    swc: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if perm_cache is not None:
        W_t = W.index_select(dim=1, index=perm_cache.to(W.device)).clone()
    else:
        W_t = W.clone()

    in_features = int(W_t.shape[1])
    if R_in_cache:
        n_blocks = _block_count(in_features, block_size)
        for b in range(n_blocks):
            if b not in R_in_cache:
                continue
            s = b * int(block_size)
            e = min((b + 1) * int(block_size), in_features)
            R = R_in_cache[b][: e - s, : e - s].to(dtype=W_t.dtype, device=W_t.device)
            W_t[:, s:e] = W_t[:, s:e] @ R

    if apply_row_rot and R_out_cache:
        out_features = int(W_t.shape[0])
        n_blocks = _block_count(out_features, block_out_size)
        for b in range(n_blocks):
            if b not in R_out_cache:
                continue
            s = b * int(block_out_size)
            e = min((b + 1) * int(block_out_size), out_features)
            R = R_out_cache[b][: e - s, : e - s].to(dtype=W_t.dtype, device=W_t.device)
            W_t[s:e, :] = R @ W_t[s:e, :]

    scales = compute_mse_scales(W_t, weight_bits, swc=swc)
    return W_t, scales


def apply_output_restore_optimized(
    y: torch.Tensor,
    pack: PackResult,
    R_out_cache: Dict[int, torch.Tensor],
    block_out_size: int,
) -> torch.Tensor:
    del pack
    if not R_out_cache:
        return y
    out_features = int(y.shape[-1])
    orig_shape = y.shape
    y_view = y.reshape(-1, out_features)
    y_out = y_view.clone()
    n_blocks = _block_count(out_features, block_out_size)
    for b in range(n_blocks):
        if b not in R_out_cache:
            continue
        s = b * int(block_out_size)
        e = min((b + 1) * int(block_out_size), out_features)
        R = R_out_cache[b][: e - s, : e - s].to(dtype=y_view.dtype, device=y_view.device)
        y_out[:, s:e] = y_view[:, s:e] @ R
    return y_out.reshape(orig_shape)


def apply_bias_row_rot_optimized(
    bias: torch.Tensor,
    pack: PackResult,
    R_out_cache: Dict[int, torch.Tensor],
    block_out_size: int,
) -> torch.Tensor:
    del pack
    if not R_out_cache:
        return bias
    out_features = int(bias.shape[-1])
    bias_out = bias.clone()
    n_blocks = _block_count(out_features, block_out_size)
    for b in range(n_blocks):
        if b not in R_out_cache:
            continue
        s = b * int(block_out_size)
        e = min((b + 1) * int(block_out_size), out_features)
        R = R_out_cache[b][: e - s, : e - s].to(dtype=bias.dtype, device=bias.device)
        bias_out[s:e] = R @ bias[s:e]
    return bias_out
