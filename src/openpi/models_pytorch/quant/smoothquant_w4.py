
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import hashlib
import os
import pathlib
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

from openpi.models_pytorch.quant.duquant_calibration import load_calibration
from openpi.models_pytorch.quant.duquant_packed_w4 import (
    pack_int4_signed,
    unpack_int4_signed,
    w4_weight_only_linear_triton,
)
from openpi.models_pytorch.quant.duquant_preprocess import (
    compute_mse_scales,
    fake_quantize_activation,
)


DEFAULT_SMOOTHQUANT_CALIB_PATH = (
    "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant/openpi_duquant_calib.pt"
)

DEFAULT_SMOOTHQUANT_INT4_CACHE_DIR = (
    "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/smoothquant/smoothquant_int4_cache"
)

DEFAULT_SMOOTHQUANT_COMMON_EXCLUDE_REGEX = (
    r"lm_head"
    r"|embed_tokens"
    r"|rotary_emb"
    r"|\.norm"
    r"|\.input_layernorm"
    r"|\.post_attention_layernorm"
)

DEFAULT_SMOOTHQUANT_VLM_INCLUDE_REGEX = (
    r"paligemma_with_expert\.paligemma\.model\..*"
)

DEFAULT_SMOOTHQUANT_ACTION_MLP_IO_INCLUDE_REGEX = (
    r"paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$"
    r"|^action_in_proj$"
    r"|^time_mlp_in$"
    r"|^time_mlp_out$"
)

DEFAULT_SMOOTHQUANT_INCLUDE_REGEX = (
    DEFAULT_SMOOTHQUANT_VLM_INCLUDE_REGEX
    + r"|"
    + DEFAULT_SMOOTHQUANT_ACTION_MLP_IO_INCLUDE_REGEX
)

DEFAULT_SMOOTHQUANT_EXCLUDE_REGEX = (
    DEFAULT_SMOOTHQUANT_COMMON_EXCLUDE_REGEX
    + r"|^action_out_proj$"
)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in {"0", "false", "no", "off"}


def _quant_mode_default_abits() -> int:
    mode = os.environ.get("OPENPI_QUANT_MODE", "").strip()
    return {
        "1": 4,    # W4A4
        "2": 8,    # W4A8
        "3": 16,   # W4A16
        "4": 2,    # W4A2, usually not recommended
    }.get(mode, 4)


def _safe_part(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _torch_load(path: pathlib.Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _hash_tensor_head(x: Optional[torch.Tensor], *, max_elems: int = 4096) -> str:
    if x is None:
        return "none"
    t = x.detach().to(torch.float32).cpu().reshape(-1)
    t = t[: min(int(t.numel()), int(max_elems))]
    return hashlib.sha1(t.numpy().astype("float32", copy=False).tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class SmoothQuantW4Config:
    calib_path: str = DEFAULT_SMOOTHQUANT_CALIB_PATH
    int4_cache_dir: str = DEFAULT_SMOOTHQUANT_INT4_CACHE_DIR
    int4_cache_tag: str = "smoothquant"
    include_regex: str = DEFAULT_SMOOTHQUANT_INCLUDE_REGEX
    exclude_regex: str = DEFAULT_SMOOTHQUANT_EXCLUDE_REGEX

    weight_bits: int = 4
    act_bits: int = 4

    alpha: float = 0.5
    lac: float = 1.0
    swc: float = 1.0

    # dynamic_symmetric: old path, uses fake_quantize_activation(...)
    # dynamic_token_affine: per-token affine min/max fake quant
    # dynamic_group_affine: per-token per-group affine min/max fake quant
    act_quant_mode: str = "dynamic_symmetric"
    act_group_size: int = 0
    # triton: fast fused x*inv_s + token affine fake quant.
    # torch: fallback PyTorch implementation.
    act_quant_impl: str = "triton"

    backend: str = "kernel"
    require_calib: bool = True
    int4_cache: bool = True
    scale_clip_min: float = 1.0e-3
    scale_clip_max: float = 1.0e3

    @classmethod
    def from_env(cls) -> "SmoothQuantW4Config":
        """Debug-only env constructor.

        Main QuantVLA path should preferably pass SmoothQuantW4Config directly
        from quantvla.py, so DuQuant env variables cannot silently pollute
        SmoothQuant experiments.

        This function intentionally does NOT read OPENPI_DUQUANT_*.
        """
        base = cls()

        abits_env = os.environ.get("OPENPI_SMOOTHQUANT_ABITS")
        abits = int(abits_env) if abits_env else _quant_mode_default_abits()

        return cls(
            calib_path=os.environ.get("OPENPI_SMOOTHQUANT_CALIB_PATH", base.calib_path),
            int4_cache_dir=os.environ.get("OPENPI_SMOOTHQUANT_INT4_CACHE_DIR", base.int4_cache_dir),
            int4_cache_tag=os.environ.get("OPENPI_SMOOTHQUANT_INT4_CACHE_TAG", base.int4_cache_tag),
            include_regex=os.environ.get("OPENPI_SMOOTHQUANT_INCLUDE", base.include_regex),
            exclude_regex=os.environ.get("OPENPI_SMOOTHQUANT_EXCLUDE", base.exclude_regex),

            weight_bits=int(os.environ.get("OPENPI_SMOOTHQUANT_WBITS", str(base.weight_bits))),
            act_bits=abits,

            alpha=float(os.environ.get("OPENPI_SMOOTHQUANT_ALPHA", str(base.alpha))),
            lac=float(os.environ.get("OPENPI_SMOOTHQUANT_LAC", str(base.lac))),
            swc=float(os.environ.get("OPENPI_SMOOTHQUANT_SWC", str(base.swc))),
            act_quant_mode=os.environ.get(
                "OPENPI_SMOOTHQUANT_ACT_QUANT_MODE",
                base.act_quant_mode,
            ).strip().lower(),
            act_group_size=int(os.environ.get("OPENPI_SMOOTHQUANT_ACT_GROUP_SIZE", str(base.act_group_size))),
            act_quant_impl=os.environ.get(
                "OPENPI_SMOOTHQUANT_ACT_QUANT_IMPL",
                base.act_quant_impl,
            ).strip().lower(),

            backend=os.environ.get("OPENPI_SMOOTHQUANT_BACKEND", base.backend).strip().lower(),
            require_calib=_env_bool("OPENPI_SMOOTHQUANT_REQUIRE_CALIB", "1" if base.require_calib else "0"),
            int4_cache=_env_bool("OPENPI_SMOOTHQUANT_INT4_CACHE", "1" if base.int4_cache else "0"),
            scale_clip_min=float(os.environ.get("OPENPI_SMOOTHQUANT_SCALE_CLIP_MIN", str(base.scale_clip_min))),
            scale_clip_max=float(os.environ.get("OPENPI_SMOOTHQUANT_SCALE_CLIP_MAX", str(base.scale_clip_max))),
        )


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _load_smoothquant_calibration(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    return load_calibration(path)


def _compute_smooth_scale(
    weight: torch.Tensor,
    *,
    act_absmax: torch.Tensor,
    alpha: float,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """Classic SmoothQuant scale: x_s = x / s, W_s = W * s."""
    if act_absmax.numel() != weight.shape[1]:
        raise ValueError(
            f"act_absmax shape mismatch: expected {weight.shape[1]}, got {act_absmax.numel()}"
        )

    a = act_absmax.detach().to(device=weight.device, dtype=torch.float32).reshape(-1).clamp_min(1e-6)
    w = torch.amax(torch.abs(weight.detach().to(torch.float32)), dim=0).reshape(-1).clamp_min(1e-6)

    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"SmoothQuant alpha must be in [0,1], got {alpha}")

    s = torch.pow(a, alpha) / torch.pow(w, 1.0 - alpha)
    s = torch.nan_to_num(s, nan=1.0, posinf=float(clip_max), neginf=float(clip_min))
    s = torch.clamp(s, min=float(clip_min), max=float(clip_max))
    return s.to(dtype=weight.dtype, device=weight.device)


def fake_quantize_activation_affine_dynamic(
    x: torch.Tensor,
    bits: int,
    *,
    group_size: int = 0,
) -> torch.Tensor:
    """Dynamic affine fake quantization for activations.

    This is still runtime dynamic quantization. It does NOT use calibration
    absmax as a fixed activation scale.

    token mode:
      xmin/xmax over the last dimension.

    group mode:
      split the last dimension into groups, then xmin/xmax per group.
    """
    bits = int(bits)
    if bits <= 0 or bits >= 16:
        return x

    qmin = 0.0
    qmax_val = float((1 << bits) - 1)

    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)

    group_size = int(group_size or 0)
    if group_size <= 0:
        xmin = torch.amin(x_fp32, dim=-1, keepdim=True)
        xmax = torch.amax(x_fp32, dim=-1, keepdim=True)

        scale = (xmax - xmin).clamp_min(1e-8) / (qmax_val - qmin)
        zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax_val)

        q = torch.round(x_fp32 / scale + zero_point).clamp(qmin, qmax_val)
        y = (q - zero_point) * scale
        return y.to(dtype=orig_dtype)

    last_dim = int(x_fp32.shape[-1])
    pad = (group_size - last_dim % group_size) % group_size
    if pad:
        x_work = torch.nn.functional.pad(x_fp32, (0, pad))
    else:
        x_work = x_fp32

    new_last = int(x_work.shape[-1])
    num_groups = new_last // group_size
    xg = x_work.reshape(*x_work.shape[:-1], num_groups, group_size)

    xmin = torch.amin(xg, dim=-1, keepdim=True)
    xmax = torch.amax(xg, dim=-1, keepdim=True)

    scale = (xmax - xmin).clamp_min(1e-8) / (qmax_val - qmin)
    zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax_val)

    q = torch.round(xg / scale + zero_point).clamp(qmin, qmax_val)
    y = (q - zero_point) * scale
    y = y.reshape(*x_work.shape)

    if pad:
        y = y[..., :last_dim]

    return y.to(dtype=orig_dtype)



def _next_power_of_2_int(x: int) -> int:
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


if _HAS_TRITON:
    @triton.jit
    def _sq_scale_token_affine_kernel(
        X,
        INV_S,
        Y,
        N: tl.constexpr,
        BITS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(INV_S + offs, mask=mask, other=1.0).to(tl.float32)
        xs = x * inv

        xs_min = tl.where(mask, xs, float("inf"))
        xs_max = tl.where(mask, xs, -float("inf"))

        xmin = tl.min(xs_min, axis=0)
        xmax = tl.max(xs_max, axis=0)

        qmax = (1 << BITS) - 1
        scale = (xmax - xmin) / qmax
        scale = tl.maximum(scale, 1.0e-8)

        # Triton versions in the current OpenPI env may not provide tl.round.
        # Use round-half-up for values that are later clamped to [0, qmax].
        zero_point = ((-xmin / scale) + 0.5).to(tl.int32).to(tl.float32)
        zero_point = tl.minimum(tl.maximum(zero_point, 0.0), qmax + 0.0)

        q = ((xs / scale + zero_point) + 0.5).to(tl.int32).to(tl.float32)
        q = tl.minimum(tl.maximum(q, 0.0), qmax + 0.0)

        y = (q - zero_point) * scale
        tl.store(Y + row * N + offs, y, mask=mask)


def smoothquant_scale_token_affine_triton(
    x: torch.Tensor,
    inv_s: torch.Tensor,
    bits: int,
    in_features: int,
) -> torch.Tensor:
    """Fast fused SmoothQuant input scaling + token affine fake quant.

    Computes:
      x_s = x * inv_s
      affine dynamic fake quant over the last dimension

    This function still outputs dequantized floating activations, so it can be
    used with the existing real packed W4 kernel without changing weight cache.
    """
    bits = int(bits)
    if bits <= 0 or bits >= 16:
        return x * inv_s.to(device=x.device, dtype=x.dtype)

    if not _HAS_TRITON or not x.is_cuda:
        inv = inv_s.to(device=x.device, dtype=x.dtype)
        return fake_quantize_activation_affine_dynamic(x * inv, bits, group_size=0)

    n = int(in_features)
    orig_shape = x.shape

    x2d = x.reshape(-1, n)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()

    inv = inv_s
    if inv.device != x2d.device:
        inv = inv.to(device=x2d.device)
    if inv.dtype != x2d.dtype:
        inv = inv.to(dtype=x2d.dtype)
    inv = inv.contiguous()

    y2d = torch.empty_like(x2d)

    block_n = _next_power_of_2_int(n)
    # Keep a sane upper bound. Current OpenPI linear dimensions are within this.
    if block_n > 32768:
        inv = inv.to(device=x.device, dtype=x.dtype)
        return fake_quantize_activation_affine_dynamic(x * inv, bits, group_size=0)

    _sq_scale_token_affine_kernel[(x2d.shape[0],)](
        x2d,
        inv,
        y2d,
        N=n,
        BITS=bits,
        BLOCK_N=block_n,
        num_warps=8,
    )

    return y2d.reshape(orig_shape)



class SmoothQuantW4Linear(nn.Module):
    """SmoothQuant + real packed W4 Linear.

    Weight is stored as real packed INT4:
      _qweight_packed: uint8 [out_features, ceil(in_features / 2)]
      _w_scales:       per-output-channel dequant scale

    Activation is fake quantized dynamically after SmoothQuant input scaling.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        name: str,
        cfg: SmoothQuantW4Config,
        calib_rec: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__()

        if int(cfg.weight_bits) != 4:
            raise ValueError(f"SmoothQuantW4Linear only supports real W4 weight, got W{cfg.weight_bits}")

        self.name = str(name)
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.cfg = cfg
        self.weight_bits = 4
        self.act_bits = int(cfg.act_bits)
        self.forward_backend = str(cfg.backend).lower()

        weight = base.weight.detach()
        device = weight.device
        dtype = weight.dtype

        act_absmax = None
        if isinstance(calib_rec, dict):
            v = calib_rec.get("absmax")
            if isinstance(v, torch.Tensor):
                act_absmax = v

        if act_absmax is None:
            if cfg.require_calib:
                raise RuntimeError(f"Missing SmoothQuant calibration absmax for layer={name!r}")
            act_absmax = torch.ones((self.in_features,), dtype=torch.float32)

        smooth_s = _compute_smooth_scale(
            weight,
            act_absmax=act_absmax,
            alpha=float(cfg.alpha),
            clip_min=float(cfg.scale_clip_min),
            clip_max=float(cfg.scale_clip_max),
        )

        inv_s = torch.reciprocal(smooth_s.to(torch.float32).clamp_min(1e-12)).to(dtype=dtype, device=device)
        self.register_buffer("_inv_s", inv_s.contiguous(), persistent=True)

        if base.bias is not None:
            self.register_buffer("_bias", base.bias.detach().clone(), persistent=True)
        else:
            self._bias = None

        self.register_buffer(
            "_weight_compat_storage",
            torch.empty(1, dtype=dtype, device=device),
            persistent=False,
        )

        packed_cols = (self.in_features + 1) // 2
        self.register_buffer(
            "_qweight_packed",
            torch.empty((self.out_features, packed_cols), dtype=torch.uint8, device=device),
            persistent=True,
        )
        self.register_buffer(
            "_w_scales",
            torch.empty((self.out_features,), dtype=dtype, device=device),
            persistent=True,
        )

        self._weight_hash = _hash_tensor_head(weight)
        self._calib_hash = _hash_tensor_head(act_absmax)
        self._build_or_load_packed_weight(weight, smooth_s)

    @property
    def weight(self) -> torch.Tensor:
        return self._weight_compat_storage.as_strided((self.out_features, self.in_features), (0, 0))

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self._bias

    def set_a_bits(self, bits: int) -> None:
        self.act_bits = int(bits)

    def set_w_bits(self, bits: int) -> None:
        if int(bits) != 4:
            raise ValueError("SmoothQuantW4Linear only supports fixed real W4 weight.")

    def _cache_prefix(self) -> pathlib.Path:
        cache_dir = pathlib.Path(self.cfg.int4_cache_dir)
        key = {
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight_bits": 4,
            "alpha": float(self.cfg.alpha),
            "swc": float(self.cfg.swc),
            "scale_clip_min": float(self.cfg.scale_clip_min),
            "scale_clip_max": float(self.cfg.scale_clip_max),
            "calib_hash": str(self._calib_hash),
            "weight_hash": str(self._weight_hash),
            "cache_version": 2,
        }
        raw = repr(sorted(key.items())).encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()[:12]
        prefix = (
            f"{_safe_part(self.cfg.int4_cache_tag)}__{_safe_part(self.name)}"
            f"__out{self.out_features}_in{self.in_features}__{digest}"
        )
        return cache_dir / prefix

    def _try_load_int4_cache(self) -> bool:
        if not self.cfg.int4_cache:
            return False

        prefix = self._cache_prefix()
        q_path = pathlib.Path(str(prefix) + "_qweight_packed.pt")
        s_path = pathlib.Path(str(prefix) + "_scales.pt")
        inv_path = pathlib.Path(str(prefix) + "_inv_s.pt")
        meta_path = pathlib.Path(str(prefix) + "_meta.pt")

        if not (q_path.exists() and s_path.exists() and inv_path.exists() and meta_path.exists()):
            return False

        try:
            meta = _torch_load(meta_path, map_location="cpu")
            expected = {
                "name": self.name,
                "in_features": self.in_features,
                "out_features": self.out_features,
                "packed_shape": tuple(self._qweight_packed.shape),
                "scale_shape": tuple(self._w_scales.shape),
                "inv_s_shape": tuple(self._inv_s.shape),
                "weight_bits": 4,
                "alpha": float(self.cfg.alpha),
                "swc": float(self.cfg.swc),
                "calib_hash": str(self._calib_hash),
                "weight_hash": str(self._weight_hash),
                "cache_version": 2,
            }
            for k, v in expected.items():
                if meta.get(k) != v:
                    print(f"[SMOOTHQUANT-W4-CACHE] meta mismatch {self.name}: {k}; rebuilding", flush=True)
                    return False

            q = _torch_load(q_path, map_location="cpu")
            s = _torch_load(s_path, map_location="cpu")
            inv_s = _torch_load(inv_path, map_location="cpu")

            if tuple(q.shape) != tuple(self._qweight_packed.shape) or q.dtype != torch.uint8:
                return False
            if tuple(s.shape) != tuple(self._w_scales.shape):
                return False
            if tuple(inv_s.shape) != tuple(self._inv_s.shape):
                return False

            self._qweight_packed.copy_(q.to(device=self._qweight_packed.device, dtype=torch.uint8))
            self._w_scales.copy_(s.to(device=self._w_scales.device, dtype=self._w_scales.dtype))
            self._inv_s.copy_(inv_s.to(device=self._inv_s.device, dtype=self._inv_s.dtype))
            return True
        except Exception as e:
            print(f"[SMOOTHQUANT-W4-CACHE] failed to load {self.name}: {e}; rebuilding", flush=True)
            return False

    def _save_int4_cache(self) -> None:
        if not self.cfg.int4_cache:
            return

        prefix = self._cache_prefix()
        prefix.parent.mkdir(parents=True, exist_ok=True)

        q_path = pathlib.Path(str(prefix) + "_qweight_packed.pt")
        s_path = pathlib.Path(str(prefix) + "_scales.pt")
        inv_path = pathlib.Path(str(prefix) + "_inv_s.pt")
        meta_path = pathlib.Path(str(prefix) + "_meta.pt")

        meta = {
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "packed_shape": tuple(self._qweight_packed.shape),
            "scale_shape": tuple(self._w_scales.shape),
            "inv_s_shape": tuple(self._inv_s.shape),
            "weight_bits": 4,
            "act_bits": int(self.act_bits),
            "alpha": float(self.cfg.alpha),
            "swc": float(self.cfg.swc),
            "scale_clip_min": float(self.cfg.scale_clip_min),
            "scale_clip_max": float(self.cfg.scale_clip_max),
            "calib_hash": str(self._calib_hash),
            "weight_hash": str(self._weight_hash),
            "cache_version": 2,
        }

        try:
            torch.save(self._qweight_packed.detach().cpu(), q_path)
            torch.save(self._w_scales.detach().cpu(), s_path)
            torch.save(self._inv_s.detach().cpu(), inv_path)
            torch.save(meta, meta_path)
        except Exception as e:
            print(f"[SMOOTHQUANT-W4-CACHE] failed to save {self.name}: {e}", flush=True)

    @torch.no_grad()
    def _build_or_load_packed_weight(self, weight: torch.Tensor, smooth_s: torch.Tensor) -> None:
        if self._try_load_int4_cache():
            return

        W_s = weight.detach().to(torch.float32) * smooth_s.detach().to(device=weight.device, dtype=torch.float32)[None, :]
        scales = compute_mse_scales(W_s, 4, swc=float(self.cfg.swc)).to(device=W_s.device)
        q = torch.round(W_s / scales[:, None])
        q = torch.clamp(q, -8, 7).to(torch.int8)
        q_packed = pack_int4_signed(q)

        self._qweight_packed.copy_(q_packed.to(device=self._qweight_packed.device, dtype=torch.uint8))
        self._w_scales.copy_(scales.to(device=self._w_scales.device, dtype=self._w_scales.dtype))

        del W_s, scales, q, q_packed
        self._save_int4_cache()

    def _linear_unpack_fake(self, x_s: torch.Tensor) -> torch.Tensor:
        q = unpack_int4_signed(self._qweight_packed.to(x_s.device), self.in_features)
        W_q = q.to(dtype=x_s.dtype) * self._w_scales.to(device=x_s.device, dtype=x_s.dtype)[:, None]
        return torch.nn.functional.linear(x_s, W_q, None)

    def _scale_and_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        mode = str(getattr(self.cfg, "act_quant_mode", "dynamic_symmetric")).strip().lower()
        impl = str(getattr(self.cfg, "act_quant_impl", "triton")).strip().lower()

        inv_s = self._inv_s
        bits = int(self.act_bits)

        if bits <= 0 or bits >= 16:
            if inv_s.device != x.device or inv_s.dtype != x.dtype:
                inv_s = inv_s.to(device=x.device, dtype=x.dtype)
            return x * inv_s

        # Fast path: fuse x * inv_s and dynamic token affine fake quant in one Triton kernel.
        if mode in {"dynamic_token_affine", "token_affine", "affine"}:
            if impl == "triton" and x.is_cuda:
                return smoothquant_scale_token_affine_triton(
                    x,
                    inv_s,
                    bits,
                    self.in_features,
                )

            if inv_s.device != x.device or inv_s.dtype != x.dtype:
                inv_s = inv_s.to(device=x.device, dtype=x.dtype)
            return fake_quantize_activation_affine_dynamic(x * inv_s, bits, group_size=0)

        # Group affine fallback. Kept for accuracy debugging; slower than token affine.
        if mode in {"dynamic_group_affine", "group_affine"}:
            if inv_s.device != x.device or inv_s.dtype != x.dtype:
                inv_s = inv_s.to(device=x.device, dtype=x.dtype)
            group_size = int(getattr(self.cfg, "act_group_size", 128))
            if group_size <= 0:
                group_size = 128
            return fake_quantize_activation_affine_dynamic(x * inv_s, bits, group_size=group_size)

        # Old symmetric path.
        if mode in {"dynamic_symmetric", "symmetric", "dynamic_token_symmetric"}:
            if inv_s.device != x.device or inv_s.dtype != x.dtype:
                inv_s = inv_s.to(device=x.device, dtype=x.dtype)
            x_s = x * inv_s
            return fake_quantize_activation(
                x_s,
                bits,
                lac=float(self.cfg.lac),
                group_size=int(self.cfg.act_group_size),
            )

        raise ValueError(f"Unknown OPENPI_SMOOTHQUANT_ACT_QUANT_MODE={mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_s = self._scale_and_quantize_activation(x)

        if self.forward_backend == "kernel":
            y = w4_weight_only_linear_triton(
                x_s,
                self._qweight_packed,
                self._w_scales,
                in_features=self.in_features,
                out_features=self.out_features,
            )
        elif self.forward_backend == "unpack_fake":
            y = self._linear_unpack_fake(x_s)
        else:
            raise ValueError(f"Unknown OPENPI_SMOOTHQUANT_BACKEND={self.forward_backend!r}")

        if self._bias is not None:
            bias = self._bias
            if bias.device != y.device or bias.dtype != y.dtype:
                bias = bias.to(device=y.device, dtype=y.dtype)
            y = y + bias

        return y


def get_smoothquant_w4_layers(model: nn.Module) -> List[SmoothQuantW4Linear]:
    return [m for m in model.modules() if isinstance(m, SmoothQuantW4Linear)]


def count_smoothquant_w4_linears(model: nn.Module) -> int:
    return len(get_smoothquant_w4_layers(model))


def enable_openpi_smoothquant_w4_all_linears(
    model: nn.Module,
    *,
    config: Optional[SmoothQuantW4Config] = None,
    dry_run: Optional[bool] = None,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
) -> int:
    if config is None:
        config = SmoothQuantW4Config.from_env()

    if dry_run is None:
        dry_run = _env_bool("OPENPI_SMOOTHQUANT_DRYRUN", "0")

    if include_regex is None:
        include_regex = config.include_regex
    if exclude_regex is None:
        exclude_regex = config.exclude_regex

    if int(config.weight_bits) != 4:
        raise ValueError(f"SmoothQuant backend requires real W4 weight, got W{config.weight_bits}")
    if int(config.act_bits) not in {2, 4, 8, 16}:
        raise ValueError(f"SmoothQuant supports A bits in {{2,4,8,16}}, got A{config.act_bits}")

    calib = _load_smoothquant_calibration(config.calib_path)
    if config.require_calib and not calib and not dry_run:
        raise RuntimeError(
            "SmoothQuant requires calibration. Set OPENPI_SMOOTHQUANT_CALIB_PATH "
            "or pass SmoothQuantW4Config(calib_path=...)."
        )

    include_re = re.compile(include_regex)
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    targets: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, SmoothQuantW4Linear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not include_re.search(name):
            continue
        if exclude_re is not None and exclude_re.search(name):
            continue
        targets.append(name)

    print(
        f"[SMOOTHQUANT-W4] matched={len(targets)} W4A{config.act_bits} alpha={config.alpha} "
        f"lac={config.lac} swc={config.swc} act_mode={config.act_quant_mode} "
        f"group={config.act_group_size} impl={config.act_quant_impl} calib_layers={len(calib)} "
        f"cache_dir={config.int4_cache_dir} backend={config.backend}",
        flush=True,
    )

    bucket_counter = Counter()
    suffix_counter = Counter()
    shape_counter = defaultdict(int)
    missing_calib: List[str] = []

    def bucket_name(name: str) -> str:
        if name.startswith("paligemma_with_expert.paligemma"):
            return "paligemma"
        if name.startswith("paligemma_with_expert.gemma_expert"):
            return "gemma_expert"
        if name in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "state_proj"}:
            return "action_io_time"
        return "other"

    replaced = 0
    t0 = time.perf_counter()

    for name in targets:
        parent, attr = _get_parent_module_and_attr(model, name)
        old = getattr(parent, attr)
        if not isinstance(old, nn.Linear):
            continue

        bucket_counter[bucket_name(name)] += 1
        suffix_counter[name.split(".")[-1]] += 1
        shape_counter[(old.in_features, old.out_features)] += 1

        calib_rec = calib.get(name)
        if config.require_calib and calib_rec is None:
            missing_calib.append(name)
            if not dry_run:
                continue

        if dry_run:
            print(f"[SMOOTHQUANT-W4][DRYRUN] {name}: Linear({old.in_features}->{old.out_features})")
            continue

        new = SmoothQuantW4Linear(old, name=name, cfg=config, calib_rec=calib_rec)
        setattr(parent, attr, new)
        del old

        replaced += 1
        if replaced % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if missing_calib:
        msg = f"[SMOOTHQUANT-W4] missing calibration for {len(missing_calib)} selected layers"
        print(msg, flush=True)
        for n in missing_calib[:20]:
            print(f"  missing: {n}", flush=True)
        if config.require_calib and not dry_run:
            raise RuntimeError(msg)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if dry_run:
        print(f"[SMOOTHQUANT-W4] dry-run total layers listed: {len(targets)}", flush=True)
        return len(targets)

    print(f"[SMOOTHQUANT-W4] total layers replaced: {replaced}, dt={time.perf_counter() - t0:.2f}s", flush=True)
    print("[SMOOTHQUANT-W4] breakdown by bucket:", dict(bucket_counter), flush=True)
    print("[SMOOTHQUANT-W4] top suffixes:", suffix_counter.most_common(20), flush=True)
    print("[SMOOTHQUANT-W4] top shapes:", sorted(shape_counter.items(), key=lambda x: -x[1])[:20], flush=True)
    return replaced


def enable_openpi_smoothquant_w4(
    model: nn.Module,
    *,
    config: Optional[SmoothQuantW4Config] = None,
) -> nn.Module:
    if config is None:
        config = SmoothQuantW4Config.from_env()

    pathlib.Path(config.int4_cache_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"[SMOOTHQUANT-W4] enabling "
        f"W4A{config.act_bits} alpha={config.alpha} "
        f"act_mode={config.act_quant_mode} group={config.act_group_size} "
        f"impl={config.act_quant_impl} "
        f"lac={config.lac} swc={config.swc} backend={config.backend} "
        f"calib={config.calib_path} cache={config.int4_cache_dir}",
        flush=True,
    )

    replaced = enable_openpi_smoothquant_w4_all_linears(model, config=config)
    print(f"[SMOOTHQUANT-W4] enabled layers={replaced}", flush=True)
    return model
