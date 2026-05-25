# Copyright (c) 2026
# DuQuant-compatible packed W4 backend for OpenPI.
#
# This version adds:
#   1. disk cache for packed INT4 weights:
#        $OPENPI_DUQUANT_INT4_CACHE_DIR/
#          <layer>_qweight_packed.pt
#          <layer>_scales.pt
#          <layer>_meta.pt
#   2. per-layer progress prints during convert_duquant_to_packed_w4()
#
# Normal run:
#   export OPENPI_DUQUANT_WEIGHT_BACKEND=packed_w4
#   export OPENPI_DUQUANT_PACKED_BACKEND=kernel
#
# Optional debug fallback:
#   export OPENPI_DUQUANT_PACKED_BACKEND=unpack_fake

from __future__ import annotations

import gc
import hashlib
import os
import pathlib
import re
import time
from typing import Optional

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


from openpi.models_pytorch.quant.duquant_layers import DuQuantLinear
from openpi.models_pytorch.quant.duquant_preprocess import (
    apply_input_transform_optimized,
    apply_output_restore_optimized,
    fake_quantize_sym,
    qmax,
    transform_weight_for_forward_optimized,
)


# --------------------------------------------------------------------------------------
# INT4 pack / unpack
# --------------------------------------------------------------------------------------


def pack_int4_signed(q: torch.Tensor) -> torch.Tensor:
    """
    Pack signed int4 tensor into uint8.

    Args:
        q: int tensor with values in [-8, 7], shape [out_features, in_features].

    Returns:
        packed uint8 tensor, shape [out_features, ceil(in_features / 2)].
        low nibble stores even k, high nibble stores odd k.
    """
    if q.dim() != 2:
        raise ValueError(f"pack_int4_signed expects 2D tensor, got shape={tuple(q.shape)}")

    q = torch.clamp(q.to(torch.int16), -8, 7)
    q_u = (q + 8).to(torch.uint8)  # [0, 15]

    if q_u.shape[1] % 2 == 1:
        pad = torch.zeros((q_u.shape[0], 1), dtype=torch.uint8, device=q_u.device)
        q_u = torch.cat([q_u, pad], dim=1)

    low = q_u[:, 0::2]
    high = q_u[:, 1::2]

    packed = low | (high << 4)
    return packed.contiguous()


def unpack_int4_signed(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """
    Unpack uint8 packed INT4 tensor back to signed int8.

    Args:
        packed: uint8 tensor, shape [out_features, ceil(in_features / 2)].
        in_features: original K.

    Returns:
        int8 tensor with values in [-8, 7], shape [out_features, in_features].
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be torch.uint8, got {packed.dtype}")
    if packed.dim() != 2:
        raise ValueError(f"unpack_int4_signed expects 2D tensor, got shape={tuple(packed.shape)}")

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    q_u = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    q_u[:, 0::2] = low
    q_u[:, 1::2] = high

    q = q_u[:, :in_features].to(torch.int16) - 8
    return q.to(torch.int8).contiguous()


def dequantize_packed_int4(
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Debug fallback: unpack packed int4 into float W_q.

    Returns:
        W_q float tensor, shape [out_features, in_features].
    """
    q = unpack_int4_signed(qweight_packed.to(device), in_features)
    return q.to(dtype=dtype) * scales.to(device=device, dtype=dtype)[:, None]


# --------------------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------------------


def _safe_part(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _torch_load(path: pathlib.Path, map_location="cpu"):
    # weights_only exists in newer torch, but not all environments have it.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _get_default_int4_cache_dir() -> pathlib.Path:
    cache_dir = os.environ.get("OPENPI_DUQUANT_INT4_CACHE_DIR", "").strip()
    if cache_dir:
        return pathlib.Path(cache_dir)

    pack_dir = os.environ.get("OPENPI_DUQUANT_PACKDIR", "").strip()
    if pack_dir:
        return pathlib.Path(pack_dir) / "duquant_int4_cache"

    return pathlib.Path("/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant_int4_cache")


def _int4_cache_enabled() -> bool:
    return os.environ.get("OPENPI_DUQUANT_INT4_CACHE", "1").lower() not in {"0", "false", "no"}


# --------------------------------------------------------------------------------------
# Triton W4 weight-only GEMM
# --------------------------------------------------------------------------------------


if _HAS_TRITON:

    @triton.jit
    def _w4_weight_only_gemm_kernel(
        X,
        QW,
        S,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        KPACK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k

            a = tl.load(
                X + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)

            byte_idx = k // 2

            # QW layout: [N, KPACK]
            packed = tl.load(
                QW + offs_n[None, :] * KPACK + byte_idx[:, None],
                mask=(offs_n[None, :] < N) & (byte_idx[:, None] < KPACK) & (k[:, None] < K),
                other=0,
            )

            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            use_high = (k[:, None] & 1) == 1
            q_u = tl.where(use_high, high, low)

            # unsigned nibble [0, 15] -> signed int4 [-8, 7]
            q = q_u.to(tl.float32) - 8.0

            scales = tl.load(S + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            w = q * scales[None, :]

            acc += tl.dot(a, w, input_precision="tf32")

        tl.store(
            Y + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def w4_weight_only_linear_triton(
    x: torch.Tensor,
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """
    Compute y = x @ dequant(qweight_packed, scales).T using Triton.

    Args:
        x: fp16/bf16/fp32 tensor, shape [..., in_features].
        qweight_packed: uint8 tensor, shape [out_features, ceil(in_features / 2)].
        scales: tensor, shape [out_features].

    Returns:
        y: tensor, shape [..., out_features], same dtype as x.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available. Use OPENPI_DUQUANT_PACKED_BACKEND=unpack_fake.")

    if x.shape[-1] != in_features:
        raise ValueError(f"x last dim={x.shape[-1]} but in_features={in_features}")

    if qweight_packed.dtype != torch.uint8:
        raise TypeError(f"qweight_packed must be uint8, got {qweight_packed.dtype}")

    orig_shape = x.shape[:-1]
    x2d = x.reshape(-1, in_features).contiguous()

    M = x2d.shape[0]
    N = int(out_features)
    K = int(in_features)
    KPACK = (K + 1) // 2

    if qweight_packed.shape != (N, KPACK):
        raise ValueError(
            f"qweight_packed shape mismatch: expected {(N, KPACK)}, got {tuple(qweight_packed.shape)}"
        )

    y2d = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Conservative defaults. Tune later.
    BM = int(os.environ.get("OPENPI_W4_TRITON_BLOCK_M", "16"))
    BN = int(os.environ.get("OPENPI_W4_TRITON_BLOCK_N", "32"))
    BK = int(os.environ.get("OPENPI_W4_TRITON_BLOCK_K", "64"))

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    _w4_weight_only_gemm_kernel[grid](
        x2d,
        qweight_packed,
        scales.contiguous(),
        y2d,
        M,
        N,
        K,
        KPACK,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=3,
    )

    return y2d.reshape(*orig_shape, N)


# --------------------------------------------------------------------------------------
# Packed DuQuant Linear
# --------------------------------------------------------------------------------------


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _get_base_weight(base: DuQuantLinear) -> torch.Tensor:
    if hasattr(base, "_weight"):
        return getattr(base, "_weight")
    if hasattr(base, "weight"):
        return getattr(base, "weight")
    raise AttributeError("Cannot find weight tensor on DuQuantLinear.")


def _get_base_bias(base: DuQuantLinear) -> Optional[torch.Tensor]:
    if hasattr(base, "bias") and getattr(base, "bias") is not None:
        return getattr(base, "bias")
    if hasattr(base, "_bias") and getattr(base, "_bias") is not None:
        return getattr(base, "_bias")
    return None


class DuQuantPackedW4Linear(nn.Module):
    """
    DuQuant-compatible packed W4 Linear.

    This module keeps DuQuant's:
        - input transform
        - W4 quantization scale
        - optional output restore
        - activation fake quant

    It changes the weight storage / compute path:
        - stores signed W4 weights packed into uint8
        - kernel backend computes directly from packed W4
    """

    def __init__(self, base: DuQuantLinear):
        super().__init__()

        self.name = getattr(base, "name", "")
        self.in_features = int(getattr(base, "in_features"))
        self.out_features = int(getattr(base, "out_features"))

        self.cfg = getattr(base, "cfg")
        self.pack = getattr(base, "pack")

        self.weight_bits = 4
        self.act_bits = int(getattr(base, "act_bits", 16))

        self._block_size = int(getattr(base, "_block_size"))
        self._block_out_size = int(getattr(base, "_block_out_size"))

        # Copy perm cache.
        perm_cache = getattr(base, "_perm_cache", None)
        if perm_cache is not None:
            self.register_buffer("_perm_cache", perm_cache.detach().clone(), persistent=False)
        else:
            self._perm_cache = None

        # Copy R_in buffers.
        self._R_in_block_indices = list(getattr(base, "_R_in_block_indices", []))
        for b in self._R_in_block_indices:
            r = getattr(base, f"_R_in_{b}")
            self.register_buffer(f"_R_in_{b}", r.detach().clone(), persistent=False)

        # Copy R_out buffers.
        self._R_out_block_indices = list(getattr(base, "_R_out_block_indices", []))
        for b in self._R_out_block_indices:
            r = getattr(base, f"_R_out_{b}")
            self.register_buffer(f"_R_out_{b}", r.detach().clone(), persistent=False)

        # Bias. In inference, buffer is enough and avoids training semantics.
        bias = _get_base_bias(base)
        if bias is not None:
            self.register_buffer("_bias", bias.detach().clone(), persistent=False)
        else:
            self._bias = None

        # Bias rotated path for row_rot_mode != restore if original base has it.
        bias_rot = getattr(base, "_bias_rot", None)
        if bias_rot is not None:
            self.register_buffer("_bias_rot", bias_rot.detach().clone(), persistent=False)
        else:
            self._bias_rot = None

        # Optional activation percentile vector from base if it already exists.
        act_vec = getattr(base, "_act_percentile_vec", None)
        if act_vec is not None and isinstance(act_vec, torch.Tensor):
            self.register_buffer("_act_percentile_vec", act_vec.detach().clone(), persistent=False)
        else:
            self.register_buffer("_act_percentile_vec", torch.empty(0), persistent=False)

        weight = _get_base_weight(base)
        packed_cols = (self.in_features + 1) // 2
        self.register_buffer(
            "_weight_compat_storage",
            torch.empty(1, dtype=weight.dtype, device=weight.device),
            persistent=False,
        )

        self.register_buffer(
            "_qweight_packed",
            torch.empty(
                self.out_features,
                packed_cols,
                dtype=torch.uint8,
                device=weight.device,
            ),
            persistent=True,
        )

        self.register_buffer(
            "_w_scales",
            torch.empty(
                self.out_features,
                dtype=weight.dtype,
                device=weight.device,
            ),
            persistent=True,
        )

        self.forward_backend = os.environ.get("OPENPI_DUQUANT_PACKED_BACKEND", "kernel").lower()
        self.act_scale_mode = os.environ.get("OPENPI_DUQUANT_PACKED_ACT_SCALE_MODE", "duquant_or_batch_amax")

        self._build_packed_weight_from_base(base)

    def _get_R_in_cache(self):
        return {b: getattr(self, f"_R_in_{b}") for b in self._R_in_block_indices}

    def _get_R_out_cache(self):
        return {b: getattr(self, f"_R_out_{b}") for b in self._R_out_block_indices}

    @property
    def bias(self):
        return self._bias
    @property
    def weight(self):
        """
        Compatibility-only fake weight tensor.

        This is NOT the real dequantized weight.
        It exists because some upstream modules only inspect:
            module.weight.dtype
            module.weight.shape
            module.weight.numel()

        We return a zero-stride view backed by a single element, so it does not
        allocate the full [out_features, in_features] matrix.
        """
        return self._weight_compat_storage.as_strided(
            (self.out_features, self.in_features),
            (0, 0),
        )

    def set_a_bits(self, bits: int) -> None:
        self.act_bits = int(bits)

    def set_w_bits(self, bits: int) -> None:
        bits = int(bits)
        if bits != 4:
            raise ValueError("DuQuantPackedW4Linear only supports fixed W4 weights.")

    def _cache_prefix(self) -> pathlib.Path:
        cache_dir = _get_default_int4_cache_dir()

        tag = os.environ.get("OPENPI_DUQUANT_INT4_CACHE_TAG", "").strip()
        if not tag:
            # Layout matters because it changes which layers are converted, but per-layer cache still
            # mostly depends on name/shape/config. This tag helps keep experiments separated.
            tag = os.environ.get("OPENPI_DUQUANT_LAYOUT", "default")

        key = {
            "name": self.name,
            "in": self.in_features,
            "out": self.out_features,
            "row_rot_mode": str(getattr(self.cfg, "row_rot_mode", "")),
            "block_size": self._block_size,
            "block_out_size": self._block_out_size,
            "weight_bits": 4,
            "pack_version": 2,
        }

        raw = repr(sorted(key.items())).encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()[:12]

        safe_name = _safe_part(self.name if self.name else "unnamed")
        prefix = f"{_safe_part(tag)}__{safe_name}__out{self.out_features}_in{self.in_features}__{digest}"
        return cache_dir / prefix

    def _try_load_int4_cache(self) -> bool:
        if not _int4_cache_enabled():
            return False

        prefix = self._cache_prefix()
        q_path = pathlib.Path(str(prefix) + "_qweight_packed.pt")
        s_path = pathlib.Path(str(prefix) + "_scales.pt")
        m_path = pathlib.Path(str(prefix) + "_meta.pt")

        if not (q_path.exists() and s_path.exists() and m_path.exists()):
            return False

        try:
            meta = _torch_load(m_path, map_location="cpu")
            expected = {
                "name": self.name,
                "in_features": self.in_features,
                "out_features": self.out_features,
                "packed_shape": tuple(self._qweight_packed.shape),
                "scale_shape": tuple(self._w_scales.shape),
                "weight_bits": 4,
                "row_rot_mode": str(getattr(self.cfg, "row_rot_mode", "")),
                "block_size": self._block_size,
                "block_out_size": self._block_out_size,
            }

            for k, v in expected.items():
                if meta.get(k) != v:
                    print(
                        f"[PACKED-W4-CACHE] meta mismatch for {self.name}: {k} "
                        f"expected={v} got={meta.get(k)}; rebuilding",
                        flush=True,
                    )
                    return False

            q = _torch_load(q_path, map_location="cpu")
            s = _torch_load(s_path, map_location="cpu")

            if tuple(q.shape) != tuple(self._qweight_packed.shape) or q.dtype != torch.uint8:
                print(f"[PACKED-W4-CACHE] qweight shape/dtype mismatch for {self.name}; rebuilding", flush=True)
                return False

            if tuple(s.shape) != tuple(self._w_scales.shape):
                print(f"[PACKED-W4-CACHE] scales shape mismatch for {self.name}; rebuilding", flush=True)
                return False

            self._qweight_packed.copy_(q.to(device=self._qweight_packed.device, dtype=torch.uint8))
            self._w_scales.copy_(s.to(device=self._w_scales.device, dtype=self._w_scales.dtype))

            print(f"[PACKED-W4-CACHE] loaded {self.name} from {q_path.parent}", flush=True)
            return True

        except Exception as e:
            print(f"[PACKED-W4-CACHE] failed to load {self.name}: {e}; rebuilding", flush=True)
            return False

    def _save_int4_cache(self) -> None:
        if not _int4_cache_enabled():
            return

        prefix = self._cache_prefix()
        prefix.parent.mkdir(parents=True, exist_ok=True)

        q_path = pathlib.Path(str(prefix) + "_qweight_packed.pt")
        s_path = pathlib.Path(str(prefix) + "_scales.pt")
        m_path = pathlib.Path(str(prefix) + "_meta.pt")

        meta = {
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "packed_shape": tuple(self._qweight_packed.shape),
            "scale_shape": tuple(self._w_scales.shape),
            "weight_bits": 4,
            "row_rot_mode": str(getattr(self.cfg, "row_rot_mode", "")),
            "block_size": self._block_size,
            "block_out_size": self._block_out_size,
            "cache_version": 2,
        }

        try:
            torch.save(self._qweight_packed.detach().cpu(), q_path)
            torch.save(self._w_scales.detach().cpu(), s_path)
            torch.save(meta, m_path)
            print(f"[PACKED-W4-CACHE] saved {self.name} to {prefix.parent}", flush=True)
        except Exception as e:
            print(f"[PACKED-W4-CACHE] failed to save {self.name}: {e}", flush=True)

    @torch.no_grad()
    def _build_packed_weight_from_base(self, base: DuQuantLinear) -> None:
        """
        Build packed W4 directly from DuQuant-transformed weight.

        Important:
            Do NOT call base._maybe_update_weight_cache().
            That would also build the full float _W_t_quantized cache, which is slow
            and defeats the memory-saving purpose during conversion.
        """
        if self._try_load_int4_cache():
            return

        weight = _get_base_weight(base).detach()
        apply_row = self.cfg.row_rot_mode != "0"

        W_t, scales = transform_weight_for_forward_optimized(
            weight,
            base.pack,
            weight_bits=4,
            apply_row_rot=apply_row,
            perm_cache=getattr(base, "_perm_cache", None),
            R_in_cache=base._get_R_in_cache(),
            R_out_cache=base._get_R_out_cache(),
            block_size=base._block_size,
            block_out_size=base._block_out_size,
        )

        if W_t.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"W_t shape mismatch for {self.name}: expected "
                f"{(self.out_features, self.in_features)}, got {tuple(W_t.shape)}"
            )

        scales = scales.to(device=W_t.device)

        # Keep mostly in W_t dtype to avoid an extra full fp32 copy.
        # For W4, q range is [-8, 7].
        q = torch.round(W_t / scales[:, None])
        q = torch.clamp(q, -8, 7).to(torch.int8)

        q_packed = pack_int4_signed(q)

        self._qweight_packed.copy_(q_packed.to(self._qweight_packed.device))
        self._w_scales.copy_(scales.to(dtype=self._w_scales.dtype, device=self._w_scales.device))

        del W_t, scales, q, q_packed

        self._save_int4_cache()

    def _current_batch_percentile(self, x: torch.Tensor) -> torch.Tensor:
        # Exact but slower path, mostly for compatibility/debug.
        x_abs = torch.abs(x.detach().to(torch.float32))
        c = x_abs.shape[-1]
        x2d = x_abs.reshape(-1, c)
        percentile = float(getattr(self.cfg, "act_percentile", 99.9))
        p = torch.quantile(x2d, percentile / 100.0, dim=0)
        return torch.clamp(p, min=1e-6).to(dtype=x.dtype, device=x.device)

    def _batch_amax(self, x: torch.Tensor) -> torch.Tensor:
        x_abs = torch.abs(x.detach())
        c = x_abs.shape[-1]
        x2d = x_abs.reshape(-1, c)
        return torch.amax(x2d, dim=0).clamp_min(1e-6).to(dtype=x.dtype, device=x.device)

    def _token_amax(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(torch.abs(x.detach()), dim=-1, keepdim=True).clamp_min(1e-6).to(dtype=x.dtype, device=x.device)

    def _get_act_scale(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        bits = int(bits)
        if bits <= 0 or bits >= 16:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        mode = self.act_scale_mode

        if mode == "duquant_percentile":
            p = self._current_batch_percentile(x)

        elif mode == "batch_amax":
            p = self._batch_amax(x)

        elif mode == "token_amax":
            p = self._token_amax(x)

        else:
            # duquant_or_batch_amax:
            # If a calibrated vector exists and matches channel size, use it.
            # Otherwise use dynamic batch_amax instead of slow quantile.
            if self._act_percentile_vec.numel() == x.shape[-1]:
                p = self._act_percentile_vec.to(dtype=x.dtype, device=x.device)
            else:
                p = self._batch_amax(x)

        return p / qmax(bits)

    def _linear_unpack_fake(self, x_t: torch.Tensor) -> torch.Tensor:
        W_q = dequantize_packed_int4(
            self._qweight_packed,
            self._w_scales,
            self.in_features,
            dtype=x_t.dtype,
            device=x_t.device,
        )
        return torch.nn.functional.linear(x_t, W_q, None)

    def _linear_kernel(self, x_t: torch.Tensor) -> torch.Tensor:
        return w4_weight_only_linear_triton(
            x_t,
            self._qweight_packed,
            self._w_scales,
            in_features=self.in_features,
            out_features=self.out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = apply_input_transform_optimized(
            x,
            self.pack,
            self._perm_cache,
            self._get_R_in_cache(),
            self._block_size,
        )

        if self.act_bits > 0 and self.act_bits < 16:
            s_a = self._get_act_scale(x_t, self.act_bits)
            x_t = fake_quantize_sym(x_t, s_a, self.act_bits, label="activation_forward")

        if self.forward_backend == "kernel":
            y = self._linear_kernel(x_t)
        elif self.forward_backend == "unpack_fake":
            y = self._linear_unpack_fake(x_t)
        else:
            raise ValueError(f"Unknown OPENPI_DUQUANT_PACKED_BACKEND={self.forward_backend}")

        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            y = apply_output_restore_optimized(
                y,
                self.pack,
                self._get_R_out_cache(),
                self._block_out_size,
            )
            if self._bias is not None:
                y = y + self._bias.to(dtype=y.dtype, device=y.device)
        else:
            if self._bias is not None:
                bias = self._bias_rot if self._bias_rot is not None else self._bias
                y = y + bias.to(dtype=y.dtype, device=y.device)

        return y


def convert_duquant_to_packed_w4(model: nn.Module) -> int:
    """
    Convert all DuQuantLinear modules into DuQuantPackedW4Linear.

    This must be called AFTER your current _enable_openpi_duquant_staged(model).
    """
    targets = []

    for name, module in model.named_modules():
        if isinstance(module, DuQuantLinear):
            targets.append(name)

    print(f"[PACKED-W4] Found DuQuantLinear layers: {len(targets)}", flush=True)
    print(f"[PACKED-W4] backend={os.environ.get('OPENPI_DUQUANT_PACKED_BACKEND', 'kernel')}", flush=True)
    print(f"[PACKED-W4] int4_cache={_get_default_int4_cache_dir()} enabled={_int4_cache_enabled()}", flush=True)

    replaced = 0
    t_all = time.perf_counter()

    for i, name in enumerate(targets, start=1):
        parent, attr = _get_parent_module_and_attr(model, name)
        old = getattr(parent, attr)

        if not isinstance(old, DuQuantLinear):
            continue

        try:
            shape = (int(old.out_features), int(old.in_features))
        except Exception:
            shape = "unknown"

        print(
            f"[PACKED-W4] [{i}/{len(targets)}] converting {name}, shape={shape}",
            flush=True,
        )

        t0 = time.perf_counter()

        new = DuQuantPackedW4Linear(old)
        setattr(parent, attr, new)

        del old
        replaced += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        gc.collect()

        print(
            f"[PACKED-W4] [{i}/{len(targets)}] done {name}, dt={time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    print(
        f"[PACKED-W4] Converted layers: {replaced}, total_dt={time.perf_counter() - t_all:.2f}s",
        flush=True,
    )

    return replaced


def count_packed_w4_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, DuQuantPackedW4Linear))


def debug_compare_packed_vs_unpack(module: DuQuantPackedW4Linear, x: torch.Tensor) -> dict[str, float]:
    """
    Debug helper for a single layer. Compares kernel backend with unpack_fake backend.
    """
    if not isinstance(module, DuQuantPackedW4Linear):
        raise TypeError("module must be DuQuantPackedW4Linear")

    old_backend = module.forward_backend

    with torch.inference_mode():
        module.forward_backend = "unpack_fake"
        y_ref = module(x)

        module.forward_backend = "kernel"
        y_ker = module(x)

    module.forward_backend = old_backend

    diff = (y_ref.to(torch.float32) - y_ker.to(torch.float32)).abs()

    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "ref_norm": float(y_ref.to(torch.float32).norm().item()),
        "ker_norm": float(y_ker.to(torch.float32).norm().item()),
    }
