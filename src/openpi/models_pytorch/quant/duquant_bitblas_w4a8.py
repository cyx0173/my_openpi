from __future__ import annotations

"""
BitBLAS W4A8 backend for DuQuant real activation path.

This module implements the fast path:
    qa[int8] @ qweight[int4].T -> int32
    int32 * activation_scale * weight_scale + bias -> bf16

It is intentionally limited to the no-output-restore A8 path. Layers with
R_out restore or group activation quantization should fall back to the existing
Triton/CUTLASS paths in duquant_real_w4a.py.
"""

import os
import warnings
from typing import Optional

import torch

# BitBLAS requires this in multi-GPU environments before Matmul construction.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

try:
    import bitblas  # type: ignore

    _HAS_BITBLAS = True
except Exception as _bitblas_import_error:  # pragma: no cover - depends on env
    bitblas = None
    _HAS_BITBLAS = False
    warnings.warn(
        f"[duquant-bitblas] failed to import bitblas: {_bitblas_import_error}. "
        "BitBLAS W4A8 backend will be disabled."
    )

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception as _triton_import_error:  # pragma: no cover - depends on env
    triton = None
    tl = None
    _HAS_TRITON = False
    warnings.warn(
        f"[duquant-bitblas] failed to import triton: {_triton_import_error}. "
        "BitBLAS W4A8 scale/bias epilogue will be disabled."
    )


# Caches are keyed by tensor pointer/device/shape and static GEMM shape.
# transform_weight is expensive and must be done once per layer weight, not per forward.
_MATMUL_CACHE: dict[tuple[int, int, int, str], object] = {}
_W_BITBLAS_CACHE_BY_PTR: dict[
    tuple[int, int, tuple[int, ...], torch.device, str], torch.Tensor
] = {}

_BITBLAS_HIT_PRINTED = False


def bitblas_w4a8_enabled() -> bool:
    """Return whether the BitBLAS backend is requested and importable."""
    flag = os.environ.get("OPENPI_DUQUANT_BITBLAS_BACKEND", "0").strip().lower()
    requested = flag in {"1", "true", "yes", "on", "bitblas", "w4a8"}
    return bool(requested and _HAS_BITBLAS and _HAS_TRITON and torch.cuda.is_available())


def _build_matmul(m: int, n: int, k: int, *, out_dtype: str = "int32"):
    if not _HAS_BITBLAS or bitblas is None:
        raise RuntimeError("BitBLAS is not available")

    cfg = bitblas.MatmulConfig(
        M=int(m),
        N=int(n),
        K=int(k),
        A_dtype="int8",
        W_dtype="int4",
        accum_dtype="int32",
        out_dtype=str(out_dtype),
        layout="nt",
        with_bias=False,
        group_size=-1,
        with_scaling=False,
        with_zeros=False,
        zeros_mode="original",
        storage_dtype="int8",
    )

    try:
        return bitblas.Matmul(config=cfg, enable_tuning=False)
    except TypeError:
        try:
            return bitblas.Matmul(config=cfg)
        except TypeError:
            return bitblas.Matmul(cfg)


def _get_or_create_matmul(m: int, n: int, k: int, *, out_dtype: str = "int32"):
    key = (int(m), int(n), int(k), str(out_dtype))
    op = _MATMUL_CACHE.get(key)
    if op is None:
        op = _build_matmul(m, n, k, out_dtype=out_dtype)
        _MATMUL_CACHE[key] = op
    return op


def _unpack_int4_signed_to_dense(qweight_packed: torch.Tensor, k: int) -> torch.Tensor:
    """Unpack DuQuant packed signed INT4 to dense int8 [N, K] in [-8, 7]."""
    if qweight_packed.dtype is not torch.uint8:
        raise TypeError(f"qweight_packed must be uint8, got {qweight_packed.dtype}")
    if qweight_packed.dim() != 2:
        raise ValueError(f"qweight_packed must be [N, ceil(K/2)], got {tuple(qweight_packed.shape)}")

    n, kpack = qweight_packed.shape
    expected_kpack = (int(k) + 1) // 2
    if int(kpack) != expected_kpack:
        raise ValueError(f"qweight_packed KPACK mismatch: expected {expected_kpack}, got {int(kpack)}")

    packed = qweight_packed.contiguous()
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    q_u = torch.empty((n, kpack * 2), device=packed.device, dtype=torch.uint8)
    q_u[:, 0::2] = low
    q_u[:, 1::2] = high

    q = q_u[:, : int(k)].to(torch.int16) - 8
    return q.to(torch.int8).contiguous()


def _get_or_create_bitblas_weight(
    qweight_packed: torch.Tensor,
    k: int,
    matmul_op,
    *,
    out_dtype: str = "int32",
) -> torch.Tensor:
    """Return BitBLAS-transformed signed-int4 weight for this packed tensor."""
    if not qweight_packed.is_cuda:
        raise ValueError("qweight_packed must be CUDA tensor for BitBLAS backend")

    qweight_packed = qweight_packed.contiguous()
    n = int(qweight_packed.shape[0])

    key = (
        int(qweight_packed.data_ptr()),
        int(k),
        tuple(qweight_packed.shape),
        qweight_packed.device,
        str(out_dtype),
    )
    cached = _W_BITBLAS_CACHE_BY_PTR.get(key)
    if cached is not None and cached.device == qweight_packed.device:
        return cached

    # IMPORTANT: transform_weight is expensive; use the already-built forward
    # operator and cache the transformed weight by the original packed tensor ptr.
    q_dense = _unpack_int4_signed_to_dense(qweight_packed, int(k))
    transformed = matmul_op.transform_weight(q_dense).contiguous()
    _W_BITBLAS_CACHE_BY_PTR[key] = transformed
    return transformed


if _HAS_TRITON:

    @triton.jit
    def _scale_i32_to_bf16_kernel(
        YI32,
        SA,
        SW,
        BIAS,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        acc = tl.load(
            YI32 + offs_m[:, None] * N + offs_n[None, :],
            mask=mask,
            other=0,
        ).to(tl.float32)

        sa = tl.load(SA + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

        out = acc * sa[:, None] * sw[None, :]

        if HAS_BIAS:
            b = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out = out + b[None, :]

        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


def _scale_i32_to_bf16(
    y_i32: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if not _HAS_TRITON or triton is None:
        raise RuntimeError("Triton is not available for BitBLAS scale epilogue")

    if y_i32.dtype is not torch.int32:
        raise TypeError(f"y_i32 must be int32, got {y_i32.dtype}")
    if y_i32.dim() != 2:
        raise ValueError(f"y_i32 must be [M,N], got {tuple(y_i32.shape)}")

    m, n = int(y_i32.shape[0]), int(y_i32.shape[1])
    y_i32 = y_i32.contiguous()
    act_scale = act_scale.reshape(m).to(device=y_i32.device, dtype=torch.float32).contiguous()
    weight_scale = weight_scale.reshape(n).to(device=y_i32.device, dtype=torch.float32).contiguous()

    has_bias = bias is not None
    if has_bias:
        bias_t = bias.reshape(n).to(device=y_i32.device, dtype=torch.bfloat16).contiguous()
    else:
        bias_t = torch.empty((n,), device=y_i32.device, dtype=torch.bfloat16)

    y = torch.empty((m, n), device=y_i32.device, dtype=torch.bfloat16)

    block_m = int(os.environ.get("OPENPI_DUQUANT_BITBLAS_SCALE_BLOCK_M", "16"))
    block_n = int(os.environ.get("OPENPI_DUQUANT_BITBLAS_SCALE_BLOCK_N", "64"))
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

    _scale_i32_to_bf16_kernel[grid](
        y_i32,
        act_scale,
        weight_scale,
        bias_t,
        y,
        m,
        n,
        HAS_BIAS=bool(has_bias),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return y


def bitblas_w4a8_linear_from_quantized_act(
    qa: torch.Tensor,
    act_scale: torch.Tensor,
    qweight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_shape_prefix: tuple[int, ...],
) -> torch.Tensor:
    """DuQuant A8/W4 BitBLAS fast path.

    Args:
        qa: int8 activation, shape [M,K].
        act_scale: float activation scale, shape [M].
        qweight_packed: uint8 signed-int4 packed weight, shape [N, ceil(K/2)].
        weight_scale: per-output-channel scale, shape [N].
        bias: optional bias, shape [N].
        out_shape_prefix: original activation prefix shape before flattening.

    Returns:
        bf16 tensor with shape [*out_shape_prefix, N].
    """
    global _BITBLAS_HIT_PRINTED

    if not bitblas_w4a8_enabled():
        raise RuntimeError("BitBLAS W4A8 backend is disabled or unavailable")

    if qa.dtype is not torch.int8:
        raise TypeError(f"qa must be int8, got {qa.dtype}")
    if qa.dim() != 2:
        raise ValueError(f"qa must be [M,K], got {tuple(qa.shape)}")
    if qweight_packed.dtype is not torch.uint8:
        raise TypeError(f"qweight_packed must be uint8, got {qweight_packed.dtype}")

    qa = qa.contiguous()
    m, k = int(qa.shape[0]), int(qa.shape[1])
    n = int(qweight_packed.shape[0])

    if qweight_packed.shape[1] != (k + 1) // 2:
        raise ValueError(
            f"qweight_packed shape mismatch: expected second dim {(k + 1) // 2}, "
            f"got {int(qweight_packed.shape[1])}"
        )

    if not _BITBLAS_HIT_PRINTED:
        print("[duquant-bitblas] HIT BitBLAS W4A8 int4 backend")
        _BITBLAS_HIT_PRINTED = True

    # Build the op for the actual M used by this forward. Build cost is cached.
    op = _get_or_create_matmul(m, n, k, out_dtype="int32")
    w_bitblas = _get_or_create_bitblas_weight(qweight_packed, k, op, out_dtype="int32")

    y_i32 = op(qa, w_bitblas)
    if y_i32.dtype is not torch.int32:
        y_i32 = y_i32.to(torch.int32)

    y_bf16 = _scale_i32_to_bf16(y_i32, act_scale, weight_scale, bias)
    return y_bf16.reshape(*out_shape_prefix, n)
