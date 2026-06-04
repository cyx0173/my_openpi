from __future__ import annotations
import os
import sys
from typing import Optional
import torch
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False
import warnings

try:
    from openpi.models_pytorch.quant_old.duquant_bitblas_w4a8 import (
        bitblas_w4a8_enabled as _duquant_bitblas_w4a8_enabled,
        bitblas_w4a8_linear_from_quantized_act as _duquant_bitblas_w4a8_linear_from_quantized_act,
    )
    _HAS_DUQUANT_BITBLAS_W4A8 = True
except Exception as _bitblas_w4a8_import_error:
    _HAS_DUQUANT_BITBLAS_W4A8 = False

    def _duquant_bitblas_w4a8_enabled() -> bool:
        return False

    def _duquant_bitblas_w4a8_linear_from_quantized_act(*args, **kwargs):
        raise RuntimeError(
            f"BitBLAS W4A8 backend import failed: {_bitblas_w4a8_import_error}"
        )


_CUTLASS_EXT_DIR = os.environ.get(
    "OPENPI_CUTLASS_EXT_DIR",
    "/home/chengyuxuan/openpi/.kernels/cutlass_ext",
)
if _CUTLASS_EXT_DIR not in sys.path:
    sys.path.insert(0, _CUTLASS_EXT_DIR)
try:
    import w8a8_cutlass_ext  # type: ignore
    _HAS_W8A8_CUTLASS_EXT = True
except Exception as _cutlass_import_error:
    w8a8_cutlass_ext = None
    _HAS_W8A8_CUTLASS_EXT = False
    warnings.warn(
        f"[duquant-cutlass] failed to import w8a8_cutlass_ext from {_CUTLASS_EXT_DIR}: "
        f"{_cutlass_import_error}. CUTLASS backend will be disabled."
    )
_QW8_CACHE_BY_PTR: dict[tuple[int, int, tuple[int, ...], torch.device], torch.Tensor] = {}

_SW_FP32_CACHE_BY_PTR: dict[tuple[int, tuple[int, ...], torch.device], torch.Tensor] = {}
_BIAS_BF16_CACHE_BY_PTR: dict[tuple[int, tuple[int, ...], torch.device], torch.Tensor] = {}
ENABLE_CUTLASS_W4A8_CACHED_BACKEND = True
_CUTLASS_HIT_PRINTED = False
_BITBLAS_HIT_PRINTED = False

def _duquant_cutlass_enabled() -> bool:
    return (
        ENABLE_CUTLASS_W4A8_CACHED_BACKEND
        and _HAS_W8A8_CUTLASS_EXT
        and torch.cuda.is_available()
    )


def _get_or_create_qw8_cache(
    qweight_packed: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if not _HAS_W8A8_CUTLASS_EXT:
        raise RuntimeError("w8a8_cutlass_ext is not available")

    assert w8a8_cutlass_ext is not None

    if qweight_packed.dtype is not torch.uint8:
        raise TypeError(f"qweight_packed must be torch.uint8, got {qweight_packed.dtype}")

    if not qweight_packed.is_cuda:
        raise ValueError("qweight_packed must be CUDA tensor for CUTLASS backend")

    qweight_packed = qweight_packed.contiguous()

    key = (
        int(qweight_packed.data_ptr()),
        int(k),
        tuple(qweight_packed.shape),
        qweight_packed.device,
    )

    cache = _QW8_CACHE_BY_PTR.get(key)

    if cache is None or cache.device != qweight_packed.device:
        cache = w8a8_cutlass_ext.unpack_w4_to_w8(qweight_packed, int(k)).contiguous()
        _QW8_CACHE_BY_PTR[key] = cache

    return cache

def _get_or_create_sw_fp32_cache(weight_scale: torch.Tensor, n: int) -> torch.Tensor:
    if not weight_scale.is_cuda:
        raise ValueError("weight_scale must be CUDA tensor for CUTLASS backend")

    key = (
        int(weight_scale.data_ptr()),
        tuple(weight_scale.shape),
        weight_scale.device,
    )

    cache = _SW_FP32_CACHE_BY_PTR.get(key)

    if cache is None or cache.device != weight_scale.device:
        cache = weight_scale.reshape(n).to(torch.float32).contiguous()
        _SW_FP32_CACHE_BY_PTR[key] = cache

    return cache


def _get_or_create_bias_bf16_cache(
    bias: Optional[torch.Tensor],
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    if bias is None:
        return torch.zeros((n,), device=device, dtype=torch.bfloat16), False

    if not bias.is_cuda:
        bias = bias.to(device)

    key = (
        int(bias.data_ptr()),
        tuple(bias.shape),
        bias.device,
    )

    cache = _BIAS_BF16_CACHE_BY_PTR.get(key)

    if cache is None or cache.device != device:
        cache = bias.reshape(n).to(torch.bfloat16).contiguous()
        _BIAS_BF16_CACHE_BY_PTR[key] = cache

    return cache, True

def _as_bf16_bias_or_zero(
    bias: Optional[torch.Tensor],
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    if bias is None:
        return torch.zeros((n,), device=device, dtype=torch.bfloat16), False

    if not bias.is_cuda:
        bias = bias.to(device)

    if bias.dtype is not torch.bfloat16:
        bias = bias.to(torch.bfloat16)

    return bias.contiguous(), True


def _duquant_cutlass_w4a8_cached_linear_from_quantized_act(
    qa: torch.Tensor,
    act_scale: torch.Tensor,
    qweight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_shape_prefix: tuple[int, ...],
) -> torch.Tensor:
    if not _duquant_cutlass_enabled():
        raise RuntimeError("CUTLASS backend is disabled")

    assert w8a8_cutlass_ext is not None

    if qa.dtype is not torch.int8:
        raise TypeError(f"qa must be int8, got {qa.dtype}")

    if qa.dim() != 2:
        raise ValueError(f"qa must be [M, K], got {tuple(qa.shape)}")

    m, k = qa.shape
    n = qweight_packed.shape[0]

    qa = qa.contiguous()
    act_scale = act_scale.reshape(m).to(torch.float32).contiguous()

    qw8_cache = _get_or_create_qw8_cache(qweight_packed, int(k))
    weight_scale = _get_or_create_sw_fp32_cache(weight_scale, int(n))
    bias_bf16, has_bias = _get_or_create_bias_bf16_cache(bias, int(n), qa.device)

    y = w8a8_cutlass_ext.w8a8_scaled_bf16(
        qa,
        act_scale,
        qw8_cache,
        weight_scale,
        bias_bf16,
        has_bias,
    )

    return y.reshape(*out_shape_prefix, n)

def _duquant_cutlass_w4a8_cached_linear_from_bf16_act(
    x2d: torch.Tensor,
    qweight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    clip_ratio: float,
    out_shape_prefix: tuple[int, ...],
) -> torch.Tensor:
    """CUTLASS cached W4A8 fast path from BF16 activation.

    This calls the C++ extension entry:
      x bf16 -> A8 quant -> W8A8 cached GEMM
    inside the extension, reducing Python gap.
    """
    if not _duquant_cutlass_enabled():
        raise RuntimeError("CUTLASS backend is disabled")

    assert w8a8_cutlass_ext is not None

    if x2d.dtype is not torch.bfloat16:
        raise TypeError(f"x2d must be bf16, got {x2d.dtype}")

    if x2d.dim() != 2:
        raise ValueError(f"x2d must be [M,K], got {tuple(x2d.shape)}")

    m, k = x2d.shape
    n = qweight_packed.shape[0]

    x2d = x2d.contiguous()

    qw8_cache = _get_or_create_qw8_cache(qweight_packed, int(k))
    weight_scale = _get_or_create_sw_fp32_cache(weight_scale, int(n))
    bias_bf16, has_bias = _get_or_create_bias_bf16_cache(bias, int(n), x2d.device)

    y = w8a8_cutlass_ext.w8a8_cached_linear_bf16_a8_dynamic(
        x2d,
        qw8_cache,
        weight_scale,
        bias_bf16,
        float(clip_ratio),
        has_bias,
    )

    return y.reshape(*out_shape_prefix, n)
REAL_W4A_BLOCK_M = 32
REAL_W4A_BLOCK_N = 64
REAL_W4A_BLOCK_K = 128

DEFAULT_REAL_BLOCK_CONFIG_BY_BITS: dict[int, tuple[int, int, int]] = {
    4: (32, 64, 128),
    8: (64, 128, 128),
    16: (32, 64, 128),
}
ALLOW_REAL_ACT_ENV_OVERRIDE = False

# DEFAULT_REAL_ACT_CONFIG_BY_BITS: dict[int, tuple[str, float, int]] = {
#     4: ("dynamic_token_group_clip", 0.95, 64),
#     8: ("dynamic_token_group_clip", 0.99, 128),
#     16: ("dynamic_token_group_clip", 0.99, 128),
# }
DEFAULT_REAL_ACT_CONFIG_BY_BITS: dict[int, tuple[str, float, int]] = {
    4: ("dynamic_token_group_clip", 0.95, 64),
    8: ("dynamic_token_clip", 0.99, 128),
    16: ("dynamic_token_clip", 0.99, 128),
}

def _block_power_of_2(x: int) -> int:
    if not _HAS_TRITON:
        return 1 << (int(x) - 1).bit_length()
    return triton.next_power_of_2(int(x))


def _is_power_of_two(x: int) -> bool:
    x = int(x)
    return x > 0 and (x & (x - 1)) == 0


def _get_block_sizes(act_bits: int) -> tuple[int, int, int]:
    act_bits = int(act_bits)
    if act_bits in DEFAULT_REAL_BLOCK_CONFIG_BY_BITS:
        block_m, block_n, block_k = DEFAULT_REAL_BLOCK_CONFIG_BY_BITS[act_bits]
    else:
        block_m, block_n, block_k = REAL_W4A_BLOCK_M, REAL_W4A_BLOCK_N, REAL_W4A_BLOCK_K
    if os.environ.get("OPENPI_REAL_W4A_ALLOW_BLOCK_ENV_OVERRIDE", "0") == "1":
        block_m = int(os.environ.get("OPENPI_REAL_W4A_BLOCK_M", str(block_m)))
        block_n = int(os.environ.get("OPENPI_REAL_W4A_BLOCK_N", str(block_n)))
        block_k = int(os.environ.get("OPENPI_REAL_W4A_BLOCK_K", str(block_k)))

    return int(block_m), int(block_n), int(block_k)


def _get_act_quant_config(act_bits: int) -> tuple[str, float, int]:
    act_bits = int(act_bits)

    if act_bits not in DEFAULT_REAL_ACT_CONFIG_BY_BITS:
        raise ValueError(
            f"No default real activation config for act_bits={act_bits}. "
            f"Available: {sorted(DEFAULT_REAL_ACT_CONFIG_BY_BITS)}"
        )

    default_mode, default_clip_ratio, default_group_size = DEFAULT_REAL_ACT_CONFIG_BY_BITS[act_bits]

    if ALLOW_REAL_ACT_ENV_OVERRIDE:
        mode = os.environ.get("OPENPI_DUQUANT_ACT_QUANT_MODE", default_mode).strip().lower()
        clip_ratio = float(os.environ.get("OPENPI_DUQUANT_ACT_CLIP_RATIO", str(default_clip_ratio)))
        group_size = int(os.environ.get("OPENPI_DUQUANT_ACT_GROUP_SIZE", str(default_group_size)))
    else:
        mode = str(default_mode).strip().lower()
        clip_ratio = float(default_clip_ratio)
        group_size = int(default_group_size)

    aliases = {
        "amax": "dynamic_token_amax",
        "token_amax": "dynamic_token_amax",
        "dynamic_amax": "dynamic_token_amax",

        "clip": "dynamic_token_clip",
        "token_clip": "dynamic_token_clip",

        "group": "dynamic_token_group_amax",
        "group_amax": "dynamic_token_group_amax",
        "token_group": "dynamic_token_group_amax",
        "token_group_amax": "dynamic_token_group_amax",

        "group_clip": "dynamic_token_group_clip",
        "token_group_clip": "dynamic_token_group_clip",
    }
    mode = aliases.get(mode, mode)

    valid_modes = {
        "dynamic_token_amax",
        "dynamic_token_clip",
        "dynamic_token_group_amax",
        "dynamic_token_group_clip",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported real activation quant mode: {mode}")

    if not (0.0 < clip_ratio <= 1.0):
        raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")

    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(f"group_size must be positive even int, got {group_size}")

    if not _is_power_of_two(group_size):
        raise ValueError(f"group_size must be power of two for Triton group kernels, got {group_size}")

    return mode, clip_ratio, group_size


if _HAS_TRITON:

    @triton.jit
    def _round_nearest_signed(x):
        return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


    # ======================================================================
    # Token-wise activation quantization kernels.
    # CLIP_RATIO=1.0 means original token-amax.
    # CLIP_RATIO<1.0 means token-clip.
    # ======================================================================

    @triton.jit
    def _quant_a8_token_amax_kernel(
        X,
        QA,
        SA,
        M: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        CLIP_RATIO: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        mask = offs_k < K

        x = tl.load(X + pid_m * K + offs_k, mask=mask, other=0.0).to(tl.float32)
        abs_x = tl.abs(x)
        p = tl.max(tl.where(mask, abs_x, 0.0), axis=0)

        clip = tl.maximum(p * CLIP_RATIO, 1.0e-12)
        scale = tl.maximum(clip / 127.0, 1.0e-12)

        x_clip = tl.minimum(tl.maximum(x, -clip), clip)
        q = _round_nearest_signed(x_clip / scale)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8)

        tl.store(QA + pid_m * K + offs_k, q, mask=mask)
        tl.store(SA + pid_m, scale)


    @triton.jit
    def _quant_a4_token_amax_pack_kernel(
        X,
        QA_PACK,
        SA,
        M: tl.constexpr,
        K: tl.constexpr,
        KPACK: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_KPACK: tl.constexpr,
        CLIP_RATIO: tl.constexpr,
    ):
        pid_m = tl.program_id(0)

        offs_k = tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(X + pid_m * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        abs_x = tl.abs(x)
        p = tl.max(tl.where(mask_k, abs_x, 0.0), axis=0)

        clip = tl.maximum(p * CLIP_RATIO, 1.0e-12)
        scale = tl.maximum(clip / 7.0, 1.0e-12)

        offs_p = tl.arange(0, BLOCK_KPACK)
        k0 = offs_p * 2
        k1 = k0 + 1

        mask0 = k0 < K
        mask1 = k1 < K

        x0 = tl.load(X + pid_m * K + k0, mask=mask0, other=0.0).to(tl.float32)
        x1 = tl.load(X + pid_m * K + k1, mask=mask1, other=0.0).to(tl.float32)

        x0 = tl.minimum(tl.maximum(x0, -clip), clip)
        x1 = tl.minimum(tl.maximum(x1, -clip), clip)

        q0 = _round_nearest_signed(x0 / scale)
        q1 = _round_nearest_signed(x1 / scale)

        q0 = tl.minimum(tl.maximum(q0, -8.0), 7.0).to(tl.int16)
        q1 = tl.minimum(tl.maximum(q1, -8.0), 7.0).to(tl.int16)

        u0 = (q0 + 8).to(tl.uint8)
        u1 = (q1 + 8).to(tl.uint8)

        packed = u0 | (u1 << 4)

        tl.store(QA_PACK + pid_m * KPACK + offs_p, packed, mask=offs_p < KPACK)
        tl.store(SA + pid_m, scale)


    @triton.jit
    def _quant_a16_token_amax_kernel(
        X,
        XQ,
        SA,
        M: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        CLIP_RATIO: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        mask = offs_k < K

        x = tl.load(X + pid_m * K + offs_k, mask=mask, other=0.0).to(tl.float32)
        abs_x = tl.abs(x)
        p = tl.max(tl.where(mask, abs_x, 0.0), axis=0)

        clip = tl.maximum(p * CLIP_RATIO, 1.0e-12)
        scale = tl.maximum(clip / 32767.0, 1.0e-12)

        x_clip = tl.minimum(tl.maximum(x, -clip), clip)
        q = _round_nearest_signed(x_clip / scale)
        q = tl.minimum(tl.maximum(q, -32768.0), 32767.0)

        y = q * scale

        tl.store(XQ + pid_m * K + offs_k, y, mask=mask)
        tl.store(SA + pid_m, scale)


    # ======================================================================
    # Group-wise activation quantization kernels.
    # One scale per token per hidden group.
    # GROUP_SIZE must be power-of-two and even.
    # ======================================================================

    @triton.jit
    def _quant_a8_token_group_kernel(
        X,
        QA,
        SA_GROUP,
        M: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        CLIP_RATIO: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)

        offs = tl.arange(0, GROUP_SIZE)
        k = pid_g * GROUP_SIZE + offs
        mask = k < K

        x = tl.load(X + pid_m * K + k, mask=mask, other=0.0).to(tl.float32)
        abs_x = tl.abs(x)
        p = tl.max(tl.where(mask, abs_x, 0.0), axis=0)

        clip = tl.maximum(p * CLIP_RATIO, 1.0e-12)
        scale = tl.maximum(clip / 127.0, 1.0e-12)

        x_clip = tl.minimum(tl.maximum(x, -clip), clip)
        q = _round_nearest_signed(x_clip / scale)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8)

        tl.store(QA + pid_m * K + k, q, mask=mask)
        tl.store(SA_GROUP + pid_m * G + pid_g, scale)


    @triton.jit
    def _quant_a4_token_group_pack_kernel(
        X,
        QA_PACK,
        SA_GROUP,
        M: tl.constexpr,
        K: tl.constexpr,
        KPACK: tl.constexpr,
        G: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUP_PACK: tl.constexpr,
        CLIP_RATIO: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)

        offs = tl.arange(0, GROUP_SIZE)
        k = pid_g * GROUP_SIZE + offs
        mask = k < K

        x = tl.load(X + pid_m * K + k, mask=mask, other=0.0).to(tl.float32)
        abs_x = tl.abs(x)
        p = tl.max(tl.where(mask, abs_x, 0.0), axis=0)

        clip = tl.maximum(p * CLIP_RATIO, 1.0e-12)
        scale = tl.maximum(clip / 7.0, 1.0e-12)

        offs_p = tl.arange(0, GROUP_PACK)
        k0 = pid_g * GROUP_SIZE + offs_p * 2
        k1 = k0 + 1

        mask0 = k0 < K
        mask1 = k1 < K

        x0 = tl.load(X + pid_m * K + k0, mask=mask0, other=0.0).to(tl.float32)
        x1 = tl.load(X + pid_m * K + k1, mask=mask1, other=0.0).to(tl.float32)

        x0 = tl.minimum(tl.maximum(x0, -clip), clip)
        x1 = tl.minimum(tl.maximum(x1, -clip), clip)

        q0 = _round_nearest_signed(x0 / scale)
        q1 = _round_nearest_signed(x1 / scale)

        q0 = tl.minimum(tl.maximum(q0, -8.0), 7.0).to(tl.int16)
        q1 = tl.minimum(tl.maximum(q1, -8.0), 7.0).to(tl.int16)

        u0 = (q0 + 8).to(tl.uint8)
        u1 = (q1 + 8).to(tl.uint8)

        packed = u0 | (u1 << 4)
        byte_idx = k0 // 2

        tl.store(QA_PACK + pid_m * KPACK + byte_idx, packed, mask=byte_idx < KPACK)
        tl.store(SA_GROUP + pid_m * G + pid_g, scale)


    @triton.jit
    def _quant_a16_token_group_kernel(
        X,
        XQ,
        SA_GROUP,
        M: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        CLIP_RATIO: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)

        offs = tl.arange(0, GROUP_SIZE)
        k = pid_g * GROUP_SIZE + offs
        mask = k < K

        x = tl.load(X + pid_m * K + k, mask=mask, other=0.0).to(tl.float32)
        abs_x = tl.abs(x)
        p = tl.max(tl.where(mask, abs_x, 0.0), axis=0)

        clip = tl.maximum(p * CLIP_RATIO, 1.0e-12)
        scale = tl.maximum(clip / 32767.0, 1.0e-12)

        x_clip = tl.minimum(tl.maximum(x, -clip), clip)
        q = _round_nearest_signed(x_clip / scale)
        q = tl.minimum(tl.maximum(q, -32768.0), 32767.0)

        y = q * scale

        tl.store(XQ + pid_m * K + k, y, mask=mask)
        tl.store(SA_GROUP + pid_m * G + pid_g, scale)


    # ======================================================================
    # Token-scale GEMM kernels.
    # A4/A8: integer dot, then multiply activation scale and weight scale.
    # A16: activation is quant-dequanted float; dot float activation with int4 weight.
    # ======================================================================

    @triton.jit
    def _w4a8_gemm_restore_bias_kernel(
        QA,
        SA,
        QW,
        SW,
        ROUT,
        BIAS,
        Y,
        M,
        N,
        K: tl.constexpr,
        KPACK: tl.constexpr,
        HAS_RESTORE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_nb = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k

            a = tl.load(
                QA + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0,
            ).to(tl.int8)
            a = tl.where(k[None, :] < K, a, 0)

            byte_idx = k // 2
            packed_w = tl.load(
                QW + offs_n[None, :] * KPACK + byte_idx[:, None],
                mask=(offs_n[None, :] < N) & (byte_idx[:, None] < KPACK) & (k[:, None] < K),
                other=0,
            )

            low_w = packed_w & 0x0F
            high_w = (packed_w >> 4) & 0x0F
            use_high_w = (k[:, None] & 1) == 1
            qwu = tl.where(use_high_w, high_w, low_w)
            w = (qwu.to(tl.int16) - 8).to(tl.int8)
            w = tl.where(k[:, None] < K, w, 0)

            acc += tl.dot(a, w, out_dtype=tl.int32)

        sa = tl.load(SA + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

        out = acc.to(tl.float32) * sa[:, None] * sw[None, :]

        if HAS_RESTORE:
            r_i = tl.arange(0, BLOCK_N)
            r_j = tl.arange(0, BLOCK_N)
            r = tl.load(
                ROUT + pid_nb * BLOCK_N * BLOCK_N + r_i[:, None] * BLOCK_N + r_j[None, :],
                mask=(r_i[:, None] < BLOCK_N) & (r_j[None, :] < BLOCK_N),
                other=0.0,
            ).to(tl.float32)
            out = tl.dot(out, r, input_precision="tf32")

        if HAS_BIAS:
            b = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out = out + b[None, :]

        tl.store(
            Y + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


    @triton.jit
    def _w4a4_gemm_restore_bias_kernel(
        QA_PACK,
        SA,
        QW,
        SW,
        ROUT,
        BIAS,
        Y,
        M,
        N,
        K: tl.constexpr,
        KPACK: tl.constexpr,
        HAS_RESTORE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_nb = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            byte_idx = k // 2

            packed_a = tl.load(
                QA_PACK + offs_m[:, None] * KPACK + byte_idx[None, :],
                mask=(offs_m[:, None] < M) & (byte_idx[None, :] < KPACK) & (k[None, :] < K),
                other=0,
            )
            low_a = packed_a & 0x0F
            high_a = (packed_a >> 4) & 0x0F
            use_high_a = (k[None, :] & 1) == 1
            qau = tl.where(use_high_a, high_a, low_a)
            a = (qau.to(tl.int16) - 8).to(tl.int8)
            a = tl.where(k[None, :] < K, a, 0)

            packed_w = tl.load(
                QW + offs_n[None, :] * KPACK + byte_idx[:, None],
                mask=(offs_n[None, :] < N) & (byte_idx[:, None] < KPACK) & (k[:, None] < K),
                other=0,
            )
            low_w = packed_w & 0x0F
            high_w = (packed_w >> 4) & 0x0F
            use_high_w = (k[:, None] & 1) == 1
            qwu = tl.where(use_high_w, high_w, low_w)
            w = (qwu.to(tl.int16) - 8).to(tl.int8)
            w = tl.where(k[:, None] < K, w, 0)

            acc += tl.dot(a, w, out_dtype=tl.int32)

        sa = tl.load(SA + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

        out = acc.to(tl.float32) * sa[:, None] * sw[None, :]

        if HAS_RESTORE:
            r_i = tl.arange(0, BLOCK_N)
            r_j = tl.arange(0, BLOCK_N)
            r = tl.load(
                ROUT + pid_nb * BLOCK_N * BLOCK_N + r_i[:, None] * BLOCK_N + r_j[None, :],
                mask=(r_i[:, None] < BLOCK_N) & (r_j[None, :] < BLOCK_N),
                other=0.0,
            ).to(tl.float32)
            out = tl.dot(out, r, input_precision="tf32")

        if HAS_BIAS:
            b = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out = out + b[None, :]

        tl.store(
            Y + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


    @triton.jit
    def _w4a16_gemm_restore_bias_kernel(
        XQ,
        QW,
        SW,
        ROUT,
        BIAS,
        Y,
        M,
        N,
        K: tl.constexpr,
        KPACK: tl.constexpr,
        HAS_RESTORE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_nb = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k

            a = tl.load(
                XQ + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            a = tl.where(k[None, :] < K, a, 0.0)

            byte_idx = k // 2
            packed_w = tl.load(
                QW + offs_n[None, :] * KPACK + byte_idx[:, None],
                mask=(offs_n[None, :] < N) & (byte_idx[:, None] < KPACK) & (k[:, None] < K),
                other=0,
            )

            low_w = packed_w & 0x0F
            high_w = (packed_w >> 4) & 0x0F
            use_high_w = (k[:, None] & 1) == 1
            qwu = tl.where(use_high_w, high_w, low_w)
            w = (qwu.to(tl.int16) - 8).to(tl.float32)
            w = tl.where(k[:, None] < K, w, 0.0)

            acc += tl.dot(a, w, input_precision="tf32")

        sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out = acc * sw[None, :]

        if HAS_RESTORE:
            r_i = tl.arange(0, BLOCK_N)
            r_j = tl.arange(0, BLOCK_N)
            r = tl.load(
                ROUT + pid_nb * BLOCK_N * BLOCK_N + r_i[:, None] * BLOCK_N + r_j[None, :],
                mask=(r_i[:, None] < BLOCK_N) & (r_j[None, :] < BLOCK_N),
                other=0.0,
            ).to(tl.float32)
            out = tl.dot(out, r, input_precision="tf32")

        if HAS_BIAS:
            b = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out = out + b[None, :]

        tl.store(
            Y + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


    # ======================================================================
    # Group-scale GEMM kernels.
    # A4/A8: per group partial dot multiplied by sa_group[m,g], then accumulated.
    # ======================================================================

    @triton.jit
    def _w4a8_group_gemm_restore_bias_kernel(
        QA,
        SA_GROUP,
        QW,
        SW,
        ROUT,
        BIAS,
        Y,
        M,
        N,
        K: tl.constexpr,
        KPACK: tl.constexpr,
        G: tl.constexpr,
        HAS_RESTORE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_nb = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, GROUP_SIZE)

        sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g0 in range(0, K, GROUP_SIZE):
            gid = g0 // GROUP_SIZE
            k = g0 + offs_k

            a = tl.load(
                QA + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0,
            ).to(tl.int8)
            a = tl.where(k[None, :] < K, a, 0)

            byte_idx = k // 2
            packed_w = tl.load(
                QW + offs_n[None, :] * KPACK + byte_idx[:, None],
                mask=(offs_n[None, :] < N) & (byte_idx[:, None] < KPACK) & (k[:, None] < K),
                other=0,
            )

            low_w = packed_w & 0x0F
            high_w = (packed_w >> 4) & 0x0F
            use_high_w = (k[:, None] & 1) == 1
            qwu = tl.where(use_high_w, high_w, low_w)
            w = (qwu.to(tl.int16) - 8).to(tl.int8)
            w = tl.where(k[:, None] < K, w, 0)

            acc = tl.dot(a, w, out_dtype=tl.int32)

            sa = tl.load(
                SA_GROUP + offs_m * G + gid,
                mask=offs_m < M,
                other=0.0,
            ).to(tl.float32)

            out += acc.to(tl.float32) * sa[:, None] * sw[None, :]

        if HAS_RESTORE:
            r_i = tl.arange(0, BLOCK_N)
            r_j = tl.arange(0, BLOCK_N)
            r = tl.load(
                ROUT + pid_nb * BLOCK_N * BLOCK_N + r_i[:, None] * BLOCK_N + r_j[None, :],
                mask=(r_i[:, None] < BLOCK_N) & (r_j[None, :] < BLOCK_N),
                other=0.0,
            ).to(tl.float32)
            out = tl.dot(out, r, input_precision="tf32")

        if HAS_BIAS:
            b = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out = out + b[None, :]

        tl.store(
            Y + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


    @triton.jit
    def _w4a4_group_gemm_restore_bias_kernel(
        QA_PACK,
        SA_GROUP,
        QW,
        SW,
        ROUT,
        BIAS,
        Y,
        M,
        N,
        K: tl.constexpr,
        KPACK: tl.constexpr,
        G: tl.constexpr,
        HAS_RESTORE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_nb = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, GROUP_SIZE)

        sw = tl.load(SW + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g0 in range(0, K, GROUP_SIZE):
            gid = g0 // GROUP_SIZE
            k = g0 + offs_k
            byte_idx = k // 2

            packed_a = tl.load(
                QA_PACK + offs_m[:, None] * KPACK + byte_idx[None, :],
                mask=(offs_m[:, None] < M) & (byte_idx[None, :] < KPACK) & (k[None, :] < K),
                other=0,
            )
            low_a = packed_a & 0x0F
            high_a = (packed_a >> 4) & 0x0F
            use_high_a = (k[None, :] & 1) == 1
            qau = tl.where(use_high_a, high_a, low_a)
            a = (qau.to(tl.int16) - 8).to(tl.int8)
            a = tl.where(k[None, :] < K, a, 0)

            packed_w = tl.load(
                QW + offs_n[None, :] * KPACK + byte_idx[:, None],
                mask=(offs_n[None, :] < N) & (byte_idx[:, None] < KPACK) & (k[:, None] < K),
                other=0,
            )
            low_w = packed_w & 0x0F
            high_w = (packed_w >> 4) & 0x0F
            use_high_w = (k[:, None] & 1) == 1
            qwu = tl.where(use_high_w, high_w, low_w)
            w = (qwu.to(tl.int16) - 8).to(tl.int8)
            w = tl.where(k[:, None] < K, w, 0)

            acc = tl.dot(a, w, out_dtype=tl.int32)

            sa = tl.load(
                SA_GROUP + offs_m * G + gid,
                mask=offs_m < M,
                other=0.0,
            ).to(tl.float32)

            out += acc.to(tl.float32) * sa[:, None] * sw[None, :]

        if HAS_RESTORE:
            r_i = tl.arange(0, BLOCK_N)
            r_j = tl.arange(0, BLOCK_N)
            r = tl.load(
                ROUT + pid_nb * BLOCK_N * BLOCK_N + r_i[:, None] * BLOCK_N + r_j[None, :],
                mask=(r_i[:, None] < BLOCK_N) & (r_j[None, :] < BLOCK_N),
                other=0.0,
            ).to(tl.float32)
            out = tl.dot(out, r, input_precision="tf32")

        if HAS_BIAS:
            b = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out = out + b[None, :]

        tl.store(
            Y + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


# ======================================================================
# Public quant wrappers.
# ======================================================================

def quantize_activation_a8_token_amax_triton(
    x2d: torch.Tensor,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available for real W4A8 activation quantization.")

    if x2d.dim() != 2:
        raise ValueError(f"x2d must be 2D, got {tuple(x2d.shape)}")

    m, k = int(x2d.shape[0]), int(x2d.shape[1])
    qa = torch.empty((m, k), device=x2d.device, dtype=torch.int8)
    sa = torch.empty((m,), device=x2d.device, dtype=torch.float32)

    block_k = _block_power_of_2(k)
    _quant_a8_token_amax_kernel[(m,)](
        x2d.contiguous(),
        qa,
        sa,
        m,
        k,
        BLOCK_K=block_k,
        CLIP_RATIO=float(clip_ratio),
        num_warps=8,
    )
    return qa, sa


def quantize_activation_a4_token_amax_pack_triton(
    x2d: torch.Tensor,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available for real W4A4 activation quantization.")

    if x2d.dim() != 2:
        raise ValueError(f"x2d must be 2D, got {tuple(x2d.shape)}")

    m, k = int(x2d.shape[0]), int(x2d.shape[1])
    kpack = (k + 1) // 2

    qa_pack = torch.empty((m, kpack), device=x2d.device, dtype=torch.uint8)
    sa = torch.empty((m,), device=x2d.device, dtype=torch.float32)

    block_k = _block_power_of_2(k)
    block_kpack = _block_power_of_2(kpack)
    _quant_a4_token_amax_pack_kernel[(m,)](
        x2d.contiguous(),
        qa_pack,
        sa,
        m,
        k,
        kpack,
        BLOCK_K=block_k,
        BLOCK_KPACK=block_kpack,
        CLIP_RATIO=float(clip_ratio),
        num_warps=8,
    )
    return qa_pack, sa


def quantize_activation_a16_token_amax_triton(
    x2d: torch.Tensor,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available for real W4A16 activation quantization.")

    if x2d.dim() != 2:
        raise ValueError(f"x2d must be 2D, got {tuple(x2d.shape)}")

    m, k = int(x2d.shape[0]), int(x2d.shape[1])
    xq = torch.empty_like(x2d)
    sa = torch.empty((m,), device=x2d.device, dtype=torch.float32)

    block_k = _block_power_of_2(k)
    _quant_a16_token_amax_kernel[(m,)](
        x2d.contiguous(),
        xq,
        sa,
        m,
        k,
        BLOCK_K=block_k,
        CLIP_RATIO=float(clip_ratio),
        num_warps=8,
    )
    return xq, sa


def quantize_activation_a8_token_group_triton(
    x2d: torch.Tensor,
    *,
    group_size: int,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available for real W4A8 group activation quantization.")

    if x2d.dim() != 2:
        raise ValueError(f"x2d must be 2D, got {tuple(x2d.shape)}")

    if group_size <= 0 or group_size % 2 != 0 or not _is_power_of_two(group_size):
        raise ValueError(f"group_size must be positive even power-of-two int, got {group_size}")

    m, k = int(x2d.shape[0]), int(x2d.shape[1])
    g = triton.cdiv(k, group_size)

    qa = torch.empty((m, k), device=x2d.device, dtype=torch.int8)
    sa_group = torch.empty((m, g), device=x2d.device, dtype=torch.float32)

    _quant_a8_token_group_kernel[(m, g)](
        x2d.contiguous(),
        qa,
        sa_group,
        m,
        k,
        g,
        GROUP_SIZE=int(group_size),
        CLIP_RATIO=float(clip_ratio),
        num_warps=4,
    )
    return qa, sa_group


def quantize_activation_a4_token_group_pack_triton(
    x2d: torch.Tensor,
    *,
    group_size: int,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available for real W4A4 group activation quantization.")

    if x2d.dim() != 2:
        raise ValueError(f"x2d must be 2D, got {tuple(x2d.shape)}")

    if group_size <= 0 or group_size % 2 != 0 or not _is_power_of_two(group_size):
        raise ValueError(f"group_size must be positive even power-of-two int, got {group_size}")

    m, k = int(x2d.shape[0]), int(x2d.shape[1])
    kpack = (k + 1) // 2
    g = triton.cdiv(k, group_size)
    group_pack = group_size // 2

    qa_pack = torch.empty((m, kpack), device=x2d.device, dtype=torch.uint8)
    sa_group = torch.empty((m, g), device=x2d.device, dtype=torch.float32)

    _quant_a4_token_group_pack_kernel[(m, g)](
        x2d.contiguous(),
        qa_pack,
        sa_group,
        m,
        k,
        kpack,
        g,
        GROUP_SIZE=int(group_size),
        GROUP_PACK=int(group_pack),
        CLIP_RATIO=float(clip_ratio),
        num_warps=4,
    )
    return qa_pack, sa_group


def quantize_activation_a16_token_group_triton(
    x2d: torch.Tensor,
    *,
    group_size: int,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available for real W4A16 group activation quantization.")

    if x2d.dim() != 2:
        raise ValueError(f"x2d must be 2D, got {tuple(x2d.shape)}")

    if group_size <= 0 or group_size % 2 != 0 or not _is_power_of_two(group_size):
        raise ValueError(f"group_size must be positive even power-of-two int, got {group_size}")

    m, k = int(x2d.shape[0]), int(x2d.shape[1])
    g = triton.cdiv(k, group_size)

    xq = torch.empty_like(x2d)
    sa_group = torch.empty((m, g), device=x2d.device, dtype=torch.float32)

    _quant_a16_token_group_kernel[(m, g)](
        x2d.contiguous(),
        xq,
        sa_group,
        m,
        k,
        g,
        GROUP_SIZE=int(group_size),
        CLIP_RATIO=float(clip_ratio),
        num_warps=4,
    )
    return xq, sa_group


# ======================================================================
# Main entry.
# ======================================================================

def real_w4a_linear_triton(
    x_t: torch.Tensor,
    qweight_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    act_bits: int,
    in_features: int,
    out_features: int,
    r_out_stack: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    block_out_size: int,
) -> torch.Tensor:
    global _CUTLASS_HIT_PRINTED
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available for real W4A backend.")

    if x_t.shape[-1] != in_features:
        raise ValueError(f"x last dim={x_t.shape[-1]} but in_features={in_features}")

    if qweight_packed.dtype != torch.uint8:
        raise TypeError(f"qweight_packed must be uint8, got {qweight_packed.dtype}")

    act_bits = int(act_bits)
    if act_bits not in {4, 8, 16}:
        raise ValueError(f"real_w4a_linear_triton only supports act_bits=4, 8, or 16, got {act_bits}")

    orig_shape = x_t.shape[:-1]
    x2d = x_t.reshape(-1, in_features).contiguous()

    m = int(x2d.shape[0])
    n = int(out_features)
    k = int(in_features)
    kpack = (k + 1) // 2

    if qweight_packed.shape != (n, kpack):
        raise ValueError(f"qweight_packed shape mismatch: expected {(n, kpack)}, got {tuple(qweight_packed.shape)}")

    block_m, block_n_default, block_k = _get_block_sizes(act_bits)
    has_restore = r_out_stack is not None
    if has_restore:
        block_n = int(block_out_size)
    else:
        block_n = int(block_n_default)

    if block_k > k:
        block_k = 64 if k >= 64 else 32

    act_quant_mode, clip_ratio, group_size = _get_act_quant_config(act_bits)
    use_group = act_quant_mode in {"dynamic_token_group_amax", "dynamic_token_group_clip"}
    use_clip = act_quant_mode in {"dynamic_token_clip", "dynamic_token_group_clip"}

    if not use_clip:
        clip_ratio = 1.0

    if use_group:
        if group_size <= 0 or group_size % 2 != 0 or not _is_power_of_two(group_size):
            raise ValueError(f"group_size must be positive even power-of-two int, got {group_size}")
    if (
        act_bits == 8
        and not use_group
        and r_out_stack is None
        and not _duquant_bitblas_w4a8_enabled()
        and _duquant_cutlass_enabled()
        and x_t.dtype == torch.bfloat16
        and qweight_packed.is_cuda
        and qweight_packed.dtype == torch.uint8
    ):
        if not _CUTLASS_HIT_PRINTED:
            print("[duquant-cutlass] HIT W4A8 cached fused-A8 backend")
            _CUTLASS_HIT_PRINTED = True

        return _duquant_cutlass_w4a8_cached_linear_from_bf16_act(
            x2d=x2d,
            qweight_packed=qweight_packed.contiguous(),
            weight_scale=weight_scales,
            bias=bias,
            clip_ratio=clip_ratio,
            out_shape_prefix=tuple(orig_shape),
        )
    y2d = torch.empty((m, n), device=x_t.device, dtype=x_t.dtype)

    has_restore = r_out_stack is not None
    if has_restore:
        if r_out_stack.device != x_t.device:
            r_out_stack = r_out_stack.to(device=x_t.device)
        if r_out_stack.dtype != x_t.dtype:
            r_out_stack = r_out_stack.to(dtype=x_t.dtype)
        r_out_stack = r_out_stack.contiguous()
    else:
        r_out_stack = torch.empty((1, block_n, block_n), device=x_t.device, dtype=x_t.dtype)

    has_bias = bias is not None
    if has_bias:
        bias = bias.to(device=x_t.device, dtype=x_t.dtype).contiguous()
    else:
        bias = torch.empty((n,), device=x_t.device, dtype=x_t.dtype)

    weight_scales = weight_scales.to(device=x_t.device).contiguous()

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

    if use_group:
        g = triton.cdiv(k, group_size)

        if act_bits == 8:
            qa, sa_group = quantize_activation_a8_token_group_triton(
                x2d,
                group_size=group_size,
                clip_ratio=clip_ratio,
            )
            _w4a8_group_gemm_restore_bias_kernel[grid](
                qa,
                sa_group,
                qweight_packed.contiguous(),
                weight_scales,
                r_out_stack,
                bias,
                y2d,
                m,
                n,
                k,
                kpack,
                g,
                HAS_RESTORE=has_restore,
                HAS_BIAS=has_bias,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                GROUP_SIZE=group_size,
                num_warps=4,
                num_stages=3,
            )

        elif act_bits == 4:
            qa_pack, sa_group = quantize_activation_a4_token_group_pack_triton(
                x2d,
                group_size=group_size,
                clip_ratio=clip_ratio,
            )
            _w4a4_group_gemm_restore_bias_kernel[grid](
                qa_pack,
                sa_group,
                qweight_packed.contiguous(),
                weight_scales,
                r_out_stack,
                bias,
                y2d,
                m,
                n,
                k,
                kpack,
                g,
                HAS_RESTORE=has_restore,
                HAS_BIAS=has_bias,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                GROUP_SIZE=group_size,
                num_warps=4,
                num_stages=3,
            )

        else:
            xq, _sa_group = quantize_activation_a16_token_group_triton(
                x2d,
                group_size=group_size,
                clip_ratio=clip_ratio,
            )
            _w4a16_gemm_restore_bias_kernel[grid](
                xq,
                qweight_packed.contiguous(),
                weight_scales,
                r_out_stack,
                bias,
                y2d,
                m,
                n,
                k,
                kpack,
                HAS_RESTORE=has_restore,
                HAS_BIAS=has_bias,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=3,
            )

    else:
        if act_bits == 8:
            qa, sa = quantize_activation_a8_token_amax_triton(
                x2d,
                clip_ratio=clip_ratio,
            )
            if (
                _duquant_bitblas_w4a8_enabled()
                and not has_restore
                and x_t.dtype == torch.bfloat16
                and qweight_packed.is_cuda
                and qweight_packed.dtype == torch.uint8
            ):
                return _duquant_bitblas_w4a8_linear_from_quantized_act(
                    qa=qa,
                    act_scale=sa,
                    qweight_packed=qweight_packed.contiguous(),
                    weight_scale=weight_scales,
                    bias=(bias if has_bias else None),
                    out_shape_prefix=tuple(orig_shape),
                )

            if (
                _duquant_cutlass_enabled()
                and not has_restore
                and x_t.dtype == torch.bfloat16
                and qweight_packed.is_cuda
                and qweight_packed.dtype == torch.uint8
            ):
                if not _CUTLASS_HIT_PRINTED:
                    print("[duquant-cutlass] HIT W4A8 cached backend")
                    _CUTLASS_HIT_PRINTED = True

                return _duquant_cutlass_w4a8_cached_linear_from_quantized_act(
                    qa=qa,
                    act_scale=sa,
                    qweight_packed=qweight_packed.contiguous(),
                    weight_scale=weight_scales,
                    bias=(bias if has_bias else None),
                    out_shape_prefix=tuple(orig_shape),
                )
            _w4a8_gemm_restore_bias_kernel[grid](
                qa,
                sa,
                qweight_packed.contiguous(),
                weight_scales,
                r_out_stack,
                bias,
                y2d,
                m,
                n,
                k,
                kpack,
                HAS_RESTORE=has_restore,
                HAS_BIAS=has_bias,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=3,
            )

        elif act_bits == 4:
            qa_pack, sa = quantize_activation_a4_token_amax_pack_triton(
                x2d,
                clip_ratio=clip_ratio,
            )
            _w4a4_gemm_restore_bias_kernel[grid](
                qa_pack,
                sa,
                qweight_packed.contiguous(),
                weight_scales,
                r_out_stack,
                bias,
                y2d,
                m,
                n,
                k,
                kpack,
                HAS_RESTORE=has_restore,
                HAS_BIAS=has_bias,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=3,
            )

        else:
            xq, _sa = quantize_activation_a16_token_amax_triton(
                x2d,
                clip_ratio=clip_ratio,
            )
            _w4a16_gemm_restore_bias_kernel[grid](
                xq,
                qweight_packed.contiguous(),
                weight_scales,
                r_out_stack,
                bias,
                y2d,
                m,
                n,
                k,
                kpack,
                HAS_RESTORE=has_restore,
                HAS_BIAS=has_bias,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=3,
            )

    return y2d.reshape(*orig_shape, n)

'''
# 原始 token amax
OPENPI_DUQUANT_ACT_BACKEND=real \
OPENPI_DUQUANT_ACT_QUANT_MODE=dynamic_token_amax

# token clip
OPENPI_DUQUANT_ACT_BACKEND=real \
OPENPI_DUQUANT_ACT_QUANT_MODE=dynamic_token_clip \
OPENPI_DUQUANT_ACT_CLIP_RATIO=0.95

# group amax
OPENPI_DUQUANT_ACT_BACKEND=real \
OPENPI_DUQUANT_ACT_QUANT_MODE=dynamic_token_group_amax \
OPENPI_DUQUANT_ACT_GROUP_SIZE=128

# group clip
OPENPI_DUQUANT_ACT_BACKEND=real \
OPENPI_DUQUANT_ACT_QUANT_MODE=dynamic_token_group_clip \
OPENPI_DUQUANT_ACT_GROUP_SIZE=128 \
OPENPI_DUQUANT_ACT_CLIP_RATIO=0.95
'''