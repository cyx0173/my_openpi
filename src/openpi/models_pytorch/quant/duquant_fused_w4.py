from __future__ import annotations
import os
import gc
import time
from typing import Optional

import torch
from torch import nn
from openpi.models_pytorch.quant.duquant_real_w4a import real_w4a_linear_triton
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
    fake_quantize_act_dynamic,
)
from openpi.models_pytorch.quant.duquant_packed_w4 import (
    DuQuantPackedW4Linear,
    _get_parent_module_and_attr,
)


# Fixed production tile parameters.
# Keep these hardcoded for clean benchmarking.
FUSED_W4_BLOCK_M = 32
FUSED_W4_BLOCK_K = 128


# --------------------------------------------------------------------------------------
# R stack builders
# --------------------------------------------------------------------------------------


def _make_r_in_stack(module: nn.Module) -> Optional[torch.Tensor]:
    """Build R_in_stack with shape [num_blocks, block_size, block_size]."""
    if not hasattr(module, "_R_in_block_indices"):
        return None

    indices = list(getattr(module, "_R_in_block_indices", []))
    if not indices:
        return None

    block = int(getattr(module, "_block_size"))
    in_features = int(getattr(module, "in_features"))
    num_blocks = (in_features + block - 1) // block

    first = getattr(module, f"_R_in_{indices[0]}")
    stack = torch.zeros((num_blocks, block, block), dtype=first.dtype, device=first.device)

    for b in range(num_blocks):
        diag = torch.arange(block, device=first.device)
        stack[b, diag, diag] = 1.0

    for b in indices:
        R = getattr(module, f"_R_in_{b}")
        rows = min(block, R.shape[0])
        cols = min(block, R.shape[1])
        stack[b, :rows, :cols] = R[:rows, :cols]

    return stack.contiguous()


def _make_r_out_stack(module: nn.Module) -> Optional[torch.Tensor]:
    """Build R_out_stack with shape [num_blocks, block_out_size, block_out_size]."""
    if not hasattr(module, "_R_out_block_indices"):
        return None

    indices = list(getattr(module, "_R_out_block_indices", []))
    if not indices:
        return None

    block = int(getattr(module, "_block_out_size"))
    out_features = int(getattr(module, "out_features"))
    num_blocks = (out_features + block - 1) // block

    first = getattr(module, f"_R_out_{indices[0]}")
    stack = torch.zeros((num_blocks, block, block), dtype=first.dtype, device=first.device)

    for b in range(num_blocks):
        diag = torch.arange(block, device=first.device)
        stack[b, diag, diag] = 1.0

    for b in indices:
        R = getattr(module, f"_R_out_{b}")
        rows = min(block, R.shape[0])
        cols = min(block, R.shape[1])
        stack[b, :rows, :cols] = R[:rows, :cols]

    return stack.contiguous()


# --------------------------------------------------------------------------------------
# Batched input transform
# --------------------------------------------------------------------------------------


def apply_input_transform_batched_fast(
    x: torch.Tensor,
    perm_cache: Optional[torch.Tensor],
    r_in_stack: Optional[torch.Tensor],
    block_size: int,
    in_features: int,
) -> torch.Tensor:
    """
    DuQuant input transform fast path.

    Equivalent to:
        x -> optional permutation -> blockwise x_block @ R_in_block

    The block rotation is executed as one torch.bmm instead of many small matmuls.
    """
    if perm_cache is not None:
        if perm_cache.device != x.device:
            perm_cache = perm_cache.to(device=x.device)
        x = x.index_select(dim=-1, index=perm_cache)

    if r_in_stack is None:
        return x

    orig_shape = x.shape
    if orig_shape[-1] != in_features:
        raise ValueError(f"x last dim={orig_shape[-1]} but in_features={in_features}")

    x2d = x.reshape(-1, in_features).contiguous()
    m = x2d.shape[0]

    num_blocks = int(r_in_stack.shape[0])
    padded_features = num_blocks * int(block_size)

    if padded_features < in_features:
        raise ValueError(
            f"invalid padded_features={padded_features}, in_features={in_features}, "
            f"num_blocks={num_blocks}, block_size={block_size}"
        )

    if padded_features != in_features:
        x2d = torch.nn.functional.pad(x2d, (0, padded_features - in_features))

    # [M, num_blocks, block] -> [num_blocks, M, block]
    xb = x2d.reshape(m, num_blocks, block_size).transpose(0, 1).contiguous()

    r = r_in_stack
    if r.device != xb.device:
        r = r.to(device=xb.device)
    if r.dtype != xb.dtype:
        r = r.to(dtype=xb.dtype)

    # [num_blocks, M, block] @ [num_blocks, block, block]
    yb = torch.bmm(xb, r)
    y2d = yb.transpose(0, 1).contiguous().reshape(m, padded_features)

    if padded_features != in_features:
        y2d = y2d[:, :in_features]

    return y2d.reshape(orig_shape)


# --------------------------------------------------------------------------------------
# Triton kernel: packed W4 GEMM + output restore + bias
# --------------------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _w4_gemm_restore_bias_kernel(
        X,
        QW,
        S,
        ROUT,
        BIAS,
        Y,
        M,
        N,
        K: tl.constexpr,
        KPACK,
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
                X + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float16)

            byte_idx = k // 2

            # QW layout: [N, ceil(K / 2)]
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
            q = (q_u.to(tl.float32) - 8.0).to(tl.float16)

            scales = tl.load(S + offs_n, mask=offs_n < N, other=0.0).to(tl.float16)
            w = q * scales[None, :]

            acc += tl.dot(a, w)

        # Output restore: y_block @ R_out
        if HAS_RESTORE:
            r_i = tl.arange(0, BLOCK_N)
            r_j = tl.arange(0, BLOCK_N)
            r = tl.load(
                ROUT + pid_nb * BLOCK_N * BLOCK_N + r_i[:, None] * BLOCK_N + r_j[None, :],
                mask=(r_i[:, None] < BLOCK_N) & (r_j[None, :] < BLOCK_N),
                other=0.0,
            ).to(tl.float32)
            acc = tl.dot(acc, r)

        if HAS_BIAS:
            b = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc = acc + b[None, :]

        tl.store(
            Y + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def fused_w4_linear_triton(
    x_t: torch.Tensor,
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    r_out_stack: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    block_out_size: int,
) -> torch.Tensor:
    """
    Fused W4 path after activation handling:
        y_raw = X @ dequant(W4).T
        y = y_raw @ R_out   if restore exists
        y = y + bias        if bias exists

    For A4/A8, X has already been fake-quantized/dequantized before this call.
    For A16, X is the normal transformed activation.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available for fused W4 backend.")

    if x_t.shape[-1] != in_features:
        raise ValueError(f"x last dim={x_t.shape[-1]} but in_features={in_features}")

    if qweight_packed.dtype != torch.uint8:
        raise TypeError(f"qweight_packed must be uint8, got {qweight_packed.dtype}")

    orig_shape = x_t.shape[:-1]
    x2d = x_t.reshape(-1, in_features).contiguous()

    m = int(x2d.shape[0])
    n = int(out_features)
    k = int(in_features)
    kpack = (k + 1) // 2

    if qweight_packed.shape != (n, kpack):
        raise ValueError(f"qweight_packed shape mismatch: expected {(n, kpack)}, got {tuple(qweight_packed.shape)}")

    block_n = int(block_out_size)
    block_m = FUSED_W4_BLOCK_M
    block_k = FUSED_W4_BLOCK_K

    if k < block_k:
        block_k = 64 if k >= 64 else 32

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

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

    _w4_gemm_restore_bias_kernel[grid](
        x2d,
        qweight_packed.contiguous(),
        scales.contiguous(),
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


# --------------------------------------------------------------------------------------
# Fused module
# --------------------------------------------------------------------------------------


class DuQuantFusedW4Linear(DuQuantPackedW4Linear):
    """
    Fused W4 module.

    A4/A8/A16 all use:
        batched input transform
        optional activation fake quant for A4/A8
        fused W4 GEMM + output restore + bias

    This preserves A4/A8 fake quant semantics while avoiding the parent path's
    old input transform and output restore overhead.
    """

    def __init__(self, base: DuQuantLinear):
        super().__init__(base)

        if not hasattr(self, "_weight_compat_storage"):
            dtype = self._w_scales.dtype
            device = self._w_scales.device
            self.register_buffer(
                "_weight_compat_storage",
                torch.empty(1, dtype=dtype, device=device),
                persistent=False,
            )

        r_in_stack = _make_r_in_stack(self)
        if r_in_stack is not None:
            target_dtype = self._w_scales.dtype
            if r_in_stack.dtype != target_dtype:
                r_in_stack = r_in_stack.to(dtype=target_dtype)
            self.register_buffer("_r_in_stack", r_in_stack.contiguous(), persistent=False)
        else:
            self._r_in_stack = None

        r_out_stack = None
        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            r_out_stack = _make_r_out_stack(self)

        if r_out_stack is not None:
            target_dtype = self._w_scales.dtype
            if r_out_stack.dtype != target_dtype:
                r_out_stack = r_out_stack.to(dtype=target_dtype)
            self.register_buffer("_r_out_stack", r_out_stack.contiguous(), persistent=False)
        else:
            self._r_out_stack = None

    @property
    def weight(self):
        # Compatibility-only fake weight tensor. Does not allocate full float weight.
        return self._weight_compat_storage.as_strided((self.out_features, self.in_features), (0, 0))

    def _input_transform(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return apply_input_transform_batched_fast(
                x,
                self._perm_cache,
                self._r_in_stack,
                self._block_size,
                self.in_features,
            )
        except Exception:
            # Silent safe fallback to original implementation.
            return apply_input_transform_optimized(
                x,
                self.pack,
                self._perm_cache,
                self._get_R_in_cache(),
                self._block_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self._input_transform(x)
        act_backend = os.environ.get("OPENPI_DUQUANT_ACT_BACKEND", "fake").strip().lower()

        if act_backend in {"real", "real_w4a", "kernel"} and self.act_bits in {4, 8 ,16}:
            if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
                bias = self._bias
                r_out_stack = self._r_out_stack
            else:
                bias = self._bias_rot if getattr(self, "_bias_rot", None) is not None else self._bias
                r_out_stack = None

            return real_w4a_linear_triton(
                x_t,
                self._qweight_packed,
                self._w_scales,
                act_bits=self.act_bits,
                in_features=self.in_features,
                out_features=self.out_features,
                r_out_stack=r_out_stack,
                bias=bias,
                block_out_size=self._block_out_size,
            )
        
        if self.act_bits > 0 and self.act_bits <= 16:
            act_mode = os.environ.get(
                "OPENPI_DUQUANT_ACT_QUANT_MODE",
                "dynamic_token_amax",
            ).strip().lower()
            x_t = fake_quantize_act_dynamic(
                x_t,
                self.act_bits,
                mode=act_mode,
                label="activation_forward",
            )

        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            bias = self._bias
            r_out_stack = self._r_out_stack
        else:
            bias = self._bias_rot if getattr(self, "_bias_rot", None) is not None else self._bias
            r_out_stack = None

        return fused_w4_linear_triton(
            x_t,
            self._qweight_packed,
            self._w_scales,
            in_features=self.in_features,
            out_features=self.out_features,
            r_out_stack=r_out_stack,
            bias=bias,
            block_out_size=self._block_out_size,
        )


def convert_duquant_to_fused_w4(model: nn.Module) -> int:
    """Convert all DuQuantLinear modules into DuQuantFusedW4Linear."""
    targets = [name for name, module in model.named_modules() if isinstance(module, DuQuantLinear)]

    print(f"[FUSED-W4] Found DuQuantLinear layers: {len(targets)}", flush=True)

    replaced = 0
    t_all = time.perf_counter()

    for i, name in enumerate(targets, start=1):
        parent, attr = _get_parent_module_and_attr(model, name)
        old = getattr(parent, attr)

        if not isinstance(old, DuQuantLinear):
            continue

        new = DuQuantFusedW4Linear(old)
        setattr(parent, attr, new)

        del old
        replaced += 1

        if i % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    print(
        f"[FUSED-W4] Converted layers: {replaced}, total_dt={time.perf_counter() - t_all:.2f}s",
        flush=True,
    )

    return replaced


def count_fused_w4_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, DuQuantFusedW4Linear))
