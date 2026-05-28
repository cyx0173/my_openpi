from __future__ import annotations
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import inspect
import time
import traceback

import torch
import bitblas


def bench_cuda(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)

    st.record()
    for _ in range(iters):
        fn()
    ed.record()

    torch.cuda.synchronize()
    return float(st.elapsed_time(ed) / iters)


def make_config(**kwargs):
    sig = inspect.signature(bitblas.MatmulConfig)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return bitblas.MatmulConfig(**allowed)


def maybe_make_matmul(config):
    # Different BitBLAS versions have slightly different Matmul signatures.
    try:
        return bitblas.Matmul(config=config, enable_tuning=False)
    except TypeError:
        try:
            return bitblas.Matmul(config=config)
        except TypeError:
            return bitblas.Matmul(config)


def run_one(M: int, N: int, K: int, cfg_kwargs: dict, tag: str) -> bool:
    print("=" * 120)
    print(f"[TRY] {tag} M={M} N={N} K={K}")
    print("cfg =", cfg_kwargs)

    try:
        torch.manual_seed(0)

        config = make_config(
            M=M,
            N=N,
            K=K,
            layout="nt",
            with_bias=False,
            group_size=None,
            with_scaling=False,
            with_zeros=False,
            zeros_mode=None,
            **cfg_kwargs,
        )

        t0 = time.perf_counter()
        matmul = maybe_make_matmul(config)
        torch.cuda.synchronize()
        print(f"[BUILD] dt={time.perf_counter() - t0:.2f}s")

        A_dtype = cfg_kwargs["A_dtype"]
        W_dtype = cfg_kwargs["W_dtype"]

        if A_dtype == "int8":
            A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
        elif A_dtype == "float16":
            A = torch.randn((M, K), device="cuda", dtype=torch.float16)
        else:
            raise ValueError(f"unsupported A_dtype={A_dtype}")

        if W_dtype == "int4":
            W = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8)
            W_ref = W.to(torch.int32)
        elif W_dtype == "uint4":
            # For uint4, BitBLAS generally interprets weight values as unsigned.
            W = torch.randint(0, 16, (N, K), device="cuda", dtype=torch.int8)
            W_ref = W.to(torch.int32)
        else:
            raise ValueError(f"unsupported W_dtype={W_dtype}")

        t0 = time.perf_counter()
        W_t = matmul.transform_weight(W)
        torch.cuda.synchronize()
        print(f"[TRANSFORM] dt={time.perf_counter() - t0:.2f}s shape={tuple(W_t.shape)} dtype={W_t.dtype}")

        def fn():
            return matmul(A, W_t)

        Y = fn()
        torch.cuda.synchronize()
        print(f"[OUT] shape={tuple(Y.shape)} dtype={Y.dtype}")

        if A_dtype == "int8":
            ref = A.to(torch.float32) @ W_ref.to(torch.float32).t()
            diff = (Y.to(torch.float32) - ref.to(torch.float32)).abs()
        else:
            ref = A.to(torch.float32) @ W.to(torch.float32).t()
            diff = (Y.to(torch.float32) - ref.to(torch.float32)).abs()

        print(
            f"[CHECK] mean={diff.mean().item():.6g} "
            f"p95={torch.quantile(diff.flatten(), 0.95).item():.6g} "
            f"max={diff.max().item():.6g}"
        )
        print("Y[0,:8]   =", Y[0, :8].detach().cpu().tolist())
        print("ref[0,:8] =", ref[0, :8].detach().cpu().tolist())

        ms = bench_cuda(fn, warmup=10, iters=50)
        print(f"[BENCH] {tag} = {ms:.4f} ms")

        # For int32 output we expect exact. For float output, allow small mismatch.
        ok = diff.max().item() == 0 if str(Y.dtype) in {"torch.int32", "torch.int64"} else diff.max().item() < 1.0
        print(f"[RESULT] ok={ok}")
        return bool(ok)

    except Exception as e:
        print(f"[FAIL] {tag}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return False


def main():
    print("bitblas =", bitblas)
    print("bitblas file =", getattr(bitblas, "__file__", None))
    print("bitblas version =", getattr(bitblas, "__version__", None))
    print("MatmulConfig signature =", inspect.signature(bitblas.MatmulConfig))

    candidates = [
        {
            "A_dtype": "int8",
            "W_dtype": "int4",
            "accum_dtype": "int32",
            "out_dtype": "int32",
        },
        {
            "A_dtype": "int8",
            "W_dtype": "int4",
            "accum_dtype": "int32",
            "out_dtype": "float32",
        },
        {
            "A_dtype": "int8",
            "W_dtype": "uint4",
            "accum_dtype": "int32",
            "out_dtype": "int32",
        },
        {
            "A_dtype": "int8",
            "W_dtype": "uint4",
            "accum_dtype": "int32",
            "out_dtype": "float32",
        },
        # sanity: known BitBLAS style path from docs
        {
            "A_dtype": "float16",
            "W_dtype": "int4",
            "accum_dtype": "float16",
            "out_dtype": "float16",
        },
    ]

    good = []

    for i, cfg in enumerate(candidates):
        ok = run_one(64, 128, 128, cfg, f"small_cfg{i}")
        if ok:
            good.append((i, cfg))

    print("=" * 120)
    print("[GOOD SMALL CONFIGS]")
    for i, cfg in good:
        print(i, cfg)

    if not good:
        print("No usable BitBLAS W4A8 config found on small shape.")
        return

    print("=" * 120)
    print("[TARGET SHAPE TEST]")
    for i, cfg in good:
        run_one(256, 4304, 1152, cfg, f"target_cfg{i}")


if __name__ == "__main__":
    main()