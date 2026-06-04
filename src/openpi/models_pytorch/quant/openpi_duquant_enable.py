from __future__ import annotations

import os
import time

import torch


def default_duquant_selective_patterns() -> tuple[str, str]:
    """Default OpenPI DuQuant selective layout.

    Quantize Paligemma model and action expert MLP / action input-time
    projections. Keep embeddings, norms, and action_out_proj out.
    """
    common_exclude = (
        r"lm_head"
        r"|embed_tokens"
        r"|rotary_emb"
        r"|\.norm"
        r"|\.input_layernorm"
        r"|\.post_attention_layernorm"
    )

    vlm_include = r"paligemma_with_expert\.paligemma\.model\..*"

    action_mlp_io_include = (
        r"paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$"
        r"|^action_in_proj$"
        r"|^time_mlp_in$"
        r"|^time_mlp_out$"
    )

    include = f"({vlm_include})|({action_mlp_io_include})"

    exclude = (
        common_exclude
        + r"|^action_out_proj$"
    )

    return include, exclude


def _setdefault_env(name: str, value: object) -> None:
    os.environ.setdefault(name, str(value))


def _configure_common_env() -> None:
    _setdefault_env("OPENPI_DUQUANT_WBITS_DEFAULT", "4")
    _setdefault_env("OPENPI_DUQUANT_ABITS", "4")
    _setdefault_env("OPENPI_DUQUANT_BLOCK", "128")
    _setdefault_env("OPENPI_DUQUANT_BLOCK_OUT", os.environ.get("OPENPI_DUQUANT_BLOCK", "128"))
    _setdefault_env("OPENPI_DUQUANT_ALPHA", "0.6")
    _setdefault_env("OPENPI_DUQUANT_LAC", "0.9")
    _setdefault_env("OPENPI_DUQUANT_SWC", "0.8")
    _setdefault_env("OPENPI_DUQUANT_PERMUTE", "1")
    _setdefault_env("OPENPI_DUQUANT_PERMUTATION_TIMES", "1")
    _setdefault_env("OPENPI_DUQUANT_ROW_ROT", "restore")
    _setdefault_env("OPENPI_DUQUANT_ACT_GROUP_SIZE", "0")
    _setdefault_env("OPENPI_DUQUANT_REQUIRE_CALIB", "1")
    _setdefault_env(
        "OPENPI_DUQUANT_PACKDIR",
        "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant/duquant_packed",
    )
    _setdefault_env(
        "OPENPI_DUQUANT_INT4_CACHE_DIR",
        "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant/duquant_int4_cache",
    )
    _setdefault_env("OPENPI_DUQUANT_INT4_CACHE", "1")
    _setdefault_env("OPENPI_DUQUANT_PACKED_BACKEND", "kernel")
    _setdefault_env(
        "TRITON_CACHE_DIR",
        "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant/.triton_cache",
    )
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

    include, exclude = default_duquant_selective_patterns()
    os.environ["OPENPI_DUQUANT_INCLUDE"] = os.environ.get("OPENPI_DUQUANT_INCLUDE", include)
    os.environ["OPENPI_DUQUANT_EXCLUDE"] = os.environ.get("OPENPI_DUQUANT_EXCLUDE", exclude)

    if not os.environ.get("OPENPI_DUQUANT_CALIB_PATH"):
        raise RuntimeError(
            "OPENPI_DUQUANT_CALIB_PATH is required. Run OpenPI DuQuant activation "
            "calibration first; do not run W4A* without calibration."
        )


def enable_openpi_duquant_reference(model):
    """Enable reference DuQuant path: nn.Linear -> DuQuantLinear."""
    from openpi.models_pytorch.quant.duquant_layers import enable_openpi_duquant_all_linears

    _configure_common_env()
    print(
        "[OPENPI-DUQUANT] enabling reference DuQuantLinear "
        f"W{os.environ['OPENPI_DUQUANT_WBITS_DEFAULT']}A{os.environ['OPENPI_DUQUANT_ABITS']} "
        f"calib={os.environ['OPENPI_DUQUANT_CALIB_PATH']} "
        f"packdir={os.environ['OPENPI_DUQUANT_PACKDIR']}",
        flush=True,
    )
    t0 = time.perf_counter()
    replaced = enable_openpi_duquant_all_linears(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[OPENPI-DUQUANT] reference enabled layers={replaced}, dt={time.perf_counter() - t0:.2f}s", flush=True)
    return model


def enable_openpi_duquant_packed_w4(model):
    model = enable_openpi_duquant_reference(model)
    from openpi.models_pytorch.quant.duquant_packed_w4 import convert_duquant_to_packed_w4

    print(
        "[OPENPI-DUQUANT] converting reference -> packed_w4 "
        f"backend={os.environ.get('OPENPI_DUQUANT_PACKED_BACKEND', 'kernel')} "
        f"int4_cache={os.environ.get('OPENPI_DUQUANT_INT4_CACHE_DIR')}",
        flush=True,
    )
    t0 = time.perf_counter()
    converted = convert_duquant_to_packed_w4(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[OPENPI-DUQUANT] packed_w4 converted layers={converted}, dt={time.perf_counter() - t0:.2f}s", flush=True)
    return model


def enable_openpi_duquant_fused_w4(model):
    model = enable_openpi_duquant_reference(model)
    from openpi.models_pytorch.quant.duquant_fused_w4 import convert_duquant_to_fused_w4

    print(
        "[OPENPI-DUQUANT] converting reference -> fused_w4 "
        f"backend={os.environ.get('OPENPI_DUQUANT_PACKED_BACKEND', 'kernel')} "
        f"int4_cache={os.environ.get('OPENPI_DUQUANT_INT4_CACHE_DIR')}",
        flush=True,
    )
    t0 = time.perf_counter()
    converted = convert_duquant_to_fused_w4(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[OPENPI-DUQUANT] fused_w4 converted layers={converted}, dt={time.perf_counter() - t0:.2f}s", flush=True)
    return model


def _enable_openpi_duquant_staged(model):
    """Backend switch: reference / packed_w4 / fused_w4."""
    backend = os.environ.get("OPENPI_DUQUANT_WEIGHT_BACKEND", "fused_w4").strip().lower()
    if backend in {"reference", "fake", "duquant"}:
        return enable_openpi_duquant_reference(model)
    if backend in {"packed", "packed_w4", "real_w4"}:
        return enable_openpi_duquant_packed_w4(model)
    if backend in {"fused", "fused_w4"}:
        return enable_openpi_duquant_fused_w4(model)
    raise ValueError(f"Unsupported OPENPI_DUQUANT_WEIGHT_BACKEND={backend!r}; expected reference/packed_w4/fused_w4")
