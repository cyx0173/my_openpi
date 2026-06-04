from __future__ import annotations

import atexit
import os
import signal
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional


# =============================================================================
# DuQuant path constants
# =============================================================================

DEFAULT_DUQUANT_CALIB_PATH = (
    "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant/openpi_duquant_calib.pt"
)

DEFAULT_DUQUANT_PACK_DIR = (
    "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant/duquant_packed"
)

DEFAULT_DUQUANT_INT4_CACHE_DIR = (
    "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant/duquant_int4_cache"
)

DEFAULT_DUQUANT_INT4_CACHE_TAG = "duquant"

DEFAULT_DUQUANT_COMMON_EXCLUDE_REGEX = (
    r"lm_head"
    r"|embed_tokens"
    r"|rotary_emb"
    r"|\.norm"
    r"|\.input_layernorm"
    r"|\.post_attention_layernorm"
)

DEFAULT_DUQUANT_VLM_INCLUDE_REGEX = (
    r"paligemma_with_expert\.paligemma\.model\..*"
)

DEFAULT_DUQUANT_ACTION_MLP_IO_INCLUDE_REGEX = (
    r"paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$"
    r"|^action_in_proj$"
    r"|^time_mlp_in$"
    r"|^time_mlp_out$"
)

DEFAULT_DUQUANT_INCLUDE_REGEX = (
    DEFAULT_DUQUANT_VLM_INCLUDE_REGEX
    + r"|"
    + DEFAULT_DUQUANT_ACTION_MLP_IO_INCLUDE_REGEX
)

DEFAULT_DUQUANT_EXCLUDE_REGEX = (
    DEFAULT_DUQUANT_COMMON_EXCLUDE_REGEX
    + r"|^action_out_proj$"
)

# These envs are from older experimental paths and can silently change semantics.
# Do not clear core config envs such as ABITS, ACT_QUANT_MODE, PACKDIR, INT4_CACHE.
_CONFLICTING_EXPERIMENTAL_ENV_KEYS = (
    "OPENPI_DUQUANT_FUSED_W4",
    "OPENPI_DUQUANT_ACT_BACKEND",           # old real activation acceleration branch
    "OPENPI_DUQUANT_PACKED_ACT_SCALE_MODE", # old calibrated-fixed activation scale branch
)


# =============================================================================
# SmoothVLA path constants
# =============================================================================
#
# SmoothVLA is intentionally separate from DuQuant.
# It does NOT set OPENPI_DUQUANT_* envs and does NOT call DuQuant pack / R_in / R_out.

DEFAULT_SMOOTHVLA_CALIB_PATH = DEFAULT_DUQUANT_CALIB_PATH

DEFAULT_SMOOTHVLA_INT4_CACHE_DIR = (
    "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/smoothquant/smoothquant_int4_cache"
)

DEFAULT_SMOOTHVLA_INT4_CACHE_TAG = "smoothvla_affine"

# Keep SmoothVLA and DuQuant layer selection identical for fair comparison.
# Expected matched layers: 346.
DEFAULT_SMOOTHVLA_INCLUDE_REGEX = DEFAULT_DUQUANT_INCLUDE_REGEX
DEFAULT_SMOOTHVLA_EXCLUDE_REGEX = (
    DEFAULT_DUQUANT_COMMON_EXCLUDE_REGEX
    + r"|\.self_attn\.(q_proj|k_proj|v_proj|o_proj|out_proj)"
    + r"|\.attn\.(q_proj|k_proj|v_proj|out_proj)"
    + r"|^action_out_proj$"
)

# =============================================================================
# Shared helpers
# =============================================================================

def _as_env_bool(value: bool) -> str:
    return "1" if bool(value) else "0"


def _set_env(name: str, value: object | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)


def _clear_conflicting_experimental_env() -> None:
    for key in _CONFLICTING_EXPERIMENTAL_ENV_KEYS:
        os.environ.pop(key, None)


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        val = os.environ.get(name)
        if val is not None and str(val).strip() != "":
            return val
    return None


def _quant_mode_default_abits(default: int | None = None) -> Optional[int]:
    # Keep compatibility with policy_config OPENPI_QUANT_MODE mapping.
    mode = os.environ.get("OPENPI_QUANT_MODE", "").strip()
    return {
        "1": 4,   # W4A4
        "2": 8,   # W4A8
        "3": 16,  # W4A16
        "4": 2,   # W4A2, usually not recommended
    }.get(mode, default)


# =============================================================================
# DuQuant config / path
# =============================================================================

def _canonical_weight_backend(raw: object | None) -> str:
    backend = str(raw or "fused_w4").strip().lower()
    aliases = {
        "fake": "reference",
        "duquant": "reference",
        "ref": "reference",
        "packed": "packed_w4",
        "real_w4": "fused_w4",  # compatibility: real W4 weight should use fused/packed fast path
        "fused": "fused_w4",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"reference", "packed_w4", "fused_w4"}:
        raise ValueError(
            f"Unsupported DuQuant weight backend={backend!r}. "
            "Expected reference / packed_w4 / fused_w4."
        )
    return backend


@dataclass(frozen=True)
class QuantVLADuQuantConfig:
    # Paths / backend.
    calib_path: Optional[str] = DEFAULT_DUQUANT_CALIB_PATH
    pack_dir: Optional[str] = DEFAULT_DUQUANT_PACK_DIR
    int4_cache_dir: Optional[str] = DEFAULT_DUQUANT_INT4_CACHE_DIR
    int4_cache_tag: Optional[str] = DEFAULT_DUQUANT_INT4_CACHE_TAG
    weight_backend: Optional[str] = None
    packed_backend: Optional[str] = "kernel"

    # Quant bits. If None, use env / OPENPI_QUANT_MODE / default.
    wbits: Optional[int] = None
    abits: Optional[int] = None

    # DuQuant transform / quant config.
    act_quant_mode: Optional[str] = None
    block_size: Optional[int] = None
    block_out_size: Optional[int] = None
    alpha: Optional[float] = None
    lac: Optional[float] = None
    swc: Optional[float] = None
    lambda_smooth: Optional[float] = None  # kept for compatibility/logging
    permute: Optional[bool] = None
    permutation_times: Optional[int] = None
    row_rot: Optional[str] = None
    act_group_size: Optional[int] = None
    require_calib: bool = True

    # Selective layout.
    include_regex: Optional[str] = DEFAULT_DUQUANT_INCLUDE_REGEX
    exclude_regex: Optional[str] = DEFAULT_DUQUANT_EXCLUDE_REGEX

    # Cache / env handling.
    int4_cache: bool = True
    clear_conflicting_experimental_env: bool = True

    def resolved_weight_backend(self) -> str:
        raw = (
            self.weight_backend
            or _env_first("OPENPI_DUQUANT_WEIGHT_BACKEND", "OPENPI_DUQUANT_BACKEND")
            or "fused_w4"
        )
        return _canonical_weight_backend(raw)

    def resolved_packed_backend(self) -> str:
        return str(self.packed_backend or _env_first("OPENPI_DUQUANT_PACKED_BACKEND") or "kernel").strip().lower()

    def resolved_calib_path(self) -> Optional[str]:
        return _env_first("OPENPI_DUQUANT_CALIB_PATH") or self.calib_path

    def resolved_pack_dir(self) -> Optional[str]:
        env_pack = _env_first("OPENPI_DUQUANT_PACKDIR")
        if env_pack:
            return env_pack
        if self.pack_dir:
            return self.pack_dir
        calib = self.resolved_calib_path()
        if calib:
            return str(Path(calib).expanduser().resolve().parent / "duquant_packed")
        return None

    def resolved_int4_cache_dir(self) -> Optional[str]:
        env_cache = _env_first("OPENPI_DUQUANT_INT4_CACHE_DIR")
        if env_cache:
            return env_cache
        if self.int4_cache_dir:
            return self.int4_cache_dir
        pack_dir = self.resolved_pack_dir()
        if pack_dir:
            return str(Path(pack_dir).expanduser().resolve() / "duquant_int4_cache")
        return None

    def resolved_wbits(self) -> int:
        val = self.wbits
        if val is None:
            env = _env_first("OPENPI_DUQUANT_WBITS_DEFAULT", "OPENPI_DUQUANT_WBITS")
            val = int(env) if env is not None else 4
        return int(val)

    def resolved_abits(self) -> int:
        val = self.abits
        if val is None:
            env = _env_first("OPENPI_DUQUANT_ABITS")
            if env is not None:
                val = int(env)
            else:
                val = _quant_mode_default_abits(default=4)
        return int(val)

    def resolved_act_quant_mode(self) -> str:
        return str(self.act_quant_mode or _env_first("OPENPI_DUQUANT_ACT_QUANT_MODE") or "dynamic_token_amax")

    def resolved_block_size(self) -> int:
        return int(self.block_size if self.block_size is not None else (_env_first("OPENPI_DUQUANT_BLOCK") or 128))

    def resolved_block_out_size(self) -> int:
        if self.block_out_size is not None:
            return int(self.block_out_size)
        env = _env_first("OPENPI_DUQUANT_BLOCK_OUT")
        if env is not None:
            return int(env)
        return self.resolved_block_size()

    def resolved_alpha(self) -> float:
        return float(self.alpha if self.alpha is not None else (_env_first("OPENPI_DUQUANT_ALPHA") or 0.6))

    def resolved_lac(self) -> float:
        return float(self.lac if self.lac is not None else (_env_first("OPENPI_DUQUANT_LAC") or 0.9))

    def resolved_swc(self) -> float:
        return float(self.swc if self.swc is not None else (_env_first("OPENPI_DUQUANT_SWC") or 0.8))

    def resolved_lambda_smooth(self) -> float:
        return float(self.lambda_smooth if self.lambda_smooth is not None else (_env_first("OPENPI_DUQUANT_LS") or 0.15))

    def resolved_permute(self) -> bool:
        if self.permute is not None:
            return bool(self.permute)
        return str(_env_first("OPENPI_DUQUANT_PERMUTE") or "1") not in {"0", "false", "False", "no"}

    def resolved_permutation_times(self) -> int:
        return int(self.permutation_times if self.permutation_times is not None else (_env_first("OPENPI_DUQUANT_PERMUTATION_TIMES") or 1))

    def resolved_row_rot(self) -> str:
        return str(self.row_rot or _env_first("OPENPI_DUQUANT_ROW_ROT") or "restore")

    def resolved_act_group_size(self) -> int:
        return int(self.act_group_size if self.act_group_size is not None else (_env_first("OPENPI_DUQUANT_ACT_GROUP_SIZE") or 0))


def configure_quantvla_duquant(
    config: QuantVLADuQuantConfig | None = None,
    **overrides,
) -> QuantVLADuQuantConfig:
    if config is None:
        config = QuantVLADuQuantConfig(**overrides)
    elif overrides:
        config = replace(config, **overrides)

    if config.clear_conflicting_experimental_env:
        _clear_conflicting_experimental_env()

    backend = config.resolved_weight_backend()
    packed_backend = config.resolved_packed_backend()
    wbits = config.resolved_wbits()
    abits = config.resolved_abits()

    if wbits != 4:
        raise ValueError(f"Current packed/fused DuQuant W4 backend requires wbits=4, got {wbits}")
    if abits not in {2, 4, 8, 16}:
        raise ValueError(f"QuantVLA DuQuant supports abits in {{2,4,8,16}}, got {abits}")

    calib_path = config.resolved_calib_path()
    pack_dir = config.resolved_pack_dir()
    int4_cache_dir = config.resolved_int4_cache_dir()

    if config.require_calib and not calib_path:
        raise RuntimeError(
            "DuQuant requires calibration. Set OPENPI_DUQUANT_CALIB_PATH or pass calib_path."
        )

    if not config.include_regex:
        raise RuntimeError("OPENPI_DUQUANT_INCLUDE would be empty; refusing to quantize all linears accidentally.")

    # Backend selection used by openpi_duquant_enable.py.
    _set_env("OPENPI_DUQUANT_WEIGHT_BACKEND", backend)
    _set_env("OPENPI_DUQUANT_BACKEND", backend)  # compatibility only
    _set_env("OPENPI_DUQUANT_PACKED_BACKEND", packed_backend)

    # Paths.
    _set_env("OPENPI_DUQUANT_CALIB_PATH", calib_path)
    _set_env("OPENPI_DUQUANT_PACKDIR", pack_dir)
    _set_env("OPENPI_DUQUANT_INT4_CACHE", _as_env_bool(config.int4_cache))
    _set_env("OPENPI_DUQUANT_INT4_CACHE_DIR", int4_cache_dir)
    _set_env("OPENPI_DUQUANT_INT4_CACHE_TAG", config.int4_cache_tag)

    # Core DuQuant config.
    _set_env("OPENPI_DUQUANT_WBITS_DEFAULT", wbits)
    _set_env("OPENPI_DUQUANT_WBITS", str(wbits))
    _set_env("OPENPI_DUQUANT_ABITS", abits)
    _set_env("OPENPI_DUQUANT_ACT_QUANT_MODE", config.resolved_act_quant_mode())
    _set_env("OPENPI_DUQUANT_BLOCK", config.resolved_block_size())
    _set_env("OPENPI_DUQUANT_BLOCK_OUT", config.resolved_block_out_size())
    _set_env("OPENPI_DUQUANT_ALPHA", config.resolved_alpha())
    _set_env("OPENPI_DUQUANT_LAC", config.resolved_lac())
    _set_env("OPENPI_DUQUANT_SWC", config.resolved_swc())
    _set_env("OPENPI_DUQUANT_LS", config.resolved_lambda_smooth())
    _set_env("OPENPI_DUQUANT_PERMUTE", _as_env_bool(config.resolved_permute()))
    _set_env("OPENPI_DUQUANT_PERMUTATION_TIMES", config.resolved_permutation_times())
    _set_env("OPENPI_DUQUANT_ROW_ROT", config.resolved_row_rot())
    _set_env("OPENPI_DUQUANT_ACT_GROUP_SIZE", config.resolved_act_group_size())
    _set_env("OPENPI_DUQUANT_REQUIRE_CALIB", _as_env_bool(config.require_calib))

    # Selective layout.
    _set_env("OPENPI_DUQUANT_INCLUDE", config.include_regex)
    _set_env("OPENPI_DUQUANT_EXCLUDE", config.exclude_regex)

    print(
        "[QUANTVLA-DUQUANT] configured "
        f"backend={backend} packed_backend={packed_backend} "
        f"W{wbits}A{abits} act_mode={os.environ['OPENPI_DUQUANT_ACT_QUANT_MODE']} "
        f"block={os.environ['OPENPI_DUQUANT_BLOCK']} "
        f"block_out={os.environ['OPENPI_DUQUANT_BLOCK_OUT']} "
        f"alpha={os.environ['OPENPI_DUQUANT_ALPHA']} "
        f"lac={os.environ['OPENPI_DUQUANT_LAC']} "
        f"swc={os.environ['OPENPI_DUQUANT_SWC']} "
        f"ls={os.environ['OPENPI_DUQUANT_LS']} "
        f"permute={os.environ['OPENPI_DUQUANT_PERMUTE']} "
        f"perm_times={os.environ['OPENPI_DUQUANT_PERMUTATION_TIMES']} "
        f"row_rot={os.environ['OPENPI_DUQUANT_ROW_ROT']} "
        f"calib_path={calib_path} pack_dir={pack_dir} "
        f"int4_cache={os.environ['OPENPI_DUQUANT_INT4_CACHE']} "
        f"int4_cache_dir={int4_cache_dir} "
        f"include={config.include_regex!r} exclude={config.exclude_regex!r}",
        flush=True,
    )

    return config


def _prepare_openpi_duquant_calibration(
    model,
    *,
    config: QuantVLADuQuantConfig | None = None,
    calib_path: Optional[str] = DEFAULT_DUQUANT_CALIB_PATH,
    include_regex: Optional[str] = DEFAULT_DUQUANT_INCLUDE_REGEX,
    exclude_regex: Optional[str] = DEFAULT_DUQUANT_EXCLUDE_REGEX,
    clear_conflicting_experimental_env: bool = True,
    save_on_exit: bool = True,
    autosave_every: int = 0,
):
    if config is None:
        config = QuantVLADuQuantConfig(
            calib_path=calib_path,
            pack_dir=None,
            weight_backend="reference",
            require_calib=False,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            clear_conflicting_experimental_env=clear_conflicting_experimental_env,
        )

    if config.clear_conflicting_experimental_env:
        _clear_conflicting_experimental_env()

    out_path = config.resolved_calib_path()
    if not out_path:
        raise RuntimeError("DuQuant calibration output path is required.")

    if not config.include_regex:
        raise RuntimeError("DuQuant calibration include_regex is empty; refusing to collect all Linear layers.")

    Path(out_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    from openpi.models_pytorch.quant.duquant_calibration import DuQuantActivationCalibrator

    include = config.include_regex
    exclude = config.exclude_regex or ""

    calibrator = DuQuantActivationCalibrator(include_regex=include, exclude_regex=exclude)
    n_hooks = calibrator.register(model)

    print(
        "[QUANTVLA-DUQUANT-CALIB] online calibration enabled "
        f"hooks={n_hooks} out_path={out_path} include={include!r} exclude={exclude!r}",
        flush=True,
    )

    save_state = {"saved": False, "forward_count": 0}

    def _save_calibration(reason: str = "manual") -> None:
        try:
            state = calibrator.state_dict()
            calibrator.save(out_path)
            save_state["saved"] = True
            print(
                "[QUANTVLA-DUQUANT-CALIB] saved "
                f"reason={reason} layers={len(state.get('layers', {}))} path={out_path}",
                flush=True,
            )
        except Exception as e:
            print(f"[QUANTVLA-DUQUANT-CALIB][ERROR] failed to save calibration: {e}", flush=True)

    model_forward_handle = None
    if autosave_every and autosave_every > 0:
        def _model_forward_hook(_module, _inputs, _outputs):
            save_state["forward_count"] += 1
            if save_state["forward_count"] % int(autosave_every) == 0:
                _save_calibration(reason=f"autosave_{save_state['forward_count']}")

        try:
            model_forward_handle = model.register_forward_hook(_model_forward_hook)
        except Exception as e:
            print(
                "[QUANTVLA-DUQUANT-CALIB][WARN] "
                f"failed to register model autosave hook: {e}",
                flush=True,
            )

    def _close_and_save(reason: str) -> None:
        _save_calibration(reason=reason)
        try:
            calibrator.close()
        except Exception:
            pass
        if model_forward_handle is not None:
            try:
                model_forward_handle.remove()
            except Exception:
                pass

    if save_on_exit:
        atexit.register(lambda: _close_and_save("atexit"))

        def _signal_handler(signum, _frame):
            _close_and_save(f"signal_{signum}")
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception as e:
            print(
                "[QUANTVLA-DUQUANT-CALIB][WARN] "
                f"failed to register signal handlers: {e}",
                flush=True,
            )

    model._duquant_calibrator = calibrator
    model._duquant_save_calibration = _save_calibration
    model._duquant_calibration_path = out_path

    return model


_calibrate_openpi_duquant_staged = _prepare_openpi_duquant_calibration


def _enable_openpi_duquant_staged(
    model,
    *,
    config: QuantVLADuQuantConfig | None = None,
    calib_path: Optional[str] = None,
    pack_dir: Optional[str] = None,
    weight_backend: Optional[str] = None,
    backend: Optional[str] = None,  # backward-compatible alias
    packed_backend: Optional[str] = None,
    wbits: Optional[int] = None,
    abits: Optional[int] = None,
    act_quant_mode: Optional[str] = None,
    block_size: Optional[int] = None,
    block_out_size: Optional[int] = None,
    alpha: Optional[float] = None,
    lac: Optional[float] = None,
    swc: Optional[float] = None,
    lambda_smooth: Optional[float] = None,
    permute: Optional[bool] = None,
    permutation_times: Optional[int] = None,
    row_rot: Optional[str] = None,
    act_group_size: Optional[int] = None,
    require_calib: bool = True,
    include_regex: Optional[str] = DEFAULT_DUQUANT_INCLUDE_REGEX,
    exclude_regex: Optional[str] = DEFAULT_DUQUANT_EXCLUDE_REGEX,
    clear_conflicting_experimental_env: bool = True,
    int4_cache: bool = True,
    int4_cache_dir: Optional[str] = None,
    int4_cache_tag: Optional[str] = None,
):
    if config is None:
        config = QuantVLADuQuantConfig(
            calib_path=calib_path or DEFAULT_DUQUANT_CALIB_PATH,
            pack_dir=pack_dir or DEFAULT_DUQUANT_PACK_DIR,
            int4_cache_dir=int4_cache_dir or DEFAULT_DUQUANT_INT4_CACHE_DIR,
            int4_cache_tag=int4_cache_tag or DEFAULT_DUQUANT_INT4_CACHE_TAG,
            weight_backend=weight_backend or backend,
            packed_backend=packed_backend or "kernel",
            wbits=wbits,
            abits=abits,
            act_quant_mode=act_quant_mode,
            block_size=block_size,
            block_out_size=block_out_size,
            alpha=alpha,
            lac=lac,
            swc=swc,
            lambda_smooth=lambda_smooth,
            permute=permute,
            permutation_times=permutation_times,
            row_rot=row_rot,
            act_group_size=act_group_size,
            require_calib=require_calib,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            clear_conflicting_experimental_env=clear_conflicting_experimental_env,
            int4_cache=int4_cache,
        )

    configure_quantvla_duquant(config)

    from openpi.models_pytorch.quant.openpi_duquant_enable import (
        _enable_openpi_duquant_staged as enable_duquant_backend,
    )

    model = enable_duquant_backend(model)

    print(
        "[QUANTVLA-DUQUANT] enabled backend "
        f"weight_backend={os.environ.get('OPENPI_DUQUANT_WEIGHT_BACKEND')} "
        f"packed_backend={os.environ.get('OPENPI_DUQUANT_PACKED_BACKEND')} "
        f"W{os.environ.get('OPENPI_DUQUANT_WBITS_DEFAULT')}A{os.environ.get('OPENPI_DUQUANT_ABITS')}",
        flush=True,
    )

    return model


# =============================================================================
# SmoothVLA config / path
# =============================================================================

def _make_openpi_smoothvla_config_from_quant_mode():
    """Single source of truth for SmoothVLA/SmoothQuant experiments.

    This function intentionally does NOT read or set OPENPI_DUQUANT_* variables.

    OPENPI_QUANT_MODE:
      1 -> W4A4
      2 -> W4A8
      3 -> W4A16
      4 -> W4A2, not recommended
    """
    from openpi.models_pytorch.quant.smoothquant_w4 import SmoothQuantW4Config

    abits = _quant_mode_default_abits(default=4)

    return SmoothQuantW4Config(
        calib_path=DEFAULT_SMOOTHVLA_CALIB_PATH,
        int4_cache_dir=DEFAULT_SMOOTHVLA_INT4_CACHE_DIR,
        int4_cache_tag=DEFAULT_SMOOTHVLA_INT4_CACHE_TAG,

        include_regex=DEFAULT_SMOOTHVLA_INCLUDE_REGEX,
        exclude_regex=DEFAULT_SMOOTHVLA_EXCLUDE_REGEX,

        weight_bits=4,
        act_bits=abits,

        alpha=0.5,
        lac=1.0,
        swc=1.0,

        # Current SmoothVLA affine strategy.
        # For old behavior, change to:
        #   act_quant_mode="dynamic_symmetric", act_group_size=0
        # For group affine test, change to:
        #   act_quant_mode="dynamic_group_affine", act_group_size=128
        act_quant_mode="dynamic_token_affine",
        act_group_size=0,

        backend="kernel",
        require_calib=True,
        int4_cache=True,
        scale_clip_min=1.0e-3,
        scale_clip_max=1.0e3,
    )


def _enable_openpi_smoothvla_staged(model, *, config=None):
    """Enable the independent SmoothVLA/SmoothQuant W4 path.

    This path is intentionally separate from _enable_openpi_duquant_staged(...).

    SmoothVLA:
      calibration absmax -> SmoothQuant scale
      W_s = W * s -> real packed W4 weight
      x_s = x / s -> dynamic activation fake quant
      W4 kernel forward

    No DuQuant:
      no perm
      no R_in / R_out
      no DuQuant pack
      no OPENPI_DUQUANT_* env fallback
    """
    if config is None:
        config = _make_openpi_smoothvla_config_from_quant_mode()

    from openpi.models_pytorch.quant.smoothquant_w4 import enable_openpi_smoothquant_w4

    print(
        "[OPENPI-SMOOTHVLA] staged enable "
        f"W4A{config.act_bits} alpha={config.alpha} "
        f"act_mode={config.act_quant_mode} group={config.act_group_size} "
        f"lac={config.lac} swc={config.swc} backend={config.backend}",
        flush=True,
    )

    return enable_openpi_smoothquant_w4(model, config=config)


# =============================================================================
# Debug
# =============================================================================

def _debug_duquant_state(model, tag: str = "unknown", max_examples: int = 50) -> None:
    from collections import Counter

    bit_counter = Counter()
    class_counter = Counter()
    examples = []

    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        has_bit_api = hasattr(module, "set_w_bits") and hasattr(module, "set_a_bits")
        looks_quantized = has_bit_api or "Quant" in class_name or "W4" in class_name
        if not looks_quantized:
            continue

        w_bits = getattr(module, "weight_bits", getattr(module, "current_w_bits", getattr(module, "w_bits", None)))
        a_bits = getattr(module, "act_bits", getattr(module, "current_a_bits", getattr(module, "a_bits", None)))

        bit_counter[(w_bits, a_bits)] += 1
        class_counter[class_name] += 1

        if len(examples) < int(max_examples):
            examples.append((name, class_name, w_bits, a_bits))

    print(f"\n[DUQUANT-DEBUG] tag={tag}", flush=True)
    print(f"[DUQUANT-DEBUG] quant_like_modules={sum(bit_counter.values())}", flush=True)
    print(f"[DUQUANT-DEBUG] bit_counter={dict(bit_counter)}", flush=True)
    print(f"[DUQUANT-DEBUG] class_counter={dict(class_counter)}", flush=True)

    for name, class_name, w_bits, a_bits in examples:
        print(
            f"[DUQUANT-DEBUG] example name={name} class={class_name} W={w_bits} A={a_bits}",
            flush=True,
        )


__all__ = [
    "DEFAULT_DUQUANT_CALIB_PATH",
    "DEFAULT_DUQUANT_PACK_DIR",
    "DEFAULT_DUQUANT_INT4_CACHE_DIR",
    "DEFAULT_DUQUANT_INT4_CACHE_TAG",
    "DEFAULT_DUQUANT_INCLUDE_REGEX",
    "DEFAULT_DUQUANT_EXCLUDE_REGEX",
    "DEFAULT_SMOOTHVLA_CALIB_PATH",
    "DEFAULT_SMOOTHVLA_INT4_CACHE_DIR",
    "DEFAULT_SMOOTHVLA_INT4_CACHE_TAG",
    "DEFAULT_SMOOTHVLA_INCLUDE_REGEX",
    "DEFAULT_SMOOTHVLA_EXCLUDE_REGEX",
    "QuantVLADuQuantConfig",
    "configure_quantvla_duquant",
    "_calibrate_openpi_duquant_staged",
    "_prepare_openpi_duquant_calibration",
    "_enable_openpi_duquant_staged",
    "_make_openpi_smoothvla_config_from_quant_mode",
    "_enable_openpi_smoothvla_staged",
    "_debug_duquant_state",
]
