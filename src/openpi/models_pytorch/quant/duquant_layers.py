from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .duquant_calibration import load_calibration
from .duquant_preprocess import (
    PackResult,
    apply_bias_row_rot_optimized,
    apply_input_transform_optimized,
    apply_output_restore_optimized,
    fake_quantize_activation,
    fake_quantize_sym,
    load_pack,
    pack_weight,
    save_pack,
    transform_weight_for_forward_optimized,
)


def _env(name: str, default: str) -> str:
    openpi_name = f"OPENPI_DUQUANT_{name}"
    if openpi_name in os.environ:
        return os.environ[openpi_name]
    return default


def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).lower() not in {"0", "false", "no", "off"}


@dataclass
class DuQuantConfig:
    weight_bits: int = 4
    act_bits: int = 4
    block_size: int = 128
    block_out_size: int = 128
    alpha: float = 0.6
    lac: float = 0.9
    swc: float = 0.8
    enable_permute: bool = True
    permutation_times: int = 1
    row_rot_mode: str = "restore"
    act_group_size: int = 0
    pack_dir: Optional[str] = None
    calib_path: Optional[str] = None
    require_calib: bool = True

    @classmethod
    def from_env(cls) -> "DuQuantConfig":
        block = int(_env("BLOCK", "128"))
        return cls(
            weight_bits=int(_env("WBITS", _env("WBITS_DEFAULT", "4"))),
            act_bits=int(_env("ABITS", "4")),
            block_size=block,
            block_out_size=int(_env("BLOCK_OUT", str(block))),
            alpha=float(_env("ALPHA", "0.6")),
            lac=float(_env("LAC", "0.9")),
            swc=float(_env("SWC", "0.8")),
            enable_permute=_env_bool("PERMUTE", "1"),
            permutation_times=int(_env("PERMUTATION_TIMES", "1")),
            row_rot_mode=_env("ROW_ROT", "restore").strip().lower(),
            act_group_size=int(_env("ACT_GROUP_SIZE", "0")),
            pack_dir=os.environ.get("OPENPI_DUQUANT_PACKDIR"),
            calib_path=os.environ.get("OPENPI_DUQUANT_CALIB_PATH"),
            require_calib=_env_bool("REQUIRE_CALIB", "1"),
        )


def _parse_per_layer_wbits(env_val: Optional[str]) -> Dict[str, int]:
    if not env_val:
        return {}
    out: Dict[str, int] = {}
    for part in env_val.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, bit = part.split(":", 1)
        try:
            out[name.strip()] = int(bit.strip())
        except ValueError:
            pass
    return out


def _expected_pack_meta(name: str, base: nn.Linear, cfg: DuQuantConfig, calib_rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Mirrors the fields used by duquant_preprocess._meta_matches.  The exact
    # calibration hash is computed inside pack_weight, so we pass None here and
    # let stale packs be safely rebuilt by not using expected_meta in this port.
    return {
        "format": "openpi_duquant_reference_v1",
        "layer_name": name,
        "in_features": int(base.in_features),
        "out_features": int(base.out_features),
        "block_size": int(cfg.block_size),
        "block_out_size": int(cfg.block_out_size),
        "enable_permute": bool(cfg.enable_permute),
        "require_calib": bool(cfg.require_calib),
        "permutation_times": int(cfg.permutation_times),
        "row_rot_mode": str(cfg.row_rot_mode),
    }


class DuQuantLinear(nn.Module):
    """Clean reference DuQuant Linear for OpenPI.

    This is intentionally not a deployment kernel.  It is the correctness path:
    calibration-driven transform + fake W/A quantization + optional restore.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        name: str,
        cfg: DuQuantConfig,
        weight_bits: Optional[int] = None,
        calib_rec: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.cfg = cfg
        self.weight_bits = int(cfg.weight_bits if weight_bits is None else weight_bits)
        self.act_bits = int(cfg.act_bits)

        self.register_buffer("_weight", base.weight.detach().clone(), persistent=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        act_absmax = None
        act_std = None
        if calib_rec is not None:
            act_absmax = calib_rec.get("absmax")
            act_std = calib_rec.get("std")

        pack = load_pack(self.name, cfg.pack_dir)
        if pack is None:
            pack = pack_weight(
                self._weight,
                layer_name=self.name,
                act_absmax=act_absmax,
                act_std=act_std,
                block_size=cfg.block_size,
                block_out_size=cfg.block_out_size,
                enable_permute=cfg.enable_permute,
                require_calib=cfg.require_calib,
                alpha=cfg.alpha,
                permutation_times=cfg.permutation_times,
                row_rot_mode=cfg.row_rot_mode,
            )
            save_pack(self.name, pack, cfg.pack_dir)
        self.pack: PackResult = pack

        if pack.perm is not None:
            self.register_buffer("_perm_cache", torch.from_numpy(pack.perm).long(), persistent=False)
        else:
            self._perm_cache = None

        self._R_in_block_indices: List[int] = []
        if pack.R_in_blocks:
            for b, R in pack.R_in_blocks.items():
                self.register_buffer(f"_R_in_{b}", torch.from_numpy(R).to(dtype=self._weight.dtype), persistent=False)
                self._R_in_block_indices.append(int(b))

        self._R_out_block_indices: List[int] = []
        if pack.R_out_blocks:
            for b, R in pack.R_out_blocks.items():
                self.register_buffer(f"_R_out_{b}", torch.from_numpy(R).to(dtype=self._weight.dtype), persistent=False)
                self._R_out_block_indices.append(int(b))

        self._block_size = int(pack.meta.get("block_size", cfg.block_size))
        self._block_out_size = int(pack.meta.get("block_out_size", cfg.block_out_size))

        self.register_buffer("_W_t", torch.zeros_like(self._weight), persistent=False)
        self.register_buffer("_W_t_quantized", torch.zeros_like(self._weight), persistent=False)
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype), persistent=False)
        self._cached_weight_key: Optional[Tuple[str, torch.dtype, int, int, float]] = None
        self._weight_quantized_cached = False
        self._bias_rot: Optional[torch.Tensor] = None

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            self._weight.copy_(value)
            self._cached_weight_key = None
            self._weight_quantized_cached = False

    def set_w_bits(self, bits: int) -> None:
        self.weight_bits = int(bits)
        self._cached_weight_key = None
        self._weight_quantized_cached = False

    def set_a_bits(self, bits: int) -> None:
        self.act_bits = int(bits)

    def _get_R_in_cache(self) -> Dict[int, torch.Tensor]:
        return {b: getattr(self, f"_R_in_{b}") for b in self._R_in_block_indices}

    def _get_R_out_cache(self) -> Dict[int, torch.Tensor]:
        return {b: getattr(self, f"_R_out_{b}") for b in self._R_out_block_indices}

    @torch.no_grad()
    def _maybe_update_weight_cache(self) -> None:
        apply_row = self.cfg.row_rot_mode == "restore"
        key = (str(self._weight.device), self._weight.dtype, int(self.weight_bits), int(apply_row), float(self.cfg.swc))
        if self._cached_weight_key == key:
            return

        W_t, scales = transform_weight_for_forward_optimized(
            self._weight,
            self.pack,
            weight_bits=self.weight_bits,
            apply_row_rot=apply_row,
            perm_cache=self._perm_cache,
            R_in_cache=self._get_R_in_cache(),
            R_out_cache=self._get_R_out_cache(),
            block_size=self._block_size,
            block_out_size=self._block_out_size,
            swc=self.cfg.swc,
        )
        self._W_t.copy_(W_t)
        self._w_scales.copy_(scales)

        if 0 < self.weight_bits < 16:
            self._W_t_quantized.copy_(fake_quantize_sym(W_t, scales[:, None], self.weight_bits))
            self._weight_quantized_cached = True
        else:
            self._weight_quantized_cached = False

        if self.bias is not None and self.cfg.row_rot_mode == "propagate" and self.pack.R_out_blocks is not None:
            self._bias_rot = apply_bias_row_rot_optimized(
                self.bias.detach(),
                self.pack,
                self._get_R_out_cache(),
                self._block_out_size,
            )
        else:
            self._bias_rot = None
        self._cached_weight_key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = apply_input_transform_optimized(
            x,
            self.pack,
            self._perm_cache,
            self._get_R_in_cache(),
            self._block_size,
        )
        x_t = fake_quantize_activation(
            x_t,
            self.act_bits,
            lac=self.cfg.lac,
            group_size=self.cfg.act_group_size,
        )

        self._maybe_update_weight_cache()
        if self._weight_quantized_cached:
            y = torch.nn.functional.linear(x_t, self._W_t_quantized, None)
        elif 0 < self.weight_bits < 16:
            W_q = fake_quantize_sym(self._W_t, self._w_scales[:, None], self.weight_bits)
            y = torch.nn.functional.linear(x_t, W_q, None)
        else:
            y = torch.nn.functional.linear(x_t, self._W_t, None)

        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            y = apply_output_restore_optimized(y, self.pack, self._get_R_out_cache(), self._block_out_size)
            if self.bias is not None:
                y = y + self.bias.to(dtype=y.dtype, device=y.device)
        else:
            if self.bias is not None:
                bias = self._bias_rot if self._bias_rot is not None else self.bias
                y = y + bias.to(dtype=y.dtype, device=y.device)
        return y


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def get_duquant_layers(model: nn.Module) -> List[DuQuantLinear]:
    return [m for m in model.modules() if isinstance(m, DuQuantLinear)]


def count_duquant_layers(model: nn.Module) -> int:
    return len(get_duquant_layers(model))


def enable_openpi_duquant_all_linears(
    model: nn.Module,
    *,
    dry_run: Optional[bool] = None,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    per_layer_wbits: Optional[Dict[str, int]] = None,
) -> int:
    if dry_run is None:
        dry_run = os.environ.get("OPENPI_DUQUANT_DRYRUN", "0").lower() not in {"0", "false", "no"}
    if include_regex is None:
        include_regex = os.environ.get("OPENPI_DUQUANT_INCLUDE", r".*")
    if exclude_regex is None:
        exclude_regex = os.environ.get("OPENPI_DUQUANT_EXCLUDE", "")
    if per_layer_wbits is None:
        per_layer_wbits = _parse_per_layer_wbits(os.environ.get("OPENPI_DUQUANT_WBITS"))

    cfg = DuQuantConfig.from_env()
    calib = load_calibration(cfg.calib_path) if cfg.calib_path else {}
    if cfg.require_calib and not calib and not dry_run:
        raise RuntimeError(
            "OPENPI_DUQUANT_REQUIRE_CALIB=1 but no calibration was loaded. "
            "Set OPENPI_DUQUANT_CALIB_PATH to a file generated by duquant_calibration.py."
        )

    include_re = re.compile(include_regex)
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    targets: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, DuQuantLinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not include_re.search(name):
            continue
        if exclude_re is not None and exclude_re.search(name):
            continue
        targets.append(name)

    # print(
    #     f"[OPENPI-DUQUANT-REF] matched={len(targets)} W{cfg.weight_bits}A{cfg.act_bits} "
    #     f"block={cfg.block_size} lac={cfg.lac} swc={cfg.swc} "
    #     f"calib_layers={len(calib)} pack_dir={cfg.pack_dir}",
    #     flush=True,
    # )

    bucket_counter = Counter()
    suffix_counter = Counter()
    shape_counter = defaultdict(int)

    def bucket_name(name: str) -> str:
        if name.startswith("paligemma_with_expert.paligemma"):
            return "paligemma"
        if name.startswith("paligemma_with_expert.gemma_expert"):
            return "gemma_expert"
        if name in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "state_proj"}:
            return "action_io_time"
        return "other"

    replaced = 0
    missing_calib: List[str] = []
    for name in targets:
        parent, attr = _get_parent_module_and_attr(model, name)
        old = getattr(parent, attr)
        if not isinstance(old, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        bucket_counter[bucket_name(name)] += 1
        suffix_counter[name.split(".")[-1]] += 1
        shape_counter[(old.in_features, old.out_features)] += 1

        if cfg.require_calib and name not in calib:
            missing_calib.append(name)
            if not dry_run:
                continue

        if dry_run:
            print(f"[OPENPI-DUQUANT-REF][DRYRUN] {name}: Linear({old.in_features}->{old.out_features}) W{wbits} A{cfg.act_bits}")
            continue

        new = DuQuantLinear(old, name=name, cfg=cfg, weight_bits=wbits, calib_rec=calib.get(name))
        setattr(parent, attr, new)
        replaced += 1

    if missing_calib:
        msg = f"[OPENPI-DUQUANT-REF] missing calibration for {len(missing_calib)} selected layers"
        print(msg, flush=True)
        for n in missing_calib[:20]:
            print(f"  missing: {n}", flush=True)
        if cfg.require_calib and not dry_run:
            raise RuntimeError(msg)

    if dry_run:
        print(f"[OPENPI-DUQUANT-REF] dry-run total layers listed: {len(targets)}", flush=True)
        return len(targets)

    print(f"[OPENPI-DUQUANT-REF] total layers replaced: {replaced}", flush=True)
    print("[OPENPI-DUQUANT-REF] breakdown by bucket:", dict(bucket_counter), flush=True)
    print("[OPENPI-DUQUANT-REF] top suffixes:", suffix_counter.most_common(20), flush=True)
    print("[OPENPI-DUQUANT-REF] top shapes:", sorted(shape_counter.items(), key=lambda x: -x[1])[:20], flush=True)
    return replaced


def set_all_duquant_bits(model: nn.Module, *, w_bits: Optional[int] = None, a_bits: Optional[int] = None) -> None:
    for layer in get_duquant_layers(model):
        if w_bits is not None:
            layer.set_w_bits(w_bits)
        if a_bits is not None:
            layer.set_a_bits(a_bits)
