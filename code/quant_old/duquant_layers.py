from __future__ import annotations
from collections import Counter, defaultdict
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import json
from pathlib import Path
import torch
from torch import nn

from .duquant_preprocess import (
    PackResult,
    apply_bias_row_rot_optimized,
    apply_input_transform_optimized,
    apply_output_restore_optimized,
    fake_quantize_sym,
    fake_quantize_act_dynamic,
    load_pack,
    pack_weight,
    qmax,
    save_pack,
    transform_weight_for_forward_optimized,
)


def _env(name: str, default: str) -> str:
    """
    Prefer OPENPI_DUQUANT_*, but fallback to GR00T_DUQUANT_* so old scripts still work.
    """
    openpi_name = f"OPENPI_DUQUANT_{name}"
    groot_name = f"GR00T_DUQUANT_{name}"
    if openpi_name in os.environ:
        return os.environ[openpi_name]
    if groot_name in os.environ:
        return os.environ[groot_name]
    return default


@dataclass
class DuQuantConfig:
    weight_bits: Optional[int] = None
    act_bits: Optional[int] = None
    block_size: Optional[int] = None
    block_out_size: Optional[int] = None
    lambda_smooth: Optional[float] = None
    enable_permute: Optional[bool] = None
    act_percentile: Optional[float] = None
    calib_batches: Optional[int] = None
    pack_dir: Optional[str] = None
    row_rot_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weight_bits is None:
            self.weight_bits = int(_env("WBITS_DEFAULT", "4"))
        if self.act_bits is None:
            self.act_bits = int(_env("ABITS", "8"))
        if self.block_size is None:
            self.block_size = int(_env("BLOCK", "64"))
        if self.block_out_size is None:
            self.block_out_size = int(_env("BLOCK_OUT", str(self.block_size)))
        if self.lambda_smooth is None:
            self.lambda_smooth = float(_env("LS", "0.15"))
        if self.enable_permute is None:
            self.enable_permute = _env("PERMUTE", "1") not in ("0", "false", "False")
        if self.act_percentile is None:
            self.act_percentile = float(_env("ACT_PCT", "99.9"))
        if self.calib_batches is None:
            self.calib_batches = int(_env("CALIB_STEPS", "32"))
        if self.pack_dir is None:
            self.pack_dir = os.environ.get("OPENPI_DUQUANT_PACKDIR", os.environ.get("GR00T_DUQUANT_PACKDIR", None))
        if self.row_rot_mode is None:
            self.row_rot_mode = _env("ROW_ROT", "restore")


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


class DuQuantLinear(nn.Module):
    """
    Drop-in fake-quantized replacement for nn.Linear.

    It does:
      x -> DuQuant input transform -> activation fake quant
      W -> DuQuant weight transform -> weight fake quant
      y = F.linear(x_q, W_q)
      optional output restore + bias
    """

    def __init__(
        self,
        base: nn.Linear,
        name: str,
        cfg: DuQuantConfig,
        weight_bits: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.cfg = cfg

        self.weight_bits = int(cfg.weight_bits if weight_bits is None else weight_bits)
        self.act_bits = int(cfg.act_bits)

        self.register_buffer("_weight", base.weight.detach().clone())
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        pack = load_pack(self.name, cfg.pack_dir)
        if pack is None:
            pack = pack_weight(
                self._weight,
                block_size=cfg.block_size,
                block_out_size=cfg.block_out_size,
                enable_permute=cfg.enable_permute,
                lambda_smooth=cfg.lambda_smooth,
            )
            save_pack(self.name, pack, cfg.pack_dir)
        self.pack: PackResult = pack

        if pack.perm is not None:
            self.register_buffer("_perm_cache", torch.from_numpy(pack.perm).long())
        else:
            self._perm_cache = None

        self._R_in_block_indices: List[int] = []
        if pack.R_in_blocks:
            for b, R in pack.R_in_blocks.items():
                self.register_buffer(f"_R_in_{b}", torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_in_block_indices.append(b)

        self._R_out_block_indices: List[int] = []
        if pack.R_out_blocks:
            for b, R in pack.R_out_blocks.items():
                self.register_buffer(f"_R_out_{b}", torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_out_block_indices.append(b)

        self._block_size = int(pack.meta.get("block_size", cfg.block_size))
        self._block_out_size = int(pack.meta.get("block_out_size", cfg.block_out_size))

        self.register_buffer("_W_t", torch.zeros_like(self._weight))
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype))
        self.register_buffer("_W_t_quantized", torch.zeros_like(self._weight))

        self._cached_weight_key: Optional[Tuple[str, torch.dtype, int, int]] = None
        self._weight_quantized_cached = False
        self._bias_rot: Optional[torch.Tensor] = None

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            self._weight.copy_(value)

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

    def _maybe_update_weight_cache(self) -> None:
        apply_row = self.cfg.row_rot_mode != "0"
        key = (str(self._weight.device), self._weight.dtype, int(self.weight_bits), int(apply_row))
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
        )

        self._W_t.copy_(W_t)
        self._w_scales.copy_(scales)

        if self.weight_bits > 0 and self.weight_bits < 16:
            self._W_t_quantized.copy_(
                fake_quantize_sym(W_t, scales[:, None], self.weight_bits, label="weight_prequant")
            )
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

        self._maybe_update_weight_cache()

        if self._weight_quantized_cached:
            y = torch.nn.functional.linear(x_t, self._W_t_quantized, None)
        elif self.weight_bits > 0 and self.weight_bits < 16:
            W_q = fake_quantize_sym(self._W_t, self._w_scales[:, None], self.weight_bits, label="weight_fallback")
            y = torch.nn.functional.linear(x_t, W_q, None)
        else:
            y = torch.nn.functional.linear(x_t, self._W_t, None)

        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            y = apply_output_restore_optimized(
                y,
                self.pack,
                self._get_R_out_cache(),
                self._block_out_size,
            )
            if self.bias is not None:
                y = y + self.bias
        else:
            if self.bias is not None:
                y = y + (self._bias_rot if self._bias_rot is not None else self.bias)

        return y


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _should_exclude(name: str, exclude_regex: Optional[str]) -> bool:
    if exclude_regex is None:
        return False
    return re.search(exclude_regex, name) is not None


def _parse_bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False")


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
    """
    Replace OpenPI model nn.Linear layers with DuQuantLinear.

    Default behavior:
      - replace all nn.Linear
      - unless OPENPI_DUQUANT_INCLUDE / OPENPI_DUQUANT_EXCLUDE is set

    Recommended first run:
      OPENPI_DUQUANT_DRYRUN=1
    """
    if dry_run is None:
        dry_run = _parse_bool_env("OPENPI_DUQUANT_DRYRUN", "0")

    if include_regex is None:
        include_regex = os.environ.get("OPENPI_DUQUANT_INCLUDE", r".*")

    if exclude_regex is None:
        exclude_regex = os.environ.get("OPENPI_DUQUANT_EXCLUDE", "")

    if per_layer_wbits is None:
        per_layer_wbits = _parse_per_layer_wbits(os.environ.get("OPENPI_DUQUANT_WBITS", None))

    cfg = DuQuantConfig() #实例化配置
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

    print(f"[OPENPI-DUQUANT] Matched nn.Linear layers: {len(targets)}")
    print(f"[OPENPI-DUQUANT] Config: W{cfg.weight_bits} A{cfg.act_bits} block={cfg.block_size} "
          f"block_out={cfg.block_out_size} perm={cfg.enable_permute} row_rot={cfg.row_rot_mode}")

    replaced = 0
    bucket_counter = Counter()
    suffix_counter = Counter()
    shape_counter = defaultdict(int)
    def bucket_name(name: str) -> str:
        if name.startswith("paligemma_with_expert.paligemma.vision_tower"):
            return "paligemma.vision_tower"
        if name.startswith("paligemma_with_expert.paligemma.language_model"):
            return "paligemma.language_model"
        if name.startswith("paligemma_with_expert.gemma_expert.model"):
            return "gemma_expert.model"
        if name in {
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        }:
            return "action_io_time"
        return "other"

    for name in targets:
        parent, attr = _get_parent_module_and_attr(model, name)
        old = getattr(parent, attr)

        if not isinstance(old, nn.Linear):
            continue

        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        bucket_counter[bucket_name(name)] += 1
        suffix_counter[name.split(".")[-1]] += 1
        shape_counter[(old.in_features, old.out_features)] += 1


        if dry_run:
            print(f"[OPENPI-DUQUANT][DRYRUN] {name}: Linear({old.in_features}->{old.out_features}) W{wbits} A{cfg.act_bits}")
            continue

        new = DuQuantLinear(old, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, new)

        #print(f"[OPENPI-DUQUANT][REPLACED] {name}: Linear({old.in_features}->{old.out_features}) -> DuQuantLinear W{wbits} A{cfg.act_bits}")
        replaced += 1

    if dry_run:
        print(f"[OPENPI-DUQUANT] Dry-run total layers listed: {len(targets)}")
        return len(targets)

    print(f"[OPENPI-DUQUANT] Total layers replaced: {replaced}")
    print("[OPENPI-DUQUANT] Breakdown by module bucket:")
    for k, v in bucket_counter.most_common():
        print(f"  {k}: {v}")

    print("[OPENPI-DUQUANT] Breakdown by layer suffix:")
    for k, v in suffix_counter.most_common():
        print(f"  {k}: {v}")

    print("[OPENPI-DUQUANT] Top Linear shapes:")
    for (in_f, out_f), v in sorted(shape_counter.items(), key=lambda x: -x[1])[:20]:
        print(f"  Linear({in_f}->{out_f}): {v}")

    return replaced
    


def set_all_duquant_bits(model: nn.Module, *, w_bits: Optional[int] = None, a_bits: Optional[int] = None) -> None:
    """
    Useful later for dynamic quantization experiments.
    """
    for layer in get_duquant_layers(model):
        if w_bits is not None:
            layer.set_w_bits(w_bits)
        if a_bits is not None:
            layer.set_a_bits(a_bits)