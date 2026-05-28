from __future__ import annotations

import json
import pathlib
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


ACTION_PRECISIONS = ("w4a4", "w4a8", "w4a16")

FIXED_W_BITS = 4

PRECISION_TO_A_BITS = {
    "w4a4": 4,
    "w4a8": 8,
    "w4a16": 16,
}


def normalize_action_precision(precision: str) -> str:
    precision = str(precision)
    if precision not in ACTION_PRECISIONS:
        raise ValueError(
            f"Unknown action precision={precision}. "
            f"Expected one of {ACTION_PRECISIONS}."
        )
    return precision


def _is_pi05_action_attention(
    name: str,
    module: nn.Module,
    scope: str = "gemma_expert",
) -> bool:
    if module.__class__.__name__ != "GemmaAttention":
        return False

    if scope == "gemma_expert":
        return (
            name.startswith("paligemma_with_expert.gemma_expert.model.layers.")
            and name.endswith(".self_attn")
        )

    if scope == "all_gemma":
        return name.endswith(".self_attn")

    return scope in name


def _is_action_fakequant_linear(
    name: str,
    module: nn.Module,
    scope: str = "gemma_expert",
) -> bool:
    if not (hasattr(module, "set_w_bits") and hasattr(module, "set_a_bits")):
        return False

    if scope == "gemma_expert":
        return name.startswith("paligemma_with_expert.gemma_expert.")

    if scope == "all_gemma":
        return "gemma" in name

    return scope in name


def _load_json(path: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _lookup_layer_entry(data: dict[str, Any], name: str) -> dict[str, Any] | None:
    return (
        data.get(name)
        or data.get(name.replace("model.", "model", 1))
        or data.get(name.replace("paligemma_with_expert.", "", 1))
    )


def select_runtime_atm_alpha(attn: nn.Module) -> torch.Tensor | None:
    """
    Used inside GemmaAttention forward.

    Runtime semantics:
      method == "atm"
      precision in {w4a4, w4a8, w4a16}
      return alpha_bank[precision]

    If runtime is not installed, fall back to old fixed ATM attribute.
    """
    if not getattr(attn, "_pi05_action_runtime_enabled", False):
        return getattr(attn, "_atm_alpha_all", None)

    method = getattr(attn, "_pi05_action_runtime_method", "none")
    precision = getattr(attn, "_pi05_action_runtime_precision", "w4a16")

    if method != "atm":
        return None

    precision = normalize_action_precision(precision)
    bank = getattr(attn, "_pi05_atm_alpha_bank", {})
    return bank.get(precision, None)


def select_runtime_ohb_beta_perhead(attn: nn.Module) -> torch.Tensor | None:
    """
    Used inside GemmaAttention forward.

    Runtime semantics:
      method == "ohb"
      precision in {w4a4, w4a8, w4a16}
      return beta_perhead_bank[precision]

    If runtime is not installed, fall back to old fixed OHB attribute.
    """
    if not getattr(attn, "_pi05_action_runtime_enabled", False):
        return getattr(attn, "_ohb_beta_perhead", None)

    method = getattr(attn, "_pi05_action_runtime_method", "none")
    precision = getattr(attn, "_pi05_action_runtime_precision", "w4a16")

    if method != "ohb":
        return None

    precision = normalize_action_precision(precision)
    bank = getattr(attn, "_pi05_ohb_beta_perhead_bank", {})
    return bank.get(precision, None)


def select_runtime_ohb_beta_scalar(attn: nn.Module) -> float | None:
    """
    Used inside GemmaAttention forward.

    Runtime semantics:
      method == "ohb"
      precision in {w4a4, w4a8, w4a16}
      return beta_scalar_bank[precision]

    If runtime is not installed, fall back to old fixed OHB scalar attribute.
    """
    if not getattr(attn, "_pi05_action_runtime_enabled", False):
        return getattr(attn, "_ohb_beta_scalar", None)

    method = getattr(attn, "_pi05_action_runtime_method", "none")
    precision = getattr(attn, "_pi05_action_runtime_precision", "w4a16")

    if method != "ohb":
        return None

    precision = normalize_action_precision(precision)
    bank = getattr(attn, "_pi05_ohb_beta_scalar_bank", {})
    return bank.get(precision, None)


def install_pi05_action_runtime_banks(
    model: nn.Module,
    *,
    method: str,
    atm_paths: dict[str, str] | None = None,
    ohb_paths: dict[str, str] | None = None,
    scope: str = "gemma_expert",
    default_precision: str = "w4a16",
) -> dict[str, int]:
    """
    Install runtime action quantization banks.

    Important:
      - This runtime only supports action precisions:
            w4a4 / w4a8 / w4a16
      - Weight bits are fixed to W4.
      - Runtime switching only changes activation bits:
            A4 / A8 / A16
      - ATM/OHB banks are selected by current precision.
      - No fp16 action runtime mode is supported here.
    """
    if method not in {"atm", "ohb", "none"}:
        raise ValueError(f"Unknown runtime method={method}. Expected atm / ohb / none.")

    default_precision = normalize_action_precision(default_precision)

    atm_data = {k: _load_json(v) for k, v in (atm_paths or {}).items() if v}
    ohb_data = {k: _load_json(v) for k, v in (ohb_paths or {}).items() if v}

    atm_data = {k: v for k, v in atm_data.items() if k in ACTION_PRECISIONS}
    ohb_data = {k: v for k, v in ohb_data.items() if k in ACTION_PRECISIONS}

    matched_attn = 0
    matched_alpha = 0
    matched_beta = 0
    matched_fakequant = 0

    for name, module in model.named_modules():
        if not _is_pi05_action_attention(name, module, scope=scope):
            continue

        matched_attn += 1

        setattr(module, "_pi05_action_runtime_enabled", True)
        setattr(module, "_pi05_action_runtime_method", method)
        setattr(module, "_pi05_action_runtime_precision", default_precision)

        alpha_bank: dict[str, torch.Tensor] = {}
        beta_perhead_bank: dict[str, torch.Tensor] = {}
        beta_scalar_bank: dict[str, float] = {}

        for precision, data in atm_data.items():
            entry = _lookup_layer_entry(data, name)
            if not entry:
                continue

            alpha_values = entry.get("all") or entry.get("alpha")
            if alpha_values is not None:
                alpha_bank[precision] = torch.tensor(alpha_values, dtype=torch.float32)
                matched_alpha += 1

        for precision, data in ohb_data.items():
            entry = _lookup_layer_entry(data, name)
            if not entry:
                continue

            beta_perhead = entry.get("beta_perhead")
            if beta_perhead is not None:
                beta_perhead_bank[precision] = torch.tensor(beta_perhead, dtype=torch.float32)
                matched_beta += 1
                continue

            beta = entry.get("beta", None)
            if beta is not None:
                beta_scalar_bank[precision] = float(beta)
                matched_beta += 1

        setattr(module, "_pi05_atm_alpha_bank", alpha_bank)
        setattr(module, "_pi05_ohb_beta_perhead_bank", beta_perhead_bank)
        setattr(module, "_pi05_ohb_beta_scalar_bank", beta_scalar_bank)

    for name, module in model.named_modules():
        if not _is_action_fakequant_linear(name, module, scope=scope):
            continue

        matched_fakequant += 1

        # Runtime action quantization always uses W4.
        # Activation bits are switched dynamically by set_precision().
        if hasattr(module, "set_w_bits"):
            module.set_w_bits(FIXED_W_BITS)

        if hasattr(module, "set_a_bits"):
            module.set_a_bits(PRECISION_TO_A_BITS[default_precision])

    runtime = Pi05ActionQuantRuntime(
        model,
        method=method,
        scope=scope,
        default_precision=default_precision,
    )

    model._pi05_action_runtime_method = method
    model._pi05_action_runtime_scope = scope
    model._pi05_action_runtime = runtime

    return {
        "matched_attn": matched_attn,
        "matched_alpha": matched_alpha,
        "matched_beta": matched_beta,
        "matched_fakequant": matched_fakequant,
        "default_precision": default_precision,
    }


class Pi05ActionQuantRuntime:
    def __init__(
        self,
        model: nn.Module,
        *,
        method: str,
        scope: str = "gemma_expert",
        default_precision: str = "w4a16",
    ):
        self.model = model
        self.method = method
        self.scope = scope
        self.default_precision = normalize_action_precision(default_precision)

        self.attn_modules: list[tuple[str, nn.Module]] = []
        self.fakequant_modules: list[tuple[str, nn.Module]] = []

        for name, module in model.named_modules():
            if _is_pi05_action_attention(name, module, scope=scope):
                self.attn_modules.append((name, module))

            if _is_action_fakequant_linear(name, module, scope=scope):
                self.fakequant_modules.append((name, module))

    @property
    def num_attn_modules(self) -> int:
        return len(self.attn_modules)

    @property
    def num_fakequant_modules(self) -> int:
        return len(self.fakequant_modules)

    def set_precision(self, precision: str) -> None:
        """
        Switch current action precision.

        Allowed:
          w4a4 / w4a8 / w4a16

        Semantics:
          W is always fixed to 4.
          A is dynamically set to 4 / 8 / 16.
          ATM/OHB forward will select bank[precision].
        """
        precision = normalize_action_precision(precision)
        a_bits = PRECISION_TO_A_BITS[precision]

        for _, module in self.fakequant_modules:
            if hasattr(module, "set_w_bits"):
                module.set_w_bits(FIXED_W_BITS)
            module.set_a_bits(a_bits)

        for _, module in self.attn_modules:
            setattr(module, "_pi05_action_runtime_method", self.method)
            setattr(module, "_pi05_action_runtime_precision", precision)

    def get_precision(self) -> str:
        for _, module in self.attn_modules:
            precision = getattr(module, "_pi05_action_runtime_precision", self.default_precision)
            return normalize_action_precision(precision)
        return self.default_precision

    @contextmanager
    def use_precision(self, precision: str | None):
        if precision is None:
            yield
            return

        precision = normalize_action_precision(precision)

        old_attn: list[tuple[nn.Module, str, str]] = []
        old_fakequant: list[tuple[nn.Module, int]] = []

        for _, module in self.attn_modules:
            old_method = getattr(module, "_pi05_action_runtime_method", self.method)
            old_precision = getattr(module, "_pi05_action_runtime_precision", self.default_precision)
            old_precision = normalize_action_precision(old_precision)
            old_attn.append((module, old_method, old_precision))

        for _, module in self.fakequant_modules:
            old_a_bits = int(getattr(module, "current_a_bits", PRECISION_TO_A_BITS[self.default_precision]))
            old_fakequant.append((module, old_a_bits))

        self.set_precision(precision)

        try:
            yield
        finally:
            for module, old_method, old_precision in old_attn:
                setattr(module, "_pi05_action_runtime_method", old_method)
                setattr(module, "_pi05_action_runtime_precision", old_precision)

            for module, old_a_bits in old_fakequant:
                if hasattr(module, "set_w_bits"):
                    module.set_w_bits(FIXED_W_BITS)
                module.set_a_bits(old_a_bits)


