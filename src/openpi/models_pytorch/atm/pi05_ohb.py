import json
import os
from dataclasses import dataclass
from typing import Callable

import torch


OPENPI_OHB_ENABLE_ENV = "OPENPI_OHB_ENABLE"
OPENPI_OHB_BETA_PATH_ENV = "OPENPI_OHB_BETA_PATH"
OPENPI_OHB_SCOPE_ENV = "OPENPI_OHB_SCOPE"

def enable_pi05_ohb_beta_constant(
    model: torch.nn.Module,
    beta: float,
    scope: str = "gemma_expert",
) -> tuple[int, int]:
    matched_layers = 0
    total_heads = 0

    for name, module in model.named_modules():
        if _is_pi05_action_attention(name, module, scope=scope):
            num_heads = int(module.q_proj.out_features // module.head_dim)
            module._ohb_beta_perhead = torch.full(
                (num_heads,),
                float(beta),
                dtype=torch.float32,
            )
            matched_layers += 1
            total_heads += num_heads

    print(
        f"[OPENPI-OHB] Enabled beta={beta} per-head OHB for "
        f"{matched_layers} layers, {total_heads} heads."
    )
    return matched_layers, total_heads

def _is_pi05_action_attention(
    name: str,
    module: torch.nn.Module,
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


def register_pi05_ohb_perhead_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "gemma_expert",
) -> int:
    count = 0

    for name, module in model.named_modules():
        if _is_pi05_action_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_ohb_perhead_capture_callback",
                lambda attn, rms, layer=name: callback(layer, rms),
            )
            setattr(module, "_atm_ohb_perhead_capture_name", name)
            count += 1

    print(f"[OPENPI-OHB] Registered per-head OHB capture on {count} PI0.5 attention layers.")
    return count


def clear_pi05_ohb_capture(model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        for attr in (
            "_atm_ohb_perhead_capture_callback",
            "_atm_ohb_perhead_capture_name",
        ):
            if hasattr(module, attr):
                delattr(module, attr)


def clear_pi05_ohb_beta(model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        if hasattr(module, "_ohb_beta_perhead"):
            delattr(module, "_ohb_beta_perhead")


def enable_pi05_ohb_beta_ones(
    model: torch.nn.Module,
    scope: str = "gemma_expert",
) -> tuple[int, int]:
    matched_layers = 0
    total_heads = 0

    for name, module in model.named_modules():
        if _is_pi05_action_attention(name, module, scope=scope):
            num_heads = int(module.q_proj.out_features // module.head_dim)
            module._ohb_beta_perhead = torch.ones(num_heads, dtype=torch.float32)
            matched_layers += 1
            total_heads += num_heads

    print(
        f"[OPENPI-OHB] Enabled beta=1 per-head OHB for "
        f"{matched_layers} layers, {total_heads} heads."
    )
    return matched_layers, total_heads


@dataclass
class _OHBSummary:
    matched_layers: int = 0
    total_heads: int = 0


def enable_pi05_ohb_if_configured(model: torch.nn.Module) -> None:
    enabled = os.environ.get(OPENPI_OHB_ENABLE_ENV, "0") not in (
        "0",
        "false",
        "False",
        "",
    )
    if not enabled:
        return

    beta_path = os.environ.get(OPENPI_OHB_BETA_PATH_ENV)
    if not beta_path:
        print("[OPENPI-OHB] OPENPI_OHB_ENABLE=1 but OPENPI_OHB_BETA_PATH is not set; skipping.")
        return

    if not os.path.exists(beta_path):
        print(f"[OPENPI-OHB] Beta JSON not found: {beta_path}; skipping.")
        return

    with open(beta_path, "r", encoding="utf-8") as f:
        beta_data = json.load(f)

    scope = os.environ.get(OPENPI_OHB_SCOPE_ENV, "gemma_expert")
    summary = _OHBSummary()

    for name, module in model.named_modules():
        if not _is_pi05_action_attention(name, module, scope=scope):
            continue

        entry = beta_data.get(name)
        if not entry:
            continue

        beta_values = entry.get("beta_perhead")
        if beta_values is None:
            continue

        beta_tensor = torch.tensor(beta_values, dtype=torch.float32)
        setattr(module, "_ohb_beta_perhead", beta_tensor)

        summary.matched_layers += 1
        summary.total_heads += len(beta_values)

    if summary.matched_layers == 0:
        print(f"[OPENPI-OHB] No PI0.5 attention layers matched beta JSON: {beta_path}")
    else:
        print(
            f"[OPENPI-OHB] Enabled per-head OHB for {summary.matched_layers} PI0.5 attention layers "
            f"({summary.total_heads} heads) using {beta_path}"
        )