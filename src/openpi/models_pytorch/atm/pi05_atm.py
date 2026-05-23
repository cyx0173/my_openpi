import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch


OPENPI_ATM_ENABLE_ENV = "OPENPI_ATM_ENABLE"
OPENPI_ATM_ALPHA_PATH_ENV = "OPENPI_ATM_ALPHA_PATH"
OPENPI_ATM_SCOPE_ENV = "OPENPI_ATM_SCOPE"


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


def register_pi05_atm_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "gemma_expert",
) -> int:
 
    count = 0

    for name, module in model.named_modules():
        if _is_pi05_action_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_capture_callback",
                lambda attn, std, layer=name: callback(layer, std),
            )
            setattr(module, "_atm_capture_name", name)
            count += 1

    print(f"[OPENPI-ATM] Registered ATM capture on {count} PI0.5 attention layers.")
    return count


def register_pi05_atm_logits_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "gemma_expert",
) -> int:
    """
    Optional debug hook. Captures full logits tensor.

    callback receives:
        layer_name: str
        logits: Tensor, shape (B, H, Q, K)

    Warning:
        This can use a lot of memory.
    """
    count = 0

    for name, module in model.named_modules():
        if _is_pi05_action_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_logits_capture_callback",
                lambda attn, logits, layer=name: callback(layer, logits),
            )
            setattr(module, "_atm_logits_capture_name", name)
            count += 1

    print(f"[OPENPI-ATM] Registered ATM logits capture on {count} PI0.5 attention layers.")
    return count

def enable_pi05_atm_alpha_ones(
    model: torch.nn.Module,
    scope: str = "gemma_expert",
) -> tuple[int, int]:
    """
    Enable PI0.5 ATM with alpha=1 for sanity check.

    This should not change model outputs. It is used to verify that
    attaching ATM attributes has no side effect when alpha is neutral.
    """
    matched_layers = 0
    total_heads = 0

    for name, module in model.named_modules():
        if _is_pi05_action_attention(name, module, scope=scope):
            if not hasattr(module, "q_proj") or not hasattr(module, "head_dim"):
                raise RuntimeError(
                    f"Cannot infer num_heads for ATM alpha=1 layer: {name}. "
                    f"module={module.__class__.__name__}"
                )

            num_heads = int(module.q_proj.out_features // module.head_dim)

            module._atm_alpha_all = torch.ones(
                num_heads,
                dtype=torch.float32,
            )

            matched_layers += 1
            total_heads += num_heads

    print(
        f"[OPENPI-ATM] Enabled alpha=1 sanity ATM for "
        f"{matched_layers} layers, {total_heads} heads."
    )

    return matched_layers, total_heads

def clear_pi05_atm_capture(model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        for attr in (
            "_atm_capture_callback",
            "_atm_capture_name",
            "_atm_logits_capture_callback",
            "_atm_logits_capture_name",
        ):
            if hasattr(module, attr):
                delattr(module, attr)


@dataclass
class _ATMSummary:
    matched_layers: int = 0
    total_heads: int = 0


def enable_pi05_atm_if_configured(model: torch.nn.Module) -> None:
    """
    Enable PI0.5 ATM from environment variables.

    Required:
        OPENPI_ATM_ENABLE=1
        OPENPI_ATM_ALPHA_PATH=/path/to/pi05_atm_alpha.json

    Optional:
        OPENPI_ATM_SCOPE=gemma_expert
    """
    enabled = os.environ.get(OPENPI_ATM_ENABLE_ENV, "0") not in (
        "0",
        "false",
        "False",
        "",
    )
    if not enabled:
        return

    alpha_path = os.environ.get(OPENPI_ATM_ALPHA_PATH_ENV)
    if not alpha_path:
        print("[OPENPI-ATM] OPENPI_ATM_ENABLE=1 but OPENPI_ATM_ALPHA_PATH is not set; skipping.")
        return

    if not os.path.exists(alpha_path):
        print(f"[OPENPI-ATM] Alpha JSON not found: {alpha_path}; skipping.")
        return

    with open(alpha_path, "r", encoding="utf-8") as f:
        alpha_data = json.load(f)

    scope = os.environ.get(OPENPI_ATM_SCOPE_ENV, "gemma_expert")
    summary = _ATMSummary()

    for name, module in model.named_modules():
        if not _is_pi05_action_attention(name, module, scope=scope):
            continue

        entry = alpha_data.get(name)
        if not entry:
            # Allow a relaxed fallback in case a wrapper prefix changes.
            short_name = name.replace("model.", "model", 1)
            entry = alpha_data.get(short_name)

        if not entry:
            continue

        alpha_values = entry.get("alpha") or entry.get("all")
        if not alpha_values:
            continue

        alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32)
        setattr(module, "_atm_alpha_all", alpha_tensor)

        summary.matched_layers += 1
        summary.total_heads += len(alpha_values)

    if summary.matched_layers == 0:
        print(f"[OPENPI-ATM] No PI0.5 action attention layers matched alpha JSON: {alpha_path}")
    else:
        print(
            f"[OPENPI-ATM] Enabled ATM for {summary.matched_layers} PI0.5 action attention layers "
            f"({summary.total_heads} heads) using {alpha_path}"
        )


def clear_pi05_atm_alpha(model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        if hasattr(module, "_atm_alpha_all"):
            delattr(module, "_atm_alpha_all")