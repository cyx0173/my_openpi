import logging
import os
import pathlib
from typing import Any
import json
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms
from openpi.policies.quantvla import _enable_openpi_duquant_staged, _debug_duquant_state, _calibrate_openpi_duquant_staged
def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    quantize: bool = False,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

        if quantize:
            wbits = int(os.environ.get("OPENPI_DUQUANT_WBITS_DEFAULT", "4"))
            abits = int(os.environ.get("OPENPI_DUQUANT_ABITS", "4"))
            print(f"[OPENPI] Quantizing with wbits={wbits}, abits={abits}", flush=True)
            if abits not in {2, 4, 8, 16}:
                raise ValueError(f"Unsupported OPENPI_DUQUANT_ABITS={abits}")
            from openpi.policies.quantvla import (
                _enable_openpi_duquant_staged,
                _enable_openpi_smoothvla_staged,
            )

            quant_backend = os.environ.get("OPENPI_QUANT_BACKEND", "duquant").strip().lower()

            if quant_backend in {"smoothvla", "smoothquant", "smooth_w4", "sq_w4"}:
                model = _enable_openpi_smoothvla_staged(model)
                model._modulewise_abits.build()
                print(
                    "[MODULEWISE-ABITS] built "
                    f"vlm_layers={len(model._modulewise_abits.vlm_layers)} "
                    f"action_layers={len(model._modulewise_abits.action_layers)} "
                    f"other_layers={len(model._modulewise_abits.other_layers)}",
                    flush=True,
                )
            else:
                model = _enable_openpi_duquant_staged(model, wbits=wbits, abits=abits)
        else:
            _calibrate_openpi_duquant_staged(model)
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    # Load norm_stats from checkpoint/assets directly, bypassing data_config's _load_norm_stats
    # which would try the local assets_dir first (causing a spurious "not found" warning).
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"
    
    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )

'''
#收益测试
if os.environ.get("OPENPI_DUMP_QUANT_BENEFIT", "0") == "1":
            from openpi.policies.quantvla import _dump_pi05_linear_quant_benefit

            _dump_pi05_linear_quant_benefit(
                model,
                out_csv="/home/chengyuxuan/openpi/lab_track/quant_benefit/linear_quant_benefit.csv",
                out_summary="/home/chengyuxuan/openpi/lab_track/quant_benefit/linear_quant_benefit_summary.txt",
            )

# atm & ohb 
        _enable_pi05_atm_capture_jsonl_if_configured(model)
        _enable_pi05_ohb_capture_jsonl_if_configured(model)

        duquant_layout = os.environ.get("OPENPI_DUQUANT_LAYOUT", "quantvla_selective").strip()
        disable_atm_ohb = os.environ.get("OPENPI_DISABLE_ATM_OHB", "0") == "1"

        if duquant_layout.startswith("naive"):
            disable_atm_ohb = True

        if disable_atm_ohb:
            print(
                "[OPENPI] Skip ATM/OHB repair: "
                f"OPENPI_DUQUANT_LAYOUT={duquant_layout}, "
                f"OPENPI_DISABLE_ATM_OHB={os.environ.get('OPENPI_DISABLE_ATM_OHB', '0')}",
                flush=True,
            )
        else:
            from openpi.models_pytorch.atm import (
                enable_pi05_atm_if_configured,
                enable_pi05_atm_alpha_ones,
                enable_pi05_ohb_if_configured,
                enable_pi05_ohb_beta_ones,
                enable_pi05_ohb_beta_constant,
            )

            if os.environ.get("OPENPI_ATM_ALPHA_ONES", "0") == "1":
                enable_pi05_atm_alpha_ones(
                    model,
                    scope=os.environ.get("OPENPI_ATM_SCOPE", "gemma_expert"),
                )
            else:
                enable_pi05_atm_if_configured(model)

            if os.environ.get("OPENPI_OHB_BETA_ONES", "0") == "1":
                enable_pi05_ohb_beta_ones(
                    model,
                    scope=os.environ.get("OPENPI_OHB_SCOPE", "gemma_expert"),
                )
            elif os.environ.get("OPENPI_OHB_BETA_CONSTANT", "") != "":
                enable_pi05_ohb_beta_constant(
                    model,
                    beta=float(os.environ["OPENPI_OHB_BETA_CONSTANT"]),
                    scope=os.environ.get("OPENPI_OHB_SCOPE", "gemma_expert"),
                )
            else:
                enable_pi05_ohb_if_configured(model)

        if quantize:
            _debug_duquant_state(
                model,
                tag=(
                    f"layout_{duquant_layout}_"
                    f"mode_{os.environ.get('OPENPI_QUANT_MODE', 'unknown')}_"
                    f"atmohb_disabled_{int(disable_atm_ohb)}"
                ),
            )
'''