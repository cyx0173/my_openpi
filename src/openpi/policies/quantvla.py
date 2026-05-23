def _enable_pi05_atm_capture_jsonl_if_configured(model):
    import json
    import os
    import pathlib
    import threading
    import time

    import torch

    capture_path = os.environ.get("OPENPI_ATM_CAPTURE_PATH", "")
    if not capture_path:
        return

    capture_tag = os.environ.get("OPENPI_ATM_CAPTURE_TAG", "unknown")

    from openpi.models_pytorch.atm.pi05_atm import register_pi05_atm_capture

    path = pathlib.Path(capture_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    f = open(path, "a", encoding="utf-8", buffering=1)

    def cb(layer_name: str, std: torch.Tensor):
        # std shape: (B, H)
        std_head = std.detach().float().mean(dim=0).cpu().tolist()

        record = {
            "timestamp": time.time(),
            "tag": capture_tag,
            "layer": layer_name,
            "std_shape": list(std.shape),
            "std_per_head": std_head,
        }

        with lock:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    count = register_pi05_atm_capture(model, cb, scope="gemma_expert")
    print(
        f"[OPENPI-ATM] capture enabled: path={capture_path}, "
        f"tag={capture_tag}, layers={count}"
    )
def _enable_pi05_ohb_capture_jsonl_if_configured(model):
    import json
    import os
    import pathlib
    import time

    import torch

    capture_path = os.environ.get("OPENPI_OHB_CAPTURE_PATH", "")
    if not capture_path:
        return

    capture_tag = os.environ.get("OPENPI_OHB_CAPTURE_TAG", "unknown")

    from openpi.models_pytorch.atm import register_pi05_ohb_perhead_capture

    path = pathlib.Path(capture_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def cb(layer_name: str, rms_per_head: torch.Tensor):
        record = {
            "timestamp": time.time(),
            "tag": capture_tag,
            "layer": layer_name,
            "branch": "perhead",
            "rms_shape": list(rms_per_head.shape),
            "rms_per_head": rms_per_head.detach().float().cpu().tolist(),
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    count = register_pi05_ohb_perhead_capture(
        model,
        cb,
        scope=os.environ.get("OPENPI_OHB_SCOPE", "gemma_expert"),
    )

    print(
        f"[OPENPI-OHB] capture enabled: path={capture_path}, "
        f"tag={capture_tag}, layers={count}"
    )
def _enable_openpi_duquant_staged(model):
    import gc
    import os
    import torch

    from openpi.models_pytorch.quant import enable_openpi_duquant_all_linears

    def _cleanup(stage_name: str):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"[OPENPI-DUQUANT-STAGED] cleanup done after {stage_name}", flush=True)

    def _run_stage(stage_name: str, include: str, exclude: str = ""):
        print("\n" + "=" * 100, flush=True)
        print(f"[OPENPI-DUQUANT-STAGED] Start {stage_name}", flush=True)
        print(f"[OPENPI-DUQUANT-STAGED] INCLUDE={include}", flush=True)
        print(f"[OPENPI-DUQUANT-STAGED] EXCLUDE={exclude}", flush=True)
        print("=" * 100, flush=True)

        os.environ["OPENPI_DUQUANT_INCLUDE"] = include
        os.environ["OPENPI_DUQUANT_EXCLUDE"] = exclude

        enable_openpi_duquant_all_linears(model)
        _cleanup(stage_name)

    # ------------------------------------------------------------------
    # Stage 1: VLM / PaliGemma 主体。
    # 包括：
    #   paligemma.model.vision_tower
    #   paligemma.model.multi_modal_projector
    #   paligemma.model.language_model
    #
    # 注意：
    #   不要量化 paligemma.lm_head。
    # ------------------------------------------------------------------
    _run_stage(
        "stage1_paligemma_model_original",
        include=r"paligemma_with_expert\.paligemma\.model\..*",
        exclude=(
            r"lm_head"
            r"|embed_tokens"
            r"|rotary_emb"
            r"|\.norm"
            r"|\.input_layernorm"
            r"|\.post_attention_layernorm"
        ),
    )

    # ------------------------------------------------------------------
    # Stage 2: Action Expert MLP + action 输入/时间投影。
    # 包括：
    #   gemma_expert.model.layers.*.mlp.gate_proj
    #   gemma_expert.model.layers.*.mlp.up_proj
    #   gemma_expert.model.layers.*.mlp.down_proj
    #   action_in_proj
    #   time_mlp_in
    #   time_mlp_out
    #
    # 保留：
    #   action_out_proj
    #   gemma_expert attention q/k/v/o
    # ------------------------------------------------------------------
    _run_stage(
        "stage2_action_expert_mlp_and_action_io",
        include=(
            r"paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$"
            r"|^action_in_proj$"
            r"|^time_mlp_in$"
            r"|^time_mlp_out$"
        ),
        exclude=(
            r"lm_head"
            r"|embed_tokens"
            r"|rotary_emb"
            r"|\.norm"
            r"|\.input_layernorm"
            r"|\.post_attention_layernorm"
            r"|\.self_attn\.(q_proj|k_proj|v_proj|o_proj|out_proj)"
            r"|\.attn\.(q_proj|k_proj|v_proj|out_proj)"
            r"|^action_out_proj$"
        ),
    )

    print("[OPENPI-DUQUANT-STAGED] All stages finished.", flush=True)
    return model