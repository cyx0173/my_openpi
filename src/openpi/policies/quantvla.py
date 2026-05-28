#OPENPI_DUQUANT_LAYOUT=naive_vlm_action_selective 这里其实不应该传入这个环境变量 我们后面应该是默认第一种量化策略
def _enable_openpi_duquant_staged(model):
    import gc
    import os
    import time

    import torch

    from openpi.models_pytorch.quant import enable_openpi_duquant_all_linears

    os.environ["OPENPI_DUQUANT_PACKDIR"] = "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant_packed"
    os.environ["OPENPI_DUQUANT_INT4_CACHE_TAG"] = "default"
    os.environ["OPENPI_DUQUANT_INT4_CACHE"] = "1"
    os.environ["OPENPI_DUQUANT_INT4_CACHE_DIR"] = (
        "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant_int4_cache"
    )

    os.environ["OPENPI_DUQUANT_WEIGHT_BACKEND"] = "fused_w4"
    os.environ["OPENPI_DUQUANT_PACKED_BACKEND"] = "kernel"
    os.environ["OPENPI_DUQUANT_FUSED_W4"] = "1"

    # Activation quantization:
    #   W4A4/W4A8: stateless dynamic per-token absmax fake quant, FP32 internal.
    #   W4A16: native BF16/FP16 activation, no activation fake quant.
    os.environ.setdefault("OPENPI_DUQUANT_ACT_QUANT_MODE", "dynamic_token_amax")
    os.environ.setdefault("OPENPI_DUQUANT_FAKEQ_INTERNAL", "fp32")
    os.environ.setdefault("OPENPI_DUQUANT_ACT_BACKEND", "fake")
    os.environ.setdefault("TRITON_CACHE_DIR", "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/.triton_cache")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/.torchinductor_cache")

    quant_mode = os.environ.get("OPENPI_QUANT_MODE", "unknown")
    print(
        "[OPENPI-DUQUANT] "
        f"mode={quant_mode} "
        "backend=fused_w4 "
        f"act_quant_mode={os.environ['OPENPI_DUQUANT_ACT_QUANT_MODE']} "
        f"int4_cache_tag={os.environ['OPENPI_DUQUANT_INT4_CACHE_TAG']}",
        flush=True,
    )

    def _cleanup(stage_name: str):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"[OPENPI-DUQUANT] cleanup after {stage_name}", flush=True)

    def _count_duquant_linear() -> int:
        from openpi.models_pytorch.quant.duquant_layers import DuQuantLinear

        return sum(1 for m in model.modules() if isinstance(m, DuQuantLinear))

    def _run_stage(stage_name: str, include: str, exclude: str = ""):
        print(f"[OPENPI-DUQUANT] start {stage_name}", flush=True)
        t0 = time.perf_counter()
        before = _count_duquant_linear()

        os.environ["OPENPI_DUQUANT_INCLUDE"] = include
        os.environ["OPENPI_DUQUANT_EXCLUDE"] = exclude

        enable_openpi_duquant_all_linears(model)
        _cleanup(stage_name)

        after = _count_duquant_linear()
        print(
            f"[OPENPI-DUQUANT] done {stage_name}: "
            f"duquant_layers_before={before}, after={after}, "
            f"added={after - before}, dt={time.perf_counter() - t0:.2f}s",
            flush=True,
        )

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

    action_exclude = (
        common_exclude
        + r"|\.self_attn\.(q_proj|k_proj|v_proj|o_proj|out_proj)"
        + r"|\.attn\.(q_proj|k_proj|v_proj|out_proj)"
        + r"|^action_out_proj$"
    )

    _run_stage(
        "stage1_vlm_paligemma_model",
        include=vlm_include,
        exclude=common_exclude,
    )

    _run_stage(
        "stage2_action_expert_mlp_and_action_io",
        include=action_mlp_io_include,
        exclude=action_exclude,
    )

    print("[OPENPI-DUQUANT] converting DuQuantLinear -> DuQuantFusedW4Linear", flush=True)
    t0 = time.perf_counter()

    from openpi.models_pytorch.quant.duquant_fused_w4 import convert_duquant_to_fused_w4

    converted = convert_duquant_to_fused_w4(model)
    _cleanup("fused_w4_conversion")

    print(
        f"[OPENPI-DUQUANT] fused_w4 converted_layers={converted}, "
        f"dt={time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    return model


def _debug_duquant_state(model, tag: str = "unknown"):
    from collections import Counter

    bit_counter = Counter()
    class_counter = Counter()
    examples = []

    for name, module in model.named_modules():
        has_bit_api = hasattr(module, "set_w_bits") and hasattr(module, "set_a_bits")
        class_name = module.__class__.__name__

        if has_bit_api or "Quant" in class_name or "Fake" in class_name:
            w_bits = getattr(module, "current_w_bits", None)
            a_bits = getattr(module, "current_a_bits", None)

            if w_bits is None:
                w_bits = getattr(module, "w_bits", None)
            if a_bits is None:
                a_bits = getattr(module, "a_bits", None)

            bit_counter[(w_bits, a_bits)] += 1
            class_counter[class_name] += 1

            if len(examples) < 50:
                examples.append((name, class_name, w_bits, a_bits))

    print(f"\n[DUQUANT-DEBUG] tag={tag}", flush=True)
    print(f"[DUQUANT-DEBUG] quant_like_modules={sum(bit_counter.values())}", flush=True)
    print(f"[DUQUANT-DEBUG] bit_counter={dict(bit_counter)}", flush=True)
    print(f"[DUQUANT-DEBUG] class_counter={dict(class_counter)}", flush=True)

    for name, class_name, w_bits, a_bits in examples:
        print(
            f"[DUQUANT-DEBUG] example name={name} class={class_name} "
            f"W={w_bits} A={a_bits}",
            flush=True,
        )

def _dump_pi05_linear_quant_benefit(
    model,
    out_csv: str = "/home/chengyuxuan/openpi/lab_track/linear_quant_benefit.csv",
    out_summary: str = "/home/chengyuxuan/openpi/lab_track/linear_quant_benefit_summary.txt",
):
    import csv
    import os
    import re
    from collections import defaultdict

    import torch
    from torch import nn

    try:
        from openpi.models_pytorch.quant.duquant_layers import DuQuantLinear
    except Exception:
        DuQuantLinear = tuple()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

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

    action_attn_include = (
        r"paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj|out_proj)$"
    )

    action_out_include = r"^action_out_proj$"

    def match(pattern: str, name: str) -> bool:
        return re.search(pattern, name) is not None

    def excluded(name: str) -> bool:
        return match(common_exclude, name)

    def in_selective(name: str) -> bool:
        if excluded(name):
            return False
        if match(vlm_include, name):
            return True
        if match(action_mlp_io_include, name):
            return True
        return False

    def in_plus_attn(name: str) -> bool:
        if in_selective(name):
            return True
        if excluded(name):
            return False
        if match(action_attn_include, name):
            return True
        return False

    def in_full(name: str) -> bool:
        if in_plus_attn(name):
            return True
        if excluded(name):
            return False
        if match(action_out_include, name):
            return True
        return False

    def branch_of(name: str) -> str:
        if name.startswith("paligemma_with_expert.paligemma.model.vision_tower"):
            return "vlm_vision_tower"
        if name.startswith("paligemma_with_expert.paligemma.model.multi_modal_projector"):
            return "vlm_multi_modal_projector"
        if name.startswith("paligemma_with_expert.paligemma.model.language_model"):
            return "vlm_language_model"
        if re.search(r"paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.self_attn", name):
            return "action_expert_attention"
        if re.search(r"paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\.mlp", name):
            return "action_expert_mlp"
        if name in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}:
            return "action_io_time"
        return "other"

    def suffix_of(name: str) -> str:
        return name.split(".")[-1]

    def is_linear_like(module) -> bool:
        if isinstance(module, nn.Linear):
            return True
        if DuQuantLinear and isinstance(module, DuQuantLinear):
            return True
        return hasattr(module, "in_features") and hasattr(module, "out_features") and hasattr(module, "weight")

    def mb(x: float) -> float:
        return x / 1024**2

    rows = []

    for name, module in model.named_modules():
        if not is_linear_like(module):
            continue

        weight = getattr(module, "weight", None)
        if weight is None:
            continue

        try:
            weight_params = int(weight.numel())
        except Exception:
            weight_params = int(module.in_features) * int(module.out_features)

        bias = getattr(module, "bias", None)
        bias_params = int(bias.numel()) if bias is not None else 0

        total_params = weight_params + bias_params

        # BF16/FP16: 2 bytes per param.
        fp16_bytes = total_params * 2

        # 理论 W4 权重部署：
        # weight: 4-bit = 0.5 byte
        # bias: keep fp16/bf16 = 2 bytes
        # scale/zero overhead 暂时忽略，通常远小于 weight。
        w4_bytes = weight_params * 0.5 + bias_params * 2

        saved_bytes = fp16_bytes - w4_bytes

        row = {
            "name": name,
            "branch": branch_of(name),
            "suffix": suffix_of(name),
            "in_features": int(getattr(module, "in_features", -1)),
            "out_features": int(getattr(module, "out_features", -1)),
            "weight_params": weight_params,
            "bias_params": bias_params,
            "total_params": total_params,
            "fp16_mb": mb(fp16_bytes),
            "w4_weight_mb": mb(w4_bytes),
            "saved_mb_if_w4": mb(saved_bytes),
            "selective": int(in_selective(name)),
            "plus_attn": int(in_plus_attn(name)),
            "full": int(in_full(name)),
        }

        rows.append(row)

    rows.sort(key=lambda x: x["fp16_mb"], reverse=True)

    fieldnames = [
        "name",
        "branch",
        "suffix",
        "in_features",
        "out_features",
        "weight_params",
        "bias_params",
        "total_params",
        "fp16_mb",
        "w4_weight_mb",
        "saved_mb_if_w4",
        "selective",
        "plus_attn",
        "full",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    layouts = ["selective", "plus_attn", "full"]

    total_fp16 = sum(r["fp16_mb"] for r in rows)
    total_w4_all = sum(r["w4_weight_mb"] for r in rows)
    total_saved_all = sum(r["saved_mb_if_w4"] for r in rows)

    by_branch = defaultdict(lambda: {"count": 0, "fp16_mb": 0.0, "saved_mb": 0.0})
    by_suffix = defaultdict(lambda: {"count": 0, "fp16_mb": 0.0, "saved_mb": 0.0})

    for r in rows:
        by_branch[r["branch"]]["count"] += 1
        by_branch[r["branch"]]["fp16_mb"] += r["fp16_mb"]
        by_branch[r["branch"]]["saved_mb"] += r["saved_mb_if_w4"]

        by_suffix[r["suffix"]]["count"] += 1
        by_suffix[r["suffix"]]["fp16_mb"] += r["fp16_mb"]
        by_suffix[r["suffix"]]["saved_mb"] += r["saved_mb_if_w4"]

    with open(out_summary, "w", encoding="utf-8") as f:
        def w(line: str = ""):
            print(line, file=f)

        w("=" * 100)
        w("[PI0.5 LINEAR QUANT BENEFIT]")
        w("=" * 100)
        w(f"Total Linear layers: {len(rows)}")
        w(f"All Linear FP16/BF16 memory: {total_fp16:.2f} MB")
        w(f"All Linear theoretical W4-weight memory: {total_w4_all:.2f} MB")
        w(f"All Linear theoretical saved memory: {total_saved_all:.2f} MB")
        w()

        w("[Layout Summary]")
        for layout in layouts:
            selected = [r for r in rows if r[layout] == 1]
            fp16_mb = sum(r["fp16_mb"] for r in selected)
            w4_mb = sum(r["w4_weight_mb"] for r in selected)
            saved_mb = sum(r["saved_mb_if_w4"] for r in selected)
            w(
                f"{layout:12s}: "
                f"layers={len(selected):4d}, "
                f"fp16={fp16_mb:10.2f} MB, "
                f"w4={w4_mb:10.2f} MB, "
                f"saved={saved_mb:10.2f} MB, "
                f"cover_linear_mem={fp16_mb / max(total_fp16, 1e-9) * 100:6.2f}%"
            )

        w()
        w("[By Branch]")
        for k, v in sorted(by_branch.items(), key=lambda x: -x[1]["fp16_mb"]):
            w(
                f"{k:32s} "
                f"layers={v['count']:4d}, "
                f"fp16={v['fp16_mb']:10.2f} MB, "
                f"saved_if_w4={v['saved_mb']:10.2f} MB"
            )

        w()
        w("[By Suffix]")
        for k, v in sorted(by_suffix.items(), key=lambda x: -x[1]["fp16_mb"]):
            w(
                f"{k:16s} "
                f"layers={v['count']:4d}, "
                f"fp16={v['fp16_mb']:10.2f} MB, "
                f"saved_if_w4={v['saved_mb']:10.2f} MB"
            )

        w()
        w("[Top 40 Largest Linear Layers]")
        for r in rows[:40]:
            w(
                f"{r['fp16_mb']:10.2f} MB | "
                f"save {r['saved_mb_if_w4']:10.2f} MB | "
                f"{r['branch']:28s} | "
                f"{r['suffix']:12s} | "
                f"Linear({r['in_features']}->{r['out_features']}) | "
                f"{r['name']}"
            )

    print(f"[PI0.5-BENEFIT] CSV saved to: {out_csv}", flush=True)
    print(f"[PI0.5-BENEFIT] Summary saved to: {out_summary}", flush=True)

'''
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
'''