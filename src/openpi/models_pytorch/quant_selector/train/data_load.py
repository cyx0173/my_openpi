#!/usr/bin/env python3
"""
Label builder for the dynamic-precision selector (clean-subset + 9-pair advantage).

它做什么:
  - 扫所有有 final_path.mp4 的有效 episode
  - 只保留 tail_baseline == "v8_a8" 的 chunk(干净子集,baseline 可信)
  - 对每个 chunk,以该 chunk 的 v8_a8 candidate 行为 baseline,
    给 9 个 precision pair 各算 advantage:
        d_success    = success(pair) - success(baseline)
        d_cost       = cost.total(pair) - cost.total(baseline)
        quality_proxy= final_chunk(pair) - final_chunk(baseline)   (timeout 罚 +TIMEOUT_PENALTY)
  - 用 feature_index.jsonl 找到该 chunk "真实执行那档" 的特征 npz 路径(选法 A)
  - 算一个难度权重(trivial chunk 权重低)
  - 输出训练表 jsonl + 一个汇总统计

它不做什么:
  - 不读 recovery_schedule(那条偏 A4 的贪心路径被弃用)
  - 不碰/不重采 npz 特征(只记录路径,训练时再 lazy-load)

用法:
  python build_selector_labels.py \
      --recovery_root /home/chengyuxuan/openpi/experiments/smooth/mode7/recovery_schedules \
      --feature_index_root /home/chengyuxuan/openpi/experiments/selector_dataset/cases \
      --out /home/chengyuxuan/openpi/experiments/selector_dataset/clean_v8a8_labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

# 9 个 precision pair 的固定顺序(axis: v=VLM bits, a=action bits)
BITS = (4, 8, 16)
PAIRS = [f"v{v}_a{a}" for v in BITS for a in BITS]   # 9 个
PAIR_TO_IDX = {p: i for i, p in enumerate(PAIRS)}

BASELINE_PAIR = "v8_a8"          # 干净子集的 baseline
CLEAN_TAIL = "v8_a8"             # 只保留这种尾巴的 chunk
TIMEOUT_PENALTY = 50             # final_chunk=null(timeout)时的额外惩罚(单位:chunk 数)


def load_candidate_outcomes(path: Path) -> dict:
    """读一个 episode 的 candidate_outcomes.jsonl.
    返回 {chunk_idx: {pair: row}}  以及 {chunk_idx: tail_baseline}
    """
    by_chunk = defaultdict(dict)
    tail_of = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            c = int(r["chunk_idx"])
            pair = r["candidate_pair"]
            by_chunk[c][pair] = r
            tail_of[c] = r.get("tail_baseline")
    return by_chunk, tail_of


def load_feature_index(path: Path) -> dict:
    """读 feature_index.jsonl.
    返回 {chunk_idx: feature_path}  只取真实执行那档(is_best_path_vlm_branch==True)
    """
    feat_of = {}
    if not path.exists():
        return feat_of
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not bool(r.get("is_best_path_vlm_branch", False)):
                continue
            c = int(r["chunk_idx"])
            feat_of[c] = r.get("feature_path")
    return feat_of


def safe_final_chunk(row, baseline_final):
    """timeout 行 final_chunk=null,用 baseline+penalty 代表'最差';否则原值。"""
    fc = row.get("final_chunk")
    if fc is None:
        return baseline_final + TIMEOUT_PENALTY
    return float(fc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery_root", required=True,
                    help="recovery_schedules 根目录(含每个 episode 文件夹)")
    ap.add_argument("--feature_index_root", required=True,
                    help="selector_dataset/cases 根目录(含每个 case 的 feature_index.jsonl)")
    ap.add_argument("--out", required=True, help="输出训练表 jsonl 路径")
    args = ap.parse_args()

    recovery_root = Path(args.recovery_root)
    feat_root = Path(args.feature_index_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(
        d for d in glob.glob(os.path.join(recovery_root, "*"))
        if os.path.isdir(d)
        and os.path.exists(os.path.join(d, "candidate_outcomes.jsonl"))
        and os.path.exists(os.path.join(d, "final_path.mp4"))  # 只要有效 episode
    )

    rows_out = []
    n_chunks_seen = 0
    n_chunks_clean = 0
    n_missing_feature = 0
    n_missing_baseline = 0
    n_incomplete_pairs = 0
    n_key_chunks = 0           # A4 失败但 baseline 成功的关键 chunk
    difficulties = []

    for d in episode_dirs:
        case_id = os.path.basename(d)
        co_path = Path(d) / "candidate_outcomes.jsonl"
        fi_path = feat_root / case_id / "feature_index.jsonl"

        by_chunk, tail_of = load_candidate_outcomes(co_path)
        feat_of = load_feature_index(fi_path)

        for c, pairs in by_chunk.items():
            n_chunks_seen += 1
            # 只留干净尾巴
            if tail_of.get(c) != CLEAN_TAIL:
                continue
            n_chunks_clean += 1

            # baseline 行
            if BASELINE_PAIR not in pairs:
                n_missing_baseline += 1
                continue
            base = pairs[BASELINE_PAIR]
            base_succ = 1.0 if base.get("success") else 0.0
            base_cost = float(base["cost"]["total"])
            base_final = base.get("final_chunk")
            base_final = float(base_final) if base_final is not None else 999.0

            # 9 个 pair 都得在
            if any(p not in pairs for p in PAIRS):
                n_incomplete_pairs += 1
                continue

            # 特征路径(真实执行那档)
            feat_path = feat_of.get(c)
            if feat_path is None:
                n_missing_feature += 1
                continue

            d_success = [0.0] * 9
            d_cost = [0.0] * 9
            quality_proxy = [0.0] * 9
            succ_vec = [0.0] * 9
            for p in PAIRS:
                row = pairs[p]
                i = PAIR_TO_IDX[p]
                s = 1.0 if row.get("success") else 0.0
                succ_vec[i] = s
                d_success[i] = s - base_succ
                d_cost[i] = float(row["cost"]["total"]) - base_cost
                quality_proxy[i] = safe_final_chunk(row, base_final) - base_final

            # 关键 chunk:A4 失败但 baseline 成功
            a4_idx = PAIR_TO_IDX["v4_a4"]
            if succ_vec[a4_idx] == 0.0 and base_succ == 1.0:
                n_key_chunks += 1
                is_key = True
            else:
                is_key = False

            # 难度权重:9 个 pair 的 d_success 方差 + d_cost 标准差(各自标准化后相加)
            var_succ = float(np.var(d_success))
            std_cost = float(np.std(d_cost))
            difficulty_raw = var_succ + std_cost  # 先存原始,稍后全局标准化
            difficulties.append(difficulty_raw)

            rows_out.append({
                "case_id": case_id,
                "chunk_idx": c,
                "task_id": int(base.get("task_id", -1)),
                "feature_path": feat_path,
                "current_vlm_a_bits": int(base.get("candidate_vlm_a_bits", 8)),  # baseline=v8
                "pairs": PAIRS,
                "d_success": d_success,
                "d_cost": d_cost,
                "quality_proxy": quality_proxy,
                "success_vec": succ_vec,
                "baseline_pair": BASELINE_PAIR,
                "baseline_cost": base_cost,
                "baseline_final_chunk": base_final,
                "is_key_chunk": is_key,
                "difficulty_raw": difficulty_raw,
            })

    # 全局标准化难度权重到 [~0, ~1],trivial chunk 趋 0
    if difficulties:
        dmin = float(np.min(difficulties))
        dmax = float(np.max(difficulties))
        rng = (dmax - dmin) if dmax > dmin else 1.0
        for r in rows_out:
            r["weight"] = float((r["difficulty_raw"] - dmin) / rng)
    # 写出
    with open(out_path, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    print("=" * 70)
    print("Label builder 完成")
    print("=" * 70)
    print(f"有效 episode 数              : {len(episode_dirs)}")
    print(f"扫到的 chunk 总数            : {n_chunks_seen}")
    print(f"其中干净尾巴(v8_a8)chunk    : {n_chunks_clean}")
    print(f"  - 缺 baseline 行跳过        : {n_missing_baseline}")
    print(f"  - 9 pair 不全跳过           : {n_incomplete_pairs}")
    print(f"  - 缺特征文件跳过            : {n_missing_feature}")
    print(f"最终写出训练样本(chunk)数   : {len(rows_out)}")
    print(f"  其中关键 chunk(A4失败/base成功): {n_key_chunks}")
    if rows_out:
        weights = [r['weight'] for r in rows_out]
        print(f"难度权重分布: min={min(weights):.3f} "
              f"median={sorted(weights)[len(weights)//2]:.3f} max={max(weights):.3f}")
        # 抽查一个关键 chunk
        for r in rows_out:
            if r["is_key_chunk"]:
                print("\n抽查一个关键 chunk:")
                print(f"  {r['case_id']} chunk={r['chunk_idx']}")
                for i, p in enumerate(PAIRS):
                    print(f"    {p:8s} d_succ={r['d_success'][i]:+.0f} "
                          f"d_cost={r['d_cost'][i]:+.0f} "
                          f"quality={r['quality_proxy'][i]:+.0f}")
                break
    print("\n输出文件:", out_path)
    print("=" * 70)


if __name__ == "__main__":
    main()