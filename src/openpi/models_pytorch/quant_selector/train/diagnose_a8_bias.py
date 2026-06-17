#!/usr/bin/env python3
"""
诊断:模型偏向 v8_a8 的根因是什么?

挑出训练集里 "A4 真的又安全又便宜" 的 chunk(真实数据里 v4_a4 success=1 且 d_cost<0),
打印模型对这些 chunk 的 9-pair 预测,重点看:
  - 模型预测的 d_cost(v4_a4) 是负的还是正的?
      负 -> 模型知道A4便宜  (那偏v8_a8是推理太保守 -> 改推理)
      正 -> 模型以为A4更贵  (没学到 -> 救数据/baseline多样性)
  - 模型预测的 d_success(v4_a4) 对不对
  - 约束式在这些chunk上到底选了谁

用法:
  python diagnose_a8_bias.py \
      --labels /home/chengyuxuan/openpi/experiments/selector_dataset/clean_v8a8_labels.jsonl \
      --ckpt   /home/chengyuxuan/openpi/experiments/selector_dataset/selector_demo.pt
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

# 复用训练脚本里的定义
import importlib.util, sys
def load_train_module(path):
    spec = importlib.util.spec_from_file_location("train_selector", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_script", default="train_selector.py",
                    help="train_selector.py 路径(复用模型/Dataset定义)")
    ap.add_argument("--n_show", type=int, default=20, help="打印多少个样例chunk")
    ap.add_argument("--cache_dir", default="/home/chengyuxuan/openpi/experiments/selector_dataset/selcache")
    ap.add_argument("--max_tokens", type=int, default=1024)
    args = ap.parse_args()

    T = load_train_module(args.train_script)
    PAIRS = T.PAIRS
    A4_IDX = T.A4_IDX
    BASE_IDX = T.BASELINE_IDX
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in open(args.labels) if l.strip()]
    print(f"读入 {len(rows)} chunk")

    # 加载模型
    ck = torch.load(args.ckpt, map_location=device)
    dcost_std = ck["dcost_std"]; dquality_std = ck["dquality_std"]
    model = T.CrossAttnSelector().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"模型已载入,dcost_std={dcost_std:.1f}")

    # 用缓存载入 items(和训练同一套)
    items = T.load_or_build_items(rows, args.cache_dir, args.max_tokens, tag="all")

    # ===== 筛选 "A4 安全且便宜" 的 chunk(真实数据) =====
    # 真实: v4_a4 success==1 且 d_cost(v4_a4) < 0(比baseline便宜)
    safe_cheap_idx = []
    for i, r in enumerate(rows):
        if r["success_vec"][A4_IDX] == 1.0 and r["d_cost"][A4_IDX] < 0:
            safe_cheap_idx.append(i)
    print(f"\n'A4安全且便宜'的chunk数: {len(safe_cheap_idx)} / {len(rows)}")
    print("(这些chunk里,A4是真实最优选择之一——模型理应敢选A4)\n")

    if not safe_cheap_idx:
        print("没有这类chunk。说明在干净子集里A4几乎从不比baseline便宜")
        print("=> 这本身就解释了偏v8_a8:数据里A4就没展示出'又安全又省'的样子")
        return

    # ===== 跑模型预测,统计 =====
    ds = T.SelectorDataset(items)
    loader = DataLoader(ds, batch_size=32, collate_fn=T.collate)
    all_dsucc, all_dcost = [], []
    with torch.no_grad():
        for batch in loader:
            ctx = batch["ctx"].to(device); mask = batch["mask"].to(device)
            state = batch["state"].to(device); cvb = batch["current_vlm_bits"]
            pred = model(ctx, mask, state, cvb)
            all_dsucc.append(pred["d_success"].cpu())
            all_dcost.append(pred["d_cost"].cpu() * dcost_std)  # 还原尺度
    P_succ = torch.cat(all_dsucc)   # (N,9)
    P_cost = torch.cat(all_dcost)   # (N,9)

    # ===== 在 "A4安全且便宜" chunk 上,模型怎么看A4 =====
    idx_t = torch.tensor(safe_cheap_idx)
    pred_a4_cost = P_cost[idx_t, A4_IDX]      # 模型预测的 d_cost(A4)
    pred_a4_succ = P_succ[idx_t, A4_IDX]      # 模型预测的 d_success(A4)
    true_a4_cost = torch.tensor([rows[i]["d_cost"][A4_IDX] for i in safe_cheap_idx])

    print("=" * 72)
    print("模型对 A4 的预测(在'A4安全且便宜'的chunk上)")
    print("=" * 72)
    print(f"  真实 d_cost(A4) 均值      : {true_a4_cost.mean():+.1f}  (负=真便宜)")
    print(f"  模型预测 d_cost(A4) 均值  : {pred_a4_cost.mean():+.1f}")
    print(f"  模型预测 d_success(A4)均值: {pred_a4_succ.mean():+.3f}  (0=认为不掉成功率)")
    pred_cheaper = (pred_a4_cost < 0).float().mean().item()
    print(f"  模型预测A4更便宜(d_cost<0)的比例: {pred_cheaper*100:.0f}%")

    print("\n  判定:")
    if pred_a4_cost.mean() < -5:
        print("  >>> 模型【知道】A4便宜(预测d_cost明显为负)。")
        print("      偏v8_a8是【推理太保守】——改推理规则,鼓励安全state选A4。")
    elif pred_a4_cost.mean() > 5:
        print("  >>> 模型【以为】A4更贵或持平(预测d_cost为正)。")
        print("      偏v8_a8是【没学到A4便宜】——救脏数据/引入baseline多样性。")
    else:
        print("  >>> 模型预测A4 d_cost接近0(没把握)。")
        print("      v8_a8的d_cost=0原点优势压过了A4——需要更强的成本信号或baseline多样性。")

    # ===== 约束式在这些chunk上实际选了谁 =====
    print("\n" + "=" * 72)
    print(f"约束式(eps=0.1)在这些chunk上的实际选择分布")
    print("=" * 72)
    choice = T.constrained_select(P_succ[idx_t], P_cost[idx_t], eps=0.1)
    from collections import Counter
    cnt = Counter(PAIRS[c] for c in choice.tolist())
    for p, c in cnt.most_common():
        print(f"  {p:8s}: {c:4d}  ({c/len(choice)*100:.0f}%)")
    a4_chosen = (choice == A4_IDX).float().mean().item()
    print(f"\n  这些'A4本是最优'的chunk里,模型实际选A4的比例: {a4_chosen*100:.0f}%")
    if a4_chosen < 0.3:
        print("  => 即使A4在这些chunk是对的,模型也很少选它,确认存在v8_a8偏向。")

    # ===== 打印几个具体样例 =====
    print("\n" + "=" * 72)
    print(f"样例(前{args.n_show}个'A4安全且便宜'chunk的逐pair预测)")
    print("=" * 72)
    for k, i in enumerate(safe_cheap_idx[:args.n_show]):
        r = rows[i]
        ch = T.constrained_select(P_succ[i:i+1], P_cost[i:i+1], eps=0.1).item()
        print(f"\n[{r['case_id']} chunk={r['chunk_idx']}] 模型选了 {PAIRS[ch]}  "
              f"(真实A4: succ={int(r['success_vec'][A4_IDX])} d_cost={r['d_cost'][A4_IDX]:+.0f})")
        print(f"  {'pair':8s} {'真d_succ':>8} {'预d_succ':>8} {'真d_cost':>8} {'预d_cost':>8}")
        for j, p in enumerate(PAIRS):
            print(f"  {p:8s} {r['d_success'][j]:+8.0f} {P_succ[i,j].item():+8.2f} "
                  f"{r['d_cost'][j]:+8.0f} {P_cost[i,j].item():+8.1f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
