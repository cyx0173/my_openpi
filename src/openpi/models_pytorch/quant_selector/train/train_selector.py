#!/usr/bin/env python3
"""
Cross-attention dynamic-precision selector (clean-subset demo).

输入: build_selector_labels.py 产出的 clean_v8a8_labels.jsonl
特征: 复用已采集的 npz (feature_path 指向),lazy-load,不重采

模型: server_state(+current_vlm_bits) 作为 query,
      selector_context_tokens(968×2048,变长,mask屏蔽) 作为 key/value,
      multi-head cross-attention -> 三个回归 head:
          d_success_hat[9], d_cost_hat[9], quality_hat[9]
      不做分类(分类 argmax 会塌)。

训练: 9-pair advantage 加权 MSE,难度权重乘进去(trivial chunk 权重低)。

推理(约束式,不 argmax cost):
      可行集 = { pair : d_success_hat ≥ -eps }
      选 = 可行集里 d_cost_hat 最小的 pair

评估(抗塌三件套):
      1) 74 个关键 chunk 上的"选对率"(是否避开 A4 升到正确精度)
      2) 扫 eps 画 Pareto: selector vs 静态A4 vs 静态baseline
      3) 整体 cost / success,对比静态基线

用法:
  python train_selector.py --labels /home/chengyuxuan/openpi/experiments/selector_dataset/clean_v8a8_labels.jsonl
  # 看泛化:加 --split_by_episode
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- 9 pair 固定顺序,与 builder 一致 ----
BITS = (4, 8, 16)
PAIRS = [f"v{v}_a{a}" for v in BITS for a in BITS]
PAIR_TO_IDX = {p: i for i, p in enumerate(PAIRS)}
A4_IDX = PAIR_TO_IDX["v4_a4"]
BASELINE_IDX = PAIR_TO_IDX["v8_a8"]
# 每个 pair 的标称 cost(用于 Pareto 评估时的静态基线参考,真实 cost 用数据里的)
PAIR_BITCOST = {p: ({4: 1, 8: 2, 16: 4}[int(p.split("_")[0][1:])] * 3
                    + {4: 1, 8: 2, 16: 4}[int(p.split("_")[1][1:])] * 1)
                for p in PAIRS}

CONTEXT_DIM = 2048   # selector_context_tokens 维度(已确认)
STATE_DIM = 32       # server_state 维度(已确认)
VLM_BITS_LIST = [4, 8, 16]
VLM_BITS_TO_ID = {4: 0, 8: 1, 16: 2}


# ============================================================
# 磁盘缓存:首次把所有 npz 预处理成 .pt 分片,之后秒读
# ============================================================
import hashlib

def _cache_key(rows, max_tokens):
    """用样本集合(feature_path 列表)+ max_tokens 算一个指纹,数据变了缓存自动失效"""
    h = hashlib.md5()
    h.update(str(max_tokens).encode())
    for r in rows:
        h.update(r["feature_path"].encode())
    return h.hexdigest()[:12]


def build_cache(rows, cache_path, max_tokens):
    """读所有 npz,存成一个 .pt 文件(变长序列用 list 存,fp16 原样保留)。"""
    print(f"[cache] 首次构建缓存 -> {cache_path} (共 {len(rows)} 样本,只跑一次)")
    items = []
    for i, r in enumerate(rows):
        with np.load(r["feature_path"], allow_pickle=False) as d:
            ctx = d["selector_context_tokens"]          # (T,2048) fp16,原样
            mask = d["selector_context_mask"].astype(np.bool_)
            state = d["server_state"].astype(np.float32)
        T = ctx.shape[0]
        if T > max_tokens:                              # 截断在缓存期做一次,训练期不再重复
            ctx = ctx[:max_tokens]; mask = mask[:max_tokens]
        items.append({
            "ctx": torch.from_numpy(np.ascontiguousarray(ctx)),     # fp16
            "mask": torch.from_numpy(mask),
            "state": torch.from_numpy(state),
            "current_vlm_bits": int(r["current_vlm_a_bits"]),
            "d_success": torch.tensor(r["d_success"], dtype=torch.float32),
            "d_cost": torch.tensor(r["d_cost"], dtype=torch.float32),
            "quality": torch.tensor(r["quality_proxy"], dtype=torch.float32),
            "success_vec": torch.tensor(r["success_vec"], dtype=torch.float32),
            "weight": float(r["weight"]),
            "is_key": bool(r["is_key_chunk"]),
            "baseline_cost": float(r["baseline_cost"]),
        })
        if (i + 1) % 200 == 0:
            print(f"  [cache] {i+1}/{len(rows)}")
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, cache_path)
    sz = os.path.getsize(cache_path) / 1e9
    print(f"[cache] 完成,缓存大小 {sz:.2f} GB")
    return items


# ============================================================
# Dataset (从内存缓存读,零磁盘 IO)
# ============================================================
class SelectorDataset(Dataset):
    """直接吃预处理好的内存 list(ctx 是 fp16,__getitem__ 时转 fp32)。"""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return {
            "ctx": it["ctx"].float(),               # fp16 -> fp32(省内存/盘,算时转)
            "mask": it["mask"],
            "state": it["state"],
            "current_vlm_bits": it["current_vlm_bits"],
            "d_success": it["d_success"],
            "d_cost": it["d_cost"],
            "quality": it["quality"],
            "success_vec": it["success_vec"],
            "weight": it["weight"],
            "is_key": it["is_key"],
            "baseline_cost": it["baseline_cost"],
        }


def load_or_build_items(rows, cache_dir, max_tokens, tag):
    """有缓存读缓存,没有就建。返回内存里的 items list。"""
    key = _cache_key(rows, max_tokens)
    cache_path = os.path.join(cache_dir, f"selcache_{tag}_{key}.pt")
    if os.path.exists(cache_path):
        print(f"[cache] 命中缓存 {cache_path},直接加载")
        return torch.load(cache_path)
    return build_cache(rows, cache_path, max_tokens)


def collate(batch):
    """pad 变长 ctx 到 batch 内最大长度"""
    B = len(batch)
    maxT = max(b["ctx"].shape[0] for b in batch)
    D = batch[0]["ctx"].shape[1]
    ctx = torch.zeros(B, maxT, D, dtype=torch.float32)
    mask = torch.zeros(B, maxT, dtype=torch.bool)
    for i, b in enumerate(batch):
        T = b["ctx"].shape[0]
        ctx[i, :T] = b["ctx"]
        mask[i, :T] = b["mask"]
    out = {
        "ctx": ctx,
        "mask": mask,
        "state": torch.stack([b["state"] for b in batch]),
        "current_vlm_bits": torch.tensor([b["current_vlm_bits"] for b in batch], dtype=torch.long),
        "d_success": torch.stack([b["d_success"] for b in batch]),
        "d_cost": torch.stack([b["d_cost"] for b in batch]),
        "quality": torch.stack([b["quality"] for b in batch]),
        "success_vec": torch.stack([b["success_vec"] for b in batch]),
        "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
        "is_key": torch.tensor([b["is_key"] for b in batch], dtype=torch.bool),
        "baseline_cost": torch.tensor([b["baseline_cost"] for b in batch], dtype=torch.float32),
    }
    return out


# ============================================================
# Cross-attention selector model
# ============================================================
class CrossAttnSelector(nn.Module):
    def __init__(self, context_dim=CONTEXT_DIM, state_dim=STATE_DIM,
                 d_model=256, n_heads=4, n_layers=2, n_pairs=9, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # context tokens -> d_model (key/value)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        # state -> d_model
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        # current_vlm_bits embedding,加到 query 上(当前精度影响判断)
        self.vlm_emb = nn.Embedding(len(VLM_BITS_LIST), d_model)

        # 几层 cross-attention:query=state,kv=context
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "ff": nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                    nn.Linear(d_model * 2, d_model)),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])

        # 三个回归头,各输出 9 个 pair 的 advantage 预测
        self.head_dsucc = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Linear(d_model, n_pairs))
        self.head_dcost = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Linear(d_model, n_pairs))
        self.head_quality = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                          nn.Linear(d_model, n_pairs))

    def forward(self, ctx, mask, state, current_vlm_bits):
        # ctx: (B,T,context_dim)  mask: (B,T) True=valid  state:(B,state_dim)
        kv = self.ctx_proj(ctx)                       # (B,T,d)
        vlm_ids = torch.tensor([VLM_BITS_TO_ID[int(b)] for b in current_vlm_bits.tolist()],
                               device=ctx.device)
        q = self.state_proj(state) + self.vlm_emb(vlm_ids)   # (B,d)
        q = q.unsqueeze(1)                            # (B,1,d) 单 query token

        key_padding_mask = ~mask                      # True = ignore(padding)
        for layer in self.layers:
            attn_out, _ = layer["attn"](q, kv, kv, key_padding_mask=key_padding_mask)
            q = layer["norm1"](q + attn_out)
            ff_out = layer["ff"](q)
            q = layer["norm2"](q + ff_out)

        h = q.squeeze(1)                              # (B,d)
        return {
            "d_success": self.head_dsucc(h),          # (B,9)
            "d_cost": self.head_dcost(h),
            "quality": self.head_quality(h),
        }


# ============================================================
# 约束式推理:可行集里选最便宜
# ============================================================
def constrained_select(d_success_hat, d_cost_hat, eps):
    """返回每个样本选中的 pair index。
    可行集 = {pair: d_success_hat >= -eps}; 选可行集里 d_cost_hat 最小。
    可行集为空时回退到 d_success_hat 最大的(最安全)。
    """
    B, P = d_success_hat.shape
    feasible = d_success_hat >= (-eps)               # (B,9) bool
    cost = d_cost_hat.clone()
    cost[~feasible] = float("inf")
    choice = cost.argmin(dim=1)                      # (B,)
    # 可行集空 -> 回退最安全
    empty = ~feasible.any(dim=1)
    if empty.any():
        safe = d_success_hat.argmax(dim=1)
        choice[empty] = safe[empty]
    return choice


# ============================================================
# 训练
# ============================================================
def train(model, train_loader, device, epochs, lr,
          dcost_std, dquality_std):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            ctx = batch["ctx"].to(device)
            mask = batch["mask"].to(device)
            state = batch["state"].to(device)
            cvb = batch["current_vlm_bits"]
            d_succ = batch["d_success"].to(device)
            d_cost = batch["d_cost"].to(device) / dcost_std
            qual = batch["quality"].to(device) / dquality_std
            w = batch["weight"].to(device)

            pred = model(ctx, mask, state, cvb)
            # 加权 MSE,每个样本乘难度权重(下限给个小底,别让 trivial 完全不学)
            ws = (w + 0.05).unsqueeze(1)              # (B,1)
            loss_succ = ((pred["d_success"] - d_succ) ** 2 * ws).mean()
            loss_cost = ((pred["d_cost"] - d_cost) ** 2 * ws).mean()
            loss_qual = ((pred["quality"] - qual) ** 2 * ws).mean()
            loss = loss_succ + loss_cost + 0.5 * loss_qual

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * ctx.shape[0]; n += ctx.shape[0]
        sched.step()
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}/{epochs}  train_loss={tot/n:.4f}")
    return model


# ============================================================
# 评估:抗塌三件套
# ============================================================
@torch.no_grad()
def evaluate(model, items, device, dcost_std, dquality_std, eps_list):
    model.eval()
    # 收集每个样本的预测和真值
    preds_succ, preds_cost = [], []
    true_succ_vec, true_dcost, true_quality = [], [], []
    is_key, baseline_cost = [], []

    ds = SelectorDataset(items)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate)
    for batch in loader:
        ctx = batch["ctx"].to(device); mask = batch["mask"].to(device)
        state = batch["state"].to(device); cvb = batch["current_vlm_bits"]
        pred = model(ctx, mask, state, cvb)
        preds_succ.append(pred["d_success"].cpu())
        preds_cost.append(pred["d_cost"].cpu() * dcost_std)  # 还原尺度
        true_succ_vec.append(batch["success_vec"])
        true_dcost.append(batch["d_cost"])
        true_quality.append(batch["quality"])
        is_key.append(batch["is_key"])
        baseline_cost.append(batch["baseline_cost"])

    P_succ = torch.cat(preds_succ)        # (N,9) 预测 d_success
    P_cost = torch.cat(preds_cost)        # (N,9) 预测 d_cost(已还原)
    T_succ = torch.cat(true_succ_vec)     # (N,9) 真实 success 0/1
    T_dcost = torch.cat(true_dcost)       # (N,9) 真实 d_cost
    key = torch.cat(is_key)               # (N,)
    base_cost = torch.cat(baseline_cost)  # (N,)
    N = P_succ.shape[0]

    # 真实每个 pair 的绝对 cost = baseline_cost + d_cost
    abs_cost = base_cost.unsqueeze(1) + T_dcost   # (N,9)

    print("\n" + "=" * 70)
    print(f"评估 (N={N} chunks, 其中关键 chunk={int(key.sum())})")
    print("=" * 70)

    # ---- 静态基线 ----
    def static_metrics(pair_idx):
        succ = T_succ[:, pair_idx].mean().item()
        cost = abs_cost[:, pair_idx].mean().item()
        return succ, cost
    a4_succ, a4_cost = static_metrics(A4_IDX)
    base_succ, base_cost_m = static_metrics(BASELINE_IDX)
    print(f"\n静态基线:")
    print(f"  永远 A4(v4_a4)   : success={a4_succ:.3f}  cost={a4_cost:.1f}")
    print(f"  永远 baseline(v8_a8): success={base_succ:.3f}  cost={base_cost_m:.1f}")

    # ---- selector 扫 eps ----
    print(f"\nselector 约束式(扫 eps):")
    print(f"  {'eps':>5} | {'success':>8} | {'cost':>7} | {'关键chunk选对率':>14} | {'选A4比例':>9}")
    pareto = []
    for eps in eps_list:
        choice = constrained_select(P_succ, P_cost, eps)   # (N,)
        chosen_succ = T_succ[torch.arange(N), choice]       # 选中pair的真实success
        chosen_cost = abs_cost[torch.arange(N), choice]
        succ = chosen_succ.mean().item()
        cost = chosen_cost.mean().item()
        # 关键 chunk 选对率:在关键chunk上,选中的pair真实success==1(即成功避开了A4陷阱)
        if key.sum() > 0:
            key_correct = chosen_succ[key].mean().item()
        else:
            key_correct = float("nan")
        a4_ratio = (choice == A4_IDX).float().mean().item()
        print(f"  {eps:5.2f} | {succ:8.3f} | {cost:7.1f} | {key_correct:14.3f} | {a4_ratio:9.3f}")
        pareto.append((eps, succ, cost, key_correct, a4_ratio))

    # ---- Pareto 支配判断 ----
    print(f"\nPareto 支配静态A4? (selector 同success下更便宜 或 同cost下更高success)")
    dominated = False
    for eps, succ, cost, kc, ar in pareto:
        if succ >= a4_succ and cost <= a4_cost and (succ > a4_succ or cost < a4_cost):
            print(f"  ✓ eps={eps:.2f}: success={succ:.3f}(≥{a4_succ:.3f}) "
                  f"cost={cost:.1f}(≤{a4_cost:.1f}) —— 支配静态A4")
            dominated = True
    if not dominated:
        print("  ✗ 没有 eps 点严格支配静态A4 —— 需检查训练/数据")
    print("=" * 70)
    return pareto


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--split_by_episode", action="store_true",
                    help="按episode切train/val看泛化;默认全量训练+全量评估")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="/home/chengyuxuan/openpi/experiments/selector_dataset/selector_demo.pt")
    ap.add_argument("--cache_dir", default="/home/chengyuxuan/openpi/experiments/selector_dataset/selcache",
                    help="磁盘缓存目录(硬盘空间换速度,首次构建后秒读)")
    ap.add_argument("--max_tokens", type=int, default=1024,
                    help="context token 截断上限(缓存期截一次)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    rows = [json.loads(l) for l in open(args.labels) if l.strip()]
    print(f"读入 {len(rows)} 个 chunk 样本")

    # train/val 划分
    if args.split_by_episode:
        cases = sorted({r["case_id"] for r in rows})
        random.shuffle(cases)
        n_val = max(1, int(len(cases) * args.val_frac))
        val_cases = set(cases[:n_val])
        train_rows = [r for r in rows if r["case_id"] not in val_cases]
        val_rows = [r for r in rows if r["case_id"] in val_cases]
        print(f"按episode切: train={len(train_rows)} val={len(val_rows)} "
              f"(val来自{n_val}个未见episode)")
    else:
        train_rows = rows
        val_rows = rows
        print(f"全量训练+全量评估: {len(rows)} chunks")

    # 标准化常数(用训练集的 d_cost / quality 尺度)
    all_dcost = np.array([r["d_cost"] for r in train_rows]).flatten()
    all_qual = np.array([r["quality_proxy"] for r in train_rows]).flatten()
    dcost_std = float(np.std(all_dcost)) or 1.0
    dquality_std = float(np.std(all_qual)) or 1.0
    print(f"标准化: d_cost_std={dcost_std:.1f}  quality_std={dquality_std:.1f}")

    # === 磁盘缓存:首次构建,之后秒读;数据全进内存,epoch内零磁盘IO ===
    train_items = load_or_build_items(train_rows, args.cache_dir, args.max_tokens,
                                      tag="train" if args.split_by_episode else "all")
    if args.split_by_episode:
        val_items = load_or_build_items(val_rows, args.cache_dir, args.max_tokens, tag="val")
    else:
        val_items = train_items  # 全量模式 train==val

    train_ds = SelectorDataset(train_items)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)  # 已在内存,无需多worker

    model = CrossAttnSelector().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M")

    print("\n开始训练...")
    model = train(model, train_loader, device, args.epochs, args.lr,
                  dcost_std, dquality_std)

    eps_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    evaluate(model, val_items, device, dcost_std, dquality_std, eps_list)

    torch.save({"model": model.state_dict(),
                "dcost_std": dcost_std, "dquality_std": dquality_std},
               args.save)
    print(f"\n模型已保存: {args.save}")


if __name__ == "__main__":
    main()