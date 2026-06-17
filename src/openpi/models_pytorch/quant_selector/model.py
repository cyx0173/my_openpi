
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

PAIRS = ["v4_a4","v4_a8","v4_a16","v8_a4","v8_a8","v8_a16","v16_a4","v16_a8","v16_a16"]
# pair索引 -> (vlm_bits, action_bits)
PAIR_BITS = [(4,4),(4,8),(4,16),(8,4),(8,8),(8,16),(16,4),(16,8),(16,16)]
A16_IDX = 8
# 真实单chunk成本: vlm_weight=3, action_weight=1, bit_cost{4:1,8:2,16:4}
PAIR_COST = np.array([4,5,7, 7,8,10, 13,14,16], dtype=np.float32)

CTX_DIM, STATE_DIM, SEQ_LEN = 2048, 32, 544

# ===== 点B 部署配置(细扫选出: critical保护0.93, 踩雷率0.089) =====
DEFAULT_PENALTY = 8.0
DEFAULT_LAMBDA  = 2.0
DEFAULT_TAU_S   = 0.5


class _Selector(nn.Module):
    """与训练 train_twosignal_v2.py 的 Selector 结构完全一致"""
    def __init__(self, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.ctx_proj = nn.Linear(CTX_DIM, d_model)
        self.state_proj = nn.Sequential(nn.Linear(STATE_DIM, d_model), nn.GELU(),
                                        nn.Linear(d_model, d_model))
        self.vlm_emb = nn.Embedding(3, d_model)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            "n1": nn.LayerNorm(d_model),
            "ff": nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)),
            "n2": nn.LayerNorm(d_model),
        }) for _ in range(n_layers)])
        self.head_succ = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 9))
        self.head_blow = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                       nn.Dropout(0.3), nn.Linear(d_model, 9))

    def forward(self, ctx, mask, state, vlm_ids):
        kv = self.ctx_proj(ctx)
        q = (self.state_proj(state) + self.vlm_emb(vlm_ids)).unsqueeze(1)
        kpm = ~mask
        for L in self.layers:
            a,_ = L["attn"](q, kv, kv, key_padding_mask=kpm)
            q = L["n1"](q + a); q = L["n2"](q + L["ff"](q))
        h = q.squeeze(1)
        return self.head_succ(h), self.head_blow(h)


class PrecisionSelector:
    def __init__(self, model, device, penalty=DEFAULT_PENALTY,
                 lam=DEFAULT_LAMBDA, tau_s=DEFAULT_TAU_S):
        self.model = model
        self.device = device
        self.penalty = float(penalty)
        self.lam = float(lam)
        self.tau_s = float(tau_s)
        self.pair_cost = torch.tensor(PAIR_COST, device=device)
        self.vlm_map = {4:0, 8:1, 16:2}

    @classmethod
    def load(cls, ckpt_path, device="cuda",
             penalty=DEFAULT_PENALTY, lam=DEFAULT_LAMBDA, tau_s=DEFAULT_TAU_S):
        ck = torch.load(ckpt_path, map_location=device)
        state_dict = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model = _Selector().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model, device, penalty=penalty, lam=lam, tau_s=tau_s)

    def _prep_ctx(self, ctx, mask):
        """对齐到训练分布: [1,544,2048] + [1,544]. 自动加batch维/pad/截断."""
        ctx = torch.as_tensor(ctx, dtype=torch.float32)
        mask = torch.as_tensor(mask)
        if ctx.dim() == 2:   # [seq,2048] -> [1,seq,2048]
            ctx = ctx.unsqueeze(0)
        if mask.dim() == 1:  # [seq] -> [1,seq]
            mask = mask.unsqueeze(0)
        mask = mask.bool()
        B, seq, dim = ctx.shape
        assert dim == CTX_DIM, f"ctx dim {dim} != {CTX_DIM}, 确认是VLM prefix hidden"
        if seq < SEQ_LEN:                     # pad到544
            pad = SEQ_LEN - seq
            ctx = torch.cat([ctx, torch.zeros(B, pad, dim, dtype=ctx.dtype)], dim=1)
            mask = torch.cat([mask, torch.zeros(B, pad, dtype=torch.bool)], dim=1)
        elif seq > SEQ_LEN:                   # 截断(训练只见过544,超长截断)
            ctx = ctx[:, :SEQ_LEN, :]
            mask = mask[:, :SEQ_LEN]
        return ctx.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def infer(self, selector_context_tokens, selector_context_mask,
              server_state, current_vlm_a_bits):
        # --- 校验 current_vlm_a_bits ---
        cvb = int(current_vlm_a_bits)
        assert cvb in self.vlm_map, f"current_vlm_a_bits={cvb} 必须是 4/8/16"
        vlm_ids = torch.tensor([self.vlm_map[cvb]], device=self.device, dtype=torch.long)

        # --- ctx/mask 对齐到 [1,544,*] ---
        ctx, mask = self._prep_ctx(selector_context_tokens, selector_context_mask)

        # --- state -> [1,32] ---
        state = torch.as_tensor(server_state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        assert state.shape[-1] == STATE_DIM, f"state dim {state.shape[-1]} != {STATE_DIM}"

        # --- forward ---
        succ_logit, blow_logit = self.model(ctx, mask, state, vlm_ids)  # [1,9]

        # --- 点B 软权衡选择 ---
        feasible = torch.sigmoid(succ_logit) >= self.tau_s          # [1,9]
        blow_p = torch.sigmoid(blow_logit)
        score = self.pair_cost.unsqueeze(0) + self.lam * blow_p * self.penalty
        score = score.clone()
        score[~feasible] = 1e9
        idx = int(score.argmin(dim=1).item())
        if not bool(feasible.any()):
            idx = A16_IDX     # 空集回退安全锚

        vlm_bits, action_bits = PAIR_BITS[idx]
        return {
            "chosen_idx": idx,
            "pair": PAIRS[idx],
            "action_a_bits": int(action_bits),     # 当前chunk action精度
            "next_vlm_a_bits": int(vlm_bits),       # 下一拍VLM精度
            "succ_prob": torch.sigmoid(succ_logit)[0].cpu().numpy(),
            "blow_prob": blow_p[0].cpu().numpy(),
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # 真实模型测试: python precision_selector.py /path/to/ckpt.pt
        sel = PrecisionSelector.load(sys.argv[1], device="cpu")
        ctx = torch.randn(60, 2048)            # 模拟变长prefix(60<544,测pad)
        mask = torch.ones(60, dtype=torch.bool)
        state = torch.randn(32)
        out = sel.infer(ctx, mask, state, current_vlm_a_bits=8)
        print("infer输出:", {k:v for k,v in out.items() if k in ["chosen_idx","pair","action_a_bits","next_vlm_a_bits"]})
        print("succ_prob:", np.round(out["succ_prob"],2))
        print("blow_prob:", np.round(out["blow_prob"],2))
    else:
        # 无ckpt时测结构/pad/拆解逻辑
        m = _Selector()
        sel = PrecisionSelector(m, "cpu")
        # 测pad: 60 -> 544
        ctx, mask = sel._prep_ctx(torch.randn(60,2048), torch.ones(60,dtype=torch.bool))
        assert ctx.shape == (1,544,2048) and mask.shape == (1,544), "pad失败"
        assert mask[0,:60].all() and not mask[0,60:].any(), "pad mask语义错"
        # 测截断: 600 -> 544
        ctx2,mask2 = sel._prep_ctx(torch.randn(600,2048), torch.ones(600,dtype=torch.bool))
        assert ctx2.shape==(1,544,2048), "截断失败"
        # 测batch维已有
        ctx3,mask3 = sel._prep_ctx(torch.randn(1,544,2048), torch.ones(1,544,dtype=torch.bool))
        assert ctx3.shape==(1,544,2048), "已有batch维处理错"
        # 测pair拆解
        for i,(v,a) in enumerate(PAIR_BITS):
            assert PAIRS[i]==f"v{v}_a{a}", f"pair拆解错 {i}"
        out = sel.infer(torch.randn(60,2048), torch.ones(60,dtype=torch.bool),
                        torch.randn(32), current_vlm_a_bits=8)
        assert out["chosen_idx"] in range(9)
        assert (out["next_vlm_a_bits"],out["action_a_bits"]) == PAIR_BITS[out["chosen_idx"]]
        print("结构/pad/截断/拆解 全部自测通过")
        print("示例输出:", {k:out[k] for k in ["pair","action_a_bits","next_vlm_a_bits"]})
        print("\n用真实ckpt测: python precision_selector.py /path/to/twosignal_v2.pt")