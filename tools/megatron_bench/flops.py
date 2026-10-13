"""Transformer FLOPs 记账（纯 CPU 解析，不需要 GPU）。

两种口径并存，并要求在文章里同时给出：
1. exact —— 逐算子精确累加（区分 GQA、区分 attention core 里的 s 项、可选 causal 折半）
2. 6ND  —— 业界常用的近似，N 为参与矩阵乘的参数量

两者必然有偏差，偏差本身就是值得写进文章的结论。
"""

from dataclasses import dataclass
from typing import Dict

from .configs import BenchConfig, MODELS, HARDWARE


@dataclass
class FlopsBreakdown:
    """每 token 的 forward FLOPs 明细。单位都是 FLOP/token。"""

    qkv_proj: float
    attn_core: float
    attn_out: float
    mlp: float
    lm_head: float

    def forward_total(self, include_lm_head: bool = True) -> float:
        t = self.qkv_proj + self.attn_core + self.attn_out + self.mlp
        if include_lm_head:
            t += self.lm_head
        return t

    def as_rows(self) -> list:
        return [
            ("QKV 投影", self.qkv_proj),
            ("attention core (QK^T + AV)", self.attn_core),
            ("输出投影 W_o", self.attn_out),
            ("MLP (SwiGLU 三个投影)", self.mlp),
            ("lm_head (vocab 投影)", self.lm_head),
        ]


def forward_flops_per_token(cfg: BenchConfig, causal: bool = False) -> FlopsBreakdown:
    """每 token 的 forward FLOPs。不计 elementwise（LN / RoPE / dropout / residual）。

    GEMM 类算子的 FLOPs = 2 * tokens * in * out。
    """
    m = MODELS[cfg.model]
    h, I = m.hidden, m.ffn_hidden
    kv_dim = m.kv_dim
    s = cfg.resolved_seq_len()

    qkv = 2 * h * h + 2 * 2 * h * kv_dim          # W_q: h->h, W_k/W_v: h->kv_dim
    attn_core = 4 * s * h                          # QK^T + AV，每个 token 要打 s 个位置
    if causal:
        attn_core /= 2                             # 下三角，实际乘一半
    attn_out = 2 * h * h
    # SwiGLU 有 gate/up/down 三个投影 -> 6hI；GELU 只有 up/down 两个 -> 4hI
    mlp = h * I * (6 if m.gated_mlp else 4)
    lm_head = 2 * h * m.vocab

    return FlopsBreakdown(qkv, attn_core, attn_out, mlp, lm_head)


def training_flops_per_token(cfg: BenchConfig, causal: bool = False) -> Dict[str, float]:
    """每 token 的训练总 FLOPs，按 recompute 策略分别给出。

    注意 forward_flops_per_token 返回的是**单层**的量，这里才乘上层数。
    backward = 2 x forward（权梯度 + 输入梯度两个 GEMM）。
    """
    m = MODELS[cfg.model]
    L = m.n_layers
    fwd = forward_flops_per_token(cfg, causal)

    hidden_forward = fwd.forward_total(include_lm_head=False) * L   # 所有 transformer 层
    full_forward = hidden_forward + fwd.lm_head

    out = {}
    out["forward"] = full_forward
    out["backward"] = 2 * full_forward
    out["total_none"] = 3 * full_forward
    # full recompute：反向阶段把所有 transformer 层再算一遍 forward
    out["total_full_recompute"] = 3 * full_forward + hidden_forward
    # selective(core_attn)：只重算 attention core
    out["total_selective_core_attn"] = 3 * full_forward + fwd.attn_core * L
    out["_hidden_forward_per_layer"] = fwd.forward_total(include_lm_head=False)
    out["_attn_core"] = fwd.attn_core
    return out


def megatron_official_flops_per_token(cfg: BenchConfig) -> Dict[str, float]:
    """复刻 Megatron 官方的 FLOPs 记账，用来和我们自己的实现交叉验证。

    公式来源：`megatron/training/training.py::num_floating_point_operations()` (L802)，
    其中 dense 分支在 L1007 收尾为 `return flops_fwd * 3`。

    官方公式的三个关键约定（都对得上它的源码注释）：
      1. 每个 GEMM 计 2mnk（FMA），且 forward / wgrad / dgrad 各算一次 -> 最后乘 3
      2. attention core 用 `sum_i(L_i^2)` 计，且 causal mask 的 /2 与 FMA 的 *2 抵消
      3. **不乘重计算系数** —— 开 recompute 时它报的数不变（这是本篇的重要发现）
      4. lm_head 计入：`2 * T * h * vocab * (1 + mtp_num_layers)`
    """
    m = MODELS[cfg.model]
    s = cfg.resolved_seq_len()
    # 按「一条长度为 s 的序列」喂进官方公式，再除以 token 数得到每 token
    T = float(s)                 # total_real_tokens_in_batch
    s2 = float(s) * s            # seqlen_squared_sum_in_batch

    h = float(m.hidden)
    n_heads = float(m.n_heads)
    p = 1.0                      # kv_channels 未显式指定时 p=1
    g = float(m.n_kv_heads)      # GQA: g = kv 组数；MHA 时等于 n_heads

    L = float(m.n_layers)          # 官方公式里是 num_attn_layers / num_mlp_layers 逐层累加

    # QKV + output 投影（forward，逐层）
    attn_proj = L * 4.0 * T * h * p * (h + h * (g / n_heads))
    # core attention（forward，已含 causal 折半，逐层）
    attn_core = L * 2.0 * s2 * h * p

    expansion = float(m.ffn_hidden) / h
    scale = 1.5 if m.gated_mlp else 1.0
    mlp = L * 4.0 * expansion * scale * T * h * h

    logits = 2.0 * T * h * float(m.vocab)

    per_token_fwd = (attn_proj + attn_core + mlp + logits) / T
    return {
        "attention_projections": attn_proj / T,
        "attention_core": attn_core / T,
        "mlp": mlp / T,
        "lm_head": logits / T,
        "forward": per_token_fwd,
        "training": per_token_fwd * 3,
    }


def cross_validate(cfg: BenchConfig) -> Dict[str, float]:
    """三种口径并列：我们的精确、Megatron 官方、6ND。"""
    ours_causal = training_flops_per_token(cfg, causal=True)["total_none"]
    ours_no_causal = training_flops_per_token(cfg, causal=False)["total_none"]
    official = megatron_official_flops_per_token(cfg)["training"]
    approx = approx_6nd(cfg)
    return {
        "ours_causal": ours_causal,
        "ours_no_causal": ours_no_causal,
        "megatron_official": official,
        "6nd": approx,
        # 同口径对比：我们也按 causal 折半
        "ours_vs_official_pct": 100.0 * (ours_causal - official) / official,
        "official_vs_6nd_pct": 100.0 * (official - approx) / approx,
    }


def approx_6nd(cfg: BenchConfig) -> float:
    """6ND 近似：N 取参与 GEMM 的参数量（不含 embedding 与 lm_head）。"""
    m = MODELS[cfg.model]
    return 6 * m.n_params_no_embedding()


def compare_exact_vs_6nd(cfg: BenchConfig, causal: bool = False) -> Dict[str, float]:
    """把精确口径和 6ND 并列出来，算出相对偏差。"""
    exact = training_flops_per_token(cfg, causal)["total_none"]
    approx = approx_6nd(cfg)
    return {
        "exact_loss_head_included": exact,
        "6nd": approx,
        "rel_diff_pct": 100.0 * (exact - approx) / approx,
        "params_no_embedding": MODELS[cfg.model].n_params_no_embedding(),
    }


def mfu(throughput_tokens_per_s: float, cfg: BenchConfig, recompute: str = "none") -> float:
    """给定实测吞吐，反算 MFU（Model FLOPs Utilization）。

    这是无 GPU 时唯一诚实的用法：把别人公布的吞吐换算成统一口径的 MFU。
    """
    key = {
        "none": "total_none",
        "full": "total_full_recompute",
        "selective": "total_selective_core_attn",
    }[recompute]
    per_token = training_flops_per_token(cfg)[key]
    hw = HARDWARE[cfg.hardware]
    achieved = throughput_tokens_per_s * per_token  # FLOP/s
    return achieved / (hw.bf16_tflops * 1e12)


def theoretical_step_time_s(cfg: BenchConfig, recompute: str = "none", mfu_assumed: float = 0.4) -> float:
    """在假设 MFU 的前提下，估算一个 step 的纯计算耗时（秒）。"""
    key = {
        "none": "total_none",
        "full": "total_full_recompute",
        "selective": "total_selective_core_attn",
    }[recompute]
    per_token = training_flops_per_token(cfg)[key]
    hw = HARDWARE[cfg.hardware]
    m = MODELS[cfg.model]
    s = cfg.resolved_seq_len()
    tokens_per_rank = cfg.micro_batch * s
    flops_per_rank = per_token * tokens_per_rank * cfg.grad_accum
    return flops_per_rank / (hw.bf16_tflops * 1e12 * mfu_assumed)
