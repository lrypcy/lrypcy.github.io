"""MFU / HFU 的口径实验：分子换成什么、分母换成什么，结果差多少。

复用 `tools/megatron_bench/` 的 FLOPs 记账与硬件档位，不重复实现。

硬件峰值来源（均为官方产品页，核验日期 2026-09-29）：
    A100 80GB SXM: BF16 312 TFLOPS（稠密）| 624 TFLOPS*（稀疏），HBM 2039 GB/s
       https://www.nvidia.com/en-us/data-center/a100/
    H100 SXM     : BF16 1979 TFLOPS*（= 稠密 989.5），FP8 3958*（= 稠密 1979），
                   HBM 3.35 TB/s
       https://www.nvidia.com/en-us/data-center/h100/
    * 官方标注 With sparsity。注意 A100 页把稠密值放前面、H100 页只给带 * 的稀疏值，
      两个页面的排版口径并不一致 —— 这本身就是分母陷阱的来源。

用法：
    cd tools && python -m profiling_bench.mfu_accounting
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron_bench import configs, flops  # noqa: E402

# 官方峰值表（TFLOPS），稠密与稀疏分列
PEAKS = {
    "A100 80GB SXM · BF16 稠密": 312.0,
    "A100 80GB SXM · BF16 稀疏 2:4": 624.0,
    "H100 SXM · BF16 稠密": 989.5,
    "H100 SXM · BF16 稀疏 2:4": 1979.0,
    "H100 SXM · FP8 稠密": 1979.0,
    "H100 SXM · TF32 稠密": 494.5,
}

# 代际对照（同 SKU、同精度、都取稠密）
GEN = {
    "A100 80GB SXM": {"bf16_dense": 312.0, "hbm_gbs": 2039.0},
    "H100 SXM": {"bf16_dense": 989.5, "hbm_gbs": 3350.0},
}


def _cfg(model="llama3-8b", **kw):
    base = dict(model=model, hardware="h100-80g", tp=1, cp=1, pp=1, ep=1, dp=8,
                num_nodes=1, micro_batch=4, grad_accum=8, seq_len=None,
                recompute="none", use_distributed_optimizer=True)
    base.update(kw)
    return configs.BenchConfig(**base)


def exp_denominator(model="llama3-8b", seq_len=8192, tf_per_gpu=400.0):
    """实验一：分子固定，只换分母 -> MFU 差多少。

    入口取「官方 log 那一行的 TFLOP/s/GPU」，避免自己编吞吐。
    400 TFLOP/s/GPU 对 8B 模型在 H100 上是常见量级（≈40% MFU）。
    """
    cfg = _cfg(model=model, seq_len=seq_len)
    per_token = flops.training_flops_per_token(cfg)["total_none"]
    tps = tf_per_gpu * 1e12 / per_token

    print("=" * 78)
    print("实验一  分母口径：同一行 log，换个分母就换一个 MFU")
    print("=" * 78)
    print("模型 %s，序列 %d，每 token %.2f GFLOP（不含重计算）" % (model, seq_len, per_token / 1e9))
    print("log 读数 = %.0f TFLOP/s/GPU  ->  折合 %.0f tokens/s/GPU"
          % (tf_per_gpu, tps))
    print("分子固定不变，只换分母：")
    print()
    print("  分母（TFLOPS）                         MFU     相对 H100 稠密 BF16")
    ref = None
    for name, peak in PEAKS.items():
        m = tf_per_gpu / peak
        if "H100 SXM · BF16 稠密" in name:
            ref = m
        print("  %-36s %7.2f%%" % (name, 100 * m))
    print()
    lo, hi = min(PEAKS.values()), max(PEAKS.values())
    print("  最大 / 最小分母之比 = %.2fx -> 同一行 log 可以报成 %.1f%% 或 %.1f%%"
          % (hi / lo, 100 * tf_per_gpu / hi, 100 * tf_per_gpu / lo))
    print()
    print("  注意 A100 那两行给出了 >100%% 的「MFU」。这个数本身就是一个诊断：")
    print("  400 TFLOP/s/GPU 超过了 A100 稠密 BF16 的峰值 312，所以这行日志不可能来自 A100。")
    print("  跨硬件套错分母，得到的是一个物理上不存在的利用率。")
    print()
    print("=> MFU 没有绝对值，只有「分子口径 + 分母口径 + 硬件」三元组。")
    print("   报 MFU 不写分母用的是稠密还是稀疏、哪种精度，等于没报。")
    print()


def exp_numerator(model="llama3-8b", seq_len=8192, tf_per_gpu=400.0):
    """实验二：分子换 recompute 策略 -> MFU 与 HFU 分道扬镳。"""
    cfg = _cfg(model=model, seq_len=seq_len)
    t = flops.training_flops_per_token(cfg)
    peak = PEAKS["H100 SXM · BF16 稠密"]

    print("=" * 78)
    print("实验二  分子口径：MFU 不变而 HFU 变，差额就是重计算的代价")
    print("=" * 78)
    print("分母统一用 H100 稠密 BF16 = %.1f TFLOPS；log 读数 %.0f TFLOP/s/GPU"
          % (peak, tf_per_gpu))
    print()
    rows = [("不重计算", t["total_none"]),
            ("full recompute", t["total_full_recompute"]),
            ("selective (core attn)", t["total_selective_core_attn"])]
    base = t["total_none"]
    print("  策略                 每 token FLOPs     MFU      HFU     HFU/MFU")
    for name, fl in rows:
        mfu = tf_per_gpu / peak                              # MFU 不随 recompute 变
        hfu = tf_per_gpu * (fl / base) / peak                # HFU 按实际发射量
        print("  %-20s %8.2f GFLOP  %6.2f%%  %6.2f%%    %.3f"
              % (name, fl / 1e9, 100 * mfu, 100 * hfu, fl / base))
    print()
    print("  full recompute 时 解析比值 = %.4f（= 4/3，每 token 从 6P 变 8P）"
          % (t["total_full_recompute"] / t["total_none"]))
    print("  selective 时     解析比值 = %.4f"
          % (t["total_selective_core_attn"] / t["total_none"]))
    print()
    print("=> Megatron 官方 log 的 `throughput per GPU (TFLOP/s/GPU)` 来自")
    print("   `num_floating_point_operations()`，收尾在 L1007 的 `return flops_fwd * 3`，")
    print("   **不含重计算系数** —— 所以它报的是 MFU 口径，开不开 recompute 这个数都不变。")
    print("   想拿 HFU 得自己乘上去；拿 MFU 去评价「重计算划不划算」是问错了指标。")
    print()


def exp_seqlen_sweep(model="llama3-8b", tok_per_s=14000.0):
    """实验三：6ND 近似随序列变长失真。"""
    cfg0 = _cfg(model=model)
    print("=" * 78)
    print("实验三  分子口径的来源：6ND 近似随序列变长而失真")
    print("=" * 78)
    print("  序列长度   精确(含 lm_head)      6ND        偏差")
    for s in (1024, 2048, 4096, 8192, 32768, 131072):
        cfg = _cfg(model=model, seq_len=s)
        exact = flops.training_flops_per_token(cfg, causal=True)["total_none"]
        approx = flops.approx_6nd(cfg)
        print("  %8d   %8.2f GFLOP   %8.2f GFLOP   %+7.2f%%"
              % (s, exact / 1e9, approx / 1e9, 100 * (exact - approx) / approx))
    print()
    print("=> 6ND 把 attention 的 L^2 项当常数忽略。序列越长，这一项越不能忽略，")
    print("   于是「同一个模型」在长序列下的 6ND-MFU 与精确-MFU 会系统性背离。")
    print()


def exp_generation():
    """实验四：代际对照 —— 算力涨得比带宽快，MFU 必然下降。"""
    print("=" * 78)
    print("实验四  代际算术强度门槛：算力涨 3.2x，带宽只涨 1.6x")
    print("=" * 78)
    a, h = GEN["A100 80GB SXM"], GEN["H100 SXM"]
    r_flops = h["bf16_dense"] / a["bf16_dense"]
    r_bw = h["hbm_gbs"] / a["hbm_gbs"]
    print("  稠密 BF16 峰值:  %.1f -> %.1f TFLOPS   = %.2fx"
          % (a["bf16_dense"], h["bf16_dense"], r_flops))
    print("  HBM 带宽:        %.0f -> %.0f GB/s     = %.2fx"
          % (a["hbm_gbs"], h["hbm_gbs"], r_bw))
    print()
    bal_a = a["bf16_dense"] * 1e12 / (a["hbm_gbs"] * 1e9)
    bal_h = h["bf16_dense"] * 1e12 / (h["hbm_gbs"] * 1e9)
    print("  机器平衡点（峰值算力 / 峰值带宽，单位 FLOP/Byte）:")
    print("    A100 80GB SXM: %.1f FLOP/Byte" % bal_a)
    print("    H100 SXM     : %.1f FLOP/Byte" % bal_h)
    print("    门槛上升     : %.2fx" % (bal_h / bal_a))
    print()
    print("=> 门槛上升意味着：一段算术强度固定为 X 的子层，在 A100 上要 X > %.1f 才是计算受限，"
          % bal_a)
    print("   在 H100 上要 X > %.1f。原本在 A100 上勉强算「计算受限」的层，" % bal_h)
    print("   换到 H100 就滑回了访存受限 —— 峰值算力上的分子涨了，能用到的那部分没涨那么多。")
    print("   所以「换新卡后吞吐上去了、但 log 里的 TFLOP/s/GPU 没同比例涨」是正常的。")
    print()


def exp_layer_mix(model="llama3-8b", seq_len=8192):
    """实验五：一次完整 forward 的 FLOPs 构成。

    注意 `forward_flops_per_token()` 返回的是**单层**的量，但 lm_head 是每次 forward
    只算一次。直接拿它和单层的项相加会得到一个没有物理含义的比例，这里按 L 折算。
    """
    cfg = _cfg(model=model, seq_len=seq_len)
    d = flops.forward_flops_per_token(cfg)
    m = configs.MODELS[model]
    L = m.n_layers

    per_layer = d.qkv_proj + d.attn_core + d.attn_out + d.mlp
    hidden_all = per_layer * L
    total = hidden_all + d.lm_head

    print("=" * 78)
    print("实验五  一次完整 forward 的 FLOPs 构成：不是所有 FLOPs 都长成 GEMM 的样子")
    print("=" * 78)
    print("模型 %s（%d 层），序列 %d，每 token 口径" % (model, L, seq_len))
    print()
    print("  组成                                    每 token FLOPs   占 forward")
    rows = [("QKV 投影 × %d 层" % L, d.qkv_proj * L),
            ("attention core (QK^T + AV) × %d 层" % L, d.attn_core * L),
            ("输出投影 W_o × %d 层" % L, d.attn_out * L),
            ("MLP (SwiGLU 三个投影) × %d 层" % L, d.mlp * L),
            ("lm_head (vocab 投影，只算一次)", d.lm_head)]
    for name, v in rows:
        print("  %-40s %10.2f MFLOP     %5.1f%%" % (name, v / 1e6, 100 * v / total))
    print("  %-40s %10.2f MFLOP     %5.1f%%" % ("合计", total / 1e6, 100.0))
    print()
    print("  attention core 占 %.1f%%，其余是各层投影；lm_head 单独占 %.1f%%。"
          % (100 * d.attn_core * L / total, 100 * d.lm_head / total))
    print("  attention core 里有一半是 softmax（elementwise）而不是 matmul —— MFU 的分子")
    print("  只算 matmul，这部分 FLOPs 一分都不计入，但真机上要花时间。")
    print("  序列越长，这个「计入分子但不完整」的区域越大。")
    print()


def exp_official_log(model="llama3-8b", seq_len=8192):
    """实验五：把官方 log 的数换回 MFU。"""
    cfg = _cfg(model=model, seq_len=seq_len)
    per_token = flops.training_flops_per_token(cfg)["total_none"]
    print("=" * 78)
    print("实验六  把 log 里的 TFLOP/s/GPU 换回 MFU")
    print("=" * 78)
    print("模型 %s / 序列 %d / 每 token %.2f GFLOP（不含重计算）"
          % (model, seq_len, per_token / 1e9))
    print()
    print("  log 里的数    tokens/s/GPU   MFU(稠密 BF16)   MFU(稀疏 BF16)")
    for tf in (100.0, 200.0, 300.0, 400.0):
        tps = tf * 1e12 / per_token
        print("  %6.0f TFLOP/s   %10.0f     %10.2f%%       %10.2f%%"
              % (tf, tps,
                 100 * tf / PEAKS["H100 SXM · BF16 稠密"],
                 100 * tf / PEAKS["H100 SXM · BF16 稀疏 2:4"]))
    print()
    print("=> 同一行 log，稠密分母下 40%、稀疏分母下 20%。")
    print("   报数的人如果不写分母口径，你无法判断他说的是哪个。")
    print()


def main():
    exp_denominator()
    exp_numerator()
    exp_seqlen_sweep()
    exp_generation()
    exp_layer_mix()
    exp_official_log()
    print("=" * 78)
    print("结论：MFU/HFU 之争不是定义之争，是口径之争。")
    print("      写清楚「分子含不含重计算、分母用哪种精度哪个稀疏假设」才算报了个数。")
    print("=" * 78)


if __name__ == "__main__":
    main()
