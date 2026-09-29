#!/usr/bin/env python3
"""重叠（overlap）收益的上界：α-β 模型下一个能藏多少、什么情况下根本藏不动。

回答三个问题
------------
1. **一次集合通信里，有多少是"可以藏"的？**
   α-β 模型 t = steps·α + bytes/BW 把时间分成两截：
   - `steps·α`（延迟项）：依赖链，**和计算量无关**，重叠能改善它的空间有限；
   - `bytes/BW`（带宽项）：有独立的计算可跑就能藏住。
   推导出一个干净的结论：两项相等发生在 `D* = n·α·BW`。
   消息小于 D* 时是延迟主导，做重叠收益很小——**这条比"通信占比"有用得多**。

2. **一次迭代里通信的理论下界是什么？**
   - 完全不重叠：`T = C + M`
   - 完美重叠（通信与计算可完全并行）：`T ≥ max(C, M)`
   - 现实：`T ≈ C + M − η·min(C, M)`，η ∈ (0,1) 是重叠效率
   给出 η 的敏感度表，用它来判断"还能榨出多少"，而不是拍一个百分比。

3. **nccl-tests 的 algbw 和 busbw 差多少、差在哪。**
   `busbw = algbw × 2(N−1)/N`（AllReduce）、`× (N−1)/N`（AllGather / ReduceScatter）。
   **但 busbw 在 NVLS / CollNet / 分层算法下不再对应单一物理链路**，
   拿它反推线速会把"算法利用率"读成"链路带宽"。

模型参数全部来自 `tools/megatron_bench/comm.py`（α-β 实现）与 `configs.py`
（模型/硬件标称值），本文件不重复定义。

用法
----
    python3 tools/profiling_bench/overlap_bound.py              # 全部三节
    python3 tools/profiling_bench/overlap_bound.py --crossover
    python3 tools/profiling_bench/overlap_bound.py --bound --tp 8 --dp 16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron_bench.comm import (  # noqa: E402
    collective_time_s, tensor_parallel_events)
from megatron_bench.configs import BenchConfig, HARDWARE, MODELS  # noqa: E402


# ----------------------------------------------------------------------------
# 第 1 节：α 项 vs β 项，交叉点 D* = n * alpha * BW
# ----------------------------------------------------------------------------


def crossover_bytes(n, alpha_us, bw_gbs):
    """α 项（依赖链延迟）与 β 项（带宽搬运）相等的消息量。

    推导（以 AllReduce ring 为例）：
        t = 2(n-1)α + 2(n-1)/n · D/BW
        两项相等  ⇒  α = D/(n·BW)  ⇒  D* = n·α·BW
    AllGather / ReduceScatter 的 steps 是 (n-1)，比值同为 n，结果一样。
    """
    return n * alpha_us * 1e-6 * bw_gbs * 1e9


def section_crossover(alpha_us=2.0):
    print("=" * 78)
    print("第 1 节  α 项与 β 项的交叉点：D* = n · α · BW")
    print("=" * 78)
    print("α = %.1f us（每次 ring 步的固定延迟，取 nccl-tests 常见量级）\n"
          % alpha_us)
    print("%-12s %-10s %8s %14s %14s" %
          ("硬件", "链路", "n", "BW (GB/s)", "D* (bytes)"))
    rows = []
    for hw_name, hw in HARDWARE.items():
        for link, bw in (("NVLink", hw.nvlink_gbs), ("IB", hw.ib_gbs)):
            for n in (8, 64):
                d = crossover_bytes(n, alpha_us, bw)
                rows.append((hw_name, link, n, bw, d))
                print("%-12s %-10s %8d %14.0f %14.0f"
                      % (hw_name, link, n, bw, d))
    print("""
读法：
  * D* 以下 = 延迟主导。此时改重叠、加 buffer 都没多少用，
    真正有效的是"合并小消息"（把 N 个小 AllReduce 合成一个）或换算法（NVLS）。
  * D* 以上 = 带宽主导。这才是重叠能发挥作用的区间。
  * D* 与 n 成正比：卡越多，依赖链越长，小消息越吃亏。
  * IB 的 D* 只有几百 KB（25 GB/s 档约 0.2 MB），
    而 NVLink 的 D* 高出两个数量级——所以"跨机通信藏不住"的第一个原因是
    它本来就落在延迟主导区，第二个原因才是带宽窄。
  * 数据并行梯度 reduce 的每层消息 = 该层参数字节数（bf16 下 7B 模型的
    attention 层约 25 MB / MLP 层约 88 MB），远在 D* 之上 → 带宽主导，可重叠。
    TP 每层每次 AR 的消息量小得多，正好落在 D* 附近 → 这就是
    `tp_comm_overlap` 收益不稳定、且强依赖 shape 的原因。""")
    return rows


# ----------------------------------------------------------------------------
# 第 2 节：重叠收益的上界
# ----------------------------------------------------------------------------


def section_bound(args):
    print("=" * 78)
    print("第 2 节  一次迭代的通信量与重叠收益上界")
    print("=" * 78)

    cfg = BenchConfig(model=args.model, hardware=args.hardware,
                      tp=args.tp, cp=args.cp, pp=args.pp, dp=args.dp,
                      micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                      seq_len=args.seq_len)
    hw = HARDWARE[args.hardware]
    m = MODELS[args.model]

    world = cfg.world_size
    seq = cfg.resolved_seq_len()
    params_per_card = m.n_params() / (cfg.tp * cfg.pp)
    print("配置：%s on %s，tp=%d cp=%d pp=%d dp=%d → 共 %d 卡"
          % (args.model, args.hardware, cfg.tp, cfg.cp, cfg.pp, cfg.dp, world))
    print("序列 %d，micro_batch %d，grad_accum %d"
          % (seq, cfg.micro_batch, cfg.grad_accum))
    print("每卡参数 %.0f M（bf16 即 %.0f MB 梯度）\n"
          % (params_per_card / 1e6, params_per_card * 2 / 1e6))

    # ---------------- TP 通信：直接复用 comm.py 的事件清单 ----------------
    L = m.n_layers
    print("-- TP 通信（复用 megatron_bench/comm.py 的 tensor_parallel_events）--")
    print("%-14s %-16s %10s %8s %14s %14s"
          % ("模式", "集合", "消息量", "每层次数", "每层(us)", "每迭代(ms)"))
    tp_tot = {}
    for sp in (False, True):
        if cfg.tp <= 1:
            tp_tot[sp] = 0.0
            continue
        evs = tensor_parallel_events(cfg, sequence_parallel=sp)
        tot = 0.0
        for e in evs:
            t1 = collective_time_s(e.kind, e.size_bytes, cfg.tp,
                                   hw.nvlink_gbs, args.alpha_us)
            tot += t1 * e.count_per_layer
            print("%-14s %-16s %10s %8d %14.2f %14.2f"
                  % ("TP+SP" if sp else "TP only", e.kind,
                     fmt_b(e.size_bytes), e.count_per_layer, t1 * 1e6,
                     t1 * e.count_per_layer * L * cfg.grad_accum * 1e3))
        tp_tot[sp] = tot * L * cfg.grad_accum
    if cfg.tp <= 1:
        print("%-14s %s" % ("TP only", "tp=1，无 TP 通信"))
    print("  每迭代 = 每层 × %d 层 × %d 个微批" % (L, cfg.grad_accum))
    if cfg.tp > 1:
        print("  TP only 总时长 %.2f ms ；TP+SP 总时长 %.2f ms ；比值 %.2f×"
              % (tp_tot[False] * 1e3, tp_tot[True] * 1e3,
                 tp_tot[False] / max(tp_tot[True], 1e-12)))
        print("""
  → 比值恒为 1.00×，这不是模型不够精细，而是结论本身：
    **序列并行（SP）不改变 TP 的通信总量。**
    恒等式 AR(d) = RS(d) + AG(d)：baseline TP 每层 4 次 AR
    （每次搬 2(n−1)/n·d），SP 每层 4 次 AG + 4 次 RS（每次搬 (n−1)/n·d），
    两边都等于 8(n−1)/n·d。SP 分的是**激活显存**——LN / dropout / residual
    从每卡存一份 (s,b,h) 变成存 (s/t,b,h)，缩小 t 倍；
    真正的提速来自"显存省下来了，可以把 activation recompute 关掉"。
    通信量口径上把 SP 当降本手段是常见的误解。
    见 Korthikanti et al. arXiv:2205.05198；通信条数的独立复述见
    arXiv:2311.02382 §3（"8 global communications per attention layer：
    4 in forward pass and 4 in backward pass"）。
    本条已用于校准 megatron_bench/comm.py 的 tensor_parallel_events()。
""")

    # ---------------- DP 梯度 reduce ----------------
    h, ffn_onepart = m.hidden, m.ffn_hidden
    attn_bytes = (h * h + 2 * h * m.kv_dim + h * h) * 2 / cfg.tp
    mlp_bytes = ((2 * h * ffn_onepart if m.gated_mlp else h * ffn_onepart)
                 + ffn_onepart * h) * 2 / cfg.tp
    per_layer_bytes = attn_bytes + mlp_bytes

    dp_kind = "reduce_scatter" if cfg.use_distributed_optimizer else "all_reduce"
    dp_link = hw.ib_gbs if world > hw.gpus_per_node else hw.nvlink_gbs
    dp_per_layer = collective_time_s(dp_kind, per_layer_bytes, cfg.dp,
                                    dp_link, args.alpha_us)
    dp_total = dp_per_layer * L
    print("-- DP 梯度 reduce（%s，n=%d，链路 %.0f GB/s）--"
          % (dp_kind, cfg.dp, dp_link))
    print("  每层参数字节 %s（已按 tp=%d 分片，pp 不计入 DP 组）"
          % (fmt_b(per_layer_bytes), cfg.tp))
    print("  每层耗时 %.3f ms × %d 层 = 每迭代 %.3f ms"
          % (dp_per_layer * 1e3, L, dp_total * 1e3))
    # 对账：朴素估算 = 本卡梯度字节 × (dp-1)/dp / 链路带宽
    naive_bytes = per_layer_bytes * L * (cfg.dp - 1) / cfg.dp
    naive_t = naive_bytes / (dp_link * 1e9)
    print("  对账（朴素估算，只算带宽项）：%s × (dp−1)/dp = %s，%.3f ms"
          % (fmt_b(per_layer_bytes * L), fmt_b(naive_bytes), naive_t * 1e3))
    print("  差额 %.3f ms 全部是 %d 层 × %d 步的 α 项；相对偏差 %.2f%%"
          % ((dp_total - naive_t) * 1e3, L, cfg.dp - 1,
             (dp_total - naive_t) / naive_t * 100.0))

    # ---------------- 计算量 ----------------
    tokens = cfg.micro_batch * seq * cfg.grad_accum * cfg.dp
    flops = 6.0 * m.n_params() * tokens
    C = flops / (world * hw.bf16_tflops * 1e12 * args.mfu)
    print("\n-- 计算量 --")
    print("  每迭代 token 数 = micro_batch %d × seq %d × grad_accum %d × dp %d = %s"
          % (cfg.micro_batch, seq, cfg.grad_accum, cfg.dp,
             "{:,}".format(int(tokens))))
    print("  6ND FLOPs = %.3e" % flops)
    print("  计算时长 C = FLOPs / (%d 卡 × %.0f TFLOP/s × MFU %.0f%%) = %.3f s"
          % (world, hw.bf16_tflops, args.mfu * 100, C))

    # ---------------- 重叠收益上界 ----------------
    M = tp_tot.get(False, 0.0) + dp_total
    print("\n-- 重叠收益上界（M = TP %.3f + DP %.3f = %.3f s，C = %.3f s）--"
          % (tp_tot.get(False, 0.0), dp_total, M, C))
    base = C + M
    print("%-38s %14s %12s" % ("口径", "迭代时间(s)", "相对不重叠"))
    print("%-38s %14.4f %11.1f%%" % ("完全不重叠  T = C + M", base, 100.0))
    for eta in (0.3, 0.5, 0.7, 0.9, 1.0):
        t = C + M - eta * min(C, M)
        print("%-38s %14.4f %11.1f%%"
              % ("重叠效率 η=%.1f  T = C+M−η·min(C,M)" % eta, t,
                 t / base * 100.0))
    floor = max(C, M)
    print("%-38s %14.4f %11.1f%%"
          % ("完美重叠  T = max(C, M)", floor, floor / base * 100.0))
    print("""读法：
  * `max(C, M)` 是任何调度器都突破不了的下界——**它是上界收益，不是目标值**。
  * η 从 0 到 1 的落差 = 「重叠这件事的全部价值」= %.1f%%。若只有几个百分点，
    所有 overlap 调参都不值得做，瓶颈在别处（通常是 C 本身或 idle）。
  * 本例 min(C, M) = M，说明通信总量小于计算总量：**重叠在原理上够用**，
    那么"通信没藏住"就一定是调度/依赖问题，而不是通信量太大。
  * 反过来若 M > C，即使 η=1 也只能降到 M：此时唯一有效的手段是减少通信量
    本身（换并行策略、增大 micro batch 摊薄、减少重计算触发的额外通信、
    改通信算法/拓扑）。注意**开序列并行不在此列**——它不改变通信总量（见上）。
  * 实测 η 的算法：从 trace 里量 `overlap_time / min(C, M)`，
    即 `trace_agg.py` 报的 `overlap / comm_total`（在 C ≥ M 时等价）。"""
          % ((base - floor) / base * 100.0))
    dump_json = {
        "config": {"model": args.model, "hardware": args.hardware,
                   "tp": cfg.tp, "pp": cfg.pp, "dp": cfg.dp,
                   "world_size": world, "mfu": args.mfu},
        "M_tp_s": tp_tot.get(False, 0.0), "M_dp_s": dp_total, "M_s": M,
        "C_s": C, "no_overlap_s": base, "perfect_overlap_floor_s": floor,
        "overlap_value_pct": (base - floor) / base * 100.0,
    }
    if args.dump:
        import json
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(dump_json, f, indent=2, ensure_ascii=False)
        print("\n结果已写出: %s" % args.dump)


def fmt_b(x):
    if x >= 1e6:
        return "%.2f MB" % (x / 1e6)
    if x >= 1e3:
        return "%.1f KB" % (x / 1e3)
    return "%.0f B" % x


# ----------------------------------------------------------------------------
# 第 3 节：algbw 与 busbw
# ----------------------------------------------------------------------------


def busbw_factor(kind, n):
    """nccl-tests 定义：busbw = algbw × factor。"""
    if n <= 1:
        return 1.0
    if kind in ("all_reduce",):
        return 2.0 * (n - 1) / n
    if kind in ("all_gather", "reduce_scatter"):
        return (n - 1) / n
    if kind in ("all_to_all", "reduce", "broadcast"):
        return (n - 1) / n
    raise ValueError(kind)


def section_busbw():
    print("\n" + "=" * 78)
    print("第 3 节  nccl-tests：algbw 与 busbw 差多少、能反推什么")
    print("=" * 78)
    print("busbw = algbw × factor，factor 只取决于集合类型与卡数：\n")
    print("%-18s %10s %10s %12s %12s" %
          ("集合", "N", "factor", "algbw", "busbw"))
    for kind in ("all_reduce", "all_gather", "reduce_scatter"):
        for n in (2, 4, 8, 16, 64):
            f = busbw_factor(kind, n)
            algbw = 10.0  # 假想读数，用于展示换算
            print("%-18s %10d %10.3f %12.2f %12.2f"
                  % (kind, n, f, algbw, algbw * f))
    print("""
读法：
  * 同一份数据，AllReduce 的 factor（2(N−1)/N）是 AllGather 的两倍。
    所以"我的 AllReduce 达到 40 GB/s"和"我的 AllGather 达到 40 GB/s"
    在物理链路上完全不是一回事。
  * N=2 时 AllReduce 的 factor 只有 1.0 —— **两卡 AllReduce 就是 SendRecv，
    没有 busbw 口径优势**。拿两卡测出来的 busbw 去推 8 卡性能会严重高估。
  * **busbw 是双向口径，链路标称带宽通常是单向口径**。所以 ring AllReduce 在
    大 N 下 busbw ≈ 2 × 单向链路是**正常的**（N=8 时 1.75×），
    看到它超过链路标称值不要急着报故障。
  * 真正会让口径失效的是 NVLS（NVLink SHARP，网内归约）、CollNet（SHARP
    交换机内归约）与分层算法：数据不在链路上被逐跳转发，甚至不在链路上归约，
    此时上面这个 factor 的推导前提（数据在 N 卡链路上都跑了一遍）不成立，
    busbw 会超出 2× 这个经验上界。**判据：busbw / 单向链路 > 2 时，
    不要再拿它反推线速，只能同类算法、同 N 之间横向比。**
  * 正确用法：同算法、同 N、同消息量下横比。busbw 曲线在某档位出现 cliff，
    就把故障定位到了"引入该档位的那一层"（小消息看 α、中消息看算法、
    大消息看链路）。""")

    # 演示：algbw 排名与 busbw 排名可以相反
    print("\n-- 演示：algbw 的排名与 busbw 的排名可以相反 --")
    link = 25.0
    print("%-16s %6s %10s %10s %12s %12s" %
          ("集合", "N", "algbw", "factor", "busbw", "busbw/单向链路"))
    cases = (("all_reduce", 8, 22.0), ("all_gather", 8, 39.0),
             ("all_reduce", 2, 24.0))
    for kind, n, algbw in cases:
        f = busbw_factor(kind, n)
        b = algbw * f
        print("%-16s %6d %10.1f %10.3f %12.1f %13.2fx"
              % (kind, n, algbw, f, b, b / link))
    print("""   看 algbw：AllGather 39.0 > AllReduce 22.0 → "AllGather 快得多"。
   看 busbw：AllReduce 38.5 > AllGather 34.1 → "AllReduce 快得多"。
   两个结论都"对"，也都没用——它们量的不是同一个东西。
   AllReduce 的 algbw 低是因为它按定义要搬 2(N−1)/N 倍的量；
   AllGather 的 factor < 1 是因为 algbw 的分母是"集合的数据量"，
   而 ring AllGather 的分母口径下每卡实际只承担 (N−1)/N 份。
   **跨集合类型比带宽数字，是这张表要防的唯一一件事。**""")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--bound", action="store_true")
    ap.add_argument("--busbw", action="store_true")
    ap.add_argument("--alpha-us", type=float, default=2.0)
    ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--hardware", default="h100-80g")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--mfu", type=float, default=0.40,
                    help="已实现的 MFU，用于把 6ND 的算术量折成实际计算时长")
    ap.add_argument("--dump", help="把上界结果写成 JSON（便于正文引用）")
    args = ap.parse_args()

    all_sections = not (args.crossover or args.bound or args.busbw)
    if all_sections or args.crossover:
        section_crossover(args.alpha_us)
    if all_sections or args.bound:
        section_bound(args)
    if all_sections or args.busbw:
        section_busbw()
    return 0


if __name__ == "__main__":
    sys.exit(main())
