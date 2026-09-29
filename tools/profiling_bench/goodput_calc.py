"""goodput 口径：统计「时间花在正事上没有」，与 MFU 是两个正交的量。

口径来源（Megatron v0.20 原生 OTel，commit 60e039626）：
    megatron/core/telemetry/span_groups.py   MegatronSpanGroup 的 span 分组
    megatron/core/telemetry/training_metrics.py  8 个 OTel 指标
    各埋点处用 is_goodput_span=True 标记「算正事」的 span
      - training/checkpointing.py:810  megatron.checkpoint.save.state_dict
      - training/checkpointing.py:896  megatron.checkpoint.save.io_write
      - training/checkpointing.py:2515 megatron.checkpoint.load.io_read
      - training/training.py:4482      megatron.startup.weight_hash_check

核心区别：
    MFU     = 模型 FLOPs / (时间 × 峰值算力)   算得快不快
    goodput = 计入 goodput 的时长 / 挂钟时长    时间用在正事上没有
两者可以背离，且背离的方向告诉你该优化哪一边。

用法：
    cd tools && python -m profiling_bench.goodput_calc
"""

from __future__ import annotations


def _fmt_span(name, ms, goodput):
    return "  %-42s %9.1f ms   %s" % (name, ms, "是" if goodput else "否")


def steady_state_iteration(iter_ms=1800.0):
    """把一个训练步拆成 span，算 goodput。

    `megatron.step` 是父 span（时长 = 整个迭代），不要把父子的时长相加。
    下面只列叶子 span，它们的和恰好等于父 span。

    span 时长是量级示意（依据 Megatron 的 span 语义与 step 时间构成），
    重点是**口径与算法**，不是绝对数。
    """
    dl, comm, opt, log = 90.0, 70.0, 70.0, 10.0
    fb = iter_ms - (dl + comm + opt + log)
    return [
        ("data_loading（等 batch）", dl, False),
        ("forward_backward（计算）", fb, True),
        ("communication（overlap 之外漏出的部分）", comm, False),
        ("optimizer（step + grad clip）", opt, True),
        ("logging / bookkeeping", log, False),
    ]


def checkpoint_cost(iter_ms=1800.0, ckpt_interval=1000, ckpt_ms=60_000.0):
    """checkpoint 的摊薄成本。存在则它本身是 goodput span，但会拉长挂钟。"""
    if ckpt_interval is None or ckpt_interval <= 0:
        return 0.0, 0.0
    per_iter = ckpt_ms / ckpt_interval
    return per_iter, ckpt_ms


def report(iter_ms=1800.0, ckpt_intervals=(None, 2000, 1000, 500)):
    print("=" * 84)
    print("goodput 与 MFU 的分工：算得快不快 vs 时间用在正事上没有")
    print("=" * 84)
    print()
    spans = steady_state_iteration(iter_ms)
    print("单个稳定迭代的 span 拆分（megatron.step 父 span = %.0f ms）：" % iter_ms)
    for name, ms, gp in spans:
        print(_fmt_span(name, ms, gp))
    print("  %-42s %9.1f ms" % ("叶子 span 合计", sum(ms for _, ms, _ in spans)))
    print()

    gp_sum = sum(ms for _, ms, gp in spans if gp)
    total = sum(ms for _, ms, _ in spans)
    print("  迭代挂钟 %.1f ms，其中计 goodput 的 %.1f ms -> 迭代内 goodput %.1f%%"
          % (total, gp_sum, 100 * gp_sum / total))
    print("  不计 goodput 的部分 = %.1f ms（%.1f%%）"
          % (total - gp_sum, 100 * (total - gp_sum) / total))
    print()

    print("-" * 84)
    print("把 checkpoint 摊进来（checkpoint 本身是 goodput span，但会拉长挂钟）")
    print("-" * 84)
    print("  ckpt 间隔     每迭代摊到   含 ckpt 的迭代时长    goodput    有效吞吐(相对)")
    ref = None
    for ci in ckpt_intervals:
        per_iter, _ = checkpoint_cost(iter_ms, ci) if ci else (0.0, 0.0)
        tot = total + per_iter
        gp = (gp_sum + per_iter) / tot if per_iter else gp_sum / total
        eff = 1.0 / tot
        if ref is None:
            ref = eff
        label = "不存" if ci is None else "%d 步" % ci
        print("  %-12s %8.1f ms      %9.1f ms        %6.2f%%      %.3f"
              % (label, per_iter, tot, 100 * gp, eff / ref))
    print()
    print("  => 存得越勤，有效吞吐掉得越多，而 goodput 几乎不动（甚至微升，因为 ckpt 算正事）。")
    print("     两个指标对同一件事的反应完全不同 —— 这就是为什么必须一起看。")
    print()

    print("-" * 84)
    print("两种背离形态：该优化哪一边")
    print("-" * 84)
    cases = [
        ("形态一：MFU 高、goodput 低",
         "计算很猛，但大量时间花在数据加载 / 通信漏出 / 日志上",
         "去查 data_loading 与 communication span，而不是去换 kernel"),
        ("形态二：MFU 低、goodput 高",
         "时间都在 forward_backward 里，但每步内部访存受限（长序列 / 小 batch / decode 形态）",
         "去查算子与并行切分，而不是去缩减非计算开销"),
    ]
    for title, desc, action in cases:
        print("  %s" % title)
        print("    现象：%s" % desc)
        print("    动作：%s" % action)
        print()

    # 构造一个形态一的例子并量化
    print("-" * 84)
    print("量化形态一：把 data_loading 与 communication 的漏出压缩掉，goodput 能回到多少")
    print("-" * 84)
    print("  漏出项                        当前      压到 1/4 后")
    dl, comm = 90.0, 70.0
    for name, v in (("data_loading", dl), ("communication(漏出)", comm)):
        print("  %-28s %7.1f ms   %7.1f ms" % (name, v, v / 4))
    new_total = total - (dl + comm) * 0.75
    new_gp = (gp_sum + (dl + comm) * 0.0)  # 这两项不计 goodput，压缩不影响 gp_sum
    print("  迭代挂钟                      %7.1f ms   %7.1f ms" % (total, new_total))
    print("  goodput                       %7.1f%%   %7.1f%%"
          % (100 * gp_sum / total, 100 * new_gp / new_total))
    print()
    print("  => 分子没动（这两项本来就不计 goodput），分母变小，goodput 自然回升。")
    print("     goodput 的变化只反映「正事占比」，不能推出 FLOPs 利用率也同步改善。")
    print()


def main():
    report()
    print("=" * 84)
    print("结论：MFU 与 goodput 必须一起看。")
    print("      只报 MFU 会漏掉「算得快但整天在等数据」，")
    print("      只报 goodput 会把「时间都在算、但算得很慢」当成健康。")
    print("=" * 84)


if __name__ == "__main__":
    main()
