#!/usr/bin/env python3
"""指标口径对齐：把队列模型的输出按 vLLM `--save-result` 的字段名落盘，逐字段对照。

为什么需要这一步
----------------
「TTFT」「吞吐」「goodput」这些词在工具之间**同名不同义**。不逐字段钉一遍，
就会拿 A 工具的字段去比 B 工具的字段，然后得出一个自己都不知道在比什么的结论。
本脚本做三件事：

1. **字段映射表**：每个能对齐的字段标明来源与可信度（已核验 / 版本相关 / 推断）。
2. **能力边界表**：明确列出**队列模型给不出的量**。
   报得出来的和报不出来的都要写清楚，否则读者会以为模型能替代压测。
3. **ITL 与 TPOT 的分母演示**：这两个量在流式输出下不是一回事，
   差多少有一个闭式表达，不依赖任何仿真。

ITL 与 TPOT 的分母（本文最容易读错的一处）
-----------------------------------------
vLLM 官方文档给的定义：

- **ITL** 记录**相邻两次流式输出**之间的间隔。一个流式输出里如果含多个
  token（投机解码下一个 engine step 接受多个 draft token 时就会这样），
  这些 token **不额外产生 ITL 样本**。
- **TPOT** 是**每条请求算一次**、排除首 token 后摊到每个输出 token 上，
  再跨请求聚合。

于是两者相差一个分母：

    mean_ITL / TPOT = (n_out - 1) / n_streamed_gaps

每个流式输出恰好一个 token 时 `n_streamed_gaps = n_out - 1`，比值 1，
两者数值相近——这正是「ITL 和 TPOT 差不多」这句经验话成立的条件。
一旦每个输出带多个 token，比值就等于「平均每个输出带几个 token」，
ITL 被同比例放大。

**后果**：开启投机解码后 mean ITL 会自己往上走，而每条请求的生成速度其实没变。
只看 ITL 会把「测量口径的放大」读成「服务变慢了」。

字段名的可信度分级（不是所有名字都同样硬）
------------------------------------------
- **已核验**：vLLM 文档或 `benchmark_serving.py` 源码里直接出现的名字。
- **版本相关**：分位数字段名（`p{N}_<metric>_ms`）来自结果文件的实际用法，
  不同版本可能增删；跨版本对比前先 `--save-result` 跑一次看实际键名。
- **推断**：由文档描述推出的名字，未经一手确认。**不要拿推断项做自动化断言。**

用法
----
    python3 tools/profiling_bench/serving_schema.py --demo      # ITL/TPOT 分母演示
    python3 tools/profiling_bench/serving_schema.py             # 字段映射与边界表
    python3 tools/profiling_bench/serving_schema.py --check     # 落盘 + 逐字段回读校验
"""

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from profiling_bench.serving_model import (  # noqa: E402
    ServerCfg, arrival_times, simulate, summarize)


# ---------------------------------------------------------------------------
# 第 1 节  字段映射表
# ---------------------------------------------------------------------------

# group: A = 模型能给（可直接对齐）；B = 模型给不出（需要真实流式时序）
# confidence: verified | versioned | inferred
FIELD_MAP = [
    # ---- 汇总：条数、时长、token 数 ----
    ("completed", "completed", "A", "verified",
     "成功完成的请求数。分母口径要一致：被拒/超时/答案错的算不算，各工具不同。"),
    ("duration", "duration", "A", "verified", "整段挂钟秒数。"),
    ("n_offered", "—", "A", "inferred",
     "发出的请求数。vLLM 结果里没有独立的 offered 字段，只能从 num_prompts 推。"),
    ("total_input_tokens", "total_input_tokens", "A", "verified", "输入 token 总数。"),
    ("total_output_tokens", "total_output_tokens", "A", "verified", "输出 token 总数。"),

    # ---- 吞吐 ----
    ("request_throughput", "request_throughput", "A", "verified", "req/s。"),
    ("output_throughput", "output_throughput", "A", "verified",
     "输出 tok/s。**这是判断 decode 是否饱和的那一个数**。"),
    ("total_token_throughput", "total_token_throughput", "A", "verified",
     "输入+输出 tok/s。分母里混了 prefill，和 output 不是一回事。"),

    # ---- TTFT ----
    ("mean_ttft_ms", "mean_ttft_ms", "A", "verified", "均值。分布式/排队场景下几乎总是骗人的。"),
    ("median_ttft_ms", "median_ttft_ms", "A", "verified", "中位数。"),
    ("p90_ttft_ms", "p90_ttft_ms", "A", "cond",
     "**条件出现**：只有 --metric-percentiles 里包含 90 才有；默认值是 99，"
     "所以默认跑出来的结果文件里没有这个键。"),
    ("p99_ttft_ms", "p99_ttft_ms", "A", "cond",
     "同样是条件出现，只不过 99 是默认值，所以默认就有。"),

    # ---- TPOT ----
    ("mean_tpot_ms", "mean_tpot_ms", "A", "verified", "每条请求算一次再聚合。"),
    ("median_tpot_ms", "median_tpot_ms", "A", "verified", "同上，中位数。"),
    ("p90_tpot_ms", "p90_tpot_ms", "A", "cond", "条件出现，同 p90_ttft_ms。"),
    ("p99_tpot_ms", "p99_tpot_ms", "A", "cond", "条件出现，默认有。"),

    # ---- E2EL ----
    ("mean_e2el_ms", "mean_e2el_ms", "A", "verified", "端到端时延。"),
    ("median_e2el_ms", "median_e2el_ms", "A", "verified", "同上，中位数。"),
    ("p99_e2el_ms", "p99_e2el_ms", "A", "cond", "条件出现，默认有。"),
    ("—", "—", "A", "self",
     "注意 goodput 结构里的 ttft_ms / tpot_ms 是 **SLO 阈值**，不是测量值；"
     "别把它和逐请求真值数组 ttfts 看混。"),

    # ---- goodput ----
    ("slo_attainment", "—", "A", "inferred",
     "达标率。由 --goodput 的 SLO 口径算出，但结果文件里通常只给 goodput 数，"
     "不给达标率本身；要做容量边界判读得自己算。"),
    ("request_goodput", "request_goodput", "A", "inferred",
     "满足 SLO 的 req/s。名字是推断的：文档只说明 --goodput 会加 goodput 指标，"
     "未给字段名。**不要拿它做自动化断言。**"),
    ("output_token_goodput", "—", "A", "inferred",
     "满足 SLO 的输出 tok/s。vLLM 侧口径请自行核对。"),

    # ---- 分位比值（本系列自造，工具里没有）----
    ("ttft_p99_over_p50", "—", "A", "self",
     "自造合成量。工具不报这个比值，但它是**唯一能把「正在变饱和」和"
     "「已经饱和」区分开**的读数。"),

    # ---- ITL：模型给不出 ----
    ("—", "mean_itl_ms", "B", "verified",
     "相邻流式输出的平均间隔。需要**流式**逐块到达时刻；队列模型只有逐请求的"
     "首/末时刻，没有块边界，算不出。"),
    ("—", "median_itl_ms", "B", "verified", "同上。"),
    ("—", "std_itl_ms", "B", "verified", "同上。"),
    ("—", "p99_itl_ms", "B", "versioned", "同上。"),
    ("—", "ttfts", "B", "verified",
     "逐请求 TTFT 原始数组（--save-detailed）。模型能给，但工具侧才是真值。"),
    ("—", "itls", "B", "verified", "逐请求 ITL 数组。模型给不出。"),

    # ---- 服务端才有 ----
    ("n_preemptions_in_model", "—", "C", "self",
     "模型内抢占次数。真实 vLLM 有 vllm:num_preemptions_total 指标，"
     "但**客户端压测结果里没有**——它是服务端 Prometheus 指标。"),
    ("—", "—", "C", "verified",
     "KV 块占用率、等待队列长度、prefix cache 命中率：都是服务端指标。"
     "客户端只能看到它们**造成的结果**（TTFT 起飞的形状），看不到原因。"),
]

GROUP_TITLE = {
    "A": "A 组 · 队列模型能给，可与压测输出逐字段对照",
    "B": "B 组 · 需要真实流式时序，队列模型给不出",
    "C": "C 组 · 服务端才有，客户端压测看不到",
}

CONF_TAG = {
    "verified": "已核验  ",
    "versioned": "版本相关",
    "cond": "条件出现",
    "inferred": "推断    ",
    "self": "自造    ",
}


def print_field_map():
    print("=" * 108)
    print("字段映射：队列模型输出 ↔ vLLM `--save-result`")
    print("=" * 108)
    print("可信度说明：")
    print("  已核验   vLLM 文档或 benchmark_serving.py 源码里直接出现的名字")
    print("  版本相关 分位数字段名 p{N}_<metric>_ms，跨版本可能增删；换版本先跑一次看键名")
    print("  推断     由文档描述推出的名字，未经一手确认，不要用于自动化断言")
    print("  自造     本系列自己的合成量，工具里没有对应字段")
    print()
    for g in ("A", "B", "C"):
        print("-" * 108)
        print(GROUP_TITLE[g])
        print("-" * 108)
        print("  %-40s %-26s %-8s %s" % ("模型字段", "vLLM 字段", "可信度", "说明"))
        for our, vllm, grp, conf, note in FIELD_MAP:
            if grp != g:
                continue
            print("  %-40s %-26s %-8s %s" % (our, vllm, CONF_TAG[conf], note))
        print()
    print("关键提醒：**不要把 B/C 组的量当成「也是模型算的」**。")
    print("队列模型回答「拐点在哪、由什么决定」；它不回答「ITL 的尾部形态」。")
    print()


# ---------------------------------------------------------------------------
# 第 2 节  ITL vs TPOT 的分母演示
# ---------------------------------------------------------------------------

def itl_vs_tpot(chunks, ttft_ms=100.0, e2el_ms=180.0):
    """`chunks` = 每个流式输出携带的 token 数序列。

    返回 (n_out, n_gaps, mean_itl, tpot, ratio)。
    ITL 只在**块边界**上产生样本，块内多个 token 不产生样本。
    """
    n_out = sum(chunks)
    n_gaps = len(chunks) - 1
    if n_gaps <= 0:
        mean_itl = float("nan")
    else:
        mean_itl = (e2el_ms - ttft_ms) / n_gaps
    tpot = (e2el_ms - ttft_ms) / (n_out - 1) if n_out > 1 else float("nan")
    ratio = (mean_itl / tpot) if tpot and tpot > 0 else float("nan")
    return n_out, n_gaps, mean_itl, tpot, ratio


def print_itl_demo():
    print("=" * 96)
    print("ITL 与 TPOT 的分母：同一个流，两个数可以差几倍")
    print("=" * 96)
    print("固定 TTFT = 100 ms、E2EL = 180 ms（即解码段共 80 ms），只改每个流式输出带几个 token。")
    print()
    print("%-22s %6s %7s %12s %12s %10s" %
          ("每个流式输出带 token 数", "n_out", "n_gaps", "mean ITL", "TPOT", "ITL/TPOT"))
    print("-" * 96)
    cases = [
        ("全部 1 个（常规解码）", [1, 1, 1, 1, 1]),
        ("2 个 / 1 个 / 2 个", [2, 1, 2]),
        ("2 个 / 2 个 / 2 个", [2, 2, 2]),
        ("3 个 / 3 个", [3, 3]),
    ]
    for label, chunks in cases:
        n_out, n_gaps, itl, tpot, ratio = itl_vs_tpot(chunks)
        print("%-22s %6d %7d %10.1f ms %10.1f ms %10.2f" %
              (label, n_out, n_gaps, itl, tpot, ratio))
    print("-" * 96)
    print("闭式关系：mean_ITL / TPOT = (n_out − 1) / n_streamed_gaps")
    print()
    print("第二行（2 个 / 1 个 / 2 个）就是 vLLM 文档里的例子：两次 40 ms 的流式间隔 →")
    print("mean ITL = 40 ms；同一份时序的 TPOT = (180 − 100) / (5 − 1) = 20 ms/token。")
    print("即 mean ITL = 2 × TPOT。**两个数都对，因为问的不是同一件事。**")
    print()
    print("第一行（每个输出恰好一个 token）是两者相近的**唯一**情形，")
    print("比值 1.00——「ITL 和 TPOT 差不多」这句话只在这种情况下成立。")
    print()
    print("实操含义：开投机解码后每个流式输出接受的 token 数上升，")
    print("n_streamed_gaps 下降，mean ITL 按 (n_out−1)/n_gaps 同比例放大，")
    print("而 TPOT（每条请求摊到每个 token）不受这个口径影响。")
    print("只盯 ITL 会把这个口径放大读成「服务变慢了」。")
    print()


# ---------------------------------------------------------------------------
# 第 3 节  落盘与逐字段回读
# ---------------------------------------------------------------------------

def build_result(summary, cfg, slo_ttft_ms, slo_tpot_ms, label="serving_model"):
    """按 vLLM `--save-result` 的形状组织一份结果。"""
    out = {
        "label": label,
        "args": {
            "max_num_seqs": cfg.max_num_seqs,
            "max_num_batched_tokens": cfg.max_num_batched_tokens,
            "block_size": cfg.block_size,
            "kv_token_capacity": cfg.kv_token_capacity,
        },
        "completed": summary["completed"],
        "duration": summary["duration"],
        "total_input_tokens": summary["total_input_tokens"],
        "total_output_tokens": summary["total_output_tokens"],
        "request_throughput": summary["request_throughput"],
        "output_throughput": summary["output_throughput"],
        "total_token_throughput": summary["total_token_throughput"],
        "mean_ttft_ms": summary["mean_ttft_ms"],
        "median_ttft_ms": summary["median_ttft_ms"],
        "p90_ttft_ms": summary["p90_ttft_ms"],
        "p99_ttft_ms": summary["p99_ttft_ms"],
        "mean_tpot_ms": summary["mean_tpot_ms"],
        "median_tpot_ms": summary["median_tpot_ms"],
        "p90_tpot_ms": summary["p90_tpot_ms"],
        "p99_tpot_ms": summary["p99_tpot_ms"],
        "mean_e2el_ms": summary["mean_e2el_ms"],
        "median_e2el_ms": summary["median_e2el_ms"],
        "p99_e2el_ms": summary["p99_e2el_ms"],
    }
    if slo_ttft_ms is not None and slo_tpot_ms is not None:
        out["goodput"] = {
            "ttft_ms": slo_ttft_ms,
            "tpot_ms": slo_tpot_ms,
            "request_goodput": summary.get("request_goodput"),
            "output_token_goodput": summary.get("output_token_goodput"),
            "slo_attainment": summary.get("slo_attainment"),
        }
    return out


def check(explicit=None):
    """跑一遍、落盘、回读，断言 A 组字段一个不缺。"""
    cfg = ServerCfg(max_num_seqs=32, max_num_batched_tokens=2048,
                    kv_token_capacity=262144)
    times, _ = arrival_times("poisson", 8.0, 300, seed=0)
    recs, meta = simulate(cfg, times, 1024, 128)
    s = summarize(recs, meta, cfg, 300.0, 30.0)
    result = build_result(s, cfg, 300.0, 30.0)

    present = set(result.keys()) | set(result.get("goodput", {}).keys())
    required = [v for _, v, g, _, _ in FIELD_MAP if g == "A" and v != "—"]
    our_keys = set(k for k, v, g, _, _ in FIELD_MAP
                   if g == "A" and k != "—" and v == "—")
    missing = [v for v in required if v not in present]
    unknown = [k for k in present
               if k not in required and k not in our_keys
               and k not in ("label", "args", "goodput",
                             "ttft_ms", "tpot_ms")]

    print("=" * 88)
    print("逐字段回读校验")
    print("=" * 88)
    print("A 组要求字段 %d 个，实际落盘 %d 个。" % (len(required), len(present)))
    if missing:
        print("**缺失**：%s" % ", ".join(missing))
    else:
        print("缺失：无")
    if unknown:
        print("额外字段（不在映射表里）：%s" % ", ".join(sorted(unknown)))
    print()
    if explicit:
        with open(explicit, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print("已写入 %s" % explicit)
        print()
    print("落盘样例（前 14 行）：")
    for i, line in enumerate(json.dumps(result, indent=2,
                                        ensure_ascii=False).splitlines()):
        if i >= 14:
            break
        print("  " + line)
    print()
    return not missing


# ---------------------------------------------------------------------------
# 第 4 节  CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="指标口径对齐：队列模型输出 ↔ vLLM --save-result 字段")
    p.add_argument("--demo", action="store_true", help="只跑 ITL/TPOT 分母演示")
    p.add_argument("--check", action="store_true",
                   help="逐字段回读校验 A 组字段（不落盘；配合 --emit 才写文件）")
    p.add_argument("--emit", default=None,
                   help="把结果 JSON 写到这个路径"
                        "（默认不写，避免产物落到仓库根目录）")
    args = p.parse_args(argv)

    if args.demo:
        print_itl_demo()
        return 0

    print_field_map()
    if args.check or args.emit:
        ok = check(args.emit)          # 不给路径就只在内存里回读校验
        return 0 if ok else 1
    print("（加 --check 回读校验 A 组字段，不落盘；"
          "加 --emit PATH 才写文件；加 --demo 看 ITL/TPOT 分母演示）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
