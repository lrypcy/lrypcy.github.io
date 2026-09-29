#!/usr/bin/env python3
"""多 rank 训练 trace 的聚合分析：时间分解、idle 三分、comm/compute overlap、掉队卡。

两条使用路径
------------
1. **合成 trace 自校验**（默认）：按预置真值构造一份多 rank chrome trace，
   再用本脚本分析它，把「量出来的值」和「造进去的真值」并排打印。
   分析实现有 bug 时，两者对不上——这是本脚本存在的主要理由。
2. **真实 trace 分析**：`--trace <file.json | file.json.gz>`。
   吃 Perfetto / kineto 的 chrome trace 格式（`traceEvents` + `ph == "X"`），
   也就是 PyTorch `torch.profiler` 导出的 `torch_profile/rank-N.json.gz`、
   以及 Megatron `--profile` 的产物、nsys 的 `.json` 导出。

为什么必须自校验
----------------
trace 分析的三个核心量（overlap、idle、straggler）都有多种同样"说得通"的算法，
不同算法在边缘 case 下差出几十个百分点是很常见的。没有真值反算，
一份看起来很漂亮的报告可能整体是错的。

口径说明（这些选择都会影响结论，必须写清楚）
--------------------------------------------
* `comm`：集合通信 kernel（名字含 nccl / AllReduce / AllGather / ReduceScatter /
  SendRecv）+ `cat` 为 `gpu_memcpy` / `communication` 的事件。
* `compute`：设备侧其余 kernel。
* `idle`：迭代墙钟里既无 compute 也无 comm 的时间。
* `overlap`：comm 区间与 compute 区间的**交集长度**（区间代数，不是"各取一半"的估算）。
* overlap 有两个分母，给出的是两个完全不同的结论：
    - `overlap / comm_total` —— "通信有多少被藏住了"（效率视角）
    - `overlap / iter_time`  —— "通信还能再藏多少"（收益上界视角）
  报告里两个都给，并明确标注是哪一种。这是 HTA `get_comm_comp_overlap()` 的语义，
  本实现按同样定义做，便于对齐。

idle 三分（沿用 HTA 的官方措辞）
-------------------------------
HTA v0.6.0 的 README 把 GPU 空闲拆成 **host wait / kernel wait / unknown**，
本实现沿用同样的三分类名，便于跨工具对照：
* `host-wait`：空隙期间 host 在跑非 launch 的活（dataloader / python_function）→ 等数据 / 等 Python。
* `kernel-wait`：空隙前 host 已经 post 了下一个 kernel（launch 事件在空隙之前），
  但设备没开工 → 流内依赖等待 / launch 开销。
* `unknown`：窗口内看不到任何 host 侧解释 → 通常是等对端（暴露出来的通信）、或 host 侧采样缺失。

三者的修法完全不同，混在一起统计会把人引向错误的优化方向。

**注意本实现的判据是自己的**（按 host 侧事件的时间重叠关系推断），
与 HTA 基于 Kineto CPU 侧算子链的推断规则**不完全一致**。
跨工具比数之前必须先对齐判据，否则同一个 trace 会给出两组不同的三分占比。

用法
----
    python3 tools/profiling_bench/trace_agg.py                  # 自校验
    python3 tools/profiling_bench/trace_agg.py --selftest       # 同上
    python3 tools/profiling_bench/trace_agg.py --trace t.json.gz --pid 3

无第三方依赖。
"""

import argparse
import gzip
import json
import statistics
import sys
from bisect import bisect_left

US = 1.0  # 时间单位统一用微秒（chrome trace 原生单位）

# ----------------------------------------------------------------------------
# 区间代数
# ----------------------------------------------------------------------------


def union(intervals):
    """合并重叠区间，返回按起点排序的不相交区间列表。"""
    if not intervals:
        return []
    iv = sorted(intervals)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def total_len(intervals):
    return sum(e - s for s, e in union(intervals))


def intersection_len(a, b):
    """两个区间集合的交集总长度（先各自并集化，再双指针求交）。"""
    ua, ub = union(a), union(b)
    if not ua or not ub:
        return 0.0
    starts_b = [s for s, _ in ub]
    total, j = 0.0, 0
    for s, e in ua:
        j = max(0, bisect_left(starts_b, s) - 1)
        while j < len(ub):
            s2, e2 = ub[j]
            if s2 >= e:
                break
            lo, hi = max(s, s2), min(e, e2)
            if hi > lo:
                total += hi - lo
            j += 1
    return total


def sparse_union_len(intervals):
    """sweep line 求并集长度（比 pairwise 快，用于大 trace）。"""
    ev = []
    for s, e in intervals:
        if e > s:
            ev.append((s, 1))
            ev.append((e, -1))
    ev.sort()
    depth, prev, total = 0, None, 0.0
    for x, d in ev:
        if depth > 0 and prev is not None:
            total += x - prev
        depth += d
        prev = x
    return total


# ----------------------------------------------------------------------------
# 事件分类
# ----------------------------------------------------------------------------

COMM_HINTS = ("nccl", "allreduce", "all_reduce", "allgather", "all_gather",
              "reducescatter", "reduce_scatter", "sendrecv", "send_recv",
              "broadcast", "alltoall", "all_to_all")


def classify(ev):
    """返回 ('compute'|'comm'|'host-<kind>'|'other', ...)。"""
    cat = (ev.get("cat") or "").lower()
    name = (ev.get("name") or "").lower()

    if cat in ("gpu_memcpy", "communication"):
        return "comm"
    if cat == "kernel":
        if any(h in name for h in COMM_HINTS):
            return "comm"
        return "compute"
    if cat in ("python_function", "dataloader", "data_loader"):
        return "host-work"
    if cat == "user_annotation":
        # torch.profiler 的 record_function / NVTX 区间。名字叫 iteration 的
        # 当作迭代标记，由 analyze() 单独处理，不参与 compute/comm 统计。
        return "marker" if "iteration" in name else "host-work"
    if cat == "cpu_op":
        return "host-launch" if "launch" in name else "host-work"
    if cat == "cuda_runtime":
        return "host-launch"
    return "other"


# ----------------------------------------------------------------------------
# 合成 trace：真值 → 事件
# ----------------------------------------------------------------------------


def build_synthetic(n_ranks=8, iters=3, compute_ms=100.0, comm_ms=40.0,
                    overlap_frac=0.5, idle_ms=20.0,
                    idle_split=(0.5, 0.3, 0.2), straggler=None,
                    straggler_penalty_ms=18.0, tail_kernel_us=100.0, seed=1234):
    """按真值构造 chrome trace。

    每个 rank、每次迭代的时间线：
        [compute C] 与 [comm M] 放在同一条时间轴上，其中 overlap_frac * M 与
        compute 区间重叠（被藏住），其余 (1-overlap_frac) * M 暴露在外。
        之后再加 idle_ms 的空隙，按 idle_split 分成三种成因。

    返回 (trace, truth)
    """
    import random
    rng = random.Random(seed)

    events = []
    truth = {"n_ranks": n_ranks, "iters": iters,
             "compute_ms": compute_ms, "comm_ms": comm_ms,
             "overlap_frac": overlap_frac, "idle_ms": idle_ms,
             "idle_split": idle_split, "straggler": straggler,
             "straggler_penalty_ms": straggler_penalty_ms,
             "per_rank": {}}

    n_gap = 10                     # 把 idle 切成 10 个空隙，让 50/30/20 不被量化吃掉
    types = ("host-wait", "kernel-wait", "unknown")
    # 按比例给每个空隙分配成因，保证总数吻合
    quota = {t: round(idle_split[i] * n_gap) for i, t in enumerate(types)}
    # 修正舍入误差
    diff = n_gap - sum(quota.values())
    if diff:
        quota["unknown"] += diff
    gap_types = []
    for t in types:
        gap_types += [t] * quota[t]
    gap_types = (gap_types * n_gap)[:n_gap]
    truth["gap_types"] = list(gap_types)
    truth["gap_len_us"] = idle_ms * 1000.0 / n_gap
    truth["tail_kernel_us"] = tail_kernel_us

    for it in range(iters):
        t0 = it * (compute_ms + comm_ms + idle_ms + 50.0) * 1000.0  # 迭代间隔留 50ms

        for rank in range(n_ranks):
            pid = rank
            # straggler == -1 表示短板轮转（每个迭代换一张卡）
            s_rank = (it % n_ranks) if straggler == -1 else straggler
            penalty = straggler_penalty_ms * 1000.0 if rank == s_rank else 0.0
            # 掉队表现为 compute 变慢（更常见）——是真实的长尾形态
            c = compute_ms * 1000.0 + penalty
            m = comm_ms * 1000.0
            idle = idle_ms * 1000.0
            if c <= 0:
                c = 1000.0
            # comm 总量 m，其中 overlap_frac * m 与 compute 重叠（被藏住）
            ov = overlap_frac * m
            exposed = m - ov

            # --- compute：切成 4 个 kernel，铺满 [0, c]
            nk = 4
            per = c / nk
            xs = t0
            for k in range(nk):
                events.append(_ev("compute_kernel_%d" % k, pid, 1, "kernel",
                                  xs, per))
                # host 侧对应的 launch（在 kernel 之前一点点）
                events.append(_ev("launch_kernel_%d" % k, pid, 100,
                                  "cuda_runtime", xs - 30.0, 12.0))
                xs += per

            # --- comm：重叠部分切成 2 段，落在 compute 区间内部
            if ov > 0:
                half = ov / 2.0
                events.append(_ev("nccl:all_reduce", pid, 2, "kernel",
                                  t0 + c * 0.15, half))
                events.append(_ev("nccl:reduce_scatter", pid, 2, "kernel",
                                  t0 + c * 0.60, half))
            # 暴露部分：放在 compute 之后（串行外露）
            if exposed > 0:
                events.append(_ev("nccl:all_gather", pid, 2, "kernel",
                                  t0 + c, exposed))

            # --- idle：真实 trace 里空隙不会互相贴着，总是被 kernel 隔开。
            #     所以尾段做成「小 kernel + 空隙」交替，空隙才切得开。
            per_gap = idle / n_gap
            cursor = t0 + c + exposed
            for gi, gt in enumerate(gap_types):
                events.append(_ev("tail_kernel_%d" % gi, pid, 1, "kernel",
                                  cursor, tail_kernel_us))
                g0 = cursor + tail_kernel_us
                cursor = g0 + per_gap
                if gt == "host-wait":
                    # host 在整个空隙里跑 dataloader / python，设备只能等
                    events.append(_ev("DataLoader._next", pid, 100,
                                      "python_function", g0, per_gap))
                elif gt == "kernel-wait":
                    # host 已经把下一个 kernel post 出去了（落在空隙开始之前），
                    # 设备却迟迟没开工 —— 典型是流内依赖 / launch 开销
                    events.append(_ev("launch_kernel_next", pid, 100,
                                      "cuda_runtime", g0 - 20.0, 10.0))
                # unknown：什么都不放（真实场景里通常是等对端）

            dev_compute = c + n_gap * tail_kernel_us
            iter_us = dev_compute + exposed + idle

            # --- 迭代标记：真实 trace 里这是 NVTX / record_function 区间，
            #     分析器靠它把 trace 切成迭代，否则只能跨迭代求总和。
            #     放在最后 append 是因为需要先知道 iter_us。
            events.append(_ev("iteration", pid, 1, "user_annotation",
                              t0, iter_us))

            truth["per_rank"].setdefault(rank, {})
            truth["per_rank"][rank][it] = {
                "compute_us": dev_compute, "comm_us": m, "overlap_us": ov,
                "exposed_comm_us": exposed,
                "idle_us": idle,
                "iter_us": iter_us,
            }

    # 补一个 metadata 事件，模拟真实 chrome trace
    events.insert(0, {"ph": "M", "name": "process_name", "pid": 0,
                      "args": {"name": "rank-0"}})
    trace = {"traceEvents": events,
             "displayTimeUnit": "ms",
             "metadata": {"synthetic": True}}
    return trace, truth


def _ev(name, pid, tid, cat, ts, dur):
    return {"ph": "X", "name": name, "pid": pid, "tid": tid,
            "cat": cat, "ts": round(ts, 3), "dur": round(dur, 3)}


# ----------------------------------------------------------------------------
# 分析
# ----------------------------------------------------------------------------


def load_trace(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze(trace, idle_host_work_min_us=50.0, verbose=False):
    """分析一份 chrome trace，返回结构化结果。

    分段策略：
      * trace 里有名字含 "iteration" 的 `user_annotation` 区间（NVTX / record_function）
        → 以它为迭代窗口，逐个迭代分析后取**平均**。真实训练 trace 都有这层标注。
      * 没有标注 → 退化成「整段跨度」口径，此时跨迭代的间隙会被算成 idle，
        结果会系统性偏大。这种情况会打印警告，不静默给数。
    """
    evs = [e for e in trace.get("traceEvents", [])
           if e.get("ph") == "X" and e.get("dur")]
    if not evs:
        return None

    DEV_CATS = ("kernel", "gpu_memcpy", "communication")
    dev_evs = [e for e in evs if e.get("cat") in DEV_CATS]
    if not dev_evs:
        return None
    ranks = sorted({e["pid"] for e in dev_evs})

    markers = {}
    for e in evs:
        if classify(e) == "marker":
            markers.setdefault(e["pid"], []).append((e["ts"], e["ts"] + e["dur"]))
    segmented = bool(markers)

    win_start = min(e["ts"] for e in dev_evs)
    win_end = max(e["ts"] + e["dur"] for e in dev_evs)

    per_rank = {}
    for r in ranks:
        rk = [e for e in evs if e["pid"] == r]
        compute = [(e["ts"], e["ts"] + e["dur"]) for e in rk
                   if classify(e) == "compute"]
        comm = [(e["ts"], e["ts"] + e["dur"]) for e in rk
                if classify(e) == "comm"]
        host_work = [(e["ts"], e["ts"] + e["dur"]) for e in rk
                     if classify(e) == "host-work"]
        host_launch = [(e["ts"], e["ts"] + e["dur"]) for e in rk
                       if classify(e) == "host-launch"]

        segs = markers.get(r) or [(min(x[0] for x in compute + comm),
                                   max(x[1] for x in compute + comm))
                                  if (compute or comm) else (win_start, win_end)]
        segs = sorted(segs)

        acc = {"iter_us": 0.0, "compute_us": 0.0, "comm_us": 0.0,
               "overlap_us": 0.0, "idle_us": 0.0,
               "host-wait": 0.0, "kernel-wait": 0.0, "unknown": 0.0}
        n_gaps = 0
        iter_durs = []
        for lo, hi in segs:
            if hi <= lo:
                continue
            iter_durs.append(hi - lo)
            cs = [iv for iv in compute if iv[0] < hi and iv[1] > lo]
            ms = [iv for iv in comm if iv[0] < hi and iv[1] > lo]
            hw = [iv for iv in host_work if iv[0] < hi and iv[1] > lo]
            hl = [iv for iv in host_launch if iv[0] < hi and iv[1] > lo]
            # 夹到窗口内，避免跨窗口的 kernel 把时长算重
            cs = [(max(s, lo), min(e, hi)) for s, e in cs]
            ms = [(max(s, lo), min(e, hi)) for s, e in ms]

            busy = union(cs + ms)
            acc["iter_us"] += hi - lo
            acc["compute_us"] += sparse_union_len(cs)
            acc["comm_us"] += sparse_union_len(ms)
            acc["overlap_us"] += intersection_len(cs, ms)
            idle_iv = [(s, e) for s, e in _complement(busy, lo, hi)
                       if e - s > 1e-6]
            acc["idle_us"] += sum(e - s for s, e in idle_iv)
            n_gaps += len(idle_iv)
            for gs, ge in idle_iv:
                for k, v in _triage_split(gs, ge, hw, hl).items():
                    acc[k] += v

        n = len([1 for lo, hi in segs if hi > lo])
        if n == 0:
            continue
        per_rank[r] = {
            "n_iters": n,
            "iter_us": acc["iter_us"] / n,
            "compute_us": acc["compute_us"] / n,
            "comm_us": acc["comm_us"] / n,
            "overlap_us": acc["overlap_us"] / n,
            "exposed_comm_us": (acc["comm_us"] - acc["overlap_us"]) / n,
            "idle_us": acc["idle_us"] / n,
            "idle_split": {"host-wait": acc["host-wait"] / n,
                           "kernel-wait": acc["kernel-wait"] / n,
                           "unknown": acc["unknown"] / n},
            "n_idle_gaps": n_gaps / n,
            "iter_durs": iter_durs,
        }

    if not per_rank:
        return None
    if not segmented and verbose:
        print("[!] trace 里没有 iteration 标注，已退化为整段跨度口径：")
        print("    跨迭代间隙会被算成 idle，读数系统性偏大。")

    iters = sorted(per_rank)
    durs = [per_rank[r]["iter_us"] for r in iters]
    slowest = max(iters, key=lambda r: per_rank[r]["iter_us"])
    fastest = min(iters, key=lambda r: per_rank[r]["iter_us"])
    med = statistics.median(durs)
    return {
        "ranks": iters,
        "per_rank": per_rank,
        "window_us": (win_start, win_end),
        "segmented": segmented,
        "slowest_rank": slowest,
        "fastest_rank": fastest,
        "iter_mean_us": sum(durs) / len(durs),
        "iter_median_us": med,
        "iter_max_us": max(durs),
        "iter_min_us": min(durs),
        "max_over_median_pct": (max(durs) / med - 1.0) * 100.0,
        "max_over_mean_pct": (max(durs) / (sum(durs) / len(durs)) - 1.0) * 100.0,
        "max_over_min_pct": (max(durs) / min(durs) - 1.0) * 100.0,
    }


def _complement(intervals, lo, hi):
    """[lo,hi] 上未被子区间覆盖的空隙。"""
    out, cur = [], lo
    for s, e in union(intervals):
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        out.append((cur, hi))
    return out


def _triage_split(gs, ge, host_work, host_launch, min_fill=0.5):
    """把一段设备空闲切成三类，返回 {成因: 微秒}。

    粒度说明：按 host 侧「真活」的边界把空隙切成原子段，再逐段归因。
    这样一段空隙可以同时含多种成因（真实 trace 里很常见）。
    归因规则（按优先级）：
      1. 原子段被 host 侧真活（dataloader / python_function）覆盖 ≥ min_fill
         → host-wait
      2. 空隙起点前 100 us 内有一个 launch 事件（host 已经 post 了下一个 kernel）
         → kernel-wait
      3. 其余 → unknown

    注意规则 2 是**整段空隙级别**的判断，不是逐原子段：一个 launch 只能解释
    紧随其后的那个空隙，不能反过来解释它前面的空隙。
    """
    out = {"host-wait": 0.0, "kernel-wait": 0.0, "unknown": 0.0}

    bnds = {gs, ge}
    for s, e in host_work:
        a, b = max(s, gs), min(e, ge)
        if b > a:
            bnds.add(a)
            bnds.add(b)
    pts = sorted(bnds)

    pending_launch = any(gs - 100.0 <= (s + e) / 2.0 <= gs + 20.0
                         for s, e in host_launch)

    for a, b in zip(pts, pts[1:]):
        if b <= a:
            continue
        covered = _covered(host_work, a, b)
        if covered >= min_fill * (b - a):
            out["host-wait"] += b - a
        elif pending_launch:
            out["kernel-wait"] += b - a
        else:
            out["unknown"] += b - a
    return out


def _covered(intervals, a, b):
    """[a,b] 被 intervals 覆盖的长度（并集，不重复计）。"""
    pieces = [(max(s, a), min(e, b)) for s, e in intervals
              if s < b and e > a]
    return sparse_union_len(pieces)


# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------


def pct(x, y):
    return 0.0 if y <= 0 else 100.0 * x / y


def report_selftest(res, truth):
    print("=" * 74)
    print("合成 trace 自校验：真值 ↔ 反算值")
    print("=" * 74)
    print("配置：%d rank × %d 迭代，compute %.1f ms，comm %.1f ms，"
          "overlap %.0f%%，idle %.1f ms，掉队 rank %s（+%.1f ms compute）"
          % (truth["n_ranks"], truth["iters"], truth["compute_ms"],
             truth["comm_ms"], truth["overlap_frac"] * 100,
             truth["idle_ms"], truth["straggler"],
             truth["straggler_penalty_ms"]))

    # 选一个在迭代 0 没有被罚时的 rank 做真值对照
    s0 = 0 if truth["straggler"] == -1 else truth["straggler"]
    ref = 0 if s0 != 0 else 1
    t = truth["per_rank"][ref][0]
    a = res["per_rank"][ref]

    rows = [
        ("compute_us", t["compute_us"], a["compute_us"]),
        ("comm_us", t["comm_us"], a["comm_us"]),
        ("overlap_us", t["overlap_us"], a["overlap_us"]),
        ("exposed_comm_us", t["exposed_comm_us"], a["exposed_comm_us"]),
        ("idle_us", t["idle_us"], a["idle_us"]),
        ("iter_us", t["iter_us"], a["iter_us"]),
    ]
    print("\n-- rank %d（非掉队）单次迭代 --" % ref)
    print("   %-18s %12s %12s %10s" % ("量", "真值(us)", "反算(us)", "相对误差"))
    worst = 0.0
    for name, tv, av in rows:
        err = 0.0 if tv == 0 else (av - tv) / tv * 100.0
        worst = max(worst, abs(err))
        flag = "  OK" if abs(err) < 2.0 else "  <<< 偏差"
        print("   %-18s %12.3f %12.3f %9.2f%%%s"
              % (name, tv, av, err, flag))

    print("\n-- idle 三分（构造时切成 %d 个空隙，每段 %.0f us）--"
          % (len(truth["gap_types"]), truth["gap_len_us"]))
    spl = truth["idle_split"]
    names = ("host-wait", "kernel-wait", "unknown")
    tcount = {n: truth["gap_types"].count(n) for n in names}
    print("   %-12s %12s %12s %10s" % ("成因", "真值(us)", "反算(us)", "占比"))
    for n in names:
        tv = tcount[n] * truth["gap_len_us"]
        av = a["idle_split"][n]
        print("   %-12s %12.3f %12.3f %9.1f%%"
              % (n, tv, av, pct(av, a["idle_us"])))
    split_err = 0.0
    for n in names:
        tv = tcount[n] * truth["gap_len_us"]
        split_err = max(split_err, abs(a["idle_split"][n] - tv))
    print("   目标占比 %.0f/%.0f/%.0f%% 被量化成 %s（每段 %.0f us）"
          % (spl[0] * 100, spl[1] * 100, spl[2] * 100,
             "/".join(str(tcount[n]) for n in names), truth["gap_len_us"]))
    print("   三分绝对偏差最大 %.1f us %s"
          % (split_err, "OK" if split_err < 1.0 else "<<< 偏差"))

    print("\n-- overlap 的两个分母（同一份数据，两个结论）--")
    om = pct(a["overlap_us"], a["comm_us"])
    oi = pct(a["overlap_us"], a["iter_us"])
    print("   overlap / comm_total = %.1f%%   ← 「通信有多少被藏住了」（效率）"
          % om)
    print("   overlap / iter_time  = %.1f%%   ← 「通信最多还能再藏多少」（上界）"
          % oi)
    print("   真值 overlap/comm = %.1f%%" % (truth["overlap_frac"] * 100))

    print("\n-- 逐迭代最慢卡：短板是固定的还是一直在换？--")
    n_iter = max(len(res["per_rank"][r]["iter_durs"]) for r in res["ranks"])
    print("   %-6s %-12s %12s %14s" % ("迭代", "最慢 rank", "该卡 iter(us)",
                                       "vs 中位数"))
    for i in range(n_iter):
        d = {r: res["per_rank"][r]["iter_durs"][i] for r in res["ranks"]
             if len(res["per_rank"][r]["iter_durs"]) > i}
        if not d:
            continue
        sr = max(d, key=d.get)
        mv = statistics.median(d.values())
        print("   %-6d %-12d %12.1f %13.2f%%"
              % (i, sr, d[sr], (d[sr] / mv - 1.0) * 100.0))
    print("   → 每轮都是同一张卡：固定短板，去查那张卡本身（NIC / 拓扑 / 掉频）。")
    print("     每轮换卡：多半是资源争抢或负载不均（数据分片、专家路由、共享卡）。")
    print("     这个区分靠『平均值』永远看不出来。")

    print("\n-- 掉队卡 --")
    print("   最慢 rank %d  iter = %.1f us" % (res["slowest_rank"],
                                              res["iter_max_us"]))
    print("   最快 rank %d  iter = %.1f us" % (res["fastest_rank"],
                                              res["iter_min_us"]))
    print("   中位数 %.1f us，均值 %.1f us" % (res["iter_median_us"],
                                              res["iter_mean_us"]))
    print("   最慢 / 中位数 - 1 = +%.2f%%" % res["max_over_median_pct"])
    print("   最慢 / 均值   - 1 = +%.2f%%" % res["max_over_mean_pct"])
    print("   最慢 / 最快   - 1 = +%.2f%%" % res["max_over_min_pct"])
    print("   → 分布式里整步时长由最慢卡决定，所以「均值」这个数永远是乐观的：")
    print("     它把掉队卡的超时摊给了其它卡。")

    ok = worst < 2.0 and split_err < 5.0
    print("\n" + "=" * 74)
    print("[ok] 真值反算通过（最大相对误差 %.2f%%）" % worst
          if ok else "[!] 真值反算未通过，检查分析实现")
    return ok


def report_real(res, top=10):
    print("=" * 74)
    print("真实 trace 分析结果")
    print("=" * 74)
    r = res["ranks"][:top]
    print("%-8s %11s %11s %11s %11s %11s" %
          ("rank", "iter(us)", "compute", "comm", "overlap", "idle"))
    for k in r:
        d = res["per_rank"][k]
        print("%-8d %11.1f %11.1f %11.1f %11.1f %11.1f" %
              (k, d["iter_us"], d["compute_us"], d["comm_us"],
               d["overlap_us"], d["idle_us"]))
    print()
    print("最慢 rank %d（%.1f us），中位数 %.1f us，+%.2f%%"
          % (res["slowest_rank"], res["iter_max_us"],
             res["iter_median_us"], res["max_over_median_pct"]))
    d = res["per_rank"][res["slowest_rank"]]
    print("最慢 rank 的 idle 三分: host-wait %.1f / kernel-wait %.1f / unknown %.1f us"
          % (d["idle_split"]["host-wait"], d["idle_split"]["kernel-wait"],
             d["idle_split"]["unknown"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="真实 chrome trace（.json / .json.gz）")
    ap.add_argument("--pid", type=int, help="只分析指定 pid（默认全部）")
    ap.add_argument("--selftest", action="store_true", help="跑合成自校验")
    ap.add_argument("--n-ranks", type=int, default=8)
    ap.add_argument("--straggler", type=int, default=3)
    ap.add_argument("--overlap-frac", type=float, default=0.5)
    ap.add_argument("--dump", help="把合成 trace 写到这个路径，便于用 Perfetto 打开")
    args = ap.parse_args()

    if args.trace:
        trace = load_trace(args.trace)
        res = analyze(trace)
        if res is None:
            print("[!] 没在 trace 里找到设备侧事件（cat=kernel/gpu_memcpy）")
            return 2
        report_real(res)
        return 0

    trace, truth = build_synthetic(n_ranks=args.n_ranks,
                                   straggler=args.straggler,
                                   overlap_frac=args.overlap_frac)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(trace, f)
        print("合成 trace 已写出: %s（可直接拖进 Perfetto UI）" % args.dump)
    res = analyze(trace)
    ok = report_selftest(res, truth)

    # 再演示一次「重叠率变化时两个分母怎么分道扬镳」
    print("\n" + "=" * 74)
    print("重叠率扫描：两个分母给出的结论差距")
    print("=" * 74)
    print("%8s %14s %14s %12s" %
          ("overlap", "ov/comm (%)", "ov/iter (%)", "iter(us)"))
    for of in (0.0, 0.25, 0.5, 0.75, 1.0):
        tr, _ = build_synthetic(n_ranks=args.n_ranks, straggler=None,
                                overlap_frac=of)
        rs = analyze(tr)
        d = rs["per_rank"][0]
        print("%7.0f%% %14.1f %14.1f %12.1f"
              % (of * 100, pct(d["overlap_us"], d["comm_us"]),
                 pct(d["overlap_us"], d["iter_us"]), d["iter_us"]))
    print("""
读法：随着重叠变好，`ov/comm` 趋近 100%（通信基本藏完），
而 `ov/iter` 反而"看起来很低"——因为分母是迭代时长，它本来就不该接近 100%。
把这两个分母混用，就会出现"我们 overlap 只有 20%，还能大改"这种结论：
那个 20% 是 ov/iter，真正该看的是 ov/comm。""")

    # 短板轮转 vs 固定短板
    print("\n" + "=" * 74)
    print("短板形态对比：固定 vs 轮转（同样的平均拖慢量）")
    print("=" * 74)
    for label, s in (("固定 rank %d" % args.straggler, args.straggler),
                     ("每轮换卡", -1)):
        tr, _ = build_synthetic(n_ranks=args.n_ranks, straggler=s)
        rs = analyze(tr)
        print("\n[%s]" % label)
        n_iter = max(len(rs["per_rank"][r]["iter_durs"]) for r in rs["ranks"])
        for i in range(n_iter):
            d = {r: rs["per_rank"][r]["iter_durs"][i] for r in rs["ranks"]
                 if len(rs["per_rank"][r]["iter_durs"]) > i}
            sr = max(d, key=d.get)
            mv = statistics.median(d.values())
            print("   迭代 %d: 最慢 rank %d  %.1f us  (+%.2f%% vs 中位数)"
                  % (i, sr, d[sr], (d[sr] / mv - 1.0) * 100.0))
        print("   均值 %.1f / 中位数 %.1f / 最大 %.1f us"
              % (rs["iter_mean_us"], rs["iter_median_us"], rs["iter_max_us"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
