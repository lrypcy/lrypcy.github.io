"""复刻 Megatron `Timer` 的同步式计时语义，量化它对重叠的破坏。

背景（源码依据，commit 60e039626）：
    megatron/core/timers.py:143  Timer.start()  -> torch.cuda.synchronize() 后读 time.time()
    megatron/core/timers.py:156  Timer.stop()   -> torch.cuda.synchronize() 后读 time.time()
    megatron/core/timers.py:171  Timer.reset()  -> 只清 _elapsed，不清 _active_time
    megatron/core/timers.py:318  _get_global_min_max_time() -> 过滤 x > 0.0 后取 min/max

同步点会把「所有流都排空」这件事算进被测量的区间里，因此：
  1. 计时读数 >= 该阶段真实占用时间，膨胀量 = 边界处仍在飞的工作量；
  2. **重叠做得越好，膨胀越大** —— 你越会优化，内置 timer 越难看；
  3. 逐阶段计时求和不是任何真实的串行分解。

本脚本用纯 CPU 真线程（sleep 模拟设备侧耗时）复现这三条。
不需要 GPU，不需要 numpy。

用法：
    python3 tools/profiling_bench/timer_semantics.py
    python3 tools/profiling_bench/timer_semantics.py --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

# ---------------------------------------------------------------- 实验一


class _ComputeWorker(threading.Thread):
    """compute 流：逐 chunk 执行，每个 chunk 完成后放一个事件通知 comm 流。"""

    def __init__(self, n_chunks: int, chunk_ms: float):
        super().__init__(daemon=True)
        self.n_chunks = n_chunks
        self.chunk_s = chunk_ms / 1000.0
        self.done = [threading.Event() for _ in range(n_chunks)]
        self.busy_s = 0.0


    def run(self) -> None:
        for i in range(self.n_chunks):
            t0 = time.perf_counter()
            time.sleep(self.chunk_s)
            self.busy_s += time.perf_counter() - t0
            self.done[i].set()


class _CommWorker(threading.Thread):
    """comm 流：第 i 个 chunk 依赖 compute 流第 i 个 chunk 完成（反向重叠语义）。"""

    def __init__(self, compute: _ComputeWorker, chunk_ms: float):
        super().__init__(daemon=True)
        self.compute = compute
        self.chunk_s = chunk_ms / 1000.0
        self.busy_s = 0.0

    def run(self) -> None:
        for i in range(self.compute.n_chunks):
            self.compute.done[i].wait()
            if self.chunk_s <= 0:
                continue
            t0 = time.perf_counter()
            time.sleep(self.chunk_s)
            self.busy_s += time.perf_counter() - t0


def _run_async(n_chunks: int, tc_ms: float, tm_ms: float):
    """一次发射到底：重叠不受干扰。返回 (makespan_ms, compute_busy_ms, comm_busy_ms)。"""
    c = _ComputeWorker(n_chunks, tc_ms)
    m = _CommWorker(c, tm_ms)
    t0 = time.perf_counter()
    c.start()
    m.start()
    c.join()
    m.join()
    return (time.perf_counter() - t0) * 1000.0, c.busy_s * 1000.0, m.busy_s * 1000.0


def _run_synced(n_chunks: int, tc_ms: float, tm_ms: float, phase_len: int):
    """每 phase_len 个 chunk 调一次 stop()（同步排空），再调 start()。

    返回 (各阶段读数之和 ms, 各阶段读数列表 ms)。
    """
    readings = []
    for beg in range(0, n_chunks, phase_len):
        k = min(phase_len, n_chunks - beg)
        c = _ComputeWorker(k, tc_ms)
        m = _CommWorker(c, tm_ms)
        t0 = time.perf_counter()          # stop() 已排空 -> start() -> 读时钟
        c.start()
        m.start()
        c.join()
        m.join()
        readings.append((time.perf_counter() - t0) * 1000.0)  # stop() 排空 -> 读时钟
    return sum(readings), readings


def experiment_sync_inflation(n_chunks=8, tc_ms=10.0, repeats=3):
    """实验一：同步点数量 × 通信占比 -> 计时膨胀。"""
    tm_values = [0.0, 2.0, 5.0, 10.0]
    phase_lens = [(n_chunks, "无中间同步点（≈时间线口径）"),
                  (n_chunks // 2, "2 个阶段"),
                  (1, "每 chunk 一个阶段（L 段）")]

    rows = []
    for tm_ms in tm_values:
        best_async = min(_run_async(n_chunks, tc_ms, tm_ms)[0] for _ in range(repeats))
        row = {"tm_ms": tm_ms, "async_ms": best_async, "sync": {}}
        for plen, label in phase_lens:
            total = min(_run_synced(n_chunks, tc_ms, tm_ms, plen)[0]
                        for _ in range(repeats))
            row["sync"][plen] = (total, label)
        rows.append(row)

    print("=" * 78)
    print("实验一  同步式计时对重叠的破坏（compute chunk = %.0f ms，共 %d 个 chunk）"
          % (tc_ms, n_chunks))
    print("=" * 78)
    print("真实时间线（一次发射到底）:")
    print("  chunk 时长 -> " + "  ".join("Tm=%4.1fms" % r["tm_ms"] for r in rows))
    print("  makespan   -> " + "  ".join("%8.1f" % r["async_ms"] for r in rows))
    ovh = (rows[0]["async_ms"] - n_chunks * tc_ms) / n_chunks
    print("  （每 chunk 的线程调度开销 ≈ %.2f ms，所以绝对值比解析式系统性偏大，看趋势与比值）"
          % ovh)
    print()
    print("同步式计时读数（各阶段求和）与膨胀率:")
    for plen, label in phase_lens:
        tot = "  ".join("%8.1f" % r["sync"][plen][0] for r in rows)
        inf = "  ".join("%7.0f%%" % (100.0 * (r["sync"][plen][0] / r["async_ms"] - 1))
                        for r in rows)
        print("  [%s]" % label)
        print("     求和 " + tot)
        print("     膨胀 " + inf)
    print()

    # 逐 chunk 同步那档的解析值对照
    print("解析对照（L 段每段都同步）: 读数 = L*(Tc+Tm)，真实 = L*Tc+Tm")
    for r in rows:
        L, tc, tm = n_chunks, tc_ms, r["tm_ms"]
        print("  Tm=%4.1fms -> 解析 %6.1f / 实测 %6.1f   真实 %6.1f"
              % (tm, L * (tc + tm), r["sync"][1][0], r["async_ms"]))
    print()
    return rows


# ---------------------------------------------------------------- 实验二


def _synth_ranks(n_ranks: int, n_phases: int, base_ms: float, jitter_ms: float,
                 straggler_rank: int, straggler_factor: float, rotate_spike: bool):
    """确定性合成各 rank 各阶段耗时（不用随机数，避免复现差异）。

    rotate_spike=False：掉队卡在**每个阶段**都慢 -> 各阶段 max 落在同一张卡上
    rotate_spike=True ：短板**每阶段换一张卡** -> 各阶段 max 落在不同的卡上
    这两种形态对「逐阶段相加」的影响完全不同，是实验二要对比的重点。
    """
    table = []
    for r in range(n_ranks):
        row = []
        for p in range(n_phases):
            frac = ((r * 7919 + p * 104729) % 1000) / 1000.0
            v = base_ms + jitter_ms * frac
            if r == straggler_rank:
                v *= straggler_factor
            if rotate_spike and r == (p * 3 + 1) % n_ranks:
                v += jitter_ms * 6.0
            row.append(v)
        table.append(row)
    return table


def experiment_minmax_additivity(n_ranks=8, n_phases=6, base_ms=100.0, jitter_ms=20.0,
                                 straggler_rank=3, straggler_factor=1.6):
    """实验二：`_get_global_min_max_time()` 报的 min/max 不可加。"""
    print("=" * 78)
    print("实验二  多 rank 的 min/max 不是同一张卡，各项 max 相加没有物理含义")
    print("=" * 78)
    print("合成设定：%d rank × %d 阶段，基准 %.0f ms ± %.0f ms" %
          (n_ranks, n_phases, base_ms, jitter_ms))
    print()

    out = {}
    for rotate, title in [(False, "形态 A：掉队卡固定（rank 3 每阶段都慢 1.6×）"),
                          (True, "形态 B：短板每阶段换一张卡（+%.0f ms 轮转尖峰）" % (jitter_ms * 6))]:
        table = _synth_ranks(n_ranks, n_phases, base_ms, jitter_ms,
                             straggler_rank, straggler_factor, rotate)
        per_phase_max = [max(table[r][p] for r in range(n_ranks)) for p in range(n_phases)]
        per_phase_min = [min(table[r][p] for r in range(n_ranks)) for p in range(n_phases)]
        per_rank_total = [sum(row) for row in table]

        print("-" * 78)
        print(title)
        print("-" * 78)
        print("阶段  " + "".join("   rank%d" % r for r in range(n_ranks)))
        for p in range(n_phases):
            print("  %d   " % p + "".join("%8.1f" % table[r][p] for r in range(n_ranks)))
        print()
        print("  按 rank 求和: " + "  ".join("r%d=%.0f" % (r, per_rank_total[r])
                                              for r in range(n_ranks)))
        sum_max, sum_min = sum(per_phase_max), sum(per_phase_min)
        real_max, real_min = max(per_rank_total), min(per_rank_total)
        print()
        print("  各阶段 max 之和 = %8.1f ms   最慢卡真实值 = %8.1f ms   偏差 %+7.1f ms (%+.1f%%)"
              % (sum_max, real_max, sum_max - real_max, 100.0 * (sum_max / real_max - 1)))
        print("  各阶段 min 之和 = %8.1f ms   最快卡真实值 = %8.1f ms   偏差 %+7.1f ms (%+.1f%%)"
              % (sum_min, real_min, sum_min - real_min, 100.0 * (sum_min / real_min - 1)))
        print("  每个 max 落在哪张卡: " +
              str([max(range(n_ranks), key=lambda r: table[r][p]) for p in range(n_phases)]))
        print("  每个 min 落在哪张卡: " +
              str([min(range(n_ranks), key=lambda r: table[r][p]) for p in range(n_phases)]))
        print()
        out[rotate] = {"sum_max": sum_max, "real_max": real_max,
                       "sum_min": sum_min, "real_min": real_min}

    print("=> 形态 A（掉队卡固定）：各阶段 max 恰好都落在同一张卡上，相加碰巧等于最慢卡真实值；")
    print("   形态 B（短板轮转）：max 来自不同的卡，相加得到一个没有任何卡跑出过的数。")
    print("   所以「把各项 max 加起来」既可能碰巧对，也可能系统性错——取决于短板是否固定。")
    print()
    return out


# ---------------------------------------------------------------- 实验三


def experiment_zero_filter(n_ranks=8, pct=0.25):
    """实验三：`filter(x > 0.0)` 会静默丢掉没进过该 timer 的 rank。"""
    full = [100.0 + 6.0 * r for r in range(n_ranks)]
    skip_ranks = list(range(int(n_ranks * pct)))
    partial = [0.0 if r in skip_ranks else full[r] for r in range(n_ranks)]

    print("=" * 78)
    print("实验三  filter(x > 0.0) 的静默丢弃：min 可能来自一个只干了一点活的 rank")
    print("=" * 78)
    print("场景：基础阶段 %d 个 rank 都进了该 timer；增量阶段只有部分 rank 进入。" % n_ranks)
    print()
    print("  无过滤时按 min 选中的 rank = %d （值 %.1f）"
          % (min(range(n_ranks), key=lambda r: partial[r]),
             min(partial)))
    kept = [v for v in partial if v > 0.0]
    kept_ranks = [r for r in range(n_ranks) if partial[r] > 0.0]
    print("  过滤 0 后按 min 选中的 rank = %d （值 %.1f）"
          % (min(kept_ranks, key=lambda r: partial[r]), min(kept)))
    print()
    print("  被过滤掉的 rank: %s" % skip_ranks)
    print("  过滤后仍在统计里的 rank: %s" % kept_ranks)
    print()
    print("=> 过滤本身是合理的（没进过 timer 不该报 0），但**它不告诉你少了谁**。")
    print("   于是 min 变成「活跃子集里的最快」，与「全体的最快」不是同一个量。")
    print()


# ---------------------------------------------------------------- 实验四


def experiment_active_time(n_iter=10, warmup_iter=3, iter_ms=50.0):
    """实验四：`reset()` 不清 `_active_time` -> elapsed/active_time 是占空比。"""
    print("=" * 78)
    print("实验四  reset() 不清 _active_time：elapsed 与 active_time 的比是占空比")
    print("=" * 78)
    elapsed_each = iter_ms
    cum_elapsed = 0.0      # 每次 reset 后归零
    cum_active = 0.0       # 从创建起累计，reset 不影响
    print("  迭代   elapsed(本次)   active_time(累计)   elapsed/active_time")
    for it in range(1, n_iter + 1):
        cum_elapsed = elapsed_each
        cum_active += elapsed_each
        ratio = cum_elapsed / cum_active
        note = "   <- warmup 阶段" if it <= warmup_iter else ""
        print("   %3d      %8.1f ms       %10.1f ms          %6.3f%s"
              % (it, cum_elapsed, cum_active, ratio, note))
    print()
    print("=> 同一个 timer 对象上 elapsed 每次归零、active_time 只增不减，")
    print("   所以早期迭代的 elapsed/active_time 会系统性偏大。")
    print("   想拿占空比必须用**稳态窗口**的差值，不能拿累计量直接除。")
    print()


def main():
    ap = argparse.ArgumentParser(description="复刻同步式计时语义（纯 CPU）")
    ap.add_argument("--repeats", type=int, default=3, help="每档取几次最小值（默认 3）")
    ap.add_argument("--skip-threads", action="store_true", help="跳过多线程实验")
    args = ap.parse_args()

    if not args.skip_threads:
        experiment_sync_inflation(repeats=args.repeats)
    experiment_minmax_additivity()
    experiment_zero_filter()
    experiment_active_time()

    print("=" * 78)
    print("结论：内置 timer 报的是「排空后的时间」，不是「阶段占用的时间」。")
    print("      overlap 越好 -> 边界处在飞的工作越多 -> 读数膨胀越大。")
    print("=" * 78)


if __name__ == "__main__":
    main()
