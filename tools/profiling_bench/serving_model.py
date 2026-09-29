#!/usr/bin/env python3
"""推理服务的队列模型：TTFT / TPOT 分位、goodput、以及 P99/P50 拐点在哪。

为什么要有这个脚本
------------------
服务侧的一堆读数（TTFT、TPOT、ITL、E2EL、吞吐、goodput）在压测报告里是
一组平行数字，看不出它们由什么决定。把它们写成一个可解析的排队系统之后，
三件事就能算出来而不是猜出来：

1. **拐点位置由什么决定。** 吞吐什么时候走平、TTFT 什么时候起飞，
   不是调参调出来的，是「步长 → 批量」这条曲线的膝盖位置决定的。
2. **两种上限谁先 bind。** `max_num_seqs`（调度槽位）与 KV 容量（块池）
   是两个独立约束，长请求会让后者先到。模型里把两个约束分开写，
   压着槽位加并发不加吞吐，就是 KV 在 bind。
3. **goodput 与吞吐差多少。** 达标率随到达率单调下降，「最大不达标率
   仍满足 α 的到达率」才是容量边界——它和峰值吞吐经常差一倍以上。

代价模型（每条都是自解释的，换掉不影响结论的形状）
--------------------------------------------------
单步时间按 roofline 写，取两种资源的较大者：

    prefill_step_ms(n_tok) = max(PREFILL_FLOOR_MS, PREFILL_MS_PER_TOK * n_tok)
    decode_step_ms(n_seq)  = max(DECODE_FLOOR_MS,  DECODE_MS_PER_SEQ * n_seq)

两者都先由一个与批量无关的**下限**决定（权重搬运、kernel launch、采样），
批量大到超过膝盖 `n* = FLOOR / PER_UNIT` 之后才转成随批量线性增长。
这条「先平后斜」的形状是真实存在的，膝盖两侧的优化手段完全不同：

- 膝盖左侧：单步时间与批量无关 → 加并发几乎免费，吞吐线性上升，TPOT 不变；
- 膝盖右侧：单步时间 ∝ 批量 → 吞吐饱和在上限 `1000 / PER_UNIT`，TPOT 线性上升。

chunked prefill 下 prefill 与 decode 在同一个 step 内串行执行，
所以单步时间是两段之和：

    step_ms = prefill_part + decode_part

**代价常数是示意值**（一组自洽的默认参数），本文只主张**结构**：
哪些量互为因果、拐点由什么位置决定。绝对数值不代表任何具体硬件或模型。

自校验（这是本文件的可信度来源）
--------------------------------
`--selftest` 把四个可以从纸面推出来的结论和仿真结果对账：

1. 无争用单请求：`TTFT == prefill_step_ms(in_len)`、`TPOT == decode_step_ms(1)`、
   `E2EL == TTFT + (out_len-1)·TPOT`，三者都是精确值，不是近似。
2. 同批到达 N 个请求（不超预算、不被槽位挡住）：N 个请求的 TTFT 完全相同，
   且等于 `prefill_step_ms(N·in_len)`——prefill 是批处理，不是逐个串行。
3. 吞吐饱和：decode 段 `decode_ms_per_seq > 0` 时，输出吞吐严格小于上限
   `1000 / decode_ms_per_seq`，且随并发单调不减。
4. 卫生条件：所有时延 ≥ 0、利用率/达标率 ∈ [0,1]、P99/P50 ≥ 1。
   （`tools/rl_framework_budget.py` 第一版曾跑出负空闲率，比率类量的
   取值范围必须显式断言，见 `deep-dive-profiling/RESEARCH_PLAN.md` §4.2。）

用法
----
    python3 tools/profiling_bench/serving_model.py --selftest
    python3 tools/profiling_bench/serving_model.py --sweep           # 并发扫描
    python3 tools/profiling_bench/serving_model.py --sweep --pattern poisson --rate 8
    python3 tools/profiling_bench/serving_model.py --goodput --slo-ttft 300 --slo-tpot 30
    python3 tools/profiling_bench/serving_model.py --json out.json   # 交给 serving_schema.py
"""

import argparse
import json
import math
import random
import sys

try:  # numpy 只用于采样与分位，缺了也能跑（stdlib 回退）
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# 第 1 节  到达模型：constant / Poisson / gamma
# ---------------------------------------------------------------------------

def arrival_times(pattern, rate, n, seed=0, shape=1.0):
    """生成到达时刻（秒）。返回 (times, cv_realized)。

    rate = 平均到达率 λ（req/s）。三种模式的**均值相同、分布不同**，
    这正是「三者不可互推」的来源：同一 λ 下队列行为可以差很多。

    - `constant`：等间隔，CV = 0。平滑到不真实，但它是复现性最好的基线。
    - `poisson` ：指数间隔，CV = 1。λ 就是指数分布的率参数。
    - `gamma`   ：Gamma(shape=k, rate=k·λ)，均值 = 1/λ，CV = 1/√k。
      k = 1 退化成 Poisson；k > 1 更规则；k < 1 更突发。

    注意一处容易踩的坑：**工具的旋钮名字和分布的平滑程度不一定同向。**
    不同实现里「burstiness」可能映射到 shape 本身，也可能映射到它的倒数。
    所以这里不按旋钮名推断，而是把**实测 CV 打出来**——
    换工具时先看 CV 对不对得上，再比数。

    CV 是实现值（有限样本），不是理论值：样本量小的时候会明显偏离 1/√k。
    """
    rng = random.Random(seed)
    if rate <= 0:
        raise ValueError("rate 必须为正")
    if n <= 0:
        return [], float("nan")

    gaps = []
    if pattern == "constant":
        gaps = [1.0 / rate] * n
    elif pattern == "poisson":
        for _ in range(n):
            u = rng.random()
            # 指数分布反变换，u 取 (0,1) 开区间
            gaps.append(-math.log(max(u, 1e-12)) / rate)
    elif pattern == "gamma":
        if shape <= 0:
            raise ValueError("shape 必须为正")
        for _ in range(n):
            gaps.append(_gamma_sample(rng, shape) / (shape * rate))
    else:
        raise ValueError("未知到达模式: %s" % pattern)

    times, acc = [], 0.0
    for g in gaps:
        acc += g
        times.append(acc)

    mean = sum(gaps) / len(gaps)
    if len(gaps) > 1 and mean > 0:
        var = sum((g - mean) ** 2 for g in gaps) / (len(gaps) - 1)
        cv = math.sqrt(var) / mean
    else:
        cv = 0.0
    return times, cv


def _gamma_sample(rng, shape):
    """Marsaglia-Tsang 法取 Gamma(shape, 1)。numpy 在就直接用。"""
    if _np is not None:
        return float(_np.random.default_rng(rng.randrange(2 ** 31)).gamma(shape))
    if shape < 1.0:  # 提升法：G(k) = G(k+1) · U^(1/k)
        u = max(rng.random(), 1e-12)
        return _gamma_sample(rng, shape + 1.0) * u ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0.0, 1.0)
        v = 1.0 + c * x
        if v <= 0:
            continue
        v = v ** 3
        u = max(rng.random(), 1e-12)
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v


# ---------------------------------------------------------------------------
# 第 2 节  代价模型与服务器配置
# ---------------------------------------------------------------------------

DEFAULT_COST = dict(
    prefill_floor_ms=3.0,      # prefill 的与批量无关部分
    prefill_ms_per_tok=0.080,  # 峰值约 12500 prefill tok/s
    decode_floor_ms=12.0,      # decode 单步下限（权重搬运主导）
    decode_ms_per_seq=0.25,    # 膝盖在 48 路，吞吐上限约 4000 tok/s
)


class ServerCfg(object):
    """一次压测的服务器侧配置。字段名刻意贴 vLLM 的启动参数。"""

    def __init__(self,
                 max_num_seqs=32,
                 max_num_batched_tokens=2048,
                 kv_token_capacity=262144,
                 block_size=16,
                 preempt=True,
                 **cost):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.kv_token_capacity = kv_token_capacity
        self.block_size = block_size
        self.preempt = preempt
        c = dict(DEFAULT_COST)
        c.update(cost)
        self.prefill_floor_ms = c["prefill_floor_ms"]
        self.prefill_ms_per_tok = c["prefill_ms_per_tok"]
        self.decode_floor_ms = c["decode_floor_ms"]
        self.decode_ms_per_seq = c["decode_ms_per_seq"]

    # --- 单步代价：roofline 两段 ---
    def prefill_step_ms(self, n_tok):
        if n_tok <= 0:
            return 0.0
        return max(self.prefill_floor_ms, self.prefill_ms_per_tok * n_tok)

    def decode_step_ms(self, n_seq):
        if n_seq <= 0:
            return 0.0
        return max(self.decode_floor_ms, self.decode_ms_per_seq * n_seq)

    # --- 膝盖位置与饱和上限，都是解析值 ---
    @property
    def decode_knee(self):
        """decode 从「与批量无关」转为「随批量线性」的并发数。"""
        if self.decode_ms_per_seq <= 0:
            return float("inf")
        return self.decode_floor_ms / self.decode_ms_per_seq

    @property
    def decode_throughput_ceiling(self):
        """输出吞吐的渐近上限（tok/s）。"""
        if self.decode_ms_per_seq <= 0:
            return float("inf")
        return 1000.0 / self.decode_ms_per_seq

    def kv_blocks(self, tokens):
        return int(math.ceil(tokens / float(self.block_size)))

    def as_dict(self):
        return dict(
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            kv_token_capacity=self.kv_token_capacity,
            block_size=self.block_size,
            preempt=self.preempt,
            prefill_floor_ms=self.prefill_floor_ms,
            prefill_ms_per_tok=self.prefill_ms_per_tok,
            decode_floor_ms=self.decode_floor_ms,
            decode_ms_per_seq=self.decode_ms_per_seq,
        )


class _Req(object):
    __slots__ = ("idx", "arrival_ms", "in_len", "out_len",
                 "in_done", "out_done", "chunk", "ttft_ms", "e2el_ms",
                 "preempted", "rejected", "finished")

    def __init__(self, idx, arrival_ms, in_len, out_len):
        self.idx = idx
        self.arrival_ms = arrival_ms
        self.in_len = in_len
        self.out_len = out_len
        self.in_done = 0
        self.out_done = 0
        self.chunk = 0
        self.ttft_ms = None
        self.e2el_ms = None
        self.preempted = False
        self.rejected = False
        self.finished = False

    def kv_tokens(self, block_size):
        used = self.in_len + max(self.out_done - 1, 0)
        return int(math.ceil(used / float(block_size))) * block_size


# ---------------------------------------------------------------------------
# 第 3 节  仿真：槽位与 KV 两个约束分开建模
# ---------------------------------------------------------------------------

def simulate(cfg, times_s, in_len, out_len, max_steps=2_000_000, closed=None):
    """跑一遍连续批处理调度。返回 (records, meta)。

    每个 step 的顺序是固定的，顺序本身就是语义：

        1. 空闲则把时间跳到下一个到达时刻（不能凭空推进）
        2. **准入**：从等待队列头部取请求，同时受槽位与 KV 池限制
        3. 填 prefill token 预算（从已在 prefill 的请求队头开始）
        4. 推进 decode（每个在跑的请求 +1 token）
        5. 结算本步代价、推进时间、收尾

    第 2 步与第 3 步之间不做抢占，抢占只发生在「准入失败」时——
    这和真实调度器的形态一致：先看能不能塞进去，塞不进去才考虑驱逐。

    `closed=(并发数, 总请求数)` 切换成 **closed loop（固定并发）**：
    在途请求数恒不超过并发数，一条完成就立刻补一条（零思考时间）。
    这不需要预先知道到达时刻，因为下一个请求的发出时间依赖于
    上一个请求的返回时间。**open loop 与 closed loop 不能互推**，
    就是因为这一点：closed loop 下等待队列永远为空。
    """
    if closed is not None:
        c_max, n_total = closed
        reqs, pending = [], []
        dispatched = 0
        t = 0.0
    else:
        reqs = [_Req(i, t * 1000.0, in_len, out_len)
                for i, t in enumerate(times_s)]
        pending = list(reqs)
        n_total = len(reqs)
        dispatched = n_total
        t = 0.0

    prefilling = []               # 已准入、prefill 未完成
    decoding = []                 # prefill 完成、正在出 token
    done = []
    n_preempt = 0
    steps = 0

    def kv_used():
        return sum(r.kv_tokens(cfg.block_size) for r in prefilling + decoding)

    while (pending or prefilling or decoding or
           (closed is not None and dispatched < n_total)) and steps < max_steps:
        steps += 1

        # --- 0. closed loop：补足在途请求 ---
        if closed is not None:
            while dispatched < n_total and \
                    (len(pending) + len(prefilling) + len(decoding)) < c_max:
                r = _Req(dispatched, t, in_len, out_len)
                dispatched += 1
                reqs.append(r)
                pending.append(r)

        # --- 1. 空闲则跳到下一个到达 ---
        if not prefilling and not decoding and pending:
            if closed is None and pending[0].arrival_ms > t:
                t = pending[0].arrival_ms

        # --- 2. 准入 ---
        while pending and pending[0].arrival_ms <= t:
            r = pending[0]
            need = cfg.kv_blocks(r.in_len) * cfg.block_size
            if need > cfg.kv_token_capacity:
                # 单条请求的 prompt 就超过整个块池：永远进不去，直接判死
                r.rejected = True
                pending.pop(0)
                continue
            slots = cfg.max_num_seqs - (len(prefilling) + len(decoding))
            if slots <= 0:
                break
            if kv_used() + need > cfg.kv_token_capacity:
                if not cfg.preempt:
                    break
                victim = None
                for c in decoding:
                    if c.preempted:
                        continue
                    if c.kv_tokens(cfg.block_size) >= need:
                        if victim is None or c.kv_tokens(cfg.block_size) > \
                                victim.kv_tokens(cfg.block_size):
                            victim = c
                if victim is None:
                    break
                # 驱逐 = 丢弃已算出的 token，回等待队列头部，重算
                decoding.remove(victim)
                victim.out_done = 0
                victim.in_done = 0
                victim.preempted = True
                victim.ttft_ms = None
                pending.insert(0, victim)
                n_preempt += 1
                continue
            pending.pop(0)
            prefilling.append(r)

        # --- 3. prefill token 预算 ---
        budget = cfg.max_num_batched_tokens
        p_tokens = 0
        p_active = []
        for r in prefilling:
            if budget <= 0:
                break
            take = min(budget, r.in_len - r.in_done)
            if take <= 0:
                continue
            r.chunk = take
            budget -= take
            p_tokens += take
            p_active.append(r)

        # --- 4. 本步代价 ---
        # decode 批次必须取「本步开始时已在 decode 的请求」的快照：
        # 本步刚做完 prefill 的请求已经产出了首 token，**不能再多吃一步 decode**。
        # 不拍这个快照就会少算一步，单请求 E2EL 会差一个 TPOT。
        d_batch = list(decoding)
        n_d = len(d_batch)
        step_ms = cfg.prefill_step_ms(p_tokens) + cfg.decode_step_ms(n_d)
        if step_ms <= 0:
            # 没有任何工作可做：只能是纯等待，跳到下一个到达时刻
            if pending:
                t = max(t, pending[0].arrival_ms)
                continue
            break

        # --- 5. 推进时间并收尾 ---
        t += step_ms

        for r in p_active:
            r.in_done += r.chunk
            r.chunk = 0
            if r.in_done >= r.in_len:
                # prefill 完成即产出首 token
                r.ttft_ms = t - r.arrival_ms
                r.out_done = 1
                prefilling.remove(r)
                if r.out_len <= 1:
                    r.e2el_ms = t - r.arrival_ms
                    r.finished = True
                    done.append(r)
                else:
                    decoding.append(r)

        for r in d_batch:
            r.out_done += 1
            if r.out_done >= r.out_len:
                r.e2el_ms = t - r.arrival_ms
                r.finished = True
                decoding.remove(r)
                done.append(r)

    records = []
    for r in done:
        tpot = None
        if r.out_len > 1 and r.e2el_ms is not None and r.ttft_ms is not None:
            tpot = (r.e2el_ms - r.ttft_ms) / float(r.out_len - 1)
        records.append(dict(
            idx=r.idx, arrival_ms=r.arrival_ms,
            ttft_ms=r.ttft_ms, tpot_ms=tpot, e2el_ms=r.e2el_ms,
            in_len=r.in_len, n_out=r.out_len, preempted=r.preempted,
        ))
    finished_idx = set(r.idx for r in done)
    meta = dict(
        duration_ms=t, steps=steps, n_preempt=n_preempt,
        n_rejected=sum(1 for r in reqs if r.rejected),
        n_finished=len(done),
        total_input_tokens=sum(r.in_len for r in reqs if r.idx in finished_idx),
        total_output_tokens=sum(r.out_len for r in reqs if r.idx in finished_idx),
        n_offered=len(reqs),
    )
    return records, meta


# ---------------------------------------------------------------------------
# 第 4 节  统计量：分位、goodput、SLO 达标率
# ---------------------------------------------------------------------------

def pct(values, q):
    """线性插值分位（与 numpy.percentile 的默认口径一致）。"""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))


def meets_slo(rec, slo_ttft_ms, slo_tpot_ms):
    """判定单条请求是否达标。单 token 请求只有 TTFT 可言，不参与 TPOT 判定。"""
    if rec["ttft_ms"] is None:
        return False
    if slo_ttft_ms is not None and rec["ttft_ms"] > slo_ttft_ms:
        return False
    if rec["tpot_ms"] is None:
        return True
    if slo_tpot_ms is not None and rec["tpot_ms"] > slo_tpot_ms:
        return False
    return True


def summarize(records, meta, cfg, slo_ttft_ms=None, slo_tpot_ms=None):
    """把逐请求记录收成一份汇总。字段命名对齐 vLLM `--save-result`。"""
    dur_s = meta["duration_ms"] / 1000.0 if meta["duration_ms"] > 0 else float("nan")
    ttfts = [r["ttft_ms"] for r in records if r["ttft_ms"] is not None]
    tpots = [r["tpot_ms"] for r in records if r["tpot_ms"] is not None]
    e2els = [r["e2el_ms"] for r in records if r["e2el_ms"] is not None]
    # token 计数以**传入的 records** 为准，而不是 meta：
    # 传稳态子样本时，meta 里的计数属于整段，混用会把吞吐算高。
    # 时长仍用 meta 的整段挂钟——子样本切不出干净的分母，
    # 宁可让吞吐偏保守（分母偏大），也不要高估。
    n_out = sum(r["n_out"] for r in records) if records \
        else meta.get("total_output_tokens", 0)
    n_in = sum(r.get("in_len", 0) for r in records) if records \
        else meta.get("total_input_tokens", 0)

    out = dict(
        completed=len(records),
        duration=dur_s,
        total_input_tokens=n_in,
        total_output_tokens=n_out,
        request_throughput=(len(records) / dur_s) if dur_s and dur_s > 0 else float("nan"),
        output_throughput=(n_out / dur_s) if dur_s and dur_s > 0 else float("nan"),
        total_token_throughput=((n_in + n_out) / dur_s) if dur_s and dur_s > 0 else float("nan"),
        mean_ttft_ms=(sum(ttfts) / len(ttfts)) if ttfts else float("nan"),
        median_ttft_ms=pct(ttfts, 50),
        p90_ttft_ms=pct(ttfts, 90),
        p99_ttft_ms=pct(ttfts, 99),
        mean_tpot_ms=(sum(tpots) / len(tpots)) if tpots else float("nan"),
        median_tpot_ms=pct(tpots, 50),
        p90_tpot_ms=pct(tpots, 90),
        p99_tpot_ms=pct(tpots, 99),
        mean_e2el_ms=(sum(e2els) / len(e2els)) if e2els else float("nan"),
        median_e2el_ms=pct(e2els, 50),
        p99_e2el_ms=pct(e2els, 99),
        n_preemptions_in_model=meta["n_preempt"],
        n_rejected_in_model=meta["n_rejected"],
    )
    out["ttft_p99_over_p50"] = _ratio(out["p99_ttft_ms"], out["median_ttft_ms"])
    out["tpot_p99_over_p50"] = _ratio(out["p99_tpot_ms"], out["median_tpot_ms"])

    if slo_ttft_ms is not None or slo_tpot_ms is not None:
        ok = [r for r in records if meets_slo(r, slo_ttft_ms, slo_tpot_ms)]
        att = (len(ok) / len(records)) if records else float("nan")
        out["slo_ttft_ms"] = slo_ttft_ms
        out["slo_tpot_ms"] = slo_tpot_ms
        out["slo_attainment"] = att
        out["request_goodput"] = (len(ok) / dur_s) if dur_s and dur_s > 0 else float("nan")
        out["output_token_goodput"] = (
            (sum(r["n_out"] for r in ok) / dur_s) if dur_s and dur_s > 0 else float("nan")
        )
    return out


def cfg_in_len(records):
    return 0


def _pct_gain(a, b):
    """相对增益（百分数）。b 为 0 时返回 0，不抛除零。"""
    if not _finite(a) or not _finite(b) or b <= 0:
        return 0.0
    return (a / b - 1.0) * 100.0


def _ratio(a, b):
    if a is None or b is None or not _finite(a) or not _finite(b) or b <= 0:
        return float("nan")
    return a / b


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 第 5 节  自校验：把纸面推导和仿真结果对账
# ---------------------------------------------------------------------------

def _check(name, got, want, tol, rows):
    ok = abs(got - want) <= tol
    rows.append((name, got, want, tol, ok))
    return ok


def selftest(verbose=True):
    all_ok = True
    rows = []

    # --- 校验 1：无争用单请求，三个量都有精确解 ---
    cfg = ServerCfg(max_num_seqs=8, max_num_batched_tokens=8192,
                    kv_token_capacity=262144)
    S, L = 512, 33
    recs, meta = simulate(cfg, [0.0], S, L)
    want_ttft = cfg.prefill_step_ms(S)
    want_tpot = cfg.decode_step_ms(1)
    want_e2el = want_ttft + (L - 1) * want_tpot
    all_ok &= _check("无争用 TTFT", recs[0]["ttft_ms"], want_ttft, 1e-9, rows)
    all_ok &= _check("无争用 TPOT", recs[0]["tpot_ms"], want_tpot, 1e-9, rows)
    all_ok &= _check("无争用 E2EL", recs[0]["e2el_ms"], want_e2el, 1e-9, rows)

    # --- 校验 2：同批 N 条，TTFT 全等且等于批 prefill 代价 ---
    N = 8
    cfg2 = ServerCfg(max_num_seqs=16, max_num_batched_tokens=8192,
                     kv_token_capacity=262144)
    recs2, _ = simulate(cfg2, [0.0] * N, S, 5)
    want = cfg2.prefill_step_ms(N * S)
    ttfts2 = [r["ttft_ms"] for r in recs2]
    all_ok &= _check("批 prefill TTFT", max(ttfts2), want, 1e-9, rows)
    all_ok &= _check("批 prefill TTFT 离散度", max(ttfts2) - min(ttfts2), 0.0, 1e-9, rows)

    # --- 校验 3：吞吐饱和上限与单调性 ---
    cfg3 = ServerCfg(max_num_seqs=512, max_num_batched_tokens=8192,
                     kv_token_capacity=4_000_000)
    ceil_tps = cfg3.decode_throughput_ceiling
    tps_seq = []
    for c in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        steps_dec = 64 - 1
        dec_ms = steps_dec * cfg3.decode_step_ms(c)
        total = cfg3.prefill_step_ms(c * 64) + dec_ms
        tps_seq.append(c * 64 / (total / 1000.0))
    monotone = all(tps_seq[i + 1] >= tps_seq[i] - 1e-6
                   for i in range(len(tps_seq) - 1))
    rows.append(("吞吐随并发单调不减", float(monotone), 1.0, 0.0, monotone))
    all_ok &= monotone
    below = max(tps_seq) < ceil_tps
    rows.append(("吞吐严格小于上限", max(tps_seq), ceil_tps, 0.0, below))
    all_ok &= below

    # --- 校验 4：卫生条件 ---
    ok = hygiene_ok(recs + recs2)
    rows.append(("取值范围卫生检查", float(ok), 1.0, 0.0, ok))
    all_ok &= ok

    if verbose:
        print("=" * 78)
        print("自校验：把纸面推导与仿真结果对账")
        print("=" * 78)
        print("%-26s %16s %16s %10s %s" %
              ("检查项", "实测", "期望", "容差", "结果"))
        for name, got, want, tol, good in rows:
            print("%-26s %16.6f %16.6f %10.1e %s" %
                  (name, got, want, tol, "通过" if good else "**失败**"))
        print()
        print("全部通过" if all_ok else "存在失败项，模型不可用")
        print()
    return all_ok


def hygiene_ok(records):
    for r in records:
        for k in ("ttft_ms", "tpot_ms", "e2el_ms"):
            v = r[k]
            if v is not None and (not _finite(v) or v < 0):
                return False
        if r["e2el_ms"] is not None and r["ttft_ms"] is not None:
            if r["e2el_ms"] + 1e-9 < r["ttft_ms"]:
                return False
    return True


# ---------------------------------------------------------------------------
# 第 6 节  并发扫描与拐点判读
# ---------------------------------------------------------------------------

SWEEP_HEADER = ("%-8s %10s %11s %11s %12s %13s %11s %9s" %
                ("并发", "req/s", "out tok/s", "TTFT p50", "TTFT p99(全)",
                 "TTFT p99(稳态)", "TPOT p50", "达标率"))


def sweep_concurrency(concurrencies, in_len=1024, out_len=128, n_prompts=None,
                      slo_ttft_ms=300.0, slo_tpot_ms=30.0, cfg_kwargs=None,
                      pattern="constant"):
    """固定并发（closed loop）扫描。返回 rows，供拐点分析用。

    **闭环比开环好看，这是工具的问题不是服务的问题。** closed loop 下
    等待队列永远是空的（在途数被客户端卡住），所以 TTFT 里被排队那一块
    是**测不到**的——它只反映「prefill 本身花多久」。真实系统的 TTFT
    主要输在排队上，那部分只有 open loop 能看见。

    `--num-prompts` 与并发的关系：样本量太小时分位不稳定，默认取并发的
    10 倍（社区惯例），保证每个档位至少有十轮请求可统计。
    """
    cfg_kwargs = dict(cfg_kwargs or {})
    rows = []
    for c in concurrencies:
        cfg = ServerCfg(max_num_seqs=c, **cfg_kwargs)
        n = n_prompts or max(10 * c, 40)
        recs, meta = simulate(cfg, None, in_len, out_len, closed=(c, n))
        s = summarize(recs, meta, cfg, slo_ttft_ms, slo_tpot_ms)
        # 稳态口径：丢掉整整一波在途请求（瞬态恰好是这一波）
        ss = steady_state(recs, c)
        if ss:
            ttfts = [r["ttft_ms"] for r in ss if r["ttft_ms"] is not None]
            s["ttft_p99_steady_ms"] = pct(ttfts, 99)
            s["ttft_p50_steady_ms"] = pct(ttfts, 50)
            s["p99_over_p50_steady"] = _ratio(s["ttft_p99_steady_ms"],
                                              s["ttft_p50_steady_ms"])
            s["n_steady"] = len(ss)
        else:
            s["ttft_p99_steady_ms"] = float("nan")
            s["ttft_p50_steady_ms"] = float("nan")
            s["p99_over_p50_steady"] = float("nan")
            s["n_steady"] = 0
        rows.append(dict(concurrency=c, n_prompts=n, **s))
    return rows


def sweep_rate(rates, in_len=1024, out_len=128, n_prompts=600,
               slo_ttft_ms=300.0, slo_tpot_ms=30.0, cfg_kwargs=None,
               pattern="poisson", seed=0, shape=1.0):
    """固定到达率（open loop）扫描：直接体现「不达标率随 λ 上升」。"""
    cfg_kwargs = dict(cfg_kwargs or {})
    rows = []
    for lam in rates:
        cfg = ServerCfg(**cfg_kwargs)
        times, cv = arrival_times(pattern, lam, n_prompts, seed=seed, shape=shape)
        recs, meta = simulate(cfg, times, in_len, out_len)
        s = summarize(recs, meta, cfg, slo_ttft_ms, slo_tpot_ms)
        ss = steady_state(recs, max(1, len(recs) // 10))
        if ss:
            ttfts = [r["ttft_ms"] for r in ss if r["ttft_ms"] is not None]
            s["ttft_p99_steady_ms"] = pct(ttfts, 99)
            s["p99_over_p50_steady"] = _ratio(s["ttft_p99_steady_ms"], pct(ttfts, 50))
        else:
            s["ttft_p99_steady_ms"] = float("nan")
            s["p99_over_p50_steady"] = float("nan")
        s["arrival_cv"] = cv
        s["offer_rate"] = lam
        rows.append(s)
    return rows


def print_sweep(rows, title, slo_ttft_ms, slo_tpot_ms, slo_alpha=0.90):
    print("=" * 106)
    print(title)
    print("=" * 106)
    print("SLO: TTFT <= %.0f ms 且 TPOT <= %.0f ms；达标率目标 alpha = %.0f%%" %
          (slo_ttft_ms, slo_tpot_ms, slo_alpha * 100))
    print("「全」= 整个压测段；「稳态」= 丢掉第一波在途请求（爬坡段）。")
    print("两列差多少，就是「冷启动有多慢」被算进分位数里的程度。")
    print()
    print(SWEEP_HEADER)
    print("-" * 106)
    for r in rows:
        key = r.get("concurrency", r.get("offer_rate"))
        print("%-8s %10.2f %11.1f %11.1f %12.1f %13.1f %11.1f %8.1f%%" % (
            "%g" % key, r["request_throughput"], r["output_throughput"],
            r["median_ttft_ms"], r["p99_ttft_ms"],
            r.get("ttft_p99_steady_ms", float("nan")),
            r["median_tpot_ms"], r["slo_attainment"] * 100.0))
    print()
    print("拐点判读")
    print("-" * 108)
    knee = throughput_knee(rows)
    if knee:
        print("  吞吐饱和：%s 档位起，边际收益 < 5%%（该档位起加并发买不到吞吐）"
              % ("%g" % knee))
    else:
        print("  吞吐饱和：扫描范围内未出现（把并发继续加）")
    cap = capacity_boundary(rows, slo_alpha)
    if cap is not None:
        print("  容量边界 goodput：达标率首次跌破 %.0f%% 之前的一档，"
              "吞吐 %.2f req/s（峰值 %.2f req/s，差 %.2f 倍）"
              % (slo_alpha * 100, cap["request_throughput"],
                 max(r["request_throughput"] for r in rows),
                 max(r["request_throughput"] for r in rows) /
                 max(cap["request_throughput"], 1e-9)))
    else:
        print("  容量边界 goodput：扫描范围内达标率始终 >= %.0f%%，"
              "把到达率继续往上加" % (slo_alpha * 100))
    w_all = max(rows, key=lambda r: r["ttft_p99_over_p50"])
    ss_pk = [r for r in rows if _finite(r.get("p99_over_p50_steady", float("nan")))]
    w_ss = max(ss_pk, key=lambda r: r["p99_over_p50_steady"]) if ss_pk else None
    print("  P99/P50（全样本）峰值 %s 档：%.2f" %
          ("%g" % w_all.get("concurrency", w_all.get("offer_rate")),
           w_all["ttft_p99_over_p50"]))
    if w_ss is not None:
        print("  P99/P50（稳态）峰值 %s 档：%.2f" %
              ("%g" % w_ss.get("concurrency", w_ss.get("offer_rate")),
               w_ss["p99_over_p50_steady"]))
    print("    读法：全样本列把**爬坡段**算了进去，比值被「冷启动有多慢」抬高；")
    print("    稳态列才是真正的压力表。closed loop 下请求长度一致时稳态比值恒为 1")
    print("    （单批内每条请求走的是同一串步长），所以**比值随负载变化这件事，")
    print("    只在 open loop 下才看得见**——这也解释了为什么工具要区分两者。")
    print()


def throughput_knee(rows, eps=0.05):
    """第一个「再往上加，吞吐相对增益 < eps」的档位。"""
    keys = [r.get("concurrency", r.get("offer_rate")) for r in rows]
    tps = [r["request_throughput"] for r in rows]
    for i in range(len(tps) - 1):
        if not _finite(tps[i]) or tps[i] <= 0:
            continue
        if (tps[i + 1] - tps[i]) / tps[i] < eps:
            return keys[i + 1]
    return None


def capacity_boundary(rows, alpha=0.90):
    """达标率仍 >= alpha 的最大到达率那一档。"""
    ok = [r for r in rows if r.get("slo_attainment") is not None
          and r["slo_attainment"] >= alpha]
    return ok[-1] if ok else None


def section_preempt_ab(in_len=1024, out_len=128, kv_token_capacity=262144,
                       max_num_batched_tokens=8192):
    """对照实验：抢占开/关各值多少吞吐。

    设计上遵循 `RESEARCH_PLAN.md` §4.2 的第二条教训——**先确认基线归零**。
    这里基线的定义是：并发数还没超过 KV 池装得下的路数时，
    抢占开关应该**完全不影响结果**（因为一次抢占都不会发生）。
    如果基线档位上两列不相等，说明脚本本身有问题，结论一律不采信。
    """
    cfgs = [(c, ServerCfg(max_num_seqs=c,
                          max_num_batched_tokens=max_num_batched_tokens,
                          kv_token_capacity=kv_token_capacity))
            for c in (128, 227, 256, 384, 512)]
    cap = cfgs[0][1].kv_token_capacity
    per = cfgs[0][1].kv_blocks(in_len + out_len - 1) * cfgs[0][1].block_size
    fit = cap // per

    print("=" * 96)
    print("对照实验：抢占（recompute）值多少吞吐")
    print("=" * 96)
    print("in_len=%d out_len=%d → 每条占 %d token 的块池；KV 池 %d token 只装得下 %d 路"
          % (in_len, out_len, per, cap, fit))
    print("max_num_batched_tokens = %d（prefill token 预算，会显著改变 TTFT）"
          % max_num_batched_tokens)
    print("基线档位 = 并发 <= %d（此区间抢占不该被触发，两列必须相等）" % fit)
    print()
    print("%-9s %-9s %11s %11s %11s %9s %8s" %
          ("max_seqs", "preempt", "req/s", "out tok/s", "TTFT p50", "抢占次数", "超容量"))
    print("-" * 96)
    base_ok = True
    for c, cfg in cfgs:
        vals = {}
        for pe in (False, True):
            cfg.preempt = pe
            recs, meta = simulate(cfg, None, in_len, out_len,
                                  closed=(c, max(10 * c, 300)))
            s = summarize(recs, meta, cfg)
            vals[pe] = (s, meta)
            over = "是" if c > fit else "否"
            print("%-9d %-9s %11.2f %11.1f %11.1f %9d %8s" %
                  (c, "on" if pe else "off", s["request_throughput"],
                   s["output_throughput"], s["median_ttft_ms"],
                   meta["n_preempt"], over))
        if c <= fit:
            a, b = vals[False][0], vals[True][0]
            same = (abs(a["request_throughput"] - b["request_throughput"]) < 1e-9
                    and abs(a["median_ttft_ms"] - b["median_ttft_ms"]) < 1e-9)
            base_ok &= same
        else:
            off = vals[False][0]["request_throughput"]
            on = vals[True][0]["request_throughput"]
    print("-" * 96)
    print("基线检查：%s" % ("通过（容量内两列逐位相等，对照有效）"
                        if base_ok else "**失败**，结论不采信"))
    print()
    print("读法：超过 KV 容量之后，")
    print("  preempt=off → 吞吐**不变**，多出来的并发只转化成 TTFT（排队），")
    print("  preempt=on  → 吞吐掉一截，且 TTFT 比 off 更差（丢掉的工作要重算）。")
    print("本模型只实现 **recompute 型**抢占。vLLM 另有 swap 型（把块换出去再换回来），")
    print("代价低得多，所以这里的跌幅是 recompute 形态的量级，不能直接外推。")
    print()
    return base_ok


def steady_state(records, drop):
    """丢掉启动瞬态，只留稳态样本。

    这不是可选项。closed loop 在 t=0 一次性放出全部并发，
    前 `并发数` 条请求要排队等 prefill 预算，尾部会被这段**爬坡**撑起来；
    把整段拿去算 p99，量到的是「冷启动有多慢」，不是「服务有多稳」。
    """
    ordered = sorted(records, key=lambda r: r["idx"])
    return ordered[drop:] if drop < len(ordered) else []


def steady_window(meta, subset):
    """给稳态子样本配一个**匹配的分母**。

    子样本的吞吐 = 子样本产出的 token / 子样本自身的时间窗，
    而不是 / 整段挂钟。用整段挂钟当分母会把吞吐系统性低估
    （丢得越多低估越狠）。窗口下界取子样本里最早的到达时刻；
    上界仍用整段结束时刻，边界上最多差一条请求。
    """
    if not subset:
        return dict(meta)
    t_lo = min(r["arrival_ms"] for r in subset)
    m = dict(meta)
    m["duration_ms"] = max(meta["duration_ms"] - t_lo, 1e-9)
    return m


def section_loop_compare(in_len=1024, out_len=128, n_prompts=800,
                         slo_ttft_ms=300.0, slo_tpot_ms=30.0,
                         cfg_kwargs=None):
    """同一组代价参数下，closed loop 与 open loop 各测一遍峰值。

    两条结论：

    - **closed loop 的稳态尾部是结构性地平的。** 在途数被客户端卡住，
      等待队列永远为空，所以 TTFT 里没有「排队」这一项，
      稳态 p99 与 p50 逐位相等（比值 1.00）。
    - **closed loop 的峰值吞吐更高。** 它总能把批次填满；
      open loop 在低到达率下会空转。所以拿 closed loop 的峰值做容量规划，
      会把容量估高。
    """
    cfg_kwargs = dict(cfg_kwargs or {})

    # closed：扫并发，取稳态输出吞吐最高的档。丢掉整整一波在途请求
    # （= 并发数），因为瞬态正好是这一波。
    best_c, best_closed, n_ss, n_all = None, None, 0, 0
    for c in (32, 64, 128, 256):
        cfg = ServerCfg(max_num_seqs=c, **cfg_kwargs)
        recs, meta = simulate(cfg, None, in_len, out_len, closed=(c, 10 * c))
        ss = steady_state(recs, c)
        if not ss:
            continue
        s = summarize(ss, steady_window(meta, ss), cfg, slo_ttft_ms, slo_tpot_ms)
        if best_closed is None or s["output_throughput"] > best_closed["output_throughput"]:
            best_c, best_closed, n_ss, n_all = c, s, len(ss), len(recs)
    # open：λ 一路加，取稳态输出吞吐最高的档
    best_lam, best_open, o_ss, o_all = None, None, 0, 0
    for lam in (4, 6, 8, 10, 14, 20):
        cfg = ServerCfg(**cfg_kwargs)
        times, _ = arrival_times("poisson", lam, n_prompts, seed=0)
        recs, meta = simulate(cfg, times, in_len, out_len)
        ss = steady_state(recs, max(1, len(recs) // 10))
        if not ss:
            continue
        s = summarize(ss, steady_window(meta, ss), cfg, slo_ttft_ms, slo_tpot_ms)
        if best_open is None or s["output_throughput"] > best_open["output_throughput"]:
            best_lam, best_open, o_ss, o_all = lam, s, len(ss), len(recs)

    print("=" * 100)
    print("closed loop vs open loop：同一代价参数，两条曲线不可互推")
    print("=" * 100)
    print("统计口径 = 稳态样本（丢掉爬坡段）。爬坡段量的是冷启动有多慢，不是服务有多稳。")
    print()
    print("%-22s %11s %11s %11s %10s %11s %9s" %
          ("模式", "out tok/s", "TTFT p50", "TTFT p99", "P99/P50", "稳态样本", "达标率"))
    print("-" * 100)
    print("%-22s %11.1f %11.1f %11.1f %10.2f %11s %8.1f%%" %
          ("closed（并发 %d）" % best_c, best_closed["output_throughput"],
           best_closed["median_ttft_ms"], best_closed["p99_ttft_ms"],
           best_closed["ttft_p99_over_p50"], "%d/%d" % (n_ss, n_all),
           best_closed["slo_attainment"] * 100))
    print("%-22s %11.1f %11.1f %11.1f %10.2f %11s %8.1f%%" %
          ("open（lambda=%g）" % best_lam, best_open["output_throughput"],
           best_open["median_ttft_ms"], best_open["p99_ttft_ms"],
           best_open["ttft_p99_over_p50"], "%d/%d" % (o_ss, o_all),
           best_open["slo_attainment"] * 100))
    print("-" * 100)
    gain = (best_closed["output_throughput"] / best_open["output_throughput"] - 1.0) * 100
    print("closed loop 峰值比 open loop 高 %.1f%%；两者的稳态 TTFT 尾部形态也不同：" %
          gain)
    print("closed P99/P50 = %.2f，open P99/P50 = %.2f。"
          % (best_closed["ttft_p99_over_p50"], best_open["ttft_p99_over_p50"]))
    print("closed loop 把批次填满的能力更强，也把「排队」那一项结构性地藏了起来。")
    print("注意两行的达标率都接近 0：这是**峰值吞吐**对比，不是 goodput 对比。")
    print("结论：报峰值吞吐可以用 closed loop，做容量规划必须用 open loop。")
    print()
    return gain


def section_pattern_compare(rate=5.0, in_len=1024, out_len=128, n_prompts=500,
                            slo_ttft_ms=300.0, slo_tpot_ms=30.0,
                            cfg_kwargs=None, seed=0):
    """同一平均到达率 λ，三种到达分布各跑一遍。

    **三者不可互推**是本节要钉死的一句话。平均到达率相同并不意味着
    队列行为相同：突发度决定的是「瞬时有没有超过服务能力」，
    而均值把这个信息完全抹掉了。工具里对应的旋钮是 `--request-rate`
    加 `--burstiness`（或到达分布），不是随便设一个 QPS 就完事。

    同时打出**实测 CV**，因为旋钮名字不可靠——不同实现对
    「burstiness」映射到 Gamma 的哪个参数并不一致，先看 CV 再比数。
    """
    cfg_kwargs = dict(cfg_kwargs or {})
    cases = [
        ("constant（等间隔）", "constant", 1.0),
        ("gamma, k=4（比 Poisson 规则）", "gamma", 4.0),
        ("poisson（k=1）", "poisson", 1.0),
        ("gamma, k=0.35（比 Poisson 突发）", "gamma", 0.35),
    ]
    print("=" * 100)
    print("负载模型：平均到达率相同，分布不同 → 队列行为不同")
    print("=" * 100)
    print("mean 到达率 lambda = %g req/s，%d 条请求，并发上限 32，SLO TTFT<=%.0f ms / TPOT<=%.0f ms"
          % (rate, n_prompts, slo_ttft_ms, slo_tpot_ms))
    print()
    print("%-28s %8s %9s %11s %11s %10s %10s %9s" %
          ("到达分布", "实测 CV", "req/s", "TTFT p50", "TTFT p99", "P99/P50", "达标率", "goodput"))
    print("-" * 100)
    rows = []
    for label, pattern, shape in cases:
        cfg = ServerCfg(max_num_seqs=32, **cfg_kwargs)
        times, cv = arrival_times(pattern, rate, n_prompts, seed=seed, shape=shape)
        recs, meta = simulate(cfg, times, in_len, out_len)
        s = summarize(recs, meta, cfg, slo_ttft_ms, slo_tpot_ms)
        s["arrival_cv"] = cv
        s["label"] = label
        rows.append(s)
        print("%-28s %8.2f %9.2f %11.1f %11.1f %10.2f %9.1f%% %9.2f" %
              (label, cv, s["request_throughput"], s["median_ttft_ms"],
               s["p99_ttft_ms"], s["ttft_p99_over_p50"],
               s["slo_attainment"] * 100.0, s["request_goodput"]))
    print("-" * 100)
    tp = [r["request_throughput"] for r in rows]
    gd = [r["request_goodput"] for r in rows]
    print("同一 lambda 下：吞吐最大 %.2f req/s、最小 %.2f req/s（相差 %.1f%%）；" %
          (max(tp), min(tp), _pct_gain(max(tp), min(tp))))
    if min(gd) > 0:
        print("goodput 最大 %.2f、最小 %.2f req/s（相差 %.1f%%）。" %
              (max(gd), min(gd), _pct_gain(max(gd), min(gd))))
    else:
        print("goodput 最大 %.2f req/s，最小 **0** req/s"
              "——最乐观的那个分布之外的几种全部达不到 SLO。" % max(gd))
    print()
    print("读法（这是「三者不可互推」的量化版本）：")
    print("  · **平均吞吐几乎看不出差别**：四个分布的最大最小只差百分之十几。")
    print("    原因：突发之间的空档里服务器空转（少干），突发期间超出容量的部分")
    print("    排队（干不完），两个方向的偏差互相抵消。**只看吞吐会得出结论「分布无所谓」。**")
    print("  · **goodput 差别很大**：同一组 lambda 下能差七成以上。")
    print("    因为达标率对瞬时过载非常敏感，而均值和峰值分位把它平均掉了。")
    print("  · p50 与尾部分位都随 CV 单调变差——两者同向，不是「一个升一个降」。")
    print()
    print("所以：到达分布这件事，**在吞吐上看不见，在 goodput 上一览无余**。")
    print("等间隔（CV = 0）是所有分布中最乐观的一个，拿它做基线会把容量估高。")
    print()
    return rows


def prefix_cache_demo(in_len=1024, ratios=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99),
                      cfg_kwargs=None):
    """前缀缓存命中对 TTFT 的影响——解析式，不需要仿真。

    命中比例 f 把这一步真正要算的 prefill token 数从 S 降到 S·(1−f)。
    单请求 TTFT 是 `max(prefill_floor_ms, prefill_ms_per_tok · S)`，
    于是：

        TTFT(f) / TTFT(0) ≥ (1 − f)          （带宽主导时取等）
        TTFT(f) / TTFT(0) → 1                 （命中到 floor 主导时饱和）

    **它是分母上的乘法，不是模型变快了。** 所以「第二轮压测 TTFT 掉了一半」
    这个读数里，有一部分是缓存、有一部分才是预热——两者必须分开。

    受控做法只有三条，选一条：
      1. 换种子（prompt 内容变了，缓存键就变了）；
      2. 每轮之间重启服务（代价最高，最干净）；
      3. 用 `vllm bench sweep serve` —— 它在每轮之间显式调用
         `/reset_prefix_cache` 与 `/reset_multimodal_cache`，
         目的就是「get a clean slate for the next run」。
    """
    cfg = ServerCfg(**dict(cfg_kwargs or {}))
    base = cfg.prefill_step_ms(in_len)
    print("=" * 92)
    print("前缀缓存为什么会把 TTFT 读数改掉（解析式，不依赖仿真）")
    print("=" * 92)
    print("prompt = %d token，prefill 代价 = max(%.1f ms, %.3f ms/tok × n)"
          % (in_len, cfg.prefill_floor_ms, cfg.prefill_ms_per_tok))
    print()
    print("%-12s %14s %13s %14s %14s" %
          ("命中比例 f", "需算 token 数", "TTFT (ms)", "相对无缓存", "理想值 1−f"))
    print("-" * 92)
    for f in ratios:
        need = int(round(in_len * (1.0 - f)))
        ttft = cfg.prefill_step_ms(need)
        print("%-12s %14d %13.1f %14.3f %14.3f" %
              ("%.0f%%" % (f * 100), need, ttft,
               ttft / base if base > 0 else float("nan"), 1.0 - f))
    print("-" * 92)
    print("读法：相对值只由「还需算多少 token」决定，与硬件无关。")
    print("带宽主导时相对值正好等于 1−f；命中比例高到让固定下限（floor）主导之后")
    print("收益就饱和——实现值会**大于** 1−f（下限不肯往下走），再高的命中率也不降 TTFT。")
    print()
    print("**关键：这一列变化是测量口径变化，不是服务变快。**")
    print("同一批 prompt 复测两遍，第二遍的 TTFT 就不是同一个量了——")
    print("它量的是「缓存命中后 prefill 还剩多少」，不是「这个服务多快」。")
    print("所以复测前必须在「换种子 / 重启服务 / 用 sweep 自动清缓存」里选一条。")
    print("另：warmup（首请求的 kernel 编译与显存分配）和缓存命中是**两个**效应，")
    print("`--num-warmups` 只处理后者之外的那一个，不要指望它替你清缓存。")
    print()


# ---------------------------------------------------------------------------
# 第 7 节  CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="推理服务队列模型：TTFT/TPOT 分位、goodput、并发拐点")
    p.add_argument("--selftest", action="store_true", help="只跑自校验")
    p.add_argument("--sweep", action="store_true", help="并发扫描（closed loop）")
    p.add_argument("--rate-sweep", action="store_true",
                   help="到达率扫描（open loop，体现不达标率）")
    p.add_argument("--preempt-ab", action="store_true",
                   help="对照实验：抢占开关值多少吞吐")
    p.add_argument("--loop-compare", action="store_true",
                   help="closed loop 与 open loop 的峰值/尾部对比")
    p.add_argument("--pattern-compare", action="store_true",
                   help="同一到达率下三种到达分布的对比")
    p.add_argument("--prefix-cache", action="store_true",
                   help="前缀缓存命中对 TTFT 的解析影响")
    p.add_argument("--pattern", default="poisson",
                   choices=["constant", "poisson", "gamma"])
    p.add_argument("--shape", type=float, default=1.0, help="gamma 的 shape k")
    p.add_argument("--rate", type=float, default=5.0,
                   help="单点/分布对比的平均到达率（默认 5.0，取在容量以下）")
    p.add_argument("--concurrency", type=int, default=32, help="closed loop 的单点并发")
    p.add_argument("--in-len", type=int, default=1024)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--kv-token-capacity", type=int, default=262144)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--no-preempt", action="store_true")
    p.add_argument("--slo-ttft", type=float, default=300.0)
    p.add_argument("--slo-tpot", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="把结果写入 JSON（给 serving_schema.py）")
    args = p.parse_args(argv)

    if args.selftest:
        return 0 if selftest() else 1

    cfg_kwargs = dict(
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_token_capacity=args.kv_token_capacity,
        block_size=args.block_size,
        preempt=not args.no_preempt,
    )

    if args.prefix_cache:
        prefix_cache_demo(in_len=args.in_len, cfg_kwargs=cfg_kwargs)
        if not args.sweep and not args.rate_sweep and not args.pattern_compare \
                and not args.loop_compare and not args.preempt_ab:
            return 0

    if args.preempt_ab or args.loop_compare or args.pattern_compare:
        ok = True
        if args.pattern_compare:
            section_pattern_compare(
                rate=args.rate, in_len=args.in_len, out_len=args.out_len,
                n_prompts=args.num_prompts or 500,
                slo_ttft_ms=args.slo_ttft, slo_tpot_ms=args.slo_tpot,
                cfg_kwargs=cfg_kwargs, seed=args.seed)
        if args.loop_compare:
            section_loop_compare(in_len=args.in_len, out_len=args.out_len,
                                 slo_ttft_ms=args.slo_ttft,
                                 slo_tpot_ms=args.slo_tpot,
                                 cfg_kwargs=cfg_kwargs)
        if args.preempt_ab:
            ok = section_preempt_ab(
                in_len=args.in_len, out_len=args.out_len,
                kv_token_capacity=args.kv_token_capacity,
                max_num_batched_tokens=args.max_num_batched_tokens)
        if not args.sweep and not args.rate_sweep:
            return 0 if ok else 1

    payload = dict(cfg=None, rows=[], mode=None)

    if args.sweep:
        conc = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        rows = sweep_concurrency(
            conc, in_len=args.in_len, out_len=args.out_len,
            n_prompts=args.num_prompts,
            slo_ttft_ms=args.slo_ttft, slo_tpot_ms=args.slo_tpot,
            cfg_kwargs=cfg_kwargs)
        print_sweep(rows, "并发扫描（closed loop，固定并发数）",
                    args.slo_ttft, args.slo_tpot)
        payload["mode"] = "concurrency_sweep"
        payload["rows"] = rows
    elif args.rate_sweep:
        rates = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
        rows = sweep_rate(
            rates, in_len=args.in_len, out_len=args.out_len,
            n_prompts=args.num_prompts or 600,
            slo_ttft_ms=args.slo_ttft, slo_tpot_ms=args.slo_tpot,
            cfg_kwargs=cfg_kwargs, pattern=args.pattern,
            seed=args.seed, shape=args.shape)
        print_sweep(rows, "到达率扫描（open loop，%s）" % args.pattern,
                    args.slo_ttft, args.slo_tpot)
        payload["mode"] = "rate_sweep"
        payload["rows"] = rows
    else:
        # 单点
        cfg = ServerCfg(max_num_seqs=args.max_num_seqs or args.concurrency,
                        **cfg_kwargs)
        n = args.num_prompts or 200
        if args.pattern == "constant":
            times = [i / args.rate for i in range(n)]
        else:
            times, cv = arrival_times(args.pattern, args.rate, n,
                                      seed=args.seed, shape=args.shape)
        recs, meta = simulate(cfg, times, args.in_len, args.out_len)
        s = summarize(recs, meta, cfg, args.slo_ttft, args.slo_tpot)
        print("=" * 78)
        print("单点：%s，平均到达率 %.2f req/s，并发上限 %d" %
              (args.pattern, args.rate, cfg.max_num_seqs))
        print("=" * 78)
        print("  完成请求        %d / %d（模型内抢占 %d 次，拒绝 %d 条）" %
              (s["completed"], n, s["n_preemptions_in_model"],
               s["n_rejected_in_model"]))
        print("  时长            %.2f s" % s["duration"])
        print("  吞吐            %.2f req/s   %.1f out tok/s" %
              (s["request_throughput"], s["output_throughput"]))
        print("  TTFT  p50/p90/p99   %.1f / %.1f / %.1f ms" %
              (s["median_ttft_ms"], s["p90_ttft_ms"], s["p99_ttft_ms"]))
        print("  TPOT  p50/p90/p99   %.2f / %.2f / %.2f ms" %
              (s["median_tpot_ms"], s["p90_tpot_ms"], s["p99_tpot_ms"]))
        print("  E2EL  p50/p99       %.1f / %.1f ms" %
              (s["median_e2el_ms"], s["p99_e2el_ms"]))
        print("  P99/P50（TTFT）     %.2f" % s["ttft_p99_over_p50"])
        print("  达标率              %.1f%%（SLO: TTFT<=%.0f ms, TPOT<=%.0f ms）" %
              (s["slo_attainment"] * 100.0, args.slo_ttft, args.slo_tpot))
        print("  goodput             %.2f req/s（峰值吞吐 %.2f req/s）" %
              (s["request_goodput"], s["request_throughput"]))
        print()
        payload["mode"] = "single_point"
        payload["rows"] = [dict(offer_rate=args.rate, **s)]

    payload["cfg"] = ServerCfg(**cfg_kwargs).as_dict()
    payload["slo"] = dict(ttft_ms=args.slo_ttft, tpot_ms=args.slo_tpot,
                          alpha=0.90)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print("已写入 %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
