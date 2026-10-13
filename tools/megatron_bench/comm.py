"""集合通信的 α-β 代价模型（纯解析）。

约定：带宽用 **单向链路带宽**；时间公式采用 nccl-tests 的 bus-bandwidth 口径，
并同时给出传统 link-bandwidth 口径的结果，避免读者对不上号。

All-Reduce (ring, n ranks, 数据量 D):
    每 rank 发送/接收 2*(n-1)/n * D
    t ≈ 2*(n-1)*α + 2*(n-1)/n * D / BW
Reduce-Scatter / All-Gather 为 AR 的一半。
All-to-All 每个 rank 给其它每个 (1/n) 份，单 rank 收发量为 (n-1)/n * D。
"""

from dataclasses import dataclass
from typing import Dict, List

from .configs import BenchConfig, MODELS, HARDWARE


def collective_bytes(kind: str, d: float, n: int) -> float:
    """单卡链路上的数据搬运字节数。d 是集合要处理的总数据量。"""
    if n <= 1:
        return 0.0
    kinds = {
        "all_reduce": 2.0 * (n - 1) / n * d,
        "reduce_scatter": (n - 1) / n * d,
        "all_gather": (n - 1) / n * d,
        "all_to_all": (n - 1) / n * d,
        "reduce": (n - 1) / n * d,
        "broadcast": d,
    }
    if kind not in kinds:
        raise ValueError("unknown collective: %s" % kind)
    return kinds[kind]


def collective_time_s(kind: str, d: float, n: int, bw_gbs: float,
                      alpha_us: float = 2.0) -> float:
    """一次集合通信的估计时长。n<=1 直接返回 0。"""
    if n <= 1:
        return 0.0
    steps = {
        "all_reduce": 2 * (n - 1),
        "reduce_scatter": n - 1,
        "all_gather": n - 1,
        "all_to_all": n - 1,
        "reduce": n - 1,
        "broadcast": n - 1,
    }[kind]
    return steps * alpha_us * 1e-6 + collective_bytes(kind, d, n) / (bw_gbs * 1e9)


@dataclass
class CommEvent:
    group: str        # tp / pp / dp / ep
    kind: str
    size_bytes: float
    count_per_layer: int
    note: str = ""

    def total_bytes(self, n: int) -> float:
        return collective_bytes(self.kind, self.size_bytes, n) * self.count_per_layer


def tensor_parallel_events(cfg: BenchConfig, sequence_parallel: bool,
                           bytes_elem: int = 2) -> List[CommEvent]:
    """一个 transformer 层的 TP 通信事件清单。

    TP only（无 SP）:  每层 4 次 All-Reduce
        —— QKV / MLP-up (ColumnParallel) 的 backward g 各一次（共 2 次）
        —— attn-dense / MLP-down (RowParallel) 的 forward f 各一次（共 2 次）
        每次搬满一个 [s, b, h] 激活：2(n-1)/n · d。

    TP + SP: 同一个 f / g 算子改变形态，**总字节数不变**。
        - f（TP 区入口）：前向 identity → all-gather；反向 identity → reduce-scatter
        - g（TP 区出口）：前向 all-reduce → reduce-scatter；反向 identity → all-gather
        于是每层：4 次 AG + 4 次 RS，每次 (n-1)/n · d
                  合计 8·(n-1)/n·d
        而 baseline TP 的 4 次 AR = 4·2(n-1)/n·d = 8·(n-1)/n·d
        —— **完全相同**。理由就是恒等式 AR(d) = RS(d) + AG(d)。

    2026-09-29 校准（此前标注"待校准"，现已核验）：
      * 恒等式 AR = RS ∘ AG 决定总量守恒；SP 分的是**激活显存**（LN/dropout/
        residual 从 O(sbh)  replicated 变成 (s/t, b, h)，缩小 t 倍），不是通信量。
      * Korthikanti et al. 的 SP 描述见 arXiv:2205.05198；对其通信条数的独立
        复述见 arXiv:2311.02382 §3（"8 global communications per attention
        layer：4 in forward pass and 4 in backward pass"），与本函数一致。
      * 因此**不要指望开 SP 降低通信量**。SP 的实际收益是显存 + 允许关掉
        activation recompute（recompute 减少才是它提速的来源）。
    """
    m = MODELS[cfg.model]
    h = m.hidden
    T = cfg.micro_batch * (cfg.resolved_seq_len() // cfg.cp)
    d = T * h * bytes_elem

    if not sequence_parallel:
        return [
            CommEvent("tp", "all_reduce", d, 2, "两层 ColumnParallel 的 backward g"),
            CommEvent("tp", "all_reduce", d, 2, "两层 RowParallel 的 forward f"),
        ]

    if cfg.tp <= 1:
        return []
    return [
        CommEvent("tp", "all_gather", d, 4,
                  "f 前向 AG + g 反向 AG（attn 与 MLP 各一对）"),
        CommEvent("tp", "reduce_scatter", d, 4,
                  "g 前向 RS + f 反向 RS（attn 与 MLP 各一对）"),
    ]


def data_parallel_grad_bytes(cfg: BenchConfig, bytes_elem: int = 2) -> float:
    """DP 组上每个 step 要搬运的梯度字节数（单卡视角，bus 口径）。

    分布式优化器走 reduce-scatter，普通 DDP 走 all-reduce + gathering。
    """
    params_local = MODELS[cfg.model].n_params() / (cfg.tp * cfg.pp)
    d = params_local * bytes_elem
    kind = "reduce_scatter" if cfg.use_distributed_optimizer else "all_reduce"
    return collective_bytes(kind, d, cfg.dp)


def pipeline_point_to_point_bytes(cfg: BenchConfig, bytes_elem: int = 2) -> float:
    """PP 相邻 stage 之间，单个 microbatch 一次收发的数据量。"""
    T = cfg.micro_batch * (cfg.resolved_seq_len() // cfg.cp)
    return T * MODELS[cfg.model].hidden * bytes_elem


def per_rank_bandwidth(cfg: BenchConfig, group: str) -> float:
    """估计某个进程组大概率跑在什么链路上。

    判据很粗：看这个组横跨的 GPU 数是否超出单机容量。
    tp / cp 一般被刻意排在节点内，dp 最容易跨机。
    """
    hw = HARDWARE[cfg.hardware]
    # 该组一共跨越的 device 数（保守地连同 tp 一起算）
    span = {"tp": cfg.tp,
            "cp": cfg.tp * cfg.cp,
            "pp": cfg.tp * cfg.cp * cfg.pp,
            "dp": cfg.world_size,
            "ep": cfg.tp * cfg.cp * cfg.pp * cfg.ep}[group]
    return hw.nvlink_gbs if span <= hw.gpus_per_node else hw.ib_gbs


def bandwidth_report(cfg: BenchConfig) -> Dict[str, float]:
    return {g: per_rank_bandwidth(cfg, g) for g in ("tp", "pp", "dp", "ep")}
