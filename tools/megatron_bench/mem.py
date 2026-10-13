"""显存账本：权重 / 梯度 / 优化器状态 / 激活 / 其他。

无 GPU 时这是整个系列里最能算准的一块——所有项都是结构决定的静态量，
不需要实测。

激活部分采用**逐张量清单**（而不是引用某个经验公式），每一项都列出名字、
形状和字节数，方便读者自己复核。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .configs import BenchConfig, HARDWARE, MODELS

BYTES_BF16 = 2
BYTES_FP32 = 4

@dataclass
class MemRow:
    """显存账本里的一行。

    sp_reducible=True 表示这个张量沿 hidden 维分布、在 sequence parallel 下
    可以被 reduce-scatter 摊薄到 tp 份 —— 这是 SP 省显存的机制所在。
    """

    name: str
    bytes_per_rank: float
    sp_reducible: bool = False

    def gb(self) -> float:
        return self.bytes_per_rank / (1024 ** 3)


@dataclass
class MemLedger:
    weights: List[MemRow] = field(default_factory=list)
    grads: List[MemRow] = field(default_factory=list)
    optimizer: List[MemRow] = field(default_factory=list)
    activations: List[MemRow] = field(default_factory=list)

    def total_bytes(self) -> float:
        return sum(r.bytes_per_rank for grp in (
            self.weights, self.grads, self.optimizer, self.activations) for r in grp)

    def group_total(self, group: str) -> float:
        return sum(r.bytes_per_rank for r in getattr(self, group))

    def summary(self) -> List[Tuple[str, float]]:
        return [
            ("权重", self.group_total("weights")),
            ("梯度", self.group_total("grads")),
            ("优化器状态", self.group_total("optimizer")),
            ("激活", self.group_total("activations")),
        ]


def _params_per_rank(cfg: BenchConfig) -> int:
    """每个 rank 持有的参数量。TP 切权重，PP 切层。"""
    m = MODELS[cfg.model]
    return int(m.n_params() / (cfg.tp * cfg.pp))


def weight_and_grad_memory(cfg: BenchConfig) -> Tuple[List[MemRow], List[MemRow]]:
    """主权重 / 梯度。

    默认 Megatron 的混合精度训练：参数 bf16，梯度 bf16（--grad-reduce-in-bf16 时），
    另有 fp32 master 记在优化器状态里。
    """
    p = _params_per_rank(cfg)
    b = cfg.bytes_per_param_elem

    weights = [
        MemRow(f"主权重 ({cfg.bytes_per_param_elem} B/elem, 按 tp{cfg.tp} pp{cfg.pp} 切分)", p * b),
    ]
    grads = [
        MemRow("梯度 bf16", p * BYTES_BF16),
    ]
    return weights, grads


def optimizer_memory(cfg: BenchConfig,
                     exp_avg_dtype_bytes: int = BYTES_BF16,
                     exp_avg_sq_dtype_bytes: int = BYTES_BF16,
                     use_distributed_optimizer: bool = True,
                     precision_aware: bool = False) -> List[MemRow]:
    """优化器状态。

    AdamW 需要 fp32 master + exp_avg + exp_avg_sq。
    开了分布式优化器后这三个都按 DP 组切碎（这是 --use-distributed-optimizer 的核心收益）。
    """
    p = _params_per_rank(cfg)
    shard = cfg.dp if use_distributed_optimizer else 1

    # 主权重副本：普通模式保留 fp32 master；precision-aware optimizer 可以省掉
    master_bytes = 0 if precision_aware else p * BYTES_FP32
    rows = [
        MemRow(f"fp32 master (precision-aware={'省掉' if precision_aware else '保留'})",
               master_bytes / shard),
        MemRow("exp_avg", p * exp_avg_dtype_bytes / shard),
        MemRow("exp_avg_sq", p * exp_avg_sq_dtype_bytes / shard),
    ]
    return rows


def activation_inventory_per_layer(cfg: BenchConfig) -> Tuple[List[MemRow], Dict[str, float]]:
    """单层激活清单（每个 microbatch）。返回 (清单, 关键中间量)。

    约定：
      b = micro_batch, s = seq_len / cp, T = b*s 为每个 CP rank 上的 token 数
      h = hidden, I = ffn_hidden, kv_dim = 本 TP rank 上的 kv 宽度

    sp_reducible 标记出那些「沿 hidden 维分布、TP 内原本被复制」的张量，
    sequence parallel 打开的正是这部分空间。
    """
    m = MODELS[cfg.model]
    h, I = m.hidden, m.ffn_hidden
    s = cfg.resolved_seq_len() // cfg.cp
    b = cfg.micro_batch
    T = b * s
    kv_dim = m.kv_dim // cfg.tp

    th = T * h * BYTES_BF16       # 一个 [T,h] 的 bf16 张量
    rows = [
        MemRow("LN1 输入（残差分支复用）", th, True),
        MemRow("LN1 输出 / QKV 输入", th, True),
        MemRow("q", th, True),
        MemRow("k", T * kv_dim * BYTES_BF16, False),
        MemRow("v", T * kv_dim * BYTES_BF16, False),
        MemRow("attention 输出（W_o 之前）", th, True),
        MemRow("W_o 输出", th, True),
        MemRow("dropout mask", T * h * 1, True),
        MemRow("LN2 输入", th, True),
        MemRow("LN2 输出 / gate-up 输入", th, True),
        MemRow("gate 中间结果", T * I * BYTES_BF16, False),
        MemRow("up 中间结果", T * I * BYTES_BF16, False),
        MemRow("SwiGLU 输出", T * I * BYTES_BF16, False),
    ]
    extras = {"T": T, "s_local": s, "I": I, "head_dim": m.head_dim}
    return rows, extras


def stored_activation_rows(cfg: BenchConfig, recompute: str) -> List[MemRow]:
    """按 recompute 策略，返回真正会被保存到显存里的那些张量。"""
    rows, _ = activation_inventory_per_layer(cfg)

    if recompute in ("selective", "core_attn"):
        recompute = "core_attn"
    if recompute == "none":
        return rows
    if recompute == "core_attn":
        # Megatron 默认的 selective：attention core 内部的中间结果不保存
        return [r for r in rows if "attention 输出" not in r.name]
    if recompute == "full":
        # 每层只保留层输入这一份
        m = MODELS[cfg.model]
        T = cfg.micro_batch * (cfg.resolved_seq_len() // cfg.cp)
        return [MemRow("层输入（唯一保存项）", T * m.hidden * BYTES_BF16, True)]
    raise ValueError("unknown recompute: %s" % recompute)


def activation_memory(cfg: BenchConfig,
                      sequence_parallel: bool = False,
                      recompute: str = "none",
                      distribute_saved_activations: bool = False) -> List[MemRow]:
    """每个 GPU 上驻留的激活总量。

    计算顺序很重要：先按 recompute 决定「存哪些」，再按 SP 决定「每份摊薄多少」。
    早先版本把 SP 的减免加在完整清单上，recompute=full 时会算出负数 —— 已修。
    """
    m = MODELS[cfg.model]
    T = cfg.micro_batch * (cfg.resolved_seq_len() // cfg.cp)
    layers_local = m.n_layers // cfg.pp
    outstanding = cfg.pp if cfg.pp > 1 else 1

    rows = stored_activation_rows(cfg, recompute)

    per_layer = 0.0
    for r in rows:
        b = r.bytes_per_rank
        if sequence_parallel and r.sp_reducible and cfg.tp > 1:
            b /= cfg.tp
        per_layer += b

    if distribute_saved_activations and recompute != "none" and cfg.tp > 1:
        # 重算边界上的保存量再按 tp 摊一次
        per_layer /= cfg.tp

    total_layers = per_layer * layers_local
    notes = ["recompute=%s" % recompute]
    if sequence_parallel and cfg.tp > 1:
        notes.append("SP 摊薄 /tp%d" % cfg.tp)
    if distribute_saved_activations and recompute != "none" and cfg.tp > 1:
        notes.append("distribute_saved /tp%d" % cfg.tp)

    label = ("激活：%d 层 × %d 个在飞 microbatch（%s）"
             % (layers_local, outstanding, "，".join(notes)))
    return [MemRow(label, total_layers * outstanding)]


def full_ledger(cfg: BenchConfig, **kw) -> MemLedger:
    w, g = weight_and_grad_memory(cfg)
    o = optimizer_memory(cfg,
                         use_distributed_optimizer=cfg.use_distributed_optimizer,
                         precision_aware=kw.get("precision_aware", False))
    a = activation_memory(cfg,
                          sequence_parallel=kw.get("sequence_parallel", False),
                          recompute=cfg.recompute,
                          distribute_saved_activations=kw.get("distribute_saved_activations", False))
    return MemLedger(weights=w, grads=g, optimizer=o, activations=a)


def bytes_to_gib(x: float) -> float:
    return x / (1024 ** 3)
