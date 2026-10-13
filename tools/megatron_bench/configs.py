"""模型与硬件的规格预设。

所有数值都是公开可查的标称值，带出处。单位一律在数据注释里写死，不做隐式换算。

模型参数量若与公开 implementation-records 有出入，以本文件注释里的来源为准。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelSpec:
    """一个 dense Transformer（GPT 系）的规格。"""

    name: str
    hidden: int          # h
    n_layers: int        # L
    n_heads: int         # attention heads
    ffn_hidden: int      # I, SwiGLU 的中间宽度（已经是乘过 ffn_mult 的值）
    vocab: int
    seq_len: int
    n_kv_heads: Optional[int] = None   # None 表示 MHA；否则是 GQA 的 kv 组数
    gated_mlp: bool = True             # True=SwiGLU(gate+up+down 三个矩阵)；False=GELU(up+down 两个)
    source: str = ""

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.hidden % self.n_heads == 0, "hidden 必须能被 head 数整除"

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def n_params(self, untie_embeddings: bool = False) -> int:
        """非 embedding 参数量 + embedding。用于和官方公布的参数量对账。"""
        attention = self.hidden * self.hidden          # W_q
        attention += 2 * self.hidden * self.kv_dim     # W_k, W_v
        attention += self.hidden * self.hidden         # W_o
        if self.gated_mlp:
            mlp = 2 * self.hidden * self.ffn_hidden    # gate + up
            mlp += self.ffn_hidden * self.hidden       # down
        else:
            mlp = 2 * self.hidden * self.ffn_hidden    # up + down
        per_layer = attention + mlp
        embed = self.vocab * self.hidden
        total = self.n_layers * per_layer + embed
        if not untie_embeddings:
            total += embed                             # output layer 独立
        return total

    def n_params_no_embedding(self) -> int:
        attention = 2 * self.hidden * self.hidden + 2 * self.hidden * self.kv_dim
        mlp = self.hidden * self.ffn_hidden * (3 if self.gated_mlp else 2)
        return self.n_layers * (attention + mlp)


MODELS: Dict[str, ModelSpec] = {
    # Llama-2 7B 结构：hidden 4096 / 32 层 / MHA / SwiGLU 中间 11008
    "llama-7b": ModelSpec(
        name="llama-7b", hidden=4096, n_layers=32, n_heads=32,
        ffn_hidden=11008, vocab=32000, seq_len=4096,
        source="Llama-2-7B config (HuggingFace meta-llama/Llama-2-7b-hf)",
    ),
    # Llama-3 8B 系：改了 GQA（8 个 kv head）+ 更大词表
    "llama3-8b": ModelSpec(
        name="llama3-8b", hidden=4096, n_layers=32, n_heads=32, n_kv_heads=8,
        ffn_hidden=14336, vocab=128256, seq_len=8192,
        source="Llama-3-8B config (meta-llama/Meta-Llama-3-8B)",
    ),
    # Qwen2.5-14B 系 calcifications（含 GQA）
    "qwen-14b": ModelSpec(
        name="qwen-14b", hidden=5120, n_layers=48, n_heads=40, n_kv_heads=8,
        ffn_hidden=13824, vocab=152064, seq_len=8192,
        source="Qwen2.5-14B config (Qwen/Qwen2.5-14B)",
    ),
    # GPT-3 175B：Megatron-LM examples/pretrain_gpt3_175B.sh
    "gpt3-175b": ModelSpec(
        name="gpt3-175b", hidden=12288, n_layers=96, n_heads=96,
        ffn_hidden=49152, vocab=50257, seq_len=2048, gated_mlp=False,
        source="Megatron-LM examples/gpt3/train_gpt3_175b_distributed.sh",
    ),
}


@dataclass
class HardwareSpec:
    """一张卡的标称 specs。带宽统一用单向 GB/s，避免 bidir 口径混淆。"""

    name: str
    bf16_tflops: float      # dense BF16 算力峰值 (TFLOP/s)
    hbm_gbs: float          # HBM 带宽 (GB/s)
    mem_gb: float           # 单卡显存 (GiB)
    nvlink_gbs: float       # NVLink 单向聚合带宽 (GB/s)
    ib_gbs: float           # 节点间网络单向带宽 (GB/s)
    gpus_per_node: int
    source: str = ""


HARDWARE: Dict[str, HardwareSpec] = {
    "a100-80g": HardwareSpec(
        name="a100-80g", bf16_tflops=312.0, hbm_gbs=2039.0, mem_gb=80.0,
        nvlink_gbs=300.0, ib_gbs=25.0, gpus_per_node=8,
        source="NVIDIA A100 datasheet (BF16 dense 312 TFLOP/s, HBM2e 2.0 TB/s)",
    ),
    "h100-80g": HardwareSpec(
        name="h100-80g", bf16_tflops=989.0, hbm_gbs=3350.0, mem_gb=80.0,
        nvlink_gbs=450.0, ib_gbs=50.0, gpus_per_node=8,
        source="NVIDIA H100 SXM datasheet (BF16 dense 989 TFLOP/s, HBM3 3.35 TB/s)",
    ),
    "b200-180g": HardwareSpec(
        name="b200-180g", bf16_tflops=2250.0, hbm_gbs=8000.0, mem_gb=180.0,
        nvlink_gbs=900.0, ib_gbs=100.0, gpus_per_node=8,
        source="NVIDIA B200 datasheet (BF16 dense 2.25 PFLOP/s, HBM3e 8 TB/s)",
    ),
}


@dataclass
class BenchConfig:
    """一次求解的完整输入。"""

    model: str = "llama-7b"
    hardware: str = "h100-80g"
    tp: int = 1
    cp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1
    num_nodes: int = 1
    micro_batch: int = 4
    grad_accum: int = 8
    seq_len: Optional[int] = None
    recompute: str = "none"          # none | full | selective
    use_distributed_optimizer: bool = True
    bytes_per_param_elem: int = 2    # bf16 = 2, fp32 = 4
    extra_notes: list = field(default_factory=list)

    def resolved_seq_len(self) -> int:
        return self.seq_len or MODELS[self.model].seq_len

    @property
    def world_size(self) -> int:
        """总卡数 = 各并行维乘积。dp 一般是被最后算出来的那一维。"""
        return self.tp * self.cp * self.pp * self.ep * self.dp
