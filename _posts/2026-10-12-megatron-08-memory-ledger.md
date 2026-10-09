---
title: "Megatron-LM 深度剖析（08）：显存账本——重计算、卸载与分布保存"
date: 2026-10-12 10:00:00 +0800
categories:
  - 分布式训练
tags: [megatron, megatron-core, activation-recompute, activation-checkpointing, selective-recompute, activation-offloading, cpu-offloading, hybrid-optimizer, cuda-graph, memory-fragmentation, distributed-training, source-code]
layout: post
mathjax: true
excerpt: "显存六项账本里，激活这一项怎么算？本文给出一套统一口径：把 recompute.py 的 189 行、transformer_config.py 的 11 个 recompute_modules、1610 行 fine-grained offload 与 477 行 hybrid_optimizer 放进同一把尺子，扫出显存-重算代价前沿，并逐条钉死 normal 与 output-discarding 的二分、DSA 与 SP 为什么互斥、以及重计算与 CUDA Graph 为什么不能同时开。"
---

> 源码基准：`megatron-core` **0.20.0**，commit [`60e039626`](https://github.com/NVIDIA/Megatron-LM/tree/60e039626)。本文所有行号均对该 commit 取证。
> 配套实验：`ipynbs` 仓库 `experiments/megatron/memory_ledger/`（三个脚本 + 固化输出）。
> <!-- TODO-ipynbs-commit: 监工 push 后把下面的 commit 号回填成本文实验目录的 permalink 前缀 -->

<!-- TODO-frontmatter: 06/07 落盘后补前置链 -->

这是模块 A 的收口。

前面七篇各自讲了通信与激活的一个局部：进程组怎么切、通信怎么与计算重叠、专家并行的 token 怎么 dispatch、张量并行与序列并行的 f / g 算子怎么实现、GTP 怎么换掉 1D TP、数据并行的梯度怎么分片、流水调度器怎么排 microbatch。每一篇都碰到「显存不够」这件事，但每篇只报了自己那一项的开销。

本文做一件前面七篇都没做的事：**把六项显存放进同一把尺子里量**。口径是上一篇（第 05 篇）已经立好的六项 —— 权重、梯度、优化器状态、激活、碎片、临时缓冲。上一篇在算这张账时把激活项显式记了 0，并写明理由：激活由 microbatch 数与重计算策略决定，与数据并行那条路无关，因此那张表不能当作训练峰值显存。

这个 0 就是本文要填的洞。

---

> **TL;DR**
>
> 1. **激活显存的总账 = 权重 + 梯度 + 优化器状态 + 激活 + 碎片 + 临时缓冲**，前两项与重计算无关，本文只处理后四项里能被结构决定的部分。
> 2. **`recompute.py` 只有 189 行、一个公共符号，而且只服务 `full`。** `selective` 的 11 个 `recompute_modules` 全部实现在各自的模块文件里。`recompute_granularity` 这个字段在仓库里被 **19 处**代码读取，`recompute.py` 一个字都没提。
> 3. **`recompute_num_layers` 在 `uniform` 下不改变显存**，只改变 chunk 边界数量 —— 因为源码的 `uniform` 分支把每一段都 checkpoint。真正能用 `k` 换显存的只有 `block`。
> 4. **11 个 `recompute_modules` 分成两类**：4 个走 normal checkpointing（`core_attn` `mlp` `moe` `shared_experts`），7 个走 output-discarding（`moe_act` `layernorm` `mla_up_proj` `gdn_norm_out` `gdp_in_proj` `gdp_qkv` `mhc`）。区别是后者连模块自己的输出一并释放，靠一次统一重算钩子恢复。
> 5. **`distribute_saved_activations` 只摊 `args[0]` 一个张量**，切完是一维的 `numel // tp_size`；它切的是扁平化后的连续块，也就是序列维 —— 与 SP 切同一个轴。所以 `transformer_config.py` **直接禁止** DSA 与 SP 同时开，实测 SP 开着时 DSA 的净收益恰好是 0.000 GiB。
> 6. **offload 是三档，不是两档**：整层 CPU offload、模块级 activation offload、optimizer offload。第一档与第二档互斥，第二档与重算之间有 4 条按数据依赖写死的互斥规则 —— 其中 `gdp_in_proj` 的重算会把 `gdp_qkv` 卸载组想留在 host 上的东西**直接丢掉**。
> 7. **`hybrid_optimizer.py` 搬的是优化器状态，不是权重。** 被卸载的参数在 GPU 上仍保留一份 bf16 计算副本，CPU 上多出的是优化器状态与可选的 fp32 主副本。分界线是「按 `param_groups` 遍历顺序、以 numel 计的贪心前缀」，与参数角色无关。
> 8. **重计算与 CUDA Graph 的关系是三条约束，不是一句「互斥」。** 捕获期与 warmup 期直接跳过 checkpointing（源码注释：*recomputation cannot run inside a captured graph*）；`full` 重计算**要求**整迭代图而非被图禁止；`selective` 的重算模块必须完全在图内或完全在图外，不许跨边界。
>
> 上面 8 条里，1 / 3 / 5 / 6 有配套脚本的实测支撑（解析模型 + CPU 微型复现），2 / 4 / 7 / 8 是源码级事实，行号逐条给出。

---

## 0. 符号与口径

先把符号定死，避免后面混用。本文的符号全部沿用来源文献或源码的既有写法，不自造。

| 符号 | 含义 | 出处 |
| --- | --- | --- |
| $$b$$ | micro batch size | 源码 config |
| $$s$$ | 单个 CP rank 上的序列长度 | 03 篇 |
| $$T = b \cdot s$$ | 单 rank 的 token 数 | 本文的记法，与 03 篇一致 |
| $$h$$ | hidden size | Llama / Qwen 结构 |
| $$I$$ | FFN 中间宽度（SwiGLU 口径） | 同上 |
| $$F$$ | 单层前向 FLOPs | 本文记法 |
| $$M$$ | 显存账本六项的合计 | 05 篇口径 |

显存一律用 GiB（$$2^{30}$$ 字节），不用 GB。FLOPs 一律用「GEMM 计 $$2MNK$$」的口径。

两个「额外代价」的量必须分清，全文不混：

- **$$\text{extra}/F$$**：额外重算 FLOPs ÷ 前向 FLOPs。全量重算所有层时等于 1.0。
- **总代价增幅**：额外重算 FLOPs ÷ $$3F$$。基线是前向 $$F$$ 加反向 $$2F$$，全量重算所有层时等于 33.33%。

之所以要分清，是因为文献里「重计算贵 30% 多」说的是后者 —— Chen 等人给出的就是 *30 percent additional runtime cost*（[arXiv:1604.06174](https://arxiv.org/abs/1604.06174)）；而配置文档里「多跑一次前向」说的是前者。本文表格两个都给。

「以计算换显存」这件事本身也不是新想法。同一篇论文把代价-显存权衡写成了显式的分段公式：把 $$n$$ 层网络切成 $$k$$ 段，代价是 $$O(n/k) + O(k)$$，取 $$k = \sqrt{n}$$ 得到 $$O(2\sqrt{n})$$ 的显存 —— 这就是本文 `recompute_method` 两种取值背后的思想，只不过 Megatron 把分段数做成了配置项而不是让它自动求解。同一篇论文也把 checkpointing 这个叫法归给了自动微分文献。

---

## 1. 显存六项账本

### 1.1 六项凭什么这么分

第 05 篇立的六项口径是：权重、梯度、优化器状态、激活、碎片、临时缓冲。这六项的分界线是**「谁负责分配、什么时候释放、能不能跨 iteration 复用」**：

| 项 | 谁分配 | 生命周期 | 能否跨 iteration 复用 |
| --- | --- | --- | --- |
| 权重 | 优化器 / 模型 | 全程 | 否（每步可能更新） |
| 梯度 | `param_and_grad_buffer` | 每个 iteration 重置 | 否 |
| 优化器状态 | 优化器 | 全程 | 否 |
| 激活 | autograd | 单次 forward→backward | 否 |
| 碎片 | allocator | — | 部分 |
| 临时缓冲 | 调用方（通信 / bucket / 卸载） | 跨 iteration | **能** |

第六项是本文要强调的：**临时缓冲是六项里唯一一项设计上就打算被复用的**。这决定了它的记账方式与前五项不同 —— 前五项按峰值算，第六项按「稳态占用」算。

### 1.2 上一篇留下的那个 0

第 05 篇的显存账本给出四行数字（7B 参数、TP=8、DP=8、bf16 权重 + fp32 优化器状态，单位 GiB）：

| 路径 | param | grad | optim | 临时桶 | fp32 暂存 | 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| DDP | 1.63 | 1.63 | 6.52 | 0.00 | 0.00 | **9.78** |
| DistOpt | 1.63 | 1.63 | 0.81 | 0.00 | 0.00 | **4.07** |
| DistOpt + overlap | 1.63 | 1.63 | 0.81 | 1.63 | 0.41 | **6.11** |
| FSDP2 / MFSDP | 1.63 | 0.20 | 0.81 | 0.00 | 0.00 | **2.65** |

这张表的分项口径来自 Megatron-LM 原文对并行训练显存构成的分解（[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)）；最后一行 FSDP2 / MFSDP 走的是另一条路 —— 参数、梯度、状态全部按 rank 分片，机制与前三行不同，PyTorch FSDP 的官方说明见 [FSDP 文档](https://docs.pytorch.org/docs/stable/fsdp.html)。

那张表里激活一栏是空的，原话是「**activation 项记 0**，它与 DP 路径无关，由 microbatch 数与重计算策略决定。因此上表**不能当作训练峰值显存**——这是账本的口径限制，不是疏漏」。

本文接的就是这句话。要把它填上，得先说清楚激活那一项由什么决定。

激活显存不是「模型多大」的函数，是「**这一层 backward 时需要回看哪些前向中间量**」的函数。把它写成账本的一项，就是把每层反向真正需要的张量逐条列出来，算字节，再乘上「有多少层同时在飞」。

### 1.3 单层保存集：逐张量清单

以一个 SwiGLU + RMSNorm 的稠密 decoder layer 为准（RMSNorm 的定义见 [PyTorch 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)，块结构本身沿用 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 的原式），TP=8 且序列并行开启，单个 rank 每个 microbatch 的保存集如下（`T` = 本 rank token 数，`kv` = 本 rank 上的 kv 宽度，$$I$$ = FFN 中间宽度）：

| 张量 | 形状 | 字节 | SP 可摊薄 |
| --- | --- | --- | --- |
| LN1 输入（残差分支复用） | $$[T, h]$$ | $$2Th$$ | 是 |
| LN1 输出 / QKV 输入 | $$[T, h]$$ | $$2Th$$ | 是 |
| q | $$[T, h]$$ | $$2Th$$ | 是 |
| k | $$[T, \mathrm{kv}]$$ | $$2T \cdot \mathrm{kv}$$ | 否 |
| v | $$[T, \mathrm{kv}]$$ | $$2T \cdot \mathrm{kv}$$ | 否 |
| core_attn 输出 / $$W_o$$ 输入 | $$[T, h]$$ | $$2Th$$ | 是 |
| $$W_o$$ 输出 | $$[T, h]$$ | $$2Th$$ | 是 |
| LN2 输入 | $$[T, h]$$ | $$2Th$$ | 是 |
| LN2 输出 / gate-up 输入 | $$[T, h]$$ | $$2Th$$ | 是 |
| gate 中间结果 | $$[T, I]$$ | $$2TI$$ | 否 |
| up 中间结果 | $$[T, I]$$ | $$2TI$$ | 否 |
| SwiGLU 输出 / down 输入 | $$[T, I]$$ | $$2TI$$ | 否 |

「SP 可摊薄」这一列是承接 03 篇的结论：序列并行只摊薄**沿 hidden 维在 TP 组内本来是复制的**那些张量，也就是 layernorm 与 dropout 之前那一段；k / v 与 FFN 中间量沿 token 维分布，SP 摊不动它们。这条边界在源码里有对应 —— 03 篇已核过 `RowParallelLinear` 里那句 `assert not self.sequence_parallel`，两者互斥是纪律，不是巧合。

这个清单是 A 级脚本 `pareto.py` 的输入。脚本对每个模块声明三件事：不被重算时保存什么、被重算时保存什么、它自己的 FLOPs 是多少；然后按配置求和。把模型换成 Llama-2 7B 结构（$$h$$ = 4096、32 层、SwiGLU 中间 11008）、序列长 4096、TP=8、micro batch 1、SP 开，基线激活是：

```text
1. 基线（不重算不卸载）激活 = 9.19 GiB。
```

这个数只声明结构，不声明性能。它的用途是和下面几节的重算、卸载数字放在同一列里比。

### 1.4 碎片：两个同名的 `GlobalMemoryBuffer`

碎片这一项最容易漏，因为它不来自任何一行显式 `torch.empty`，而来自 allocator 的分配粒度与复用策略。

Megatron 对此的正面回答是 [`GlobalMemoryBuffer`](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/utils.py#L727-L753)：通信的输入输出 buffer 不每次分配，而是从一个按 `(name, dtype)` 索引的全局池里取子张量。

```python
class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (self.buffer.get((name, dtype), None) is None
                or self.buffer[(name, dtype)].numel() < required_len):
            ...
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype,
                device=torch.cuda.current_device(), requires_grad=False)

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
```

三个要点，每一个都直接影响账本：

1. **只涨不缩。** 池子只在不够大时重新分配，够大就永远不释放。所以「峰值 = 这个池的最大需求」，而不是平均值。通信 group 的个数一旦变大，这块就是常驻开销。
2. **按 `(name, dtype)` 索引，不按形状。** 同一个名字下所有形状共用一块，取子张量用 `[0:required_len]`。于是「不同通信事件用不同 name」是必要的隔离 —— docstring 第一句就写了 *Caller should ensure that buffers of the same name are not used concurrently*。
3. **它算第六项还是第四项？** 算第六项。它不随 iteration 释放，是设计上就要复用的东西。第 01 篇讲通信时引过这个类（`GlobalMemoryBuffer` 在 forward 的 SP 分支被用到）。

还有一个容易被漏的事实：这个类在仓库里**有两份**，互不相关 ——

| 位置 | 行号 | 归属 |
| --- | --- | --- |
| `megatron/core/utils.py` | L727 | 主线，通信用 |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/utils.py` | L769 | Megatron-FSDP 自带一份 |

所以「引用 `GlobalMemoryBuffer`」这句话必须带文件与行号，不带就是错的。这也是本系列行号规范（每 150 行至少 1 个行内 permalink）的直接来源。

至于 allocator 本身的行为 —— 分块粒度、cached block 的复用、碎片化之后的 `cudaMalloc` 失败 —— 属于 PyTorch 文档范畴，见第 6 节。

### 1.5 临时缓冲：bucket padding 的尾巴

第六项在第 05 篇里已经算过一次，本文不重算，只引用结论。

对齐常数三个：`pad_param_start` 对齐到 64、`bucket_end_divisor` 取 $$\mathrm{lcm}(\text{dp}, 128)$$、开 `pad_buckets_for_high_nccl_busbw` 时升到 $$\mathrm{lcm}(\text{dp}, 128, 2^{16})$$。函数在 [`optimizer/param_layout.py`](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/param_layout.py#L19-L42)：

```python
def pad_to_divisor(value: int, divisor: int) -> int:
    """Round up ``value`` to the nearest multiple of ``divisor``."""
    return int(math.ceil(value / divisor) * divisor)

def pad_param_start(param_start_index: int) -> int:
    """Align parameter start index to a 64-element boundary."""
    return pad_to_divisor(param_start_index, 64)

def bucket_end_divisor(data_parallel_world_size: int, pad_for_high_nccl_busbw: bool) -> int:
    """Divisor used to pad bucket ends for DP-divisibility (and optional NCCL busbw)."""
    if pad_for_high_nccl_busbw:
        return math.lcm(data_parallel_world_size, 128, 2**16)
    return math.lcm(data_parallel_world_size, 128)
```

padding 只在分布式优化器那条路上存在，默认 DDP 路一个元素都不补。

对齐要求来自集合通信本身：reduce-scatter 要把缓冲区均分给各 rank，所以桶边界必须被 `dp_world_size` 整除（[NCCL 集合通信文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) 给出各原语对缓冲区尺寸的要求；通信域与分组语义见 [NCCL 用户指南概览](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)）。实测浪费率：稠密模型的参数 numel 基本都是 128 的倍数，`lcm(dp, 128)` 在 `dp \le 128` 时恒等于 128，补不出东西；真正制造浪费的是 `pad_param_start(64)` 遇到非对齐参数（MoE 单专家权重最常见），每个最多补 63 个元素。开高总线带宽对齐后 `lcm` 跃升到 65536，非对齐场景的浪费率从 0.003% 涨到 **1.869%**。

**一个诚实的缺口**：这三个常数的选型动机源码里没有写。本系列不猜。要换基准实测，只能在真机上量 bucket 起始地址的对齐分布 —— 本机没有 GPU，**待验证**。

---

## 2. 重计算：189 行薄封装背后的判断逻辑

### 2.1 第一个反差：唯一的公共符号只服务一半的取值

先把 `recompute.py` 的规模说清楚：全文 **189 行**，公共符号**只有一个** —— `checkpointed_forward`，在 [`recompute.py` L21](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/recompute.py#L21)。

然后是本篇要讲的第一个反差。这个符号只在**一个**地方被调用：

```python
# megatron/core/transformer/transformer_block.py L662-L676
if self.config.recompute_granularity == 'full' and self.training:
    checkpointed_result = checkpointed_forward(
        self,
        hidden_states=hidden_states,
        ...
    )
```

全仓库 `checkpointed_forward` 的引用只有三处：定义在 `recompute.py` L21，`transformer_block.py` L24 导入 + L663 调用，`models/hybrid/hybrid_block.py` L27 导入 + L360 调用。`hybrid_block.py` 的那一处条件相同（`hybrid_block.py` L359）：`if self.config.recompute_granularity == 'full' and self.training:`。

也就是说：**`recompute.py` 这 189 行只实现 `full` 这一种粒度。** `selective` 完全不走它。

`selective` 走的是另一条路 —— 每个模块在**自己的文件里**判断要不要重算自己。以 `core_attn` 为例，判断在构造期就做完，存成一个布尔量：

```python
# megatron/core/transformer/attention.py L392-L395
self.checkpoint_core_attention = (
    self.config.recompute_granularity == 'selective'
    and "core_attn" in self.config.recompute_modules
)
```

同样的两行在 [`transformer_block.py` L299-L302](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_block.py#L299-L302) 又出现一次。`TransformerConfig` 的 docstring 把分工写得很清楚（[`transformer_config.py` L526-L533](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L526-L533)）：

> Determines which type of activation recompute to use. Megatron-core supports 'selective' activation checkpointing where the submodules set in `--recompute-modules` is checkpointed. The default is “core_attn” which is the memory intensive part of attention. These memory intensive activations are also less compute intensive which makes activation checkpointing more efficient for LLMs (20B+). ... **'selective' always uses all layers.**

最后那句 *'selective' always uses all layers* 是配置层的硬约束的措辞版本，源码里对应 L1904-L1910 的 `raise`：selective 模式下 `recompute_num_layers` 必须为 `None`。所以 selective 没有「只重算前 k 层」这种玩法。

### 2.2 判断逻辑散在 19 处

把 `recompute_granularity` 在仓库里的读取点全列出来，会看到判断逻辑的真正形状 —— 它不是一个函数，是散布在十几个文件里的一组同形判断：

| 文件 | 行号 | 判断内容 |
| --- | --- | --- |
| `transformer/transformer_block.py` | L300 / L330 / L662 | `core_attn` 谓词、`mhc` 谓词、`full` 分派 |
| `transformer/attention.py` | L393 | `core_attn` 谓词 |
| `transformer/multi_latent_attention.py` | L172 | MLA 的 selective 分支 |
| `transformer/experimental_attention_variant/absorbed_mla.py` | L183 | 吸收型 MLA |
| `transformer/moe/moe_layer.py` | L248 / L253 | MoE 层 |
| `transformer/moe/experts.py` | L270 | 专家 |
| `transformer/moe/shared_experts.py` | L157 | 共享专家 |
| `transformer/multi_token_prediction.py` | L1532 | MTP 的 `full` 分派 |
| `models/hybrid/hybrid_block.py` | L359 | `full` 分派 |
| `ssm/gated_delta_net/common.py` | L271 | GDN |
| `ssm/gated_delta_product.py` | L271 / L276 | GatedDeltaProduct |
| `transformer/cuda_graphs.py` | L1024 | CUDA Graph 分段捕获时的重算判定 |
| `distributed/torch_fully_sharded_data_parallel.py` | L140 | FSDP 包装期 |
| `models/huggingface/fastconformer_model.py` | L49 / L62 | 语音模型 |
| `transformer/transformer_config.py` | 30 余处 | 校验与联动 |

这个散布本身是个结论：**重计算不是一个开关，而是一组每个模块各自作出的局部决定。** 加一个新的 `recompute_modules` 取值，要动的是那个模块自己的文件，而不是 `recompute.py`。

### 2.3 `full`：uniform 与 block 的真实区别

`recompute.py` 里 `recompute_method` 的两个分支在 L157-L183。逐字看：

```python
# megatron/core/recompute.py L157-L166
if self.config.recompute_method == 'uniform':
    layer_idx = 0
    while layer_idx < self.num_layers_per_pipeline_rank:
        chunk_end = min(
            layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank
        )
        chunk_runner(layer_idx, chunk_end, True)
        layer_idx += self.config.recompute_num_layers
elif self.config.recompute_method == 'block':
    recompute_skip_num_layers = 0
    for layer_idx in range(self.num_layers_per_pipeline_rank):
        if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
            recompute_skip_num_layers += 1
        use_checkpoint = (
            layer_idx >= recompute_skip_num_layers
            and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
        )
        chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
else:
    raise ValueError("Invalid activation recompute method.")
```

`chunk_runner(layer_idx, chunk_end, True)` 的第三个参数在 `uniform` 分支里是**硬编码的 `True`**。这就是全部：`uniform` 把所有层划成连续的 chunk，逐段 checkpoint，**一段都不跳过**。因此 `recompute_num_layers` 在 `uniform` 下只决定 chunk 有多少段，不决定哪些层被重算 —— 显存与 `k` 无关。

`block` 才用 `k` 控制：只有 `layer_idx < recompute_num_layers` 的层被 checkpoint，其余层整份保存激活。

实测印证这一点（`pareto.py` 输出）：

```text
2. full+uniform 的 k 取 1/2/4/8 时激活显存恒为 [0.125] GiB —— k 不改变显存。
```

对比 `block`：

```text
3. full+block k=1: 8.90 GiB, extra/F=3.1%
3. full+block k=2: 8.62 GiB, extra/F=6.2%
3. full+block k=4: 8.05 GiB, extra/F=12.5%
3. full+block k=8: 6.92 GiB, extra/F=25.0%
```

这是 `uniform` 与 `block` 的本质区别：**前者是「切段重算」，显存降到每段的层输入之和；后者是「挑层重算」，显存随 `k` 线性下降。** 想要用 `k` 换显存，只能走 `block`。

`block` 分支里还有一段只在 FP8 / FP4 下生效的逻辑（`recompute.py` L175-L180）：当 `hidden_states.requires_grad` 为 `False` 时把该层跳过并累加 `recompute_skip_num_layers`。源码注释说明了原因 —— 重入式 autograd 引擎至少需要一个带梯度的输入张量，所以这些槽位被推到重算窗口之外。

### 2.4 precision-aware：两条 checkpoint 路径

`chunk_runner` 里 `use_checkpoint` 为真时，分派到两个不同实现（[`recompute.py` L130-L144](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/recompute.py#L130-L144)）：

```python
if use_checkpoint:
    # Precision-aware activation checkpoint: TE under FP8/FP4,
    # tensor_parallel under BF16/FP16/FP32.
    if self.config.fp8 or self.config.fp4:
        hidden_states, context = te_checkpoint(
            cf, self.config.distribute_saved_activations,
            tensor_parallel.random.get_cuda_rng_tracker, self.pg_collection.tp, *args)
    else:
        hidden_states, context = tensor_parallel.checkpoint(
            cf, self.config.distribute_saved_activations, *args)
```

「precision-aware」这个词在这里有确切含义（TE 侧的量化上下文见 [Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/index.html)）：**低精度走 Transformer Engine 的 checkpoint（它知道 FP8 的 amax 记账需要什么），中高精度走 `tensor_parallel` 自己的 checkpoint。** 第 09 篇（低精度）会接着讲这条线。

注意两个调用都把 `distribute_saved_activations` 当第二个位置参数传下去 —— 这是第 3 节的入口。

### 2.5 selective 为什么默认只重算 `core_attn`

默认值在 [`transformer_config.py` L1918-L1919](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1918-L1919)：

```python
if self.recompute_modules is None:
    self.recompute_modules = ["core_attn"]
```

理由写在字段 docstring 里（`transformer_config.py` L528-L532）：`core_attn` 是注意力里最吃显存的部分，而它**重算起来又不贵** —— *These memory intensive activations are also less compute intensive which makes activation checkpointing more efficient for LLMs (20B+)*，并直接给了文献：*See Reducing Activation Recomputation in Large Transformer Models*（[arXiv:2205.05198](https://arxiv.org/abs/2205.05198)）。

那篇论文的核心论点与本节的实测一致：注意力内部那些张量（softmax 前的分数、dropout 掩码）体积大但生成代价低，所以「只重算注意力」比「重算整层」划算。Korthikanti 等人给出的结论是，选择性重算能在几乎不影响吞吐的前提下拿到接近全量重算的显存收益。

本文的 `pareto.py` 在结构上复现了这个方向性。以 `extra/F` 为代价轴：

| 配置 | 激活（GiB） | $$\text{extra}/F$$ | 总代价增幅 |
| --- | --- | --- | --- |
| 不重算 | 9.19 | 0.0% | 0.00% |
| `selective core_attn` | 9.06 | 9.7% | 3.23% |
| `selective layernorm` | 9.06 | 0.0% | 0.00% |
| `selective moe_act` | 6.50 | 0.0% | 0.00% |
| `selective mlp` | 6.50 | 78.2% | 26.06% |
| `full uniform` | 0.125 | 100.0% | 33.33% |

读这张表要说清两件事，否则容易误读：

**第一，`moe_act` 在这张表里是 0 代价换 2.69 GiB。** 这不是脚本算错了，而是本文的建模口径：output-discarding 类模块释放的是「模块输出」这条，而 `mla_up_proj` / `moe_act` 这类逐元素或窄投影算子的重算 FLOPs 相对 GEMM 可忽略（见脚本 `module_flops` 里这几项返回 0.0）。真实重算仍要占 kernel 时间，**那部分本机无法建模**。

**第二，`core_attn` 只省 0.13 GiB，是因为本节用的是 FlashAttention 路径。** 上表里 `core_attn` 省下的正是「core_attn 输出 / $$W_o$$ 输入」那一条 $$[T, h]$$。论文里说的「注意力内部张量体积大」，指的是**显式materialize 分数矩阵**的实现（$$[T, T]$$ 量级）；当注意力核用融合 kernel 时（IO 感知分块那块工作见 [FlashAttention](https://arxiv.org/abs/2205.14135)），那块本来就不占显存，选择性重算自然也省不到多少。**这一条是本篇最需要读者自己判断适用边界的地方** —— 表里的数只对「注意力核用融合 kernel」的配置成立。

### 2.6 normal 与 output-discarding 的二分

这是 `recompute_modules` 最需要讲出来的一条。11 个合法取值里，`docstring` 明写了一个二分（[`transformer_config.py` L572-L574](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L572-L574)，逐字）：

> “moe_act”, “layernorm”, “mla_up_proj”, “gdn_norm_out”, “gdp_in_proj”, “gdp_qkv”, and “mhc” use output-discarding checkpointing, “core_attn”, “mlp”, “moe”, and “shared_experts” use normal checkpointing.

| 类别 | 取值 | 个数 |
| --- | --- | --- |
| output-discarding | `moe_act` `layernorm` `mla_up_proj` `gdn_norm_out` `gdp_in_proj` `gdp_qkv` `mhc` | 7 |
| normal | `core_attn` `mlp` `moe` `shared_experts` | 4 |

两类在**代码里是两套完全不同的实现**，不只是策略不同。

**normal** 是 `CheckpointFunction`（[`tensor_parallel/random.py` L600-L672](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/tensor_parallel/random.py#L600-L672)）。「重算」的准确含义由 autograd 的 saved-tensor 语义定义 —— 前向存什么、反向取什么，见 [PyTorch autograd 文档](https://docs.pytorch.org/docs/stable/autograd.html)。它的 docstring 开头就点明来源：*This has been directly copied from torch.utils.checkpoint*（`random.py` L678-L679）。前向在 `no_grad` 下跑，L628 `ctx.save_for_backward(*args)` 存下全部输入；反向取出输入、重跑前向、再 `torch.autograd.backward`。

**output-discarding** 是 `CheckpointWithoutOutput` 系列（`random.py` L748-L1010）。它的类 docstring 把差别讲得很清楚：

> For the normal 'checkpoint` function, the outputs of it may be saved by the following modules for their backward computation. However, the output of the checkpointed function is re-generated at recomputation, so the output store is not technically needed. This method can manually discard the output in the forward pass and restore it by recomputation in the backward pass to reduce the memory usage.
>
> Due to the reason above, to save memory with this method, the caller should make sure that the discarded output tensors are directly saved in the following modules for backward computation.

翻译过来就是差别所在：normal checkpointing 释放模块**内部**的中间量，但模块**输出**还在（下游可能把它存下来了）；output-discarding 连输出一起释放 —— 因为输出在重算时会重新生成，那个存储位技术上不需要。代价是调用方必须保证被弃的输出不会被前向后续用到。

### 2.7 output-discarding 的实现：丢弃与 zero-copy 恢复

丢弃动作是 [`random.py` L985-L996](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/tensor_parallel/random.py#L985-L996)：

```python
def _discard_outputs(self):
    """Release output storage, preserving outputs that alias retained inputs."""
    if self.retain_input_tensors:
        for output in self.outputs:
            if output.untyped_storage().data_ptr() not in self._saved_input_ptrs:
                output.untyped_storage().resize_(0)
    else:
        for output in self.outputs:
            output.untyped_storage().resize_(0)
```

注意用的是 `untyped_storage().resize_(0)`（API 见 [PyTorch 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html)）而不是丢掉 Python 引用。**张量句柄与形状都还在，只有字节被释放** —— 因为下游可能已经对它 `.reshape()` 过或 `torch.split()` 过，那些 view 还指着同一块 storage，直接置空会让它们变野指针。

恢复动作在 L967-L978：

```python
# Zero-copy: make output's StorageImpl point to recomputation_output's data.
# This operates at the UntypedStorage level (below TensorImpl), so:
#   - ALL views / reshapes that reference output's StorageImpl see the data
#     (e.g. TE GroupedLinear's inp.reshape() + torch.split() saved for backward)
#   - No tensor version-counter bump (no autograd complaint)
share_storage = _get_share_storage()
for output, recomputation_output in zip(self.outputs, outputs):
    if output.untyped_storage().data_ptr() != recomputation_output.untyped_storage().data_ptr():
        share_storage(output, recomputation_output)
```

换指针这件事由一个 C++ 扩展完成（`random.py` L32 的注释：*C++ extension: zero-copy storage sharing for CheckpointWithoutOutput*）。为什么非做不可，注释给了两层理由：必须在 `UntypedStorage` 层（`TensorImpl` 之下）操作，所有引用该 `StorageImpl` 的 view 才会一起看到新数据；以及不能触发 tensor 的 version counter，否则 autograd 会报「变量在反向中被原地修改」。

这条注释提到的具体受害者就是注意力/MLP 里那些 `inp.reshape()` + `torch.split()` 后被保存的张量。

**多个 output-discarding 点如何协调**：`CheckpointWithoutOutputManager`（`random.py` L803-L845）。一个 `TransformerBlock` 里若有多个 output-discarding 点，且它们有先后依赖（一个的输出是下一个的输入），就注册到同一个 manager，然后**统一丢弃、统一重算**：

```python
def discard_all_outputs_and_register_unified_recompute(self, hook_tensor):
    """Discard all checkpoint outputs to save memory and register unified recompute hook."""
    for ckpt in self.checkpoints:
        ckpt._discard_outputs()
    if hook_tensor.requires_grad:
        hook_tensor.register_hook(self._unified_recompute_hook)

def _unified_recompute_hook(self, grad_output):
    for ckpt in self.checkpoints:
        # Call _recompute for each checkpoint in forward order
        ckpt._recompute(None)
```

只注册**一个** autograd 钩子，挂在 block 输出张量上；触发时按**前向顺序**逐个重算。这就是为什么整个 block 只需要一次钩子，而不是每个模块一个。

### 2.8 一个必须原样保留的上游怪癖

`CheckpointWithoutOutput.__init__` 的 `fp8` 参数有个看起来像 bug 的地方，源码自己写了长注释解释（[`random.py` L862-L875](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/tensor_parallel/random.py#L862-L875)）：

```python
fp8: Quantization recipe, or a bool. Note that the default `fp8=False`
    still evaluates to `self.fp8 = True`; every caller that constructs
    `CheckpointWithoutOutput()` with no arguments therefore takes the
    TE `activation_recompute_forward` path. That is long-standing
    behavior which several selective-recompute modules ("layernorm",
    "moe_act", "gdn_norm_out") depend on for correct FP8 amax
    bookkeeping, so do NOT "fix" this to `bool(fp8)` here — tightening
    it changes FP8 numerics and needs its own PR with FP8
    functional-test evidence.
```

对应的实现是 L883 的 `self.fp8 = fp8 is not None`。默认 `fp8=False` 走进来，`False is not None` 求值为 `True`。

**这是逐字忠实引用，不是错误。** 上游明确写了不要「修」它 —— `layernorm` / `moe_act` / `gdn_norm_out` 这几个 selective 模块依赖它才能把 FP8 的 amax 记账做对，改成 `bool(fp8)` 会改动 FP8 数值。

### 2.9 `mhc` 的额外约束

`mhc`（HyperConnection 的中间激活）是 output-discarding 那 7 个里约束最多的一个。配置层写了四条（[`transformer_config.py` L2005-L2029](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L2005-L2029)）：

| 约束 | 行号 | 触发条件 |
| --- | --- | --- |
| 必须开 `enable_mhc_connections` | L2007-L2008 | 否则 `ValueError` |
| 不能与 `mlp` 同用 | L2009-L2013 | 两者机制不同，会冲突 |
| `mhc_recompute_layer_num` 必须是正整数 | L2014-L2022 | 否则 `ValueError` |
| 不能与 fine-grained 卸载同开 | L2023-L2029 | `NotImplementedError` |

最后一条的理由写得非常具体，值得逐字引：

> 'mhc' in recompute_modules + fine_grained_activation_offloading is not yet supported. The mHC recompute hook currently fires before the offloading backward chunk is initialized, causing `tensor_pop` on a None chunk. Disable one of them.

这是**重计算与卸载直接打架**的一个实例，而且给出了机制层面的原因：重算钩子触发得比卸载的反向 chunk 初始化更早，于是 `tensor_pop` 拿到一个 `None`。第 4.6 节会看到这不是孤例，而是四类互斥中的一条。

另外 `enable_mhc_connections` 与 `full` 粒度也不兼容（`transformer_config.py` L2047-L2052，`NotImplementedError`）—— 即 mHC 只能配 selective。

---

## 3. `distribute_saved_activations`

### 3.1 它摊的是哪一个张量

字段定义只有一行 docstring（[`transformer_config.py` L550-L551](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L550-L551)）：

> If True, distribute recomputed activations across the model parallel group.

「across the model parallel group」这个措辞容易读成「把所有保存的激活都摊到 MP 组上」。源码不是这样。实际实现只摊**第一个**张量 —— [`random.py` L619-L625](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/tensor_parallel/random.py#L619-L625)：

```python
# Divide hidden states across model parallel group and only keep
# the chunk corresponding to the current rank.
if distribute_saved_activations:
    ctx.input_0_shape = args[0].data.shape
    safely_set_viewless_tensor_data(
        args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
    )

# Store everything.
ctx.save_for_backward(*args)
```

`args[0]`，硬编码的下标 0（字段的官方 API 页面见 [Megatron-Core API 指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/index.html)）。反向对应 L647-L650：

```python
inputs = ctx.saved_tensors
if ctx.distribute_saved_activations:
    safely_set_viewless_tensor_data(
        inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
    )
```

`inputs[0]`，同样是下标 0。`ctx.input_0_shape` 这个变量名本身就说明了意图。

所以 DSA 的收益上限是**一个张量的 $$1 - 1/\mathrm{tp}$$**，与该层保存了多少张量无关。这是理解第 3 节的关键。

### 3.2 形状变换：一维，不是 `(T, h)`

切分函数在 [`tensor_parallel/utils.py` L48-L76](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/tensor_parallel/utils.py#L48-L76)：

```python
tp_group = get_tensor_model_parallel_group_if_none(tp_group)
partition_size = torch.numel(tensor) // tp_group.size()
start_index = partition_size * tp_group.rank()
end_index = start_index + partition_size
if new_buffer:
    data = torch.empty(partition_size, dtype=tensor.dtype,
                       device=torch.cuda.current_device(), requires_grad=False)
    data.copy_(tensor.view(-1)[start_index:end_index])
else:
    data = tensor.view(-1)[start_index:end_index]
return data
```

三点：

1. **先 `view(-1)` 再切。** 返回的是一维张量，长度 `numel // tp_size`。形状不再是 `(T, h)`。
2. **`new_buffer=True` 时是拷贝，不是 view。** 调用点写死了 `new_buffer=True`（`random.py` L624），所以每次都要重新分配一块 `partition_size` 的显存并拷贝进去。
3. **分组是张量并行组**（`get_tensor_model_parallel_group_if_none`，默认取 TP 组），不是「模型并行组」这个更宽泛的概念。

聚合端 [`utils.py` L79-L93](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/tensor_parallel/utils.py#L79-L93)：

```python
numel_gathered = torch.numel(tensor) * tp_group.size()
gathered = torch.empty(numel_gathered, dtype=tensor.dtype, ...)
dist_all_gather_func(gathered, tensor, group=tp_group)
return gathered
```

长度按 `numel * tp_size` 还原，再 `.view(ctx.input_0_shape)` 恢复原形状。

`dsa_shapes.py` 逐条推出每 rank 的形状（`(4096, 4096)`、tp=8）：

```text
rank      start        end      numel         保存形状           字节
   0          0    2097152    2097152   (2097152,)      4194304
   1    2097152    4194304    2097152   (2097152,)      4194304
   2    4194304    6291456    2097152   (2097152,)      4194304
   3    6291456    8388608    2097152   (2097152,)      4194304
   4    8388608   10485760    2097152   (2097152,)      4194304
   5   10485760   12582912    2097152   (2097152,)      4194304
   6   12582912   14680064    2097152   (2097152,)      4194304
   7   14680064   16777216    2097152   (2097152,)      4194304
```

单 rank 持有量恰为原来的 $$1/\mathrm{tp}$$，往返逐位一致。

### 3.3 与 TP 的关系：切的是序列维

上面这张表里 `start` / `end` 的间隔是 2097152，正好等于 $$T \cdot h / 8$$。由于 `view(-1)` 是行主序，第 `r` 个连续块覆盖的是**第 $$8r$$ 到 $$8r+1$$ 行**，也就是一段连续的 token。

**所以 DSA 实际切的是序列维（token 维），不是 hidden 维。** 这一点是理解与 SP 关系的前提，也是最容易误解的地方 —— 从「把激活摊到 TP 组上」这句话完全推不出这一点。

代价侧也清楚：反向时多一次 all-gather，通信量与该张量同阶（集合通信与张量并行的语义边界见 [PyTorch 分布式文档](https://docs.pytorch.org/docs/stable/distributed.html)）。这就把 DSA 变成了一个**显存换通信**的开关，而 TP 本身的通信量第 03 篇已经算过。用不用它，取决于显存是不是瓶颈。

### 3.4 与 SP 的冲突：不是「语义会变」，是硬禁止

第 03 篇的结论是**序列并行不降低 TP 通信量**：AR = RS ∘ AG ⇒ 总量守恒，实测比值恒 1.00×。SP 的收益在**激活显存**。

现在把 DSA 放到这张图上：它摊的正是层输入那条张量，也就是 SP 摊的那条。两者切同一个轴。沿序列维分片以支撑长上下文的做法见 [Ultra-Long Sequence Distributed Transformer](https://arxiv.org/abs/2311.02382)。

`transformer_config.py` 的处理不是警告，是直接 `raise`（`transformer_config.py` L1912-L1916）：

```python
if self.distribute_saved_activations and self.sequence_parallel:
    raise ValueError(
        f"distribute_saved_activations: {self.distribute_saved_activations} must be "
        f"false when sequence parallel is enabled: {self.sequence_parallel}"
    )
```

注意这个校验的嵌套位置：它在 `if self.recompute_granularity is not None:`（`transformer_config.py` L1879）**之内**。也就是说，DSA 只有开了重计算才会被校验到。

结构上说明了问题：DSA 是 checkpoint 的一个可选项（它在 `CheckpointFunction` 里生效），单独开 `full` 却不校验。

`pareto.py` 的实测印证了「不是可叠加的两项」：

| SP | DSA 前 | DSA 后 | 省 |
| --- | --- | --- | --- |
| on | 0.12 GiB | 0.12 GiB | **0.000 GiB** |
| off | 1.00 GiB | 0.12 GiB | 0.875 GiB |

SP 开着时净收益**恰好是 0**，不是「很少」，是 0。因为层输入那条张量在 SP 下已经只有 $$1/\mathrm{tp}$$，DSA 再切一次切的是同一个轴上已经被切过的分片 —— 而 `split_tensor_into_1d_equal_chunks` 对它再除一次 `tp_size`，得到的分片已经属于别的 rank 了，全组对不齐。

`dsa_shapes.py` 把这个机制算到底：SP 下本 rank 的形状已是 `(512, 4096)`（`numel` = 2097152），DSA 再切一次得到 `partition_size` = 262144，占原始 `numel` 的

$$\frac{1}{\mathrm{tp}^2} = \frac{1}{64} = 0.01562500$$

也就是说，如果强行同时开，这条张量会被摊到 $$1/\mathrm{tp}^2$$，而全组合计对不齐原长度。`transformer_config.py` 的 `raise` 正是拦这个。

### 3.5 一个没有余数处理的边界

`partition_size = torch.numel(tensor) // tp_group.size()` 用的是整除向下取整。如果 `numel` 不能被 `tp_size` 整除，**余数被静默丢弃**；而聚合端按 `numel * tp_size` 还原，长度对不上，`.view(ctx.input_0_shape)` 必然抛形状不匹配。

`dsa_shapes.py` 列了几组：

```text
     numel   tp  numel//tp  tp×(numel//tp)     余数     能还原?
  16777216    8    2097152        16777216      0        是
      4095    8        511            4088      7        否
       100    8         12              96      4        否
      4096    5        819            4095      1        否
      1023    7        146            1022      1        否
        10    4          2               8      2        否
```

实际配置里 `(T, h)` 的 `numel` 一般能被 `tp` 整除（`h` 通常是 `tp` 的倍数），所以这条边界平时碰不到。**但它说明 DSA 没有任何余数处理逻辑** —— 如果哪条路径让 `args[0]` 的首维不整除，就会以形状不匹配的形式失败，而不是明确报错。

---
## 4. offload 三档

### 4.1 三档总表

把激活或状态搬到 host 是这一系列的通用手段，方法学可追到 ZeRO-Offload（[arXiv:2101.06840](https://arxiv.org/abs/2101.06840)）与 ZeRO-Infinity（[arXiv:2104.07857](https://arxiv.org/abs/2104.07857)）。Megatron 侧则有三条互相独立的卸载路径，官方文档入口见 [Megatron-Core 用户指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)。先把落点与粒度摆清楚：

| 档 | 开关 | 粒度 | 落点 | 行数 |
| --- | --- | --- | --- | --- |
| 1. 整层 CPU offload | `cpu_offloading` + `cpu_offloading_num_layers` | **层** | `transformer_block.py` L304-L318 | — |
| 2. 模块级 activation offload | `fine_grained_activation_offloading` | **模块** | `pipeline_parallel/fine_grained_activation_offload.py` | **1610** |
| 3. optimizer offload | `optimizer_cpu_offload` + `optimizer_offload_fraction` | **参数** | `optimizer/cpu_offloading/hybrid_optimizer.py` | **477** |

第 1 档与第 2 档互斥，配置层直接 `assert`（[`transformer_config.py` L2105-L2108](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L2105-L2108)）：

```python
if self.fine_grained_activation_offloading:
    assert not self.cpu_offloading, \
        "fine_grained_activation_offloading cannot be enabled with cpu_offloading."
```

两者的粒度差异在字段 docstring 里写明了（`transformer_config.py` L1309-L1311）：

> Fine-grained activation offloading is a module-level offloading method instead of a layer-level offloading method like cpu_offloading.

**「同名不同目录」本身就是一条论据。** `hybrid_optimizer.py` 在 `megatron/core/optimizer/cpu_offloading/` 下，`fine_grained_activation_offload.py` 在 `megatron/core/pipeline_parallel/` 下 —— 而后者整份文件顶部第一句注释写的是：

```python
# megatron/core/pipeline_parallel/fine_grained_activation_offload.py L11
# CPU offload implementation for pipeline parallelism
```

放在 `pipeline_parallel/` 下、写的是「for pipeline parallelism」、通篇用 `forward_chunk` / `backward_chunk` 命名，不是偶然：第 2 档在架构上是**按流水并行的 chunk 边界**组织的卸载，而不是全局的层堆叠。这也解释了它为什么能做「按 PP rank 差异化卸载」（第 4.4 节）。

### 4.2 第 1 档：整层 CPU offload 的两个硬校验

第 1 档的实现挂在 `TransformerBlock` 构造期（[`transformer_block.py` L304-L318](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_block.py#L304-L318)），来源是 `get_cpu_offload_context`。它有两条硬校验和一条 TE 门禁。

**校验一：层数范围**（[`transformer_config.py` L1862-L1867](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1862-L1867)）：

```python
if self.cpu_offloading and (
    self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
):
    raise ValueError(f"CPU offloading can be done only for layers less than {self.num_layers}")
```

`cpu_offloading_num_layers` 必须在 `[0, num_layers)` 内。

**校验二：不支持流水并行**（`transformer_config.py` L1869-L1872）：

```python
if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
    raise ValueError("Currently there is no support for Pipeline parallelism with CPU offloading")
```

这条常被忽略：**第 1 档与 PP 互斥**。而第 2 档恰恰是按 PP chunk 组织的 —— 两条路径的适用场景因此完全不同。

**校验三：不支持重计算**（`transformer_config.py` L1874-L1877）：

```python
if self.cpu_offloading and self.recompute_granularity is not None:
    raise ValueError("CPU offloading does not work when activation recomputation is enabled")
```

**重计算与整层卸载互斥。** 机制上说得通：重计算要重跑前向，那么「需要重新读到的输入」必须还在原处；一旦把层输入搬到 host，重算就得先搬回来，省下的显存被往返搬运吃掉。

**TE 门禁**（`transformer_block.py` L319-L325）：

```python
else:
    assert (
        self.config.cpu_offloading is False
    ), "CPU Offloading is enabled when TE is not present"
    self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
```

**没有 Transformer Engine 时第 1 档直接不可用。**

### 4.3 第 2 档：1610 行是怎么组织的

第 2 档的核心文件有 1610 行，但结构是清楚的五个类 + 一组 `autograd.Function`。按职责分：

| 类 | 行号 | 职责 |
| --- | --- | --- |
| `OffloadTensorPool` | L150 | host 侧张量池，按 `(shape, dtype)` 复用 |
| `OffloadTensorGroup` | L384 | 一个卸载组：push / pop、事件记录、字节统计（CUDA stream 与 event 的语义见 [PyTorch 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams)） |
| `PipelineOffloadManager` | L443 | 全局单例，管 chunk 队列与流 |
| `ChunkOffloadHandler` | L881 | 单个 chunk 的实际搬运（bulk offload / reload） |
| `FineGrainedOffloadingGroupCommitFunction` | L1357 | autograd 钩子，标记卸载组边界 |

配套的 `FineGrainedOffloadingInterface`（`fine_grained_activation_offload.py` L1496）是对外的静态入口。

#### 关键机制一：`saved_tensors_hooks` 桥

这是整套东西最值得讲的一处。它不手动搬张量，而是在 autograd 的保存/取回两个时机插钩（[`fine_grained_activation_offload.py` L863-L878](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L863-L878)）：

```python
def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
    """Hook called when autograd saves a tensor for backward pass.
    Returns a tag to identify the tensor later."""
    debug_rank(f"------on_save_for_backward {tensor.shape}")
    assert self.inside_context, "Must be inside offload context"
    return self.cur_forward_chunk().tensor_push(tensor)

def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
    """Hook called when autograd retrieves a saved tensor during backward pass."""
    debug_rank("----on_get_saved_tensor")
    return self.cur_backward_chunk().tensor_pop(saved_state)
```

保存时返回一个 **tag**（不是张量），取回时才把张量还原。**autograd 全程不知道这些张量在 host 上。** 好处是所有会 `save_for_backward` 的算子自动被覆盖，不需要逐个模块改；代价是必须靠 chunk 的进出来配对前后向，所以有 `assert self.inside_context`。

这个模式在 PyTorch 里有正式文档，见 [`torch.autograd.graph.saved_tensors_hooks`](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks)。

#### 关键机制二：非连续 view 的整块搬运

`ChunkOffloadHandler.offload` 有一处很细的优化（[L895-L934](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L895-L934)）。对一个非连续 view，如果它覆盖了底层 storage 的绝大部分，就**直接搬整个 storage，把 view 元数据一起带走**，而不是先 gather 成连续副本：

```python
view_meta = None
if not src_tensor.is_contiguous():
    storage = src_tensor.untyped_storage()
    element_size = src_tensor.element_size()
    covered_bytes = src_tensor.numel() * element_size
    if (storage.nbytes() % element_size == 0
            and covered_bytes >= self.BASE_OFFLOAD_MIN_COVERAGE * storage.nbytes()):
        view_meta = (src_tensor.size(), src_tensor.stride(), src_tensor.storage_offset())
        # Flat alias of the full storage; contiguous by construction.
        src_tensor = torch.empty(0, dtype=src_tensor.dtype,
                                 device=src_tensor.device).set_(storage)
    else:
        src_tensor = src_tensor.contiguous()
```

阈值 `BASE_OFFLOAD_MIN_COVERAGE = 0.5`（`fine_grained_activation_offload.py` L892）。省掉的是一次 gather 出来的临时连续副本 —— 对显存紧张的场景，这个临时副本本身就是峰值的一部分。

docstring 还说明了为什么要用 `untyped_storage()` 而不是 `._base`：

> The base is resolved through ``untyped_storage()`` rather than ``._base`` because autograd hands the saved-tensor hooks a detach()-ed alias (e.g. **CheckpointWithoutOutput's saved input**), which drops the view metadata but not the storage.

**这句话把第 2 档和第 2.6 节的 output-discarding 直接连起来了**：output-discarding 交给 saved-tensor hook 的输入是一个 detach 过的别名，`._base` 没了但 storage 还在。写这段代码的人必须知道 output-discarding 的存在，才能选对 `untyped_storage()`。

#### 关键机制三：跨库契约

TE 会给自己的张量打标记，`_te_do_not_offload`（`fine_grained_activation_offload.py` L36-L49）读这个标记：

```python
def _te_do_not_offload(tensor):
    """Return whether TE marked a tensor-like object as non-offloadable."""
    if getattr(tensor, "_TE_do_not_offload", False):
        return True
    ...
```

**卸载路径依赖 TE 的私有属性**（TE 的公开接口见 [Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/index.html)）。 两侧的版本要是对齐，兼容性就没保证 —— 这是一个隐含的跨库耦合，不写在任何文档里。

### 4.4 `post_warmup_callback`：三段决策都来自实测字节

第 2 档最有意思的部分在 `PipelineOffloadManager.post_warmup_callback`（[L609-L706](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L609-L706)），1610 行里信息密度最高的 100 行。它在 warmup 之后、正式训练之前跑一次，用**实测到的字节数**决定最终卸载哪些组。三段：

**第一段 · offload margin**（`fine_grained_activation_offload.py` L622-L639）。先把 margin 抬到「去重后的组数」的最大值，然后**把每个名字最后一次出现的组关掉**，留出 reload 的带宽余量：

```python
self._offload_margin = max(self._offload_margin, chunk.get_max_deduplicated_groups())
...
# Mark the last group with the same name as not offloadable to make sure
# the reloading won't block the main stream.
for name, group in last_group_with_same_name.items():
    if self._offload_margin > 0:
        group.offload = False
        self._offload_margin -= 1
```

理由写在注释里：不留余量的话，最后那次 reload 会挡住主流。`get_max_deduplicated_groups`（`fine_grained_activation_offload.py` L1168-L1174）返回的是**不同组名**的个数，所以 margin 等于模块种数。

**第二段 · 按 PP rank 差异化**（`fine_grained_activation_offload.py` L640-L648）：

```python
keep_on_gpu_bytes = self._pp_rank * self._delta_offload_bytes_across_pp_ranks
for chunk in self._cached_chunks_backward:
    for group in chunk.offload_groups:
        if group.offload and keep_on_gpu_bytes > 0:
            keep_on_gpu_bytes -= group.total_offload_bytes
            group.offload = False
```

对应配置字段 `delta_offload_bytes_across_pp_ranks`（`transformer_config.py` L1337-L1341，*Difference of offload bytes across PP ranks to balance the offload load*）。注意它是**按 rank 线性递增**的：rank 0 留 0 字节、rank 1 留 $$\Delta$$、rank 2 留 $$2\Delta$$…… 于是一个 stage 内各 PP rank 留在 GPU 上的字节数相同。

**第三段 · 按比例抽样**（`fine_grained_activation_offload.py` L649-L668），对应 ``activation_offload_fraction`（`transformer_config.py` L1343-L1347，默认 1.0）：

```python
disabled_groups_count = int(offloaded_groups_count * (1 - self._activation_offload_fraction))
# Prefer keeping earlier forward groups offloaded because releasing
# those activations sooner gives the longest memory-pressure relief.
for group in reversed(eligible_offload_groups):
    if disabled_groups_count > 0:
        disabled_groups_count -= 1
        group.offload = False
```

**从后往前关。** 理由是前向靠前的组释放得早，关掉靠后的、把靠前的留着卸载，显存压力缓解的时间最长。这是整段代码里最「工程」的一句注释。

还有一处统计上的细节值得记（`fine_grained_activation_offload.py` L693-L696）：

```python
# Stop statistics at the first backward chunk after which 1F1B is running,
# where the memory cost will not increase anymore.
if chunk is self._cached_chunks_backward[0]:
    break
```

**激活峰值出现在 1F1B 刚进入稳态的那个点**，之后不再增长。这是第 6 篇（PP 调度器）要展开的同一个性质，在这里被当作卸载统计的边界用了。

另外 L669-L679 记了一类**重复搬运**：同一个组内若有多个 view 覆盖同一块 storage，第一个走整块路径后，其余的拷贝就是冗余的。源码只统计不删除（*The copies are left in place -- this is only a diagnostic*），但这一项对账本有意义 —— 它是「实际流量 > 结构需求」的一个已知来源。

### 4.5 卸载的 9 个模块，与重算的 11 个不是一套名字

`offload_modules` 的合法取值有 9 个（[L1313-L1327](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1313-L1327)）：`attn_norm` `qkv_linear` `core_attn` `attn_proj` `mlp_norm` `expert_fc1` `moe_act` `fused_group_mlp` `gdp_qkv`。配置层的 `allowed_modules` 在 L2110-L2120，逐字一致。

**两套名字有两个交集**：`core_attn`、`moe_act`、`gdp_qkv`。这三个可以同时出现在 `recompute_modules` 与 `offload_modules` 里，而且**除了 `gdp_qkv` 之外没有任何互斥检查**。也就是说 `core_attn` 可以既被重算又被卸载，两个开关正交。

这不是设计缺陷。两个动作作用在不同时机：重算改变的是「反向时要不要重新生成」，卸载改变的是「前向存下的东西放哪」。对一个模块同时开两个，等于「前向把它搬到 host、反向搬回来重算一次」。

### 4.6 `recompute_modules` × `offload_modules`：四类互斥

配置层在这两种名字之间写了四条约束。它们的形态一致 —— 全是**数据依赖**或**机制重叠**导致的矛盾，不是随便设的限制。

**其一 · 消费者必须先于生产者存在**（`transformer_config.py` L2126-L2131）：

```python
if "attn_proj" in self.offload_modules and "core_attn" not in self.offload_modules:
    raise ValueError(
        "attn_proj cannot be set to offload_modules alone without core_attn "
        "because the input of attn_proj is the output of core_attn, "
        "which is needed in core_attn.backward()")
```

`attn_proj` 的输入就是 `core_attn` 的输出，而那个输出在 `core_attn.backward()` 里还要用。**把它卸载了就没人搬运回来，重算无从下手。**

**其二 · output-discarding 重算会把卸载想留的东西丢掉**（`transformer_config.py` L2132-L2142）：

```python
if ("gdp_qkv" in self.offload_modules
        and self.recompute_granularity == "selective"
        and "gdp_in_proj" in self.recompute_modules):
    raise ValueError(
        "gdp_qkv cannot be set in offload_modules together with gdp_in_proj in "
        "recompute_modules, because gdp_in_proj discards the input of the causal "
        "conv and rematerializes it at the start of the mixer backward, leaving "
        "nothing for the gdp_qkv offload group to keep on the host.")
```

这是四类里最深的一条：**output-discarding 重算会消除卸载组的载荷。** `gdp_in_proj` 的重算在前向丢掉了 causal conv 的输入、反向开头重新物化，那么 `gdp_qkv` 卸载组想在 host 上留的那份东西**根本不存在**。

第 2.9 节那条 `mhc` + 卸载的 `NotImplementedError` 是同一族问题的另一个实例（钩子时序早于 chunk 初始化）。**两条一起说明：重计算与卸载不是两个正交的旋钮，源码用四类互斥把它们的相容性显式钉住了。**

**其三 · 整段重算与段内卸载冗余**（`transformer_config.py` L2143-L2153）：

```python
if self.recompute_granularity == "selective" and "moe" in self.recompute_modules:
    offload_inside_moe = {"moe_act", "expert_fc1", "fused_group_mlp"} & set(self.offload_modules)
    assert not offload_inside_moe, (
        f"Cannot offload {offload_inside_moe} while recomputing the entire MoE layer. "
        f"'moe' in recompute_modules wraps the full MoE forward in a checkpoint, "
        f"so offloading activations inside it is redundant and will cause errors.")
```

**其四 · paged stash 覆盖了同样的激活**（`transformer_config.py` L2188-L2205）：

```python
if self.moe_paged_stash:
    ...
    moe_offload_conflict = {"expert_fc1", "moe_act", "fused_group_mlp"} & set(self.offload_modules)
    if moe_offload_conflict:
        raise ValueError(
            "When moe_paged_stash is enabled, offload_modules must not include "
            f"expert_fc1, moe_act, or fused_group_mlp "
            f"(paged stash covers those activations). ")
```

这一条直接连到第 02 篇。`moe_paged_stash` 已经在为反向暂存同样那批专家激活，卸载就是纯重复。第 02 篇讲的是「分页暂存 + dropless 重放」这条溢出兜底路径，这里是它与卸载的边界。

### 4.7 第 3 档：optimizer offload

第 3 档的实现只有 477 行，官方 README 只有 13 行，定位写得很清楚（[`cpu_offloading/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/cpu_offloading/README.md)）：

> Add these flags to enable optimizer cpu offload in MCore.
> ```bash
> --optimizer-cpu-offload
> --optimizer-offload-fraction 1.0
> --use-precision-aware-optimizer
> ```
> Gradient copy from GPU to CPU, CPU optimizer step, and subsequent parameter copy from CPU to GPU can be time-consuming operations, and it is recommended to use the flag `--overlap-cpu-optimizer-d2h-h2d` to execute them concurrently.

一个类 `HybridDeviceOptimizer`（`hybrid_optimizer.py` L14）。第 5 节单独讲它 —— 契约要求它必须写成可独立阅读的一节，不许只在别篇提名字。

---

## 5. `hybrid_optimizer.py`：搬的是状态，不是权重

### 5.1 分界线怎么定

判定函数是 `_get_sub_optimizer_param_groups`（[L251-L302](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py#L251-L302)）。核心七行：

```python
params_total_numel = sum([param.numel() for param in params])
gpu_params_total_numel = sum([param.numel() for param in params if param.is_cuda])
cpu_params_total_numel = params_total_numel - gpu_params_total_numel
offload_threshold = gpu_params_total_numel * offload_fraction
offload_params_numel = 0
...
for param in group["params"]:
    orig_param = param
    cpu_copy = False
    if offload_params_numel < offload_threshold and param.is_cuda:
        param = param.detach().clone().cpu().pin_memory()
        offload_params_numel += param.numel()
        cpu_copy = True
```

四条性质，每条都影响账本：

1. **预算是 GPU 侧 numel 的倍数**，不是全部参数的倍数（`gpu_params_total_numel` 只累加 `param.is_cuda` 的）。已经在 host 上的参数不占预算。
2. **贪心前缀，按遍历顺序。** 一旦 `offload_params_numel >= offload_threshold`，后面全部留在 GPU。所以 `offload_fraction=0.5` **不是「后一半层搬走」，而是「按 `param_groups` 声明顺序，前若干个参数凑够一半 numel 就停」**。分界线与参数角色无关 —— 可能是 embedding，也可能正好切在某个 attention 权重中间。
3. **搬运是 `detach().clone().cpu().pin_memory()`**：新分配 host 内存、拷贝、pin 住（跨设备传输语义见 [PyTorch `Tensor.to` 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html)）。`orig_param` 仍在 GPU。
4. **只有 `is_cuda` 的参数会被搬。** 已经是 host 张量的参数直接进 CPU 组。

如果 `param_update_in_fp32` 且原 dtype 不是 fp32，还要**再 clone 一次**转 fp32（`hybrid_optimizer.py` L277-L281），并在 pin 的情况下再 pin 一次。也就是说一个被卸载的 bf16 参数在 host 上占：一份 bf16 拷贝 + 一份 fp32 主副本。

### 5.2 被卸载参数的显存账

这是本节最需要说清的一点：**被卸载的参数，GPU 上仍保留一份 bf16 计算副本。**

看 L283-L290：

```python
if cpu_copy:
    gpu_params_map_cpu_copy[orig_param] = param
    cpu_copys_map_gpu_param[param] = orig_param

if param.is_cuda:
    gpu_group["params"].append(param)
else:
    cpu_group["params"].append(param)
```

`param` 在 L274 已被重绑到 CPU 副本，所以 L287 的 `param.is_cuda` 为假，**CPU 副本只进 CPU 组**。`orig_param` 不进任何子优化器的参数列表，它只通过 `cpu_copys_map_gpu_param` 被引用。

那 GPU 上那份是谁在更新？答案是 step 之后的钩子（`hybrid_optimizer.py` L117-L127）：

```python
def param_copy_back_gpu_hook_closure():
    def param_copy_back_gpu_hook(optimizer, args, kwargs):
        self._h2d_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._h2d_stream):
            for param in _param_generator(optimizer):
                gpu_param = self.cpu_copys_map_gpu_param[param]
                gpu_param.data.copy_(param.data, non_blocking=True)
        self._h2d_stream.record_event().wait(torch.cuda.current_stream())
    return param_copy_back_gpu_hook
```

`gpu_param.data.copy_(param.data)` —— **写进 GPU 上那份 `orig_param` 的 storage。** 它是模型注册的参数，前向反向都要用。

于是账是这样（混合精度下主副本的语义见 [PyTorch AMP 文档](https://docs.pytorch.org/docs/stable/amp.html)）：

| 位置 | 被卸载参数 | 未卸载参数 |
| --- | --- | --- |
| GPU · bf16 计算副本 | **有**（`orig_param`，前反向要用） | 有 |
| GPU · fp32 主副本 | 无 | 视 `param_update_in_fp32` |
| GPU · `exp_avg` / `exp_avg_sq` | **无**（随 CPU 子优化器走） | 有 |
| host · bf16 副本 | 有 | 无 |
| host · fp32 主副本 | 视 `param_update_in_fp32` | 无 |
| host · 优化器状态 | 有 | 无 |

**结论：`--optimizer-cpu-offload` 省的是优化器状态（以及可选的 fp32 主副本），不是权重。** `offload_fraction` 决定的是「多少比例的参数的**优化器状态**搬到 host」，GPU 上的 bf16 计算副本一份都不少。这与第 05 篇「DistOpt 省的是优化器状态，不是梯度」是同一类结论 —— 降的是状态项，不是前向要读的那份权重。

### 5.3 `step()` 的五阶段

`step` 被覆写成五步（[L150-L179](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py#L150-L179)）：

```python
def step(self, closure=None):
    self._sync_hdo_param_groups_to_sub_optimizers()

    self._d2h_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(self._d2h_stream):
        self._set_sub_optimizer_grads()

    # Step the sub-optimizers.
    if self.gpu_optimizer:
        self.gpu_optimizer.step(closure)

    for cpu_optimizer in self.cpu_optimizers:
        d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
        if d2h_event is not None:
            d2h_event.synchronize()
        cpu_optimizer.step(closure)

    self._sync_sub_optimizers_state_to_hdo()
```

| 阶段 | 动作 | 流 |
| --- | --- | --- |
| 1 | 把 `param_groups`（lr / wd 等）同步给子优化器 | 主 |
| 2 | 梯度 D2H 异步拷贝 | `_d2h_stream` |
| 3 | GPU 子优化器 step，**不等 D2H** | 主 |
| 4 | 逐个 CPU 子优化器：取 D2H 事件 → `synchronize()` → step | CPU |
| 5 | 子优化器状态同步回 HDO | 主 |

第 3 步不等第 2 步、第 4 步才同步 —— 这就是 `--overlap-cpu-optimizer-d2h-h2d`（README 推荐）带来的重叠。第 4 步那次 `synchronize()` 等的是 CUDA event（[API 文档](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)），不是 CPU 侧的阻塞等待。流本身在 `_init_sub_optimizers`（`hybrid_optimizer.py` L217-L221）按开关决定：

```python
self._d2h_stream = torch.cuda.current_stream()
self._h2d_stream = torch.cuda.current_stream()
if self.overlap_cpu_optimizer_d2h_h2d:
    self._d2h_stream = torch.cuda.Stream()
    self._h2d_stream = torch.cuda.Stream()
```

**不开这个 flag 时 D2H/H2D 都在主流上同步做，一点重叠都没有。**

CPU 侧的子优化器是**每个参数一个实例**（`hybrid_optimizer.py` L227-L249）：

```python
"""Build several cpu optimizers to enable overlap. Currently we naively
assign each parameter to an individual optimizer."""
...
for param in params:
    _cpu_param_group = group_defaults.copy()
    _cpu_param_group["params"] = [param]
    cpu_optimizers.append(cpu_optimizer_cls([_cpu_param_group]))
```

源码自己写了 *naively*。粒度这么细是为了让每个参数的 D2H 事件能独立重叠，代价是 CPU 侧要维护很多个优化器对象。

还有一个细节在 L99-L115：梯度来源是 `getattr(gpu_param, "decoupled_grad", gpu_param.grad)` —— **读的是 `.grad`，不是 `main_grad`**。这一条决定了它和第 05 篇的碰撞方式。

### 5.4 与数据并行的四处咬合

`HybridDeviceOptimizer` 是在 `DistributedOptimizer` **内部**被包装的 —— `distrib_optimizer.py` 里有 10 处 `isinstance(self.optimizer, HybridDeviceOptimizer)`。四处关键：

**其一 · 分片布局算完之后重建**（[`distrib_optimizer.py` L774-L784](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/distrib_optimizer.py#L774-L784)）：

```python
) = self._build_model_and_main_param_groups(
    self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges, config
)

if isinstance(self.optimizer, HybridDeviceOptimizer):
    self.optimizer = HybridDeviceOptimizer(
        params=[g["orig_group"] for g in self.opt_group_ranges], **self.optimizer.defaults
    )
else:
    self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
    self.optimizer.load_state_dict(self.optimizer.state_dict())
```

**注意顺序**：重建发生在 `_build_model_and_main_param_groups` **之后**。也就是说 CPU/GPU 分界是在 DP 分片区间算完之后才定的，`offload_fraction` 是对**本 rank 自己那份分片**施加的。这解释了为什么分界线「与参数角色无关」在分布式下更明显 —— 遍历顺序是 `opt_group_ranges` 决定的，而那是分片布局的产物。

**其二 · 读 checkpoint 时双向同步状态**（`distrib_optimizer.py` L2117-L2118 与 L2406-L2407，两处同样的调用）：

```python
if isinstance(self.optimizer, HybridDeviceOptimizer):
    self.optimizer._sync_hdo_state_to_sub_optimizers()
```

**其三 · 微调重载走另一条路**（`distrib_optimizer.py` L3075-L3077）：

```python
if isinstance(self.optimizer, HybridDeviceOptimizer):
    self.optimizer.update_fp32_param_by_new_param()
    return
```

**其四 · 从 checkpoint 取 step 号要遍历子优化器**（`distrib_optimizer.py` L841-L850）：

```python
elif isinstance(self.optimizer, HybridDeviceOptimizer):
    step = None
    for optimizer in self.optimizer.sub_optimizers:
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            if len(optimizer.state) == 0:
                continue
            steps = list(set([s["step"].item() for s in optimizer.state.values()]))
            assert len(steps) == 1, f"steps: {optimizer.state}"
            step = steps[0]
            break
```

因为每个参数一个优化器实例，step 号要靠遍历找到第一个有状态的。AdamW 的更新式见 [PyTorch 优化器文档](https://docs.pytorch.org/docs/stable/optim.html) —— 它是逐元素运算，这一点第 05 篇已用来论证分片更新与非分片更新等价。

**碰撞的性质**：第 05 篇的梯度主路径是 `main_grad` 累积进 `_ParamAndGradBuffer` 的连续内存再 reduce-scatter；而 `_set_sub_optimizer_grads`（`hybrid_optimizer.py` L102）读的是 `gpu_param.grad`，**并且在 D2H 时才 `torch.empty` 出一块 pinned host 缓冲**（`hybrid_optimizer.py` L109-L112）：

```python
self.cpu_copy_map_grad[param] = torch.empty(
    param.shape, dtype=param.dtype, pin_memory=self.pin_cpu_grads, device="cpu"
)
param.grad = self.cpu_copy_map_grad[param]
self.cpu_copy_map_grad[param].data.copy_(grad, non_blocking=True)
```

这块 pinned host 缓冲是**第 5 项与第 6 项的交界**：它按参数逐个分配、生命周期跨 step、不参与跨 iteration 复用，但也不进任何 buffer 管理。第 05 篇的六项表里没有它的位置 —— 那张表记的「临时桶」是 `overlap_param_gather` 的 all-gather 桶，不是这个。

**这一项的大小本机无法测**（取决于 `offload_fraction` 与实际配置），**待验证**。

---

## 6. CUDA Graph 与卸载 / 重计算的冲突

### 6.1 重计算与图捕获在根上互斥

这是本篇最容易写出深度的一节，因为源码的注释把矛盾说得很直白。`tensor_parallel/checkpoint` 的入口（[`random.py` L675-L687](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/tensor_parallel/random.py#L675-L687)）：

```python
def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    from megatron.core.transformer.cuda_graphs import is_graph_capturing, is_graph_warmup

    # Skip checkpointing during CUDA graph warmup and capture, matching the behavior of
    # CheckpointWithoutOutput. The graph captures all ops directly; recomputation cannot
    # run inside a captured graph.
    if is_graph_warmup() or is_graph_capturing():
        return function(*args)
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)
```

**`recomputation cannot run inside a captured graph`** —— 这不是调优取舍，是机制冲突：

- CUDA graph 捕获的是一段固定的 kernel launch 序列，回放时按图里的拓扑执行（捕获与回放的语义见 [CUDA C++ 编程指南的 Graphs 章](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) 与 [PyTorch `torch.cuda.CUDAGraph`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)）。
- 重计算要**在反向时重新跑一遍前向**，而反向的 Python 控制流（哪些张量需要重算、重算几次）在捕获期还没确定。
- 图里的节点数与 launch 拓扑在捕获时固化，回放期无法插入新的 host 侧决策。

源码给的解法是**在 warmup 与捕获期直接不 checkpoint**（`return function(*args)`）。注意这一句注释还提到 *matching the behavior of CheckpointWithoutOutput* —— 两套实现独立地做了同一个决定。

但「捕获期跳过」不等于「重计算与图不兼容」。配置层给的是一条**相反方向**的约束（[`transformer_config.py` L2832-L2836](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L2832-L2836)）：

```python
if self.recompute_granularity:
    if self.recompute_granularity != "selective":
        assert (
            self.cuda_graph_impl == "full_iteration"
        ), "full recompute is only supported with full iteration CUDA graph."
```

**`full` 重计算不是被图禁止，而是要求整迭代图**（`cuda_graph_impl == "full_iteration"`）。两者能同时开的前提是：整迭代被一张图罩住，图内不含需要 host 侧重算的边界。`selective` 走另一条约束（`transformer_config.py` L2837-L2846）：重算模块必须**完全在图内或完全在图外**，不允许跨越图的边界 —— `transformer_engine` 图且 `recompute_modules` 含 `moe` 时，`moe_router` 不得进图。

所以准确的说法是三种组合而不是「互斥 / 不互斥」两分：

| 组合 | 判定 | 出处 |
| --- | --- | --- |
| `full` + 非 `full_iteration` 图 | **禁止**（`assert`） | L2832-L2836 |
| `selective` + `transformer_engine` 图 + `moe` 重算 + `moe_router` 进图 | **禁止**（`assert`） | L2837-L2846 |
| `full` + `full_iteration` 图 | 允许（显式要求） | L2834-L2836 |
| 捕获期 / warmup 期 | 静默跳过重算 | `random.py` L685-L686 |

`CheckpointWithoutOutput.checkpoint` 里同样的处理，理由略有不同（`random.py` L901-L906）：

```python
# If in cuda graph warmup, disable checkpointing, as 'discard_output_and_register_recompute'
# may be called in a separate graph warmup.
from megatron.core.transformer.cuda_graphs import is_graph_warmup

if is_graph_warmup():
    return run_function(*args)
```

差别在于 output-discarding 多一层顾虑：**丢弃动作和重算钩子的注册可能发生在不同的 warmup 轮次里**，那么丢弃时算出的状态到重算时就对不上了。所以它连「捕获期」都不必判断，只判 warmup 就够。

反向侧还有一处一致的判断，`CheckpointFunction.backward` 开头（`random.py` L637-L643）：

```python
from megatron.core.transformer.cuda_graphs import is_graph_capturing

if not torch.autograd._is_checkpoint_valid() and not is_graph_capturing():
    raise RuntimeError(
        "Checkpointing is not compatible with .grad(), please use .backward() if possible"
    )
```

**捕获期豁免这条检查** —— 捕获时不能用 `.grad()` 触发反向，那会生成图外的操作。

### 6.2 卸载与图重放：把 host 侧决策推迟到图外

第 2 档面对的是同一个矛盾的另一面。图捕获/回放期间，host 侧不能插入新的调度决策，但卸载恰恰需要 host 侧决策（哪些组现在搬、插在哪个流上）。

解法是**推迟**。`PipelineOffloadManager` 有一个 `_in_replay` 标志，由 `_te_cuda_graph_replay` 的包装层在图外切换（`transformer_layer.py` L1344-L1351）：

```python
if self.config.delay_offload_until_cuda_graph:
    self.off_interface.enter_replay()

try:
    return self._te_cuda_graph_replay_impl(args, kwargs, context)
finally:
    if self.config.delay_offload_until_cuda_graph:
        self.off_interface.exit_replay()
```

`enter_replay` / `exit_replay` 就是置位 / 清位（`fine_grained_activation_offload.py` L1602-L1610）：

```python
@staticmethod
def enter_replay():
    """Enter CUDA graph replay mode to enable delayed offloading."""
    PipelineOffloadManager.get_instance()._in_replay = True
```

标志在提交钩子里被读（[L1364-L1375](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L1364-L1375)）：

```python
def forward(ctx, tensor, cur_forward_chunk, name, forced_released_tensors, delay_offload):
    if delay_offload and PipelineOffloadManager.get_instance()._in_replay:
        # During TE CUDA graph replay, queue D2H work and launch it after
        # replay returns, where CPU scheduling can overlap with graph/comm gaps.
        PipelineOffloadManager.get_instance().push_offload_groups(
            cur_forward_chunk.on_group_commit_forward, name, forced_released_tensors
        )
    else:
        cur_forward_chunk.on_group_commit_forward(name, forced_released_tensors)
```

**图回放期间只入队，不发射。** D2H 的发射与 join 都走 CUDA stream / event 原语（[CUDA Runtime API 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)），这些 host 侧决策因此必须发生在图外。真正发射在图外、且在回放返回之后（`transformer_layer.py` L1357-L1361）：

```python
# Flush delayed offload groups from previous layers after graph replay.
# The CPU is idle during the sync between graph replay and a2a comm,
# so we use that time to execute the delayed offload operations.
if self.config.delay_offload_until_cuda_graph:
    self.off_interface.flush_delayed_groups()
```

注释点出了真正的优化机会：**图回放与 all-to-all 通信之间的同步窗口里 CPU 是闲的**，卸载的 D2H 正好塞在那里。默认关（`delay_offload_until_cuda_graph: bool = False`，L1331），因为它只有在用 CUDA Graph 时才有意义。

### 6.3 为什么要给 in-flight 卸载数设上限

第三个耦合点是 `fine_grained_offloading_max_inflight_offloads`（[L1367-L1373](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1367-L1373)）：

> Per fine-grained offloading group name, max number of inflight offloads for that name not yet joined on the main stream (wait_event on D2H). ... None = do not insert these joins. **This feature is particularly useful when using with full-iteration CUDA graphs**

实现是 `_drain_offload_pending`（`fine_grained_activation_offload.py` L1250-L1260）：

```python
def _drain_offload_pending(self, group_name: str) -> None:
    """For ``group_name``, have the main stream wait on older D2H events
    when that name's pending count exceeds ``_max_inflight_offloads``
    (same cap for every name; 0 = wait on each commit for that name)."""
    if self._max_inflight_offloads is None:
        return
    cur = torch.cuda.current_stream()
    q = self._offload_pending_by_name[group_name]
    while len(q) > self._max_inflight_offloads:
        old_evt = q.popleft()
        cur.wait_event(old_evt)
```

**它做的事就是限制「已发射但尚未 join 的 D2H 事件数」。** 默认 `None` = 不插入 join，于是所有 D2H 都可以一直挂着 —— 省掉了主流等待，代价是 host 侧的 pinned 缓冲（在飞 D2H 都要一块 host 内存）持续增长，**峰值显存因此不可控**。

设了上限就会周期性 join，等于用一个可调的参数把「显存峰值」和「主流等待」换算开。所以这个字段的存在本身说明：**卸载 + 全迭代 CUDA Graph 需要显式管理在飞数量，否则显存不可预测。**

### 6.4 三个开关的相容性汇总

把本章三节的结论并成一张表：

| 组合 | 能否同开 | 源码依据 |
| --- | --- | --- |
| `full` 重计算 + 非整迭代图 | **禁止**，必须 `cuda_graph_impl == "full_iteration"` | `transformer_config.py` L2832-L2836 |
| `selective` 重算跨越图边界 | **禁止**，模块须完全在图内或图外 | `transformer_config.py` L2837-L2846 |
| 捕获期 / warmup 期内的重计算 | 静默跳过（机制不支持图内重算） | `random.py` L682-L686 |
| 重计算 + SP | 可以（正是 SP 的用途） | 03 篇 |
| 重计算 + DSA + SP | **禁止** | `transformer_config.py` L1912-L1916 |
| 重计算 + 整层 CPU offload | **禁止** | L1874-L1877 |
| 重计算 + `mhc` + fine-grained 卸载 | **禁止** | L2023-L2029 |
| 整层 offload + PP | **禁止** | L1869-L1872 |
| 整层 offload + 模块级 offload | **禁止** | L2106-L2108 |
| 模块级 offload + 延迟到图后 | 可以（只在图路径生效） | `transformer_layer.py` L1344-L1351 |

**六条禁止里，五条的措辞都是「机制上做不到」而不是「暂不支持」。** 唯一一条「尚未支持」的是 `mhc` + fine-grained 卸载（`transformer_config.py` L2024 给的措辞是 *is not yet supported*，并说明是钩子时序问题）。这个区别值得记：前五条即使等多久也不会自己好，因为那是数据依赖与机制冲突；第六条是一次实现顺序问题。

---

## 7. 引用关系网图

```mermaid
graphLR
  CLAIM["显存六项账本<br/>激活项由重计算与卸载决定"]

  A1["recompute.py 仅 189 行<br/>只服务 full 粒度"]
  A2["recompute_granularity<br/>散在 19 处判断"]
  A3["uniform 的 k 不改显存<br/>block 才行"]
  A4["11 个模块 normal 与<br/>output-discarding 二分"]
  A5["DSA 只摊 args 0<br/>且与 SP 互斥"]
  A6["offload 三档<br/>粒度为层 模块 参数"]
  A7["hybrid_optimizer<br/>搬状态不搬权重"]
  A8["重计算与图捕获互斥<br/>卸载推迟到图外"]

  L1["arXiv 2205.05198<br/>选择性重算"]
  L2["arXiv 1511.01856<br/>次线性内存反向"]
  L3["arXiv 2104.04473<br/>Megatron-LM"]
  L4["arXiv 1910.02054<br/>ZeRO 状态分片"]
  D1["PyTorch 文档<br/>saved_tensors_hooks"]
  D2["PyTorch 文档<br/>CUDA 缓存分配器"]
  N1["NVIDIA 官方文档<br/>细粒度激活卸载"]
  N2["NVIDIA 官方文档<br/>optimizer CPU offload"]
  X1["A 级脚本<br/>显存与重算代价前沿"]
  X2["C 级脚本<br/>两种 checkpointing 等价性"]
  X3["C 级脚本<br/>DSA 逐 rank 形状"]

  A1 -->|"支撑 单点实现只覆盖一种粒度"| CLAIM
  A2 -->|"支撑 判断逻辑分布式"| CLAIM
  A3 -->|"支撑 k 只对 block 有效"| CLAIM
  A4 -->|"支撑 释放对象不同"| CLAIM
  A5 -->|"支撑 摊薄对象单一"| CLAIM
  A6 -->|"支撑 粒度逐档变细"| CLAIM
  A7 -->|"支撑 降的是状态项"| CLAIM
  A8 -->|"支撑 显存受调度形态约束"| CLAIM

  L1 -->|"支撑 注意力最值得选择性重算"| A3
  L2 -->|"支撑 重计算以算力换显存"| A1
  L3 -->|"支撑 六项账本的来源"| CLAIM
  L4 -->|"支撑 状态项可分片"| A7
  D1 -->|"支撑 卸载的 autograd 桥"| A6
  D2 -->|"支撑 碎片项的来源"| CLAIM
  N1 -->|"支撑 模块级卸载的官方定位"| A6
  N2 -->|"支撑 optimizer offload 的官方定位"| A7
  X1 -->|"支撑 前沿形状与 DSA 净收益"| CLAIM
  X2 -->|"支撑 重算数值恒等"| A4
  X3 -->|"支撑 DSA 形状与 SP 冲突"| A5
```

*图 1：核心论断「激活项由重计算与卸载决定」与八条源码事实（A1–A8）、四条文献与四篇官方文档构成的支撑关系。数据出处（已逐条核实）：A1 [`megatron/core/recompute.py` L21-L34](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/recompute.py#L21-L34)（全文 189 行，唯一的公共符号 `checkpointed_forward` 在 L21，只服务 `full` 粒度）；A4 [`transformer_config.py` L1918–L1919 `recompute_modules` 默认 `["core_attn"]`](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1918-L1919)与 [L1923–L1935 `allowed_modules`](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1923-L1935)（恰 11 项：core_attn、moe_act、layernorm、mla_up_proj、mlp、moe、shared_experts、gdn_norm_out、gdp_in_proj、gdp_qkv、mhc）、[L1936–L1940 非法项断言](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1936-L1940)；A6 [`pipeline_parallel/fine_grained_activation_offload.py` L443-L465](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/pipeline_parallel/fine_grained_activation_offload.py#L443-L465)（全文 1610 行，`PipelineOffloadManager` 单例在 L443）；A7 [`optimizer/cpu_offloading/hybrid_optimizer.py` L251-L302](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py#L251-L302)（全文 477 行，CPU/GPU 分界判定在 L251-L302）；A8 [L2832–L2845 `assert self.cuda_graph_impl == "full_iteration"`](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L2832-L2845)（*full recompute is only supported with full iteration CUDA graph.*）。A2 的 enforcing 代码是 `transformer_block.py` [L662](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_block.py#L662)（`recompute_granularity == 'full'` 才调 `checkpointed_forward`）与 `attention.py` [L392-L395](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/attention.py#L392-L395)；A3 是 `recompute.py` [L157-L166](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/recompute.py#L157-L166)（`uniform` 分支每段都 `use_checkpoint=True`）；A5「DSA 与 SP 互斥」的校验在 `transformer_config.py` [L1912-L1916](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1912-L1916)，两者同开直接 `raise ValueError`（*must be false when sequence parallel is enabled*）。*

这张图的结构：中心论断是「激活项由重计算与卸载决定」，八条源码事实（A1-A8）是它的直接支撑，四条文献与四篇官方文档给出外部锚点，三个实验脚本做交叉验证。**即使读者不接受某一条源码证据，只要接受实验或文献中的任一条，论断仍站得住** —— 这是本系列引用关系网图的统一画法。

图里论文类来源的引用遵循 arXiv 官方「Permissions and Reuse」页（[reuse](https://info.arxiv.org/help/license/reuse.html)、[license](https://info.arxiv.org/help/license/index.html)）的判定：本文只用文字引用其结论，不复制任何一张论文图。这张图的结构：

---

## 8. 小结与下一篇

本篇钉死了七件事：

1. **六项账本里，激活项的填法**是逐张量清单，而不是某个经验公式。基线（Llama-2 7B 结构、seq 4096、TP=8、SP 开、micro batch 1）是 9.19 GiB。
2. **`recompute.py` 的 189 行只服务 `full`**；`selective` 的 11 个模块各自在模块文件里判断，`recompute_granularity` 在仓库里被 19 处读取。`recompute_num_layers` 在 `uniform` 下不改变显存（实测恒为 0.125 GiB），只有 `block` 能用 k 换显存。
3. **11 个 `recompute_modules` 分成两类**，4 个 normal、7 个 output-discarding，实现是两套（`CheckpointFunction` vs `CheckpointWithoutOutput`）。后者用 `untyped_storage().resize_(0)` 释放字节，用 C++ 扩展在 `StorageImpl` 层换指针恢复，避免 version counter 触发 autograd 报错。
4. **`distribute_saved_activations` 只摊 `args[0]`**，切完是一维 `numel // tp_size`，切的是序列维 —— 与 SP 同一个轴，所以配置层直接禁止组合，实测净收益 0.000 GiB。
5. **offload 是三档**，粒度依次为层、模块、参数。模块级那一档的 `post_warmup_callback` 在 warmup 后用实测字节做三段决策：留 reload margin、按 PP rank 线性差异化、从后往前抽样。
6. **`recompute_modules` 与 `offload_modules` 之间有四类互斥**，形态都是数据依赖或机制冲突；最深的一条是 `gdp_in_proj` 的 output-discarding 重算会让 `gdp_qkv` 卸载组没有东西可留。
7. **`hybrid_optimizer.py` 省的是优化器状态不是权重** —— 被卸载参数在 GPU 上仍留 bf16 计算副本，H2D 钩子每步把 CPU 的更新写回那份副本。它在 `DistributedOptimizer` 内部被包装，分片布局算完后重建，四处需要显式同步状态。

重计算与 CUDA Graph 的关系也一并钉死，且**不是一句「互斥」**：图内不能重算，所以捕获期与 warmup 期跳过 checkpointing；但 `full` 重计算与图不是互斥而是**绑定**——配置层要求它必须配整迭代图，`selective` 则要求重算模块不跨图边界。卸载那一侧的解法不同：把 host 侧决策推迟到图回放之后，理由是图与通信之间的同步窗口 CPU 闲。

三个明确的**待验证**（本篇不编造）：

- `uniform` 切段策略下的实际吞吐损失（本机无 GPU，只能给 `extra/F` 这个结构量；真机侧该用 PyTorch profiler 逐 iteration 读数，见 [profiler 文档](https://docs.pytorch.org/docs/stable/profiler.html)）
- `recompute_num_layers` 三档对齐常数的动机（源码未写，第 05 篇已核）
- `hybrid_optimizer` 在真实配置下 pinned host 缓冲的规模（依赖 `offload_fraction` 与实际模型）

两个必须说清的适用边界：

- §2.5 的 `core_attn` 收益数只对**注意力核用融合 kernel**的配置成立；显式 materialize 分数矩阵时收益会大得多。
- §4.3 的整块搬运依赖 TE 私有属性 `_TE_do_not_offload`，跨库版本不对齐时行为无保证。

下一篇写**流水并行调度器**：`schedules.py` 里 1F1B 的实际代码结构、warmup / steady / cooldown 三段、virtual pipeline 省掉多少气泡。本篇 §4.4 已经埋了一个引子 —— 激活峰值出现在 1F1B 刚进稳态的那一点，那篇会把它讲透。

<!-- TODO-next: 06（PP）落盘后把上一篇指向它 -->

---

## 参考

### 源码（本文对应 commit `60e039626`，`megatron-core` 0.20.0）

仓库 `https://github.com/NVIDIA/Megatron-LM`。本篇引用文件与行号范围：

- `megatron/core/recompute.py` 189 行 —— L21 `checkpointed_forward`；L125-L144 `chunk_runner` 与两条 checkpoint 路径；L157-L166 `uniform`；L167-L183 `block`
- `megatron/core/transformer/transformer_config.py` 3311 行 —— L525 `recompute_granularity`；L536 `recompute_method`；L544 `recompute_num_layers`；L550 `distribute_saved_activations`；L553-L575 `recompute_modules`（11 个取值与二分在 L572-L574）；L1308-L1373 细粒度卸载字段组；L1862-L1916 六条硬校验；L1918-L1921 `recompute_modules` 默认值；L1923-L1990 模块级校验；L2005-L2029 `mhc` 约束；L2105-L2172 `offload_modules` 校验；L2188-L2205 `moe_paged_stash` 冲突
- `megatron/core/pipeline_parallel/fine_grained_activation_offload.py` 1610 行 —— L11 文件头注释；L36-L49 `_te_do_not_offload`；L150 `OffloadTensorPool`；L384 `OffloadTensorGroup`；L443 `PipelineOffloadManager`；L609-L706 `post_warmup_callback`；L863-L878 `saved_tensors_hooks` 桥；L881-L952 `ChunkOffloadHandler`；L892 `BASE_OFFLOAD_MIN_COVERAGE`；L895-L934 整块搬运；L985-L996 `_discard_outputs`；L1168-L1174 `get_max_deduplicated_groups`；L1250-L1260 `_drain_offload_pending`；L1357-L1387 `FineGrainedOffloadingGroupCommitFunction`；L1496 `FineGrainedOffloadingInterface`；L1602-L1610 `enter_replay` / `exit_replay`
- `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py` 477 行 —— L14 `HybridDeviceOptimizer`；L83-L115 `_set_sub_optimizer_grads`；L117-L148 H2D / fp32 回写钩子；L150-L179 `step`；L181-L224 `_init_sub_optimizers`；L227-L249 `build_cpu_optimizer_list`；L251-L302 `_get_sub_optimizer_param_groups`
- `megatron/core/optimizer/cpu_offloading/README.md` 13 行 —— 三个开关与重叠建议
- `megatron/core/optimizer/distrib_optimizer.py` —— L774-L784 分片布局后重建；L841-L850 step 号提取；L2117-L2118 / L2406-L2407 状态同步；L3075-L3077 微调重载
- `megatron/core/tensor_parallel/random.py` 1022 行 —— L32 C++ 扩展注释；L600-L672 `CheckpointFunction`；L675-L687 `checkpoint`；L748-L800 `CheckpointWithoutOutputFunction`；L803-L845 `CheckpointWithoutOutputManager`；L848-L1010 `CheckpointWithoutOutput` —— L862-L875 `fp8` 怪癖；L967-L978 换指针；L985-L996 丢弃
- `megatron/core/tensor_parallel/utils.py` —— L48-L76 `split_tensor_into_1d_equal_chunks`；L79-L93 `gather_split_1d_tensor`
- `megatron/core/transformer/transformer_block.py` —— L299-L302 `core_attn` 谓词；L304-L318 整层 offload 接入；L319-L325 TE 门禁；L655-L690 `checkpointed_forward` 调用点 —— L662 分派条件
- `megatron/core/transformer/transformer_layer.py` —— L1344-L1351 `enter_replay` / `exit_replay`；L1357-L1361 `flush_delayed_groups`
- `megatron/core/transformer/attention.py` L392-L395 —— `core_attn` 谓词
- `megatron/core/optimizer/param_layout.py` —— L19-L21 `pad_to_divisor`；L24-L26 `pad_param_start`；L29-L33 `bucket_end_divisor`；L36-L42 `pad_bucket_end`
- `megatron/core/utils.py` —— L727-L753 `GlobalMemoryBuffer`
- `megatron/core/distributed/fsdp/src/megatron_fsdp/utils.py` L769 —— 第二份 `GlobalMemoryBuffer`
- `megatron/core/transformer/moe/paged_stash.py` 1326 行 —— 第 02 篇已详
- `recompute_granularity` 的其余读取点 —— `transformer/moe/moe_layer.py` L248 / L253、`moe/experts.py` L270、`moe/shared_experts.py` L157、`multi_latent_attention.py` L172、`multi_token_prediction.py` L1532、`models/hybrid/hybrid_block.py` L359、`transformer/cuda_graphs.py` L1024、`ssm/gated_delta_net/common.py` L271、`ssm/gated_delta_product.py` L271 / L276、`transformer/experimental_attention_variant/absorbed_mla.py` L183

### 论文

- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*：https://arxiv.org/abs/2205.05198
  —— `transformer_config.py` L531 的 docstring 直接引这篇；其 abs 页**无 CC 标识**，故本文相关图一律自绘。
- Chen, Xu, Zhang, Guestrin, *Training Deep Nets with Sublinear Memory Cost*：https://arxiv.org/abs/1604.06174
  —— 「以计算换显存」与分段代价 $$O(n/k) + O(k)$$ 的出处，也是 §0 那个 30% 数字的来源。
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*：https://arxiv.org/abs/2104.04473
- Shoemaker et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*：https://arxiv.org/abs/1909.08053
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*：https://arxiv.org/abs/1910.02054
- Rajbhandari et al., *ZeRO-Offload: Democratizing Billion-Scale Model Training*：https://arxiv.org/abs/2101.06840
- Rajbhandari et al., *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*：https://arxiv.org/abs/2104.07857
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*：https://arxiv.org/abs/2205.14135
- Liu et al., *Ultra-Long Sequence Distributed Transformer*：https://arxiv.org/abs/2311.02382
- Vaswani et al., *Attention Is All You Need*：https://arxiv.org/abs/1706.03762

### 官方文档

- NVIDIA Megatron-LM · 细粒度激活卸载（`transformer_config.py` L1334 / L1340 / L1346 三处 docstring 链接的锚点）：https://github.com/NVIDIA/Megatron-LM/blob/60e039626/docs/user-guide/features/fine_grained_activation_offloading.md
- NVIDIA Megatron-LM · optimizer CPU offload：https://github.com/NVIDIA/Megatron-LM/blob/60e039626/docs/user-guide/features/optimizer_cpu_offload.md
- Megatron-Core 用户指南（并行策略、卸载等特性的总入口）：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html
- Megatron-Core API 指南（`TransformerConfig` 等配置字段页）：https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/index.html
- Megatron Core 并行策略指南（03 篇引 SP 官方措辞处）：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- Transformer Engine 文档（`te_checkpoint` 与 FP8 量化上下文）：https://docs.nvidia.com/deeplearning/transformer-engine/index.html
- NCCL 集合通信文档（缓冲区尺寸与均分要求）：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
- NCCL 用户指南概览：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
- CUDA C++ 编程指南（Graphs 章，捕获与回放语义）：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- CUDA Runtime API 文档（stream / event 原语）：https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
- PyTorch · `torch.utils.checkpoint`（`CheckpointFunction` 的抄写来源）：https://docs.pytorch.org/docs/stable/checkpoint.html
- PyTorch · autograd 与 saved-tensor 语义：https://docs.pytorch.org/docs/stable/autograd.html
- PyTorch · `torch.autograd.graph.saved_tensors_hooks`（第 4.3 节卸载桥所依赖的机制）：https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks
- PyTorch · `Tensor.untyped_storage`（`_discard_outputs` 所用的 API）：https://docs.pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html
- PyTorch · `Tensor.to`（跨设备传输与 pin 住语义）：https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html
- PyTorch · CUDA 缓存分配器与 `PYTORCH_CUDA_ALLOC_CONF`：https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
- PyTorch · CUDA streams：https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams
- PyTorch · `torch.cuda.Event`：https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html
- PyTorch · `torch.cuda.CUDAGraph`：https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html
- PyTorch · 分布式与集合通信：https://docs.pytorch.org/docs/stable/distributed.html
- PyTorch · FSDP：https://docs.pytorch.org/docs/stable/fsdp.html
- PyTorch · 自动混合精度：https://docs.pytorch.org/docs/stable/amp.html
- PyTorch · 优化器（含 AdamW 更新式）：https://docs.pytorch.org/docs/stable/optim.html
- PyTorch · `torch.nn.RMSNorm`：https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
- PyTorch · profiler：https://docs.pytorch.org/docs/stable/profiler.html
- arXiv 官方「Permissions and Reuse」页（本系列判定论文图许可的依据）：https://info.arxiv.org/help/license/reuse.html
- arXiv 官方许可条款（CC BY 4.0 允许 reuse / remix / adapt）：https://info.arxiv.org/help/license/index.html

### 配套实验与本系列

- 配套实验（`ipynbs` 仓库）：`experiments/megatron/memory_ledger/`
  - `pareto.py` —— 逐张量保存集、11 个模块绑定、显存与重算代价双轴扫描
  - `selective_equivalence.py` —— normal 与 output-discarding 的前向与梯度逐位对照、丢弃字节验证、RNG 恢复必要性
  - `dsa_shapes.py` —— `split_tensor_into_1d_equal_chunks` 下标算术复刻、SP 双重摊分、非整除边界
  - `results/stdout.txt` —— 固化输出
14. 本系列前作：《（00）代码地图、配置系统与一次迭代的控制流全图》[/2026/09/28/megatron-00-foundation/]、《（01）进程组——网格切分、通信域与计算组织》[/2026/10/08/megatron-01-process-groups-and-comm-overlap/]、《（02）专家并行——进程组、token dispatch 全链路与负载均衡》[/2026/10/08/megatron-02-expert-parallel/]、《（03）张量并行与序列并行——f / g 算子与 RS + AG 的真实实现》[/2026/09/28/megatron-03-tensor-parallel-and-sequence-parallel/]、《（04）GTP：新一代张量并行》[/2026/10/11/megatron-04-generalized-tensor-parallelism/]、《（05）数据并行与优化器——三套实现的分野》[/2026/10/10/megatron-05-data-parallel-and-optimizers/]
