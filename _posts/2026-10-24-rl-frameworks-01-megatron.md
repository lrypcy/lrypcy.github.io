---
title: "Megatron 训练栈的框架视角：数据怎么进来、HF 模型怎么改、自动切分切到哪、怎么落到卡上"
date: 2026-10-24 09:00:00 +0800
categories:
  - 分布式训练
tags: [megatron, megatron-bridge, data-pipeline, gpt-dataset, hf-conversion, tensor-parallel, column-parallel, row-parallel, sequence-parallel, qkv, sharded-state, checkpoint-reshard, source-code]
layout: post
mathjax: true
---

> **系列导航** ｜ Megatron 系列已完成 14 篇（[00 地基](/2026/09/28/megatron-00-foundation/) → [01 进程组与通信重叠](/2026/10/08/megatron-01-process-groups-and-comm-overlap/) → … → [16 混合与 MoE 模型装配](/2026/10/14/megatron-16-hybrid-and-moe-model-assembly/)），讲的是并行原理与源码层；本文是其中的**框架视角补充篇**，只回答「一个外部模型与一份外部数据，怎么变成一个能在多卡上跑起来的 Megatron 作业」。RL 框架系列：[02 slime](/2026/10/24/rl-frameworks-02-slime/) ｜ [03 veRL](/2026/10/24/rl-frameworks-03-verl/) ｜ [08 两族横评](/2026/10/24/rl-frameworks-08-landscape/)

**TL;DR**

> * **数据链路是「三重切分」叠加**：拆文档 → 按 `seq_len` 打包成样本 → 按 DP rank 分样本。前两层决定一条样本长什么样，第三层决定它被哪张卡读到。三层的边界分别落在 `GPTDataset.__getitem__`、`build_pretraining_data_loader` 的 sampler、以及 `consumed_samples` 这个续训游标上。
> * **普通 HF 模型不能直接用，缺的是三样东西**：并行层替换（`nn.Linear` → `ColumnParallelLinear` / `RowParallelLinear`）、参数名与布局改写（q/k/v 三段合并成 GQA 交错布局）、以及一份**可逆的映射表**（Bridge 的 `MegatronMappingRegistry`）。
> * **切分点的正确性可以被证明**：`ColumnParallelMapping` 沿 dim 0 切、`RowParallelMapping` 沿 dim 1 切，两者在 `Y = XW` 的矩阵乘法里天然互补——上游按输出维切、下游按输入维切，中间的局部结果相加即得全局结果。这就是「为什么这样切是等价的」的全部答案。
> * **QKV 是唯一的例外，也是唯一需要逐个模型写代码的地方**。HF 是三个独立矩阵，Megatron 是一个按 GQA 交错的融合矩阵：顺序是 $$[q_1\ldots q_n, k_1, v_1, q_1\ldots q_n, k_2, v_2, \ldots]$$，其中 $$n = n_{\rm heads}/n_{\rm query\_groups}$$。这个交错布局不能靠「沿某个维度 concat」得到。
> * **自动切分有三个静默退化的边界**：`num_query_groups` 不能被 TP 整除、`sequence_parallel` 开着但注意力没按序列切、以及 embedding 与 LM head 共享权重时 vocab 不能被 TP 整除。三者都不报错，只是算出来的东西不对或者性能莫名其妙地掉。
> * **「切完怎么落到卡上」的答案不只是广播**。每个 rank 拿到的是自己那一份分片，但**初始化必须各 rank 各自做**：带 rank 的随机种子 + 本地分片形状，而不是「主 rank 建完整模型再切」。后者在 TP 大于 1 时会因为形状不一致直接崩，在 PP 下更是不可能。
> * **RL 场景下这条数据链路会被整段绕过**：`args.perform_rl_step` 为真时 Megatron 直接把三个 dataloader 全置为 `None`，数据由外部框架（slime / veRL）自己塞进来。理解这一点才能看清 RL 框架与 Megatron 的真实分工边界。

---

## 目录

- [1. 本文的边界：14 篇讲过的与没讲的](#1-本文的边界14-篇讲过的与没讲的)
- [2. 数据链路（一）：从文档到样本](#2-数据链路一从文档到样本)
- [3. 数据链路（二）：从样本到卡](#3-数据链路二从样本到卡)
- [4. HF 模型改造：缺的是哪三样东西](#4-hf-模型改造缺的是哪三样东西)
- [5. 矩阵视角：为什么沿 dim 0 / dim 1 切是等价的](#5-矩阵视角为什么沿-dim-0--dim-1-切是等价的)
- [6. QKV 与 gate_up：唯一需要逐模型写代码的地方](#6-qkv-与-gateup唯一需要逐模型写代码的地方)
- [7. 自动切分的三个静默边界](#7-自动切分的三个静默边界)
- [8. 切完怎么落到卡上](#8-切完怎么落到卡上)
- [9. RL 场景：这条链路被整段绕过](#9-rl-场景这条链路被整段绕过)
- [10. 小结：Megatron 在 RL 栈里的真实位置](#10-小结megatron-在-rl-栈里的真实位置)

---

## 1. 本文的边界：14 篇讲过的与没讲的

Megatron 系列已经把并行机制讲透了。本文刻意**不重复**这些内容：

| 主题 | 已有位置 |
| --- | --- |
| 进程组、rank / world size 语义、通信与计算重叠 | `megatron-01` |
| TP / SP / GTP 的切分原理 | `megatron-03`、`megatron-04` |
| 三套 DP 优化器与分布式参数管理 | `megatron-05` |
| PP 调度器 | `megatron-06` |
| 重计算与显存账本 | `megatron-08` |
| refit 与权重导出 | `megatron-11` |
| 模块规格与模型装配 | `megatron-14`、`megatron-15`、`megatron-16` |

本文只补四件事，它们共同回答一个问题：**一个外部世界的东西（HF 模型、JSONL 数据）怎么变成 Megatron 作业**。

> **基准声明**：本文引用 Megatron-LM `core_v0.19.2`（`4b4acac9a1d28ea6829c8d4f566d75698a21249d`）与 Megatron-Bridge `v0.6.2`（`c0e164ed2aedac4ad1c877780e2564a19d5d54ec`）。`megatron-` 系列的行号锁在 `60e039626`（v0.20.0），**两者不同，行号不可跨篇比对**。

---

## 2. 数据链路（一）：从文档到样本

### 2.1 三层结构

一份 JSONL 语料要变成一张卡上的一个 batch，中间隔三层：

```text
JSONL / HF dataset
   │  ① 索引：按 document + shuffle 索引做采样（不把全量读进内存）
   ▼
GPTDataset[idx]  →  一个长度为 seq_len 的 token 序列
   │  ② 打包：把连续文档拼到恰好 seq_len，跨越边界的地方记 document_lengths
   ▼
按 DP rank 分配样本下标
   │  ③ 分片：sampler 决定 idx 的分布
   ▼
micro-batch → 前向
```

三层的代码位置完全不同：① 在 `GPTDataset._query_document_sample_shuffle_indices`，② 在 `GPTDataset.__getitem__`，③ 在 `megatron/training/datasets/data_samplers.py` 的 sampler 构造里。

### 2.2 第一层：按索引采样而不是预切

`GPTDataset.__getitem__` 的第一件事是查索引，不是读缓存：

```python
if idx is None:
    # Batch padding sequence so the index does not matter
    text, _, document_lengths = self._query_document_sample_shuffle_indices(0)
else:
    text, _, document_lengths = self._query_document_sample_shuffle_indices(idx)
```

`idx is None` 是一个专门为**流水线尾部的 padding 样本**留的分支：当 micro-batch 不足以填满时，Megatron 会用 `None` 索引取一条padding 样本并把 loss mask 全清零：

```python
# Batch padding sequence so we mask the loss
if idx is None:
    loss_mask = torch.zeros_like(loss_mask)
```

这个设计意味着**padding 样本也走完整的 tokenize 与 mask 构造流程**，只是最后把 loss 抹掉。好处是形状与真实样本完全一致，不需要为 padding 单独写一条前向路径。

### 2.3 第二层：tab 与 label 的错位

样本拿到之后的第二件事是造 `tokens` / `labels` 的错位：

```python
if self.config.add_extra_token_to_sequence:
    tokens = text[:-1].contiguous()
    labels = text[1:].contiguous()
else:
    tokens = text
    labels = torch.roll(text, shifts=-1, dims=0)
    labels[-1] = self._pad_token_id
```

两条分支的差别在于**样本长度是 `seq_len + 1` 还是 `seq_len`**。`add_extra_token_to_sequence` 为真时多取一个 token 供错位使用，此时 `text[:-1]` 与 `text[1:]` 长度都是 `seq_len`；为假时用 `roll` 循环移位，末位补 pad。

这个细节在数据管线里经常被忽略，但它决定了 `drop_last` 与总样本数怎么算——**多取一个 token 意味着每个样本实际消耗 `seq_len + 1` 个 token 的语料**。

### 2.4 第三层：跨文档边界怎么记

打包会把多个文档拼进一个样本，于是 `document_lengths` 记录每个文档占了多少 token。当 `inter_document_masking` 打开时，这些长度被转成 `cu_seqlens` 交给 flash attention 的变长接口：

```python
if self.config.add_extra_token_to_sequence:
    document_lengths[-1] -= 1
    if document_lengths[-1] == 0:
        document_lengths.pop()
# If the sample was padded (e.g., the last validation sample),
# fold the padding into the last document so cu_seqlens[-1]
# equals sequence_length.
```

注释说明了这段减一的理由：`_query_document_sample_shuffle_indices` 取的是 `seq_len + 1` 个 token，多出来的那一个被追加到最后一个文档里（用来造错位的 label），所以计算 `cu_seqlens` 之前必须把它减掉。**这是一个典型的「两个子系统各自正确、接口处差一」的问题**，只能靠注释守。

### 2.5 掩码与位置编码的缓存

`_get_ltor_masks_and_position_ids` 的产物（attention_mask / loss_mask / position_ids）在 `masks_and_position_ids_are_cacheable` 为真时只算一次并缓存。原因很直接：**同一个 `seq_len` 下，这三个张量与内容无关**，只与长度和 padding 有关。省掉它们的重复构造对长序列（8k、32k）的训练吞吐有可见影响。

---

## 3. 数据链路（二）：从样本到卡

### 3.1 sampler 决定 idx 的分布

`build_pretraining_data_loader` 是第三层切分的落点。它先按 split 决定 batch size，再按四种情况选 sampler：

| 条件 | sampler | 语义 |
| --- | --- | --- |
| `split == valid` 且 `full_validation` | `MegatronFullValidationSampler` | 所有 rank 跑同样步数，长度取 DP 间最大值 |
| `dataloader_type == single` 且 `hybrid_context_parallel` | `HybridCPMegatronPretrainingSampler` | CP 与 DP 混合切 |
| `dataloader_type == single` | `MegatronPretrainingSampler` | 标准 DP 切 |
| 其他 | 按 `cyclic` / `external` 走 | — |

构造时传入的四个参数里，`consumed_samples` 是最容易出问题的那个——它是**断点续训的游标**。

### 3.2 `consumed_samples` 是一个跨三处的隐式状态

`build_train_valid_test_data_loaders` 开头有一段专门为「按 iteration 训练」做的向后兼容：

```python
# Backward compatibility, assume fixed batch size.
if args.iteration > 0 and args.consumed_train_samples == 0:
    assert args.train_samples is None, 'Only backward compatiblity support for iteration-based training'
    args.consumed_train_samples = args.iteration * args.global_batch_size
```

也就是说 **`consumed_train_samples` 必须与 `iteration` 保持一致**，而这个一致性没有任何机制保证，只能靠调用方维护。续训时如果 checkpoint 里的 `consumed_train_samples` 与 `iteration * global_batch_size` 对不上，数据会被重复读或跳过，而**训练不会报错**——只是收敛曲线会莫名其妙地变差。这是 Megatron 数据管线里最隐蔽的一类 bug。

验证集的游标更绕：`consumed_valid_samples` 的计算里带 `start_eval_at_iter` 与 `eval_interval`，因为「从第几步开始评测」会改变已经消费了多少验证样本。

### 3.3 TP rank 0 建完再广播

数据管线的构造是**有条件的**：

```python
if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:
    # 建 dataset 与 dataloader
    ...
    flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], ...)
else:
    flags = torch.tensor([0, 0, 0], ...)

torch.distributed.broadcast(flags, 0)
args.do_train = getattr(args, "do_train", False) or flags[0].item()
```

只有 TP rank 0 会真正构造 dataset（除非 dataset 自己声明 `is_distributed`），然后把「这一阶段要不要 train / valid / test」三个标志广播给所有 rank。**被广播的不是数据本身，而是「有没有数据」这件事**。

为什么 TP 内只需一份：TP 组内所有 rank 读的是**同一批样本**（正是 TP 的定义——同一条样本的不同切片）。所以 dataset 对象在 TP 组内是冗余的，只留 rank 0 一份即可省内存与文件句柄。真正需要每 rank 一份的是 DP 维度，这一点由 sampler 的 `data_parallel_rank` 处理。

---

## 4. HF 模型改造：缺的是哪三样东西

### 4.1 为什么 HF 模型不能直接塞进去

一个 HuggingFace 的 `LlamaForCausalLM` 扔进 Megatron 会立刻遇到三个不同层次的问题：

| 层 | 问题 | 表现 |
| --- | --- | --- |
| **计算层** | HF 用 `nn.Linear`，没有并行切分 | 即使给了 TP 配置也不会切，全部算在每张卡上 |
| **参数层** | HF 的 q/k/v 是三个独立矩阵，Megatron 是一个融合矩阵 | 名字都对不上，`load_state_dict` 直接报 missing |
| **映射层** | 没有「HF 名 ↔ Megatron 名 + 切分方式」的表 | 即使逐个手动改也容易漏 |

Bridge 的解法是给出三样对应的东西：**并行层替换（provider 负责）**、**参数布局转换（mapping 负责）**、**可逆映射表（registry 负责）**。

### 4.2 映射层：`MegatronMappingRegistry`

`mapping_registry.py` 的核心是一个注册表，支持按参数名模式做双向查找：

```python
def megatron_to_hf_lookup(self, megatron_param_name: str) -> Optional[MegatronParamMapping]:
...
def hf_to_megatron_lookup(self, hf_param_name: str) -> Optional[MegatronParamMapping]:
```

注意它是**双向**的：既能从 HF 转进 Megatron（加载预训练权重），也能从 Megatron 转回 HF（导出、给推理引擎用）。RL 框架最依赖的正是后一个方向——slime 的 `megatron_to_hf/<model>.py` 干的同一件事，只是自己写而不是用 Bridge。

注册表还额外做了两件自动补全：

- `_add_separate_layernorm_mappings`：当模型用独立的 LayerNorm 参数（而非 fused）时自动补齐映射
- `_add_quantization_mappings`：量化权重的映射
- `_convert_pattern_to_regex`：把 `decoder.layers.*.self_attention...` 这种通配模式转成正则

**这个设计的关键在于「模式匹配而不是枚举」**：40 层的模型不需要写 40 条映射，一条带 `*` 的模式就能覆盖所有层。代价是模式冲突时后注册的会覆盖先注册的，调试时只能靠 `describe()` 打印全部映射来排查。

### 4.3 三种基础映射

`param_mapping.py` 里最基础的是三个类，它们覆盖了绝大多数参数：

| 类 | 语义 | 切分维度 |
| --- | --- | --- |
| `DirectMapping` | 直接搬运，不变形 | 无 |
| `ColumnParallelMapping` | 列并行（embedding、QKV、MLP up） | dim 0 |
| `RowParallelMapping` | 行并行（attention out、MLP down） | dim 1 |
| `ReplicatedMapping` | 复制，每 rank 一份 | 无 |

`ColumnParallelMapping` 的文档写得最清楚：

> Column-parallel layers in Megatron split the output dimension across tensor parallel ranks. […] The weight matrix is partitioned along dimension 0 (rows), so each TP rank holds a subset of output features while maintaining all input features.

- 原始权重：$$[\,d_{\rm out},\, d_{\rm in}\,]$$
- 每个 rank：$$[\,d_{\rm out}/t,\, d_{\rm in}\,]$$

`RowParallelMapping` 则切**输入维**（dim 1），并且它的文档里有一段值得单独看的说明：

> **Forward path (external → Megatron)**
> 1. Rank 0 validates that the *second* dimension is divisible by `tp_size`.
> 2. Rank 0 splits the tensor with `torch.chunk(..., dim=1)` producing `tp_size` equally-sized shards.
> 3. The shards are **scattered** so that every TP rank receives exactly one shard matching the shape of its local Megatron parameter.

三个动作：**rank 0 校验整除 → rank 0 切 → scatter 到各 rank**。校验放在 rank 0 是刻意的：不整除时只需要一个 rank 报错就够了，让所有 rank 都去算一遍整除判断纯属浪费。

---

## 5. 矩阵视角：为什么沿 dim 0 / dim 1 切是等价的

这是本文唯一需要一点数学的地方，但它同时也是整篇文章里最容易被当作「显然」而跳过的地方。

### 5.1 列并行 + 行并行是一对

考虑一个两层 MLP：$$Y = f(X W_1) W_2$$，其中 $$W_1 \in \mathbb{R}^{d_{\rm in} \times d_{\rm hidden}}$$、$$W_2 \in \mathbb{R}^{d_{\rm hidden} \times d_{\rm out}}$$。

**列并行**把 $$W_1$$ 的**输出维**（即 hidden 维）切成 $$t$$ 份：

$$W_1 = \begin{bmatrix} W_1^{(1)} & W_1^{(2)} & \cdots & W_1^{(t)} \end{bmatrix},\qquad W_1^{(i)} \in \mathbb{R}^{d_{\rm in} \times d_{\rm hidden}/t}$$

每个 rank 独立算出自己的局部激活：$$Z^{(i)} = f(X W_1^{(i)})$$，不需要任何通信。

**行并行**把 $$W_2$$ 的**输入维**（同样是 hidden 维）切成同样的 $$t$$ 份：

$$W_2 = \begin{bmatrix} W_2^{(1)} \\ W_2^{(2)} \\ \vdots \\ W_2^{(t)} \end{bmatrix},\qquad W_2^{(i)} \in \mathbb{R}^{d_{\rm hidden}/t \times d_{\rm out}}$$

每个 rank 算局部乘积：$$P^{(i)} = Z^{(i)} W_2^{(i)}$$。因为矩阵乘法的分块性质，

$$Y = Z W_2 = \sum_{i=1}^{t} Z^{(i)} W_2^{(i)} = \sum_{i=1}^{t} P^{(i)}$$

所以最后只需要一次 all-reduce 求和。**这就是「切分是等价的」的全部证明**——它不依赖任何近似，是精确的代数恒等式。

### 5.2 数值上并不完全等价

代数等价不等于浮点等价。分块求和的顺序由 TP rank 的归约顺序决定，$$t$$ 路 all-reduce 的求和顺序与单卡算完整矩阵乘的顺序不同，因此**最后一个 bit 会不一样**。这一点在 RL 里会变得重要：当推理引擎用另一套并行度算 logprob 时，累积的浮点差异会进入重要性比的分母。这就是训推一致「数值层」的根源之一，详见第 7 节。

### 5.3 TEG 的融合边界

真实的 Transformer 里，列并行与行并行总是成对出现：attention 的 QKV 投影是列并行，输出投影是行并行；MLP 的 up 投影是列并行，down 投影是行并行。**融合算子（fused attention、fused MLP）会把这对边界吞掉**，于是通信次数从「每个子层两次」降到「每个子层一次」。

理解这一点才能解释一个常见的困惑：**为什么开了 TP 之后算力利用率反而低于单卡**——因为即使代数上零冗余，通信与融合边界上的同步开销仍然在。

---

## 6. QKV 与 gate_up：唯一需要逐模型写代码的地方

### 6.1 为什么 QKV 不能靠 concat

如果模型是标准多头注意力（$$n_{\rm heads} = n_{\rm kv\_heads}$$），HF 的 `q_proj` / `k_proj` / `v_proj` 形状相同，Megatron 的 `linear_qkv` 就是把三者沿输出维拼接。这种情形下 `QKVMapping` 退化成一次 concat。

**但只要引入 GQA，交错方式就变了**。`QKVMapping` 的文档给出了 Megatron 的实际布局：

> Interleaving order: `[q1...qn, k1, v1, q1...qn, k2, v2, ...]`
> Where `n = num_attention_heads / num_query_groups`

也就是说当 $$n_{\rm query\_groups} = 8$$、$$n_{\rm heads} = 32$$ 时，每个 KV 头对应 4 个 Q 头，融合矩阵的输出维排列是：

$$\underbrace{q_1 q_2 q_3 q_4}_{\text{group 1}},\, k_1,\, v_1,\, \underbrace{q_5 q_6 q_7 q_8}_{\text{group 2}},\, k_2,\, v_2,\, \ldots$$

这个布局**不是任何一个简单维度上的 concat**，它要求先 reshape 再 transpose 再 flatten。这就是为什么每个模型族都需要自己的转换器，而不能用一个通用规则解决。

### 6.2 gate 让事情更复杂

`QKVGMapping` 处理的是带额外 scalar gate（`g_proj`）的变体。文档说明 Megatron 一侧有两种布局：

> For regular QKV modules, gate rows are appended after the standard GQA-interleaved QKV block. For Megatron-Core `attention_output_gate` modules, the scalar gate is expanded across each head dimension and interleaved as the module's output-gate block.

**同名的 `linear_qkv`，在两个不同的 attention 模块里可能是两种完全不同的布局**。这意味着映射表必须知道「当前用的是哪个 provider」，这也是 Bridge 里 `AutoMapping` 存在的理由——它按实际模块形状反推该用哪种映射，而不是让用户指定。

`QKVGMapping` 还有一条明确的限制：

> QKVGMapping does not support bias tensors; add_qkv_bias must be False.

不支持 bias 是显式的 `NotImplementedError`，而不是静默算错。这类「明确拒绝」在转换层里非常重要，因为静默错误会一路传到训推一致问题上，查起来要花几天。

### 6.3 一个可以直接抄的判据

把上面的内容压缩成一句话：**如果 HF 侧的参数能通过「沿单一维度 concat/split」还原成 Megatron 侧，就用基础映射；一旦涉及 reshape + 交错，就必须写专用映射。**

按这个判据，需要专用映射的只有三处：**QKV（含 gate）**、**gate_up（MLP 的融合）**、以及**MoE 的专家参数**（因为专家在 EP 下跨 rank 分布，名字与布局都不规则）。

---

## 7. 自动切分的三个静默边界

### 7.1 边界一：`num_query_groups` 不能被 TP 整除

GQA 模型的 KV 头数通常很小（Llama-3-8B 是 8、Qwen2.5-7B 是 4、更小的模型是 1 或 2）。而 TP 通常是 8。

- $$n_{\rm query\_groups} = 8$$、TP = 8：每卡 1 个 KV 头，正好
- $$n_{\rm query\_groups} = 4$$、TP = 8：**每卡 0.5 个 KV 头**
- $$n_{\rm query\_groups} = 2$$、TP = 8：**每卡 0.25 个 KV 头**

Megatron 的 `ColumnParallelLinear` 在 QKV 上切分时，切的是「输出维」，而输出维的构成是 $$n_{\rm heads} \cdot d_{\rm head} + 2 \cdot n_{\rm query\_groups} \cdot d_{\rm head}$$。当 TP 大于 $$n_{\rm query\_groups}$$ 时，一种做法是让每个 KV 头被复制到多个 rank 上（**KV 头重复**），另一种是直接报错。

**问题在于「重复」这件事不会报错**。它会带来两个后果：

1. 显存里 KV 头被重复存放，TP 越大冗余越多
2. **推理端（SGLang / vLLM）如果按「KV 头数 = 全局头数 / TP」去理解，恢复出来的布局会与训练端不同**

第二条正是 RL 训推不一致的一个真实来源。排查时的判据是：**看 SGLang 侧收到的 KV 权重是否在多个 rank 间完全相同**。

### 7.2 边界二：`sequence_parallel` 对激活切分的隐含要求

序列并行（SP）切的是**激活值的序列维**，而不是权重。它的前提是：注意力和 MLP 里的 LayerNorm 是沿 hidden 维做的，所以序列维可以自由切分而不影响结果。

但有两个前提容易被忽略：

1. **只对 LayerNorm 后的激活切**。LayerNorm 之前的激活如果被切，算子就需要跨 rank 的统计量；
2. **TP 通信要改成 RS ∘ AG**。SP 打开后，列并行层之后的 all-gather 变成 reduce-scatter、行并行层之前的 all-reduce 变成 all-gather。

关于 SP 有一个已核正的事实值得重申：**SP 不降低 TP 通信总量**，因为 AR = RS ∘ AG，总量守恒。它的收益在**激活显存**——省下来的显存能让长序列关掉重计算，那才是延迟的来源（见 `megatron-03`）。

回到本文的语境：**「激活值切分」这个说法在 Megatron 里对应的是 SP，而 SP 是否生效取决于注意力实现是否走 `sequence_parallel=True` 的路径**。如果某个 fused kernel 只在 `sequence_parallel=False` 下可用，打开 SP 会静默地走进另一条实现，性能与数值都变。

### 7.3 边界三：embedding 与 LM head 共享权重

Tie word embeddings 的模型（Qwen 小模型、Gemma 系列）里，输入 embedding 与输出 LM head 是同一块权重。而这两者在 Megatron 里的并行语义不同：

- 输入 embedding 是**列并行**（按 vocab 切）
- LM head 的输出投影是**列并行**（按 vocab 切，但输出的是 logits）

它们能共享，前提是 **vocab 能被 TP 整除**。当 vocab = 151936、TP = 8 时 151936 / 8 = 18992 正好整除；而 vocab = 151936、TP = 16 时也是整除。但很多自定义词表不满足，此时要么 pad 词表要么改用分开的 embedding。

**pad 词表会污染 logprob**：多加的 pad 位置在 softmax 里虽然是 0，但 vocab 维变大后归一化的浮点误差分布会变。对 RL 来说这是可测量的一阶差异，不是可以忽略的小数。

---

## 8. 切完怎么落到卡上

### 8.1 初始化必须各 rank 各自做

一个常见的错误心智模型是：主 rank 建好完整模型，再按 TP 切开分给各卡。**这个模型在 Megatron 里是错的**。

正确的顺序是：

```text
每个 rank
   ├─ 设置 TP / PP / DP / EP / CP 的世界大小与 rank（set_*_world_size）
   ├─ 用【带 rank 的随机种子】初始化本地分片形状的张量
   └─ 建通信组
```

**为什么必须带 rank 种子**：如果所有 rank 用同一个种子初始化「自己的分片」，那么切分后的各分片之间会存在不该有的相关性——因为同一段随机序列的前 N 个元素在每张卡上都被用了一次。带 rank 的种子（通常是 `seed + tp_rank * 因子 + pp_rank * 因子`）保证每个分片拿到的都是独立随机数。

这个细节只在**从头训练**时影响收敛，加载预训练权重时根本看不出来——所以它属于「只在特定场景咬人」的那类设计，一旦咬到就很难归因。

### 8.2 加载：`--load` 的 reshard 路径

从 checkpoint 加载时，如果**保存时的并行度与当前的并行度不同**，需要一次重切（reshard）。这是 RL 框架里最常踩的一步，因为：

- 预训练 checkpoint 通常是 TP=8、PP=1
- RL 训练可能要用 TP=4、PP=2
- 推理引擎又可能是 TP=8

三次转换的规则不同：

| 转换 | 能不能直接做 | 需要什么 |
| --- | --- | --- |
| TP 8 → TP 4 | 可以 | 每卡把两个相邻分片 concat 起来 |
| TP 4 → TP 8 | 可以 | 每卡把分片 split 后 scatter |
| PP 1 → PP 2 | 可以 | 按层号范围切分，**但需要每卡都知道全量权重** |
| PP 2 → PP 1 | 可以 | 收集两段再拼接 |

**PP 的转换成本远高于 TP**：TP 转 PP 时，每个 rank 手上只有一部分层，必须先把完整模型 gather 到某个位置再按层切分。这就是为什么 RL 框架通常固定训练的 PP，只让 TP 变化。

### 8.3 一个判断「落没落到正确位置」的检查

无法直接观察显存里每个参数属于哪个 rank，但可以间接验证：**加载完成后做一次前向，比较损失**。

- 单卡完整模型算出的 loss 与多卡切分算出的 loss 应该**在合理浮点误差内相等**
- 如果差异在 1e-3 量级以上，几乎一定是切分错了（通常是 QKV 交错顺序错，或者 row/column 弄反）

这个检查成本很低，值得固化到任何框架的启动流程里。

---

## 9. RL 场景：这条链路被整段绕过

### 9.1 Megatron 自己知道 RL 不用 dataloader

`build_train_valid_test_data_loaders` 里有一段专门为 RL 准备的分支：

```python
if args.perform_rl_step:
    # we don't need to build any dataloaders for RL training
    train_dataloader = None
    valid_dataloaders = None
    test_dataloader = None
    do_train = (args.train_iters or 0) > 0
    do_valid = (args.full_validation or args.eval_iters > 0)
    do_test = (args.full_validation or args.eval_iters > 0)
```

也就是说：**在 RL 训练里，前面第 2、3 节讲的三层数据切分完全不起作用**。RL 的数据是 rollout 产出的，由外部框架（slime、veRL）构造好后直接送进 `train_step`。

同一段代码在优化器那边也有对应分支：

```python
has_rl_optimizer = args.perform_rl_step and not args.no_load_optim
```

注释解释了原因：*the Forward pre-hooks are part of the distributed optimizer's overlapped param-gather; so in order to disable them, we must check that the optimizer actually exists.*——**RL 场景下即使 `--skip-train`，优化器仍然存在**，因为它要参与权重更新的前置工作。这个细节如果不看注释，几乎不可能从代码推断出来。

### 9.2 真实分工边界

| 层 | 谁负责 | Megatron 是否参与 |
| --- | --- | --- |
| 数据切分（第 2、3 节） | RL 框架 | **不参与** |
| 模型转换（第 4~6 节） | Bridge / RL 框架的转换器 | 提供目标布局与并行层 |
| 并行切分（第 7 节） | Megatron | 完全负责 |
| 落到卡上（第 8 节） | Megatron | 完全负责 |
| 权重导出给推理引擎 | RL 框架 | 提供 `megatron_to_hf` 方向的转换 |

结论很清楚：**在 RL 栈里，Megatron 承担的是「并行执行与优化器」这一层，前面的数据链路整段由 RL 框架接管，后面的权重导出由 RL 框架对接。** 这也解释了为什么 slime 只需要 `megatron_to_hf/<model>.py` 一个方向的重活——它不需要 Megatron 的数据管线。

---

## 10. 小结：Megatron 在 RL 栈里的真实位置

### 10.1 四件事的答案

| 问题 | 答案 |
| --- | --- |
| 数据怎么进来 | 三层：索引采样 → 打包成 `seq_len` → sampler 按 DP rank 分。RL 下整段绕过 |
| HF 模型怎么改 | 三样：并行层替换、布局改写成 GQA 交错、可逆映射表 |
| 自动切分怎么做到、有没有问题 | 沿 dim 0 / dim 1 切，代数上精确等价；三个静默边界在 GQA、SP、tied embedding |
| 怎么落到卡上 | 各 rank 带 rank 种子各自初始化；加载时按 TP / PP 重切，PP 的成本高一个量级 |

### 10.2 给选型的一句话

**如果你的模型族在 Bridge 的支持列表里，用 Bridge；如果不在，就要自己写 `megatron_to_hf` 转换器——工作量集中在 QKV 交错与 MoE 专家两块。**

这也解释了为什么 slime 的模型支持面窄：它没有 Bridge 那样的自动映射表，只有逐模型的转换器文件。Bridge 用「模式匹配 + 自动推断」换广度，slime 用「显式逐模型代码」换确定性。两种做法在下一篇分别代表了两个极端。

---

**上一篇**：[Megatron-LM 深度剖析（16）：混合与 MoE 模型装配](/2026/10/14/megatron-16-hybrid-and-moe-model-assembly/) ｜ **下一篇**：[02 slime 源码拆解](/2026/10/24/rl-frameworks-02-slime/)

## 参考

- Megatron-LM：<https://github.com/NVIDIA/Megatron-LM>，基准 `core_v0.19.2` = `4b4acac9a1d28ea6829c8d4f566d75698a21249d`
- Megatron-Bridge：<https://github.com/NVIDIA-NeMo/Megatron-Bridge>，基准 `v0.6.2` = `c0e164ed2aedac4ad1c877780e2564a19d5d54ec`
- 数据迭代器：[`megatron/training/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/training/training.py#L4601-L4690)
- RL 分支：[`training.py` L4553-4560](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/training/training.py#L4553-L4560)
- sampler 构造：[`megatron/training/datasets/data_samplers.py`](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/training/datasets/data_samplers.py#L19-L58)
- 样本构造：[`megatron/core/datasets/gpt_dataset.py` L230-L300](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/datasets/gpt_dataset.py#L230-L300)
- 映射注册表：[`mapping_registry.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/conversion/mapping_registry.py)
- 列 / 行并行映射：[`param_mapping.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/conversion/param_mapping.py#L919-L935)
