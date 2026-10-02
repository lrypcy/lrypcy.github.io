---
title: "性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准"
date: 2026-09-29 23:55:00 +0800
categories:
  - 性能分析
tags: [profiling, mfu, hfu, flops, timer, goodput, otel, megatron, nccl-tests, methodology, performance]
layout: post
mathjax: true
---

> **系列导航** ｜ 工具总纲：[NV 卡性能剖析工具全景：从 nvidia-smi 到 Nsight Compute](/2026/08/26/nv-gpu-profiling-toolkit/) ｜ 本系列：**00 度量口径** → [01 训练栈 profiling](/2026/09/29/profiling-01-training/) → 02 推理服务 profiling
>
> 前置总纲回答的是「有哪些工具、各自回答什么问题」。从本篇起进入下一层：**这些工具报出来的数字是怎么算出来的，以及哪些数字彼此不可比。**

**TL;DR**

> * **同一行训练日志，可以合法地报成 20.2% 或 128.2% 的 MFU。** 差异全部来自分母：用的是稠密还是带 2:4 稀疏的规格、是 BF16 还是 FP8、是不是同一代卡。MFU 不是一个数，是「分子口径 + 分母口径 + 硬件」的三元组；报 MFU 不写分母，等于没报。
> * **官方规格页自己就有排版陷阱。** [A100 产品页](https://www.nvidia.com/en-us/data-center/a100/)写作「312 TFLOPS \| 624 TFLOPS\*」（稠密在前、带星号的是稀疏）；[H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)只给「1,979 TFLOPS\*」，稠密 989.5 要靠脚注的一半推出来。照抄首页数字当分母，MFU 直接腰斩。
> * **MFU 与 HFU 的差别只有一处：重计算的 FLOPs 算不算数。** 开满 activation recompute 时每 token 从 6P 变成 8P，HFU = MFU × 4/3。**Megatron 官方日志里的 `throughput per GPU (TFLOP/s/GPU)` 是 MFU 口径**——它的分子收尾在 `training.py:1007` 的 `return flops_fwd * 3`，不含重计算系数，所以开不开 recompute 这个对数都不变。想评价「重计算划不划算」得自己换算。
> * **内置 `Timer` 测的不是「阶段花了多久」，而是「这个阶段排空所有流花了多久」。** `megatron/core/timers.py:143` 用的是 `torch.cuda.synchronize()` + `time.time()`，不是 CUDA Event。同步点会把别的流上还在飞的工作一起算进区间——**重叠做得越好，读数膨胀越大**。实测：通信占计算 1/4 时，逐 chunk 同步计时的读数比真实时间线高 85%。
> * **多 rank 的 min/max 不可加。** `timers.py:318` 的 `_get_global_min_max_time()` 分别取各阶段的最小与最大，**min 与 max 可能来自不同的卡**。短板固定时「各项 max 相加」碰巧等于最慢卡；短板在各阶段轮转时，相加结果比最慢卡的真实总时长高 **29.6%**——那是一个没有任何卡跑出过的数。
> * **`filter(x > 0.0)` 会静默丢 rank。** 没进过该 timer 的 rank 被过滤掉不报，于是 min 变成「活跃子集里的最快」。过滤本身合理，但它不告诉你少了谁。
> * **goodput 与 MFU 是两个正交的量。** 把 checkpoint 间隔从 2000 步缩到 500 步，有效吞吐掉 6.2%，而 goodput 几乎不动（90.56% → 91.15%）——因为 checkpoint 本身被标了 `is_goodput_span=True`，算正事。**只报 MFU 会漏掉「算得快但整天在等数据」，只报 goodput 会把「时间都在算、但算得很慢」当成健康。**
> * **Megatron v0.20 已经有原生 OTel 观测栈**（`megatron/core/telemetry/`，版权头是 2026）：8 个 span group 预设 + 8 个训练指标 + goodput 标记，开关是 `--otel-enable` / `--otel-service-name` / `--otel-span-groups`，环境变量前缀 `MEGATRON_OTEL`（回退到 `NEMO_LENS`）。
> * **本篇所有数字都来自三个纯 CPU notebook**，输出已固化、可直接复算：[Timer 计时语义](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-timer-semantics/profiling-00-timer-semantics.ipynb)（同步式计时语义）、[MFU/HFU 口径](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-mfu-accounting/profiling-00-mfu-accounting.ipynb)（分子分母口径）、[goodput 与 MFU](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-goodput/profiling-00-goodput.ipynb)（两者的分工），FLOPs 记账复用 `tools/megatron_bench/flops.py`。

---

## 1. 为什么口径要排在工具前面

把 GPU 剖析的工具链条摆清楚之后，下一个卡住人的问题通常不是「该用 nsys 还是 ncu」，而是**同一个东西为什么有两套数**。

三个具体现场：

- 一份训练日志写着 `throughput per GPU (TFLOP/s/GPU): 400.0`。换成 MFU 是 40% 还是 20%？回答取决于分母取稠密 BF16 峰值还是带稀疏的规格数字。
- 一次优化的复盘写着「本阶段耗时从 1.9 s 降到 1.2 s」。这个「阶段耗时」是某个 timer 报出来的，而那个 timer 每次 `stop()` 都做了设备同步——它测的到底是这个阶段的工作量，还是包含别的流排空的时间？
- 一篇论文报 MFU 46.2%，另一篇报 38%。这两个数能不能直接比？取决于双方的分子含不含重计算、分母用的什么精度。

这三件事都不是「换个工具就能解决」的问题。**工具负责采集，口径负责解释；工具报数很快，口径错了后面全部白干。**

所以本系列的第一篇不讲工具——总纲篇已经讲完了——只做一件事：把尺子做准。

本篇带三个可复算的 notebook，全部纯 CPU，**输出已固化，点开就能看到跑出来的数**：

| Notebook（GitHub 上可直接读 / 运行） | 它回答的问题 |
|:---|:---|
| [MFU/HFU 口径实验](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-mfu-accounting/profiling-00-mfu-accounting.ipynb) | 分子换一下、分母换一下，MFU 差多少 |
| [Timer 计时语义](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-timer-semantics/profiling-00-timer-semantics.ipynb) | 内置 timer 报的是「阶段耗时」还是「排空耗时」 |
| [goodput 与 MFU 的分工](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-goodput/profiling-00-goodput.ipynb) | goodput 和 MFU 为什么会给出相反的结论 |

> notebook 里的代码是从本仓库 `tools/profiling_bench/` **逐字抽取**的（用 `ast` 按顶层符号抽，不是手抄），所以正文引用的数字与 notebook 的输出是对得上的。仓库内的脚本是同一份源码，命令行跑法见文末 Lab Exercises。

FLOPs 记账复用 `tools/megatron_bench/flops.py`（已有三种口径的交叉验证实现），不重复造。

---

## 2. MFU 与 HFU：分子和分母都必须写清楚

### 2.1 定义

Model FLOPs Utilization 由 PaLM 报告引入（[arXiv:2204.02311](https://arxiv.org/abs/2204.02311)），用来回答「我花了这么多卡的峰值算力，有多少真的用在模型定义的算术上」：

$$\text{MFU} = \frac{\text{模型定义所需的 FLOPs} \div \text{时间}}{\text{硬件峰值 FLOPs}}$$

分母是硬件峰值，分子是**模型定义**的 FLOPs。这个「模型定义」是关键限定：它只算实现无关的算术，不计任何实现引入的冗余。

Hardware FLOPs Utilization 换掉了分子：

$$\text{HFU} = \frac{\text{真正发射到设备上的 FLOPs} \div \text{时间}}{\text{硬件峰值 FLOPs}}$$

于是 **HFU 恒大于等于 MFU**，差额就是实现引入的冗余——主要来自 activation recompute。

### 2.2 分母：官方规格页自己就排版不一致

先看两份官方产品页，注意数字的排列方式：

| 来源 | BF16 Tensor Core | 脚注 |
|:---|:---|:---|
| [A100 产品页](https://www.nvidia.com/en-us/data-center/a100/)（80GB SXM） | `312 TFLOPS \| 624 TFLOPS*` | `* With sparsity` |
| [H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)（SXM） | `1,979 TFLOPS*` | `* With sparsity` |

A100 页把**稠密值放在前面**（312），带星号的 624 是稀疏；H100 页**只给了一个带星号的 1979**，稠密值 989.5 要靠「一半」推出来。两份页面都合规、都写清了脚注，但读者的默认解读完全不同：从 A100 页抄下 312 的人拿到了稠密值，从 H100 页抄下 1979 的人拿到了稀疏值。

**这不是文档错误，是阅读陷阱。** 而分母乘 2 或除 2，MFU 就乘 2 或除 2。

### 2.3 实测：同一行日志，六个分母

固定分子（llama3-8b，序列 8192，每 token 57.91 GFLOP，不含重计算），固定日志读数 400 TFLOP/s/GPU，只换分母：

```
实验一  分母口径：同一行 log，换个分母就换一个 MFU
模型 llama3-8b，序列 8192，每 token 57.91 GFLOP（不含重计算）
log 读数 = 400 TFLOP/s/GPU  ->  折合 6907 tokens/s/GPU
分子固定不变，只换分母：

  分母（TFLOPS）                         MFU
  A100 80GB SXM · BF16 稠密               128.21%
  A100 80GB SXM · BF16 稀疏 2:4            64.10%
  H100 SXM · BF16 稠密                     40.42%
  H100 SXM · BF16 稀疏 2:4                 20.21%
  H100 SXM · FP8 稠密                      20.21%
  H100 SXM · TF32 稠密                     80.89%

  最大 / 最小分母之比 = 6.34x -> 同一行 log 可以报成 20.2% 或 128.2%
```

跨了 6.34 倍。这个跨度全部由分母贡献，分子一个 bit 都没变。

顺带一个副产品：A100 那两行给出了 **128.21%** 的「MFU」。这个数本身就是一个诊断——400 TFLOP/s/GPU 已经超过 A100 稠密 BF16 的峰值 312，所以**这行日志不可能来自 A100**。跨硬件套错分母，得到的是物理上不存在的利用率。反过来说，如果你的换算表给出了超过 100% 的 MFU，先怀疑分母而不是怀疑代码。

### 2.4 分子：6P 的来历，以及 attention 项为什么不能省

对稠密 Transformer，每参数的训练 FLOPs 常被近似成 6N（tokens 维度上是 6P/token）：

- 前向：每个参数在每个 token 上做一次乘加，即 2 FLOPs
- 反向：关于**输入**的梯度 GEMM 和关于**权重**的梯度 GEMM，各与一次前向同量级，即 4 FLOPs

合计 6 FLOPs/参数/token。attention 的 core 部分另算：

$$\text{FLOPs/token} \approx 6P + 12 \, n_{\text{layer}} \, d_{\text{model}} \, S$$

第二项随序列长度线性增长。注意一个反直觉的约定：**causal mask 实际只算下三角，但口径必须按整张 $$S \times S$$ 矩阵计**。这样不同实现、不同论文之间才可比——FlashAttention 省掉的是真实计算量，不是口径上的记账量。

6ND 近似随序列失真的程度，用同一个模型扫出来：

```
实验三  分子口径的来源：6ND 近似随序列变长而失真
  序列长度   精确(含 lm_head)      6ND        偏差
      1024      45.83 GFLOP      41.88 GFLOP     +9.45%
      2048      46.64 GFLOP      41.88 GFLOP    +11.37%
      4096      48.25 GFLOP      41.88 GFLOP    +15.22%
      8192      51.47 GFLOP      41.88 GFLOP    +22.91%
     32768      70.80 GFLOP      41.88 GFLOP    +69.07%
    131072     148.11 GFLOP      41.88 GFLOP   +253.68%
```

6ND 把 $$S^2$$ 项当常数忽略掉了，序列越长这一项越不能忽略。所以「同一个模型在长序列下的 MFU 变低了」这件事，可能根本与性能无关——是分子换了口径。

### 2.5 HFU：差额就是重计算的账单

activation recompute 用「重算一次前向」换取「不保存激活」。多算的那部分在 HFU 里算数、在 MFU 里不算数：

```
实验二  分子口径：MFU 不变而 HFU 变，差额就是重计算的代价
分母统一用 H100 稠密 BF16 = 989.5 TFLOPS；log 读数 400 TFLOP/s/GPU

  策略                 每 token FLOPs     MFU      HFU     HFU/MFU
  不重计算                    57.91 GFLOP   40.42%   40.42%    1.000
  full recompute          76.17 GFLOP   40.42%   53.17%    1.315
  selective (core attn)    62.21 GFLOP   40.42%   43.42%    1.074

  full recompute 时 解析比值 = 1.3152（= 4/3，每 token 从 6P 变 8P）
  selective 时     解析比值 = 1.0742
```

full recompute 的解析比值恰好是 4/3：每 token 从 6P 变成 8P，$$\text{HFU} = \text{MFU} \times \tfrac{4}{3}$$。开满重计算时，**HFU 会凭空高出 31.5%**——这解释了为什么现场常常同时出现「MFU 只有 40%」和「HFU 有 53%」两个都正确的说法。

### 2.6 已发表的锚点

作为量级参照（均为原论文口径，引用时请注意分母是各自的硬件）：

| 模型 | 规模 | 报告的 MFU | 来源 |
|:---|:---|:---|:---|
| GPT-3 | 175B | 21.3% | 公开汇总口径，待与原论文核对 |
| MT-NLG | 530B | 30.2% | 同上 |
| Gopher | 280B | 32.5% | 同上 |
| PaLM | 540B | 46.2% | [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) |
| Llama 3 | 405B | 38–43%（8,192 卡 43%，16,384 卡 41%） | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |

工程上的经验区间是大模型训练落在 30–50%，低于 25% 通常指向配置问题（气泡过大、TP 跨机、micro-batch 过小、checkpoint 开销）。**但这句话必须配上「分母是稠密值、精度与训练精度一致」的前提**，否则区间本身就失效。

另一个常被误用的判断：decode 阶段 MFU 极低是**正常的物理结果**，不是故障。逐 token 生成时瓶颈在读权重和 KV cache，不在算术，所以分子本来就小。拿训练时的 MFU 区间去衡量推理服务，是范畴错误。

### 2.7 一个反直觉的代际效应

换新卡时，峰值算力涨得比带宽快，结果 MFU 反而下降：

```
实验四  代际算术强度门槛：算力涨 3.2x，带宽只涨 1.6x
  稠密 BF16 峰值:  312.0 -> 989.5 TFLOPS   = 3.17x
  HBM 带宽:        2039 -> 3350 GB/s     = 1.64x

  机器平衡点（峰值算力 / 峰值带宽，单位 FLOP/Byte）:
    A100 80GB SXM: 153.0 FLOP/Byte
    H100 SXM     : 295.4 FLOP/Byte
    门槛上升     : 1.93x
```

（数据取自两份官方产品页，同为 80GB SXM 档、同取稠密 BF16。）

含义：一段算术强度固定为 X 的子层，在 A100 上 X 超过 153 就算计算受限，在 H100 上要超过 295。**原本在 A100 上勉强算计算受限的层，换到 H100 就滑回了访存受限**；norm、softmax、优化器 step 这些天然低算术强度的部分尤其如此。

所以「换卡之后吞吐上去了，但日志里的 TFLOP/s/GPU 没有同比例上涨」是正常的。分子没变、分母涨得更快，MFU 必然下降，这不是实现退化。

---

## 3. Megatron 的 FLOPs 记账：读源码确认口径

口径不能靠猜，要对着源码确认。基准是 v0.20.0（commit `60e039626`，2026-08-21 main）。

### 3.1 逐架构族的记账

`megatron/training/training.py:802` 的 `num_floating_point_operations()` 是唯一的 FLOPs 入口，内部按架构族分成若干闭式子函数：

| 子函数 | 覆盖的层型 | 关键约定 |
|:---|:---|:---|
| `attn_layer_flops` | 标准 attention | 分成 token 线性项（QKV + 输出投影）与 $$S^2$$ 项（QK^T 与 AV 两次 GEMM） |
| `mlp_layer_flops` | 稠密 MLP | SwiGLU 时乘 1.5（三个投影），否则 4 倍扩展 |
| `moe_layer_flops` | MoE | routed 专家按实际路由数算；有 `moe_latent_size` 时额外计 up/down 投影；shared expert 单独算 |
| `mamba_layer_flops` | Mamba | in_proj + scan + out_proj 三段 |
| `gated_delta_product_layer_flops` | GDP | 明确写清是「best-case recurrent 估计」，比 chunk kernel 实际做的事少 |
| `gdn_layer_flops` | Gated DeltaNet | 区分 GDN 与 GDN2 的 in_proj 维度 |
| `hybrid_flops` | 混合架构 | 各层型按数量加权，末尾加 logits 项 |

两个值得注意的地方：

- **它已经能算混合架构**（attention + Mamba + GDP + GDN + MoE 混合），这是 2026 年的架构现实倒逼出来的；
- GDP 那条注释很诚实：`This keeps the implementation-agnostic lower bound explicit.` 也就是说这个数是**下界**，真实 kernel 做的比它多。看到 MFU 偏低估时，先回来看看是不是分子记少了。

### 3.2 收尾那一行决定了它是 MFU 还是 HFU

```python
# megatron/training/training.py:991-1007（节选）
flops_fwd = (
        num_attn_layers * attn_layer_flops(...) +
        num_mlp_layers * mlp_layer_flops(...) +
        ...
        (2 * total_tokens * hidden_size * vocab_size * (1 + mtp_num_layers))  # logits
)
return flops_fwd * 3
```

**`flops_fwd * 3` 里没有重计算系数。** 前向一次、两个反向各一次，正好三倍——这就是「模型定义」的记账，与 recompute 开不开无关。

于是可以直接下结论：**`--log-throughput` 打出来的 `throughput per GPU (TFLOP/s/GPU)` 是 MFU 口径。** 想拿 HFU，得自己按 recompute 策略乘上去（full recompute 时乘 4/3，selective 要按具体重算的层型算）。

推论：**用这个数去评价「recompute 值不值」是问错了指标。** 它天然对 recompute 无感。

### 3.3 THD 路径：打包训练会让分母变小

v0.20 里新增了一条 THD（packed sequence）路径，函数签名多了两个可选参数：

```python
def num_floating_point_operations(
    args, batch_size,
    seqlen_squared_sum_in_batch=None,   # sum_i(L_i ** 2)，只算真实（未 padding）子序列
    total_real_tokens_in_batch=None,    # sum_i(L_i)，只算真实 token
):
```

两个参数合起来做一件事：**让 padding 与因果分块不再进入 FLOPs。** 文档字符串写得很清楚——THD 下 `seqlen_squared_sum_in_batch` 严格小于 BSHD 值，因为「padding token 不产生 attention 分数」。

这意味着：**同一个模型、同样一批数据，换成打包训练后 MFU 会变，而变的不是性能而是分母。** 打包后分母变小（不再为 padding 记账），于是给定的实际吞吐反算出的 MFU 会变。拿「打包前 vs 打包后」的 MFU 直接做对比，是把两个口径放在一起比。

### 3.4 一次 forward 的构成

把单层的量按层数折算、再把只算一次的 lm_head 单独拎出来：

```
实验五  一次完整 forward 的 FLOPs 构成：不是所有 FLOPs 都长成 GEMM 的样子
模型 llama3-8b（32 层），序列 8192，每 token 口径

  组成                                    每 token FLOPs   占 forward
  QKV 投影 × 32 层                               1610.61 MFLOP       8.3%
  attention core (QK^T + AV) × 32 层           4294.97 MFLOP      22.2%
  输出投影 W_o × 32 层                             1073.74 MFLOP       5.6%
  MLP (SwiGLU 三个投影) × 32 层                   11274.29 MFLOP      58.4%
  lm_head (vocab 投影，只算一次)                     1050.67 MFLOP       5.4%
  合计                                         19304.28 MFLOP     100.0%
```

两点：

1. **8192 序列下 attention core 已经占到 forward 的 22.2%**，不再是「可以忽略的二次项」。序列再长它就成主角。
2. attention core 里有一半是 softmax，是 elementwise 而不是 matmul。MFU 的分子只算 matmul，这部分工作量在分子里体现不出来，但在真机上要花时间。**分子计入得不完整，是 MFU 天然低于 100% 的构成性原因之一。**

### 3.5 三种口径的交叉验证

`tools/megatron_bench/flops.py` 用三种口径同时对同一个模型求解并摊开偏差：

```
### FLOPs 记账 —— llama-7b
每 token forward FLOPs（逐算子）:
    QKV 投影                         100.66 MFLOP
    attention core (QK^T + AV)     67.11 MFLOP
    输出投影 W_o                       33.55 MFLOP
    MLP (SwiGLU 三个投影)              270.53 MFLOP
    lm_head (vocab 投影)             262.14 MFLOP
    合计（含 lm_head）                  734.00 MFLOP

训练每 token（backward = 2x forward）:
    total_none                     46.08 GFLOP
    total_selective_core_attn      48.23 GFLOP
    total_full_recompute           61.18 GFLOP

精确口径 vs 6ND 近似:
    6ND                        38.86 GFLOP
    精确                       46.08 GFLOP
    相对偏差                   +18.60%
```

三种口径（逐算子精确 / 复刻官方公式 / 6ND）的偏差不是要藏掉的噪声，**它本身就是结论**：18.6% 的差距足够让一次「优化」的收益看起来存在或不存在。

> **实践建议**：任何 MFU 数字旁边都应该写三样东西——分子含不含重计算、分母是哪个精度哪种稀疏假设、是不是打包训练。三样缺一样，这个数就只在自己的上下文里有效。

---

## 4. 计时器在骗人

这一节是最容易踩、后果最严重的一节。结论先给：**内置 timer 报的不是「这个阶段花了多久」，而是「这个阶段排空所有流花了多久」。**

### 4.1 源码：同步式计时

`megatron/core/timers.py` 的 `Timer` 实现（v0.20）：

```python
# megatron/core/timers.py:143-167（节选）
def start(self, barrier=False):
    if barrier:
        torch.distributed.barrier(group=self._barrier_group)
    torch.cuda.synchronize()      # <- 注意这里
    self._start_time = time.time()
    self._started = True

def stop(self, barrier=False):
    if barrier:
        torch.distributed.barrier(group=self._barrier_group)
    torch.cuda.synchronize()      # <- 和这里
    elapsed = time.time() - self._start_time
    self._elapsed += elapsed
    self._active_time += elapsed
    self._started = False
```

三个细节：

1. 用的是 **`torch.cuda.synchronize()` + `time.time()`**，不是 CUDA Event。CUDA Event 记录的是**设备侧时间戳**，可以异步打点、不改变执行；`synchronize()` 是**等所有流排空**，会改变执行。
2. `time.time()` 是墙钟，不是 `time.perf_counter()`。单机场景差别可忽略，但语义上它是「可被 NTP 调整的时间」。
3. `stop()` 里的 `torch.cuda.synchronize()` 会等**所有流**排空，不只是这个 timer 关心的那条流。

### 4.2 后果：越会优化，读数越难看

第 3 点是全部问题的根源。`stop()` 排空所有流，意味着**边界处还在飞的工作被算进了这个区间**。而「在飞的工作」正是你辛苦做重叠的那部分。

用真线程复刻这个语义（[Timer 计时语义 notebook](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-timer-semantics/profiling-00-timer-semantics.ipynb)）：两条流——compute 流逐 chunk 计算，comm 流的第 i 个 chunk 依赖 compute 流第 i 个 chunk 完成（1F1B 反向重叠的语义）。sleep 模拟设备侧耗时，chunk 时长 = 10 ms，共 8 个 chunk：

```
实验一  同步式计时对重叠的破坏（compute chunk = 10 ms，共 8 个 chunk）
真实时间线（一次发射到底）:
  chunk 时长 -> Tm= 0.0ms  Tm= 2.0ms  Tm= 5.0ms  Tm=10.0ms
  makespan   ->     91.6      96.0     100.2     106.7

同步式计时读数（各阶段求和）与膨胀率:
  [2 个阶段]
     求和     94.5      98.6     108.2     119.5
     膨胀       3%        3%        8%       12%
  [每 chunk 一个阶段（L 段）]
     求和     98.3     122.2     150.4     197.3
     膨胀       7%       27%       50%       85%

解析对照（L 段每段都同步）: 读数 = L*(Tc+Tm)，真实 = L*Tc+Tm
  Tm= 0.0ms -> 解析   80.0 / 实测   98.3   真实   91.6
  Tm= 2.0ms -> 解析   96.0 / 实测  122.2   真实   96.0
  Tm= 5.0ms -> 解析  120.0 / 实测  150.4   真实  100.2
  Tm=10.0ms -> 解析  160.0 / 实测  197.3   真实  106.7
```

读法（`Tm` 是通信 chunk 时长，占比越高重叠收益越大）：

- **真实时间线**（一次发射到底）随 `Tm` 只从 91.6 涨到 106.7——重叠把它们藏住了，符合预期。
- **同步式计时**在「每 chunk 都同步」这一档，从 98.3 涨到 197.3，**膨胀率 7% → 85%**。
- 膨胀量的解析式就是边界处被迫串行掉的通信：读数 = $$L(T_c + T_m)$$，真实 = $$L \cdot T_c + T_m$$，差 $$(L-1)T_m$$。

**这是一条与直觉相反的规律：重叠做得越好（$$T_m$$ 越大、越值得藏），内置 timer 的读数就膨胀得越厉害。** 优化的方向和指标的方向反了。

（实测绝对值比解析式系统性偏大，因为每个 chunk 有约 1.4 ms 的线程调度开销；所以看趋势和比值，不要看绝对值。开销本身在脚本里被显式打印出来，便于核对。）

**正确的做法**：要「阶段耗时」就用时间线口径——CUDA Event 打点、或者从 nsys / torch.profiler 的 trace 里读该区间的实际跨度。要「端到端步时」就只在外层打一对点，中间不要插同步点。

### 4.3 `reset()` 不清 `_active_time`

`timers.py:171` 的 `reset()` 只清 `_elapsed`：

```python
def reset(self):
    # Don't reset _active_time
    self._elapsed = 0.0
    self._started = False
```

而 `_active_time` 从 timer 构造起只增不减（`timers.py:212`）。于是两者的比值是一个**占空比**，但只在稳态才成立：

```
实验四  reset() 不清 _active_time：elapsed 与 active_time 的比是占空比
  迭代   elapsed(本次)   active_time(累计)   elapsed/active_time
     1          50.0 ms             50.0 ms           1.000   <- warmup 阶段
     2          50.0 ms            100.0 ms           0.500   <- warmup 阶段
     3          50.0 ms            150.0 ms           0.333   <- warmup 阶段
     4          50.0 ms            200.0 ms           0.250
    10          50.0 ms            500.0 ms           0.100
```

迭代 1 的比值是 1.000（100% 占空比），迭代 10 是 0.100。同一个数在早期迭代里完全没有意义。**要拿占空比，必须用稳态窗口内的差值，不能拿累计量直接除。**

### 4.4 多 rank 的 min/max 不是同一张卡

`timers.py:318` 的 `_get_global_min_max_time()`：

```python
for i, name in enumerate(names):
    # filter out the ones we did not have any timings for
    rank_to_time = list(filter(lambda x: x > 0.0, rank_name_to_time[i]))
    if len(rank_to_time) > 0:
        name_to_min_max_time[name] = (
            min(rank_to_time) / normalizer,
            max(rank_to_time) / normalizer,
        )
```

默认 `--timing-log-option minmax`，所以日志里每个 timer 报的是**各阶段在全部 rank 上的 min 与 max**。

**这带来了三种误读：**

**误读一：把各项 max 相加当成「总时长」。** min 与 max 各自可能来自不同的卡，相加得到的是一个「把所有卡的短板拼在一起」的虚拟机器。

```
实验二  多 rank 的 min/max 不是同一张卡，各项 max 相加没有物理含义
合成设定：8 rank × 6 阶段，基准 100 ms ± 20 ms

形态 A：掉队卡固定（rank 3 每阶段都慢 1.6×）
  各阶段 max 之和 =   1071.3 ms   最慢卡真实值 =   1071.3 ms   偏差    +0.0 ms (+0.0%)
  各阶段 min 之和 =    613.3 ms   最快卡真实值 =    650.1 ms   偏差   -36.8 ms (-5.7%)
  每个 max 落在哪张卡: [3, 3, 3, 3, 3, 3]

形态 B：短板每阶段换一张卡（+120 ms 轮转尖峰）
  各阶段 max 之和 =   1387.9 ms   最慢卡真实值 =   1071.3 ms   偏差  +316.7 ms (+29.6%)
  每个 max 落在哪张卡: [1, 4, 7, 2, 5, 0]
```

形态 A（掉队卡固定）下，各项 max 恰好都落在同一张卡上，相加**碰巧**等于最慢卡的真实总时长——偏差 +0.0%。形态 B（短板轮转）下，max 来自 6 张不同的卡，相加比最慢卡真实值高 **29.6%**。

**所以「把 max 加起来」既可能碰巧对，也可能系统性错**——取决于短板是否固定在少数几张卡上。而固定 vs 轮转恰恰是两种需要不同处置的故障：固定掉队要修那张卡，轮转抖动要修通信/调度。这个区分只能靠逐 rank 的完整数据，靠 min/max 两列看不出来。

**误读二：`filter(x > 0.0)` 静默丢 rank。**

```
实验三  filter(x > 0.0) 的静默丢弃：min 可能来自一个只干了一点活的 rank
  无过滤时按 min 选中的 rank = 0 （值 0.0）
  过滤 0 后按 min 选中的 rank = 2 （值 112.0）

  被过滤掉的 rank: [0, 1]
  过滤后仍在统计里的 rank: [2, 3, 4, 5, 6, 7]
```

过滤本身是合理的（没进过这个 timer 的 rank 不该报 0），**但它不告诉你少了谁**。于是 min 变成「活跃子集里的最快」，与「全体的最快」不是同一个量。想排查这类问题，得开 `--timing-log-option all` 拿到逐 rank 的原始值。

**误读三：忽略 `barrier=True` 的时间也被量进去了。**

`training.py:3503` 里 `interval-time` 的读取方式是：

```python
elapsed_time = timers('interval-time').elapsed(barrier=True, reset=should_reset)
```

`barrier=True` 会在读表前做一次全局 barrier——**即等待最慢的 rank 到达读表点**。这个等待发生在被测量区间之外还是之内要看实现细节，但语义上「这一步的耗时」已经包含了「等最慢的卡跟上」。对步时来说这是对的（步时本来就由最慢的卡决定），但把它当成「计算耗时」就错了。

### 4.5 何时该看哪个口径

| 想回答的问题 | 该用的口径 | 不该用的口径 |
|:---|:---|:---|
| 端到端步时、吞吐 | 外层一对点（timer 或墙钟） | 各类子阶段 timer 相加 |
| 某个阶段占了多少 | 时间线口径（Event / nsys / torch.profiler 区间） | 同步式 timer（会膨胀） |
| 有没有掉队卡 | 逐 rank 的全部数据（`--timing-log-option all`） | 只看 min/max 两列 |
| 通信藏住了没有 | overlap 百分比（见 01 篇）+ 时间线 | max 之和减去 compute 之和 |
| 占空比 | 稳态窗口内 elapsed 的差值 | 累计 `active_time` 直接做除 |

---

## 5. 配置开关：`--timing-log-level` 与 `--timing-log-option`

`Timer` 由全局 `Timers` 容器管理，开关定义在 `megatron/training/config/training_config.py:276`：

```python
timing_log_option: Literal["max", "minmax", "all"] = "minmax"
```

三个取值对应三种上报方式（`timers.py:318` / `338` / `357`）：

| 取值 | 上报内容 | 适合 |
|:---|:---|:---|
| `max` | 只报各阶段在各 rank 上的最大值 | 日常训练，关心最慢路径 |
| `minmax`（默认） | 报最小与最大 | 想看抖动幅度 |
| `all` | 报全部 rank 的逐项数值 | 排查掉队卡、验证 §4.4 的 min/max 误读 |

另一个开关是 `--timing-log-level`，`Timers` 构造时传入（`training/global_vars.py:332`）：

```python
_GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)
```

单个 timer 的日志级别由注册时的 `log_level` 决定（`timers.py:241` 的 `__call__`），只有级别不超过全局阈值的 timer 才会被打印。所以「某个 timer 不出现」不一定是它没被启用，可能是它的级别被过滤了。

还有 `timers.py:452` 的 `write()`，把计时结果落盘（配套 `--timing-log-file`），适合跑完之后再做分析而不是盯日志。

**实践建议**：日常用默认 `minmax`；一旦出现「整体变慢但每一项看起来都正常」，切到 `all` 把逐 rank 数据拉出来——这正是 §4.4 形态 B 的场景。

---

## 6. goodput：比 MFU 更贴近「有没有在干正事」

### 6.1 v0.20 的原生 OTel 观测栈

Megatron v0.20 新增了整个 `megatron/core/telemetry/` 包（版权头写的是 2026），把训练过程暴露成 OpenTelemetry 的 span 与 metric。三个文件：

| 文件 | 内容 |
|:---|:---|
| `telemetry/span_groups.py` | `MegatronSpanGroup`：span 分组定义与预设 |
| `telemetry/training_metrics.py` | 8 个 OTel 指标 |
| `telemetry/fallbacks.py` | 无 `nemo-lens` 时的全 no-op 实现 |

span 分组分两层。基础层与通用训练语义对齐：

```
job / checkpoint / evaluate / model_init / load_checkpoint / step / forward_backward / optimizer
```

Megatron 专有层细化到实现细节：

```
microbatch        每个微批的前向/反向
layer             每层 transformer 的前向（attention 与 MLP 分开）
communication     P2P send/recv 与梯度 AllReduce/ReduceScatter
activation_offload  GPU<->CPU 激活卸载与重载
data_loading      数据加载与 batch 准备
first_iteration   本进程实际执行的第一次迭代（含编译、CUDA graph 捕获、prefetch）
trace_region      与 perfetto 原生 trace_region 标记一一对应的影子 span（约 85 个）
inference         推理服务请求
```

预设四档：`default`（job / checkpoint / evaluate / first_iteration / inference）、`per_step`（加上 model_init、load_checkpoint、step、forward_backward、optimizer、communication、data_loading）、`profiling` 与 `all`（全开）。

指标侧 `training_metrics.py` 暴露 8 个：

```
megatron.training.step_duration_ms      直方图
megatron.training.loss                  gauge
megatron.training.throughput_tflops     gauge   <- 就是 §3.2 那个 MFU 口径的数
megatron.training.grad_norm             gauge
megatron.training.skipped_iters         counter
megatron.training.learning_rate         gauge
megatron.training.tokens_per_sec        gauge
megatron.training.memory_allocated_gb   gauge
```

启用方式（`training/global_vars.py:490` 的 `_set_telemetry`）：

```bash
python pretrain_gpt.py \
    --otel-enable \
    --otel-service-name my-train-job \
    --otel-span-groups per_step \
    ...
```

环境变量前缀是 `MEGATRON_OTEL`，回退到 `NEMO_LENS`。真正的实现在外部包 `nemo-lens` 里；没装时全部退化成 no-op，不会让训练挂掉。另外注意一个工程细节：`nvmlInit()` 只在 telemetry 真的启用时才调用，关掉时不会碰 NVML。

埋点的分布本身值得看一眼——它覆盖的正是训练栈里最容易出问题的地方：

| 文件 | 埋的是哪一段 |
|:---|:---|
| `training/checkpointing.py` | checkpoint 的 save / load |
| `training/training.py` | 训练循环与 startup（含 weight-hash check） |
| `core/distributed/distributed_data_parallel.py` | DP 梯度归约 |
| `core/pipeline_parallel/p2p_communication.py` | PP 的点到点收发 |
| `core/pipeline_parallel/schedules.py` | 1F1B 等调度 |
| `core/pipeline_parallel/fine_grained_activation_offload.py` | 细粒度激活卸载 |
| `core/transformer/transformer_layer.py` | 每层前向 |
| `core/ssm/mamba_layer.py` | Mamba 层 |

### 6.2 goodput 是怎么算的

关键在于 `managed_span(..., is_goodput_span=True)` 这个标记。被标的 span 算「正事」，其余不算。目前已标记的包括：

```python
# training/checkpointing.py:810
with _otel_managed_span('checkpoint', 'megatron.checkpoint.save.state_dict', is_goodput_span=True):
# training/checkpointing.py:896
with _otel_managed_span('checkpoint', 'megatron.checkpoint.save.io_write', is_goodput_span=True):
# training/checkpointing.py:2515
with _otel_managed_span('load_checkpoint', 'megatron.checkpoint.load.io_read', is_goodput_span=True):
# training/training.py:4482
with _otel_managed_span('job', 'megatron.startup.weight_hash_check', is_goodput_span=True):
```

于是：

$$\text{goodput} = \frac{\sum \text{计入 goodput 的时长}}{\text{挂钟时长}}$$

这与 MFU 是**两个正交的量**：MFU 说「算得快不快」，goodput 说「时间用在正事上没有」。

### 6.3 两个指标会给出不同的结论

用 [goodput notebook](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-goodput/profiling-00-goodput.ipynb) 把一次稳定迭代拆成 span（叶子 span 之和等于父 span，不要重复计）：

```
单个稳定迭代的 span 拆分（megatron.step 父 span = 1800 ms）：
  data_loading（等 batch）                           90.0 ms   否
  forward_backward（计算）                          1560.0 ms   是
  communication（overlap 之外漏出的部分）                  70.0 ms   否
  optimizer（step + grad clip）                     70.0 ms   是
  logging / bookkeeping                           10.0 ms   否
  叶子 span 合计                                    1800.0 ms

  迭代挂钟 1800.0 ms，其中计 goodput 的 1630.0 ms -> 迭代内 goodput 90.6%
  不计 goodput 的部分 = 170.0 ms（9.4%）
```

现在把 checkpoint 摊进来（checkpoint 的成本是 60 s，按不同间隔摊）：

```
  ckpt 间隔     每迭代摊到   含 ckpt 的迭代时长    goodput    有效吞吐(相对)
  不存                0.0 ms         1800.0 ms         90.56%      1.000
  2000 步           30.0 ms         1830.0 ms         90.71%      0.984
  1000 步           60.0 ms         1860.0 ms         90.86%      0.968
  500 步           120.0 ms         1920.0 ms         91.15%      0.938
```

**存得越勤，有效吞吐掉得越多（1.000 → 0.938，掉 6.2%），而 goodput 几乎不动，甚至微升。** 因为 checkpoint 本身被标成了 goodput span——它算正事。

这就是为什么两个指标必须一起看：

| 形态 | 现象 | 该查什么 |
|:---|:---|:---|
| MFU 高、goodput 低 | 计算很猛，但大量时间花在数据加载、通信漏出、日志上 | 查 `data_loading` 与 `communication` span，不是去换 kernel |
| MFU 低、goodput 高 | 时间都在 forward_backward 里，但每步内部访存受限 | 查算子与并行切分，不是去缩减非计算开销 |

反过来，把漏出的非 goodput 项压掉之后 goodput 的回升幅度也要正确解读：

```
量化形态一：把 data_loading 与 communication 的漏出压缩掉
  漏出项                        当前      压到 1/4 后
  data_loading                    90.0 ms      22.5 ms
  communication(漏出)               70.0 ms      17.5 ms
  迭代挂钟                       1800.0 ms    1680.0 ms
  goodput                          90.6%      97.0%
```

**分子没动**（这两项本来就不计 goodput），分母变小，goodput 自然回升 6.4 个百分点。goodput 的变化只说明「正事占比提高了」，**不能推出 FLOPs 利用率也同步改善**。

goodput 目前还没有统一口径——哪些 span 算正事是各家自定义的。上面这些 `is_goodput_span` 的选择就是 Megatron 的答案，读别人的 goodput 数字前先确认它的定义。

---

## 7. 实验卫生：让两次测量可比

口径对了，还要保证测量本身可比。

### 7.1 warmup 不是随便取的

官方 profiler 集成在 `megatron/training/training.py:4452-4458`：

```python
prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=max(args.profile_step_start - 1, 0),
        warmup=1 if args.profile_step_start > 0 else 0,
        active=args.profile_step_end - args.profile_step_start,
        repeat=1,
    ),
    on_trace_ready=trace_handler,
    record_shapes=args.pytorch_profiler_collect_shapes,
    with_stack=args.pytorch_profiler_collect_callstack,
    execution_trace_observer=et,
)
```

配套的命令行（`examples/llama/train_llama3_8b_h100_fp8.sh:168-171`）：

```bash
--log-throughput
--profile
--profile-step-start 4
--profile-step-end 6
```

`wait = start - 1`、`warmup = 1`、`active = end - start` 这组数字不是随手写的：从第 4 步开始采、第 6 步结束，意味着**采到的是第 5、6 步**，前面留了 1 步 warmup。torch.profiler 的 `wait` 是「前 N 步先不采」，`warmup` 是「采但不记录（让 profiler 自身的开销稳定下来）」，`active` 才是真正落盘的窗口。**把 warmup 设成 0 是最常见的采样污染源。**

迁移到别的场景时，同样的思路：第一次迭代包含编译、CUDA graph 捕获、权重预取——v0.20 甚至专门为此设了一个 `first_iteration` span 来捕捉这些一次性成本（它的文档字符串明确写着「不一定是第 1 次迭代，也可能是 checkpoint 恢复后的第一次」）。**首迭代的数据不该进稳态统计。**

### 7.2 锁定变量

- **重复取中位**：单次测量受抖动影响大，至少 3 次取中位。本篇的线程实验就是每档取 3 次最小值。
- **固定随机性**：数据顺序、dropout 都要固定，否则「变快了」可能只是运气。
- **区分锁频与不锁频**：`ncu --clock-control none` 与默认的锁频策略回答的是不同问题——锁频追求**可比性**（排除功耗墙/温度墙干扰），不锁频追求**真实性**（真实运行时就是会降频）。两个都采、明确标注是哪个，比只采一个强。这条在总纲篇里已展开过，此处只强调「必须标注」。
- **确认「只是变快了，结果没变」**：性能优化的正确性门槛是 loss 曲线落在噪声范围内。跑一个短窗口、对比两条曲线，而不是看最终 loss。

### 7.3 明确「没测什么」

比测了什么更具信息量的是**没测什么**。profiler 上线的场景通常有：采样窗口短（1–2 步）、单机单卡、合成数据、关闭了某些优化（如 CUDA graph 无法与 profiler 共存）、`profile_ranks` 只選了部分 rank。

`training.py:3496` 的 `profile_ranks` 决定了哪些 rank 落盘：

```python
should_prof_rank = (args.profile_ranks == [] or safe_get_rank() in args.profile_ranks)  # [] is all ranks
```

`[]` 表示全部 rank。默认只 profile 一个 rank 是常见做法（控制 trace 体积），但**单 rank 的 trace 看不到掉队**——恰好是分布式训练最需要诊断的问题。要做掉队分析就得全 rank 落盘，这个问题 01 篇会展开。

还有一个容易忽略的：Megatron 会把每个 rank 的 trace 导出成独立文件（`training.py:4448-4451` 的 `trace_handler`）：

```python
def trace_handler(p):
    profile_dir = Path(f"{args.tensorboard_dir}/../torch_profile")
    profile_dir.mkdir(parents=True, exist_ok=True)
    p.export_chrome_trace(f"{profile_dir}/rank-{torch.distributed.get_rank()}.json.gz")
```

**这些 per-rank 的 chrome trace 正好是多 rank trace 分析工具（如 HTA）的输入格式**——一个目录下放全部 rank 的 trace。这条链路在 01 篇里会用到。

顺带一提，同一个函数上方还有一个分支（`training.py:4442`）在 `pytorch_profiler_collect_chakra` 打开时注册 `ExecutionTraceObserver`，导出 Chakra ET 格式，用于回放式仿真。它是另一种用途——不做「看时间去哪了」，而是做「换一种硬件配置会怎样」。

---

## 8. 什么不能用什么测

把本篇的结论收成一张表。这张表的用法是反的：**先看你想回答的问题在不在左列，再看右边有没有你打算用的工具。**

| 你想回答的 | 可用的 | 不可用的，以及为什么 |
|:---|:---|:---|
| 端到端步时 / 吞吐 | 外层一对点；`interval-time` timer；墙钟 | 各子阶段 timer 相加——§4.4，min/max 不可加 |
| 某阶段占多少时间 | 时间线口径（CUDA Event / nsys / torch.profiler 区间） | 同步式 `Timer`——§4.2，重叠越好膨胀越大 |
| 通信有没有被藏住 | overlap 百分比 + 时间线 | 「max 之和 − compute 之和」——两个数来自不同 rank |
| 哪张卡是掉队卡 | 逐 rank 全量数据（`--timing-log-option all`）、逐 rank trace | `minmax` 两列——看不出短板是固定还是轮转 |
| 算力利用率 | MFU（分子 = 模型 FLOPs，分母写清精度与稀疏假设） | 拿 A100 页的稠密值配 H100 页的稀疏值；不写分母的 MFU |
| 重计算划不划算 | HFU 与 MFU 的差额；实测步时 | 官方 log 的 `TFLOP/s/GPU`——§3.2，它对 recompute 无感 |
| 时间用在正事上没有 | goodput（并注明哪些 span 算正事） | 用 MFU 替代——§6.3，两个指标可以相反 |
| decode 阶段的效率 | 显存带宽利用率；TTFT / TPOT（见 02 篇） | MFU——访存受限阶段 MFU 低是物理结果 |
| 占空比 / 活跃时间比 | 稳态窗口内 elapsed 的差值 | 累计 `active_time` 直接做除——§4.3 |
| 打包训练的收益 | 实际吞吐（tokens/s） | MFU 的前后对比——§3.3，分母变了 |

---

## 9. 批判与展望

三件本篇没能给出满意答案的事，如实写在这里。

**其一，MFU 作为跨栈比较指标的局限在扩大。** 它的分子是「模型定义」的 FLOPs，这个定义在稠密 Transformer 时代是清楚的；到了 MoE、Mamba/GDN 混合、MTP 多头预测的今天，同一个模型的「定义 FLOPs」在不同实现里可以差出几十个百分点（`gated_delta_product_layer_flops` 的注释就直接承认自己是个下界）。**当分子本身有实现相关的不确定性时，用它做跨栈比较要格外小心。**

**其二，FP8 / FP4 时代的分母问题会更严重。** BF16 时代还有「稠密一半、稀疏一半」这个还算整齐的关系；FP8 训练的峰值是 BF16 的两倍，FP4 又是 FP8 的两倍，而实际训练往往混用多种精度（前向 FP8、部分层 BF16、优化器状态 FP32）。**「用哪个精度当分母」正在从「选一个数」变成「选一套加权方案」。** 目前没有公认做法，各家各写各的。

**其三，goodput 没有统一定义，而且短期内不会有。** 哪些 span 算正事是框架作者的价值判断——Megatron 把 checkpoint 算正事（§6.3），换一个团队可能认为 checkpoint 是纯开销。**这让 goodput 很适合做纵向对比（自己的同一个作业在不同配置下），不适合做横向对比。**

一条本篇可以确定下来的结论：**口径问题不会随工具进步自动消失。** Nsight Systems 2026.5.1 加了 NCCL 掉队分析 recipe、Nsight Compute 2026.3 支持了 Tile IR、还新出了 Nsight Python 做脚本化内核剖析（[CUDA 13.4 发布说明](https://developer.nvidia.com/blog/cuda-toolkit-13-4-adds-windows-on-arm-support-and-greater-control-over-shared-gpus/)）——工具在变强，但「MFU 的分母是哪个数」「timer 测的是什么」这类问题，仍然只能靠读源码和写清楚来回答。

---

## 10. Lab Exercises

全部纯 CPU 可跑。

**Exercise 1：复现同步式计时的膨胀**

```bash
cd tools && python -m profiling_bench.timer_semantics
```

把工作负载改成 16 个 chunk 重跑，观察「每 chunk 同步」那一档的膨胀率怎么变。用解析式 $$(L-1)T_m / (L \cdot T_c + T_m)$$ 预测，再和实测对照，解释偏差来自哪里。

**Exercise 2：给一个 MFU 数字补口径**

拿任意一个公开的 MFU 数字（论文或技术博客），把它需要的三项口径补齐：分子含不含重计算、分母是哪种精度哪种稀疏假设、是否打包训练。补不齐的地方明确标注「未说明」。然后用 `mfu_accounting.py` 算出「在补齐不同假设下这个数会变成多少」的区间。

**Exercise 3：设计一个不会被 min/max 误导的观测方案**

给定 8 rank、每 rank 三个阶段，写一段代码生成合成数据，使得：① 各项 max 相加恰好等于最慢卡真实值；② 各项 max 相加比最慢卡真实值高 20% 以上。两种数据分别对应什么真实的故障形态？要区分它们，最少需要采集哪些字段？

**Exercise 4：验证「打包训练让 MFU 变化」的幅度**

用 `tools/megatron_bench/flops.py` 对同一个模型算两种口径：一种是 BSHD（`seqlen_squared_sum = batch × S²`、`total_tokens = batch × S`），一种是 THD（给定一批真实长度参差的序列，用真实的 $$\sum L_i^2$$ 与 $$\sum L_i$$）。给定固定的实测 tokens/s，比较两种口径下的 MFU，并回答：这个差异是性能差异还是记账差异？

---

## 参考清单

**官方规格（分母口径的来源）**

- [NVIDIA A100 产品页](https://www.nvidia.com/en-us/data-center/a100/) —— A100 80GB SXM 的 BF16 312 TFLOPS（稠密）/ 624 TFLOPS（带稀疏）、HBM 2039 GB/s
- [NVIDIA H100 产品页](https://www.nvidia.com/en-us/data-center/h100/) —— H100 SXM 的 BF16 1979 TFLOPS、FP8 3958 TFLOPS（均带稀疏脚注）、HBM 3.35 TB/s

**框架与源码（基准 commit `60e039626`，v0.20.0，2026-08-21 main）**

- [Megatron-LM 仓库](https://github.com/NVIDIA/Megatron-LM)
- `megatron/training/training.py:802` `num_floating_point_operations()`；`:1007` `return flops_fwd * 3`
- `megatron/training/training.py:3503` `interval-time`；`:3533` `--log-throughput`；`:4436-4464` torch.profiler 集成；`:3496` `profile_ranks`
- `megatron/core/timers.py:109` `Timer`；`:171` `reset()`；`:212` `active_time`；`:270` `_get_elapsed_time_all_ranks()`；`:318` `_get_global_min_max_time()`；`:422` `log()`
- `megatron/training/config/training_config.py:276` `timing_log_option`
- `megatron/core/telemetry/span_groups.py` `MegatronSpanGroup`；`telemetry/training_metrics.py` 8 个指标；`training/global_vars.py:490` `_set_telemetry()`

**论文**

- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) —— MFU 指标的出处，540B 报告 46.2%
- Dubey et al., *The Llama 3 Herd of Models*, [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) —— 405B 的 MFU 与并行配置对照

**工具文档**

- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) ／ [torch.cuda.Event](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html)
- [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) ／ [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [OpenTelemetry 规范](https://opentelemetry.io/docs/specs/otel/) ／ [NVML](https://developer.nvidia.com/management-library-nvml) ／ [DCGM](https://developer.nvidia.com/dcgm)
- [CUDA Toolkit 13.4 发布说明](https://developer.nvidia.com/blog/cuda-toolkit-13-4-adds-windows-on-arm-support-and-greater-control-over-shared-gpus/) —— Nsight 2026.x 一系更新与 Nsight Python 1.0

**站内互引**

- [NV 卡性能剖析工具全景：从 nvidia-smi 到 Nsight Compute](/2026/08/26/nv-gpu-profiling-toolkit/) —— 本系列前置总纲，工具分层地图与 ncu/SOL 语义
- [算子开发与优化（04）：性能分析方法论](/2026/09/03/op-04-performance-analysis/) —— 单 kernel 侧的「nsys → ncu → 假设-实验」三层下钻
- [Megatron-LM 深度剖析（00）：代码地图、配置系统与一次迭代的控制流全图](/2026/09/28/megatron-00-foundation/) —— 控制流与配置系统的全景
- [分布式训练全景（00）：为什么大模型必须分布式](/2026/08/31/dist-train-00-overview/) —— 并行策略与通信代价的解析推导

**本片配套脚本**：`tools/profiling_bench/timer_semantics.py`、`tools/profiling_bench/mfu_accounting.py`、`tools/profiling_bench/goodput_calc.py`，FLOPs 记账复用 `tools/megatron_bench/flops.py`。全部纯 CPU 可跑（miniconda python）。

**在线读 / 复算**（输出已固化，GitHub 上直接看）：

- [MFU/HFU 口径实验](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-mfu-accounting/profiling-00-mfu-accounting.ipynb)
- [Timer 计时语义](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-timer-semantics/profiling-00-timer-semantics.ipynb)
- [goodput 与 MFU 的分工](https://github.com/lrypcy/ipynbs/blob/main/experiments/profiling/profiling-00-goodput/profiling-00-goodput.ipynb)

---

下一篇进入训练栈：一次迭代到底慢在哪——三层观测面（OTel span / Perfetto 区间 / nsys 时间线）怎么分工、通信有没有真的被藏住、GPU 饿了的三种成因怎么区分、以及掉队卡怎么找。
