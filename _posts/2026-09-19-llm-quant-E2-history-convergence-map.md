---
title: "扩展篇 E2：PTQ / QAT 发展史与收敛地图（2015–2026）"
date: 2026-09-19 00:05:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, ptq, qat, history, survey, convergence]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **扩展篇 E2** ｜ 不与主线同序
>
> 主线：[00 量化全景](/2026/08/24/ptq-00-overview/) ｜ [23 统一视角](/2026/08/29/llm-quant-23-unified-view/) ｜ [24 KV Cache](/2026/08/29/llm-quant-24-kv-cache/)

> **TL;DR**
>
> * **核心结论**：PTQ/QAT 这十年不是「算法越出越多」的线性积累，而是**三次范式迁移**：① 2015–2019 的 CV 时代解决了「梯度怎么过 round()」（STE / 可学习 scale-clip）；② 2022–2023 的 LLM 第一波发现并攻克了「权重好压、激活难压」的不对称（LLM.int8() / GPTQ / SmoothQuant / AWQ）；③ 2024–2026 的第三波把战场从**算法**搬到了**数值格式 × 粒度 × kernel**（MXFP4 / NVFP4 / FP8 × block scaling × 融合算子），同时 QAT 从「补 PTQ 的坑」升级为「低比特的唯一解」。
> * **反直觉发现**：① **收敛的不是算法，是问题的难度分层**——W4A16 收敛是因为它只有「权重误差」一个自由度，W4A4 不收敛是因为它同时叠加了激活 outlier、逐层误差累积和 FC2 瓶颈。② **为整数格式设计的旋转类方法搬到 MXFP4 上反而有害**：ACL 2026 的 MXFP 系统性 benchmark 显示 QuaRot / SpinQuant 在 MXFP4 上**稳定差于 RTN 基线**，原因是全局旋转破坏了 MXFP 块级局部统计的保存机制。③ **2-bit 的失败和 4-bit 的失败不是一回事**：4-bit 是「信号退化」（计算通路完好、精度受损，可无训练修复），2-bit 是「计算崩溃」（关键组件失效、信号在浅层就被摧毁，必须结构重建）——这是 2026 年最有解释力的一个结论。
> * **本篇定位**：**[23 篇](/2026/08/29/llm-quant-23-unified-view/)给的是「地质学」（同一优化问题的不同自由度），本篇给的是「年代史」（这条河是怎么流到这里的）+「天气预报」（哪一段已经封冻、哪一段还在改道）**。读完它你应该能回答：一个新的量化论文出来时，它是在已封冻的河段上做微创新，还是在真正未收敛的河段上推进。所有年份、会议、数字均标注出处，可在 §11 年表与 §12 论文索引中核对。

---

## 0. 符号与缩写表

本篇是历史综述，主要沿用主线记号，只新增少量缩写：

| 缩写 | 全称 | 含义 |
|---|---|---|
| W$b$A$b'$ | Weight-$b$-bit / Activation-$b'$-bit | 例：W4A16 = 权重 INT4、激活 BF16 |
| STE | Straight-Through Estimator | 直通估计器，令 $\partial \hat{w}/\partial w = 1$ |
| PTQ / QAT | Post-Training / Quantization-Aware Training | 训练后量化 / 量化感知训练 |
| MXFP | Microscaling Floating Point | OCP 标准的块浮点，块内共享 E8M0 尺度 |
| NVFP4 | NVIDIA FP4 | Blackwell 上的 4-bit 浮点变体（块更小、尺度更细） |
| SQNR | Signal-to-Quantization-Noise Ratio | 量化信噪比，$\approx 6.02b$ dB 是均匀量化的经典上界 |
| ILP | Integer Linear Programming | 混合精度里做 bit allocation 的标准工具 |

---

## 1. 为什么要单独写一篇「历史」

主线写到这里有一个结构性问题：**按算法组织的内容天然是「横向切片」，它无法表达「这个领域正在往哪走」**。

你读完 GPTQ、AWQ、SmoothQuant、QuaRot 之后，知道它们各自在干什么，但不知道：

* 为什么 2022 年大家突然都在做 weight-only？
* 为什么 2023 年中之后所有人开始做 activation？
* 为什么 2024 年底开始 QAT 论文数量暴增？
* 为什么 2025–2026 年的论文标题里越来越多出现 MXFP4 / NVFP4 / microscaling，而不是新的「XXQuant」？

这些问题的答案不在任何一篇论文里，而在**约束条件的变化**里：

```text
2015–2019  约束：CV 模型小、可重训      → 问题：梯度怎么过 round()
2022–2023  约束：LLM 太大、不能重训      → 问题：不训练怎么压到 4-bit
2023–2024  约束：显存吃完、prefill 也要快 → 问题：激活也得压
2024–2026  约束：硬件开始原生支持 FP4/MX → 问题：算法与格式谁迁就谁
2025–2026  约束：4-bit 已够用、要更极致  → 问题：2-bit 与 reasoning 能不能活
```

每一波的「主流问题」变了，所以「主流算法」才会变。把这个时间轴补齐，你才有能力判断：**2026 年的一篇新量化论文，是在解决一个真问题，还是在解决一个已经被硬件解决掉的问题。**

---

## 2. 史前史（2015–2019）：CV 时代留下三个遗产

这一波的主角是 CNN，模型小（几十 MB）、训练便宜、可以随便重训。所以这一波几乎全是 **QAT**——因为重训不是问题。

### 2.1 遗产一：STE（BinaryConnect / BNN，2015–2016）

量化函数 $\hat{w} = s \cdot \mathrm{clip}\big(\lfloor w/s\rceil, q_{\min}, q_{\max}\big)$ 里的 $\lfloor\cdot\rceil$ 几乎处处导数为零，反向传播直接断掉。

BinaryConnect（2015）给的解法粗暴但有效：**前向用量化值，反向假装量化不存在**。

```text
forward:   w → sign(w)          （用量化值）
backward:  ∂L/∂w ← ∂L/∂ŵ · 1    （梯度原样穿过）
```

这个「$\cdot 1$」就是 STE。此后十年，几乎所有 QAT 都在它的变体上打补丁。

### 2.2 遗产二：量化器参数可学习（PACT 2018 / LSQ 2019）

STE 之后大家发现一个尴尬事实：**scale $s$ 和 clip 上界 $\alpha$ 是人工常数，模型能调权重，却调不了网格本身**。

* **PACT**（2018）把 clip 上界 $\alpha$ 变成参数，并给出 $\partial \hat{y}/\partial \alpha$ 的梯度；
* **LSQ**（2019）把 step size $s$ 变成参数，从 STE 推出 $\partial \hat{v}/\partial s$ 的三段式梯度，并用 $g = 1/\sqrt{n_Q Q_P}$ 做尺度归一化，让 $s$ 的更新步长与权重可比。

这条线在本系列的 [18 篇](/2026/08/29/llm-quant-18-lsq-pact-dsq/)里有完整推导。**它们今天的地位是：所有现代低比特 QAT 的默认底座。**

### 2.3 遗产三：per-channel 粒度与「不是所有东西都该一样压」

2017–2019 年的工程实践（TFLite、TensorRT）确立了粒度阶梯：per-tensor → per-channel → per-group，并确认了 **per-channel weight + per-tensor activation** 是 INT8 部署的默认配置。

这个「不同对象用不同粒度」的思想，是十年后 **混合精度**与**MoE 逐专家量化**的祖先。

### 2.4 这一波的边界

2019 年时，社区的主流结论是：**INT8 基本无损，INT4 需要 QAT 且掉点明显**。这个结论在三年后被 LLM 打破——但不是被算法打破，而是被**问题重新定义**打破。

---

## 3. 第一波 LLM 量化（2022–2023）：outlier 的发现与「绕开 / 搬走 / 补偿」

### 3.1 起点是一个「不该发生」的失败

2022 年之前，把 BERT 时代的 INT8 方案直接套到 GPT 级模型上会**直接崩掉**。原因不是算法不好，而是 LLM 有一个 CNN 没有的现象：**系统性 emergent outlier**——少数隐藏维度上出现幅度超出其余维度几十倍的激活值，且这些维度与语义无关、位置固定。

这是 [02 篇（LLM.int8()）](/2026/08/24/llm-quant-01-llmint8-outlier-mixture/)讲的故事。LLM.int8() 给出的第一代解法是**绕开**： outlier 维度走 FP16 小路，其余走 INT8。

### 3.2 「绕开」之后的三种解法

一旦 outlier 被识别为病根，2022–2023 年迅速分化出三条路，它们至今仍是三条并行主线：

| 路线 | 代表 | 核心动作 | 解决的是哪种误差 |
|---|---|---|---|
| **补偿**（改答案） | GPTQ（2022） | 用 $H=2XX^{\top}$ 做二阶误差补偿：量化一个权重，让其余权重补偿它的误差 | 权重量化的**事后**误差 |
| **保护**（改问题） | AWQ（2023） | 用激活幅度找显著通道，scale up 后量化，理论上保持 FP 计算等价 | 权重量化的**事前**分布 |
| **搬走**（改问题） | SmoothQuant（2023） | 用恒等式 $XW = (X\,\mathrm{diag}(\tau)^{-1})(\mathrm{diag}(\tau)W)$ 把难度从激活搬到权重 | 激活量化的 outlier |

**这三条路不是互斥的，这是选型时最常见的误解。** 工业界 SOTA 基本都是叠加结果（[23 篇 §统一视角](/2026/08/29/llm-quant-23-unified-view/)）。

### 3.3 这一波为什么能快速收敛

因为 **W4A16 只有一个自由度**：权重误差。

```text
W4A16 的推理路径：
INT4 weight → dequant → FP16/BF16 GEMM
```

激活不量化，误差不跨层累积成「激活分布漂移」，所以：

* 误差可用单层二阶近似精确刻画 → GPTQ 的 Hessian 补偿有效；
* 误差可用激活统计量预判 → AWQ 的显著性保护有效；
* 误差可用少量校准集估计 → 不需要训练。

**一个自由度、可解析、可离线估计——这是它收敛的全部原因。**

### 3.4 同期并行的「另一条路」：outlier 单独存

SpQR（2023）与 SqueezeLLM（2023）给出一个很不相同的哲学：

> **既然少数 outlier 压不动，那就不要压它们。**

SpQR 把权重分成「主体 INT3/4 + 少量 FP16 outlier」，SqueezeLLM 用「非均匀量化 + 稀疏敏感值」。这条线后来在 [06 篇](/2026/08/24/ptq-04-spqr-owq-hqq/)与 [08 篇](/2026/08/24/ptq-09-squeezellm-vptq-claq/)展开，也在 2026 年的 **pQuant** 里以「1-bit 主分支 + 高精度小分支」的形式复活。

---

## 4. 第二波（2023–2024）：等价变换的极限与码本换赛道

### 4.1 当缩放不够用时：旋转

SmoothQuant 的 $\mathrm{diag}(\tau)$ 只是对角缩放。2024 年的 QuaRot / SpinQuant 把它升级为**正交旋转**：

$$XW = (XR)(R^{\top}W), \quad R^{\top}R = I$$

对角缩放只能重新分配「哪个通道难」，旋转能把 outlier 的**能量摊薄到所有坐标**——因为旋转后每个新坐标都是原坐标的线性组合，极端值被稀释。

这条线的意义在于：它第一次让 **W4A4（连激活一起压到 4-bit）**在 7B/70B 上逼近 FP16（[12 篇](/2026/08/24/ptq-07-quarot-spinquant/)）。

### 4.2 但旋转不是终点：FlatQuant 的「平坦度」批判

ICML 2025 的 **FlatQuant**（[arXiv:2410.09426](https://arxiv.org/abs/2410.09426)）直接指出：SmoothQuant 与 QuaRot 之后，**权重和激活的分布仍然 steep and dispersed**，并不平坦。

它的做法是把等价变换从「固定矩阵」变成**可学习仿射变换**，并用 Kronecker 分解 $P = P_1 \otimes P_2$ 把 $O(N^2)$ 的开销压下来，再配一个融合 Triton kernel。结果：LLaMA-3-70B 的 W4A4 掉点 **< 1%**，相对 SpinQuant 提升 7.5%，prefill 加速 2.3×、decode 加速 1.7×。

**这个结果很重要，因为它说明「旋转类方法」这条线本身还在演进，且已经从「数学构造」转向「学习 + 系统实现」。**

### 4.3 2-bit 换赛道：向量量化

当 4-bit 也压不下去时，标量量化（每个权重独立取整）的信息论上限就到了。QuIP / QuIP# / AQLM / VPTQ 的做法是**多个权重联合编码**：

```text
标量量化：  w1→q1,  w2→q2,  w3→q3,  w4→q4
向量量化： [w1,w2,w3,w4] → code → codebook lookup
```

QuIP# 用 Hadamard 变换 + E8 格码本把 2-bit 推到实用区间；VPTQ 进一步做到 < 2-bit 并提供 70B/405B 方案（[07 篇](/2026/08/24/ptq-05-quip-aqlm/)）。

**但这条线至今没成为部署主流**，原因不是精度不够，而是：

> **codebook lookup → dequant → GEMM 打不过 INT4 Tensor Core。**

这就是「算法最优 ≠ 系统最优」的第一个大型案例。

### 4.4 工程侧同步发生的事：格式标准化

2023–2024 年，量化从「算法问题」外溢成「生态问题」：GGUF k-quants（llama.cpp 生态）、FP8 E4M3/E5M2（Hopper）、GPTQ/AWQ 进入 vLLM/TensorRT-LLM 主干（[15 篇](/2026/08/24/ptq-08-gguf-fp8-mxfp4/)）。

到 2026 年，vLLM 主干已支持 **20+ 种量化方法**，包括 FP8、GPTQ/AWQ-Marlin、GGUF、bitsandbytes、compressed-tensors、ModelOpt（FP8 / NVFP4）、TorchAO、MXFP4 / NVFP4、INT8 experts，以及 KV cache 的 FP8 / INT8 / NVFP4。

**这个「多格式并存」本身就是最强的一个信号：这个领域没有出现赢家，出现的是生态分层。**

---

## 5. 第三波（2024–2026）：QAT 复兴与格式战争

### 5.1 为什么 QAT 会复兴

因为 PTQ 的能力曲线在 **2–3 bit 断崖**。2026 年的机制研究给了这件事一个远比「误差变大」更精确的解释。

**ACL 2026 Findings，[arXiv:2604.19884](https://arxiv.org/abs/2604.19884)（中科院自动化所等）**用机制可解释性工具（Logit Lens 逐层投影、跨模型因果激活补丁、注意力熵/JSD 诊断、FFN key-value memory 的 SFR/Jaccard/cosine 度量）在事实知识召回任务上系统对比 FP16 / 4-bit / 2-bit，发现两种**性质不同**的失效：

| | **信号退化**（4-bit） | **计算崩溃**（2-bit） |
|---|---|---|
| 计算通路 | 完好，只是精度受损 | 关键组件功能失效 |
| 信号出现位置 | 中后层仍出现，但强度变弱 | 浅层就被摧毁，根本没生成 |
| 误差性质 | 累积噪声（量变） | 组件失效（质变） |
| 无训练修复 | **有效**（源保护 + 峰值信号放大） | **无效**（多米诺实验证明不可逆） |
| 需要的手段 | 误差补偿类 PTQ | **结构重建**（微调 / QAT） |

原文结论直接写着：**「解决 computation collapse 需要 structural reconstruction 而非简单的补偿」**。

这句话基本给「2-bit 到底该不该继续卷 PTQ 算法」判了刑：**卷不动，得训练。**

### 5.2 QAT 的工程化：从论文到框架

2024–2026 年 QAT 的第一变化是**它变得能用**：

* **TorchAO 官方 QAT 流程**（PyTorch blog, 2026-03）：在 Llama-3 上相对 PTQ 恢复 **96% 的 HellaSwag 精度退化、68% 的 WikiText PPL 退化**；
* **Unsloth 集成 INT4 QAT**：恢复 **66.9%** 精度退化，相对 BF16 推理加速 **1.73×**；
* **Axolotl 集成 NVFP4 QAT（prototype）**：恢复 **71.6%** 精度退化，B200 上加速 1.35×、HBM 降到 1/4；
* **QAT + LoRA**：训练加速 **1.89×**，显存降 36.1%；
* **TorchAO 0.16.0 支持的组合**：INT4/FP32·BF16、INT4/FP8、INT4/INT8、NVFP4/NVFP4（prototype）。

关键点是 **prepare → fake quant → train → convert** 这套工作流已经稳定（本系列 [17 篇](/2026/08/26/llm-quant-11-fake-quant-insertion/)）：

> **QAT 的「工作流」收敛了，但「最优 recipe」远没有收敛。**

### 5.3 QAT 的算法分化：四条并行的路

#### (a) 分块 + 端到端两段式：EfficientQAT（ACL 2025）

[arXiv:2407.11062](https://arxiv.org/abs/2407.11062) 解决的是「70B 怎么做 QAT」：

```text
Stage 1  Block-AP：逐 block 训练全部参数（W, s, z），block 级重构损失
Stage 2  E2E-QP ：冻结量化权重，只端到端训练 step size（约 1.6% 参数）
```

**单张 A100-80GB、41 小时完成 Llama-2-70B 的 2-bit 量化，69.48 vs FP16 72.41（掉点 < 3）**；INT2 70B 占 19.2GB，反而比 Llama-2-13B（67.81 / 24.2GB）更好。

**这篇是理解「LLM QAT 为什么能真正做起来」的关键论文**——它把 QAT 从「需要全套预训练算力」降到了「PTQ 量级成本」。

#### (b) 从启发式到优化理论：PARQ（ICML 2025）

[arXiv:2503.15748](https://arxiv.org/abs/2503.15748)（Meta 等）换了整个视角：**不再用 STE 这种启发式，而是用凸的分段仿射正则把参数「吸」到离散网格上**：

$$\Psi(w) = \max_{k \in \{0,\dots,m\}} \big\{ a_k (|w| - q_k) + b_k \big\}$$

配合 **AProx**（aggregate proximal SGD）最小化 $\mathcal{L} + \lambda\Psi$，并证明 **last-iterate convergence**（此前 BinaryConnect 一类只能给 average-iterate 保证）。论文还给出一个漂亮结论：**STE 是 PARQ 的渐近形式**——十年启发式终于有了理论解释。

TorchAO 已将其作为 prototype 集成：**3-bit per-row 与 4-bit per-group 基线精度相当，显存仅约 58%，decode 吞吐快约 1.57×**。

#### (c) 极低比特 scaling law：ParetoQ（ICML 2025）

[arXiv:2502.02631](https://arxiv.org/abs/2502.02631) 是第一个把 1 / 1.58 / 2 / 3 / 4 bit 放进统一 QAT 框架做严格比较的工作。核心发现是一个 **2-bit 与 3-bit 之间的「学习相变」**：

> 3-bit 及以上：微调后模型仍贴近原预训练分布；
> 2-bit 及以下：**表示发生剧变**，本质上是「重新学习一个表示」而不是「补偿误差」。

因此它的结论是：**ternary / 2-bit / 3-bit 在 size-accuracy 权衡上基本打平，且总体优于 4-bit 与 binary**；binary 显著更差。它提出的 **Stretched Elastic Quant（SEQ）** 解决了 2-bit 网格「含 0 导致 bin 分配不均」的问题，并给出 2-bit GPU kernel：相对 FP16 最高 **4.14×**、相对 Machete 4-bit kernel **1.24×**。

**注意正确读法**：这不是「2-bit 已解决」，而是「2-bit 在**足够好的训练配方**下，Pareto 前沿上可以打败 4-bit 的小模型」。它恰恰证明 **ultra-low-bit 是一个训练问题**。

#### (d) 渐进式与结构改造：Bit-by-Bit / BSQAT / pQuant（2026）

* **Bit-by-Bit**（ACL 2026，[arXiv:2604.07888](https://arxiv.org/abs/2604.07888)）：直击「直接低比特 QAT 的 loss spike 与跨层误差累积」。三件套——逐 block 渐进降精度、嵌套整数网格实现 **train once / deploy any precision**、rounding-aware outlier channel splitting（恒等变换，不改量化输出）。配自研 W2A2/W2A16 kernel，相对 BF16 最高 **11×** 加速。Llama2-7B 上 W2A2 的 WikiText2 PPL 差距仅 **2.25**，相对 BitDistiller 优 24.19、相对 EfficientQAT 优 20.59；训练 token 需求相对 ParetoQ 降 **3600×**。
* **BSQAT**：指出主流 QAT 把 Transformer block **孤立处理**，忽略了相邻 block 的结构依赖。用滑窗合并相邻 block 并**共享量化参数**（LQAT），再加低秩缩放模块保权重保真（GQAT）。LLaMA-3-70B INT2 上 zero-shot 平均精度 **65.18% → 67.87%**（对比 EfficientQAT），显存再降约 7%，覆盖 6.7B–123B。
* **pQuant**（ACL 2026 Findings，[arXiv:2602.22592](https://arxiv.org/abs/2602.22592)）：发现 sub-2-bit 的一个瓶颈叫 **parameter democratization**——所有参数的敏感度被「同质化」，模型分不清谁重要，表达力崩塌。解法是把线性层拆成**主导 1-bit 分支 + 紧凑高精度分支**，用 feature scaling 显式引导敏感参数流向高精度分支，并把该分支扩展成稀疏激活的多专家。相对 1-bit SOTA **PPL 降 32.0%**，精度超过 2-bit 模型，**吞吐 +18.2%**。

**这四篇共同说明：2026 年的 QAT 已经不是「怎么让梯度穿过 round()」，而是「用什么数据、什么 loss、什么 schedule、什么结构让模型学会适应量化」。**

### 5.4 同一时期：格式战争开打

这是我认为最值得现在关注的变化。算法的竞争正在让位于**数值格式**的竞争：

```text
INT（INT4 / INT8）        —— 成熟，生态最厚
FP （FP8 E4M3/E5M2）      —— 已标准化，接近无损
MX （MXFP8 / MXFP4）      —— OCP 标准，块内共享 E8M0 尺度，块大小 32
NVFP4                     —— Blackwell 上的 FP4 变体
NF / 非均匀 / PoT / 码本   —— 各自服务特定分布与硬件
```

**ACL 2026 的 MXFP 系统性 benchmark（[arXiv:2601.09555](https://arxiv.org/abs/2601.09555)）**给出了一组非常有代表性的数字（7+ PTQ 算法 × 15 个 benchmark × 3 个模型族）：

| 配置 | 精度恢复率 | 结论 |
|---|---|---|
| MXFP8 (W8A8) | 通常 **> 98%** | 近无损，**不需要复杂 PTQ 算法** |
| MXFP4 权重 (W4A8) | **95–98%** | 有可见损失，非推理任务可接受 |
| MXFP4 (W4A4) | 常 **< 95%**，个别低至 **86–87%** | 极其困难，算法选择变得关键 |

更关键的是三条**反直觉**结论：

1. **旋转类方法在 MXFP4 上是有害的**：QuaRot / SpinQuant 稳定差于 RTN 基线。原因是 MXFP 的机制是「块内局部统计保存」，而全局旋转破坏了这种局部性。**一个在整数格式上 SOTA 的方法，换格式后变成负优化。**
2. **RTN 基线的竞争力惊人**：说明多数为整数设计的 PTQ 方法，直接搬到 MXFP 上并没有真正对齐 MXFP 的量化机制。
3. **MXFP4 的主要误差源是 scale 本身**（E8M0 尺度只有 8 位指数、0 位尾数，粗量化造成 block 内所有值的尺度失配）；一个简单的 **pre-scale（输入先乘 3/4 再量化）** 就能显著缓解。

针对这一点，2026 年的 **DuQuant++**（[arXiv:2604.17789](https://arxiv.org/abs/2604.17789)）给出了格式对齐的正确姿势：把旋转的**块大小对齐到 MX 的 microscaling group（B=32）**，因为每个 MX group 有独立尺度，跨块方差问题不复存在，于是「双重旋转 + zigzag 置换」可以简化为**单次 outlier-aware 旋转**，在线旋转开销减半，并在 LLaMA-3 的 MXFP4 W4A4 上取得 SOTA。

> **这一节的教训：2026 年选量化方案时，「你的目标格式是什么」可能比「你的算法是什么」更重要。**

---

## 6. 三条时间线合一：完整年表

| 年份 | 代表工作 | 它解决了什么 | 它留下了什么（至今仍在用） |
|---|---|---|---|
| 2015–16 | BinaryConnect / BNN / XNOR-Net | 梯度怎么过 $\mathrm{sign}$ / $\mathrm{round}$ | **STE** |
| 2017–18 | DoReFa / PACT / TFLite INT8 | clip 上界怎么定 | **可学习 clip**、per-channel 粒度 |
| 2019 | LSQ / DSQ | scale 怎么学 | **可学习 scale**（现代 QAT 默认件） |
| 2022.08 | LLM.int8() | LLM INT8 为什么崩 | **outlier 现象**、混合精度分解 |
| 2022.10 | GPTQ | 不训练如何压到 3/4-bit 权重 | **二阶误差补偿**，$H=2XX^{\top}$ |
| 2023.03 | SmoothQuant | 激活怎么压到 INT8 | **等效缩放迁移**，难度守恒 |
| 2023.06 | AWQ / SpQR / SqueezeLLM | 权重重要性 / outlier 处理 | **激活感知保护**、稀疏 outlier 存储 |
| 2023.07 | QLoRA | 消费级卡微调 | **NF4 + LoRA**，Q-PEFT 范式 |
| 2023.10 | OmniQuant / QuIP / AQLM | 可学习 PTQ / 2-bit 码本 | **LWC / LET**、incoherence + 码本 |
| 2024.03 | KIVI / KVQuant | KV cache 显存 | **per-channel K / per-token V** |
| 2024.04 | QuaRot / SpinQuant | W4A4 / KV4 | **正交旋转**消除 outlier |
| 2024.07 | EfficientQAT | 70B 怎么做 QAT | **Block-AP + E2E-QP**，低成本 QAT |
| 2024.10 | FlatQuant / VPTQ | 平坦度 / <2-bit | **可学习仿射变换**、码本工程化 |
| 2025.02 | ParetoQ | 极低 bit scaling law | **2↔3-bit 相变**、SEQ 量化函数 |
| 2025.03 | PARQ | QAT 的理论基础 | **分段仿射正则 + AProx**，STE 的理论解释 |
| 2025.05 | QAT Scaling Law | W4A4 误差从哪来 | **FC2 激活 outlier 是主瓶颈** |
| 2025–26 | TorchAO / Unsloth / Axolotl | QAT 工程化 | **INT4 / NVFP4 QAT 生产可用** |
| 2026.01 | MXFP PTQ Benchmark | 算法 vs 格式 | **格式兼容性 > 算法先进性** |
| 2026.02 | pQuant | sub-2-bit 表达力 | **参数民主化**、1-bit 主分支 + HP 分支 |
| 2026.04 | Bit-by-Bit / BSQAT / DuQuant++ | 低比特稳定性 / 块间依赖 / 格式对齐 | **渐进降精度**、**块间共享参数**、**group 对齐旋转** |
| 2026 (ICLR) | ReasoningQAT | 推理模型低比特 | **混合域校准 + 教师引导奖励修正** |
| 2026 (ACL) | 两种失效模式 | 2-bit 为什么崩 | **信号退化 vs 计算崩溃** |

---

## 7. 收敛地图（截至 2026-09）

**「收敛」的判据**（这是本篇的定义，不是数学意义）：

> 主流方案之间的差距，是否已经小到工程上主要取决于**格式 / 粒度 / kernel / 硬件**，而不是论文算法本身。

| 层级 | 方向 | 代表算法 / 格式 | 收敛度 | 主要依据 |
|---|---|---|---|---|
| **已封冻** | INT8 / FP8 / MXFP8 PTQ | SmoothQuant、FP8 E4M3 | 🟢🟢 | MXFP8 恢复率 > 98%，无需复杂算法 |
| **已封冻** | W4A16 PTQ | GPTQ / AWQ / HQQ | 🟢🟢 | 单自由度、可解析、生态统一（Marlin） |
| **已封冻** | QAT 工作流 | prepare→fake→train→convert + STE | 🟢🟢 | TorchAO / Unsloth / Axolotl 全线支持 |
| **已封冻** | W4A16 QAT | standard QAT / QAT+LoRA | 🟢 | 恢复 66.9–96% 退化，成本已 PTQ 化 |
| **正在收敛** | W8A8 的主格式之争 | INT8 vs FP8 vs MXFP8 | 🟢/🟡 | FP8 正在侵占 INT8 的高 batch 场景 |
| **正在收敛** | W4A8 | QQQ / AWQ variants | 🟡 | 同时优化 prefill 与 decode，方案未统一 |
| **正在收敛** | 等价变换类 | SmoothQuant → QuaRot → FlatQuant | 🟡 | 仍在学习化 + kernel 融合方向演进 |
| **正在收敛** | FP8 / NVFP4 QAT | TorchAO NVFP4 prototype | 🟡→🟢 | 恢复 71.6%，仍标 prototype |
| **未收敛** | W4A4 | rotation + QAT 组合 | 🟡 | MXFP4 W4A4 恢复率可低至 86–87% |
| **未收敛** | W3A16 / W3A3 | GPTQ / OmniQuant / VQ | 🟡 | 处于 2↔3-bit 相变边缘 |
| **未收敛** | 量化格式 | INT4 / FP4 / MXFP4 / NVFP4 / 码本 | 🔴 | 硬件厂商各自押注，尚未统一 |
| **未收敛** | W2A16 / W2A2 | EfficientQAT / Bit-by-Bit / pQuant | 🔴 | 计算崩溃需结构重建，非补偿可解 |
| **未收敛** | Sub-2-bit | pQuant / ParetoQ 1-bit 分支 | 🔴 | 参数民主化问题刚被提出 |
| **未收敛** | Reasoning LLM 量化 | ReasoningQAT | 🔴 | 2-bit 上 PTQ 近乎归零，靠训练配方救 |
| **未收敛** | MoE 量化 | 逐专家混合精度 / DynaExq | 🔴 | 路由驱动的敏感度排序仍在探索 |
| **未收敛** | 长上下文 / KV | KIVI / KVQuant / FP8 KV | 🟡/🔴 | 误差随长度累积的机制未完全清楚 |
| **未收敛** | 量化感知预训练 | BitNet / 原生低比特预训练 | 🔴 | 需要全套预训练算力，仅少数机构能做 |

### 7.1 一句话版本

```text
PTQ   ：4/8-bit 主干已封冻；创新点从「新算法」转向 format × granularity × kernel
QAT   ：工作流已封冻，最优 recipe 未封冻；W4A16 成熟，W4A4 / W2 / W3 是主战场
真战场：ultra-low-bit × activation quantization × reasoning / MoE / long-context
        × hardware co-design
```

---

## 8. 为什么 W4A16 收敛而 W4A4 不收敛：机制层面的解释

这是收敛地图上最重要的一条分界线，值得单独拆开。

### 8.1 W4A16：单自由度

$$\hat{y} = (\hat{W})x, \quad \hat{W} = W + \Delta W$$

误差只有 $\Delta W$，且**不进入下一层的输入分布**（激活仍是 FP16）。于是误差是可加、可局部、可解析的：

$$\min_{\hat{W}} \ \mathrm{tr}\big(\Delta W\, C\, \Delta W^{\top}\big), \quad C = \mathbb{E}[xx^{\top}]$$

这就是 [23 篇](/2026/08/29/llm-quant-23-unified-view/)的统一目标。**一个凸性良好的局部问题，自然会被几篇论文解决完。**

### 8.2 W4A4：四重叠加

```text
ΔW（权重量化误差）
 +  Δx（激活量化误差，含 outlier 劫持 scale）
 +  逐层累积（Δx 会作为下一层的输入继续被量化）
 +  FC2 层的激活 outlier（主瓶颈）
```

第二条和第三条是**非线性耦合**的：激活误差会改变下一层的输入分布，而量化参数是基于**未受扰动的**校准集估计的——这就是所谓的 **distribution shift under quantization**，它不能靠单层二阶近似刻画。

### 8.3 实验证据：QAT Scaling Law（arXiv:2505.14302）

2025 年这份研究做了 **268 次 QAT 实验、27.6 万 A100 GPU 小时**，模型规模 74M–594M、训练 token 10B–100B，把量化误差建模为「模型规模 × 数据量 × group size」的函数。结论：

1. 量化误差**随模型规模增大而减小**，随**训练 token 增多**和**粒度变粗**而增大；
2. 把 W4A4 误差分解为权重项与激活项后，**FC2 层的激活量化误差（由 outlier 引起）是 W4A4 的主要瓶颈**；
3. 用混合精度处理 FC2 后，权重误差与激活误差可收敛到相近水平；
4. **但随着训练数据继续增加，权重量化误差最终会超过激活误差**——也就是说瓶颈会转移。

第 4 点特别重要：**W4A4 的最优解不是一个固定配方，而是随训练预算迁移的**。这本身就是「未收敛」的定义。

---

## 9. 2026 年的六个开放问题

按我判断的「重要性 × 活跃度」排序：

### 9.1 Reasoning 模型的低比特（正在被攻，远未解决）

推理模型经过 SFT + 偏好优化后形成了**异质知识结构**：预训练得来的常识与 post-training 得来的推理能力，对量化的敏感度完全不同。ICLR 2026 的 [ReasoningQAT](https://openreview.net/forum?id=Azsd2qyK6C) 做了一个关键实验——把校准集中推理数据的比例从 0 调到 100%，**常识类任务几乎不动，而数学/代码/科学显著上升**；t-SNE 显示预训练输入激活聚得很紧，推理输入激活分散得很开。

它的两阶段方案：

```text
Stage 1  混合域校准（80% 推理 OpenThoughts + 20% 预训练 FineWeb-Edu）
         + block-wise 只微调 scale（EfficientQAT 风格，4096 样本 × 2048 ctx）
Stage 2  教师引导的奖励修正：L_t(θ) = L_SFT(θ) · sg(π_T(y*|x))
         + β · KL(π_T ‖ π_S)（top-20 概率上计算，α=0.2, β=1.0）
```

结果：混合域校准相对单域在 6 个任务上平均提升 **2.74%**；**2-bit Qwen3-8B 在 5 个推理基准上平均超过 PTQ 基线 50.45%**，并以 < 1B token 超过 BitNet-2B4T 约 2% 的数学精度。

> 注意这个「50.45%」的含义：它说明 **2-bit 上 PTQ 已经近乎归零**，而不是说明 2-bit 已解决。

### 9.2 MoE 量化

MoE 的量化难点不在算法，而在**「谁重要」这件事由路由动态决定**。2026 年的几个进展：

* **ICLR 2026（arXiv:2604.06515）**：从理论上证明**训练过程中 router L2 norm 变化最小**的专家，往往捕捉稀有但关键的特征 → 对量化噪声更敏感 → 应分配更高精度；辅以 **max intra-neuron variance** 修正排序。对拿不到初始权重的预训练模型，可用**最终 router 的 L2 norm** 作为代理指标。
* **DynaExq（arXiv:2511.15015）**：把问题从离线变成**在线**——用路由轨迹估计专家热度，按 HBM 预算做 budget-feasible top-$n$ 选择，通过 stable expert handle 异步升降精度。Qwen3-80B 上静态 PTQ **73.09% → 77.57%**，batch 32 下吞吐最高 **2.73×** 于 offloading 基线。
* **MODE（EMNLP 2026，arXiv:2606.17118）**：指出 MoE-MLLM 里专家频率统计被**视觉 token 的数值优势**污染（跨模态偏置）与**冗余视觉 token**（模态内偏置），需要先做模态分解 + 去噪，再用 **ILP** 分配逐专家 bit。W3A16 下平均损失 **≤ 2.9%**。

### 9.3 2-bit 的结构重建（而非误差补偿）

§5.1 的「计算崩溃」结论意味着：**2-bit 的正确打开方式是改变模型结构或训练配方，而不是继续设计更精巧的 PTQ 目标**。2026 年的 Bit-by-Bit（渐进 + 嵌套网格 + outlier 通道拆分）、pQuant（1-bit 主分支 + 高精度分支）、BSQAT（块间共享）都是这个方向的不同答案，尚未统一。

### 9.4 格式与硬件协同设计

MXFP4 的 scale 量化（E8M0）本身成为主要误差源这件事，说明**格式的细节决定了算法的上限**。未来的竞争不是「GPTQ vs AWQ」，而是：

> **INT4 vs FP4 vs MXFP4 vs NVFP4 × 各家 kernel × 各家 compiler。**

### 9.5 长上下文与 KV

见 [24 篇](/2026/08/29/llm-quant-24-kv-cache/)。核心未解问题是**误差随上下文长度的累积规律**与 RoPE / 位置编码与量化的相互作用。

### 9.6 量化感知预训练

BitNet b1.58 一类「从预训练开始就是低比特」的路线效果最好，但需要全套预训练算力。**它对绝大多数团队不可及**，但对硬件厂商与头部实验室是终局路线的候选。

---

## 10. 2026 年的实用选型表

> **注意**：本节是**算法侧**的选型（哪个算法适合哪个场景）。算法能不能真正在你的机器上跑起来，是另一件事——
> vLLM / SGLang 的支持矩阵、kernel 后端、硬件门槛见 **[E3 篇：部署侧视角](/2026/09/19/llm-quant-E3-deployment-support/)**。
> 两套排序并不一致（典型例子：AQLM 精度很好但已被 vLLM 移除；RTN 是最朴素的 baseline 却是官方推荐的入门路径）。

把历史收敛结论落成可执行的判断：

| 你的场景 | 推荐配置 | 说明 |
|---|---|---|
| 通用 LLM serving，Hopper/Blackwell | **FP8 W8A8** | 近无损，kernel 成熟 |
| 低 batch / decode 密集 / 显存吃紧 | **W4A16 + GPTQ 或 AWQ**（group 128/32） | 已封冻，Marlin kernel 统一 |
| 高 batch / prefill 密集 | **FP8** | INT8 正在被 FP8 替代 |
| 想同时优化 prefill 与 decode | **W4A8**（QQQ 一类） | 未收敛，需实测 |
| Blackwell 上追求极限吞吐 | **NVFP4 / MXFP4** | 必须用格式对齐的算法（FlatQuant、DuQuant++），**慎用全局旋转** |
| 端侧 / CPU / 边缘 | **GGUF k-quants**，或 **Q4_K_M** 一档 | llama.cpp 生态 |
| 需要 ≤3-bit | **QAT**（EfficientQAT / Bit-by-Bit / PARQ） | PTQ 在 2-bit 会计算崩溃 |
| 推理 / 数学 / 代码类模型低比特 | **QAT + 混合域校准 + 蒸馏** | 单域校准会丢推理能力 |
| MoE | **逐专家混合精度 + 路由热度** | 频率低的专家反而可能更敏感 |

**永远做的一件事**：用你自己的 Golden Set 对比量化前后的输出，而不是只看 PPL / 平均分。多个 2026 年的 benchmark 都指出，**4-bit 在数学、代码、推理类任务上的损失显著高于常识类任务**（可达 1%–8%），这个分布差异是平均分掩盖掉的。

---

## 11. 附录 A：按年份的论文索引（可核对清单）

> 下列为本文引用并在检索中核实过出处的工作。主线文章已详述的（GPTQ / AWQ / SmoothQuant / QuIP# / AQLM / VPTQ 等）只列条目，细节见对应篇目。

| 年份 / 会议 | 工作 | 一句话 |
|---|---|---|
| ICML 2025 | [PARQ](https://arxiv.org/abs/2503.15748) | 分段仿射正则 + AProx，给 STE 一个理论解释 |
| ICML 2025 | [ParetoQ](https://arxiv.org/abs/2502.02631) | 统一 1/1.58/2/3/4-bit QAT，发现 2↔3-bit 相变 |
| ICML 2025 | [FlatQuant](https://arxiv.org/abs/2410.09426) | 可学习仿射 + Kronecker 分解，W4A4 掉点 <1% |
| ACL 2025 | [EfficientQAT](https://arxiv.org/abs/2407.11062) | Block-AP + E2E-QP，单卡 41h 做 70B 2-bit |
| arXiv 2025.05 | [QAT Scaling Law](https://arxiv.org/abs/2505.14302) | 268 次实验，FC2 激活 outlier 是 W4A4 主瓶颈 |
| arXiv 2025.06 | [UPQ](https://arxiv.org/abs/2506.09104) | FP16→INT4→INT2 渐进 + 蒸馏 QAT，面向 instruct 模型 |
| ACL 2026 | [MXFP PTQ Benchmark](https://arxiv.org/abs/2601.09555) | MXFP8 >98%；旋转类在 MXFP4 上劣于 RTN |
| ICLR 2026 | [ReasoningQAT](https://openreview.net/forum?id=Azsd2qyK6C) | 混合域校准 + 教师引导奖励修正 |
| ICLR 2026 | [MoE 量化泛化保证](https://arxiv.org/abs/2604.06515) | router L2 norm 变化识别脆弱专家 |
| ACL 2026 | [Bit-by-Bit](https://arxiv.org/abs/2604.07888) | 渐进降精度 + 嵌套网格 + outlier 通道拆分 |
| ACL 2026 Findings | [pQuant](https://arxiv.org/abs/2602.22592) | 参数民主化；1-bit 主分支 + 高精度分支 |
| ACL 2026 Findings | [两种失效模式](https://arxiv.org/abs/2604.19884) | 4-bit 信号退化 vs 2-bit 计算崩溃 |
| arXiv 2026.04 | [DuQuant++](https://arxiv.org/abs/2604.17789) | 旋转块对齐 MX group，单次旋转减半开销 |
| arXiv 2025.11 | [DynaExq](https://arxiv.org/abs/2511.15015) | 在线路由热度驱动的 MoE 混合精度服务 |
| EMNLP 2026 | [MODE](https://arxiv.org/abs/2606.17118) | 模态分解的 MoE-MLLM 逐专家混合精度 |
| PyTorch Blog 2026.03 | [TorchAO QAT (II)](https://pytorch.org/blog/quantization-aware-training-in-torchao-ii) | QAT 生产化数字与 NVFP4 prototype |

---

## 12. 附录 B：给继续研究者的问题清单

如果你打算在 2026 年做量化相关的研究或工程，下面这些问题我认为**还没有好答案**：

1. 有没有一个统一判据，能在**量化之前**预测某一层/某一专家会「信号退化」还是「计算崩溃」？
2. MXFP4 上，为什么全局旋转有害、块对齐旋转有益？这个「局部性原理」能否推广成格式选择的一般准则？
3. Reasoning 模型的能力损失，能否用**不依赖教师模型**的方式修复（教师引导的奖励修正需要 fp16 teacher）？
4. QAT 的瓶颈随训练预算从「激活」转移到「权重」（scaling law 结论 4），是否存在一个**自适应的混合精度调度**跟随这个转移？
5. 2-bit kernel 与 4-bit Tensor Core 的吞吐差距在 Blackwell 上还剩下多少？这决定了 ultra-low-bit 是否值得继续卷。
6. 长上下文下 KV 量化误差的累积是线性、对数还是相变？

---

> **姊妹篇**：**[E3 部署侧视角](/2026/09/19/llm-quant-E3-deployment-support/)**——本篇讲「算法处于收敛地图的哪个格子」，
> E3 讲「这个格子在你的机器上跑不跑得起来」（vLLM / SGLang 支持矩阵、kernel 后端、llm-compressor 白名单）。
>
> **与主线的最短路径**：如果你只想从本篇回到主线，按这个顺序读——
> [00 全景](/2026/08/24/ptq-00-overview/) → [01 量化器数学](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) → [03 GPTQ](/2026/08/24/ptq-02-gptq/) → [09 SmoothQuant](/2026/08/24/llm-quant-02-smoothquant-w8a8/) → **[23 统一视角](/2026/08/29/llm-quant-23-unified-view/)** → 本篇 → **E3**。
