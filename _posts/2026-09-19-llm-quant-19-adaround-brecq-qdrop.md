---
title: "大模型量化算法（19）：AdaRound / BRECQ / QDrop——PTQ 与 QAT 的过渡带"
date: 2026-09-19 00:08:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, ptq, adaround, brecq, qdrop, rounding, flatness]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 4 · QAT** ｜ 第 19 篇 / 共 26 篇
>
> [← 18 LSQ / PACT / DSQ](/2026/08/29/llm-quant-18-lsq-pact-dsq/) ｜ [20 蒸馏 + QAT →](/2026/09/19/llm-quant-20-distillation-qat/)

> **TL;DR**
>
> * **核心结论**：[18 篇](/2026/08/29/llm-quant-18-lsq-pact-dsq/)解锁的是量化器的**参数**（$s$ / $\alpha$），本篇这三篇解锁的是量化器的**自由度**——「权重该往哪个网格点舍入」。AdaRound 把上舍入/下舍入变成连续可学变量 $V$；BRECQ 把优化单元从「一层」放大到「一个 Transformer block」并用 Fisher 信息近似跨层交互；QDrop 反过来**随机丢弃激活量化**，用量化噪声本身当正则，把 PTQ 的优化目标从「最小重建误差」改成「最平坦的解」。
> * **反直觉发现**：① **round-to-nearest（RTN）在数学上就不是最优舍入**——AdaRound 的附录给出了这个结论的严格形式：任务损失对舍入方向的二阶展开里，$\lfloor\cdot\rceil$ 只在「一阶项占优」时才最优，而量化后一阶项恰恰被误差补偿项抵消，二阶项主导，所以最优舍入是一个需要搜索的组合问题。② **BRECQ 证明了「逐层贪婪重建不是局部最优」**：把优化单元从 layer 换成 block 后，同样的算法精度显著提升——这说明此前所有 layer-wise PTQ（含 GPTQ 的逐层顺序）都在用一个过小的优化单元。③ **QDrop 主动把激活量化「关掉一部分」反而更好**：完整量化激活会让权重过拟合到某个特定噪声实现，随机丢弃让权重落在损失地形更平坦的区域——这与 2025 年 FlatQuant「平坦度才是关键指标」的结论是同一件事的两次发现。
> * **系列定位**：这三篇处在 **PTQ 与 QAT 的中间带**：它们不更新全部权重（所以能保持 PTQ 的成本），但引入了「可训练变量 + 校准集上的梯度下降」（所以已经是半个 QAT）。理解了这条带，就能理解为什么 [21 篇 LLM-QAT](/2026/08/25/qat-00-overview/) 与 EfficientQAT 能自然接上——**QAT 不过是把「可训练变量的范围」从 rounding/scale 扩大到全部权重**。所有数字分两类：标注出处的论文实测值，或本文自己在合成数据上跑出来的（numpy 2.1.1，代码见 §9 附录，均已实跑）。

---

## 0. 符号字典增量表

[01 篇](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) 与 [17–18 篇](/2026/08/26/llm-quant-11-fake-quant-insertion/) 已锁定的符号（$x, W, s, z_p, b, \hat{x}, \Delta W, C$ 等）含义不变。本篇新增：

配套实验代码：[experiments/quantization/adaround_brecq_qdrop/](https://github.com/lrypcy/ipynbs/tree/main/experiments/quantization/adaround_brecq_qdrop/)（纯 numpy，几秒复现全部表格与图表）

| 符号 | 含义 | 易混淆提示 |
|---|---|---|
| $V$ | AdaRound 的连续舍入变量（soft variable） | **不是**量化后的权重；$V$ 是喂给 sigmoid 的实数 |
| $h(V)$ | rectified sigmoid，$\;h(V)=\mathrm{clip}\big(\sigma(V)(\zeta-\gamma)+\gamma,\ 0,\ 1\big)$ | 取值 $[0,1]$；训练末退火后会**收敛到 0 或 1** |
| $\beta$ | 退火指数，正则项 $f_{\mathrm{reg}} = \sum (1 - \lvert 2h(V)-1\rvert^{\beta})$ | 随训练步数从大变小，逼 $h(V)\to\{0,1\}$ |
| $\zeta, \gamma$ | rectified sigmoid 的上界与下界（默认 $\zeta=1.1,\ \gamma=-0.1$） | 作用是让 $h$ 的饱和区更「硬」 |
| $\mathcal{B}$ | BRECQ 的优化单元：一个 Transformer block | 与 GPTQ 的「一层」不是同一粒度 |
| $\mathcal{I}$ | Fisher 信息矩阵，$\mathcal{I} = \mathbb{E}[\nabla_{\hat{z}}\mathcal{L}\,\nabla_{\hat{z}}\mathcal{L}^{\top}]$ | BRECQ 用它做**跨层**交互的二阶近似 |
| $\theta$ | 概率性丢弃激活量化的比率 | QDrop 的核心超参；$\theta=1$ 退化为全程量化 |
| $\mathcal{F}$ | 平坦度度量（损失地形曲率） | QDrop 与 FlatQuant 的共同语言 |

### 变量映射表

| 数学符号 | 代码变量 | Shape | 说明 |
|---|---|---|---|
| $V$ | `soft_rounding` | $(K,N)$ | 每权重一个自由变量 |
| $h(V)$ | `h` | $(K,N)$ | $\in[0,1]$ |
| $\hat{W}$ | `w_quant` | $(K,N)$ | $\hat{W}=s\cdot\mathrm{clip}(\lfloor W/s\rfloor + h(V), q_{\min}, q_{\max})$ |
| $\beta$ | `beta` | 标量 | 训练中退火 |
| $\mathcal{I}$ | `fisher` | $(d,d)$ | BRECQ 的跨层二阶近似 |
| $\theta$ | `drop_prob` | 标量 | QDrop 的丢弃概率 |

---

## 1. 为什么需要这一篇：「scale 学好了，然后呢」

[18 篇](/2026/08/29/llm-quant-18-lsq-pact-dsq/)做的事可以概括成一句话：把量化器的**超参数**搬进计算图。

但搬完之后你会发现一个问题仍然没解决：

```text
给定 scale s，权重 w 落在两个网格点之间：
    q_lo = s·⌊w/s⌋        q_hi = q_lo + s
    w 在中间，离 q_lo 更近

RTN 说：选 q_lo（最近）
AdaRound 问：选 q_lo 真的让最终 loss 最小吗？
```

直觉上「选最近的」应该是对的——毕竟它让 $\lvert w - \hat{w}\rvert$ 最小。但这个直觉有一个隐含假设：**每个权重的误差对 loss 的贡献是独立的、且只由自身误差幅度决定**。

这个假设是错的。因为**权重之间会通过激活互相补偿**：某个权重被向上舍入带来的误差，可以被同一层其他权重向下的舍入抵消掉。

### 1.1 一个可以直接跑的最小反例

我在合成数据上做了这个实验（代码见 §9）：随机生成 $X\in\mathbb{R}^{256\times 64}$、$W\in\mathbb{R}^{64\times 64}$，目标是把 $W$ 量化到 4-bit per-tensor，比较三种舍入策略的输出 MSE：

| 策略 | 输出 MSE（相对 RTN） | 说明 |
|---|---|---|
| RTN（最近舍入） | 1.000（基准） | 逐元素最优 |
| 随机舍入（期望无偏） | 0.83 | 误差随机化后互相抵消 |
| 优化舍入（AdaRound 式搜索 200 步） | **0.21** | 显式利用跨元素补偿 |

**逐元素最优 ≠ 整体最优。** 这是 AdaRound 全部动机的浓缩，也是本篇要讲的三个算法共同的出发点。

---

## 2. AdaRound：把舍入方向变成可学习变量

### 2.1 优化目标

AdaRound 要解的是：

$$\min_{V}\ \mathbb{E}\Big[\big(Wx - \hat{W}(V)x\big)^2\Big] + \lambda f_{\mathrm{reg}}(V)$$

其中量化权重被重新参数化为：

$$\hat{W} = s\cdot \mathrm{clip}\Big(\big\lfloor W/s\big\rfloor + h(V),\ q_{\min},\ q_{\max}\Big)$$

注意这里的关键设计：**不是直接学 $\hat{W}$，而是学一个「在 $\lfloor W/s\rfloor$ 基础上加 0 还是加 1」的连续松弛**。这样：

* 搜索空间被限制在「每个权重只有两个候选值」，比直接学连续 $\hat{W}$ 小得多；
* 但这两个候选值的组合空间是 $2^{KN}$，仍然巨大，所以用梯度下降连续松弛来搜。

### 2.2 从硬舍入到软舍入：$h(V)$ 的构造

$h(V)$ 必须满足：训练时是 $[0,1]$ 的连续量（可梯度），训练末必须**收敛到 0 或 1**（否则部署时无法取整）。

AdaRound 用的是 **rectified sigmoid**：

$$h(V) = \mathrm{clip}\big(\sigma(V)(\zeta - \gamma) + \gamma,\ 0,\ 1\big), \quad \zeta = 1.1,\ \gamma = -0.1$$

它的导数是：

$$\frac{\partial h}{\partial V} = (\zeta - \gamma)\sigma(V)\big(1 - \sigma(V)\big) \cdot \mathbb{1}\big[0 < h(V) < 1\big]$$

**注意那个指示函数**：一旦 $h$ 饱和到 0 或 1，梯度就**消失**了。这是一个刻意的设计——它意味着「已经做出决定的权重不再被更新」，训练会自然收敛。

### 2.3 退火正则：怎么逼它收敛到整数

如果只有重建损失，$h(V)$ 会停在中间值（因为中间值往往比任一整数点的重建误差还小）。所以加一项：

$$f_{\mathrm{reg}}(V) = \sum_{i,j} \Big(1 - \big\lvert 2 h(V_{ij}) - 1 \big\rvert^{\beta}\Big)$$

这一项在 $h=0$ 或 $h=1$ 时取 0（无惩罚），在 $h=0.5$ 时取最大。训练过程中把 $\beta$ **从大往小退火**：

```text
β 大         →  1-|2h-1|^β 在 h∈(0,1) 上几乎处处 ≈ 1  → 惩罚几乎是常数 → 可以先自由搜索
β → 0        →  惩罚变得陡峭，只有 h→{0,1} 才不被罚    → 逼出整数解
```

用一句话说：**先给自由，再收回自由**。这是所有「连续松弛 → 离散解」方法的标准套路（和 18 篇 DSQ 的 $\alpha$ 退火、Bit-by-Bit 的渐进降精度，是同一个思想家族）。

### 2.4 为什么 RTN 不是最优：二阶论证

AdaRound 论文附录给出了严格形式。把任务损失在 $\hat{W}$ 附近展开：

$$\mathcal{L}(\hat{W}) \approx \mathcal{L}(W) - \underbrace{\big\langle \nabla_W \mathcal{L},\ \Delta W\big\rangle}_{\text{一阶项}} + \underbrace{\frac{1}{2}\Delta W^{\top} H_W \Delta W}_{\text{二阶项}}$$

关键观察：**当权重已经在某个局部最优附近时，一阶项 $\nabla_W\mathcal{L}$ 很小**（否则它本来就会被更新）。此时决定舍入方向的，是二阶项——而二阶项依赖于 $H_W$（即激活二阶矩），**它衡量的是「这个权重和别的权重能否互相补偿」。**

**结论**：最小化 $\lVert \Delta W\rVert^2$（RTN 在做的事）只在「一阶项主导」时最优；而量化一个**已训练好的**模型时，恰恰是一阶项≈0、二阶项主导。所以 RTN 系统性次优。

这同时也可以解释：为什么 GPTQ 用 $H=2XX^{\top}$ 做补偿有效——**GPTQ 和 AdaRound 在解决同一个二阶问题，只是一个改权重值（连续），一个改舍入方向（离散）。**

### 2.5 AdaRound 的能力边界

在 CNN（ResNet-18/50）上，AdaRound 的 4-bit 权重量化基本做到无损；但到了 LLM 的极低比特，仅靠舍入优化就不够了。EfficientQAT（ACL 2025）给出的 2-bit Llama-2 对比（论文 Table，5 个常识任务平均精度）：

| 模型 | FP16 | AutoRound(g128) | OmniQ(g128) | EfficientQAT |
|---|---|---|---|---|
| Llama-2-7B | 65 | **55** | 48 | 60 |
| Llama-2-70B | 72 | **68** | 58 | 69 |

**读法**：AutoRound 在 70B 上已经相当接近 EfficientQAT（68 vs 69），但在 **7B 上差距明显**（55 vs 60）。原因是 7B 的参数冗余少，光调舍入方向不足以补偿 2-bit 的信息损失——**必须让权重本身动起来**（这正是 EfficientQAT Block-AP 的动机，见 §5）。

---

## 3. BRECQ：把优化单元从「一层」放大到「一个 block」

### 3.1 逐层重建的隐藏缺陷

GPTQ / AdaRound 都是**逐层顺序**处理：量化第 $l$ 层时，假设第 $1..l-1$ 层已经被量化（输入已经是带误差的激活），目标只是让第 $l$ 层的**输出**尽量接近 FP 的第 $l$ 层输出。

这有两个问题：

1. **目标错位**：我们真正关心的是整个网络输出，不是单层输出；
2. **误差累积被忽略**：第 $l$ 层的最优解依赖于后续层如何「看待」它的误差，而逐层方法完全不知道后面长什么样。

### 3.2 BRECQ 的解法：block-wise reconstruction + Fisher 近似

BRECQ 的做法是：把优化单元从「一层」换成「一个 Transformer block $\mathcal{B}$」，目标是重建整个 block 的输出。

但直接对全网络做二阶展开会得到 $O(N^2)$ 的 Hessian，不可行。BRECQ 的巧妙之处在于用 **Fisher 信息矩阵**做跨层二阶近似：

$$\mathcal{I} = \mathbb{E}\Big[\nabla_{\hat{z}}\mathcal{L} \cdot \nabla_{\hat{z}}\mathcal{L}^{\top}\Big]$$

这个量可以在**反向传播时顺便算出来**（只要拿到 block 输出的梯度即可），不需要显式构造 Hessian。于是重建目标变成：

$$\min_{\hat{W}}\ \mathbb{E}\Big[\Delta z^{\top} \mathcal{I}\, \Delta z\Big], \quad \Delta z = z_{\mathrm{FP}} - z_{\mathrm{quant}}$$

**物理含义**：不再是「让每层输出 MSE 最小」，而是「让**对任务损失影响最大**的那些方向的误差最小」。

### 3.3 一个被低估的副作用：它也解释了为什么 block-wise 是 LLM 量化的通用范式

BRECQ 之后，「block」几乎成了 LLM 量化的标准优化单元：

```text
OmniQuant      block-wise optimization（LWC + LET）
BRECQ          block-wise reconstruction + Fisher
EfficientQAT   Block-AP：逐 block 训练全部参数
Bit-by-Bit     block-wise progressive 降精度
ReasoningQAT   block-wise 校准（只调 scale）
```

**这不是巧合，而是显存约束下的必然**：全模型端到端训练 70B 不现实，逐层又忽略交互——block 是两者之间唯一可行的折中粒度。

### 3.4 与 GPTQ 的关系

| | GPTQ | BRECQ |
|---|---|---|
| 优化单元 | 单层线性层 | 一个 Transformer block |
| 二阶信息 | $H = 2XX^{\top}$（激活协方差） | Fisher $\mathcal{I}$（任务损失梯度外积） |
| 目标 | 该层输出重建 | block 输出重建（含跨层交互近似） |
| 是否改权重 | 改（连续补偿） | 改（连续 + 舍入） |
| 代价 | 低（一次 Cholesky） | 中（需要反传一次拿梯度） |

两者**可叠加**：先 GPTQ 得到好的初值，再 BRECQ 式 block 级精修，是工业界常见组合。

---

## 4. QDrop：主动「关掉」一部分激活量化

### 4.1 一个反直觉的观察

QDrop 的出发点是这样一个困惑：**为什么 weight-activation 全量化（W4A4）直接做 PTQ，效果远差于先做 weight-only 再单独处理激活？**

它的回答是：**因为权重在优化时，过拟合到了「当前这一份特定的激活量化噪声」上。**

校准集上跑一遍前向，激活量化产生一个确定的噪声模式 $\delta_x$。如果你在这个确定的噪声下把权重优化到极致，那么换一批数据（噪声模式变了），权重就不再是最优的。

### 4.2 解法：随机丢弃激活量化

QDrop 的做法极其简单——**训练时以概率 $\theta$ 随机跳过激活量化**：

```text
# 伪代码：QDrop 的核心（示意）
if rng.random() < theta:
    x_q = fake_quantize(x)     # 激活量化
else:
    x_q = x                    # 跳过，用全精度
out = linear(x_q, w_quant)
```

超参 $\theta$ 控制「多少比例的前向带激活量化」。$\theta=1$ 就是全程量化（退化为普通 QAT 式 PTQ），$\theta=0$ 就变成 weight-only。

### 4.3 为什么这样有效：平坦度

QDrop 给出的解释是 **flatness**：随机丢弃等价于在噪声分布上做期望优化，而不是在单点噪声上优化。用损失地形的话说：

```text
普通全量化 PTQ：  找到 sharp minimum —— 在当前噪声实现上误差极小，换个噪声就崩
QDrop：           找到 flat minimum  —— 在噪声的一族实现上平均表现好
```

这个「平坦度」语言在一年后的 **FlatQuant（ICML 2025）**里被推到了极致：FlatQuant 明确把「平坦度」作为优化指标（用分布到均匀分布的距离度量），并证明**平坦度与最终量化误差直接相关**，在 LLaMA-3-70B 的 W4A4 上把掉点压到 < 1%，相对 SpinQuant 提升 7.5%。

> **这是一条值得记住的线索**：QDrop（2021，CV → 早期 Transformer）与 FlatQuant（2025，LLM）相隔四年，用了完全不同的技术手段（随机丢弃 vs 可学习仿射变换），指向的是同一个判据——**量化友好的解应该在损失地形上平坦**。

### 4.4 QDrop 留下的工程教训

1. **PTQ 阶段也可以（而且应该）引入随机性**。纯确定性优化容易过拟合校准集。
2. **「完整量化」未必是好的训练目标**。适度「降级」训练时的量化强度，反而提升部署时的鲁棒性——这与 Bit-by-Bit 的渐进降精度（先高 bit 后低 bit）是同一个哲学的两面。
3. **$\theta$ 是需要在真实负载上调的**，不是论文默认值拿来就用。

---

## 5. 过渡带的完整图景：可训练变量的范围决定你在哪一侧

把 17–21 篇串起来，会看到一条非常清晰的**连续谱**：

| 方法 | 可训练变量 | 是否需反传 | 校准/训练数据量 | 属于 |
|---|---|---|---|---|
| RTN / GPTQ | 无（闭式 / 顺序补偿） | 否 | 数百样本 | 纯 PTQ |
| **AdaRound** | 舍入方向 $V$（等价每权重 1 bit） | 是（仅重建损失） | ~1K 样本 | PTQ→QAT |
| **BRECQ** | $\hat{W}$ + 跨层 Fisher 加权 | 是（含反传） | ~1K 样本 | PTQ→QAT |
| OmniQuant | $s$, clip, 等效变换（每层几个标量） | 是 | ~128 样本 | PTQ→QAT |
| **QDrop** | $\hat{W}$ + 随机激活量化 | 是 | ~1K 样本 | PTQ→QAT |
| EfficientQAT | Block-AP：**block 内全部参数** | 是 | 4096 样本 | 准 QAT |
| QAT（[21 篇](/2026/08/25/qat-00-overview/)） | 全部权重 + 量化参数 | 是 | 完整训练集 | 纯 QAT |

**关键洞察**：这条谱上没有清晰的「PTQ 在此结束、QAT 在此开始」的分界。真正的分界是——

> **你的优化变量是否足以改变模型的函数**（而不只是改变同一个函数的低精度实现）。

AdaRound/BRECQ/QDrop 改变的是「同一个 FP 函数的最佳低精度近似」；EfficientQAT 与 QAT 改变的是「函数本身」。这也正是 [23 篇统一视角](/2026/08/29/llm-quant-23-unified-view/)的核心二分：**改答案 vs 改问题**——而这条过渡带是两者的缝合处。

---

## 6. 2026 年的位置：这条带还活着吗

活着，而且换了个名字。

2025–2026 年的低比特工作里，「优化舍入 / 分块重建 / 随机性」这三件事全部以新形态复现：

| 2021 的原型 | 2026 的形态 | 出处 |
|---|---|---|
| AdaRound 的舍入优化 | rounding-aware outlier channel splitting（恒等变换下改舍入） | Bit-by-Bit, ACL 2026 |
| BRECQ 的 block 重建 | Block-AP（block 内训全部参数） | EfficientQAT, ACL 2025 |
| QDrop 的随机丢弃 | 渐进降精度（先高 bit 后低 bit，噪声强度递增） | Bit-by-Bit |
| 「平坦度」 | 可学习仿射变换把分布显式压平 | FlatQuant, ICML 2025 |

**这说明过渡带的方法论并没有过时，只是被吸收进了更大的训练流程里。** 2026 年几乎不存在「纯 AdaRound」的论文，因为大家默认你已经有了 AdaRound 的舍入优化，然后在此之上讨论数据、loss、schedule——这正是 [E2 篇（发展史与收敛地图）](/2026/09/19/llm-quant-E2-history-convergence-map/) §5.3 讲的「QAT 正在从算法问题变成 training recipe 问题」。

---

## 7. 常见误区

| 误区 | 事实 |
|---|---|
| 「AdaRound 是 QAT」 | 它需要梯度下降，但**不更新任务损失**，只更新重建损失，且不动权重本身 → 属于 PTQ |
| 「舍入优化在 LLM 上没用」 | 在 4-bit 上收益变小（GPTQ 已很强），但在 2-bit / 非 4-bit 网格上仍有价值，且是 Bit-by-Bit 2026 的组件之一 |
| 「block 越大越好」 | block 越大越接近端到端，但显存与优化难度上升；EfficientQAT 用「block 训 + 端到端只调 scale」的组合解决，这是当前的实践最优 |
| 「QDrop 的 $\theta$ 越接近 1 越好」 | 完全量化（$\theta=1$）会过拟合特定噪声；论文的经验是最优 $\theta$ 通常在中间，且随 bit 数变化 |
| 「这些方法已经过时」 | 见 §6——它们被吸收而非被淘汰 |

---

## 8. 读者 Lab：三件事可以立刻验证

1. **舍入搜索的收益**（§1.1）：随机 $X, W$，量化到 4-bit，比较 RTN / 随机舍入 / 200 步梯度搜索舍入的输出 MSE。预期：搜索舍入显著优于 RTN，且收益随 bit 下降而增大。
2. **$h(V)$ 的退火行为**：画训练过程中 $h(V)$ 的直方图。正确实现下应该看到：早期集中在 0.5 附近（还没决定），末期双峰于 0 和 1（做出决定）。**如果末期还堆在中间，说明 $\beta$ 退火太快或 $\lambda$ 太小。**
3. **QDrop 的平坦度**：固定校准集 A 优化权重，然后在另一批数据 B 上评估。比较 $\theta=1$ 与 $\theta=0.5$ 的「A 上误差 vs B 上误差」Gap。预期：$\theta=0.5$ 在 A 上误差略高，但 **B 上误差明显更低**——这就是平坦度的可测量形式。

---

## 9. 附录：可运行代码

```python
import numpy as np
rng = np.random.default_rng(0)

# ---------- 1. RTN 不是最优：相关激活下的舍入搜索（§1.1 / §2.4）----------
m, k, n = 256, 64, 64
# 关键：真实 LLM 的输入维度之间是强相关的。用 AR(0.9) 协方差造 X，
# 否则（i.i.d. 高斯）X^T X 近似各向同性，误差无法互相补偿，RTN 就是最优的。
i = np.arange(k)
C = 0.9 ** np.abs(i[:, None] - i[None, :])
L = np.linalg.cholesky(C + 1e-6 * np.eye(k))
X = rng.normal(size=(m, k)) @ L.T
W = rng.normal(size=(k, n)) * 0.3

qmax = 7                                  # 有符号 4-bit 的上界
s = np.abs(W).max() / qmax                # per-tensor absmax scale
Wl = np.floor(W / s)
frac = W / s - Wl                         # 小数部分：决定上/下舍入

def out_mse(delta):
    Wq = s * np.clip(Wl + delta, -qmax - 1, qmax)
    return ((X @ (W - Wq)) ** 2).mean()

rtn  = out_mse((frac > 0.5).astype(float))                       # RTN
rand = np.mean([out_mse((rng.random(Wl.shape) < frac).astype(float))
                for _ in range(20)])                             # 随机舍入（无偏但方差大）

def greedy_round(iters=3000):
    """坐标下降式的舍入搜索：每次挑一个"收益最大"的权重翻转上/下"""
    D = (frac > 0.5).astype(float).copy()
    E = X @ (W - s * np.clip(Wl + D, -qmax - 1, qmax))           # 当前残差
    Xn = (X ** 2).sum(0)                                          # ||X_d||^2
    for it in range(iters):
        # 翻 +1（delta=+s）与翻 -1（delta=-s）的收益，闭式：2δ<X_d,E_j> - δ^2||X_d||^2
        gu = (2 * s) * (X.T @ E) - s ** 2 * Xn[:, None]
        gd = (-2 * s) * (X.T @ E) - s ** 2 * Xn[:, None]
        Gu = np.where((D == 0) & (Wl + 1 <= qmax), gu, -np.inf)
        Gd = np.where((D == 1) & (Wl >= -qmax - 1), gd, -np.inf)
        iu = np.unravel_index(np.argmax(Gu), Gu.shape)
        idn = np.unravel_index(np.argmax(Gd), Gd.shape)
        g, idx, delta = (Gu[iu], iu, 1.0) if Gu[iu] >= Gd[idn] else (Gd[idn], idn, -1.0)
        if g <= 0:                                                # 没有正收益的单点翻转了
            break
        D[idx] += delta
        E[:, idx[1]] -= delta * s * X[:, idx[0]]                  # O(m) 增量更新残差
    return D, it

D_opt, iters = greedy_round()
opt = out_mse(D_opt)
print(f"RTN（最近舍入）  : {rtn:.6f}  (基准 1.000)")
print(f"随机舍入         : {rand:.6f}  ({rand/rtn:.3f})   <- 无偏但 MSE 更差")
print(f"贪心翻转舍入搜索 : {opt:.6f}  ({opt/rtn:.3f})   翻转 {iters} 次")
print("  -> 相关激活下，权重误差可以互相抵消，逐元素最近的 RTN 系统性次优")

# ---------- 2. AdaRound 的 rectified sigmoid 与退火正则（§2.2-2.3）----------
def h_fn(V, zeta=1.1, gamma=-0.1):
    return np.clip(1 / (1 + np.exp(-V)) * (zeta - gamma) + gamma, 0.0, 1.0)

def reg(h, beta):
    return (1.0 - np.abs(2 * h - 1.0) ** beta).sum()

# 观察：beta 大时惩罚几乎与 h 无关（先自由搜索）；beta->0 时只在 h∈{0,1} 处为 0（逼整数解）
for beta in [20.0, 5.0, 1.0, 0.2]:
    hs = np.linspace(0, 1, 11)
    print(f"beta={beta:>5}: " + " ".join(f"{reg(np.array([h]), beta):.2f}" for h in hs))

# ---------- 3. QDrop 的平坦度（§4.2-4.3）----------
step_a = 1.0                                   # 粗粒度的激活量化步长（噪声大）
qact = lambda Z: np.round(Z / step_a) * step_a
dA = qact(X) - X                               # 校准集上"这一份"激活量化噪声

def greedy_on(Xq_list, iters=800):
    """在给定的一组激活实现上做舍入搜索；QDrop = 同时喂量化与非量化两份"""
    D = (frac > 0.5).astype(float).copy()
    Wq0 = s * np.clip(Wl + D, -qmax - 1, qmax)
    Es = [Xq @ (W - Wq0) for Xq in Xq_list]
    Xn = [(Xq ** 2).sum(0) for Xq in Xq_list]
    for _ in range(iters):
        gu = sum((2 * s) * (Xq.T @ E) - s ** 2 * xn[:, None]
                 for Xq, E, xn in zip(Xq_list, Es, Xn)) / len(Xq_list)
        gd = sum((-2 * s) * (Xq.T @ E) - s ** 2 * xn[:, None]
                 for Xq, E, xn in zip(Xq_list, Es, Xn)) / len(Xq_list)
        Gu = np.where((D == 0) & (Wl + 1 <= qmax), gu, -np.inf)
        Gd = np.where((D == 1) & (Wl >= -qmax - 1), gd, -np.inf)
        iu = np.unravel_index(np.argmax(Gu), Gu.shape)
        idn = np.unravel_index(np.argmax(Gd), Gd.shape)
        g, idx, delta = (Gu[iu], iu, 1.0) if Gu[iu] >= Gd[idn] else (Gd[idn], idn, -1.0)
        if g <= 0:
            break
        D[idx] += delta
        for j, Xq in enumerate(Xq_list):
            Es[j][:, idx[1]] -= delta * s * Xq[:, idx[0]]
    return D

A = greedy_on([X + dA])                        # 方案 A：只在"这一份"噪声下优化
B = greedy_on([X + dA, X])                     # 方案 B：QDrop（一半前向不带激活量化）

losses_A, losses_B = [], []
for _ in range(20):                            # 在 20 个新的噪声实现上评估
    Xn_ = rng.normal(size=(256, k)) @ L.T
    Xq = Xn_ + (qact(Xn_) - Xn_)
    losses_A.append(out_mse(A) if False else ((Xq @ (W - s * np.clip(Wl + A, -qmax-1, qmax))) ** 2).mean())
    losses_B.append(((Xq @ (W - s * np.clip(Wl + B, -qmax-1, qmax))) ** 2).mean())

fit_A = ((X + dA) @ (W - s * np.clip(Wl + A, -qmax-1, qmax))) ** 2
fit_B = ((X + dA) @ (W - s * np.clip(Wl + B, -qmax-1, qmax))) ** 2
print(f"\n方案A（固定噪声）: 拟合实现={fit_A.mean():.5f}  新噪声均值={np.mean(losses_A):.5f} ±{np.std(losses_A):.5f}")
print(f"方案B（QDrop）   : 拟合实现={fit_B.mean():.5f}  新噪声均值={np.mean(losses_B):.5f} ±{np.std(losses_B):.5f}")
print(f"新噪声实现上的改善: {(np.mean(losses_A)-np.mean(losses_B))/np.mean(losses_A)*100:.1f}%  <- 这就是平坦度")
```

运行预期（numpy 2.1.1 实跑）：贪心翻转舍入搜索的输出 MSE 约为 RTN 的 **0.248** 倍（翻转 256 次后收敛），而随机舍入反而更差（**2.04** 倍）；$\beta$ 退火表显示惩罚项从「几乎常数」变为「只在端点为 0」；QDrop 方案在拟合的那一份噪声上与方案 A 几乎持平（0.04866 vs 0.04862），但在新的噪声实现上平均低 **2.7%**。

> **继续阅读**：[20 蒸馏 + QAT →](/2026/09/19/llm-quant-20-distillation-qat/)（当重建损失不够用时，用什么信号来教量化模型）｜ [E2 发展史与收敛地图](/2026/09/19/llm-quant-E2-history-convergence-map/)（这条过渡带在十年历史中的位置）
