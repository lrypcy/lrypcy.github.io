---
title: "大模型量化算法（23）：LLM PTQ 统一视角——所有算法到底在优化什么"
date: 2026-08-29 10:00:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, ptq, qat, survey, unified-view]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 5 · 横向专题** ｜ 第 23 篇 / 共 26 篇
>
> 22 Reasoning QAT（待写） ｜ [24 KV Cache 量化 →](/2026/08/29/llm-quant-24-kv-cache/)
>
> **本篇是全系列的承重墙**。读它不需要先读完前 22 篇，但读完前 22 篇再来读它会更有收获——因为它要做的事，是把你已经见过的二十来个算法名字，还原成**同一个优化问题在不同自由度上的不同解法**。

> **TL;DR**
>
> * **核心结论**：所有 LLM PTQ 算法都在解同一个问题——$\min_{\hat{W}}\ \mathrm{tr}\big(\Delta W\,C\,\Delta W^\top\big)$，其中 $\Delta W = W - \hat{W}$、$C=\mathbb{E}[xx^\top]$。它们之所以看起来千差万别，是因为这个问题的**设计空间有八个自由度**，而每个算法选择在其中一两个上发力。把这八个自由度摊开，GPTQ / AWQ / SmoothQuant / QuaRot / SpQR / AQLM 之间的区别就不再是"又一个新方法"，而是**同一张表上的不同格子**。
> * **反直觉发现**：① **绝大多数算法并没有改变被量化的函数**——SmoothQuant、AWQ、QuaRot、OS+ 施加的都是数学上严格恒等的变换，它们改的是"这个函数长什么样"而不是"这个函数是什么"；真正改变函数的只有 QAT 一类。② **"改答案"类方法（GPTQ/AdaRound）和"改问题"类方法（SmoothQuant/AWQ）可以叠加，而且工业界的 SOTA 基本都是叠加的结果**——把它们当成互斥选项去比较，是选型时最常见的错误。③ **bit 数不是算法的核心属性**：同样是 W4A16，per-tensor + RTN 与 per-group + GPTQ + AWQ 缩放是两种完全不同的精度水平，把它们都叫"4-bit"会让选型失去意义。
> * **系列定位**：本篇不引入任何新算法，而是给前 22 篇做一次**横向收束**。如果说 [00 篇](/2026/08/24/ptq-00-overview/)给的是"地图"，本篇给的是"地质学"——解释这片地形为什么长成这样。读完它，你应该能对一个从没见过的新量化论文，在五分钟内定位它动的是哪个自由度、预期收益上限在哪。

---

## 1. 为什么要写这一篇：论文列表化的陷阱

这个系列写到第 22 篇的时候，出现了一个值得警惕的信号。

如果我问"GPTQ 和 AWQ 有什么区别"，一个读完前面所有文章的人很可能回答："GPTQ 用 Hessian 做二阶补偿，AWQ 用激活幅度找显著通道。" 这个回答没错，但它是一个**记住了答案**的回答，而不是一个**能推导**的回答。

再追问一句："那 SmoothQuant 和 AWQ 呢？它们都用等效变换，可以互相替代吗？"

能答上来的人会少一大半。

问题出在学习方式上。按年代逐个读算法，得到的是一张**清单**：

```text
RTN → LLM.int8() → GPTQ → AWQ → OmniQuant → SpQR → HQQ
    → QuIP# → AQLM → SmoothQuant → ZeroQuant → QuaRot
    → SpinQuant → SqueezeLLM → VPTQ → RPTQ → ATOM → OliVe → QServe
```

清单的问题不是它不对，而是它**不产生判断力**。知道 18 个算法名字的人，面对第 19 个算法时，仍然只能从头读一遍，然后把它挂在清单末尾。

更糟的是，清单会诱导一种错误的心智模型：**这些算法是彼此竞争的同类选项，我们要从中挑一个最好的。**

这个模型是错的，而且错得很有代价。真实情况是——它们大多在解决**同一个优化问题的不同侧面**，彼此之间主要是**可叠加**关系，而不是**互斥**关系。

这一篇要做的事，就是把清单换成结构。

---

## 2. 一个问题：所有 PTQ 算法共同的起点

先立目标。整个系列反复出现的那个层损失，在这里正式成为主角。

对单个线性层，量化权重 $\hat{W}$ 的合理目标不是"让 $\hat{W}$ 接近 $W$"，而是"让 $\hat{W}$ 的**输出**接近 $W$ 的输出"：

$$\min_{\hat{W}}\ \mathcal{L}(\hat{W}) = \mathbb{E}_x\Big[\ \lVert Wx - \hat{W}x \rVert_2^2\ \Big]$$

把 $\Delta W \triangleq W - \hat{W}$ 代进去，利用 $\mathbb{E}[\cdot]$ 的线性与迹的循环性：

$$\mathcal{L} = \mathbb{E}_x\Big[\ x^\top \Delta W^\top \Delta W\, x\ \Big] = \mathbb{E}_x\Big[\ \mathrm{tr}\big(\Delta W^\top \Delta W\, x x^\top\big)\ \Big] = \mathrm{tr}\big(\Delta W\, C\, \Delta W^\top\big)$$

其中

$$C \triangleq \mathbb{E}_x\big[\,x x^\top\,\big] \ \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

就是**[01 篇](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) §6.2** 反复出现的激活二阶矩矩阵（GPTQ 里叫 Hessian，$H = 2XX^\top$，差一个常数因子 2）。

这一行推导是整个系列的枢纽，值得停下来看清楚它说了三件事：

1. **误差是被 $C$ 加权的，不是等权的**。$C$ 的大特征方向上的权重误差，代价远高于小特征方向。这是 AWQ "问激活而不是问权重" 的全部依据。
2. **$\hat{W}$ 的取值被约束在一个离散集合里**（量化网格），所以这不是普通的二次优化，而是**离散约束下的二次优化**——本质上难。
3. **这个目标只覆盖一层**。真实网络是层的复合，层间误差会累积、会被 LayerNorm 重新归一化、会在 attention 里被 softmax 非线性放大。所以层最优不等于网络最优——这是所有 layer-wise PTQ 方法的共同软肋（后面 §6 会回到这一点）。

好，现在关键的一步：

> **前面那 22 篇文章里的所有算法，都可以看作是在回答"如何在一个巨大的离散约束下，让 $\mathrm{tr}(\Delta W C \Delta W^\top)$ 尽量小"。**

它们之所以长得不一样，是因为这个问题的**设计空间是八维的**。下面把八维摊开。

---

## 3. 八个自由度：一张完整的算法设计空间表

给定层权重 $W$、校准激活统计量 $C$、目标比特 $b$，要把 $W$ 变成 $\hat{W}$，你其实要做八个决定：

| # | 自由度 | 你在决定什么 | 常见取值 |
|---|---|---|---|
| **F1** | **网格类型**（grid） | 允许 $\hat{W}$ 取哪些值 | 均匀 / 非均匀(NF4、k-means) / 向量码本 / 浮点(FP8、MXFP4) |
| **F2** | **网格位置与间距**（scale & zero-point） | 网格放在数轴哪里、多密 | min-max / percentile / MSE 最优 / 解析(HQQ) / 学习(LSQ) |
| **F3** | **粒度**（granularity） | 有多少个独立网格 | per-tensor / per-channel / per-group / per-token |
| **F4** | **落点规则**（rounding） | 每个 $w$ 落到网格哪个点 | 就近(RTN) / 二阶补偿(GPTQ) / 优化(AdaRound) / 码本搜索 |
| **F5** | **等效变换**（reparameterization） | 能否先恒等变换 $W,x$ 再量化 | 对角缩放(SmoothQuant/AWQ) / 正交旋转(QuaRot) / $\gamma$ 迁移(OS+) |
| **F6** | **事后补偿**（compensation） | 量化完还能不能再补 | Hessian 补偿 / 低秩补偿(LoRC) / bias correction |
| **F7** | **混合精度**（outlier handling） | 是否给一部分元素特殊待遇 | 均匀 / 通道级混合(OWQ) / 元素级稀疏(SpQR) / 配对编码(OliVe) |
| **F8** | **模型适应**（adaptation） | 是否允许改 $W$ 本身 | 不动(PTQ) / 局部重构(BRECQ) / 全局微调(QAT) |

这八个自由度构成了完整的算法坐标。任何一个量化算法，都可以在这张表上标出一个坐标。

下面逐条讲清楚每个自由度的**数学抓手、收益来源、代价**。

---

### F1 · 网格类型：允许 $\hat{W}$ 取哪些值

最朴素的假设是**均匀网格**：$\hat{w} \in \{s(q - z) \mid q \in \{0,\dots,2^b-1\}\}$。它便宜（整数 GEMM 直接吃），但对高斯型权重是浪费——权重大概率集中在 0 附近，均匀网格把宝贵的码字匀给了很少出现的尾部。

三条逃逸路线：

- **非均匀标量网格**：让网格点本身按分布密度排布。QLoRA 的 NF4 是典型——按标准正态的分位数取 $2^4$ 个码字，实现"信息论最优的分位量化"，代价是查表而非整数运算。GGUF 的 k-quants 则是"分组 + 变步长"的位布局工程化（[15 篇](/2026/08/24/ptq-08-gguf-fp8-mxfp4/)）。
- **浮点网格**：FP8 (E4M3/E5M2)、MXFP4 用指数位换取动态范围，让网格在对数尺度上近似均匀。硬件原生支持时，它比整数网格更贴合重尾分布。
- **向量量化**：不再给每个标量单独选点，而让**一批**权重共享一个码本、用索引代替数值。QuIP# 用 E8 格做免训练码本，AQLM 用多个小码本相加（[07 篇](/2026/08/24/ptq-05-quip-aqlm/)）。这是 2-bit 及以下能保住精度的几乎唯一路线。

**收益来源**：网格贴合分布，等价于在相同比特下降低有效量化噪声。
**代价**：非均匀网格几乎都要查表或解码，失去整数 GEMM 的硬件红利。**这是"算法精度"与"硬件速度"之间最经典的一次交换**。

---

### F2 · 网格位置与间距：scale 与 zero-point

给定网格类型，还要决定它放在数轴何处、间距多大。这个决定比很多人以为的重要得多。

[01 篇 §4](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) 已经推导过：教科书公式 $s = \max|w| / 2^{b-1}$ 在高斯型权重上**远非最优**，因为它让极少数极端值决定全局格距。主动裁掉一部分尾部（引入裁剪误差）反而能降低总误差：

$$\min_{s}\ \mathbb{E}\Big[\big(w - \hat{w}(s)\big)^2\Big] \quad\text{s.t.}\quad \hat{w} = s\cdot\mathrm{clip}\!\left(\big\lfloor w/s \rceil,\ q_{\min},\ q_{\max}\right)$$

因为总误差 = 舍入误差 + 裁剪误差，而两者随 $s$ 此消彼长。

工程上的选择谱系：

| 方法 | 怎么用 | 代价 |
|---|---|---|
| min-max | 直接用极值 | 对 outlier 极敏感 |
| percentile / 分位数 | 截掉 $p\%$ 尾部 | $p$ 需调，且分布依赖 |
| MSE 最优 | 网格搜索或闭式解 | 需遍历候选 $s$ |
| KL 校准 | 最小化量化前后分布 KL | 传统 INT8 推理常用，LLM 上较少 |
| 解析求解（HQQ） | 稀疏化 + $\ell_p$ 范数闭式 | 免校准集，精度有上限 |
| 学习（LSQ/PACT） | 把 $s$、clip 界放进计算图 | 需训练（[18 篇](/2026/08/29/llm-quant-18-lsq-pact-dsq/)） |

**这一自由度常被低估**。在 4-bit 下，从 min-max 换到 MSE 最优 scale 的收益，往往不比换一个量化算法小。

---

### F3 · 粒度：有多少个独立网格

一个 scale 管整个矩阵（per-tensor），还是每行一个（per-channel），还是每 128 个元素一个（per-group）？

[01 篇 §7](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) 的实测给出了粒度的收益来源：**outlier 的空间局部性**。行间 outlier 用 per-channel 就能隔离；行内散点必须 per-group。

**代价是元数据**：group 越小，scale/zero-point 的存储占比越高。以 FP16 scale + 4-bit 权重、$g=128$ 为例，每个 scale 摊到 128 个权重上，额外开销约 $16/128 = 0.125$ bit/权重，即 ~3%；$g=32$ 时涨到 ~12%。所以**粒度不是越细越好**，它有一条明确的收益递减曲线。

这里还埋着一个 AWQ 的关键前提：[04 篇](/2026/08/24/llm-quant-03-awq-scale-search/)实测发现，**在逐输入通道粒度下 AWQ 的缩放严格无效**（扫描曲线逐点完全相同），因为每个通道独享 scale 时，放大这个动作被自己的 scale 精确抵消了。AWQ 有效的前提是**组内共享 scale**——这正是一个"自由度之间存在依赖关系"的例子：F5 的收益依赖 F3 的取值。

---

### F4 · 落点规则：每个权重往哪落

网格定了、scale 定了，剩下的就是每个 $w$ 落到哪个点。这是最"算法"的一维。

- **RTN**：逐元素就近。它是在**忽略 $C$ 加权、忽略元素间耦合**前提下的最优解——[01 篇 §6.1](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/)证明了它的逐元素最优性，也证明了这个最优性与层最优性之间的鸿沟。
- **GPTQ**：把"已经量化的那些权重造成的误差"用**未量化权重**补偿掉。用 $H=2XX^\top$ 做二阶近似，Lagrange 乘子法给出闭式更新：
  $$\delta = -\frac{w_q - \hat{w}_q}{[H^{-1}]_{qq}} \cdot (H^{-1})_{:,q}$$
  核心洞察是：**量化误差不是白噪声，可以通过调整其他权重主动抵消**（[03 篇](/2026/08/24/ptq-02-gptq/)）。
- **AdaRound / BRECQ**：更直接——把"向上还是向下"本身当成**待优化变量** $V$，用软松弛 + 局部重构误差去优化它（[19 篇](/2026/08/29/llm-quant-19-adaround-brecq/)）。

**收益来源**：从"每个元素独立最优"升级到"元素之间协同最优"。
**代价**：需要校准集、需要计算 $H$（及其 Cholesky/逆），且存在顺序依赖与局部最优。

---

### F5 · 等效变换：改问题，而不是改答案

这是整个设计空间里**最优雅、也最容易被误解**的一维。

核心想法：对任意可逆矩阵 $D$，有严格恒等式

$$XW = \big(X D^{-1}\big)\big(D W\big)$$

函数**分毫未变**，但 $X$ 和 $W$ 的**数值分布**变了，量化的难度也随之改变。这不是近似，是恒等。

沿着这条思路的三个里程碑：

| 方法 | 变换形式 | 干了什么 |
|---|---|---|
| **SmoothQuant**（[09 篇](/2026/08/24/llm-quant-02-smoothquant-w8a8/)） | 对角 $\tau_j = a_j^{\beta}/w_j^{1-\beta}$ | 把激活的量化难度**迁移**给权重，实现 W8A8 |
| **AWQ**（[04 篇](/2026/08/24/llm-quant-03-awq-scale-search/)） | 对角 $\kappa_j = a_j^{\gamma}$ | 在**组内**给显著通道分配更细的格距 |
| **QuaRot / SpinQuant**（[12 篇](/2026/08/24/ptq-07-quarot-spinquant/)） | 正交旋转 $R$：$XW = (XR)(R^{-1}W)$ | 把 outlier 从少数坐标**摊薄**到所有坐标 |
| **OS+**（[11 篇](/2026/08/24/ptq-10-outlier-suppression/)） | $\gamma$ 迁移 + shift | 拆掉 LayerNorm 这个"outlier 放大器" |

**一个必须讲清的区分**：SmoothQuant 和 AWQ 都做对角缩放，但它们**优化的目标不同**——SmoothQuant 追求的是"让激活变得可量化"（迁移难度），AWQ 追求的是"让权重的重要通道得到更细格距"（分配精度）。前者服务于 W8A8，后者服务于 W4A16。把它们当成同一种东西，就会得出"二选一"的错误结论。

**代价**：F5 是恒等变换，所以**数学上免费**；代价全在工程侧——变换必须能融合进相邻权重（否则多一次矩阵乘），残差分支、LayerNorm、RoPE 都会给变换链的传播制造障碍。这是 QuaRot 一类方法落地时真正难的地方。

---

### F6 · 事后补偿：量化完还能不能再捞回来

量化做完了，$\Delta W$ 已经产生，还能补救吗？

- **Hessian 补偿**（GPTQ）：见 F4，它其实是边量化边补偿。
- **低秩补偿（LoRC）**：把残差 $\Delta W$ 做低秩近似 $\Delta W \approx U V^\top$，额外存这两个小矩阵，推理时补上。[16 篇](/2026/08/24/ptq-13-qserve-qqq/)实测指出：**它的收益高度依赖残差谱的集中度**——残差谱越集中收益越大，这解释了为什么 LoRC 在不同论文里时好时坏。
- **Bias correction**：量化引入的误差往往有系统性偏置（不是零均值），直接把它补偿到层的 bias 项上。便宜，但只能修一阶矩。

**代价**：都要额外的存储与访存，而且补偿本身也会引入新的量化误差。

---

### F7 · 混合精度：给 outlier 特殊待遇

如果 99% 的权重很好量化，但 1% 极端敏感，那最经济的做法就是**给这 1% 开小灶**。

- **通道级**：OWQ 按 Hessian 对角元挑敏感**通道**，整通道保留 FP16。
- **元素级**：SpQR 挑敏感**元素**，用稀疏格式存；SqueezeLLM 用对角 Fisher 信息挑（[08 篇](/2026/08/24/ptq-09-squeezellm-vptq-claq/)）。
- **格式级**：OliVe 把"outlier-受害者配对"直接编码进数据格式，避开索引与全局协调（[14 篇](/2026/08/24/ptq-12-olive-abfloat/)）。

**收益来源**：精度收益非常显著，尤其在 2–3 bit。
**代价**：**部署噩梦**。稀疏矩阵、混合精度、双路径调度，每一项都是 kernel 的敌人。这是"论文精度"与"工程速度"落差最大的一维——很多方法精度惊艳却从未能进入生产。

---

### F8 · 模型适应：允许改 $W$ 吗

前面七个自由度都在"给定 $W$，找最好的 $\hat{W}$"。第八个自由度问的是一个不同的问题：**能不能改 $W$ 本身，让它更容易被量化？**

- **不动**（纯 PTQ）：F1–F7 的全部领域。
- **局部重构**（BRECQ / OmniQuant）：只优化变换参数或舍入变量，不动主权重。
- **全局微调**（QAT / LSQ / QLoRA）：把量化搬进计算图，让权重与量化参数共同适应（[17](/2026/08/26/llm-quant-11-fake-quant-insertion/)、[18](/2026/08/29/llm-quant-18-lsq-pact-dsq/)、[21 篇](/2026/08/25/qat-00-overview/)）。

**这是唯一真正改变被量化函数的自由度**，也是威力最大、代价最高的一维。

---

## 4. 全景表：所有算法，一张坐标图

把前面 22 篇的算法放回这张表。**"●" = 主要发力点，"○" = 附带涉及**：

| 算法 | F1 网格 | F2 scale | F3 粒度 | F4 落点 | F5 变换 | F6 补偿 | F7 混合 | F8 适应 | 系列 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| RTN | | | | ● | | | | | [01](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) |
| LLM.int8() | | | ○ | | | | ● | | [02](/2026/08/24/llm-quant-01-llmint8-outlier-mixture/) |
| GPTQ | | | ● | ● | | ○ | | | [03](/2026/08/24/ptq-02-gptq/) |
| AWQ | | ● | ● | | ● | | | | [04](/2026/08/24/llm-quant-03-awq-scale-search/) |
| OmniQuant | | ● | | | ● | | | ○ | [05](/2026/08/24/ptq-03-awq-omniq/) |
| SpQR / OWQ / HQQ | | ○ | | ● | | | ● | | [06](/2026/08/24/ptq-04-spqr-owq-hqq/) |
| QuIP# / AQLM | ● | | | ● | | | | | [07](/2026/08/24/ptq-05-quip-aqlm/) |
| SqueezeLLM / VPTQ | ● | | | | | | ● | | [08](/2026/08/24/ptq-09-squeezellm-vptq-claq/) |
| SmoothQuant | | ● | | | ● | | | | [09](/2026/08/24/llm-quant-02-smoothquant-w8a8/) |
| Outlier Suppression+ | | | | | ● | | | | [11](/2026/08/24/ptq-10-outlier-suppression/) |
| QuaRot / SpinQuant | | | | | ● | | | | [12](/2026/08/24/ptq-07-quarot-spinquant/) |
| RPTQ / QUIK / ATOM | | ● | ● | | | | ● | | [13](/2026/08/24/ptq-11-rptq-quik-atom/) |
| OliVe | ● | | | | | | ● | | [14](/2026/08/24/ptq-12-olive-abfloat/) |
| QServe (QoQ) | | ● | | | ● | ● | | | [16](/2026/08/24/ptq-13-qserve-qqq/) |
| QAT / LSQ / QLoRA | | ● | | | | | | ● | [17](/2026/08/26/llm-quant-11-fake-quant-insertion/)–[21](/2026/08/25/qat-00-overview/) |

看这张表会发现两件有意思的事：

**第一，没有哪个算法在超过三个自由度上发力。** 这是工程现实的约束——每个自由度都带来实现复杂度与部署风险。

**第二，F5（等效变换）被用得最多，而它数学上是免费的。** 这解释了为什么 2023 年之后的研究大量涌向这个方向：它是唯一一个"收益显著而直接代价最小"的自由度。

---

## 5. 三个元策略：改答案 / 改问题 / 改模型

八个自由度可以进一步收敛成**三个元策略**。这比记八个格子更好用：

```text
                 min tr(ΔW · C · ΔWᵀ)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ① 改答案            ② 改问题            ③ 改模型
   在给定网格上        用恒等变换把       让 W 本身
   找更好的 Ŵ          W,x 变好量化       适应网格
        │                 │                 │
      RTN            SmoothQuant           QAT
      GPTQ               AWQ               LSQ
    AdaRound          QuaRot            OmniQuant
    QuIP#/AQLM          OS+               QLoRA
        │                 │                 │
   需校准集/算 H      数学免费          需训练算力
   局部最优           工程要融合        收益上限最高
```

### ① 改答案（F1, F2, F3, F4, F6, F7）

接受"这个问题很难"，然后在离散约束下尽力求解。

**收益曲线**：RTN → GPTQ → AdaRound → VQ，每一步都在逼近"给定网格下的信息论极限"。到 2-bit，这条路基本走到头了，QuIP#/AQLM 换向量量化是最后的一次跃迁。

**判断标准**：当你的瓶颈是"网格太粗"，改答案有用；当瓶颈是"分布太糟"（outlier 主导），改答案收益有限。

### ② 改问题（F5）

不改答案，改题目的形式。用恒等变换把 outlier 摊薄或迁移走。

**为什么它这么受欢迎**：数学上零代价，收益直接。SmoothQuant 把 W8A8 从"不可用"变成"可用"，QuaRot 让 W4A4 第一次在 7B 级别逼近 FP16。

**它的边界**：变换必须能融合进权重，且必须在非线性/残差处断开重估。能变换的空间受网络拓扑严格约束——这就是为什么旋转类方法要花大量篇幅论证"哪些地方能塞进 Hadamard 变换"。

### ③ 改模型（F8）

让参数自己走到好量化的位置上去。

**它的主场**：2–4 bit 的极低比特、以及需要保住复杂能力（推理链、长上下文）的场景。在 4-bit 以上，现代 PTQ（GPTQ/AWQ + 旋转）往往已经能打平轻量 QAT——[21 篇](/2026/08/25/qat-00-overview/)已经指出过这一点。

**关键洞察**：②③ 不是互斥的。近年最强的方案基本是**三层叠加**——QServe 的 W4A8KV4 就是典型：SmoothAttention（②）+ 量化误差补偿（①的 F6）+ 精心挑选的粒度与 scale（①的 F2/F3）。

---

## 6. 统一视角能预测什么：四条可检验的判断

一个框架的价值，在于它能不能产生**可证伪的推断**。这一节给出四条，都是从上面的分析直接推出来的。

**判断一：叠加同类自由度收益递减，叠加异类自由度收益叠加。**
在同一个层上同时做 GPTQ + AdaRound（都在 F4），收益会明显小于各自单独做的收益之和——它们在争夺同一个自由度。而 GPTQ（F4）+ SmoothQuant（F5）+ AWQ（F3/F5）叠加，收益基本可加。**这也是为什么工业方案总是"挑每类里最好的一个"而不是"挑精度最高的三个"。**

**判断二：bit 数下降时，发力点必须沿 F1 → F7 → F8 的方向迁移。**
4-bit 时优化 F2/F3/F4 收益巨大；3-bit 时 F7（保 outlier）开始成为必需；2-bit 时 F1（向量量化）或 F8（QAT）几乎是唯一出路。**如果你的方法在 2-bit 还在优化 F2 的 scale，它几乎注定失败。**

**判断三：任何声称"通用最优"的单点方法都值得怀疑。**
因为设计空间是八维的，而场景约束（硬件支持哪些格式、有没有校准集、能不能训练、延迟预算 vs 显存预算）决定了哪几个自由度可用。**不存在脱离约束的最优解**——只存在"在你的约束下，哪几个自由度最值得动"。

**判断四：这个方法会部署得顺利吗？看它动了哪一维。**
主要动 F5（恒等变换）的，部署相对顺；动 F1（非均匀网格）的，要查表；动 F7（混合精度）的，要稀疏 kernel 或多路径调度——**这是精度漂亮但落地困难的方法最集中的一维**。

---

## 7. 用这个框架五分钟定位一篇新论文

拿到一篇新的量化论文，按这个顺序问：

1. **它改的是被量化的函数吗？**（F8）——如果不是，它一定在某个环节存在精度上限。
2. **它的变换是恒等的吗？**（F5）——如果是，问"能不能融合进权重"，这决定它有没有工程价值。
3. **它需要校准集吗？**——需要的话，问"校准集分布漂移会怎样"，这决定鲁棒性。
4. **它的输出是均匀整数网格吗？**——如果不是（F1/F7），问"目标硬件原生支持这个格式吗"，这决定它能不能真的跑快。
5. **它在哪个 bit 数上评估？**——只在 4-bit 上报告结果的 2-bit 方法，多半在 2-bit 上不行。

这五个问题问完，你基本知道这篇论文在你的场景里值不值得投入。

---

## 8. 批判与展望

**本篇解决了什么**：把 22 篇里出现的二十多个算法，从一个扁平的清单，还原成"一个优化目标 × 八个自由度 × 三个元策略"的立体结构。给出了它们之间的**可叠加关系**（而非互斥关系），以及四条可用于判断新方法的可证伪推断。

**致命局限**：有三个东西是这套框架**覆盖不到**的，必须诚实说明。

1. **它只讲单层**。真实网络是层的复合，误差会跨层传播、被 LayerNorm 重归一化、被 softmax 非线性放大。所有 layer-wise PTQ 都在这个假设下工作，而"层最优 ≠ 网络最优"是它们的系统性盲区。
2. **它只讲权重与激活，没讲 KV Cache**。在长上下文、大 batch 的服务场景里，KV cache 才是显存与带宽的主角——那是 [24 篇](/2026/08/29/llm-quant-24-kv-cache/) 的主题。
3. **它假设你能自由选 bit 数**。真实部署里，bit 数往往是被硬件格式与 kernel 支持反过来决定的（"W4 不一定比 W8 快"是 [16 篇](/2026/08/24/ptq-13-qserve-qqq/) 反复强调的事）。真正的选型要反过来做：**先看硬件支持什么，再在允许的自由度里挑**——也就是 [25 篇](/2026/08/29/llm-quant-25-mixed-precision/) 要讲的硬件感知 bit 分配。

**Takeaway 三件套**

> **解决什么痛点**：面对二十多个量化算法名字，只能死记、无法推导、无法判断新方法。
>
> **致命局限**：框架建立在 layer-wise 二次近似之上，跨层误差累积与 KV cache 不在其射程内。
>
> **如何用起来**：拿到任何量化方案，先问它动的是 F1–F8 里的哪几个自由度，再看它属于"改答案/改问题/改模型"哪一类。同类的不要叠加，异类的尽量叠加。

---

## 参考清单

**论文（ID 已逐一核验）**

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) —— F4 的代表，"误差可主动抵消"
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) —— F5 + F3 依赖关系的实例
- Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) —— F5 难度迁移
- Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) —— F5 正交旋转
- Liu et al., *SpinQuant: LLM Quantization with Learned Rotations*, [arXiv:2405.16465](https://arxiv.org/abs/2405.16465) —— 可学习旋转
- Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) —— F7 的起点
- Kim et al., *SqueezeLLM: Dense-and-Sparse Quantization*, [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) —— F7 + F1
- Dettmers et al., *SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression*, [arXiv:2306.03078](https://arxiv.org/abs/2306.03078) —— F7 元素级
- Tseng et al., *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*, [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) —— F1 向量量化
- Egiazarian et al., *Extreme Compression of Large Language Models via Additive Quantization*, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) —— AQLM，加性码本
- Nagel et al., *Up or Down? Adaptive Rounding for Post-Training Quantization*, [arXiv:2004.10568](https://arxiv.org/abs/2004.10568) —— F4 优化派起点
- Wei et al., *Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling*, [arXiv:2304.09145](https://arxiv.org/abs/2304.09145) —— F5 的 $\gamma$ 迁移
- Nagel et al., *A White Paper on Neural Network Quantization*, [arXiv:2106.08295](https://arxiv.org/abs/2106.08295) —— 本文设计空间的规范参考

**代码与规范**

- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) —— GPTQ 参考实现
- [llm-awq](https://github.com/mit-han-lab/llm-awq) —— AWQ 官方实现
- [Qserve](https://github.com/mit-han-lab/qserve) —— W4A8KV4 服务端
- [llama.cpp](https://github.com/ggerganov/llama.cpp) —— k-quants 非均匀网格的工程标杆

**系列导航**

- 系列规划：见站内 [模型量化课程路线图](/quantization-roadmap/)（全 26 篇目录与阅读路径）

> **Lab 练习（动手）**
> 1. 挑一篇你最近看到的量化论文，用 §7 的五个问题过一遍，看能不能在五分钟内定位它的自由度坐标。
> 2. 回到 [01 篇](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) 的配套实验，把 bit 数从 4 降到 2，观察"优化 scale"（F2）的收益如何塌缩——验证判断二。
> 3. 在同一个层上先做 SmoothQuant 再做 GPTQ，与只做 GPTQ 对比；再试 GPTQ + AdaRound，观察同类叠加的递减效应——验证判断一。
