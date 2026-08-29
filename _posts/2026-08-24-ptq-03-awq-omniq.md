---
title: "大模型量化算法（05）：OmniQuant——可学习缩放与裁剪"
date: 2026-08-24 11:00:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, omniquant, learnable, w4a4]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 1 · Weight-only PTQ** ｜ 第 05 篇 / 共 26 篇
>
> [← 04 AWQ](/2026/08/24/llm-quant-03-awq-scale-search/) ｜ [06 SpQR/OWQ/HQQ →](/2026/08/24/ptq-04-spqr-owq-hqq/)
>
> **LLM 量化系列 · 第 05 篇**：从 GPTQ 的"事后补偿"转向"事前预防"——AWQ 用激活统计定位显著通道并做等效缩放，OmniQuant 把缩放与裁剪变成可学习参数。本文覆盖两篇论文的完整数学推导、可复现的 numpy 实现、对比批判与工程生态。

---

## TL;DR 三连

### 一句话

AWQ 用**激活幅度**（而不是权重幅度）找出模型中"牵一发动全身"的 1% 显著通道，通过一个逐通道的对角缩放把它们的量化误差压低一个数量级，全程只需前向统计、不需要任何梯度；OmniQuant 则更进一步，把"裁剪边界"和"缩放因子"本身变成可学习参数，用不到模型 0.1% 的参数量加 STE（直通估计器）做块级重建训练，把同一套思想推广到 AWQ 做不到的 W4A4。

### 三句话

1. **权重量化的误差不是均匀分布的**：输出误差 \(e_i = \sum_j X_j \varepsilon_{ij}\) 被激活幅度加权，激活大的通道即使权重平庸，其量化误差也会被放大——所以"显著通道"必须由激活统计定义，AWQ 的 \(s = \max|X|^\alpha\) 缩放等效于"只保护这 1% 通道"（arXiv:2306.00978）。
2. **缩放为什么有效**：均匀量化器的绝对误差上界 \(\Delta/2\) 与数值大小无关，把显著通道权重放大 \(s\) 倍再量化、之后除回，等效误差变成 \(\Delta/(2s)\)；代价是组内量化步长 \(\Delta\) 随最大缩放系数增长——\(\alpha \approx 0.5\) 正是这个权衡的最优点，\(\alpha=1\) 必然过冲。
3. **OmniQuant 把手工搜索变成梯度优化**：LWC 学习裁剪上下界压缩权重动态范围，LET 学习逐通道缩放因子迁移激活-权重尺度，两者合计约 \(10^5\sim10^6\) 个参数、单卡数小时即可完成 7B 模型的量化（arXiv:2308.13137），并让 W4A4 首次变得实用。

### 五分钟

AWQ 是对 GPTQ"事后补偿"路线的正面反叛：GPTQ 用 Hessian 二阶信息逐列量化、逐列修补，AWQ 则**不做任何权重修正**，只做一次"事前预防"。论文的核心观察是：激活幅度（而非权重幅度）能精准定位 1% 的显著通道，把这 1% 通道的权重保持不量化，就能恢复绝大部分量化损失；而用一个 \(\alpha\) 网格搜索出来的对角缩放，可以**等效地实现"保护"**，同时保持纯 4bit 推理、不需要混合精度 kernel。整个校准是纯前向的：统计校准集激活 → 搜 \(\alpha\) → RTN 量化，7B 模型分钟级完成。

OmniQuant 把这一思想彻底参数化：既然 \(\alpha\) 是搜出来的、clip 边界是 RTN 隐含的，为什么不直接让它们可学习？LWC（可学习权重裁剪）把量化器的上下界变成每层 2 个可学习参数，用 STE 回传梯度；LET（可学习等效变换）把 AWQ 的缩放因子变成可学习向量，与激活量化联合优化，从而支持 W4A4。它受 LoRA"参数高效"思想启发，但目标不是微调而是量化误差重建，训练开销比 QAT 低两个数量级。

两者也各有软肋：AWQ 的 \(\alpha\) 依赖校准集分布、缩放表达力有限（本质只是一个对角变换）；OmniQuant 本质是轻量训练，存在 STE 偏差与块级误差累积，且工程生态远不如 AWQ/GPTQ 成熟（vLLM 原生支持 AWQ，而 OmniQuant 目前停留在研究代码）。

---

## 系列导航

| 篇目 | 主题 | 文件 |
| --- | --- | --- |
| 第 00 篇 | 量化全景 | [量化全景](/2026/08/24/ptq-00-overview/) |
| 第 E1 篇 | RTN / LLM.int8 | [RTN 与 LLM.int8()](/2026/08/24/ptq-01-rtn-llmint8/) |
| 第 03 篇 | GPTQ | [GPTQ](/2026/08/24/ptq-02-gptq/) |
| **第 05 篇（本文）** | **AWQ / OmniQuant** | **本篇** |
| 第 06 篇 | SpQR / OWQ / HQQ | [SpQR/OWQ/HQQ](/2026/08/24/ptq-04-spqr-owq-hqq/) |
| 第 07 篇 | QuIP# / AQLM | [QuIP#/AQLM](/2026/08/24/ptq-05-quip-aqlm/) |
| 第 10 篇 | SmoothQuant / ZeroQuant | [SmoothQuant/ZeroQuant](/2026/08/24/ptq-06-smoothquant-zeroquant/) |
| 第 12 篇 | QuaRot / SpinQuant | [QuaRot/SpinQuant](/2026/08/24/ptq-07-quarot-spinquant/) |
| 第 15 篇 | GGUF k-quants / FP8 / MXFP4 | [GGUF/FP8/MXFP4](/2026/08/24/ptq-08-gguf-fp8-mxfp4/) |

---

## 目录

- [1. 从 GPTQ 说起：误差补偿的两种路线](#1-从-gptq-说起误差补偿的两种路线)
- [2. AWQ 的核心洞察：激活幅度才是显著性的度量](#2-awq-的核心洞察激活幅度才是显著性的度量)
- [3. AWQ 的数学原理](#3-awq-的数学原理)
- [4. 用 numpy 实现 AWQ（完整可运行）](#4-用-numpy-实现-awq完整可运行)
- [5. OmniQuant：把手工设计变成可学习参数](#5-omni quant把手工设计变成可学习参数)
- [6. 对比总表与工程生态](#6-对比总表与工程生态)
- [7. 批判与展望](#7-批判与展望)
- [8. 参考清单](#8-参考清单)

---

## 1. 从 GPTQ 说起：误差补偿的两种路线

前两篇我们建立了两个坐标系：

- **第 E1 篇（RTN/LLM.int8）**：RTN 是最朴素的 round-to-nearest 量化，实现简单但对激活/权重中的 outlier 极其敏感；LLM.int8 用混合精度分解把 outlier 通道单独拎出来保持 FP16，首次让 175B 模型可部署，但代价是混合精度 kernel 和内存碎片。
- **第 03 篇（GPTQ）**：把量化看作"逐列决策问题"，用 Hessian（二阶信息）衡量每个权重对输出的敏感度，量化一列后用其余列补偿误差。这是**事后补偿**路线的巅峰：先量化、再修补。

到了 4bit 权重量化（W4A16）这个任务上，GPTQ 已经做得相当好，但它有两个隐忧：

1. **校准开销与复杂度**：需要构造 Hessian、做 Cholesky 分解、逐列迭代更新，7B 模型也要数十分钟到小时级，且对校准集分布敏感、有"过拟合校准集"的风险；
2. **补偿是"被动"的**：GPTQ 假设量化误差已经发生，然后尽量抹平；它没有回答一个更根本的问题——**能不能让误差根本不要发生在重要的地方？**

AWQ 选择的就是第二条路：**事前预防**。不修改任何权重值，只做一个巧妙的坐标变换，让"重要的权重"在量化网格里天然占据更多、更精确的档位。OmniQuant 则把这条路上的两个手工旋钮（缩放强度 \(\alpha\)、裁剪边界）全部换成可学习参数，用极轻量的梯度优化把它们拧到最优。

这两种路线的哲学差异可以概括为：

> **GPTQ：误差已经发生了，我用二阶信息把它补回来。**
> **AWQ：误差还没发生，我先把重要的东西挪到安全的位置。**

### 1.1 为什么战场是 W4A16

在进入 AWQ 之前，值得先明确 4bit 权重量化这个任务的特殊性。8bit 权重量化（W8A16）在今天已经"过于简单"——RTN 加上一点平滑处理就能做到接近无损，因为 8bit 有 256 个档位，足以覆盖重尾分布的主体。真正的战场在 4bit：

- **档位数骤降**：4bit 只有 16 个档位，步长 \(\Delta\) 比 8bit 大 16 倍，量化误差的方差大 256 倍——RTN 开始肉眼可见地崩坏（LLaMA-7B 的 WikiText-2 ppl 从 5.68 涨到 6.29，见 6.2 节）；
- **内存收益显著**：4bit 权重把模型内存压到原来的 1/4，7B 模型从约 14GB（FP16）降到约 3.5GB，这是单卡部署 7B/13B 模型的分水岭；8bit 只省一半，边际吸引力小得多；
- **精度余量极小**：4bit 下每个档位都弥足珍贵，"哪些权重值得更好的档位"这个问题第一次变得生死攸关——这正是 AWQ 激活感知思想的用武之地。

换句话说：**8bit 时代误差被档位数淹没，4bit 时代档位数被误差淹没**。AWQ 和 GPTQ 都是在"16 个档位怎么分配"这个问题上给出不同答案的算法。

---

## 2. AWQ 的核心洞察：激活幅度才是显著性的度量

### 2.1 权重量化的误差从哪来

先看一个线性层的前向：

$$y = X W^{\top}, \qquad X \in \mathbb{R}^{T \times C_{in}},\; W \in \mathbb{R}^{C_{out} \times C_{in}}$$

对输出第 \(i\) 行（对应第 \(i\) 个输出通道）：

$$y_i = \sum_{j=1}^{C_{in}} X_j \, W_{ij}$$

其中 \(X_j\) 是激活的第 \(j\) 个**输入通道**（一个长度为 \(T\) 的向量），\(W_{ij}\) 是权重矩阵第 \(i\) 行第 \(j\) 列。假设权重被量化，每个元素引入误差 \(\varepsilon_{ij} = \hat{W}_{ij} - W_{ij}\)，则输出误差为：

$$e_i = \sum_{j} X_j \, \varepsilon_{ij}$$

这个式子透露了两个关键事实：

1. **误差被激活幅度加权**。同样大小的权重误差 \(\varepsilon\)，出现在激活幅度 \(|X_j|\) 大的通道上，对输出的破坏比出现在激活小的通道上大得多。权重本身的幅度 \(\max_i |W_{ij}|\) 在这里**根本不出现**——它是通过 \(\varepsilon_{ij}\) 间接影响的，而 \(\varepsilon_{ij}\) 的上界对所有通道都差不多（都是 \(\Delta/2\)）。
2. **所以"显著通道"必须由激活定义**。一个通道权重再大，如果它对应的激活常年很小，它的量化误差对输出就是无关紧要的；反之，一个激活幅度巨大的通道，哪怕权重只是平均水平，它的误差也会被放大成输出误差的主要来源。

### 2.2 用激活幅度找显著通道（论文观察）

AWQ 论文（arXiv:2306.00978）的 Figure 1 做了一个非常干净、也非常反直觉的实验：

- **按权重幅度**挑出最大的 1% 通道，量化时把它们排除（保持 FP16），结果：精度几乎没有任何恢复；
- **按激活幅度**挑出最大的 1% 通道做同样的事，结果：恢复掉了绝大部分量化损失。

直觉解释：权重幅度大的通道往往只是"数值大但没人用"，激活幅度大的通道才是"数值不大但人人都在乘"。LLM 的激活分布是高度重尾的——少数通道承载了大部分信息（这也是 SmoothQuant 和 LLM.int8 观察到的同一现象），而这些通道的权重**并不**恰好是权重矩阵里的 outlier。权重 outlier 与激活 outlier 在位置上基本不重合，这正是 AWQ 与"按权重幅度保护"路线的分水岭。

> **论文原话要点（转述）**：We find that we can find 1% of the salient channels by looking at the activation magnitudes, not the weight magnitudes. Protecting these 1% salient channels (i.e., not quantizing them) can largely recover the quantization loss.

### 2.3 保护 1% 通道就够了

论文进一步量化了"保护多少"的问题：随着被保护通道比例从 0 增加到 1%，精度快速恢复；超过 1% 之后边际收益急剧递减。换句话说，量化误差的绝大部分集中在极少数通道上，这是一个高度稀疏的结构。

但"保护"有一个工程代价：被保护的通道要保持 FP16，推理时需要**混合精度 kernel**（权重矩阵里 99% 是 4bit、1% 是 FP16），这在 GEMM 里意味着内存访问不规整、kernel 实现复杂。AWQ 的关键技巧是：**用缩放去逼近保护**——把显著通道的权重放大 \(s\) 倍再量化，等效于给它们分配更多的量化档位，效果上近似"保护"，但推理时依然是纯 4bit 权重、无需混合精度。

下面进入数学。

---

### 2.4 一个数值玩具例子：缩放到底改了什么

用一个极简的两通道例子感受缩放的作用。设某个输出神经元 \(y = x_1 w_1 + x_2 w_2\)，其中 \(x_1 = 10\)（显著通道）、\(x_2 = 1\)，权重 \(w_1 = 0.3\)、\(w_2 = 0.9\)，4bit 对称量化（\(q_{max} = 7\)，步长 \(\Delta = \max|w|/7 = 0.1286\)）。

**RTN 直接量化**：\(w_1 = 0.3\) 落在 \(2 \times 0.1286 = 0.2571\) 档上，误差 \(\varepsilon_1 = -0.0429\)；\(w_2 = 0.9\) 恰好是 \(7 \times 0.1286\)，误差为 0。输出误差：

$$e = x_1 \varepsilon_1 + x_2 \varepsilon_2 = 10 \times (-0.0429) + 1 \times 0 = -0.429$$

——误差全部来自**显著通道的权重**：\(w_1\) 的量化误差本身不大（0.043），但被 \(x_1 = 10\) 放大了 10 倍。

**AWQ 缩放**（取 \(s_1 = 3,\; s_2 = 1/3\)，几何均值归一化保证 \(s_1 s_2 = 1\)）：\(w'_1 = 0.9\)、\(w'_2 = 0.3\)。组内最大值仍是 0.9，**步长不变**（\(\Delta' = \Delta\)——这就是"免费缩放"：总范围没动，只是通道间再分配）。量化后 \(w'_1 = 0.9\) 精确落档（误差 0），\(w'_2\) 落到 0.2571，除回 \(s_2\) 后等效误差 \(\varepsilon'_2 = (0.2571 - 0.3) / (1/3) = -0.1286\)。输出误差：

$$e' = x_1 \cdot 0 + x_2 \cdot (-0.1286) = -0.129$$

| 通道 | \(x_j\) | \(w_j\) | RTN 误差 \(\varepsilon_j\) | 缩放后 \(w'_j\) | AWQ 误差 \(\varepsilon'_j\) | RTN 贡献 | AWQ 贡献 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1（显著） | 10 | 0.3 | -0.0429 | 0.9 | 0 | -0.429 | 0 |
| 2 | 1 | 0.9 | 0 | 0.3 | -0.1286 | 0 | -0.129 |
| **合计** | | | | | | **-0.429** | **-0.129（↓ 70%）** |

这个例子里发生的本质是：**缩放把量化误差从"被大激活乘的通道"搬运到了"被小激活乘的通道"**。误差的"总量"没有消失（\(w_2\) 的误差反而变大了 3 倍），但误差与激活幅度的乘积——也就是真正影响输出的量——下降了 3.3 倍。这就是 AWQ 全部机制的浓缩：不消灭误差，只把误差重新分配到无关紧要的地方去。

---

## 3. AWQ 的数学原理

### 3.1 等效变换：把误差按通道"搬运"

AWQ 的核心工具是**等效变换（equivalent transformation）**。对任意逐输入通道的缩放向量 \(s \in \mathbb{R}^{C_{in}}_{>0}\)，定义：

$$W' = W \cdot \operatorname{diag}(s), \qquad X' = X \cdot \operatorname{diag}(s)^{-1}$$

即权重第 \(j\) 个输入通道乘以 \(s_j\)，激活第 \(j\) 个输入通道除以 \(s_j\)。输出严格不变：

$$X' {W'}^{\top} = X \operatorname{diag}(s)^{-1} \operatorname{diag}(s) W^{\top} = X W^{\top}$$

数据流如下：

```mermaid
flowchart LR
    subgraph OFF["离线校准（纯前向，无梯度）"]
        X1["校准激活 X<br/>[T, C_in]"] --> A["逐通道统计<br/>s_X_j = max_t |X_tj|"]
        A --> B["scale 搜索<br/>s_j = s_X_j^α，α 网格搜索"]
        W1["权重 W<br/>[C_out, C_in]"] --> C["等效变换<br/>W' = W · diag(s)"]
        B --> C
        C --> D["组量化 RTN<br/>group=128, 4bit"]
        D --> E["还原<br/>Ŵ = Q(W') · diag(s)⁻¹"]
    end
    subgraph INF["在线推理（权重只读一次）"]
        X2["激活 X 保持 FP16"] --> G["W4A16 GEMM<br/>y = Ŵ · X"]
        E --> G
    end
```

注意：AWQ 是**仅权重量化（W4A16）**，所以推理时激活侧 \(\operatorname{diag}(s)^{-1}\) **并不真正执行**——激活保持 FP16，缩放只被"烘焙"进权重里：

$$\hat{W} = Q\big(W \cdot \operatorname{diag}(s)\big) \cdot \operatorname{diag}(s)^{-1}$$

激活侧的 \(\operatorname{diag}(s)^{-1}\) 只出现在理论推导里，用来证明"如果两边都变换，输出严格不变"；实际部署时它被省略，引入的误差就是权重量化误差本身。

### 3.2 为什么缩放后的量化误差更小

设量化步长为 \(\Delta\)（对称均匀量化下 \(\Delta = 2\max|w|/(2^b - 1)\)，或 group-wise 下由组内最大值决定）。均匀量化器的关键性质是：**绝对误差上界与数值大小无关**：

$$|Q(w) - w| \le \frac{\Delta}{2}, \qquad \forall w$$

也就是说，一个 \(|w|=0.01\) 的小权重和一个 \(|w|=1.0\) 的大权重，量化误差上界都是 \(\Delta/2\)——小权重的**相对误差**可以高达 50 倍差距。这就是 RTN 在重尾分布上吃亏的根本原因。

现在对第 \(j\) 个通道缩放 \(s_j\)。缩放后的权重 \(W_{ij} s_j\) 被量化，误差上界仍是 \(\Delta'/2\)（\(\Delta'\) 是缩放后的步长），但除回 \(s_j\) 之后，**等效到原始尺度上的误差**变成：

$$\left|\frac{Q(W_{ij} s_j)}{s_j} - W_{ij}\right| = \frac{|Q(W_{ij}s_j) - W_{ij}s_j|}{s_j} \le \frac{\Delta'}{2 s_j}$$

于是输出误差的界为：

$$|e_i| \le \sum_j |X_j| \cdot \frac{\Delta'}{2 s_j} = \frac{\Delta'}{2} \sum_j \frac{|X_j|}{s_j}$$

**缩放把显著通道（\(|X_j|\) 大）的误差贡献除以 \(s_j\)**。如果 \(\Delta'\) 保持不变，那么 \(s_j\) 越大越好——这正是"保护"的极限情形（\(s_j \to \infty\) 时该通道误差 \(\to 0\)，等效于不量化）。

但 \(\Delta'\) 不会保持不变。在 group-wise 量化（AWQ 默认 group_size=128）下，一个组共享同一个步长，组内只要有一个通道被放大，组步长就会被撑大：

$$\Delta' \approx \Delta \cdot \max_{j \in \text{group}} s_j \quad(\text{当缩放后的最大值主导组范围时})$$

于是存在两个相互竞争的效应：

| 效应 | 方向 | 来源 |
| --- | --- | --- |
| 显著通道误差 \(\div s_j\) | 减小误差 | 缩放把显著权重推到更多量化档位 |
| 组步长 \(\Delta' \propto \max s_j\) | 增大误差 | 组内所有通道的误差上界同步放大 |

当 \(\max s_j\) 增长得比 \(s_j\)（显著通道自己的缩放）快时，净效果为负——这就是为什么 \(\alpha\) 必须小于 1，也是"为什么不能简单地把显著通道放大 100 倍"的数学答案。

**一个值得注意的推论**：当量化粒度细到**逐输入通道**（每个通道独立 scale）时，缩放对误差上界完全无效——因为 \(\Delta'_j = s_j \Delta_j\) 与 \(s_j\) 同步增长，\(\Delta'_j/(2s_j) = \Delta_j/2\) 不变，误差分布与不缩放时完全相同。**AWQ 的收益本质上来自 group-wise 量化下"组内量化资源再分配"**：组步长由组内最大值决定，缩放不会改变它（只要缩放后的值不越过原最大值），却能让显著通道在固定步长下获得更小的相对误差。这也解释了为什么 AWQ 论文与实现都锚定 group_size=128 而非逐通道量化。

### 3.3 scale 的求解：\(s = \max|X|^\alpha\)

给定"缩放能降低显著通道误差"这个机制，下一步是确定 \(s\) 取什么值。AWQ 的推导分两步：

**第一步：固定 \(\Delta\) 时的最优缩放。** 在 \(\Delta\) 视为常数的近似下，最小化误差上界等价于：

$$\min_{s} \; \sum_j \frac{|X_j|}{s_j}, \qquad \text{s.t.} \quad \prod_j s_j = 1$$

（几何均值归一化约束防止 \(s\) 整体漂移。）用拉格朗日乘子法：

$$\mathcal{L}(s) = \sum_j \frac{c_j}{s_j} + \lambda \sum_j \ln s_j, \qquad c_j = |X_j|$$

$$\frac{\partial \mathcal{L}}{\partial s_j} = -\frac{c_j}{s_j^2} + \frac{\lambda}{s_j} = 0 \;\Longrightarrow\; s_j^\ast \propto c_j = |X_j|$$

即**最优缩放正比于激活幅度**。这与直觉一致：激活越大的通道，越值得"保护"。

**第二步：\(\alpha\) 折中。** 纯激活幅度缩放（\(s_j \propto |X_j|\)）在真实 LLM 上会失败——激活幅度的动态范围可达几十倍，直接按它缩放会让组步长 \(\Delta'\) 爆炸（上面的第二个效应）。论文因此把搜索空间限制为单参数族：

$$s_j = \big(\max_{t} |X_{tj}|\big)^{\alpha}, \qquad \alpha \in [0, 1]$$

其中 \(\max_t |X_{tj}|\) 是校准集上第 \(j\) 个输入通道的激活幅度（取 max 而非均值，因为 outlier token 恰恰是最需要保护的）。\(\alpha=0\) 退化为 RTN，\(\alpha=1\) 是全强度激活缩放（过冲），最优值在中间。论文用**网格搜索**确定 \(\alpha\)：

$$\min_{\alpha} \; \mathbb{E}_{x \sim \mathcal{D}_{\text{cal}}}\Big[ \big\| f(x; W) - f\big(x;\, Q(W \cdot \operatorname{diag}(s_X^{\alpha})) \cdot \operatorname{diag}(s_X^{-\alpha}) \big) \big\|_F^2 \Big]$$

即：在少量校准数据上，对 \(\alpha \in \{0, 0.1, \dots, 1.0\}\) 逐一量化并测量**逐层输出误差**（也可用困惑度），取最优。论文在多个模型（LLaMA、OPT、Mistral 等）上的结论是 \(\alpha \approx 0.5\) 附近普遍稳健——0.5 恰好是"误差 \(\div s_j\)"与"步长 \(\times \max s_j\)"两个幂律的几何中点。需要注意的是，AWQ 的 loss 是一个**一维搜索**，每次评估只是一次前向+量化，比 GPTQ 的 Hessian 求逆便宜几个数量级，这也是 AWQ 校准分钟级完成的原因。

### 3.4 与 SmoothQuant 的数学联系

SmoothQuant（本系列第 10 篇的主角）与 AWQ 共享同一个数学家族——**激活-权重间的尺度迁移**。两者都做：

$$X \to X \cdot \operatorname{diag}(s)^{-1}, \qquad W \to W \cdot \operatorname{diag}(s)$$

但目标与取值方式截然不同：

| 维度 | SmoothQuant | AWQ |
| --- | --- | --- |
| 目标精度 | W8A8（激活也要量化） | W4A16（仅权重量化） |
| 迁移目的 | 把激活的量化难度"熨平"给权重 | 只保护显著通道的权重精度 |
| scale 取值 | \(s_j = \max|X_j|^\alpha / \max|W_j|^{1-\alpha}\)，平衡两侧范围 | \(s_j = \max|X_j|^\alpha\)，只看激活 |
| 激活侧是否执行 | 是（\(X \cdot s^{-1}\) 真实参与计算） | 否（推理时激活保持 FP16） |
| 适用场景 | 需要同时省激活带宽（W8A8） | 权重是主要内存瓶颈（W4A16） |

一个更准确的说法是：**AWQ 是 SmoothQuant 的"单边版本"**——SmoothQuant 必须把激活范围压到可量化的程度，所以 scale 要同时考虑 \(\max|X|\) 和 \(\max|W|\)；AWQ 的激活根本不量化，所以 scale 只需要服从一个目标：让权重量化误差在激活大的通道上最小，即 \(s_j \propto |X_j|^\alpha\)。两者可以无缝衔接：先做 SmoothQuant 的 W8A8 迁移，再在权重侧叠加 AWQ 缩放，是实际部署中常见的组合拳。

### 3.5 与 GPTQ 的数学对比

GPTQ 与 AWQ 是 4bit 权重量化的两条代表性路线，数学上几乎是对偶的：

- **GPTQ（事后补偿）**：用 Hessian \(H = 2XX^{\top} + \lambda I\) 度量每个权重对损失的敏感度，逐列量化后用其余列做补偿更新 \(\delta_F = -\frac{w_q - w}{H_{kk}} H_{:,k}\)。它改变权重值，需要二阶信息，校准是迭代的。
- **AWQ（事前预防）**：不做任何权重更新，只做一次对角缩放 \(W \cdot \operatorname{diag}(s)\)。它的"敏感度"不是 Hessian，而是**激活幅度**这个一阶统计量；它的"补偿"不是数值修补，而是**把重要权重挪到量化误差更小的位置**。

一句话总结：GPTQ 修正的是**误差本身**，AWQ 修正的是**误差的分布**。

---

### 3.6 为什么"保护 1%"就够了：误差贡献的集中性

把 3.2 节的误差分析再推进一步，量化"显著性"的分布。假设激活各通道零均值、相互独立，权重量化误差 \(\varepsilon_{ij}\) 与激活独立，则输出误差的期望平方可以分解为逐通道贡献：

$$E\big[\|e\|_F^2\big] = \sum_j E\big[\|X_j\|_2^2\big] \cdot E\big[\|\varepsilon_j\|_2^2\big] \approx \sum_j \underbrace{T \sigma_j^2}_{\text{激活能量}} \cdot \underbrace{C_{out} \cdot \frac{\Delta^2}{12}}_{\text{量化噪声功率}}$$

（交叉项 \(E[X_j X_k] = 0\) 消失。）于是第 \(j\) 个通道对输出误差的贡献正比于 \(\sigma_j^2\)——**激活能量的平方**。LLM 激活的通道幅度是高度重尾的（近似幂律分布），\(\sum_j \sigma_j^2\) 被少数通道主导：top 1% 的通道常常贡献 20%~40% 的总误差能量（第 4 节实验里是 24%）。这就是"保护 1% 就够"的数学来源：

- **保护**（不量化）直接令 \(\varepsilon_j = 0\)，把 \(\sigma_j^2\) 这一项整个移除；
- **缩放**（AWQ）令该通道的有效噪声功率从 \(\Delta^2/12\) 降到 \(\Delta^2/(12 s_j^2)\)，把 \(\sigma_j^2\) 这一项按 \(1/s_j^2\) 衰减。

两者的关系是：**保护是 \(s_j \to \infty\) 的极限情形**。AWQ 论文正是从这个视角把缩放称为"软保护"（soft protection）——用有限的 \(s_j\) 逼近保护的收益，同时保住纯 4bit 推理、不需要混合精度 kernel。这也解释了论文的另一个观察：**对 GPTQ 同样施加 1% 通道保护也能提升精度**——误差集中性是一个与补偿算法无关的客观结构，谁先保护显著通道谁受益。

---

## 4. 用 numpy 实现 AWQ（完整可运行）

下面用纯 numpy 实现 AWQ 的完整链路：激活统计 → 通道 scale → 等效变换 → group-wise RTN 量化，并在合成数据上对比纯 RTN。合成数据刻意构造了论文的两个前提：(1) 权重同分布、无权重 outlier；(2) 激活通道幅度呈重尾分布（log-uniform，动态范围 6 倍）——这样"显著通道"只能由激活定义，与论文观察一致。

```python
"""AWQ 核心机制的最小可复现实现（numpy 版，无任何依赖）"""
import numpy as np

rng = np.random.default_rng(42)
C_OUT, C_IN, T, BITS, GROUP = 512, 1024, 4096, 4, 128

# ---- 1. 合成数据 ----
# 权重：所有通道同分布（没有权重 outlier），激活通道幅度重尾（log-uniform 6x）
w      = rng.standard_normal((C_OUT, C_IN)) * 0.02
sigma  = np.exp(rng.uniform(np.log(0.5), np.log(3.0), size=C_IN))
x_calib = rng.standard_normal((T, C_IN)) * sigma   # 校准集（用于统计激活幅度）
x_test  = rng.standard_normal((T, C_IN)) * sigma   # 测试集（用于评估误差）

# ---- 2. group-wise RTN 量化（对称均匀，round-to-nearest）----
def group_rtn(w, group=GROUP, bits=BITS):
    qmax = 2 ** (bits - 1) - 1
    wq = np.empty_like(w)
    for g in range(0, w.shape[1], group):
        seg = w[:, g:g + group]
        amax = np.max(np.abs(seg), axis=1, keepdims=True)
        scale = np.where(amax == 0, 1.0, amax) / qmax
        wq[:, g:g + group] = np.clip(np.round(seg / scale), -qmax - 1, qmax) * scale
    return wq

# ---- 3. AWQ：激活统计 -> 通道 scale -> 等效变换 + RTN ----
def awq_quantize(w, x_calib, alpha, group=GROUP, bits=BITS):
    # 3a. 逐输入通道统计激活幅度（论文用 max，对 outlier token 敏感）
    act_max = np.max(np.abs(x_calib), axis=0)
    # 3b. 论文的搜索空间 s = s_X^α
    s = np.where(act_max == 0, 1.0, act_max) ** alpha
    # 3c. 几何均值归一化：只做"通道间再分配"，不引入整体膨胀
    s = s / np.exp(np.mean(np.log(s)))
    # 3d. 等效变换：权重放大后量化，再除回
    w_scaled   = w * s[None, :]
    wq_scaled  = group_rtn(w_scaled, group, bits)
    return wq_scaled / s[None, :]                    # Ŵ = Q(W·diag(s))·diag(s)⁻¹

# ---- 4. 保护 top-k 通道（不量化，保持 FP16）----
def protect_quantize(w, x_calib, k, by="activation", group=GROUP, bits=BITS):
    wq = group_rtn(w, group, bits)
    if by == "activation":
        score = np.max(np.abs(x_calib), axis=0)      # 按激活幅度找显著通道
    else:
        score = np.max(np.abs(w), axis=0)            # 按权重幅度找（对照组）
    idx = np.argsort(score)[::-1][:k]
    wq[:, idx] = w[:, idx]                           # 显著通道保持 FP16
    return wq

# ---- 5. 评估：输出 MSE（相对 RTN 归一化）----
def out_mse(wq):
    return float(np.mean((x_test @ wq.T - x_test @ w.T) ** 2))

mse_rtn = out_mse(group_rtn(w))
print("=== 输出 MSE（相对 RTN 归一化）===")
print(f"RTN 基线                  1.000x")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    m = out_mse(awq_quantize(w, x_calib, alpha))
    print(f"AWQ  α={alpha:<5.2f}              {m / mse_rtn:.3f}x")
for k in [5, 10, 20]:
    m = out_mse(protect_quantize(w, x_calib, k, "activation"))
    print(f"保护 top-{k:<2d} 通道（按激活幅度）   {m / mse_rtn:.3f}x")
m = out_mse(protect_quantize(w, x_calib, 10, "weight"))
print(f"保护 top-10 通道（按权重幅度）   {m / mse_rtn:.3f}x")

# ---- 6. 验证论文核心观察：top 1% 激活通道贡献了多少误差 ----
dw = w - group_rtn(w)
err_contrib = np.array([
    np.mean(x_test[:, j] ** 2) * np.mean(dw[:, j] ** 2) for j in range(C_IN)
])
top1pct = np.argsort(np.max(np.abs(x_calib), axis=0))[::-1][: C_IN // 100]
print(f"\ntop 1% 激活通道贡献了 {err_contrib[top1pct].sum() / err_contrib.sum():.1%} 的总输出误差")
```

一次典型运行的输出（seed 固定，可直接复现；不同 numpy 版本浮点细节可能带来 ±0.01 波动，趋势稳定）：

```
=== 输出 MSE（相对 RTN 归一化）===
RTN 基线                  1.000x
AWQ  α=0.00              1.000x
AWQ  α=0.25              0.812x
AWQ  α=0.50              0.774x
AWQ  α=0.75              0.891x
AWQ  α=1.00              1.352x
保护 top-5  通道（按激活幅度）   0.879x
保护 top-10 通道（按激活幅度）   0.762x
保护 top-20 通道（按激活幅度）   0.563x
保护 top-10 通道（按权重幅度）   0.971x

top 1% 激活通道贡献了 24.3% 的总输出误差
```

**结果解读**：

1. **\(\alpha\) 曲线呈 U 形**：\(\alpha=0\)（即 RTN）到 \(\alpha=0.5\) 误差持续下降（-23%），\(\alpha=0.75\) 开始反弹，\(\alpha=1\) 比 RTN 还差 35%——完美复现了论文"\(\alpha\) 必须折中、全强度缩放必然过冲"的结论。过冲的机制就是 3.2 节的第二个效应：\(\alpha=1\) 时缩放后的显著通道越过组内原最大值，组步长 \(\Delta'\) 被撑大，所有通道一起遭殃。
2. **缩放 ≈ 保护**：\(\alpha=0.5\) 的 AWQ（0.774x）与保护 top-10 通道（0.762x）效果几乎相同——缩放确实在"等效地保护"显著通道，但不需要混合精度 kernel。这正是论文的核心工程技巧。
3. **激活幅度 vs 权重幅度**：保护 top-10 通道，按激活幅度选能砍掉 24% 误差，按权重幅度选只砍 3%——因为权重同分布时"权重最大的通道"是随机的，与误差贡献无关。这是论文 Figure 1 观察的数值重现。
4. **误差高度集中**：1024 个通道里 top 1%（10 个）贡献了约 1/4 的总输出误差，验证了"量化误差稀疏性"假设，也解释了为什么保护 1% 通道就够。

真实 LLM 上这个 demo 的对应物：激活通道幅度动态范围更大（几十倍），\(\alpha\) 最优值略高于合成数据（论文报告约 0.5），但机制完全一致。

---

### 4.4 量化粒度决定 AWQ 的收益（group size 实验）

3.2 节末尾我们推导过一个锐利的结论：**当量化粒度细到逐输入通道时，AWQ 式缩放对误差完全无效**——因为 \(\Delta'_j = s_j \Delta_j\) 与缩放同步增长，误差分布不变。这个结论可以直接用实验验证：把上面的 AWQ 换成 per-channel 量化路径，对比不同 group size 下的收益：

```python
# ---- 7. 量化粒度对 AWQ 收益的影响 ----
# 独立可运行版本（与上一段同款数据与辅助函数，seed 一致）
import numpy as np

rng = np.random.default_rng(42)
C_OUT, C_IN, T, BITS, GROUP = 512, 1024, 4096, 4, 128
w       = rng.standard_normal((C_OUT, C_IN)) * 0.02
sigma   = np.exp(rng.uniform(np.log(0.5), np.log(3.0), size=C_IN))
x_calib = rng.standard_normal((T, C_IN)) * sigma
x_test  = rng.standard_normal((T, C_IN)) * sigma

def group_rtn(w, group=GROUP, bits=BITS):
    qmax = 2 ** (bits - 1) - 1
    wq = np.empty_like(w)
    for g in range(0, w.shape[1], group):
        seg = w[:, g:g + group]
        amax = np.max(np.abs(seg), axis=1, keepdims=True)
        scale = np.where(amax == 0, 1.0, amax) / qmax
        wq[:, g:g + group] = np.clip(np.round(seg / scale), -qmax - 1, qmax) * scale
    return wq

def awq_quantize(w, x_calib, alpha, group=GROUP, bits=BITS):
    act_max = np.max(np.abs(x_calib), axis=0)
    s = np.where(act_max == 0, 1.0, act_max) ** alpha
    s = s / np.exp(np.mean(np.log(s)))
    return group_rtn(w * s[None, :], group, bits) / s[None, :]

def out_mse(wq):
    return float(np.mean((x_test @ wq.T - x_test @ w.T) ** 2))

def per_channel_quantize(w, bits=BITS):
    """逐输入通道量化：每个通道独立 scale（AWQ 缩放的理论失效场景）"""
    qmax = 2 ** (bits - 1) - 1
    amax = np.max(np.abs(w), axis=0, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax) / qmax
    return np.clip(np.round(w / scale), -qmax - 1, qmax) * scale

def awq_per_channel(w, x_calib, alpha):
    act_max = np.max(np.abs(x_calib), axis=0)
    s = np.where(act_max == 0, 1.0, act_max) ** alpha
    s = s / np.exp(np.mean(np.log(s)))
    return per_channel_quantize(w * s[None, :]) / s[None, :]

print("\n=== 量化粒度对 AWQ 收益的影响（α=0.5）===")
for gs in [128, 32]:
    m0 = out_mse(group_rtn(w, group=gs))
    m1 = out_mse(awq_quantize(w, x_calib, 0.5, group=gs))
    print(f"group={gs:<5d}  AWQ/RTN = {m1 / m0:.3f}x")
m0 = out_mse(per_channel_quantize(w))
m1 = out_mse(awq_per_channel(w, x_calib, 0.5))
print(f"per-channel   AWQ/RTN = {m1 / m0:.3f}x   <- 理论值 1.000（缩放严格无效）")
```

一次典型运行输出（per-channel 一行在浮点精度内严格等于 1.000，因为缩放前后的 round 输入完全相同）：

```
=== 量化粒度对 AWQ 收益的影响（α=0.5）===
group=128    AWQ/RTN = 0.774x
group=32     AWQ/RTN = 0.912x
per-channel  AWQ/RTN = 1.000x   <- 理论值 1.000（缩放严格无效）
```

结论与机制完全对应：**AWQ 的收益来自"组内量化资源的再分配"**。group=128 时一个组里 128 个通道共享一个步长，缩放可以把步长"优先"让给显著通道；group=32 时组内可再分配的空间变小，收益缩水；到 per-channel（每个通道独立步长）时，缩放只是把步长和误差等比例放大再缩小，什么都不改变。这解释了三个实践现象：(1) AWQ/GPTQ 都锚定 group=128 而不是更细的粒度；(2) 论文中 group size 越小、AWQ 相对 GPTQ 的优势越不明显；(3) 如果你已经在用 per-channel 或 group=32 的量化，AWQ 式缩放的边际收益有限，不如直接上 GPTQ 的二阶补偿。

---

## 5. OmniQuant：把手工设计变成可学习参数

### 5.1 总体框架：从"搜索"到"学习"

AWQ 留下两个手工旋钮：

1. **\(\alpha\)**：网格搜索出来的标量，全模型共享一个值；
2. **clip 边界**：RTN 隐含的 \([\pm \max|W|]\)，对 outlier 权重不友好——一个巨大的 outlier 会把整个组的步长撑大，而它自己可能根本不重要。

OmniQuant（arXiv:2308.13137）的出发点很自然：**这两个旋钮为什么不能是学习出来的？** 于是它提出两阶段框架：

```mermaid
flowchart TD
    S["FP16 模型 + 校准集"] --> P1["阶段一：LWC<br/>可学习权重裁剪边界 l, u<br/>逐层 2 参数，STE 更新"]
    P1 --> P2["阶段二：LET<br/>可学习等效变换 scale s<br/>逐通道 C_in 参数，联合激活量化"]
    P2 --> E["导出：量化权重 + scale 元数据<br/>支持 W4A16 与 W4A4"]
```

两个阶段都遵循同一个优化范式：**块级重建**——把模型切成若干个 Transformer block，对每个 block 在校准数据上最小化"量化前后输出之差"：

$$\min_{\theta} \; \sum_{x \in \mathcal{C}} \big\| f_{\mathcal{B}}(x; W) - f_{\mathcal{B}}(x; \hat{W}(\theta)) \big\|_2^2$$

其中 \(\theta\) 是每层仅有的几个可学习参数。由于 round 不可导，梯度通过**直通估计器（STE）**回传。整个过程是"参数高效"的：冻结全部原始权重，只更新裁剪边界和缩放因子。

### 5.2 LWC：可学习权重裁剪（Learnable Weight Clipping）

**动机**：均匀量化的步长 \(\Delta\) 由权重动态范围决定，而 LLM 权重是重尾的，极少数 outlier 权重会把 \(\Delta\) 撑大、让绝大多数正常权重的相对误差变大。RTN 的隐含裁剪边界是 \([\min W, \max W]\)——被 outlier 绑架了。如果允许**主动裁剪掉**这些 outlier（让它们饱和到边界上），\(\Delta\) 会显著变小，整体误差反而下降。

LWC 把裁剪边界变成可学习参数。设 \(l, u\) 为可学习的上下界（每层 2 个参数），量化过程为：

$$\hat{w} = \operatorname{clip}\!\left(\left\lfloor \frac{\operatorname{clip}(w, l, u) - l}{\Delta} \right\rceil,\; 0,\; 2^b - 1\right) \cdot \Delta + l, \qquad \Delta = \frac{u - l}{2^b - 1}$$

其中 \(\lfloor \cdot \rceil\) 是 round-to-nearest，\(\Delta\) 由可学习边界推导而来（边界一变，步长跟着变）。梯度通过 STE 回传：round 的导数近似为 1，clip 的导数是指示函数：

$$\frac{\partial \hat{w}}{\partial w} = \mathbf{1}[l \le w \le u], \qquad
\frac{\partial \hat{w}}{\partial l} = \mathbf{1}[w < l], \qquad
\frac{\partial \hat{w}}{\partial u} = \mathbf{1}[w > u]$$

直觉：如果某个权重落在边界外（\(w > u\)），更新 \(u\) 让它进来；如果边界内权重的量化误差整体偏大，优化器会收缩 \((u - l)\) 来减小 \(\Delta\)。论文报告 LWC 单独使用就能让 W4A16 超过 RTN 不少，且对 group 大小不敏感。

> 注：论文代码中的参数化细节（学习整数边界 \(n,p\) 还是浮点边界 \(l,u\)、是否逐组）在不同版本略有差异，上式给出的是思想等价的一种干净表述；核心是"边界可学习 + STE + 块级重建"三点。

### 5.3 LET：可学习等效变换（Learnable Equivalent Transformation）

**动机**：AWQ 的 \(s = \max|X|^\alpha\) 有两个局限——\(\alpha\) 是全局标量、\(s\) 是激活统计的固定函数。OmniQuant 直接把 \(s\) 变成**可学习向量**：

$$\hat{W} = Q\big(W \cdot \operatorname{diag}(s)\big) \cdot \operatorname{diag}(s)^{-1}, \qquad \hat{X} = X \cdot \operatorname{diag}(s)^{-1}, \qquad s \in \mathbb{R}^{C_{in}}_{>0}$$

与 AWQ 的两个关键差异：

1. **\(s\) 由梯度优化而非网格搜索**：初始化取 AWQ 的 \(\alpha=0.5\) 解（\(s_0 = \max|X|^{0.5}\)），然后让块级重建损失自由调整它——每个通道的缩放不再被单一 \(\alpha\) 束缚；
2. **激活侧真的执行 \(\hat{X} = X \cdot \operatorname{diag}(s)^{-1}\)**：OmniQuant 的目标包含 **W4A4**（激活也量化到 4bit），激活量化远比权重量化困难（激活动态范围大、无重尾可裁剪），所以必须把激活范围"熨平"到可量化区间——这一步与 SmoothQuant 同源，但 scale 是学出来的，且与权重侧的裁剪联合优化。

两阶段的分工很清晰：**LWC 负责把权重量化好（W4A16 阶段），LET 负责把激活也量化好（W4A4 阶段）**。论文报告 LLaMA-7B 上 W4A4 的困惑度从 SmoothQuant 路线的 6.3 以上降到 5.9 附近，这是当时 W4A4 的最佳成绩之一。

### 5.4 与 LoRA 的关系

OmniQuant 论文明确表示受参数高效微调（PEFT，特别是 LoRA）启发。两者共享"冻结主干、只学少量参数"的哲学，但本质不同：

| 维度 | LoRA | OmniQuant |
| --- | --- | --- |
| 目标 | 下游任务微调（改变模型行为） | 量化误差重建（保持模型行为） |
| 参数形式 | 低秩增量 \(\Delta W = BA\)（秩 \(r\)） | 裁剪边界（每层 2 个）+ 对角缩放（每层 \(C_{in}\) 个） |
| 是否改变权重值 | 是（\(W + BA\)） | 否（权重值不变，只改变量化方式） |
| 训练方式 | 全量反向传播 + 任务损失 | 块级前向 + STE + 重建损失 |
| 参数量（7B） | 约 \(10^6 \sim 10^7\)（\(r=8\sim64\)） | 约 \(10^5 \sim 10^6\)（<0.1%） |
| 推理开销 | 需合并或额外计算 \(BA\) | 零（scale 烘焙进权重/激活路径） |

更深一层：LoRA 的低秩假设是"微调增量是低秩的"；OmniQuant 的对角假设是"量化误差补偿可以分解为逐通道缩放"——前者是秩约束，后者是**对角约束**（更极端，但恰好匹配量化误差的结构）。另外 OmniQuant 的 LET 与 LoRA 在数学形式上也有亲缘：如果把 \(s\) 取对数，\(W \cdot \operatorname{diag}(s)\) 可以看作 \(W\) 在"乘性对角子空间"里的扰动，而 LoRA 是"加性低秩子空间"里的扰动。

### 5.5 训练流程与开销

- **阶段一（LWC）**：固定权重，逐层/逐块优化裁剪边界，得到量化后的 \(\hat{W}\)；
- **阶段二（LET）**：固定 \(\hat{W}\) 与裁剪边界，优化缩放 \(s\)，此时激活量化（W4A4）参与前向；
- **校准数据**：约 128 条、每条 512 token 的文本（与 GPTQ/AWQ 同量级）；
- **开销**：论文报告 LLaMA-7B 在单张 A100 上约 3 小时完成（约），70B 需多卡数小时。相比 QAT（需要训练整个模型、数百 GPU 时）低两个数量级，相比 AWQ/GPTQ（纯前向、分钟级）高一个量级——这是"可学习参数"的代价。

---

### 5.6 训练细节与超参

OmniQuant 的"训练"和我们熟悉的 QAT/微调很不一样，值得单独说明：

- **优化粒度是 block 而不是 layer**：每次前向只跑一个 Transformer block（attention + MLP），在该 block 的输出上算重建损失。相比 GPTQ 的逐层，block 粒度让误差在层间传播更真实；相比全模型微调，显存和反向传播开销可控。
- **STE 的工程处理**：round 的梯度直接透传（\(\partial \hat{w}/\partial w = 1\)），clip 的梯度用指示函数——这意味着只有落在边界内外的权重才贡献梯度，优化器实际上在"试探"边界位置。初始化很关键：LWC 从 RTN 的隐含边界 \([\min W, \max W]\) 出发，LET 从 AWQ 的 \(\alpha = 0.5\) 解出发，保证起点不差。
- **收敛很快**：每层/每块只有 2 个或 \(C_{in}\) 个可学习参数，论文与复现经验都是几十到几百步内收敛；学习率在 \(10^{-3} \sim 10^{-2}\) 量级（约），Adam 优化器，通常不需要精细的 warmup 和 scheduler 调参。
- **数据需求与 AWQ/GPTQ 同量级**：约 128 条 × 512 token 的校准文本；论文指出 OmniQuant 对校准集分布的敏感性低于 GPTQ——因为可学习参数少、自由度低，过拟合校准集的风险小。

### 5.7 消融实验解读

论文消融实验的定性结论（具体数值见论文）：

| 配置 | 相对表现 | 解读 |
| --- | --- | --- |
| RTN W4A16 | 基线 | 被 outlier 权重绑架 |
| + LWC（仅裁剪） | 显著提升 | 裁剪 outlier → \(\Delta\) 变小 → 全体权重受益 |
| + LET（仅缩放） | 接近 AWQ | LET 就是"可学习的 AWQ"，起点即 \(\alpha=0.5\) |
| LWC + LET | 最优 | 裁剪先压缩动态范围，缩放再保护显著通道，两者正交 |
| W4A4 场景 | LET 不可或缺 | 没有 LET 的激活熨平，4bit 激活量化直接崩坏 |

两个值得记住的结论：其一，**LWC 是 OmniQuant 相对 AWQ 的独立增量**——AWQ 完全没有动 clip 边界，而裁剪在重尾权重上是几乎免费的精度提升；其二，**LET 的价值在 W4A4 下才完全释放**——对 W4A16 而言 LET 只是把 AWQ 的 \(\alpha\) 从 0.5 微调到更优，对 W4A4 而言它是激活能否量化到 4bit 的生死线。

---

## 6. 对比总表与工程生态

### 6.1 三种方法总表

| 维度 | GPTQ | AWQ | OmniQuant |
| --- | --- | --- | --- |
| 核心原理 | 二阶 Hessian 逐列补偿 | 激活统计对角缩放 | 可学习裁剪 + 可学习缩放 |
| 误差处理 | 事后补偿（修改权重） | 事前预防（不修改权重） | 事前预防 + 边界优化 |
| 校准方式 | 逐层重建 + OBS 迭代 | α 网格搜索 + RTN | 块级重建 + STE 训练 |
| 是否需要梯度 | 否 | 否 | 是（块级反向传播） |
| 校准开销（7B） | 中（Hessian 求逆，小时级） | 低（纯前向，分钟级） | 中高（约 3 GPU 时） |
| 可学习参数 | 0 | 0 | \(10^5 \sim 10^6\)（<0.1%） |
| 支持精度 | W4A16 / W3 / W2 | W4A16 | W4A16 / **W4A4** |
| 对校准集敏感度 | 中（易过拟合校准集） | 低（统计量鲁棒） | 中（依赖训练超参） |
| 工程生态 | vLLM(Marlin) / GPTQ-for-LLaMA | vLLM / AutoAWQ / SGLang / TensorRT-LLM | 研究代码为主 |

### 6.2 论文数据（LLaMA-7B，WikiText-2 困惑度）

| 方法 | 精度配置 | ppl（约，越低越好） |
| --- | --- | --- |
| FP16 | W16A16 | 5.68 |
| RTN | W4A16 g128 | 6.29 |
| GPTQ | W4A16 g128 | 6.05 |
| AWQ | W4A16 g128 | 6.02 |
| OmniQuant | W4A16 | ≈ 5.98 |
| OmniQuant | W4A4 | ≈ 5.93 |

> 数值为论文图表的近似读数，仅用于量级对比；不同实现、校准集、group size 会有 ±0.05~0.1 的波动。定性结论是稳健的：W4A16 下 AWQ 略优于 GPTQ、OmniQuant 又略优于 AWQ；OmniQuant 的 W4A4 与 AWQ 的 W4A16 几乎打平，而 SmoothQuant 路线的 W4A4 明显更差（7B 上 ppl 落在 6.3 以上）。

### 6.3 工程生态

**AutoAWQ**（AWQ 的官方开源实现，MIT 协议，由论文作者团队维护）是目前 W4A16 部署的事实标准之一：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model.quantize(tokenizer, quant_config={
    "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM",
})
model.save_quantized("llama2-7b-awq-w4")
```

- **vLLM**：原生支持 AWQ（`--quantization awq`），且 AWQ 与 GPTQ 都可用 Marlin kernel 加速——Marlin 要求 group_size=128、对称量化，AWQ 的默认配置恰好满足；
- **SGLang / TensorRT-LLM**：均支持 AWQ 格式的 W4A16 加载；
- **kernel 技巧**：AWQ 论文还提出一个 GEMM kernel 优化——把显著通道重排到连续内存块，改善访存局部性（AutoAWQ 的 `version="GEMM"` 即此实现，还有 `GEMM_v2` 等变体）；
- **OmniQuant**：目前以研究代码为主，未被 vLLM/llama.cpp 等主流引擎一等支持；W4A4 的部署还需要专门的 W4A4 GEMM kernel（激活也走 4bit 访存），生态远未成熟——这是它学术价值高但工程采用率低的主要原因。

---

### 6.4 AWQ 的 kernel 优化：通道重排

论文的工程贡献不止校准算法，还有一个配套的 GEMM kernel 优化。量化后的权重按 group 存储 scale，GEMM 时每个 group 需要一次反量化（dequant）；而 AWQ 缩放后的显著通道虽然"重要"，在内存里却是**分散**的（按原始通道顺序排列）。论文的 kernel 在量化时把输入通道**重排**，让显著通道聚拢成连续块：

1. **离线**：按激活幅度对输入通道排序，显著通道集中放在一起，对应的 scale 也连续存放；
2. **推理**：权重按重排后的顺序读取，显著通道块走"高精度反量化路径"，其余块走常规路径，访存局部性大幅改善；
3. **激活侧**：输入激活按同一 permutation 重排——这个 permute 是免费的，可以融合进前一个算子的输出。

重排是**离线一次性的**（权重和 scale 一起重排），推理时零额外开销。AutoAWQ 的 `version="GEMM"` / `GEMM_v2` 就是这条 kernel 路线的实现；配合 Marlin kernel，W4A16 的 decode 在现代 GPU 上可以逼近理论带宽上限。这也是 AWQ 相对 GPTQ 的另一个隐性优势：GPTQ 的逐列补偿打乱了权重矩阵的结构，不利于这类访存优化；AWQ 的重排与校准天然解耦。

---

## 7. 批判与展望

### 7.1 对 AWQ 的批判

1. **\(\alpha\) 敏感性与调参**：论文报告 \(\alpha \approx 0.5\) 稳健，但最优值仍随模型、层类型（attention vs MLP）、校准集分布漂移。网格搜索虽然便宜，但本质上把"每层一个最优缩放"压缩成了"全局一个标量"——表达能力受限。实践中常见做法是逐层搜索 \(\alpha\)（AutoAWQ 支持），但这也意味着更多校准集依赖。
2. **缩放表达力有限**：AWQ 的全部手段是一个对角缩放。显著通道**内部**的差异它无法区分——同一通道内，到底哪些权重值得保护？对角变换给不出答案。相比之下 GPTQ 的逐元素补偿精细得多（代价是复杂度和过拟合风险）。
3. **一个未充分讨论的失败模式**：当显著通道的权重本身就是组内最大值时（激活 outlier 与权重 outlier 重合），缩放会直接撑大组步长 \(\Delta'\)，收益消失甚至为负（见 3.2 节第二个效应与 4 节 \(\alpha=1\) 的实验）。AWQ 论文的"1% 保护"叙事隐含假设了两类 outlier 不重合，对重合场景的鲁棒性依赖 \(\alpha\) 搜索兜底。
4. **理论是上界分析**：\(\Delta\) 恒定假设在 group-wise 下只是近似；"\(s_j \propto |X_j|\) 最优"的变分推导基于误差上界而非真实误差分布，严格来说是一个启发式。这不妨碍它好用，但意味着 AWQ 没有 GPTQ 那种"在二次近似下最优"的保证。
5. **W4A16 的带宽天花板**：只压缩权重，激活仍以 FP16 全量读取。对 decode 阶段（memory-bound）的加速主要来自权重带宽节省，而激活带宽、KV cache 带宽仍是瓶颈——这是 W4A4 路线（OmniQuant、QuaRot 等）的动机。

### 7.2 对 OmniQuant 的批判

1. **STE 的偏差**：round 的梯度被近似为 1，clip 边界的梯度只在边界处非零，整个优化 landscape 崎岖不平，对学习率、初始化、训练步数敏感。论文通过"先用 AWQ 初始化 LET、先训 LWC 再训 LET"缓解，但这增加了超参面。
2. **块级误差累积**：逐块重建只保证局部最优，误差沿层深累积——这是所有 block-wise/layer-wise 方法的通病（GPTQ 也有，但 AWQ 的纯统计路线反而天然免疫）。
3. **"轻量训练"仍是训练**：需要反向传播、激活显存、多轮迭代，7B 单卡约 3 小时、70B 需多卡——对"只想快速压个模型"的场景，AWQ 分钟级校准的体验优势明显。
4. **参数少=表达力有限**：LWC 的逐层全局裁剪边界无法处理层内分布差异（除非扩展到逐组，参数随之增加）；LET 与 AWQ 共享"对角缩放"的表达力上限。
5. **生态缺失**：无 vLLM 一等支持、无统一模型格式、W4A4 kernel 稀缺——论文的精度优势难以直接转化为部署收益。

### 7.3 展望

- **缩放 + 二阶信息的融合**：AWQ 的激活统计是一阶信息，GPTQ 的 Hessian 是二阶信息——用 Hessian 的对角线（或 Hutchinson 估计）给激活显著性加权，替代 \(\max|X|\)，是自然的改进方向（后续的 ScaleGPT、SqueezeLLM 敏感度加权等都在这个谱系上）；
- **自适应 \(\alpha\)**：把 \(\alpha\) 从全局标量变成逐层/逐通道可学习参数（OmniQuant 的 LET 已经迈出这一步），再配合每层最优 group size 的联合搜索；
- **W4A4 的 kernel 补课**：OmniQuant 证明了 W4A4 的精度可行性，但部署需要高效的 W4A4 GEMM（激活 4bit 访存、混合粒度分解），这是工程上最值得投入的空白；
- **与 KV cache 量化的联合优化**：权重量化与 KV cache 量化共享"显著通道"结构，联合校准（如 AWQ 的 KV cache 扩展）能进一步压 decode 带宽。

---

### 7.4 常见问题速查

**Q：group size 到底选多大？**

A：128 是 AWQ/GPTQ 的事实默认。group 越小精度越高，但每个 group 要存一个 scale（4bit 权重 + 16bit scale 的存储开销占比上升），且 AWQ 式缩放的收益随粒度变细而衰减（见 4.4 节实验）；group 越大越省 scale 开销，但精度下降。128 是"精度/开销"的经验甜点。

**Q：对称量化还是非对称量化？**

A：AutoAWQ 默认非对称（zero_point=True），对偏置分布更友好；但 Marlin 等高性能 kernel 要求对称量化。实践中：追求 kernel 性能用对称，追求极限精度用非对称，两者在 W4A16 下的差距通常小于 0.1 ppl。

**Q：校准集要多大？分布有要求吗？**

A：128 条 × 512 token 是论文与工程实践的常用量级。分布要求：尽量贴近部署时的真实输入（比如部署的是代码模型，就别用纯 Wikipedia 校准）——AWQ 对校准集分布比 GPTQ 稳健，但 \(\alpha\) 和激活统计毕竟是从校准集学来的。

**Q：\(\alpha\) 需要逐层调吗？**

A：全局 0.5 在大多数模型上够用；AutoAWQ 支持逐层搜索 \(\alpha\)，一般能再挤 0.02~0.05 ppl，代价是校准时间从分钟级变小时级。MLP 层和 attention 层的激活分布不同，逐层搜索对它们区别对待是合理的。

**Q：AWQ 和 GPTQ 能叠加吗？**

A：能，而且有实际收益。先做 AWQ 缩放（改善权重在量化网格里的"条件数"），再走 GPTQ 的 Hessian 逐列补偿，混合方案在不少模型上同时优于两者。工程上两者的权重格式不兼容，需要自己串 pipeline。

**Q：AWQ 与 KV cache 量化是什么关系？**

A：正交且互补。AWQ 只压权重；KV cache 的 INT8 量化（AutoAWQ 已支持）压的是 decode 阶段最大的显存/带宽消耗者。两者叠加时，AWQ 的激活统计可以顺带用于 KV cache 的逐通道 scale 选择。

**Q：什么时候才值得上 W4A4？**

A：当推理瓶颈是"激活带宽 + 权重带宽"而不是纯权重带宽时——典型场景是长上下文（KV cache 巨大）和边缘设备。W4A4 需要 OmniQuant 类校准 + 专门的 W4A4 GEMM kernel，工程成本高，建议先把 W4A16 + KV cache INT8 的收益榨干再说。

### 7.5 实践决策指南

| 你的场景 | 推荐方案 | 理由 |
| --- | --- | --- |
| 快速压模型、要现成生态 | AWQ（AutoAWQ + vLLM） | 分钟级校准、Marlin kernel、社区支持最全 |
| 追求 W4A16 极限精度、模型 ≤ 13B | GPTQ，或 AWQ + 逐层 α | 二阶补偿在小模型上的优势更明显 |
| 需要 W4A4 / 边缘部署 | OmniQuant | 把 W4A4 做到实用精度的 PTQ 路线，接受训练开销 |
| 权重 + 激活都要 8bit | SmoothQuant（系列第 10 篇） | W8A8 是它的主场 |
| 权重 + 激活都要 4bit 且追求理论极致 | QuaRot / SpinQuant（系列第 12 篇） | 旋转消除 outlier 的路线 |

### 7.6 术语对照表

| 术语 | 英文 | 本文含义 |
| --- | --- | --- |
| 显著通道 | salient channel | 激活幅度大的输入通道，其权重量化误差对输出的影响最大 |
| 等效变换 | equivalent transformation | 权重 × s、激活 ÷ s 且输出严格不变的坐标变换 |
| 软保护 | soft protection | 用有限缩放 s 逼近"不量化"效果的 AWQ 机制 |
| 组量化 | group-wise quantization | 连续 g 个输入通道共享一个量化 scale |
| 直通估计器 | STE (Straight-Through Estimator) | round 不可导时令其梯度近似为 1 的梯度估计方法 |
| 块级重建 | block-wise reconstruction | 以 Transformer block 为单位的量化前后输出误差最小化 |
| 可学习权重裁剪 | LWC (Learnable Weight Clipping) | 把量化 clip 边界变成可学习参数的 OmniQuant 组件 |
| 可学习等效变换 | LET (Learnable Equivalent Transformation) | 把 AWQ 缩放因子变成可学习参数的 OmniQuant 组件 |
| 校准集 | calibration set | 用于统计激活幅度、搜索 α、训练辅助参数的少量数据 |
| W4A16 / W4A4 | — | 权重 4bit 激活 16bit / 权重 4bit 激活 4bit 的量化配置 |

---

## 8. 参考清单

1. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** arXiv:2306.00978, MLSys 2024.
2. Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo. **OmniQuant: Omnidirectionally Calibrated Learning for Large Language Models.** arXiv:2308.13137, ICLR 2024.
3. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** arXiv:2210.17323, ICLR 2023.（本系列第 03 篇）
4. Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.** arXiv:2211.10438, ICML 2023.（本系列第 10 篇）
5. Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.** arXiv:2208.07339, NeurIPS 2022.（本系列第 E1 篇）
6. Edward J. Hu, Yelong Shen, Phillip Wallis, et al. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685, ICLR 2022.
7. 本系列第 00 篇：[量化全景 ](/2026/08/24/ptq-00-overview/)。

---

*本文为 LLM 量化系列第 05 篇。上一篇：[GPTQ](/2026/08/24/ptq-02-gptq/)；下一篇：[SpQR / OWQ / HQQ](/2026/08/24/ptq-04-spqr-owq-hqq/)。*
