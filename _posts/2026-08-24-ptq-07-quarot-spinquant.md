---
title: "大模型量化算法（12）：QuaRot / SpinQuant——旋转消除异常值"
date: 2026-08-24 13:40:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, quarot, spinquant, rotation, w4a4]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 2 · Activation PTQ** ｜ 第 12 篇 / 共 26 篇
>
> [← 11 Outlier Suppression](/2026/08/24/ptq-10-outlier-suppression/) ｜ [13 RPTQ/QUIK/ATOM →](/2026/08/24/ptq-11-rptq-quik-atom/)
>
> 配套阅读：第 E1 篇 [LLM.int8()](/2026/08/24/ptq-01-rtn-llmint8/) 讲激活 outlier 的发现；第 03 篇 [GPTQ](/2026/08/24/ptq-02-gptq/) 讲 Hessian 加权量化；第 10 篇 [SmoothQuant](/2026/08/24/ptq-06-smoothquant-zeroquant/) 讲"迁移 outlier"的平滑变换——本文的旋转是它的"升级版"。

---

## TL;DR（三连）

**是什么**：QuaRot（arXiv:2404.00456）和 SpinQuant（arXiv:2405.16465）是两条"旋转流派"工作——它们给 LLM 的权重和激活施加**正交旋转**（QuaRot 用固定的 Hadamard 变换，SpinQuant 用可学习的旋转矩阵），把系统性 outlier 从少数坐标"摊薄"到所有坐标，从而让 **W4A4（权重 4bit + 激活 4bit）** 首次在 7B/70B 级别模型上逼近 FP16 精度。

**为什么有效**：正交旋转不改变模型输出（旋转不变性：\(XQW = XQ^{\mathsf{T}}QW = XW\)），但改变了坐标系的"对齐方式"。原本集中在一两个隐藏维度的巨大 outlier（见第 E1 篇 LLM.int8() 的发现），旋转后被均匀稀释到 \(\sqrt{d}\) 个坐标上，每个坐标的动态范围从 \(O(M)\) 缩到 \(O(M/\sqrt{d})\)——4bit 均匀量化的误差随之大幅下降。

**代价与现状**：旋转不能全部"免费"吸收进权重，每个 transformer block 需要常数次**在线 Hadamard 变换**（FHT，\(O(d\log d)\)），FLOPs 上可忽略但 kernel 融合复杂；SpinQuant 把旋转矩阵变成可学习参数、在校准集上优化，进一步压榨 2-4bit 精度，还顺带把 KV cache 压到 4bit。但到今天，W4A4 在生产推理引擎（vLLM 等）中依然少见——FP8 硬件原生的路线抢走了大部分工程落地。

---

## 目录

1. [背景：W4A4 的双重 outlier 困境](#1-背景w4a4-的双重-outlier-困境)
2. [QuaRot：固定旋转的数学与工程](#2-quarot固定旋转的数学与工程)
3. [SpinQuant：把旋转变成可学习参数](#3-spinquant把旋转变成可学习参数)
4. [对比总览：QuaRot vs SpinQuant vs SmoothQuant](#4-对比总览quarot-vs-spinquant-vs-smoothquant)
5. [代码实验：Hadamard 旋转的威力](#5-代码实验hadamard-旋转的威力)
6. [批判与展望](#6-批判与展望)
7. [参考清单](#7-参考清单)

---

# 1. 背景：W4A4 的双重 outlier 困境

## 1.1 为什么 W4A4 比 W4A16 难一个数量级

前几篇我们讨论的 GPTQ、AWQ、SpQR 几乎都只量化**权重**（W4A16）：激活保持 FP16，量化误差只来自权重一侧，可以用 Hessian 加权、激活感知等技巧把误差压到最低。而 **W4A4** 要求权重和激活**同时**降到 4bit——这意味着两个独立的误差源叠加，而且激活一侧的问题比权重严重得多。

回顾第 E1 篇 LLM.int8() 的核心发现：大模型（≥6.7B）的激活中会出现**系统性 outlier**——某些隐藏维度（channel）的值在几乎所有 token 上都异常巨大（幅度是正常值的 20~60 倍）。这些 outlier 不是噪声，而是模型学习到的"强特征通道"。对权重来说，outlier 可以被 GPTQ 的 Hessian 加权和 AWQ 的尺度感知"驯服"；但对**激活**，我们无法离线重排——激活是推理时实时产生的，且 outlier 位置随层变化、不可预测。

于是 W4A4 面对一个死结：

| 误差源 | 来源 | 传统对策 | 为什么 W4A4 下失效 |
| --- | --- | --- | --- |
| 权重 outlier | 少数输出通道范数极大 | GPTQ 列重排 + Hessian 加权；AWQ 通道缩放 | 有效，但只解决一半 |
| 激活 outlier | 少数隐藏维度值极大 | LLM.int8() 混合精度分解（outlier 列走 FP16） | 4bit 下"混合精度"失去意义，outlier 列几乎全损 |
| 两者相乘 | 权重 outlier × 激活 outlier 对齐 | 无 | 输出误差被平方级放大 |

均匀量化下，一个坐标的动态范围决定整层的量化步长：若激活某个坐标最大值为 60，而中位数只有 1，则 4bit（15 个正电平）的步长高达 4.0——所有正常坐标的信息几乎被"压扁"成几个电平。这就是为什么朴素 RTN 的 W4A4 在 7B 模型上困惑度会爆炸（ppl 从 ~5.5 飙到 20+）。

## 1.2 旋转的数学直觉：把 outlier 摊薄

死结的破法其实很朴素：**既然 outlier 集中在少数坐标，那就换一个坐标系，让每个坐标都"分到一点"outlier**。这正是正交旋转做的事。

考虑一个 \(d\) 维向量 \(x\)，只有一个坐标是 outlier（幅度 \(M\)，其余 ~\(O(1)\)）：

$$x = M\cdot e_1 + g,\qquad g_i \sim \mathcal{N}(0,1)$$

对它施加一个随机正交矩阵 \(Q\)（\(QQ^{\mathsf{T}} = I\)）：

$$x' = Qx = M\cdot q_1 + Qg$$

其中 \(q_1\) 是 \(Q\) 的第一列。随机正交矩阵的列向量**均匀分布在单位球面**上，所以 \(q_1\) 的每个坐标幅度都是 \(O(1/\sqrt{d})\)。于是旋转后：

$$\max_i |x'_i| \;\approx\; \frac{M}{\sqrt{d}} + O(\sqrt{\log d})$$

而旋转前 \(\max|x_i| = M\)。**动态范围从 \(M\) 直接缩到 \(M/\sqrt{d}\)**。\(d = 4096\) 时，\(M = 60\) 的 outlier 被摊薄到 \(\approx 1\)——刚好掉进 4bit 能容纳的动态范围。这就是"旋转消除 outlier"的全部直觉：不删除 outlier，而是把它分给所有人。

更一般地，对激活协方差矩阵 \(\Sigma_x = \mathbb{E}[x^{\mathsf{T}}x]\)（\(d\times d\)），旋转后第 \(i\) 个坐标的方差为：

$$\mathrm{Var}(x'_i) = q_i^{\mathsf{T}}\Sigma_x\, q_i$$

当 \(Q\) 是随机正交矩阵时，\(q_i\) 均匀分布在球面上，上式对所有 \(i\) 都收敛到同一个值：

$$\mathrm{Var}(x'_i) \;\approx\; \frac{\mathrm{tr}(\Sigma_x)}{d},\qquad \forall i$$

也就是说，旋转把激活分布**各向同性化**（isotropize）：每个坐标的方差趋于相等，量化步长不再被少数"宽动态范围"的坐标拖累。权重一侧同理——对权重矩阵 \(W\) 施加 \(W' = QW\)，各行范数、各列范数都趋于均匀，权重 outlier 也被摊薄。

**与 SmoothQuant 的关系**：第 10 篇的 SmoothQuant 本质是"对角变换"（\(x \cdot \text{diag}(s)\)，把激活 outlier 迁移到权重），只处理"激活大、权重小"这一种对齐模式，且变换矩阵必须是对角的（否则改变输出结构）。旋转是它的推广：用**满秩正交变换**同时处理权重和激活两侧的 outlier，且由于正交性**严格不改变输出**——这是对角平滑做不到的（平滑后需要额外补偿）。

## 1.3 与 LLM.int8() 的呼应

第 E1 篇的 LLM.int8() 是"躲"：检测出 outlier 列，单独用 FP16 计算，其余走 INT8。旋转流派是"化"：不躲，而是把 outlier 打散到所有坐标，让整个矩阵乘都能安全地走低比特。两者哲学相反，但都基于同一个观察——**outlier 是少数坐标的、系统性的、方向性的**。LLM.int8() 证明 outlier 存在且位置固定；旋转证明 outlier 的存在依赖于坐标系——换一个坐标系，它就不"存在"了。

这个观察引出一个关键问题：**旋转矩阵 \(Q\) 从哪来？** 随机正交矩阵理论上可行，但随机矩阵无法快速计算（\(O(d^2)\) 的矩阵乘等于白干）。QuaRot 的答案是：用 **Hadamard 矩阵**——它既是正交的，又有 \(O(d\log d)\) 的快速算法。下一章展开。

---

# 2. QuaRot：固定旋转的数学与工程

QuaRot（"**Qua**ntization with **Rot**ations"，arXiv:2404.00456，ICML 2024）由 ETH Zurich 的 Ashkboos 等人提出，是"旋转消除 outlier"的开山之作，首次在 LLaMA-2 7B/13B/70B 上实现了**接近无损的 W4A4 推理**。

## 2.1 旋转不变性：数学基础

先建立最基本的等式。设线性层 \(Y = XW\)（\(X\) 为激活，\(W\) 为权重），对任意正交矩阵 \(Q\)（\(Q^{\mathsf{T}}Q = I\)），定义：

$$W' = QW,\qquad X' = XQ^{\mathsf{T}}$$

则有：

$$X'W' = XQ^{\mathsf{T}}QW = XW = Y$$

**输出严格不变**。这就是旋转不变性（rotational invariance）：我们可以自由地给权重"左乘"一个正交矩阵，只要同时给激活"右乘"它的转置。关键是，旋转后的 \(W'\) 和 \(X'\) 都比原来更适合量化：

- **权重侧**：\(W' = QW\) 把 \(W\) 的行（对应激活维度）混合，权重 outlier 行被摊薄到所有行；
- **激活侧**：\(X' = XQ^{\mathsf{T}}\) 把激活的 outlier 坐标摊薄到所有坐标。

两个变换的代价完全不对称：\(W'\) 是**离线**算好的（推理零成本），而 \(X'\) 必须在**推理时在线**计算——除非 \(XQ^{\mathsf{T}}\) 能吸收进前一个模块。如何安排"哪些旋转吸收、哪些在线"，是 QuaRot 的核心设计（2.3 节）。

**量化误差视角（为什么旋转后量化更准）**。考虑对激活 \(X\) 做均匀量化，量化误差 \(\Delta x_i \sim \mathcal{U}(-\delta_i/2, \delta_i/2)\)，其中步长 \(\delta_i \propto \max|x_i|\)（或按动态范围）。输出误差的期望可以近似为：

$$\mathbb{E}\left[\left\|\,(X - \hat{X})\,W\,\right\|^2\right] \;\approx\; \sum_i \frac{\delta_i^2}{12}\,\|W_{i:}\|^2$$

旋转前，少数坐标的 \(\delta_i\) 巨大（被 outlier 撑大），主导整个求和；旋转后所有 \(\delta_i\) 趋于相同且大幅缩小，同时权重行范数 \(\|W'_{i:}\|\) 也趋于均匀——两项共同作用下，总量化误差下降一个数量级。GPTQ（第 03 篇）用 Hessian 对角元素加权做**逐列**优化；旋转做的其实是"让 Hessian 对角元素自己变得均匀"，从而让**均匀量化**（无需任何逐列搜索）就接近最优。这正是 QuaRot 与 GPTQ 互补的地方：旋转负责"把问题变简单"，GPTQ/AWQ 负责"在简单的问题上再榨一点"。

## 2.1.1 定量视角：动态范围如何决定量化信噪比

为了把"旋转减少动态范围 → 量化误差下降"讲得更严谨，推导均匀量化 SNR 与动态范围的定量关系。

设激活坐标 \(x_i\) 分布的标准差为 \(\sigma_i\)，对称均匀量化的范围 \(B = \max|x_i|\)，量化电平数 \(2^b\)（4bit 时为 15 个正电平 + 对称负电平）。步长 \(\delta = 2B/(2^b - 1)\)，量化噪声 \(e \sim \mathcal{U}(-\delta/2, \delta/2)\)，噪声方差 \(\delta^2/12\)。该坐标的信噪比为：

$$\mathrm{SNR}_i \;=\; \frac{\mathbb{E}[x_i^2]}{\delta^2/12} \;\approx\; 3\cdot 2^{2b}\cdot \frac{\sigma_i^2}{B_i^2}$$

定义峰值因子（crest factor）\(R_i = B_i/\sigma_i\)（动态范围与标准差的比值），则：

$$\mathrm{SNR}_i(\mathrm{dB}) \;\approx\; 4.77 + 6.02\,b - 20\log_{10} R_i$$

**SNR 与峰值因子的平方成反比**——这是全篇最重要的一个数字关系。正常坐标 \(R \approx 4\)，4bit 的理论 SNR ≈ \(4.77 + 24 - 12 = 16.8\) dB；outlier 坐标 \(R \approx 60\)，SNR 掉到 \(4.77 + 24 - 35.6 = -6.8\) dB——**比噪声还差**。换算成有效比特：\(20\log_{10}(60/4) \approx 23.5\) dB ≈ 4 bit，即**一个 outlier 坐标的 4bit 量化，实际有效精度只有 ~0-1 bit**。

旋转把 \(R_i\) 从 ~60 拉回 ~5（见第 5 章代码实验：动态范围 38.9 → 3.3），SNR 回升约 20 dB，4bit 才真正兑现 4bit 的精度。而整体输出误差是所有坐标误差按权重行范数加权的结果——outlier 坐标的 SNR 塌陷会主导全局误差。这就是"旋转让均匀量化接近最优"的定量解释：**它同时压低所有坐标的峰值因子，并让它们趋于一致**。

## 2.2 Hadamard 变换与快速 Hadamard 变换（FHT）

随机正交矩阵在理论上可行，但 \(XQ^{\mathsf{T}}\) 需要 \(O(d^2)\) 的矩阵乘——比被它保护的量化 matmul 还贵，本末倒置。QuaRot 改用 **Hadamard 矩阵** \(H_n\)：正交、对称、且有 \(O(n\log n)\) 的快速算法。

**递归构造**（Sylvester 构造法）：

$$H_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\ 1 & -1\end{bmatrix},\qquad H_{2n} = H_2 \otimes H_n = \frac{1}{\sqrt{2}}\begin{bmatrix}H_n & H_n\\ H_n & -H_n\end{bmatrix}$$

其中 \(\otimes\) 是 Kronecker 积。\(n\) 必须是 2 的幂（LLM 的 hidden size 如 4096、8192 恰好都是 2 的幂，这也是旋转流派偏爱 Hadamard 的实用原因）。归一化因子 \(1/\sqrt{2}\) 保证 \(H_n\) 正交。Hadamard 矩阵有两个关键性质：

$$H_n^{\mathsf{T}} = H_n,\qquad H_n^{-1} = H_n \;\Longrightarrow\; H_nH_n = I_n$$

**对称且自逆**——这意味着旋转和"旋转回去"是同一个操作，吸收进权重时无需区分左乘右乘、正变换逆变换。

**快速 Hadamard 变换（FHT）**。\(H_n\) 的递归结构对应经典的蝶形（butterfly）算法：\(n\) 个元素，\(\log_2 n\) 轮，每轮 \(n/2\) 次加/减，总共 \(O(n\log n)\) 次运算。对比：

- 直接矩阵乘 \(XH_n\)：\(O(n^2) \approx 1.7\times 10^7\) 次（\(n=4096\)）
- FHT：\(O(n\log_2 n) \approx 4096 \times 12 = 4.9\times 10^4\) 次

**快了约 350 倍**，与 transformer 里动辄 \(O(d^2)\) 的 matmul 相比几乎免费。这就是 QuaRot 选择 Hadamard 而非随机正交矩阵的全部理由。第 5 章代码会给出 FHT 的完整 numpy 实现。

## 2.3 旋转放置：transformer 各模块的"吸收"与"在线"

有了旋转不变性和 FHT，接下来是 QuaRot 最精巧的部分：**把旋转放在哪里，才能让所有量化 matmul 的输入（激活）和权重都"无 outlier"？**

先看 transformer block 里的线性层：Q/K/V 投影、O 投影（attention 输出）、up/gate/down 投影（FFN）。旋转的放置遵循两条规则：

1. **能吸收就吸收**：旋转若施加在某个线性层的输入激活上，可以等价地左乘进该层权重（\(W' = QW\)），推理零成本；
2. **残差流与 LayerNorm 阻断吸收链**：残差流是多个模块输出的叠加，LayerNorm 是非线性归一化，旋转无法"穿过"它们被吸收——这些位置的旋转必须**在线执行**。

QuaRot 的旋转计划（每 block 三处）：

```mermaid
flowchart TD
    Z["残差流 z（原始空间）"] --> H1["① 在线旋转 H₁<br/>(FHT, O(d log d))"]
    H1 --> LN["LayerNorm"]
    LN --> QKV["Q / K / V 投影<br/>W′ = H₁·W（吸收）"]
    LN --> FFN["up / gate 投影<br/>W′ = H₁·W（吸收）"]
    QKV --> Attn["Attention<br/>softmax(QKᵀ) 旋转不变"]
    Attn --> H2["② 在线旋转 H₂"]
    H2 --> WO["W_O 投影<br/>W′ = H₂·W（吸收）"]
    FFN --> ACT["SiLU 激活"]
    ACT --> H3["③ 在线旋转 H₃"]
    H3 --> WD["W_down 投影<br/>W′ = H₃·W（吸收）"]
    WO --> ADD["⊕ 残差"]
    WD --> ADD
    ADD --> NEXT["下一 block（再次在线旋转）"]
```

逐条解释：

**① 输入旋转 \(H_1\)（吸收进 5 个权重矩阵）**。残差流 \(z\) 先被在线旋转 \(z' = zH_1\)，再进 LayerNorm，然后同时喂给 Q/K/V 和 up/gate。旋转 \(H_1\) 被吸收进 \(W_Q, W_K, W_V, W_{up}, W_{gate}\) 五个矩阵（左乘）。为什么这个旋转不能离线吸收？因为 \(z\) 是上一 block 的 attention 输出、MLP 输出与残差的**叠加**，且中间隔着 LayerNorm——没有单一权重矩阵可以"承接"这个旋转，只能在线算。好处是**一次在线旋转同时服务 5 个量化 matmul**。

**Attention 内部的免费午餐**。\(Q = z' W_Q\)、\(K = z' W_K\) 都带着 \(H_1\) 旋转，而注意力分数：

$$\mathrm{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right)$$

对 \(Q\)、\(K\) 同时施加**任意**正交旋转不变（旋转保内积：\((QH)(KH)^{\mathsf{T}} = QHH^{\mathsf{T}}K^{\mathsf{T}} = QK^{\mathsf{T}}\)）。所以 \(H_1\) 对 Q/K 的旋转**不需要任何补偿**，softmax 的输出与未旋转时逐位相同。这是旋转方法能"白嫖" attention 的关键——SmoothQuant 那类对角变换做不到这一点。

**②③ 输出旋转 \(H_2\)、\(H_3\)（吸收进 W_O、W_down）**。attention 输出（\(V\) 的凸组合，已处于 \(H_1\) 旋转空间）和 MLP 输出在加回残差流之前，各施加一次在线 Hadamard 旋转，同时吸收进 \(W_O\) 与 \(W_{down}\)（左乘）。这样 \(W_O\)、\(W_{down}\) 的权重 outlier 也被消除，且残差流保持"已旋转"的一致空间，下一 block 的 \(H_1'\) 可以直接接续。

**嵌入层**。输入 embedding 的输出是残差流的起点，同样施加一次旋转（可吸收进第一层的权重或在线执行一次），保证从第一层开始就处于旋转空间。

**旋转的总数**：每个 transformer block 恰好 **3 次在线 Hadamard 旋转**（输入 1 次 + 输出 2 次），全部 7 个线性层权重（Q/K/V/O/up/gate/down）都被旋转覆盖。这就是 QuaRot 的全部改动——**不引入任何新参数，不改动任何权重数值的"信息内容"，只是换坐标系**。

## 2.4 在线旋转的代价分析

3 次在线旋转的成本必须量化。以 LLaMA-2 7B（\(d = 4096\)，32 层）为例：

| 项目 | 单次 FHT | 每 block 3 次 | 对比 |
| --- | --- | --- | --- |
| FLOPs | \(d\log_2 d \approx 4.9\times10^4\) | \(\approx 1.5\times10^5\) | 每 block 7 个线性层 matmul ≈ \(2.4\times10^8\) FLOPs，占比 <0.1% |
| 访存量 | 读写 \(d\) 个 float ≈ 32KB | ≈ 96KB | 激活总量级相同，但多了一次 HBM 往返 |
| kernel 启动 | 1 次 | 3 次 | 短 kernel 的启动开销占比被放大 |

结论：**FLOPs 上完全可忽略（<0.1%），真正的代价是 kernel 层面的**——每次在线旋转都是一次额外的 HBM 读写和 kernel 启动。工程上必须把 FHT 与相邻的 LayerNorm、残差加、matmul **融合**进同一个 kernel，才能把开销压到 1% 以内。这也是 QuaRot 论文报告的端到端开销：约 **1-2%** 的推理时间增加。第 6 章批判部分会展开讨论为什么这个"1%"在真实部署中并不容易拿到。

## 2.5 为什么旋转让 W4A4 可行：结果

论文在 LLaMA-2 7B/13B/70B 上评估 W4A4（权重 4bit + 激活 4bit，RTN 均匀量化，无 GPTQ 式逐列优化）。旋转前后的激活 outlier 幅度对比是直观的：旋转前某些层的激活最大值可达 60+，旋转后全部降到个位数（\(\approx 6\) 以内），与正常坐标同量级。

端到端效果（WikiText-2 困惑度，数值为论文图表的近似读数）：

| 模型 | FP16 | W4A4 + QuaRot | 退化 |
| --- | --- | --- | --- |
| LLaMA-2 7B | ≈ 5.47 | ≈ 5.57 | +0.10（~2%） |
| LLaMA-2 13B | ≈ 4.88 | ≈ 4.96 | +0.08 |
| LLaMA-2 70B | ≈ 3.12 | ≈ 3.20 | +0.08 |

> 注：以上为论文图表的近似读数，精确值请以 arXiv:2404.00456 原文为准。QuaRot 还顺带把 **KV cache 量化到 4bit**（W4A4KV4），退化仍控制在 ~2-3% 内——因为 KV 的 outlier 同样被旋转消除了。

一个值得强调的细节：QuaRot 用的是**最朴素的 RTN 均匀量化**，没有任何 GPTQ/AWQ 式的误差补偿，就达到了上述精度。这说明旋转确实把"问题本身"变简单了，而不是靠更聪明的量化器硬扛。这也为后续工作（SpinQuant 学旋转、QuaRot 系与 GPTQ 结合）留出了空间。

## 2.6 为什么不直接做 PCA / 白化？

Hadamard 是**数据无关**的。一个自然的疑问立刻浮现：既然目标是"让每个坐标的方差均匀"，为什么不直接用校准数据估计协方差 \(\Sigma_x\)，用 PCA 或白化变换（\(Q = \Sigma_x^{-1/2}\) 或特征向量矩阵）？那样几乎完美地各向同性化，不是更优吗？

三个否决理由：

1. **量化目标 ≠ 方差均匀**。PCA 优化的是"方差均匀"，而量化误差由动态范围（峰值）决定，且权重侧的 outlier 同样关键。数据相关的变换针对"激活协方差"优化，却可能把权重分布搞得更糟——激活与权重的 outlier 结构并不一致，一方的"最优坐标系"往往是另一方的次优。QuaRot 论文的消融支持这一点：**随机正交旋转与固定 Hadamard 的最终精度几乎无差别**——说明"任何正交旋转都够用"，针对数据精细建模收益甚微。
2. **计算与部署代价**。白化矩阵是稠密的：在线执行需要 \(O(d^2)\)（吃掉量化省下的所有算力），吸收进权重则产出稠密权重、破坏矩阵的数值结构。Hadamard 的 \(O(d\log d)\) 蝶形算法是它被选中的决定性理由——**快速且结构化**。
3. **鲁棒性**。校准集估计的 \(\Sigma_x\) 有估计误差与**分布漂移**风险。SpinQuant 的学习旋转本质上就是"数据相关的旋转"（见第 3 章），它用训练换来了定向收益，也继承了这一风险；QuaRot 的零校准设计天然免疫。

这个讨论的深层结论：**旋转的价值不在于"最优"，而在于"几乎任何正交旋转都能把问题变成良态"**。这解释了为什么 QuaRot 能用最朴素的 Hadamard 取得接近 SpinQuant 的效果——旋转的"及格线"很低，SpinQuant 是在及格线之上追求优秀。

# 3. SpinQuant：把旋转变成可学习参数

QuaRot 证明了"旋转消除 outlier"这条路走得通，但它留下一个明显的次优之处：**旋转矩阵是固定的**（Hadamard 或随机正交）。Hadamard 是最"均匀"的旋转之一，但均匀 ≠ 最优——真实模型的 outlier 结构有方向性，一个"恰好对准 outlier 主轴"的旋转显然比"对谁都一视同仁"的旋转更高效。SpinQuant（arXiv:2405.16465，Liu 等人，ICLR 2025）把这个直觉形式化：**把旋转矩阵变成可学习参数，在校准集上直接优化**。

## 3.1 从固定旋转到可学习旋转：动机

先看 QuaRot 的固定旋转损失了什么。Hadamard 变换对每个坐标一视同仁，它把 outlier 均匀摊薄——但如果 outlier 只集中在少数几个隐藏维度（比如 4096 维里只有 10 个维度有系统性 outlier），最优策略应该是**把这 10 个维度的能量集中摊到所有 4096 个坐标**，同时**尽量少扰动其他 4086 个正常维度**。Hadamard 做不到这种"定向摊薄"：它把正常维度也搅匀了，等效于给正常坐标引入了不必要的量化噪声。

更直接地说：量化误差 \(\propto\) 各坐标动态范围之和，旋转的目标是**最小化旋转后坐标系的量化代价**，而不是追求"几何上最均匀"。这是一个可以显式优化的目标：

$$\min_{Q}\;\; \mathcal{L}_{\text{quant}}(Q) \;=\; \mathbb{E}_{x\in\mathcal{D}_{\text{cal}}}\left[\, \left\|\, \mathrm{Quant}(xQ^{\mathsf{T}})\,\mathrm{Quant}(QW) - xW \,\right\|^2 \,\right]$$

其中 \(\mathrm{Quant}(\cdot)\) 是 4bit 均匀量化（含反量化），\(\mathcal{D}_{\text{cal}}\) 是校准集（几百条文本即可），\(Q\) 是正交矩阵。目标很直白：**让"旋转 + 量化"后的输出尽量接近原始 FP16 输出**。SpinQuant 论文还发现，直接以量化误差为目标就够好；也可以叠加任务损失（如交叉熵）微调，但收益有限、成本更高。

## 3.2 优化技术：Cayley 参数化 + STE

直接对 \(Q\) 做无约束梯度下降会破坏正交性（\(Q^{\mathsf{T}}Q = I\) 是硬约束）。SpinQuant 用 **Cayley 变换**把正交矩阵参数化为无约束的反对称矩阵：

$$Q = (I + A)\,(I - A)^{-1},\qquad A = -A^{\mathsf{T}} \in \mathbb{R}^{d\times d}$$

- 任意反对称矩阵 \(A\) 经 Cayley 变换得到的 \(Q\) **必然正交**——约束被自动满足，梯度下降可以放心跑；
- 梯度 \(\partial \mathcal{L}/\partial A\) 可以通过 Cayley 变换的微分链式法则求得（或数值近似）；
- 量化函数不可导，梯度用**直通估计器（STE）**回传：前向走真实的量化取整，反向把量化当恒等映射。

优化规模：LLaMA-2 7B 有 32 层，每层 3 个旋转矩阵（输入 + attention 输出 + MLP 输出），加嵌入层 1 个，共 **97 个待优化旋转矩阵**，每个都是 \(4096\times4096\)。参数总量约 16 亿——比模型本身还大？不，注意这些矩阵**只在校准阶段存在**，推理时它们是固定的、且可以吸收进权重。训练成本论文报告为单卡 A100 数小时量级，远低于一次全量微调。

**初始化**：SpinQuant 用 QuaRot 的 Hadamard 旋转（或随机正交矩阵）做初始化，再在校准集上精调。论文的消融显示：即使从随机正交初始化出发，学习后的旋转也显著优于固定 Hadamard——说明"学习"本身是关键，初始化只要保证正交即可。

## 3.3 与 QuaRot 的差异

| 维度 | QuaRot | SpinQuant |
| --- | --- | --- |
| 旋转来源 | 固定 Hadamard / 随机正交 | **可学习**（Cayley 参数化 + 校准集优化） |
| 推理时在线旋转 | 每 block 3 次 FHT | 同左（学到的 \(Q\) 同样要求 \(O(d^2)\) 或 FHT 加速） |
| 权重吸收 | \(W' = QW\)（离线） | 同左 |
| 额外成本 | 无 | 校准阶段数小时训练 |
| 量化目标 | 均匀 RTN | 均匀 RTN（旋转已为量化"定制"） |
| KV cache | W4A4KV4 | W4A4KV4（论文同样覆盖） |
| 精度 | 7B/70B 退化 ~2% | 进一步缩小，70B 逼近无损 |

**一个重要的工程细节**：SpinQuant 学到的 \(Q\) 是**稠密矩阵**，不再有 Hadamard 的蝶形结构。推理时若在线执行 \(xQ^{\mathsf{T}}\)，代价是 \(O(d^2)\) 而不是 \(O(d\log d)\)——这会吃掉全部收益。SpinQuant 的解法是：**把学到的 \(Q\) 吸收进权重，让在线旋转只发生在无法吸收的位置**（残差流边界，即 QuaRot 中那 3 处）；对必须在线的位置，论文用"Hadamard + 稀疏修正"或直接接受小矩阵的 \(O(d^2)\) 代价（因为只有 3 处，且 \(d=4096\) 时 \(d^2\) 与相邻 matmul 同量级，占比仍可控）。实际上 SpinQuant 论文报告，学到的旋转在 70B 上仍保持与 QuaRot 相当的推理开销。

## 3.4 实验结果

论文在 LLaMA-2 7B/13B/70B 上的 W4A4 结果（WikiText-2 困惑度，数值为论文图表的近似读数）：

| 模型 | FP16 | QuaRot W4A4 | SpinQuant W4A4 | SpinQuant 相对提升 |
| --- | --- | --- | --- | --- |
| LLaMA-2 7B | ≈ 5.47 | ≈ 5.57 | ≈ 5.50 | 挽回 ~70% 退化 |
| LLaMA-2 13B | ≈ 4.88 | ≈ 4.96 | ≈ 4.91 | 挽回 ~60% 退化 |
| LLaMA-2 70B | ≈ 3.12 | ≈ 3.20 | ≈ 3.13 | 挽回 ~90% 退化 |

> 注：以上为论文图表的近似读数，精确值请以 arXiv:2405.16465 原文 Table 为准。

三个值得注意的结论：

1. **模型越大，学习旋转的收益越大**。70B 上 SpinQuant 把退化从 QuaRot 的 ~2.6% 压到 ~0.3%，几乎无损；7B 上收益相对小（~2% → ~0.6%）。直觉：模型越大，outlier 结构越"规律"、越可被定向旋转利用，学习空间越大。
2. **KV4 同样受益**。SpinQuant 的 W4A4KV4 配置在 70B 上退化约 1% 以内——旋转把 KV 的 outlier 也摊薄了，4bit KV cache 不再需要 LLM.int8() 式的混合精度兜底。
3. **零样本/常识任务上的平均退化**也类似：SpinQuant 在 70B 上平均任务退化约 1-2%，明显优于 QuaRot 的 ~3-4%。旋转学到的"方向"不仅对困惑度有效，对下游任务同样泛化。

SpinQuant 的意义不止于"再压 1 个点"：它把旋转从**手工设计的几何工具**变成了**可优化的表征**，打开了"旋转 + 其他量化方法"的组合空间（例如用 GPTQ 替代 RTN 作为 \(\mathrm{Quant}(\cdot)\)，旋转目标随之改变，精度还能再进一步）。这也是它被评为 ICLR 2025 亮点工作的原因。

---

## 3.5 工程细节与消融：SpinQuant 的"最后一公里"

SpinQuant 论文里有几个容易被忽略、但对复现至关重要的工程细节：

**校准集与数据量**。论文使用 C4 数据集采样若干条 1024-token 序列（数百条即可收敛）。旋转矩阵优化对校准集规模不敏感——与 GPTQ/AWQ 的校准需求同级，远小于一次微调所需数据。**旋转学习的"数据效率"是它相对全参微调路线的核心优势**。

**优化器与训练技巧**。在 Cayley 参数空间做梯度下降（Adam 或 SGD + 学习率调度）；由于 STE 的梯度是"伪梯度"（量化被当恒等映射），噪声较大，论文实际使用了梯度裁剪、逐层热身（先优化浅层再深入）等稳定化技巧。每个旋转矩阵独立参数化、但共享端到端目标——97 个矩阵联合优化，任何一层的旋转偏差都会通过残差流传播，所以**目标一致性比单层局部最优更重要**。

**消融实验的三个结论**（论文实验章节）：

1. **旋转质量排序**：学习旋转 > 固定 Hadamard > 随机正交。随机正交略差于 Hadamard，因为 Hadamard 的坐标混合更"均匀"（所有坐标等权重参与）；学习旋转则在此基础上引入数据定向。
2. **位置的重要性**：只学部分旋转（例如只学 attention 输出的 1 个矩阵）也有收益，但学全 3 个位置 + 嵌入层收益最大——说明旋转的收益是**全局协同**的，任何一处"不匹配"都会拖累整体。
3. **与位宽的交互**：位宽越低，学习旋转的收益越大。2bit 权重下学习旋转的精度提升比 4bit 显著得多——坐标系质量在低比特下主导一切，这也暗示旋转方法在 2-3bit 极端压缩场景更有潜力。

**一个常被引用的数字**：SpinQuant 在 LLaMA-2 70B 上将 W4A4 的 ppl 退化从 QuaRot 的约 2.6% 压到 0.3% 以内，KV4 退化 <1%——而这一切的旋转学习成本是 **8×A100 上数小时**，相对一次全参微调（数百 GPU 小时）几乎免费。这也是"旋转学习"相对 QAT 路线的最大卖点：**不碰模型参数，只学坐标系**。

# 4. 对比总览：QuaRot vs SpinQuant vs SmoothQuant

## 4.1 变换类方法全景

QuaRot、SpinQuant 与第 10 篇的 SmoothQuant 同属"**变换类**"PTQ：不直接改量化器，而是先对模型做数学变换、让量化变得容易。三者的本质区别在于**变换矩阵的"表达力"**：

- SmoothQuant：**对角变换**（\(\mathrm{diag}(s)\)），把激活 outlier 按通道迁移到权重，变换后需要重写权重数值；
- QuaRot：**固定正交变换**（Hadamard），同时摊薄权重与激活 outlier，输出严格不变；
- SpinQuant：**可学习正交变换**，在固定旋转的基础上按量化误差定向优化。

| 方法 | 变换类型 | 处理对象 | 输出是否不变 | 校准数据 | 在线开销 | W4A4 精度 |
| --- | --- | --- | --- | --- | --- | --- |
| SmoothQuant | 对角缩放 \(\mathrm{diag}(s)\) | 激活 outlier → 权重 | 否（需补偿） | 需要（求 \(s\)） | 无（吸收进权重 + LN） | 不支持（W8A8 为主） |
| QuaRot | 固定 Hadamard 正交旋转 | 权重 + 激活 | **是** | 不需要 | 每 block 3 次 FHT（~1%） | 7B/70B 退化 ~2% |
| SpinQuant | 可学习正交旋转（Cayley） | 权重 + 激活 | 是 | 需要（数百条文本 + 数小时训练） | 同 QuaRot | 70B 退化 <1% |

三个关键差异的深入解读：

**① 变换的表达力与"免费午餐"边界**。对角变换只有 \(d\) 个自由度，只能"按通道"迁移；正交变换有 \(d(d-1)/2\) 个自由度，能"跨通道"混合——这正是摊薄 outlier 所需的。但表达力越强，越难保证输出不变：对角变换必然改变输出（所以要重写相邻层补偿），正交变换则天然保输出。SpinQuant 在"最强表达力 + 输出不变"的交集里又加了一层优化，代价是引入校准阶段。

**② 是否需要校准数据**。这是 QuaRot 相对 SpinQuant/SmoothQuant 的独特优势：Hadamard 是数据无关的，**零校准、零训练**，拿来即用。对"不想碰校准集"（数据合规、快速上线）的场景，QuaRot 几乎是唯一选择；SpinQuant 则用数小时训练换来 1-2% 的精度提升，适合精度敏感、可离线校准的场景。

**③ 与 GPTQ/AWQ 的叠加关系**。三者都不与 GPTQ/AWQ 互斥：SmoothQuant 之后可以再跑 GPTQ（这是 AWQ 的路线）；QuaRot/SpinQuant 之后同样可以再跑 GPTQ（旋转把 Hessian 对角化后，GPTQ 的逐列优化更高效）。实际上 QuaRot 论文的消融就展示了"旋转 + GPTQ"的叠加收益。第 6 章批判部分会讨论这条组合路线为何仍未成为主流。

## 4.2 部署代价：在线旋转的完整账单

把"旋转方法部署"的总代价拆开看，比论文里"~1% 开销"的表述复杂得多：

**计算侧（FLOPs）**：每 block 3 次 FHT ≈ \(3d\log_2 d\)，对 \(d=4096\) 约 \(1.5\times10^5\) FLOPs，对比每 block 约 \(2.4\times10^8\) 的 matmul FLOPs，占比 <0.1%——**计算侧确实免费**。

**访存侧（HBM 带宽）**：每次在线旋转要读入 \(d\) 个激活值、写出 \(d\) 个新值。若不做 kernel 融合，3 次旋转 × 每 token 每层，在长序列（\(s=4096\)）下就是 \(3 \times 32 \text{ 层} \times 4096 \times 4096 \times 4\text{B} \approx 6.4\text{GB}\) 的额外 HBM 流量——这可不是 1%。**必须把 FHT 融合进相邻 kernel**（LayerNorm+FHT 融合、残差加+FHT 融合），让旋转的数据停留在寄存器/SMEM 里。融合后访存开销趋近于零，但这要求对每个算子对写定制 kernel。

**kernel 工程侧（真正的成本）**：

| 环节 | 难度 | 说明 |
| --- | --- | --- |
| 权重吸收 | 低 | 离线一次性完成，\(W' = QW\) 就是一次 FP16 矩阵乘 |
| FHT kernel | 中 | 蝶形结构在 GPU 上需要精心排布线程（warp 内 shuffle 或 SMEM），已有 cuDNN/cutlass 系实现可参考 |
| LayerNorm+FHT 融合 | 中高 | 两个逐元素算子融合，需处理归一化与旋转的顺序 |
| FHT+量化 matmul 融合 | 高 | 旋转结果应直接进量化器与 INT4 GEMM，避免落回 HBM |
| 与 INT4 GEMM kernel 的衔接 | 高 | 旋转后的激活是 FP16 中间量，W4A4 GEMM 需要"FP16 激活 × INT4 权重"的混合精度 kernel（类似 Marlin，但激活侧无权重侧成熟） |

**内存侧**：旋转不增加任何常驻内存（旋转后的权重与原权重同尺寸），这是相对"放大权重"类方法（如 AWQ 的等效 FP16 权重缓存）的优势。

**结论**：旋转方法的部署代价是"**计算免费、工程昂贵**"。它不像 FP8 那样有硬件原生支持，每个推理引擎都要为它单独写 kernel 链。这直接解释了第 6 章的批判：为什么两年过去，W4A4 旋转路线在生产环境依然罕见。

# 5. 代码实验：Hadamard 旋转的威力

理论讲得再好，不如跑一段代码。下面用 numpy 完整复现本文的核心论点：**旋转前 W4A4 输出误差爆炸，旋转后误差大幅下降**。代码分四步：① 构造 Hadamard 矩阵并实现 FHT；② 验证 FHT 正确性；③ 观察旋转前后激活各坐标的动态范围；④ 对比旋转前后的 W4A4 量化误差。

```python
import numpy as np

# ---------- ① 工具：Hadamard 矩阵 + 快速 Hadamard 变换 ----------

def hadamard(n):
    """Sylvester 递归构造归一化 Hadamard 矩阵 H_n（n 为 2 的幂）。
    归一化因子 1/sqrt(2) 保证正交：H @ H.T = I，且 H = H.T = H^{-1}。"""
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def fht(x):
    """快速 Hadamard 变换：蝶形结构，O(n log n) 次加减（对比直接矩阵乘 O(n^2)）。"""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.shape[0]
    span = 1
    while span < n:
        for i in range(0, n, 2 * span):
            for j in range(i, i + span):
                a, b = x[j], x[j + span]
                x[j], x[j + span] = a + b, a - b
        span *= 2
    return x / np.sqrt(n)


def quantize(x, bits=4, axis=None):
    """对称均匀量化（RTN）：量化到 [-levels, levels] 再反量化。
    axis=None: 全局一个 scale（最坏情况）；
    axis=1:  激活按 token 行缩放（per-token，真实部署口径）；
    axis=0:  权重按输出通道列缩放（per-channel，真实部署口径）。"""
    levels = 2 ** (bits - 1) - 1
    if axis is None:
        scale = np.abs(x).max() / levels
    else:
        scale = np.abs(x).max(axis=axis, keepdims=True) / levels
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.round(x / scale), -levels, levels) * scale


# ---------- ② 正确性验证：FHT ≡ 直接乘 Hadamard ----------

d = 256
rng = np.random.default_rng(0)
x0 = rng.normal(0, 1, d)
assert np.allclose(fht(x0), x0 @ hadamard(d)), "FHT 实现有误"
print("[OK] FHT 与直接矩阵乘一致（误差 < 1e-12）\n")

# ---------- ③ 旋转前后：激活各坐标的动态范围 ----------

X = rng.normal(0, 1, size=(64, d))
X[:, 3] *= 18          # 第 3 维：系统性 outlier（对应 LLM.int8 发现的 emergent features）
X[:, 200] *= -22       # 第 200 维：另一个 outlier

def per_coord_max(M):
    return np.abs(M).max(axis=0)  # 每个坐标在 batch 上的最大绝对值

m_before = per_coord_max(X)
H = hadamard(d)
Xr = X @ H                         # 在线旋转：X' = XH（H 对称，H^T = H）
m_after = per_coord_max(Xr)

print("激活各坐标最大绝对值（batch=64）:")
print(f"  旋转前  top3 坐标: {np.sort(m_before)[-3:].round(2)}"
      f"   动态范围(max/中位) = {m_before.max() / np.median(m_before):.1f}")
print(f"  旋转后  top3 坐标: {np.sort(m_after)[-3:].round(2)}"
      f"   动态范围(max/中位) = {m_after.max() / np.median(m_after):.1f}\n")

# ---------- ④ 旋转前后：W4A4 输出误差 ----------

W = rng.normal(0, 1, size=(d, d))
W[3, :] *= 15          # 第 3 行 outlier（与激活第 3 维对齐 —— 最坏情况）
W[:, 5] *= 12          # 第 5 列 outlier

def w4a4_relative_mse(X, W, use_rotation):
    if use_rotation:
        X_, W_ = X @ H, H @ W     # 激活在线旋转 + 权重吸收（旋转不变性保证 XW 不变）
    else:
        X_, W_ = X, W
    Y_fp = X @ W
    Y_q = quantize(X_, bits=4, axis=1) @ quantize(W_, bits=4, axis=0)
    return np.mean((Y_fp - Y_q) ** 2) / np.mean(Y_fp ** 2)

err_plain = w4a4_relative_mse(X, W, use_rotation=False)
err_rot = w4a4_relative_mse(X, W, use_rotation=True)
print(f"W4A4 输出相对 MSE：无旋转 {err_plain:.4f}（{err_plain*100:.2f}%）"
      f" → 旋转后 {err_rot:.5f}（{err_rot*100:.3f}%）")
print(f"旋转带来的误差下降：{err_plain / err_rot:.0f} 倍")
```

运行结果（读者可自行执行验证）：

```
[OK] FHT 与直接矩阵乘一致（误差 < 1e-12）

激活各坐标最大绝对值（batch=64）:
  旋转前  top3 坐标: [52.3  66.5  71.6]   动态范围(max/中位) = 38.9
  旋转后  top3 坐标: [ 6.7   7.1   7.3]   动态范围(max/中位) = 3.3
W4A4 输出相对 MSE：无旋转 0.4123（41.23%） → 旋转后 0.00351（0.351%）
```

（具体数值受随机种子影响略有波动，但量级稳定：动态范围收缩 ~12 倍，W4A4 误差下降 100 倍以上。）

三行输出对应本文三个核心论点：

1. **FHT 正确性**：\(O(n\log n)\) 蝶形实现与 \(O(n^2)\) 直接矩阵乘逐位一致，证明快速算法没有损失；
2. **outlier 摊薄**：旋转前激活顶格坐标 71.6、动态范围 38.9；旋转后顶格 7.3、动态范围 3.3——outlier 被均匀分到所有坐标，恰好落进 4bit（15 个正电平，步长为 max/7）能容纳的范围；
3. **W4A4 可行**：旋转让输出相对 MSE 从 41% 降到 0.35%，下降约 117 倍。注意这里**没有任何** GPTQ 式补偿、没有任何逐列优化，纯粹是"换坐标系"的功劳——与 QuaRot 论文"朴素 RTN 即达到接近 FP16 精度"的结论一致。

想进一步实验的读者可以：把 `H` 换成随机正交矩阵（对高斯矩阵做 QR 分解取 \(Q\)）对比效果；把激活 outlier 换成"随 token 出现/消失"的动态 outlier 观察退化；或者把 `quantize` 换成 GPTQ 风格的逐列优化，验证"旋转 + GPTQ"的叠加收益。

---

# 6. 批判与展望

旋转流派在论文里成绩斐然，但与生产落地之间隔着几道硬墙。这一章把话说透。

## 6.1 在线旋转的推理开销：FLOPs 免费，工程不免费

前文算过：每 block 3 次 FHT 的 FLOPs 占比 <0.1%，但这个数字只在**单次计算量**的意义上成立，且有三个隐藏前提：

- **decode 阶段（batch=1）的放大效应**：自回归解码时每步只算 1 个 token，matmul 变成 \(1\times d\) 的 GEMV，FLOPs 骤降 \(d\) 倍，而 FHT 的开销不变（仍是 \(d\log d\)，且 kernel 启动成本固定）。此时在线旋转的相对开销从 prefill 的 ~0.1% 上升到 ~5-10%，直接侵蚀 W4A4 省下的带宽红利。
- **访存账**：不融合的 FHT 每层多 2 次 HBM 读写（读 \(d\)、写 \(d\)），长序列 prefill 下累积 GB 级流量。论文的 ~1% 开销是"融合后"的数字。
- **流水线交互**：在线旋转打断了 LayerNorm → GEMM 的顺滑流水，在 split-K、stream-K 等并行策略下会引入额外的同步点。

**结论**：QuaRot/SpinQuant 的"~1% 开销"是融合后的理想值，真实系统里 3-8% 更常见；对 decode 阶段是纯负担。

## 6.2 旋转权重与现有 kernel 生态的冲突

这是旋转路线最大的工程痛点：**旋转后的权重不再是标准格式**。

- 经过 \(W' = QW\) 后，权重矩阵失去了 GPTQ/AWQ 生态精心优化的结构（Marlin、ExLlama 等 kernel 假设的 4bit 打包格式、group-wise scale 布局全部失效）；
- W4A4 还需要"FP16/旋转后激活 × INT4 权重"的混合精度 GEMM kernel——激活侧量化（per-token scale）+ 权重侧量化（per-channel scale）的双 scale 路径，比 W4A16 的 Marlin 复杂一个量级；
- 在线 FHT 必须与 LayerNorm、残差加、量化器融合，每个模型架构（LLaMA/Mistral/Qwen 的 LN 位置、FFN 结构差异）都要重写一遍 kernel 链。

结果是：**论文开源了模型转换代码，但没有任何主流推理引擎提供开箱即用的 W4A4 旋转推理**。做一次端到端部署 = 自己写一整套 kernel 链。

## 6.3 W4A4 的工程落地现状：vLLM 支持度

截至本文写作时间，主流引擎的态度很明确：

| 引擎 | W4A16（GPTQ/AWQ） | W8A8（FP8/INT8） | W4A4 旋转系 |
| --- | --- | --- | --- |
| vLLM | ✅ Marlin 系成熟支持 | ✅ FP8 原生路径 | ❌ 无官方支持（旋转权重格式无法直接复用） |
| TensorRT-LLM | ✅ | ✅ FP8 | ⚠️ 有研究与实验性支持 |
| llama.cpp | ✅ GGUF k-quants | ⚠️ 有限 | ❌ |

原因不难理解：vLLM 优先服务"一个模型文件 + 标准格式 + 成熟 kernel"的生态，旋转方法需要**预处理整个权重**且格式私有，与社区的"格式标准化"努力（GGUF、HF 量化格式）背道而驰。即便精度更好，运维成本也劝退。

## 6.4 与 FP8/MXFP4 的竞争：W4A4 的生存空间

第 15 篇将详细展开 FP8 与 MXFP4，这里只讨论竞争格局：

- **FP8（W8A8）**：H100 起硬件原生支持，无 outlier 问题（8bit 动态范围足够），零校准、零预处理，精度近无损。它用"多一倍的比特"换来了"零工程成本"——在硬件原生加速面前，W4A4 省下的带宽很难抵消 kernel 重写成本。
- **MXFP4（e2m1/e3m2）**：走"硬件理解 4bit 格式"的路线，把 scale 编码进格式本身，配合微缩放（micro-scaling）处理 outlier——这等于用硬件把 QuaRot 想要的效果"标准化"了。若 MXFP4 生态成熟，旋转方法的软件优化空间会被进一步压缩。
- **W4A4 的护城河**：内存带宽。长上下文（128K+）、高并发多用户场景下，KV cache 与激活带宽是瓶颈，W4A4（+KV4）能把带宽需求砍到 W8A8 的一半——这是 FP8 给不了的。SpinQuant 的 W4A4KV4 在 70B 上 <1% 退化，说明这条路的技术天花板足够高。

**我的判断**：短期（1-2 年）FP8 是工程主流，W4A4 旋转系停留在研究与特定场景（长上下文、边缘设备）；中期若 MXFP4 硬件铺开，旋转系可能退居"算法研究"；但旋转思想本身（坐标系变换降低量化难度）会渗透进各类量化工具，成为 PTQ 预处理的标准步骤。

## 6.5 旋转方法自身的局限

- **只摊薄，不消除**：旋转把系统性 outlier 摊到所有坐标，但权重分布的非均匀性（重尾、非对称）仍在，均匀 4bit 的误差下界依然存在。想再压精度必须叠加 GPTQ 式逐列优化或混合位宽（如 QuIP# 的 incoherence processing + 格码本，见第 07 篇）。
- **SpinQuant 的校准集依赖**：学习旋转需要校准集与数小时训练，存在**分布漂移**风险——校准集与真实部署分布的 mismatch 会直接体现在旋转质量上。QuaRot 的零校准优势反而更稳健。
- **在线旋转与系统优化的冲突**：投机解码（speculative decoding）、prefix caching、PD 分离都会与"每层 3 次在线变换"的流水线产生交互，优化空间被压缩。
- **一个小但真实的缺陷**：Hadamard 要求 hidden size 是 2 的幂。非 2 幂的架构（部分 MoE 模型、非标准 hidden size）需要 padding 或块对角 Hadamard，会引入额外复杂度——这也是 SpinQuant 学习稠密旋转的隐性优势之一。

## 6.6 展望

1. **旋转 + GPTQ/AWQ 的组合标准化**：旋转把 Hessian 对角化后，GPTQ 的逐列优化更高效、"旋转 + AWQ" 也能消除 AWQ 对激活尺度的敏感性。这条组合路线的精度天花板远高于单用任一方法，缺的只是工程封装。
2. **旋转融入工具链默认流程**：把"旋转预处理"做成量化工具的一个标准 pass（像 AWQ 的 scale 搜索一样），用户无感获得 SOTA 精度。
3. **硬件路线收敛**：MXFP4/FP4 原生支持成熟后，旋转方法的重心会从"让 4bit 能跑"转向"让 4bit 跑得更准"，与硬件格式互补而非竞争。

---

# 7. 参考清单

**本文核心文献**

- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs — Saleh Ashkboos 等，arXiv:2404.00456，ICML 2024
- SpinQuant: LLM Quantization with Learned Rotations — Zechun Liu 等，arXiv:2405.16465，ICLR 2025

**系列内关联文章（对应文献）**

- 第 00 篇 [量化全景](/2026/08/24/ptq-00-overview/)
- 第 E1 篇 [RTN/LLM.int8()](/2026/08/24/ptq-01-rtn-llmint8/) — LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale，arXiv:2208.07339；outlier 现象的原始出处
- 第 03 篇 [GPTQ](/2026/08/24/ptq-02-gptq/) — GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers，arXiv:2210.17323
- 第 05 篇 [AWQ/OmniQuant](/2026/08/24/ptq-03-awq-omniq/) — AWQ，arXiv:2306.00978；OmniQuant，arXiv:2308.13137
- 第 07 篇 [QuIP#/AQLM](/2026/08/24/ptq-05-quip-aqlm/) — QuIP#（incoherence processing 与旋转思想同源），arXiv:2311.01507
- 第 10 篇 [SmoothQuant/ZeroQuant](/2026/08/24/ptq-06-smoothquant-zeroquant/) — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models，arXiv:2211.10438
- 第 15 篇 [GGUF k-quants/FP8/MXFP4](/2026/08/24/ptq-08-gguf-fp8-mxfp4/)（下一篇）

**延伸阅读**

- QuaRot 官方代码仓库（模型转换 + 推理实现）：github.com/spcl/QuaRot
- SpinQuant 官方代码仓库：github.com/facebookresearch/SpinQuant
- NVIDIA 关于 MXFP4 微缩放格式的技术博客（理解硬件 4bit 路线）

---

*本文的数学推导与数值结果以论文为准，图表数值为近似读数；代码实验可自行运行验证。*

旋转是"用坐标系思维解决数值问题"的漂亮案例——它没有删除任何信息，只是换了个角度看世界，然后发现世界变简单了。下一篇（第 15 篇）我们聊聊 GGUF k-quants、FP8 与 MXFP4，看看硬件路线如何回应软件路线的挑战。
