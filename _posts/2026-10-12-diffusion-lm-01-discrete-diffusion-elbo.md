---
title: "扩散语言模型（01）：离散扩散的 ELBO——为什么「掩码 + 1/t 权重」就够用了"
date: 2026-10-12 12:30:00 +0800
categories:
  - 生成模型
tags: [diffusion-llm, discrete-diffusion, masked-diffusion, elbo, d3pm, ctmc, score-entropy, mdlm, llada, tau-leaping]
layout: post
mathjax: true
excerpt: "文本上的扩散不能照搬图像：离散 token 加不了高斯噪声，也没法定义 score。本文从 D3PM 的转移矩阵推到连续时间 CTMC，再把吸收态下的变分下界一步步化简成「带 1/t 权重的掩码交叉熵」，并用 numpy 实跑验证三件事：前向核/反向核的表达式、1/t 是损失成为 NLL 上界的必要条件（去掉它损失会掉到 NLL 之下 1.74）、以及固定 15% 掩码率训练的模型在 20 万样本里一次都没见到全掩码上下文。"
---

> **系列导航** ｜ 本系列承接 [Flow Matching 建模之三：条件生成、离散域与潜空间](/2026/08/22/flow-matching-03-guidance-discrete-latent/) 里的离散域部分，把它展开成完整的一条线。
> 上一篇：无（系列首篇）｜ 下一篇：02 块扩散与 KV Cache（写作中）

> **TL;DR**
>
> * 文本的“噪声”必须是离散状态转移，主流是**吸收态掩码**：token 被换成 `[MASK]` 且永不自愈。
> * 吸收态下，连续时间变分下界化简为**掩码位置上的交叉熵乘以 $$1/t$$**——这就是 LLaDA / Dream 训练损失的全部理论依据。
> * 这个权重不是技巧：实跑显示带 $$1/t$$ 的损失等于 $$-\mathbb{E}[\log p_\theta(x_0)]$$（间隙 0.0000），去掉后掉到 NLL **之下 1.7382**，不再构成上界。
> * 掩码率必须随机（$$t\sim\mathcal{U}[0,1]$$）。固定 15% 训练时，全掩码上下文在 20 万样本中被观测 **0 次**，$$t=1$$ 处的交叉熵退化成 $$\ln 4$$，生成质量比随机掩码差 2.33 倍。
> * 并行解码的代价来自 $$\tau$$-leaping 的条件独立假设：两个相关性 0.9 的 token 被一步同时解出时，一致概率从 0.90 掉到 0.50。

配套实验：[`ipynbs` commit `c90dcf8`](https://github.com/lrypcy/ipynbs/tree/c90dcf8/experiments/diffusion-lm/) 下的 `elbo_masked_diffusion/` 与 `parallel_decoding_independence/`，纯 numpy、CPU 可跑，本文所有数字都来自这两份脚本的实际输出。

---

## 0. 符号约定

| 符号 | 含义 | 备注 |
|---:|:---|:---|
| $$\mathcal{V},\;V$$ | 词表及其大小，含特殊符号 $$\mathrm{M}=$$`[MASK]` | 掩码符号的 id 通常取 $$V$$ 或 $$V+1$$ |
| $$L$$ | 序列长度 | 朴素掩码扩散下是固定的 |
| $$x_0$$ | 干净序列 $$(x_0^1,\ldots,x_0^L)$$ | 下标是时间，上标是位置 |
| $$z_t$$ / $$x_t$$ | $$t$$ 时刻的加噪序列 | 本文统一写 $$z_t$$ |
| $$\alpha_t$$ | **未被掩码**的概率，$$\alpha_0=1$$、$$\alpha_1=0$$ | 扮演连续扩散中信噪比的角色 |
| $$t$$ | 连续时间，$$t\sim\mathcal{U}[0,1]$$ | 也是“掩码率” |
| $$\mathbf{Q}_t,\mathbf{R}_t$$ | 离散时间转移矩阵 / 连续时间速率矩阵 | $$\mathbf{P}_t=\exp(t\mathbf{R})$$（齐次时） |
| $$\mathcal{M}$$ | 被掩码位置集合 | 损失只对它求和 |
| $$p_\theta(\cdot\mid z_t)$$ | 掩码预测器输出的逐位置分布 | 双向 Transformer，**无 causal mask** |

> **读原文前先确认记号**：LLaDA 用 $$t$$ 表示**掩码概率**（权重写成 $$1/t$$）；MDLM 与 Kuleshov 组讲义用 $$\alpha_t$$ 表示**保留概率**（权重写成 $$-\alpha'_t/(1-\alpha_t)$$）。两者在 $$\alpha_t=1-t$$ 下等价，混用会算错分母。

---

## 1. 为什么连续扩散不能照搬到文本

图像扩散的成功建立在两个前提上，而这两个前提在文本上都不成立。

**第一，加噪本身要有意义。** 连续扩散的前向过程是 $$q(x_t\mid x_0)=\mathcal{N}(x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)\mathbf{I})$$ [1](https://arxiv.org/abs/2006.11239)：给干净样本加一点高斯噪声，样本仍然“像”原图，只是更模糊。文本没有这个性质——对离散 token 施加 $$\epsilon$$ 级别的高斯扰动，量化回词表之后要么还是同一个 token（没有污染），要么跳到一个语义完全无关的 token（污染过度）。中间不存在“轻微破坏但可恢复”的状态。

**第二，score 要能定义。** 连续扩散的去噪目标等价于估计 $$\nabla_x\log p_t(x)$$；词表是有限集，这个梯度无处可求 [2](https://ar5iv.arxiv.org/html/2506.13759)。

能继承的只有**思路**：定义一条可控的破坏过程，学它的逆，用变分下界当训练目标。下面四节就是把这条思路在离散状态上重新实现一遍。

---

## 2. 离散状态上的噪声：D3PM 的转移矩阵

D3PM 把前向过程写成离散状态上的马尔可夫链，每一步作用一个转移矩阵 $$\mathbf{Q}_t$$，边际分布由累积乘积给出 [3](https://aiwiki.ai/wiki/discrete_diffusion)：

$$
q(x_t\mid x_0)=\mathrm{Cat}\left(x_t;\;x_0\,\bar{\mathbf{Q}}_t\right),\qquad \bar{\mathbf{Q}}_t=\mathbf{Q}_1\mathbf{Q}_2\cdots\mathbf{Q}_t
$$

$$\mathbf{Q}_t$$ 的选法决定“噪声”的性质：

| 转移类型 | 行为 | 特点 |
|---|---|---|
| 均匀（uniform） | 以某概率换成词表内均匀随机的 token | 破坏最彻底；被污染的 token 偶尔恰好是正确 token，仍保留部分信息 |
| **吸收态（absorbing）** | 以某概率换成 `[MASK]`，`[MASK]` 永不还原 | 与掩码语言模型同构，2024 年之后的主流 |
| 嵌入近邻 | 按学习到的嵌入相似度换成相邻 token | 破坏温和，实现复杂 |

吸收态的关键性质是**单调性**：一旦戴上 `[MASK]` 就不会自己摘下来。于是 $$t=1$$ 时整条序列全是 `[MASK]`，反向过程就是“从全掩码画布反复填空”;反过来，若已知真实序列 $$x$$，从 $$z_t$$ 还原到 $$z_s\;(s<t)$$ 只需要处理那些被掩码的位置。

D3PM 还指出了一个容易被忽略的事实：**自回归模型是这条谱系的一个端点**——当掩码时刻被确定性地安排成从左到右时，吸收态扩散就退化成自回归 [3](https://aiwiki.ai/wiki/discrete_diffusion)。自回归、掩码语言模型、扩散，三者坐在同一条谱系上，差别只在“破坏顺序怎么安排”。

---

## 3. 连续时间：把求和换成积分

把时间步长趋于零，前向过程成为连续时间马尔可夫链（CTMC），由速率矩阵 $$\mathbf{R}_t\in\mathbb{R}^{V\times V}$$ 驱动：非对角元非负、行和为零；在无穷小区间 $$[t,t+\mathrm{d}t)$$ 内从状态 $$j$$ 跳到 $$k\neq j$$ 的概率是 $$[\mathbf{R}_t]_{jk}\,\mathrm{d}t$$。转移概率满足 Kolmogorov 前向方程 [4](https://arxiv.org/html/2607.13431v1)：

$$
\frac{\mathrm{d}}{\mathrm{d}t}\mathbf{P}_t=\mathbf{P}_t\mathbf{R}_t,\qquad \mathbf{P}_0=\mathbf{I}
$$

时间齐次时 $$\mathbf{P}_t=\exp(t\mathbf{R})$$。换成连续时间有两个实打实的好处：

1. **下界写成积分而不是求和**，表达式更简洁，并且能看清一件事：目标只依赖时间表的端点，不依赖它的函数形式（MDLM 与 MD4 都指出了这一点）[3](https://aiwiki.ai/wiki/discrete_diffusion)。
2. **训练与推理的粒度解耦**。模型在连续时间上训练，推理时可以任意选择 $$[0,1]$$ 的离散化份数 $$T'$$，不必等于训练时的 $$T$$——这就是“步数是自由旋钮”的来源，也是 LLaDA 能用更少步数换速度的理论基础 [5](https://arxiv.org/html/2502.09992)。

反向采样要模拟反向 CTMC。精确模拟代价高，实践用 $$\tau$$-leaping：一个时间步内允许多个位置同时跳变，并假设这些跳变**条件独立**。第 9 节会量化这个假设的代价。

---

## 4. 吸收态的前向核与反向核

令 $$\alpha_t$$ 表示“token 仍未被掩码”的概率，单调递减，$$\alpha_0=1$$、$$\alpha_1=0$$。它在离散扩散里扮演的正是连续扩散中信噪比的角色 [6](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/)。

对 $$s<t$$，前向核定义为：把 $$z_s$$ 中**尚未被掩码**的每个 token 以概率 $$(\alpha_s-\alpha_t)/\alpha_s$$ 换成 `[MASK]`，已被掩码的保持不动：

$$
q(z_t^i\mid z_s^i)=
\begin{cases}
1, & z_s^i=\mathrm{M},\; z_t^i=\mathrm{M}\\[4pt]
\dfrac{\alpha_t}{\alpha_s}, & z_s^i\neq\mathrm{M},\; z_t^i=z_s^i\\[6pt]
\dfrac{\alpha_s-\alpha_t}{\alpha_s}, & z_s^i\neq\mathrm{M},\; z_t^i=\mathrm{M}\\[6pt]
0, & \text{其他}
\end{cases}
$$

第二项的 $$\alpha_t/\alpha_s$$ 保证“到 $$t$$ 时刻仍未掩码”的边缘概率恰为 $$\alpha_t$$——可以验算：$$\alpha_s\cdot(\alpha_t/\alpha_s)=\alpha_t$$。

若已知真实序列 $$x$$，最优反向核是：把 $$z_t$$ 中的掩码位置以概率 $$(\alpha_s-\alpha_t)/(1-\alpha_t)$$ 换成 $$x$$ 中对应的值，否则保留掩码 [6](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/)。**分母是 $$1-\alpha_t$$ 而不是 $$\alpha_t$$**，因为条件概率必须归一化在“$$t$$ 时刻处于掩码状态”这一事件上——这是全文最容易写错分母的地方。

> **直觉**：给每个位置分配一个“掩码时刻” $$\tau\sim\mathcal{U}[0,1]$$，$$t$$ 时刻被掩码等价于 $$\tau\le t$$。于是
> $$\mathrm{P}(\text{在 } t \text{ 被掩码}\mid \text{在 } s \text{ 未掩码})=\mathrm{P}(\tau\le t\mid \tau>s)=\dfrac{t-s}{1-s}$$，
> $$\mathrm{P}(\text{在 } s \text{ 未掩码}\mid \text{在 } t \text{ 被掩码})=\mathrm{P}(\tau>s\mid \tau\le t)=\dfrac{t-s}{t}$$。
> 取 $$\alpha_t=1-t$$ 后，这两式正是上面的两个转移概率。

**数值验证**（`elbo_masked_diffusion/run.py` 的 A 组，$$s=0.3$$、$$t=0.7$$、40 万次采样）：

| 量 | 经验值 | 理论值 | 偏差 |
|---|---|---|---|
| 前向 $$\mathrm{P}(\text{mask}@t\mid \text{unmask}@s)$$ | 0.57067 | 0.57143 | 7.6e-4 |
| 反向 $$\mathrm{P}(\text{unmask}@s\mid \text{mask}@t)$$ | 0.57129 | 0.57143 | 1.4e-4 |

---

## 5. 连续时间 ELBO 的化简：只剩“掩码位置的加权交叉熵”

把连续时间 ELBO 写开，单项贡献形如“权重 $$\times$$ 逐位置对数概率”，权重为 $$-\alpha'_t/(1-\alpha_t)$$。取线性表 $$\alpha_t=1-t$$（于是 $$\alpha'_t=-1$$、$$1-\alpha_t=t$$）[6](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/)：

$$
\mathcal{L}_{\mathrm{ELBO}}=\mathbb{E}_{t\sim\mathcal{U}[0,1]}\;\mathbb{E}_{q(z_t\mid x)}\left[\frac{1}{t}\,\log p_\theta(x\mid z_t)\right]
$$

再往前走两步就到终点：

**第一步，只有掩码位置需要预测。** 吸收态下未被掩码的位置满足 $$z_t^i=x^i$$，它的对数概率是常数，不产生学习信号。

**第二步，因子化。** 掩码预测器一次前向输出所有位置的分布，训练时按 $$p_\theta(x\mid z_t)=\prod_{i\in\mathcal{M}}p_\theta(x^i\mid z_t)$$ 分解（$$\mathcal{M}$$ 为被掩码位置集合）。于是对数概率变成对被掩码位置求和，得到 LLaDA 采用的形式 [5](https://arxiv.org/html/2502.09992)：

$$
\mathcal{L}(\theta)\triangleq-\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}[x_t^i=\mathrm{M}]\;\log p_\theta\!\left(x_0^i\mid x_t\right)\right]
$$

**$$1/t$$ 的解释是归一化**：$$t$$ 是被掩码 token 的期望占比，除以它才不会让高掩码率的样本在损失里权重过小 [6](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/)。第 6 节会看到，它还有一个更硬的作用。

### 5.1 权重是怎么来的：把下界写成积分再代入吸收态

上面的 $$1/t$$ 是“$$\alpha_t=1-t$$”这一个特定时间表下的结果。完整形式值得自己推一遍，因为它是判断“换时间表要不要改权重”的唯一依据。

**第一步，离散时间的下界求和。** D3PM 的变分下界由逐时间步的 KL 项与最后一步的熵项组成；把它写成对时间步求和，每一项的权重由该步的转移矩阵决定 [3](https://aiwiki.ai/wiki/discrete_diffusion)。

**第二步，取极限变成积分。** 时间步长趋于零后，求和变成对 $$t$$ 的积分。对一般的转移核，被积函数的形状是“该时刻发生跳变的速率 $$\times$$ 跳变位置的负对数概率”，速率项正是 $$\mathbf{R}_t$$ 的非对角元之和。

**第三步，代入吸收态。** 吸收态下跳变只有一种：未掩码 $$\to\mathrm{M}$$。它的速率由 $$\alpha_t$$ 决定——单位时间内“仍未掩码”的比例变化是 $$-\alpha'_t$$，而此时处于掩码状态的比例是 $$1-\alpha_t$$，于是权重为

$$w(t)=\frac{-\alpha'_t}{1-\alpha_t}$

取 $$\alpha_t=1-t$$（$$\alpha'_t=-1$$）即得 $$w(t)=1/t$$。

**由此得到两条实用结论：**

1. **权重只依赖时间表的端点与单调性，不依赖函数形式。** 换成正弦表或余弦表时，只要仍满足 $$\alpha_0=1,\alpha_1=0$$ 且单调，带 $$-\alpha'_t/(1-\alpha_t)$$ 权重的损失仍是同一个量的估计——MDLM 与 MD4 都把这一点列为吸收态框架的主要简化 [3](https://aiwiki.ai/wiki/discrete_diffusion)。动手练习 ① 就是让你亲自验证这件事。
2. **不要在代码里硬编码 $$1/t$$。** 一旦换时间表（不少实践用非线性表以改善低掩码率处的训练信号），硬编码的 $$1/t$$ 会静默变成错误的目标。

至此，训练代码与一个掩码语言模型几乎没有区别——差别只在掩码率的采样方式与那个权重。LLaDA 官方也明确指出：如果你有一套自回归的训练代码，改成 LLaDA 只需动几行 [7](https://github.com/ML-GSAI/LLaDA)。

---

## 6. 它就是 NLL 的上界：实跑验证

第二条式子必须和第一条一起记住 [5](https://arxiv.org/html/2502.09992)：

$$
-\mathbb{E}_{p_{\mathrm{data}}(x_0)}\left[\log p_\theta(x_0)\right]\le\mathcal{L}(\theta)
$$

第一条让训练可行，第二条让"它是一个生成模型"成立——dLLM 因此有似然可谈，可以与自回归在同一把尺子上比困惑度。

### 6.1 实验设计

在一个能精确枚举的小问题上（$$V=4$$、$$L=3$$，64 条序列、125 个上下文状态）做三件事：

1. 用**真实后验收敛器** $$p_\theta(x^i=v\mid z_t)\propto\sum_{x:\,x \text{ 与 } z_t \text{ 相容},\,x^i=v}p_{\mathrm{data}}(x)$$，按第 4 节的反向核传播状态分布，精确算出反向过程诱导的模型分布 $$p_\theta(x_0)$$，从而得到真实的 $$-\mathbb{E}[\log p_\theta(x_0)]$$；
2. 用数值积分算出带 $$1/t$$ 的损失；
3. 再算一遍去掉 $$1/t$$ 的版本。

### 6.2 结果

| 量 | 数值 | 与 NLL 的关系 |
|---|---|---|
| $$-\mathbb{E}[\log p_\theta(x_0)]$$ | 3.6602 | — |
| 带 $$1/t$$ 的损失 | 3.6602 | 差 $$+0.0000$$，**是上界** |
| 去掉 $$1/t$$ 的损失 | 1.9220 | 差 $$-1.7382$$，**不是上界** |

用真实后验收敛器时，连续时间变分间隙为 0，所以带 $$1/t$$ 的损失与 NLL **精确相等**——这是对第 5 节推导的强验证。而去掉 $$1/t$$ 之后，损失掉到 NLL 之下 1.74：它不再界定任何东西，只是一个"看起来也在下降"的交叉熵。

> **为什么去掉 $$1/t$$ 就会变小？**
> 损失里 $$\sum_i\mathbf{1}[x_t^i=\mathrm{M}]$$ 这一项的期望是 $$tL$$。不除以 $$t$$ 时，低掩码率样本对损失的贡献被系统性压低（$$t\to0$$ 时几乎没有掩码位置，损失趋于 0），整个期望被拉到 NLL 之下。第 7 节的区间贡献数据也印证了这一点：不带权重时，掩码率 $$[0.75,1)$$ 区间独占 45.94% 的贡献，而带上权重后四个区间大致均衡（21.89% / 24.34% / 26.22% / 27.54%）。

代价是方差：同样 16 个样本的蒙特卡洛估计，带 $$1/t$$ 的方差是 1.2387，不带是 0.1683（约 7.4 倍）。这也解释了为什么实践中要靠大批次与更多训练 token 去摊——$$1/t$$ 换来的是**正确性**，不是**稳定性**。

### 6.3 推导自查清单

读完前六节，下面每一条都应当能独立推出或证伪：

- [ ] 前向核第二项为什么是 $$\alpha_t/\alpha_s$$（提示：验算边缘概率）；
- [ ] 反向核的分母为什么是 $$1-\alpha_t$$ 而不是 $$\alpha_t$$；
- [ ] $$-\alpha'_t/(1-\alpha_t)$$ 在 $$\alpha_t=1-t$$ 下退化为 $$1/t$$；
- [ ] 为什么吸收态使得"只需在掩码位置算交叉熵"；
- [ ] 为什么去掉 $$1/t$$ 之后损失会掉到 NLL 之下（提示：$$\sum_i\mathbf{1}[x_t^i=\mathrm{M}]$$ 的期望是 $$tL$$）；
- [ ] 用真实后验收敛器时，变分间隙为什么是 0。

---

## 7. 与 BERT 的两处差别，两处都致命

掩码扩散的损失看起来就是 BERT 的 MLM 损失，但有两处不同，而这两处正是"能不能生成"的分水岭 [5](https://arxiv.org/html/2502.09992)：

| | BERT / MLM | 掩码扩散 |
|---|---|---|
| 掩码率 | 固定约 15% | $$t\sim\mathcal{U}[0,1]$$，每条样本独立采样 |
| 损失权重 | 无（或常数） | $$1/t$$ |
| 是否生成模型 | 否，无似然 | 是，损失是 NLL 上界 |
| 能否从全 `[MASK]` 生成 | 不能 | 能 |

### 7.1 掩码率：一个几乎不可能被观测到的上下文

在一个 $$L=6$$、$$V=4$$ 的一阶马尔可夫链上，用**查表式 MLE**（不做权值共享，没见过的上下文只能回退到均匀分布）训两个预测器，各喂 20 万条样本：

| 训练掩码率 | 全掩码上下文被观测次数 | CE@0.15 | CE@0.50 | CE@1.00 | 生成序列的 $$-\log p$$ |
|---|---|---|---|---|---|
| 固定 15% | **0 次** / 200000 | 1.1935 | 1.3236 | **1.3863** | 8.3927 |
| $$t\sim\mathcal{U}[0,1]$$ | 28399 次 / 200000 | 1.2126 | 1.2171 | **1.2929** | 7.6964 |

参考值：真实分布的熵 $$H=7.1709$$，均匀分布基线 $$-\log p=8.3178$$。

数字非常直白：固定 15% 时，全掩码上下文的出现概率是 $$0.15^6\approx1.1\times10^{-5}$$，20 万样本里**一次都没出现**；于是 $$t=1$$ 处的预测只能回退到均匀分布，交叉熵正好等于 $$\ln 4=1.3863$$。生成质量上，固定 15% 的样本超出熵 $$+1.2219$$，随机掩码只超出 $$+0.5256$$——**2.33 倍**的差距，且前者几乎贴着均匀分布的基线（8.39 vs 8.32）。

机制清楚了：掩码率随机化让模型见过"从几乎全掩码到几乎不掩码"的全部状态，这是能从全 `[MASK]` 出发生成整句的前提。LLaDA 论文也强调，这个看似细微的差别在大规模下影响显著 [5](https://arxiv.org/html/2502.09992)。

> **一个诚实的补充**：早期用带权值共享的线性模型做同款实验时，两个方案的差距只有 1.005 倍——因为线性模型会把"邻居被掩码"这一特征外推到全掩码上下文，掩盖了数据覆盖率的差距。真实 Transformer 不具备这种线性外推能力，所以查表式实验才是这里该用的对照。

---

## 8. 一般转移怎么办：concrete score 与 score entropy

第 5 节的化简依赖吸收态。换成均匀转移或一般转移，"只需在掩码位置算交叉熵"不再成立，需要直接处理**比值**。

**concrete score** 把连续扩散里的 $$\nabla\log p$$ 换成离散情形下的分布比值 [2](https://ar5iv.arxiv.org/html/2506.13759)：

$$
s_\theta(x,t)_y\approx\frac{p_t(y)}{p_t(x)}
$$

**score entropy** 是可 tractable 的 Bregman 散度目标，只需在观测到的转移上计算：

$$
\mathcal{L}_{\mathrm{SE}}(\theta)=\mathbb{E}_{x_t\sim p_t}\left[\sum_{y\neq x_t}\left(s_\theta(x_t,t)_y-\frac{p_t(y)}{p_t(x_t)}\log s_\theta(x_t,t)_y\right)\right]
$$

第一项压低比值整体规模，第二项让模型把真实比值学对。SEDD 凭此在 GPT-2 尺度上把此前离散扩散的困惑度降低 25%–75%，首次与自回归可比，获 ICML 2024 最佳论文 [3](https://aiwiki.ai/wiki/discrete_diffusion)。

还有一条有用的工程结论：**RADD 发现吸收态下 concrete score 可以分解为"时间相关标量 $$\times$$ 时间无关的条件分布"** [8](https://arxiv.org/html/2406.03736v1)，于是网络主体不必以 $$t$$ 为输入，还能配合缓存减少函数评估次数。这是"掩码扩散可以把时间条件与网络解耦"的理论依据。

---

## 9. 采样视角：并行解码到底贵在哪

训练讲完了，回到采样。反向过程用 $$\tau$$-leaping 在一次前向里揭示多个 token：

$$
p(x_s^i,x_s^j\mid z_t)\;\approx\;p(x_s^i\mid z_t)\,p(x_s^j\mid z_t)
$$

右边就是条件独立假设。被同时揭示的 token 之间若有依赖，这个近似就会产出"局部合理、全局不通"的组合。

### 9.1 量化这个代价

`parallel_decoding_independence/run.py` 的 A 组：两个 token 的真实联合是"以 0.9 概率取相同值"。

| 解码方式 | KL(真实 vs 近似) | $$\mathrm{P}(\text{一致})$$ | 相对误差 |
|---|---|---|---|
| 串行（每步 1 个） | 0.000000 | 0.9000 | +0.00% |
| 并行（每步 2 个） | **0.368064** | **0.5000** | **−44.44%** |

一步并行就丢掉了 44% 的联合结构。三个 token 的链式分布上，每步提交 1 / 2 / 3 个，KL 分别是 0.000000 / 0.368064 / 0.736128——**误差随每步提交数线性增长**。

这正是 Fast-dLLM 提出置信度感知并行解码的动机：每步只提交超过置信阈值的位置，把"没把握"的位置继续留成掩码 [9](https://nvlabs.github.io/Fast-dLLM)。

### 9.2 提交顺序比步数更关键

B 组在 $$L=10$$、$$V=4$$ 的一阶马尔可夫链上跑三种 unmasking 策略（每步都做精确的前向后向推断，Viterbi 序列的 $$-\log p=7.5050$$ 作为参考）：

| 策略 | 步数 | 每步提交 | 与 Viterbi 一致率（greedy） | 生成 $$-\log p$$（sampling） |
|---|---|---|---|---|
| random | 1 | 10 | 60.0% | 17.2066 |
| random | 10 | 1 | 58.95% | 11.2987 |
| confidence | 1 | 10 | 60.0% | 17.4818 |
| confidence | 5 | 2 | **100%** | 13.2728 |
| confidence | 10 | 1 | **100%** | 11.2044 |
| semi_ar | 5 | 2 | **100%** | 14.0027 |
| semi_ar | 10 | 1 | **100%** | 11.3618 |

两条规律：

1. **步数是硬约束**：每步提交数从 10 降到 1，生成序列的 $$-\log p$$ 从 17.2 降到 11.2（真实最优 7.5）。这解释了为什么朴素实现里"采样步数等于回答长度时效果最好"——步数一压，条件独立假设的代价立刻显现 [7](https://github.com/ML-GSAI/LLaDA)。
2. **顺序比步数更关键**：随机顺序即使降到每步 1 个，一致率也只有约 59%；而 confidence 与 semi-AR 在每步 ≤2 个时就能 100% 复现 Viterbi。LLaDA 的半自回归重掩码（semi-autoregressive remasking）在指令微调模型上表现最好 [5](https://arxiv.org/html/2502.09992)，与这里的结论一致。

> **口径提示**：本实验每一步都做精确推断，置信度取自真实后验的最大概率；真实 dLLM 的置信度来自模型自身且会被校准误差污染，因此这里的策略差距是乐观估计，真实系统上差距通常更大。

---

## 10. 从公式到代码

### 10.1 变量映射

| 数学符号 | 代码变量 | Shape | 说明 |
|---:|:---|:---|:---|
| $$x_0$$ | `input_ids` | $$(B,L)$$ | 干净 token id |
| $$t$$ | `t` | $$(B,)$$ | $$t\sim\mathcal{U}[0,1]$$，**每条样本独立** |
| $$z_t$$ | `noisy_ids` | $$(B,L)$$ | 以概率 $$t$$ 把位置替换为 `[MASK]` |
| $$\mathbf{1}[z_t^i=\mathrm{M}]$$ | `is_mask` | $$(B,L)$$ | 布尔掩码 |
| $$1/t$$ | `loss_weight` | $$(B,)$$ | 广播到序列维 |
| $$p_\theta(\cdot\mid z_t)$$ | `logits` | $$(B,L,V)$$ | 双向 Transformer 输出（**无 causal mask**） |
| $$\mathcal{L}$$ | `loss` | 标量 | 掩码位置上的加权平均 |

### 10.2 最小实现

```python
import numpy as np

rng = np.random.default_rng(0)
B, L, V, MASK_ID = 4, 8, 32, 31

x0 = rng.integers(0, V - 1, size=(B, L))        # 干净序列
t  = rng.random(B)                              # 掩码率：每条样本独立采样

# 前向过程：吸收态，每个位置独立以概率 t 变成 [MASK]
is_mask = rng.random((B, L)) < t[:, None]
z_t = np.where(is_mask, MASK_ID, x0)

# 掩码预测器（真实实现里是去掉 causal mask 的 Transformer）
logp = np.log(rng.random((B, L, V)) + 1e-8)
token_logp = np.take_along_axis(logp, x0[..., None], axis=-1)[..., 0]

# 1/t 权重：t 是被掩码 token 的期望占比，除以它做归一化
w = 1.0 / np.maximum(t, 1e-3)
loss = -(w * (token_logp * is_mask).sum(axis=1) / np.maximum(is_mask.sum(axis=1), 1)).mean()
```

三处最容易写错：

1. `t` 必须**每条样本独立采样**；整批共用一个值会显著减少模型见到的掩码率多样性。
2. 别忘了指示函数：未掩码位置的信息已经泄漏，算进去会污染损失。
3. 架构上唯一必需的改动是**去掉 causal mask**（双向注意力）。这也是后续一切麻烦的起点——双向注意力与 KV cache 天生冲突，本系列下一篇就讲这件事怎么和解。

### 10.3 真实训练里还有四个坑

前三个来自"从自回归改过来"的惯性，第四个是掩码扩散独有的：

| 坑 | 现象 | 处理 |
|---|---|---|
| padding 参与损失 | 批内短序列的 pad 位置被当成要预测的 token | 用 label padding（如 `-100`）或在掩码指示里排除；LLaDA 的做法相反——它**故意在 padding 上训练**，靠它学会输出长度 [5](https://arxiv.org/html/2502.09992) |
| SFT 阶段掩码范围 | 对整个 prompt+response 一起掩码 | 只掩码 response，prompt 保持可见，这与自回归"只在 response 上算损失"是同一件事的扩散版 [5](https://arxiv.org/html/2502.09992) |
| 输出长度 | 朴素实现固定生成 $$L$$ 个 token | 让一部分训练样本长度随机取自 $$[1,L]$$，并把 `EOS` 也纳入掩码与损失，模型才能自己决定何时停 [5](https://arxiv.org/html/2502.09992) |
| 掩码 token 的嵌入 | 训练初期 loss 异常、不稳定 |  masked token embedding 上加一点独立噪声可以抑制早期数值不稳定（厂商/二手来源，见 [LLaDA 2.0 技术报告](https://github.com/inclusionAI/LLaDA2.0)） |

其中第三条最容易被忽视：变长生成不是"推理时截断"，而是训练时就要让模型见过不填满的序列。这是块扩散（本系列下一篇）要解决的核心问题之一。

---

## 11. 小结

- 文本扩散的"噪声"是离散状态转移，主流是吸收态掩码；`[MASK]` 一旦戴上不会自愈。
- 吸收态下连续时间 ELBO 化简为**带 $$1/t$$ 权重的掩码交叉熵**，这是 dLLM 能规模化的全部理论基础。
- $$1/t$$ 是**正确性**而非技巧：实跑显示带它时损失与 NLL 精确相等，去掉后掉到 NLL 之下 1.7382。
- 掩码率必须随机化。固定 15% 在 20 万样本里一次都没见到全掩码上下文，$$t=1$$ 处退化成均匀分布。
- 并行解码的代价来自 $$\tau$$-leaping 的条件独立假设，误差随每步提交数线性增长；提交顺序（confidence / semi-AR 优于 random）比步数本身更关键。

下一篇进入块扩散：为什么双向注意力与 KV cache 冲突，块化怎么同时解决变长生成与缓存复用，以及 Fast-dLLM / dInfer / SGLang 这条工程线是怎么落地的。

---

## 参考链接

* Nie et al., Large Language Diffusion Models (LLaDA) —— [arXiv:2502.09992](https://arxiv.org/html/2502.09992)
* Ho, Jain, Abbeel, Denoising Diffusion Probabilistic Models —— [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
* Discrete Diffusion in Large Language and Multimodal Models: A Survey —— [arXiv:2506.13759](https://ar5iv.arxiv.org/html/2506.13759)
* Discrete Diffusion Models: A Unified Framework from Tokenization to Generation —— [arXiv:2607.13431](https://arxiv.org/html/2607.13431v1)
* Discrete diffusion language model（D3PM / SEDD / MDLM / MD4 脉络梳理）—— [aiwiki](https://aiwiki.ai/wiki/discrete_diffusion)
* Diffusion Language Models（含 Plaid、连续潜空间路线）—— [aiwiki](https://aiwiki.ai/wiki/diffusion_language_models)
* Kuleshov Group, How to Build a Diffusion Language Model（ICLR 2026 / MLSS 2026 讲义改编）—— [kuleshov-group.github.io](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/)
* LLaDA 官方仓库与 FAQ（"改成 LLaDA 只需动几行"）—— [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)
* Ou et al., Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data (RADD) —— [arXiv:2406.03736](https://arxiv.org/html/2406.03736v1)
* Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding —— [项目页](https://nvlabs.github.io/Fast-dLLM) / [arXiv:2505.22618](https://arxiv.org/abs/2505.22618)
* 配套实验：`ipynbs` commit [`c90dcf8`](https://github.com/lrypcy/ipynbs/tree/c90dcf8/experiments/diffusion-lm/) 的 `elbo_masked_diffusion/` 与 `parallel_decoding_independence/`

> 🧪 **动手练习**：① 把 `elbo_masked_diffusion/run.py` 里的线性表 $$\alpha_t=1-t$$ 换成余弦表，重跑 C 组，看带 $$1/t$$ 的损失是否仍等于 NLL（预期：仍相等，因为目标对时间表形式不敏感，只依赖端点）；② 把 B 组的序列长度从 6 加到 8，观察固定 15% 方案的全掩码观测次数与 CE@1.00 如何变化；③ 在 `parallel_decoding_independence/run.py` 里把两个 token 的相关性从 0.9 降到 0.6，验证并行解码的相对误差是否随之缩小——这能直接给出“什么时候可以放心并行”的经验阈值。

> **上一篇**：无（系列首篇）｜ **下一篇**：02 块扩散与 KV Cache（写作中）
