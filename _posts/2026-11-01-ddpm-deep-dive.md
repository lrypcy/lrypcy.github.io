---
title: "DDPM 深度剖析：从理论到实践"
date: 2026-11-01 17:30:00 +0800
categories:
  - 生成模型
tags: [diffusion, DDPM, DDIM, noise-schedule, U-Net, sampling]
layout: post
mathjax: true
---

> **系列导航** ｜ [扩散模型综述：算法与应用](/2026/08/24/diffusion-survey-algorithms-applications/) ｜ [Score、扩散与 SDE](/2026/08/22/flow-matching-02-score-diffusion-sde/) ｜ [Flow Matching 原理篇](/2026/10/20/flow-matching-00-principles/) ｜ [扩散 Transformer 之一：Patchify、AdaLN-Zero 与 Token 预算](/2026/10/14/dit-01-patchify-adaln-zero-token-budget/)

> **TL;DR**
>
> * **前向过程只有一个目的**：把数据逐步推成标准高斯，并且任意步可以闭式采样——$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$。训练时不用真跑 $$t$$ 次前向，随机抽一个 $$t$$、按上式直接合成样本即可。
> * **简化目标把变分下界换成噪声回归**：$$L_{\mathrm{simple}}=\mathbb E\big\lVert\epsilon-\epsilon_\theta(x_t,t)\big\rVert^2$$。它丢掉了 ELBO 里的时间权重，换来更低方差和更好的生成质量，代价是训练目标不再是严格的似然下界，不能用它去比较不同模型的对数似然。
> * **时间步是全局条件，不是层号**：同一个 U-Net 在整个采样过程中被反复调用 $$T$$ 次；$$t$$ 经正弦位置编码后在每个残差块内注入，决定网络当前应该“去多少噪”。
> * **适用边界**：串行去噪意味着一次采样至少 $$T$$ 次网络前向，这是 DDPM 慢的结构性原因；DDIM（$$\eta=0$$）与高阶求解器把步数从 1000 压到几十步，代价是引入离散化误差，需要在采样预算与质量之间取舍。

## 一、DDPM 盯的是什么问题

生成模型的任务是从一个易采样的分布出发，把样本变换成目标数据分布 $$p_{\mathrm{data}}(x_0)$$。VAE 用一个编码器把这个变换一次性学出来，GAN 用判别器间接逼迫Generator 匹配分布，而 DDPM 选择了第三条路：**先定义一个固定的、能把数据毁成噪声的过程，再学它的逆过程**。

这条路有两个结构上的好处。第一，正向破坏过程完全由人指定、无需学习，所以中间状态的分布是可知的；第二，逆过程的每一步只需要做局部去噪，把一个极难的一步到位的生成任务拆成 $$T$$ 个相对容易的小任务。

下面沿着这条线索展开：先写清前向过程（§2），再看时间步如何进入网络（§3），然后是用什么目标训练（§4）、怎么采样与加速（§5）、网络长什么样（§6），最后是工程取舍与局限（§7–§8）。

## 二、前向扩散过程

### 2.1 单步马尔可夫链

每一步加噪遵循：

$$q(x_t\mid x_{t-1})=\mathcal N\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big),\qquad \epsilon\sim\mathcal N(0,I)$$

等价地写成采样形式：

$$x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon$$

其中 $$\beta_t\in(0,1)$$ 是预设的方差序列（噪声调度），不参与学习。整个前向过程是一条马尔可夫链：

$$q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1})$$

### 2.2 闭式：DDPM 效率的基石

令 $$\alpha_t=1-\beta_t$$、$$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$$，反复代入并利用两个独立高斯之和仍是高斯（方差直接相加），可以得到任意时刻的边际分布：

$$q(x_t\mid x_0)=\mathcal N\big(x_t;\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)$$

也就是一步到位的采样式：

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$

这条式子是整个算法的关键。训练时在 $$1,\ldots,T$$ 上均匀采一个 $$t$$、再采一个标准高斯的 $$\epsilon$$，就能直接合成任意噪声强度的样本 $$(x_t,t,\epsilon)$$，**不需要真的执行 $$t$$ 次前向传播**。这把一个 $$T$$ 步的序列化过程变成了可随机访问的数据集，训练复杂度与 $$T$$ 解耦。

为了让 $$x_T$$ 确实接近标准高斯，调度需要满足 $$\bar\alpha_T\approx 0$$。

### 2.3 时间步到底指的是什么

这是理解 DDPM 最容易出错的地方：**$$t$$ 不是神经网络的层编号，而是输入图像的噪声水平**。同一个 U-Net 在不同的 $$t$$ 上被反复调用：

| 时间步 | $$x_t$$ 的状态（$$T=1000$$ 时） |
| --- | --- |
| $$t=0$$ | 干净数据，无噪声 |
| $$t\approx 100$$ | 轻度噪声，局部纹理受损 |
| $$t\approx 500$$ | 中度噪声，全局轮廓尚存、细节丢失 |
| $$t=1000$$ | 几乎不可分辨，接近标准高斯 |

所以 DDPM **没有 1000 层网络**，它只有一个网络、$$T$$ 种不同的工作条件。$$t$$ 是整个 U-Net 的**全局条件变量**，它告诉网络“当前输入里的信噪比是多少”——信噪比高时该做细微修正，信噪比低时该先把大体结构猜出来。

由于 $$x_t$$ 中的信号被 $$\sqrt{\bar\alpha_t}$$ 缩放，从输入本身其实能反推出一部分噪声水平信息，但显式地喂给网络 $$t$$ 更有效也更稳定。

## 三、时间步如何进网络

### 3.1 时间嵌入

$$t$$ 是标量，需要先映射成高维向量。DDPM 用的是 Transformer 那套正弦位置编码：

$$\mathrm{TE}(t)=\big[\sin(\omega_1 t),\cos(\omega_1 t),\ldots,\sin(\omega_d t),\cos(\omega_d t)\big],\qquad \omega_k=\frac{1}{10000^{\,2k/d}}$$

它有三个实用性质：取值有界、不同步的表示近似正交、相邻的 $$t$$ 编码距离小——最后一点让网络能在相邻时间步之间插值，而不需要每个 $$t$$ 都记住一套独立行为。编码之后通常再过一层 MLP 得到条件向量 $$e_t$$。

### 3.2 注入策略

常见的注入方式有三种：

1. **加法注入**：把 $$e_t$$ 投影后直接加到残差块的特征图上，实现最简单。
2. **缩放偏移（AdaGN / scale-shift modulation）**：先做 GroupNorm，再用 $$e_t$$ 学出的 $$\gamma_t,\beta_t$$ 做仿射变换。这是目前最常用的形式，与 DiT 里的 AdaLN 属于同一类机制。
3. **交叉注意力**：把条件作为额外的 key / value 引入注意力层，多用于文本等更复杂的条件。

一个值得注意的事实：在同一次前向传播里，所有残差块收到的是**同一个** $$t$$。它们之所以能学到不同的去噪行为，靠的是各层参数不同、各分辨率的特征尺度不同，以及 $$e_t$$ 在进入不同块之前各自独立的投影。

## 四、训练目标

### 4.1 从变分下界到简化目标

严格的目标是对数似然的变分下界。把 ELBO 展开，每一项都是两个高斯的 KL：真正的后验 $$q(x_{t-1}\mid x_t,x_0)$$（高斯，且是闭式的）与模型 $$p_\theta(x_{t-1}\mid x_t)$$。两个高斯的 KL 有闭式解，因此目标可以写成：

$$L_{\mathrm{vlb}}=\mathbb E\Big[\sum_{t>1}D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)-\log p_\theta(x_0\mid x_1)\Big]$$

其中真实后验的均值为：

$$\tilde\mu_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

DDPM 的做法是：既然网络可以直接预测噪声 $$\epsilon$$，就用它把 $$\tilde\mu_t$$ 重参数化，再把目标换成预测噪声与真噪声之间的均方误差，并**丢掉 KL 带来的那组时间权重**：

$$L_{\mathrm{simple}}=\mathbb E_{x_0,\,t,\,\epsilon}\Big\lVert\epsilon-\epsilon_\theta(x_t,t)\Big\rVert^2$$

这个简化是有代价的，必须写清楚：

- **收益**：去掉权重后目标的方差显著降低，训练更稳，主观生成质量更好（原论文 FID 指标也更优）。
- **代价**：$$L_{\mathrm{simple}}$$ 不再是 log-likelihood 的严格下界，**不同模型之间的 $$L_{\mathrm{simple}}$$ 数值不可直接比较**，也不能拿它去做似然相关的评测。想优化真似然，还得回到带权重的 $$L_{\mathrm{vlb}}$$（其中 $$L_{t-1}$$ 项用预测均值与 $$\tilde\mu_t$$ 的加权 MSE 计算）或离格式 loss。

### 4.2 训练流程

对每个 batch：

1. 采数据 $$x_0\sim p_{\mathrm{data}}$$；
2. 均匀采时间步 $$t\sim\mathcal U\{1,\ldots,T\}$$，采噪声 $$\epsilon\sim\mathcal N(0,I)$$；
3. 用 §2.2 的闭式合成本步的带噪样本 $$x_t$$；
4. 前向 $$\hat\epsilon=\epsilon_\theta(x_t,t)$$，对 $$\lVert\epsilon-\hat\epsilon\rVert^2$$ 求梯度。

注意这里没有任何序列：不同样本、不同 batch 的 $$t$$ 完全独立。

## 五、采样与 DDIM 加速

### 5.1 DDPM 的欧拉式串行去噪

学习到的反向转移是一个高斯：

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N\big(x_{t-1};\ \mu_\theta(x_t,t),\ \sigma_t^2 I\big)$$

用上述 $$\tilde\mu_t$$ 做模板、把其中的 $$x_0$$ 换成从当前预测反解出来的干净图像

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$$

代入即得：

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)$$

完整采样循环：

```python
x_T = torch.randn_like(x)                 # 从标准高斯出发
for t in reversed(range(T)):              # t = T-1, ..., 0
    eps = unet(x, t)                      # 同一个 U-Net，输入不同 x 与 t
    eps_coef = beta[t] / (1 - alpha_bar[t]).sqrt()
    mean = (x - eps_coef * eps) / alpha[t].sqrt()
    z = torch.randn_like(x) if t > 0 else 0      # 最后一步不加随机项
    x = mean + sigma[t] * z               # sigma_t^2 = beta_t 或 beta_t 的复取值
```

整个过程是串行的：必须算出 $$x_{t-1}$$ 才能算下一步。生成一张图要做 $$T$$ 次网络前向，这就是 DDPM 慢的结构性原因，也是后续加速工作的共同出发点。

### 5.2 DDIM：把随机过程换成确定性 ODE

DDIM 的关键观察是：**前向过程不必非要马尔可夫**。只要满足边际条件 $$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$$，前向链有无穷多种取法；换一种取法，反向采样就变成：

$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t,t)+\sigma_t\epsilon_t$$

$$\sigma_t^2=\eta\cdot\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\Big(1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}\Big)$$

其中 $$\hat x_0$$ 用 5.1 的式子反解，$$\epsilon_t\sim\mathcal N(0,I)$$。参数行为：

- $$\eta=1$$：等价回到 DDPM 的随机采样；
- **$$\eta=0$$**：$$\sigma_t=0$$，随机项消失，过程完全确定——同一个初始 $$x_T$$ 永远得到同一张图，这也让潜空间插值和编辑变得可以做。

真正的加速来自 DDIM 允许**只跑原时间序列的一个子序列** $$[\tau_1,\ldots,\tau_S]$$（$$S\ll T$$），把式中的 $$\bar\alpha_{t-1}$$ 换成子序列里上一个时刻的 $$\bar\alpha$$。因为不再要求每步的分布恰好匹配反向马尔可夫链，跳步引入的是 ODE 离散化误差而非分布偏差，于是 50 步甚至 20 步也能得到合理结果。

这只是三类加速路线中的一类：

| 路线 | 代表 | 思路 | 代价 |
| --- | --- | --- | --- |
| 免训练跳步 | DDIM、DPM-Solver | 用高阶 ODE 求解器支持更大的步长 | 步长越大，离散化误差越明显 |
| 蒸馏 | Consistency Models、LCM | 训练模型在少数几步内到达同样的终点 | 需要额外训练，容量受限时细节下降 |
| 换空间 | Latent Diffusion | 在压缩后的潜空间扩散，像素上的计算交给自编码器 | 生成质量受自编码器重建上限约束 |

## 六、U-Net 骨架

DDPM 用的是带注意力机制的 U-Net：

- **编码器**：多级下采样，逐级提高通道数、降低分辨率，提取多尺度特征；
- **瓶颈层**：最低分辨率上接自注意力，做全局信息混合；
- **解码器**：上采样重建分辨率；
- **跳连**：把编码器对应分辨率的特征直接拼回到解码器，保住下采样过程中丢掉的高频细节；
- **残差块**：每一层的基本单元，也是时间嵌入的注入点（§3.2）。

之所以选 U-Net 而不是纯 Transformer：图像是二维网格，卷积的局部归纳偏置天然契合，参数效率更高；DDPM 需要在每个分辨率上都输出、并能接收注入的条件，U-Net 的多尺度对称结构刚好满足。后续 DiT 的工作才把它整体换成 Transformer，那需要更大的算力预算来换 scalability。

## 七、噪声调度与工程取舍

### 7.1 调度怎么选

三条设计原则：单调递增、相邻步平滑变化、$$\bar\alpha_T$$ 足够接近 0。

- **线性调度**（DDPM 原论文）：$$\beta_t$$ 从 $$10^{-4}$$ 线性增长到 $$0.02$$、$$T=1000$$。简单，但在接近纯噪声的区间上“消耗”了太多步数——它们对应的 SNR 已经极低，对最终质量贡献很小。
- **余弦调度**（Improved DDPM）：直接对 $$\bar\alpha_t$$ 而不是 $$\beta_t$$ 做约束，

  $$\bar\alpha_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)$$

  偏移 $$s=0.008$$ 用来防止 $$t$$ 很小时 $$\beta_t$$ 过小。它在中间段近似线性，两端刻意放缓，让采样珍贵的步数集中在对质量最敏感的中等噪声区间。

实践里余弦调度更常见，但两者不是“哪个绝对更好”：在高分辨率或 latent 空间上，线性调度仍然被广泛使用，选择要看具体的 anchor 指标。

### 7.2 训练稳定性手段

- **EMA**：对参数维护指数移动平均，采样时用平均后的权重，通常能明显提升样本质量；
- **学习率预热**：前若干千步线性升温，避免训练早期 Adam 的方差估计不稳；
- **混合精度**：显存减半、带宽减半；需要配合 loss scaling 防止下溢；
- **梯度检查点 / 梯度累积**：前者用重计算换显存，后者用小 batch 拼出等效大 batch 的统计量。

这些都是通用技巧，不属于 DDPM 的贡献，但扩散模型的训练实践几乎离不开它们。

## 八、局限与改进方向

1. **采样慢**：1000 步串行推理，延迟对交互式场景不可接受。路线见 §5.2 的表格：跳步、蒸馏、换潜空间，三者也可以叠加。
2. **似然优化与质量问题割裂**：$$L_{\mathrm{simple}}$$ 不是似然下界（§4.1），需要似然时要回到 $$L_{\mathrm{vlb}}$$ 或改用另一套目标。
3. **条件生成能力有限**：原版只支持类标签条件。文本到图像需要 CFG 之类的方法把条件信号放大，但那已经是启发式，会牺牲多样性。
4. **像素空间开销**：原版直接在像素上扩散，256 以上的分辨率成本陡增。Latent Diffusion 把压缩这一步交给预训练自编码器，这也是 Stable Diffusion 系列的地基。

## 九、小结

DDPM 的价值可以概括成三句话：

1. **把生成拆解成可监督的去噪任务**——前向过程人为设计且闭式可知，于是每一步都有精确的监督信号；
2. **训练完全可并行**：$$\bar\alpha_t$$ 的闭式让任意时间步可随机访问，训练复杂度与前进步数解耦；
3. **留下了一个清晰的可改进点**：采样的串行性。DDIM → DPM-Solver → Consistency Models → Latent Diffusion 这条线的每一环，都是在“效率”和“质量”之间重新取一个折中点。

## 参考文献

1. **Denoising Diffusion Probabilistic Models** — Ho, Jain, Abbeel, NeurIPS 2020（arXiv:2006.11239）。本文 §2、§4、§6 的公式来源。<https://arxiv.org/abs/2006.11239>
2. **Denoising Diffusion Implicit Models** — Song, Meng, Ermon, ICLR 2021（arXiv:2010.02502）。§5.2 的非马尔可夫前向与跳步采样。<https://arxiv.org/abs/2010.02502>
3. **Improved Denoising Diffusion Probabilistic Models** — Nichol & Dhariwal, ICML 2021（arXiv:2102.09672）。余弦调度与 learned variance。<https://arxiv.org/abs/2102.09672>
4. **High-Resolution Image Synthesis with Latent Diffusion Models** — Rombach et al., CVPR 2022（arXiv:2112.10752）。<https://arxiv.org/abs/2112.10752>
5. **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps** — Lu et al., NeurIPS 2022（arXiv:2206.00927）。<https://arxiv.org/abs/2206.00927>
6. **Consistency Models** — Song et al., ICML 2023（arXiv:2303.01469）。<https://arxiv.org/abs/2303.01469>

## 上一篇 / 下一篇

- 上一篇：[扩散 Transformer 之一：Patchify、AdaLN-Zero 与 Token 预算](/2026/10/14/dit-01-patchify-adaln-zero-token-budget/)
- 下一篇：[Flow Matching 原理篇：从概率路径到一边解方程一边生成](/2026/10/20/flow-matching-00-principles/)
