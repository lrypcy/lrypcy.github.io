---
title: "从概率路径到 Flow Matching：用一套数学框架理解现代扩散模型——从连续性方程、条件流匹配，到 Score、SDE 与 Classifier-Free Guidance 的完整推导"
date: 2026-10-20 22:00:00 +0800
categories:
  - 生成模型
tags: [flow-matching, diffusion, probability-path, continuity-equation, score-matching, cfg, rectified-flow, sde, fokker-planck]
layout: post
mathjax: true
---

> **系列导航** ｜ [建模之一：概率路径与边际向量场（数值核验存档）](/2026/08/22/flow-matching-01-algorithm-evolution/) ｜ [建模之二：score、扩散与 SDE](/2026/08/22/flow-matching-02-score-diffusion-sde/) ｜ [建模之三：条件生成、离散域与潜空间](/2026/08/22/flow-matching-03-guidance-discrete-latent/) ｜ [应用篇：SD3 到蛋白质设计](/2026/08/22/flow-matching-04-applications/) ｜ [扩散模型综述](/2026/08/24/diffusion-survey-algorithms-applications/)

> **TL;DR**
>
> * **一条主线**：先设计概率分布如何随时间变化（概率路径 $$p_t$$），再求出输运它所需的动力学（速度场 $$u_t$$，由连续性方程与路径绑定），最后用神经网络近似动力学、数值积分完成生成。
> * **训练不解 ODE，靠一条恒等式**：$$\mathcal L_{\mathrm{CFM}}=\mathcal L_{\mathrm{FM}}+C$$（$$C$$ 与参数无关）——把“回归算不出来的边际速度场”换成“回归有闭式的条件速度场”，CondOT 线性路径下就是回归 $$z-\epsilon$$，一行 MSE。
> * **高斯路径下 Flow Matching 与扩散模型是同一件事**：$$u_t=\frac{\dot a_t}{a_t}x+\left(\frac{\dot a_tb_t^2}{a_t}-b_t\dot b_t\right)s_t$$。velocity / score / denoiser / noise 四种参数化可互相换算，差别只在有效时间权重——系数接近零的时间步上，换算会放大误差。
> * **采样期的随机性是免费旋钮**：漂移项加 $$\tfrac{g_t^2}{2}\nabla\log p_t$$ 的 SDE 与 ODE 边际分布相同；直接加噪声不改漂移会多出扩散项，毁掉概率路径。
> * **CFG 采的是倾斜分布**：$$\tilde p_t(x)\propto p_t(x)\,p_t(y\mid x)^w$$。$$w=1$$ 才是正确的条件后验；$$w>1$$ 进一步放大条件似然——更贴提示，但牺牲多样性，且它是启发式而非无偏采样。

当我们使用一个文生图模型时，输入的是一段文本，输出的却是一张包含数百万像素的图像。模型究竟如何完成这件事？

一种常见的解释是：模型先生成一张随机噪声图，再逐步去噪，最终得到图像。

这个解释没有错，但它只描述了生成过程的表象。它没有回答更本质的问题：**为什么逐步去噪能够生成符合真实数据分布的样本？为什么预测噪声、预测干净图像和预测速度都能成为有效的训练目标？为什么一个确定性的 ODE 和一个带随机噪声的 SDE，可以产生相同的边际概率分布？**

要回答这些问题，我们需要把扩散模型从具体的算法实现中抽象出来，建立一个统一的数学框架。

本文以 [MIT 6.S184：An Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/) 的理论体系为基础，结合 Flow Matching、Score Matching 和现代生成模型的相关研究，从最基本的概率分布开始，逐步推导整个框架。

读完之后，你应该能够理解以下几个问题：

- Flow Matching 为什么可以不求解微分方程就完成训练？
- 连续性方程如何把一条概率路径与一个速度场联系起来？
- Conditional Flow Matching 为什么能够替代难以直接优化的 Flow Matching 目标？
- Velocity、Score、Denoiser 和 Noise Prediction 之间究竟有什么数学关系？
- 为什么扩散模型可以在采样时引入随机性，却仍然保持目标概率分布不变？
- Classifier-Free Guidance（CFG）为什么能够增强文本与图像的匹配程度，又为什么会牺牲多样性？

整个推导围绕一条主线展开：

**先设计概率分布如何随时间变化，再求出输运这些分布所需的动力学，最后用神经网络近似动力学，并通过数值积分完成生成。**

---

## 一、生成模型的本质：从生成一个样本，到生成一个概率分布

### 1.1 为什么生成任务首先是一个概率问题？

假设我们希望生成一张猫的图片。

符合要求的图片显然不止一张：猫可以处于不同姿态、不同光照条件和不同背景中。生成模型不应该只输出某一张固定图片，而应该能够产生具有合理多样性的样本。

因此，我们首先将一张图片表示为一个高维向量：

$$
x\in\mathbb R^d
$$

这里的 $$d$$ 可以是图像所有像素和颜色通道展开后的维度，也可以是视频或其他连续数据经过向量化后的维度。

真实数据并不是均匀分布在整个空间中。真实图片通常集中在具有特定结构的区域，而随机生成的像素向量大概率不对应一张自然图像。

我们用概率分布 $$p_{\mathrm{data}}(x)$$ 描述真实数据的统计规律。

严格来说，如果数据具有连续概率密度，那么 $$p_{\mathrm{data}}(x)$$ 是一个密度函数；如果数据分布是离散的、奇异的，或者集中在低维流形上，则还需要使用更一般的概率测度来描述。本文为了便于推导，主要讨论具有适当密度和正则性的连续情形。

生成任务于是可以写成：

$$
x\sim p_{\mathrm{data}}
$$

这句话意味着：我们希望构造一个算法，使它输出的随机变量具有与真实数据相同或足够接近的分布。

### 1.2 为什么从高斯噪声开始？

真实数据分布复杂，通常很难直接采样；但标准高斯分布非常容易采样：

$$
x_0\sim p_0=\mathcal N(0,I)
$$

因此，一个自然的想法是构造一个随时间演化的过程，将简单分布 $$p_0$$ 转换为复杂的数据分布 $$p_1=p_{\mathrm{data}}$$。

如果把时间归一化到区间 $$[0,1]$$，那么我们的目标就是找到一族概率分布：

$$
\{p_t\}_{t\in[0,1]}
$$

满足：

$$
p_{t=0}=p_0,\qquad
p_{t=1}=p_{\mathrm{data}}
$$

在生成过程中，我们从 $$p_0$$ 采样，然后沿着某种动力学演化状态。最终得到的随机变量 $$X_1$$ 应满足：

$$
X_1\sim p_{\mathrm{data}}
$$

注意，这里真正的目标不是让每一个初始噪声都对应某一张预先指定的图片，而是让整个随机群体在终点服从正确的数据分布。

这一区别非常重要。生成模型需要学习的是分布之间的转换规律，而不只是单个样本的恢复过程。

### 1.3 生成过程有三个可以区分的问题

把生成任务写成概率分布的演化后，就可以将问题拆成三个层次。

| 层次 | 核心问题 | 对应的数学对象 |
|---|---|---|
| 路径设计 | 中间时刻的概率分布应该是什么？ | 概率路径 $$p_t$$ |
| 动力学设计 | 怎样让样本按照这条路径运动？ | 速度场 $$u_t(x)$$ 或 SDE 漂移项 |
| 函数学习 | 如何用神经网络表示动力学？ | 参数化模型 $$u_\theta(t,x)$$ 等 |

这三个问题相关，却不能混为一谈。

一条概率路径规定了每个时刻的分布，但没有直接告诉我们每个样本应该朝哪个方向移动。速度场负责描述局部运动规律，而神经网络只是近似这个速度场的一种工具。

Flow Matching 的重要思想，正是把这三个层次拆开处理。

在进一步讨论之前，我们先建立描述连续运动的数学工具。

---

## 二、ODE 与 Flow：用确定性动力学搬运概率分布

### 2.1 从速度场定义轨迹

在经典力学中，如果已知一个物体在每个位置、每个时刻的速度，就可以通过积分计算它的运动轨迹。

生成模型中的 Flow 具有类似的结构。

定义一个时变向量场：

$$
u:[0,1]\times\mathbb R^d\to\mathbb R^d
$$

其中 $$u_t(x)$$ 表示粒子在时刻 $$t$$、位置 $$x$$ 处的瞬时速度。

我们希望粒子遵循如下常微分方程：

$$
\boxed{
\frac{dX_t}{dt}=u_t(X_t)
}
$$

并给定初始条件：

$$
X_0=x_0
$$

这两条式子共同决定了粒子的演化。

需要区分三个概念：

- **向量场**：规定每个位置的瞬时速度。
- **轨迹**：某个初始点在动力学下实际经过的路径。
- **Flow**：把所有初始点映射到各个时刻位置的映射。

将最后一个概念写成：

$$
X_t=\psi_t(X_0)
$$

其中 $$\psi_t$$ 是由 ODE 诱导的流映射。它满足：

$$
\frac{d}{dt}\psi_t(x)
=u_t(\psi_t(x)),
\qquad
\psi_0(x)=x
$$

在向量场满足适当的存在性、唯一性和正则性条件时，初值决定唯一轨迹；在更强的条件下，流映射还具有可微性和可逆性。

这也是为什么 Flow 模型的核心网络通常预测速度，而不是直接预测整个流映射：网络只需要给出局部运动规律，完整的演化过程交给 ODE 求解器完成。

### 2.2 如何通过 ODE 实现生成？

现在令初始状态随机：

$$
X_0\sim p_0
$$

然后求解：

$$
\frac{dX_t}{dt}=u_\theta(t,X_t)
$$

最终得到：

$$
X_1=\psi_1^\theta(X_0)
$$

模型训练的目标是使：

$$
\boxed{
(\psi_1^\theta)_\#p_0
\approx p_{\mathrm{data}}
}
$$

这里的 $$\#$$ 表示概率分布在映射下的推前（pushforward）。

直观地说，初始噪声经过神经网络所描述的流映射后，应该变成符合真实数据分布的样本。

这带来一个重要认识：

**Flow 模型中的随机性可以完全来自初始采样，演化过程本身不必包含随机噪声。**

只要初始点是随机的，不同初始点就可能得到不同的生成结果。

### 2.3 为什么还需要数值积分？

除了少数特殊情况，神经网络定义的 ODE 通常无法得到解析解，因此必须使用数值积分。

最简单的方法是 Euler 法：

$$
X_{t+h}
=
X_t+h\,u_\theta(t,X_t)
$$

其思想是：在一个足够小的时间间隔内，暂时认为当前速度近似不变。

如果把总时间划分成 $$N$$ 个等长区间，则步长为：

$$
h=\frac1N
$$

反复执行更新，最终得到 $$X_1$$ 的近似值。

Euler 法的一步局部截断误差在常规光滑条件下为 $$O(h^2)$$，全局误差通常为 $$O(h)$$。更高阶的方法可以利用多个位置或时刻的速度来修正更新方向。

这意味着采样质量不仅取决于网络是否准确，还取决于速度场的几何结构和数值积分误差。

如果轨迹很弯，粗粒度的积分可能难以准确跟踪；如果轨迹足够直，在相同误差要求下，就有机会使用更少的采样步数。

这也为后面理解 Rectified Flow 和现代少步数生成模型埋下了伏笔。

---

## 三、从 ODE 到概率路径：什么才是正确的速度场？

前面我们已经知道，可以用 ODE 将初始分布搬运到目标分布。

但现在出现了一个关键问题：

**如果我们事先指定了每个时刻的目标分布 $$p_t$$，怎样才能判断某个速度场是否真的实现了这条概率路径？**

回答这个问题，需要从单个粒子的运动转向整个粒子群体的概率密度。

### 3.1 连续性方程：概率质量守恒

设 $$p_t(x)$$ 是时刻 $$t$$ 的概率密度，$$u_t(x)$$ 是对应的速度场。

在一个很短的时间间隔内，某个空间区域中的概率质量会发生变化。其变化只能来自概率质量穿过区域边界的流入和流出。

这与流体力学中的质量守恒、电磁学中的电荷守恒具有相同的数学结构。

定义概率流密度：

$$
j_t(x)=p_t(x)u_t(x)
$$

那么，概率密度的演化满足连续性方程：

$$
\boxed{
\partial_t p_t(x)
+
\nabla\cdot\left(p_t(x)u_t(x)\right)=0
}
$$

也可以写成：

$$
\partial_t p_t=-\nabla\cdot(p_tu_t)
$$

这里：

- $$\partial_t p_t$$ 描述某个位置的概率密度随时间的变化。
- $$p_tu_t$$ 描述概率质量的流动。
- 散度 $$\nabla\cdot(p_tu_t)$$ 描述该点附近概率流的净流出程度。

因此，如果某个区域的净概率流出为正，那么该区域内的概率密度就会下降；反之则会上升。

### 3.2 从微观轨迹到宏观密度

为什么 ODE 会导出连续性方程？

可以从概率质量守恒出发理解。对于任意足够规则的区域 $$\Omega$$，其内部的概率质量为：

$$
P_t(\Omega)=\int_\Omega p_t(x)\,dx
$$

概率质量的变化率等于边界上的净流入：

$$
\frac{d}{dt}\int_\Omega p_t(x)\,dx
=
-\int_{\partial\Omega}
p_t(x)u_t(x)\cdot n(x)\,dS
$$

其中 $$n(x)$$ 是区域边界的外法向量。

使用散度定理：

$$
\int_{\partial\Omega}p_tu_t\cdot n\,dS
=
\int_\Omega\nabla\cdot(p_tu_t)\,dx
$$

于是：

$$
\int_\Omega
\left[
\partial_t p_t+
\nabla\cdot(p_tu_t)
\right]dx=0
$$

由于该关系对任意合适的区域都成立，因此在相应正则性条件下，可以得到连续性方程：

$$
\partial_t p_t+\nabla\cdot(p_tu_t)=0
$$

这条方程是 Flow Matching 整个理论体系的核心。

它将两个原本不同的对象联系起来：一边是我们想要的概率路径，另一边是推动样本运动的速度场。

### 3.3 一个容易忽略的数学细节：速度场不一定唯一

连续性方程并不意味着给定 $$p_t$$ 后，速度场 $$u_t$$ 在所有情况下都唯一。

原因在于，连续性方程约束的是：

$$
\nabla\cdot(p_tu_t)
$$

而不是直接约束 $$u_t$$ 的每一个分量。

例如，如果向概率流中加入某个满足

$$
\nabla\cdot(p_t w_t)=0
$$

的向量场 $$w_t$$，那么 $$u_t+w_t$$ 仍然满足相同的连续性方程。

因此，同一条概率路径可能对应多个不同的速度场。要获得唯一性，需要额外的约束或选择准则。

这并不妨碍我们训练生成模型。Flow Matching 的关键不是声称任意路径都存在唯一速度场，而是构造一个明确、可学习且能够输运目标路径的速度场。

接下来，我们就会看到如何构造这样的速度场。

---

## 四、概率路径的设计：先规定中间分布，再求运动规律

### 4.1 条件概率路径与边际概率路径

假设我们有一个数据样本：

$$
z\sim p_{\mathrm{data}}
$$

我们希望定义一条条件概率路径：

$$
p_t(x\mid z)
$$

它描述在给定数据样本 $$z$$ 的条件下，中间状态 $$X_t$$ 的分布如何随时间变化。

例如，在生成过程中，我们可以让 $$t=0$$ 时的状态接近随机噪声，而让 $$t=1$$ 时的状态接近给定的数据样本。

但是，真正的生成任务不是提前给定一个目标数据样本 $$z$$，而是从整个数据分布中随机产生样本。因此，我们需要对所有可能的 $$z$$ 进行边际化：

$$
\boxed{
p_t(x)
=
\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz
}
$$

这就是边际概率路径。

如果条件路径的起点和终点设计正确，那么边际路径就能满足：

$$
p_0=p_{\mathrm{noise}},
\qquad
p_1=p_{\mathrm{data}}
$$

这里必须区分两件事：

- 条件路径 $$p_t(x\mid z)$$ 是围绕单个目标样本构造的。
- 边际路径 $$p_t(x)$$ 才是整个生成过程在时刻 $$t$$ 的真实分布。

这一区别直接决定了后面为什么不能简单地将每个训练样本对应的条件速度直接当作生成时的速度。

### 4.2 一个最重要的例子：高斯插值路径

在实践中，一个简单且广泛使用的选择是高斯条件路径。

设：

$$
\boxed{
X_t=a_t z+b_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
}
$$

其中 $$a_t$$ 和 $$b_t$$ 是随时间变化的调度函数。

采用本文的时间方向约定：

$$
a_0=0,\quad b_0=1
$$

$$
a_1=1,\quad b_1=0
$$

于是：

$$
X_0=\epsilon,\qquad X_1=z
$$

当 $$t$$ 从 0 增加到 1 时，样本从噪声逐渐转向数据。

对于给定的 $$z$$，该条件分布为：

$$
p_t(x\mid z)
=
\mathcal N(a_tz,b_t^2I)
$$

当 $$b_t>0$$ 时，它具有明确的高斯密度，也很容易采样。

一个尤其重要的特例是线性插值：

$$
\boxed{
X_t=(1-t)\epsilon+tz
}
$$

它对应：

$$
a_t=t,\qquad b_t=1-t
$$

这种路径通常称为 CondOT（Conditional Optimal Transport）形式的线性路径，在 Rectified Flow 和多种现代 Flow Matching 模型中具有重要地位。

为什么这条路径值得关注？因为它让条件轨迹成为直线，条件速度也不再依赖时间。

但要注意，**条件轨迹是直线，并不保证边际速度场的轨迹也都是直线。** 边际速度是多个条件速度的加权平均，整体几何结构仍然取决于数据分布以及条件样本之间的关系。

### 4.3 为什么概率路径可以有很多种？

我们并不一定非要选择线性插值。

原则上，只要满足适当的端点条件和正则性要求，就可以设计不同的概率路径。例如，可以调整信号与噪声的衰减速度，也可以选择非线性插值。

不同路径会影响：

1. 中间时刻的数据分布；
2. 对应速度场的复杂程度；
3. 训练目标在不同时间步上的尺度；
4. 采样轨迹的曲率和数值积分难度。

因此，概率路径不是无关紧要的预处理步骤，而是生成模型设计空间中的重要自由度。

传统扩散模型通常从一个逐步加噪的过程出发，得到特定的概率路径；Flow Matching 则允许我们先选择希望学习的路径，再构造与之匹配的速度场。

这正是两种思路在建模顺序上的重要区别。

---

## 五、Flow Matching 的关键突破：从不可计算的边际速度到可计算的条件速度

现在，我们已经知道如何定义概率路径，也知道正确的速度场需要满足连续性方程。

接下来才进入 Flow Matching 最关键的部分：

**如何训练一个网络，使它学会推动整个概率分布沿着目标路径演化？**

### 5.1 条件速度场可以直接求出

先考虑高斯路径：

$$
X_t=a_tz+b_t\epsilon
$$

对时间求导，得到：

$$
\frac{dX_t}{dt}
=
\dot a_tz+\dot b_t\epsilon
$$

因此，条件速度场可以写成：

$$
\boxed{
u_t(x\mid z)
=
\dot a_tz+\dot b_t\epsilon
}
$$

这里的 $$\epsilon$$ 与 $$z$$ 通过当前状态满足：

$$
x=a_tz+b_t\epsilon
$$

在合适的时间区间内，可以由这个关系恢复对应的条件噪声，从而计算条件速度。

对于线性插值路径：

$$
X_t=(1-t)\epsilon+tz
$$

直接求导即可得到：

$$
\boxed{
u_t(X_t\mid z)=z-\epsilon
}
$$

这非常简单：给定一个目标数据样本和一份初始噪声，条件速度就是从噪声指向数据的位移。

而且它与 $$t$$ 无关。

但这还没有解决真正的问题。

### 5.2 为什么不能直接使用条件速度进行生成？

训练时，我们可以同时取得真实样本 $$z$$ 和随机噪声 $$\epsilon$$，因此可以构造出条件状态 $$X_t$$ 和对应的条件速度 $$u_t(X_t\mid z)$$。

然而，生成时我们并不知道应该对应哪个目标样本 $$z$$。

如果把真实数据样本 $$z$$ 固定后，直接沿条件速度运动，那么生成轨迹就会被引导至那个预先指定的目标。这不是我们希望的无条件生成过程。

真正需要的是边际速度场：

$$
u_t(x)
$$

它必须综合考虑所有可能的数据样本，并让整体分布沿着边际概率路径演化。

由条件速度得到边际速度的关键公式是：

$$
\boxed{
u_t(x)
=
\mathbb E
\left[
u_t(X_t\mid Z)
\mid X_t=x
\right]
}
$$

也就是说，给定当前状态 $$x$$，我们应该对所有可能的目标样本所对应的条件速度求后验加权平均。

进一步展开：

$$
u_t(x)
=
\int
u_t(x\mid z)
\frac{
p_t(x\mid z)p_{\mathrm{data}}(z)
}{
p_t(x)
}
\,dz
$$

其中：

$$
p(z\mid x,t)
=
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}
$$

是给定当前状态后的目标样本后验分布。

这个式子揭示了一个关键事实：边际速度场本身通常涉及一个复杂的后验积分，而这个积分一般无法直接计算。

### 5.3 为什么后验加权平均能够得到正确的速度场？

我们可以从连续性方程直接验证。

对于每个固定的 $$z$$，假设条件概率路径与条件速度场满足：

$$
\partial_t p_t(x\mid z)
+
\nabla\cdot
\left(
p_t(x\mid z)u_t(x\mid z)
\right)=0
$$

边际概率路径为：

$$
p_t(x)
=
\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz
$$

对时间求导：

$$
\partial_t p_t(x)
=
\int
\partial_t p_t(x\mid z)
p_{\mathrm{data}}(z)\,dz
$$

代入条件连续性方程：

$$
\partial_t p_t(x)
=
-\int
\nabla\cdot
\left(
p_t(x\mid z)u_t(x\mid z)
\right)
p_{\mathrm{data}}(z)\,dz
$$

在允许交换积分与散度运算的条件下：

$$
\partial_t p_t(x)
=
-\nabla\cdot
\left[
\int
p_t(x\mid z)u_t(x\mid z)
p_{\mathrm{data}}(z)\,dz
\right]
$$

根据边际速度的定义：

$$
p_t(x)u_t(x)
=
\int
p_t(x\mid z)u_t(x\mid z)
p_{\mathrm{data}}(z)\,dz
$$

所以：

$$
\boxed{
\partial_t p_t(x)
+
\nabla\cdot\left(p_t(x)u_t(x)\right)=0
}
$$

这就证明了：在适当正则性条件下，后验加权平均得到的边际速度场满足连续性方程，因此能够输运相应的边际概率路径。

这一步是整个框架的数学枢纽。它让我们能够从容易构造的条件路径出发，得到一个具有正确边际分布演化规律的速度场。

### 5.4 从 Flow Matching 到 Conditional Flow Matching

最直接的训练目标是让神经网络拟合边际速度：

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{t,X_t}
\left[
\left\|
u_\theta(t,X_t)-u_t(X_t)
\right\|^2
\right]
$$

其中：

$$
X_t\sim p_t
$$

问题在于，边际速度 $$u_t(x)$$ 包含前面提到的后验积分，因此通常难以直接计算。

解决方法是改为回归条件速度：

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{t,Z,X_t\mid Z}
\left[
\left\|
u_\theta(t,X_t)-u_t(X_t\mid Z)
\right\|^2
\right]
}
$$

这就是 Conditional Flow Matching（CFM）。

它的训练样本全部可以通过以下方式获得：

1. 从数据集中采样 $$Z$$；
2. 采样随机噪声；
3. 随机选择时间 $$t$$；
4. 构造条件状态 $$X_t$$；
5. 根据条件路径计算目标速度。

整个训练过程不需要先求解 ODE，也不需要显式计算边际速度场。

更重要的是，CFM 并不是随意替换训练目标。两种损失之间存在严格的数学关系。

### 5.5 为什么两个训练目标只差一个与参数无关的常数？

记：

$$
U=u_t(X_t\mid Z),\qquad
m=u_t(X_t)
$$

根据边际化公式：

$$
m=\mathbb E[U\mid X_t]
$$

因此，条件速度可以分解为：

$$
U=m+(U-m)
$$

对任意只依赖 $$X_t$$ 和 $$t$$ 的网络输出 $$v=u_\theta(t,X_t)$$，有：

$$
\begin{aligned}
\mathbb E\|v-U\|^2
&=
\mathbb E\|v-m-(U-m)\|^2\\
&=
\mathbb E\|v-m\|^2
+\mathbb E\|U-m\|^2\\
&\quad
-2\mathbb E\langle v-m,U-m\rangle
\end{aligned}
$$

最后一项为零，因为：

$$
\begin{aligned}
&\mathbb E[\langle v-m,U-m\rangle]\\
&=
\mathbb E\left[
\left\langle
v-m,\mathbb E[U-m\mid X_t]
\right\rangle
\right]\\
&=0
\end{aligned}
$$

所以：

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathcal L_{\mathrm{FM}}(\theta)+C
}
$$

其中：

$$
C=
\mathbb E\|U-m\|^2
$$

不依赖于模型参数 $$\theta$$。

这意味着两种目标具有相同的最优解。更准确地说，在无限表达能力和总体期望的理想条件下，它们具有相同的最优边际速度预测；在有限模型和有限数据的实际训练中，具体优化轨迹和泛化误差仍然可能不同。

这条恒等式解释了为什么我们能够使用简单、可采样的条件目标，去学习一个本身难以直接计算的边际速度场。

### 5.6 为什么训练不需要解 ODE？

以线性插值路径为例：

$$
X_t=(1-t)\epsilon+tZ
$$

其条件速度为：

$$
u_t(X_t\mid Z)=Z-\epsilon
$$

因此训练目标可以写成：

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E
\left[
\left\|
u_\theta(t,X_t)-(Z-\epsilon)
\right\|^2
\right]
}
$$

这就是一个普通的均方误差回归问题。

整个训练过程只需要采样数据、噪声和时间，并构造相应的监督目标。微分方程只在生成阶段用于求解。

这也是 Flow Matching 易于规模化训练的重要原因之一：训练目标不需要通过反复模拟整条生成轨迹才能获得。

不过，“不需要解 ODE”不代表训练没有其他成本。模型前向和反向传播、数据读取、分布式训练以及不同时间步的目标尺度仍然会影响实际效率。

到这里，我们已经建立了 Flow Matching 的主干逻辑：

**概率路径决定目标分布如何演化，连续性方程刻画正确的输运规律，条件速度场则提供了一个可以直接训练的目标。**

接下来，我们将看到，这套框架如何与扩散模型中常见的 Score、Denoiser 和 Noise Prediction 联系起来。

---

## 六、Score 与 Velocity：扩散模型和 Flow Matching 的数学桥梁

### 6.1 什么是 Score？

在扩散模型中，一个重要的学习对象是 Score Function：

$$
\boxed{
s_t(x)=\nabla_x\log p_t(x)
}
$$

它是概率密度对数关于输入的梯度。

直观上，Score 指向概率密度在局部增长最快的方向，因此能够为样本提供关于概率分布结构的信息。

但必须注意：Score 并不是简单地指向某个唯一的真实样本。对于多模态分布，一个点附近可能存在多个可能的数据来源，Score 描述的是整个分布的局部变化，而不是单个目标样本的位移。

为什么扩散模型会学习 Score？因为在适当的随机动力学下，Score 可以用于构造反向生成过程，让受噪声污染的样本逐步回到目标数据分布。

而在 Flow Matching 中，我们直接学习的是 Velocity。两者看起来不同，但在高斯路径下存在明确的换算关系。

### 6.2 从高斯条件分布推导 Score

继续使用：

$$
X_t=a_tZ+b_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
$$

在给定 $$Z=z$$ 的条件下：

$$
p_t(x\mid z)
=
\mathcal N(a_tz,b_t^2I)
$$

当 $$b_t>0$$ 时，其对数密度为：

$$
\log p_t(x\mid z)
=
-\frac d2\log(2\pi b_t^2)
-\frac{\|x-a_tz\|^2}{2b_t^2}
$$

对 $$x$$ 求梯度：

$$
\boxed{
s_t(x\mid z)
=
\nabla_x\log p_t(x\mid z)
=
-\frac{x-a_tz}{b_t^2}
}
$$

由于：

$$
x-a_tz=b_t\epsilon
$$

所以：

$$
\boxed{
s_t(x\mid z)=-\frac{\epsilon}{b_t}
}
$$

这说明，对于高斯条件路径，条件 Score 与条件噪声之间存在直接关系。

因此，预测噪声和预测 Score 并不是完全无关的两个任务：在适当的时间参数化下，它们能够相互转换。

### 6.3 推导 Velocity 与 Score 的关系

我们已经知道，条件速度为：

$$
u_t(x\mid z)
=
\dot a_tz+\dot b_t\epsilon
$$

利用：

$$
z=\frac{x-b_t\epsilon}{a_t}
$$

可以得到：

$$
u_t(x\mid z)
=
\frac{\dot a_t}{a_t}x
+
\left(
\dot b_t-\frac{\dot a_tb_t}{a_t}
\right)\epsilon
$$

再代入：

$$
\epsilon=-b_t s_t(x\mid z)
$$

得到：

$$
\boxed{
u_t(x\mid z)
=
\frac{\dot a_t}{a_t}x
+
\left(
\frac{\dot a_tb_t^2}{a_t}
-b_t\dot b_t
\right)s_t(x\mid z)
}
$$

对条件变量 $$z$$ 进行边际化，同样可以得到边际版本：

$$
\boxed{
u_t(x)
=
\frac{\dot a_t}{a_t}x
+
\left(
\frac{\dot a_tb_t^2}{a_t}
-b_t\dot b_t
\right)s_t(x)
}
$$

这里假设相应的系数在所讨论的时间区间内有定义，并且条件与边际量满足必要的可积性条件。

这个公式非常重要。

它表明，对于给定的高斯概率路径，Velocity 与 Score 可以通过已知的时间系数进行转换。只要相关系数不退化，知道其中一个，就可以恢复另一个。

因此，Flow Matching 和 Score-Based Diffusion 在这种路径下并不是两套互不相关的数学体系，而是可以通过不同的参数化方式描述相同的分布演化。

### 6.4 四种常见参数化：Velocity、Score、Denoiser 和 Noise

除了 Velocity 与 Score，实际模型还经常预测干净样本或噪声。

| 参数化 | 网络预测什么 | 直观含义 |
|---|---|---|
| Velocity Prediction | $$u_t(x)$$ | 当前状态应该朝哪个方向运动 |
| Score Prediction | $$\nabla_x\log p_t(x)$$ | 当前状态附近的概率密度如何变化 |
| Denoiser / Data Prediction | $$\hat z_\theta(x,t)$$ | 当前带噪样本对应的干净数据估计 |
| Noise Prediction | $$\hat\epsilon_\theta(x,t)$$ | 构造当前状态时所使用的噪声估计 |

对于前面定义的高斯路径：

$$
x=a_tz+b_t\epsilon
$$

只要相关系数非零，就可以通过代数关系在这些参数化之间转换。

例如：

$$
\epsilon=\frac{x-a_tz}{b_t}
$$

因此，在已知干净数据和当前状态的关系时，可以将干净数据预测转换为噪声预测，反之亦然。

但需要强调一个关键区别：

**在理想的函数空间中，某些参数化可以互相转换，并不意味着使用不同训练损失、时间权重和有限容量网络时，训练效果必然完全相同。**

因为参数化会影响目标的数值尺度、不同时间步的权重，以及模型误差在换算过程中如何被放大。

这也是为什么实际系统需要仔细选择预测目标，而不能只依据它们在理想数学条件下的等价性做决定。

### 6.5 为什么不同参数化会产生不同的训练行为？

假设模型预测 Score，真实 Score 为 $$s_t(x)$$，模型误差为：

$$
\delta s=\hat s_t(x)-s_t(x)
$$

由 Velocity 与 Score 的换算关系：

$$
u_t(x)
=
\frac{\dot a_t}{a_t}x
+
c_t s_t(x)
$$

其中：

$$
c_t=
\frac{\dot a_tb_t^2}{a_t}-b_t\dot b_t
$$

那么 Score 误差对应的 Velocity 误差为：

$$
\delta u=c_t\delta s
$$

所以：

$$
\|\delta u\|^2
=
c_t^2\|\delta s\|^2
$$

反过来，如果 $$c_t$$ 很小，那么从 Velocity 还原 Score 时，误差可能被显著放大。

同样，若噪声预测与速度预测之间的换算包含一个随时间变化且接近零的系数，那么某些时间步上的误差也会变得特别敏感。

因此，参数化选择影响的不只是网络最后一层输出什么，而是整个训练目标的条件数和有效时间权重。

在现代生成模型中，Velocity Prediction 受到广泛关注，一个重要原因正是它在某些常用路径下具有简洁的目标形式，并且能避免部分噪声预测参数化在特定时间区间中的误差放大问题。

不过，这不是一个适用于所有路径和所有模型的绝对结论。具体表现仍取决于路径设计、时间采样、损失权重、网络预条件化以及采样算法。

---

## 七、从确定性 Flow 到随机 Diffusion：Fokker–Planck 方程

到目前为止，我们主要讨论了 ODE。

但经典扩散模型还引入了一个关键元素：沿着时间演化过程持续注入随机噪声。

为了理解它与 Flow Matching 的关系，我们需要将确定性动力学推广到随机微分方程。

### 7.1 布朗运动：连续时间的随机扰动

布朗运动 $$W_t$$ 是一种连续时间随机过程，其增量满足：

$$
W_{t+h}-W_t
\sim\mathcal N(0,hI)
$$

并且不相交时间区间的增量相互独立。

因此，一个足够短的时间间隔内，布朗运动的增量可以写成：

$$
\Delta W_t=\sqrt h\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
$$

这里的 $$\sqrt h$$ 非常重要：随机增量的标准差随 $$\sqrt h$$ 缩放，而不是随 $$h$$ 缩放。

这意味着当我们将时间间隔缩短时，随机扰动并不会像普通的确定性误差那样按相同阶数衰减。

### 7.2 随机微分方程的形式

一般的 SDE 可以写成：

$$
\boxed{
dX_t=f_t(X_t)\,dt+g_t\,dW_t
}
$$

其中：

- $$f_t(x)$$ 是漂移项，描述确定性的平均运动趋势；
- $$g_t$$ 是扩散系数，控制随机扰动的强度；
- $$W_t$$ 是布朗运动。

与 ODE 不同，SDE 的轨迹除了受初始状态影响，还受沿途随机增量影响。因此，相同初始状态重复模拟时，也可能得到不同的终点。

当 $$g_t=0$$ 时，SDE 退化为确定性 ODE。

这说明 ODE 与 SDE 可以放在同一套动力学框架中讨论，但它们的随机性来源和密度演化规律并不相同。

### 7.3 Fokker–Planck 方程：随机动力学对应的密度演化规律

对于上述具有各向同性扩散系数的 SDE，概率密度满足 Fokker–Planck 方程：

$$
\boxed{
\partial_t p_t
=
-\nabla\cdot(f_tp_t)
+
\frac{g_t^2}{2}\Delta p_t
}
$$

其中：

$$
\Delta p_t=\nabla\cdot\nabla p_t
$$

是拉普拉斯算子。

与连续性方程相比，Fokker–Planck 方程多出了：

$$
\frac{g_t^2}{2}\Delta p_t
$$

这一项描述随机扩散对概率密度的影响。

如果把漂移项理解为推动概率质量整体移动，那么扩散项就会使概率密度在空间中进一步扩散、平滑。

当 $$g_t=0$$ 时，Fokker–Planck 方程退化为：

$$
\partial_t p_t
+
\nabla\cdot(f_tp_t)=0
$$

这正是前面得到的连续性方程。

所以，两种动力学的关键差别不仅在于是否存在随机噪声，还在于随机噪声会改变概率密度的演化方程。

### 7.4 能否在保持同一概率路径的情况下引入随机性？

现在考虑一个已经满足连续性方程的速度场 $$u_t(x)$$：

$$
\partial_t p_t
=
-\nabla\cdot(p_tu_t)
$$

我们希望构造一个 SDE，使它具有与原来相同的边际概率路径 $$p_t$$。

设新的 SDE 为：

$$
dX_t=f_t(X_t)\,dt+g_t\,dW_t
$$

根据 Fokker–Planck 方程，要使其边际密度仍然满足原来的连续性方程，就需要：

$$
-\nabla\cdot(p_tf_t)
+
\frac{g_t^2}{2}\Delta p_t
=
-\nabla\cdot(p_tu_t)
$$

利用：

$$
\Delta p_t
=
\nabla\cdot\left(p_t\nabla\log p_t\right)
$$

可以将扩散项写成：

$$
\frac{g_t^2}{2}\Delta p_t
=
\nabla\cdot
\left(
p_t\frac{g_t^2}{2}\nabla\log p_t
\right)
$$

因此，只需要选择：

$$
\boxed{
f_t(x)
=
u_t(x)+\frac{g_t^2}{2}s_t(x)
}
$$

其中：

$$
s_t(x)=\nabla_x\log p_t(x)
$$

就可以得到：

$$
\boxed{
dX_t
=
\left[
u_t(X_t)+\frac{g_t^2}{2}s_t(X_t)
\right]dt
+
g_t\,dW_t
}
$$

在适当的正则性和解的唯一性条件下，这个 SDE 与原来的 ODE 具有相同的边际概率路径。

注意这里的关键不是“随机噪声不会影响样本”，而是：**噪声确实改变了单条轨迹，但通过对漂移项进行相应修正，可以让整个随机群体保持相同的边际分布演化。**

这为理解概率流 ODE、扩散模型采样以及随机性调节提供了统一的数学解释。

### 7.5 为什么随机项不能随意添加？

假设我们直接在原来的 ODE 上添加随机噪声：

$$
dX_t=u_t(X_t)\,dt+g_t\,dW_t
$$

那么它的概率密度满足：

$$
\partial_t p_t
=
-\nabla\cdot(p_tu_t)
+
\frac{g_t^2}{2}\Delta p_t
$$

只要 $$g_t\neq0$$，一般就会多出一个扩散项。

因此，原来的概率路径通常不再成立。

这解释了为什么在保持边际概率路径的构造中，必须同时调整漂移项。随机噪声不是可以无代价添加的装饰，而是需要与概率密度演化规律相协调的动力学组成部分。

---

## 八、CFG：从条件生成到分布引导

前面主要讨论的是无条件生成，即从 $$p_{\mathrm{data}}$$ 中采样。

实际应用中，我们通常希望生成结果满足某个条件，例如给定一段文本提示 $$y$$，生成与其语义相符的图片。

数学上，这意味着目标从无条件分布转向条件分布：

$$
p_{\mathrm{data}}(x\mid y)
$$

模型不仅要生成合理的样本，还需要使样本与条件信息相匹配。

Classifier-Free Guidance（CFG）就是现代扩散模型中常用的一种引导方法。

### 8.1 条件模型与无条件模型

我们可以让神经网络显式接收条件 $$y$$：

$$
u_\theta(t,x,y)
$$

同时，也可以让模型在没有条件信息时预测无条件速度：

$$
u_\theta(t,x,\varnothing)
$$

其中 $$\varnothing$$ 表示空条件。

CFG 的核心是将这两个预测进行线性组合：

$$
\boxed{
u_{\mathrm{cfg}}
=
u_\theta(t,x,\varnothing)
+
w\left[
u_\theta(t,x,y)
-
u_\theta(t,x,\varnothing)
\right]
}
$$

这里 $$w$$ 是引导强度。

在这种约定下：

- $$w=0$$：只使用无条件预测；
- $$w=1$$：恢复条件模型的预测；
- $$w>1$$：沿着条件预测相对于无条件预测的差异方向进一步放大。

需要注意，不同文献和实现对引导尺度的记号可能不同。有些实现把条件预测作为基准项，因此其参数数值与这里的 $$w$$ 不一定相同。

### 8.2 为什么放大条件方向能够增强文本匹配？

我们可以从概率论解释这种组合。

对于给定的提示 $$y$$，贝叶斯公式给出：

$$
p_t(x\mid y)
=
\frac{p_t(x)p_t(y\mid x)}{p_t(y)}
$$

对 $$x$$ 取对数并求梯度：

$$
\nabla_x\log p_t(x\mid y)
=
\nabla_x\log p_t(x)
+
\nabla_x\log p_t(y\mid x)
$$

因此：

$$
\boxed{
s_t(x\mid y)
=
s_t(x)
+
\nabla_x\log p_t(y\mid x)
}
$$

这意味着条件 Score 可以理解为无条件 Score 加上一个条件相关的修正项。

如果有一个能够估计 $$p_t(y\mid x)$$ 的分类器，那么它的对数似然梯度就可以用来引导生成过程朝着更符合条件的方向运动。这是 Classifier Guidance 的基本思想。

CFG 则通过同一个模型分别预测有条件和无条件的结果，再利用两者的差异近似实现这种引导，而不需要单独训练一个额外的分类器。

在相应的参数化与概率路径条件下，Velocity 也可以使用类似的线性组合，因为 Velocity 与 Score 之间存在相应的线性关系。

### 8.3 CFG 到底在采样什么分布？

这一问题很容易被忽略。

在理想化条件下，假设无条件 Score 与条件 Score 都是精确的，那么 CFG 组合可以写成：

$$
s_{\mathrm{cfg}}
=
s_t(x)+w\left[s_t(x\mid y)-s_t(x)\right]
$$

由贝叶斯分解：

$$
s_t(x\mid y)-s_t(x)
=
\nabla_x\log p_t(y\mid x)
$$

因此：

$$
s_{\mathrm{cfg}}
=
\nabla_x\log p_t(x)
+
w\nabla_x\log p_t(y\mid x)
$$

合并后得到：

$$
\boxed{
s_{\mathrm{cfg}}
=
\nabla_x
\log\left[
p_t(x)\,p_t(y\mid x)^w
\right]
}
$$

忽略归一化常数，可以将其对应的密度写为：

$$
\boxed{
\tilde p_t(x)
\propto
p_t(x)\,p_t(y\mid x)^w
}
$$

这揭示了 CFG 的本质：它不仅仅是“让模型更加关注提示”，而是在理想化条件下，将条件似然以幂次 $$w$$ 重新加权。

当 $$w=1$$ 时：

$$
\tilde p_t(x)
\propto p_t(x)p_t(y\mid x)
\propto p_t(x\mid y)
$$

此时恢复了正确的条件分布。

当 $$w>1$$ 时，条件似然被进一步放大。模型会更加偏向那些更符合提示的样本区域，但这通常也会降低样本的多样性，并可能带来过饱和、细节失真等现象。

这里的结论需要一个重要限定：上述分布解释建立在精确 Score 以及相应条件关系之上。真实神经网络存在估计误差，实际离散采样也会引入数值误差，因此 CFG 的实际生成分布未必严格等于上述倾斜分布。

这也是为什么 CFG 更适合被理解为一种有效的引导启发式，而不是对原始条件分布进行无偏采样的保证。

---

## 九、把理论连接到工程：为什么路径设计和参数化如此重要？

前面的推导说明，Flow Matching、Score-Based Diffusion 和 CFG 可以在一套数学框架下讨论。

但理论上能够成立，并不意味着所有设计在工程上都同样好用。

在实际生成模型中，路径设计、时间采样、参数化和数值积分都会直接影响训练效率和生成质量。

### 9.1 为什么直线路径具有吸引力？

以线性插值为例：

$$
X_t=(1-t)\epsilon+tZ
$$

对应的条件速度是：

$$
u_t(X_t\mid Z)=Z-\epsilon
$$

它不依赖时间 $$t$$。

这带来了一个简单而明确的监督目标，也避免了条件速度本身随时间剧烈变化的问题。

另一方面，如果整体边际速度场对应的生成轨迹比较直，那么使用有限步数的数值积分就有机会更准确地跟踪轨迹。

需要强调的是，直线条件路径并不意味着边际路径一定直，也不意味着少步数采样必然准确。它的优势是提供了一个简洁的条件目标，而实际轨迹的几何性质仍然取决于边际化后的速度场。

Rectified Flow 的相关研究进一步探索了如何通过重新配对样本、重新训练等方式改善轨迹结构，以提高生成效率。

参考论文：

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482)

### 9.2 为什么时间采样会影响训练质量？

在 CFM 中，训练时通常需要采样时间 $$t$$。

一种简单的选择是：

$$
t\sim\mathcal U(0,1)
$$

但这并不意味着均匀采样在所有模型中都是最优的。

原因在于，不同时间步的状态难度、目标尺度和对最终生成质量的影响可能不同。

例如，在某些路径下：

- 接近噪声端时，模型面对的信息较少；
- 接近数据端时，样本可能已经包含较多数据结构；
- 中间区域可能更需要模型学习如何将噪声结构转换为数据结构。

如果训练过程中对所有时间步一视同仁，模型未必能获得最合适的优化效果。

因此，现代生成模型会探索不同的时间采样分布和损失权重，以重新分配训练资源。

Stable Diffusion 3 的相关研究就系统比较了多种训练配方，并报告了 Rectified Flow 与 Logit-Normal 时间采样等设计的效果。

参考论文：

[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)

### 9.3 为什么不能只看训练损失？

假设两个模型使用不同的预测参数化，但在理想数学条件下能够互相转换。

这并不意味着它们的训练损失数值可以直接比较。

原因包括：

1. 不同参数化具有不同的目标尺度；
2. 不同时间步的误差可能受到不同系数的放大或缩小；
3. 有限容量网络无法保证所有时间步同时达到理想最优解；
4. 训练损失与最终生成质量之间不存在普遍的一一对应关系。

因此，比较生成模型时，除了训练损失，还应关注采样步数、生成质量、条件一致性、样本多样性以及实际推理成本。

如果某种参数化在某些时间区间中会显著放大预测误差，那么即使它在理想函数空间中与其他参数化等价，也可能在有限算力和有限训练时间下表现得更差。

### 9.4 为什么 ODE 与 SDE 的采样表现可能不同？

在连续时间的理想条件下，我们可以构造具有相同边际概率路径的 ODE 和 SDE。

但是在实际系统中，两者的数值求解方式不同：

- ODE 采样误差来自确定性积分误差；
- SDE 采样误差还受到随机离散化的影响；
- 不同噪声强度可能改变有限步数下的误差表现；
- 不同采样器的计算开销和稳定性也不相同。

所以，理论上的边际分布不变性并不意味着任意噪声强度、任意步数和任意数值求解器都能得到完全相同的实际结果。

工程上需要在理论正确性、数值稳定性、生成质量和计算成本之间进行权衡。

---

## 十、总结：用一条主线理解 Flow Matching 与 Diffusion

现在，我们可以把整套理论重新串联起来。

首先，生成任务被定义为从简单分布采样，并通过某种动力学将其转换为目标数据分布。

接着，我们选择一条概率路径 $$p_t$$，规定从初始噪声到真实数据的整个分布演化过程。

然后，连续性方程将概率路径与速度场联系起来：

$$
\partial_t p_t
+
\nabla\cdot(p_tu_t)=0
$$

由于边际速度场通常包含难以计算的后验积分，Flow Matching 利用条件路径和条件速度构造可训练的监督目标，并通过 Conditional Flow Matching 的恒等式，保证其与原始 Flow Matching 目标具有相同的最优解。

对于高斯路径，Velocity、Score、Denoiser 和 Noise Prediction 又可以通过相应的代数关系互相转换。这让 Flow Matching 与扩散模型能够在统一框架下理解，同时也解释了不同参数化为什么会影响实际训练效果。

进一步地，通过 Fokker–Planck 方程，我们能够构造在边际分布意义下与确定性 Flow 相容的随机动力学；而 CFG 则利用有条件与无条件预测之间的差异，对生成过程施加更强的条件引导。

整套理论可以浓缩成四个核心认识：

**第一，生成模型学习的不只是去噪，而是概率分布的演化规律。**

去噪只是某些路径和参数化下的具体表现。更一般的目标是构造动力学，使简单分布能够演化为目标数据分布。

**第二，概率路径、速度场和神经网络参数化是三个不同的设计层次。**

路径决定我们希望分布如何变化；速度场描述如何输运分布；网络则近似这个速度场或与之等价的其他量。

**第三，Flow Matching 的核心价值在于将难以计算的边际速度学习，转化为可直接采样的条件监督问题。**

这不是简单地换一个损失函数，而是利用概率论与连续性方程建立的严格数学关系。

**第四，理论等价不等于工程表现完全相同。**

不同的路径、预测目标、时间权重和采样算法，即使在理想条件下能够互相转换，也可能在有限模型和有限计算资源下表现出明显差异。

理解了这些关系，就不必再把 Flow Matching、Diffusion、Score Matching、SDE 和 CFG 看作一组互不相关的技巧。它们分别处理的是概率路径设计、分布输运、函数学习、随机动力学和条件引导等不同层次的问题。

这正是理解现代生成模型最值得建立的整体视角。

---

## 参考文献与延伸阅读

以下论文分别对应本文的主要理论支柱，建议按照“Flow Matching → Score-Based Diffusion → Rectified Flow → CFG”的顺序阅读。

1. **Flow Matching 的基础理论**

   Lipman et al., *Flow Matching for Generative Modeling* (2023).

   https://arxiv.org/abs/2210.02747

2. **扩散模型的随机微分方程框架**

   Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (2021).

   https://arxiv.org/abs/2011.13456

3. **DDPM 与噪声预测目标**

   Ho et al., *Denoising Diffusion Probabilistic Models* (2020).

   https://arxiv.org/abs/2006.11239

4. **Rectified Flow 与直线路径**

   Liu, Gong, and Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* (2022).

   https://arxiv.org/abs/2209.03003

5. **最优传输与 Flow Matching 的结合**

   Tong et al., *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*.

   https://arxiv.org/abs/2302.00482

6. **Stable Diffusion 3 与 Rectified Flow Transformer**

   Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (2024).

   https://arxiv.org/abs/2403.03206

7. **Classifier-Free Guidance**

   Ho and Salimans, *Classifier-Free Diffusion Guidance* (2022).

   https://arxiv.org/abs/2207.12598

8. **Classifier Guidance**

   Dhariwal and Nichol, *Diffusion Models Beat GANs on Image Synthesis* (2021).

   https://arxiv.org/abs/2105.05233

9. **统一的随机插值视角**

   Albergo and Vanden-Eijnden, *Building Normalizing Flows with Stochastic Interpolants* (2023).

   https://arxiv.org/abs/2209.15571

10. **课程讲义**

    Peter Holderrieth and Ezra Erives, *An Introduction to Flow Matching and Diffusion Models*, MIT 6.S184 (2026).

    https://diffusion.csail.mit.edu/

## 上一篇 / 下一篇

- 上一篇：[变分自编码器之一：ELBO、重参数化与潜空间](/2026/10/13/vae-01-elbo-reparameterization-latent-space/)
- 下一篇：[扩散 Transformer 之一：架构与规模化](/2026/10/14/dit-01-patchify-adaln-zero-token-budget/)
