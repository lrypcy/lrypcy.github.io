---
title: "RL 符号与计算全景：GAE、优势估计、概率与重要性比的定义与实算"
date: 2026-09-19 15:00:00 +0800
categories:
  - 强化学习
tags: [rl, gae, ppo, grpo, advantage, importance-sampling, kl, rlhf, notation, estimator, survey]
layout: post
mathjax: true
---

> **系列导航** ｜ [从 MDP 到 GRPO（四）：PPO 与裁剪代理目标](/2026/08/21/mdp-to-grpo-04-ppo-clipped-surrogate/) ｜ [（五）：GRPO 组相对优势](/2026/08/21/mdp-to-grpo-05-grpo-group-relative/) ｜ [RL 信用分配与奖励系数](/2026/09/19/rl-credit-assignment-settlement/) ｜ **本篇是这三篇的“符号与计算”底座**

> **TL;DR**
>
> * **带帽 = 估计量，不带帽 = 真值；这条线必须在每个公式里守住（§2）**。$$\rho_t$$、$$\ell_t$$、$$\mathcal H_t$$ **不加帽**——给定模型它们能被精确算出来；$$\hat A$$、$$\hat{\mathbb D}_{\rm KL}$$、$$\hat R_t$$ **必须加帽**——分别因为函数近似、对指数级序列空间求和算不起、以及自举。而 $$\hat A$$ 上只有一顶帽子，却同时掩盖了 $$V_\phi\neq V^\pi$$ 和 $$\lambda<1$$ 两层近似。PPO 与 GRPO 的论文都把 $$A$$ 的帽子丢了，这就是坑一的根源。
> * **RL 的符号混乱不是风格问题，是数值问题**。同一个 $$A$$ 在 PPO 里是 GAE 估计（量纲 = 奖励），在 GRPO 里是组内 $$z$$ 分数（量纲 = 标准差），两者能差两个数量级。把 GRPO 的 advantage 直接喂给 PPO 的 value loss，critic 会立刻崩。
> * **GAE 的 $$\lambda$$ 主要买的是方差，不是偏差**。实测：$$\lambda$$ 从 0 到 1，优势估计的方差从 0.375 涨到 10.03（27 倍），而偏差项在完美 critic 下恒等于采样噪声地板。$$\lambda$$ 对偏差的影响**取决于 critic 误差的空间结构**——误差是“局部的”（critic 分辨不出相邻状态）时 $$\lambda\uparrow$$ 降偏差，误差是“全局缩放”时 $$\lambda\uparrow$$ 反而升偏差。
> * **$$\hat A_t + V(s_t)$$ 只有在 $$\lambda=1$$ 时才等于 return-to-go**。$$\lambda=0.95$$ 时它与真 $$G_t$$ 的相关系数是 0.954，$$\lambda=0$$ 时只有 0.087。也就是说，你喂给 critic 做回归的 target，在 $$\lambda<1$$ 时根本不是 $$G_t$$，而是一个被 $$\lambda$$ 截断的量——critic 拟合的是另一个函数。
> * **$$\gamma=1$$、确定性转移、仅终端奖励时（这正是 RLVR 的标准设定），GAE 有闭式解**：$$\hat A_t=-V_t+(1-\lambda)\sum_{j\ge1}\lambda^{j-1}V_{t+j}+\lambda^{T-1-t}R$$。$$\lambda=0$$ 时退化为“逐 token 价值增量” $$V_{t+1}-V_t$$，$$\lambda=1$$ 时退化为 $$R-V_t$$。若 critic 塌缩成常数 $$c$$，则 $$\hat A_t=\lambda^{T-1-t}(R-c)$$——**信号从句尾向句首指数衰减**，$$\lambda=0$$ 时只有最后一个 token 有梯度。
> * **KL 的 $$k_3$$ 估计器有个符号陷阱**：作为估计量它无偏且非负，是 [Schulman 2020](http://joschu.net/blog/kl-approx.html) 推荐的选择；但如果你把它当 loss 并且**不反向传播穿过它**（很多实现的做法），策略梯度的系数会从正确的 $$-\log r$$ 变成 $$k_3\approx\frac12(\log r)^2$$——当 $$r>1$$（策略比参考模型更不偏好该 token）时**符号直接相反**，惩罚变成了奖励。
> * **token 级重要性比在长序列上是病态的**：假设每 token 的 $$\log$$ ratio 是 $$\mathcal N(0.01,0.05^2)$$，序列长度 1024 时序列级 ratio 的中位数是 $$2.8\times10^4$$，4096 时是 $$6\times10^{17}$$。这就是为什么 GSPO 要把 ratio 换成序列级的几何平均 $$\big(\pi_\theta(o\mid q)/\pi_{\rm old}(o\mid q)\big)^{1/\lvert o\rvert}$$。
>
> 全文公式都配了可复现的数值验证（numpy，CPU 上几十秒），脚本见 [§14](#14-附录复现脚本)。

---

## 目录

- [0. 三个最容易踩的符号坑](#0-三个最容易踩的符号坑)
- [1. 符号与形状：一张表](#1-符号与形状一张表)
- [2. 帽子规则：什么时候加 hat](#2-帽子规则什么时候加-hat)
- [3. 五个价值量与它们之间的恒等式](#3-五个价值量与它们之间的恒等式)
- [4. 优势估计的家族谱](#4-优势估计的家族谱)
- [5. GAE：λ 到底在买什么](#5-gaeλ-到底在买什么)
- [6. off-policy 修正：重要性比家族](#6-off-policy-修正重要性比家族)
- [7. 概率侧：logprob、熵、温度、top-p](#7-概率侧logprob熵温度top-p)
- [8. KL：三种估计器与梯度符号陷阱](#8-kl三种估计器与梯度符号陷阱)
- [9. 重要性比：token 级 vs 序列级与 PPO 裁剪](#9-重要性比token-级-vs-序列级与-ppo-裁剪)
- [10. 归一化与量纲：GRPO / Dr.GRPO / whitening](#10-归一化与量纲grpo--drgrpo--whitening)
- [11. RLHF 特例：γ=1 下 GAE 的闭式解](#11-rlhf-特例γ1-下-gae-的闭式解)
- [12. 训练时要盯的监控量](#12-训练时要盯的监控量)
- [13. 速查总表与 17 个坑](#13-速查总表与-17-个坑)
- [14. 附录：复现脚本](#14-附录复现脚本)

---

## 0. 三个最容易踩的符号坑

**坑一：$$A$$ 不是同一个 $$A$$。**（本质是帽子问题，见 §2）

| 出处 | $$A$$ 的定义 | 量纲 | 典型数值范围 |
|:---|:---|:---|:---|
| PPO / TRPO（[Schulman et al. 2015](https://arxiv.org/abs/1506.02438)、[2017](https://arxiv.org/abs/1707.06347)） | $$\hat A_t^{\rm GAE}$$ | 奖励的单位 | 取决于 reward scale，可能 $$\pm 10$$ |
| GRPO（[DeepSeekMath](https://arxiv.org/abs/2402.03300)） | $$(R_i-\hat\mu_g)/\hat\sigma_g$$ | 无量纲（$$z$$ 分数） | $$[-2.65, 2.65]$$，$$G=8$$ 二值奖励 |
| Dr.GRPO（[2503.20783](https://arxiv.org/abs/2503.20783)） | $$R_i-\hat\mu_g$$ | 奖励的单位（不除 std） | $$[-1, 1]$$，二值奖励 |

注意这三行**都是带帽的估计量**（原论文里都没加帽，这是混乱的起点）。真正不带帽的 $$A^\pi=Q^\pi-V^\pi$$ 你永远拿不到。

把 GRPO 的 advantage 拿去算 value loss 或者和 KL 项直接相加，量纲立刻打架。

**坑二：$$\pi_{\rm old}$$ 与 $$\pi_{\rm ref}$$ 是两个东西。**

- $$\pi_{\rm old}$$（也叫 $$\pi_{\theta_{\rm old}}$$、behavior policy $$\mu$$）：**生成当前这批 rollout 的那份策略**，每轮 rollout 刷新一次。只出现在重要性比 $$r_t=\pi_\theta/\pi_{\rm old}$$ 里。
- $$\pi_{\rm ref}$$：**RL 开始前的 SFT 模型**，全程冻结。只出现在 KL 惩罚 $$\mathbb D_{\rm KL}(\pi_\theta\Vert\pi_{\rm ref})$$ 里。

TRPO 的信任域约束是对 $$\pi_{\rm old}$$ 的；RLHF 的 KL 惩罚是对 $$\pi_{\rm ref}$$ 的。混用会让裁剪失效或者把模型往错误的锚点拉。

**坑三：$$\gamma$$ 和 $$\lambda$$ 管的是两件事。**

- $$\gamma$$（折扣）：决定**你在优化哪个目标**。$$\gamma<1$$ 时优化的是折扣回报，梯度是“biased for the true undiscounted objective”——Schulman 在 GAE 论文里专门用“$$\gamma$$-just”这个词强调这一点。RLHF 里通常取 $$\gamma=1$$（有限 horizon，不需要折扣）。
- $$\lambda$$（GAE 参数）：决定**你用什么估计量去逼近那个目标的梯度**。$$\gamma$$ 固定不动时，$$\lambda$$ 仍然要调。

---

## 1. 符号与形状：一张表

LLM RL 里所有量的“时间维”就是 token 位置，状态转移是**确定性的**（$$s_{t+1}=s_t$$ 拼上 $$a_t$$），这一点后面会反复用到。

| 符号 | 定义 | LLM 场景对应 | Tensor 形状 |
|:---|:---|:---|:---|
| $$q,\ x$$ | prompt / 条件 | 输入问题 | `[B]` |
| $$o,\ y$$ | 生成的回答 | 一条 rollout | `[B, T]` |
| $$s_t$$ | 状态 | $$(q, y_{<t})$$ 拼起来的上下文 | — |
| $$a_t$$ | 动作 | 第 $$t$$ 个 token $$y_t$$ | `[B]` |
| $$T,\ \vert o\vert$$ | 回答长度 | — | `[B]` |
| $$\pi_\theta(a_t\mid s_t)$$ | 策略 | next-token 分布 | `[B, T, V]` |
| $$\log\pi_\theta(y_t\mid q,y_{<t})$$ | token 对数概率 | 由 logits 的 log-softmax 取对应项 | `[B, T]` |
| $$r_t$$ | 即时奖励 | RLVR 里只有 $$t=T-1$$ 非零；KL shaping 时每步非零 | `[B, T]` |
| $$R,\ R(o)$$ | 轨迹总奖励 / 终端奖励 | 答案对=1，错=0 | `[B]` |
| $$\gamma$$ | 折扣因子 | RLHF 常取 1 | 标量 |
| $$\lambda$$ | GAE 参数 | 0.95（PPO 默认） | 标量 |
| $$G_t$$ | return-to-go $$\sum_{l\ge0}\gamma^l r_{t+l}$$ | — | `[B, T]` |
| $$V_\phi(s_t)$$ | 状态价值（critic） | “这个前缀最终能拿多少分” | `[B, T]` |
| $$Q^\pi(s_t,a_t)$$ | 动作价值 | — | `[B, T]` |
| $$A^\pi(s_t,a_t)$$ | 优势 $$=Q^\pi-V^\pi$$ | — | `[B, T]` |
| $$\delta_t$$ | TD error $$=r_t+\gamma V(s_{t+1})-V(s_t)$$ | — | `[B, T]` |
| $$\hat A_t$$ | 优势的估计量 | GAE / 组相对 / RLOO … | `[B, T]` |
| $$r_t(\theta)$$ 或 $$\rho_t$$ | 重要性比 $$=\pi_\theta(a_t\mid s_t)/\pi_{\rm old}(a_t\mid s_t)$$ | — | `[B, T]` |
| $$\epsilon$$ | PPO 裁剪半径 | 0.2（policy），0.1~0.5（value） | 标量 |
| $$\beta$$ | KL 系数 | 0.001~0.1 | 标量 |

> 注意 $$r_t$$ 这个字母在文献里既被用作**奖励**又被用作**重要性比**。本文用 $$r_t$$ 表示奖励，用 $$\rho_t$$ 表示重要性比。

---

## 2. 帽子规则：什么时候加 hat

RL 的符号混乱有一半来自同一个字母在“真值”和“估计”之间来回切换。$$\hat A$$ 与 $$A$$ 不是书写习惯的差别，它们**是两种不同的数学对象**：一个是随机变量，一个是常数。这一节给出本文统一遵守的帽子规则，后面所有公式都按它来读。

### 2.1 一句话判据

$$\hat x = \text{你从有限数据里算出来的那个数（随机变量，有分布）},\qquad x = \text{定义出来的那个量（真值，通常不可知）}$$

从 $$x$$ 走到 $$\hat x$$ 有三个“污染源”，RL 三个都有，监督学习只有前两个：

| 污染源 | 具体表现 | 监督学习里有吗 |
|:---|:---|:---|
| ① 有限样本 | 用 $$\frac1N\sum_i$$ 代替 $$\mathbb E$$ | 有 |
| ② 函数近似 | 用 $$V_\phi$$ 代替 $$V^\pi$$ | 有（用 $$f_\theta$$ 代替 $$f^*$$） |
| ③ **自举** | 用估计量自己当训练目标（$$\hat R=\hat A+V_\phi$$ 拿去训练 $$V_\phi$$） | **没有** |

第三个是 RL 特有的，也是 RL 里“帽子”最容易被写漏的地方——见 §2.5。

### 2.2 判据的推论：帽子标的是“算不算得出来”，不只是“随不随机”

最反直觉的一条：**下面这些量永远不加帽，尽管它们作用在采样出来的 token 上**。

$$\ell_t=\log\pi_\theta(y_t\mid q,y_{<t}),\qquad \rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm old}(a_t\mid s_t)},\qquad \mathcal H_t=-\sum_v p_v\log p_v$$

- $$\ell_t$$：给定模型和 token，它是**模型前向的确定性输出**，不是估计。
- $$\rho_t$$：两个分布的**精确比值**，$$\rho_t=1$$ 当且仅当策略没变。
- $$\mathcal H_t$$：对整个词表求和，128k 维一次前向就算完，**精确**。

对照：**KL 必须加帽**。

$$\mathbb D_{\rm KL}(\pi_\theta\Vert\pi_{\rm ref})=\sum_{o}\pi_\theta(o)\log\frac{\pi_\theta(o)}{\pi_{\rm ref}(o)}\ \text{—— 要对长度为 }T\text{ 的序列空间求和，}V^T\text{ 项，算不动}$$

于是只能采样估计，于是必须有 $$\hat{\mathbb D}_{\rm KL}$$。注意这里的分界线不是“精确 vs 不精确”，而是**算得起 vs 算不起**：熵是在 128k 维上求一次和（算得起），KL 是在指数级序列空间上求和（算不起）。**如果你哪天用采样 softmax 去估熵，那个熵也要加帽变成 $$\hat{\mathcal H}$$**。

同理，$$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$$：给定 $$V$$ 和一条轨迹，它是确定的数，**不加帽**；但实践里 $$V$$ 是 $$V_\phi$$（学出来的），所以你手上永远是 $$\delta_t^{V_\phi}$$——本文在强调“这是估计”时写成 $$\hat\delta_t$$，在只讨论代数恒等式时写 $$\delta_t$$。

### 2.3 全篇帽子清单

| 量 | 无帽（真值 / 定义 / 可精确计算） | 有帽（估计 / 经验） | 加帽的理由 |
|:---|:---|:---|:---|
| 优势 | $$A^\pi(s,a)=Q^\pi-V^\pi$$ | $$\hat A_t^{\rm GAE}$$、$$\hat A_i^{\rm GRPO}$$、$$\hat A_i^{\rm RLOO}$$ | 依赖未知的 $$Q^\pi,V^\pi$$ |
| 状态价值 | $$V^\pi$$ | $$V_\phi$$（带下标 $$\phi$$ 时不再加帽） | ② 函数近似 |
| 回报 | $$G_t$$（给定轨迹后是确定的数） | 当它充当 $$A$$ 的估计量时，写成 $$\hat A_t=G_t-\hat b$$ | 角色决定，见 §2.4 ② |
| TD error | $$\delta_t$$（对给定 $$V$$ 精确） | $$\hat\delta_t=\delta_t^{V_\phi}$$ | ② |
| 期望 | $$\mathbb E_\pi[\cdot]$$ | $$\hat{\mathbb E}=\frac1N\sum_i$$ | ① 有限样本 |
| 策略梯度 | $$\nabla_\theta J(\theta)$$ | $$\hat g$$ | ① + $$\hat A$$ |
| KL | $$\mathbb D_{\rm KL}$$ | $$\hat{\mathbb D}_{\rm KL}$$（实现里就是 $$k_3$$ 的平均） | 算不起 |
| value target | — | $$\hat R_t=\hat A_t+V_\phi(s_t)$$ | 由估计量构造 |
| **logprob / 重要性比 / 熵** | — | $$\ell_t,\ \rho_t,\ \mathcal H_t$$ **无帽** | 精确可算 |
| Bellman 方程 | $$V^\pi(s)=\mathbb E_\pi[r+\gamma V^\pi(s')]$$ | 更新式 $$\hat V\leftarrow\hat V+\alpha(\hat R-\hat V)$$ | 见 §2.4 ① |

### 2.4 逐条公式里的帽子

**① Bellman 方程永远不带帽**

$$V^\pi(s)=\mathbb E_\pi\big[r+\gamma V^\pi(s')\big]\quad\text{（定义，恒等式，描述真值的不动点）}$$
$$\hat V(s)\leftarrow \hat V(s)+\alpha\big(\hat R_t-\hat V(s)\big)\quad\text{（更新式，全带帽）}$$

把帽子写进 Bellman 方程是概念错误：那个方程说的是**真值满足什么关系**，不是你手上的数满足什么关系。你手上的 $$\hat V$$ 一般并不满足它——差值就是 TD error，而 TD error 之所以非零，正是因为你还在用估计量。

**② $$G_t$$ 带不带帽，取决于它在句子里扮演什么角色**

- 作为“这条轨迹的折扣回报”：$$G_t$$ 是**确定的数**（轨迹已经采到了），无帽；
- 作为“$$A^\pi(s_t,a_t)$$ 的蒙特卡洛估计”：$$\hat A_t^{\rm MC}=G_t-V(s_t)$$，**有帽**——因为它的随机性来自“这条轨迹是从 $$\pi$$ 里随机采出来的”，跨轨迹它会变。

同一个符号、两种身份，这是文献里最常被写混的一处。判据是：**你在对什么取期望？** 如果对轨迹取期望，那它就是随机变量，加帽。

**③ GAE 这条链上，帽子掩盖了两层近似**

$$A^\pi(s_t,a_t)\ \xrightarrow[\ \text{② 函数近似}\ ]{V^\pi\to V_\phi}\ \hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}^{V_\phi}\ \xrightarrow[\ \text{① 有限样本}\ ]{}\ \hat g=\frac1N\sum_{i,t}\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\,\hat A_{i,t}$$

**$$\hat A$$ 上只有一顶帽子，但它同时掩盖了 $$V_\phi\neq V^\pi$$ 和 $$\lambda<1$$ 两层近似**，从公式上根本看不出偏差来自哪一层。这就是为什么 §5 必须单独做偏差-方差分解、并且要构造四种不同的 critic 误差形态——光看 $$\hat A$$ 的公式你什么也判断不了。

**④ PPO 的裁剪目标：混着戴**

$$L^{\rm CLIP}=\mathbb E\Big[\min\big(\rho_t\,\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\Big]$$

- $$\rho_t$$ **无帽**：两个模型的精确比值（§2.2）；
- $$\hat A_t$$ **有帽**：估计出来的；
- 外层 $$\mathbb E$$ **无帽**：这是**目标函数的定义**；只有当你把它换成 $$\frac1N\sum$$ 去做 minibatch 更新时，才变成估计 $$\hat L$$。

也就是说：PPO 的“目标”和“你每步实际在优化的那个数”是两个东西，差一顶帽子。

**⑤ GRPO：baseline 带不带帽不是重点，独立不独立才是**

$$\hat A_i=\frac{R_i-\hat\mu_g}{\hat\sigma_g},\qquad \hat\mu_g=\frac1G\sum_{j=1}^{G}R_j,\qquad \hat\sigma_g=\sqrt{\frac1G\sum_j(R_j-\hat\mu_g)^2}$$

$$\hat\mu_g,\hat\sigma_g$$ 都带帽（经验统计量）。真正要命的不是帽子，而是**$$\hat\mu_g$$ 里包含了 $$R_i$$ 自己**，于是 $$\hat A_i$$ 与分子 $$R_i$$ 相关，梯度有 $$O(1/G)$$ 的偏差。RLOO 换成留一法均值 $$\hat\mu_{-i}=\frac{1}{G-1}\sum_{j\ne i}R_j$$ 后，baseline 与 $$R_i$$ 独立，估计**严格无偏**。

$$\text{baseline 的帽子不是重点，}\ \mathrm{Cov}(\hat b,\,R_i)\ \text{才是}$$

**⑥ KL 的帽子落在哪一层**

$$\mathbb D_{\rm KL}=\mathbb E_{o\sim\pi_\theta}\big[-\log r\big]\ \text{（无帽，定义）}\ \longrightarrow\ \hat{\mathbb D}_{\rm KL}=\frac1M\sum_{m=1}^{M}k_3^{(m)}\ \text{（有帽）}$$

$$k_3^{(m)}$$ 是**单个样本上的估计量**，不是真值；$$k_1,k_2,k_3$$ 的区别（§8.1）是“用哪个函数当这个估计量”，与帽子在哪一层无关。

### 2.5 嵌套帽子：RL 特有的自举循环

监督学习里，标签是“真的”，$$y$$ 不带帽。RL 里没有这种东西：

$$\underbrace{\hat R_t}_{\text{训练 critic 的 target}}=\underbrace{\hat A_t}_{\text{用 }V_\phi\text{ 算的}}+\underbrace{V_\phi(s_t)}_{\text{被训练的那个函数}}$$

target 由估计量构造，而这个 target 又要拿去训练构造它的那个函数。**这是一个固定点迭代，不是一次回归**。后果：

- $$\hat R_t$$ 必须带帽——它看起来像标签，其实不是；
- 但训练时我们把它**当标签用**（算 $$\ell_2$$ 损失），这是 bootstrap 的定义，不是笔误；
- 于是 $$V_\phi$$ 的“偏差”会循环放大：偏高的 $$V$$ → 偏低的 $$\hat A$$ → 偏低的 $$\hat R$$ → 拉低 $$V$$…… 收敛性靠 $$\gamma<1$$ 和 $$\lambda$$ 的收缩性保证（V-trace 论文专门证明了这一点，见 §6）。

### 2.6 一顶帽子不够时：用 tilde，不要叠帽子

| 记号 | 含义 | 例子 |
|:---|:---|:---|
| $$\hat{\ }$$ | 估计（随机性来自**数据**） | $$\hat A,\ \hat{\mathbb D}_{\rm KL},\ \hat g$$ |
| $$\tilde{\ }$$ | 变换（**确定性**后处理，不再引入新的随机性来源） | $$\tilde A_i=(\hat A_i-\mu)/\sigma$$（whitening 后的版本） |
| 下标 $$\phi,\theta$$ | 参数化（“用 $$\phi$$ 表示的那个 $$V$$”） | $$V_\phi,\ \pi_\theta$$ —— 加了下标就不再加帽 |

写成 $$\hat{\hat A}$$ 是不可读的。如果你叠加了“估计 + 归一化 + 再裁剪”三件事，用文字说明而不是叠符号。

### 2.7 三问判定法

遇到一个量，按顺序问：

1. **它是定义/恒等式的一侧吗？**（如 $$A^\pi=Q^\pi-V^\pi$$、Bellman 方程）→ 不加帽；
2. **它能被精确算出来吗？**（$$\ell_t,\rho_t,\mathcal H_t$$）→ 不加帽；
3. **它的随机性来自哪里？** 有限样本 → $$\hat{\ }$$；学出来的函数 → 用下标 $$\phi$$ 或 $$\hat{\ }$$；自举 target → $$\hat{\ }$$。

### 2.8 文献现状：为什么你读论文时会被搞混

| 出处 | 写法 | 实际含义 | 按本文规则 |
|:---|:---|:---|:---|
| GAE 论文 | 严格区分 $$A^{\pi,\gamma}$$ 与 $$\hat A_t$$ | 真值 / 估计 | 一致 ✅ |
| PPO 论文 | 全文写 $$A_t$$（**无帽**） | 实际是 GAE 估计量 | 应为 $$\hat A_t$$ ❌ |
| GRPO / DeepSeekMath | 写 $$A_i$$（**无帽**） | 实际是经验组统计量 | 应为 $$\hat A_i$$ ❌ |

同一个 $$A$$，三篇顶会论文三个意思——这就是 §0 坑一的根源。本文**凡是估计量一律加帽**，读到不带帽的 $$A^\pi$$、$$V^\pi$$、$$\mathbb D_{\rm KL}$$ 时，它们指的都是你拿不到的那个真值。

### 2.9 落到代码里

verl / TRL 的字段命名其实已经隐含了帽子规则，建议沿用：

| 代码字段 | 符号 | 带帽 |
|:---|:---|:---|
| `advantages` | $$\hat A_t$$ | ✅ |
| `returns` | $$\hat R_t=\hat A_t+V_\phi$$ | ✅（它是 target，不是真回报） |
| `values` | $$V_\phi(s_t)$$ | 下标 $$\phi$$ |
| `old_log_probs` | $$\ell_t^{\rm old}$$ | ❌（精确） |
| `ref_log_probs` | $$\ell_t^{\rm ref}$$ | ❌（精确） |
| `token_level_rewards` | $$r_t$$ | ❌（环境给的） |
| `kl_penalty` / `kld` | $$\hat{\mathbb D}_{\rm KL}$$ 的逐 token 项 | ✅ |

自己写 estimator 时，变量名里保留 `hat_` 前缀或写清注释——半年后你会忘记 `adv` 到底是 GAE 还是组相对。

---

## 3. 五个价值量与它们之间的恒等式

先定义清楚，再谈估计。**这一节里所有符号都不带帽**——它们全是定义出来的真值，你拿不到，只能估计：

$$V^\pi(s_t)=\mathbb E_\pi\!\left[G_t\mid s_t\right],\qquad Q^\pi(s_t,a_t)=\mathbb E_\pi\!\left[G_t\mid s_t,a_t\right],\qquad A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)$$

等你把它们换成“用 $$V_\phi$$ 算的、用 $$N$$ 条轨迹平均的”，才变成 §4 那些带帽的量。

三条恒等式，后面的所有估计量都是它们的推论：

**(1) 优势的零均值性质**

$$\mathbb E_{a\sim\pi(\cdot\mid s)}\!\left[A^\pi(s,a)\right]=0$$

这是“baseline 不改变梯度期望”的根源：任何只依赖 $$s$$ 的 $$b(s)$$ 都可以从回报里减掉而保持无偏。

注意这条恒等式**只对无帽的 $$A^\pi$$ 成立**。你手上的 $$\hat A$$ 一般**不满足** $$\mathbb E[\hat A]=0$$——$$\hat A^{\rm GAE}$$ 在 $$V_\phi\neq V^\pi$$ 时有偏，$$\hat A^{\rm GRPO}$$ 因为 baseline 含自身而均值非零、方差被 $$\hat\sigma_g$$ 强行改成 1。**“组内 advantage 均值为零”是构造出来的，不是定理**。

**(2) TD error 的恒等式（对任意 $$V$$ 成立，不要求 $$V=V^\pi$$）**

$$G_t - V(s_t)=\sum_{l=0}^{\infty}\gamma^l\,\delta_{t+l},\qquad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$$

证明就是望远镜求和：$$\sum_l\gamma^l\delta_{t+l}=\sum_l\gamma^l r_{t+l}+\sum_l\gamma^{l+1}V(s_{t+l+1})-\sum_l\gamma^l V(s_{t+l})=G_t-V(s_t)$$。

推论：**GAE 在 $$\lambda=1$$ 时就是 $$G_t-V(s_t)$$**，这正是 Schulman 论文里的式 (10)。

**(3) 确定性转移下的简化**

LLM 生成中 $$s_{t+1}=(s_t,a_t)$$ 是确定的，所以

$$Q^\pi(s_t,a_t)=r_t+\gamma V^\pi(s_{t+1})\quad\Longrightarrow\quad A^\pi(s_t,a_t)=r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)=\delta_t^{\pi}$$

**在确定性环境里，“优势”就是 TD error 的真值**。这是 LLM RL 与经典 RL 最重要的结构差异，也是 §11 的出发点。

---

## 4. 优势估计的家族谱

所有 critic-free 方法都可以写成同一个模板（[GPO 统一框架，2503.19523](https://arxiv.org/abs/2503.19523) 的表 1）：先选一个重要性权重函数 $$\omega(\cdot)$$，再选一个 baseline $$B$$，最后得到 $$A(\cdot)$$。**下面这张表里所有的 $$A$$ 都是估计量，原论文统一不加帽，本文一律写成 $$\hat A$$**：

| 估计量 | $$\omega(\cdot)$$ | 裁剪 | baseline $$B$$ | $$\hat A$$ | 出处 |
|:---|:---|:---|:---|:---|:---|
| REINFORCE | $$1$$ | — | $$0$$ | $$R(\mathbf y^i)$$ | [Williams 1992](https://link.springer.com/article/10.1007/BF00992696) |
| RLOO | $$1$$ | — | $$\hat\mu_{-i}=\frac{1}{N-1}\sum_{j\ne i}R(\mathbf y^j)$$ | $$R(\mathbf y^i)-\hat\mu_{-i}$$ | [2402.14740](https://arxiv.org/abs/2402.14740) |
| ReMax | $$1$$ | — | $$\hat b=R(\mathbf y)^{\rm greedy}$$ | $$R(\mathbf y^i)-\hat b$$ | [2310.10505](https://arxiv.org/abs/2310.10505) |
| GRPO | $$\mathrm{clip}(\cdot)$$ | $$[1-\epsilon,1+\epsilon]$$ | $$\hat\mu_g$$ | $$(R(\mathbf y^i)-\hat\mu_g)/\hat\sigma_g$$ | [2402.03300](https://arxiv.org/abs/2402.03300) |
| REINFORCE++ | $$\mathrm{clip}(\cdot)$$ | $$[1-\epsilon,1+\epsilon]$$ | $$\hat\mu_g$$ | $$\gamma^{T_{\max}-t}(R(\mathbf y^i)-\hat\mu_g)+\sum_{k\ge t}\gamma^{k-t}\log\frac{\pi_\theta}{\pi_{\rm ref}}$$ | [2501.03262](https://arxiv.org/abs/2501.03262) |
| PPO / GAE | $$\mathrm{clip}(\cdot)$$ | $$[1-\epsilon,1+\epsilon]$$ | $$V_\phi(s_t)$$ | $$\sum_l(\gamma\lambda)^l\hat\delta_{t+l}$$ | [1506.02438](https://arxiv.org/abs/1506.02438) |

注意 RLOO 的 baseline 用**留一法均值**（$$\hat\mu_{-i}$$），这样 $$R(\mathbf y^i)$$ 不出现在自己的 baseline 里，$$\mathrm{Cov}(\hat\mu_{-i},R_i)=0$$，梯度**严格无偏**；GRPO 用的是含自身的 $$\hat\mu_g$$，有 $$O(1/G)$$ 的偏差，换来的是 std 归一化带来的稳定性。**区别不在帽子，在于 baseline 与 $$R_i$$ 是否独立**（§2.4 ⑤）。

ReMax 的 baseline 是**贪婪解码的奖励**，它不依赖采样轨迹，因此也是独立于 $$R_i$$ 的——但它是 $$\pi_\theta$$ 的确定性函数，会随训练变化，所以严格说是“条件独立”。

基于 critic 的估计量单独一张表（这是本文的重点，全部加帽）：

| 估计量 | 公式 | 需要 critic | $$\lambda$$ 的角色 |
|:---|:---|:---|:---|
| MC / REINFORCE | $$\hat A_t=G_t-V(s_t)$$（或 $$G_t$$） | 可选 | — |
| TD(0) | $$\hat A_t=\hat\delta_t$$ | 是 | 等价于 $$\lambda=0$$ |
| $$n$$-step | $$\hat A_t=\sum_{l=0}^{n-1}\gamma^l r_{t+l}+\gamma^n V(s_{t+n})-V(s_t)$$ | 是 | — |
| $$\mathrm{GAE}(\gamma,\lambda)$$ | $$\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\hat\delta_{t+l}$$ | 是 | 偏差-方差旋钮 |
| $$\mathrm{TD}(\lambda)$$（**估价值**用） | 后向视图的资格迹累积 | 是 | 估 $$V$$ 而非估 $$A$$ |
| $$\mathrm{Q}(\lambda)$$ / SARSA($$\lambda$$) | 离线修正版 | 是 | 见 §6 |
| V-trace | 见 §6 | 是 | off-policy 修正 |

> GAE 与 TD($$\lambda$$) 用的是同一个 $$\gamma\lambda$$ 折扣，但**目标不同**：TD($$\lambda$$) 估计的是 $$V$$，GAE 估计的是 $$A$$。名字像，不能换。

---

## 5. GAE：$$\lambda$$ 到底在买什么

**先把帽子戴上**：本节所有 $$\hat A$$ 都是 $$A^\pi$$ 的估计量，所有 $$\hat\delta$$ 都是用 $$V_\phi$$ 算的（不是真值 $$\delta^\pi$$）。GAE 论文把这一点写得很清楚——它定义的是 $$\hat A^{\rm GAE(\gamma,\lambda)}_t$$，并单独证明了“当 $$V=V^{\pi,\gamma}$$ 时它是 $$\gamma$$-just 的”；很多派生工作把这个帽子丢了，于是读者会误以为 $$\lambda$$ 改变的是目标本身，其实它改变的只是**估计量**。

$$A^\pi(s_t,a_t)\ \xrightarrow{\ V^\pi\to V_\phi\ }\ \hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\hat\delta_{t+l},\qquad \hat\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$$

帽子只有一顶，但掩盖了两层近似（$$V_\phi\neq V^\pi$$ 与 $$\lambda<1$$），所以下面必须做分解才能看清楚。

### 5.1 递归式与展开式是同一个东西（数值验证）

实践中都写反向递归：

$$\hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1}$$

它与定义式 $$\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$$ 完全等价。实测（随机 MDP，$$S=10,A=3,\gamma=0.95,T=30$$，1200 条轨迹）：

```
lambda=0.00 : max|递归 - 展开| = 0.000e+00
lambda=0.50 : max|递归 - 展开| = 8.882e-16
lambda=0.95 : max|递归 - 展开| = 1.776e-15
lambda=1.00 : max|递归 - 展开| = 4.441e-15
```

纯浮点误差，两种写法随便选。**但边界处理必须一致**：序列末尾要 bootstrap（用 $$V(s_T)$$，episode 终止时置 0），且跨 episode 不能把 $$\hat A_{t+1}$$ 传过去。

### 5.2 偏差-方差分解：四种 critic 误差形态

设置：随机 MDP（$$S=10,A=3,\gamma=0.95,T=30$$），精确解出 $$V^\pi,Q^\pi$$，得到真值 $$A^\pi(s,a)$$（$$\mathbb E_\pi[A]=0$$ 已校验）。对每个 $$(s,a)$$ 采样 1200 条轨迹，逐 $$(s,a)$$ 先算 bias 与 var 再对状态求平均（**跨状态直接平均会让正负偏差抵消，这是最容易犯的统计错误**）。

**① 完美 critic（$$V_\phi=V^\pi$$）**

| $$\lambda$$ | bias$$^2$$ | var | MSE | 噪声地板 var$$/N$$ |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0003 | 0.3754 | 0.3756 | 0.0003 |
| 0.60 | 0.0008 | 0.9023 | 0.9032 | 0.0008 |
| 0.90 | 0.0040 | 3.3667 | 3.3707 | 0.0028 |
| 0.95 | 0.0062 | 5.1892 | 5.1954 | 0.0043 |
| 1.00 | 0.0112 | 10.0265 | 10.0377 | 0.0084 |

bias$$^2$$ **恰好等于噪声地板**（$$N$$ 条样本均值的标准误平方），说明理论上确实无偏——$$V=V^\pi$$ 时任意 $$\lambda$$ 都无偏。方差则从 0.375 单调涨到 10.03，**27 倍**。

**② 系统性缩放误差（$$V_\phi=0.8\,V^\pi$$，critic 全局低估）**

| $$\lambda$$ | bias$$^2$$ | var | MSE |
|---:|---:|---:|---:|
| 0.00 | 0.0311 | 0.2402 | 0.2714 |
| 0.60 | 0.0264 | 0.8004 | 0.8268 |
| 0.95 | 0.0358 | 5.1280 | 5.1638 |
| 1.00 | 0.0446 | 10.0133 | 10.0579 |

**③ 低容量 critic（10 个状态聚成 3 类，取类内均值——“分辨不出相邻状态”）**

| $$\lambda$$ | bias$$^2$$ | var | MSE |
|---:|---:|---:|---:|
| 0.00 | 0.4402 | 0.1527 | 0.5930 |
| 0.60 | 0.3414 | 0.6501 | 0.9915 |
| 0.95 | 0.3272 | 5.0280 | 5.3552 |
| 1.00 | 0.3336 | 10.0090 | 10.3426 |

**④ 逐状态独立噪声（$$V_\phi=V^\pi+\mathcal N(0,(0.15\sigma_V)^2)$$）**

| $$\lambda$$ | bias$$^2$$ | var | MSE |
|---:|---:|---:|---:|
| 0.00 | 0.0113 | 0.4493 | 0.4606 |
| 0.60 | 0.0105 | 0.9457 | 0.9562 |
| 0.95 | 0.0171 | 5.2115 | 5.2286 |
| 1.00 | 0.0244 | 10.0332 | 10.0575 |

**读出来的结论：**

1. **$$\lambda$$ 对方差的影响是单调、巨大、与 critic 质量无关的**：四种形态下方差都从 $$\sim$$ 0.15–0.45 涨到 10.0。这是 GAE 的第一性作用。
2. **$$\lambda$$ 对偏差的影响取决于 critic 误差的空间结构**：
   - 误差是**局部的**（低容量 critic 分辨不出相邻状态）→ $$\lambda\uparrow$$ 降偏差（0.4402 → 0.3336）。因为 $$\lambda=1$$ 时估计只依赖 $$V(s_t)$$ 这一个点的误差，$$\lambda=0$$ 时每个 $$\delta$$ 都要吃相邻状态的误差。
   - 误差是**全局的**（缩放偏差）→ $$\lambda\uparrow$$ 升偏差（0.0311 → 0.0446）。因为 $$\lambda=1$$ 时 $$V$$ 的偏差原封不动进入 $$G_t-V(s_t)$$。
3. **在这张表里 $$\lambda=0$$ 的 MSE 永远最低，但不要因此把 PPO 的 $$\lambda$$ 设成 0**。原因是：这里的判据是“单步优势估计的 MSE”，而 PPO 真正关心的是**整条轨迹上策略梯度估计的方差**，且实践中 critic 与 policy 交替更新、$$V_\phi$$ 永远滞后于当前策略（GAE 论文 §4 明确讨论了这一点）。$$\lambda=0.95$$ 是经验值，不是从这张表推出来的。

### 5.3 $$\hat A+V$$ 不等于 return-to-go

训练 critic 的 target 通常写成 $$\hat R_t=\hat A_t+V(s_t)$$。它到底是不是 $$G_t$$？

```
真 G_0: mean=-0.237  std=2.988 | V^pi(s0)=-0.497 | A^pi(s0,a0)=0.416
lambda=0.00 : mean(R_0)=-0.077  bias vs E[G_0] = +0.1595  corr(R_0,G_0)=0.0865
lambda=0.50 : mean(R_0)=-0.115  bias vs E[G_0] = +0.1213  corr(R_0,G_0)=0.4667
lambda=0.90 : mean(R_0)=-0.189  bias vs E[G_0] = +0.0480  corr(R_0,G_0)=0.8731
lambda=0.95 : mean(R_0)=-0.209  bias vs E[G_0] = +0.0276  corr(R_0,G_0)=0.9544
lambda=1.00 : mean(R_0)=-0.237  bias vs E[G_0] = -0.0000  corr(R_0,G_0)=1.0000
```

只有 $$\lambda=1$$ 时 $$\hat R_t$$ 才精确等于 $$G_t$$（相关系数 1.0000）。$$\lambda=0.95$$ 时相关系数 0.954、系统性高估 0.028；$$\lambda=0$$ 时 $$\hat R_t=r_t+\gamma V(s_{t+1})$$ 就是一步 TD target，与 $$G_t$$ 的相关性只剩 0.087。

**实践含义**：用 $$\lambda<1$$ 时，critic 拟合的不是“从这里出发能拿多少分”，而是“$$\lambda$$ 步截断的回报”。这本身不是 bug（TD 学习本来就是这么做的），但如果你拿 $$V$$ 去做“这个 prompt 值不值得做”的推断（比如 dynamic sampling、难度过滤），要记住它在这个意义上有偏。

---

## 6. off-policy 修正：重要性比家族

只要采样策略 $$\mu$$ 与目标策略 $$\pi$$ 不同（异步训练、rollout 复用多个 epoch、经验回放），就需要修正。核心量都是重要性比 $$\rho_t=\pi(a_t\mid s_t)/\mu(a_t\mid s_t)$$，区别在**截断方式**：

| 方法 | 目标量 | 公式要点 | 特点 |
|:---|:---|:---|:---|
| 朴素 IS | $$G_t$$ | $$\prod_{k\ge t}\rho_k$$ | 方差爆炸，几乎不用 |
| per-decision IS | $$G_t$$ | $$\sum_l\gamma^l\big(\prod_{k=t}^{t+l}\rho_k\big)r_{t+l}$$ | 利用了“因果性”，方差低于朴素 IS |
| $$\mathrm{Retrace}(\lambda)$$ | $$Q$$ | $$Q(s_t,a_t)\!+\!\sum_l\gamma^l\big(\prod_{k=t+1}^{t+l}\!c_k\big)\delta_{t+l}$$，$$c_k=\lambda\min(1,\rho_k)$$ | [Munos et al. 2016](https://arxiv.org/abs/1606.02647) |
| V-trace | $$V$$ | 见下 | [IMPALA, Espeholt et al. 2018](https://arxiv.org/abs/1802.01561) |
| Tree-backup($$\lambda$$) | $$Q$$ | 用 $$\pi$$ 的期望替代采样动作，不产生 IS 乘积 | Precup, Sutton & Singh (2000)，见 [Sutton & Barto 第 12 章](http://incompleteideas.net/book/the-book-2nd.html) |
| $$\mathrm{Q}(\lambda)$$ | $$Q$$ | 采样到偏离为止，之后 bootstrap | [Sutton & Barto 2018](http://incompleteideas.net/book/the-book-2nd.html) 第 12 章 |

**V-trace（IMPALA）的完整定义**（核对自原文式 (1)）：

$$v_s=V(x_s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}\Big(\prod_{i=s}^{t-1}c_i\Big)\,\delta_t V,\qquad \delta_t V=\rho_t\big(r_t+\gamma V(x_{t+1})-V(x_t)\big)$$
$$\rho_t=\min\Big(\bar\rho,\frac{\pi(a_t\mid x_t)}{\mu(a_t\mid x_t)}\Big),\qquad c_i=\min\Big(\bar c,\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}\Big),\qquad \bar\rho\ge\bar c$$

两个截断常数分工明确：$$\bar\rho$$ 决定**收敛到哪个策略的价值**（$$\bar\rho=\infty$$ 收敛到 $$V^\pi$$，$$\bar\rho\to0$$ 收敛到 $$V^\mu$$），$$\bar c$$ 决定**收敛多快**。on-policy 时（$$\pi=\mu$$ 且 $$\bar c\ge1$$）所有 $$c_i=\rho_t=1$$，V-trace 退化为标准 $$n$$-step Bellman target——这是它比 Retrace 更适合“on/off 混用”的原因。

策略梯度侧，V-trace 的优势是：

$$\text{pg\_adv}_s=\rho_s\big(r_s+\gamma v_{s+1}-V(x_s)\big)$$

---

## 7. 概率侧：logprob、熵、温度、top-p

### 7.1 定义

$$\log\pi_\theta(y_t\mid q,y_{<t})=z_{t,y_t}-\mathrm{logsumexp}(z_t),\qquad z_t\in\mathbb R^{V}$$
$$\log\pi_\theta(o\mid q)=\sum_{t=1}^{\lvert o\rvert}\log\pi_\theta(y_t\mid q,y_{<t})$$
$$\mathcal H_t=-\sum_{v}p_{t,v}\log p_{t,v},\qquad \mathrm{PPL}(o)=\exp\Big(-\frac{1}{\lvert o\rvert}\log\pi_\theta(o\mid q)\Big)$$

带温度采样时（$$p\propto\exp(z/T)$$）：

$$\log p_T(y_t)=z_{t,y_t}/T-\mathrm{logsumexp}(z_t/T)$$

### 7.2 数值稳定性与精度（实测）

设置：$$\text{logits}\sim\mathcal N(0,8)$$，词表 128000。

| 做法 | 得到的 logprob | 与 fp64 的差 |
|:---|---:|---:|
| fp64 | -37.4472572469 | — |
| fp32 | -37.4472579956 | 7.5e-07 |
| fp16 | -37.44544220 | 1.8e-03 |
| bf16（尾数截断模拟） | -37.34581375 | **1.0e-01** |
| naive（先 `exp` 再 `log`，logits=800） | `-inf` | **溢出** |

两个实践要点：

1. **必须 log-sum-exp 稳定化**：`log(exp(x).sum())` 在 logits 稍大时就溢出成 `inf`（实测 logits=800 时直接 `-inf`，而稳定版正确给出 $$-\ln 10=-2.3026$$）。
2. **bf16 存 logits 的代价是 logprob 有 $$10^{-1}$$ 量级误差**。这个误差在两处放大：
   - **重要性比**：$$\rho=\exp(\Delta\log p)$$，两个 $$10^{-1}$$ 量级的误差相减后取指数 → ratio 百分级误差；
   - **长序列**：$$T$$ 个 token 的 logprob 累加，序列级 ratio 的误差随 $$T$$ 指数放大（见 §9）。
   
   > 说明：这里的 bf16 用“尾数截断”模拟，真实硬件是 round-to-nearest，误差大约减半，量级不变。多数框架在 log-softmax 前会把 logits 提到 fp32，此时误差回到 fp32 的 $$10^{-7}$$ 量级——**请确认你的框架做了这一步**。

### 7.3 top-p / top-k 采样的 mismatch

采样时用的是**截断分布** $$p^{\rm trunc}$$，训练时算 logprob 用的是**完整 softmax** $$p^{\rm full}$$。两者的差是常数：

$$\log p^{\rm trunc}(a)-\log p^{\rm full}(a)=-\log\Big(\sum_{i\in S}p_i\Big)$$

| top-p | 集合内的概率质量 | 对应的 log 差 |
|---:|---:|---:|
| 1.00 | 1.000 | 0.000 |
| 0.95 | ~0.95 | 0.051 |
| 0.90 | ~0.90 | 0.105 |
| 0.80 | ~0.80 | 0.223 |

也就是说 top-p=0.9 时，**每个 token 的 logprob 都系统性偏大 0.105**，一条 1000 token 的回答累积偏大 105。这个偏差在重要性比和 PPL 里都会出现，但因为它对所有 token 是同向的常数，在**组内归一化**（GRPO）下会被减掉，在**跨样本比较**（PPL 排序、best-of-n 打分）下不会。

### 7.4 熵的两个尺度

- **token 级** $$\mathcal H_t\in[0,\log V]$$：分布越平越大。
- **序列级** $$\sum_t\mathcal H_t$$：随长度线性增长，不能直接跨长度比较，一般用 token 平均。

RL 里加的 entropy bonus 通常是 token 级平均 $$\frac{1}{\sum_i\lvert o_i\rvert}\sum_{i,t}\mathcal H_{i,t}$$。注意：**熵奖励会让分布变平**，与 KL 惩罚（拉回 $$\pi_{\rm ref}$$）方向相反，两者系数要一起调。

---

## 8. KL：三种估计器与梯度符号陷阱

### 8.1 三个估计量

记 $$r=\pi_{\rm ref}(o)/\pi_\theta(o)$$，采样 $$o\sim\pi_\theta$$（on-policy）。三个估计量都来自 [Schulman 的 KL 近似笔记](http://joschu.net/blog/kl-approx.html)：

$$k_1=-\log r,\qquad k_2=\tfrac12(\log r)^2,\qquad k_3=r-\log r-1$$

**帽子落在哪里**：$$r$$ 无帽（两个分布的精确比值，§2.2），而 $$k_1,k_2,k_3$$ 是**单个样本上的估计量**——它们各自的期望才是 $$\mathbb D_{\rm KL}$$ 的估计：

$$\hat{\mathbb D}_{\rm KL}=\frac1M\sum_{m=1}^{M}k_n^{(m)},\qquad \mathbb E\big[\hat{\mathbb D}_{\rm KL}\big]\overset{?}{=}\mathbb D_{\rm KL}\ \text{（对 }k_1,k_3\text{ 成立，对 }k_2\text{ 不成立）}$$

$$k_1/k_2/k_3$$ 三者之争（下面这张数值表）是“用哪个函数当这个估计量”的问题，与帽子在哪一层无关。

实测（词表 2000，40 万样本）：

**regime A：策略轻微漂移（logit 噪声 sd=0.15，真 KL=0.01148）**

| | 均值 | 偏差 | 标准差 | 负值占比 |
|:---|---:|---:|---:|---:|
| $$k_1$$ | 0.01166 | +0.00018 | 0.15130 | 47.6% |
| $$k_2$$ | 0.01151 | +0.00003 | 0.01635 | 0.0% |
| $$k_3$$ | 0.01144 | -0.00004 | 0.01642 | 0.0% |

**regime B：策略大幅漂移（logit 噪声 sd=1.20，真 KL=0.79225）**

| | 均值 | 偏差 | 标准差 | 负值占比 |
|:---|---:|---:|---:|---:|
| $$k_1$$ | 0.79182 | -0.00043 | 1.26718 | 26.4% |
| $$k_2$$ | 1.11636 | **+0.32412** | 1.42332 | 0.0% |
| $$k_3$$ | 0.79440 | +0.00216 | 1.39687 | 0.0% |

结论与 Schulman 的结论一致，但要补两条边界：

- **$$k_1$$ 的标准差是真值的 13 倍**（regime A），且 47.6% 的样本给出负值——“KL 散度为负”在数值上很荒谬，作为 reward shaping 项会注入大量噪声。
- **$$k_2$$ 在大 KL 时严重高估**（+0.324，相对偏差 41%）。它在 $$q\approx p$$ 附近与 KL 二阶等价（$$f''(1)=1$$），离开这个邻域就不成立。
- **$$k_3$$ 保持无偏，但“低方差”的优势只在小 KL 时成立**：regime A 里 std=0.0164（比 $$k_1$$ 小 9 倍），regime B 里 std=1.397，与 $$k_1$$ 的 1.267 相当甚至略差。$$k_3$$ 里含 $$r$$ 项，当 $$\pi_\theta(o)\ll\pi_{\rm ref}(o)$$ 时 $$r\gg1$$，方差会爆。**异步训练（rollout 落后 learner 很多步）时，这正是你会走进的 regime**。

### 8.2 梯度侧：$$k_3$$ 的符号陷阱

真正容易错的是“KL 项怎么进梯度”。设目标里加了 $$-\beta\,\mathbb D_{\rm KL}$$，用 $$k_3$$ 估计：

$$\nabla_\theta\,\mathbb E_{o\sim\pi_\theta}\!\left[k_3\right]=\mathbb E\Big[\nabla_\theta\log\pi_\theta\cdot k_3+\nabla_\theta k_3\Big]$$

第二项：$$\nabla_\theta k_3=\nabla_\theta r-\nabla_\theta\log r=(r-1)\nabla_\theta\log r=-(r-1)\nabla_\theta\log\pi_\theta$$。代回：

$$\nabla_\theta\,\mathbb E[k_3]=\mathbb E\Big[\nabla_\theta\log\pi_\theta\cdot\big(k_3+1-r\big)\Big]=\mathbb E\Big[\nabla_\theta\log\pi_\theta\cdot(-\log r)\Big]$$

而真 KL 的梯度是 $$\mathbb E[\nabla_\theta\log\pi_\theta\cdot(-\log r)]$$（因为 $$\mathbb E[\nabla_\theta\log\pi_\theta\cdot k_1]=\nabla{\rm KL}$$）。**两者完全相同**——前提是**反向传播要穿过 $$k_3$$**。

如果实现里对 $$k_3$$ 做了 stop-gradient（只保留 REINFORCE 项，很多框架这么写），系数就从 $$-\log r$$ 变成了 $$k_3$$：

| $$r$$ | 正确系数 $$-\log r$$ | $$k_3=r-\log r-1$$ | $$k_2=\frac12(\log r)^2$$ | |
|---:|---:|---:|---:|:---|
| 0.25 | +1.3863 | 0.6363 | 0.9609 | |
| 0.50 | +0.6931 | 0.1931 | 0.2402 | |
| 0.80 | +0.2231 | 0.0231 | 0.0249 | |
| 1.00 | 0.0000 | 0.0000 | 0.0000 | |
| 1.25 | **-0.2231** | **+0.0269** | 0.0249 | **符号相反** |
| 2.00 | **-0.6931** | **+0.3069** | 0.2402 | **符号相反** |
| 4.00 | **-1.3863** | **+1.6137** | 0.9609 | **符号相反** |

$$r>1$$ 意味着 $$\pi_\theta$$ 比 $$\pi_{\rm ref}$$ **更不**偏好这个 token，此时正确的 KL 梯度系数是负的（惩罚项是“你偏离了”的反向推力），而 $$k_3$$ 恒非负——**惩罚变成了奖励**。另外在 $$r\approx1$$ 附近，$$k_3\approx\frac12(\log r)^2$$ 是 $$-\log r$$ 的二阶小量（$$u=0.1$$ 时 0.005 vs 0.1，低估 20 倍），正则强度被严重削弱。

> 这一节的结论：**用 $$k_3$$ 估计 KL 没问题（它是无偏的），但要确认梯度是否穿过它**。检查方法：在你的框架里搜 `kl_penalty` / `kld`，看它进入 loss 前有没有 `.detach()`。

---

## 9. 重要性比：token 级 vs 序列级与 PPO 裁剪

### 9.1 PPO 的裁剪到底切断了什么

$$L^{\rm CLIP}=\mathbb E\Big[\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\big)\Big]$$

实测（$$\epsilon=0.2$$，数值求导）：

```
A = +1（好动作）：
     r |     loss | d loss/dr | 状态
  0.50 |   0.5000 |    1.0000 | 正常更新
  1.00 |   1.0000 |    1.0000 | 正常更新
  1.19 |   1.1900 |    1.0000 | 正常更新
  1.21 |   1.2000 |    0.0000 | 梯度被切断（已推得够远）
  3.00 |   1.2000 |    0.0000 | 梯度被切断（已推得够远）
A = -1（坏动作）：
  0.30 |  -0.8000 |    0.0000 | 梯度被切断（已压得够低）
  0.79 |  -0.8000 |    0.0000 | 梯度被切断（已压得够低）
  0.81 |  -0.8100 |   -1.0000 | 正常更新
  2.00 |  -2.0000 |   -1.0000 | 正常更新
```

裁剪是**单向**的：只在“优势方向已经把 ratio 推到界外且方向不变”时才切断。所以 $$\epsilon$$ 不是“限制更新幅度”的软约束，而是**硬阈值**——超过就完全不更新。$$\epsilon=0.2$$ 意味着一个 token 的概率在一次 rollout 复用期内最多被推高 20%（相对），超过就停止。

### 9.2 长序列上 token 级 ratio 是病态的

假设一次 mini-batch 更新后每个 token 的 $$\log$$ ratio 独立地服从 $$\mathcal N(\mu,0.05^2)$$，看序列级 $$\rho=\exp(\sum_t\log\rho_t)$$ 的分布：

| $$T$$ | $$\mu$$ | 序列级 ratio 中位数 | p99 | $$P(\rho>2)$$ | GSPO 形式 $$\rho^{1/T}$$ 中位数 |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.00 | 1.002 | 2.55 | 0.042 | 1.000 |
| 64 | +0.01 | 1.896 | 4.81 | 0.446 | 1.010 |
| 256 | 0.00 | 0.998 | 6.38 | 0.192 | 1.000 |
| 256 | +0.01 | 12.987 | 83.96 | 0.990 | 1.010 |
| 1024 | 0.00 | 0.999 | 42.15 | 0.333 | 1.000 |
| 1024 | +0.01 | **27917.7** | $$1.15\times10^6$$ | 1.000 | 1.010 |
| 4096 | +0.01 | **$$6.0\times10^{17}$$** | $$1.0\times10^{21}$$ | 1.000 | 1.010 |

即使策略**平均而言完全没变**（$$\mu=0$$），4096 token 的序列级 ratio 的 p99 也有 1684——因为 $$\sum_t\log\rho_t$$ 的标准差是 $$0.05\sqrt{T}$$，$$T=4096$$ 时是 3.2，取指数后跨越 3 个数量级。

这就是 [GSPO](https://arxiv.org/abs/2507.18071) 把重要性比改成**序列级几何平均**的动机：

$$\rho^{\rm GSPO}(o)=\Big(\frac{\pi_\theta(o\mid q)}{\pi_{\rm old}(o\mid q)}\Big)^{1/\lvert o\rvert}=\exp\Big(\frac{1}{\lvert o\rvert}\sum_{t=1}^{\lvert o\rvert}\log\frac{\pi_\theta(y_t\mid q,y_{<t})}{\pi_{\rm old}(y_t\mid q,y_{<t})}\Big)$$

上表最后一列：无论 $$T$$ 多大，GSPO 的 ratio 始终稳定在 1.000~1.010。**长度归一化把 $$O(\sqrt T)$$ 的漂移压回了 $$O(1/\sqrt T)$$**。

> 说明：这里的 $$\mathcal N(\mu,0.05^2)$$ 是示意性假设（真实分布取决于学习率、token 位置、模型），用来展示 $$T$$ 的放大效应，不是实测数据。

---

## 10. 归一化与量纲：GRPO / Dr.GRPO / whitening

### 10.1 除不除 std，改变了什么

$$G=8$$、二值奖励、组内 $$k$$ 条答对：

| $$k$$ | GRPO: $$A$$（正确） | GRPO: $$A$$（错误） | Dr.GRPO: $$A$$（正确） | Dr.GRPO: $$A$$（错误） |
|---:|---:|---:|---:|---:|
| 1 | **2.6458** | -0.3780 | 0.8750 | -0.1250 |
| 2 | 1.7321 | -0.5774 | 0.7500 | -0.2500 |
| 3 | 1.2910 | -0.7746 | 0.6250 | -0.3750 |
| 4 | 1.0000 | -1.0000 | 0.5000 | -0.5000 |
| 5 | 0.7746 | -1.2910 | 0.3750 | -0.6250 |
| 6 | 0.5774 | -1.7321 | 0.2500 | -0.7500 |
| 7 | 0.3780 | -2.6458 | 0.1250 | -0.8750 |

组内正负比值两者都是 $$7:1$$，但**跨组（跨难度）的权重不同**：

- GRPO：$$k=1$$ 的题（难题）正样本权重 2.646，$$k=4$$ 的题是 1.000 → **难题被放大 2.65 倍**
- Dr.GRPO：0.875 vs 0.500 → 只放大 1.75 倍

除以 std 等价于“按题目难度反向加权”：组内方差越小（几乎全对或全错）的题，advantage 的绝对值越大。这就是 Dr.GRPO 说的**难度偏置**——它会把优化压力集中到“模型已经很确定”的题目上。

### 10.2 零梯度组：有多少采样是白做的

组内奖励全同时（全对或全错），$$\hat\mu_g=R$$、$$\hat\sigma_g=0$$，所有 advantage 恒为 0，整组不产生梯度：

| pass rate | $$G=8$$ | $$G=16$$ |
|---:|---:|---:|
| 0.10 | 43.05% | 18.53% |
| 0.30 | 5.77% | 0.33% |
| 0.50 | **0.78%** | 0.00% |
| 0.70 | 5.77% | 0.33% |
| 0.90 | 43.05% | 18.53% |
| 0.95 | **66.34%** | 44.01% |
| 0.99 | **92.27%** | 85.15% |

**pass rate 一旦超过 0.9，$$G=8$$ 时三分之二的 rollout 是纯浪费**。这就是 [DAPO](https://arxiv.org/abs/2503.14476) 的 dynamic sampling（超采样后丢弃全同组）和“提高 $$G$$”的直接动机。注意 $$G$$ 从 8 提到 16 在 pass rate=0.95 时只把浪费从 66% 降到 44%——**$$G$$ 的收益是对数级的，而成本是线性的**。

### 10.3 whitening 不是“免费的方差缩减”

batch 级 whitening $$\tilde A=(A-\mu_{\rm batch})/\sigma_{\rm batch}$$ 引入了**跨样本耦合**：一个样本的 advantage 会依赖同 batch 里所有其他样本。后果包括：

- 梯度的期望不再是原始梯度的正标量倍（方向会变），尤其 batch 内难度分布不均时；
- 不同 micro-batch 之间梯度不可交换（不能再随意切分）；
- reward hacking 时，一旦多数样本被 hack 到高分，$$\mu_{\rm batch}$$ 抬高，诚实的样本反而拿到巨大负 advantage。

如果你要用 whitening，建议只在**每个 prompt 的组内**做（GRPO 已经是），不要在全局 batch 上再做一次。

---

## 11. RLHF 特例：$$\gamma=1$$ 下 GAE 的闭式解

这一节是本篇相对“非标准”的部分，但它直接对应 RLVR 的训练设定：**$$\gamma=1$$、状态转移确定、奖励只在最后一个 token 给出、终止状态价值为 0**。

在这个设定下，$$r_t=0\ (t<T-1)$$、$$r_{T-1}=R$$、$$V(s_T)=0$$，于是

$$\delta_t=V_{t+1}-V_t\ (t<T-1),\qquad \delta_{T-1}=R-V_{T-1}$$

把 GAE 的定义式展开并做望远镜求和（过程见 §3 的恒等式 (2)），得到**闭式解**：

$$\boxed{\ \hat A_t=-V_t+(1-\lambda)\sum_{j=1}^{T-1-t}\lambda^{j-1}V_{t+j}+\lambda^{T-1-t}R\ }$$

数值验证（$$T=12$$，$$V$$ 随机单调，$$R=1$$）：

```
lambda=0.00 : max|递归 - 闭式| = 0.000e+00
lambda=0.30 : max|递归 - 闭式| = 1.527e-16
lambda=0.60 : max|递归 - 闭式| = 1.110e-16
lambda=0.90 : max|递归 - 闭式| = 8.327e-17
lambda=0.95 : max|递归 - 闭式| = 1.110e-16
lambda=1.00 : max|递归 - 闭式| = 1.110e-16
```

### 11.1 两个极端的含义

| $$t$$ | $$V(s_t)$$ | $$\lambda=0$$：$$V_{t+1}-V_t$$ | $$\lambda=1$$：$$R-V_t$$ |
|---:|---:|---:|---:|
| 0 | 0.280 | 0.0105 | 0.7204 |
| 3 | 0.353 | 0.0918 | 0.6474 |
| 6 | 0.529 | 0.1195 | 0.4705 |
| 9 | 0.686 | 0.0159 | 0.3142 |
| 11 | 0.797 | 0.2026 | 0.2026 |
| $$\sum_t\hat A_t$$ | | **0.7204** $$=R-V_0$$ | **5.7808** |

- **$$\lambda=0$$：$$\hat A_t=V_{t+1}-V_t$$**，即“写下这个 token 让期望得分涨了多少”。这是**真正的逐 token 信用分配**，而且 $$\sum_t\hat A_t=R-V_0$$ 是望远镜和——总梯度量守恒，不重复计入。
- **$$\lambda=1$$：$$\hat A_t=R-V_t$$**，即“最终结果比这个位置的预期好多少”。注意 $$\sum_t\hat A_t=5.78$$，是 $$\lambda=0$$ 的 8 倍——**每个 token 都把整条轨迹的剩余价值完整算了一遍**。这意味着 $$\lambda\to1$$ 时梯度尺度随 $$T$$ 线性放大，通常需要配合 advantage 归一化或更小的学习率。

### 11.2 critic 塌缩时会发生什么

若 critic 失去分辨力、对所有位置输出同一个常数 $$c$$（RLHF 里很常见的 value collapse），则 $$\delta_t=0\ (t<T-1)$$、$$\delta_{T-1}=R-c$$，闭式解退化成：

$$\hat A_t=\lambda^{T-1-t}(R-c)$$

实测（$$T=12,\lambda=0.95,c=0.5,R=1$$）：

```
c=0.5 : delta = [0 0 0 0 0 0 0 0 0 0 0 0.5]
        GAE(lam=0.95) = [0.2844 0.2994 0.3151 0.3317 0.3492 0.3675 0.3869 0.4073 0.4287 0.4512 0.4750 0.5000]
        闭式 lam^{T-1-t}(R-c) = [0.2844 0.2994 ... 0.5000]
c=1.0 : delta = [0 ... 0]   ->  GAE 全为 0
```

**$$\lambda=0.95$$ 时信号从句尾向句首指数衰减**（$$0.95^{11}=0.57$$），$$\lambda=0$$ 时**只有最后一个 token 拿到梯度**，前面全部为 0。这解释了两个实践现象：

1. 纯 outcome reward + 无 KL shaping + critic 塌缩 → 训练早期只有靠近答案的 token 在学；
2. 加入**每 token 的 KL shaping 奖励** $$r_t=-\beta\,{\rm KL}_t$$ 后，每个位置都有了非零即时奖励，信号不再依赖 $$\lambda$$ 的指数传播：

```
lambda=0.00 : A[:6] = [0.009  0.047  0.0083 0.0881 0.0415 0.033 ] ...
lambda=0.95 : A[:6] = [0.4717 0.4871 0.4632 0.4789 0.4114 0.3894] ...
lambda=1.00 : A[:6] = [0.6673 0.6583 0.6113 0.6031 0.515  0.4735] ...
```

---

## 12. 训练时要盯的监控量

| 监控量 | 定义 | 健康范围 | 异常时意味着什么 |
|:---|:---|:---|:---|
| clip fraction | $$\mathbb P\big(\vert\rho_t-1\vert>\epsilon\big)$$ | 0.05 ~ 0.3 | > 0.5：策略跑太快（lr 太大或 epoch 太多）；≈ 0：裁剪没起作用，等于普通策略梯度 |
| approx KL | $$\frac12\mathbb E[(\log\pi_\theta-\log\pi_{\rm old})^2]$$ 或 $$\mathbb E[\log\pi_{\rm old}-\log\pi_\theta]$$ | 与 $$\beta$$ 的目标 KL 同量级 | 持续上升 = 策略在漂移，考虑早停 |
| explained variance | $$1-\dfrac{\mathrm{Var}(R-\hat V)}{\mathrm{Var}(R)}$$ | 0.4 ~ 0.9 | < 0：critic 还不如常数；> 0.95：可能过拟合了 batch |
| value loss | $$\mathbb E[(\hat V-R)^2]$$（常取 0.5×，且做 clipped 版） | 缓慢下降 | 暴涨 = reward scale 变化或 critic lr 过大 |
| ratio 分位数 | $$\rho_t$$ 的 p1 / p50 / p99 | p99 < $$1+\epsilon$$ 的若干倍 | p99 随 $$T$$ 增长 → 见 §9.2，考虑 GSPO |
| advantage 均值 / std | — | 组内应近似零均值 | 非零 = baseline 有偏；std 为 0 = 零梯度组（§10.2） |
| 熵 | $$\frac{1}{\sum\vert o\vert}\sum\mathcal H$$ | 缓慢下降，不能塌到 0 | 骤降 = 模式塌缩；上升 = lr 过大或熵系数过大 |
| KL to ref | $$k_3$$ 的序列平均 | 与 $$\beta$$ 设定一致 | 见 §8.2，注意梯度是否穿过 |

两个容易搞错的细节：

- **approx KL 的两种写法不等价**：$$\mathbb E[\log\pi_{\rm old}-\log\pi_\theta]$$（$$k_1$$ 形式）可以为负、方差大；$$\frac12\mathbb E[(\Delta\log p)^2]$$（$$k_2$$ 形式）非负、有偏。监控用哪个都行，但**切换时阈值要跟着改**。
- **explained variance 用 $$R$$ 的真值方差做分母**，如果你的 $$R$$ 是二值的（0/1），$$\mathrm{Var}(R)$$ 很小，EV 会显得异常低，此时更该看 value loss 的绝对量级。

---

## 13. 速查总表与 17 个坑

### 13.1 速查总表

最后一列是帽子（§2）：✅ = 估计量，❌ = 真值或精确可算的量。

| 量 | 符号 | 计算式 | 帽子 | 备注 |
|:---|:---|:---|:---:|:---|
| 回报 | $$G_t$$ | $$\sum_{l\ge0}\gamma^l r_{t+l}$$ | — | RLVR 中常退化为 $$G_t=R$$（$$\gamma=1$$） |
| 状态价值（真） | $$V^\pi$$ | $$\mathbb E_\pi[G_t\mid s_t]$$ | ❌ | 拿不到 |
| 状态价值（critic） | $$V_\phi(s_t)$$ | 网络输出 | 下标 $$\phi$$ | 训练 target 见 §5.3 |
| 动作价值 | $$Q^\pi$$ | $$r_t+\gamma V^\pi(s_{t+1})$$ | ❌ | 确定性环境下成立 |
| 优势（真） | $$A^\pi$$ | $$Q^\pi-V^\pi$$ | ❌ | $$\mathbb E_\pi[A]=0$$ 只对它成立 |
| TD error | $$\hat\delta_t$$ | $$r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$$ | ✅ | 确定性环境下 $$\delta^\pi=A^\pi$$（无帽版） |
| GAE | $$\hat A_t$$ | $$\hat\delta_t+\gamma\lambda\hat A_{t+1}$$ | ✅ | 反向递归，跨 episode 断开 |
| n-step | $$\hat A_t^{(n)}$$ | $$\sum_{l<n}\gamma^l r_{t+l}+\gamma^n V(s_{t+n})-V(s_t)$$ | ✅ | GAE 的 $$(1-\lambda)\lambda^{n-1}$$ 加权平均 |
| value target | $$\hat R_t$$ | $$\hat A_t+V_\phi(s_t)$$ | ✅ | 只有 $$\lambda=1$$ 时才 $$=G_t$$ |
| token logprob | $$\ell_t$$ | $$z_{t,y_t}-\mathrm{logsumexp}(z_t)$$ | ❌ | 精确；必须稳定化 |
| 序列 logprob | $$\ell(o)$$ | $$\sum_t\ell_t$$ | ❌ | bf16 下误差随 $$T$$ 放大 |
| 熵 | $$\mathcal H_t$$ | $$-\sum_v p_v\log p_v$$ | ❌ | 精确；用 token 平均，别用序列和 |
| 重要性比（token） | $$\rho_t$$ | $$\exp(\ell_t^\theta-\ell_t^{\rm old})$$ | ❌ | 精确；在 log 空间算再 exp |
| 重要性比（序列） | $$\rho(o)$$ | $$\exp(\sum_t\Delta\ell_t)$$ | ❌ | 精确但病态，见 §9.2 |
| GSPO 比 | $$\rho^{\rm GSPO}$$ | $$\exp(\frac1T\sum_t\Delta\ell_t)$$ | ❌ | 长度归一化 |
| KL（真） | $$\mathbb D_{\rm KL}$$ | $$\sum_o\pi_\theta(o)\log\frac{\pi_\theta}{\pi_{\rm ref}}$$ | ❌ | 算不起 |
| KL（估计） | $$\hat{\mathbb D}_{\rm KL}$$ | $$r-\log r-1,\ r=\pi_{\rm ref}/\pi_\theta$$ | ✅ | 无偏、非负 |
| 组内均值 / std | $$\hat\mu_g,\hat\sigma_g$$ | $$\frac1G\sum_jR_j$$，$$\sqrt{\frac1G\sum_j(R_j-\hat\mu_g)^2}$$ | ✅ | 含 $$R_i$$ 自身 |
| 组内优势 | $$\hat A_i$$ | $$(R_i-\hat\mu_g)/\hat\sigma_g$$ 或 $$R_i-\hat\mu_g$$ | ✅ | 见 §10.1 |
| 归一化后 | $$\tilde A_i$$ | $$(\hat A_i-\mu)/\sigma$$ | 用 tilde | 确定性后处理，不叠帽子 |

### 13.2 17 个坑

1. **跨 episode 传 $$\hat A_{t+1}$$**——反向递归时遇到终止必须把累加器清零。
2. **末步忘记 bootstrap**——$$\delta_{T-1}$$ 需要 $$V(s_T)$$，终止时是 0，未终止时是 critic 的预测。
3. **$$\hat A+V$$ 当 $$G_t$$ 用**——只有 $$\lambda=1$$ 时成立（§5.3）。
4. **偏差-方差分解时跨状态平均**——先逐点算 bias 再平方，否则正负抵消（§5.2）。
5. **把 $$\pi_{\rm ref}$$ 当 $$\pi_{\rm old}$$**——前者冻结、后者每轮刷新，用途完全不同。
6. **$$\mathrm{TD}(\lambda)$$ 与 GAE 混用**——一个估 $$V$$，一个估 $$A$$。
7. **$$\gamma$$-just ≠ 无偏**——$$\gamma<1$$ 时 GAE 对折扣目标的梯度无偏，对无折扣目标仍是有偏代理。
8. **naive `log(sum(exp))`**——用 log-sum-exp，否则 logits 稍大就 `inf`（§7.2）。
9. **bf16 logits 直接算 logprob**——$$10^{-1}$$ 量级误差，长序列放大（§7.2）。
10. **top-p 采样但用完整分布算 logprob**——每 token 系统性偏 $$-\log(\text{mass})$$（§7.3）。
11. **$$k_1$$ 当 KL 用**——47.6% 的样本给出负值（§8.1）。
12. **$$k_2$$ 在大 KL 时用**——41% 相对高估（§8.1）。
13. **对 $$k_3$$ stop-gradient**——$$r>1$$ 时梯度符号相反（§8.2）。
14. **pass rate > 0.9 还用 $$G=8$$**——66% 的 rollout 零梯度（§10.2）。
15. **batch 级 whitening + 组内归一化叠加**——引入跨样本耦合，梯度方向改变（§10.3）。
16. **给 $$\rho_t$$、$$\ell_t$$、$$\mathcal H_t$$ 加帽子**——它们是精确可算的量（§2.2），加帽会让你误以为存在需要处理的采样噪声。反过来，**KL 必须加帽**：它要对指数级的序列空间求和，算不起。
17. **把 $$\hat R_t=\hat A_t+V_\phi$$ 当真标签**——它是自举 target，由估计量构造（§2.5）。只有 $$\lambda=1$$ 时它才等于 return-to-go。用 $$\lambda<1$$ 时 critic 拟合的是“$$\lambda$$ 步截断回报”，不是 $$G_t$$。

---

## 14. 附录：复现脚本

三个脚本，纯 numpy，CPU 上总共约 40 秒。本文所有数值表都来自这些脚本的输出。

**脚本 1（§5、§3）：GAE 的递归-展开一致性、偏差-方差分解、$$\hat A+V$$ 与 $$G_t$$ 的对比**

```python
import numpy as np
rng = np.random.default_rng(7)
S, A, gamma, T, N = 10, 3, 0.95, 30, 1200
P = rng.dirichlet(np.ones(S) * 0.25, size=(S, A))
R = rng.normal(0.0, 1.0, size=(S, A))
pi = rng.dirichlet(np.ones(A) * 1.0, size=S)
Ppi = np.einsum('sa,sat->st', pi, P); Rpi = np.einsum('sa,sa->s', pi, R)
V = np.linalg.solve(np.eye(S) - gamma * Ppi, Rpi)
Q = R + gamma * np.einsum('sat,t->sa', P, V)
Adv = Q - V[:, None]                       # 真优势

def gae_zero(Vhat, lam, s_seq, a_seq):
    r_seq = R[s_seq[:-1], a_seq]; Vs = Vhat[s_seq]
    delta = r_seq + gamma * Vs[1:] - Vs[:-1]
    return float(np.sum((gamma * lam) ** np.arange(T) * delta))

def gae_recursive(Vhat, lam, s_seq, a_seq):
    r_seq = R[s_seq[:-1], a_seq]; Vs = Vhat[s_seq]
    delta = r_seq + gamma * Vs[1:] - Vs[:-1]
    g = 0.0
    for t in range(T - 1, -1, -1):
        g = delta[t] + gamma * lam * g
    return float(g)

def sample_traj(s0, a0, n):
    ss = np.zeros((n, T + 1), int); aa = np.zeros((n, T), int)
    ss[:, 0] = s0; aa[:, 0] = a0
    for t in range(T):
        for i in range(n):
            ss[i, t + 1] = rng.choice(S, p=P[ss[i, t], aa[i, t]])
        if t + 1 < T:
            for i in range(n):
                aa[i, t + 1] = rng.choice(A, p=pi[ss[i, t + 1]])
    return ss, aa

trajs = {(s0, a0): sample_traj(s0, a0, N) for s0 in range(S) for a0 in range(A)}
# 偏差-方差分解：逐 (s,a) 算 bias/var，再对状态平均
for lam in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]:
    errs = []
    for (s, a), (sseq, aseq) in trajs.items():
        e = np.array([gae_zero(V, lam, sseq[i], aseq[i]) - Adv[s, a] for i in range(N)])
        errs.append((e.mean(), e.var()))
    b2 = np.mean([m ** 2 for m, _ in errs]); var = np.mean([v for _, v in errs])
    print(lam, b2, var, b2 + var)
```

**脚本 2（§7–§10）：KL 估计器、PPO 裁剪、长序列 ratio、GRPO 归一化、logprob 精度**

```python
import numpy as np
rng = np.random.default_rng(11)
Vv = 2000
q = rng.dirichlet(np.ones(Vv) * 0.8)
logits_p = np.log(q) + rng.normal(0, 0.15, size=Vv)     # 0.15 -> 轻微漂移; 1.2 -> 大幅漂移
def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()
p = softmax(logits_p)
true_kl = float(np.sum(p * np.log(p / q)))
idx = rng.choice(Vv, size=400000, p=p)
r = q[idx] / p[idx]
k1, k2, k3 = -np.log(r), 0.5 * np.log(r) ** 2, r - np.log(r) - 1
print("true KL", true_kl, [k.mean() for k in (k1, k2, k3)], [k.std() for k in (k1, k2, k3)])

# PPO 裁剪的梯度切断
eps = 0.2
obj = lambda r, A: np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)
grad = lambda r, A, h=1e-6: (obj(r + h, A) - obj(r - h, A)) / (2 * h)

# 长序列 token 级 ratio
for T in [64, 256, 1024, 4096]:
    lr_ = rng.normal(0.01, 0.05, size=(200000, T))
    seq = np.exp(lr_.sum(axis=1)); gspo = seq ** (1.0 / T)
    print(T, np.median(seq), np.percentile(seq, 99), np.median(gspo))

# logprob 精度
def lse(x):
    m = x.max(); return x - (m + np.log(np.exp(x - m).sum()))
lg = rng.normal(0, 8, size=128000).astype(np.float32)
ref = lse(lg.astype(np.float64))
def to_bf16(x32):
    return (x32.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
print(lse(lg)[0], lse(to_bf16(lg))[0], lse(lg.astype(np.float16).astype(np.float32))[0])
```

**脚本 3（§11）：$$\gamma=1$$ 下 GAE 的闭式解**

```python
import numpy as np
rng = np.random.default_rng(23)
T, gamma, R = 12, 1.0, 1.0
V = np.sort(rng.uniform(0.2, 0.9, size=T))

def gae_recursive(lam):
    Vs = np.append(V, 0.0); r = np.zeros(T); r[-1] = R
    delta = r + gamma * Vs[1:] - Vs[:-1]
    A = np.zeros(T); g = 0.0
    for t in range(T - 1, -1, -1):
        g = delta[t] + gamma * lam * g; A[t] = g
    return A

def gae_closed(lam):
    A = np.zeros(T)
    for t in range(T):
        m = T - 1 - t
        s = sum((lam ** j) * V[t + 1 + j] for j in range(m)) if m > 0 else 0.0
        A[t] = -V[t] + (1 - lam) * s + (lam ** m) * R
    return A

for lam in [0.0, 0.3, 0.6, 0.9, 0.95, 1.0]:
    print(lam, np.max(np.abs(gae_recursive(lam) - gae_closed(lam))))
```

---

## 相关阅读

- [High-Dimensional Continuous Control Using Generalized Advantage Estimation（Schulman et al., ICLR 2016）](https://arxiv.org/abs/1506.02438) — GAE 原文，“$$\gamma$$-just”定义与 Proposition 1
- [Proximal Policy Optimization Algorithms（Schulman et al., 2017）](https://arxiv.org/abs/1707.06347) — 裁剪代理目标
- [DeepSeekMath: GRPO（2024）](https://arxiv.org/abs/2402.03300) — 组相对优势与 $$k_3$$ KL
- [IMPALA / V-trace（Espeholt et al., ICML 2018）](https://arxiv.org/abs/1802.01561) — off-policy 修正的截断 IS
- [Back to Basics: Revisiting REINFORCE-Style Optimization（RLOO, 2024）](https://arxiv.org/abs/2402.14740) — 留一法 baseline
- [REINFORCE++（2025）](https://arxiv.org/abs/2501.03262) — 全局 batch baseline 与 token 级 KL
- [One Framework to Rule Them All（2025）](https://arxiv.org/abs/2503.19523) — RL-based 与 RL-free 方法的 GPO 统一表
- [DAPO（2025）](https://arxiv.org/abs/2503.14476) — dynamic sampling 与 token 级 loss
- [Dr. GRPO（2025）](https://arxiv.org/abs/2503.20783) — 去掉 std 归一化与长度归一化
- [GSPO（2025）](https://arxiv.org/abs/2507.18071) — 序列级重要性比
- [Approximating KL Divergence（Schulman, 2020 博客）](http://joschu.net/blog/kl-approx.html) — $$k_1/k_2/k_3$$ 的来源
- [Sutton & Barto, Reinforcement Learning: An Introduction（第 12 章：eligibility traces）](http://incompleteideas.net/book/the-book-2nd.html)

> **未验证声明清单**：
> ① §7.2 的 bf16 数值用“尾数截断”模拟，真实 round-to-nearest 误差约减半，量级不变。
> ② §9.2 的每 token $$\log$$ ratio 分布 $$\mathcal N(\mu,0.05^2)$$ 是示意性假设，用于展示 $$T$$ 的放大效应，非实测。
> ③ §8.2 的“stop-gradient 导致符号相反”是对实现的静态分析，具体行为取决于你用的框架是否对 $$k_3$$ 做 `.detach()`，请自行核对 `kl_penalty` 的实现。
> ⑤ 本文所有数值均在 CPU numpy 上复现（Python 3.11 / numpy 2.1.1），随机种子已固定。
