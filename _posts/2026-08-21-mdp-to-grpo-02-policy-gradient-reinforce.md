---
title: "从 MDP 到 GRPO（二）：策略梯度——REINFORCE 与方差的战争"
date: 2026-08-21 21:10:00 +0800
categories:
  - 强化学习
tags: [rl, policy-gradient, reinforce, baseline, advantage, variance-reduction]
layout: post
mathjax: true
---

> **TL;DR**
>
> * **策略梯度定理是 RL 里最重要的恒等式**：$\nabla_\theta J(\theta) = \mathbb{E}\big[\, \nabla_\theta \log \pi_\theta(a \mid s)\, A^\pi(s,a) \,\big]$。它把"无法求导的期望回报"变成了"可以采样的梯度估计"，靠的是一个初等的恒等式 $\nabla p = p\, \nabla \log p$（log-derivative trick），以及一个关键的观察——**环境动力学项在求导时消失**，所以这个梯度不需要知道环境的模型。
> * **REINFORCE 能跑，但方差大到离谱**。本篇的主线是一场持续到 GRPO 的"方差战争"：reward-to-go（各步只用自己之后的回报）→ baseline（减一个与动作无关的量，期望不变）→ advantage（用 $V^\pi$ 当 baseline）。**第五篇 GRPO 的"组内减均值除标准差"，就是这场战争在大模型时代的最新战役**。
> * **on-policy 是策略梯度的原罪**：数据按当前策略采样、更新一次就作废。想复用数据就得重要性采样，而比率连乘在长轨迹上方差爆炸——这个死结将在下一篇由 TRPO 用"限制新旧策略距离"来解开。

```mermaid
flowchart LR
    A["朴素策略梯度<br/>∇logπ · G_t"] --> B["+ reward-to-go<br/>各步只看未来"]
    B --> C["+ baseline<br/>减去与动作无关的量"]
    C --> D["+ advantage<br/>A = Q − V"]
    D --> E["GRPO<br/>组内统计估计 A<br/>（第五篇）"]
    B -.->|"期望不变"| C
    C -.->|"期望不变"| D
```

---

## 1. 承接：值函数会打分，但不会改进

第一篇我们解决了"给策略打分"的问题：贝尔曼方程 + 动态规划可以在表格世界里精确算出 $V^\pi$ 甚至 $V^\ast$。但通往 GRPO 的路上还差三样东西：

1. **状态是无限集**。LLM 的状态是所有可能的前缀序列，表格放不下，$V(s)$ 必须参数化；
2. **没有环境模型**。$p(s' \mid s, a)$ 未知，$\max_a \sum_{s'}$ 这一步无从谈起；
3. **策略本身需要显式表达**。词表十万维上做 $\arg\max_a$，或者干脆想要随机策略时，"从值函数反推策略"这条路走不通。

最直接的思路：**跳过值函数，直接把策略参数化成 $\pi_\theta(a \mid s)$（比如一个神经网络），然后对目标函数求梯度**。这就是策略梯度（policy gradient）方法，也是后面 TRPO / PPO / GRPO 整条演化线的共同起点。

## 2. 目标函数：一个采样得动、却求不了导的东西

把第一篇的"最大化期望折扣回报"写成参数化形式。一条轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ 的总回报记为 $R(\tau)$，目标函数就是它的期望：

$$J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta(\tau)}\big[\, R(\tau) \,\big] \qquad\text{其中}\quad R(\tau) = \sum_{t=0}^{T-1} \gamma^t\, r_{t+1}$$

轨迹的概率由初始分布、策略、环境动力学三部分连乘给出：

$$p_\theta(\tau) \;=\; \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t)$$

麻烦在于：$J(\theta)$ 是对**未知分布**的期望。采样没问题——跑几个 episode 就行；但求导看起来没戏：

$$\nabla_\theta J(\theta) = \int \nabla_\theta\, p_\theta(\tau)\, R(\tau)\, \mathrm{d}\tau$$

被积函数里有 $\nabla_\theta p_\theta(\tau)$——对概率本身求导，而我们只能从 $p_\theta$ 里**采样**，没法算出它的解析式。

## 3. log-derivative trick：两行代换撬动整个领域

第一步是一个平凡却威力巨大的恒等式（对任意可导的正函数成立）：

$$\nabla_\theta\, p_\theta(\tau) = p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)$$

代回去，积分号下的东西重新组合成一个期望：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\big[\, \nabla_\theta \log p_\theta(\tau)\, R(\tau) \,\big]$$

现在只需要算 $\nabla_\theta \log p_\theta(\tau)$。对连乘取对数：

$$\log p_\theta(\tau) = \log \rho_0(s_0) + \sum_{t=0}^{T-1} \Big[\, \log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t) \,\Big]$$

关键观察来了：**三项里只有策略项含 $\theta$**。初始分布 $\rho_0$ 和环境动力学 $P$ 都与 $\theta$ 无关，求导后直接消失：

$$\nabla_\theta \log p_\theta(\tau) \;=\; \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

这一步值得停下来体会：**环境怎么转移的，完全不出现在梯度里**。这就是策略梯度"model-free"的数学根源——我们不需要知道环境，只需要能和环境交互。

于是得到**策略梯度定理**（episodic 形式）：

$$\boxed{\;\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}\left[\; \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t \;\right]\;}$$

其中 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ 是从时刻 $t$ 起的 reward-to-go。直觉读法非常直白：

* **表现好的轨迹**（$G_t$ 大）：拉高这些轨迹里每个动作的对数概率；
* **表现差的轨迹**（$G_t$ 小甚至为负）：压低它们的概率。

每个 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 就像一根橡皮筋，把概率质量往"好动作"方向拽。

> **考究备注**：折扣因子放哪里，文献里有两种不完全等价的写法——Sutton & Barto 把 $\gamma^t$ 乘在整个 $\nabla \log \pi$ 项外面（对应"每时刻平均奖励"的目标），DeepMind 系教材则把它折进 $G_t$（对应"起始状态分布上的期望回报"）。两者在 $\gamma < 1$ 时梯度略有差别，但对现代深度 RL 实践影响可忽略。本系列采用后者。

## 4. REINFORCE：最小可行策略梯度

把上面的期望用蒙特卡洛估计代替，就得到 1992 年的 **REINFORCE** 算法：

```
初始化策略参数 θ
repeat:
    用 π_θ 采样完整轨迹 τ = (s_0, a_0, r_1, ..., s_T)     # on-policy：必须用当前策略采
    从后往前递推折扣回报：G_T = 0；G_t = r_{t+1} + γ·G_{t+1}
    θ ← θ + α · Σ_t ∇θ log π_θ(a_t | s_t) · G_t           # 梯度上升
until 收敛
```

PyTorch 实现（CartPole 为例，可直接运行）：

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

env = gym.make("CartPole-v1")
policy = nn.Sequential(nn.Linear(4, 128), nn.Tanh(), nn.Linear(128, 2))
optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.99

for episode in range(1, 601):
    state, _ = env.reset()
    logps, rewards = [], []
    done = False
    while not done:                                   # ① 采样一整条轨迹
        dist = Categorical(logits=policy(torch.tensor(state, dtype=torch.float32)))
        action = dist.sample()
        logps.append(dist.log_prob(action))           #    记下 log π(a_t | s_t)
        state, r, term, trunc, _ = env.step(action.item())
        rewards.append(r)
        done = term or trunc

    G, returns = 0.0, []                              # ② 从后往前递推 G_t
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns = torch.tensor(list(reversed(returns)))
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # ③ 回报标准化

    loss = -(torch.stack(logps) * returns).sum()      # ④ 负号：优化器做的是梯度下降
    optim.zero_grad(); loss.backward(); optim.step()

    if episode % 100 == 0:
        with torch.no_grad():                         #    评估：贪心策略跑 20 个回合
            scores = []
            for _ in range(20):
                s, _ = env.reset(); d, total = False, 0
                while not d:
                    a = policy(torch.tensor(s, dtype=torch.float32)).argmax().item()
                    s, _, t, tr, _ = env.step(a); d = t or tr; total += 1
                scores.append(total)
        print(f"ep {episode}: 平均存活 {sum(scores)/len(scores):.0f} 步")
```

四个容易踩的实现细节都标了号：**①** 必须采完整条轨迹才能算 loss（蒙特卡洛的天性）；**②** $G_t$ 从后往前一次递推，别写 $O(T^2)$ 的双重循环；**③** 回报标准化是最便宜的方差缩减手段（见下一节）；**④** PyTorch 优化器只会下降，所以 loss 取负号。

真实运行结果（Apple M1 CPU，约 3 分钟；CartPole-v1 满分 500 步）：

```
ep 100: 平均存活 73 步
ep 200: 平均存活 202 步
ep 300: 平均存活 447 步
ep 400: 平均存活 486 步
ep 500: 平均存活 449 步
ep 600: 平均存活 497 步
```

曲线的形状也很有教育意义：前 100 个 episode 几乎在随机水平徘徊（策略梯度信号弱、方差大），随后快速爬升——这正是第二篇说的"学习信号被方差淹没"与回报标准化合力生效的过程。500 附近的震荡则来自策略接近确定性后探索不足。

## 5. 方差战争：REINFORCE 的阿喀琉斯之踵

### 5.1 方差为什么大

策略梯度是无偏估计，但方差高得吓人。三个来源：

| 来源 | 机制 | 后果 |
|:---|:---|:---|
| **回报尺度大且非负** | $G_t$ 常年 $[0, +\infty)$，均值远大于波动幅度 | 所有梯度同向"加强"，学习信号被巨大的常数分量淹没 |
| **轨迹内共享尺度** | 同一条轨迹的所有时间步共用同一个 $R(\tau)$ 的量级 | 早期动作被后期奖励"殃及"，credit assignment 噪声大 |
| **环境与策略双重随机** | 两条同样好的轨迹回报可能天差地别 | 需要海量样本才能平均出可靠梯度 |

深度学习的经验是 batch 越大方差越小，但 RL 的样本是拿真金白银（与环境交互）换来的。**所以 RL 的方差缩减不能靠堆样本，只能靠改造估计量本身**——好在数学给了两个完美的工具。

### 5.2 第一件武器：reward-to-go

朴素版本里每个动作乘的是整条轨迹的总回报 $R(\tau)$。但因果律告诉我们：$t$ 时刻的动作影响不了 $t$ 之前的奖励。把之前那些"无关奖励"从乘积里去掉，得到策略梯度的 reward-to-go 形式：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\; \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t \;\right], \qquad G_t = \sum_{k=t}^{T-1} \gamma^{k-t}\, r_{k+1}$$

为什么合法？被扔掉的项 $\sum_{k<t} \gamma^k r_{k+1}$ 在给定 $(s_t, a_t)$ 时与当前动作无关，套用下面 5.3 的论证可知减掉它不改变期望。**方差立刻下降**：少加了一堆纯噪声。

### 5.3 第二件武器：baseline——期望不变，方差骤降

**定理**：设 $b(s)$ 是任意只依赖状态、不依赖动作的函数，则：

$$\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\big[\, \nabla_\theta \log \pi_\theta(a \mid s)\; b(s) \,\big] \;=\; 0$$

证明只要三行（这是全系列最重要的三行之一）：

$$\sum_a \pi_\theta(a \mid s)\, \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)}\, b(s) \;=\; b(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) \;=\; b(s)\, \nabla_\theta \underbrace{\sum_a \pi_\theta(a \mid s)}_{=\,1} \;=\; 0$$

直觉：baseline 项在每个状态内部对所有动作"加权求和归零"，所以减掉它**不改变梯度的期望**；但它削掉了 $G_t$ 里的公共直流分量，**方差显著下降**。这就像信号处理里先去 DC 再放大——同样的信噪比下，有效信号被相对放大了。

那减什么最好？理论上最优 baseline 是按梯度范数加权的期望（推导略），但实践中最好的选择几乎总是**值函数本身**：

$$\nabla_\theta J(\theta) \;=\; \mathbb{E}\left[\; \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; \big(\, G_t - V^\pi(s_t) \,\big) \;\right]$$

括号里的 $G_t - V^\pi(s_t)$ 正是第一篇埋下伏笔的**优势函数** $A^\pi(s_t, a_t)$ 的蒙特卡洛估计——"这个动作比该状态的平均水准好多少"。至此第一篇的暗线正式接通：**V 是 baseline，A 是去掉了 baseline 的学习信号**。

### 5.4 战线汇总

| 技术 | 改动 | 期望 | 方差 |
|:---|:---|:---|:---|
| 朴素 REINFORCE | 乘 $R(\tau)$ | 无偏 | 极高 |
| reward-to-go | 乘 $G_t$ | 无偏 | ↓ |
| + baseline | 乘 $G_t - b(s_t)$ | 无偏 | ↓↓ |
| + advantage | 乘 $G_t - V^\pi(s_t)$ | 无偏 | 更低 |
| 回报标准化（代码③） | batch 内 z-score | 有轻微偏差* | ↓↓↓ |

\* 严格说 batch 统计量引入了样本间耦合，是有偏的，但实践中人人在用、效果稳定——RL 工程里"理论有偏、实践无敌"的第一个例子，后面 PPO 的 clip 会再遇到同类取舍。

## 6. on-policy 困境：好数据只能用一次

还有最后一座大山。注意策略梯度定理里的期望是对 $p_\theta(\tau)$ 取的——**梯度估计只在数据来自当前策略时无偏**。可是神经网络一更新，$\theta$ 变了，刚才采的那批数据立刻"过期"。于是 REINFORCE 的训练变成：

```
采样一批轨迹（贵）→ 更新一步参数 → 数据作废 → 再采样……
```

GPU 时代这尤其令人心痛：前向推理那么贵，数据却是一次性的。自然的想法是用旧数据凑合更新几步——数学上这叫**重要性采样**，把旧策略 $p_{\theta'}$ 下的期望修正回新策略 $\theta$ 下：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}}\left[\; \frac{p_\theta(\tau)}{p_{\theta'}(\tau)}\; \nabla_\theta \log p_\theta(\tau)\; R(\tau) \;\right]$$

问题是那个修正系数是**逐时间步比率的长连乘**：

$$\frac{p_\theta(\tau)}{p_{\theta'}(\tau)} = \prod_{t} \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta'}(a_t \mid s_t)}$$

新旧策略稍有偏离，连乘就指数级爆炸或归零，方差比 REINFORCE 本身还惨。**结论：长轨迹上直接重要性采样不可行，除非新旧策略"离得不远"**。

这个"除非"就是下一篇的全部内容：与其事后用比率纠偏，不如**主动约束每次更新不要偏离太远**——信任域（trust region）思想，TRPO 登场。

## 7. Takeaway

1. **策略梯度定理**：$\nabla_\theta J = \mathbb{E}\big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\big]$。log-derivative trick 让动力学项消失——**model-free 的数学根源**。
2. **REINFORCE = 策略梯度的蒙特卡洛实现**，能跑但方差极高；实现四坑：整轨迹采样、逆序递推 $G_t$、回报标准化、loss 取负。
3. **方差战争三板斧**：reward-to-go（因果）、baseline（与动作无关的量，期望不变方差降）、advantage（$A = Q - V$，接通第一篇暗线）。**GRPO 的组内减均值就是 baseline 思想的免 critic 版本**。
4. **on-policy 困境**：数据一次性使用；重要性采样修正在长轨迹上方差爆炸 → 必须限制新旧策略的距离。

**下一篇预告**：TRPO 将证明"新旧策略的 KL 距离有界 ⇒ 性能单调不降"的保证，把这个约束变成一个带二阶信息的优化问题——然后用共轭梯度把它做得勉强能跑。正是"勉强能跑"这四个字，催生了第四篇的主角。

---

**系列导航**

1. [从 MDP 到 GRPO（一）：强化学习的地基——MDP、贝尔曼方程与值函数](/2026/08/21/mdp-to-grpo-01-mdp-bellman-foundation/)
2. **从 MDP 到 GRPO（二）：策略梯度——REINFORCE 与方差的战争**（本篇）
3. [从 MDP 到 GRPO（三）：TRPO——信任域，让更新别摔死](/2026/08/21/mdp-to-grpo-03-trpo-trust-region/)
4. [从 MDP 到 GRPO（四）：PPO——用一阶方法驯服策略更新](/2026/08/21/mdp-to-grpo-04-ppo-clipped-surrogate/)
5. [从 MDP 到 GRPO（五）：GRPO——组相对优势与大模型时代的 RL](/2026/08/21/mdp-to-grpo-05-grpo-group-relative/)

**参考与延伸阅读**

* Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist RL" (1992) —— REINFORCE 原始论文，baseline 技巧也出自这里
* Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 13 —— 策略梯度定理的标准推导
* Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (arXiv:1506.02438) —— advantage 与方差-偏差权衡的深入分析
* OpenAI Spinning Up, "Policy Gradients" —— 本篇第 5 节结构与 spinning up 的讲法相互印证
