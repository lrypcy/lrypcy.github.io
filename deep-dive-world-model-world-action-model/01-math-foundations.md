# 01 · WAM 的数学基础

> 本篇是整个系列的公共底座。后面每个方向的技术文档都会引用这里的符号与结论。
> 三个 Lab 全部实跑（miniconda python 3.11 / numpy），输出为真实结果。

---

## 1. 问题设定

### 1.1 POMDP

机器人/驾驶的交互过程写成部分可观测马尔可夫决策过程：

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, E, R, \gamma \rangle$$

- 状态 $s_t \in \mathcal{S}$（真实物理状态，**观测不到**）
- 动作 $a_t \in \mathcal{A}$
- 观测 $o_t \in \mathcal{O}$（图像 / 多视角图像 / 本体感觉）
- 转移 $T(s_{t+1} \mid s_t, a_t)$，**未知**
- 发射 $E(o_t \mid s_t)$
- 回报 $R(s_t, a_t)$，折扣 $\gamma$

目标是策略 $\pi(a_t \mid o_{\le t}, l)$ 最大化 $J(\pi) = \mathbb{E}\left[\sum_{t} \gamma^t R(s_t, a_t)\right]$。

关键困难是**部分可观测**：$o_t$ 里没有速度、没有接触力、没有遮挡物。所以任何只吃 $o_t$ 的反应式策略都在被迫“盲决策”——这正是 WAM 要解决的问题。

### 1.2 为什么需要世界模型

定义信念状态 $b_t = p(s_t \mid o_{\le t}, a_{<t})$。最优策略原则上应写成 $\pi^*(a_t \mid b_t)$，但 $b_t$ 不可tractable。

世界模型提供替代路径：**不推断 $s_t$，直接预测未来的观测序列**。

$$\underbrace{p(o_{t+1:t+H} \mid o_{\le t}, a_{t:t+H-1})}_{\text{观测空间动力学}}$$

这就绕过了状态推断。代价是：**预测的对象是高维观测，而控制只需要其中一小部分**——这个矛盾贯穿整个领域（见 §6 与方向 D6/D7）。

---

## 2. 三个分布：VLA / World Model / WAM

| 范式 | 建模对象 | 数学表达 | 能否回答"如果我这么做会怎样" |
|:---|:---|:---|:---|
| VLA | 条件动作分布 | $p(a_{t:t+H} \mid o_{\le t}, l)$ | ✗ |
| World Model | 观测动力学 | $p(o_{t+1:t+H} \mid o_{\le t}, a_{t:t+H-1})$ | ✓（但不产生动作） |
| **WAM** | 联合分布 | $p(o_{t+1:t+H}, a_{t:t+H} \mid o_{\le t}, l)$ | ✓ 且产生动作 |

### 2.1 两种因子分解

联合分布可以按两种方式拆，对应两大架构族：

$$\underbrace{p(o', a \mid o, l)}_{\text{Joint}} = \underbrace{p(a \mid o, l)}_{\text{policy}} \cdot \underbrace{p(o' \mid o, a, l)}_{\text{forward dynamics}} \tag{2.1}$$

$$\underbrace{p(o', a \mid o, l)}_{\text{Joint}} = \underbrace{p(o' \mid o, l)}_{\text{marginal future}} \cdot \underbrace{p(a \mid o, o', l)}_{\text{inverse dynamics}} \tag{2.2}$$

- **式 (2.1)** = **action-first**：先提动作，再推演后果。τ₀-WM 的 ACVS、所有"propose → simulate → rank"都属此类。
- **式 (2.2)** = **scene-first**：先想象未来，再反解动作。DriveWAM 明确走这条（"先生成未来视频潜变量，再从中解码 ego action，实现 inverse-dynamics 动作生成"）；VERA 的无动作视频规划器 + Jacobian IDM 也是。

**两者在数学上等价，但在学习上不等价**：

- 式 (2.1) 的困难是 $p(a|o,l)$ 是多峰的（多个动作都能完成任务），单个确定性回归头会学到"平均动作"→ 不做任何事。
- 式 (2.2) 的困难是 $p(o'|o,l)$ 的边缘分布把动作的影响**边缘化掉了**，学到的未来可能是"不做任何动作的未来"，逆动力学再怎么准也救不回来。

这就是 cascade 架构的系统性缺陷，也是 2026 年大家转向 **joint** 的数学理由（详见方向 D3）。

### 2.2 判定边界

[NUS WAM Survey (2606.20781)](https://arxiv.org/abs/2606.20781) 给出的操作性定义：只要预测的末来**服务于下列三者之一**，就是 WAM：

1. $p(a \mid o, \hat{o}')$ —— 未来**生成**动作（scene-first）
2. $s(a; \hat{o}'(a))$ —— 未来**给动作打分**（action-first / planning）
3. $\nabla_\theta \mathcal{L}_{\text{action}}$ 经过未来通路 —— 未来**训练**动作通路（SimWAM 的"视频生成只作训练信号"）

第 3 条很重要：它意味着**推理时不需要真的生成视频**，只要训练时未来预测塑造了表征，就算 WAM。[SimWAM (2608.07468)](https://arxiv.org/abs/2608.07468) 与 [Faster-WAM (2608.02365)](https://arxiv.org/abs/2608.02365) 都落在这条上。

---

## 3. WAM = Foundation-Model 化的 Model-Based RL

### 3.1 经典 MBRL 的形式

Dreamer 系列的 latent imagination：

$$
\begin{aligned}
\text{表征：}&\quad z_t \sim q_\phi(z_t \mid z_{t-1}, a_{t-1}, o_t) \\
\text{动力学：}&\quad \hat{z}_t \sim p_\phi(\hat{z}_t \mid z_{t-1}, a_{t-1}) \\
\text{解码/回报/价值：}&\quad \hat{o}_t \sim p_\phi(\hat{o}_t \mid z_t),\ \ \hat{r}_t \sim p_\phi(\hat{r}_t \mid z_t),\ \ v_t \approx V_\phi(z_t)
\end{aligned}
$$

训练目标（变分下界 + 回报/价值回归）：

$$\mathcal{L} = \underbrace{\mathbb{E}_q\left[\log p(\hat{o}_t \mid z_t)\right]}_{\text{重建}} - \beta \underbrace{\mathbb{E}_q\left[\mathrm{KL}\left(q(z_t \mid \cdots) \,\|\, p(\hat{z}_t \mid \cdots)\right)\right]}_{\text{动力学正则}} + \mathcal{L}_r + \mathcal{L}_V \tag{3.1}$$

然后**在潜空间想象**：从 $z_t$ 出发，用 $p_\phi$ 和 actor 展开 $H$ 步，累积 $\hat{r}$，用 $\lambda$-return 更新 actor/critic：

$$V^\lambda_t = \hat{r}_t + \gamma\left[(1-\lambda) v_{t+1} + \lambda V^\lambda_{t+1}\right] \tag{3.2}$$

### 3.2 WAM 与它的差别

WAM 把 $p_\phi(\hat{z}_{t+1} \mid z_t, a_t)$ 换成了**视频扩散/流匹配模型**，把 $z$ 换成了**视频 latent 序列**。差别有三：

| | 经典 MBRL | WAM |
|:---|:---|:---|
| 潜变量来源 | 学到的 RSSM 状态（紧凑、任务特定） | 预训练 VAE 的视频 latent（冗余、通用） |
| 动力学 | 单步马尔可夫转移 | **chunk 级**序列生成（一次 $H$ 帧） |
| 监督 | 单任务交互数据 | 互联网视频 + 机器人轨迹 + 人类第一视角 |
| 规划 | 潜空间 rollout + 梯度/MPC | 采样候选 → 世界模型打分 → rectify |

**数学上的核心差别**：WAM 的"dynamics"不是 $p(z_{t+1}|z_t,a_t)$ 的一步转移，而是**条件分布**

$$p_\theta\big(x^{(0)}_{1:H} \,\big|\, x^{(0)}_0 = \text{obs},\ a_{0:H},\ l\big)$$

其中 $x^{(0)}$ 是干净视频 latent，生成过程由 flow matching 参数化（见 §4）。这带来两个后果：

1. **多步一致性天然更好**（一次生成整个 chunk，而不是自回归递推）——代价是动作必须是**已知的** chunk，不能边想边改。
2. **不能做单步反事实查询**——要问"第 3 步换个动作会怎样"，必须重新生成整个 chunk。这是 chunked 部署的根本限制，[WALL-WM (2606.01955)](https://arxiv.org/abs/2606.01955) 的"事件级切分"正是针对这一点。

---

## 4. 生成引擎：Flow Matching 与 Rectified Flow

### 4.1 条件流匹配（Conditional Flow Matching）

给定数据 $x_1 \sim p_{\text{data}}$，噪声 $x_0 \sim \mathcal{N}(0, I)$，构造**概率路径** $x_t = \psi_t(x_0, x_1)$，其速度场为 $u_t = \dot{\psi}_t$。

**Rectified flow 取直线插值**：

$$x_t = (1-t)\,x_0 + t\,x_1, \qquad u_t(x \mid x_1) = x_1 - x_0 \tag{4.1}$$

注意这个速度场**与 $t$ 无关**，且对给定的 $(x_0,x_1)$ 是常数。

训练目标是回归这个条件速度场：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1, c}\left[\big\|\, v_\theta(x_t, t, c) - (x_1 - x_0)\,\big\|^2\right], \quad t \sim \mathcal{U}[0,1] \tag{4.2}$$

其中 $c$ 是条件（文本、观测帧、动作）。**注意 (4.2) 是条件期望**，因此学到的边缘速度场是

$$v_\theta(x, t, c) = \mathbb{E}\big[x_1 - x_0 \,\big|\, x_t = x, c\big]$$

这是一个**后验均值**，不是某一个样本的速度——这也是为什么流匹配在模式覆盖上天然是"平均化"的，多峰问题时需要额外机制（见方向 D9 的 MM-Future）。

### 4.2 为什么比 DDPM 省步数

DDPM / VP-SDE 的路径是弯的：$x_t = \sqrt{\bar\alpha_t} x_1 + \sqrt{1-\bar\alpha_t} x_0$，其速度场 $\dot{x}_t$ 强烈依赖 $t$。用 Euler 法解 ODE：

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)$$

局部截断误差 $\propto \|\ddot{x}_t\| \Delta t^2$，全局误差 $\propto \|\ddot{x}\| \Delta t$。**直线路径的 $\ddot{x} \equiv 0$**，所以理论上一步到位；弯路径误差随步数线性下降。

**Lab C：实测**

```
== Lab C: 直线路径 vs 曲线路径的 Euler 积分误差 (d=64) ==
  NFE=  1   直线(rectified flow) = 4.264e-16    余弦曲线 = 8.290e+00
  NFE=  2   直线(rectified flow) = 6.939e-16    余弦曲线 = 4.041e+00
  NFE=  4   直线(rectified flow) = 7.805e-16    余弦曲线 = 2.009e+00
  NFE=  8   直线(rectified flow) = 1.545e-15    余弦曲线 = 1.003e+00
  NFE= 16   直线(rectified flow) = 3.209e-15    余弦曲线 = 5.014e-01
```

直线路径在任何 NFE 下都是机器精度；余弦路径误差精确地 $\propto 1/\text{NFE}$。这就是"**NFE 从 50 降到 4**"能成立的数学根据，也是 [MotuBrain](https://arxiv.org/abs/2604.27792) 的 step reduction、[DreamZero](https://arxiv.org/abs/2602.15922) 的 38× 加速能落地的前提。

> 现实中直线路径达不到机器精度，因为 $v_\theta$ 是神经网络拟合的后验均值，有拟合误差；但**误差随 NFE 的衰减速率**这一结论保留。

### 4.3 动作与观测的联合流匹配

把视频 latent $x$ 与动作 chunk $a$ 拼成“联合样本” $y = [x; a]$，共用一个速度场：

$$y_t = (1-t) y_0 + t y_1, \quad \mathcal{L} = \mathbb{E}\big\|v_\theta(y_t, t, c) - (y_1 - y_0)\big\|^2 \tag{4.3}$$

这就是 [MotuBrain](https://arxiv.org/abs/2604.27792) 的 UniDiffuser 形式与 [DreamZero](https://arxiv.org/abs/2602.15922) 的 joint flow matching。

**关键设计自由度在“可见性掩码”**：哪些 token 能看见哪些 token。设掩码矩阵 $M \in \{0,1\}^{N \times N}$，注意力

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \log M\right) V$$

- $M$ 全 1 → 完全联合，双向耦合（MotuBrain）
- 未来视频 token 与动作 token 互相不可见 → 模态解耦（[UNIVERSE (2607.05133)](https://arxiv.org/abs/2607.05133) 的 Modality-Decoupling Visibility Mask，防未来目标泄漏）
- 动作 token 看不见**未来**视频帧，只看历史 → 隔离掩码（[SimWAM](https://arxiv.org/abs/2608.07468)），使推理时可丢弃视频分支

三种掩码的取舍见方向 D3、D9。

---

## 5. 动作条件注入的四种数学形式

给定观测 latent $z^{\text{obs}}$ 与动作 $a$，注入方式决定了动作能“管到哪里”：

| 方式 | 公式 | 代表 | 特点 |
|:---|:---|:---|:---|
| 拼接 token | $z = [z^{\text{obs}};\, \phi(a)]$ | τ₀-WM、SimWAM | 动作与视频同一序列，注意力自然交互 |
| 自适应归一化 | $z' = \gamma(a) \odot \frac{z-\mu}{\sigma} + \beta(a)$ | DiT 标准 AdaLN | 参数量小，但只能做**全局**调制，空间定位弱 |
| 交叉注意力 | $\mathrm{Attn}(Q_{\text{vid}}, K_{\text{act}}, V_{\text{act}})$ | DriveWAM 的 VLM 引导 | 语义层引导，密度低 |
| **潜帧注入** | 把 $a$ 编码成“额外的一帧”塞进视频序列 | **Cosmos Policy** | **不改架构**，动作/未来/值都是帧 |

### 5.1 潜帧注入（latent frame injection）

[Cosmos Policy (2601.16163)](https://arxiv.org/abs/2601.16163) 的做法值得单独说：它把一段视频 latent 序列的 11 个“帧槽”分配为

$$\big[\,\varnothing,\ q^{\text{proprio}},\ I^{\text{wrist}},\ I^{\text{cam}_1},\ I^{\text{cam}_2},\ \mathbf{a}^{\text{chunk}},\ \hat{q}^{\text{future}},\ \hat{I}^{\text{wrist}},\ \hat{I}^{\text{cam}_1},\ \hat{I}^{\text{cam}_2},\ \mathbf{V}\,\big]$$

即 **action chunk 是帧，future value 也是帧**。整个模型仍是“生成 11 帧视频”，损失函数一行都不用改。

数学含义：把条件分布 $p(o', a, V \mid o, l)$ 编码成**一个序列生成问题**，而不是三个头。
- 优点：零架构改动，直接继承视频预训练权重。
- 代价：动作被强行塞进视频的时空归纳偏置里（相邻帧应该“空间连续”），这在物理上是错的——动作帧和图像帧不应该共享空间平滑先验。**这是“不改架构”路线的理论上限**，也解释了为什么后来有 Faster-WAM 的显式解耦。

---

## 6. 动作对齐（Action Alignment）：WAM 最重要的形式化

### 6.1 问题的数学表述

设世界模型 $f_\theta(o, a) \mapsto \hat{o}'$。定义两个量：

**预测误差**（评测常用）：

$$\varepsilon_{\text{pred}} = \mathbb{E}\big[\,d(\hat{o}', o')\,\big]$$

**动作敏感度**（真正关心的）：

$$\mathcal{S}(f) = \mathbb{E}_{o, a_1, a_2}\left[\frac{d\big(f(o,a_1),\, f(o,a_2)\big)}{d(a_1, a_2)}\right] \tag{6.1}$$

**决定性结论**：$\varepsilon_{\text{pred}}$ 小 **推不出** $\mathcal{S}$ 大。

形式化地，把未来分解为动作相关部分与无关部分：$o' = g(o, a) + \xi$，其中 $\xi \perp a$（光照、纹理、背景、不可控物体）。若 $\mathrm{Var}[\xi] \gg \mathrm{Var}[g]$，则

$$\inf_f \varepsilon_{\text{pred}} \approx \mathrm{Var}[\xi]$$

而“忽略动作”的模型 $f_0(o,a) = \mathbb{E}[o' \mid o]$ 也能达到 $\varepsilon_{\text{pred}} \approx \mathrm{Var}[\xi] + \mathrm{Var}[g]$，**与最优模型的差距只有 $\mathrm{Var}[g]$**——但 $\mathcal{S}(f_0) = 0$。

### 6.2 信息论刻画

动作对齐的严格版本是**条件互信息**：

$$I\big(A ;\, \hat{O}' \mid O \big) = H(\hat{O}' \mid O) - H(\hat{O}' \mid O, A)$$

模型对动作不敏感 ⟺ $I \to 0$。这给出一个可训练的目标（最大化互信息的下界），但目前**没有工作把它作为主损失**——这是领域里一个明显的空白。

### 6.3 Lab A：实测

构造 toy 动力学 $s' = A s + B a + \xi$，其中刻意把动作增益 $B$ 做小、噪声 $\xi$ 做大（这模拟真实机器人数据里“动作只解释未来变化的一小部分”）。拟合两个线性模型：一个用 $(s,a)$，一个只用 $s$。

```
== Lab A: MSE 看起来差不多，决策能力已经塌了 ==
  MSE   动作感知模型 = 7.9780
  MSE   动作无关模型 = 8.8042   (比值 1.104 —— 只差 10.4%)
  动作敏感度  感知 = 0.4371    无关 = 0.000000
  Top-1 命中(K=16, 随机=0.062)  感知 = 0.963   无关 = 0.060
  与真值 Pearson          感知 = 0.999   无关 = nan
```

读法：

- 两个模型的 MSE 只差 **10.4%**——如果只看像素/状态空间误差，你会认为“动作无关模型也还行”。
- 但动作敏感度一个是 **0.437**，一个是 **0**。
- 下游决策（16 个候选里挑最好的）：感知模型 **96.3%** 命中，无关模型 **6.0%** ——正好是随机猜（1/16 = 6.25%）。
- Pearson 相关系数对无关模型是 **nan**：因为它对所有候选给出同一个预测，方差为 0，相关系数无定义。这个 nan 本身就是最好的诊断信号。

**结论**：WAM 的评测不能只看 $\varepsilon_{\text{pred}}$。至少要同时报 $\mathcal{S}(f)$ 和一个决策层指标（Top-1 命中 / 排序相关）。这正是 [OSCAR](https://arxiv.org/abs/2606.04463) 报告 Spearman/Pearson 排序相关、[WorldGym](https://arxiv.org/abs/2506.00613) 报告“排序保持”的原因（详见方向 D5）。

---

## 7. 长视野误差累积：谱半径定理

### 7.1 定理

设模型推演 $\hat{s}_{t+1} = f(\hat{s}_t, a_t)$，真动力学 $s_{t+1} = F s_t + \cdots$。若

$$\|f(s, a) - f(s', a)\| \le \rho \|s - s'\|, \qquad \|f(s,a) - F s - G a\| \le \varepsilon$$

则**开环**推演（不重新观测）的误差 $e_t = \|\hat{s}_t - s_t\|$ 满足

$$e_H \le \varepsilon \sum_{k=0}^{H-1} \rho^k = \begin{cases}
\varepsilon \dfrac{1-\rho^H}{1-\rho} & \rho \ne 1 \\[6pt]
\varepsilon H & \rho = 1
\end{cases} \tag{7.1}$$

三个regime：

- $\rho < 1$：**有界**，$e_\infty \le \varepsilon/(1-\rho)$。这是“收缩动力学”的好处。
- $\rho = 1$：线性增长 $\varepsilon H$。
- $\rho > 1$：**指数爆炸** $\varepsilon \rho^H$。32 步、$\rho=1.05$、$\varepsilon=0.02$ 时界为 $1.5$ 量级——已经完全不可用。

### 7.2 观测刷新把开环变成“分段有界”

若每 $k$ 步用真实观测重置一次（$\hat{s} \leftarrow s$），则误差上界变成

$$e_H^{\text{refresh}} \le \varepsilon \sum_{j=0}^{k-1} \rho^j = \varepsilon\frac{1-\rho^k}{1-\rho} \tag{7.2}$$

**与总视野 $H$ 无关**。这是“periodic observation refresh”这一工程技巧的完整数学根据。

### 7.3 Lab B：实测

```
== Lab B: 开环 vs 周期观测刷新 (H=32, eps=0.02, 报全程均值 / 峰值) ==
  rho=0.90  开环 均值0.125/峰值0.146 | k=16 均值0.103/峰值0.131 | k=8 均值0.085/峰值0.113 | k=4 均值0.070/峰值0.098 | k=1 均值0.047/峰值0.050
  rho=0.99  开环 均值0.266/峰值0.561 | k=16 均值0.143/峰值0.225 | k=8 均值0.099/峰值0.149 | k=4 均值0.075/峰值0.111 | k=1 均值0.047/峰值0.050
  rho=1.00  开环 均值0.302/峰值0.692 | k=16 均值0.150/峰值0.245 | k=8 均值0.101/峰值0.155 | k=4 均值0.076/峰值0.112 | k=1 均值0.047/峰值0.050
  rho=1.05  开环 均值0.644/峰值2.156 | k=16 均值0.194/峰值0.389 | k=8 均值0.112/峰值0.185 | k=4 均值0.079/峰值0.121 | k=1 均值0.047/峰值0.050
```

三条结论：

1. **$\rho$ 是主旋钮**：同样 $\varepsilon=0.02$，$\rho=0.90 \to 1.05$ 让开环末误差从 0.15 涨到 **2.16**（14 倍）。
2. **刷新周期 $k$ 的收益是次线性的**：$k=16 \to k=4$ 只把峰值从 0.389 降到 0.121，而 $k=4 \to k=1$ 只从 0.121 降到 0.050。**$k=4$–$8$ 是性价比拐点**——这对实时部署很关键，因为刷新要等真机观测，有延迟成本。
3. **刷新把 $\rho$ 的影响几乎抹平**：$k=1$ 时四个 $\rho$ 的误差完全一样（0.047）。这说明**长程一致性的治本方法是闭环刷新，不是把 context window 做大**。

对应到工程：[MotuBrain](https://arxiv.org/abs/2604.27792) 的“real-time chunked closed-loop execution”、[DriveWAM](https://arxiv.org/abs/2605.28544) 的“selective KV memory”都是这条定理的不同实现。

---

## 8. 规划的形式化

### 8.1 规划问题

给定世界模型 $f$ 与打分器 $J$（可以是学到的 value，也可以是 VLM 裁判）：

$$a^* = \arg\max_{a \in \mathcal{A}} J\big(f(o, a)\big) \tag{8.1}$$

$\mathcal{A}$ 是连续高维空间，无法枚举。三条主流解法：

**（a）Best-of-N 采样**（τ₀-WM、Cosmos Policy）

$$a_1, \dots, a_N \sim p_\theta(a \mid o, l), \qquad a^* = \arg\max_i J\big(f(o, a_i)\big)$$

样本复杂度：要覆盖 $\delta$-最优动作，需要 $N \gtrsim 1/p(\text{good})$。**没有 rectify 时，N 的收益是对数级的**——这是它便宜的原因。

**（b）CEM / MPPI**（经典，正在回流）

维护高斯 $\mathcal{N}(\mu_t, \Sigma_t)$，每轮：采样 $N$ 个 → 按 $J$ 取 top-$k$ → 重拟合 $\mu, \Sigma$：

$$\mu_{t+1} = \frac{\sum_{i \in \text{top-}k} w_i a_i}{\sum w_i}, \quad \Sigma_{t+1} = \frac{\sum_{i \in \text{top-}k} w_i (a_i - \mu_{t+1})(a_i - \mu_{t+1})^\top}{\sum w_i}$$

比 best-of-N 高效，但连续动作空间里 $J$ 往往是**不可微的神经网络**，只能用零阶优化。

**（c）Rectify**（τ₀-WM 的新机制）

不满足于“从已有候选里选”，而是**用最好的想象未来重新生成动作**：

$$a^{\text{new}} \sim p_\theta\big(a \mid o, l, \hat{o}'_{*}\big), \quad \hat{o}'_{*} = f(o, a_{i^*}) \tag{8.2}$$

数学上这是在做一步 **EM / 坐标上升**：固定未来 → 改进动作 → 再固定动作 → 改进未来。它把式 (2.1) 和 (2.2) 两个因子分解**交替使用**，这是对 §2.1 那个“cascade 缺陷”的直接修补。

### 8.2 一致性打分（re-denoising consistency）

τ₀-WM 的排序分数：把采样的候选动作 $a_i$ **加噪再重新去噪**，看它是否回到自己：

$$s(a_i) = -\Big\|\, \underbrace{v_\theta \circ \cdots \circ v_\theta}_{\text{re-denoise}}\big(a_i + \sigma \epsilon\big) - a_i \,\Big\|$$

这是**无需外部裁判**的质量度量，本质是检查 $a_i$ 是否位于学到的条件分布 $p(a|o,l)$ 的高密度区。它与 value 打分互补：一致性高 = “这个动作像模型会做的”，value 高 = “这个动作的后果好”。两者都高才执行。

---

## 9. 评测：为什么 FVD / PSNR 不够

把 §6 和 §7 的结论合起来，可以给出 WAM 应该报的四类指标：

| 层次 | 指标 | 衡量什么 | 是否与控制相关 |
|:---|:---|:---|:---|
| L1 视觉保真 | PSNR / SSIM / LPIPS / FVD | 生成得像不像 | ✗ |
| L2 物理一致 | 物体持久性、接触、遮挡正确性 | 物理是否合理 | 间接 |
| L3 **动作对齐** | 敏感度 $\mathcal{S}(f)$、反事实准确率 $I(A;\hat{O}'\mid O)$ | 未来是否随动作变 | ✓✓ |
| L4 **决策相关** | Top-1 命中、Spearman/Pearson 排序相关、任务成功率 | 选出来的动作好不好 | ✓✓✓ |

L1/L2 高而 L3/L4 低，是“漂亮但没用的世界模型”——Lab A 里那个 MSE 只差 10% 的模型就是典型。

[OSCAR](https://arxiv.org/abs/2606.04463) 的 PSNR 24.24 / SSIM 0.846 / LPIPS 0.094 / FVD 7.08 属于 L1；它真正有价值的数字是 **Spearman 0.750 / Pearson 0.852**（L4）。

---

## 10. 本篇要点

1. WAM 的联合分布有两种因子分解（action-first / scene-first），**数学等价、学习不等价**——cascade 的系统性缺陷来源于此（§2.1）。
2. WAM 是“foundation-model 化的 MBRL”：把单步潜转移换成 chunk 级条件生成（§3.2）。
3. Rectified flow 的**直线路径**使 Euler 误差与步数无关（理论上），这是 NFE 从 50 → 4 的根据（§4.2，Lab C）。
4. **动作对齐是核心难题**：$\varepsilon_{\text{pred}}$ 小推不出敏感度高（§6.1，Lab A）。
5. **长视野误差由谱半径 $\rho$ 主导，观测刷新把它变成与 $H$ 无关**（§7，Lab B），且 $k=4$–$8$ 是性价比拐点。
6. 评测必须报 L3（动作对齐）和 L4（决策相关），只报 L1 会系统性高估模型（§9）。

---

**上一篇**：[00 · 发展方向地图](./00-wam-directions-map.md) ｜ **下一篇**：[02 · 视频基座模型作为世界先验](./02-video-foundation-prior.md)
