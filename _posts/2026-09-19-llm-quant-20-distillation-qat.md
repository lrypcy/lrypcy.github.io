---
title: "大模型量化算法（20）：知识蒸馏 + QAT——低比特为什么需要老师"
date: 2026-09-19 00:11:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, qat, distillation, kd, jsd, reward-rectification]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 4 · QAT** ｜ 第 20 篇 / 共 26 篇
>
> [← 19 AdaRound / BRECQ / QDrop](/2026/09/19/llm-quant-19-adaround-brecq-qdrop/) ｜ [21 LLM-QAT / QLoRA →](/2026/08/25/qat-00-overview/)

> **TL;DR**
>
> * **核心结论**：量化模型在低比特上掉的不是「正确答案」，而是**整个输出分布的长尾结构**。逐 token 的交叉熵（CE）只监督正确 label 那一个位置，无法恢复被量化抹平的分布形状——所以低比特 QAT 必须引入一个**老师**（FP16 模型）提供稠密监督。监督信号的演进是一条清晰的谱系：**局部重建 MSE → logits KL → 特征/注意力对齐 → 分布级 JSD → 类 RL 的奖励修正**。
> * **反直觉发现**：① **在 2-bit 指令模型上，「用 next-token prediction 做 QAT」几乎完全无效**：Qualcomm 的 UPQ 给出的数据是，1B token 预算下 NTP-QAT 的 MMLU 只有 **20.5**，而换成「先 INT4 PTQ 打底 + 广义 JSD 蒸馏」的 UPQ 达到 **39.5**——同样的算力，差了一倍。原因不是训练不够，而是 **NTP 只优化条件似然，不会恢复指令跟随能力**。② **KL 的方向和支撑集很关键**：量化学生的分布尾部会「塌陷到零」，正向 KL（$\mathrm{KL}(T\|S)$）要求学生在老师有质量的地方必须有质量，反向 KL 允许学生忽略尾部；实践中常用 **top-K 截断的 KL**（如 top-20）或**对称的 JSD**来避免尾部噪声主导梯度。③ **老师不一定可靠**：ReasoningQAT（ICLR 2026）指出直接用量化学生自己的概率做 reweighting 会**放大量化误差**，必须改用 FP16 老师的概率作为权重——「谁当老师」本身是一个设计决策。
> * **系列定位**：[19 篇](/2026/09/19/llm-quant-19-adaround-brecq-qdrop/)讲了「不训练也能优化到什么程度」，本篇讲「一旦要训练，用什么信号训」。它直接决定 [22 篇 Reasoning QAT](/2026/09/19/llm-quant-22-reasoning-llm-lowbit/) 能不能成立——推理模型的低比特失败，本质上就是**监督信号选错**。所有数字分两类：标注出处的论文实测值，或本文在合成分布上跑出来的（numpy 2.1.1，代码见 §9 附录，均已实跑）。

---

## 0. 符号字典增量表

| 符号 | 含义 | 易混淆提示 |
|---|---|---|
| $\pi_T(\cdot\mid x)$ | **教师**（FP16）在输入 $x$ 上的输出分布 | 下标 T = Teacher，不是温度 |
| $\pi_S(\cdot\mid x)$ | **学生**（量化）的输出分布 | 下标 S = Student |
| $\tau$ | 蒸馏温度 | 与 Teacher 的下标 T 完全无关，本篇用 $\tau_{\mathrm{kd}}$ 避免歧义 |
| $V$ | 词表大小 | 与 19 篇的舍入变量 $V$ 无关（不同篇） |
| $\mathrm{KL}(P\|Q)$ | $\sum_i P_i \log (P_i/Q_i)$ | **非对称**；顺序有意义 |
| $\mathrm{JSD}_{\beta}$ | 广义 Jensen-Shannon 散度 | UPQ 用的目标，对称且有界 |
| $r(x,y)$ | 奖励信号 | ReasoningQAT 里取 $\mathbb{1}[y=y^*]$ 的软版本 |
| $\mathrm{sg}(\cdot)$ | stop-gradient（阻断梯度） | reward rectification 的核心算子 |
| $\mathcal{L}_{\mathrm{NTP}}$ | next-token prediction 损失 | 即普通 CE |

### 变量映射表

| 数学符号 | 代码变量 | Shape | 说明 |
|---|---|---|---|
| $\pi_T$ | `teacher_logits` / `p_t` | $(T, V)$ | FP16 模型输出 |
| $\pi_S$ | `student_logits` / `p_s` | $(T, V)$ | 量化模型输出 |
| $\tau_{\mathrm{kd}}$ | `temperature` | 标量 | 常用 1.0–2.0 |
| $\mathrm{sg}(\cdot)$ | `.detach()` | — | PyTorch 中写法 |
| $\beta$ | `jsd_beta` | 标量 | 广义 JSD 的加权系数 |

---

## 1. 问题的起点：量化到底损失了什么

先做一个实验，看清楚「掉点」的物理形态。我在一个 64 类的合成分类分布上，把 logits 量化到 INT4 per-tensor，然后比较 FP 与量化后分布的差异（代码见 §9）：

| 指标 | FP16 → INT4 后的变化 |
|---|---|
| top-1 是否改变 | 基本不变（**准确率看起来没掉**） |
| 完整分布的 KL | 显著增大 |
| 分布尾部（rank 10–64）的质量占比 | **系统性压缩** |
| 分布的熵 | 下降（**分布变「尖」**） |

**这就是低比特量化最容易骗人的地方**：用 accuracy / top-1 评测时它看起来完全没掉，因为 argmax 是鲁棒的；但分布形状已经被改写了（KL 0.0616、熵下降 0.08）。一旦任务从「选一个答案」变成**采样**（生成、推理链、agent 多步决策），这个形状差异会在多步之后被指数放大（见 [22 篇 §1.1](/2026/09/19/llm-quant-22-reasoning-llm-lowbit/)的定量推演）。

**注意一个反直觉的细节**：尾部质量占比在本例中是**上升**而不是下降的。说明量化对分布做的不是「均匀压缩」，而是**重排**——主峰附近被压尖，其余质量被重新分配。所以「用熵或 KL 单独判断量化好坏」都不可靠，真正该看的是**下游采样行为**。

这解释了一个行业里反复出现的现象：

> **量化模型在 MMLU / 选择题上掉 1–2 分，但在长链推理、代码、多轮 agent 上可能掉 20–50%。**

[E2 篇 §9.1](/2026/09/19/llm-quant-E2-history-convergence-map/)引用的 ReasoningQAT 数据就是这个现象的一个极端版本：**2-bit 的 PTQ 模型在 MATH-500 上近乎归零**，而常识类任务只掉一点。

### 1.1 所以监督信号必须换

```text
CE / NTP    ：只监督「正确 token 的概率要高」
             → 不约束其余 V-1 个位置的相对形状
             → argmax 保住了，分布塌了

KD / KL     ：监督「整个分布要像老师」
             → 约束全部 V 个位置的相对形状
             → 分布结构被恢复
```

---

## 2. 监督信号的谱系

把从 PTQ 到现代 QAT 的目标函数排成一列，会看到一条清晰的**监督粒度递增**曲线：

| 层级 | 目标 | 代表 | 能恢复什么 | 不能恢复什么 |
|---|---|---|---|---|
| L0 | 局部重建 $\lVert Wx - \hat{W}x\rVert^2$ | GPTQ / AdaRound / BRECQ | 单层输出 | 跨层、分布形状 |
| L1 | 特征对齐 $\lVert h_T - h_S\rVert^2$ | 中间层 KD | 表示空间 | 输出分布的校准 |
| L2 | logits KL $\mathrm{KL}(\pi_T\|\pi_S)$ | 经典 KD、LLM-QAT | 输出分布形状 | 长尾采样行为 |
| L3 | 广义 JSD $\mathrm{JSD}_\beta(\pi_T\|\pi_S)$ | UPQ | 分布的**双向**一致性 | 序列级奖励 |
| L4 | 奖励修正 $\mathcal{L}_{\mathrm{SFT}}\cdot \mathrm{sg}(\pi_T(y^*\mid x))$ | ReasoningQAT | **序列级**正确性偏好 | — |

**粒度越往上越贵，但低比特场景下只有 L2 以上才够用**——这是本篇的核心判断。

---

## 3. KL 的三个设计选择（每一个都会翻车）

### 3.1 方向：mode-seeking vs mode-covering

$$\mathrm{KL}(\pi_T \| \pi_S) = \sum_i \pi_T(i) \log \frac{\pi_T(i)}{\pi_S(i)}$$

* **$\mathrm{KL}(\pi_T\|\pi_S)$（forward / mode-covering）**：只要老师在某个 token 上有质量，学生也必须给质量，否则损失爆炸。**它会逼学生去覆盖老师的长尾**——但量化学生的表达力有限，硬覆盖会导致主峰被拉平。
* **$\mathrm{KL}(\pi_S\|\pi_T)$（reverse / mode-seeking）**：学生可以在老师的某些 mode 上「放弃」，只要自己给质量的地方老师也有质量。**它会让学生更集中**，但对量化模型来说，这恰好放任了「尾部塌陷」。

量化场景的推荐：**用 forward KL + top-K 截断**。forward 保证不放弃老师的 mode；top-K 把「学生无法表达的超长尾」排除在梯度之外，避免梯度被一堆接近零的数值主导。

### 3.2 支撑集：为什么 top-K 是必需品

我在合成分布上做了这个对比（§9 代码）：老师分布有 64 类，学生被 INT4 量化。三种 KL 的训练结果：

| 变体 | 最终 $\mathrm{KL}(T\|S)$ | 头部 20 类质量误差 | 相对老师的熵差 |
|---|---|---|---|
| 全词表 KL | 0.0347 | 0.0333 | +0.111 |
| **top-20 截断 KL** | **0.0318** | **0.0309** | +0.111 |
| 温度 $\tau_{\mathrm{kd}}=2$ 软化目标 | 0.2839 | 0.1183 | **+1.056** |

（学生是 fake-quantized 的 4-bit logits，代码见 §9。）

三点读法：① top-20 截断在**头部质量**和整体 KL 上都略优于全词表——因为它不把梯度浪费在学生学不动的长尾上；② 温度 2 的软化目标**灾难性变差**（熵差 +1.06，学生被训练成明显过平的分布），验证了 §3.3 的提醒；③ 三种变体的收益差距在这个 toy 上都不大，**但它们在同一方向上**——真实模型上的差距会被放大。

**注意**：top-K 截断的代价是「被截断的尾部完全不受监督」。所以实践中常配合「保留一个 other 桶」或直接用对称的 JSD。

### 3.3 温度：不是越高越好

温度 $\tau_{\mathrm{kd}}$ 的作用是软化分布、暴露 dark knowledge。但在量化场景有个额外约束：

> **学生在训练时是 fake-quantized 的，它的 logits 动态范围本身已经被量化器改变了。**

温度过高会把已经压扁的 logits 再拉平，反而让「主峰 vs 尾部」的对比度消失。**量化 QAT 里 $\tau_{\mathrm{kd}}\in[1,2]$ 是常见区间，跟经典 KD（常用 2–4）不一样**，这一点很多复现会踩坑。

---

## 4. 经典案例拆解

### 4.1 LLM-QAT（2023）：没有训练数据怎么办

LLM-QAT 的问题是：**闭源模型的原始训练数据拿不到**。

它的解法是 **data-free 自蒸馏**：

```text
FP 教师 → 采样生成合成数据 → 用这份数据训练量化学生（蒸馏）
```

同时把 weight / activation / KV cache 全部纳入 QAT，最低做到 4-bit。

**这个方案的天才之处和隐患是同一件事**：数据是老师自己生成的，所以**数据分布天然偏向老师的高概率区域**——长尾被系统性欠采样。换句话说：

> **自蒸馏会放大 §1 里那个「分布变尖」的问题，而不是修复它。**

这也是为什么后来的工作（UPQ、ReasoningQAT）都强调：**校准/训练数据的域，必须覆盖你想保住的能力**。

### 4.2 UPQ（Qualcomm AI Research, 2025）：指令模型的 2-bit 为什么必须换目标

UPQ（[arXiv:2506.09104](https://arxiv.org/abs/2506.09104)）解决的是一个很具体的痛点：**ParetoQ 那套 NTP-QAT 在 base 模型上很有效，但直接用在指令微调模型上会崩**。

原因是：指令模型的能力来自 post-training（SFT + 偏好优化）数据，而这份数据通常是**闭源的**。用开源预训练数据去做 NTP-QAT，等于在教模型「预测网页文本」，而不是「遵循指令」。

UPQ 的两阶段方案：

```text
Stage 1  FP16 → INT4：block-wise PTQ（提供一个好的低比特初值）
Stage 2  INT4 → INT2：蒸馏式 QAT，最小化广义 JSD
                     JSD_β(π_T ‖ π_S) = β·KL(π_T ‖ M) + (1-β)·KL(π_S ‖ M)
                     M = β·π_T + (1-β)·π_S
```

广义 JSD 的好处是**对称且有界**：既不让学生的尾部塌陷（forward 项），也不让学生硬扛无法表达的长尾（有界）。

论文给出的 MMLU 对比（2-bit 指令模型，随训练 token 预算变化）：

| 训练 token | NTP-QAT | INT4 PTQ → NTP-QAT | **UPQ** |
|---|---|---|---|
| 1B | 20.5 | 21.5 | **39.5** |
| 2B | 21.5 | 22.5 | **42.0** |
| 3B | 20.5 | 21.0 | 40.5 |

**三个读法**：

1. 加训练 token 对 NTP-QAT **几乎没用**（20.5 → 21.5 → 20.5，甚至回落）——说明这不是算力问题，是目标函数问题；
2. 只加 INT4 PTQ 打底也**不够**（21.5）——说明初值重要但不是决定性的；
3. 换成 JSD 蒸馏后**直接翻倍**——说明「恢复输出分布」才是 2-bit 指令模型的关键。

还有一个被低估的结论：**渐进量化（FP16 → INT4 → INT2）必须在中间插一级**。直接从 FP16 跳到 INT2 的误差上界显著更大；插了 INT4 之后，INT2 阶段的 QAT 是在一个「已经比较干净的低比特盆地」里继续下降。

### 4.3 ReasoningQAT（ICLR 2026）：连「学生自己的概率」都不能信

[ReasoningQAT](https://openreview.net/forum?id=Azsd2qyK6C) 把 reward rectification 引入 QAT。原始形式是：

$$\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{SFT}}(\theta)\cdot \mathrm{sg}\big(1/w\big)$$

当 $w = 1/\pi_\theta(y\mid x)$ 时，它的梯度等价于一个带奖励 $\mathbb{1}[y=y^*]$ 的 on-policy 策略梯度更新——**不用真的做 RL 采样，就能拿到 RL 式的泛化收益**。

但作者指出了一个致命细节：**量化学生的概率 $\pi_\theta$ 本身被精度损失污染了**，用 $1/\pi_\theta$ 做 reweighting 会放大误差。所以改成用**老师**的概率：

$$\mathcal{L}_t(\theta) = \mathcal{L}_{\mathrm{SFT}}(\theta)\cdot \mathrm{sg}\big(\pi_T(y^*\mid x)\big)$$

含义是：**当学生对正确答案的置信度低于老师时，损失被放大，强迫它追上来**。

完整目标（含分布对齐）：

$$\mathcal{L}(\theta) = \alpha\,\mathcal{L}_t(\theta) + \beta\, \mathrm{KL}\big(\pi_T(\cdot\mid x)\,\|\,\pi_S(\cdot\mid x)\big), \quad \alpha=0.2,\ \beta=1.0$$

KL 只在 **top-20** 概率上计算——与 §3.2 的结论一致。

（完整拆解见 [22 篇](/2026/09/19/llm-quant-22-reasoning-llm-lowbit/)。）

---

## 5. 老师从哪来：三种来源与各自的坑

| 来源 | 优点 | 坑 |
|---|---|---|
| **FP16 原模型**（最常见） | 监督质量最高、无需额外训练 | 需要一个能跑 FP16 的副本，显存/算力翻倍；且蒸馏时 teacher 必须保持 eval 模式 + 不更新 |
| **自蒸馏**（模型自己当老师） | 无需额外模型 | 见 §4.1：**放大分布塌缩**，且无法纠正已有偏差 |
| **更强的外部模型** | 可以「提升」而不只是「保真」 | 分布差异过大，学生学不动；且评估时会混淆「量化损失」与「蒸馏增益」 |

**实践建议**：量化 QAT 的目标应该是**保真**（fidelity），不是提升。所以用 FP16 原模型当老师，并且**在报告结果时明确区分「相对 FP16 教师恢复了百分之多少」与「相对原始 FP16 掉了多少」**——这两者在很多论文里被混用。

---

## 6. 成本账：KD-QAT 到底贵在哪

| 组件 | 相对普通 QAT 的额外开销 | 可否优化 |
|---|---|---|
| teacher 前向 | +1× 前向（且无法省，因为要 logits） | 可预先算好 logits 缓存（若数据是固定的） |
| KL / JSD 计算 | 词表维度上的逐元素运算，$O(TV)$ | top-K 截断可大幅降低 |
| 梯度反传 | 与普通 QAT 相同 | — |
| 数据 | 需要高质量的域匹配数据 | **最贵的一项**，见 §7 |

**一个容易忽略的事实**：在 LLM QAT 里，teacher 前向往往只占总成本的 20–30%，真正贵的是**数据**。ReasoningQAT 声称用 **< 1B token** 就超过了 BitNet-2B4T 约 2% 的数学精度；Bit-by-Bit 声称相对 ParetoQ **训练 token 需求降低 3600×**——这些对比的核心都是**数据效率**，而不是 FLOPs。

---

## 7. 数据域匹配：比 loss 设计更重要的一件事

这是 2026 年低比特 QAT 最重要的一条经验，值得单独强调。

**ReasoningQAT 的关键实验**：把校准/训练数据中「推理类数据」的比例从 0 逐步调到 100%，观察两类任务：

```text
常识类任务（预训练能力）：几乎不受推理数据比例影响
推理类任务（数学/代码/科学）：随推理数据比例显著上升
```

同时 t-SNE 可视化显示：**预训练输入的激活聚得很紧，推理输入的激活分散得很开**——两者是两种截然不同的分布，量化器必须分别照顾。

结论是混合域校准（80% 推理 + 20% 预训练），相对单域校准在 6 个任务上平均提升 **2.74%**。

> **工程含义**：如果你的量化模型要干推理类活，**用预训练语料做校准/蒸馏是错的**，哪怕那份语料更大、更「通用」。域不匹配造成的损失，比换一个更好的量化算法大得多。

---

## 8. 常见误区

| 误区 | 事实 |
|---|---|
| 「KD 只是锦上添花，QAT 用 CE 就行」 | 4-bit 以上也许行；**2-bit / 指令模型 / 推理模型上，NTP-QAT 与 JSD-QAT 的差距可达 2 倍**（UPQ：20.5 vs 39.5） |
| 「温度越高越好」 | 量化学生的 logits 动态范围已被改变，高温度会抹平主峰；常见区间 1–2 |
| 「KL 方向无所谓」 | forward 逼覆盖长尾、reverse 放任塌陷，量化场景应优先 forward + top-K |
| 「教师越强越好」 | 量化 QAT 的目标是保真，用更强的外部教师会混淆量化损失与蒸馏增益 |
| 「自蒸馏省事又有效」 | 自蒸馏数据偏向高概率区，会放大分布塌缩（LLM-QAT 的固有隐患） |
| 「数据越多越好」 | 见 §7：域不匹配时，加 token 完全无效（UPQ 的 NTP-QAT 20.5→21.5→20.5） |

---

## 9. 附录：可运行代码

```python
import numpy as np
rng = np.random.default_rng(0)

# ---------- 1. 量化到底损失了什么（§1）----------
V = 64
logits = rng.normal(size=V) * 4.0                       # 峰比较尖的分布（真实 LM 如此）
p_fp = np.exp(logits); p_fp /= p_fp.sum()

def quantize_logits(z, b=4):
    s = np.abs(z).max() / (2 ** (b - 1) - 1)
    return np.round(z / s) * s

z_q = quantize_logits(logits)
p_q = np.exp(z_q); p_q /= p_q.sum()

kl_full = (p_fp * np.log(p_fp / (p_q + 1e-12))).sum()
tail = np.argsort(-p_fp)[10:]                           # 按概率降序，rank 11 以后视为尾部
ent = lambda p: -(p * np.log(p + 1e-12)).sum()
print(f"top-1 是否改变        : {p_fp.argmax()} -> {p_q.argmax()}   <- argmax 很鲁棒")
print(f"完整分布 KL           : {kl_full:.4f}")
print(f"尾部质量占比 FP / Q   : {p_fp[tail].sum():.4f} / {p_q[tail].sum():.4f}")
print(f"熵 FP / Q             : {ent(p_fp):.4f} / {ent(p_q):.4f}   <- 分布被压尖或压平")
print("  -> 掉的不是 argmax，是分布形状；采样类任务（生成/推理链）会放大这个差异")

# ---------- 2. KL 变体对比（§3.1-3.3）----------
# 学生是 fake-quantized 的：前向用 4-bit 网格上的 logits，反传用 STE（这正是 QAT 的设定）
def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

zs0 = rng.normal(size=V) * 0.5
s_grid = np.abs(zs0).max() / 7

def train(target, steps=600, lr=0.5):
    z = zs0.copy()
    for _ in range(steps):
        ps = softmax(np.round(z / s_grid) * s_grid)     # 前向：量化
        z = z - lr * (ps - target)                      # 反传：softmax+KL 的解析梯度，STE
    return softmax(np.round(z / s_grid) * s_grid)

kl = lambda p, q: (p * np.log(p / (q + 1e-12))).sum()
head_err = lambda p, q, k=20: abs(p[np.argsort(p)[-k:]].sum() - q[np.argsort(p)[-k:]].sum())

t20 = np.zeros(V); idx20 = np.argsort(p_fp)[-20:]
t20[idx20] = p_fp[idx20]; t20 /= t20.sum()               # top-20 截断后重新归一化
t_t2 = p_fp ** 0.5; t_t2 /= t_t2.sum()                   # 温度 2 软化后的目标

print()
for name, tgt in [("全词表 KL", p_fp), ("top-20 KL ", t20), ("温度2 软化", t_t2)]:
    ps = train(tgt)
    print(f"{name}: KL(T||S)={kl(p_fp, ps):.4f}   头部20质量误差={head_err(p_fp, ps):.4f}   "
          f"熵差={ent(ps)-ent(p_fp):+.4f}")

# ---------- 3. 广义 JSD：UPQ 的目标（§4.2）----------
def jsd(p, q, beta=0.5):
    m = beta * p + (1 - beta) * q
    f = lambda a, b: (a * np.log(a / (b + 1e-12))).sum()
    return beta * f(p, m) + (1 - beta) * f(q, m)

print(f"\nJSD(beta=0.5) = {jsd(p_fp, p_q):.4f}   （对称、有界，上限 log 2 = {np.log(2):.4f}）")
print(f"对照 KL(T||S) = {kl_full:.4f}    （非对称、无界，尾部会主导梯度）")

# ---------- 4. Reward rectification 的 reweighting 对比（§4.3）----------
p_correct_teacher, p_correct_student = 0.72, 0.41
w_teacher = p_correct_teacher                            # 老师概率：学生低于老师 -> 放大损失
w_student = 1.0 / p_correct_student                      # 学生概率（错误做法）
print(f"\n教师加权  : loss x {w_teacher:.3f}")
print(f"学生加权  : loss x {w_student:.3f}   "
      f"相对放大 {w_student / w_teacher:.2f}x  -> 不可用")
```

运行预期（numpy 2.1.1 实跑）：量化后 top-1 不变、但尾部质量与熵都变了；三种蒸馏目标里 **top-20 截断 KL 的最终 $\mathrm{KL}(T\|S)$ 与头部质量误差都略优于全词表 KL**（0.0318 vs 0.0347），而**温度 2 软化后的目标会显著变差**（0.2839，学生分布被训练成明显过平、熵差 +1.06）——这正是 §3.3 说的「量化学生的动态范围已经被改变，高温度会抹平主峰」。

> **继续阅读**：[21 LLM-QAT / QLoRA →](/2026/08/25/qat-00-overview/)（data-free 自蒸馏与参数高效路线）｜ [22 Reasoning LLM 低比特量化](/2026/09/19/llm-quant-22-reasoning-llm-lowbit/)（本篇 §4.3 的完整展开）｜ [E2 发展史与收敛地图](/2026/09/19/llm-quant-E2-history-convergence-map/)
