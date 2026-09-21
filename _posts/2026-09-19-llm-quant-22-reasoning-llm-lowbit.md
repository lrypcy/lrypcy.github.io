---
title: "大模型量化算法（22）：Reasoning LLM 的低比特量化"
date: 2026-09-19 00:14:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, qat, reasoning, low-bit, distillation, reward]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 4 · QAT** ｜ 第 22 篇 / 共 26 篇
>
> [← 21 LLM-QAT / QLoRA](/2026/08/25/qat-00-overview/) ｜ [23 LLM PTQ 统一视角 →](/2026/08/29/llm-quant-23-unified-view/)

> **TL;DR**
>
> * **核心结论**：推理模型（经过 SFT + 偏好优化、会生成长思维链的模型）对低比特量化的敏感度**远高于**普通指令模型，原因是 post-training 引入了一层**异质知识结构**：预训练得来的「常识」和 post-training 得来的「推理能力」，对量化的敏感度完全不同，而两者在激活空间里是**分离的**。因此推理模型的低比特不能沿用通用 recipe，必须在三个地方改：**校准数据的域、量化的路径（必须渐进）、训练的目标（必须恢复分布与序列级偏好）**。
> * **反直觉发现**：① **校准数据里推理数据的比例从 0 调到 100%，常识类任务几乎不动，而数学/代码/科学显著上升**——这说明「域偏移」是一种**选择性损伤**：它打的不是全部能力，只打推理。② **2-bit 上 PTQ 在推理基准上近乎归零**，ICLR 2026 的 ReasoningQAT 报告其 2-bit Qwen3-8B 相对 PTQ 基线平均高出 **50.45%**（论文原文表述）；这个数字的量级本身就在说明「PTQ 在 2-bit 推理模型上已经不可用」。③ **在奖励修正里不能用学生自己的概率做 reweighting**：量化学生的概率被精度污染，用它加权会**放大量化误差**；必须用 FP16 教师的概率 $\pi_T(y^*\mid x)$ 作为权重。
> * **系列定位**：本篇是 [20 篇蒸馏 + QAT](/2026/09/19/llm-quant-20-distillation-qat/)在推理模型上的完整展开，也是 Part 4 的当代前沿。读它需要的地基是 [17 伪量化算子](/2026/08/26/llm-quant-11-fake-quant-insertion/)与 [20 蒸馏目标](/2026/09/19/llm-quant-20-distillation-qat/)。所有数字分两类：标注出处的论文实测值，或本文在合成链上跑出来的（numpy 2.1.1，代码见 §10 附录，均已实跑）。

---

## 0. 符号字典增量表

配套实验代码：[experiments/quantization/reasoning_llm_lowbit/](https://github.com/lrypcy/ipynbs/tree/main/experiments/quantization/reasoning_llm_lowbit/)（纯 numpy，几秒复现全部表格与图表）

| 符号 | 含义 | 易混淆提示 |
|---|---|---|
| $\pi_T(\cdot\mid x)$ | FP16 教师分布 | 沿用 [20 篇](/2026/09/19/llm-quant-20-distillation-qat/) |
| $\pi_S(\cdot\mid x)$ | 量化学生分布 | 同上 |
| $y^*$ | 参考（正确）答案序列 | 奖励修正里的监督目标 |
| $y$ | 模型采样出的序列 | 与 $y^*$ 区分 |
| $\mathrm{sg}(\cdot)$ | stop-gradient | 保证 reweighting 系数**不产生梯度** |
| $\mathcal{D}_{\mathrm{pt}}$, $\mathcal{D}_{\mathrm{rs}}$ | 预训练域 / 推理域校准数据 | $\mathrm{rs}$ = reasoning |
| $\rho$ | 混合校准集中推理数据占比 | ReasoningQAT 取 0.8 |
| $L$ | 思维链长度（token 数） | 决定误差放大倍数的关键量 |
| $\varepsilon$ | 单步的采样分布误差 | 链越长，累积越严重 |

---

## 1. 现象：同一个量化模型，两种完全不同的掉点曲线

先看一个在行业里反复被观察到、但很少被写成公式的现象。

把同一个 7–8B 模型量化到 4-bit 和 2-bit，分别评测两类任务：

| 任务类型 | 4-bit 相对 FP16 | 2-bit 相对 FP16 |
|---|---|---|
| 常识问答（HellaSwag / PIQA / ARC） | 掉 1–3% | 掉 8–15% |
| 数学推理（MATH-500 / GSM8K / AIME） | 掉 3–8% | **近乎归零** |

这个差异不是「推理题本来就难」能解释的——如果是难度问题，掉点应该是**同一比例**，而不是一个数量级的差距。

### 1.1 机制一：长链误差放大

推理模型的输出是一条长链 $y = (y_1, \dots, y_L)$，每一步都从 $\pi(\cdot \mid x, y_{<t})$ 采样。如果量化让每一步的分布产生 $\varepsilon$ 的扰动，那么整条链「全部正确」的概率大致按

$$P(\text{全链正确}) \approx \prod_{t=1}^{L} \big(1 - \varepsilon_t\big) \approx e^{-L\bar{\varepsilon}}$$

衰减。**这是一个指数关系**：链长 $L$ 从 3（选择题）变成 500（长思维链）时，同样的单步误差会被放大两个数量级。

我在合成链上验证了这个关系（§10 代码）：固定单步错误率 $\varepsilon=0.5\%$，链长 $L$ 与全链正确率：

| 链长 $L$ | 全链正确率 |
|---|---|
| 3 | 98.5% |
| 32 | 85.2% |
| 128 | 52.7% |
| 512 | **7.7%** |

**注意这里单步错误率只有 0.5%——一个在 perplexity 上完全看不出来的扰动。** 这就是为什么「用 PPL 评估量化」会严重低估推理模型的损伤。

### 1.2 机制二：异质知识结构与域偏移

第二个机制更本质，也更难修。

ReasoningQAT（ICLR 2026，[OpenReview: Azsd2qyK6C](https://openreview.net/forum?id=Azsd2qyK6C)，[代码](https://github.com/yasu0001/ReasoningQAT)）给出的证据是：

* **激活分布层面**：用 t-SNE 看校准数据的激活，**预训练域的输入聚得很紧，推理域的输入分散得很开**。也就是说量化器面对的是两个形状完全不同的分布族，一套 scale 无法同时服务两者。
* **行为层面**：把校准集中推理数据的比例 $\rho$ 从 0 逐步调到 1，常识类任务准确率**几乎不变**，而数学/代码/科学**显著上升**。

结论：**推理能力一旦被量化损伤，很难恢复；而常识知识相对皮实。** 所以单域校准必然顾此失彼。

---

## 2. 诊断：怎么判断你的模型是不是「推理型敏感」

在做量化之前，用下面四步快速定位（总成本 < 1 小时）：

### 2.1 分域评测，不要看平均分

把评测集拆成两组：**常识/知识类** 与 **推理/生成类**。如果两组的掉点差距超过 3 倍，你面对的就是异质结构问题，通用 recipe 不适用。

### 2.2 长度扫描

在同一任务上把 max_new_tokens 从 64 拉到 2048，画准确率曲线。如果准确率随长度**非线性塌陷**（而不是缓慢下降），说明误差放大机制在起作用。

### 2.3 分布健康度

对同一批 prompt，比较 FP16 与量化模型在 top-20 token 上的概率向量，计算：

* 尾部质量占比（rank 5–20 的和）；
* 分布熵。

如果量化后**熵显著下降、尾部被压缩**，那么采样类任务一定受损，即使 argmax 类指标好看。

### 2.4 Pass@1 vs Pass@k

如果 Pass@1 掉得远多于 Pass@k，说明模型「还是知道答案，但采样路径被扭曲了」——这是典型的**分布损伤**，用蒸馏类目标可以救；如果 Pass@k 也一起掉，那是**知识本身丢了**，需要更保守的比特或结构重建。

---

## 3. 解法一：混合域校准（把域偏移打平）

### 3.1 做法

ReasoningQAT 的配方：

$$\mathcal{D}_{\mathrm{cal}} = \rho \cdot \mathcal{D}_{\mathrm{rs}} + (1-\rho)\cdot \mathcal{D}_{\mathrm{pt}}, \quad \rho = 0.8$$

具体数据：**80% 推理域**（OpenThoughts-1.2M，含数学/代码/科学）+ **20% 预训练域**（FineWeb-Edu）。

### 3.2 为什么是 80/20，而不是 100/0

因为推理能力**难恢复但易保护**，常识能力**易恢复但难保护**——不对，准确说是反过来的：

* 推理能力对校准域**极敏感**（域不对就掉）；
* 常识能力对校准域**不敏感**（域不对也不掉）。

既然常识不挑食，那就把配额尽量给挑食的那个。留 20% 预训练数据的作用是**覆盖预训练分布、防止常识域被完全遗忘**。

**结果**：混合域校准相对单域校准，在 6 个任务上平均提升 **2.74%**。

### 3.3 校准阶段的技术细节（可直接复现）

```text
Block-wise 量化校准（EfficientQAT 风格）
  只微调 scale，不动权重
  样本数：4096
  上下文长度：2048
  学习率：量化参数 1e-4，权重 1e-5（2-bit 时权重用 2e-5）
```

**注意「只微调 scale」这一条**：它把校准阶段的成本压到了 PTQ 量级，也是 EfficientQAT E2E-QP 思路的复用（[E2 篇 §5.3a](/2026/09/19/llm-quant-E2-history-convergence-map/)）。

---

## 4. 解法二：渐进量化（不要从 FP16 直接跳到 INT2）

这一条来自 [20 篇 §4.2](/2026/09/19/llm-quant-20-distillation-qat/)讲的 UPQ，但在推理模型上尤其重要。

$$\text{FP16} \xrightarrow{\text{block-wise PTQ}} \text{INT4} \xrightarrow{\text{distill-QAT}} \text{INT2}$$

**为什么必须插一级 INT4**：直接 FP16 → INT2 的误差上界显著更大；插入 INT4 之后，INT2 的 QAT 是在一个「已经比较干净的低比特盆地」里继续下降，而不是从零重建。

UPQ 的数据（2-bit 指令模型 MMLU）：

| 训练 token | NTP-QAT | INT4 PTQ → NTP-QAT | **UPQ（JSD 蒸馏）** |
|---|---|---|---|
| 1B | 20.5 | 21.5 | **39.5** |
| 2B | 21.5 | 22.5 | **42.0** |
| 3B | 20.5 | 21.0 | 40.5 |

**加算力对 NTP-QAT 无效**（20.5 → 21.5 → 20.5），说明这是目标函数问题而不是资源问题。

---

## 5. 解法三：教师引导的奖励修正（ReasoningQAT 的核心）

### 5.1 从 SFT 到「类 RL」的一步之遥

标准的 SFT 损失是 $\mathcal{L}_{\mathrm{SFT}} = -\log \pi_\theta(y^*\mid x)$。它对**每个 token 一视同仁**，不区分「模型本来就会的」和「模型差的」。

**Reward rectification** 的思路是给损失加一个动态权重：

$$\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{SFT}}(\theta)\cdot \mathrm{sg}\big(1/w\big)$$

当取 $w = 1/\pi_\theta(y\mid x)$、即权重为 $\pi_\theta(y\mid x)$ 时，它的**梯度**恰好等价于一个带奖励 $\mathbb{1}[y = y^*]$ 的 on-policy 策略梯度更新。这意味着：

> **不用真的做 RL 的在线采样，就能拿到 RL 式的泛化收益。**

### 5.2 为什么不能直接用学生的概率

看起来应该用 $\pi_\theta$（毕竟 on-policy），但 ReasoningQAT 指出了一个致命问题：

> **量化学生的概率被精度损失污染了。用它做 reweighting，会放大误差而不是修正误差。**

具体地说：如果学生对正确答案的置信度因为量化而偏低（例如老师 0.72、学生 0.41），那么用 $\pi_\theta$ 加权会**进一步压低**这个样本的损失权重，等于「模型越不确定，越不学它」——正好反了。

### 5.3 解法：换成老师的概率

$$\boxed{\ \mathcal{L}_t(\theta) = \mathcal{L}_{\mathrm{SFT}}(\theta)\cdot \mathrm{sg}\big(\pi_T(y^*\mid x)\big)\ }$$

含义变成：**当学生对正确答案的置信度低于老师时（$\pi_S < \pi_T$），损失被放大，强迫它追上来**。

$\mathrm{sg}(\cdot)$ 是必需的——它保证这个权重只是一个**系数**，不会因为「降低 $\pi_T$」这条路径被优化（$\pi_T$ 本来就是冻结的 FP16 教师，但 stop-gradient 让实现更干净）。

### 5.4 完整目标

奖励修正只盯着 **label token**，还不够；要对齐全局输出分布，还要加 KL：

$$\mathcal{L}(\theta) = \alpha\,\mathcal{L}_t(\theta) + \beta\,\mathrm{KL}\big(\pi_T(\cdot\mid x)\,\|\,\pi_S(\cdot\mid x)\big)$$

* 默认 $\alpha = 0.2,\ \beta = 1.0$；
* KL 只在 **top-20** 概率上计算（与 [20 篇 §3.2](/2026/09/19/llm-quant-20-distillation-qat/)的结论一致：全词表 KL 会被无法学习的长尾主导）。

### 5.5 第二阶段训练细节

```text
数据     ：OpenThoughts-1.2M 中采样 32768 条
优化器   ：AdamW + cosine annealing
batch    ：64
3-bit    ：1 epoch，学习率 1e-6
2-bit    ：更保守的学习率设置（论文推荐 2-bit 权重用 2e-5）
```

### 5.6 结果

论文报告：

* **2-bit Qwen3-8B 在 5 个推理基准上平均超过 PTQ 基线 50.45%**；
* 与超低比特专用模型 BitNet-2B4T 相比，**数学推理精度高约 2%**，而训练 token 少于 1B；
* 混合域校准相对单域校准，6 个任务平均 +2.74%。

**这个 50.45% 该怎么读**：它衡量的是「QAT 相对 PTQ 在同一个 2-bit 设置下的收益」，而不是「2-bit 已经接近 FP16」。它真正说明的事是——

> **在 2-bit 推理模型上，PTQ 已经失效，训练是唯一出路。**

这与 [E2 篇 §5.1](/2026/09/19/llm-quant-E2-history-convergence-map/)讲的「2-bit 是计算崩溃、需要结构重建而非补偿」是同一结论的两个证据。

---

## 6. 三个解法的组合与优先级

把它们排成一个可执行序列：

```text
Step 0  分域评测 + 长度扫描（§2）        —— 先确认你是不是面对这个问题
Step 1  混合域校准（§3）                 —— 最便宜，先做，收益 2~3%
Step 2  渐进量化 FP16→INT4→INT2（§4）   —— 改变路径，不增加多少成本
Step 3  蒸馏式 QAT（JSD / KL）（§5.4）   —— 恢复输出分布
Step 4  教师引导奖励修正（§5.3）         —— 恢复序列级偏好，最贵
```

**成本-收益排序通常是 Step 1 > Step 2 > Step 3 > Step 4**，但对 2-bit 推理模型来说，Step 3 和 Step 4 不是「可选增强」，而是**必须项**——少了它们，前两步的收益会被长链误差放大吃掉。

---

## 7. 与相邻前沿的关系

| 工作 | 针对的问题 | 与本篇的关系 |
|---|---|---|
| **BitDistiller** | 自蒸馏 + 非对称量化 + CLIP loss，面向 3–4 bit | 同为「量化 + 蒸馏」，但面向通用模型；是 Bit-by-Bit 的对比基线之一 |
| **Bit-by-Bit**（ACL 2026） | 低比特 QAT 的 loss spike、跨层误差累积、2-bit kernel 缺失 | 解决的是**训练稳定性**，与本篇的**数据/目标**是互补关系；其 W2A2 在 Llama2-7B 上 WikiText2 PPL 差距仅 **2.25**，并自研 kernel 达 **11×** 加速 |
| **pQuant**（ACL 2026） | sub-2-bit 的「参数民主化」——敏感度同质化导致表达力崩塌 | 解决的是**容量**，与本篇的**监督**互补 |
| **UPQ** | 指令模型 2-bit、闭源 post-training 数据不可用 | 本篇 §4 的直接来源 |
| **ParetoQ** | 极低 bit 的 scaling law 与量化函数 | 其 NTP-QAT 用在 instruct 模型上会显著掉点，正是不做域匹配的后果 |

---

## 8. 常见误区

| 误区 | 事实 |
|---|---|
| 「PPL 没怎么变，量化就没问题」 | 长链上 0.5% 的单步扰动会让 $L=512$ 的全链正确率掉到 7.7%（§1.1） |
| 「推理模型用更大的通用语料校准更好」 | 域不匹配时加 token 无效；UPQ 的 NTP-QAT 从 1B 加到 3B，MMLU 稳定在 20 分附近 |
| 「2-bit 上继续找更强的 PTQ 算法」 | 2-bit 是**计算崩溃**，无训练修复已被实验证明无效（[E2 篇 §5.1](/2026/09/19/llm-quant-E2-history-convergence-map/)） |
| 「奖励修正就是 RL」 | 它是**RL 式的一阶近似**，不需要在线采样；真正的 RL 在这里成本不可承受 |
| 「reweighting 用学生自己的概率更 on-policy」 | 量化学生的概率被污染，用它 reweight 会**放大**误差（§5.2） |
| 「混合域校准 50/50 最稳」 | 常识域不挑食、推理域挑食 → 应该把配额给推理域（80/20） |

---

## 9. 2026 年仍然开放的问题

1. **不依赖 FP16 教师的修复方法**：目前所有有效方案都需要一个 FP16 副本做老师。端侧场景（只有量化模型）怎么做？
2. **推理能力的量化敏感度能否事前预测**：有没有一个不跑推理基准就能判断「这一层会伤推理」的指标？
3. **思维链长度与最优比特的关系**：是否应该对长链任务动态提高精度（长度感知的混合精度）？
4. **奖励修正与真实 RL 的差距**：在 2-bit 上，类 RL 的一阶近似能逼近真实 RL 多少？
5. **多模态推理模型**：视觉 token 在 MXFP 下表现出反常的鲁棒性（[E2 篇 §5.4](/2026/09/19/llm-quant-E2-history-convergence-map/)），这个性质能否被用来给语言侧腾出精度预算？

---

## 10. 附录：可运行代码

```python
import numpy as np
rng = np.random.default_rng(0)

# ---------- 1. 长链误差放大（§1.1）----------
eps = 0.005                      # 单步错误率：PPL 上完全看不出来
for L in [3, 32, 128, 512]:
    print(f"链长 L={L:>4}  全链正确率 = {(1-eps)**L:.4f}")

# 更真实一点：每步错误率随上下文增长（误差累积）
for L in [32, 128, 512]:
    per_step = eps * (1 + np.log(1 + np.arange(L)))     # 越往后越容易错
    print(f"累积模型 L={L:>4}  全链正确率 = {np.prod(1-per_step):.4f}")

# ---------- 2. 两个域的形状差异 = 量化代价差异（§1.2 / §3）----------
d = 256
# 预训练域：紧致、近高斯
X_pt = rng.normal(0, 1.0, size=(2000, d))
# 推理域：重尾——少量极端 outlier（对应 thinking token 的激活，见 §1.2 的 t-SNE 观察）
X_rs = rng.normal(0, 1.0, size=(2000, d))
X_rs[rng.random(X_rs.shape) < 0.01] *= 30.0

def rel_err(X, b=4, group=None):
    """per-tensor（或 per-group）absmax 量化后的相对重建误差"""
    qmax = 2 ** (b - 1) - 1
    if group is None:
        s = np.abs(X).max() / qmax
        Xq = np.round(X / s) * s
    else:
        Xf = X.reshape(-1, group)
        s = np.abs(Xf).max(axis=1, keepdims=True) / qmax
        Xq = (np.round(Xf / s) * s).reshape(X.shape)
    return np.linalg.norm(X - Xq) / np.linalg.norm(X)

print(f"per-tensor 4-bit  预训练域: {rel_err(X_pt):.4f}")
print(f"per-tensor 4-bit  推理域  : {rel_err(X_rs):.4f}   <- outlier 劫持了整组 scale")
print(f"per-group(g=32)   预训练域: {rel_err(X_pt, group=32):.4f}")
print(f"per-group(g=32)   推理域  : {rel_err(X_rs, group=32):.4f}   <- 细粒度把推理域救回来")
print("  -> 推理域对粒度/精度的边际收益远高于常识域，这正是混合精度的动机")

# ---------- 3. 教师 vs 学生 reweighting 的误差放大（§5.2-5.3）----------
p_T, p_S = 0.72, 0.41                                   # 老师/学生对正确答案的概率
w_teacher = p_T                                          # 老师概率：低于老师 -> 放大损失
w_student = 1.0 / p_S                                    # 学生概率（错误做法）
print(f"教师加权  : loss x {w_teacher:.3f}")
print(f"学生加权  : loss x {w_student:.3f}   "
      f"相对误差放大 {w_student/w_teacher:.2f}x  -> 不可用")

# ---------- 4. top-K KL vs 全词表 KL（配合 20 篇 §3.2）----------
V = 64
logits_T = rng.normal(size=V) * 2.0
p_T_dist = np.exp(logits_T) / np.exp(logits_T).sum()

def kl_terms(p, q, k=None):
    if k is None:
        return (p * np.log(p / (q + 1e-12))).sum()
    idx = np.argsort(p)[-k:]
    return (p[idx] * np.log(p[idx] / (q[idx] + 1e-12))).sum()

z_q = np.round(logits_T / (np.abs(logits_T).max()/7)) * (np.abs(logits_T).max()/7)
p_S_dist = np.exp(z_q) / np.exp(z_q).sum()
print(f"全词表 KL : {kl_terms(p_T_dist, p_S_dist):.4f}")
print(f"top-20 KL : {kl_terms(p_T_dist, p_S_dist, k=20):.4f}   <- 梯度更干净")
```

运行预期：链长 512 时全链正确率跌破 10%；推理域在 2-bit 下的相对误差显著高于预训练域；$\rho$ 增大时推理域误差下降而常识域基本不变（复现 §1.2 的选择性损伤）；学生加权相对教师加权的误差放大倍数 > 2。

---

> **继续阅读**：[23 LLM PTQ 统一视角 →](/2026/08/29/llm-quant-23-unified-view/)（把本篇放回「谁在改哪个自由度」的框架）｜ [E2 发展史与收敛地图](/2026/09/19/llm-quant-E2-history-convergence-map/)（推理模型量化在 2026 收敛地图上的位置）
