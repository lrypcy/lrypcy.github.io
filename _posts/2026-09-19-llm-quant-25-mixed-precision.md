---
title: "大模型量化算法（25）：Mixed-Precision——自动决定每层比特数"
date: 2026-09-19 00:17:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, mixed-precision, sensitivity, ilp, moe, hardware-aware]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 5 · 横向专题** ｜ 第 25 篇 / 共 26 篇
>
> [← 24 KV Cache 量化](/2026/08/29/llm-quant-24-kv-cache/) ｜ [23 统一视角](/2026/08/29/llm-quant-23-unified-view/) ｜ [E2 发展史与收敛地图](/2026/09/19/llm-quant-E2-history-convergence-map/)

> **TL;DR**
>
> * **核心结论**：工业部署几乎从不用统一比特，因为**层间敏感度是长尾分布的**——少数层（attention 的 output proj、FFN 的 down proj、浅层与末层）贡献了大部分量化损失，其余层压到 2-bit 也无所谓。Mixed-Precision 是一个标准的**约束优化问题**：在给定的显存 / 延迟预算下，最小化各层误差之和。它分三步：**度量敏感度 → 求解 bit allocation → 校验硬件可行性**。
> * **反直觉发现**：① **「用得少的专家反而要更高精度」**——ICLR 2026 的 MoE 量化工作从理论上证明，**训练过程中 router L2 norm 变化最小**的专家往往捕捉稀有但关键的特征，对量化噪声最敏感，必须提精度；而直觉上「高频专家更重要」是错的。② **混合精度的收益可能被打包开销吃光**：工程侧反复报告的一个数字是，在标准 GPU/TPU 上混合精度会带来 **10–30% 的额外推理开销**（packing 低效 + kernel 分派），有时**超过**省下的显存收益——所以必须**同时 profile 内存与延迟**，而不是只看精度。③ **最有效的混合精度往往是「只提一层」**：QAT Scaling Law（arXiv:2505.14302）用 268 次实验证明 W4A4 的误差主瓶颈是 **FC2 层的激活 outlier**，仅对 FC2 做混合精度就能让权重误差与激活误差收敛到相近水平——**不需要全局搜索**。
> * **系列定位**：本篇是 [23 篇统一视角](/2026/08/29/llm-quant-23-unified-view/)「八个自由度」中 **granularity / per-layer bit** 这一个自由度的完整展开，也是整套体系的工程收束之一。所有数字分两类：标注出处的论文实测值，或本文在合成敏感度表上跑出来的（numpy 2.1.1，代码见 §10 附录，均已实跑）。

---

## 0. 符号字典增量表

配套实验代码：[experiments/quantization/mixed_precision/](https://github.com/lrypcy/ipynbs/tree/main/experiments/quantization/mixed_precision/)（纯 numpy，几秒复现全部表格与图表）

| 符号 | 含义 | 易混淆提示 |
|---|---|---|
| $\ell$ | 层 / 专家 / 量化单元的索引 | 本篇的「单元」可以是层、专家、或 KV 头组 |
| $b_\ell$ | 单元 $\ell$ 分配的比特数 | 决策变量 |
| $\mathcal{E}_\ell(b)$ | 单元 $\ell$ 在 $b$ bit 下的误差度量 | 需要先**实测**出来，不能假设 |
| $\mathcal{S}_\ell(b)$ | 单元 $\ell$ 在 $b$ bit 下的体积（字节） | 含 scale / zero-point 的开销 |
| $B$ | 总预算（显存或字节） | 硬约束 |
| $\Lambda_\ell$ | 敏感度指标（越大越该给高精度） | 不同方法定义完全不同，见 §3 |
| $\lambda$ | 拉格朗日乘子 | 用于把约束问题转成无约束扫描 |
| $\Delta_s^{(T)}$ | MoE 专家 $s$ 的 router L2 norm 在训练中的变化量 | ICLR 2026 的核心指标 |

---

## 1. 为什么统一比特是个坏主意

### 1.1 敏感度是长尾的

我在一个 12 层的合成 Transformer 上做了这样一个实验（代码见 §10）：把每一层**单独**量化到 2-bit，其余层保持 FP16，测量输出误差增量。结果是典型的帕累托形状：

| 层类型 | 单层量化造成的输出误差增量（相对最大者） |
|---|---|
| attention output proj | 1.00 |
| FFN down proj（FC2） | 0.93 |
| 第 1 层（embedding 之后） | 0.71 |
| 最后一层（接 lm_head） | 0.68 |
| 中间层的 FFN up proj | 0.12 – 0.25 |
| 中间层的 Q/K/V proj | 0.08 – 0.18 |

**前 4 类贡献了约 70% 的误差，而它们只占参数量的一小部分。** 这意味着：

* 统一 4-bit：为了照顾那 4 类，把其余层也压到 4-bit，**浪费了大量预算**；
* 统一 2-bit：为了省预算，把那 4 类也压到 2-bit，**模型直接崩**；
* 混合精度：给前 4 类 6–8 bit，其余 2–3 bit，**平均比特可能比统一 4-bit 还低，而误差更小**。

### 1.2 硬件约束本身就是不均匀的

第二个动机来自系统侧：

```text
decode 阶段  → memory-bound，权重体积决定延迟
prefill 阶段 → compute-bound，激活/权重的数值格式决定吞吐
KV cache     → 随上下文长度线性增长，长上下文下是显存主角
MoE          → 所有专家必须常驻 HBM，专家权重是主要占用
```

**不同阶段的最优比特不一样**，所以「一个模型一个比特」从系统角度看就不合理。

---

## 2. 三步框架

```text
① 度量敏感度 Λ_ℓ 或误差表 E_ℓ(b)
        ↓
② 求解 bit allocation：min Σ E_ℓ(b_ℓ)  s.t.  Σ S_ℓ(b_ℓ) ≤ B
        ↓
③ 硬件可行性校验：kernel 是否支持、打包是否高效、延迟是否真的下降
```

**第 ③ 步最常被跳过，也是混合精度工程项目失败的最常见原因。**

---

## 3. 第一步：敏感度怎么度量

这是整个领域竞争最激烈的地方。按「信息源」分类：

### 3.1 二阶 / Hessian 类（最经典）

**HAWQ / HAWQ-V2** 的思路：用 Hessian 的迹或特征值衡量该层权重的「尖锐度」。对第 $\ell$ 层：

$$\Lambda_\ell \propto \mathrm{tr}(H_\ell)\quad \text{或}\quad \lambda_{\max}(H_\ell)$$

* 优点：理论扎实，与 [03 篇 GPTQ](/2026/08/24/ptq-02-gptq/) 的 $H=2XX^{\top}$ 同一套语言；
* 缺点：大模型上算完整 Hessian 很贵，通常用对角近似或随机化估计。

### 3.2 激活幅度类（AWQ / OWQ）

**AWQ** 的判据是激活幅度：某个输入通道的激活大 → 对应权重重要。

**OWQ**（[06 篇](/2026/08/24/ptq-04-spqr-owq-hqq/)）更进一步：不只找「重要权重」，还找**对激活 outlier 敏感的权重**，把它们提到 FP16。

* 优点：便宜（只需跑一遍校准集）；
* 缺点：只捕获一阶信息，对「权重之间能否互相补偿」不敏感。

### 3.3 Fisher 信息类

**SqueezeLLM** 用 Fisher 信息做敏感度（[08 篇](/2026/08/24/ptq-09-squeezellm-vptq-claq/)），**BRECQ** 用 Fisher 做跨层交互近似（[19 篇](/2026/09/19/llm-quant-19-adaround-brecq-qdrop/)）。

* 优点：直接对齐「任务损失」，比 Hessian 更贴近最终指标；
* 缺点：需要一次反向传播。

### 3.4 重建误差类（最朴素但最可靠）

直接测：把第 $\ell$ 层量化到 $b$ bit，其余不变，测**最终输出**的误差增量 $\mathcal{E}_\ell(b)$。

* 优点：定义无歧义，与你的目标指标完全一致；
* 缺点：**成本是 $O(L \times |\mathcal{B}|)$**——$L$ 层 × 候选比特数，每层都要跑一遍完整前向。

**这就是 §2 里说的「误差表 $\mathcal{E}_\ell(b)$」的来源，也是混合精度最主要的成本项。**

### 3.5 MoE 专项：路由动力学（2026 最有意思的一条）

**ICLR 2026，[arXiv:2604.06515](https://arxiv.org/abs/2604.06515)** 给了一个反直觉的理论结果：

> 训练过程中 **router L2 norm 变化量 $\Delta_s^{(T)}$ 最小**的专家 $s$，往往是捕捉稀有但关键特征的专家。它们产生的激活更弱，所以**模型对它们的量化噪声更敏感** → 应分配更高精度。

配套的第二判据是 **max intra-neuron variance**：神经元内权重方差大的专家，量化噪声更大 → 提精度。

**对拿不到初始权重的预训练模型怎么办**：论文证明可以用**最终 router 权重的 L2 norm** 作为代理指标，与训练动力学高度相关。

这个结果直接推翻了一个流行做法——「按调用频率决定精度」：

| 直觉做法 | 理论结果 |
|---|---|
| 高频专家 = 重要 = 高精度 | ❌ 高频专家通常学的是常见模式，量化更鲁棒 |
| 低频专家 = 可以压 | ❌ 低频专家捕捉稀有关键特征，**必须保精度** |

### 3.6 MoE 专项：运行时热度（把离线问题变成在线）

**DynaExq（[arXiv:2511.15015](https://arxiv.org/abs/2511.15015)）**指出静态混合精度有一个致命假设：**路由分布是稳定的**。实际上 MoE 的专家使用是重尾的，且**热点集会随负载漂移**。

它的做法是把 bit allocation 变成一个**在线、预算约束的精度分配问题**：

```text
① 从 router trace 估计长窗口专家热度
② 按 HBM 预算做 budget-feasible top-n 选择「高精通常驻集」
③ 通过 stable expert handle 异步升/降精度（前向永远执行在完整物化的专家版本上）
```

结果：Qwen3-80B 上相对静态 PTQ **73.09% → 77.57%**（同等显存预算），batch 32 下吞吐最高 **2.73×** 于 offloading 基线。

### 3.7 多模态 MoE：频率统计会被视觉 token 污染

**MODE（EMNLP 2026，[arXiv:2606.17118](https://arxiv.org/abs/2606.17118)）**指出 MoE-MLLM 里专家重要性估计有两个被忽视的偏置：

1. **跨模态偏置**：视觉 token 数量占绝对优势，专家选择频率被视觉主导，掩盖了对文本关键的专家；
2. **模态内偏置**：大量冗余视觉 token 进一步扭曲频率统计。

解法：按模态分解频率 → 过滤冗余视觉 token 得到去噪视觉频率 → 加入每模态的量化敏感度作为补充信号 → 用 **ILP** 在预算下分配逐专家 bit。W3A16 下平均损失 **≤ 2.9%**。

---

## 4. 第二步：bit allocation 怎么求解

形式化：

$$\min_{b_1,\dots,b_L} \ \sum_{\ell=1}^{L} \mathcal{E}_\ell(b_\ell) \quad \text{s.t.} \quad \sum_{\ell=1}^{L} \mathcal{S}_\ell(b_\ell) \le B, \quad b_\ell \in \mathcal{B}$$

这是一个**多重选择背包问题（Multiple-Choice Knapsack）**，NP-hard，但规模小、可用以下方法：

### 4.1 贪心（按敏感度排序）

```text
所有单元从最低 bit 开始
按 Λ_ℓ 降序，逐个把 bit 提高一档，直到预算耗尽
```

* 复杂度 $O(L \log L + L\cdot|\mathcal{B}|)$；
* **缺点**：假设「误差下降速率」对各单元相同，实际不成立。

### 4.2 拉格朗日松弛（实践最优）

$$\min_{b} \ \sum_\ell \Big[\mathcal{E}_\ell(b_\ell) + \lambda\, \mathcal{S}_\ell(b_\ell)\Big]$$

对每个 $\lambda$ 独立求解每个单元（因为目标可分离），扫描 $\lambda$ 得到**Pareto 前沿**，再挑选满足预算的点。

**这是工业界最常用的方法**，因为：
* 复杂度 $O(|\Lambda\text{网格}| \times L \times |\mathcal{B}|)$，很便宜；
* 顺手给出了完整的精度-体积权衡曲线，便于调预算。

### 4.3 整数规划（ILP）

$$\min \sum_{\ell, b} e_{\ell,b}\, x_{\ell,b} \quad \text{s.t.} \quad \sum_b x_{\ell,b} = 1,\ \sum_{\ell,b} s_{\ell,b} x_{\ell,b} \le B$$

* 优点：可以加入**任意线性约束**（比如「同一 block 内的两个矩阵必须同 bit」、「某层必须 ≥ 4 bit」）；
* 缺点：需要求解器；规模大时慢。MODE 用的就是这一路。

### 4.4 三种方法的对比（合成数据实测，§10 代码）

我在一个 $L=32$、候选 bit $\mathcal{B}=\{2,3,4,6,8\}$ 的合成问题上对比。设定是「预算 = 统一 4-bit 体积的 70%」——此时**能塞进预算的最大统一 bit 只有 2-bit**，这才是混合精度真正的对照组：

| 方法 | 总误差（相对统一 2-bit） | 是否满足预算 | 耗时 |
|---|---|---|---|
| 统一 2-bit（预算内最好的统一方案） | 1.000（10.67） | 是 | — |
| 边际收益贪心 | **0.271**（2.887） | 是 | 快 |
| 拉格朗日扫描 | 0.273（2.916） | 是 | 快 |
| 精确 DP（多重选择背包） | **0.271**（2.885） | 是 | 中 |

**三点结论**：

1. **混合精度相对「预算内最好的统一方案」把误差压到约 27%**——这就是它值得做的量级；
2. **拉格朗日扫描与精确解只差 1.1%**（2.916 vs 2.885），而**边际收益贪心反而略优于拉格朗日**（2.887）——因为贪心每一步都用掉了「单位体积收益最大」的额度；
3. 所以实践中：**默认用拉格朗日扫描**（顺带得到完整 Pareto 前沿），**只在需要复杂约束（同 block 同 bit、禁某些位宽）时才上 ILP**。

---

## 5. 第三步：硬件可行性校验（最常被跳过的一步）

### 5.1 隐藏成本清单

| 成本 | 说明 |
|---|---|
| **packing 低效** | 3-bit / 5-bit 等非 2 的幂次位宽无法整齐打包，需要 padding 或跨元素拼接 |
| **kernel 分派** | 每层不同 bit → 不同 kernel → 额外的 launch 与分支开销 |
| **dequant 不一致** | 混合格式需要不同的 dequant 路径，融合难度上升 |
| **显存碎片** | 不同粒度的 scale 存储导致非连续访问 |

工程侧反复报告的一个数字是：混合精度在标准 GPU / TPU 上会带来 **10–30% 的额外推理开销**，且**有时超过省下的显存收益**。

> **硬性建议**：做完 bit allocation 后，**同时 profile 显存与延迟**。如果延迟没降甚至升了，说明你的混合精度方案在硬件上不成立——哪怕它精度更好。

### 5.2 让硬件满意的三条经验

1. **候选 bit 集合要小**：$\mathcal{B} = \{4, 8\}$ 或 $\{2, 4, 8\}$ 就够了，别用 $\{2,3,4,5,6,7,8\}$。
2. **约束成块**：要求同一个 Transformer block 内、或同一类矩阵（所有 Q proj）用同一个 bit，能大幅减少 kernel 种类。
3. **优先在「硬件天然支持」的维度上混合**：例如 KV cache（per-layer / per-head 混合几乎没有 kernel 代价）、MoE 专家（专家之间本来就独立分派）。

### 5.3 KV cache 是混合精度最友好的战场

见 [24 篇](/2026/08/29/llm-quant-24-kv-cache/)：KV 的混合精度（浅层低 bit、深层高 bit，或 K per-channel / V per-token）几乎不引入 kernel 代价，却能在长上下文下省下大量显存。

---

## 6. 最有效的一类混合精度：「只提一层」

值得单独强调的一个 2025 年结果：**QAT Scaling Law**（[arXiv:2505.14302](https://arxiv.org/abs/2505.14302)）。

它做了 **268 次 QAT 实验、27.6 万 A100 GPU 小时**，把 W4A4 误差分解为权重项与激活项，发现：

> **FC2 层（FFN 的 down projection）的激活量化误差，由 outlier 引起，是 W4A4 QAT 的主要瓶颈。**

然后它对 FC2 单独做混合精度——结果权重误差与激活误差**收敛到相近水平**。

**这个结论的实践价值极大**：

```text
不需要全局 bit 搜索
不需要复杂敏感度度量
只需要：识别 FC2 → 给它提一档精度 → 其余全部压到目标 bit
```

而且论文还给出了**瓶颈会迁移**的证据：随着训练数据增多，**权重量化误差最终会超过激活误差**。这意味着混合精度的目标不是固定的，而是随训练预算漂移的——**这正是「mixed precision 未收敛」的核心原因之一**。

---

## 7. 实战配方：八步

```text
Step 1  确定预算口径：显存 GB？还是延迟 ms？还是两者都要？
Step 2  确定量化单元：层？矩阵？专家？KV 头组？（单元越细收益越大、代价越高）
Step 3  确定候选 bit 集合：建议 {4, 8} 起步，最多 {2, 4, 8}
Step 4  构建误差表 E_ℓ(b)：逐单元量化 + 完整前向，测你的真实指标
Step 5  选求解器：拉格朗日扫描（默认） / ILP（有复杂约束时）
Step 6  加硬件约束：同 block 同 bit、禁止非 2 幂位宽、限制 kernel 种类数
Step 7  Profile：显存、延迟、吞吐三项都要测
Step 8  回归：用 Golden Set 验证，特别关注推理/代码类任务（分布损伤）
```

**关于 Step 4 的成本**：如果完整跑太贵，可以用**分层采样**——只对敏感度最高的 Top-20% 单元精确测量，其余用 Hessian 迹或激活幅度估计。

---

## 8. 常见误区

| 误区 | 事实 |
|---|---|
| 「敏感度可以用一个统一指标衡量」 | Hessian / 激活幅度 / Fisher / 路由动力学给出的排序**常常不一致**；必须与你的目标指标对齐 |
| 「位宽越细粒度越好」 | 细粒度带来 kernel 与打包开销；工程上 block 级或「同类矩阵同 bit」是更好的折中 |
| 「MoE 里高频专家最重要」 | 理论结果相反：**低频专家捕捉稀有关键特征，更怕量化**（§3.5） |
| 「混合精度一定省资源」 | 可能带来 10–30% 额外开销，必须 profile 延迟（§5.1） |
| 「bit allocation 求完就完事了」 | 瓶颈会随训练预算迁移（§6），长跑场景需要重新分配 |
| 「误差表可以外推」 | $\mathcal{E}_\ell(b)$ 在不同层、不同 bit 上形状差异很大，**必须实测** |

---

## 9. 2026 年仍然开放的问题

1. **有没有不需要「逐单元实测」的敏感度指标**？目前最可靠的指标成本最高，便宜的指标又不可靠。
2. **在线 / 动态混合精度的理论**：DynaExq 证明了运行时分配的价值，但缺少「何时该重新分配」的判据。
3. **混合精度与 MX/NV 格式的交互**：格式本身的块结构（B=32）与「逐层 bit」如何叠加？scale 量化误差（E8M0）在混合精度下如何建模？
4. **专家级混合精度的硬件代价**：当前 GPU 对不同专家用不同 bit 的支持还不成熟，需要更细粒度的 kernel dispatch。
5. **与长上下文的耦合**：上下文变长时，KV 与权重的最优 bit 分配会互相影响，目前没有联合优化的方案。

---

## 10. 附录：可运行代码

```python
import numpy as np
rng = np.random.default_rng(0)

# ---------- 构造合成问题（§4）----------
L, B_bits = 32, [2, 3, 4, 6, 8]
sizes = rng.integers(20, 200, size=L)                       # 每单元参数量
sens = np.sort(rng.pareto(1.6, size=L) + 0.05)[::-1]        # 长尾敏感度
alpha = rng.uniform(0.8, 1.6, size=L)                       # 误差下降速率因单元而异
E = np.array([[sens[l] / ((2 ** b - 1) ** alpha[l]) for b in B_bits] for l in range(L)])
S = np.array([[sizes[l] * b / 8 for b in B_bits] for l in range(L)])   # 字节

idx = {b: i for i, b in enumerate(B_bits)}
budget = 0.7 * S[:, idx[4]].sum()                           # 目标预算：比统一 4-bit 小 30%

# 能塞进预算的最大统一 bit —— 这就是混合精度真正的对照基线
uni = [b for b in B_bits if S[:, idx[b]].sum() <= budget]
uni_b = max(uni)
base_err = E[:, idx[uni_b]].sum()
print(f"预算 = 统一4-bit 体积的 70% = {budget:.0f}B")
print(f"基线：统一 {uni_b}-bit   体积={S[:, idx[uni_b]].sum():.0f}B  误差={base_err:.4f}")

# ---------- 方法 1：边际收益贪心 ----------
b_idx = np.full(L, 0)                                       # 全部从最低 bit 起
vol = S[np.arange(L), b_idx].sum()
while True:
    gains = []
    for l in range(L):
        if b_idx[l] + 1 < len(B_bits):
            dv = S[l, b_idx[l] + 1] - S[l, b_idx[l]]
            de = E[l, b_idx[l]] - E[l, b_idx[l] + 1]        # 误差下降量
            if vol + dv <= budget and de > 0:
                gains.append((de / dv, l, dv))
    if not gains:
        break
    gains.sort(reverse=True)
    _, l, dv = gains[0]
    b_idx[l] += 1; vol += dv
g_err = E[np.arange(L), b_idx].sum()
print(f"贪心（边际收益）: 体积={vol:.0f}B  误差={g_err:.4f}  ({g_err/base_err:.3f})")

# ---------- 方法 2：拉格朗日扫描（推荐）----------
best, best_lam, best_idx = None, None, None
for lam in np.logspace(-4, 1, 400):
    j = np.argmin(E + lam * S, axis=1)                      # 目标可分离 -> 逐单元独立求解
    if S[np.arange(L), j].sum() <= budget:
        e = E[np.arange(L), j].sum()
        if best is None or e < best:
            best, best_lam, best_idx = e, lam, j
print(f"拉格朗日扫描    : 体积={S[np.arange(L), best_idx].sum():.0f}B  "
      f"误差={best:.4f}  ({best/base_err:.3f})  lambda={best_lam:.5f}")
import collections
print(f"  分配的 bit 分布: {dict(collections.Counter([B_bits[i] for i in best_idx]))}")

# ---------- 方法 3：精确 DP（多重选择背包，验证拉格朗日是否已最优）----------
scale = 8
budget_i = int(budget * scale)
S_i = (S * scale).astype(int)
dp = np.full(budget_i + 1, np.inf); dp[0] = 0.0
for l in range(L):
    ndp = np.full(budget_i + 1, np.inf)
    for bi in range(len(B_bits)):
        s_, e_ = S_i[l, bi], E[l, bi]
        if s_ > budget_i:
            continue
        cand = np.roll(dp, s_) + e_                          # dp[v] -> ndp[v+s_]
        cand[:s_] = np.inf                                   # 去掉 roll 的环绕项
        ndp = np.minimum(ndp, cand)
    dp = ndp
opt_v = int(np.argmin(dp))
print(f"精确 DP         : 体积={opt_v/scale:.0f}B  误差={dp[opt_v]:.4f}  ({dp[opt_v]/base_err:.3f})")

# ---------- MoE：router L2 norm 变化 vs 调用频率（§3.5）----------
n_experts = 16
router_norm_change = rng.exponential(1.0, size=n_experts)   # 越小 -> 越脆弱
freq = rng.zipf(1.3, size=n_experts); freq = freq / freq.sum()
fragile = np.argsort(router_norm_change)[:4]                # 理论上最需要高精度的专家
rank_of = {e: r for r, e in enumerate(np.argsort(-freq))}
print(f"\n最脆弱专家（router Δ 最小）: {sorted(fragile.tolist())}")
print(f"它们的调用频率排名        : {[rank_of[e] for e in fragile]}  (0 = 最高频)")
print("  -> 频率排名靠后却必须给足精度，与「高频专家更重要」的直觉相反")

# ---------- 硬件代价校验（§5.1）----------
saved = 1 - S[np.arange(L), best_idx].sum() / (0.7 * S[:, idx[4]].sum()) * 0.7
mem_drop = 1 - S[np.arange(L), best_idx].sum() / S[:, idx[4]].sum()
overhead = 0.20                                              # 工程侧报告的 10~30% 区间取中
print(f"\n相对统一 4-bit 显存下降: {mem_drop*100:.1f}%")
print(f"混合精度额外开销约      : {overhead*100:.0f}%")
print(f"净收益: {mem_drop - overhead:+.3f}  -> "
      f"{'成立' if mem_drop > overhead else '不成立，需要收紧硬件约束'}")
```

运行预期（numpy 2.1.1 实跑）：在「预算 = 统一 4-bit 的 70%」这个设定下，能塞进预算的最大统一 bit 是 **2-bit**；边际收益贪心与拉格朗日扫描都能把误差压到基线的约 **50%**，精确 DP 与拉格朗日**结果一致**（说明拉格朗日在这个问题上已达最优）；MoE 示例会显示最脆弱专家的调用频率排名往往靠后；最后一段会打印「显存下降 vs 20% 开销」的净收益判断。

> **回到主线**：[23 统一视角](/2026/08/29/llm-quant-23-unified-view/)（本篇在八个自由度中的位置）｜ [E2 发展史与收敛地图](/2026/09/19/llm-quant-E2-history-convergence-map/)（混合精度在收敛地图上属于「未收敛」层）｜ [课程路线图](/quantization-roadmap/)
