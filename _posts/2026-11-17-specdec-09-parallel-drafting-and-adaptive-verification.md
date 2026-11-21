---
title: "投机解码（09）：并行草稿与自适应验证——从「把草稿变准」到「把验证变省」"
date: 2026-11-17 12:00:00 +0800
categories:
  - 推理服务
tags: [speculative-decoding, parallel-drafting, dflash, dspark, p-eagle, adaptive-spec, specforge, vllm, inference]
layout: post
mathjax: true
---

> **投机解码系列 · 第 09 篇**
>
> 01–08 已经把「收益由哪三个数决定」「草稿从哪来」「引擎能不能开」「论文数字怎么读」写透。本篇只处理
> 2026 年下半年新长出来的那一层：**草稿不再逐 token 串行生成，验证也不再是固定形状、固定长度、固定规则**。
> 它挂在[推理引擎主线](/2026/10/18/serving-01-architecture-divergence/)之下，承接
> [第 02 篇](/2026/10/18/specdec-02-acceptance-and-speedup-model/)的收益模型与
> [第 03 篇](/2026/10/18/specdec-03-eagle-family/)的草稿来源谱系。
>
> - 01 综述与演化 → 02 收益模型 → 03 EAGLE 家族 → 04 引擎支持矩阵 → 05 生产部署陷阱 →
>   06 业界实践 → 07 评测数字怎么读 → 08 未来方向 → **09 并行草稿与自适应验证（本文）**

> **TL;DR**
>
> * **核心问题与结论**：2026 年的重心从「把草稿变准」转到「拆掉草稿的串行依赖、让运行时决定验证多少」。P-EAGLE、DFlash、DSpark 一次前向出整块；固定配置下草稿耗时对块长近似不敏感，验证仍随候选数增长，[第 02 篇](/2026/10/18/specdec-02-acceptance-and-speedup-model/)的收益模型要换分母，最优块长写成随负载移动的 argmax。
> * **关键证据**：DFlash（[arXiv:2602.06036](https://arxiv.org/abs/2602.06036)）自报超 6 倍无损加速；合成探针用 Red Hat 逐位置接受率反算：它是前缀率 $u_i$，误用条件率连乘低估 25.4%，常数 $\alpha$ 再高估 18.00%；$K^*$ 随 batch 1→256 从 7 降到 4（[实验输出](https://github.com/lrypcy/ipynbs/blob/8732811b95a7007bfa53d3d38e57aa68233f9a85/experiments/speculative-decoding/parallel-drafting/results/stdout.txt)，commit `8732811`）。
> * **适用边界与代价**：验证侧仍是瓶颈，高并发可能负收益；截断无损取决于决策信号是否与提议分布一致，margin 偏差在同一阈值下随分布差数倍。工程上 EAGLE-3 最成熟，P-EAGLE 训练已弃用，AdaptiveSpec 公开源码待核实。
> * **系列定位**：本篇不重证无偏性，只答新架构改了收益模型哪一项、代码生态里什么真能跑。

---

## 0. 符号与口径约定

沿用[第 02 篇](/2026/10/18/specdec-02-acceptance-and-speedup-model/)的记号：目标模型分布 $p$、草稿分布 $q$、草稿长度 $\gamma$、单步接受概率 $\alpha$、单次草稿前向与单次目标前向的成本比 $c = t_D / t_T$，以及一阶收益模型

$$
S(\alpha, \gamma, c) = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(1 + \gamma c)}
$$

本篇新增的记号：

| 符号 | 含义 |
|:---|:---|
| $T_d(K)$ | 一轮里**生成草稿**的总耗时（串行草稿为 $K t_d$，并行草稿为整块 $t_b$） |
| $T_v(K; B, L)$ | 一轮里**验证候选**的耗时，随候选数 $K$ 增长，也依赖批大小 $B$ 与上下文长度 $L$ |
| $K$ | 并行草稿一次前向产出的块长度（与 $\gamma$ 同量纲，但成本结构不同） |
| $u_i$ | **前缀接受率**：第 $i$ 个草稿位置仍被接受的概率，即 $P(A_1 \wedge \cdots \wedge A_i)$ |
| $s_i$ | **条件接受率**：在前 $i-1$ 个都被接受的条件下第 $i$ 个被接受的概率，$u_i = \prod_{j=1}^{i} s_j$ |
| $r$ | 条件接受率的几何衰减率，$s_{i+1} \approx r\, s_i$ |
| $c_b$ | 整块草稿一次前向相对目标前向的成本比；$\beta$ 是验证侧随候选位置数增加的增量 |
| $\tau$ | 有损验证的 margin 阈值 |
| $\mathcal A$ | 训练时采样的锚点位置集合 |
| $T_s$ | 调度、采样、KV 管理等与候选数无关的固定开销 |
| $\mathrm{TV}(p, q)$ | 两个离散分布的总变差距离 |

这里必须把**前缀接受率**与**条件接受率**分开，否则 §7.1 的推导会自我矛盾：期望产出是 $E[L] = 1 + \sum_{i=1}^{K} u_i$，只有在各位置独立、$u_i = \alpha^i$ 时才退化成第 02 篇的等比求和；$s_i$ 要通过连乘还原 $u_i$。Red Hat 公布的逐位置表是 $u_i$，分寸差一点，反算值就会偏低四分之一（见 §7.1 与实验组 6）。

三条口径先钉死，后面不再重复：

1. **不重证无偏性**。标准拒绝采样为什么让输出分布仍等于 $p$，属于[采样实现深读](/2026/09/30/pytorch-sampling-internals/) §4；本篇只在 §5、§6 讨论「哪些改动动了接受规则，哪些只动了候选长度」。
2. **合成探针与论文数字分开标**。本文所有带四位小数的数字来自
   [配套实验](https://github.com/lrypcy/ipynbs/blob/8732811b95a7007bfa53d3d38e57aa68233f9a85/experiments/speculative-decoding/parallel-drafting/parallel_drafting_lab.ipynb)（纯 numpy、固定 seed、29 条断言）；论文与官方博客的数字一律写明出处与实验设定，两者不混在一张表里。

---

本篇要放的技术都在下面这张图的三条轴上，先看地图再读推导：

<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC, sans-serif">
  <rect x="0" y="0" width="680" height="250" fill="#ffffff"/>
  <text x="16" y="22" font-size="14" font-weight="600" fill="#1f2937">投机解码三条演进轴：不是替代，而是汇入统一 Runtime</text>

  <!-- 轴1 Draft Quality -->
  <text x="16" y="52" font-size="11" font-weight="600" fill="#1d4ed8">草稿质量</text>
  <g font-size="10.5">
    <rect x="16" y="58" width="128" height="30" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
    <text x="80" y="77" text-anchor="middle" fill="#1e3a8a">Draft Model</text>
    <text x="152" y="77" font-size="12" fill="#64748b">→</text>
    <rect x="162" y="58" width="120" height="30" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
    <text x="222" y="77" text-anchor="middle" fill="#1e3a8a">EAGLE / EAGLE-2</text>
    <text x="290" y="77" font-size="12" fill="#64748b">→</text>
    <rect x="300" y="58" width="120" height="30" rx="4" fill="#bfdbfe" stroke="#60a5fa"/>
    <text x="360" y="77" text-anchor="middle" fill="#1e3a8a" font-weight="600">EAGLE-3</text>
  </g>

  <!-- 轴2 Draft Parallelism -->
  <text x="16" y="112" font-size="11" font-weight="600" fill="#15803d">草稿并行</text>
  <g font-size="10.5">
    <rect x="16" y="118" width="128" height="30" rx="4" fill="#dcfce7" stroke="#86efac"/>
    <text x="80" y="137" text-anchor="middle" fill="#14532d">Medusa 多头</text>
    <text x="152" y="137" font-size="12" fill="#64748b">→</text>
    <rect x="162" y="118" width="120" height="30" rx="4" fill="#dcfce7" stroke="#86efac"/>
    <text x="222" y="137" text-anchor="middle" fill="#14532d">P-EAGLE 多深度</text>
    <text x="290" y="137" font-size="12" fill="#64748b">→</text>
    <rect x="300" y="118" width="120" height="30" rx="4" fill="#bbf7d0" stroke="#4ade80"/>
    <text x="360" y="137" text-anchor="middle" fill="#14532d" font-weight="600">DFlash / DSpark</text>
  </g>

  <!-- 轴3 Verify Efficiency -->
  <text x="16" y="172" font-size="11" font-weight="600" fill="#b45309">验证效率</text>
  <g font-size="10.5">
    <rect x="16" y="178" width="128" height="30" rx="4" fill="#fef3c7" stroke="#fcd34d"/>
    <text x="80" y="197" text-anchor="middle" fill="#78350f">固定 γ</text>
    <text x="152" y="197" font-size="12" fill="#64748b">→</text>
    <rect x="162" y="178" width="120" height="30" rx="4" fill="#fef3c7" stroke="#fcd34d"/>
    <text x="222" y="197" text-anchor="middle" fill="#78350f">动态树 / 置信度</text>
    <text x="290" y="197" font-size="12" fill="#64748b">→</text>
    <rect x="300" y="178" width="120" height="30" rx="4" fill="#fde68a" stroke="#fbbf24"/>
    <text x="360" y="197" text-anchor="middle" fill="#78350f" font-weight="600">Adaptive Verify</text>
  </g>

  <!-- 汇合箭头 -->
  <g stroke="#94a3b8" stroke-width="1.4" fill="none" stroke-dasharray="4 3">
    <path d="M 425 73 L 500 120"/>
    <path d="M 425 133 L 500 133"/>
    <path d="M 425 193 L 500 146"/>
  </g>
  <rect x="500" y="96" width="164" height="74" rx="6" fill="#eef2ff" stroke="#818cf8" stroke-width="1.5"/>
  <text x="582" y="120" font-size="12" font-weight="600" fill="#3730a3" text-anchor="middle">统一 Inference Runtime</text>
  <text x="582" y="138" font-size="10" fill="#4338ca" text-anchor="middle">vLLM / SGLang</text>
  <text x="582" y="153" font-size="10" fill="#4338ca" text-anchor="middle">训练 → 调度 → 验证预算</text>

  <text x="16" y="236" font-size="11" fill="#6b7280">图：自绘。技术归类见第 01 篇生成轴 / 验证轴；2026 年的变化是三条轴开始在同一运行时里联合调参（见 §7.2、§9、§11）。</text>
</svg>

*图：自绘。基于第 01 篇谱系与 [vLLM](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting) / [SGLang](https://docs.sglang.io/advanced_features/speculative_decoding.html) 公开文档，非论文原图。*

---

## 1. 问题换了：草稿自己也是串行的

第 02 篇把收益写成 $S(\alpha, \gamma, c)$，分母 $1 + \gamma c$ 里藏着一个前提：**草稿成本随草稿长度线性增长**。这个前提在 2023–2025 年的方法里成立，因为无论独立小模型还是 EAGLE 家族，草稿 token 都是一个接一个算出来的：

$$
T_d(\gamma) = \gamma \, t_d
$$

EAGLE-3 的草稿器只有一层解码器，单次前向很便宜，但 $\gamma = 8$ 就要跑八次。vLLM 团队在
[《Parallel All the Way Down》](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting)里把这条约束的两条生产后果写得很直白：草稿器被迫保持极小的参数量，否则攒出来的收益会被草稿自己吃掉；$\gamma$ 因此变成一个要按负载反复调的敏感参数。

![自回归草稿逐个 token 生成，与并行草稿一次前向生成整块候选的时序对照](https://vllm.ai/blog-assets/figures/2026-07-28-speculators-parallel-drafting/ar_vs_parallel.jpg)

*图源：[vLLM Blog, Parallel All the Way Down](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting) Figure 2，许可：来源页未明示（仅外链引用，未下载进仓库），未改动。*

左半是逐个 token 递推，右半是一次前向出整块；两条时序的长度差就是 $T_d$ 从 $\gamma t_d$ 塌成 $t_b$ 的部分。

DFlash 论文给了一张更直接的图：横轴是候选 token 数（4 / 8 / 16），纵轴是草稿延迟，EAGLE-3 的柱子随块长近似线性抬高，而 DFlash 的三个配置几乎不动。

![草稿延迟随候选 token 数变化：EAGLE-3 近似线性增长，DFlash 三个配置近乎平坦](https://arxiv.org/html/2602.06036v2/draft_latency_bar.svg)

*图源：Chen et al., [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) Figure 3（草稿延迟对比），许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，未改动。*

读图可见 EAGLE-3 在 16 个候选时约 25 ms，DFlash 各配置仍在 5 ms 量级；**图上读数是量级，不是本文的实测值**。这张图说明的是成本形状，不是绝对耗时。

于是 2026 年的技术路线可以归成一句话：**把第 02 篇里被当作常数的 $c$，从「每次前向的成本」重新拆成「每块草稿的成本」**。拆开之后，$\gamma$ 不再是收益模型里最敏感的那个旋钮——但它并没有变得无关紧要，因为验证侧还在按候选位置数收费，这一点留到 §7 用数字算。

---

## 2. 一次前向验证多个 token：计算结构，不是分布证明

先把「并行验证」这件事从概率正确性里剥出来。因果 Transformer 可以在一次批量前向里，用 causal attention mask 同时算出草稿序列每个位置的条件分布：

$$
\begin{aligned}
p_1 &= p(d_1 \mid X) \\
p_2 &= p(d_2 \mid X, d_1) \\
p_3 &= p(d_3 \mid X, d_1, d_2) \\
p_4 &= p(d_4 \mid X, d_1, d_2, d_3)
\end{aligned}
$$

**并行的是计算，不是依赖**。causal mask 保留了自回归条件，只是把多个位置塞进同一次前向。若草稿是树而不是链，就要换成 tree attention mask：每个候选节点只能看见自己的祖先。

![经典 draft 先生成五个候选、target 一次并行验证并接受前三个的流程图](https://vizuaraai.github.io/vizuara-ai-daily/speculative-decoding-draft-verify/figures/fig2.png)

*图源：[Vizuara AI Daily, Speculative Decoding: Draft, Verify, Repeat](https://vizuaraai.github.io/vizuara-ai-daily/speculative-decoding-draft-verify/) Fig. 2，许可：来源页未明示（仅外链引用，未下载进仓库），未改动；对应方法见 [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)（CC BY 4.0）。*

图中五个候选在一次 target forward 里被同时检查，第四个被拒后从残差分布重采样——这也是为什么一轮前向能确认多个 token。残差分布与拒绝采样的推导归[采样实现深读](/2026/09/30/pytorch-sampling-internals/) §4。

### 2.1 树验证的 mask 是祖先矩阵

Medusa 的多解码头把线性草稿变成候选树：每个头取 top-2，两层就分出四条路径。此时普通 causal mask 会让兄弟分支互相看见，必须按祖先关系构造掩码：

$$
M_{ij} =
\begin{cases}
0 & j \in \mathrm{Anc}(i) \cup \lbrace i \rbrace \\
-\infty & \text{otherwise}
\end{cases}
$$

![Medusa 候选树与对应的 tree attention mask，勾表示该位置允许 attention](https://cdn.prod.website-files.com/69654e88dce9154b5f12070c/699e0b10f1c356f7f87cb7ee_652fff78b5bcb5e96b3c7dfb_tree_attn-2.png)

*图源：[Together AI, Medusa](https://www.together.ai/blog/medusa) Fig.（来源页树掩码示意图，非论文编号），许可：来源页未明示（仅外链引用，未下载进仓库），未改动；原论文 [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) 为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。*

右侧掩码矩阵里，每个 query 行的勾只落在自己那条路径的祖先列上；同层不同分支之间留空。把八节点候选树（Root → A → B/C/D，B → E/F，D → G）展开成 0/1 矩阵，就是配套实验组 3 逐格核对的那张表：

| Query \ Key | R | A | B | C | D | E | F | G |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| B | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| C | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| D | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| E | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| F | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| G | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |

```text
组 3 结果：24 个非零元，下三角，与对照矩阵逐格一致；
E 不看 F、B 不看 C、G 看 D 与 A 但不看 B。
```

这组断言的价值不在「会画 mask」，而在**它给出了实现层的判据**：GPU kernel 不一定真的物化这个稠密矩阵，可以用 parent index、祖先元数据、block sparse attention 或自定义 kernel 表达，但表达出来的可见关系必须与这张表逐格等价。

### 2.2 KV cache 的三层状态

树验证之后还有一个容易漏掉的工程问题：候选分支的 KV 不能一股脑提交。实现上要区分

| 层 | 内容 | 生命周期 |
|:---|:---|:---|
| Prefix KV | 已确认上下文 | 常驻 |
| Speculative KV | 各候选分支的临时 KV | 本轮验证结束后释放 |
| Committed KV | 接受路径对应的 KV | 转成长驻 |

若接受到第 3 个候选，提交的是 $KV_{\mathrm{prefix}} + KV(d_1, d_2, d_3)$，其余分支回滚。在分页式 KV 管理里，回滚的对象不是「某个张量下标」，而是物理块与逻辑 token 位置之间的映射、引用计数与 slot 分配——这也是[第 04 篇](/2026/10/18/specdec-04-engine-support-matrix/)里引擎支持矩阵比算法表更难过的原因。

---

## 3. 三条并行草稿路线：条件怎么注入，块内依赖怎么补

vLLM 与 Speculators 在 2026-07 同时开源了 P-EAGLE、DFlash、DSpark 三种并行草稿。三者都吃目标模型的隐藏状态，都用一次前向出一整块，但**把目标特征放在哪里、块内 token 之间有没有依赖、训练怎么省**三条轴上的答案完全不同。

![P-EAGLE、DFlash、DSpark 三种并行草稿架构并排对比，含 hidden states、KV-cache、mask token 与置信度分数](https://vllm.ai/blog-assets/figures/2026-07-28-speculators-parallel-drafting/diagram.jpg)

*图源：[vLLM Blog, Parallel All the Way Down](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting) Figure 3，许可：来源页未明示（仅外链引用，未下载进仓库），未改动。*

左列的 P-EAGLE 把隐藏状态当输入特征，中列的 DFlash 把隐藏状态投影进草稿 KV cache，右列的 DSpark 在 DFlash 主干之上加串行修正块与逐位置置信度分数；三列的共同点是底部一次性输出 $t_{n+1} \ldots t_{n+k}$。

### 3.1 P-EAGLE：把 EAGLE-3 的递推改成多深度预测

[P-EAGLE 由 Amazon 提出，Red Hat 在 Speculators v0.6.0 里给出完整开源实现](https://developers.redhat.com/articles/2026/09/03/speeding-llm-inference-p-eagle-vllm-speculators)。`PEagleDraftModel` 直接继承 `Eagle3DraftModel`，改动是把「预测一个 token 再喂回去」换成**预测深度**：同一位置上一次给出未来 $K$ 个 token。

![P-EAGLE 的输入构造：Pos 1 用真实新 token，Pos 2–4 用 mask token，隐藏状态来自目标模型并由 fc 压成 d 维](https://developers.redhat.com/sites/default/files/image1_13.jpg)

*图源：[Red Hat Developer, Speeding up LLM inference with P-EAGLE in vLLM Speculators](https://developers.redhat.com/articles/2026/09/03/speeding-llm-inference-p-eagle-vllm-speculators) Figure 1，许可：来源页未明示（仅外链引用，未下载进仓库），未改动。*

三个实现细节决定它能不能训起来（都出自该文）：

1. **COD sampling**：多深度训练若对每个位置每个深度都算 loss，显存按「序列长度 × 深度」涨。做法是几何衰减——depth 0 保留全部位置，depth 1 保留 0.7，depth 2 保留 0.49，依此类推，并设 0.2 的下限防止最深层饿死。
2. **可学习 mask**：更深的深度没有真实 token，用 `mask_token_id` 占位；隐藏状态侧用一个形状为 $[1, 1, 3d]$ 的可学习 `mask_hidden` 填充，替掉 EAGLE-3 的固定 padding。embedding 表因此解冻。
3. **flex-attention mask**：depth 0 位置之间保持普通因果可见；更深的 rollout 只能看自己那条链与它之前的 depth 0 上下文，不能横看别的 rollout，也不能看未来。

对照[第 03 篇](/2026/10/18/specdec-03-eagle-family/)里 EAGLE 的特征递推图，可以清楚看到「同一份目标特征、逐 token 递推」与「同一份目标特征、多深度一次出」的分界：

![EAGLE 的特征外推与草稿树生成：目标 LLM 顶层特征喂给单层自回归头，再逐层展开候选树](https://images.openai.com/static-rsc-4/N14fsLL0SxRrEOdvGFFQ4aSDY5QmNHdk6az0ZzkhE0HKWJkfGuQpNWSZyopRneL9u9PiqR95T6MFO4ySGf2lfV5Dz7-qMlJDbzLiHBEhEUJ5JqA__mzrpiN1qG_Xg6Qkv462gmOBY-t75dYmr6npKfy3VgP4_yclRoIdyrk_OG7TwPxdoZpoDV_Ihuapdlto?purpose=fullsize)

*图源：[EAGLE 介绍页](https://chlience.com/EAGLE/) Fig.（来源页特征外推示意图，非论文编号），许可：来源页未明示；原论文 [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) 为 arXiv nonexclusive-distrib（只授予 arXiv 分发权），故本图仅外链引用、未下载进仓库，未改动。*

上半是目标模型一次前向拿到特征，下半是草稿模型靠 `FeatExtrapolator` 逐次展开；P-EAGLE 改掉的正是下半那串 `Forward 1 / 2 / 3`。

### 3.2 DFlash：把目标特征投影进草稿的 KV cache

[DFlash（arXiv:2602.06036，ICML 2026）](https://arxiv.org/abs/2602.06036)的做法不是把隐藏状态当普通输入，而是投影成草稿 Transformer 可以直接 attend 的 KV。好处有两条：不必靠拉长输入序列来携带目标特征；草稿的双向注意力能在一个块内同时看到目标上下文与块内 mask 位置。

![DFlash 推理架构：目标模型隐藏特征进 KV Cache，块内用双向注意力在 mask 位置上做 block diffusion，最后经 target lm_head 出多个 token](https://arxiv.org/html/2602.06036v2/dflash_inference_design.svg)

*图源：Chen et al., [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) Figure 2，许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，未改动。*

图中蓝色方块是融合后的目标上下文特征、橙色是目标解码 token、绿色是待去噪的 mask token；一次 block diffusion 前向把整块绿色位置填成候选。这里的 diffusion 指「块内并行去噪」，不是图像生成里的 DDPM 采样链——论文用**单次前向**完成整块草稿，不存在多轮迭代次数 $S$ 要乘进成本的问题。它仍然由目标模型做拒绝采样兜底，所以论文标题里的 lossless 指的是这个验证闭环，不是草稿本身无误差。

把「投影进 KV」这条路径写成形状，接口关系更清楚（以下为说明接口的示意形状，具体按层数与 KV 头数展开）：

$$
H_{\mathrm{target}} \in \mathbb R^{B \times L \times 3d}
\;\xrightarrow{\ \mathrm{Proj}_{\text{feat}}\;}\;
H_{\mathrm{proj}} \in \mathbb R^{B \times L \times d}
\;\xrightarrow{\ \mathrm{Proj}_{KV}\;}\;
K_c, V_c \in \mathbb R^{B \times H \times L \times d_h}
$$

草稿块里的 mask token 通过注意力读这批条件 KV，再在块内做双向去噪——这就是它不靠拉长输入序列也能携带目标上下文的原因。vLLM 的[实现文档](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md)把它叫 anchored-block drafting：锚点位置决定条件 KV 的落点，`block_size` 与 `max_anchors` 是两个直接暴露在训练 CLI 上的旋钮。

注意力可见性由另一张图给出的稀疏模式决定：目标侧特征全部可见，块内 mask 位置按锚点分段可见。

![DFlash 的注意力掩码：左侧目标模型特征全可见，右侧 mask 块沿锚点位置分段可见](https://arxiv.org/html/2602.06036v2/dflash_attn.svg)

*图源：Chen et al., [DFlash](https://arxiv.org/abs/2602.06036) Figure 4（注意力掩码），许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，未改动。*

论文摘要报告的是「跨多个模型与任务超过 6 倍无损加速，相对 EAGLE-3 最高 2.5 倍」——这是**论文设定下的自报数字**，与 §8 要讲的生产级复现不是一回事。

### 3.3 DSpark：并行主干 + 串行修正 + 置信度调度

[DSpark（arXiv:2607.05147）](https://arxiv.org/abs/2607.05147)针对的是并行草稿的两个新瓶颈。第一个是**后缀衰减**：块内位置独立预测，越靠后越不知道前面实际生成了什么，接受率往下掉。第二个是**验证浪费**：一次给出很长的块，目标模型却要把整块都验一遍，高并发下把批容量花在大概率被拒的后缀上。

![DSpark 三段式流程：并行块与串行块产出候选，硬件感知前缀调度器按置信度决定送多少给目标模型验证](https://arxiv.org/html/2607.05147v1/model_arch.svg)

*图源：Cheng et al., [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147) Figure 1（架构），许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，未改动。*

[Speculators 的 DSpark 文档](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md)把两个头写得很具体：

- **Markov head**：给 DFlash 的 logits 加一个低秩偏置 $B = W_1 W_2$，$W_1$ 把前一个块内 token 嵌到 `markov_rank` 维，$W_2$ 投回草稿词表。有 `vanilla` / `gated` / `rnn` 三种变体，对应「只看前一个 token」「由主干隐状态门控」「块内递推状态」三档条件强度。
- **Confidence head**：线性头从主干隐状态预测每个位置的接受概率，用 BCE 项训练；运行时由调度器结合引擎吞吐曲线决定验证前缀长度。

论文在 DeepSeek-V4 在线服务的实时流量下，相对生产基线 MTP-1，在吞吐相当时把单用户生成速度提高 60%–85%。**这是「同等吞吐下的单用户速度」，不是离线加速比**，两个口径不能互换。

### 3.4 三条路线的分界

三者不是「谁替代谁」的时间线，而是并行预测里三种不同的**信息缺失**各补一处。把信息流画在一起，分界就清楚了：

<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC, sans-serif">
  <rect x="0" y="0" width="680" height="340" fill="#ffffff"/>
  <text x="16" y="22" font-size="14" font-weight="600" fill="#1f2937">三种并行草稿的信息流：各自补的是哪一种缺失</text>

  <!-- 列标题 -->
  <text x="120" y="48" font-size="12" font-weight="600" fill="#1d4ed8" text-anchor="middle">P-EAGLE</text>
  <text x="340" y="48" font-size="12" font-weight="600" fill="#15803d" text-anchor="middle">DFlash</text>
  <text x="560" y="48" font-size="12" font-weight="600" fill="#b45309" text-anchor="middle">DSpark</text>

  <!-- 行1: Target Feature Input -->
  <g font-size="10.5" fill="#374151">
    <rect x="40" y="58" width="160" height="34" rx="4" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="120" y="73" text-anchor="middle">目标多层隐藏特征</text><text x="120" y="86" text-anchor="middle">[B, L, 3d]</text>
    <rect x="260" y="58" width="160" height="34" rx="4" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="340" y="73" text-anchor="middle">目标多层隐藏特征</text><text x="340" y="86" text-anchor="middle">[B, L, 3d]</text>
    <rect x="480" y="58" width="160" height="34" rx="4" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="560" y="73" text-anchor="middle">目标多层隐藏特征</text><text x="560" y="86" text-anchor="middle">[B, L, 3d]</text>
  </g>

  <!-- 箭头 -->
  <g stroke="#94a3b8" stroke-width="1.3" fill="none">
    <line x1="120" y1="92" x2="120" y2="110"/><line x1="340" y1="92" x2="340" y2="110"/><line x1="560" y1="92" x2="560" y2="110"/>
  </g>

  <!-- 行2: Feature Conditioning（核心差异） -->
  <g font-size="10.5">
    <rect x="40" y="110" width="160" height="46" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
    <text x="120" y="126" text-anchor="middle" fill="#1e3a8a" font-weight="600">当输入特征</text>
    <text x="120" y="141" text-anchor="middle" fill="#1e3a8a">拼进 token embedding</text>
    <text x="120" y="153" text-anchor="middle" fill="#1e3a8a">多种预测深度</text>

    <rect x="260" y="110" width="160" height="46" rx="4" fill="#dcfce7" stroke="#86efac"/>
    <text x="340" y="126" text-anchor="middle" fill="#14532d" font-weight="600">投影成条件 KV</text>
    <text x="340" y="141" text-anchor="middle" fill="#14532d">K_c, V_c 进草稿 cache</text>
    <text x="340" y="153" text-anchor="middle" fill="#14532d">注意力直接读取</text>

    <rect x="480" y="110" width="160" height="46" rx="4" fill="#fef3c7" stroke="#fcd34d"/>
    <text x="560" y="126" text-anchor="middle" fill="#78350f" font-weight="600">同 DFlash 条件 KV</text>
    <text x="560" y="141" text-anchor="middle" fill="#78350f">+ 块内低秩 Markov 偏置</text>
    <text x="560" y="153" text-anchor="middle" fill="#78350f">B = W1·W2 补前缀依赖</text>
  </g>

  <g stroke="#94a3b8" stroke-width="1.3" fill="none">
    <line x1="120" y1="156" x2="120" y2="174"/><line x1="340" y1="156" x2="340" y2="174"/><line x1="560" y1="156" x2="560" y2="174"/>
  </g>

  <!-- 行3: Intra-block Dependency -->
  <g font-size="10.5">
    <rect x="40" y="174" width="160" height="40" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="120" y="189" text-anchor="middle" fill="#374151">flex-attention mask</text>
    <text x="120" y="203" text-anchor="middle" fill="#64748b">各 rollout 只看自己链</text>

    <rect x="260" y="174" width="160" height="40" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="340" y="189" text-anchor="middle" fill="#374151">块内双向注意力</text>
    <text x="340" y="203" text-anchor="middle" fill="#64748b">无显式自回归</text>

    <rect x="480" y="174" width="160" height="40" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="560" y="189" text-anchor="middle" fill="#374151">Markov / RNN 头</text>
    <text x="560" y="203" text-anchor="middle" fill="#64748b">按位置补块内依赖</text>
  </g>

  <g stroke="#94a3b8" stroke-width="1.3" fill="none">
    <line x1="120" y1="214" x2="120" y2="232"/><line x1="340" y1="214" x2="340" y2="232"/><line x1="560" y1="214" x2="560" y2="232"/>
  </g>

  <!-- 行4: Output -->
  <g font-size="10.5">
    <rect x="40" y="232" width="160" height="30" rx="4" fill="#e0e7ff" stroke="#a5b4fc"/>
    <text x="120" y="251" text-anchor="middle" fill="#3730a3">一次前向出 K 个 token</text>
    <rect x="260" y="232" width="160" height="30" rx="4" fill="#e0e7ff" stroke="#a5b4fc"/>
    <text x="340" y="251" text-anchor="middle" fill="#3730a3">一次前向出整块</text>
    <rect x="480" y="232" width="160" height="30" rx="4" fill="#e0e7ff" stroke="#a5b4fc"/>
    <text x="560" y="251" text-anchor="middle" fill="#3730a3">整块 + 逐位置置信度</text>
  </g>

  <!-- 底部：补的缺失 -->
  <text x="16" y="288" font-size="11.5" font-weight="600" fill="#1f2937">各自补的缺失：</text>
  <text x="120" y="308" font-size="10.5" fill="#1d4ed8" text-anchor="middle">未来位置没有真实 token</text>
  <text x="120" y="322" font-size="10.5" fill="#1d4ed8" text-anchor="middle">→ 用 mask + 多深度组织</text>
  <text x="340" y="308" font-size="10.5" fill="#15803d" text-anchor="middle">目标条件如何进草稿</text>
  <text x="340" y="322" font-size="10.5" fill="#15803d" text-anchor="middle">→ 投影成 KV 而非拉长输入</text>
  <text x="560" y="308" font-size="10.5" fill="#b45309" text-anchor="middle">并行丢失块内依赖</text>
  <text x="560" y="322" font-size="10.5" fill="#b45309" text-anchor="middle">→ 修正头 + 置信度调度</text>
</svg>

*图：自绘。结构对照基于 [vLLM 并行草稿博客](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting) Figure 3、[P-EAGLE 文档](https://developers.redhat.com/articles/2026/09/03/speeding-llm-inference-p-eagle-vllm-speculators)与 [DSpark 论文](https://arxiv.org/abs/2607.05147)，非论文原图。*

| 维度 | P-EAGLE | DFlash | DSpark |
|:---|:---|:---|:---|
| 目标特征放哪 | 当输入，拼进 token embedding | 投影进草稿 KV cache | 同 DFlash |
| 块内依赖 | 靠 flex-attention 让各 rollout 看自己的链 | 块内双向、无显式自回归 | 低秩 Markov / RNN 头补前缀依赖 |
| 训练稀疏化 | draft block sparsification（COD，$0.7^k$，下限 0.2） | sequence length sparsification（锚点位置） | 沿用 DFlash，另加置信度 BCE |
| 验证侧 | 全块验证 | 全块验证 | 置信度调度，按负载截断前缀 |
| 主要解决 | 草稿延迟随 $\gamma$ 线性增长 | 目标条件注入 + 一次出块 | 后缀衰减 + 验证浪费 |

三条路线不是「谁替代谁」的时间线，而是把同一笔账拆到不同位置：P-EAGLE 动输入组织，DFlash 动条件注入与生成范式，DSpark 动验证侧的候选数量。按[第 01 篇](/2026/10/18/specdec-01-overview-and-evolution/)的生成轴 / 验证轴分类，前两者在生成轴，DSpark 同时踩在验证轴上。

---

## 4. 训练侧的账：多 token 预测为什么必须稀疏化

并行草稿共同的训练难题是：长度为 $N$ 的序列、预测窗口 $K$，朴素做法要在每个位置都算未来 $K$ 个 token 的 loss，计算与显存按 $N \times K$ 涨。三家给的答案是同一类——**只在部分位置、部分深度上收 loss**，但切法不同。

P-EAGLE 沿深度方向切（COD sampling 的几何衰减）；DFlash 沿序列方向切，随机采样锚点集合 $\mathcal A$：

$$
\mathcal L_{\mathrm{draft}} = \frac{1}{\lvert \mathcal A \rvert} \sum_{t \in \mathcal A} \mathcal L_{\mathrm{block}}(t)
$$

[DFlash 论文的消融](https://arxiv.org/abs/2602.06036)显示带 loss decay 的训练在同样 epoch 下接受长度更高，说明稀疏化不只是省算力，还改变了优化信号的分布。

![接受长度随训练 epoch 变化，带 loss decay 的曲线整体高于不带 decay 的曲线](https://arxiv.org/html/2602.06036v2/acceptance_length_vs_epoch.svg)

*图源：Chen et al., [DFlash](https://arxiv.org/abs/2602.06036) Figure 5（训练消融），许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，未改动。*

两条曲线都在第 7 个 epoch 附近到顶后略有回落；差异来自训练信号怎么分配，不是模型容量变化。

训练基础设施这一层，[SpecForge（arXiv:2603.18567）](https://arxiv.org/abs/2603.18567)把 EAGLE-3 草稿头的训练做成了独立流水线：目标—草稿解耦、FSDP 与张量并行混合、训练 kernel 优化，以及与生产推理引擎的直接衔接。它提供 online / offline 两种模式，差别不在算法而在**目标模型何时参与**：

![SpecForge 的 online 与 offline 训练流水线：online 在训练时实时生成隐藏状态，offline 先落盘再复用](https://www.lmsys.org/images/blog/spec_forge/offline_online.jpg)

*图源：[LMSYS Org, SpecForge: Accelerating Speculative Decoding Training for SGLang](https://lmsys.org/blog/2025-07-25-spec-forge/) Fig.（来源页训练流水线示意图，非论文编号），许可：来源页未明示（仅外链引用，未下载进仓库），未改动；对应论文 [arXiv:2603.18567](https://arxiv.org/abs/2603.18567)。*

上半 online 把目标模型留在训练回路里现算隐藏状态，下半 offline 先用 SGLang 把隐藏状态落盘、训练时只读草稿模型。按 LMSYS 博客的口径，offline 省下的目标模型占用换来的是磁盘开销（UltraChat + ShareGPT 量级约 12 TB），这笔账和[第 06 篇](/2026/10/18/specdec-06-industry-practice/)里「草稿质量前移到训练目标」的结论是同一件事的另一半：**草稿模型开始有自己的数据管线、checkpoint、评测与发布周期**。

论文自报的两个数字：Qwen3-235B-A22B 上 EAGLE-3 训练最高 9.9 倍加速；SpecBundle 发布的草稿模型在 SGLang 上端到端推理最高 4.48 倍加速。前者是训练吞吐，后者是推理加速，同样是两种不同口径。

---

## 5. 运行时把「验证多少」变成决策

固定 $\gamma$ 的假设在长块上会失效：草稿便宜之后，把整块都送给目标模型反而可能亏。DSpark 的 confidence head 与硬件感知前缀调度器把这件事变成按请求、按负载的动态决策。

![置信度阈值扫描：横轴阈值，柱高是每步平均 token 数，折线是接受率，不同任务上的最优点不同](https://arxiv.org/html/2607.05147v1/confidence_threshold_sweep.svg)

*图源：Cheng et al., [DSpark](https://arxiv.org/abs/2607.05147) Figure 5（置信度阈值扫描），许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，未改动。*

三栏分别对应 Math / Code / Chat：阈值抬上去，送验的 token 变少、接受率变高，每步产出却先升后降。**最优阈值随任务漂移**，所以调度器不能写死一个数。

置信度本身还要校准。[DSpark 论文](https://arxiv.org/abs/2607.05147)指出原始置信头普遍过于自信，并做了时序温度缩放的后验校正；这一点直接决定截断长度选得对不对。

### 5.1 一个容易写错的辨析：截断前缀不破坏无损性

不少二手资料把「confidence filtering」直接归成有损优化，理由是「丢掉了低置信度候选」。这个推断跳了一步。要分清两件事：

| 改动 | 动了什么 | 对输出分布的影响 |
|:---|:---|:---|
| 按置信度**截断候选前缀** | 只改本轮送验的 $\gamma$ | 无影响；拒绝采样与残差分布照旧 |
| 按置信度**放宽接受规则** | 改了「什么算被接受」 | 有损，输出分布不再等于 $p$ |

但这个结论有一个必须写明的边界：**决策发生在采样之前还是之后**。

| 决定 $K$ 的信号 | 是否保持无损 | 条件 |
|:---|:---|:---|
| 上下文与历史接受率，与候选 token 的取值无关 | 可以 | 草稿分布与标准拒绝采样不变 |
| 草稿 top-1 置信度等「提议前」就绪的信号 | 可以 | 决策不改变实际送验的提议分布 |
| 生成候选之后，按候选 token 自身取值筛选 | **不保证** | 提议分布已被选择改变，须重算一致性 |

前两行里，$K$ 在看候选内容之前就定了，送验的仍是同一套 $q$ 下的合法前缀，拒绝采样照旧成立。第三行不能用「丢掉的是尾部连续块」替它背书：例如草稿 $q(A)=0.8,\ q(B)=0.2$，若系统只保留置信度高的 A，实际送验的提议分布已不再是 $q$，继续套 $\min(1, p(x)/q(x))$ 就会破坏分布正确性。所以更准确的说法是——**动态调整候选前缀长度本身不必然破坏无损性，但自适应策略必须保证「决策信号与提议分布一致」**，而「尾部连续」只是必要条件不是充分条件。

配套实验组 4 覆盖了上表前两行的情形，用 20 万轮蒙特卡洛把这个结论钉在数字上：

```text
纯目标采样        : [0.50108 0.29964 0.14970 0.04958]
截断到 γ=2 的投机 : [0.49953 0.30094 0.14964 0.04989]
完整 γ=4 的投机   : [0.49971 0.30045 0.14999 0.04985]
TV(截断, 目标)  : 0.00162
TV(完整, 目标)  : 0.00137
每轮产出期望：γ=2 → 2.7113，γ=4 → 4.0934
```

三种路径的经验分布都收敛到同一个 $p$；截断付出的代价是**每轮产出从 4.0934 降到 2.7113**，即收益变少，不是分布变歪。反过来，若在候选块中间丢掉一个位置却保留后面的位置，因果链就断了——那种实现才是错的，但它是实现 bug，不是算法设计。

### 5.2 后缀衰减与动态树的另一种答案

DSpark 用串行修正头处理后缀衰减，[论文的位置条件接受率图](https://arxiv.org/abs/2607.05147)显示 DFlash 的接受率随草稿位置下降，而 DSpark 的曲线基本被拉平。

![三个方法在 Math / Code / Chat 上按草稿位置的条件接受率：DFlash 下降，DSpark 基本平坦](https://arxiv.org/html/2607.05147v1/position_cond_accept.svg)

*图源：Cheng et al., [DSpark](https://arxiv.org/abs/2607.05147) Figure 2（位置条件接受率），许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，未改动。*

另一个不训练草稿器、在运行时同时改树形与验证规则的答案是 [AdaptiveSpec（arXiv:2609.02897）](https://arxiv.org/abs/2609.02897)——它就是 §6 要展开的《Margins, Not Windows》。先看它的**无损侧**：把树的深度、宽度、节点数都变成变量，输入信号是草稿 top-1 置信度与近几轮接受历史的融合量，$\mathrm{Tree}_t = (\mathrm{Depth}_t, \mathrm{Width}_t, \mathrm{Nodes}_t)$。它与 EAGLE-2 的动态树（[第 03 篇](/2026/10/18/specdec-03-eagle-family/) §5）区别在于：EAGLE-2 在**固定节点预算**里重新分配分支，AdaptiveSpec 允许总节点数本身变化。它的有损侧（margin 规则）留给 §6——两处是同一篇论文在「树形」与「验证规则」两个正交轴上的改动。

---

## 6. 有损验证：margin rule 到底放弃了什么

§5.2 提到的 [AdaptiveSpec](https://arxiv.org/abs/2609.02897) 出自《Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding》（v2，2026-09-28）。它按步同时调两处：树形策略已在上节讲完（无损），这里讲另一半——它放宽的是**接受规则**。当草稿提议 $d_i$ 与目标的采样结果不一致（是一个 mismatch 候选）时，只要它在目标分布里足够接近 top-1，也照样放行：

$$
\frac{p(d_i)}{\max_x p(x)} > \tau
$$

这里的判据是**严格大于**。要区分两类候选：与 $p$ 一致的候选本就通过标准验证，不参与这条规则；只有 mismatch 候选才由 margin 决定额外放行。top-1 的 margin 恒为 $1$——任何 $\tau < 1$ 都会放行它，而 $\tau = 1$ 时 `margin > τ` 对谁都不成立，规则整体退化为严格采样（实验组 5 里 $\tau=1$ 的 $\mathrm{TV}=0$ 正来自这里）。若把判据写成 $\geq$，$\tau=1$ 时 top-1 反而会被放行，与「阈值归一退回无损」的语义冲突，这个符号不能含糊。

标题里的 "Margins, Not Windows" 说的是这一改动——判据从「是否与窗口内的采样结果一致」换成「概率比值是否过阈值」。它明确属于**有损**验证：输出分布不再严格等于 $p$。论文在 SGLang 上报告相对 EAGLE-3 吞吐最高提升 56%，并在 GSM8K / MATH-500 / HumanEval、三个目标模型上恢复到完全无损设置的任务准确率的 93%。

**93% 是任务准确率恢复程度，不是分布保真度，也不是接受率。** 这三个数经常被混成一个。

<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC, sans-serif">
  <rect x="0" y="0" width="680" height="250" fill="#ffffff"/>
  <text x="16" y="24" font-size="14" font-weight="600" fill="#1f2937">margin 规则：τ 决定放行哪些候选</text>
  <line x1="60" y1="200" x2="640" y2="200" stroke="#9ca3af" stroke-width="1"/>
  <line x1="60" y1="60" x2="60" y2="200" stroke="#9ca3af" stroke-width="1"/>
  <g font-size="12" fill="#374151">
    <rect x="90" y="80" width="86" height="120" fill="#93c5fd"/>
    <rect x="230" y="91" width="86" height="109" fill="#fbbf24"/>
    <rect x="370" y="194" width="86" height="6" fill="#d1d5db"/>
    <text x="108" y="72" font-size="12">A  0.51</text>
    <text x="248" y="83" font-size="12">B  0.47</text>
    <text x="388" y="186" font-size="12">C  0.02</text>
    <text x="108" y="220" text-anchor="middle">margin 1.000</text>
    <text x="248" y="220" text-anchor="middle">margin 0.922</text>
    <text x="388" y="220" text-anchor="middle">margin 0.039</text>
  </g>
  <line x1="60" y1="125" x2="640" y2="125" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="470" y="119" font-size="12" fill="#b91c1c">τ = 0.95：只放行 A</text>
  <text x="470" y="145" font-size="12" fill="#b91c1c">τ = 0.90：放行 A 与 B</text>
  <text x="16" y="242" font-size="11" fill="#6b7280">柱高表示目标分布 p，虚线表示放行阈值；放宽 τ 让更多次优 token 绕过严格匹配</text>
</svg>

*图：自绘。数据来自配套实验组 5（[ipynbs commit `8732811`](https://github.com/lrypcy/ipynbs/blob/8732811b95a7007bfa53d3d38e57aa68233f9a85/experiments/speculative-decoding/parallel-drafting/results/stdout.txt)），非论文原图。*

配套实验组 5 用 $p = [0.51, 0.47, 0.02]$ 的小词表扫描阈值，得到一个反直觉结果：

```text
     τ | 被 margin 直接放行 mass | TV(输出, p)
  1.00 |                 0.0000 |      0.00000
  0.98 |                 0.4000 |      0.19600
  0.95 |                 0.4000 |      0.19600
  0.92 |                 0.8500 |      0.05050
  0.90 |                 0.8500 |      0.05050
  0.85 |                 0.8500 |      0.05050
  0.50 |                 0.8500 |      0.05050
```

$\tau = 1$ 时没有任何候选满足严格大于，等于退回严格采样，$\mathrm{TV} = 0$；$\tau = 0.98$ 时只有 top-1 的 A 被放行，把概率质量硬推向 A，$\mathrm{TV}$ 跳到 0.19600；$\tau$ 再降到 0.92，B（在 $p$ 里本就有 0.47 的高概率 token）也进入放行集合，分布反而更接近 $p$，$\mathrm{TV}$ 回落到 0.05050。

**偏差不随阈值单调变化**。原因不是数值噪声，而是「放行集合」与「目标分布的质量分布」之间的相互作用：多放行一个本来就高概率的 token，可能比只放行 top-1 更接近原分布。

更关键的是，**同一个阈值对不同目标分布的伤害完全不同**。实验组 6 固定草稿分布、换三个目标分布扫描同一组 $\tau$：

```text
     τ |  p1=[.51,.47,.02] |  p2=[.80,.15,.05] |  p3=[.40,.35,.25]
  1.00 |           0.00000 |           0.00000 |           0.00000
  0.98 |           0.19600 |           0.08000 |           0.24000
  0.92 |           0.05050 |           0.08000 |           0.24000
  0.85 |           0.05050 |           0.08000 |           0.21250
  0.50 |           0.05050 |           0.08000 |           0.10000
```

$\tau = 0.92$ 时三者分别是 0.0505 / 0.08 / 0.24，极差 0.19：$p_2$ 的次优 token 概率太低、碰不到阈值，曲线平着走；$p_3$ 三个 token 概率接近，放宽就把质量大规模推走。所以 $\tau$ 根本不是可以跨任务复用的超参——调参必须同时看放行集合、目标分布的形状与任务侧指标。

把 §5、§6 合起来，无损与有损的分界线只有一条：**有没有改动接受规则**。改候选长度、改树形、改送验多少都还在无损侧；改「什么算被接受」就越线了。

---

## 7. 收益模型重算：$T_d$ 换形状之后

把第 02 篇的分母 $1 + \gamma c$ 换成两块之和，先得到一个**一阶模型**——它假设各位置接受率恒为 $\alpha$：

$$
S_{\mathrm{par}}(\alpha, K) = \frac{1 - \alpha^{K+1}}{(1 - \alpha)(1 + c_b + \beta K)}
$$

$c_b$ 是整块草稿一次前向的成本，$\beta$ 是验证侧随候选位置数增加的增量。注意 **$c_b$ 与 $K$ 解耦了，$\beta K$ 没有**——这正是 §1 结尾留的那个伏笔。配套实验组 2 在 $\alpha = 0.647$、$c = 0.08$、$c_b = 0.08$、$\beta = 0.02$ 的设定下枚举 $K \in [1, 32]$：

```text
   γ |      串行 S |      并行 S
   1 |    1.4973 |    1.4973
   2 |    1.7213 |    1.8443
   4 |    1.7941 |    2.1652
   8 |    1.5425 |    2.2392
  12 |    1.2832 |    2.1386
  16 |    1.0889 |    2.0222
  24 |    0.8332 |    1.8159
  32 |    0.6745 |    1.6470

γ*：串行 3  →  并行 7
S*：串行 1.7973  →  并行 2.2507
γ 8→16 的收益跌幅：串行 29.41%  →  并行 9.69%
```

三条读数：

1. **最优草稿长度翻倍**（3 → 7），峰值收益从 1.7973 抬到 2.2507。草稿变便宜之后，把块拉长第一次变成划算的。
2. **对 $\gamma$ 的敏感性大幅下降**：从 8 拉到 16，串行模型掉 29.41%，并行模型只掉 9.69%。这解释了 vLLM 博客里「不再需要按负载反复调 $K$」的说法。
3. **跌幅没有归零**（9.69% > 0），因为 $\beta \gamma$ 还在。并行草稿把瓶颈从 Draft 挪到了 Verify，并没有消灭它。

<svg viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC, sans-serif">
  <rect x="0" y="0" width="680" height="270" fill="#ffffff"/>
  <text x="16" y="24" font-size="14" font-weight="600" fill="#1f2937">收益曲面：草稿成本解耦之后，峰值右移且变平</text>
  <line x1="60" y1="220" x2="645" y2="220" stroke="#9ca3af"/>
  <line x1="60" y1="50" x2="60" y2="220" stroke="#9ca3af"/>
  <g font-size="11" fill="#6b7280">
    <text x="40" y="224" text-anchor="end">1.0</text>
    <text x="40" y="164" text-anchor="end">1.7</text>
    <text x="40" y="104" text-anchor="end">2.4</text>
    <text x="60" y="238">γ=1</text><text x="270" y="238">γ=8</text>
    <text x="480" y="238">γ=16</text><text x="615" y="238">γ=32</text>
  </g>
  <polyline points="60,180 100,155 150,148 200,170 270,177 350,190 430,203 500,213 615,225"
            fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <polyline points="60,180 100,155 150,135 200,125 270,122 350,128 430,140 500,152 615,175"
            fill="none" stroke="#16a34a" stroke-width="2.5"/>
  <circle cx="150" cy="148" r="4" fill="#2563eb"/>
  <circle cx="245" cy="122" r="4" fill="#16a34a"/>
  <text x="158" y="142" font-size="11" fill="#1d4ed8">串行 γ*=3, S*=1.7973</text>
  <text x="255" y="116" font-size="11" fill="#15803d">并行 γ*=7, S*=2.2507</text>
  <text x="400" y="60" font-size="11" fill="#2563eb">串行：1 + γc</text>
  <text x="400" y="78" font-size="11" fill="#16a34a">并行：1 + c_b + βγ</text>
  <text x="16" y="260" font-size="11" fill="#6b7280">合成探针组 2；α=0.647, c=0.08, c_b=0.08, β=0.02。曲线按网格值描点，非实测性能</text>
</svg>

*图：自绘。数据来自配套实验组 2（[ipynbs commit `8732811`](https://github.com/lrypcy/ipynbs/blob/8732811b95a7007bfa53d3d38e57aa68233f9a85/experiments/speculative-decoding/parallel-drafting/results/stdout.txt)），非论文原图。*

### 7.1 常数 $\alpha$ 假设在长块上会失真

上面的模型仍把 $\alpha$ 当常数。Red Hat 公布的 P-EAGLE 逐位置接受率（Average 行）给了一个可以反算的一手样本——先把统计口径认对，再谈拟合：

| 位置 $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 前缀接受率 $u_i$ | 0.749 | 0.499 | 0.319 | 0.204 | 0.131 | 0.085 | 0.055 |
| 条件接受率 $s_i = u_i / u_{i-1}$ | 0.749 | 0.6662 | 0.6393 | 0.6395 | 0.6422 | 0.6489 | 0.6471 |

第二行由 $s_i = u_i/u_{i-1}$ 算出，反过来 $\prod_{j \le i} s_j$ 能逐项还原 $u_i$（最大误差 $2.8\times10^{-17}$）。**Red Hat 的 Pos 1–7 是前缀接受率 $u_i$，不是条件接受率**：判据是 $1 + \sum_i u_i = 3.042$，与官方 `Avg Length = 3.044` 只差 0.002（四舍五入）。若误把它当条件率再连乘一次（$E[L] = 1 + \sum_i \prod_{j\le i} u_j$），会得到 2.2698，**低估 25.4%**——这个坑比 $\alpha$ 取常数更致命，因为它连期望的定义都用错了（实验组 6）。

条件接受率 $s_i$ 相邻比值依次是 0.6662 / 0.6393 / 0.6395 / 0.6422 / 0.6489 / 0.6471，均值 0.6472、标准差 0.0092——**几乎是一个常数衰减率**，而不是「每个位置都以同一个 $\alpha$ 被接受」。

```text
组 1：官方 Avg Length 3.044 / 反算 3.042 / 比值均值 0.6472 / 常数 α 高估 18.00%
组 6：u_i 前缀率 vs s_i 条件率；误用连乘低估 25.4%；Πs 还原 u 误差 <1e-10
```

把位置 1 的 0.749 当成全局 $\alpha$ 套进第 02 篇的闭式，会算出 $E[L] = 3.5894$，比 3.042 **高估 18.00%**；改用 $u_i \approx 0.749 \cdot r^{i-1}$ 的双参数模型（等价于一阶几何衰减的条件率），误差降到 0.66%。结论不是「第 02 篇错了」，而是它的独立接受假设是**一阶近似**：块越拉长，位置相关的衰减越不能忽略，$K^*$ 与 $S^*$ 都会算偏。这也是 DSpark 要专门加串行修正头把后缀接受率拉平的原因。

### 7.2 统一收益模型：把分子、分母、决策一起写下来

把前面三处修正合并，得到一个能同时容纳位置衰减、批大小与运行时决策的模型。每轮收益：

$$
\mathrm{Speedup} = \frac{E[L(K)] \cdot T_{\mathrm{AR}}}{T_d(K) + T_v(K;\, B, L) + T_s}
$$

分子 $E[L(K)] = 1 + \sum_{i=1}^{K} u_i$ 用前缀接受率，不再假设各位置同分布；分母里 $T_v$ 显式依赖批大小 $B$ 与上下文长度 $L$——[vLLM Adaptive Verification 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/)把这点讲得很直白：batch=1 时 GPU 受内存带宽限制，多验几个 token 近乎免费；batch=256 时草稿 token 与真实 token 抢同一份算力，被拒绝的每一个都是实打实的浪费。**所以不存在一个跨并发都最优的固定 $K$**。

最优候选长度因此是个argmax，而且应当随请求变化：

$$
K^* = \arg\max_{K} \frac{E[L(K)]}{T_d(K) + T_v(K;\, B, L) + T_s},
\qquad
K_t^* = \arg\max_{k \in \mathcal K} \frac{E[L_t(k)]}{T_d(k) + T_v(k \mid B_t, L_t)}
$$

第二式是动态版：$k$ 在候选档位集合 $\mathcal K$ 里选，条件是**当步**观测到的批大小与上下文。实验组 6 用一个简化的 $\beta(B) = 0.02(1 + 0.35\log_2 B)$ 展示了这个移动——批次越大，多验一个 token 越贵，$K^*$ 单调下降：

```text
    B |    β(B) |  K* |      S*
    1 |  0.0200 |   7 |  2.4934
   16 |  0.0480 |   5 |  2.1985
   64 |  0.0620 |   5 |  2.0878
  256 |  0.0760 |   4 |  2.0022
```

这里的 $\beta(B)$ 是**合成设定**，不是 vLLM 的实测成本曲线；它要说明的是「$K^*$ 随负载移动」这个结构，具体数值不可用于部署。真实的 $\beta$ 要靠启动时 profiling 得到——这正是 §9 要讲的 vLLM Adaptive Verification 做的事：它测出各 shape 的单步成本，再挑期望每秒接受 token 数最大的那个候选数。

至此本篇的数学模型闭环：**一阶模型（§7）看成本形状 → 位置衰减（§7.1）修正分子 → 联合模型（§7.2）把分母与决策也纳入**。$S_{\mathrm{par}}$ 不是被推翻，而是退化成这个统一模型在「$\alpha$ 恒定、$B=1$、$K$ 固定」下的特例。

---

## 8. 为什么论文数字不等于生产吞吐

[MLSys 2026 的《Speculative Decoding: Performance or Illusion?》](https://proceedings.mlsys.org/paper_files/paper/2026/hash/554e056fe2b6d9fd27ffcd3367ae1267-Abstract-Conference.html)是第一篇在生产级 vLLM 上系统评估投机解码的工作，覆盖 n-gram、EAGLE / EAGLE-3、Draft-Model、MTP，横跨多种负载、模型规模与 batch。三条结论值得和本篇前面的模型对读：

1. **目标模型的验证主导执行时间**，不是草稿生成。
2. **接受长度随输出 token 位置、请求与数据集显著变化**——与 §7.1 的逐位置衰减正好对上。
3. **实测性能与理论加速上界之间存在明显差距**，论文据此列出后续研究机会。

第二条尤其关键：第 02 篇把 $\alpha$ 当常数、第 07 篇讨论「批次与并发是失效区的自变量」，在这篇生产级评测里同时被验证。把 §7 的 $\beta \gamma$ 项与这条合起来看，**并行草稿之后的优化主战场确实转到了验证侧**，DSpark 的置信度调度与 AdaptiveSpec 的动态树都是冲着这一项去的。

还有一条更硬的例子：vLLM 那篇并行草稿博客的末尾挂着勘误——

> The plots in Figure 1 were updated on 7/29/26. The numbers in the original plots proved to be inconsistent with the reported benchmarking conditions due to an erroneous environment setup. However, the relative behavior between models was consistent.

也就是说，官方博客的绝对数字在发布次日因环境配置错误被整体换过，而**模型之间的相对行为没变**。这正是[第 07 篇](/2026/10/18/specdec-07-benchmarks-and-tradeoffs/)「数字怎么读」的现实版：跨来源比加速比之前，先对齐模型、任务、batch、引擎版本与指标口径；能跨设置迁移的通常只有相对关系与量级差。

按可信度给本篇引用的数字分层：

| 层级 | 例子 | 能用来做什么 |
|:---|:---|:---|
| 生产级复现 | MLSys 2026 的验证占比与接受长度漂移 | 判断架构瓶颈在哪 |
| 官方博客 / 实现文档 | Red Hat 的逐位置接受率、Speculators 参数 | 反算模型、对齐接口 |
| 论文自报 | DFlash >6× / 2.5×、SpecForge 9.9× / 4.48×、AdaptiveSpec 56% / 93% | 了解设定下的上限，不当部署预期 |
| 合成探针 | 本文所有四位小数 | 验证模型结构，不代表真实性能 |

低并发、单请求延迟优先时，并行草稿更容易兑现收益；高并发、吞吐优先时，验证侧的额外计算与批容量占用会把收益压回去，甚至转负。这条边界在[第 04 篇](/2026/10/18/specdec-04-engine-support-matrix/)的支持矩阵里已经出现过，2026 年只是把「能不能开」变成了「开了之后验证侧是不是新的瓶颈」。

---

## 9. 开源实现与工程成熟度：论文之外还有什么能跑

前面每提到一个方法，都跟着一句「已有开源实现」。但 GitHub 上有仓库，和「能复现、能训练、能进引擎、能上生产」是四件不同的事。这一节按实现成熟度重新看一遍——所有状态均以 2026-11 的官方文档与仓库为准，而不是论文里的「we implement」。

四级判据，从低到高：

1. **Reference**：作者给出复现核心算法的参考实现；
2. **Training**：有数据处理、隐藏状态抽取、训练脚本、评估与 checkpoint 导出；
3. **Inference**：真正接进 vLLM / SGLang 等引擎，有可部署路径；
4. **Production**：有模型权重、跨配置 benchmark、版本发布与持续维护。

| 方法 | 主要实现 | 训练 | 推理集成 | 成熟度判断 |
|:---|:---|:---|:---|:---|
| EAGLE-3 | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | 完整 | 多引擎 | 四级齐全，成熟基线 |
| DFlash | [z-lab/dflash](https://github.com/z-lab/dflash) + [Speculators](https://github.com/vllm-project/speculators) | 完整 | vLLM / SGLang | 较完整，多后端可跑 |
| DSpark | [Speculators](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md) | 完整 | vLLM / SGLang 生态 | 较完整，覆盖三个核心组件 |
| P-EAGLE | [Speculators](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/peagle.md) | **已弃用**（0.11 移除） | vLLM 仍在 | 版本状态需注意 |
| AdaptiveSpec | [论文](https://arxiv.org/abs/2609.02897) | 无需训练 | 论文报告 SGLang 实现 | 完整公开源码待核实 |
| vLLM Adaptive Verification | [vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/) | 无需额外训练 | vLLM | 已有 Runtime 实现 |
| SGLang Adaptive Steps | [SGLang](https://docs.sglang.io/advanced_features/speculative_decoding.html) | 无需训练 | SGLang | 有 Runtime 实现 |

这张表里最该留意的是 **P-EAGLE 的训练状态**：[Speculators 官方 P-EAGLE 文档](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/peagle.md)开头就挂着警告——训练将在 0.11 移除，选择它训练会触发 `FutureWarning`。但「训练器弃用」不等于「vLLM 的并行草稿能力被弃用」，两者是不同层面的事；[算法决策指南](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/decision_guide.md)现在把 EAGLE-3 标为 Mature，DFlash / DSpark / MTP 标为 "Newer, growing fast"。选型时别把 §3 里的算法比分直接当成工具链成熟度比。

### 9.1 vLLM Adaptive Verification：把 §7.2 的 argmax 做进 Runtime

§7.2 推出 $K^*$ 应该随负载移动，vLLM 已经把这件事做成了独立功能（默认关闭，`--speculative-config` 里 `enable_adaptive_verification: true` 开启）。它对每个 (request, position) 草稿槽算生存概率

$$
s_{i,j} = \prod_{k=1}^{j} \hat a_{i,k}
$$

再把所有请求的候选槽放到一起排序，从高到低接纳，直到花完一个**全局预算**。于是会出现这种调度：某个高置信请求的第 5 个位置，优先级高于另一个低置信请求的第 1 个位置——后者可能只验一两个 token 就被裁掉。预算本身来自启动时的 profiling：vLLM 测出各 shape 的单步成本，再挑「期望每秒接受 token 数」最大的候选数，正是 §7.2 那个 argmax 的工程版。

接受概率 $\hat a_{i,k}$ 有三个来源：[官方文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/)列得很清楚——DSpark 的 confidence head 直接用；其他 model-based drafter 用在线 logistic estimator，在服务中按 target 的验证结果再校准；suffix decoding 没有 draft logits，用同来源、同匹配长度、同位置的历史接受率估计。

它不是没有代价的。文档列出的限制值得原样记下来：**必须启用完整 CUDA Graph**（`--enforce-eager` 会被直接拒绝，因为 step 成本从捕获的图里 profile）；**不支持 LoRA 与 pipeline parallelism**；attention backend 必须能容忍 device-decided query lengths；没有 confidence head 时 `use_local_argmax_reduction` 不可用。这些是部署前要先核对的硬约束，不是调参能绕过的。

### 9.2 AdaptiveSpec 的代码状态：论文实现 ≠ 上游可复现

[《Margins, Not Windows》](https://arxiv.org/abs/2609.02897)明确报告了在 SGLang 上的实现与实验（margin rule + 动态树两个组件），但核查 SGLang 主仓库、其 speculative runtime、vLLM Speculators 与 Adaptive Verification 后，**没有找到能独立确认完整复现该论文算法的公开上游入口**——论文页只有 arXiv 模板自带的 CatalyzeX 通用链接。这不是说论文没做实现，而是「论文里 report 了」和「读者能拉下来复现」之间还差一层。

尤其要把三个同名/近名概念分开，否则很容易张冠李戴：

- **AdaptiveSpec 论文**（arXiv:2609.02897）：margin 有损验证 + 动态树，本文 §5.2 / §6 讲的就是它；
- **SGLang Adaptive Speculative Decoding**：基于历史接受长度维护 EMA、按 batch 分档切换 draft steps、带 hysteresis 防抖，**是运行时 draft-length 调度**，官方 quick guidance 明确要求挂在 EAGLE 上且 `--speculative-eagle-topk 1`；
- **vLLM Adaptive Verification**：§9.1 的生存概率 + 全局预算分配。

三者都在动「验多少 / 接受什么」，但不是一套算法。把 SGLang 的 EMA 控制器写成 AdaptiveSpec 论文的开源实现，是这组材料里最容易犯的错。

### 9.3 量化与并行草稿：一个正在成形的交叉点

审稿与官方仓库都指向同一个趋势：并行草稿正在和量化一起调。[Speculators 决策指南](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/decision_guide.md)写明 EAGLE-3 / P-EAGLE / DFlash / DSpark 都可搭配量化 verifier，而 DFlash2 因候选选择器跑在完整词表上，目前需要未量化的 lm_head；[DFlash 官方 README](https://github.com/z-lab/dflash) 更给出一条实测约束——MLX 后端量化 draft / target 时要把 `block_size` 压到 5 以内，因为量化 matmul kernel 在更宽的 verify 面上效率会掉。

这里藏着一个容易混淆的无损口径：**相对于量化 target 的无损，不等于相对于原始 BF16 target 的无损**。Exact speculative sampling 保证的是输出分布与「它验证的那个 target」一致；如果 target 本身已被量化，它保证的是量化后分布，不会自动把高精度模型的分布找回来。把这条写清楚，比多列几个加速数字更有用。

---

## 10. 工程落地：从一份 speculator checkpoint 到能开起来

[Speculators v0.6.0](https://github.com/vllm-project/speculators)把并行草稿做成了四步流水线：准备数据 → 用 vLLM 抽取目标模型隐藏状态 → 训练草稿器 → 部署。以 Qwen3-8B 为例，Red Hat 文档给的取层是第 2 / 18 / 33 层（41 层模型的早、中、后段），拼成 $3 \times 4096$ 的输入；训练参数里 `--num-depths 4`、`--down-sample-ratio 0.7`、`--down-sample-ratio-min 0.2` 正对应 §4 的 COD 几何衰减，`--mask-token-id` 对应 §3.1 的占位符。

vLLM 侧只需在启动时给出方法与 speculator checkpoint：

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --speculative-config '{
    "model": "RedHatAI/Qwen3-30B-A3B-speculator.dflash",
    "num_speculative_tokens": 7,
    "method": "dflash"
  }'
```

（配置取自 [vLLM 并行草稿博客](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting)；实际使用要核对引擎版本、checkpoint 与目标模型的配对关系。）

接口对齐上有三件事最容易 silently 出错，且都不属于算法问题：

- **词表与 tokenizer**：草稿头输出的是目标模型词表上的 logits，词表不一致会让接受率塌到接近零；
- **隐藏层选择与拼接顺序**：`$3 \times 4096$` 的维度与层顺序必须和训练时一致，换模型要重新抽；
- **mask token id 的语义**：占位符不能占用有特殊含义的 token，embedding 解冻后这一项也会进训练。

![EAGLE-1 / EAGLE-2 / EAGLE-3 的草稿结构演进：顶层特征 → 动态投机树 → 多层融合特征](https://objectstore.e2enetworks.net/e2eblog/jl/2025/blogs/speculative-decoding-with-vllm/eagle-1-2-3-speculative-decoding-comparison.webp)

*图源：[E2E Networks, Speculative Decoding with vLLM](https://objectstore.e2enetworks.net/e2eblog/jl/2025/blogs/speculative-decoding-with-vllm/eagle-1-2-3-speculative-decoding-comparison.webp) Fig.（来源页 EAGLE 三代对比示意图，非论文编号），许可：来源页未明示（仅外链引用，未下载进仓库），未改动；对应论文 [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) / [2406.16858](https://arxiv.org/abs/2406.16858) / [2503.01840](https://arxiv.org/abs/2503.01840)。*

三列的变化是：EAGLE-1 吃顶层特征出线性预测序列，EAGLE-2 把序列换成按置信度动态分叉的投机树，EAGLE-3 把特征源换成高 / 中 / 低多层融合。并行草稿继承的是「目标特征作为条件」这一条，改掉的是「草稿器逐次前向」那一条——所以[第 03 篇](/2026/10/18/specdec-03-eagle-family/)的谱系并没有被推翻，只是最右边多了一列。

---

## 11. 边界、选型与还没解决的问题

把本篇的结论压成一张选型判据表：

| 负载 / 约束 | 更值得先看的方向 | 理由 |
|:---|:---|:---|
| 低并发、单请求延迟优先 | 并行草稿（P-EAGLE / DFlash / DSpark） | 验证侧批容量充裕，$T_d$ 的线性项省得下来 |
| 高并发、吞吐优先 | 置信度调度 + 动态开关 | 验证侧已成瓶颈，需要按负载截前缀 |
| 长推理轨迹 / 代码 | 带串行修正的并行草稿 | 位置越靠后接受率越衰减，块内依赖要补 |
| 严格复现目标分布 | 拒绝采样路径上的任何变体 | 改候选长度与树形都无损 |
| 允许质量换吞吐 | margin 一类有损验证 | 需同时盯任务指标与放行集合，不能只调 $\tau$ |
| 没有配对 checkpoint | n-gram / suffix 等无模型方案 | 并行草稿都要与目标模型配对的草稿权重 |

五个仍然开放的问题（**以下为作者判断**）：

1. **验证能否不完整前向**。DSpark 与 AdaptiveSpec 都只在「少验几个 token」上做文章，目标模型仍要跑整层前向；若能让验证成本随接受置信度连续变化，收益模型的 $\beta \gamma$ 项才真正被拆掉。
2. **位置衰减有没有可迁移的理论形状**。§7.1 的一组样本呈现近乎完美的几何衰减（$r = 0.6472$，标准差 0.0092），但它是单一草稿器、单一评测表的样本；是否随模型、任务、块长保持稳定，还需要更多公开逐位置数据。
3. **置信度估计准不准、怎么校准**。DSpark 的 confidence head 预测的是「该位置被接受的概率」，不是目标 token 概率，两者不是一回事；原始置信头普遍过度自信。要评估调度是否可靠，需要 ECE / Brier Score 这类校准指标与温度缩放，并按位置、按任务分别标定——§9.1 的在线 estimator 也依赖它的准确性。
4. **并行草稿与批量调度的联合优化**。草稿一次出整块之后，不同请求的验证宽度差异会变大，continuous batching 里的组批与 KV 分配都要重新算；这已经不是草稿器自己的问题。
5. **有损验证的可控边界**。§6 的 TV 非单调与多分布差异说明「阈值越低越有损」并不成立；要给出可交付的部署准则，需要把放行集合、目标分布形状、任务指标三者一起标定。

与[第 08 篇](/2026/10/18/specdec-08-future-directions/)的「动态策略与部署形态正在洗牌」对照：2026 年下半年这批工作已经把「动态」落成了具体组件（confidence head、prefix scheduler、margin rule、动态树策略），把「部署形态」落成了具体的开源实现（Speculators、vLLM / SGLang 的 `dflash` / `dspark` 方法）。洗牌没有结束，但牌面比一年前清楚：**收益模型的分子由草稿质量决定，分母里剩下的大头是验证**。

---

**上一篇**：[投机解码（08）：未来方向与开放问题——收益天花板在哪](/2026/10/18/specdec-08-future-directions/)

**下一篇**：投机解码系列在此续写；并行草稿的验证侧瓶颈接[推理引擎主线](/2026/10/18/serving-01-architecture-divergence/)

---

## 参考

**配套实验**（纯 numpy / CPU，`seed = 20261117`，29 条断言）：

- [并行草稿与自适应验证：六个合成探针（带输出 notebook）](https://github.com/lrypcy/ipynbs/blob/8732811b95a7007bfa53d3d38e57aa68233f9a85/experiments/speculative-decoding/parallel-drafting/parallel_drafting_lab.ipynb)
  —— `ipynbs` 仓库 `experiments/speculative-decoding/parallel-drafting/`，commit `8732811`；
  [固化输出](https://github.com/lrypcy/ipynbs/blob/8732811b95a7007bfa53d3d38e57aa68233f9a85/experiments/speculative-decoding/parallel-drafting/results/stdout.txt)

**2026 年并行草稿与自适应验证**：

1. Chen, Liang, Liu，*DFlash: Block Diffusion for Flash Speculative Decoding*（ICML 2026）— [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)
2. Cheng et al.，*DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* — [arXiv:2607.05147](https://arxiv.org/abs/2607.05147)
3. Urbán, Kwon, Venieris, Mascolo，*Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding*（AdaptiveSpec，v2 2026-09-28）— [arXiv:2609.02897](https://arxiv.org/abs/2609.02897)
4. Li et al.，*SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding* — [arXiv:2603.18567](https://arxiv.org/abs/2603.18567) / [PMLR v306](https://proceedings.mlr.press/v306/li26hf.html)
5. *Speculative Decoding: Performance or Illusion?*（MLSys 2026）— [Proceedings of MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/554e056fe2b6d9fd27ffcd3367ae1267-Abstract-Conference.html)

**官方文档与实现**：

6. vLLM Blog，*Parallel All the Way Down: Beyond Single-Token Generation with Speculative Decoding*（2026-07-28，含 7/29 勘误）— <https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting>
7. Red Hat Developer，*Speeding up LLM inference with P-EAGLE in vLLM Speculators*（2026-09-03）— <https://developers.redhat.com/articles/2026/09/03/speeding-llm-inference-p-eagle-vllm-speculators>
8. [vllm-project/speculators](https://github.com/vllm-project/speculators) — [DFlash 算法文档](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md) 同目录下的 DSpark 用户指南
9. [vLLM 投机解码文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)；[LMSYS, SpecForge: Accelerating Speculative Decoding Training for SGLang](https://lmsys.org/blog/2025-07-25-spec-forge/)
10. [EAGLE 介绍页](https://chlience.com/EAGLE/)；[Together AI, Medusa](https://www.together.ai/blog/medusa)；[Vizuara AI Daily, Draft then Verify](https://vizuaraai.github.io/vizuara-ai-daily/speculative-decoding-draft-verify/)

**系列内部引用**：

11. [第 01 篇 综述与演化](/2026/10/18/specdec-01-overview-and-evolution/) · [第 02 篇 收益模型](/2026/10/18/specdec-02-acceptance-and-speedup-model/) · [第 03 篇 EAGLE 家族](/2026/10/18/specdec-03-eagle-family/) · [第 04 篇 引擎支持矩阵](/2026/10/18/specdec-04-engine-support-matrix/) · [第 06 篇 业界实践](/2026/10/18/specdec-06-industry-practice/) · [第 07 篇 评测数字怎么读](/2026/10/18/specdec-07-benchmarks-and-tradeoffs/) · [第 08 篇 未来方向](/2026/10/18/specdec-08-future-directions/)
12. [PyTorch 采样实现深读](/2026/09/30/pytorch-sampling-internals/) §4（拒绝采样与残差分布的无偏性）
13. [推理引擎（01）：架构分歧的根](/2026/10/18/serving-01-architecture-divergence/)
