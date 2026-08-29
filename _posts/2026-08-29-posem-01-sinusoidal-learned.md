---
title: "大模型位置编码（01）：正余弦位置编码与可学习位置编码"
date: 2026-08-29 08:20:00 +0800
categories:
  - 位置编码
tags: [llm, positional-encoding, sinusoidal, learned-pe, transformer]
layout: post
mathjax: true
---

> **系列导航** ｜ 第 01 篇 / 共 08 篇
>
> [← 00 总览](/2026/08/29/posem-00-overview/) ｜ [02 相对位置编码 →](/2026/08/29/posem-02-relative-t5/)

> **TL;DR**
>
> * **正余弦位置编码（Sinusoidal PE）** 是 2017 年原始 Transformer 的方案：$\boldsymbol{p}_m$ 的第 $2i$ 维是 $\sin(m\theta_i)$、第 $2i+1$ 维是 $\cos(m\theta_i)$，频率 $\theta_i = 10000^{-2i/d}$ 按几何级数从高频排到低频。它有三个被精心设计的性质：**每个维度有界（±1）**、**不同位置的编码彼此线性无关**、以及最关键的——$\boldsymbol{p}_m \cdot \boldsymbol{p}_n$ 只依赖相对距离 $m-n$，且随距离增大整体趋于衰减。
> * **可学习位置编码（Learned PE）** 是 BERT / GPT-2 的方案：把 $\boldsymbol{p}_m$ 当普通参数训练。它放弃了手工结构，换来的是简单和略好的同长度性能，代价是**严格锁死训练长度**——这是长上下文时代它被彻底淘汰的直接原因。
> * **历史裁决**：在纯粹的"同长度"竞赛里两者五五开（原始论文自己也说结果差不多）；但 Sinusoidal 的"内积只依赖 $m-n$"这一性质，在六年后被 RoPE 发扬成"用绝对形式实现相对位置"的完整框架。起点即伏笔。

---

## 1. 设计问题：给每个下标配一个向量

绝对位置编码的做法最直白：为位置 $m$ 分配一个 $d$ 维向量 $\boldsymbol{p}_m$，加到 token embedding 上：

$$
\tilde{\boldsymbol{x}}_m = \boldsymbol{x}_m + \boldsymbol{p}_m
$$

注意这是一个**加性、只在输入层做一次**的方案。经过第一层之后，位置信息与内容信息在残差流里混合，此后再无法干净地分离（[00 篇](/2026/08/29/posem-00-overview/)的"解耦标尺"在这里已经是低分）。

现在的设计问题是：$\{\boldsymbol{p}_m\}_{m=0}^{\infty}$ 这个（理想上无限的）向量族该怎么选？两个候选：手工构造一个解析函数（Sinusoidal），或为有限个位置直接学一张表（Learned）。

## 2. Sinusoidal 位置编码

### 2.1 定义

对位置 $m \in \{0, 1, 2, \dots\}$、维度 $i \in \{0, 1, \dots, d/2 - 1\}$：

$$
\begin{aligned}
p_{m, 2i} &= \sin(m \cdot \theta_i), \\
p_{m, 2i+1} &= \cos(m \cdot \theta_i),
\end{aligned}
\qquad
\theta_i = 10000^{-2i/d}
$$

直觉：把 $d$ 维向量切成 $d/2$ 个二维平面，每个平面里位置编码是一个以角频率 $\theta_i$ 旋转的单位圆上的点。频率从 $\theta_0 = 1$（最高频，位置每 +1 转一大格）按几何级数递减到 $\theta_{d/2-1} \approx 1/10000$（最低频，上万个位置才转完一圈）。整个向量像一块**多刻度的表盘**：高频维是秒针，低频维是时针，任意位置组合出的刻度图案几乎唯一。

### 2.2 性质一：任意相对位移都是线性变换

对固定的位移 $\delta$，由三角恒等式：

$$
\begin{pmatrix} \sin((m+\delta)\theta_i) \\ \cos((m+\delta)\theta_i) \end{pmatrix}
=
\begin{pmatrix} \cos(\delta\theta_i) & \sin(\delta\theta_i) \\ -\sin(\delta\theta_i) & \cos(\delta\theta_i) \end{pmatrix}
\begin{pmatrix} \sin(m\theta_i) \\ \cos(m\theta_i) \end{pmatrix}
$$

即 $\boldsymbol{p}_{m+\delta} = \boldsymbol{R}(\delta)\, \boldsymbol{p}_m$，$\boldsymbol{R}(\delta)$ 是块对角的旋转矩阵——**位移对应旋转**。记住这个画面，它就是 RoPE 的种子（[03 篇](/2026/08/29/posem-03-rope/)会把这个"旋转"从位置编码本身搬到 Q/K 向量上）。

### 2.3 性质二：内积只依赖相对距离

同一个二维平面内：

$$
\sin(m\theta)\sin(n\theta) + \cos(m\theta)\cos(n\theta) = \cos\big((m-n)\theta\big)
$$

于是：

$$
\boldsymbol{p}_m \cdot \boldsymbol{p}_n = \sum_{i=0}^{d/2-1} \cos\big((m-n)\,\theta_i\big)
$$

**$m$ 和 $n$ 只以 $m-n$ 出现。** 这是一个绝对位置编码偷偷携带相对位置信息的第一个证据。此外，$\sum_i \cos(\delta\theta_i)$ 在 $\delta=0$ 时取最大值 $d/2$；$|\delta|$ 增大时，不同频率的余弦相位逐渐错开、正负相消，内积整体呈衰减震荡——"距离越远，位置相似度越低"，正是语言归纳偏置想要的形状。

需要说清楚的一点：这个性质**不会**让带 Sinusoidal PE 的 Transformer 自动获得相对注意力——因为 PE 是加在输入 embedding 上的，$\tilde{\boldsymbol{x}}_m \cdot \tilde{\boldsymbol{x}}_n = \boldsymbol{x}_m\cdot\boldsymbol{x}_n + \boldsymbol{x}_m\cdot\boldsymbol{p}_n + \boldsymbol{p}_m\cdot\boldsymbol{x}_n + \boldsymbol{p}_m\cdot\boldsymbol{p}_n$，四项里只有最后一项是纯相对的；且这只是输入层，注意力权重经过 W_Q/W_K 投影与多层传播后早已面目全非。它只是"有这个潜力"。

### 2.4 为什么是几何级数、为什么是 10000

频率比是 $r = 10000^{2/d}$。选几何级数保证相邻两维的周期比恒定，位置刻度在"分辨率轴"上均匀铺开（类似浮点数的尾数）。底数 10000 意味着：最高频维周期 $2\pi$，最低频维周期约 $2\pi \cdot 10000$——编码器训练长度通常远小于此，最低频维"几乎线性单调"，可用作粗粒度的绝对坐标。这个设计哲学八年后在 RoPE 上被原样继承，而"底数 10000 该多大"在长上下文时代重新成为核心超参（[04 篇](/2026/08/29/posem-04-extrapolation-pi-ntk/)的 NTK-aware 与 [05 篇](/2026/08/29/posem-05-yarn-longrope/)的 LongRoPE 都在动它）。

## 3. 可学习位置编码

### 3.1 定义与实现

BERT / GPT-2 直接设一张 $\boldsymbol{P} \in \mathbb{R}^{L_{\max} \times d}$ 的可训练表：

```python
import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)   # P: (L_max, d)

    def forward(self, x):                                # x: (B, L, d)
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device)           # m = 0..L-1
        return x + self.pos_emb(pos).unsqueeze(0)        # broadcast: (B, L, d)
```

| 数学符号 | 代码变量 | Shape | 说明 |
|---:|---|:---:|---|
| $m$ | `pos` | (L,) | 位置下标 |
| $\boldsymbol{P}$ | `pos_emb.weight` | (L_max, d) | 可学习位置表 |
| $\boldsymbol{p}_m$ | `pos_emb(pos)` | (L, d) | 取出第 m 行 |
| $\tilde{\boldsymbol{x}}_m$ | 返回值 | (B, L, d) | 加上 PE 后的输入 |

### 3.2 与 Sinusoidal 的对照实验

原始 Transformer 论文的表述是：两者在机器翻译任务上"几乎相同"（learned 略好）。后续大量复现的共识是：**同长度训练与评估时，learned PE 通常持平或小胜**（毕竟容量更大、可针对任务调整）；差异远小于数据与训练的影响。

但劣势是结构性的：

| 维度 | Sinusoidal | Learned |
|---|---|---|
| 参数量 | 0 | $L_{\max} \times d$（GPT-2 1024×768 ≈ 0.8M） |
| 长度上限 | 理论无限 | 硬上限 $L_{\max}$，超出即崩溃 |
| 外推能力 | 有理论基础，实测平平 | **零**：位置 $L_{\max}+1$ 无 embedding |
| 内积性质 | $\boldsymbol{p}_m\cdot\boldsymbol{p}_n = f(m-n)$ | 无任何结构保证 |

### 3.3 外推为什么是零，以及"外推错觉"的历史

一个常被引用的细节：GPT-2/原始 GPT 的位置 embedding 只在训练时用过前 $L_{\max}$ 个位置，推理时若强行外插（例如把位置 1025 映射到随机初始化的行），输出立刻不可用。BERT 系任务长度固定 512，问题被任务本身掩盖了。

有趣的是，Sinusoidal 在当年的机器翻译评测里外推同样**不好**（BLEU 在超长句上骤降）——理论上无约束 ≠ 实际可外推，因为训练时模型从未见过低频维转过大角度时的组合分布。这个"理论可外推、实测不可外推"的矛盾，正是 [04 篇](/2026/08/29/posem-04-extrapolation-pi-ntk/)外推性分析的起点：**外推失败不是编码不存在，而是训练分布外的坐标值让注意力分布崩坏**。

## 4. 实现细节与常见坑

1. **维度奇偶配对方式**：原始论文是 $(\sin, \cos)$ 相邻成对；有的实现（如某些 GPT 复现）把前一半维全放 sin、后一半全放 cos。两者数学等价（只是排列），但**加载预训练权重时必须匹配实现**，混用会导致性能静默劣化。
2. **Sinusoidal 要注册为 buffer 而非 parameter**，否则会被优化器更新、被保存进 checkpoint，白白破坏结构。
3. **加性 PE 与 LayerNorm 的交互**：PE 加在 embedding 上、之后紧接 LayerNorm 的架构里，PE 的能量会被归一化部分吸收；GPT 系把 PE 加在 token embedding 后再 dropout，两家的顺序不同，迁移实现时容易踩坑。

```python
import math
import torch

def sinusoidal_pe(max_len: int, d: int, base: float = 10000.0):
    # theta_i = base^(-2i/d), 形状 (d/2,)
    theta = torch.exp(torch.arange(0, d, 2).float() / d * (-math.log(base)))
    m = torch.arange(max_len).float().unsqueeze(1)          # (L, 1)
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(m * theta)                      # p_{m,2i}  = sin(m θ_i)
    pe[:, 1::2] = torch.cos(m * theta)                      # p_{m,2i+1}= cos(m θ_i)
    return pe                                               # (L, d), 无参数

# 验证性质二：p_m · p_n 只依赖 m-n
pe = sinusoidal_pe(2048, 128)
dot = pe @ pe.t()
m, n = 100, 50
print(torch.allclose(dot[m, n], dot[m + 500, n + 500], atol=1e-4))  # True
```

## 5. 总结与下一篇

Sinusoidal 的遗产不是它本身的性能，而是它埋下的三个观念：**多尺度频率谱**、**位移即旋转**、**内积只依赖相对距离**。Learned PE 的遗产则是一个反面教训：**与位置绑定的自由参数是长上下文的死穴**。两条线索在 2021 年交汇——RoPE 把"旋转"从 $\boldsymbol{p}_m$ 本身搬到 Q/K 上，第一次让"绝对形式、相对内容"不打折扣地实现。

但在这之前，研究界先走了一段"直接改注意力公式"的弯路——T5 bias、Transformer-XL、XLNet 的相对位置编码家族，正是 [02 篇](/2026/08/29/posem-02-relative-t5/)的主角。

**Lab 练习**：
1. 用上面的代码画出 $\boldsymbol{p}_m\cdot\boldsymbol{p}_n$ 随 $|m-n|$ 的曲线（$d=128$，base=10000），观察衰减震荡；再换 base=100、base=1000000 对比衰减速度——直观感受"底数决定频率谱覆盖"；
2. 在一个小模型上把 Sinusoidal 换成 learned PE，分别在第 100/1000/10000 步可视化前几层 attention map 的"位置偏置"（对角线集中度），观察两种 PE 学出的注意力局部性差异。

## 参考文献

1. Vaswani et al., *Attention Is All You Need*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
3. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019. [OpenAI](https://openai.com/index/better-language-models/)
4. 苏剑林，《让研究人员绞尽脑汁的Transformer位置编码》，科学空间. [spaces.ac.cn/archives/8130](https://spaces.ac.cn/archives/8130)
