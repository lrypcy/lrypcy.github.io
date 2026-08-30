---
title: "大模型位置编码（02）：相对位置编码——从 Transformer-XL 到 T5 bias"
date: 2026-08-29 08:40:00 +0800
categories:
  - LLM算法
tags: [llm, 位置编码, relative-pe, transformer-xl, t5]
layout: post
mathjax: true
---

> **系列导航** ｜ 第 02 篇 / 共 08 篇
>
> [← 01 正余弦与可学习](/2026/08/29/posem-01-sinusoidal-learned/) ｜ [03 RoPE 详解 →](/2026/08/29/posem-03-rope/)

> **TL;DR**
>
> * **动机**：绝对 PE 只在输入层加一次，位置信息经过多层传播早已与内容不可分。相对位置编码学派的主张更激进——**位置信息应该直接出现在注意力打分公式里**，且只以 $$m-n$$ 的形式：$$\text{score}(m, n) = f(\boldsymbol{q}_m, \boldsymbol{k}_n, m-n)$$。
> * **三代方案**：Transformer-XL（2019）把打分公式逐项重写为相对形式；**T5（2020）**用一句"实验发现其余项都可以删"的惊人简化，把整个方案压缩成一张可学习的标量偏置表 $$b_{\text{bucket}(m-n)}$$——简单到粗暴，却成了那一代最常用的方案；XLNet/DeBERTa 是两支中间变体。
> * **为什么它们最终都让位于 RoPE**：① 改写了注意力的矩阵形式，与 FlashAttention/张量并行/KV cache 等工程优化摩擦大；② T5 的 bucket 截断使相对距离上限=外推上限；③ 加性 bias 无法随 $$\vertm-n\vert$$ 连续衰减成可调的注意力局部性。RoPE 用"不改公式、只旋转 Q/K"一次解决了全部三点。

---

## 1. 从"该注入什么"到"该在哪注入"

[01 篇](/2026/08/29/posem-01-sinusoidal-learned/)结尾指出：加性绝对 PE 的位置信息在第一层之后就和内容混在一起了。相对位置编码学派把设计目标直接写进注意力打分。标准的注意力打分（无 PE 时）：

$$
s_{mn} = \frac{\boldsymbol{q}_m \cdot \boldsymbol{k}_n}{\sqrt{d}}
$$

希望的新形式：

$$
s_{mn} = g(\boldsymbol{q}_m, \boldsymbol{k}_n, \delta), \qquad \delta = m - n
$$

问题变成：$$g$$ 该长什么样？把 $$\boldsymbol{q}_m\cdot\boldsymbol{k}_n$$ 展开（假设 $$\boldsymbol{q}, \boldsymbol{k}$$ 已含位置信息，$$\boldsymbol{q}_m = \boldsymbol{W}_q(\boldsymbol{x}_m + \boldsymbol{p}_m)$$ 等），会产生四类项：

$$
\underbrace{\boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n}_{\text{内容-内容}}
+ \underbrace{\boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{p}_n}_{\text{内容-位置}}
+ \underbrace{\boldsymbol{p}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n}_{\text{位置-内容}}
+ \underbrace{\boldsymbol{p}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{p}_n}_{\text{位置-位置}}
$$

三代相对方案的差异，本质是**对这四项做不同的替换/删减**。

## 2. Transformer-XL：逐项重写

Transformer-XL（Dai et al., 2019）为处理超长文档引入段级循环，需要位置编码能跨段一致，于是提出：

$$
s_{mn} = \underbrace{\boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n}_{\text{内容-内容}}
+ \underbrace{\boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_{k,r}\, \boldsymbol{r}_{\delta}}_{\text{内容-位置}}
+ \underbrace{\boldsymbol{u}^\top \boldsymbol{W}_k \boldsymbol{x}_n}_{\text{位置-内容（u 与 m 无关）}}
+ \underbrace{\boldsymbol{v}^\top \boldsymbol{W}_{k,r}\, \boldsymbol{r}_{\delta}}_{\text{位置-位置（v 与 m,n 无关）}}
$$

其中 $$\boldsymbol{r}_\delta$$ 是相对距离 $$\delta$$ 的 Sinusoidal 编码，$$\boldsymbol{u}, \boldsymbol{v}$$ 是每个头两个可学习向量。关键改动：

- 绝对位置 $$\boldsymbol{p}_n$$ 出现的地方全部换成相对编码 $$\boldsymbol{r}_\delta$$；
- 涉及 $$\boldsymbol{p}_m$$ 的两项里把 query 侧的投影干掉，换成与位置无关的可学习向量——因为"当前查询自身的绝对位置"不重要，重要的只是它和别人**相距多少**。

数学上干净，工程上灾难：打分不再是 $$\boldsymbol{Q}\boldsymbol{K}^\top$$ 这一个矩阵乘，而是四项异质计算（其中两项还依赖 $$\delta$$ 构成的 Toeplitz 矩阵结构），无法塞进标准 GEMM，训练慢、推理难优化。

## 3. T5 bias：一次著名的暴力简化

T5（Raffel et al., 2020）的实验结论堪称位置编码史上最"莽"的一段：**上述四项里，只有"位置-位置"项是位置编码真正需要的，其余全部删掉**，然后把删剩下的项压成一个标量偏置：

$$
s_{mn} = \frac{\boldsymbol{q}_m \cdot \boldsymbol{k}_n}{\sqrt{d}} + b_{\,c(\delta)}
$$

细节设计：

1. **分桶（bucketing）**：相对距离 $$\delta$$ 不是一比一映射到参数，而是先分桶。前 16 个距离各占一桶（精确的近距离），之后按对数间隔合并（远距离粗化）：每个头一张 $$B$$ 个桶的标量表 $$b \in \mathbb{R}^{B}$$（共享实现中 $$B=32$$，双向注意力时对称扩展到 64 桶）。这基于"远距离只需粗粒度感知"的语言学直觉，也把参数量从 $$O(L)$$ 压到 $$O(\log L)$$；
2. **bias 只在打分上加标量**，恢复了矩阵形式：$$\boldsymbol{S} = \boldsymbol{Q}\boldsymbol{K}^\top/\sqrt{d} + \boldsymbol{B}$$，其中 $$\boldsymbol{B}$$ 是逐头不同的 Toeplitz 偏置矩阵，可以预计算后广播加上——工程上比 Transformer-XL 友好得多；
3. 每个注意力头有**独立的** bias 表，所以不同头可以学出不同的"距离偏好"（有的头盯局部，有的头看远程）。

```python
import torch
import torch.nn as nn

class T5RelativeBias(nn.Module):
    """T5 风格的对数分桶相对位置偏置（单向简化版）"""
    def __init__(self, n_heads: int, n_buckets: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.bias = nn.Embedding(n_buckets, n_heads)     # b: (B, H)

    def bucket(self, delta: torch.Tensor) -> torch.Tensor:
        # delta: (L, L)，值域 -(L-1)..0（causal）
        # 前 16 个距离精确分桶，之后对数粗化
        d = -delta                                           # 距离取正
        log = torch.floor(torch.log2(d.clamp(min=16) / 16)) + 16
        return torch.where(d < 16, d, log).long().clamp(max=self.bias.num_embeddings - 1)

    def forward(self, L: int, device):
        m = torch.arange(L, device=device).unsqueeze(1)     # (L, 1)
        n = torch.arange(L, device=device).unsqueeze(0)     # (1, L)
        delta = n - m                                       # δ = m - n 的镜像（causal 下 ≤0）
        b_idx = self.bucket(delta)                          # (L, L) 桶下标
        return self.bias(b_idx)                             # (L, L, H)
```

| 数学符号 | 代码变量 | Shape | 说明 |
|---:|---|:---:|---|
| $$\delta$$ | `delta` | (L, L) | 相对距离矩阵 |
| $$c(\delta)$$ | `b_idx` | (L, L) | 距离→桶下标 |
| $$b$$ | `bias.weight` | (B, H) | 每头偏置表 |
| $$\boldsymbol{B}$$ | 返回值 | (L, L, H) | 加到 logits 上的偏置 |

## 4. 中间变体：XLNet 与 DeBERTa

- **XLNet**（Yang et al., 2019）：基本沿用 Transformer-XL 公式，但因排列语言模型的目标，相对编码在其双流注意力里穿行，实现复杂度进一步上升；
- **DeBERTa**（He et al., 2020）：提出"解耦注意力（disentangled attention）"——内容项与位置项完全分离计算：

$$
s_{mn} = \boldsymbol{x}_m^\top\boldsymbol{W}_q^\top\boldsymbol{W}_k\boldsymbol{x}_n
+ \boldsymbol{p}_\delta^\top \boldsymbol{W}_{q,p}^\top \boldsymbol{W}_{k,p}\boldsymbol{p}_\delta
$$

  内容和相对位置各自算打分再相加。DeBERTa 用它配合 ELECTRA 式训练长期霸榜 SuperGLUE，证明了"内容与位置解耦"这个方向本身是对的——它缺的只是一个更优雅的算子。这个算子一年后由 RoPE 给出。

## 5. 相对编码学派的历史教训

回头看，这一代方案确立了两个正确观念（相对距离才是本质；内容与位置应解耦），但输在了工程形态上：

| 方案 | 位置项形式 | 是否保持 $$\boldsymbol{Q}\boldsymbol{K}^\top$$ | 外推上限 |
|---|---|:---:|---|
| Transformer-XL | 四项重写 + 相对向量 | ✗ | 无硬限（$$\boldsymbol{r}_\delta$$ 解析定义） |
| T5 | 标量 bias 分桶 | 近似（加一个 Toeplitz 矩阵） | **bucket 最大距离**（约 128） |
| DeBERTa | 解耦的相对内容项 | ✗ | 无硬限，实测一般 |
| （对照）RoPE | 乘性旋转 Q/K | **✓ 原封不动** | 需外推技巧（[04](/2026/08/29/posem-04-extrapolation-pi-ntk/)/[05](/2026/08/29/posem-05-yarn-longrope/)篇） |

三条具体教训：

1. **别改公式**。任何对 $$\text{softmax}(\boldsymbol{Q}\boldsymbol{K}^\top)$$ 结构的破坏都会在 CUDA kernel、FlashAttention、张量并行上付出代价——纯软件生态的演化速度远快于论文的修正速度；
2. **加性标量 bias 表达力不足**。$$b_\delta$$ 对所有 $$(m, n)$$ 对一视同仁且与内容无关；RoPE 的乘性旋转让位置调制作用于**向量几何**上，注意力分布因此随距离和内容联合变化；
3. **对数截断 = 外推天花板**。T5 的 bucket 让"训练时距离最大 128"成为模型的世界观，推理时超出的距离全部坍缩进最后一个桶——[04 篇](/2026/08/29/posem-04-extrapolation-pi-ntk/)会看到 RoPE 的高频维度也犯了同款"训练长度内高频震荡"的错误，而解法恰恰相反：不是截断，而是重整频率谱。

**Lab 练习**：
1. 在一个 2 层小 Transformer 上分别接 T5 bias 与不加位置编码，训练一个"复制间隔 k 的 token"的合成任务（如 `a b c ...`，输出与首 token 相距 10 的那个字符），对比两种设置对精确距离的敏感度——预期 T5 bias 在 k≤16 时准确率接近 100%，k>32 后因 bucket 粗化而显著下降；
2. 打印训练后 T5 bias 各头的桶偏置曲线 $$b_c$$，观察哪些头学到了单调递减（局部性）、哪些头学到远程偏好——验证"逐头距离偏好"的说法，预期前几个头的近程桶偏置明显高于远程桶。

## 参考文献

1. Dai et al., *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*, 2019. [arXiv:1901.02860](https://arxiv.org/abs/1901.02860)
2. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), 2020. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
3. Yang et al., *XLNet: Generalized Autoregressive Pretraining*, 2019. [arXiv:1906.08237](https://arxiv.org/abs/1906.08237)
4. He et al., *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*, 2020. [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)
5. 苏剑林，《让研究人员绞尽脑汁的Transformer位置编码》，科学空间. [spaces.ac.cn/archives/8130](https://spaces.ac.cn/archives/8130)
