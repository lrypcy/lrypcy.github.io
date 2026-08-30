---
title: "大模型位置编码（07）：多模态 M-RoPE——当位置从一维坐标变成时空坐标"
date: 2026-08-29 10:20:00 +0800
categories:
  - LLM算法
tags: [llm, 位置编码, mrope, multimodal, qwen2-vl, video]
layout: post
mathjax: true
---

> **系列导航** ｜ 第 07 篇 / 共 08 篇
>
> [← 06 ALiBi](/2026/08/29/posem-06-alibi/) ｜ [00 总览](/2026/08/29/posem-00-overview/)

> **TL;DR**
>
> * **问题**：多模态序列里"位置"不再是下标。一张图的视觉 token 有**二维坐标 $$(h, w)$$**，一段视频有**时间 $$t$$ 和每帧内的 $$(h, w)$$**，音频有自己的帧率。如果沿用 1D 下标（把图像 patch 按光栅序排号），"同一行相邻"和"跨行距离一行"会被编码成完全不同的距离——位置编码与数据的几何结构**错位**。
> * **M-RoPE（Qwen2-VL，2024）**：把 RoPE 的 $$d$$ 个旋转维度切成 3 组，分别用 $$t$$、$$h$$、$$w$$ 三个坐标做旋转角：$$\boldsymbol{q}'_m = \boldsymbol{R}_\Theta(\boldsymbol{t}, \boldsymbol{h}, \boldsymbol{w})\,\boldsymbol{q}$$。注意力打分变成三维距离的混合函数。**纯文本时三个坐标取相同值，精确退化为 1D RoPE**——一个多模态方案不伤文本底座，这是它被后续大量 VLM 沿用的关键。
> * **更大的图景**：M-RoPE 是"位置 = 任意结构化坐标"这个统一观念的落地。ViT 的 2D 位置编码、视频的 3D 编码、语音的帧位置，乃至 Agent 场景的 GUI 坐标（UI 截图的 $$(h,w)$$ 直接对应可点击位置），都是同一观念在不同模态上的实例。位置编码研究的终点，是给**注意力**提供数据的几何。

---

## 1. 一维下标在图像上错在哪里

设图像切成 $$H \times W$$ 的 patch 网格，按光栅序（逐行）展开成一维序列，喂给带 1D RoPE 的 LLM。两个 patch 的"模型距离"是它们的一维下标差：

- 同一行相邻两个 patch：距离 1 ✓（几何距离也是 1）
- 上下相邻两个 patch：距离 $$W$$（几何距离只是 1！）

模型的"距离感"与图像几何**各向异性地错位**：水平方向的局部性被保留，垂直方向的局部性被拉伸 $$W$$ 倍。$$W$$ 还随分辨率动态变化（Qwen2-VL 恰恰主打原生动态分辨率），错位程度本身不可预测。视频更糟：帧间隔 $$t$$ 与帧内距离混在同一个下标轴上，模型无从分辨"下一帧同一位置"和"本帧远处"。

早期 VLM（LLaVA 系）绕开而非解决：图像压到固定网格、视觉编码器内部自己带 2D PE，语言模型看到的"位置"只是 patch 序号，几何信息靠视觉编码器的输出特征隐式携带。这对低分辨率可用，但高分辨率文档（OCR 要精确定位"第 3 行第 14 列"）、长视频（"第 5 秒左上角"）要求**语言模型注意力层面**就拥有时空几何。

## 2. M-RoPE：三维坐标的 RoPE

### 2.1 定义

把 $$d$$ 个旋转维（如 $$d=128$$，即 64 个二维平面）均分三组（余数维分给 $$w$$）：

$$
\boldsymbol{R}_\Theta(t, h, w) =
\mathrm{diag}\big(
\underbrace{\boldsymbol{R}(t\theta_0), \dots}_{d_t \text{ 维：时间组}},
\underbrace{\boldsymbol{R}(h\theta_j), \dots}_{d_h \text{ 维：高组}},
\underbrace{\boldsymbol{R}(w\theta_k), \dots}_{d_w \text{ 维：宽组}}
\big)
$$

$$
\boldsymbol{q}'_m = \boldsymbol{R}_\Theta(t_m, h_m, w_m)\,\boldsymbol{q}_m, \qquad
\boldsymbol{k}'_n = \boldsymbol{R}_\Theta(t_n, h_n, w_n)\,\boldsymbol{k}_n
$$

注意力打分（沿用 [03 篇](/2026/08/29/posem-03-rope/)的推导）：

$$
s_{mn} = \frac{1}{\sqrt{d}}\Big[\text{内容项}_t(t_m-t_n) + \text{内容项}_h(h_m-h_n) + \text{内容项}_w(w_m-w_n)\Big]
$$

$$m, n$$ 各自的三对坐标差分别进入自己那组维度——**时间差调时间维、行差调行维、列差调列维**，三维几何在注意力里同时在场。

### 2.2 三个模态下的取值规则

| 输入 | $$t$$ | $$h$$ | $$w$$ |
|---|---|---|---|
| 纯文本 token | $$m$$ | $$m$$ | $$m$$（三者相同 ⇒ 退化为 1D RoPE） |
| 图像 patch | 该图在序列中的"段号"（图内所有 patch 共享，同一图像内 $$t$$ 不变） | patch 行号 | patch 列号 |
| 视频帧 patch | 帧号 | 帧内行号 | 帧内列号 |

两个关键的工程设计：

1. **文本退化（text-degenerate）**：$$t=h=w=m$$ 时三个 diag 块的旋转角都是 $$m\theta$$，拼起来恰是标准 RoPE（只是频率被重排到三段）。所以同一个模型可以无缝混排文本与视觉 token，文本底座的预训练完全复用——**多模态扩展不必伤单模态**；
2. **跨模态段的 $$t$$ 单调推进**：每来一张新图/新帧，$$t$$ 段整体 +1（实现上常对每个模态段分配一个递增的时间戳）。这保证因果性（后面的图"看见"前面的图）与"第几张图"的粗粒度顺序感，同时不与帧内 $$h/w$$ 干扰。

### 2.3 实现

```python
import torch

def mrope_positions(text_lens, images, base_d=128):
    """
    text_lens: List[int]                        —— 各文本段长度
    images:    List[(H, W)]                     —— 各图像 patch 网格尺寸
    返回: (t, h, w)，每个 (total_len, base_d/2) 的旋转角输入
    """
    dt = dh = dw = base_d // 3                    # 128 -> 42/42/44（余数给 w）
    ts, hs, ws = [], [], []
    m = 0                                          # 全局文本位置指针
    seg_t = 0                                      # 模态段时间戳
    for i, (kind, obj) in enumerate(
            [("text", l) for l in text_lens] + [("img", g) for g in images]):
        if kind == "text":
            pos = torch.arange(m, m + obj)
            ts.append(pos.repeat_interleave(1))    # 文本：t=h=w=m
            hs.append(pos.clone()); ws.append(pos.clone())
            m += obj
        else:
            H, W = obj
            h, w = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            flat_h, flat_w = h.flatten(), w.flatten()
            t_seg = torch.full((H * W,), seg_t, dtype=torch.long)
            ts.append(t_seg); hs.append(flat_h); ws.append(flat_w)
            m += H * W
        seg_t += 1                                 # 下一个模态段 t 推进
    return torch.cat(ts), torch.cat(hs), torch.cat(ws)

# 旋转本身：与 03 篇 apply_rope 相同，只是 cos/sin 分别按三段坐标、三段 θ 拼接计算。
```

| 符号 | 变量 | Shape | 说明 |
|---:|---|:---:|---|
| $$(t, h, w)$$ | `ts/hs/ws` | (L,) 各 | 每个 token 的三维坐标 |
| $$d_t, d_h, d_w$$ | `dt/dh/dw` | — | 旋转维三段切分 |
| $$\boldsymbol{R}_\Theta(t,h,w)$$ | 三段 cos/sin 拼接 | (L, d/2) 角度 | 分组旋转角 |

（工程注：HF transformers 中 Qwen2-VL 的 `Qwen2VLRotaryEmbedding` 就是这个结构，config 里的 `mrope_section=[16,24,24]` 表示 head_dim=128 下按 16/24/24 个**频率对**切分——比例可调，不必然三等分。）

## 3. M-RoPE 的效果与边界

### 3.1 论文证据

Qwen2-VL 论文的消融：在同一视频问答基准上，M-RoPE 相对 1D RoPE 有明显增益，视频越长、分辨率越动态，差距越大——这符合第 1 节的错位分析（错位随 $$W$$ 与帧数放大）。文本任务上 M-RoPE 不掉点（退化性质保证了这一点）。

### 3.2 诚实的边界

1. **三维距离的"混合"没有语义保证**。$$s_{mn}$$ 把三个维度的贡献相加，权重由 $$d_t:d_h:d_w$$ 的切分比例硬编码。"$$t$$ 差 1 帧重要还是 $$h$$ 差 10 行重要"没有原理性答案，是超参；
2. **外推问题原样继承**。M-RoPE 的每一维仍是标准 RoPE 频率谱，[04](/2026/08/29/posem-04-extrapolation-pi-ntk/)/[05](/2026/08/29/posem-05-yarn-longrope/) 篇的全部外推病（相位外推、熵膨胀）在 $$t/h/w$$ 各轴上独立存在——高分辨率大图（$$w$$ 轴超训练范围）会触发与长文本同款的崩溃。YaRN 类手术原则上可对各轴分别做，但"对哪个轴做、系数怎么搜"是尚未充分研究的开放问题；
3. **坐标语义在序列里不传递**。$$t$$ 段的时间戳只在"段间"有意义，图内所有 patch 的 $$t$$ 相同——两张图之间的关系只有"先后"，没有"相隔多远（时间语义）"。做长视频时这仍是粗糙的。

## 4. 更大的家族：位置 = 任意结构化坐标

M-RoPE 的真正启示是方法论层面的，它的亲戚们：

| 方案 | 坐标 | 出处 |
|---|---|---|
| ViT 2D PE | $$(h, w)$$，可学习或 sin-cos 2D | ViT (2020) |
| 3D 视频编码 | $$(t, h, w)$$（时间维低频采样） | ViViT / Video Swin 等 |
| M-RoPE | $$(t, h, w)$$，作用在 LLM 注意力上 | Qwen2-VL (2024) |
| Interleaved M-RoPE | 文本/图像交替段的扩展变体 | Qwen2.5-VL 等后续 |
| GUI 坐标 | 屏幕像素 $$(h, w)$$ 直接做位置 | 各类 UI Agent（M-RoPE 的坐标即点击目标） |

甚至可以把 [02 篇](/2026/08/29/posem-02-relative-t5/)的 T5 bucket、[06 篇](/2026/08/29/posem-06-alibi/)的 ALiBi 都放进这个框架：T5 bias 定义的是"距离→标量"的查表函数，ALiBi 是"距离→线性标量"的解析函数，RoPE/M-RoPE 是"坐标→正交变换"的解析函数。**位置编码 = 数据几何到注意力调制的一个映射**，2025 年之后的新方案（如按内容定义位置的 CoPE、把位置信息做进 logits 残差的各类工作）都在这个映射空间里继续探索。

## 5. 系列总结

八篇走完，位置编码的完整故事线：

1. **问题**（[00](/2026/08/29/posem-00-overview/)）：attention 是集合算子，位置是被丢弃后必须买回来的信息；
2. **史前**（[01](/2026/08/29/posem-01-sinusoidal-learned/)–[02](/2026/08/29/posem-02-relative-t5/)）：加性绝对 PE → 改公式的相对 PE，确立了"相对距离 + 内容解耦"两个正确观念，却输在工程形态；
3. **主角**（[03](/2026/08/29/posem-03-rope/)）：RoPE 用旋转把两个观念一次兑现，且不改注意力公式；
4. **攻坚**（[04](/2026/08/29/posem-04-extrapolation-pi-ntk/)–[05](/2026/08/29/posem-05-yarn-longrope/)）：外推性 = 逐维频率重整 + 注意力分布重整，从 PI 的刀切到 LongRoPE 的逐维搜索；
5. **另一条路**（[06](/2026/08/29/posem-06-alibi/)）：ALiBi 证明天生外推可以用硬先验换，市场最终按"表达力上限"裁决给 RoPE 阵营；
6. **泛化**（[07](/2026/08/29/posem-07-mrope-multimodal/)）：位置 → 时空坐标，M-RoPE 把 RoPE 推广到任意结构化几何。

回头看，苏剑林 2021 年那篇 RoPE 原文里"绝对形式实现相对位置"的构造，几乎是这条整条演进链的种子——频率谱的几何级数带来了外推战场，正交性带来了工程兼容，绝对形式带来了向多维坐标的自然推广。一个好的数学构造的红利，能吃很多年。

**Lab 练习**：
1. 用 2.3 的代码生成"一段文本 + 一张 4×6 图 + 一段文本"的三维坐标，打印 $$(t,h,w)$$ 表——预期文本段满足 $$t=h=w=m$$（退化性质），图像段所有 patch 的 $$t$$ 相同但 $$h/w$$ 各不同，亲手验证 M-RoPE 的文本退化与空间编码；
2. 复现第 1 节的错位实验：构造一个合成检索任务（答案位于二维网格中的 $$(h_0, w_0)$$，query 给出坐标描述），对比 1D 光栅序 RoPE 与 M-RoPE 的小模型在"水平相邻/垂直相邻/跨行"三类位置的检索准确率——预期 1D RoPE 在垂直相邻和跨行任务上准确率低 20-40%，M-RoPE 三者接近，各向异性差异会直接出现在准确率矩阵里。

## 参考文献

1. Wang et al., *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*, 2024. [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
2. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
3. 苏剑林，《Transformer升级之路：4、RoPE是一种很好的位置编码》. [spaces.ac.cn/archives/8265](https://spaces.ac.cn/archives/8265)
4. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT), 2020. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
5. Qwen2.5-VL 技术报告（Interleaved-MRoPE 及后续改进）. [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
