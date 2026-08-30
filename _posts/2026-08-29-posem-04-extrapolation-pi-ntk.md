---
title: "大模型位置编码（04）：长度外推（上）——从外推性分析到 PI 与 NTK-aware"
date: 2026-08-29 09:20:00 +0800
categories:
  - LLM算法
tags: [llm, 位置编码, rope, extrapolation, position-interpolation, ntk-aware]
layout: post
mathjax: true
---

> **系列导航** ｜ 第 04 篇 / 共 08 篇
>
> [← 03 RoPE 详解](/2026/08/29/posem-03-rope/) ｜ [05 长度外推（下）YaRN/LongRoPE →](/2026/08/29/posem-05-yarn-longrope/)

> **TL;DR**
>
> * **外推失败的机理**：RoPE 的每个二维平面是一个频率为 $$\theta_i$$ 的振荡器。训练长度 $$L_{\text{train}}$$ 内，第 $$i$$ 维只"见过"相位区间 $$[0, L_{\text{train}}\theta_i]$$；推理超出后，**高频维（短波长 $$\lambda_i$$）率先进入未见的相位组合**，注意力 logits 的分布整体漂移，softmax 之后要么过度聚焦要么平均化——困惑度雪崩。一句话：外推病是**逐频率**发作的。
> * **位置插值 PI（Chen et al., 2023）**：把位置下标线性压缩 $$m \to m/s$$（$$s$$ 为扩展倍数），等价于所有频率除以 $$s$$。粗暴但有效，且**需要的微调极少**（约 1000 步）——因为压缩后的所有相位都落在训练见过的范围内。代价是短程分辨率受损（相邻位置的距离被压近了）。
> * **NTK-aware（社区方案，2023.06）**：治好 PI 的短程损伤——低频维才需要压缩（它们负责长程），高频维保持原样（它们负责局部分辨率且本身不外推去多远）。实现上不逐维缩放，而是**改底数**：$$\text{base}' = \text{base}\cdot s^{d/(d-2)}$$，一次改动同时实现"低频压、高频保"。零训练即可小幅外推（1.5–2 倍）的性价比，使其成为 Dynamic NTK 与 YaRN 的直接前身。

---

## 1. 预备：把 RoPE 看成频率谱

[03 篇](/2026/08/29/posem-03-rope/)给了 $$\theta_i = \text{base}^{-2i/d}$$。做频率谱分析，用**波长**更直观：

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi\cdot \text{base}^{2i/d}
$$

以 LLaMA-1 为例（$$d=128$$，base=10000）：$$\lambda_0 \approx 6.3$$（最高频维：6 个 token 就震荡一个周期），$$\lambda_{63} \approx 6.3\times10^4$$（最低频维：训练长度 2048 内只转了 1/30 圈）。定义第 $$i$$ 维在训练长度内的**振荡周期数**：

$$
r_i = \frac{L_{\text{train}}}{\lambda_i}
$$

- $$r_i \gg 1$$（高频维）：训练中转了很多圈，见过各种相位；
- $$r_i \ll 1$$（低频维）：训练中几乎单调，相当于绝对坐标。

外推到 $$L_{\text{test}} = s \cdot L_{\text{train}}$$ 时，问题出在哪？科学空间的分析视角（苏剑林对 RoPE 外推的系列讨论）指出：**两类维各有各的病**：

1. **高频维的相位外推**：虽见过全部相位值域，但"位置 $$m$$ 上的相位 $$m\theta_i$$"这个组合在训练分布外的高密度区出现，注意力打分 $$\cos(\delta\theta_i)$$ 的取值对未见过的 $$\delta$$ 精确值敏感（$$\delta\theta_i$$ 稍微错过 $$2\pi$$ 的整数倍就完全换了符号）——**距离感错乱**；
2. **低频维的坐标外推**：训练中单调递增的部分继续单调递增，问题不大——低频维反而是外推最安全的部分。

这个"逐维诊断"是理解 PI/NTK/YaRN 全家桶的钥匙：**好的外推方案 = 低频压、高频保，并按维度连续过渡**。

## 2. Position Interpolation（PI）：线性压缩坐标

### 2.1 做法

Chen et al.（Meta，2023.06）的方案朴素到一句话：把位置除以 $$s$$ 再喂给 RoPE：

$$
m \;\longrightarrow\; m' = \frac{m}{s}
\qquad\Longleftrightarrow\qquad
\theta_i \;\longrightarrow\; \theta_i' = \frac{\theta_i}{s}
$$

所有频率统一除以 $$s$$，所有波长统一乘以 $$s$$。推理长度 $$s\cdot L_{\text{train}}$$ 处的相位 = 原训练长度边界处的相位——**所有相位都回到训练见过的范围**。论文里作者说得很形象："外推是走出训练区间，插值是把长序列塞回训练区间"。

### 2.2 效果与代价

- LLaMA-7B 从 4k 外推 32k：**只需约 1000 步微调**即可恢复到接近原生质量（对比"直接在长数据上继续训练"的 10B token 级开销）；
- 代价一：**短程分辨率受损**。相邻 token 的角度差从 $$\theta_i$$ 缩为 $$\theta_i/s$$，模型对"距离 1 和距离 2"的区分变钝——精细的局部任务（代码缩进、字符级复制）会掉点；
- 代价二：**统一缩放错配病根**。高频维本来就只负责局部、几乎不承担长程外推压力，却被压缩了分辨率；低频维需要大幅压缩才能在 $$sL$$ 内转完训练见过的角度，只压 $$s$$ 倍可能还不够——一刀切对两头都不最优。

### 2.3 一个常被忽视的点：PI 是"插值"，不是免费午餐

PI 的名字容易造成误解——它并不是免训练的。零训练直接 PI 通常严重掉点（比不 PI 略好），1000 步微调才是标准用法。真正"零训练可用"的是下面 NTK 家族的温和版本。

## 3. NTK-aware：按频率非均匀缩放

### 3.1 思想

社区（Code Llama 作者背景的 bhaias/emozilla 等，2023 年中 Reddit r/LocalLLaMA 与 HF 讨论中成形）提出的修正：**按波长区分对待**——

- 高频维（$$\lambda_i$$ 小）：保持原频率（局部分辨率是刚需，且高频外推病主要靠微调治）；
- 低频维（$$\lambda_i$$ 大）：按 PI 的思路压缩（长程坐标必须回到训练范围）。

理想形式是逐维选取缩放系数 $$s_i$$，从 1（最高频）平滑过渡到 $$s$$（最低频）。但逐维调参太不工程——NTK-aware 的聪明之处是发现**改一个参数就能近似做到**。

### 3.2 改底数：数学

关键观察：$$\theta_i = \text{base}^{-2i/d}$$ 是几何级数，改 base 等价于沿维度轴**整体重新标定**频率谱的斜率。把 base 换成：

$$
\boxed{\;\text{base}' = \text{base}\cdot s^{\,d/(d-2)}\;}
$$

（$$d$$ 为旋转维数，如 LLaMA 的 128；常用近似 $$d/(d-2) \approx 1$$，即 base' ≈ base·s，社区也常用这个粗版。）验证一下它实现了什么：在位置 $$m$$、维度 $$i$$：

$$
m\,\theta_i' = m\cdot \text{base}'^{-2i/d}
$$

- 最高频维 $$i=0$$：$$\theta_0' = \theta_0 = 1$$，**完全不变**；
- 最低频维 $$i=d/2-1$$：$$\theta' \approx \theta \cdot$$ 相当于缩了约 $$s$$ 倍（推导：$$\theta_i' = \text{base}^{-2i/d}\, s^{-2i/(d-2)}$$，$$i$$ 越大缩放越狠，指数 $$-2i/(d-2)$$ 在 $$i = d/2-1$$ 处约为 $$-1$$，即缩 $$s$$ 倍）；
- 中间维度：连续插值。

一次改底数 = "高频 1 倍、低频 $$s$$ 倍"的连续谱缩放。这正是第 1 节诊断开出的药方。

### 3.3 效果与局限

- **零微调**即可把 4k 模型外推到 5k–8k（约 1.5–2 倍）且几乎无损——PI 做不到这一点；
- 再往外就需要微调了，且注意一个结构性问题：训练见过的**最长绝对相位**仍然受 $$L_{\text{train}}$$ 限制，高频维在中长距离上的相位组合依旧未见过。NTK-aware 只是缓解了 PI 的短程损伤，没有"治愈"外推病；
- 边界模糊：base 改大后，波长谱整体右移，模型对中程距离的区分也发生漂移——这就是为什么后续 YaRN 要引入**逐维精确控制**与**注意力温度**（[05 篇](/2026/08/29/posem-05-yarn-longrope/)）。

### 3.4 Dynamic NTK：推理时的自适应版本

推理输入长度逐步增长时（流式解码），固定 $$s$$ 会在序列还短的时候过早缩放。Dynamic NTK（kaiokendev，2023）的做法：随当前序列长度动态更新

$$
s(t) = \max\!\left(1, \; \frac{l(t)\ \text{的长度}}{L_{\text{train}}}\right), \qquad
\text{base}'(t) = \text{base}\cdot s(t)^{d/(d-2)}
$$

序列没超训练长度时完全不缩放，超过多少缩多少。vLLM 等推理框架早期内置的 `rope_scaling="dynamic"` 即此。缺点：每次越界都要重建 cos/sin 缓存并重算 KV（缓存过的 K 是用旧 base 旋转的，位置相关的旋转不可简单事后修正——RoPE 的可逆性救不了缓存里已经乘上去的矩阵，只能在新 base 下重新旋转原始 K，即重算或存"未旋转 K"）。

## 4. 方案对照

| 方案 | 操作 | 零训练可用 | 需微调时步数 | 短程分辨率 | 逐维控制 |
|---|---|:---:|:---:|:---:|:---:|
| 直接外推 | 什么都不做 | 1.0–1.2 倍 | — | ✓ | — |
| PI | $$\theta_i \to \theta_i/s$$ | ✗ | ~1000 | ✗ | ✗（一刀切） |
| NTK-aware | base 乘 $$s^{d/(d-2)}$$ | ~2 倍 | 少量更好 | ✓ | 半（谱斜率） |
| Dynamic NTK | 随长度更新 base | ~2 倍 | 少量更好 | ✓ | 同上 |
| YaRN（[05 篇](/2026/08/29/posem-05-yarn-longrope/)） | 逐维 ramp + 温度 | ✗（需少量微调） | ~400 | ✓ | **✓** |

一条清晰的演进逻辑线：**统一缩放（PI）→ 谱斜率缩放（NTK）→ 逐点缩放 + 分布修正（YaRN）→ 逐点非均匀搜索（LongRoPE）**。每一代都在把"缩放"做得更细，同时补上前一代漏掉的另一半问题——注意力分布本身的熵漂移。

## 5. 实验：亲眼看外推崩溃

```python
import torch

def attn_entropy(L: int, d: int = 64, base: float = 10000.0, seed: int = 0):
    """随机 Q/K 下，最后一个位置对全序列的注意力分布熵。训练长度假设为 512。"""
    torch.manual_seed(seed)
    q = torch.randn(L, d); k = torch.randn(L, d)
    inv = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    def rot(x, m):                       # R_Θ(m) x，奇偶配对
        x1, x2 = x[..., 0::2], x[..., 1::2]
        ang = m * inv
        return torch.cat([x1 * ang.cos() - x2 * ang.sin(),
                          x1 * ang.sin() + x2 * ang.cos()], dim=-1)
    qm = rot(q, L - 1)
    km = torch.stack([rot(k, n) for n in range(L)])
    p = torch.softmax(qm @ km.T / d**0.5, dim=-1)
    return -(p * p.clamp_min(1e-12).log()).sum().item()

for L in [512, 1024, 2048, 4096]:
    print(L, attn_entropy(L))
# 观察：L 翻倍时熵持续上升（注意力被拉平）——外推病的最小复现。
# 再把 base 换成 10000 * 8**(64/62) 重跑，看 NTK-aware 如何压住熵漂移。
```

| 符号 | 变量 | Shape | 说明 |
|---:|---|:---:|---|
| $$L_{\text{train}}, s$$ | `512`, `L/512` | — | 训练长度与扩展倍数 |
| $$\theta_i$$ | `inv` | (d/2,) | 角频率 |
| $$s_{mn}$$ | `qm @ km.T` | (L,) | 末位 query 对各 key 的打分 |

**Lab 练习**：
1. 跑上面的熵实验，分别对 base ∈ {10000, 40000, 10000·8^{64/62}}、L ∈ {512 … 16384} 画熵曲线族——预期底数越大、熵漂移发作越晚：base=10000 时 L=2048 熵已达 3.5 bits，base=10000·8^{64/62} 时 L=4096 熵仍低于 2.5 bits，验证"改底数延缓外推崩溃"；
2. 取任意一个 4k 上下文的开源模型（如 LLaMA-2-7B），用 HF transformers 的 `rope_scaling={"type": "linear", "factor": 8.0}` 直接推理一个 16k 的 needle-in-haystack 提示，记录失败模式——预期零训练 PI 会出现"找到 needle 但引用错位"或"完全找不到"的掉点，亲手体验 PI 零训练的缺陷。

## 参考文献

1. Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation*, 2023. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)
2. Rozière et al., *Code Llama: Open Foundation Models for Code*, 2023（NTK-aware base 缩放的首次大规模应用）. [arXiv:2308.12950](https://arxiv.org/abs/2308.12950)
3. bhaias / emozilla，*NTK-Aware Scaled RoPE*（社区方案原帖）. [Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
4. kaiokendev，*Dynamic NTK* 说明. [kaiokendev.github.io](https://kaiokendev.github.io/context)
5. 苏剑林，《Transformer升级之路》系列关于 RoPE 外推与底数选择的讨论. [spaces.ac.cn](https://spaces.ac.cn/archives/8265)
