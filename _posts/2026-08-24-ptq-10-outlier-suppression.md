---
title: "大模型量化算法（11）：Outlier Suppression / OS+——从保护到抑制"
date: 2026-08-24 15:40:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, outlier-suppression, w8a8, layernorm]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 2 · Activation PTQ** ｜ 第 11 篇 / 共 26 篇
>
> [← 10 ZeroQuant](/2026/08/24/ptq-06-smoothquant-zeroquant/) ｜ [12 QuaRot/SpinQuant →](/2026/08/24/ptq-07-quarot-spinquant/)
>
> - **第 11 篇 Outlier Suppression / OS+(本文)** -- W8A8 补遗:γ 迁移、token-wise 裁剪与 shift+scale 等效变换。后续姊妹篇:[第 13 篇 RPTQ/QUIK/ATOM](/2026/08/24/ptq-11-rptq-quik-atom/)、[第 14 篇 OliVe](/2026/08/24/ptq-12-olive-abfloat/)、[第 16 篇 QoQ/QServe 与 QQQ](/2026/08/24/ptq-13-qserve-qqq/)

---

## TL;DR

1. **离群值处理有两条路线,本篇走的是第二条**。LLM.int8()(第 E1 篇)、SpQR/OWQ(第 06 篇)乃至 OliVe(第 14 篇)都在"保护"离群值--把它们挑出来用高精度算;Outlier Suppression(OS,arXiv:2209.13325)反过来问:**离群值是从哪里被制造出来的?**答案是 LayerNorm 的尺度参数 $\gamma$--它在训练中吸收并放大了特定通道的幅值,是名副其实的"离群值放大器"。把 $\gamma$ 迁移进后续权重(Gamma Migration),放大器被拆除,分布自然变乖。
2. **OS+(arXiv:2304.09145)补上了另一半拼图:不对称性**。光拆放大器不够--离群通道的分布往往整体偏移(大的正直流分量 + 小幅波动),只做 SmoothQuant 式的缩放救不了它们自己(本文 §5 Demo B 实测:**仅 scale 时离群通道的有效级数与不处理完全相同**,1.9 级 vs 1.9 级)。OS+ 加入 **channel-wise shift**(减去逐通道均值)再 scale,同一实验里离群通道有效级数从 1.9 提到 36,层输出误差相对仅-scale 方案再降一半。
3. **所有变换都必须能"免费"迁移回权重**,这是这条路线的立身之本。shift 迁入前一跳 LayerNorm 的 $\beta$,scale 迁入后继线性层的输入通道--浮点模型的数学输出分毫不变(等效变换),只有量化器看到的分布变了。这套"变换 + 统一迁移"的模式后来被 OmniQuant(第 05 篇,把变换参数改成可学习)和旋转路线(第 12 篇,变换换成正交矩阵)继承,是理解 2023 之后激活量化工作的通用语法。

---

## 目录

1. [引言:保护还是抑制?](#1-引言保护还是抑制)
2. [共同的物理源头:LayerNorm 的仿射参数](#2-共同的物理源头layernorm-的仿射参数)
   - 2.1 [30 秒回顾 LayerNorm](#21-30-秒回顾-layernorm)
   - 2.2 [γ 为什么是离群值放大器](#22-γ-为什么是离群值放大器)
3. [Outlier Suppression(v1):拆掉放大器 + 聪明地裁剪](#3-outlier-suppressionv1拆掉放大器--聪明地裁剪)
   - 3.1 [Gamma Migration:等效变换推导](#31-gamma-migration等效变换推导)
   - 3.2 [token-wise 裁剪](#32-token-wise-裁剪)
4. [Outlier Suppression+:shift + scale 的完整形态](#4-outlier-suppressionshift--scale-的完整形态)
   - 4.1 [新观察:通道间的不对称形状](#41-新观察通道间的不对称形状)
   - 4.2 [为什么仅靠 scale 救不了离群通道:一个算术事实](#42-为什么仅靠-scale-救不了离群通道一个算术事实)
   - 4.3 [Channel-wise Shift:数学形式与最优性](#43-channel-wise-shift数学形式与最优性)
   - 4.4 [统一迁移模式:每一步都免费](#44-统一迁移模式每一步都免费)
   - 4.5 [与 SmoothQuant 的精确关系](#45-与-smoothquant-的精确关系)
5. [代码实现(numpy):γ 迁移与 shift+scale 对照实验](#5-代码实现numpyγ-迁移与-shiftscale-对照实验)
6. [批判与展望](#6-批判与展望)
7. [常见问题 FAQ](#7-常见问题-faq)
8. [参考清单](#8-参考清单)

---

## 1. 引言:保护还是抑制?

把本系列已经讲过的离群值方案排成一列,会发现它们共享同一个思维定式:

| 方案 | 对离群值的姿态 | 代价 |
|:---|:---|:---|
| LLM.int8()(第 E1 篇) | 挑出来走 fp16 | 分裂 GEMM,kernel 复杂 |
| SpQR/OWQ/SqueezeLLM(第 4/9 篇) | 挑出来存高精度 | 稀疏格式 + 双路径 kernel |
| AWQ(第 05 篇) | 缩放保护敏感通道 | 需要 grid search |
| ATOM/OliVe(第 11/12 篇) | 混合精度/配对编码 | 定制硬件或复杂编码 |

它们的共同点是**接受离群值的存在**,围绕它做架构文章。Outlier Suppression(OS)的第一作者贡献是换了一个问题:

> 与其围着离群值修围栏,能不能让它**根本不长出来**?

这个问题的答案藏在所有 Transformer 都有的一个部件里:LayerNorm 的可学习尺度 $\gamma$。OS 论文的两个核心发现:

1. **$\gamma$ 是离群值放大器**:训练后的 $\gamma$ 在特定通道上取值极大,归一化后本来温和的隐状态经过 $\gamma\odot\hat x+\beta$ 一乘,离群值才真正"长"出来;
2. **激进裁剪是安全的**:覆盖面积较大的离群值可以被 token-wise 地剪切而不掉点。

于是 OS 的处方有两味药:**Gamma Migration**(把 $\gamma$ 迁出归一化层,拆除放大器)和 **token-wise clipping**(按 token 自适应裁剪)。一年后的 OS+ 发现光这两招还差一步--离群分布不只是"大",而且"偏"(不对称),于是加入 **channel-wise shift**,并把所有操作统一成一套**可逆迁移**的语法。最终效果(论文口径):W8A8 下多个模型/任务超过 SmoothQuant,且不需要调 $\alpha$ 超参。

一句话定位:**OS 家族 = "分布整形"路线的最小完备集**。读懂它,第 05 篇 OmniQuant 的可学习变换和第 12 篇 QuaRot 的旋转变换都只是"换了更强整形函数"的递归。

## 2. 共同的物理源头:LayerNorm 的仿射参数

### 2.1 30 秒回顾 LayerNorm

对单个 token 的隐向量 $x \in \mathbb{R}^d$(为简洁省略 layer index):

$$
\hat{x}_j = \frac{x_j - \mu(x)}{\sigma(x)},\qquad
y_j = \gamma_j \hat{x}_j + \beta_j,\quad j=1,\dots,d
$$

其中 $\mu,\sigma$ 是**逐 token** 跨通道统计(所以 LN 本身不改变通道间的相对格局),$ \gamma, \beta \in \mathbb{R}^d$ 是**逐通道**可学习参数。关键结构:$y$ 对 $x$ 做的是"逐 token 归一化",对通道维做的却是"逐通道仿射"--**通道间的一切不平等都是 $\gamma,\beta$ 干的**。

**变量映射表**:

| 数学符号 | 代码变量 | Shape | 说明 |
|:---|:---|:---:|:---|
| $x$ | `x` | $(T, d)$ | 进入 LN 的残差流隐状态 |
| $\hat{x}$ | `xhat` | $(T, d)$ | 归一化后、未加仿射的中间量 |
| $\gamma_j$ | `gamma[j]` | $(d,)$ | LN 尺度参数(离群值放大器) |
| $\beta_j$ | `beta[j]` | $(d,)$ | LN 偏移参数(shift 的迁移宿主) |
| $y$ | `a_ln` | $(T, d)$ | LN 输出,下游线性层的输入 |
| $W$ | `W` | $(d_{out}, d_{in})$ | 后继线性层权重 |
| $\delta_j$ | `mu[j]` | $(d,)$ | OS+ 的逐通道移位量(校准均值) |
| $s_j$ | `s_os[j]` | $(d,)$ | OS+ 的逐通道缩放量 |

### 2.2 γ 为什么是离群值放大器

训练动力学的经验事实(第 E1 篇讲过):少数通道存在跨 token 稳定的幅值优势,attention/MLP 依赖这些通道传递"全局信号"。训练会把这份依赖写进 $\gamma$--这些通道对应的 $\gamma_j$ 长到 5~20 倍于其他通道。于是 LN 输出的幅值谱变成:

$$
|y_j| \approx |\gamma_j|\cdot|\hat{x}_j|
$$

即便 $\hat x$ 是温良的(逐 token 归一化保证 $\|\hat x\|$ 有界),乘上悬殊的 $\gamma$ 之后,$y$ 的 per-tensor absmax 就被少数几个通道劫持。OS 论文观察到:**离群值主要在 LN 乘法之后成型**--$\hat x$ 上本来只有"苗头",$\gamma$ 把苗头浇成了离群值。这就是"放大器"一词的准确含义:它不是离群值的源头(源头上游还有 attention 的结构性偏好),但它是最直接的增益级。

> 💡 **为什么这很重要**:如果离群值是 $\gamma$ 造出来的,那它就是**模型参数的一部分**,而不是数据的属性--参数可以重写,数据不行。这就是"抑制"路线可行的全部依据:对数据我们只能分流保护,对参数我们可以做等效重写。

## 3. Outlier Suppression(v1):拆掉放大器 + 聪明地裁剪

### 3.1 Gamma Migration:等效变换推导

考虑 LN 输出 $y$ 进入线性层 $Y = yW^\top + b$(行向量约定,$W\in\mathbb{R}^{d_{out}\times d_{in}}$)。代入 $y=\gamma\odot\hat x+\beta$:

$$
Y = (\gamma\odot\hat x + \beta)W^\top + b
  = \hat x\,(\mathrm{diag}(\gamma)\, W^\top) + (\beta W^\top + b)
$$

定义迁移后的参数:

$$
W' = W\,\mathrm{diag}(\gamma),\qquad b' = b + W\beta
$$

则 $Y = \hat x W'^\top + b'$ **严格成立**(浮点意义下 bit 级等效,舍入顺序差异除外)。也就是说:$\gamma$ 从 LN 里搬进了后继权重的对应输入通道,$\beta$ 搬进了 bias,LN 退化为无参归一化($\gamma\equiv 1,\beta\equiv 0$),而网络输出不变。

收益在哪里?对比量化器看到的张量:

* 迁移前:量化对象是 $y=\gamma\odot\hat x+\beta$,其 per-tensor absmax 被 $\gamma$ 放大的通道决定;
* 迁移后:量化对象是 $\hat x$,absmax 只由归一化后的天然幅度决定。

§5 Demo A 用合成数据实测:同样 per-tensor int8,**直接量化 LN 输出相对误差 0.136,Gamma Migration 后 0.029,降低约 79%**。

三个工程注脚:

1. **多消费者要同步迁移**。一个 LN 输出通常同时喂给 Q/K/V 三个投影(或 gate/up 两个),每个消费者的 $W$ 都要各自乘 $\mathrm{diag}(\gamma)$,漏一个是静默的数值错误;
2. **残差流不受影响**。Transformer 的残差连接取的是 LN 的**输入** $x$ 而非输出 $y$,所以迁移不会污染恒等支路--这也是后面 OS+ 能"逐块安全迁移"的结构前提;
3. **迁移后 $\gamma$ 不消失,只是搬家**。权重的对应列被放大了同样的倍数,权重侧 per-channel 量化(每个输出通道一个 scale)恰好能吃下这种列间悬殊--这正是第 10 篇说过的"权重易量化、激活难量化"不对称性的又一次胜利。

### 3.2 token-wise 裁剪

拆掉放大器后仍有残余离群。v1 的第二味药是**按 token 自适应地确定裁剪范围**,而非全网固定 $[-a,a]$:

$$
\tilde{x}^{(t)} = \mathrm{clip}\big(x^{(t)},\ [-c_t,+c_t]\big),\qquad
c_t = \arg\min_{c}\ \big\|Q_c\big(x^{(t)}\big)-x^{(t)}\big\|^2
$$

其中 $Q_c$ 是以 $c$ 为 absmax 的对称量化器。直觉:不同 token 的离群程度天差地别(BOS token、首 token 往往幅值普遍偏高),固定阈值要么对所有 token 太松、要么对多数 token 太紧。逐 token 搜索 $c_t$ 让每个 token 在"裁剪损失 vs 舍入损失"之间取自己的平衡点。代价是推理时要为每个 token 记录/计算 $c_t$(动态量化,类似 ZeroQuant 的 per-token scale),工程上并入 scale 的计算流程即可。

## 4. Outlier Suppression+:shift + scale 的完整形态

### 4.1 新观察:通道间的不对称形状

OS+ 重新审视离群通道的分布形状,发现了 v1 没有处理的一件事:**离群通道的分布不是"以零为中心的重尾",而是"整体偏移的小方差分布"**--大量样本堆积在一个很大的正值附近(比如恒为正、均值 40、波动 ±2),跨 token 形状稳定。这种不对称带来两重恶果:

1. 张量的支撑集被拉得很宽(从 0 到 40+),per-tensor 网格被撑爆;
2. 更隐蔽的是,**即使做了 SmoothQuant 式逐通道缩放,该通道自己的分辨率也不会改善**--下一小节用一个算术事实说明这一点。

### 4.2 为什么仅靠 scale 救不了离群通道:一个算术事实

设某离群通道 $j$ 的值为 $x_{ij} = \mu_j + \epsilon_{ij}$,$\mu_j = 40$,$|\epsilon|\le 2$,全张量 absmax 由它决定,$M \approx 42$。对称 int8 的步长:

$$
\Delta = \frac{M}{127} \approx 0.33
$$

现在做 SmoothQuant 式处理:除以逐通道 scale $s_j = \max_i|x_{ij}| \approx 42$,再对结果做 per-tensor int8(此时网格步长归一化为 $2/254$)。反推回原始单位,**该通道的实际步长**:

$$
\Delta_j^{(\text{scale})} = \frac{s_j}{127} \approx \frac{42}{127} = \Delta
$$

**分毫未变**。逐通道 scale 只是给每个通道发了"自己的网格",而该通道的网格密度由它自己的 absmax 决定--absmax 没变(减不掉那个 40 的直流分量),密度就不会变。scale 真正救的是**其他通道**(它们借用了更细的全局步长),这一点 §5 Demo B 会给出直观的级数对比。

要让该通道自己受益,必须先把 40 的偏移去掉--这就是 shift 的职责。

### 4.3 Channel-wise Shift:数学形式与最优性

对激活的第 $j$ 个通道,减去一个校准集上估计的移位量:

$$
\tilde{x}_{ij} = x_{ij} - \delta_j
$$

移位后分布以零为中心、支撑集收窄,再叠加逐通道 scale $s_j$(以及可选的裁剪),量化器看到的就是一组"均值零、幅度均匀"的通道。$\delta_j$ 的选择:

* **最小化绝对动态范围**准则:对单通道样本集合,使 $\max_i \tilde x - \min_i \tilde x$ 最小的移位是**中点** $\delta_j^\* = (\max_i x_{ij} + \min_i x_{ij})/2$;
* **矩准则**:对单峰近对称的残差,取**均值** $\delta_j = \mathbb{E}[x_{ij}]$(校准集逐通道平均)更稳健--它不受单个极端样本摆布,且让二阶矩(量化噪声的能量上限)最小化。

两种准则在"重尾但偏移明确"的离群通道上给出相近的结果;OS+ 在校准集上估计移位与缩放(scale 的估计沿用"激活难度向权重迁移"的思想),不再像 SmoothQuant 那样依赖人工调节的 $\alpha$ 网格搜索(论文的具体估计式以原文为准,此处保留两种准则的推导供对照)。

### 4.4 统一迁移模式:每一步都免费

等效变换是这条路线的纪律:**任何对激活的改造,都必须能在浮点图中找到等价的参数重写**。OS+ 给出了统一的迁移清单:

| 变换 | 施加点 | 迁移宿主 | 等价性 |
|:---|:---|:---|:---|
| shift $-\delta$ | LN 输出 | LN 的 $\beta' = \beta - \delta$ | 严格(仿射吸收) |
| shift $-\delta$(内部激活) | 线性层输入 | $b' = b + \delta W^\top$ | 严格(§3.1 同款) |
| scale $\div s$ | 线性层输入 | $W' = W\,\mathrm{diag}(s)$ | 严格(SmoothQuant 同款) |
| scale $\div s$(内部激活) | 生产侧线性层 | 反量化 epilogue 中并入 $s$ | 严格(dequant 因子重排) |

两个结构性红利让这张表在实践中真的成立:

1. **残差流免疫**:如 §3.1 注脚 2,分支内的激活变换不触碰恒等支路,因此可以逐块独立实施而不累积误差;
2. **消费者有限**:一个激活张量的消费者就是同分支的一两个线性层,权重重写的范围可控、一次性完成。

于是整个方法可以压缩成一个循环:*校准前向 → 统计每条边的 $(\delta, s)$ → 重写两端参数 → 浮点输出不变、量化分布变乖*。这个循环正是后来 OmniQuant"把 $(\delta,s)$ 改成梯度可学的参数"、QuaRot"把仿射变换换成哈达玛旋转"的共同骨架。

### 4.5 与 SmoothQuant 的精确关系

把两者的变换族摆在一起,包含关系一目了然:

$$
\text{SmoothQuant}: \tilde{x} = \frac{x}{s(\alpha)}\quad\subset\quad
\text{OS+}: \tilde{x} = \frac{x - \delta}{s}
$$

* SmoothQuant 是 OS+ 变换族中 $\delta \equiv 0$ 的特例;
* 对称分布($\delta\approx 0$)下两者退化为同一方法--所以 SmoothQuant 在离群不严重的层/模型上也够用;
* 分布显著偏移时(§4.1 的形态),shift 项不可替代--§5 Demo B 实测:同样条件下 shift+scale 比仅 scale 层输出误差再降约一半,且离群通道自身的有效分辨率提高一个数量级以上。

论文口径的结果(未逐一复验,以原文为准):OPT/BLOOM 系列多规模上,W8A8/W6A6 等设置的困惑度与零样本任务准确率全面优于 SmoothQuant,且免去 $\alpha$ 调参。

## 5. 代码实现(numpy):γ 迁移与 shift+scale 对照实验

两个可运行实验:**Demo A** 验证"$\gamma$ 是放大器"+ Gamma Migration 的误差收益;**Demo B** 在带不对称离群通道的激活上对比 直接量化 / 仅 scale / shift+scale 三种处理。依赖 numpy 即可。

```python
import numpy as np

rng = np.random.default_rng(0)

# ---------- 公共构件 ----------
def per_tensor_sym_quant(A, bits=8):
    """对称 per-tensor 量化（激活量化的最坏基线）"""
    s = np.abs(A).max() / (2 ** (bits - 1) - 1)
    return np.round(A / s).clip(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s

def rel_err(Y, Y_ref):
    return float(np.linalg.norm(Y - Y_ref) / np.linalg.norm(Y_ref))

# ---------- Demo A: γ 放大效应 与 Gamma Migration ----------
d, n_tokens, n_outlier = 512, 4096, 3
xhat = rng.normal(0, 1.0, (n_tokens, d))
oc = rng.choice(d, n_outlier, replace=False)
xhat[:, oc] *= 6                      # 归一化后的常驻离群通道（真实 LLM 中存在）
gamma = np.ones(d); gamma[oc] *= 5    # 训练后的 γ 进一步放大这些通道
beta = rng.normal(0, 0.1, d)
W = rng.normal(0, 0.02, (d, d))       # 下游线性层

a_ln = gamma * xhat + beta            # LN 输出（fp16 路径的真实值）
Y_ref = a_ln @ W.T
print("[DemoA] γ 迁移前后的动态范围与量化误差")
print(f"  xhat    per-tensor absmax = {np.abs(xhat).max():7.2f}")
print(f"  LN 输出 per-tensor absmax = {np.abs(a_ln).max():7.2f}"
      f"   （γ 放大 {np.abs(a_ln).max()/np.abs(xhat).max():.1f}x）")

# 路径 1：直接量化 LN 输出
e_direct = rel_err(per_tensor_sym_quant(a_ln) @ W.T, Y_ref)
# 路径 2：Gamma Migration——γ 折进权重，β 折进 bias，量化对象变成 xhat
W_mig = W * gamma[None, :]
b_mig = beta @ W.T                    # 常数项 fp16 计算，不参与激活量化
e_mig = rel_err(per_tensor_sym_quant(xhat) @ W_mig.T + b_mig, Y_ref)
print(f"  直接量化 LN 输出   : 输出相对误差 = {e_direct:.4f}")
print(f"  Gamma Migration 后 : 输出相对误差 = {e_mig:.4f}   （降低 {(1-e_mig/e_direct)*100:.1f}%）")
```

```python
# ---------- Demo B: OS+ 的 shift+scale vs 仅 scale（SmoothQuant 型） ----------
def make_activation(n_tokens=4096, d=512, n_outlier=16):
    """不对称离群通道：大的正均值(直流分量)+小幅波动,形状跨 token 稳定"""
    X = rng.normal(0, 1.0, (n_tokens, d))
    oc = rng.choice(d, n_outlier, replace=False)
    mus = rng.uniform(20.0, 50.0, n_outlier)      # 每个离群通道自己的偏移量
    sigmas = rng.uniform(0.4, 1.0, n_outlier)     # 波动反而很小
    for k, j in enumerate(oc):
        X[:, j] = mus[k] + sigmas[k] * rng.normal(0, 1.0, n_tokens)
    return X, oc, mus, sigmas

X, oc, mus, sigmas = make_activation()
W2 = rng.normal(0, 0.02, (256, X.shape[1]))
Y_ref2 = X @ W2.T

mu = X.mean(0)                       # 校准集通道均值 → shift 向量
s_smooth = np.abs(X).max(0)          # SmoothQuant 口径的逐通道 scale
s_os = np.abs(X - mu).max(0)         # OS+ 口径：先去均值再取逐通道 absmax
print("[DemoB] 激活侧 W8A8（per-tensor int8）三种处理对比")
print(f"  原始激活  per-tensor absmax = {np.abs(X).max():6.2f}")
print(f"  shift 后  per-tensor absmax = {np.abs(X-mu).max():6.2f}"
      f"   （去均值使动态范围缩小 {np.abs(X).max()/np.abs(X-mu).max():.1f}x）")

const = mu @ W2.T                    # shift 的常数补偿项（迁移进 LN β 的对应物）
Y_direct = per_tensor_sym_quant(X) @ W2.T
Y_scale  = per_tensor_sym_quant(X / s_smooth) @ (W2 * s_smooth[None, :]).T
Y_os     = per_tensor_sym_quant((X - mu) / s_os) @ (W2 * s_os[None, :]).T + const
for name, Y in [("直接量化           ", Y_direct),
                ("仅 scale(Smooth型) ", Y_scale),
                ("shift+scale(OS+)   ", Y_os)]:
    print(f"  {name}: 输出相对误差 = {rel_err(Y, Y_ref2):.4f}")

# 附：离群通道波动项拿到的有效级数（σ ÷ 该通道在原始单位下的网格步长）
j = oc[np.argmax(mus)]; k = list(oc).index(j)
step_direct = np.abs(X).max() / 127        # 直接量化：全局 absmax 定步长
step_smooth = s_smooth[j] / 127            # 仅 scale：该通道自己的 absmax 决定其步长
step_os     = s_os[j] / 127                # shift+scale：去均值后 scale 大幅缩小
print(f"  离群通道 {j} (mu={mu[j]:.1f}, 波动σ≈{sigmas[k]:.2f}) 的波动项拿到的 int8 有效级数:")
print(f"    直接量化    : {sigmas[k]/step_direct:6.1f} 级   (该通道步长 {step_direct:.4f})")
print(f"    仅 scale    : {sigmas[k]/step_smooth:6.1f} 级   (步长几乎不变! scale 救的是其他通道)")
print(f"    shift+scale : {sigmas[k]/step_os:6.1f} 级   (去均值后步长缩小 {step_smooth/step_os:.0f}x)")
```

**运行结果解读**(实测输出,随机种子固定可复现):

* **Demo A**($\gamma$ 在 3 个离群通道上取 5 倍):LN 输出的 per-tensor absmax 被放大 5.0 倍(29.99 → 149.86);直接量化 LN 输出的层输出相对误差 **0.1363**,Gamma Migration 后降到 **0.0291**(--78.6%)。注意迁移后权重的对应列被放大了同样倍数,但权重走 per-channel 量化,吃得下这种列间悬殊--这正是"不对称性"被搬到了它能被便宜处理的一侧;
* **Demo B**(16/512 个通道为"大均值 + 小波动"形态):三种处理的层输出相对误差依次为 **0.0179 → 0.0026 → 0.0013**。最有信息量的是附注那三行:仅 scale 时,离群通道自身的有效级数与不处理**完全相同**(1.9 级 vs 1.9 级,因为步长由它自己的 absmax 决定)--scale 救的是其余 496 个正常通道;shift 把该通道步长缩小 18 倍,波动项拿到 36 级,这才是离群通道自身的救赎。这也解释了为什么 OS+ 的增益集中体现在离群占比高的设置里,而 SmoothQuant 在"离群少而弱"的场景已经够用。

> 说明:Demo B 中 shift 的常数补偿项 `const = mu @ W.T` 对应 OS+ 里迁入 LN $\beta$(或线性层 bias)的那部分;为了隔离激活侧的效果,两个 Demo 的权重都保持 fp16。合成分布的"偏移 + 小波动"形态是对 OS+ 论文观察的教学化还原,真实 LLM 各层偏移量与占比不同,结论方向一致、幅度浮动。

## 6. 批判与展望

**批判**:

1. **shift/scale 都是静态的**。$(\delta, s)$ 从校准集估计后冻结,对分布漂移(长文本、多语言、代码等域外输入)没有自适应能力;ZeroQuant 式的动态量化是另一个极端(在线统计、零校准),两者之间没有免费午餐;
2. **等效变换的层数受限**。迁移依赖"LN 输出只喂同分支线性层"的结构假设,遇到更复杂的拓扑(交叉注意力、共享 LN、MoE 的 router)要逐一重新论证等价性,工程泛化成本被论文轻描淡写;
3. **增益的天花板**。shift+scale 只能整形一阶统计(位置与幅度),对"通道间相关性"无能为力--这正是旋转路线(第 12 篇 QuaRot/SpinQuant)的出发点:哈达玛变换把离群能量摊到所有通道,处理的是协方差结构。可以说 OS+ 把"仿射整形"做到了头,再往前就必须换工具;
4. **工程影响力不及 SmoothQuant**。原因不在效果而在时机与简单性:SmoothQuant 早半年、名字响、有 torch-int 参考 实现,而部署侧随后又被 FP8 格式截胡--OS+ 的 shift 思想更多是以"被继承"的方式活在后续工作(OmniQuant 的 learnable shift/scale)里的。

**展望**:仿射整形的下一个台阶是把 $(\delta,s)$ 从"校准统计"升级为"优化变量"(OmniQuant 已做),再往上是用数据驱动的正交变换替代固定函数族(Rotation/SpinQuant)。而 OS 家族留下的最持久资产是那条**纪律**:任何分布改造都必须附带一张迁移表,否则就不是免费的。今天所有声称"training-free"的激活量化方案,都还在遵守它。

## 7. 常见问题 FAQ

**Q1:Gamma Migration 和 SmoothQuant 的 scale 迁移有什么区别?**
对象不同:$\gamma$ 迁移搬的是**训练出来的 LN 参数**(一次性、无需校准),SmoothQuant 搬的是**校准集统计出的 scale**(需要数据、需要调 $\alpha$)。前者拆除放大器,后者均衡剩余分布;实践中可以串联(先迁移 $\gamma$,再做平滑),OS 论文正是这样组合的。

**Q2:shift 减掉的均值,信息去哪了?会不会丢信息?**
不丢。$x-\delta$ 丢掉的常数被 $b' = b + \delta W^\top$ 或 $\beta'=\beta-\delta$ 完整接住,浮点输出严格不变。丢的只是量化器的"负担":常数不需要分辨率,波动才需要。

**Q3:为什么不直接对激活做 per-channel 量化,而要绕这么大圈子?**
per-channel 动态 scale 的 GEMM 要为每个输入通道维护 scale 并在 kernel 内做缩放累加,Tensor Core 的整齐流水线被打断(第 10 篇详述过);而 OS+ 的变换是**离线一次性**重写参数,运行时仍是干净的 per-tensor W8A8 GEMM。绕圈是为了把在线成本降到零。

**Q4:这套方法和 KV cache 量化有什么关系?**
KV cache 的 per-channel 统计同样呈现"稳定偏移 + 重尾"形态,OS+ 的 shift+scale 可以原样用于 K/V 的量化预处理(QServe 系(第 16 篇)的 KV4 处理里能看到同款思想:非对称 zero-point 本质就是一个内置的 shift)。

## 8. 参考清单

| 论文/工具 | 链接 |
|:---|:---|
| Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models | [arXiv:2209.13325](https://arxiv.org/abs/2209.13325) · [GitHub](https://github.com/wimh966/outlier_suppression) |
| Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling | [arXiv:2304.09145](https://arxiv.org/abs/2304.09145) · [GitHub](https://github.com/ModelTC/Outlier_Suppression_Plus) |
| SmoothQuant(对比基线,第 10 篇) | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) |
| LLM.int8()(离群值现象的开山作,第 E1 篇) | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |
| OmniQuant(把 $(\delta,s)$ 变成可学习参数,第 05 篇) | [arXiv:2308.13137](https://arxiv.org/abs/2308.13137) |
| 本文配套实验 | 配套实验脚本（纯 numpy 实现，种子固定可复现） |

> **下一篇**:[RPTQ/QUIK/ATOM](/2026/08/24/ptq-11-rptq-quik-atom/)--当 scale 和 shift 都压不住离群值,W4A4 时代的三板斧:通道聚类重排、混合精度 GEMM、以及把 reorder 融进 LayerNorm 的工程学。

