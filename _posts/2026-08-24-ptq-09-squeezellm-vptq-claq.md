---
title: "大模型量化算法（08）：SqueezeLLM / VPTQ / CLAQ——敏感度度量与非均匀量化"
date: 2026-08-24 15:00:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, squeezellm, vptq, claq, sensitivity]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **Part 1 · Weight-only PTQ** ｜ 第 08 篇 / 共 26 篇
>
> [← 07 QuIP#/AQLM](/2026/08/24/ptq-05-quip-aqlm/) ｜ [09 SmoothQuant →](/2026/08/24/llm-quant-02-smoothquant-w8a8/)
>
> - **第 08 篇 SqueezeLLM/APTQ/VPTQ/CLAQ(本文)** -- 权重侧低比特的"度量与码本"补遗,后续姊妹篇:[第 11 篇 Outlier Suppression](/2026/08/24/ptq-10-outlier-suppression/)、[第 13 篇 RPTQ/QUIK/ATOM](/2026/08/24/ptq-11-rptq-quik-atom/)、[第 14 篇 OliVe](/2026/08/24/ptq-12-olive-abfloat/)、[第 16 篇 QoQ/QServe 与 QQQ](/2026/08/24/ptq-13-qserve-qqq/)

---

## TL;DR

1. **本篇回答两个遗留问题**。第 06 篇讲 SpQR/OWQ 时留下一个悬念:"敏感度到底该怎么度量?"--激活统计(LLM.int8/AWQ)、Hessian 对角元(OWQ)、OBC 式二阶量(SpQR)之外,SqueezeLLM 给出了第四种答案:**对角 Fisher 信息的梯度平方近似**;而 APTQ 把校准视角从线性层输出搬进了 Attention 内部,用 **attention 输出重构误差**统一校准块内全部四个投影矩阵。
2. **均匀网格不是低比特的终点**。3-bit 以下,固定间距的电平排布浪费严重;SqueezeLLM 用 **Fisher 加权 k-means** 直接优化非均匀码本,VPTQ 进一步把"逐权重标量量化"升级为"**逐向量量化**"--把 GPTQ 的二阶补偿框架套在向量量化上,在平均 ~2.x bit 下逼近有训练的 AQLM,且解码只是查表。
3. **自适应比特分配是 2-bit 的生存技能**。CLAQ 证明:与其全网统一 2-bit,不如按列的离群顺序统计把精度和离群保留额度花在刀刃上--列级 AP(自适应精度)与 OR(离群保留)正交可叠加。
4. **历史判词**:这一族方法的共同代价是"查表/稀疏/变长"带来的 kernel 复杂度。它们没有像 AWQ/GPTQ 那样成为部署默认项,但把两个事实钉进了共识:**(a) 敏感度必须进入量化目标函数;(b) 低比特下电平排布本身就是一个值得优化的对象**。

---

## 目录

1. [引言:权重侧低比特的两块拼图](#1-引言权重侧低比特的两块拼图)
2. [敏感度度量的四条谱系](#2-敏感度度量的四条谱系)
   - 2.1 [从损失函数到单权重敏感度:一阶泰勒展开](#21-从损失函数到单权重敏感度一阶泰勒展开)
   - 2.2 [对角 Fisher 信息的梯度平方近似](#22-对角-fisher-信息的梯度平方近似)
   - 2.3 [四种度量的成本与信息量对比](#23-四种度量的成本与信息量对比)
3. [SqueezeLLM:Fisher 加权非均匀量化 + 稠密稀疏分解](#3-squeezellmfisher-加权非均匀量化--稠密稀疏分解)
   - 3.1 [加权 Lloyd 迭代:非均匀码本的闭式更新](#31-加权-lloyd-迭代非均匀码本的闭式更新)
   - 3.2 [dense-and-sparse 分解与 LUT 反量化](#32-dense-and-sparse-分解与-lut-反量化)
   - 3.3 [实验结果与工程代价](#33-实验结果与工程代价)
4. [APTQ:把校准视角搬进 Attention 内部](#4-aptq把校准视角搬进-attention-内部)
5. [VPTQ:向量量化遇上二阶补偿](#5-vptq向量量化遇上二阶补偿)
   - 5.1 [30 秒向量量子课程](#51-30-秒向量量子课程)
   - 5.2 [GPTQ 框架下的 VQ 形式化](#52-gptq-框架下的-vq-形式化)
   - 5.3 [多级残差码本与初始化](#53-多级残差码本与初始化)
6. [CLAQ:列级自适应的两板斧](#6-claq列级自适应的两板斧)
7. [四方法对比与路线定位](#7-四方法对比与路线定位)
8. [代码实现(numpy):加权 Lloyd 与迷你 VQ-GPTQ](#8-代码实现numpy加权-lloyd-与迷你-vq-gptq)
9. [批判与展望](#9-批判与展望)
10. [常见问题 FAQ](#10-常见问题-faq)
11. [参考清单](#11-参考清单)

---

## 1. 引言:权重侧低比特的两块拼图

第 06 篇结束时我们说过,outlier 拆分路线(SpQR/OWQ)被码本路线(QuIP#/AQLM)和旋转路线(QuaRot/SpinQuant)取代是工程必然,但留下了两个没讲完的故事:

* **故事一:敏感度的度量学**。AWQ 用激活幅度找敏感通道,OWQ 用 Hessian 对角元找弱列,SpQR 用 OBC 二阶量找敏感元素--它们各测一面。SqueezeLLM 补上了统计学上最"正统"的一种:**Fisher 信息**,并证明它便宜到可以逐权重计算。APTQ 则指出这些度量全都只盯着线性层输出,Attention 内部的耦合被系统性低估了。
* **故事二:电平的排布学**。从第 00 篇起我们用的都是均匀网格(scale + zero-point)。但 2~3-bit 时网格只有 4~8 个点,"等距"这个先验毫无道理--分布是重尾的,误差目标又是 Hessian 加权的,为什么不让电平自己站到该站的位置?这就是**非均匀量化**(SqueezeLLM)与它的极端形态**向量量化**(VPTQ/CLAQ,也是第 07 篇 AQLM 的同门)。

一句话串联:**本篇四个算法 = 四种敏感度度量 × 三种电平排布方式的组合实验**。理解了这个组合视角,这四篇论文就塌缩成一张二维表(见 §7)。

## 2. 敏感度度量的四条谱系

### 2.1 从损失函数到单权重敏感度:一阶泰勒展开

设量化扰动 $\Delta w_j$ 加在第 $j$ 个权重上,损失变化的一阶近似:

$$
\Delta L \approx \frac{\partial L}{\partial w_j}\Delta w_j
$$

一阶项的问题是对称扰动下正负抵消,且量化误差不满足"无穷小"假设。更稳的做法是用二阶项:

$$
\Delta L \approx g_j \Delta w_j + \tfrac{1}{2} h_{jj} (\Delta w_j)^2
$$

其中 $g_j = \partial L/\partial w_j$ 是梯度,$h_{jj}$ 是 Hessian 对角元。这正是 OBD(Optimal Brain Damage, LeCun 1990)的出发点,也是第 03 篇 GPTQ 家谱(OBD→OBS→OBQ→GPTQ)的起点。问题在于 $h_{jj}$ 的精确计算需要对整个损失的二次反向传播,代价高得不可接受。

### 2.2 对角 Fisher 信息的梯度平方近似

统计学里,参数 $w_j$ 携带的样本信息量由 **Fisher 信息**刻画:

$$
F_{jj} = \mathbb{E}\left[\left(\frac{\partial}{\partial w_j}\log p(y\mid x; w)\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2}{\partial w_j^2}\log p(y\mid x; w)\right]
$$

第一个等号(平方形式)与第二个等号(负曲率形式)相等是 Fisher 信息的核心恒等式(证明见任何数理统计教材,此处直接使用)。关键观察是:

$$
-\frac{\partial^2}{\partial w_j^2}\log p(y\mid x; w)\ \approx\ \left(\frac{\partial \ell}{\partial w_j}\right)^2
$$

即 **Hessian 对角元可以用单个样本梯度的平方来近似**--这就是经验 Fisher。它只需要一次普通反向传播取梯度、再做一次逐元素平方,成本与训练一步完全相同。SqueezeLLM 在校准集上累积:

$$
\widehat{F}_{jj} = \frac{1}{N}\sum_{n=1}^{N}\left(\frac{\partial \ell_n}{\partial w_j}\right)^2
$$

并把 $\widehat F_{jj}$ 当作权重 $w_j$ 的敏感度。直觉:如果某个权重的梯度长期很大,说明损失对这个权重的局部曲率大,动它一下代价就高。

**变量映射表**:

| 数学符号 | 代码变量 | Shape | 说明 |
|:---|:---|:---:|:---|
| $w_j$ | `W.flatten()[j]` | 标量 | 展平后的单个权重 |
| $g_j^{(n)}$ | `grad[j]` | 同 W | 第 $n$ 条校准样本的反传梯度 |
| $\widehat F_{jj}$ | `sens` | 同 W | 平方均值,即对角 Fisher 近似 |
| $c_k$ | `codebook[k]` | $(K,)$ | 非均匀量化电平(码本) |
| $a_j$ | `assign` | 同 W | 每个权重归属的电平索引 |

### 2.3 四种度量的成本与信息量对比

| 度量 | 代表算法 | 计算成本 | 信息粒度 | 盲区 |
|:---|:---|:---|:---|:---|
| 激活幅度统计 | AWQ / LLM.int8() | 仅前向 | 通道级 | 只反映输入侧,忽略输出侧 |
| Hessian 对角(层内重构) | OWQ / GPTQ | 一次前向收集 $\sum \mathbf{x}\mathbf{x}^\top$ | 列级/元素级 | 层间耦合不可见 |
| OBC 二阶量(含行列结构) | SpQR | 同上 + 结构分析 | 元素级 + 结构 | 同上 |
| 对角 Fisher(全模型反传) | SqueezeLLM | 每校准样本一次完整反向 | 元素级 | 一阶近似的方差较大 |
| Attention 输出重构 | APTQ | 块级前向 + 反传 | 块内跨矩阵 | 计算比单层贵 |

> 💡 **没有免费午餐**:Fisher 近似最"全"(覆盖到最终损失),但也最吵(单样本梯度平方的方差极大,需要足够多的校准样本);Hessian 类方法最安静,却只能看见本层的输出空间。APTQ 的贡献正是把"看哪里"这个旋钮再拧一格--从层输出拧到 attention 输出。

## 3. SqueezeLLM:Fisher 加权非均匀量化 + 稠密稀疏分解

SqueezeLLM(arXiv:2306.07629,UC Berkeley)的两个组件各自独立成章:敏感度驱动的**非均匀量化**与 **dense-and-sparse 分解**。

### 3.1 加权 Lloyd 迭代:非均匀码本的闭式更新

给定 $K = 2^b$ 个电平的码本 $\mathcal{C} = \{c_1,\dots,c_K\}$ 与逐权重敏感度 $s_j$,量化目标不再是普通的 MSE,而是 Fisher 加权的重构误差:

$$
\min_{\mathcal{C},\,a}\ \sum_{j=1}^{M} s_j\left(w_j - c_{a_j}\right)^2
$$

这是经典的加权 k-means 目标,用 **Lloyd 迭代**交替求解,两步都有闭式解:

* **固定码本,最优分配**:$a_j = \arg\min_k\ (w_j - c_k)^2$--每个权重挑最近的电平;
* **固定分配,最优码本**:对 $c_k$ 求导置零,

$$
\frac{\partial}{\partial c_k}\sum_{j:\,a_j=k} s_j (w_j-c_k)^2 = -2\sum_{j:\,a_j=k} s_j (w_j-c_k) = 0
\;\Rightarrow\;
c_k^\* = \frac{\sum_{j:\,a_j=k} s_j w_j}{\sum_{j:\,a_j=k} s_j}
$$

即**新电平 = 组内权重的敏感度加权均值**。对比普通 k-means(权重系数全为 1),敏感度大的权重对电平位置的"话语权"更大,电平会向它们偏移--重尾分布 + 敏感度集中在少数大权重上时,这个偏移恰好把电平推向了最需要精度的区域。

初始化用敏感度加权的 k-means++ 或分位数初始化,迭代至收敛。**注意码本是每层甚至每组一份**,需要随权重一起存储:$K$ 个 fp16 电平相对 $M$ 个 3-bit 索引的开销可以忽略($K=8$ 时仅 $16K$ 字节 vs 每 100 万权重约 $375$KB 索引)。

### 3.2 dense-and-sparse 分解与 LUT 反量化

非均匀量化解决了"电平怎么摆",但仍有极少数权重连最近的电平都离得太远。SqueezeLLM 的处理与 SpQR 同路数但更简:

$$
W \;\approx\; \underbrace{Q_{nu}(W_{\text{dense}})}_{\text{非均匀低比特,稠密}} \;+\; \underbrace{W_{\text{sparse}}}_{\text{fp16 离群值,CSR 稀疏}}
$$

* **稠密部分**:$b$-bit 索引矩阵 + 每组一份码本,反量化就是一次 **gather**(`codebook[idx]`),对 GPU 友好;
* **稀疏部分**:敏感度超阈值或量化残差超阈值的元素以 fp16 进 CSR,配合专门的双路径 GEMV kernel(稠密分支走 int4 LUT,稀疏分支走 fp16 gather-add)。

与 SpQR 的区别在"抠谁":SpQR 按 OBC 二阶结构抠出成行成列的敏感组;SqueezeLLM 按 Fisher 标量阈值逐元素抠,且抠出来的部分不做低比特压缩、直接 fp16。

### 3.3 实验结果与工程代价

论文口径(未逐一复验,以原文为准):LLaMA-7B 上 3-bit 非均匀量化 + 稀疏分解的困惑度显著低于 RTN/GPTQ(如 WikiText2 上约 7.3 vs GPTQ 约 8~9 的量级差异,具体数值随设置浮动);4-bit 下与 AWQ 相当或略优。代价同样明确:**稀疏分支使端到端吞吐低于纯稠密的 GPTQ/AWQ kernel**,加速比落后于困惑度的领先幅度--这也是它没有成为部署主流的原因。

## 4. APTQ:把校准视角搬进 Attention 内部

APTQ(Attention-aware Post-Training Quantization,arXiv 编号 2408.11793,未验证,以标题检索为准)的出发点是一个此前所有 Hessian 类方法共有的盲区:

> GPTQ 及其变体在量化第 $i$ 个线性层时,最小化的是**该层输出**的重构误差 $\|Wx - \hat W x\|^2$。但 Transformer 的 attention 块里有四个投影($W_Q, W_K, W_V, W_O$),它们的误差会在 softmax 与乘加中**交叉放大**--单独最优不等于联合最优。

APTQ 的做法:

1. **校准目标换成 attention 输出**:对块内四个矩阵施加量化扰动后,最小化

$$
\min_{\{W_Q,W_K,W_V,W_O\}\ \text{的量化}}\quad \big\|\,\mathrm{Attn}(X; W_Q,W_K,W_V,W_O) - \mathrm{Attn}(X; \cdot_{\text{fp16}})\,\big\|^2
$$

2. **敏感度来自 attention 输出对每个权重的梯度/Hessian**:通过一次带扰动的反传即可同时拿到四个矩阵的二阶代理量,天然包含交叉项的影响;
3. **混合精度分配**:按敏感度排序,敏感权重给 4-bit、其余压到 2-bit(块内混合精度,粒度可到组级),在 W4/W2 混合下整体逼近 W4 均匀量化的精度、显存却接近 W2。

一句话评价:APTQ 没有发明新的量化器,它改的是**校准信号的采样位置**--这与第 05 篇 OmniQuant"学习量化参数"的目标函数选择异曲同工,属于同一类"目标函数工程"。

## 5. VPTQ:向量量化遇上二阶补偿

### 5.1 30 秒向量量子课程

标量量化把每个数映射到一个电平;**向量量化(VQ)把一组数(比如权重的 $d$ 个连续元素拼成的向量)整体映射到码本里的一个向量(codeword)**:

$$
\text{存储:索引 } i \in \{1,\dots,K\},\quad \text{重建:}\ \hat{\mathbf{w}} = \mathbf{c}_i,\quad \text{码本 } \mathcal{C}=\{\mathbf{c}_1,\dots,\mathbf{c}_K\}
$$

位宽账很好算:$K = 2^B$ 个码字、每组 $d$ 个元素,则**平均位宽 $B/d$ bit/权重**。$K=65536, d=8$ 时仅 2 bit/权重--这就是码本路线在 2-bit 领域碾压标量均匀量化的算术基础。代价是码本搜索(离线)与查表(在线)。

### 5.2 GPTQ 框架下的 VQ 形式化

回忆第 03 篇 GPTQ 的层目标($\mathbf{H}=2XX^\top$ 为 Hessian):

$$
\min_{\hat W}\ \mathrm{tr}\big((W-\hat W)^\top X X^\top (W-\hat W)\big)
$$

RTN 版 GPTQ 逐列做"舍入 + 补偿";VPTQ(Microsoft,arXiv:2409.17066)把这个循环里的**舍入替换为最近邻码字分配**:

```
for 每个待量化列组(按 Hessian 重要度排序):
    在当前"已补偿"的权重上,为该组向量选最近码字 c_i     ← VQ 替换 round()
    计算量化误差 δ 并按 GPTQ 公式回写到尚未量化的权重    ← 二阶补偿保持不变
```

数学上,VPTQ 把每步的子问题写成带 Hessian 加权的 VQ 分配:

$$
\min_{i}\ \big(\mathbf{w}_{col} - \mathbf{c}_i\big)^\top \mathbf{H}_{col}\ \big(\mathbf{w}_{col} - \mathbf{c}_i\big)
$$

其中 $\mathbf{H}_{col}$ 是该列对应的 Hessian 子块--**分配不再看欧氏距离,而是看马氏距离**,误差大的方向被自动避开。这是"二阶优化指导 VQ"的核心含义。

### 5.3 多级残差码本与初始化

单一码本装不下权重的动态范围,VPTQ 采用**多级残差量化**(residual VQ):

$$
\hat{\mathbf{w}} = \mathbf{c}^{(1)}_{i_1} + \mathbf{c}^{(2)}_{i_2} + \cdots + \mathbf{c}^{(R)}_{i_R}
$$

第一级码本抓主方向,第二级量化第一级的残差,依此叠加;总位宽 = 各级索引位宽之和。初始化用**重要性加权 k-means**(权重即 Hessian 对角元),并在量化过程中用 Lloyd 迭代微调码字。与 AQLM(第 07 篇)的区别:AQLM 用梯度下降端到端学码本并微调下游参数,效果更强但要训练;VPTQ 全程免训练、纯 PTQ,平均 ~2.x bit 下拿到接近的成绩,解码只需逐级查表相加。

## 6. CLAQ:列级自适应的两板斧

CLAQ(arXiv:2405.17233)面向 2~3-bit 的极端场景,主张"**预算要花在刀刃上**"。三个组件:

1. **列级 k-means 码本**:为每列(或共享组)生成自适应中心点--同 SqueezeLLM 的非均匀思想,只是作用域在列;
2. **离群顺序(outlier order)**:构造列级敏感度排序(基于范数与离群程度的统计量),回答"哪几列最碰不得";
3. **两个正交策略**:
   * **AP(Adaptive Precision)**:按排序给不同列分配不同位宽(敏感列 3~4-bit,迟钝列 2-bit);
   * **OR(Outlier Retention)**:对最敏感列内的离群值按阈值保留原始精度,其余进码本。

AP 决定"每列多少 bit",OR 决定"省下的 bit 里哪些值例外",二者作用维度不同、可叠加。消融显示 AP+OR 联合优于任一单独使用(原文口径)。它与 APTQ 的混合精度互补:APTQ 的粒度是**块内权重矩阵**,CLAQ 的粒度是**列**,后者更细但需要额外的列元数据。

## 7. 四方法对比与路线定位

```mermaid
flowchart TD
    A["权重侧低比特"] --> B{"敏感度怎么测?"}
    B -->|"激活统计"| C["AWQ 系列<br>第 05 篇"]
    B -->|"层内 Hessian"| D["GPTQ / OWQ / SpQR<br>第 2/4 篇"]
    B -->|"全模型 Fisher"| E["SqueezeLLM<br>本文"]
    B -->|"attention 输出"| F["APTQ<br>本文"]
    A --> G{"电平怎么摆?"}
    G -->|"均匀网格"| H["RTN / GPTQ<br>第 1/2 篇"]
    G -->|"标量非均匀码本"| I["SqueezeLLM / CLAQ-AP<br>本文"]
    G -->|"向量码本"| J["AQLM 第 07 篇 / VPTQ / CLAQ<br>本文"]
```

| 维度 | SqueezeLLM | APTQ | VPTQ | CLAQ |
|:---|:---|:---|:---|:---|
| 敏感度度量 | 对角 Fisher(梯度平方) | attention 输出重构 | Hessian 马氏距离 | 列离群顺序统计 |
| 电平/码本 | 标量非均匀(加权 Lloyd) | 均匀网格 + 混合精度 | 多级残差向量码本 | 列级 k-means 码本 |
| 典型位宽 | 3~4-bit | W2/W4 混合 | 平均 ~2.x-bit | 2~3-bit |
| 是否需要校准数据 | 需要(反传) | 需要(前向+反传) | 需要(少量) | 需要(少量) |
| 解码开销 | LUT gather | 普通 dequant | 逐级查表相加 | 查表 + 列元数据 |
| 工程友好度 | 中(稀疏分支拖速) | 中 | 中高(免训练) | 中 |
| 最佳定位 | 精度极限型 W3/W4 | 极低比特混合精度 | 免训练 2-bit 码本 | 2-bit 自适应兜底 |

## 8. 代码实现(numpy):加权 Lloyd 与迷你 VQ-GPTQ

两个可运行实验:**Demo A** 验证 Fisher 加权非均匀量化对重尾分布的优势;**Demo B** 实现迷你版 VQ-GPTQ,对比 RTN/均匀 GPTQ/VQ-GPTQ 在层输出空间的误差。依赖 numpy/matplotlib(matplotlib 可注释掉)。

```python
import numpy as np

rng = np.random.default_rng(0)

# ---------- 公共构件 ----------
def make_layer(M=256, N=256, n_outlier_cols=4):
    """带离群列的合成权重层与输入,返回 W, X, H"""
    W = rng.normal(0, 0.05, size=(M, N))
    W[:, :n_outlier_cols] *= 20          # 少数输入通道幅值极大(模拟 outlier channel)
    X = rng.normal(0, 1.0, size=(4096, M))
    H = 2.0 * (X.T @ X)                   # GPTQ 口径的 Hessian
    return W, X, H

def sym_uniform_quant(W, bits):
    """对称均匀量化(逐张量),返回反量化权重"""
    s = np.max(np.abs(W)) / (2 ** (bits - 1) - 1)
    return np.round(W / s).clip(-(2**(bits-1)), 2**(bits-1)-1) * s

def weighted_lloyd(w, sens, K, iters=50):
    """Fisher 加权 k-means:返回码本与分配(SqueezeLLM 核心)"""
    # 初始化:分位点
    qs = np.quantile(w, np.linspace(0, 1, K + 1)[1:-1])
    C = qs.copy()
    assign = np.zeros_like(w, dtype=int)
    for _ in range(iters):
        assign = np.argmin((w[:, None] - C[None, :]) ** 2, axis=1)      # 分配步
        for k in range(K):                                               # 更新步(闭式)
            m = assign == k
            if m.any():
                C[k] = (sens[m] * w[m]).sum() / sens[m].sum()
        order = np.argsort(C); C = C[order]; remap = np.argsort(order)
        assign = remap[assign]                                           # 保持电平有序
    return C, assign

def fisher_quant(W, bits, sens):
    """SqueezeLLM 风格:逐列非均匀量化"""
    What = np.empty_like(W); K = 2 ** bits
    for j in range(W.shape[1]):
        C, a = weighted_lloyd(W[:, j], sens[:, j], K)
        What[:, j] = C[a]
    return What

# ---------- Demo A: 加权 Lloyd vs 均匀量化 ----------
W, X, H = make_layer()
grad2 = (np.abs(W) > 0.3).astype(float) * 100 + 1.0     # 合成"梯度平方"敏感度: 大权重更敏感
for bits in [2, 3]:
    e_uni = ((X @ (W - sym_uniform_quant(W, bits)).T) ** 2).mean()
    e_fsh = ((X @ (W - fisher_quant(W, bits, grad2)).T) ** 2).mean()
    print(f"[DemoA] {bits}-bit  均匀 MSE={e_uni:.4e}  Fisher加权非均匀 MSE={e_fsh:.4e}  "
          f"降低 {(1 - e_fsh/e_uni)*100:.1f}%")
```

```python
# ---------- Demo B: 共享码本 VQ-GPTQ ----------
def kmeans_w(vecs, wts, K, iters=25):
    """加权向量 k-means: vecs (n,d), wts (n,)"""
    Cm = vecs[np.random.default_rng(1).choice(len(vecs), K, replace=False)].copy()
    for _ in range(iters):
        a = ((vecs[:, None, :] - Cm[None])**2).sum(-1).argmin(1)
        for k in range(K):
            m = a == k
            if m.any(): Cm[k] = (wts[m, None] * vecs[m]).sum(0) / wts[m].sum()
    return Cm

def vq_gptq(W, X, d=4, K=256):
    """迷你 VQ-GPTQ: 全组共享码本(有效位宽 log2(K)/d bit/权重)
       + GPTQ 式误差回写"""
    M, N = W.shape
    H = 2.0 * (X.T @ X) + 1e-6 * np.eye(N)
    Hinv = np.linalg.inv(H); hdiag = np.diag(Hinv)
    # 码本构建: 所有组的行向量(每个向量是 d 个连续权重的拼接)
    blocks   = [W[:, j0:j0+d] for j0 in range(0, N, d)]           # 每块 (M, d)
    vecs_all = np.concatenate(blocks, 0)                           # (N/d*M, d)
    wts_all  = np.concatenate(
        [np.full(M, np.abs(hdiag[j0:j0+d]).mean()) for j0 in range(0, N, d)])
    Cm = kmeans_w(vecs_all, wts_all, K)                            # Hessian 加权码本
    Wres = W.copy(); Wq = np.empty_like(W)
    for j0 in range(0, N, d):                                      # 逐组量化 + 补偿
        cols = np.arange(j0, min(j0+d, N))
        vecs = Wres[:, cols]                                       # (M, d) 每行一个待量化向量
        a    = ((vecs[:, None, :] - Cm[None])**2).sum(-1).argmin(1)
        What = Cm[a]
        delta = Wres[:, cols] - What                               # 本组量化误差
        Wq[:, cols] = What
        nxt = np.arange(cols[-1] + 1, N)
        if len(nxt):                                               # 二阶补偿: 后续列减去与本组
            Wres[:, nxt] -= delta @ (Hinv[np.ix_(cols, nxt)]       # 误差的 Hessian 相关项
                                      / hdiag[cols][:, None])
    return Wq

W, X, _ = make_layer(n_outlier_cols=6)
rtn = sym_uniform_quant(W, 2)
vq  = vq_gptq(W, X)
ref = X @ W.T
print("[DemoB] 2-bit(effective) 层输出相对误差:")
print("  RTN     :", round(float(np.linalg.norm(X @ rtn.T - ref) / np.linalg.norm(ref)), 4))
print("  VQ-GPTQ :", round(float(np.linalg.norm(X @ vq.T  - ref) / np.linalg.norm(ref)), 4))
```

**运行结果解读**（实测输出，随机种子固定可复现）：

* **Demo A**（合成敏感度与幅值强相关）：2-bit 下 Fisher 加权非均匀量化的输出 MSE 比均匀量化低 **67.2%**，3-bit 低 **77.3%**——当敏感度集中在尾部大权重时，加权 Lloyd 会把电平拉向它们，收益巨大；若把敏感度换成与幅值无关的随机数，增益会收窄到普通 k-means 的水平。结论不是“位宽越低越重要”，而是**“敏感度与幅值相关性越强，非均匀码本越值”**；
* **Demo B**（共享码本 VQ-GPTQ）：同样有效位宽 2-bit（log2(256)/4），层输出相对误差从 RTN 的约 **0.84** 降到 **0.37~0.38**（具体数值随随机状态微幅浮动，单跑与连跑 Demo A 后结果略异）——码字可以“站”到离群方向上去，而均匀网格只能靠 scale 迁就它们；再叠加 GPTQ 式误差回写，后续列在已补偿的权重上继续量化。

> 说明：Demo B 为教学简化——真实 VPTQ 的分配用的是 Hessian 子块的马氏距离而非欧氏距离，码本带多级残差级联且免训练地逐层精化；这里的欧氏版本保留了“共享码本分配 + 二阶回写”的骨架，足以展示两条机制叠加的收益来源。

## 9. 批判与展望

**批判**:

1. **敏感度的噪声问题**:经验 Fisher 是逐样本梯度平方的平均,方差大、对校准集选择敏感--同一个模型换一套校准数据,SqueezeLLM 的敏感图可能明显不同,论文未系统讨论这一点;
2. **APTQ 的块内混合精度落地难**:变长位宽意味着 kernel 要么按位宽分组执行(碎片化),要么走 LUT(开销),工程上远不如"全网统一 4-bit + AWQ scale"干净;
3. **VPTQ/CLAQ 的对比公平性**:码本方法在同等"bit 数"下额外存储了码本与元数据,某些论文按"索引位宽"报告会轻微美化压缩率;
4. **本篇四个算法都没有大规模生产部署**--它们的价值更多是把"度量"与"排布"两个设计维度推到了极限,为后来的 MXFP4/NF4(第 15 篇)提供了"非均匀性可以吃掉多少误差"的经验数值。

**展望**:敏感度度量正在与旋转/变换路线合流--QuaRot 之后的工作开始"先转再测",让 Hessian 近对角后再做非均匀分配;向量量化则朝着"码本可学习 + 硬件可查表"收敛(AQLM 微调 + LUT kernel 已是雏形)。可以预期下一个分水岭是 **2-bit 以下**:那里标量方法基本出局,竞争只在码本与残差/变换组合之间展开。

## 10. 常见问题 FAQ

**Q1:Fisher 加权 k-means 和普通 k-means 差别真的那么大吗?**
取决于敏感度与幅值的相关性。若敏感度与 $|w|$ 强相关(常见),加权会把电平拉向尾部大权重,等效于"自动学习裁剪范围";若敏感度接近均匀,两者退化相同。建议在自己的模型上先画 $s_j$-$|w_j|$ 散点图再决定要不要付这个复杂度。

**Q2:VPTQ 与 AQLM 怎么选?**
有训练资源、追求极限精度 → AQLM(端到端学码本 + 下游微调);零训练、快速出活 → VPTQ。推理侧两者都是查表,kernel 成本相近。

**Q3:这些方法和 NF4 什么关系?**
NF4(第 15 篇 QLoRA 用的 4-bit 格式)可以看作"对高斯分布预计算好的静态非均匀码本"--分位数电平,不含逐层敏感度。SqueezeLLM 是它的"逐层自适应版"。理解了本篇再看 NF4,它就是 $s_j\equiv 1$、码本全局共享的特例。

**Q4:为什么本篇方法都没进 vLLM/TensorRT-LLM?**
三个字:kernel 难。LUT gather、变长位宽、稀疏分支都会打断 Tensor Core 的整齐流水线;而 AWQ/GPTQ 的"sScale + 均匀 int4"恰好完美贴合 dequant-matmul 融合范式。精度领先 5%、速度落后 30% 的方案不会赢。

## 11. 参考清单

| 论文/工具 | 链接 |
|:---|:---|
| SqueezeLLM: Dense-and-Sparse Quantization | [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) · [GitHub](https://github.com/SqueezeAILab/SqueezeLLM) |
| APTQ: Attention-aware Post-Training Mixed-Precision Quantization | arXiv 2408.11793(编号未验证,以标题检索为准) |
| VPTQ: Extreme Low-bit Vector Post-Training Quantization | [arXiv:2409.17066](https://arxiv.org/abs/2409.17066) · [GitHub](https://github.com/microsoft/VPTQ) |
| CLAQ: Ultra-Low Bit LLM Quantization | [arXiv:2405.17233](https://arxiv.org/abs/2405.17233) |
| LeCun, OBD: Optimal Brain Damage(敏感度思想的源头,1990) | 可搜标题获取 |
| 中文社区解读:知乎《大模型量化技术原理》系列(SpQR/SqueezeLLM 各有一篇) | 知乎搜「吃果冻不吐果冻皮 量化」 |
| 本系列参照文 | 第 03 篇 GPTQ(二阶补偿框架)、第 06 篇 SpQR/OWQ/HQQ(敏感度拆分)、第 07 篇 QuIP#/AQLM(码本路线主线) |

> **下一篇**:[Outlier Suppression 与 OS+](/2026/08/24/ptq-10-outlier-suppression/)--当所有人都在"保护 outlier"时,它选择"抑制 outlier":γ 迁移、token-wise 裁剪、以及比 SmoothQuant 多走一步的 shift+scale 等效变换。

