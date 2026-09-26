---
title: "超越函数在 GPU 上到底怎么算：从 MUFU.EX2 到 FlashAttention-3 的 softmax 之战"
date: 2026-09-26 19:00:00 +0800
categories:
  - 算子开发
  - 芯片架构
tags: [transcendental, mufu, sfu, softmax, flashattention, rope, numerical, minimax, blackwell]
layout: post
mathjax: true
---

> **本篇是[《算子开发与优化（10）：超越函数，GPU / NPU 是怎么算 exp、log、sin 的》](/2026/09/03/op-10-transcendental-math/)的纵深续篇。**
>
> 第 10 篇做的是**横向**扫描：NVIDIA / AMD / Intel / 昇腾 / TPU 各家用什么单元、吞吐多少。本篇做**纵向**深挖：只盯 NVIDIA 一家，把「硬件到底怎么算」和「LLM 为什么会撞上这个瓶颈」这两件事打通，并且给出可以复现的数值实验。文末附第 10 篇的数据勘误。

**TL;DR**

> * **一个数字定生死**：H100 上 FP16 Tensor Core 是 989 TFLOPS，而负责 `exp` 的 SFU 只有 3.9 T 次/秒 —— 差 **256 倍**。head dim 128 时，exp 花掉的时间约等于 matmul 的 **50%**；换 FP8 就变成 **100%**。这是 attention 里所有非矩阵运算开销的物理根源。
> * **硬件不「算」超越函数，只逼近**：唯一范式是**归约 → 逼近 → 重建**。`exp2`/`log2` 之所以是原语，是因为它们的归约和重建能被 IEEE-754 的指数域**白嫖**，误差 100% 集中在中间那条多项式上。
> * **为什么是 minimax 不是 Taylor**：实跑数据 —— 在 $$[0,1)$$ 上逼近 $$2^f$$ 达到 `MUFU.EX2` 的 $$2^{-22}$$ 精度，Taylor 要 **8 阶**，minimax 只要 **5 阶**，而「32 段查表 + 2 阶多项式」同样达标却只需两次乘法。后者正是 SFU 的选择。
> * **softmax 是 exp2 的完美用例**：max-subtraction 让 ex2 的参数归约被算法**免费完成** —— 参数恒为负、小数部分恒落在 $$[0,1)$$、结果恒 $$\le 1$$。这不是巧合，是 softmax 与硬件的双向奔赴。
> * **英伟达的答案不是加速 exp，而是藏起 exp**：FlashAttention-3 用 warp specialization + 二级流水线把 `MUFU.EX2` 塞进 `WGMMA` 的影子里（570 → 660 TFLOPS）。Blackwell 的实测是 100% co-issue，softmax 的开销被完全重叠。
> * **最大的坑在 RoPE**：`__sinf` 的归约误差随缠绕数**线性增长**，position 1e6 时高频通道相位偏 1.7e-2 rad，1e7 时偏 0.71 rad（彻底崩）。解法是预计算表 —— 顺带把 SFU 负载换成访存。

---

## 1. 起点：一条 256 倍的吞吐悬崖

先看清楚问题有多大。CUDA C 编程指南给的特殊函数吞吐是**每 SM 每周期 16 次操作**，H100 SXM5 有 132 个 SM、算 FP16 峰值用的时钟是 1.83 GHz：

$$
16 \times 132 \times 1.83\times 10^{9} \approx 3.86 \times 10^{12}\ \text{次/秒}
$$

同一块卡上，FP16 Tensor Core 是 989 TFLOPS、FP8 是 1978 TFLOPS。三个数字摆在一起：

| 执行单元 | 吞吐 | 相对 SFU |
|---|---|---|
| FP8 Tensor Core | 1978 TFLOPS | 512×（按 FLOPS 口径） |
| FP16 Tensor Core | 989 TFLOPS | **256×** |
| FP32 CUDA Core | 30.9 T ops/s | 8× |
| **SFU / MUFU** | **3.86 T ops/s** | **1×** |

FlashAttention-3 论文里的「256 倍」用的就是上表第二行的口径（989 TFLOPS ÷ 3.9 TFLOP-equivalent）。注意这个口径下的「TFLOPS」对 SFU 而言实际是「每秒多少次操作」，量纲不完全等价，作为吞吐比值是合理的，但不要当成 FLOPS 去参与算力加总。

真正有决策价值的是**每个 attention score 元素的时间预算**。设 head dim 为 $$d$$，每个 score 元素对应两个 GEMM（$$QK^{\top}$$ 和 $$PV$$）共 $$4d$$ 个 FLOP，以及 **1 次 exp**：

```text
Lab 1  H100 throughput cliff: Tensor Core vs SFU
==========================================================================
SFU (MUFU)          :     3.86 T ops/s   (16 ops/clk/SM)
FP32 ALU            :    30.92 T ops/s   (128 ops/clk/SM)
FP16 Tensor Core    :   989.00 TFLOPS
FP8  Tensor Core    :  1978.00 TFLOPS

SFU is 1/8 of the FP32 ALU rate
FP16 MAC vs SFU     :    127.9 x
FP8  MAC vs SFU     :    255.9 x

Per-score-element budget, head dim d = 128:
  d= 64  FP16  matmul  256 FLOP/elem | exp/matmul = 100.0% | exp/(exp+matmul) =  50.0%
  d= 64  FP8  matmul  256 FLOP/elem | exp/matmul = 199.9% | exp/(exp+matmul) =  66.7%
  d=128  FP16  matmul  512 FLOP/elem | exp/matmul =  50.0% | exp/(exp+matmul) =  33.3%
  d=128  FP8  matmul  512 FLOP/elem | exp/matmul = 100.0% | exp/(exp+matmul) =  50.0%
  d=256  FP16  matmul 1024 FLOP/elem | exp/matmul =  25.0% | exp/(exp+matmul) =  20.0%
  d=256  FP8  matmul 1024 FLOP/elem | exp/matmul =  50.0% | exp/(exp+matmul) =  33.3%

Add the FP32 ALU work softmax also needs (max reduce, sum, rescale,
F2FP cast): ~5 FP32 ops per element at 1/8 the SFU cost each.
  FP16 d=128 non-matmul share of (matmul+non-matmul) = 44.8%
```

几件事立刻清楚了：

1. **$$d=128$$ + FP16 时，exp 花的时间正好是 matmul 的一半**（50.0%）——这就是 FlashAttention-3 论文说的那个「50%」。
2. **$$d=64$$ + FP8 时，exp 占总时间 66.7%** —— 这也是论文里「FP8 下更糟」的来源。
3. **真正的 softmax 不止 exp**。还带约 5 条 FP32 ALU 指令（取 max、累加、rescale、F2FP 转换）。按 FP32 是 SFU 的 8 倍速折算，这部分又贡献 $$5/8$$ 个「SFU 单位」，于是 $$d=128$$ FP16 的非矩阵开销占总时间 **44.8%**。

最后一行才是工程现实：**你以为你在优化 exp，其实你在优化一整条 non-matmul 指令流**。

---

## 2. 通式：归约 → 逼近 → 重建

所有超越函数的硬件实现都是同一套三段式，没有例外。

```mermaid
graph LR
    A["输入 x"] --> B["① 归约<br>argument reduction"]
    B --> C["② 逼近<br>minimax poly / LUT"]
    C --> D["③ 重建<br>reconstruction"]
    D --> E["输出"]

    style B fill:#E6F1FB
    style C fill:#FAEEDA
    style D fill:#E6F1FB
```

### 2.1 为什么 exp2 和 log2 是「原语」

以 $$2^x$$ 为例。把 $$x$$ 拆成整数部分和小数部分：

$$
y = x,\quad n = \lfloor y \rfloor,\quad f = y - n \in [0, 1)
$$

$$
2^{y} = 2^{n} \cdot 2^{f}
$$

关键在于——**$$2^n$$ 不用算**。浮点数本身就是 $$1.m \times 2^{e}$$ 的形式，乘 $$2^n$$ 就是往指数位加 $$n$$，一条位操作，误差为零。于是整条指令的误差 100% 来自 $$2^f$$ 那条多项式，而 $$f$$ 被限制在 $$[0,1)$$ 这个极窄、极温和的区间里。

`log2` 完全对称：

$$
x = 2^{e} \cdot m,\ m \in [1,2) \quad\Longrightarrow\quad \log_2 x = e + \log_2 m
$$

$$e$$ 从位域直接抠出来，只需逼近 $$[1,2)$$ 上的 $$\log_2 m$$。

**这解释了 GPU 上最重要的一个事实：底数 2 的版本才是原语，自然指数/对数是二等公民。**

$$
e^{x} = 2^{x \log_2 e},\qquad \ln x = \ln 2 \cdot \log_2 x
$$

所以 CUDA 的 `__expf(x)` 编译出来就是两条指令：

```text
FMUL     R0, R0, 1.442695       ; x * log2(e)
MUFU.EX2 R0, R0                 ; 2^(x*log2e) = e^x
```

而 `expf(x)`（libdevice 的多项式版本，带完整归约和多轮修正）是几十条。softmax 里那个 $$1/\sqrt{d}$$ 缩放可以顺手把 $$\log_2 e$$ 也吸收进去——换底是**免费**的。

### 2.2 另外三条路线

| 路线 | 用在 | 说明 |
|---|---|---|
| 多项式 / 有理逼近 | `exp2` `log2` `sin` `cos` | minimax（Remez），**不是 Taylor** |
| 查表 + 低阶插值 | SFU 的实际实现 | 见 §3 |
| Newton-Raphson / Goldschmidt | `rcp` `rsqrt` `sqrt` 除法 | 硬件只给种子，精修靠迭代 |
| CORDIC | **GPU 上不用** | 移位+加法迭代，适合无乘法器的 FPGA/ASIC |

顺带埋掉一个浪漫的误会：Quake III 那个 `0x5F3759DF` 快速逆平方根，是「没有 `rsqrt` 指令的年代」的产物。现在一条 `MUFU.RSQ` 就结束了，别再往 kernel 里抄。

---

## 3. 逼近的核心：为什么必须是 minimax

这一段用实跑数据说话。目标是在 $$[0,1)$$ 上逼近 $$2^f$$，精度基准取 `MUFU.EX2` 的 $$2^{-22} \approx 2.38\times 10^{-7}$$。

```text
Lab 2  Approximating 2^f on [0,1): minimax vs Taylor vs table+quadratic
==========================================================================
target: MUFU.EX2 class accuracy  2^-22 = 2.384e-07

scheme                               max rel err   vs 2^-22
Taylor deg 3                           5.561e-03   23324.95x
Taylor deg 4                           7.520e-04    3154.27x
Taylor deg 5                           8.536e-05     358.02x
Taylor deg 6                           8.342e-06      34.99x
Taylor deg 8                           5.465e-08       0.23x

minimax deg 2                          1.742e-03    7305.91x
minimax deg 3                          7.621e-05     319.66x
minimax deg 4                          2.642e-06      11.08x
minimax deg 5                          7.657e-08       0.32x

Hardware-style two-level scheme: LUT of S segments + degree-2 poly
  S=   8 segments x deg 2            3.388e-06      14.21x
  S=  16 segments x deg 2            4.235e-07       1.78x
  S=  32 segments x deg 2            5.293e-08       0.22x
  S=  64 segments x deg 2            6.617e-09       0.03x
  S= 128 segments x deg 2            8.271e-10       0.00x

Coefficients, minimax degree 4 for 2^f on [0,1) (Horner order):
  c0 = +1.000002540413
  c1 = +0.693004713379
  c2 = +0.241441241224
  c3 = +0.052010549410
  c4 = +0.013535874748
```

三个结论：

**（1）Taylor 是灾难。** 它在展开点附近极准、在远端极烂，最大误差全部堆在 $$f=1$$ 处。要到 $$2^{-22}$$ 需要 **8 阶**，而 minimax 只要 **5 阶** —— 差 3 个乘法器，在一个每 SM 要放几十份的单元里，这是决定性的面积差异。注意 deg 5 的 minimax 系数和 Taylor 系数已经明显不同（$$c_2$$ 的 Taylor 值是 $$(\ln2)^2/2 = 0.24023$$，minimax 给 0.24144）——它**故意**在展开点引入误差，换取区间内误差的均匀摊平（等波纹，equioscillation）。

**（2）真正的硬件选择是「32 段表 + 2 阶多项式」。** 精度 $$5.29\times 10^{-8}$$，比单条 5 阶 minimax 还好，但求值只需要 2 次乘法 + 1 次 FMA，外加一张 32 项的表。表在 SRAM/ROM 里，面积远小于 3 个乘法器 + 3 个加法器。**这就是 SFU 的真实形态**，也正是 Oberman & Siu 在 [ARITH'05 *A high-performance area-efficient multifunction interpolator*](https://doi.org/10.1109/ARITH.2005.7) 里描述的多级插值器结构。

**（3）「面积 - 精度」是可调旋钮。** 想要更高精度就加段数（表变大），想要更省面积就降段数。Oberman & Siu 那篇论文的核心贡献就是**一份硬件复用出多个函数**：`EX2`/`LG2`/`RCP`/`RSQ`/`SIN`/`COS` 共享同一套表 + 乘加阵列，只换系数 ROM。这才是 SFU「面积小」的根本原因。

---

## 4. NVIDIA 的 MUFU 全景

每个 SM 里有一组 **SFU（Special Function Unit）**，独立于 FP32 / INT32 / Tensor Core 流水线，执行 **MUFU**（Multi-Function Unit）指令。每个 SM 子分区（SMSP）一个，四个 SMSP 共用一个 SM。

### 4.1 指令与精度

| SASS | PTX | 语义 | 精度 | 引入 |
|---|---|---|---|---|
| `MUFU.EX2` | `ex2.approx.f32` | $$2^x$$ | ~2 ulp | 全架构 |
| `MUFU.LG2` | `lg2.approx.f32` | $$\log_2 x$$ | 尾数绝对误差 $$2^{-22.6}$$ | 全架构 |
| `MUFU.RCP` | `rcp.approx.f32` | $$1/x$$ | ~1 ulp（约 23 bit） | 全架构 |
| `MUFU.RSQ` | `rsqrt.approx.f32` | $$1/\sqrt{x}$$ | 相对误差 $$2^{-22.9}$$ | 全架构 |
| `MUFU.SIN` / `COS` | `sin/cos.approx.f32` | 约 22 bit | 全架构 |
| `MUFU.TANH` | `tanh.approx.f32` | $$\tanh$$ | 约 22 bit | **sm_75+** |
| `MUFU.RCP64H` / `RSQ64H` | — | FP64 高精度种子 | sm_80+ |
| `MUFU.SQRT` | `sqrt.approx` | $$\sqrt{x}$$ | sm_89+ |

（精度数字出自 [PTX ISA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/) 与对 `ptxas` 生成行为的实测；不同架构略有出入，以官方文档为准。）

注意没有 `MUFU.EXP` / `MUFU.LOG`——因为底数 2 的版本加一条 FMA 就够了，没必要占面积。`MUFU.TANH` 从 Turing 才加入，明摆着是为 GELU 准备的。

### 4.2 关键事实一：MUFU 只给种子，IEEE 靠软件迭代

这是最容易踩的认知断层。`MUFU.RCP` 一条指令出的是 **~23 bit** 的近似值，不是 IEEE 正确舍入。要正确舍入就得 Newton-Raphson 精修。实跑一下收敛速度：

```text
Lab 3  Newton-Raphson refinement from a 23-bit MUFU.RSQ seed
==========================================================================
seed error injected: 2^-23 relative (MUFU.RSQ class, ~1 ulp)
  seed            : max rel err = 1.192e-07  (-23.0 bits)
  + Newton iter 1 : max rel err = 2.154e-14  (-45.4 bits)   > FP32
  + Newton iter 2 : max rel err = 3.331e-16  (-51.4 bits)   > FP32
  + Newton iter 3 : max rel err = 2.220e-16  (-52.0 bits)   > FP32

Same for reciprocal (MUFU.RCP -> division):
  seed            : max rel err = 1.192e-07
  + Newton iter 1 : max rel err = 1.443e-14
  + Newton iter 2 : max rel err = 2.220e-16

Instruction cost on NVIDIA hardware:
  rsqrt.approx.f32 : 1 MUFU                      -> ~23 bit
  div.full.f32     : MUFU.RCP + 1 Newton (FFMA)  -> IEEE FP32
  div.rn.f64       : template with 2 Newton      -> ~100 SASS insns
```

Newton 是**二次收敛**的：误差自乘。23 bit → 一次迭代到 45 bit（远超 FP32 的 24 bit）→ 两次到 51 bit（逼近 FP64 的 53 bit）。所以：

- FP32 的 IEEE 除法（`div.full.f32`）只需要**一次**迭代；
- FP64 的 IEEE 除法（`div.rn.f64`）需要**两次**，加上各种边界处理，展开成上百条 SASS。

**工程含义**：`x / y` 和 `1.0f / sqrtf(x)` 在默认编译选项下是「贵」的。LLM kernel 里清一色用 `__frcp_rn` / `rsqrt` 近似版，或者开 `--prec-div=false --prec-sqrt=false`，代价是 1 ulp 级误差——对 softmax 和 RMSNorm 完全无所谓。

### 4.3 关键事实二：现代架构上连 MUFU 都被包了一层

硬件 MUFU 的有效输入域有限（大致 $$[0.5, 4)$$）。所以 `ptxas` 生成 `rcp.approx.f32` 时会插入 **range-reduction wrapper**：subnormal 测试、overflow 测试、两次 FSEL 选缩放因子、pre-scale、`MUFU`、post-scale —— 一共 7 条 SASS。

延迟因此分成三档（sm_120 实测）：

| 档位 | 指令 | 有效延迟 | 原因 |
|---|---|---|---|
| 40–44 cyc | `rcp` `rsqrt` `lg2` | 最贵 | wrapper 指令最多 |
| 18–22 cyc | `sin` `cos` | 中 | 三角归约 |
| ~18 cyc | `ex2` `tanh` | 最便宜 | 输出域有界，wrapper 最少 |

**`ex2` 是最便宜的那一个**——又一次说明 softmax 走 exp2 路线在硬件上是被优待的。

### 4.4 关键事实三：`MUFU.SIN/COS` 的 2π 语义

据 SASS 层分析，`MUFU.SIN/COS` 的硬件语义其实是 $$\sin(2\pi x)$$ ——**输入单位是「圈数」，不是弧度**。π 归约被推给了编译器和软件。这一条直接决定了 §7 里 RoPE 的长上下文陷阱。

---

## 5. softmax：为 exp2 量身定做的完美用例

现在把硬件和算法接上。softmax 的标准写法是：

$$
p_i = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}},\quad m = \max_j s_j
$$

那个减 $$m$$ 通常被教成「防止溢出」。它其实还做了一件更重要的事：**把 ex2 的参数归约免费做完了**。

```text
Lab 4  Softmax numerics: exp2 route, max-subtraction, FP16 overflow
==========================================================================
logits: 4096 elements, mean -0.09, std 5.94, max +19.16, min -21.97

route A  FP32 libm exp        max abs err vs FP64 = 9.298e-08
route B  FP32 exp2 + 2^-22    max abs err vs FP64 = 1.299e-07
route C  no max-subtraction   -> 139/4096 elements overflow FP16 (max 65504)

Why max-subtraction makes exp2 exact-friendly:
  after max-subtraction, ex2 argument range = [-41.13, 0.00]
  integer parts n = floor(arg) all <= 0 -> result stays <= 1
  fractional parts f land in [0,1) -> exactly where MUFU.EX2 is fitted
  fraction of arguments with f in [0,1): 100.0%
```

三个收益叠在一起：

1. **数值稳定**：参数恒 $$\le 0$$，$$n = \lfloor y\rfloor$$ 恒 $$\le 0$$，结果恒 $$\le 1$$，累加不可能溢出。实跑里 3.4% 的元素在不减 max 时会冲爆 FP16（65504）。
2. **精度最优**：小数部分 100% 落在 $$[0,1)$$，正是那条 minimax 多项式被拟合的区间——硬件精度最高的地方。
3. **归约免费**：不需要额外的 range-reduction 指令。

而且从 Lab 4 看，**route B（exp2 + $$2^{-22}$$ 扰动）的误差 1.30e-7 与 route A（libm exp）的 9.30e-8 几乎在同一量级**。也就是说：用快 10 倍的 `MUFU.EX2` 换来的精度损失，对 softmax 而言完全可以忽略。cuDNN 和 FlashAttention 内部敢用 `__expf`，是有数值依据的。

### 5.1 online softmax：一次 exp2，不是两次

FlashAttention 的 online softmax 常被说成「省内存」，但它在**指令层面**也省了东西。朴素分块实现要对每个元素做两轮（先算 exp 求分母，再归一化）；online 版本把它压成一轮：

$$
O_{\text{cur}} \leftarrow O_{\text{cur}} \cdot 2^{\,m_{\text{old}} - m_{\text{new}}} + 2^{\,s_i - m_{\text{new}}} \cdot V_i
$$

重点在 $$2^{\,m_{\text{old}} - m_{\text{new}}}$$ ——这是**每块一个标量**，不是每元素一个。第二遍扫描整个 score 矩阵的操作凭空消失了。

---

## 6. Hopper / Blackwell 的答案：既然干不掉，就藏起来

`exp` 慢是物理事实——SFU 就那么大，塞不下更多。NVIDIA 的思路是：**它慢没关系，只要 Tensor Core 不空转就行**。

FlashAttention-3（[arXiv:2407.08608](https://arxiv.org/abs/2407.08608)，NeurIPS 2024）用三招做到这一点：

| 招式 | 机制 | FP16 前向吞吐 |
|---|---|---|
| baseline（换 WGMMA 指令） | FA2 算法 + Hopper 原生指令 | ~570 TFLOPS |
| + warp specialization / pingpong | 两个 consumer warpgroup 交替：一个跑 GEMM，另一个跑 softmax，`bar.sync` 协调 | ~620 TFLOPS |
| + 二级 GEMM-softmax 流水线 | 单 warpgroup 内跨迭代流水：发射 `WGMMA(Q,K_{j+1})` 后**不等**，立刻对 $$S_j$$ 做 softmax | **640–660 TFLOPS** |
| FP8 | 叠加分块量化 + Hadamard 非相干处理 | ~1.2 PFLOPS |

第二招和第三招的灵魂都是**WGMMA 的异步性**：发射矩阵乘指令后不需要等结果，Tensor Core 在后台跑，CUDA core 可以同时做 softmax。等到真正需要 GEMM 输出时再显式 wait。

这不是纸上推演 —— FA3 团队反汇编了 SASS 来确认，`HGMMA` 指令之间确实穿插着 `MUFU.EX2` 和 `F2FP`：

```text
// softmax 的 row_max 和 exp2
FMNMX.FTZ  R0, R24, R6, !PT ;
MUFU.EX2   R185, R184 ;
...
// 第一个 WGMMA，和 softmax 的 exp2 / F2FP 交织
WARPGROUP.ARRIVE ;
HGMMA.64x192x16.F32 R24, gdesc[UR44], RZ, !UPT ;
F2FP.F16.F32.PACK_AB R214, R214, R187 ;
MUFU.EX2   R234, R5 ;
...
// 第二个 WGMMA 独立执行
WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
```

到了 Blackwell，对 sm_120 的微基准结论更干脆：**`MUFU.EX2` 与 Tensor Core 操作可以 100% co-issue**，FlashAttention 里 softmax 那约 25% 的墙钟时间被完全重叠，净增延迟约为零。

**这是本文最重要的一句结论：现代 NVIDIA 架构解决超越函数瓶颈的方式，不是让它更快，而是让它和矩阵乘并发。**

由此也得到一个调优判据：**如果你的 attention kernel 里 SFU 利用率高、Tensor Core 利用率低，问题不在 exp 太慢，而在你没把它们重叠起来。**

---

## 7. RoPE：sin/cos 的精度陷阱

这是本篇最实用的一节，也是和第 10 篇差异最大的部分。

RoPE 要算 $$\cos(p\theta_i)$$ 和 $$\sin(p\theta_i)$$，其中 $$p$$ 是位置索引、$$\theta_i = \text{base}^{-2i/d}$$。问题在 §4.4 埋的雷：**`MUFU.SIN/COS` 不负责 π 归约**。用 FP32 的 $$2\pi$$ 做归约会发生什么？

$$
\tilde{r} = x - k \cdot \widetilde{2\pi},\quad k = \left\lfloor \frac{x}{2\pi} \right\rfloor
$$

误差是 $$\lvert k \rvert \cdot \lvert 2\pi - \widetilde{2\pi} \rvert$$——**随缠绕数 $$k$$ 线性增长**。

```text
Lab 5  RoPE phase precision at long context
==========================================================================
2*pi in FP32   : 6.283185482025
2*pi true      : 6.283185307180
abs error      : 1.748e-07 (= 2.78e-08 relative)

sinf-style reduction: r = x - round(x / 2pi_f32) * 2pi_f32, all in FP32
  position      theta     x=p*theta  phase err (rad)     cos err
      1000      1e+00       1000.00        2.542e-05   2.102e-05
      1000      1e-02         10.00       -3.497e-07   1.902e-07
     10000      1e+00      10000.00       -4.566e-05   1.395e-05
     10000      1e-02        100.00       -2.798e-06   1.417e-06
    100000      1e+00     100000.00        4.274e-03   1.619e-04
    100000      1e-02       1000.00        2.542e-05   2.102e-05
   1000000      1e+00    1000000.00       -1.744e-02   6.245e-03
   1000000      1e-02      10000.00       -4.566e-05   1.395e-05
  10000000      1e+00   10000000.00        7.075e-01   4.911e-01
  10000000      1e-02     100000.00        4.274e-03   1.619e-04

Second failure mode: storing the table in bf16 erases neighbours.
  p=1e+05 low freq  theta=1.155e-04  cos(p)-cos(p+1) = 9.831e-05  bf16 ulp ~ 3.91e-03  -> ERASED
  p=1e+05 high freq theta=1.000e+00  cos(p)-cos(p+1) = 4.293e-01  bf16 ulp ~ 3.91e-03  -> survives
  p=1e+06 low freq  theta=1.155e-04  cos(p)-cos(p+1) = 7.961e-05  bf16 ulp ~ 3.91e-03  -> ERASED
  p=1e+06 high freq theta=1.000e+00  cos(p)-cos(p+1) = 1.361e-01  bf16 ulp ~ 3.91e-03  -> survives
```

读这张表：

- **position 1e6、最高频通道（$$\theta=1$$）**：相位偏 **1.74e-2 rad**（约 1 度），cos 值误差 6.2e-3。相对位置信息在这一通道开始失真。
- **position 1e7**：相位偏 **0.71 rad**（40 度），cos 误差 0.49 —— 这个通道彻底报废。
- 误差**只跟 $$x = p\theta$$ 的绝对值有关**，跟 $$p$$ 本身没有直接关系。所以高频通道总是先崩。

第二个失效模式完全不同：

- **低频通道（$$\theta = 1.155\times10^{-4}$$）在 bf16 下被「抹平」**。相邻位置 $$p$$ 与 $$p+1$$ 的 cos 差约 $$8\times10^{-5}$$，而 bf16 只有 8 位尾数、ulp 约 $$3.9\times10^{-3}$$ —— 差了 50 倍，直接被吞掉。这一通道**静默地停止编码位置**。
- 高频通道差值大（0.14–0.43），bf16 存得下，所以你不会看到明显的「数值爆炸」，只会看到长上下文检索莫名其妙掉点 —— 这比崩溃更难查。

MegaBeam 的技术报告（[arXiv:2505.08651](https://arxiv.org/html/2505.08651v1)）记录的就是这个现象：NIAH 基准上模型回忆长数字会丢最后一位，根因是 bf16 下大 position 索引的 RoPE 精度，把 RoPE 单独强制回 FP32 才修好。

### 7.1 工业界的解法：预计算表

vLLM / SGLang / FlashInfer / llama.cpp 的通行做法是**预计算 cos/sin 表**：用 FP32（甚至 FP64）的精确 `sinf`（libdevice 里带完整 Payne-Hanek 归约）生成，kernel 里只做查表 + 旋转。

这一个动作同时解决了三件事：

1. **精度**：归约在 host 端或一次性用高精度算完，绕开 $$k \cdot \epsilon_{2\pi}$$ 累积项；
2. **性能**：把 SFU 负载换成了访存 —— 表通常只有 $$\text{max\_seq\_len} \times d/2$$ 个元素， resident 在 cache 里；
3. **数值一致性**：训练和推理用同一张表，避免两端归约路径不同导致的偏差。

**这是「用算法规避硬件瓶颈」的最佳范例：不要把 sin/cos 放在 token 循环里。**

---

## 8. 英伟达没做的事，以及别人做的

值得明说：**NVIDIA 至今没有独立的 softmax 硬件单元。** 理由不难理解：

- Tensor Core 是规整的 MAC 阵列，往里塞非线性单元会破坏数据流；
- softmax 的瓶颈不只是 exp，还有 max 归约、跨 lane/warp 通信、访存；
- 不同场景对 softmax 的精度要求差别极大（FP8 训练 vs FP32 推理），专用单元很难通用。

对比之下，学术界和别的厂商走得更激进：

| 方案 | 做法 | 效果 |
|---|---|---|
| [Softermax](https://doi.org/10.1145/3489517.3530486)（DAC 2021） | $$e^x \to 2^x$$，把 max 融进在线归一化 | 省掉独立归约通路 |
| [ConSmax](https://arxiv.org/abs/2402.10930)（ICCAD 2024） | 用可学习的 $$\beta, \gamma$$ **干掉** max 搜索和分母求和；bitwidth-split LUT | 16nm 下 0.2 mW / 0.0008 mm²，省 3.35× 功耗、2.75× 面积 |
| TurboAttention 的 SAS | LUT + 多项式混合逼近 + 稀疏剪枝 | 融入 FlashAttention 风格 kernel |
| Google TPU / Habana Gaudi | 独立的非线性 / 超越函数单元 | 与矩阵单元解耦 |
| 线性注意力（Mamba2 / DeltaNet / GLA / RetNet）、ReLA | **直接消灭 softmax** | 从算法侧绕过 SFU |

最后一行是 2025–2026 的真实趋势：既然非线性单元是确定性的瓶颈，那就别用需要 exp 的注意力。这与本文 §1 的算账是同一个逻辑——**当 exp 占总时间一半时，去掉 exp 就等于翻倍**。

---

## 9. 实战 checklist

### 9.1 指令选择

| 你要什么 | 用这个 | 别用这个 | 省下 |
|---|---|---|---|
| softmax 的 exp | `__expf`（FMUL + `MUFU.EX2`） | `expf` | 2 条 vs 几十条 |
| RMSNorm 归一化 | `rsqrt.approx.f32` | `1.0f/sqrtf(x)` | 1 条 vs 多轮迭代 |
| FP16 softmax | `h2exp` / `h2rsqrt`（`f16x2` 双发） | 逐元素 `expf` | 吞吐翻倍 |
| GELU | tanh 近似（FP16 走 `MUFU.TANH`） | `erf` 精确版 | 一个数量级 |
| RoPE | 预计算表 + 查表 | kernel 内 `__sinf` | 精度 + 性能双赢 |

编译选项：`-use_fast_math`，或至少 `--prec-div=false --prec-sqrt=false --ftz=true`。

### 9.2 精度陷阱

1. **FP16 下 ex2 输入超过 16 就是 inf**（$$2^{16} = 65536 > 65504$$）。减 max 后天然安全，但如果你手写了不带 max 的 softmax，必炸。
2. **denormal 会拖慢 MUFU**。加 `.ftz` 或全局 `--ftz=true`。
3. **`__sinf` 在大 $$\lvert x\rvert$$ 上精度崩**。只用于 $$\lvert x\rvert < 100$$ 的场景；RoPE 一律走表。
4. **RoPE 表不要存 bf16**。低频通道会被抹平。
5. **FP8 注意力里 softmax 仍要留在 FP32/FP16**。E4M3 只有 3 位尾数，概率分布量化会炸——这也是为什么 FP8 反而放大了 SFU 的相对占比。

### 9.3 用 Nsight Compute 定位

看这两个 metric：

```text
smsp__inst_executed_pipe_sfu.avg.pct_of_peak_sustained_active
sm__pipe_sfu_cycles_active.avg.pct_of_peak_sustained_active
```

判据很直接：

- **SFU 利用率高 + Tensor Core 利用率低** → 典型的 softmax 未被重叠，参考 §6 做 warp specialization / 流水线。
- **SFU 利用率高 + Tensor Core 利用率也高** → 已经重叠好了，别再优化 exp 了，去看访存。
- **两者都低** → 瓶颈在别处（launch 配置、occupancy、memory bound）。

---

## 10. 对第 10 篇的数据勘误

本篇写作时重新核对了数据，第 10 篇（[op-10](/2026/09/03/op-10-transcendental-math/)）有几处需要更新：

| 位置 | 原文 | 应为 | 说明 |
|---|---|---|---|
| §3.1 SFU 数量 | 「32 个」 | 每 SMSP 一个，SM 内 4 个 | 数字口径不同，32 容易与 FP32 core 数混淆 |
| §3.1 吞吐比 | 「FP32 的 1/4」 | A100 上约 1/4（FP32 64/clk/SM）；**H100 上是 1/8**（FP32 128/clk/SM） | 不同架构 FP32 速率不同，比例随之变化 |
| §3.3 `__expf` 精度 | 「约 1e-5」 | 约 $$2^{-22} \approx 2.4\times 10^{-7}$$（~2 ulp） | 原文偏保守约 100 倍；Lab 4 实测 softmax 端到端误差 1.3e-7 |
| §7 TPU | 「没有专用超越函数单元」 | TPU 有独立的非线性单元，但公开文档少 | 措辞过于绝对 |

其余内容（三步算法、AMD TALU、Intel EM、昇腾的多项式+查表）与本篇一致，可放心参照。

---

## 11. Lab Exercises

配套脚本在 [`tools/transcendental_lab.py`](https://github.com/lrypcy/lrypcy.github.io/blob/main/tools/transcendental_lab.py)，运行：

```bash
/Users/congyuan/Software/miniconda3/bin/python3 tools/transcendental_lab.py
```

**Exercise 1 — 复算吞吐悬崖。** 改 `lab1()` 里的 `d`（64/128/256）和精度，观察 exp 占比如何随 head dim 下降、随精度下降而上升。思考：为什么 $$d$$ 越大 softmax 越便宜？

**Exercise 2 — 自己做一次 Remez。** 在 `lab2()` 里把目标函数从 $$2^f$$ 换成 $$\log_2 m$$（$$m\in[1,2)$$）或 $$\tanh$$，看需要几阶、几段表。再验证等波纹性：画出残差曲线，数一数极值点是否正好是 $$\text{deg}+2$$ 个、符号是否交替。

**Exercise 3 — 牛顿迭代的边界。** 在 `lab3()` 里把种子误差从 $$2^{-23}$$ 改成 $$2^{-11}$$（模拟 FP16 种子），看需要几次迭代才能回到 FP32。结论会告诉你为什么 FP16 的除法不能只靠一次迭代。

**Exercise 4 — softmax 端到端。** 改 `lab4()` 里 logits 的 std（1 / 6 / 20），观察：溢出比例如何变化？为什么 std 越大反而越依赖 max-subtraction？

**Exercise 5 — RoPE 精度。** 在 `lab5()` 里换成你实际用的 `base`（10000 / 500000 / 25e6）和 $$d$$，算出你的模型在第几个 token 处高频通道开始失真。再试试把表存成 FP16 而不是 bf16，看是否还有通道被抹掉。

**Exercise 6（需要 GPU）— 实测 SFU 瓶颈。** 写两个 attention kernel：一个朴素串行（GEMM → wait → softmax → wait → GEMM），一个按 §6 做二级流水线。用 `ncu` 采集第 9.3 节的两个 metric，验证 SFU 与 Tensor Core 是否真的并发了。

---

## 12. 参考资料

1. S. F. Oberman, M. Y. Siu. *A High-Performance Area-Efficient Multifunction Interpolator*. ARITH'05. [doi:10.1109/ARITH.2005.7](https://doi.org/10.1109/ARITH.2005.7) —— SFU 的经典设计论文，多级表 + 插值、一份硬件复用多函数。
2. J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. NeurIPS 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) —— 256× 吞吐比、warp specialization、二级流水线、SASS 级证据。
3. NVIDIA. *PTX ISA — Floating Point Instructions*（`ex2.approx` / `lg2.approx` / `rcp.approx` / `rsqrt.approx` 的精度边界）. [docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/)
4. NVIDIA. *CUDA C++ Programming Guide — Maximize Instruction Throughput*（特殊函数吞吐 16 ops/cycle/SM）. [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
5. J. Stevens et al. *Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers*. DAC 2021. [doi:10.1145/3489517.3530486](https://doi.org/10.1145/3489517.3530486)
6. S. Liu et al. *ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters*. ICCAD 2024. [arXiv:2402.10930](https://arxiv.org/abs/2402.10930)
7. *Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing*. [arXiv:2505.08651](https://arxiv.org/html/2505.08651v1) —— bf16 RoPE 导致 NIAH 掉数字的一手记录。
8. *Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base*. [arXiv:2602.10959](https://ar5iv.labs.arxiv.org/html/2602.10959) —— RoPE base 的有限精度上界分析。
9. zartbot. *Dissecting SM_120 through Microbenchmarking*. [zartbot.github.io](https://zartbot.github.io/micro_arch/nvidia/sm_120/05_instruction_level_characterization.html) —— sm_120 上 MUFU 的延迟分档与 co-issue 实测。
10. 本博客[《算子开发与优化（10）：超越函数》](/2026/09/03/op-10-transcendental-math/) —— 横向对比各家硬件，本篇的姊妹篇。

---

*姊妹篇：[算子开发与优化（10）：超越函数，GPU / NPU 是怎么算 exp、log、sin 的](/2026/09/03/op-10-transcendental-math/) —— 本篇是它的纵深续篇。*

*相关：[算子开发与优化（07）：Attention 实战](/2026/09/03/op-07-case-attention/) · [（08）：归一化实战](/2026/09/03/op-08-case-normalization/)*
