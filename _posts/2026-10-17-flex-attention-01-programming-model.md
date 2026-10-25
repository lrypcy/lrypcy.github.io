---
title: "FlexAttention（01）：把注意力变体编译成一个 FlashAttention 内核——online softmax、score_mod 的可编程边界与 BlockMask"
date: 2026-10-17 22:40:00 +0800
categories:
  - 编译器技术
tags: [flexattention, attention, flash-attention, triton, torch-compile, torch-inductor, score-mod, mask-mod, blockmask, block-sparse, online-softmax, paged-attention, software-lottery]
layout: post
mathjax: true
excerpt: "FlexAttention 的全部内容可以压缩成一句话：把「五花八门的注意力变体」压平成「softmax 之前的一个逐点修正函数」，然后由编译器把它塞回手写 FlashAttention 模板里。本文把这句话拆开：为什么任意逐点 score_mod 能和 online softmax 的重标定循环共存、为什么 -inf 掩码必须单列成 mask_mod、BlockMask 是怎么把一份掩码压成两个索引向量的、反向传播是怎么用 AOTAutograd 的 joint graph 自动出来的、表达力边界上到底卡住了哪三类东西。配套 numpy 实验把六个变体的 online 与一次性计算比对到 1.776e-15，并把稀疏度损失、元数据字节数、页表索引转换这三笔账算成具体数字。"
---

> **系列导航** ｜ 本系列承接 [编译器知识体系深度解析](/2026/08/25/compiler-01-lexing-parsing/)（torch.compile / Triton 工程三件套落在哪一层）与 [算子开发与优化](/2026/09/03/op-01-hardware/)（单算子为什么写成那样），把两者交界处补齐：**当一个「算子的变体空间」大到手写不完的时候，编译器该怎么介入**。
> 上一篇：无（系列首篇）｜ 下一篇：02 BlockMask 与稀疏度的工程实践——数据依赖稀疏、确定性与组合爆炸的另一半（待写）

> **TL;DR**
>
> * FlexAttention 的定义只有一个式子（论文式 1）：$$\mathrm{FlexAttention}(\mathbf Q,\mathbf K,\mathbf V)=\mathrm{softmax}\big(\mathrm{mod}(\mathbf Q\mathbf K^{T}/\sqrt{d_k})\big)\mathbf V$$。[arXiv:2412.05496](https://arxiv.org/abs/2412.05496) 原文把它概括为相对原生 FlashAttention 的**两个扩展**：「对 softmax 前分数的逐点修改 + 可从全局内存任意加载」与「前向反向都支持的块稀疏迭代」（[PyTorch 官方博客，2026-03-05](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)）。
> * 这三个字「softmax 前」是全部的关键。修正是在「取最大值之前」施加的，所以 online softmax 的运行最大值 $$m_i$$ 天然量的是**修正之后**的分数，任意逐点修正都能塞进那条 $$\alpha$$-重标定循环里。配套实验把 noop / causal / ALiBi / soft-capping / 相对位置偏置 / causal+ALiBi 六种变体在「分块流式」与「一次性材料化」两条路径上比对，**最差 1.776e-15**。
> * 但 $$m_i$$ 取在哪一侧其实不是数学必须（softmax 对参考值的选择不变），而是**数值必须**：给所有分数加常数 $$+160$$，取修正前做 $$m_i$$ 会让实际求过的 $$\log_2$$ 指数顶到 230.83，超过 float32 的 $$2^{127}$$ 溢出阈值；取修正后是 0.00。差值恰好是 $$\text{bias}/\ln 2$$。
> * **mask 必须单列成 mask_mod 有两个理由**：一个是语义（靠「喂一个 0 分数进去、看它是否返回 $$-\infty$$」来反推哪些位置被掩，这种特判不可靠），一个是真金白银的性能——给每个元素施加掩码要掉 15–20%（[PyTorch 博客 FAQ](https://pytorch.org/blog/flexattention/)）。
> * **稀疏度是第二个维度**，而且它有损：causal 掩码在 N=512 / BS=64 下元素密度 50.1%，但块级密度是 56.2%——一个 128×128 的块里只要有一个元素要算，整块都得算。
> * **反向不需要用户写**：PyTorch 对 score_mod 单独跑一次 AOTAutograd 的 `create_joint`，得到 forward+backward 的合一图，再把这块 IR 同样塞进手写反向模板（`torch/_higher_order_ops/flex_attention.py:646-763`，锁定 v2.14.1）。
> * **边界上的三类不可表达**：依赖「整行统计量」的修正（流式相对误差 4.365e-02，且部分行均值要到最后一块才收敛）、作用在 softmax **之后**或分母上的修正（如 attention sink 的可学习常数项）、以及元素级（非块级）的真实稀疏跳过。
> * 2026 年这个模型还在跑，而且边界在动：2.11 起多了 **FlashAttention-4 后端**（计算受限 workload 上比 Triton 实现快 1.2–3.2×），2.13 起上了 **Apple Silicon**（最高 12×）并补齐了**确定性反向**。

配套实验：[`ipynbs` commit `5954fb5`](https://github.com/lrypcy/ipynbs/tree/5954fb521a279b997005a6f514144c3b380c8e36/experiments/flex-attention/flex_core/) 下的 `flex_core/`，纯 numpy、CPU 可跑、固定 seed `20261017`，27 条断言全过；本文所有自测数字都来自它的实际输出（[`results/summary.md`](https://github.com/lrypcy/ipynbs/blob/5954fb521a279b997005a6f514144c3b380c8e36/experiments/flex-attention/flex_core/results/summary.md)）。论文与官方博客的数字一律标注出处。

---

## 0. 符号约定

| 符号 | 含义 | 备注 |
|---:|:---|:---|
| $$\mathbf Q,\mathbf K,\mathbf V$$ | 查询 / 键 / 值 | 形状 $$(B,H,Q_{\rm LEN},D)$$ 与 $$(B,H,KV_{\rm LEN},D)$$ |
| $$B,H$$ | batch 数、注意力头数 | $$H_q$$ 与 $$H_{kv}$$ 分开写时才区分（GQA） |
| $$Q_{\rm LEN},KV_{\rm LEN}$$ | 查询序列长、键值序列长 | prefill 里相等，decode 里 $$Q_{\rm LEN}$$ 远小于 $$KV_{\rm LEN}$$ |
| $$d_k$$ / $$D$$ | query-key 的头维 | 源码里叫 `E`；`Ev` 是 value 的头维，可以与它不等 |
| $$i,j$$ | query 位置 / key-value 位置 | 源码里是 `m` 与 `n`（Triton 习惯），论文里是 q_idx / kv_idx |
| $$s_{ij}$$ | 原始分数 | $$s_{ij}=\langle q_i,k_j\rangle/\sqrt{d_k}$$ |
| $$\mathrm{mod}(\cdot)$$ | **本文的核心对象**：softmax 之前的修正 | 含 score_mod 与 mask_mod 的复合 |
| $$\tilde s_{ij}$$ | 修正后的分数 | $$\tilde s_{ij}=\mathrm{mod}(s_{ij},b,h,i,j)$$ |
| $$m_i,l_i,\mathrm{acc}_i$$ | online softmax 的三个运行量 | 运行最大值、运行分母、运行分子 |
| $$b,h$$ | batch 下标、头下标 | 在 score_mod 里它们是**张量**不是整数 |
| $$BS$$ | BlockMask 的块边长 | 默认 128（`_DEFAULT_SPARSE_BLOCK_SIZE`） |

> **一处容易混的记号**：PyTorch 源码内部把 query 位置变量写成 `m`、把 key 位置变量写成 `n`（`m` 也顺手被用成了 BLOCK_M）。本文统一用 $$i$$（query）与 $$j$$（key），只在贴源码时用回它的名字。

**本文锁定的版本**：PyTorch `v2.14.1` = commit [`5c488690`](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070)（2026-09-30 发布）；vLLM `v0.31.0` = commit [`db9527a4`](https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6)。所有行号都挂在这两个 commit 上。

---

## 1. 软件彩票：一篇关于 FlexAttention 的文章为什么要先讲清楚背景

2019 年以后，注意力这个词的含义一直在长。它不是「一个算子」，而是「一族算子」，而且这族算子的成员数量还在涨：

- **省算力**：Longformer 的滑窗注意力、BigBird 的随机+滑窗+全局混合（[Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)）
- **稳训练**：Gemma-2 的 logit soft-capping（[arXiv:2408.00118](https://arxiv.org/abs/2408.00118)）、Grok-1 同款
- **长度外推**：ALiBi 的线性位置惩罚（[Press et al., arXiv:2108.12409](https://arxiv.org/abs/2108.12409)）
- **批长短不一的序列**：文档掩码（document masking / sample packing）
- **服务吞吐**：PagedAttention 的分页 KV cache（[Kwon et al., arXiv:2309.06180](https://arxiv.org/abs/2309.06180)）
- **换到别的模态**：邻域注意力 Neighborhood Attention（把 2D 邻域当掩码）

这还不是最糟的。真正的问题是它们会**组合**：滑窗 + ALiBi、文档掩码 + 滑窗、PagedAttention + 软截断……如果每个组合都要一个手写 kernel，写上不完。

论文给了一张表衡量这个「支持矩阵」有多参差（Table 1，原文这样写）：

| 注意力变体 | FAv2 | FAv3 (c1d146c) | SDPA-cuDNN (v9.1.1) | SDPA-mem_efficient | FlexAttention |
|:--|:--|:--|:--|:--|:--|
| noop | native | native | native | native | native |
| causal | native | native | native | native | native |
| alibi_bias | native | x | itemized mask | itemized mask | native |
| sliding_win | native | native | itemized mask | itemized mask | native |
| prefix_lm | x | x | itemized mask | itemized mask | native |
| soft_cap | native | x | x | x | native |
| various lengths | native | native | jagged tensors | jagged tensors | native via document_mask |
| neighbor attention | x | x | x | x | native |

> 出处：[arXiv:2412.05496](https://arxiv.org/abs/2412.05496) Table 1（*Native* 指该库把这个变体写进了 kernel；*itemized mask* 指用户得先构造一个稠密布尔掩码再喂进去；*jagged tensors* 指用嵌套张量表示变长）。

「x」那一列之外的另一个事实是：即便退化成 itemized mask，也慢。论文 §6.2 给的数字是：在 FAv2 **不支持**的那些变体上，FlexAttention 相对「SDPA + 逐元素掩码」是 **5.49×–8.00×**；而在 FAv2 支持的变体上，官方博客的相对性能是「forward 达到 FAv2 的 90%、backward 85%」。

所以「软件彩票」（software lottery）这个词的意思是：

> 你的想法能不能被人看见，取决于有没有一个写 CUDA/Triton 的人正巧对它有空。

FlexAttention 的回答不是「再写一个更强的注意力库」，而是**换一层抽象**：让用户声明「想要什么样的注意力」，让编译器负责「怎么把它变成一个 FlashAttention」。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 340" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="340" fill="#ffffff"/>
<text x="24" y="26" font-size="15" font-weight="600" fill="#111827">两条技术路线：为每个组合写一个 kernel，还是把「变体」当成输入</text>
<text x="24" y="46" font-size="11.5" fill="#4b5563">上面是组合爆炸，下面是把「变体空间」从手写代码里搬走</text>

<text x="24" y="76" font-size="12.5" font-weight="600" fill="#b91c1c">路线 A：手写 kernel（传统）</text>
<g>
<rect x="24" y="86" width="96" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
<text x="30" y="100" font-size="10.5" fill="#991b1b">开关 1 causal?</text>
<rect x="128" y="86" width="96" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
<text x="134" y="100" font-size="10.5" fill="#991b1b">开关 2 sliding?</text>
<rect x="232" y="86" width="96" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
<text x="238" y="100" font-size="10.5" fill="#991b1b">开关 3 alibi?</text>
<rect x="336" y="86" width="96" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
<text x="342" y="100" font-size="10.5" fill="#991b1b">开关 4 softcap?</text>
<rect x="440" y="86" width="96" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
<text x="446" y="100" font-size="10.5" fill="#991b1b">开关 5 doc?</text>
<rect x="544" y="86" width="112" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
<text x="550" y="100" font-size="10.5" fill="#991b1b">开关 6 paged?</text>
</g>
<path d="M72 110 L72 132" stroke="#b91c1c" stroke-width="1.2"/>
<path d="M68 128 L72 134 L76 128 Z" fill="#b91c1c"/>
<text x="86" y="130" font-size="11" fill="#991b1b">6 个正交开关 → 2⁶ = 64 种组合（实验第 6 组实测）</text>
<rect x="24" y="140" width="632" height="34" fill="#fef2f2" stroke="#fecaca" rx="4"/>
<text x="34" y="156" font-size="11" fill="#7f1d1d">手写路线：每种组合各写一个 kernel；新变体 = 把上次的代码再抄一遍</text>
<text x="34" y="170" font-size="11" fill="#7f1d1d">现实版本：大多数组合根本没人写，于是退化成「构造一个 B×H×Q×KV 的布尔掩码」→ 显存先炸</text>

<text x="24" y="204" font-size="12.5" font-weight="600" fill="#1d4ed8">路线 B：FlexAttention（把变体当作编译器输入）</text>
<rect x="24" y="214" width="150" height="52" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
<text x="34" y="232" font-size="11.5" font-weight="600" fill="#1e3a8a">score_mod</text>
<text x="34" y="248" font-size="10" fill="#1e40af">s, b, h, i, j → s'</text>
<text x="34" y="261" font-size="10" fill="#1e40af">（逐点，写几行 Py）</text>
<rect x="190" y="214" width="150" height="52" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
<text x="200" y="232" font-size="11.5" font-weight="600" fill="#1e3a8a">mask_mod</text>
<text x="200" y="248" font-size="10" fill="#1e40af">b, h, i, j → bool</text>
<text x="200" y="261" font-size="10" fill="#1e40af">（or/and 可组合）</text>
<path d="M174 240 L186 240" stroke="#1d4ed8" stroke-width="1.4"/>
<path d="M340 240 L358 240" stroke="#1d4ed8" stroke-width="1.4"/>
<path d="M352 236 L358 240 L352 244 Z" fill="#1d4ed8"/>
<rect x="362" y="206" width="140" height="68" rx="5" fill="#ede9fe" stroke="#c4b5fd"/>
<text x="372" y="226" font-size="11.5" font-weight="600" fill="#5b21b6">torch.compile</text>
<text x="372" y="242" font-size="10" fill="#6d28d9">Dynamo 抓图 → Inductor</text>
<text x="372" y="256" font-size="10" fill="#6d28d9">生成 Triton 片段</text>
<text x="372" y="268" font-size="10" fill="#6d28d9">+ 注入手写模板</text>
<path d="M502 240 L520 240" stroke="#1d4ed8" stroke-width="1.4"/>
<path d="M514 236 L520 240 L514 244 Z" fill="#1d4ed8"/>
<rect x="524" y="206" width="132" height="68" rx="5" fill="#dcfce7" stroke="#86efac"/>
<text x="534" y="226" font-size="11.5" font-weight="600" fill="#14532d">1 个融合 kernel</text>
<text x="534" y="242" font-size="10" fill="#166534">fwd / bwd / decode</text>
<text x="534" y="256" font-size="10" fill="#166534">不材料化 N×N</text>
<text x="534" y="268" font-size="10" fill="#166534">BlockMask 跳过整块</text>
<text x="24" y="300" font-size="11.5" fill="#374151">论文 §1.1 的原话：</text>
<text x="24" y="316" font-size="11.5" fill="#1e40af">“We introduce FlexAttention, a novel compiler-driven programming model that allows implementing the majority of attention variants in a few lines of idiomatic PyTorch code.”</text>
<text x="24" y="332" font-size="10.5" fill="#6b7280">出处：arXiv:2412.05496（Dong, Feng, Guessous, Liang, He），2024-12-07</text>
</svg>

这一节剩下的问题是：凭什么「写几行 Python」就能生成出一个跟手写 FlashAttention 打平的 kernel？答案全在第 2、3 节。

---

## 2. score_mod：softmax 之前的那个修正函数

### 2.1 定义与位置

标准的缩放点积注意力（论文式 2）：

$$
\mathbf S=\mathrm{softmax}\!\left(\frac{\mathbf Q\mathbf K^{T}}{\sqrt{d_k}}\right),\qquad \text{out}=\mathbf S\mathbf V
$$

FlexAttention 的全部改动是在等号右边中间插了一层（论文式 1）：

$$
\mathrm{FlexAttention}(\mathbf Q,\mathbf K,\mathbf V)=\mathrm{softmax}\Big(\underbrace{\mathrm{mod}\Big(\frac{\mathbf Q\mathbf K^{T}}{\sqrt{d_k}}\Big)}_{\text{用户写的就是这个}}\Big)\mathbf V
$$

逐元素写出来，用户提供的就是一个函数

$$
\tilde s_{ij}=\mathrm{mod}\big(s_{ij},\,b,\,h,\,i,\,j\big),\qquad s_{ij}=\langle q_i,k_j\rangle/\sqrt{d_k}
$$

它有五个输入：一个标量（分数）和四个标量位置索引。注意在真实 API 里这五个量**全都是 0 维张量**，不是 Python 整数——这是后面「 = 5 个位置参数」能被自动识别出来的前提。

### 2.2 四种典型变体，写成同一个形状

| 变体 | 数学形式 | 为什么它符合「逐点」定义 |
|:--|:--|:--|
| causal | $$\tilde s_{ij}=s_{ij}\ \text{if}\ i\ge j\ \text{else}\ -\infty$$ | 只依赖 $$(i,j)$$，不看别的分数的值 |
| ALiBi | $$\tilde s_{ij}=s_{ij}-\lambda_h\lvert i-j\rvert$$ | 同上，$$\lambda_h$$ 是按头编号取的一个常数或一个被捕获的张量 |
| soft-capping | $$\tilde s_{ij}=\kappa\tanh(s_{ij}/\kappa)$$ | 单调压缩，逐点；$$\kappa$$ 由 Py 变量捕获 |
| 相对位置偏置（T5 式） | $$\tilde s_{ij}=s_{ij}+\mathbf B_{i-j}$$ | $$\mathbf B$$ 是被「捕获」进 score_mod 的张量，取值只依赖 $$(i,j)$$ |

写成 Py 的话是这个样子（官方例子，见 [PyTorch 博客](https://pytorch.org/blog/flexattention/)）：

```python
# ALiBi：按头给斜率，按距离给线性惩罚
def alibi_mod(score, b, h, q_idx, kv_idx):
    return score + alibi_bias[h] * (kv_idx - q_idx)

# soft-capping：Gemma-2 / Grok-1 用来压住过大的 logit
cap = 20
def soft_cap(score, b, h, q_idx, kv_idx):
    score = score / cap
    score = torch.tanh(score)
    return score * cap

# 反之亦然：任意张量从外面「捕获」进来也不用重编译
bias = torch.randn(1024, 1024, device="cuda")
def score_mod(score, b, h, q_idx, kv_idx):
    return score + bias[q_idx][kv_idx]   # bias 可以换！
```

最后一个例子值得停一下：`bias` 是被**闭包捕获**的张量，`torch.compile` 的前端 TorchDynamo 不要求编译时用到的张量必须通过参数传进来，所以换一个新 `bias` **不会触发重编译**。这一点被专门记在博客里，也正因为如此，「保持同样的 .py 函数，靠改变捕获张量的值来切换模式」成了 FlexAttention 推理侧的标准打法（第 10 节会看到它在 decode offset 上怎么用）。

反过来，**标量**是会被烤进 kernel 的：2026 年的 FA4 博客明确写了

> “However, scalars captured in `score_mod` or `mask_mod` are baked into the compiled kernel. If you have a `soft_cap` value that changes between calls, each unique value triggers a recompilation.”

（[FlexAttention + FlashAttention-4: Fast and Flexible](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)，2026-03-05）

张量 vs 标量，一字之差，一个是「不用重编译」，一个是「每个新值重编译一次」。这是实践中踩得最频繁的一脚。

### 2.3 「逐点」这个词是契约，不是修辞

`mod` 的签名只给了它五个参数，而且五个里没有一个是「别的分数」。这不是偷懒，而是**能不能编译成 FlashAttention 的前提**。第 3 节给出数学上的理由，这里先说结论：

> **契约**：$$\tilde s_{ij}$$ 的值只能依赖 $$s_{ij}$$ 本身、位置 $$(b,h,i,j)$$，以及被捕获张量在**这些位置**上的取值。它不能依赖同一行里其它位置的分数值，也不能依赖 softmax 的分母。

违反这条契约的修正不是「性能差」，而是「算出来的东西不对」——第 8 节会用数字说明。

---

## 3. online softmax：为什么任意逐点修正都能被塞进那条循环

这是全文最该看明白的一节。

### 3.1 没有 mod 的时候

FlashAttention 的核心是「online softmax」：不材料化 $$N\times N$$ 的分数矩阵，而是按 KV 分块一路扫过去，只维护三个运行量。对第 $$i$$ 行而言，扫到第 $$t$$ 个 KV 块之后手上拿着

$$
m_i^{(t)}=\max_{j\le j_t}s_{ij},\qquad
l_i^{(t)}=\sum_{j\le j_t}e^{\,s_{ij}-m_i^{(t)}},\qquad
\mathrm{acc}_i^{(t)}=\sum_{j\le j_t}e^{\,s_{ij}-m_i^{(t)}}v_j
$$

最后 $$o_i=\mathrm{acc}_i/l_i$$。关键步是接上新块时的**重标定**：因为运行中最大值会从 $$m_i^{(t-1)}$$ 长到 $$m_{ij}=\max\big(m_i^{(t-1)},\max_{j\in B_t}s_{ij}\big)$$，旧的累加量必须乘上修正因子

$$
\alpha_i=2^{\,m_i^{(t-1)}-m_{ij}}\quad(\text{或}\ e^{\,\cdot}\ \text{，取决于底数，见 3.4})
$$

于是

$$
l_i^{(t)}=\alpha_i\, l_i^{(t-1)}+\sum_{j\in B_t}e^{\,s_{ij}-m_{ij}},\qquad
\mathrm{acc}_i^{(t)}=\alpha_i\,\mathrm{acc}_i^{(t-1)}+\sum_{j\in B_t}e^{\,s_{ij}-m_{ij}}v_j
$$

### 3.2 加上 mod 之后：什么都没坏

现在把 $$s_{ij}$$ 全部换成 $$\tilde s_{ij}$$。逐点修正是在「算 $$\max$$ 之前」施加的，所以：

$$
m_{ij}=\max\Big(m_i^{(t-1)},\ \max_{j\in B_t}\tilde s_{ij}\Big)
$$

是一个**关于修正后分数的**运行最大值。既然 $$\tilde s_{ij}$$ 已经被算好了、且它在后面不会再被修改，那么：

- 第 $$t$$ 步的行一眼就能看到的 $$\max_{j\in B_t}\tilde s_{ij}$$ 是「真值」的一部分，不需要回头重算；
- 已经进了 $$\mathrm{acc}$$ 的旧贡献，全部可以通过乘一个标量 $$\alpha$$ 重新对齐到新的基准上；
- 分母 $$l$$ 同理。

这正是 PyTorch 模板里的那几行（锁定 `v2.14.1`，`torch/_inductor/kernel/flex/templates/common.py.jinja:111-140`）：

```python
m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
if not ROWS_GUARANTEED_SAFE:
    masked_out_rows = (m_ij == float("-inf"))
    m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
else:
    m_ij_masked = m_ij

alpha = tl.math.exp2(m_i - m_ij_masked)
p     = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

l_i = l_i * alpha + tl.sum(p, 1)
acc = acc * alpha[:, None]
...
acc = tl.dot(p.to(MATMUL_PRECISION), v.to(q.dtype), acc, input_precision=FLOAT32_PRECISION)
m_i = m_ij
```

包含这段代码的循环与它在整个 kernel 里的位置，如下图所示：

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 300" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="300" fill="#ffffff"/>
<text x="24" y="24" font-size="15" font-weight="600" fill="#111827">带 mod 的 online softmax：三个运行量 + 一个重标定因子 α</text>
<text x="24" y="42" font-size="11.5" fill="#4b5563">每一步只有一个新块进入视野，但它可能抬升整行的运行最大值 → 旧的 acc 与 l 全部乘 α 重新对齐</text>

<text x="24" y="70" font-size="11.5" font-weight="600" fill="#111827">KV 方向（按块进入）</text>
<g>
<rect x="120" y="56" width="70" height="26" rx="3" fill="#e0f2fe" stroke="#7dd3fc"/>
<text x="128" y="73" font-size="10.5" fill="#0c4a6e">B₁ (已扫)</text>
<rect x="196" y="56" width="70" height="26" rx="3" fill="#e0f2fe" stroke="#7dd3fc"/>
<text x="204" y="73" font-size="10.5" fill="#0c4a6e">B₂ (已扫)</text>
<rect x="272" y="56" width="70" height="26" rx="3" fill="#fef08a" stroke="#facc15"/>
<text x="280" y="73" font-size="10.5" fill="#713f12">Bₜ (当前)</text>
<rect x="348" y="56" width="70" height="26" rx="3" fill="#f1f5f9" stroke="#cbd5e1"/>
<text x="356" y="73" font-size="10.5" fill="#64748b">Bₜ₊₁ (未见)</text>
<text x="424" y="73" font-size="10.5" fill="#94a3b8">…</text>
</g>

<path d="M307 82 L307 106" stroke="#2563eb" stroke-width="1.3"/>
<path d="M303 100 L307 108 L311 100 Z" fill="#2563eb"/>
<text x="316" y="108" font-size="11" fill="#1d4ed8">① s = Q·Kᵀ / √d   ② s̃ = mod(s, b, h, i, j)   ③ 取本块行 max</text>

<rect x="24" y="122" width="300" height="46" rx="5" fill="#f8fafc" stroke="#94a3b8"/>
<text x="34" y="140" font-size="11.5" font-weight="600" fill="#0f172a">运行最大值</text>
<text x="130" y="140" font-size="11.5" fill="#1d4ed8">m_ij = max( m_i , max_j s̃_ij )</text>
<text x="34" y="158" font-size="10" fill="#64748b">注意 max 作用在「修正之后」的 s̃ 上 —— 这是全部正确性的来源</text>

<rect x="340" y="122" width="316" height="46" rx="5" fill="#f8fafc" stroke="#94a3b8"/>
<text x="350" y="140" font-size="11.5" font-weight="600" fill="#0f172a">重标定因子</text>
<text x="446" y="140" font-size="11.5" fill="#b45309">α = 2^( m_i − m_ij )</text>
<text x="350" y="158" font-size="10" fill="#64748b">标量，一行一个；把已经算好的旧贡献整体缩回去</text>

<path d="M150 168 L150 190" stroke="#94a3b8" stroke-width="1.1"/>
<path d="M480 168 L480 190" stroke="#94a3b8" stroke-width="1.1"/>

<rect x="24" y="192" width="300" height="46" rx="5" fill="#fff7ed" stroke="#fdba74"/>
<text x="34" y="210" font-size="11.5" font-weight="600" fill="#7c2d12">运行分母 l_i</text>
<text x="130" y="210" font-size="11.5" fill="#9a3412">l_i ← α·l_i + Σ_j 2^( s̃_ij − m_ij )</text>
<text x="34" y="228" font-size="10" fill="#7c2d12">l 只会单调增长：它永远不小于已扫描部分的和</text>

<rect x="340" y="192" width="316" height="46" rx="5" fill="#fff7ed" stroke="#fdba74"/>
<text x="350" y="210" font-size="11.5" font-weight="600" fill="#7c2d12">运行分子 acc_i</text>
<text x="446" y="210" font-size="11.5" fill="#9a3412">acc ← α·acc + Σ_j 2^( s̃_ij − m_ij )·v_j</text>
<text x="350" y="228" font-size="10" fill="#7c2d12">矩阵形式就是 acc += P·V，P 是已经归一化到 α 基准的概率块</text>

<path d="M340 268 L172 268" stroke="#2563eb" stroke-width="1.2"/>
<path d="M180 264 L172 268 L180 272 Z" fill="#2563eb"/>
<text x="352" y="272" font-size="11.5" fill="#1d4ed8">收尾：o_i = acc_i / l_i ，lse_i = m_i + log₂ l_i</text>
<text x="24" y="290" font-size="10.5" fill="#6b7280">对应源码：torch/_inductor/kernel/flex/templates/common.py.jinja:111-140 与 flex_attention.py.jinja:193-216（commit 5c48869）</text>
</svg>

实验**第 1 组**直接验证了这件事：把 noop、causal、ALiBi、soft-capping、相对位置偏置、causal+ALiBi 六种变体各跑一遍「分块 32 流式」与「一次性材料化」，比对输出与 logsumexp：

| 变体 | $$\max\lvert\Delta o\rvert/\max\lvert o\rvert$$ | $$\max\lvert\Delta \mathrm{lse}\rvert$$ |
|:--|--:|--:|
| noop | 9.009e-16 | 1.776e-15 |
| causal | 2.498e-16 | 1.776e-15 |
| alibi | 1.413e-15 | 8.882e-16 |
| softcap | 1.495e-15 | 1.776e-15 |
| rel_bias | 9.259e-16 | 1.776e-15 |
| causal+alibi | 2.351e-16 | 8.882e-16 |
| **最差** | **1.776e-15** | |

这不是「六个巧合」，是**同一条恒等式在六个不同的 mod 上**。

### 3.3 严谨一点：为什么前一个 JVM 五分钟智商税收酒吧还得「随便选个参考值都行」

值得把上面的直觉写成一句定理，因为它解释了「为什么 $$m_i$$ 的取值方式可以自由改」：

> **参考值不变性**。对第 $$i$$ 行，记 $$A_i=\sum_j 2^{\tilde s_{ij}}$$，$$B_i=\sum_j 2^{\tilde s_{ij}}v_j$$。则对任意标量 $$c$$，
> $$o_i=\frac{B_i}{A_i}=\frac{2^{-c}B_i}{2^{-c}A_i}$$
> 与 $$c$$ 无关。

把 $$c$$ 取成运行最大值 $$m_i^{(t)}$$ 得到的就是 3.1 的递推。**所以把 $$c$$ 取成 max 不是为了保证正确性，而是为了保证 $$2^{\tilde s_{ij}-c}\le 1$$**——也就是为了不溢出。

这一点在有了 mod 之后会变得非常具体。实验**第 3 组**把一个「看上去最无害」的 mod——给所有分数加一个常数——跑了三遍（PyTorch 的实现、取修正后 max、取修正前 max）：

| 常数偏置 | 参考 vs 取**修正后** max | 参考 vs 取**修正前** max | 实际求过的 $$\log_2$$ 指数上界（前 / 后） |
|:--|--:|--:|--:|
| $$+80$$ | 2.109e-15 | 2.776e-15 | 115.42 / **0.00** |
| $$+160$$ | 3.386e-15 | 4.774e-15 | 230.83 / **0.00** |

两件事同时成立：

1. **数学上两种取法都对**（误差同为 $$10^{-15}$$ 级）——这印证了上面的参考值不变性；
2. **数值上取修正前会死**。差值恰好是 $$\text{bias}/\ln 2$$（$$80/0.6931=115.42$$、$$160/0.6931=230.83$$），而 float32 的溢出阈值是 $$2^{127}\approx1.70\times10^{38}$$。$$+160$$ 那个偏置在 float32 下直接溢出。

一句话总结 3.2–3.3：

> **「取 $$\max$$ 在修正之后」不是为了让数学成立（数学对任何参考值都成立），而是为了让数值裕度不被用户在 mod 里随手加的常数吃掉。**

### 3.4 还有一个坑：为什么模板内部用的是 $$\exp_2$$

上面的式子我一直写 $$2^{(\cdot)}$$ 而不是 $$e^{(\cdot)}$$，这不是偷懒。源码里有一个常量：

```python
RCP_LN2: tl.constexpr = 1.44269504      # common.py.jinja:169
...
post_mod_scores *= RCP_LN2              # common.py.jinja:106-107
```

意思是：把修正后的分数先乘上 $$1/\ln 2=\log_2 e$$，然后就可以全程用硬件更快的实现 $$\exp_2(\cdot)$$ 代替 $$\exp(\cdot)$$。代价是**所有与「对数量纲」有关的东西都自动变成 2 底**：

- 累加出来的 $$l_i$$ 是「以 2 为底的 exp 之和」；
- 模板返回的是 `lse = m_i + tl.math.log2(l_i)`（`flex_attention.py.jinja:212`）——**单位是 $$\log_2$$**；
- PyTorch 在把这东西交回给用户之前，要乘回去：`nn/attention/flex_attention.py:2549-2558` 里 `ln2 = math.log(2.0); lse_scaled = lse_scaled * ln2`，并且这个转换**只在非 FLASH 后端做**（`stats_are_log2=kernel_options["BACKEND"] != "FLASH"`，`:2592`）。

实验**第 2 组**把三条路径两两比对：自然对数实现、base-2 内部实现再换算回去、以及定义式 $$m+\log\sum_j e^{s_{ij}-m}$$。三条路径两两相差 ≤ 8.882e-16，且 `RCP_LN2 = 1/ln2 = 1.44269504` 与模板里的常量逐位相等。

**实践含义**：如果你自己 `return_lse=True` 之后拿 lse 去复现 softmax 或者做上下文并行/推测解码的对数概率对齐，务必确认你拿到的是哪个单位。**同一个 API，换 backend 会换单位**（虽然 PyTorch 已经替你换算了，但如果你走的不是 `flex_attention()` 这一层而是直接对着 HOP 编程，就要自己管）。

### 3.5 两个边界必须先规定好

**整行被掩掉**。此时 $$l_i=0$$ 且 $$m_i=-\infty$$。模板的做法（`flex_attention.py.jinja:193-198`）：

```python
# Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
l_i = tl.where(l_i == 0.0, 1, l_i)
acc = acc / l_i[:, None]
```

把 $$l$$ 从 0 顶回 1，于是 $$\mathrm{acc}/l=0$$，输出是零向量、lse 是 $$-\infty$$。这是一个**约定**，不是推导：它决定了「一个 query 完全看不见任何 key」时模型拿到的是零而不是 NaN。

**全掩行的安全检查也可以关掉**。`ROWS_GUARANTEED_SAFE` 这个 kernel option（`nn/attention/flex_attention.py:226-230`）允许你声明「每行至少有一个不被掩」，从而省掉上面那段判断。文档给的例子是 causal——每个 query 至少能看见自己。

---

## 4. mask_mod：掩码为什么必须单列

既然 `score_mod` 严格强于 `mask_mod`（任何掩码都能写成 $$\tilde s = \text{where}(\text{mask}, s, -\infty)$$），为什么还要多一个 API？

官方 FAQ 给了两个理由，都是硬理由。

**理由一（语义）**：用 score_mod 表达掩码，需要在「打分 = 0」这个假输入上跑一遍用户的 score_mod，看它是否返回 $$-\infty$$。原文：

> “However, this is hacky – what if the user's score_mod returned -inf when the input is 0? Or what if the user's score_mod masked out with a large negative value instead of -inf?”

（[PyTorch 博客 FAQ](https://pytorch.org/blog/flexattention/)）

**理由二（性能，这才是关键）**：给每个已算元素施加掩码要掉 15–20%。causal 有 50% 稀疏，跳过一半计算本来该省 2×，结果被逐元素掩码吃掉一大截。

解决办法是把掩码**从「每个元素的事」升级成「每个 128×128 块的事」**：一个块如果完全不需要掩码，就完全不跑 mask_mod。而要做到这件事，编译器必须**先知道**哪些块是需要掩码的——于是就有了 `BlockMask`，也就必须有 `mask_mod` 这个「不接收分数、只回答 bool」的独立入口。

> 推论：**掩码成本是有两档的**。第一档是「这块要不要算」（块级，便宜，`kv_indices` 一次查表），第二档是「这块里哪些元素要掩」（元素级，贵，只有对角线上的块才需要）。第 5 节的图片会把这两档分开画。

顺带一个很容易踩的工程细节：PyTorch **靠位置参数个数**判定你给的是哪一种（`nn/attention/flex_attention.py:315-347`）：

```python
if num_positional_args != 5 and num_positional_args != 4:
    raise AssertionError(f"Expected 4 or 5 positional args, got {num_positional_args}")
if num_positional_args == 5:
    return _ModificationType.SCORE_MOD
elif num_positional_args == 4:
    return _ModificationType.MASK_MOD
```

给 mask_mod 加了个默认值参数？不好意思，它会被判成第三种东西或者在别的地方报错。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 268" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="268" fill="#ffffff"/>
<text x="24" y="24" font-size="15" font-weight="600" fill="#111827">同一个 causal 掩码的三种写法，成本完全不同</text>
<text x="24" y="42" font-size="11.5" fill="#4b5563">从左到右：逐元素 score_mod → 逐元素 mask_mod → 块级 mask_mod（前两种都要逐元素判断）</text>

<text x="24" y="72" font-size="12" font-weight="600" fill="#b91c1c">① score_mod：对每个标量判断</text>
<g>
<rect x="24" y="82" width="150" height="150" fill="#ffffff" stroke="#94a3b8"/>
<rect x="24" y="82" width="30" height="30" fill="#1d4ed8"/><rect x="54" y="82" width="30" height="30" fill="#dbeafe"/><rect x="84" y="82" width="30" height="30" fill="#dbeafe"/><rect x="114" y="82" width="30" height="30" fill="#dbeafe"/><rect x="144" y="82" width="30" height="30" fill="#dbeafe"/>
<rect x="24" y="112" width="30" height="30" fill="#1d4ed8"/><rect x="54" y="112" width="30" height="30" fill="#1d4ed8"/><rect x="84" y="112" width="30" height="30" fill="#dbeafe"/><rect x="114" y="112" width="30" height="30" fill="#dbeafe"/><rect x="144" y="112" width="30" height="30" fill="#dbeafe"/>
<rect x="24" y="142" width="30" height="30" fill="#1d4ed8"/><rect x="54" y="142" width="30" height="30" fill="#1d4ed8"/><rect x="84" y="142" width="30" height="30" fill="#1d4ed8"/><rect x="114" y="142" width="30" height="30" fill="#dbeafe"/><rect x="144" y="142" width="30" height="30" fill="#dbeafe"/>
<rect x="24" y="172" width="30" height="30" fill="#1d4ed8"/><rect x="54" y="172" width="30" height="30" fill="#1d4ed8"/><rect x="84" y="172" width="30" height="30" fill="#1d4ed8"/><rect x="114" y="172" width="30" height="30" fill="#1d4ed8"/><rect x="144" y="172" width="30" height="30" fill="#dbeafe"/>
<rect x="24" y="202" width="30" height="30" fill="#1d4ed8"/><rect x="54" y="202" width="30" height="30" fill="#1d4ed8"/><rect x="84" y="202" width="30" height="30" fill="#1d4ed8"/><rect x="114" y="202" width="30" height="30" fill="#1d4ed8"/><rect x="144" y="202" width="30" height="30" fill="#1d4ed8"/>
</g>
<text x="186" y="130" font-size="10.5" fill="#991b1b">每个格子都要跑一遍</text>
<text x="186" y="146" font-size="10.5" fill="#991b1b">`where(i≥j, s, -inf)`；</text>
<text x="186" y="162" font-size="10.5" fill="#991b1b">25 个格子 = 25 次判断</text>
<text x="186" y="184" font-size="10.5" font-weight="600" fill="#991b1b">−15~20% 吞吐</text>
<text x="186" y="204" font-size="10" fill="#b91c1c">（博客实测）</text>

<text x="256" y="72" font-size="12" font-weight="600" fill="#b45309">② mask_mod（元素级）</text>
<g>
<rect x="256" y="82" width="150" height="150" fill="#ffffff" stroke="#94a3b8"/>
<rect x="256" y="82" width="30" height="30" fill="#ea580c"/><rect x="286" y="82" width="30" height="30" fill="#fed7aa"/><rect x="316" y="82" width="30" height="30" fill="#fed7aa"/><rect x="346" y="82" width="30" height="30" fill="#fed7aa"/><rect x="376" y="82" width="30" height="30" fill="#fed7aa"/>
<rect x="256" y="112" width="30" height="30" fill="#ea580c"/><rect x="286" y="112" width="30" height="30" fill="#ea580c"/><rect x="316" y="112" width="30" height="30" fill="#fed7aa"/><rect x="346" y="112" width="30" height="30" fill="#fed7aa"/><rect x="376" y="112" width="30" height="30" fill="#fed7aa"/>
<rect x="256" y="142" width="30" height="30" fill="#ea580c"/><rect x="286" y="142" width="30" height="30" fill="#ea580c"/><rect x="316" y="142" width="30" height="30" fill="#ea580c"/><rect x="346" y="142" width="30" height="30" fill="#fed7aa"/><rect x="376" y="142" width="30" height="30" fill="#fed7aa"/>
<rect x="256" y="172" width="30" height="30" fill="#ea580c"/><rect x="286" y="172" width="30" height="30" fill="#ea580c"/><rect x="316" y="172" width="30" height="30" fill="#ea580c"/><rect x="346" y="172" width="30" height="30" fill="#ea580c"/><rect x="376" y="172" width="30" height="30" fill="#fed7aa"/>
<rect x="256" y="202" width="30" height="30" fill="#ea580c"/><rect x="286" y="202" width="30" height="30" fill="#ea580c"/><rect x="316" y="202" width="30" height="30" fill="#ea580c"/><rect x="346" y="202" width="30" height="30" fill="#ea580c"/><rect x="376" y="202" width="30" height="30" fill="#ea580c"/>
</g>
<text x="418" y="130" font-size="10.5" fill="#92400e">语义清晰：只回答 bool，</text>
<text x="418" y="146" font-size="10.5" fill="#92400e"> −inf 的事由编译器统一做</text>
<text x="418" y="168" font-size="10.5" font-weight="600" fill="#92400e">但仍然要逐元素</text>

<text x="490" y="72" font-size="12" font-weight="600" fill="#15803d">③ BlockMask（块级）</text>
<g>
<rect x="490" y="82" width="166" height="150" fill="#ffffff" stroke="#94a3b8"/>
<rect x="490" y="82" width="33" height="30" fill="#22c55e"/><rect x="523" y="82" width="33" height="30" fill="#e5e7eb"/><rect x="556" y="82" width="33" height="30" fill="#e5e7eb"/><rect x="589" y="82" width="33" height="30" fill="#e5e7eb"/><rect x="622" y="82" width="33" height="30" fill="#e5e7eb"/>
<rect x="490" y="112" width="33" height="30" fill="#22c55e"/><rect x="523" y="112" width="33" height="30" fill="#22c55e"/><rect x="556" y="112" width="33" height="30" fill="#e5e7eb"/><rect x="589" y="112" width="33" height="30" fill="#e5e7eb"/><rect x="622" y="112" width="33" height="30" fill="#e5e7eb"/>
<rect x="490" y="142" width="33" height="30" fill="#22c55e"/><rect x="523" y="142" width="33" height="30" fill="#22c55e"/><rect x="556" y="142" width="33" height="30" fill="#22c55e"/><rect x="589" y="142" width="33" height="30" fill="#e5e7eb"/><rect x="622" y="142" width="33" height="30" fill="#e5e7eb"/>
<rect x="490" y="172" width="33" height="30" fill="#22c55e"/><rect x="523" y="172" width="33" height="30" fill="#22c55e"/><rect x="556" y="172" width="33" height="30" fill="#22c55e"/><rect x="589" y="172" width="33" height="30" fill="#22c55e"/><rect x="622" y="172" width="33" height="30" fill="#e5e7eb"/>
<rect x="490" y="202" width="33" height="30" fill="#22c55e"/><rect x="523" y="202" width="33" height="30" fill="#22c55e"/><rect x="556" y="202" width="33" height="30" fill="#22c55e"/><rect x="589" y="202" width="33" height="30" fill="#22c55e"/><rect x="622" y="202" width="33" height="30" fill="#22c55e"/>
<rect x="490" y="82" width="33" height="30" fill="none" stroke="#f59e0b" stroke-width="2.5"/>
<rect x="523" y="112" width="33" height="30" fill="none" stroke="#f59e0b" stroke-width="2.5"/>
<rect x="556" y="142" width="33" height="30" fill="none" stroke="#f59e0b" stroke-width="2.5"/>
<rect x="589" y="172" width="33" height="30" fill="none" stroke="#f59e0b" stroke-width="2.5"/>
<rect x="622" y="202" width="33" height="30" fill="none" stroke="#f59e0b" stroke-width="2.5"/>
</g>
<text x="490" y="248" font-size="10.5" fill="#14532d">绿 = full（完全不用掩）  灰 = 跳过（一次都不算）</text>
<text x="490" y="262" font-size="10.5" fill="#92400e">橙框 = partial（只有对角线上的块需要逐元素掩码）</text>
<text x="24" y="248" font-size="10.5" fill="#374151">第 ③ 种把「25 次判断」降成「5 次块级判断 + 5 个对角线块的元素级掩码」</text>
</svg>

---

## 5. BlockMask：把一份掩码压缩成两个索引向量

### 5.1 数据结构

`BlockMask` 的官方定义写在 `nn/attention/flex_attention.py:855-929` 的类文档里。它的本体是「介于 BCSR 与非稀疏格式之间的东西」，核心是两个张量：

- `kv_num_blocks`：形状 $$B\times H\times \lceil Q_{\rm LEN}/BS\rceil$$，每行表示一个 query 块要看**几个** KV 块；
- `kv_indices`：形状 $$B\times H\times \lceil Q_{\rm LEN}/BS\rceil\times \lceil KV_{\rm LEN}/BS\rceil$$，存这些 KV 块的**列号**（行尾之后的内容是未定义的）。

类文档里还列了另外三组：`<b>可选的</b>` `(full_kv_num_blocks, full_kv_indices)`（放那些完全不用掩码的块，官方说 causal 上值 ~15% 提速）、`<b>自动生成的</b>` `(q_num_blocks, q_indices)` 与 `(full_q_num_blocks, full_q_indices)`（反向要沿 Q 方向迭代，所以必须有反向的索引）、以及 2.13 之后为了确定性反向加的 `dq_write_order` 系列。

把这堆名字落到具体数字上，一眼就明白了（$$N=512$$、$$BS=64$$ 的 causal，$$B=H=1$$）：

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 236" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="236" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">BlockMask 的两个核心张量：一个记「几个」，一个记「哪几个」</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">8×8 的块网格（N=512、BS=64）上跑 causal；右边是同一份事实的两个扁平表示</text>

<g>
<rect x="24" y="56" width="192" height="160" fill="#ffffff" stroke="#94a3b8"/>
<rect x="24" y="56" width="24" height="20" fill="#22c55e"/><rect x="48" y="56" width="24" height="20" fill="#e5e7eb"/><rect x="72" y="56" width="24" height="20" fill="#e5e7eb"/><rect x="96" y="56" width="24" height="20" fill="#e5e7eb"/><rect x="120" y="56" width="24" height="20" fill="#e5e7eb"/><rect x="144" y="56" width="24" height="20" fill="#e5e7eb"/><rect x="168" y="56" width="24" height="20" fill="#e5e7eb"/><rect x="192" y="56" width="24" height="20" fill="#e5e7eb"/>
<rect x="24" y="76" width="24" height="20" fill="#22c55e"/><rect x="48" y="76" width="24" height="20" fill="#22c55e"/><rect x="72" y="76" width="24" height="20" fill="#e5e7eb"/><rect x="96" y="76" width="24" height="20" fill="#e5e7eb"/><rect x="120" y="76" width="24" height="20" fill="#e5e7eb"/><rect x="144" y="76" width="24" height="20" fill="#e5e7eb"/><rect x="168" y="76" width="24" height="20" fill="#e5e7eb"/><rect x="192" y="76" width="24" height="20" fill="#e5e7eb"/>
<rect x="24" y="96" width="24" height="20" fill="#22c55e"/><rect x="48" y="96" width="24" height="20" fill="#22c55e"/><rect x="72" y="96" width="24" height="20" fill="#22c55e"/><rect x="96" y="96" width="24" height="20" fill="#e5e7eb"/><rect x="120" y="96" width="24" height="20" fill="#e5e7eb"/><rect x="144" y="96" width="24" height="20" fill="#e5e7eb"/><rect x="168" y="96" width="24" height="20" fill="#e5e7eb"/><rect x="192" y="96" width="24" height="20" fill="#e5e7eb"/>
<rect x="24" y="116" width="24" height="20" fill="#22c55e"/><rect x="48" y="116" width="24" height="20" fill="#22c55e"/><rect x="72" y="116" width="24" height="20" fill="#22c55e"/><rect x="96" y="116" width="24" height="20" fill="#22c55e"/><rect x="120" y="116" width="24" height="20" fill="#e5e7eb"/><rect x="144" y="116" width="24" height="20" fill="#e5e7eb"/><rect x="168" y="116" width="24" height="20" fill="#e5e7eb"/><rect x="192" y="116" width="24" height="20" fill="#e5e7eb"/>
<rect x="24" y="136" width="24" height="20" fill="#22c55e"/><rect x="48" y="136" width="24" height="20" fill="#22c55e"/><rect x="72" y="136" width="24" height="20" fill="#22c55e"/><rect x="96" y="136" width="24" height="20" fill="#22c55e"/><rect x="120" y="136" width="24" height="20" fill="#22c55e"/><rect x="144" y="136" width="24" height="20" fill="#e5e7eb"/><rect x="168" y="136" width="24" height="20" fill="#e5e7eb"/><rect x="192" y="136" width="24" height="20" fill="#e5e7eb"/>
<rect x="24" y="156" width="24" height="20" fill="#22c55e"/><rect x="48" y="156" width="24" height="20" fill="#22c55e"/><rect x="72" y="156" width="24" height="20" fill="#22c55e"/><rect x="96" y="156" width="24" height="20" fill="#22c55e"/><rect x="120" y="156" width="24" height="20" fill="#22c55e"/><rect x="144" y="156" width="24" height="20" fill="#22c55e"/><rect x="168" y="156" width="24" height="20" fill="#e5e7eb"/><rect x="192" y="156" width="24" height="20" fill="#e5e7eb"/>
<rect x="24" y="176" width="24" height="20" fill="#22c55e"/><rect x="48" y="176" width="24" height="20" fill="#22c55e"/><rect x="72" y="176" width="24" height="20" fill="#22c55e"/><rect x="96" y="176" width="24" height="20" fill="#22c55e"/><rect x="120" y="176" width="24" height="20" fill="#22c55e"/><rect x="144" y="176" width="24" height="20" fill="#22c55e"/><rect x="168" y="176" width="24" height="20" fill="#22c55e"/><rect x="192" y="176" width="24" height="20" fill="#e5e7eb"/>
<rect x="24" y="196" width="24" height="20" fill="#22c55e"/><rect x="48" y="196" width="24" height="20" fill="#22c55e"/><rect x="72" y="196" width="24" height="20" fill="#22c55e"/><rect x="96" y="196" width="24" height="20" fill="#22c55e"/><rect x="120" y="196" width="24" height="20" fill="#22c55e"/><rect x="144" y="196" width="24" height="20" fill="#22c55e"/><rect x="168" y="196" width="24" height="20" fill="#22c55e"/><rect x="192" y="196" width="24" height="20" fill="#22c55e"/>
<rect x="24" y="56" width="24" height="20" fill="none" stroke="#f59e0b" stroke-width="2"/><rect x="48" y="76" width="24" height="20" fill="none" stroke="#f59e0b" stroke-width="2"/><rect x="72" y="96" width="24" height="20" fill="none" stroke="#f59e0b" stroke-width="2"/><rect x="96" y="116" width="24" height="20" fill="none" stroke="#f59e0b" stroke-width="2"/><rect x="120" y="136" width="24" height="20" fill="none" stroke="#f59e0b" stroke-width="2"/><rect x="144" y="156" width="24" height="20" fill="none" stroke="#f59e0b" stroke-width="2"/><rect x="168" y="176" width="24" height="20" fill="none" stroke="#f59e0b" stroke-width="2"/><rect x="192" y="196" width="24" height="20" fill="none" stroke="#f59e0b" stroke-width="2"/>
</g>
<text x="24" y="228" font-size="10.5" fill="#374151">绿=要算的块（36 个）  灰=跳过（28 个）  橙框=partial（8 个，要逐元素掩码）</text>

<path d="M228 136 L262 136" stroke="#64748b" stroke-width="1.3"/>
<path d="M256 132 L262 136 L256 140 Z" fill="#64748b"/>

<text x="272" y="60" font-size="11.5" font-weight="600" fill="#1e3a8a">kv_num_blocks（每行几个）</text>
<rect x="272" y="66" width="196" height="22" rx="3" fill="#eff6ff" stroke="#93c5fd"/>
<text x="280" y="81" font-size="10.5" fill="#1e40af">[1, 2, 3, 4, 5, 6, 7, 8]</text>
<text x="272" y="102" font-size="10.5" fill="#475569">形状 (1, 1, 8) —— 第 r 行要看 r+1 个 KV 块</text>

<text x="272" y="126" font-size="11.5" font-weight="600" fill="#1e3a8a">kv_indices（哪几个，按行存）</text>
<rect x="272" y="132" width="196" height="100" rx="3" fill="#eff6ff" stroke="#93c5fd"/>
<text x="280" y="148" font-size="10.5" fill="#1e40af">row0: [0, ·, ·, ·, ·, ·, ·, ·]</text>
<text x="280" y="164" font-size="10.5" fill="#1e40af">row1: [0, 1, ·, ·, ·, ·, ·, ·]</text>
<text x="280" y="180" font-size="10.5" fill="#1e40af">row2: [0, 1, 2, ·, ·, ·, ·, ·]</text>
<text x="280" y="196" font-size="10.5" fill="#475569">… 行尾之后的内容未定义（源码原话）</text>
<text x="280" y="212" font-size="10.5" fill="#1e40af">row7: [0, 1, 2, 3, 4, 5, 6, 7]</text>

<text x="490" y="60" font-size="11.5" font-weight="600" fill="#5b21b6">为什么还要 full_* 一份</text>
<rect x="490" y="66" width="166" height="100" rx="3" fill="#f5f3ff" stroke="#c4b5fd"/>
<text x="498" y="84" font-size="10.5" fill="#5b21b6">full_kv_num_blocks</text>
<text x="498" y="100" font-size="10.5" fill="#6d28d9">[0, 1, 2, 3, 4, 5, 6, 7]</text>
<text x="498" y="120" font-size="10.5" fill="#5b21b6">partial = 36 − 28 = 8</text>
<text x="498" y="140" font-size="10" fill="#6d28d9">28/36 = 77.8% 的块可以</text>
<text x="498" y="154" font-size="10" fill="#6d28d9">完全不跑 mask_mod</text>

<text x="490" y="184" font-size="11.5" font-weight="600" fill="#14532d">反向还要一份转置</text>
<rect x="490" y="190" width="166" height="34" rx="3" fill="#ecfdf5" stroke="#6ee7b7"/>
<text x="498" y="205" font-size="10" fill="#166534">(q_num_blocks, q_indices)</text>
<text x="498" y="219" font-size="10" fill="#166534">沿 Q 方向迭代才能算 dKV</text>
</svg>

实验**第 5 组**按这个定义把 causal 掩码真的展开成了一组数字（$$N=512$$、$$BS=64$$、$$B=H=1$$）：

| 掩码 | $$BS$$ | 总块数 | 保留 | full | partial | 整块跳过 | 元素密度 | **块级密度** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| causal | 64 | 64 | 36 | 28 | 8 | 28 | 50.1% | **56.2%** |
| sliding ($$w=256$$) | 64 | 64 | 30 | 18 | 12 | 34 | 37.5% | **46.9%** |
| document | 64 | 64 | 26 | 3 | 23 | 38 | 20.8% | **40.6%** |
| causal | 128 | 16 | 10 | 6 | 4 | 6 | 50.1% | **62.5%** |

请注意最后一列：**块级密度总是 ≥ 元素密度**。这就是「粒度是有损的」这句话的量化版本：

- causal 在理想情况下只有 50.1% 的元素要算，但按 128×128 切块之后实际要算 56.2% 的块；
- document 掩码更极端：元素密度 20.8%，块级密度 40.6%——**块有这么大，掩码只能按游击队员的粒度执行**；
- $$BS$$ 越大越吃亏：causal 从 $$BS=64$$ 到 $$BS=128$$，块级密度从 56.2% 涨到 62.5%。

### 5.2 显存这笔账

论文给出的复杂度是对的：元数据随 $$\lceil Q_{\rm LEN}/BS\rceil\times\lceil KV_{\rm LEN}/BS\rceil$$ 增长，而不是 $$O(N^2)$$——但注意前者本身就是后者除以 $$BS^2$$。实验按 int32 索引、bool 按 1 字节算出了具体的数：

| 序列长 | $$BS$$ | 稠密布尔掩码 | BlockMask 元数据 | 比值 |
|--:|--:|--:|--:|--:|
| 8192 | 128 | 67.11 MB | 0.0166 MB | 4033× |
| 32768 | 128 | 1073.74 MB | 0.2632 MB | 4080× |
| 32768 | 1024 | 1073.74 MB | 0.0042 MB | 254200× |
| 1048576 | 128 | 1099511.63 MB | 268.4682 MB | 4096× |

四千倍到二十五万倍，看着很爽，但**最后一行才是值得记住的**：一百万长度的序列，$$BS=128$$ 的元数据是 **268 MB**。它比 1.1 TB 的稠密掩码小四千倍，却也不是可以随手每人一份的东西——官方 FAQ 的建议是「确实在乎就调大 BLOCK_SIZE」。官方博客把它概括为「对大多数用例 overhead 可以忽略」，这个说法成立的前提是你接受 268 MB。

### 5.3 `create_block_mask` 自己的成本

这一条在实践中比上面两笔账更常咬人。`nn/attention/flex_attention.py:1967` 的 `create_block_mask` 内部第一行实际工作是

```python
mask_tensor = create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device)   # :2063
```

而 `create_mask` 的做法是**真的把那个 $$B\times H\times Q_{\rm LEN}\times KV_{\rm LEN}$$ 的布尔张量材料化出来**（`:1828`），再由 `_convert_mask_to_block_mask` 降采样成块级。所以：

- **时间**：要跑 $$Q_{\rm LEN}\times KV_{\rm LEN}$$ 次 mask_mod；
- **峰值显存**：真的有一份 $$B\cdot H\cdot Q\cdot KV$$ 的 bool 张量短暂存在过。

官方 FAQ 给了三条对策，按推荐度排：

1. **跨 batch / head 广播**：`create_block_mask(mod, B=None, H=None, ...)`——绝大多数掩码（causal、滑窗）与 batch 和头无关，设成 `None` 直接省两个维度。
2. **编译它**。FAQ 里原本写「`torch.compile` 因为一些限制不能直接作用在 `create_block_mask` 上，请用 `_compile=True`」，但**这个限制已经被修复了**——2.14.1 里 `_compile=True` 会先抛 `DeprecationWarning`（`:2040-2045`）：

   > “_compile flag on create_block_mask was originally added to work around a torch.compile limitation. That limitation has since been addressed. So, to compile create_block_mask, we suggest doing torch.compile(create_block_mask).”

3. **自己写构造函数**。真的在意就别走 mask_mod，直接给出 `kv_num_blocks` 和 `kv_indices`——FAv2 之后这在 vLLM 里是常规操作（第 9 节）。

顺便把 https://github.com/vllm-project/vllm 的实践摆在这里（锁定 `db9527a4`）：

```python
torch._dynamo.config.recompile_limit = 16                       # vllm/.../flex_attention.py:54
create_block_mask_compiled = torch.compile(                     # :55-57
    create_block_mask, fullgraph=True, mode="reduce-overhead"
)
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)   # :58
```

注意 `reduce-overhead` 这个 mode：它是给**每个 step 都要重建一次掩码**（文档掩码、data-dependent 稀疏）的场景准备的，代价是 CUDA graph 捕获。这不是「更快」，是「换一种摊销方式」。

官方 FAQ 也给了一张「什么时候该什么时候缓存 BlockMask」的清单，照抄就够用：

| 掩码的变化频率 | 建议 |
|:--|:--|
| 不变（causal） | 全局预构造一次，`functools.partial(flex_attention, block_mask=bm)` |
| 每 batch 变（document masking） | 模型开头构造一次，**穿层透传**，让所有层摊销 |
| 每层都变（data-dependent sparsity） | 最难；收益取决于稀疏度与构造速度，考虑直接写 `kv_indices` |

---

## 6. 编译链路：从 Python 函数到一个 Triton kernel

这一节回答「compiler-driven」这四个字到底做了什么。整条链路五步，全部行号锁在 `5c48869`。

**第 1 步 · TorchDynamo 抓图。** `flex_attention()` 在 Dynamo 编译期间走的是专门分支（`nn/attention/flex_attention.py:2571-2593`），它把 `block_mask` 摊平成元组（`as_tuple()`，`:1200`）再交给 HOP。这个函数还会把**头数与头维标成静态**：

```python
for x in [query, key, value]:
    torch._dynamo.mark_static(x, -3)     # 头数 H
    torch._dynamo.mark_static(x, -1)     # 头维 E
```

头数与头维参与几十上百个编译期决策（BLOCK_M/N 的选择、寄存器分配、是否需要 packing），让它们动态只会换来一堆冗余编译产物。

**第 2 步 · 把用户的两个函数 traced 成 FX 图。** `torch/_higher_order_ops/flex_attention.py:408` 的 `trace_flex_attention` 做的事是：用**零维假张量**当输入调用一遍 score_mod 与 mask_mod，各自 `make_fx` 出一个 GraphModule，注册成 `sdpa_score` / `sdpa_mask`：

```python
example_vals = [query.new_zeros((), requires_grad=query.requires_grad)] + \
               [query.new_zeros((), dtype=torch.int) for _ in range(4)]
mask_example_vals = [query.new_zeros((), dtype=torch.int) for _ in range(4)]
with TransformGetItemToIndex():
    score_graph = reenter_make_fx(score_mod)(*example_vals, *score_mod_other_buffers)
    mask_graph  = reenter_make_fx(mask_mod)(*mask_example_vals, *mask_mod_other_buffers)
```

那个 `TransformGetItemToIndex` 值得一提：它把 `bias[q_idx][kv_idx]` 这种**连续两次 `__getitem__`** 改写成 `bias[q_idx, kv_idx]`，因为后者才能被 Inductor 翻译成一次 Triton `tl.load`。写 `bias[q][kv]` 与写 `bias[q, kv]` 在 eager 下完全等价，在这里不是。

**第 3 步 · Inductor 把这两张图编译成 Triton IR 片段。** 入口是 `torch/_inductor/kernel/flex/flex_attention.py:155` 的 `register_lowering`：

```python
subgraph_buffer = build_subgraph_buffer(
    placeholder_inps + list(score_mod_other_buffers), subgraph)   # :253-255
freeze_irnodes(subgraph_buffer)
mask_graph_buffer = build_subgraph_buffer(
    mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph)  # :267-269
```

`build_subgraph_buffer` 是通用机制：它把 FX 图里的每个节点逐个 lowering 成 Triton IRNode。`freeze_irnodes` 之后这些片段就不再变动，可以安全地往模板里塞。

**第 4 步 · 用手写模板拼装。** 模板声明在 `torch/_inductor/kernel/flex/flex_attention.py:145-150`：

```python
flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=load_flex_template("flex_attention")
         + load_flex_template("utilities")
         + load_flex_template("common"),
    always_freeze_layout=True,
    ...
)
```

拼装的动作是 `maybe_append_choice(..., subgraphs=[subgraph_buffer, mask_graph_buffer], ...)`（`:514-538`）。**这一步是整个设计的枢纽**：online softmax、Tile Scheduler、TMA 描述符、块边界与序列边界处理全部是手写 Triton 代码，只有那两小块是生成出来的。编译器从不负责「发明 FlashAttention」，只负责「把用户的一小段 expressions 嵌进去」。

模板里留给用户的插槽长这样（`templates/common.py.jinja:53-73`，为避 Liquid 误会这里用代码块给出）：

{% raw %}
```python
{{ modification(
    subgraph_number=0,
    output_name="post_mod_scores",
    score="qk", b="off_z", h="off_h", m="m", n="n",
    out="qk",
    input_shapes={"score": ("BLOCK_M", "BLOCK_N"),
                  "m": ("BLOCK_M", "1"), "n": ("1", "BLOCK_N")},
    input_dtypes={"b": "index", "h": "index", "m": "index", "n": "index"},
) | indent_except_first(1) }}
```
{% endraw %}

注意 `input_shapes`：`score` 是 $$(B_M,B_N)$$，而 `m` 是 $$(B_M,1)$$、`n` 是 $$(1,B_N)$$。广播在这里就定死了——这也解释了为什么 score_mod 必须是「能对这两个广播形状做逐点运算」的东西。

**第 5 步 · 选后端。** 同一个入口会按条件分到不同的模板（`flex_attention.py:174-360`）：

| 顺序 | 分支 | 触发条件 | 位置 |
|:--|:--|:--|:--|
| 1 | CPU / C++ 模板 | `device.type == "cpu"` | `:174-185` |
| 2 | MPS / Metal 模板 | `device.type == "mps"` | `:186-199` |
| 3 | FlexDecode（split-KV） | `Q_LEN < 128` 且 batch/heads 静态且 GQA 比值是 2 的幂 | `:281-304` + `flex_decoding.py:66-132` |
| 4 | FLASH（CuTeDSL + FA4） | score/mask mod 足够简单且 Hopper/Blackwell | `:334-360` |
| 5 | 基础 Triton 模板 | 兜底 | `:306` 之后 |

有意思的是 `:167-173` 那份 docstring 还停留在「四种模板（Base Triton / Flex Decode / CPU CPP / MPS Metal）」，而代码已经多了 FLASH——**docstring 会过期，实际以 dispatch 为准**。这种事情在做源码引用时只能以 dispatch 分支为准，不能拿 docstring 当结论。

还有一个很实际的点：**如果忘了把它包进 torch.compile 会怎样**？源码会直接警告你（`nn/attention/flex_attention.py:2595-2605`）：

> “flex_attention called without torch.compile() - this will use an unfused implementation that materializes the full scores matrix instead of generating a fused kernel.”

也就是说：**没配 compile 的 FlexAttention 会退化成「比 FlashAttention 还糟」的实现**——它真会把 $$N\times N$$ 分数矩阵材料化出来。这不是慢一半，是回到 2022 年之前。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 250" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="250" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">编译链路：五个环节，只有中间两个环节是「生成」的</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">手写部分（绿）负责性能，生成部分（紫）负责语义 —— 这个分工是 FlexAttention 的全部工程取舍</text>

<g>
<rect x="24" y="58" width="118" height="52" rx="5" fill="#ede9fe" stroke="#c4b5fd"/>
<text x="34" y="76" font-size="11" font-weight="600" fill="#5b21b6">① torch.compile</text>
<text x="34" y="92" font-size="10" fill="#6d28d9">Dynamo 抓图</text>
<text x="34" y="104" font-size="9.5" fill="#7c3aed">mark_static(H, E)</text>

<rect x="158" y="58" width="128" height="52" rx="5" fill="#ede9fe" stroke="#c4b5fd"/>
<text x="168" y="76" font-size="11" font-weight="600" fill="#5b21b6">② make_fx 成两张图</text>
<text x="168" y="92" font-size="10" fill="#6d28d9">sdpa_score / sdpa_mask</text>
<text x="168" y="104" font-size="9.5" fill="#7c3aed">TransformGetItemToIndex</text>

<rect x="302" y="58" width="128" height="52" rx="5" fill="#ede9fe" stroke="#c4b5fd"/>
<text x="312" y="76" font-size="11" font-weight="600" fill="#5b21b6">③ Inductor</text>
<text x="312" y="92" font-size="10" fill="#6d28d9">build_subgraph_buffer</text>
<text x="312" y="104" font-size="9.5" fill="#7c3aed">→ Triton IR 片段</text>

<rect x="446" y="58" width="122" height="52" rx="5" fill="#dcfce7" stroke="#86efac"/>
<text x="456" y="76" font-size="11" font-weight="600" fill="#14532d">④ 手写 Triton 模板</text>
<text x="456" y="92" font-size="10" fill="#166534">online softmax + 分块</text>
<text x="456" y="104" font-size="9.5" fill="#15803d">TMA / epilogue 融合</text>

<rect x="584" y="58" width="72" height="52" rx="5" fill="#dcfce7" stroke="#86efac"/>
<text x="592" y="76" font-size="11" font-weight="600" fill="#14532d">⑤ kernel</text>
<text x="592" y="92" font-size="10" fill="#166534">autotune</text>
<text x="592" y="104" font-size="9.5" fill="#15803d">autotune 选一份配置</text>
</g>

<path d="M142 84 L154 84" stroke="#64748b" stroke-width="1.3"/><path d="M148 80 L154 84 L148 88 Z" fill="#64748b"/>
<path d="M286 84 L298 84" stroke="#64748b" stroke-width="1.3"/><path d="M292 80 L298 84 L292 88 Z" fill="#64748b"/>
<path d="M430 84 L442 84" stroke="#64748b" stroke-width="1.3"/><path d="M436 80 L442 84 L436 88 Z" fill="#64748b"/>
<path d="M568 84 L580 84" stroke="#64748b" stroke-width="1.3"/><path d="M574 80 L580 84 L574 88 Z" fill="#64748b"/>

<rect x="24" y="126" width="632" height="30" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="34" y="145" font-size="10.5" fill="#475569">→ 用户那两个 mod 从头到尾只写 Python 表达式；被捕获的张量不会被复制，只会被编译成 tl.load</text>

<rect x="24" y="166" width="309" height="60" rx="5" fill="#fef2f2" stroke="#fecaca"/>
<text x="34" y="184" font-size="11" font-weight="600" fill="#991b1b">生成的是哪两块</text>
<text x="34" y="200" font-size="10.5" fill="#7f1d1d">① score_mod 的 forward 计算（以及它的 joint graph）</text>
<text x="34" y="216" font-size="10.5" fill="#7f1d1d">② mask_mod 的 bool 计算（只在对角块上运行）</text>

<rect x="347" y="166" width="309" height="60" rx="5" fill="#ecfdf5" stroke="#a7f3d0"/>
<text x="357" y="184" font-size="11" font-weight="600" fill="#14532d">手写的是哪些</text>
<text x="357" y="200" font-size="10.5" fill="#166534">online softmax 的重标定、Triton tile 调度、间接访存</text>
<text x="357" y="216" font-size="10.5" fill="#166534">epilogue 融合、fp32 化的 lse/max，以及块边界与序列边界的处理</text>

<text x="24" y="244" font-size="10" fill="#6b7280">为什么不让编译器自动生成整个 kernel：因为 online softmax 的重标定与块稀疏访存是代数改写，通用编译器做不出来（参见论文 §2.3 对 Mirage 的讨论）</text>
</svg>

---

## 7. 反向是怎么自动出来的

用户只写了正向的三行 Python，梯度是从哪来的？

答案是：**PyTorch 对 score_mod 单独跑一次 AOTAutograd 的 joint graph，而不是对整个注意力跑。**

`torch/_higher_order_ops/flex_attention.py:646` 的 `create_fw_bw_graph` 是关键：

```python
dummy_aot_config = AOTConfig(fw_compiler=None, bw_compiler=None, ...)   # :661-669
...
def joint_f(score, b, h, m, n, example_grad, *other_buffers):
    def fw_with_masks(*args):
        captured_args = [mod_index(buffer, []) if (isinstance(buffer, torch.Tensor)
                         and buffer.requires_grad and buffer.ndim == 0) else buffer
                         for buffer in args[5:]]
        fw_out = score_mod(*args[:5], *captured_args)
        return ((fw_out,), (fw_out.requires_grad,))
    joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)   # :746
    args = [score, b, h, m, n] + list(other_buffers)
    _, grads = joint(args, [example_grad])
    ...
joint_graph = make_fx(joint_f)(*unwrapped_score_mod_indexes, example_grad, *unwrapped_other_buffers)
```

`create_joint` 是 AOTAutograd 的老工具：它把「前向函数」和「它的后继梯度」拼成一张图，这样一次 trace 就同时拿到两边的 IR。因为 `score_mod` 通常只有十几二十个节点，这张 joint graph 小到可以放心在空间出其生成 Triton。

生成出来的两张图随后和 forward 走**同一套 lowering 流程**：joint graph 被 lowering 成 Triton IR，注入到手写**反向模板**（`templates/flex_backwards.py.jinja`）里。反向 kernel 本身仍然是手写的（它要解决 dQ / dKV 两遍遍历、沿着 mask 的 Q 方向迭代、以及 dQ 的累加顺序），注入的依旧只有「用户那一段」。

### 7.1 三条源码里写死的规则

1. **score_mod 的输出必须依赖 score**（`:750-755`）：

   ```python
   if grads[0] is None:
       raise RuntimeError(
           "flex_attention backward requires the output of score_mod to "
           "depend on score. Got a score_mod whose output does not "
           "require gradients with respect to score."
       )
   ```

   换句话说，一个把 score 扔掉、只返回关于位置的函数的 score_mod 是没法反传的——因为它压根不影响 loss。

2. **mask_mod 里捕获的需要梯度的张量不支持**（`FlexAttentionAutogradOp.forward`，`:785-793`）。

3. **`max_scores` 不可求导**（`:813`）：`ctx.mark_non_differentiable(max_scores)`。它不是一个网络输出，是一个 Attn stat。

### 7.2 确定性：一个 2026 年才补上的口子

`dQ` 的累加天然带竞争：多个 KV 块（或多个 split）要往同一个 $$\mathrm{d}Q$$ 位置原子加。PyTorch 2.13 的解法是**预先算好写入顺序**，而不是事后去重：

- `BlockMask` 多了 `dq_write_order` / `dq_write_order_full` / `dq_kv_order` 三个字段；
- `create_block_mask` 多了 `compute_dq_write_order` 与 `dq_kv_order` 两个开关（`nn/attention/flex_attention.py:1977-1978`）；
- 通过 `torch.use_deterministic_algorithms(True)` 开启。

官方给的成本数字是：**`create_block_mask` 端到端开销远低于 1%，序列长 $$S=32768$$ 时是 $$+0.2\%$$**（[PyTorch 2.13 release blog](https://pytorch.org/blog/pytorch-2-13-release-blog/)）。「几乎零成本的确定性」是这句数字的含义。

值得注意的另一件事：**这个确定性是 Flash 后端专有的**——从源码的层叠可以看到 `(compute_dq_write_order)` 的 docstring 写的是 “precompute dQ write-order metadata needed by deterministic block-sparse FLASH backward”（`:1998-1999`）。而 FA4 博客的限制清单里也明确留着一条：

> “**Deterministic backward with block-sparsity:** The Flash backend's backward pass is not yet deterministic when block-sparsity is enabled (score-mod-only workloads are deterministic).”

---

## 8. 表达力边界：到底什么东西写不出来

这一节把前面积累的东西收敛成一张清单。

### 8.1 写得出来的

只要满足「逐点 + 只依赖自身与位置」，几乎什么都能写。实验**第 1 组**验证了六类，其中「causal + ALiBi」这种组合 **不需要任何新代码**——这正是论文的卖点之一。

更进一步，两个 mask_mod 可以用布尔运算组合（`nn/attention/flex_attention.py:1759-1785`）：

```python
def or_masks(*mask_mods):        # :1759
    def or_mask(b, h, q_idx, kv_idx):
        return functools.reduce(lambda x, y: x | y, [m(b,h,q_idx,kv_idx) for m in mask_mods])
    return or_mask

def and_masks(*mask_mods):       # :1773
    ...
```

实验**第 6 组**的实测：

- `or(prefix, causal)`（prefixLM）：与手写掩码**逐元素一致**，掩码密度 53.27%；
- `and(sliding, causal, document)`：同样逐元素一致，掩码密度 17.38%。

而 `or_masks` / `and_masks` 的实现本身只有十几行。

顺便把组合数也量化一下：6 个正交开关 → $$2^6=64$$ 种组合。手写路线要 64 个 kernel；FlexAttention 每个开关只写一个 mask_mod，组合交给布尔运算。

### 8.2 写不出来（之一）：依赖「整行统计量」的修正

这是最重要的一条边界，也是「为什么 score_mod 必须逐点」的答案。

设想这样一个修正：让整行的 logits 除以一个依赖行均值的行标量

$$
\tilde s_{ij}=s_{ij}\cdot\frac{1}{1+\exp\big(-\tfrac{1}{N}\textstyle\sum_j s_{ij}\big)}
$$

它违反了契约：$$\tilde s_{ij}$$ 的值依赖同一行里**所有列**的分数值，包括那些还没被加载的列。

实验**第 4 组**把这个东西摊开来看。令 score 矩阵被切成四块，流式算法在扫第 $$t$$ 块时只能知道前 $$t$$ 块的均值，它与真值的偏差随块序变化：

| 块序 | 部分行均值与真值的平均绝对偏差 |
|:--|--:|
| 1 | 0.125398 |
| 2 | 0.058475 |
| 3 | 0.034071 |
| 4 | 0.000000 |

**只有扫到最后一块，那个「行均值」才是行均值。** 于是：

- 逐点的 soft-capping 在流式与一次性两条路径上相差 **4.996e-16**（可表达）；
- 依赖行均值的修正在两条路径上相差 **4.365e-02**（不可表达）。

而且这个差值不是精度问题。 $$\alpha$$-重标定能做的一件事，也只有一件事：**把旧分母整体缩回去**。它没法回头修改已经混进 $$\mathrm{acc}$$ 的分子——因为那些分子在原定义下本来就该是别的值。

### 8.3 写不出来（之二）：作用在 softmax 之后或分母上的东西

`mod` 的位置被规定死了：在 $$\max$$ 之前、在 $$\exp$$ 之前。任何「softmax 之后」或「只进分母」的操作都不在这个位置上：

- **attention sink / sink token 的可学习常数项**：它往softmax 分母里塞一个额外的 $$\exp(c)$$ 项，但分子里没有对应的 value 贡献。位置够不着。
- **输出门控（attention gate，对 $$o_i$$ 乘一个可学习标量或向量）**：那是对注意力**输出**的操作，得写到 attention 外面去（或者靠 Inductor 的 epilogue 融合顺手做掉）。
- **top-k 稀疏（按分数值挑 k 个）**：这是一个「依赖别的元素的值」的掩码，属于 8.2 的范畴；真要做必须先跑一遍 pass 拿到 argpartition 结果，那就是 data-dependent sparsity，落回到「每步重建 BlockMask」的成本上。

### 8.4 写不出来（之三）：元素级真实稀疏

这一条不是 Flow 的边界，是当前的工程取舍。官方博客的原话：

> “Also note that despite FlexAttention being able to express sparsity in attention, we currently do not support embedding sparsity directly into the kernel (i.e. actual sparsity).”

今天能做到的是**块级跳过**（一个块要么全算要么不算）。 $$BS=128$$ 时，一个块里只要有一个元素要算，$$128\times128=16384$$ 个 MAC 就得全部走完——第 5 节那张表里「元素密度 20.8% vs 块级密度 40.6%」就是这件事的直接后果。

### 8.5 一张总结表

| 类别 | 例子 | 能不能写 | 代价/说明 |
|:--|:--|:--|:--|
| 加性逐点修正 | ALiBi、T5 bias、可学习 bias | ✅ | 近乎零成本，被捕获张量可变值不重编译 |
| 单调逐点压缩 | soft-capping、logit scaling | ✅ | 需要有 “tanh” 之类算子能 lower 到 Triton |
| 任意位置掩码 | causal、滑窗、文档掩码、prefixLM | ✅ | 用 mask_mod 而非 score_mod，走 BlockMask |
| 掩码的组合 | `and` / `or` 任意嵌套 | ✅ | 逐元素等价（实测 0 个不一致元素） |
| 依赖行的整体统计量 | 行归一化、top-k 掩码、按行列 σ | ❌ | 流式差 4.365e-02；须两遍 + 预 pass |
| 作用在 softmax 之后/分母上 | attention sink 的可学习常数、输出门控 | ❌ | 位置不够；sink 通常靠拼接一列 K/V 实现 |
| 元素级稀疏跳过 | 任意散点稀疏 | ❌（当前） | 只有块级；粒度 $$BS$$ 越大越亏 |

---

## 9. 案例：PagedAttention，或者说「BlockMask 就是页表」

这是论文 §5.1 的精彩之处：**vLLM 的 PagedAttention 本质上就是 BlockMask，只不过「块」换成了「页」。**

一个 paged KV cache 有两层索引：

- **逻辑层**：序列的第 $$i$$ 个 token 属于第 $$\lfloor i/P\rfloor$$ 个逻辑页（$$P$$ 是页大小）；
- **物理层**：逻辑页可能被放到任意物理页 $$p$$ 上。

用户写的 mask_mod 想要「逻辑」索引（因为 causal 的定义是 $$i\ge j$$ 在逻辑序列上），但 kernel 实际看到的是**物理**索引。桥就是一层转换：

$$
\text{logical\_block}=\text{physical\_to\_logical}[\,\text{req},\ \lfloor \text{kv\_idx}/P\rfloor\,],\qquad
\text{logical\_kv\_idx}=\text{logical\_block}\cdot P+(\text{kv\_idx}\bmod P)
$$

vLLM 里这段写得非常直白（锁定 `db9527a4`，`vllm/v1/attention/backends/flex_attention.py:427-460`）：

```python
def _convert_physical_to_logical(self, request_lookup, q_idx, physical_kv_idx):
    q_req = request_lookup[q_idx]
    physical_kv_block = physical_kv_idx // self.block_size          # :445
    physical_kv_offset = physical_kv_idx % self.block_size
    logical_block_idx = self.physical_to_logical[q_req, physical_kv_block]   # :447
    logical_kv_idx = logical_block_idx * self.block_size + physical_kv_offset
    live_block = logical_block_idx >= 0
    within_upper_bound = logical_kv_idx < self.seq_lens[q_req]
    within_lower_bound = logical_kv_idx >= 0
    is_valid = live_block & within_upper_bound & within_lower_bound
    ...
    return is_valid, logical_q_idx, logical_kv_idx
```

然后把它和用户原本的 mask_mod 组合起来（`:476-487`）：

```python
def final_mask_mod(b, h, q_idx, physical_kv_idx):
    (is_valid, logical_q_idx, logical_kv_idx) = (
        self._convert_physical_to_logical(self.doc_ids, q_idx, physical_kv_idx))
    return is_valid & self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx)
```

**结论性的观察**：`final_mask_mod` 又是「filter 与 user mask 的 and 组合」——跟第 8 节里 `and_masks` 的模式一模一样。论文 §5.1 把它叫做 “converted_mask_mod”，vLLM 全文把它直接叫 `final_mask_mod`。

而且 vLLM 的 `get_mask_mod`（`:625`）把多种 Pattern 叠在一起：paged / bidirectional 作为底座，然后 `and_masks(mask_mod, sliding_window_mask_mod)`、``or_masks(mask_mod, prefix_lm_mask_mod)`、`and_masks(mask_mod, self.get_rswa_mask_mod())`。**「一个 unbundled attention backend 用布尔运算拼装而成」这件事在生产系统里已经落地了。**

论文给的开销数字：FlexAttention 实现的 PagedAttention **平均 <1% 的运行时开销**，而 vLLM 自己的方法论 reported 20–26%（论文 §6.4）。页大小从 16 到 256 变化「没有显著影响」。

实验**第 7 组**用一个 4 页、随机乱序的页表 (`[2, 3, 0, 1]`) 验证了这个语义：物理布局 + 反查后的输出，与连续逻辑布局的输出相差 **1.877e-16**。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 232" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="232" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">PagedAttention：逻辑序列 → 物理页 → 反查回逻辑 index → 套用原 mask_mod</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">vLLM 的 final_mask_mod = is_valid &amp; logical_mask_mod(b, h, logical_q, logical_kv)（v0.31.0 :476-487）</text>

<text x="24" y="64" font-size="11" font-weight="600" fill="#1e40af">逻辑序列（连续）</text>
<g>
<rect x="24" y="72" width="56" height="22" rx="3" fill="#dbeafe" stroke="#60a5fa"/><text x="40" y="87" font-size="9.5" fill="#1e3a8a">L0</text>
<rect x="82" y="72" width="56" height="22" rx="3" fill="#dbeafe" stroke="#60a5fa"/><text x="98" y="87" font-size="9.5" fill="#1e3a8a">L1</text>
<rect x="140" y="72" width="56" height="22" rx="3" fill="#dbeafe" stroke="#60a5fa"/><text x="156" y="87" font-size="9.5" fill="#1e3a8a">L2</text>
<rect x="198" y="72" width="56" height="22" rx="3" fill="#dbeafe" stroke="#60a5fa"/><text x="214" y="87" font-size="9.5" fill="#1e3a8a">L3</text>
</g>
<text x="266" y="87" font-size="10.5" fill="#475569">页表 perm = [2, 3, 0, 1]</text>

<path d="M52 94 L52 104 L140 104 L140 116" stroke="#94a3b8" stroke-width="1" fill="none"/>
<path d="M110 94 L110 108 L198 108 L198 116" stroke="#94a3b8" stroke-width="1" fill="none"/>
<path d="M168 94 L168 112 L24 112 L24 116" stroke="#94a3b8" stroke-width="1" fill="none"/>
<path d="M226 94 L226 116 L82 116 L82 120" stroke="#94a3b8" stroke-width="1" fill="none"/>

<text x="24" y="138" font-size="11" font-weight="600" fill="#92400e">物理内存（乱序）</text>
<g>
<rect x="24" y="146" width="56" height="22" rx="3" fill="#fed7aa" stroke="#fb923c"/><text x="36" y="161" font-size="9.5" fill="#7c2d12">P0 = L2</text>
<rect x="82" y="146" width="56" height="22" rx="3" fill="#fed7aa" stroke="#fb923c"/><text x="94" y="161" font-size="9.5" fill="#7c2d12">P1 = L3</text>
<rect x="140" y="146" width="56" height="22" rx="3" fill="#fed7aa" stroke="#fb923c"/><text x="152" y="161" font-size="9.5" fill="#7c2d12">P2 = L0</text>
<rect x="198" y="146" width="56" height="22" rx="3" fill="#fed7aa" stroke="#fb923c"/><text x="210" y="161" font-size="9.5" fill="#7c2d12">P3 = L1</text>
</g>

<path d="M262 157 L300 157" stroke="#1d4ed8" stroke-width="1.3"/>
<path d="M294 153 L300 157 L294 161 Z" fill="#1d4ed8"/>

<rect x="304" y="126" width="188" height="62" rx="5" fill="#f8fafc" stroke="#94a3b8"/>
<text x="314" y="144" font-size="10.5" font-weight="600" fill="#0f172a">physical → logical 反查</text>
<text x="314" y="160" font-size="10" fill="#1e40af">block = kv_idx // P</text>
<text x="314" y="174" font-size="10" fill="#1e40af">offset = kv_idx % P</text>
<text x="314" y="186" font-size="10" fill="#1e40af">logical = ptbl[block]·P + offset</text>

<path d="M492 157 L520 157" stroke="#1d4ed8" stroke-width="1.3"/>
<path d="M514 153 L520 157 L514 161 Z" fill="#1d4ed8"/>

<rect x="524" y="126" width="132" height="62" rx="5" fill="#ecfdf5" stroke="#6ee7b7"/>
<text x="534" y="144" font-size="10.5" font-weight="600" fill="#14532d">logical_mask_mod</text>
<text x="534" y="160" font-size="10" fill="#166534">causal / sw / prefix</text>
<text x="534" y="176" font-size="10" fill="#166534">用户写的那一个</text>

<text x="304" y="206" font-size="10.5" fill="#374151">整套转换是 O(1) 的查表 + 一次整除/取余 —— 这就是为什么「融合间接访存」几乎免费：论文测到平均 &lt;1% 开销（评论：vLLM 的同类报告是 20–26%）。</text>
<text x="304" y="222" font-size="10.5" fill="#6b7280">实验第 7 组：物理布局输出 vs 逻辑布局输出 = 1.877e-16</text>
</svg>

---

## 10. 推理：FlexDecoding 与那个「必须是张量」的 offset

decode 阶段的形状是极端的：$$Q_{\rm LEN}$$ 很小（batch 推理时常常等于请求数），$$KV_{\rm LEN}$$ 很长（KV cache）。这时候「按 Q 分块 + 每个 block 扫完所有 KV」的策略会欠寄生：SM 数量远多于 query 行，大部分 SM 在空转。

解法是 **split-KV**（FlashDecoding）：把 KV 方向也切开，每个 SM 负责一段 KV 算出局部的 $$(m, l, \mathrm{acc})$$，最后做一次归约。PyTorch 的这个分支叫 FlexDecoding。

它的触发条件是明写在源码里的（`torch/_inductor/kernel/flex/flex_decoding.py:66-132`）：

| 条件 | 值 |
|:--|:--|
| query 长度 | $$<128$$（`guard_or_false`，不成立就退回主 kernel） |
| batch、head 数 | 必须**编译期静态可知** |
| GQA 比值 | 必须是 2 的幂 |
| BlockMask 的头维 | 必须是 1 或等于 query 头数（GQA 时） |
| `FORCE_USE_FLEX_ATTENTION` | 不能为 True |

grid 是三维的：`(num_block_m, batch*kv_heads, SPLIT_KV)`（`:138-146`）——第三维就是沿着 KV 切出来的份数。

### 10.1 offset 这个坑

decode 时新 token 的 query 位置在涨，所以 causal 里的 `q_idx >= kv_idx` 必须带上一个偏移量。天真的写法是把 offset 当 Python 整数：

```python
def causal_w_offset(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx + offset >= kv_idx, score, -float("inf"))
```

如果 `offset` 是 Python int，**每次 step 换一个值 → 重新 trace → 重新 compile**。官方博客给的表述是：这会把一次 2ms 的 decode step 变成一次几十秒的编译（[FlexAttention Part II: FlexAttention for Inference](https://pytorch.org/blog/flexattention-for-inference/)）。

正确写法是把它做成**张量**：

```python
offset = torch.tensor(i, device="cuda")

def causal_w_offset(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx + offset >= kv_idx, score, -float("inf"))

# 第一次调用会编译
out = flex_attn(q_tok, k_cache, v_cache, score_mod=causal_w_offset)
# 之后每次只需要：
offset.fill_(i + 1)      # 原地改值 —— 不用重编译
```

这个技巧和第 2 节那句「张量可以随便换，标量会被烤进去」是同一件事：**被捕获的张量走 load，被捕获的标量走常量折叠**。写代码的时候记得把「会变的量」都弄成张量。

### 10.2 BlockMask 在 decode 里怎么用

策略是「预构造一次大方块，然后取一行切片」：

```python
block_mask = create_block_mask(causal_mask, B=None, H=None,
                               Q_LEN=MAX_SEQ_LEN, KV_LEN=MAX_SEQ_LEN)
# 第 i 个 token：
block_offset = i // block_mask.BLOCK_SIZE[0]
block_mask_slice = block_mask[:, :, block_offset]     # 注意它切片后 mask_mod 会失效
```

源码里对此有一个硬性保护（`nn/attention/flex_attention.py:2486-2489`）：切片后的 BlockMask 的 mask_mod 会被替换成一个专门报错的桩，一旦有人想用它就抛 `RuntimeError("Cannot use mask_mod from a sliced BlockMask")`。

---

## 11. 2025–2026：边界在动

写这篇文章的时候（2026 年 10 月），FlexAttention 已经从「一个新 API」变成了「一条被承认的路」。三件大事改变了它的边界。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 214" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="214" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">FlexAttention 的时间线：从「一个 API」到「一条被承认的路」</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">上半是官方材料的发布时间，下半是这条时间线改变了什么边界</text>

<line x1="40" y1="86" x2="656" y2="86" stroke="#cbd5e1" stroke-width="2"/>
<g>
<circle cx="60" cy="86" r="6" fill="#2563eb"/><text x="48" y="106" font-size="10.5" fill="#1e40af">2024-12</text>
<circle cx="228" cy="86" r="6" fill="#2563eb"/><text x="206" y="106" font-size="10.5" fill="#1e40af">2024-12 / 2025</text>
<circle cx="424" cy="86" r="6" fill="#7c3aed"/><text x="400" y="106" font-size="10.5" fill="#6d28d9">2026-03（2.11）</text>
<circle cx="620" cy="86" r="6" fill="#16a34a"/><text x="596" y="106" font-size="10.5" fill="#15803d">2026-07（2.13）</text>
</g>

<rect x="24" y="116" width="176" height="66" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
<text x="34" y="134" font-size="11" font-weight="600" fill="#1e3a8a">论文 arXiv:2412.05496</text>
<text x="34" y="150" font-size="10" fill="#1e40af">score_mod / mask_mod</text>
<text x="34" y="164" font-size="10" fill="#1e40af">BlockMask 块稀疏</text>
<text x="34" y="176" font-size="10" fill="#1e40af">Triton 模板注入</text>

<rect x="212" y="116" width="184" height="66" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
<text x="222" y="134" font-size="11" font-weight="600" fill="#1e3a8a">博客（上/下两篇）</text>
<text x="222" y="150" font-size="10" fill="#1e40af">Part I：性能与 FAQ</text>
<text x="222" y="164" font-size="10" fill="#1e40af">Part II：FlexDecoding 与</text>
<text x="222" y="176" font-size="10" fill="#1e40af">decode offset 陷阱</text>

<rect x="408" y="116" width="184" height="66" rx="5" fill="#f5f3ff" stroke="#c4b5fd"/>
<text x="418" y="134" font-size="11" font-weight="600" fill="#5b21b6">FA4 后端（CuTeDSL）</text>
<text x="418" y="150" font-size="10" fill="#6d28d9">Hopper / Blackwell 上</text>
<text x="418" y="164" font-size="10" fill="#6d28d9">1.6–3.2× 于 Triton（fwd）</text>
<text x="418" y="176" font-size="10" fill="#6d28d9">1.85–2.3×（bwd）</text>

<rect x="604" y="116" width="72" height="66" rx="5" fill="#ecfdf5" stroke="#6ee7b7"/>
<text x="612" y="134" font-size="11" font-weight="600" fill="#14532d">2.13</text>
<text x="612" y="150" font-size="10" fill="#166534">Apple Silicon</text>
<text x="612" y="164" font-size="10" fill="#166534">最高 12×</text>
<text x="612" y="176" font-size="10" fill="#166534">确定性反向</text>

<text x="24" y="200" font-size="10.5" fill="#374151">同一套 score_mod / mask_mod 语法贯穿全程 —— 换后端、换硬件都不用改用户代码，这是「编程模型」这四个字的实际含义。</text>
</svg>

### 11.1 FlashAttention-4 后端（PyTorch 2.11，2026-03）

[PyTorch 官方博客](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)（Guessous, Stern, Hoehnerbach, Xie, Zadouri, Shah, Dao，2026-03-05）给出的信息密度极高，几条值得原文记住：

> “In practice, this has proven true: dozens of papers cite FlexAttention, and over a thousand repos have adopted it.”

> “At the time of the original blog post, we compared against FlashAttention-3 (FA3) on Hopper GPUs and were roughly at 80% of its performance. If you measure this today, FlexAttention achieves roughly 60% of FlashAttention-3's throughput, despite improvements to both implementations!”

第二句是非常少见的自我揭短，而且它解释了为什么必须换后端：**两边都在改进，差距反而变大**（Blackwell 上 Triton 表达不了 warp-specialized 的深度流水）。作者把它概括为

> “What was once a small gap has grown to a chasm!”

实现方式是把 FA4 用 CuTeDSL 重写并把用户的 score_mod 内联进去：

- 前向把 $$\mathbf S$$ tile 从 tensor memory 搬回寄存器，在那里应用 mod、算 row-wise max/sum、生成 $$\mathbf P$$ tile；
- 避开可变参数捕获，改传一个 `aux_tensors` 列表表示 mod 用到的全局内存区域；
- 反向用一个生成的 `score_mod_bwd`；但因为用户导数可能需要 pre-softmax score，标准的 FA4 liveness（$$\mathbf S$$ 与 $$\mathrm{d}\mathbf S$$ 不同时活跃）会被打破，代价是寄存器压力。

数字（同一篇博客 + PyTorch 2.11 release blog）：

| 对比对象 | forward | backward |
|:--|--:|--:|
| vs Triton 实现（计算受限 workload） | 1.2–3.2× | — |
| vs Triton 实现（GB200，SDPA 支持的 pattern） | 1.6–3.2× | 1.85–2.3× |
| vs cuDNN（forward） | 仍有差距，causal 尤其明显 | 部分 pattern 下能追平甚至超过 |

启用法：

```python
flex_flash = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}), dynamic=False
)
```

**限制**（同样来自官方）：

| 限制 | 说明 |
|:--|:--|
| PagedAttention | 暂不支持；FA4 在 Hopper 上是 128×128、Blackwell 上是 256×128 调参出来的，块大小不灵活 |
| 需要梯度的被捕获 buffer 的反向 | 不支持，请用 Triton 后端 |
| 块稀疏下的确定性反向 | 暂不支持（纯 score-mod 的 workload 是确定的） |
| 硬件 | 仅 Hopper / Blackwell |

### 11.2 Apple Silicon 与 Intel XPU（PyTorch 2.13，2026-07）

2.13 release blog 的标题句：「FlexAttention 登陆 Apple Silicon，带来了高达 12 倍的加速」。在源码层面这对应 `torch/_inductor/kernel/flex/flex_attention.py:186-199` 那个 `lower_mps` 分支 + `templates/` 里的 Metal 模板。这一类移植正是「编程模型与后端解耦」的收益：换硬件不用重写五年积累的注意力变体。

### 11.3 确定性反向（PyTorch 2.13）

前面 7.2 已经说了：`torch.use_deterministic_algorithms(True)` + `compute_dq_write_order`，开销 $$+0.2\%$$（$$S=32768$$）。

### 11.4 生态

落在真实系统里的证据：

- **vLLM**：`vllm/v1/attention/backends/flex_attention.py` 是完整的一等公民后端（锁定 `db9527a4`，1492 行），用 `torch.compile(create_block_mask, fullgraph=True, mode="reduce-overhead")` + `torch.compile(flex_attention, fullgraph=True)`，并把 `recompile_limit` 提到 16（默认小得多，说明它确实在赌编译次数）。
- **训练侧**：torchtune 的 sample packing 用 FlexAttention 替换掉那个 $$B\times H\times N\times N$$ 的布尔掩码，**吞吐 +71%**（[PyTorch 博客](https://pytorch.org/blog/flexattention/)）；论文 Table 4 的端到端数字是 torchtune 训练最高 2.4×、gpt-fast 推理在 LLaMA3.1-8B 上 1.22×–2.04×（70B 上 0.99×–1.66×）。
- **论文环节**：原文列举的 adoption 是「dozens of papers cite FlexAttention, and over a thousand repos have adopted it」。

---

## 12. 什么时候该用，怎么用

### 12.1 决策表

| 场景 | 建议 | 理由 |
|:--|:--|:--|
| 只有因果注意力、 MHA | 继续用 SDPA | SDPA 会派发到 FlashAttention/cuDNN，更快更稳 |
| 滑窗 / ALiBi / soft-cap / 文档掩码，且已在用 torch.compile | FlexAttention | 省掉一个 $$B\times H\times N\times N$$ 的 bool 掩码 + 省掉手写 kernel |
| 需要短时间试一种新 attention | FlexAttention | 几行 Py 就是上层用户 API 的全部成本 |
| 多个变体要叠加 | FlexAttention | `and_masks` / `or_masks`；手写路线是 64 个 kernel |
| 生产热点，必须拿到 FA3/FA4 的 98% 以上 | 先在 Hopper/Blackwell 上试 `BACKEND="FLASH"`，不行就手写 | 官方承认 Triton 版本只剩 ~60% FA3 |
| 序列很短（$$<512$$） | 不一定值得 | 编译开销与 block 额粒度都摊不开 |
| 需要长期维护的生产系统 | 谨慎 | API 至今还是 **prototype**（源码 `:2437-2440` 的 warning 仍在） |

### 12.2 落地清单（按踩坑频率排）

1. **必须 `torch.compile`**。否则会退化成材料化完整分数矩阵的实现，源码会警告你（`:2595-2605`）。
2. **把「会变的量」写成张量**。`offset`、阈值、缩放因子这类标量会被烤进 kernel，每个新值 = 一次重编译。
3. **缓存 BlockMask**。causal 就全局构造一次；document mask 就在模型入口构造一次穿层透传。别忘了 `B=None, H=None` 广播。
4. **掩码一定要用 mask_mod，不要用 score_mod 表达**。差 15–20%。
5. **注意 lse 的单位**。Triton 后端内部是 $$\log_2$$，PyTorch 会乘 $$\ln 2$$ 换回来；但如果 BACKEND 是 FLASH，那条换算**不做**（`:2592`）。
6. **GQA 要显式开 `enable_gqa=True`**，否则会因 $$H_q\ne H_{kv}$$ 直接报错（`:2451-2456`）。
7. **`create_block_mask` 内部会材料化一份完整的 bool 掩码**。它不便宜：要么编译它，要么自己写 `kv_indices`。
8. **审查时的 perf knobs**：`PRESCALE_QK`（更快，但数值误差更大）、`ROWS_GUARANTEED_SAFE`（省掉全掩行的判断）、`BLOCKS_ARE_CONTIGUOUS`（省掉逐块间接跳转）、`BLOCK_M/N`、`WRITE_DQ`。文档见 `FlexKernelOptions`（`:144-275`）。
9. **compile 环境变量/内存 bug 很难查**。官方博客列了几个常见 failure 与其对应的 kernel option，改之前先把 `torch._dynamo.config.recompile_limit` 调大一点观察重编译次数。

---

## 13. 收束

七条值得带走的结论：

1. **FlexAttention 的定义只有一个式子**，位置比内容重要：$$\mathrm{mod}$$ 必须在 softmax 之前、在 $$\max$$ 之前。这个位置决定了它为什么能和 online softmax 共存。
2. **为什么共存得了**：因为运行最大值 $$m_i$$ 是在**修正之后**的分数上取的，而 $$\alpha$$-重标定能把任何旧贡献整体缩到新基准。六种变体的流式与一次性计算相差 1.776e-15。
3. **为什么 $$m_i$$ 要取在那里**：不是数学（对任意参考值都成立），是数值——$$+160$$ 的常数偏置会把 $$\log_2$$ 指数顶到 230.83，超过 float32 的 $$2^{127}$$。
4. **mask 单列成 mask_mod 是为了稀疏性**：块级跳过 + 全块免掩码，比逐元素掩码省 15–20%。
5. **BlockMask 是有损压缩**：causal 从 50.1% 的元素密度涨到 56.2% 的块级密度；文档掩码更惨，20.8% → 40.6%。选 $$BS$$ 就是在「省元数据」和「浪费算力」之间选。
6. **表达力边界是三条**：依赖整行统计量的修正（流式差 4.365e-02）、作用在 softmax 之后/分母上的修正、以及当前不支持的元素级真实稀疏。
7. **反向是自动的，但不是「从整个注意力」自动**：PyTorch 只对 score_mod 跑一次 AOTAutograd 的 joint graph，然后把它塞进手写反向模板——这个分工才是「既灵活又够快」的原因。

最后放一句官方在 2026 年自己说的话，它比任何宣传语都更能说明这个模型的位置：

> “A common pattern emerges: researchers experiment with Flex, find something that works, then hit a wall when performance becomes critical. At the time of the original blog post, we compared against FlashAttention-3 (FA3) on Hopper GPUs and were roughly at 80% of its performance. If you measure this today, FlexAttention achieves roughly 60% of FlashAttention-3's throughput.”

FlexAttention 不是终点，它是「先让它跑起来、先别管它有多快」那一层。真正的价值在于：**当你终于要写那个 CUDA/Triton kernel 的时候，你已经确切知道你要写的是什么了。**

下一篇（02）写 BlockMask 的另一半：data-dependent 稀疏每步重建的代价、`compute_dq_write_order` 的确定性实现细节、以及如何为一个真实的长上下文任务设计 $$BS$$ 与 GSQ 的组合。

---

## 参考

**论文与官方博客**

- Dong, Feng, Guessous, Liang & He, *Flex Attention: A Programming Model for Generating Optimized Attention Kernels*, 2024. [arXiv:2412.05496](https://arxiv.org/abs/2412.05496) · [DOI 10.48550/arXiv.2412.05496](https://doi.org/10.48550/arXiv.2412.05496)
- PyTorch Team, *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*, 2024. [pytorch.org/blog/flexattention](https://pytorch.org/blog/flexattention/)
- PyTorch Team, *FlexAttention Part II: FlexAttention for Inference*, [pytorch.org/blog/flexattention-for-inference/](https://pytorch.org/blog/flexattention-for-inference/)
- Guessous, Stern, Hoehnerbach, Xie, Zadouri, Shah & Dao, *FlexAttention + FlashAttention-4: Fast and Flexible*, 2026-03-05. [pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- *PyTorch 2.11 Release Blog*（FA4 backend / MPS / XPUGraph），[pytorch.org/blog/pytorch-2-11-release-blog/](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- *PyTorch 2.13 Release Blog*（Apple Silicon 12×、确定性反向 $$+0.2\%$$），[pytorch.org/blog/pytorch-2-13-release-blog/](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- PyTorch 文档：`torch.nn.attention.flex_attention`，[docs.pytorch.org/docs/stable/nn.attention.flex_attention.html](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)

**前置工作**

- Dao, Fu, Ermon, Rudra & Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Press, Smith & Lewis, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi), ICLR 2022. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Beltagy, Peters & Cohan, *Longformer: The Long-Document Transformer*, 2020. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang & Stoica, *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Gemma Team, *Gemma 2: Improving Open Language Models at a Practical Size*（logit soft-capping），2024. [arXiv:2408.00118](https://arxiv.org/abs/2408.00118)
- Wu, Wang, Xiao, Zheng, Zhang & Chen, *Mirage: A Multi-Level Superoptimizer for Tensor Programs*（论文里用来对比「为什么通用编译器生成不了 attention」），MLSys 2025. [arXiv:2405.05751](https://arxiv.org/abs/2405.05751)

**源码（行号锁定）**

- PyTorch `v2.14.1` = [`5c488690`](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070)：
  [`torch/nn/attention/flex_attention.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/attention/flex_attention.py)（`:2365` flex_attention、`:1967` create_block_mask、`:855` BlockMask、`:315` 改函数类型判别、`:1759`/`:1773` 组合）·
  [`torch/_higher_order_ops/flex_attention.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_higher_order_ops/flex_attention.py)（`:408` trace、`:646` 反向图生成）·
  [`torch/_inductor/kernel/flex/flex_attention.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/flex/flex_attention.py)（`:155` lowering、`:145` 模板声明）·
  [`torch/_inductor/kernel/flex/flex_decoding.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/flex/flex_decoding.py)（`:66` 分支条件）·
  模板 [`flex_attention.py.jinja`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/flex/templates/flex_attention.py.jinja) 与 [`common.py.jinja`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/flex/templates/common.py.jinja)
- vLLM `v0.31.0` = [`db9527a4`](https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6)：[`vllm/v1/attention/backends/flex_attention.py`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flex_attention.py)（`:54` compile 设置、`:427` physical→logical、`:625` get_mask_mod）

**本站配套**

- [编译器知识体系深度解析（01）](/2026/08/25/compiler-01-lexing-parsing/) —— torch.compile / Triton 这套工程三件套落在编译原理的哪一层
- [算子开发与优化（01）](/2026/09/03/op-01-hardware/) —— 为什么「先算 Roofline 上界」是写任何 kernel 之前的第一步
- 本实验：[`ipynbs` `experiments/flex-attention/flex_core/`](https://github.com/lrypcy/ipynbs/tree/5954fb521a279b997005a6f514144c3b380c8e36/experiments/flex-attention/flex_core/)（commit `5954fb5`，27 条断言全过）
