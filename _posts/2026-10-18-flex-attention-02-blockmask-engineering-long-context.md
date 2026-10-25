---
title: "FlexAttention（02）：BlockMask 的工程与长上下文——构造成本、确定性 dQ、粒度选型与上下文并行"
date: 2026-10-18 22:10:00 +0800
categories:
  - 编译器技术
tags: [flexattention, blockmask, block-sparse, context-parallel, ptrr, load-balance, deterministic, dq-write-order, gqa, torch-compile, kv-cache, long-context]
layout: post
mathjax: true
excerpt: "FlexAttention 的数学在 01 那篇已经讲完，剩下的是工程：create_block_mask 内部真的会物化一份 Q×KV 的布尔张量（8×16×8192² 是 8192 MiB），每步还是每层重建掩码的阈值不到 2%，确定性 dQ 靠的是一张 rank 表把累加顺序钉死（按 rank 累加与串行累加逐位相等），块大小选错了滑窗掩码会多算 25% 的元素，以及上下文并行里 BlockMask 的 kv_num_blocks 直接就是负载均衡的代价模型（PTRR 把 causal 的不均衡从 1.875 压到 1.000）。配套 numpy 实验五组 19 条断言。"
---

> **系列导航** ｜ 本系列承接 [编译器知识体系深度解析](/2026/08/25/compiler-01-lexing-parsing/) 与 [算子开发与优化](/2026/09/03/op-01-hardware/)，讲编译器与算子的交界。
> 上一篇：[FlexAttention（01）：把注意力变体编译成一个 FlashAttention 内核](/2026/10/17/flex-attention-01-programming-model/)（online softmax 为什么能吃下任意逐点修正）｜ 下一篇：03 torch.compile 与后端调度（autotune、max-autotune、CUDA graph、重编译治理）（待写）

> **TL;DR**
>
> * **`create_block_mask` 不是免费的，而且它的成本结构很反直觉**。它内部先 `create_mask` 物化一份 $$B\times H\times Q\times KV$$ 的**布尔张量**，再降采样成块级。8×16×8192² 是 **8192 MiB**；`B=None, H=None` 广播之后降到 **64 MiB**（128×）；而压缩完的 BlockMask 元数据只有 **0.0159 MiB**（516222×）。
> * **官方推荐 `B=None, H=None` 的第一个理由不是省时间，是省这份峰值显存。**
> * **「每层重建」在纯 FLOPs 口径下几乎是划算的**：一次重建相对一次完整注意力的成本是 $$\kappa/(4D)$$，$$\kappa=4$$ 时也只有 **1.562%**；三种掩码的净收益都是 +45%~+73%。它之所以被官方列为「最难的场景」，难在**另外三件事**——物化的 bool 峰值、独立的 kernel 序列、动态 shape 触发的重编译。
> * **确定性 dQ 修的是浮点非结合性，不是竞态逻辑**。float32 下同一组部分和换顺序就有差别（升/降序差 7.153e-07，64 个随机顺序最大 1.192e-06）。PyTorch 的做法是预先算一张 rank 表（$$\mathrm{rank}=\mathrm{cumsum}(\mathrm{dense})-\mathrm{dense}$$）把「谁先写」钉死：按 rank 累加与串行累加**逐位相等**。
> * **块粒度选错很贵**：滑窗 $$w=1024$$ 在 $$BS=64/128/256$$ 下的浪费率是 6.24% / 12.49% / **24.99%**。但**文档长度对齐块边界时浪费率恒为 0.00%**——元素密度与块级密度逐位相等。
> * **上下文并行里 BlockMask 还是一份成本账**：`kv_num_blocks + full_kv_num_blocks` 就是每个 query 块的「处理时间」。朴素连续切分在 causal + CP=8 下 max/理想 = **1.875**（峰谷差 224），PTRR 压到 **1.000**；而且 head-tail 只对单调代价曲线成立（document 掩码上退化到 1.333），PTRR 是通用的。
> * **CP + FlexAttention 有一条硬约束**：`Q_LEN` 必须能被 $$cp\_world\_size \times BLOCK\_SIZE$$ 整除，否则直接 `NotImplementedError`——因为 local→global 的索引换算发生在**块粒度**。

配套实验：[`ipynbs` commit `5d22352`](https://github.com/lrypcy/ipynbs/tree/5d223527089c816f464f37b345af35ce2b778e0f/experiments/flex-attention/blockmask_lab/) 下的 `blockmask_lab/`，纯 numpy、CPU 可跑、固定 seed `20261018`，19 条断言全过；本文所有自测数字都来自它的实际输出（[`results/summary.md`](https://github.com/lrypcy/ipynbs/blob/5d223527089c816f464f37b345af35ce2b778e0f/experiments/flex-attention/blockmask_lab/results/summary.md)）。01 那篇的实验在同目录 [`flex_core/`](https://github.com/lrypcy/ipynbs/tree/5954fb521a279b997005a6f514144c3b380c8e36/experiments/flex-attention/flex_core/)。

**本文锁定的版本**：PyTorch `v2.14.1` = commit [`5c488690`](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070)；vLLM `v0.31.0` = commit [`db9527a4`](https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6)。

---

## 0. 符号约定

| 符号 | 含义 | 备注 |
|---:|:---|:---|
| $$BS$$ | BlockMask 的块边长 | 默认 128（源码 `_DEFAULT_SPARSE_BLOCK_SIZE`） |
| $$B_Q,B_N$$ | kernel 的 tile 边长 `BLOCK_M` / `BLOCK_N` | **不是** $$BS$$，两者是整除关系 |
| $$N,Q_{\rm LEN},KV_{\rm LEN}$$ | 序列长 | 本文默认 $$Q_{\rm LEN}=KV_{\rm LEN}=N$$ |
| $$D$$ | query-key 头维 | 源码 `E` |
| $$\kappa$$ | 一次 mask_mod 求值的 op 数 | 纯比较 $$\kappa\approx1$$，带算术 $$\kappa\approx4$$ |
| $$L$$ | 层数 | 摊销的分母 |
| $$CP$$ | 上下文并行度（world size） | |
| $$n_b$$ | 块数 | $$n_b=\lceil N/BS\rceil$$ |
| $$\mathrm{cost}[m]$$ | 第 $$m$$ 个 query 块的处理时间 | $$=\mathrm{kv\_num\_blocks}[m]+\mathrm{full\_kv\_num\_blocks}[m]$$ |
| $$r$$ | CP rank 编号 | |

---

## 1. 01 留下的四个问题

01 把 FlexAttention 的**数学**讲完了：为什么任意逐点 `score_mod` 能和 online softmax 的 $$\alpha$$-重标定共存，为什么运行最大值必须取在修正之后，反向是怎么用 AOTAutograd 的 joint graph 自动出来的。

但决定「要不要上 FlexAttention」的其实是另一批问题，而且官方材料对它们只给了**定性**建议：

1. `create_block_mask` 到底做了什么？官方只说「it is unfortunately fairly expensive」，没说贵在哪。
2. 掩码该多久重建一次？官方给了三种模式（不变 / 每 batch / 每层），没给阈值。
3. 确定性反向到底修的是什么？「预先算好的写入顺序」长什么样？
4. $$BS$$ 该选多大？和 tile 大小、GQA 比值怎么配合？

这四个问题的答案都藏在 BlockMask 这个数据结构里。而 BlockMask 最妙的一点是：**它不只是一份掩码，还是一份成本账**——第 8 节会把这件事讲透。

---

## 2. `create_block_mask` 到底做了什么

### 2.1 三段流水线

源码路径（`nn/attention/flex_attention.py`）：

```
create_block_mask(mod, B, H, Q_LEN, KV_LEN, ...)          # :1967
   └─ create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device)  # :2063 → :1828
        材料化一个 (B, H, Q_LEN, KV_LEN) 的布尔张量
   └─ _convert_mask_to_block_mask(mask_tensor, ...)       # :2064 → :1706
        降采样成块级：(B, H, n_b, n_b)，并分出 full / partial
   └─ _create_sparse_block_from_block_mask(...)           # :2070 → :1802
        压成 (kv_num_blocks, kv_indices) 两个索引张量
```

关键在于**第 1 步**。看 `create_mask` 的实现（`:1860-1876`，简化）：

```python
def create_mask(mod_fn, B, H, Q_LEN, KV_LEN, device="cuda"):
    if B is None: B = 1
    if H is None: H = 1
    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, Q_LEN, device=device)
    n = torch.arange(0, KV_LEN, device=device)
    ...
    mask = mask_mod(b, h, m, n)      # ← 广播出完整的 (B, H, Q_LEN, KV_LEN)
    return mask
```

而 `create_block_mask` 在调用它之前就已经把 `None` 折成了 1（`:2030-2033`）：

```python
if B is None:
    B = 1
if H is None:
    H = 1
```

所以 **`B=None, H=None` 的收益不是在元数据上（元数据本来就广播），而是在这份中间张量上**。这解释了官方 FAQ 那句话的真正含义：

> “create_block_mask is unfortunately fairly expensive, both from a memory and compute perspective, as determining whether a block is completely sparse requires evaluating mask_mod at **every single point in the block**.”

（[PyTorch 博客 FAQ](https://pytorch.org/blog/flexattention/)）

「every single point in the block」——判定一个块是否全空，需要把这个块里 $$BS^2$$ 个点全跑一遍。所以 mask_mod 的求值次数是 $$Q_{\rm LEN}\times KV_{\rm LEN}$$，跟块大小无关。

### 2.2 峰值显存这笔账

实验**第 1 组**按源码路径算出了具体的字节数（bool 按 1 字节、int32 索引）：

| 配置 | $$B$$ | $$H$$ | $$Q=KV$$ | 物化的 bool 掩码 | BlockMask 元数据 | 比值 |
|:--|--:|--:|--:|--:|--:|--:|
| 8B 训练 / 不广播 | 8 | 16 | 8192 | **8192.00 MiB** | 0.0159 MiB | 516222× |
| 8B 训练 / 广播 H | 8 | 1 | 8192 | 512.00 MiB | 0.0159 MiB | 32264× |
| 8B 训练 / 广播 B,H | 1 | 1 | 8192 | **64.00 MiB** | 0.0159 MiB | 4033× |
| 长文训练 | 1 | 1 | 32768 | 1024.00 MiB | 0.2510 MiB | 4080× |
| 长文训练 / 不广播 | 4 | 8 | 32768 | **32768.00 MiB** | 0.2510 MiB | 130562× |

三个数量级的事实：

- **元数据本身小到可以忽略**（几万分之一）；
- **但构造过程会在峰值上放一份 GB 级的 bool 张量**——不广播的 8×16×8192 就是 8 GiB；
- **`B=None, H=None` 把这份峰值砍掉 128 倍**，这是它成为官方首推的真正原因。

元数据随 $$BS$$ 的平方下降（$$N=32768$$）：

| $$BS$$ | 元数据 |
|--:|--:|
| 64 | 1026.00 KiB |
| 128 | 257.00 KiB |
| 256 | 64.50 KiB |
| 1024 | 4.12 KiB |

$$BS$$ 从 64 提到 1024（16×）→ 元数据降 **249×**（差在 $$n_b$$ 的取整）。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 268" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="268" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">create_block_mask 的三段流水线：峰值出现在第一段</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">元数据是压缩后的成本，但构造过程里那份稠密 bool 张量是真实存在的峰值</text>

<rect x="24" y="56" width="192" height="72" rx="5" fill="#fef2f2" stroke="#fca5a5"/>
<text x="34" y="76" font-size="11.5" font-weight="600" fill="#991b1b">① create_mask</text>
<text x="34" y="94" font-size="10.5" fill="#7f1d1d">物化 (B,H,Q,KV) 的 bool</text>
<text x="34" y="110" font-size="10.5" fill="#7f1d1d">mask_mod 求值 Q·KV 次</text>
<text x="34" y="124" font-size="10.5" font-weight="600" fill="#991b1b">← 峰值在这里</text>

<path d="M216 92 L236 92" stroke="#64748b" stroke-width="1.3"/>
<path d="M230 88 L236 92 L230 96 Z" fill="#64748b"/>

<rect x="240" y="56" width="192" height="72" rx="5" fill="#fff7ed" stroke="#fdba74"/>
<text x="250" y="76" font-size="11.5" font-weight="600" fill="#9a3412">② 降采样到块级</text>
<text x="250" y="94" font-size="10.5" fill="#7c2d12">(B,H,⌈Q/BS⌉,⌈KV/BS⌉)</text>
<text x="250" y="110" font-size="10.5" fill="#7c2d12">分出 full / partial</text>

<path d="M432 92 L452 92" stroke="#64748b" stroke-width="1.3"/>
<path d="M446 88 L452 92 L446 96 Z" fill="#64748b"/>

<rect x="456" y="56" width="200" height="72" rx="5" fill="#ecfdf5" stroke="#6ee7b7"/>
<text x="466" y="76" font-size="11.5" font-weight="600" fill="#14532d">③ 压成索引张量</text>
<text x="466" y="94" font-size="10.5" fill="#166534">kv_num_blocks + kv_indices</text>
<text x="466" y="110" font-size="10.5" fill="#166534">（+ 反向用的 q_* 两份）</text>

<text x="24" y="152" font-size="12" font-weight="600" fill="#111827">同一组输入的三档字节数（8×16×8192²）</text>
<rect x="24" y="162" width="620" height="18" rx="3" fill="#dc2626"/>
<text x="32" y="176" font-size="10.5" fill="#ffffff">物化的 bool（不广播）            8192.00 MiB  =  8.0 GiB</text>
<rect x="24" y="184" width="38.75" height="18" rx="3" fill="#f59e0b"/>
<text x="76" y="198" font-size="10.5" fill="#92400e">广播 H → 512.00 MiB</text>
<rect x="24" y="206" width="4.84" height="18" rx="3" fill="#16a34a"/>
<text x="76" y="220" font-size="10.5" fill="#14532d">广播 B,H → 64.00 MiB</text>
<rect x="24" y="228" width="1.2" height="18" rx="3" fill="#2563eb"/>
<text x="76" y="242" font-size="10.5" fill="#1e40af">BlockMask 元数据 → 0.0159 MiB（条形已缩到不可见，实际非零）</text>
<text x="24" y="262" font-size="10.5" fill="#374151">注：条形按 log 之外的线性比例画，最后一档只有 16 KiB，画不出来——这正是要点。</text>
</svg>

### 2.3 三条官方对策，以及它们的现状

官方 FAQ 给了三条，按推荐度排。**注意第 2 条在 2.14.1 已经变了**：

**① 跨 batch / head 广播**。`create_block_mask(mod, B=None, H=None, ...)`。绝大多数掩码（causal、滑窗）与 batch 和头无关。

**② 编译它**。FAQ 原文写的是「`torch.compile` 因为一些限制不能直接作用在 `create_block_mask` 上，请用 `_compile=True`」。**这个限制已经被修复了**——2.14.1 里 `_compile=True` 会先抛 `DeprecationWarning`（`:2040-2045`）：

> “_compile flag on create_block_mask was originally added to work around a torch.compile limitation. That limitation has since been addressed. So, to compile create_block_mask, we suggest doing torch.compile(create_block_mask).”

真实系统已经在用新写法了（vLLM `db9527a4`：`:55-57`）：

```python
create_block_mask_compiled = torch.compile(
    create_block_mask, fullgraph=True, mode="reduce-overhead"
)
```

**③ 自己写构造函数**。真的在意就别走 mask_mod，直接构造 `kv_num_blocks` 与 `kv_indices`。01 那篇提到过这一点，第 9 节会给一个能跑的写法。

---

## 3. 三种缓存策略，以及「每层重建」到底难在哪

官方 FAQ 给的分类是：

| 掩码的变化频率 | 建议 |
|:--|:--|
| 不变（causal） | 全局预构造一次，`functools.partial(flex_attention, block_mask=bm)` |
| 每 batch 变（文档掩码） | 模型入口构造一次，**穿层透传**，让所有层摊销 |
| 每层都变（data-dependent 稀疏） | 「the hardest setting」 |

第三行那句 “the hardest setting” 需要一个量化解释。实验**第 2 组**做了这件事。

### 3.1 把两笔成本放在同一把尺子上

一次注意力的算力（两次 matmul，每次 $$2QKV D$$ 个 MAC）：

$$
C_{\rm attn}=4\,Q_{\rm LEN}\,KV_{\rm LEN}\,D
$$

一次重建的成本：mask_mod 求值 $$Q_{\rm LEN}KV_{\rm LEN}$$ 次，每次 $$\kappa$$ 个 op；再加块级归约。相对注意力：

$$
\frac{C_{\rm build}}{C_{\rm attn}}\approx\frac{\kappa}{4D}
$$

$$D=64$$ 时：

| $$\kappa$$ | 单次重建 / 一次注意力 | 每步建一次、32 层复用 | 每层各建一次 |
|:--|--:|--:|--:|
| 1（纯比较） | 0.391% | 0.0122% | 0.391% |
| 4（带算术） | 1.562% | 0.0488% | 1.562% |

### 3.2 break-even：阈值不到 2%

重建划算的条件是「跳过的算力 > 重建成本」，即

$$
\text{skip\_ratio} > \frac{\kappa}{4D}
$$

$$D=64$$、$$\kappa=4$$ 时阈值是 **1.562%**——也就是说**只要掩码能让超过 1.5% 的块被跳过，每层重建就回本**。三种实际掩码的净收益（$$\kappa=4$$）：

| 掩码 | 块级稀疏度（跳过比例） | 扣掉重建后净收益 |
|:--|--:|--:|
| causal | 46.9% | **+45.3%** |
| sliding $$w=512$$ | 72.7% | **+71.1%** |
| document | 74.2% | **+72.7%** |

**结论：纯 FLOPs 口径下，「每层重建」根本不是问题。**

那官方为什么说它最难？因为真实成本不在 FLOPs，在这三处：

1. **物化的 bool 峰值显存**（第 2 节）：每层重建意味着这份峰值每层都要出现一次，而且无法用「构造完就释放」以外的办法躲开；
2. **独立的 kernel 序列**：每次 `create_block_mask` 是一串独立的 GPU kernel（构造索引张量、arange、比较、归约），不是融合进注意力里的；它的**launch 开销与同步开销**在 FLOPs 账上看不见；
3. **动态 shape 触发重编译**：如果每层的掩码形状不同，或者序列长度在变，`torch.compile` 会为每个新形状重新编译——这才是真正的杀手（vLLM 把 `recompile_limit` 从默认提到 16，`db9527a4:54`，就是在赌这件事）。

所以正确的读法是：

> **「每层重建」难的从来不是算力，是显存峰值 + kernel 序列 + 重编译。**

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 214" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="214" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">三种缓存策略：同一份构造成本，三种摊销方式</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">横轴是一次前向的 L 层；深蓝是构造，浅蓝是复用的收益</text>

<text x="24" y="66" font-size="11.5" font-weight="600" fill="#14532d">① 不变（causal）：全局构造一次</text>
<rect x="190" y="54" width="26" height="16" rx="2" fill="#1d4ed8"/>
<rect x="216" y="54" width="440" height="16" rx="2" fill="#dbeafe"/>
<text x="470" y="66" font-size="10" fill="#1e40af">32 层全部复用 → 摊到 0.05%</text>

<text x="24" y="98" font-size="11.5" font-weight="600" fill="#14532d">② 每 batch 变（文档掩码）：入口构造，穿层透传</text>
<rect x="300" y="86" width="26" height="16" rx="2" fill="#1d4ed8"/>
<rect x="326" y="86" width="330" height="16" rx="2" fill="#dbeafe"/>
<text x="470" y="98" font-size="10" fill="#1e40af">同样 32 层复用，但要每 step 重来</text>

<text x="24" y="130" font-size="11.5" font-weight="600" fill="#b91c1c">③ 每层都变（data-dependent）：无从摊销</text>
<g>
<rect x="190" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="230" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="270" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="310" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="350" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="390" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="430" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="470" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="510" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="550" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="590" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
<rect x="630" y="118" width="26" height="16" rx="2" fill="#dc2626"/>
</g>
<text x="470" y="130" font-size="10" fill="#991b1b">每层一份构造 → 纯粹的 L 倍</text>

<rect x="24" y="148" width="632" height="46" rx="4" fill="#fff7ed" stroke="#fdba74"/>
<text x="34" y="166" font-size="10.5" fill="#7c2d12">但即便如此，FLOPs 口径下仍然划算：净收益 = 跳过比例 − κ/(4D) = 46.9% − 1.562% = +45.3%（causal）。</text>
<text x="34" y="184" font-size="10.5" fill="#7c2d12">真正让 ③ 变难的是另外三笔：每层一次 GB 级 bool 峰值 · 每层一串独立 kernel · 形状一变就重编译。</text>

<text x="24" y="210" font-size="10.5" fill="#374151">官方原话：“This is the hardest setting, since we're unable to amortize the block mask computation across multiple FlexAttention invocations.”</text>
</svg>

---

## 4. 确定性 dQ：修的不是竞态，是浮点

### 4.1 问题长什么样

反向的 $$dQ$$ 是所有 KV 块贡献的**和**。多个块（或多个 split）要往同一个 $$dQ$$ 位置累加，源码默认用原子操作。原子操作本身没有逻辑错误——但**浮点加法不满足结合律**：

$$
(a+b)+c \neq a+(b+c)\quad\text{在 float32 下是真的}
$$

于是「哪个块先到」会影响最后一位。实验**第 3 组**把这个量了出来（float32，8 个 query 块 × 16 个 KV 块）：

| 对比 | 最大偏差 |
|:--|--:|
| 升序串行 vs 降序串行 | 7.153e-07 |
| 64 个随机顺序 vs 升序（最大 / 均值） | 1.192e-06 / 6.049e-07 |

这些数字看着小，但它们的后果是**同一次训练两次跑不出逐位相同的结果**——回归测试、数值 bisect、可复现研究全部失效。

### 4.2 PyTorch 的解法：一张 rank 表

2.13 的做法（源码 `_compute_dq_write_order_from_block_mask`，`nn/attention/flex_attention.py:1879-1964`）是**把累加顺序提前算好**，而不是事后去重。算法三步：

**第 1 步**，把 partial 与 full 两份块列表合并成一个稠密的「贡献者矩阵」，并做前缀和：

```python
dense_partial = _ordered_to_dense(kv_num_blocks, kv_indices, num_n)
if has_full:
    dense_full = _ordered_to_dense(full_kv_num_blocks, full_kv_indices, num_n)
    dense = (dense_partial + dense_full).clamp(max=1)     # :1939-1940
...
cumsum = dense.cumsum(dim=-1)
rank_table = (cumsum - dense).to(torch.int32)             # :1944-1945
```

**第 2 步**，可选地把顺序反过来（`dq_kv_order=True`，即 SPT）：

```python
if dq_kv_order:
    total_per_m = cumsum[:, :, :, -1:]
    rank_table = (total_per_m - 1 - rank_table).to(torch.int32)   # :1947-1949
```

**第 3 步**，按反向遍历顺序（沿 Q 方向的 `q_indices`）把 rank 收集出来：

```python
def _gather_write_order(bwd_idx: Tensor) -> Tensor:
    ...
    m_vals = bwd_idx.long().clamp(0, num_m - 1)
    return rank_table[b_i, h_i, m_vals, n_i].to(torch.int32)      # :1956

dq_write_order = _gather_write_order(q_indices)                   # :1958
```

于是 `rank_table[m, n]` 的含义非常干净：

$$
\mathrm{rank}[m,n]=\#\{\,n'<n : \text{块 }(m,n')\text{ 有贡献}\,\}
$$

也就是「在按 $$n$$ 升序的顺序里，这一对排第几」。kernel 拿它当 semaphore 值，累加顺序就被钉死了。

实验**第 3 组**在 numpy 里复现了这个算法并验证：

| 验证 | 结果 |
|:--|--:|
| $$\mathrm{rank}_{\rm asc}[m,n] =$$ 「$$n$$ 之前的贡献者数」 | 逐对通过 |
| 按 $$\mathrm{rank}_{\rm asc}$$ 累加 vs 串行升序 | **0.000e+00**（逐位相等） |
| 按 $$\mathrm{rank}_{\rm spt}$$ 累加 vs 串行降序 | **0.000e+00**（逐位相等） |

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 236" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="236" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">确定性 dQ 的 rank 表：把「谁先写」提前算成一张整数表</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">rank[m,n] = n 之前有多少个块对第 m 行有贡献；按它累加 == 串行升序累加（逐位）</text>

<text x="24" y="66" font-size="11" font-weight="600" fill="#1e3a8a">dense（贡献者矩阵，1 表示 (m,n) 要算）</text>
<g>
<rect x="24" y="74" width="26" height="20" fill="#1d4ed8"/><rect x="50" y="74" width="26" height="20" fill="#e5e7eb"/><rect x="76" y="74" width="26" height="20" fill="#1d4ed8"/><rect x="102" y="74" width="26" height="20" fill="#e5e7eb"/><rect x="128" y="74" width="26" height="20" fill="#1d4ed8"/><rect x="154" y="74" width="26" height="20" fill="#1d4ed8"/><rect x="180" y="74" width="26" height="20" fill="#e5e7eb"/><rect x="206" y="74" width="26" height="20" fill="#1d4ed8"/>
<text x="240" y="89" font-size="10.5" fill="#374151">第 0 行：n = 0, 2, 4, 5, 7</text>
</g>

<path d="M120 100 L120 112" stroke="#64748b" stroke-width="1.2"/>
<path d="M116 106 L120 112 L124 106 Z" fill="#64748b"/>
<text x="130" y="122" font-size="10.5" fill="#475569">cumsum → (1,1,2,2,3,4,4,5)</text>

<path d="M280 122 L300 122" stroke="#64748b" stroke-width="1.2"/>
<path d="M294 118 L300 122 L294 126 Z" fill="#64748b"/>

<text x="310" y="112" font-size="11" font-weight="600" fill="#14532d">rank = cumsum − dense</text>
<g>
<rect x="310" y="74" width="26" height="20" fill="#16a34a"/><rect x="336" y="74" width="26" height="20" fill="#e5e7eb"/><rect x="362" y="74" width="26" height="20" fill="#16a34a"/><rect x="388" y="74" width="26" height="20" fill="#e5e7eb"/><rect x="414" y="74" width="26" height="20" fill="#16a34a"/><rect x="440" y="74" width="26" height="20" fill="#16a34a"/><rect x="466" y="74" width="26" height="20" fill="#e5e7eb"/><rect x="492" y="74" width="26" height="20" fill="#16a34a"/>
<text x="312" y="89" font-size="9.5" fill="#ffffff">0</text><text x="364" y="89" font-size="9.5" fill="#ffffff">1</text><text x="416" y="89" font-size="9.5" fill="#ffffff">2</text><text x="442" y="89" font-size="9.5" fill="#ffffff">3</text><text x="494" y="89" font-size="9.5" fill="#ffffff">4</text>
</g>
<text x="526" y="89" font-size="10.5" fill="#374151">→ 该行的累加顺序 0→1→2→3→4</text>

<rect x="24" y="140" width="632" height="40" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="34" y="158" font-size="10.5" fill="#334155">dq_kv_order=True（SPT）：rank ← (该行总数 − 1 − rank)，把顺序反过来。两种都是确定的，只是基准不同。</text>
<text x="34" y="174" font-size="10.5" fill="#334155">实验：按 rank_asc 累加 vs 串行升序 = 0.000e+00；按 rank_spt 累加 vs 串行降序 = 0.000e+00</text>

<rect x="24" y="188" width="308" height="34" rx="4" fill="#fef2f2" stroke="#fecaca"/>
<text x="34" y="206" font-size="10.5" fill="#991b1b">非确定性来源：升/降序差 7.153e-07，64 个随机顺序最大 1.192e-06</text>
<rect x="348" y="188" width="308" height="34" rx="4" fill="#ecfdf5" stroke="#a7f3d0"/>
<text x="358" y="206" font-size="10.5" fill="#14532d">2.13 官方成本：S=32768 时 create_block_mask 端到端 +0.2%</text>
</svg>

### 4.3 怎么开，以及当前的边界

```python
torch.use_deterministic_algorithms(True)

block_mask = create_block_mask(
    mask_mod, B, H, Q_LEN, KV_LEN,
    compute_dq_write_order=True,     # :1977
    dq_kv_order=True,                # :1978  False=升序，True=降序/SPT
)
```

三条要记住的边界：

1. **这是 Flash 后端专有的**（`compute_dq_write_order` 的 docstring：`needed by deterministic block-sparse FLASH backward`，`:1998-1999`）。
2. **需要 `q_num_blocks` / `q_indices`**，否则直接报错（`:1902-1905`）——而这两个是 `BlockMask` 自动生成或由调用方提供的。
3. **FA4 后端在块稀疏下还没做**：

   > “**Deterministic backward with block-sparsity:** The Flash backend's backward pass is not yet deterministic when block-sparsity is enabled (score-mod-only workloads are deterministic).”

   （[FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)）

成本方面，官方给的是：

> “在更长序列长度下，create_block_mask 的实测端到端开销远低于 1%（例如，在 S=32768 时为 +0.2%）”

（[PyTorch 2.13 release blog](https://pytorch.org/blog/pytorch-2-13-release-blog/)）

---

## 5. 粒度：块大小选错了，多算的不止一点点

### 5.1 浪费率

块级稀疏是有损的：一个 $$BS\times BS$$ 的块里只要有一个元素必须算，$$BS^2$$ 个 MAC 就得全走完。定义

$$
\text{waste}=\frac{\text{实算元素数}-\text{必需元素数}}{\text{必需元素数}}
$$

实验**第 5 组**（$$N=8192$$）：

| 掩码 | $$BS=64$$ | $$BS=128$$ | $$BS=256$$ |
|:--|--:|--:|--:|
| causal | 0.77% | 1.55% | 3.11% |
| sliding $$w=1024$$ | 6.24% | 12.49% | **24.99%** |
| document | 0.00% | 0.00% | 0.00% |

两个结论：

- **滑窗掩码对块大小最敏感**。$$w$$ 与 $$BS$$ 同量级时（$$w=1024=8\times128$$），窗口边界每个块都要陪算，$$BS=256$$ 时浪费到 **24.99%**——等于四分之一的算力白烧。
- **document 掩码在文档长度对齐块边界时浪费率为零**：元素密度 15.62%，块级密度也是 15.62%，差 **0.00 个百分点**。这是「把文档 pad 到 $$BS$$ 的整数倍」的一个正面理由——它不只是为了形状规整，它直接消灭粒度损失。

同一件事的两种说法（$$BS=128$$）：

| 掩码 | 元素密度 | 块级密度 | 差 |
|:--|--:|--:|--:|
| causal | 50.01% | 50.78% | 0.78 pp |
| sliding $$w=1024$$ | 11.72% | 13.18% | 1.46 pp |
| document | 15.62% | 15.62% | **0.00 pp** |

### 5.2 硬约束：块大小必须整除 tile 大小

$$BS$$ 不是可以随便填的。Inductor 在 lowering 时会检查（`torch/_inductor/kernel/flex/flex_attention.py:492-507`）：

```python
if (
    cur_kernel_options["SPARSE_KV_BLOCK_SIZE"] % cur_kernel_options["BLOCK_N"] != 0
    or cur_kernel_options["SPARSE_Q_BLOCK_SIZE"] % cur_kernel_options["BLOCK_M"] != 0
):
    invalid_block_options = cur_kernel_options
    ...
    raise_flex_kernel_options_error(...)     # :500-506
```

合法的 $$(BS, B_N)$$ 组合（实验第 5 组枚举）：

| | $$B_N=32$$ | $$B_N=64$$ | $$B_N=128$$ |
|:--|:--|:--|:--|
| $$BS=64$$ | ✓ | ✓ | ✗ |
| $$BS=128$$ | ✓ | ✓ | ✓ |
| $$BS=256$$ | ✓ | ✓ | ✓ |

$$BS=64$$ 配 $$B_N=128$$ 直接报错。这条约束的来源是模板里的静态断言：

```python
tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)
```

（`torch/_inductor/kernel/flex/templates/flex_attention.py.jinja:32-33`）

为什么必须是整除关系：一个 $$BS\times BS$$ 的稀疏块要被切成若干个 $$B_M\times B_N$$ 的 tile 来执行；除不尽就会出现「半个 tile 有掩码、半个 tile 没有」的混合块，而模板里的 full/partial 判定是按 tile 走的。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 220" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="220" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">块粒度的浪费率：BS 越大，陪算的元素越多</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">N=8192，浪费率 =（整块陪算 − 真正需要）/ 真正需要</text>

<text x="24" y="64" font-size="11" font-weight="600" fill="#111827">causal（元素密度 50%）</text>
<rect x="180" y="54" width="12" height="16" fill="#16a34a"/><text x="198" y="67" font-size="10" fill="#14532d">0.77%</text>
<rect x="290" y="54" width="24" height="16" fill="#16a34a"/><text x="320" y="67" font-size="10" fill="#14532d">1.55%</text>
<rect x="400" y="54" width="48" height="16" fill="#22c55e"/><text x="454" y="67" font-size="10" fill="#14532d">3.11%</text>
<text x="24" y="82" font-size="10" fill="#6b7280">      BS=64                  BS=128                 BS=256</text>

<text x="24" y="106" font-size="11" font-weight="600" fill="#111827">sliding w=1024（元素密度 11.7%）</text>
<rect x="180" y="96" width="97" height="16" fill="#f59e0b"/><text x="283" y="109" font-size="10" fill="#92400e">6.24%</text>
<rect x="290" y="96" width="194" height="16" fill="#ea580c"/><text x="490" y="109" font-size="10" fill="#7c2d12">12.49%</text>
<rect x="400" y="96" width="388" height="16" fill="#dc2626"/><text x="24" y="130" font-size="10" fill="#991b1b">BS=256 → 24.99%（条形按 25% 满幅归一，越界部分截断显示）</text>

<text x="24" y="152" font-size="11" font-weight="600" fill="#111827">document（长度对齐 BS 时）</text>
<rect x="180" y="142" width="1" height="16" fill="#2563eb"/><text x="198" y="155" font-size="10" fill="#1e40af">0.00%</text>
<rect x="290" y="142" width="1" height="16" fill="#2563eb"/><text x="320" y="155" font-size="10" fill="#1e40af">0.00%</text>
<rect x="400" y="142" width="1" height="16" fill="#2563eb"/><text x="454" y="155" font-size="10" fill="#1e40af">0.00%</text>

<rect x="24" y="170" width="632" height="34" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
<text x="34" y="188" font-size="10.5" fill="#1e40af">判据很简单：掩码的「边界」是否落在块边界上。causal 的对角线必然斜穿块 → 有损失；</text>
<text x="34" y="200" font-size="10.5" fill="#1e40af">document 的边界只要对齐 BS 就完全落在块边界上 → 零损失。</text>
</svg>

---

## 6. GQA 的额外约束

分组查询注意力（GQA）在 FlexAttention 里是「可选 + 有代价」的。

### 6.1 正向：只需要一个开关

```python
out = flex_attention(q, k, v, enable_gqa=True)
```

源码的两条检查（`nn/attention/flex_attention.py:2451-2464`）：不开 `enable_gqa` 时 $$H_q\neq H_{kv}$$ 直接报错；开了则要求 $$H_q$$ 是 $$H_{kv}$$ 的整数倍。内部把它翻译成一个编译期常量：

```python
gqa_shared_heads = FloorDiv(Hq, Hkv)
kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)
```

（`torch/_inductor/kernel/flex/flex_attention.py:418-419`）

### 6.2 反向：FlexDecoding 要求比值是 2 的幂

decode 分支（`torch/_inductor/kernel/flex/flex_decoding.py:66-132`）的准入条件里有一条很具体：

```python
ratio = FloorDiv(Hq, Hkv)
pw_of_two = V.graph.sizevars.guard_or_false(
    sympy.And(sympy.Gt(ratio, 0), sympy.Eq(ratio & (ratio - 1), 0))
)                                                        # :108-112
```

实验第 5 组的枚举：

| GQA 比值 | 是 2 的幂？ | FlexDecoding 可用？ |
|--:|:--|:--|
| 1 | ✓ | ✓ |
| 2 | ✓ | ✓ |
| 3 | ✗ | ✗ |
| 4 | ✓ | ✓ |
| 6 | ✗ | ✗ |
| 8 | ✓ | ✓ |

**6 出局**。这不是吹毛求疵：LLaMA-3 8B 的比值是 4（32/8），但一些模型的比值是 3 或 5（比如 $$H_q=24, H_{kv}=8$$），这类模型在 decode 阶段会自动退回主 kernel（不会报错，只是拿不到 split-KV 的收益）。

其余准入条件（同一处，`:114-123`）：

| 条件 | 值 |
|:--|:--|
| query 长度 | $$<128$$ |
| batch、head 数 | 必须编译期静态可知 |
| BlockMask 头维 | 必须为 1 或等于 query 头数 |
| `FORCE_USE_FLEX_ATTENTION` | 必须为 False |

注意源码里那句注释（`:75-83`），它把设计意图说得很清楚：

> “Decode eligibility is an optimization choice, not a user-visible contract... We also do NOT use guard_or_true: an unknown predicate should disable decode, not silently enable it.”

**判不出来就退回通用 kernel，绝不为了启用 decode 而给用户强加运行时约束。**

---

## 7. 长上下文（一）：上下文并行 × BlockMask

这是全篇最值得看的一节，因为它展示了 BlockMask 的另一重身份。

### 7.1 一条硬约束

上下文并行（CP）把序列切开分给多个 rank，每个 rank 拿一段 query、看全局 KV。FlexAttention 的 CP 入口是 `_create_cp_block_mask`（`torch/distributed/tensor/experimental/_context_parallel/_attention.py:1184`）。它有这样一条检查：

```python
if Q_LEN % (device_mesh.size() * _DEFAULT_SPARSE_BLOCK_SIZE) != 0:
    raise NotImplementedError(
        f"Q_LEN {Q_LEN} is not divisible by CP mesh world size {device_mesh.size()} * "
        f"BLOCK_SIZE {_DEFAULT_SPARSE_BLOCK_SIZE}. This is not supported yet. "
    )                                                            # :1234-1238
```

docstring 解释了为什么（`:1222-1228`）：

> “Currently requires Q_LEN to be divisible by CP mesh world size * BLOCK_SIZE (BLOCK_SIZE defaults to 128). This constraint exists because the BlockMask must handle both padding and offsets correctly. For example, if Q_LEN is 384, CP world size is 2, and BLOCK_SIZE is 128, the local Q_LEN would be 192. In such cases, both rank0 and rank1 would have paddings in their local BlockMasks. Support for padding in this scenario is planned for future work.”

**原因**：local → global 的索引换算发生在**块粒度**，不是 token 粒度。看 `_rewrite_mask_mod` 里的换算（`:1275-1284`）：

```python
def local_q_idx_to_q_idx(local_q_idx: torch.Tensor) -> torch.Tensor:
    local_blk_idx, local_blk_offset = (
        local_q_idx // block_size,
        local_q_idx % block_size,
    )
    local_num_blocks = local_q_size // block_size
    blk_idx = local_num_blocks * rank + local_blk_idx
    return blk_idx * block_size + local_blk_offset
```

注意 `blk_idx = local_num_blocks * rank + local_blk_idx`：**rank $$r$$ 拿到的是块区间 $$[\,r\cdot n_b^{\rm local},\ (r+1)\cdot n_b^{\rm local})$$**，是块的连续区间，不是把 token 区间切成块。如果局部长度不是 $$BS$$ 的整数倍，块就会跨 rank 边界，两边都要补 padding —— 目前没实现。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 208" role="img" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
<rect x="0" y="0" width="680" height="208" fill="#ffffff"/>
<text x="24" y="22" font-size="15" font-weight="600" fill="#111827">CP × FlexAttention：切分单位是「块」，不是 token</text>
<text x="24" y="40" font-size="11.5" fill="#4b5563">local_q_idx → global 的换算先拆成 (块号, 块内偏移)，块号整体加上 rank·本地块数</text>

<text x="24" y="68" font-size="11" font-weight="600" fill="#1e3a8a">Q_LEN=1024，BS=128，CP=4 → 全局 8 块，每 rank 2 块</text>
<g>
<rect x="24" y="80" width="64" height="22" fill="#dbeafe" stroke="#60a5fa"/><text x="40" y="95" font-size="9.5" fill="#1e3a8a">b0</text>
<rect x="88" y="80" width="64" height="22" fill="#dbeafe" stroke="#60a5fa"/><text x="104" y="95" font-size="9.5" fill="#1e3a8a">b1</text>
<rect x="152" y="80" width="64" height="22" fill="#dcfce7" stroke="#6ee7b7"/><text x="168" y="95" font-size="9.5" fill="#14532d">b2</text>
<rect x="216" y="80" width="64" height="22" fill="#dcfce7" stroke="#6ee7b7"/><text x="232" y="95" font-size="9.5" fill="#14532d">b3</text>
<rect x="280" y="80" width="64" height="22" fill="#fef9c3" stroke="#facc15"/><text x="296" y="95" font-size="9.5" fill="#713f12">b4</text>
<rect x="344" y="80" width="64" height="22" fill="#fef9c3" stroke="#facc15"/><text x="360" y="95" font-size="9.5" fill="#713f12">b5</text>
<rect x="408" y="80" width="64" height="22" fill="#fae8ff" stroke="#e879f9"/><text x="424" y="95" font-size="9.5" fill="#86198f">b6</text>
<rect x="472" y="80" width="64" height="22" fill="#fae8ff" stroke="#e879f9"/><text x="488" y="95" font-size="9.5" fill="#86198f">b7</text>
<text x="552" y="95" font-size="10" fill="#475569">全局块号</text>
</g>
<text x="24" y="122" font-size="10.5" fill="#1e40af">rank0（蓝）</text><text x="152" y="122" font-size="10.5" fill="#14532d">rank1（绿）</text>
<text x="280" y="122" font-size="10.5" fill="#713f12">rank2（黄）</text><text x="408" y="122" font-size="10.5" fill="#86198f">rank3（紫）</text>

<path d="M56 104 L56 128" stroke="#64748b" stroke-width="1.2"/>
<path d="M52 122 L56 128 L60 122 Z" fill="#64748b"/>
<text x="66" y="142" font-size="10.5" fill="#374151">local_q_idx=130 → 块号 1、块内偏移 2 → 全局块号 = 2·0 + 1 = 1 → 全局 idx = 1·128+2 = 130</text>
<path d="M184 104 L184 128" stroke="#64748b" stroke-width="1.2"/>
<path d="M180 122 L184 128 L188 122 Z" fill="#64748b"/>
<text x="194" y="160" font-size="10.5" fill="#374151">rank1 上 local_q_idx=130 → 全局块号 = 2·1 + 1 = 3 → 全局 idx = 3·128+2 = 386</text>

<rect x="24" y="172" width="632" height="26" rx="4" fill="#fef2f2" stroke="#fecaca"/>
<text x="34" y="190" font-size="10.5" fill="#991b1b">若 Q_LEN=384、CP=2、BS=128：本地 192，块号 0 完整、块号 1 只有半个 → 两个 rank 都要 padding，当前直接 NotImplementedError</text>
</svg>

### 7.2 BlockMask 是代价模型

CP 的核心工程问题是**负载均衡**：causal 掩码下，靠前的 rank 只需要看很少的 key，靠后的 rank 要看全部。朴素连续切分会让 rank 0 闲死、最后一个 rank 累死。

PyTorch 为此准备了三个负载均衡器（`torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py`）：

| 类 | 位置 | 适用 |
|:--|:--|:--|
| `_HeadTailLoadBalancer` | `:87` | 首尾配对，要求 $$\text{seq} \bmod (2\cdot ws)=0$$ |
| `_PerDocumentHeadTailLoadBalancer` | `:176` | 每篇文档内做首尾配对 |
| `_PTRRLoadBalancer` | `:300` | **只能用 FlexAttention** |

第三个的 docstring 是本节最关键的一句话（`:301-304`）：

> “Processing-Time based Round-Robin (PTRR) load balancer. This load balancer should **only be used for flex_attention()** since it leverages `BlockMask`.”

它「leverage」的是什么？看 `_generate_indices`（`:435-442`）：

```python
block_mask = self.block_mask
kv_num_blocks = block_mask.kv_num_blocks
full_kv_num_blocks = block_mask.full_kv_num_blocks
non_sparse_kv_num_blocks = (
    kv_num_blocks + full_kv_num_blocks
    if full_kv_num_blocks is not None
    else kv_num_blocks
)
```

**代价模型就是 BlockMask 自己**：一个 query 块要算多少个 KV 块（partial + full），就是它的处理时间。换句话说：

> **BlockMask 不只是「哪些块要算」，它同时也写着「算它们要多久」。**

`ptrr_scheduling` 的算法（`:354-372`）是经典的 LPT 轮转，带方向交替：

```python
_, sorted_indices_descending = torch.sort(process_time, descending=True, stable=True)
sorted_indices_descending_reversed = torch.flip(
    sorted_indices_descending.view(-1, group_size), dims=[1]).view(-1)
tasks_in_group = torch.where(
    torch.arange(num_tasks) // group_size % 2 == 0,
    sorted_indices_descending,
    sorted_indices_descending_reversed,
)
tasks_in_group = tasks_in_group.view(-1, group_size).transpose(0, 1)
```

按处理时间降序排好，每 $$group\_size$$ 个一轮，**偶数轮顺序发、奇数轮逆序发**（这就是那个「方向交替」，它避免所有大任务都落在同一批 rank 上），然后转置成「第 $$i$$ 个任务进第 $$i\bmod CP$$ 组」。

实验**第 4 组**（$$N=4096$$、$$BS=128$$、$$CP=8$$、32 个 query 块）在 numpy 里复现了这个算法：

| 掩码 | 切分方式 | 各 rank 代价 (max/min) | 理想均值 | max/理想 | 峰谷差 |
|:--|:--|--:|--:|--:|--:|
| causal | 朴素连续 | 240 / 16 | 128.0 | **1.875** | 224 |
| causal | head-tail | 128 / 128 | 128.0 | **1.000** | 0 |
| causal | PTRR | 128 / 128 | 128.0 | **1.000** | 0 |
| sliding $$w=1024$$ | 朴素连续 | 64 / 16 | 56.0 | 1.143 | 48 |
| sliding $$w=1024$$ | head-tail | 64 / 36 | 56.0 | 1.143 | 28 |
| sliding $$w=1024$$ | PTRR | 63 / 49 | 56.0 | **1.125** | 14 |
| document | 朴素连续 | 96 / 32 | 72.0 | 1.333 | 64 |
| document | head-tail | 96 / 64 | 72.0 | 1.333 | 32 |
| document | PTRR | 80 / 64 | 72.0 | **1.111** | 16 |

三个观察：

1. **causal 上朴素切分的不均衡是 1.875**——最忙的 rank 干 240，最闲的只干 16，差 15 倍。整段训练的时间由最忙那个决定，等于 CP 白做了一半。
2. **head-tail 在 causal 上也是完美的**（1.000），但它只对「代价随位置单调」的掩码成立。
3. **一旦换到 sliding / document，head-tail 退化到 1.143 / 1.333，PTRR 仍是 1.125 / 1.111**。这就是 PTRR 存在的理由：它不假设代价曲线的形状，直接读 BlockMask 里的真实代价。

另外两个源码细节值得一提：

- 默认负载均衡器的选择（`_create_default_load_balancer`，`:475-483`）：`_cp_options.enable_load_balance` 且 `seq_length % (world_size*2) == 0` 才用 head-tail，否则返回 `None`（不重排）。
- PTRR 要求 `q_blk_size == kv_blk_size`（`:458-461`），不等就 `AssertionError`。
- `_create_cp_block_mask` 内部把 `create_block_mask` 编译后**缓存成全局单例**（`:1240-1245`）——又一次印证「编译并缓存」是标准做法。

---

## 8. 长上下文（二）：怎么把这些拼起来

把前面的约束收拢成一张配置清单（$$N$$ 全局序列长，$$CP$$ 上下文并行度，$$BS$$ 块边长）：

| 约束 | 表达式 | 出处 |
|:--|:--|:--|
| CP 整除 | $$N \bmod (CP\cdot BS)=0$$ | `_attention.py:1234` |
| 块/tile 整除 | $$BS \bmod B_M = 0$$ 且 $$BS \bmod B_N = 0$$ | `flex_attention.py:492-507` |
| head-tail 重排 | $$N \bmod (2\cdot CP)=0$$ | `_load_balancer.py:480` |
| PTRR | $$B_Q = B_{KV}$$（块边长两维相等） | `_load_balancer.py:458-461` |
| FlexDecoding | $$Q_{\rm local}<128$$、$$B,H$$ 静态、$$H_q/H_{kv}$$ 是 2 的幂 | `flex_decoding.py:84-123` |
| 确定性 dQ | 需要 `q_indices`，且当前只服务 FLASH 后端 | `flex_attention.py:1902-1905` |

一个能同时满足这些的配置大概长这样（$$N=32768$$、$$CP=8$$、$$BS=128$$）：

- $$N \bmod (8\cdot128)=32768 \bmod 1024 = 0$$ ✓
- $$B_N$$ 取 64 或 128（都能整除 128）✓
- head-tail 需要 $$32768 \bmod 16 = 0$$ ✓
- 每 rank 本地 $$4096$$ token = 32 个 query 块 ✓

---

## 9. BlockMask 的其余 API

### 9.1 切片（推理侧）

decode 时预构造一个大方块，然后按行切片（`nn/attention/flex_attention.py:1245` 的 `__getitem__`）：

```python
block_mask = create_block_mask(causal_mask, B=None, H=None,
                               Q_LEN=MAX_SEQ_LEN, KV_LEN=MAX_SEQ_LEN)
block_offset = i // block_mask.BLOCK_SIZE[0]
block_mask_slice = block_mask[:, :, block_offset]
```

但切片之后 `mask_mod` 会被替换成一个专门报错的桩（`:2486-2489`）：

```python
if getattr(block_mask, "mask_mod", None) is _sliced_mask_mod_error:
    raise RuntimeError("Cannot use mask_mod from a sliced BlockMask")
```

原因很实在：切片后的块级索引已经对应不上原来的 `mask_mod` 语义了，与其算错不如直接报错。

### 9.2 诊断

| API | 位置 | 用途 |
|:--|:--|:--|
| `sparsity()` | `:1446` | 未被计算的块占比（$$100\times(1-\text{computed}/\text{total})$$） |
| `numel()` | `:1437` | 不考虑稀疏度的元素总数 |
| `to_dense()` | `:1459` | 还原成块级稠密掩码 |
| `to_string(grid_size=(20,20))` | `:1475` | 直接把掩码画成字符网格，调试神器 |
| `_adjust(new_q_len, new_kv_len)` | `:1390` | 改序列长度而不重算 |

`sparsity()` 的实现值得注意（`:1452-1454`）：`computed_blocks = kv_num_blocks.sum() + full_kv_num_blocks.sum()`——partial 与 full 是「被计算的块」的一个划分，所以加起来才对。

### 9.3 手写构造函数

真的在意构造成本，就绕开 mask_mod 直接给索引。官方 FAQ 给的例子（针对 causal）：

```python
def create_causal_mask(S):
    BLOCK_SIZE = 128
    num_blocks = torch.arange(S // BLOCK_SIZE, device="cuda") + 1
    indices = torch.arange(S // BLOCK_SIZE, device="cuda").expand(
        S // BLOCK_SIZE, S // BLOCK_SIZE
    )
    num_blocks = num_blocks[None, None, :]
    indices = indices[None, None, :]
    return BlockMask(num_blocks, indices, BLOCK_SIZE=BLOCK_SIZE, mask_mod=causal_mask)
```

这个写法把「$$Q\times KV$$ 次 mask_mod 求值 + 一份 GB 级 bool 张量」整个换成了两次 `arange`。代价是你得自己保证索引正确。

---

## 10. 决策清单

| 你面对的情况 | 做法 |
|:--|:--|
| 掩码与 batch/head 无关 | `B=None, H=None`——省的不是时间，是 GB 级峰值 |
| 掩码每 step 变（文档掩码） | 入口构造一次，穿层透传；`torch.compile(create_block_mask, fullgraph=True)`（`reduce-overhead` 若每 step 都重建） |
| 掩码每层变 | 先算 `skip_ratio > κ/(4D)`（$$D=64$$ 时 ≈1.5%）。划算就上，但要把 bool 峰值与重编译次数盯住 |
| 序列长 $$>8192$$ 且要 causal | 考虑 CP；用 PTRR 而不是 head-tail（除非你确定代价曲线单调） |
| 用 CP + FlexAttention | 先确认 $$N \bmod (CP\cdot BS)=0$$，否则当场 `NotImplementedError` |
| 要逐位可复现的训练 | `torch.use_deterministic_algorithms(True)` + `compute_dq_write_order=True`；注意 FA4 后端 + 块稀疏目前还不支持 |
| decode 阶段 | 确认 $$H_q/H_{kv}$$ 是 2 的幂，否则拿不到 split-KV |
| 滑窗掩码 | $$BS$$ 别选太大；$$w=1024$$ 时 $$BS=256$$ 会多算 25% |
| 文档掩码 | 把文档长度 pad 到 $$BS$$ 的整数倍——粒度损失直接归零 |
| 不确定掩码长什么样 | `block_mask.to_string()` 直接看 |

---

## 11. 收束

五条值得带走的结论：

1. **`create_block_mask` 的成本不在元数据，在构造过程**。元数据几万分之一，但构造会物化一份 $$B\cdot H\cdot Q\cdot KV$$ 的 bool 张量——8×16×8192² 是 **8192 MiB**。这就是 `B=None, H=None` 成为首推的原因。
2. **「每层重建」在 FLOPs 上几乎总是划算的**（阈值 ≈1.5%，净收益 +45%~+73%）。官方说它最难，难在显存峰值、独立 kernel 序列、动态 shape 重编译这三件事。把这三件事和 FLOPs 分开看，决策才做得对。
3. **确定性 dQ 修的是浮点非结合性**，不是竞态逻辑。一张 $$\mathrm{rank}=\mathrm{cumsum}-\mathrm{dense}$$ 的表把累加顺序钉死，按 rank 累加与串行累加**逐位相等**。
4. **块粒度是显式可调的损失**。滑窗 $$w=1024$$ 在 $$BS=256$$ 下浪费 **24.99%**；而文档长度对齐块边界时浪费 **0.00%**。
5. **BlockMask 的第二重身份是成本账**：`kv_num_blocks + full_kv_num_blocks` 就是每个 query 块的处理时间，PTRR 直接拿它做负载均衡，把 causal + CP=8 的不均衡从 **1.875** 压到 **1.000**，而且在 head-tail 失效的 document 掩码上仍然有效（1.333 → 1.111）。

01 回答「这个编程模型为什么成立」，02 回答「它要付多少成本、怎么付得少一点」。剩下没讲的是**编译器那一侧的时间都花在哪**——autotune 怎么选 tile、`max-autotune` 值不值、CUDA graph 能不能包、重编译怎么治理。那是 03。

---

## 参考

**官方材料**

- PyTorch Team, *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*（FAQ：BlockMask 缓存策略、自定义构造函数），[pytorch.org/blog/flexattention](https://pytorch.org/blog/flexattention/)
- *PyTorch 2.13 Release Blog*（确定性反向、Apple Silicon），[pytorch.org/blog/pytorch-2-13-release-blog/](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- Guessous, Stern, Hoehnerbach, Xie, Zadouri, Shah & Dao, *FlexAttention + FlashAttention-4: Fast and Flexible*（限制清单、LPT 调度），[pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- Dong, Feng, Guessous, Liang & He, *Flex Attention: A Programming Model for Generating Optimized Attention Kernels*, [arXiv:2412.05496](https://arxiv.org/abs/2412.05496)
- PyTorch 文档：`torch.nn.attention.flex_attention`，[docs.pytorch.org/docs/stable/nn.attention.flex_attention.html](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)

**源码（行号锁定 PyTorch `v2.14.1` = `5c488690`）**

- [`torch/nn/attention/flex_attention.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/attention/flex_attention.py)：`:1967` create_block_mask、`:1828` create_mask、`:1706` 块级降采样、`:1879` dQ 写入顺序算法、`:1245` 切片、`:1446` sparsity、`:1475` to_string、`:2486` 切片后 mask_mod 的报错
- [`torch/_inductor/kernel/flex/flex_attention.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/flex/flex_attention.py)：`:492-507` 块/tile 整除检查、`:418-419` GQA_SHARED_HEADS
- [`torch/_inductor/kernel/flex/flex_decoding.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/flex/flex_decoding.py)：`:66-132` FlexDecoding 准入条件
- [`torch/_inductor/kernel/flex/templates/flex_attention.py.jinja`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/flex/templates/flex_attention.py.jinja)：`:32-33` 静态断言
- [`torch/distributed/tensor/experimental/_context_parallel/_attention.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/tensor/experimental/_context_parallel/_attention.py)：`:1184` CP BlockMask、`:1234` 整除约束、`:1275-1284` 块粒度索引换算
- [`torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py)：`:87` head-tail、`:300` PTRR、`:354-372` 调度算法、`:435-442` 以 BlockMask 为代价模型
- vLLM `v0.31.0` = [`db9527a4`](https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6)：[`vllm/v1/attention/backends/flex_attention.py`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/attention/backends/flex_attention.py) `:54-58`

**本站配套**

- [FlexAttention（01）：把注意力变体编译成一个 FlashAttention 内核](/2026/10/17/flex-attention-01-programming-model/)
- 本实验：[`ipynbs` `experiments/flex-attention/blockmask_lab/`](https://github.com/lrypcy/ipynbs/tree/5d223527089c816f464f37b345af35ce2b778e0f/experiments/flex-attention/blockmask_lab/)（commit `5d22352`，19 条断言全过）
- 01 的实验：[`flex_core/`](https://github.com/lrypcy/ipynbs/tree/5954fb521a279b997005a6f514144c3b380c8e36/experiments/flex-attention/flex_core/)（commit `5954fb5`）
