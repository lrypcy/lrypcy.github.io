---
title: "推理引擎（06）：benchmark 交叉验证 —— 谁在快、快多少、能不能信"
date: 2026-10-18 20:00:00 +0800
categories:
  - 推理服务
tags: [serving-engine, vllm, sglang, benchmark, methodology, cross-validation, throughput, ttft, inference]
layout: post
mathjax: true
---

> **推理引擎系列 · 第 06 篇**
>
> 本系列面向**已经能把模型跑起来、但还没读过任何一个推理引擎源码**的工程师：把 vLLM 与
> SGLang 的功能差异、性能差异、选型结论，全部还原到**源码里可点开核验的那一行**。
>
> 取证基准：**vLLM `v0.31.0`**（commit `db9527a4`）／ **SGLang `v0.5.21`**（commit `e00930c5`）。
> 本文所有行号均锁在这两个 commit 上。
>
> - 00 引擎在栈里的位置 → 01 架构分歧的根 → 02 调度与批处理 → 03 支持矩阵（上）→
>   04 支持矩阵（下）→ 05 PD 分离与 KV 传输 → **06 benchmark 交叉验证（本文）** →
>   07 选型手册 → 08 前缀命中率

> **TL;DR**
>
> * **锁定版本（v0.31.0 / v0.5.21）不存在任何同配置的第三方 head-to-head**。
>   本文逐条核验了互联网上流传的 20 个性能数字，结论是：**没有一个能直接回答
>   「这两个版本谁更快」**。
> * **传播最广的「SGLang 领先 29%」测的是 SGLang v0.2.3 vs vLLM 0.11.0**——
>   两个版本相差约两年；且它的口径是**离线批处理聚合吞吐**、prompt 基本唯一，
>   **根本不是 RadixAttention 的主场**。把它当前缀负载的证据是用错了论据。
> * **流传的「反面证据」同样有问题**：「unique prompt 下 vLLM 60 vs SGLang 52.7」
>   是另一个指标（单流 decode 速率）、另一个模型（70B distill）、另一种硬件
>   （2×H100）—— 两组数字被拼成一个「前缀决定论」的故事，**两边都用错了**。
> * **唯一条件清晰的量化因果链**是前缀命中率 → TTFT：SGLang v0.5.20 自报
>   命中率 43.8%→60.8% 对应 mean TTFT 1.57s→1.07s。companion 实验用一阶模型
>   核验其内部自洽（反解全量 prefill 时间两点相差 2.32%），并把口诀
>   「每 +10pp 降 19%」还原为**区间中点的瞬时斜率**——它随命中率变化，不是常数。
> * **本文的立场**：性能选型不能靠公开 benchmark。该信的是机制（01/02/05 篇）
>   与自测（§8 给最小方案），不是任何一张流传的对比表。

---

## 0. 速览：三类数字的命运

把本文核验过的数字按「能不能当论据」分三档，这就是全文的地图：

| 档 | 含义 | 典型例子 | 本文用途 |
|:---|:---|:---|:---|
| **A** | 第三方实测，配置透明、可溯源、可复现 | GPUStack GPT-OSS-120B（脚本全公开）；SGLang issue #2558（完整输出） | 可作论据，须带版本与条件 |
| **B** | 厂商 / 项目方自报 | SGLang v0.5.20/v0.5.21 release notes；vLLM v0.31.0 release notes | 只作厂商口径引用，**不与 A 级混排** |
| **C** | 自媒体转载 / 无配置 / 无法溯源 | 「+29%」的二次传播；「p50 TTFT 112 vs 120ms」 | 只作现象提及，**不作论据** |

三个已经在系列前文用过的纪律在这里继续生效：**任何计数必须附口径**
（03 篇的「387 vs 282」）、**任何数字必须附版本**（本文通篇）、
**厂商自报不与第三方混排**。

---

## 1. 传播最广的数字：+29% 的完整解剖

「SGLang 吞吐比 vLLM 高 29%」大概是推理引擎领域被引用最多的单个数字。
它的原始形态是：

| 引擎 | 吞吐（tok/s） | 版本 |
|:---|---:|:---|
| SGLang | 16,215 | **v0.2.3** |
| LMDeploy | 16,132 | 0.10.2 |
| vLLM | 12,553 | **0.11.0**（FlashInfer enabled） |

测试条件（据原始方法学）：单张 H100 80GB、Llama-3.1-8B-Instruct bf16、
1,000 条 ShareGPT_Vicuna_unfiltered prompt（50–4,000 字符、seed 42）、
每引擎 10 轮全量、约 150 万 token、gpu_memory_utilization 0.8。
原始发布方自己指出「same attention kernel as SGLang; gap attributed to
orchestration, not kernel」——两边用的是同一个 FlashInfer kernel，
差距被归因于调度编排。

**三个事实决定了这个数字的用法**：

1. **版本差约两年**。vLLM 0.11.0 与 SGLang v0.2.3 不是同一时代的产品；
   deepinfra 的评论写得很直接：「The published setup pairs SGLang v0.2.3
   with vLLM 0.11.0, releases roughly two years apart. A number produced
   that way can't tell you which engine is faster today, in either direction.」
2. **口径是离线批处理聚合吞吐**，不是 serving 延迟，也不是单流速率。
3. **prompt 基本唯一**（方法学转述者的原话：「batch throughput on identical,
   mostly-unique prompts」）。也就是说，**这一场根本没有前缀可复用**——
   它不是 RadixAttention 的主场，+29% 是在对 SGLang 最不利的负载形状下
   测出来的。把 +29% 当作「共享前缀负载下 SGLang 更快」的证据，是用错了论据。

顺带一提，同表里 LMDeploy 与 SGLang 相差不到 0.6%，发布方自己称之为
统计意义上的平手——三个引擎挤在 16,100–16,200 区间，vLLM 独自落后。

---

## 2. 五处必须纠正的流传错误

本节是本文存在的主要理由。每条都核到了原始出处。

### 2.1 「+29%」被当成 2026 年实测

大量 2026 年的「对比指南」「选型表」把 16,215 vs 12,553 标注为当年实测。
实际版本是 v0.2.3 vs 0.11.0（见 §1）。连带受害的是同源转载的
TTFT 79ms vs 103ms 与 ITL 6.0ms vs 7.1ms——**这三个精确数字不是三份独立证据，
而是一份旧基准的三次转载**，并列引用会制造「多源互相印证」的假象。

### 2.2 「同一基准的反面行」：60 vs 52.7 是被拼进来的

流传结构是：「共享前缀 SGLang 16,215 vs 12,553；唯一 prompt vLLM 60 vs 52.7；
所以前缀决定胜负。」前半截的问题见 §1；后半截是**另一场测试**：

| 维度 | 16,215 vs 12,553 | 60 vs 52.7 |
|:---|:---|:---|
| 模型 | Llama-3.1-8B | DeepSeek-R1-Distill-Llama-70B |
| 硬件 | 1×H100 | 2×H100 |
| 指标 | **离线批处理聚合吞吐** | **单请求 decode 速率** |
| 版本 | v0.2.3 / 0.11.0 | 未公布 |
| 来源 | AIMultiple | RunPod 测试（经 Particula 等转述） |

deepinfra 把这种拼法叫做 **unit conflation**：「Those are single-stream
decode rates for one user. AI Multiple's 16,215 tok/s is aggregate server
throughput across a saturated batch. Both get labeled 'tokens per second'
and they answer different questions.」同一场景下还有第三组数字：
开缓存后 SGLang 35 vs vLLM 32.8——**依然是 SGLang 领先**。
所以「unique prompt 下 vLLM 反超」至多说明「单流 decode 速率下 vLLM 不落后」，
推不出「前缀决定吞吐胜负」。

### 2.3 「markaicode 独立 70B 实测 2.5%–4.6%」无法在该页核验

多个中文转述称「markaicode 用透明方法测得 70B 上 SGLang 仅领先 2.5%–4.6%」。
打开该页核对：它深度测试的是 **Llama 3.1 8B**，且 8B 表**自承是转述**——
「AIMultiple's own reported methodology, reproduced here rather than a
recreation, since we can't independently re-run it without H100 access」。
该页唯一的 70B 段把数字归给 **Spheron（2026-03）**，且结论与转述相反：
「TensorRT-LLM leading at every concurrency level, with the gap to vLLM
as small as ~8% at low concurrency」。
**把转述当独立实测、再把结论改成相反的意思，是双重错误。**

### 2.4 「8×H200：SGLang 5450」归属记反了

广泛流传的「IoT Digital Twin 8×H200：vLLM 5,300 vs SGLang 5,450」是错的。
原文表格（Llama-3.3-70B 4-bit，Batch 16 req）实际为：

| 引擎 | tok/s |
|:---|---:|
| vLLM | 5,100 |
| TGI | 4,200 |
| SGLang | 5,300 |
| **Triton-TRT（TensorRT-LLM）** | **5,450** |

**5,450 属于 Triton-TRT，不是 SGLang**；vLLM 是 5,100、SGLang 是 5,300。
归属和数值都记反了。需要说明：该表没有版本、seed、harness 信息，
作为证据仍是 C 级——但「归属记反」这个错误本身，已经被原文表格证伪。

### 2.5 「p50 TTFT 112 vs 120ms」无法追溯

该数字找不到任何原始出处；同场景（Llama-3.3-70B、512-token 4-bit prompt）
可溯源的 p50 实测与它对不上。**无法溯源的数字一律降级为现象**，
不进入任何论证链。

---

## 3. 锁定版本有没有第三方 head-to-head？

**没有。明确没有。**

- 公开互联网上不存在任何一份「同时、同硬件、同配置、同数据集」在
  v0.31.0 / v0.5.21 上测两个框架的第三方 benchmark。§2.3 说明连最接近的
  候选页也是转述。
- **两个项目的官方 release notes 都只报自家数字**，且已逐字核验：
  - vLLM v0.31.0 release notes：全文没有跨引擎对比，性能数字全是组件级
    自报（如 GLM-5.3-Flash 的「metadata ops 1.6-4.8x faster」、
    ROCm「13x lower decode latency」）；
  - SGLang v0.5.21 release notes：同样只报自家（「DeepSeek-V4.1 gets 22%
    faster first token on long prompts」「Kimi K3 gets 20.6% higher prefill
    throughput in PD serving」）。
- 方法论最透明的现存第三方数据（GPUStack，见 §5.2）版本是
  **v0.10.2 / v0.5.3rc0**，同样不是锁定版本。

**结论**：任何在 v0.31.0 / v0.5.21 语境下写「SGLang 快 29%」的句子，
都是**误导性引用**。想在锁定版本上给结论，唯一诚实的路是自建基准（§7）。

---

## 4. 唯一条件清晰的因果链：命中率 → TTFT

全部核验过的数字里，只有一条同时满足「条件清晰、机制可解释、同源前后对比」：
SGLang v0.5.20 release notes（PR #34565，DeepSeek-V4-Flash + 共享
system prompt）：

| 指标 | 前 | 后 | 变化 |
|:---|---:|---:|---:|
| token 命中率 | 43.8% | 60.8% | +17pp |
| mean TTFT | 1.57s | 1.07s | 约 −32% |

两点细节让这条链比同类数字硬：

1. **机制具体**：变的是 SWA 组件的 branching-point caching——请求从共享前缀
   分叉时保留滑窗状态，分叉复用而不重算。不是泛泛的「缓存变好了」。
2. **同源前后对比**：同一次发布、同一模型、只改缓存机制。
   这排除了版本、硬件、调度器等混杂因素——而**跨引擎对比**（§1）恰恰
   把这些全混在一起。

### 4.1 一阶模型核验（companion 实验）

companion 实验 `prefix-hitrate-ttft-model` 把两点代进

$$
\mathrm{TTFT}(h) = (1-h)\,P
$$

（命中部分零成本、未命中部分付全额 prefill 时间 $$P$$）：

- 由前点反解 $$P = 2.7936\,s$$，由后点反解 $$P = 2.7296\,s$$，
  **相差 2.32%**；平均 $$P = 2.7616\,s$$ 下两点复现误差都小于 3%。
  **上报数字内部自洽**。
- 流传的口诀「命中率每 +10pp，TTFT 约降 19%」是这个模型在区间中点的
  瞬时斜率。它不是常数：

| 命中率 $$h$$ | 每 +10pp 的 TTFT 降幅 |
|---:|---:|
| 20% | 12.5% |
| 43.8% | 17.8% |
| 60% | 25.0% |
| 80% | 50.0% |

也就是说，**每 +10pp 的 TTFT 相对降幅是 $$0.1/(1-h)$$，随命中率增大**：
同一句口诀在低命中率区高估收益（20% 处实际只有 12.5%）、在高命中率区低估
（80% 处已有 50.0%）。按绝对值算，每 +10pp 恒定省 $$0.1P$$——
从 20% 爬到 60% 与从 60% 爬到 100% 省下的绝对值相同（都是 $$0.4P$$），
但前者只把 TTFT 砍半，后者直接清零。

### 4.2 吞吐侧的平行数字（B 级，量纲不同）

v0.4 的 cache-aware 负载均衡自报：命中率 20%→75%（3.8×）时
聚合吞吐 82,665→158,596 tok/s（≈1.9×）。同款直线模型
$$\mathrm{throughput}(h)=\mathrm{base}\,(1+kh)$$ 反解
$$k=2.5077$$、$$\mathrm{base}=55{,}054\ \mathrm{tok/s}$$，
中点瞬时斜率每 +10pp 约 **+11.4%**（区间平均粗估 +14%）。

> ⚠️ 这个 +11.4% 与上表的 −21% **量纲不同**：一个是单请求延迟的相对变化，
> 一个是集群聚合吞吐的相对变化，负载、并发、硬件全都不同。
> 数值接近是巧合，**不能互相推导，更不能在同一张表里混排**。

---

## 5. 反面证据：vLLM 赢的场景与 SGLang 的已知回退

只列 SGLang 赢的场景是宣传。以下每条都核到可溯源出处。

### 5.1 overlap scheduler 在小模型单卡上反而变慢（A 级）

SGLang issue #2558（标题就是「Improve the Zero-Overhead Batch Scheduler
performance for the small model」）：Qwen2.5-0.5B-Instruct、单张 A100、
SGLang 0.4.0.post1、`bench_serving` 随机负载（4096 in / 2048 out、500 prompts）：

| 配置 | 总吞吐（tok/s） |
|:---|---:|
| 默认（开 overlap） | **22,025.22** |
| `--disable-overlap` | **25,555.94** |

关掉 overlap 快约 16%。报告者自述多次复现结果一致。
这与官方 v0.4 博客的限定措辞并不矛盾——那里说的是收益在
「small models and large TP sizes」最显著；单卡 + 低并发下，
bookkeeping 开销超过它隐藏的调度开销。
**「默认开启」≠「always faster」**——这句话对两个引擎的所有默认值都成立。
SGLang 自己也在代码里留了后门：连续两个 prefill 批次会主动关闭 overlap
以改善 TTFT（`scheduler.py:2032` 的 `CONSECUTIVE_PREFILL_OVERLAP`，
环境变量 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`，02 篇 §4 已引用）。

### 5.2 默认配置下 vLLM 大幅领先（A 级，单一来源）

GPUStack 性能实验室（脚本与完整输出公开）：GPT-OSS-120B、2×H100、
ShareGPT 1,000 prompts、**各引擎默认配置**：

| 引擎 | 版本 | 总吞吐（tok/s） | mean TTFT |
|:---|---:|---:|---:|
| vLLM | v0.10.2 | **6,095.37** | 24,830 ms |
| TensorRT-LLM | v1.0.0 | 2,139.54 | 55,897 ms |
| SGLang | v0.5.3rc0 | 1,814.32 | 132,613 ms |

vLLM 默认配置领先 SGLang 默认配置约 **3.4 倍**。
三条必须跟着念的注脚：这是**默认配置**（GPUStack 自注「not an exhaustive
search over every possible configuration」）；这是**单一来源**
（GPUStack 自身以 vLLM 为默认引擎，有立场）；SGLang 的 mean TTFT
高达 132.6s，说明这个负载下它的默认参数明显没调开（长 prompt 高并发）。
它证明的是「开箱即用的体验 vLLM 更稳」，不是「vLLM 更快」。

### 5.3 70B 上差距收窄（多源一致）

- Spheron（2026-03，H100、Llama-3.3-70B FP8）：TensorRT-LLM 各并发档领先，
  与 vLLM 的差距最小约 8%；
- Cerebrium（Llama 3.1 70B、batch-1、H100）：vLLM 的 TTFT 三者最低；
- 多个中文实测提到 70B 极限高并发下 vLLM 反超约 5%（C 级，仅作现象）。

**定性结论**：模型越大、并发越高，前缀机制的优势越被「通用调度效率 +
显存效率」抵消。这与 01 篇的机制分析一致：radix 的收益取决于负载的
**前缀形状**，与模型大小只是相关而非因果。

### 5.4 vLLM 在系统性追回调度差距

vLLM PR #32618 自测：async scheduling + PP 在
Qwen3-30B-A3B-Thinking-2507-FP8、PP4 上 E2E 吞吐 **+30.8%**、TPOT **+31.8%**。
更大的图景是 V1 架构本身：EngineCore 进程拆分（ZMQ 通信）、
Model Runner V2 在 v0.25.0 成为 dense 模型默认——攻的正是
「CPU 调度开销」，也就是 SGLang overlap scheduler 当年的优势项。
**2026 年两者的调度差距已远小于 2024 年**，这本身就否定了
「拿 2024 的基准推 2026 的结论」。

---

## 6. 指标口径对照：为什么 5% 以内的差距没有意义

本文所有翻车案例的共同根源是**指标混用**。把常见口径摆在一起：

| 口径 | 回答的问题 | 常见陷阱 |
|:---|:---|:---|
| 离线批处理聚合吞吐 | 饱和 batch 下服务器能吃多少 token | 与在线 serving 延迟不可比；对 prompt 分布极敏感 |
| 在线 serving 总吞吐 | 带调度、带排队的端到端 token 速率 | 受并发上限、到达过程影响；两边默认 `max_num_seqs` 不同 |
| 单请求 decode 速率（tok/s/user） | 一个用户看到文字出现的速度 | 与聚合吞吐差一两个数量级，**都被叫做 tok/s** |
| mean / p50 / p99 TTFT | 首 token 延迟分布 | 均值被长尾拉爆；分位数插值方式两边可能不同 |
| TPOT / ITL | 解码步稳定性 | 与 batch 大小强耦合，不锁并发就没意义 |

由此得到**自测的第一条纪律**：引用任何差距之前，先确认两边
（a）同一指标、（b）同一负载、（c）同一并发档。三条缺一，
5% 以内的差距不值得讨论——§1 的表里 LMDeploy 与 SGLang 差 0.6%，
发布方自己称之为平手，这就是标杆行为。

---

## 7. 自己测的最小方案

公开数据不可用，唯一诚实的路径是自测。最小可执行方案：

### 7.1 必须锁住的四个变量

| 变量 | 怎么锁 |
|:---|:---|
| 代码版本 | 同一天拉取，**记录 commit SHA**（本文通篇纪律） |
| 模型与精度 | 同一 checkpoint、同一量化格式；注意 vLLM 无 gguf（04 篇 §6 已列） |
| 硬件与并行 | 同型号卡、同卡数、同 TP/EP/PD 配置 |
| **负载前缀复用率** | **最关键也最常被忽略**——见下 |

### 7.2 前缀复用率必须作为自变量

理由在 §4：命中率是少数有清晰因果链的变量，且 01 篇的 companion 实验
（`prefix-reuse-granularity`）算过：块对齐税是**绝对额 ≤15 token 的固定税**，
长前缀负载上不到 1%——**负载形状比粒度更决定胜负**。
建议至少三档：

- 低复用（<20%，所有 prompt 唯一）；
- 中复用（约 50%，共享 system prompt + 变化用户问题）；
- 高复用（>80%，多轮对话 / RAG 反复命中同一批文档）。

只测「唯一 prompt」的 benchmark 在结构上不可能测出 radix 的优势，
也测不出它在无复用时的开销——两头都瞎。

### 7.3 工具与口径

- vLLM：`vllm bench serve`；SGLang：`sglang bench_serving`；
- 跨框架：`genai-perf` / `genai-bench`（可固定 prefix-reuse 比例）、
  `guidellm`（自动扫并发梯度，产出分位数曲线）；
- ⚠️ 用两家自带工具时，**必须核对分位数插值口径**，否则 5% 以内没意义（§6）。

### 7.4 先测命中率，再决定要不要测性能

这一步成本比性能 benchmark 低两个数量级，信息量更大：
tokenize 一天真实请求，统计两两请求的公共前缀 token 数，估算理论命中率。
依据 §4 的模型，命中率直接映射到 TTFT 预期——比单点性能测试稳定得多。
这正是 08 篇（前缀命中率）的主题，这里只强调顺序：**先定性，再定量**。

---

## 8. 引用纪律与取证边界

**引用本文系数字的硬规则**：

1. 不要在 v0.31.0 / v0.5.21 语境下写「+29%」；要引用必须带版本脚注
   （v0.2.3 / 0.11.0，2024 级）。
2. 不要把「16,215 vs 12,553」当前缀负载证据——那一场 prompt 基本唯一。
3. 不要把「60 vs 52.7」当它的反面行——指标、模型、硬件全不同。
4. 不要把转述当独立实测（markaicode 8B 表自承转述 AIMultiple）。
5. IoT DT 的 5,450 是 Triton-TRT 不是 SGLang；且该表无方法学，C 级。
6. 所有 release notes 数字一律标 B（厂商自报），不与 A 级混排。
7. 无法溯源（如 p50 TTFT 112/120）一律降级为现象。

**本文的取证边界**：

- §1/§2 的版本与方法学核验自 AIMultiple 原始方法学及其转述页、
  deepinfra 的版本指正、markaicode 页面自述；IoT DT 表格数字核自原文，
  但其自述的「illustrative」定位未能在本次核验中确认，故整体仍列 C 级。
- §5.1 的完整 benchmark 输出核自 issue #2558 原文；「社区已复现」
  本文弱化为「报告者多次复现一致」。
- §5.2 的数字核自 GPUStack 文档一手页面；单一来源 + 厂商立场已在正文标注。
- §4.1 的模型核验是**内部自洽性检验，不是复现**；一阶模型是下界简化
  （命中块也有寻址成本、批处理会摊薄单位成本）。
- 本文不主张「谁更快」。在锁定版本上，这个问题**没有公开答案**；
  本文主张的是「哪些数字不能用、为什么、以及你自己该测什么」。

## 实验与数据

companion 实验 `prefix-hitrate-ttft-model`（纯 CPU、确定性输出）：
<https://github.com/lrypcy/ipynbs/blob/9a091b5/experiments/serving-engine/prefix-hitrate-ttft-model/run.py>

- 组 1：TTFT = (1−h)·P 一阶模型对 v0.5.20 上报两点的反解与互预测；
  「每 +10pp 降幅」随命中率的变化表。
- 组 2：吞吐直线模型对 v0.4 cache-aware LB 两点的反解；
  两种斜率的量纲对照。

## 上一篇 / 下一篇

[《推理引擎（05）：PD 分离与 KV 传输》](/2026/10/18/serving-05-pd-disaggregation-kv-transfer/)；
前缀复用的粒度代价见
[《推理引擎（01）：架构分歧的根》](/2026/10/18/serving-01-architecture-divergence/)；
chunked prefill 的延迟交易见
[《推理引擎（02）：调度与批处理》](/2026/10/18/serving-02-scheduling-and-batching/)；
支持矩阵的口径纪律见
[《推理引擎（03）：支持矩阵（上）》](/2026/10/18/serving-03-support-matrix-models/)。

## 参考

1. AIMultiple 推理引擎对比（+29% 的原始方法学来源）— <https://research.aimultiple.com/inference-engines>
2. deepinfra：vLLM vs SGLang（版本指正与 unit conflation 分析）— <https://deepinfra.com/blog/vllm-vs-sglang>
3. markaicode：Llama 3.1 8B H100 吞吐（自承转述 AIMultiple）— <https://markaicode.com/benchmarks/sglang-llama-31-h100-throughput-benchmark>
4. markaicode：70B / RTX 4090 汇总（RunPod 单流数字、Spheron 70B FP8）— <https://markaicode.com/benchmarks/sglang-llama-33-rtx-4090-throughput-benchmark/>
5. horizonlux：SGLang vs vLLM（「前缀决定论」叙事的典型）— <https://horizonlux.com/blog/sglang-vs-vllm>
6. mlai.qa：三引擎对比（Spheron 70B FP8 的再转述）— <https://mlai.qa/blog/vllm-vs-sglang-vs-tensorrt-llm>
7. GPUStack 性能实验室：GPT-OSS-120B on H100（默认配置三方对比，脚本公开）— <https://docs.gpustack.ai/latest/performance-lab/gpt-oss-120b/h100/>
8. IoT Digital Twin Q2 2026（归属纠正的原文表格）— <https://iotdigitaltwinplm.com/?p=3741/>
9. SGLang issue #2558（overlap 回退，A 级）— <https://github.com/sgl-project/sglang/issues/2558>
10. SGLang v0.5.20 release notes（命中率 43.8%→60.8%，PR #34565）— <https://github.com/sgl-project/sglang/releases/tag/v0.5.20>
11. SGLang v0.5.21 release notes（PD +20.6% 等）— <https://github.com/sgl-project/sglang/releases/tag/v0.5.21>
12. vLLM v0.31.0 release notes（无跨引擎对比）— <https://github.com/vllm-project/vllm/releases/tag/v0.31.0>
13. vLLM PR #32618（async scheduling + PP 自测）— <https://github.com/vllm-project/vllm/pull/32618>
14. SGLang 论文（6.4× 吞吐，前缀密集负载）— <https://arxiv.org/abs/2312.07104>
15. SGLang v0.4 release blog（overlap / cache-aware LB / DP attention）— <https://lmsys.org/blog/2024-12-04-sglang-v0-4/>
16. EAGLE-3（2.36× 自报）— <https://arxiv.org/abs/2503.01840>
17. companion 实验：prefix-hitrate-ttft-model — <https://github.com/lrypcy/ipynbs/blob/9a091b5/experiments/serving-engine/prefix-hitrate-ttft-model/run.py>
18. companion 实验：prefix-reuse-granularity — <https://github.com/lrypcy/ipynbs/blob/ac9f51c/experiments/serving-engine/prefix-reuse-granularity/run.py>
