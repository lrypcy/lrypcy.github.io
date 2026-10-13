# 08 公开 benchmark 交叉验证

> 锁定版本：**vLLM `v0.31.0`**（2026-10-05）／ **SGLang `v0.5.21`**（2026-10-02）
> 检索日期：2026-10-09

## TL;DR

1. **网上流传最广的「SGLang 领先 vLLM 29%」是2024 年的数字**，
   测试对象是 SGLang v0.2.3 vs vLLM 0.11.0 —— 版本相差约两年。
2. **锁定版本（v0.31.0 / v0.5.21）不存在任何同配置的第三方 head-to-head。**
   两个项目的官方 release notes 也都没有给出任何形式的 vLLM↔SGLang 端到端对比。
3. 因此本文的立场是：**性能选型不能靠公开 benchmark，只能靠自测+ 机制推理。**
   本文给出的是「哪些机制会决定差距」和「怎么自己测」，不是「谁更快」。

---

## 1. 可信度分级

全文使用三级标注，读者可据此判断每条数字的用途。

| 等级 | 含义 | 用法 |
|:---|:---|:---|
| **A** | 第三方实测，配置透明可溯源、理论上可复现 | 可作论据，须标注版本与条件 |
| **B** | **厂商/项目方自报** | 只作厂商口径引用，**不与第三方混排** |
| **C** | 自媒体转载 / 无配置 / AI 生成 / 无法溯源 | **只作现象提及，不作论据** |

---

## 2. 全部流传数字的分级表

| # | 数字 | 声称来源 | 可信度 | 真实测试条件 / 问题 |
|:---|:---|:---|:---|:---|
| 1 | 16,215 vs 12,553 tok/s（SGLang +29%） | “H100 实测对比” | **C** | 实为 **SGLang v0.2.3 vs vLLM 0.11.0**（版本相差约两年）。单 H100 80GB、Llama-3.1-8B bf16、1000 ShareGPT×10 runs。无独立复现 |
| 2 | TTFT 79ms vs 103ms | Inference.net 对比表 | **C** | 与 #1 **同源**，同一份旧基准 |
| 3 | ITL 6.0ms vs 7.1ms | Inference.net 对比表 | **C** | 与 #1 **同源** |
| 4 | 6.4× throughput | SGLang 论文 / LMSYS | **B** | [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)。前缀密集负载（few-shot/agent/RAG/多轮）。**不是 v0.5.21，也不是对比 v0.31.0** |
| 5 | 8B +29%、70B +4% | Turion 引 Spheron | **A→版本不对** | 8B 数字即 #1 的旧数，疑搬运；70B 版本约 v0.18.0/v0.5.9 |
| 6 | Llama-3.3-70B FP8：仅领先 2.5%–4.6% | markaicode | **A** | 方法透明（seed=42、200 prompts、自写 aiohttp harness、512in/256out、unique-prompt）。单 H100，**v0.18.0 / v0.5.9** |
| 7 | 8×H200 4-bit 70B：vLLM 5100 / SGLang 5300 / **Triton 5450** | IoT Digital Twin Q2 2026 | **C** | 原文自述 “illustrative/internal”。**注意归属**：5450 属 Triton 不是 SGLang |
| 8 | p50 TTFT 112ms vs 120ms | — | **C** | **无法追溯到任何原始出处**。IoT DT 同场景 p50 实为 SGLang 79ms / vLLM 82ms，与之对不上 |
| 9 | EAGLE-3：158.34→373.25 tok/s（2.36×） | SGLang team | **B** | 单 H100、MT-Bench。方法公开但属自报（[arXiv:2503.01840](https://arxiv.org/abs/2503.01840)） |
| 10 | xgrammar 结构化输出 up to 10× | SGLang v0.4 博客 | **B** | 约束解码场景。无独立复现 |
| 11 | overlap：1.1× vs v0.3、1.3× vs SOTA | SGLang v0.4 博客 | **B** | 官方限定「小模型 + 大 TP 最显著」。无独立复现 |
| 12 | cache-aware LB：82,665→158,596 tok/s | SGLang v0.4 博客 | **B** | 8×A100、平衡共享前缀负载；命中率 20%→75% |
| 13 | DP Attention：1.9× | SGLang v0.4 博客 | **B** | DeepSeek-Coder-V2、8×H100 TP8 |
| 14 | **命中率 43.8%→60.8%，TTFT 1.57s→1.07s** | SGLang v0.5.20 release | **B** | DeepSeek-V4-Flash + 共享 system prompt。**前缀缓存关系的核心量化证据**，见 §5 |
| 15 | sampling-mask+overlap：batch1 +17% / batch64 +52% | SGLang v0.5.20 release | **B** | Qwen3-8B |
| 16 | DeepSeek-V4.1 长 prompt 首 token 快 22% | SGLang v0.5.21 release | **B** | |
| 17 | Kimi K3 PD prefill +20.6% | SGLang v0.5.21 release | **B** | PD 分离部署 |
| 18 | **vLLM v0.31.0 无端到端吞吐数字** | vLLM release | **B** | 仅组件级（如 GLM-5.3-Flash 元数据 1.6–4.8×） |
| 19 | **关overlap 反而更快：22,025→25,555 tok/s** | SGLang issue #2558 | **A** | Qwen2.5-0.5B-Instruct、单 A100、0.4.0.post1。社区已复现 |
| 20 | async scheduling + PP：+30.8% E2E、+31.8% TPOT | vLLM PR #32618 | **B** | Qwen3-30B-A3B-Thinking-2507-FP8、PP4 |

---

## 3. ⚠️ 三处必须纠正的流传错误

这一节是本文存在的主要理由之一。

### 3.1 「+29%」是 2024 年的数字

`16,215 vs 12,553 tok/s`（Llama-3.1-8B / H100，+29%）被大量 2026 年的「对比指南」当作当年实测引用。
实际上它的测试对象是 **SGLang v0.2.3 vs vLLM 0.11.0**，两个版本相差约两年。

连带地，`TTFT 79ms vs 103ms` 与 `ITL 6.0ms vs 7.1ms`
**全部回溯到同一份基准** —— 它们是那一份旧基准的三次转载，
并列引用会制造「多源独立印证」的假象。

### 3.2 归属被记反了

广泛流传的「IoT Digital Twin 8×H200：vLLM 5300 vs SGLang 5450」是错的。
原文（Batch16）实际为：

$$
\text{vLLM } 5100 \;<\; \text{SGLang } 5300 \;<\; \text{Triton-TRT } 5450
$$

即 5450 属于 **Triton**，不是 SGLang；而 SGLang 是 5300，vLLM 是 5100。
数值和归属都记反了。

### 3.3 「p50 TTFT 112 vs 120ms」无法追溯

该数字找不到原始出处。同场景（IoT DT，Llama-3.3-70B、512-token 4-bit prompt）的 p50
实测是 **SGLang 79ms / vLLM 82ms**，与之对不上。

---

## 4. 锁定版本有没有第三方 head-to-head？

**没有。明确没有。**

- 公开互联网上**不存在**任何一份「同时、同硬件、同配置、同数据集」在
  v0.31.0 / v0.5.21 上测两个框架的第三方 benchmark。
- 现存 2026 年第三方数据里方法论最透明的 markaicode，版本是 **v0.18.0 / v0.5.9**（2026-03），
  结果是 70B unique-prompt 仅领先 **2.5%–4.6%**（不是 +29%）。
- 两个项目的官方 release notes **都只报各自组件/模型收益，
  没有给出任何 vLLM↔SGLang 端到端对比**。

**结论**：任何在v0.31.0 / v0.5.21 语境下写「SGLang 快 29%」的句子，
都是**误导性引用**。

---

## 5. 唯一可信的因果链：前缀命中率 → TTFT

这是全文最有价值的一条量化证据，也是唯一一条条件清晰的。

SGLang v0.5.20 release notes（DeepSeek-V4-Flash + 共享 system prompt，
Unified Radix Tree分支点缓存）：

| 指标 | 前 | 后 | 变化 |
|:---|---:|---:|:---|
| token 命中率 | 43.8% | 60.8% | **+17pp** |
| 平均 TTFT | 1.57s | 1.07s | **约 −32%** |

$$
\text{经验关系：命中率每+10pp}\;\Rightarrow\;\text{TTFT 约降 19\%}
$$

（长 prompt / 共享 system prompt 场景。**厂商自报**，但条件清晰、机制可解释，
是这类数字里质量最高的一条。）

### 5.1 为什么这条链可信

因为它是**同一次发布、同一模型、只改缓存机制**的前后对比 ——
不是两个引擎之间的对比。这排除了大量混杂因素。

而两个引擎之间的对比（#1）恰恰混杂了太多东西：版本、调度器、kernel、
默认参数全都不一样。

### 5.2 命中率 → 吞吐的定性模型

| 命中率区间 | 预期表现 | 依据 |
|:---|:---|:---|
| **> 60%** | RadixAttention 的 token 级最长前缀匹配 + cache-aware 路由收益最大 | Turion 引 Spheron：>60% 前缀复用时 prefill 延迟 3–5× 改善 |
| **20% – 60%** | 有可观收益，取决于负载形状与是否开 cache-aware 路由 | 上述 v0.5.20 的 +17pp 落在此区间 |
| **< 20%** | 差距消失甚至反转 —— radix 树维护开销成为净负担 | 机制推理（见 [01](01-architecture-divergence.md) §1） |

聚合吞吐侧的一个自报点（B 级）：v0.4 cache-aware LB 实测命中率 20%→75%（3.8×）时，
82,665→158,596 tok/s（≈1.9×），即命中率每 +10pp 约 +14% 聚合吞吐。

> ⚠️ 这两组数字**口径不同**（一个是单请求 TTFT，一个是集群聚合吞吐），
> **不能换算或相加**。列在这里只是给出「量级」概念。

---

## 6. vLLM 赢的场景与 SGLang 的已知回退

只列 SGLang 赢的场景就是宣传。这一节列反面。

### 6.1 overlap 在小模型单卡上反而变慢（A 级）

[SGLang issue #2558](https://github.com/sgl-project/sglang/issues/2558)：
Qwen2.5-0.5B-Instruct在单张 A100 上跑 `bench_serving` 随机负载（4096 in / 2048 out），
**`--disable-overlap-schedule` 反而更快**：22,025 → 25,555 total tok/s。社区已复现。

这与 v0.4 官方措辞不矛盾 —— 那里限定「gains most significant on
**small models and large tensor parallelism sizes**」。两句合起来读：

> 小模型 + 单卡 + 低并发时，overlap 引入的 bookkeeping 开销可能超过它隐藏的调度开销。
> 「默认开启」≠「always faster」。

SGLang 自己也在代码里承认这点：**连续两个 prefill 批次会主动关闭 overlap 以改善 TTFT**
（`scheduler.py:2015`，`:2031-2035`，环境变量 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`）。

### 6.2 大模型默认配置下 vLLM 压过 SGLang

GPUStack 的 GPT-OSS-120B、2×H100、**默认配置**实测（版本 v0.10.2 / v0.5.3rc0）：

| 引擎 | tok/s |
|:---|---:|
| **vLLM v0.10.2** | **6,095** |
| TensorRT-LLM v1.0.0 | 2,139 |
| SGLang v0.5.3rc0 | 1,814 |

即开箱即用的大模型场景，**vLLM 默认配置领先 SGLang 默认配置 3.4 倍**。
（版本较旧，标 B→A 之间，但量级差异足够大。）

### 6.3 模型越大差距越小（多源一致）

- AIMultiple 自注：8B 优势最大，70B「大幅收窄」
- markaicode（v0.18/v0.5.9，A 级方法论）：70B unique-prompt 仅 **2.5%–4.6%**
- 多个中文实测提到 70B 极限高并发下 **vLLM 反超约 5%**

**定性结论**：模型越大、并发越高，RadixAttention 的相对优势越被
「通用调度效率 + 显存效率」抵消，乃至反转。

### 6.4 vLLM 正在系统性追回调度差距

vLLM PR #32618（自测）：async scheduling + PP 在 Qwen3-30B-A3B-Thinking-2507-FP8 / PP4 上
E2E 吞吐 **+30.8%**、TPOT **+31.8%**。

vLLM V1 的主线工作（EngineCore 进程拆分、Model Runner V2、ZMQ 优化）
本质上都是在攻「调度开销」—— 也就是 SGLang overlap scheduler 当年的优势项。
2026 年两者的调度差距已远小于 2024 年。

---

## 7. TensorRT-LLM 的位置

一句话：**它不是「慢」，而是贵在编译、窄在硬件、僵在多模型。**

- 每模型 / 每硬件需专门编译，单次 25–40 分钟
- 仅支持 NVIDIA
- 编译产物与具体模型强绑定，频繁换模型 / 多模型路由场景不灵活

它与 vLLM/SGLang 的对比本质是「专用编译优化 vs 通用推理框架」，**不在同一选型维度**，
故不入本文主线。§6.2 的数据也显示它在大模型开箱即用场景落后 vLLM 2.8 倍。

---

## 8. 那该怎么选？—— 自己测的方法

既然公开数据不可用，唯一诚实的路径是自测。下面是可执行的最小方案。

### 8.1 必须控制的四个变量

任何自建对比如果没锁住这四项，结论同样不可信：

| 变量 | 怎么锁 |
|:---|:---|
| 拉取时间 | 同一天从两个仓库拉取，**记录 commit SHA** |
| 模型与精度 | 同一 checkpoint、同一量化格式（注意 vLLM 无 gguf/bitsandbytes，见 [02](02-feature-matrix.md) §7） |
| 硬件与并行 | 同型号卡、同卡数、同 TP/EP/PD 配置 |
| **负载前缀复用率** | ⚠️ **这是最关键也最常被忽略的变量** —— 见下|

### 8.2 前缀复用率必须作为自变量

理由已在 §5 说清：前缀命中率是决定两边差距符号的**唯一主变量**。
一个只测「唯一 prompt」负载的 benchmark 在结构上就不可能测出 RadixAttention 的优势，
也测不出它在无复用时的开销。

建议至少跑三档：

- **低复用**（< 20%，所有 prompt 唯一）
- **中复用**（约 50%，共享 system prompt + 变化的用户问题）
- **高复用**（> 80%，多轮对话 / RAG 反复命中同一批文档）

### 8.3 工具选择

- vLLM：`vllm bench serve`
- SGLang：`sglang bench_serving`
- 跨框架第三方工具：`genai-perf`、`guidellm`

> ⚠️ 用两个框架自带的 bench 工具时，注意它们对 TTFT/ITL/吞吐的**分位数统计口径可能不同**
> —— 必须核对 `p50`/`p95`/`p99` 的插值方式，否则 5% 以内的差距没有意义。

### 8.4 先测前缀命中率，再决定要不要测性能

这一步的成本比性能 benchmark 低两个数量级，但信息量更大：

> tokenize 一天真实请求，统计两两请求的公共前缀 token 数，
> 估算理论命中率 → 直接映射到「该选谁」。

理由：§5 已经说明，命中率每 +10pp 对应 TTFT 约 −19%，
这个斜率在多样本上比单点性能测试稳定得多。

配套动作（SGLang 侧）：
- 把 system prompt 与 tool 定义里**所有可变内容移出开头**（把公共前缀真正做长）
- 起一个副本开 `--enable-metrics`，回放真实流量
- 在 `fcfs` 与 `lpm` 两种策略下分别记录命中率、TTFT、吞吐
- 在 `--mem-fraction-static` 与 `--chunked-prefill-size` 上按**真实最长 prompt**调，
  而不是按benchmark 调
- 权重要更新时必须 `/flush_cache`，并对命中率突降告警

---

## 9. 引用本文数字的纪律

给后续作者（和读者）的硬规则：

1. **不要**在v0.31.0 / v0.5.21 语境下写「+29%」。要引用必须带版本脚注。
2. **不要**把「200+ 模型架构」和「300+ 模型架构」当作数量差距的论据 ——
   两者口径不同（见 [02](02-feature-matrix.md) §1.3）。
3. **不要**把 release notes 的组件级收益当作两个引擎的差距。
4. 所有 release 自报数字一律标 **B（厂商自报）**，且**不与第三方数据混排在同一张表**。
5. 厂商自报数字之间也可能不一致 —— 引用前务必回到对应tag 的 release notes 原文核对。

---

## 参考

- SGLang 论文：<https://arxiv.org/abs/2312.07104>
- SGLang v0.4 release blog（overlap / cache-aware LB / DP attention / xgrammar）：
  <https://openlm.ai/sglang-v0.4/>
- SGLang v0.5.20 release notes（命中率 43.8%→60.8%）：<https://github.com/sgl-project/sglang/releases/tag/v0.5.20>
- SGLang v0.5.21 release notes：<https://github.com/sgl-project/sglang/releases/tag/v0.5.21>
- vLLM v0.31.0 release notes：<https://github.com/vllm-project/vllm/releases/tag/v0.31.0>
- vLLM async scheduling PR #32618：<https://github.com/vllm-project/vllm/pull/32618>
- SGLang issue #2558（overlap 回退，A 级证据）：<https://github.com/sgl-project/sglang/issues/2558>
- markaicode 70B 实测（方法透明，A 级）：<https://markaicode.com/benchmarks/sglang-llama-31-h100-throughput-benchmark/>
- IoT Digital Twin Q2 2026（归属需纠正）：<https://iotdigitaltwinplm.com/?p=3741/>
- AIMultiple（旧版数字的原始出处）：<https://research.aimultiple.com/inference-engines>