# SGLang vs vLLM 性能 Benchmark 交叉验证报告

> 验证对象：中文 deep dive 文章中引用的 SGLang / vLLM 性能数字
> 锁定版本：**vLLM `v0.31.0`**（2026-10-05 发布）/ **SGLang `v0.5.21`**（2026-10-02 发布）
> 数据采信原则：仅采信 2026 年数据；2024/2025 仅作“历史锚点”标注，不作 2026 年论据。
> 生成日期：2026-10-09

---

## 0. 方法学与可信度分级

### 0.1 三级可信度定义
| 等级 | 含义 | 在文章中的用法 |
|------|------|----------------|
| **A** | 第三方实测，配置透明可溯源、理论上可复现 | 可作论据，但仍须注明测试条件与版本 |
| **B** | 厂商/项目方自报（SGLang team、vLLM team、LMSYS） | 必须标注「厂商自报 / 有夸大动机」，不与独立数据混用 |
| **C** | 自媒体 / 无配置 / AI 生成 / 无法溯源 | 只作“现象提及”，绝不作论证依据 |

### 0.2 核心结论（先说最重要的发现）
**本文锁定的两个版本（v0.31.0 / v0.5.21，均为 2026-10）在公开互联网上没有任何一份“同时、同配置、同硬件”测两框架的第三方 head-to-head benchmark。** 文章里流传最广的一批“2026 年数字”实际来源如下，均与锁定版本对不上：

- **16,215 / 12,553 tok/s（+29%）** —— 出自 AIMultiple 2026 年文章，但其测试配置是 **SGLang v0.2.3 vs vLLM 0.11.0**（deepinfra 明确指出两版相差约两年）。这不是 2026-10 的数字。
- **Spheron / Turion 引用的 8B +29%、70B +4%** —— 即便确为 2026-03 实测，所用版本约为 **vLLM v0.18.0 / SGLang v0.5.9**，仍非锁定版本。
- **markaicode 独立 70B 基准** —— 唯一方法论透明（seed=42、200 prompts、aiohttp 自写 harness）的第三方，但版本为 v0.18.0 / v0.5.9，且结果是 SGLang 仅领先 **2.5%–4.6%**（非 +29%）。

> 写作建议：凡引用上述数字，必须加版本脚注。任何“2026 年 SGLang 领先 vLLM 29%”的句式，在锁定版本语境下属于**误导性引用**。

---

## 1. 流传数字逐条核实

表格列：| 数字/说法 | 声称来源 | 原始出处 URL | 可信度 | 测试条件 | 独立复现 |

| # | 数字 / 说法 | 声称来源 | 原始出处 URL | 可信度 | 测试条件 | 独立复现 |
|---|------------|----------|--------------|--------|----------|----------|
| 1 | 16,215 vs 12,553 tok/s（SGLang +29%） | “H100 实测对比” | https://research.aimultiple.com/inference-engines | **C**（版本严重过期） | 单 H100 80GB、Llama-3.1-8B bf16、1,000 ShareGPT prompts×10 runs、gpu_mem 0.8 | **无**；且实为 **SGLang v0.2.3 vs vLLM 0.11.0**（deepinfra 指正版本差约两年） |
| 2 | TTFT 79ms vs 103ms（SGLang 快 23%） | Inference.net 对比表 | https://inference.net/content/sglang-complete-guide/ | **C** | 同上 AIMultiple 配置（v0.2.3/v0.11.0） | 无；与 #1 同源，版本不对 |
| 3 | ITL 6.0ms vs 7.1ms（SGLang 快 15%） | Inference.net 对比表 | https://inference.net/content/sglang-complete-guide/ | **C** | 同上 | 无；与 #1 同源 |
| 4 | SGLang 吞吐 6.4× vs 基线 | SGLang 原论文 / LMSYS | arXiv:2312.07104（2024-06 更新） | **B**（厂商/LMSYS 自报） | few-shot / agent / RAG / multi-turn 等前缀密集负载，对比“state-of-the-art 推理系统” | 论文方法可复现，但非 v0.5.21、非对比 vLLM v0.31.0 |
| 5 | Spheron 2026-03：8B +29%（16,215 vs 12,500） | Turion 博客引用 | https://www.edgechat.ai/sglang-mai01029（汇总） | **A→但版本不对** | H100、Llama-3.1-8B；但所引数字即 AIMultiple 的 v0.2.3/v0.11.0 旧数，疑为搬运而非新测 | 无法确认 Spheron 是否独立重测；与锁定版本无关 |
| 6 | Spheron 2026-03：70B FP8 +4%（1,920 vs 1,850） | Turion / Spheron | https://www.edgechat.ai/sglang-mai01029 | **A→但版本不对** | H100、Llama-3.3-70B FP8；版本约 v0.18.0/v0.5.9 | 无第三方复现；与锁定 v0.31.0/v0.5.21 无关 |
| 7 | p50 TTFT 112ms vs 120ms | （用户提及，未给源） | 无法追溯 | **C**（无法追溯） | 未知模型/硬件/并发 | 无；未能找到此数字原始出处 |
| 8 | IoT Digital Twin 8×H200 4-bit Llama-3.3-70B：vLLM ~5,300 vs SGLang ~5,450 | IoT DT 基准（用户记忆） | iotdigitaltwinplm.com（2026 Q2 基准） | **C**（且记忆有误） | 8×H200 SXM、4-bit Llama-3.3-70B；**原文 Batch16 为 vLLM 5100 / SGLang 5300 / Triton 5450** | 自述“illustrative/internal”，非严格基准；**5450 是 Triton 而非 SGLang**，用户记忆混淆 |
| 9 | markaicode Llama-3.3-70B FP8：SGLang 领先 2.5%–4.6% | markaicode 独立基准 | https://markaicode.com/benchmarks/sglang-llama-31-h100-throughput-benchmark/ | **A**（方法透明） | 单 H100、unique-prompt、vLLM v0.18.0 / SGLang v0.5.9、custom aiohttp、200 prompts、seed 42、512in/256out | 方法可复现；**版本非锁定** |
| 10 | EAGLE-3：Llama-3.1-8B 158.34→373.25 tok/s（2.36×） | SGLang team 自测 | arXiv:2503.01840v2 + SGLang docs（speculative_decoding） | **B**（厂商自测） | 单 H100、MT-Bench、EAGLE-3 投机解码 | 方法公开，但属 SGLang 方自报 |
| 11 | xgrammar 结构化输出 up to 10× | SGLang v0.4 博客 | openlm.ai/sglang-v0.4 | **B**（厂商自报） | 约束解码（JSON 等）场景 | 无独立复现 |
| 12 | v0.4 overlap scheduler：1.1× vs v0.3 / 1.3× vs SOTA | SGLang v0.4 博客 | openlm.ai/sglang-v0.4 | **B** | CPU 调度与 GPU 计算重叠 | 无独立复现 |
| 13 | v0.4 cache-aware LB：82,665→158,596 tok/s（命中率 20%→75%） | SGLang v0.4 博客 | openlm.ai/sglang-v0.4 | **B** | Rust router 前缀命中率预测路由 | 无独立复现 |
| 14 | v0.4 DP Attention：1.9× | SGLang v0.4 博客 | openlm.ai/sglang-v0.4 | **B** | DeepSeek-Coder-V2、8×H100 TP8、MLA 数据并行 | 无独立复现 |
| 15 | v0.5.20：DS-V4-Flash 共享前缀命中率 43.8%→60.8%，TTFT 1.57s→1.07s | SGLang release notes | sglang.ai/blog + release #34565 | **B**（厂商自报，但条件清晰） | DeepSeek-V4-Flash、共享 system prompt | 无独立复现；**这是前缀缓存关系的核心量化证据** |
| 16 | v0.5.21：DeepSeek-V4.1 长 prompt 首 token 快 22% | SGLang release notes | sglang.ai/blog + release #40352 | **B** | DeepSeek-V4.1 长 prompt | 无独立复现 |
| 17 | v0.5.21：Kimi K3 PD prefill +20.6% | SGLang release notes | sglang.ai/blog + release #40045 | **B** | Kimi K3 PD 分离部署 | 无独立复现 |
| 18 | vLLM v0.31.0：**无端到端吞吐数字** | vLLM release（freedom.tech 等转载） | vllm.ai releases / 转载 | **B** | 仅组件级（GLM-5.3-Flash 元数据 1.6–4.8× 等） | 无；release 未给 vLLM↔SGLang 对比 |
| 19 | SGLang Issue #2558：关 overlap 反而更快（22,025→25,555 total tok/s） | SGLang GitHub issue | github.com/sgl-project/sglang/issues/2558 | **A**（真实 issue） | Qwen2.5-0.5B-Instruct、A100、SGLang 0.4.0.post1 | 社区复现确认；**小模型/单卡回退证据** |
| 20 | vLLM async scheduling + PP：+30.8% E2E 吞吐、+31.8% TPOT | vLLM GitHub PR | github.com/vllm-project/vllm/pull/32618 | **A→B**（PR 自测） | Qwen3-30B-A3B-Thinking-2507-FP8、PP4 | 无第三方复现 |

> 关于 #1–#3 的特别提示：Inference.net、horizonlux.com 等“对比表”里的 16,215/12,553、TTFT 79/103、ITL 6.0/7.1 全部回溯到 AIMultiple 同一篇（v0.2.3/v0.11.0）基准。换言之，**这三个“精确数字”不是三份独立证据，而是一份旧基准的三次转载**。文章中若并列引用，会制造“多个来源互相印证”的假象。

---

## 2. 2026 年方法论透明的第三方 Benchmark（同时测两框架）

**结论：未找到在锁定版本（v0.31.0 / v0.5.21）上同时测两框架的第三方 harness。** 现存的 2026 年第三方数据分两类：

### 2.1 最贴近“方法论透明”的第三方：markaicode（Llama-3.3-70B）
- 来源：https://markaicode.com/benchmarks/sglang-llama-31-h100-throughput-benchmark/
- 配置透明：自写 aiohttp 客户端、200 条 prompts、seed=42、512in/256out、单 H100、Llama-3.3-70B FP8。
- 版本：**vLLM v0.18.0 / SGLang v0.5.9**（2026-03，非锁定）。
- 结果：unique-prompt 下 SGLang 仅领先 2.5%–4.6%——与流传的 +29% 量级完全不同。
- 评级：**A（方法透明，但版本对不上）**。

### 2.2 其余第三方多为“搬运/再叙述”旧基准
- Turion、edgechat.ai 引用的 Spheron 2026-03 数字，其 8B 16,215/12,500 与 AIMultiple 旧数完全一致，疑似搬运而非新测（见 #5）。
- 多个中文博客（CSDN、掘金）直接复制 AIMultiple 的 v0.2.3/v0.11.0 数字并标注“2026 实测”，属 **C 级误导**。

### 2.3 推荐用于文章自身复测的方法论（若作者要自建基准）
| 工具 | 用途 | 备注 |
|------|------|------|
| `vllm bench serve` / `sglang.bench_serving` | 同构 serving 压测 | 必须**同一 commit 日**拉两框架、同模型、同硬件、同数据集 |
| genai-bench / genai-perf（Triton 生态） | 标准化请求分布、固定 prefix-reuse 比例 | 可控制“前缀复用率”这一关键变量 |
| guidellm | 自动扫描并发梯度 | 产出 p50/p95 TTFT、TPOT 曲线 |
| production-bench / vllm-optimizer | 生产形态压测 | 注意是否含前缀复用 |

> 写作建议：文章若想给出“2026-10 锁定版本”结论，**最诚实的做法是作者自建一份同版本基准**，而非引用任何现存第三方数字。现存数字最高只能用作“历史趋势锚点”。

---

## 3. 前缀缓存命中率 ↔ 性能关系

这是 SGLang（RadixAttention）相对 vLLM（APC，block 级哈希）的核心差异化论点，也是少数有清晰量化证据的方向。

### 3.1 核心量化证据（厂商自报，B 级）
| 指标 | 前 | 后 | 变化 | 来源 |
|------|----|----|------|------|
| 共享前缀命中率 | 43.8% | 60.8% | +17pp | SGLang v0.5.20（#34565），DeepSeek-V4-Flash 共享 system prompt |
| TTFT | 1.57s | 1.07s | **约 −32%** | 同上 |

> 解读：命中率提升 17 个百分点，对应 TTFT 下降约 32%。量级上，**前缀命中率每提升 ~10pp，TTFT 约降 ~19%**（在长 prompt / 共享 system prompt 场景）。这是“前缀密集负载 SGLang 占优”最直接的因果证据。

### 3.2 命中率 → 吞吐的定性模型
- **高复用（>60%）**：RadixAttention 的 token 级最长前缀匹配 + cache-aware 调度（优先把同前缀请求排到一起）收益最大。Turion 引 Spheron：>60% 前缀复用时 prefill 延迟 3–5× 改善。
- **低复用（<20%）**：两框架差距消失甚至反转——SGLang 的 radix tree 维护开销成为净负担（见 #19 回退、第 4 节）。
- **吞吐模型（经验式）**：`有效吞吐 ≈ 基线吞吐 × (1 + k·命中率)`，k 随模型/负载而异；v0.4 cache-aware LB 实测 命中率 20%→75% 时 82,665→158,596 tok/s（≈ 1.9×），即命中率每 +10pp 约 +14% 聚合吞吐（B 级自报）。

### 3.3 机制对比（写作可用）
| | vLLM APC | SGLang RadixAttention |
|---|----------|----------------------|
| 索引结构 | block 级哈希表 | token 级基数树（radix tree） |
| 匹配粒度 | 整块精确匹配 | 最长前缀匹配（支持部分/分支重叠） |
| 典型受益负载 | 可预测模板前缀 | agent 多轮 / RAG / few-shot（前缀天然分支） |
| 无复用代价 | 哈希开销小 | radix 树维护开销相对明显 |

---

## 4. 反面证据：vLLM 赢 / SGLang 回退的场景

### 4.1 SGLang 性能回退：Issue #2558（A 级，真实）
- 场景：Qwen2.5-0.5B-Instruct、A100、SGLang 0.4.0.post1。
- 现象：**关闭 overlap scheduler 反而更快**——22,025 → 25,555 total tok/s。
- 含义：zero-overhead batch scheduler 在小模型 / 单卡 / 低并发下收益为负，CPU 调度重叠的额外开销盖过了 GPU 重叠收益。**“overlap 永远更快”是错的。**

### 4.2 vLLM 默认配置获胜：GPUStack 基准（markaicode，A 级）
- 场景：GPT-OSS-120B、2×H100、**默认配置**。
- 结果：vLLM v0.10.2 = **6,095 tok/s** > TRT-LLM v1.0.0 = 2,139 > SGLang v0.5.3rc0 = 1,814。
- 含义：在“开箱即用、不手工调优”的大模型场景，vLLM 默认配置可以压过 SGLang 默认配置。**默认配置 ≠ 调优后配置**，但说明 vLLM 开箱体验在大模型上不弱。

### 4.3 vLLM 自身增益：Async Scheduling + PP（A→B 级，PR 自测）
- PR #32618：async scheduling + Pipeline Parallelism。
- 结果：E2E 吞吐 **+30.8%**、TPOT **+31.8%**，测于 Qwen3-30B-A3B-Thinking-2507-FP8、PP4。
- 含义：vLLM V1 把“调度开销”作为主攻方向（进程拆分 ZMQ、Model Runner V2），正在系统性追回与 SGLang overlap scheduler 的差距。2026 年两框架调度差距已显著小于 2024。

### 4.4 模型变大，差距缩小（多源一致）
- AIMultiple 自身注明：8B 上 SGLang/LMDeploy 优势最大，70B 上“大幅收窄”。
- markaicode（v0.18/v0.5.9）：70B unique-prompt 仅 2.5%–4.6% 领先。
- 多个中文实测（CSDN 掘金）：70B 极限高并发下 **vLLM 反超 5%**。
- 定性结论：**模型越大、并发越高，RadixAttention 的相对优势越被“通用调度/显存效率”抵消**，甚至反转。

### 4.5 无复用负载 vLLM 占优（horizonlux 等）
- 单轮、完全独立 prompt：vLLM 60 tok/s vs SGLang 52.7 tok/s（该数字来自同一 AIMultiple 旧基准的“反面行”，C 级，仅作现象）。
- 机制：无前缀可复用，SGLang 的 radix 树维护成为纯开销。

---

## 5. 为什么 TensorRT-LLM 不在对比主线

（基于既有认知 + 多源一致描述；未做新检索，标注为综合判断）

| 维度 | TensorRT-LLM | 对“为何不进主线”的影响 |
|------|--------------|------------------------|
| 编译成本 | 每模型/每硬件需专门编译，**单次 25–40 分钟**（视模型与 GPU） | 无法像 vLLM/SGLang 那样“拉下来就跑”，迭代/换模型成本极高 |
| 硬件绑定 | **NVIDIA only**，无 AMD/Intel/TPU/ Trainium 支持 | 与 vLLM 的“全硬件覆盖”定位不在一个维度；开源社区对比通常默认同硬件跨引擎 |
| 多模型场景 | 编译产物与具体模型强绑定，频繁切换模型代价大 | 生产“多模型路由/按需加载”场景不灵活；vLLM/SGLang 动态加载更适配 |
| 优势场景 | 单一固定模型、极致延迟/吞吐、纯 NVIDIA 集群 | 适合大厂固定 workload，不适合“框架对比”类文章的通用结论 |

> 结论：TensorRT-LLM 不是“慢”，而是**贵在编译、窄在硬件、僵在多模型**。它和 vLLM/SGLang 的对比本质是“专用编译优化 vs 通用推理框架”，不在同一选型维度，故不入本文主线。GPUStack 基准（4.2）显示即使在大模型默认配置下，TRT-LLM 也落后于 vLLM——进一步说明它当前在“开箱即用”赛道不占优。

---

## 附：给文章作者的可直接用的“诚实性清单”

1. **删除或加版本脚注**所有 16,215/12,553、TTFT 79/103、ITL 6.0/7.1 的“2026 实测”表述——它们实为 v0.2.3/v0.11.0（2024 级）。
2. **+29% 仅在 ≤8B、共享前缀、旧版本下成立**；锁定版本下无证据，markaicode 同法测得 70B 仅 +2.5~4.6%。
3. **IoT Digital Twin 的 5450 是 Triton，不是 SGLang**；原文 vLLM 5100 / SGLang 5300 / Triton 5450。
4. 所有 release notes 数字（v0.4 / v0.5.20 / v0.5.21 / vLLM v0.31.0）一律标 **B（厂商自报）**，且 vLLM v0.31.0 自身**没有**端到端 vLLM↔SGLang 对比数字。
5. 前缀缓存关系是**最强因果证据**（v0.5.20 命中率 43.8%→60.8% 对应 TTFT −32%），可放心引用但标 B。
6. 若想给“2026-10 锁定版本”结论，**自建同版本基准**是唯一诚实路径；现存第三方最高作趋势锚点。

---
*报告结束。全文约 230 行，所有 A/B/C 评级均按“可溯源 + 独立复现”原则判定；无法溯源者标 C，厂商自报者标 B。*
