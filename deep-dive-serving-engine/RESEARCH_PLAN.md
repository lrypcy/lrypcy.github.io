# Deep Dive 研究计划：SGLang vs vLLM

> 目标读者：会部署 LLM 推理服务、需要在两个引擎之间做决策的 AI Infra 工程师。
> 主视角：**功能与支持矩阵优先**（这是选型问题的真实约束）。
> 性能：**汇总各家公开 benchmark 做交叉表**，明确标注口径差异，不在本机实跑（macOS 无多卡 GPU）。
> 基准：源码锁到 release tag，行号引用给 permalink。

---

## 0. 基准与口径（写作前先钉死）

| 项目 | vLLM | SGLang |
| --- | --- | --- |
| 仓库 | `vllm-project/vllm` | `sgl-project/sglang` |
| 锁定 tag | `v0.31.0`（2026-10-05） | `v0.5.21`（2026-10-02） |
| 锁定 commit | 待 fetch 后填 | 待 fetch 后填 |
| License | Apache-2.0 | Apache-2.0 |

**性能数字的处理原则**（本篇的诚实性基础）：

1. 公开 benchmark 几乎**全部**有厂商或自媒体倾向，标题党多。本篇只引用：
   - 可追溯到具体 commit / 硬件 / 模型 / 并发度的第三方实测；
   - 官方 release notes / 博客自报数据（**必须标注「官方自报」**）。
2. 每个数字必须带**四元组**：`模型 + 精度/量化 + 硬件/卡数 + 并发度`。缺任一项的数字一律不引用。
3. **不同来源不可直接相减**。ShareGPT 类负载与 RAG 类负载的差距方向相反，必须分开列表。
4. 明确写出一节「**这些数字为什么互相矛盾**」，解释口径差异来源（prefix 命中率、是否开投机解码、chunked prefill 阈值、是否 PD 分离）。

---

## 1. 文档结构

```
deep-dive-serving-engine/
├── README.md                      # 阅读指南 + 选型速查表
├── RESEARCH_PLAN.md               # 本文件
├── PROGRESS.md                    # 进度与待核清单
├── 01-architecture-divergence.md  # 架构分歧的根：三条设计路线
├── 02-feature-matrix.md           # 功能支持矩阵（主篇）
├── 03-model-hardware-support.md   # 模型与硬件覆盖
├── 04-scheduling-and-cache.md     # 调度器与 KV 缓存机制
├── 05-speculative-decoding.md     # 投机解码实现对比
├── 06-pd-disaggregation.md        # PD 分离与大规模部署
├── 07-quantization-and-kernels.md # 量化格式与 kernel 生态
├── 08-benchmark-crosscheck.md     # 公开 benchmark 交叉表
├── 09-production-and-ecosystem.md # 生产运维、周边生态、社区
└── 10-selection-playbook.md       # 选型决策树 + 迁移路径
```

---

## 2. 各篇要点

### 01-architecture-divergence.md —— 三条设计路线

不是「谁更先进」，而是三个独立选择：

- **KV 缓存组织**：vLLM = 固定 block + hash 索引（块粒度复用）；
  SGLang = radix tree over token 序列（token 粒度复用，自动最长前缀匹配）。
  要讲清 radix tree 的 **node 切分规则**（在什么位置断 child）与 **LRU 淘汰**，
  以及为什么这在「前缀不完全对齐」时反而可能不如 block hash。
- **调度器**：vLLM V1 = Scheduler / KVCacheManager / BlockPool 三段式；
  SGLang = overlap scheduler（CPU 跑在 GPU 前面一个 batch）+ prefix-aware 策略（`lpm` 等）。
  **overlap 不是无条件赢**——要引 issue #2558（小模型单卡反而变慢）作为反证。
- **前后端进程模型**：vLLM V1 = EngineCore 独立进程（每 DP rank 一个）+ 前端解耦；
  SGLang = Tokio runtime + Rust 前端/router（`sgl-router`）+ `sgl-kernel` 独立 ABI 包。
  这一点直接影响「引擎崩了会不会带崩整个 server」——vLLM 有 `fault_tolerance/engine_core_sentinel.py`。

### 02-feature-matrix.md —— 主篇

一张大表，行为能力维度，列为 `vLLM / SGLang / 说明`：

- API 兼容面：Chat/Completions、Responses API、Embeddings、多模态、音频
- 批处理与调度：continuous batching、chunked prefill、PD 分离、priority scheduling
- 缓存：APC / radix cache、HiCache 分层、KV offload（Hybrid Memory Allocator、Mooncake）
- 解码加速：n-gram / EAGLE / Medusa / MTP / DSpark / 自适应投机
- 约束解码：xgrammar / outlines / guidance，支持哪些 grammar
- LoRA / 多适配器：热切换、多 LoRA batch
- RL / 训练侧：rollout 后端、权重同步（`return_sampling_mask` 这类特性）
- 可观测性：Prometheus 指标、OTel、tracing
- 弹性与运维：多节点、DP coordinator、Ray、Helm/K8s 成熟度

每行都要**两个仓库各给一个 permalink**（文档或源码行号）。

### 03-model-hardware-support.md

- 模型：新架构首发支持谁更快（找 day-0 支持的 PR 号）
- 硬件矩阵：NVIDIA 全系（Hopper / Blackwell / GB 系列）、AMD ROCm（含 gfx950）、Intel XPU、
  TPU（vLLM 主线 vs SGLang 走 SGLang-Jax）、Apple Silicon、CPU、国产 NPU
- 量化格式逐个对比：FP8（E4M3）、NVFP4、MXFP4、INT8（W8A8）、W4A16、AWQ、GPTQ、bitsandbytes、GGUF
- **边界要诚实写**：vLLM 0.31.0 已退役 CUDA 12 产物；SGLang 0.5.20 移除 prefill CP v1。
  这类 breaking change 是「选型时看版本」的实证论据。

### 04-scheduling-and-cache.md

- 两套缓存的**命中语义差异**：hash 需精确对齐 block 边界，radix tree 任意 token 位置切分
- 什么负载下 radix 的优势归零（唯一 prompt）、什么负载下放大（RAG / 多轮 / agent）
- 缓存淘汰策略对交互式负载的影响
- 度量方法：怎么在生产里测自己的 prefix 命中率（tokenize 一天真实请求数公共前缀 token 数）

### 05-speculative-decoding.md

- 两者支持的 drafter 列表与各自的实现路径（内置 vs 插件）
- 官方自报加速比（标注官方自报）
- 一个反直觉点：Mamba/hybrid 模型上开 APC 或 MTP 的已知问题（引 issue）

### 06-pd-disaggregation.md

- 两边的 PD 分离实现：KV 传输通道（NIXL / Mooncake / LMCache）、拓扑要求、失败模式
- vLLM 的 DP coordinator 与 vLLM 的 DP attention
- SGLang 的 **dynamic PD role switching**（v0.5.21 新增，运行时切换 prefill/decode 角色不重启）
  —— 这是 v0.5.21 最值得写的一条新特性

### 07-quantization-and-kernels.md

- kernel 来源：vLLM = `csrc/` + 大量 backend 插件；SGLang = 独立 `sgl-kernel` 包
- attention backend 可选矩阵对照（FlashInfer / FlashAttention / MLA / TokenSpeed 等）
- FP8 KV cache、NVFP4 compressed KV cache 这类 2026 新路径两边进展

### 08-benchmark-crosscheck.md

交叉表，来源至少覆盖：IoT Digital Twin 8×H200 70B-4bit、Turion Q2 2026、
PremAI H100 Llama-3.1-8B（多站转载）、markaicode Llama-3.3-70B FP8、agentnative 汇总、
flozi 架构对比、spheron 2026、官方 release notes 自报。

**每个数字标注可信度等级**：`第三方实测` / `官方自报` / `二手转载（存疑）`。

### 09-production-and-ecosystem.md

- K8s / Helm / 官方镜像成熟度
- 社区规模与 contributor 结构（用 GitHub API 客观统计，不用「~15k star」这种约数）
- 治理与商业化动向（要标注时间与出处）
- 安全记录（vLLM 2026-08~10 的 CVE 情况，需核实具体条目）

### 10-selection-playbook.md —— 落地篇

- 决策树：模型规模 / 负载形状 / 硬件 / 团队能力 四个维度
- 「先 vLLM 起步，需要时再切」这个默认建议在什么条件下成立、什么条件下不成立
- 迁移成本核算：OpenAI API 兼容 ≠ 零成本，列出会真正卡住的地方（prefix cache 语义、采样参数、
  structured output 行为差异、`extra_body` 私有参数）
- 一份可直接用的 checklist

---

## 3. 核心问题清单

1. 两套 KV 缓存的复用粒度差异，在**什么 prefix 命中率区间**产生可测的端到端差距？
2. SGLang 的 overlap scheduler 在什么条件下反而劣化（要有 issue 级证据）？
3. vLLM EngineCore 进程隔离带来的故障隔离，实际挡住了什么故障？
4. 「vLLM 支持 200+ 架构」这个说法在 2026 年还成立吗？需要用源码里的模型注册表核对。
5. 两边对同一模型的首日支持差距，是工程投入差异还是纯人力差异？
6. SGLang 的 Rust 前端 / Rust radix cache 替换（v0.5.21）带来了多少可测收益？
7. 国产 NPU（昇腾）在两个框架上的支持现状如何？（对国内用户是决定性的）
8. RL rollout 场景下，两边分别需要绕什么坑？
9. TPU 上 vLLM 主线与 SGLang-Jax 的实际差距有多大？
10. 从一个框架迁到另一个，真实会卡住的地方清单是什么？

---

## 4. Lab 设想（本机可跑的部分）

macOS 无 GPU，但以下实验可以真实跑：

1. **前缀分析器**：输入一段真实请求日志，tokenize 后统计两两请求的公共前缀 token 数，
   估算 prefix cache 理论命中率 → 直接映射到「该选谁」。
2. **静态差异对照**：同一批模型 config，分别用两边的 model registry 判断支持状态
   （纯 Python import，不需 GPU）。
3. **迁移兼容性检查器**：给定一组 OpenAI API 请求，标出哪些参数在两边语义不同。

---

## 5. 写作红线（沿用 `.agents/WRITING.md`）

- 不写第一人称叙述生成过程
- 公开 benchmark 数字**必须**带四元组 + 可信度等级；无法核实的标「待验证」
- 引用源码锁 commit + permalink 行号；上游自带 typo 原样保留
- 行内公式 `$$...$$`，LaTeX 用 `\mathrm{}` 不用`{\rm }`；正文禁裸双花括号与 Liquid 开标记
- Mermaid 只用 `<br>`，节点含括号加引号
- 草稿目录已在 `_config.yml` 的 `exclude` 里（`deep-dive-*` 通配，无需改）
- 交稿前跑 `python3 tools/check_liquid.py` 与 `tools/check_post.py`