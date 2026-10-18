---
title: "推理引擎（04）：支持矩阵（下）——API、可观测与迁移的真实成本"
date: 2026-10-18 19:00:00 +0800
categories:
  - 推理服务
tags: [serving-engine, vllm, sglang, api-compatibility, structured-output, lora, observability, migration, inference]
layout: post
mathjax: true
---

> **推理引擎系列 · 第 04 篇**
>
> 本系列面向**已经能把模型跑起来、但还没读过任何一个推理引擎源码**的工程师：把 vLLM 与
> SGLang 的功能差异、性能差异、选型结论，全部还原到**源码里可点开核验的那一行**。
>
> 取证基准：**vLLM `v0.31.0`**（commit `db9527a4`）／ **SGLang `v0.5.21`**（commit `e00930c5`）。
> 本文所有行号均锁在这两个 commit 上。
>
> - 00 引擎在栈里的位置 → 01 架构分歧的根 → 02 调度与批处理 → 03 支持矩阵（上）→ **04 支持矩阵（下）（本文）** →
>   05 PD 分离 → 06 benchmark 交叉验证 → 07 选型手册 → 08 前缀命中率

**TL;DR**

> * **迁移的真实成本不在端点数量，在同名参数的语义差**：`enable_prefix_caching` 与
>   `disable_radix_cache` 一个正一个反、默认值都是「开」；`async_scheduling` 与
>   `disable_overlap_schedule` 同样一正一反、默认值都是「关/开相反」。
>   **照抄参数名会得到相反的行为**。
> * **结构化输出的同名参数，两边支持的后端集合不同**：`disable_additional_properties`
>   在 vLLM 仅 `guidance` 支持，在 SGLang 是 `xgrammar` + `llguidance`。
>   同名字段静默不生效，比报错更难查。
> * **可观测性 38 vs 156 不是「多少」的问题，是「有没有那一类」的问题**：
>   vLLM 没有 grammar 编译耗时、KV 传输分段耗时这两类指标，做 PD 分离容量规划会很被动。
> * **硬件平台恰好互补**：vLLM 有 TPU（但只是委托给 PyPI 包的 20 行文件）无 NPU；
>   SGLang 有 NPU 无 TPU（TPU 在独立仓库 `sglang-jax`）。
> * **三处必须纠正的预设**：vLLM 也有 `return_sampling_mask`（18 文件完整链路）；
>   TokenSpeed MLA 两边都有；**两个 metrics 路径都不存在**（一个迁走了、一个内联散布）。

---

## 1. 论点：端点数量不构成差异，默认值与语义才构成

「都兼容 OpenAI API，改个 base_url 就能迁」是选型讨论里最常见的一句话。
把它和源码对照之后，结论是：**端点层面的兼容基本是真的，迁移成本全在端点之外**。

三个层面，按成本从低到高：

| 层面 | 兼容程度 | 典型成本 |
|:---|:---|:---|
| 端点存在性 | 高（核心 `/v1/*` 两边都有） | 几乎为零 |
| 参数名与默认值 | 中（同名参数语义可能相反） | 行为静默反转，难排查 |
| 行为与副作用 | 低（后端选择、存储默认值、版本漂移） | 需要逐项核对 |

本篇按这个顺序展开：先讲参数语义（§2–§3），再讲可观测与 API 组织（§4–§5），
最后给一张**升级/迁移必查的默认开关表**（§7）——那张表是本篇最该被单独摘出去用的部分。

---

## 2. 结构化输出：同名参数的后端集合不一致

### 2.1 两边各有四个后端，但命名不同

**vLLM**：4 个后端，分派链在 `vllm/v1/structured_output/__init__.py:134-160`——
`xgrammar`(`:134`) → `guidance`(`:140`) → `outlines`(`:146`，懒加载) →
`lm_format_enforcer`(`:155`，懒加载)。
约束类型枚举 `StructuredOutputOptions` 在 `backend_types.py:19-25`：
`JSON`、`JSON_OBJECT`、`REGEX`、`GRAMMAR`、`CHOICE`、`STRUCTURAL_TAG`。

**SGLang**：`GRAMMAR_BACKEND_CHOICES = ["xgrammar","outlines","llguidance","none"]`
（`python/sglang/srt/arg_groups/choices.py:129`），
分派在 `base_grammar_backend.py`：`outlines`(`:366-367`) → `xgrammar`(`:373-374`) →
`llguidance`(`:405-406`)。

两个命名差异先记住：**SGLang 有 `none` 选项**（vLLM 无）；
**SGLang 叫 `llguidance`，vLLM 叫 `guidance`**——同一个项目的两个名字，
迁移时按字符串匹配会漏。

### 2.2 ⚠️ 同名参数的支持后端集合不一致

这是迁移时最容易踩的坑，三行表说完：

| 参数 | vLLM 支持的后端 | SGLang 支持的后端 |
|:---|:---|:---|
| `disable_any_whitespace` | `xgrammar` + `guidance` | `outlines` + `llguidance` |
| `disable_additional_properties` | **仅** `guidance` | `xgrammar` + `llguidance` |
| `max_whitespace` | 未提供 | **仅** `xgrammar` |

vLLM 侧出处 `vllm/config/structured_outputs.py:63-71`；
SGLang 侧 `python/sglang/srt/arg_groups/fields/serving.py:280-288`
（该处注释直接点名后端）。

> **为什么这比报错更难查**：参数名合法、值合法、请求成功返回——
> 只是约束没生效。客户端以为 JSON 里不会有额外字段，实际上有。
> 这类问题只有在输出 schema 校验那一层才会暴露，且是概率性的。

另外两条：**Lark grammar 两边都未找到支持**；
SGLang 独有 `reasoner_grammar_backend.py`（思维链模型的结构化输出适配）与
`outlines_jump_forward.py`，以及 **per-backend 的 schema 校验**
（`json_schema_validation.py`：`validate_xgrammar_json_schema()` `:167`、
`validate_outlines_json_schema()` `:203`）——
**同一个 schema 在不同后端上的合法性判定可能不同**，SGLang 把这件事显式化了。

---

## 3. LoRA：多租户的结构性差异

单 adapter 场景两边都能用；差别在**多租户**——一个基座 + 几十个业务 adapter：

| | vLLM | SGLang |
|:---|:---|:---|
| 并发 adapter 数 | `max_loras` 默认 **1**（`config/lora.py:37`） | `max_loras_per_batch` 默认 **8**（`fields/lora.py:66`） |
| rank 上限 | `max_lora_rank` 固定 **16**（`config/lora.py:35`） | 默认 **None，从 `--lora-paths` 自动推断**（`fields/lora.py:42-45`） |
| CPU 侧上限 | `max_cpu_loras`（`config/lora.py:122-126`） | `max_loaded_loras`（`fields/lora.py:62-65`） |
| 淘汰策略 | 未见同名参数 | `lora_eviction_policy`：`lru`（默认）/ `fifo`（`fields/lora.py:70-76`） |
| kernel 后端可换 | 否（Triton 固定） | **是**：`lora_backend` = `triton`/`csgmv`/`ascend`/`torch_native`，默认 `csgmv`（`fields/lora.py:77-83`） |
| chunk 大小 | — | `max_lora_chunk_size` = 16/32/64/128，默认 16（`fields/lora.py:84-90`） |

> **选型含义**：多租户 LoRA 服务（一个基座 + 几十个业务 adapter）是 SGLang 的
> 结构性优势项——默认就支持 8 个 adapter 同 batch，vLLM 默认只支持 1 个。
> 参数名还不同（`max_loras` vs `max_loras_per_batch`），**照抄名字会直接启动失败**，
> 这倒是几种坑里最好的一种。

SGLang 的 LoRA 后端目录 `python/sglang/srt/lora/backend/` 有 7 个实现
（`triton` / `chunked` / `ascend` / `torch` / `uno_cublas` + `base_backend` + `lora_registry`），
另有 `lmhead_mixing.py`、`lora_overlap_loader.py`、`lora_drainer.py`、
`lora_moe_runner_marlin.py`——**overlap loader 与 drainer 说明它把 LoRA 加载
也放进了 overlap 调度**（与第 02 篇的 overlap 主线一致）。
vLLM 侧 `vllm/lora/ops/triton_ops/` 有 8 个 kernel 文件
（expand / shrink / fused_moe × 4 + metadata + fp8 utils）。

---

## 4. 可观测性：38 vs 156

### 4.1 ⚠️ 两个路径都不存在

先纠正两个常见预设：

- `vllm/engine/metrics.py` **不存在** → 已迁移到 **`vllm/v1/metrics/`**
  （7 个文件：`loggers.py`（1309 行，指标定义主体）、`prometheus.py`(82)、
  `stats.py`、`buckets.py`、`perf.py`、`reader.py`、`ray_wrappers.py`）；
- `python/sglang/srt/metrics/` **不存在** → 指标以 `name="sglang:*"`
  **内联散布**在 `python/sglang/srt/` 各处。

**按文件名找指标定义，两边都会扑空**——一个搬了家，一个根本没有家。

### 4.2 数量与差异

vLLM 核心引擎 `name="vllm:*"` 去重得 **38 个**（仅 `v1/metrics/loggers.py`；
`kv_connector/v1/metrics.py`、`orca_metrics.py`、realtime 等处另有，故总数 ≥ 38）。
`prometheus.py` 只做 multiprocess 管理（`setup_multiprocess_prometheus()` `:17`、
`get_prometheus_registry()` `:39`、`unregister_vllm_metrics()` `:56`、
`shutdown_prometheus()` `:72`）。

SGLang 全树去重得 **156 个**。多出的是**整类**指标，不是零头：

| 类别 | 代表指标 |
|:---|:---|
| **grammar（10 个）** | `grammar_compilation_time_seconds`、`grammar_ebnf_size`、`grammar_tree_traversal_time_avg`/`_max`、`num_grammar_aborted_total`、`num_grammar_cache_hit_total`、`num_grammar_queue_reqs` |
| **hicache / 存储（约 20 个）** | `hicache_backup_*`(4)、`hicache_host_*`(2)、`storage_prefetch_*`(3)、`load_back_*`(3)、`prefetch_bandwidth`、`hicache_dropped_tokens_total` |
| **KV 传输 / PD（9 个）** | `kv_transfer_alloc_ms`、`kv_transfer_bootstrap_ms`、`kv_transfer_latency_ms`、`kv_transfer_speed_gb_s`、`num_bootstrap_failed_reqs_total`、`per_stage_req_latency_seconds` |
| **投机解码（7 个）** | `spec_accept_length`、`spec_accept_rate`、`spec_block_accept_length`、`spec_cap_length`、`spec_num_draft_tokens`、`spec_num_steps`、`spec_verify_calls_total` |
| **PD 队列深度（5 个）** | `num_decode_transfer_queue_reqs`、`num_prefill_bootstrap_queue_reqs` 等 |
| LoRA（3 个） | `lora_pool_slots_total`、`lora_pool_slots_used`、`lora_pool_utilization` |
| **encoder（14 个）** | `encoder_*` |
| 路由（4 个） | `num_unique_running_routing_keys`、`routing_key_all_req_count` 等 |
| 其它 | `eplb_balancedness`、`fwd_occupancy`、`estimated_flops_per_gpu_total`、`prefill_delayer_*`(3)、`swa_*`(4)、`mamba_*`(4) |

> **选型含义**：如果可观测性方案要「从引擎指标直接算 SLO」，
> SGLang 能直接给的指标多得多——尤其 **KV 传输分段耗时**与 **grammar 编译耗时**，
> 这两个在 vLLM 侧没有对应指标，做 PD 分离的容量规划会很被动。
> 反过来，只做单模型服务的话 vLLM 的 38 个覆盖核心请求/缓存/抢占，**是够用的**。

### 4.3 开关与默认值

**SGLang**：`--enable-metrics` 默认 **False**
（`python/sglang/srt/arg_groups/fields/observability.py:70`）；
`--enable-metrics-for-all-schedulers`（`:82`）开启后所有 TP rank 而非仅 TP 0
单独记录请求指标，`dp_attention` 下尤其有用。

**vLLM**：未找到同名开关，Prometheus 导出为常开 + multiprocess 管理。

> **一个迁移陷阱**：从 vLLM 切到 SGLang，指标**默认是全关的**。
> 灰度对比期如果忘了 `--enable-metrics`，会得到「SGLang 没有任何指标」的假象。

SGLang 的 Rust 网关还有独立 metrics 实现
（`sgl-model-gateway/src/observability/metrics.rs`，用 `metrics` +
`metrics_exporter_prometheus`，`counter!`/`gauge!`/`histogram!` 分别出现
45 / 31 / 23 处，含 `STRING_INTERNER` 标签 interning 以控制基数，L28），
另有 `src/core/metrics_aggregator.rs`。
⚠️ **具体 router 指标名称未取到**（字符串字面量 grep 命中 0）。

---

## 5. API 兼容面：组织方式比端点列表更重要

### 5.1 两种组织哲学

**vLLM 已重构为按域拆分的 `api_router.py`**，共 **30 个**，各以 `attach_router(app)` 挂载。
**SGLang 是单文件集中注册**——`python/sglang/srt/entrypoints/http_server.py`
（2936 行），共 **55 个**路由装饰器。

这个差异的后果不在功能，在**可维护性**：vLLM 加一个域是加一个文件，
SGLang 加一个端点是往一个近三千行的文件里加几行——
单人 review 时后者的 diff 更难读，但 grep 一个端点名时后者更快找到。

### 5.2 OpenAI 兼容端点对照

| 端点 | vLLM | SGLang |
|:---|:---|:---|
| `/v1/chat/completions` | `openai/chat_completion/api_router.py:41` | `http_server.py:1777` |
| `/v1/completions` | `openai/completion/api_router.py:35` | `http_server.py:1769` |
| `/v1/embeddings` | `pooling/embed/api_router.py:29` | `http_server.py:1787-1791` |
| `/v1/models` | `openai/models/api_router.py:20` | `http_server.py:1898` |
| `/v1/responses` | `openai/responses/api_router.py:48` | `http_server.py:1971` |
| `/v1/responses/{id}` | `:80` | `:1990` |
| `/v1/responses/{id}/cancel` | `:110` | `:1998` |
| `/v1/score` | `pooling/scoring/api_router.py:53` | `http_server.py:1955` |
| `/v1/rerank` | `pooling/scoring/api_router.py:85` | **未找到** |
| `/v1/classify` | `pooling/classify/api_router.py` | `http_server.py:1799-1803` |
| `/v1/pooling` | `pooling/pooling/api_router.py` | **未找到** |
| `/v1/audio/transcriptions` | `speech_to_text/transcription/api_router.py:32` | `http_server.py:1847` |
| `/v1/tokenize` + `/tokenize` | `serve/tokenize/` | `http_server.py:1811-1821`（双路由） |
| `/v1/detokenize` + `/detokenize` | — | `http_server.py:1829-1839`（双路由） |

**SGLang 独有扩展**：`/v1/decisions`（`:1963`，chat 模型当分类器/打分器、不做生成）、
`/v1/systemone`（`:2086`）、`/v1/messages`（`:2065`，Anthropic 风格）、
`/api/chat` 等（`:2036` 起，**Ollama 兼容**）、`/vertex_generate`（`:2112`）、
`/hicache/storage-backend`（`:1146`）。

`/v1/decisions` 的机制值得一说（v0.5.21 新增）：实现类
`OpenAIServingDecisions`（`python/sglang/srt/entrypoints/openai/serving_decisions.py:64`）
**复用 chat serving**（构造函数 `:70`），官方文档
（`docs/docs/supported-models/decision_models.mdx:6`）的说法是
「each answer comes back with the probability of every option, read from the model's
**next-token scores at the answer position**」——即复用 scoring 路径，不解析生成输出。
⚠️ 文档与 tag 有错位：该文档仍写 “install a nightly build”，
**不应断言「v0.5.21 已正式发布该端点」**；且文档明确它不是 OpenAI API 的一部分。

---

## 6. 硬件平台：恰好互补

| | vLLM | SGLang |
|:---|:---|:---|
| 平台文件 | `vllm/platforms/`：`__init__` `interface` `cuda` `rocm` `cpu` `xpu` **`tpu`** `zen_cpu` | `python/sglang/srt/platforms/`：`__init__` `interface` `device_mixin` `cuda` `rocm` `cpu` **`npu`** `xpu` |
| **TPU** | ✅ `tpu.py`（**全文仅 20 行**） | ❌ 主仓**无** TPU 代码 |
| **昇腾 NPU** | ❌ `platforms/` 下无 npu/ascend | ✅ `npu.py` + `disaggregation/ascend/` + `lora/backend/ascend_backend.py` + 专用量化 |

### 6.1 TPU：形态完全不同的两种「支持」

**vLLM——委托给 PyPI 包**。`vllm/platforms/tpu.py` 全文 20 行，核心是 try/except：

```python
from tpu_inference.platforms import TpuPlatform as TpuInferencePlatform
TpuPlatform = TpuInferencePlatform
USE_TPU_INFERENCE = True
```

（`:10-15`；ImportError 时 log error 后 `pass`，`:17-19`）
即 **vLLM 仓库内无 XLA/TPU kernel 实现**。另支持 Pathways TPU proxy
（`__init__.py:36-50`，`VLLM_TPU_USING_PATHWAYS` `:37`）。

**SGLang——独立仓库 `sgl-project/sglang-jax`**。**不是同仓库子目录**
（顶层目录实测无 `jax`；全仓 grep `jax|tpu` 只命中 docs）。
官方文档（`docs/docs/hardware-platforms/tpu.mdx:9`）原文：
「SGLang TPU support is implemented via the **SGLang-JAX backend, a dedicated
JAX-based inference engine maintained as a separate repository**」。
安装 `pip install sglang-jax`（`:234`）；⚠️ TPU Docker 支持仍在开发中（`:248`）。

> **选型含义**：TPU 上 vLLM 与 SGLang 是**两个不同项目的竞争关系**，
> 不是「同一个项目的两条后端路径」。比 TPU 性能时，实际在比
> `tpu_inference` 包与 `sglang-jax` 仓库——这超出了本篇的取证范围。

⚠️ 「vLLM 无 NPU platform 文件」这一结论的搜索范围是 `vllm/platforms/` 目录
ls-tree，未穷尽全库。

---

## 7. 默认开关对照（升级/迁移必查）

这是本篇最该被单独摘出去用的一张表。**两边的默认值多处相反**：

| 特性 | vLLM `v0.31.0` | SGLang `v0.5.21` |
|:---|:---|:---|
| prefix caching | **默认开**（`config/cache.py:142`） | **默认开**（`fields/memory.py:64-70`） |
| chunked prefill | **默认开**（`DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048`，`config/scheduler.py:42`） | **默认开**（`chunked_prefill_size` 默认 `None`；`-1` 才禁，`fields/schedule.py:52-55`） |
| CPU/GPU overlap | **默认关**（`async_scheduling=None`，`config/scheduler.py:209`） | **默认开**（`disable_overlap_schedule=False`，`fields/schedule.py:193`） |
| 长 prefill 公平性阈值 | **默认关**（`long_prefill_token_threshold=0`，`config/scheduler.py:80`；adaptive 也默认 `False` `:87`） | 不适用（用 9 种 schedule-policy 替代） |
| prefix-aware 调度 | 无 | **默认关**（`schedule_policy="fcfs"`，`fields/schedule.py:98`） |
| 分层 KV / HiCache | 无内建分层 | **默认关**（`enable_hierarchical_cache=False`，`fields/memory.py:99`） |
| PD role switching | **该 commit 未找到** | **默认关**（`enable_pd_role_switch=False`，`fields/disagg.py:115`） |
| 投机解码 | 默认关 | 默认关（`SpeculativeAlgorithm.NONE`） |
| Prometheus 指标 | 常开 | **默认关**（`--enable-metrics`，`fields/observability.py:70`） |
| 多 LoRA 并发 | **1**（`config/lora.py:37`） | **8**（`fields/lora.py:66`） |

### 7.1 四组「一正一反」的参数对

迁移时照抄参数名最危险的就是这四组——**名字相反、默认值也可能相反**：

| vLLM | SGLang | 关系 |
|:---|:---|:---|
| `enable_prefix_caching` | `disable_radix_cache` | 一正一反，**默认都是「开」** |
| `async_scheduling` | `disable_overlap_schedule` | 一正一反，**默认都是「关/开相反」**（vLLM 关 overlap、SGLang 开） |
| `max_loras` | `max_loras_per_batch` | 名字不同，**默认值差 8 倍** |
| `max_lora_rank`（固定 16） | `max_lora_rank`（自动推断） | 同名，**语义不同** |

> **迁移 checklist 的第一条应该是**：把两边所有「语义相反」的参数对列出来，
> 逐对确认目标值，而不是逐参数抄名字。第 07 篇会给完整的 checklist。

### 7.2 近期 breaking change（版本相关性的实证）

**vLLM v0.31.0 移除/收紧**：多模态 per-request kwargs **默认拒绝**
（需 `--trust-request-mm-kwargs`）；`tokenizer_mode="slow"` 移除；
`--enable-mamba-fine-grained-prefix-cache` 移除；`quantization="fp8"` 改 spec；
Quark 量化改隐式在线量化；AllSpark INT8 W8A16 移除；`--enforce-eager` 移除；
`VLLM_XPU_ENABLE_XPU_GRAPH` 移除。

更早的节奏：v0.30.0 引入「Fast Start」与 `HiSparse`；v0.29.0 把
**Model Runner V2 设为所有模型默认**；v0.25.0 把 V2 设为 dense 模型默认并
「deprecate PagedAttention」；v0.26.0 引入 Rust frontend（多模态）；
v0.21 周期引入 KV offloading（Hybrid Memory Allocator、Mooncake）、
RayExecutorV2 默认。

**SGLang 侧**：v0.5.20 把 `/v1/responses` 服务端存储**默认关闭**
（需 `--enable-response-store`；依赖存储的请求返回 400，PD 部署下不支持该 flag），
同时移除 prefill CP v1 与 CUDA 12 产物；v0.5.21 移除
`SGLANG_EXPERIMENTAL_CPP_RADIX_TREE`、`SWARadixCache` / `MambaRadixCache`
（并入 `UnifiedRadixCache`）、`HiRadixCache` 及所有废弃 2+ 版本的别名。

> **「vLLM 支持某模型」这句话必须带版本号**——这是本篇和上一篇共同的结论。
> 两边的默认值与注册表都在以月为单位变动。

---

## 8. 取证边界与三处纠正

| 项 | 状态 |
|:---|:---|
| vLLM `gguf` / `bitsandbytes` 量化 | **确认不存在**（目录与枚举 AST 均已完整枚举） |
| SGLang `torchao` | **确认不存在**（以 `humming`/`modelslim`/`mlx` 替代） |
| vLLM NPU platform | **未找到**（搜索范围 `platforms/` 目录 ls-tree） |
| vLLM `shared_storage` connector | **未找到**（16 个 connector 全枚举，见第 05 篇） |
| vLLM `sampling_mask_max_tokens` | **未找到**（全树 grep 0 命中） |
| Lark grammar（两边） | **未找到** |
| vLLM PD 拓扑文档 | **未找到**（`docs/` 全树文件名 grep 命中 0） |
| SGLang router 指标名称 | **存在但未定位到路径**（`metrics.rs` 用 `metrics_exporter_prometheus`，字面量 grep 命中 0） |
| `sgl-project/sglang-jax` 内部实现 | **未取证**（不在两仓范围内） |

三处需要纠正的常见预设：

1. **vLLM 有 `return_sampling_mask`**（配置字段 `vllm/config/model.py:252`，
   18 个文件的完整链路，sampler 核心 `vllm/v1/worker/gpu/sample/sampler.py:201-205`）。
   真正差异是 vLLM 无 token 上限参数、且开启后会关闭 flashinfer/xpu sampler
   快速路径（`sampler.py:89`、`:93`）、与 batch-sharded gather 不兼容
   （`config/vllm.py:3205-3208`）。
2. **TokenSpeed MLA 两边都有**（vLLM `registry.py:75`、SGLang `attention_registry.py:96`），
   但 SGLang 版仅限 MLA 模型（`:99` 抛错）。
3. **两个 metrics 路径都不存在**：vLLM 迁到 `vllm/v1/metrics/`，
   SGLang 是内联散布——按老路径找指标定义会扑空。

---

**上一篇**：[推理引擎（03）：支持矩阵（上）——「200+ 模型」为什么不可比](/2026/10/18/serving-03-support-matrix-models/)

**下一篇**：[推理引擎（05）：PD 分离与 KV 传输](/2026/10/18/serving-05-pd-disaggregation-kv-transfer/)

**站内承接**：调度与默认开关的机制见
[《推理引擎（02）》](/2026/10/18/serving-02-scheduling-and-batching/)；
注册机制与架构计数见
[《推理引擎（03）》](/2026/10/18/serving-03-support-matrix-models/)；
投机解码的引擎支持见
[《投机解码（04）》](/2026/10/18/specdec-04-engine-support-matrix/)；
结构化输出与投机的交互见
[《投机解码（05）：生产部署的陷阱》](/2026/10/18/specdec-05-production-traps/)；
指标口径见[《性能分析（02）：推理》](/2026/09/29/profiling-02-inference/)。

## 参考

1. vLLM `v0.31.0`，`vllm/v1/structured_output/__init__.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/structured_output/__init__.py>
2. vLLM `v0.31.0`，`vllm/config/structured_outputs.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/structured_outputs.py>
3. vLLM `v0.31.0`，`vllm/config/lora.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/lora.py>
4. vLLM `v0.31.0`，`vllm/v1/metrics/loggers.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/metrics/loggers.py>
5. vLLM `v0.31.0`，`vllm/platforms/tpu.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/platforms/tpu.py>
6. SGLang `v0.5.21`，`python/sglang/srt/arg_groups/fields/serving.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/arg_groups/fields/serving.py>
7. SGLang `v0.5.21`，`python/sglang/srt/arg_groups/fields/lora.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/arg_groups/fields/lora.py>
8. SGLang `v0.5.21`，`python/sglang/srt/arg_groups/fields/observability.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/arg_groups/fields/observability.py>
9. SGLang `v0.5.21`，`python/sglang/srt/entrypoints/http_server.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/entrypoints/http_server.py>
10. SGLang decision models 文档 — <https://docs.sglang.io/supported-models/decision_models/>
11. SGLang TPU 文档（独立仓库说明）— <https://docs.sglang.io/hardware_platforms/tpu/>
12. SGLang PD 分离文档 — <https://docs.sglang.io/advanced_features/pd_disaggregation/>
13. vLLM LoRA 文档 — <https://docs.vllm.ai/en/latest/features/lora.html>
14. SGLang LoRA 文档 — <https://docs.sglang.ai/advanced_features/LoRA/>
