# 02 功能与支持矩阵

> 基准：vLLM `v0.31.0`（`db9527a4`）／ SGLang `v0.5.21`（`e00930c5`）。
> 本文是主篇。所有单元格都带源码行号，可点开permalink 核验。

先给一张速览，再逐维度展开。

## 0.速览：数量对照

| 维度 | vLLM | SGLang | 口径说明 |
|:---|---:|---:|:---|
| 总文件数 | 7431 | 9912 | `git ls-tree -r` |
| `.py` 文件数 | 4880 | 7024 | |
| 模型架构 | **387** | **282** | ⚠️ 口径不同，见§1 |
| Attention backend | **44** | **24** | vLLM 用 Enum，SGLang 用装饰器 |
| PD / KV 传输 connector | **16** | **5** | vLLM 含 push/pull 双向 |
| 量化方法 | **30** | 25 base + 平台追加 | 机制不同，见 §7 |
| 结构化输出后端 | 4 | 4 | 命名与能力档位都不同 |
| Prometheus 指标 | 38+ | **156** | 覆盖范围差异巨大，见 §9 |
| 投机解码算法 | 8 类别名 | **9** 枚举 | SGLang 有插件注册机制 |
| 默认多 LoRA 并发 | **1** | **8** | 参数名也不同 |
| 硬件平台文件 | 6+ TPU | 6+ NPU | 恰好互补 |

---

## 1. 模型架构支持：「200+」这个数字

### 1.1 vLLM：静态显式注册，387 个唯一架构名

vLLM 的注册表 `vllm/model_executor/models/registry.py` 是 **10 个模块级 dict 字面量**
加一个 `**` 合并字典，AST 精确解析得**唯一架构名 387 个**。

| 字典 | 行号区间 | 条目数 |
|:---|:---|---:|
| `_TEXT_GENERATION_MODELS` | L71–217 | 125 |
| `_EMBEDDING_MODELS` | L219–270 | 33 |
| `_LATE_INTERACTION_MODELS` | L272–286 | 11 |
| `_REWARD_MODELS` | L288–292 | 3 |
| `_TOKEN_CLASSIFICATION_MODELS` | L294–313 | 6 |
| `_SEQUENCE_CLASSIFICATION_MODELS` | L315–342 | 10 |
| `_MULTIMODAL_MODELS` | L344–624 | 121 |
| `_SPECULATIVE_DECODING_MODELS` | L626–708 | 63 |
| `_TRANSFORMERS_SUPPORTED_MODELS` | L710–736 | 14 |
| `_TRANSFORMERS_BACKEND_MODELS` | L738–771 | 10 |
| `_VLLM_MODELS`（`**` 合并 10 个） | L773–784 | — |
| `_PREVIOUSLY_SUPPORTED_MODELS`（已移除，附移除版本） | L792–846 | 50 |
| `_OOT_SUPPORTED_MODELS`（转插件） | L848–853 | 4 |

> 条目简单相加为 400，差13 是跨字典重名。

「seamlessly supports 200+ model architectures」这句话的出处在
`docs/README.md:66`。**387 > 200，属实且保守**。

核心类 `_ModelRegistry` 在 `registry.py:1115`，`register_model()` 在 `:1122`，
未支持架构的报错 `_raise_for_unsupported()` 在 `:1164`
（会区分「旧版本支持过」与「已转 OOT 插件」两种情况 —— 这个区分对排查问题很有用）。

### 1.2 SGLang：运行期动态发现，282 个注册条目

SGLang**没有** vLLM 式显式列表。`registry.py` 只有 159 行，
靠 `pkgutil.iter_modules` 遍历 `python/sglang/srt/models/` 下每个模块的
**`EntryClass`** 变量动态注册（读取逻辑 `registry.py:134-147`，支持 list 多类形式 `:137-143`）。

- `import_model_classes()` 在 `:115`（带 `@lru_cache`，`:114`）
- 重复实现断言 `:141`、`:145`；**导入失败仅 warn 不致命** `:129-132`
- 可禁用架构的环境变量 `SGLANG_DISABLED_MODEL_ARCHS` `:124`
- 外部模型包 `SGLANG_EXTERNAL_MODEL_PACKAGE` 分支 `:152-157`

AST 遍历全部 282 个模型文件、解析每个 `EntryClass` 赋值，得**282 个注册条目，分布 221 个模块**。

一个模块可注册多个类（例：`transformers.py:1673` 注册 12 个，`llama.py:928` 注册 4 个，
`deepseek_v2.py:3472` 注册 DeepseekV2 / V3 / V32）。

### 1.3 ⚠️ 这两个数字不能直接相减

- vLLM 的 387 是**唯一架构名字符串**
- SGLang 的 282 是**EntryClass 条目数**

两者口径不同（vLLM 一个 arch 字符串对应一个类文件；SGLang 一个模块可注册多类）。

**不能得出「vLLL 多支持 105 个架构」**。可以确证的是：两者都在数百级，
vLLM 明确更大，但差距不可直接量化。目录规模佐证：vLLM `models/` 有 306 个 `.py`，
SGLang 有 282 个。

---

## 2. 投机解码

### 2.1 vLLM：类型别名 + if-elif 链，无插件机制

**这个 commit 里没有集中的 drafter 注册表字典**。权威清单是 config 里的 `Literal`：

`SpeculativeMethod = Literal[...]` —— `vllm/config/speculative.py:74-85`：

| 类别 | 值 | 行号 |
|:---|:---|:---|
| 内置方法 | `ngram` / `medusa` / `mlp_speculator` / `draft_model` / `suffix` / `custom_class` | `:74-85` |
| Eagle 类| `eagle` / `eagle3` / `extract_hidden_states` + MTP/DFlash | `:71-73` |
| GPU n-gram | `ngram_gpu` | `:68` |
| DSpark | `dspark` | `:70` |

其中 **`MTPModelTypes` 独占 `:39-67`，共 27 个值**：
`deepseek_mtp`、`dots3_note_mtp`、`mimo_mtp`、`mimo_v2_mtp`、`glm4_moe_mtp`、
`glm4_moe_lite_mtp`、`glm_ocr_mtp`、`ernie_mtp`、`nemotron_h_mtp`、`exaone_moe_mtp`、
`exaone4_5_mtp`、`qwen3_next_mtp`、`qwen4_exp_mtp`、`qwen3_5_mtp`、`longcat_flash_mtp`、
`bailing_hybrid_v3_mtp`、`minimax_m3_mtp`、`bailing_hybrid_mtp`、`mtp`、`kimi_k3_mtp`、
`pangu_ultra_moe_mtp`、`step3p5_mtp`、`hy_v3_mtp`、`hy_v4_mtp`、`gemma4_mtp`、
`inkling_mtp`、`glm5_next_mtp`

采样方法：`RejectionSampleMethod = Literal["standard","synthetic","block"]`（`:86`）、
`DraftSampleMethod = Literal["greedy","probabilistic"]`（`:87`）。

实际 dispatch 在 `vllm/v1/worker/gpu_model_runner.py:617-676`的 if-elif 链：

| method | 类 | 行号 |
|:---|:---|:---|
| `custom_class` | `create_custom_proposer` | `:618` |
| `ngram` | `NgramProposer` | `:624` |
| 用 draft model | `DraftModelProposer` | `:626` |
| n-gram GPU | `NgramProposerGPU` | `:632` |
| Gemma4 MTP | `Gemma4Proposer` | `:649` |
| Step3.5 MTP | `Step3p5MTPProposer` | `:651` |
| DFlash | `DFlashProposer` | `:653` |
| `suffix` | `SuffixDecodingProposer` | `:656` |
| Eagle | `EagleProposer` | `:658` |
| `medusa` | `MedusaProposer` | `:664` |
| `extract_hidden_states` | `ExtractHiddenStatesProposer` | `:668` |

实现目录 `vllm/v1/spec_decode/`（22 文件），基类 `SpecDecodeBaseProposer` 在 `llm_base_proposer.py`。

### 2.2 SGLang：真正的插件注册表

`class SpeculativeAlgorithm(Enum)` —— `python/sglang/srt/speculative/spec_info.py:32`，
9 个内置成员（`:39-47`）：`DFLASH`、`UNO`、`DSPARK`、`EAGLE`、`EAGLE3`、
`FROZEN_KV_MTP`、`STANDALONE`、`NGRAM`、`NONE`。

字符串解析入口 `from_string()` 在 `:50`，三级回退（`:57` / `:60-61` / `:62-64`）：
枚举 → 保留别名 → 插件注册表，否则抛 `ValueError`。
保留别名表只有 **一个**：`NEXTN` → EAGLE（`spec_registry.py:183`）。

插件入口是装饰器 `SpeculativeAlgorithm.register(name, *, supports_overlap, validate_server_args, spec_class)`
—— `spec_info.py:67-91`，转发到 `spec_registry.register_algorithm`。
插件以 `class CustomSpecAlgo`（`spec_registry.py:26`）存在，duck-type 同一套接口。

`spec_registry.py:38-41` 的注释指出 **spec V1 worker 路径已移除**，
`supports_overlap=False` 的插件会走 V2 schema 且关闭 overlap。

> **选型含义**：要接一个自研 drafter，SGLang 有正式插件位（写个类+ 一行装饰器）；
> vLLM 需要走 `custom_class`，但要自己在 `gpu_model_runner.py` 的 if-elif 链里加分支。

---

## 3. Attention backend：44 vs 24

### 3.1 vLLM：Enum，44 个成员

`class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta)` ——
`vllm/v1/attention/backends/registry.py:34-200`。自定义 metaclass（`:18-31`）
把 `KeyError` 转成含全部合法取值的友好报错（`:28-31`）。

按族分组（行号在同一文件）：

| 族 | 成员与行号 |
|:---|:---|
| FlashAttention | `FLASH_ATTN`(L44)、`FLASH_ATTN_DIFFKV`(L45)、`FLASH_ATTN_MLA`(L121)、`FLASH_ATTN_MLA_SPARSE`(L122) |
| Triton | `TRITON_ATTN`(L48)、`TRITON_ATTN_DIFFKV`(L49)、`TRITON_FLASH_ATTN`(L52)、`TRITON_MLA`(L90)、`TRITON_MSA`(L132) |
| FlashInfer | `FLASHINFER`(L68)、`TRITON_FLASHINFER`(L69)、`FLASHINFER_MLA`(L72)、+SPARSE 系列(L78/82/86) |
| FlashMLA | `FLASHMLA`(L92)、`FLASHMLA_SPARSE`(L93)、+DSV4/DSV41 变体(L97/100/104/110/113/117) |
| ROCm / AITER | `ROCM_ATTN`(L55)、`ROCM_AITER_MLA`(L56)、`ROCM_AITER_TRITON_MLA`(L57)、`ROCM_AITER_FA`(L60)、`ROCM_AITER_MLA_SPARSE`(L63)、`ROCM_FLASHMLA_SPARSE_DSV4`(L104)、`ROCM_AITER_UNIFIED_ATTN`(L144) |
| CPU / XPU | `XPU_MLA_SPARSE`(L66)、`TORCH_SDPA`(L67, 仅 ViT)、`CPU_ATTN`(L148)、`CPU_MLA`(L149)、`AMX_MLA`(L150)、`HPC_ATTN`(L143) |
| CUTLASS | `CUTLASS_MLA`(L91)、`CUTLASS_MSA`(L128) |
| 特殊 | `NO_ATTENTION`(L136)、`FLEX_ATTENTION`(L137)、`CUSTOM`(L154)、`B12X`(L120)、`MINIMAX_M3_SPARSE`(L125)、`TURBOQUANT`(L151)、**`TOKENSPEED_MLA`(L75)** |

**`xformers` 未找到**（`vllm/v1/attention/backends/` 下无实现；
`flex_attention.py` 是 PyTorch `torch.nn.attention.flex_attention`）。

### 3.2 SGLang：装饰器注册，24 个

`ATTENTION_BACKENDS = {}`（`attention_registry.py:40`）+
`register_attention_backend(name)`（`:43-48`），逐个 `@register_attention_backend("...")`。

| 名称 | 行号 | | 名称 | 行号 |
|:---|:---|:---|:---|:---|
| `flashinfer` | L51 | | `fa3` | L227 |
| `trtllm_mla` | L78 | | `fa4` | L253 |
| **`tokenspeed_mla`** | **L96** | | `ascend` | L130 |
| `cutedsl_mla` | L107 | | `intel_xpu` | L550 |
| `aiter` | L116 | | `hpc_ops` | L271 |
| `wave` | L123 | | `intel_amx` | L288 |
| `triton` | L195 | | `minicpm_flashattn` | L326 |
| `torch_native` | L206 | | `minicpm_flashinfer` | L333 |
| `flex_attention` | L213 | | `dsv4` | L166 |
| `flashmla` | L220 | | `dsa` | L139 |
| `dsa` | L139 | | `qsa` | L146 |
| `nsa` | L155 | | `trtllm_mha` | L262 |

SGLang 独有值得注意的：`dsv4` / `dsa` / `qsa` / `nsa`（DeepSeek 稀疏注意力系列）、
`cutedsl_mla`、`fa3` / `fa4`、`wave`、`ascend`、`minicpm_*`。

vLLM 有而 SGLang 注册名里没有的：`FLASH_ATTN_DIFFKV`、`TRITON_ATTN_DIFFKV`、
`ROCM_AITER_UNIFIED_ATTN`、`CUTLASS_MSA`、`TRITON_MSA`、`MINIMAX_M3_SPARSE`、
`TURBOQUANT`、`B12X`。

### 3.3 ⚠️ 纠正：TokenSpeed MLA 两边都有

不是 SGLang 独有。vLLM 是 `TOKENSPEED_MLA`（`registry.py:75`），
指向 `vllm/v1/attention/backends/mla/tokenspeed_mla.py`；
SGLang 是 `tokenspeed_mla`（`attention_registry.py:96`），
指向 `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`。

但有一处**可用性差异**：SGLang 版仅限 MLA 模型，否则抛错
`"tokenspeed_mla backend can only be used with MLA models."`（`:99`）；
vLLM 的枚举无此守卫。

---

## 4. 结构化输出

### 4.1 后端与分派

**vLLM**：4 后端，分派链在 `vllm/v1/structured_output/__init__.py:134-160` ——
`xgrammar`(:134) → `guidance`(:140) → `outlines`(:146，懒加载) → `lm_format_enforcer`(:155，懒加载)。

约束类型枚举 `StructuredOutputOptions` 在 `backend_types.py:19-25`：
`JSON`、`JSON_OBJECT`、`REGEX`、`GRAMMAR`、`CHOICE`、`STRUCTURAL_TAG`。
抽象基类 `StructuredOutputGrammar` 在 `:31`，`StructuredOutputBackend` 在 `:97`。

**SGLang**：`GRAMMAR_BACKEND_CHOICES = ["xgrammar","outlines","llguidance","none"]`
—— `python/sglang/srt/arg_groups/choices.py:129`。

> 注意两个差异：**SGLang 有 `none` 选项**（vLLM 无）；
> **SGLang 用 `llguidance` 命名**，vLLM 用 `guidance`。

分派在 `base_grammar_backend.py`：`outlines`(:366-367) → `xgrammar`(:373-374) → `llguidance`(:405-406)。

SGLang 独有 `reasoner_grammar_backend.py`（思维链模型的结构化输出适配）与
`outlines_jump_forward.py`；另有 `json_schema_validation.py` 提供 **per-backend** schema 校验
（`validate_xgrammar_json_schema()` `:167`、`validate_outlines_json_schema()` `:203`）。

目录也不同：SGLang 是 `python/sglang/srt/constrained/`（不是 `structured_outputs/`）。

### 4.2 ⚠️ 同名参数的支持后端集合不一致

这是迁移时最容易踩的坑：

| 参数 | vLLM 支持的后端 | SGLang 支持的后端 |
|:---|:---|:---|
| `disable_any_whitespace` | `xgrammar` + `guidance` | `outlines` + `llguidance` |
| `disable_additional_properties` | **仅** `guidance` | `xgrammar` + `llguidance` |
| `max_whitespace` | 未提供 | **仅** `xgrammar` |

vLLM 侧见 `vllm/config/structured_outputs.py:63-71`；
SGLang 侧见 `python/sglang/srt/arg_groups/fields/serving.py:280-288`
（该处注释直接点名后端）。

**Lark grammar：两边都未找到支持。**

---

## 5. LoRA

| | vLLM | SGLang |
|:---|:---|:---|
| 并发 adapter 数 | `max_loras` 默认 **1**（`config/lora.py:37`） | `max_loras_per_batch` 默认 **8**（`fields/lora.py:66`） |
| rank 上限 | `max_lora_rank` 固定 **16**（`config/lora.py:35`） | `max_lora_rank` 默认 **None，从 `--lora-paths` 自动推断**（`fields/lora.py:42-45`） |
| CPU 侧上限 | `max_cpu_loras`（`config/lora.py:122-126`） | `max_loaded_loras`（`fields/lora.py:62-65`） |
| 淘汰策略 | 未见同名参数 | `lora_eviction_policy`：`lru`（默认）/ `fifo`（`fields/lora.py:70-76`） |
| kernel 后端可换 | 否（Triton 固定） | **是**：`lora_backend` = `triton`/`csgmv`/`ascend`/`torch_native`，默认 `csgmv`（`fields/lora.py:77-83`） |
| chunk 大小 | — | `max_lora_chunk_size` = 16/32/64/128，默认 16（`fields/lora.py:84-90`） |

SGLang 的 LoRA 后端目录 `python/sglang/srt/lora/backend/` 有 7 个实现
（`triton` / `chunked` / `ascend` / `torch` / `uno_cublas` + `base_backend` + `lora_registry`），
另有 `lmhead_mixing.py`、`lora_overlap_loader.py`、`lora_drainer.py`、
`lora_moe_runner_marlin.py` 等。

vLLM 侧 `vllm/lora/ops/triton_ops/` 有 8 个 kernel 文件（expand / shrink / fused_moe × 4 + metadata + fp8 utils）。

> **选型含义**：多租户 LoRA 服务（一个基座+ 几十个业务 adapter）是 SGLang 的明显优势项 ——
> 默认就支持 8 个 adapter 同 batch，vLLM 默认只支持 1 个，要开还得调 `max_loras`。

---

## 6. PD 分离与KV 传输

### 6.1 vLLM：16 个 connector，字符串插件式注册

机制是 `KVConnectorFactory` + `register_connector(name, module_path, class_name)`
**字符串懒加载** —— `vllm/distributed/kv_transfer/kv_connector/factory.py:27-40`
（`_registry` `:28`、`register_connector` `:31`、重复注册报错 `:34`、懒import `:36-40`）。

16 个注册项（行号 = `register_connector(` 调用行，`factory.py`）：

| 行 | 名称 | | 行 | 名称 |
|:---|:---|:---|:---|:---|
| L153 | `ExampleConnector` | | L201 | `HiSparseConnector` |
| L159 | `ExampleHiddenStatesConnector` | | **L207** | **`MoRIIOConnector`** |
| **L165** | **`LMCacheConnectorV1`** | | L213 | `OffloadingConnector` |
| **L171** | **`LMCacheMPConnector`** | | L219 | `DecodeBenchConnector` |
| **L177** | **`NixlConnector`** | | **L225** | **`MooncakeConnector`** |
| **L183** | **`NixlPullConnector`** | | **L230** | **`MooncakeStoreConnector`** |
| **L189** | **`NixlPushConnector`** | | L235 | `FlexKVConnectorV1` |
| L195 | `MultiConnector`（HMA，`:135-146`） | | L240 | `SimpleCPUOffloadConnector` |
| | | | **L245** | **`HF3FSKVConnector`** |

NIXL 类定义：`NixlBaseConnector` `v1/nixl/connector.py:77`、
`NixlPullConnector` `:339`、`NixlPushConnector` `:367`，
`supports_divergent_local_hybrid_hits()` `:81`；子模块含 `pull_scheduler.py` / `pull_worker.py` /
`push_scheduler.py` / `push_worker.py` / `tp_mapping.py`（**TP 感知**）等 9+ 文件。

**vLLM 特有能力：HMA（Hybrid Memory Attention）是一等公民**——
`SupportsHMA` mixin、`MultiConnector.all_children_support_hma()`（`factory.py:146`）、
`disable_hybrid_kv_cache_manager` 配置检查（`:55-58`）、connector 不支持时报错（`:58`）。

> **未找到名为 `shared_storage` 的 connector**（常被提及）。
> 最接近的是 `SimpleCPUOffloadConnector`(L240) 与 HF3FS 存储（`vllm/v1/hf3fs/`）。

### 6.2 SGLang：5 个传输后端 + encoder 分离

`class TransferBackend(Enum)` —— `python/sglang/srt/disaggregation/utils.py:565-570`，
共 5 个：`MOONCAKE`(L566)、`MORI`(L567)、`NIXL`(L568)、`ASCEND`(L569)、`FAKE`(L570)。

实现目录 `python/sglang/srt/disaggregation/`（40 文件）：

| 类别 | 文件 |
|:---|:---|
| 后端 conn | `nixl/conn.py`、`mooncake/conn.py`、`mori/conn.py`、`ascend/conn.py`(+`transfer_engine.py`)、`fake/conn.py`、`common/conn.py`、`base/conn.py` |
| 调度 | `prefill.py`、`decode.py`、`decode_schedule_batch_mixin.py`、`decode_hicache_mixin.py`、**`role_switch.py`** |
| 传输 | `common/staging_buffer.py`、`staging_handler.py`、`dcp_pack.py`、`checksum.py`、`kv_events.py` |
| **encoder（多模态编码器）分离** | `encoder/`（`grpc_server.py`、`http_server.py`、`preprocessor.py`、`receiver.py`、`runtime.py`、`server.py`） |

> **vLLM 侧未见对应的 encoder 分离目录** —— 这在多模态 PD 分离场景下是个真实差距。

### 6.3 ⚠️ 极易混淆的一点

SGLang 的 `lmcache` / `hf3fs` / `flexkv` / `aibrix_kvcache` / `eic` / `file`
**不在 `disaggregation/`**，而在 `python/sglang/srt/mem_cache/storage/`（HiCache 分层存储），
由 `backend_factory.py` 选择。**这些是存储后端，不是 PD 传输后端。**

HiCache 抽象基类 `HiCacheStorage(ABC)` 在 `python/sglang/srt/mem_cache/hicache_storage.py:170`，
后端工厂 `StorageBackendFactory` 在 `python/sglang/srt/mem_cache/storage/backend_factory.py:16`
（`register_backend()` `:44` 懒加载、`create_backend()` `:66`）。

**12 个已注册存储后端**（注册调用在 `backend_factory.py:202` 起）：

| 名称 | 实现类 | 行号 |
|:---|:---|:---|
| `file` | `HiCacheFile` | `:202-204` |
| `sim` | `SimHiCacheStorage` | `:206-208` |
| `nixl` | `HiCacheNixl` | `:210-214` |
| `mooncake` | `MooncakeStore` | `:216-220` |
| `npu_memcache` | `NpuMemcacheStore` | `:222-226` |
| `hf3fs` | `HiCacheHF3FS` | `:228-232` |
| `aibrix` | `AibrixKVCacheStorage` | `:234-238` |
| `eic` | `EICStorage` | `:240-244` |
| `simm` | `HiCacheSiMM` | `:246-250` |
| `mori` | `UMBPStore` | `:252-256` |
| `shm` | `HiCacheShm` | `:258-262` |
| `tensorcast` | — | `:264` |

GPU / CPU 两级不是「后端」而是内存池：`hicache_ratio`（`fields/memory.py:107-110`，默认 `None`，
cache 模式解析为 2.0、buffer_only 模式 1.2）、`hicache_host_memory_mode`（`:100-106`，默认 `cache`）。

### 6.4 拓扑要求

**SGLang 有明确文档**（`docs/docs/advanced_features/pd_disaggregation.mdx`）：

- **同机NVLink**：`SGLANG_MOONCAKE_CUSTOM_MEM_POOL` 取 `NVLINK`（推荐 NVL72）或
  `INTRA_NODE_NVLINK`（推荐 A100/H20/H100），见 `:194-208`
- ⚠️ **重要限制**：NVLink 只作用于 KV cache 传输，**auxiliary data 仍走 TCP**
  （`:195`、`:202` 明确写"auxiliary data transfer will still use TCP as a temporary workaround"）
- **跨机 RDMA**：文档 L91 有「IB Device Configuration」章节，支持 AWS EFA
  （`/opt/amazon/efa/lib`，`:71`、`:78`；EFA wheel 仅为 x86_64 且不含 Mooncake EP/PG 与 NVLink/MNNVL 组件）
- `SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT = EnvInt(60)`（`environ.py:710`），
  注释 "A wedged RDMA stack fails startup here instead of at the scheduler watchdog"

**vLLM 侧拓扑要求未找到**：`docs/` 下无任何 disaggregation 文档
（`git ls-tree` grep "disaggregat" 命中 0）；`nixl/connector.py` grep
`RDMA|rdma|nvlink|IB` 仅命中类定义行。
可间接确证的是有 `mooncake/rdma_utils.py`（文件名含 rdma）。

---

## 7. 量化格式

### 7.1 机制差异（这才是关键）

**vLLM**：`QuantizationMethods` 是 `typing.Literal[...]`，**30 个字符串**，
定义 `vllm/model_executor/layers/quantization/__init__.py:16-48`；
`QUANTIZATION_METHODS` 在 `:50` 生成。分派表 `method_to_config` 在 `:156-184`。
量化方法以**顶层 `.py` 模块**形式存在，子目录仅 6 个且非一一对应方法。

**SGLang**：普通 dict `BASE_QUANTIZATION_METHODS`（`:62-90`，**25 个键**）
+ 按平台条件 `.update()` 追加（CPU/CUDA/ROCm/XPU/NPU/MPS 各不相同），
最终 `QUANTIZATION_METHODS` 在 `:157`。采用**「每方法一子包」**结构
（`awq/`、`gptq/`、`compressed_tensors/`、`quark/`、`modelslim/`…），
每个子包内含 `schemes/` 分发不同 kernel 后端（cpu / linear / marlin / moe / xpu）。

> **选型含义**：同一个量化方法在两边能选到的 kernel 后端可能不同 ——
> 这比「支不支持」这个布尔值更影响实际吞吐。

### 7.2 逐格式对照

| 格式 | vLLM | SGLang |
|:---|:---|:---|
| fp8 | ✅ `__init__.py:160` | ✅ `:63`；另有 `mxfp8`(`:64`)、`w8a8_fp8`(`:72`) |
| **nvfp4** | ✅ `:164`（`modelopt_fp4`）、`:45`（`nvfp4_per_token` shorthand） | ✅ `:69`（`nvfp4_online`）、`:81`（`petit_nvfp4`） |
| **mxfp4** | ✅ `:176`、`:177`（`gpt_oss_mxfp4`）、`:45`（`mxfp8`） | ✅ `:100`（条件注册）；**NPU 下 `:112` 语义变为 W4A4**；`:89`（`mxfp_w4a8`） |
| int8 / w8a8 | ✅ `:171`（`experts_int8`）、`:44`（shorthand）。**无 `w8a8` 键** | ✅ `:65`（`blockwise_int8`）、`:71`（`w8a8_int8`）、`:85`（`auto-round-int8`） |
| awq | ✅ `:157-159`（`awq`/`awq_marlin`/`auto_awq`） | ✅ `:73`、`:74`；CPU 下 `:124`、XPU 下 `:133` 有特化 |
| gptq | ✅ `:167-169`（`auto_gptq`/`gptq`/`gptq_marlin`） | ✅ `:77`（`gptq_marlin`）；NPU/CPU/XPU 各有特化（`:108`/`:124`/`:132`） |
| **bitsandbytes** | ❌ **不存在**（无枚举值、无分派项、无 `.py`） | ✅ `:75` |
| **gguf** | ❌ **不存在**（同上） | ✅ `:76` |
| compressed-tensors | ✅ `:170` | ✅ `:79` |
| modelopt | ✅ `:163-166` | ✅ `:66-68`、`:70` |
| quark | ✅ `:172` | ✅ `:82-83`、`:87`（含 `quark_mxfp4`、`quark_int4fp8_moe`） |
| torchao | ✅ `:174` | ❌ **未找到**（以 `humming`(`:88`)/`modelslim`(`:86`)/`mlx` 替代） |

> ⚠️ **gguf 与 bitsandbytes 在 vLLM v0.31.0 的量化注册表中完全不存在**。
> 这是实质功能差异，不是搜索遗漏（目录与枚举AST 均已完整枚举）。
> 如果你的链路依赖这两者（例如 llama.cpp 产出的 GGUF 权重），
> **vLLM 在这个版本上不是可选项**。

SGLang 独有方法：`auto-round`(`:84`)、`modelslim`(`:86`)、`humming`(`:88`)、
`petit_nvfp4`(`:81`)、`w4afp8`(`:80`)、`mxfp8`(`:64`)。

vLLM 另有 `DEPRECATED_QUANTIZATION_METHODS = ["fbgemm_fp8","fp_quant"]`（`:52-55`，仍在分派表中但标记弃用），
以及 `register_quantization_config("my_quant")`（`:61`）的自定义注册入口（`:97-100`会同步进平台列表）。

---

## 8. RL / rollout

### 8.1 ⚠️ 纠正：vLLM 也有 `return_sampling_mask`

常见的说法是「SGLang 有 `return_sampling_mask` 用于 RL rollout 复现，vLLM 没有」。
**这在 v0.31.0 上不成立** —— vLLM 有完整实现，覆盖 18 个文件。

vLLM 侧：配置字段 `return_sampling_mask: bool = False` —— `vllm/config/model.py:252`
（docstring “Whether to return the post-processing token support for each sample.”）。

关键链路：

| 环节 | 位置 |
|:---|:---|
| CLI 参数 | `vllm/engine/arg_utils.py:471`、`:967`、`:1942` |
| 与 batch-sharded gather 冲突检查 | `vllm/config/vllm.py:3205`（检查体 `:3208`） |
| 文档 | `vllm/entrypoints/llm.py:166` |
| **sampler 核心** | `vllm/v1/worker/gpu/sample/sampler.py:201-205`（`SamplingMaskTensors.from_logits(...)`） |
| 调度器侧 | `scheduler.py:406`、`:2098`、`:2189-2192`、`:2210` |
| 协议 | `vllm/entrypoints/scale_out/token_in_token_out/protocol.py` |

**但 vLLM 侧有两个约束**：

1. 开启 mask 会**主动关闭 flashinfer sampler 与 xpu sampler 快速路径**
   （`sampler.py:89`、`:93`）
2. **与 batch-sharded gather 不兼容**（`config/vllm.py:3205-3208`）

### 8.2 真正的差异

| | vLLM | SGLang |
|:---|:---|:---|
| `return_sampling_mask` | ✅ | ✅ |
| **token 数上限参数** | ❌ **全树grep 0 命中** | ✅ `--sampling-mask-max-tokens` 默认 **4096**（`fields/exec_.py:101-107`） |
| PD 分离下支持 | 未定位到 | ✅ `disaggregation/prefill.py:853,922`、`decode.py:2557`、`utils.py:520` |
| 依赖约束 | — | ⚠️ **必须同时开 `return_meta_info`**（`serving_chat.py:1030-1031`） |
| 跨节点一致性要求 | — | PD 两侧必须同值（`arg_groups/validation_hook.py:511`） |

SGLang 的 mask capture 性能保护：`:sampling_mask_max_tokens` 读在 `layers/sampler.py:128`，
且 `sampler.py:335` 注释 "Capture work is performed only when return_sampling_mask is enabled."——
**不开就不付这个开销**。

环境开关 `SGLANG_ENABLE_DISAGG_SAMPLING_MASK = EnvBool(False)`（`environ.py:701`，
注释 L700 “Enable on both P and D with the same --sampling-mask-max-tokens value.”）；
旧变量 `SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS` 已废弃、仅用于拒绝启动（`:702-703`）。

---

## 9. 可观测性：38 vs 156

### 9.1 ⚠️ 两个路径都不存在

- `vllm/engine/metrics.py` **不存在** → 已迁移到 **`vllm/v1/metrics/`**（7 文件：
  `loggers.py`（1309 行，指标定义主体）、`prometheus.py`(82)、`stats.py`、`buckets.py`、
  `perf.py`、`reader.py`、`ray_wrappers.py`）
- `python/sglang/srt/metrics/` **不存在** → 指标以 `name="sglang:*"` **内联散布**在
  `python/sglang/srt/` 各处

### 9.2 数量与差异

vLLM 核心引擎 `name="vllm:*"` 去重得 **38 个**（仅 `v1/metrics/loggers.py`；
`kv_connector/v1/metrics.py`、`orca_metrics.py`、realtime 等处另有，
故 vLLM 总数 ≥ 38）。`prometheus.py` 只做 multiprocess 管理
（`setup_multiprocess_prometheus()` `:17`、`get_prometheus_registry()` `:39`、
`unregister_vllm_metrics()` `:56`、`shutdown_prometheus()` `:72`）。

SGLang 全树去重得 **156 个**。且有 vLLM 缺失的**整类**指标：

| 类别 | 代表指标 |
|:---|:---|
| **grammar（10个）** | `sglang:grammar_compilation_time_seconds`、`grammar_ebnf_size`、`grammar_schema_count`、`grammar_tree_traversal_time_avg` / `_max`、`num_grammar_aborted_total`、`num_grammar_cache_hit_total`、`num_grammar_queue_reqs`、`num_grammar_timeout_total`、`num_grammar_total` |
| **hicache / 存储（~20 个）** | `hicache_backup_*`(4)、`hicache_host_*`(2)、`hicache_prefetch_*`、`storage_prefetch_*`(3)、`load_back_*`(3)、`backup_*`(3)、`prefetch_bandwidth`、`prefetch_pgs`、`hicache_dropped_tokens_total` |
| **KV 传输 / PD（9 个）** | `kv_transfer_alloc_ms`、`kv_transfer_bootstrap_ms`、`kv_transfer_latency_ms`、`kv_transfer_speed_gb_s`、`kv_transfer_total_mb`、`num_bootstrap_failed_reqs_total`、`num_transfer_failed_reqs_total`、`num_prefill_retries_total`、`per_stage_req_latency_seconds` |
| **投机解码（7 个）** | `spec_accept_length`、`spec_accept_rate`、`spec_block_accept_length`、`spec_cap_length`、`spec_num_draft_tokens`、`spec_num_steps`、`spec_verify_calls_total` |
| **PD 队列深度（5 个）** | `num_decode_transfer_queue_reqs`、`num_decode_prealloc_queue_reqs`、`num_prefill_bootstrap_queue_reqs`、`num_prefill_inflight_queue_reqs`、`num_decode_host_receive_queue_reqs` |
| LoRA（3 个） | `lora_pool_slots_total`、`lora_pool_slots_used`、`lora_pool_utilization` |
| **encoder（14 个）** | `encoder_*` |
| 路由（4 个） | `num_unique_running_routing_keys`、`routing_key_all_req_count`、`routing_key_running_req_count`、`routing_keys_active` |
| 其它 | `eplb_balancedness`、`fwd_occupancy`、`estimated_flops_per_gpu_total`、`estimated_read_bytes_per_gpu_total`、`estimated_write_bytes_per_gpu_total`、`prefill_delayer_*`(3)、`swa_*`(4)、`mamba_*`(4) |

> **选型含义**：如果你的可观测性方案依赖「从引擎指标直接算 SLO」，
> SGLang 能直接给的指标多得多 —— 尤其是 **KV 传输分段耗时**和 **grammar 编译耗时**，
> 这两个在 vLLM 侧没有对应指标，做 PD 分离的容量规划会很被动。

### 9.3 开关

**SGLang**：`--enable-metrics` 默认 **False** —— `python/sglang/srt/arg_groups/fields/observability.py:70`；
`--enable-metrics-for-all-schedulers`（`:82`，help `:84` 说明开启后所有 TP rank 而非仅 TP 0
单独记录请求指标，`dp_attention` 下尤其有用）。

**vLLM**：未找到同名开关，Prometheus 导出为常开+ `prometheus.py` multiprocess 管理。

### 9.4 Rust router 的指标

SGLang router（`sgl-model-gateway/`）有独立 metrics 实现：
`src/observability/metrics.rs`（用 `metrics` + `metrics_exporter_prometheus`，
`counter!`/`gauge!`/`histogram!` 分别出现 45 / 31 / 23 处，
含 `STRING_INTERNER` 标签 interning 以控制基数，L28），
另有 `src/core/metrics_aggregator.rs`。

⚠️ **具体 router 指标名称未取到**（字符串字面量 grep 命中 0）。

---

## 10. API 兼容面

### 10.1 组织方式

**vLLM 已重构为按域拆分的 `api_router.py`**，共 **30 个** `api_router.py`，各以 `attach_router(app)` 挂载。
**SGLang 是单文件集中注册** —— `python/sglang/srt/entrypoints/http_server.py`（2936 行），
共 **55 个**路由装饰器。

### 10.2 OpenAI 兼容端点

| 端点 | vLLM | SGLang |
|:---|:---|:---|
| `/v1/chat/completions` | `openai/chat_completion/api_router.py:41` | `http_server.py:1777` |
| `/v1/completions` | `openai/completion/api_router.py:35` | `http_server.py:1769` |
| `/v1/embeddings` | `pooling/embed/api_router.py:29` | `http_server.py:1787-1791` |
| `/v1/models` | `openai/models/api_router.py:20` | `http_server.py:1898` |
| `/v1/models/{path}` | — | `http_server.py:1930` |
| `/v1/responses` | `openai/responses/api_router.py:48` | `http_server.py:1971` |
| `/v1/responses/{id}` | `openai/responses/api_router.py:80` | `http_server.py:1990` |
| `/v1/responses/{id}/cancel` | `openai/responses/api_router.py:110` | `http_server.py:1998` |
| `/v1/score` | `pooling/scoring/api_router.py:53` | `http_server.py:1955` |
| `/v1/rerank` | `pooling/scoring/api_router.py:85` | 未找到 |
| `/v1/classify` | `pooling/classify/api_router.py` | `http_server.py:1799-1803` |
| `/v1/pooling` | `pooling/pooling/api_router.py` | 未找到 |
| `/v1/audio/transcriptions` | `speech_to_text/transcription/api_router.py:32` | `http_server.py:1847` |
| `/v1/audio/translations` | `speech_to_text/translation/api_router.py:32` | 未找到 |
| `/v1/tokenize` + `/tokenize` | `serve/tokenize/` | `http_server.py:1811-1821`（双路由） |
| `/v1/detokenize` + `/detokenize` | — | `http_server.py:1829-1839`（双路由） |

### 10.3 SGLang 独有的扩展

| 端点 | 行号 | 用途 |
|:---|:---|:---|
| **`/v1/decisions`** | `:1963` | chat 模型当分类器/打分器，**不做生成** |
| `/v1/systemone` | `:2086` | System One 兼容 decisions API |
| `/v1/messages` | `:2065` | Anthropic 风格 |
| `/v1/messages/count_tokens` | `:2075` | |
| `/api/chat`、`/api/generate`、`/api/tags`、`/api/show` | `:2036` / `:2042` / `:2050` / `:2056` | **Ollama 兼容**（路径可用环境变量覆盖） |
| `/vertex_generate` | `:2112` | Vertex AI（env `AIP_PREDICT_ROUTE`） |
| `/invocations` | `:2101` | |
| `/hicache/storage-backend` | `:1146` | |
| `/add_external_corpus` 等 | `:1005` / `:1030` / `:1048` | 外部语料管理 |

**vLLM 也有非 OpenAI 面**：`anthropic/`（`/v1/messages`）、`cohere/`、
`generate/`、`scale_out/`（`derender`/`render`/`token_in_token_out`）、
`serve/`（11 个子域，含 `fault_tolerance`、`elastic_ep`、`lora`、`sagemaker`、`rlhf` 等）。

### 10.4 `/v1/decisions` 的实现细节（v0.5.21 新增）

路由 `http_server.py:1963`，handler `v1_decisions_request()` `:1964`，
委托 `app.state.openai_serving_decisions`。

实现类 `OpenAIServingDecisions` —— `python/sglang/srt/entrypoints/openai/serving_decisions.py:64`；
构造函数 `:70`（**复用 chat serving**）；服务端能力校验 `_validate_server()` `:134`；
`_apply_chat_template()` `:287`；`render_text()` `:369`。

机制（官方文档 `docs/docs/supported-models/decision_models.mdx:6`）：
「turns a chat model into a decision model… each answer comes back with the
probability of every option, read from the model's **next-token scores at the answer position**」
—— 即复用 scoring 路径，不解析生成输出。每响应携带 `prompt_format_version`（服务端拥有 prompt 措辞并版本化，`:137`）。

> ⚠️ **文档与 tag 的错位**：`decision_models.mdx:6` 与 `:9` 都写
> “Until a release contains it, install a nightly build”。即在该 tag 上文档仍标为 nightly。
> **写作时不应断言「v0.5.21 已正式发布该端点」**。另外 `:9` 明确它是
> SGLang 对 `/v1` 的扩展，**不属 OpenAI API**。

---

## 11. 硬件平台

| | vLLM | SGLang |
|:---|:---|:---|
| 平台文件 | `vllm/platforms/`：`__init__` `interface` `cuda` `rocm` `cpu` `xpu` **`tpu`** `zen_cpu` | `python/sglang/srt/platforms/`：`__init__` `interface` `device_mixin` `cuda` `rocm` `cpu` **`npu`** `xpu` |
| 检测逻辑 | `__init__.py` TPU`:34-53` / CUDA`:79-105` / ROCm`:110-138`（含 WSL 回退`:136`）/ XPU`:145-162` / CPU`:175-187` | — |
| **TPU** |✅ `tpu.py`（**全文仅 20 行**） | ❌ 主仓**无** TPU 代码 |
| **昇腾 NPU** | ❌ `platforms/` 下无 npu/ascend | ✅ `npu.py` + `disaggregation/ascend/` + `lora/backend/ascend_backend.py` + `quantization/npu_mxfp4*.py` |

**两边恰好互补**：vLLM 有 TPU 无 NPU，SGLang 有 NPU 无 TPU。

> ⚠️ vLLM 无 NPU platform 文件这一结论的搜索范围是 `vllm/platforms/` 目录 ls-tree，
> 未穷尽全库。

### 11.1 TPU 支持方式：形态完全不同

**vLLM —— 委托给 PyPI 包**。`vllm/platforms/tpu.py` 全文 20 行，核心是 try/except：

```python
from tpu_inference.platforms import TpuPlatform as TpuInferencePlatform
TpuPlatform = TpuInferencePlatform
USE_TPU_INFERENCE = True
```

（`:10-15`；ImportError 时 log error 后 `pass`，`:17-19`）
即 **vLLM 仓库内无 XLA/TPU kernel 实现**。另支持 Pathways TPU proxy
（`__init__.py:36-50`，`VLLM_TPU_USING_PATHWAYS` `:37`）。

**SGLang —— 独立仓库 `sgl-project/sglang-jax`**。
**不是同仓库子目录**（顶层目录实测无 `jax`；全仓 grep `jax|tpu` 只命中 docs）。
官方文档 `docs/docs/hardware-platforms/tpu.mdx:9` 原文：
「SGLang TPU support is implemented via the **SGLang-JAX backend, a dedicated
JAX-based inference engine maintained as a separate repository**」。

安装：`pip install sglang-jax`（`:234`）。
文档已列支持的模型（`:184-224` 链接到 sglang-jax cookbook）：
Qwen3、Qwen3-MoE、Qwen2.5-VL、Llama 3.1、Llama 3.3 70B、DeepSeek V3/R1、
Kimi-Linear、Grok-2、Gemma 2、MiMo-V2-Flash / V2.5-Pro、Ling-2.6。
⚠️ **TPU Docker 支持仍在开发中**（`:248`）。

> **选型含义**：TPU 上 vLLM 与 SGLang 是**两个不同项目的竞争关系**，
> 不是「同一个项目的两条后端路径」。SGLang-JAX 的架构数/量化/后端细节不在本文取证范围。

---

## 12. 默认开关对照（升级/迁移必查）

| 特性 | vLLM `v0.31.0` | SGLang `v0.5.21` |
|:---|:---|:---|
| prefix caching | **默认开**（`config/cache.py:142`） | **默认开**（`fields/memory.py:64-70`） |
| chunked prefill | **默认开**（`DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048`，`config/scheduler.py:42`） | **默认开**（`chunked_prefill_size` 默认 `None`；`-1` 才禁，`fields/schedule.py:52-55`） |
| CPU/GPU overlap | **默认关**（`async_scheduling=None`，`config/scheduler.py:209`） | **默认开**（`disable_overlap_schedule=False`，`fields/schedule.py:193`） |
| 长 prefill 公平性阈值 | **默认关**（`long_prefill_token_threshold=0`，`config/scheduler.py:80`；adaptive 也默认 `False` `:87`） | 不适用（用 9 种 schedule-policy 替代） |
| prefix-aware 调度 | 无 | **默认关**（`schedule_policy="fcfs"`，`fields/schedule.py:98`） |
| 分层 KV / HiCache | 无内建分层 | **默认关**（`enable_hierarchical_cache=False`，`fields/memory.py:99`） |
| PD role switching | **该 commit 未找到**（无此功能） | **默认关**（`enable_pd_role_switch=False`，`fields/disagg.py:115`） |
| 投机解码 | 默认关 | 默认关（`SpeculativeAlgorithm.NONE`） |
| Prometheus 指标 | 常开 | **默认关**（`--enable-metrics`，`fields/observability.py:70`） |

### 12.1 近期 breaking change（选型时看版本的实证）

**vLLM v0.31.0 移除/收紧**：

| 项 | 变化 |
|:---|:---|
| 多模态 per-request kwargs | **默认拒绝**（需 `--trust-request-mm-kwargs`） |
| `tokenizer_mode="slow"` | 移除 |
| `--enable-mamba-fine-grained-prefix-cache` | 移除 |
| `quantization="fp8"` | 改spec |
| Quark 量化 | 改隐式在线量化 |
| AllSpark INT8 W8A16 | 移除 |
| `--enforce-eager` | 移除 |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | 移除 |

另v0.30.0 引入「Fast Start」、`HiSparse`；v0.29.0 把 **Model Runner V2 设为所有模型默认**；
v0.25.0把 Model Runner V2 设为 dense 模型默认并「deprecate PagedAttention」；
v0.26.0 引入 Rust frontend（多模态）；v0.21 周期引入 KV offloading
（Hybrid Memory Allocator、Mooncake）、RayExecutorV2 默认、Model Runner V2。

**SGLang 侧**：

- **v0.5.20**：`/v1/responses` 服务端存储**默认关闭**，需 `--enable-response-store`；
  依赖存储的请求（retrieval、`previous_response_id`、后台任务）返回 400
  （PD 部署下不支持该 flag）。同时移除 **prefill CP v1** 与 CUDA 12 产物。
- **v0.5.21**：移除 `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE`、`SWARadixCache` / `MambaRadixCache`
  （并入 `UnifiedRadixCache`）、`HiRadixCache`，以及所有废弃 2+ 版本的端点/环境变量/别名。

> **选型含义**：两边都在快速破坏兼容性。**「vLLM 支持某模型」这句话必须带版本号**，
> 否则半年后可能就变了。

---

## 13. 本文的取证边界（明确「未找到」的项）

| 项| 状态 |
|:---|:---|
| vLLM `gguf` / `bitsandbytes` 量化 | **确认不存在**（已完整枚举目录与 AST） |
| SGLang `torchao` | **确认不存在**（全量化目录无；以 humming/modelslim/mlx 替代） |
| vLLM NPU platform | **未找到**（搜索范围 `vllm/platforms/` 目录 ls-tree） |
| vLLM `shared_storage` connector | **未找到**（16 个 connector 全枚举） |
| vLLM `sampling_mask_max_tokens` | **未找到**（`git grep` 全树 `vllm/`，0 命中） |
| Lark grammar（两边） | **未找到** |
| vLLM PD 拓扑文档 | **未找到**（`docs/` 全树文件名 grep 命中 0） |
| `e4m3` 精确行号（两边） | **存在但未定位到路径**（仅在 `__init__.py` 层未见该字面量，未逐行核查 `fp8.py` / `fp8_utils.py` 内部 dtype 分支） |
| SGLang router 指标名称 | **存在但未定位到路径**（已确证 `metrics.rs` 用 `metrics_exporter_prometheus`，字面量 grep 命中 0） |
| SGLang `page_size` 解析成具体数字的代码 | **未追到**（字段默认 `None`，`:285` 只是透传） |
| `sgl-project/sglang-jax` 内部实现 | **未取证**（不在两仓范围内） |
| vLLM recovery instruction 白名单 | **未找到**（`instruction` 是裸 `str`，当前只实现 `retry`） |

### 三处需要纠正的常见预设

1. **vLLM 有 `return_sampling_mask`**（18 文件完整链路），
   真正差异是 vLLM 无 token 上限参数、且开启后会关闭 flashinfer/xpu sampler 快速路径、
   与 batch-sharded gather 不兼容。
2. **TokenSpeed MLA 两边都有**（vLLM `:75`、SGLang `:96`）。
3. **两个 metrics 路径都不存在**（vLLM 迁到 `v1/metrics/`；SGLang 是内联散布），
   且 **SGLang 主仓完全没有 TPU/JAX 代码**。

## 参考

- vLLM `v0.31.0` release notes：<https://github.com/vllm-project/vllm/releases/tag/v0.31.0>
- SGLang `v0.5.21` release notes：<https://github.com/sgl-project/sglang/releases/tag/v0.5.21>
- SGLang structured output 决策模型文档：<https://docs.sglang.io/supported-models/decision_models/>
- SGLang PD 分离文档：<https://docs.sglang.io/advanced_features/pd_disaggregation/>
- SGLang TPU 文档：<https://docs.sglang.io/hardware_platforms/tpu/>
- vLLM 投机解码配置文档：<https://docs.vllm.ai/en/latest/features/spec_decode/>
- SGLang-JAX（独立仓库）：<https://github.com/sgl-project/sglang-jax>