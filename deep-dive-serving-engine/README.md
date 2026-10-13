# SGLang vs vLLM 深度对比

> 源码级取证基准：**vLLM `v0.31.0`**（commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`，
> 2026-10-05）／ **SGLang `v0.5.21`**（commit `e00930c5489053f26d86b179cee0d087f846acbb`，
> 2026-10-02）。
> 全部分析基于这两个 tag 的源码，所有行号引用可点开 permalink 核验。

## 这篇 deep dive 的一句话结论

**默认选 vLLM**，因为覆盖更广、故障隔离成体系、大模型默认配置更可靠 ——
**不是因为它更快**（没有证据支持）。

**偏离默认的四个明确理由**：前缀复用率高、自建 RL rollout、昇腾 NPU / TPU、
多 LoRA 租户或依赖 gguf/bitsandbytes 量化。

---

## 一个必须先说的前提：性能对比的现状

在动手调研时发现一件事，它改变了整篇文章的写法：

> **网上流传最广的「SGLang 领先 vLLM 29%」是 2024 年的数字**
> （测试对象是 SGLang v0.2.3 vs vLLM 0.11.0，版本差约两年），
> 而且它是**同一份基准的三次转载**，并列引用会制造「多源印证」的假象。
>
> **在锁定的 v0.31.0 / v0.5.21 上，不存在任何同配置的第三方 head-to-head。**
> 两个项目的官方 release notes 也都没给出互相的对比数字。

所以本系列**不给「谁更快」的结论**，给的是：哪些机制决定差距、怎么自己测。
详见 [08-benchmark-crosscheck.md](08-benchmark-crosscheck.md)。

---

## 文档地图

| 文档 | 内容 | 什么时候读 |
|:---|:---|:---|
| **[01 架构分歧的根](01-architecture-divergence.md)** | 三条独立设计选择：KV 缓存组织、调度器重叠、进程/语言边界 | **先读这个**。后面所有差异都能从它推出 |
| **[02 功能与支持矩阵](02-feature-matrix.md)** | 模型/量化/后端/LoRA/PD 分离/结构化输出/API/硬件/默认开关全表 | 做选型时查表 |
| **[08 Benchmark 交叉验证](08-benchmark-crosscheck.md)** | 全部流传数字的可信度分级 + 三处广泛流传的错误 + 自测方法 | 任何引用性能数字之前 |
| **[10 选型决策手册](10-selection-playbook.md)** | 决策树 + 两侧明确理由 + 灰色地带 + 迁移 checklist | 要做决策时 |

### 阅读路径

- **要选型** → 10 → 08（前缀命中率那节）→ 02 查表
- **要理解架构** → 01 → 02
- **要引用性能数字** → 08（必读，否则会引错）

---

## 速查表

### 三条架构路线

| | vLLM | SGLang |
|:---|:---|:---|
| KV 缓存 | 固定 block + 链式哈希（16 token 对齐） | radix tree over token（任意位置） |
| CPU/GPU 重叠 | **默认关**（`async_scheduling`） | **默认开**（overlap scheduler） |
| 进程/语言 | Python +每 DP rank 一个 EngineCore 进程 | Python + Tokio + 内嵌 Rust 网关 |

### 数量对照（⚠️ 模型数口径不同，不可相减）

| | vLLM | SGLang |
|:---|---:|---:|
| 模型 | 387 唯一架构名 | 282 EntryClass |
| Attention backend | **44** | 24 |
| KV 传输 connector | **16** | 5 |
| Prometheus 指标 | 38+ | **156** |
| 默认多 LoRA 并发 | 1 | **8** |
| 缓存淘汰策略 | 隐式 LRU | **8 种可插拔** |

### 硬件恰好互补

| | vLLM | SGLang |
|:---|:---|:---|
| 昇腾 NPU | ❌ | ✅ |
| TPU | ✅（委托 `tpu_inference` 包） | ⚠️ 主仓无代码，在独立仓库 `sgl-project/sglang-jax` |

---

## 本文推翻的六个流传说法

调研过程中逐条核过源码，以下六条都与事实不符：

| # | 流传说法 | 事实 |
|:---|:---|:---|
| 1 | SGLang 领先 vLLM 29% | 2024 年数字（v0.2.3 vs 0.11.0），且是同一基准的三次转载 |
| 2 | IoT DT 的5450 tok/s 是 SGLang | 属Triton；原文 vLLM 5100 / SGLang 5300 / Triton 5450 |
| 3 | SGLang 默认用 longest prefix match 调度 | 默认是 `fcfs`（`fields/schedule.py:98`），`lpm` 需显式指定 |
| 4 | TokenSpeed MLA 是 SGLang 独有 | 两边都有（vLLM `:75`、SGLang `:96`） |
| 5 | vLLM 没有 `return_sampling_mask` | v0.31.0 有完整实现，18 个文件 |
| 6 | SGLang 的 Rust router 叫 `sgl-router` | v0.5.21 实际目录名是 `sgl-model-gateway/`（在主仓内） |

另外两条**参数陷阱**（[10](10-selection-playbook.md) §5.1 有详述）：

- `enable_prefix_caching`（vLLM）vs `disable_radix_cache`（SGLang）—— **一正一反**
- `async_scheduling`（vLLM）vs `disable_overlap_schedule`（SGLang）—— **一正一反**

照抄参数名会得到**相反的行为**。

---

## 最有价值的单条量化证据

SGLang v0.5.20 release notes 里这一组是全文质量最高的性能数字 ——
因为它是**同一次发布、同一模型、只改缓存机制**的前后对比，排除了大量混杂因素：

| 指标 | 前 | 后 | 变化 |
|:---|---:|---:|:---|
| token 命中率 | 43.8% | 60.8% | **+17pp** |
| 平均 TTFT | 1.57s | 1.07s | **约 −32%** |

即：**前缀命中率每 +10pp → TTFT 约降 19%**（长 prompt / 共享 system prompt 场景）。
厂商自报（B 级），但机制可解释、条件清晰。

这也解释了为什么性能选型的第一步不是跑 benchmark，
而是**先算自己负载的前缀命中率** —— 成本低两个数量级，信息量更大。

---

## 未找到的项（本文明确划界）

为便于后续核查，把「确认不存在」与「没查到」分开列：

**确认不存在**（已完整枚举源码）：
- vLLM 的 `gguf` / `bitsandbytes` 量化（无枚举值、无分派项、无对应文件）
- SGLang 的 `torchao` 量化
- SGLang 主仓的 TPU / JAX 代码
- vLLM 的 NPU platform 文件
- vLLM 名为 `shared_storage` 的 connector
- vLLM 的 `sampling_mask_max_tokens` 参数
- 两边的 Lark grammar 支持
- vLLM 的 drafter 插件注册表（只有 `Literal` + if-elif 链）

**没查到路径**（存在但未定位到）：
- `e4m3` 在两边量化配置里的精确行号（仅在 `__init__.py` 层未见该字面量）
- SGLang Rust router 的具体指标名称（已确证用了 `metrics_exporter_prometheus`）
- vLLM PD 分离的拓扑要求文档（`docs/` 全树无disaggregation 文档）
- `sgl-project/sglang-jax` 内部实现（不在取证范围的两仓内）
- vLLM 的 recovery instruction 白名单（`instruction` 是裸 `str`）

---

## 关于版本锁定

两个框架的 breaking change 都很频繁：

- **vLLM v0.29.0** 把 Model Runner V2 设为**所有模型**默认；
  v0.31.0 移除 8 个配置项（含 `--enforce-eager`、AllSpark INT8、`tokenizer_mode="slow"`）
- **SGLang v0.5.20** 移除 prefill CP v1 与 CUDA 12 产物，
  `/v1/responses` 服务端存储改为 opt-in；
  v0.5.21 移除 `SWARadixCache` / `MambaRadixCache` / `HiRadixCache`

所以「vLLM 支持某模型」这句话**必须带版本号**，
否则半年后可能就变了。逐行号引用同理。

---

## 复现方式

本系列的可核验性依赖源码锁定：

```bash
cd ~/Desktop/Projects/github/vllm && git rev-list -n1 v0.31.0     # db9527a4...
cd ~/Desktop/Projects/github/sglang && git rev-list -n1 v0.5.21   # e00930c5...
```

读指定 commit 的文件用 `git show "<sha>:<path>"`（**引号必须加** ——
zsh 下 `$C:path` 形式的展开会静默污染 `git show` 输出）。