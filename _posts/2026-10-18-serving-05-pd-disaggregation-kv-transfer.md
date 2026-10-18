---
title: "推理引擎（05）：PD 分离与 KV 传输"
date: 2026-10-18 19:00:00 +0800
categories:
  - 推理服务
tags: [serving-engine, vllm, sglang, pd-disaggregation, kv-transfer, nixl, mooncake, rdma, inference]
layout: post
mathjax: true
---

> **推理引擎系列 · 第 05 篇**
>
> 本系列面向**已经能把模型跑起来、但还没读过任何一个推理引擎源码**的工程师：把 vLLM 与
> SGLang 的功能差异、性能差异、选型结论，全部还原到**源码里可点开核验的那一行**。
>
> 取证基准：**vLLM `v0.31.0`**（commit `db9527a4`）／ **SGLang `v0.5.21`**（commit `e00930c5`）。
> 本文所有行号均锁在这两个 commit 上。
>
> - 00 引擎在栈里的位置 → 01 架构分歧的根 → 02 调度与批处理 → 03 支持矩阵（上）→
>   04 支持矩阵（下）→ **05 PD 分离与 KV 传输（本文）** → 06 benchmark 交叉验证 →
>   07 选型手册 → 08 前缀命中率

**TL;DR**

> * **PD 分离不提高吞吐，这是 vLLM 官方文档的原话**（`docs/features/disagg_prefill.md`
>   开篇的 note：「Disaggregated prefill DOES NOT improve throughput」）。它买的是两样东西：
>   TTFT 与 ITL 可以分开调，以及尾延迟可控。当成加速器用会失望。
> * **该不该传 KV，第一笔账与 prompt 长度无关**。传输时间与重算时间都正比于 token 数，
>   比值只由「每 token KV 字节数 ÷ 激活参数量」和链路带宽决定。companion 实验算出
>   Llama-3-8B 的盈亏平衡带宽是 **4.04 GB/s**，DeepSeek-V3（MLA）只有 **0.47 GB/s** ——
>   同一根 10GbE 链路上，8B 应该重算，MLA 模型应该无脑传。
> * **vLLM 的 PD 连接器注册表里有 17 项，文档只讲 9 种**；`NixlConnector` 只是
>   拉取式（READ）的别名，推送式是另一个类，两者协议不兼容，握手哈希会拦住混搭。
> * **P 侧显存什么时候解放，是一笔独立的账**。vLLM 用「30s 短租约 + 5s 心跳续租」
>    取代了早期的单一 480s 超时：D 崩溃时回收从 480s 降到 **27s**（仿真实测 17.8 倍差距），
>    同时不误杀健康但排长队的请求。
> * **SGLang 把同一问题拆成了更多层**：5 个传输后端、12 个 HiCache 存储后端、
>    独立的 encoder 分离、运行期 P/D 角色切换。区分「传输后端」与「存储后端」
>    是读它代码的第一道坎。

---

## 0. 速览：两个引擎的 PD 分离面

| 维度 | vLLM `v0.31.0` | SGLang `v0.5.21` |
|:---|:---|:---|
| 连接器 / 后端数量 | 注册表 **17 项**（`factory.py` L153–L245） | 传输后端 **5 个**（`utils.py` L565–570） |
| 注册机制 | 字符串懒加载，`register_connector(name, module, class)` | `TransferBackend` 枚举 + `KVClassType` 分派 |
| 传输方向 | pull（READ）/ push（WRITE）两个类，协议不兼容 | 后端内部实现（mooncake conn 3024 行） |
| 多模态 encoder 分离 | 有（`docs/features/disagg_encoder.md`） | 有（`disaggregation/encoder/` + 8 个配置字段） |
| 运行期角色切换 | 未见 | `/pd_role_switch` 端点（`http_server.py:1577`） |
| 故障回收机制 | lease + 心跳（设计文档 + 实现） | decode 侧重传备份（`cpu_tensor`/`host_pool`/`none`） |

两个引擎的共同点比差异更重要：**PD 分离在两边都是「实验性」功能**，
vLLM 的文档标题里就挂着 experimental；生产可用性取决于具体 connector / 后端，
不取决于引擎本身。

---

## 1. PD 分离解决什么问题（以及它明确不解决什么）

vLLM `docs/features/disagg_prefill.md` 开篇列了两条理由，第一条是**分别调优**：

> Tuning time-to-first-token (TTFT) and inter-token-latency (ITL) separately …
> assign different parallel strategies (e.g. `tp` and `pp`) to tune TTFT without
> affecting ITL.

第二条是**尾 ITL**：

> Without disaggregated prefilling, vLLM may insert some prefill jobs during the
> decoding of one request. This results in higher tail latency.

紧接着文档自己给出了对照：

> Chunked prefill with a proper chunk size also can achieve the same goal, but in
> practice it's hard to figure out the correct chunk size value. So disaggregated
> prefilling is a much more reliable way to control tail ITL.

这句话与 02 篇的结论对上了：chunked prefill 调 $$\tau$$ 是在**延迟分布**上做交易，
收益集中在短请求尾延迟、代价是长请求延迟单调恶化；PD 分离用「物理隔离」绕开了调参，
但把问题转移给了部署拓扑。同一份文档最后还有一句必须印在脑子里的话：

> **Disaggregated prefill DOES NOT improve throughput.**

SGLang 的 `docs/docs/advanced_features/pd_disaggregation.mdx` 给出了同向的动机叙述：
统一调度下的两个病症是 **Prefill Interruption**（prefill 打断 decode）与
**DP Attention Imbalance**（一个 DP worker 在跑 prefill、另一个在跑 decode，拖慢后者）。
它没有说「不提吞吐」，但列出的解法（分开两阶段、各自优化）与 vLLM 一致。

**本节要点**：PD 分离是一个**延迟特性的交易**，不是吞吐工具。
把它当吞吐优化手段部署，会在 benchmark 上得到「加了设备、QPS 没动」的结果。

---

## 2. vLLM 的三层抽象：Pipe / LookupBuffer / Connector

`vllm/distributed/kv_transfer/README.md` 把整个子系统拆成三层，
这个拆分解释了后面所有的设计选择：

| 层 | 职责 | 关键 API |
|:---|:---|:---|
| **KV pipe** | torch tensor 的 FIFO 管道 | `send_tensor` / `recv_tensor` |
| **KV lookup buffer** | 按 token 查 KV，语义类似 SQL | `insert`（非阻塞）/ `drop_select`（阻塞） |
| **KV connector** | 把上面两层接到 vLLM | `send_kv_caches_and_hidden_states` / `recv_kv_caches_and_hidden_states` |

README 解释了为什么 FIFO 不够用：

> prefill vLLM worker may process requests in order A -> B -> C, but the decode
> vLLM worker may process request C first.

生产者与消费者的**处理顺序不一致**，所以中间必须有一层能按 key 查找的缓冲。
README 还明确了两条「可绕过」路径：如果底层通信本身就支持 KV 查找（如 Redis、
RDMA database），pipe 层可以整体跳过；如果想改 vLLM 的执行流（比如只对部分 token
做 prefill），可以跳过前两层直接实现 connector —— 但文档警告这样会
「likely be broken when vLLM has new updates」。

连接器本身在 v1 里被拆成**两个角色**（`kv_connector/v1/base.py:137-143`）：

```text
KVConnectorRole:
    SCHEDULER = 0   # 与 scheduler 同进程，只做元数据决策
    WORKER = 1      # 与 worker 同进程，只做真正的收发
```

`base.py` 顶部的 docstring 把两边的 API 列得很清楚：scheduler 侧是
`get_num_new_matched_tokens`（能白拿多少 token 的 KV）、`update_state_after_alloc`、
`update_connector_output`、`request_finished`、`take_events`；
worker 侧是 `handle_preemptions`、`start_load_kv`、`wait_for_layer_load`、
`save_kv_layer`、`wait_for_save`、`get_transfer_results`。
工厂注释里写明了原因（`factory.py:66-73`）：scheduler connector 只应该在
Scheduler 类里被调用，worker connector 只应该在 forward context 与 attention 层里
被调用，**分开构造是为了强制这个边界**。

注册机制是字符串懒加载（`factory.py:28-40`）：`_registry` 字典存的是
`(module_path, class_name)`，`register_connector` 只在 import 时登记，
真正用到才 `importlib.import_module`。这样做的好处是**加载一个 connector
不会拖累其他 16 个的依赖**——Mooncake、NIXL、MoRI-IO 各有各的 native 依赖。
同名重复注册直接报错（`:34`）。

`factory.py` L153–L245 一共 **17 个注册项**（草稿阶段流传的「16 个」是旧数，
以本 commit 为准）：

| 行 | 名称 | 备注 |
|:---|:---|:---|
| L153 | `ExampleConnector` | 教程用 |
| L159 | `ExampleHiddenStatesConnector` | 传 hidden states 而非 KV |
| L165 / L171 | `LMCacheConnectorV1` / `LMCacheMPConnector` | 后者是独立 `lmcache server` 多进程模式 |
| L177 / L183 / L189 | `NixlConnector` / `NixlPullConnector` / `NixlPushConnector` | 拉取、拉取、推送 |
| L195 | `MultiConnector` | 多个 connector 串联，HMA 特判 |
| L201 | `HiSparseConnector` | 需要 `host_pool_gib` 配置 |
| L207 | `MoRIIOConnector` | ROCm only |
| L213 | `OffloadingConnector` | KV 卸到 CPU |
| L219 | `DecodeBenchConnector` | benchmark 专用 |
| L225 / L230 | `MooncakeConnector` / `MooncakeStoreConnector` | 直传 vs 走 store |
| L235 | `FlexKVConnectorV1` | 分布式 KV store |
| L240 | `SimpleCPUOffloadConnector` | 最简 CPU 卸货 |
| L245 | `HF3FSKVConnector` | 走 HF3FS 存储 |

官方文档 `disagg_prefill.md` 只讲了 **9 种**（Example / LMCacheV1 / Nixl / Mooncake /
MoRI-IO / Multi / Offloading / FlexKV，加一句「Now supports 9 types of connectors」）。
**17 与 9 的差不是「藏了 8 个功能」**：注册表里含 benchmark 专用（`DecodeBenchConnector`）、
教程（两个 Example）和被文档合并叙述的变体（NIXL 一家三口、Mooncake 一家两口）。
数注册项不能当成支持面。

---

## 3. 一次请求的生命周期：1P1D 的硬约束与 `kv_transfer_params`

vLLM 的 PD 协调几乎全部藏在一个字典里：请求的 `kv_transfer_params`。
`pull_scheduler.py:59` 与 `:69` 两个分支读它来决定行为：

- **`do_remote_prefill`**：本实例是 decode 侧，把所有 prompt 块从远端 prefill 引擎
  异步拉过来（`pull_scheduler.py:59-73`）；
- **`do_remote_decode`**：本实例是 prefill 侧，已经算好了请求的一部分 KV，
  把块信息（`remote_engine_id` / `remote_request_id` / `remote_host` / `remote_port` /
  `remote_block_ids`）回给路由，由路由转交给 decode 实例（`:69-104`）。

文档还给了一个省事的优化：decode 侧不必重新模板化 + tokenize。
prefill 请求带 `return_token_ids`，从响应里读 `prompt_token_ids`，
填到 decode 请求的 `kv_transfer_params["prompt_token_ids"]`；
`messages` 仍然必填，但内容不再被 tokenize（`disagg_prefill.md`「Reusing prefill
token ids on decode」一节，标注 experimental）。

三个硬约束值得先知道：

1. **`kv_rank` 的文档字符串写着「Currently only 1P1D is supported」**
   （`config/kv_transfer.py:79-82`）。1 prefill 实例对 1 decode 实例；
   多 P 多 D 的拓扑要自己在路由层拼。
2. **加载失败策略默认是 `fail`**（`kv_load_failure_policy`，`:103`）：
   KV 传输失败默认让请求直接失败；想要「失败了就地重算」得显式设 `recompute`。
3. **`kv_role` 与 `kv_connector` 必须成对出现**（`__post_init__`，`:135-137`）：
   只设 connector 不设角色会被拒。

---

## 4. pull 还是 push：一次握手定终身

`NixlConnector` 在文档里是主角，在代码里却只是 `NixlPullConnector` 的
向后兼容别名（`nixl/connector.py` 顶部 docstring 明说）。
真正的类谱系是：

- `NixlBaseConnector`（`:77`）—— pull/push 共享，声明
  `supports_divergent_local_hybrid_hits`（`:81`）；
- `NixlPullConnector`（`:339`）—— READ 方向，decode 侧主动拉；
- `NixlPushConnector`（`:367`）—— WRITE 方向，prefill 侧主动推。

**两者协议不兼容，混搭会被握手哈希拦住**。
`docs/features/nixl_connector_compatibility.md` 的「What must match between P and D」
列出的必相同项：

- vLLM 版本与 NIXL connector 版本；
- 模型（架构、dtype、KV head 数、head size、层数）；
- attention backend；
- `cache_dtype`；
- 投机解码方法与 draft 模型配置；
- **传输方向（push 与 pull 用不兼容的传输协议，绝不能配对）**。

可以安全不同的项同样重要，它决定了 P/D 能不能用不同并行度：

- `tensor-parallel-size`（异构 TP，受模型限制）；
- `block-size`（异构块大小，**仅支持 P 块小于 D 块**）；
- KV cache block 数（各实例按显存自定）；
- `num_speculative_tokens`（P/D 可以用不同 draft 深度）；
- draft 模型的 `attention_backend`（各自独立选，布局在握手时校验）。

架构 × 能力矩阵（同文档，原文七列）把边界画得很细，摘录四行四列：

| 模型类型 | Basic PD | Spec Decode | Hetero TP | Hetero block size |
|:---|:---:|:---:|:---:|:---:|
| Dense Transformer | ✅ | ✅ | ✅ | 🟠 |
| MLA（DeepSeek-V2/V3） | ✅ | ✅ | 🟠 | 🟠 |
| Hybrid SSM / Mamba | ✅ | ❔ | 🚧 | ❌ |
| Encoder-Decoder | ❌ | ❌ | ❌ | ❌ |

脚注里的信息量比表格本身大：MLA 的异构 TP 能做但**不做 head 拆分**
（KV 在 TP worker 间复制，P TP > D TP 时只发一次读，多余 rank 跳过）；
Hybrid SSM **要求同构 TP**；异构块大小只在非混合模型上支持，
且 **HMA 不支持不同的远端块大小**。

pull 与 push 还各有各的并行形态禁区（`nixl/connector.py`）：

- pull：PCP（prefill context parallel）**只支持 `kv_producer`**（`:105-109`），
  PCP + DCP 要求 TP1 且 DCP 跨满整个 PCP 组（`:111-122`），
  PCP producer 不支持双向传输（`:124-130`）；
- push：**不支持 `decode_context_parallel_size > 1`**（`:377-379`）。

`MultiConnector`（L195）允许把多个 connector 串成有序列表，
但 HMA 支持要**每一个子 connector 都支持**才算数
（`factory.py:132-146`：包装类自己实现 `SupportsHMA`，
实际支持度取决于全部配置项）。

---

## 5. KV cache 布局：传输性能逼出来的约束

`NixlBaseConnector.get_required_kvcache_layout`（`nixl/connector.py:141-158`）
做了一件很「物理」的事：**连接器反过来要求引擎改 KV 布局**。
非 MLA 模型直接返回 `"LBHNC"`（head-major，旧称 HND），日志原话是
「setting KV cache layout to LBHNC for better xfer performance」；
MLA 返回 `None`——压缩后的 latent 只有一份，布局不影响传输。

兼容矩阵文档把布局规则写全了：

- `LBHNC`（head-major）是默认；
- `LBNHC`（token-major，旧称 NHD）**支持但不允许异构 TP 的 head 拆分**；
- 两种布局间的 permute 是实验特性，用
  `enable_permute_local_kv` 打开（`config/kv_transfer.py:100`），**不与 HMA 兼容**。

为什么布局要由传输方定：RDMA 按连续内存块搬数据，head-major 让同一个 head 的
全部 token 连续，一次读就是一整块；token-major 则要把同一个 head 的切片从
每一层、每个 token 位置抠出来。兼容矩阵还提到 cross-layer 连续性要靠
`BLHNC` 布局实现（`VLLM_KV_CACHE_LAYOUT=BLHNC`）。

Pipeline parallelism 是另一个坑：**默认的 pull connector 不支持
`pipeline-parallel-size > 1` 配 HMA 布局**——region index 在 P/D 分层切分后
不是稳定身份，连接器直接在启动时报错。要 PP 就用 push connector：
它按 layer-name（成员身份）路由传输，所以 PP 分片的 prefill 可以写进
PP=1 的 decode。push 的 PP + HMA 限制也很长：只有 prefiller 可以 PP 分片、
不支持混合 SSM 布局、HMA 要求 P/D 同块大小、非 MLA 的 attention-HMA 层路由
要求 decode TP 不大于 prefill TP。

---

## 6. 量化 KV 穿过 P/D 边界

量化 KV cache（如 FP8）在 PD 分离下的规则，兼容矩阵写得很短但很硬：

- P/D 必须用**相同的 `cache_dtype`**，不匹配会挂握手哈希；
- **静态量化 ✅**（scale 从 checkpoint 加载，各实例各自读）；
- **动态量化 ❌**（scale 运行时算，per-block scale 不随 KV 一起传）；
- **packed-layout ✅**（scale 与权重内联存放，随 KV 块一起传）。

一句话：**P/D 边界上能传的是「数据 + 内联元数据」，任何需要传额外 sidecar
的量化方案都过不来**。这条对选型有直接后果——想 PD 分离又想要 FP8 KV，
先确认你的量化是静态或 packed 的。

---

## 7. 租约与心跳：P 侧显存什么时候解放

prefill 算完之后，P 的 KV 块不能立刻释放：要等 D 通过 RDMA 读走。
D 什么时候读、读不读，P 说了不算——于是有了租约。
`docs/design/nixl_kv_cache_lease.md` 记录了这段设计史，是少见的
「把失败模式写进文档」的案例：

**旧设计是一个大超时**（`VLLM_NIXL_ABORT_REQUEST_TIMEOUT`，默认 480s），
它同时踩两个坑：

- D 崩溃或断连时，P 要把可能几 GB 的「死块」攥到 8 分钟，期间新请求可用缓存减少；
- 流量突发时请求会在 D 的等待队列里排很久，超时调短了，块在 D 读到之前就被释放，
  造成无谓重算。

**新设计是短租约 + 心跳续租**：

- prefill 完成即授予 **30s 初始租约**（`kv_lease_duration` 默认值）；
- D 侧只要请求**在排队或在飞**，就周期发心跳，每次把租约续
  **20s**（`lease_duration * 2/3`），心跳间隔 **5s**（`lease_duration // 6`）；
- 心跳不新开通道，**搭载在 NIXL 的通知系统上**（`send_notif` / `get_new_notifs`），
  传输介质随后端自动从 IB/RoCE 回退到 TCP；
- 一条心跳消息续该 D 在对应 P 上的**全部**在钉请求——按 (D, P) 配对批量发，
  成本与在飞请求数无关；
- 心跳跟踪从 `on_new_request()` 就开始（请求进 D 的 scheduler 那一刻，
  不是被调度执行的那一刻），因为排队时长没有上界；
- 心跳的发送与处理都在 **forward loop** 里做，不开后台线程——时间不精确，
  但默认参数下心跳间隔与续租量比一次前向大一个数量级，够用。

companion 实验 `pd-disaggregation-transfer-cost` 组 2 把这个状态机跑了出来
（参数即文档默认值，事件驱动仿真）：

| 场景 | 设计 | 心跳数 | 块驻留 | 白占 |
|:---|:---:|---:|---:|---:|
| A 健康：排队 100s 后传输 | lease+hb | 21 | 102.5s | 0.0s |
| B 崩溃：D 在 t=10s 死掉 | lease+hb | 2 | **27.0s** | 17.0s |
| C 过载：心跳间隔拉到 4.8s | lease+hb | 26 | 122.5s | 0.0s |
| D 旧设计：单一 480s 超时，D 崩溃 | single-timeout | 0 | **480.0s** | 480.0s |

场景 B 的 27.0s 是这么来的：最后一次心跳在 t=7s，续租 20s → t=27s 到期回收。
**崩溃回收从 480s 降到 27s，17.8 倍差距**；而场景 A/C 里健康但排队的请求
一次都没被误杀。设计不变量也很直白：**续租时长必须大于心跳间隔**
（20s > 5s），差值就是「单次心跳丢失还能撑多久」（15s）。

---

## 8. 划得来吗：传输与重算的盈亏平衡

PD 分离最常见的误用动机是「让 prefill 快点」。把账算清楚，
第一笔是**传输 vs 重算**：把 KV 从 P 传到 D，和 D 自己把 prefill 重算一遍，哪个快？

设每 token KV 字节数为 $$b$$，链路带宽 $$BW$$，激活参数量 $$N_{\mathrm{active}}$$，
前向算力 $$F$$。前向 FLOPs 约 $$2 N_{\mathrm{active}} L$$（$$L$$ 为 prompt 长度），于是

$$
t_{\mathrm{xfer}} = \frac{L\,b}{BW}, \qquad
t_{\mathrm{recompute}} = \frac{2\,N_{\mathrm{active}}\,L}{F}
$$

两边都有 $$L$$，**比值与 prompt 长度无关**：

$$
\frac{t_{\mathrm{xfer}}}{t_{\mathrm{recompute}}}
= \frac{b\,F}{2\,N_{\mathrm{active}}\,BW}, \qquad
BW^{\ast} = \frac{b\,F}{2\,N_{\mathrm{active}}}
$$

$$BW^{\ast}$$ 是盈亏平衡带宽：链路快于它，传输便宜；慢于它，重算便宜。
companion 实验组 1 用闭式解与逐层累加对拍后算出（BF16 KV，H100 BF16 dense
494.7 TFLOPS 标称值，MoE 用激活参数量）：

| 模型 | $$b$$（KiB/token） | $$BW^{\ast}$$（GB/s） | NVLink 4 | PCIe Gen5 | IB/RoCE | TCP 10GbE |
|:---|---:|---:|---:|---:|---:|---:|
| Llama-3-8B | 128.0 | 4.037 | 0.01 | 0.08 | 0.08 | **3.23** |
| Qwen3-32B | 256.0 | 1.977 | 0.00 | 0.04 | 0.04 | 1.58 |
| Llama-3-70B | 320.0 | 1.148 | 0.00 | 0.02 | 0.02 | **0.92** |
| DeepSeek-V3（MLA） | 68.6 | **0.470** | 0.00 | 0.01 | 0.01 | 0.38 |

（单元格 = $$t_{\mathrm{xfer}} / t_{\mathrm{recompute}}$$，<1 表示传输更快。
实验同时在 $$L \in [512, 131072]$$ 五个数量级上验证了比值恒为 0.08。）

三个可以直接拿去用的结论：

1. **「长请求才值得传」是错的**。比值与 $$L$$ 无关，长短请求一样划算
   （就传输/重算这一笔而言）。
2. **比值由 $$b / N_{\mathrm{active}}$$ 决定，也就是 KV 缓存相对模型的「密度」**。
   dense 模型 $$b$$ 随层数线性涨，$$BW^{\ast}$$ 落在 1–4 GB/s；
   MLA 把每 token 缓存压到 576 个元素（512 latent + 64 rope），
   MoE 又只激活 37B/671B 参数，$$BW^{\ast}$$ 掉到 0.47 GB/s——
   **这种模型在任何现代链路上都不该重算**。
3. **10GbE TCP 是个分水岭**：8B 在上面重算快 3.2 倍，70B 传输已经反超。
   跨机房、只有 TCP 的场景，先算 $$BW^{\ast}$$ 再决定部不部署 PD。

还有第二笔账常被忘掉：**P 侧驻留**。租约窗口内 P 钉住的 KV 是实打实的显存
（实验组 1 的 1d 节，BF16）：

| 模型 | $$L=4096$$ | $$L=32768$$ |
|:---|---:|---:|
| Llama-3-8B | 0.50 GB | 4.00 GB |
| Qwen3-32B | 1.00 GB | 8.00 GB |
| Llama-3-70B | 1.25 GB | 10.00 GB |
| DeepSeek-V3（MLA） | 0.27 GB | 2.14 GB |

D 崩溃时这些块要白占到租约到期——正是第 7 节那笔账。

引擎内部其实有同款决策：`pull_scheduler.py:90-104` 的
`kv_recompute_threshold`，远端 token 数低于阈值就**跳过拉取、就地重算**。
差别是引擎只看 token 数，而我们这里按模型与链路算 $$BW^{\ast}$$——
把引擎的阈值按第 8 节的公式反推，比拍脑袋设一个数靠谱。

---

## 9. SGLang 侧：同一问题的另一套答案

SGLang 把 PD 分离做成了一个**配置面很宽**的子系统。
`python/sglang/srt/disaggregation/` 下 40 个文件，
传输后端只有 5 个（`utils.py:565-570`）：

```text
class TransferBackend(Enum):
    MOONCAKE / MORI / NIXL / ASCEND / FAKE
```

`mooncake/conn.py` 一个文件 3047 行，是 SGLang 侧的主力实现。
它的传输路径值得细看，因为「hetero TP 怎么传」的答案在这里：

`send_kvcache_staged`（`:626`）的 docstring 一行写完了三步：
**gather → bulk RDMA → scatter**。P 侧把 KV head 切片 gather 进一块连续
staging buffer，bulk RDMA 整块推过去，D 侧再 scatter 进正确的 KV page。
`compute_head_slice_params` 负责算「本地 TP rank 该发哪些 head」，
PP > 1 时还要按 `[all K, all V]` 顺序配对 staging slot。
SGLang 文档给的量化收益：高并发下比默认的 per-token slice
**快 2–5 倍**，与同构 TP 基线差距在 5% 以内（`pd_disaggregation.mdx:294`）。

auxiliary data（元数据）是另一条路，代码里留着 TODO：
`send_aux`（`:1475`）在 NVLink 自定义内存池下直接改走
`send_aux_tcp`（`:1499`），注释原话是
「Fix me when nvlink_transport of Mooncake is bug-free」——
文档也两次明说 auxiliary data 仍走 TCP（`:195`、`:202`）。
**KV 走 NVLink、元数据走 TCP，这是当前实现的真实边界。**

`arg_groups/fields/disagg.py` 把可调项全列了出来，几个有信息量的：

- `disaggregation_enable_kv_checksum`（默认关）：prefill 侧对每请求的 KV 页
  算 Adler-32，decode 侧校验，不匹配就中止请求——**默认不开的传输校验**；
- `disaggregation_decode_retraction_backup`：decode 侧重传时 KV 的退路，
  `cpu_tensor` / `host_pool` / `none`；选 `none` 时被重传的请求直接 503 让客户端重试；
- `num_reserved_decode_tokens`（默认 512）：新请求进 running batch 时预留的 decode 槽位；
- `optimistic_prefill_attempts`：乐观 prefill 前向次数，跳过 bootstrap 等待；
- `enable_pdmux` + `sm_group_num`（默认 8）：PD 跑在 greenctx stream 上做 SM 分区复用。

**encoder 分离**是 SGLang 的一个完整子特性，配置字段就有 8 个：
`encoder_only` / `language_only` / `language_model_only` /
`encoder_transfer_backend`（`auto` / `zmq_to_scheduler` / `zmq_to_tokenizer` /
`mooncake`）/ `encoder_urls` / `encoder_bootstrap_port`（默认 8997）/
`encoder_register_urls` / `enable_adaptive_dispatch_to_encoder`。
注意 `language_model_only` 与 `language_only` 不是一回事：前者是**独立模式**
（连 encoder 权重都不加载，多模态请求直接拒绝），后者是 encoder/decoder
分离的一部分。

**运行期角色切换**是 vLLM 侧看不到的能力：
`/pd_role_switch` 端点（`http_server.py:1577`）→
`handle_pd_role_switch`（`role_switch.py:43`）。
切换条件很严：实例必须**完全 idle**（`is_fully_idle()`），
目标 decode 角色要检查 CUDA graph 显存余量；
文档字符串把失败语义写得很直白——
「teardown + 角色翻转 + 重建是一个逻辑原子步骤，任何一步抛异常，
实例就处于半拆除状态，**没有原地回滚**」。
这个端点**没有用户文档**（`pd_disaggregation.mdx` 里搜不到 role switch），
属于「代码里有、文档里没有」的功能。

最后一个必须厘清的混淆点：**存储后端不是传输后端**。
`mem_cache/storage/` 下的 HiCache 体系有 **12 个已注册存储后端**
（`backend_factory.py:202-266`：`file` / `sim` / `nixl` / `mooncake` /
`npu_memcache` / `hf3fs` / `aibrix` / `eic` / `simm` / `mori` / `shm` /
`tensorcast`），抽象基类 `HiCacheStorage(ABC)` 在
`hicache_storage.py:170`，工厂 `register_backend` 在 `backend_factory.py:44`。
它们回答「KV 卸到哪」；`TransferBackend` 的 5 个回答「KV 怎么从 P 到 D」。
两件事都碰 `mooncake` 这个名字，但一个走 store、一个走 transfer engine，
**读 SGLang 的 PD 代码先分清这两个命名空间**。

---

## 10. 拓扑：NVLink、RDMA、EFA 与它们的明暗线

PD 分离的所有约束最后都落在**物理拓扑**上。SGLang 文档
（`pd_disaggregation.mdx`）是两边唯一系统的拓扑说明：

- **同机 NVLink**：`SGLANG_MOONCAKE_CUSTOM_MEM_POOL` 取 `NVLINK`
  （文档推荐 NVL72）或 `INTRA_NODE_NVLINK`（推荐 A100/H20/H100）。
  但**auxiliary data 仍走 TCP**（`:195`、`:202` 两次明说），
  代码侧对应 `send_aux` 的 TODO。
- **跨机 RDMA**：`:91` 起有「IB Device Configuration」章节；
  AWS EFA 要换 `mooncake-transfer-engine-efa-cuda13` 包、
  设 `MOONCAKE_PROTOCOL=efa`。
- **EFA 的暗线**（`:78-89`）：wheel 仅 x86_64，且**不含 Mooncake EP/PG
  与 NVLink/MNNVL 组件**；libfabric 的 EFA provider **传不了 CUDA VMM
  支撑的 buffer**（引用 libfabric issue #12853），所以 EFA 下不能设
  `NVLINK` / `BAREX`，`INTRA_NODE_NVLINK` 可以；EFA 协议自动关闭
  post-capture KV sizing；`MC_FORCE_MNNVL` / `NCCL_MNNVL_ENABLE` 不要设。
- **启动即失败优于运行时报错**：
  `SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT = EnvInt(60)`，
  注释原话「A wedged RDMA stack fails startup here instead of at the
  scheduler watchdog」。

vLLM 侧要纠正一个此前的说法：**`db9527a4` 上 vLLM 有 5 篇 PD 相关文档**——
`docs/features/disagg_prefill.md`（总览 + 9 种 connector + token id 复用）、
`docs/features/nixl_connector_compatibility.md`（兼容矩阵）、
`docs/design/nixl_kv_cache_lease.md`（租约）、
`docs/design/nixl_kv_push_connector.md`（push 设计）、
`docs/features/disagg_encoder.md`（encoder 分离）。
但**没有一篇讲拓扑**：全文搜不到 RDMA/NVLink 的部署要求，
`nixl/connector.py` 里也搜不到 `RDMA|nvlink|IB` 的实质内容
（只有类定义行）。可间接确证的是存在 `mooncake/rdma_utils.py`（文件名带 rdma）。
同理，encoder 分离也**不是 SGLang 独有**：vLLM 有
`examples/disaggregated/disaggregated_encoder/`（1E1PD、1E1P1D 两个脚本 +
EPD proxy）与 `ECMooncakeConnector`（走 Mooncake TransferEngine 传 encoder 输出），
配置项是 `--ec-transfer-config`。真实差距在别处：vLLM 侧没有 SGLang 的
`/pd_role_switch`，SGLang 侧没有 vLLM 的 lease 设计文档与 TP 感知的
`tp_mapping.py`。

---

## 11. 小结：一正一反与取证边界

**一正一反**：

| 说法 | 核对结果 |
|:---|:---|
| 「PD 分离能提吞吐」 | ❌ vLLM 文档原话 DOES NOT improve throughput；它交易的是 TTFT/ITL 独立性与尾延迟 |
| 「prompt 越长越值得传 KV」 | ❌ 传输/重算比值与 $$L$$ 无关，由 $$b\,F/(2N_{\mathrm{active}}BW)$$ 定；10GbE 上 8B 该重算、70B 该传 |
| 「NixlConnector 是双向的」 | ❌ pull/push 是两个类、两套协议，握手哈希禁止混搭；`kv_both` 在 NixlConnector 上已废弃（启动即 warning） |
| 「MLA 模型异构 TP 会拆 head」 | ❌ MLA 的 KV 在 TP 间复制，只发一次读；dense GQA 才拆 head（`tp_mapping.py` 的 `rank_offset_factor`） |
| 「FP8 KV 随便配配就能 PD」 | ❌ 动态量化过不了 P/D 边界（per-block scale 不随 KV 传）；静态与 packed 可以 |
| 「SGLang 的 PD 后端有十几个」 | ❌ 传输后端 5 个；「十几个」是 HiCache 的 12 个**存储**后端，两个命名空间 |
| 「vLLM 没有 PD 文档 / 没有 encoder 分离」 | ❌ 本文基准 commit 上 5 篇文档、`disagg_encoder` 示例与 `ECMooncakeConnector` 都在 |
| 「D 崩了 P 要攥 8 分钟显存」 | ⚠️ 旧设计如此（480s）；新设计 30s 租约 + 心跳，仿真实测 27s 回收 |

**取证边界**：

- 本文所有行号锁在 vLLM `db9527a4`（v0.31.0）与 SGLang `e00930c5`（v0.5.21）。
  connector 数量、字段默认值都是**该 commit 的快照**，两边都在高频变动。
- 第 8 节全部数字来自 companion 实验的**解析模型**：链路带宽与 GPU 算力取标称值，
  未实测；MoE 用激活参数量是建模假设；没算 PD 引入的调度、路由、序列化开销，
  也没算 prefill 批处理对重算成本的摊薄。
- 第 7 节是**设计文档参数的状态机复现**，不是对 vLLM 运行时的测量；
  心跳在 forward loop 里发送导致的抖动未建模。
- SGLang 文档的「2–5x」是其自报数字，本文未复现。
- `pd_role_switch` 无用户文档，行为以代码为准；`disaggregation_decode_host_receive_threshold`
  的 help 文本自承「No built-in backend currently supports this」——
  这类「配置存在、内置后端不支持」的字段在 SGLang 里不止一个，选型时以代码为准。

## 实验与数据

companion 实验 `pd-disaggregation-transfer-cost`（纯 CPU、确定性输出、
`--selftest` 可挂 CI）：
<https://github.com/lrypcy/ipynbs/blob/1ecce83/experiments/serving-engine/pd-disaggregation-transfer-cost/run.py>

- 组 1：每 token KV 字节闭式 vs 数值对拍；比值随 $$L$$ 不变的五点验证；
  四模型 × 四链路 $$BW^{\ast}$$ 表；单请求 KV 体量。
- 组 2：lease + heartbeat 状态机四场景（健康 / 崩溃 / 过载 / 旧设计对照）。

## 上一篇 / 下一篇

[《推理引擎（04）：支持矩阵（下）》](/2026/10/18/serving-04-support-matrix-api/)；
调度与批处理见[《推理引擎（02）》](/2026/10/18/serving-02-scheduling-and-batching/)；
投机解码与 PD 的交互约束见
[《投机解码（04）：引擎支持矩阵》](/2026/10/18/specdec-04-engine-support-matrix/)；
prefix caching 的粒度代价见
[《推理引擎（01）》](/2026/10/18/serving-01-architecture-divergence/)。

## 参考

1. vLLM `v0.31.0`，`vllm/distributed/kv_transfer/README.md` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/README.md>
2. vLLM `v0.31.0`，`vllm/distributed/kv_transfer/kv_connector/factory.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/factory.py>
3. vLLM `v0.31.0`，`vllm/distributed/kv_transfer/kv_connector/v1/base.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/base.py>
4. vLLM `v0.31.0`，`vllm/config/kv_transfer.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/kv_transfer.py>
5. vLLM `v0.31.0`，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py>
6. vLLM `v0.31.0`，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py>
7. vLLM `v0.31.0`，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py` — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py>
8. vLLM 文档：Disaggregated Prefilling — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/features/disagg_prefill.md>
9. vLLM 文档：NixlConnector Compatibility Matrix — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/features/nixl_connector_compatibility.md>
10. vLLM 设计文档：NIXL KV Cache Lease Renewal — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/nixl_kv_cache_lease.md>
11. vLLM 设计文档：NIXL push-mode KV transfer — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/nixl_kv_push_connector.md>
12. vLLM 文档：Disaggregated Encoder — <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/features/disagg_encoder.md>
13. SGLang `v0.5.21`，`python/sglang/srt/disaggregation/utils.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/disaggregation/utils.py>
14. SGLang `v0.5.21`，`python/sglang/srt/disaggregation/mooncake/conn.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/disaggregation/mooncake/conn.py>
15. SGLang `v0.5.21`，`python/sglang/srt/disaggregation/prefill.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/disaggregation/prefill.py>
16. SGLang `v0.5.21`，`python/sglang/srt/disaggregation/role_switch.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/disaggregation/role_switch.py>
17. SGLang `v0.5.21`，`python/sglang/srt/arg_groups/fields/disagg.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/arg_groups/fields/disagg.py>
18. SGLang `v0.5.21`，`python/sglang/srt/mem_cache/storage/backend_factory.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/mem_cache/storage/backend_factory.py>
19. SGLang `v0.5.21`，`python/sglang/srt/mem_cache/hicache_storage.py` — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/mem_cache/hicache_storage.py>
20. SGLang 文档：PD Disaggregation — <https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/docs/docs/advanced_features/pd_disaggregation.mdx>
21. companion 实验：pd-disaggregation-transfer-cost — <https://github.com/lrypcy/ipynbs/blob/1ecce83/experiments/serving-engine/pd-disaggregation-transfer-cost/run.py>
