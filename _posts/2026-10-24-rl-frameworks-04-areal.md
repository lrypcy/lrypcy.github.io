---
title: "AReaL 2.1 源码拆解：把全异步做成第一公民"
date: 2026-10-24 21:00:00 +0800
categories:
  - 强化学习
tags: [rl, rlvr, grpo, ppo, areal, async, off-policy, staleness, decoupled-ppo, sglang, fsdp, megatron, rpc, benchmark]
layout: post
mathjax: true
---

> **系列导航** ｜ 承接 [RL 训练框架全景横评（2026）](/2026/09/29/rl-training-frameworks-survey/) 的三族划分与四轴定位，与 [RL 框架里的那些变量到底怎么算](/2026/09/30/rl-variables-in-frameworks/) 的逐行对照互补 ｜ 同系列：[Megatron 训练栈](/2026/10/24/rl-frameworks-01-megatron/) ｜ [slime](/2026/10/24/rl-frameworks-02-slime/) ｜ [veRL](/2026/10/24/rl-frameworks-03-verl/) ｜ 本篇为第 4 篇 · AReaL。
>
> 基准：`v2.1.0` = `ecc8b0e4dfb4e3965f67121a5c87866a07c390ae`。本文只谈 AReaL 自己的设计，不复述算法推导与并行原理。

**TL;DR**

> * **AReaL 是 BRIEF 四级异步定义里唯一落在最高一级的框架**：训练端与推理端是**两个独立服务**，用**样本队列**连接，staleness 由队列深度而不是全局屏障控制。`max_head_offpolicyness=0` 时它退化成同步系统，官方文档自述同步比异步慢约 2 倍 `[官方]`。
> * **队列不是一个数据结构，而是一个「准入控制器」**。§4 拆开的 `StalenessManager.get_capacity()` 是全文最核心的一行：容量 $$=\min(\text{并发槽},\;(\eta+v+1)B-N_{\rm accepted+running})$$。超限的样本**既不丢弃也不降权，而是让生产者停下来**——限流点在入口，不在出口。
> * **可中断 rollout 与 slime 的 partial rollout 是两件不同的事**。AReaL 中断后**丢 KV、把已生成前缀当 prompt 重算**，整条轨迹的每个 token 都记着自己的生成版本（`versions` 张量）；slime 中断后**把未完成样本回收进 buffer 续跑**，可选地把跨版本前缀的 `loss_mask` 置零。前者保留数据、承担陈旧；后者丢弃陈旧 token、承担样本浪费。
> * **off-policy 修正落在损失函数里，不在数据管道里**。AReaL 实现的是 decoupled PPO：行为策略 $$\pi_{\rm behav}$$、近端策略 $$\pi_{\rm prox}$$、当前策略 $$\pi_\theta$$ 三者分离，$$\pi_{\rm prox}/\pi_{\rm behav}$$ 作为行为重要性权重乘进 PPO 的裁剪目标；额外的 token 过滤由 `rejection_sampling` 做（`ratio` / `kl_k1` / `kl_k2` / `kl_k3` / `binary_kl`，`mask` 或 `clamp`）。
> * **2.x 的目录结构已经不只是「api / engine / controller」三层**：`areal/v2/` 下另有 `training_service` `inference_service` `agent_service` `weight_update` 四个**微服务**，每个都带 `gateway / router / guard / worker` 四件套。1.x 的 `infra/controller` 仍在，两套并存。
> * **论文口径的加速比必须带着前提读**：`[论文]` 端到端最快 **2.77×**（LiveCodeBench，32B，48 节点），但**同一张表里 7B 数学任务是 2.27×，1.5B 是 2.77×**；强扩展（strong scaling）口径下最多 **2.5×**。所有数字都在 H800 + 75/25 推理/训练卡比 + `rollout.n=32` + 32K 上限下取得，**跨框架横向引用只在论文同一张表内成立**。

---

## 目录

- [1. 设计出发点：为什么同步屏障不可接受](#1-设计出发点为什么同步屏障不可接受)
- [2. 模块初始化：从入口脚本到所有 worker 就绪](#2-模块初始化从入口脚本到所有-worker-就绪)
- [3. 进程与角色：控制面、数据面与 RPC 通道](#3-进程与角色控制面数据面与-rpc-通道)
- [4. 样本队列与 staleness 控制](#4-样本队列与-staleness-控制)
- [5. 可中断 rollout：与 slime partial rollout 的对照](#5-可中断-rollout与-slime-partial-rollout-的对照)
- [6. off-policy 修正：decoupled PPO 与 token 过滤](#6-off-policy-修正decoupled-ppo-与-token-过滤)
- [7. 训推一致三层](#7-训推一致三层)
- [8. 监控系统：指标层、追踪层、归因层](#8-监控系统指标层追踪层归因层)
- [9. 性能：论文口径与不可比性](#9-性能论文口径与不可比性)
- [10. 小结：它解决到什么程度、还剩什么](#10-小结它解决到什么程度还剩什么)

---

## 1. 设计出发点：为什么同步屏障不可接受

### 1.1 问题陈述

LLM 的 RL 训练里，一次迭代是「生成 → 打分 → 训练」三段。同步系统把这三段串成流水线的三个阶段，每个阶段跑完才进下一个；一个 batch 里所有 rollout 都由同一个策略版本生成。论文 `[论文]` 对这种做法的判词是「训练稳定，但遭受严重的系统级低效：生成必须等到 batch 里最长的输出完成才能更新模型，导致 GPU 利用率不足」<a href="#ref-paper">[1]</a>。

这个低效有两个来源，AReaL 分别用两种手段处理：

| 低效来源 | 表现 | AReaL 的处置 | 对应章节 |
| --- | --- | --- | --- |
| **批内长尾** | 同一 batch 内最长序列决定屏障时刻，短序列的卡空转 | 可中断生成：不等最长，随时换权重 | §5 |
| **阶段间串行** | 训练时推理卡空转、推理时训练卡空转 | 训练端与推理端解耦成两个服务，各自跑满 | §3、§4 |

这两个处置不是同义词。**前者是时间维度上打破批内屏障，后者是空间维度上打破阶段间屏障**。slime / veRL 的 partial rollout 只做前者；AReaL 两件都做，这才是它落在「全异步」那一级的原因。

### 1.2 放弃了什么

解耦是有代价的，且代价是**算法层面**的，不是工程层面的：

1. **on-policy 假设失守**。数据由旧策略生成，训练时策略已经前进若干步。朴素 PPO 的裁剪中心是行为策略，一旦行为策略变旧，裁剪区间就围绕一个错的中心展开。论文实测：`η=4` 时朴素 PPO 的 AIME24 从 42.0 掉到 **23.3** `[论文]`，而配上 decoupled 目标后回到 **42.2**。
2. **必须引入新的超参**。$$\eta$$（`max_head_offpolicyness`）把一个原本不存在的调优项塞进了用户的配置面。论文自己的建议是「采用较大的 $$\eta$$ 以获得最好的系统吞吐」——也就是说，**算法稳定性上让步、换取系统吞吐，再用算法手段把这个让步补回来**。
3. **中断让一条轨迹跨多个版本**。这在同步系统里根本不会发生，于是产生了一个新问题：一条轨迹的不同 token 由不同版本生成，loss 该怎么归因？AReaL 的答案是**逐 token 记录版本**（§5.4 + §8）。

### 1.3 1.x 到 2.x 的架构演进

README 的 News 区把版本线写得很清楚 `[官方]`：

| 版本 | 时间 | 架构形态 | 关键点 |
| --- | --- | --- | --- |
| v0.1 | 2025-02 | 单进程/单控制器雏形 | 1.5B、7B 可复现 |
| v0.2「boba」 | 2025-03 | 引入 SGLang 推理后端 | 推理侧独立成进程 |
| v0.3「boba²」 | 2025-06 | 全异步首次成型 | 论文口径 2.77× |
| v1.0 | 2025 年内 | 全异步 + `RolloutController` / `TrainController` 双控制器 | 稳定期，本文讲的 1.x 主干 |
| **v2.0** | 2026-07 | **微服务架构**：training / inference / agent / weight-update 四个独立服务 | 技术报告 arXiv 2607.01120 |
| **v2.1.0** | 2026-08-25 | 在 2.0 基础上补齐服务间协议与 CLI | **本文基准** |

2.0 的定性是「一个重大的架构里程碑……把 AReaL 重构成微服务架构，带有独立的训练、推理、agent 和权重更新服务」`[官方]`。落到代码上，`areal/v2/` 与 1.x 的 `areal/infra/` **并存**而不是替换关系——这一点决定了读这份源码时必须先分清自己在看哪一套（§3.1）。

---

## 2. 模块初始化：从入口脚本到所有 worker 就绪

### 2.1 入口脚本极薄

官方 GSM8K 示例的完整主体只有不到 50 行（<a href="#ref-entry">[2]</a>）：读 YAML 配置 → 加载 tokenizer → 构造数据集 → `with PPOTrainer(...) as trainer: trainer.train(workflow=...)`。所有分布式细节都藏在 `PPOTrainer.__init__` / `train` 里。这是 AReaL 的 API 设计取向：**算法侧只写 workflow，系统侧全交给 trainer**。

### 2.2 `PPOTrainer` 的初始化序列

`_init_impl` 的顺序是「环境 → 分配 → 引擎 → 数据服务」（<a href="#ref-rltrainer">[3]</a>）：

1. **决定单控还是 SPMD**。`is_single_controller()` 读环境变量 `AREAL_SPMD_MODE`；非 SPMD（默认）才建 `scheduler`、才建 `DataController`。这是全文第一处分支，后面几乎所有路径都要先问它。
2. **解析资源分配**。`ModelAllocation.from_str(config.actor.backend)` 和 `... config.rollout.backend` 各解析一次，得到 actor 与 rollout 两套并行策略。**训练与推理的并行度从配置里就分开写**，这是解耦在配置面上的第一体现。
3. **判断要不要 offload**。`_should_offload_rollout = _is_actor_rollout_colocated(config)`：colocate 时两边抢同一批卡，必须靠 onload/offload 轮流上；分离部署时不用。
4. **建引擎**。actor 用 `_create_train_engine` → 返回 `TrainController`；critic / ref / teacher 按需；rollout 用 `_init_rollout` → 返回 `RolloutController`（或 2.x 的 `RolloutControllerV2`）。

### 2.3 `TrainController.initialize`：训练端怎么起

训练端的起法是**先起壳、再注入引擎**，分四步（<a href="#ref-trainctl">[4]</a>）：

```
Job(replicas=world_size, tasks=scheduling_spec, role=role)
      │
      ├─ scheduler.create_workers(job)   → 拉起 N 个 worker 进程
      ├─ scheduler.get_workers(role)     → 等全部就绪
      │
      ├─ 取 rank0 的 engine_ports[1] 当 MASTER_PORT，rank0 的 ip 当 MASTER_ADDR
      │
      ├─ Phase 1: 给每个 worker set_worker_env(RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT)
      │           然后 create_engine(engine="areal.engine.fsdp_engine.FSDPPPOActor",
      │                               engine_name="actor/<rank>")
      │
      └─ Phase 2: async_call_engine(method="create_process_group")  → 建 NCCL 进程组
                  async_call_engine(method="initialize", ft_spec=...) → 载模型、建优化器
```

两个细节值得单独拎出来：

- **`LOCAL_RANK` 恒为 0**。`TrainController._async_create_engines` 里把 `LOCAL_RANK` 硬编码成 `"0"`，注释写明「每个进程只用一张 GPU」<a href="#ref-trainctl">[4]</a>。这意味着**一个 worker 进程 = 一张卡 = 一个引擎实例**，TP/PP 是靠引擎内部的进程组跨 worker 拼出来的，不是靠单进程内多卡。这是「微服务化」的前提：只有一卡一进程，才能在任意时刻增删一个 worker 而不影响别的。
- **引擎是按需动态导入的**。`engine` 参数是字符串形式的类路径，worker 侧用 `import_from_string` 反序列化。所以「训练后端可换」不是编译期的事，而是**配置里换个字符串**的事——FSDP / Megatron / Archon 三套后端共享同一个 `TrainEngine` 抽象。

### 2.4 `RolloutController.initialize`：推理端怎么起

推理端多了一层「先起推理服务器、再起引擎客户端」（<a href="#ref-rolloutctl">[5]</a>）：

1. **按实例粒度算资源**。`instance_size = tp_size × pp_size`，一个 worker 吃 `instance_size` 张卡、`instance_size` 份 cpu/mem。`replicas = dp_size`。也就是**推理侧的 worker 是「一个推理实例」而不是「一张卡」**——这和训练侧的一卡一进程刚好相反。
2. **分三条路起服务器**：
   - 已有 `server_infos`（评测复用主 rollout 的服务器）；
   - `scheduling_strategy.type == colocation`（AWEX 路径，多个服务器共享节点，必须显式算 `base_gpu_id`）；
   - 否则是分离路径，SLURM 已经按 worker 隔离了 `CUDA_VISIBLE_DEVICES`，用 `launch_server` 集体 RPC 起服务器。
3. **再 `initialize` 引擎客户端**，把 `addr` 传给 worker，worker 侧的 `RemoteInfEngine` 拿到地址后当 HTTP 客户端用。
4. **建 staleness manager 与 dispatcher**（这一步才是「样本队列」的出生点，见 §4）。
5. **起 callback 服务器**。`_start_callback_server` 起一个 Flask 应用监听 `find_free_ports(1)`，暴露 `/callback/rollout_complete`、`/callback/update_weights_xccl`、`/callback/pause_generation` 等端点。

第 5 步是理解 AReaL 通信模型的关键：**worker 完成一条 rollout 不返回到调用方，而是回调控制器的 HTTP 端点**。`submit` 返回的不是结果而是 `task_id`，结果异步从回调进入 `_pending_futures[task_id]`。

### 2.5 2.x 的服务化初始化

2.x 把上面的每条链路拆成独立服务进程，`RolloutControllerV2` 的注释直接写明它「是通过 gateway HTTP 栈（Gateway → Router → Data Proxy → 推理后端）路由推理与 pause/continue 流量；所有服务器都作为 worker 进程由 scheduler 拉起；推理服务器进程通过 RPCGuard（一个轻量进程管理器）fork 出来」<a href="#ref-rcv2">[6]</a>。

服务拓扑因此变成：

```text
                    ┌─────────────┐
   训练侧            │ Gateway     │ ← admin_api_key + router_addr
   (TrainController) │ (thin proxy)│
        │            └──────┬──────┘
        │                   │ 只问路由，不代理流量
        │            ┌──────▼──────┐
        │            │ Router      │ 持有 worker 健康 / session→worker 映射
        │            └──────┬──────┘
        │                   │
        │            ┌──────▼──────┐
        │            │ Data Proxy  │ session / tokenize / pause
        │            └──────┬──────┘
        │                   │
        │            ┌──────▼──────┐
        └───────────>│ SGLang/vLLM │
                     └─────────────┘
```

Gateway 的注释特意说明「gateway 只持有 `admin_api_key` 和 `router_addr`，所有 worker 状态、session pinning 与路由策略都活在 Router 服务里」<a href="#ref-gw2">[7]</a>；Router 则「从不代理流量，只回答路由查询」<a href="#ref-router2">[8]</a>。这种职责切分让 Router 可以无状态地水平扩，而 Gateway 保持薄到可以随便重启。

---

## 3. 进程与角色：控制面、数据面与 RPC 通道

### 3.1 两套栈并存

读 2.1.0 必须先立一张对照表，否则会以为自己在同一层里跳：

| 概念 | 1.x 主干（`areal/infra/`） | 2.x 服务（`areal/v2/`） |
| --- | --- | --- |
| 训练控制器 | `infra/controller/train_controller.py` → `TrainController` | `v2/training_service/controller/controller.py` → `GatewayTrainController` |
| 推理控制器 | `infra/controller/rollout_controller.py` → `RolloutController` | `v2/inference_service/controller/controller.py` → `RolloutControllerV2` |
| 引擎抽象 | `areal/api/engine_api.py` 的 `TrainEngine` / `InferenceEngine` ABC | 不变（服务层照样用同一套 ABC） |
| 控制通道 | Flask RPC（`/call`、`/create_engine`、`/set_env`） | FastAPI Gateway + Router（HTTP 转发） |
| 权重更新 | `TrainEngine.update_weights(meta)` 直连推理服务器 | `v2/weight_update` 独立服务（`/connect`、`/update_weights`） |

`RolloutControllerV2` 的 docstring 明确说它是 `RolloutController` 的**并行实现**（`parallel implementation`，NOT a subclass），但「duck-type 兼容：训练器可以用任意一个而不改代码」<a href="#ref-rcv2">[6]</a>。所以两套不是迁移中间态，而是**两个可选的部署形态**：单机调试走 1.x，生产服务化走 2.x。

### 3.2 角色清单

| 角色 | 进程 | 持有权重？ | 控制面还是数据面 |
| --- | --- | --- | --- |
| `PPOTrainer` 主进程 | 1 | 否 | **控制面** |
| `actor/<rank>` worker | `world_size` 个（训练侧） | **是**（参数 + 优化器） | 数据面（算力） |
| `ref/<rank>`、`critic/<rank>`、`teacher/<rank>` | 按需 | 是（推理态） | 数据面 |
| `rollout/<rank>` worker | `rollout dp_size` 个 | 否（只有 HTTP 客户端） | 数据面（产物） |
| SGLang/vLLM 服务器进程 | 每个 rollout worker 起一组 | **是**（推理态权重） | 数据面（生成） |
| `data/<n>` worker | `num_workers` 个 | 否 | 数据面（读数据集） |
| `proxy-rollout/<rank>` | 按需 fork | 否 | 数据面（OpenAI 兼容层） |

关键结论有两条：

- **权重有两份、分居两个进程组**。训练侧的那份是 fp32 master + 优化器状态；推理侧那份是 bf16 计算权重。两者之间只能靠 refit 通信（§7.2），没有任何共享内存路径。
- **控制面不持有权重，也不做计算**。它只做三件事：发任务、收结果、记账（版本号、统计量）。

### 3.3 RPC 通道怎么建

1.x 的 RPC 不是 gRPC 也不是 Ray actor call，而是**每个 worker 起一个 Flask 应用**（`areal/infra/rpc/rpc_server.py`）。这个应用由三个 blueprint 拼成（<a href="#ref-rpcserver">[9]</a>）：

| Blueprint | 端点 | 语义 |
| --- | --- | --- |
| `data_bp` | `PUT/GET /data/<shard_id>`、`POST /data/batch`、`DELETE /data/clear` | **张量分片存储**，按 shard_id 存取 |
| `engine_bp` | `POST /set_env`、`/create_engine`、`/call` | **引擎生命周期与任意方法调用** |
| （Guard 主应用） | `/health`、`/configure`、`/kill_forked_worker` | 健康检查、播种、进程管理 |

最要紧的设计是 **engine 单线程串行执行**。`engine_blueprint.py` 起了唯一一个 `EngineWorker` 守护线程，所有 `/call` 请求先进 `Queue`，由该线程逐个执行并 `future.result()` 阻塞等待：

```text
POST /call {method, engine_name, args, kwargs}
      │
      ├─ deserialize_value(args/kwargs)
      ├─ RTensor.localize(...)            ← 把 meta tensor 换成真数据（§3.4）
      └─ _submit_to_engine_thread(...)
             │
             ├─ 可选：broadcast_tensor_container(src_rank=DP head)
             ├─ perf_tracer.trace_scope(f"rpc.{method}")
             └─ getattr(engine, method)(*args, **kwargs)
```

为什么必须串行？源码注释给的答案是「这是 NCCL 兼容性要求」<a href="#ref-enginebp">[10]</a>。NCCL 的集合通信要求同一条流上的操作顺序确定，而 Flask 默认多线程会打乱顺序。把引擎侧做成单线程 + 队列，等于**用「一个 worker 进程内只有一条执行序」换掉了 NCCL 的顺序保证问题**。

### 3.4 `RTensor`：控制面不搬运张量

这是 AReaL 工程上最省事的一处设计。控制器与 worker 之间的 RPC payload 里，**张量不是真的张量，而是 `TensorShardInfo(shard_id, node_addr)` 形式的指针**（`RTensor` 的 `data` 字段是 `meta` device 上的空壳）<a href="#ref-rtensor">[11]</a>：

- **写侧**（控制器）：`RTensor.remotize(obj, node_addr)` 把张量 detach→cpu→`store()` 上传到本机 `/data/<shard_id>`，返回带 `meta` 占位的 `RTensor`。顺带做一件事：对带 `attention_mask` 的轨迹字典先 `split_and_unpad_tensor` 裁掉尾部 padding，「让每个 RTensor 更紧凑」。
- **读侧**（worker）：`RTensor.localize(args)` 把每个 `meta` tensor 换回真数据，走 `HttpRTensorBackend.fetch()`；同一 node_addr 的 shard 合并成 `POST /data/batch` 批量取，每批最多 32 个；带 3 次重试。
- **客户端侧还有一层 fetch buffer**：同名 shard 重复取会命中本地缓存。清理靠训练循环末尾的 `clear_batches`，且要**扇出到每一个 localize 过的角色**——源码注释直接点出了 issue 号（#1209）：一次 DELETE 只能清掉存储 owner 的 `_storage`，清不掉各消费者进程的 `_fetch_buffer`。

这套设计的收益很直观：控制器可以只发一个大 batch 的「元数据」，让所有 8 个 DP worker 各自并行走 HTTP 拉自己那一份，控制器的网络和内存都不成瓶颈。代价是**生命周期管理变复杂**（谁负责删 shard、什么时候删），这也是 `train.clear_batches` 这一步在训练循环里独占一个 trace scope 的原因。

---

## 4. 样本队列与 staleness 控制

这一节是全文的技术核心。BRIEF 的四级定义说全异步的判据是「训练端与推理端是两个独立服务，用样本队列连接，staleness 由队列深度控制」。AReaL 的代码把这个「队列」拆成了**三层**，每层解决一个不同的问题。

### 4.1 三层结构

```text
   数据源          ① pending_inputs（无界 deque）        ② AsyncTaskRunner 输入队列（有界）
  dataloader ──> [ _RemoteRolloutTaskInput, ... ] ──> [ task_fn, ... ] ──> 推理服务器
   (cycle)         │ 准入由 StalenessManager 决定         │ 上限 = queue_size
                   │ submit_task_input() 无阻塞入队       │
                   ▼                                      ▼
              生产线程 _commit_loop                 异步协程池执行
                   ▲                                      │
                   │ 有新容量就 notify                      ▼
   训练端 <──── ③ pending_results（dict[task_id]）<──── 消费线程 _fetch_loop
   取 B 条         wait_results(count=B)
```

三层各自的语义：

| 层 | 数据结构 | 有没有界 | 谁管它 |
| --- | --- | --- | --- |
| ① 待提交队列 `_pending_inputs` | `collections.deque` | **无界** | 但入队被 `get_pending_limit()=(η+1)B` 间接约束 |
| ② 执行队列 `runner.input_queue` | `queue.Queue(maxsize=qsize)` | **有界**，`qsize = queue_size or max_concurrent_rollouts × 16` | 满了抛 `TaskQueueFullError`，任务塞回队头 |
| ③ 结果队列 `_pending_results` | `dict[int, TimedResult]` | 无显式上限，但受「已接受数 ≤ (η+v+1)B」约束 | 训练端 `wait_results(count)` 取走 |

### 4.2 `StalenessManager`：准入公式

整个 staleness 控制浓缩在 `get_capacity()` 里（<a href="#ref-stale">[12]</a>）：

```python
def get_capacity(self) -> int:
    with self.lock:
        current_version = self.version_provider.get_version()
        max_concurrent_rollouts = max(1, self.max_concurrent_rollouts)
        concurrency_capacity = max_concurrent_rollouts - self.rollout_stat.running

        ofp = self.max_staleness                        # 即 η
        sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
        consumer_bs = max(1, self.consumer_batch_size)  # 即 B
        staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

        capacity = min(concurrency_capacity, staleness_capacity)
        return capacity
```

把它写成公式：

$$
\text{capacity} = \min\Bigl(\underbrace{C_{\max} - N_{\rm running}}_{\text{并发槽}},\;\; \underbrace{(\eta + v + 1)\,B - (N_{\rm accepted} + N_{\rm running})}_{\text{staleness 槽}}\Bigr)
$$

其中 $$v$$ 是推理端当前权重版本，$$B$$ 是 `consumer_batch_size`（训练端一次消费多少条），$$\eta$$ 是 `max_head_offpolicyness`。

与论文公式 (3) 的对应关系值得逐项对一遍。论文写的是：

$$
\left\lfloor \frac{N_r - 1}{B} \right\rfloor \le i + \eta
$$

其中 $$N_r$$ 是已生成轨迹总数，$$i$$ 是当前策略版本，$$\eta$$ 是最大允许 staleness。改写不等式得 $$N_r \le (\eta + i + 1)B$$ ——**与代码里 `staleness_capacity` 的右侧完全同形**。差别只在代码把 $$N_r$$ 拆成 `accepted + running`（已落地 + 在途），把论文的整数除法换成乘法避免取整误差。

### 4.3 超限后怎么处理：既不是丢弃也不是降权

这是 BRIEF 明确点名要回答的问题。答案是**第三种：根本不让它发生**。

- `get_capacity() <= 0` 时，`_get_next_task_for_submission()` 拿不到任务，生产线程在 `_input_cv.wait()` 上挂起；
- 训练端消费一条、`_fetch_loop` 里 `_input_cv.notify()`，容量释放，生产线程被唤醒；
- 结果就是**rollout 提交速率自动跟随训练消费速率**——一个纯反馈控制系统，不需要显式的丢弃或降权策略。

为什么这个选择重要？因为丢弃和降权都在**出口**处理，意味着算力已经花掉了；限流在**入口**处理，算力根本没花。对一个推理成本远高于训练成本的系统（论文口径下推理占 3/4 的卡），入口限流的收益是数量级的。

副作用也很清楚：**队列会在长尾上卡住**。论文自己承认「当 $$\eta$$ 太小时，正在生成某些极长轨迹时生成吞吐会被拖慢；因此经验上建议采用较大的 $$\eta$$」`[论文]`。这等于说：**限流参数必须松到让长尾也能跑完，否则吞吐崩**。

### 4.4 两个边界处理

参数交换里还有两处容易忽略的实现细节，都是 bug 驱动出来的：

**(a) 并发上限与队列深度的分工。**`get_pending_limit()` 返回 `(η+1)B`，这是**待提交队列**的软上限，用在 `active_submit_and_wait` 里：

```python
cap_staleness = self.staleness_manager.get_pending_limit() - pending_inputs
cap_queue = self.runner.max_queue_size - (self.runner.get_input_queue_size() + batch_size)
capacity = min(cap_staleness, cap_queue)
```

两者语义不同：`cap_staleness` 保证「入队的样本在被消费时不会超过 η 步」，`cap_queue` 保证「不会把执行队列撑爆」。而 `_has_runner_capacity()` 用的是**第三个**判据 `get_capacity() > 0`，它同时看并发槽与 staleness 槽。**三个上限、三处判定**，这是读这段代码最容易迷路的地方。

**(b) checkpoint 恢复时的版本跳变。**`on_version_recovered(version)` 的存在理由写在 docstring 里：从 checkpoint 恢复时版本号从 0 跳到恢复值，如果不调整 `accepted`，容量公式会给出 `(η+version+1)B` 而不是意图中的 `(η+1)B`，「导致一波提交和无界 staleness 增长」<a href="#ref-stale">[12]</a>。修法是把 `accepted` 直接对齐到 `version × consumer_bs`——一个诚实的「假装这些样本都已经消费过了」。

### 4.5 版本号是怎么传的

`v` 的来源链值得单独画一下，因为它决定了反馈环的延迟：

```text
训练循环 global_step
   └─ new_version = global_step + 1
        ├─ weight_update_meta.with_version(new_version)
        ├─ actor.update_weights(versioned_meta)          ← refit 真的搬权重
        ├─ actor.set_version(new_version)                ← 训练侧记账
        ├─ critic.set_version(new_version)
        └─ rollout.set_version(new_version)              ← 控制面记账 + 扇出
              └─ _collective_rpc("set_version")  → 每个 rollout worker
                    └─ RemoteInfEngine.set_version() → self._version = v
```

注意 `RolloutController.set_version` 除了改本地 `self._version`，还 `_collective_rpc("set_version")` 扇出到所有 rollout worker，并且**如果 proxy 也启动了，还要再扇出一次**。所以 `version_provider.get_version()` 读到的是控制面自己那份副本，而这个副本**无条件假设扇出一定成功**——`get_capacity()` 里没有「权重是否真的到位」的校验。这一点在 §7.2 会作为权重层的一个风险点重新提。

---

## 5. 可中断 rollout：与 slime partial rollout 的对照

### 5.1 AReaL 的做法：中断 → 丢 KV → 前缀当 prompt 重算

`RemoteInfEngine.agenerate` 的主循环是理解「可中断」的全部（<a href="#ref-agenerate">[13]</a>）：

```python
while (stop_reason not in ["stop", "tool_calls", "length"]
       and len(accumulated_output_tokens) < ori_max_new_tokens):
    # 1. 若被暂停，先等（0.5s 轮询）
    while self.workflow_executor.is_paused():
        await asyncio.sleep(0.5)

    # 2. 钉住本段使用的版本
    request_version = self.get_version()

    # 3. 用当前 req.input_ids 发请求
    http_req = self.backend.build_generation_request(req, with_lora=..., version=request_version)
    result = await arequest_with_retry(...)

    # 4. 累积 tokens / logprobs / versions
    accumulated_output_tokens.extend(gen_result.output_tokens)
    accumulated_output_logprobs.extend(gen_result.output_logprobs)
    accumulated_versions.extend([request_version] * len(gen_result.output_tokens))

    # 5. 把已生成部分拼进 prompt，扣减剩余预算
    req.input_ids += gen_result.output_tokens
    req.gconfig.max_new_tokens -= len(gen_result.output_tokens)
```

关键在**第 5 步**：下一次迭代发给推理服务器的是 `req.input_ids`，也就是「原始 prompt + 这一轮已经生成的全部 token」。这意味着中断后**旧的 KV cache 被丢弃**（请求变了，prefix 不匹配），服务器要重新 prefill 整个前缀。

为什么愿意付这个代价？因为**重算保证了一条轨迹的每个 token 都由它标注的版本真正生成**。`accumulated_versions` 逐 token 记录版本，训练时 `versions` 张量进入 loss（§6.4）。

中断的触发点有两个层次，且它们不等价：

| 层次 | 方法 | 作用 | 代码位置 |
| --- | --- | --- | --- |
| **控制面调度层** | `RolloutController.pause()` / `resume()` | 停生产、停提交，**不影响在途请求** | `rollout_controller.py` |
| **引擎请求层** | `RemoteInfEngine.pause_generation()` / `continue_generation()` | 向推理服务器发 `/pause_generation`，**中止在途请求** | `remote_inf_engine.py` |

SGLang 后端的 `get_pause_requests()` 返回**两个**请求，注释解释了为什么必须两个（<a href="#ref-sglangpause">[14]</a>）：

1. 第一个是默认 mode 的 `/pause_generation`：中止在途请求、返回 partial output。「客户端通过扩展 prompt 来续跑它们」；同时把 scheduler 完全闲置，SGLang 释放显存前要求这个状态。
2. 第二个是 `mode="in_place"`：注释说第二个「看起来冗余」，但 abort 路径根本到不了 scheduler——`TokenizerManager` 只对非 abort mode 转发请求，其他情况直接走 `abort_request()` drain。**abort 抬高的是 tokenizer 自己的闸门，scheduler 还在调度**。只有 in-place pause 才抬高 scheduler 的 paused flag。

这段注释是整份源码里信息密度最高的地方之一：它说明**「中断」在 SGLang 里是两套独立机制的组合**（tokenizer 闸门 + scheduler 闸门），AReaL 必须两个都调才能既中止在途、又释放显存。

另外 `pause_generation()` 末尾还会 `time.sleep(config.pause_grace_period)`，注释是「上面的 http 请求可能需要一些时间才被调度执行；这一行等到所有请求真的被丢弃」。`pause_grace_period` 默认 **0.0**，也就是说默认情况下 AReaL **不等**——中断是「尽力而为」的，靠这个超参兜底。

### 5.2 slime 的做法：中断 → 回收 → 下一轮续跑

slime 的 partial rollout 走的是另一条路（基准 `v0.4.0` = `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`）。参数说明写得很直白：

> `--partial-rollout`：是否使用 partial rollout。若设置，动态采样期间**未完成的样本会被回收进 data buffer**。对长响应有用。
>
> `--mask-offpolicy-in-partial-rollout`：是否在 partial rollout 中 mask 掉之前的生成。若设置，**只有 on-policy 生成的 token 会用于训练**。
>
> <a href="#ref-slimeargs">[15]</a>

中断发生在 `generate_rollout_async` 的 drain 阶段（<a href="#ref-slimedrain">[16]</a>）：SGLang 服务器被 `abort_servers_until_idle` 中止到「确认空闲」，然后遍历已完成的 task，凡是含「ABORTED 且 response_length > 0」样本的 group 就整组回收进 `aborted_samples`，并给样本打上 `start_rollout_id` 标记。

续跑时在 `generate_and_rm` 开头：

```python
# mask previous off-policy generation for partial rollout
if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
    sample.loss_mask = [0] * sample.response_length
```

**把之前生成的那段 token 的 `loss_mask` 全部置零**，然后从这个已有 response 继续生成。

### 5.3 两者的差别（本文最重要的一组对照）

| 维度 | AReaL 可中断 rollout | slime partial rollout |
| --- | --- | --- |
| **中断触发** | 权重更新前主动 `pause_generation()`，随训练步节奏 | roll_ut 结束/动态采样阶段 drain，随 rollout 轮次节奏 |
| **中断后旧 token 怎么办** | **保留**，拼进 prompt 当上下文；丢 KV 重算 | **保留**在 `sample.response` 里继续生成；可选 `loss_mask=0` |
| **KV cache** | 丢弃（前缀变了）→ 重算整个前缀 | 保留（同一次生成会话）→ 续跑 |
| **旧 token 是否进 loss** | **进**。版本记进 `versions` 张量，由 decoupled PPO 承担 off-policy | 默认进；开 `mask_offpolicy_in_partial_rollout` 则**不进** |
| **off-policy 负担落在哪** | **算法**：重要性权重 + 逐 token 版本 | **数据**：mask 掉跨版本 token，或显式接受 |
| **代价** | prefill 重算（算力） | 部分样本的 token 被浪费（样本效率） |
| **谁在跨步** | 一条轨迹的多个**版本**跨步；轨迹本身不跨步 | 一条轨迹跨**rollout_id**；版本跟着轮次走 |
| **staleness 记录粒度** | **token 级**（`output_versions`） | **样本级**（`start_rollout_id` metadata） |
| **能否一条轨迹多版本混训** | 可以，`versions` 直接喂 loss | 不能，只能全留或全 mask |

一句话概括：**AReaL 选择「让一条轨迹横跨多个策略版本，然后用算法消化」，slime 选择「让一条轨迹在同一轮内完成，跨轮的部分要么丢要么不算」**。

这个差别不是风格问题，而是**算法-系统联合设计**与**系统优化**的分野。论文自己有一句判词：「我们的可中断生成技术概念上类似同步 RL 系统中的 partial rollout。但 AReaL 不是设一个固定的长度预算，而是动态中断生成，同时通过缓冲维持一致的训练 batch 大小，从而保持 PPO 的稳定性」`[论文]`。

「通过缓冲维持一致 batch 大小」指的是 `active_submit_and_wait` 保证返回**恰好 `batch_size` 条结果**（`dynamic_bs=False` 时），即使中间有大量 rejected。这一条在 slime 里对应 `dynamic_sampling` + `over_sampling_batch_size` 的机制，是不同形状的同一个目标。

### 5.4 版本逐 token 记账

`versions` 张量的构造在 workflow 层（<a href="#ref-rlvr">[17]</a>）：

```python
versions = [-1] * resp.input_len + resp.output_versions
```

prompt 部分记 **-1**（不是策略生成的，无版本），生成部分逐 token 记版本。这个约定贯穿全链路：

- 训练时 `_log_version_staleness_stats` 用 `versions >= 0` 过滤出真正的生成 token；
- `sample_staleness_proximal = (current_version - 1) - v_behave`，`sample_staleness_theta = current_version - v_behave`；
- 记录 avg / max / min 六个数。

**这是全异步框架里唯一能把「我这个 batch 有多 off-policy」量化出来的机制**。没有逐 token 版本，$$\eta$$ 只是个配置数字，你永远不知道自己实际跑在多少 staleness 上。

### 5.5 中断的代价：论文的消融

`[论文]` 的图 6(b) 消融了可中断生成：「没有可中断生成时，控制器必须等最长的响应。可中断生成在 4 节点上对 1.5B 和 7B 模型分别带来 **12%** 和 **17%** 的吞吐提升」。这两个数字是「中断 + 重算」这一整套的净收益——**已经扣掉了重算 prefill 的成本**。

注意 12%/17% 的量级：它远小于端到端的 2.77×。这提示一个判断——**全异步的主要收益来自阶段解耦（§4），可中断只是把长尾这一块补上**。把两者混为一谈是读 AReaL 论文时最常见的误读。

---

## 6. off-policy 修正：decoupled PPO 与 token 过滤

### 6.1 论文的目标函数

论文式 (4)(5) 的 decoupled PPO（<a href="#ref-paper">[1]</a>）是：

$$
J(\theta) = \mathbb{E}_{q \sim \mathcal{D},\, a_t \sim \pi_{\rm behav}}\left[\sum_{t=1}^{H} \frac{\pi_{\rm prox}}{\pi_{\rm behav}} \min\Bigl(u_t^{\rm prox}(\theta)\hat{A}_t,\; \mathrm{clip}\bigl(u_t^{\rm prox}(\theta), 1-\epsilon, 1+\epsilon\bigr)\hat{A}_t\Bigr)\right]
$$

其中

$$
u_t^{\rm prox}(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\rm prox}(a_t \mid s_t)}
$$

三个策略的分工是：

- $$\pi_{\rm behav}$$：**实际生成数据的策略**（rollout 时推理服务器上的权重），可能很旧；
- $$\pi_{\rm prox}$$：**近端策略**，一个参考点，用来定义信任域的中心。代码里取 `current_version - 1`，即「上一步训练完的策略」；
- $$\pi_\theta$$：**当前正在更新的策略**。

论文对这个设计的解释是：如果直接用 $$\pi_{\rm behav}$$ 作裁剪中心，「近端策略会把最新策略 $$\pi_\theta$$ 往旧版本、低质量的策略上拉，拖慢模型改进」。改用一个近期的高质量策略作中心，「模型更新发生在高质量 $$\pi_{\rm prox}$$ 周围的信任域内，从而稳定训练」`[论文]`。

还有一个关键性质：「decoupled PPO 目标天然放宽了“一个训练 batch 内所有数据应由单一策略生成”的要求。这个性质对在可中断生成与策略更新结合时维持算法正确性至关重要」`[论文]`。**这正是 §5 敢让一条轨迹跨多个版本的算法依据**。

### 6.2 代码里怎么落：`_resolve_proximal_logp`

`grpo_loss_fn` 拿到 `logprobs`（当前策略前向）、`old_logp`（行为策略，来自推理服务器）、`prox_logp`（近端，可能来自额外前向）三个量，然后由 `_resolve_proximal_logp` 决定近端策略的 logprob 从哪来（<a href="#ref-resolveprox">[18]</a>）：

| `prox_logp_method` | 近端 logprob 的来源 | 代价 |
| --- | --- | --- |
| `recompute`（默认） | 训练前用当前权重额外跑一次前向，得 ground truth | 一次额外 forward |
| `loglinear` | 用 `versions` 在 behavior 与 current 之间**插值近似** | 免费，但有近似误差 |
| `metrics` | 训练用 ground truth，同时算近似值仅作对照指标 | 一次 forward + 指标开销 |

函数里还有两道防御：`prox_logp` 为 None 而当前方法不允许跳过前向时直接 `ValueError`；算完检查 NaN/Inf，命中直接 `RuntimeError`。这类「宁可炸也不带毒训练」的写法在有 off-policy 的系统里很必要——**一个 NaN 的 prox_logp 会静默污染整批梯度**。

注意这个选择带来的一个隐含约束：`recompute` 方法下**近端策略的版本必须就是「当前训练引擎里的权重」**。这要求 `update_weights` 之后、下一次 `compute_logp` 之前，训练侧与推理侧的版本记账必须一致——落到 §7.2 的权重层问题。

### 6.3 `rejection_sampling`：第二个修正层

除了 decoupled 目标，AReaL 还有一层**逐 token 的离群过滤**，配置类是 `RejectionSamplingConfig`（<a href="#ref-rsconfig">[19]</a>），实现是 `apply_rejection_sampling`（<a href="#ref-rs">[20]</a>）。它有两个正交的维度：

**维度一：度量（`metric`）**

| `metric` | 定义 | 语义 |
| --- | --- | --- |
| `ratio` | $$\pi_{\rm prox} / \pi_{\rm behav}$$ | 直接的重要性比 |
| `kl_k1` | $$\log r$$ | 前向 KL 无偏估计（**可负**） |
| `kl_k2` | $$\tfrac{1}{2}(\log r)^2$$ | 非负二次近似 |
| `kl_k3` | $$r - \log r - 1$$ | 非负精确前向 KL 估计 |
| `binary_kl` | $$\max\bigl(\mathrm{KL}(\pi_{\rm prox}\|\pi_{\rm behav}), \mathrm{KL}(\pi_{\rm behav}\|\pi_{\rm prox})\bigr)$$ | KPop：双向二值 KL |

（`r` 即重要性比，代码里先算 `log_ratio = prox - behav` 再按需 `exp`。）

**维度二：动作（`action`）**

| `action` | 行为 | 效果 |
| --- | --- | --- |
| `mask`（默认） | 超界的 token/序列 `loss_mask` 置零 | 拒绝：完全排除出梯度 |
| `clamp` | 重要性权重 clamp 到 `[lower, upper]` | 截断：token 仍参与梯度，但权重有界 |

**两个维度的粒度**都由 `level`（`token` / `sequence`）控制。`level='sequence'` 且 `metric='ratio'` 时，过滤决策与修正权重**都用序列级几何平均**——注释说明这是「为避免长度陷阱、匹配 GSPO 语义」；用算数平均会让长序列的重要性比被系统性放大。

实现里有几处正确性细节值得单独点出：

1. **fp32 上采样再相减**。`log_ratio = proximal.detach().float() - old.detach().float()`，注释写「避免 bf16/fp16 的精度损失」。在 bf16 下两个接近的 logprob 相减会损失有效位，重要性比的误差会被 `exp` 放大。
2. **非有限值消毒**。`log_ratio = torch.where(torch.isfinite(log_ratio), log_ratio, 0.0)`，注释点名了 `-inf - (-inf) = NaN` 这个具体场景。
3. **分母用原始 mask 计数**。`ppo_actor_loss_fn` 在应用 rejection 之前先存 `loss_mask_count = loss_mask.count_nonzero() or 1`，注释解释：`loss_weight_fn` 永远用原始 mask，如果不这样做，`mask` 模式会把每 token 梯度放大 `N_original / N_kept` 倍。

第 3 点是一个**极易踩的坑**：过滤样本会让剩下的 token 梯度被放大，而放大倍数恰好等于被过滤比例。在大 staleness 下这个比例可能很高，梯度会爆。

### 6.4 三个修正层叠加后的实际数据流

把 §6.1–§6.3 串起来，一次 actor 更新的数据流是：

```text
rollout 产物                      训练端加工                         loss
──────────                        ──────────                        ─────
logprobs ──────┐                  prox_logp（额外前向）──┐
versions ────┐ │                                          │
             │ └─> old_logp = π_behav 的 logprob ─────────┼──> log_ratio = prox - behav
             │                                            │        │
             │                                            │        ├─> metric → 与 upper/lower 比
             │                                            │        │      ├─ mask: loss_mask 置零
             │                                            │        │      └─ clamp: 权重夹紧
             │                                            │        │
             └────────────────────────────────────────────┴──> ratio = exp(logprobs - prox)
                                                                     │
                                                          pg_loss = -A·ratio 与 -A·clip(ratio) 取 max
                                                                     │
                                                          pg_loss *= behave_imp_weight   ← 行为重要性权重
```

注意 `behave_imp_weight` 是**乘在 PPO 裁剪目标外面的**，不是替代 ratio。也就是说 AReaL 是「decoupled PPO 的裁剪 + 行为重要性的外层加权」的组合，而不是单一的 TIS 截断。源码 docstring 对两者关系的表述很明确：`importance_sampling_level` 控制 $$\pi_\theta/\pi_{\rm prox}$$ 的聚合粒度（GSPO），与 `rejection_sampling.level` 控制 $$\pi_{\rm prox}/\pi_{\rm behav}$$ 的过滤粒度**是正交的**<a href="#ref-ppoactorloss">[21]</a>。

### 6.5 `use_decoupled_loss=False` 时的退化

配置项 `use_decoupled_loss` 关掉时（<a href="#ref-advantage">[22]</a>）：

```python
if not self.config.use_decoupled_loss and self.config.recompute_logprob:
    # 用训练侧重算的 logp 覆盖推理侧产的 logp
    old_logp = data["logprobs"] = prox_logp_value
else:
    old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
    if not self.config.use_decoupled_loss:
        data["prox_logp"] = old_logp   # 近端 = 行为
```

关掉之后 `prox_logp = old_logp`，两个策略被强行捏成同一个，decoupled 目标退化成标准 PPO。**这时 `versions` 仍然被记录，但不再参与 loss**——只剩监控价值。这解释了为什么 §8 的归因层能独立于算法开关工作。

---

## 7. 训推一致三层

按 BRIEF §3 的三层拆。AReaL 在三层上的着力点很不均匀：**权重层做得最重，数值层基本靠交给推理引擎，样本层是它的立身之本**。

### 7.1 数值层：交付给推理引擎，AReaL 只做两件事

AReaL 不在数值层做任何 kernel 级对齐——logprob 由 SGLang / vLLM 自己的 `/generate` 返回（`return_logprob: True`）。它能做的是：

1. **采样参数原样传递**。`SGLangBackend.build_generation_request` 把 `top_p` / `top_k` / `temperature` / `stop_token_ids` / `ignore_eos` / `frequency_penalty` 逐项塞进 `sampling_params`，`greedy` 时 `temperature` 硬置 0.0。**没有重新定义采样顺序**，所以数值层的正确性责任在引擎（见 serving 系列）。
2. **`logprobs` 与 `versions` 同长同序**。`parse_generation_response` 从 `meta_info["output_token_logprobs"]` 拆成两个平行的 list，长度不一致会导致 `versions` 错位——这种错位不会报错，只会静默让 staleness 归因失效。

换句话说：**AReaL 的数值层不是一个设计，而是一个委托**。这也是它能同时接 SGLang 与 vLLM 两个后端而不用维护两套 kernel 的原因。

### 7.2 权重层：refit 的三个契约

权重层是 AReaL 最重的地方。三条路径并存：

| 路径 | 机制 | 适用 |
| --- | --- | --- |
| **xccl**（默认） | 训练侧建 NCCL 组，逐参数 `DTensor.full_tensor()` → bf16 → 分桶 broadcast 到推理服务器 | 同机/同网，主流 |
| **disk** | 训练侧存 checkpoint，推理侧 reload | 跨集群、LoRA（SGLang XCCL 不支持 LoRA，直接 `ValueError`） |
| **awex**（2.x） | 权重增量同步（roadmap 里 `Weight delta update with awex` 仍标注为 planned） | colocate / 增量 |

**契约一：广播前必须停生成。**`_update_weights_from_distributed` 的第一件事是 `if main_rank: self.rollout_engine.pause_generation()`，最后一件事是 `continue_generation()`，中间用 `dist.barrier(group=self.cpu_group)` 夹住（<a href="#ref-fsdpupd">[23]</a>）。这段代码的注释解释了桶式流水线：**只有 rank 0 广播**，其他 rank 只贡献 `full_tensor()` 这个集合通信；桶超限时先 `_wait_pending_weight_update_bucket` 再发下一桶，单 pending 桶的设计。

注意 `abort_all_requests: True` 也直接写在 SGLang 的请求 payload 里（`build_distributed_weight_update_requests`）。也就是说 **refit 本身就是一个中断事件**，不只是推一个版本号。这是 §5 的中断与本节 refit 的连接点：**中断的动机就是权重层的正确性**。

**契约二：版本号必须与权重同时到达。**训练循环里的顺序是「先 `update_weights` 再 `set_version`」：

```python
new_version = global_step + 1
versioned_meta = self.weight_update_meta.with_version(new_version)
self.actor.update_weights(versioned_meta)   # 真的搬权重
self.actor.set_version(new_version)         # 训练侧记账
self.rollout.set_version(new_version)       # 推理侧记账 + 扇出
```

这个顺序是对的，但**中间没有校验**：`update_weights` 返回不代表推理服务器的权重真的全到位了（`areal_update_weights` 在 vLLM 侧返回 `(bool, str)`，失败时只 `logger.error`）。而在 `_update_weights_from_distributed` 末尾，`current_platform.synchronize()` + `dist.barrier()` 只保证**训练侧**的 collective 完成。所以权重层的真实保证是「**尽力而为的一致**」，兜底靠 `pause_grace_period` 与下一次 refit。

**契约三：LoRA 走版本化 adapter，不走覆盖。**这是 AReaL 权重层最有意思的一处设计。LoRA 模式下不做「把权重刷进去」，而是给每个版本注册一个**独立的 adapter**：

```python
lora_name = get_versioned_lora_name(meta.lora_name, meta.version)   # "xxx-v3"
requests = [
    HttpRequest(endpoint="/load_lora_adapter",
                payload={"lora_name": lora_name, "lora_path": str(meta.path)},
                best_effort=True),
    # 卸载掉掉出保留窗口的那个版本
    HttpRequest(endpoint="/unload_lora_adapter",
                payload={"lora_name": get_versioned_lora_name(meta.lora_name, meta.version - keep)},
                best_effort=True),
]
```

而生成请求按版本选 adapter：`payload["lora_path"] = get_versioned_lora_name(lora_name, version)`（<a href="#ref-sglanggen">[24]</a>）。注释解释了保留窗口的含义：「保留的版本覆盖 off-policy rollout（`max_head_offpolicyness`）。不卸载的话 SGLang 会积累每步一个 adapter，泄漏显存并最终挂住」。

**这是全异步下权重层唯一能做到「真正多版本共存」的实现**：因为 adapter 按版本号隔离，一条轨迹的某个片段用 `-v3` 生成、下一片段用 `-v5`，服务器上两者同时在线。全量权重没有这个能力（一份权重只有一个版本），只能靠「中断 + 重算」把老版本的数据用新版本重新前向。

3.x / 2.x 的 `weight_update` 服务把这条链路做成独立服务，接口层面是 `/connect`（注册 train/inference worker 对）→ `/update_weights {pair_name, version}` → `/disconnect`，另有 `/weight_meta/*` 一组 KV 接口给 delta 同步用（<a href="#ref-wugw">[25]</a>）。`/update_weights` 分三种模式：`colocate` / `disk` / `awex`，返回 `WeightUpdateResult(status, version, duration_ms)`——**把 refit 的耗时也变成了一等指标**。

### 7.3 样本层：这是 AReaL 的全部

样本层是 AReaL 的立身之本，也是它与其他框架差别最大的地方。三个机制组合：

1. **队列深度即 staleness 上限**（§4.2）。这是**事前**控制：保证任何样本在被消费时 $$\lfloor (N_r-1)/B \rfloor \le i + \eta$$。
2. **逐 token 版本记录**（§5.4）。这是**事中**度量：训练时算出 `sample_staleness_theta_max/avg/min`。
3. **decoupled PPO + rejection sampling**（§6）。这是**事后**补偿：承认样本是旧的，用重要性权重与过滤把它用起来。

三层的关系可以写成一句话：**用队列保证上界，用版本记录真实值，用算法消化差值**。这也是 BRIEF 里「样本层：staleness 步数、partial rollout 的续用、队列深度」三个判据在 AReaL 上的完整落地。

### 7.4 三层的优先级结论

| 层 | AReaL 的投入 | 谁承担主要风险 |
| --- | --- | --- |
| 数值层 | 低（委托给引擎） | 推理引擎的 logprob 实现 |
| 权重层 | **高**（三条 refit 路径 + 版本化 LoRA + 独立服务） | refit 的「尽力而为」性质 |
| 样本层 | **最高**（队列 + 版本 + 算法三层） | 用户必须调对 $$\eta$$ |

一个反直觉的观察：**AReaL 把异步的风险几乎全部集中在样本层，然后给样本层配了最厚的一整套机制**。这与其他框架（把风险摊到三层、每层都做一点）是不同的架构选择。

---

## 8. 监控系统：指标层、追踪层、归因层

### 8.1 指标层：`stats_tracker` + `stats_tracker.scope`

指标是**按 scope 分层组织**的，每个 scope 有自己的分母：

| scope | 关键指标 | 分母 |
| --- | --- | --- |
| `rollout` | `accepted`、`rejected` | 计数 |
| `update` | `importance_weight`、`approx_kl`、`new_logp`、`old_logp`、`entropy`、`actor_loss`、`clip_ratio`、`dual_clip_ratio`、`logp_diff` | `n_valid_tokens` |
| `update`（rejection） | `behave_imp_weight`、`behave_approx_kl`、`behave_filtered_ratio` | `unclipped_behave_tokens` / `n_valid_tokens` |
| `version_stats` | `sample_staleness_proximal_{avg,max,min}`、`sample_staleness_theta_{avg,max,min}` | `n_valid_tokens` |

这套 scope 设计让「同一批数据在三个不同口径下的表现」可以并排看：`old_logp`（推理侧产）vs `new_logp`（训练侧前向）vs `prox_logp`（近端重算）。

指标的聚合走 `export_stats`：FSDP 引擎里是 `stats_tracker.export_all(reduce_group=self.data_parallel_group)`，**按 DP 组做 all-reduce**；控制器的 `export_stats` 再把多个 worker 的 `__count` 加权平均：

```python
for k, v in raw_stats.items():
    if k.endswith("__count"):
        counts[k] += v
    else:
        stats[k] += v * raw_stats.get(k + "__count", 0)
final_stats[k] = v / counts[count_key]
```

这是个**加权平均而非算术平均**——正确做法，但也意味着分母缺失的指标会静默消失（`if count_key in counts and counts[count_key] > 0`）。

训练循环最后把三份统计合起来提交：`stats = actor.export_stats() | rollout.export_stats() | eval_rollout.export_stats()`。**训练侧与推理侧的指标在同一个 step 对齐**，这是全异步下判断「是不是某一边出问题」的唯一入口——因为两边的 wall-clock 早已不同步，只能按 step 对齐。

下游 sink 支持 wandb / swanlab / trackio / tensorboardX 四家。

### 8.2 追踪层：`perf_tracer` 的两套 trace

`perf_tracer` 输出**两个 jsonl 文件**（<a href="#ref-perftracer">[26]</a>）：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `traces.jsonl` | span 级性能事件，带 7 类 category | Perfetto / Chrome Tracing 打开看时间线 |
| `sessions.jsonl` | 会话级生命周期（`GENERATE_START/END`、`REWARD_START/END`、`TOOLCALL_START/END`、`FINALIZED`） | 看一条 rollout 的时间都花在哪一段 |

7 个 category 是：`compute` / `comm` / `io` / `sync` / `scheduler` / `instr` / `misc`。**这个分类法本身就是一份诊断清单**：`sync` 类事件占比高 → 在等屏障；`comm` 高 → refit 或参数 broadcast 是瓶颈；`scheduler` 高 → 队列在卡。

最值得讲的是 **RPC 层的自动打点**。`engine_blueprint.py` 的 `/call` 处理器不看调用方，而是**按方法名关键词猜 category**<a href="#ref-enginebp">[10]</a>：

| 方法名含 | category |
| --- | --- |
| `submit` / `wait` | `scheduler` |
| `update_weights` / `broadcast` | `comm` |
| `save` / `load` | `io` |
| `train` / `eval` / `forward` / `compute` / `step` / `update` / `optimizer` / `zero_grad` / `lr_scheduler` | `compute` |

也就是说**任何一次跨节点的引擎调用都会被自动记进 trace，不需要在业务代码里手动埋点**。这是 RPC 抽象层的一个隐藏红利：因为所有调用都走同一个 `/call`，所以打点可以集中在一个地方。

`ContextVar` 的用法也值得一提：`_current_task_id`、`_current_session_id`、`_current_global_step` 都是 ContextVar，所以**在 async 协程里跨 await 点也能保持归因**——这是把 rollout 的并发协程与 trace 的层级结构对上账的前提。

### 8.3 归因层：从「慢」到「为什么慢」

AReaL 的归因能力集中在三处：

**(a) 会话级相位分解。**`sessions.jsonl` 里每条 rollout 都记 `submit_ts` / `finalized_ts` / `total_s` / `generate_s` / `reward_s` / `toolcall_s`，配套 `areal/tools/plot_session_trace.py` 出图（`plotly` + `make_subplots`，`TIMESTAMP_COLUMNS` / `DURATION_COLUMNS` 两组列名硬编码）。

这个分解对 agentic 任务尤其关键：**多轮 agent 的时间主要花在 toolcall 和 reward 上，不是 generate 上**。如果只看 GPU 利用率，会得出完全错误的优化方向。

**(b) 逐 token 版本归因。**`versions` 张量让你能回答「我这个 batch 里最旧的那个 token 是几步前生成的」。这是全异步独有的归因维度，同步系统里这个问题不存在。

**(c) staleness 四元组实时反馈。**`StalenessManager.get_stats()` 返回 `enqueued / running / accepted / rejected` 四个计数，在 `enable_rollout_tracing=True` 时每次提交/完成都打一行：

```text
Submit rollout. enqueued: 12, running: 34, accepted: 128, rejected: 5.
```

这四个数的组合有诊断意义：

| 现象 | 读数特征 | 含义 |
| --- | --- | --- |
| 生产端被限流 | `enqueued` 持续增长，`running` 不涨 | staleness 容量耗尽，在等训练消费 |
| 推理端堵住 | `running` 贴近 `max_concurrent_rollouts` | 推理服务器打满 |
| 数据质量差 | `rejected` 比例高 | `should_accept_fn` 或格式检查在大量拒样本 |
| 队列空转 | `enqueued` ≈ 0，`running` 低 | 数据源供不上（dataloader 慢） |

**(d) 2.x 的 refit 耗时。**权重更新服务的 `/update_weights` 返回 `duration_ms`，直接进日志与指标。这是把「refit 花了多久」从「需要自己掐表」变成「框架自报」。

### 8.4 监控在架构上的位置

```text
     ┌──────────────────────────────────────────────────┐
     │ 控制器（PPOTrainer）                              │
     │   stats_tracker（指标）  perf_tracer（trace）     │
     │        ▲                    ▲                    │
     │        │ export_stats       │ trace_scope        │
     └────────┼────────────────────┼────────────────────┘
              │                    │
     ┌────────┴────────┐  ┌────────┴─────────┐
     │ TrainController │  │ RolloutController│
     │  （聚合 DP 组）  │  │  （四元组统计）   │
     └────────┬────────┘  └────────┬─────────┘
              │                    │
     ┌────────┴────────────────────┴─────────┐
     │ worker 进程（每个 /call 自动打点）      │
     │  engine_blueprint → perf_tracer       │
     └───────────────────────────────────────┘
```

一个结构性观察：**监控是「控制面聚合、数据面打点」**——所有原始事件在 worker 侧产生，所有聚合在控制器侧完成。这与 §3.2 的控制面/数据面分工一致，好处是控制器可以只拉一次 `export_stats` 就拿到全局视图，不用连每个 worker。

---

## 9. 性能：论文口径与不可比性

### 9.1 端到端对比（论文表 1）

以下全部标 `[论文]`，出处为 arXiv 2505.24298v5 表 1，实验硬件为 **64 节点 × 8 张 H800**（NVLink 节点内 + RoCE 3.2Tbps 节点间），基线为「最新 verl 代码」，AReaL 与 AReaL 的同步变体（Sync.AReaL）同表对比：

| 模型 / 任务 | 方法 | 节点数 | PPO 步数 | 训练小时 | 相对 Sync.AReaL |
| --- | --- | --- | --- | --- | --- |
| 1.5B / AIME24 | w/ verl | 16 | 250 | 33.6 | — |
| 1.5B / AIME24 | w/ Sync.AReaL | 16 | 250 | **41.0** | 1.00× |
| 1.5B / AIME24 | **w/ AReaL** | 16 | 250 | **14.8** | **2.77×** |
| 7B / AIME24 | w/ Sync.AReaL | 24 | 250 | 57.7 | 1.00× |
| 7B / AIME24 | **w/ AReaL** | 24 | 250 | **25.4** | **2.27×** |
| 14B / LiveCodeBench | w/ Sync.AReaL | 32 | 80 | 48.8 | 1.00× |
| 14B / LiveCodeBench | **w/ AReaL** | 32 | 80 | **21.9** | **2.23×** |
| 32B / LiveCodeBench | w/ Sync.AReaL | 48 | 60 | 51.1 | 1.00× |
| 32B / LiveCodeBench | **w/ AReaL** | 48 | 60 | **31.1** | **1.64×** |

三个必须一起读的前提：

- **2.77× 只在 1.5B 上成立**。模型越大**加速比越小**（7B 为 2.27×、32B 为 1.64×）。论文摘要里的「up to 2.77×」是最乐观的那一格，不是典型值。
- **基线是 Sync.AReaL 而不是 verl**。论文对 verl 的对比只到「AIME24 43.1 / 33.6 h」这一行，且带 `*` 标记（引自 DeepScaler 的最佳已知可复现结果）。**论文没有在表 1 里直接给出 verl(nodes=16) 与 AReaL(nodes=16) 的训练小时差**，所以「AReaL 比 verl 快多少」这个问题，严格说法只有 Sync.AReaL 可以做同表比值。
- **评测条件统一**：最大生成长度 **32K** token，每题采样 **32** 条（即 `rollout.n = 32`），报告 pass@1 平均。

### 9.2 强扩展（论文图 4）

同一论文的口径：在 16K 与 32K 两种上下文长度、多个模型规模下与 verl 对齐，「AReaL 在所有设置下都呈现近似线性的扩展趋势，而同步系统通常无法有效扩展」；「大部分设置下 AReaL 吞吐超过基线，最多可达 **2.5×** 加速」`[论文]`。

论文还给了两条限定，都没有被摘要提到：

- **短上下文时优势会变小**，因为「生成吞吐跟不上训练吞吐——虽然生成了很多序列，但它们没有被训练过程有效消费」；
- **verl 在 32K 上下文 + 32B 模型下持续 OOM**，所以这些数据点缺失。

第二条尤其重要：**图 4 的「最多 2.5×」是在 verl 能跑通的那些点上取得的**，OOM 的那些点根本没有比值。把 OOM 当作「基线更差」来算加速比是错的。

### 9.3 系统消融（论文 §7.5）

| 优化项 | 增益 `[论文]` | 口径 |
| --- | --- | --- |
| 动态 microbatch 分配 | 平均 **30%** 吞吐提升 | 标准设定 32 个 micro-batch；动态批用每 micro-batch 32768 token 预算 |
| 可中断生成 | **12%**（1.5B）/ **17%**（7B） | 4 节点，与「必须等最长响应」对比 |

这两个数合起来仍然解释不了 2.77×。**差额来自阶段解耦本身**——也就是「训练与推理同时跑」这件事，论文没有把它单独消融（因为那需要重构系统）。

### 9.4 算法消融：staleness 与 decoupled 的交叉表

论文表 2（1.5B，数学任务，250 步）是本系列里信息量最高的一张表，`[论文]`：

| Max.Stale. $$\eta$$ | AIME24 无/有 decoupled | AIME25 无/有 | AMC23 无/有 | MATH500 无/有 |
| --- | --- | --- | --- | --- |
| 0（Oracle） | 42.0 / — | 32.9 / — | 84.4 / — | 89.2 / — |
| 1 | 41.8 / 42.1 | 30.7 / 31.9 | 83.3 / 85.2 | 89.9 / 89.8 |
| 2 | 40.0 / 41.8 | 32.1 / 32.5 | 82.3 / 84.3 | 89.6 / 89.6 |
| **4** | **23.3 / 42.2** | **23.1 / 32.0** | **58.5 / 85.1** | **66.9 / 89.5** |
| 8 | 35.7 / 41.0 | 27.8 / 31.1 | 81.2 / 82.9 | 87.8 / 89.2 |
| 16 | 35.8 / 38.7 | 26.2 / 32.5 | 78.4 / 83.2 | 87.4 / 89.1 |
| $$\infty$$ | 34.0 / 36.9 | 26.9 / 29.9 | 79.4 / 81.0 | 87.1 / 88.1 |

读这张表的三个要点：

1. **$$\eta=4$$ 是朴素 PPO 的崩塌点**（AIME24 从 42.0 掉到 23.3，MATH500 从 89.2 掉到 66.9），而 decoupled 版几乎无损（42.2 / 89.5）。这个断崖说明**朴素 PPO 在 off-policy 下不是「缓慢退化」而是「突然崩」**，用同步系统的经验去外推是危险的。
2. **$$\eta=8$$ 以后 decoupled 也开始退化**（AIME24 从 42.2 降到 41.0、38.7）。论文的结论是「即使有 decoupled 目标，无界 staleness（$$\eta \to \infty$$）仍然劣于零 staleness oracle；当约束得当时，中等 staleness（如 $$\eta \le 8$$）对最终性能影响极小」。
3. **论文的实验默认值就是 $$\eta=8$$（数学）/ $$\eta=4$$（代码）**，落在「几乎无损」区间的上沿。这个选择和 §4.3 的工程建议（「建议采用较大的 $$\eta$$」）方向一致。

### 9.5 不可比性声明

以下几条必须写清楚，否则读者会误用上面的数字：

| 声明 | 内容 |
| --- | --- |
| **硬件** | 全部为 H800；本文无任何 `[本机]` 数字，本机无 GPU |
| **基线版本** | 「最新 verl 代码」，论文未锁 commit；跨框架横向引用只在论文表 1 内成立 |
| **并行度** | 节点数从 16 到 48 不等，同表内对比才是受控的 |
| **colocate 与否** | AReaL 侧固定 **75% 推理 / 25% 训练** 的卡比（论文称比 50/50 吞吐更高，属经验配置）；这个比例对加速比影响极大，换比例后 2.77× 不成立 |
| **`rollout.n`** | 32 |
| **序列长度** | prompt 上限 1K，总长上限 32K |
| **首次编译** | 论文称吞吐在 warmup 步之后测量 |
| **不可横向比较** | 与 veRL fully-async 的 2.35–2.67×、slime 的任何数字**不在同一份 benchmark 内**，一律视为「厂商口径，不可横向比较」 |

---

## 10. 小结：它解决到什么程度、还剩什么

### 10.1 它解决到什么程度

| 问题 | AReaL 的答案 | 完成度 |
| --- | --- | --- |
| 批内长尾浪费 | 可中断生成 + 前缀重算 | **解决**，净收益 12%/17% `[论文]` |
| 阶段间串行 | 训练/推理解耦成两服务 + 样本队列 | **解决**，这是 2.77× 的主要来源 |
| 异步带来的分布偏移 | decoupled PPO（$$\pi_{\rm prox}/\pi_{\rm behav}$$ + 裁剪中心迁移） | **解决**，$$\eta \le 8$$ 近无损 `[论文]` |
| 离群 token 污染梯度 | `rejection_sampling`（5 种度量 × 2 种动作 × 2 种粒度） | **解决**，且可只当监控不开 |
| staleness 可观测 | 逐 token `versions` + 6 个 staleness 指标 | **解决**，这是它比其他框架强的地方 |
| 一条轨迹多版本混训 | 版本化 LoRA adapter + decoupled 目标 | **解决**（LoRA 全量共存） |
| 权重层的强一致 | 中断 + broadcast + 版本号 | **部分解决**，尽力而为，无事后校验 |
| 跨框架可比的加速比 | 论文给了同表对比 | **部分解决**，需带 75/25、H800、`n=32` 等一串前提 |

### 10.2 还剩什么

**(a) 权重层没有端到端校验。**`update_weights` 返回成功并不等于推理服务器权重全到位。训练侧的 barrier 只覆盖训练侧 collective；推理侧的 `areal_update_weights` 在 vLLM 里失败只 `logger.error` 返回 `(False, msg)`。全异步下这个问题比同步系统更严重，因为**没有屏障会把它暴露出来**——错误会一直传播到下一次 refit，期间所有 rollout 都用着混搭权重。

**(b) $$\eta$$ 的调参责任甩给了用户。**队列深度能保证上界，但「上界设多少」需要用户按模型规模与更新频率自己试。论文给了经验值（数学 8 / 代码 4）与方向建议（宁大勿小），但没有自动调节机制。

**(c) RPC 的 category 靠方法名猜。**`engine_blueprint.py` 用子串匹配 `["update_weights", "broadcast"]` 判断是不是 `comm`。这在方法名演化时会静默失准（比如方法改名成 `sync_params` 就落到 `misc`），而 trace 的 category 是给人看的第一层分类，错了不会报错。

**(d) 1.x 与 2.x 双栈并存的维护成本。**`RolloutController` 与 `RolloutControllerV2` 是「并行实现、duck-type 兼容」，意味着**同一处语义要改两遍且没有编译器帮忙保证一致**。ROADMAP 里 `Effective AReaL CLI support` 仍标为 on-going，说明服务化的用户体验还没收口。

**(e) rollout worker 的资源粒度与训练 worker 相反。**训练侧一卡一进程（`LOCAL_RANK=0` 硬编码），推理侧一个实例一进程（`instance_size = tp × pp` 张卡）。这个不对称让「按卡数统一调度」变复杂，colocate 路径里必须靠 `slots_per_node` 与 `base_gpu_id` 手工算偏移。

**(f) agentic 场景下的时间归因还没到「按轮」粒度。**`sessions.jsonl` 的 phase 是 `generate` / `reward` / `toolcall` 三段，多轮 agent 的每一轮都混在这三段里（虽然有 `TOOLCALL_START/END` 的多次执行）。要回答「第 3 轮的工具调用特别慢」需要另外的埋点。

### 10.3 一句话定位

**AReaL 把「异步」从一个优化开关变成了架构前提**：它的队列不是缓冲，是节流阀；它的中断不是容错，是常态；它的损失函数不是 PPO，是 decoupled PPO。代价是用户必须接受一个新的超参 $$\eta$$ 与一套新的心智模型（一条轨迹可以横跨多个策略版本），收益是在长序列、长尾、agentic 场景下把 GPU 利用率从「最长的那个 rollout 决定」变成「训练与推理各自跑满」。

在「全异步」这一级上，AReaL 目前是**唯一把抽象、机制、算法三层都做齐的开源实现**——这不是说它没有短板（§10.2 列了六条），而是说在同一级别上竞品的实现深度还没到可以逐行对照的程度。

---

## 参考

**论文与文档**

- [1] Fu, Gao, Shen, Zhu, Mei, He, Xu, Wei, Mei, Wang, Yang, Yuan, Wu. *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning*. arXiv:2505.24298v5, 2026-03-02. <https://arxiv.org/abs/2505.24298>（本文表 1、表 2、图 4、图 6、式 (3)(4)(5)、§5.1、§7.1、§7.5 均出此处）
- AReaL 2.0 技术报告：arXiv:2607.01120（README News 2026-07-01 条引用）
- AReaL 异步 RL 指南：`docs/en/algorithms/async.md`（`max_head_offpolicyness` 与 decoupled 目标的官方说明）

**源码（全部锁 `v2.1.0` = `ecc8b0e4dfb4e3965f67121a5c87866a07c390ae`）**

<a id="ref-entry"></a>[2] `examples/math/gsm8k_rl.py` — <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/examples/math/gsm8k_rl.py#L1-L47>

<a id="ref-rltrainer"></a>[3] `areal/trainer/rl_trainer.py` — `_init_impl` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/trainer/rl_trainer.py#L124-L260>；`train` 主循环 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/trainer/rl_trainer.py#L640-L966>；`_init_rollout` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/trainer/rl_trainer.py#L1177-L1265>

<a id="ref-trainctl"></a>[4] `areal/infra/controller/train_controller.py` — `initialize` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/train_controller.py#L254-L323>；`_async_create_engines`（`LOCAL_RANK=0`）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/train_controller.py#L332-L356>；`_async_initialize_engines`（两阶段）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/train_controller.py#L358-L384>；`_dispatch_tensors`（按 token 数均衡切分）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/train_controller.py#L76-L122>

<a id="ref-rolloutctl"></a>[5] `areal/infra/controller/rollout_controller.py` — `initialize`（含 staleness manager 与 dispatcher 构造）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/rollout_controller.py#L162-L231>；`_async_initialize`（三条起服路径）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/rollout_controller.py#L233-L370>；callback 服务器 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/rollout_controller.py#L605-L713>；`_create_submit_callback` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/rollout_controller.py#L851-L936>；`set_version` / `pause` / `resume` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/controller/rollout_controller.py#L1174-L1206>

<a id="ref-rcv2"></a>[6] `areal/v2/inference_service/controller/controller.py` — `RolloutControllerV2` docstring <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/v2/inference_service/controller/controller.py#L1-L78>

<a id="ref-gw2"></a>[7] `areal/v2/inference_service/gateway/app.py` — <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/v2/inference_service/gateway/app.py#L1-L7>

<a id="ref-router2"></a>[8] `areal/v2/inference_service/router/app.py` — <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/v2/inference_service/router/app.py#L1-L12>

<a id="ref-rpcserver"></a>[9] `areal/infra/rpc/rpc_server.py` — <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/rpc/rpc_server.py#L1-L66>

<a id="ref-enginebp"></a>[10] `areal/infra/rpc/guard/engine_blueprint.py` — 单线程 engine worker <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/rpc/guard/engine_blueprint.py#L108-L181>；`/call` 处理器与 category 猜测 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/rpc/guard/engine_blueprint.py#L425-L560>

<a id="ref-rtensor"></a>[11] `areal/infra/rpc/rtensor.py` — 客户端 fetch buffer 与清理不变量 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/rpc/rtensor.py#L287-L348>；`to_local` / `remotize` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/rpc/rtensor.py#L351-L420>；批量取 shard <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/rpc/rtensor.py#L208-L250>；`/data/*` 端点 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/rpc/guard/data_blueprint.py#L1-L76>

<a id="ref-stale"></a>[12] `areal/infra/staleness_manager.py` — 全文 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/staleness_manager.py#L20-L182>；`get_pending_limit` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/staleness_manager.py#L69-L77>；`get_capacity` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/staleness_manager.py#L79-L113>；`on_version_recovered` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/staleness_manager.py#L115-L131>

<a id="ref-dispatcher"></a>[12b] `areal/infra/workflow_executor.py` — `BatchTaskDispatcher` 三层结构 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/workflow_executor.py#L288-L341>；`_has_runner_capacity` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/workflow_executor.py#L356-L361>；生产/消费线程 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/workflow_executor.py#L387-L477>；`wait_results` 与 `_select_results` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/workflow_executor.py#L574-L633>；`active_submit_and_wait` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/workflow_executor.py#L664-L761>；`_compute_output_versions`（head/tail/RLE）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/workflow_executor.py#L864-L878>；`WorkflowExecutor.initialize`（DP 缩放）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/workflow_executor.py#L1046-L1102>；`_create_workflow_task`（accept/reject）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/workflow_executor.py#L1171-L1314>

<a id="ref-agenerate"></a>[13] `areal/infra/remote_inf_engine.py` — `agenerate`（中断续跑主循环）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/remote_inf_engine.py#L889-L1060>；`pause_generation` / `continue_generation` / `pause` / `resume` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/remote_inf_engine.py#L1400-L1429>；`set_version` / `get_version` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/remote_inf_engine.py#L601-L609>

<a id="ref-sglangpause"></a>[14] `areal/engine/sglang_remote.py` — `get_pause_requests`（两段式暂停）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/sglang_remote.py#L339-L372>；abort 前缀解析 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/sglang_remote.py#L104-L140>

<a id="ref-sglanggen"></a>[24] `areal/engine/sglang_remote.py` — `build_generation_request`（版本化 LoRA）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/sglang_remote.py#L54-L102>；`build_distributed_weight_update_requests` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/sglang_remote.py#L227-L253>；`build_disk_weight_update_requests`（adapter 保留窗口）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/sglang_remote.py#L176-L225>

<a id="ref-fsdpupd"></a>[23] `areal/engine/fsdp_engine.py` — `update_weights` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/fsdp_engine.py#L620-L635>；`_update_weights_from_distributed`（桶式流水 + barrier）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/fsdp_engine.py#L1596-L1692>；`export_stats` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/fsdp_engine.py#L922-L926>

<a id="ref-wugw"></a>[25] `areal/v2/weight_update/gateway/app.py` — `/update_weights` 路由 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/v2/weight_update/gateway/app.py#L704-L758>；`ConnectRequest` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/v2/weight_update/gateway/app.py#L31-L50>

<a id="ref-vllmworker"></a>[25b] `areal/engine/vllm_ext/vllm_worker_extension.py` — `areal_update_weights`（失败只记日志）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/engine/vllm_ext/vllm_worker_extension.py#L39-L56>

<a id="ref-resolveprox"></a>[18] `areal/trainer/ppo/actor.py` — `_resolve_proximal_logp` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/trainer/ppo/actor.py#L851-L930>；`grpo_loss_fn` 主体 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/trainer/ppo/actor.py#L527-L760>；`_log_version_staleness_stats` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/trainer/ppo/actor.py#L1229-L1270>

<a id="ref-advantage"></a>[22] `areal/trainer/ppo/actor.py` — `use_decoupled_loss` 分支（old_logp / prox_logp 赋值）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/trainer/ppo/actor.py#L214-L247>

<a id="ref-rsconfig"></a>[19] `areal/api/cli_args.py` — `RejectionSamplingConfig` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/api/cli_args.py#L1504-L1570>；`InferenceEngineConfig`（`max_head_offpolicyness` / `queue_size` / `consumer_batch_size` / `pause_grace_period`）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/api/cli_args.py#L2477-L2572>

<a id="ref-rs"></a>[20] `areal/utils/functional/functional.py` — `apply_rejection_sampling` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/utils/functional/functional.py#L185-L436>；`compute_binary_kl_divergence` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/utils/functional/functional.py#L440-L450>

<a id="ref-ppoactorloss"></a>[21] `areal/utils/functional/functional.py` — `ppo_actor_loss_fn`（decoupled 目标 + 分母保护）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/utils/functional/functional.py#L452-L570>；序列级 ratio <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/utils/functional/functional.py#L56-L149>

<a id="ref-rlvr"></a>[17] `areal/workflow/rlvr.py` — 轨迹字典构造（`versions = [-1]*input_len + output_versions`）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/workflow/rlvr.py#L164-L180>；`areal/api/io_struct.py` `ModelResponse.output_versions` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/api/io_struct.py#L60-L80>

<a id="ref-perftracer"></a>[26] `areal/utils/perf_tracer.py` — `PerfTraceCategory` 七类 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/utils/perf_tracer.py#L63-L119>；`SessionTraceEvent` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/utils/perf_tracer.py#L247-L295>；ContextVar 定义 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/utils/perf_tracer.py#L31-L41>

<a id="ref-scheduler"></a>[27] `areal/api/scheduler_api.py` — `Worker` / `Job` / `Scheduler` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/api/scheduler_api.py#L13-L80>；`areal/infra/scheduler/local.py` `LocalScheduler` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/scheduler/local.py#L92-L98>；默认 worker 命令 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/infra/scheduler/local.py#L257-L272>

<a id="ref-engineapi"></a>[28] `areal/api/engine_api.py` — `TrainEngine` ABC（含 `update_weights` / `set_version` / `prepare_batch`）<https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/api/engine_api.py#L32-L292>；`InferenceEngine` ABC 起点 <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/api/engine_api.py#L551-L560>

<a id="ref-singlec"></a>[29] `areal/utils/environ.py` — `is_single_controller()` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/utils/environ.py#L74-L75>

<a id="ref-versionedlora"></a>[30] `areal/api/io_struct.py` — `get_versioned_lora_name` <https://github.com/areal-project/AReaL/blob/ecc8b0e4dfb4e3965f67121a5c87866a07c390ae/areal/api/io_struct.py#L162-L164>

**跨框架对照（slime 侧，锁 `v0.4.0` = `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`）**

<a id="ref-slimeargs"></a>[15] `slime/utils/arguments.py` — `--partial-rollout` / `--mask-offpolicy-in-partial-rollout` 定义 <https://github.com/slime-project/slime/blob/0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e/slime/utils/arguments.py#L489-L508>；`--buffer-sort-by-staleness` <https://github.com/slime-project/slime/blob/0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e/slime/utils/arguments.py#L561-L569>

<a id="ref-slimedrain"></a>[16] `slime/rollout/sglang_rollout.py` — partial 样本回收 <https://github.com/slime-project/slime/blob/0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e/slime/rollout/sglang_rollout.py#L440-L473>；`loss_mask` 置零 <https://github.com/slime-project/slime/blob/0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e/slime/rollout/sglang_rollout.py#L288-L304>；`slime/backends/sglang_utils/server_control.py` `abort_servers_until_idle` <https://github.com/slime-project/slime/blob/0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e/slime/backends/sglang_utils/server_control.py#L237-L254>

**仓库与版本**

- AReaL：`v2.1.0` = `ecc8b0e4dfb4e3965f67121a5c87866a07c390ae`（2026-08-25），<https://github.com/areal-project/AReaL>
- slime：`v0.4.0` = `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`（2026-10-09），<https://github.com/slime-project/slime>

---

**上一篇**：[veRL 源码拆解](/2026/10/24/rl-frameworks-03-verl/) ｜ **下一篇**：[RLinf：具身与 LLM 后训练的统一](/2026/10/24/rl-frameworks-05-rlinf/)
