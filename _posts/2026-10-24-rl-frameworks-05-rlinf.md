---
title: "RLinf 源码拆解：当 rollout 不再是「一次生成」，而是「几十步的仿真」"
date: 2026-10-24 11:00:00 +0800
categories:
  - 强化学习
tags: [rl, rlinf, embodied, vla, m2flow, simulation, scheduler, channel, placement, isaac-lab, maniskill, libero, fsdp, sglang, source-code]
layout: post
mathjax: true
---

> **系列导航** ｜ 本系列：[01 Megatron 训练栈](/2026/10/24/rl-frameworks-01-megatron/) ｜ [02 slime](/2026/10/24/rl-frameworks-02-slime/) ｜ [03 veRL](/2026/10/24/rl-frameworks-03-verl/) ｜ [04 AReaL](/2026/10/24/rl-frameworks-04-areal/) ｜ 本文 05 ｜ [06 SGLang 接缝](/2026/10/24/rl-frameworks-06-sglang/) ｜ [07 vLLM 接缝](/2026/10/24/rl-frameworks-07-vllm/) ｜ [08 两族横评](/2026/10/24/rl-frameworks-08-landscape/) ｜ 全景：[RL 训练框架横评](/2026/09/29/rl-training-frameworks-survey/) ｜ 算法侧：[从 MDP 到 GRPO](/2026/08/21/mdp-to-grpo-05-grpo-group-relative/)

**TL;DR**

> * **RLinf 的存在理由不是「又一个 LLM RL 框架」，而是「rollout 的粒度变了」**。LLM 的 rollout 是一次自回归生成，几秒到几分钟；具身智能的 rollout 是一个 episode，几十到几百步，每步都要渲染、物理仿真、动作下发。同一个框架要同时吃下这两种时间尺度，原本理所当然的「同步屏障」直接失效。
> * **它自建了一套调度层，而不是用 Ray 的 actor 模型硬凑**。`rlinf/scheduler/` 下有 `worker` / `placement` / `channel` / `collective` / `dynamic_scheduler` / `hardware` / `cluster` / `manager` 八个子模块——**这是一套完整的分布式编排基础设施**，承担的职责与 veRL 的 `single_controller` 同级，只是设计目标不同。
> * **`Channel` 是这套调度层的心脏**。它被设计成「带 key 路由与权重的 FIFO 队列」，语义对齐 `asyncio.Queue`。文档里点明了存在理由：*multi-turn scenarios in agent and embodied RL, where a data is processed by a fixed set of workers*——**多轮与具身场景需要「数据去找固定的 worker」而不是「worker 抢数据」**，这是普通队列做不到的。
> * **放置策略把「机器人」也当成一种硬件**。`FlexiblePlacementStrategy` 的文档原话是 *place processes on any hardware (accelerators, robots, etc.)*。把仿真器/真机作为一等硬件资源参与放置，是它与其他所有 RL 框架最根本的架构差异。
> * **异构调度的解法是「按顺序创建 worker」**。`init_workers` 的注释是 *create worker in order to decrease the maximum memory usage*：先起 rollout，再起 env，再起 reward，再起 actor。因为仿真器与推理引擎的初始化峰值内存都很大，同时起会 OOM。
> * **性能数字必须看口径**：BEHAVIOR 仿真器的 **1028.7 ms/step → 41.2 ms/step（25×）** 是**单个仿真步的延迟**，不是训练吞吐，也不是端到端训练加速比。官方博客把它们分开写了，引用时不能混。

---

## 目录

- [1. 设计出发点：rollout 的粒度变了](#1-设计出发点rollout-的粒度变了)
- [2. 模块初始化逻辑：八个调度子模块](#2-模块初始化逻辑八个调度子模块)
- [3. 进程与角色：仿真器是 CPU 密集，模型是 GPU 密集](#3-进程与角色仿真器是-cpu-密集模型是-gpu-密集)
- [4. 数据与样本流：episode 轨迹怎么变成训练样本](#4-数据与样本流episode-轨迹怎么变成训练样本)
- [5. 权重同步：小模型改变了什么](#5-权重同步小模型改变了什么)
- [6. 训推一致三层：具身场景的特有难点](#6-训推一致三层具身场景的特有难点)
- [7. 异步形态：长尾是常态而不是异常](#7-异步形态长尾是常态而不是异常)
- [8. 监控系统](#8-监控系统)
- [9. 性能 benchmark：口径比数字重要](#9-性能-benchmark口径比数字重要)
- [10. 小结：它到底解决了什么、还剩什么](#10-小结它到底解决了什么还剩什么)

---

## 1. 设计出发点：rollout 的粒度变了

### 1.1 两种 rollout 的对照

所有为 LLM 后训练设计的 RL 框架，骨子里都假设了同一件事：**一次 rollout = 一次带 continuous batching 的自回归生成**。这个假设撑起了一整套设计——批量请求、KV 缓存复用、按 token 计费、组内 n 条样本的同步返回。

具身 RL 打破了这个假设：

| 维度 | LLM rollout | 具身 rollout |
| --- | --- | --- |
| 单位 | 一次 completion | 一个 episode |
| 长度 | 几十到几万 token | 几十到几百个控制步 |
| 每步做什么 | 一次 decode | 渲染 + 物理仿真 + 模型前向 + 动作下发 |
| 谁在算 | GPU（decode） | **GPU（策略）+ CPU（仿真器）交替** |
| 耗时分布 | 长尾（少数超长） | **每一步都可能长尾** |
| 状态 | 无（每步独立） | 有（物理状态、物体姿态、机器人位姿） |
| 奖励 | 结尾一次 | 多步稀疏或逐步稠密 |
| 失败模式 | 超长、重复、格式错 | **仿真崩溃、真机通信超时、动作越界** |

这张表里最关键的一行是「**谁在算**」。仿真器是 CPU 密集且高度并行单线程的，模型是 GPU 密集的。两者在同一次 rollout 里交替出现，意味着**GPU 在仿真期间是空闲的，CPU 在模型前向期间是空闲的**。这是所有具身 RL 框架的首要优化目标。

### 1.2 M2Flow：为什么需要一种新范式

RLinf 的论文（[arXiv 2509.15965](https://arxiv.org/abs/2509.15965)）把它的做法称为 **M2Flow**（macro-to-micro flow）。名字里的两级对应两种尺度：

- **macro**：跨 worker 组的大流程——rollout 组、env 组、actor 组的编排
- **micro**：单个 worker 内部的小流程——episode 内的步进循环

两级分开设计的原因是：**macro 层的调度单位是「组」，micro 层的调度单位是「步」**，把两者塞进同一个调度器会让抽象变得不可读。这也是为什么 RLinf 的调度层有 `worker_group` 与 `worker` 两层，而不是只有一层。

### 1.3 它放弃了什么

M2Flow 的代价很明确：

1. **不支持 LLM 特有的优化**。continuous batching、prefix caching 这些是在 micro 层做的，RLinf 把它们透传给底层的推理引擎（SGLang）。
2. **抽象层次多，上手门槛高**。要写一个新的仿真器接入，得理解 `Worker` / `WorkerGroup` / `Placement` / `Channel` 四个概念。
3. **论文口径的加速比不可横向比较**。它的基线是「同一个仿真器不做系统优化」，而不是「另一个 RL 框架」。

---

## 2. 模块初始化逻辑：八个调度子模块

### 2.1 调度层清单

`rlinf/scheduler/` 的八个子模块各管一件事：

| 子模块 | 职责 | 关键抽象 |
| --- | --- | --- |
| `worker` | 执行单元 | `Worker` / `WorkerGroup` |
| `placement` | 摆位 | `PlacementStrategy`（Packed / Node / Flexible） |
| `channel` | 通信 | `Channel`（带 key 路由与权重的队列） |
| `collective` | 集合通信 | `AsyncWork` 系列句柄 |
| `dynamic_scheduler` | 运行时调度 | 动态增删 worker |
| `hardware` | 硬件抽象 | `Accelerator`（含 MUSA / CANN / ROCm） |
| `cluster` | 集群视图 | `NodeInfo` / `NodeGroupInfo` |
| `manager` | 生命周期 | worker 的启停与健康 |

**这套东西的规模与 veRL 的 `single_controller` 相当，但目标不同**：veRL 要表达「同一份算法在不同 placement 上跑」，RLinf 要表达「不同类型的工作负载放在不同类硬件上」——后者多了 CPU 侧与仿真器这两类资源。

### 2.2 `Channel` 的设计细节

`Channel` 的类文档把它与 `asyncio.Queue` 显式对齐：

> **Interface**: Similar as the `asyncio.Queue`, the `Channel` provides interfaces like `put`, `get`, `put_no_wait`, and `get_no_wait`, as well as query interfaces like `qsize`, `empty`, and `full`. The semantics of these interfaces are identical to those of `asyncio.Queue`.

在队列语义之上，它加了四个特性，其中两个决定了它能用在哪：

**特性一：异步操作可以脱离 asyncio 上下文使用**。

> This async can be used not only in asyncio context with `await channel.get(async_op=True).async_wait()`, but also in non-asyncio contexts by generating a communication handle that can be waited later, like async `torch.distributed.send()`.

这句话描述的模式与 PyTorch 的 `isend`/`irecv` 一致：**发起时拿到句柄，稍后再等**。这个能力对仿真器接入是必需的——仿真器 API 通常是同步阻塞的，不能 `await`，但又不希望阻塞整个 worker。

**特性二：按 key 路由**。

> Key-based routing: Channel allows specifying a `key` for each data item, which can be used to identify and route messages. […] This is useful in multi-turn scenarios in agent and embodied RL, where a data is processed by a fixed set of workers.

**「数据去找固定的 worker」而不是「worker 抢数据」** 是这里的关键语义。在普通的请求-响应系统里，哪个 worker 处理无关紧要；但在多轮 agent 与具身场景里，**同一个 episode 的后续步骤必须回到持有该 episode 状态的 worker**——物理状态、随机数状态、仿真器内部缓存都在那个进程里。用 key 路由而不是「把状态序列化了搬过去」，是唯一不需要在每步之间序列化仿真器状态的做法。

**特性三：权重与批量提取**。`get_batch` 按权重取一批消息。这在多环境异步推进时用来做「按优先级取」，避免慢环境饿死。

### 2.3 放置策略：机器人也是硬件

`FlexiblePlacementStrategy` 的文档第一句就点明了它的野心：

> This placement strategy allows processes to be placed on any hardware (accelerators, robots, etc.) by specifying a list of *global* hardware ranks for each process.

三种策略的分工：

| 策略 | 语义 | 用在哪 |
| --- | --- | --- |
| `PackedPlacementStrategy` | 尽量挤在一台机器 | 小规模、单机 |
| `NodePlacementStrategy` | 按节点整块分配 | 多机、需要节点内 NVLink |
| `FlexiblePlacementStrategy` | 显式列出每个进程的全局 rank | 异构、真机、仿真器混部 |

「机器人也是硬件」这件事在实现上表现为：`Cluster` 的节点信息里不只有 GPU 数，还有「这一组节点是什么类型」（`node_group`），可以把「4 台带 GPU 的服务器 + 8 台带机械臂的工作站」放进同一个集群描述里。

### 2.4 初始化是串行的，理由在注释里

`EmbodiedRunner.init_workers` 只有十几行，但顺序是有意为之：

```python
def init_workers(self):
    # create worker in order to decrease the maximum memory usage
    rollout_handle = self.rollout.init_worker()
    env_handle = self.env.init_worker()

    if self.reward is not None:
        self.reward.init_worker().wait()

    rollout_handle.wait()
    env_handle.wait()
    self.actor.init_worker().wait()
```

两点值得注意：

1. **注释直接给出理由**：*create worker in order to decrease the maximum memory usage*。`init_worker()` 返回句柄后**不立即等待**，而是继续发起下一组；但 reward 与 actor 之间插了一个 `.wait()`。这个插法的含义是：rollout 与 env 的初始化可以并发（它们抢的是不同资源：GPU 与 CPU），但 actor 必须在 reward 之后——因为 actor 要加载的模型与 reward 模型可能共享显存池。
2. **actor 最后起**。这是显存考虑：actor 持有优化器状态，是三者里显存占用最大的；如果先起 actor，后面的 rollout 引擎就没有空间了。

这个「按内存峰值排序的串行初始化」是异构调度在初始化阶段的具体表现。

---

## 3. 进程与角色：仿真器是 CPU 密集，模型是 GPU 密集

### 3.1 八类 worker

`rlinf/workers/` 下按角色分目录：

| 目录 | 角色 | 资源类型 |
| --- | --- | --- |
| `rollout` | 生成/执行策略 | GPU |
| `env` | 环境交互（仿真器或真机） | **CPU + 可选 GPU（渲染）** |
| `actor` | 训练 | GPU |
| `critic` | 价值函数 | GPU |
| `reward` | 奖励计算 | GPU 或 CPU |
| `inference` | 独立推理服务 | GPU |
| `agent` | 多轮 agent 控制 | CPU |
| `sft` | 监督微调 | GPU |

**`env` 被拆成一个独立的 worker 类别，是这个框架最核心的结构决定**。在纯 LLM 框架里，环境是 reward function 的一部分，跑在 driver 进程里；在这里，环境是一个需要摆位、需要独立生命周期的资源。

### 3.2 三类 env worker

`rlinf/workers/env/` 下有三个实现：

- `env_worker.py`：标准同步环境
- `async_env_worker.py`：异步推进的环境
- `rtc_env_worker.py`：实时控制环境（真机用）

**同步 / 异步 / 实时三条路径对应仿真器 API 的三种形态**：Gymnasium 风格是同步 `step()`，Isaac Lab 可以做异步，真机则是实时约束（控制周期必须稳定）。把它们分成三个 worker 类而不是在同一个类里加分支，是因为它们的中断语义完全不同——真机上的握手超时不能像仿真那样重试。

### 3.3 异构调度：谁在等谁

一次 episode 期间的资源占用：

```text
时间
 |
 |  GPU:  [策略前向][空闲 ][策略前向][空闲 ]... [策略前向]
 |  CPU:  [仿真步进][渲染][仿真步进][渲染][仿真步进]...
 |
 +--------------------------------------------------->
```

理想情况下这两个占空比能被叠起来，实际受两个约束限制：

1. **回合内的因果依赖**：第 $$t+1$$ 步的策略前向必须等第 $$t$$ 步的仿真结果。所以**同一 episode 内无法重叠**。
2. **回合间的重叠是唯一出路**：把多个环境并行推进，当 A 环境在仿真时 B 环境在做策略前向。这就是为什么具身 RL 天然需要大量并行环境（vectorized env），而这恰好也是 RL 本身的方差需求。

所以 RLinf 的重叠策略与 LLM 框架完全不同：**LLM 框架重叠的是「生成」与「训练」，具身框架重叠的是「这一批环境的仿真」与「那一批环境的推理」**。

---

## 4. 数据与样本流：episode 轨迹怎么变成训练样本

### 4.1 两种样本形态并存

RLinf 同时支持两类任务，它们的样本结构不同：

| | reasoning（LLM） | embodied（VLA） |
| --- | --- | --- |
| 样本 | prompt + response tokens | 观测序列 + 动作序列 + 奖励序列 |
| runner | `reasoning_runner.py` | `embodied_runner.py` / `async_embodied_runner.py` |
| 优势单位 | 组内（GRPO） | 可在组内或按 episode |
| 奖励 | 结尾一次 | 多步 |

两类 runner 分开而不是合并，是因为**优势估计的粒度根本不同**：LLM 的 token 级优势是从序列级回报摊下来的，具身是多步折扣回报。统一抽象会迫使两边都退到最弱的共同处。

### 4.2 runner 是「薄控制面」

`EmbodiedRunner` 的方法列表读起来像一份流程清单：`init_workers` / `update_rollout_weights` / `evaluate` / `_maybe_eval_and_checkpoint` / `_log_step_metrics` / `_finish_run`。

**它不持有数据**。数据在 `Channel` 里流动，runner 只负责在正确的时间发出正确的调用并收集指标。这个分工与 veRL 的 trainer 有本质差别：veRL 的 trainer 是数据流的中心节点，RLinf 的 runner 只是时序编排者。

`update_rollout_weights` 是唯一值得细看的方法：

```python
def update_rollout_weights(self):
    rollout_handle: Handle = self.rollout.sync_model_from_actor()
    actor_handle: Handle = self.actor.sync_model_to_rollout()
    actor_handle.wait()
    rollout_handle.wait()
```

**两侧同时发起，然后一起等**。发送端与接收端是配对的集合通信，必须两边都进入才成立；如果串行调用（先 actor 发完再 rollout 收），就会死锁。这个「双向发起再汇合」是分布式握手的基本模式，但在 RL 框架里经常被写成串行。

---

## 5. 权重同步：小模型改变了什么

### 5.1 关键差异：模型尺寸

VLA 模型（OpenVLA 7B、π₀ 3B、GR00T 3B 量级）比 LLM RL 里的 32B/72B 小一个数量级。这改变了权重同步的成本结构：

| | LLM RL（7B–72B） | 具身（3B–7B） |
| --- | --- | --- |
| 一份 bf16 权重 | 14–144 GB | 6–14 GB |
| 单次全量同步耗时 | 秒级到十几秒 | **百毫秒级** |
| 是否需要 delta 同步 | 需要（72B 时差 3.1×） | **不需要** |
| 是否需要专门的 resharding 引擎 | 需要 | 通常不需要 |

**结论：RLinf 在权重同步上不需要做激进优化，因此也不应该把「权重同步性能」作为评价它的维度。** 这是选型时最容易搞错的地方——拿 LLM 框架的权重同步指标去要求一个具身框架。

### 5.2 同步路径

`sync_model_to_rollout` / `sync_model_from_actor` 这一对方法名揭示了方向：**actor → rollout**。具体实现依赖底层引擎，`rlinf/models/` 下有一套模型抽象，`rlinf/hybrid_engines/` 下有引擎适配（含 SGLang）。

**`hybrid_engines` 这个目录名是关键**：它意味着 RLinf 不假设「训练引擎」与「推理引擎」是同一个东西，而是为「混合引擎」（同一个进程既能训也能推）留了位置。这与 slime 的 `--colocate` 达成同一目的，但通过不同手段——slime 靠 Ray 摆位，RLinf 靠引擎适配层的统一接口。

---

## 6. 训推一致三层：具身场景的特有难点

### 6.1 数值层：预处理链是一阶误差源

LLM 的数值层问题集中在 attention kernel 与 rope 上。具身场景多了一整条**视觉预处理链**：

```text
原始渲染帧 → resize → crop → normalize → 颜色空间转换 → patchify → 模型
```

这条链上每一环都可能有「训练时用 A 实现、推理时用 B 实现」的分歧。最典型的三个：

1. **resize 插值方式**（bilinear vs bicubic vs antialias）
2. **normalize 的均值方差**（ImageNet 统计量 vs 模型自带）
3. **颜色通道顺序**（RGB vs BGR，OpenCV 默认是 BGR 而 PyTorch 是 RGB）

**这三处任意一处不一致，都会让同一帧图像产生可见的像素差异**，进而在 logprob 上放大。它们比 attention kernel 的差异大两个数量级——这一点与 LLM 场景的直觉相反。

### 6.2 数值层：动作归一化

VLA 模型的动作输出通常要经过归一化（action normalization）落到真实控制器的量程上。归一化统计量（每维的 min/max 或均值方差）从哪来：

- 从训练数据集统计 → 推理时必须用同一份
- 从当前 episode 在线估计 → 训练与推理的统计口径不同

第二种做法在 RL 微调里更常见（因为要适应探索），但它天然引入训推不一致——**因为统计量在变化**。这不是 bug，而是设计选择，需要在训练监控里显式记录统计量的漂移。

### 6.3 权重层与样本层

权重层在具身场景里反而简单（第 5 节），样本层则更复杂：一个 episode 的几十步是用**同一份权重**生成的，还是跨了几个权重版本？具身 RL 的 episode 比 LLM 的 completion 长得多，**如果采用异步更新，一个 episode 内跨版本几乎是必然的**。

RLinf 的解法在 `async_ppo_embodied_runner.py` 与 `async_weight_sync_mixin.py` 里——用一个 mixin 把「换权重」这个动作编织进 episode 推进流程，而不是在 episode 边界上做屏障。

---

## 7. 异步形态：长尾是常态而不是异常

### 7.1 具身的 staleness 问题更严重

LLM 的 staleness 单位是「一个 step 的生成」，通常是分钟级。具身的 staleness 单位是「一个 episode」，可以是几分钟到几十分钟（真机场景）。

这意味着：**如果按「episode 全部完成才更新权重」做同步，GPU 的有效利用率会非常低**。所以 RLinf 从设计上就偏向异步：

| runner | 形态 |
| --- | --- |
| `embodied_runner.py` | 同步（基准实现） |
| `async_embodied_runner.py` | 异步 |
| `async_ppo_embodied_runner.py` | 异步 + PPO（带 critic） |

### 7.2 异步的代价

按本系列的判定标准，RLinf 的异步属于**off-policy 异步**（训练第 $$k$$ 步时 rollout 已在产第 $$k+1$$ 步），但**不是全异步**（训练端与推理端仍由同一个 runner 编排，不是两个独立服务）。

代价是 off-policy 程度随时间线性增长。对具身场景来说，这个代价比 LLM 更大，因为：

- **环境状态分布会漂移**：策略变强后，episode 的观测分布整体平移，旧数据不只是「策略旧了」，而是「环境状态也不在分布内了」
- **PPO 的 clip 假设会失效**：clip 的有效区间是按「新旧策略接近」设计的

所以具身异步必须配更强的重要性修正或更保守的衰减。这是目前所有具身 RL 框架都还没完全解决的问题。

---

## 8. 监控系统

### 8.1 runner 里直接内置了指标聚合

`EmbodiedRunner` 的指标相关方法密度异常高——在一份编排代码里出现了 `print_metrics_table_async` / `_log_ranked_metrics` / `_aggregate_numeric_metrics` / `_process_ranked_numeric_results` / `_process_ranked_eval_results` / `_log_step_metrics` 六个方法。

这个设计选择反映了一个实现现实：**具身 RL 的指标来自多个异构 worker**（环境侧的成功率、推理侧的延迟、训练侧的 loss），把它们统一成一张表需要跨 worker 的归集逻辑。RLinf 把这个逻辑放在 runner 里，而不是抽一个独立的 metric 模块。

### 8.2 指标的分类

从方法名可以还原指标的三大类：

| 类 | 来源 | 例子 |
| --- | --- | --- |
| **ranked** | 多环境并行时的分排名指标 | 每个 env 的成功率，取 min/分位数 |
| **numeric aggregate** | 标量聚合 | 平均回报、平均 episode 长度 |
| **step metrics** | 单步 | 动作范数、推理时延 |

**`ranked` 这一类是具身特有的**。LLM 场景里所有样本地位等价，取平均即可；具身场景里不同环境的难度差异巨大（LIBERO 的不同任务、BEHAVIOR 的不同场景），取平均值会把「一个简单环境刷高分」和「所有环境都略有进步」混为一谈。所以必须保留分环境的排名。

---

## 9. 性能 benchmark：口径比数字重要

### 9.1 那个 25× 到底是什么

官方 README 的原文：

> RLinf achieves **25×** end-to-end speedup for the BEHAVIOR simulator through system-level optimizations (slimming, on-demand observation, hybrid pipeline parallelism), reducing rollout latency from 1028.7 ms/step to 41.2 ms/step.

拆解这段话的每一部分：

| 论断 | 实际口径 | 不能这么理解 |
| --- | --- | --- |
| 25× | **针对 BEHAVIOR 仿真器**的加速 | 不是所有仿真器的普遍加速 |
| end-to-end | 从「仿真器被调用」到「观测返回」 | **不是训练的端到端时间** |
| 1028.7 → 41.2 ms/step | 单个仿真步的延迟 | 不是训练吞吐（steps/s）的提升 |
| slimming / on-demand observation / hybrid pipeline parallelism | 三项系统优化 | 不包含模型或算法改动 |

**最关键的一点**：`ms/step` 是延迟指标。延迟降低 25× 与吞吐提高 25× 是两回事——如果优化手段是「减少每步的同步等待」，那么延迟的改善会远大于吞吐，因为吞吐还受并行度约束。

### 9.2 三项优化各自在做什么

README 列出的三项可以从名字还原：

1. **slimming**：精简仿真器每步要传递的状态。BEHAVIOR 的观测很大，全局精简与按需取子集能省下大量 CPU 时间
2. **on-demand observation**：按需渲染，不是每步都渲染全部相机视角
3. **hybrid pipeline parallelism**：把仿真步与策略前向流水起来

**第三项本质就是第 3.3 节讨论的「跨环境重叠」**，只是做在了仿真器内部。

### 9.3 论文口径的对比数字

论文与 v0.1 release note 里给出的「相对 veRL / slime 1.07×–1.70×（推理 RL）」属于**[论文] 口径**，必须注意两点：

1. 基线是那两个框架对**同一类任务**的开销，而那两个框架的具身支持本来就弱
2. 这个数字与 25× 不可叠加——一个横比框架、一个纵比仿真器

### 9.4 本机无法实跑

本文所有性能数字都来自官方 README 或论文，属于 `[官方]` / `[论文]` 口径，**没有本机实测**。且 RLinf 的典型工作负载需要 GPU + 仿真器 + 渲染，即使有 GPU 也难以在单机上复现 BEHAVIOR 的配置。

---

## 10. 小结：它到底解决了什么、还剩什么

### 10.1 解决到的程度

| 问题 | 解决程度 |
| --- | --- |
| 仿真器与模型的异构调度 | **好**：独立的 env worker 类别 + 三类实现 + 硬件抽象层 |
| 多轮/episode 状态保持 | **好**：`Channel` 的 key 路由让数据回到持有状态的 worker |
| 异构硬件（NVIDIA / ROCm / 昇腾 / MUSA） | **好**：`hardware` 子模块专门抽象 |
| 与主流仿真器集成 | **好**：Isaac Lab v3.0 官方采用、LIBERO / ManiSkill3 / RoboTwin 均有接入 |
| 真机在线 RL | **可用**：`rtc_env_worker` 有实时约束处理，但生态案例仍少 |
| 异步下的算法正确性 | **未解决**：具身 staleness 的修正方法仍无公认答案 |
| 上手门槛 | **偏高**：需要理解四个调度概念 |

### 10.2 与 LLM 框架的三条本质差异

把第 1、3、5 节的结论压缩成三条：

1. **调度单位从「一批请求」变成「一组环境 + 一步仿真」**，因此必须有跨环境的流水，而不是跨 step 的重叠
2. **环境是一等资源**，参与放置与生命周期管理，而不是一个 reward 函数
3. **权重同步不是瓶颈**，因此不该拿 LLM 框架的同步指标来比较

### 10.3 什么时候选它

**适合**：做 VLA / 具身 RL、需要仿真器与真机混合训练、需要 Isaac Lab 生态、关心多轮 agent 与具身统一抽象。

**不适合**：纯 LLM 后训练（它的调度层是为此设计的额外负担）、单机小规模实验（抽象成本高）、需要现成的 GRPO/DAPO 全家桶（算法覆盖不如 veRL）。

---

**上一篇**：[04 AReaL 全异步架构](/2026/10/24/rl-frameworks-04-areal/) ｜ **下一篇**：[06 SGLang 作为 rollout 引擎](/2026/10/24/rl-frameworks-06-sglang/)

## 参考

- 仓库：<https://github.com/RLinf/RLinf>，基准 `v0.4` = `2348ece3df113874671478be1360ae3115b346e3`（2026-10-20）
- 论文：RLinf，[arXiv 2509.15965](https://arxiv.org/abs/2509.15965)
- 调度层：[`rlinf/scheduler/`](https://github.com/RLinf/RLinf/tree/2348ece3df113874671478be1360ae3115b346e3/rlinf/scheduler)
- `Channel`：[`rlinf/scheduler/channel/channel.py`](https://github.com/RLinf/RLinf/blob/2348ece3df113874671478be1360ae3115b346e3/rlinf/scheduler/channel/channel.py#L43-L75)
- 放置策略：[`rlinf/scheduler/placement/flexible.py`](https://github.com/RLinf/RLinf/blob/2348ece3df113874671478be1360ae3115b346e3/rlinf/scheduler/placement/flexible.py#L27-L45)
- 编排器初始化：[`rlinf/runners/embodied_runner.py` L176-L198](https://github.com/RLinf/RLinf/blob/2348ece3df113874671478be1360ae3115b346e3/rlinf/runners/embodied_runner.py#L176-L198)
- BEHAVIOR 系统优化：[官方博客](https://rlinf.readthedocs.io/en/latest/rst_source/resources/blog/behavior_system_optimization.html)
