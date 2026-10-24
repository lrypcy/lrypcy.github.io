---
title: "RL 训练框架与推理引擎的两族横评（2026）：九个维度、两把尺子、一张选型表"
date: 2026-10-24 18:00:00 +0800
categories:
  - 强化学习
tags: [rl, framework, comparison, megatron, slime, verl, areal, rlinf, vllm, sglang, selection, benchmark, community, survey]
layout: post
mathjax: true
---

> **系列导航** ｜ 本文是本系列的收口篇。逐篇：[01 Megatron 训练栈](/2026/10/24/rl-frameworks-01-megatron/) ｜ [02 slime](/2026/10/24/rl-frameworks-02-slime/) ｜ [03 veRL](/2026/10/24/rl-frameworks-03-verl/) ｜ [04 AReaL](/2026/10/24/rl-frameworks-04-areal/) ｜ [05 RLinf](/2026/10/24/rl-frameworks-05-rlinf/) ｜ [06 SGLang 接缝](/2026/10/24/rl-frameworks-06-sglang/) ｜ [07 vLLM 接缝](/2026/10/24/rl-frameworks-07-vllm/) ｜ 全景（不读源码的版本）：[RL 训练框架横评](/2026/09/29/rl-training-frameworks-survey/) ｜ 引擎架构：[推理引擎深度对比（01）](/2026/10/18/serving-01-architecture-divergence/)

**TL;DR**

> * **两族不能放一张表比。** 训练框架与推理引擎优化的目标函数不同：前者是「一次 RL step 的墙钟时间」，后者是「单位显存下的 token 吞吐」。把 veRL 和 vLLM 放同一列比较，比出来的既不是 veRL 也不是 vLLM，而是「谁恰好更适合被当成 rollout 后端」。
> * **正确比法是三张表**：训练框架族内部横评、推理引擎在「挂 RL 的适配度」这一维上的横评、以及**接缝表**（谁配谁最省事）。
> * **训练框架族的分水岭不是性能，是「编排层的厚度」**。按这个维度排开是连续的：slime（几乎零编排）→ AReaL（服务解耦 + 队列）→ RLinf（自建完整调度层）→ veRL（WorkerGroup + ResourcePool + hook）。厚度决定表达能力，也决定调试成本。
> * **推理引擎族的分水岭是「让不让显存」**。vLLM 的 sleep mode 有两个等级、五种能力标记、可插拔后端；SGLang 有三档暂停语义与可查询的权重版本。**一个擅长让显存，一个擅长换权重**，这决定了它们的默认搭档。
> * **异步是 2026 的主轴，但四个框架落在四个不同的级**：slime 的部分异步、veRL 的 off-policy 异步（V1 三条路径）、AReaL 的全异步、RLinf 的具身长尾异步。**没有一个是「更好的异步」，只有「延迟容忍度是否匹配」**。
> * **执行层四维比评估层四维更能决定成败**：单机 8 卡能否跑通、能否锁版本复现、一个 worker 挂了是整轮重跑还是续上、要不要改模型代码。这四条里任何一条不达标，前四条评估维度写得再高也用不上。
> * **三条选型的反直觉结论**：① 模型族不在支持列表里时，框架的「性能优势」全部作废；② `experimental` 目录的稳定性要看 import 关系而不是目录名（veRL 的 `experimental/agent_loop` 比 `trainer/ppo/v1/` 里多数东西更稳定）；③ 分离部署的瓶颈往往不是权重同步，而是**版本传播的一窗口期**——这一条 AReaL 与 SGLang 解决得比别家好。

---

## 目录

- [1. 为什么必须分成两族比](#1-为什么必须分成两族比)
- [2. 维度定义：评估层四维 + 执行层四维](#2-维度定义评估层四维--执行层四维)
- [3. 训练框架族：按编排层厚度排开](#3-训练框架族按编排层厚度排开)
- [4. 模型支持程度：最容易一票否决的维度](#4-模型支持程度最容易一票否决的维度)
- [5. 异步形态的四级归位](#5-异步形态的四级归位)
- [6. 推理引擎族：只评「挂 RL 的适配度」](#6-推理引擎族只评挂-rl-的适配度)
- [7. 接缝表：谁配谁最省事](#7-接缝表谁配谁最省事)
- [8. 社区活跃度快照与解读](#8-社区活跃度快照与解读)
- [9. 选型决策树与三张反直觉结论](#9-选型决策树与三张反直觉结论)
- [10. 2026 的共同未解问题](#10-2026-的共同未解问题)

---

## 1. 为什么必须分成两族比

### 1.1 目标函数不同

| | 训练框架族 | 推理引擎族 |
| --- | --- | --- |
| 优化目标 | 一次 RL step 的墙钟时间 | 单位显存下的 token 吞吐 |
| 时间尺度 | 一个 step（分钟级） | 一次请求（毫秒到分钟） |
| 关键约束 | 训练与生成的**资源交替** | 显存与带宽的**持续占用** |
| 失效模式 | 训练崩、不收敛、权重不同步 | 请求超时、OOM、吞吐塌陷 |
| 它是不是「框架」 | 是（编排 + 算法 + 训练执行） | **不是**（它是被编排的一方） |

**第五行是分族的根本理由**：推理引擎不是一个能独立跑 RL 的东西，它一定被某个训练框架调用。所以「vLLM 和 veRL 哪个好」这个问题本身不成立，能问的只有「vLLM 作为 veRL 的 rollout 后端好不好」。

### 1.2 三张表而不是一张

| 表 | 比什么 | 结论形态 |
| --- | --- | --- |
| 训练框架族 | 编排能力、算法覆盖、模型支持、活跃度 | 「选哪个框架」 |
| 推理引擎族 | 换权重、让显存、数值透明、组采样 | 「选哪个引擎」 |
| 接缝 | 组合成本、接口对齐程度 | 「选哪个组合」 |

**第三张表是本文独有的**。逐篇看下来会发现，真正决定工程体验的不是单个组件的能力，而是**组合起来之后的接缝数量**——每多一个接缝就多一类只能靠实测发现的 bug（详见第 7 节）。

---

## 2. 维度定义：评估层四维 + 执行层四维

### 2.1 评估层（原目标里列的四维）

| 维度 | 判据 | 怎么量 |
| --- | --- | --- |
| **易用性** | 从 git clone 到跑通一个例子要几步 | 官方 quickstart 的步骤数、是否需要改 yaml |
| **使用复杂度** | 读者必须理解几个概念才能改代码 | 核心抽象个数（Worker / Placement / Channel / Trainer…） |
| **模型支持程度** | 现成支持的模型族与并行度 | 是否有转换器、是否支持 MoE / VLM / VLA |
| **社区活跃度** | 可核验的四个数字 | 见第 8 节快照 |

### 2.2 执行层（本文补充的四维）

这四条才真正决定「能不能用起来」：

| 维度 | 判据 | 为什么它更重要 |
| --- | --- | --- |
| **单机 8 卡可跑** | 官方是否有单机 recipe | 决定 90% 的团队能否验证 |
| **可复现性** | 能否锁版本跑出同样数字 | 决定实验结果是否可信 |
| **故障行为** | worker 挂了 → 整轮重跑 / 续上 / 静默挂死 | 决定能否把一次训练真的跑完 |
| **侵入性** | 要不要改模型代码、要不要自己写环境 | 决定迁移成本 |

**这四条的判据都是「能不能」，不是「好不好」**——它们是一票否决型维度，而前四条是排序型维度。把两类混在一起打分是选型失败的第二大来源。

### 2.3 不可比声明（必须放在每个数字旁边）

本系列所有性能数字分四级：`[官方]` / `[论文]` / `[第三方]` / `[本机]`。**本系列不存在 `[本机]`**——本机没有 GPU 环境，全部是引用。

跨框架比较只有在**同一份 benchmark 内部**才成立（例如 AReaL 论文里对 veRL / slime 的同一任务对比）。其余一律标注为「厂商口径」，不可横向比较。不可比的原因至少有六个：GPU 型号、并行度、是否 colocate、`rollout.n`、序列长度、是否含首次编译。

---

## 3. 训练框架族：按编排层厚度排开

### 3.1 厚度谱

把五个框架按「自己造了多少编排层」排开，得到的是一条连续的谱：

```text
zero orchestration                                            full orchestration
      |                                                              |
   slime  ────────  AReaL  ────────  RLinf  ────────  veRL
   (Ray 壳)      (双服务+队列)    (自建调度器)    (WorkerGroup+
                                                   ResourcePool+hook)
```

| 框架 | 编排层规模 | 核心抽象 | 表达能力 | 调试成本 |
| --- | --- | --- | --- | --- |
| **slime** | `slime/ray/` 六个文件 | `RayTrainGroup` + rollout manager | 布局只有 `--colocate` 布尔 | **最低**：主循环 50 行可直接读 |
| **AReaL** | 双控制器 + 三层队列 | `RolloutController` / `TrainController` | 全异步、staleness 可调 | 中：要理解三个队列 |
| **RLinf** | `rlinf/scheduler/` 八个子模块 | `Worker` / `WorkerGroup` / `Placement` / `Channel` | 异构硬件、仿真器一等公民 | 中高：四个概念 |
| **veRL** | `single_controller` + V1 trainer | WorkerGroup / ResourcePool / decorator | **最强**：colocate / 分离 / 混合 | **最高**：调的是 placement 与 resharding |

### 3.2 这张谱的读法

**越靠左，代码越可读、越容易被 fork；越靠右，能力越强、越难 debug。**

具体到每个框架的一句话：

- **slime**：把复杂度推给 Megatron 与 SGLang，自己只留数据流。**代价是模型族与引擎都不可换**。
- **AReaL**：把「训练端与推理端是两个服务」做成了架构前提，而不是一个开关。**代价是同步模式反而是退化路径**（官方自述同步比异步慢约 2 倍 `[官方]`）。
- **RLinf**：编排层的厚度与 veRL 同级，但服务的是另一个问题——**异构硬件**（仿真器、真机）的放置。
- **veRL**：编排层本身成了框架的核心资产。**代价在 [03 veRL](/2026/10/24/rl-frameworks-03-verl/) 里写得很直接**：数据不再经控制器所以崩了看不见、完成顺序不确定所以要严格复现得退回 V0 训练器。

### 3.3 评估层四维的打分

| 维度 | Megatron 原生 | slime | veRL | AReaL | RLinf |
| --- | --- | --- | --- | --- | --- |
| **易用性** | 低（要写训练脚本） | **高** | 中 | 中 | 低 |
| **使用复杂度** | 高（14 篇才讲完） | **低** | 高 | 中 | 中高 |
| **模型支持** | 广（需 Bridge 转换） | 窄（逐模型转换器） | **最广** | 中 | 窄（具身为主） |
| **社区活跃度** | 高（NVIDIA） | 中 | **最高** | 中 | 中 |

**注意「易用性高」与「使用复杂度低」不是同一件事**：slime 两者都好，是因为它真的少写代码；veRL 是易用性中等（quickstart 能跑）但复杂度高（改算法要理解 placement）。

---

## 4. 模型支持程度：最容易一票否决的维度

### 4.1 支持方式决定迁移成本

四个框架对「一个新模型族」的支持方式完全不同：

| 框架 | 支持方式 | 加一个新模型要做什么 | 风险 |
| --- | --- | --- | --- |
| **Megatron + Bridge** | 模式匹配 + 自动推断（`MegatronMappingRegistry`） | 通常只需 provider 声明 | **自动推断错了会静默错位** |
| **slime** | **逐模型写转换器**（`megatron_to_hf/<model>.py`） | 写一个文件 | 显式，错了能看出来 |
| **veRL** | 后端抽象（FSDP / Megatron 双路径） | 取决于后端 | 两条路径行为可能不同 |
| **RLinf** | 模型抽象（`rlinf/models/`） | 实现模型接口 | 生态集中在 VLA |

**这里有一个与直觉相反的判断**：Bridge 的「自动」并不总是更省事。`QKVMapping` 的文档里提到一处限制：

> For regular QKV modules, gate rows are appended after the standard GQA-interleaved QKV block. For Megatron-Core `attention_output_gate` modules, the scalar gate is expanded across each head dimension and interleaved as the module's output-gate block.

**同名的 `linear_qkv` 在两个 attention 模块里可能是两种布局**。自动推断必须知道「当前用的是哪个 provider」，这就是 `AutoMapping` 存在的原因。**自动化的代价是把「显式写错」换成了「自动推断错」**，后者的排查成本更高。

### 4.2 一票否决判据

下列任一条不满足，框架的其他优势全部作废：

| 判据 | 不满足的后果 |
| --- | --- |
| 你的模型族有现成转换器 / provider | 要自己写，工作量集中在 QKV 交错与 MoE 专家 |
| 你要的并行度组合被支持 | 例如 `num_query_groups` 不能被 TP 整除时的静默退化 |
| 你要的推理引擎被支持 | slime 只接 SGLang，换 vLLM 要重写整个 `sglang_utils` |
| 你的 dtype 组合被支持 | fp8 KV cache、fp4 权重的组合 | 

---

## 5. 异步形态的四级归位

### 5.1 归位表（跨篇统一）

| 形态 | 判据 | Megatron | slime | veRL | AReaL | RLinf |
| --- | --- | --- | --- | --- | --- | --- |
| **同步** | 每步 rollout 全完成后才更新 | 默认 | 默认 | `trainer_mode=sync` | `max_head_offpolicyness=0` | `embodied_runner` |
| **部分异步** | 在途生成跨步续用，权重更新仍是全局屏障 | — | `--partial-rollout` | `colocate_async` | — | — |
| **off-policy 异步** | 训练第 $$k$$ 步时 rollout 在产第 $$k+1$$ 步 | RolloutPipeline | `--fully-async` | `fully_async_policy` / `one_step_off_policy` / `separate_async` | — | `async_embodied_runner` |
| **全异步** | 训练端与推理端是两个独立服务 | — | — | `separate_async` + `hybrid_engine=False` | **原生形态** | — |

### 5.2 读这张表的三条注意事项

**第一，veRL 的 `separate_async` 不是纯的第四级。** [03 veRL](/2026/10/24/rl-frameworks-03-verl/) 的判定是：它默认形态（hybrid replicas 开着）介于第三与第四之间——有独立的 standalone rollout 池，同时训练卡上还挂着 hybrid 副本在 idle 时借给生成。**它是把第三级和第四级拼在一起**。

**第二，AReaL 的「同步」是退化路径而不是默认路径。** 把 `max_head_offpolicyness` 设成 0 就退化成同步，但官方文档自述同步比异步慢约 2 倍 `[官方]`。**这说明它的同步路径不是为性能设计的，只是为正确性兜底。**

**第三，slime 的全异步不换控制流。** 它换的是 rollout 后端（`generate_rollout_fully_async` 替换默认生成逻辑），主循环、权重同步、checkpoint 全复用。**这是「薄编排」架构带来的可组合性红利**——换异步程度不需要改控制流。

### 5.3 延迟容忍度匹配表

异步选型的真正判据不是「哪个更先进」，而是**你的延迟容忍度**：

| 你的情形 | 该落到哪一级 |
| --- | --- |
| 小模型、单机、要严格复现 | 同步 |
| 长 CoT 但能接受同一步内的续用 | 部分异步 |
| 多机、生成占绝对大头、能接受 off-policy 修正 | off-policy 异步 |
| 推理端有独立资源池、想要稳定的服务化 | 全异步 |
| 具身 episode 长尾 | 具身异步（RLinf 那一档） |

---

## 6. 推理引擎族：只评「挂 RL 的适配度」

### 6.1 六个子维度

本节不评引擎的推理性能（那是 `serving` 系列的事），只评「把它当 rollout 后端」的六个子维度：

| 子维度 | vLLM | SGLang | 判据 |
| --- | --- | --- | --- |
| **换权重** | 三条路径（tensor / distributed / disk） | 三条路径 + **显式建组协议** | 是否支持 colocate 与分离两种 |
| **让显存** | **最强**：sleep mode 两级 + 可插拔后端 + 五种能力标记 | 无对等实现 | colocate 是否可行 |
| **暂停语义** | 主要是 abort | **三档**（abort / retract / in_place） | 能否保住在途请求 |
| **版本可见** | 无统一字段 | **`weight_version` 可查询** | 多引擎时能否确认全部到位 |
| **数值透明** | 配置项齐全 | 配置项齐全 | 需框架侧保证 |
| **组采样** | 引擎无组概念 | 引擎无组概念 | 必须框架侧展开 |

### 6.2 两种「擅长」

把六个子维度压缩成一句话：

- **vLLM 擅长「让显存」**：`SleepModeBackend` 是一个带五类能力标记的可插拔后端（RFC #34303），它的 `preserves_communicators()` / `preserves_compiled_artifacts()` / `preserves_graphs_with_communicators()` / `supports_durable_storage()` 四个查询，回答的正是「唤醒后要重建什么」。**这是 colocate 场景的核心能力**。
- **SGLang 擅长「换权重」**：三档暂停语义 + 显式建组协议 + 可查询的 `weight_version`。**这是分离部署的核心能力**。

### 6.3 一张容易被忽略的表：sleep 的两个等级

vLLM 的 `sleep(level)` 不是「省不省显存」的二选一，而是两种完全不同的机制：

| | level 1 | level 2 |
| --- | --- | --- |
| 权重 | 保留在可被他人占用的池里 | **先备份到 CPU 再释放** |
| 唤醒代价 | 小 | **一次完整的 CPU → GPU 拷贝** |

**选错等级的后果是灾难性的**（slime 的 [02](/2026/10/24/rl-frameworks-02-slime/) 与 [07 vLLM](/2026/10/24/rl-frameworks-07-vllm/) 都讲了这一点）：level 1 留下的权重残留会让训练端 OOM，level 2 的完整回拷如果之后立刻又睡就白拷了一次。

---

## 7. 接缝表：谁配谁最省事

### 7.1 三个矩阵

**矩阵一：训练框架 × 推理引擎的现成支持**

| | vLLM | SGLang | Megatron 原生生成 |
| --- | --- | --- | --- |
| **veRL** | 支持（一等公民） | 支持（一等公民） | 支持（Megatron 后端） |
| **AReaL** | 支持（服务解耦） | 支持（默认） | — |
| **slime** | 不接 | 支持（**深度耦合**） | 不支持 |
| **RLinf** | 支持（`hybrid_engines`） | 支持 | — |
| **Megatron 原生 RL** | 不支持 | 不支持 | 支持（**根本不做同步**） |

**矩阵二：接缝数量**

每个「训练端 → 推理端」的连接点都对应一类只能靠实测发现的 bug。数一数：

| 组合 | 接缝 | 潜在 bug 类别 |
| --- | --- | --- |
| slime + SGLang | 权重格式（逐模型转换器）+ IPC + 暂停 + 版本 | 4 类 |
| veRL + vLLM | 权重格式 + 并行重排 + rpc + 数值 | 4 类 |
| AReaL + SGLang | 队列 + 权重 + staleness 修正 | 3 类（队列本身是一个接缝） |
| **Megatron 原生 RL** | **零** | **0 类** |

**Megatron 原生 RL 栈的「零接缝」是它唯一但决定性的优势**：训练与推理同进程、同格式、同布局，3D-HybridEngine 那笔搬运账直接归零。代价是 HF 模型必须经 Bridge 转换、算法与环境基本要自己写。

**矩阵三：默认搭档**

| 框架 | 默认搭档 | 为什么 |
| --- | --- | --- |
| slime | **SGLang** | 深度耦合，迁移成本最高 |
| veRL | vLLM 或 SGLang | 两者都是一等公民 |
| AReaL | **SGLang** | 服务解耦形态与 SGLang 的建组协议契合 |
| RLinf | 均可 | 有引擎无关适配层 |

### 7.2 接缝的三种典型失效

从逐篇的源码里能提炼出三类反复出现的失效模式：

| 失效 | 表现 | 出处 |
| --- | --- | --- |
| **缓冲复用竞态** | 权重静默污染，所有 barrier 通过，只有训练曲线歪 | slime `_build_flattened_tensor_data` 的注释 |
| **`rank_offset` 算错** | NCCL 挂住或行为诡异，报错位置离根因很远 | SGLang 建组协议 |
| **refit 无端到端校验** | `update_weights` 返回成功 ≠ 权重到位 | AReaL §10 与 veRL §10 都把它列为未解决 |

**第三类是最普遍的**：veRL 的「refit 后没有框架层校验」与 AReaL 的「权重层没有端到端校验」是同一个问题的两种表述。**唯一能证明权重真的同步到位的动作是逐项比对或算 logprob**，而绝大多数框架默认不做。

---

## 8. 社区活跃度快照与解读

### 8.1 快照（采集于 2026-10-24）

| 仓库 | stars | open issues | 最近 90 天 commit | 90 天贡献者 | 最新 release |
| --- | --- | --- | --- | --- | --- |
| vllm | 95,268 | 9,377 | 729 | 343 | `v0.32.1rc0` 2026-10-18 |
| sglang | 37,688 | 6,136 | 4,327 | 665 | `v0.5.22` 2026-10-13 |
| verl | 23,998 | 1,386 | 219 | 97 | `v0.9.1` 2026-09-20 |
| trl | 19,594 | 278 | — | — | `v1.16.0` 2026-10-15 |
| Megatron-LM | 18,257 | 1,696 | 48 | 20 | `core_v0.19.2` 2026-09-18 |
| OpenRLHF | 10,114 | 403 | — | — | `v0.11.2` 2026-09-14 |
| slime | 8,727 | 549 | 76 | 26 | `v0.4.0` 2026-10-09 |
| AReaL | 5,866 | 141 | 45 | 14 | `v2.1.0` 2026-08-25 |
| RLinf | 5,488 | 311 | 131 | 53 | `v0.4` 2026-10-20 |
| Megatron-Bridge | 969 | 492 | 140 | 16 | `v0.6.2` 2026-09-18 |

> 90 天 commit 数按各仓库的目标 commit 计量，不是按 `main` 头部。

### 8.2 三个解读（不要只看 stars）

**第一，「stars / open issues」比值反映维护压力而不是热度。**

| 仓库 | stars / open issues | 读法 |
| --- | --- | --- |
| AReaL | 41.6 | **最高**：issue 消化得最好 |
| slime | 15.9 | 中上 |
| trl | 70.5 | 高，但仓库定位不同 |
| veRL | 17.3 | 中 |
| RLinf | 17.6 | 中 |
| vLLM | 10.2 | **偏低**：用户基数太大，issue 必然堆积 |

这个比值不能直接当质量指标——**用户量越大的项目这个比值越差**，因为提交 issue 的门槛远低于提 PR 的门槛。

**第二，Megatron-Bridge 是一个值得警惕的数字。** 969 stars、492 open issues、90 天仅 16 个贡献者。**它是全表里唯一「依赖面极广但社区极小」的组件**——所有走 Megatron 路线的 RL 框架（slime、veRL 的 Megatron 后端、AReaL 的 megatron_utils）都间接依赖它。这是当前 RL 栈的一个结构性风险点。

**第三，commit 数的绝对值会骗人。** SGLang 90 天 4,327 commit 是因为它包含大量测试与 CI 改动，不能与 veRL 的 219 直接比。**有意义的是「最近 release 距今多久」**——AReaL 的 `v2.1.0` 距今两个月，是五个训练框架里最久的。

### 8.3 活跃度的实战判据

不看数字，看三个具体的东西：

1. **最近三个月有没有新的 release**（有 → 有人在维护发布流程）
2. **open issue 里有没有 maintainer 在回应**（随机抽 3 个看回复时间）
3. **`experimental` 目录是不是在动**（动 → 有路线图；不动 → 可能已停滞）

---

## 9. 选型决策树与三张反直觉结论

### 9.1 决策树

```text
你有 GPU 集群吗？
├─ 没有 → 用 TRL（单机、单进程、无需编排）
└─ 有
   ├─ 你要训的是 VLA / 具身模型？
   │  └─ 是 → RLinf
   ├─ 你的模型族在 Megatron 支持列表里，且已有 Megatron 预训练栈？
   │  ├─ 是 → 训练侧用 Megatron（原生 RL 栈，零接缝）
   │  │       └─ 需要现成算法与生态 → slime（若模型族支持）或 veRL
   │  └─ 否 → veRL（FSDP 路径，无需 Bridge 转换）
   ├─ 你的推理端是独立资源池、要服务化？
   │  └─ 是 → AReaL（全异步原生）
   └─ 你只要「能跑通 GRPO」且不想理解 placement？
      └─ slime
```

### 9.2 反直觉结论一：性能优势会被模型支持度一票否决

**现象**：很多团队会先比 benchmark，再选框架。

**问题**：模型族不在支持列表里时，所有性能数字都是别人的机器的数字。**加一个模型族的工作量（写转换器、调 QKV 交错、验证 MoE 专家映射）往往比选错框架带来的损失更大。**

**判据**：**先确认支持，再看性能**。支持列表是第一道筛子，不是最后一道。

### 9.3 反直觉结论二：`experimental` 目录的稳定性要看 import 关系

**现象**：看到 `experimental/` 前缀就认为不稳定。

**事实**（见 [03 veRL](/2026/10/24/rl-frameworks-03-verl/) 的量化表）：

| 子目录 | 被主流程 import？ | 真实稳定性 |
| --- | --- | --- |
| `experimental/agent_loop` | **是** | **比 `trainer/ppo/v1/` 里多数东西更稳定** |
| `experimental/fully_async_policy` | 否 | **已被弃用，下版本迁出** |

**判据：只看 import 关系，不看目录名。** 被主流程 import 的 `experimental` 模块只是还没搬家。

### 9.4 反直觉结论三：分离部署的瓶颈往往不是权重同步

**现象**：讨论分离部署时，大家默认瓶颈是「把 144 GB 权重搬过去」。

**事实**：真正难的是**版本传播的窗口期**——多个推理引擎实例是逐个推送的，中间窗口里不同实例是不同版本。这段时间产出的样本，它们的版本标记是错的。

**谁解决得更好**：

| 方案 | 机制 |
| --- | --- |
| SGLang | `weight_version` 可逐个查询 + `abort_all_requests` 默认 True |
| AReaL | 逐 token 的版本记录，$$\eta$$ 可量化 |
| veRL | `trajectory_spans` / `trajectory_staleness` 把「轨迹内不一致」与「整体陈旧」分开计量 |
| slime | `weight_version` 递增 + `pause_rollout_admission` |

**判据**：选分离部署的框架时，问一个具体问题——**「权重更新到一半时产出的样本，你们怎么打版本标记？」** 回答不上来的框架，意味着它的版本一致性只能靠「更新期间不接单」这种粗放手段。

---

## 10. 2026 的共同未解问题

逐篇看下来，五个问题在所有框架里都还没解决。**它们是选型时要问的自检问题，而不是某一家的缺陷**。

### 10.1 refit 没有框架级端到端校验

- veRL：`update_weights` 返回只保证传输完成
- AReaL：权重层没有端到端校验
- slime：**有** `check_weights(action="compare")`，但默认不在每步跑

**为什么难**：比对全部参数本身的成本与同步相当，等于把同步做两遍。

**务实的解法**：抽样比对 + 定期全量。**但这需要框架提供抽样接口，目前只有 slime 有。**

### 10.2 严格可复现与高性能训练器互斥

veRL 的官方文档直接写明：`trainer.use_v1` 一行是 `Must be false under full_determinism`，因为 V1 按完成顺序收集输出。**这不是 veRL 的缺陷，是异步/乱序设计的必然代价。**

**对所有框架都成立**：只要训练器按「完成的先后」组批，批的组成就不可复现。

### 10.3 崩溃窗口内的轨迹丢失

veRL 的 RFC #7579 描述得很清楚：V1 能在 checkpoint 边界快照 TransferQueue、重启后重发在途 prompt，**但上次 checkpoint 之后产出的全部轨迹都丢了**。而且提高 checkpoint 频率不能缩小窗口，因为快照时要停下来拷贝 buffer。

**这是所有异步框架的共同问题**：轨迹在队列里，队列在内存里。

### 10.4 静默失败多于显式失败

三个例子：

1. 资源不足 → driver 无限静默轮询（veRL issue #7656，已修但暴露了模式）
2. 权重错位 → 所有 barrier 通过，只有训练曲线歪（缓冲复用竞态）
3. `num_query_groups` 与 TP 不整除 → KV 头重复，不报错

**共同点：失效不体现在报错上，体现在收敛曲线上。** 这意味着排查的第一步总是「先证明不是同步问题」，而这个证明目前只能靠人。

### 10.5 MoE 的数值稳定性

- veRL：项目与社区都在做，但官方 roadmap 里多项仍 WIP
- slime：Routing Replay 实现较深，MoE 权重同步有独立的 `_expert_transfer_plan`
- 通用做法：Routing Replay（回放路由）或序列级重要性比（GSPO）

**这一条的现状是「有解但都不完美」**：Routing Replay 要额外显存存路由，GSPO 改的是算法而非系统。

### 10.6 小结判据表

| 问题 | 有解吗 | 谁能解 | 代价 |
| --- | --- | --- | --- |
| refit 端到端校验 | **半解** | slime 有接口 | 需定期跑，成本高 |
| 严格可复现 | **有解（退到同步）** | 所有 | 放弃乱序带来的吞吐 |
| 崩溃丢轨迹 | **未解** | 无人 | — |
| 静默失败 | **未解** | 无人 | — |
| MoE 数值稳定 | **有解但都不完美** | Routing Replay / GSPO | 显存或算法改动 |

---

## 11. 收口：一句话选型表

| 你的情形 | 选 | 一句话理由 |
| --- | --- | --- |
| 单机、单卡、跑通 GRPO | TRL | 无需编排 |
| 已有 Megatron 预训练栈、模型族在支持列表 | **Megatron 原生 RL** | 零接缝 |
| 要现成算法全家桶、能接受调试 placement | **veRL** | 表达能力最强 |
| 要代码可读、能 fork、模型族受限但够用 | **slime** | 最薄、最可读 |
| 推理端已服务化、要全异步 | **AReaL** | 全异步原生 |
| 做 VLA / 具身 | **RLinf** | 唯一把仿真器当一等资源 |
| colocate 且显存紧张 | vLLM 作后端 | sleep mode 两级 |
| 分离部署且要精确换权重 | SGLang 作后端 | 三档暂停 + 版本可查 |

**最后一条判据**：如果你的候选框架之间差别在你阈值之内，**选那个你有能力 debug 的**。本系列七篇逐篇读下来，真正决定工程成败的不是 benchmark 上的百分之几，而是**出事时你能不能在自己的代码里找到那一行**。

---

**上一篇**：[07 vLLM 作为 rollout 引擎](/2026/10/24/rl-frameworks-07-vllm/) ｜ **系列导航**：[01 Megatron](/2026/10/24/rl-frameworks-01-megatron/) ｜ [02 slime](/2026/10/24/rl-frameworks-02-slime/) ｜ [03 veRL](/2026/10/24/rl-frameworks-03-verl/) ｜ [04 AReaL](/2026/10/24/rl-frameworks-04-areal/) ｜ [05 RLinf](/2026/10/24/rl-frameworks-05-rlinf/) ｜ [06 SGLang](/2026/10/24/rl-frameworks-06-sglang/)

## 参考

本系列逐篇的源码引用与基准 commit 见各自「参考」节。活跃度快照采集于 2026-10-24。

- veRL：[RFC #5400](https://github.com/verl-project/verl/issues/5400)、[roadmap #4880](https://github.com/verl-project/verl/issues/4880)、[RFC #7579](https://github.com/verl-project/verl/issues/7579)、[issue #7656](https://github.com/verl-project/verl/issues/7656)
- vLLM sleep mode backend：RFC #34303
- AReaL 论文：[arXiv 2505.24298](https://arxiv.org/abs/2505.24298)
- RLinf 论文：[arXiv 2509.15965](https://arxiv.org/abs/2509.15965)
- HybridFlow 论文：[arXiv 2409.19256](https://arxiv.org/abs/2409.19256)
- 社区活跃度查询：GitHub API，2026-10-24 采集
