---
title: "RL 训练框架全景横评（2026）：从 TRL 到 veRL、AReaL、slime 的设计取舍"
date: 2026-09-29 20:30:00 +0800
categories:
  - 强化学习
tags: [rl, rlhf, rlvr, grpo, verl, openrlhf, trl, nemo-rl, slime, areal, roll, chatlearn, ray, vllm, sglang, megatron, survey]
layout: post
mathjax: true
---

> **系列导航** ｜ 算法侧：[从 MDP 到 GRPO（五）：GRPO 组相对优势](/2026/08/21/mdp-to-grpo-05-grpo-group-relative/) ｜ [RL 信用分配与奖励系数](/2026/09/19/rl-credit-assignment-settlement/) ｜ 变量侧：[RL 框架里的那些变量到底怎么算](/2026/09/30/rl-variables-in-frameworks/) ｜ 系统侧：[Megatron-LM 深度剖析（00）](/2026/09/28/megatron-00-foundation/) ｜ [分布式训练（08）：组合实战](/2026/08/31/dist-train-08-combined-practice/)
>
> 本篇讲的是**承载这些算法的工程系统**，与前面几篇的算法视角互补。

**TL;DR**

> * **RL 训练框架不是「一个 Trainer」，而是四个正交选择轴的叉积**：控制器范式（单控 / 多控 / 混合）、资源布局（colocate / 分离 / 自动映射）、权重同步（进程内 reshard / CUDA IPC / NCCL / checkpoint reload）、时间模型（同步 / 异步 / 流式）。看懂一个框架 = 在这四根轴上定位它。
> * **瓶颈从来不在梯度**。OpenRLHF 论文实测生成占 RLHF 总时长 **>90%**；NeMo-Aligner 70B 单步 53.7s 里，生成 45.2s、训练 8.5s。所以框架竞争的胜负手是**推理引擎的接入深度**和**权重搬运的代价**，不是谁的优化器写得快。
> * **本文跑的量化模型显示**：同步屏障下，长尾输出造成的槽位空闲率上界约 **41%（短 CoT）→ 56%（agentic）**；把步时预算压到同步步时的 60% 做截断，完成率仍有 94–97%、有效吞吐 **1.6×**。这就是所有框架 2025–2026 集体转向异步与 partial rollout 的量化理由。
> * **经典 RL 框架（RLlib / SB3）套不到 LLM 上**，本质原因是它们把采样抽象成 `env.step()`，而 LLM 的「采样」是**一次带 continuous batching 的自回归生成**，且策略权重每步都变、还要在两种并行布局之间搬 140 GB。RLlib 的 `num_env_runners` 表达不了这件事。
> * **veRL（HybridFlow）是当前事实标准**：v0.9.0（2026-08-14）统一了同步/异步训练器、默认走 Megatron-Bridge、上了 delta-sharded 权重同步（72B 权重更新提速 3.1×）与硬件插件层（ROCm / 昇腾 / 国产芯片）。代价是调试对象从 Python 控制流变成了 **placement + resharding**。
> * **TRL 走另一极端**：v1.0（2026-03-31）确立稳定层/实验层双轨契约，而 v1.13.0（2026-09-10）**直接把 PPOTrainer 删掉了**——这是「不为昨天的假设付维护成本」的极端表达。单机到三十 B 的 GRPO 首选。
> * **异步是 2026 的主线**：AReaL 全异步 2.77×、veRL fully async 2.35–2.67×@128 GPU、LlamaRL 分离+异步在 405B 上 10.7×。代价是 staleness 引入 off-policy，必须配 TIS / IcePop / CISPO 修正。
> * **MoE 的 RL 有专属坑**：专家路由在一次更新后约 10% 变化，token 级重要性比剧烈抖动。Qwen 用 Routing Replay 兜，GSPO 改序列级重要性比根治，NeMo-RL 至今还在做 R3（Router Replay Rollouts）。
> * **环境接口正在收敛**：OpenEnv（HF，Gymnasium 风格 + MCP 一等公民，2026-06 治理移交多组织委员会）和 Harbor（Terminal-Bench 作者，ATIF 轨迹格式）是两个主要候选，MCP 是实际公约数。
> * **本篇初稿漏掉了 RLinf**（清华 + 无问芯穹 + 中关村学院，[arXiv 2509.15965](https://arxiv.org/abs/2509.15965)），已补。它是唯一把**具身智能（VLA + 仿真器 + 真机）**与 LLM 后训练塞进同一套基础设施的框架，核心是 M2Flow 范式；在 BEHAVIOR-1K 上把端到端延迟从 1028 ms/step 压到 **41 ms/step（25×）**，RLinf-VLA 拿了 CVPR 2026 ScaleBot Workshop 最佳论文。
> * **多模态 RL 有专门的答案**：**EasyR1**（LLaMA-Factory 作者 hiyouga，5.1k★）是 veRL 的干净分叉，补齐了 VLM 原生支持、内置 LoRA、padding-free 与「奖励函数就是一个 `compute_score()`」，20+ 研究项目用它。具身侧还有 **SimpleVLA-RL**（上海 AI Lab，ICLR 2026，二元 0/1 奖励 + 并行渲染）。
> * **veRL 已成为上游底座**：EasyR1、SimpleVLA-RL、RL-Factory、SkyRL、rLLM、verl-agent 全是它的 fork 或扩展。2026 年做 RL 框架的主流动作不是从零写，而是 fork veRL 后垂直做深。
> * **选对框架不等于能收敛**。第 9 章给了九个必看指标与八种失败模式的对照（reward hacking、熵崩塌、长度爆炸、MoE 崩、异步不收敛等）。另外三组纯 numpy 实验（`tools/rl_obj_numeric_lab.py`）把算法差异算成了数字：**难度两极化时约 17% 的 rollout 是零梯度的**（DAPO 的靶心）、**序列级比率把波动压低 71×**（GSPO 免 Routing Replay 的量化理由）、**训推噪声到 0.30 时 27% 的 token 假装越界**。
> * **框架的价值有一半藏在「不出事」的地方**。第 10 章列了七个容易被忽略的系统问题——容错与弹性、检查点语义（在途 rollout 属于谁）、verifier 基础设施、样本管道、系数调度、超长序列、可复现性——每个都附了选型时该问的自检问题。**它们都不影响跑通一个 demo，但决定了能不能把一次训练真的跑完。**

---

## 目录

- [1. 全景：三族框架，别混着比](#1-全景三族框架别混着比)
- [2. 一次 RL 迭代的解剖：钱花在哪](#2-一次-rl-迭代的解剖钱花在哪)
- [3. 四个正交选择轴（全文骨架）](#3-四个正交选择轴全文骨架)
- [4. 主力框架逐个解剖](#4-主力框架逐个解剖)
- [5. 学术界与中国厂商：异步、agentic 与「后 Trainer」形态](#5-学术界与中国厂商异步agentic-与后-trainer-形态)
- [6. 经典 RL 框架：还活着吗](#6-经典-rl-框架还活着吗)
- [7. 三个硬问题的工程解法](#7-三个硬问题的工程解法)
- [8. 算法 × 框架支持矩阵](#8-算法--框架支持矩阵)
- [9. 训练跑歪时看什么：指标与失败模式](#9-训练跑歪时看什么指标与失败模式)
- [10. 框架还该解决什么：七个容易被忽略的系统问题](#10-框架还该解决什么七个容易被忽略的系统问题)
- [11. 选型决策树](#11-选型决策树)
- [12. 2026 的趋势与未解问题](#12-2026-的趋势与未解问题)

---

## 1. 全景：三族框架，别混着比

「RL 训练框架」这个词在 2026 年至少指三种东西，把它们放同一张表里比较是选型失败的第一来源。

| 族 | 代表 | 采样是什么 | 模型规模 | 优化目标 |
|---|---|---|---|---|
| **经典通用 RL** | SB3、CleanRL、RLlib、Acme、TorchRL、Tianshou | `env.step()`，环境是物理/游戏仿真 | 几 M – 几十 M | 环境步进吞吐、算法覆盖 |
| **LLM 后训练 RL** | TRL、OpenRLHF、veRL、NeMo-RL、slime、ROLL、AReaL | vLLM/SGLang 自回归生成 | 几 B – 数千 B | 生成吞吐、权重同步、显存 |
| **Agentic RL Runtime** | verl-agent、rLLM、SkyRL-agent、AReaL v2、prime-rl + verifiers | 多轮工具调用 + 沙箱，轨迹分钟级 | 同上，且轨迹更长 | 异步编排、轨迹协议、I/O |

第三族是 2025 年下半年才真正分出来的。它与第二族的差别不在算法，而在**控制对象**：第二族的 rollout 是「一批独立的 completion」，第三族是「一个有状态的、可能跑几十步的 episode」，中间还要插入工具返回、环境反馈、上下文 compaction。这直接把「同步屏障」这个设计变成了不可接受的开销——于是异步从优化项变成了必需品。

**这个三族划分有一个重要的反例：RLinf**。它同时出现在第一族和第三族里——采样端是 ManiSkill3 / LIBERO / IsaacLab / RoboTwin 这类**物理仿真器**（第一族的地盘），被训练的模型却是 OpenVLA / π₀ / GR00T 这种**数十亿参数的 VLA 大模型**（第二族的地盘），而且已经做到真机上的在线 RL。它证明「仿真器 RL」和「大模型 RL」并不天然分居，只是此前没人把两者塞进同一套基础设施。SimpleVLA-RL（5.7）走的是同一个交叉点的另一条路。详见第 5.1 节。

后面第 4、5、6 章按这三族展开；第 3 章是贯穿它们的解剖框架。

---

## 2. 一次 RL 迭代的解剖：钱花在哪

先说清楚一次迭代到底干什么。以 GRPO 为例（PPO 多一个 critic 前向/反向和 value loss）：

$$\text{step} = \underbrace{\text{rollout}}_{\text{生成}} \to \underbrace{\text{reward}}_{\text{打分}} \to \underbrace{\text{ref/old logprob}}_{\text{前向}} \to \underbrace{A_i}_{\text{优势}} \to \underbrace{\text{update}}_{\text{训练}} \to \underbrace{\text{refit}}_{\text{权重同步}}$$

公开实测的时间分布高度一致地指向同一个结论：

| 来源 | 配置 | 时间分布 |
|---|---|---|
| [OpenRLHF 论文](https://arxiv.org/abs/2405.11143) | RLHF 全流程 | 生成占 **>90%**，模型越大、上下文越长越极端 |
| [NeMo-Aligner 论文](https://arxiv.org/abs/2405.01481) | 70B 单步 53.7s | response 生成 35.8s、logprobs 4.0s、**TRT refit 3.1s**、训练 8.5s |
| [RLHFuse](https://arxiv.org/abs/2409.13221) | PP 流水线 | 流水线气泡可吃掉**高达 50%** 的 GPU 时间 |
| [LlamaRL](https://arxiv.org/abs/2505.24034) | 8B/70B/405B 分离+异步 | 相对同步基线 **2.52× / 3.98× / 10.7×** |

本文的量级模型（脚本已入库：`tools/rl_framework_budget.py`，miniconda Python 实跑；文中所有自算数字都出自它）把这件事拆开。核心是**槽位木桶模型**：4096 条 prompt 随机分到 256 个生成槽位，每槽串行处理 16 条，一步的墙钟 = 最慢槽位的时间。

```
设定：4096 prompts / step，256 槽位（每槽 16 条），单槽 35.0 tok/s

形态                            均值     P50     P95      P99       最长   理想步时   同步步时   空闲率
-----------------------------------------------------------------------------------------------
短 CoT (median 800)           967     808    2175     3176     6139     442.0     754.4   41.4%
长 CoT (median 4000)         6060    3999   17756    32033   101133    2770.1    5456.0   49.2%
agentic (median 12000)     21646   11851   73499   152031   200000    9895.5   22367.1   55.8%
```

**这个 41%–56% 是上界**，因为模型假设槽位之间不能「偷任务」。真实的 vLLM/SGLang continuous batching 会持续把新请求填进空闲槽位，实测浪费会低不少。但趋势是硬的：**输出越长尾，同步屏障越亏**，而且是单调的。这正是长 CoT（2025）和 agentic（2026）两波浪潮各自把框架往前推了一代的直接原因。

第二个自算结果回答「截断划不划算」：

```
形态                             预算占同步步时      完成率       有效吞吐      相对同步
------------------------------------------------------------------------
长 CoT (median 4000)              100%   100.0%      0.751     1.00x
长 CoT (median 4000)               80%    99.3%      0.932     1.24x
长 CoT (median 4000)               60%    95.8%      1.199     1.60x
长 CoT (median 4000)               40%    77.8%      1.460     1.95x
```

结论：**把步时压到 60% 是划算的**（丢 4% 样本换 1.6× 有效吞吐）；压到 40% 就开始伤样本分布了——被截断的偏偏是难题和长题，而它们往往是最有价值的梯度信号。这个 trade-off 正是 [DAPO dynamic sampling](https://arxiv.org/abs/2503.14476) 存在的理由：它不截断，而是**过采样再过滤掉组内全对/全错的 prompt**（优势恒为 0，白算），Qwen2.5-32B 用它在 AIME 2024 拿到 50 分。

> 模型脚本已入库：`tools/rl_framework_budget.py`。它是量级模型，用于给选型讨论定标，不是任何框架的实测吞吐。

---

## 3. 四个正交选择轴（全文骨架）

看懂任何一个 RL 训练框架，只要回答四个问题。这四个选择彼此独立，可以任意叉乘——所谓「框架之争」大多是不同叉乘点之间的取舍，而非谁对谁错。

### 轴一：控制器范式

| 范式 | 机制 | 谁在用 | 代价 |
|---|---|---|---|
| **Single-controller** | 一个中心进程（通常 Ray）编排全部数据流，worker 只执行 | OpenRLHF、NeMo-RL、SkyRL、AReaL | 把算子派发到数千加速器有控制开销 |
| **Multi-controller (SPMD)** | 每卡一个控制器，靠 P2P/collective 自协调，无中心 | DeepSpeed-Chat、Megatron 原生 RL | 改一个 RL 数据流要重写跨模型的通信-计算交织代码 |
| **Hybrid** | 节点间单控编排，节点内多控 SPMD 跑重计算 | **veRL（HybridFlow）** | 两层心智模型，调试复杂 |

HybridFlow 论文（[arXiv 2409.19256](https://arxiv.org/abs/2409.19256)，EuroSys 2025）的论证很干脆：RLHF 本质上是**多个 LLM 之间的数据流**——四模型、三阶段、负载异构、边是多对多 reshard 广播。纯单控不可扩展，纯多控不可编程，所以分层。它的实测是相对 DeepSpeed-Chat / OpenRLHF / NeMo-Aligner 的 **1.53×–20.57×**，配合 3D-HybridEngine 在训练/生成之间做零冗余 reshard。

### 轴二：资源布局

- **Colocate（同卡时分复用）**：KV cache 与训练激活不共存，显存峰值得以错峰。代价是生成与训练互相抢占、长尾拖尾被放大。
- **Disaggregated（分离池）**：推理池与训练池各自按最优并行度部署，权重跨网络同步。代价就是下面轴三的那笔账。
- **自动映射**：veRL 的 device mapping 按集群规模自动选 placement。Crossover 点随互联带宽与模型尺寸移动，**没有固定的最优解**。

同一套脚本给出的显存账解释了另一件事：**GRPO（无 critic）全参训 70B、外加一个 7B 奖励模型、ZeRO-3 全分片，全角色 colocate 的最小可行规模在 32 卡量级**——低于这个数，优化器状态（AdamW 的 fp32 m+v 单卡就要占掉大头）就把显存吃穿了。这就是社区里「70B 的 RL 起步就是两机四机」的真正原因，也是下面这四条优化各有市场的理由：

| 省显存的四条路 | 省掉什么 |
|---|---|
| 去掉奖励模型（RLVR 的规则奖励） | 一整个模型的常驻副本 |
| ref 只算 logprob，可 offload 或复用推理引擎 | reference 的常驻副本 |
| LoRA | optimizer 状态从 $$8P$$ 降到近似 0 |
| colocate 时分复用 | KV cache 与训练激活不共存，错峰掉一整块 KV cache |

### 轴三：权重同步（refit）

这是「训练侧布局 → 推理侧布局」的搬运，70B bf16 = 140 GB，**每步都要搬一次**。差异在三轴：分片布局（FSDP flat-shard vs Megatron TP-PP）、dtype、算子实现。

不同同步路径差的是**数量级**，不是百分比（70B bf16 权重约 140 GB）：

| 同步路径 | 量级 | 能否做热同步 |
|---|---|---|
| 进程内 reshard（colocate / 3D-HybridEngine） | 亚秒级，且优化器状态完全不搬 | 最优 |
| 机内直连（CUDA IPC / 共享内存） | 亚秒级 | 可用 |
| 跨机直连（NCCL / RDMA） | 秒级 | 可用 |
| 经 CPU / object store 中转 | 十秒到百秒级 | 勉强 |
| checkpoint reload（存储往返） | 百秒级 | 只能做容错，不能做热同步 |

LlamaRL 论文给出的实测很刺眼：**OpenRLHF 70B 权重同步 111.65s，LlamaRL 1.15s**，差两个数量级，主因是前者走 Ray object store / CPU 往返、后者直连 NCCL。**选框架时先问它的 refit 走哪条路径**，这一项能决定整体吞吐是几倍还是零点几倍。

优化路线有两条，各框架站队不同：

1. **减少搬运量**：veRL v0.9.0 的 `delta_sharded` 只搬变化分片，官方口径 72B 权重更新提速 **3.1×**。
2. **根本不搬**：Megatron 原生 RL 让训练进程直接当推理进程用，3D-HybridEngine 逐层 gather、优化器状态**从不搬移**——70B 的 AdamW fp32 m+v 是 560 GB，误搬就是 4× 慢。

### 轴四：时间模型

同步 → 一步一屏障；异步 → rollout 与 training 常驻并发；流式 → 生成完一条就进训练队列。用 Amdahl 算收益上界（$$f$$ 是生成占比，$$\text{ov}$$ 是可重叠比例）：

```
    生成占比 f    ov=50%    ov=70%    ov=90%   ov=100%
----------------------------------------------------
       60%      1.43x      1.72x      2.17x      2.50x
       70%      1.54x      1.96x      2.70x      3.33x
       80%      1.67x      2.27x      3.57x      5.00x
       90%      1.82x      2.70x      5.26x     10.00x
```

对照公开实测：AReaL **2.77×**、veRL fully async **2.35–2.67×**（128 GPU）、LlamaRL **10.7×**（405B）。全部落在 $$f\approx 0.8$$、$$\text{ov}\approx 0.7$$–$$0.95$$ 的区间内，量级自洽。

异步不是免费的：数据变 stale，rollout 用的是几步之前的策略，on-policy 假设破了。这就是第 7 章要讲的 TIS / IcePop / CISPO。

---

## 4. 主力框架逐个解剖

### 4.1 veRL（HybridFlow）——当前事实标准

*DeepWiki：[verl-project/verl](https://deepwiki.com/verl-project/verl)*



- **出身与许可**：字节 Seed 团队发起，现归 `verl-project` 组织（2026-01 迁移），Apache-2.0，约 22.9k stars。论文 [HybridFlow, EuroSys 2025](https://arxiv.org/abs/2409.19256)。
- **四轴定位**：混合控制 / 自动映射 / 进程内 reshard + CUDA IPC + NCCL 三种后端 / v0.9 起同步异步同一套控制流。
- **v0.9.0（2026-08-14）是一次大改**：统一 V1 PPO 训练器（同步与异步共用控制流）、**Megatron-Bridge 成为默认 Megatron 路径**、vLLM 要求 ≥0.18、新增 `delta_sharded` 异步权重同步（Qwen2.5-72B 提速 3.1×）、引入 `verl.plugin.platform` 硬件抽象层（AMD ROCm、昇腾 NPU、国产芯片）、支持 vLLM prefill-decode 分离 rollout（NIXL / Mooncake 传输层）。
- **算法覆盖**：PPO(GAE)、GRPO、DAPO、VAPO、RLOO、ReMax、PRIME、SPPO、GSPO、PF-PPO。
- **训练后端**：FSDP / FSDP2 / Megatron-LM（含 EP，官方口径支持到 671B）。**rollout 后端**：vLLM、SGLang、HF generate。

**优势**：placement 灵活性是全场最强的——同一套代码可以表达 colocate、分离、混合三种布局；算法覆盖最广；Megatron + FSDP 双后端意味着既有 5D/6D 并行的上限，又有 FSDP 的易用；硬件抽象层让它成为国产芯片上少有的可行选择。

**不足**（这些是真正会咬人的）：
- **调试对象变了**。有开发者的评价很准确：TRL 的脚本可以自上而下读，veRL 的运行是「一张分布在 Ray worker group 上的图」，挂起时你调试的是 **placement 与 resharding，不是 Python 控制流**。
- **版本锁死**。v0.9.0 要求 vLLM ≥0.18，lockfile 假设 transformers <5.11 与 torch 2.11.0；升级前必须确认自定义脚本是否依赖旧训练器路径（`verl/trainer/ppo` 非 v1 子目录），**这是破坏性变更**。
- **experimental 区仍在漂移**：`transfer_queue`、`fully_async_policy`、`one_step_off_policy`、`vla` 都还在 `verl.experimental`，README 明示「计划合入主库」。任何从这条路径 import 的代码都面临搬迁。
- **recipe 目录的迁移成本**：2026-01 目录被移到独立仓库并作为 git submodule。在那之前 fork 过 recipe 的人，副本已经分叉了。
- **一项依赖要单独审许可**：`TransferQueue` 来自 `github.com/Ascend/TransferQueue`，是独立项目、独立条款，且**总是被安装**。

**选它的时机**：已有 FSDP/Megatron 训练栈与 vLLM/SGLang 部署，想表达 GRPO/PPO 数据流而不想自己写控制器。**不选它的时机**：单卡、单进程、或者你没法锁住 transformers/torch 版本。

### 4.2 OpenRLHF —— 中小集群的性价比之选

*DeepWiki：[OpenRLHF/OpenRLHF](https://deepwiki.com/OpenRLHF/OpenRLHF)*



- **四轴定位**：Ray 单控 / 默认分离、`--colocate_all_models` 可切 colocate / vLLM NCCL 权重同步 / v0.8.0（2025-05）起支持异步。
- **现状**：v0.10.0（2026-04-12）、v0.11.0（2026-08-13）；vLLM 已升到 0.29.0（2026-09-10）。算法 PPO、REINFORCE++(-baseline)、GRPO、RLOO、DAPO、GSPO、DPO/IPO/cDPO。**2026-09 新增 FlashREINFORCE**：critic-free、single-rollout 异步 RL，用 vLLM logprobs 上的 binary-KL 信任域 + sample-mean 聚合。
- **新后端 Molt**：官方称其为 Automodel 驱动、比 DeepSpeed 更强、可扩到数百 B 参数，同时保持原有 OpenRLHF 工作流。
- **Sleep/wake 机制**：`--vllm_enable_sleep` / `--deepspeed_enable_sleep` 让 vLLM 与 ZeRO 在阶段切换时休眠唤醒以避免 OOM——这是中小集群上很实用的一招。

**优势**：Ray + vLLM + DeepSpeed 三件套组合最直白，70B 全参训练路径成熟；统一 agent 范式（AgentExecutorBase / MultiTurnAgentExecutor）让单轮与多轮写法一致；off-policy 修正内置（TIS / IcePop / seq-mask-TIS）；社区自报比 veRL/TRL 快约 3×（**这是版本与硬件相关的自测值，别当结论用**）。

**不足**：强绑定 vLLM 版本（v0.9.3 要求 vLLM ≥0.15.1）；DeepSpeed ZeRO 路线在超大规模（>70B 多节点）吞吐弱于 Megatron 系；**LlamaRL 论文实测其 70B 权重同步 111.65s**，是分离部署下的明显短板。

### 4.3 TRL —— 易用性的极端，以及「敢于删除」

*DeepWiki：[huggingface/trl](https://deepwiki.com/huggingface/trl)*



- **四轴定位**：单进程 + Accelerate，无 Ray / colocate 原生 / 进程内无同步开销 / 同步为主（异步 GRPO 在路线图上）。
- **v1.0（2026-03-31）** 确立双轨契约：稳定层（SFTTrainer、DPOTrainer、GRPOTrainer、RLOOTrainer、RewardTrainer）遵循语义化版本；实验层 `trl.experimental` 不承诺稳定。
- **v1.13.0（2026-09-10）把 PPOTrainer、PPOConfig 和 value-head 模型直接删掉了**。这个动作值得单独说：TRL 的设计哲学是「不为今天看起来稳定的形态建抽象，因为在这个领域强假设的半衰期极短」。PPO 在 v1.0 进实验层，五个月后被移除。
- **其他**：DistillationTrainer 转正（chunked JSD loss + vLLM 生成）；vLLM 两种模式（colocate 提速 1.3–1.7× / server）；通过 `environment_factory` 接 [Harbor](https://huggingface.co/blog/openenv) 做沙箱 agent 训练；19.3k stars，月下载 3M。v1.13.0 还记录了 **Qwen3-8B 单步 1,048,576 tokens**（380 s/step，单个 8 卡节点即可承载）。

**优势**：上手成本全场最低；Unsloth、Axolotl 直接构建在它的 Trainer API 上，生态位稳固；GRPO/RLOO 是稳定 API，可以放心进生产；colocate vLLM 免去了跨进程同步这一整类问题。

**不足**：分布式扩展不是它的战场——70B+ 多节点远不如 veRL / NeMo-RL；**没有 PPO 了**，需要 PPO 的团队得去别处；DAPO 的动态采样官方未支持。

### 4.4 NVIDIA NeMo-RL —— 全栈派的主力

*DeepWiki：[NVIDIA-NeMo/RL](https://deepwiki.com/NVIDIA-NeMo/RL)*



- **四轴定位**：Ray 单控 / 支持 colocate / CUDA IPC refit（MoE 优化 10×）/ 异步 RL + replay buffer。
- **两套训练后端**：DTensor（PyTorch 原生 FSDP2 + TP/SP/PP/CP）与 **Megatron Core（6D 并行，面向 100B+ 与 MoE）**；三套生成后端：vLLM、SGLang、**Megatron 原生生成**（免权重转换，对新模型 day-0 友好）。
- **算法**：GRPO、DAPO、CISPO、PPO、GSPO、GDPO、SFT（含 LoRA）、DPO、奖励建模、MOPD（on-policy 蒸馏）、跨 tokenizer 蒸馏。v0.6.0（2026-04-30）加入 SGLang 后端、Muon 优化器、投机解码；**Nemotron-3-Ultra 用它训练**，25+ 篇任务指南（SWE RL、音频后训练、量化感知 RL、YaRN 长上下文）。
- **前代**：[NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) 已于 2025-05-15 停止维护，新项目一律迁移 NeMo-RL。NeMo-Skills 是流水线编排层而非 RL 框架，训练后端可挂 NeMo-RL 或 veRL。

**优势**：NVIDIA 全栈（Megatron-Core 6D + TensorRT-LLM + NCCL refit），100B+/MoE 场景最扎实；原生 Megatron 生成免权重转换是独一份；算法清单里 GSPO / GDPO / CISPO 这些 2025–2026 的新目标跟得最快；文档与 recipe 质量高。

**不足**：生态绑定 NVIDIA 硬件；v0.7.0 roadmap（ETA 2026-06-30）里多项仍标记 WIP，包括 Router Replay Rollouts（R3）、PPO with MCore / dTensor、RDMA refit 与 delta refit——说明 **MoE RL 的数值稳定性在 NVIDIA 自己也还没完全收口**；对非 NVIDIA 平台基本不可选。

### 4.5 Megatron-LM 原生 RL 栈 —— 免 refit 的极端

*DeepWiki：[NVIDIA/Megatron-LM](https://deepwiki.com/NVIDIA/Megatron-LM)*



- **四轴定位**：无独立 Ray 控制器，RL 内嵌于 `train_rl.py` 主循环 / 天然 colocate / **根本不做权重同步** / RolloutPipeline 异步编排。
- **核心差异**：`MegatronLocal` 原地推理——**训练进程自身充当推理服务**。训练与推理同进程、同格式、同布局，因此 3D-HybridEngine 那笔搬运账直接归零。
- 2026 年持续迭代：2026-01 `train_rl` 启用 CUDA graphs for RL；2026-03 RL Hybrid MoE cudagraphs。`examples/rl` 提供 countdown / dapo / gsm8k 等 recipe。

**优势**：彻底消灭了主轴三；复用 Megatron Core 的全部并行能力（TP/PP/EP/DP/CP）；对已经把预训练栈建在 Megatron 上的团队，这是最省事的路径——也是本站 [Megatron 系列](/2026/09/28/megatron-00-foundation/)要接着往下走的一条线。

**不足**：HF 模型必须经过 **Megatron Bridge** 转换才能进；相比 veRL/NeMo-RL 生态封闭，算法与环境基本要自己写；只适合已经全量落地 Megatron 栈的团队。

### 4.6 DeepSpeed-Chat —— 已经退场，但值得一读

- Hybrid Engine 在**同一模型**上动态切换 ZeRO 训练与 TP 推理，天然 colocate、无需跨引擎同步。论文 [arXiv 2308.01320](https://arxiv.org/abs/2308.01320) 是 InstructGPT 三阶段的忠实复现。
- **现状**：2023 年后基本冻结，仅有零星 bugfix；只支持 PPO；生成端是自研推理而非 vLLM/SGLang，长 CoT 场景落后一个量级。
- **定位**：**教学参考实现**。DeepSpeed 本体已于 2025 年转入 Linux Foundation AI & Data。新项目请用 veRL / OpenRLHF / NeMo-RL。

---

## 5. 学术界与中国厂商：异步、agentic 与「后 Trainer」形态

这一族的共同特征是：**先把异步做成一等公民，再谈别的**。

### 5.1 RLinf（清华 + 无问芯穹 + 中关村学院）

*DeepWiki：[RLinf/RLinf](https://deepwiki.com/RLinf/RLinf)*



**这是本篇最值得补的一个**——它不在上面任何一个族里，而是横跨两族。

- **出身**：清华大学、北京中关村学院、**无问芯穹（Infinigence AI）**、北京大学、UC Berkeley、北航、上海交大联合发布。2025-09 开源，论文 [arXiv 2509.15965](https://arxiv.org/abs/2509.15965)。名字里的 `inf` 双关 **Infrastructure** 与 **Infinite**。2026 年已加入 PyTorch Ecosystem Landscape。
- **核心范式 M2Flow（Macro-to-Micro Flow）**：开发者用命令式接口粗粒度地描述「组件之间怎么通信、怎么同步」（宏逻辑流），系统再自动把它拆解、重组成时空两个维度上的优化执行流（微执行流）。**逻辑编程与执行规划彻底解耦**——这是它区别于 veRL 的「手动指定 placement」的根本思路。
- **三个支撑机制**：**Worker 抽象**（把每个 RL 组件封装成可灵活放置的单元，内置自适应通信，跨 GPU/NPU/CPU 异构）、**弹性流水线 + 自动上下文切换**（不改逻辑流就能调流水线粒度、做加速器的时分复用）、**profiling 引导的调度策略**（自动选执行模式）。
- **三种执行模式 + 两种调度**：colocated / disaggregated / **hybrid**（两者可定制组合）；dynamic scheduling（运行时动态调整资源）与 static scheduling（按负载自动选模式，免手工配 GPU 与并行度）。
- **具身智能是它的主场，也是它唯一性所在**：VLA 模型覆盖 OpenVLA、OpenVLA-OFT、**π₀ / π₀.₅**、GR00T-N1.5、LingBot-VLA、Dexbotic；仿真器覆盖 ManiSkill3、LIBERO、IsaacLab、RoboTwin、MetaWorld、CALVIN、RoboCasa、BEHAVIOR-1K；并且做到了**首个 π₀ 系列（flow-matching action expert）的 RL 微调**。
- **版本节奏很快**：v0.1（2025-12）打通仿真-训练-推理一体化；**v0.2（2026-03）扩展到真实世界**，支持全异构、全异步的真机 RL（口号是「像使用 GPU 一样使用机器人」）、跨域多机真实世界 RL、人在环；v0.3.0 已上 PyPI。2026 年上半年陆续加了世界模型（Wan、OpenSora）、Sim-Real 协同训练（RLinf-Co）、多智能体 RL（WideSeek-R1）、DSRL、SAC-Flow、FUSCO（MoE All-to-All 加速）等。
- **算法**：GRPO、PPO、DAPO、REINFORCE++、SAC、SFT（全参 / LoRA），以及 DAgger。后端 FSDP + HF（快速适配新模型）与 Megatron + SGLang/vLLM（大规模）二选一，rollout 支持 SGLang / vLLM / HF。

**优势**：
- **唯一同时吃下「仿真器 RL」与「大模型 RL」的框架**。别人的 rollout 是生成 token，它的 rollout 是「跑仿真器拿轨迹」，同时被训的又是十亿到数十亿参数的 VLA——这两件事此前分属两族。
- 公开性能数字很硬：论文口径相对 veRL / slime **1.07×–1.70×**（推理 RL），具身 RL 最高 **2.13×**（v0.1 release note 口径 2.434×）；**2026-05 被 StanfordVL / BEHAVIOR-1K 官方上游仓库集成**，端到端延迟从 1028 ms/step 压到 **41 ms/step（25×）**；RLinf-VLA 拿 CVPR 2026 ScaleBot Workshop 最佳论文（1.61×–1.88× 训练加速）；2026-03 入选 EAI-100 具身智能年度十大突破项目。
- Sim-Real 协同训练的效果很实在：OpenVLA 平均成功率 **16.5% → 64.0%**，π₀.₅ **26.7% → 66.2%**；真实轨迹只有 10–20 条时优势最大，50 条真机轨迹就能追平 SFT 用 200+ 条的效果。
- 国产硬件适配认真：华为云与昇腾完成了适配、精度对齐与性能优化并合入社区，支持「昇腾卡训推 + 渲染卡仿真」的跨节点异构训练。

**不足**：
- **具身场景的门槛在框架之外**：仿真器资产、真机硬件、相机标定这些才是真正的工作量，框架只解决调度与吞吐。
- 生态与社区规模仍小于 veRL；LLM 推理/agentic 那条线上，它的 SOTA 是拿 DeepSeek-R1-Distill-Qwen 做的（RLinf-math-1.5B：AIME24 48.44 / AIME25 35.63 / GPQA-d 38.46；7B：68.33 / 52.19 / 48.18），强但不是唯一卖点。
- 抽象层（Worker / Scheduler / Channel）比 veRL 更厚，读懂成本更高；安装推荐走 Docker，依赖（Maniskill3、LeRobot、openpi）较重。

### 5.2 slime（THUDM / 智谱）

*DeepWiki：[THUDM/slime](https://deepwiki.com/THUDM/slime)*



- **架构**：Megatron-LM（训练）+ SGLang（rollout）经 Data Buffer 桥接，**SGLang-native 单一推理后端、引擎透传**。设计上刻意保持低抽象。
- **身份**：GLM-4.5 / 4.6 / 4.7 的训练底座。异步有 `examples/fully_async`；MoE 有 GLM-4.5、Qwen3-30B-A3B、DeepSeek-R1 的原生示例，CI 覆盖 dense + MoE。
- **生态**：阿里的 Dressage（Accio）与 RadixArk 的 Miles 都基于它；Miles 补了 FP8、R3、speculative RL、VLM 多轮；vime 则保留 slime 训练栈、把 rollout 换成 vLLM + vllm-router。

**优势**：Megatron + SGLang 这条组合调得最深，大规模 agentic 与长视野 rollout 是主场；低抽象意味着看得懂、改得动。**不足**：单一推理后端（想用 vLLM 得走 vime 这类 fork）；文档与社区规模不如 veRL。

### 5.3 AReaL（清华交叉信息院 + 蚂蚁）

*DeepWiki：[inclusionAI/AReaL](https://deepwiki.com/inclusionAI/AReaL)*



- **全异步是它的定义**：rollout 与 training 彻底解耦，rollout worker 持续生产、learner 独立消费，靠 staleness 上限 $$\eta$$ 控制新鲜度。论文 [arXiv 2505.24298](https://arxiv.org/abs/2505.24298)，实测 **2.77×**。
- **三个独门机制**：**可中断 rollout**（生成中途换权重、丢弃旧 KV 重算，维持 on-policy 正确性）、**GPU-Direct RDMA 权重同步**（1000 GPU 集群 <3s）、**radix cache 刷新**（SGLang 后端在权重更新后自动刷，保证 on-policy 正确）。
- **版本线**：v0.3 = AReaL-boba²，v1.0 全异步，**v2.0（2026-07）转向微服务运行时**（training / inference / agent / weight-update 各自独立服务，[技术报告](https://github.com/inclusionAI/AReaL)）。
- **agentic 接入方式极聪明**：`ArealOpenAI` 提供 OpenAI 兼容 SDK，**任何能通过 API 访问的 agent 都能被训练**——不需要重写 agent 逻辑。

**优势**：异步做得最彻底，公开实测数字最完整；agentic 的接入门槛最低；自带 Archon 引擎（PyTorch 原生 5D 并行）与 FSDP/Megatron 后端可换。**不足**：异步引入的 off-policy 修正（IcePop / KPop）需要调参才稳；v2.0 微服务化后运维复杂度上升。

### 5.4 ROLL（阿里巴巴）

*DeepWiki：[alibaba/ROLL](https://deepwiki.com/alibaba/ROLL)*



- **架构**：Ray 多角色 + Megatron-Core（5D：DP/TP/PP/CP/EP）+ SGLang/vLLM 训推分离；Rollout Scheduler + AutoDeviceMapping 做异构调度。
- **异步**：ROLL Flash（[arXiv 2510.11345](https://arxiv.org/abs/2510.11345)）生产者-消费者解耦，用 **Asynchronous Ratio** 控制 stale，集成 Decoupled PPO / TOPR / TIS / CISPO。
- **算法**：PPO / GRPO / GSPO / REINFORCE++ / TOPR / GiGPO / StarPO / RAFT++。2026-06 有 OSDI'26 的 RollArt；支持 Ascend NPU 与 AMD GPU。

**优势**：国产芯片支持最认真；算法清单里 GiGPO / StarPO 这类 agentic 专用目标是一手的；异构调度对企业集群友好。**不足**：文档与社区规模弱于 veRL；抽象层较厚，二次开发成本高。

> 顺带纠两个常见误传：**ROLL 没有官方中文名「如流」**；**ROME 不是 Prime Intellect 的项目**，它是 ROLL 团队的 agentic 生态（ALE + ROME，2026-01）。

### 5.5 ChatLearn（阿里云 PAI）

*DeepWiki：[alibaba/ChatLearn](https://deepwiki.com/alibaba/ChatLearn)*



- **定位**：计算图式编程，封装几个函数即定义算法。**训练** Megatron / FSDP2，**推理** vLLM / SGLang；Sequence Packing、Ulysses SP、Group GEMM 加速；资源独占/共享调度可选。
- **性能口径**：官方称 70B+70B 相对 DeepSpeed-Chat / OpenRLHF 提速 **137%–208%**（厂商自测值）。GRPO 与 GSPO（Qwen GSPO 的第一时间复现）都支持；配合 SGLang + LangGraph 做多轮工具调用。

**优势**：PAI 平台上开箱即用，与企业调度集成好；GSPO 跟进快。**不足**：开源社区活跃度明显低于 veRL/AReaL；MoE RL 教程仍在 roadmap。

### 5.6 prime-rl（Prime Intellect）

*DeepWiki：[PrimeIntellect-ai/prime-rl](https://deepwiki.com/PrimeIntellect-ai/prime-rl)*



- **全异步 + FSDP2 + vLLM**，FP8 推理、PD 分离、EP/CP，目标 1T+ MoE / 1000+ GPU。
- **两个去中心化组件很有意思**：**SHARDCAST**（权重树状广播）与 **TOPLOC**（可验证推理哈希）——后者用来在不可信的推理 worker 上验证 rollout 真实性，服务于 INTELLECT-2 那种全球分布式异步训练（32B）。
- **环境协议**：`verifiers` 仓库 4.6k stars，已成事实上的「RL 环境 + 评测协议」。

**优势**：为大规模分布式/去中心化场景设计，是少数认真处理「不可信推理 worker」的框架；verifiers 生态好。**不足**：面向特定基础设施形态，常规企业集群用不上那些去中心化机制。

### 5.7 三个不该漏的：EasyR1、SimpleVLA-RL、MARTI

**EasyR1（hiyouga，LLaMA-Factory 作者）**——多模态 RL 训练的默认答案，5.1k★，Apache-2.0。它是 veRL 的一个干净分叉，但解决了 veRL 没解决的事：**VLM 原生支持**（Qwen2-VL / 2.5-VL / 3-VL、视频输入）、**内置 LoRA**、**padding-free 训练**、**YAML + CLI 点号覆盖**（改配置不用动代码）、**奖励函数就是一个普通 Python 文件里的 `compute_score()`**、预置 Docker 镜像。算法 7 种（GRPO、DAPO、REINFORCE++、ReMax、RLOO、GSPO、CISPO）。MMR1、Vision-R1、Seg-Zero、GUI-R1、Long-RL、MetaSpatial 等 20+ 研究项目用它训练。DeepWiki：[hiyouga/EasyR1](https://deepwiki.com/hiyouga/EasyR1)。

> 定位对照：直接用 veRL 做 VLM 要自己适配、自己写 reward 接口、没有 LoRA；EasyR1 把这些补齐了。**做多模态 RL，先看 EasyR1 再看 veRL。**

**SimpleVLA-RL（上海 AI Lab，PRIME-RL 组织，ICLR 2026）**——具身 RL 的另一条路，1.9k★。同样基于 veRL，但扩展方向是 VLA：**并行多环境渲染**加速轨迹采样、**二元 0/1 奖励**（不需要任何 reward shaping）、探索增强的 GRPO。结果很硬：单条轨迹 SFT + RL 就能到 96.9% 成功率、**反而超过全轨迹 SFT**；机器人自主涌现出演示数据里没有的动作（著名的 “pushcut” 现象）；Sim-to-Real 迁移 +21%，长时程灵巧操作相对提升 300%。DeepWiki：[PRIME-RL/SimpleVLA-RL](https://deepwiki.com/PRIME-RL/SimpleVLA-RL)。

> 它与 5.1 的 RLinf 是具身 RL 的两个直接竞品，路线差别很清楚：**RLinf 是自建底座（M2Flow + Worker/Scheduler/Channel），SimpleVLA-RL 是站在 veRL 肩上做垂直扩展**。要通用基础设施选前者，要快速在 LIBERO / RoboTwin 上跑出结果选后者。

**MARTI（清华 C3I，ICLR 2026）**——LLM **多智能体** RL 训练框架，基于 OpenRLHF。支持图式工作流与第三方多智能体框架；**MARTI-v2（2026-02）** 加入多智能体树搜索（MARS²，面向代码生成），并集成 GSPO 序列级损失、**TIS 修正**（解决 vLLM 采样不匹配）、动态数据过滤、overlong buffer（超长 token 惩罚），支持到 **32K token** 超长序列与异构多智能体训练。ReviewRL、CoMAS 构建在它之上。DeepWiki：[TsinghuaC3I/MARTI](https://deepwiki.com/TsinghuaC3I/MARTI)。

### 5.8 其他值得记的

| 框架 | 一句话 | 适用 |
|---|---|---|
| **SkyRL**（UC Berkeley / NovaSky） | 源自 veRL fork，拆成 skyrl-train / gym / agent / tx 四模块；2026-02 官方集成 Harbor | 长程多轮工具、学术研究 |
| **rLLM**（agentica，Berkeley Sky + Together） | veRL 深度 fork + AgentExecutionEngine 全异步；产出 DeepSWE-32B（SWE-Bench-Verified 59%） | SWE / coding agent |
| **verl-agent** | veRL 扩展，**逐步独立交互**（不拼接历史）故可扩到 50+ 步；提出 GiGPO | 长程 LLM/VLM agent |
| **RL-Factory**（Simple-Efficient） | 基于 veRL，专攻多轮工具调用（asyncio 异步工具、MCP 工具、loss mask）；Search-R1 + Qwen3-4B 吞吐 6.8× | 搜索/工具类 agent |
| **MARTI**（清华 C3I） | 见 5.7；LLM 多智能体 RL，MARTI-v2 加树搜索 | 多智能体系统 |
| **ms-swift**（ModelScope） | GRPO/DAPO/PPO/DPO/KTO 全，单卡 LoRA GRPO，4×80G 可训 72B | 国内单机/小规模 |
| **LLaMA-Factory** | 模型与方法覆盖最广（100+ LLM/VLM），YAML 驱动，WebUI | 快速试验、教学 |
| **Unsloth** | 构建在 TRL 上的单卡 GRPO，自研 Triton kernel；官方口径 8B GRPO 显存降到标准实现的约十分之一；支持 GSPO / Dr.GRPO | 消费级显卡 |
| **Tunix**（Google） | JAX 原生，MaxText 训练 + vLLM/SGLang-JAX rollout；PPO/GRPO/GSPO-Token/DAPO/Dr.GRPO；2025-12 起 agentic RL | TPU / JAX 栈 |

**一个值得注意的现象：veRL 已经变成了上游底座。** 上面这个表里 SkyRL、rLLM、verl-agent、RL-Factory 是 veRL 的 fork 或扩展，5.7 的 EasyR1 是它的干净分叉、SimpleVLA-RL 是它的 VLA 扩展，slime 生态之外的 Miles / vime / Dressage 也在这个谱系附近。**2026 年做 RL 框架，很多团队做的事不再是「从零写一个」，而是「fork veRL 然后垂直做深」**——这跟 2024 年人人自己写 PPO 循环的局面完全不同。

> Unsloth 有一条许可条款要注意：核心 Apache-2.0，但 **SaaS 暴露 API 时触发 AGPL-3.0**。

---

## 6. 经典 RL 框架：还活着吗

这一族对 LLM 后训练基本无用（**例外是 5.1 的 RLinf 与 5.7 的 SimpleVLA-RL**——它们把仿真器和 VLA 大模型接在了一起，是两族之间的桥），但对机器人/控制/游戏/MARL 仍是主力，而且它们的抽象影响了后来所有人。这里给一张「2026 年活跃度真相」表——很多选型文档还在推荐已经停更的项目。

| 框架 | 设计特点 | 现状（2026-09） | 判定 |
|---|---|---|---|
| **Stable-Baselines3** | 统一 `model.learn()`，monolithic policy；SBX 是 JAX 版（约 20×） | 13.8k★，v2.9.0（2026-06），周级提交 | **活跃**，科研/教学首选 |
| **Ray RLlib** | new API stack：EnvRunner / Learner / RLModule，Actor 图 | Ray 2.58.0（2026-08），new stack 已成默认 | **活跃**，大规模生产 RL |
| **TorchRL / tensordict** | TensorDict 贯穿 env/collector/buffer/loss | 3.5k★，0.13–0.14（2026），日级提交 | **活跃**，PyTorch 官方 |
| **CleanRL** | 单文件实现（ppo_atari.py 约 340 行），可读可改 | 8.6k★，2025 下半年仍有提交 | **活跃**，学算法首选 |
| **Tianshou**（清华） | 高度模块化 collector/buffer/trainer，2.0 重构 | 2.0.1（2026-04）后约 5 个月无提交 | **半静默** |
| **Acme**（DeepMind） | 可复用 actor/learner/replay 组件，JAX + TF 双后端 | 4.1k★，最近提交 2026-02 | **低活跃但存活** |
| **Sample Factory 2.0** | RolloutWorker/InferenceWorker/Batcher/Learner + 共享内存 | 最近提交 2025-05 | **基本停滞** |
| **Dopamine**（Google） | 极简库 + gin 配置，JAX agents | 10.9k★，2026-03，无新 release | **稀疏存活** |
| **Stoix / JaxMARL** | 端到端 JAX，vmap/pmap 向量化 | Stoix 活跃（2026-03） | **活跃**，JAX 系研究 |
| **Brax / MJX** | 纯 JAX 物理仿真 + 训练 | Brax 0.14.2（仅 training 维护），envs 被 MuJoCo Playground 取代 | **部分退役** |
| **PufferLib** | CUDA/Cython 加速 env，1M+ steps/s | 最近提交 2025-11 | **中等** |
| **OpenAI Baselines / Spinning Up** | 历史地位 | 2021 后零提交 | **已停** |
| **Meta ReAgent** | — | README 明示归档，建议转 Pearl | **已归档** |

### 6.1 补一张国产 / 中文社区的表

上面那张表全是英文社区项目，而中文社区其实有自己完整的一族。这一族在国内的机器人、多智能体、决策智能教研里仍是默认选择，且有几个能力是英文社区没有的：

| 框架 | 设计特点 | 与英文社区的差异化 | 适用 |
|---|---|---|---|
| **DI-engine**（上海 AI Lab OpenDILab） | 算法覆盖最全：PPO/IMPALA/SAC 之外还有 QMIX/MAPPO 多智能体、GAIL 等模仿学习、CQL/DT 离线 RL | 把**在线 RL + 多智能体 + 模仿学习 + 离线 RL** 塞进一个库，SB3/RLlib 都只覆盖其中一部分 | 决策智能教研、算法覆盖优先 |
| **PARL**（百度 PaddlePaddle） | 基于 PaddlePaddle，含百度参赛方案，**原生支持分布式** | 中文文档与国产硬件链路完整；但绑定 Paddle 生态 | 已在飞桨栈上的团队 |
| **OpenRL**（OpenRL-Lab） | 三层架构（封装层 / 组件层 / 工具层），封装 Gymnasium、PettingZoo、Isaac Gym；内置 Gallery（可复现）+ Arena（对抗评测） | **唯一同时支持 NLP/RLHF、多智能体、自博弈、离线 RL 与 DeepSpeed** 的通用框架；在对话任务上 FPS 比 RL4LMs 高 17.2%，GPT-2 / OPT-1.3B 上 DeepSpeed 集成带来 30–35% FPS 提升 | 想用一套 API 横跨 NLP 与控制的研究 |
| **XuanCe 玄策**（东南大学 agi-brain） | **40+ 算法**（含大量 MARL：VDN/QMIX/QTRAN/MAPPO/COMA 等），[arXiv 2312.16248](https://arxiv.org/abs/2312.16248) | **唯一同时支持 PyTorch / TensorFlow / MindSpore 三后端**，且支持 CPU / GPU / **昇腾** 与 Ubuntu / Windows / macOS / EulerOS | 多后端对比研究、昇腾环境 |

同生态里还有一批更聚焦的：MARLlib（多智能体）、RL4LMs 与 trlx（语言模型 RL 的早期形态，后者是 TRL 的前身脉络之一）、d3rlpy（离线 RL）、MushroomRL、skrl（机器人）、FinRL（金融）。**它们共同的特点是：都不做大模型后训练**——这也是整族与第 4、5 章的分界线。

### 为什么 RLlib 的抽象套不到 LLM 上

这是本章唯一真正重要的结论，值得写开：

1. **采样成本的性质不同**。经典 RL 的采样是跑环境（物理仿真/游戏帧），RLlib 用 `num_env_runners` 表达它；LLM 的采样是一次**带 continuous batching、PagedAttention、prefill/decode 分离的自回归生成**，占单步墙钟 60–90%。
2. **`RLModule` 只是 NN wrapper**。它没有「推理引擎」这个概念，而 LLM RL 必须是**训练引擎 + 推理引擎双引擎循环**，两者对硬件的诉求还相反（训练要显存放激活与优化器，推理要显存放 KV cache，且要更高的 batch 并发）。
3. **权重每步都变**。经典 RL 里 actor 拉一次权重可以跑几千步；LLM RL 每一步都要把数十到数百 GB 在两种并行布局之间搬一次（第 3 章轴三）。RLlib 没有这个抽象的位置。
4. **状态是 token 序列**。action 是「从数万词表里选下一个 token」，reward 是 RM/verifier 对整段 completion 的打分，还涉及 reference logprob、token 级 mask、多轮轨迹的 token-id 持久化。

一句话：**经典 RL 优化的是「采样 = 环境步进」的循环；LLM RL 优化的是「生成吃掉预算」的双引擎循环**。这就是 OpenRLHF / veRL / slime 全部另起炉灶的原因。

---

## 7. 三个硬问题的工程解法

框架之间的差异，最后都落在三个具体问题上。选型时直接问：你打算怎么解这三个？

### 7.1 训推数值不一致（on-policy 假设的破洞）

vLLM/SGLang 推理出的 logprob 与训练引擎前向的 logprob 对同权重给出不同值——不同的 kernel、不同的 dtype、FP8/INT4 量化会放大这个 gap。后果是重要性比失真，off-policy 程度被低估。**有一条经验规律值得记住：rollout 的精度越低，重要性比的失真越大**——有报告称 FP8 rollout 会把重要性比放大 1.7×–2.9×（**该数字来自二手来源，待验证**，但方向是确定的）。

| 修法 | 机制 | 谁在用 |
|---|---|---|
| **TIS**（Truncated IS） | 截断 token 级重要性比上界，软修正 | veRL（需显式开启）、OpenRLHF |
| **IcePop** | 双边 mask，越界 token 梯度直接置零 | AReaL / Ring-1T，MoE 上优于 TIS（AIME25 +14%） |
| **CISPO** | 裁剪 **IS 权重**并 stop-gradient，保留所有 token 梯度 | MiniMax-M1（[arXiv 2506.13585](https://arxiv.org/abs/2506.13585)）、NeMo-RL、ScaleRL |
| **FP32 LM Head** | 只把输出头提精度，成本极低 | ScaleRL 报告渐近线 0.52→0.61（**二手来源，待验证**） |

**注意**：veRL 里 TIS/IcePop **不是默认开启的**，要显式设 `--algo.advantage.is_correction_enable`。如果你用了 FP8 rollout 却没开修正，训练是在一个你不知道的 off-policy 状态下跑的。

### 7.2 MoE 的路由抖动

GRPO 用在 MoE 上会崩，原因是**专家激活波动**：一次更新后约 10% 的 token 路由会变，导致 token 级重要性比剧烈抖动。

- **Routing Replay**（Qwen 原方案）：缓存旧策略激活的专家并在训练时重放。有效，但吃显存与通信。
- **GSPO**（Qwen3，[arXiv 2507.18071](https://arxiv.org/abs/2507.18071)）：把重要性比改成**序列级**，对 token 概率波动天然不敏感，**彻底免掉 Routing Replay**，还顺带允许直接用推理引擎的似然、容忍精度差异（利好 partial rollout / 多轮 / 训推分离）。
- **CISPO / TIS** 亦可缓解。NeMo-RL 至今仍在 roadmap 里做 R3（Router Replay Rollouts），说明这条路还没完全收敛。

### 7.3 长尾 rollout

前面第 2 章已经量化了问题。工程解法四类：

1. **Partial rollout**：veRL 的 `sleep()/resume()` 在同步点暂停在途生成、下一轮续用；slime 的 active partial rollout（超额 64 请求、最快 32 完成即终止并保留 KV）；AReaL 的可中断 rollout（换权重后重算）。
2. **动态采样**：[DAPO](https://arxiv.org/abs/2503.14476) 过采样后过滤组内全对/全错的 prompt；DUPO（[arXiv 2507.02592](https://arxiv.org/abs/2507.02592)）复制非零标准差样本填充。
3. **序列级均衡**：chunked prefill、continuous batching、sequence packing（NeMo-RL 实测减少 padding、提吞吐）。
4. **PD 分离**：veRL v0.9 支持把 prefill 与 decode 拆到不同节点（NIXL / Mooncake 传输层）——prefill 是算力瓶颈、decode 是带宽瓶颈，拆开后各自可以按需扩。

### 7.4 数值实验：把上面这些算法差异算出来

算法那层的事情光看公式不容易建立直觉。`tools/rl_obj_numeric_lab.py`（纯 numpy，CPU 可跑）做了三组实验，把前面散落的几个论点落到数字上。

**实验一：GRPO 有多少组是白算的。** 用 Beta(1.5, 3.0) 建模题目难度（偏难、长尾，接近数学题 RL 中期），每组 $$G=8$$ 条 rollout，二值奖励：

```
题目正确率分布：P10=0.09  P50=0.31  P90=0.63

  全错组（0/8）占比        :  15.40%
  全对组（8/8）占比        :   1.23%
  **零梯度组合计**          :  16.63%

  分难度看：
       题目正确率区间      组数占比        零梯度组占比
    [0.0, 0.1)    12.14%        62.78%
    [0.1, 0.3)    36.41%        19.61%
    [0.3, 0.7)    46.01%         2.16%
    [0.7, 0.9)     5.22%        14.56%
    [0.9, 1.0)     0.22%        50.00%
```

**约 17% 的 rollout 算力完全不产生梯度**，而且浪费高度集中在难度两极——正确率低于 0.1 的题有 **63%** 的组是白算的。这正是 DAPO dynamic sampling 与 DUPO 的靶心。

同一组实验还量化了 Dr.GRPO 的论点：组内只答对 1 条（难组）时，含 std 归一化的 advantage 幅度是 **2.65**，答对 4 条时只有 **1.00**——**除以组内标准差等于给难题发了一张放大券**，这就是 Dr.GRPO 主张去掉 std 的原因。

**实验二：token 级 vs 序列级比率的方差。** 序列长度 4096、token 级 log-ratio 标准差 0.35 时：

```
      token 级 log-ratio  std = 0.3501
      序列级 log-ratio    std = 0.0053
      → 序列级比率的波动被压低 71.1×
```

这就是 **GSPO 在 MoE 上不需要 Routing Replay 的全部理由**：专家路由变化是 token 级扰动，取序列均值后自然被平滑掉了。同一个实验里，PPO/GRPO 的 token 级裁剪（$$\varepsilon=0.2$$）会让 **28% 的 token 完全没有梯度流过**，而 CISPO 保留全部 100%。

**实验三：训推不一致怎么假装重要性比变了。** 建模为推理引擎的 logprob 与训练引擎真值差一个噪声 $$\varepsilon$$，训练时若拿它当分母，重要性比变成 $$\text{ratio}_{\text{true}}\cdot e^{-\varepsilon}$$：

```
  噪声 sigma   ratio P95   ratio P99   超界(>1.2)占比
        0.00       1.086       1.123           0.01%
        0.05       1.123       1.178           0.48%
        0.15       1.297       1.445          12.40%
        0.30       1.648       2.028          27.37%
```

两引擎完全一致时几乎不超界；噪声到 0.30（激进量化 / FP8 rollout 的量级）时 **27% 的 token 越界**——**这些 token 的重要性比其实没变，是引擎差异假装它变了**。三种修法的差别也随之清楚：TIS 与 CISPO 保留全部 token 只压权重，IcePop 直接丢弃越界 token 的梯度（本例丢 2.3%），GSPO 则是治本（改在序列级算比率）。

### 7.5 环境接口：正在收敛，但还没统一

| 标准 | 形态 | 现状 |
|---|---|---|
| **OpenEnv**（HF） | Gymnasium 风格 `reset()/step()/state()`，HTTP/WebSocket + Docker，**MCP 一等公民**，奖励无关（只管接口与部署） | 2026-06 治理移交给多组织委员会（Meta-PyTorch、NVIDIA、HF、vLLM、SkyRL、Stanford 等） |
| **Harbor** | Dockerfile + 测试脚本定义任务，统一可训练轨迹格式 **ATIF** | SkyRL 官方集成（2026-02），TRL 通过 `environment_factory` 接入 |
| **verifiers** | 协议层（环境 + 评测） | Prime Intellect，4.6k★ |
| **skyrl-gym** | Gymnasium 环境集合 | SkyRL |

**MCP 是跨训练与生产一致性的实际公约数**——OpenEnv、Harbor、RL-Factory 都把它当一等公民。如果你的 agent 在生产环境里用 MCP，训练时也应该用同一套。

---

## 8. 算法 × 框架支持矩阵

选型时最常被问的一句是「某某算法这个框架支持吗」。下面这张表按算法横向扫（✓ = 官方支持或有一等公民 recipe，— = 未见官方支持或需自行实现）：

| 算法 | TRL | OpenRLHF | veRL | NeMo-RL | slime | AReaL | ROLL | RLinf | EasyR1 |
|---|---|---|---|---|---|---|---|---|---|
| **GRPO** | ✓ 稳定 API | ✓ | ✓ | ✓ | ✓ 主力 | ✓ | ✓ | ✓ | ✓ |
| **PPO** | ✗ v1.13 起移除 | ✓ | ✓ | ✓ | ✓ | ✓ Decoupled | ✓ | ✓ | — |
| **DAPO** | ✓ 默认 loss | ✓ | ✓ | ✓ | — | — | — | ✓ | ✓ |
| **GSPO** | — | ✓ | ✓ | ✓ | — | — | ✓ | — | ✓ |
| **CISPO** | — | ✓ | — | ✓ | — | — | ✓ TOPR | — | ✓ |
| **REINFORCE++** | — | ✓ 原创 | — | — | — | — | ✓ | ✓ | ✓ |
| **RLOO** | ✓ 稳定 API | ✓ | ✓ | — | — | — | — | — | ✓ |
| **ReMax / VAPO / PRIME** | — | — | ✓ | — | — | — | — | — | ✓ ReMax |
| **离线族（DPO / KTO / SFT）** | ✓ 稳定 | ✓ DPO/IPO/cDPO | 部分 | ✓ 含 RM、蒸馏 | — | — | — | ✓ SFT/LoRA/SAC | — |

三点读法：

- **GRPO 已经是无争议的默认项**，几乎所有框架都支持。真正的分化在它之上：GSPO / CISPO / DAPO 这些 2025–2026 的新目标，谁跟得快谁就是「算法前沿」那一档（NeMo-RL 与 EasyR1 跟得最全）。
- **PPO 在退潮**。TRL v1.13 直接删掉 PPOTrainer；但它在长视野 agent 场景又回来了（见 5.3 与第 11 章的未解问题），所以 OpenRLHF / veRL / NeMo-RL / RLinf 都还留着。
- **离线族（DPO/KTO/SFT）与在线 RL 是否同一个库**，是个实际的工程分界：TRL、OpenRLHF、NeMo-RL 是的（一个库跑完 SFT → RM → RL），veRL / slime / AReaL 则专注在线 RL，SFT 要回到别的栈。

## 9. 训练跑歪时看什么：指标与失败模式

这一节是横评文章里通常没有、但工程上最容易卡住的地方。**框架选对了不代表训练能收敛**，下面是必看指标和失败模式对照。

### 9.1 九个必看指标

| 指标 | 健康的样子 | 异常意味着什么 |
|---|---|---|
| **reward / pass@1** | 缓慢上升，最终平台 | 长期平坦 → 奖励饱和、探索不足或数据太难 |
| **response length** | 略增后趋稳 | 单调暴涨 → 长度爆炸，多半伴随奖励 hacking |
| **entropy** | 缓慢下降 | 骤降到接近 0 → 熵崩塌；反而上升 → 训练不稳 |
| **clip ratio**（被裁剪 token 占比） | 个位数到 20% | 超过一半 → 步长过大，或 off-policy 严重 |
| **KL 到 ref** | 小而稳 | 飙升 → 策略漂移；恒为 0 → KL 项根本没生效 |
| **组内全对/全错占比** | 越低越好 | 超过三成 → 该上 DAPO dynamic sampling（见 7.4 实验一） |
| **rollout 与训练引擎的 logprob 差** | 1e-3 以下 | 偏大 → 必须开 TIS / IcePop（见 7.4 实验三） |
| **importance ratio 的 max / P99** | 靠近 1 | 长尾重 → MoE 路由抖动或训推不一致 |
| **生成占墙钟比** | 60–90% | 过低 → 瓶颈其实在训练侧，框架选型的前提就错了 |

### 9.2 八种常见失败模式

| 现象 | 最可能的原因 | 怎么定位 | 怎么处理 |
|---|---|---|---|
| reward 涨但人工评测不涨 | **reward hacking** | 抽样本人工看；看 length 是否同步暴涨 | 改奖励、加长度惩罚、换更强的 verifier |
| 熵骤降到接近 0 | **熵崩塌**，策略收敛到单一解 | 看 entropy 曲线拐点 | 加熵正则、降学习率、增大 group size |
| 长度持续暴涨 | 长度被隐性奖励，或没有长度约束 | 看 response length 的 P95/P99 | 加长度惩罚或截断、DAPO 的 overlong buffer |
| loss 变 NaN | 重要性比爆炸 / MoE 路由抖动 | 看 ratio 的 max 与 P99 | 开 TIS / IcePop；MoE 上换 GSPO 或 Routing Replay |
| GPU 利用率不低但吞吐不涨 | **长尾 rollout 的同步屏障** | 看单步里 rollout 时长的分布 | partial rollout、异步、dynamic sampling |
| 权重同步占了整步的两位数百分比 | refit 走了慢路径（CPU 中转 / reload） | 单独计时 refit | 改 colocate、delta update、直连 NCCL |
| MoE 上训练突然崩 | 专家路由在一次更新后大幅变化 | 看 ratio 方差是否突然放大 | GSPO（序列级比率）或 Routing Replay |
| 异步训练不收敛 | **staleness 过大**，数据过度 off-policy | 看 staleness 分布与 ratio | 收紧 $$\eta$$ / Asynchronous Ratio，配 off-policy 修正 |

### 9.3 上线前的六条自检

1. **确认 off-policy 修正真的开了**。veRL 里 TIS / IcePop 不是默认项，要显式设 `--algo.advantage.is_correction_enable`。用了 FP8 rollout 却没开，训练就是在一个不知道的 off-policy 状态下跑的。
2. **确认 KL 项真的在算**。不少 GRPO 配置默认不带 KL（DAPO 就主张去掉），如果你需要它，先验证 KL 曲线非零。
3. **先跑一遍空转**：固定模型只做 rollout，确认 reward 分布、长度分布、全对/全错占比合理，再开训练。这一步能省掉后面大量无效排查。
4. **量一下 refit 占整步的比例**。超过 10% 就该换同步路径（第 3 章轴三）。
5. **保存每一步的 rollout 样本**。出问题时回溯样本比回溯曲线有用得多。
6. **奖励先做单元测**。规则奖励也要测：边界情况、空输出、超长输出、格式错误，各给什么分。

## 10. 框架还该解决什么：七个容易被忽略的系统问题

第 7 章讲的是**算法侧**的硬问题（训推不一致、MoE 抖动、长尾 rollout）。但一个 RL 训练框架作为**系统**，还有一批必须解决的问题——它们不出现在任何框架的宣传页上，却是决定「能不能真的把训练跑完」的地方。下面七个，每个都给「问题是什么 / 不解决会怎样 / 典型做法 / 选型时该问什么」。

### 10.1 容错与弹性

**问题**：一次 RL 训练跑几天到几周，几十到上千卡。节点故障、被抢占、网络抖动是**必然事件而非例外**——按单卡 MTBF 折算，千卡集群上每天都有节点出问题是常态。

**不解决会怎样**：跑到第 5 天挂一次，如果恢复不正确，等于从零开始（或者更糟：从一个静默错误的状态继续）。

**难在哪**：三件事纠缠在一起——checkpoint 本身很贵（70B 的全量状态是分钟级开销，太频繁会显著拖慢训练）；单控框架把状态放在调度器 actor 里，actor 挂了状态就没了；异步模式下缓冲区里还有「用旧策略生成、尚未训练」的数据。

**典型做法**：Ray 的 actor 重启 + 状态外置；NeMo-RL 的 worker isolation（RL actor 之间进程隔离，一个组件崩不带走全部）；RLinf 的自动在线扩缩容（把 GPU 切换压到秒级，同时保留 on-policy 性质）；AReaL v2.0 把训练/推理/agent/权重更新拆成独立服务，好处之一就是故障域变小。

**选型时该问**：从 checkpoint 恢复后，**第一步的 on-policy 数据是怎么产生的**？是重新 rollout，还是复用缓冲区里那批旧策略数据？

### 10.2 检查点的语义

**问题**：RL 的一个 checkpoint 远不止模型权重。它至少包含：policy 权重、优化器状态、学习率与各系数调度器的进度、数据加载器的位置，以及——最容易被忽略的——**在途的 rollout 数据及其策略版本号**。

**不解决会怎样**：GRPO 的「一个组」是跨样本的。一个 prompt 的 $$G$$ 条 rollout 分散在不同 worker 上，checkpoint 时如果只存了一半，恢复后这个组的组内均值和标准差算不出来，整组样本只能作废（静默地降低有效 batch）。

**典型做法**：veRL 用统一的管理器接管模型/优化器/数据加载器状态；NeMo-RL 的 v0.7 roadmap 里单列了一项 generation trajectory checkpointing——**说明这在大规模 MoE 训练里仍是个活跃的未收敛问题**。

**选型时该问**：checkpoint 里到底存了什么？恢复时在途 rollout 是丢弃、作废整个组、还是续用？

### 10.3 奖励服务与 verifier 基础设施

**问题**：RLVR 时代奖励的主要来源是规则与 verifier（跑单测、比对答案、调工具）。这意味着**奖励计算本身是一个必须被调度、被隔离、被监控的服务**，而不是一个函数调用。

**不解决会怎样**：模型输出一段死循环代码，verifier 卡住，整个 rank 的 rollout 全部空等；或者沙箱没有严格隔离，让训练环境执行了任意代码。

**难在哪**：五个要求同时成立——沙箱隔离（安全）、高并发（不成为新瓶颈）、超时与重试（模型会产出跑不完的东西）、防作弊（模型可能反向猜测测试用例）、结果缓存（同一份输出不必重复验证）。

**典型做法**：veRL 把 reward 抽象成可外挂的 manager 与自定义函数；SkyRL 用独立的 skyrl-gym 承载环境；Harbor 用 Dockerfile + 测试脚本定义任务，并在沙箱里跑；OpenEnv 走 HTTP/WebSocket 服务化并把 MCP 当一等公民；EasyR1 则把奖励简化成一个普通 Python 文件里的 `compute_score()`。

**选型时该问**：verifier 跑在哪个进程/容器里？一个跑不完的样本会不会拖垮整批？奖励结果缓存吗？

### 10.4 样本管理与数据管道

**问题**：prompt 集合不是静态的。需要难度筛选、去重、课程安排，以及决定 replay buffer 里数据的新鲜度策略。

**量化依据**：第 2 章与 7.4 节的实验都指向同一件事——**难度两极化时约 17% 的 rollout 是零梯度的**，正确率低于 0.1 的题有 63% 白算。这不是算法细节，是直接的算力浪费。

**典型做法**：DAPO 的 dynamic sampling（过采样 + 过滤掉组内无方差的 prompt）与 DUPO（复制非零标准差样本补齐 batch）是算法层的解；框架层需要「过采样 + 过滤」作为一等公民配置项，以及独立的数据过滤 worker（EasyR1 有）。MS 侧还要考虑 prompt 去重与难度分档。

**选型时该问**：过采样与过滤是开个开关就用，还是要自己在数据管道里实现？过滤后 batch 缺额怎么补？

### 10.5 系数与超参的调度

**问题**：RL 的超参几乎都不是常数——KL 系数（PPO 经典的自适应 KL）、裁剪范围（DAPO 的 clip-higher 用的是非对称范围）、熵系数、长度惩罚、学习率，通常都要按步数或按指标反馈调整。

**不解决会怎样**：系数调度逻辑散落在用户代码里，实验之间不可比；或者系数写死，模型早期被 KL 拽住、后期又漂移（第 9 章那两个失败模式的常见根因）。

**典型做法**：成熟框架会把自适应 KL 做成内置（按目标 KL 反推系数），把裁剪范围、熵系数暴露成 config 项，并把「按步数/按指标」的调度回调留给用户。

**选型时该问**：KL 系数能不能按目标 KL 自适应？裁剪范围是否支持非对称？有没有统一的调度回调接口？

### 10.6 超长序列与变长 batch

**问题**：长 CoT 与 agentic 让序列长度差异从几倍涨到几十倍，带来三个具体麻烦：padding 浪费、单个超长样本拖垮整批、以及超过 `max_len` 的样本怎么处理。

**典型做法**：sequence packing（NeMo-RL 实测显著减少 padding）；DAPO 的 overlong buffer / token 惩罚（不直接丢弃超长样本，而是给超出部分加惩罚，保住梯度）；按长度分桶；chunked prefill 与 continuous batching 由推理引擎侧承担。

**选型时该问**：超长样本是截断、丢弃，还是加惩罚后继续训练？截断后 token 级 loss mask 怎么算？

### 10.7 可复现性与实验管理

**问题**：RL 的不确定性来源比监督学习多得多——采样温度、rollout 引擎的 batching 顺序、异步带来的 staleness、多 worker 的浮点归约顺序。同一个 config 跑两次结果不同，调参就会退化成玄学。

**典型做法**：TRL 的「稳定层 / 实验层双契约」是从上游缓解这一类问题的一种思路（v1.0 引入，稳定层承诺语义化版本）；此外还要固定种子、把**实际生效的完整 config**（而非提交的 config）写进日志、给数据集打版本。

**选型时该问**：固定种子能不能复现？日志里记的是解析后的最终 config 吗？数据集的版本怎么追踪？

### 10.8 一页速查

下表七行对应上面七个小节，最后一行是交叉项（可观测性的详细清单在第 9 章）：

| 维度 | 一句话自检 | 对应小节 |
|---|---|---|
| 容错 | 挂一次要付多少代价？恢复后第一步的数据从哪来？ | 10.1 / 10.2 |
| 检查点 | 存了权重之外的东西吗？在途 rollout 怎么办？ | 10.2 |
| 奖励服务 | verifier 在哪跑？隔离、超时、缓存、防作弊齐了吗？ | 10.3 |
| 数据管道 | 过采样与难度过滤是开关还是自己写？ | 10.4 |
| 系数调度 | KL / 裁剪范围 / 熵系数能不能自适应？ | 10.5 |
| 变长序列 | 超长样本怎么处理？loss mask 怎么算？ | 10.6 |
| 可复现性 | 固定种子能复现吗？日志记的是最终 config 吗？ | 10.7 |
| 可观测性 | 第 9 章那九个指标，框架默认打点几个？ | 第 9 章 |

**这七个问题有个共同特征**：它们都不影响「跑通一个 demo」，但决定了「能不能把一次训练真的跑完」。选型时用 demo 比较框架，看到的差距往往只有 10%；上面这些地方差起来，是训练能不能收敛的差别。

## 11. 选型决策树

按问题顺序走，不要按「哪个框架 star 多」走：

1. **你在训什么？** 机器人/控制/游戏/多智能体 → 第 6 章那一族（SB3 快速起步，RLlib 上规模，JAX 系做可复现研究；国产栈看 6.1 的 DI-engine / PARL / OpenRL / XuanCe）。**但如果被训的是 VLA 大模型（OpenVLA / π₀ / GR00T）且要接仿真器或真机 → 5.1 的 RLinf（自建底座，通用）或 5.7 的 SimpleVLA-RL（veRL 扩展，快速出结果）**。**如果是 VLM 多模态 → 5.7 的 EasyR1**。纯 LLM 后训练 → 往下。
2. **单机 / ≤30B / 想尽快跑通** → **TRL**（GRPO 稳定 API，colocate vLLM）。显存不够 → **Unsloth**。
3. **30B–70B，一个集群，要异步与多轮** → **OpenRLHF**（直白、TIS 内置）或 **veRL**（placement 灵活、算法最全）。
4. **100B+ / MoE / 已有 NVIDIA 全栈** → **NeMo-RL**（Megatron 6D + 原生生成免转换）；若已全量落地 Megatron 预训练栈 → **Megatron 原生 RL**（连 refit 都省了）。
5. **已确定要异步** → **AReaL**（最彻底、有公开实测、agentic 接入门槛最低）或 **prime-rl**（去中心化/大规模）。
6. **长程多轮 agent** → **verl-agent / rLLM / SkyRL-agent**；环境先用 **OpenEnv 或 Harbor** 定义，别自造。
7. **多智能体 LLM 系统** → **MARTI**（5.7，含树搜索的 MARTI-v2）。
8. **国产芯片 / 昇腾 / ROCm** → **veRL**（plugin.platform 抽象层）、**ROLL** 与 **RLinf**（Ascend/AMD 支持）；AReaL 也有 NPU 支持；经典 RL 场景看 **XuanCe**（昇腾）与 **PARL**（飞桨）。
9. **TPU / JAX** → **Tunix**。

三条跨框架的通用建议：

- **先量化再选型**。跑一遍 `tools/rl_framework_budget.py` 那类量级模型，把你的 $$\text{median}$$ 输出长度、槽位数、互联带宽代进去。如果生成占比 $$f < 0.6$$，异步收益不到 1.4×，不值得付那笔复杂度。
- **把 off-policy 修正当默认项**，尤其是用了 FP8 rollout 或 colocate 的时候。
- **先定环境协议，再定框架**。OpenEnv / Harbor 的迁移成本远低于框架的迁移成本。

---

## 12. 2026 的趋势与未解问题

**八个已经发生的方向**（都有可核验的落地）：

1. **异步成为默认**。AReaL、prime-rl 从设计上就是异步的；veRL v0.9 把异步合入同一套 V1 训练器；OpenRLHF、slime、ROLL、Tunix 全部支持。同步框架（TRL）把异步 GRPO 挂在路线图上。
2. **RL 与预训练共用同一套并行栈**。NeMo-RL 直接用 Megatron-Core；slime 把 Megatron 与 SGLang 深度耦合；veRL 默认走 Megatron-Bridge。「RL 框架自己实现一套并行」这条路基本被放弃了。
3. **训推一体的精度路线**。FP8 rollout 的吞吐增益很可观（有报告称最高约 44%，**待验证**），代价是训推不一致放大，必须配 TIS/MIS 校正——**精度与吞吐现在是同一个旋钮**，不能分开调。
4. **PD 分离进入训练环**。veRL v0.9 支持 prefill/decode 离散化 rollout（NIXL / Mooncake）。推理侧的服务化架构正在反向输入给训练侧。
5. **环境标准化与 MCP 化**。OpenEnv 交出治理权、Harbor 被 SkyRL 与 TRL 采纳，训练环境与生产环境用同一套协议这件事正在变成共识。
6. **具身智能把「仿真器」重新带回 RL 框架**。RLinf 从真机 RL（v0.2）一路做到世界模型（Wan / OpenSora）、Sim-Real 协同训练与多智能体，并被 BEHAVIOR-1K 上游集成（25× 加速）；SimpleVLA-RL 则在 veRL 上用并行渲染 + 二元奖励把这条路走通（ICLR 2026）。**仿真器不再只是环境，而是训练流水线里一个要被调度、要被流水线化的一等公民**——这直接把第 1 章那个「三族划分」的边界又模糊了一层。
7. **veRL 变成上游底座，创新下沉到垂直 fork**。EasyR1（多模态）、SimpleVLA-RL（VLA）、RL-Factory（工具调用）、SkyRL、rLLM、verl-agent 全是 veRL 的 fork 或扩展。**2026 年做 RL 框架，主流做法已经不是从零写一个，而是 fork veRL 后在某个垂直方向做深**。好处是生态收敛、不再重复造轮子；风险是 veRL 上游的一次破坏性变更（如 v0.9 统一训练器）会同时传导给整个下游。
8. **系统侧那些「以前要自己写」的东西正在进入框架主线**（第 10 章那七个问题）。故障隔离成了框架特性（NeMo-RL 的 worker isolation、AReaL v2 的微服务化）；弹性扩缩容被做进调度器（RLinf 压到秒级）；verifier 沙箱被标准化（Harbor / OpenEnv）；连「在途生成轨迹的 checkpoint」都上了 NeMo-RL v0.7 的 roadmap。**框架的竞争焦点正从「支持哪些算法」往「能不能把一次长训练稳稳跑完」迁移。**

**三个还没解决的问题**：

- **异步下的收敛性理论**。staleness 上限（AReaL 的 $$\eta$$、ROLL 的 Asynchronous Ratio）目前都是经验旋钮，没有像同步 PPO 那样的可信域保证。TIS/IcePop/CISPO 都是截断式的补丁，不是理论解。
- **MoE RL 的数值稳定性**。GSPO 免掉了 Routing Replay，但 NVIDIA 的 R3 还在做，说明「序列级比率 vs token 级比率」的最优边界还没划清。
- **长视野 agent 的信用分配与工程相互纠缠**。轨迹 compaction 之后，同一 prompt 的 rollout 产生的可训练 sub-trace 数量与长度差异极大——「一组干净可比的 rollout」这个 GRPO 的基本假设直接崩掉，有团队因此在长视野阶段放弃组相对优化、改回 PPO。这个问题在[信用分配那篇](/2026/09/19/rl-credit-assignment-settlement/)里被列为「欠条四」，它现在**既是算法问题也是系统问题**：框架必须能表达「不整齐的一组 rollout」，而现有的 advantage 抽象都假设它们是整齐的。

---

## 参考与延伸

**论文**
- HybridFlow（veRL）：[arXiv 2409.19256](https://arxiv.org/abs/2409.19256)，EuroSys 2025
- OpenRLHF：[arXiv 2405.11143](https://arxiv.org/abs/2405.11143)
- NeMo-Aligner：[arXiv 2405.01481](https://arxiv.org/abs/2405.01481)
- LlamaRL：[arXiv 2505.24034](https://arxiv.org/abs/2505.24034)
- AReaL：[arXiv 2505.24298](https://arxiv.org/abs/2505.24298)
- DAPO：[arXiv 2503.14476](https://arxiv.org/abs/2503.14476)；DUPO：[arXiv 2507.02592](https://arxiv.org/abs/2507.02592)
- GSPO（Qwen3）：[arXiv 2507.18071](https://arxiv.org/abs/2507.18071)；CISPO（MiniMax-M1）：[arXiv 2506.13585](https://arxiv.org/abs/2506.13585)
- RLinf（M2Flow）：[arXiv 2509.15965](https://arxiv.org/abs/2509.15965)；RLinf-VLA 技术报告获 CVPR 2026 ScaleBot Workshop 最佳论文
- AsyncFlow：[arXiv 2507.01663](https://arxiv.org/abs/2507.01663)；StreamRL：[arXiv 2504.15930](https://arxiv.org/abs/2504.15930)
- ROLL Flash：[arXiv 2510.11345](https://arxiv.org/abs/2510.11345)；ROLL：[arXiv 2506.06122](https://arxiv.org/abs/2506.06122)

**仓库与 DeepWiki**

[DeepWiki](https://deepwiki.com/) 会对 GitHub 仓库自动生成带行号引用的架构文档——读源码之前先过一遍它的 Overview 与组件图，比直接 clone 快得多。下表已逐个验证过页面可用性（唯一没有页面的是 Stoix，其余都实测可达）。

*主力与厂商*

| 项目 | GitHub | DeepWiki |
|---|---|---|
| veRL | [verl-project/verl](https://github.com/verl-project/verl) | [deepwiki](https://deepwiki.com/verl-project/verl) |
| OpenRLHF | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | [deepwiki](https://deepwiki.com/OpenRLHF/OpenRLHF) |
| TRL | [huggingface/trl](https://github.com/huggingface/trl) | [deepwiki](https://deepwiki.com/huggingface/trl) |
| NeMo-RL | [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | [deepwiki](https://deepwiki.com/NVIDIA-NeMo/RL) |
| NeMo-Aligner（前代） | [NVIDIA/NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) | [deepwiki](https://deepwiki.com/NVIDIA/NeMo-Aligner) |
| Megatron-LM | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | [deepwiki](https://deepwiki.com/NVIDIA/Megatron-LM) |
| DeepSpeedExamples（含 DeepSpeed-Chat） | [deepspeedai/DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples) | [deepwiki](https://deepwiki.com/deepspeedai/DeepSpeedExamples) |

*学术界与中国厂商*

| 项目 | GitHub | DeepWiki |
|---|---|---|
| RLinf | [RLinf/RLinf](https://github.com/RLinf/RLinf) | [deepwiki](https://deepwiki.com/RLinf/RLinf) |
| slime | [THUDM/slime](https://github.com/THUDM/slime) | [deepwiki](https://deepwiki.com/THUDM/slime) |
| AReaL | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) | [deepwiki](https://deepwiki.com/inclusionAI/AReaL) |
| ROLL | [alibaba/ROLL](https://github.com/alibaba/ROLL) | [deepwiki](https://deepwiki.com/alibaba/ROLL) |
| ChatLearn | [alibaba/ChatLearn](https://github.com/alibaba/ChatLearn) | [deepwiki](https://deepwiki.com/alibaba/ChatLearn) |
| prime-rl | [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | [deepwiki](https://deepwiki.com/PrimeIntellect-ai/prime-rl) |
| SkyRL | [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) | [deepwiki](https://deepwiki.com/NovaSky-AI/SkyRL) |
| rLLM | [agentica-project/rllm](https://github.com/agentica-project/rllm) | [deepwiki](https://deepwiki.com/agentica-project/rllm) |
| verl-agent | [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) | [deepwiki](https://deepwiki.com/langfengQ/verl-agent) |
| RL-Factory | [Simple-Efficient/RL-Factory](https://github.com/Simple-Efficient/RL-Factory) | [deepwiki](https://deepwiki.com/Simple-Efficient/RL-Factory) |
| EasyR1 | [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1) | [deepwiki](https://deepwiki.com/hiyouga/EasyR1) |
| SimpleVLA-RL | [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) | [deepwiki](https://deepwiki.com/PRIME-RL/SimpleVLA-RL) |
| MARTI | [TsinghuaC3I/MARTI](https://github.com/TsinghuaC3I/MARTI) | [deepwiki](https://deepwiki.com/TsinghuaC3I/MARTI) |
| ms-swift | [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | [deepwiki](https://deepwiki.com/modelscope/ms-swift) |
| LLaMA-Factory | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | [deepwiki](https://deepwiki.com/hiyouga/LLaMA-Factory) |
| Unsloth | [unslothai/unsloth](https://github.com/unslothai/unsloth) | [deepwiki](https://deepwiki.com/unslothai/unsloth) |
| Tunix | [google/tunix](https://github.com/google/tunix) | [deepwiki](https://deepwiki.com/google/tunix) |

*经典 RL 与国产库*

| 项目 | GitHub | DeepWiki |
|---|---|---|
| Stable-Baselines3 | [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | [deepwiki](https://deepwiki.com/DLR-RM/stable-baselines3) |
| CleanRL | [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | [deepwiki](https://deepwiki.com/vwxyzjn/cleanrl) |
| Ray（含 RLlib） | [ray-project/ray](https://github.com/ray-project/ray) | [deepwiki](https://deepwiki.com/ray-project/ray) |
| Acme | [google-deepmind/acme](https://github.com/google-deepmind/acme) | [deepwiki](https://deepwiki.com/google-deepmind/acme) |
| Sample Factory | [alex-petrenko/sample-factory](https://github.com/alex-petrenko/sample-factory) | [deepwiki](https://deepwiki.com/alex-petrenko/sample-factory) |
| TorchRL | [pytorch/rl](https://github.com/pytorch/rl) | [deepwiki](https://deepwiki.com/pytorch/rl) |
| Tianshou | [thu-ml/tianshou](https://github.com/thu-ml/tianshou) | [deepwiki](https://deepwiki.com/thu-ml/tianshou) |
| Dopamine | [google/dopamine](https://github.com/google/dopamine) | [deepwiki](https://deepwiki.com/google/dopamine) |
| JaxMARL | [FLAIROx/JaxMARL](https://github.com/FLAIROx/JaxMARL) | [deepwiki](https://deepwiki.com/FLAIROx/JaxMARL) |
| Brax | [google/brax](https://github.com/google/brax) | [deepwiki](https://deepwiki.com/google/brax) |
| PufferLib | [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib) | [deepwiki](https://deepwiki.com/PufferAI/PufferLib) |
| Gymnasium | [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | [deepwiki](https://deepwiki.com/Farama-Foundation/Gymnasium) |
| Stoix | [EdanToledo/Stoix](https://github.com/EdanToledo/Stoix) | 暂无页面 |
| DI-engine | [opendilab/DI-engine](https://github.com/opendilab/DI-engine) | [deepwiki](https://deepwiki.com/opendilab/DI-engine) |
| PARL | [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL) | [deepwiki](https://deepwiki.com/PaddlePaddle/PARL) |
| OpenRL | [OpenRL-Lab/openrl](https://github.com/OpenRL-Lab/openrl) | [deepwiki](https://deepwiki.com/OpenRL-Lab/openrl) |
| XuanCe 玄策 | [agi-brain/xuance](https://github.com/agi-brain/xuance) | [deepwiki](https://deepwiki.com/agi-brain/xuance) |

*推理引擎与环境协议*

| 项目 | GitHub | DeepWiki |
|---|---|---|
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | [deepwiki](https://deepwiki.com/vllm-project/vllm) |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | [deepwiki](https://deepwiki.com/sgl-project/sglang) |
| OpenEnv | [huggingface/OpenEnv](https://github.com/huggingface/OpenEnv) | [deepwiki](https://deepwiki.com/huggingface/OpenEnv) |

**本文的自算脚本**
- `tools/rl_framework_budget.py`：槽位木桶 / 预算截断 / refit 量级 / 显存账 / Amdahl 上界（选型量化，见第 2、3 章）
- `tools/rl_obj_numeric_lab.py`：GRPO 零梯度组 / 四种裁剪的梯度权重 / 训推不一致与三种修法（见 7.4 节）

两个脚本都是纯 numpy，CPU 可跑。文中所有自算数字均为量级模型，已明确标注，请勿当作任何框架的实测吞吐。

---

*上一篇：[RL 信用分配与奖励系数（2025–2026）](/2026/09/19/rl-credit-assignment-settlement/) ｜ 下一篇：待定*
