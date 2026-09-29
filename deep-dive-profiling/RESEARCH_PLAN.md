# 训练 / 推理 Profiling 方法学 · 补充计划

> 本文件是计划与提纲，不是正文。正文按项目惯例**每篇直接进 `_posts/`**，命名 `profiling-NN-*.md`，
> `categories: 性能分析`。本目录只放计划与脚本说明，不放文章副本。

## 0. 定位：填的是哪块空白

2026-09-29 全站盘点结论（见 `.workbuddy/memory/2026-09-29.md`）：

| 层次 | 现有内容 | 状态 |
|:---|:---|:---|
| GPU 工具栈总纲 | `_posts/2026-08-26-nv-gpu-profiling-toolkit.md`（465 行） | 已覆盖，**很深** |
| 单算子 / kernel | `op-00` `op-03` `op-04` `op-13` | 已覆盖，**很深** |
| 编译器 cost model | `chip-for-ai-compiler-10` | 已覆盖，中等 |
| **训练栈层** | `dist-train 00–08` 纯解析、零实测；Megatron 模块 C 三篇未写 | **空缺** |
| **推理服务层** | 只有 nv-toolkit §5 与投机解码 07 侧面提及 | **空缺** |

本系列只补后两格，以及它们共同依赖的一层——**度量口径本身**。

### 0.1 与已有内容的边界（硬约束，写正文时逐条对照）

已有文章**已经讲过**、本系列**不重复**的内容：

- **`nv-gpu-profiling-toolkit`**：五层工具栈地图（快速观测 / 时间线 / kernel 深潜 / 框架内嵌 / CUPTI）、
  `nvidia-smi dmon·pmon` 与 NVML `utilization` 口径、DCGM 定位、nsys 基本采集命令与 `nsys stats` 报表、
  ncu 的 SOL 与 replay 语义、`ncu --clock-control` 对可比性的影响、`nvidia-smi utilization 100% ≠ 打满`、
  「decode 阶段 SM 低 + HBM 近峰才是健康形态」、基础的坑清单。
- **`op-03`**：跨硬件工具链选型（NVIDIA / ROCm / VTune / 昇腾 msprof / 国产）、跨硬件通用层、实操对照。
- **`op-04`**：单 kernel 的「nsys 全局 → ncu 微观 → 假设-实验」三层下钻、算子级常见陷阱。
- **`chip-for-ai-compiler-10`**：ncu 计数器 ↔ 硬件单元的对照词典、分层 Roofline、参数可信度分级（datasheet=C / 微基准=B / ncu=A）。

本系列**只做**这三件已有内容没做的事：

1. **口径**：数字是怎么被算出来的、以及哪些数字根本不可比（MFU/HFU、FLOPs 记账、timer 语义、goodput）。
2. **栈级归因**：从「单个 kernel 慢」上升到「一次迭代慢在哪、哪张卡是掉队的、通信有没有真的被藏住」。
3. **服务级测量**：推理服务的负载模型与 SLO 口径，以及「测出来的到底是模型还是缓存」。

### 0.2 一条容易被忽略的立场

本博客已有一条约定（2026-09-29）：Megatron 系列正文**不写围绕硬件展开的内容**，判断标准是
「这段内容换成在 CPU 上讲还成不成立」。那条约定是**针对 Megatron 系列**定的，
因为该系列定位是源码走读。

本系列定位就是性能分析，硬件与工具是主题本身，不受该约定约束——
与已发布的 `nv-gpu-profiling-toolkit`（同为 `categories: 性能分析`，通篇 GPU 内容）保持一致。

但有一条**继承下来并加强**：**没有实验就没有，不要凑**。
所以本系列所有可执行实验都必须能在**纯 CPU** 上跑出真实数字（见 §1.2）。

---

## 1. 实验方法论（本机无 GPU 约束）

### 1.1 数字只有三个合法来源

沿用 Megatron 系列已建立的 A/B/C 分级，措辞对齐，避免两套标准：

| 级别 | 来源 | 要求 | 本系列实例 |
|:---|:---|:---|:---|
| **A** | `tools/profiling_bench/` 下脚本实跑产出 | 脚本随文章公开，参数可改，他人可复算 | FLOPs 三口径交叉验证、合成 trace 聚合、队列模型、timer 语义复现 |
| **B** | 官方文档 / 论文 / 官方仓库 benchmark | 必须挂链接与核验日期 | Nsight 版本行为、AIPerf 指标定义、nccl-tests busbw 公式、已发表 MFU 数字 |
| **C** | 解析模型 / 模拟 | 明确标注为模型而非测量 | α-β 通信代价、overlap 收益上界、队列论的 TTFT 分位 |

**纪律**：吞吐、MFU、busbw、TTFT 这类必须真机才能定的数字，**一律走 B 级并写清卡型、规模、软件版本**，
不假装是自己测的。凡引用无法核验的二手数字，显式标注「待验证」。

### 1.2 三条替代原则（无 GPU 怎么办）

1. **口径可验证**：FLOPs 记账、MFU 分母、busbw 换算、TTFT/TPOT 定义都是**纯算术**，
   可以在 CPU 上用真实参数跑通并与官方公式逐项对齐。这正是「尺子做准」的可行部分。
2. **聚合可验证**：多 rank trace 的聚合（temporal breakdown、idle 三分、comm/compute overlap）
   是对**已有 JSON 的变换**。我们可以按 PyTorch/Kineto 的 chrome trace 字段规范**合成 trace**，
   跑通聚合逻辑，并用合成数据里已知的真值反向验证实现——包括验证我们自己踩到的坑（见 §4.1）。
3. **行为可复现**：`torch.cuda.synchronize()` 破坏重叠、队列在饱和后把压力转成 TTFT 这类机制，
   可以在 CPU 上用线程/队列把语义复刻出来，量出方向与量级，不冒充真机绝对数。

**明确不做**：不编造 profiler 截图、不写「实测得到 X TFLOPS」而实际没有机器、不把解析模型包装成测量结果。

---

## 2. 系列总览

三篇 + 一个已有的前置总纲，构成一条完整的下钻链：

```
[nv-gpu-profiling-toolkit]  工具是什么（已有，不重复）
            │
            ▼
[profiling-00]  度量口径：数字是怎么来的，哪些数字不可比
            │
    ┌───────┴────────┐
    ▼                ▼
[profiling-01]    [profiling-02]
训练栈 profiling   推理服务 profiling
一次迭代慢在哪     这个服务能扛多少
```

| # | 标题（拟定） | 回答的问题 | 目标行数 |
|:---|:---|:---|:---|
| 00 | 度量口径：MFU、HFU、Timer 与 goodput，先把尺子做准 | 我看到的那个数字，是谁算的、怎么算的？ | 500+ |
| 01 | 训练栈 profiling：从时间线到掉队卡，一次迭代到底慢在哪 | 慢在计算、通信、数据还是某一张卡？ | 550+ |
| 02 | 推理服务 profiling：TTFT、goodput 与负载模型 | 这个服务在什么负载下、对谁、不满足什么 SLO？ | 500+ |

三篇都是 `categories: 性能分析`，带 `mathjax: true`。互相之间用 `> **系列导航**` 引用块连成一条线，
并在末尾指回 `nv-gpu-profiling-toolkit` 作为工具总纲。

---

## 3. 逐篇提纲

### profiling-00 · 度量口径：MFU、HFU、Timer 与 goodput

**为什么排第一**：后面两篇的每一个结论都建立在这把尺子上。口径不清，“优化了 20%” 就是自欺。

**骨架**

1. **为什么先讲口径**：同一份代码，三种 MFU 口径能报出三倍差距；本文只做一件事——把尺子做准
2. **MFU 到底怎么算**
   - 分母：dtype 决定峰值（BF16 / FP8 / FP4 不是同一个数），**分母口径不匹配则比值无意义**
   - 分子：6P 的来历（前向 2P + 反向 4P），attention 的 $$12 n_{layer} d_{model} S$$ 项
   - **causal mask 只算一半，但口径必须按整张矩阵算**——为什么（跨论文可比性）
   - HFU 与 MFU 的关系：HFU ≥ MFU 恒成立；full recompute 把 6P 变成 8P，HFU/MFU = 4/3
   - 已发表的锚点（B 级引用）：GPT-3 21.3%、MT-NLG 30.2%、Gopher 32.5%、PaLM 46.2%、Llama 3 38–43%；
     30–50% 算好，<25% 提示配置有问题
   - **反直觉**：A100 → H100 峰值 FLOPs 涨 3.1×，显存带宽只涨 2.1×，
     访存受限的子层（norm / softmax / optimizer）不跟着涨，**吞吐上升而 MFU 下降是正常的**
3. **Megatron 官方的 FLOPs 记账（源码走读）**
   - `megatron/training/training.py:802` `num_floating_point_operations`：逐架构族拆分
     （attn / MLP / MoE / Mamba / GDP / GDN / MTP / logits），`training.py:1007` `return flops_fwd * 3`
   - **不含重计算系数** → 官方 log 的 `TFLOP/s/GPU` 是 **MFU 口径**，不是 HFU
   - **THD（packed）路径**：`seqlen_squared_sum_in_batch` / `total_real_tokens_in_batch`，
     让 padding 与因果分块不再计入 FLOPs——**同一模型换成打包后，MFU 会变，变的不是性能而是分母**
   - 与 `tools/megatron_bench/flops.py` 的三口径交叉验证（逐算子精确 / 官方复刻 / 6ND），把偏差摊开写
4. **计时器在骗人（源码级）**
   - `megatron/core/timers.py:109` `Timer` 用的是 `time.time()` + **`torch.cuda.synchronize()`**，
     不是 CUDA Event——**测量动作本身把重叠毁掉了**，所以 timer 求和 ≠ 时间线
   - `timers.py:171` `reset()` **不重置** `_active_time`（构造至今累计），`elapsed` 与 `active_time` 的比就是占空比
   - `timers.py:270` 多 rank 聚合走 `all_gather`；`timers.py:318` 的 min/max **会过滤掉 0 值**，
     未进入该 timer 的 rank 静默消失
   - min 与 max **可能来自不同的 rank**：把各项 max 相加得到的「总时长」不是任何一张卡的真实值，只是上界
   - `training.py:3503` `interval-time` 用 `barrier=True`：**barrier 本身进入了被测量的区间**
   - `training.py:3530` 的 `elapsed time per iteration` 与 `--log-throughput` 的口径关系
5. **配置开关的语义**：`--timing-log-level` / `--timing-log-option {max,minmax,all}`
   （`megatron/training/config/training_config.py:276`，默认 `minmax`），什么时候该开 `all`
6. **goodput：比 MFU 更贴近「有没有在干正事」**
   - Megatron v0.20 原生 OTel（`megatron/core/telemetry/`，2026 新增）：span group 划分
     （`job / checkpoint / evaluate / model_init / load_checkpoint / step / forward_backward / optimizer`
     + `microbatch / layer / communication / activation_offload / data_loading / first_iteration / trace_region / inference`）、
     `--otel-enable` / `--otel-service-name` / `--otel-span-groups`（含 `default / per_step / profiling / all` 预设）、
     日志桥接到 trace 关联
   - `is_goodput_span=True` 标了什么（checkpoint save/load、weight-hash check、RL 各阶段）
   - goodput 与 MFU 的分工：MFU 管「算得快不快」，goodput 管「时间花在正事上没有」
7. **实验卫生**（这一节决定读者能不能复现别人）
   - warmup 与取中位：官方 `--profile` 的 `wait=start-1 / warmup=1 / active=end-start` 是怎么定的
   - 固定随机性、去前 N step、重复次数、**锁频与不锁频各适合回答什么问题**
   - 可比性清单：换配置后如何确认「只是变快了，结果没变」（loss 曲线落在噪声内）
8. **一张「什么不能用什么测」的表**：每个工具/指标的适用边界与失效条件
9. **批判与展望**：口径之争（MFU 作为跨栈比较指标的局限）、goodput 缺乏统一定义、
   FP8/FP4 时代的分母问题
10. **Lab Exercises**（纯 CPU 可跑）

**产出**

- 复用并扩展 `tools/megatron_bench/flops.py`（已有 `mfu()` / `cross_validate()` / `approx_6nd()`）
- 新增 `tools/profiling_bench/timer_semantics.py`：在 CPU 上复刻 `Timer` 的「同步式计时 vs 异步发射」，
  量化同步点对重叠的破坏（合成数据 + 真值对照）
- 新增 `tools/profiling_bench/goodput_calc.py`：从 span 时长表算 goodput，并把「goodput 高但 MFU 低」与
  「MFU 高但 goodput 低」两种形态构造出来

---

### profiling-01 · 训练栈 profiling：从时间线到掉队卡

**骨架**

1. **从 kernel 到迭代**：`op-04` 讲完了单 kernel，这一篇回答栈级问题——一次迭代的时间去哪了
2. **三个观测面，管三种时间尺度**（本文的结构骨架）
   | 观测面 | 时间尺度 | 回答什么 | 代价 |
   |:---|:---|:---|:---|
   | OTel spans（Megatron v0.20 原生） | 全训练期 | 长周期漂移、goodput、checkpoint 占比 | 极低，可常开 |
   | Perfetto / NVTX 区间（`get_nvtx_range`） | 单次迭代 | 阶段分解、RL 各阶段占比 | 低 |
   | nsys / ncu / torch.profiler | 单次迭代内的微秒 | 谁在等谁、单 kernel 为什么慢 | 高，需窗口化 |
   **核心论点：这三层不是替代关系。用 nsys 看整轮训练的漂移、用 OTel 看单 kernel 都是用错层级。**
3. **一次迭代的解剖**：microbatch 切分 → forward → backward → grad reduce → optimizer step → 日志/checkpoint，
   逐段的观测点与正常时长占比
4. **通信到底藏住了没有**
   - Megatron 的 overlap 相关开关与它们的观测点（`--overlap-grad-reduce`、
     `--overlap-param-gather`、`tp_comm_overlap` 全家桶）
   - **comm / compute overlap 百分比**怎么算才有意义（重叠区间 / 通信总时长 vs 重叠区间 / 迭代时长——
     两种分母给出完全不同的结论）
   - 用 HTA 的 `get_comm_comp_overlap()` 语义对齐我们的合成实现
   - **收益上界**：可被掩盖的通信时长 / 总迭代时长。承认这是上界，实测必然更低
5. **GPU 饿了的三种成因与区分方法**（本文最实用的一节）
   - host-wait（数据管道 / Python 侧 / launch 开销）、kernel-gap（流内依赖等待）、unknown
   - 为什么必须三分：这三类的修法完全不同，混在一起会去优化错误的东西
6. **掉队卡（straggler）诊断**
   - 为什么「平均值」在分布式里几乎总是骗人的：一个掉队 rank 决定整步时长
   - nsys 2026.5.1 的 **NCCL straggler analysis recipe**（B 级引用 + 判读方法）
   - hta 的 temporal breakdown 让每 rank 的 compute / comm / idle 三色并排
7. **通信微基准怎么读**：nccl-tests 的 `algbw` vs `busbw`
   - AllReduce ring：`busbw = algbw × 2(N-1)/N`；AllGather / ReduceScatter：`busbw = algbw × (N-1)/N`；
     N=8 时系数 1.75
   - **`busbw` 在 NVLS / CollNet / 分层算法下不再对应单一物理链路**，不要拿它去反推线速
   - 验收读法：flat 是好，某个档位出现 cliff 就是把故障定位到了那一层引入的链路
8. **多 rank trace 的聚合：HTA 及其等价物**
   - HTA 能做什么：temporal breakdown、idle 三分、kernel breakdown、comm-comp overlap、
     增强计数器（内存带宽 / 队列长度）、trace diff、critical path、CUPTI 计数器
   - Megatron 官方 `--profile` 导出的 `torch_profile/rank-N.json.gz` **正好是 HTA 的输入格式**——
     两者是无缝衔接的，这一点要写清楚
   - 换配置后怎么做 A/B：trace diff 而不是靠感觉
9. **八种常见病的排除树**（表格：症状 / 可能原因 / 验证方法 / 无 GPU 时的替代手段）
   —— 这一节直接吸收 Megatron 计划模块 C·15 的排除树并补全
10. **一次完整的调优记账**：从一个 baseline 配置出发，逐笔写「改了什么 / 收益多少 / 代价多少 / 怎么确认」
11. **批判与展望**：trace 的体积与采样偏差、critical path 的实用性争议、
    OTel 语义约定的碎片化（nemo-lens 的 `MEGATRON_OTEL_*` vs `NEMO_LENS_*` 双前缀本身就是信号）
12. **Lab Exercises**

**产出**

- 新增 `tools/profiling_bench/trace_agg.py`：合成多 rank chrome trace → temporal breakdown +
  idle 三分 + comm/compute overlap + 掉队卡识别。**用合成数据里预置的真值反向验证实现**
- 新增 `tools/profiling_bench/overlap_bound.py`：α-β 模型下的 overlap 收益上界（复用 `megatron_bench/comm.py`）
- 复用 `tools/rl_framework_budget.py` 的 Amdahl / 木桶思路（该脚本已踩过一次「拍脑袋模型导致负空闲率」的坑，
  本系列必须避免同类错误，见 §4.2 教训）

---

### profiling-02 · 推理服务 profiling：TTFT、goodput 与负载模型

**骨架**

1. **「能出字」不是标准**：服务侧要回答的是「多少并发下、对谁、不满足什么 SLO」
2. **四个数字的定义与主导项**（这一节决定后面全部推理）
   - TTFT（队列 + prefill 主导）、TPOT / ITL（decode 主导）、E2EL、吞吐（req/s 与 tok/s）
   - **ITL 是逐 token 间隔，TPOT 是除首 token 外的平均**——两者在长尾下不是一回事
   - 分位而非均值：为什么要报 p50 / p90 / p99，以及 `--percentile-metrics` / `--metric-percentiles` 的用法
   - **goodput**：满足 SLO 的那部分吞吐，与「峰值吞吐」的差别
3. **负载模型：你测的是哪种负载**
   - `--max-concurrency`（固定并发数）vs `--request-rate`（固定到达率）vs 到达分布
     （constant / Poisson / gamma，可调突发度）——**三者不可互推**
   - `--num-prompts` 与并发的关系（保持约 10× 才有稳定均值）
   - 固定种子的合成负载 vs trace replay（ShareGPT / Mooncake / Baseten / WEKA AgentX）
   - `--ignore-eos` 对可比性的影响
4. **客户端不能是瓶颈**（2026 年这条有了明确答案）
   - vLLM 自带 bench 是**单进程 asyncio**，高并发下测量工具自身与被测服务抢资源
   - **AIPerf**（GenAI-Perf 的指定继承者，2026-09-18 发布）改为多进程 + ZMQ：worker 生成负载、
     独立进程处理结果，客户端不再成为瓶颈
   - GenAI-Perf 已弃用（不再加新特性）、LLMPerf 已于 2025-12-17 归档
   - **纪律：换工具就不能比数**——同一批配置必须在同一工具内比较
5. **并发扫描与拐点判读**
   - 扫描法：并发按倍数抬升，看聚合吞吐何时走平、TTFT 何时起飞
   - **P99/P50 比值是队列压力表**：比值稳定说明缓存是按负载配的，比值冲高说明 KV cache 成了约束——
     此时改重试与超时都没有用
   - 两个上限谁先到：`--max-num-seqs`（调度上限）与 KV cache 容量（显存上限）——
     **哪个先 bind 取决于请求有多长**，超过后是排队还是抢占（preempt + 重算）也是两种形态
6. **被测对象其实是谁**：网关限流、路由 fallback、多租户共享卡都会改变读数；
   量到的是「这一层的行为」还是「模型的容量」
7. **最大的坑：量了缓存，没量模型**
   - 前缀缓存在服务的默认下开着，用同一批 prompt 复测会把 TTFT 从 265 ms 测成 61 ms，
     这个「改善」毫无意义；变种子或重启服务关缓存，二者必选其一
   - warmup、模型冷启动、连续多轮会话导致的 KV 占用累积——受控压测覆盖不到，要单独说
8. **decode 阶段的 profile 形态**：为什么低 SM 利用率 + 接近峰值的 HBM 带宽是健康的
   （与 nv-toolkit §5 呼应但不重复：那里讲形态，这里讲**怎么用指标确认形态**）
9. **一张决策表**：单机冒烟 / 容量规划 / 回归门禁 / 线上排障 → 各自该用哪个工具、哪组参数、看哪几个数
10. **批判与展望**：LLM 服务的 SLO 缺乏行业统一口径、goodput 的定义人各一套、
    agentic 长会话对「请求」这一计量单位的冲击
11. **Lab Exercises**

**产出**

- 新增 `tools/profiling_bench/serving_model.py`：纯 numpy 队列模型。输入到达模式（constant / Poisson / gamma）、
  批上限、KV 容量、prefill/decode 单位代价；输出 TTFT / TPOT / E2EL 分位、goodput-到-SLO、
  **P99/P50 拐点位置**。用途有二：① 在没有服务的情况下把「拐点在哪、为什么在那儿」讲清楚；
  ② 生成的 JSON 字段与 vLLM / AIPerf 输出对齐，验证我们对指标口径的理解
- 新增 `tools/profiling_bench/serving_schema.py`：把上述输出按 vLLM `--save-result` 的字段名落盘，
  便于逐字段对照

---

## 4. 实验与脚本清单

统一放在 `tools/profiling_bench/`，与 `tools/megatron_bench/` 并列。
数值代码一律用 miniconda python（managed python 3.13 无 numpy）。

| 脚本 | 用途 | 服务于 | 状态 |
|:---|:---|:---|:---|
| `mfu_accounting.py` | 分子/分母换来换去，MFU 差多少；官方日志换算 | 00 §2–3 | 已落地 |
| `timer_semantics.py` | 复刻同步式计时对重叠的破坏，量出方向与量级 | 00 §4 | 已落地 |
| `goodput_calc.py` | span 时长表 → goodput；构造 goodput/MFU 背离的两种形态 | 00 §6 | 已落地 |
| `trace_agg.py` | 合成多 rank chrome trace → temporal / idle 三分 / overlap / straggler；也能读真实 chrome trace | 01 §4–6、§8 | 已落地，真值反算 0.00% |
| `overlap_bound.py` | `D* = n·α·BW` 交叉点、`max(C,M)` 收益上界与 η 敏感度、TP/SP 对账、algbw↔busbw | 01 §4、§7 | 已落地 |
| `serving_model.py` | roofline 步长队列模型 → TTFT/TPOT 分位、goodput 边界、拐点；覆盖/到达分布/抢占三组对照 + 前缀缓存解析式 | 02 §2–3、§5、§7 | 已落地，自校验 8 项全精确通过 |
| `serving_schema.py` | 输出字段与 vLLM `--save-result` 对齐；含 A/B/C 三组能力边界表与 ITL–TPOT 分母演示 | 02 §2 | 已落地 |

辅助工具（放 `tools/`，不属于本系列专属，但本系列依赖）：

| 脚本 | 用途 |
|:---|:---|
| `tools/check_post.py` | kramdown 离线渲染校验：先剥离围栏代码块，再验公式转义与表格数（2026-09-29 新增） |
| `tools/scan_garbled.py` | 乱码词 / 可疑 Unicode / 单 `$` 公式扫描（白名单已补 `²³`、`ηχψ`、`∘`） |
| `tools/fix_quotes.py` | 半角 `"` → 中文弯引号，按空行分段配对；**不会动 YAML front matter** |

复用已有的：`tools/megatron_bench/flops.py`（→ `mfu()` / `cross_validate()` / `approx_6nd()`）、
`megatron_bench/comm.py`（α-β，**SP 分支已于 2026-09-29 校准**）、`megatron_bench/configs.py`（模型与硬件参数表）。

### 4.1 合成 trace 的自校验设计（关键）

`trace_agg.py` 不能只是「跑通了」。做法是：

1. 按已知的真值**构造** trace：选定的 compute / comm / idle 占比、选定的掉队 rank、选定的重叠区间；
2. 用聚合器反算这些量；
3. **断言反算值 == 构造值**，容差写进输出。

只有这样，当文章说「overlap 只有 12%」时读者才相信公式没写错。
顺带能验证一个真实存在的坑：**重叠区间该用什么分母**——
换分母会得到完全不同的结论，而两种写法在社区文章里都能看到。

### 4.2 写量化模型的两条硬教训（已在本项目踩过）

- 第一版模型在极端参数下必须**自检不出现负数、不出现大于 1 的比率**。
  `tools/rl_framework_budget.py` 第一版就是因为「槽位时间 = 均值 + max×0.5」拍脑袋，
  跑出了**负空闲率**。队列模型（`serving_model.py`）里 goodput、利用率、重叠率都是比率，同一风险。
- **做对照实验前先确认基线归零**。同一项目里另一次实验因为基线本身有 11% 的干扰，
  自变量的效果被污染。（该教训来自 `tools/rl_obj_numeric_lab.py` 的迭代记录。）

---

## 5. 2026 事实底座（可直接引用，核验日期 2026-09-29）

写正文时**逐条复核版本号**，不要凭记忆。

**工具链现状**

- **CUDA Toolkit 13.4**（2026-09）：Rubin GPU 预览支持、Windows on Arm、cuBLASLt 对 MoE Grouped GEMM 的
  动态 SM 调度、FP8 的 VEC32/VEC128 UE8M0 缩放布局
- **Nsight Systems 2026.5.1**：CUDA 13.4 / Rubin / WoA 支持；NVTX range 投影进 All Streams 层级；
  NVTX binary payload 可导出为 **SQLite / Arrow / Parquet** 关系表（→ 可以程序化分析，不再是只能肉眼看）；
  **NCCL straggler analysis recipe**（分析集合通信时序、识别反复拖慢 communicator 的 rank）；
  `--pytorch=functions-trace-shapes`（给 trace 的函数加上张量形状与训练参数）；
  CPU Topdown metric sets；NIC 高频指标（DOCA Telemetry）；S3 访问汇总 recipe；
  多节点对齐的 **vClock 插件**（实验性，无需改系统时钟 / 特权）；**Nsight Cloud**（headless 远程看报告）
- **Nsight Compute 2026.3**：Tile IR 支持（CUDA Tile 工作负载，源码页可关联 Tile IR 与生成代码）、
  OptiX 寄存器溢出信息改进、Nsight Copilot 增强
- **Nsight Python 1.0**（新）：Python 内核剖析接口，装饰器 + 上下文管理器，
  一个脚本里完成内核基准、架构指标采集、**防 GPU 降频**与可视化
- **HTA（Holistic Trace Analysis）** v0.6.0（MIT；PR #341 “Release v0.6.0” 日期 2026-04-22；
  要求 Python ≥ 3.10；末次核验 2026-09-29）：
  输入是 **PyTorch Profiler / Kineto** trace，同一 job 的 trace 必须放进一个独立目录（`trace_dir`）。
  README 列出的 features：temporal breakdown（compute/comm/memory/idle 四色分解）/
  idle 三分（**host wait / kernel wait / unknown** ← 官方措辞）/ kernel breakdown（各 rank 最耗时的小 kernel）/
  comm-comp overlap / kernel duration distribution / frequent kernel patterns /
  CUDA kernel launch statistics / 增强计数器（内存带宽利用率 + 每 stream 的未决操作数即队列长度）/
  trace comparison（diff）/ CUPTI 计数器分析（实验性，把 kernel 指标归因到 PyTorch 算子做 roofline）。
  **critical path 在代码里（`critical_path_analysis.py`）但未列入 README feature 列表**——
  引用时不要写成“官方支持的关键路径分析”。
  另有 Triton kernel backend 解析、NCCL `seq_num` / `collective_name` 解析、AMD 与 MTIA 支持。
  **修正**：此前计划里记的 v0.5.0 已过期，实际已是 v0.6.0。
  **修正**：idle 三分的官方第二类是 `kernel wait`，不是 early plan 里写的 `kernel-gap`
  （两者指同一现象：设备空闲、host 已 post 下一个 kernel）。正文与脚本统一用 HTA 措辞。

**推理压测**

- **AIPerf**（`ai-dynamo/aiperf`）：GenAI-Perf 的**指定继承者**，2026-09-18 发布，全新重写而非小版本更新；
  文档列为 v0.12.0（2026-06 时为 v0.10.0）；多进程架构 + ZMQ 协调，避免客户端成为瓶颈；
  15+ endpoint 类型；到达模式 constant / Poisson / gamma + 可调突发度 + 渐进抬升；
  trace replay（Mooncake / Baseten / WEKA AgentX）；TTFT / ITL / 请求延迟 / 输出吞吐 + p50/p90/p95/p99；
  有 DCGM 或 pynvml 时附 GPU telemetry；支持多节点分布式压测
- **GenAI-Perf 已弃用**（README 声明不再加新特性，指向 AIPerf）；**LLMPerf 于 2025-12-17 归档**
- **vLLM bench**：`vllm bench serve`，`--backend openai-chat`，`--dataset-name random`、
  `--random-input-len` / `--random-output-len`、`--ignore-eos`、`--max-concurrency`、
  `--percentile-metrics ttft,tpot,itl,e2el`、`--metric-percentiles 50,99`、`--save-result`（落 JSON）。
  注意：**单进程 asyncio**，高并发下测量精度受限；`--no-enable-prefix-caching` 是 serve 侧开关，
  bench 侧关不掉

**口径锚点**

- MFU 由 PaLM 报告引入；已发表值：GPT-3 21.3%、MT-NLG 30.2%、Gopher 32.5%、PaLM 46.2%、
  Llama 3 约 38–43%（8,192 卡 43%，16,384 卡 41%）
- HFU ≥ MFU 恒成立；full activation recompute 把每 token 从 6P 提到 8P，HFU = MFU × 4/3
- 实操阈值：大模型训练 30–50% MFU 算好；<25% 提示配置问题（气泡过大、TP 跨机、micro-batch 过小、
  checkpoint 开销）
- 分母必须与实际精度匹配（BF16 / FP8 / FP4 峰值差异可达数量级）
- decode 阶段 MFU 极低是**正常的物理结果**（访存受限），不是故障

**通信**

- nccl-tests：`algbw = size / time`；ring AllReduce `busbw = algbw × 2(N-1)/N`；
  AllGather / ReduceScatter `busbw = algbw × (N-1)/N`；N=8 时系数 1.75
- NVLS / CollNet / 分层算法下 `busbw` 不再对应单一物理链路流量
- 验收读法：小消息看 `time`（延迟），大消息看 `busbw`（吞吐）；随节点数缩放时 busbw 应**保持平坦**，
  某一档出现 cliff 即把故障定位到该档引入的链路层

**Megatron 侧（源码基准 v0.20.0 / commit `60e039626`，2026-08-21 main）**

- `megatron/core/timers.py`（485 行）：`Timer`（109）、`active_time`（212）、`Timers._get_elapsed_time_all_ranks`（270）、
  `_get_global_min_max_time`（318）、`log`（422）、`write`（452）
- `megatron/training/training.py`：`num_floating_point_operations`（802，`return flops_fwd * 3` 在 1007）、
  吞吐日志（3503–3534）、torch.profiler 集成（4436–4464）、`profile_ranks`（3496）
- `megatron/training/config/training_config.py:276`：`timing_log_option: Literal["max","minmax","all"] = "minmax"`
- `megatron/core/telemetry/`（2026 新增，版权头为 2026）：`span_groups.py`（`MegatronSpanGroup`）、
  `training_metrics.py`（8 个指标：`step_duration_ms` / `loss` / `throughput_tflops` / `grad_norm` /
  `skipped_iters` / `learning_rate` / `tokens_per_sec` / `memory_allocated_gb`）、`fallbacks.py`（无 nemo-lens 时全 no-op）
- OTel 埋点分布（8 个文件、约 72 处引用）：`training/checkpointing.py`、`training/training.py`、
  `core/distributed/distributed_data_parallel.py`、`core/pipeline_parallel/p2p_communication.py`、
  `core/pipeline_parallel/schedules.py`、`core/pipeline_parallel/fine_grained_activation_offload.py`、
  `core/transformer/transformer_layer.py`、`core/ssm/mamba_layer.py`
- OTel 初始化：`training/global_vars.py:490` `_set_telemetry`，env 前缀 `MEGATRON_OTEL`，
  fallback 前缀 `NEMO_LENS`；只在 enabled 时才 `nvmlInit()`
- 官方 profile 示例：`examples/llama/train_llama3_8b_h100_fp8.sh:168-171`
  （`--log-throughput` / `--profile` / `--profile-step-start 4` / `--profile-step-end 6`）
- RL 侧的 NVTX 区间命名（`megatron/rl/rl_utils.py`）：`rl/offload-optimizer-before-inference`、
  `rl/rollout-collection`、`rl/get-logprobs`、`rl/prefetch-weights-to-gpu`、
  `rl/synchronize-cuda-and-collect-garbage` 等——**RL 训练的 refit 阶段已有现成的区间划分可直接讲**

---

## 6. 与 Megatron 系列的关系：模块 C 的外迁

`deep-dive-megatron/RESEARCH_PLAN.md` 的模块 C 是三篇「实测与调优」：

| 原计划篇目 | 原内容 | 去向 |
|:---|:---|:---|
| 13 · 度量方法学 | MFU 口径、nsys/ncu/torch profiler/mcore timers 分工、nccl-tests busbw/algbw、实验卫生 | **并入 profiling-00**（口径部分） |
| 14 · 调优 Playbook | 七步法、组合决策树、端到端记账案例 | **并入 profiling-01 §10** |
| 15 · Scaling 与瓶颈诊断 | 强/弱扩展测法、α-β + 屋顶线、八种常见病排除树 | **并入 profiling-01 §5–6、§9** |

**为什么外迁**：模块 C 的内容天然不是「Megatron 源码走读」，而 Megatron 系列已定
「正文不写围绕硬件展开的内容、没有实验就没有」。留在原地会两头不讨好：既违反该系列的定位，
又让它长期悬空。

**外迁后**：

- `deep-dive-megatron/` 的模块 C 改为一段交叉引用，正文范围收为模块 0 + A + B + D；
- 本系列承接全部实测量化的部分，落点为 `categories: 性能分析`，与 `nv-gpu-profiling-toolkit` 同族；
- 两边都保留指针，读者从任一侧都能走通。

（Megatron 侧的计划文件已同步改动，见其模块 C 章节的「已外迁」标注。）

---

## 7. 写作硬性约定（沿用项目既定规则）

- **中文引号必须写全角** `“ ”`；写完后跑 `python3 tools/fix_quotes.py`
- **行内公式一律 `$$...$$`**，禁止单 `$`（kramdown 只保护双 `$$`；单 `$` 里的 `_` 会生成 `<em>`、
  裸 `|` 会让整段被判成表格）
- 公式内绝对值/范数写 `\lvert x\rvert`，禁止裸 `|`
- **所有代码路径、类名、行号必须对着 commit `60e039626` 核一遍**，不允许凭记忆写
- 每条重要论断挂出处（官方 docs / 论文 / 源码行号 / 官方仓库）；不可核验的数据显式标注「待验证」
- 示例数值必须实跑（miniconda python）；**没有实验就没有，不要凑**
- **正文不用第一人称**叙述生成过程（「初版实现没有区分 X」而非「我一开始没区分 X」）
- Mermaid：只用 `<br>` 换行、禁止字面 `\n`；节点文本含 `()[]{}` 需加引号
- 长文档写完后跑 `python3 tools/scan_garbled.py FILE` 扫乱码词与单 `$` 公式
- 离线校验用 kramdown 渲染，判据只看两件事：公式是否被正确转义、有没有误生成 `<table>`
  （本地无 GFM parser，代码块渲染结果不可信）
- **离线校验统一走 `python3 tools/check_post.py FILE`**（2026-09-29 新增）：它先剥离围栏代码块再渲染，
  避免代码块里的 `*` `_` `|` 漏进正文伪造出 `<em>` / `<table>`——这是本地 kramdown 2.5.2 的已知行为
  （本地把 ``` 围栏降级成 `<p><code>`，多行块会散成段落；只有 4 空格缩进块才变 `<pre>`）。
  输出三项：公式转换数、表格数是否一致、`<em>` 出现处清单。

---

## 8. 进度

- [x] 全站盘点，确定与已有内容的边界（§0.1）
- [x] 核对 Megatron 源码基准，定位 timers / FLOPs / profiler / OTel 的具体行号
- [x] 调研 2026 年工具链现状，形成 §5 事实底座
- [x] 确定模块 C 外迁方案（§6）
- [x] `tools/profiling_bench/timer_semantics.py`、`mfu_accounting.py`、`goodput_calc.py`（2026-09-29）
- [x] `tools/profiling_bench/trace_agg.py`（2026-09-29）：真值反算六项主量 + idle 三分，最大相对误差 0.00%
- [x] `tools/profiling_bench/overlap_bound.py`（2026-09-29）：`D* = n·α·BW`、`max(C,M)` 上界、algbw↔busbw
- [x] **profiling-00 度量口径**（`_posts/2026-09-29-profiling-00-metrics.md`，821 行）
- [x] **profiling-01 训练栈 profiling**（`_posts/2026-09-29-profiling-01-training.md`，645 行）
- [x] `tools/profiling_bench/serving_model.py`（2026-09-30）：roofline 步长队列模型，CLI 覆盖
      `--selftest / --sweep / --rate-sweep / --pattern-compare / --loop-compare / --preempt-ab / --prefix-cache`
- [x] `tools/profiling_bench/serving_schema.py`（2026-09-30）：字段映射（A/B/C 三组能力边界）+ ITL/TPOT 分母演示
- [x] **profiling-02 推理服务 profiling**（`_posts/2026-09-29-profiling-02-inference.md`，572 行）

### 8.2 profiling-02 写作中确认 / 修正的结论（回灌建议）

1. **`P99/P50` 只在 open loop 下有信息量。** closed loop 稳态下比值恒为 1.00
   （在途数被客户端卡住 → 等待队列恒空 → 单批内每条请求走同一串步长）。
   全样本口径的高比值来自**爬坡段**：并发 128 时全样本 p99 = 10467.2 ms、
   稳态 195.3 ms，差 53.6 倍。→ **任何 p99 报数前先声明稳态样本怎么切。**
2. **绑住 goodput 的通常是 TPOT 而不是 TTFT。** 并发 16→32，TPOT 中位数 21.0→31.4 ms
   跨过 30 ms 的 SLO，达标率 91.2%→8.8%。TPOT 的转折点就是 decode 膝盖 `F_d / a_d`。
3. **输出吞吐的饱和值不是模型能力，是输入输出 token 比的函数。**
   同一条 KV 池下 4K prompt 的输出吞吐上限（352.7 tok/s）只有 1K prompt（1125.3 tok/s）的 1/3。
   → 报 tok/s 必须同时报 prompt/completion 长度分布。
4. **KV 池满之后 recompute 型抢占是纯负收益。** 对照实验（基线档位两列逐位相等，已验证）：
   并发 256 时 `preempt=off` 8.79 req/s / TTFT 2228.4 ms vs `preempt=on` 5.13 req/s（−41.6%）/ 25231.2 ms。
   → 边界：模型只实现 recompute，vLLM 的 **swap 型**代价低得多，不可外推。
5. **ITL 与 TPOT 相差一个分母** `(n_out−1)/n_streamed_gaps`。
   vLLM 文档自带例子：同一时序 mean ITL = 40 ms、TPOT = 20 ms。
   → 开投机解码后 mean ITL 会被口径放大，不能读成“变慢”。
6. **工具面（2026-09）**：vLLM 自带 bench 为单进程 asyncio；**AIPerf**（`ai-dynamo/aiperf`）
   为 GenAI-Perf 指定继承者，**多进程 + ZMQ**；**GenAI-Perf 已在 `perf_analyzer` 仓库挂弃用声明**；
   **LLMPerf 于 2025-12-17 归档**；GuideLLM 在 vLLM 项目内维护。
   → 换工具不能比数：两者不产出逐位相同结果。
7. **`--goodput` 需要显式给 SLO**（`KEY:VALUE`，值毫秒，可选 ttft/tpot/e2el），
   文档直接引用 DistServe（arXiv:2401.09670）。
   → 分位字段是**条件出现**的：`--metric-percentiles` 默认只有 `99`，所以默认结果里没有 `p90_*`。
8. **前缀缓存的坑有了官方证据**：`vllm bench sweep serve` 在每轮之间显式调用
   `/reset_prefix_cache` 与 `/reset_multimodal_cache`，文档原话是 “to get a clean slate for the next run”。
   → 「换种子 / 重启服务 / 用 sweep」三选一；`--num-warmups` **不清缓存**（两个独立效应）。
   → 计划 §7 里那组“265 ms → 61 ms”**未能回溯到一手来源，已在正文标注待验证**。

### 8.3 脚本侧修掉的两个工具 bug（均已在 `--selftest` 里固化为回归项）

1. **decode 批次没拍快照**：本步刚完成 prefill 的请求又多吃了一步 decode，
   单请求 E2EL 恰好差一个 TPOT。只有对账（而非“跑起来”）能发现。
2. **`check_post.py` 的源码侧块级公式计数漏掉引用块前缀**：`>   $$...$$`
   被判成行内，于是报出假的“行内公式没被转换”。已修（先 `re.sub(r"^(?:>\s*)+", "", s)`）。
   修后 profiling-02 源码 23 行内 / 6 块级 == 渲染 23 / 6，逐位相等。

### 8.1 profiling-01 写作中确认 / 修正的结论（这些要回灌到 §5 事实底座）

1. **序列并行（SP）不降低 TP 通信量**，恒等式 AR = RS ∘ AG 决定总量守恒。
   已用 `overlap_bound.py --bound` 验证比值恒为 1.00×。
   → 已据此**校准** `megatron_bench/comm.py::tensor_parallel_events()` 的 SP 分支
   （此前标注“待校准”，现已改写并附 arXiv:2205.05198 与 arXiv:2311.02382 引用）。
   → **推论：不要再把“开 SP”当成降通信量的手段。**
2. **HTA 版本更正**：实际最新为 **v0.6.0**（PR #341，2026-04-22），此前记的 v0.5.0 已过期；
   Python ≥ 3.10。idle 三分官方措辞是 **host wait / kernel wait / unknown**，
   计划早期写的 `kernel-gap` 已统一改成 `kernel-wait`（脚本与正文同口径）。
   `critical_path_analysis.py` 在代码里但**未列入 README feature 列表**，引用时要注明。
3. **OTel trace 按 `save_interval` 重新生根**（`training.py:625`），分块用独立计数器 `_otel_trace_interval_step`；
   `save_interval <= 0` 时退化为单条长 trace。→ 决定 OTel 数据的聚合粒度，写 profiling-02 时若引用 OTel 要注意同一件事。
4. **`is_goodput_span=True` 共 23 处**（`training.py` 20 + `checkpointing.py` 3）——这是 goodput 口径的源码级定义清单，
   可用于 00 篇的口径论证。
5. **`training.py:4535-4541` 的注释**是“代码里明确写出机制与验证手段”的范例：
   async checkpoint 的 `save.finalize` 在关键路径上，表现为 checkpoint 边界附近的 per-iteration 空隙。
   → idle 三分的一个可被 span 直接确认的病例。

---

## 参考清单（计划阶段已核验的入口）

**官方文档**

- [CUDA Toolkit 13.4 发布说明（NVIDIA 开发者博客）](https://developer.nvidia.com/blog/cuda-toolkit-13-4-adds-windows-on-arm-support-and-greater-control-over-shared-gpus/)
- [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) ／ [产品页](https://developer.nvidia.com/nsight-systems)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) ／ [产品页](https://developer.nvidia.com/nsight-compute)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) ／ [torch.cuda.Event](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html)
- [Holistic Trace Analysis 文档](https://hta.readthedocs.io/) ／ [GitHub](https://github.com/facebookresearch/HolisticTraceAnalysis)
- [NVIDIA AIPerf 大规模推理基准测试](https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf) ／ 仓库 `ai-dynamo/aiperf`
- [nccl-tests](https://github.com/NVIDIA/nccl-tests) ／ [NCCL](https://developer.nvidia.com/nccl)
- [NVML](https://developer.nvidia.com/management-library-nvml) ／ [DCGM](https://developer.nvidia.com/dcgm) ／ [CUPTI](https://developer.nvidia.com/cupti)

**论文与来源**

- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
  —— MFU 指标的出处
- Dubey et al., *The Llama 3 Herd of Models*, [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
  —— 已发表 MFU 与并行配置对照
- Kwon et al., *PagedAttention / vLLM*, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  —— 服务侧 KV cache 形态的背景
- Megatron-LM 仓库 `NVIDIA/Megatron-LM`，基准 commit `60e039626`（v0.20.0，2026-08-21）

**站内互引**

- [NV 卡性能剖析工具全景](/2026/08/26/nv-gpu-profiling-toolkit/) —— 本系列的前置总纲（工具是什么）
- [算子开发与优化（04）性能分析方法论](/2026/09/03/op-04-performance-analysis/) —— 单 kernel 侧的下钻方法论
- [给 AI 编译器工程师的芯片课（十）](/2026/08/23/chip-for-ai-compiler-10-codesign-perf-modeling/) —— cost model 与参数可信度分级
- [分布式训练全景（00）](/2026/08/31/dist-train-00-overview/) —— 并行策略与通信代价的解析推导
