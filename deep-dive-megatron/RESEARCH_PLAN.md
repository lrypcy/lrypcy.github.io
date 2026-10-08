# Megatron-LM 深度剖析系列 · 研究计划与提纲

> 本系列做的是「**源码层 + 实测层**」，不是并行原理的第二遍。
> 原理层已在 `dist-train-00..08` 系列讲透，本系列一律引用不重推。

---

## 0. 基准与定位

### 读代码基准（路径与行号均已核对）

- 仓库：`NVIDIA/Megatron-LM`，本地路径 `~/Desktop/Projects/github/Megatron-LM`
- 版本：**`megatron-core` 0.20.0**，commit `60e039626`（2026-08-21，main 分支）
- 每篇开头必须标注 `commit`，因为 mcore 的代码迭代极快（一个月内动过 `parallel_state.py`、`finalize_model_grads.py`、`tensor_parallel/`）

### 本系列要填补的三处空白

`dist-train` 系列的结尾是一张「通信账本」，但没有兑现这三件事：

| 空白 | 表现 | 本系列如何补 |
|---|---|---|
| **没进源码** | 只给结论 + flag 名，`ColumnParallelLinear` 长什么样、`ParallelState` 怎么切 grid 都没看 | 模块 A 逐个拆解核心类/函数，贴真实代码片段与行号 |
| **没做实测** | 零 profiler 数据，零 MFU 曲线，数值全靠解析估算或引二手贴 | 本系列提供可复算的解析模型；**实测量化部分移交 `deep-dive-profiling/`**（本机无 GPU，见模块 C 说明） |
| **没讲 2026 的新东西** | 还在 1D TP + Ring Attention 时代 | GTP、MLA/DSA、Megatron-FSDP v2、Dynamic CP、Full CUDA Graph、原生 RL 栈全部缺席，本系列纳入 |

### 交付形态（按既定偏好）

- 本目录（`deep-dive-megatron/`）放**研究素材与中间产物**（提纲、benchmark harness、实测数据、源码笔记）
- **对外发布默认每个模块收成 1 篇长文进 `_posts/`**（`megatron-01-*.md` …），不拆 15 篇碎片；仅在单篇超过 600 行时按内部主题拆分
- **交付形态（2026-09-28 定稿，2026-09-29 修订）**：**每个模块直接产出 1 篇深度长文进 `_posts/`**，本目录下不放正文。
  正文命名（**编号已顺延，以本段末尾的第三次顺延说明为准**）：
  `_posts/2026-09-28-megatron-00-foundation.md`、`megatron-01-parallelism-internals`、
  `megatron-02-expert-parallel`、`megatron-03-optimization-toolbox`、`megatron-04-rl-refit-bridge`。
  **原定的 `megatron-03-measurement-and-tuning` 已取消**，其实测与调优内容整体移交
  `deep-dive-profiling/`（见「模块 C」章的说明）。2026-09-29 那次修订曾把 `rl-refit-bridge`
  从 04 顺延为 03，该编号已被下方的第三次顺延再次覆盖。
  **编号第三次顺延（2026-10-08）**：EP 从模块 B 提前独立成篇，占 `megatron-02`，
  `optimization-toolbox` 顺延为 `megatron-03`、`rl-refit-bridge` 顺延为 `megatron-04`。
  理由是 EP 单主题的源码体量已超出「一个模块一篇」的承载上限：`token_dispatcher.py` 2085 行 +
  `moe_utils.py` 1762 行 + `experts.py` 1567 行 + 62 个 `moe_*` 配置字段 + 三套 dispatcher 四套后端，
  且负载均衡、CUDA graph 分段捕获、通信重叠三条线索各自都能独立成篇。
  本目录只保留：本计划、`labs/`（CPU 实验脚本）。解析工具在 `tools/megatron_bench/`

---

## 1. 实验方法论（无 GPU 约束）

**本机没有任何 GPU**，所以本系列不存在「真机 profiling 一遍」这条路。为此先订三条纪律，所有文章一律照办。

### 1.1 数字只有三个合法来源

| 等级 | 来源 | 要求 | 例子 |
|---|---|---|---|
| **A 解析/推导** | `tools/megatron_bench/` 下的 CPU 脚本实跑产出 | 脚本必须随文章公开，参数可改，结果可被他人复算 | 显存账本、α-β 通信代价、1F1B 气泡率 |
| **B 官方来源** | NVIDIA 技术博客 / Megatron 论文 / repo 里的 unit test 与 benchmark | 必须挂链接或文件路径，并注明卡型与规模 | 某配置下的实测 MFU、吞吐对照表 |
| **C CPU 微型复现** | numpy / CPU torch 手撸的等价实现，本机实跑 | 验证**数值等价性**或**算法正确性**，不假装是端到端性能 | online softmax vs 全量 softmax；E4M3 量化误差；Masked-Orthogonal rank 分组 |

**禁止 D 级**：凭印象写的数字（“大约快 30%”“通信量差不多翻倍”）一律不许出现。推不出来、又查不到出处的，就明写「**待验证**」。

### 1.2 三条替代原则

1. **能严格算的，就不要估**：通信量、显存占用、气泡率、理论 kernel launch 数都是可解析量，用脚本算出精确值而不是给个经验数。
2. **能验证等价性的，就不要只看代码**：凡是换了数据流或算法路径的地方（SP 用 RS+AG 替代 AR、selective recompute 只重算 core_attn、融合 kernel），一律在 CPU 上跑一次数值对照，把「它确实等价」落实成一个数字。
3. **性能结论只能来自 B 级**：吞吐/MFU 这类必须真机才能定的数字，老实引用官方数据并写清卡型与规模，不假装是自己测的。

### 1.3 每篇的实验适配（统一写在各篇「实验设计」末尾）

- 模块 A（02–07）：以 **C 级为主** —— 进程组拓扑、f/g 算子等价性、Masked-Orthogonal 分组、online softmax、SP 的 RS+AG 数学等价，全部手撸复现；CP/PP 维度再叠加 A 级代价模型。
- 模块 B（08–12）：以 **A 级 + C 级**为主 —— 重叠收益上限用时间线模型算，显存用六项账本扫参数画帕累托前沿，FP8/MX 用手撸量化器跑误差，MoE 用 numpy 模拟均衡策略；吞吐类结论引用官方 blog。
- 模块 C（13–15）：**已外迁**。原定的 A 级为主 + B 级校准路线由 `deep-dive-profiling/` 承接，
  该方法论（A/B/C 分级）在两处保持一致，避免出现两套标准。

---

## 2. 系列总览

```
模块 0 · 地基          00 导览与实验台    01 控制流全图
模块 A · 并行源码层     02 进程组  03 TP/SP  04 GTP  05 DP/优化器  06 PP  07 CP/长上下文
模块 B · 优化工具箱     08 通信重叠  09 显存/重计算  10 低精度  11 融合/CUDA Graph  12 MoE
模块 C · 实测与调优     ← 已整体外迁至 deep-dive-profiling（见该章）
模块 D · 延伸(选做)     E1 RL/Refit  E2 Checkpoint/容错  E3 Bridge/互操作
```

本系列交付 **13 篇**（模块 0 + A + B），延伸 **3 篇**按需。建议执行顺序：0 → A → B → D。
模块 C 原定的 3 篇移交给 `deep-dive-profiling/`，理由与映射见该章。

---

## 模块 0 · 地基

### 00 · 系列导览与实验台

**核心设问**：这一系列要解决什么？实验怎么跑才算数？

**骨架**
- 系列地图与阅读路径（速查版 / 深入版 / 调优从业者版）
- 三个组件的分工：Megatron-LM（参考实现） vs Megatron Core（库） vs Megatron Bridge（HF 双向转换）
- 实验台搭建：安装路径（PyPI `megatron-core` vs 源码 editable）、NGC 容器、依赖矩阵（TE / Apex / NCCL / nvshmem）
- **可复现性契约**：commit 锁定、`--mock-data` 与真实数据的差别、确定性 baseline（`--seed` 一致性、`NVTE_*` 环境变量）

**实验设计**
- **A 级**：搭 `tools/megatron_bench/`，四个纯 CPU 解析器：`flops.py`（FLOPs 记账与 MFU 口径）、`mem.py`（显存六项账本）、`comm.py`（α-β 通信代价）、`roofline.py`（吞吐上限）。CLI 可调参，输出表格 + JSON
- **自校验**：同一配置在三种口径下交叉验证（逐算子累加 vs 6ND 近似 vs 官方公布的实测点），把偏差写进表格而不是藏起来

---

### 01 · 一次迭代的控制流全图

**核心设问**：从 `pretrain_gpt.py` 回车到 loss.backward，中间到底发生了什么？

**代码入口**
- `pretrain_gpt.py` → `megatron/training/training.py::pretrain()` → `model_provider` / `get_forward_backward_func` → `forward_step` / `backward_step`
- `megatron/core/pipeline_parallel/schedules.py::forward_backward_no_pipelining` / `forward_backward_pipelining_with_interleaving`
- `GlobalMemoryBuffer`（`megatron/core/utils.py`）：跨 iteration 复用缓冲，是显存记账里容易漏的一项

**骨架**
- 主循环时序图：模型构建 → 初始化（opt/ckpt/数据）→ microbatch 切分 → 每个 rank 在干啥
- `num_microbatches_calculator.py`：microbatch 数怎么算，rampup batch size
- 训练日志里每一个字段分别是哪里打出来的

**实验设计**
- **C 级**：手撸一个 40 行 mini training loop，复刻同样的控制流结构（microbatch 切分 → forward → backward → grad 归约），CPU 上用 `cProfile` 打出调用热点与层级耗时占比，对着上面推导的控制流逐项对账
- **B 级**：引用官方发布过的 training step timeline 截图做对照

---

## 模块 A · 并行体系源码层

### 02 · 进程组：world_size 如何被切成一张网格

**核心设问**：给定 1024 卡和一组并行 size，谁负责把它切成 TP×CP×PP×EP×DP？两个 rank 之间的 NCCL comm 是建在哪一层？

**代码入口**（已核对行号）
- `megatron/core/parallel_state.py::initialize_model_parallel()` —— **L601 起约 1000 行**，是整个体系的地基
- `generate_masked_orthogonal_rank_groups()` (L269) —— 通信邻居分散，避免同一交换机过载
- `create_hierarchical_groups()` (L378) / `create_hybrid_dp_cp_groups()` (L440) —— 层级 CP、DP-CP 混合组
- `megatron/core/process_groups_config.py` —— 新一代 declarative 进程组配置
- `megatron/core/hyper_comm_grid.py`（447 行）—— 多维通信网格抽象

**骨架**
- rank → 坐标的双向换算；「我在哪个组」的全部 getter（`get_tensor_model_parallel_group` 等 50+ 个）
- `_process_groups_from_order`：组创建的顺序为什么重要
- embedding / position-embedding 组的特殊处理（L560）
- TP×PP 与 GPU 亲和性的关系：为什么 `generate_masked_orthogonal_rank_groups` 能提速

**实验设计**
- **C 级**：把 `generate_masked_orthogonal_rank_groups` 用手撸实现重写一遍，给定 `(world_size, tp, pp, dp)` 打印 rank 矩阵，并用 numpy 验证「正交性」——任意两组的交集大小、跨节点/节点内通信占比是否真被摊平
- **A 级**：`comm.py` 计算 k-ring step 在不同拓扑下的跨节点字节数，证明这个摊平不是玄学

---

### 03 · 张量并行与序列并行的真实实现

**核心设问**：论文里那对虚拟算子 `f` / `g`，在代码里是 autograd.Function 的哪一段？SP 是怎么把 All-Reduce 换出 RS+AG 的？

**代码入口**
- `megatron/core/tensor_parallel/layers.py`（1583 行）
  - `ColumnParallelLinear` L920 / `RowParallelLinear` L1308
  - **`LinearWithGradAccumulationAndAsyncCommunication` L573** —— 通信与 GEMM 重叠的真身
  - `_linear_forward` L444 / `_wgrad_gemm` L554 / `LinearWithFrozenWeight` L403
  - `VocabParallelEmbedding` L230
- `megatron/core/tensor_parallel/cross_entropy.py` —— vocab-parallel CE，显存杀手解法
- `megatron/core/tensor_parallel/mappings.py` —— `_reduce_scatter_to_sequence_parallel_region` 等

**骨架**
- f 算子的 forward=f 矩阵 / backward=g；g 的 all-reduce 怎么与 dgrad GEMM 重叠
- `sequence_parallel=True` 时 f 切换为 reduce-scatter、g 切换为 all-gather 的代码分支
- 为什么 Grad 累加要 fuse（`--gradient-accumulation-fusion`），以及它和 `wgrad_deferral` 的关系

**实验设计**
- **C 级**：numpy 严格验证三件事的数值等价性：① f 的列切 + All-Reduce ≡ 未切版本；② SP 下 f 换 RS、g 换 AG 与原来 AR 的结果一致；③ gradient-accumulation-fusion 累加路径 ≠ 朴素累加（精度差异量化）
- **A 级**：TP=1/2/4/8 的每层通信字节数精确解，配合 NVLink 与 IB 的带宽比值定位性能拐点

---

### 04 · GTP：2026 年的新一代张量并行

**核心设问**：传统的 activation-centric TP 有什么上限？GTP 改了什么？它为什么必须依赖 TransformerEngine？

**代码入口**
- `megatron/core/tensor_parallel/gtp_api.py` —— 门面层，`HAVE_GTP` 门控
- `megatron/core/tensor_parallel/generalized_tensor_parallelism.py` —— 实现主体
- `megatron/core/tensor_parallel/gtp_cuda_graphs.py` —— 与 CUDA Graph 协作
- 关键符号：`GTPChain`, `gtp_remat_shard_dim0`, `classify_gtp_remat_chains`, `attach_gtp_to_presharded_module`, `wait_async_comms`, `dequantize_gtp_native_fp8`
- 相关 flag：`--gtp-remat-reduce-scatter-with-fp32-accumulation`

**骨架**
- weight-centric vs activation-centric：前者切权重本身，后者只切激活
- remat（重 materialization）策略与 `classify_gtp_remat_chains` 的分类逻辑
- AG/RS 专用 CUDA stream 的管理（`get_ag_stream` / `get_rs_stream`）
- native FP8 权重路径
- parallel_state 里的 `get_gtp_weight_remat_group`（L1713）—— 说明它建了专属进程组

**实验设计**
- **B 级**：源码走读 + 官方 GTP 材料（逐处核对，不编数据）
- **A 级**：把传统 TP 与 GTP 的每层通信字节数做成可对比模型，重点回答「TP 超过 NVLink 域之后，两者差多少」
- **C 级**：`gtp_remat_shard_dim0` 的切分语义手工跑一遍 shape 推演（给定 hidden 与 tp_size，逐步给出每个 rank 上的张量形状）

---

### 05 · 数据并行与优化器：三套实现的分野

**核心设问**：DDP / Megatron-FSDP / FSDP2 在代码的哪一层分叉？`param_and_grad_buffer` 里的 bucket 到底怎么切？

**代码入口**
- `megatron/core/distributed/param_and_grad_buffer.py`（**1794 行**，本系列最硬的一块）
- `megatron/core/distributed/finalize_model_grads.py`（713 行）
- `megatron/core/distributed/distributed_data_parallel.py` / `distributed_data_parallel_config.py`
- `megatron/core/distributed/fsdp/` —— Megatron-FSDP（MFSDP v2）
- `megatron/core/distributed/torch_fully_sharded_data_parallel.py` —— FSDP2 适配
- `megatron/core/optimizer/`：
  - `distrib_optimizer.py`（DistributedOptimizer 的分片布局）
  - `fully_sharded_optimizer.py` / `layer_wise_optimizer.py` / `cpu_offloading/`
  - `param_layout.py`、`optimizer_cuda_graph.py`、`clip_grads.py`、`grad_scaler.py`
  - `muon.py` + `emerging_optimizers.py`（2026/04 新增）+ `qk_clip.py`

**骨架**
- `param_and_grad_buffer` 的核心数据结构：param bucket / grad bucket 如何 padding 与对齐（`--ddp-pad-buckets-for-high-nccl-busbw`）
- 重叠状态机：`overlap_grad_reduce` + `overlap_param_gather` + `overlap_param_gather_with_optimizer_step` 三种组合干了什么、实测哪种最高
- `DistributedOptimizer` 的分片布局 vs ZeRO-1：为什么 mcore 的分片布局和 ZeRO-1 不完全是一回事（`param_layout.py` 是干什么的）
- 三条 DP 路径的取舍表（显存 / 通信量 / 支持 / 坑）
- `--ddp-reduce-scatter-with-fp32-accumulation` / `reduce_scatter_with_fp32_accumulation.py` —— 精度相关的 RS
- Muon / emerging optimizers 接入方式（承接 2026/04 官方 blog）

**实验设计**
- **A 级**：`mem.py` 算出 DDP / Megatron-FSDP / FSDP2 三条路径的显存六项账本，给一张完整对照表（含 bucket 粒度带来的 tail 浪费）
- **C 级**：numpy 验证 reduce-scatter + 分片优化器更新与非分片 AdamW 的**数学等价**（这正是这些 overlap 开关能安全打开的前提）
- **A 级**：overlap 组合的时间线模型，给出「什么规模该开哪个」的决策表

---

### 06 · 流水并行：调度器的细节不在论文里

**核心设问**： 论文上的 1F1B 只画了方块，`schedules.py` 里要为它填多少工程细节？

**代码入口**
- `megatron/core/pipeline_parallel/schedules.py`
- `p2p_communication.py`、`combined_1f1b.py`、`hybrid_cp_schedule.py`
- `bridge_communicator.py` / `multimodule_communicator.py`（VLM 等多模块场景）
- `fine_grained_activation_offload.py`
- `megatron/core/transformer/pipeline_parallel_layer_layout.py` —— 非均匀切层

**骨架**
- forward/backward/Wgrad/Dgrad 四种 tensor 形状在 schedule 里怎么保证 sender == receiver
- warmup / steady / cooldown 三阶段的实际代码结构 + 每个 steady step 的 checklist 顺序
- virtual pipeline (interleaving) 到底省掉了多少气泡
- p2p 与 `--overlap-p2p-communication` 的实现位置
- 非均匀切层：`--pipeline-model-parallel-layout` 与 `--decoder-first-pipeline-num-layers` 解决第一/末 stage 显存不均
- 激活 offload：`--fine-grained-activation-offload`

**实验设计**
- **A 级**：气泡率的精确解析解（1F1B / interleaved 各自的公式），扫 PP 与 virtual stages 出表
- **C 级**：用 numpy 画一张甘特图模拟 1F1B 的执行时序（气泡在图上肉眼可见），再和解析解对账
- **A 级**：`mem.py` 解析第一/末 stage 的显存溢出（embedding + loss + MTP），对比均匀切 vs `--decoder-first-pipeline-num-layers`

---

### 07 · 上下文并行与长上下文

**核心设问**：长序列这一刀怎么切？CP 的三种实现对 attention kernel 分别要求什么？

**代码入口**
- `megatron/core/transformer/dot_product_attention.py` —— CP dispatch 主体
- `megatron/core/packed_seq_params.py` —— 打包序列（THD 格式）的元数据
- `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` —— hybrid CP 与 PP 联合调度
- `megatron/core/models/multimodal/context_parallel.py` —— 多模态 CP
- `megatron/core/ssm/mamba_context_parallel.py` / `gdp_context_parallel.py` —— SSM/Hybrid 的 CP
- Attention 变体：`multi_latent_attention.py`（**1502 行**，MLA）、`experimental_attention_variant/{absorbed_mla,dsa}.py`（DSA = DeepSeek Sparse Attention）
- 相关 flag：`--cp-comm-type`、`--sequence-packing-scheduler`

**骨架**
- 三条路线对照：Ring/KV-p2p vs All-Gather(THD) vs Hybrid，各自的通信量与 kernel 要求
- `PackedSeqParams` 里 cu_seqlens 怎么算，为什么 THD 能省 padding
- Dynamic CP（2026/01，可变长 + 自适应 CP size，官方称 1.48x）—— 实现机制
- MLA / absorbed MLA / DSA 在实现层的关键差异
- SSM/Hybrid 模型的 CP 为什么不能直接用 ring

**实验设计**
- **C 级**：numpy 验证 online softmax（分块递推）与全量 softmax 的**数值等价**，量化累积误差随块数增长的曲线——这是 ring attention 正确性的基础
- **A 级**：ring / THD(all-gather) / hybrid 三种路线的 KV 通信字节数解析模型，扫 CP size 画边际收益
- **A 级**：用采样出来的真实长度分布算 padding 浪费率，给出 THD packed 的收益上界

---

## 模块 B · 性能优化工具箱

> 这一模块的组织原则：**每一篇回答「解决什么问题 / 怎么实现的 / 什么时候开 / 有什么坑」四个问题**，并给出开关清单表。

### 08 · 通信与计算重叠全景

本篇是全系列的「通信账本兑现篇」。

**代码入口**
- `megatron/core/distributed/finalize_model_grads.py`（713 行）—— 处理 8 种 overlap 开关组合的分派
- `model_parallel_config.py` L265-410 的 `tp_comm_overlap_*` 全家桶（12 个开关：`bulk_wgrad` / `bulk_dgrad` / `ag` / `rs` / `rs_dgrad` / `ag_dgrad` / `rs_fprop` / `disable_qkv` / `disable_fc1` …）
- `--use-nccl-ub`（UserBuffer）、`--suggested-communication-unit-size`
- `reduce_scatter_with_fp32_accumulation.py`
- `delay_wgrad_compute`、`overlap_moe_expert_parallel_comm`、`ep_overlap_early_attn_memory_release`

**骨架**
- 一张「重叠矩阵」：哪种通信可以被哪种计算掩盖（RS↔wgrad GEMM、AG↔下一个 GEMM、all-reduce↔bwd、P2P↔邻近 microbatch）
- `finalize_model_grads` 为什么写成一张查找表：8 个条件的组合
- TP-comm-overlap 的 SplitGEMM 思想：把 GEMM 拆 chunks 与 AG/RS 流水
- NCCL UserBuffer：绕开 NCCL 的通用路径换取更少 SM 占用与更低延迟
- MoE 的 EP overlap： dispatch backward 与 wgrad 如何叠；为什么必须配 memory release

**实验设计**
- **A 级**：时间线模型算每个开关的**理论收益上限** = 可被掩盖的通信时长 / 总迭代时长，产出「收益上界 / 触发条件 / 副作用」表（承认这是上界，实测会更低）
- **B 级**：用官方给出的实测增益校准这个模型，偏差写进文章
- 用一个固定 case（7B、TP4 PP2）从头走到尾

---

### 09 · 显存账本：重计算、卸载与分布保存

**代码入口**
- `megatron/core/recompute.py`（189 行，只有一个 `checkpointed_forward`）只是一个薄封装，真正的判断逻辑散在调用侧
- `megatron/core/transformer/transformer_config.py` L522-575 —— **recompute 配置全量字段**（已核对）
  - `recompute_granularity`：`full` / `selective`
  - `recompute_method`：`uniform` / `block`；`recompute_num_layers`
  - `recompute_modules` 全部选项（源码注明 default=`[core_attn]`）：`core_attn`(默认) / `moe_act` / `layernorm` / `mla_up_proj` / `mlp` / `moe` / `shared_experts` / `gdn_norm_out` / `gdp_in_proj` / `gdp_qkv` / `mhc`
  - 区分 normal checkpointing vs **output-discarding checkpointing**（后者注意 discrepancy）
  - `distribute_saved_activations` —— 把保存的激活摊到模型并行组上（配合 TP 用时要注意语义）
- `fine_grained_activation_offload.py` / `cpu_offloading/` / `--optimizer-cpu-offload` / `--overlap-cpu-optimizer-d2h-h2d`

**骨架**
- 显存账本六项：权重 / 梯度 / 优化器状态 / 激活 / 碎片 / 临时缓冲（别忘了 `GlobalMemoryBuffer` 与 NCCL allocator）
- selective recompute 的原理回溯（arXiv:2205.05198）：为什么只重算 attention 最划算
- `distribute_saved_activations` 与 TP/CP 的相互作用
- offload 三档：全部 CPU offload / fine-grained activation offload / optimizer offload

**实验设计**
- **A 级**：`mem.py` 扫全部 recompute 组合 × distribute × offload 三档，画「显存 vs 额外重算 FLOPs」的帕累托前沿（纵轴不能用吞吐——那是真机才能定的数，这里用**额外计算量占比**作为代价指标）
- 落到具体结论：在某型号 GPU / 某模型规模下，帕累托前沿上该选哪个点

---

### 10 · 低精度：FP8 全链路与 FP4 到哪里了

**代码入口**
- `megatron/core/fp8_utils.py`（1056 行）/ `fp4_utils.py`
- `megatron/core/extensions/transformer_engine.py` —— TE 集成层
- `megatron/core/quantization/`、`post_training/modelopt/`
- 相关 flag：`--fp8-param-gather`、`--fp4-param-gather`、`--grad-reduce-in-bf16`、`--reuse-grad-buf-for-mxfp8-param-ag`、`--main-params-dtype`、`--main-grads-dtype`、`--exp-avg-dtype`、`--use-precision-aware-optimizer`、`--num-distributed-optimizer-instances`

**骨架**
- FP8 的三种 scaling 策略在干什么（delayed / current / MX），各自的 amax 归约在哪里算（`get_amax_reduction_group`）
- param gather 的精度路径：为什么要保留 fp32 主权重、更新后再 cast；在 gather 阶段降精度能省多少带宽
- grad reduce 在 bf16 做 + fp32 累加的必要性（`reduce_scatter_with_fp32_accumulation` 在这里也出场）
- **NVFP4 / MXFP4 训练**：目前能到哪一步、哪些层不能降、精度损失的实际量级
- loss 曲线对照：FP8 vs BF16，看是否出现发散或系统性偏差

**实验设计**
- **C 级**（本篇重头戏，可接 `llm-quant` 主线）：手撸 E4M3 / E5M2 / MX-block-scaling 三种量化器，在**合成权重张量**（正态分布 + 注入 outlier）上跑 RMSE / SNR / 极值保留率，与 per-tensor 和 per-channel INT8 并排对比
- **A 级**：给每种 scaling 策略算 amax 归约的通信代价与 histogram 开销

---

### 11 · 算子融合与 CUDA Graph：把 step 图化到什么程度

**代码入口**
- `megatron/core/fusions/`（13 个融合模块）：`fused_bias_gelu` / `fused_bias_swiglu` / `fused_bias_geglu` / `fused_bias_dropout` / `fused_layer_norm` / `fused_softmax` / `fused_cross_entropy` / `fused_mla_yarn_rope_apply` / `fused_indices_converter` / `fused_pad_routing_map` / `fused_weighted_squared_relu`
- `megatron/core/full_cuda_graph.py`（267 行）+ `megatron/core/transformer/cuda_graphs.py`（**3311 行**）
- `megatron/core/optimizer/optimizer_cuda_graph.py`、`gtp_cuda_graphs.py`
- flag：`--cuda-graph-scope`、`--cuda-graph-modules`、`--optimizer-cuda-graph`

**骨架**
- **单次迭代图化（full iteration CUDA graph）的边界条件**：哪些部分能图化（shape 固定的 attention / MLP / optimizer step）、哪些不行（含 CPU-GPU 同步、动态 shape 的部分集合通信）
- CUDA Graph 与 NCCL 通信的冲突：为什么集合通信必须放到图外或独立的异步流上
- `fused_cross_entropy` 为什么在长词表下极其重要（显式 materialize logits 的显存开销）
- 写 CUDA graph 时的内存池管理（`graph_pool_handle` 复用）与多 stream 依赖的正确处理
- 常见坑：图化后 hang、确定性丢失、和 `--tp-comm-overlap` 的互相影响

**实验设计**
- **A 级**：kernel launch 次数的精确计数模型（逐层列出算子清单并累加），给出图化前后的差值——这是图上发生的事里唯一能纯算出来的
- **C 级**：numpy 验证融合算子的数值等价（bias+gelu、swiglu、fused CE），并算「融合前后 HBM 读写字节数」差值——融合的真实收益在带宽而非算力，这个能算准

---

### 12 · MoE 训练的全链路优化

**代码入口**
- `megatron/core/transformer/moe/`（20 个文件）
  - `router.py`（含 aux loss / z-loss / 多种均衡策略）、`experts.py`（GroupedMLP / TEGroupedMLP / SequentialMLP）
  - `token_dispatcher.py`（AG vs A2A 两条路线）、`fused_a2a.py`、`shared_experts.py`
  - `upcycling_utils.py`（dense → MoE 升级训练）、`router_replay.py` / `router_trace.py`（可复现路由调试）
  - `paged_stash.py`、`batch_invariant.py`（RL 场景确定性）
- `megatron/core/transformer/moe/ops/` —— 底层算子
- flag：`--moe-aux-loss-coeff`、`--moe-router-load-balancing-type`、`--num-experts`、`--moe-layer-freq`、`--moe-use-upcycling`

**骨架**
- Dispatcher 二选：All-Gather（适合同机少量卡） vs All-to-All（大规模）—— 各自通信量与显存开销怎么权衡
- Grouped GEMM vs 逐 expert GEMM：什么时候 grouped 反而不划算（tokens per expert 太少）
- 负载均衡六种策略的机理与适用场景；router collapse 的识别与缓解
- Shared expert 与 dense expert 的通信 overlap
- 附录：对照官方 MoE Q3–Q4 2025 Roadmap，看哪些承诺已落地、哪些还是坑

**实验设计**
- **A 级**：All-Gather 与 All-to-All 路由的精确通信字节数模型，随 EP/tokens 变化求拐点（这是个方程，有解析解）
- **C 级**：numpy 模拟四种负载均衡策略下的 expert 分配，输出负载直方图与“最大/均值比”作为均衡度指标
- Grouped GEMM 的 padding 浪费：随 tokens-per-expert 变化的收益/失效边界，用解析模型给出

---

## 模块 C · 实测与调优 —— **已整体外迁**（2026-09-29）

> **本模块不再在本系列交付。** 三篇（13 / 14 / 15）的内容已移到新计划
> **`deep-dive-profiling/RESEARCH_PLAN.md`**，落点为 `_posts/profiling-*.md`，`categories: 性能分析`。
>
> **外迁理由**：模块 C 天然不是「Megatron 源码走读」——它讲的是度量口径、调优 playbook、scaling 诊断，
> 与本系列已定的「正文不写围绕硬件展开的内容、没有实验就没有」直接冲突。留在原地会两头不讨好：
> 既违反本系列定位，又因该约定而长期无法动笔。
>
> **映射关系**：
> - 13 度量方法学（MFU 口径 / 工具分工 / nccl-tests / 实验卫生）→ `profiling-00`
> - 14 调优 Playbook（七步法 / 决策树 / 端到端记账）→ `profiling-01` §10
> - 15 Scaling 与瓶颈诊断（强弱扩展 / α-β + 屋顶线 / 八种常见病排除树）→ `profiling-01` §5–6、§9
>
> 需要 Megatron 源码细节时（timers、FLOPs 记账、OTel span groups）由 `profiling-*` 反向引用本系列。
>
> 以下原始提纲**保留**，作为迁移内容的存根，不再按本系列的写作约定执行。

### 13 · 度量方法学：先把尺子做准

**骨架**
- **MFU 到底怎么算才对**：Transformer FLOPs 的记账细项（含 attention 里 QK^T 与 AV 两次 GEMM、mask / softmax 是否计入），mcore 自己 log 的 `TFLOPs` 是哪个口径、与 HFU 口径的差异
- 工具链：nsys / ncu / torch profiler / Megatron 内置 timers（`megatron/core/timers.py`）分别解决什么
- nccl-tests 怎么读：busbw vs algbw 的换算
- 一致性校验：同配置两次运行的 loss 曲线应落在噪声范围内；换配置后如何确认「只是变快了，结果没变」
- 实验卫生：warmup / 取中位 / 去掉前 N step / 固定 randomness / GPU 锁频

**产出**
- 一份可复用的 `tools/megatron_bench/report.py`：多组 JSON → 并排表格 + 图

**实验设计**
- **交叉验证**：`flops.py` 用三种口径同时对同一个模型求解（逐算子精确累加 / 6ND 近似 / 官方实测点反推），把差异做成表格摊开写，而不是悄悄挑一个用
- 每种工具的适用边界写成一张表：什么东西**不能**用它测

---

### 14 · 调优 Playbook

**骨架**
- 七步法：先验粗估 → 单变量实验（一次只改一个开关）→ 先把显存压进预算，再谈吞吐 → 每次加一项就记一笔账 → 加完回头复测 baseline
- **组合决策树**：给你的 N卡 / M模型 / S序列，产出初始配置表
- 常见反直觉结论收集（如 TP 跨机会崩、DP 太大通信拖、GBS 太小 GPU 饿）
- **一个端到端完整案例**：从 baseline config 出发，逐笔记账到最终配置，每笔写出“改了什么 / 收益多少 / 代价多少”

**实验设计**
- **A 级**：给定（卡数、卡型号带宽、模型规模、序列长度），用 `tools/` 自动排出一套初始配置，并给出每一步的依据账
- **B 级**：找一个官方公开的真实训练配置做整体验证——模型推荐的和 NVIDIA 自己选的有多大偏差，偏差理由是什么

---

### 15 · Scaling 与瓶颈诊断

**骨架**
- 强/弱扩展的正确测法（哪些指标不该直接当结论）
- 一个简单的通信开销模型：α-β 模型 + 屋顶线，预估 “这个规模上理论应该跑到多少”
- **八种常见病 + 排除树**：
  | 症状 | 可能原因 | 验证方法 |
  |---|---|---|
  | 吞吐低且 GPU 利用率低 | 数据加载 / Host-side / GPU 饿（GBS 太小） | nsys 看 idle gap |
  | 显存炸在每个 step 末尾 | 优化器状态 / 激活保存超限 | memory snapshot |
  | DP 增大吞吐不涨 | 通信已成瓶颈 | nccl-tests + busbw |
  | 多机带宽只有机内 1/10 | 进程组绑卡不对 / IB 未用上 | `generate_masked_orthogonal` 检查 |
  | 中途 hang | CUDA graph + 集合通信死锁 / NCCL timeout | TORCH_DISTRIBUTED_DEBUG |
  | FP8 掉点 | amax 归约组不对 / 某些层敏感度 | 逐层对比 |
  | loss NaN | 精度 accumulate / clip | grad scaler 日志 |
  | 周期性减速 | ckpt save / dataloader 重读 | timeline 找周期峰 |

**实验设计**
- **A 级**：用 `comm.py` 的 α-β 模型去拟合官方公布的 weak scaling 曲线，把拟合残差写出来（拟合不上就说明模型漏了哪一项）
- 上面那张排除树里的每条「验证方法」都要注明：无 GPU 时该用什么替代手段

---

## 模块 D · 延伸（按需选做）

### E1 · RL 训练栈与 Refit（**强烈建议做，与 GRPO 主线交叉**）

v0.20 已有原生 RL：`megatron/rl/`（`agent/`、`inference/`、`server/`、`rollout_bank.py`、`sequence_packing_utils.py`）+ 近 40 个 `--grpo-*` / `--rl-*` flag。

- **核心看点**：训练与推理两套并行布局之间的权重搬运 —— `megatron/core/resharding/`：`refit.py` → `planner.py`（每 rank 重放同一确定性排期，只保留自己的 op）→ `execution.py` → `copy_services/`（NCCL / Gloo / NVSHMEM 三种传输），`transforms.py` 的 `MXFP8ReshardTransform`
- `batch_invariant.py`：训练-推理 logprob 一致性（承接 on-policy distillation 主线）
- 产出一份「训练/推理同构 vs 异构 resharding」的代价测算

### E2 · Checkpoint 与容错

`dist_checkpointing/`（fully-parallel save / load 策略）、`--ckpt-format fsdp_dtensor|torch_dist`、`rerun_state_machine.py`、`fault_injector.py`、`nvidia-resiliency-ext`、`--inprocess-*`（进程内重启）。

- 重点调优：把大模型 ckpt 从十几分钟压到几十秒，到底要拧哪几个旋钮

### E3 · Bridge、导出与模型家族落地

Megatron Bridge（HF↔mcore 双向 + recipes）、`export/`、modelopt 后训练（量化/distill）。
横向落地：DeepSeek-V4（含 DSA / MLA）、Qwen3-Next（Gated DeltaNet）、Falcon-H1 混合架构 + BitNet 三值量化、GPT-OSS（YaRN + attention sink）。

---

## 3. 写作硬性约定（沿用项目既定规则）

- **中文引号必须写全角** `“ ”`，正文写半角 `"` 会被 kramdown smart quotes 全转成右引号；写完后跑 `python3 tools/fix_quotes.py`
- **行内公式一律 `$$...$$`**，禁止单 `$`（kramdown 只保护双 `$$`；单 `$` 里的 `_` 会生成 `<em>`、裸 `|` 会让整段被判成表格）
- 公式内绝对值/范数写 `\lvert o\rvert`，禁止裸 `|`
- **所有贴出来的代码路径、类名、行号必须对着 commit `60e039626` 核一遍**，不允许凭记忆写
- 每条重要论断挂出处（arXiv / 官方 docs / 源码行号）；不可核验的数据显式标注「待验证」
- 示例数值必须实跑（用 miniconda python，managed python 3.13 无 numpy）
- Mermaid：只用 `<br>` 换行、节点文本含 `()[]{}` 需加引号
- 长文档写完后扫乱码词（伪 Unicode / 出现 1 次且长度 >6 的英文词）

## 4. 进度

- [x] 确定源码基准（v0.20.0 / `60e039626`），本地仓库已就位
- [x] 完成 16 + 3 篇系列提纲
- [x] 确认无 GPU，确定 A/B/C 三级证据体系与替代实验路线
- [x] 模块 0 地基 → `_posts/2026-09-28-megatron-00-foundation.md`（585 → 550 行，已按「去第一人称 + 去 GPU 内容」改写）
- [x] 模块 C 三篇外迁至 `deep-dive-profiling/`（2026-09-29）
- [x] 模块 B 的 MoE 主题提前独立成篇 → `_posts/2026-10-08-megatron-02-expert-parallel.md`
      （1897 行。进程组双表与 masked-orthogonal 分组数学、连续块不变量、路由与三种 aux loss、
      expert bias、group-limited routing、容量与 drop、permute 的 argsort 推导、
      All-to-All 七步与变长元数据、AllGather/Flex 四套后端、Grouped GEMM padding 分工、
      五槽 overlap 调度、partial CUDA graph 两套实现、paged_stash 重放、62 个 `moe_*` 字段表。
      纯源码 + 公式 + 图示，**未附 C 级实验**——本篇的数字全部是解析推导，
      §2 的网格算例已用 `generate_masked_orthogonal_rank_groups` 的语义重算复核：
      两套网格的 PP 组逐 rank 相同、EP 组等于 dense DP 组两条不变量均成立。
      若后续补 C 级实验，落点 `ipynbs` 的 `experiments/megatron/expert_parallel/`）
- [x] 模块 A·02 进程组 → `_posts/2026-10-08-megatron-01-process-groups-and-comm-overlap.md`
      （1397 行。两层初始化与 `order` 实参、59 个模块级全局的三类配对、`create_group`
      唯一收口与进程组台账、`RankGenerator` 的 order 归一化、
      masked-orthogonal 的混合基数分解与 1024 卡位序阶梯、
      「同序创建 + 成员过滤」两重隔离、组选择的三层机制（含 `ctx.tp_group`）、
      三个时间尺度与正交性纪律、两种 stream 流派与
      `CUDA_DEVICE_MAX_CONNECTIONS` 的两条相反硬断言、GlobalMemoryBuffer /
      wgrad deferral / grad accumulation fusion 三个计算组织手段、backward launch 序列。
      11 张自绘 Mermaid 图，纯源码 + 公式 + 图示，**未附 C 级实验**。
      文中算例全部实跑复核：masked-orthogonal 的两个 docstring 算例、1024 卡各 token 的
      组数与组大小、world=64 的正交性自检（tp∩dp 恒为 1）、59 个全局的分类计数。
      行号用 `git show 60e039626:` 逐条核过 85 个区间，修正 6 处偏差。
      注：本篇覆盖的是原计划「模块 A·02 进程组」+ 通信组织机制，
      **不含** overlap 开关组合决策表 / `finalize_model_grads` 八种分派 / NCCL UserBuffer，
      那部分留给优化工具箱那一块）
- [ ] 模块 A 其余篇目（TP+SP 真实实现 / GTP / DP+优化器 / PP / CP）
- [ ] 模块 B 优化工具箱（顺延为 megatron-03）
- [ ] 模块 D 延伸选做（E1 RL/Refit 与 GRPO 主线交叉，优先级最高）
