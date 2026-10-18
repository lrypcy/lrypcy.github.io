# TODO

> 研究待办清单。本文件不发布（已加进 `_config.yml` 的 `exclude`）。
> 状态：⬜ 待开始 ｜ 🟨 进行中 ｜ ✅ 完成

---

## 1. SFT 训练到什么程度可以做 RL ⬜

**核心问题**：什么时候该停 SFT、切 RL？判据是什么？切早了 / 切晚了各自损失什么？

**待查证的子问题**

- [ ] 收敛判据：验证 loss 平台期、下游任务 pass@k 饱和、格式/指令遵循率达标，哪个是真正可用的信号？
  - 关注「SFT 过拟合之后再 RL 会怎样」——是 RL 修得回来，还是容量被锁死？
- [ ] SFT 阶段数与 RL 的衔接：多轮 SFT（rejection sampling / STaR 式迭代）与 RL 的边界在哪？
  迭代式 SFT 是不是 RL 的低样本效率近似？
- [ ] 数据规模与质量：多少条 SFT 数据、什么质量分布下 RL 才有边际收益？
  （LIMA 式「少而精」 vs 大规模 SFT，两者后续 RL 的收益曲线是否不同）
- [ ] 基础模型能力门槛：数学/代码类任务上，SFT 到什么水平 RL 才「爬得动」？
  基座完全不会的任务，RL 是学不会还是只是学得慢？（ reward 稀疏 vs 能力天花板）
- [ ] 可验证奖励（RLVR）场景下 SFT 是否还需要「做满」？
- [ ] 工程判据：RL 前要不要冻结 SFT 的某些层 / 保留 KL 锚点的强度怎么定？

**参考线索**

- 站内已有：MDP → GRPO 系列 01–05、蒸馏与 RL 的统一谱系、`2026-09-19-rl-credit-assignment-settlement`
- 待查：DeepSeek-R1 / Kimi / Qwen 技术报告里关于 SFT→RL 阶段切换的描述，需锁版本引用
- 待查：是否有论文系统做过「SFT 程度 × RL 收益」的消融（找不到就标注「未见系统研究」）

**产出**：一篇 Deep-Dive，落到 `_posts/`，默认单篇长文直进（500+ 行 / 10 个一级章节）。
有可跑的判据就配 companion lab（放 ipynbs 仓库），没有就不凑。

---

## 2. SFT loss 与 PPO loss 怎么协同计算 🟨

**核心问题**：两个 loss 同处一个训练步时，梯度怎么合、量纲怎么对齐、谁主谁辅？

**待查证的子问题**

- [ ] 组合形式盘点：加性混合（$$\mathcal{L} = \mathcal{L}_{\rm PPO} + \lambda \mathcal{L}_{\rm SFT}$$）、
  两阶段交替、预热后切换，各自适用什么场景？
- [ ] 「SFT loss 作为 PPO 的辅助项」到底在正则什么？
  常见说法是防灾难性遗忘 / 保格式 / 稳熵，各自有没有实验支撑？
- [ ] 系数 $$\lambda$$ 的量纲问题：SFT 是 token 级 NLL（每 token 一份），PPO 是序列级 surrogate
  （整条再乘 advantage）——**直接相加是不是量纲不一致**？常见实现里各怎么归一化？
- [ ] SFT 项用哪些 token：全序列？只算 prompt？只算正样本（RAFT / 拒绝采样式）？
  用 on-policy 采样出的序列算 SFT loss 时，梯度与 off-policy 演示数据有何不同？
- [ ] 梯度冲突与工程处理：两个 loss 的梯度方向相反时怎么办（PCGrad / 梯度裁剪 / 分参数更新）？
  actor 与 critic 的更新频率、loss 权重是否要分开调？
- [ ] 与 KL 惩罚项的关系：SFT loss 项和 KL-to-SFT-reference 是重复正则还是互补？
- [ ] 框架实现对齐：TRL / verl / OpenRLHF 里这一项分别叫什么、默认值多少？

**参考线索**

- 站内已有：`2026-09-19-rl-notation-and-estimators-deepdive`（符号对照表先对齐，别自造字母）、
  `2026-09-30-rl-variables-in-frameworks`、`2026-09-29-rl-training-frameworks-survey`
- 待查：InstructGPT 的 pretrain gradient 混合（原文有没有给出混合系数与动机）、
  RLHF 开源实现里的 `sft_loss` / `aux_loss` 命名与默认权重（锁 commit 引用）
- 可跑实验：小模型上做 $$\lambda$$ 扫描 + 梯度余弦相似度统计，验证「梯度冲突」是不是真问题

**产出**：先做 $$\lambda$$ 扫描的 pilot 实验拿到数字，再决定写不写；写法同 §1。

---

## 3. 集群调度、故障恢复与弹性训练 🟨

> **2026-10-18 核实**：站内真空白。`kubernetes` / `k8s` / `多租户` / `弹性训练` / `Lustre` 全站零命中，
> `slurm` 只在 `megatron-12` 里提过一次。这是目前缺口里最大的一块。

> **第 01 篇已交付**（commit `7e3d112`）：`_posts/2026-10-18-cluster-fault-tolerance-01-checkpoint-cost.md`
> （368 行），配套 lab 在 ipynbs `experiments/cluster-scheduling/03-checkpoint-cost-model/`。
> 新分类「集群与容错」+ 系列 `fault-tolerance` 已入 `_data/knowledge_graph.yml`。
> **本篇只覆盖 checkpoint 成本模型**，调度语义 / 多租户 / 弹性训练均未动（见下方子问题）。
**核心问题**：一个千卡训练任务从提交到跑完，集群这一层到底做了什么？失败之后怎么续？

**待查证的子问题**

- [ ] 调度语义：gang scheduling 为什么是刚需？K8s 原生的 `Job` 缺什么，才需要 volcano / kueue /
      Slurm 的 `--exclusive` + 拓扑感知分配？（把「调度单元」和 Megatron 的「进程组」对齐起来讲）
- [ ] 拓扑感知分配：NVLink 域 / NVSwitch / 机架 / 胖树，调度器看到的是哪一层？
      分配错了代价是多少（和 `dist-train-07` §3 的 NCCL 分段带宽模型接上）
- [ ] 故障恢复：故障检测（hang 判定、NCCL watchdog）→ 隔离（drain / cordon）→ 恢复（重启 vs 热替换）
      三段的时延分布；重试次数与训练步损失的关系
- [x] checkpoint 策略：写频次的成本模型 —— **已在 01 篇做完**（闭式最优 $$I^\* = \sqrt{2wM}$$，
      写开销与重算开销在最优处相等；异步写压低 $$w$$、分层检查点把重算上限降到 $$i/2$$；
      周期故障最坏相位损失翻倍）。
      异步写 / 分层写（显存 → host → 并行 FS）的取舍。**注意**：`megatron-12` 的变更史第 9 条记着
      `dist_checkpointing/` 未展开且未分配编号 —— 那是 Megatron 源码向，本文是集群策略向，别合并
- [ ] 弹性训练：resize 世界大小时，优化器状态 / 数据分片 / LR 怎么重配？
      与 `megatron-11` refit & resharding 的边界在哪（它是权重搬运，这里是作业级伸缩）
- [ ] 多租户隔离：配额、优先级抢占、GPU 共享（MPS / MIG / 时间片）各自的隔离强度与性能代价

**参考线索**

- 站内可接：`2026-08-31-dist-train-07-collective-comm`（NCCL 拓扑感知，但只有 134 行，深度不够，见 §5）、
  `2026-10-20-megatron-12-deterministic-rerun-and-fault-injection`（故障注入，确定性视角）
- 待查：volcano / kueue / Slurm 官方文档的调度插件与拓扑感知实现（锁版本）；
  公开的事故复盘（Meta / 字节 / 阿里云的千卡训练故障率与 MTBF 数据，找不到就标「未见公开数据」）
- **硬约束**：本条很难做纯 CPU 可跑的 companion lab。若拿不到可核验的故障率 / 时延数据，
  按 `WRITING.md` §6「没有实验就不写」处理 —— 要么只写机制与源码证据，要么不写

**产出**：先做调度语义 + checkpoint 成本模型两块的取证，能拿到数字就写单篇长文。
分类需新建（现有 14 个分类没有「集群系统」），动 `_data/knowledge_graph.yml` 后必跑
`node tools/kg_layout_check.mjs`。

---

## 4. 数据 pipeline 与存储 I/O 🟨

> **2026-10-18 核实**：站内真空白。`dataloader` / `数据加载` / `并行文件系统` / `对象存储` /
> `Lustre` / `GPFS` / `Ceph` 全站零命中。153 篇里没有一个字讲数据怎么进 GPU。
> **第 01 篇已交付**（commit `c035035`）：`_posts/2026-10-18-data-pipeline-01-loader-and-sharding.md`，
> 538 行，配套 lab 在 ipynbs `experiments/data-pipeline/`（两个纯 CPU numpy 实验）。
> 新分类「数据与存储」+ 系列 `data-pipeline` 已入 `_data/knowledge_graph.yml`。

**核心问题**：训练侧的「喂料」瓶颈 —— 为什么 GPU 在等数据，以及存储这一层能做什么。

**待查证的子问题**

- [x] 加载流水线：读盘 / 解码 / tokenize / batch 组装的时间线在哪一段断？
      DataLoader worker 数与 pinned memory 的真实收益曲线（可跑 CPU 实验）
      —— **已在 01 篇做完**。结论：worker 数收益是阶跃的，越过 $$L/T_s \ge B/T_g$$ 后归零；
      prefetch 深度在几乎所有配置下无影响，起作用的是生产盈余 $$\alpha$$
      （1.00 → 1.25 就把损失从 2.5% 压到 0）。
- [x] 数据切分语义：分布式下谁负责 shuffle、seed 怎么同步 —— **已在 01 篇做完**
      （四种策略的正确性 + 分布偏斜 + shard 是 cluster）。核心发现：真正的病不是重复/遗漏
      而是每个 rank 的类别分布偏斜，TV 是随机分配的 12 倍，且常规指标查不出来。
- [ ] 数据格式：webdataset / mds / arrow / parquet 的随机访问代价差异
      —— **未做**，是 02 篇的主候选。
- [ ] 存储层：并行文件系统（Lustre / GPFS）与对象存储（S3）在吞吐、元数据、成本上的分野；
      本地 NVMe 缓存层（Alluxio / JuiceFS）放在哪一级
- [ ] 与训练的耦合：checkpoint 写与数据读争抢同一条 I/O 路径时会发生什么；
      多机同时读同一份数据集的元数据风暴（hot spot）
- [ ] 长上下文 / 多模态场景：样本大小从 KB 到 MB 量级后，上面哪些结论失效

**参考线索**

- 站内可接：`2026-09-29-profiling-01-training`（迭代时间线里已经能看到 dataloader 空档，但没展开）
- **写 02 篇前先想清楚怎么突破 01 篇的模型边界**：01 篇的 prefetch 结论明确限定在
  「worker 连续生产、无 batch 边界、无 IPC 开销」的模型内。真实 DataLoader 的 prefetch
  还负责隐藏跨进程传 batch 张量的延迟，突破不了这个边界就别重复 01 篇。
- 待查：PyTorch DataLoader 源码锁 commit；webdataset / litdata 的实现取舍

**产出**：02 篇聚焦「存储层 + 格式随机访问代价」，需先做格式扫描的 pilot 实验
（纯 CPU 可跑：扫 shard 大小 / 格式 / 随机访问模式，量读放大与吞吐）。

---

## 5. NCCL / 通信栈本体：把 `dist-train-07` 补成深文 ⬜

> **2026-10-18 核实**：**通信-计算重叠已经不缺** —— `megatron-01` 的规划条目就是
> 「进程组 + 通信与计算重叠」，已发 1507 行源码级长文，06/07/10 各篇里也有 overlap 分析。
> 真正浅的是 **NCCL 本体**：`dist-train-07` 全文 134 行、6 节，NCCL 只占 §3 一节（约 20 行）；
> 全站 26 篇提到 NCCL，全是散布引用，没有一篇以它为**主体**。

**核心问题**：NCCL 拿到一个集合操作之后，到底做了什么决定？为什么同样的 All-Reduce 在不同拓扑上差几倍？

**待查证的子问题**

- [ ] 算法选择：ring / tree / 递归 halving-doubling 的选择条件（消息大小 × 节点数）在源码里怎么落地
- [ ] 传输与协议：P2P_DIRECT / SHM / NET / IB / Socket 五条路径；GPUDirect RDMA 与 NVLink 的分工；
      PXN、SHARP（在网计算）各自省掉了什么
- [ ] 拓扑感知：拓扑发现（XML / 图划分）→ 通道搜索的建模代价；为什么建通信域要几秒
- [ ] 调优面：`NCCL_TOPO_FILE`、`NCCL_ALGO`、`NCCL_PROTO`、buffer 大小、线程数各自的真实影响
- [ ] 可观测：nccl-tests 的 busbw 口径为什么容易读错；与 profiling 系列 §0 的度量口径对齐

**边界（别重复写）**

- Ring All-Reduce 的数学推导 —— `dist-train-07` §2 已有，只引用不重写
- 进程组与网格切分 —— `megatron-01` 已有 1507 行，不碰
- 各并行策略里的通信模式 —— `dist-train-01~06` 各篇自带，不重写

**产出**：一篇以 NCCL 为主体的源码级长文，或并入「分布式训练」分类做扩展篇。
**必须锁 NCCL 版本 + commit**，行号给 permalink。无 GPU 实测条件时，只写机制与参数语义，
不编造实测带宽数字（`WRITING.md` §6）。

---

## 6. 成本与效率：从 roofline 走到钱 ⬜

> **2026-10-18 核实**：**硬件与 roofline 不缺** —— `op-00` 整篇讲 Roofline 模型、23 篇提及，
> `chip-09` 是 A100→H100→B200 的选型案例，`chip-10` 讲性能建模，`op-01` 讲硬件架构。
> 真空白的是**成本侧**：`TCO` / `成本优化` / `每 token 成本` 全站零命中。

**核心问题**：一个大模型服务 / 一次训练的钱花在哪，怎么算，怎么省。

**待查证的子问题**

- [ ] 成本模型：把 roofline 的「算力 / 带宽」翻译成「每 token 成本 / 每 step 成本」，
      需要哪几个中间量（MFU、卡时单价、利用率、能耗）？
      与 `profiling-00` 的 MFU / HFU / goodput 口径接上
- [ ] 选型：同算力下不同卡的每美元吞吐怎么比（算力、显存、带宽、互连四项加权）；
      国产卡的价格/性能比要不要写（数据难核验，倾向不写具体数字）
- [ ] 推理侧成本杠杆：量化、投机解码、continuous batching、PD 分离各自省的是哪一笔钱，
      量级差多少 —— **站内已有素材可直接复用**：量化系列 25 篇、投机解码 02 篇（α/γ/c 收益模型）、
      `serving-01`
- [ ] 训练侧成本杠杆：MFU 每提升 5% 对应多少钱；重计算 / 卸载省显存是否划算（与 `megatron-08` 显存账本接上）
- [ ] 闲置与碎片：预留、排队、故障重跑的隐性成本占多少

**风险**：本条天然偏商业与采购，且几乎不可能做可复现实验，容易写成没有证据的空谈。
**建议只写「成本模型 + 已有站内杠杆的成本换算」这一半**，选型与报价部分不写。动笔前先确认这一点。

---

## 7. 推理引擎对比矩阵：从两家扩到 N 家，并做成可维护的活页 ⬜

> **2026-10-18 核实**：**不是「缺」，是「没扩出去、没发布」**。草稿
> `deep-dive-serving-engine/` 里已有 `02-feature-matrix.md`（720 行 / 13 节，含投机解码、
> attention backend 44 vs 24、结构化输出、LoRA、PD 分离、量化格式、可观测性 38 vs 156、默认开关）、
> `10-selection-playbook.md`、`08-benchmark-crosscheck.md`；已发布 `serving-01`。
> 缺的是：只覆盖 **vLLM / SGLang 两家**，没有 TensorRT-LLM / LMDeploy / MLC-LLM / TGI。

**待查证的子问题**

- [ ] 第三、第四引擎的取证：TensorRT-LLM（编译期 vs 运行期、engine 构建时间、支持的模型面）、
      LMDeploy（TurboMind / PyTorch 双后端、W4A16 与国产卡支持）。同样要**锁 commit + permalink**
- [ ] 矩阵维度是否需要加列：CUDA graph 捕获范围、多 LoRA 热切换、structured output 的 grammar 后端、
      PD 分离的 KV 传输协议
- [ ] 「持续维护」怎么做：版本号 + 取证日期进表格；每次引擎发版是重跑全表还是只更新差异行？
      定一个能坚持的节奏（建议季度）
- [ ] 性能数字单独分层：功能矩阵（可静态取证，稳）与性能对比（需实测，易过期）分开维护，
      性能部分沿用 `08-benchmark-crosscheck.md` 的 A/B/C 三级可信度口径

**产出**：先补 TensorRT-LLM 一家的源码取证做 pilot，验证矩阵维度够不够用，再决定全量扩到几家。
矩阵以 `_posts/` 单篇形式发布（别做成 Jekyll data 文件，改起来重），文首标注「取证版本与日期」。

---

## 8. 安全 / 多租户 Infra —— 暂不启动 ⏸

> **2026-10-18 核实**：站内确实是空白（沙箱 / 多租户 / 审计的命中全在 RL 与具身语境，不是 Infra 侧）。
> 但**不建议进 TODO 排期**，理由：与现有四条主线（Megatron / 量化 / RL / world model）都不咬合；
> 且很难做可复现实验，按 `WRITING.md` §6 基本写不成。等 Infra 系列成规模后再看。

---

## 备注

- §1 §2 属于 RL 主线；§3–§8 属于 AI Infra 主线。动笔前先读
  `.workbuddy/memory/topics/series-boundaries.md` 确认不与已有系列撞车
- 公式一律用来源文献的通行写法，不自造符号；行内公式用 `$$...$$`，写完跑 `python3 tools/check_liquid.py`
- **2026-10-18 覆盖盘点的已否决项**（别重复立项）：
  - 编译与算子层 —— **不缺**。`op-` 14 篇 + `compiler-` 13 篇（含 TVM、XLA/MLIR/IREE、
    torch.compile/Triton/TensorRT、从零写迷你 AI 编译器）+ `xdsl` + `chip-` 10 篇 + `flex-attention-01`
  - KV cache 显存管理 —— **不缺**。`llm-quant-24-kv-cache` 1042 行讲 KV 量化；
    `serving-01` §2/§3/§4 源码级对比了 vLLM 的 block 链式哈希 vs SGLang 的 radix tree、
    prefix 粒度差、8 种驱逐策略（LRU/LFU/FIFO/MRU/FILO/priority/SLRU/TLRU）
  - 唯一没单独成篇的是 **PagedAttention 原论文本身**（全文只在 `op-07` 表格里提一句），
    但 serving-01 §2.1 已经把 vLLM 侧的实现讲透了 —— 不值得单独立项
