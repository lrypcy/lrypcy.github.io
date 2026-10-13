# 引用池 · 05 / 08 / 16 三篇的外部引用候选

> 监工 = Sisyphus。契约见 `SERIES_SPEC.md` §3.2（引用规范）与 §3.4（许可白名单）。
>
> **三篇的阻塞项是同一个：外部引用密度。** 本文件把**已核验**的候选集中在这里，
> 省掉三个窗口各自重复检索。

---

## 0. 使用纪律（先读，否则白做）

§3.2 原文：

- 每条重要论断**至少挂 1 条外部或源码出处**
- **禁止**「据官方文档」这种不落 URL 的说法
- **禁止**把二手贴的转述当出处 —— **必须追到原论文 / 原仓库**
- **禁止为凑数堆 URL** —— 每条必须挂在具体论断上

**本文件给的是候选，不是成品。** 引用前必须：

1. `webfetch` 那个 URL，确认能打开、且内容确实支撑你要写的论断
2. 在正文对应论断处挂上（不是堆在参考节）
3. **arXiv 图另需看 abs 页有无 CC 标识**（§3.4）—— 没有 CC 标识的论文，其图**一律自绘**

**计数口径**：排除 `lrypcy.github.io`（站内互链）与 `github.com/lrypcy/ipynbs`（A 级实验脚本 permalink）。
门槛 = `行数 ÷ 300 × 8`，向上取整。

| 篇 | 行数 | 门槛 | 现有 | 需补 |
| --- | --- | --- | --- | --- |
| 05 | 1446 | **≥ 38** | 5 | **+33** |
| 08 | 1411 | **≥ 38** | 11 | **+27** |
| 16 | 667 | **≥ 18** | 9 | **+9** |

---

## 1. 论文：已核验的 arXiv ID

以下 ID 我**逐个打开过 abs 页或检索到原文**，可直接用。

| arXiv ID | 标题 / 出处 | 能支撑什么 | CC 标识 |
| --- | --- | --- | --- |
| `1910.02054` | Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* | ZeRO-DP 阶段划分；05 §8「为什么不是 ZeRO-1」的**主要对照** | 未核，引用前看 abs 页 |
| `2104.04473` | Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* | Megatron 论文本身；DP + TP + PP 组合的原始出处 | 未核 |
| `2205.05198` | Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* | **已在 01/03/04/08 引用**；selective recompute 的原始出处 | **已核：无 CC 标识 → 其图不可复制，只能自绘** |
| `2507.20534` | Kimi AI, *Kimi K2: Open Agentic Intelligence* | **QK-Clip 与 MuonClip 的唯一原始出处**（§2.1 提出 QK-Clip、per-head 变体、MLA 下只裁 unshared 分量）。**05 §12.4 `qk_clip.py` 的正确出处** | 未核 |
| `2409.20325` | Bernstein & Newhouse, *Old Optimizer, New Norm: An Anthology* | Newton–Schulz 迭代作为正交化手段的文献依据（05 §12 讲 `newton_schulz_tp`） | 未核 |
| `1511.01856` | *cudnn* | **已在 08 引用** | 未核 |

### Muon 本身没有 arXiv 论文 —— 用官方 blog

Muon 的原始出处是作者的 blog，**不是 arXiv**：

- `https://kellerjordan.github.io/posts/muon/` — Jordan et al., *Muon: An optimizer for hidden layers in neural networks*（2024-12-08）
  - 含 Newton–Schulz 五步迭代的系数推导、`Ortho(G) = argmin{O}` 的定义、与 Tuddenham et al. Orthogonal-SGDM 的对比、bfloat16 下可稳定运行
  - 05 §12.1 说「`muon.py` 只是个工厂壳，实现不在 Megatron 仓库里」，**这篇是那个实现的上游出处**

配套：

- `https://arxiv.org/abs/2202.07052` — Tuddenham et al., *Orthogonalising gradients to speed up neural network optimisation*（Muon 的前身 Orthogonal-SGDM）

### 一条待你核的线索

检索时出现 `arXiv:2601.19156`（*Convergence of Muon with Newton–Schulz*，2026-01），
是 Muon 的**收敛性理论**。05 若要论证「Muon 换掉 SVD 后的理论依据」，这篇比 blog 更合适。
**但我未核其 abs 页与 CC 状态，用前必须自己打开确认。**

---

## 2. 官方文档：已核验可直接复用的 URL

以下 7 条**在 04 篇里已被使用且验收通过**（`audit_permalinks` 0 broken），URL 可达：

| URL | 支撑什么 |
| --- | --- |
| `https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html` | 各并行策略的适用边界与推荐组合 |
| `https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/generalized_tensor_parallel.html` | GTP 的官方 API 定义 |
| `https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html` | 进程组 getter 的官方签名 |
| `https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.generalized_tensor_parallelism.html` | GTP 实现类的官方签名 |
| `https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.gtp_api.html` | GTP 门面层职责 |
| `https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced/fp8_primer.html` | FP8 scaling 与 amax 归约 |
| `https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/distributed_weight.py` | TE `DistributedWeight` 协议 |

**同一套 URL 结构适用于 05 / 08 / 16** —— 把 `core.parallel_state` 换成你要的模块名即可，例如
`.../apidocs/core/core.distributed.param_and_grad_buffer.html`。
**每个都要 `webfetch` 确认可达再引**，不要照抄格式就算数。

---

## 3. 按篇分配

### 05 · 数据并行与优化器（需 +33）

| 章节 | 该引什么 | 去哪找 |
| --- | --- | --- |
| §1 三套 DP 分叉 | DDP / Megatron-FSDP / FSDP2 三条路径的**官方定义** | `docs.nvidia.com/megatron-core/…/apidocs/core/core.distributed.*`；PyTorch FSDP 官方文档（已在参考节）+ FSDP2 的官方说明 |
| §2 `_ParamAndGradBuffer` | 连续 buffer 的语义、`BufferType` 的官方定义 | `core.distributed.param_and_grad_buffer` API 页 |
| §2.5 NVFP4 打包 | NVFP4 格式定义、block scaling | TransformerEngine FP8/FP4 文档（**04 只用了 fp8_primer，NVFP4 是新的**） |
| §3/§4 bucket 装箱与 padding | 通信分桶、bucket size 上限 | NCCL 官方文档的 collective / bucket 说明；`training.py` 里 `resolve_ddp_bucket_size` 的 docs 链接 |
| §5 三段式同步 | 集合通信异步语义、CUDA stream 与通信 | CUDA Programming Guide 的 stream 语义；NCCL 文档 |
| §6 overlap 三组合 | overlap 机制的官方定位 | Megatron-Core 文档；DeepSpeed ZeRO 文档作对照 |
| §7/§8 DistOpt 分片 vs ZeRO-1 | **ZeRO-2 / ZeRO-3 的分片语义** | `1910.02054` 是 ZeRO 主论文；ZeRO-3 与 ZeRO-Infinity 需另找（**ID 我未核**） |
| §11 fp32 累加归约 | reduce-scatter 的 fp32 累加 | NCCL 文档；PyTorch `distributed` 文档 |
| §12 Muon / qk_clip | **§1 表里的 3 条** | `kellerjordan.github.io` + `2507.20534` + `2409.20325` |

**特别提示**：05 参考节里那 19 条 `github.com/lrypcy/ipynbs/...` **不计入密度**，
别以为已经够了。

### 08 · 显存账本（需 +27）

08 现有 9 条唯一外部来源（4 arXiv + 3 PyTorch + 1 Megatron 指南 + 1 arXiv 许可页）。
**缺口集中在低精度与 offload 两块，这两块现在一条官方文档都没有。**

| 章节 | 该引什么 |
| --- | --- |
| §2.5 NVFP4 打包 | TransformerEngine NVFP4 / block scaling 文档（与 05 共用） |
| §2 normal vs output-discarding | `torch.utils.checkpoint` 已在参考节；需 `saved_tensors_hooks` 的机制说明（已在参考节）→ **再补 autograd saved tensor 的官方语义页** |
| §3 DSA 与 SP 互斥 | Megatron-Core `parallelism-guide`（已在）+ `param_layout` API 页 |
| §4 offload 三档 | `docs/user-guide/features/fine_grained_activation_offloading.md` 与 `optimizer_cpu_offload.md`（**`transformer_config.py` L1334/L1340/L1346 三处 docstring 直接指向这些锚点**，这是现成的官方出处） |
| §5 `hybrid_optimizer.py` | `core.optimizer` API 页；PyTorch CPU offload 文档 |
| §6 卸载与 CUDA Graph 冲突 | `transformer_config.py` docstring 里那条 `#cuda-graph-integration` 锚点；CUDA Graph 官方文档的 stream 约束 |
| §1 账本与碎片 | PyTorch CUDA 缓存分配器（已在参考节）→ 补 `torch.cuda.memory` API 页 |

### 16 · hybrid / MoE 装配（需 +9）

16 现有 9 条唯一外部来源（6 arXiv + 3 Python/pytest 文档），**PyTorch 侧一条都没有，明显偏科**。

优先补：

- **PyTorch 官方文档**：16 讲装配，必然涉及 `torch.nn.Module`、`Protocol`、`torch.distributed` 的语义 →
  `docs.pytorch.org` 的对应 API 页（现在只有 `docs.python.org`，不合理）
- **Megatron-Core hybrid / Mamba 相关 API 页**：`models/hybrid/` 的 `HybridLayerAllocation` 等类的官方签名
- **TransformerEngine 对应模块文档**：`GatedDeltaNet` / `linear_attention` 的官方页
- `transformer_config.py` 里 hybrid 与 mamba 字段的 docstring 常直接链官方文档 —— **先 grep 这些 docstring 的 URL，比外部搜索准**

---

## 4. 各窗口只许写自己那一篇

| 窗口 | 篇 | 只许写 |
| --- | --- | --- |
| `lrypcy_github_io:6` | 05 | `_posts/2026-10-10-megatron-05-data-parallel-and-optimizers.md` |
| `lrypcy_github_io:5` | 08 | `_posts/2026-10-12-megatron-08-memory-ledger.md` |
| `lrypcy_github_io:2` | 16 | `_posts/2026-10-14-megatron-16-hybrid-and-moe-model-assembly.md` |

**三篇都不许 `git add` / `commit` / `push`。** 监工逐篇验收后统一提交。
14 / 15 两篇已由监工提交推送（`803bc13` / `7cc7b60`），**别动它们**。