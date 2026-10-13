# 工单 · 10 CUDA Graph：把 step 图化到什么程度

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**（§2.1 孤儿归属 / §3 引用规范 / §3.4 许可白名单）。
> 引用候选见同目录 `CITATION-POOL.md`。

---

## 0. 你只许写一个文件

`_posts/2026-10-18-megatron-10-cuda-graph.md`，目标 1300–1800 行。

实验脚本落 **`ipynbs` 仓库**的 `experiments/megatron/cuda_graph/`（新建）。
本仓库 `tools/megatron_bench/` **只读，不许扩写**（契约 §7）。

---

## 1. 范围（监工已核定）

编号 10 原定「算子融合与 CUDA Graph」。**监工按契约 §2 变更史第 4 条的先例拆开**
（那条写的是：「EP 单主题 1897 行已超『一个模块一篇』承载上限；六主题合并至少
6000 行，不可读也不可审」——本篇若合并约 7000 行，正是同一情形）。

### ✅ 本篇负责：CUDA Graph 主线（约 3900 行）

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `megatron/core/transformer/cuda_graphs.py` | **3311** | **全系列最大单文件**。已核的公开符号：`_get_cuda_graph_stream` L107 / `_get_tensor_alias_chain` L154 / `_apply_cudagraph_buffer_metadata` L166 / `_tag_cudagraph_buffer_saved_for_backward` L185 / `get_gtp_runner_streams` L200 / `_set_skip_fp8_weight_update_tensor` L207 / `is_graph_capturing` L244 / `_set_capture_start` L249 / `_set_capture_end` L255 / `is_graph_warmup` L261 / `_set_warmup_start` L266 / `_set_warmup_end` L272 / `CudagraphBufferMetadata` L279 / `ArgMetadata` L310 / `alloc_tensor_from_graph_mempool` L336 / `tree_map` L350 |
| `megatron/core/full_cuda_graph.py` | 267 | 整 iteration 图化的边界判定 |
| `megatron/core/optimizer/optimizer_cuda_graph.py` | **68** | **契约 §2.1 指定孤儿 → 必须写成独立一节** |
| `megatron/core/tensor_parallel/gtp_cuda_graphs.py` | 249 | 与 04 篇的 GTP 相扣 |

`optimizer_cuda_graph.py` 的调用链（监工已核）：

```
optimizer_config.py:398      optimizer_cuda_graph: bool = False
arguments.py:2897            '--optimizer-cuda-graph', action='store_true'
optimizer/__init__.py:553    "capturable": config.optimizer_cuda_graph
training.py:88               from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
training.py:4405             if args.optimizer_cuda_graph:
```

**关键点**：它不是「把 optimizer step 图化」这么简单，而是**通过 `capturable`
把整个优化器切到 capturable 模式**（Adam 的 `step` 必须能被 CUDA graph 捕获）。
68 行改的是优化器的可用性边界，不是性能选项。

### ❌ 本篇不负责：算子融合（约 3100 行，另立编号，**编号等监工与用户定**）

`megatron/core/fusions/` 11 个模块：`fused_bias_swiglu.py` 341 /
`fused_bias_geglu.py` 442 / `fused_mla_yarn_rope_apply.py` 996 /
`fused_softmax.py` 359 / `fused_indices_converter.py` 287 /
`fused_layer_norm.py` 169 / `fused_cross_entropy.py` 148 /
`fused_weighted_squared_relu.py` 110 / `fused_pad_routing_map.py` 100 /
`fused_bias_dropout.py` 94 / `fused_bias_gelu.py` 55。

**你不许自己给它们编号。** 但**可以**在正文里引用它们说明「哪些算子已经是融合的、
为什么它们是图化的前提」—— 一句话带过可以，写成独立一节不行。

---

## 2. ⚠️ 一处弃用迁移，是本篇可以写进正文的一个论点（监工 2026-10-09 核正）

训练侧真实开关是 **`cuda_graph_impl`**（`transformer_config.py` **L1068**）：

```
cuda_graph_impl: Literal['none', 'local', 'transformer_engine', 'full_iteration'] = "none"
```

而 `cuda_graph_scope` 是它的**旧名**，正在被弃用。同一段 docstring 里能看完整条迁移链：

| 字段 | 行号 | 状态 |
| --- | --- | --- |
| `enable_cuda_graph: bool` | **L1042** | docstring 首行即 `"""DEPRECATED and replaced by cuda_graph_impl.` |
| `external_cuda_graph: bool` | **L1064** | 同上，`"""DEPRECATED and replaced by cuda_graph_impl.` |
| `cuda_graph_scope` | L1049 | 只在 `cuda_graph_use_single_mempool` 的 docstring 里作为**旧名**出现 |
| `cuda_graph_impl` | **L1068** | 现行字段，四档取值 |

`arguments.py` **L282-L289** 还有一条 CLI 弃用告警：`--cuda-graph-modules` 传某些 scope 串时
提示 `deprecated and has no effect`，引导改用 `--cuda-graph-impl=local`；
**L614** 另有 `Do not use --enable-cuda-graph or --external-cuda-graph with --cuda-graph-impl`。

**训练与推理是两条正交的轴**（`cuda_graph_impl` docstring L1069-L1081 原文）：

- 训练由 `cuda_graph_modules` 控制（枚举见 `megatron/core/transformer/enums.py` **L61-L73**：
  `attn` / `mlp` / `moe` / `moe_router` / `moe_preprocess` / `mamba`）
- 推理由 `inference_cuda_graph_scope` 控制（`InferenceCudaGraphScope`）
- `"transformer_engine"` 与 `"full_iteration"` 两档 **不支持推理图** ——
  docstring 明写 `inference_cuda_graph_scope must be "none"`
- `"full_iteration"` 档下 `cuda_graph_modules` 无效且必须为空（L1080-L1081）

两张约束表在 `megatron/core/transformer/cuda_graph_config.py`（**136 行**）：

- `CUDA_GRAPH_MODULES_DEPRECATIONS`（**L10-L13**）：旧 scope 串 → 新字段与新值
- `ALLOWED_INFERENCE_SCOPES`（**L17-L22**）：每个 `cuda_graph_impl` 取值允许的推理粒度集合，
  由 `transformer_config.__post_init__` 与 `validate_args` 共用（L15-L16 注释说明是为了避免重复）

**三个可写的论点**：

1. 一个 flag 的弃用要跨 `transformer_config`（字段）、`arguments.py`（CLI 告警）、
   `cuda_graph_config.py`（迁移表）三处同步 —— 只改一处会留下静默失效的旧名。
2. 训练图化与推理图化共用一套「粒度」词汇，但**允许的组合是一张白名单而不是笛卡尔积**。
3. `full_iteration` 档把整个 forward-backward 路径（不含 optimizer step）图化成一张 ——
   这是本篇的主线，它与 §1 的边界判定是一回事。

> （监工此前在本工单写「`cuda_graph_scope` 属于推理路径、训练侧开关不叫这个名字」是**错的**：
> 该名字在训练侧 `transformer_config.py:1049` 就有，是旧名而非推理专用名。已更正。）

---

## 3. 源码基准

仓库 `~/Desktop/Projects/github/Megatron-LM`，**commit `60e039626`**。
**不许 `pull` / `checkout`**，先 `git -C ~/Desktop/Projects/github/Megatron-LM log -1 --format=%H` 确认。

行号必须**现取现核**：`git show 60e039626:<path> | sed -n 'A,Bp'`。

---

## 4. 骨架（照此写，可细化）

### §1 图化的边界条件
哪些部分能图化（shape 固定的 attention / MLP / optimizer step）、
哪些不行（含 CPU-GPU 同步、动态 shape 的集合通信）。
`full_cuda_graph.py` 267 行是判定入口。

### §2 内存池管理
`alloc_tensor_from_graph_mempool`（L336）与 `ArgMetadata`（L310）、
`CudagraphBufferMetadata`（L279）：`graph_pool_handle` 复用、
以及多 stream 依赖的正确处理。**这是图化最容易 hang 的地方。**

### §3 warmup 与 capture 两阶段
`is_graph_warmup` L261 / `_set_warmup_start` L266 / `_set_warmup_end` L272 /
`is_graph_capturing` L244 / `_set_capture_start` L249 / `_set_capture_end` L255
—— 为什么必须先 warmup 再 capture。

### §4 集合通信为什么必须留在图外
CUDA Graph 与 NCCL 通信的冲突。**这是本篇最值得写深的一节** ——
它与 01 篇的 stream 流派、05 篇的 overlap 状态机、04 篇的 GTP AG/RS stream 全都扣在一起。

### §5 `_apply_cudagraph_buffer_metadata` 与 saved_for_backward
L166 / L185：图化后反向的 buffer 复用与生命周期，
`_tag_cudagraph_buffer_saved_for_backward` 为什么必须打标。

### §6 optimizer step 图化（**契约 §2.1 的孤儿，独立成节**）
`optimizer_cuda_graph.py` 68 行 + §1 给出的调用链。
**不许只在别篇提一句名字就算交差。**

### §7 GTP 与图化
`gtp_cuda_graphs.py` 249 行；`get_gtp_runner_streams`（L200）——
回引 04 篇的 prefetch 链与 AG/RS stream。

### §8 常见坑 + 图 + 引用关系网
图化后 hang、确定性丢失、与 `--tp-comm-overlap` 互相影响。
每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面。

---

## 5. 读大文件的正确打法（契约 §1，**已发生 2 次 30 分钟后台任务被杀**）

`cuda_graphs.py` **3311 行** —— **不要整文件读，也不要丢给 subagent 去「读懂」**：

```bash
M=~/Desktop/Projects/github/Megatron-LM
git -C $M grep -n '^class \|^def ' 60e039626 -- megatron/core/transformer/cuda_graphs.py
git -C $M show 60e039626:megatron/core/transformer/cuda_graphs.py | sed -n '107,200p'
```

⚠️ **本环境下后台 subagent 任务会静默超时、零输出返回。** 监工派出的三个后台任务
（`deep` ×1、`librarian` ×2）全部是 `Total messages: 1` 就 30 分钟超时。
**外部资料检索请自己用 `websearch` / `webfetch`，或看 `CITATION-POOL.md`，不要派后台 librarian。**

---

## 6. 实验设计（本机**无 GPU**，只有 A / C 级）

⚠️ **本篇没有 GPU 就没有真图化实验** —— CUDA graph 必须在真卡上跑。
所以本篇只能做解析类实验，**不许假装测过**（契约 §4 禁止 D 级证据）。

### A 级 · kernel launch 次数的精确计数模型（**本篇唯一能纯算出来的**）
逐层列出算子清单并累加，给出图化前后的差值。
脚本落 `ipynbs/experiments/megatron/cuda_graph/`。

### C 级 · 可选
若 A 级顺利，可加「融合前后 HBM 读写字节数」的解析对照
（**注意：`fusions/` 本身不属本篇**，只做 CUDA Graph 相关的）。

用 `~/Software/miniconda3/bin/python3`（managed python 3.13 无 numpy）。
`results/` 固化输出。

**正文必须写明**：本机无 GPU，图化的实际收益上界**引官方数据并注明卡型规模**（B 级证据），
不许自己编一个吞吐数字。

---

## 7. 引用密度（**这是 05/08/16 三篇全部退回的唯一原因，务必一次做够**）

契约 §3.2：**每 300 行至少 8 条外部引用**。本篇 1300–1800 行 → **需 ≥ 35–48 条**。

前车之鉴（监工亲跑的数）：

| 篇 | 行数 | 唯一外部来源 | 门槛 | 结果 |
| --- | --- | --- | --- | --- |
| 09（修完前） | 1393 | 29 | 38 | ❌ |
| **09（修完后）** | 1417 | **41** | 38 | ✅ |
| **08** | 1483 | **35** | 39 | ✅ |

**计数口径**（监工用的就是这个）：排除 `lrypcy.github.io`（站内互链）与
`github.com/lrypcy/ipynbs`（A 级实验脚本）。

**靠的是几十个不同的真实来源，不是把同一个 URL 重复几十次。**

本篇该找的官方来源（**每条引用前必须自己 `webfetch` 确认可达且真能支撑该论断**）：

- **CUDA Graphs 官方文档** —— `docs.nvidia.com/cuda/cuda-c-programming-guide/`
  的 Graph 章节；**stream capture 的语义与限制**（这是本篇最该引的一类）
- **CUDA C++ Best Practices Guide** —— 图化对内存池与延迟的影响
- `docs.nvidia.com/megatron-core/developer-guide/…` —— 图化相关 user-guide 页
- `docs.nvidia.com/deeplearning/nccl/…` —— 集合通信与图化的冲突（§4 的核心）
- `docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html` —— 作为对照

⚠️ **arXiv / 文档 ID 不许凭印象写。** 监工发现 `qk_clip` 的常见误引是 `2109.12142`
（实际是 Kimi K2 的 `2507.20534`）。**每个 ID 都要打开页面确认标题对得上。**

---

## 8. 禁区

**绝对不许动**：

- `_posts/2026-09-28-megatron-00-foundation.md`、`_posts/2026-10-08-megatron-02-expert-parallel.md`
- `_posts/2026-10-16-megatron-06-pipeline-parallel-schedules.md` ← 另一窗口在写
- `_posts/2026-10-17-megatron-07-context-parallel-and-long-context.md` ← 另一窗口在写
- `SERIES_SPEC.md` 的编号表与模块 E 章节
- `TASK-05.md` / `TASK-06.md` / `TASK-07.md` / `TASK-08-fix.md` / `TASK-16-fix.md` / `CITATION-POOL.md`
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`
- `tools/megatron_bench/` 全部（存量，不许扩写）
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，一条 git 写命令都不许

已入库不许动的篇目：`04`（`6729660`）、`05`（`43b8616`）、`08`（`3c648ec`）、
`09`（`61914f0`）、`14`（`803bc13`）、`15`（`7cc7b60`）、`16`（`db8e842`）。

---

## 9. Git 纪律（契约 §6，第三次即终止写权限）

**不许 `git add` / `git commit` / `git push` / `git add -A` / `git add .`。**
写完 → 自查 → 报监工 → 等指令。**就这三步。**

已记录两次越权（`7842859`、`ff675e6`）。

---

## 10. 写完的自查

四步（`.agents/WRITING.md` §8）：

1. `python3 tools/check_post.py <file>`
2. `python3 tools/scan_garbled.py <file>`
3. `python3 tools/fix_quotes.py --dry-run <file>` —— **必须带 `--dry-run`**
   （不带会直接写盘。2026-10-08 监工已因此踩了两次，修了 10 处被误改的引文）
4. **回填一致性比对** —— 每个 ` ```text ` 块的数字与 `ipynbs` 脚本真实输出按数字字面量对齐

契约 §5 另加：

5. `python3 tools/check_mermaid.py <file>`
6. `~/Software/miniconda3/bin/python3 tools/audit_permalinks.py <file>` —— broken 必须 0
7. `grep -nE '（L[0-9]' <file>` —— 裸行号必须 0
8. **外部引用密度 ≥ 35 条**（监工口径，见 §7）
9. 图注判据：紧跟围栏之后、以 `> ` 或 `图 N：` 开头、且含「出处」/「来源」/ URL / 行号锚点
   （引用关系网图豁免 per-edge permalink）

### 红线

无第一人称叙述生成过程 · 无硬件向内容 · 没实验不写实测数 · 没出处不写数字 ·
不自造数学符号 · 公式内绝对值写 `\lvert o\rvert` 禁裸 `|` · 行内公式 `$$...$$` 禁单 `$` ·
中文引号全角 `“ ”` · 元叙述不入正文

**图**：默认一律自绘。NVIDIA 官方 blog / docs 的图版权保留，**一律不许复制**（§3.4）。

---

## 11. 交付

报告监工：篇名、行数、图数、**外部引用唯一来源数 / 出现次数**、实验断言结果、
自查结果、**没能核实的部分**（不许猜、不许编，写「待验证」）。然后**停下等指令**。