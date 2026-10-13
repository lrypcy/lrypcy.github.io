# 工单 · 08 篇退回修复（两项文字缺陷）

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`。原始工单见 `TASK-08.md`。

**实验侧全部通过，不用返工**：`selective_equivalence.py` 7 条断言全过
（normal / output-discarding 前向逐位恒等、段内梯度逐位恒等、段外梯度相对误差 2.20e-16、
被丢弃输出 storage 归零、去掉 RNG 恢复后梯度确实变了）；
§8.4 回填 5 个 ` ```text ` 块数字全部对得上脚本真实输出；
29 个 permalink **0 broken**；G1/G2/G4/G9 全过。

只修下面两项文字缺陷。

---

## 缺陷一 · 参考节 2 处裸行号

`audit_permalinks.py` 的 `bare （L<n> refs` 判据抓到 2 处：

| 行号 | 位置 | 现状 |
| --- | --- | --- |
| L1384 | `megatron/core/tensor_parallel/random.py` 那条 bullet | `……（L862-L875 fp8 怪癖；L967-L978 换指针；L985-L996 丢弃）` |
| L1386 | `megatron/core/transformer/transformer_block.py` 那条 bullet | `……（L662 分派条件）` |

问题不在于「这条 bullet 没写文件名」—— 文件名在 bullet 开头写了。
问题在于**同一条 bullet 内其余行号都是无括号风格**（`—— L32`、`—— L600-L672`、
`—— L748-L800`），只有这两处多套了一层全角括号。04 篇与 02 篇的参考节都是 **0 处**，
所以系列惯例是严格写法，这两处属于破例。

**改法**：去掉那两处全角括号，与同 bullet 内其余写法保持一致。改完确认：

```bash
~/Software/miniconda3/bin/python3 tools/audit_permalinks.py _posts/2026-10-12-megatron-08-memory-ledger.md
```

期望输出里 `bare （L<n> refs` 那一项消失。

---

## 缺陷二 · 外部引用密度不足（硬门禁，§3.2）

契约 §3.2 目标：**每 300 行至少 8 条外部引用**。本篇 1411 行 → **需 ≥ 38 条**。
当前 **11 条**（1/128 行），**缺 ≥ 27 条**。

### 当前全部外部引用（严格口径 9 个唯一来源）

```
arXiv:1511.01856                              cuDNN
arXiv:1910.02054                              ZeRO
arXiv:2104.04473                              Megatron
arXiv:2205.05198                              激活重计算
docs.nvidia.com/megatron-core/…/parallelism-guide.html
docs.pytorch.org/docs/stable/checkpoint.html
docs.pytorch.org/docs/stable/generated/torch.autograd.graph.saved_tensors_hooks.html
docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
info.arxiv.org/help/license/reuse.html
```

> 自己 ipynbs 仓库的脚本 permalink（`github.com/lrypcy/ipynbs/...`）是 **A 级证据，不计入外部引用**。

### 禁止的做法（§3.2 原文）

- 「据官方文档」这种**不落 URL** 的说法
- 把二手贴的转述当出处 —— 必须追到原论文/原仓库
- **为凑数堆 URL** —— 每条必须挂在具体论断上

### 至少该覆盖的官方来源（按本篇章节）

| 章节 | 论断 | 该找的官方出处 |
| --- | --- | --- |
| §2.5 | NVFP4 打包进 param buffer，numel 减半 | TransformerEngine 的 FP8/NVFP4 block scaling 文档；`fp4_utils.py` 官方 API 页 |
| §2.3 | normal vs output-discarding checkpointing 二分 | `torch.utils.checkpoint` 与 `CheckpointWithoutOutput` 的上游设计说明；PyTorch autograd 文档里 saved tensor 的语义 |
| §3 | `distribute_saved_activations` 与 SP 互斥、只摊 args[0] | Megatron-Core API guide 的 `param_layout` / transformer_config 具体页面 |
| §4 | offload 三档的官方定位 | `docs/user-guide/features/fine_grained_activation_offloading.md`（`transformer_config.py` L1334/L1340/L1346 三处 docstring 直接指向的锚点）与 `optimizer_cpu_offload.md` |
| §5 | hybrid_optimizer 搬状态不搬权重 | Megatron-Core API guide 的 `HybridDeviceOptimizer` 页面 |
| §6 | 卸载的 autograd 桥、CUDA Graph 冲突 | `torch.autograd.graph.saved_tensors_hooks` 的官方说明；CUDA Graph 与流相关的 CUDA/NVIDIA 文档 |
| §1 | 账本分桶、NCCL 通信语义 | NCCL 集合通信与通信分桶官方文档 |

**自绘图注里带的外部出处同样计入密度** —— 但每张图仍然必须有图注（§3.3）。

---

## 非门禁 · 由你判断

全篇 1411 行**只有 1 张图**（那张引用关系网图），而 04 篇 7 张、05 篇 12 张。

契约没规定图数下限，所以这不构成退回理由。但这几处很适合自绘：

- §2 `recompute_modules` 的 11 个取值 × normal / output-discarding 二分
- §4 offload 三档的粒度对比（层 / 模块 / 参数）
- §5 hybrid_optimizer「搬状态不搬权重」的数据流
- §6 卸载与重计算的冲突关系

加不加你定。**加的话每张都要带图注，且不许复制 NVIDIA 官方图**（契约 §3.4）。

---

## 禁区（不变）

- `_posts/2026-09-28-megatron-00-foundation.md` ← 窗口 2 在改
- `_posts/2026-10-08-megatron-02-expert-parallel.md` ← 窗口 2 在改
- `_posts/2026-10-10-megatron-05-data-parallel-and-optimizers.md` ← 窗口 6 在改
- `SERIES_SPEC.md`、`TASK-05.md` ← 不许动
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`
- `tools/megatron_bench/` 全部（存量，不许扩写）
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，不许 pull / checkout

**你只许写两个文件**：本篇正文，以及
`~/Desktop/Projects/sandbox/ipynbs/experiments/megatron/memory_ledger/` 下的实验文件
（若需补实验；三个脚本已通过，优先只改文字）。

## Git 纪律

**不许 `git add` / `git commit` / `git push`。** 改完 → 跑自查 → 报监工 → 等指令。
已记录两次越权（`7842859`、`ff675e6`），第三次即终止写权限。

## 自查

1. `~/Software/miniconda3/bin/python3 tools/audit_permalinks.py <file>` —— `bare （L<n> refs` 必须消失
2. `python3 tools/check_post.py <file>`
3. `python3 tools/check_mermaid.py <file>`
4. `python3 tools/scan_garbled.py <file>`
5. `python3 tools/fix_quotes.py <file>` —— 必须 0 转换
6. 外部引用重数 ≥ 38（严格口径，监工用同口径复核）
7. 图注判据（若加了图）：紧跟围栏之后、以 `> ` 或 `图 N：` 开头、含「出处」/ URL / 行号锚点