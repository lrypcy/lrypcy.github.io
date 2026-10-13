# 工单 · 06 流水并行：调度器的细节不在论文里

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**（§2.1 孤儿归属 / §3 引用规范 / §3.4 许可白名单）。
> 引用候选见同目录 `CITATION-POOL.md`。

---

## 0. 你只许写一个文件

`_posts/2026-10-16-megatron-06-pipeline-parallel-schedules.md`，目标 1300–1800 行。

实验脚本落 **`ipynbs` 仓库**的 `experiments/megatron/pipeline_parallel/`（新建）。
本仓库 `tools/megatron_bench/` **只读，不许扩写**（契约 §7）。

**为什么这篇重要**：`dist-train` 系列只给了 1F1B 的方块图，原理层已讲透。
本篇是模块 A 的**骨架缺口** —— Megatron 作为「流水线机器」的核心，目前完全空白。
不要重推并行原理，**原理引用 `dist-train-*` 系列，篇目只做源码走读 + 实测**。

---

## 1. 源码清单（行号已由监工核过，**你必须用 `git show 60e039626:` 逐条重核**）

仓库 `~/Desktop/Projects/github/Megatron-LM`，基准 **commit `60e039626`**。
**不许 `pull` / `checkout`**，先 `git -C ~/Desktop/Projects/github/Megatron-LM log -1 --format=%H` 确认 HEAD 仍是它。

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `megatron/core/pipeline_parallel/schedules.py` | **2509** | 本篇主体。**三个入口**：`forward_backward_no_pipelining` **L723** / `forward_backward_pipelining_with_interleaving` **L1019** / `forward_backward_pipelining_without_interleaving` **L2147** |
| `megatron/core/pipeline_parallel/p2p_communication.py` | 675 | 四种 tensor 形状的 p2p 收发 |
| `megatron/core/pipeline_parallel/combined_1f1b.py` | 512 | 1F1B + interleaving 合并 |
| `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` | 669 | hybrid CP 与 PP 联合调度（**07 篇会重用，先把接口讲清**） |
| `megatron/core/pipeline_parallel/bridge_communicator.py` | 1136 | VLM 等多模块场景 |
| `megatron/core/transformer/pipeline_parallel_layer_layout.py` | 321 | 非均匀切层 |
| `megatron/training/arguments.py` | — | flag 的互斥断言，见 §2 |

> ⚠️ **`forward_backward_no_pipelining` 在 `schedules.py` L723，不在 `training.py`。**
> `.workbuddy/memory/topics/series-boundaries.md` 专门记过这条 —— 早期版本写过
> `training.py`，是错的。**这个位置本身就是正文的一个论点**：
> 「不进 PP 就没有调度器」，无 PP 路径与有 PP 路径是两个独立入口。

### 读大文件的正确打法（契约 §1，**已发生 2 次 30 分钟后台任务被杀**）

`schedules.py` 2509 行 —— **不要整文件读，也不要丢给 subagent 去「读懂」**：

```bash
M=~/Desktop/Projects/github/Megatron-LM
git -C $M grep -n 'def forward_backward_no_pipelining' 60e039626 -- megatron/core/pipeline_parallel/schedules.py
git -C $M show 60e039626:megatron/core/pipeline_parallel/schedules.py | sed -n '723,780p'
```

后台 `explore` / `librarian` 只用于**外部资料检索**，不用来读本地源码。

> ⚠️ **本环境下后台 subagent 任务会静默超时、零输出返回。**
> 监工派出的两个 librarian 任务（论文类 / 官方文档类）都是
> `Total messages: 1`（只收到 prompt）就 30 分钟超时。
> **外部资料检索请用 `websearch` / `webfetch` 自己跑，或看 `CITATION-POOL.md`，
> 不要派后台 librarian。**

---

## 2. 已核实的两处坑

### 坑一 · 两个 flag 互斥，撞了直接 assert

`megatron/training/arguments.py`：

```
L804  assert args.decoder_first_pipeline_num_layers is None, (
L825  assert args.pipeline_model_parallel_layout is None, (
L923  + int(args.pipeline_model_parallel_layout is not None)
L936  num_stages = PipelineParallelLayerLayout.get_num_stages_from_str(args.pipeline_model_parallel_layout)
```

两种非均匀切分方式**互斥**，同时给会 assert 失败。
`get_num_stages_from_str` 是从布局串反推 PP 度的入口。

### 坑二 · 无 PP 路径与有 PP 路径是两个独立函数

L723 的 `forward_backward_no_pipelining` 和 L2147 的
`forward_backward_pipelining_without_interleaving` 完全不同构 ——
前者没有 warmup / steady / cooldown 三阶段，没有气泡。
**本篇必须讲清「不切流水线时那些代码根本不存在」这件事**，
而不是笼统说「PP 就是省显存」。

---

## 3. 骨架（照此写，可细化）

### §1 四种 tensor 形状与 sender == receiver
forward / backward / Wgrad / Dgrad 四种形状在 schedule 里怎么保证收发匹配；
`_communicate` 的形状分发逻辑（`p2p_communication.py`）。

### §2 1F1B 的真实代码结构
warmup / steady / cooldown 三阶段各自的循环边界；
每个 steady step 里的 checklist 顺序（哪些 tensor 先收发、哪些后）。
**不要只画方块图** —— 方块图 `dist-train` 已经给过了。

### §3 interleaving 省掉了多少气泡
L1019 的 interleaving 路径；virtual pipeline 与气泡率的解析关系。
**这里必须有 A 级精确解，见 §5**。

### §4 非均匀切层
`pipeline_parallel_layer_layout.py` 321 行 + `get_num_stages_from_str`；
`--pipeline-model-parallel-layout` 与 `--decoder-first-pipeline-num-layers`
分别解决第一/末 stage 显存不均的什么问题。

### §5 p2p 与 overlap
`--overlap-p2p-communication` 的实现位置；
通信与哪个 microbatch 的计算重叠。

### §6 hybrid CP 与多模块
`hybrid_cp_schedule.py` 的接口（**07 篇会重用，只讲接口不深挖**）；
`bridge_communicator.py` 处理的 VLM 多模块场景。

### §7 图 + 引用关系网
每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面。
**建议画一张甘特图**（见 §5 的 C 级实验）—— 气泡在图上肉眼可见。

---

## 4. 引用密度（**这是 05/08/16 三篇全部退回的唯一原因，务必一次做够**）

契约 §3.2：**每 300 行至少 8 条外部引用**。本篇 1300–1800 行 → **需 ≥ 35–48 条**。

前车之鉴（监工亲跑的数）：

| 篇 | 行数 | 唯一外部来源 | 出现次数 | 门槛 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 05（修完前） | 1446 | 4 | 5 | 38 | ❌ |
| **08** | 1483 | **35** | 66 | 39 | ✅ |
| 14 | 653 | 15 | — | 18 | ✅ |
| 15 | 534 | 11 | — | 15 | ✅ |
| 16（修完前） | 667 | 9 | 9 | 18 | ❌ |

**计数口径**（监工用的就是这个）：排除 `lrypcy.github.io`（站内互链）与
`github.com/lrypcy/ipynbs`（A 级实验脚本）。

**08 能过靠的是 35 个不同的真实来源**，不是把同一个 URL 重复 66 次。
契约 §3.2 明写「禁止为凑数堆 URL」「每条必须挂在具体论断上」。

本篇该找的官方来源（**每条引用前必须自己 `webfetch` 确认可达且真能支撑该论断**）：

- `docs.nvidia.com/megatron-core/developer-guide/…` —— pipeline 并行的
  user-guide 页与 API 页（**URL 结构见 `CITATION-POOL.md` §2，但必须逐个 fetch 确认**）
- **PipeDream（arXiv:1806.03377）** —— 1F1B 的原始出处，本篇最该引的一篇
- **DeepSpeed 3D 并行（arXiv:2205.05198 已在本系列引用过，但那是激活重计算；
  3D 并行那篇要另找）** —— interleaved 1F1B 的出处
- `docs.nvidia.com/deeplearning/nccl/…` —— `send` / `recv` 的语义
- `docs.pytorch.org/…` —— `torch.distributed.pipelining` 作为对照实现

⚠️ **arXiv ID 不许凭印象写。** 监工发现 `qk_clip` 的常见误引是 `2109.12142`
（实际是 Kimi K2 的 `2507.20534`）。**每个 ID 都要打开 abs 页确认标题对得上。**

---

## 5. 实验设计（本机**无 GPU**，只有 A / C 级）

证据分级见 `RESEARCH_PLAN.md` §1。**禁止 D 级**（凭印象的数字）。
推不出来又查不到出处的，明写「**待验证**」。

### A 级 · 气泡率精确解析解

- 1F1B 与 interleaved 各自的气泡率公式，扫 PP 度 × microbatch 数 × virtual stages 出表
- 第一/末 stage 显存不均的解析解：对比均匀切 vs 非均匀切

### C 级 · 甘特图模拟（**本篇最值得做的一项**）

用 numpy 模拟 1F1B 的执行时序，**画一张甘特图**，让气泡在图上肉眼可见，
再和 A 级的解析解对账。这一项直接补上「论文只画方块图」的缺口。

脚本落 `ipynbs/experiments/megatron/pipeline_parallel/`，
结构 `<name>.py` + `README.md` + `results/`（输出固化，读者不必自己跑）。
用 `~/Software/miniconda3/bin/python3`（managed python 3.13 无 numpy）。

---

## 6. 禁区

**绝对不许动**：

- `_posts/2026-09-28-megatron-00-foundation.md`、`_posts/2026-10-08-megatron-02-expert-parallel.md`
- `_posts/2026-10-10-megatron-05-data-parallel-and-optimizers.md` ← 另一窗口在改
- `_posts/2026-10-15-megatron-09-low-precision-fp8-fp4.md` ← 另一窗口在写
- `SERIES_SPEC.md` 的编号表与模块 E 章节
- `TASK-05.md` / `TASK-08-fix.md` / `TASK-16-fix.md` / `TASK-09.md` / `CITATION-POOL.md`
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`
- `tools/megatron_bench/` 全部（存量，不许扩写）
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，一条 git 写命令都不许

已入库不许动的篇目：`04`（`6729660`）、`08`（`3c648ec`）、`14`（`803bc13`）、
`15`（`7cc7b60`）、`16`（`db8e842`）。

---

## 7. Git 纪律（契约 §6，第三次即终止写权限）

**不许 `git add` / `git commit` / `git push` / `git add -A` / `git add .`。**
写完 → 自查 → 报监工 → 等指令。**就这三步。**

已记录两次越权（`7842859`、`ff675e6`）。

---

## 8. 写完的自查

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
8. **外部引用密度 ≥ 35 条**（监工口径，见 §4）
9. 图注判据：紧跟围栏之后、以 `> ` 或 `图 N：` 开头、且含「出处」/ URL / 行号锚点
   （引用关系网图豁免per-edge permalink）

### 红线

无第一人称叙述生成过程 · 无硬件向内容 · 没实验不写实测数 · 没出处不写数字 ·
不自造数学符号 · 公式内绝对值写 `\lvert o\rvert` 禁裸 `|` · 行内公式 `$$...$$` 禁单 `$` ·
中文引号全角 `“ ”` · 元叙述不入正文

**图**：默认一律自绘。NVIDIA 官方 blog / docs 的图版权保留，**一律不许复制**（§3.4）。
arXiv 图只在 abs 页**有 CC 标识**时才可引，并标许可。

---

## 9. 交付

报告监工：篇名、行数、图数、**外部引用唯一来源数 / 出现次数**、实验断言结果、
自查结果、**没能核实的部分**（不许猜、不许编，写「待验证」）。然后**停下等指令**。