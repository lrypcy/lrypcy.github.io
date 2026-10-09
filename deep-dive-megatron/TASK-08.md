# 工单 · 新窗口（megatron-08 显存账本）

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**。
> 本工单只说你这一轮干什么。上位文件 `.agents/AGENTS.md`（工程）、`.agents/WRITING.md`（写作）。

---

## 0. 你是谁、你在哪

- tmux 窗口：`lrypcy_github_io:4`（监工新建，模型已锁定）
- 仓库：`~/Desktop/Projects/github/lrypcy.github.io`
- 同工作区还有另外两条线在跑，**禁区见 §6**

---

## 1. 交付物

**唯一交付物**：`_posts/2026-10-12-megatron-08-memory-ledger.md`，1200–1800 行。

标题：**Megatron-LM 深度剖析（08）：显存账本——重计算、卸载与分布保存**

定位：这一篇是**模块 A 的收口**。01–07 各自讲了通信与激活的一个局部，
本篇给出统一的显存六项账本口径，把前面各篇的通信/激活开销放进同一把尺子里量。
所以本篇要大量**反向引用** 01（`GlobalMemoryBuffer`）、02（EP 的 `paged_stash` 与显存释放时机）、
03（SP 省激活显存）、04（GTP 的 remat）、05（`param_and_grad_buffer` 与 bucket padding 浪费）。

### 前置链的特别说明

本篇**写的时候 06（PP）与 07（CP）还不存在**。所以：

- 开头的「前置」行**只链已存在的 00–05**，不要链 06/07 的 URL（会 404）
- 在 06/07 落盘后，监工会让你回来补这一行。**留一个 `<!-- TODO-frontmatter: 06/07 落盘后补前置链 -->` 注释**，不要写在正文可见处

---

## 2. 源码清单（行号已由监工核过，但**你必须逐条重核**）

基准：`megatron-core` **0.20.0**，commit **`60e039626`**。**不许 pull**，
先 `git -C ~/Desktop/Projects/github/Megatron-LM log -1 --format=%H` 确认 HEAD 仍是它。

| 文件 | 行数 | 关键位置 |
| --- | --- | --- |
| `megatron/core/recompute.py` | 189 | 全文只有**一个**公共符号 `checkpointed_forward`（L21）。真正的判断逻辑散在调用侧 —— 这是本篇要讲清的第一个反差 |
| `megatron/core/transformer/transformer_config.py` | 3311 | `recompute_granularity` L525 / `recompute_method` L536 / `recompute_num_layers` docstring L545-546 / `distribute_saved_activations` L550 / `recompute_modules` L553-575 / `fine_grained_activation_offloading` L1308 / `fine_grained_offloading_max_inflight_offloads` L1367 / 校验逻辑 L1862-1887 |
| `megatron/core/pipeline_parallel/fine_grained_activation_offload.py` | **1610** | 见 §3 的路径纠错 |
| `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py` | 477 | **契约 §2.1 指定的孤儿之一，必须写成独立一节** |
| `megatron/core/optimizer/cpu_offloading/README.md` | 13 | 官方对这条路径的定位说明 |

### `recompute_modules` 的 11 个合法取值（docstring 原文，L554-575）

`core_attn`（**默认**）/ `moe_act` / `layernorm` / `mla_up_proj` / `mlp` / `moe` /
`shared_experts` / `gdn_norm_out` / `gdp_in_proj` / `gdp_qkv` / `mhc`

**docstring 里明写了一个必须讲出来的二分**（L573-575）：

- 用 **output-discarding** checkpointing：`moe_act` `layernorm` `mla_up_proj` `gdn_norm_out`
  `gdp_in_proj` `gdp_qkv` `mhc`（7 个）
- 用 **normal** checkpointing：`core_attn` `mlp` `moe` `shared_experts`（4 个）

为什么这个二分重要：normal checkpointing 保存全部输入激活、峰值显存高但反传能重算；
output-discarding 只保存「必要输入」、显存低但**不参与重算的部分在反向时被丢弃** ——
`CHECKPOINT_WITHOUT_OUTPUT` / `CheckpointWithoutOutputManager` 是 2026 的新机制。
`mhc` 还有额外约束（需 `enable_mhc_connections=True`，且不能与 `mlp` 同用）。

### 取证方法（契约 §1，禁止整文件丢给 subagent 读）

```bash
M=~/Desktop/Projects/github/Megatron-LM
git -C $M grep -n 'def checkpointed_forward' 60e039626 -- megatron/core/recompute.py
git -C $M show 60e039626:megatron/core/recompute.py | sed -n '21,60p'
git -C $M show 60e039626:megatron/core/transformer/transformer_config.py | wc -l
```

---

## 3. ⚠️ 计划书里的一处路径错误（监工已核）

`RESEARCH_PLAN.md` 模块 B 的 08 条目把 offload 脚本写成
`fine_grained_activation_offload.py`，读起来像在 `megatron/core/transformer/` 下。
**实际路径是 `megatron/core/pipeline_parallel/fine_grained_activation_offload.py`，1610 行。**

`git ls-tree -r 60e039626 | grep -i offload` 的完整结果：

```
docs/user-guide/features/fine_grained_activation_offloading.md
docs/user-guide/features/optimizer_cpu_offload.md
megatron/core/optimizer/cpu_offloading/README.md
megatron/core/optimizer/cpu_offloading/__init__.py
megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py
megatron/core/pipeline_parallel/fine_grained_activation_offload.py
```

`hybrid_optimizer.py` 在 `optimizer/` 下、`fine_grained_*.py` 在 `pipeline_parallel/` 下 ——
这个「同名不同目录」本身就是本篇 offload 三档结构的一个论据：用它。

---

## 4. 骨架（照此写，但可细化）

### §1 显存六项账本

权重 / 梯度 / 优化器状态 / 激活 / 碎片 / 临时缓冲。

**碎片与临时缓冲这两项最容易被漏**，必须给出源码依据：

- `GlobalMemoryBuffer`（`megatron/core/utils.py`，01 篇已讲）—— 跨 iteration 复用，是「临时」项
- NCCL / CUDA allocator 的 caching behaviour
- 05 篇的 bucket padding 浪费（`pad_to_divisor` 的 64 / 128 / 2^16 三个常数）—— 直接引用 05 的结论，不要重算

### §2 重计算：189 行薄封装背后的判断逻辑

`recompute.py` 只有一个 `checkpointed_forward`，所以要讲：

- 判断逻辑实际散在哪些调用侧（`grep` 出来列全，给行号）
- `recompute_granularity` 的 `full` / `selective` 分支
- `recompute_method` 的 `uniform` / `block` 与 `recompute_num_layers` 的关系
- selective 为什么默认只重算 `core_attn`（回溯 arXiv:2205.05198；**注意该论文 abs 页无 CC 标识，图不可复制，只能自绘** —— 契约 §3.4）
- normal vs output-discarding 的二分（§2 的那张表）

### §3 `distribute_saved_activations`

把保存的激活摊到模型并行组上。**重点讲它与 TP / CP 的相互作用**：
开 TP 时语义会变（01/03 篇的 group 结构决定谁能拿到哪一片）。
这一节要反向引用 01 的组选择机制与 03 的 SP。

### §4 offload 三档

| 档 | 开关 | 落点 |
| --- | --- | --- |
| 全部 CPU offload | `--cpu-offloading` + `cpu_offloading_num_layers` | `cpu_offloading/` |
| fine-grained activation offload | `--fine-grained-activation-offloading` | `pipeline_parallel/fine_grained_activation_offload.py` 1610 行 |
| optimizer offload | `hybrid_optimizer.py` 477 行 | 见 §5 |

`transformer_config.py` L1862-1863 有一条硬校验：`cpu_offloading_num_layers < 0`
或 `>= num_layers` 会直接报错。这个约束要引。

### §5 `hybrid_optimizer.py`（契约 §2.1 指定的孤儿，必须独立成节）

477 行。核心是「哪些参数留在 GPU、哪些搬到 CPU」的判定 —— 优化器状态在 CPU 而 hot
parameter 在 GPU，分界怎么定、搬运会和 05 篇的哪个阶段撞。
**不许只在别篇提一句名字就算交差**（契约 §2.1 执行纪律）。

### §6 CUDA Graph 与 offload 的冲突

`transformer_config.py` L1334/1340/1346 直接链到了官方文档的
`#cuda-graph-integration` / `#tuning-parameters` / `#activation-offload-fraction` 锚点。
这两个特性**互相牵制**，是本篇最容易写出深度的一节。

### §7 引用关系网图

`graphLR`，中心是「显存账本六项」核心论断节点，四周挂支撑来源，边标「支撑」。
每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面。**不许出现无出处的图**。

---

## 5. 实验设计（本机**无 GPU**，只有 A / C 级）

证据分级见 `RESEARCH_PLAN.md` §1。**禁止 D 级**（凭印象的数字）。
推不出来又查不到出处的，明写「**待验证**」。

### A 级 · 显存账本帕累托前沿

脚本落点：**`ipynbs` 仓库的 `experiments/megatron/memory_ledger/`**（新建），
结构 `<name>.py` + `README.md` + `results/`。

- 扫 `recompute_granularity` × `recompute_method` × `recompute_num_layers`
  × `distribute_saved_activations` × offload 三档
- 纵轴显存、横轴**额外重算 FLOPs 占比**
- **代价轴不能用吞吐** —— 那是真机才能定的数，本机没 GPU。写清这个取舍

> ⚠️ `tools/megatron_bench/mem.py` 在本仓库里，**只读参考，不许扩写、不许删**
> （契约 §7 明确「存量暂时留在原地不许删也不许扩写」）。你的新脚本放 ipynbs，
> 需要的话读它的逻辑重写一份干净的。

### C 级 · 至少一项

候选（挑你验得动的）：

- selective recompute 的**数值等价性**：重算与不重算两条路径的输出逐位对照
- output-discarding 的语义：证明被 discard 的那部分激活确实不参与反向，
  且重算后结果与 normal 一致
- `distribute_saved_activations` 的形状变换：给定 `(hidden, tp_size)`，
  逐 rank 推出摊分后的张量形状（这一项与 04 篇的 `gtp_shard.py` 方法论同源，
  可以互相参照，但**不许抄它的代码**）

用 `~/Software/miniconda3/bin/python3`（managed python 3.13 无 numpy）。
`results/` 里固化输出，读者不必自己跑。

---

## 6. 禁区（另一个 tmux 窗口正在改这些，碰了就撞车）

**绝对不许动**：

- `_posts/2026-09-28-megatron-00-foundation.md` ← 窗口 2 在改
- `_posts/2026-10-08-megatron-02-expert-parallel.md` ← 窗口 2 在改
- `SERIES_SPEC.md` 的**编号表与模块 E 章节** ← 窗口 2 待批准后要改
  （§2.1 那一节是监工刚落的，**只读**，别再动）
- `_posts/2026-10-11-megatron-04-*.md`、`_posts/2026-10-10-megatron-05-*.md`
  ← 监工在验收，**这两篇只有你能反向引用，不能改**
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`、
  `_posts/2026-08-22-distill-rl-unified-spectrum.md`、
  `_posts/2026-09-30-rl-variables-in-frameworks.md`
- `tools/megatron_bench/` 全部（存量，不许扩写）
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，一条 git 写命令都不许
  （尤其 `pull` / `checkout`；行号取证一律 `git show 60e039626:<path>`）

**你只许写一个文件**：本篇正文。加上 `ipynbs` 仓库里 `experiments/megatron/memory_ledger/` 下的文件。

---

## 7. Git 纪律（契约 §6，第三次即终止写权限）

**不许 `git add` / `git commit` / `git push` / `git add -A` / `git add .`。**
写完文件 → 跑自查 → 报告监工 → 等指令。**就这三步。**

已记录两次越权（`7842859`、`ff675e6`），内容都合格但流程越权。
流程越权比内容出错更危险 —— 它意味着未验收的内容可以自己上线。

---

## 8. 写完的自查（`.agents/WRITING.md` §8 四步 + 契约 §5）

四步（`.agents/WRITING.md` §8）：

1. `python3 tools/check_post.py <file>`
2. `python3 tools/scan_garbled.py <file>`
3. `python3 tools/fix_quotes.py <file>` —— 必须是 0 转换
4. **回填一致性比对** —— 每个 ` ```text ` 块的数字与 ipynbs 脚本真实输出按数字字面量对齐
   （LaTeX 科学计数法与 Python e 记法先归一化；脚本改过必须重跑）

契约 §5 另加：

5. `python3 tools/check_mermaid.py <file>`
6. `~/Software/miniconda3/bin/python3 tools/audit_permalinks.py <file>`
7. **行号必须带文件名** —— grep 裸 `（L\d+`，逐个核。正文行号还要给 permalink：
   `https://github.com/NVIDIA/Megatron-LM/blob/60e039626/<path>#L<a>-L<b>`
8. **外部引用密度** —— 目标每 300 行 ≥ 8 条真实外部 URL

### 红线（`.agents/WRITING.md` §2/§4/§6 + 契约 §5 第 9 项）

- 无第一人称叙述生成过程（不写「我」「我们」讲怎么做的过程）
- 无硬件向内容（不围绕具体卡型/硬件展开）
- 没实验就不写实测数；没出处不写数字
- 不自造数学符号 —— 一律用来源文献的通行写法
- 公式内绝对值写 `\lvert o\rvert`，禁裸 `|`
- 行内公式 `$$...$$`，禁单 `$`
- 中文引号全角 `“ ”`
- 元叙述不入正文

**图**：默认一律自绘（契约 §3.4）。NVIDIA 官方 blog / docs 的图**版权保留，一律不许复制**，
需要就自绘并在图注写「据官方文档 X 重绘」。arXiv 图只在 abs 页**有 CC 标识**时才可引。

---

## 9. 交付

写完 → 跑完 §8 自查 → 报告监工：篇名、行数、图数、引用数、自查结果、
**以及你没能核实的部分**（不许猜、不许编，写「待验证」）。
然后**停下等指令**，不要 commit。
