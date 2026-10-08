# 工单 · agents:1（megatron-comm → TP/SP 线）

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**。
> 本工单只说你这一轮干什么。

---

## 本轮两件事，按顺序做

### 任务 A：修 01 篇的引用规范（退回重做，优先）

**背景**：你写的 `_posts/2026-10-08-megatron-01-process-groups-and-comm-overlap.md`（1401 行）
内容质量我已验收通过 —— 抽验 17 条行号全部命中，连上游源码自带的 typo
（`parallel_state.py:253` 的 `tiemout`、`:364` 的 `unmaksed`）都原样保留了，忠实度没问题。

**但引法违反红线**。`AGENTS.md` §8 明写：「引用源码必须锁版本 + commit，**行号给 permalink**」。
你用的是正文行号（如「`create_group()` L232–L266」），参考节只给文件级链接。两个问题：

1. **不可点击** —— 读者要手工跳。
2. **行号不带文件名** —— 本篇跨 6 个文件、137 个不同行号，读者看到「（L612）」无法判断是哪个文件。
   已核实两处：`ctx.tp_group = tp_group` 实际在 `layers.py:612`（不是 `parallel_state.py`），
   `def get_pg_rank` 在 `utils.py:650`。数字对，指向不明。

**要改成什么样**（照 `SERIES_SPEC.md` §3.1）：

- permalink 格式：`https://github.com/NVIDIA/Megatron-LM/blob/60e039626/<path>#L<a>-L<b>`
- **TL;DR 的每一条**（现在 10 条）末尾挂 permalink —— 这是读者第一眼看的，追溯性最重要
- **每个 `##` / `###` 小节的起始机制论断**挂 permalink，全篇**不少于 12 个**行内 permalink
  （1401 行 ÷ 150 ≈ 9，取 12 留余量）。02 篇是 99 个 / 1897 行，可作上限参照
- **所有正文残留的裸行号补上文件名**：写「（L612）」的地方改成
  「（`layers.py` L612）」或直接换成 permalink。全文 grep `（L\d+` 自查，别漏
- 参考节保留「源文件 + 行号范围」总账（已有，写法不用改），但**不能只有它**

**顺带补外部引用**（`SERIES_SPEC.md` §3.2）：现在全篇只有 2 条 arXiv。
目标 **每 300 行至少 8 条外部引用** → 本篇 **不少于 8 条**，且 TL;DR 每条论断至少 1 条出处。
可用的料：`mappings.py` 的 SP reduce-scatter/all-gather 对应 arXiv:2205.05198（Korthikanti）、
arXiv:2311.02382；NCCL UserBuffer 有官方文档页；`CUDA_DEVICE_MAX_CONNECTIONS` 的语义有 PyTorch 官方文档。
去 NVIDIA 官方文档站找**具体页面 URL**，不接受「据官方文档」这种不落 URL 的说法。

**再加 1 张引用关系网图**（`SERIES_SPEC.md` §3.3）：`graphLR`，中心是本篇核心论断
（如「进程组状态是模块级全局变量」「同序创建互不影响」「SP 不减通信量」），
四周挂支撑来源（论文 / 源码 permalink / 官方文档），边标「支撑」。节点文本含 `()[]{}` 要加引号。

**任务 A 的硬约束**：
- 不许改任何**行号**。数字你已经核过是对的，这次只改**引法**。
- 不许改技术内容、结论、TL;DR 的判断。
- 引文里的上游 typo **保持原样**，不要「顺手修好」。
- 改完跑五步自查（见下）。

### 任务 B：写 `megatron-03-tensor-parallel-and-sequence-parallel`

任务 A 验收通过后立刻开写。题纲在 `RESEARCH_PLAN.md` 的「### 03 · 张量并行与序列并行的真实实现」。

核心设问：论文里那对虚拟算子 `f` / `g`，在代码里是 autograd.Function 的哪一段？
SP 是怎么把 All-Reduce 换出 RS+AG 的？

必须覆盖（提纲已给，先自己核行号）：
- `ColumnParallelLinear` / `RowParallelLinear`
- `LinearWithGradAccumulationAndAsyncCommunication` —— 通信与 GEMM 重叠的真身
- `_linear_forward` / `_wgrad_gemm` / `LinearWithFrozenWeight`
- `VocabParallelEmbedding`
- `tensor_parallel/cross_entropy.py` —— vocab-parallel CE，显存杀手解法
- `mappings.py` 的 `_reduce_scatter_to_sequence_parallel_region` 等
- grad 累加为什么要 fuse（`--gradient-accumulation-fusion`），与 `wgrad_deferral` 的关系

**一条必须核正的历史事实**（`series-boundaries.md` 已钉死，别写反）：
**序列并行不降低 TP 通信量** —— AR = RS ∘ AG ⇒ 总量守恒，实测比值恒 1.00×。
SP 的收益在**激活显存**，提速来自可以关掉 activation recompute。
**不要把 SP 当降通信量的手段。**

按 `SERIES_SPEC.md` §3 全部引用要求、§4 证据分级（A/B/C，禁 D 级「大约快 X%」）。

**目标体量**：1200–1800 行。引用密度按篇长自查（≥1 permalink / 150 行；≥8 条外部引用）。
**建议配 C 级实验**：numpy 验三件事 —— ① f 列切 + All-Reduce ≡ 未切版本；
② SP 下 f 换 RS、g 换 AG 与原 AR 数值一致；③ gradient-accumulation-fusion 与朴素累加的精度差。
脚本落 `ipynbs` 的 `experiments/megatron/tensor_parallel/`（**不在本仓库 `tools/`**，见 `AGENTS.md` §5）。

---

## 读源码的正确打法（重要，已发生 2 次 30 分钟超时）

本系列目标文件都很大，**不要让 subagent 去「读懂」大文件**：

```bash
git grep -n 'def create_group' 60e039626 -- megatron/core/parallel_state.py
git show 60e039626:megatron/core/parallel_state.py | sed -n '232,266p'
git show 60e039626:<path> | wc -l
```

后台 `librarian` 只用于**外部资料检索**（论文、官方文档、许可），
不用来读本地源码。上次三个 librarian 同时 30 分钟无输出被杀 —— 拆小、或改同步短任务。

源码基准固定 **`60e039626`**（`megatron-core` 0.20.0），已确认本地 HEAD == origin/main，无需再 pull。

## Git 纪律

**不许自己 commit、不许 push。** 写文件、改文件，然后报我。
只碰 `_posts/2026-10-08-megatron-01-*.md` 和新建的 03。
**绝对不许** `git add -A` / `git add .`。
**不许动**：`_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`、
`_posts/2026-08-22-distill-rl-unified-spectrum.md`、`_posts/2026-09-30-rl-variables-in-frameworks.md`。

## 交付前自查五步

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io
F=_posts/2026-10-08-megatron-01-process-groups-and-comm-overlap.md
python3 tools/check_post.py $F
python3 tools/check_mermaid.py $F
python3 tools/scan_garbled.py $F     # 第 1/2/4 项必须干净；第 3 项是英文词表，人工过目
python3 tools/fix_quotes.py $F      # 期望 0 转换
~/Software/miniconda3/bin/python3 tools/audit_permalinks.py $F   # 我已装好，见下
```

`tools/audit_permalinks.py` 是监工写的 permalink 审计器（验文件在 commit 存在 + 行号在范围内）。
若它不存在，用 `/tmp/audit_pl.py` 的同款逻辑自检。

## 写完怎么报告我

1. 任务 A：改了哪些（TL;DR 几条挂了 permalink、全篇几个、行号带文件名自查结果、外部引用几条、网图 1 张）
2. 任务 B：起个草稿先报结构（章节骨架 + 打算挂哪些 permalink），再往下写
3. 有拿不准的事实**立刻问我**，不要靠猜写进正文
