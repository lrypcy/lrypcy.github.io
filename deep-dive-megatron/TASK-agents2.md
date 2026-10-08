# 工单 · agents:2（megatron-ep → DP/优化器线）

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**。
> 本工单只说你这一轮干什么。

---

## 先回答你的两个提问

**Q：要不要把 `_data/knowledge_graph.yml` 单独提一个 commit？**
要，但**先别动** —— 监工统一提交。而且**内容要改，不只是提 commit**：
yml 上的 desc 还写着「5 篇…03 实测调优 → 04 RL/refit 桥接」，但 03 实测调优早已外迁到 profiling 系列，
现在实际编号是 02=EP、03=优化工具箱、04=rl-refit。已发布页面上这句描述是错的，要改到与实际一致。

**新编号（监工 2026-10-08 第四次定稿，六合一已拆开）：**

| 编号 | 主题 | 状态 |
| --- | --- | --- |
| 00 | 地基 | 待修引用后提交 |
| 01 | 进程组 + 通信与计算重叠 | 初稿已落盘（另一个 agent 在修） |
| 02 | 专家并行 | **已发布**（你的） |
| 03 | 张量并行与序列并行 | agents:1 在写 |
| 04 | GTP | 待写 |
| **05** | **数据并行与优化器（三套实现）** | **← 你这篇** |
| 06 | 流水并行调度器 | 待写 |
| 07 | 上下文并行与长上下文 | 待写 |
| 08 | 显存账本 | 待写 |
| 09 | 低精度 FP8/FP4 | 待写 |
| 10 | 融合算子与 CUDA Graph | 待写 |
| 11 | RL 栈与 Refit | 待写 |
| 12 | Checkpoint 与容错 | 待写 |
| 13 | Bridge / 导出 | 待写 |

原定「01 = 进程组/TP+SP/GTP/DP/PP/CP」六合一**按主题拆开**，理由与你把 EP 独立成 02 完全一致
（EP 单主题 1897 行已超「一个模块一篇」的承载上限，六主题合一起码 6000 行）。
**你 EP 破先例的做法现在是本系列的既定规范。**

**Q：Pages 构建本地看不到？**
对。本地只能过静态门禁，实际渲染以线上构建为准，这条不用管。

---

## 本轮两件事，按顺序做

### 任务 A：修 00 篇（**当前线上有 404，这是最高优先级**）

**问题比你报告的更严重**：你已发布的 02 篇里有 **3 处**链接指向 00 地基
（L13 系列导航、L1893 参考、L1897 上一篇），而 00 从未进入任何 commit。
**也就是说线上现在点「00 地基」是 404。** 已发布内容的活链，必须先修。

文件：`_posts/2026-09-28-megatron-00-foundation.md`（570 行，从未提交）

三件事：

**(1) 补 permalink（当前 0 个）**
570 行的「源码层」长文，一个行内 permalink 都没有，只有参考节列了个仓库根目录。
对比你的 02 篇 99 个。目标是每 150 行至少 1 个，全篇**不少于 8 个**，
优先给：代码地图（各文件规模怎么核的）、`TransformerConfig` / `ModelParallelConfig` 两个 dataclass 的字段出处、
参数量自检与官方公式对照、`pretrain()` 的关键节点、`no_pipelining` 那个循环、训练日志字段产出点。

**(2) 修断链（`SERIES_SPEC.md` §7）**
00 篇 L207 和 L569 两处引用 `tools/megatron_bench/flops.py`，但：
- 该脚本**未被 git 跟踪**
- 所在 `tools/` 在 `_config.yml` 的 `exclude` 里

**读者在站点和 GitHub 上都取不到。** `AGENTS.md` §5 的规矩是：新代码进 `ipynbs` 仓库，
正文用**锁 commit 的 permalink** 引用。所以：

1. 把 `flops.py`（以及它依赖的 `configs.py` / `comm.py` / `mem.py`，`tools/megatron_bench/` 共 6 个 .py）
   迁到 `~/Desktop/Projects/sandbox/ipynbs` 的 `experiments/megatron/foundation/`
2. 补 `README.md`：写清**对应文章（带链接）+ 如何运行 + 结论**，以及它验证了本文哪个数字
3. **在 ipynbs 仓库 commit 并 push**（`origin` = `git@github.com:lrypcy/ipynbs.git`，分支 `main`）。
   push 前把仓库根目录已有的 `?? .omo/ .workbuddy/ experiments/.omo/` **排除掉**，只提交
   `experiments/megatron/` 下你新建的文件
4. 拿到 commit SHA 后，把 00 篇里那两处引用换成锁 commit 的 permalink
5. 校验它可解析：`git -C ~/Desktop/Projects/sandbox/ipynbs log -1 --format=%H`

`tools/megatron_bench/` 留在原地**不许删**（暂时还有引用），但不许扩写。

**(3) 监工已核出的硬错误，必须改（已逐条核过 `training.py`，共 5606 行）**

第 220 行写「`num_floating_point_operations()` 的 dense 分支（**L862-1007**）」——**错了**：

```
L802   def num_floating_point_operations(     ← 函数在这里开始
L862   def attn_layer_flops(                  ← 这里只是它内部的一个 helper
L1007  return flops_fwd * 3                  ← 这条是对的
```

正确写法是函数起于 **L802**。L1007 那条 `return flops_fwd * 3` 经核**准确**，不用动。
`.workbuddy/memory/topics/series-boundaries.md` 里记的就是 `training.py:802`，
是文章写错了、记忆没错 —— 不要去「修正」记忆。

我另外抽验的这些**全部准确**，别去动：
`pretrain()` L1500 ✓、`setup_model_and_optimizer` L1820 ✓、`update_num_microbatches` L2159 ✓、
`get_model` L2351 ✓、L2710 的 `return get_model(...)` ✓、
`forward_backward_no_pipelining` L723（在 **`schedules.py`**，不在 `training.py`）✓。

**(4) 跑五步自查**（见下）

任务 A 完成后**先报我**，我验收 + 立刻 commit & push（会一并解掉线上 404）。

### 任务 B：写 `megatron-05-data-parallel-and-optimizers`

任务 A 验收通过后立刻开写。题纲在 `RESEARCH_PLAN.md` 的「### 05 · 数据并行与优化器：三套实现的分野」。

核心设问：DDP / Megatron-FSDP / FSDP2 在代码的哪一层分叉？
`param_and_grad_buffer` 里的 bucket 到底怎么切？

这是本系列**最硬的一块**（`param_and_grad_buffer.py` 1794 行）。必须覆盖（提纲已给，先自己核行号）：
- `param_and_grad_buffer.py` 的 bucket padding 与对齐（`--ddp-pad-buckets-for-high-nccl-busbw`）
- `finalize_model_grads.py` 的三种重叠组合：`overlap_grad_reduce` / `overlap_param_gather` /
  `overlap_param_gather_with_optimizer_step`
- `DistributedOptimizer` 的分片布局 vs ZeRO-1 —— **为什么不完全是一回事**，`param_layout.py` 是干什么的
- 三条 DP 路径（DDP / Megatron-FSDP / FSDP2）取舍表：显存 / 通信量 / 支持面 / 坑
- `--ddp-reduce-scatter-with-fp32-accumulation` / `reduce_scatter_with_fp32_accumulation.py`
- `muon.py` + `emerging_optimizers.py`（2026/04 新增）+ `qk_clip.py` 怎么接入

**建议配 A 级解析脚本 + C 级实验**（都在 `ipynbs`，不进本仓库）：
- A 级：`mem.py` 算三条路径的显存六项账本，出对照表（含 bucket 粒度带来的 tail 浪费）
- C 级：numpy 验 reduce-scatter + 分片优化器更新与非分片 AdamW 的**数学等价** ——
  这正是那些 overlap 开关能安全打开的前提，是本篇最该有的实验

**一条别写错的**：`core/timers.py:109` 的 `Timer` 用 `time.time()` + `torch.cuda.synchronize()`
（不是 CUDA Event）→ **测量动作本身就毁掉重叠**。别把内置 timer 的读数当重叠收益的证据。

按 `SERIES_SPEC.md` §3 全部引用要求、§4 证据分级（A/B/C，**禁 D 级**「大约快 X%」——
推不出来又查不到出处的，明写「**待验证**」）。本机无 GPU，吞吐/MFU 只能引官方数据并写清卡型与规模。

**目标体量**：1400–2000 行（这块最重）。
**建议配 1 张引用关系网图**：中心是「三套 DP 实现的分叉点」这个论断，
四周挂 `param_and_grad_buffer` permalink、`distrib_optimizer.py`、官方 FSDP/ZeRO 文档、相关论文，边标「支撑」。

---

## 读源码的正确打法（重要，已发生 2 次 30 分钟超时）

`param_and_grad_buffer.py` 1794 行，**不要让 subagent 去「读懂」大文件**：

```bash
git grep -n 'class ParamAndGradBuffer' 60e039626 -- megatron/core/distributed/param_and_grad_buffer.py
git show 60e039626:megatron/core/distributed/param_and_grad_buffer.py | sed -n 'A,Bp'
git show 60e039626:<path> | wc -l
```

后台 `librarian` 只用于**外部资料检索**（论文、官方文档、许可），不用来读本地源码。
上次三个 librarian 同时 30 分钟无输出被杀 —— 拆小、或改同步短任务。

源码基准固定 **`60e039626`**（`megatron-core` 0.20.0），已确认本地 HEAD == origin/main，无需再 pull。

## Git 纪律

**本仓库不许自己 commit、不许 push。**（唯一例外：任务 A 步骤 3 要你在 **ipynbs 仓库** push，
那是另一个仓库，不受此限。）

本仓库里只碰 `_posts/2026-09-28-megatron-00-foundation.md` 和新建的 05。
**绝对不许** `git add -A` / `git add .`。
**不许动**：`_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`（我统一改）、
`_posts/2026-08-22-distill-rl-unified-spectrum.md`、`_posts/2026-09-30-rl-variables-in-frameworks.md`。

## 交付前自查五步

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io
F=_posts/2026-09-28-megatron-00-foundation.md
python3 tools/check_post.py $F
python3 tools/check_mermaid.py $F
python3 tools/scan_garbled.py $F     # 第 1/2/4 项必须干净；第 3 项是英文词表，人工过目
python3 tools/fix_quotes.py $F      # 期望 0 转换
~/Software/miniconda3/bin/python3 tools/audit_permalinks.py $F   # 监工已装好，见下
```

注意：`scan_garbled.py` 第 3 项报的英文词里，**引文自带的 typo 不算错**。
引文忠实性是红线 —— 上游 `parallel_state.py:253` 就是 `tiemout`、`:364` 就是 `unmaksed`，
原样保留才是对的，改掉才是伪造引文。核到源码确认即可。

`tools/audit_permalinks.py` 是监工写的 permalink 审计器（验文件在 commit 存在 + 行号在范围内）。

## 写完怎么报告我

1. 任务 A：permalink 补了几个、外部引用几条、`megatron-00-foundation` 在 ipynbs 的 permalink（含 commit SHA）、线上 404 是否已解
2. 任务 B：起个草稿先报结构（章节骨架 + 打算挂哪些 permalink），再往下写
3. 有拿不准的事实**立刻问我**，不要靠猜写进正文
