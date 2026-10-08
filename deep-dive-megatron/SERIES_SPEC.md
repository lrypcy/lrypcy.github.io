# Megatron-LM 深度剖析系列 · 监工契约（Sisyphus  supervisory spec）

> 本文件是**两个执行 agent 与监工的共享契约**。agent 每次开工前必须重读本文件。
> 上位文件：`.agents/AGENTS.md`（工程约定）、`.agents/WRITING.md`（写作规范）。
> 本文件只补充「系列专属」的编号、边界与验收门禁。三者冲突时以 `.agents/` 为准。

---

## 1. 源码基准（不变）

- 仓库 `NVIDIA/Megatron-LM`，本地 `~/Desktop/Projects/github/Megatron-LM`
- 版本 **`megatron-core` 0.20.0**，commit **`60e039626`**（2026-08-21 main）
- **每篇开头必须标注 commit**。所有代码引用锁这个 commit。
- 行号必须**现取现核**：`git show 60e039626:<path> | sed -n 'A,Bp'`，不许凭记忆写。

### 读大文件的正确打法（P7 教训）

本系列目标文件都很大，**逐行读会超时**（已发生 2 次 30 分钟后台任务被杀）：

| 文件 | 行数 |
| --- | --- |
| `core/parallel_state.py` | 2692 |
| `core/transformer/cuda_graphs.py` | 3311 |
| `core/distributed/param_and_grad_buffer.py` | 1794 |
| `core/tensor_parallel/generalized_tensor_parallelism.py` | 2647 |
| `core/transformer/moe/token_dispatcher.py` | 2085 |

**正确做法**（不要把大文件丢给 subagent 去「读懂」）：

```bash
# 1. 先定位符号在哪一行
git grep -n 'def create_group' 60e039626 -- megatron/core/parallel_state.py

# 2. 再精确取段，不要整文件读
git show 60e039626:megatron/core/parallel_state.py | sed -n '232,266p'

# 3. 数行数
git show 60e039626:<path> | wc -l
```

后台 `explore`/`librarian` 只用于**外部资料检索**（论文、官方 blog、许可），
不用来读本地源码 —— 读源码一律用上面的 `git show` / `git grep`。

---

## 2. 编号表（2026-10-08 定稿 · 第四次编号）

| 编号 | 主题 | 主源码 | 状态 |
| --- | --- | --- | --- |
| 00 | 地基：代码地图 / 配置系统 / 一次迭代控制流 | `initialize.py` `training.py` `utils.py` | 待补引用后提交 |
| 01 | 进程组 + 通信与计算重叠 | `parallel_state.py` `mappings.py` `layers.py` | 初稿已落盘，待补引用 |
| 02 | 专家并行 | `moe/token_dispatcher.py` 等 | **已发布** |
| 03 | 张量并行与序列并行（TP + SP） | `tensor_parallel/layers.py` `mappings.py` `cross_entropy.py` | 待写 |
| 04 | GTP：新一代张量并行 | `generalized_tensor_parallelism.py` `gtp_api.py` | 待写 |
| 05 | 数据并行与优化器（三套实现） | `param_and_grad_buffer.py` `distrib_optimizer.py` `fsdp/` | 待写 |
| 06 | 流水并行调度器 | `pipeline_parallel/schedules.py` | 待写 |
| 07 | 上下文并行与长上下文 | `dot_product_attention.py` `multi_latent_attention.py` `packed_seq_params.py` | 待写 |
| 08 | 显存账本：重计算 / 卸载 / 分布保存 | `recompute.py` `transformer_config.py` `fine_grained_activation_offload.py` | 待写 |
| 09 | 低精度：FP8 全链路与 FP4 | `fp8_utils.py` `fp4_utils.py` `extensions/transformer_engine.py` | 待写 |
| 10 | 算子融合与 CUDA Graph | `fusions/` `full_cuda_graph.py` `transformer/cuda_graphs.py` | 待写 |
| 11 | RL 训练栈与 Refit | `megatron/rl/` `core/resharding/` | 待写 |
| 12 | Checkpoint 与容错 | `dist_checkpointing/` `rerun_state_machine.py` | 待写 |
| 13 | Bridge、导出与模型家族落地 | Megatron Bridge `export/` modelopt | 待写 |

**编号变更史**（不要再改，再改要在本文件追加一条）：

1. 原 15 篇 → 模块 C（实测调优 3 篇）外迁至 `deep-dive-profiling/`
2. 原 `rl-refit-bridge` 由 03 顺延为 04
3. EP 从模块 B 提前独立成篇，占 02
4. **第四次（2026-10-08）**：原定「01 = 进程组/TP+SP/GTP/DP/PP/CP」六合一**拆开**，
   按主题各自独立成篇（TP+SP→03、GTP→04、DP→05、PP→06、CP→07）。
   **通信重叠**不再单列，并入 01。理由与 EP 破先例时完全一致：
   EP 单主题 1897 行已超「一个模块一篇」承载上限；六主题合并至少 6000 行，不可读也不可审。

---

## 3. 引用规范（本次新增的硬要求）

用户明确要求：**丰富的网络引用 + 图，注明引用何处；代码引用可追溯。**
以下三条为**硬门禁**，不满足即退回。

### 3.1 代码引用：行号必须给 permalink

`AGENTS.md` §8 红线原文：「引用源码必须锁版本 + commit，**行号给 permalink**」。

- ✅ 每个论断的行内 permalink：`https://github.com/NVIDIA/Megatron-LM/blob/60e039626/<path>#L<a>-L<b>`
- ✅ 正文行号**必须带文件名**。`（L612）` 是不可接受的 —— 本系列跨 6+ 文件、100+ 行号，
  读者无法判断 L612 属于哪个文件。写成 `（`layers.py` L612）` 或直接给 permalink。
- ❌ 只在参考节列文件级链接（那等于没有行号锚点）
- 参考节仍需列出本篇引用的全部源文件 + 行号范围，作为总账。

**目标密度**：每 150 行至少 1 个行内 permalink（02 篇是 99 个 / 1897 行，可作基准）。

### 3.2 外部引用：论文 / 官方文档 / 官方 blog

- 每条重要论断（尤其 TL;DR 的每一条）**至少挂 1 条外部或源码出处**
- **目标密度**：每 300 行至少 8 条外部引用
- arXiv 用 `https://arxiv.org/abs/<id>`；官方文档给具体页面而非仓库根；
  官方 blog 给具体文章 URL
- ❌ 禁止「据官方文档」这种不落 URL 的说法
- 禁止把二手贴的转述当出处，要追到原论文/原仓库

### 3.3 图：三种都要

| 类型 | 要求 |
| --- | --- |
| **自绘图** | 主体。Mermaid 只用 `<br>` 换行；节点文本含 `()[]{}` 需加引号；纯中文标签。**每张图下方注明数据/结论出处**（行内 permalink 或论文链接）。 |
| **现成图** | **只在许可明确时才引**。arXiv 论文配 CC BY 许可的图可引，必须标 `（图源：arXiv:<id>，CC BY）`。NVIDIA 官方 blog / 文档站图片默认**版权保留**，一律不引 —— 改用自绘复现，并在图注写明「据官方文档重绘」。 |
| **引用关系网图** | 每篇正文配 1 张 `graphLR`：中心是本篇核心论断节点，四周挂支撑来源（论文 / 源码 permalink / 官方文档），边标「支撑」。系列级总网图放 00 篇。 |

**禁止**：来历不明的图；自己无法复现却标「官方图」的图。

### 3.4 引用许可白名单（2026-10-08 核定，有原文依据）

结论：**默认一律自绘。** 现成图只在许可明确时才引。

#### arXiv 论文图 —— 逐篇看 abs 页有没有 CC 标识

arXiv 官方「Permissions and Reuse」页（<https://info.arxiv.org/help/license/reuse.html>）的原文：

> **Can I reuse figures from an arXiv paper?**
> The short answer is "it depends". More specifically:
> - If the license applied to the work allows for remixing or reuse with citation, then yes.
> - If not, then the version is assigned one of the arXiv perpetual non-exclusive licenses,
>   and you will need to contact the submitter or copyright holder (if published) to determine
>   applicable permissions.

> All e-prints submitted to arXiv are subject to copyright protections.
> **arXiv is not the copyright holder on any of the e-prints in our corpus.**
> The overwhelming majority of e-prints are submitted using the arXiv perpetual non-exclusive
> license, **which does not grant further reuse permissions directly.**

**判定方法**（同页给出）：所有 abs 页在 "Download:" 选项下方标注该版本 assigned license；
若提交者选了 Creative Commons，页面会出现 **CC 标识**。

| abs 页状态 | 能否复制该论文的图 |
| --- | --- |
| 有 CC 标识（CC BY 4.0 / CC BY-SA 4.0） | ✅ 可，须署名 + 标注许可 |
| 只有 "view license"（默认 arXiv 永久非独占许可） | ❌ **不可**，须自绘 |
| 论文正文自带明确授权语句 | ✅ 按该语句条款 |

- 「CC BY 4.0 允许 reuse、remix、adapt、build upon，commercial use 亦可，只要署名」
  （<https://info.arxiv.org/help/license/index.html>）
- CC BY-SA 额外要求：改作须同许可发布。
- **本系列已核**：arXiv:2205.05198（Korthikanti，01/03 两篇都在引）的 abs 页**只有
  "view license"、无 CC 标识** → 走默认许可 → **其图不可复制，只能自绘**。
  引用其余论文图前一律先看 abs 页有没有 CC 标识，不许凭印象。

#### 论文正文自带授权（少数，逐篇确认）

存在但少见，引用前必须在该论文正文里找到原句。例如 *Attention Is All You Paper*（arXiv:1706.03762）：

> Provided proper attribution is provided, Google hereby grants permission to reproduce the
> tables and figures in this paper solely for use in journalistic or scholarly works.

这种「按论文逐条给授权」的情况才可复制，且必须照抄其限定条件。

#### NVIDIA 官方 blog / docs / 仓库文档 —— 一律不可复制

NVIDIA 法务页（<https://www.nvidia.com/legal-info>）原文：

> **OWNERSHIP OF MATERIALS** Materials are copyrighted and are protected by worldwide copyright
> laws and treaty provisions. **They may not be copied, reproduced, modified, published, uploaded,
> posted, transmitted, or distributed in any way, without NVIDIA's prior written permission.**

CUDA SDK License 更明确把图列入受版权标的（<https://developer.nvidia.com/downloads/license/nsclv1>）：

> **Intellectual Property Ownership:** All rights, title, interest, and copyrights in and to the
> Materials (**including but not limited to all images, photographs, animations, video, audio,
> music, text, and other information** incorporated into the Materials), are owned by NVIDIA, or its suppliers.

NVIDIA Technology Access Terms §5(d) 亦禁止把 NVIDIA Content 「copy, reproduce, publish, blog,
disclose, transmit, or otherwise disseminate elsewhere」。

**结论：NVIDIA 的一切图（blog 配图、docs 截图、README 图、架构示意图）默认版权保留，
一律不复制。** 需要时自绘并在图注写「据官方文档 X 重绘」，链接到原页面。

#### 图注模板

| 情形 | 图注写法 |
| --- | --- |
| 自绘、结论来自源码 | 数据/结论来源：<permalink>（或 `文件 L<a>-L<b>`） |
| 自绘、据官方文档重绘 | 据 <官方文档 URL> 重绘 |
| 自绘、据 CC BY 论文重绘 | 据 arXiv:<id>（CC BY）重绘 |
| 直接复制 CC BY 论文图 | 图源：arXiv:<id>，CC BY 4.0 —— **必须逐篇确认 CC 标识** |
| 直接复制 NVIDIA 图 | **禁止** |

#### 落地要求

- 用户要求「丰富的网络引用图」→ **用「图多 + 每张图都注明出处」来满足，不是靠搬别人的图**。
  本系列的价值在源码级精度，自绘的示意图比官方截图更贴合作者的代码行号与推导。
- 每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面，**不许出现无出处的图**。

---

## 4. 证据分级（沿用 RESEARCH_PLAN.md §1，不得自创）

| 级 | 来源 | 要求 |
| --- | --- | --- |
| **A** | `tools/megatron_bench/` 解析脚本实跑 | 脚本随文可复算 |
| **B** | NVIDIA 官方 blog / 论文 / repo 内 test 与 benchmark | 挂链接，注明卡型与规模 |
| **C** | CPU 微型复现（numpy / CPU torch） | 验数值等价或算法正确性，不假装端到端性能 |

**禁止 D 级**：凭印象的数字（"大约快 30%"）。推不出来又查不到出处的，明写「**待验证**」。

**本机无 GPU** —— 不存在真机 profiling 这条路。吞吐 / MFU 只能引官方数据并写清卡型规模。

---

## 5. 验收门禁（监工执行，agent 自查不算数）

每篇交付后由监工逐项跑，任一不过即退回：

1. `python3 tools/check_post.py <file>` —— 公式 / 表格 / `<em>` 全过
2. `python3 tools/check_mermaid.py <file>` —— 纯中文标签、无 LaTeX、无危险字符
3. `python3 tools/scan_garbled.py <file>` —— 第 1/2/4 项干净；**第 3 项监工人工过目**
4. `python3 tools/fix_quotes.py <file>` —— 0 转换
5. **`~/Software/miniconda3/bin/python3 tools/audit_permalinks.py <file>`**
   —— 每个 permalink 的文件在 commit 存在 + 行号在文件范围内（本系列自研工具）
6. **行号带文件名** —— grep 出裸 `（L\d+` 逐个核
7. **外部引用密度** —— 按 §3.2 统计，低于阈值退回
8. `node tools/kg_layout_check.mjs` —— 仅当动了 `_data/knowledge_graph.yml`
9. **红线自检**（监工人工）：
   - 无第一人称叙述生成过程（不写「我」「我们」讲怎么做的过程）
   - 无硬件向内容（不围绕具体卡型/硬件展开）
   - 没实验就不写实测数
   - 没出处不写数字
   - 不自造数学符号（一律用来源文献的通行写法）
   - 公式内绝对值写 `\lvert o\rvert`，禁裸 `|`
   - 行内公式 `$$...$$`，禁单 `$`
   - 中文引号全角 `“ ”`

### 引文忠实性红线

引用源码注释里的 typo（如上游 `parallel_state.py:253` 的 `tiemout`、`:364` 的 `unmaksed`）
**必须原样保留**。这是逐字忠实，不是错误 —— 改掉才是伪造引文。
`scan_garbled` 会在第 3 项报出这类词，监工核到源码确认即可放行。

---

## 6. Git 纪律（两个 agent 共用一个工作区）

**两个 agent 不许自己 commit、不许自己 push。** 写文件即可。监工验收通过后统一提交并立即推送。

> **违规记录**
>
> 1. `7842859`（2026-10-08 08:08）——01 篇。收到禁令后仍自建并推送。
> 2. `ff675e6`（2026-10-08 08:51）——03 篇。**在监工完成验收之前**就推送。
>    监工当时正在核第三处引文（`layers.py` L1503 的 assert），推送发生在核完前 2 分钟。
>
> 两次都是「内容合格、流程越权」：事后核验都通过，但**流程越权比内容出错更危险** ——
> 它意味着未验收的内容可以自己上线，而验收门禁正是本契约存在的唯一理由。
>
> **第三次即终止该 agent 的写权限**，由监工接管其全部文件写入。
>
> 这是明确禁令，不是建议。写完文件 → 报告监工 → 等指令，就这三步。

如果 agent 需要暂存，**只允许按路径精确 stage 自己的那一个文件**：

```bash
git add _posts/<自己的文件名>.md      # ✅
git add -A                            # ❌ 绝对禁止
git add .                             # ❌
git commit / git push                 # ❌ agent 不许做，等监工指令
```

两个 agent 共用工作区，已知会被别人改到的文件（**不许动**）：
`_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`、
`_posts/2026-08-22-distill-rl-unified-spectrum.md`、`_posts/2026-09-30-rl-variables-in-frameworks.md`。

这些是监工与其他工作线留下的未提交改动，agent 绕开。

---

## 7. 代码放哪（AGENTS.md §5 红线）

- 新写的随文实验脚本 → **`ipynbs` 仓库** 的 `experiments/megatron/<topic>/`，
  结构 `<name>.py|.ipynb` + `README.md` + `results/`；README 写清对应文章（带链接）+ 如何运行 + 结论
- 改完 push 到 GitHub，正文引用 **permalink 锁 commit**
- 本仓库 `tools/` 只剩校验脚本 + 访问统计后端；`tools/megatron_bench/` 是**存量**，
  已两处被正文引用（00 篇），**暂时留在原地不许删也不许扩写** ——
  但 00 篇必须补上它的可访问路径，否则读者够不到（见下）

### 当前唯一断链（P3，待修）

`00-foundation` 两处引用 `tools/megatron_bench/flops.py`（L207、L569），
但该脚本 (a) 未被 git 跟踪，(b) 所在 `tools/` 在 `_config.yml` 的 `exclude` 里。
**读者在站点和 GitHub 上都取不到。** 修法：把脚本迁到 `ipynbs`、push、
正文换成锁 commit 的 permalink。

---

## 8. 交付节奏

- 两篇并行，主题不重叠（见 §2 表），避免同文件互相踩
- 每篇写完 → agent 自查五步 → 报监工 → 监工跑 §5 全部门禁 + 人工红线
- 不合格退回，**agent 不得自行 commit 绕过验收**
- **每验收通过一篇，监工立刻 commit & push 到 `origin/master`，不攒到最后**
  （用户 2026-10-08 明确要求：每完成一个就及时提交）
- commit 只 stage 该篇正文 + 本契约/计划的改动，**不带其他工作线的未提交文件**
- 全部 13 篇推进完、且每篇都过了 §5，才算目标达成
