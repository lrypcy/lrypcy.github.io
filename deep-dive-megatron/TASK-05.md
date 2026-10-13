# 工单 · 05 篇退回修复（两项硬缺陷）

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**。
> 本工单只说你这一轮干什么。

---

## 0. 背景

`2026-10-10-megatron-05-data-parallel-and-optimizers.md`（1422 行）已写完，验收**退回两项**。
两项都是契约硬要求，不是风格偏好。

参考对照：**同系列的 04 篇 7/7 张图都有图注、02 篇 4/4 张都有**。05 是唯一的例外。

---

## 缺陷一：12 张 mermaid 全部无图注（违反契约 §3.3）

§3.3 原文：「**每张图下方注明数据/结论出处**（行内 permalink 或论文链接）」
以及：「每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面，**不许出现无出处的图**」

现状：12 个 mermaid 块后面接的都是**正文散文**，不是图注。例如

- 图1（L64）后面接「这个继承结构带来一个重要后果：……」—— 这是 §1.2 的正文
- 图3（L318）后面接「三处不同：」—— 这是 §3.2 的引导句

### 逐张位置与所属章节（监工已核）

| 图 | 起始行 | 所属章节 | 该图在论证什么 |
| --- | --- | --- | --- |
| 1 | L64 | §1.2 继承链：三者共享同一个抽象基类 | 三个类的继承关系与 `param_and_grad_buffer` 的服务边界 |
| 2 | L147 | §2.3 缓冲区类型与 NVFP4 分支 | param / grad buffer 的双索引结构 |
| 3 | L318 | §3.2 DistOpt 路径的装箱：三处不同 | 默认路 vs DistOpt 路装箱逻辑差异 |
| 4 | L391 | §3（bucket 到底怎么切） | 桶到 rank 的分片：一个容易搞混的细节 |
| 5 | L538 | §5.3 梯度同步：本文最重要的一段代码 | 梯度同步的三段式流程 |
| 6 | L703 | §6 三种 overlap 组合的精确分派 | 三种组合的分派路径 |
| 7 | L857 | §7 `DistributedOptimizer`：分片布局怎么算出来的 | buffer → rank 按 dtype 分组 |
| 8 | L968 | §8.1 三个数据类 | `param_layout.py` 的三个数据类与两层布局 |
| 9 | L992 | §8.2 为什么不是 ZeRO-1 | 分片布局与 ZeRO-1 的差异 |
| 10 | L1077 | §9.1 它是梯度归约的总调度 | `finalize_model_grads` 的分派查找表 |
| 11 | L1187 | §11.2 关键：为什么先 all-to-all | reduce-scatter + fp32 累加的通信顺序 |
| 12 | L1322 | §13 引用关系网 | 引用关系网图（见下方特殊要求） |

### 图注写法（照 04 篇的既有形式，**不要自创格式**）

04 篇的图注长这样，用 `*图 N：…*` 斜体行，紧跟在 ``` 之后：

```
*图 1：两种切法的差别在「权重是否被再次切分、是否需要聚回来」。数据出处：GTP 设计文档的
TP / GTP_remat 对照表与 [`initialize_model_parallel` L601-L700](permalink)。
```

三种可接受的出处写法，至少用一种：

1. `数据出处：` + Megatron 源码行内 permalink
   `https://github.com/NVIDIA/Megatron-LM/blob/60e039626/<path>#L<a>-L<b>`
2. `数据出处：` + 官方文档具体页面
3. `据 <官方文档 URL> 重绘`

**每张图至少一个 permalink 或具体 URL，不许写「数据出处见参考节」这种空话。**

### 图 12（引用关系网图）的特殊要求

它是 `graphLR` 引用关系网，中心是核心论断、四周挂支撑来源、边标「支撑」。
图注要说明**中心论断是哪几条、支撑来源分别落在哪些章节**，但**不要求每条边都给 permalink**
—— 关系网图本身��出处载体，permalink 在各图注与参考节里。

对照 02 篇的图4 写法：
`*图 4：四条核心论断与支撑来源，分别落在 §2、§7、§6 与 §8、§11。完整 permalink 见各图注与参考节。*`

### 行号取证（禁止凭记忆写）

```bash
M=~/Desktop/Projects/github/Megatron-LM
git -C $M log -1 --format=%H          # 必须仍是 60e039626，不许 pull / checkout
git -C $M grep -n 'def <符号>' 60e039626 -- megatron/core/distributed/param_and_grad_buffer.py
git -C $M show 60e039626:megatron/core/distributed/param_and_grad_buffer.py | sed -n 'A,Bp'
```

写完图注后跑一次 `~/Software/miniconda3/bin/python3 tools/audit_permalinks.py` 确认新加的 permalink 全部可解析。

---

## 缺陷二：外部引用密度差一个数量级（违反契约 §3.2）

§3.2 目标：**每 300 行至少 8 条外部引用**。05 有 1422 行 → **需 ≥ 38 条**。

现状：真实外部引用**只有 4 条**：

```
arXiv:1910.02054                      ZeRO（Rajbhandari et al.）
arXiv:2104.04473                      Megatron（Narayanan et al.）
pytorch.org/docs/stable/fsdp.html      PyTorch FSDP 文档
pypi.org/project/emerging-optimizers/  emerging-optimizers 包
```

> **注意**：参考节里另外 19 条 URL 全是**你自己 ipynbs 仓库的脚本 permalink**
> （`github.com/lrypcy/ipynbs/...`）。那是 **A 级证据**，不是外部引用。
> §3.2 说的外部引用是「论文 / 官方文档 / 官方 blog」。
> 这 19 条已经占着参考节的位置，但它们不计入密度。

两种计数口径都 FAIL，所以没有解释空间：

| 口径 | 实数 | 门槛 | 结果 |
| --- | --- | --- | --- |
| 严格（只算论文/文档/blog） | 4 | 38 | ❌ |
| 宽松（连 ipynbs 自链也算） | 23 | 38 | ❌ |

### 要补的量：≥ 34 条，且必须**逐条对应具体论断**

**绝不允许为凑数堆 URL。** §3.2 原文：「每条重要论断（尤其 TL;DR 的每一条）**至少挂 1 条外部或源码出处**」，
以及「禁止『据官方文档』这种不落 URL 的说法」「禁止把二手贴的转述当出处，要追到原论文/原仓库」。

监工已经准备好一份**候选引用清单**（见同目录 `TASK-05-refs.md`），
按 05 的实际章节逐条给出可用来源。**你必须逐条核到原文再写进正文** ——
清单里给的是起点，不是可以直接抄的成品。

### 图注里也可以带外部出处

12 张图注本身也要出处，其中若干张用外部来源比用源码 permalink 更合适
（例如 §8.2 讲「为什么不是 ZeRO-1」的那张图，出处就是 ZeRO 论文）。
**图注的出处可以同时计入 §3.2 的外部引用密度**，但必须是真实可核的 URL。

---

## 禁区（其他 tmux 窗口正在改这些）

**绝对不许动**：

- `_posts/2026-09-28-megatron-00-foundation.md` ← 窗口 2 在改
- `_posts/2026-10-08-megatron-02-expert-parallel.md` ← 窗口 2 在改
- `SERIES_SPEC.md` ← 窗口 2 待批准后要改编号表（§2.1 是监工落的，只读）
- `_posts/2026-10-12-megatron-08-memory-ledger.md` ← 窗口 5 正在写
- `deep-dive-megatron/TASK-08.md` ← 同上
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`、
  `_posts/2026-08-22-distill-rl-unified-spectrum.md`、
  `_posts/2026-09-30-rl-variables-in-frameworks.md`
- `tools/megatron_bench/` 全部（存量，不许扩写）
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，一条 git 写命令都不许

**你只许写一个文件**：`_posts/2026-10-10-megatron-05-data-parallel-and-optimizers.md`。

---

## Git 纪律（契约 §6，第三次即终止写权限）

**不许 `git add` / `git commit` / `git push` / `git add -A` / `git add .`。**
写完 → 跑自查 → 报告监工 → 等指令。已记录两次越权（`7842859`、`ff675e6`）。

---

## 写完的自查

四步（`.agents/WRITING.md` §8）：

1. `python3 tools/check_post.py <file>`
2. `python3 tools/scan_garbled.py <file>`
3. `python3 tools/fix_quotes.py <file>` —— 必须 0 转换
4. **回填一致性比对** —— 正文数字与 ipynbs `experiments/megatron/data_parallel/` 的真实输出对齐

契约 §5 另加：

5. `python3 tools/check_mermaid.py <file>`
6. `~/Software/miniconda3/bin/python3 tools/audit_permalinks.py <file>`
7. grep 裸 `（L\d+` —— 必须 0
8. **外部引用密度 ≥ 38 条**（严格口径）
9. **图注 12/12，且每张带出处**

### 监工验收用的图注判据

图注 = 紧跟 mermaid 结束围栏之后的非空行，且满足：

- 以 `> ` 或 `*图 N：` / `* *图 N：` 开头
- 含「出处」二字，**或**含 `http`，**或**含 `L\d+` 行号锚点

图 12 例外：只需含「图 12」编号 + 说明中心论断落在哪几章。

### 红线（不变）

- 无第一人称叙述生成过程（注意：描述代码行为的第一人称同样禁）
- 无硬件向内容（不围绕具体卡型/硬件展开）
- 没实验就不写实测数；没出处不写数字
- 不自造数学符号
- 公式内绝对值写 `\lvert o\rvert`，禁裸 `|`
- 行内公式 `$$...$$`，禁单 `$`
- 中文引号全角 `“ ”`
- 元叙述不入正文
- **图**：默认一律自绘。NVIDIA 官方 blog / docs 的图版权保留，**一律不许复制**。
  arXiv 图只在 abs 页**有 CC 标识**时才可引，并标许可。

---

## 交付

报告监工：图注补了几张、外部引用从 4 补到多少、自查结果、**没能核实的部分**
（不许猜、不许编，写「待验证」）。然后**停下等指令**，不要 commit。