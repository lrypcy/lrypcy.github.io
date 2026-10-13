# 工单 · 07 上下文并行与长上下文

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**（§2.1 孤儿归属 / §3 引用规范 / §3.4 许可白名单）。
> 引用候选见同目录 `CITATION-POOL.md`。

---

## 0. 你只许写一个文件

`_posts/2026-10-17-megatron-07-context-parallel-and-long-context.md`，目标 1300–1800 行。

实验脚本落 **`ipynbs` 仓库**的 `experiments/megatron/context_parallel/`（新建）。
本仓库 `tools/megatron_bench/` **只读，不许扩写**（契约 §7）。

---

## 1. 范围（监工已核定，**这一条比什么都重要**）

07 的全部相关文件加起来约 **6300 行**，塞不下一篇。契约 §2 的编号变更史第 4 条
已经写过这条规律：「EP 单主题 1897 行已超『一个模块一篇』承载上限；六主题合并
至少 6000 行，不可读也不可审。」**07 严格照此办理。**

### ✅ 本篇负责（CP 分派内核，约 2300 行）

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `megatron/core/transformer/dot_product_attention.py` | **271** | `DotProductAttention` 类在 **L26**；**L58** 是 `self.config.context_parallel_size == 1` 的短路 |
| `megatron/core/packed_seq_params.py` | **69** | THD 打包元数据，`PackedSeqParams` |
| `megatron/core/models/multimodal/context_parallel.py` | 536 | 多模态 CP |
| `megatron/core/ssm/mamba_context_parallel.py` | 433 | SSM 的 CP |
| `megatron/core/ssm/gdp_context_parallel.py` | 340 | GatedDeltaProduct 的 CP |
| `megatron/core/extensions/transformer_engine.py` **L2028–L2113** | 片段 | `cp_comm_type` 分派，**只取这一段**（该文件 3678 行，见 §5 打法） |
| `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` | 669 | **06 篇已在写同一文件 —— 本篇只讲它暴露给 CP 的接口，不重复调度器细节** |

### ❌ 本篇不负责（建议另立编号，**先不要动，等监工与用户定**）

| 文件 | 行数 | 理由 |
| --- | --- | --- |
| `megatron/core/transformer/multi_latent_attention.py` | **1502** | MLA |
| `megatron/core/transformer/experimental_attention_variant/absorbed_mla.py` | **953** | absorbed MLA |
| `megatron/core/transformer/experimental_attention_variant/dsa.py` | **2244** | DeepSeek Sparse Attention |

三个合计 **4699 行**，与 CP 分派内核是**两条不同的线索**（一个是「序列怎么切」，
一个是「attention 变体怎么实现」）。塞进来会让本篇超过 3000 行，重演 01 被拆的剧本。

**处理方式与模块 E（14–17）完全一致**：接在编号末尾，本篇只写 CP 内核，
MLA / absorbed MLA / DSA 另立新号。**编号由监工与用户定，你不许自己编号。**

---

## 2. 源码基准

仓库 `~/Desktop/Projects/github/Megatron-LM`，**commit `60e039626`**。
**不许 `pull` / `checkout`**，先 `git -C ~/Desktop/Projects/github/Megatron-LM log -1 --format=%H` 确认。

---

## 3. ⚠️ 计划书有一处错，已核正（**写进正文**）

`RESEARCH_PLAN.md` 的 07 条目把通信方式写成 CLI flag `--cp-comm-type`。
**实际不是 flag。** `cp_comm_type` 是 TE 扩展层的构造参数：

```
transformer_engine.py L2028   cp_comm_type: Optional[str] = "p2p",
                       L2079   if cp_comm_type == "a2a+p2p":
                       L2107       extra_kwargs["cp_comm_type"] = "p2p"
                       L2113       extra_kwargs["cp_comm_type"] = "a2a+p2p"
```

而 `--sequence-packing-scheduler` 对应的读取点在
`megatron/core/datasets/data_schedule.py` L710（`scheduler_type = config.sequence_packing_scheduler`）。

**「为什么一个切序列的方式藏在 attention 扩展的构造参数里」是个值得写的论点** ——
CP 的三种实现选择落在 kernel 侧，不在 Python 侧的策略层。

---

## 4. 骨架（照此写，可细化）

### §1 长序列这一刀怎么切
CP 与 TP / SP / PP 的职责边界；`dot_product_attention.py` L58 那个
`context_parallel_size == 1` 短路说明了什么（不进 CP 时这些代码根本不存在）。

### §2 三条路线的对照
Ring / KV-p2p（`"p2p"`）vs All-Gather（THD）vs Hybrid（`"a2a+p2p"`）。
各自的通信量与对 attention kernel 的要求 —— **本篇的核心对照表**。

### §3 `PackedSeqParams` 与 THD
`cu_seqlens` 怎么算，为什么 THD 能省掉 padding。
本机无 GPU，只能给**解析模型**，不给实测吞吐。

### §4 SSM / Hybrid 的 CP 为什么不能直接用 ring
`mamba_context_parallel.py` 433 行与 `gdp_context_parallel.py` 340 行。
这是本篇最容易写出深度的一节 —— 序列的状态维度让 ring 的 K/V 交换模型失效。

### §5 多模态 CP
`context_parallel.py` 536 行；图像 token 与文本 token 的切法差异。

### §6 图 + 引用关系网
每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面。

---

## 5. 读大文件的正确打法（契约 §1，**已发生 2 次 30 分钟后台任务被杀**）

`dsa.py` 2244 行、`transformer_engine.py` 3678 行 —— **不要整文件读**：

```bash
M=~/Desktop/Projects/github/Megatron-LM
git -C $M grep -n 'def <符号>' 60e039626 -- megatron/core/extensions/transformer_engine.py
git -C $M show 60e039626:megatron/core/extensions/transformer_engine.py | sed -n '2028,2113p'
```

⚠️ **本环境下后台 subagent 任务会静默超时、零输出返回。** 监工派出的三个
后台任务（`deep` ×1、`librarian` ×2）全部是 `Total messages: 1` 就 30 分钟超时。
**外部资料检索请自己用 `websearch` / `webfetch`，或看 `CITATION-POOL.md`，不要派后台 librarian。**

---

## 6. 实验设计（本机**无 GPU**，只有 A / C 级）

证据分级见 `RESEARCH_PLAN.md` §1。**禁止 D 级**（凭印象的数字）。
推不出来又查不到出处的，明写「**待验证**」。

### C 级 · online softmax 分块递推 ≡ 全量 softmax（**本篇最值得做的一项**）

这是 ring attention 的正确性地基：**分块递推的 online softmax 与一次性全量 softmax
必须数值等价**，而误差随块数增长的曲线就是「切得越碎、误差越大」的定量表述。

用 `~/Software/miniconda3/bin/python3`（managed python 3.13 无 numpy）。
脚本落 `ipynbs/experiments/megatron/context_parallel/`，结构
`<name>.py` + `README.md` + `results/`（输出固化，读者不必自己跑）。

**做法**：`scores = Q @ K.T`，先算全局 max，再分块累加 exp 并修正 ——验证与
`softmax(scores)` 逐位/近逐位一致；扫块数 1/2/4/8/16/32，输出相对误差曲线。

### A 级 · 三条路线的 KV 通信量解析模型
ring / THD(all-gather) / hybrid 三种路线，扫 CP size 出每层每 token 的 KV 字节数，
给边际收益递减点。**只给解析值，不给实测吞吐。**

---

## 7. 引用密度（**这是 05/08/16 三篇全部退回的唯一原因，务必一次做够**）

契约 §3.2：**每 300 行至少 8 条外部引用**。本篇 1300–1800 行 → **需 ≥ 35–48 条**。

前车之鉴（监工亲跑的数）：

| 篇 | 行数 | 唯一外部来源 | 门槛 | 结果 |
| --- | --- | --- | --- | --- |
| 05（修完前） | 1446 | 4 | 38 | ❌ |
| 16（修完前） | 667 | 9 | 18 | ❌ |
| **05（修完后）** | 1559 | **44** | 42 | ✅ |
| **08** | 1483 | **35** | 39 | ✅ |

**计数口径**（监工用的就是这个）：排除 `lrypcy.github.io`（站内互链）与
`github.com/lrypcy/ipynbs`（A 级实验脚本）。

**靠的是几十个不同的真实来源，不是把同一个 URL 重复几十次。**
契约 §3.2 明写「禁止为凑数堆 URL」「每条必须挂在具体论断上」。

本篇该找的官方来源（**每条引用前必须自己 `webfetch` 确认可达且真能支撑该论断**）：

- **Ring Attention（arXiv:2310.01889）** —— ring CP 的原始出处，**本篇最该引的一篇**
- **DeepSpeed Ulysses（arXiv:2309.14509）** —— a2a 路线
- `docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html`
  —— CP 的官方定位（该 URL 已在 04/05/16 验收通过，可直接用）
- `docs.nvidia.com/deeplearning/transformer-engine/…` —— TE 对 CP 的约束
- `docs.pytorch.org/…` —— flex attention / 序列并行作为对照

⚠️ **arXiv ID 不许凭印象写。** 监工发现 `qk_clip` 的常见误引是 `2109.12142`
（实际是 Kimi K2 的 `2507.20534`）。**每个 ID 都要打开 abs 页确认标题对得上。**

---

## 8. 禁区

**绝对不许动**：

- `_posts/2026-09-28-megatron-00-foundation.md`、`_posts/2026-10-08-megatron-02-expert-parallel.md`
- `_posts/2026-10-15-megatron-09-low-precision-fp8-fp4.md` ← 另一窗口在改
- `_posts/2026-10-16-megatron-06-pipeline-parallel-schedules.md` ← 另一窗口在写
- `SERIES_SPEC.md` 的编号表与模块 E 章节
- `TASK-05.md` / `TASK-06.md` / `TASK-08-fix.md` / `TASK-16-fix.md` / `CITATION-POOL.md`
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`
- `tools/megatron_bench/` 全部（存量，不许扩写）
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，一条 git 写命令都不许

已入库不许动的篇目：`04`（`6729660`）、`05`（`43b8616`）、`08`（`3c648ec`）、
`14`（`803bc13`）、`15`（`7cc7b60`）、`16`（`db8e842`）。

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
9. 图注判据：紧跟围栏之后、以 `> ` 或 `图 N：` 开头、且含「出处」/ URL / 行号锚点
   （引用关系网图豁免 per-edge permalink）

### 红线

无第一人称叙述生成过程 · 无硬件向内容 · 没实验不写实测数 · 没出处不写数字 ·
不自造数学符号 · 公式内绝对值写 `\lvert o\rvert` 禁裸 `|` · 行内公式 `$$...$$` 禁单 `$` ·
中文引号全角 `“ ”` · 元叙述不入正文

**图**：默认一律自绘。NVIDIA 官方 blog / docs 的图版权保留，**一律不许复制**（§3.4）。
arXiv 图只在 abs 页**有 CC 标识**时才可引，并标许可。

---

## 11. 交付

报告监工：篇名、行数、图数、**外部引用唯一来源数 / 出现次数**、实验断言结果、
自查结果、**没能核实的部分**（不许猜、不许编，写「待验证」）。然后**停下等指令**。