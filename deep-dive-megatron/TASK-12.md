# 工单 · 12 确定性重跑与故障注入

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**（§2.1 孤儿归属 / §3 引用规范 / §3.4 许可白名单）。
> 引用候选见同目录 `CITATION-POOL.md`。

---

## 0. 你只许写一个文件

`_posts/2026-10-20-megatron-12-deterministic-rerun-and-fault-injection.md`，目标 1300–1800 行。

实验脚本落 **`ipynbs` 仓库**的 `experiments/megatron/rerun/`（新建）。
本仓库 `tools/megatron_bench/` **只读，不许扩写**（契约 §7）。

---

## 1. 范围（监工已核定）

### ✅ 本篇负责：确定性重跑与故障注入（约 1650 行）

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `megatron/core/rerun_state_machine.py` | **1420** | 本篇主体。确定性重跑的状态机 |
| `megatron/core/fault_injector.py` | **233** | 故障注入 |

### ❌ 本篇不负责（另立编号，**编号等监工与用户定，你不许自己编号**）

`megatron/core/dist_checkpointing/` —— **22 个文件、合计 7793 行**
（其中 `serialization.py` 463 行、`core.py` 81 行）。
体量是本篇的五倍，属另一条线索。

**可以**在正文里引用它说明「重跑与存档是两件事」，**不许**展开写存档格式。

---

## 2. 源码基准

仓库 `~/Desktop/Projects/github/Megatron-LM`，**commit `60e039626`**。
**不许 `pull` / `checkout`**，先 `git -C ~/Desktop/Projects/github/Megatron-LM log -1 --format=%H` 确认。

行号必须**现取现核**：`git show 60e039626:<path> | sed -n 'A,Bp'`。

---

## 3. 骨架（照此写，可细化）

### §1 要解决的到底是什么问题
硬件会挂、位翻转会毁一个 step 的算子输出、单个 rank 的 NCCL 超时会拖死整个作业。
「重跑这一个 step」听起来简单，但**重跑必须与原跑逐位一致**，否则梯度累加就错了。

### §2 状态机记的是什么
`rerun_state_machine.py` 1420 行。核心是**每个 step 上挂了哪些「需要校验的中间量」**。
`scan_garbled.py` 那类校验不是逐算子做，而是挂在这套机制上。

### §3 校验的时机与代价
什么时候查、查多贵、误报怎么处理。**这是全篇最该写深的一节** ——
「用多少算力换容错」是一个显式的成本核算，不是免费的保险。

### §4 `fault_injector.py`：怎么主动制造故障
233 行。**这是本篇最独特的一节** —— 有了它才能证明机制真的有效，
而不是「理论上能重跑」。主动注入故障 + 观察是否正确恢复，
是把「机制存在」升级为「机制有效」的唯一办法。

### §5 与其他篇的接口
- 与 05 篇的梯度归约：重跑后梯度累加不能重复计入
- 与 01 篇的进程组：哪些 rank 参与重跑、哪些等待

### §6 图 + 引用关系网
每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面。

---

## 4. 读大文件的正确打法（契约 §1）

⚠️ **本环境下后台 subagent 任务会静默超时、零输出返回。** 监工派出的三个后台任务
（`deep` ×1、`librarian` ×2）全部是 `Total messages: 1` 就 30 分钟超时。
**外部资料检索请自己用 `websearch` / `webfetch`，或看 `CITATION-POOL.md`，不要派后台 librarian。**

---

## 5. 引用密度（**这是 05/08/09/16 四篇退回的唯一原因，务必一次做够**）

契约 §3.2：**每 300 行至少 8 条外部引用**。本篇 1300–1800 行 → **需 ≥ 35–48 条**。

前车之鉴（监工亲跑的数）：

| 篇 | 行数 | 唯一外部来源 | 门槛 | 结果 |
| --- | --- | --- | --- | --- |
| 09（修完前） | 1393 | 29 | 38 | ❌ |
| **09（修完后）** | 1417 | **41** | 38 | ✅ |
| **06** | 1504 | **42** | 41 | ✅ |
| **07** | 1381 | **51** | 37 | ✅ |

**计数口径**（监工用的就是这个）：排除 `lrypcy.github.io`（站内互链）与
`github.com/lrypcy/ipynbs`（A 级实验脚本）。

**靠的是几十个不同的真实来源，不是把同一个 URL 重复几十次。**

本篇该找的官方来源（**每条引用前必须自己 `webfetch` 确认可达且真能支撑该论断**）：

- **确定性计算与位级复现** —— 这是本篇的主题，找得到原始论文
- `docs.nvidia.com/megatron-core/developer-guide/…` —— 容错 / 重跑相关 user-guide 页
- `docs.nvidia.com/deeplearning/nccl/…` —— 超时与集合通信失败的语义
- PyTorch 分布式文档里与 in-process restart 相关的页
- 硬件故障与静默数据损坏的公开资料（**只引论文/官方文档，不引新闻**）

⚠️ **arXiv / 文档 ID 不许凭印象写。** 监工发现 `qk_clip` 的常见误引是 `2109.12142`
（实际是 Kimi K2 的 `2507.20534`）。**每个 ID 都要打开页面确认标题对得上。**

---

## 6. 实验设计（本机**无 GPU**，只有 A / C 级）

**禁止 D 级**（凭印象的数字）。推不出来又查不到出处的，明写「**待验证**」。

### C 级 · 重跑正确性的数值验证（本篇唯一能做的）
构造一个「同一个 step 跑两次」的场景，验证两次的**输出与梯度逐位一致** ——
这是整套机制成立的前提。若能演示「不挂校验项时误差被静默累加」，
那比「挂上就一致」更有说服力。

脚本落 `ipynbs/experiments/megatron/rerun/`，
结构 `<name>.py` + `README.md` + `results/`（输出固化）。
用 `~/Software/miniconda3/bin/python3`（managed python 3.13 无 numpy）；
**若脚本需要 torch，用 `~/Software/miniconda3/envs/torch/bin/python`**
（监工在 08 篇踩过这个坑：用错解释器会误判脚本坏了）。

### A 级 · 校验开销的核算模型
挂 N 个校验项时每个 step 的额外开销。**只给算术模型，不许假装实测。**

---

## 7. 禁区

**绝对不许动**：

- `_posts/2026-09-28-megatron-00-foundation.md`、`_posts/2026-10-08-megatron-02-expert-parallel.md`
- `_posts/2026-10-18-megatron-10-cuda-graph.md` ← 另一窗口在写
- `_posts/2026-10-19-megatron-11-refit-resharding.md` ← 另一窗口在写
- `SERIES_SPEC.md` 的编号表与模块 E 章节
- `TASK-05.md` / `TASK-06.md` / `TASK-07.md` / `TASK-08-fix.md` /
  `TASK-10.md` / `TASK-11.md` / `TASK-16-fix.md` / `CITATION-POOL.md`
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`
- `tools/megatron_bench/` 全部（存量，不许扩写）
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，一条 git 写命令都不许

已入库不许动的篇目：`04`（`6729660`）、`05`（`43b8616`）、`06`（`6ecbcb2`）、
`07`（`cf3a399`）、`08`（`3c648ec`）、`09`（`61914f0`）、`14`（`803bc13`）、
`15`（`7cc7b60`）、`16`（`db8e842`）。

---

## 8. Git 纪律（契约 §6，第三次即终止写权限）

**不许 `git add` / `git commit` / `git push` / `git add -A` / `git add .`。**
写完 → 自查 → 报监工 → 等指令。**就这三步。**

已记录两次越权（`7842859`、`ff675e6`）。

---

## 9. 写完的自查

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
8. **外部引用密度 ≥ 35 条**（监工口径，见 §5）
9. 图注判据：紧跟围栏之后、以 `> ` 或 `图 N：` 开头、且含「出处」/「来源」/ URL / 行号锚点
   （引用关系网图豁免 per-edge permalink）

### 红线

无第一人称叙述生成过程 · 无硬件向内容 · 没实验不写实测数 · 没出处不写数字 ·
不自造数学符号 · 公式内绝对值写 `\lvert o\rvert` 禁裸 `|` · 行内公式 `$$...$$` 禁单 `$` ·
中文引号全角 `“ ”` · 元叙述不入正文

**图**：默认一律自绘。NVIDIA 官方 blog / docs 的图版权保留，**一律不许复制**（§3.4）。

---

## 10. 交付

报告监工：篇名、行数、图数、**外部引用唯一来源数 / 出现次数**、实验断言结果、
自查结果、**没能核实的部分**（不许猜、不许编，写「待验证」）。然后**停下等指令**。