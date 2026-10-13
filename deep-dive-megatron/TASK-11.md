# 工单 · 11 Refit / Resharding：训练与推理之间的权重搬运

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**（§2.1 孤儿归属 / §3 引用规范 / §3.4 许可白名单）。
> 引用候选见同目录 `CITATION-POOL.md`。

---

## 0. 你只许写一个文件

`_posts/2026-10-19-megatron-11-refit-resharding.md`，目标 1300–1800 行。

实验脚本落 **`ipynbs` 仓库**的 `experiments/megatron/refit/`（新建）。
本仓库 `tools/megatron_bench/` **只读，不许扩写**（契约 §7）。

**本篇是你这一轮的「跨主线交叉点」**：本博客已有 `rl-*` / `toolrl` / `search-rl` /
`distillation-*` / `online-distillation` / `opd-*` 六条 RL 线，
Refit 是 Megatron 侧唯一与它们对接的机制。正文该链到那些篇目的 permalink，
**不要重复推导 RL 原理**。

---

## 1. 范围（监工已核定）

`megatron/core/resharding/` 全目录 **5800 行**，塞不下一篇。
按契约 §2 变更史第 4 条先例（01 被拆、06/07/10 按主题拆）处理。

### ✅ 本篇负责：Refit 内核与四种 copy service

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `megatron/core/resharding/refit.py` | **495** | 门面 |
| `megatron/core/resharding/planner.py` | **524** | **本篇最值得写的部分**，见 §3 |
| `megatron/core/resharding/execution.py` | **260** | 排期落地 |
| `megatron/core/resharding/transforms.py` | **295** | `MXFP8ReshardTransform` |
| `megatron/core/resharding/copy_services/base.py` | — | 传输抽象 |
| `.../copy_services/nccl_copy_service.py` | — | NCCL 传输 |
| `.../copy_services/gloo_copy_service.py` | — | Gloo 传输 |
| `.../copy_services/nixl_copy_service.py` | — | NIXL 传输 |
| `.../copy_services/nvshmem_copy_service.py` | — | NVSHMEM 传输 |

### ❌ 本篇不负责（另立编号，**编号等监工与用户定，你不许自己编号**）

`megatron/rl/` —— **29 个文件，四组子目录 + 8 个顶层模块**（监工 2026-10-09 完整核验，未截断）：

| 子目录 | 文件数 | 内容 |
| --- | --- | --- |
| `agent/` | 9 | `api.py` / `registry.py` / `remote_agent.py` / `reward_only_agent.py` / `rollout_pipeline.py` / `weighted_multi_task.py` / `huggingface_dataset_agent.py` / `pass_at_evaluation_agent.py` / `__init__.py` |
| `inference/` | 4 | `api.py` / `inference_interface.py` / `megatron.py` / `__init__.py` |
| **`server/`** | **6** | **`agent/fastapi_env_server.py` 205 行** / **`inference/inference_interface_server.py` 95 行** / `api.py` 49 行 / 三个 `__init__.py` |
| **`types/`** | **2** | `rollout.py` 72 行 / `__init__.py` 25 行 |

顶层另有 8 个：`README.md` / `__init__.py` / `logging.py` / `rl_profiling.py` /
`rl_utils.py` / `rollout_bank.py` / `rollout_granularity.py` /
`sequence_packing_utils.py`。

**可以**在正文里引用它们说明「Refit 服务的是谁」，**不许**展开写 RL 框架本身。
「四组」这个结构本身可以是个论点：`agent/` 定义接口、`inference/` 对接推理引擎、
`server/` 是 FastAPI 环境服务端、`types/` 是 rollout 的类型契约 ——
**训练侧的类型契约与训练侧的权重搬运是两件事**，这个区分值得写。

> ⚠️ **`RESEARCH_PLAN.md` 有一处错，是本篇可以写进正文的一个论点**（监工 2026-10-09 核正）：
> 计划书 L485 把 `copy_services/` 写成「**NCCL / Gloo / NVSHMEM 三种传输**」，
> **实际有四种** —— `nccl_copy_service.py` / `gloo_copy_service.py` /
> **`nixl_copy_service.py`** / `nvshmem_copy_service.py`，另有 `base.py` 抽象。
> 「三种传输」这个数是抄旧版留下的，四条路径各自的适用场景见 §3。
>
> （监工此前在本工单里写「`megatron/rl/` 没有 `server/`」是**错的**，
> 已在下面更正 —— 那是监工核验时用 `head -20` 截断了 29 个文件的清单所致。
> `megatron/rl/` 实为四组：`agent/` 9 文件、`inference/` 4、`server/` 6、`types/` 2。
> RESEARCH_PLAN L483 列的 `agent/` / `inference/` / `server/` /
> `rollout_bank.py` / `sequence_packing_utils.py` **全部存在，计划书是对的**。）

---

## 2. 源码基准

仓库 `~/Desktop/Projects/github/Megatron-LM`，**commit `60e039626`**。
**不许 `pull` / `checkout`**，先 `git -C ~/Desktop/Projects/github/Megatron-LM log -1 --format=%H` 确认。

行号必须**现取现核**：`git show 60e039626:<path> | sed -n 'A,Bp'`。

---

## 3. 骨架（照此写，可细化）

### §1 训练与推理的并行布局为什么不同
训练侧是 DP+TP+PP+EP 的网格（回引 01 篇），推理侧是另一套。
Refit 要解决的就是「同一份权重在两套布局之间怎么搬」。

### §2 planner：每 rank 重放同一份确定性排期
**本篇的核心设计**。每个 rank 跑同一个 planner，得出完整排期，
**只保留属于自己那一片的 op** —— 于是 N 个 rank 得到 N 份互补的排期，
而不是 N 份相同的排期再互相协调。
这个设计的代价与收益都要讲清（省掉了协调，但要求排期严格确定）。

### §3 四种 copy service 的取舍
NCCL / Gloo / NIXL / NVSHMEM 四条传输路径：
各自适合什么场景（GPU 间 / CPU 跨机 / RDMA / 高速互联）。
**这是本篇最容易写出对照表的一节。**

### §4 `transforms.py` 与 MXFP8ReshardTransform
搬运不是纯字节拷贝 —— 量化参数在重分片后要保持量化语义。
`MXFP8ReshardTransform` 是怎么处理的。**回引 09 篇**（FP8/FP4 那一篇）。

### §5 代价测算
「训练/推理同构 vs 异构 resharding」的通信量与耗时解析模型。
**本机无 GPU，只给解析量**，不给实测数字。

### §6 图 + 引用关系网
每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面。

---

## 4. 读大文件的正确打法（契约 §1，**已发生 2 次 30 分钟后台任务被杀**）

⚠️ **本环境下后台 subagent 任务会静默超时、零输出返回。** 监工派出的三个后台任务
（`deep` ×1、`librarian` ×2）全部是 `Total messages: 1` 就 30 分钟超时。
**外部资料检索请自己用 `websearch` / `webfetch`，或看 `CITATION-POOL.md`，不要派后台 librarian。**

---

## 5. 引用密度（**这是 05/08/16/09 四篇退回的唯一原因，务必一次做够**）

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
契约 §3.2 明写「禁止为凑数堆 URL」。

本篇该找的官方来源（**每条引用前必须自己 `webfetch` 确认可达且真能支撑该论断**）：

- `docs.nvidia.com/deeplearning/nccl/…` —— NCCL 传输语义
- NVSHMEM 与 NIXL 的官方文档 —— 两种高速传输路径的定位
- **RL 与推理侧权重重排的论文**（本博客 `rl-*` / `online-distillation` 系列
  已有的结论，链到本博客自己的 permalink，不算外部引用）
- `docs.nvidia.com/megatron-core/developer-guide/…` —— resharding 相关页

⚠️ **arXiv / 文档 ID 不许凭印象写。** 监工发现 `qk_clip` 的常见误引是 `2109.12142`
（实际是 Kimi K2 的 `2507.20534`）。**每个 ID 都要打开页面确认标题对得上。**

---

## 6. 实验设计（本机**无 GPU**，只有 A / C 级）

**禁止 D 级**（凭印象的数字）。推不出来又查不到出处的，明写「**待验证**」。

### A 级 · 排期与通信量解析模型（本篇唯一能纯算的）
给定两套并行布局（训练的 TP×PP×DP vs 推理的另一套），算出
每个 rank 要搬多少字节、总共多少 op。**这只是模型，不许假装实测过。**

### C 级 · planner 的确定性验证
同一输入跑两次 planner，排期是否逐位相同 —— 这是「每 rank 重放同一份排期」的前提。
若能构造一个反例（说明什么输入会让它不确定），那更有价值。

脚本落 `ipynbs/experiments/megatron/refit/`，
结构 `<name>.py` + `README.md` + `results/`（输出固化）。
用 `~/Software/miniconda3/bin/python3`（managed python 3.13 无 numpy）。

---

## 7. 禁区

**绝对不许动**：

- `_posts/2026-09-28-megatron-00-foundation.md`、`_posts/2026-10-08-megatron-02-expert-parallel.md`
- `_posts/2026-10-18-megatron-10-cuda-graph.md` ← 另一窗口在写
- `SERIES_SPEC.md` 的编号表与模块 E 章节
- `TASK-05.md` / `TASK-06.md` / `TASK-07.md` / `TASK-08-fix.md` / `TASK-10.md` /
  `TASK-16-fix.md` / `CITATION-POOL.md`
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