# 工单 · 16 篇退回修复（一项硬缺陷）

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`。原始工单见 `SERIES_SPEC.md` §2.1 模块 E。

**其余门禁全过，实验/图注/引文都干净，只修这一项。**

---

## 已通过的项（不要动，返工容易改坏）

| 门禁 | 结果 |
| --- | --- |
| G1 `check_post` | 通过（公式 / 表格 / `<em>`） |
| G2 `check_mermaid` | 通过（3 块全纯中文标签） |
| G3 `scan_garbled` | 第 1/2/4 项干净 |
| G4 `fix_quotes --dry-run` | 0 转换 |
| G5 `audit_permalinks` | 58 处 / 47 distinct，**0 broken**；7 条无 `#L` 的全在参考节（正文内 0），符合 §3.1 |
| G6 裸 `（L\d+` | **0 处** |
| G9 红线 | 0 第一人称 / 0 单 `$` / 0 元叙述 |
| 图注 | **3/3 合规**（含引用关系网图） |
| §8.4 回填 | 正文 0 个 ` ```text ` 块 → 不适用 |

对照参考：同批的 14 篇（653 行）与 15 篇（534 行）已全过门禁并提交，
所以这一篇的差距**只在引用密度这一项**。

---

## 唯一缺陷 · 外部引用密度不足（§3.2 硬门禁）

契约 §3.2：**每 300 行至少 8 条外部引用**。本篇 667 行 → **需 ≥ 18 条**。

**当前真实外部引用只有 9 条**（下面第 1 条是自己站内互链，按 §3.2 不算外部引用，
我先前统计时把它算进去了，这里更正）：

| # | URL | 性质 |
| --- | --- | --- |
| — | `https://lrypcy.github.io/2026/10/13/megatron-15-gpt-and-llama-model-assembly/` | **站内互链，不计入** |
| 1 | `https://docs.python.org/3/library/typing.html#typing.Protocol` | Python 官方文档 |
| 2 | `https://docs.python.org/3/library/typing.html#typing.Optional` | Python 官方文档 |
| 3 | `https://docs.pytest.org/en/stable/how-to/assert.html` | pytest 文档 |
| 4 | `https://arxiv.org/abs/2412.19437` | 论文 |
| 5 | `https://arxiv.org/abs/2405.04434` | 论文 |
| 6 | `https://arxiv.org/abs/2404.06440` | 论文 |
| 7 | `https://arxiv.org/abs/2401.04088` | 论文 |
| 8 | `https://arxiv.org/abs/2312.00752` | 论文 |
| 9 | `https://arxiv.org/abs/2101.03961` | 论文 |

**要补 ≥ 9 条**（按不含站内互链的严格口径；按含站内互链的宽松口径也仍缺 8 条，
两种算法都 FAIL）。

### 禁止的做法（§3.2 原文）

- 「据官方文档」这种**不落 URL** 的说法
- 把二手贴的转述当出处 —— 必须追到原论文 / 原仓库
- **为凑数堆 URL** —— 每条必须挂在具体论断上
- 自己 ipynbs 仓库的脚本 permalink（`github.com/lrypcy/...`）是 **A 级证据，不计入外部引用**

### 该往哪找（按本篇实际章节）

本篇讲 hybrid / MoE 模型装配，缺口集中在这些论断上：

- **Mamba / SSM 类混合架构** → 原始论文已在引（`2401.04088` 等），但**架构混排的动机与显存/序列并行取舍**
  需要补：`docs.nvidia.com/megatron-core` 的并行策略指南、hybrid 相关的官方 user-guide 页
- **层分配（layer allocation）** → `TransformerLayerSubmodules` 与 hybrid layer allocation 的**官方 API 页面**
- **`Mamba` / `GatedDeltaNet` / `linear_attention` 的实现依据** → 除现有 6 篇论文外，
  TransformerEngine 对应模块的官方文档页
- **专家路由 / MoE 混排** → 承接 02 篇（`megatron-02-expert-parallel`）；但**本篇不重复 02 的论断**，
  只引 02 已确立的结论时链到 02 的 permalink，不算外部引用
- **PyTorch `Module` / `Protocol` / `torch.distributed`** 类机制 → `docs.pytorch.org` 具体 API 页
  （当前只有 `docs.python.org` 3 条，PyTorch 侧一条都没有，明显偏科）

优先补 **官方文档类**（`docs.nvidia.com/megatron-core/developer-guide/...` 的具体页面、
`docs.pytorch.org` 的具体 API 页），因为本篇是源码走读，最缺的是「官方怎么定义这个机制」的锚点。

---

## 禁区

- `_posts/2026-09-28-megatron-00-foundation.md` / `_posts/2026-10-08-megatron-02-expert-parallel.md`
- `SERIES_SPEC.md`（除本篇状态列外不改结构）
- `_posts/2026-10-10-megatron-05-*.md` ← 另一窗口在改
- `_posts/2026-10-12-megatron-08-*.md` ← 另一窗口在改
- `TASK-05.md` / `TASK-08-fix.md`
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`、`tools/megatron_bench/`
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，不许 `pull` / `checkout`；
  行号取证一律 `git show 60e039626:<path>`

**你只许写一个文件**：`_posts/2026-10-14-megatron-16-hybrid-and-moe-model-assembly.md`。

## Git 纪律

**不许 `git add` / `git commit` / `git push`。** 改完 → 跑自查 → 报监工 → 等指令。
已记录两次越权（`7842859`、`ff675e6`），第三次即终止写权限。
14 / 15 两篇已由监工验收通过并提交推送（`803bc13` / `7cc7b60`），**你不要自己提交 16**。

## 自查

1. 重数外部引用（排除 `lrypcy.github.io` 与 `github.com/lrypcy`），**≥ 18 条**
2. `~/Software/miniconda3/bin/python3 tools/audit_permalinks.py <file>` —— broken 必须仍为 0
3. `python3 tools/check_post.py <file>`
4. `python3 tools/check_mermaid.py <file>`
5. `python3 tools/scan_garbled.py <file>`
6. `python3 tools/fix_quotes.py --dry-run <file>` —— 必须 0 转换
   （**注意必须带 `--dry-run`**；不带会直接写盘，2026-10-08 已因此事故一次）
7. `grep -nE '（L[0-9]' <file>` —— 必须 0 处
8. 红线自查：第一人称 0 / 元叙述 0 / 单 `$` 0
9. 图注仍是 3/3（新增图才需要补图注）