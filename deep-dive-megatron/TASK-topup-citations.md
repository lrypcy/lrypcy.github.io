# 工单 · 引用密度补引（04 / 08 / 14 / 15 四篇）

> 监工 = Sisyphus。契约见 `SERIES_SPEC.md` §3.2。口径已由 `tools/megatron_gate.py` 定死。

---

## 0. 背景：这是监工自己的错，不是你的

2026-10-09 监工跑 §5-7 这道门禁时，**对不同篇目用了不同口径**：
04 / 08 / 14 / 15 按「出现次数」判，其余按「唯一来源」判。

后果：04 **两项口径都不达标**（唯一 13 / 门槛 37，出现 29）却被放行入库 ——
靠重复挂那 13 个来源刷到了 29 次。

现在口径统一为**唯一来源**，理由写在 `tools/megatron_gate.py` 的文件头：
§3.2 明写「禁止为凑数堆 URL」，而出现次数恰好能被重复挂同一个 URL 刷高。

**你只补引用，不许改动任何技术论断、行号、图注或实验数字。**

---

## 1. 缺口

| 篇 | 行数 | 唯一来源 | 门槛 | **缺** |
| --- | --- | --- | --- | --- |
| **04** GTP | 1359 | 13 | 37 | **24** |
| **08** 显存账本 | 1483 | 35 | 40 | **5** |
| **14** 层栈与 spec | 654 | 15 | 18 | **3** |
| **15** GPT/Llama 装配 | 535 | 11 | 15 | **4** |

**一轮做完四篇**，或按缺口大小分两轮。

---

## 2. 计数口径（与 `tools/megatron_gate.py` 完全一致）

**算**：论文 abs 页、官方文档**具体页面**、官方 blog 具体文章、官方仓库的具体文件。

**不算**：

- `github.com/lrypcy/...`（配套实验脚本，A 级证据）
- `github.com/NVIDIA/Megatron-LM/blob/...`（源码 permalink，属 §3.1 的「源码出处」）
- `lrypcy.github.io`（站内互链）
- **文档目录页 / index 页** —— `.../index.html`、`.../developer-guide/latest/index.html`
  这类**不算**一条外部引用（§3.2 要求「官方文档给具体页面而非仓库根」）

⚠️ 有几条现有引用形如 `https://arxiv.org/abs/2205.05198（**激活重计算**的原始出处——即`
—— URL 后面直接粘了中文。**补引时顺手把这些 URL 与中文之间的分隔补干净**
（URL 本身是对的，是分隔符缺失）。

---

## 3. 每篇该往哪找

### 04 · GTP（缺 24，最重）

现有 13 条几乎全是 `docs.nvidia.com/megatron-core/...` 的 API 索引页 + TE 的 `fp8_primer`。
**缺口方向**：

| 章节 | 该找什么 |
| --- | --- |
| §2 门面与门控、§9 TE `DistributedWeight` 协议 | TransformerEngine 的**其他具体文档页**：FP8 recipe、NVFP4、distributed checkpointing、`make_graphed_callables`（§10 CUDA graph 用得上） |
| §4 切分语义与 §5 五组不变量 | **激活重计算 / 显存优化的原始论文**（不止 `2205.05198` 一篇）；重计算与通信重叠的经典结论 |
| §3 `GTP = TP × GTP_remat` 两个正交子轴 | 权重分片 / 参数高效训练的综述或原始论文 |
| §7 prefetch 链（`GRAPHED` / `UNGRAPHED`） | 参数预取与通信重叠的论文 |
| §8 AG / RS 专用 stream | NCCL 文档的**具体章节**（stream 语义、`ncclCommInitRank`、group 创建） |
| §10 CUDA graph | CUDA Programming Guide 的 **Graph / stream capture 具体章节** |
| 全篇 | `docs.pytorch.org` 的 `distributed` / `Tensor.sub` / autograd saved tensor 等具体 API 页 |

### 08 · 显存账本（缺 5）

已覆盖得很好（PyTorch 15 条 + CUDA 2 + NCCL 2 + TE 1 + Megatron-Core 4 + arXiv 10）。
**缺口方向**：`docs.pytorch.org` 的 `memory` / `profiler` 已引，可补

- `transformer_config.py` docstring 里那三条 `#cuda-graph-integration` /
  `#tuning-parameters` / `#activation-offload-fraction` 锚点对应的**官方文档具体页**
- `docs.nvidia.com/deeplearning/transformer-engine/…` 的 FP8/NVFP4 block scaling 页
- CUDA allocator 的 `cudaMallocAsync` / expandable segments 文档

### 14 · 层栈与 spec（缺 3）

现有全是 Python 标准库文档（`typing` / `dataclasses` / `importlib` / `functools` …）+ PEP 544。
**缺口方向**（这一篇讲的是「模型怎么被造出来」，该引的是**模型侧的官方文档**）：

- `docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/` 下
  `core.transformer.spec_utils` / `core.transformer.transformer_layer` /
  `core.transformer.transformer_block` 的**具体 API 页**（现有只引了 index）
- `docs.pytorch.org` 的 `torch.nn.Module` 之外的：`register_module` /
  `apply` / `named_modules` 等具体 API 页（正文讲模块装配时用得上）

### 15 · GPT/Llama 装配（缺 4）

现有 11 条里有 3 条是 index/release 页。**缺口方向**：

- Megatron-Core API 页：`core.models.gpt.gpt_model` / `core.models.gpt.gpt_layer_specs` /
  `core.models.backends` 的**具体页**
- **GPT 与 Llama 的原始论文**（正文讲这两个模型的装配，原始出处必引）
- `docs.nvidia.com/deeplearning/transformer-engine/user-guide/` 下
  `TransformerLayer` / `fp8` 的**具体页**（现在只引了 index）

---

## 4. 硬要求

- 每条引用**必须挂在具体论断上**，不许在参考节堆一串 URL 充数
- **必须自己 `webfetch` 确认可达**且内容确实支撑该论断
- arXiv / 文档 ID **不许凭印象写**。监工已抓到 `qk_clip` 的常见误引：
  `2109.12142` 其实是错的，QK-Clip 的出处是 Kimi K2 的 **`2507.20534`**。
  **每个 ID 都要打开页面确认标题对得上。**
- 「每 300 行至少 8 条」是**最低线不是目标**。参考已入库的 07 篇：
  1382 行 / 51 个唯一来源（门槛 37），比值 2.5。

## 5. 禁区

- **不许**改技术论断、行号 permalink、图注、实验数字、` ```text ` 块
- **不许**改 `_config.yml` / `tools/kg_*.mjs` / `_data/knowledge_graph.yml`
- 正在被别人写的：`2026-10-18-megatron-10-*`（窗口 5）、
  `2026-10-19-megatron-11-*`（窗口 2）、`2026-10-20-megatron-12-*`（窗口 6）
- `~/Desktop/Projects/github/Megatron-LM` **只读**，不许 `pull` / `checkout`
- `ipynbs` 只读

## 6. Git 纪律

**不许 `git add` / `git commit` / `git push`。** 改完 → 自查 → 报监工 → 等指令。

## 7. 自查

```bash
python3 tools/megatron_gate.py _posts/2026-10-11-megatron-04-generalized-tensor-parallelism.md
```

九项门禁 + permalink 密度 + 图注一次跑完。**只要 §5-7 一项 FAIL 就继续补。**
`fix_quotes` 在该脚本内部**已固定带 `--dry-run`**，不会写盘。