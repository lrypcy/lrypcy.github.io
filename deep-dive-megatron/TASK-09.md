# 工单 · 09 低精度：FP8 全链路与 FP4

> 监工 = Sisyphus。契约见同目录 `SERIES_SPEC.md`，**开工前必读**（尤其 §2.1 孤儿归属、§3 引用规范、§3.4 许可白名单）。
> 引用候选见同目录 `CITATION-POOL.md`。

---

## 0. 你只许写一个文件

`_posts/2026-10-15-megatron-09-low-precision-fp8-fp4.md`，目标 1200–1800 行。

实验脚本落 **`ipynbs` 仓库**的 `experiments/megatron/low_precision/`（新建）。
**本仓库 `tools/megatron_bench/` 只读，不许扩写**（契约 §7）。

---

## 1. 源码清单（行号已由监工核过，但**你必须逐条用 `git show 60e039626:` 重核**）

仓库 `~/Desktop/Projects/github/Megatron-LM`，基准 **commit `60e039626`**。
**不许 `pull` / `checkout`**，先 `git -C ~/Desktop/Projects/github/Megatron-LM log -1 --format=%H` 确认 HEAD 仍是它。

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `megatron/core/fp8_utils.py` | **1056** | 本篇主体。公开符号与行号：`_unwrap_parameter_data` L104 / `_is_instance_or_param_data` L115 / `is_float8tensor` L122 / `is_mxfp8tensor` L134 / `is_grouped_tensor` L139 / `is_grouped_tensor_with_quantized_storage` L144 / `_get_grouped_quantized_recipe` L153 / `is_grouped_mxfp8tensor` L165 / `get_grouped_quantized_members` L173 / `copy_tensor_to_quantized_param` L193 / `copy_tensors_to_quantized_params` L229 / `modify_grouped_tensor_rowwise_storage` L269 / `dequantize_fp8_tensor` L303 / `_resolve_callable_from_python_import_path` L311 |
| `megatron/core/fp4_utils.py` | **302** | NVFP4 / MXFP4 路径 |
| `megatron/core/extensions/transformer_engine.py` | **3678** | TE 集成层，本系列最大单文件之一。**不要整file 读**（见 §5 打法） |
| `megatron/core/optimizer/clip_grads.py` | 277 | **契约 §2.1 指定孤儿 → 归本篇** |
| `megatron/core/optimizer/grad_scaler.py` | 165 | **契约 §2.1 指定孤儿 → 归本篇** |
| `megatron/training/arguments.py` | — | flag 的**静默降级**逻辑，见 §2 |

> `clip_grads.py` + `grad_scaler.py` 是契约 §2.1 明确划给本篇的两个孤儿
> （在 05 写完时提及次数是 1 和 **0**）。**每篇必须写成可独立阅读的一节，
> 不许只在别篇提一句名字就算交差**（§2.1 执行纪律）。

---

## 2. 已核实的两个坑（本篇的骨架就搭在这上面）

###坑一 · flag 会被**静默关掉**

`megatron/training/arguments.py` L1043–L1054：

```
L1043  if args.fp8_param_gather and is_te_min_version("2.0.0"):
L1044      args.fp8_param_gather = False
L1047      'all_gather instead, turning off fp8_param_gather'
L1050  if args.fp4_param_gather and not is_te_min_version("2.7.0.dev0"):
L1051      args.fp4_param_gather = False
L1054      'all_gather instead, turning off fp4_param_gather'
```

**你显式传了 flag，程序不报错，只打一行 log 就把它关掉。** 这一点必须写进正文 ——
本系列已经吃过一次同类亏：`omo.jsonc` 里 `opencode-go/kimi-k2.6` 与
`opencode-go/qwen3.7-max` 已不存在，路由静默掉到兜底档，没人发现。

同时 `arguments.py` L1074–L1075：`main_grads_dtype` / `main_params_dtype`
经 `map_dtype` 归一化 —— 这是精度路径的入口，核清楚它接受什么写法。

### 坑二 · fp8 param buffer 的存储类型切换

`fp8_utils.py` 有一整套 `is_*tensor` / `copy_tensor*_to_quantized_param` /
`modify_grouped_tensor_rowwise_storage` / `dequantize_fp8_tensor`
（L122–L303）。这条链决定了 param gather 阶段能不能降精度、
gather 完怎么还原、`GroupedTensor` 与 `QuantizedTensor` 如何并存。

**05 篇 §2.5 已经讲过 NVFP4 打包让 param buffer numel 减半**——
本篇要接着往下走：这些flag 打开后，buffer 的存储类型与生命周期怎么变。

---

## 3. 骨架（照此写，可细化）

### §1 三种 scaling 策略与 amax 归约
delayed / current / MX 三种策略分别在干什么；`get_amax_reduction_group`
（**行号现取**）怎么决定跨哪些 rank 归约 amax；归约发生在什么时机。

### §2 param gather 的精度路径
为什么要保留 fp32 主权重、更新后再 cast；`--fp8-param-gather` /
`--fp4-param-gather` 打开后在 gather 阶段降精度能省多少带宽。
**这里必须回引05 §2.5**（NVFP4 打包的 numel 减半）。

### §3 grad reduce 的精度
bf16 做归约 + fp32 累加的必要性；05 §11 的 `reduce_scatter_with_fp32_accumulation`
在本篇的语境下是什么关系（**只引 05 的结论，不重复推导**）。

### §4 grad clip 与 loss scaling（**契约 §2.1 的两个孤儿，各写一节**）
- `clip_grads.py` 277 行：连续 buffer 上对整块做 `torch.norm` 的前提
  （05 §2 已经铺过这个前提，本篇只讲 clip 本身的算法与阈值语义）
- `grad_scaler.py` 165 行：fp16 loss scaling 在 bf16 时代的地位

### §5 NVFP4 / MXFP4 到哪一步
目前能训到哪一步、哪些层不能降、精度损失量级。
**不许写「大约损失 X%」这种无出处数字**（契约 §4 禁止 D 级证据）。

### §6 TE 集成层
`transformer_engine.py` 3678 行里与低精度相关的部分怎么组织；
`_resolve_callable_from_python_import_path`（`fp8_utils.py` L311）说明 mcore
对 TE 的依赖是**运行期动态解析**的——这与 04 篇「GTP 必须依赖 TE」是同一条线索。

### §7 图 + 引用关系网
每张图下方必须有图注，指向具体 permalink / 论文 / 官方页面。

---

## 4. 引用密度（**这是 05 / 08 / 16 三篇全部退回的唯一原因，务必一次做够**）

契约 §3.2：**每 300 行至少 8 条外部引用**。本篇 1200–1800 行 → **需 ≥ 32–48 条**。

前车之鉴（监工亲跑的数）：

| 篇 | 行数 | 唯一外部来源 | 出现次数 | 门槛 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 05 | 1446 | 4 | 5 | 38 | ❌ 退回 |
| 08（修完） | 1483 | **35** | 66 | 39 | ✅ 通过 |
| 16 | 667 | 9 | 9 | 17 | ❌ 退回 |

**计数口径**（监工用的就是这个，你自查也用同一个）：
排除 `lrypcy.github.io`（站内互链）与 `github.com/lrypcy/ipynbs`（A 级实验脚本）。

**08 修完后能过，靠的是 35 个不同的真实来源**，不是把同一个 URL 重复 66 次。
契约 §3.2 明写「禁止为凑数堆URL」，「每条必须挂在具体论断上」。

本篇该找的官方来源（**每条引用前必须自己 `webfetch` 确认可达且真能支撑该论断**）：

- `docs.nvidia.com/deeplearning/transformer-engine/…` —— FP8 recipe、
  MXFP8 block scaling、NVFP4 格式定义（04 篇只用过 `fp8_primer.html`，**FP4 侧是新的**）
- `docs.nvidia.com/megatron-core/developer-guide/latest/…` —— 低精度相关的
  API 页与 user-guide 页
- `docs.pytorch.org/docs/stable/amp.html` —— autocast 与 FP16 loss scaling 的官方语义
- arXiv：低精度训练的误差补偿、NVFP4/MXFP8 的提出论文、量化误差的原始出处
  （**ID 必须打开 abs 页核，不要凭印象写**）

候选清单见 `CITATION-POOL.md` §1 与 §2。里面已核验的 ID / URL 可直接用，
但 §2 那套 Megatron-Core URL 结构**必须逐个 fetch 确认**，不要照抄格式就算数。

---

## 5. 读大文件的正确打法（契约 §1，**已发生 2 次 30 分钟后台任务被杀**）

`transformer_engine.py` 3678 行 —— **不要整文件读，也不要丢给 subagent 去「读懂」**：

```bash
M=~/Desktop/Projects/github/Megatron-LM
git -C $M grep -n 'def <符号>' 60e039626 -- megatron/core/extensions/transformer_engine.py
git -C $M show 60e039626:megatron/core/extensions/transformer_engine.py | sed -n 'A,Bp'
git -C $M show 60e039626:megatron/core/extensions/transformer_engine.py | wc -l
```

后台 `explore` / `librarian` **只用于外部资料检索**，不用来读本地源码。

---

## 6. 实验设计（本机**无 GPU**，只有 A / C 级）

证据分级见 `RESEARCH_PLAN.md` §1。**禁止 D 级**（凭印象的数字）。
推不出来又查不到出处的，明写「**待验证**」。

### C 级（本篇重头戏，可接 `llm-quant-00..25 + E1..E4` 主线）

在 `ipynbs/experiments/megatron/low_precision/` 建脚本，**手撸三种量化器**：
E4M3 / E5M2 / MX-block-scaling，在**合成权重张量**（正态分布 + 注入 outlier）上跑：

- RMSE / SNR / 极值保留率
- 与 per-tensor 和 per-channel INT8 **并排对比**
- 量化误差随 scale 粒度（per-tensor / per-channel / per-block）的变化曲线
- 验证「per-block 粒度换来的误差下降是否值得它引入的 amax 归约开销」
  —— 这就是 delayed / current / MX 三种策略的取舍的量化表达

> 正文若引用 `llm-quant` 主线的结论，**链到它的 permalink，不复制推导**
> （避免与主线重复，dist-train 系列的教训）。

用 `~/Software/miniconda3/bin/python3`（managed python 3.13 无 numpy）。
`results/` 固化输出，读者不必自己跑。

### 顺带可做（时间允许）
`clip_grads.py` 的数值等价性验证：连续 buffer 上整块 `torch.norm`
与逐参数循环 `torch.norm` 的结果一致性。

---

## 7. 禁区

**绝对不许动**：

- `_posts/2026-09-28-megatron-00-foundation.md`、`_posts/2026-10-08-megatron-02-expert-parallel.md` ← 窗口 2
- `_posts/2026-10-10-megatron-05-data-parallel-and-optimizers.md` ← 窗口 6 在改
- `_posts/2026-10-14-megatron-16-hybrid-and-moe-model-assembly.md` ← 窗口 2 在改
- `SERIES_SPEC.md` 的编号表与模块 E 章节 ← 窗口 2
- `TASK-05.md` / `TASK-08-fix.md` / `TASK-16-fix.md` / `CITATION-POOL.md`
- `_config.yml`、`tools/kg_*.mjs`、`_data/knowledge_graph.yml`
- `tools/megatron_bench/` 全部（存量，不许扩写）
- `~/Desktop/Projects/github/Megatron-LM` —— **只读**，一条 git 写命令都不许

已入库不许动的篇目：`04`（`6729660`）、`08`（`3c648ec`）、`14`（`803bc13`）、`15`（`7cc7b60`）。

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
   （不带会直接写盘。2026-10-08 监工已因此踩过一次，修了 10 处被误改的引文；
   护栏已加，但你还可能漏 `--dry-run`）
4. **回填一致性比对** —— 每个 ` ```text ` 块的数字与 `ipynbs` 脚本真实输出按数字字面量对齐

契约 §5 另加：

5. `python3 tools/check_mermaid.py <file>`
6. `~/Software/miniconda3/bin/python3 tools/audit_permalinks.py <file>` —— broken 必须 0
7. `grep -nE '（L[0-9]' <file>` —— 裸行号必须 0
8. **外部引用密度 ≥ 32 条**（监工口径，见 §4）
9. 图注：每张图下方带出处，判据 =紧跟围栏之后、以 `> ` 或 `图 N：` 开头、
   且含「出处」/ URL / 行号锚点（引用关系网图豁免per-edge permalink）

### 红线

无第一人称叙述生成过程 · 无硬件向内容 · 没实验不写实测数 · 没出处不写数字 ·
不自造数学符号 · 公式内绝对值写 `\lvert o\rvert` 禁裸 `|` · 行内公式 `$$...$$` 禁单 `$` ·
中文引号全角 `“ ”` · 元叙述不入正文

**图**：默认一律自绘。NVIDIA 官方 blog / docs 的图版权保留，**一律不许复制**
（契约 §3.4）。arXiv 图只在 abs 页**有 CC 标识**时才可引，并标许可。

---

## 10. 交付

报告监工：篇名、行数、图数、**外部引用唯一来源数 / 出现次数**、实验断言结果、
自查结果、**没能核实的部分**（不许猜、不许编，写「待验证」）。然后**停下等指令**。