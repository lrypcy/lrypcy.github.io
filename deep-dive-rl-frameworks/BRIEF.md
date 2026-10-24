# rl-frameworks 系列：写作契约（所有撰稿人必读）

本文件不属于任何一篇正文，属于系列内部契约。**不要把它写进文章**。

## 0. 系列定位

承接已有主线：`rl-training-frameworks-survey`（全景导航，828 行）+ `rl-variables-in-frameworks`（变量侧源码对照）。
本系列做的是**源码级拆解**：每篇只回答「这个框架凭什么这样设计、它解决了什么、代价是什么」。
**不复述算法推导**（那是 `mdp-to-grpo-*` 的事），**不复述并行原理**（那是 `dist-train-*` 的事）。

## 1. 取证基准（已全部落到本机，可直接 `git show`）

| 仓库 | 本地路径 | 锁定基准 | commit |
| --- | --- | --- | --- |
| Megatron-LM | `~/Desktop/Projects/github/Megatron-LM` | `core_v0.19.2` | `4b4acac9a1d28ea6829c8d4f566d75698a21249d` |
| Megatron-Bridge | `~/Desktop/Projects/github/Megatron-Bridge` | `v0.6.2` | `c0e164ed2aedac4ad1c877780e2564a19d5d54ec` |
| verl | `~/Desktop/Projects/github/verl` | `v0.9.1` | `1876b06d0a3e4e71e06230be10af14492ca8a75b` |
| slime | `~/Desktop/Projects/github/slime` | `v0.4.0` | `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e` |
| sglang | `~/Desktop/Projects/github/sglang` | `v0.5.22` | `baa9befa381511cd4d56f83980951e5a45fffc6d` |
| vllm | `~/Desktop/Projects/github/vllm` | `v0.32.1rc0` | `2365177ddda7714502e4df7a1669cc79c0a4dfd0` |
| AReaL | `~/Desktop/Projects/github/AReaL` | `v2.1.0` | `ecc8b0e4dfb4e3965f67121a5c87866a07c390ae` |
| RLinf | `~/Desktop/Projects/github/RLinf` | `v0.4` | `2348ece3df113874671478be1360ae3115b346e3` |

**取证规则（硬性）**

1. 工作树的 HEAD **不等于**上面任何一个 commit。**只许 `git show "<commit>:<path>"` 读源码**，
   **不许 `git checkout` / `git pull` / `git switch`** —— 会毁掉别人正在读的树。
2. 行号必须用 `git show <commit>:<path> | sed -n 'A,Bp'` 或 `git grep -n <pat> <commit> -- <path>` 落到本地精确提取。
   **不许靠浏览器目测行号**。
3. permalink 一律写
   `https://github.com/<org>/<repo>/blob/<完整 40 位 commit>/<path>#L<起>-L<止>`
   组织名以 §1 表为准（verl 已迁至 **`verl-project`**，`volcengine/verl` 会 301）。
4. **每条行号自检**：`git show <commit>:<path> | sed -n '<起>,<止>p'` 打印出来的内容必须与正文描述一致。
   自检不通过就不许写进正文。

## 2. 跨篇统一的骨架（每个框架篇都按这个顺序，不许换序）

1. **设计出发点**：它要解决哪一类问题？为此放弃了什么？（要有出处：论文 / README / release note）
2. **模块初始化逻辑**：从入口脚本到「所有 worker 就绪」中间发生了什么
3. **进程与角色**：有几个进程/actor、谁持有权重、谁是控制面谁是数据面、rank/world size 怎么落地
4. **数据与样本流**：一条样本从哪来、怎么进训练、怎么被切分与聚合
5. **权重同步**：训练端 → 推理端到底搬了什么、搬多少、什么机制
6. **训推一致三层**（见 §3）
7. **异步形态**：按 §4 的四级定义先归位，再讲实现
8. **监控系统**：指标层 / 日志与追踪层 / 归因层
9. **性能 benchmark**：按 §5 分级，写明不可比原因
10. **小结：它解决到什么程度、还剩什么**

> 某些框架某一节天然为空（例如 Megatron 原生 RL 栈没有独立 refit）——**写明「此节为空的原因」，不要删节号**。

## 3. 训推一致三层（统一术语，不许自造）

| 层 | 内容 | 典型判据 |
| --- | --- | --- |
| **数值层** | 同一份权重在推理引擎与训练端的 logprob 偏移 | KV cache dtype、attention kernel、rope 实现、采样温度与 top-p 的应用顺序 |
| **权重层** | refit 是否真的完成 | 部分参数没同步、delta 同步的累积误差、并行形状重排后的数值等价性 |
| **样本层** | rollout 用的策略版本与训练用的版本是否同一个 | staleness 步数、partial rollout 的续用、队列深度 |

## 4. 异步形态四级定义（跨篇统一，先归位再讲实现）

| 形态 | 判定条件 | 代表 |
| --- | --- | --- |
| **同步** | 每步 rollout 全部完成后才更新，训练端与推理端轮流占用同一批卡 | TRL、OpenRLHF 默认 |
| **部分异步**（partial rollout） | 同一步内允许在途生成跨步续用，但**权重更新仍是全局屏障** | slime、veRL `sleep/resume` |
| **off-policy 异步** | 训练第 $$k$$ 步时 rollout 已在产第 $$k+1$$ 步，**同一组卡上流水** | veRL `fully_async_policy`、OpenRLHF async |
| **全异步**（服务解耦） | 训练端与推理端是**两个独立服务**，用样本队列连接，staleness 由队列深度控制 | AReaL |

## 5. 性能数字分级（本机无 GPU，一律不实跑）

每个数字必须带**等级标签**：

- `[官方]` 厂商 README / release note 自报
- `[论文]` 论文实测口径
- `[第三方]` 独立复现或第三方博客
- `[本机]` —— **本系列不存在此类，不许出现**

并且必须写明不可比原因：GPU 型号、并行度、是否 colocate、`rollout.n`、序列长度、是否含首次编译。
**跨框架比较只允许引用同一份 benchmark 内的数字**（例如 AReaL 论文里对 veRL/slime 的对比），
其余一律写成「厂商口径，不可横向比较」。

## 6. 社区活跃度快照（采集于 2026-10-24，统一引用这一份，不许各篇自己重算）

| 仓库 | stars | open issues | 最近 90 天 commit | 90 天贡献者 | 最新 release |
| --- | --- | --- | --- | --- | --- |
| vllm | 95,268 | 9,377 | 729 | 343 | `v0.32.1rc0` 2026-10-18 |
| sglang | 37,688 | 6,136 | 4,327 | 665 | `v0.5.22` 2026-10-13 |
| verl | 23,998 | 1,386 | 219 | 97 | `v0.9.1` 2026-09-20 |
| trl | 19,594 | 278 | — | — | `v1.16.0` 2026-10-15 |
| Megatron-LM | 18,257 | 1,696 | 48 | 20 | `core_v0.19.2` 2026-09-18 |
| OpenRLHF | 10,114 | 403 | — | — | `v0.11.2` 2026-09-14 |
| slime | 8,727 | 549 | 76 | 26 | `v0.4.0` 2026-10-09 |
| AReaL | 5,866 | 141 | 45 | 14 | `v2.1.0` 2026-08-25 |
| RLinf | 5,488 | 311 | 131 | 53 | `v0.4` 2026-10-20 |
| Megatron-Bridge | 969 | 492 | 140 | 16 | `v0.6.2` 2026-09-18 |

## 7. 边界：不许重复的既有内容

| 已有系列 | 已讲透的，不要再写 |
| --- | --- |
| `megatron-00..16` | 进程组与通信重叠（01）、EP（02）、TP/SP/GTP（03/04）、DP 与三套优化器（05）、PP 调度（06）、CP（07）、重计算与显存账（08）、FP8/FP4（09）、CUDA Graph（10）、refit（11）、容错（12）、模块规格（14）、模型装配（15）、Bridge 与导出（16） |
| `serving-01..08` | vLLM/SGLang 架构分歧（01）、调度与批处理（02）、支持矩阵（03/04）、PD 分离（05）、benchmark 交叉验证（06）、选型手册（07）、前缀命中率（08） |
| `dist-train-00..08` | 并行原理通论 |
| `rl-training-frameworks-survey` | 三族划分、四轴定位、各家一段式简介、算法矩阵 |
| `rl-variables-in-frameworks` | advantage / ratio / KL / reward / 损失聚合 / mask 的逐行对照 |
| `profiling-00..02` | TTFT/TPOT/goodput 口径 |

**Megatron 基准差异声明**：`megatron-00..16` 十八篇的行号锁在 `60e039626`（v0.20.0）。
本系列 Megatron 篇用 `core_v0.19.2`，**两者不同**。每篇引用 Megatron 的地方都要在首次出现处写明
「本文基准 `core_v0.19.2`，与 `megatron-` 系列的 `60e039626` 不同，行号不可跨篇比对」。

## 8. 五步校验（每篇写完必跑，全过才交）

```bash
cd ~/Desktop/Projects/github/lrypcy.github.io
python3 tools/check_liquid.py <FILE>      # 必须第一步，挡整站构建失败
python3 tools/check_post.py <FILE>
python3 tools/scan_garbled.py <FILE>
python3 tools/fix_quotes.py <FILE>        # 这是修复不是 dry-run，会直接改文件
python3 tools/check_links.py <FILE>
```

无实验 → 跳过「回填一致性比对」那一步（本系列不做配套实验）。

## 9. 写作红线（摘自 `.agents/WRITING.md`，逐条遵守）

- 不用第一人称叙述生成过程；不写「我做了个实验发现」
- **元叙述不入正文**：不写「本文所有行号锁定在 commit X 是因为……」「拉最新代码核对后才发现」「为了完整性补充」「本节经过 N 轮修订」
- 中文引号必须全角 `“ ”`
- 行内公式 `$$...$$`，禁裸 `|`（绝对值写 `\lvert x\rvert`）
- 不自造符号，沿用 `rl-variables-in-frameworks` 已定的那套
- Mermaid 只用 `<br>` 换行，节点含 `()[]{}` 要加引号
- **仓库里不写主目录绝对路径**，一律 `~/...`
- 没出处不写，没实验不编数
