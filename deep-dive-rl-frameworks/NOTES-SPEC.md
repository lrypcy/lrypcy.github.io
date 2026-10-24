# rl-frameworks 深度改写：调研规范

本文件是调研子代理的工作契约。**不要写文章**，只产出结构化的实现细节素材。

## 0. 深度标准（这一轮改写的核心）

上一版文章被指出「过于简单、缺少实现细节、缺少建模图」。本轮要求：

1. **必须有代码走读**：贴出关键函数的真实片段（≤15 行），标 `[行号]` 与 permalink
2. **必须有量化建模**：给出公式、复杂度、显存/时间/通信量的算式，并代入具体规模算一遍
3. **必须有状态机 / 时序图素材**：Mermaid 的 `stateDiagram-v2` / `sequenceDiagram` / `flowchart`
4. **必须有「为什么这么设计」的实现理由**：优先引用源码注释原文，其次引用官方文档
5. **必须有边界与坑**：静默失效、性能悬崖、不支持的组合

**判据**：一个读者照着你的素材，能在自己的机器上定位到那一行代码、算出那个数字、复现那个状态。

## 1. 取证基准（已全部在本机，可 `git show`）

| 仓库 | 本地路径 | 基准 commit |
| --- | --- | --- |
| Megatron-LM | `~/Desktop/Projects/github/Megatron-LM` | `4b4acac9a1d28ea6829c8d4f566d75698a21249d`（core_v0.19.2） |
| Megatron-Bridge | `~/Desktop/Projects/github/Megatron-Bridge` | `c0e164ed2aedac4ad1c877780e2564a19d5d54ec`（v0.6.2） |
| verl | `~/Desktop/Projects/github/verl` | `1876b06d0a3e4e71e06230be10af14492ca8a75b`（v0.9.1） |
| slime | `~/Desktop/Projects/github/slime` | `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`（v0.4.0） |
| AReaL | `~/Desktop/Projects/github/AReaL` | `ecc8b0e4dfb4e3965f67121a5c87866a07c390ae`（v2.1.0） |
| RLinf | `~/Desktop/Projects/github/RLinf` | `2348ece3df113874671478be1360ae3115b346e3`（v0.4） |
| SGLang | `~/Desktop/Projects/github/sglang` | `baa9befa381511cd4d56f83980951e5a45fffc6d`（v0.5.22） |
| vLLM | `~/Desktop/Projects/github/vllm` | `2365177ddda7714502e4df7a1669cc79c0a4dfd0`（v0.32.1rc0） |

**规则**

- 只读：`git -C <repo> show "<commit>:<path>"` 或 `git -C <repo> grep -n <pat> <commit> -- <path>`
- **绝不** `checkout` / `pull` / `switch` / `commit` / `stash`——多个代理共享同一个工作树
- **文档也要读**：仓库里的 `docs/` 是官方对设计的自述，优先级高于外部 blog
- 行号必须用 `git show <commit>:<path> | sed -n 'A,Bp'` 落到本地，核对内容后再写进素材

## 2. veRL 官方文档清单（已确认存在）

`docs/advance/` 下：`placement.rst`、`fully_async.md`、`v1_async_trainer.md`、`one_step_off.md`、
`agent_loop.rst`、`reward_loop.rst`、`rollout_trace.rst`、`delta_weight_sync.md`、`determinism.md`、
`megatron_extension.rst`、`fsdp_extension.rst`、`dynamic_schedule.md`、`skip_manager.rst`、`checkpoint.rst`
`docs/algo/` 下：`ppo.md`、`grpo.md`、`dapo.md`、`gpg.md`、`entropy.md`、`opd.md`

## 3. 每篇需要的素材格式

每个小节按下面的模板交给 Lead：

```markdown
### <小节标题>

**结论**：一句话。

**代码**（`<path>` L<a>-L<b>）：
```python
<≤15 行真实片段>
```

**读法**：这段在做什么、为什么这么做（优先引注释原文）。

**量化**：（如有公式或数字）

**图**：（Mermaid 源码，标注应画什么）

**坑**：（如有）
```

## 4. 输出

写到 `deep-dive-rl-frameworks/notes/<篇号>-<主题>.md`，然后在回报里给出：
文件路径、覆盖的小节清单、**发现的三个最有价值的非显然事实**、没能验证的疑点。

## 5. Mermaid 硬约束（站内规范）

- 只用 `<br>` 换行，**禁止字面 `\n`**
- 节点标签含 `()[]{}` 等字符要加引号
- 不用 `▶ ◄ ► ↓ ∪ 『』 ⇏ ✔ ✘`（`scan_garbled.py` 会报）；可用 `▲ ▼ → ← ─ │ └ ├ ┬ ┴ ┼`
