# Deep Dive：世界模型（World Model）与世界动作模型（World Action Model）

> 覆盖文献：2018 ~ 2026-09 ｜ 版本：v2（2026-09-19）

## 正文

**→ [《世界模型与世界动作模型（WAM）：从 Dreamer V4 到 Cosmos Policy / DreamZero 的完整地图（2025–2026）》](/2026/09/05/world-model-world-action-model/)**

单篇长文（787 行 / 3.3 万字符），进 `_posts/`，由 GitHub Pages 自动构建。

## 本目录文件

| 文件 | 作用 |
|:---|:---|
| `RESEARCH_PLAN.md` | 研究计划 v2（术语与形式化、三条谱系、Cascaded vs Joint、双轴评测、12 条开放问题、Q1–Q12、未验证清单、论文索引） |
| `PROGRESS.md` | 进度与调研确认的关键事实 |
| `README.md` | 本文件：指向正文 + 概念速查 |

## 概念速查

| 术语 | 形式化 | 一句话 |
|:---|:---|:---|
| **VLA** | $p(a\mid o,l)$ | 观测+语言直接映射动作，反应式，不显式表示未来 |
| **World Model** | $p(o'\mid o,a)$ | 预测世界如何演化，本身不是策略 |
| **World Action Model** | $p(o',a\mid o,l)$ | 未来状态合成与可执行动作在一个策略框架内联合学习 |
| **判定边界** | — | 未来状态预测必须是**策略的一部分**，而非辅助 backbone 或外部模拟器 |
| **Cascaded** | 先 $o'$ 再 $a$ | 模块化清晰，瓶颈在两级耦合 |
| **Joint** | 联合监督 | 自回归 / 扩散式，问题是耦合怎么做 + 延迟 |
| **latent frame injection** | — | 把本体/动作/未来/值编成潜帧塞进视频扩散序列，**不改架构** |
| **shortcut forcing** | — | Dreamer V4 的动力学目标，在数据空间去噪 + 步数条件，K=4 前向/帧 |

## 三条谱系

1. **Latent 世界模型 / MBRL** — Dreamer V3（Nature 2025，symlog+two-hot+固定超参）、Dreamer V4（2025-09，纯离线挖钻）、IRIS、TD-MPC2
2. **生成式视频世界模型** — Genie 3、NVIDIA Cosmos、驾驶线（Vista/HERMES/GAIA-2）、GE-Sim V2（WorldArena Track1 冠军）
3. **VLA → WAM** — π0/GR00T → **Cosmos Policy**（arXiv:2601.16163）、**DreamZero**（arXiv:2602.15922）

## 核心 WAM 论文

| 模型 | arXiv | 关键事实 |
|:---|:---|:---|
| **WAM Survey** | [openmoss.ai/Awesome-WAM](https://openmoss.ai/Awesome-WAM/) | 首个 WAM 系统综述：定义、Cascaded vs Joint、评测、7 条 open challenges |
| **Cosmos Policy** | [2601.16163](https://arxiv.org/abs/2601.16163) | latent frame injection；LIBERO 98.5% / RoboCasa 67.1%；单阶段后训练无架构修改 |
| **DreamZero** | [2602.15922](https://arxiv.org/abs/2602.15922) | 14B Wan2.1-I2V 骨干，video+action 联合 flow matching，KV cache 写回；38× → 7Hz |
| **Dreamer V3** | [2301.04104](https://arxiv.org/abs/2301.04104) | Nature 2025；symlog + two-hot + 回报归一化 |
| **Dreamer V4** | [2509.24527](https://arxiv.org/abs/2509.24527) | shortcut forcing；首个纯离线拿到 Minecraft 钻石，数据少 100× |
| **GE-Sim V2** | [2605.27491](https://arxiv.org/abs/2605.27491) | WorldArena Track1 冠军 68.26 分；用 rollout 对比+混淆矩阵验证"世界模型当裁判" |

## 评测速查

- **世界建模三层次**：视觉保真（PSNR/SSIM/LPIPS/DINO/FVD）→ 物理常识（WorldModelBench / Physics-IQ / WorldScore / EWMBench）→ 动作可信性（WorldSimBench / IDM Turing Test）
- **综合榜**：WorldArena（CVPR 2026，16 指标 + 3 任务）
- **策略基准**：LIBERO、RoboCasa、ManiSkill、RLBench、CALVIN；RoboTwin / HumanoidBench；BEHAVIOR-1K；真机 RoboArena / RoboChallenge
- **本文提出的口径**：**选优效率 = 模型挑出的 top-k 真实回报增益 / 全知 oracle 的 top-k 增益**（见正文 §4.3）

## 相关（本站）

- [RL 信用分配与奖励系数](/2026/09/19/rl-credit-assignment-settlement/) — 想象轨迹中的信用分配、模型误差与 λ-return（正文 §5.3 交叉引用）
- [具身智能 05：VLA](/2026/08/25/embodied-ai-05-vla-foundation-models/) — WAM 的前身
