# Progress Tracking

> 结构已按"收紧成单篇"调整：不产 01–07 七篇，一篇文章进 `_posts/`，本目录只留索引。

## 状态

| 项 | 状态 | 说明 |
|:---|:---:|:---|
| `RESEARCH_PLAN.md` | ✅ | v2（2026-09-19）：术语与形式化、三条谱系时间线、Cascaded vs Joint、双轴评测地图、12 条开放问题、Q1–Q12、产出结构、未验证清单、论文索引 |
| `README.md` | ✅ | 索引页：指向正文 + 概念速查 + 核心论文表 + 评测速查 |
| `PROGRESS.md` | ✅ | 本文件 |
| 文献调研（2025-2026 WAM 现状） | ✅ | 已核实 WAM Survey、DreamZero、Cosmos Policy、GE-Sim V2 / WorldArena、Genie 3 / Cosmos、Dreamer V3/V4、TD-MPC2 / IRIS、驾驶线 |
| **`_posts/2026-09-05-world-model-world-action-model.md`** | ✅ | **正文已完成**：787 行 / 3.26 万字符，4 个 Lab 全部实跑并回填真实输出 |

## 正文结构（与 RESEARCH_PLAN §7 对齐）

| 章节 | 内容 | 回答 |
|:---|:---|:---|
| §0 TL;DR | 5 条结论 | — |
| §1 定义与形式化 | 三式 $p(a\mid o,l)$ / $p(o'\mid o,a)$ / $p(o',a\mid o,l)$；判定边界四条反例；变量映射表；"为什么是 2026"三前提 | Q1 |
| §2 三条谱系 | latent MBRL（Dreamer V3/**V4**、IRIS、TD-MPC2）+ **Lab 1**；生成式（Genie 3 / Cosmos / 驾驶线 / GE-Sim V2）；VLA→WAM（Cosmos Policy / DreamZero）+ 时间线 mermaid | Q3, Q9, Q11 |
| §3 架构 Cascaded vs Joint | **latent frame injection 详解 + Lab 2**；条件掩码三功能表；best-of-N 与双 checkpoint；DreamZero 联合流匹配 + KV cache 写回；对照表 | Q2, Q4, Q5 |
| §4 评测双轴 | 世界建模三层次 + WorldArena + 策略基准家族 + **"世界模型当裁判" Lab 3（选优效率口径）** | Q8 |
| §5 工程 | **模型误差累积 Lab 4**；38× 加速拆解；数据混合四类；成本；**与信用分配的接口 §5.3** | Q6, Q10, Q12 |
| §6 开放问题与选型 | 12 条开放问题表 + "该不该上 WAM"决策表 + 最小可行路线 | Q7, Q12 |
| §7 附录 | 论文索引 + 未验证清单 + Lab 清单 | — |

## 本轮新增核实的事实（写作前补的原文精读）

- **Dreamer V4**（arXiv:2509.24527，已精读方法与实验部分）：
  - 三阶段：世界模型预训练 → Agent 微调（插入 task token 长出 policy/reward/value 头）→ 想象训练。
  - **shortcut forcing** 明确在**数据空间**去噪，理由是"prevent accumulating errors caused by high-frequency network outputs"；训练时把步数 $d$ 作为条件，$K=4$ 次前向生成一帧 → 单 GPU 实时。
  - tokenizer 用 masked autoencoding、causal attention 做时间压缩；多模态损失用 running RMS 归一。
  - 离线挖钻：需选 **20,000+** 鼠标键盘动作序列；比 OpenAI VPT offline agent 少用 **100×** 数据。
  - 数据混合 50% uniform + 50% relevant；BC loss 只在 relevant 上，dynamics loss 只在 uniform 上（避免乐观生成）。
- **Cosmos Policy**（arXiv:2601.16163，已核对摘要与 §4/附录）：
  - LIBERO **98.5%** / RoboCasa **67.1%** 是**摘要原文数字**（之前标"待验证"的这条可以去掉）。
  - 11 帧潜序列布局（双第三方相机 + 腕相机）：blank / proprio / wrist / cam1 / cam2 / **action chunk** / future proprio / future wrist / future cam1 / future cam2 / **future value**。
  - 训练数据混合 50/25/25（策略 / 世界模型 / 值函数）；噪声分布改为 0.7 log-normal + 0.3 U[1,85]；推理 $\sigma_{\min}$ 从 0.002 提到 4。
  - **规划模式约 5 秒生成一个 action chunk**（论文自己写的限制）；直接模式 5 步去噪 + 并行解码。
  - best-of-N 用两个 checkpoint（policy model + planning model），3×5=15 次预测 + majority mean 聚合。

## Lab 实跑结果（环境：miniconda python / numpy 2.1.1）

| Lab | 关键输出 |
|:---|:---|
| L1 §2.1 | 逐样本梯度 max/min：MSE **2.63e6** → symlog **4.26e2** → two-hot **6.07**；小量级子集（21.4%）相对误差 MSE **259.7** vs symlog **0.786** |
| L2 §3.2 | 11 帧潜序列 shape `(11,16,28,16)`；action chunk 16×7=112 → 复制 64 次填满 7168；反解无需 VAE 解码 |
| L3 §4.3 | 选优效率：eps=0.02 全程 **1.000**；eps=0.05 **0.955–0.984**；eps=0.10 **0.78–0.91**（Pearson 0.84） |
| L4 §5.1 | 单步误差 0.02 下 32 步误差：ρ=0.90 **0.16**（有界）/ ρ=1.00 **9.20** / ρ=1.05 **42.69** |

## 踩坑记录

- **第一个 symlog 实验设计错了**：用标量预测去拟合跨量级分布，三种 loss 的相对误差都是 1.0，无区分度。改成"带特征的回归 + 按量级分桶"才有信号。
- **想象步数扫描实验失败两次**：随机采样 shooting 的规划器在长视野下反而更差（选出的是"后期运气好"的序列，第一步动作是随机的）；换成 CEM 后结论仍被噪声淹没。最终放弃"扫描 H 找最优"，改成**两个确定性更强的实验**（误差累积的谱半径分析 + 裁判可靠性的选优效率）。
- **教训**：要证明"模型误差随 H 累积"这种机制，直接测误差增长比测下游任务回报干净得多。
- **粘贴输出前必须重跑正文里的代码**：本次因为文章里的代码块做了简化（变量顺序、`acts` 生成位置、`make_model` 是否扰动 B），4 个块里有 3 个的输出与 /tmp 版本不同，逐个校准后才一致。

## 剩余待办

- [ ] DreamZero 的 62.2% / 39.5% / +42% / 38× 数字仍为二手来源，未复现（已在 §7.2 标注）
- [ ] Cosmos Policy 的"每任务 50 条演示"与"规划 +12.5%"未见于摘要，标为待验证
- [ ] 需要 GPU 的实验（真实 LIBERO/RoboCasa 跑分、跨本体视频迁移）未做，正文给了 §6.3 最小可行路线
