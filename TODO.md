# TODO

> 研究待办清单。本文件不发布（已加进 `_config.yml` 的 `exclude`）。
> 状态：⬜ 待开始 ｜ 🟨 进行中 ｜ ✅ 完成

---

## 1. SFT 训练到什么程度可以做 RL ⬜

**核心问题**：什么时候该停 SFT、切 RL？判据是什么？切早了 / 切晚了各自损失什么？

**待查证的子问题**

- [ ] 收敛判据：验证 loss 平台期、下游任务 pass@k 饱和、格式/指令遵循率达标，哪个是真正可用的信号？
  - 关注「SFT 过拟合之后再 RL 会怎样」——是 RL 修得回来，还是容量被锁死？
- [ ] SFT 阶段数与 RL 的衔接：多轮 SFT（rejection sampling / STaR 式迭代）与 RL 的边界在哪？
  迭代式 SFT 是不是 RL 的低样本效率近似？
- [ ] 数据规模与质量：多少条 SFT 数据、什么质量分布下 RL 才有边际收益？
  （LIMA 式「少而精」 vs 大规模 SFT，两者后续 RL 的收益曲线是否不同）
- [ ] 基础模型能力门槛：数学/代码类任务上，SFT 到什么水平 RL 才「爬得动」？
  基座完全不会的任务，RL 是学不会还是只是学得慢？（ reward 稀疏 vs 能力天花板）
- [ ] 可验证奖励（RLVR）场景下 SFT 是否还需要「做满」？
- [ ] 工程判据：RL 前要不要冻结 SFT 的某些层 / 保留 KL 锚点的强度怎么定？

**参考线索**

- 站内已有：MDP → GRPO 系列 01–05、蒸馏与 RL 的统一谱系、`2026-09-19-rl-credit-assignment-settlement`
- 待查：DeepSeek-R1 / Kimi / Qwen 技术报告里关于 SFT→RL 阶段切换的描述，需锁版本引用
- 待查：是否有论文系统做过「SFT 程度 × RL 收益」的消融（找不到就标注「未见系统研究」）

**产出**：一篇 Deep-Dive，落到 `_posts/`，默认单篇长文直进（500+ 行 / 10 个一级章节）。
有可跑的判据就配 companion lab（放 ipynbs 仓库），没有就不凑。

---

## 2. SFT loss 与 PPO loss 怎么协同计算 🟨

**核心问题**：两个 loss 同处一个训练步时，梯度怎么合、量纲怎么对齐、谁主谁辅？

**待查证的子问题**

- [ ] 组合形式盘点：加性混合（$$\mathcal{L} = \mathcal{L}_{\rm PPO} + \lambda \mathcal{L}_{\rm SFT}$$）、
  两阶段交替、预热后切换，各自适用什么场景？
- [ ] 「SFT loss 作为 PPO 的辅助项」到底在正则什么？
  常见说法是防灾难性遗忘 / 保格式 / 稳熵，各自有没有实验支撑？
- [ ] 系数 $$\lambda$$ 的量纲问题：SFT 是 token 级 NLL（每 token 一份），PPO 是序列级 surrogate
  （整条再乘 advantage）——**直接相加是不是量纲不一致**？常见实现里各怎么归一化？
- [ ] SFT 项用哪些 token：全序列？只算 prompt？只算正样本（RAFT / 拒绝采样式）？
  用 on-policy 采样出的序列算 SFT loss 时，梯度与 off-policy 演示数据有何不同？
- [ ] 梯度冲突与工程处理：两个 loss 的梯度方向相反时怎么办（PCGrad / 梯度裁剪 / 分参数更新）？
  actor 与 critic 的更新频率、loss 权重是否要分开调？
- [ ] 与 KL 惩罚项的关系：SFT loss 项和 KL-to-SFT-reference 是重复正则还是互补？
- [ ] 框架实现对齐：TRL / verl / OpenRLHF 里这一项分别叫什么、默认值多少？

**参考线索**

- 站内已有：`2026-09-19-rl-notation-and-estimators-deepdive`（符号对照表先对齐，别自造字母）、
  `2026-09-30-rl-variables-in-frameworks`、`2026-09-29-rl-training-frameworks-survey`
- 待查：InstructGPT 的 pretrain gradient 混合（原文有没有给出混合系数与动机）、
  RLHF 开源实现里的 `sft_loss` / `aux_loss` 命名与默认权重（锁 commit 引用）
- 可跑实验：小模型上做 $$\lambda$$ 扫描 + 梯度余弦相似度统计，验证「梯度冲突」是不是真问题

**产出**：先做 $$\lambda$$ 扫描的 pilot 实验拿到数字，再决定写不写；写法同 §1。

---

## 备注

- 两条都属于 RL 主线，动笔前先读 `.workbuddy/memory/topics/series-boundaries.md` 确认不与已有系列撞车
- 公式一律用来源文献的通行写法，不自造符号；行内公式用 `$$...$$`，写完跑 `python3 tools/check_liquid.py`
