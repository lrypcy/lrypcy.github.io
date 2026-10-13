# Train-Time Scaling / Scaling RL —— 资料补遗（对应 CS329A Lec 6）

> 用途：补上 [Notion 原笔记](https://app.notion.com/p/captaincaptain/Stanford-CS329A-Self-Improving-AI-Agents-Part-6-Train-Time-Scaling-Scaling-RL-3e6b1f9d11a080e68b7cf982df45b8c7)
> 断掉的后半段。那份笔记写到 Quiet-STaR 就停了，**DeepSeekMath 与 DAPO 只有标题没有内容**。
> 课程页：<https://cs329a.stanford.edu/>，本讲三篇必读：STaR / DeepSeekMath / DAPO。

---

## 0. 先把四种算力位置分开

讨论「怎么分配算力」之前，四种花的钱必须分开算，否则结论会串味：

| 阶段 | 算力花在何时 | 改的是什么 |
|:---|:---|:---|
| 预训练 | 部署前 | 权重里的通用表示与能力 |
| SFT | 部署前 | 从示范数据学到的行为 |
| Test-time scaling | 请求发生时 | 权重不动，靠搜索/重采样挑答案 |
| **Train-time scaling** | 迭代式后训练期间 | 把「上次采样到并被验证器认可的行为」写回权重 |

一句话区别：**test-time 是在「预训练模型已经能表达的东西」里搜；train-time 是提高「下次一次就采样到它」的概率。**
两者都需要验证器，但 train-time 还额外要求基座有**非零成功率**——爬坡得有个落脚点。

> 一条常被引用的经验曲线（o1 的 AIME 随训练算力与推理算力同时上升）只是**某一次具体实现**的观测，
> 不是普适效率定律。别把它写成 scaling law。

---

## 1. STaR 家族的位置（原笔记已有，补一条现代视角）

- **STaR**（[arXiv:2203.14465](https://arxiv.org/abs/2203.14465)）：少样本种子 → 生成 rationale → 只留答案对的 →
  错的把正确答案当 hint 让模型「倒着补证明」→ 用这些数据 SFT → 迭代。
  本质是 **off-policy 的 rejection-sampling bootstrap**，不是完整 RL：没有重要性比，没有负样本梯度。
- 三个隐含假设：**答案对 ≈ rationale 好**、**给答案后能反推出合理推导**、**基座已经能解一部分题**。
  第三条不成立时循环没有爬坡信号——这和后文 §4 的 RLVR 边界之争是同一件事。
- **V-STaR**（2402.06457）：加独立 verifier，把对/错路径配成偏好对用 DPO 训，**让错样本也有信息量**，推理时做重排序。
- **Quiet-STaR**（2403.09629）：把「思维」从输出层挪到每个 token 之前，REINFORCE 奖励那些**提高下一 token 预测概率**的 thought。

**现代视角**：STaR 的 neighbor 是 RFT / iterative rejection-sampling fine-tuning；
RLVR（GRPO 及变体）做的就是「STaR 的监督信号 + 负样本的显式梯度项 + 序列级 importance ratio + clip」。
即 STaR → RFT → GRPO → DAPO 是一条连续光谱，差别只在**负样本怎么用、要不要重要性比、损失怎么归一化**。

---

## 2. DeepSeekMath + GRPO（原笔记缺失的第二篇）

> [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)，Shao et al.，DeepSeek-AI。

### 2.1 两个贡献，别混

一是**数据**：用 fastText 分类器（OpenWebMath 正例 vs 随机 CC 负例）迭代挖 Common Crawl，四轮后得到
**3550 万页面 / 120B token** 的 DeepSeekMath Corpus；并用 URL 路径标注 + 去重 + 10-gram 去污染。
起点特意选了 **DeepSeek-Coder-Base-v1.5 7B**（先训过代码比通用 LLM 更适合做数学）。
**Base 7B**：GSM8K 64.2%、MATH 36.2%（超过 Minerva 540B 的 33.6%）。

二是**算法 GRPO**：GRPO = **去掉 critic 的 PPO**。对每个问题采样 $$G$$ 条回答，用组内统计量当 baseline：

$$\hat{A}_i = \frac{R_i - \mathrm{mean}(\{R_i\}_{i=1}^{G})}{\mathrm{std}(\{R_i\}_{i=1}^{G})}$$

再套 PPO 的 clipped surrogate，KL 惩罚从**奖励里挪到损失里**（不再是 per-token KL reward）。
省掉的是一个**通常与策略同规模的价值网络**，代价是每条 prompt 必须多采样。
原文一个重要发现：**RL 主要抬高 $$\mathrm{Maj}@K$$，基本不动 $$\mathrm{Pass}@K$$**——
RL 让「多次里有正确解」这件事更稳，而不是让模型会解原来解不了的题。

### 2.2 结果（只用英文 CoT 数据做 RL）

| 指标 | Instruct 7B | + GRPO |
|:---|:---|:---|
| GSM8K | 82.9% | **88.2%** |
| MATH | 46.8% | **51.7%** |
| CMATH（域外） | 84.6% | 88.8% |

自一致性 64 样本下 MATH **60.9%**。作为参照，同期 GPT-4 在 MATH 上是 52.9%、Gemini Ultra 53.2%。

> ⚠️ **勘误**：课程 slide 把 51.7% 标成了 AIME。**这是 MATH 的分数**，AIME 的 50% 是 DAPO 的结果。
> 写文章时别跟着错。

### 2.3 值得抄的一个分析工具

论文 §5.2 把 **SFT / RFT / DPO / PPO / GRPO 写成同一个梯度形式**：每个 token 的 score function
乘一个**方法专属的梯度系数**。这个视角把设计选择拆成三个正交问题：
① 训练用的输出从哪来（自己的 / 示范的）→ ② 谁提供奖励（标注 / 参考模型 / 规则）→
③ 奖励怎么变成系数（0/1 硬门 / 偏好差 / 优势）。
这个统一范式对后续写「算法谱系」类文章非常好用。

> 训练配置（$$G=64$$、lr $$1\times10^{-6}$$、$$\beta=0.04$$）来自二手整理，写进正文前建议回 arXiv v3 附录核对。

---

## 3. DAPO（原笔记缺失的第三篇，最重要的工程增量）

> [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)，ByteDance Seed + 清华 AIR。代码/数据全开源，基于 verl。

**动机**：直接用 GRPO 在 Qwen2.5-32B 上跑 AIME 只有 **30 分**（同期 DeepSeek-R1-Zero-Qwen-32B 是 47 分）。
差距不在算法名字上，在四个工程细节上。

### 3.1 目标函数

$$\mathcal{J}_{\mathrm{DAPO}}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}\left[ \frac{1}{\sum_{i=1}^{G}\lvert o_i\rvert}\sum_{i=1}^{G}\sum_{t=1}^{\lvert o_i\rvert} \min\Big( r_{i,t}(\theta)\hat{A}_{i,t},\ \mathrm{clip}\big(r_{i,t}(\theta),\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}}\big)\hat{A}_{i,t} \Big) \right]$$

$$\text{s.t.}\quad 0 < \big\lvert\{o_i \mid \mathrm{is\_equivalent}(a,o_i)\}\big\rvert < G$$

相对 GRPO 的四处改动：**去掉 KL**、**上下 clip 解耦**、**损失按 batch 内 token 总数归一化**、**加动态采样约束**。

### 3.2 四个 trick

| trick | 解决什么 | 具体做法 |
|:---|:---|:---|
| **Clip-Higher** | 熵崩塌、探索退化 | $$\varepsilon_{\mathrm{low}}=0.2,\ \varepsilon_{\mathrm{high}}=0.28$$。上界放大给低概率（探索性）token 留出提升空间；下界收紧防分布塌缩 |
| **Dynamic Sampling** | 全对/全错的组优势全为 0，梯度为空 | 过采样后过滤掉组内正确数 $$=0$$ 或 $$=G$$ 的 prompt，直到缓冲区填满；**等于把算力从没信号的题上挪走** |
| **Token-Level Loss** | 长 CoT 被稀释 | GRPO 是 $$\frac{1}{G}\sum_i \frac{1}{\lvert o_i\rvert}\sum_t$$（每条样本等权，短回答单 token 权重更大）；DAPO 改成 $$\frac{1}{\sum_i\lvert o_i\rvert}\sum_i\sum_t$$，**长链条里的每个 token 说话音量相同** |
| **Overlong Reward Shaping** | 截断样本的奖励噪声 | Soft Overlong Punishment，长度惩罚：$$\lvert y\rvert \le L_{\max}-L_{\mathrm{cache}}$$ 为 0；在 $$[L_{\max}-L_{\mathrm{cache}},\,L_{\max}]$$ 区间线性增到 $$-1$$；超出 $$L_{\max}$$ 固定 $$-1$$ |

### 3.3 超参（§4.1，可直接写进复现表格）

- 数据集 **DAPO-Math-17K**（AoPS 等来源，答案统一转成整数便于规则判分）
- $$G=16$$；rollout prompt batch **512**；mini-batch **512**（每个 rollout step 做 16 次梯度更新）
- AdamW，常数 lr **$$1\times10^{-6}$$**，20 个 rollout step 线性 warm-up
- $$L_{\max}=20{,}480$$ token，$$L_{\mathrm{cache}}=4{,}096$$ token（即 16,384 以后开始软惩罚）
- 评测：AIME 重复 32 次取 avg@32，temperature 1.0，top_p 0.7

### 3.4 逐层消融（AIME 2024，avg@32）

| 配置 | 分数 |
|:---|:---:|
| Naive GRPO 基线 | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-Level Loss | 42 |
| **+ Dynamic Sampling（完整 DAPO）** | **50** |
| *参照：DeepSeek-R1-Zero-Qwen-32B* | *47* |

50 分还只用了对方一半的训练步数。**启示**：顶层四分在这条基线上几乎全部来自「先把 GRPO 用对」，
而不是换了什么新算法。这也解释了为什么复现工作普遍卡在 30 分附近。

---

## 4. 争到现在的核心问题：train-time RL 到底能不能扩展边界

笔记里那句 “Teaching the model new fundamental capabilities” 和旁边那个疑问 “what fundamental?”，
正是 2025–2026 最吵的一件事。三方实验互不买账：

| 立场 | 出处 | 主张 | 关键证据 |
|:---|:---|:---|:---|
| **不能，RLVR 只提高采样效率** | [arXiv:2504.13837](https://arxiv.org/abs/2504.13837)（清华 LeapLab + 上海交大，NeurIPS 2025 Best Paper Runner-Up） | RLVR **没有**扩展基座已有的推理能力边界，只是把分布锐化到「基座本来就会」的题上 | $$\mathrm{pass}@k$$：小 $$k$$ 时 RL 模型领先；$$k$$ 涨到几十上百时**基座反超**（Minerva 32B 基座在 $$k=128$$ 高约 9 个点）。训练越久，训练集 $$\mathrm{pass}@1$$ 涨（26.1→42.5）而 $$\mathrm{pass}@256$$ 反降。PPO / GRPO / REINFORCE++ **三者采样效率差距很小**，都没到基座边界定义的最优；**只有蒸馏能扩展可解题集合** |
| **能，前提是训得够久** | [arXiv:2505.24864](https://arxiv.org/abs/2505.24864) **ProRL**（NVIDIA） | 长时间稳定 RL 能发现基座模型在大 $$k$$ 采样下也拿不出的**新推理策略** | 配方三件套：**KL 控制 + 参考策略周期性重置 + 任务多样性**（数学/代码/STEM/逻辑谜题/指令跟随，136k 题）。边界扩展的幅度与**基座任务能力、训练时长**两者强相关 |
| **瓶颈不是 RL 的上限，是探索不足** | [arXiv:2510.01180](https://arxiv.org/abs/2510.01180) **BroRL**（NVIDIA + Stanford/UW） | 每步加 Rollout 数 $$N$$（16 → **512**）比加步数更有效 | 质量守恒分析：已采样 token 总在扩张正确概率质量，未采样空间的贡献符号不定；$$N$$ 足够大时未采样项噪声被平均掉，净变化为正。在已饱和的 ProRLv2（3000 步）上继续训：ProRL 停滞甚至回退，BroRL 持续涨。且吞吐 36.5 → 72.4 sample/s（生成从访存受限变计算受限），动态采样通过率 41% → 62%，**token 用量反而下降**（数学 −745、代码 −717） |

**怎么读这张表**：三方并不完全是同一命题。
「不能」那方的判据是 0/1 二值奖励 + 无价值网络的 GRPO 族；
「能」的两方强调的是**长时间/宽探索 + 稳定化手段**（KL、参考重置、任务多样性、大 $$N$$）。
2504.13837 的作者自己也在网站上承认：结论限于 0/1 reward 设定，并列出仍有可能突破的方向——
稠密/过程奖励、带 value network 的信用分配、数据与课程、更强的探索机制、Agent 式工具使用。

**可以写进文章的落点**：train-time scaling 的收益大致可以分成两块，
①「把基座已有的正确路径变得更可采样」（$$=$$ 提高 $$\mathrm{pass}@1$$ / $$\mathrm{Maj}@K$$，这是稳赚的部分），
②「真的扩展可解题集合」（$$=$$ 抬高 $$\mathrm{pass}@k$$ 在大 $$k$$ 处的值，有争议且依赖上述条件）。
评估第二件事时**必须看大 $$k$$ 的 $$\mathrm{pass}@k$$**，只看 avg@n 会得出相反的结论。

---

## 5. 与本站现有文章的接缝

| 本篇材料 | 该往哪写 | 备注 |
|:---|:---|:---|
| GRPO 组内优势、DAPO 四 trick 的 loss 归一化 | `2026-09-19-rl-credit-assignment-settlement.md` §3「序列级隐形信用分配」 | token-level vs sequence-level 归一化那张表正好可以接 §3.3 |
| Clip-Higher 的上下界解耦 | 同上 §3 的 clip-higher/CISPO 段 | $$\varepsilon_{\mathrm{low}}/\varepsilon_{\mathrm{high}}$$ 用原论文记号，别自造字母 |
| Overlong Reward Shaping 的分段公式 | 同上 §5「长度惩罚定标」 | 可以直接抄 §3.2 的公式做标定例子 |
| DeepSeekMath §5.2 的统一梯度范式 | `2026-10-05-rl-advantage-estimation-lineage.md` | 与现有四支坐标系正好互补：那边按 $$\Psi_t$$ 形状 × 信息源分，这边按「奖励怎么变成系数」分 |
| $$\mathrm{Maj}@K$$ 涨 / $$\mathrm{Pass}@K$$ 不涨 + §4 三方对峙 | 值得单独成一节：*train-time scaling 的收益分解* | DAPO / ProRL / BroRL 三张消融表天然是同一叙事的三个数据点 |
| STaR → RFT → GRPO → DAPO 光谱 | 可放在上述节的开头做动机 | 强调差别只在「负样本怎么用」 |

---

## 6. 参考与待核

**一手**：
- STaR <https://arxiv.org/abs/2203.14465>｜V-STaR <https://arxiv.org/abs/2402.06457>｜Quiet-STaR <https://arxiv.org/abs/2403.09629>
- DeepSeekMath <https://arxiv.org/abs/2402.03300>
- DAPO <https://arxiv.org/abs/2503.14476>（项目页 <https://dapo-sia.github.io/>）
- RLVR 边界 <https://arxiv.org/abs/2504.13837>（项目页 <https://limit-of-rlvr.github.io/>，含 Q&A）
- ProRL <https://arxiv.org/abs/2505.24864>｜BroRL <https://arxiv.org/abs/2510.01180>
- 相关但本站已覆盖：Dr. GRPO 2503.20783、GSPO、RLOO、REINFORCE++

**待核清单**（写正文前必须回到原文）：
- [ ] DeepSeekMath 的 $$G$$、lr、$$\beta$$ 具体数值 → 核 arXiv v3 附录
- [ ] BroRL 各 benchmark 分数与耗时 → 核 arXiv §4 的 Table（当前数字来自 NVIDIA blog）
- [ ] 2504.13837 的 Minerva 32B / $$k=128$$ 差值 → 核原文 Figure
- [ ] ProRL v2 与 BroRL 的关系（blog 里的 v2 是否单独出过技术报告）
