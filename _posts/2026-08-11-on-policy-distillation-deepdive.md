---
title: "On-Policy Distillation 深度剖析：把 Agent 的探索成本外包给老师"
date: 2026-08-11 14:00:00 +0800
categories:
  - AI Infra
tags: [opd, distillation, llm, agent, rl, verl, retool]
layout: post
mathjax: true
---

> **TL;DR**
>
> * **OPD 回答了 post-training 的成本问题**：Agentic RL 贵在「多轮 rollout × 环境交互 × 奖励模型」三件套。On-Policy Distillation 把探索外包给老师——**学生自己走前缀（on-policy 状态分布），老师在前缀上补全并提供逐 token 监督**。学生学到的是「在我自己会遇到的状态下，老师会怎么做」，而不是「老师在它的状态分布里怎么做」——这是它与传统蒸馏的本质区别。
> * **两个反直觉发现**（清华 OPD，[arXiv:2604.13016](https://arxiv.org/abs/2604.13016)）：① 弱到强反向蒸馏实验显示，同家族的 1.5B 和 7B 老师，从学生视角看**分布上不可区分**——「更聪明的老师」未必是更好的蒸馏源；② 成功的 OPD 本质是**高概率 token 的渐进对齐**，一个小的共享 token 集合集中了 97%-99% 的概率质量——OPD 的"免费午餐"（dense token 级奖励）有隐性代价，能否扩展到长视野蒸馏存疑。
> * **Agent 场景的落地**（ReOPD，[arXiv:2607.04763](https://arxiv.org/abs/2607.04763)）：多轮 OPD 有「前缀陷阱」——让历史更贴近学生，会让学生在老师**目标不可靠**的位置上查询老师。ReOPD 用预收集的教师轨迹做 **prefix replay**（采样位置 \(p_t \propto \kappa^t\)，\(\kappa=0.6\)），实现**零环境交互、4-9× 更快的 rollout**，准确率不降反升。与 ToolRL（工具奖励）、Search RL（搜索动作）一起，构成 2025-2026 Agentic RL 训练的第三条技术路线。

```mermaid
flowchart LR
    subgraph OFF["off-policy 蒸馏（传统）"]
        A1["老师离线生成轨迹<br/>（老师自己的状态分布）"] --> A2["学生批量模仿"]
    end
    subgraph OPD["on-policy 蒸馏"]
        B1["学生生成前缀<br/>（学生自己的状态分布）"] --> B2["老师在前缀上补全"]
        B2 --> B3["学生学老师的续写<br/>（token 级监督）"]
        B3 -->|"更新后再生成新前缀"| B1
    end
    subgraph RL["强化学习"]
        C1["学生 rollout<br/>（环境交互）"] --> C2["奖励模型/规则打分"]
        C2 --> C3["策略梯度更新"]
        C3 -->|"重新采样"| C1
    end
    OFF -->|"分布不匹配<br/>student≠teacher occupancy"| OPD
    OPD -->|"无环境、无 RM，<br/>但受老师上限约束"| RL
```

---

## 1. 背景：Agentic RL 的成本困境

前两篇我们拆了 ToolRL（工具调用奖励设计）和 Search RL（搜索动作的 RL）。它们都属于同一条主线——**用 RL 训练 agent 与环境交互**。但这条主线有一个绕不开的现实问题：**贵**。

- **多轮 rollout**：一个 agent 轨迹动辄 4-8 轮工具调用，每轮都要模型生成 + 环境执行；
- **环境交互**：搜索 API、代码解释器、网页环境，延迟从几百 ms 到几秒，还有限流、不稳定、需要 mock；
- **奖励模型**：开放式任务没有规则奖励，需要训练 RM 或 LLM-as-judge，本身就是噪声源。

2025-2026 年，一个替代思路迅速成为 post-training 的核心技术（MiniCPM5、veRL 都把它写进官方管线）：**不让学生自己探索，让一个更强的老师替它探索**——这就是 On-Policy Distillation（OPD）。

## 2. 核心概念：什么让 OPD 是 "on-policy" 的

蒸馏（Distillation）不是新东西：小模型学大模型的输出，知识蒸馏 2015 年就有了。**OPD 的特殊之处在"on-policy"三个字上**——它指的是**学生从自己的状态分布出发**。

### 2.1 传统蒸馏的病：分布不匹配

Off-policy 蒸馏的做法：老师（或老师的推理引擎）批量生成轨迹 → 学生做 SFT 模仿。问题在于：

- 老师的状态分布（老师会遇到的 prompt/上下文/中间步骤）**≠** 学生的状态分布；
- 学生学到的是「老师在它的世界里怎么做」，但学生自己生成的轨迹会**漂移到老师没见过的区域**，那里的模仿信号是错的；
- 越小的学生漂移越严重（capacity gap + 自回归误差累积）。

### 2.2 OPD 的机制：学生走前缀，老师补全

OPD 的流程（以 RL 风格描述）：

```
1. 学生从 prompt 生成一段前缀（prefix，比如前 128 个 token）
2. 老师读「prompt + 学生前缀」，续写后续内容（补全，completion）
3. 学生在「prompt + 学生前缀」上，最大化老师补全内容的概率
4. 更新学生，回到第 1 步
```

关键在**第 2 步的输入是学生的前缀，不是老师自己的前缀**。这样：

- 学生学习的都是**自己会访问的状态**（on-policy occupancy）——分布匹配问题被绕开；
- 老师的续写提供了**逐 token 的稠密监督**（dense supervision），相当于免费的 token 级奖励信号，不需要奖励模型、不需要环境。

OPD 的损失函数就是标准的 teacher-forcing 交叉熵（或 KL）：

$$\mathcal{L}_{\mathrm{OPD}}(\theta) = -\mathbb{E}_{x\sim\mathcal{D}}\left[\sum_{t=1}^{T}\log \pi_{\theta}\left(y_t \mid x \oplus \underbrace{\pi_{\theta_{\mathrm{old}}}(x)}_{\text{学生前缀}} \oplus y_{<t}\right)\right]$$

其中 \(y\) 是老师在学生前缀上补全的内容。**注意监督的分布是 \(\pi_{\theta_{\mathrm{old}}}(x)\)（学生刚采样的前缀），而概率是在新参数 \(\theta\) 下计算的**——和 RL 的 on-policy 更新是同构的：旧策略采样、新策略更新。

### 2.3 与 RL 的关系：免费的稠密奖励？

OPD 常被描述为「把 RL 的探索替换成老师监督」。两者对比如下：

| 维度 | RL | OPD |
|---|---|---|
| 探索 | 学生自己探索（奖励稀疏、方差大） | 老师补全（探索外包，信号稠密） |
| 奖励 | 环境/RM 打分（可能噪声） | 老师 token 概率（确定性，但受老师上限约束） |
| 环境 | 必需 | **不需要**（ReOPD 做到零环境交互） |
| 上限 | 理论无上限 | **学生的能力上限 ≤ 老师**（除非用 G-OPD 的 reward extrapolation） |

这就是清华论文标题里的"免费午餐"质疑：**dense token 级奖励让训练又快又稳，但学生永远学不到老师没展示过的东西**——而 RL 的价值恰恰在于发现超越老师/超越数据的策略。

### 2.4 数学原理：OPD 的梯度为什么是 REINFORCE

先厘清一个文献里经常混为一谈的问题：**OPD 有两条数据流实现，数学上并不等价**。

- **DAgger 流**（THUNLP / ReOPD，即 §2.2 的 CE 公式）：学生只生成前缀，老师**补全**剩余内容，学生 teacher-forcing 学习老师补全的 token。监督是"学生状态 + 老师 token"——**状态 on-policy、目标 off-policy**，落在"前缀来源 × KL 方向"四象限的"学生前缀 × forward KL"格（详见姊妹篇《在线蒸馏方法全景》§5.2）。优点是更新稳定（每个 token 都是监督式拟合）、老师主动帮助学生探索；缺点是老师补全的内容与学生会自己生成的内容可能有风格差——这正是 THUNLP 说的"思考模式兼容性"问题（§3 条件一）。
- **REINFORCE 流**（MiniLLM / GKD / TML / veRL）：学生采样**完整轨迹**，老师不补全，只对这些 token 给 logprob 当稠密奖励，用反向 KL 的策略梯度更新。这才是"序列级反向 KL 的无偏估计"（四象限的"学生前缀 × reverse KL"格）。优点是目标语义与"学生学老师"完全一致；缺点是方差高——MiniLLM 为此上了三件套稳定技巧，GKD 则干脆不通过采样过程回传梯度。

下面推导 REINFORCE 流的数学。令学生 \(\pi_\theta\)、老师 \(\pi_T\)，学生自回归采样 \(y \sim \pi_\theta(\cdot \mid x)\)。目标写成序列级**反向 KL**（学生 \(\to\) 老师）：

$$\mathcal{L}_{\mathrm{OPD}}(\theta) = D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x) \;\|\; \pi_T(\cdot \mid x)\big) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\left[\log\frac{\pi_\theta(y \mid x)}{\pi_T(y \mid x)}\right]$$

注意期望内层是 \(y \sim \pi_\theta\)——这是"on-policy"三个字在数学上的位置：**目标函数自己依赖采样分布，而采样分布随 \(\theta\) 移动**。所以不能直接对目标做普通梯度下降（\(\theta\) 既出现在 \(\pi_\theta(y)\) 里、又出现在取期望的分布里），必须走策略梯度。用 score function 技巧：

$$\nabla_\theta \mathcal{L}_{\mathrm{OPD}} = \mathbb{E}_{y \sim \pi_\theta}\left[\underbrace{\log\frac{\pi_\theta(y \mid x)}{\pi_T(y \mid x)}}_{\text{序列级 log-ratio，当作奖励}}\;\nabla_\theta \log \pi_\theta(y \mid x)\right]$$

推导只有两步：乘积法则把 \(\nabla_\theta \pi_\theta(y)\) 拆成 \(\pi_\theta(y)\nabla_\theta \log \pi_\theta(y)\)；另一项 \(\mathbb{E}_{y\sim\pi_\theta}[\nabla_\theta \log \pi_T(y)] = 0\)——老师与 \(\theta\) 无关，且 \(\sum_y \nabla_\theta \pi_\theta(y) = \nabla_\theta \sum_y \pi_\theta(y) = \nabla_\theta 1 = 0\)。**结果和 REINFORCE 一模一样：log-ratio 是"奖励"，\(\nabla_\theta \log \pi_\theta\) 是策略梯度**——OPD 与 RL 在数学上没有本质区别，只是把 reward 换成了师生 log-ratio。

自回归分解后，序列级 log-ratio 拆成逐 token 求和，每个 token 有即时奖励 \(r_t\) 和 reward-to-go \(R_t\)：

$$r_t = \log\frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_T(y_t \mid x, y_{<t})}, \qquad R_t = \sum_{t'=t}^{T} r_{t'}, \qquad \nabla_\theta \mathcal{L}_{\mathrm{OPD}} = \mathbb{E}_{y\sim\pi_\theta}\Big[\sum_{t=1}^{T} R_t\, \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\Big]$$

（最后一步是策略梯度定理：动作 \(y_t\) 不影响它之前的 reward，所以未来奖励可以归到当前 token。）从这里产生两个**工程上最关键的决定**：

1. **log-ratio 必须 stop-gradient**。展开的梯度里 \(\log\frac{\pi_\theta}{\pi_T}\) 中的 \(\pi_\theta\) 也是 \(\theta\) 的函数，但标准 REINFORCE 把它当常数奖励处理：期望层面 \(\nabla r\) 的贡献合计为 0（上面第二项消零是同一回事），逐样本保留反而增大方差。实现时对 log-ratio 做 `.detach()`，只留 \(\nabla_\theta \log \pi_\theta\) 一条回传路径。
2. **discount = 0（只看当前 token）**。\(R_t\) 里装着未来所有 token 的 log-ratio——这是 MiniLLM 高方差、需要三件套稳定技巧的根源。Thinking Machines 的复现直接取 **discount = 0**：\(R_t \approx r_t\)，每个 token 只用自己的 log-ratio 当奖励。数学上是"只保留梯度展开的对角项"，方差显著更低；配合 sampled-token 估计，单 token 梯度项为：

$$\nabla_\theta \mathcal{L}_{\mathrm{OPD}}^{(t)} \approx \mathbb{E}_{y_t \sim \pi_\theta}\Big[\big(\log\pi_\theta(y_t \mid \cdot) - \log\pi_T(y_t \mid \cdot)\big)\,\nabla_\theta \log \pi_\theta(y_t \mid \cdot)\Big]$$

这就是 veRL `ADV_ESTIMATOR=token_reward_direct` 做的事：**token 级奖励 = 学生 logprob − 老师 logprob**，一个带权 REINFORCE 项。§6.1 的代码就是这一行的落地。

### 2.5 对照：off-policy SFT 蒸馏在数学上是什么

把同样的推导套到传统蒸馏，差距一目了然。off-policy SFT 的目标是**正向 KL**（老师 \(\to\) 学生）在固定老师数据上的期望：

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = D_{\mathrm{KL}}\big(\pi_T(\cdot \mid x) \;\|\; \pi_\theta(\cdot \mid x)\big) = \mathbb{E}_{y\sim\pi_T(\cdot \mid x)}\left[\log\frac{\pi_T(y \mid x)}{\pi_\theta(y \mid x)}\right]$$

关键区别：**期望内层是 \(y \sim \pi_T\)，与 \(\theta\) 无关**——没有策略梯度问题，直接求导：

$$\nabla_\theta \mathcal{L}_{\mathrm{SFT}} = \mathbb{E}_{y\sim\pi_T(\cdot \mid x)}\big[-\nabla_\theta \log \pi_\theta(y \mid x)\big] = -\mathbb{E}_{y\sim\pi_T}\Big[\sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\Big]$$

这正是"在老师生成的轨迹上做 teacher-forcing 最大似然"的梯度。**off-policy 蒸馏 = 老师分布下的 MLE，仅此而已**——梯度在老师的分布上取期望，学到的是"老师在它的世界里怎么做"；而 OPD 的梯度在学生分布上取期望、且每个 token 多一个 log-ratio 权重，学的才是"在我（学生）会遇到的状态下老师会怎么做"。两者只差"期望放在哪个分布上"，但这一差就是 exposure bias 的全部来源：训练分布（老师前缀）与推理分布（学生前缀）不一致。

**mode-seeking vs mode-covering 的机制也藏在梯度里**。反向 KL 的权重 \(\log\frac{\pi_\theta}{\pi_T}\) 在 \(\pi_\theta > \pi_T\)（学生自信区）为正、在 \(\pi_\theta < \pi_T\)（学生低估区）为负——**学生已经擅长的 token 被强化，学生回避的区域被压制**，质量向"师生共识的高概率模式"集中（mode-seeking / zero-avoiding）。正向 KL 的梯度没有权重，老师分布里每个高概率 token 都被推着学，学生容量不足时只能把质量"摊开"覆盖（mode-covering / zero-forcing），甚至会把质量放到老师认为几乎不可能的区域。直观例子：老师分布有两个峰、学生只装得下一个峰——正向 KL 的最优解是摊在两峰之间（两边都不像），反向 KL 的最优解是干脆挑一个峰集中（至少有一个模式是对的）。GKD 的实验指向同一结论：temperature sampling 评估下，mode-seeking 的散度（reverse KL 或 JSD(0.9)）普遍优于 forward KL。

**温度缩放**再补一刀。Hinton 2015 蒸馏中老师分布除以温度软化：\(\pi_T^{(T)}(y_t) = \frac{\exp(z_{y_t}/T)}{\sum_v \exp(z_v/T)}\)。\(T\to 0\) 退化为 one-hot（hard label），\(T>1\) 把"次优 token 的相对排序"也传给学生。forward KL 蒸馏里 \(T\) 是核心旋钮（分类任务常用 2-4），因为学生要模仿老师的全部模式；**OPD 里 \(T=1.0\) 是默认**——反向 KL 本来只需要老师的高概率模式，softening 边际收益小，而 TML/veRL 的实现里老师直接给 logprob（一次前向，不用采样）。

## 3. 机制拆解：OPD 什么时候成功、什么时候失败（清华 OPD，[arXiv:2604.13016](https://arxiv.org/abs/2604.13016)）

清华这篇（2026-04，ICML 2026 FoGen Workshop）是 OPD 的第一篇系统性机制研究。核心结论是两个**成败条件**：

### 条件一：师生思考模式必须兼容

> (i) the student and teacher should share compatible thinking patterns

如果学生的"思考模式"（推理风格、格式、token 习惯）和老师不一致，OPD 会失败——学生在自己的前缀上，老师续写的内容对学生来说是"外星语言"，学不动。

### 条件二：老师必须提供"真正的新能力"

> (ii) even with consistent thinking patterns and higher scores, the teacher must offer genuinely new capabilities beyond what the student has seen during training

老师分数更高**不等于**蒸馏有效。如果老师只是在学生已见过的模式上做得更好，学生早已饱和，学无可学。

### 3.1 验证方法：弱到强反向蒸馏

作者用一个巧妙的实验验证这两个条件——**反向蒸馏**（让学生教老师，看哪个方向有效）：

- 同家族的 1.5B 和 7B 模型互相蒸馏；
- 从学生的视角看，**1.5B 和 7B 老师"分布上不可区分"**——即 7B 老师比 1.5B 老师多出来的能力，落在学生从未访问的状态上，学生根本接触不到；
- 这解释了为什么「用更大模型做老师」不是银弹：**老师的有效信息 = 老师能力 ∩ 学生会访问的状态**。

### 3.2 Token 级机制：高概率 token 的渐进对齐

- 成功的 OPD 表现为：在**学生访问的状态**上，学生和老师的高概率 token 集合逐步对齐；
- 一个关键数字：**97%-99% 的概率质量集中在一个很小的共享 token 集合**上——即学生与老师一致的部分是高度集中的，对齐过程本质上是「把共享 token 集合的概率调准」，而不是全面逼近老师的分布。

### 3.3 两个实用的失败恢复策略

1. **Off-policy cold start**：先用离线蒸馏（老师轨迹 SFT）把学生预热到"会说老师的语言"，再做 OPD——先解决思考模式兼容性问题；
2. **Teacher-aligned prompt selection**：选择老师擅长的 prompt 分布做蒸馏，避免在老师也会翻车的区域硬学。

## 4. Agent 场景：ReOPD 与前缀陷阱（arXiv:2607.04763）

清华的研究是单轮/短视野的。多轮 agent 场景（工具调用、搜索、网页）里，OPD 有一个**全新的问题**——ReOPD 论文称之为 prefix trap（前缀陷阱）。

### 4.1 前缀陷阱：双面分布偏移

多轮 OPD 的师生交互发生在**对话历史**上。这里有两股相反的力量：

- **往 on-policy 拉**：历史越贴近学生自己生成的轨迹，学生学得越相关（occupancy 匹配）；
- **往 reliable 拉**：但越贴近学生的历史，越可能是老师**没见过/不擅长**的历史——老师在那些位置的目标（续写）不可靠。

这就是论文说的 *"making histories more student-on-policy improves relevance to the student, but can query the teacher on histories where its target is unreliable"*——**两难**。完全 online 的 OPD 每一步都重新 rollout 学生 + 查询老师，成本高且每一步都踩在这个两难上。

### 4.2 ReOPD 的解法：prefix replay + 位置加权

ReOPD 的改动非常工程化：

1. **离线 prefix 池**：用领域老师（如 ReTool 训练的数学 agent、搜索 agent）在 RL 过程中**顺手收集**轨迹，切成「前缀 + 监督目标」存进池子——收集几乎零成本（老师 RL 本来就要 rollout）；
2. **学生只做"受监督的一步"**：从池子里采样前缀，学生在前缀上续写，老师的目标作为 dense 监督，**全程零环境调用**；
3. **位置加权采样**：多轮轨迹越靠后的位置偏移越大（学生的轨迹漂移随轮次累积），所以按 \(p_t \propto \kappa^{t}\)（\(\kappa=0.6\)）采样，**偏向早期、低偏移的前缀**——损失的公式不变，只是数据采样概率变了。

效果（论文报告）：**零工具调用、4-9× 更快的 rollout，准确率匹配甚至超过 online OPD**。另一个附带好处：离线池把**环境与训练解耦**——不同环境（数学、搜索、网页）的轨迹由各自的领域老师收集，合并进一个池子，单个学生可以联合蒸馏，不需要把所有环境同时在线。

## 5. 生态全景：OPD 的 2025-2026

| 工作 | 定位 | 亮点 |
|---|---|---|
| **thunlp/OPD**（arXiv:2604.13016） | 机制研究 | 成败条件、token 级分析、恢复策略；top-k overlap 诊断已合入 veRL（`distillation/overlap_ratio`） |
| **ReOPD**（arXiv:2607.04763） | 多轮 agent | prefix replay、零环境交互、4-9× 加速；框架基于 THUDM slime |
| **G-OPD**（RUCBM） | 超越老师 | *Learning beyond Teacher*：奖励外推（reward extrapolation），试图突破"学生 ≤ 老师"的天花板 |
| **RLCSD**（THU-BPM） | 自蒸馏变体 | 对比式 on-policy self-distillation，学生向**自己**蒸馏 |
| **CaOPD**（Salesforce） | 校准感知 | 让学生继承老师的校准特性（calibration-aware） |
| **MiniCPM5-1B** | 工业落地 | 官方管线明确采用 "RL + OPD" 组合——1B 模型靠 OPD 学大模型能力 |
| **veRL PR #6469** | 框架集成 | OPD overlap 诊断指标进入 veRL 官方（`overlap_ratio` / `overlap_token_advantage`） |

还有三份 Awesome 列表（thinkwee/AwesomeOPD 820⭐、chrisliu298 653⭐、nick7nlp 509⭐）和一篇综述（arXiv:2604.00626，OPDHub）——一个 2025 年还不存在的方向，2026 年已经长成完整的生态。

## 6. 工程骨架：最小 OPD 实现

### 6.1 Online OPD：先跑通 DAgger 流（最简，§2.2 的 CE）

##### 变量映射表（数学 ↔ 代码）

| 数学符号 | 代码变量 | Shape / 类型 | 含义 |
|---|---|---|---|
| \(r_t=\log\pi_\theta(y_t\mid\cdot)-\log\pi_T(y_t\mid\cdot)\) | `student_logp - teacher_logp.detach()` | `(T,)` | token 级稠密奖励 |
| \(L=-\mathbb{E}\sum_t r_t\log\pi_\theta(y_t\mid\cdot)\) | `loss = -(r_t * student_logp).sum()` | 标量 | 策略梯度式蒸馏损失 |
| replay 条目 `{prefix, targets}` | 经验池样本 | dict | DAgger 式前缀回放 |

`rewards/r` 为规则奖励标量、`advantages` 为组内标准化后的优势、`ratio/logp/old_logp`
为 token 级重要性比率三件套、`format_score/correctness` 类为分项奖励。若与具体框架
命名有出入，以你所用版本的 reward_score 插件签名为准。


```python
# opd.py —— DAgger 流：学生走前缀，老师补全，学生学补全（最简可跑版本）
def opd_step(student, teacher, prompt: str, prefix_len: int, teach_len: int):
    # 1. 学生生成前缀（学生自己的状态分布 —— on-policy 的关键）
    prefix = student.generate(prompt, max_new_tokens=prefix_len,
                              sampling_params={"temperature": 0.7})

    # 2. 老师在「prompt + 学生前缀」上补全（dense 监督）
    with torch.no_grad():                                  # 老师永远不训练
        completion = teacher.generate(prompt + prefix, max_new_tokens=teach_len)
    completion = completion[len(prompt) + len(prefix):]

    # 3. 学生在自己的前缀上学习老师补全（teacher-forcing CE / forward KL 的 MC 估计）
    logits = student(prompt + prefix, completion)          # 前向
    loss = cross_entropy(logits, completion_tokens)        # token 级监督
    loss.backward(); student.step()
```

**DAgger 流最容易错的地方**：第 2 步老师的输入必须是 `prompt + 学生前缀`，不是 `prompt + 老师自己生成的前缀`——写错就退化成 off-policy 蒸馏，distribution mismatch 会回来。

### 6.2 Online OPD：REINFORCE 流完整训练循环（§2.4 的数学落地）

```python
# opd_train.py —— REINFORCE 流：学生采样完整轨迹，老师打 logprob 当稠密奖励
# 数学对应（§2.4）：min L = E_{y~pi_theta}[ sum_t r_t * log pi_theta(y_t|x,y_<t) ]，
#   r_t = log pi_theta(y_t|·) - log pi_T(y_t|·)，detach 后当作稠密 token 奖励
# 常用超参（TML 复现 Qwen3 配方）：前缀 128 token，batch 64 prompts × 4 samples，
#   lr 全参 1e-6 / LoRA(rank 128) 1e-4~3e-4，学生采样温度 0.7~1.0，老师温度 1.0
def opd_train_loop(student, teacher, prompts, steps, args):
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    for _ in range(steps):
        # 1. on-policy：学生采样完整轨迹（前缀要短，老师信号随前缀长度衰减，见 §6.5）
        xb = sample(prompts, args.batch)
        y = student.generate(xb, max_new_tokens=args.response_len,
                             temperature=args.temperature, top_p=0.9)

        # 2. 老师打分：对学生的轨迹做一次前向 compute_logprobs（T=1.0，无需采样）
        with torch.no_grad():
            t_logp = teacher.compute_logprobs(xb, y)       # log pi_T(y_t | x, y_<t)

        # 3. 学生前向 + 构造稠密奖励并 detach（§2.4 决定 1）
        s_logp = student.compute_logprobs(xb, y)           # log pi_theta(y_t | x, y_<t)
        r_t = (s_logp - t_logp).detach()                   # ★ stop-gradient，只留 PG 路径

        # 4. 带权 REINFORCE：loss = mean(r_t * log p)，
        #    梯度下降 = 最小化反向 KL（负号已含在 r_t 的符号里，这里不要再加负号）
        loss = (r_t * s_logp).mean()
        loss.backward()
        clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); opt.zero_grad()
```

**REINFORCE 流三个容易错的地方**：

1. **`r_t` 必须 `.detach()`**——它是"奖励"不是"目标"。回传后 log-ratio 里的 \(\pi_\theta\) 若参与梯度，等价于把 mode-seeking 变成了未知目标，方差爆炸；
2. **loss 不加负号**。\(r_t = \log\pi_\theta - \log\pi_T\) 自带符号：学生比老师更可能的 token（\(r_t>0\)）被正向强化，学生低估老师高概率的 token（\(r_t<0\)）被压低——这就是 mode-seeking 梯度（§2.5）；再加负号会反转成"学老师低概率区域"，直接崩；
3. **老师输入必须是学生的轨迹**（prompt + student-completed response），不能是老师自己采样的轨迹。第 2 步和第 3 步的条件必须完全相同，否则 log-ratio 在错误的状态上计算。

如果要更精确（也更贵），把第 3-4 步换成 full-vocabulary 反向 KL——每步需要师生全词表 logits：

```python
# 备选：full-vocab 反向 KL（每 token 全词表，最准；THUNLP 三种粒度之一）
loss = 0.0
for t in range(args.response_len):
    p_s = softmax(student_logits[:, t])                    # 学生 next-token 分布
    p_t = softmax(teacher_logits[:, t])                    # 老师 next-token 分布
    loss += (p_s * (p_s.log() - p_t.log())).sum(dim=-1)    # token 级 D_KL(p_s || p_t)
loss = loss.mean() / args.response_len
```

三种粒度（THUNLP 分类，verl 都支持）：full-vocab 最准但每步要全词表 logits 最贵；**sampled-token**（§6.2 的写法）用学生采到的 token 做无偏 MC 估计，最便宜且实践中够用；top-k 折中（k≈16，避开 Top-1 会崩）。

### 6.3 ReOPD：离线 prefix 池（多轮 agent 场景）

```python
# reOPD.py —— prefix replay：零环境交互
# 池子里的每条：{"prefix": 老师轨迹的前 t 个 token, "targets": 第 t+1 步及之后的 token}
# 收集方式：老师 RL 时顺手存（零成本）

def reOPD_step(student, pool, kappa: float = 0.6):
    # 位置加权采样：p_t ∝ kappa^t，偏向早期低偏移前缀
    batch = weighted_sample(pool, weight_fn=lambda item: kappa ** item["turn_index"])

    for item in batch:
        logits = student(item["prefix"])            # 学生只做受监督的一步
        loss = cross_entropy(logits, item["targets"])  # 老师的 dense 监督
    loss.backward(); student.step()
    # 全程没有环境调用、没有老师在线推理
```

### 6.4 与 veRL 结合

veRL 的 OPD 支持（v0.7.0+）：rollout 用学生权重采样 prefix，teacher 作为另一个 actor 补全，loss 走 `distillation` 模式；训练时用 `distillation/overlap_ratio`（师生 top-k token 重叠率）和 `overlap_token_advantage`（重叠 token 的优势贡献）诊断——这两个指标是清华论文的 token 级分析直接落地的。

对应 §2.4 的数学：veRL 里 `ADV_ESTIMATOR=token_reward_direct` 就是把优势函数设成逐 token 的 `log π_student − log π_teacher`，然后走标准的 GRPO 更新循环（组内归一化那一步直接跳过或设 n=1）——**从 GRPO 训练脚本到 OPD 是配置级改动**：KL 正则的 reference model 换成老师、reward 换成 token 级 log-ratio，这正是 GKD 论文和 TML 都强调的"一行代码"论断。

### 6.5 训练超参数速查

| 超参 | 推荐值（来源） | 说明 |
|------|---------------|------|
| prefix / response 长度 | 前缀 128 token（TML）；响应 1K-7K 是甜点区（THUNLP） | 前缀过长 → 老师信号衰减（+0.37 @1K → +0.02 @16K）；响应超 10K-15K 训练后期坍缩 |
| 学生采样温度 | 0.7-1.0（GKD 用 1.0） | 温度放开保多样性，on-policy 数据覆盖面才够 |
| 老师温度 | 1.0（TML / veRL 默认） | reverse KL 场景不需要软化（§2.5） |
| learning rate | 全参 1e-6（Qwen3 复现）；LoRA 1e-4~3e-4（GKD 的 T5 用 3e-4） | **reverse KL 对高 lr 敏感**（GKD 论文明确），先小后大 |
| batch | 64 prompts × 4 samples（TML）；GRPO 风格 n=8 | 比 RL 小很多就能稳（dense 监督的信息密度高） |
| λ（GKD 数据混合比） | 0.5-1.0 | 0.5 掺固定数据防 on-policy 分布坍缩，1.0 纯 on-policy |
| discount | 0（TML / veRL token_reward_direct） | 只看当前 token 的 log-ratio，方差最小 |
| 采样 n / 梯度步 | 单 prompt 20 步 × 256 rollouts 即可接近老师（TML） | dense 监督下数据效率远超 RL（见姊妹篇《在线蒸馏方法全景》§8 效率账） |

## 7. 调优清单与陷阱

1. **先验思考模式兼容性**：OPD 前先做短实验看师生 top-k 重叠率（overlap_ratio）。如果 <30%，大概率思考模式不兼容，先 off-policy cold start 或用同家族老师；
2. **别迷信"更大的老师"**：弱到强反向蒸馏说明，老师的有效信息 = 能力 ∩ 学生会访问的状态。学生太小/太弱时，换 70B 老师不如换一个**思考风格更贴近**的老师；
3. **监控 prefix 漂移**：多轮场景每轮学生轨迹都会漂移（prefix trap 的早期信号），ReOPD 的 \(\kappa\) 就是漂移速度的旋钮——\(\kappa\) 太小浪费数据（只用早期），太大踩进不可靠区；
4. **OPD 不是 RL 的免费替代**：OPD 学不到老师没见过的东西。需要**超越老师**时用 G-OPD 式奖励外推，或回到 RL 本体；
5. **长视野衰减**：清华论文的开放问题——OPD 的 dense token 监督在长轨迹上会稀释（后面的 token 学不动），long-horizon OPD 是当前没有定论的边界；
6. **日志监控**：除了 loss，记录师生重叠率、prefix 长度分布、学生独立 rollout 的准确率（防止"蒸馏过头"——学生只会模仿不会自己决策）。

## 8. 批判与展望

1. **免费午餐的代价**：OPD 的稠密监督来自老师，本质是**把探索风险从学生转移到老师的选择上**。老师的选择受限于它的训练分布——OPD 适合"把已验证的能力下放"，不适合"发现新能力"。
2. **学生 ≤ 老师的铁律正在被打破**：G-OPD 的 reward extrapolation、RLCSD 的自蒸馏，都在试图突破这个上限；但机制研究（清华）表明突破的路径必须绕开"学生访问不到老师的有效状态"这个结构性障碍。
3. **与 RL 的合流不是二选一**：MiniCPM5-1B 的管线是 "RL + OPD" 组合——OPD 做能力下放和冷启动，RL 做最后的超越与对齐。ToolRL（奖励设计）解决"RL 怎么学工具"，Search RL（搜索动作）解决"学什么"，OPD 解决"能不能便宜地学"——三者拼起来，才是完整的 Agentic post-training 版图。
4. **诊断工具会越来越重要**：OPD 的成败在 token 级（97-99% 概率质量集中），肉眼不可见。veRL 把 overlap 指标做成框架级诊断，说明这个领域正在从"玄学"走向"工程"——这和 ToolRL 把奖励设计工程化是同一个趋势。

> 🧪 **动手练习**：① 学生采样温度扫 {0.7, 1.0}，对比稠密奖励信号的方差；② prefix 长度取 64/128/256 三档，画出达到同等 BLEU 所需的真实教师 token 数。

## 参考与延伸阅读

- 论文：Yaxuan Li et al., *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*, arXiv:2604.13016（清华 NLP，ICML 2026 FoGen）
- 论文：Baohao Liao et al., *Multi-Turn On-Policy Distillation with Prefix Replay*, arXiv:2607.04763（ReOPD，2026-07）
- 论文：Jiazhan Feng et al., *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs*, arXiv:2504.11536（ReOPD 的 math 环境基础）
- 综述：*A Survey of On-Policy Distillation for Large Language Models*, arXiv:2604.00626（OPDHub）
- 代码：[thunlp/OPD](https://github.com/thunlp/OPD) （veRL v0.7.0 底座）；[BaohaoLiao/ReOPD](https://github.com/BaohaoLiao/ReOPD) （slime 底座，含 math/search 两环境评估）
- 生态：G-OPD（RUCBM，reward extrapolation）、RLCSD（THU-BPM）、CaOPD（Salesforce）、MiniCPM5-1B（OpenBMB）、veRL PR #6469（overlap 诊断）
- 系列前篇：本站《ToolRL 深度剖析》（2026-08-10）、《Search RL 深度剖析》（2026-08-11）
* 2026-08 新作速览："Step-Level On-Policy Distillation" ([arXiv:2608.16333](https://arxiv.org/abs/2608.16333)，SFT↔OPD 光谱插值)；"Group-Calibrated OPD for Long-Context Reasoning" ([arXiv:2608.19181](https://arxiv.org/abs/2608.19181))；"Open-MOPD 多教师能力失衡诊断" ([arXiv:2608.19098](https://arxiv.org/abs/2608.19098))；"OPD 泛化性的双刃剑" ([arXiv:2608.16647](https://arxiv.org/abs/2608.16647))
* 生态索引持续更新：[AwesomeOPD](https://github.com/thinkwee/AwesomeOPD)（2026-07-23 版新增审读 191 条，覆盖白盒/黑盒教师、OPSD、OPD-RL 混合、Agent OPD 等分类）
* 系列延伸（象限Ⅱ）：《[逆强化学习五十年](/2026/08/22/irl-fifty-years-from-demonstrations/)》 · 《[正向学策略，反向学奖励：IRL 在 LLM 对齐里的复活](/2026/08/22/irl-renaissance-in-llm-alignment/)》（X-KD 一节与本篇直接相关）
