> 系列文档：扩散语言模型（Diffusion LM）学习资料

# 09 开放问题与分级阅读清单

本篇把前 8 篇的结论收口到“还没解决什么”以及“该读什么”。A 部分列出仍然开放的难题，B 部分给出按能力分档的阅读清单与节奏建议。所有论断均附来源，标注为“二手”或“主张”的表示并非一手论文结论。

## A. 开放问题

下面每条都按“问题 → 为什么难 → 现有线索 → 怎么判断被解决了”四段式展开。

### 1. 似然与可评估性

**问题**：dLLM 通常只能优化证据下界（ELBO），拿不到像自回归（AR）那样精确的负对数似然（NLL），跨模型比较缺乏统一口径。

**为什么难**：吸收态／掩码扩散的边缘分布 $$q(x_t)$$ 解析困难，精确 $$\log p_\theta(x_0)$$ 需要昂贵的蒙特卡洛（如 SEDD 的 denoising variant）或借助任意序 AR 等价（RADD）来近似；不同噪声表、不同 $$\bar\alpha_t$$ 下 ELBO 的松弛程度不一致。

**现有线索**：RADD 给出 denoising cross-entropy 形式的近似 NLL；MD4 给出闭式加权的连续时间 ELBO；社区普遍用 GPT-2 规模 perplexity 作代理比较。

**怎么判断被解决**：出现无需额外蒙特卡洛、与 AR 同口径的精确似然估计，且在不同噪声表下数值稳定、可直接与自回归 perplexity 正面对比。

### 2. 并行度与质量的本质权衡

**问题**：每步解出越多 token，生成越快，但质量会掉；能否既快又好？

**为什么难**：$$\tau$$-leaping 假设已解 token 之间条件独立，而真实语言 token 强依赖；多解一个 token 就会破坏依赖链。

**现有线索（二手）**：ParallelBench 上从每步 1 个 token 到每步 2 个 token，GSM8K 掉约 10 个百分点（来源 [zylos.ai 研究笔记](https://zylos.ai/research/2026-06-16-diffusion-language-models-agentic-inference)，二手）。Fast-dLLM 的置信度阈值并行解码在 LLaDA／Dream 上最高取得 27.6× 吞吐且精度几乎不降。

**怎么判断被解决**：在保持质量（如 GSM8K／HumanEval 跌幅小于 2 点）前提下，每步并行度稳定大于 4，且能给出依赖保持的理论保证。

### 3. 变长生成与长度控制

**问题**：朴素 MDLM 是定长的；如何让 dLLM 生成任意长度序列？

**为什么难**：吸收态过程在固定 $$T$$ 步内把整条序列从全掩码去噪到完整，长度在 $$x_0$$ 空间里是固定的。

**现有线索**：块扩散（BD3-LM）在已生成块上做 KV 缓存并续写下一块；编辑流／集合扩散用可变长度状态；semi-AR 采样器（MDLM 自带）按块生成。代价是块边界引入近似、KV 管理与并行度下降。

**怎么判断被解决**：单一模型在开放生成任务上长度可任意指定、困惑度不退化，且不需要事后截断或拼接。

### 4. KV cache 与长上下文（二手）

**问题**：128K+ 上下文下，双向注意力如何做 KV 缓存与量化？

**为什么难**：双向注意力没有因果掩码，标准因果 KV cache 失效；长序列的 KV 占用与时间步采样耦合。

**现有线索（二手摘要）**：HKUST 博士答辩摘要（[cse.hkust.edu.hk](https://cse.hkust.edu.hk/pg/defenses/S26/gheaf-28-04-2026.html)）归纳出三条线——稀疏注意力、KV 管理、低比特量化。Fast-dLLM 的块级近似 KV cache 已能在双向模型上复用缓存。

**怎么判断被解决**：在 128K 上下文上实现接近 AR 的 cache 命中率与显存占用，且量化到 4-bit 不显著掉点。

### 5. 蒸馏：少步化到单步

**问题**：能否把多步 dLLM 蒸馏成 1–2 步仍保住质量？

**为什么难**：多步去噪携带的迭代修正信息在单步被压缩；离散空间的分布匹配比连续空间更难。

**现有线索（二手）**：ai-beat 综述（[ai-beat.github.io](https://ai-beat.github.io/news/2026/08/diffusion-lm-two-roads)）区分两条路线——离散一致性蒸馏与连续侧 flow map 蒸馏。

**怎么判断被解决**：单步 dLLM 在基准上与多步版本差距小于 5%，且单步吞吐相对 AR 有数倍优势。

### 6. 后训练／RL

**问题**：如何对 dLLM 做偏好对齐与推理 RL？

**为什么难**：似然估计方差大；基于 ELBO 的对数似然近似在晚期（低噪声）会坍塌；负优势下界在掩码比例极端时失效。

**现有线索**：AGRPO（[arXiv:2510.04019](https://arxiv.org/html/2510.04019)）把优化下沉到单步去噪，用 novel timestep 估计得到无偏梯度，在 Countdown／Sudoku 上相对 LLaDA 提升 $$+59.4\%$$／$$+69.7\%$$。

**怎么判断被解决**：出现稳定、可复现的 RL 配方，使 dLLM 在数学／代码／对齐上系统性超过 SFT 基线，且方差可显式控制。

### 7. Scaling law（主张）

**问题**：dLLM 的 scaling law 到底长什么样？

**为什么难**：数据效率、参数效率与计算效率三者关系尚未在大规模上摸清。

**现有线索（主张）**：量子位报道（[qbitai.com](https://www.qbitai.com/2025/12/360425.html)）主张——计算受限下 dLLM 更 data-hungry、被称为 “Super Data Learner”，同 dense 参数下可更小；但社区共识尚未形成。

**怎么判断被解决**：出现多团队复现的、跨 $$0.1\mathrm{B}$$–$$100\mathrm{B}$$ 的 dLLM scaling 曲线，并能预测 NLL 与下游准确率。

### 8. 与 AR 的关系

**问题**：dLLM 与 AR 是替代还是互补？

**为什么难**：两者目标可互转，但训练分布、推理代价、可控性不同。

**现有线索**：DiffuLLaMA 证明 AR→dLLM 转换只需持续预训练；RADD 证明吸收态扩散上界可写成任意序 AR 的 NLL 期望；dLLM 可作 AR 的 draft，AR 可作 dLLM 的先验。

**怎么判断被解决**：出现统一目标或混合架构，在速度-质量前沿同时优于纯 AR 与纯 dLLM。

### 9. 可控生成与约束满足

**问题**：如何在并行去噪中满足硬约束（格式、词典、语法）？

**为什么难**：并行解码无法像 AR 那样逐位约束；纠错需重新加噪（remask）。

**现有线索**：离散 CBG／CFG、约束解码、纠错／重噪声带来“提交-重噪声”新交互范式；LLaDA 的 remask 策略保证反向一致。

**怎么判断被解决**：约束满足率（如 JSON／SQL 合法率）接近 $$100\%$$ 且生成速度不退化。

### 10. 评测生态

**问题**：如何公平比较 dLLM？

**为什么难**：基准污染；步数 NFE、温度、并行度、remask 比例等推理超参对分数影响巨大；报告规范缺失。

**现有线索**：社区开始固定 steps／block 配置（如 Dream 全并行、LLaDA 半 AR 块大小）做对照，但仍缺统一 leaderboard。

**怎么判断被解决**：主流基准给出 “NFE-质量” 帕累托曲线与标准推理协议，且结果可复现。

### 11. 多模态与科学领域

**问题**：dLLM 能否主导蛋白／核酸等非自然语言序列设计？

**为什么难**：字母表与结构约束不同，需把扩散建在领域拓扑上；评测非文本化。

**现有线索**：ESM3、NT-v3 类工作把离散扩散用于生物序列；离散扩散框架（[arXiv:2607.13431](https://arxiv.org/abs/2607.13431)）把 tokenization／字母表列为设计空间。

**怎么判断被解决**：在湿实验可验证的设计任务上，dLLM 生成序列的功能成功率超过 AR 与进化搜索。

### 12. 量化／低比特部署

**问题**：dLLM 能否 4-bit 部署且保持并行优势？

**为什么难**：双向注意力的激活分布与 AR 不同，量化敏感点尚未系统定位。

**现有线索**：前述 HKUST 长上下文三条线已包含低比特量化；社区尚无系统报告。

**怎么判断被解决**：4-bit dLLM 在端侧达到与 FP16 同质量且吞吐翻倍。

### 13. MoE 与 dLLM 的相互作用

**问题**：MoE 路由在并行去噪下是否仍有效？

**为什么难**：每步所有位置同时激活，路由统计量随噪声水平变化。

**现有线索**：目前主流 dLLM 多为 dense；MoE 与扩散结合仅零星探索。

**怎么判断被解决**：MoE-dLLM 在同等激活参数量下匹配 dense-dLLM 质量且更快。

## B. 分级阅读清单

按“入门 → 核心 → 规模化 → 系统与加速 → 后训练 → 综述与教学”六档组织。表格列：标题（中英）、年份、arXiv／URL、为什么读、建议读法。

### 入门

| 标题（中英） | 年份 | arXiv／URL | 为什么读 | 建议读法 |
| --- | --- | --- | --- | --- |
| Denoising Diffusion Probabilistic Models（DDPM） | 2020 | arXiv:2006.11239 | 连续扩散基础，离散扩散的术语都源于此 | 读第 2 节前向-反向过程与 ELBO 推导 |
| Structured Denoising Diffusion Models in Discrete State Spaces（D3PM） | 2021 | arXiv:2107.03006 | 离散扩散开山，含吸收态／均匀／嵌入近邻转移核 | 读第 3 节转移矩阵与吸收态定义 |

### 核心

| 标题（中英） | 年份 | arXiv／URL | 为什么读 | 建议读法 |
| --- | --- | --- | --- | --- |
| Score Entropy Discrete Diffusion（SEDD） | 2024 | arXiv:2310.16834 | 把 score matching 推广到离散空间 | 读 score entropy 损失与 concrete score |
| Simple and Effective Masked Diffusion Language Models（MDLM） | 2024 | arXiv:2406.07524 | 吸收态变分下界化简为加权 MLM 交叉熵 + Rao-Blackwell | 读第 3 节 SUBS 参数化；MD4 即其连续时间闭式加权目标 |
| Reparameterized Absorbing Discrete Diffusion（RADD） | 2024 | arXiv:2406.03736 | concrete score 重参数化为时间相关标量 × 时间无关条件分布，可去时间条件加速 | 读 Theorem 1 与缓存采样 |
| A Continuous Time Framework for Discrete Denoising Diffusion Models（Campbell 等 CTMC） | 2022 | arXiv:2202.09978 | 连续时间离散扩散的一般框架 | 读 CTMC 与精确去噪条件 |
| Discrete Flow Matching（Gat 等） | 2024 | arXiv:2407.15595 | 从 flow matching 视角统一离散生成 | 读 probability velocity 与 MaskGit 是其特例 |

### 规模化

| 标题（中英） | 年份 | arXiv／URL | 为什么读 | 建议读法 |
| --- | --- | --- | --- | --- |
| LLaDA | 2025 | arXiv:2502.09992 | 首个 8B 掩码扩散 LLM | 读训练目标与 remask 采样 |
| Dream 7B | 2025 | arXiv:2508.15487 | 当前最强开源 dLLM，AR 初始化 + token 级噪声重调度 | 读实验配置与任意序生成 |
| Block Diffusion（BD3-LM） | 2025 | arXiv:2503.09573 | 块扩散支持变长与 KV 缓存 | 读训练算法与数据驱动噪声表 |
| DiffuLLaMA | 2025 | [GitHub](https://github.com/jwkirchenbauer/DiffuLLaMA) | AR→dLLM 转换，证明目标连通（arXiv 编号待核实，先给仓库） | 读目标等价性证明 |
| Mercury | 2025 | arXiv:2506.17298 | 商业级扩散 LLM，极致吞吐 | 读吞吐-质量前沿 |
| Seed Diffusion | 2025 | arXiv:2508.02193 | 大规模离散扩散，2146 token/s | 读速度-质量帕累托 |
| Efficient-DLM | 2025 | arXiv:2512.14067 | 位置相关掩码缩小 train／test gap | 读位置相关掩码与块注意力 |

### 系统与加速

| 标题（中英） | 年份 | arXiv／URL | 为什么读 | 建议读法 |
| --- | --- | --- | --- | --- |
| Fast-dLLM | 2025 | arXiv:2505.22618 | 免训练 KV cache + 置信度并行解码 | 读块级近似 KV cache 与并行解码 |
| Stable-DiffCoder | 2025 | [GitHub](https://github.com/ByteDance-Seed/Stable-DiffCoder) | 代码扩散 LLM 的系统化训练／推理 | 读仓库 README 与配置 |

### 后训练

| 标题（中英） | 年份 | arXiv／URL | 为什么读 | 建议读法 |
| --- | --- | --- | --- | --- |
| LLaDA 1.5：Variance-Reduced Preference Optimization（VRPO） | 2025 | arXiv:2505.19223 | 降方差偏好优化，奠定 dLLM 对齐 | 读方差界与 antithetic 采样 |
| Simple Policy Gradients for Reasoning with Diffusion Language Models（AGRPO） | 2025 | arXiv:2510.04019 | 单步去噪策略梯度，无偏 RL | 读 timestep 估计与逐步优化 |

### 综述与教学

| 标题（中英） | 年份 | arXiv／URL | 为什么读 | 建议读法 |
| --- | --- | --- | --- | --- |
| How to Build a Diffusion Language Model（Kuleshov 组教学博客） | 2026 | [博客](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/) | 手把手从零实现 | 通读并跑示例代码 |
| A Survey on Diffusion Language Models | 2025 | arXiv:2508.10875 | 领域全景 | 读 taxonomy 与未来方向 |
| Discrete Diffusion in Large Language and Multimodal Models: A Survey | 2025 | arXiv:2506.13759 | LLM／多模态视角综述 | 读应用章节 |
| Discrete Diffusion Models: A Unified Framework from Tokenization to Generation | 2026 | arXiv:2607.13431 | 以离散状态空间构建为统一视角 | 读 tokenization／字母表设计空间 |

### 代码仓库清单

- MDLM 官方实现：<https://github.com/kuleshov-group/mdlm>
- LLaDA 官方仓库：<https://github.com/ML-GSAI/LLaDA>
- Fast-dLLM：<https://github.com/NVlabs/Fast-dLLM>
- dllm 训练框架：<https://github.com/efficient-reasoning/dllm>
- dInfer 推理框架：<https://github.com/inclusionAI/dInfer>
- bd3lms：<https://github.com/kuleshov-group/bd3lms>
- Megatron-Bridge 中的 Nemotron diffusion 训练 recipe（位于 Megatron-Bridge 仓库内）
- Stable-DiffCoder：<https://github.com/ByteDance-Seed/Stable-DiffCoder>

### 4 周／8 周两种读法

- **4 周速通**：第 1 周 DDPM + D3PM；第 2 周 MDLM／MD4 + SEDD + RADD；第 3 周 LLaDA + Dream + BD3-LM；第 4 周两篇综述 + 跑 MDLM／Fast-dLLM 代码。
- **8 周深入**：前 4 周同上；第 5 周 Campbell CTMC + 离散 flow matching；第 6 周 Efficient-DLM + Mercury + Seed Diffusion；第 7 周 VRPO + AGRPO；第 8 周框架综述 + 自定方向精读。

## 参考链接

| 来源 | 说明 | 链接 |
| --- | --- | --- |
| ParallelBench 二手结论 | 每步 2 token 的 GSM8K 掉点 | <https://zylos.ai/research/2026-06-16-diffusion-language-models-agentic-inference> |
| HKUST 长上下文答辩摘要 | KV cache 三条线 | <https://cse.hkust.edu.hk/pg/defenses/S26/gheaf-28-04-2026.html> |
| ai-beat 蒸馏综述 | 两条蒸馏路线 | <https://ai-beat.github.io/news/2026/08/diffusion-lm-two-roads> |
| AGRPO | dLLM 策略梯度 RL | <https://arxiv.org/html/2510.04019> |
| 量子位 scaling 主张 | Super Data Learner | <https://www.qbitai.com/2025/12/360425.html> |
| 框架综述 | 统一视角 | <https://arxiv.org/abs/2607.13431> |
