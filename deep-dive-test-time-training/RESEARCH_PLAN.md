# Test-Time Training (TTT) Deep-Dive Research Plan

## Research Scope
Test-Time Training (TTT) 是一种在推理时通过自监督学习更新模型参数的技术，使模型能够适应特定测试输入。本调研覆盖 TTT 的两条主线：序列建模层（TTT as RNN layer）和 LLM 推理增强（TTT for inference），并深入数学原理、工程优化与实际应用。

---

## Document Tree

> 注：原始 9 篇方案已按用户要求合并为 4 篇正文（内容不丢失，只做分组整合）。

```
deep-dive-test-time-training/
├── 00-overview.md                    # TTT 领域全景图与知识图谱 + 基础概念
├── 01-math-sequence.md               # 核心数学推导 + TTT 作为序列建模层
├── 02-llm-inference.md               # TTT 用于 LLM 推理增强 + 工程优化
├── 03-landscape-practical.md         # 相关工作全景 + 前沿方向 + 实践指南
├── PROGRESS.md                       # 进度跟踪
├── README.md                         # 阅读指南
└── RESEARCH_PLAN.md                  # 本计划文档
```

---

## Key Questions This Research Must Answer

1. **What is TTT?** 核心思想：为什么要在 test time 训练？解决了什么问题？
2. **How does TTT work mathematically?** 内循环（fast weights）与外循环（slow weights）的数学形式化
3. **What are the different TTT variants?** Token-wise, chunk-wise, mini-batch, dual form 的区别与权衡
4. **How does TTT relate to other efficient sequence models?** DeltaNet, GLA, Mamba, RetNet 的统一视角
5. **What are the engineering challenges?** 内存 I/O 瓶颈、硬件利用率、并行化策略
6. **How is TTT applied to LLMs?** TTT-NearestNeighbors, TTT-FewShot, LoRA-TTT 的实现差异
7. **What are the theoretical guarantees?** TTT 改善 in-context learning 的理论分析
8. **What are the practical deployment considerations?** 何时选择 TTT vs. alternatives？性能基准？

---

## Detailed Coverage Points per Document

### 00-overview.md - TTT 领域全景图
- TTT 定义与核心直觉（"test time 训练一个模型来处理当前输入"）
- 两条发展主线：序列建模层 vs LLM 推理增强
- 历史脉络：2020 Sun et al. → 2024 TTT-Linear/MLP → 2024 TTT-NN → 2025 LaCT/E2-TTT/MesaNet
- 知识图谱 Mermaid 图
- 与其他 test-time compute 方法（CoT, Best-of-N, TTRL）的关系

### 01-ttt-fundamentals.md - 基础概念
- 统一视角：所有序列建模层 = 隐藏状态 + 更新规则
- Fast weights vs slow weights 概念
- TTT 作为 fast weight programming 的特例
- 与 Dynamic Evaluation 的关系
- 自监督任务设计：reconstruction loss, next-token prediction
- 变量映射表

### 02-ttt-sequence-modeling.md - TTT 作为序列建模层
- TTT-Linear：隐藏状态为线性模型
- TTT-MLP：隐藏状态为两层 MLP
- 前向传播：output rule + update rule
- 与 Transformer 和 Mamba 的对比（125M-1.3B scale）
- 长上下文表现：perplexity 持续下降 vs Mamba 16k 饱和
- 内存 I/O 挑战

### 03-ttt-llm-inference.md - TTT 用于 LLM 推理增强
- TTT-Nearest Neighbors (Hardt & Sun, ICLR 2024)
- TTT-Few-Shot (Akyürek et al., ICML 2025)
- LoRA-TTT 与 Visual TTT
- qTTT (query-only TTT)
- Self-Guided TTT
- 与 standard fine-tuning, ICL 的对比

### 04-ttt-math-formulation.md - 核心数学推导
- 统一数学框架：隐藏状态 $W_t \in \mathbb{R}^{d \times d}$
- 自监督损失：$\ell(f_W(k), v)$
- 梯度更新：$W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)$
- 外循环（meta-training）与内循环（test-time）
- 对偶形式（Dual Form）推导：提高 5x 训练速度
- 线性 vs 非线性 fast weights

### 05-ttt-engineering.md - 工程优化
- Token-wise 更新的并行化瓶颈
- Mini-batch 更新策略
- Chunk-wise 更新（LaCT 范式）
- 内存 I/O 分析：计算密度 vs 内存带宽
- 硬件利用率：为何 token-wise <5% FLOPs
- GPU/TPU 实现考量

### 06-ttt-landscape.md - 相关工作全景
- DeltaNet (Schlag et al., 2021)：delta rule + 线性 attention
- GLA (Yang et al., 2024)：gated linear attention
- Mamba (Gu & Dao, 2023)：selective state spaces
- RetNet (Sun et al., 2023)：retention mechanism
- 统一视角：test-time regression framework (Wang et al., 2025)
- Titans (Behrouz et al., 2024)：momentum in batched GD
- 性能对比表

### 07-ttt-research-frontiers.md - 前沿方向
- TTRL (Test-Time Reinforcement Learning)：RL 在 test time 的应用
- 理论分析 (Gozeten et al., 2025)：TTT 如何改善 Transformers as ICL
- MesaNet：最优 fast weight programming，CG 方法
- LaCT：大 chunk 更新 + 滑动窗口注意力
- E2-TTT：表达性与效率的平衡，闭式 scalar kernel
- VDS-TTT：验证器驱动的样本选择

### 08-ttt-practical-guide.md - 实践指南
- 决策树：何时使用 TTT？
- 与 alternatives 的权衡（full fine-tuning, LoRA, ICL）
- 实现要点：损失设计、学习率、初始化
- 性能基准汇总
- 常见陷阱与调试技巧

---

## Search Strategy

**Round 1 (已完成)**: 全局探索 - arxiv survey, 2024-2025 papers
**Round 2 (进行中)**: 数学原理 - TTT 变体公式化, dual form, convergence
**Round 3**: 工程实现 - GitHub 代码库, CUDA kernel, parallelization
**Round 4**: 中文社区 - 知乎/掘金解读, 实践经验
**Round 5**: 对抗验证 - TTT vs alternatives, limitations, failure modes

---

## Estimated Effort
- 9 documents × ~500-800 lines each = ~4500-7200 lines total
- 5 rounds of deep search
- Mermaid diagrams: ~15-20 figures
- Math derivations: ~30-40 display equations
- Variable mapping tables: ~10 tables

## Status
- [x] Phase 1: Research Plan created
- [ ] User confirmation (pending)
- [ ] Phase 2: In-depth research execution
- [ ] Phase 3: Self-review
- [ ] Phase 4: Final delivery
