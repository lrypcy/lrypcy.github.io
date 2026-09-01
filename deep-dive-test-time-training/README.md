# Test-Time Training (TTT) Deep-Dive

> 在推理时训练模型：从序列建模层到 LLM 推理增强的完整进阶路线

## 阅读指南

本系列共 4 篇文档，从入门到实践循序渐进。建议按顺序阅读。

### 学习路径

```mermaid
graph LR
    A[00-全景图<br>入门] --> B[01-数学原理<br>核心]
    B --> C[02-LLM推理<br>应用]
    C --> D[03-相关工作<br>进阶]
```

| 顺序 | 文档 | 主要内容 | 前置要求 |
|:---:|:---|:---|:---|
| 1 | [00-overview.md](00-overview.md) | TTT 领域全景图：定义、历史、两大路线 | 无 |
| 2 | [01-math-sequence.md](01-math-sequence.md) | 核心数学推导：fast weights、内/外循环、TTT-Linear/MLP 实现 | 基础线性代数、梯度下降 |
| 3 | [02-llm-inference.md](02-llm-inference.md) | TTT-NN、TTT-FewShot、LoRA-TTT、工程优化 | 了解 Transformer、LoRA |
| 4 | [03-landscape-practical.md](03-landscape-practical.md) | DeltaNet/GLA/Mamba 统一视角、TTRL、Titans、实践决策 | 完成前三篇 |

### 各文档核心问题

| 文档 | 你要能回答的问题 |
|:---|:---|
| **00** | TTT 是什么？与 RAG、ICL 有何不同？ |
| **01** | TTT 的隐藏状态为什么是矩阵？dual form 如何加速 5x？ |
| **02** | 如何在 LLM 推理中应用 TTT？有哪些工程陷阱？ |
| **03** | TTT 与 DeltaNet/Mamba/Titans 有何统一视角？何时选哪个？ |

## 关键概念速查

| 概念 | 一句话总结 | 详见 |
|:---|:---|:---|
| **Fast weights** | 在 test time 更新的"隐藏状态"模型 | 00, 01 |
| **Slow weights** | 预训练模型参数，test time 冻结 | 00 |
| **双偶形式** | 将顺序更新转化为矩阵乘法，5x 加速 | 01 |
| **TTT-NN** | 检索近邻微调模型，代码生成 -70% | 02 |
| **Delta rule** | 先擦除旧关联再写入新关联 | 03 |
| **遗忘机制** | Titans 的 weight decay，解决记忆饱和 | 03 |

## 快速上手

### 最小 TTT-Linear 实现

```python
import torch

def ttt_linear_forward(x, W_init=0, eta=1.0):
    """最小 TTT-Linear 前向传播
    x: (T, d) token 序列
    """
    T, d = x.shape
    W = torch.zeros(d, d) if W_init == 0 else W_init
    outputs = []
    for t in range(T):
        z = W @ x[t]                     # 输出规则
        grad = 2 * (W @ x[t]).outer(x[t]) / d  # reconstruction 梯度
        W = W - eta * grad               # 更新规则
        outputs.append(z)
    return torch.stack(outputs), W
```

### 尝试实验

每篇文档末尾有 **Lab Exercise**。建议按以下顺序尝试：

1. **01**: 验证 TTT-Linear 更新公式、mini-batch 大小影响
2. **02**: 实现 TTT-NN 检索微调、对比推理方法
3. **03**: 对比 TTT 与 DeltaNet、测试遗忘机制

## 关键论文索引

| 论文 | 年份 | 核心贡献 | 本系列中的位置 |
|:---|:---:|:---|:---|
| [TTT (Sun et al.)](https://arxiv.org/abs/2006.11061) | 2020 | 首次提出 TTT 概念 | 00 |
| [TTT-NN (Hardt & Sun)](https://arxiv.org/abs/2305.18466) | 2024 | LLM 推理增强 TTT | 02 |
| [TTT-Linear/MLP (Sun et al.)](https://arxiv.org/abs/2407.04620) | 2024 | TTT 作为序列建模层 | 00, 01 |
| [Titans (Behrouz et al.)](https://arxiv.org/abs/2501.00663) | 2024 | 遗忘机制 + 动量 | 03 |
| [Test-Time Regression (Wang et al.)](https://arxiv.org/abs/2501.12352) | 2025 | 统一框架 | 03 |
| [Gated DeltaNet (Yang et al.)](https://arxiv.org/abs/2412.06464) | 2025 | 门控 + Delta rule | 03 |

## 未验证声明

以下内容引用了无法在当前环境直接验证的论文细节，标记为"未验证"：

- TTT-Linear/MLP 各上下文长度下的精确 perplexity 数值（来自论文摘要及公开资料）
- MesaNet 的 CG 方法与 Gated Linear Attention 的精确等价关系
- E2-TTT 闭式 kernel 的理论推导细节

建议查阅原始论文确认。

## 进度

见 [PROGRESS.md](PROGRESS.md)
