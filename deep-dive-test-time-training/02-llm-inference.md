# TTT 用于 LLM 推理与工程优化

## 概述

TTT 在 LLM 推理中的应用与序列建模层不同：**不改变模型架构，而是在推理时用测试输入微调现有模型**。主要方法包括：

1. **TTT-Nearest Neighbors**：用检索到的近邻训练
2. **TTT-Few-Shot**：用 few-shot prompt 中的示例训练
3. **LoRA-TTT**：只更新 LoRA adapter 参数
4. **TTRL**：用强化学习在 test time 训练

## TTT-Nearest Neighbors (TTT-NN)

### 核心思想

对于每个测试输入，从大规模索引中检索其最近邻，然后用这些近邻微调模型：

```mermaid
graph LR
    A[测试输入] --> B[检索近邻<br>~200M 向量]
    B --> C[微调模型<br>1 梯度步/样本]
    C --> D[用微调后的模型<br>处理测试输入]
```

### 系统架构

| 组件 | 实现 | 规模 |
|:---|:---|:---|
| **索引** | FAISS 分布式索引 | ~200M 向量，1TB 数据 |
| **嵌入模型** | RoBERTa-large | 355M 参数 |
| **基座模型** | GPT-2/GPT-Neo | 117M-1.3B 参数 |
| **检索时间** | 单次查询 | ~1 秒 |

### 算法流程

```python
# 伪代码
def ttt_nn(test_input, model, index, num_neighbors=20):
    # 1. 检索近邻
    neighbors = index.search(test_input, k=num_neighbors)
    
    # 2. 微调模型（每个邻居 1 步梯度）
    for neighbor in neighbors:
        loss = compute_lm_loss(model, neighbor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # 3. 用微调后的模型处理测试输入
    output = model(test_input)
    return output
```

### 实验结果

| 模型 | 任务 | 无 TTT | TTT-NN (20 neighbors) | 提升 |
|:---|:---|:---:|:---:|:---:|
| GPT-2 (117M) | Pile All | 1.07 | 0.85 | **-20.6%** |
| GPT-2 (117M) | GitHub | 1.95 | 0.51 | **-73.8%** |
| GPT-2 (117M) | Enron | 1.44 | 0.72 | **-50.0%** |
| GPT-Neo (1.3B) | Pile All | 0.95 | 0.83 | **-12.6%** |

**关键发现**：
- 即使只用 20 个邻居、每个只训练 1 步，也能显著提升性能
- 对代码生成任务提升最大（GitHub: -73.8%）
- 小模型 + TTT 可以接近大模型的性能

### 与 RAG 的对比

| 维度 | RAG (Retrieval-Augmented Generation) | TTT-NN |
|:---|:---|:---|
| **训练需求** | 需要训练时也使用检索 | 无需特殊训练 |
| **上下文长度** | 线性增长（检索内容加入 prompt） | 固定（微调模型权重） |
| **计算复杂度** | $O(n^2)$ attention on context | $O(k \times \text{forward+backward})$ |
| **内存占用** | KV cache 线性增长 | 固定 |

## TTT-Few-Shot

### 核心思想

用 few-shot prompt 中的示例作为训练数据，在推理前微调模型：

$$
\theta_{\text{test}} = \theta - \eta \sum_{i=1}^k \nabla_\theta \mathcal{L}(x_i, y_i)
$$

其中 $(x_i, y_i)$ 是 prompt 中的示例，$k$ 是示例数量。

### 理论保证

Gozeten et al. (2025) 证明：

**定理**：对于线性 Transformer，TTT 可以将 in-context learning 所需的样本数从 $\Omega(d)$ 降低到 $o(d)$。

**直觉**：标准 ICL 需要足够多的示例来"覆盖"任务空间，而 TTT 通过直接优化模型权重来"记忆"任务，所需示例更少。

### 实验验证

| 设置 | 标准 ICL | TTT (1 step) | TTT (5 steps) |
|:---|:---:|:---:|:---:|
| 200 示例 | 0.45 | 0.72 | 0.78 |
| 1000 示例 | 0.72 | 0.78 | 0.81 |

**结论**：TTT 用 200 示例就能达到标准 ICL 用 1000 示例的效果（5x 减少）。

## LoRA-TTT

### 核心思想

不更新整个模型，只更新低秩 adapter（LoRA）：

$$
\theta_{\text{test}} = \theta + BA
$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$。

### 优势

| 维度 | Full TTT | LoRA-TTT |
|:---|:---|:---|
| **可训练参数** | 全部 $\theta$ | 仅 $B, A$（~0.1%） |
| **内存占用** | 高 | 低 |
| **训练速度** | 慢 | 快 |
| **效果** | 最好 | 接近 full TTT |

### 实现要点

```python
# LoRA-TTT 实现
class LoRA_TTT(nn.Module):
    def __init__(self, model, rank=8):
        self.model = model
        self.lora_A = nn.Linear(d, rank, bias=False)
        self.lora_B = nn.Linear(rank, d, bias=False)
        
    def ttt_forward(self, x):
        # 原始前向
        h = self.model(x)
        # LoRA 增量
        delta = self.lora_B(self.lora_A(x))
        return h + delta
    
    def ttt_update(self, x, y, lr=1e-4):
        # 只更新 LoRA 参数
        loss = cross_entropy(self.ttt_forward(x), y)
        loss.backward()
        # 只更新 lora_A, lora_B
        self.lora_A.weight.data -= lr * self.lora_A.weight.grad
        self.lora_B.weight.data -= lr * self.lora_B.weight.grad
```

## 工程优化

### 问题：Token-wise 更新的瓶颈

朴素 TTT 逐 token 更新，导致：

1. **低并行度**：每个 token 的梯度计算是顺序的
2. **低硬件利用率**：$<5\%$ FLOPs 效率
3. **内存 I/O 瓶颈**：频繁读写 $W_t$

### Chunk-wise 更新（LaCT 范式）

将序列分成大 chunk（如 $C=512$），每个 chunk 做一次更新：

$$
G^{[r]} = -\nabla_W \left(\sum_{t=1}^C \eta_t \mathcal{L}(f_{W^{[r-1]}}(k_t), v_t)\right)
$$

$$
W^{[r]} = W^{[r-1]} + \beta M^{[r-1]} + G^{[r]}
$$

**优势**：
- 并行度：$C$ 个 token 可以并行计算梯度
- 硬件利用率：矩阵乘法充分利用 GPU/TPU
- 内存效率：只保存 $T/C$ 个 checkpoint

### 对偶形式（Dual Form）

将顺序的外积运算转化为矩阵乘法：

| 操作 | Primal form | Dual form |
|:---|:---|:---|
| **计算 $W_b$** | 顺序更新 $b$ 次 | 一次矩阵乘法 |
| **计算输出** | 顺序依赖 | 矩阵乘法 + mask |
| **内存** | 存储 $G_t, W_t$ | 只存最终 $W_b$ |
| **速度** | 基准 | 5x 加速 |

### 混合精度训练

TTT 的数值稳定性需要注意：

| 部分 | 精度 | 原因 |
|:---|:---|:---|
| **矩阵乘法** | bf16/fp16 | 利用 Tensor Core |
| **梯度累积** | fp32 | 防止数值下溢 |
| **LayerNorm** | fp32 | 数值稳定性 |
| **快权重更新** | fp32 | 防止精度损失 |

### Gradient Checkpointing

TTT 的隐藏状态是 $W_t$，标准实现保存所有中间状态，内存 $O(T \times d^2)$。

**优化**：只保存每个 mini-batch 结束时的 $W$，内存降到 $O(\kappa \times d^2)$，其中 $\kappa = T/b$。

```mermaid
graph LR
    subgraph "内存对比"
        A["朴素: W1, ..., WT<br>O(T×d²)"] --> B["Checkpoint: Wb, W2b, ...<br>O(κ×d²)"]
    end
```

### GPU Kernel 优化

TTT-MLP 的 GPU kernel 实现要点：

| 技术 | 实现 |
|:---|:---|
| **Tensor 并行** | 隐藏状态分片到多个 SM |
| **共享内存** | 使用 distributed shared memory |
| **流水线** | Input staging 和 pipelining |
| **混合精度** | bf16 matmul + fp32 累积 |

**代码示例**（ThunderKittens kernel）：

```cpp
// 简化的 TTT forward kernel
__global__ void ttt_forward_kernel(
    float* XQ, float* XK, float* XV,
    float* W, float* output,
    int batch_size, int seq_len, int d
) {
    // 每个 block 处理一个 mini-batch
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // 共享内存存储中间结果
    __shared__ float W_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    // 加载 W 到共享内存
    W_shared[tid][tid] = W[block_idx * d + tid];
    __syncthreads();
    
    // 计算输出: z = W @ q
    float z = 0;
    for (int i = 0; i < d; i++) {
        z += W_shared[tid][i] * XQ[block_idx * d + i];
    }
    output[block_idx * d + tid] = z;
}
```

## 性能基准

### 墙钟时间对比（A100 GPU）

| 模型 | 8k tokens | 16k tokens | 32k tokens |
|:---|:---:|:---:|:---:|
| Transformer | 0.28s | 0.91s | 3.21s |
| Mamba | 0.19s | 0.34s | 0.62s |
| **TTT-Linear** | 0.21s | 0.38s | 0.71s |
| **TTT-MLP** | 0.45s | 0.89s | 1.78s |

### 吞吐量对比

| 模型 | Prefill (tokens/s) | Decode (tokens/s) |
|:---|:---:|:---:|
| Transformer | 12,500 | 850 |
| **TTT-Linear** | 15,200 | 920 |
| **TTT-MLP** | 8,300 | 680 |

### 内存占用

| 模型 | 8k KV cache | 32k KV cache | TTT 状态 |
|:---|:---:|:---:|:---:|
| Transformer | 0.5 GB | 2.0 GB | N/A |
| **TTT-Linear** | N/A | N/A | 0.125 GB (固定) |
| **TTT-MLP** | N/A | N/A | 0.5 GB (固定) |

## 实践指南

### 何时使用 TTT？

```mermaid
graph TD
    A[你的任务是什么？] --> B{需要长上下文？}
    B -->|是| C{内存受限？}
    B -->|否| D[标准 Transformer]
    C -->|是| E[考虑 TTT]
    C -->|否| F[标准 Transformer]
    E --> G{需要个性化？}
    G -->|是| H[TTT-NN 或 TTT-FewShot]
    G -->|否| I[TTT-Linear 替代 Transformer]
```

### 实现清单

- [ ] 选择 TTT 变体：TTT-Linear（简单） vs TTT-MLP（更强）
- [ ] 设置 mini-batch 大小：$b=16$ 是推荐起点
- [ ] 选择学习率：TTT-Linear 用 $\eta=1.0$，TTT-MLP 用 $\eta=0.1$
- [ ] 实现对偶形式：提高 5x 训练速度
- [ ] 添加 gradient checkpointing：节省内存
- [ ] 混合精度训练：bf16 matmul + fp32 累积

### 常见陷阱

| 问题 | 原因 | 解决方案 |
|:---|:---|:---|
| **训练不稳定** | 学习率太大 | 降低 $\eta$，添加 warmup |
| **内存溢出** | 没用 checkpointing | 启用 time-wise gradient checkpointing |
| **速度慢** | 用 primal form | 实现 dual form |
| **效果差** | mini-batch 太大 | 减小 $b$，增加梯度步数 |

## Lab Exercise

### 实验 1：TTT-NN 检索与微调

```python
import faiss
import torch
from transformers import AutoModel, AutoTokenizer

# 1. 构建索引
index = faiss.IndexFlatIP(1024)  # 内积相似度
embeddings = load_pile_embeddings()  # 加载 Pile 数据集嵌入
index.add(embeddings)

# 2. 检索近邻
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModel.from_pretrained("roberta-large")

query = "def fibonacci(n):"
query_emb = model(**tokenizer(query, return_tensors="pt")).last_hidden_state[:, 0]
neighbors = index.search(query_emb.numpy(), k=20)

# 3. 微调 GPT-2
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = torch.optim.SGD(gpt2.parameters(), lr=2e-5)

for neighbor_text in neighbors:
    inputs = tokenizer(neighbor_text, return_tensors="pt")
    loss = gpt2(**inputs, labels=inputs["input_ids"]).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 4. 生成
output = gpt2.generate tokenizer(query, return_tensors="pt")["input_ids"], max_length=100)
print(tokenizer.decode(output[0]))
```

### 实验 2：对比 TTT 与标准推理

```python
# 对比不同方法的 perplexity
methods = {
    "baseline": lambda x: model(x),
    "ttt_nn": lambda x: ttt_nn(model, x, index, k=20),
    "ttt_fewshot": lambda x: ttt_fewshot(model, x, examples),
}

for name, method in methods.items():
    ppl = evaluate_ppl(method, test_data)
    print(f"{name}: PPL = {ppl:.2f}")
```

---

**下一篇**：[03-相关工作全景、前沿方向与实践指南](/2026/09/01/ttt-03-landscape-practical/) — DeltaNet/GLA/Mamba 统一视角、TTRL、MesaNet、决策树。
