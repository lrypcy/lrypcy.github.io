# 03 EAGLE 家族深度解析：Medusa → EAGLE → EAGLE-2/3 与 DeepSeek-V3 MTP

> 系列文档：投机解码（Speculative Decoding）业内现状
> 上一篇：[02 核心技术原理与分类](02-core-methods.md) | 下一篇：[04 开源推理引擎支持矩阵](04-inference-engine-support.md)

---

## 1. 从"要不要草稿模型"到"草稿从哪来"

[02 核心原理](02-core-methods.md) 已证明：加速比的第一决定因素是**接受率 $\alpha$**。2024 年之前的方案要么用独立小模型（部署贵、接受率受分布漂移限制），要么用 n-gram（免训练但 $\alpha$ 低）。EAGLE 家族的贡献在于把接受率推上了一个台阶，且把这个能力做成"目标模型自带"。

本家族的演进主线：

```mermaid
graph LR
    M["Medusa<br>多解码头<br>arXiv 2401.10774"] --> E["EAGLE<br>特征级自回归<br>arXiv 2401.15077"]
    E --> E2["EAGLE-2<br>动态草稿树<br>arXiv 2406.16858"]
    E --> E3["EAGLE-3<br>training-time test<br>arXiv 2503.01840"]
    E2 --> E3
    E3 --> MTP["DeepSeek-V3 MTP<br>训练目标内建<br>arXiv 2412.19437"]
```

---

## 2. Medusa：用多解码头"自掏腰包"（2024-01, ICML 2024）

### 2.1 动机与设计

Medusa 的目标是绕开"准备并部署外部 draft 模型"的麻烦，直接在目标模型顶部挂 $K$ 个**解码头**（decoding head），分别预测第 $t+1,\dots,t+K+1$ 位的 token，再用**树注意**并行验证多条候选 [1](https://arxiv.org/abs/2401.10774)。

**解码头结构**（极其轻量）：

$$p_t^{(k)} = \text{softmax}\Big( h_t W_k^{(out)} \Big), \qquad h_t' = h_t + \text{GeLU}(h_t W_k^{(1)})W_k^{(2)}$$

即每个头是一个**带残差连接的单层 FFN** + 输出投影 [1](https://arxiv.org/abs/2401.10774)。$h_t \in \mathbb{R}^{d}$ 为主干最后一层隐藏状态，$W_k^{(1)}\in\mathbb{R}^{d\times d_{ff}}$、$W_k^{(2)}\in\mathbb{R}^{d_{ff}\times d}$、$W_k^{(out)}\in\mathbb{R}^{d\times V}$。相比独立草稿模型（完整 transformer 任意层），Medusa 头参数量与一个 LM head 同级。

### 2.2 训练策略两档

| 方案 | 冻结主干? | 加速 | 数据需求 |
|---|---|:---:|---|
| Medusa-1 | 冻结 | >2.2× | 需要训练数据；无数据时用 **self-distillation**（以主干生成数据训练头） |
| Medusa-2 | 联合微调 | 2.3–2.8× | 需特殊训练配方保护主干能力 |

### 2.3 Typical acceptance（关键工程取舍）

Medusa 提出**典型接受**：不用严格的 $\min(1,p/q)$（需要草稿头的 $q$ 分布），而是当目标模型对候选 token 的概率"足够可信"（典型性在阈值内）即接受 [1](https://arxiv.org/abs/2401.10774)。这大幅抬高 $\alpha$ 且降低解码方差，但**放弃了严格无偏**。工程上以可控的质量损失换取更稳的加速，是"无损 vs 够快"的明确取舍样本（生产评估见 [05](05-production-deployment.md)）。

---

## 3. EAGLE：把自回归搬到特征空间（2024-01, ICML 2024）

### 3.1 两个关键观察

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）[2](https://arxiv.org/abs/2401.15077) 提出两个被后续工作反复引用的结论：

1. **特征层自回归比 token 层自回归"更简单"**：这里"特征"指目标模型的 **second-to-top-layer**（LM head 前的倒数第二层隐藏状态）。相比 token 序列（自然语言的离散变换），特征序列更具规律性。实验：同样的草稿器在特征层做自回归比在 token 层做自回归快 1.9× vs 1.5×。
2. **特征自回归受采样随机性制约**：目标模型从分布采样 token 引入随机性，而特征是高维连续量，无法"采样"。若用 $f_I$ 预测下一个 token，则无法区分"am"还是"always"两条特征路径。

**解法：把 token 序列提前一个时间步喂给草稿模型（feature & shifted-token）**。即预测"always"时输入 $f_I$ + $t_{\text{always}}$。加入采样结果后，加速从 1.9× 提升到 2.8× [2](https://arxiv.org/abs/2401.15077)。

### 3.2 草稿模型结构（数学 + Shape）

EAGLE 草稿模型仅三块 [2](https://arxiv.org/abs/2401.15077)：

1. **Embedding 层**（复用目标模型参数，不训练）；
2. **LM Head**（复用目标模型参数，不训练）——草稿特征经它得到 token 分布；
3. **自回归头（Autoregression Head）**：一个 FC 层 + 一个解码器层。

**前向数学描述**（设 batch $B$、序列长 $L$、隐藏维 $d$）：

$$\mathbf{z}_i = \text{FC}\big([\,\mathbf{f}_i \,;\, \text{Emb}(t_{i+1})\,]\big), \qquad \hat{\mathbf{f}}_{i+1} = \text{DecoderLayer}\big(\mathbf{z}_{\le i}\big)$$

其中 $[\cdot\,;\,\cdot]$ 为拼接，$t_{i+1}$ 是**提前一位的 token**。

| 张量 | Shape | 说明 |
|---:|:---:|:---|
| $\mathbf{f}_i$ （second-to-top-layer 特征序列） | $(B, L, d)$ | 目标模型逐层提取 |
| $t_{i+1}$（移位 token 序列） | $(B, L)$ | 采样结果显式注入 |
| Emb 输出 | $(B, L, d)$ | 与特征拼接 |
| 拼接后 | $(B, L, 2d)$ | 特征-语义融合 |
| FC 后 | $(B, L, d)$ | 降维回 $d$ |
| $\hat{\mathbf{f}}$（预测特征） | $(B, L+1, d)$ | 解码器输出 |
| LM Head 后 logits | $(B, V)$ | 得到草稿 token 分布 |

### 3.3 训练与收益

- 训练数据仅 **2–4B tokens**（对比独立草稿模型 TinyLLaMA 需 3000B tokens [2](https://arxiv.org/abs/2401.15077)），训练成本两个数量级更低，且**无需修改目标模型**。
- 结果（示例）：MT-bench 上比 baseline 快 **2.1–3.8×**、比 Lookahead 快 1.7–2.1×、比 Medusa 快 1.5–1.6×；LLaMA2-Chat 70B 上延迟加速 **2.7–3.5×**、吞吐翻倍 [2](https://arxiv.org/abs/2401.15077)。
- 关键数字对比：EAGLE 草稿接受准确率 ≈ **0.8**，Medusa ≈ 0.6，Lookahead 更低 [2](https://arxiv.org/abs/2401.15077)——这就是"$\alpha$ 上台阶"的证据。
- 完全无损：验证阶段保证输出分布与目标模型一致。

### 3.4 为什么特征级更准？（直觉解释）

token 分布是特征的**确定性函数**（softmax(LM head·f)）。特征里"记住"了序列的语义轨迹、风格、格式等 token 层面被压缩掉的信息。草稿模型在特征空间自回归 = 在更丰富的表示上做预测，等价于把"猜词"升级为"续写语义"，因此同样参数下 $\alpha$ 更高。EAGLE-2/3 的核心改进都是在这个框架上进一步推高 $\alpha$。

---

## 4. EAGLE-2：动态草稿树，让好草稿长得更深（2024-06, EMNLP 2024）

### 4.1 观察：接受率不仅依赖位置，还依赖上下文

EAGLE 与 Medusa 使用**静态草稿树**：每层固定展开 $k$ 个候选，隐含假设"接受率只与树中位置有关"。EAGLE-2 系统测量发现：同一位置上接收率方差极大——接受率是**上下文相关的** [3](https://arxiv.org/abs/2406.16858)（论文 Fig.5：P1 位置接受率最高、P6 最低表明了位置依赖，但同位置的方差揭示了上下文依赖）。

### 4.2 关键发现：草稿模型是"校准良好"的

EAGLE 草稿模型输出的置信度（confidence score）可以近似接受率，且误差很小 [3](https://arxiv.org/abs/2406.16858)（Fig.6：置信度 <0.05 的 token 接受率约 0.04；置信度 >0.95 的 token 接受率约 0.98）。这让它**不调用目标模型**就能估算每个草稿 token 的真实被接受概率——这是动态树的可行基础。

### 4.3 动态草稿树：展开 + 重排

定义草稿树中节点 $t_i$ 的**全局接受率（value）**为其到根路径上所有节点置信度的乘积：

$$V_i = \prod_{u \in \text{path}(\text{root} \to t_i)} c_u$$

- **展开阶段**：只选当前层全局接受率 top-$k$ 的节点继续加深（避免每层指数爆炸）。
- **重排阶段**：由于实际被接受需要整条路径接受，不能只看局部置信度——所有候选按 $V_i$ 重排，取 top-$m$ 作为验证输入。
- **注意力掩码**：按树结构调整掩码，保证每个 token 只能看到其祖先节点（不能看到兄弟分支），与朴素自回归完全一致。

EAGLE-2 **不需要额外训练**（草稿模型完全不变，只改树结构调度）。结果：加速比 **3.05–4.26×**，比 EAGLE-1 快 20–40%（约 1.3×），MT-bench 上约为 Medusa 的 2 倍速 [3](https://arxiv.org/abs/2406.16858)。

---

## 5. EAGLE-3：放弃特征预测，直接预测 token（2025-03, NeurIPS 2025）

### 5.1 动机：EAGLE 系列随数据扩展收益递减

Li et al. 发现：扩大训练数据对 EAGLE 的收益有限。根因是 **feature prediction 约束**——训练损失同时包含特征预测损失 $L_{\text{fea}}$ 与 token 预测损失 $L_{\text{token}}$，特征回归约束限制了草稿模型的表达能力，使其**无法从更多数据中获益** [4](https://arxiv.org/abs/2503.01840)。

### 5.2 两处改动

1. **弃用特征预测，直接预测 token**：草稿模型输出直接是 token 分布（经 LM head），去掉 $L_{\text{fea}}$。这带来一个副作用——训练时草稿模型输出 $\hat{a}_{t+1}$ 与真实特征 $f_{t+1}$ 偏差增大，推理时把 $\hat{a}$ 反馈回输入会导致**训练-推理分布失配**、第二个草稿 token 接受率骤降。

2. **Training-time test 技术**：训练时就把上一时刻的**模型输出**（而非真实特征）反馈回输入，模拟自回归推理路径（类似 RNN 时代的 scheduled sampling，但配合树注意掩码）。这让模型在训练阶段就"见过"自己的错误轨迹，消除失配 [4](https://arxiv.org/abs/2503.01840)。

3. **多层特征融合**：输入从"仅顶层特征"变为**低/中/高层特征的融合**——顶层特征天然偏向 next-token 预测，不适合多步草稿；中层融合提供更丰富的语义信息 [4](https://arxiv.org/abs/2503.01840)。

### 5.3 收益

- 最高加速 **6.5×**，相对 EAGLE-2 提升约 **1.4×**；
- 训练数据扩大约 **8×** 后仍持续受益（EAGLE-2/HASS 在该数据规模下已陷入平台期，Fig.8）；
- 打破"大 batch 下投机无用"的常识：**SGLang 中 bs=64 时吞吐提升 1.38×（+38%）** [4](https://arxiv.org/abs/2503.01840)；
- 训练成本：约 2 卡天级别即可为 Llama-3 系列训练，代码开源 [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)，官方推荐 [SpecForge](https://github.com/sgl-project/SpecForge) 在 SGLang 生态中开箱训练。

> ⚠️ **引用勘误提示**：EAGLE-3 的 arXiv 编号是 **2503.01840**（2025-03-03 提交，NeurIPS 2025 录用）。早期社区流传的 2412.xxxx 编号有误，以官方 GitHub 引用的 abspdf 链接为准。

---

## 6. DeepSeek-V3 MTP：把投机能力写进训练目标（2024-12）

### 6.1 MTP 是什么

DeepSeek-V3 在预训练阶段引入 **Multi-Token Prediction (MTP)** 目标：每个位置顺序预测后续 $D$ 个 token，保持完整因果链 [5](https://arxiv.org/abs/2412.19437)。与 Gloeckle et al.（2024，多 token 并行独立头）不同，DeepSeek 的 MTP **按深度顺序预测**，与 EAGLE 的 eagerness principle 类似 [5](https://arxiv.org/abs/2412.19437)。

### 6.2 MTP 模块的数学结构

第 $k$ 个 MTP 模块由共享 Embedding $\mathrm{Emb}(\cdot)$、共享输出头 $\mathrm{OutHead}(\cdot)$、一个 Transformer 块 $\mathrm{TRM}(\cdot)$、投影矩阵 $M_k \in \mathbb{R}^{d \times 2d}$ 组成 [5](https://arxiv.org/abs/2412.19437)。第 $k$ 深度、位置 $i$ 的表示：

$$\mathbf{h}_i^{'(k)} = M_k\big[\,\mathrm{RMSNorm}(\mathbf{h}_i^{k-1});\; \mathrm{Emb}(t_{i+k})\,\big]$$

$$\mathbf{h}_i^{k} = \mathrm{TRM}(\mathbf{h}_i^{'(k)})  \quad\xrightarrow{\text{共享 } \mathrm{OutHead}}\quad  p_i^{k} = \mathrm{Softmax}(\mathrm{OutHead}(\mathbf{h}_i^{k}))$$

**MTP 训练损失**（加权平均）：

$$\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D}\sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^{k}, \qquad \mathcal{L}_{\text{MTP}}^{k} = -\frac{1}{T}\sum_{i=2+k}^{T+1} \log P_i^{k}[t_i]$$

### 6.3 双用途设计（训练 + 推理）

- **训练期**：辅助目标增强主模型在基准上的表现（MTP 密集化训练信号、帮助表征预规划）；
- **推理期**：可以直接丢弃 MTP 模块使主模型独立工作，也可**复用作投机解码**进一步降低延迟 [5](https://arxiv.org/abs/2412.19437)。

> 生态信号：DeepSeek-V3 权重约 685B = 主模型 671B + MTP 模块 14B；HuggingFace `transformers` 已支持 `num_mtp_layers` 配置与 `generate(..., use_mtp=True)` [6](https://huggingface.co/docs/transformers/main/model_doc/deepseek_v3)。这是"投机能力内建训练目标"从论文走向落地的明确标志（更多生态见 [04](04-inference-engine-support.md)）。

---

## 7. 家族横向对比

| 方法 | 草稿来源 | 额外训练 | 动态树 | 无偏 | 峰值加速（论文报） | 关键创新 |
|---|---|:---:|:---:|:---:|:---:|---|
| 经典 SD [7](https://arxiv.org/abs/2211.17192) | 独立小模型 | 否 | 否 | ✓ | 2–3× | 范式开创 |
| Medusa [1](https://arxiv.org/abs/2401.10774) | 多解码头 | 是（轻量） | 否 | typical 可选 | 2.3–2.8× | 免外部模型 |
| EAGLE [2](https://arxiv.org/abs/2401.15077) | 特征级单层解码器 | 是（2–4B tokens） | 否 | ✓ | 3.8× (MT-bench) | 特征级自回归 |
| EAGLE-2 [3](https://arxiv.org/abs/2406.16858) | 同 EAGLE | 否（复用） | **是** | ✓ | 3.05–4.26× | 动态草稿树 |
| EAGLE-3 [4](https://arxiv.org/abs/2503.01840) | 直接 token + 多层特征 | 是（~8× 数据） | 续用 | ✓ | **6.5×**；bs64 吞吐 1.38× | training-time test |
| DeepSeek-V3 MTP | 训练目标模块 | 预训练已含 | 否（模块级） | ✓ | 用于提升主模型+投机 | 训练-推理协同 |

**选型要点**：
- **已有独立小模型且不介意部署成本** → 经典 SD 够用，换来最低接入成本；
- **单卡本地 / bs=1、想免外部模型** → Medusa 简单直接（但 typical acceptance 非严格无偏）；
- **生产服务追求最高 α** → EAGLE-3 + SGLang 是当前公认组合（详见 [04](04-inference-engine-support.md)/[06](06-industry-practice.md)）；
- **希望投机能力"免费"内建** → 关注 MTP 系模型（DeepSeek-V3/R1 生态）。

---

## 8. 最小复现资源与验证路径

EAGLE-3 训练与推理的最低门槛（详细配置见后续文档与官方仓库）：

1. **推理（已有权重）**：SGLang 一行启用 `--speculative-algorithm EAGLE3 --speculative-draft-length N`（需先下载对应 `--eagle3-weight` 或经 SpecForge 训练）。
2. **训练（自训草稿头）**：SpecForge 提供开箱训练流程，约 2 卡天即可覆盖 Llama-3 系列。
3. **验证接受率**：用框架日志观测 `accept_length`，对照第 [02](02-core-methods.md) 节的 $\alpha \to \mathbb{E}[\text{tokens}]$ 曲线核对收益预期。

---

### 参考文献

1. Cai, et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. arXiv:2401.10774; ICML 2024. https://arxiv.org/abs/2401.10774
2. Li, Wei, Zhang, Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. arXiv:2401.15077; ICML 2024, PMLR 235:28935–28948. https://arxiv.org/abs/2401.15077
3. Li, Wei, Zhang, Zhang. *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*. arXiv:2406.16858; EMNLP 2024, pp.7421–7432. https://arxiv.org/abs/2406.16858
4. Li, Wei, Zhang, Zhang. *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*. arXiv:2503.01840; NeurIPS 2025. https://arxiv.org/abs/2503.01840
5. DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437. https://arxiv.org/abs/2412.19437
6. HuggingFace Transformers, DeepSeek-V3 文档（num_mtp_layers / use_mtp）. https://huggingface.co/docs/transformers/main/model_doc/deepseek_v3
7. Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding*. arXiv:2211.17192; ICML 2023. https://arxiv.org/abs/2211.17192