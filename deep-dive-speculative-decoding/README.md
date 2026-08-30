# 投机解码（Speculative Decoding）业内现状 —— 深度调研系列

> 工程优先视角的投机解码全景调研：从原理数学、方法谱系、EAGLE 家族深挖，到推理引擎支持矩阵、生产部署陷阱、业界头部实践、性能数据与选型权衡、未来方向。
>
> - 输出语言：中文（技术术语/论文标题保留英文）
> - 定位：给 **负责大模型推理/训练推理协同的工程师** 的一份可落地参考，而非纯论文综述。
> - 补充阅读：`RESEARCH_PLAN.md`（研究计划与知识图谱）、`PROGRESS.md`（进度清单）。

---

## 文档地图

```mermaid
graph LR
    subgraph 入门
        A["01 综述与演化史<br>为什么需要 从哪来 到哪去"]
    end
    subgraph 原理
        B["02 核心技术原理<br>数学推导 无偏性证明"]
        C["03 EAGLE 家族深挖<br>Medusa EAGLE MTP"]
    end
    subgraph 工程落地
        D["04 推理引擎支持矩阵<br>5 引擎配置与示例"]
        E["05 生产部署陷阱<br>收益折损 硬约束"]
        F["06 业界实践<br>DeepSeek Meta Together 端侧"]
    end
    subgraph 决策与前沿
        G["07 评测与选型权衡<br>失效区 何时别用"]
        H["08 未来方向<br>训练推理协同 开放问题"]
    end
    A --> B --> C --> D --> E --> F --> G --> H
```

## 各文档导航

| 文档 | 一句话内容 | 适合谁先读 |
|---|---|---|
| [01-overview-and-evolution.md](01-overview-and-evolution.md) | 从自回归瓶颈到 2023 原始投机解码，再到 2024–2026 方法谱系的演化史与分类框架（survey 2502.19732） | 刚接触投机解码的任何人 |
| [02-core-methods.md](02-core-methods.md) | draft/verify 数学框架、Rejection Sampling 无偏性证明、接受率-加速比推导 $S = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(1+\gamma c)}$、时序图、PyTorch 最小实现 | 需要理解"为什么无损"与性能边界的读者 |
| [03-eagle-family-deep-dive.md](03-eagle-family-deep-dive.md) | Medusa 多分支头 → EAGLE-1/2/3 → DeepSeek-V3 MTP 的逐代差异与实验增益 | 计划选型 EAGLE 系自投机的读者 |
| [04-inference-engine-support.md](04-inference-engine-support.md) | vLLM / SGLang / TensorRT-LLM / llama.cpp / Hugging Face 的算法支持、配置项、draft 模型获取、最小可运行示例 | 决定"用哪个引擎来跑"的读者 |
| [05-production-deployment.md](05-production-deployment.md) | 生产与实验室 40–60% 差距的成因、词汇表一致性等硬件约束、陷阱清单、红绿灯规则 | 要把投机解码上生产的读者 |
| [06-industry-practice.md](06-industry-practice.md) | DeepSeek-V3 MTP、Meta Llama-at-Scale、Together ATLAS/Turbo、端侧 sd.npu 等真实生产案例 | 想参照头部团队方案与踩坑的读者 |
| [07-benchmarks-and-tradeoffs.md](07-benchmarks-and-tradeoffs.md) | 论文数字横向对比表、草稿来源/验证/lookahead/batch 四维权衡、"何时不要用"清单、数字读数检查点 | 正在做选型权衡与采购/排期决策的读者 |
| [08-future-directions.md](08-future-directions.md) | 训练-推理协同、动态/多草稿、端侧检索式、投机缩放定律、开放问题清单 | 关注技术演进的读者 |

## 三条阅读路径

1. **快速总览**（30 分钟）：01 → 07 的结论表与负向清单 → 08 的开放问题 → 需要细节时回头。
2. **生产落地**（半天）：05 → 06 → 04 → 07。先知道收益会打几折，再选引擎与算法。
3. **研究视角**（一天）：02 → 03 → 07 → 08 的论文引用链，按需复现核心公式。

## 核心结论速览

- **原理**：投机解码用草稿快速生成 $K$ 个 token、目标模型一次前向并行验证，Rejection Sampling 保证输出分布与目标模型逐 token 采样**无偏等价**。
- **收益天花板**：由草稿-验证成本比 $c$ 主导，$\lim_{\gamma\to\infty} S = 1/(1+c)$；大 batch 失效区是 $(batch, seq, model, hw)$ 的四元函数（MagicDec 拐点）。
- **生产现实**：论文最高 3–6×，生产实测普遍收敛在 **1.4–2.5×**；差距来源是 batch/并发/验证开销，部分可通过调度与动态策略缓解。
- **选型主线**：独立小模型（成本高上限高）→ 自投机 EAGLE/MTP（免双模型，主流）→ 无模型检索（强复用场景零成本）。
- **生态重要变化**：DeepSeek-V3 把投机能力内建为训练目标（MTP），llama.cpp 2026-04 统一 CLI（`--spec-type`）将其列为一等公民——投机正从"后装件"变成"预装件"。

## 配套资源

- 研究计划与知识图谱：`RESEARCH_PLAN.md`
- 进度与状态：`PROGRESS.md`
- 本系列全部引用均为原生链接；个别无法验证的引用在对应文档中已标注 unverified。

---

*生成时间：2026-08-30。本系列采用 deep-dive 工作流，最终交付以 PROGRESS.md 的 Phase 3/4 状态为准。*