# 06 推理系统与加速：把“理论上的并行”变成真实的 TPS

> 系列文档：扩散语言模型（Diffusion LM）学习资料。本文为第 06 篇，聚焦推理系统与加速；上承块扩散与混合范式，下接实测与部署。

## 1. 为什么“并行解码”一开始没换来速度

离散扩散在纸面上“全序列并行去噪”，但早期实现并未线性转化为吞吐，根因是 **计算瓶颈的迁移**：

- 小 batch、短序列时，瓶颈在 **memory-bound**（频繁搬运权重与激活），并行解码省下的 FLOP 被访存吞掉；
- 大 batch、长序列时，瓶颈才转到 **compute-bound**，此时并行解码的收益才真正显现 [1]。

因此评估 dLLM 必须同时看三件套指标，单看“并行度”会误判：

| 指标 | 全称 | 含义 | 何时关键 |
| --- | --- | --- | --- |
| NFE | Number of Function Evaluations | 生成全程消耗的前向次数 | 衡量算法步数效率 |
| tokens-per-forward | 每次前向解码 token 数 | 单步并行产出 | 衡量硬件利用率 |
| TPS | Tokens Per Second | 端到端每秒 token | 最终用户体验 |

三者口径不同：少 NFE 不等于高 TPS，若每次前向只多产几个 token 但 cache 失效，TPS 仍上不去 [1][4]。一个常见误判是拿 batch size 1 的 tok/s 当通用结论——此时系统往往还停在 memory-bound 区，换大 batch 后名次可能反转 [1][4]。

一个直觉类比：并行解码像把“串行小任务”合并成“并行大任务”，只有当单次大任务足够填满算力时，合并才划算；小任务合并只会增加调度与访存开销 [1]。

## 2. 块级 KV Cache：双向注意力与朴素 cache 的不兼容

双向注意力要求每个 token 看到全部上下文，朴素 KV cache 假设“前缀不变”，在掩码去噪的每一步几乎全部 token 都在变化，于是 cache 失效 [1][3]。

近似兼容的三种思路：

- **prefix cache**：只缓存尚未变化的块外前缀 KV，块内每步仍全量前向 [1]；
- **dual cache**：额外缓存被掩码的后缀 token，用 `replace_position` 只更新发生变化位置的 KV [1]；
- **分层 cache**：块级 cache 存历史、子块 dual cache 支持块内并行，兼顾块间因果与块内并行 [3]。

Fast-dLLM v1 的核心观察是：相邻去噪步之间，prefix / suffix 的 KV 激活 **余弦相似度接近 1**，因此可“近似复用”而非每次重算 [1]。其工程做法是把输出区切成块、自左向右逐个求解：进块前预缓存块外 KV，块内每步只对当前块做前向 [1]。

下面三档是 Fast-dLLM v1 报告的训练无关加速（论文自测口径）：

| 档位 | 机制 | 加速区间 |
| --- | --- | --- |
| 无 cache | 标准双向前向 | 1.0× |
| prefix cache | 缓存块外前缀 KV | 约 2.0–3.3× |
| DualCache | 额外缓存被掩码后缀、按位更新 | 约 2.0–3.6× |

注意 DualCache 未必总高于 prefix cache：后缀 token 多时额外显存与更新开销会抵消收益，需按序列长度取舍 [1]。

### Fast-dLLM v2：从预训练 AR 微调更省

Fast-dLLM v2 反过来证明“从预训练 AR 出发”更省：只需约 1B tokens 微调（对比 Dream 的 580B tokens，约 1/500）即可把预训练 AR 改成块扩散 dLLM [2]。关键改动有三：

- **token shift**：用前一个 token 的 logit 预测掩码 token，保留 AR 的位置特征；
- **block-wise causal attention mask + complementary masks**：保证每个位置都被学到，避免块边界信息漏学；
- **分层 cache**：块级 cache 存历史，子块 dual cache 支持块内并行 [2]。

报告上，7B 相对 Qwen2.5-7B-Instruct 约 **2.54× 吞吐**；单 A100 上 batch size 1 约 **102.5 tok/s**、batch size 4 约 **217.5 tok/s**（论文自测口径）[2]。这一定调提示：若已有 AR 底座，走“AR 微调成块扩散”比从头训扩散更经济 [2]。

v1 与 v2 的差异点明一条路线分野：v1 是“训练无关、改推理”，v2 是“少量微调、改结构”，二者并非互斥，可叠加使用 [1][2]。

## 3. 置信度感知并行解码与 NFE 压缩

置信度感知并行解码（confidence-aware parallel decoding）只对被模型“高置信度”预测的掩码位做提交，低置信度位留待下一步，从而在不破坏质量的前提下压缩有效 NFE [1][4]。

阈值设定是关键：阈值过高则并行度退化回近串行，过低则错误传播需更多修正步。实践中阈值常随去噪步数退火——靠近干净端放宽、靠近噪声端收紧 [1][4]。

SGLang 的 LowConfidence 把该阈值暴露为可配置项，便于在不同模型上复用同一套退火策略，而不必为每个模型重写道解码循环 [4]。

Fast-dLLM v1 把该策略与 KV cache 组合，在 LLaDA / Dream 上报告端到端最高 **27.6× 吞吐提升**（论文自测口径）[1]。SGLang 侧以 `--dllm-algorithm LowConfidence` 与 `--dllm-algorithm-config` 把算法与框架代码解耦，便于替换策略而不动后端 [4]。

## 4. 蒸馏：把步数压下去

蒸馏是 dLLM 加速的主线之一：

- **progressive distillation**：每次递归把采样步数减半；
- **self-distillation through time**：沿时间维度自蒸馏；
- **discrete consistency distillation**：离散一致性蒸馏，把多步去噪对齐到单步 [6]。

这三类本质都可解释成一种 **on-policy 训练**：用模型自身在当前分布上的采样结果做监督，而非离线数据 [6]。

现实部署还需权衡大 batch、少步：

| 选择 | 收益 | 风险 |
| --- | --- | --- |
| 步数压到极少 | NFE 大幅下降 | 长程一致性受损 |
| 大 batch 蒸馏 | 吞吐高、方差低 | 显存与采样覆盖 |
| 配合置信度解码 | 步数进一步可省 | 需调阈值防错误传播 |

少步蒸馏后必须用置信度解码兜底，否则 NFE 的下降会换来质量塌方 [1][6]。

## 5. 长上下文与稀疏

把 dLLM 推到 128K+ 上下文时，三个彼此牵连的方向浮现：

| 方向 | 做法 | 与双向注意力的耦合点 |
| --- | --- | --- |
| 上下文裁剪 | 丢弃低贡献历史块 | 双向全局依赖使裁剪更敏感 |
| 稀疏注意力 | 限制每 token 可见集合 | 需保留块内双向与块间因果 |
| 位置外推 | RoPE / 位置编码外推 | 双向下位置对称性假设与 AR 不同 |

双向注意力的位置对称性假设与 AR 不同：AR 的位置是严格有序的，扩散的掩码过程对位置更“对称”，因此 RoPE 在双向下的外推行为需要单独验证，不能照搬 AR 的经验 [7]。

这也意味着：把 AR 上验证过的位置外推配方直接套到双向 dLLM，可能得到看似合理、实则退化的长程生成，需以长文档采样而非 perplexity 为主要判据 [7]。

HKUST 博士资格考 survey 摘要把“长上下文扩到 128K+ 后再谈稀疏注意力与量化”列为三个方向之一（**二手摘要，引用时说明**）[7]。

## 6. 量化与低精度部署

量化是 dLLM 落消费级硬件的关键。DiffusionGemma 实例（**Google 官方口径，厂商数据**）：

| 项目 | 数值 |
| --- | --- |
| 模型结构 | 26B MoE，推理仅激活 3.8B |
| 量化后显存 | 可塞进 18GB VRAM 消费级 GPU |
| NVFP4 版本 | 单张 RTX 5090 可跑 |
| 官方吞吐 | H100 上 1000+ tok/s、RTX 5090 上 700+ tok/s |
| 对比 | 约比同尺寸自回归 Gemma 4 快 4× |

NVFP4 等低精度格式在双向注意力下需额外校验：量化误差可能在长程依赖上被放大，导致生成中途“跑题”而 AR 下不明显 [5]。建议量化后跑长上下文生成样本，而非只测短 prompt 的 PPL [5][7]。

NVFP4 的 4-bit 权重加较高精度累加，对 MoE 的路由专家尤其敏感；若某专家激活稀疏却被激进量化，可能出现偶发的语义跳变，需做专家级量化敏感度分析 [5]。

## 7. 服务框架支持现状

| 框架 | dLLM 支持 | 关键特性 | 后端 / 版本 |
| --- | --- | --- | --- |
| dInfer（蚂蚁开源） | 是 | 拆分为 model / iteration manager / decoder / KV-cache manager 四可替换组件 | LLaDA 走 vLLM==0.10.2；LLaDA2 走 sglang==0.5.3.post1 |
| SGLang | 是 | `--dllm-algorithm` 解耦算法；以 block（如 32 tokens）为单位的流式输出 | 支持 LLaDA2 类模型 |
| vLLM | 部分 | 经 dInfer 间接支持 LLaDA | 见 dInfer 封装 |

dInfer 的四组件拆分让“算法组合”变容易：想换解码策略只改 decoder，想换 cache 只改 KV-cache manager，模型权重不动 [3]。它已支持 block diffusion 与批量推理、量化版本，覆盖 LLaDA、LLaDA-MoE、LLaDA2 [3]。

后端版本被显式钉死（vLLM 0.10.2 / sglang 0.5.3.post1），说明 dLLM 对框架内部 KV 处理逻辑依赖很深，升级后端需做回归测试，不能简单跟随主版本号 [3]。

dLLM 的流式输出粒度与 AR 不同：AR 逐 token 流式，dLLM 以 **block 为单位** 出，客户端需按块缓冲渲染，否则会出现“长时间静默后成块吐出”的体感 [4]。

有报道称 LLaDA2.0-flash-CAP 在某任务上 **935 tok/s** 对比 gpt-oss-120B 的 **263 tok/s**（**二手厂商口径，未核验**）[4]。

## 8. 与投机解码互操作

dLLM 与投机解码（speculative decoding）有两种组合方向：

- **dLLM 当 draft**：利用 dLLM 的并行性产出多候选草稿，供 AR target 验证，即“DLLMs as complementary drafts”路线 [7]；
- **dLLM 当 target**：AR 出草稿、dLLM 验证，或结合第 05 篇的 self-speculation 模式（扩散出草稿、AR 验证，温度 0 下相对纯 AR 无损）[05 篇][7]。

二者共享“并行草稿 + 串行验证”的骨架，差异只在谁提供草稿、谁做最终接受。dLLM 当 draft 时，其并行产出天然匹配投机解码“一次出多候选”的结构，是更有潜力的组合 [7]。

值得强调：dLLM 当 target 时，验证端仍是 AR 的贪心或采样逻辑，因此 self-speculation 与经典投机解码在“AR 验证”这一步是同一套机制，差异仅在草稿来源 [7]。

## 9. 性能体检清单

自行实测 dLLM 吞吐时，建议按以下口径，避免常见误判：

1. **固定指标口径**：同时报 NFE、tokens-per-forward、TPS，不要只报“加速倍数”；
2. **固定 batch 与长度**：1 vs 多 batch 的 TPS 不可直接比（开篇的 memory/compute-bound 迁移）；
3. **区分厂商 / 二手**：NVIDIA、Google 的 tok/s 多为厂商自测，标注“未独立核验”；
4. **流式粒度**：确认框架是逐 token 还是逐 block 流式，影响体感延迟；
5. **质量兜底**：压步数 / 量化后务必测长程一致性，防止 NFE 下降换来质量塌方 [1][4][5][7]。

实测时优先复现同 batch、同长度的对照，再谈架构优劣；跨框架比较务必对齐 `--dllm-algorithm` 与块大小等超参 [1][4]。若报告来自厂商博客，应在表格或脚注显式标注“未独立核验”，避免被当作中立基准引用 [1][4][5]。

最后提醒：TPS 是端到端结果，受网络、预处理、流式缓冲共同影响，单看模型前向速度会高估真实收益，应以完整请求链路的端到端计时为准 [1][4]。

综上，dLLM 加速是“算法（蒸馏 / 置信度解码）+ 系统（KV cache / 量化 / 框架）”的协同工程，单点优化若不与其它环节配套，往往难以转化为可观测的 TPS 提升 [1][3][4]。

## 参考链接

[1] Fast-dLLM v1（ICLR 2026，训练无关 KV cache）：https://nvlabs.github.io/Fast-dLLM 、https://arxiv.org/abs/2505.22618 、结构说明 https://deepwiki.com/NVlabs/Fast-dLLM/1.1.1-kv-cache-system
[2] Fast-dLLM v2（从预训练 AR 微调、token shift、分层 cache）：https://nvlabs.github.io/Fast-dLLM/v2
[3] dInfer（蚂蚁开源，四组件拆分）：https://github.com/inclusionAI/dInfer （镜像 https://gitee.com/natt1e/dInfer ）
[4] SGLang 支持与流式口径（含 935 tok/s 二手厂商数据）：https://www.yingzheng.com/review/sgllang-empowers-llada-2-0
[5] DiffusionGemma 量化与 NVFP4（Google 官方厂商数据）：https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation 、https://developers.googleblog.com/diffusiongemma-the-developer-guide/
[6] 蒸馏（progressive / self-distillation through time / discrete consistency）：https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/
[7] 与投机解码互操作及 128K+ 长上下文（HKUST 博士资格考 survey 二手摘要）：https://cse.hkust.edu.hk/pg/defenses/S26/gheaf-28-04-2026.html
