---
title: "大模型位置编码（05）：长度外推（下）——YaRN、LongRoPE 与生产配置实操"
date: 2026-08-29 09:40:00 +0800
categories:
  - 位置编码
tags: [llm, positional-encoding, rope, yarn, longrope, rope-scaling, huggingface]
layout: post
mathjax: true
---

> **系列导航** ｜ 第 05 篇 / 共 08 篇
>
> [← 04 长度外推（上）](/2026/08/29/posem-04-extrapolation-pi-ntk/) ｜ [06 ALiBi →](/2026/08/29/posem-06-alibi/)

> **TL;DR**
>
> * **YaRN = 逐频插值 ramp + 注意力温度 + 高效微调**。三个组件各治一个病：ramp 函数按"波长/上下文比"逐维决定插值系数（低频全插、高频不动、中间线性过渡），解决 PI 一刀切与 NTK 谱斜率都不够细的问题；**注意力温度 $t = 0.1\ln s + 1$**（logits 除以 $t$）压住长上下文下 softmax 的熵膨胀——这是 YaRN 的独门发现：外推掉点不只来自相位错乱，还来自**注意力分布被拉平**；lazy migration 让 128k 微调只要约 400 步。
> * **LongRoPE**：把"逐维缩放系数"从 YaRN 的手工 ramp 升级为**进化搜索出的非均匀系数向量**，并配"两段式"搜索（搜索 + 8k 短窗微调 + 再搜索）；借助注意力 sink 现象，把上下文推到 **2048k**。核心洞察：最优缩放系数**不是**平滑的——某些维度上存在跳变，手工函数永远拟不出来。
> * **生产实操**：HF transformers 的 `rope_scaling` 配置（`linear`/`dynamic`/`yarn`/`longrope`）逐项给出，包括 KV cache 与 base 变更的相互作用、混合精度下 cos/sin 精度、以及"下载的模型 config 里已有 rope_scaling 时不要二次套用"这类高频踩坑。

---

## 1. 从 NTK 到 YaRN：还剩什么没治？

[04 篇](/2026/08/29/posem-04-extrapolation-pi-ntk/)结束时留了两个尾巴：

1. NTK-aware 的"改底数"只是把频率谱的斜率整体调平，**逐维的插值系数仍然是间接、非精确的**——最高频维严格不缩、最低频维近似缩 $s$ 倍，中间的过渡形状完全由几何级数形状决定，没有优化余地；
2. 所有的谱手术都只管相位，**不管注意力分布本身的漂移**：上下文变长后，softmax 的分母项数变多，注意力被摊薄（熵增大），即便每个打分值都"正确"，整体分布也与训练时的局部性先验不符。

YaRN（Peng et al., 2023，Nous Research）同时对这两点出手。

## 2. YaRN 的三个组件

### 2.1 逐频 ramp 插值

对维度 $i$，定义"外推压力"指标——该维波长与目标上下文的比：

$$
r_i = \frac{\lambda_i / (2\pi)}{L_{\text{train}}} = \frac{1}{L_{\text{train}}\,\theta_i}
$$

$r_i \gg 1$：波长远大于训练长度（低频维）→ 必须插值；$r_i \ll 1$：波长短、训练中已充分见过各种相位（高频维）→ 保持原样。YaRN 用一个带两个超参（$\alpha=32$、$\beta=1$，论文默认）的 **ramp 函数**逐维给出插值比例：

$$
\gamma(r) =
\begin{cases}
0, & r < \beta \;(\text{高频，不插})\\[2pt]
\dfrac{r-\beta}{\alpha-\beta}, & \beta \le r \le \alpha \;(\text{线性过渡})\\[6pt]
1, & r > \alpha \;(\text{低频，全插，即 PI 式 } \theta_i/s)
\end{cases}
\qquad
\theta_i' = \theta_i\cdot\Big(1-\gamma_i + \frac{\gamma_i}{s}\Big)
$$

即每维的频率在"原值"与"$1/s$ 倍"之间按 $\gamma_i$ 取凸组合。与 NTK-aware 的关键区别：**过渡的边界和形状由波长直接决定且可调（$\alpha, \beta$），而 NTK 只能调一个底数**。论文证明这个形式在 $\alpha\to\infty$ 时退化为 PI、在特定极限下近似 NTK-aware——它是前两代的严格推广。

### 2.2 注意力温度：治熵膨胀

YaRN 最有启发性的发现：只做频率手术，长上下文的 KV cache 端还有一个系统性问题——key 数量随长度线性增长，softmax 注意力的**熵随长度增大**（哪怕打分分布形状不变，归一化本身就摊薄了权重），而模型在训练中建立的一切"注意力预算"先验都是按 4k 的熵水平校准的。

对策简单到不像话：给 logits 除一个温度

$$
\boxed{\;t = 1 + 0.1\,\ln s\;}, \qquad
\text{logits} \;\to\; \frac{\text{logits}}{t}
$$

$s=16$ 时 $t \approx 1.28$，16 倍上下文只压 28% 的 logits 尺度。论文的消融显示这一项贡献显著——它不是玄学修补，而是对"注意力分布随长度的重整化"的显式补偿（一个类比：温度 0.7 的采样让长尾更尖，$1/t<1$ 的推理温度同理让摊薄的注意力重新聚焦）。后续大量长上下文工作（包括 DeepSeek 的部分长程配置）都沿用了这个技巧。

### 2.3 Lazy migration：400 步微调 128k

数据策略：微调时跳过"困惑度已经低于阈值的 token"（模型已会的部分不再花算力），长上下文数据用量降至 Code Llama 长上下文方案的 **1/10**，步数（400 步 vs 10000+ 步）、效果全面占优。Mistral-7B 微调到 128k、Llama-2-7B 微调到 128k 的配方均开源。

## 3. LongRoPE：把插值系数交给搜索

Liu et al.（Microsoft，2024）的观察：YaRN 假设最优插值系数沿频率维度**平滑单调**，但实际测量推翻了这一点——用 perplexity 直接做目标去搜每一维的缩放系数，得到的解呈现**非均匀、非单调**的模式（论文还发现把两维一组搜索的"成对搜索"效果更好，单个维度独立搜索会互相打架）。

LongRoPE 的三件套：

1. **进化搜索（evolutionary search）**逐维搜索缩放系数 $\boldsymbol{s} \in \mathbb{R}^{d/2}$，目标函数是插值后模型在长序列上的 perplexity。搜索空间里同时允许"整体系数 × 逐维扰动"，以 YaRN 式解为初始化；
2. **两段式扩展**：先搜 256k 的系数并微调（8k 短窗微调即可，用"anchor"——即 attention sink——时甚至不微调也行），然后在 256k 模型上**再搜 2048k 的系数**，直接零训练推理 2048k；
3. **注意力 sink / anchor tokens**：远距离注意力集中砸在开头几个 token 上（"sink"现象，Streaming LLM 系列工作的发现），LongRoPE 借此保持远程依赖的稳定性。

结果：Llama-2-7B（4k 训练）→ 2048k 上下文，长文档 perplexity 与检索任务都保持可用。工程上最大的贡献是把"插值系数"从超参变成了**可以离线搜索产出的数据**——这也是它进 HF transformers（`rope_scaling.type="longrope"`，系数放在模型 config 里随 checkpoint 分发）的原因。

## 4. 生产实操：HF transformers / vLLM 配置速查

### 4.1 rope_scaling 类型对照

```python
# 1) PI（线性插值）——需配合长上下文微调过的 checkpoint
{"rope_type": "linear", "factor": 8.0}

# 2) Dynamic NTK——零训练，推理时随长度自适应（老版本写 "type" 而非 "rope_type"）
{"rope_type": "dynamic", "factor": 8.0}   # factor 为上限倍数

# 3) YaRN——常配合官方已微调的 yarn checkpoint
{"rope_type": "yarn", "factor": 16.0,
 "original_max_position_embeddings": 4096,
 "beta_fast": 32, "beta_slow": 1,          # 即 α、β
 "attention_factor": 1.2833}               # 也可留空让库按 t=0.1·ln s+1 计算

# 4) LongRoPE——系数由搜索产出，存在 config 的 longrope_scaling 里
{"rope_type": "longrope", "factor": 208.0,
 "original_max_position_embeddings": 4096,
 "longrope_short_factor": [...], "longrope_long_factor": [...]}
```

注意各框架的历史包袱：transformers 4.44 之前字段名是 `type`，之后改 `rope_type`；vLLM 参数名又是另一套（`--rope-scaling` CLI / `rope_scaling` in serving config）。混用旧教程抄配置是新手中头号坑。

### 4.2 高频踩坑清单

1. **二次套用**：现在的主流长上下文模型（Qwen2.5-128k、Llama-3.1-128k 等）的 base 与 rope_scaling 已烧进 config 并在预训练时使用——推理时**不要再改任何 rope 参数**；只有"拿短上下文模型自己做扩展"才需要这些配置；
2. **KV cache 与旋转的不可分离性**：已按旧频率旋转并缓存的 K，无法直接迁移到新频率（[04 篇](/2026/08/29/posem-04-extrapolation-pi-ntk/) Dynamic NTK 一节的讨论）。因此动态缩放方案在长流式会话里要么提前定死 factor，要么接受缓存重算；
3. **数值精度**：$m\theta_i$ 在 $m\sim 10^5$ 时 FP16 的 mantissa 不够用，主流库在 FP32 里算 cos/sin。自研推理引擎时这是必查项；
4. **训练长度 ≠ 有效长度**：config 里的 `max_position_embeddings` 说的是位置编号范围，模型**实际**的长程能力要用 needle-in-haystack / RULER 之类的检索基准验证——尤其微调不足时，常见"困惑度正常、检索全错"的隐性失败（长上下文的注意力熵问题通常先杀检索、后杀困惑度，与 2.2 节的温度机理一致）。

### 4.3 一个最小验证脚本

```python
from transformers import AutoConfig, AutoModelForCausalLM
import torch

cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
cfg.rope_scaling = {"rope_type": "yarn", "factor": 16.0,
                    "original_max_position_embeddings": 4096}
cfg.max_position_embeddings = 4096 * 16
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", config=cfg,
    torch_dtype=torch.bfloat16, device_map="auto")
# 注意：此模型权重从未按 yarn 训练过——该配置仅用于验证加载链路；
# 要得到质量，还需按 YaRN 论文做约 400 步 lazy-migration 微调。
```

## 5. 阶段总结

外推这条线到此收束成一张完整地图：

| 代际 | 粒度 | 是否管分布 | 训练成本 |
|---|---|:---:|---|
| PI | 所有维一刀切 | ✗ | ~1000 步 |
| NTK / Dynamic | 谱斜率（1 个参数） | ✗ | 0–少量 |
| YaRN | 逐维 ramp（2 超参） | ✓ 温度 | ~400 步 |
| LongRoPE | 逐维自由系数（搜索） | ✓（沿用 sink/温度） | 搜索 + 8k 窗微调 |

从"缩放"到"重整化"，从"手工函数"到"搜索产物"——这条演进线的尽头是一个更大的问题：**位置编码本身能不能设计得天生可外推？** 这正是 [06 篇](/2026/08/29/posem-06-alibi/)ALiBi（从 2021 年就给出了"训练 1k 推理 16k"的答案）以及后续 CoPE（Contextual Position Encoding）等新方案要讨论的话题。

**Lab 练习**：
1. 复现 2.2 节的熵膨胀：用 [04 篇](/2026/08/29/posem-04-extrapolation-pi-ntk/)的熵实验脚本，先按 YaRN 公式对 logits 除 $t=0.1\ln s+1$ 再观察熵曲线是否被拉回训练长度水平；
2. 用 `transformers` 分别以 `linear`、`dynamic`、`yarn`（factor=4）加载同一个 4k 模型，对同一段 15k 文本各画"各位置的困惑度曲线"（滑窗逐段计算），对比三种方案掉点的**起始位置**——你会直观看到 dynamic 前段无损、后段崩，linear 全程均匀劣化，yarn 整体最平。

## 参考文献

1. Peng, Quesnelle, Fan, Povey, *YaRN: Efficient Context Window Extension of Large Language Models*, 2023. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)（配套 [HF 博客](https://huggingface.co/blog/yarn)）
2. Liu et al., *Extending LLM's Context Window to 2048K via LongRoPE*, 2024. [arXiv:2402.13753](https://arxiv.org/abs/2402.13753)
3. Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023（attention sink / anchor）. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
4. Chen et al., *Position Interpolation*, 2023. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)
5. HF Transformers rope_scaling 文档. [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers/main_classes/text_generation)
