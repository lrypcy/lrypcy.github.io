---
title: "扩展篇 E3：部署侧视角——vLLM / SGLang 对量化算法的开箱支持现状"
date: 2026-09-19 00:22:00 +0800
categories:
  - 模型量化
tags: [llm-inference, quantization, vllm, sglang, deployment, kernel, llm-compressor, nvfp4]
layout: post
mathjax: true
---

> **系列导航** ｜ [课程路线图](/quantization-roadmap/) ｜ **扩展篇 E3** ｜ 不与主线同序
>
> 姊妹篇：[E2 发展史与收敛地图](/2026/09/19/llm-quant-E2-history-convergence-map/)（算法侧）｜ 本篇是**部署侧**｜ [23 统一视角](/2026/08/29/llm-quant-23-unified-view/)

> **TL;DR**
>
> * **核心结论**：量化算法的「论文分数」和「部署分数」是两套分，而且**经常不一致**。判定一个算法能不能用的真正标准不是它的 PPL，而是**四件事能不能同时成立**：① 有没有官方工具链把它写成 checkpoint；② 推理框架有没有对应的 `QuantizationConfig`；③ 有没有匹配你 GPU 架构的 kernel 后端；④ 这条路径有没有被 CI 覆盖。缺任何一条，它就只是「能加载」，不是「能用」。
> * **反直觉发现**：① **AQLM 被 vLLM 移除了**——vLLM v0.10.1 的 breaking change 明确写着 “Removed AQLM quantization support”，而 VPTQ 从未进入过主干。这是 [E2 篇](/2026/09/19/llm-quant-E2-history-convergence-map/) §4.3 那个判断最硬的证据：**码本/向量量化路线的精度再好，打不过 INT4 Tensor Core 的工程惯性**。② **llm-compressor 里没有 AQLM 和 VPTQ，但有 SpinQuant 和 QuIP**——官方工具链的选择已经替你做了一轮筛选：旋转类进了，码本类没进。③ **同一份 FP8 checkpoint 在 Ada 和 Hopper 上走的是完全不同的 kernel**（Ada 只有 Marlin 一条路，Hopper 上有 CUTLASS / DeepGEMM / Triton 三条），所以「FP8 很快」这句话必须带上限定语：在哪一代卡上、哪个后端、哪个 batch。
> * **本篇定位**：如果说 [E2 篇](/2026/09/19/llm-quant-E2-history-convergence-map/)回答「这个算法处于收敛地图的哪个格子」，本篇回答「这个格子在你的机器上跑不跑得起来」。所有支持矩阵均来自 vLLM / SGLang / llm-compressor 的公开文档与仓库状态（**截至 2026-09**），并在 §11 给出核对命令——这类信息变化极快，**动手前请自己再跑一次**。

---

## 0. 名词表

| 名词 | 是什么 | 备注 |
|---|---|---|
| **Marlin** | vLLM 的 INT4/INT8/FP8 权重量化 kernel | Turing+（SM 7.5+），**CUDA-only**；Turing 不支持 Marlin MXFP4 |
| **Machete** | Hopper 上的 mixed-input GEMM kernel | 专为 W4A16 设计，用 TMA 把反量化开销藏在访存里 |
| **FlashInfer** | JIT 生成的 attention / GEMM kernel 库 | 按 batch size 与 head dim 在运行时生成代码 |
| **DeepGEMM** | DeepSeek-V3 的 block-wise FP8 GEMM | 128×128 块尺度，FP32 累加防溢出 |
| **CUTLASS / Triton** | 通用 GEMM 后端 | 兜底路径，通常不是最快但最稳 |
| **compressed-tensors** | llm-compressor 的输出容器格式 | vLLM 一等公民，**所有官方量化产物的落点** |
| **NVFP4 / MXFP4** | Blackwell 上的 4-bit 块浮点 | 需要 SM100（B200/GB200） |
| **W4AFP8 / W4A8** | INT4 权重 + FP8 激活 | 需要 Hopper（SM 9.0+） |
| **Aiter** | AMD ROCm 上的加速 kernel 集合 | SGLang 在 AMD 上大量依赖它 |
| **Humming** | llm-compressor v0.13.0 的任意位宽打包 | 支持 3/5/6/7 bit 无浪费打包 |

---

## 1. 为什么要单独写这一篇

[E2 篇](/2026/09/19/llm-quant-E2-history-convergence-map/)把 PTQ/QAT 分成「已封冻 / 正在收敛 / 未收敛」，但那个分类是**算法侧**的。到了部署侧，判据完全不同。

看一个真实例子：

```text
算法侧：AQLM 在 2-bit 上精度非常好，论文数字漂亮，HuggingFace 上有官方 checkpoint
部署侧：vLLM v0.10.1 的 breaking change 直接写着 "Removed AQLM quantization support"
```

一个算法被**从主干里删掉**，说明它在部署栈里已经没有活路了。这种信息不会出现在任何一篇算法论文的 related work 里，但它决定了你今天能不能用。

反过来也有：

```text
算法侧：RTN 是最朴素的 baseline，论文里永远是对照组
部署侧：llm-compressor 的官方教程第一步就是 data-free 的 W8A8-FP8（RTN），
        vLLM 文档把它列为推荐的入门路径
```

**这两套排序不一致，是本篇存在的全部理由。**

---

## 2. 四层分级模型

我用一个四层模型来组织「开箱即用」的程度：

| 层级 | 判定标准 | 意味着什么 |
|---|---|---|
| **T0 一等公民** | 官方工具链可产出 + 框架有 Config + **多架构 kernel** + CI 覆盖 | 直接 `vllm serve` 就能跑，性能符合预期 |
| **T1 硬件受限** | 前两条成立，但**只支持特定架构**（如 NVFP4 只上 Blackwell） | 卡对了就开箱即用，卡不对直接报错 |
| **T2 能跑没收益** | 能加载，但走通用/回退 kernel，或需要手工转换 | 显存省了，**延迟可能反而变慢** |
| **T3 不支持** | 框架无对应实现，或已被移除 | 只能自己在 HF Transformers 里跑研究用途 |

**关键提醒**：T2 是最危险的层级——它「看起来能用」，但你在 benchmark 上会发现量化后吞吐没提升甚至下降，然后误以为「量化没用」。

---

## 3. vLLM 支持矩阵（截至 2026-09）

### 3.1 按算法 × 硬件

来源：vLLM 官方 `docs/features/quantization/README.md` 的硬件兼容表。
（Volta=SM7.0，Turing=SM7.5，Ampere=SM8.0/8.6，Ada=SM8.9，Hopper=SM9.0）

| 实现 | Volta | Turing | Ampere | Ada | Hopper | AMD | Intel GPU | x86 CPU | Arm CPU |
|---|---|---|---|---|---|---|---|---|---|
| AWQ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| GPTQ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Marlin（GPTQ/AWQ/FP8/FP4） | ❌ | ✅* | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| llm-compressor INT8 (W8A8) | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| llm-compressor INT8 (**W4A8**) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| llm-compressor **FP8 (W8A8)** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| BitsAndBytes | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

\* Turing 不支持 Marlin MXFP4。

**三行最该记住的读法**：

1. **FP8 W8A8 需要 Ada（SM 8.9）起步**——Ampere（A100）跑不了 FP8，只能走 INT8 或 INT4。这是很多团队第一次上 FP8 就踩的坑。
2. **W4A8（INT4 权重 + INT8 激活）在 GPU 上全军覆没，只有 Arm CPU 支持**。所以 [E2 篇](/2026/09/19/llm-quant-E2-history-convergence-map/)里说的「W4A8 是甜点位」目前主要是**研究结论**，不是生产结论；生产上能落的是 **W4AFP8**（INT4 权重 + FP8 激活，需 Hopper+）。
3. **Marlin 是 CUDA-only**——AMD 与 CPU 路径上 AWQ/GPTQ 走的是别的 kernel。

### 3.2 按算法 × kernel 后端

vLLM 在初始化时按「量化方法 × 平台 × 环境变量」选后端（见 `vllm/model_executor/layers/quantization/fp8.py`、`mxfp4.py`）：

| 场景 | 优先级 1 | 优先级 2 | 优先级 3 |
|---|---|---|---|
| FP8，SM89+ | CUTLASS | DeepGEMM（block） | Triton |
| FP8，SM<89 | Marlin | — | — |
| FP8，ROCm | AITER | Triton | — |
| MXFP4（无 LoRA），SM100 | FlashInfer | MXFP8 / FlashInfer BF16 | Marlin |
| MXFP4（无 LoRA），SM90 | FlashInfer BF16 | Triton | Marlin |
| MXFP4（带 LoRA） | Triton | Marlin | — |
| GPTQ / AWQ（CUDA） | Marlin | — | — |

MoE 层还有一套独立选择逻辑（SM100 优先 FlashInfer+TRTLLM/CUTLASS；SM90 优先 FlashInfer/CUTLASS；SM<89 回退 Marlin；block 量化优先 DeepGEMM），可用 `VLLM_USE_DEEP_GEMM`、`VLLM_MOE_USE_DEEP_GEMM`、`VLLM_USE_FLASHINFER_MOE_FP8` 干预。

> **工程含义**：同一份 FP8 checkpoint，在 A100 上只有 Marlin 一条路，在 H100 上有三条路，在 B200 上又有别的优先级。**「FP8 提速 X 倍」这句话如果不带架构，是没有意义的。**

### 3.3 典型模型的实际配方（社区/厂商验证路径）

| 模型族 | 配方 | 权重/激活 | vLLM 实现 |
|---|---|---|---|
| DeepSeek-V3 / R1 | FP8 block-wise | W8A8 (E4M3) | DeepGEMM（MoE + MQA logits） |
| Kimi K2 Thinking | 原生 INT4（QAT） | W4A16 | compressed-tensors + INT4 kernel |
| GPT-OSS (120B/20B) | MXFP4 | W4A8 (FP4/FP8) | Marlin MXFP4（MoE） |
| Llama 3.1 / 3.3 | FP8 (FBGEMM) | W8A8 | Machete（Hopper） |
| Qwen 2.5 / Mistral | GPTQ / AWQ | W4A16 | Marlin（Ampere+） |
| 通用 | AutoRound | W4A8 | llm-compressor 插件 |

### 3.4 已被移除 / 从未进入主干的

| 算法 | 状态 | 说明 |
|---|---|---|
| **AQLM** | **已移除** | vLLM v0.10.1 breaking change：“Removed AQLM quantization support”，官方建议迁移到 BitsAndBytes 或 ModelOpt |
| **VPTQ** | 未入主干 | 2-bit 以下向量量化，仅 HF Transformers 侧可用 |
| **2:4 稀疏** | 已移除 | llm-compressor 明确写道：“no longer supported due to lack of hardware support and user interest” |

**这三条合起来是一个很强的信号**：精度优势敌不过工程惯性。码本、向量量化、结构化稀疏——三个在论文里都很漂亮的方向，在 2026 年的生产栈里全部退出。

---

## 4. SGLang 支持矩阵（截至 2026-09）

SGLang 的机制与 vLLM 类似（注册式 `QuantizationConfig`），但**跨平台覆盖更激进**，且在 AMD/Ascend 上的差异必须单独看。

| 方法 | NVIDIA | AMD (MI300X/MI325X/MI350X) | Ascend (A2/A3) | 备注 |
|---|---|---|---|---|
| fp8 | ✅ | ✅ | WIP | AMD 走 Aiter 或 Triton |
| mxfp4 | ✅ | ✅（需 CDNA3/4） | WIP | 用 Aiter blockwise |
| w8a8_int8 | ✅ | ✅ | ❌ | Triton 实现，两平台通用 |
| w8a8_fp8 | ✅ | ✅ | ❌ | AMD 走 Aiter/Triton |
| awq | ✅ | ✅（Triton dequant） | ✅（CANN） | NVIDIA 是优化过的 CUDA kernel |
| gptq | ✅ | ✅（Triton 或 vLLM kernel） | ✅（CANN） | 同上 |
| compressed-tensors | ✅ | ✅（FP8/MoE 有 Aiter 路径） | 部分（FP8 尚未） | — |
| auto-round | ✅ | ✅ | 部分 | Intel 出品，平台无关 |
| **awq_marlin / gptq_marlin** | ✅ | ❌ | ❌ | **Marlin 是 CUDA-only** |
| gguf | ✅ | ❌ | ✅ | CUDA-only kernel（Ascend 上预反量化） |
| modelopt / modelopt_fp8 | ✅（SM90+） | ❌ | ❌ | 需 NVIDIA 硬件 |
| **modelopt_fp4** | ✅（**SM100+**） | ❌ | ❌ | Blackwell 原生 FP4 |
| **petit_nvfp4** | ❌ | ✅（MI250/MI300X/MI325X） | ❌ | **AMD 上的 NVFP4 路径** |
| bitsandbytes | ✅ | 实验性 | ❌ | 依赖 bnb 的 ROCm 支持 |
| torchao（int4wo 等） | ✅ | 部分（int4wo 不支持） | ❌ | — |
| quark / quark_int4fp8_moe | ❌ | ✅ | ❌ | AMD 专用 |
| modelslim / mxfp8(diffusion) | ❌ | ❌ | ✅ | Ascend 专用 |

**与 vLLM 的三个结构性差异**：

1. **SGLang 把 AMD 当一等公民**：`petit_nvfp4`、`quark`、`quark_int4fp8_moe` 都是 AMD-only 路径；NVIDIA 的 `modelopt_fp4` 与 AMD 的 `petit_nvfp4` 是同一目标的两种实现。
2. **Marlin 是 vLLM 的护城河**：`awq_marlin` / `gptq_marlin` 在 SGLang 上 CUDA-only，AMD 用户只能用 `awq` / `gptq`（Triton 反量化），性能特征不同。
3. **SGLang 支持在线量化**（加载时量化，无需预量化 checkpoint），例如 `--quantization fp8`；但官方文档明确建议：**离线量化优先，且已量化的模型不要再叠加 `--quantization`**。

### 4.1 GEMM 后端可显式指定

仅 blockwise FP8 与 NVFP4 支持后端选择：

```bash
--fp8-gemm-backend {auto, deep_gemm, ...}     # SM90 / SM100 上 deep_gemm 为 JIT 编译
--fp4-gemm-backend {auto, ...}
```

---

## 5. 工具链：llm-compressor 是唯一的官方通道

这一节回答「算法怎么变成 checkpoint」。

**llm-compressor**（Red Hat AI + vLLM 项目维护）输出 `compressed-tensors` 格式，vLLM 直接加载。它的算法清单本身就是一份「什么能用」的白名单：

| 类别 | 支持项 |
|---|---|
| **算法** | 简单 PTQ（RTN）、GPTQ、AWQ、SmoothQuant、AutoRound、**旋转类（SpinQuant、QuIP）**、REAP 专家剪枝 |
| **激活量化 scheme** | W8A8（INT8 / FP8）、W4AFP8、Microscale（NVFP4、MXFP4、MXFP8） |
| **混合精度** | W4A16、W8A16、MXFP8A16、MXFP4A16、NVFP4A16 |
| **低/任意位宽** | WNA4 / WNA8 / WNA16；v0.13.0 起 Humming 支持 3/5/6/7 bit 密集打包 |
| **KV / Attention** | FP8、NVFP4 |
| **硬件门槛** | W4A16/W8A16 与 W8A8-INT8：SM 7.5+；W8A8-FP8：SM 8.9+；MXFP8 / NVFP4 / MXFP4：**SM 10.0（Blackwell）**；W4AFP8：SM 9.0+ |
| **已砍掉** | 2:4 稀疏（无硬件支持） |

**三个必须注意的点**：

1. **白名单里没有 AQLM 和 VPTQ**——码本/向量量化路线在官方工具链里缺席，与 §3.4 的移除是同一件事的两面。
2. **旋转类（SpinQuant / QuIP）在白名单里**——所以 [12 篇 QuaRot/SpinQuant](/2026/08/24/ptq-07-quarot-spinquant/)这条线是**有部署出路的**；但请同时记住 [E2 篇 §5.4](/2026/09/19/llm-quant-E2-history-convergence-map/)的警告：旋转类在 **MXFP4** 上反而劣于 RTN，所以「有出路」不等于「任何格式下都该用」。
3. **v0.12.0 起支持多 modifier 叠加**（AWQ + GPTQ 一次通过），这对应 [23 篇统一视角](/2026/08/29/llm-quant-23-unified-view/)的核心结论——**这些算法本来就是可叠加的，不是互斥选项**。工具链终于跟上了这个认识。

---

## 6. QAT 模型的落地路径

这是 [Part 4 QAT](/quantization-roadmap/) 与部署栈的接口处，也是很多人卡住的地方。

```text
训练侧                          序列化                        部署侧
TorchAO QAT  ──►  Int4WeightOnlyConfig(group_size=32)  ──►  HF checkpoint
（Unsloth / Axolotl）              │                            │
                                   │                     TorchAoConfig
                                   └────────────────────►  vLLM（quantization=torchao）
                                                          或 compressed-tensors
```

**现状（截至 2026-09）**：

* **TorchAO 已被 vLLM 列为一等支持的量化格式之一**，且有官方 INT4/HQQ 预量化模型可直接 `vllm serve`（例如官方示例中的 `pytorch/Phi-4-mini-instruct-int4wo-hqq`）。
* **QAT 产出的 checkpoint 与 PTQ 走的是同一条部署路径**——框架不区分你这个 INT4 是训出来的还是压出来的，它只认格式与 config。这一点很重要：**QAT 的部署成本为 0，成本全在训练侧。**
* **原生 INT4 QAT 的大模型已经开始出现**（例如 Kimi K2 Thinking 走 compressed-tensors 加载），说明「训练时就决定低比特」这条路线正在进入生产。
* **NVFP4 QAT 目前是 prototype**：TorchAO 侧与部署侧都还没到「开箱即用」，属于 T1/T2 之间。

**一句话**：如果你做了 QAT，只要输出的是 `Int4WeightOnlyConfig` 或 `compressed-tensors` 能表达的格式，**部署侧不需要额外工作**。真正需要额外工作的是**自定义量化函数**（如 SEQ、非均匀网格、自定义码本）——那些必须自己写 `QuantizationConfig`，甚至要写 kernel。

---

## 7. 到底哪些开箱即用，哪些必须调优

### 7.1 开箱即用（Tier 0）

| 场景 | 做法 | 预期 |
|---|---|---|
| Ada/Hopper/Blackwell 上要吞吐 | FP8 W8A8（llm-compressor `FP8_BLOCK` 或官方 FP8 checkpoint） | 精度损失极小，**这是首选** |
| Ampere 上要省显存 | GPTQ / AWQ W4A16（Marlin） | 成熟稳定，生态最厚 |
| 本地/CPU/Mac | GGUF（Q4_K_M 一档） | llama.cpp 生态 |
| 长上下文 | KV cache FP8（Hopper+） | 显存减半，与权重量化独立可叠加 |
| Blackwell 上要极限压缩 | NVFP4 / MXFP4（需 SM100） | 需验证精度，格式对齐算法才有效 |

### 7.2 必须调优（Tier 1/2）

| 场景 | 必须调什么 | 不调会怎样 |
|---|---|---|
| **MXFP4 / NVFP4** | 选**格式对齐**的算法（FlatQuant、DuQuant++、MR-GPTQ）；必要时 pre-scale | 直接套 QuaRot/SpinQuant 会**劣于 RTN**（[E2 §5.4](/2026/09/19/llm-quant-E2-history-convergence-map/)） |
| **W4A8 / W4AFP8** | 选定具体 scheme（FP8 激活 vs INT8 激活）并确认架构 ≥ Hopper | GPU 上 INT8 激活版本根本不支持 |
| **MoE** | 逐专家敏感度 / 运行时热度分配；确认 fused MoE 后端支持该格式 | 静态统一比特会明显掉点 |
| **混合精度** | 硬件约束（同 block 同 bit、禁非 2 幂位宽）；profile 延迟而不只看显存 | 10–30% 打包开销可能吃掉收益（[25 篇 §5.1](/2026/09/19/llm-quant-25-mixed-precision/)） |
| **≤3-bit** | 必须 QAT，且要自建或改造 kernel | 无通用 kernel，加载后走回退路径，比不量化还慢 |
| **推理/代码类模型低比特** | 混合域校准 + 蒸馏（[22 篇](/2026/09/19/llm-quant-22-reasoning-llm-lowbit/)） | 常识题没事，数学/代码崩掉 |
| **AMD ROCm** | 确认走的是 Aiter/Triton 路径；`SGLANG_USE_AITER=1` | 用错方法名（如 awq_marlin）直接报错 |

### 7.3 必须写代码的（Tier 3）

* 自定义量化函数（SEQ、非均匀网格、自定义码本）
* 论文 prototype（多数 QAT 新算法）
* 已被移除的（AQLM）或未入主干的（VPTQ）

vLLM 提供了 out-of-tree 注册机制（`@register_quantization_config`），SGLang 也是注册式扩展——**框架门口是开的，但进去之后 kernel 要自己写**。

---

## 8. 决策表：给定硬件选什么

| 你的卡 | 首选 | 次选 | 别碰 |
|---|---|---|---|
| A100 / A800（Ampere, SM80） | GPTQ/AWQ W4A16（Marlin） | INT8 W8A8 | **FP8**（不支持） |
| L40S / 4090（Ada, SM89） | **FP8 W8A8** | W4A16 | NVFP4/MXFP4（需 SM100） |
| H100 / H800（Hopper, SM90） | **FP8 W8A8**（CUTLASS/DeepGEMM） | **W4AFP8**（Machete） | MXFP4 的最优后端不在此代 |
| B200 / GB200（Blackwell, SM100） | **NVFP4 / MXFP4** | FP8 | 直接套旋转类而不做格式对齐 |
| MI300X（AMD） | FP8（Aiter）、AWQ/GPTQ（Triton） | petit_nvfp4（AMD 的 NVFP4 路径） | awq_marlin / modelopt_* |
| CPU / 边缘 | GGUF | INT8 W8A8（x86/Arm） | 任何需要 CUDA kernel 的 |

---

## 9. 2026 年部署侧的新变化

1. **NVFP4 / MXFP4 成为新战场**：vLLM 0.10.1 起为 Marlin 与 FlashInfer 后端加入 MXFP4/NVFP4 支持；Blackwell 是主要目标。
2. **KV cache 走向 2-bit**：vLLM 0.20.x 引入 TurboQuant 2-bit KV cache（官方宣称 KV 容量 4×），长上下文并发能力大幅上升。
3. **在线量化（online quantization）进入前端**：serve 时直接量化，不必预先产出 checkpoint——适合快速试验，但**官方仍推荐离线**。
4. **任意位宽不再浪费比特**：llm-compressor v0.13.0 的 Humming 支持 3/5/6/7 bit 密集打包，并新增 W2–W8 × A4/A8/A16 的 16 种预设。这会让 [25 篇混合精度](/2026/09/19/llm-quant-25-mixed-precision/)里「候选 bit 只能是 {2,4,8}」的约束逐渐松动。
5. **2:4 稀疏退出**：llm-compressor 已明确移除。
6. **后端可显式指定**：vLLM 支持 `--linear-backend` 与 per-quant 覆盖（`--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"humming"}}'`），说明后端选择已经从「自动」走向「可运营」。

---

## 10. 上线前检查清单

```text
□ 1. 确认 GPU 计算能力（SM 版本）—— 这决定 FP8/FP4 是否可用
□ 2. 确认算法 → checkpoint 的通道（llm-compressor 白名单内？）
□ 3. 确认框架有对应 QuantizationConfig（不是"能加载"而是"有实现"）
□ 4. 确认 kernel 后端落在你期望的那一条（而不是静默回退到 Triton/BF16）
□ 5. 用你自己的 Golden Set 验证，且**分任务看**（常识 vs 推理/代码）
□ 6. profile 三项：显存、TTFT、吞吐 —— 不能只看显存
□ 7. 长上下文场景单独测 KV cache 精度与容量
□ 8. 记录版本：vLLM / SGLang / llm-compressor / CUDA / 驱动
```

**第 4 条最容易被忽略**：框架常常静默回退。vLLM 会在日志里打印后端选择，SGLang 也有 kernel dispatch 日志——**上线前务必看一眼它到底选了哪个后端**。

---

## 11. 附录 A：自己核对的命令

这类支持矩阵变化极快，动手前请自己再跑一次：

```bash
# vLLM：列出当前版本支持的所有量化方法
python -c "from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS; print(QUANTIZATION_METHODS)"

# vLLM：看它为一个 checkpoint 选了什么配置/后端
vllm serve <model> --quantization <method> 2>&1 | grep -i -E "using|backend|marlin|cutlass|flashinfer|fallback"

# SGLang：看支持的量化方法注册表
python -c "from sglang.srt.layers.quantization import QUANTIZATION_METHODS; print(QUANTIZATION_METHODS)"

# llm-compressor：确认算法与 scheme 是否还在白名单
python -c "import llmcompressor; help(llmcompressor.oneshot)"
```

文档入口：
- vLLM 量化总表：`docs/features/quantization/README.md`（含硬件兼容矩阵）
- SGLang 量化页：`docs/advanced_features/quantization.html`（含 NVIDIA/AMD/Ascend 三列）
- llm-compressor README：`vllm-project/llm-compressor`（算法与 scheme 清单、硬件门槛）

---

## 12. 附录 B：与主线文章的对应关系

| 主线篇目 | 算法 | 本篇给出的部署结论 |
|---|---|---|
| [03 GPTQ](/2026/08/24/ptq-02-gptq/) | GPTQ | T0：Marlin，Ampere+ 全支持，生态最厚 |
| [04 AWQ](/2026/08/24/llm-quant-03-awq-scale-search/) | AWQ | T0：同上；AMD 走 Triton，Intel GPU/CPU 也支持 |
| [09 SmoothQuant](/2026/08/24/llm-quant-02-smoothquant-w8a8/) | SmoothQuant | T0（作为 W8A8 的预处理）：llm-compressor 白名单内 |
| [12 QuaRot / SpinQuant](/2026/08/24/ptq-07-quarot-spinquant/) | 旋转类 | T0/T1：工具链有，**但 MXFP4 上劣于 RTN**，必须看格式 |
| [07 QuIP# / AQLM](/2026/08/24/ptq-05-quip-aqlm/) | 码本/向量量化 | **T3**：AQLM 已被 vLLM 移除；QuIP 在白名单但主要用于预处理 |
| [06 SpQR / HQQ](/2026/08/24/ptq-04-spqr-owq-hqq/) | outlier 拆分 / 免校准 | HQQ 在 vLLM 方法列表内，且有官方 int4wo-hqq 预量化模型 |
| [08 SqueezeLLM / VPTQ](/2026/08/24/ptq-09-squeezellm-vptq-claq/) | VPTQ | **T3**：未入主干 |
| [13 QServe / QQQ](/2026/08/24/ptq-13-qserve-qqq/) | W4A8KV4 | T1/T2：GPU 上 INT8 激活的 W4A8 不支持，W4AFP8 需 Hopper+ |
| [16 GGUF/FP8/MXFP4](/2026/08/24/ptq-08-gguf-fp8-mxfp4/) | 格式层 | FP8 需 Ada+；MXFP4/NVFP4 需 Blackwell |
| [21 LLM-QAT / QLoRA](/2026/08/25/qat-00-overview/) | QAT | T0（部署侧零成本）：格式对了直接可用 |
| [22 Reasoning QAT](/2026/09/19/llm-quant-22-reasoning-llm-lowbit/) | 推理模型低比特 | T2：算法侧有效，但需自建校准/蒸馏流程 |
| [25 混合精度](/2026/09/19/llm-quant-25-mixed-precision/) | Mixed precision | T1：可行但需硬件约束 + profile 延迟 |

---

> **回到系列**：[E2 发展史与收敛地图](/2026/09/19/llm-quant-E2-history-convergence-map/)（算法侧的收敛判断）｜ [23 统一视角](/2026/08/29/llm-quant-23-unified-view/)（算法在改哪个自由度）｜ [课程路线图](/quantization-roadmap/)
