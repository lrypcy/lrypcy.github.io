---
layout: page
title: 模型量化课程路线图
permalink: /quantization-roadmap/
---

<div class="post-block page">
  <h1 class="post-title">模型量化课程路线图</h1>

  > **这是什么**：本站「大模型量化算法」系列的**单一权威目录**。主线共 **26 篇（00–25）**，按
  > **总纲 → 地基 → Weight-only PTQ → Activation PTQ → 格式与部署 → QAT → 横向专题** 六个 Part 组织；
  > 另有 **扩展篇 E1 / E2 / E3 / E4** 不与主线同序。全站所有量化文章的导航栏都以此页为准。
  >
  > **状态**：主线 26 篇**全部已发布**（截至 2026-09）。扩展篇 **E1 / E2 / E3 / E4** 已全部发布。

  ---

  ## 🚀 快速指引

  ### 只读一篇？
  - **想建立系统认知**：读 **[23｜LLM PTQ 统一视角](/2026/08/29/llm-quant-23-unified-view/)** ——把所有算法统一到同一个优化目标下解释
  - **想知道领域现状**：读 **[E2｜发展史与收敛地图](/2026/09/19/llm-quant-E2-history-convergence-map/)** ——2015–2026 年表、收敛度矩阵、选型表
  - **要选方案做部署**：读 **[E3｜部署支持现状](/2026/09/19/llm-quant-E3-deployment-support/)** ——vLLM / SGLang 支持矩阵、硬件门槛、检查清单

  ### 按目标选择路径

  | 你的目标 | 建议路径 | 篇数 |
  |---|---|---|
  | **第一次学量化，要打牢地基** | 00 → 01 → 02 → 03 → 04 → 09 → 15 → 23 | 8 篇 |
  | **做 LLM 推理部署，要会选型** | 00 → 01 → 03 → 04 → 09 → 12 → 15 → 16 → 24 → **E2 §10** → **E3** | 11 篇 |
  | **要上线一个量化模型** | **E3** → 回主线补算法细节 → llm-compressor 实操 | 1 + N |
  | **研究低比特 / QAT，追前沿** | 01 → 17 → 18 → 19 → 20 → 21 → 22 → 25 → **E2** → **E4** | 10 篇 |
  | **技术路线调研，判断成熟度** | **E2** → 按其中标注的未收敛方向深挖 | 1 + N |

  ---

  ## 📚 主线内容

  ### Part 0 · 总纲与地基

  | # | 文章 | 状态 |
  |---|---|---|
  | **00** | [量化全景：设计空间、算法族谱与决策树](/2026/08/24/ptq-00-overview/) | ✅ |
  | **01** | [量化器数学地基与 RTN 基线](/2026/08/23/llm-quant-00-quantizer-fundamentals-rtn/) | ✅ |

  ---

  ### Part 1 · Weight-only PTQ（W4A16）

  把权重压到 4-bit、激活保持 16-bit。这是 LLM 量化最先成熟的路线，也是今天部署量最大的配置。

  | # | 文章 | 状态 |
  |---|---|---|
  | **02** | [LLM.int8()：激活 outlier 与混合精度分解](/2026/08/24/llm-quant-01-llmint8-outlier-mixture/) | ✅ |
  | **03** | [GPTQ：Hessian 二阶权重量化](/2026/08/24/ptq-02-gptq/) | ✅ |
  | **04** | [AWQ：激活感知的权重量化](/2026/08/24/llm-quant-03-awq-scale-search/) | ✅ |
  | **05** | [OmniQuant：可学习缩放与裁剪](/2026/08/24/ptq-03-awq-omniq/) | ✅ |
  | **06** | [SpQR / OWQ / HQQ：异常值拆分与免数据量化](/2026/08/24/ptq-04-spqr-owq-hqq/) | ✅ |
  | **07** | [QuIP# / AQLM：极低位宽的码本路线](/2026/08/24/ptq-05-quip-aqlm/) | ✅ |
  | **08** | [SqueezeLLM / VPTQ / CLAQ：敏感度度量与非均匀量化](/2026/08/24/ptq-09-squeezellm-vptq-claq/) | ✅ |

  ---

  ### Part 2 · Activation PTQ（W8A8 → W4A4）

  激活也要压。难点在于激活只能 per-token / per-group 量化，常驻 outlier 通道会劫持整组 scale。

  | # | 文章 | 状态 |
  |---|---|---|
  | **09** | [SmoothQuant：W8A8 的等效缩放迁移](/2026/08/24/llm-quant-02-smoothquant-w8a8/) | ✅ |
  | **10** | [ZeroQuant 与 W8A8 工程实践](/2026/08/24/ptq-06-smoothquant-zeroquant/) | ✅ |
  | **11** | [Outlier Suppression / OS+：从保护到抑制](/2026/08/24/ptq-10-outlier-suppression/) | ✅ |
  | **12** | [QuaRot / SpinQuant：旋转消除异常值](/2026/08/24/ptq-07-quarot-spinquant/) | ✅ |
  | **13** | [RPTQ / QUIK / ATOM：W4A4 三块基石](/2026/08/24/ptq-11-rptq-quik-atom/) | ✅ |
  | **14** | [OliVe：离群值 - 受害者配对与 abfloat](/2026/08/24/ptq-12-olive-abfloat/) | ✅ |

  ---

  ### Part 3 · 量化格式与部署

  从算法回到硬件。生产环境里跑的往往不是论文算法，而是容器格式与 kernel。

  | # | 文章 | 状态 |
  |---|---|---|
  | **15** | [GGUF k-quants / FP8 / MXFP4：格式与生态](/2026/08/24/ptq-08-gguf-fp8-mxfp4/) | ✅ |
  | **16** | [QServe / QQQ：W4A8KV4 部署甜点](/2026/08/24/ptq-13-qserve-qqq/) | ✅ |

  ---

  ### Part 4 · QAT（量化感知训练）

  前 17 篇的共同假设是「训练已结束，只调权重」。Part 4 打破这个假设。

  | # | 文章 | 状态 |
  |---|---|---|
  | **17** | [伪量化算子插入——QAT 的地基](/2026/08/26/llm-quant-11-fake-quant-insertion/) | ✅ |
  | **18** | [LSQ / PACT / DSQ：可学习的 scale 与 clip](/2026/08/29/llm-quant-18-lsq-pact-dsq/) | ✅ |
  | **19** | [AdaRound / BRECQ / QDrop：PTQ 与 QAT 的过渡带](/2026/09/19/llm-quant-19-adaround-brecq-qdrop/) | ✅ |
  | **20** | [知识蒸馏 + QAT：低比特为什么需要老师](/2026/09/19/llm-quant-20-distillation-qat/) | ✅ |
  | **21** | [LLM-QAT / QLoRA：大模型时代的 QAT](/2026/08/25/qat-00-overview/) | ✅ |
  | **22** | [Reasoning LLM 的低比特量化](/2026/09/19/llm-quant-22-reasoning-llm-lowbit/) | ✅ |

  ---

  ### Part 5 · 横向专题

  不按算法年代排，而按**问题**排。这部分是整套体系的「上层建筑」。

  | # | 文章 | 状态 |
  |---|---|---|
  | **23** | [LLM PTQ 统一视角：所有算法到底在优化什么](/2026/08/29/llm-quant-23-unified-view/) | ✅ |
  | **24** | [KV Cache 量化：KIVI / KVQuant / 长上下文](/2026/08/29/llm-quant-24-kv-cache/) | ✅ |
  | **25** | [Mixed-Precision：自动决定每层比特数](/2026/09/19/llm-quant-25-mixed-precision/) | ✅ |

  ---

  ## 🔌 扩展篇

  不与主线同序，作为特定主题的补充视角。

  | # | 文章 | 说明 |
  |---|---|---|
  | **E1** | [RTN 与 LLM.int8() 工程视角](/2026/08/24/ptq-01-rtn-llmint8/) | 与 01/02 篇主题重叠，侧重工程落地细节与历史脉络 |
  | **E2** | [PTQ / QAT 发展史与收敛地图（2015–2026）](/2026/09/19/llm-quant-E2-history-convergence-map/) | 完整年表、收敛度矩阵、开放问题、实用选型表。**建议主线读完后再读** |
  | **E3** | [部署侧视角：vLLM / SGLang 支持现状](/2026/09/19/llm-quant-E3-deployment-support/) | 四层分级、支持矩阵、算法白名单、QAT 落地路径。**做部署 / 选型时必读** |
  | **E4** | [1-bit / 三元权重量化：BitNet b1.58 与 Bonsai](/2026/09/22/llm-quant-E4-onebit-ternary-bonsai/) | **主线唯一没覆盖的整块**：换的是权重字母表本身（三元 {-1,0,+1} 与二值 {-1,+1}） |

  ---

  ## 🛠️ 维护约定

  1. **新增文章**必须在此页登记编号与所属 Part，并保证该篇文首导航栏指向本页。
  2. **编号一经分配不再变更**——URL 由文件名决定，改名会导致外链 404，故文件名只增不改。
  3. **状态从 🚧 转为 ✅** 时，同步更新该篇文首导航栏与相邻篇的「上一篇 / 下一篇」链接。
  4. 本页是系列唯一权威目录；文章中任何指向外部 `ROADMAP.md` 的引用均视为失效，应改指本页。
</div>
