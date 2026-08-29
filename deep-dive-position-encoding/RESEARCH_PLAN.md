# Deep Dive：LLM 位置编码（Positional Encoding）

## 目标
系统研究 Transformer 位置编码，产出 8 篇中文系列博客（发布到 `_posts/`，文件名前缀 `posem-`），参考科学空间（spaces.ac.cn）的推导视角。

## 文档规划

| # | 文件（_posts/2026-08-29-） | 标题 | 核心内容 |
|---|---|---|---|
| 00 | posem-00-overview.md | 位置编码总览 | 排列不变性问题、分类学（绝对/相对/旋转）、外推性主线 |
| 01 | posem-01-sinusoidal-learned.md | 正余弦与可学习位置编码 | Sinusoidal PE 推导与性质、GPT/BERT 可学习 PE、对比 |
| 02 | posem-02-relative-t5.md | 相对位置编码 | Transformer-XL、T5 bias、XLNet、DeBERTa |
| 03 | posem-03-rope.md | RoPE 详解 | 绝对形式实现相对信息、复数推导、旋转矩阵、科学空间视角 |
| 04 | posem-04-extrapolation-pi-ntk.md | 长度外推（上） | 外推性分析、PI、Linear、NTK-aware、Dynamic NTK |
| 05 | posem-05-yarn-longrope.md | 长度外推（下） | YaRN 逐频插值与注意力温度、LongRoPE、HF rope_scaling 实操 |
| 06 | posem-06-alibi.md | ALiBi 与线性偏置 | ALiBi 推导、外推机制、静态/动态偏置、Giraffe |
| 07 | posem-07-mrope-multimodal.md | 多模态 M-RoPE 与多维位置编码 | Qwen2-VL M-RoPE、2D/3D 位置、视觉与视频 |

## 关键问题
1. 为什么 self-attention 天然丢失顺序信息？
2. Sinusoidal PE 为什么用不同频率的三角函数？与可学习 PE 的实验对比？
3. 相对位置编码三代方案（XL/T5/XLNet）各自的取舍？
4. RoPE 如何用绝对位置的形式实现相对位置？复数视角与旋转矩阵视角的等价性？
5. RoPE 为什么外推性差？哪些频率分量是罪魁？
6. PI / NTK-aware / YaRN / LongRoPE 的演进逻辑与数学形式？
7. ALiBi 的线性偏置为什么天然可外推？
8. M-RoPE 如何分解 t/h/w，如何在纯文本时退化为 1D RoPE？

## 主要来源
- 科学空间《Transformer升级之路》系列（spaces.ac.cn/archives/8265 等）
- Attention is All You Need (1706.03762)、RoFormer (2104.09864)、T5 (1910.10683)
- Transformer-XL (1901.02860)、ALiBi (2108.12409)、PI (2306.15595)
- YaRN (2309.00071)、LongRoPE (2402.13753)、Qwen2-VL (2409.12191)
