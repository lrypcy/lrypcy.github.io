# Deep Dive：LLM 位置编码——阅读指南

研究成果为发布在 `_posts/` 的 8 篇系列博客（2026-08-29，前缀 `posem-`）。

## 学习路径

```
00 总览（分类学 + 外推主线）
├── 01 正余弦 / 可学习 PE      —— 史前·绝对派
├── 02 相对 PE（XL / T5）      —— 史前·相对派
├── 03 RoPE 详解               —— 主角（必读，系列推导核心）
│    ├── 04 外推上：PI / NTK   —— 频率谱诊断
│    ├── 05 外推下：YaRN / LongRoPE + 生产配置
│    └── 07 M-RoPE 多模态      —— 时空坐标推广
└── 06 ALiBi                   —— 与 RoPE 对照的另一条路（独立可读）
```

- 工程师最短路径：03 → 05（覆盖 RoPE 与全部生产 rope_scaling 配置）
- 研究视角：顺序通读；02 → 03 的"从改公式到不改公式"是最重要的思想跳跃

## 各篇一句话

| 篇 | 文件 | 一句话 |
|---|---|---|
| 00 | posem-00-overview | attention 是集合算子；三分类 + 外推主线 |
| 01 | posem-01-sinusoidal-learned | Sinusoidal 内积只依赖 m−n，是 RoPE 的种子 |
| 02 | posem-02-relative-t5 | 相对派确立了正确观念，输在"改公式" |
| 03 | posem-03-rope | 旋转是"绝对形式实现相对内容"的唯一解 |
| 04 | posem-04-extrapolation-pi-ntk | 外推病逐频率发作；PI 一刀切，NTK 改底数 |
| 05 | posem-05-yarn-longrope | YaRN 逐频 ramp + 注意力温度；LongRoPE 搜索系数 |
| 06 | posem-06-alibi | 线性负偏置天生外推，表达力换的 |
| 07 | posem-07-mrope-multimodal | 位置 = 任意结构化坐标；文本时精确退化 |

## 未尽事项 / 开放问题

- M-RoPE 各轴的外推（YaRN 手术如何按 t/h/w 分别做）尚无系统研究，文中标注为开放问题；
- CoPE（Contextual Position Encoding）等 2024 后新方案仅在 05/07 提及，未单独成篇；
- spaces.ac.cn 部分文章因反爬无法逐篇核验归档号，仅引用了已确认的 8265/8130/8701。
