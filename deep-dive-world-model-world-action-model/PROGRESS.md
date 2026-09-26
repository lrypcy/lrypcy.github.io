# Progress Tracking

## 状态

| 项 | 状态 | 说明 |
|:---|:---:|:---|
| `RESEARCH_PLAN.md` | ✅ | v2：术语与形式化、三条谱系、Cascaded vs Joint、双轴评测、12 条开放问题、Q1–Q12、论文索引 |
| `README.md` | ✅ | **v3（2026-09-19）**：新增十方向深潜系列索引、13 个 Lab 结果表、扩充后的论文表与评测四层框架 |
| `PROGRESS.md` | ✅ | 本文件 |
| `_posts/2026-09-05-world-model-world-action-model.md` | ✅ | 单篇长文正文：787 行 / 3.26 万字符，4 个 Lab 实跑并回填 |
| **WAM 十方向深潜系列 `00`–`11`** | ✅ | **本次新增**：12 篇，13 个 Lab 全部实跑 |

## 本轮产出：WAM 十方向深潜系列（2026-09-19）

来源：`congyuan_blogs/Clippings/WAM最新进展分析.md`（ChatGPT 对话归档）。用户要求：先整理方向文档，再对每个方向写含数学原理的深入技术文档。

| 文件 | 行数级 | 覆盖 | Lab |
|:---|:---|:---|:---|
| `00-wam-directions-map.md` | ~190 | 方向地图 + 论文总表 | — |
| `01-math-foundations.md` | ~330 | 数学底座 | A / B / C |
| `02-video-foundation-prior.md` | ~200 | D1 | D1 |
| `03-action-conditioned-wm.md` | ~230 | D2 | D2 |
| `04-joint-world-action.md` | ~230 | D3 | D3 |
| `05-world-model-planning.md` | ~240 | D4 | D4b / D4c |
| `06-neural-simulator.md` | ~250 | D5 | D5 |
| `07-latent-jepa-wam.md` | ~240 | D6 | D6 |
| `08-geometry-action-repr.md` | ~230 | D7 | D7 |
| `09-efficient-inference.md` | ~250 | D8 | D8 |
| `10-driving-wam.md` | ~230 | D9 | D9 |
| `11-training-recipe-wm-rl.md` | ~240 | D10 | D10 |

## 文献核实（本轮新增，全部逐条查证）

Clipping 里的 arXiv 编号全部真实存在，且挖到若干 Clipping 未列出的重要工作：

- **MotuBrain** 2604.27792（v5，2026-07）— UniDiffuser + 三流 MoT；RoboTwin 2.0 95.8/96.1；EWMScore；50–100 轨迹跨本体；>50× / 11 Hz
- **τ₀-WM** 2606.01027（v2，2026-08）— VAM + ACVS 双接口；27,300 h（17.8k 遥操作 65% / 6.5k UMI 24% / 3.0k 人类视频 11%）
- **OSCAR** 2606.04463 — 骨架条件；单 GH200 微调 Cosmos-Predict2.5-2B；PSNR 24.24 / SSIM 0.846 / LPIPS 0.094 / FVD 7.08；Spearman 0.750 / Pearson 0.852
- **DriveWAM** 2605.28544 — Wan2.2-5B → 自回归 video-action policy；VLM chunk 引导；选择性 KV 记忆；4k→100k clips
- **UNIVERSE** 2607.05133 — 单掩码调制 DiT + 模态解耦可见性掩码；91.0 PDMS（vs Two-DiT 89.6）；4.3×
- **SimWAM** 2608.07468（v4）— 隔离注意力掩码；视频分支训练后丢弃；91.5 PDMS + RL
- **WA-JEPA** 2608.20974（v2）— 未来掩码 + 潜空间条件流匹配 + 联合未来-动作预测；EPDMS 91.7；HUGSIM HD-Score 0.4462
- **SG-WAM** 2608.01397（v2）— dynamics token + EMA 自引导 + 几何监督；0.9B；LIBERO 98.5 / LIBERO-Plus 73
- **Faster-WAM** 2608.02365 — DoT（KV-Fusion + RoPE 对齐）；66.5 ms vs 211.7 ms；LIBERO 98.5 / RoboTwin 89.17 / LIBERO-Plus 75.0
- **MM-Future** 2609.20377（2026-09-17，两天前）— 多模态配对假设 + MM-Tokens；94.0 PDMS / 91.5 EPDMS / 32.3 HD-Score
- **WorldGym** 2506.00613（Stanford/NYU/GDM）— Clipping 里引用缺失，已补
- **WALL-WM** 2606.01955、**JOPAT** 2605.23856、**VERA** 2605.27817 — 从 NUS survey 主页挖到，Clipping 未列
- 两篇综述：**2609.16074**（机器人向，2026-09-13）+ **2606.20781**（NUS，109 篇，主页 world-action-models.github.io）

## 踩坑记录（本轮）

1. **可达矩阵列序写反**：`reach()` 里按 `[B, AB, A²B…]` 排列，但轨迹时序要求 `[A^{H-1}B, …, B]`。因为 $u$ 是自由向量，列置换不改变可达集，**所以 Oracle 上界看起来“正常”却拿不到 0 回报**——症状是 oracle 比模型还差。修正后 oracle 立刻回到 −0.000。教训：**先验证上界是否等于理论最优，再信下游数字**。
2. **潜空间塌缩实验没复现**：线性 encoder + 线性 predictor 下不塌缩，而是尺度爆炸（std 1.7，特征值 5.26）。JEPA 塌缩需要非线性 predictor 才有常数解。已放弃该实验，换成“潜空间维度 k 的信息/噪声权衡”。
3. **正则系数 λ 扫描引入选择噪声**：取 max over λ 会让小 k 看起来更好。改用固定任务集 + 明确标注 λ 的选择方式。
4. **目标可达性**：随机采样 goal 时，“什么都不做”（−6.0）会打败所有规划器（−7.4），因为 2 个执行器 10 步到不了 6 维随机目标。改成**从随机动作序列反推可达目标**，oracle 才回到 0。
5. **lstsq 的转置方向**：`lstsq([S,A], Sn)` 得到 $W$ 的形状是 $(d+d_a, d)$，取 $\hat A = W[:d]^\top$ 而不是 $W[:d]$。踩了两次。
6. **动作头占比的基准要说清**：“占 4%”（对 50 步基线）和“砍掉带来 2× 加速”（对 4 步基线）不矛盾，但必须在文中点明，否则读者会以为自相矛盾。

## 剩余待办

- [ ] Cosmos Policy 的“每任务 50 条演示”与“规划 +12.5%”未见于摘要，标为待验证
- [ ] DreamZero 的 62.2% / 39.5% / +42% / 38× 数字仍为二手来源
- [ ] `σ_min` 从 0.002 提到 4 的动机是**我的推断**（论文未解释），已在 11 §2.2 标注
- [ ] Lab D6 的“有限数据下高维模型过拟合”在 toy 上没拿到稳定结论，已标注为定性判断
- [ ] 需要 GPU 的实验（真实 LIBERO/RoboCasa 跑分、跨本体视频迁移）全部未做
- [ ] 十三篇文档尚未决定是否发进 `_posts/`——等用户拍板
