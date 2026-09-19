# Deep Dive：世界模型（World Model）与世界动作模型（World Action Model）

> 覆盖文献：2018 ~ 2026-09 ｜ 版本：v3（2026-09-19）
> v3 变更：在原有单篇长文之外，新增 **WAM 十方向深潜系列**（`00`–`11`）。

## 一、WAM 十方向深潜系列（本次新增）

**入口：[00 · WAM 发展方向地图](./00-wam-directions-map.md)** —— 定义、判定边界、全景图、10 个方向的核心问题/代表工作/收敛度、已核实论文总表、七层阅读路线。

| 编号 | 文档 | 方向 | 核心结论 |
|:---:|:---|:---|:---|
| — | [00 · 发展方向地图](./00-wam-directions-map.md) | 全部 | 架构骨架已收敛，**表征层（D7）与训练层（D10）没有收敛** |
| — | [01 · 数学基础](./01-math-foundations.md) | 公共底座 | POMDP、两种因子分解、rectified flow、动作对齐、谱半径误差界 |
| D1 | [02 · 视频基座作为世界先验](./02-video-foundation-prior.md) | 视频先验 | 无动作视频只能学到 marginal 动力学；**动作解释 72% 预测误差** |
| D2 | [03 · 动作条件世界模型](./03-action-conditioned-wm.md) | forward dynamics | 反事实可辨识性；**空间对齐表示在数据少时好 3.24×** |
| D3 | [04 · 联合世界-动作建模](./04-joint-world-action.md) | cascade → joint | 多峰任务上 cascade 成功率 **0 vs joint 99.6%**；阈值 $D\approx4\sigma$ |
| D4 | [05 · 世界模型与规划](./05-world-model-planning.md) | test-time 规划 | 模型误差比采样预算更重要；**$N\approx64$ 饱和** |
| D5 | [06 · 神经模拟器与策略评估](./06-neural-simulator.md) | 世界模型当裁判 | 排序对单调畸变免疫；**加 rollout 救不了 per-policy 误差** |
| D6 | [07 · 潜空间与 JEPA](./07-latent-jepa-wam.md) | latent / JEPA | 潜维度收益阶跃、成本平方；**压到控制相关子空间即可** |
| D7 | [08 · 几何感知与动作表征](./08-geometry-action-repr.md) | 跨本体 | 共享接口把 $n=10$ 的误差从 233 压到 0.427，但**有接口损失天花板** |
| D8 | [09 · 效率与实时推理](./09-efficient-inference.md) | 推理优化 | 理想算术 320× vs 现实 50×；**步数蒸馏是最大单项（12.5×）** |
| D9 | [10 · 自动驾驶 WAM](./10-driving-wam.md) | 驾驶 | 单模态碰撞率 **14.9%** vs 多模态 **0%**；代价是通行时间 +50% |
| D10 | [11 · 训练配方与世界模型 RL](./11-training-recipe-wm-rl.md) | 训练 / WM-RL | 模态专属掩码；经典误差界**松 20–325 倍**，实际放大 $(1-\gamma)^{-1}$ |

### 系列内的数值实验（全部实跑，miniconda python / numpy）

| Lab | 位置 | 关键输出 |
|:---|:---|:---|
| A | 01 §6.3 | MSE 只差 10.4%，但 Top-1 命中 **0.963 vs 0.060**（随机 0.062） |
| B | 01 §7.3 | $\rho=0.90\to1.05$ 开环误差 0.146→2.156；**观测刷新把 $\rho$ 影响抹平** |
| C | 01 §4.2 | 直线路径任意 NFE 都是机器精度；余弦路径误差 $\propto 1/\text{NFE}$ |
| D1 | 02 §4.2 | 无动作预测器贴住因果下界（70.23 vs 70.05）；动作额外降 72% |
| D2 | 03 §3.4 | 空间渲染 vs 紧凑向量：200 样本 **3.24×**，8000 样本 1.69× |
| D3 | 04 §1.3 | Cascade Top-1 **0.0** vs Joint **0.996**；阈值 $D\approx4\sigma$ |
| D4 | 05 §5 | $\|A-\hat A\|$ 0.002→0.124 吃掉 0.26–0.38 回报；**$\hat B$ 误差更致命** |
| D5 | 06 §3 | 保守压缩绝对偏差 −0.19，Spearman 仍 **0.970**；$\tau=0.2$ 时加 100× 算力只换 0.013 |
| D6 | 07 §6 | $k<6$ 规划退化；$k\ge6$ 全部最优，**维度代价差 107 倍** |
| D7 | 08 §4.1 | $n=10$：从零学 233 / 共享接口 2.04 / **预训练+残差微调 0.427** |
| D8 | 09 §5 | 瀑布：12.5× → 24× → 89× → 178× → 267× → **320×**（现实 50×） |
| D9 | 10 §4.3 | 单模态碰撞 **14.9%** vs 多模态 **0%**；与 oracle 残余差距 1.58 |
| D10 | 11 §5.2 | $\epsilon_p=0.2$ 造成 ~10% 价值偏差；界松 **20–325×**，实际放大 $(1-\gamma)^{-1}$ |

---

## 二、正文（单篇长文，进 `_posts/`）

**→ [《世界模型与世界动作模型（WAM）：从 Dreamer V4 到 Cosmos Policy / DreamZero 的完整地图（2025–2026）》](/2026/09/05/world-model-world-action-model/)**

单篇长文（787 行 / 3.3 万字符），由 GitHub Pages 自动构建。

## 三、计划与进度

| 文件 | 作用 |
|:---|:---|
| `RESEARCH_PLAN.md` | 研究计划 v2（术语与形式化、三条谱系、Cascaded vs Joint、双轴评测、12 条开放问题、Q1–Q12、未验证清单、论文索引） |
| `PROGRESS.md` | 进度与调研确认的关键事实 |
| `README.md` | 本文件 |

## 四、概念速查

| 术语 | 形式化 | 一句话 |
|:---|:---|:---|
| **VLA** | $p(a\mid o,l)$ | 观测+语言直接映射动作，反应式，不显式表示未来 |
| **World Model** | $p(o'\mid o,a)$ | 预测世界如何演化，本身不是策略 |
| **World Action Model** | $p(o',a\mid o,l)$ | 未来状态合成与可执行动作在一个策略框架内联合学习 |
| **判定边界** | — | 未来预测必须**服务于**三者之一：① 生成动作 ② 给动作打分 ③ 训练动作通路 |
| **Cascaded** | 先 $o'$ 再 $a$ | 模块化清晰，多峰任务上成功率崩为 0（Lab D3） |
| **Joint** | 联合监督 | 自回归 / 扩散式；UniDiffuser 用噪声水平切换五种推理模式 |
| **latent frame injection** | — | 把本体/动作/未来/值编成潜帧塞进视频扩散序列，**不改架构** |
| **shortcut forcing** | — | Dreamer V4 的动力学目标，数据空间去噪 + 步数条件，$K=4$ 前向/帧 |
| **DoT** | — | 视频骨干当表征枢纽，轻量动作头通过对接接口访问所有层 |
| **动作对齐 / 敏感度** | $\mathcal{S}(f)=\mathbb{E}\frac{d(f(o,a_1),f(o,a_2))}{d(a_1,a_2)}$ | WAM 的核心判据；$\varepsilon_{\text{pred}}$ 小推不出它大 |

## 五、三条谱系

1. **Latent 世界模型 / MBRL** — Dreamer V3（Nature 2025，symlog+two-hot+固定超参）、Dreamer V4（2025-09，纯离线挖钻）、IRIS、TD-MPC2
2. **生成式视频世界模型** — Genie 3、NVIDIA Cosmos、驾驶线（Vista/HERMES/GAIA-2）、GE-Sim V2（WorldArena Track1 冠军）
3. **VLA → WAM** — π0/GR00T → **Cosmos Policy**（arXiv:2601.16163）、**DreamZero**（arXiv:2602.15922）

## 六、核心 WAM 论文（已核实）

| 模型 | arXiv | 关键事实 |
|:---|:---|:---|
| **WAM Survey（机器人向）** | [2609.16074](https://arxiv.org/abs/2609.16074) | 2026-09-13；taxonomy = 表征/转移/动作接口/架构/训练/数据/scaling |
| **WAM: A Survey（NUS）** | [2606.20781](https://arxiv.org/abs/2606.20781) | 109 篇；两视图（生成什么 × 解剖学轴）；[主页](http://world-action-models.github.io/) |
| **MotuBrain** | [2604.27792](https://arxiv.org/abs/2604.27792) | UniDiffuser + 三流 MoT，五种推理模式；RoboTwin 2.0 95.8/96.1；>50×，11 Hz |
| **τ₀-WM** | [2606.01027](https://arxiv.org/abs/2606.01027) | VAM + ACVS；27,300 h；propose → rank → rectify |
| **OSCAR** | [2606.04463](https://arxiv.org/abs/2606.04463) | 2D 骨架条件；Spearman 0.750 / Pearson 0.852 |
| **Faster-WAM** | [2608.02365](https://arxiv.org/abs/2608.02365) | DoT：单层动作头 × 30 层骨干；66.5 ms vs 211.7 ms |
| **SG-WAM** | [2608.01397](https://arxiv.org/abs/2608.01397) | EMA 自引导 + 几何监督；0.9B；LIBERO 98.5% / LIBERO-Plus 73% |
| **WA-JEPA** | [2608.20974](https://arxiv.org/abs/2608.20974) | 未来掩码 + 潜空间流匹配 + 联合去噪；EPDMS 91.7 |
| **DriveWAM** | [2605.28544](https://arxiv.org/abs/2605.28544) | Wan2.2-5B → AR policy；4k→100k clips scaling |
| **UNIVERSE** | [2607.05133](https://arxiv.org/abs/2607.05133) | 单掩码调制 DiT；91.0 PDMS，仅轨迹推理 4.3× |
| **SimWAM** | [2608.07468](https://arxiv.org/abs/2608.07468) | 视频只作训练信号；91.5 PDMS + RL |
| **MM-Future** | [2609.20377](https://arxiv.org/abs/2609.20377) | 多模态配对假设；94.0 PDMS / 91.5 EPDMS |
| **WALL-WM** | [2606.01955](https://arxiv.org/abs/2606.01955) | 事件级切分；Camera RoPE + 视锥掩码 |
| **JOPAT** | [2605.23856](https://arxiv.org/abs/2605.23856) | 点追踪 + 可见性通道 |
| **WorldGym** | [2506.00613](https://arxiv.org/abs/2506.00613) | 世界模型当评估环境，保持策略排序 |
| **Cosmos Policy** | [2601.16163](https://arxiv.org/abs/2601.16163) | latent frame injection；LIBERO 98.5% / RoboCasa 67.1% |
| **DreamZero** | [2602.15922](https://arxiv.org/abs/2602.15922) | 14B Wan2.1-I2V，联合 flow matching，38× → 7 Hz |
| **Dreamer V4** | [2509.24527](https://arxiv.org/abs/2509.24527) | shortcut forcing；纯离线挖钻，数据少 100× |

## 七、评测速查

- **四层框架**（01 §9）：L1 视觉保真（PSNR/SSIM/LPIPS/FVD）→ L2 物理常识 → **L3 动作对齐**（敏感度 $\mathcal{S}$、反事实准确率）→ **L4 决策相关**（Top-1 命中、Spearman/Pearson、任务成功率）
- **只报 L1 会系统性高估模型**（Lab A）
- **综合榜**：WorldArena（CVPR 2026，16 指标 + 3 任务；EWMScore 为 16 项归一化均值）
- **策略基准**：LIBERO / LIBERO-Plus、RoboCasa、RoboTwin 2.0、ManiSkill、RLBench、CALVIN；真机 RoboArena / RoboChallenge
- **驾驶**：NAVSIM/navtest（PDMS、EPDMS，开环）、HUGSIM（HD-Score，闭环）、nuScenes、Bench2Drive、PhysicalAI-AV

## 八、相关（本站）

- [RL 信用分配与奖励系数](/2026/09/19/rl-credit-assignment-settlement/) — 想象轨迹中的信用分配、模型误差与 λ-return
- [具身智能 05：VLA](/2026/08/25/embodied-ai-05-vla-foundation-models/) — WAM 的前身
