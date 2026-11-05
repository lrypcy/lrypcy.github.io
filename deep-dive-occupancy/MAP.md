# Deep Dive：Occupancy 系列模型 —— 事实底座与写作地图

> v0 · 2026-11-05 · **pilot 产物，不发布**。确认后写单篇长文直进 `_posts/`（不拆篇）。
> 覆盖：隐式占用表征（2019–2022）→ 驾驶 3D 语义占用预测（2022–2026）→ 4D 占用世界模型（2023–2026）。
>
> 核证状态标记：**✅** 已在 `arxiv.org/abs` 亲自核到标题 / v1 日期 / 摘要；**🔎** 子代理核证、URL 可查，写正文前抽核；
> **△** 二手来源或论文原表未读到，引用前必须回原文表格；**✗** 未见，禁止当作事实写。

---

## 0. 范围与三条硬边界

「Occupancy」在这个题目里是**同一个词的三种数学对象**，全文的第一件事就是把它们拆开：

| 段 | 时期 | 状态是什么 | 监督信号 | 典型指标 |
|:---|:---|:---|:---|:---|
| I 隐式占用表征 | 2019–2022 | 连续函数 $o:\mathbb{R}^3\to\lbrace 0,1\rbrace$，网络学概率代理 $f_\theta(p,x)\in[0,1]$ | 水密 mesh 上采样的点标签 | IoU / Chamfer-L1 / F-Score |
| II 3D 语义占用预测 | 2022–2026 | 离散体素网格 $V\in\mathbb{R}^{X\times Y\times Z}$ + 类别分布 | LiDAR 多帧聚合出的稠密**伪真值** | mIoU / RayIoU / FPS |
| III 4D 占用世界模型 | 2023–2026 | 时间轴上的场景状态 $V_{t:t+T}$，或动作条件接口 | 历史 occupancy / 相机 / 轨迹 | future-frame mIoU、FID、L2、闭环 DS/SR |

**不写**：CUDA warp occupancy（同形异义，只在 §1 用一句话点掉）；纯视频 / 纯 latent 世界模型；3D 检测与 BEV 分割；VLA 策略与机器人控制；硬件型号堆料。

**与站内既有系列的硬边界**：

| 系列 | 它讲什么 | 本篇不碰什么 |
|:---|:---|:---|
| `4d-foundation-`（空间智能，01 已发） | 前馈几何重建：点图、相机、运动的 4D 基础模型 | 不重复 DUSt3R/VGGT/D4RT 的四代「消元」叙事、不写查询接口五元组 |
| `wam`（具身智能，全景 + 01 已发） | 未来怎么预测：latent / 视频世界模型、动作联合生成 | 不重复 WAM 全景的谱系与 flow matching 推导；本篇只在 §9–10 交接口 |
| `embodied-ai-*` / `pi-vla-*` | 控制地基、VLA 策略 | 不写策略与本体迁移 |

---

## 1. 一条主线，三条张力

**主线**：连续二值场 → 离散语义体素 → 稀疏 / 高斯 / token 化状态 → 动作条件接口。
不是精度堆叠史，而是「**场景状态该用什么数学对象承载**」的三次换义。

- **监督张力**：3D 点标签（ONets）→ 2D 像素体渲染（NeRF/UNISURF）→ LiDAR 伪真值（SurroundOcc/Occ3D）→ 自监督 / VLM 对齐（GaussTR/AutoOcc）。
- **表征张力**：稠密体素（立方代价）→ TPV / BEV 计算 → 稀疏 query → 3D 语义高斯 → 离散 token。
- **用途张力**：重建输出 → 感知输出 → 可度量 cost function → 与动作联合生成的状态。

---

## 2. 论文谱系（核证状态逐条标注）

### 2.1 段 I：隐式占用表征

| 工作 | 会议/年份 · arXiv | 机制一句话 | 指标与坑 |
|:---|:---|:---|:---|
| Occupancy Networks | CVPR 2019 · [1812.03828](https://arxiv.org/abs/1812.03828) 🔎 | 把表面学成 MLP 的连续决策边界；MISE 八叉树提 mesh | IoU / Chamfer-L1 / Normal C.；**没有 F-Score**；`f_θ(p,x)` 不是 `f_θ(x)` |
| IM-Net | CVPR 2019 · [1812.02822](https://arxiv.org/abs/1812.02822) 🔎 | 隐式 field decoder 替换显式 decoder | MSE/IoU 对 64³ GT；CD/LFD 需 MC 阈值 0.5 |
| DeepSDF | CVPR 2019 · [1901.05103](https://arxiv.org/abs/1901.05103) 🔎 | auto-decoder 回归连续 SDF，零水平集为面 | 与 ONets 是**同期独立**工作，不是继承链；推理需逐形状优化 latent |
| ConvONet | ECCV 2020 · [2003.04618](https://arxiv.org/abs/2003.04618) 🔎 | 卷积编码器 + 隐式解码，引入平移等变 | F-Score 在此进入该谱系（默认 1% 阈值）；ONet 7.7G vs 2D 版 1.6G 显存 |
| IF-Net | **CVPR 2020** · [2003.01456](https://arxiv.org/abs/2003.01456) 🔎 | 空间对齐的多尺度 3D 特征张量 | Chamfer-**L2**，不是 L1；常被误记成 NeurIPS |
| NeRF → UNISURF / NeuS / VolSDF | ECCV 2020 · [2003.08934](https://arxiv.org/abs/2003.08934)；ICCV/NeurIPS 2021 · [2104.10078](https://arxiv.org/abs/2104.10078) / [2106.10689](https://arxiv.org/abs/2106.10689) / [2106.12052](https://arxiv.org/abs/2106.12052) 🔎 | σ 无表面约束 → 用 SDF 零水平集或有界 occupancy 重新塞回体渲染 | **NeRF 之后胜出的是 SDF，不是 occupancy**；这是全文第一个反直觉点 |

### 2.2 段 II：驾驶 3D 语义占用预测（按表征形态，不按时间）

| 形态 | 工作 | arXiv · 年份 | 官方 repo | 自报数（**口径见 §3，禁跨行对表**） |
|:---|:---|:---|:---|:---|
| 单目 SSC | MonoScene | [2112.00726](https://arxiv.org/abs/2112.00726) CVPR'22 🔎 | [astra-vision/MonoScene](https://github.com/astra-vision/MonoScene) | SemanticKITTI mIoU 11.08 |
| 稠密体素 / BEV 提升 | BEVDet-Occ | 无独立论文 ✗ | [HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet) | repo 自报 36.1 / 42.0 mIoU |
| 同上 | FlashOCC | [2311.12058](https://arxiv.org/abs/2311.12058) 🔎 | [Yzichen/FlashOCC](https://github.com/Yzichen/FlashOCC) | 31.95 mIoU / 197.6 FPS（RTX3090 · TRT-FP16）；4D-Stereo Swin-B 43.52 |
| 同上 | FastOcc | [2403.02710](https://arxiv.org/abs/2403.02710) ICRA'24 🔎 | ✗ | 40.75 mIoU；63 ms → TRT 32 ms |
| 三视图 | TPVFormer | [2302.07817](https://arxiv.org/abs/2302.07817) CVPR'23 🔎 | [wzzheng/TPVFormer](https://github.com/wzzheng/TPVFormer) | 27.83 mIoU（第三方复现表，△） |
| 稀疏 query | VoxFormer | [2302.12251](https://arxiv.org/abs/2302.12251) CVPR'23 🔎 | [NVlabs/VoxFormer](https://github.com/NVlabs/VoxFormer) | SemanticKITTI IoU 44.15 / mIoU 13.35；显存 <16 GB |
| 稀疏 query | SparseOcc | [2312.17118](https://arxiv.org/abs/2312.17118) **ECCV'24 ✅** | [MCG-NJU/SparseOcc](https://github.com/MCG-NJU/SparseOcc) | **RayIoU 34.0 @ 17.3 FPS（7 帧）**、15 帧 35.1；提出 RayIoU |
| 稀疏 query | OctreeOcc | [2312.03774](https://arxiv.org/abs/2312.03774) NeurIPS'24 🔎 | [4DVLab/OctreeOcc](https://github.com/4DVLab/OctreeOcc) | 仅核到「算力 −15%~24%」；mIoU 统一口径 ✗ |
| 双通路 Transformer | OccFormer | [2304.05316](https://arxiv.org/abs/2304.05316) ICCV'23 🔎 | [zhangyp15/OccFormer](https://github.com/zhangyp15/OccFormer) | SemanticKITTI 13.50 / 12.37 mIoU |
| 高斯 | GaussianFormer / -2 | [2405.17429](https://arxiv.org/abs/2405.17429) ECCV'24 · [2412.04384](https://arxiv.org/abs/2412.04384) 🔎 | [huang-yh/GaussianFormer](https://github.com/huang-yh/GaussianFormer) | 显存为 SOTA 的 17.8%–24.8%；19.10→20.33 mIoU（SurroundOcc 口径） |
| 高斯自监督 | GaussianOcc | [2408.11447](https://arxiv.org/abs/2408.11447) 🔎 | [GANWANSHUI/GaussianOcc](https://github.com/GANWANSHUI/GaussianOcc) | 训练 2.7× / 渲染 5×；mIoU ✗ |
| 稠密真值流水线 | SurroundOcc | [2303.09551](https://arxiv.org/abs/2303.09551) ICCV'23 🔎 | [weiyithu/SurroundOcc](https://github.com/weiyithu/SurroundOcc) | 自家 GT 上 SC IoU 31.49 / mIoU 20.30（△），迁到 Occ3D 口径被复现为 34–37 |
| 基准 | Occ3D + CTF-Occ | [2304.14365](https://arxiv.org/abs/2304.14365) **NeurIPS'23 ✅** | [Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D) | 基线 6.06（MonoScene）～ 28.53（CTF-Occ）；**CVPR'23 只是 challenge** |
| 基准 | OpenOccupancy | [2303.03991](https://arxiv.org/abs/2303.03991) ICCV'23 🔎 | [JeffWang987/OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy) | AAP 使标注密度约 2×（约 400K 占据体素/帧） |
| 规划接口 | OccNet / Scene as Occupancy | [2306.02851](https://arxiv.org/abs/2306.02851) 🔎 | [OpenDriveLab/OccNet](https://github.com/OpenDriveLab/OccNet) | 3D occ 41.08 mIoU；规划 avg.Col 0.703 |
| 工业界 | Tesla Occupancy Network | 2022 AI Day 演示，**无论文 / 无 arXiv / 无可核 mIoU ✗** | ✗ | 只写「工业界先行演示」，不编指标 |

> 术语纠正：SurroundOcc / Occ3D / OpenOccupancy 做的是「LiDAR 多帧聚合生成稠密**伪真值**提供 dense supervision」，
> **不是 teacher–student 知识蒸馏**。

### 2.3 段 III：4D 占用世界模型（三层分野）

**L1 感知层外推**（历史 → 未来 occupancy；动作至多是联合输出，无动作输入；全部开环）：

| 工作 | arXiv · 状态 | 机制 | 指标口径 |
|:---|:---|:---|:---|
| Cam4DOcc | [2311.17663](https://arxiv.org/abs/2311.17663) CVPR'24 🔎 | 定义 camera-only 4D forecasting 任务 + GMO/GSO + backward centripetal flow | 自成一个评测协议，不是生成模型 |
| OccWorld | [2311.16038](https://arxiv.org/abs/2311.16038) **ECCV'24 ✅** | VQ-VAE 把 200×200×16 压成 50×50 token + GPT 式自回归，**联合解码 ego 轨迹** | 3 s mIoU 17.14 / IoU 26.63、L2 1.17 m（△，来自后续对比表） |
| OccLLaMA | [2409.03272](https://arxiv.org/abs/2409.03272) 🔎 | 稀疏场景 tokenizer + 视觉/语言/动作统一词表 + LLaMA 自回归 | mIoU 19.93 / IoU 29.17（△）；项目页称 ICRA 2026，**未见独立编号** |
| GaussianWorld | [2412.10373](https://arxiv.org/abs/2412.10373) **CVPR'25 ✅** | 场景演化三分解（自车对齐 / 动态局部运动 / 新观测补全），流式增强**当前帧**感知 | 19.85 → 22.13 mIoU（repo）；**不是未来预测器** |
| DFIT-OccWorld | [2412.13772](https://arxiv.org/abs/2412.13772) 🔎 | 去 VQ-VAE：动态体素 warp + 静态位姿变换 + 图像辅助训练 | GT-occ mIoU 22.71 / camera-only 10.50（同篇内可对比） |
| OccTENS | [2509.03887](https://arxiv.org/abs/2509.03887) RA-L 🔎 | temporal next-scale prediction，逐尺度 × 逐帧解耦 | mIoU 22.06 / IoU 31.03；0.21–0.56 s 延迟 |
| SparseWorld | [2510.17482](https://arxiv.org/abs/2510.17482) AAAI'26 🔎 | 稀疏动态 query + 回归式 forecasting | 绝对值 ✗ |
| GAST | [2608.15279](https://arxiv.org/abs/2608.15279) **2026-08-15 ✅** | 连续 BEV 端到端，pose-driven warping 针对离散 token 的几何失真 | 相对值 mIoU +7.67% / IoU +6.44% / 2.84×（绝对值 △） |

**L2 生成可交互世界**（给定轨迹 / 指令 / 布局生成可控未来，但仍非闭环）：

| 工作 | arXiv · 状态 | 条件 | 指标陷阱 |
|:---|:---|:---|:---|
| OccSora | [2405.20337](https://arxiv.org/abs/2405.20337) 🔎 | trajectory prompt + 4D tokenizer（压缩率 512）+ DiT | Table 1 的 37.0/27.4 是**重建**不是预测；生成质量用 FID 8.348，**无 FVD** |
| DOME | [2410.10429](https://arxiv.org/abs/2410.10429) 🔎 | trajectory resampling + 连续 occupancy 上的时空 DiT | 只报相对提升（+36.0% mIoU），绝对值 ✗ |
| DynamicCity | [2410.18084](https://arxiv.org/abs/2410.18084) ICLR'25 🔎 | HexPlane + DiT，支持轨迹/指令/layout/inpainting | v1 题名是 LiDAR Generation，v2 才改 4D Occupancy |
| UniScene | [2412.05435](https://arxiv.org/abs/2412.05435) CVPR'25 🔎 | occupancy 作 meta-representation，分层生成 video + LiDAR | 摘要无数值 ✗ |
| COME | [2506.13260](https://arxiv.org/abs/2506.13260) 🔎 | ControlNet 注入 scene-centric 分支，解耦 ego-motion 与 scene evolution | 只报相对 DOME +26.3%、UniScene +23.7% |
| GenieDrive | [2512.12751](https://arxiv.org/abs/2512.12751) 🔎 | 4D occupancy 作物理先验再生成多视图视频 | mIoU 与 FVD 属两个指标族，禁混排 |
| OccSim | [2603.28887](https://arxiv.org/abs/2603.28887) **2026-03-30 / v2 2026-10-17 ✅** | 单初始帧 + 一串 ego-actions，W-DiT + Layout Generator | >3000 帧 / >9 km / 80× 稳定长度；**v1 写 4+ km，引用须标版本** |

**L3 occupancy 当动作条件接口**（闭环最稀缺）：

| 工作 | arXiv · 状态 | 接口 | 是否闭环 |
|:---|:---|:---|:---|
| Drive-OccWorld | [2408.14197](https://arxiv.org/abs/2408.14197) AAAI'25 🔎 | 四类 action conditions + occupancy-based cost 选轨迹 | **原文自认 open-loop**，常被误写成闭环 |
| SparseWorld-TC | [2511.22039](https://arxiv.org/abs/2511.22039) CVPR'26 Oral 🔎 | 任意未来轨迹条件的稀疏 occupancy；无 VAE / 无 BEV | 开环；数值 ✗ |
| OWMDrive | [2606.30421](https://arxiv.org/abs/2606.30421) 🔎 | 多步 occupancy 作 diffusion planner 条件先验 + GRPO 安全奖励 | NAVSIM 闭环 PDMS 90.8（**△ 二手，回原表核**） |
| OccWAM | [2610.17688](https://arxiv.org/abs/2610.17688) **2026-10-13 ✅** | 三专家 MoT + 两阶段 scene→ego joint flow matching，occupancy 与轨迹联合生成 | **Bench2Drive 闭环** DS 90.45 / SR 77.27%；nuScenes 开环 L2 0.30 m |

> 对照项（**不算 occupancy 状态**）：World4Drive ([2507.00603](https://arxiv.org/abs/2507.00603))、OccVLA ([2509.05578](https://arxiv.org/abs/2509.05578)，occupancy 只是可跳过的训练监督)、
> AD-LiST-JEPA ([2602.12540](https://arxiv.org/abs/2602.12540))、AutoWorld ([2603.28963](https://arxiv.org/abs/2603.28963) **2026-03-30 ✅**)、DriveVLA-W0。**「OccVD」未见**。

### 2.4 2025–2026 其余新方向（标注侧，供 §10 收尾）

| 方向 | 工作 | arXiv | 要点 |
|:---|:---|:---|:---|
| 开放词表 / 自监督 | GaussTR | [2412.13193](https://arxiv.org/abs/2412.13193) CVPR'25 🔎 | 基础模型对齐的稀疏高斯，零样本 11.70 → 12.27 mIoU |
| 开放世界 | AGO | [2504.10117](https://arxiv.org/abs/2504.10117) 🔎 | VLM grounding，闭集自监督 +4.09 mIoU |
| 自监督预训练 | PreWorld | [2502.07309](https://arxiv.org/abs/2502.07309) ICLR'25 🔎 | 体渲染用 2D 标签预训练，统一 3D occ / 4D 预测 / planning |
| LLM 接口 | Occ-LLM | [2502.06419](https://arxiv.org/abs/2502.06419) ICRA'25 🔎 | MS-VAE 动静分离，4D 预测 + 规划 + 场景 QA；绝对值 △ |
| 自动标注 | AutoOcc | [2502.04981](https://arxiv.org/abs/2502.04981) ICCV'25 🔎 | VLM 引导 3DGS 自动开放式标注 |
| 生成式预测 | DiffOcc | [2505.23115](https://arxiv.org/abs/2505.23115) 🔎 | 判别式 → 扩散生成式 |

---

## 3. 数据与评测口径：正文任何数字必须带四元组

**四元组 = 数据集 / 输入设定（GT-occ 或 camera-only、单帧或多帧）/ 时间跨度 / 指标族**。

| 数据集 | 网格 · 体素 | 类别 / 规模 | 真值流水线 | 评估掩码 |
|:---|:---|:---|:---|:---|
| SemanticKITTI SSC | 0.2 m · 256×256×32 | 20 类 | 单帧 LiDAR 体素化 + 可见性补全 | invalid / occluded |
| Occ3D-nuScenes | 0.4 m · **200×200×16** · ±40 m | 17 类 · 700/150/150 场景 | 体素稠密化 → **ray-casting 遮挡推理** → 图像引导细化 | **只用 `mask_camera`** ✅ |
| Occ3D-Waymo | 0.4 m · 200×200×16（另有 0.1 m 版） | 15 类 | 同上 | 另有 `mask_fov`（非 360°） |
| nuScenes-Occupancy（OpenOccupancy） | 0.2 m · 512×512×40 | 17 类 · 28,130/6,019 帧 | 多帧叠加 → 伪标签 → 空白增广 → 约 4000 人工小时净化 | 自家协议 |
| SurroundOcc 稠密 GT | 0.5 m · 200×200×16 | 16 类 | 动静分离多帧聚合 → **Poisson 重建** → 体素化 → NN 传播语义 | 自家协议 |

**RayIoU 为什么出现**（[2312.17118](https://arxiv.org/abs/2312.17118) ✅）：visible mask 训练下表面后无监督，模型把墙「刷厚」可让旧 mIoU 虚高 5–15 点；
且预测墙偏深时 IoU 直接归零、偏近仍有 0.5，深度惩罚不对称。RayIoU 改为沿 query ray 取「类别一致 + 首命中深度误差 < 2 m」的 TP，并允许从过去/未来自车位置投射。

**最该写进正文的一条**：SurroundOcc 在自家 GT 上 20.30 mIoU，同一方法在 Occ3D 口径被复现为 34.40 / 34.6 / 37.1 —— **差 14+ 点不是方法进步，是真值口径差异**。

---

## 4. 十个关键争议点（每条都要在正文给出「现有证据 + 判定」）

1. **ONets ≠ 3D occupancy prediction**：前者是物体级连续隐式曲面，后者是驾驶离散体素语义预测，谱系源自机器人 occupancy grid 与 SSCNet。
2. **NeRF 之后胜出的是 SDF 不是 occupancy**：UNISURF / NeuS / VolSDF 把表面约束塞回体渲染，occupancy 被「渲染可微性」重新定义。
3. **「任意分辨率」只在表征层成立**：提 mesh 仍需 MISE + marching cubes，数字绑定采样点数与 MC 分辨率。
4. **真值才是隐形主角**：三套 GT 流水线、三种分辨率，跨论文 mIoU 不可对表。
5. **mIoU 与 RayIoU 不可比较**：体素 TP/FP/FN vs 沿射线首命中 + 2 m 深度阈值。
6. **BEV 加速 ≠ 任务退化成 BEV**：FlashOcc 是 channel-to-height，输出仍是 3D 体素；TPV 是三平面。
7. **伪真值不是真实 GT**：Occ3D 评估只算相机可见体素，遮挡区不计入。
8. **token 化该不该去**：DOME / DFIT / GAST 一致指 VQ-VAE 是表示瓶颈，但 OccTENS 用多尺度 next-scale 证明离散化仍可兼顾保真与效率 —— **未收敛**。
9. **action-conditioned ≠ closed-loop**：连 AAAI'25 的 Drive-OccWorld 都自认 open-loop；真正闭环仅少数（NavSim / NAVSIM / Bench2Drive）。
10. **GT occupancy 依赖是 occupancy-as-state 的软肋**：同一篇 DFIT 里 GT-occ 22.71 mIoU vs camera-only 10.50；若世界状态必须靠标注 occupancy 才能拿到，规模化叙事不完整 —— 这正是 WAM 主线 latent 派攻击 occupancy 派的地方。

**另有 4 条纯事实陷阱**：Occ3D 是 NeurIPS 2023（CVPR 2023 只是 challenge）；GaussianWorld 是 2412.10373（部分列表误作 2412.04380）；VoxFormer 是 2302.12251（2209.14742 是别的论文）；OccTENS 正文把 OccLLaMA 的 29.17/19.93 误标成 OccWorld（表格里 OccWorld 是 26.63/17.14）。

---

## 5. 十章大纲（单篇长文，目标 500+ 行 / 约 10 个一级章节）

| 章 | 标题 / 要回答的问题 | 内容与关键数字 | 图 / 边界 |
|:---|:---|:---|:---|
| §1 | **三个 occupancy，三种数学对象**：先把词拆开 | 符号表（$o$ / $f_\theta$ / $V$ / $V_{t:t+T}$ / $\hat V$）、三段换义、GPU warp 同形异义一句话 | 图 1 三段谱系时间轴；不写元叙述 |
| §2 | **连续隐式场：为什么 2019 是「撞车」不是继承** | ONets / IM-Net / DeepSDF 同期；BCE vs SDF 回归；MISE 与 marching cubes；ConvONet 等变偏置；NeRF 之后 SDF 回归胜出 | 站内无重复；不展开 NeuS 推导 |
| §3 | **驾驶为什么改用体素：任务定义与真值** | 3D box 的盲区、Tesla 演示（无指标）、四套数据集规格表、三套 GT 流水线与 20.30→37.1 漂移 | 图 2 真值流水线对照（手排 SVG） |
| §4 | **从 2D 到 3D：提升、三平面与双通路** | MonoScene / BEVDet-Occ / TPVFormer / OccFormer / SurroundOcc / CTF-Occ；LSS、channel-to-height、preserve-pooling | 与 `4d-foundation` 边界：不写点图与相机估计 |
| §5 | **稀疏化主线：体素 → query → 高斯** | 立方代价（0.4 m→0.2 m 是 8×；Occ3D 640k vs OpenOcc 10.5M 是 16.4×）、VoxFormer / SparseOcc (top-k 32000 ≈ 5%) / OctreeOcc / GaussianFormer (17.8%–24.8% 显存) | 图 3 表征稀疏化的体素数—显存双轴示意 |
| §6 | **评测为什么会失信：mIoU、RayIoU 与可见性掩码** | RayIoU 公式与动机、虚高 5–15 点、四元组口径、跨论文数值禁对表 | 表宽控制在 3–4 列（站点 610px 内容栏） |
| §7 | **效率账：省掉的到底是什么** | 稀疏卷积 / 2D BEV / 高斯 splatting 各自省什么；FlashOcc 197.6 FPS、SparseOcc 17.3 FPS 的硬件与精度口径；**禁止跨行排序 FPS** | 不写硬件型号堆料，只写口径 |
| §8 | **从 3D 到 4D：token 化与去 token 化** | Cam4DOcc 定义任务 → OccWorld VQ-VAE+GPT → OccSora/DOME/DynamicCity 扩散 → DFIT warp / GAST 连续 / OccTENS next-scale；GT-occ vs camera-only 的 2 倍差 | 图 4 L1/L2/L3 三层分野 + 闭环稀缺标注 |
| §9 | **动作条件与闭环：occupancy 变成接口** | Drive-OccWorld 四类条件 + cost（open-loop）、COME 解耦、SparseWorld-TC 任意轨迹、GenieDrive、OWMDrive、**OccWAM joint flow matching**（DS 90.45 / SR 77.27%） | 与 `wam` 边界：不重复 WAM 全景推导，只交接口 |
| §10 | **未收敛的四个问题与站内衔接** | 去 token 化、坐标系与因果性、GT 依赖与可扩展性、评估失信（mIoU 高 ≠ 驾驶更安全）；occupancy vs latent / 像素 / Gaussian 的四维对照表 | 收尾给 `4d-foundation` 与 `wam` 的站内链接 |

**参考 + 附录**：源文件 permalink（锁 commit 的 GitHub 行号）、术语表、核证状态表（哪些数字是论文自报 / 复现表 / 二手）。

**配图至少 3 张，全部手排 SVG**（`assets/img/occupancy/`）：图 1 三段换义时间轴、图 2 真值流水线、图 3 稀疏化体素数对照、图 4 三层分野。
不用 Mermaid 画架构；字体 PingFang SC；每张出 PNG 肉眼过一遍；外链 SVG 必须写固有 width/height。

---

## 6. 站内衔接与下一轮的站点改动

- **分类**：沿用已有「空间智能」，**不新增分类**（省掉 catalog / categories 三处同步）。
- **图谱**：`_data/knowledge_graph.yml` 新增 `occupancy` 节点（空间智能 / layer 2 / order 13），
  `prefixes: [occupancy]`；边 `4d-foundation -> occupancy`、`occupancy -> wam`；
  学习路线「空间智能与 4D 重建」改为 `[embodied, 4d-foundation, occupancy, wam]`。
  改完必跑 `node tools/kg_layout_check.mjs` 与 `python3 tools/graph_sync.py --links`。
- **命名**：单篇不拆篇，建议 `_posts/2026-11-XX-occupancy-models-deepdive.md`；front matter 的
  `categories: 空间智能`，`tags` 用 `Occupancy / 3D 感知 / 世界模型 / 自动驾驶`。
- **草稿目录**：`deep-dive-occupancy/` 已加进 `_config.yml` 的 `exclude`，只有这一份 MAP，不拆篇、不发布。
- **交付门禁**：六步校验（`check_post` → `scan_garbled` → `check_tldr` → `fix_quotes` → 回填比对 → `check_links`）
  + `check_liquid` + 本地 `jekyll build`（tar 副本 + 删 Gemfile + 删非 ASCII 资源）+ 无头 Chrome 渲染验收，最后 push `master`。

---

## 7. 已核正、下一轮可直接采信的事实

| 事实 | 状态 |
|:---|:---|
| Occ3D 是 **NeurIPS 2023**（comments 明写 Accepted to NeurIPS 2023），CVPR 2023 只是 challenge | ✅ |
| SparseOcc 提出 **RayIoU**，ECCV 2024，摘要自报 34.0 RayIoU / 17.3 FPS（7 帧）、15 帧 35.1 | ✅ |
| OccWorld 摘要写「simultaneously predict ego movement and scene evolution」，**没有动作输入** | ✅ |
| GaussianWorld 是 2412.10373，CVPR 2025；自报比单帧 mIoU 提升 >2% 且无额外计算 | ✅ |
| OccWAM 是 2610.17688（2026-10-13），摘要明确 hierarchical / parallel 两类接口的批评与 joint flow matching | ✅ |
| OccSim 2603.28887 有 v1（3-30）与 v2（10-17）两个版本，>3000 帧 / >9 km / 80× / 67% zero-shot | ✅ |
| AutoWorld 2603.28963（v3 10-26）是 LiDAR occupancy 上训练的 **latent** WM，不是 occupancy 状态 | ✅ |
| GAST 2608.15279 只给相对提升 +7.67% mIoU / +6.44% IoU / 2.84×，绝对值来自二手 | ✅ |
| Tesla Occupancy Network 无公开论文 / 无可核指标 | ✗ 未见 |
| OccVD（作为占用生成论文）不存在 | ✗ 未见 |
| OccLLaMA 项目页称 ICRA 2026、题名扩写，但 arXiv 仅 v1 无 comments | △ 待验证 |
| OccWorld / OccLLaMA / OccTENS 的绝对值来自后续论文对比表，需回原表 | △ 待验证 |
| OWMDrive / Occ-LLM / World4Drive / GAST 的绝对值来自二手解读 | △ 待验证 |
| 所有 arXiv 行号 / GitHub permalink 尚未锁定 commit | 下一轮做 |

---

## 8. 下一轮需要拍板的三件事

1. **是否配随文实验**：倾向**不配**（与 `4d-foundation-01` 一致，纯论文解读）。若要配，只做一个纯 CPU 合成探针：
   构造已知真值体素 → 故意把表面刷厚 / 沿深度偏移 → 反算 mIoU 与 RayIoU，量化「虚高 5–15 点」与深度惩罚不对称，
   落在 `ipynbs` 的 `experiments/occupancy/ray-iou-probe/`，正文给锁 commit permalink。
2. **§7 效率表的口径**：只引同一篇复现表（如 [2503.22087 Table 1](https://arxiv.org/html/2503.22087v1)）还是逐列标硬件？
   建议前者为主 + 少量论文自报值标设定，避免宽表。
3. **第三段要不要压到 2026 年 10 月**：OccWAM（2026-10-13）与 OccSim v2（2026-10-17）距今不到一个月，
   正文须写「preprint，未见录用」，趋势判断全部标「作者判断」。

---

## 附：一手 URL 索引（写正文时优先引这些）

```
段 I   1812.03828 / 1812.02822 / 1901.05103 / 2003.04618 / 2003.01456
       2003.08934 / 2104.10078 / 2106.10689 / 2106.12052
段 II  2112.00726 / 2112.11790 / 2206.10092 / 2302.07817 / 2302.12251
       2303.09551 / 2303.03991 / 2304.14365 / 2304.05316 / 2306.02851
       2311.12058 / 2312.17118 / 2312.03774 / 2403.02710 / 2405.17429
       2412.04384 / 2408.11447 / 2412.13193 / 2504.10117 / 2502.07309
       2502.06419 / 2502.04981 / 2505.23115 / 2503.22087（第三方同表）
段 III 2311.17663 / 2311.16038 / 2405.20337 / 2408.14197 / 2409.03272
       2410.10429 / 2410.18084 / 2412.05435 / 2412.10373 / 2412.13772
       2502.06419 / 2506.13260 / 2507.00603 / 2509.03887 / 2509.05578
       2510.17482 / 2511.22039 / 2512.12751 / 2602.12540 / 2603.28887
       2603.28963 / 2606.30421 / 2608.15279 / 2610.17688
官方仓库（写正文前用 git ls-remote 取 commit 锁死）
       autonomousvision/occupancy_networks · autonomousvision/convolutional_occupancy_networks
       facebookresearch/DeepSDF · jchibane/if-net
       Tsinghua-MARS-Lab/Occ3D · JeffWang987/OpenOccupancy · weiyithu/SurroundOcc
       wzzheng/TPVFormer · NVlabs/VoxFormer · MCG-NJU/SparseOcc · 4DVLab/OctreeOcc
       zhangyp15/OccFormer · Yzichen/FlashOCC · huang-yh/GaussianFormer
       wzzheng/OccWorld · wzzheng/OccSora · zuosc19/GaussianWorld
       haomo-ai/Cam4DOcc · MSunDYY/SparseWorld · synsin0/COME
```
