# Awesome-WAM 榜单数据分析：哪些 WAM 论文效果最好

> 数据抓取日期：2026-09-24。榜单 `last_updated`：**2026-05-16**（此后发布的论文不在榜内，需单独核对）。
> 配套脚本：`tools/wam_leaderboard_analysis.py`

## 1. 数据来源与可信度

Awesome-WAM 页面是 JS 动态渲染的，直接抓 HTML 拿不到论文库。真实数据在三处：

| 数据 | 地址 | 规模 |
| --- | --- | --- |
| 论文库 | `https://raw.githubusercontent.com/OpenMOSS/Awesome-WAM/main/README.md` | 74 篇 WAM（Cascaded + Joint/AR + Joint/Diffusion） |
| 榜单 | `https://openmoss.ai/Awesome-WAM/leaderboard/data/leaderboard.json` | 2229 条结果，18 个 benchmark |
| 论文精读报告 | `https://openmoss.github.io/Awesome-WAM/report/<arxiv_id>/index.en.html` | 每篇一份，含原始基准表 |

匹配方式：arXiv ID 精确匹配 `model_paper` 为主，模型名前缀匹配兜底。74 篇中有 **37 篇**能在榜内找到成绩，其余 37 篇（多为 2026-05 后发布，如 FAWAM、World Pilot、MotionWAM、AHA-WAM、Dream-Tac、Efficient-WAM、MV-WAM、MemoryWAM）榜内无数据。

> 名称匹配存在少量噪声：榜单里 `UVA` 与 `UniVLA` 缩写混用，个别条目会被归错主。判断 top 模型时请以 ID 匹配结果与原论文为准。

**三个必须注意的陷阱**：

1. **协议不可比**。LIBERO 有 50/500 demos 之分；RoboTwin 2.0 有 Protocol A/B 与 8/14/50 任务之分；RoboCasa 有 12/24 任务之分。榜内 `notes` 字段已标注，跨协议排序无意义。
2. **数据预算差几个数量级**。X-WAM 预训练 5874 小时 / 149 万条轨迹；Cosmos Policy 在 RoboCasa 上只用 50 demos/task。绝对分数高不等于方法强。
3. **指标混用**。CALVIN 榜内同时存在 `avg_len`(0–5) 与成功率(%)，直接排序会把 4.64 排到 84.18 后面。

## 2. 分榜结果

### LIBERO（4 套件标准协议，748 条）

| 模型 | 分数 | 榜内排名 | 类别 |
| --- | --- | --- | --- |
| LaST-R1 / AcceRL / CORAL / SimpleVLA-RL | 99.90 / 99.38 / 99.30 / 99.22 | 1–7 | 非 WAM，RL 后训练 |
| **Cosmos Policy + World2Act** | **98.58** | 19 | WAM |
| **DiT4DiT** | **98.55** | 20 | WAM（500 demos） |
| **LingBot-VA** | 98.45 | 27 | WAM |
| MWM / WAV / SayDreamAct | 98.25 / 98.15 / 98.05 | 38 / 42 / 45 | WAM |
| Motus / Fast-WAM / RynnVLA-002 / VLA-JEPA | 97.70 / 97.60 / 97.40 / 97.20 | 73–104 | WAM |
| CoT-VLA / WorldVLA | 83.92 / 81.80 | 549 / 568 | 早期 WAM |

结论：WAM 在 LIBERO 上最强约 98.5，**仍低于 RL 后训练类的 99.2–99.9**。LIBERO 已接近饱和，区分度主要来自后训练范式而非是否建模世界。

### RoboCasa（24 原子任务，160 条）

| 模型 | 分数 | 每任务 demos | 备注 |
| --- | --- | --- | --- |
| SymSkill | 85.0 | — | 只评 12/24 任务，不可比 |
| **X-WAM** | **79.2** | 300 + 5874h 预训练 | 超此前最佳 12.1 点 |
| GR00T-N1.6-ft + World2Act | 72.6 | 50 | 非 WAM 核心 |
| **FLARE** | 70.1 | 300 | |
| **Cosmos Policy** | 67.1 | **50** | 数据效率最优 |
| Video Policy / DUST | 66.0 / 66.3 | 300 / 1000 | |
| **DreamZero** / UWM | 62.4 / 60.8 | — / 1000 | |

### RoboTwin 2.0（176 条，多为 Protocol B / 50 任务）

| 模型 | 分数 (easy/hard) | 排名 | 训练数据 |
| --- | --- | --- | --- |
| **MotuBrain** | **95.94** (95.8/96.1) | 1 | 大规模预训练金字塔 |
| **AIM** | 93.05 (94.0/92.1) | 2 | 30K 轨迹 + GRPO 后训练 |
| **LingBot-VA + Consistency-Consensus** | 93.00 | 3 | |
| LingBot-VA | 92.24 | 5 | |
| Fast-WAM | 91.83 | 8 | 2500 clean + 25000 DR |
| X-WAM | 90.25 (89.8/90.7) | 10 | |
| Motus / GigaWorld-Policy | 87.85 / 86.0 | 16 / 21 | |

这是 WAM 唯一**整体压过非 WAM**的榜（最强非 WAM 为 HoloBrain-0-QW 92.10）。

### 其余基准

- **CALVIN**：UD-VLA 4.64（榜首 Xiaomi-Robotics-0 4.75），VPP 4.33，GR-MG 4.04，GR-1 3.06。老基准，参考价值下降。
- **LIBERO-Plus（7 维扰动鲁棒性）**：VLA-JEPA 79.5（#18/71）最好，榜首 ACoT-VLA 87.5；WorldVLA 仅 25.6 —— **WAM 在分布外扰动上反而是弱项**，值得单独研究。
- **SimplerEnv**：F1-VLA 72.9（#52/363）、villa-X 70.1（#69），WAM 不占优，非主战场。

## 3. 综合推荐

**第一梯队（多榜通吃，优先精读）**

1. **Cosmos Policy**（NVIDIA/Stanford, 2601.16163）—— Joint/Diffusion。LIBERO 98.5、RoboCasa 67.1（**仅 50 demos/task**）、真机 ALOHA 4 任务平均 93.6（vs π0.5 88.6）。最强论据是数据效率与真机长程任务，而非绝对分数。诚实点：ALOHA 的 OOD 均值 89.3 略低于 π0.5 的 92.5。
2. **LingBot-VA**（2601.21998）—— 因果自回归扩散 + MoT 双流 + KV cache 闭环。RoboTwin 92.2 / LIBERO 98.5 / 真机 6 任务 50 demos 适配 progress 79.2%（vs π0.5 65.4%）。开源了 checkpoint，复现友好度最高。
3. **X-WAM**（清华/小米, 2604.26694）—— 首个 4D（RGB-D）统一 WAM。RoboCasa 79.2、RoboTwin 90.3、4D 重建全指标第一（CD 0.0049 vs 两阶段 0.0680）、真机 15Hz 闭环。代价是 5874h 预训练。
4. **MotuBrain**（2604.27792）—— Motus 路线的工程化升级，RoboTwin 2.0 全库第一 95.94，含 50× 推理加速与真机长程家务。无开源代码。

**第二梯队（单项冠军 / 特定预算最优）**

- **UD-VLA**：CALVIN 4.64，接近全库第一。
- **FLARE**：RoboCasa 70.1 @300 demos，中等预算的强基线。
- **AIM**：显式 spatial value map 作为“未来→动作”的接口 + GRPO 后训练，RoboTwin 93.1。思路（而非分数）最值得借鉴。
- **DiT4DiT** / **MWM** / **WAV**：LIBERO 98.5 档，可作为 LIBERO 上的 WAM 基线。
- **Fast-WAM**：RoboTwin 91.8 且主打低延迟，效率方向代表。

**第三梯队（真机泛化与榜单后的新工作）**

- **DreamZero**（NVIDIA, 2602.15922）：仿真分一般（RoboCasa 62.4），但真机**未见任务 39.5% vs 最强 VLA 16.3%**，跨本体 video-only 迁移 10–20 分钟数据带来 +42% 相对提升，38× 加速做到 7Hz 闭环。若关心“泛化”而非“榜单”，这篇是第一优先级。
- **Riemann-1.0**（2608.27033，2026-08）：自报 RoboTwin2.0 94.3 / LIBERO 99.0 / RoboCasa-365 62.6，200K+ 小时交互数据，真机长程 SR 85.0 / PSR 94.4。**目前公开宣称最强的 WAM**，但未经统一协议核验。
- **SG-WAM**（2608.01397，2026-08）：0.9B、无大规模具身预训练，LIBERO 98.5 / LIBERO-Plus 73。效率-性能比突出。
- DobotWAM（越疆，工业界）：LIBERO 99.25，未开放细节。

## 4. 可验证的观察结论

1. WAM 的优势集中在**双臂/长程/接触密集**（RoboTwin、RoboCasa），在单臂短程的 LIBERO 上并不比 RL 后训练强。
2. 世界建模带来的是**数据效率与 OOD 泛化**，不是同分布下的绝对上限——Cosmos Policy 50 demos 打平 300–1000 demos 的对手是最硬的证据。
3. **LIBERO-Plus 是 WAM 的软肋**：扰动鲁棒性上 WAM 集体落后，说明“预测未来”尚未转化为对视觉扰动的稳健性。
4. 显式 3D/4D 几何（X-WAM 的深度分支）与显式空间接口（AIM 的 value map）是 2026 年最有效的两个增益点，两者都指向同一个结论：**纯 RGB 像素预测的表征对动作解码不够**。
