# Deep Dive 研究计划：《具身智能：从小白到精通》系列

> 主题：具身智能（Embodied AI）知识体系
> 目标读者：从小白到精通者的学习路径
> 输出位置：`_posts/`（Jekyll 博客文章，系列 slug 前缀 `embodied-ai-`）
> 语言：中文（术语保留英文原文）
> 日期约定：2026-08-25 起连续发布

## 一、关键问题清单（本系列必须回答）

1. 具身智能和传统机器人学 / 大语言模型的本质区别是什么？"具身"到底带来了什么？
2. 为什么莫拉维克悖论（Moravec's Paradox）是理解这个领域的钥匙？
3. 小脑级控制（locomotion）为什么走"RL + sim-to-real"路线？数学与工程上如何闭环？
4. 大脑级控制（manipulation/VLA）为什么走"模仿学习 + 大模型"路线？
5. Diffusion Policy、ACT、π0 的动作生成数学本质是什么？自回归 token vs 扩散头 vs 流匹配的取舍？
6. 数据从哪来？遥操作、UMI、仿真合成、世界模型各占什么生态位？数据 scaling law 成立吗？
7. sim-to-real gap 到底卡在哪（物理、渲染还是分布）？域随机化为什么有效又为什么不够？
8. VLA 的架构谱系（RT 系列 → OpenVLA → π0 → GR00T/Helix 双系统）如何演化？瓶颈在哪？
9. 如何评测一个机器人策略？仿真 benchmark 与真机评测的鸿沟怎么填？
10. 一个小白按什么路径学习，能在 3~6 个月内做出第一个可运行的具身智能项目？

## 二、目录结构（最终交付）

```
deep-dive-embodied-ai/
├── RESEARCH_PLAN.md          # 本文件
├── PROGRESS.md               # 进度追踪
└── README.md                 # 阅读指南（Phase 4 产出）

_posts/
├── 2026-08-25-embodied-ai-00-overview.md              # 总览：什么是具身智能
├── 2026-08-25-embodied-ai-01-control-foundations.md   # 数学与控制地基
├── 2026-08-26-embodied-ai-02-rl-il-sim2real.md        # 学习范式：RL/IL/sim-to-real
├── 2026-08-27-embodied-ai-03-perception.md            # 感知：3D 视觉/位姿/触觉
├── 2026-08-28-embodied-ai-04-manipulation.md          # 操作：抓取/灵巧手/BC 家族
├── 2026-08-29-embodied-ai-05-vla-foundation-models.md # VLA 与基础模型
├── 2026-08-30-embodied-ai-06-navigation-planning.md   # 导航与任务规划
├── 2026-08-31-embodied-ai-07-data-sim-benchmark.md    # 数据·仿真·评测
└── 2026-09-01-embodied-ai-08-industry-roadmap.md      # 产业版图与学习路线图
```

## 三、每篇文章的核心覆盖点

### 00 总览篇《什么是具身智能》
- 定义三要素（身体×环境×交互闭环）；图灵 1950 预言、Brooks 无表征智能、莫拉维克悖论
- 发展史时间线：控制论 → Shakey → subsumption → 深度 RL → RT-1 → VLA 元年
- "大脑-小脑-本体"分层架构 Mermaid 图；数据金字塔
- 为什么是现在：LLM 先验 × 并行仿真 × 低成本硬件
- 中外产业版图速览（Figure/Tesla/PI vs 宇树/智元/银河通用…）
- 系列导航图

### 01 数学与控制地基
- 运动学：DH 参数、正逆解、雅可比、微分逆运动学（含 2-link 数值例）
- 动力学：拉格朗日方程推导 $M(q)\ddot q + C\dot q + g = \tau$
- PID/LQR/MPC 三部曲：Riccati 方程推导 + 凸 MPC（MIT Cheetah）
- 阻抗/导纳控制与力控（接触密集任务的命门）
- 全身控制 WBC 任务优先级 QP；人形机器人的质心动力学
- Math-Code 绑定：cart-pole LQR 可运行示例

### 02 学习范式：RL、IL 与 Sim-to-Real
- MDP 回顾（内链到本站《从 MDP 到 GRPO》系列）
- Isaac Gym 大规模并行仿真原理（2108.10470）
- 四足 locomotion 标准范式：teacher-student、地形课程、奖励整形、AMP 风格奖励
- 域随机化数学表述与失效边界；执行器网络 actuator net
- 模仿学习：BC 协变量偏移定理、DAgger、GAIL、离线 RL
- 人形 RL：HumanPlus/OmniH2O/Expressive Whole-Body Control

### 03 感知篇
- 传感器矩阵：RGB-D/LiDAR/IMU/F-T/触觉（GelSight/DIGIT/AnySkin）
- 相机模型与标定；单目深度（Depth Anything）
- 点云深度学习：PointNet/PointNet++ 对称函数与 max-pooling 数学
- NeRF → 3DGS：场景表示革命与 real2sim 用途
- 6-DoF 位姿估计：MegaPose render&compare、FoundationPose
- SLAM 最小知识包：前端/后端/回环，机器人需要什么级别的 SLAM

### 04 操作篇（Manipulation）
- 抓取理论：力旋量、form/force closure、Dex-Net 2.0、GraspNet-1B、6-DoF GraspNet
- BC 家族深潜：协方差偏移 → Diffusion Policy 完整推导（含 shape 表）→ ACT/CVAE 分块
- 数据采集硬件谱系：ALOHA 主从臂、UMI 手持夹爪、DexCap 外骨骼、MimicGen 自动扩增
- 多模态动作分布问题：为什么 MSE 会输出"平均轨迹"
- QT-Opt/TossingBot 的 RL 操作路线及其数据成本

### 05 VLA 与基础模型
- 前史：Gato/PaLM-E/RT-1（Robotics Transformer）
- RT-2 动作 token 化与涌现语义能力；OXE 跨本体归一化
- 架构三大流派对比表：自回归 token（OpenVLA/FAST）vs 扩散头（Octo/RDT）vs 流匹配（π0）
- π0 flow matching 推导 + action expert 设计；π0.5 开放泛化
- 双系统架构：GR00T N1 / Helix S1-S2 / Gemini Robotics
- 世界模型线：DreamerV3 → Genie → Cosmos，用于策略评估与数据生成
- 微调实战：OpenVLA LoRA / SmolVLA / LeRobot 工作流

### 06 导航与任务规划
- 经典栈：占据栅格、A*/RRT*、局部规划器（数学+伪代码）
- 学习式导航：GNM→ViNT→NoMaD 扩散目标掩码统一框架
- LLM 规划器三部曲：SayCan 价值接地、Inner Monologue 反馈闭环、VoxPoser 3D 价值图
- Code as Policies；TAMP（PDDL/PDDLStream）
- 分层架构：LLM 规划层 → 技能库（VLA/原语）→ 控制层的误差复合分析

### 07 数据·仿真·评测
- 仿真器对比大表：MuJoCo/Isaac Lab/SAPIEN-ManiSkill/PyBullet/Genesis/MJX
- 数据集对比大表：OXE/DROID/BridgeV2/AgiBot World/RoboMIND/RH20T
- Benchmark：LIBERO/CALVIN/RLBench/RoboCasa/SimplerEnv 及其饱和问题
- Real2sim 流水线：扫描→资产→场景随机化→数字孪生
- 数据 scaling law（2410.18647 清华 RPT 结论）；数据飞轮设计模式

### 08 产业版图与学习路线图
- 硬件解剖：QDD 执行器/谐波减速器/灵巧手/算力平台（Jetson Thor）
- 公司全景表（美国/中国/欧洲）与技术路线分歧（纯 VLA vs 分层 vs world model）
- 岗位技能矩阵与课程地图（CS285/Underactuated/Modern Robotics/LeRobot）
- 90 天上手动线：SO-ARM100 → Push-T → LIBERO → OpenVLA 微调 → legged_gym
- 十大开放问题

## 四、引用来源策略（已完成验证）

- ✅ arXiv API 批量验证 60+ 论文 ID 与标题（2026-08-25 执行，全部真实存在）
- ✅ 官方网站可达性验证：figure.ai / unitree.com / genesis-embodied-ai.github.io / underactuated.mit.edu / mujoco.readthedocs.io / lilianweng.github.io 等
- ⚠️ github.com / huggingface.co 在当前网络环境不可直连：相关仓库链接为规范地址，标注"未本地验证"
- ⚠️ 中文社区（知乎/CSDN）正文需登录，不直接引用具体帖子 URL；产业新闻数字标注"待核实"

## 五、写作规范（继承博客既有风格）

- front matter：categories [具身智能]，tags 含 embodied-ai/vla/robotics 等，mathjax: true
- 每篇开头 TL;DR 引用块（2~4 条）；系列导航引用块
- 关键公式给完整推导 + 变量映射表（数学符号 ↔ 代码变量 ↔ Shape）
- Mermaid 图遵守 `<br>` 换行规范
- 重要论断句尾附引用链接；Lab Exercises 收尾
