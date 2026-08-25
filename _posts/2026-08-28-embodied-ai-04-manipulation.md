---
title: "具身智能全景（04）：操作——抓取、灵巧手与 Diffusion Policy 的数学深潜"
date: 2026-08-28 10:00:00 +0800
categories:
  - 具身智能
tags: [embodied-ai, manipulation, diffusion-policy, act, aloha, grasping, imitation-learning]
layout: post
mathjax: true
---

> **系列导航**：走路（第 02 篇）已被 RL 攻克，而"用双手做事"仍是皇冠上的明珠。本篇先讲解析抓取的几何理论，再深潜模仿学习两大明星架构——Diffusion Policy 与 ACT/ALOHA，最后盘点数据采集硬件的"军备竞赛"。本篇是理解第 05 篇 VLA 的必要前置。

## TL;DR

> **TL;DR 1｜多模态之死**：同一任务有多条等价轨迹时，MSE 回归会输出所有模式的**平均值**——向左绕和向右绕平均成"撞柱子直行"。这是 BC 十年停滞的元凶，也是扩散模型进入机器人学的根本理由。

> **TL;DR 2｜Diffusion Policy 一句话**：把动作序列当作图像去噪。训练学 $\nabla\log p(a|o)$，推理从高斯噪声迭代去噪出动作块；它天然表达多模态分布，代价是需要 $N$ 次前向传播（DDIM 蒸馏可缓解）[1](https://arxiv.org/abs/2303.04137)。

> **TL;DR 3｜ALOHA 的贡献一半在硬件**：低延迟主从双臂 + **action chunking**（一次预测未来 $k$ 步）把真机成功率从个位数推到 80%+，证明"数据质量 × 架构细节"比算法炫技更重要[2](https://arxiv.org/abs/2304.13705)。

## 1. 任务分类学与难度金字塔

| 类别 | 例 | 核心难点 | 代表方法 |
|------|-----|---------|---------|
| Pick-and-place | 分拣、码垛 | 视觉定位 | 解析抓取 + 运动规划 |
| 接触密集 | 插销、拧盖、擦拭 | 力控+摩擦不确定性 | 阻抗控制 / 学习方法 |
| 形变物体 | 叠衣服、系鞋带 | 状态不可枚举 | 数据驱动为主 |
| 长时程 | 做一顿饭 | 误差复合+子目标选择 | 分层/VLA |
| 动态 | 抛接、颠勺 | 高频闭环 | RL |

## 2. 解析抓取：三十年几何理论的速成

### 2.1 力闭合与形闭合

对刚体的稳定抓取有经典判据：

- **形闭合（form closure）**：接触点几何上完全约束刚体运动（夹钳全包）；
- **力闭合（force closure）**：存在一组指尖力能抵抗任意扰动力旋量——数学表述为各接触力旋量正张成整个 $\mathbb{R}^6$：

$$
\exists \lambda_i \ge 0,\quad -\begin{bmatrix} f_{ext} \\ \tau_{ext} \end{bmatrix} = \sum_i G_i(\text{法向}\times\text{摩擦锥})\lambda_i
$$

其中 $G_i$ 是接触雅可比映射。库仑摩擦锥 $|f_t| \le \mu f_n$ 让它成为非线性约束（实践中常线性化为内接多面体）。

### 2.2 从几何到学习：Dex-Net 与 GraspNet 家族

手工设计抓取度量在杂乱场景失效后，数据驱动接管：

- **Dex-Net 2.0**（Berkeley, 2017）：在仿真中生成 670 万个平行夹爪候选点+力学分析标签，训练 GQ-CNN 输出"这个抓取点可靠的概率"，实机 93% 成功率（对已知物体集）[3](https://arxiv.org/abs/1703.09312)；
- **GraspNet-1Billion**（上海AI Lab 等）：97,280 张真实 RGB-D 图像 + 十亿级标注抓取位姿的 benchmark，提供端到端检测器与统一评测协议（RA-L 版本论文 [4](https://arxiv.org/abs/1912.13470)，项目页 graspnet.net）；
- **6-DOF GraspNet**：把抓取空间扩展到任意旋转（CVAE 生成式建模），不再限于自顶向下[5](https://arxiv.org/abs/1905.10520)；
- **DexGraspNet**：面向灵巧手（Shadow Hand 多指）的大规模合成抓取数据集，基于力闭合解析检验[6](https://arxiv.org/abs/2210.02697)。

这条线的共同哲学：**把"抓取"压缩成一次前向推理**，给上层策略当原语调用——它至今仍是最可靠的工业方案，VLA 并没有完全取代它（见第 05 篇 GraspVLA 案例）。

## 3. RL 操作的光辉与悲壮

Google 的 QT-Opt 是里程碑也是墓志铭：7 台机械臂 × 数个月真机试错 = 58 万次抓取经验，换来泛化到未见物体的抓取能力[7](https://arxiv.org/abs/1806.10293)；TossingBot 用"残差物理"学会把物体抛进盒子（物理先验 + NN 补偿），但同样烧掉数月真机时间[8](https://arxiv.org/abs/1903.11239)。结论写进了行业集体记忆：

> **真机 RL 的数据成本不可持续** → 操作研究转向模仿学习 → 一切瓶颈归结为"演示从哪来、怎么用好"。

## 4. 行为克隆的救赎：两条技术路线

### 4.1 先看 MSE 为什么必然失败

设专家对"绕开左侧障碍"和"绕开右侧"各演示 50 条，两簇动作均值恰指向障碍物：

$$
a_{MSE} = \arg\min_a \sum_j \|a - a^{(j)}\|^2 = \frac{1}{100}\sum_j a^{(j)} \quad (\text{撞向障碍})
$$

形式化地：MSE 学的是条件均值 $\mathbb{E}[a|o]$，而多模态分布的均值不在任何模式上。早期解法（混合密度网络、离散化动作）各有毛病，直到生成式模型进场。

### 4.2 Diffusion Policy 完整推导

**核心思想**：把动作序列 $a_{0:T_a}$ 视为高维向量，用去噪扩散过程建模其条件分布 $p(a|o)$。

**前向加噪**（训练时人为破坏）：定义噪声调度 $\alpha_t$、$\bar\alpha_t=\prod_{i\le t}\alpha_i$，闭式采样任意时刻的加噪动作：

$$
q(a_t \mid a_0) = \mathcal{N}\!\left(a_t;\ \sqrt{\bar\alpha_t}\, a_0,\ (1-\bar\alpha_t)\, I\right)
$$

**训练目标**：网络 $\epsilon_\theta(a_t, t, o)$ 预测被加入的噪声，损失即加权去噪分数匹配：

$$
L = \mathbb{E}_{a_0\sim p_{data},\ t\sim[1,T],\ \epsilon\sim\mathcal N(0,I)}\left[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}a_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t,\ o\big)\big\|^2\right]
$$

推导要点：DDPM 论文证明该简化目标与变分下界的加权版本等价，权重恰好让网络专注预测分数函数 $\nabla_a \log p(a_t)$——即"往哪个方向去噪能让动作更像人话"。

**推理**：从纯噪声出发，逆过程迭代 $T$ 步（每步一步网络前向）：

$$
a_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(a_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(a_t,t,o)\right) + \sigma_t z,\quad z\sim\mathcal N(0,I)
$$

变量映射表（以 CNN 版本、观测窗口 $T_o{=}2$、动作块 $T_a{=}8$、7-DoF 臂为例）：

| 数学符号 | 代码变量 | Shape | 说明 |
|---:|:---|:---:|:---|
| $o$ | `obs` | `(B, T_o·d_obs)` 或图像张量 | 条件输入 |
| $a_0$ | `action_gt` | `(B, T_a, 7)` | 干净动作块 |
| $a_t$ | `noisy_action` | `(B, T_a, 7)` | 第 $t$ 步加噪动作 |
| $\epsilon_\theta$ | `noise_pred_net` | `(B,T_a,7)→(B,T_a,7)` | U-Net/Transformer |
| $t$ | `timesteps` | `(B,)` | 整型噪声等级 |
| $\epsilon$ | `noise` | `(B, T_a, 7)` | 目标高斯噪声 |

```python
import torch
B, Ta, Da = 32, 8, 7
obs = torch.randn(B, 20)                       # 低维观测示例
noise_pred_net = MyUNet1D(in_dim=Da, cond_dim=20, out_dim=Da)
alphas_bar = torch.linspace(0.999, 0.01, 100)  # 简化调度表 (T,)
t = torch.randint(0, 100, (B,))                # 随机噪声等级
eps = torch.randn(B, Ta, Da)                   # ε ~ N(0, I)
x0 = torch.randn(B, Ta, Da)                    # 专家动作块（示意）
xt = alphas_bar[t].view(B,1,1).sqrt() * x0 \
     + (1 - alphas_bar[t].view(B,1,1)).sqrt() * eps   # q(x_t|x_0) 闭式采样
loss = torch.nn.functional.mse_loss(
    eps, noise_pred_net(xt, t, global_cond=obs))      # L = ‖ε - εθ(x_t,t,o)‖²
```

（代码骨架对应官方 `diffusion_policy` 仓库的训练循环结构，GitHub: `real-stanford/diffusion_policy`，未本地验证）

工程要点：CNN 主干推理快适合高频控制；Transformer 主干吃图像观测更强。DDIM 可把推理步数从 100 压到 10 且几乎不损性能。后续演化：DP3 用点云编码器替换视觉编码器提升泛化；π0 把扩散头换成流匹配并塞进 VLA 底座（第 05 篇）。

### 4.3 ACT/ALOHA：分块与条件 VAE

ALOHA 团队发现精细双臂任务的杀手不是架构而是**误差复合**：单步小错 → 状态偏移 → 后续步步惊心。ACT（Action Chunking Transformer）的两味药：

1. **Chunking**：每个观测一次性输出未来 $k{=}100$ 步（50Hz 下 2 秒）动作，执行期间不看新观测——把"决策频率"降为原来的 $1/k$，复合误差被截断；
2. **CVAE 建模多模态**：训练时编码器 $q(z|s,a)$ 把整段专家轨迹压成风格隐变量 $z$（如"这次从左绕"），解码器以 $(z, s)$ 生成分块；推理时 $z$ 取先验期望或置零：

$$
L_{ACT} = \underbrace{\|a_{pred} - a_{gt}\|^2}_{\text{重建}} + w_{KL}\cdot D_{KL}\big(q(z|s,a)\,\|\,\mathcal N(0,I)\big)
$$

配合 4~8 台低成本主从臂（约 2 万美元）每周可采数百条高质量演示，最终在插电池片、拧瓶盖这类"RL 和解析方法都绝望"的任务上达到 80–96% 真机成功率[2](https://arxiv.org/abs/2304.13705)。Mobile ALOHA 给底盘装上轮子，用同样的遥操作框架采集全身移动操纵数据[9](https://arxiv.org/abs/2401.02117)。ACT 已是 LeRobot 开源库的内置算法之一（GitHub: `huggingface/lerobot`，未本地验证）。

## 5. 数据采集硬件军备竞赛

| 系统 | 形态 | 成本量级 | 特长 | 引用 |
|------|------|---------|------|------|
| ALOHA 主从臂 | 双 JOINT 臂对拷 | ~$20k | 高保真双臂 | [2](https://arxiv.org/abs/2304.13705) |
| UMI 手持夹爪 | 手持 + GoPro + AR 标记 | ~几百美元 | 无需机器人即可采、场景自由 | [10](https://arxiv.org/abs/2402.10329) |
| DexCap 外骨骼 | 手部动捕外骨骼 | 中 | 灵巧手指级数据 | [11](https://arxiv.org/abs/2403.07788) |
| HumanPlus/OmniH2O | RGB 人体姿态→人形影子 | 中高 | 人形全身 | 见第 02 篇 |
| MimicGen | 从少量演示自动扩增千条 | 软件 | 合成放大 50× | [12](https://arxiv.org/abs/2310.17596) |
| VR 遥操（Open TeleVision/Bunny-VisionPro 类） | 头显第一视角 | 中低 | 沉浸感、上手快 | 社区开源项目为主 |

趋势判断：**遥操作正在从"实验室技能"变成"产业基础设施"**（智元等公司已建数百人规模的数据工厂，第 07 篇展开）。UMI 证明了另一条激进路线：只要标定得当，手持设备采集的人类操作可以直接训练真机策略——embodied gap 可以被工程手段填平。

## 6. 评测的正确姿势

报告一个成功率数字远远不够，社区共识要报**泛化轴矩阵**：

- 物体泛化：未见实例 / 未见类别
- 位姿泛化：位置扰动 / 姿态扰动 / 光照变化
- 场景干扰物：有无 distractors
- 统计口径：几次随机种子 × 几次 rollouts（<20 次的成功率没有意义）

这套方法论后来被 LIBERO/CALVIN/SimplerEnv 等基准制度化（第 07 篇详述）。

## Lab Exercises

1. **亲手感受多模态失败**：构造一维玩具任务——状态 $s=0$ 时专家动作 50% 为 $+1$、50% 为 $-1$（双峰）。分别用 MSE 回归和一个小型 diffusion head（或直接输出高斯混合 MDN）拟合，画预测分布图，验证 MSE 收敛到 0（撞墙解）。
2. **跑通 Push-T**：LeRobot 提供 Push-T 任务 + ACT/Diffusion Policy 实现（GitHub: `huggingface/lerobot`，`examples/` 目录，未本地验证），用 50 条脚本生成的演示训 Diffusion Policy，记录成功率随演示数的幂律曲线；再故意只用 5 条对比，体会数据饥饿。
3. **DDIM 加速实验**：在 Diffusion Policy 推理循环中把 100 步采样换成 DDIM 10 步，测量单步决策延迟变化（ms 级），并统计 Push-T 成功率差——理解"采样步数 vs 控制频率"的真实权衡。

## 参考文献与延伸阅读

- [1] Chi et al., *Diffusion Policy*, RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- [2] Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT/ALOHA）. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)；项目页 aloha.mee.toronto.edu
- 扩散模型数学基础：本站后续可参考 Lilian Weng《What are Diffusion Models》[链接](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)。
- 中文解读：知乎检索"Diffusion Policy 解读""ACT ALOHA 源码"，多个高质量专栏逐行分析（正文需登录，站内自行检索）。

*下一篇：《05 VLA》——当 Diffusion Policy 的"手感"装上 GPT 的"大脑"，RT-2 到 π0 再到 GR00T/Helix 的架构演进史，以及流匹配为什么正在取代扩散头。*
