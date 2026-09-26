# RL 信用分配与奖励系数（2025–2026）

已整合为**单篇博客文章**，不再拆分：

> **→ [RL 信用分配与奖励系数：GRPO 欠条的结算清单（2025–2026）](/2026/09/19/rl-credit-assignment-settlement/)**
>
> 源文件：`_posts/2026-09-19-rl-credit-assignment-settlement.md`

覆盖内容（原 7 篇拆分的合并版）：

| 章节 | 内容 |
|:---|:---|
| §1 定义 | $\Delta_u(q_u;\rho)$ 协议特定的因果对比、信用依赖 $q_u$/$z_u^-$/$\rho$、五级分配单元 |
| §2 为什么被放大 | 六诊断（T/O/R/H/V/C）与 42 篇审计的阳性计数 |
| §3 主线一 | 序列级“隐形信用分配”：每 token 有效权重 $w_{i,t}$ 表、长度/std 偏差、GSPO 序列级 IS、clip-higher/CISPO、IcePop |
| §4 主线二 | 显式细粒度：统一乘性权重框架、token/segment/step 三级、依赖×开销总表 |
| §5 主线三 | 奖励系数：GDPO 塌缩、条件化奖励、长度惩罚定标、KL/熵、尺度、MEDS、hacking 面 |
| §6 主线四 | 长视野与 Agentic：GiGPO → GACA、SR-PPO、GLM-5.2 的 critic 回归、多智能体空白 |
| §7 实践 | 决策树、六项诊断、统一接口、公平比较协议、CA-ID Card、十个坑、系数参考表 |
| §8–9 | 2026 下半年方向、29 篇论文索引、未验证声明清单 |

核心论文（详见文末索引）：[综述 2604.09459](https://arxiv.org/abs/2604.09459)、[GDPO 2601.05242](https://arxiv.org/abs/2601.05242)、[80/20 高熵 token 2506.01939](https://arxiv.org/abs/2506.01939)、[GiGPO 2505.10978](https://arxiv.org/abs/2505.10978)、[GACA 2609.12424](https://arxiv.org/abs/2609.12424)、[SR-PPO 2606.25451](https://arxiv.org/abs/2606.25451)。
