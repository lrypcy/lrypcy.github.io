---
title: 具身智能日报 · 10/26：三维原生 WAM 与自伴随流策略
date: 2026-10-26
summary: UNITAS 用 1.7B 参数超过 8B 模型，SAFLOW 让流策略可算似然，SteerAct 冻权重提升 21 个百分点。
---

## UNITAS：1.7B 的三维原生世界动作模型，超过 6B 与 8B 对手

**来源**：[青稞社区](https://qingkeai.online/blog/UNITAS) ｜ 论文：[arXiv:2610.12099](http://arxiv.org/abs/2610.12099) ｜ 代码：[DexForce/UNITAS](http://github.com/DexForce/UNITAS)

世界动作模型（WAM）的主流路线是借预训练视频模型联合学习未来画面与动作。UNITAS 换了建模对象：把交互中的运动直接表达为**共享坐标系下的三维点轨迹**——动作流记录夹爪上的点如何运动，场景流描述环境里的点如何随交互变化，两者在同一预测窗口内共享真实物理尺度。

在论文报告的设置下，LIBERO 成功率 99.8%（长程子集 LIBERO-10 为 99.4%），LIBERO-Plus 89.1%，RoboTwin 2.0 双臂 Clean / Randomized 分别为 92.94% / 93.04%，VLABench 56.2%。RoboTwin 2.0 双臂任务上，1.7B 的 UNITAS 超过 6B 的 Fast-WAM 和 8B 的 Motus，参数量约为 Motus 的五分之一。真机整理书本上架与装球拉拉链各测 20 次，完成 18 次和 16 次。

值得注意的一处消融：LIBERO-Plus 上去掉视觉三维位置编码后，相机扰动下的成功率从 79.11% 降至 66.73%。显式几何信息对视角鲁棒性的贡献是可测的。

## SAFLOW：用 Verlet 对称分裂让流策略能算准似然

**来源**：[青稞社区](https://qingkeai.online/blog/SAFLOW) ｜ 论文：[OpenReview](https://openreview.net/forum?id=V4bnZAJAx7#discussion)（NeurIPS 2026，评审 5/5/5/4，6 分制）

生成式策略能表达多峰动作分布，但在线强化学习需要回答「这个动作在当前策略下概率多大」。流匹配策略的似然评估涉及基础噪声逆向恢复与雅可比行列式；常用的显式欧拉法是一阶、非时间对称的离散更新，**反转步长无法严格撤销正向更新**，前后向不一致的偏差会传递到对数似然、策略比率和优化目标。

SAFLOW 的做法是把动作空间扩展为 $$2d$$ 维的 $$(x,y)$$ 对，每步流变换采用 Verlet 风格的 kick-drift-kick 对称组合，使离散映射具备时间对称性与精确可逆性，采样与似然评估沿同一条数值路径往返。在 8 项 Isaac Lab 连续控制任务上取得最高平均回报。

## SteerAct：冻住动作模型权重，24 条轨迹换 21 个百分点

**来源**：[青稞社区](https://qingkeai.online/blog/SteerAct) ｜ 论文：[arXiv:2610.25632](https://arxiv.org/abs/2610.25632) ｜ 代码：[kaichen-z/SteerAct](https://github.com/kaichen-z/SteerAct)

斯坦福李飞飞与 MIT 团队的思路是不微调模型：用跨任务共享的预训练稀疏自编码器（SAE）把隐藏激活分解成稀疏特征，从成功与失败轨迹中选出一个最能区分成败的特征，再沿该特征的解码方向修改隐藏状态。每任务 24 条参考轨迹，部署时动作模型与 SAE 权重均不更新，也不训练额外的强化学习控制器。三项真实机器人任务上平均成功率从 31.1% 提升至 52.2%。

关键点在于选特征时看的是**局部时序模式**而非平均强度：同一特征在成败轨迹里均值可能很接近，但变化顺序不同。系统从激活历史取三个值记录相对大小顺序（六种排序模式），加熵、并列值比例与可用性标记，在两种采样间隔下形成 18 维描述符。

## Long-WAM：能访问历史，不等于会用历史

**来源**：[青稞社区](https://qingkeai.online/blog/Long-WAM) ｜ 论文：[arXiv:2610.10528](https://arxiv.org/abs/2610.10528) ｜ 模型：[Efficient-Large-Model/long-wam](https://huggingface.co/collections/Efficient-Large-Model/long-wam)

NVIDIA、MIT、港大与 UCSD 的工作给出一个反直觉的发现：更长的上下文只有在**视频基础模型以自回归方式预训练**时才带来收益。RoboCasa GR-1 上把上下文从 0 秒增至 19.2 秒，成功率从 63.3% 提升到 78.7%；换成双向预训练的初始化，长历史几乎不带来净收益。

工程侧通过流式观察编码、异步执行与硬件加速部署到 RTX 5090 / DGX Spark / Jetson AGX Thor，RTX 5090 上每个 action chunk（含未来视频 latent 预测）107.4 ms。真机部署在 Unitree G1 与 YAM 上，动态叠杯子 95% 成功率，而 Pi0.5 与 Fast-WAM 在 20 次试验中均未成功。

## 立场论文：机器人的身体也该进化

**来源**：[青稞社区](https://qingkeai.online/blog/Embracing-Evolution) ｜ 论文：[arXiv:2510.03081](https://arxiv.org/abs/2510.03081)（NeurIPS 2026 Position Paper Track）

《Embracing Evolution》主张把「机器人应该长什么样」也变成由任务表现回答的问题。现有流程通常先定硬件再学控制，关节配置、肢体尺寸、执行器能力被视为既定条件，于是任务遇到瓶颈时无法区分是策略没学好还是身体限制了发挥。

论文提出 body-control co-design：内层给定身体学控制，外层根据表现调整身体，形成「设计—学习—评估—更新」循环。文中明确指出不能直接比较两套硬件套同一个控制器——身体变了，适配它的策略也会变，可比的是**身体与适配策略的组合**。作为立场论文，它给出的是研究主张与开放问题，不含实验数字。

---

本期 5 条，来源：青稞社区。
