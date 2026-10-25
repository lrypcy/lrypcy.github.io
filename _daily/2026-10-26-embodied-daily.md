---
title: 具身智能日报 · 10/26：三维原生 WAM、自伴随流策略与「别预测像素」
date: 2026-10-26
summary: UNITAS 1.7B 超过 8B 对手；DRILL 指出像素 MSE 更低反而可能损伤策略梯度；FRED 把人机迁移成功率翻倍。
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

## DRILL：世界模型的像素误差更低，策略却可能更差

**来源**：[arXiv:2610.30814](http://arxiv.org/abs/2610.30814)

用世界模型减少环境交互、对 VLA 策略做强化学习后训练，是当下的常见做法。但世界模型通常按「一步预测像素」训练，而它在这一流程里的实际作用是**产生能改进策略的学习信号**，不是渲染逼真的视频。

论文指出这个错配相当实质：**更低的像素 MSE 可能产生更差的策略梯度**；一个按策略效用训练的世界模型，在下一帧像素 MSE 高出 17% 的情况下，仍能训出更好的下游策略。DRILL 用双层优化更新世界模型，目标是最大化内层 GRPO 更新后的策略回报。DRILL-VWI 是闭式 surrogate，按策略访问频率、GRPO group-relative advantage 和模型置信度重加权预测误差；DRILL-IMG 则走完整的 meta-gradient。在五个操作仿真器和两个 VLA 骨干上，DRILL-IMG 相比 WMPO 提高平均成功率 13.1 个点，相比 RLVR-World 提高 7.9 个点。

## FRED：自蒸馏提升未来视频表示，人机迁移成功率 19.4% → 50.0%

**来源**：[HuggingFace Daily Papers](https://huggingface.co/papers/2610.30320)（⭐37） ｜ 论文：[arXiv:2610.30320](https://arxiv.org/abs/2610.30320) ｜ 代码：[suhyeok-jang/FRED](https://github.com/suhyeok-jang/FRED)

一类 WAM 为省掉扩散模型的迭代去噪，改为把单独的动作解码器条件化在**单步去噪得到的中间未来视频表示**上。这个设计让动作预测直接取决于该表示的质量。FRED 用自蒸馏从噪声预测未来表示，以真实未来观测提取的表示作目标，并额外把历史观测条件化进去提供时序上下文。

论文报告在长程与分布外任务上优势尤其明显：在机器人训练数据中不存在、仅由人类视频演示的任务上达到 50.0% 成功率，对比最好的 competing WAM 为 19.4%。

## AtomicRL：用「原子子任务」边界阻断信用污染

**来源**：[arXiv:2610.30097](http://arxiv.org/abs/2610.30097)

VLA 策略做长程操作时，轨迹级稀疏奖励会带来延迟且有偏的信用分配，而各子任务掌握程度不均又会形成瓶颈。AtomicRL 的核心是 Semantic-Aware Policy Optimization：用语义感知门控**阻止失败子任务的 TD residual 跨任务边界传播**，在缓解信用污染的同时保留子任务内部的信用分配与价值引导。配套的 BATE 按各原子子任务的掌握程度动态调整任务采样与局部探索激励。

LIBERO-LONG 上相比 PPO 取得 3.6% 与 4.0% 的绝对提升（两个 VLA 架构）。论文另附 Real2Sim2Real 流程连接仿真在线 RL 与真机部署。

## DiffDreamer：可微仿真只管本体，世界模型只管外界

**来源**：[arXiv:2610.31617](http://arxiv.org/abs/2610.31617)

基于世界模型的强化学习常把环境演化与机器人自身动力学纠缠在同一次 latent rollout 里，模型偏差会影响策略优化效率。DiffDreamer 的做法是分头传播：在同一步优化中，**学习到的世界模型传播捕捉外部动力学与历史依赖特征的 latent 状态，可微仿真器传播机器人显式的物理状态**。actor 目标也拆成两条通路——critic 驱动处理不可微的任务奖励，仿真器驱动处理可微的运动奖励。

论文在激进跟踪与动态穿门竞速上报告训练收敛优于 model-free、model-based 与可微 RL 基线，最终表现也优于这些基线及经典规划控制方法，并给出真机飞行的概念验证。

## NVIDIA Cosmos：开放世界模型的训练与评估工具链

**来源**：[NVIDIA Blog](https://blogs.nvidia.com/blog/cosmos-open-train-evaluate-deploy-world-models/)

NVIDIA 扩展了 Cosmos 平台，新增训练工具、仿真 checkpoint 与 Cosmos Data Factory blueprints，用于构建帮助物理 AI 系统理解并仿真真实世界的世界模型。官方称 Cosmos 在六个不同的开放模型排行榜中位列前五。

## Kyutai CADRE：把世界模型的响应延迟压到 230 ms 以下

**来源**：[MarkTechPost](https://www.marktechpost.com/2026/10/23/kyutai-cadre-video-autoencoders-interactive-world-models/)

Kyutai 发布 CADRE——一组 7 个开放的**因果扩散视频自编码器**，面向交互式世界模型。官方数据：基于 CADRE 构建的世界模型在单张 H100 上以 20 fps 运行，反应延迟低于 230 ms。交互式世界模型的瓶颈一直是延迟而非画质，这条路线把自编码器单独作为优化对象。

## 立场论文：机器人的身体也该进化

**来源**：[青稞社区](https://qingkeai.online/blog/Embracing-Evolution) ｜ 论文：[arXiv:2510.03081](https://arxiv.org/abs/2510.03081)（NeurIPS 2026 Position Paper Track）

《Embracing Evolution》主张把「机器人应该长什么样」也变成由任务表现回答的问题。现有流程通常先定硬件再学控制，关节配置、肢体尺寸、执行器能力被视为既定条件，于是任务遇到瓶颈时无法区分是策略没学好还是身体限制了发挥。

论文提出 body-control co-design：内层给定身体学控制，外层根据表现调整身体，形成「设计—学习—评估—更新」循环。文中明确指出不能直接比较两套硬件套同一个控制器——身体变了，适配它的策略也会变，可比的是**身体与适配策略的组合**。作为立场论文，它给出的是研究主张与开放问题，不含实验数字。

---

本期 10 条，来源：青稞社区 / HuggingFace Daily Papers / arXiv / NVIDIA Blog / MarkTechPost。
