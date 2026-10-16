# VAE 02 素材：从 01 挪出的部分 + 已确认大纲

> 草稿目录，不发布（已加进 `_config.yml` 的 `exclude`）。
> 来源：2026-10-17 从 `_posts/2026-10-13-vae-01-*.md` 的 §8.3 / §8.4 / §10 挪出，等 02 落地后本文件可删。

## A. 从 01 挪出的原文（02 要接住这些内容）

### A1. VQ-VAE 的三项损失（原 01 §8.3）

$$
\begin{aligned}
\mathcal L_{\mathrm{VQ}}={}&
-\log p_\theta(x\mid z_q)\\
&+\lVert\operatorname{sg}[z_e]-e\rVert_2^2\\
&+\beta\lVert z_e-\operatorname{sg}[e]\rVert_2^2
\end{aligned}
$$

其中 $$\operatorname{sg}$$ 是 stop-gradient。三块的分工：重建项负责质量，码本损失把码本向量拉向
Encoder 输出，commitment loss 防止 Encoder 输出远离选中的码本向量。最近邻量化本身不可微，
梯度靠 straight-through estimator 穿过。

### A2. VQGAN 的目标组合（原 01 §8.4）

$$
\mathcal L =
\mathcal L_{\mathrm{recon}}
+\lambda_{\mathrm{perceptual}}\mathcal L_{\mathrm{perceptual}}
+\lambda_{\mathrm{GAN}}\mathcal L_{\mathrm{GAN}}
+\mathcal L_{\mathrm{codebook}}
$$

### A3. 连续 VAE 与 VQ-VAE 的选型表（原 01 §10，整节挪到 02 §9）

| 目标 | 优先考虑 |
|---|---|
| 为图像 / 视频扩散模型提供压缩潜空间 | 连续 VAE |
| 把视觉内容编码成可预测的离散 Token | VQ-VAE / VQGAN |
| 学习可解释的潜在因子 | $$\beta$$-VAE 等变分表示学习方法 |
| 极高分辨率生成 | 压缩率合适的 Autoencoder + 潜空间生成器 |
| 连续时间 / 连续状态建模 | 连续潜变量模型通常更自然，但不是唯一方案 |

这不是绝对边界。连续与离散都可以用于多种生成任务，最终取决于表示质量、生成器架构、
训练稳定性与推理成本。

## B. 已确认的 02 大纲（2026-10-17 用户确认「可以这样写」）

标题：变分自编码器（02）：离散化与码本——从 VQ-VAE 到 VQGAN 的量化细节
定位：只讲「连续向量 → 离散索引」这一层。不重讲 ELBO / 重参数化（01 已覆盖），
不重讲扩散训练（DiT / flow 系列）。

| § | 章节 | 要回答的问题 |
| --- | --- | --- |
| 0 | 符号约定与本篇边界 | 与 01 的分工；$$z_e$$ / $$e_k$$ / $$z_q$$ / $$K$$ / $$d$$ 写法统一 |
| 1 | 连续潜空间过不去的那一道坎 | 先验从 $$\mathcal N(0,I)$$ 变分类分布；$$q(z\mid x)$$ 退化 one-hot → KL 变常数 $$\log K$$；必须另学自回归先验 |
| 2 | 量化算子与它的梯度谎言 | $$\arg\min$$ 不可微；STE 的梯度是「假装 $$z_q$$ 随 $$z_e$$ 变」；**为什么必须有 commitment loss** |
| 3 | 三项损失的梯度流图 | 重建 / codebook / commitment 的 $$\operatorname{sg}$$ 位置为什么不能挪；$$\beta=0.25$$ 出处（接 A1） |
| 4 | 码本怎么更新：梯度 vs EMA | codebook loss 只拉动被选中项；EMA（计数 $$N_k$$ + Laplace 平滑 $$\epsilon$$）为什么更稳 |
| 5 | 码本坍塌与四种解药 | 富者愈富；困惑度 / 使用率 / 唯一索引数；reset、k-means 初始化、L2 归一化余弦最近邻、Gumbel-Softmax |
| 6 | LFQ：把码本直接删掉 | 二值化到 $$\{-1,1\}^d$$，隐式 $$K=2^d$$；熵惩罚逼利用率；为何更契合 Transformer 先验 |
| 7 | FSQ：标量量化也不坍塌 | 逐维量化到 $$L$$ 水平、$$K=L^d$$；码本无参数 → 梯度天然直通，不需要 commitment loss；bound 函数防什么 |
| 8 | VQGAN 补的三件事 | LPIPS、PatchGAN、自适应 $$\lambda_{\mathrm{GAN}}$$（梯度范数比值）；codebook reset 实践（接 A2） |
| 9 | 离散还是连续：判据表 | 接 A3，并补 $$d$$、$$K$$、$$\beta$$、EMA decay、reset 阈值取值区间与监控清单 |
| 10 | 小结 + 03 预告 | 03 候选：压缩率与「重建–生成」的权衡（01 §11 点名的坑） |

## C. 配套实验清单（numpy → `ipynbs` 的 `experiments/vae/vq_quant/`）

1. STE 梯度校验：有限差分验证 $$\partial\mathcal L/\partial z_e=\partial\mathcal L/\partial z_q$$；量化误差随 $$K,d$$ 缩放（理论 $$\sim K^{-1/d}$$）
2. 去掉 commitment 会发生什么：实测 $$\lVert z_e\rVert$$ 漂移、最近邻距离变大、量化误差爆炸
3. EMA vs 梯度更新：$$K=512$$、高斯混合 encoder 输出，2000 step，比困惑度 / 使用率 / 重建 MSE
4. 坍塌复现与解药对照：初始化故意聚簇 → 使用率掉到 x%，再分别上 reset / L2 归一 / Gumbel，看各恢复多少
5. LFQ：sign 量化 + 熵惩罚，扫系数看 $$2^d$$ 里真正用到的比例
6. FSQ：扫 levels 组合对 $$K$$，利用率与 VQ 对比；不加 bound 时多少维跑出界外

## D. 动笔前必须锁版本核实的三条（不靠记忆填）

1. VQGAN 的 codebook reset 阈值与自适应 $$\lambda_{\mathrm{GAN}}$$ 的原文式子
2. LFQ（MagViT-v2，arXiv:2310.05737）的维度、$$K=2^d$$ 的数字
3. FSQ（arXiv:2309.15505）的 levels 配置与实验数字

LPIPS / FID 类指标不在本地实验里：§8 的感知与对抗部分只能引论文数字并标「引论文，未实测」。
