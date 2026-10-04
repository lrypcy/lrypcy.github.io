---
title: "RL 框架里的那些变量到底怎么算：优势、概率比、KL 与损失聚合的源码对照"
date: 2026-09-30 21:30:00 +0800
categories:
  - 强化学习
tags: [rl, grpo, ppo, gspo, cispo, dapo, advantage, importance-sampling, kl-divergence, loss-aggregation, verl, trl, source-code, rlvr]
layout: post
mathjax: true
---

> **系列导航** ｜ 系统侧：[RL 训练框架全景横评（2026）](/2026/09/29/rl-training-frameworks-survey/) ｜ 算法侧：[从 MDP 到 GRPO（五）：GRPO 组相对优势](/2026/08/21/mdp-to-grpo-05-grpo-group-relative/) ｜ [RL 信用分配与奖励系数](/2026/09/19/rl-credit-assignment-settlement/)
>
> 上一篇横评回答「选哪个框架」，这一篇回答「**同一个公式，各框架实现出来为什么不一样**」。

**TL;DR**

> * **同一份 rollout，三个框架能算出三套 advantage**。不是因为谁错了，而是「组内归一化」这件事有四种合法做法：GRPO 除以组内 std、Dr.GRPO 只减均值、RLOO 乘留一系数 $$\frac{n}{n-1}$$、REINFORCE++-baseline 减均值后再做**跨组白化**。源码里它们的差别只有几行。
> * **一个几乎没人提的源码细节**：verl 用 `torch.std()` 算组内标准差（[`core_algos.py:321`](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L321)），而 torch 的 `std` **默认 unbiased（除以 $$n-1$$）**；换成 numpy 默认口径，$$G=4$$ 时 advantage 尺度差 **1.1547 倍**。
> * **GSPO 的「序列级比率」在 veRL 和 TRL 里不是同一个东西**。veRL 用**组合比率** `sg[log s_i] + log_prob − sg[log_prob]`：裁剪决策用序列级标量、梯度仍逐 token 回传；TRL 则把序列级 ratio 直接 broadcast 到所有 token，梯度整条共享。同样叫「序列级」，梯度路径不同。
> * **CISPO 的精髓就一行**：`-sg(clamp(ratio)) * A * log_prob`。梯度走 `log_prob` 而不是 `ratio`，所以**永远有梯度**，裁剪只改权重、不把 token 踢出训练。数值实验里它的非零梯度覆盖率是 6/6，而普通 PPO 裁剪是 4/6。
> * **KL 有两个落点，很多团队混着用**。veRL 把 k1 估计**注入 reward**（`r - (old_logp - ref_logp) * beta`），TRL 把 **k3 估计加在 loss**（`exp(ref−logp) − (ref−logp) − 1`，恒非负）。前者会经过 advantage 归一化，后者不会；**两个口径的 $$\beta$$ 不可比**。
> * **verl 里藏着一个 Schulman 博客里的技巧**：`kl_penalty` 若以 `+` 结尾（如 `k3+`），前向用 k3 的值、反向用 k2 的梯度——`backward - backward.detach() + forward.detach()`，因为「k1/k3 的期望是 KL，但它们的梯度不是 KL 的梯度」。
> * **损失聚合有四种口径，直接决定长回答被放大几倍**。verl 的 `agg_loss` 实现了 `token-mean` / `seq-mean-token-sum` / `seq-mean-token-mean` / `seq-mean-token-sum-norm`；TRL 里对应 `bnpo` / `grpo` / `dr_grpo` / `cispo·dapo` 四个 `loss_type` 分支。同一份数据在四个口径下的 loss 差了近 7 倍。
> * **本文所有梯度结论都用数值微分验过**（`tools/rl_var_grad_lab.py`，纯 numpy）。含 `detach()` 的表达式不能用朴素数值微分验证——**要用双变量技巧** `x_var - x_fixed` 才能正确模拟 stop-gradient，否则测出来的是「假设没写 detach」的梯度。
> * **同一个配置在不同版本上是不同的东西**（这条是拉着最新代码核对后才发现的）。`k3+`（KL 的 straight-through）在 veRL `0.7.0.dev` 上**根本跑不通**——`"k3+"` 匹配不到 `kl_penalty_forward` 里的 `("low_var_kl", "k3")` 分支，会一路抛 `NotImplementedError`，`v0.9.1` 加了后缀剥离才修好。GSPO 的损失聚合也从硬编码 `seq-mean-token-mean` 改成了可配置，**默认值变成 `token-mean`**，源码注释明确写了「偏离 GSPO 原论文」。**引用源码必须连版本一起锁定。**

---

## 目录

- [1. 为什么这些变量值得单独解剖](#1-为什么这些变量值得单独解剖)
- [2. 一次迭代里，变量是怎么流动的](#2-一次迭代里变量是怎么流动的)
- [3. 变量一：advantage](#3-变量一advantage)
- [4. 变量二：概率比 ratio](#4-变量二概率比-ratio)
- [5. 变量三：KL](#5-变量三kl)
- [6. 变量四：reward](#6-变量四reward)
- [7. 变量五：损失聚合](#7-变量五损失聚合)
- [8. 变量六：mask](#8-变量六mask)
- [9. 数值实验：把上面每条都用微分验一遍](#9-数值实验把上面每条都用微分验一遍)
- [10. 速查表与踩坑清单](#10-速查表与踩坑清单)
- [11. 训练时要盯的监控量](#11-训练时要盯的监控量)

**源码基准（本文写作时拉到的最新版本）**：

| | 版本 | commit | 日期 | 文件行数 |
|---|---|---|---|---|
| veRL | `v0.9.1` | [`1876b06d`](https://github.com/verl-project/verl/verl/trainer/ppo/tree/1876b06d) | 2026-09-20 | `verl/trainer/ppo/core_algos.py` **2549 行** |
| TRL | `v1.13.0` | [`3d9261f1`](https://github.com/huggingface/trl/trl/trainer/tree/3d9261f1) | 2026-09-09 | `trl/trainer/grpo_trainer.py` **3497 行** |

本文所有行号都锁定在这两个 commit 上，并给出**可点击的 permalink**（`blob/<commit>/<path>#L<起始>-L<结束>`）。锁定 commit 而不是引用 `main`，是因为这两个文件的改动速度很快——同一篇文章在两个月内，veRL 的文件从 2200 行长到 2549 行（**+349**），TRL 从 2549 行长到 3497 行（**+948**），行号整体漂移了几百到近千行。

> 顺带一个教训：不要用浏览器打开大文件去「目测行号」。对一个 2000+ 行的文件，直接读取工具给出的行号定位并不可靠；可信的做法是 `git show <commit>:<path>` 落到本地再精确提取（本文所有行号都是这么核对的）。

---

## 1. 为什么这些变量值得单独解剖

读 RL 论文时，你觉得一切都很清楚：优势函数就是 $$A_t$$，重要性比就是 $$\pi_\theta/\pi_{\theta_{old}}$$，KL 就是 $$\mathrm{KL}(\pi_\theta \| \pi_{ref})$$。

打开框架源码，同一个符号下至少有四种不同的东西：

| 论文里的符号 | 框架里的实际含义（不同的可能取值） |
|---|---|
| $$A_t$$ | 组内减均值后除不除 std？要不要再做一次 batch 白化？留一法系数乘不乘？ |
| $$\rho_t$$ | token 级还是序列级？序列级要不要 stop-gradient？分母是 old policy 还是 rollout engine 的实际输出？ |
| $$\mathrm{KL}$$ | k1 / k2 / k3 哪个估计器？加在 reward 里还是 loss 里？有没有 straight-through？ |
| $$\mathcal{L}$$ | 全局 token 平均？序列平均？还是除以固定长度？ |

这些选择在论文里通常被一句话带过（一句「采用标准 PPO 目标」），但在实现里它们是**几十行代码的差异，且直接改变梯度落在哪些 token 上**。这篇文章把它们逐个摊开。

---

## 2. 一次迭代里，变量是怎么流动的

先把流水线画清楚，后面每一节都能挂上去：

```text
rollout（vLLM/SGLang）
  ├─→ 产出 token 序列 + rollout 引擎给出的 log_prob
  │
  ├─→ reward 计算
  │     ├─ token_level_scores：稀疏（只在 EOS 位置有值）或密集
  │     └─ [veRL] 顺手注入 KL：token_level_rewards = scores - (old_logp - ref_logp) * beta
  │
  ├─→ advantage 估计（吃 token_level_rewards + index 分组）
  │     └─→ advantages, returns（GRPO 系两者相同）
  │
  └─→ 训练
        ├─ 重算 log_prob（当前策略）  → ratio = exp(log_prob - old_log_prob)
        ├─ policy loss（vanilla / gspo / cispo / sapo …）
        ├─ [+ TRL] 在这里加 KL 项与本 rollout 重要性权重
        └─ agg_loss 聚合成标量
```

三个值得先记住的分界点：

1. **KL 加在哪**：veRL 在 reward 阶段（进入 advantage 之前），TRL 在 loss 阶段（advantage 之后）。
2. **ratio 的粒度**：vanilla 逐 token，GSPO 序列级，CISPO 保留 token 粒度但改变梯度路径。
3. **谁被归一化**：组内（GRPO）／跨组（白化）／全局 token（agg_loss）——三层归一化常常同时存在，而且是**串联**的。

---

## 3. 变量一：advantage

### 3.1 veRL 把估计器做成了注册表

`verl/trainer/ppo/core_algos.py` 用一个枚举 + 注册表管理所有估计器（[第 88–110 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L88-L110)），一共有 14 个：

| 枚举值 | 含义 | 核心公式 |
|---|---|---|
| `GAE` | 需要 critic | 逆序递推 TD 残差 |
| `GRPO` / `GRPO_VECTORIZED` | 组相对 | $$(r_i-\bar r)/\sigma_r$$ |
| `GRPO_PASSK` | 用 pass@k 做 baseline | 组均值替换为 pass@k 估计 |
| `RLOO` / `RLOO_VECTORIZED` | 留一法 | $$\frac{n}{n-1}(r_i-\bar r)$$ |
| `REINFORCE_PLUS_PLUS` | 全局 baseline | 全 batch 均值 |
| `REINFORCE_PLUS_PLUS_BASELINE` | 组 baseline + 白化 | 减组均值后再 batch 白化 |
| `REMAX` | 贪心解码作 baseline | $$r_i - r_{\text{greedy}}$$ |
| `OPO` | 组均值只算一次 | 变体 |
| `GPG` | 组策略梯度 | 只有组均值 baseline |
| `OPTIMAL_TOKEN_BASELINE` | 逐 token 最优 baseline | 含中间奖励的多轮场景 |
| `GDPO` | 多奖励解耦归一化后求和 | v0.9.1 新增（[第 110 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L110)） |

设计上有一点值得学：注释明确写了「这个枚举创建后不可变，但**扩展新估计器不必改枚举**」——直接用 `register_adv_est("my_estimator")` 注册字符串名字即可。这是插件式设计，也解释了为什么这个清单能涨到 13 个。

### 3.2 四个实现的逐行对照

**GRPO（[第 304–331 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L304-L331)）**

```python
scores = token_level_rewards.sum(dim=-1)          # 先按 response 求和 → 标量
# 按 index 分组，逐组算 mean/std
if norm_adv_by_std_in_grpo:                       # 默认 True
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
else:                                             # Dr.GRPO 路径
    scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1) * response_mask     # 广播到每个 token
return scores, scores                             # advantage 和 returns 是同一个东西
```

四个可以挑出来讲的细节：

- **策略**：`token_level_rewards.sum(dim=-1)`。GRPO 只支持 **outcome 级**奖励（一条 response 一个标量），这也是为什么它天然适合 verifier 场景。
- **`norm_adv_by_std_in_grpo=False` 就是 Dr.GRPO**，源码注释直接引了 `arXiv:2503.20783`。
- **样本数为 1 的组**（[第 315–317 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L315-L317)）：`mean=0, std=1`，即**直接返回原始 reward 不做归一化**。这是个不显眼的边界行为。
- **advantage 与 returns 返回同一个张量**。因为没有 critic，returns 没有独立含义——这一点在接 `compute_value_loss` 时容易踩坑。

**RLOO（[第 628–633 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L628-L633)）**

```python
response_num = len(id2score[index[i]])
if response_num > 1:
    scores[i] = scores[i] * n/(n-1) - id2mean[index[i]] * n/(n-1)
```

即 $$\frac{n}{n-1}(r_i - \bar r_{-i})$$，展开后就是留一法。$$n=4$$ 时放大 1.3333 倍。**没有 std 归一，也没有白化**。

**REINFORCE++-baseline（[第 578–582 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L578-L582)）**

```python
scores[i] = scores[i] - id2mean[index[i]]           # 减组均值
scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
scores = verl_F.masked_whiten(scores, response_mask) * response_mask
```

比 RLOO 多了一步 **`masked_whiten`**——对全 batch 的有效 token 做整体标准化。后果很直接：**组间的难度差被抹平**。原本「简单题组」和「难题组」的 advantage 尺度不同，白化之后统一了。这是一把双刃剑：减小方差，但丢失了「这道题难」这个信息。

**GAE（[第 245–262 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L245-L262)）**

```python
for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam_ = delta + gamma * lam * lastgaelam
    # 关键两行：mask=0 的 token 上，nextvalues 和 lastgaelam 保持不变（不是清零）
    nextvalues = values[:, t] * mask[:, t] + (1 - mask[:, t]) * nextvalues
    lastgaelam = lastgaelam_ * mask[:, t] + (1 - mask[:, t]) * lastgaelam
advantages = torch.stack(advantages_reversed[::-1], dim=1)
returns = advantages + values
advantages = verl_F.masked_whiten(advantages, response_mask)
```

两个细节：

- [第 255–256 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L255-L256) 的 mask 处理是**保持**而非**清零**：遇到被 mask 的 token（如 padding、多轮里的 observation token）时，递推量原样传递。这等价于「跳过这些 token 继续往前传」，而不是「在这里断掉」。
- GAE 的 advantage 最后也做了 `masked_whiten`（[第 262 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L262)）。**GRPO 没有这一步**——同样是 advantage，两条路径的后期处理不同。

### 3.3 TRL 把归一化暴露成了三个开关

TRL 的 advantage 计算（[`grpo_trainer.py:2786-2820`](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L2786-L2820)）把上述选择做成了配置：

| 开关 | 可选值 | 对应 |
|---|---|---|
| `scale_rewards` | `"group"` / `"batch"` / `"none"` | 组内 std ／ 全局 std ／ **完全不除**（即 Dr.GRPO） |
| `multi_objective_aggregation` | `"sum_then_normalize"` / `"normalize_then_sum"` | 先求和再归一（会塌缩）／**先逐目标归一再加权**（GDPO 的解法） |
| `num_generations > 1` | — | 组大小为 1 时 `std_rewards = 0`，与 veRL 的 `mean=0,std=1` **行为不同** |

`v1.13.0` 相比早期版本还有一处统计口径的改进：组内统计从 `mean` / `std` 换成了 **`torch.nanmean` / `nanstd`**，并引入 `unscorable_mask` 把不可打分的样本置成 NaN（[第 2790–2796 行](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L2790-L2796)）：

```python
rewards[unscorable_mask] = torch.nan
mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)
```

意义在于：奖励函数报错或超时的样本，以前要么污染组内均值、要么整组作废；现在它们以 NaN 参与统计、被跳过。这是把「坏样本」从数据管道问题降级成了统计问题。

`scale_rewards="none"` 落在源码里就是一行：

```python
advantages = rewards - mean_grouped_rewards
if self.scale_rewards != "none":
    advantages = advantages / (std_rewards + 1e-4)
```

最后那个 `is_std_zero` 会专门打点到日志里（[第 2814 行](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L2814)）——**正是排查「零梯度组」的指标**（横评篇 7.4 节实验一量化过：难度两极化时约 17% 的 rollout 落在这种组里）。

另外注意：**组大小为 1 时两个框架的行为不一致**。veRL 给 `(mean=0, std=1)`，TRL 给 `std=0`。前者等于不做归一化，后者在 `scale_rewards != "none"` 时会把 advantage 除爆（靠 `+1e-4` 兜住）。如果做跨框架对照实验，这是必须对齐的一处。

---

## 4. 变量二：概率比 ratio

### 4.1 token 级：先 clamp 再 exp

veRL 的 vanilla loss（[第 1255–1259 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1255-L1259)）：

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)   # 稳定化
ratio = torch.exp(negative_approx_kl)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
```

三点需要注意：

1. **先对 log 差做 clamp 到 $$[-20, 20]$$，再 exp**。这不是可选的优化——没有它，一个异常 token 就能让 $$\exp$$ 溢出成 inf，把整个 batch 的 loss 变成 NaN。
2. `ppo_kl` 是把 log 差取负后求平均，也就是 $$\overline{\log \pi_{old} - \log \pi_\theta}$$。这是**有符号**的 KL 一阶估计，不是真正的散度。
3. 变量名叫 `negative_approx_kl` 而不是 `log_ratio`，是因为 PPO 传统上直接用「近似 KL」当裁剪信号，这两个概念在实现里被合并了。

裁剪本身要区分两个层次（[第 1266–1280 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1266-L1280)）：

```python
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
# 第二层：dual-clip，专门处理 advantage < 0
pg_losses3 = -advantages * clip_ratio_c        # clip_ratio_c 默认 3.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

- **`clip_ratio_low` / `clip_ratio_high` 非对称**：这是 DAPO 的 clip-higher，通过 config 传两个不同的值即可，无需改代码。
- **dual-clip（`clip_ratio_c = 3.0`）**：当 advantage 为负时，再加一个下界 $$\min(-A \cdot c,\ \cdot)$$，防止「负 advantage × 极小 ratio」产生过大的梯度。这个下界只对负 advantage 生效。

### 4.2 序列级：veRL 和 TRL 写得不一样

这是全文最值得细看的一处。

GSPO 论文的核心主张是：把 token 级比率换成**序列级**比率

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i)}{\pi_{old}(y_i)}\right)^{1/|y_i|} = \exp\left(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta(y_{i,t})}{\pi_{old}(y_{i,t})}\right)$$

然后拿 $$s_i$$ 去做裁剪。问题来了：直接用 $$s_i$$ 意味着整条序列共享一个标量，**token 级的梯度结构就丢了**。

**veRL 的解法是组合比率**（[第 1583–1594 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1583-L1594)）：

```python
seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

# s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x,y_i,<t) / sg[π_θ(y_i,t|x,y_i,<t)]
# log 空间：log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
seq_importance_ratio = torch.exp(log_seq_importance_ratio)
```

关键在那一行加法：

- **前向值**：`log_prob - log_prob` 抵消，只剩 `negative_approx_kl_seq`——**整条序列同一个标量**。所以裁剪判断确实用的是序列级比率。
- **反向梯度**：第一项 `log_prob` 的导数是 1，后两项都 `.detach()` 了。所以 $$\partial \log s_{i,t} / \partial \log\pi_\theta(y_{i,t}) = 1$$——**梯度仍然逐 token 流动**。

也就是说：**裁剪用序列粒度、梯度用 token 粒度**，一个 stop-gradient 同时拿到两边的性质。注释里 `sg[·]` 的写法（signaling stop-gradient）在论文里是一个符号，在代码里就是 `.detach()`。

**TRL 的实现是另一种**（[`grpo_trainer.py:3179-3188`](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L3179-L3188)）：

```python
elif self.importance_sampling_level == "sequence":
    log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
    log_importance_weights = log_importance_weights.unsqueeze(-1)
coef_1 = torch.exp(log_importance_weights)
```

序列级 log 比算完后 `unsqueeze(-1)` **直接 broadcast 到所有 token**，然后进入和 token 级完全相同的 loss 分支。这里**没有 stop-gradient**，梯度会经由这个标量回传到每个 token 的 log_prob 上。

| | veRL（`compute_policy_loss_gspo`） | TRL（`importance_sampling_level="sequence"`） |
|---|---|---|
| 裁剪判断依据 | 序列级标量 | 序列级标量 |
| 梯度粒度 | **逐 token**（靠组合比率） | 整条序列共享一个系数 |
| 聚合方式 | **v0.9.1 起可配置**（默认变成 `token-mean`，见下） | 由 `loss_type` 分支决定 |
| 裁剪方向 | `torch.maximum`（单向，无 dual-clip） | 同左 |

**结论：两个框架里叫同一个名字的「序列级」，梯度路径不等价。** 做跨框架对照实验时，这足以解释一部分结果差异。

**还有一个会随时间变化的选择**：GSPO 的**损失聚合口径**。在 veRL `0.7.0.dev` 里它是硬编码的：

```python
loss_agg_mode="seq-mean-token-mean",   # 写死的
```

到了 `v0.9.1` 变成了可配置，而且默认值换了——源码注释写得很直白（[第 1604 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1604-L1607)）：

> NOTE: differ from GSPO original paper that aggregate the loss at the sequence level (seq-mean-token-mean), we support aggregate the loss in different modes, **default is token-mean**.

也就是说：**实现主动偏离了原论文，并把默认值改成了 `token-mean`**。同一份配置在 0.7 和 0.9 上跑，梯度尺度会不一样。这类「默认值在某次发版里悄悄改了」的变更，比函数签名变更更难察觉——如果你做过跨版本复现，这类差异值得优先排查。

---

## 5. 变量三：KL

### 5.1 三种估计器都在

`kl_penalty_forward`（[第 2216–2250 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L2216-L2250)）把 Schulman 博客里的几种都实现了：

| 名称 | 公式 | 特性 |
|---|---|---|
| `kl` / `k1` | $$\log p - \log q$$ | **有符号**，无偏但方差大 |
| `abs` | $$\lvert \log p - \log q\rvert$$ | 非负，但有偏 |
| `mse` / `k2` | $$\tfrac12 (\log p - \log q)^2$$ | **恒非负，且梯度是无偏的** |
| `low_var_kl` / `k3` | $$r - \log r - 1$$，其中 $$r = \frac{q}{p}$$ | 非负、低方差；源码里把中间量 clip 到 $$[-20,20]$$、结果 clip 到 $$[-10,10]$$ |

三者的取舍关系正是 Schulman 那篇博客的结论：**k1 与 k3 的期望是 KL，但它们的梯度不是 KL 的梯度；只有 k2 的梯度是对的**。

### 5.2 verl 里那个 `+` 后缀

源码 [第 2200–2214 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L2200-L2214) 藏了一个技巧：

```python
forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
    return forward_score

backward_score = 0.5 * (logprob - ref_logprob).square()      # k2
return backward_score - backward_score.detach() + forward_score.detach()
```

这就是经典的 **straight-through 技巧**：前向返回 k3 的值（低方差），反向返回 k2 的梯度（正确）。**用的时候写 `k3+`，只多一个加号。**

**但这个加号在旧版本里是坏的。** 对比两个版本会发现一处真实的 bug 修复——`v0.9.1` 在调用前多做了一步剥离（[第 2201 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L2201-L2203)）：

```python
# Strip the optional '+' suffix so e.g. "k3+" dispatches to "k3".
base_kl_penalty = kl_penalty[:-1] if kl_penalty.endswith("+") else kl_penalty
forward_score = kl_penalty_forward(logprob, ref_logprob, base_kl_penalty)
```

旧版直接把 `"k3+"` 传下去，而 `kl_penalty_forward` 里的分支判断是 `if kl_penalty in ("low_var_kl", "k3")` —— `"k3+"` 不等于 `"k3"`，**匹配不上，会一路落到的 `raise NotImplementedError`**。也就是说：在 `0.7.0.dev` 上，`kl_penalty="k3+"` 这个配置是跑不通的；`v0.9.1` 修了它。

这件事本身很小，但它说明一个更普遍的判断：**引用源码时要连版本一起锁定**。同一个 `k3+`，在两个月前的版本上是崩溃配置，在最新版本上是推荐用法。

同一份 `detach()` 手法在这篇文章里出现了三次（GSPO、CISPO、KL straight-through），模式完全一样：**「要 A 的值，但不要 A 的梯度」**。

### 5.3 两种落点，β 不可比

**veRL：注入 reward**（[第 1136–1137 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1136-L1137)，只有两行）

```python
kl = old_log_prob - ref_log_prob
return token_level_scores - kl * kl_ratio
```

注意用的是 **k1、有符号**。这带来一个容易被忽略的性质：当 `old_log_prob < ref_log_prob`（当前策略比参考策略更不自信）时，KL 项为负，**reward 被加上去**。稀疏奖励场景下，这一项把「只有一个 EOS 分数」变成「逐 token 的密集信号」。

**TRL：加在 loss**（[`grpo_trainer.py:3192-3243`](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L3192-L3243)）

```python
per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
if self.args.use_bias_correction_kl:
    per_token_kl = per_token_kl * coef_1
...
per_token_loss = per_token_loss + self.beta * per_token_kl
```

用的是 **k3、恒非负**，且直接加在 loss 上。

两种落点有三个实质差别：

1. **KL 是否参与 advantage 归一化**。注入 reward 时，KL 会和奖励一起进入组内均值/标准差；加在 loss 时它完全独立。
2. **数值口径不同**：k1 可正可负、k3 恒非负。**同样的 `beta=0.04`，两者约束强度不是一回事**，跨框架比较必须换算是先对齐估计器。
3. **能否被裁剪**。进 reward 的 KL 会改变 advantage 的符号与大小，进而影响 `torch.where(advantages < 0, ...)` 走哪个分支——KL 项大的 token 可能因此进入 dual-clip 路径。

### 5.4 自适应控制器

`AdaptiveKLController`（[第 153–174 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L153-L174)）来自 InstructGPT 论文：

```python
proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
mult = 1 + proportional_error * n_steps / self.horizon
self.value *= mult
```

比例误差被 **clip 到 ±20%**，所以单次更新的调整幅度有界。要固定系数就换 `FixedKLController`（[第 177–190 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L177-L190)，`update` 是 no-op）。

---

## 6. 变量四：reward

reward 这一环在框架里往往被说得含糊，其实就三个问题：

**第一，reward 的粒度。** veRL 的数据结构始终是 `(bs, response_length)` 的 **token 级**张量，outcome reward 的处理方式是「放在最后一个有效 token 上，其余为 0」；GRPO 的估计器拿到后再 `sum(dim=-1)` 把序列压回标量。这个「先铺开再收拢」的设计让 outcome 与 process reward 可以共用同一条通路。

**第二，与 mask 的对齐。** 所有 reward 在进入 advantage 之前都必须乘上 `response_mask`。源码里到处是这一步（如 [第 329 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L329) `scores.unsqueeze(-1) * response_mask`）。漏掉它，padding 位置的 reward 就会污染组内均值。

**第三，引擎给的 logprob 与训练侧重算的 logprob 是两个量。** [第 1255 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1255) 用的是 `old_log_prob`（rollout 时记录的），而 `log_prob` 是训练时前向重算的。**训推不一致就发生在这两个量之间**——横评篇 7.4 节实验三量化过：噪声到 0.30 时 27% 的 token 会「假装越界」，然后被裁剪规则误伤。

至于 off-policy 修正的落点，源码里非常干净——就是一行乘法：

```python
if rollout_is_weights is not None:
    pg_losses = pg_losses * rollout_is_weights     # veRL 第 1365-1366 行
```

TRL 里对应两个开关：`vllm_importance_sampling_correction`（[第 2682 行](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L2682)）和 `off_policy_mask_threshold`（[第 3161 行](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L3161)，越界样本直接置零——即 IcePop 式的硬 mask）。

---

## 7. 变量五：损失聚合

`agg_loss`（[第 1140–1207 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1140-L1207)）实现多种口径。这是**最容易被忽略、但对长回答影响最大**的一环：

| 模式 | 实现 | 长序列的相对权重 |
|---|---|---|
| `token-mean` | `masked_sum(loss) / 全局有效token数` | ∝ 长度 |
| `token-sum` | `masked_sum(loss) * dp_size` | ∝ 长度，但**不做归一化**（v0.9.1 新增） |
| `seq-mean-token-sum` | 先按序列 `sum`，再按序列平均 | 每条序列等权（长序列总贡献更大） |
| `seq-mean-token-mean` | 序列内先 `mean`，再序列平均 | 每条序列等权，长度已被除掉 |
| `seq-mean-token-sum-norm` | 同上，末尾再除以固定长度 | 与真实长度无关 |

`token-sum` 是 `v0.9.1` 新增的口径，源码注释解释了它的用途（[第 1176–1180 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1176-L1180)）：DDP/FSDP 会对各 rank 的梯度取平均，把每个 rank 的局部 token 和乘上 `dp_size`，还原后正好等于全局 batch 上所有有效 token 的和。

同一版还做了一件更硬的事：**把「并行度不变性」从注释里的约定变成了运行时强制**——三处归一化分支各加了一个守卫：

```python
if dp_size > 1:
    raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
```

（[第 1173 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L1173) 等三处）。旧版本只在注释里要求调用方传全局计数，新版本直接**拒绝在缺参数时继续算**。对多机训练来说这是个很实际的改进：以前传错只会让 loss 尺度悄悄不对，现在会直接报错。

TRL 里这套口径换了个形式出现在 `loss_type` 分支（[第 3246–3275 行](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L3246-L3275)）：

| TRL `loss_type` | 归一化代码 | 等价口径 |
|---|---|---|
| `grpo` / `sapo` | `(per_token_loss*mask).sum(-1)/mask.sum(-1)).mean()` | `seq-mean-token-mean` |
| `bnpo` | `(per_token_loss*mask).sum()/mask.sum()` | `token-mean` |
| `dr_grpo` | `(per_token_loss*mask).sum()/(bs*max_completion_length)` | `seq-mean-token-sum-norm` |
| `cispo` / `dapo` | `(per_token_loss*mask).sum()/(num_items_in_batch/nproc)` | `token-mean`（跨进程） |

**注意最后一行**：TRL 里 `dapo` 与 `cispo` 共用 `num_items_in_batch` 归一化，且要除以进程数——这是为了在数据并行下保持全局口径一致。`v1.13.0` 在这一行上又多补了一层梯度累积的校正（[第 3261–3266 行](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py#L3261-L3266)）：

```python
normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
if mode == "train":   # in eval, the batch is neither split across steps nor accumulated
    normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
loss = (per_token_loss * mask).sum() / normalizer
```

因为 `num_items_in_batch` 覆盖的是**整个 generation batch**，而一次 optimizer step 只消费其中一个累积窗口，所以要按 `steps_per_generation` 缩放回来。**改梯度累积步数会导致 loss 尺度变化**——这是同一类「归一化分母必须跟着并行/累积配置走」的问题，两个框架都在往同一个方向补：让 loss 与并行配置解耦。veRL 的 `agg_loss` 里也有对应的 `dp_size` 乘法，注释写得很直白：

> Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

**这是一条硬约束**：改变并行度不应该改变 loss 的数值。如果换卡数后 loss 曲线跟着变，先怀疑归一化分母有没有乘对并行度。

顺带一提，veRL 的注释还留了另一条实现约束：FSDP 下聚合后的 loss 可直接反向，但 **Megatron 下还需要除以 `num_microbatches` 与 `cp_size`**，因为流水线调度会改变梯度的累加方式。

---

## 8. 变量六：mask

`response_mask` 的语义在源码里有明确定义（GAE 函数注释，[第 231 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L231)）：

> `[EOS]` mask. The token after `[EOS]` have mask zero.

即**只有真正生成的、且在 EOS 之前的 token 参与训练**。它同时承担四个职责：

1. 决定哪些位置算 loss（`agg_loss` 的分母与分子）
2. 决定哪些位置算 advantage（GAE 的递推保持语义）
3. 决定白化/组内统计的范围（`masked_whiten`、`masked_mean`）
4. 决定 reward 的落点（outcome reward 放在最后一个有效 token）

一个实践建议：**把 mask 的和当作首要 sanity check**。如果一步之后 `mask.sum()` 突然变化很大，通常意味着截断策略或 EOS 判定出了变化，而不是模型行为变了——这跟 loss 曲线是完全独立的信号。

---

## 9. 数值实验：把上面每条都用微分验一遍

`tools/rl_var_grad_lab.py`（纯 numpy、CPU 可跑）做了四组验证。这里的公式都逐行对照上面引用的源码。

### 9.1 方法论：含 `detach()` 的表达式怎么数值验证

朴素做法（中心差分）在这里**会得出错误结论**，因为数值微分只能测前向，而 `detach()` 恰恰是「前向用这个值、反向不回传」。直接对带 `detach()` 的表达式做差分，测出来的是「假设没写 detach」的梯度。

解法是**双变量技巧**：令 $$g(x_{var}, x_{fixed}) = x_{var} - x_{fixed}$$。当两者相等时前向值为 0，而对 $$x_{var}$$ 的偏导恒为 1。把 `sg[·]` 的部分用 `x_fixed` 计算、只对 `x_var` 做差分，就精确复现了 stop-gradient 的行为。

### 9.2 实验一：梯度落在哪些 token 上

6 个 token，`clip=[1-0.2, 1+0.28]`（DAPO 风格偏上裁剪）：

```
   token    ratio     vanilla       CISPO        GSPO
  ----------------------------------------------------
       0    0.980     -0.1634     -0.1634     -0.1661
       1    1.010      0.1683      0.1683      0.1661
       2    1.350      0.0000     -0.2133     -0.1661     ← 越界
       3    2.460      0.0000     -0.2133     -0.1661     ← 越界
       4    0.301     -0.0502     -0.1333     -0.1661     ← 越界
       5    0.990      0.1650      0.1650      0.1661
```

三种行为一目了然：

- **vanilla**：2/6 个 token 梯度为 0——被裁剪的 token 直接退出训练。且没被裁剪时梯度大小 ∝ ratio，各 token 不同。
- **CISPO**：**0/6 为 0**，全部保留。梯度是 `−clip(ratio)·A`，越界的 token 权重被压到阈值（t=2 实测 −0.2133 = −1.28×1/6，与 `clip(1.35→1.28)` 完全一致）。
- **GSPO**：所有 token 的梯度**完全相同**（−0.1661 = −exp(seq_log_ratio)/6）。这正是组合比率的直接后果。

GSPO 的「整条序列一起裁」在数值上表现为：

```
  场景                        seq-ratio    越界?     A>0 有梯度     A<0 有梯度
  --------------------------------------------------------------------
  logp 整体偏移 +0.000              1.000      否         4/4         2/2
  logp 整体偏移 +0.182              1.200      否         4/4         2/2
  logp 整体偏移 +0.470              1.600      是         0/4         2/2
```

阈值一越界，**同号的 token 整批失去梯度**（4/4 → 0/4），而另一符号的 token 因为 `torch.maximum` 的分支结构仍保留。**关键在于判断粒度**：不再是「这个 token 的 ratio 超了没有」，而是「整条轨迹的 ratio 超了没有」。

### 9.3 实验二：四种 advantage 在同一份 rollout 上的差异

2 个 prompt × 4 条 rollout，二值奖励：

```
  prompt     rollout奖励     GRPO(含std)     Dr.GRPO        RLOO     RF++-baseline
  -----------------------------------------------------------------------------
        0           1.0         0.8660      0.5000      0.6667            1.0690
        0           1.0         0.8660      0.5000      0.6667            1.0690
        0           0.0        -0.8660     -0.5000     -0.6667           -1.0690
        0           0.0        -0.8660     -0.5000     -0.6667           -1.0690
        1           1.0         1.5000      0.7500      1.0000            1.6036
        1           0.0        -0.5000     -0.2500     -0.3333           -0.5345
```

看点有三：

- **同一个 rollout，advantage 从 0.5 到 1.069 不等**（组 0 的第一条），相差一倍以上。
- **RLOO 的 1.3333 倍系数**可以被手工验算：$$0.5 \times \frac{4}{3} = 0.6667$$，实测一致。
- **RF++-baseline 抹平了组间难度差**：组 0 与组 1 的原始均值是 0.5 和 0.25，白化之后都到同一尺度。

**一个源码级的尺度细节**：veRL 用 `torch.std()`（[第 321 行](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py#L321)），torch 默认 `unbiased=True`（`ddof=1`）；如果换成 numpy 默认口径（`ddof=0`），$$G=4$$ 时标准差不同，advantage 整体差 **1.1547 倍**（$$\sqrt{4/3}$$）。这个系数会一路传到有效学习率上。

### 9.4 实验三与实验四

**KL 的两种落点**（5 个 token，`beta=0.04`）：

```
  verl: reward_t = score_t - (old_logp - ref_logp) * beta
        逐 token reward = [ 0.004 -0.006  1.008 -0.006  0.002]
  TRL : per_token_loss += beta * (exp(ref-logp) - (ref-logp) - 1)
```

veRL 路径下，原本只在最后一个 token 有值的稀疏奖励被 KL 项**铺成了逐 token 的密集信号**，而且出现了负值（第二项 −0.006）。这件事的影响比看上去大：advantage 估计器的输入分布变了。

**loss 聚合**（4 条序列，长度 10/6/3/8，总 27 个有效 token）：

```
  口径                                数值   含义
  ------------------------------------------------------------------------------
  token-mean                   -0.4425   除以全局有效 token 数
  seq-mean-token-sum           -2.9871   先按序列求和，再按序列平均
  seq-mean-token-mean          -0.5464   序列内先平均，再序列间平均
  seq-mean-token-sum-norm      -1.1948   按固定长度归一（与真实长度无关）
```

**同一份数据，四个口径下的 loss 相差近 7 倍**。这不是数值噪声，是四种不同的加权哲学。换口径而不调学习率，等价于换了学习率。

---

## 10. 速查表与踩坑清单

### 10.1 变量对照速查

| 变量 | veRL | TRL | 最容易踩的坑 |
|---|---|---|---|
| advantage 归一化 | 组内 mean/std（`norm_adv_by_std_in_grpo`） | `scale_rewards` = group/batch/none | 组大小为 1 时两者行为不同 |
| advantage 后处理 | GAE 与 RF++ 做 `masked_whiten`，GRPO/RLOO 不做 | — | 白化会抹平组间难度差 |
| ratio 粒度 | vanilla 逐 token；GSPO 组合比率 | `importance_sampling_level` = token/sequence | 序列级在两框架梯度路径不同 |
| ratio 稳定化 | `clamp(log_diff, -20, 20)` | 同左 | 去掉它就会 NaN |
| KL 估计器 | k1/abs/k2/k3，`k3+` 触发 straight-through | k3 | k1 与 k3 的 β 不可比 |
| KL 落点 | 注入 reward（k1） | 加在 loss（k3） | 是否参与 advantage 归一化 |
| loss 聚合 | `agg_loss` 四模式 | `loss_type` 分支 | 换口径＝换有效学习率 |
| off-policy 修正 | `rollout_is_weights` 一行乘法 | `vllm_importance_sampling_correction` / `off_policy_mask_threshold` | 默认不开 |

### 10.2 八条踩坑清单

1. **换框架做对照实验，先对齐 advantage 口径**（含 std 与否、白化与否、`ddof` 是 0 还是 1）。否则你会把「归一化差异」误读成「算法差异」。
2. **`torch.std` 默认 unbiased**。手写 numpy 复现时若用默认 `ddof=0`，advantage 尺度会差 $$\sqrt{n/(n-1)}$$ 倍。
3. **KL 的估计器要跟落点配对**。用 k1 注入 reward、又声称在「加 KL 惩罚」，和用 k3 加在 loss，不是同一件事。
4. **`k3+` 只多一个加号，但拿到的是正确梯度**。用了 k3 却没加加号，等于在用有偏梯度做惩罚。
5. **改了并行度必须验 loss 不变**。归一化分母里要乘 `dp_size`；Megatron 下还要处理 `num_microbatches` 与 `cp_size`。
6. **长回答的权重取决于聚合口径**。做长 CoT 训练时，`seq-mean-token-sum`（原版 GRPO）会让长回答整体被放大，DAPO 主张的 `token-mean` 才是每个 token 一视同仁。
7. **组大小为 1 是条隐藏分支**。veRL 与 TRL 的处理不同，且它恰恰是「零梯度组」之外的另一个边界。
8. **`detach()` 的位置决定了梯度落在哪**。GSPO 的裁剪、CISPO 的权重、KL 的 straight-through，三处用的都是同一招：「要这个变量的值，但不要它的梯度」。

**这篇文章真正想说的是**：RL 框架之间的差异，很大一部分不在算法清单上，而在这些「论文里一句话带过」的实现选择里。看懂它们，比多试三个框架更有用。

---

## 11. 训练时要盯的监控量

前面十节讲的是「这些变量怎么算」，这一节讲「算完之后看哪个数」。放在源码对照之后，是因为每个监控量都对应上面某个具体实现开关——不看实现，阈值没有意义。

| 监控量 | 定义 | 健康范围 | 异常时意味着什么 |
|:---|:---|:---|:---|
| clip fraction | $$\mathbb P\big(\lvert\rho_t-1\rvert>\epsilon\big)$$ | 0.05 ~ 0.3 | > 0.5：策略跑太快（lr 太大或 epoch 太多）；≈ 0：裁剪没起作用，等于普通策略梯度 |
| approx KL | $$\frac12\mathbb E[(\ell^\theta-\ell^{\rm old})^2]$$ 或 $$\mathbb E[\ell^{\rm old}-\ell^\theta]$$ | 与 $$\beta$$ 的目标 KL 同量级 | 持续上升 = 策略在漂移，考虑早停 |
| explained variance | $$1-\dfrac{\mathrm{Var}(R-\hat V)}{\mathrm{Var}(R)}$$ | 0.4 ~ 0.9 | < 0：critic 还不如常数；> 0.95：可能过拟合了 batch |
| value loss | $$\mathbb E[(\hat V-R)^2]$$（常取 0.5×，且做 clipped 版） | 缓慢下降 | 暴涨 = reward scale 变化或 critic lr 过大 |
| ratio 分位数 | $$\rho_t$$ 的 p1 / p50 / p99 | p99 < $$1+\epsilon$$ 的若干倍 | p99 随序列长度增长 → 见 §4.2，考虑 GSPO |
| advantage 均值 / std | — | 组内应近似零均值 | 非零 = baseline 有偏；std 为 0 = 零梯度组 |
| 熵 | $$\frac{1}{\sum\lvert o\rvert}\sum\mathcal H$$ | 缓慢下降，不能塌到 0 | 骤降 = 模式塌缩；上升 = lr 过大或熵系数过大 |
| KL to ref | $$k_3$$ 的序列平均 | 与 $$\beta$$ 设定一致 | 见 §5，注意梯度是否穿过 |

两个容易搞错的细节：

- **approx KL 的两种写法不等价**：$$\mathbb E[\ell^{\rm old}-\ell^\theta]$$（$$k_1$$ 形式）可以为负、方差大；$$\frac12\mathbb E[(\Delta\ell)^2]$$（$$k_2$$ 形式）非负、有偏。监控用哪个都行，但**切换时阈值要跟着改**——这正是 §5.1 里 k1/k2/k3 差异在监控侧的投影。
- **explained variance 用 $$R$$ 的真值方差做分母**。如果你的 $$R$$ 是二值的（0/1），$$\mathrm{Var}(R)$$ 很小，EV 会显得异常低，此时更该看 value loss 的绝对量级。

---

## 参考

**源码**
- veRL `core_algos.py`（基准 `v0.9.1`）：[源文件 permalink](https://github.com/verl-project/verl/blob/1876b06d/verl/trainer/ppo/core_algos.py) ｜ [仓库](https://github.com/verl-project/verl) ｜ [DeepWiki](https://deepwiki.com/verl-project/verl)
- TRL `grpo_trainer.py`（基准 `v1.13.0`）：[源文件 permalink](https://github.com/huggingface/trl/blob/3d9261f1/trl/trainer/grpo_trainer.py) ｜ [仓库](https://github.com/huggingface/trl) ｜ [DeepWiki](https://deepwiki.com/huggingface/trl)

**算法**

- GAE：[arXiv 1506.02438](https://arxiv.org/abs/1506.02438) ｜ PPO：[arXiv 1707.06347](https://arxiv.org/abs/1707.06347) ｜ Dual-Clip PPO：[arXiv 1912.09729](https://arxiv.org/pdf/1912.09729)
- GRPO（DeepSeekMath）：[arXiv 2402.03300](https://arxiv.org/abs/2402.03300) ｜ Dr.GRPO：[arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- DAPO：[arXiv 2503.14476](https://arxiv.org/abs/2503.14476) ｜ GSPO：[arXiv 2507.18071](https://arxiv.org/abs/2507.18071) ｜ CISPO：[arXiv 2506.13585](https://arxiv.org/abs/2506.13585)
- RLOO：[arXiv 2402.14740](https://arxiv.org/abs/2402.14740) ｜ REINFORCE++：[arXiv 2501.03262](https://arxiv.org/abs/2501.03262)
- KL 估计器：[John Schulman, Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) ｜ 自适应 KL（InstructGPT）：[arXiv 2203.02155](https://arxiv.org/abs/2203.02155)

**本文的自算脚本**
- `tools/rl_var_grad_lab.py`：四个实验（梯度路径 / advantage 口径 / KL 落点 / 聚合口径），纯 numpy，CPU 可跑
- 配套系统侧文章：[RL 训练框架全景横评（2026）](/2026/09/29/rl-training-frameworks-survey/)

---

*上一篇：[RL 训练框架全景横评（2026）](/2026/09/29/rl-training-frameworks-survey/) ｜ 下一篇：[PyTorch 采样实现深读](/2026/09/30/pytorch-sampling-internals/)*
