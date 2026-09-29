"""
RL 框架在算什么：三组纯 numpy 数值实验（无 GPU、无框架依赖）。

配套文章：_posts/2026-09-29-rl-training-frameworks-survey.md

实验一：GRPO 组相对优势 —— 为什么「全对/全错组」是纯浪费（DAPO 的动机）
实验二：四种裁剪/加权策略在同一份假数据上的梯度权重差异
        （PPO-clip / GRPO token-clip / CISPO / GSPO 序列级）
实验三：训推不一致（rollout 引擎 ≠ 训练引擎）如何放大重要性比，
        以及 TIS / IcePop / CISPO 三种修法各自丢掉多少梯度

运行：
  /Users/congyuan/Software/miniconda3/bin/python3 tools/rl_obj_numeric_lab.py
"""

import numpy as np

rng = np.random.default_rng(20260929)
np.set_printoptions(precision=4, suppress=True)


def section(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ----------------------------------------------------------------------
# 实验一：组相对优势与「零梯度组」
# ----------------------------------------------------------------------
section("实验一：GRPO 组相对优势 —— 有多少组是白算的？")

G = 8                 # 每条 prompt 采 G 个 rollout
N_PROMPT = 20000

print(f"\n设定：{N_PROMPT} 条 prompt，每组 G={G} 条 rollout。")
print("关键：**题目难度不是均匀的**，用 Beta(1.5, 3.0) 建模每条 prompt 的正确率")
print("（偏难、长尾，接近数学题 RL 中期的真实分布）。\n")

p_prompt = rng.beta(1.5, 3.0, N_PROMPT)
print(f"  题目正确率分布：P10={np.percentile(p_prompt,10):.2f}  "
      f"P50={np.percentile(p_prompt,50):.2f}  P90={np.percentile(p_prompt,90):.2f}")

# 二值奖励：答对 1、答错 0
r = (rng.random((N_PROMPT, G)) < p_prompt[:, None]).astype(float)
group_mean = r.mean(axis=1)
group_std = r.std(axis=1)

# GRPO 的组相对优势（去均值、除以组内标准差）
adv_std = (r - group_mean[:, None]) / (group_std[:, None] + 1e-4)
# Dr.GRPO 的变体：只去均值，不除标准差
adv_nostd = r - group_mean[:, None]

n_right = r.sum(axis=1)
uniform = (n_right == 0) | (n_right == G)   # 全错或全对
print(f"\n  全错组（0/{G}）占比        : {np.mean(n_right == 0):>7.2%}")
print(f"  全对组（{G}/{G}）占比        : {np.mean(n_right == G):>7.2%}")
print(f"  **零梯度组合计**          : {uniform.mean():>7.2%}")
# 按 token 计：简单题答对时通常也短，这里按平均长度近似
print(f"  → 约 {uniform.mean():.0%} 的 rollout 算力完全不产生梯度。")

print(f"\n  分难度看：真正浪费算力的是两极")
print(f"  {'题目正确率区间':>16}{'组数占比':>10}{'零梯度组占比':>14}")
print("  " + "-" * 42)
for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)):
    m = (p_prompt >= lo) & (p_prompt < hi)
    if m.sum() == 0:
        continue
    z = np.mean(uniform[m])
    print(f"  [{lo:.1f}, {hi:.1f}){m.mean():>10.2%}{z:>14.2%}")

print(f"\n  → 太简单（全都答对）和太难（全都答错）的题，整组 rollout 白算。")
print(f"    这就是 DAPO dynamic sampling 的动机：过采样 → 过滤掉组内奖励")
print(f"    无方差的 prompt → 同样的算力换到更多有效梯度。")
print(f"    DUPO 的做法更省：直接复制非零标准差的样本把 batch 填满。")

# 标准差归一带来的偏差（Dr.GRPO 论点）
print(f"\n  去 std vs 不去 std：难度不同的组被放大的倍数")
print(f"  {'组内答对数':>10}{'std':>8}{'含std的 advantage 幅度':>22}{'Dr.GRPO 幅度':>14}")
print("  " + "-" * 56)
for k in range(1, G):
    rv = np.zeros(G); rv[:k] = 1.0
    sd = rv.std()
    a_std = (1.0 - rv.mean()) / (sd + 1e-4)
    a_no = 1.0 - rv.mean()
    print(f"  {k:>10}{sd:>8.4f}{abs(a_std):>22.4f}{abs(a_no):>14.4f}")
print("  → 只答对 1 条 vs 答对 4 条，含 std 的 advantage 幅度相差数倍：")
print("    难组（方差小）被放大，这就是 Dr.GRPO 主张去掉 std 的原因。")


# ----------------------------------------------------------------------
# 实验二：四种裁剪/加权策略的梯度权重
# ----------------------------------------------------------------------
section("实验二：PPO-clip / GRPO / CISPO / GSPO 在同一份数据上的梯度权重")

# 构造一份有代表性的数据：多数 token 比率接近 1，少数偏离（长尾）
N = 200000
log_ratio = rng.normal(0.0, 0.35, N)     # log(pi_new / pi_old)
ratio = np.exp(log_ratio)
adv = rng.normal(0.0, 1.0, N)            # advantage（已归一化）

EPS_LOW, EPS_HIGH = 0.2, 0.2             # PPO 的经典裁剪
CISPO_C = 3.0                            # CISPO 的 IS 权重上界

print(f"\n设定：{N} 个 token，log-ratio ~ N(0, 0.35)，advantage ~ N(0,1)\n")
print(f"  重要性比 ratio 的分位数："
      f"P50={np.percentile(ratio,50):.3f}  "
      f"P95={np.percentile(ratio,95):.3f}  "
      f"P99={np.percentile(ratio,99):.3f}  "
      f"max={ratio.max():.3f}")

# PPO / GRPO 的 token 级裁剪：超出 [1-e, 1+e] 的 ratio 被截断（梯度为 0）
ppo_mask = ((ratio >= 1 - EPS_LOW) & (ratio <= 1 + EPS_HIGH)) | \
           ((adv > 0) == (ratio > 1))          # 方向一致时也不截断（min 的语义）
ppo_clipped = np.clip(ratio, 1 - EPS_LOW, 1 + EPS_HIGH)
ppo_grad = ppo_mask.astype(float)              # 是否有梯度流过

# CISPO：不截断 ratio 本身，而是把 IS 权重截断到 C，且对权重做 stop-gradient
cispo_w = np.minimum(ratio, CISPO_C)
cispo_grad = np.ones_like(ratio)               # 所有 token 都有梯度

# GSPO：序列级比率 —— 用长度归一后的序列比率代替 token 比率
# 这里模拟：序列级比率 = exp( mean(log_ratio) )，长度越长方差越小
SEQ_LEN = 4096
n_seq = N // SEQ_LEN
seq_log_ratio = log_ratio[: n_seq * SEQ_LEN].reshape(n_seq, SEQ_LEN).mean(axis=1)
seq_ratio = np.exp(seq_log_ratio)
s_len_norm = 1.0 / SEQ_LEN
gspo_seq_ratio = np.exp(seq_log_ratio * s_len_norm * SEQ_LEN / SEQ_LEN)

print(f"\n  {'策略':<28}{'有梯度的 token 占比':>20}{'有效梯度权重均值':>18}")
print("  " + "-" * 68)
print(f"  {'PPO/GRPO token-clip (e=0.2)':<28}{ppo_grad.mean():>20.2%}"
      f"{(ppo_clipped * ppo_grad).mean():>18.4f}")
print(f"  {'CISPO (截断 IS 权重 C=3)':<28}{cispo_grad.mean():>20.2%}"
      f"{cispo_w.mean():>18.4f}")
print(f"  {'原始（不修正）':<28}{1.0:>20.2%}{ratio.mean():>18.4f}")

print(f"\n  序列级 vs token 级比率的方差（GSPO 的核心论点）")
print(f"    序列长度 {SEQ_LEN} 时：")
print(f"      token 级 log-ratio  std = {log_ratio.std():.4f}")
print(f"      序列级 log-ratio    std = {seq_log_ratio.std():.4f}")
print(f"      → 序列级比率的波动被压低 {log_ratio.std()/seq_log_ratio.std():.1f}×")
print(f"    这就是为什么 MoE 上 GSPO 不需要 Routing Replay：")
print(f"      专家路由变化是 token 级的扰动，取序列均值后自然被平滑掉。")


# ----------------------------------------------------------------------
# 实验三：训推不一致 + 三种修法
# ----------------------------------------------------------------------
section("实验三：训推不一致放大重要性比，以及三种修法各丢多少梯度")

print("\n建模：推理引擎给出的 logprob 与训练引擎前向的真值差一个噪声。")
print("      FP8 / 不同 kernel / 不同 dtype 会放大这个噪声。\n")

N2 = 200000
# 真实的重要性比：ratio_true = pi_new / pi_old，on-policy 一步内很接近 1
log_ratio_true = rng.normal(0.0, 0.05, N2)
# 推理引擎记录的 logprob 与训练引擎真值差一个噪声 eps：
#   log pi_rollout = log pi_old + eps
# 训练时若拿 pi_rollout 当分母，算出的 ratio 会变成 ratio_true * exp(-eps)
SIGMA_LIST = [0.00, 0.05, 0.15, 0.30]   # 0 = 两引擎完全一致；越大 = 精度越差

print(f"  {'噪声 sigma':>10}{'ratio P95':>12}{'ratio P99':>12}"
      f"{'超界(>1.2)占比':>16}{'权重均值':>12}")
print("  " + "-" * 64)
rows = {}
for sg in SIGMA_LIST:
    eps = rng.normal(0.0, sg, N2)
    log_ratio_obs = log_ratio_true - eps
    rt = np.exp(log_ratio_obs)
    over = np.mean(rt > 1.2)
    rows[sg] = rt
    print(f"  {sg:>10.2f}{np.percentile(rt,95):>12.3f}{np.percentile(rt,99):>12.3f}"
          f"{over:>16.2%}{rt.mean():>12.4f}")

print("\n  → sigma 从 0.05 涨到 0.30（相当于从 bf16 掉到激进量化 / FP8 rollout），")
print("    超界 token 占比从个位数涨到两位数：")
print("    这些 token 的重要性比其实没变，是引擎差异假装它变了。")

print(f"\n  三种修法在 sigma={SIGMA_LIST[-1]} 下各丢掉多少梯度信号")
sg = SIGMA_LIST[-1]
rt = rows[sg]
TIS_C, ICEPOP_LO, ICEPOP_HI = 2.0, 0.5, 2.0
ice = (rt >= ICEPOP_LO) & (rt <= ICEPOP_HI)     # IcePop 的双边 mask

print(f"\n  {'修法':<26}{'保留梯度的 token':>18}")
print("  " + "-" * 46)
print(f"  {'不修正':<26}{1.0:>18.2%}")
print(f"  {'TIS (IS 上界 C=2)':<26}{1.0:>18.2%}")
print(f"  {'IcePop (双边 0.5~2)':<26}{ice.mean():>18.2%}")
print(f"  {'CISPO (截断+stop-grad)':<26}{1.0:>18.2%}")

print(f"\n  各自怎么处理「越界」的 token：")
print(f"    不修正    → 全部保留，但权重失真（训练在一个你不知道的 off-policy 状态下跑）")
print(f"    TIS       → 保留，把权重压到上界（软修正）")
print(f"    IcePop    → 越界 token 的梯度直接归零（硬修正，丢 {1-ice.mean():.1%}）")
print(f"    CISPO     → 全部保留，截断权重且对权重 stop-gradient")
print(f"    GSPO      → 不动 token，改在序列级算比率（治本，见实验二）")

print(f"\n  → MiniMax 与 ScaleRL 选 CISPO 而非 IcePop 的理由：长 CoT 里被判")
print(f"    「不可信」的 token 往往集中在推理的关键转折处，直接丢弃等于")
print(f"    丢掉最想要的那些梯度。代价是它仍然依赖一个合理的截断阈值。")

print("\n" + "=" * 72)
print("注：这是数值示意，不是任何框架的实测。用来建立「修法之间差在哪」的直觉。")
print("=" * 72)
