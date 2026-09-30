"""
RL 框架里的典型变量：逐行对照源码的数值验证（纯 numpy，无 torch）。

配套文章：_posts/2026-09-30-rl-variables-in-frameworks.md

每个实现的公式都逐行对照真实源码（文件与行号写在注释里），
用有限差分（中心差分）数值微分求出解析梯度，验证源码里
stop-gradient / clamp 到底造成了什么差别。

实验 A：三种 policy loss 的梯度路径（vanilla / cispo / gspo）
实验 B：四种 advantage 估计器的组内归一化差异（grpo / rloo / rf++-baseline / gae 的白化）
实验 C：KL 的两种落点（注入 reward vs 加在 loss）
实验 D：agg_loss 四种聚合口径的差异

运行：
  /Users/congyuan/Software/miniconda3/bin/python3 tools/rl_var_grad_lab.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def section(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


# ======================================================================
# 实验 A：三种 policy loss 的梯度路径
# ======================================================================
section("实验 A：vanilla / CISPO / GSPO 的梯度到底落在哪些 token 上")

# 构造一条序列：6 个 token，人为做出三种情形
#   t=0,1  ratio 正常（≈1）
#   t=2    ratio 偏高（被 clip_high 截部分）
#   t=3    ratio 偏高很多（一定被裁剪）
#   t=4    ratio 偏低很多（一定被低裁）
#   t=5    正常
OLD_LOGP = np.array([-1.00, -1.00, -1.00, -1.00, -1.00, -1.00])
NEW_LOGP = np.array([-1.02, -0.99, -0.70, -0.10, -2.20, -1.01])   # 训练时会变
ADV = np.array([1.0, -1.0, 1.0, 1.0, 1.0, -1.0])                  # 有正有负
MASK = np.ones(6)
EPS_LOW, EPS_HIGH = 0.2, 0.28     # DAPO 风格的偏上裁剪
CISPO_C = 1.0                     # TRL 里 cispo 只做单边 max 截断
T = len(OLD_LOGP)


def ratio_of(logp):
    """verl core_algos.py:1210-1213 —— 先算 log 比再 exp，且 clamp 到 [-20,20]"""
    neg_kl = np.clip(logp - OLD_LOGP, -20.0, 20.0)
    return np.exp(neg_kl)


# --- 三种 loss 的实现（forward 值），逐行对照源码 ---
# --- 三种 loss 的实现，逐行对照源码 ---
# 注意方法论：源码里的 .detach() / sg[] 意味着「前向取该值、反向不回传」。
# 纯数值微分只能测前向，所以对含 stop-gradient 的表达式必须用**双变量**建模：
#   令 g(x_var, x_fixed) = x_var - x_fixed  → 前向(二者相等时)=0，对 x_var 偏导 = 1
# 把 sg[·] 的部分用 x_fixed 计算，就精确复现了 stop-gradient 的行为。

def loss_vanilla(x_var, x_fixed):
    """verl core_algos.py:1210-1243 (compute_policy_loss_vanilla)，无 detach"""
    r = ratio_of(x_var)
    l1 = -ADV * r
    l2 = -ADV * np.clip(r, 1 - EPS_LOW, 1 + EPS_HIGH)
    clip_part = np.maximum(l1, l2)
    dual = -ADV * 3.0
    per_token = np.where(ADV < 0, np.minimum(dual, clip_part), clip_part)
    return (per_token * MASK).sum() / MASK.sum()


def loss_cispo(x_var, x_fixed):
    """verl core_algos.py:1746-1758
    L = -sg(clamp(ratio)) * A * log_prob，梯度只走 log_prob"""
    clipped_sg = np.clip(ratio_of(x_fixed), 1 - EPS_LOW, 1 + EPS_HIGH)   # sg 部分
    per_token = -clipped_sg * ADV * x_var
    return (per_token * MASK).sum() / MASK.sum()


def loss_gspo(x_var, x_fixed, eps_low=EPS_LOW, eps_high=EPS_HIGH):
    """verl core_algos.py:1290-1314
    log s_{i,t} = sg[log s_i] + log_prob - sg[log_prob]
      → 前向值 = sg[log s_i]（整条序列同一个标量）
      → 对 log_prob_t 的偏导 = 1（逐 token 都能回传）
    用双变量写：sg 部分取自 x_fixed，token 部分取自 x_var"""
    log_ratio_fixed = np.log(ratio_of(x_fixed))
    seq_log_ratio = (log_ratio_fixed * MASK).sum() / MASK.sum()          # sg[log s_i]
    log_s = seq_log_ratio + (x_var - x_fixed)                            # 前向 = seq_log_ratio
    log_s = np.clip(log_s, None, 10.0)                                   # 1298
    s1 = np.exp(log_s)
    s2 = np.exp(np.clip(log_s, np.log(1 - eps_low), np.log(1 + eps_high)))
    per_token = np.maximum(-ADV * s1, -ADV * s2)                         # 1305
    return (per_token * MASK).sum() / MASK.sum()


def numeric_grad(fn, x, h=1e-6):
    """中心差分。fn 取 (x_var, x_fixed)，这里 x_fixed 固定为原始 x"""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (fn(xp, x) - fn(xm, x)) / (2 * h)
    return g


print(f"\n设定：{T} 个 token，clip=[1-{EPS_LOW}, 1+{EPS_HIGH}]"
      f"（DAPO 风格偏上裁剪），advantage 有正有负\n")
print(f"  {'token':>6}{'old_logp':>10}{'new_logp':>10}{'ratio':>9}{'adv':>7}"
      f"{'是否超界':>10}")
print("  " + "-" * 54)
r0 = ratio_of(NEW_LOGP)
for t in range(T):
    over = "是" if (r0[t] < 1 - EPS_LOW or r0[t] > 1 + EPS_HIGH) else "否"
    print(f"  {t:>6}{OLD_LOGP[t]:>10.2f}{NEW_LOGP[t]:>10.2f}{r0[t]:>9.3f}"
          f"{ADV[t]:>7.1f}{over:>10}")

g_v = numeric_grad(loss_vanilla, NEW_LOGP)
g_c = numeric_grad(loss_cispo, NEW_LOGP)
g_g = numeric_grad(loss_gspo, NEW_LOGP)

print("\n  数值微分得到的 dLoss/d(new_logp)：0 表示该 token 拿不到梯度")
print(f"\n  {'token':>6}{'ratio':>9}{'vanilla':>12}{'CISPO':>12}{'GSPO':>12}")
print("  " + "-" * 52)
for t in range(T):
    print(f"  {t:>6}{r0[t]:>9.3f}{g_v[t]:>12.4f}{g_c[t]:>12.4f}{g_g[t]:>12.4f}")

n_zero_v = int(np.sum(np.abs(g_v) < 1e-9))
n_zero_c = int(np.sum(np.abs(g_c) < 1e-9))
n_zero_g = int(np.sum(np.abs(g_g) < 1e-9))
print("\n  --- 读法 ---")
print(f"  vanilla：{n_zero_v}/{T} 个 token 梯度为 0 —— 被裁剪的 token 直接不贡献梯度")
print(f"  CISPO  ：{n_zero_c}/{T} 个 token 梯度为 0 —— 梯度走 log_prob，裁剪只改权重")
print(f"           t=2 的梯度 = -clip(1.35→1.28) × A / N = {-1.28*1/6:.4f}，与实测一致")
print(f"  GSPO   ：{n_zero_g}/{T} 个 token 梯度为 0，且**所有非零梯度完全相同**"
      f"（{g_g[0]:.4f}）")

# GSPO 的序列级开关：把 seq-ratio 推到界内/界外
print("\n  --- GSPO 的「整条序列一起」是什么意思 ---")
print(f"  {'场景':<24}{'seq-ratio':>11}{'越界?':>7}{'A>0 有梯度':>12}{'A<0 有梯度':>12}")
print("  " + "-" * 68)
pos_idx = ADV > 0
neg_idx = ADV < 0
for delta in (0.0, np.log(1.20), np.log(1.60), np.log(2.60)):
    x = OLD_LOGP + delta
    seq_r = np.exp((np.log(ratio_of(x)) * MASK).sum() / MASK.sum())
    gg = numeric_grad(loss_gspo, x)
    nz = np.abs(gg) > 1e-9
    print(f"  {'logp 整体偏移 ' + f'{delta:+.3f}':<24}{seq_r:>11.3f}"
          f"{('是' if seq_r > 1 + EPS_HIGH else '否'):>7}"
          f"{f'{nz[pos_idx].sum()}/{pos_idx.sum()}':>12}"
          f"{f'{nz[neg_idx].sum()}/{neg_idx.sum()}':>12}")

print("\n  → 关键不在「有没有梯度」，而在**判断的粒度**：")
print("     · vanilla：每个 token 各自判断自己的 ratio 是否越界（逐 token 一刀）")
print("     · GSPO   ：整条序列共用一个 seq-ratio 做判断（整条轨迹一刀）")
print("     阈值一变，被切掉的是「整条序列里的一类 token」，而不是零散的几个 token。")
print("     边界情况：ratio 在界内时，A>0 与 A<0 的 token 都保留梯度；")
print("     越界后 maximum 的双分支让 A>0 的一类整体失去梯度，A<0 的仍然保留。")

# ======================================================================
# 实验 B：advantage 估计器的组内归一化
# ======================================================================
section("实验 B：四种 advantage 估计器，同一份 rollout 算出不同的数")

# 2 个 prompt，每个 4 条 rollout（G=4），二值奖励
GROUP = np.array([
    [1.0, 1.0, 0.0, 0.0],     # prompt 0：2/4 正确
    [1.0, 0.0, 0.0, 0.0],     # prompt 1：1/4 正确
])
FLAT = GROUP.reshape(-1)
G_IDX = np.repeat([0, 1], 4)

print(f"\n  奖励矩阵（2 个 prompt × G=4 条 rollout）：\n")
for i, row in enumerate(GROUP):
    print(f"    prompt {i}: {row}   组均 {row.mean():.2f}  "
          f"组 std(unbiased) {row.std(ddof=1):.4f}")


def adv_grpo(scores, ddof=1):
    """verl core_algos.py:303-330 —— 减组均，除组内 std（torch.std 默认 unbiased）"""
    out = np.zeros_like(scores)
    for i in range(len(scores)):
        m = G_IDX == G_IDX[i]
        mu = scores[m].mean()
        sd = scores[m].std(ddof=ddof)
        out[i] = (scores[i] - mu) / (sd + 1e-6)
    return out


def adv_drgrpo(scores):
    """verl core_algos.py:326-327 —— norm_adv_by_std_in_grpo=False（Dr.GRPO）"""
    return np.array([scores[i] - scores[G_IDX == G_IDX[i]].mean()
                     for i in range(len(scores))])


def adv_rloo(scores):
    """verl core_algos.py:517-522 —— 留一法系数 n/(n-1)"""
    out = np.zeros_like(scores)
    for i in range(len(scores)):
        m = G_IDX == G_IDX[i]
        n = m.sum()
        if n > 1:
            out[i] = scores[i] * n / (n - 1) - scores[m].mean() * n / (n - 1)
        else:
            out[i] = scores[i]
    return out


def adv_rfpp_baseline(scores, whiten=True):
    """verl core_algos.py:467-471 —— 减组均后，再做 batch 级白化"""
    base = np.array([scores[i] - scores[G_IDX == G_IDX[i]].mean()
                     for i in range(len(scores))])
    if whiten:
        base = (base - base.mean()) / (base.std(ddof=0) + 1e-6)
    return base


print(f"\n  {'prompt':>7}{'rollout奖励':>14}{'GRPO(含std)':>15}"
      f"{'Dr.GRPO':>12}{'RLOO':>12}{'RF++-baseline':>18}")
print("  " + "-" * 78)
a_g, a_d, a_r, a_f = adv_grpo(FLAT), adv_drgrpo(FLAT), adv_rloo(FLAT), adv_rfpp_baseline(FLAT)
for i in range(len(FLAT)):
    print(f"  {G_IDX[i]:>7}{FLAT[i]:>14.1f}{a_g[i]:>15.4f}{a_d[i]:>12.4f}"
          f"{a_r[i]:>12.4f}{a_f[i]:>18.4f}")

print("\n  --- 差异来源 ---")
print(f"  GRPO         : (r - 组均) / 组内std，G=4 时 torch.std 默认 unbiased（除以 n-1=3）")
print(f"  Dr.GRPO      : 只减组均，不除 std —— 难组不再被放大（见文章 7.4 节实验一）")
print(f"  RLOO         : n/(n-1) × (r - 组均)，留一法 baseline，n=4 时放大 1.3333 倍")
print(f"                 → 第 0 组第一条：0.5 × 1.3333 = {0.5*4/3:.4f}，实测 {a_r[0]:.4f}")
print(f"  RF++-baseline: 组内减均值后再做 batch 级白化 —— 是唯一跨组重新标准化的")
print(f"                 → 组间均值被抹平，组 0 与组 1 的难度差消失")

# unbiased vs biased 的差别（源码细节）
print(f"\n  源码细节：verl 用 torch.std(scores_tensor)（core_algos.py:320）")
print(f"    torch.std 默认 unbiased=True（ddof=1），numpy 默认 ddof=0")
print(f"    G=4 时两者相差 {np.sqrt(4/3):.4f} 倍 —— 直接改变 advantage 的尺度")
print(f"\n    {'分组':>6}{'std(ddof=1)':>14}{'std(ddof=0)':>14}{'GRPO幅度之比':>16}")
print("    " + "-" * 50)
for i, row in enumerate(GROUP):
    s1, s0 = row.std(ddof=1), row.std(ddof=0)
    print(f"    {i:>6}{s1:>14.4f}{s0:>14.4f}{s0/s1:>16.4f}")


# ======================================================================
# 实验 C：KL 的两种落点
# ======================================================================
section("实验 C：KL 惩罚加在哪 —— reward 里（verl）还是 loss 里（TRL）")

BETA = 0.04
T2 = 5
old_lp = np.array([-1.20, -0.80, -1.50, -0.60, -1.10])
ref_lp = np.array([-1.10, -0.95, -1.30, -0.75, -1.05])
score = np.array([0, 0, 1, 0, 0], dtype=float)     # 稀疏 outcome reward

print(f"\n  beta = {BETA}\n")
print(f"  {'token':>6}{'old_logp':>10}{'ref_logp':>10}{'log差':>9}"
      f"{'k1':>9}{'k3':>9}")
print("  " + "-" * 54)
log_diff = old_lp - ref_lp
k1 = log_diff                                              # verl: kl_penalty "k1"
k3 = np.exp(ref_lp - old_lp) - (ref_lp - old_lp) - 1        # TRL: k3 形式
for t in range(T2):
    print(f"  {t:>6}{old_lp[t]:>10.2f}{ref_lp[t]:>10.2f}{log_diff[t]:>9.4f}"
          f"{k1[t]:>9.4f}{k3[t]:>9.4f}")

print(f"\n  verl 落点（core_algos.py:1021-1022 compute_rewards）：")
r_verl = score - k1 * BETA
print(f"    reward_t = token_score - (old_logp - ref_logp) * beta")
print(f"    逐 token reward = {r_verl}")
print(f"    → k1 是**有符号**的：old 比 ref 概率高时 KL 项为正、reward 被扣；反之 reward 被加")
print(f"    → 稀疏场景下（只有最后一个 token 有分数），KL 变成逐 token 的密集信号")
print(f"    → 这一步之后，advantage 的估计器看到的是**已经扣过 KL 的 reward**")

r_trl = score.copy()
print(f"\n  TRL 落点（grpo_trainer.py:2333-2385）：")
print(f"    per_token_loss += beta * (exp(ref-logp) - (ref-logp) - 1)")
print(f"    → 用的是 k3 形式，**恒非负**，直接加在 loss 上，不进入 reward")
print(f"    → advantage 估计器看到的仍是纯奖励")

print(f"\n  两种落点的实质差别：")
print(f"    · 加在 reward 里：KL 会经过 advantage 估计器（组内归一/白化），")
print(f"      相当于先合并再标准化 —— 组内做差时，同一 prompt 的 KL 差异会被保留")
print(f"    · 加在 loss 里：KL 是独立的逐 token 惩罚，不参与 advantage 归一化")
print(f"    · 数值上：k1 可用，k3 恒非负，混用两个口径会让 beta 不可比")


# ======================================================================
# 实验 D：agg_loss 四种口径
# ======================================================================
section("实验 D：agg_loss 四种聚合口径（verl core_algos.py:1055-1076）")

rng = np.random.default_rng(7)
B, Tm = 4, np.array([10, 6, 3, 8])      # 4 条序列，长度不同
loss_mat = np.zeros((B, Tm.max()))
mask = np.zeros_like(loss_mat)
for i in range(B):
    loss_mat[i, : Tm[i]] = rng.normal(0.0, 1.0, Tm[i])
    mask[i, : Tm[i]] = 1.0

print(f"\n  4 条序列，长度分别是 {[int(x) for x in Tm]}（总有效 token = {int(mask.sum())}）\n")
n_tok = mask.sum()
GP = 1

m_token_mean = (loss_mat * mask).sum() / n_tok * GP
seq_sum = (loss_mat * mask).sum(-1)
seq_mask = (mask.sum(-1) > 0).astype(float)
m_seq_mean_token_sum = (seq_sum * seq_mask).sum() / seq_mask.sum() * GP
seq_tok_cnt = mask.sum(-1)
seq_mean = seq_sum / (seq_tok_cnt + 1e-8)
m_seq_mean_token_mean = (seq_mean * seq_mask).sum() / seq_mask.sum() * GP
m_seq_mean_token_sum_norm = seq_sum.sum() / mask.shape[-1]

print(f"  {'口径':<26}{'数值':>10}   含义")
print("  " + "-" * 78)
print(f"  {'token-mean':<26}{m_token_mean:>10.4f}   除以全局有效 token 数")
print(f"  {'seq-mean-token-sum':<26}{m_seq_mean_token_sum:>10.4f}   先按序列求和，再按序列平均")
print(f"  {'seq-mean-token-mean':<26}{m_seq_mean_token_mean:>10.4f}   序列内先平均，再序列间平均")
print(f"  {'seq-mean-token-sum-norm':<26}{m_seq_mean_token_sum_norm:>10.4f}   按固定长度归一（与真实长度无关）")

print(f"\n  长序列（len=10）与短序列（len=3）在两个口径下的权重：")
print(f"    token-mean            ：每条序列的权重 ∝ 它的长度（长序列权重 10，短序列 3）")
print(f"    seq-mean-token-sum    ：每条序列权重相同（都占 1/4），长序列总贡献反而更大")
print(f"    seq-mean-token-mean   ：每条序列权重相同，且长度已被除掉")
print(f"    → DAPO 主张 token-mean（长序列的每个 token 一视同仁），")
print(f"      原版 GRPO 是 seq-mean-token-sum（长回答整体被放大）")

print("\n" + "=" * 76)
print("注：以上公式逐行对照 verl / TRL 源码，数值为自己算的示意。")
print("     不涉及任何框架的实测吞吐或收敛结论。")
print("=" * 76)
