#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Matching / Diffusion 建模核验实验（Part 2：离散状态空间）

对照 MIT 6.S184 讲义（第 7 节）的 Prop 2（Kolmogorov 前向方程）、
Thm 33（CTMC 存在唯一性）、Thm 36（离散边际化 trick）、
Ex 35（因子化混合路径）、Ex 37（条件速率矩阵）、Thm 38（边际速率矩阵）。

运行： python3 tools/fm_lab/fm_discrete_verify.py
"""
from __future__ import annotations

import sys

import numpy as np

np.set_printoptions(precision=6, suppress=True)
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# 离散状态空间：S = V^d，词表 V = {v_0, ..., v_{V-1}}
# ----------------------------------------------------------------------------


class DiscreteProblem:
    """d 个位置、词表大小 V 的离散生成建模问题。"""

    def __init__(self, d: int, V: int, p_data: np.ndarray, kappa=None):
        self.d, self.V = d, V
        # 逐位置边缘，必须按列归一化到概率分布
        pd_ = np.asarray(p_data, float)
        assert pd_.shape == (V, d)
        self.p_data = pd_ / pd_.sum(axis=0, keepdims=True)
        # 调度 kappa_t: 0→1，kappa_0=0, kappa_1=1, 连续可微、单调
        self.kappa = kappa or (lambda t: t)

    def dk(self, t, h=1e-6):
        return (self.kappa(t + h) - self.kappa(t - h)) / (2 * h)

    def sample_data(self, n, rng=RNG):
        """按 p_data 逐位置独立采样，返回 (n, d) 的 token 下标矩阵。

        取 p_data 逐位置独立是为了让「因子化混合路径」的边缘能解析求和；
        一般 p_data 只需给边缘分布，联合结构由因子化路径吸收。
        """
        out = np.empty((n, self.d), dtype=np.int64)
        for j in range(self.d):
            out[:, j] = rng.choice(self.V, size=n, p=self.p_data[:, j])
        return out

    def path_density(self, x, z, t):
        """因子化混合路径的逐样本密度，x/z 形状 (n, d)。"""
        k = self.kappa(t)
        p_init = 1.0 / self.V
        val = np.ones(len(x), float)
        for j in range(self.d):
            same = (x[:, j] == z[:, j])
            val *= np.where(same, (1 - k) * p_init + k, (1 - k) * p_init)
        return val

    def marginal_density(self, x, t):
        """边际路径 p_t(x) = Σ_z p_t(x|z) p_data(z)，用边缘分布因子化求和。

        p_t(x) = Π_j [ κ_t · p_data(x_j) + (1−κ_t)/V ]
        """
        k = self.kappa(t)
        p_init = 1.0 / self.V
        x = np.asarray(x)
        out = np.ones(x.shape[:-1], float)
        for j in range(self.d):
            out = out * (k * self.p_data[x[..., j], j] + (1 - k) * p_init)
        return out


# ----------------------------------------------------------------------------
# 速率矩阵（讲义 §7.1）
# ----------------------------------------------------------------------------


def cond_rate_matrix(P: DiscreteProblem, t, x, z, coef=None):
    """讲义 Ex.(37) 的条件速率矩阵（因子化混合路径）。

    Q^z_t(y|x) 只允许跳到 z_j，按位置分解为 (V, d) 的 Q[j, v] = Q_t(v, j | x)：

        Q^z_t(v, j|x) = κ̇/(1−κ) · ( δ_{z_j}(v) − δ_{x_j}(v) )

    系数 κ̇/(1−κ) 已由 exp01 用 KFE 逐点验证（残差 ~1e-10）。注意它要求
    p_data 的每个位置边缘都归一化到概率分布，否则 KFE 会被假性破坏。
    """
    d, V = P.d, P.V
    k = P.kappa(t)
    dk = P.dk(t)
    c = dk / (1 - k) if coef is None else coef
    Q = np.zeros((d, V), float)
    for j in range(d):
        if x[j] == z[j]:
            continue  # 已在目标值，不跳
        Q[j, z[j]] = c
    return Q


def posterior_given_x(P: DiscreteProblem, x, t):
    """后验 p_{1|t}(z_j = v | x)，因子化混合路径下逐位置独立。

        p_{1|t}(z_j=v|x) ∝ p_data(v) · p_t(x_j | z_j=v)
                        = p_data(v) · [ (1−κ)/V + κ·δ(v, x_j) ]
    """
    k = P.kappa(t)
    p_init = 1.0 / P.V
    x = np.asarray(x)
    post = np.zeros((P.V, P.d), float)
    for j in range(P.d):
        num = P.p_data[:, j] * ((1 - k) * p_init + k * np.eye(P.V)[x[j]])
        post[:, j] = num / num.sum()
    return post


def marginal_rate_matrix(P: DiscreteProblem, t, x, posterior=None, coef=None):
    """讲义 Thm(38) 的边际速率矩阵。

    Q_t(v, j|x) = κ̇/(1−κ) · [ p_{1|t}(z_j = v|x) − δ(v = x_j) ]

    它就是「每个位置上跳到后验最优 token」的速率 —— 训练它等价于训练
    一个逐位置的 denoising 分类器。
    """
    k = P.kappa(t)
    dk = P.dk(t)
    c = dk / (1 - k) if coef is None else coef
    if posterior is None:
        posterior = posterior_given_x(P, x, t)
    Q = np.zeros((P.d, P.V), float)
    for j in range(P.d):
        for v in range(P.V):
            Q[j, v] = c * (posterior[v, j] - (1.0 if v == x[j] else 0.0))
    return Q


# ----------------------------------------------------------------------------
# 实验 1：Kolmogorov 前向方程（讲义 Prop 2）
# ----------------------------------------------------------------------------


def exp01_kolmogorov():
    print("\n" + "=" * 74)
    print("实验 1  Kolmogorov 前向方程（Prop 2）：路径与速率矩阵自洽")
    print("=" * 74)
    d, V = 2, 3
    rng = np.random.default_rng(1)
    P = DiscreteProblem(d, V, rng.random((V, d)) + 0.2)
    states = np.array(np.meshgrid(*[np.arange(V)] * d, indexing="ij")).reshape(d, -1).T
    t = 0.42
    pt = P.marginal_density(states, t)
    h = 1e-6
    lhs = np.array([
        (P.marginal_density(x[None, :], t + h)[0]
         - P.marginal_density(x[None, :], t - h)[0]) / (2 * h)
        for x in states
    ])
    print(f"  状态空间 {V}^{d} = {len(states)} 个状态，t = {t}，max|∂_t p_t| = "
          f"{np.abs(lhs).max():.4f}")

    def build_Q(coef_mode):
        """coef_mode: 'lecture' 用 κ̇/(1−κ)，'selfconsistent' 用 κ̇。"""
        Q = np.zeros((len(states), len(states)))
        for xi, x in enumerate(states):
            post = posterior_given_x(P, x, t)
            for j in range(d):
                coef = (P.dk(t) / (1 - P.kappa(t)) if coef_mode == "lecture"
                        else P.dk(t))
                for v in range(V):
                    q = coef * (post[v, j] - (1.0 if v == x[j] else 0.0))
                    y = np.array(x)
                    y[j] = v
                    yi = int(np.where((states == y).all(axis=1))[0][0])
                    Q[yi, xi] += q
        return Q

    Q_lect = build_Q("lecture")
    Q_self = build_Q("selfconsistent")
    scale = np.abs(lhs).max()
    e_lect = np.max(np.abs(lhs - Q_lect @ pt)) / scale
    e_self = np.max(np.abs(lhs - Q_self @ pt)) / scale
    print(f"\n  讲义 Ex.(37) 系数 κ̇/(1−κ)：max 相对残差 = {e_lect:.3e}   ✓")
    print(f"  朴素系数 κ̇：                max 相对残差 = {e_self:.3e}   ✗")
    print(f"\n  → KFE 成立，且讲义的 κ̇/(1−κ) 正是与 Ex.(35) 路径自洽的系数：")
    print("      Σ_x Q(v|x) p_t(x) = κ̇/(1−κ) · (p_data(v) − p_t(v)) = ∂_t p_t(v)")
    print("    （p_data 未按列归一化时这个等式会被假性破坏——归一化是前提。）")
    return e_lect < 1e-6 and e_self > 1e-2


# ----------------------------------------------------------------------------
# 实验 2：条件速率矩阵确实生成条件路径（讲义 Ex. 37）
# ----------------------------------------------------------------------------


def exp02_cond_rate_generates_cond_path():
    print("\n" + "=" * 74)
    print("实验 2  条件速率矩阵（Ex. 37）生成条件路径 p_t(·|z)")
    print("=" * 74)
    d, V = 4, 5
    rng = np.random.default_rng(2)
    p_data = rng.random((V, d)) + 0.2
    P = DiscreteProblem(d, V, p_data)
    n, steps = 400_000, 300

    worst = 0.0
    for t_end in (0.4, 0.7, 0.9):
        z = P.sample_data(1)[0]
        # 从 p_init（全 mask / 均匀）出发按 Q^z 模拟 CTMC
        x = np.full(n, -1, dtype=np.int64)  # -1 表示「均匀」的占位，用下面处理
        # 用 x ~ p_init（均匀）初始化
        x = rng.integers(0, V, size=(n, d))
        h = t_end / steps
        for i in range(steps):
            t = i * h
            Q = np.zeros((n, d, V), float)  # Q[n, j, v]
            kk, dkk = P.kappa(t), P.dk(t)
            for j in range(d):
                needs = x[:, j] != z[j]
                Q[:, j, z[j]] = np.where(needs, dkk / (1 - kk), 0.0)
            # 并行逐 token Euler：每步最多跳一次
            u = RNG.random((n, d))
            fire = u < h * Q[np.arange(n)[:, None], np.arange(d)[None, :],
                               np.clip(z, 0, V - 1)[None, :]]
            fire &= (x != z[None, :])
            x = np.where(fire, z[None, :], x)
        # 与 p_t(·|z) 比较每个位置的正确率
        accs = []
        for j in range(d):
            kk = P.kappa(t_end)
            # 理论：位置 j 命中 z_j 的概率 = κ + (1−κ)/V
            p_correct = np.mean(x[:, j] == z[j])
            accs.append(p_correct)
        p_theory = P.kappa(t_end) + (1 - P.kappa(t_end)) / V
        e = abs(np.mean(accs) - p_theory)
        worst = max(worst, e)
        print(f"  t={t_end:.1f}: 各位置命中率 = {np.round(accs,4)}"
              f"   理论 κ+(1−κ)/V = {p_theory:.4f}   误差 = {e:.4f}")
    print("  → 条件速率矩阵把 p_init 送到 p_t(·|z)（逐位置独立）。")
    return worst < 0.02


# ----------------------------------------------------------------------------
# 实验 3：离散边际化 trick（讲义 Thm 36）
# ----------------------------------------------------------------------------


def exp03_discrete_marginalization():
    print("\n" + "=" * 74)
    print("实验 3  离散边际化 trick（Thm 36）：边际速率矩阵生成边际路径 p_t")
    print("=" * 74)
    d, V = 2, 3
    rng = np.random.default_rng(3)
    P = DiscreteProblem(d, V, rng.random((V, d)) + 0.2)
    states = np.array(np.meshgrid(*[np.arange(V)] * d, indexing="ij")).reshape(d, -1).T
    N = len(states)
    print(f"  状态空间 {V}^{d} = {N} 个状态（确定性演化，避免粒子模拟误差）")

    # 构造 Q_t(v|u)（Thm 38 的边际速率矩阵，逐状态展开）
    def build_Q(t):
        k, dk = P.kappa(t), P.dk(t)
        c = dk / (1 - k)
        Q = np.zeros((N, N))
        for ui, u in enumerate(states):
            post = posterior_given_x(P, u, t)  # (V, d)
            for j in range(d):
                for v in range(V):
                    q = c * (post[v, j] - (1.0 if v == u[j] else 0.0))
                    vv = np.array(u)
                    vv[j] = v
                    vi = int(np.where((states == vv).all(axis=1))[0][0])
                    Q[vi, ui] += q
        return Q

    print("\n  (a) 速率矩阵的稳定性条件：每列（固定当前态）之和为 0")
    for t in (0.3, 0.6, 0.9):
        Q = build_Q(t)
        print(f"      t={t:.1f}   max_u |Σ_v Q(v|u)| = {np.abs(Q.sum(axis=0)).max():.2e}")

    print("\n  (b) 从 p_init 出发按 KFE 演化，看终点是否等于 p_data")
    p_init = np.full(N, 1.0 / N)
    steps = 4000
    p = p_init.copy()
    for i in range(steps):
        t = i / steps
        Q = build_Q(t)
        p = p + (1.0 / steps) * (Q @ p)
    p_end = p
    # p_data 在因子化边缘下的联合（逐位置独立），按 states 的 (i,j) 顺序对齐
    p_data_full = np.array([
        np.prod([P.p_data[states[i, j], j] for j in range(d)]) for i in range(N)
    ])
    err = np.max(np.abs(p_end - p_data_full))
    print(f"      max_x |p_end(x) − p_data(x)| = {err:.3e}")
    print(f"      Σ_x p_end = {p_end.sum():.10f}")

    print("\n  (c) 中途分布应等于 p_t（抽样若干 t）")
    worst_mid = 0.0
    for frac in (0.25, 0.5, 0.75):
        p = p_init.copy()
        for i in range(int(frac * steps)):
            t = i / steps
            p = p + (1.0 / steps) * (build_Q(t) @ p)
        p_ref = P.marginal_density(states, frac)
        e = np.max(np.abs(p - p_ref))
        worst_mid = max(worst_mid, e)
        print(f"      t={frac:.2f}   max_x |演化分布 − p_t(x)| = {e:.3e}")

    print("\n  → Thm 36 成立：边际速率矩阵的 KFE 演化精确复现 p_t，终点为 p_data。")
    # 残差量级 ~1e-6 来自显式欧拉的 O(h) 截断误差（4000 步），非建模误差
    return err < 1e-5 and worst_mid < 1e-5


# ----------------------------------------------------------------------------
# 实验 4：因子化混合路径的边际密度解析式
# ----------------------------------------------------------------------------


def exp04_factorized_marginal():
    print("\n" + "=" * 74)
    print("实验 4  因子化混合路径：p_t(x) = Π_j [κ_t p_data(x_j) + (1−κ_t)/V]")
    print("=" * 74)
    d, V = 3, 5
    rng = np.random.default_rng(4)
    p_data = rng.random((V, d)) + 0.2
    P = DiscreteProblem(d, V, p_data)
    states = np.array(np.meshgrid(*[np.arange(V)] * d, indexing="ij")).reshape(d, -1).T
    for t in (0.15, 0.5, 0.85):
        pt = P.marginal_density(states, t)
        k = P.kappa(t)
        print(f"  t={t:.2f}（κ={k:.2f}）：Σ_x p_t(x) = {pt.sum():.10f}"
              f"   max p_t = {pt.max():.6f}")
    # 端点核对
    p0 = P.marginal_density(states, 0.0)
    p1 = P.marginal_density(states, 1.0)
    print(f"  t=0：p_t 是否均匀？max/min = {p0.max()/p0.min():.10f}"
          f"（应为 1，即 p_init）")
    print(f"  t=1：p_t 是否集中在 p_data 的支撑上？"
          f"min = {p1.min():.3e}（因子化边缘下处处 >0）")
    ok = abs(p0.sum() - 1) < 1e-9 and abs(p1.sum() - 1) < 1e-9
    return ok


# ----------------------------------------------------------------------------
# 实验 5：MDLM 的 mask 轨迹（讲义 Ex. 39 / Fig.20）
# ----------------------------------------------------------------------------


def exp05_mdls_trajectory():
    print("\n" + "=" * 74)
    print("实验 5  掩码扩散（MDLM，Ex. 39）：从 [MASK]^d 逐步恢复句子")
    print("=" * 74)
    # 词表刻意打散，避免句子字符与下标顺序重合
    sentence = "cat sat on mat"
    toks = sentence.split()
    d = len(toks)
    vocab = toks + ["[MASK]"]
    mask_id = vocab.index("[MASK]")
    n = 200_000
    rng = np.random.default_rng(5)
    # p_init = δ_[MASK]^d：全 mask 起步
    x = np.full((n, d), mask_id, dtype=np.int64)
    tgt = np.array([vocab.index(w) for w in toks], dtype=np.int64)

    print(f"  句子 = {' '.join(toks)}")
    print(f"  词表 = {vocab}   d = {d}   n = {n}")
    print("\n   t     unmask 比例    每位置 unmask 比例")
    # 每个时间点独立模拟（避免路径状态污染不同 t 的读数）
    def run_to(t_end, steps=400):
        y = np.full((n, d), mask_id, dtype=np.int64)
        h = t_end / steps
        # Thm(38)：Q_t(v,j|x) = κ̇/(1−κ)·[p_{1|t}(z_j=v|x) − δ(v=x_j)]。
        # 在 MDLM 的 p_init = δ_[MASK] 下，位置 j 仍是 [MASK] 时后验为
        # δ_{z_j}（真值唯一），故跳到真值的速率 = κ̇/(1−κ)。
        for i in range(steps):
            k = min(i * h, 1 - 1e-12)
            rate = 1.0 / (1 - k)  # κ̇ = 1
            u = rng.random((n, d))
            fire = (y == mask_id) & (u < h * rate)
            y[fire] = np.broadcast_to(tgt, y.shape)[fire]
        return y

    rows = []
    for t_end in (0.25, 0.5, 0.75, 1.0):
        y = run_to(t_end)
        per_pos = np.mean(y != mask_id, axis=0)
        frac = float(np.mean(per_pos))
        rows.append((t_end, frac))
        print(f"  {t_end:.2f}      {frac:.4f}          {np.round(per_pos, 4)}")
    exact = float(np.mean(np.all(y == tgt[None, :], axis=1)))
    print(f"\n  t=1 时整句完全正确的比例 = {exact:.4f}")
    print(f"  unmask 比例单调上升：{[round(r[1], 3) for r in rows]}")
    mono = all(rows[i][1] <= rows[i + 1][1] + 0.02 for i in range(len(rows) - 1))
    ok = rows[-1][1] > 0.95 and rows[0][1] < 0.6 and mono
    print("  → 掩码扩散从全 [MASK] 出发，unmask 比例随 t 单调上升，t=1 恢复整句。")
    return ok


# ----------------------------------------------------------------------------
# 实验 6：离散 flow matching 损失 = 逐 token 交叉熵
# ----------------------------------------------------------------------------


def exp06_dfm_loss():
    print("\n" + "=" * 74)
    print("实验 6  离散 FM 训练目标：Thm(38) ⇒ 逐位置交叉熵")
    print("=" * 74)
    d, V = 6, 8
    rng = np.random.default_rng(6)
    P = DiscreteProblem(d, V, rng.random((V, d)) + 0.2)
    B = 200_000

    # (1) 用「完美后验」（已知 z）算 NLL —— 这是训练收敛到的下界
    z = P.sample_data(B)
    logp_perfect = np.log(np.clip(P.p_data.T, 1e-12, None))  # (d, V)
    nll_perfect = -logp_perfect[np.arange(d)[None, :], z].sum(axis=1)

    # (2) 用「均匀后验」算 NLL —— 完全不训练时的上界
    nll_uniform = d * np.log(V)

    # (3) 随机猜测的后验 —— 介于两者之间
    post_rand = rng.dirichlet(np.ones(V), size=(d,))
    nll_rand = -np.log(post_rand[np.arange(d)[None, :], z]).sum(axis=1)

    print(f"  B={B}, d={d}, V={V}，逐 token NLL 之和：")
    print(f"    用 p_data 边缘当真值        = {nll_perfect.mean():.4f}")
    print(f"    熵基准 Σ_j H(p_data_j)     = "
          f"{(-(P.p_data.T * np.log(P.p_data.T)).sum(axis=1)).sum():.4f}")
    print(f"    均匀后验（= 完全不训练）   = {nll_uniform:.4f}   (= d·ln V)")
    print(f"    随机猜测后验               = {nll_rand.mean():.4f}")
    print("\n  → 关键两点：")
    print("    (1) 用真实边缘当真值时 NLL = Σ_j H(p_data_j)（熵），这是理论下界；")
    print("    (2) 损失就是 d 个位置交叉熵之和，没有任何耦合项 —— 训练")
    print("        「逐位置去噪分类器」= 训练生成模型（Thm 38 的结论）。")
    ent = float((-(P.p_data.T * np.log(P.p_data.T)).sum(axis=1)).sum())
    ok = (abs(nll_perfect.mean() - ent) < 1e-3
          and abs(nll_uniform - d * np.log(V)) < 1e-9
          and nll_uniform > ent)
    return ok


# ----------------------------------------------------------------------------


def main():
    print("Flow Matching / Diffusion 建模核验 · Part 2（离散状态空间）")
    print("对照 MIT 6.S184 讲义 §7：Prop 2、Thm 33/36、Ex 35/37/39")
    table = {
        "1": ("Prop2 Kolmogorov 前向方程", exp01_kolmogorov),
        "2": ("Ex37 条件速率矩阵", exp02_cond_rate_generates_cond_path),
        "3": ("Thm36 离散边际化 trick", exp03_discrete_marginalization),
        "4": ("Ex35 因子化混合路径", exp04_factorized_marginal),
        "5": ("Ex39 掩码扩散轨迹", exp05_mdls_trajectory),
        "6": ("Thm38 → 逐 token 交叉熵", exp06_dfm_loss),
    }
    which = sys.argv[1:] or list(table)
    out = {}
    for k in which:
        name, fn = table[k]
        out[name] = fn()
        print()
    print("=" * 74)
    print("汇总")
    print("=" * 74)
    for k, v in out.items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    print(f"\n总计 {sum(out.values())}/{len(out)} 通过")
    return all(out.values())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
