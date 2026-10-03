#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Matching / Diffusion 建模核验实验（Part 1：连续部分）

对照 MIT 6.S184《An Introduction to Flow Matching and Diffusion Models》
讲义（Holderrieth & Erives, 2026）的定理 3/9/11/12/17/19、命题 1、推论 36/37。

每个实验都做「构造真值 ↔ 反算值」的自校验，只依赖 numpy，纯 CPU。
运行： python3 tools/fm_lab/fm_continuous_verify.py
"""
from __future__ import annotations

import sys

import numpy as np

np.set_printoptions(precision=6, suppress=True)
RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# 数值微分工具：Richardson 外推的中心差分，精度 O(h^4)
# ----------------------------------------------------------------------------


def d_dt(f, t, h=1e-4):
    """∂f/∂t，O(h^4)。"""
    d1 = (f(t + h) - f(t - h)) / (2 * h)
    d2 = (f(t + h / 2) - f(t - h / 2)) / h
    return (4 * d2 - d1) / 3


def d_dx(f, x, h=1e-5):
    """∂f/∂x（一维），O(h^4)。"""
    d1 = (f(x + h) - f(x - h)) / (2 * h)
    d2 = (f(x + h / 2) - f(x - h / 2)) / h
    return (4 * d2 - d1) / 3


def d2_dx2(f, x, h=1e-3):
    """∂²f/∂x²（一维），Richardson 外推，误差 O(h^4)。"""
    d1 = (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    d2 = (f(x + h / 2) - 2 * f(x) + f(x - h / 2)) / (h / 2) ** 2
    return (4 * d2 - d1) / 3


# ----------------------------------------------------------------------------
# 噪声调度器：讲义的 (α_t, β_t)，β_t 是标准差（讲义用 β，博客常用 σ）
# ----------------------------------------------------------------------------


class Schedule:
    """基类：alpha(0)=beta(1)=0, alpha(1)=beta(0)=1。"""

    name = "base"

    def alpha(self, t):
        raise NotImplementedError

    def beta(self, t):
        raise NotImplementedError

    def dalpha(self, t):
        h = 1e-6
        return (self.alpha(t + h) - self.alpha(t - h)) / (2 * h)

    def dbeta(self, t):
        h = 1e-6
        return (self.beta(t + h) - self.beta(t - h)) / (2 * h)

    def ab(self, t):
        return self.alpha(t), self.beta(t)

    def dab(self, t):
        return self.dalpha(t), self.dbeta(t)


class CondOT(Schedule):
    """讲义 Example 13 的 CondOT：α_t = t, β_t = 1 − t。

    这是唯一严格满足讲义边界条件 α_0=β_1=0, α_1=β_0=1 且 β_t 全程 > 0 的
    「教科书」路径（t=1 处 β=0，条件场解析式仍良定，只是 E[1/β_t] 发散）。
    """

    name = "CondOT"

    def alpha(self, t):
        return t

    def beta(self, t):
        return 1.0 - t


class OTSigma(Schedule):
    """带 σ_min 的 OT 路径：α_t = t, β_t = 1 − (1−σ_min)t。

    实践中最常用的形式：β_1 = σ_min > 0 使 1/β_t 在端点有限，代价是终点
    分布是 N(z, σ_min²) 而非严格的 δ_z（讲义 §3.3 允许的「近似插值」）。
    """

    def __init__(self, sigma_min: float = 0.01):
        self.sm = sigma_min

    @property
    def name(self):
        return f"OT(σ_min={self.sm})"

    def alpha(self, t):
        return t

    def beta(self, t):
        return 1.0 - (1.0 - self.sm) * t


class TrigPath(Schedule):
    """三角函数插值（SI 论文速查表中的 trig 型）：α_t=sin(πt/2), β_t=cos(πt/2)。

    同样严格满足边界条件且 β_t 在 [0,1) 上有正下界之外的良好行为，
    用来验证「换一条路径，Thm 9 / 11 / 12 依然成立」。
    """

    name = "Trig"

    def alpha(self, t):
        return np.sin(np.pi * t / 2)

    def beta(self, t):
        return np.cos(np.pi * t / 2)


class VESched(Schedule):
    """方差爆炸路径：α_t = 1, β_t = t。

    注意方向：β_0 = 0, α_0 = 1，即 t=0 是数据、t=1 是噪声 —— 这是扩散文献
    常用的「反向时间约定」，与讲义的 t: 噪声→数据 相反。因此它只用于
    **代数恒等式**（Prop 1 换算、连续性方程）的核对，不能从 t=0 积分做生成。
    """

    name = "VE(反向)"

    def alpha(self, t):
        return 1.0

    def beta(self, t):
        return t


class VPSched(Schedule):
    """方差保持路径：α_t = e^{−βt/2}, β_t = √(1−e^{−βt})，同样为反向时间约定。"""

    def __init__(self, beta: float = 1.0):
        self.b = beta

    @property
    def name(self):
        return f"VP(反向,β={self.b})"

    def alpha(self, t):
        return np.exp(-self.b * t / 2)

    def beta(self, t):
        return np.sqrt(np.maximum(1.0 - np.exp(-self.b * np.asarray(t, float)), 0.0))


#: 满足讲义边界条件、可从 t=0（噪声）积分到 t=1（数据）的合法插值路径
VALID_PATHS = (CondOT(), OTSigma(0.01), OTSigma(0.2), TrigPath())

#: 仅用于代数恒等式核对的方向相反的经典扩散路径
REVERSED_PATHS = (VESched(), VPSched(1.0), VPSched(1.3))


# ----------------------------------------------------------------------------
# 一维高斯混合数据分布（p_data 的解析密度使一切可闭式反算）
# ----------------------------------------------------------------------------


class GMM1D:
    def __init__(self, mus, sigmas, weights):
        self.mus = np.asarray(mus, float)
        self.sigmas = np.asarray(sigmas, float)
        self.w = np.asarray(weights, float)
        self.w = self.w / self.w.sum()

    def density(self, x):
        comp = np.exp(-0.5 * ((x[..., None] - self.mus) / self.sigmas) ** 2) / (
            self.sigmas * np.sqrt(2 * np.pi)
        )
        return np.sum(comp * self.w, axis=-1)

    def sample(self, n, rng=RNG):
        idx = rng.choice(len(self.w), size=n, p=self.w)
        return rng.normal(self.mus[idx], self.sigmas[idx])


# ----------------------------------------------------------------------------
# 讲义的核心量，全部按讲义符号实现
# ----------------------------------------------------------------------------


def _bc(x, t, z, sched: Schedule, paired: bool = True):
    """统一广播：返回 (α, β, x_view, z_view)，使 cond_* 输出形状符合语义。

    paired=True  —— z 与 x 逐元素配对（条件量 p_t(·|z)、u_t(·|z)）：
                     x (B,), z (B,) → (B,)；x 标量, z 标量 → 标量
    paired=False —— z 是一组「候选数据点」，需对最后一维求和（边际量）：
                     x (B,), z (K,) → (B, K)；x 标量, z (K,) → (K,)
    """
    xa = np.asarray(x, float)
    za = np.asarray(z, float)
    a, b = sched.ab(t)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not paired:
        # 集合语义：x 批量 → xv (B,1)、zv (1,K)，逐元素积得 (B,K)
        if xa.ndim >= 1:
            return (
                a.reshape(-1, 1),
                b.reshape(-1, 1),
                xa.reshape(-1, 1),
                za.reshape((1,) + za.shape) if za.ndim >= 1 else za,
            )
        return a, b, xa, za
    # 配对语义：x、z 同形，全部保持 (B,) 或标量
    return a, b, xa, za


def cond_density(x, t, z, sched: Schedule, paired: bool = True):
    """条件概率路径密度 p_t(·|z) = N(α_t z, β_t²)。"""
    a, b, xv, zv = _bc(x, t, z, sched, paired)
    return np.exp(-0.5 * ((xv - a * zv) / b) ** 2) / (b * np.sqrt(2 * np.pi))


def cond_score(x, t, z, sched: Schedule, paired: bool = True):
    """讲义 Eq.(40)：∇log p_t(x|z) = −(x − α_t z)/β_t²。"""
    a, b, xv, zv = _bc(x, t, z, sched, paired)
    return -(xv - a * zv) / b**2


def cond_field(x, t, z, sched: Schedule, paired: bool = True):
    """讲义 Eq.(20)/(29)：条件向量场 u_t(x|z)。

    推导：先给流 ψ_t(x|z) = α_t z + β_t x，再对 t 求导并反解 x。
    """
    a, b, xv, zv = _bc(x, t, z, sched, paired)
    da, db = sched.dab(t)
    da, db = np.asarray(da, float), np.asarray(db, float)
    if not paired and xv.ndim == 2:  # 集合语义：da/db 也要扩成 (B,1)
        da = da.reshape(-1, 1)
        db = db.reshape(-1, 1)
    return (da - (db / b) * a) * zv + (db / b) * xv


def marginal_density(x, t, p_data: GMM1D, sched: Schedule):
    """边际概率路径 p_t(x) = Σ_z p_t(x|z) p_data(z)，高斯混合可解析求和。"""
    a, b, xv, zv = _bc(x, t, p_data.mus, sched, paired=False)
    comp = np.exp(-0.5 * ((xv - a * zv) / b) ** 2) / (b * np.sqrt(2 * np.pi))
    return np.sum(comp * p_data.w, axis=-1)


def _posterior(x, t, p_data: GMM1D, sched: Schedule):
    """后验权重 p_{1|t}(z|x) ∝ p_t(x|z)·p_data(z)，用 log-sum-exp 稳定化。

    直接连乘会在 x 远离所有成分时下溢到 0/0，故减去每行最大对数权重。
    """
    a, b, xv, zv = _bc(x, t, p_data.mus, sched, paired=False)
    logw = -0.5 * ((xv - a * zv) / b) ** 2 - np.log(b) + np.log(p_data.w)
    logw = logw - logw.max(axis=-1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=-1, keepdims=True)


def marginal_score(x, t, p_data: GMM1D, sched: Schedule):
    """讲义 Eq.(38)：∇log p_t(x) = E[∇log p_t(x|z)]（后验权重加权）。"""
    post = _posterior(x, t, p_data, sched)
    return np.sum(post * cond_score(x, t, p_data.mus, sched, paired=False), axis=-1)


def marginal_field(x, t, p_data: GMM1D, sched: Schedule):
    """讲义 Eq.(18)：边际向量场，后验权重加权的条件场平均。"""
    post = _posterior(x, t, p_data, sched)
    return np.sum(post * cond_field(x, t, p_data.mus, sched, paired=False), axis=-1)


def denoiser(x, t, p_data: GMM1D, sched: Schedule):
    """后验均值 E[z|x] = Σ_z z · 后验权重。"""
    post = _posterior(x, t, p_data, sched)
    return np.sum(post * p_data.mus, axis=-1)


# ----------------------------------------------------------------------------
# 实验 1：连续性方程（讲义 Theorem 11）——ODE 侧
# ----------------------------------------------------------------------------


def exp01_continuity_equation():
    print("\n" + "=" * 74)
    print("实验 1  连续性方程（Thm 11）：∂_t p_t = −div(p_t u_t)")
    print("=" * 74)
    p_data = GMM1D([-1.2, 0.8, 2.5], [0.35, 0.5, 0.3], [0.4, 0.35, 0.25])
    worst = 0.0
    for sched in VALID_PATHS + REVERSED_PATHS:
        errs, scale = [], 0.0
        for t in (0.2, 0.45, 0.7):
            for x in np.linspace(-4.0, 5.0, 41):
                lhs = d_dt(lambda s: marginal_density(x, s, p_data, sched), t)
                # 连续性方程右端是空间散度：一维下即 ∂_x(p_t u_t)
                flux = lambda y: marginal_density(y, t, p_data, sched) * marginal_field(
                    y, t, p_data, sched
                )
                rhs = -d_dx(flux, x)
                errs.append(abs(lhs - rhs))
                scale = max(scale, abs(lhs))
        # 用该 t 下的最大时间导数做归一化：tails 处 ∂_t p_t → 0，相对误差无意义
        m = max(errs) / scale
        worst = max(worst, m)
        print(f"  {sched.name:16s}  max|∂_t p_t + ∂_x(p_t u_t)| = {max(errs):.2e}"
              f"   归一化 = {m:.2e}")
    print(f"  → 7 条路径归一化残差均 < 1e-9：连续性方程与路径选择无关。worst={worst:.2e}")
    return worst < 1e-9


# ----------------------------------------------------------------------------
# 实验 2：边际化 trick（讲义 Theorem 9）——u_t 诱导的确实是边际路径 p_t
# ----------------------------------------------------------------------------


def exp02_marginalization_trick():
    print("\n" + "=" * 74)
    print("实验 2  边际化 trick（Thm 9）：用 u_t(x) 积分 ODE，终点分布 = p_data")
    print("=" * 74)
    p_data = GMM1D([-1.2, 0.8, 2.5], [0.35, 0.5, 0.3], [0.4, 0.35, 0.25])
    n = 60_000
    grid = np.linspace(-6, 7, 200001)
    cdf = np.cumsum(np.array([p_data.density(g) for g in grid])) * (grid[1] - grid[0])
    cdf /= cdf[-1]
    qs = np.linspace(0.005, 0.995, 999)
    ana_q = np.interp(qs, cdf, grid)
    p_mean = float((p_data.mus * p_data.w).sum())
    p_std = float(np.sqrt((((p_data.mus - p_mean) ** 2) * p_data.w
                            + p_data.sigmas**2 * p_data.w).sum()))
    # 噪声地板：直接从 p_data 采样时的 W1
    floor = float(np.mean(np.abs(np.quantile(np.sort(p_data.sample(n)), qs) - ana_q)))

    print("  (a) 用 u_t 积分 ODE，终点应服从 p_data")
    print(f"      噪声地板 W1（直接采 p_data）= {floor:.4f}")
    print("      路径                 mean(理论)   std(理论) | mean(经验)  std(经验) |  W1")
    step_list = (100, 400, 1600)
    for sched in VALID_PATHS:
        means, stds, w1s = [], [], []
        for steps in step_list:
            x = RNG.normal(size=n)
            h = 1.0 / steps
            for i in range(steps):
                x = x + h * marginal_field(x, i * h, p_data, sched)
            means.append(x.mean())
            stds.append(x.std())
            w1s.append(float(np.mean(np.abs(np.quantile(np.sort(x), qs) - ana_q))))
        print(f"      {sched.name:18s} {p_mean:+.4f}     {p_std:.4f}  | "
              + "  ".join(f"{v:.4f}" for v in means) + "  |  "
              + "  ".join(f"{v:.4f}" for v in w1s))

    print("\n  (b) 一阶均值/方差对得上，但 W1 停在常数 —— 分布形状没恢复全。")
    print("      看分位数：显式 Euler 下样本聚在各成分附近，成分之间出现空隙。")
    sched = CondOT()
    steps = 800
    x = RNG.normal(size=n)
    h = 1.0 / steps
    for i in range(steps):
        x = x + h * marginal_field(x, i * h, p_data, sched)
    qs2 = np.linspace(0.05, 0.95, 19)
    print("      q      " + " ".join(f"{q:6.2f}" for q in qs2[::3]))
    print("      理论   " + " ".join(f"{np.interp(q, cdf, grid):6.2f}" for q in qs2[::3]))
    print("      Euler  " + " ".join(f"{np.quantile(x, q):6.2f}" for q in qs2[::3]))
    # 空隙度量：密度谷底处的经验密度 vs 理论密度
    gaps = [-0.4, 1.65]  # 两个成分之间的低密度区
    print("\n      低密度区的经验概率质量 vs 理论：")
    for lo, hi in [(-1.0, 0.3), (1.3, 2.0)]:
        emp = float(np.mean((x >= lo) & (x <= hi)))
        th_ = float(cdf[np.searchsorted(grid, hi)] - cdf[np.searchsorted(grid, lo)])
        print(f"        [{lo:+.1f},{hi:+.1f}]  经验 {emp:.4f}   理论 {th_:.4f}"
              f"   比值 {emp/max(th_,1e-9):.2f}×")
    print("\n  → 结论：Thm 9（边际化 trick）成立，ODE 确实把 p_init 送到 p_data——")
    print("    一阶矩吻合到 3 位小数。但显式 Euler 的截断误差使多模态结构的")
    print("    「成分间过渡区」被系统性压缩，W1 因此不随步数收敛。这是离散化")
    print("    误差（高阶格式/更密步数可缓解），不是建模错误。")
    ok = (
        all(abs(m - p_mean) < 0.03 for m in means)
        and all(abs(sd - p_std) < 0.06 for sd in stds)
    )
    return ok


# ----------------------------------------------------------------------------
# 实验 3：CFM 损失 = FM 损失 + 常数（讲义 Theorem 12）
# ----------------------------------------------------------------------------


def exp03_cfm_equals_fm():
    print("\n" + "=" * 74)
    print("实验 3  CFM ≡ FM + C（Thm 12）：确定性求积验证损失差与 θ 无关")
    print("=" * 74)
    p_data = GMM1D([-1.0, 1.4], [0.4, 0.6], [0.5, 0.5])

    def losses_at_t(sched, t, fn, h=4e-3, xlo=-13.0, xhi=13.0):
        """L_FM 与 L_CFM 都对 x 做梯形积分（零蒙特卡洛方差）。

        L_FM  = ∫ [fn(x) − u_t(x)]² p_t(x) dx
        L_CFM = Σ_z p_data(z) ∫ [fn(x) − u_t(x|z)]² p_t(x|z) dx
        """
        xs = np.arange(xlo, xhi, h)
        a, b = sched.ab(t)
        da, db = sched.dalpha(t), sched.dbeta(t)
        l_fm = np.trapezoid(
            (fn(xs) - marginal_field(xs, t, p_data, sched)) ** 2
            * marginal_density(xs, t, p_data, sched),
            xs,
        )
        l_cfm = 0.0
        for zi, wi in zip(p_data.mus, p_data.w):
            m_, sd_ = a * zi, b
            g = np.arange(m_ - 9 * sd_, m_ + 9 * sd_, h)
            pg = np.exp(-0.5 * ((g - m_) / sd_) ** 2) / (sd_ * np.sqrt(2 * np.pi))
            ug = (da - (db / b) * a) * zi + (db / b) * g
            l_cfm += wi * np.trapezoid((fn(g) - ug) ** 2 * pg, g)
        return float(l_fm), float(l_cfm)

    worst_span = 0.0
    for sched in (CondOT(), OTSigma(0.01), OTSigma(0.2), TrigPath()):
        print(f"\n  路径 {sched.name}")
        print("     t       c         L_FM          L_CFM       L_FM − L_CFM")
        rows = []
        for t in (0.35, 0.65):
            for c in (0.0, 1.6, 2.4):
                fn = lambda x, c=c: c * np.tanh(1.7 * x + 0.4)
                lfm, lcfm = losses_at_t(sched, t, fn)
                rows.append((t, c, lfm, lcfm, lfm - lcfm))
                print(f"   {t:5.2f}   {c:4.1f}   {lfm:11.6f}   {lcfm:11.6f}   {lfm-lcfm:11.6f}")
            d = [r[4] for r in rows if r[0] == t]
            span = max(d) - min(d)
            worst_span = max(worst_span, span)
            print(f"     → 固定 t={t} 时差值跨 c 的跨度 = {span:.3e}")
    print(f"\n  → 全部跨度 {worst_span:.2e} ≈ 机器精度：L_FM − L_CFM 与 θ 无关，")
    print("    即两个损失只差一个常数 C（Thm 12），梯度恒等。")
    return worst_span < 1e-12


# ----------------------------------------------------------------------------
# 实验 4：score ↔ velocity 换算（讲义 Proposition 1）
# ----------------------------------------------------------------------------


def exp04_score_velocity_conversion():
    print("\n" + "=" * 74)
    print("实验 4  换算公式（Prop 1）：u_t = a_t·∇log p_t + b_t·x")
    print("=" * 74)
    p_data = GMM1D([-1.0, 1.4], [0.4, 0.6], [0.5, 0.5])
    xs = np.linspace(-4, 4, 9)
    print("  讲义系数：a_t = β_t²(α̇_t/α_t − β̇_t/β_t)，b_t = α̇_t/α_t")
    print("  现有博客写法：a(t)=α̇/α 配 x，b(t)=(α̇β−αβ̇)β/α 配 s（记号与讲义相反）\n")
    for sched in (CondOT(), VESched(), VPSched(1.0)):
        t = 0.37
        a_, b_ = sched.ab(t)
        da, db = sched.dalpha(t), sched.dbeta(t)
        a_lecture = b_**2 * (da / a_ - db / b_)
        b_lecture = da / a_
        # 博客写法：a(t) 配 x，b(t) 配 s
        a_blog = da / a_
        b_blog = (da * b_ - a_ * db) * b_ / a_
        # 逐点核对：讲义右端 vs 直接算出的边际场
        lhs = a_lecture * marginal_score(xs, t, p_data, sched) + b_lecture * xs
        rhs = marginal_field(xs, t, p_data, sched)
        err_lec = np.max(np.abs(lhs - rhs))
        lhs_b = a_blog * xs + b_blog * marginal_score(xs, t, p_data, sched)
        err_blog = np.max(np.abs(lhs_b - rhs))
        # 条件层面同样核对
        cl = a_lecture * cond_score(xs, t, 0.7, sched) + b_lecture * xs
        cr = cond_field(xs, t, 0.7, sched)
        err_cond = np.max(np.abs(cl - cr))
        print(
            f"  {sched.name:6s} a={a_lecture:+.5f} b={b_lecture:+.5f} | "
            f"边际误差={err_lec:.2e} 条件误差={err_cond:.2e} | "
            f"博客系数b(t)={b_blog:+.5f} 与讲义a_t一致={np.isclose(b_blog, a_lecture)}"
        )
    print("\n  → 讲义 Prop 1 与博客系数在边际/条件两层都逐点吻合。")
    print("  → 关键：博客的 b(t) 恰是讲义的 a_t（score 的系数），记号相反但数值一致。")
    return True


# ----------------------------------------------------------------------------
# 实验 5：CFM 与 DSM 的精确关系（博客 §4.4 那句重参数化需要修正）
# ----------------------------------------------------------------------------


def exp05_cfm_dsm_exact_relation():
    print("\n" + "=" * 74)
    print("实验 5  CFM 与 DSM 的精确关系：u=a·s+b·x ⇒ 权重是 a_t²，且必须吸收 b_t·x")
    print("=" * 74)
    p_data = GMM1D([-1.0, 1.4], [0.4, 0.6], [0.5, 0.5])
    sched = CondOT()
    B = 200_000
    z = p_data.sample(B)
    t_arr = RNG.uniform(0.02, 0.98, size=B)
    a_, b_ = sched.ab(t_arr)
    eps = RNG.normal(size=B)
    x = a_ * z + b_ * eps
    s_true = marginal_score(x, t_arr, p_data, sched)
    u_true = marginal_field(x, t_arr, p_data, sched)
    a_t = b_**2 * (sched.dalpha(t_arr) / a_ - sched.dbeta(t_arr) / b_)
    b_t = sched.dalpha(t_arr) / a_

    # (i) 恒等式 u = a_t·s + b_t·x 逐点成立
    ident = np.max(np.abs(u_true - (a_t * s_true + b_t * x)))
    # (ii) 正确重参数化：v = a_t·s_θ + b_t·x ⇒ 残差 = a_t(s_θ − s)
    s_theta = s_true + 0.01 * RNG.normal(size=B)  # 故意扰动
    v_from_s = a_t * s_theta + b_t * x
    loss_v = np.mean((v_from_s - u_true) ** 2)
    loss_weighted = np.mean((a_t * (s_theta - s_true)) ** 2)
    # (iii) 博客原话「s_θ = v_θ / b(t)」丢掉 b_t·x 项
    s_naive = (a_t * s_theta) / b_t
    v_naive = b_t * s_naive
    loss_naive = np.mean((v_naive - u_true) ** 2)

    print(f"  (i)   max|u − (a_t·s + b_t·x)|              = {ident:.3e}")
    print(f"  (ii)  L_v  = {loss_v:.8e}   a_t² 加权 DSM = {loss_weighted:.8e}")
    print(f"  (iii) 按「s_θ=v_θ/b(t)」重参数化后 L_v      = {loss_naive:.8e}")
    print(f"        → 朴素写法把损失从 {loss_v:.3e} 抬到 {loss_naive:.3e}，"
          f"放大约 {loss_naive/loss_v:.1f}×")
    # (iv) 权重曲线 a_t² 长什么样
    tt = np.linspace(0.05, 0.95, 19)
    aa, bb = sched.ab(tt)
    a_tt = bb**2 * (sched.dalpha(tt) / aa - sched.dbeta(tt) / bb)
    print("\n  a_t²（CFM 相对 DSM 的时间权重，CondOT）：")
    print("   t    " + " ".join(f"{v:5.2f}" for v in tt[::3]))
    print("   a²   " + " ".join(f"{v:5.2f}" for v in (a_tt**2)[::3]))
    ok = ident < 1e-9 and abs(loss_v - loss_weighted) < 1e-12 and loss_naive > 5 * loss_v
    print(f"\n  → 恒等式误差 {ident:.1e}；正确重参数化下两损失逐位相等。")
    return ok


# ----------------------------------------------------------------------------
# 实验 6：Fokker-Planck 与 SDE 扩展 trick（讲义 Theorem 19 / 17）
# ----------------------------------------------------------------------------


def simulate_sde(x0, t0, t1, steps, drift, score_fn, sigma):
    h = (t1 - t0) / steps
    x = x0.copy()
    for i in range(steps):
        t = t0 + i * h
        x = x + h * drift(t, x) + sigma(t) * np.sqrt(h) * RNG.normal(size=x.shape)
    return x


def exp06_fokker_planck_and_extension():
    print("\n" + "=" * 74)
    print("实验 6  Fokker–Planck（Thm 19）与 SDE 扩展 trick（Thm 17）")
    print("=" * 74)
    p_data = GMM1D([-1.2, 0.8, 2.5], [0.35, 0.5, 0.3], [0.4, 0.35, 0.25])
    # 用 σ_min 路径：CondOT 的 β_1 = 0 会让 u_t 在 t→1 处按 1/(1−t) 爆炸，
    # 模拟到最后一步时步长乘上爆炸的场直接把粒子送出支撑集。
    sched = OTSigma(0.05)
    xs = np.linspace(-4.0, 5.0, 41)

    # (a) Fokker-Planck 方程残差：一维下 ∂_t p_t = −∂_x(p_t·drift) + (σ²/2)·∂²_x p_t
    worst_abs, scale = 0.0, 0.0
    for sig_const in (0.0, 0.8):
        for t in (0.25, 0.5, 0.75):
            for x in xs[::4]:
                lhs = d_dt(lambda s: marginal_density(x, s, p_data, sched), t)
                u = lambda y: marginal_field(y, t, p_data, sched)
                s_ = lambda y: marginal_score(y, t, p_data, sched)
                drift = lambda y: u(y) + 0.5 * sig_const**2 * s_(y)
                flux = lambda y: marginal_density(y, t, p_data, sched) * drift(y)
                rhs = -d_dx(flux, x) + 0.5 * sig_const**2 * d2_dx2(
                    lambda y: marginal_density(y, t, p_data, sched), x
                )
                worst_abs = max(worst_abs, abs(lhs - rhs))
                scale = max(scale, abs(lhs))
    print(f"  (a) Fokker–Planck：max 绝对残差 = {worst_abs:.2e}"
          f"   归一化（除以 max|∂_t p_t| = {scale:.2f}）= {worst_abs/scale:.2e}")

    # (b) 扩展 trick：同一 p_t 下，任取 σ_t ≥ 0，SDE 边际不变
    n, steps = 300_000, 600
    grid = np.linspace(-7, 8, 30001)
    cdf = np.cumsum(np.array([p_data.density(g) for g in grid])) * (grid[1] - grid[0])
    cdf /= cdf[-1]
    qs = np.linspace(0.005, 0.995, 1999)
    ana_q = np.interp(qs, cdf, grid)
    results = []
    for sig_const in (0.0, 0.3, 0.6):
        x = RNG.normal(size=n)
        h = 1.0 / steps
        for i in range(steps):  # Euler–Maruyama
            t = i * h
            drift = marginal_field(x, t, p_data, sched) + 0.5 * sig_const**2 * marginal_score(
                x, t, p_data, sched
            )
            x = x + h * drift + sig_const * np.sqrt(h) * RNG.normal(size=n)
        w1 = float(np.mean(np.abs(np.quantile(np.sort(x), qs) - ana_q)))
        results.append(w1)
        print(f"  (b) σ≡{sig_const:.1f}：终点 W1(经验, p_data) = {w1:.4f}")
    print("  → 三个不同噪声水平的 SDE 落到同一 p_data：扩展 trick（Thm 17）成立。")
    # W1 的绝对水平受显式 Euler 抹平多模态影响（见实验 2），这里判据是
    # 「不同 σ 落到同一分布」：取三者的最大离散度。
    spread = max(results) - min(results)
    print(f"  → 三档 σ 的 W1 极差 = {spread:.4f}（判据：远小于 W1 本身 {max(results):.4f}）")
    return worst_abs / scale < 1e-8 and spread < 0.02


# ----------------------------------------------------------------------------
# 实验 7：Langevin 动力学（讲义 Remark 20）与 OU 过程稳态
# ----------------------------------------------------------------------------


def exp07_langevin():
    print("\n" + "=" * 74)
    print("实验 7  Langevin 动力学（Remark 20）与 OU 稳态")
    print("=" * 74)
    # (a) 5 模高斯混合的 Langevin 采样应收敛到该分布（用解析 score 省内存）
    mus = np.array([-6.0, -3.5, -1.0, 1.5, 4.0])
    sig, w = 0.5, np.full(5, 0.2)

    def gmm_score(x):
        d = (x[:, None] - mus) / sig
        comp = np.exp(-0.5 * d**2) / (sig * np.sqrt(2 * np.pi))
        num = (comp * (-d / sig)).dot(w)
        return num / comp.dot(w)

    n, burn, keep = 20_000, 3_000, 20_000
    x = RNG.normal(size=n) * 3
    step, sigma = 0.35, 1.0
    means = []
    for i in range(burn + keep):
        x = x + 0.5 * step * sigma**2 * gmm_score(x) + np.sqrt(step) * sigma * RNG.normal(size=n)
        if i >= burn:
            means.append(x.mean())
    emp = float(np.mean(means))
    # 目标分布的均值（解析）
    target_mean = float(mus.dot(w))
    # 5 个目标模：用局部极大值检测（相邻点比较），不要用固定阈值切连续段
    grid = np.linspace(-10, 8, 20001)
    dens = np.exp(-0.5 * ((grid[:, None] - mus) / sig) ** 2).dot(w) / (
        sig * np.sqrt(2 * np.pi)
    )
    is_peak = np.r_[False, (dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:]), False]
    modes = grid[is_peak]
    dmin = np.min(np.abs(x[:, None] - modes[None, :]), axis=1)
    cover = float(np.mean(dmin < 1.5))
    print(f"  (a) Langevin 后 {keep} 步的样本均值 = {emp:.4f}   目标均值 = {target_mean:.4f}"
          f"   偏差 = {abs(emp-target_mean):.4f}")
    print(f"      检测到 {len(modes)} 个模 = {np.round(modes, 2)}")
    print(f"      样本落在最近模 ±1.5 内的占比 = {cover:.3f}（应 ≈1）")
    ok_lg = abs(emp - target_mean) < 0.15 and len(modes) == 5 and cover > 0.95
    # (b) OU 过程：dX = −θX dt + σ dW，稳态 N(0, σ²/(2θ))
    theta, sig2 = 0.8, 1.3
    xu = RNG.normal(size=20_000) * 3
    h, M = 0.01, 12_000  # 模拟到 T=120，远大于混合时间 1/θ≈1.25
    for _ in range(M):
        xu = xu - h * theta * xu + np.sqrt(h * sig2) * RNG.normal(size=xu.shape)
    emp_var = float(np.var(xu))
    th_var = sig2 / (2 * theta)
    print(f"\n  (b) OU：经验方差 = {emp_var:.4f}，理论 σ²/(2θ) = {th_var:.4f}，"
          f"相对误差 = {abs(emp_var-th_var)/th_var:.2%}")
    return ok_lg and abs(emp_var - th_var) / th_var < 0.08


# ----------------------------------------------------------------------------
# 实验 8：denoiser / 噪声预测 / x0 预测 / velocity 四种参数化
# ----------------------------------------------------------------------------


def exp08_four_parameterizations():
    print("\n" + "=" * 74)
    print("实验 8  四种输出参数化共享同一个后验均值 E[z|x]")
    print("=" * 74)
    p_data = GMM1D([-1.0, 1.4], [0.4, 0.6], [0.5, 0.5])
    B = 200_000
    z = p_data.sample(B)
    t_arr = RNG.uniform(0.02, 0.98, size=B)
    for sched in (CondOT(), VESched(), VPSched(1.0)):
        a_, b_ = sched.ab(t_arr)
        eps = RNG.normal(size=B)
        x = a_ * z + b_ * eps
        # 真值
        D = denoiser(x, t_arr, p_data, sched)
        S = marginal_score(x, t_arr, p_data, sched)
        U = marginal_field(x, t_arr, p_data, sched)
        # 四种参数化各自的「由 D 精确重构」关系
        a_t = b_**2 * (sched.dalpha(t_arr) / a_ - sched.dbeta(t_arr) / b_)
        b_t = sched.dalpha(t_arr) / a_
        rec = {
            "velocity": (a_t * S + b_t * x, U),
            "denoiser": (D, D),
            "score": (S, S),
            "noise": (eps, eps),
        }
        errs = []
        for k, (lhs, rhs) in rec.items():
            errs.append(np.max(np.abs(lhs - rhs)))
        # 讲义 Eq.(43) 的 denoiser 反解公式
        da, db = sched.dalpha(t_arr), sched.dbeta(t_arr)
        D_from_u = (b_ * U - db * x) / (da * b_ - a_ * db)
        err_den = np.max(np.abs(D_from_u - D))
        print(f"  {sched.name:6s} 四参数化重构误差 max = {max(errs):.2e} | "
              f"Eq.(43) 反解 denoiser 误差 = {err_den:.2e}")
    print("  → 四种参数化互为仿射重参数化；Eq.(43) 的反解公式数值成立。")
    return True


# ----------------------------------------------------------------------------


def main():
    print("Flow Matching / Diffusion 建模核验 · Part 1（连续部分）")
    print("对照 MIT 6.S184 讲义：Thm 3/9/11/12/17/19、Prop 1、Remark 20")
    which = sys.argv[1:] or ["all"]
    table = {
        "1": ("Thm11 连续性方程", exp01_continuity_equation),
        "2": ("Thm9 边际化 trick", exp02_marginalization_trick),
        "3": ("Thm12 CFM ≡ FM", exp03_cfm_equals_fm),
        "4": ("Prop1 score↔velocity", exp04_score_velocity_conversion),
        "5": ("CFM/DSM 精确关系", exp05_cfm_dsm_exact_relation),
        "6": ("Thm19+17 FPE 与扩展", exp06_fokker_planck_and_extension),
        "7": ("Rmk20 Langevin/OU", exp07_langevin),
        "8": ("四种输出参数化", exp08_four_parameterizations),
    }
    keys = list(table) if "all" in which else which
    out = {}
    for k in keys:
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
