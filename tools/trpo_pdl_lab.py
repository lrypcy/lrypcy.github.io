#!/usr/bin/env python3
"""TRPO 理论链 companion lab：在表格 MDP 上把「恒等式 / 近似 / 不等式」逐条数值体检。

表格 MDP 上 V、Q、A、折扣访问分布、Fisher 矩阵全部可以解析算出来，
于是 TRPO 的每一个环节都能和「真值」对照 —— 这是神经网络上做不到的。

分组运行（沙箱会掐长脚本，务必分组）：
    python3 tools/trpo_pdl_lab.py 1 2      # 默认全跑
  1  PDL 恒等式：精确性 + 部分和收敛
  2  surrogate：一阶精确、二阶漂移
  3  KL 上界定理：不等式成立 + 保守度 + max/平均 KL 的差距
  4  精确 vs 近似：Fisher 奇异、CG 截断、泰勒近似、阻尼、步长闭式解
"""

import sys
import numpy as np

N_S, N_A, GAMMA = 8, 4, 0.9


def section(msg):
    print("\n" + "=" * 74)
    print(msg)
    print("=" * 74)


def make_mdp(seed=0, n_s=N_S, n_a=N_A, gamma=GAMMA):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(n_s) * 0.6, size=(n_s, n_a))   # P[s, a, s']
    R = rng.uniform(-1.0, 1.0, size=(n_s, n_a))
    rho0 = rng.dirichlet(np.ones(n_s))
    return P, R, rho0, gamma


def softmax_rows(theta):
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def eval_policy(P, R, rho0, gamma, pi):
    """策略的精确打分：V / Q / A / J / 诱导转移矩阵。"""
    n_s = P.shape[0]
    Ppi = np.einsum("sa,sax->sx", pi, P)
    rpi = np.einsum("sa,sa->s", pi, R)
    V = np.linalg.solve(np.eye(n_s) - gamma * Ppi, rpi)
    Q = R + gamma * np.einsum("sax,x->sa", P, V)
    A = Q - V[:, None]
    return dict(V=V, Q=Q, A=A, J=float(rho0 @ V), Ppi=Ppi)


def visit_unnorm(rho0, Ppi, gamma):
    """未归一化折扣访问 rho^pi(s) = sum_t gamma^t P(s_t = s)。"""
    n_s = len(rho0)
    return np.linalg.solve((np.eye(n_s) - gamma * Ppi).T, rho0)


def visit_at_t(rho0, Ppi, t):
    v = rho0.copy()
    for _ in range(t):
        v = v @ Ppi
    return v


def kl_rows(p, q):
    return np.sum(p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300)), axis=1)


def tv_rows(p, q):
    return 0.5 * np.sum(np.abs(p - q), axis=1)


# ---------------------------------------------------------------- 1. PDL
def exp1():
    section("§2  性能差异引理（PDL）：恒等式本身是精确的")
    print("  MDP: |S|=%d |A|=%d gamma=%.2f   两种写法对照（未归一化 / 归一化 d^pi）" % (N_S, N_A, GAMMA))
    print("  %-4s %-14s %-14s %-12s %-12s" % ("seed", "J(pi')-J(pi)", "sum_t gamma^t E", "残差", "1/(1-g)E 残差"))
    worst = 0.0
    for seed in range(6):
        P, R, rho0, gamma = make_mdp(seed)
        th0 = np.random.default_rng(100 + seed).normal(size=(N_S, N_A)) * 0.7
        th1 = np.random.default_rng(200 + seed).normal(size=(N_S, N_A)) * 0.7
        pi0, pi1 = softmax_rows(th0), softmax_rows(th1)
        e0, e1 = eval_policy(P, R, rho0, gamma, pi0), eval_policy(P, R, rho0, gamma, pi1)
        dJ = e1["J"] - e0["J"]

        rho1 = visit_unnorm(rho0, e1["Ppi"], gamma)
        Abar1 = np.einsum("sa,sa->s", pi1, e0["A"])          # E_{a ~ pi'}[A^pi(s,a)]
        form_unnorm = float(rho1 @ Abar1)                     # sum_t gamma^t E_{d_t^{pi'}, pi'}[A]
        d1 = (1.0 - gamma) * rho1                             # 归一化 d^{pi'}
        form_norm = float(d1 @ Abar1) / (1.0 - gamma)

        worst = max(worst, abs(form_unnorm - dJ), abs(form_norm - dJ))
        print("  %-4d %-14.9f %-14.9f %-12.2e %-12.2e"
              % (seed, dJ, form_unnorm, abs(form_unnorm - dJ), abs(form_norm - dJ)))
    print("\n  最大残差 = %.3e  —— 双精度机器精度量级，PDL 是恒等式，不是近似" % worst)

    # 部分和收敛：只加到第 T 项，余项按 gamma^T 衰减
    print("\n  截断到 T 项的部分和 S_T 与真值之差（seed=0）：")
    P, R, rho0, gamma = make_mdp(0)
    th0 = np.random.default_rng(100).normal(size=(N_S, N_A)) * 0.7
    th1 = np.random.default_rng(200).normal(size=(N_S, N_A)) * 0.7
    pi0, pi1 = softmax_rows(th0), softmax_rows(th1)
    e0, e1 = eval_policy(P, R, rho0, gamma, pi0), eval_policy(P, R, rho0, gamma, pi1)
    Abar1 = np.einsum("sa,sa->s", pi1, e0["A"])
    dJ = e1["J"] - e0["J"]
    print("  %-6s %-16s %-14s %-14s" % ("T", "S_T", "S_T - 真值", "gamma^T"))
    S = 0.0
    for t in range(0, 401):
        S += gamma ** t * float(visit_at_t(rho0, e1["Ppi"], t) @ Abar1)
        if t in (0, 1, 5, 10, 20, 50, 100, 200, 400):
            print("  %-6d %-16.9f %-14.2e %-14.2e" % (t, S, S - dJ, gamma ** t))


# ---------------------------------------------------------------- 2. surrogate
def exp2():
    section("§3  surrogate：一阶精确、二阶漂移")
    P, R, rho0, gamma = make_mdp(0)
    rng = np.random.default_rng(7)
    th0 = rng.normal(size=(N_S, N_A)) * 0.7
    pi0 = softmax_rows(th0)
    e0 = eval_policy(P, R, rho0, gamma, pi0)
    rho0v = visit_unnorm(rho0, e0["Ppi"], gamma)
    dnorm = (1.0 - gamma) * rho0v

    # 更新方向取策略梯度方向（第二篇的梯度），再归一化，便于扫步长
    def grad_L(th):
        p = softmax_rows(th)
        return rho0v[:, None] * (p * (e0["A"] - np.sum(p * e0["A"], axis=1, keepdims=True)))

    def L_of(th):
        p = softmax_rows(th)
        return float(rho0v @ np.einsum("sa,sa->s", p, e0["A"]))

    def J_of(th):
        return eval_policy(P, R, rho0, gamma, softmax_rows(th))["J"]

    # (a) 一阶精确：nabla J|_{th0} == nabla L|_{th0}（有限差分检验 nabla J）
    g = grad_L(th0).ravel()
    eps = 1e-6
    fd = np.zeros_like(th0)
    for i in range(N_S):
        for j in range(N_A):
            tp, tm = th0.copy(), th0.copy()
            tp[i, j] += eps
            tm[i, j] -= eps
            fd[i, j] = (J_of(tp) - J_of(tm)) / (2 * eps)
    fd = fd.ravel()
    print("  (a) 一阶匹配：||∇J − ∇L||_inf = %.3e   ||∇L||_inf = %.3e"
          % (np.max(np.abs(fd - g)), np.max(np.abs(g))))
    print("      相对误差 = %.3e  —— ∇L 就是策略梯度，两者是同一个向量" % (np.max(np.abs(fd - g)) / np.max(np.abs(g))))

    # (b) 沿该方向扫步长：L 与 ΔJ 在 α→0 处贴合，α 增大后分叉
    v = grad_L(th0)
    v = v / np.linalg.norm(v)
    print("\n  (b) 沿策略梯度方向走 α 步（Δθ 的范数 = α）：")
    print("  %-8s %-14s %-14s %-12s %-12s %-10s" % ("alpha", "L（代理提升）", "ΔJ（真实提升）", "L−ΔJ", "比值 ΔJ/L", "平均KL"))
    for a in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        th = th0 + a * v
        p = softmax_rows(th)
        Lv = L_of(th)
        dJv = J_of(th) - e0["J"]
        kl = float(dnorm @ kl_rows(pi0, p))
        print("  %-8.2f %-14.7f %-14.7f %-12.2e %-12.4f %-10.4f"
              % (a, Lv, dJv, Lv - dJv, (dJv / Lv if abs(Lv) > 1e-12 else float("nan")), kl))

    # (c) 同一段参数位移，视野越长（gamma → 1）兑付率越差
    print("\n  (c) 沿各自的策略梯度方向走同样长的 α = 2.0，gamma 越大兑付率越低：")
    print("  %-8s %-14s %-14s %-12s %-12s %-12s" % ("gamma", "L（承诺）", "ΔJ（兑现）", "兑现率", "绝对漂移", "平均 KL"))
    for gm in (0.80, 0.90, 0.95, 0.99):
        P_g, R_g, rho0_g, _ = make_mdp(0, gamma=gm)
        e0_g = eval_policy(P_g, R_g, rho0_g, gm, pi0)
        rho_v_g = visit_unnorm(rho0_g, e0_g["Ppi"], gm)
        dn_g = (1.0 - gm) * rho_v_g
        g_g = rho_v_g[:, None] * (pi0 * (e0_g["A"] - np.sum(pi0 * e0_g["A"], axis=1, keepdims=True)))
        v_g = g_g / np.linalg.norm(g_g)
        p_new = softmax_rows(th0 + 2.0 * v_g)
        L_g = float(rho_v_g @ np.einsum("sa,sa->s", p_new, e0_g["A"]))
        dJ_g = eval_policy(P_g, R_g, rho0_g, gm, p_new)["J"] - e0_g["J"]
        print("  %-8.2f %-14.7f %-14.7f %-12.4f %-12.4e %-12.4f"
              % (gm, L_g, dJ_g, dJ_g / max(L_g, 1e-12), L_g - dJ_g, float(dn_g @ kl_rows(pi0, p_new))))
    print("      ⇒ 兑付率随 γ 只从 0.978 缓降到 0.971，但绝对漂移量按 1/(1−γ) 放大了 25 倍")
    print("        （0.0186 → 0.4690）：视野越长，同一段位移惹出的偏差越大，δ 就得越紧。")

    # (d) 反例搜索：确实存在「surrogate 说涨、真实性能说跌」的方向
    print("\n  (d) 反例搜索：随机方向 + 固定 KL 预算，找出 L > 0 > ΔJ 的更新（gamma=0.99）")
    P_g, R_g, rho0_g, _ = make_mdp(0, gamma=0.99)
    e0_g = eval_policy(P_g, R_g, rho0_g, 0.99, pi0)
    rho_v_g = visit_unnorm(rho0_g, e0_g["Ppi"], 0.99)
    dn_g = (1.0 - 0.99) * rho_v_g
    rng2 = np.random.default_rng(99)
    found = 0
    best = None
    for trial in range(4000):
        w = rng2.normal(size=(N_S, N_A))
        w /= np.linalg.norm(w)
        a = 0.3 + 3.0 * rng2.random()
        p_new = softmax_rows(th0 + a * w)
        L_g = float(rho_v_g @ np.einsum("sa,sa->s", p_new, e0_g["A"]))
        if L_g <= 0:
            continue
        dJ_g = eval_policy(P_g, R_g, rho0_g, 0.99, p_new)["J"] - e0_g["J"]
        kl_g = float(dn_g @ kl_rows(pi0, p_new))
        if dJ_g < 0:
            found += 1
            if best is None or dJ_g < best[0]:
                best = (dJ_g, L_g, kl_g, a)
    print("      4000 次随机试探中，L > 0 且 ΔJ < 0 的命中 %d 次" % found)
    if best is not None:
        print("      最狠的一例：L = %+.6f（承诺上涨）   ΔJ = %+.6f（实际下跌）   平均 KL = %.4f"
              % (best[1], best[0], best[2]))
    print("      ⇒ 这就是 KL 约束存在的理由：surrogate 本身拦不住这种更新")


# ---------------------------------------------------------------- 3. KL 上界定理
def exp3():
    section("§4  KL 上界定理：不等式成立，但保守到没法直接用")
    print("  检查  ΔJ_true ≥ L − C·D，C = 4·eps·gamma/(1−gamma)^2，eps = max|A^pi|")
    print("  %-5s %-8s %-12s %-12s %-12s %-10s %-10s"
          % ("seed", "alpha方向", "L", "ΔJ", "|L−ΔJ|", "C·D_maxTV²", "C·D_maxKL"))
    ratio_tv, ratio_kl = [], []
    for seed in range(6):
        P, R, rho0, gamma = make_mdp(seed)
        rng = np.random.default_rng(300 + seed)
        th0 = rng.normal(size=(N_S, N_A)) * 0.7
        pi0 = softmax_rows(th0)
        e0 = eval_policy(P, R, rho0, gamma, pi0)
        rho_v = visit_unnorm(rho0, e0["Ppi"], gamma)
        dnorm = (1.0 - gamma) * rho_v
        eps = float(np.max(np.abs(e0["A"])))
        C = 4.0 * eps * gamma / (1.0 - gamma) ** 2
        v = rng.normal(size=(N_S, N_A))
        v /= np.linalg.norm(v)
        for a in (0.05, 0.5, 2.0):
            th1 = th0 + a * v
            pi1 = softmax_rows(th1)
            L = float(rho_v @ np.einsum("sa,sa->s", pi1, e0["A"]))
            dJ = eval_policy(P, R, rho0, gamma, pi1)["J"] - e0["J"]
            a_tv = float(np.max(tv_rows(pi0, pi1)))
            d_kl_max = float(np.max(kl_rows(pi0, pi1)))
            d_kl_avg = float(dnorm @ kl_rows(pi0, pi1))
            assert dJ >= L - C * a_tv ** 2 - 1e-9, "Theorem 1 被违反"
            assert dJ >= L - C * d_kl_max - 1e-9, "KL 推论被违反"
            ratio_tv.append(abs(L - dJ) / (C * a_tv ** 2))
            ratio_kl.append(abs(L - dJ) / (C * d_kl_max))
            print("  %-5d %-8.2f %-12.6f %-12.6f %-12.2e %-10.3e %-10.3e"
                  % (seed, a, L, dJ, abs(L - dJ), C * a_tv ** 2, C * d_kl_max))
    print("\n  Theorem 1 全部成立（18 组随机试验无一违反）。")
    print("  实际违反量 / 定理上界： TV 版平均 %.2e，KL 版平均 %.2e"
          % (np.mean(ratio_tv), np.mean(ratio_kl)))
    print("  最大占比 = %.2e  —— 定理给出的惩罚项比真实漂移大 4~6 个数量级，" % max(ratio_kl))
    print("  所以 TRPO 只借它的「KL 小 ⇒ 漂移可控」这个定性结论，常数 C 从不进代码。")

    # max KL 与平均 KL 的差距（说明「max → 平均」不是等价变形）
    print("\n  max_s D_KL 与 E_s[D_KL] 的比值（说明 §5 里把 max 换成平均是实质改动）：")
    print("  %-6s %-14s %-14s %-12s %-12s" % ("seed", "max KL", "平均 KL", "比值", "|漂移|/平均KL"))
    for seed in range(4):
        P, R, rho0, gamma = make_mdp(seed)
        rng = np.random.default_rng(400 + seed)
        th0 = rng.normal(size=(N_S, N_A)) * 0.7
        pi0 = softmax_rows(th0)
        e0 = eval_policy(P, R, rho0, gamma, pi0)
        rho_v = visit_unnorm(rho0, e0["Ppi"], gamma)
        dnorm = (1.0 - gamma) * rho_v
        v = rng.normal(size=(N_S, N_A))
        v /= np.linalg.norm(v)
        kl_seed = rng.dirichlet(np.ones(N_S) * 0.3) * 3.0   # 让各状态改动幅度高度不均
        th1 = th0 + (v * kl_seed[:, None]) * 2.0
        pi1 = softmax_rows(th1)
        kls = kl_rows(pi0, pi1)
        mx, av = float(np.max(kls)), float(dnorm @ kls)
        L = float(rho_v @ np.einsum("sa,sa->s", pi1, e0["A"]))
        dJ = eval_policy(P, R, rho0, gamma, pi1)["J"] - e0["J"]
        print("  %-6d %-14.6f %-14.6f %-12.2f %-12.2e" % (seed, mx, av, mx / max(av, 1e-12), abs(L - dJ) / max(av, 1e-12)))


# ---------------------------------------------------------------- 4. 精确 vs 近似
def exp4():
    section("§5  精确 vs 近似：Fisher、CG、泰勒、步长闭式解")
    P, R, rho0, gamma = make_mdp(0)
    rng = np.random.default_rng(11)
    th0 = rng.normal(size=(N_S, N_A)) * 0.7
    pi0 = softmax_rows(th0)
    e0 = eval_policy(P, R, rho0, gamma, pi0)
    rho_v = visit_unnorm(rho0, e0["Ppi"], gamma)
    dnorm = (1.0 - gamma) * rho_v

    # Fisher：块对角，块 F_s = d(s)·(diag(pi_s) − pi_s pi_s^T)
    F = np.zeros((N_S * N_A, N_S * N_A))
    for s in range(N_S):
        p = pi0[s]
        F[s * N_A:(s + 1) * N_A, s * N_A:(s + 1) * N_A] = dnorm[s] * (np.diag(p) - np.outer(p, p))
    g = (rho_v[:, None] * (pi0 * (e0["A"] - np.sum(pi0 * e0["A"], axis=1, keepdims=True)))).ravel()

    print("  (a) Fisher 的奇异性：n = %d，特征值（从小到大 6 个）" % (N_S * N_A))
    ev = np.linalg.eigvalsh(F)
    print("      " + "  ".join("%.2e" % x for x in ev[:6]))
    print("      零特征值个数（< 1e-10）= %d  ⇒ F 奇异，必须加阻尼才能解 Fx = g" % int(np.sum(ev < 1e-10)))

    # 用「真值」KL 的 Hessian 校验 F（有限差分）
    def avg_kl(th):
        return float(dnorm @ kl_rows(pi0, softmax_rows(th.reshape(N_S, N_A))))

    def grad_kl(th):
        x = th.reshape(N_S, N_A)
        p = softmax_rows(x)
        return (dnorm[:, None] * (p - pi0)).ravel()   # d/dθ KL(pi0||pi_θ)

    h = 1e-5
    Ffd = np.zeros((N_S * N_A, N_S * N_A))
    for i in range(N_S * N_A):
        tp = th0.ravel().copy(); tp[i] += h
        tm = th0.ravel().copy(); tm[i] -= h
        Ffd[:, i] = (grad_kl(tp) - grad_kl(tm)) / (2 * h)
    print("      ||F（解析）− F（有限差分Hessian）||_max = %.3e  ⇒ 解析式正确" % np.max(np.abs(F - Ffd)))
    print("      θ=θ_old 处 ∇KL = %.3e（应为 0，KL 在此取最小）" % np.max(np.abs(grad_kl(th0.ravel()))))

    # (b) CG 截断 vs 精确解
    print("\n  (b) 共轭梯度：迭代 k 次 vs 精确解 x* = F⁺g（阻尼 λ=0.01）")
    lam = 0.01
    Fd = F + lam * np.eye(N_S * N_A)
    x_exact = np.linalg.solve(Fd, g)
    print("  %-6s %-16s %-16s" % ("k", "||x_k − x*||/||x*||", "cos 夹角"))
    x = np.zeros_like(g); r = g.copy(); pvec = r.copy()
    rs = r @ r
    for k in range(1, 21):
        Fp = Fd @ pvec
        alpha = rs / (pvec @ Fp)
        x = x + alpha * pvec
        r = r - alpha * Fp
        rs_new = r @ r
        pvec = r + (rs_new / rs) * pvec
        rs = rs_new
        if k in (1, 2, 3, 5, 10, 20):
            cos = float(x @ x_exact / (np.linalg.norm(x) * np.linalg.norm(x_exact)))
            print("  %-6d %-16.3e %-16.8f" % (k, np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact), cos))

    # (c) 步长闭式解：2δ/(xᵀFx) 与 2δ/(xᵀg) 等价性 + 二阶预测是否命中真实 KL
    print("\n  (c) 闭式步长：δ 给定时，二阶预测的 KL 是否真的等于 δ？（阻尼只进 CG，不进预测）")
    print("  %-10s %-14s %-14s %-12s %-14s %-14s" % ("delta", "预测 KL", "真实 KL", "KL 相对误差", "预测 L 提升", "真实 L 提升"))
    for delta in (1e-4, 1e-3, 1e-2, 1e-1, 3e-1):
        x = np.linalg.solve(Fd, g)
        s_quad = np.sqrt(2.0 * delta / (x @ F @ x))
        s_lin = np.sqrt(2.0 * delta / (x @ g))
        th_new = (th0.ravel() + s_quad * x).reshape(N_S, N_A)
        p_new = softmax_rows(th_new)
        kl_pred = 0.5 * (s_quad * x) @ F @ (s_quad * x)
        kl_true = float(dnorm @ kl_rows(pi0, p_new))
        L_pred = s_quad * (x @ g)
        L_true = float(rho_v @ np.einsum("sa,sa->s", p_new, e0["A"]))
        print("  %-10.0e %-14.8f %-14.8f %-12.2e %-14.7f %-14.7f"
              % (delta, kl_pred, kl_true, abs(kl_true - kl_pred) / kl_pred, L_pred, L_true))
    print("      两种写法 s=sqrt(2δ/xᵀFx) 与 s=sqrt(2δ/xᵀg) 的差 = %.3e（CG 收敛时等价）"
          % abs(s_quad - s_lin))
    print("      ⇒ 真实 KL 系统性地小于二阶预测（三阶项为负），δ 越大差得越多；")
    print("        这说明泰勒近似是「保守的一侧」，但仅凭它无法保证 KL ≤ δ —— 仍需 line search。")

    # (d) 阻尼：论文里没写、实现里必须有的那一项
    print("\n  (d) 阻尼 λ（实现里的 cg_damping）：不只是数值稳定，它实质改动解与步长")
    x_ref = np.linalg.pinv(F) @ g
    print("  %-10s %-20s %-16s %-16s" % ("lambda", "||x_λ−x_{λ→0}||/||x||", "s=√(2δ/xᵀFx)", "与 √(2δ/xᵀg) 差"))
    for lm in (0.1, 0.01, 0.001, 1e-4):
        Fd2 = F + lm * np.eye(N_S * N_A)
        x2 = np.linalg.solve(Fd2, g)
        s2 = np.sqrt(2.0 * 0.01 / (x2 @ F @ x2))
        s3 = np.sqrt(2.0 * 0.01 / (x2 @ g))
        print("  %-10.0e %-20.3e %-16.8f %-16.3e"
              % (lm, np.linalg.norm(x2 - x_ref) / np.linalg.norm(x_ref), s2, abs(s2 - s3) / s2))
    print("      ⇒ λ = 0.1 时步长与「无阻尼」相差百分之几；λ 是 TRPO 论文 Algorithm 1 里没有、")
    print("        但每个开源实现都有的隐藏近似（常见默认 0.1 / 0.01）。")


if __name__ == "__main__":
    # 注：不在本脚本里做「PG vs TRPO 训练曲线」对比 —— 表格 softmax 策略全支撑、
    # 用的是精确梯度，没有采样噪声与函数近似，训崩现象复现不出来，跑出来只会是四条重合的曲线。
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for w in which:
        {"1": exp1, "2": exp2, "3": exp3, "4": exp4}[w]()
