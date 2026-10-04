"""
RL 符号与计算全景：GAE、优势估计、概率与重要性比的数值实验（纯 numpy，无 torch）。

配套文章：_posts/2026-09-19-rl-notation-and-estimators-deepdive.md

文章里每一张数值表都对应下面一个小节，编号与正文一致。合成数据全部做
「构造真值 <-> 反算值」自校验：MDP 里用线性方程组解出的 V^pi / Q^pi 会用
蒙特卡洛回报再核一遍，KL 的真值用全词表求和得到、与采样估计对照。

  §5.1  GAE 递归式 vs 展开式
  §5.2  四种 critic 误差形态下的偏差-方差分解
  §5.3  A_hat + V 与真实 return-to-go 的对比
  §5.4  gamma=1 下 GAE 的闭式解
  §5.5  lambda=0 / lambda=1 两端的量纲差异
  §5.6  critic 塌缩与 KL shaping
  §7.2  logprob 的数值精度（fp64/fp32/fp16/bf16/naive）
  §7.3  top-p 截断带来的 logprob 系统性偏移
  §8.1  KL 的 k1/k2/k3 三个估计器（两种漂移 regime）
  §8.2  k3 的梯度符号陷阱
  §9.1  PPO 裁剪在哪个点上切断梯度
  §9.2  长序列上 token 级重要性比的病态程度
  §10.1 GRPO / Dr.GRPO 的组内优势
  §10.2 零梯度组占比

运行：
  ~/Software/miniconda3/bin/python tools/rl_notation_lab.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 公用：随机 MDP

S, A_n, GAMMA, T, N = 10, 3, 0.95, 30, 1200


def build_mdp(seed=7):
    """构造一个随机 MDP，并用线性方程组解出真值 V^pi / Q^pi / A^pi。"""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S) * 0.25, size=(S, A_n))   # (s, a, s')
    Rw = rng.normal(0.0, 1.0, size=(S, A_n))
    pi = rng.dirichlet(np.ones(A_n) * 1.0, size=S)
    Ppi = np.einsum("sa,sat->st", pi, P)
    Rpi = np.einsum("sa,sa->s", pi, Rw)
    V = np.linalg.solve(np.eye(S) - GAMMA * Ppi, Rpi)
    Q = Rw + GAMMA * np.einsum("sat,t->sa", P, V)
    return rng, P, Rw, pi, V, Q, Q - V[:, None]


def sample_trajs(rng, P, pi, s0, a0, n):
    """从 (s0, a0) 出发采样 n 条长度为 T 的轨迹。用逆 CDF 采样（向量化）。"""
    ss = np.zeros((n, T + 1), dtype=int)
    aa = np.zeros((n, T), dtype=int)
    ss[:, 0] = s0
    aa[:, 0] = a0
    for t in range(T):
        u = rng.random(n)
        cum = np.cumsum(P[ss[:, t], aa[:, t]], axis=1)
        ss[:, t + 1] = (u[:, None] > cum).sum(axis=1)
        if t + 1 < T:
            u = rng.random(n)
            cum = np.cumsum(pi[ss[:, t + 1]], axis=1)
            aa[:, t + 1] = (u[:, None] > cum).sum(axis=1)
    return ss, aa


def gae_from_traj(Vhat, lam, ss, aa, Rw):
    """(n, T+1) 状态、(n, T) 动作 -> (n,) 的 A_hat_0。定义式直接求和。"""
    r_seq = Rw[ss[:, :-1], aa]
    Vs = Vhat[ss]
    delta = r_seq + GAMMA * Vs[:, 1:] - Vs[:, :-1]
    w = (GAMMA * lam) ** np.arange(T)
    return (w[None, :] * delta).sum(axis=1)


def gae_recursive_from_traj(Vhat, lam, ss, aa, Rw):
    r_seq = Rw[ss[:, :-1], aa]
    Vs = Vhat[ss]
    delta = r_seq + GAMMA * Vs[:, 1:] - Vs[:, :-1]
    out = np.zeros(len(ss))
    g = np.zeros(len(ss))
    for t in range(T - 1, -1, -1):
        g = delta[:, t] + GAMMA * lam * g
    out[:] = g
    return out


# ============================================================ §5.1 递归 vs 展开

def lab_5_1():
    section("§5.1  GAE 递归式与展开式的一致性")
    rng, P, Rw, pi, V, Q, Adv = build_mdp()
    ss, aa = sample_trajs(rng, P, pi, 0, 0, 400)
    for lam in [0.0, 0.5, 0.95, 1.0]:
        a = gae_from_traj(V, lam, ss, aa, Rw)
        b = gae_recursive_from_traj(V, lam, ss, aa, Rw)
        print(f"lambda={lam:.2f} : max|递归 - 展开| = {np.max(np.abs(a - b)):.3e}")


# ==================================================== §5.2 四种 critic 误差形态

def make_critics(rng, V):
    """四种 critic：完美 / 全局缩放 / 低容量聚类 / 逐状态独立噪声。"""
    sigma_V = V.std()
    cls = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])   # 按状态编号分 3 类，与 V 的大小无关
    Vclus = np.array([V[cls == k].mean() for k in range(3)])[cls]
    return {
        "① 完美 critic  V_phi = V^pi": V.copy(),
        "② 全局缩放     V_phi = 0.8 V^pi": 0.8 * V,
        "③ 低容量聚类   10 状态 -> 3 类取类内均值": Vclus,
        "④ 逐状态噪声   V_phi = V^pi + N(0,(0.15 sigma_V)^2)":
            V + rng.normal(0.0, 0.15 * sigma_V, size=S),
    }


def lab_5_2():
    section("§5.2  偏差-方差分解：四种 critic 误差形态（N=1200，逐 (s,a) 先算再平均）")
    rng, P, Rw, pi, V, Q, Adv = build_mdp()

    # 自校验 1：E_pi[A^pi] = 0
    zero = np.einsum("sa,sa->s", pi, Adv)
    print(f"[自校验] max_s |E_pi[A^pi]| = {np.abs(zero).max():.2e}  (应 ~ 0)")

    trajs = {}
    for s0 in range(S):
        for a0 in range(A_n):
            trajs[(s0, a0)] = sample_trajs(rng, P, pi, s0, a0, N)

    # 自校验 2：蒙特卡洛回报的均值应还原 V^pi
    mc = np.array([(GAMMA ** np.arange(T) * Rw[trajs[(s, 0)][0][:, :-1],
                                                trajs[(s, 0)][1]][0]).sum() for s in range(S)])
    print(f"[自校验] MC 单条轨迹回报 vs V^pi 的相关系数 = {np.corrcoef(mc, V)[0, 1]:.3f}")

    critics = make_critics(rng, V)
    lams = [0.0, 0.3, 0.6, 0.9, 0.95, 1.0]

    for name, Vhat in critics.items():
        print(f"\n--- {name}")
        print(f"{'lambda':>7} | {'bias^2':>8} | {'var':>8} | {'MSE':>8} | {'噪声地板 var/N':>14}")
        for lam in lams:
            b, v = [], []
            for (s, a), (ss, aa) in trajs.items():
                e = gae_from_traj(Vhat, lam, ss, aa, Rw) - Adv[s, a]
                b.append(e.mean())
                v.append(e.var())
            b2, var = np.mean(np.square(b)), np.mean(v)
            print(f"{lam:7.2f} | {b2:8.4f} | {var:8.4f} | {b2 + var:8.4f} | {var / N:14.4f}")


# =============================================== §5.3  A_hat + V vs return-to-go

def lab_5_3():
    section("§5.3  R_hat = A_hat + V 到底是不是 G_t（在 (s0,a0)=(0,0) 上）")
    rng, P, Rw, pi, V, Q, Adv = build_mdp()
    s0, a0 = 0, 0
    ss, aa = sample_trajs(rng, P, pi, s0, a0, 4000)
    G0 = (GAMMA ** np.arange(T) * Rw[ss[:, :-1], aa]).sum(axis=1)
    print(f"真 G_0: mean={G0.mean():+.3f}  std={G0.std():.3f} | "
          f"V^pi(s0)={V[s0]:+.3f} | A^pi(s0,a0)={Adv[s0, a0]:+.3f}")

    def gae_terminate(Vhat, lam):
        """episode 真正终止：末步 bootstrap 值取 0。"""
        r_seq = Rw[ss[:, :-1], aa]
        Vs = Vhat[ss].copy()
        Vs[:, -1] = 0.0
        delta = r_seq + GAMMA * Vs[:, 1:] - Vs[:, :-1]
        return ((GAMMA * lam) ** np.arange(T)[None, :] * delta).sum(axis=1)

    for lam in [0.0, 0.5, 0.9, 0.95, 1.0]:
        Rh = gae_terminate(V, lam) + V[s0]
        print(f"lambda={lam:.2f} : mean(R_0)={Rh.mean():+.3f}  "
              f"bias vs E[G_0]={Rh.mean() - G0.mean():+.4f}  "
              f"corr(R_0,G_0)={np.corrcoef(Rh, G0)[0, 1]:.4f}")

    # 对照：若用 critic 的 V(s_T) 做 bootstrap（截断而非终止），lambda=1 会多出 gamma^T V(s_T)
    Rh_tr = gae_from_traj(V, 1.0, ss, aa, Rw) + V[s0]
    print(f"\n[对照] 末步用 critic 的 V(s_T) bootstrap：lambda=1 时 "
          f"mean(R_0 - G_0) = {np.mean(Rh_tr - G0):+.4f}，"
          f"corr = {np.corrcoef(Rh_tr, G0)[0, 1]:.4f}")
    print(f"       理論上多出 gamma^T V(s_T)，gamma^T = {GAMMA ** T:.4f}，"
          f"E[gamma^T V(s_T)] = {GAMMA ** T * V[ss[:, -1]].mean():+.4f}")


# ================================================= §5.4  gamma=1 下 GAE 的闭式解

def lab_5_4():
    section("§5.4  gamma=1、仅终端奖励时 GAE 的闭式解（T=12）")
    rng = np.random.default_rng(23)
    Tl, Rw_end = 12, 1.0
    Vv = np.sort(rng.uniform(0.2, 0.9, size=Tl))

    def rec(lam):
        Vs = np.append(Vv, 0.0)
        r = np.zeros(Tl)
        r[-1] = Rw_end
        delta = r + Vs[1:] - Vs[:-1]
        out = np.zeros(Tl)
        g = 0.0
        for t in range(Tl - 1, -1, -1):
            g = delta[t] + lam * g
            out[t] = g
        return out

    def closed(lam):
        out = np.zeros(Tl)
        for t in range(Tl):
            m = Tl - 1 - t
            s = sum((lam ** j) * Vv[t + 1 + j] for j in range(m)) if m > 0 else 0.0
            out[t] = -Vv[t] + (1 - lam) * s + (lam ** m) * Rw_end
        return out

    print("V =", np.round(Vv, 3))
    for lam in [0.0, 0.3, 0.6, 0.9, 0.95, 1.0]:
        print(f"lambda={lam:.2f} : max|递归 - 闭式| = {np.max(np.abs(rec(lam) - closed(lam))):.3e}")
    return Vv


# ================================================= §5.5  lambda=0 / lambda=1 两端

def lab_5_5(Vv):
    section("§5.5  两个极端的量纲：lambda=0 逐 token 增量 vs lambda=1 整段重算（T=12, R=1）")
    Tl = len(Vv)
    a0 = np.append(Vv[1:], 1.0) - Vv          # lambda=0: V_{t+1} - V_t（末步是 R - V）
    a0[-1] = 1.0 - Vv[-1]
    a1 = 1.0 - Vv                              # lambda=1: R - V_t
    print(f"{'t':>3} | {'V(s_t)':>7} | {'lam=0':>8} | {'lam=1':>8}")
    for t in range(Tl):
        print(f"{t:3d} | {Vv[t]:7.3f} | {a0[t]:+8.4f} | {a1[t]:+8.4f}")
    print(f"sum_t A_hat :  lambda=0 -> {a0.sum():.4f} (= R - V_0 = {1.0 - Vv[0]:.4f})")
    print(f"               lambda=1 -> {a1.sum():.4f}   （是 lambda=0 的 {a1.sum() / a0.sum():.1f} 倍）")


# ============================================ §5.6  critic 塌缩与 KL shaping

def lab_5_6():
    section("§5.6  critic 塌缩成常数 c 时，信号如何从句尾衰减")
    Tl, Rw_end = 12, 1.0
    for c in [0.5, 1.0]:
        delta = np.zeros(Tl)
        delta[-1] = Rw_end - c
        for lam in [0.0, 0.95]:
            g, out = 0.0, np.zeros(Tl)
            for t in range(Tl - 1, -1, -1):
                g = delta[t] + lam * g
                out[t] = g
            cf = (lam ** (Tl - 1 - np.arange(Tl))) * (Rw_end - c)
            print(f"c={c}  lam={lam:.2f} : GAE={np.round(out, 4)}")
            print(f"{'':>16} 闭式 lam^(T-1-t)(R-c) 的最大偏差 = {np.max(np.abs(out - cf)):.2e}")

    print("\n--- 加入逐 token 的 KL shaping 奖励后（r_t = -beta*KL_t，beta*KL_t ~ U(0,0.1)）")
    rng = np.random.default_rng(23)
    Vv = np.sort(rng.uniform(0.2, 0.9, size=Tl))
    kl = rng.uniform(0.0, 0.1, size=Tl)
    r = -kl
    r[-1] += Rw_end
    Vs = np.append(Vv, 0.0)
    for lam in [0.0, 0.95, 1.0]:
        delta = r + Vs[1:] - Vs[:-1]
        g, out = 0.0, np.zeros(Tl)
        for t in range(Tl - 1, -1, -1):
            g = delta[t] + lam * g
            out[t] = g
        print(f"lambda={lam:.2f} : A[:6] = {np.round(out[:6], 4)}  |  "
              f"非零 token 数 = {int((np.abs(out) > 1e-12).sum())}/{Tl}")


# ==================================================== §7.2  logprob 的数值精度

def lab_7_2():
    section("§7.2  logprob 的数值精度（logits ~ N(0,8)，词表 128000）")
    rng = np.random.default_rng(11)
    lg = rng.normal(0, 8, size=128000)

    def lse(x):
        m = x.max()
        return x - (m + np.log(np.exp(x - m).sum()))

    def to_bf16(x32):
        return (np.asarray(x32, dtype=np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)

    ref = float(lse(lg.astype(np.float64))[0])
    rows = [
        ("fp64", lse(lg.astype(np.float64))[0]),
        ("fp32", lse(lg.astype(np.float32).astype(np.float64))[0]),
        ("fp16", lse(lg.astype(np.float16).astype(np.float64))[0]),
        ("bf16（尾数截断模拟）", float(lse(to_bf16(lg).astype(np.float64))[0])),
    ]
    for name, val in rows:
        print(f"{name:<22} {val:+.10f}   与 fp64 之差 = {abs(val - ref):.2e}")

    # naive：先 exp 再 log，用一个小词表演示（V=10，正确答案 = -ln 10）
    z = np.full(10, 800.0)
    try:
        with np.errstate(over="ignore"):
            naive = float(np.log(np.exp(z).sum()))
    except Exception:
        naive = float("nan")
    print(f"\nnaive log(sum(exp))  logits=800, V=10 -> logprob = {naive}   （溢出）")
    print(f"稳定版                            -> logprob = {np.log(1.0 / 10):+.4f} = -ln 10")
    print(f"（V=128000 时正确答案是 -ln 128000 = {np.log(1.0 / 128000):+.4f}）")


# ==================================================== §7.3  top-p 截断的偏移

def lab_7_3():
    section("§7.3  top-p 截断分布与完整分布的 logprob 之差")
    rng = np.random.default_rng(11)
    z = rng.normal(0, 3, size=128000)      # sd=8 会让分布塌到单个 token 上，不真实
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    print(f"分布形态：p_max = {p.max():.4f}（sd=3 的 logits，词表 128000）")
    print(f"{'top-p':>6} | {'截断集合大小 |S|':>16} | {'集合内概率质量':>14} | {'log 差 -ln(mass)':>16}")
    for tp in [1.00, 0.95, 0.90, 0.80]:
        k = min(int(np.searchsorted(cum, tp)) + 1, len(p) - 1)
        mass = cum[k - 1]
        print(f"{tp:6.2f} | {k:16d} | {mass:14.4f} | {-np.log(mass):16.4f}")


# ==================================================== §8.1  KL 的三个估计器

def lab_8_1():
    section("§8.1  KL 的 k1 / k2 / k3（词表 2000，40 万样本；真值用全词表求和）")
    Vv = 2000

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    for tag, sd in [("A 轻微漂移 (logit 噪声 sd=0.15)", 0.15),
                    ("B 大幅漂移 (logit 噪声 sd=1.20)", 1.20)]:
        rng = np.random.default_rng(11)          # 两个 regime 各自独立
        q = rng.dirichlet(np.ones(Vv) * 0.8)
        logits_p = np.log(q) + rng.normal(0, sd, size=Vv)
        p = softmax(logits_p)
        true_kl = float(np.sum(p * np.log(p / q)))
        idx = rng.choice(Vv, size=400000, p=p)
        rho_ref = q[idx] / p[idx]           # pi_ref / pi_theta
        k1 = -np.log(rho_ref)
        k2 = 0.5 * np.log(rho_ref) ** 2
        k3 = rho_ref - np.log(rho_ref) - 1
        print(f"\n--- regime {tag}   真 KL = {true_kl:.5f}")
        print(f"{'':>4} | {'均值':>9} | {'偏差':>9} | {'标准差':>9} | {'负值占比':>8}")
        for nm, k in [("k1", k1), ("k2", k2), ("k3", k3)]:
            print(f"{nm:>4} | {k.mean():9.5f} | {k.mean() - true_kl:+9.5f} | "
                  f"{k.std():9.5f} | {(k < 0).mean() * 100:7.1f}%")


# ==================================================== §8.2  k3 的梯度符号陷阱

def lab_8_2():
    section("§8.2  KL 项进梯度的系数：正确值 -log(rho_ref) vs stop-gradient 后的 k3")
    print(f"{'rho_ref':>8} | {'正确 -log r':>12} | {'k3':>10} | {'k2':>10} | 判定")
    for r_ in [0.25, 0.50, 0.80, 1.00, 1.25, 2.00, 4.00]:
        corr, k3, k2 = -np.log(r_), r_ - np.log(r_) - 1, 0.5 * np.log(r_) ** 2
        same = "" if abs(r_ - 1) < 1e-12 else (
            "符号相反" if corr * k3 < 0 else "同号但量级差 %.1fx" % (corr / k3 if k3 else float("nan")))
        print(f"{r_:8.2f} | {corr:+12.4f} | {k3:+10.4f} | {k2:+10.4f} | {same}")


# ==================================================== §9.1  PPO 裁剪切断点

def lab_9_1():
    section("§9.1  PPO 裁剪在哪个 ratio 上切断梯度（eps=0.2，中心差分）")
    eps = 0.2
    obj = lambda r_, adv: np.minimum(r_ * adv, np.clip(r_, 1 - eps, 1 + eps) * adv)
    grad = lambda r_, adv, h=1e-6: (obj(r_ + h, adv) - obj(r_ - h, adv)) / (2 * h)
    for adv in [+1.0, -1.0]:
        print(f"\nA = {adv:+.0f}（{'好动作' if adv > 0 else '坏动作'}）")
        print(f"{'rho':>6} | {'loss':>9} | {'d loss/d rho':>13} | 状态")
        grid = [0.5, 1.0, 1.19, 1.21, 3.0] if adv > 0 else [0.3, 0.79, 0.81, 1.2, 2.0]
        for r_ in grid:
            g = grad(r_, adv)
            state = "梯度被切断" if abs(g) < 1e-3 else "正常更新"
            print(f"{r_:6.2f} | {obj(r_, adv):9.4f} | {g:13.4f} | {state}")


# ==================================================== §9.2  长序列 token 级 ratio

def v_size(v):
    return abs(float(v))


def lab_9_2():
    section("§9.2  长序列上 token 级 ratio 的病态程度（log ratio ~ N(mu, 0.05^2)）")
    rng = np.random.default_rng(11)
    print(f"{'T':>5} | {'mu':>5} | {'序列 ratio 中位数':>18} | {'p99':>12} | {'P(rho>2)':>9} | {'GSPO 中位数':>11}")
    for Tl in [64, 256, 1024, 4096]:
        for mu in [0.00, 0.01]:
            lr = rng.normal(mu, 0.05, size=(200000, Tl))
            seq = np.exp(lr.sum(axis=1))
            med, p99 = np.median(seq), np.percentile(seq, 99)
            fmt = (lambda v: f"{v:.3f}") if v_size(med) < 1e5 else (lambda v: f"{v:.2e}")
            print(f"{Tl:5d} | {mu:5.2f} | {fmt(med):>18} | {fmt(p99):>12} | "
                  f"{(seq > 2).mean():9.3f} | {np.median(seq ** (1.0 / Tl)):11.4f}")


# ==================================================== §10.1 GRPO / Dr.GRPO

def lab_10_1():
    section("§10.1  G=8、二值奖励下组内 k 条答对时的 advantage")
    G = 8
    print(f"{'k':>3} | {'GRPO 正确':>10} | {'GRPO 错误':>10} | {'Dr.GRPO 正确':>12} | {'Dr.GRPO 错误':>12}")
    for k in range(1, G):
        mu = k / G
        sd = np.sqrt(mu * (1 - mu))
        print(f"{k:3d} | {(1 - mu) / sd:10.4f} | {(0 - mu) / sd:10.4f} | "
              f"{1 - mu:12.4f} | {0 - mu:12.4f}")
    print(f"\n难题(k=1) vs 中等题(k=4) 的正样本权重比："
          f"GRPO {np.sqrt(7 / 1) / np.sqrt(4 / 4):.3f}x，Dr.GRPO {0.875 / 0.5:.3f}x")


# ==================================================== §10.2 零梯度组

def lab_10_2():
    section("§10.2  组内奖励全同时的零梯度组占比")
    print(f"{'pass rate':>10} | {'G=8':>8} | {'G=16':>8}")
    for pr in [0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99]:
        print(f"{pr:10.2f} | {(pr ** 8 + (1 - pr) ** 8) * 100:7.2f}% | "
              f"{(pr ** 16 + (1 - pr) ** 16) * 100:7.2f}%")


if __name__ == "__main__":
    lab_5_1()
    lab_5_2()
    lab_5_3()
    Vv = lab_5_4()
    lab_5_5(Vv)
    lab_5_6()
    lab_7_2()
    lab_7_3()
    lab_8_1()
    lab_8_2()
    lab_9_1()
    lab_9_2()
    lab_10_1()
    lab_10_2()
    print("\n完成。")
