#!/usr/bin/env python3
"""
L5: 连续动作的"采样"——DDPM / DDIM / flow matching / CVAE 先验。

LLM 的采样是「一个 argmax 或 multinomial 调用」；
VLA 里连续动作的采样是「一次数值积分」或「一次 latent 重参数化」。
本脚本在同一套一维高斯混合目标上把四者摆在一起，用解析 score / 解析速度场
（不训练任何网络）把每一步都算准，从而把「算法本身」和「拟合误差」分开看。

目标分布: 0.35*N(-2, 0.5^2) + 0.65*N(2, 0.4^2)

纯 CPU 可跑。

用法:
    python l5_vla_transport.py
    python l5_vla_transport.py --selftest
"""

import argparse
import math

import torch

torch.manual_seed(0)

MU = torch.tensor([-2.0, 2.0], dtype=torch.float64)
SIGMA = torch.tensor([0.5, 0.4], dtype=torch.float64)
W = torch.tensor([0.35, 0.65], dtype=torch.float64)

T_DIFF = 500
BETAS = torch.linspace(1e-4, 0.02, T_DIFF, dtype=torch.float64)
ALPHAS = 1.0 - BETAS
ABAR = torch.cumprod(ALPHAS, 0)


# --------------------------------------------------------------- 解析边缘量


def target_moments() -> tuple[float, float]:
    m = float((W * MU).sum())
    v = float((W * (SIGMA**2 + MU**2)).sum() - m**2)
    return m, math.sqrt(v)


def diffusion_score(x: torch.Tensor, t: int) -> torch.Tensor:
    """p_t 仍是高斯混合，直接写出 ∇ log p_t(x)。"""
    ab = ABAR[t]
    means = ab.sqrt() * MU
    stds = (ab * SIGMA**2 + (1 - ab)).sqrt()
    logw = torch.log(W)[None, :] - 0.5 * ((x[:, None] - means[None, :]) / stds[None, :]) ** 2 - torch.log(stds)[None, :]
    gamma = torch.softmax(logw, dim=-1)
    return -(gamma * (x[:, None] - means[None, :]) / stds[None, :] ** 2).sum(-1)


def eps_theta(x: torch.Tensor, t: int) -> torch.Tensor:
    """ε_θ = -sqrt(1-ᾱ_t) · score，即 DDPM 参数化下的噪声预测。"""
    return -(1 - ABAR[t]).sqrt() * diffusion_score(x, t)


# ------------------------------------------------------------------- 采样器


def sample_ddpm(n: int, steps: int | None = None) -> torch.Tensor:
    """DDPM 祖先采样。steps=None 表示用全部 T_DIFF 步。"""
    gen = torch.Generator().manual_seed(2026)
    x = torch.randn(n, dtype=torch.float64, generator=gen)
    if steps is None:
        idx = list(range(T_DIFF - 1, -1, -1))
    else:
        idx = torch.linspace(T_DIFF - 1, 0, steps).round().to(torch.int64).tolist()
    for t in idx:
        eps = eps_theta(x, t)
        x0 = (x - (1 - ABAR[t]).sqrt() * eps) / ABAR[t].sqrt()
        if t == 0:
            x = x0
            break
        c0 = ABAR[t - 1].sqrt() * BETAS[t] / (1 - ABAR[t])
        c1 = ALPHAS[t].sqrt() * (1 - ABAR[t - 1]) / (1 - ABAR[t])
        mean = c0 * x0 + c1 * x
        var = BETAS[t] * (1 - ABAR[t - 1]) / (1 - ABAR[t])
        z = torch.randn(n, dtype=torch.float64, generator=gen)
        x = mean + var.sqrt() * z
    return x


def sample_ddim(n: int, steps: int = 20, eta: float = 0.0, x_init: torch.Tensor | None = None,
                seed: int = 2026) -> torch.Tensor:
    """DDIM。eta=0 完全确定性；eta=1 退化为 DDPM 的方差。"""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dtype=torch.float64, generator=gen) if x_init is None else x_init.clone()
    idx = torch.linspace(T_DIFF - 1, 0, steps).round().to(torch.int64).tolist()
    for k, t in enumerate(idx):
        eps = eps_theta(x, t)
        x0 = (x - (1 - ABAR[t]).sqrt() * eps) / ABAR[t].sqrt()
        if t == 0:
            x = x0
            break
        t_prev = idx[k + 1]
        sigma = eta * ((1 - ABAR[t_prev]) / (1 - ABAR[t])).sqrt() * (1 - ABAR[t] / ABAR[t_prev]).sqrt()
        dir_coef = (1 - ABAR[t_prev] - sigma**2).clamp(min=0.0).sqrt()
        x = ABAR[t_prev].sqrt() * x0 + dir_coef * eps
        if eta > 0:
            x = x + sigma * torch.randn(n, dtype=torch.float64, generator=gen)
    return x


def fm_velocity(x: torch.Tensor, tau: float) -> torch.Tensor:
    """
    线性-Gaussian 路径 x_tau = tau*x1 + (1-tau)*eps 下的解析边缘速度场。

    v(x,tau) = (E[x1 | x_tau=x] - x) / (1 - tau)
    """
    m = tau * MU
    s2 = (1 - tau) ** 2 + tau**2 * SIGMA**2
    s = s2.sqrt()
    logw = torch.log(W)[None, :] - 0.5 * ((x[:, None] - m[None, :]) / s[None, :]) ** 2 - torch.log(s)[None, :]
    gamma = torch.softmax(logw, dim=-1)
    # 每个模态下 E[x1 | x_tau] 的线性后验
    coef = (SIGMA**2 * tau) / s2
    m_k = MU[None, :] + coef[None, :] * (x[:, None] - m[None, :])
    x1_hat = (gamma * m_k).sum(-1)
    return (x1_hat - x) / (1 - tau)


def sample_flow(n: int, steps: int = 10, x_init: torch.Tensor | None = None, seed: int = 2026) -> torch.Tensor:
    """
    π0 式前向 Euler：tau 从 0（纯噪声）积分到 1（干净动作）。

    注意 π0 论文把目标速度场写作 u = eps - A_t，同时把 noisy action 定义为
    A^tau = tau*A + (1-tau)*eps 并从 tau=0 积到 tau=1；本脚本用自洽的
    v = (E[x1|x_tau] - x_tau)/(1-tau)，即 dA^tau/dtau，方向与之相反但终态相同。
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dtype=torch.float64, generator=gen) if x_init is None else x_init.clone()
    dt = 1.0 / steps
    tau = 0.0
    for _ in range(steps):
        v = fm_velocity(x, tau)
        x = x + dt * v
        tau = min(tau + dt, 1.0 - 1e-6)
    return x


# ------------------------------------------------------------------ 统计量


def summarize(name: str, x: torch.Tensor) -> tuple[float, float]:
    m, s = float(x.mean()), float(x.std())
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    q = torch.quantile(x, qs).tolist()
    print(f"    {name:<26} 均值={m:+8.4f} 标准差={s:7.4f}  "
          + "  ".join(f"q{int(p*100)}={v:+.3f}" for p, v in zip(qs.tolist(), q)))
    return m, s


def target_quantiles() -> list[float]:
    """用大样本蒙特卡洛给出目标分布的分位数（作为参照）。"""
    gen = torch.Generator().manual_seed(999)
    comp = (torch.rand(400_000, generator=gen, dtype=torch.float64) > W[0]).long()
    x = MU[comp] + SIGMA[comp] * torch.randn(400_000, generator=gen, dtype=torch.float64)
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    return torch.quantile(x, qs).tolist()


def tv_hist(a: torch.Tensor, b: torch.Tensor, lo: float = -5.0, hi: float = 5.0, bins: int = 80) -> float:
    ha = torch.histc(a.float(), bins=bins, min=lo, max=hi)
    hb = torch.histc(b.float(), bins=bins, min=lo, max=hi)
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return 0.5 * float((ha - hb).abs().sum())


# --------------------------------------------------------------------- 各节


def section_target() -> None:
    print("=" * 96)
    print("[0] 目标分布与参照")
    print("=" * 96)
    m, s = target_moments()
    print(f"    p_data = 0.35*N(-2, 0.5^2) + 0.65*N(2, 0.4^2)  (双模态，故意不对称)")
    print(f"    解析均值 = {m:+.4f}   解析标准差 = {s:.4f}")
    print(f"    解析分位数 (大样本 MC 参照) = {[round(v,4) for v in target_quantiles()]}")
    print()
    gen = torch.Generator().manual_seed(999)
    comp = (torch.rand(400_000, generator=gen, dtype=torch.float64) > W[0]).long()
    ref = MU[comp] + SIGMA[comp] * torch.randn(400_000, generator=gen, dtype=torch.float64)
    summarize("目标分布 (400k 样本)", ref)
    print()
    return ref


def section_ddpm(ref: torch.Tensor) -> None:
    print("=" * 96)
    print("[1] DDPM 祖先采样：T=500 全步 vs 20 步子序列")
    print("=" * 96)
    for steps in (None, 50, 20, 10):
        x = sample_ddpm(40_000, steps=steps)
        tag = "全 500 步" if steps is None else f"抽 {steps} 步"
        m, s = summarize(f"DDPM {tag:<12}", x)
        print(f"      {'':<26} 与目标直方图 TV = {tv_hist(x, ref):.4f}")
    print()


def section_ddim(ref: torch.Tensor) -> None:
    print("=" * 96)
    print("[2] DDIM：eta=0 的确定性")
    print("=" * 96)
    n = 20_000
    gen = torch.Generator().manual_seed(42)
    x0 = torch.randn(n, dtype=torch.float64, generator=gen)

    a = sample_ddim(n, steps=20, eta=0.0, x_init=x0)
    b = sample_ddim(n, steps=20, eta=0.0, x_init=x0)
    c = sample_ddim(n, steps=20, eta=1.0, x_init=x0)
    print(f"    同一初始噪声，eta=0 跑两次：逐元素完全相同 = {bool(torch.equal(a, b))}")
    print(f"    同一初始噪声，eta=1 跑两次：逐元素完全相同 = {bool(torch.equal(a, c))}")
    print(f"    eta=0 与 eta=1 结果的最大差异 = {float((a-c).abs().max()):.4f}")
    print()
    for eta in (0.0, 0.5, 1.0):
        for steps in (10, 20, 50):
            x = sample_ddim(40_000, steps=steps, eta=eta)
            summarize(f"DDIM eta={eta} steps={steps:<3}", x)
            print(f"      {'':<26} 与目标直方图 TV = {tv_hist(x, ref):.4f}")
        print()


def section_flow(ref: torch.Tensor) -> None:
    print("=" * 96)
    print("[3] Flow matching（π0 式）：tau 从 0（噪声）积到 1（动作）")
    print("=" * 96)
    n = 20_000
    gen = torch.Generator().manual_seed(42)
    x0 = torch.randn(n, dtype=torch.float64, generator=gen)
    a = sample_flow(n, steps=10, x_init=x0)
    b = sample_flow(n, steps=10, x_init=x0)
    print(f"    同一初始噪声跑两次：逐元素完全相同 = {bool(torch.equal(a, b))}  (ODE, 无随机性)")
    print()
    for steps in (1, 2, 5, 10, 20, 100):
        x = sample_flow(40_000, steps=steps)
        summarize(f"Flow steps={steps:<4}", x)
        print(f"      {'':<26} 与目标直方图 TV = {tv_hist(x, ref):.4f}")
    print()
    print("    注意: 步数越小越像「把噪声整体平移」，多模态结构会被积分误差抹平；")
    print("          10 步是 π0 的默认值，在 1D 混合上已经能恢复双峰。")
    print()


def section_cvae() -> None:
    print("=" * 96)
    print("[4] CVAE 的 latent 采样：训练用后验，推理用什么")
    print("=" * 96)
    print("    构造 q(z|x) ~ N(mu_phi(x), sigma_phi(x)^2) 的可观测后验（不训练网络），")
    print("    比较推理时三种 latent 取法对输出分布的影响：z~q(z|x) / z=0 / z~N(0,1)。")
    print()

    n = 60_000
    gen = torch.Generator().manual_seed(5)
    comp = (torch.rand(n, generator=gen, dtype=torch.float64) > W[0]).long()
    a_true = MU[comp] + SIGMA[comp] * torch.randn(n, generator=gen, dtype=torch.float64)
    cond = torch.randn(n, generator=gen, dtype=torch.float64)  # 观测条件

    # 一个"训练好"的后验：mu 与 z=0 时的基准动作 a0 相关，sigma 随 |cond| 变化
    a0 = torch.where(cond > 0, MU[1], MU[0]) + 0.8 * SIGMA[comp] * cond
    mu = 0.6 * (a_true - a0)
    log_sigma = -1.2 + 0.15 * cond.abs()
    sigma = log_sigma.exp()

    def decode(z: torch.Tensor) -> torch.Tensor:
        return a0 + z / 0.6

    z_post = mu + sigma * torch.randn(n, generator=gen, dtype=torch.float64)
    z_zero = torch.zeros_like(mu)
    z_prior = torch.randn(n, generator=gen, dtype=torch.float64)

    out_post, out_zero, out_prior = decode(z_post), decode(z_zero), decode(z_prior)
    print(f"    {'latent 取法':<22} {'输出均值':>10} {'输出标准差':>12} {'与目标 TV':>12} {'命中真值误差':>14}")
    for tag, z, out in [("z ~ q(z|x) 训练同款", z_post, out_post),
                        ("z = 0       (ACT 做法)", z_zero, out_zero),
                        ("z ~ N(0,1)  (纯先验)", z_prior, out_prior)]:
        err = float((out - a_true).abs().mean())
        print(f"    {tag:<22} {float(out.mean()):>10.4f} {float(out.std()):>12.4f}"
              f" {tv_hist(out, a_true, -6, 6, 60):>12.4f} {err:>14.4f}")
    print()
    kl = (0.5 * (sigma**2 + mu**2 - 1 - torch.log(sigma**2))).mean()
    print(f"    训练时 KL(q(z|x) || N(0,1)) 的均值 = {float(kl):.4f} nat")
    print(f"    z=0 与 z~q(z|x) 的输出平均差 = {float((out_zero-out_post).abs().mean()):.4f}")
    print(f"    z~N(0,1) 与 z~q(z|x) 的输出平均差 = {float((out_prior-out_post).abs().mean()):.4f}")
    print()
    print("    要点: 推理时把 z 固定为 0，等价于把整批动作塌到后验均值对应的单一模态；")
    print("          它在确定性、可复现上更优，但丢掉了训练时见过的 latent 多样性。")
    print()


def section_selftest() -> None:
    print("=" * 96)
    print("[-] 自校验")
    print("=" * 96)
    # 1) eta=0 确定性
    gen = torch.Generator().manual_seed(1)
    x0 = torch.randn(1000, dtype=torch.float64, generator=gen)
    a = sample_ddim(1000, steps=10, eta=0.0, x_init=x0)
    b = sample_ddim(1000, steps=10, eta=0.0, x_init=x0)
    print(f"    DDIM eta=0 两次结果完全相同 = {bool(torch.equal(a, b))}")
    # 2) flow 的 ODE 确定性
    a = sample_flow(1000, steps=5, x_init=x0)
    b = sample_flow(1000, steps=5, x_init=x0)
    print(f"    Flow 两次结果完全相同      = {bool(torch.equal(a, b))}")
    # 3) eta=1 的 DDIM 方差公式应退化为 DDPM 的 β̃_t
    t, t_prev = 300, 299
    s_ddim = ((1 - ABAR[t_prev]) / (1 - ABAR[t])).sqrt() * (1 - ABAR[t] / ABAR[t_prev]).sqrt()
    s_ddpm = (BETAS[t] * (1 - ABAR[t_prev]) / (1 - ABAR[t])).sqrt()
    print(f"    eta=1 的 sigma 与 DDPM 的 β̃_t^{{1/2}}: {float(s_ddim):.8f} vs {float(s_ddpm):.8f}"
          f"  相对差={abs(float(s_ddim)-float(s_ddpm))/float(s_ddpm):.2e}")
    # 4) 解析速度场在 tau->0 时退化为 E[x1]
    x = torch.tensor([0.0, -3.0, 3.0], dtype=torch.float64)
    v = fm_velocity(x, 1e-6)
    print(f"    tau->0 时 flow 速度 = {[round(float(z),4) for z in v]}  (应≈目标均值 {target_moments()[0]:.4f} 的位移)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    print(f"torch {torch.__version__} | 设备 cpu")
    print()
    section_selftest()
    if args.selftest:
        return
    ref = section_target()
    section_ddpm(ref)
    section_ddim(ref)
    section_flow(ref)
    section_cvae()


if __name__ == "__main__":
    main()
