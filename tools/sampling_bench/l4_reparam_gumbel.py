#!/usr/bin/env python3
"""
L4: 重参数化与 Gumbel-Softmax —— 梯度怎么穿过一次采样。

三件事：
  1. Distribution.sample() 默认转发到 rsample() 并套 no_grad；rsample 的
     loc + eps*scale 是可微的。用 gradcheck 验证，并对比 sample 的梯度为 None。
  2. ExpRelaxedCategorical.rsample 用 -log(-log(U)) 造 Gumbel；F.gumbel_softmax
     用 -log(Exp(1)) 造 Gumbel。同一个分布两条数值路径，验证它们是否逐元素相同。
  3. hard=True 的 straight-through 装修（y_hard - y_soft.detach() + y_soft）
     梯度确实等于 y_soft 的梯度；且它的 argmax 就是真正的离散采样。

纯 CPU 可跑。

用法:
    python l4_reparam_gumbel.py
    python l4_reparam_gumbel.py --selftest
"""

import argparse
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def section_rsample() -> None:
    print("=" * 92)
    print("[1] sample() 与 rsample()：梯度能不能穿过采样")
    print("=" * 92)
    loc = torch.tensor([0.5, -1.0], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([1.5, 0.5], dtype=torch.float64, requires_grad=True)

    d = torch.distributions.Normal(loc, scale)
    print(f"    has_rsample = {d.has_rsample}")
    print()

    s = d.sample((3,))
    print(f"    sample()  输出 requires_grad = {s.requires_grad}  (基类走 with torch.no_grad())")
    r = d.rsample((3,))
    print(f"    rsample() 输出 requires_grad = {r.requires_grad}  (loc + eps*scale 保留计算图)")
    print()

    loss = r.sum()
    g_loc, g_scale = torch.autograd.grad(loss, [loc, scale])
    print(f"    d(sum rsample)/d(loc)   = {g_loc.tolist()}   (解析解 = 采样个数 3.0)")
    print(f"    d(sum rsample)/d(scale) = {g_scale.tolist()}  (解析解 = sum(eps))")
    print()

    print("    把 rsample 展开：rsample 到底是不是 loc + eps*scale")
    torch.manual_seed(4242)
    r2 = d.rsample((3,))
    torch.manual_seed(4242)
    eps = torch.empty(3, 2, dtype=torch.float64).normal_()
    expand_ok = bool(torch.allclose(r2, loc.detach() + eps * scale.detach()))
    print(f"      重新播种后手动复现 eps，逐元素相等 = {expand_ok}")
    print(f"      eps =\n{eps}")
    print()
    print("    gradcheck：直接对 rsample 做会失败（每次调用都重抽 eps，数值 Jacobian 无意义）")
    try:
        torch.autograd.gradcheck(
            lambda l, s_: torch.distributions.Normal(l, s_).rsample((4,)).sum(),
            (loc.clone().requires_grad_(True), scale.clone().requires_grad_(True)),
        )
        direct = "通过了"
    except Exception as e:  # noqa: BLE001
        direct = f"失败（{type(e).__name__}）"
    print(f"      gradcheck(rsample) 直接调用 = {direct}")
    print("      原因：rsample 不是 loc/scale 的确定性函数，扰动输入时不变量 eps 也变了。")
    print()
    print("    正确做法：固定 eps 后对展开式做 gradcheck")
    ok = torch.autograd.gradcheck(
        lambda l, s_: (l + eps * s_).sum(), (loc.clone().requires_grad_(True), scale.clone().requires_grad_(True)),
        eps=1e-6, atol=1e-8,
    )
    print(f"      gradcheck(loc + eps*scale) = {ok}")
    print("      结论：重参数化把随机性完全推到 eps 上，loc/scale 走的是普通确定性微分。")
    print()


def section_two_gumbel_paths() -> None:
    print("=" * 92)
    print("[2] 两条造 Gumbel 噪声的路径是否逐元素相同")
    print("=" * 92)
    n = 100_000
    g = torch.Generator().manual_seed(2026)

    u = torch.rand(n, dtype=torch.float64, generator=g)
    u_clamped = torch.distributions.utils.clamp_probs(u)
    path_u = -torch.log(-torch.log(u_clamped))                       # ExpRelaxedCategorical 的写法
    g.manual_seed(2026)
    e = torch.empty(n, dtype=torch.float64)
    e.exponential_(generator=g)
    path_e = -torch.log(e)                                            # F.gumbel_softmax 的写法

    print(f"    路径 A  -log(-log(U)), U~U(0,1)   : 均值={path_u.mean():.6f} 方差={path_u.var():.6f}")
    print(f"    路径 B  -log(E),       E~Exp(1)   : 均值={path_e.mean():.6f} 方差={path_e.var():.6f}")
    print(f"    Gumbel(0,1) 理论值                : 均值={0.5772156649:.6f} 方差={math.pi**2/6:.6f}")
    print()

    print(f"    A 与 B 的差: 最大 = {float((path_u-path_e).abs().max()):.3e}, "
          f"均值 = {float((path_u-path_e).abs().mean()):.3e}")
    print(f"    A 与 B 逐元素完全相同 = {bool(torch.equal(path_u, path_e))}")
    print()
    print("    exponential_ 到底做了什么变换（同一 seed 下与候选变换比对）:")
    g2 = torch.Generator().manual_seed(2026)
    e2 = torch.empty(n, dtype=torch.float64)
    e2.exponential_(generator=g2)
    g2.manual_seed(2026)
    u2 = torch.rand(n, dtype=torch.float64, generator=g2)
    print(f"      E == -log(U)     : {bool(torch.allclose(e2, -torch.log(u2)))}")
    print(f"      E == -log(1-U)   : {bool(torch.allclose(e2, -torch.log1p(-u2)))}")
    print()
    print("    结论: 两条路径服从同一个 Gumbel(0,1)，但不是逐元素同一个数 ——")
    print("          路径 A 是 -log(-log(U))，路径 B 是 -log(-log(1-U))，同一个 U 的两种变换。")
    print("          分布一致、样本不一致，正是「同一分布的不同采样器」的典型形态。")
    print()


def section_temperature() -> None:
    print("=" * 92)
    print("[3] gumbel_softmax 的温度行为与 straight-through 梯度")
    print("=" * 92)
    logits = torch.tensor([[2.0, 0.5, -1.0]], dtype=torch.float64, requires_grad=True)
    print(f"    logits = {logits.tolist()[0]}, 采样 20000 次")
    print()
    print(f"    {'tau':>8} {'max 分量均值':>16} {'one-hot 比例':>16} {'argmax 分布 (0/1/2)':>26}")
    for tau in (2.0, 1.0, 0.5, 0.1, 0.01):
        mx, onehot, counts = 0.0, 0.0, torch.zeros(3, dtype=torch.float64)
        reps = 20000
        lg = logits.detach().repeat(reps, 1)
        y = F.gumbel_softmax(lg, tau=tau, hard=False, dim=-1)
        mx = float(y.max(dim=-1).values.mean())
        onehot = float((y.max(dim=-1).values > 0.99).double().mean())
        idx = F.gumbel_softmax(lg, tau=tau, hard=True, dim=-1).argmax(dim=-1)
        counts = torch.bincount(idx, minlength=3).to(torch.float64) / reps
        print(f"    {tau:>8} {mx:>16.6f} {onehot:>16.4f}   "
              f"[{counts[0]:.4f}, {counts[1]:.4f}, {counts[2]:.4f}]")
    print()
    exact = F.softmax(logits.detach(), dim=-1)[0]
    print(f"    真实 softmax 概率 = [{exact[0]:.4f}, {exact[1]:.4f}, {exact[2]:.4f}]")
    print("    注: hard=True 的 argmax 分布与 tau 无关，恒等于真实分类分布（Gumbel-max 定理）。")
    print("        温度只影响梯度回传的软硬程度，不改变离散采样的正确性。")
    print()

    print("    straight-through 的梯度检查（固定同一份噪声 + 非退化目标函数）：")
    wt = torch.tensor([[0.7, -1.3, 2.1]], dtype=torch.float64)
    torch.manual_seed(777)
    y_soft = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
    torch.manual_seed(777)
    y_hard = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
    torch.manual_seed(777)
    y_again = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
    g_soft = torch.autograd.grad((y_soft * wt).sum(), logits, retain_graph=True)[0]
    g_hard = torch.autograd.grad((y_hard * wt).sum(), logits)[0]
    print(f"      loss = <y, w>, w = {wt.tolist()[0]}")
    print(f"      重播种后 soft 分支可复现 = {bool(torch.equal(y_soft, y_again))}")
    print(f"      y_hard 取值          = {y_hard.detach().tolist()[0]}  (恰好 one-hot)")
    print(f"      d loss/dlogit (soft) = {[round(v, 10) for v in g_soft[0].tolist()]}")
    print(f"      d loss/dlogit (hard) = {[round(v, 10) for v in g_hard[0].tolist()]}")
    print(f"      两者相同 = {bool(torch.allclose(g_soft, g_hard))}   "
          f"(straight-through: 前向 one-hot, 反向等于同一份噪声下 soft 的梯度)")
    print( "      两个坑: ① 不固定噪声则两次调用抽的是不同 y_soft，梯度当然不同；")
    print( "              ② 拿 y_soft.sum() 当 loss 会得到全 0（softmax 输出和恒为 1）。")
    print()


def section_gumbel_vs_multinomial() -> None:
    print("=" * 92)
    print("[4] Gumbel-max 与 multinomial 是否同一个分布（K=6，20 万次）")
    print("=" * 92)
    K, N = 6, 200_000
    logits = torch.tensor([[1.5, 0.2, -0.5, 0.8, -2.0, 1.1]], dtype=torch.float64)
    probs = F.softmax(logits, dim=-1)

    g = torch.Generator().manual_seed(2026)
    a = torch.multinomial(probs.expand(N, -1), 1, generator=g).squeeze(-1)

    g.manual_seed(2026)
    noise = -torch.log(-torch.log(torch.rand(N, K, dtype=torch.float64, generator=g)))
    b = (logits + noise).argmax(dim=-1)

    g.manual_seed(2026)
    q = torch.empty(N, K, dtype=torch.float64)
    q.exponential_(generator=g)
    c = (probs.expand(N, -1) / q).argmax(dim=-1)

    ca = torch.bincount(a, minlength=K).to(torch.float64) / N
    cb = torch.bincount(b, minlength=K).to(torch.float64) / N
    cc = torch.bincount(c, minlength=K).to(torch.float64) / N
    print(f"    {'idx':>4} {'理论':>10} {'multinomial':>13} {'Gumbel-max':>13} {'p/Exp(1)':>13}")
    for i in range(K):
        print(f"    {i:>4} {probs[0,i]:>10.5f} {ca[i]:>13.5f} {cb[i]:>13.5f} {cc[i]:>13.5f}")
    print()
    print(f"    Gumbel-max 与 multinomial 逐元素一致率 = {float((a==b).double().mean()):.4f}")
    print(f"    p/Exp(1)  与 multinomial 逐元素一致率 = {float((a==c).double().mean()):.4f}")
    print()


def section_selftest() -> None:
    print("=" * 92)
    print("[0] 自校验")
    print("=" * 92)
    # 重参数化的解析解：Normal(0,1).rsample 的样本标准差应接近 1
    d = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64),
                                   torch.tensor(1.0, dtype=torch.float64))
    s = d.rsample((200_000,))
    print(f"    Normal(0,1).rsample 200k: 均值={float(s.mean()):.5f} 标准差={float(s.std()):.5f}")
    # gradcheck(rsample) 本身不可能通过（每调用一次就重抽 eps）；固定 eps 后验证展开式
    loc = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    sc = torch.tensor([1.2], dtype=torch.float64, requires_grad=True)
    torch.manual_seed(11)
    r = torch.distributions.Normal(loc, sc).rsample((3,))
    torch.manual_seed(11)
    eps = torch.empty(3, 1, dtype=torch.float64).normal_()
    print(f"    rsample 展开为 loc+eps*scale 逐元素相等 = {bool(torch.allclose(r, loc.detach()+eps*sc.detach()))}")
    ok = torch.autograd.gradcheck(lambda l, s_: (l + eps * s_).sum(), (loc, sc))
    print(f"    gradcheck(loc + eps*scale) = {ok}")
    # clamp_probs 的边界
    cp = torch.distributions.utils.clamp_probs(torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
    print(f"    clamp_probs([0, 0.5, 1]) = {cp.tolist()}  eps={torch.finfo(torch.float64).eps:.3e}")
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
    section_rsample()
    section_two_gumbel_paths()
    section_temperature()
    section_gumbel_vs_multinomial()


if __name__ == "__main__":
    main()
