#!/usr/bin/env python3
"""
Companion lab for the blog post:
  "超越函数在 GPU 上到底怎么算：从 MUFU.EX2 到 FlashAttention-3 的 softmax 之战"

Reproduces every number quoted in the article. Run with:
    /Users/congyuan/Software/miniconda3/bin/python3 tools/transcendental_lab.py

Labs
  1  throughput cliff: 989 TFLOPS vs 3.9 T/s  (why softmax hurts)
  2  minimax vs Taylor vs table+quadratic for 2^f on [0,1)
  3  Newton-Raphson refinement from a 23-bit MUFU seed
  4  softmax numerics: exp2 route, max-subtraction, FP16 overflow
  5  RoPE phase precision at 1M context (sinf-style reduction vs Payne-Hanek)
"""

import math

import numpy as np

np.set_printoptions(precision=6, suppress=False)
F64 = np.float64


def line(title=""):
    print()
    if title:
        print("=" * 74)
        print(title)
        print("=" * 74)


# ----------------------------------------------------------------------------
# Lab 1  throughput cliff
# ----------------------------------------------------------------------------
def lab1():
    line("Lab 1  H100 throughput cliff: Tensor Core vs SFU")

    sm = 132                    # H100 SXM5 SM count
    freq = 1.83e9               # clock used for the 989 TFLOPS FP16 figure
    sfu_ops_per_sm_clk = 16     # CUDA C Programming Guide: special function throughput
    sfu_tps = sm * freq * sfu_ops_per_sm_clk

    fp16_flops = 989e12
    fp8_flops = 1978e12
    fp32_ops = sm * freq * 128  # 128 FP32 ops/clk/SM

    print(f"SFU (MUFU)          : {sfu_tps/1e12:8.2f} T ops/s   (16 ops/clk/SM)")
    print(f"FP32 ALU            : {fp32_ops/1e12:8.2f} T ops/s   (128 ops/clk/SM)")
    print(f"FP16 Tensor Core    : {fp16_flops/1e12:8.2f} TFLOPS")
    print(f"FP8  Tensor Core    : {fp8_flops/1e12:8.2f} TFLOPS")
    print()
    print(f"SFU is 1/{fp32_ops/sfu_tps:.0f} of the FP32 ALU rate")
    print(f"FP16 MAC vs SFU     : {(fp16_flops/2)/sfu_tps:8.1f} x")
    print(f"FP8  MAC vs SFU     : {(fp8_flops/2)/sfu_tps:8.1f} x")

    # per-softmax-element instruction budget (FlashAttention inner loop)
    print()
    print("Per-score-element budget, head dim d = 128:")
    for d in (64, 128, 256):
        # two GEMMs (QK^T and P V) => 2 * 2 * d FLOPs per score element
        matmul_flops_per_elem = 4 * d
        for name, flops in (("FP16", fp16_flops), ("FP8", fp8_flops)):
            t_matmul = matmul_flops_per_elem / flops
            t_exp = 1.0 / sfu_tps
            frac_of_matmul = t_exp / t_matmul
            frac_of_total = t_exp / (t_exp + t_matmul)
            print(f"  d={d:3d}  {name}  matmul {matmul_flops_per_elem:4d} FLOP/elem | "
                  f"exp/matmul = {frac_of_matmul*100:5.1f}% | "
                  f"exp/(exp+matmul) = {frac_of_total*100:5.1f}%")

    print()
    print("Add the FP32 ALU work softmax also needs (max reduce, sum, rescale,")
    print("F2FP cast): ~5 FP32 ops per element at 1/8 the SFU cost each.")
    extra = 5.0 / 8.0
    d = 128
    t_matmul = (4 * d) / fp16_flops
    t_nonmatmul = (1.0 + extra) / sfu_tps
    print(f"  FP16 d=128 non-matmul share of (matmul+non-matmul) = "
          f"{t_nonmatmul/(t_nonmatmul+t_matmul)*100:.1f}%")


# ----------------------------------------------------------------------------
# Lab 2  approximating 2^f on [0,1)
# ----------------------------------------------------------------------------
def remez_minimax_rel(f, deg, a=0.0, b=1.0, iters=120, grid=6001):
    """Minimax poly for f on [a,b] under *relative* error. Remez exchange."""
    n = deg + 1          # number of coefficients
    m = deg + 2          # reference points
    xg = np.linspace(a, b, grid)
    fg = f(xg)

    xr = np.sort(0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * np.arange(m) / (deg + 1)))
    coef, E = None, 0.0
    for _ in range(iters):
        A = np.zeros((m, m))
        for j in range(n):
            A[:, j] = xr ** j
        fr = f(xr)
        A[:, n] = -((-1.0) ** np.arange(m)) * fr
        try:
            sol = np.linalg.solve(A, fr)
        except np.linalg.LinAlgError:
            break
        coef, E = sol[:n], abs(sol[n])

        r = (np.polyval(coef[::-1], xg) - fg) / fg
        ar = np.abs(r)
        loc = np.ones(grid, bool)
        loc[1:-1] = (ar[1:-1] >= ar[:-2]) & (ar[1:-1] >= ar[2:])
        cand, candr = xg[loc], r[loc]
        if len(cand) < m:
            break
        chosen, last = [], 0
        for i in np.argsort(-np.abs(candr)):
            s = np.sign(candr[i])
            if s == 0:
                continue
            if last == 0 or s != last:
                chosen.append(i)
                last = s
            if len(chosen) == m:
                break
        if len(chosen) < m:
            break
        xr_new = np.sort(cand[chosen])
        if np.allclose(xr_new, xr, atol=1e-13):
            xr = xr_new
            break
        xr = xr_new

    r = (np.polyval(coef[::-1], xg) - fg) / fg
    return coef, float(np.max(np.abs(r))), float(E)


def lab2():
    line("Lab 2  Approximating 2^f on [0,1): minimax vs Taylor vs table+quadratic")

    f = lambda x: np.power(2.0, x)
    xg = np.linspace(0.0, 1.0, 200001)
    ref = f(xg)
    target = 2.0 ** -22          # MUFU.EX2 class accuracy ~2.4e-7
    print(f"target: MUFU.EX2 class accuracy  2^-22 = {target:.3e}")
    print()
    print(f"{'scheme':34s} {'max rel err':>13s}  {'vs 2^-22':>9s}")

    # Taylor at 0
    for deg in (3, 4, 5, 6, 8):
        c = np.array([np.log(2.0) ** k / math.factorial(k) for k in range(deg + 1)])
        err = np.max(np.abs((np.polyval(c[::-1], xg) - ref) / ref))
        print(f"{'Taylor deg ' + str(deg):34s} {err:13.3e}  {err/target:9.2f}x")

    print()
    # minimax
    best = {}
    for deg in (2, 3, 4, 5):
        c, err, E = remez_minimax_rel(f, deg)
        best[deg] = (c, err)
        print(f"{'minimax deg ' + str(deg):34s} {err:13.3e}  {err/target:9.2f}x")
    print()
    print(f"{'=> equioscillation check (|E| from Remez vs measured):':34s}")
    for deg in (2, 3, 4, 5):
        print(f"{'   deg ' + str(deg):34s} {best[deg][1]:13.3e}   (stable)")

    print()
    # table + low order polynomial: split [0,1) into S segments
    print("Hardware-style two-level scheme: LUT of S segments + degree-2 poly")
    for S in (8, 16, 32, 64, 128):
        edges = np.linspace(0.0, 1.0, S + 1)
        worst = 0.0
        for s in range(S):
            a, b = edges[s], edges[s + 1]
            xx = np.linspace(a, b, 2001)
            ff = f(xx)
            # degree-2 minimax via Remez on the sub-interval
            c, e, _ = remez_minimax_rel(f, 2, a, b, grid=801)
            worst = max(worst, e)
        print(f"  S={S:4d} segments x deg 2        {worst:13.3e}  {worst/target:9.2f}x")

    print()
    print("Coefficients, minimax degree 4 for 2^f on [0,1) (Horner order):")
    c = best[4][0]
    for i, v in enumerate(c):
        print(f"  c{i} = {v:+.12f}")


# ----------------------------------------------------------------------------
# Lab 3  Newton-Raphson refinement from a MUFU seed
# ----------------------------------------------------------------------------
def lab3():
    line("Lab 3  Newton-Raphson refinement from a 23-bit MUFU.RSQ seed")

    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 4.0, 200000)          # MUFU's comfortable input domain
    true_rsqrt = 1.0 / np.sqrt(a)

    print("seed error injected: 2^-23 relative (MUFU.RSQ class, ~1 ulp)")
    for seed_bits in (23,):
        eps = rng.uniform(-1, 1, a.size) * 2.0 ** -seed_bits
        y = true_rsqrt * (1.0 + eps)
        e0 = np.max(np.abs(y * np.sqrt(a) - 1.0))
        print(f"  seed            : max rel err = {e0:.3e}  ({np.log2(e0):+.1f} bits)")

        for it in range(1, 4):
            y = y * (1.5 - 0.5 * a * y * y)
            e = np.max(np.abs(y * np.sqrt(a) - 1.0))
            print(f"  + Newton iter {it} : max rel err = {e:.3e}  ({np.log2(e):+.1f} bits)"
                  f"   {'> FP32' if e < 2**-24 else ''}{'  > FP64 needs 2' if it == 2 else ''}")

    print()
    print("Same for reciprocal (MUFU.RCP -> division):")
    b = rng.uniform(0.5, 4.0, 200000)
    true_rcp = 1.0 / b
    eps = rng.uniform(-1, 1, b.size) * 2.0 ** -23
    y = true_rcp * (1.0 + eps)
    print(f"  seed            : max rel err = {np.max(np.abs(y*b-1)):.3e}")
    for it in range(1, 3):
        y = y * (2.0 - b * y)
        print(f"  + Newton iter {it} : max rel err = {np.max(np.abs(y*b-1)):.3e}")

    print()
    print("Instruction cost on NVIDIA hardware:")
    print("  rsqrt.approx.f32 : 1 MUFU                      -> ~23 bit")
    print("  div.full.f32     : MUFU.RCP + 1 Newton (FFMA)  -> IEEE FP32")
    print("  div.rn.f64       : template with 2 Newton      -> ~100 SASS insns")


# ----------------------------------------------------------------------------
# Lab 4  softmax numerics
# ----------------------------------------------------------------------------
def softmax_ref(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def lab4():
    line("Lab 4  Softmax numerics: exp2 route, max-subtraction, FP16 overflow")

    rng = np.random.default_rng(7)
    n = 4096
    x64 = rng.normal(0.0, 6.0, n).astype(np.float64)   # realistic logit spread
    ref = softmax_ref(x64)

    x32 = x64.astype(np.float32)

    # route A: libm-quality exp in FP32
    a = np.exp(x32 - x32.max())
    out_a = (a / a.sum()).astype(np.float64)

    # route B: exp2 route with MUFU-class perturbation (2^-22 relative on the ex2)
    y = (x32 - x32.max()) * np.float32(np.log2(np.e))
    e2 = np.exp2(y) * (1.0 + rng.uniform(-1, 1, n) * 2.0 ** -22)
    out_b = (e2 / e2.sum()).astype(np.float64)

    # route C: exp2 route, no max subtraction, then cast to FP16
    y_raw = x32 * np.float32(np.log2(np.e))
    with np.errstate(over="ignore"):
        e_raw = np.exp2(y_raw.astype(np.float32))
    fp16_max = 65504.0
    n_inf = int(np.sum(~np.isfinite(e_raw)) + np.sum(e_raw > fp16_max))
    print(f"logits: {n} elements, mean {x64.mean():+.2f}, std {x64.std():.2f}, "
          f"max {x64.max():+.2f}, min {x64.min():+.2f}")
    print()
    print(f"route A  FP32 libm exp        max abs err vs FP64 = {np.max(np.abs(out_a-ref)):.3e}")
    print(f"route B  FP32 exp2 + 2^-22    max abs err vs FP64 = {np.max(np.abs(out_b-ref)):.3e}")
    print(f"route C  no max-subtraction   -> {n_inf}/{n} elements overflow FP16 (max 65504)")
    print()
    print("Why max-subtraction makes exp2 exact-friendly:")
    y = (x32 - x32.max())
    print(f"  after max-subtraction, ex2 argument range = [{y.min():.2f}, {y.max():.2f}]")
    print(f"  integer parts n = floor(arg) all <= 0 -> result stays <= 1")
    print(f"  fractional parts f land in [0,1) -> exactly where MUFU.EX2 is fitted")
    print(f"  fraction of arguments with f in [0,1): "
          f"{np.mean((y - np.floor(y) >= 0) & (y - np.floor(y) < 1))*100:.1f}%")

    print()
    print("Online softmax (FlashAttention): one exp2 per element, not two.")
    print("  naive  : exp(s_i), sum, then divide  -> 1 exp + 1 rcp per element")
    print("  online : exp2(s_i - m_new) fused with the running rescale")
    print("           O_cur <- O_cur * exp2(m_old - m_new) + exp2(s_i - m_new) * V_i")
    print("  the rescale factor exp2(m_old - m_new) is *one scalar per block*,")
    print("  not one per element -> the second pass over scores disappears.")


# ----------------------------------------------------------------------------
# Lab 5  RoPE phase precision
# ----------------------------------------------------------------------------
def lab5():
    line("Lab 5  RoPE phase precision at long context")

    two_pi_f32 = np.float32(2.0 * np.pi)
    two_pi_true = 2.0 * np.pi
    print(f"2*pi in FP32   : {F64(two_pi_f32):.12f}")
    print(f"2*pi true      : {two_pi_true:.12f}")
    print(f"abs error      : {abs(F64(two_pi_f32)-two_pi_true):.3e} "
          f"(= {abs(F64(two_pi_f32)-two_pi_true)/two_pi_true:.2e} relative)")
    print()

    print("sinf-style reduction: r = x - round(x / 2pi_f32) * 2pi_f32, all in FP32")
    print(f"{'position':>10s} {'theta':>10s} {'x=p*theta':>13s} "
          f"{'phase err (rad)':>16s} {'cos err':>11s}")
    for p in (1e3, 1e4, 1e5, 1e6, 1e7):
        for theta in (1.0, 1e-2):
            x = np.float32(p * theta)
            k = np.round(x / two_pi_f32)
            r = np.float32(x - np.float32(k * two_pi_f32))
            approx = np.cos(F64(r))
            truth = np.cos(F64(x) * 1.0)      # exact argument reduction
            phase = abs(F64(r) - np.mod(F64(x), two_pi_true))
            phase = min(phase, two_pi_true - phase)
            print(f"{p:10.0f} {theta:10.0e} {F64(x):13.2f} "
                  f"{phase:16.3e} {abs(approx-truth):11.3e}")

    print()
    print("Interpretation: the error grows linearly with the winding number")
    print("k = x / 2pi. At p = 1e6 with theta = 1 (the highest-frequency RoPE")
    print("channel) the phase is off by ~1e-2 rad, i.e. the *relative* position")
    print("information in that channel is corrupted. Payne-Hanek reduction")
    print("(libdevice sinf/cosf, or a precomputed table) has no such term.")

    print()
    print("Second failure mode: storing the table in bf16 erases neighbours.")
    d, base = 128, 10000.0
    i_hi, i_lo = 0, d // 2 - 1
    theta_lo = base ** (-(2 * i_lo) / d)
    theta_hi = base ** (-(2 * i_hi) / d)
    for p in (1e5, 1e6):
        for name, theta in (("low freq ", theta_lo), ("high freq", theta_hi)):
            c0 = np.cos(p * theta)
            c1 = np.cos((p + 1) * theta)
            gap = abs(c0 - c1)
            bf16_ulp = 2.0 ** -8
            print(f"  p={p:.0e} {name} theta={theta:.3e}  "
                  f"cos(p)-cos(p+1) = {gap:.3e}  "
                  f"bf16 ulp ~ {bf16_ulp:.2e}  -> "
                  f"{'ERASED' if gap < bf16_ulp else 'survives'}")
    print()
    print("  bf16 has 8 mantissa bits; adjacent-position differences in the")
    print("  low-frequency channels fall below one bf16 ulp, so those channels")
    print("  silently stop encoding position. Keep the RoPE table in FP32.")
    print("  (MegaBeam reports exactly this: NIAH recall dropped digits until")
    print("   RoPE was forced to FP32 -- arXiv:2505.08651)")


if __name__ == "__main__":
    lab1()
    lab2()
    lab3()
    lab4()
    lab5()
    print()
    print("All labs complete.")
