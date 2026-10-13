#!/usr/bin/env python3
"""
L1: 采样原语的等价性与边界行为。

验证四件事：
  1. 有放回采样的三种写法（torch.multinomial / Gumbel-max / vLLM 的 p/Exp(1)）
     是否服从同一分布；
  2. 无放回的 log(U)/w + topk 与「逐次剔除重归一化」的教科书定义是否一致；
  3. torch.multinomial 在退化输入（零概率、全零行、n_sample 越界）上的真实行为；
  4. torch.multinomial / Tensor.exponential_ 每次调用消耗多少个均匀数
     （逆变换每行 1 个；Gumbel、指数竞赛每行 K 个——用于反推内部路径）。

纯 CPU 可跑。判据用标准化偏差 max-z（每个桶期望计数 >= 30 时近似 N(0,1)），
阈值 5 sigma；样本量故意取大，让真实差异无法被随机波动掩盖。

用法:
    python l1_sampler_primitives.py            # 全部
    python l1_sampler_primitives.py --selftest # 只跑自校验
"""

import argparse
import math

import torch

PROBS = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05], dtype=torch.float64)
N_SAMPLES = 400_000
Z_THRESHOLD = 5.0


def max_z(counts: torch.Tensor, probs: torch.Tensor, n: int) -> float:
    """每桶标准化偏差 |obs - n*p| / sqrt(n*p*(1-p)) 的最大值。"""
    expected = n * probs.to(torch.float64)
    sd = torch.sqrt(expected * (1.0 - probs.to(torch.float64)))
    return float(((counts.to(torch.float64) - expected).abs() / sd).max())


def tv_distance(counts: torch.Tensor, probs: torch.Tensor, n: int) -> float:
    return 0.5 * float((counts.to(torch.float64) / n - probs.to(torch.float64)).abs().sum())


def bincount(idx: torch.Tensor, k: int) -> torch.Tensor:
    return torch.bincount(idx.reshape(-1).to(torch.int64), minlength=k).to(torch.float64)


def uniforms_consumed(op, seed: int = 20260930, probe_len: int = 64, repeat: int = 1) -> int:
    """测量 op 单次调用从 generator 里取走多少个均匀数。

    做法：用同一 seed 先生成参考流 r_0..r_{probe_len-1}；再把 generator 复位、
    执行 op（可重复 repeat 次），随后取一个均匀数，在参考流里定位它的下标——
    该下标就是消耗量。不依赖任何内部计数器，纯行为观测。

    这条测量是判别内部路径的硬证据：逆变换采样每行只需 1 个均匀数，
    而走 Gumbel/指数竞赛的实现每行要 K 个（K = 类别数）。
    """
    g = torch.Generator().manual_seed(seed)
    ref = torch.rand(probe_len, dtype=torch.float64, generator=g)
    g = torch.Generator().manual_seed(seed)
    for _ in range(repeat):
        op(g)
    nxt = float(torch.rand(1, dtype=torch.float64, generator=g))
    hits = (ref == nxt).nonzero(as_tuple=True)[0]
    return int(hits[0]) if hits.numel() else -1


# ---------------------------------------------------------------- 有放回三写法


def sample_multinomial(probs: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """A: torch.multinomial，CPU 上走 ATen 的 multinomial_stub。"""
    g = torch.Generator().manual_seed(seed)
    rows = probs.to(torch.float64).unsqueeze(0).expand(n, -1)
    return torch.multinomial(rows, 1, replacement=True, generator=g).squeeze(-1)


def sample_gumbel_max(probs: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """B: Gumbel-max，argmax(log p + g), g = -log(-log(U))。"""
    g = torch.Generator().manual_seed(seed)
    logits = probs.to(torch.float64).log()
    u = torch.rand(n, probs.numel(), dtype=torch.float64, generator=g)
    gumbel = -torch.log(-torch.log(u))
    return (logits.unsqueeze(0) + gumbel).argmax(dim=-1)


def sample_exp_race_vllm(probs: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """C: vLLM random_sample 的写法，argmax(p / Exp(1))。"""
    g = torch.Generator().manual_seed(seed)
    p = probs.to(torch.float64).unsqueeze(0).expand(n, -1).clone()
    q = torch.empty_like(p)
    q.exponential_(generator=g)
    return p.div_(q).argmax(dim=-1)


# -------------------------------------------------------------- 无放回三种写法


def wo_direct(probs: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """D: torch.multinomial(replacement=False)，当前实现 = log(U)/w 后 topk。"""
    g = torch.Generator().manual_seed(seed)
    rows = probs.to(torch.float64).unsqueeze(0)
    return torch.multinomial(rows, k, replacement=False, generator=g).squeeze(0)


def wo_manual_race(probs: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """E: 手写 Efraimidis-Spirakis / exponential race。"""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(1, probs.numel(), dtype=torch.float64, generator=g)
    keys = u.log() / probs.to(torch.float64).unsqueeze(0)
    return keys.topk(k, dim=-1).indices.squeeze(0)


def wo_sequential(probs: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """F: 教科书定义——每采一个就把该项概率置零并重新归一化。"""
    g = torch.Generator().manual_seed(seed)
    w = probs.to(torch.float64).clone()
    picked = []
    for _ in range(k):
        idx = torch.multinomial(w.unsqueeze(0), 1, replacement=True, generator=g).item()
        picked.append(idx)
        w[idx] = 0.0
    return torch.tensor(picked, dtype=torch.int64)


def wo_joint_distribution(sampler, probs: torch.Tensor, k: int, trials: int, seed: int):
    """统计 k 元组（作为升序集合）的出现频率，用于比较无放回采样器。"""
    counts: dict[tuple, int] = {}
    for t in range(trials):
        key = tuple(sorted(sampler(probs, k, seed + t).tolist()))
        counts[key] = counts.get(key, 0) + 1
    return counts


def exact_wo_joint(probs: torch.Tensor, k: int) -> dict[tuple, float]:
    """无放回（顺序无关）的精确联合概率：P(S) = sum over orders of prod w_i / (partial sums)。"""
    from itertools import combinations, permutations

    w = [float(x) for x in probs]
    out: dict[tuple, float] = {}
    for combo in combinations(range(len(w)), k):
        total = 0.0
        for order in permutations(combo):
            remaining = sum(w)
            p = 1.0
            for i in order:
                p *= w[i] / remaining
                remaining -= w[i]
            total += p
        out[tuple(sorted(combo))] = total
    return out


# ---------------------------------------------------------------------- 各小节


def section_replacement_true() -> None:
    print("=" * 78)
    print("[1] 有放回：三种写法的分布一致性")
    print("=" * 78)
    print(f"    目标概率   = {PROBS.tolist()}")
    print(f"    样本量     = {N_SAMPLES}")
    print()

    impls = [
        ("A torch.multinomial", sample_multinomial),
        ("B Gumbel-max      ", sample_gumbel_max),
        ("C vLLM p/Exp(1)   ", sample_exp_race_vllm),
    ]
    rows = []
    for name, fn in impls:
        idx = fn(PROBS, N_SAMPLES, seed=20260930)
        counts = bincount(idx, PROBS.numel())
        z = max_z(counts, PROBS, N_SAMPLES)
        tv = tv_distance(counts, PROBS, N_SAMPLES)
        rows.append((name, counts, z, tv, idx[:12].tolist()))

    for name, counts, z, tv, head in rows:
        print(f"  {name}  max-z={z:6.3f}  TV={tv:.5f}  前 12 个样本={head}")
    print()

    print("  两两一致性（同一随机种子下逐元素相等的比例，说明实现是否bit级一致）:")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a = impls[i][1](PROBS, N_SAMPLES, seed=20260930)
            b = impls[j][1](PROBS, N_SAMPLES, seed=20260930)
            same = float((a == b).double().mean())
            print(f"    {impls[i][0].strip()} vs {impls[j][0].strip()}: {same:.4f}")
    print()
    print(f"  判据: max-z < {Z_THRESHOLD} 视为与目标分布无显著差异")
    verdict = all(r[2] < Z_THRESHOLD for r in rows)
    print(f"  结论: {'三种写法分布一致' if verdict else '存在显著差异，需检查'}")
    print()

    print("  均匀数预算（每行消耗多少个 U(0,1)，用于反推内部走的是哪条路径）:")
    one_row = PROBS.unsqueeze(0).expand(1, -1).clone()
    k = PROBS.numel()

    def _mult(g):
        return torch.multinomial(one_row, 1, replacement=True, generator=g)

    def _exp(g):
        return torch.empty(1, k, dtype=torch.float64).exponential_(generator=g)

    print(f"multinomial(1x{k}) 消耗均匀数 = {uniforms_consumed(_mult)}")
    print(f"exponential_(1x{k}) 消耗均匀数 = {uniforms_consumed(_exp)}")
    print(f"multinomial(1x{k}) 反复 3 次后单次增量: {uniforms_consumed(_mult, repeat=3)}")
    print(f"对照 torch.rand(1) 消耗均匀数 = "
          f"{uniforms_consumed(lambda g: torch.rand(1, dtype=torch.float64, generator=g))}")
    print()


def section_replacement_false() -> None:
    print("=" * 78)
    print("[2] 无放回：log(U)/w + topk 是否等价于逐次剔除重归一化")
    print("=" * 78)
    k = 2
    trials = 60_000
    print(f"    权重 = {PROBS.tolist()}, k = {k}, 试验次数 = {trials}")
    print()

    exact = exact_wo_joint(PROBS, k)
    impls = [
        ("D torch.multinomial(rep=False)", wo_direct),
        ("E 手写 log(U)/w + topk        ", wo_manual_race),
        ("F 逐次剔除 + 重归一化          ", wo_sequential),
    ]

    print(f"    {'子集':<10} {'精确概率':>10}  " + "  ".join(f"{n.strip()[:14]:>14}" for n, _ in impls))
    observed = {}
    for name, fn in impls:
        observed[name] = wo_joint_distribution(fn, PROBS, k, trials, seed=7)

    for combo, p in sorted(exact.items()):
        line = f"    {str(combo):<10} {p:>10.5f}  "
        for name, _ in impls:
            emp = observed[name].get(combo, 0) / trials
            line += f"{emp:>14.5f}"
        print(line)
    print()

    for name, _ in impls:
        tv = 0.5 * sum(abs(observed[name].get(c, 0) / trials - p) for c, p in exact.items())
        print(f"  {name.strip():<32} TV vs 精确 = {tv:.5f}")
    print()

    d_seed_a = wo_direct(PROBS, k, seed=99)
    e_seed_a = wo_manual_race(PROBS, k, seed=99)
    print(f"  同一 seed 下 D={d_seed_a.tolist()} E={e_seed_a.tolist()}"
          f" -> {'一致' if torch.equal(d_seed_a, e_seed_a) else '不一致（RNG 消耗方式不同）'}")
    print()


def section_degenerate() -> None:
    print("=" * 78)
    print("[3] 退化输入的真实行为")
    print("=" * 78)

    print("  3.1 零概率项会不会被采到（有放回，w=[1,0,0]，各采 5 万次）")
    g = torch.Generator().manual_seed(1)
    w = torch.tensor([[1.0, 0.0, 0.0]])
    idx = torch.multinomial(w.expand(50_000, -1), 1, replacement=True, generator=g)
    print(f"      bincount = {torch.bincount(idx.reshape(-1), minlength=3).tolist()}")
    print()

    print("  3.2 全零行（有放回）")
    try:
        r = torch.multinomial(torch.zeros(1, 4), 1, replacement=True)
        print(f"      未报错，返回 {r.tolist()}")
    except RuntimeError as e:
        print(f"      RuntimeError: {e}")
    print()

    print("  3.3 全零行（无放回）")
    try:
        r = torch.multinomial(torch.zeros(1, 4), 1, replacement=False)
        print(f"      未报错，返回 {r.tolist()}")
    except RuntimeError as e:
        print(f"      RuntimeError: {e}")
    print()

    print("  3.4 无放回时 n_sample 的上限：文档说受非零元素数约束，实测看哪个")
    for w_list, k in [([1.0, 0.0, 0.0], 2), ([1.0, 0.0, 0.0], 3), ([1.0, 1.0, 1.0], 3), ([1.0, 1.0, 1.0], 4)]:
        wt = torch.tensor([w_list], dtype=torch.float64)
        try:
            r = torch.multinomial(wt, k, replacement=False)
            print(f"      w={w_list} k={k} -> 返回 {r.tolist()}")
        except RuntimeError as e:
            print(f"      w={w_list} k={k} -> RuntimeError: {e}")
    print()

    print("  3.5 负数 / inf 输入")
    for w_list, tag in [([-1.0, 2.0], "含负数"), ([float("inf"), 1.0], "含 inf")]:
        wt = torch.tensor([w_list], dtype=torch.float64)
        for rep in (True, False):
            try:
                r = torch.multinomial(wt, 1, replacement=rep)
                print(f"      {tag} replacement={rep} -> 返回 {r.tolist()}")
            except RuntimeError as e:
                print(f"      {tag} replacement={rep} -> RuntimeError: {str(e)[:78]}")
    print()


def section_uncertainty() -> None:
    print("=" * 78)
    print("[4] 自校验：判据本身是否可信")
    print("=" * 78)
    g = torch.Generator().manual_seed(12345)
    idx = torch.randint(0, PROBS.numel(), (N_SAMPLES,), generator=g)
    counts = bincount(idx, PROBS.numel())
    print(f"    均匀随机流对非均匀目标: max-z = {max_z(counts, PROBS, N_SAMPLES):8.2f}"
          f" (应当远大于 {Z_THRESHOLD}，否则判据太松)")
    idx2 = sample_multinomial(PROBS, N_SAMPLES, seed=5)
    counts2 = bincount(idx2, PROBS.numel())
    print(f"    真采样对目标分布:      max-z = {max_z(counts2, PROBS, N_SAMPLES):8.2f}"
          f" (应当小于 {Z_THRESHOLD})")
    print(f"    每个桶的期望计数 >= {int(N_SAMPLES * float(PROBS.min()))}，正态近似成立")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="只跑判据自校验")
    args = ap.parse_args()

    torch.set_num_threads(1)
    print(f"torch {torch.__version__} | 设备 cpu | 线程 1")
    print()
    if args.selftest:
        section_uncertainty()
        return
    section_uncertainty()
    section_replacement_true()
    section_replacement_false()
    section_degenerate()


if __name__ == "__main__":
    main()
