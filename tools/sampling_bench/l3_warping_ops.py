#!/usr/bin/env python3
"""
L3: logits 塑形算子（top-k / top-p / min-p）的实现分歧。

同一个 top-p，transformers 是「升序排列 + cumsum <= 1-p 掩掉」，
教科书定义是「降序排列 + 保留累积到 p 的最小前缀」。两者声称等价，
本脚本把它们的 mask 逐元素比一遍，并把差异样本找出来。

min-p 同理：transformers 在概率域比较，vLLM 的 triton kernel 在 log 域比较。
top-k 再加一条「秩」实现。

纯 CPU 可跑。

用法:
    python l3_warping_ops.py
    python l3_warping_ops.py --selftest
"""

import argparse
import time

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ top-p 实现


def topp_transformers(scores: torch.Tensor, top_p: float, min_keep: int = 1) -> torch.Tensor:
    """transformers.TopPLogitsWarper：升序 + cumsum <= 1-p。"""
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -min_keep:] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return scores.masked_fill(indices_to_remove, float("-inf"))


def topp_textbook(scores: torch.Tensor, top_p: float, min_keep: int = 1) -> torch.Tensor:
    """教科书定义：降序，保留累计概率达到 p 的最小前缀。"""
    probs = scores.softmax(dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    # 第 k 个（0-based）保留 <=> 前 k 个之和仍 < p
    keep = torch.cat([torch.ones_like(cum[..., :1], dtype=torch.bool), cum[..., :-1] < top_p], dim=-1)
    keep[..., :min_keep] = True
    remove_desc = ~keep
    indices_to_remove = remove_desc.scatter(1, sorted_indices, remove_desc)
    return scores.masked_fill(indices_to_remove, float("-inf"))


def topp_vllm(scores: torch.Tensor, top_p: float, min_keep: int = 1) -> torch.Tensor:
    """vLLM apply_top_k_top_p_pytorch 的 top-p 部分（原地改 sort 后 scatter 回去）。"""
    logits_sort, logits_idx = scores.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_p_mask = probs_sum <= 1 - top_p
    top_p_mask[:, -min_keep:] = False
    logits_sort.masked_fill_(top_p_mask, float("-inf"))
    return scores.scatter(dim=-1, index=logits_idx, src=logits_sort)


# ------------------------------------------------------------------ min-p 实现


def minp_prob_domain(scores: torch.Tensor, min_p: float) -> torch.Tensor:
    """transformers.MinPLogitsWarper：先 softmax，再按 top*p 比较。"""
    probs = torch.softmax(scores, dim=-1)
    top_probs = probs.amax(dim=-1, keepdim=True)
    return scores.masked_fill(probs < min_p * top_probs, float("-inf"))


def minp_log_domain(scores: torch.Tensor, min_p: float) -> torch.Tensor:
    """vLLM _min_p_kernel：threshold = max(logits) + log(min_p)。"""
    threshold = scores.amax(dim=-1, keepdim=True) + torch.log(torch.tensor(min_p, dtype=scores.dtype))
    return scores.masked_fill(scores < threshold, float("-inf"))


# ------------------------------------------------------------------ top-k 实现


def topk_broadcast(scores: torch.Tensor, k: int) -> torch.Tensor:
    """transformers.TopKLogitsWarper：拿第 k 大的值做广播比较。"""
    k = min(k, scores.size(-1))
    kth = torch.topk(scores, k, dim=-1)[0][..., -1, None]
    return scores.masked_fill(scores < kth, float("-inf"))


def topk_rank(scores: torch.Tensor, k: int) -> torch.Tensor:
    """秩实现：rank = argsort(argsort(-x))，保留 rank < k。"""
    order = torch.argsort(scores, dim=-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(scores.size(-1), device=scores.device).expand_as(order))
    return scores.masked_fill(rank >= k, float("-inf"))


def compare_masks(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    ma, mb = torch.isinf(a) & (a < 0), torch.isinf(b) & (b < 0)
    same = bool(torch.equal(ma, mb))
    diff = int((ma != mb).sum())
    print(f"      {name:<36} mask 完全相同={same}  不同元素数={diff}")
    if not same:
        rows, cols = torch.nonzero(ma != mb, as_tuple=True)
        show = min(5, rows.numel())
        for i in range(show):
            r, c = int(rows[i]), int(cols[i])
            print(f"          行{r} 列{c}: A 掩掉={bool(ma[r,c])} B 掩掉={bool(mb[r,c])}")
    return same


# ---------------------------------------------------------------------- 各小节


def section_topp() -> None:
    print("=" * 92)
    print("[1] top-p 的三种写法")
    print("=" * 92)
    torch.manual_seed(0)
    B, V = 8, 40
    scores = torch.randn(B, V, dtype=torch.float64) * 2.0
    for top_p in (0.9, 0.95, 0.5):
        print(f"\n    top_p = {top_p}")
        a = topp_transformers(scores, top_p)
        b = topp_textbook(scores, top_p)
        c = topp_vllm(scores, top_p)
        compare_masks("transformers  vs 教科书", a, b)
        compare_masks("transformers  vs vLLM", a, c)
        # 保留 token 数分布
        def kept(x):
            return (~torch.isinf(x)).sum(-1).tolist()
        print(f"      transformers 保留数 = {kept(a)}")
        print(f"      教科书      保留数 = {kept(b)}")
        print(f"      vLLM        保留数 = {kept(c)}")
    print()

    print("    边界情形：出现并列概率时（logits 含重复值）")
    dup = torch.tensor([[2.0, 2.0, 2.0, 1.0, 0.0]] * 1, dtype=torch.float64)
    a, b = topp_transformers(dup, 0.6), topp_textbook(dup, 0.6)
    print(f"      logits = {dup.tolist()[0]}, top_p=0.6")
    print(f"      transformers 保留 = {(~torch.isinf(a)).sum(-1).tolist()}"
          f"   教科书 保留 = {(~torch.isinf(b)).sum(-1).tolist()}")
    print()


def section_minp() -> None:
    print("=" * 92)
    print("[2] min-p：概率域比较 vs log 域比较")
    print("=" * 92)
    torch.manual_seed(1)
    B, V = 8, 40
    scores = torch.randn(B, V, dtype=torch.float64) * 2.0
    for min_p in (0.05, 0.1, 0.3):
        a = minp_prob_domain(scores, min_p)
        b = minp_log_domain(scores, min_p)
        print(f"    min_p = {min_p}")
        compare_masks("概率域 vs log 域 (fp64)", a, b)
        a32 = minp_prob_domain(scores.float(), min_p).double()
        compare_masks("概率域 vs log 域 (fp32)", a32, b)
    print()
    print("    说明: 两者数学上等价 —— softmax 之后 p_i < min_p * p_max")
    print("          等价于 logits_i < logits_max + log(min_p)，公共的 logsumexp 被约掉。")
    print("          log 域写法省掉一次全词表 softmax。")
    print()


def section_topk() -> None:
    print("=" * 92)
    print("[3] top-k 的三种写法")
    print("=" * 92)
    torch.manual_seed(2)
    B, V = 8, 40
    scores = torch.randn(B, V, dtype=torch.float64) * 2.0
    for k in (5, 20):
        print(f"    k = {k}")
        a = topk_broadcast(scores, k)
        b = topk_rank(scores, k)
        compare_masks("broadcast 阈值 vs 秩", a, b)
    print()

    print("    并列值下的行为（logits 含重复最大值）")
    dup = torch.tensor([[3.0, 3.0, 3.0, 1.0, 0.0]], dtype=torch.float64)
    for k in (1, 2):
        a = topk_broadcast(dup, k)
        b = topk_rank(dup, k)
        print(f"      k={k}: broadcast 保留 {(~torch.isinf(a)).sum(-1).item()} 个, "
              f"秩 保留 {(~torch.isinf(b)).sum(-1).item()} 个  "
              f"-> {'一致' if torch.equal(torch.isinf(a), torch.isinf(b)) else '不一致'}")
    print()


def section_cost() -> None:
    print("=" * 92)
    print("[4] 成本：全排序 / topk / 双 argsort，在 LLM 词表尺度上的耗时")
    print("=" * 92)
    torch.set_num_threads(4)
    B, V = 32, 152_064  # 一个 150K 词表的 batch=32 decode 步
    x = torch.randn(B, V)
    reps = 20

    def timed(fn, reps=reps):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2] * 1e3

    print(f"    batch={B}, vocab={V}, 取 {reps} 轮中位数")
    t_sort = timed(lambda: torch.sort(x, dim=-1))
    t_topk = timed(lambda: torch.topk(x, 50, dim=-1))
    t_argsort = timed(lambda: torch.argsort(x, dim=-1, descending=True))
    t_rank = timed(lambda: torch.argsort(torch.argsort(x, dim=-1, descending=True), dim=-1))
    t_minp_log = timed(lambda: x.amax(dim=-1, keepdim=True))
    print(f"      torch.sort(全排序)            {t_sort:8.3f} ms")
    print(f"      torch.topk(k=50)              {t_topk:8.3f} ms   ({t_sort/t_topk:5.1f}x 快于全排序)")
    print(f"      torch.argsort(降序)           {t_argsort:8.3f} ms")
    print(f"      双 argsort 求秩               {t_rank:8.3f} ms   ({t_rank/t_sort:5.2f}x 全排序)")
    print(f"      amax(仅求阈值, min-p 用)       {t_minp_log:8.3f} ms   ({t_sort/t_minp_log:7.1f}x 快于全排序)")
    print()
    torch.set_num_threads(1)
    print("    结论: top-p 必须排序；top-k 只需部分排序；min-p 只需要一个 max。")
    print("          这就是为什么 min-p 在同等筛选强度下比 top-p 便宜一个数量级。")
    print()


def section_end_to_end() -> None:
    print("=" * 92)
    print("[5] 端到端：不同实现 + 同一随机种子，采出的 token 是否相同")
    print("=" * 92)
    torch.manual_seed(3)
    B, V = 4096, 200
    logits = torch.randn(B, V, dtype=torch.float64) * 2.0

    variants = [
        ("原始分布           ", lambda s: s),
        ("top-k=20          ", lambda s: topk_broadcast(s, 20)),
        ("top-p=0.9 (tf)    ", lambda s: topp_transformers(s, 0.9)),
        ("top-p=0.9 (教科书) ", lambda s: topp_textbook(s, 0.9)),
        ("min-p=0.1 (prob)  ", lambda s: minp_prob_domain(s, 0.1)),
        ("min-p=0.1 (log)   ", lambda s: minp_log_domain(s, 0.1)),
    ]
    outs = {}
    for name, fn in variants:
        s = fn(logits)
        g = torch.Generator().manual_seed(20260930)
        probs = F.softmax(s, dim=-1)
        outs[name] = torch.multinomial(probs, 1, generator=g).squeeze(-1)
    base = outs[variants[0][0]]
    print(f"    batch={B}, vocab={V}, 与「原始分布」采样结果的一致率")
    for name, _ in variants[1:]:
        same = float((outs[name] == base).double().mean())
        print(f"      {name} {same:.4f}")
    print()
    print("      top-p 两种写法之间的一致率: "
          f"{float((outs[variants[2][0]] == outs[variants[3][0]]).double().mean()):.4f}")
    print("      min-p 两种写法之间的一致率: "
          f"{float((outs[variants[4][0]] == outs[variants[5][0]]).double().mean()):.4f}")
    print()


def section_selftest() -> None:
    print("=" * 92)
    print("[0] 自校验")
    print("=" * 92)
    s = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    print(f"    均匀 logits top-p=0.5 应保留 2 个: "
          f"transformers={(~torch.isinf(topp_transformers(s, 0.5))).sum(-1).item()}, "
          f"教科书={(~torch.isinf(topp_textbook(s, 0.5))).sum(-1).item()}")
    s2 = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    print(f"    尖锐 logits top-p=0.5 应保留 1 个: "
          f"transformers={(~torch.isinf(topp_transformers(s2, 0.5))).sum(-1).item()}, "
          f"教科书={(~torch.isinf(topp_textbook(s2, 0.5))).sum(-1).item()}")
    s3 = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    print(f"    均匀 logits top-k=2 应保留 2 个: "
          f"broadcast={(~torch.isinf(topk_broadcast(s3, 2))).sum(-1).item()}, "
          f"秩={(~torch.isinf(topk_rank(s3, 2))).sum(-1).item()}")
    # min-p 解析解：logits=[0, -log(5)] 时 p_1/p_0 = 1/5，min_p=0.2 恰好在边界
    s4 = torch.tensor([[0.0, -torch.log(torch.tensor(5.0, dtype=torch.float64)).item()]], dtype=torch.float64)
    print(f"    min-p=0.2 边界样本: 概率域保留 {(~torch.isinf(minp_prob_domain(s4, 0.2))).sum(-1).item()} 个, "
          f"log 域保留 {(~torch.isinf(minp_log_domain(s4, 0.2))).sum(-1).item()} 个")
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
    section_topp()
    section_minp()
    section_topk()
    section_cost()
    section_end_to_end()


if __name__ == "__main__":
    main()
