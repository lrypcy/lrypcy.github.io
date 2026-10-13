#!/usr/bin/env python3
"""
L2: 同一个 logprob，五条写法，到底差多少。

torch.distributions 的 Categorical 给出的是"参数化对象 + log_prob"这条路径；
LLM 训练框架给的是"logits 张量 + gather"这条路径。两者数学等价，数值不等价。
本脚本在正常 / 极端 / 含零概率 / 含 -inf 四类输入上把它们摆在一起比。

纯 CPU 可跑。

用法:
    python l2_logprob_paths.py
    python l2_logprob_paths.py --selftest
"""

import argparse
import math

import torch
import torch.nn.functional as F

torch.set_printoptions(precision=12)


# ------------------------------------------------------------------ 五条路径


def p1_categorical_probs(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Categorical(probs=softmax(logits))，probs 走 clamp(eps, 1-eps) 再 log。"""
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    d = torch.distributions.Categorical(probs=probs)
    return d.log_prob(idx)


def p2_categorical_logits(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Categorical(logits=logits)，构造时做 logits - logsumexp 归一化，log_prob 直接 gather。"""
    d = torch.distributions.Categorical(logits=logits.to(torch.float64))
    return d.log_prob(idx)


def p3_log_softmax(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """log_softmax 全量物化再 gather。"""
    return F.log_softmax(logits.to(torch.float64), dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def p4_selective(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """TRL selective_log_softmax：log x_i - logsumexp(x)，不物化整个 log_softmax。"""
    x = logits.to(torch.float64)
    selected = x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(x, dim=-1)


def p5_log_of_softmax(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """朴素 log(softmax(x))，数值上最不稳，但很多手写实现就是这样。"""
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    return probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).log()


PATHS = [
    ("P1 Categorical(probs)", p1_categorical_probs),
    ("P2 Categorical(logits)", p2_categorical_logits),
    ("P3 log_softmax + gather", p3_log_softmax),
    ("P4 selective (TRL)", p4_selective),
    ("P5 log(softmax)", p5_log_of_softmax),
]


# ------------------------------------------------------------------ 场景定义

SCENARIOS = {
    "S1 常规 logits": (torch.tensor([2.0, 1.0, 0.5, -1.0, -3.0]), 0),
    "S2 极端 logits": (torch.tensor([800.0, 0.0, -800.0]), 0),
    "S3 含 0 概率": (torch.tensor([1.0, 0.0, 0.0]), 2),
    "S4 含 -inf(mask)": (torch.tensor([1.0, float("-inf"), 0.5]), 2),
    "S5 目标为被 mask 项": (torch.tensor([1.0, float("-inf"), 0.5]), 1),
    "S6 概率极小的项": (torch.tensor([0.0, -60.0, -120.0]), 2),
}


def run_scenarios() -> None:
    print("=" * 96)
    print("[1] 五条路径在六类输入上的输出（float64 计算）")
    print("=" * 96)
    for name, (logits, idx) in SCENARIOS.items():
        target = torch.tensor([idx])
        print(f"\n  {name}")
        print(f"    logits = {logits.tolist()}")
        print(f"    目标索引 = {idx}")
        ref = None
        for pname, fn in PATHS:
            try:
                val = fn(logits.unsqueeze(0), target).item()
                txt = f"{val:.12f}"
            except Exception as e:  # noqa: BLE001
                txt = f"<异常 {type(e).__name__}>"
                val = None
            mark = ""
            if ref is None and val is not None:
                ref = val
                mark = "  <- 基准"
            elif val is not None and ref is not None:
                if val == ref:
                    mark = "  一致"
                elif math.isinf(ref) or math.isinf(val):
                    mark = "  不一致(无穷)"
                else:
                    mark = f"  差 {val - ref:+.3e}"
            print(f"      {pname:<26} {txt}{mark}")
    print()


def run_float32_vs_float64() -> None:
    print("=" * 96)
    print("[2] float32 下的漂移：为什么服务端 logits 一律 upcast 到 fp32 再算 logprob")
    print("=" * 96)
    print(f"  {'场景':<22} {'路径':<26} {'float32':>16} {'float64':>16} {'|Δ|':>10}")
    for name, (logits, idx) in SCENARIOS.items():
        target = torch.tensor([idx])
        for pname, fn in PATHS:
            try:
                v32 = fn(logits.unsqueeze(0).to(torch.float32), target).item()
            except Exception:  # noqa: BLE001
                v32 = float("nan")
            try:
                v64 = fn(logits.unsqueeze(0).to(torch.float64), target).item()
            except Exception:  # noqa: BLE001
                v64 = float("nan")
            d = abs(v32 - v64) if not (math.isinf(v32) or math.isinf(v64) or math.isnan(v32) or math.isnan(v64)) else float("nan")
            dtxt = f"{d:.3e}" if d == d else "inf/nan"
            print(f"  {name:<22} {pname:<26} {v32:>16.10f} {v64:>16.10f} {dtxt:>10}")
    print()


def lp_pipeline(logits: torch.Tensor, idx: torch.Tensor, accum: torch.dtype) -> torch.Tensor:
    """不改写输入精度的 logprob：gather 后减 logsumexp（TRL selective_log_softmax 的做法）。"""
    sel = logits.gather(-1, idx.unsqueeze(-1)).squeeze(-1).to(accum)
    lse = torch.logsumexp(logits.to(accum), dim=-1)
    return sel - lse


def report_ratio(tag: str, new_lp: torch.Tensor, old_lp: torch.Tensor) -> None:
    r = (new_lp - old_lp).exp()
    finite = torch.isfinite(r)
    n_bad = int((~finite).sum())
    dev = (r[finite] - 1.0).abs()
    print(f"  {tag}")
    print(f"      max|ratio-1| = {float(dev.max()):.3e}   mean|ratio-1| = {float(dev.mean()):.3e}"
          f"   ratio 范围 [{float(r[finite].min()):.6f}, {float(r[finite].max()):.6f}]"
          f"   inf/nan = {n_bad}")


def run_ratio_identity() -> None:
    print("=" * 96)
    print("[3] RL 的 ratio 恒等式：误差源能把 exp(Δlogp) 推离 1 多远")
    print("=" * 96)
    torch.manual_seed(0)
    V, B = 32, 8192
    logits = torch.randn(B, V, dtype=torch.float64) * 3.0
    g = torch.Generator().manual_seed(11)
    tok = torch.multinomial(F.softmax(logits, -1), 1, generator=g).squeeze(-1)
    ref = lp_pipeline(logits, tok, torch.float64)
    print(f"  batch={B}, vocab={V}；采样分布 = softmax(logits)（fp64）")
    print()

    report_ratio("(a) 同精度、改走 log_softmax+gather 全量物化再取", p3_log_softmax(logits, tok), ref)

    f32 = logits.float()
    report_ratio("(b) 打分侧降到 fp32（推理引擎常见）", lp_pipeline(f32, tok, torch.float32).double(), ref)

    bf16 = logits.bfloat16()
    report_ratio("(c) 打分侧降到 bf16（训练 forward 常见）", lp_pipeline(bf16, tok, torch.float32).double(), ref)

    print()
    print("  说明: (b)(c) 只让打分精度退化，采样还是 fp64 —— 这就是「rollout 用高精度引擎、")
    print("        训练用低精度 forward」的真实配置。即便策略权重一字未改，ratio 也不再恒等于 1，")
    print("        这类偏差会被 PPO/GRPO 当成真实策略漂移来惩罚。")
    print()

    print("  (d) 截断口径不一致：proposal 用 top-k=4 截断，target 用完整分布")
    topk_logits = logits.clone()
    kth = logits.topk(4, dim=-1).values[..., -1:]
    topk_logits[logits < kth] = float("-inf")
    tok_k = torch.multinomial(F.softmax(topk_logits, -1), 1, generator=g).squeeze(-1)
    lp_target = lp_pipeline(logits, tok_k, torch.float64)
    lp_naive_proposal = lp_pipeline(logits, tok_k, torch.float64)  # 错误做法：拿 target 当 proposal
    report_ratio("    错误做法（proposal 也用 target 打分）", lp_naive_proposal, lp_naive_proposal)
    lp_proposal = lp_pipeline(topk_logits, tok_k, torch.float64)
    report_ratio("    正确做法（proposal 用截断后的分布打分）", lp_target, lp_proposal)
    print(f"     注意：截断后 proposal 的概率质量更集中，w = p_target/p_proposal <= 1 的比例 = "
          f"{float((lp_target <= lp_proposal).double().mean()):.4f}")
    print()
    print("  说明: 若打分分布没做同样的 top-k 截断，proposal 里概率为 0 的 token 会让")
    print("        log p_proposal = -inf，正确写法下的 ratio 会直接爆成 inf/nan。")
    print("        截断采样 + 未截断打分，是 RL 里最常见的「采样器与打分器口径不一致」事故。")
    print()


def run_sampled_logprob() -> None:
    print("=" * 96)
    print("[4] 采样得到的 token，用各路径回算 logprob（是否出现 -inf / 精度损失）")
    print("=" * 96)
    torch.manual_seed(3)
    V, B = 1000, 20000
    logits = torch.randn(B, V, dtype=torch.float64) * 2.0
    # 把后一半 token mask 掉，模拟 allowed_token_ids
    logits[:, V // 2:] = float("-inf")

    g = torch.Generator().manual_seed(21)
    probs = F.softmax(logits, -1)
    tok = torch.multinomial(probs, 1, generator=g).squeeze(-1)
    print(f"  batch={B}, vocab={V}（后半被 mask 成 -inf）")
    print(f"  采样结果全部落在前 {V//2} 个 token 内: {bool((tok < V // 2).all())}")
    print()
    for pname, fn in PATHS:
        try:
            v = fn(logits, tok)
            finite = torch.isfinite(v)
            print(f"    {pname:<26} -inf/nan 个数={int((~finite).sum()):<6}"
                  f" mean={float(v[finite].mean()):.6f} min={float(v[finite].min()):.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"    {pname:<26} <异常 {type(e).__name__}: {str(e)[:50]}>")
    print()


def run_selftest() -> None:
    print("=" * 96)
    print("[0] 自校验")
    print("=" * 96)
    # 构造一个已知答案的例子：logits=[0,0]，两个 token 的 logprob 都应是 -log 2
    lg = torch.zeros(1, 2, dtype=torch.float64)
    idx = torch.tensor([0])
    for pname, fn in PATHS:
        v = fn(lg, idx).item()
        expect = -math.log(2.0)
        print(f"    {pname:<26} {v:.12f}  期望 {-math.log(2):.12f}  差 {abs(v-expect):.2e}")
    # 已知答案 2：logits=[a,0] 时 token0 的 logprob = a - log(1+e^a)
    a = 3.0
    lg2 = torch.tensor([[a, 0.0]], dtype=torch.float64)
    expect2 = a - math.log(1 + math.exp(a))
    print(f"\n    logits=[{a}, 0] 时 token0 的解析解 = {expect2:.12f}")
    for pname, fn in PATHS:
        v = fn(lg2, torch.tensor([0])).item()
        print(f"    {pname:<26} {v:.12f}  差 {abs(v-expect2):.2e}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    print(f"torch {torch.__version__} | 设备 cpu")
    print()
    run_selftest()
    if args.selftest:
        return
    run_scenarios()
    run_float32_vs_float64()
    run_ratio_identity()
    run_sampled_logprob()


if __name__ == "__main__":
    main()
