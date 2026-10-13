"""Megatron-LM 解析工具箱入口。

无 GPU 环境下的统一入口：所有数字都能在 CPU 上复算出来。

用法:
    python3 -m megatron_bench --model llama-7b --hardware h100-80g --tp 2 --dp 8
    python3 -m megatron_bench --model gpt3-175b --hardware a100-80g --mem --recompute full
"""

from __future__ import annotations

import argparse
import sys

from . import comm, configs, flops, mem
from .mem import activation_inventory_per_layer, bytes_to_gib, full_ledger


Gb = 1024 ** 3


def _fmt(v: float, unit: str = "") -> str:
    if v >= 1e12:
        return f"{v / 1e12:,.2f} T{unit}"
    if v >= 1e9:
        return f"{v / 1e9:,.2f} G{unit}"
    if v >= 1e6:
        return f"{v / 1e6:,.2f} M{unit}"
    return f"{v:,.2f} {unit}"


def report_flops(cfg):
    m = configs.MODELS[cfg.model]
    print("\n### FLOPs 记账 —— %s" % cfg.model)
    print("    source: %s" % m.source)
    fwd = flops.forward_flops_per_token(cfg)
    print("\n每 token forward FLOPs（逐算子）:")
    for name, v in fwd.as_rows():
        print("    %-30s %s" % (name, _fmt(v, "FLOP")))
    print("    %-30s %s" % ("合计（含 lm_head）", _fmt(fwd.forward_total(), "FLOP")))

    tr = flops.training_flops_per_token(cfg)
    print("\n训练每 token（backward = 2x forward）:")
    for k in ("total_none", "total_selective_core_attn", "total_full_recompute"):
        print("    %-30s %s" % (k, _fmt(tr[k], "FLOP")))

    cmp = flops.compare_exact_vs_6nd(cfg)
    print("\n精确口径 vs 6ND 近似:")
    print("    6ND                        %s" % _fmt(cmp["6nd"], "FLOP"))
    print("    精确                       %s" % _fmt(cmp["exact_loss_head_included"], "FLOP"))
    print("    相对偏差                   %+.2f%%" % cmp["rel_diff_pct"])
    return cmp


def report_mem(cfg, args):
    print("\n### 显存账本 —— %s / %s" % (cfg.model, cfg.hardware))
    led = full_ledger(
        cfg,
        sequence_parallel=args.sequence_parallel,
        distribute_saved_activations=args.distribute_saved,
        precision_aware=args.precision_aware,
    )
    hw = configs.HARDWARE[cfg.hardware]
    print("    并行: tp=%d cp=%d pp=%d ep=%d dp=%d   recompute=%s  SP=%s"
          % (cfg.tp, cfg.cp, cfg.pp, cfg.ep, cfg.dp, cfg.recompute, args.sequence_parallel))
    print("\n四项合计:")
    for name, v in led.summary():
        print("    %-12s %s GiB" % (name, _fmt(bytes_to_gib(v))))
    print("    %-12s %s GiB" % ("总计", _fmt(bytes_to_gib(led.total_bytes()))))
    print("    %-12s %s GiB (%.1f%% 用满)"
          % ("单卡容量", f"{hw.mem_gb:.1f}",
             100 * bytes_to_gib(led.total_bytes()) / hw.mem_gb))

    if args.verbose:
        rows, extras = activation_inventory_per_layer(cfg)
        print("\n单层激活清单 (micro_batch=%d, seq/cp=%d, tp=%d):"
              % (cfg.micro_batch, int(extras["s_local"]), cfg.tp))
        for r in rows:
            print("    %-34s %s" % (r.name, _fmt(r.bytes_per_rank, "B")))
    return led


def report_comm(cfg, args):
    print("\n### 通信账本 —— %s" % cfg.model)
    bw = comm.bandwidth_report(cfg)
    print("    链路带宽估计: " + "  ".join(f"{k}={v:.0f}GB/s" for k, v in bw.items()))

    events = comm.tensor_parallel_events(cfg, sequence_parallel=args.sequence_parallel)
    layers_local = configs.MODELS[cfg.model].n_layers // cfg.pp
    total_bytes = 0.0
    print("\n每层 TP 通信:")
    for e in events:
        b = e.total_bytes(cfg.tp) * layers_local
        total_bytes += b
        print("    %-14s x%-2d  %-10s  %s  %s"
              % (e.kind, e.count_per_layer, e.group,
                 _fmt(e.total_bytes(cfg.tp) * layers_local), e.note))
    print("    单层 TP 总线字节: %s" % _fmt(total_bytes / max(layers_local, 1)))
    print("    本 stage 全部层:  %s" % _fmt(total_bytes))

    dpg = comm.data_parallel_grad_bytes(cfg)
    print("\nDP 梯度归约（每 step，单卡 bus 口径）: %s" % _fmt(dpg))
    ppb = comm.pipeline_point_to_point_bytes(cfg)
    print("PP 点到点（单 microbatch 一次收发）: %s" % _fmt(ppb))
    return dict(tp_bytes=total_bytes, dp_bytes=dpg, pp_bytes=ppb)


def build_cfg(args) -> configs.BenchConfig:
    return configs.BenchConfig(
        model=args.model, hardware=args.hardware,
        tp=args.tp, cp=args.cp, pp=args.pp, ep=args.ep, dp=args.dp,
        num_nodes=args.num_nodes, micro_batch=args.micro_batch,
        grad_accum=args.grad_accum, seq_len=args.seq_len,
        recompute=args.recompute,
        use_distributed_optimizer=not args.no_distributed_optimizer,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Megatron-LM 解析工具箱（无 GPU）")
    ap.add_argument("--model", default="llama-7b", choices=sorted(configs.MODELS))
    ap.add_argument("--hardware", default="h100-80g", choices=sorted(configs.HARDWARE))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--num-nodes", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--recompute", default="none",
                    choices=["none", "full", "selective"])
    ap.add_argument("--sequence-parallel", action="store_true")
    ap.add_argument("--distribute-saved", action="store_true")
    ap.add_argument("--precision-aware", action="store_true")
    ap.add_argument("--no-distributed-optimizer", action="store_true")
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--comm", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    cfg = build_cfg(args)
    print("=" * 70)
    print("Megatron-LM 解析工具箱 | model=%s hardware=%s | world=%d"
          % (cfg.model, cfg.hardware, cfg.tp * cfg.cp * cfg.pp * cfg.ep * cfg.dp))
    print("=" * 70)

    run_all = not (args.flops or args.mem or args.comm)
    if run_all or args.flops:
        report_flops(cfg)
    if run_all or args.mem:
        report_mem(cfg, args)
    if run_all or args.comm:
        report_comm(cfg, args)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
