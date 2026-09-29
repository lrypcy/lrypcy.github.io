"""
RL 训练框架选型：把「为什么必须异步/partial rollout/高效 refit」算出来。

三组蒙特卡洛 + 一个带宽账：
  A. 同步屏障下，输出长度长尾造成的 GPU 空闲率
  B. partial rollout（截断到分位数）能救回多少
  C. 权重同步（refit）在不同互联下的耗时与占比
  D. colocate vs disaggregated 的显存账

运行：
  /Users/congyuan/Software/miniconda3/bin/python3 tools/rl_framework_budget.py
"""

import numpy as np

rng = np.random.default_rng(20260929)


def section(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


# ----------------------------------------------------------------------
# A. 同步屏障的长尾代价（槽位木桶模型）
# ----------------------------------------------------------------------
section("A. 同步 GRPO 一步：输出长度长尾 → 木桶效应吃掉多少 GPU")

# 建模：N_PROMPT 条请求随机分到 N_SLOT 个生成槽位，每槽位串行处理若干条。
# 一步的墙钟 = 最慢槽位的时间（同步屏障），利用率 = mean(slot)/max(slot)。
# 这是所有同步 RL 框架（TRL / DeepSpeed-Chat / 早期 OpenRLHF）的真实处境。

profiles = {
    "短 CoT (median 800)": dict(median=800, sigma=0.6),
    "长 CoT (median 4000)": dict(median=4000, sigma=0.9),
    "agentic (median 12000)": dict(median=12000, sigma=1.1),
}

N_PROMPT = 4096
N_SLOT = 256
PER_SLOT = N_PROMPT // N_SLOT
TOK_PER_SEC = 35.0   # 单槽位有效生成吞吐（70B 级、含 batching 的量级）
CAP = 200_000

print(f"\n设定：{N_PROMPT} prompts / step，{N_SLOT} 槽位（每槽 {PER_SLOT} 条），"
      f"单槽 {TOK_PER_SEC} tok/s\n")
print(f"{'形态':<24}{'均值':>8}{'P50':>8}{'P95':>8}{'P99':>9}{'最长':>9}"
      f"{'理想步时':>10}{'同步步时':>10}{'空闲率':>9}")
print("-" * 92)

results = {}
for name, p in profiles.items():
    L = np.clip(rng.lognormal(np.log(p["median"]), p["sigma"], N_PROMPT), 1, CAP)
    # 随机洗牌后按 PER_SLOT 分组
    Ls = rng.permutation(L)[: N_SLOT * PER_SLOT].reshape(N_SLOT, PER_SLOT)
    slot_time = Ls.sum(axis=1) / TOK_PER_SEC
    t_ideal, t_sync = slot_time.mean(), slot_time.max()
    idle = 1.0 - t_ideal / t_sync
    results[name] = dict(L=L, slot_time=slot_time, t_ideal=t_ideal,
                         t_sync=t_sync, idle=idle)
    print(f"{name:<24}{L.mean():>8.0f}{np.percentile(L,50):>8.0f}"
          f"{np.percentile(L,95):>8.0f}{np.percentile(L,99):>9.0f}{L.max():>9.0f}"
          f"{t_ideal:>10.1f}{t_sync:>10.1f}{idle:>8.1%}")

# ----------------------------------------------------------------------
# B. 固定 wall-clock 预算：截断 vs 有效样本率
# ----------------------------------------------------------------------
section("B. 固定时间预算：步时降下来了，但「没跑完的请求没有奖励」")

print("\n关键约束：一条 rollout 只有跑到 EOS 才能算奖励（RLVR 场景）。")
print("被 budget 截断的请求要么丢弃（有效样本率下降），要么保留 KV 下一步续跑。\n")
print(f"{'形态':<24}{'预算占同步步时':>14}{'完成率':>9}{'有效吞吐':>11}{'相对同步':>10}")
print("-" * 72)

for name, r in results.items():
    t_sync = r["t_sync"]
    base = N_PROMPT / t_sync     # 同步模式下每秒产出多少条**可用**样本
    for frac in (1.0, 0.8, 0.6, 0.4):
        T_b = t_sync * frac
        # 每槽位在 T_b 内能跑完多少条（按随机顺序串行）
        Ls = rng.permutation(r["L"])[: N_SLOT * PER_SLOT].reshape(N_SLOT, PER_SLOT)
        cum = np.cumsum(Ls, axis=1) / TOK_PER_SEC
        done = (cum <= T_b).sum()
        rate = done / (N_SLOT * PER_SLOT)
        thr = done / T_b
        print(f"{name:<24}{frac:>13.0%}{rate:>9.1%}{thr:>11.3f}{thr/base:>9.2f}x")
    print()

# ----------------------------------------------------------------------
# C. 权重同步（refit）的账
# ----------------------------------------------------------------------
section("C. 权重同步：70B bf16 = 140 GB，每步都要从训练布局搬到推理布局")

PARAMS_B = 70
BYTES = PARAMS_B * 1e9 * 2  # bf16
print(f"\n模型：{PARAMS_B}B，bf16 权重 = {BYTES/1e9:.0f} GB\n")

links = [
    ("NVLink (机内, 400 GB/s 有效)", 400),
    ("IB HDR (跨机 200 Gbps)", 25),
    ("PCIe Gen4 x16 (机内慢路径)", 12),
    ("以太网 100 Gbps", 12.5),
    ("NFS / checkpoint reload", 1.2),
]
print(f"{'链路':<34}{'有效带宽':>12}{'全量同步(s)':>13}{'占 60s 步时':>12}")
print("-" * 72)
for nm, gb in links:
    t = BYTES / (gb * 1e9)
    print(f"{nm:<34}{gb:>9.1f} GB/s{t:>13.2f}{t/60:>11.1%}")

print("\n优化 1：delta update（只搬变化的权重分片，verl 0.9 delta_sharded）")
print("  → 官方口径 Qwen2.5-72B 权重更新提速 3.1x（verl v0.9.0 release notes）")
for nm, gb in links[:3]:
    t = BYTES / (gb * 1e9) / 3.1
    print(f"  {nm:<32}{t:>10.2f} s")

print("\n优化 2：逐层流式 gather（3D-HybridEngine）——优化器状态从不搬移")
print(f"  70B AdamW fp32 m+v = {PARAMS_B*1e9*8/1e9:.0f} GB，")
print(f"  若误搬则慢 {PARAMS_B*1e9*8/BYTES:.0f}x。这是『同进程 reshard』相对『跨进程 reload』的核心优势。")

# ----------------------------------------------------------------------
# D. colocate vs disaggregated 显存账
# ----------------------------------------------------------------------
section("D. 显存账：为什么 70B 的 RL 至少要 4~8 卡，colocate 又省在哪")

P, PRM = 70, 7          # policy 70B，reward model 7B
GPU_MEM = 80            # H100 80GB
ACT_TRAIN, ACT_REF, KV, ACT_RM = 6.0, 3.0, 20.0, 2.0

# 每卡常驻 = 权重2P/N + 梯度2P/N + AdamW(fp32 m+v)8P/N + 激活
def per_gpu_policy(N):  return (2 + 2 + 8) * P / N + ACT_TRAIN
def per_gpu_ref(N):     return 2 * P / N + ACT_REF
def per_gpu_vllm(N):    return 2 * P / N + KV
def per_gpu_rm(N):      return 2 * PRM / N + ACT_RM

print(f"\nGRPO（无 critic）全参训 70B，reward model 7B，单卡 H100 {GPU_MEM}GB。")
print("假设 ZeRO-3 把 policy 的权重/梯度/AdamW 状态全部平摊，vLLM 副本同样按 N 分片。\n")
print(f"{'卡数':>6}{'policy':>9}{'ref':>8}{'vLLM':>8}{'RM':>7}{'合计/卡':>10}{'判定':>9}")
print("-" * 60)
for N in (8, 16, 32, 64):
    a, b, c, d = per_gpu_policy(N), per_gpu_ref(N), per_gpu_vllm(N), per_gpu_rm(N)
    tot = a + b + c + d
    ok = "OK" if tot <= GPU_MEM else "OOM"
    print(f"{N:>6}{a:>9.1f}{b:>8.1f}{c:>8.1f}{d:>7.1f}{tot:>10.1f}{ok:>9}")

print(f"\n  → 全角色 colocate 时，最小可行规模约 {32} 卡；这也是社区里 70B RL 起步就是")
print(f"    两机/四机的原因。省显存的四条路：去掉 RM（规则奖励，RLVR 主流）、")
print(f"    ref 只做 logprob 可 offload 或复用推理引擎、LoRA（ optimizer 从 8P 降到 ~0）、")
print(f"    colocate 时分复用（KV cache 与训练激活不共存，可再省 {KV:.0f} GB/卡）。")
print(f"  → disaggregated 则把 policy 池与 vLLM 池拆开，各自按自己 N 算，")
print(f"    代价是每步一次跨机权重同步 —— 见 C 节。")

# ----------------------------------------------------------------------
# E. 异步收益上界
# ----------------------------------------------------------------------
section("E. 异步化的收益上界：Amdahl 视角")

print("\n设生成占同步步时的比例 f（文献实测 60%~90%），异步能把生成与训练重叠，")
print("理论加速 = 1 / (1 - overlap × f)，overlap 为可重叠比例：\n")
print(f"{'生成占比 f':>10}" + "".join(f"{f'ov={o:.0%}':>10}" for o in (0.5, 0.7, 0.9, 1.0)))
print("-" * 52)
for f in (0.6, 0.7, 0.8, 0.9, 0.95):
    line = f"{f:>10.0%}"
    for o in (0.5, 0.7, 0.9, 1.0):
        line += f"{1/(1-o*f):>10.2f}x"
    print(line)

print("\n对照公开实测：AReaL 2.77x（arXiv 2505.24298）、veRL fully async 2.35~2.67x（128 GPU）、")
print("LlamaRL 分离+异步 2.52/3.98/10.7x（8B/70B/405B，arXiv 2505.24034）。")
print("→ 都落在 f≈0.8、overlap≈0.7~0.95 的区间内，量级自洽。")

print("\n" + "=" * 68)
print("注：本脚本是量级模型，用于给选型讨论定标，不是某框架的实测吞吐。")
print("=" * 68)
