#!/usr/bin/env python3
"""复刻 Megatron-LM `forward_backward_no_pipelining` 的控制流，并验证两件事。

无水播出 Sweep：本机没有 GPU，也不装 torch，所以整个 lab 用 numpy 手写前向/反向。

验证目标（C 级证据）：
 1. DDP 等价性 —— 多 rank 各自算一半 batch 再reduce grads 平均，与单 rank 算整个 batch
    得到的更新必须一致（这正是数据并行成立的前提）。
 2. microbatch 梯度累加 —— 一个 step 拆成若干 microbatch 累加梯度，与一次性算完必须一致
    （这是 gradient accumulation 成立的前提）。

用法:
    python3 deep-dive-megatron/labs/01_mini_train_loop.py
"""

import argparse
import cProfile
import pstats
from io import StringIO

import numpy as np


# ---------------------------------------------------------------- 参数与工具

class Linear:
    """一个 nn.Linear。手写是为了把反向 happens-where 暴露出来。"""

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, scale: float = 0.02):
        self.W = rng.normal(0, scale, size=(d_out, d_in))
        self.b = np.zeros(d_out)
        self.cache = None
        # 模拟 Megatron 里 param + main_param 的双份语义（这里只留一份 wgrad）
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        # x: [T, d_in]
        y = x @ self.W.T + self.b
        self.cache = x
        return y

    def backward(self, dy):
        # dy: [T, d_out]
        x = self.cache
        self.dW += dy.T @ x          # 权梯度
        self.db += dy.sum(axis=0)
        dx = dy @ self.W             # 输入梯度（要往回传）
        return dx

    def reset_grads(self):
        self.dW[:] = 0.0
        self.db[:] = 0.0


class TransformerBlock:
    """极简 transformer 层：LN -> attn 用一层线性近似 -> MLP(gelu)。"""

    def __init__(self, h: int, ffn: int, rng: np.random.Generator):
        self.qkv = Linear(h, 3 * h, rng)
        self.out_proj = Linear(h, h, rng)
        self.up = Linear(h, ffn, rng)
        self.down = Linear(ffn, h, rng)
        self.cache = []

    def params(self):
        return [self.qkv, self.out_proj, self.up, self.down]

    def forward(self, x):
        self.cache = []
        h0 = layernorm(x)
        self.cache.append(h0)
        qkv = self.qkv.forward(h0)
        # 用一层廉价近似替代真实 attention：保持 [T,h] 形状，不做 s^2 计算
        attn = np.tanh(qkv[:, : x.shape[-1]])
        self.cache.append(attn)
        a = self.out_proj.forward(attn)
        self.cache.append(a)
        h1 = layernorm(x + a)
        self.cache.append(h1)
        u = self.up.forward(h1)
        u = gelu(u)
        self.cache.append(u)
        y = self.down.forward(u)
        return x + y

    def backward(self, dy):
        u, h1, a, attn, h0 = self.cache[4], self.cache[3], self.cache[2], self.cache[1], self.cache[0]
        du = self.down.backward(dy)
        du = gelu_backward(du, u)
        dh1 = self.up.backward(du)
        dh1 = layernorm_backward(dh1)      # 近似：忽略 LN 的可学习参数反传细节
        da = self.out_proj.backward(dh1)
        dattn = da * (1.0 - attn ** 2)   # d(tanh)/dx
        dqkv = np.zeros((da.shape[0], 3 * da.shape[1]))
        dqkv[:, : da.shape[1]] = dattn
        dh0 = self.qkv.backward(dqkv)
        dh0 = layernorm_backward(dh0)
        return dy + dh0                     # 残差分支


def layernorm(x, eps: float = 1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def layernorm_backward(dy):
    """近似版：只做中心化部分的反传，够用就行，lab 里不追求严格。"""
    return dy - dy.mean(axis=-1, keepdims=True)


def gelu(x):
    """用 tanh 近似 erf，避开 scipy 依赖。"""
    return x * 0.5 * (1.0 + np.tanh(1.1283791671 * x / np.sqrt(2.0)))


def _erf(v):
    # numpy 没有内置 erf，用 tanh 近似（lab 精度足够，且省依赖）
    return np.tanh(1.1283791671 * v)


def gelu_backward(dy, u):
    return dy * np.where(u > 0, 1.0, 0.05)   # 近似导数


# ---------------------------------------------------------------- 训练流程

def build_model(h, ffn, n_layers, rng):
    return [TransformerBlock(h, ffn, rng) for _ in range(n_layers)]


def all_params(model):
    ps = []
    for blk in model:
        ps.extend(blk.params())
    return ps


def forward_step(model, batch):
    """对应 Megatron 的 forward_step：返回 (loss_scalar, grad_of_loss)。"""
    x, target = batch
    for blk in model:
        x = blk.forward(x)
    diff = x - target
    loss = float((diff ** 2).mean())
    return loss, diff


def backward_blocks(model, dy):
    for blk in reversed(model):
        dy = blk.backward(dy)


def train_step_single_rank(model, batches, lr):
    """一个 step 的完整控制流：累加 microbatch 梯度 -> 更新。

    对应 forward_backward_no_pipelining 的简化版（没有 PP，也没有通信）。
    """
    scale = 2.0 / (len(batches) * next(iter(batches))[0].size)  # d(mean sq)/dx
    losses = []
    for batch in batches:
        loss, diff = forward_step(model, batch)
        losses.append(loss)
        backward_blocks(model, diff * scale)
    for p in all_params(model):
        p.W -= lr * p.dW
        p.b -= lr * p.db
        p.reset_grads()
    return float(np.mean(losses))


def clone_params(model):
    return [(p.W.copy(), p.b.copy()) for p in all_params(model)]


def assign_params(model, snapshot):
    for p, (W, b) in zip(all_params(model), snapshot):
        p.W[:] = W
        p.b[:] = b


def make_data(rng, n_tokens, h, micro_batch, n_micro, seed=0):
    r = np.random.default_rng(seed)
    batches = []
    for _ in range(n_micro):
        x = r.normal(0, 1, size=(micro_batch, h))
        t = r.normal(0, 1, size=(micro_batch, h))
        batches.append((x, t))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--microbatches", type=int, default=3)
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    h, n_micro = args.hidden, args.microbatches
    rng = np.random.default_rng(1234)

    # ---- 实验 1：microbatch 累加 vs 一次性算完 ---------------------------------
    model_a = build_model(h, args.ffn, args.layers, np.random.default_rng(7))
    model_b = build_model(h, args.ffn, args.layers, np.random.default_rng(7))
    batches = make_data(rng, 1, h, args.micro_batch, n_micro, seed=11)

    loss_a = train_step_single_rank(model_a, batches, args.lr)

    # 把所有 microbatch 拼成一个大 batch 一次性算
    mega_x = np.concatenate([b[0] for b in batches], axis=0)
    mega_t = np.concatenate([b[1] for b in batches], axis=0)
    loss_b = train_step_single_rank(model_b, [(mega_x, mega_t)], args.lr)

    pa, pb = clone_params(model_a), clone_params(model_b)
    max_grad_diff = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(pa, pb))
    print("=" * 66)
    print("实验 1：microbatch 梯度累加 == 一次性计算 ？")
    print("=" * 66)
    print(f"  累加 {n_micro} 个 microbatch 的 loss : {loss_a:.8f}")
    print(f"  一次性算同样数据的 loss          : {loss_b:.8f}")
    print(f"  更新后权重的最大绝对差           : {max_grad_diff:.3e}")

    # ---- 实验 2：DDP 等价性 ----------------------------------------------------
    print()
    print("=" * 66)
    print("实验 2：多 rank 分数据 + 梯度平均 == 单 rank 全量 ？")
    print("=" * 66)
    model_c = build_model(h, args.ffn, args.layers, np.random.default_rng(7))
    # 关键：两条路径必须吃**同一份**数据，否则比的是随机数而不是算法。
    # 先生成按 rank 切好的小批次，再拼成单 rank 的全量批次。
    ranks_data = make_data(rng, 1, h, args.micro_batch, args.ranks, seed=11)
    all_batches = [(
        np.concatenate([b[0] for b in ranks_data], axis=0),
        np.concatenate([b[1] for b in ranks_data], axis=0),
    )]
    loss_single = train_step_single_rank(model_c, all_batches, args.lr)
    ref_after = clone_params(model_c)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    model_d = build_model(h, args.ffn, args.layers, np.random.default_rng(7))
    n_elements_per_rank = ranks_data[0][0].size
    for rank_batches in ranks_data:
        loss_rank, diff = forward_step(model_d, rank_batches)
        # 每个 rank 算自己那份的平均 loss，梯度除以全局元素数后再求和 == 全量梯度
        scale = 2.0 / (args.ranks * n_elements_per_rank)
        backward_blocks(model_d, diff * scale)
    for p in all_params(model_d):
        p.W -= args.lr * p.dW
        p.b -= args.lr * p.db
        p.reset_grads()
    if args.profile:
        pr.disable()
        sio = StringIO()
        pstats.Stats(pr, stream=sio).sort_stats("cumulative").print_stats(8)
        print("\n--- cProfile: 单 step CPU 热点 top8 ---")
        print("\n".join(sio.getvalue().split("\n")[:18]))

    cls = clone_params(model_d)
    max_diff = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(ref_after, cls))
    print(f"  单 rank 全量 loss               : {loss_single:.8f}")
    print(f"  {args.ranks} 个 rank 各自半份数据 + 梯度平均 (lr={args.lr})")
    print(f"  更新后权重的最大绝对差           : {max_diff:.3e}")
    verdict = "通过" if max_diff < 1e-9 else "未通过"
    print(f"  等价性结论: {verdict}  (最大差 {max_diff:.3e})")


if __name__ == "__main__":
    main()
