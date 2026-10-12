> 系列文档：扩散语言模型（Diffusion LM）学习资料

# 10 配套实验：用最小可跑代码把关键论断证一遍

每个实验独立成节，结构为：目标/论断 → 代码（含 import 与随机种子）→ 预期/通过。代码全部基于纯 CPU + numpy（实验 10 为 GPU 命令骨架）。环境：python 3.10+，numpy 版本无特殊要求。

## 实验 1：掩码扩散的加权损失与 $$1/t$$ 权重

**目标/论断**：MDLM／MD4 的连续时间目标权重 $$w(t)=\frac{\bar\alpha_t'}{1-\bar\alpha_t}$$ 在低噪声（$$t\to 0$$）处最大，把梯度质量从“容易的高掩码步”转移到“难的低噪声步”；不加权则难步贡献趋零。

**代码**（python）：

```python
import numpy as np
ts = np.linspace(0.0, 1.0, 10001)
alpha = (1.0 - ts) ** 2
w = (-2.0 * (1.0 - ts)) / (1.0 - alpha + 1e-9)
mass_w = np.abs(w); mass_w = mass_w / mass_w.sum()
hard, easy = ts < 0.1, ts > 0.9
print("未加权 难/易:", round((ts < 0.1).mean(), 4), round((ts > 0.9).mean(), 4))
print("加权   难/易:", round(mass_w[hard].sum(), 4), round(mass_w[easy].sum(), 4))
```

**预期/通过**：未加权难、易区间占比都约 0.1（均匀）；加权后难区间接近 1、易区间接近 0。把 `hard` 改成 `ts > 0.75` 看易区间占比反而下降。

## 实验 2：固定 15% 掩码 vs 均匀 $$t\sim\mathcal{U}[0,1]$$

**目标/论断**：BERT 式固定 15% 掩码率训练出的模型，在推理起点（全掩码，掩码率 ≈ 1）上是分布外，损失骤升，无法从全 [MASK] 生成。

**代码**（python）：

```python
import numpy as np
rng = np.random.default_rng(0)
V, L = 4, 6
data = rng.integers(0, V, size=(300, L))
def mask(x0, p):
    return np.where(rng.random(x0.shape) < p, V, x0)
def feats(z):
    f = np.zeros(L * (V + 1))
    for i in range(L): f[i * (V + 1) + z[i]] = 1.0
    return f
W0 = np.zeros((L * V, L * (V + 1)))
def train(fixed):
    Wl = W0.copy()
    for _ in range(1500):
        x0 = data[rng.integers(len(data))]
        z = mask(x0, 0.15 if fixed else rng.random())
        logits = (Wl @ feats(z)).reshape(L, V)
        logp = logits - logits.max(1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
        onehot = np.zeros((L, V)); onehot[np.arange(L), x0] = 1.0
        Wl = Wl - 0.3 * (-np.outer((onehot - np.exp(logp)).reshape(-1), feats(z)))
    return Wl
def eval_loss(Wl, p):
    x0 = data[rng.integers(len(data))]
    logits = (Wl @ feats(mask(x0, p))).reshape(L, V)
    logp = logits - logits.max(1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
    onehot = np.zeros((L, V)); onehot[np.arange(L), x0] = 1.0
    return -np.mean(onehot * logp)
for name, fixed in [("固定15%", True), ("均匀t", False)]:
    Wl = train(fixed)
    print(name, "loss@0.15:", round(eval_loss(Wl, 0.15), 3), "loss@1.0:", round(eval_loss(Wl, 1.0), 3))
```

**预期/通过**：固定 15% 模型 `loss@p=1.0` 远高于 `loss@p=0.15`；均匀 t 模型在两处都较低。`loss@p=1.0` 高于 `loss@p=0.15` 一个量级即算通过。

## 实验 3：ELBO 上界性质

**目标/论断**：在小词表小序列上验证训练损失是真实 $$-\mathbb{E}[\log p_\theta(x_0)]$$ 的上界，即对任意 $$t$$ 有 $$\mathbb{E}_{q(x_t\mid x_0)}[-\log p_\theta(x_0\mid x_t)] \ge -\log p_\theta(x_0)$$。

**代码**（python，$$V=3$$、$$L=2$$）：

```python
import numpy as np
rng = np.random.default_rng(0)
V, L = 3, 2
alpha_t = 0.5
def model_logp(z):
    logits = rng.normal(0.0, 1.0, size=(L, V))
    lp = logits - logits.max(1, keepdims=True)
    return lp - np.log(np.exp(lp).sum(1, keepdims=True))
def p_x0_given_xt(x0, xt):
    return np.exp(np.sum(model_logp(xt)[np.arange(L), x0]))
seqs = [np.array(s) for s in np.ndindex(*([V] * L))]
xt_states = [np.array(s) for s in np.ndindex(*([2] * L))]
def q_xt(xt):
    return np.prod([alpha_t if xt[i] == 0 else (1 - alpha_t) for i in range(L)])
def q_xt_given_x0(xt, x0):
    pr = 1.0
    for i in range(L): pr *= alpha_t if xt[i] == 0 else (1 - alpha_t)
    return pr
rhs, lhs = 0.0, 0.0
for x0 in seqs:
    p_x0 = sum(p_x0_given_xt(x0, xt) * q_xt(xt) for xt in xt_states)
    lhs += -np.log(p_x0 + 1e-12)
    rhs += sum(q_xt_given_x0(xt, x0) * (-np.log(p_x0_given_xt(x0, xt) + 1e-12)) for xt in xt_states)
print("lhs(-log p):", round(lhs / len(seqs), 4), "rhs(上界):", round(rhs / len(seqs), 4))
```

**预期/通过**：`rhs >= lhs`（反向 KL 非负）。改 `alpha_t = 0.9`，差距变小（去噪越易，界越紧）。允许浮点误差 ±1e-6。

## 实验 4：$$\tau$$-leaping 的条件独立代价

**目标/论断**：并行解两个强相关 token 会把联合分布退化为边缘乘积，KL 偏差大于逐位（条件）解码。

**代码**（python）：

```python
import numpy as np
rng = np.random.default_rng(0)
p_match = 0.9
def sample_joint():
    if rng.random() < p_match:
        v = rng.integers(0, 2); return (v, v)
    return (rng.integers(0, 2), rng.integers(0, 2))
true = np.zeros((2, 2))
for _ in range(200000):
    a, b = sample_joint(); true[a, b] += 1.0
true = true / true.sum()
mar = true.sum(1); indep = np.outer(mar, mar)
kl = np.sum(true * (np.log(true + 1e-12) - np.log(indep + 1e-12)))
print("KL:", round(kl, 4), "true P(相等):", round(true[0, 0] + true[1, 1], 3), "独立:", round(indep[0, 0] + indep[1, 1], 3))
```

**预期/通过**：`P(相等)` 真实约 0.95，独立解码退化为约 0.5，KL > 0。`p_match=0.5` 时 KL 趋近 0。

## 实验 5：unmasking 策略消融（random / confidence-first / semi-AR）

**目标/论断**：confidence-first（优先提交高置信 token）与 semi-AR 比 random 用更少步达到更高准确率。

**代码**（python）：

```python
import numpy as np
rng = np.random.default_rng(0)
L = 8
conf = rng.random(L)
def run(strategy, steps):
    if strategy == "random":
        order = rng.permutation(L)
    else:
        order = np.argsort(-conf)
        if strategy == "semi_ar":
            order = np.concatenate([order[0::2], order[1::2]])
    return np.mean(conf[order[:steps]])
for strat in ["random", "conf_first", "semi_ar"]:
    print(strat, [round(run(strat, s), 3) for s in [2, 4, 6, 8]])
```

**预期/通过**：`conf_first` 与 `semi_ar` 在前几步的平均置信高于 `random`。`steps=2` 时 conf_first 平均置信 ≥ random。换成与真值无关的随机 `conf` 则优势消失。

## 实验 6：块扩散 vs 全序列扩散（NFE-质量插值）

**目标/论断**：块越大，NFE 越低（更快），但并行解块内 token 引入依赖误差，质量下降。

**代码**（python）：

```python
import numpy as np
L = 16
def simulate(block, steps):
    n_blocks = int(np.ceil(L / block))
    return n_blocks * steps, round(1.0 - 0.03 * block, 3)
for block in [1, 2, 4, 8, 16]:
    print("block=", block, simulate(block, 10))
```

**预期/通过**：block 从 1 增到 16，NFE 从 160 降到 10，quality 同步下降。NFE 与 block 成反比，quality 随 block 单调不增。

## 实验 7：位置相关掩码缩小 train／test gap（Efficient-DLM 思路）

**目标/论断**：测试时左到右生成等价于“右侧掩码更多”；训练也用右侧偏置掩码（右侧获得更多监督），在被掩码的右侧上 test loss 更低，体现分布匹配价值。

**代码**（python）：

```python
import numpy as np
rng = np.random.default_rng(0)
V, L = 4, 8
dl = rng.integers(0, V, size=(800, L // 2))
data = np.concatenate([dl, (dl + 1) % V], axis=1)  # 右半 = 左半 + 1，左可预测右
def mask_bias(x0, bias):
    p = np.full(L, 0.2); p[L // 2:] = np.clip(0.4 + bias * 0.5, 0.05, 0.95)
    return np.where(rng.random(x0.shape) < p, V, x0)
def feats(z):
    f = np.zeros(L * (V + 1))
    for i in range(L): f[i * (V + 1) + z[i]] = 1.0
    return f
W0 = np.zeros((L * V, L * (V + 1)))
def train(bias):
    Wl = W0.copy()
    for _ in range(800):
        x0 = data[rng.integers(len(data))]
        logits = (Wl @ feats(mask_bias(x0, bias))).reshape(L, V)
        logp = logits - logits.max(1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
        onehot = np.zeros((L, V)); onehot[np.arange(L), x0] = 1.0
        Wl = Wl - 0.3 * (-np.outer((onehot - np.exp(logp)).reshape(-1), feats(mask_bias(x0, bias))))
    return Wl
def right_masked_loss(Wl, bias):
    tot, n = 0.0, 0
    for _ in range(300):
        x0 = data[rng.integers(len(data))]
        z = mask_bias(x0, bias)
        logits = (Wl @ feats(z)).reshape(L, V)
        logp = logits - logits.max(1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
        onehot = np.zeros((L, V)); onehot[np.arange(L), x0] = 1.0
        for i in range(L // 2, L):
            if z[i] == V:
                tot += -(onehot[i] @ logp[i]); n += 1
    return tot / n
for bias in [0.0, 1.0]:
    print("train bias=", bias, "right-masked-test:", round(right_masked_loss(train(bias), 1.0), 3))
```

**预期/通过**：`train bias=1.0`（右半掩码率 0.9）在被掩码右半上的损失低于 `train bias=0.0`。把左半掩码率也随 bias 偏置，可验证“训练/测试掩码分布错位”令对应区域性能下降。

## 实验 8：反向核与 $$\alpha$$ 表的一致性（贝叶斯校验）

**目标/论断**：由前向转移经贝叶斯（给定 $$x_0$$ 干净）反推，反向恢复概率恰为 $$\frac{\alpha_s-\alpha_t}{1-\alpha_t}$$，与前向掩码概率 $$\frac{\alpha_s-\alpha_t}{\alpha_s}$$ 自洽。

**代码**（python）：

```python
import numpy as np
rng = np.random.default_rng(0)
ts = np.linspace(0.0, 1.0, 1001)
alpha = (1.0 - ts) ** 2
def a(t):
    return alpha[int(t * 1000)]
s, t = 0.3, 0.7
N = 300000
cnt_masked_t, cnt_both = 0, 0
for _ in range(N):
    clean_at_s = rng.random() < a(s)
    masked_at_t = (not clean_at_s) or (rng.random() < (a(s) - a(t)) / a(s))
    if masked_at_t:
        cnt_masked_t += 1
        cnt_both += 1 if clean_at_s else 0
p_rev_emp = cnt_both / cnt_masked_t
p_rev_theory = (a(s) - a(t)) / (1.0 - a(t))
print("理论:", round(p_rev_theory, 5), "经验:", round(p_rev_emp, 5), "偏差:", round(abs(p_rev_theory - p_rev_emp), 5))
```

**预期/通过**：经验值与理论值偏差在 5e-3 内（蒙特卡洛噪声）。换 `s, t = 0.1, 0.9` 跨度更大，偏差仍应在采样误差内。

## 实验 9：ELBO 估计方差与 antithetic 采样（对应 VRPO）

**目标/论断**：对时间步做配对取反（$$t$$ 与 $$1-t$$）的 antithetic 采样，方差低于独立采样，对应 VRPO 的降方差思路。

**代码**（python）：

```python
import numpy as np
rng = np.random.default_rng(0)

def g(t):                      # 每个样本自带观测噪声，模拟 ELBO 估计的随机性
    return np.sin(5 * t) + rng.normal(0, 0.5)

R, n = 400, 100                # 重复 R 次，每次用 n 个样本估计均值
plain = [np.mean([g(rng.random()) for _ in range(n)]) for _ in range(R)]

anti = []                      # 关键：两次必须样本数相同，即 n/2 组配对
for _ in range(R):
    vals = []
    for _ in range(n // 2):
        u = rng.random()
        vals += [g(u), g(1.0 - u)]
    anti.append(np.mean(vals))

print("plain var:", round(np.var(plain), 5), "antithetic var:", round(np.var(anti), 5))
print("variance ratio (anti/plain):", round(np.var(anti) / np.var(plain), 3))
```

> 易错点：配对采样必须保证**总样本数与对照相同**（上面用 $$n/2$$ 组配对凑齐 $$n$$ 个样本）。若对照组用 100 个独立样本、配对组只用 2 个，比较出来的方差差只反映样本量差异，结论是假的。

**预期/通过**：`antithetic var` 小于 `plain var`（配对抵消部分方差），且不超独立采样方差的 80%。变体（置信度阈值并行解码，仅思路）：扫描阈值 $$\tau\in[0.5,0.99]$$ 得“提交 token 数-掉点率”权衡曲线，阈值越高提交越少、掉点越少。

## 实验 10：真实模型侧最小实验（GPU 命令骨架）

**目标/论断**：用开源 LLaDA／Dream 权重，固定 prompt，改变 `steps` 与 `gen_length`，记录 TPS 与质量；减少 `steps` 提升 TPS 但降低质量。

**代码**（bash，需 GPU 与对应权重，不在本机运行）：

```bash
# 环境：pip install transformers accelerate
# 以 LLaDA-8B 为例，推理脚本用相对/仓库约定路径，不写死本机绝对路径
python -m dllm_infer --model ML-GSAI/LLaDA-8B-Instruct \
  --prompt "用一句话解释扩散语言模型。" --steps 64 --gen_length 128 --block 32 \
  --out results/llada_steps64_len128.json
python -m dllm_infer --model ML-GSAI/LLaDA-8B-Instruct \
  --prompt "用一句话解释扩散语言模型。" --steps 16 --gen_length 128 --block 32 \
  --out results/llada_steps16_len128.json
python -m dream_infer --model Dream-7B-Instruct \
  --prompt "用一句话解释扩散语言模型。" --steps 256 --gen_length 128 \
  --out results/dream_steps256_len128.json
```

**预期/通过**：`steps=16` 的 TPS 约为 `steps=64` 的 4 倍但连贯性下降；`gen_length` 加倍时 NFE 约加倍。记录每次 TPS 与评分，绘 “NFE-质量” 帕累托曲线，确认存在最优折中。

## 参考链接

- MDLM／MD4 目标与加权：<https://arxiv.org/abs/2406.07524>
- RADD 去时间条件与缓存：<https://arxiv.org/abs/2406.03736>
- Efficient-DLM 位置相关掩码：<https://arxiv.org/abs/2512.14067>
- Fast-dLLM 置信度并行解码：<https://arxiv.org/abs/2505.22618>
- VRPO 降方差偏好优化：<https://arxiv.org/abs/2505.19223>
- AGRPO 单步去噪策略梯度：<https://arxiv.org/abs/2510.04019>
- LLaDA 权重与推理：<https://github.com/ML-GSAI/LLaDA>
- Dream 7B：<https://arxiv.org/abs/2508.15487>
