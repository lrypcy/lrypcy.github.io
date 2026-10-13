---
title: "PyTorch 采样实现深读：multinomial 的 fast path、LLM 的 logits 塑形与 VLA 的连续动作运输"
date: 2026-09-30 23:30:00 +0800
categories:
  - 系统实现
tags: [pytorch, sampling, multinomial, gumbel-max, categorical, logprob, top-k, top-p, min-p, transformers, vllm, spec-decoding, reparameterization, gumbel-softmax, diffusion, flow-matching, vla, rl, reproducibility]
layout: post
mathjax: true
---

> **系列导航** ｜ 相关前作：[vLLM 推理服务 profiling](/2026/09/29/profiling-02-inference/) ｜ [RL 符号与估计器全景](/2026/09/19/rl-notation-and-estimators-deepdive/) ｜ [World Model 与 WAM](/2026/09/05/world-model-world-action-model/) ｜ [GPU 超越函数与 softmax](/2026/09/26/gpu-transcendental-softmax-llm/)

> **TL;DR**
>
> * **`torch.multinomial` 的 fast path 是一条 Gumbel 路径，不是逆变换。** 源码里的条件是 `if (!with_replacement || n_sample == 1)`——**只要只采一个样本**（LLM 每步推理的全部情形），就走 `argmax(p/q)`，`q ~ Exp(1)`。实测在 40 万次采样上，它与 vLLM 手写的 `random_sample` **逐元素 100% 一致**；而每行消耗的均匀数正好是类别数 $$K$$，稳居逆变换采样的 $$K$$ 倍。这也解释了为什么 vLLM 敢直接手写 `probs.div_(q).argmax(-1)` 替代它——两者本来就是同一段数学，替换的收益只在避开 CPU-GPU 同步，不在改变分布（§2）。
> * **`Categorical` 的 `probs` 与 `logits` 两条构造路径不是等价的。** 传 `probs` 要过 `clamp_probs`，夹到 $$[\varepsilon, 1-\varepsilon]$$，float64 下 $$\varepsilon = 2.22\times 10^{-16}$$，于是任何概率低于 $$\varepsilon$$ 的 token，其 `log_prob` 一律被抬到 $$\log \varepsilon = -36.04$$。实测一个真实 logprob 应为 $$-120.0$$ 的 token，走 `Categorical(probs=...)` 会报成 $$-36.04$$，**偏差 83.96 nat**；而被 mask 成 $$-\infty$$ 的 token 报成 $$-36.04$$ 而不是 $$-\infty$$。RL 训练里传错参数，ratio 直接失真（§5）。
> * **采样器与打分器的口径必须逐项对齐。** 只把打分侧从 fp64 降到 bf16（采样侧不动），`exp(Δlogp)` 就偏离 1 达 $$3.5\times 10^{-2}$$；proposal 做了 top-k 截断而 target 没做，由于截断后概率质量更集中，正确的重要性权重实测落在 $$[0.415, 0.9998]$$——**恒小于 1，均值偏 14%**。这两类偏差都会被 PPO/GRPO 当成真实策略漂移来惩罚（§6）。
> * **连续动作的“采样”是一次数值积分，少步数会塌分布。** 在双模态目标上，flow matching 用 1 步 Euler 时输出标准差为 **0.0000**——所有噪声样本被推到同一个点（条件均值），多模态信息被积分误差完全抹平；DDPM 更麻烦，把 500 步抽成 20 步子序列会让与目标的直方图 TV 从 $$0.0156$$ 暴涨到 $$0.7164$$，因为祖先采样的后验方差公式绑死在相邻步上。这解释了为什么 VLA 里“10 步”是个需要小心对待的默认值（§8）。

---

## 1. 采样在 PyTorch 里是三层抽象

讨论“PyTorch 怎么实现采样”时，最容易混淆的是把三件不同层面的事当成一件。它们各自的代码路径、性能特征、正确性陷阱都不一样：

| 层 | 代表 API | 职责 | 典型用户 |
| :--- | :--- | :--- | :--- |
| **分布对象层** | `torch.distributions.Categorical`、`Normal` | 参数校验、归一化、`log_prob` / `entropy` / `rsample` | 教学、小规模实验、变分推断 |
| **采样原语层** | `torch.multinomial`、`torch.rand` / `exponential_`、`torch.Generator` | 从已归一化的权重里抽索引；提供随机源 | 几乎所有人 |
| **logits 塑形层** | `top-k` / `top-p` / `min-p` / 温度 / 惩罚 | 修改分布本身 | transformers、vLLM、SGLang |

本文按这三层往下走，再分别看它们在 LLM 与 VLA 里的具体形态。

### 1.1 分布对象层的架构：`sample` 默认就是 `rsample`

`torch.distributions.Distribution` 基类只有 349 行，但其中两段决定了整个体系的形状（`torch/distributions/distribution.py:168-182`）：

```python
def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
    with torch.no_grad():
        return self.rsample(sample_shape)

def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
    raise NotImplementedError
```

**默认实现是重参数化采样，而不是反过来。** 基类把 `sample` 定义为“套了 `no_grad` 的 `rsample`”，把 `rsample` 定义为抽象方法。任何子类只要实现了可微的那条路径，就免费得到不可微的那条。

这也意味着一个反常识的事实：**`Normal.sample()` 是有梯度的阻塞版 `rsample`，而不是两套独立实现**。但子类可以推翻它——`Normal` 就推翻了两者（`torch/distributions/normal.py:78-86`）：

```python
def sample(self, sample_shape=torch.Size()):
    shape = self._extended_shape(sample_shape)
    with torch.no_grad():
        return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
    shape = self._extended_shape(sample_shape)
    eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
    return self.loc + eps * self.scale
```

`rsample` 就是 $$\text{loc} + \varepsilon \cdot \text{scale}$$，教科书式的重参数化；`sample` 则用 `torch.normal(mean, std)` 一次生成，省掉先造 $$\varepsilon$$ 再乘的开销。两者数学同分布，但 RNG 消耗路径不同。

配套的机制是一个布尔标志 `has_rsample`：`Normal` 和 `Gamma` 为 `True`，`Categorical` / `Bernoulli` 等离散分布为 `False`（离散分布的 `rsample` 在基类里直接 `raise NotImplementedError`）。框架层靠这个标志决定要不要在计算图里保留采样节点。

### 1.2 `Categorical`：采样与打分走的是两条不同的张量路径

`Categorical` 的实现只有 171 行，但采样与 logprob 用的是**完全不同的两条路径**。这构成了本文 §5 那一节全部问题的根源，所以先看源码（`torch/distributions/categorical.py`）。

构造函数先做归一化，两种参数走不同分支（`:67-79`）：

```python
if probs is not None:
    self.probs = probs / probs.sum(-1, keepdim=True)
else:
    self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
```

传 `logits` 时用 `logits - logsumexp(logits)` 归一化，这是数值稳定的标准写法；传 `probs` 时直接除以行和，**没有任何 clamp**。

采样（`:143-148`）：

```python
def sample(self, sample_shape=torch.Size()):
    probs_2d = self.probs.reshape(-1, self._num_events)
    samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
    return samples_2d.reshape(self._extended_shape(sample_shape))
```

`self.probs` 是一个 `lazy_property`：如果构造时传的是 `logits`，它由 `logits_to_probs` 现算，也就是 `F.softmax(logits, dim=-1)`（`torch/distributions/utils.py:89-98`）。所以 `Categorical(logits=...).sample()` 的实际路径是 **softmax → multinomial**。

logprob（`:150-156`）：

```python
def log_prob(self, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, self.logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)
```

直接对 `self.logits` 做 gather。如果构造时传的是 `logits`，`self.logits` 是归一化后的值，gather 出来就是精确 logprob；**如果传的是 `probs`，`self.logits` 由 `probs_to_logits` 现算**（`utils.py:127-137`）：

```python
def probs_to_logits(probs: Tensor, is_binary: bool = False) -> Tensor:
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)
```

而 `clamp_probs` 是（`utils.py:101-124`）：

```python
def clamp_probs(probs: Tensor) -> Tensor:
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)
```

**两条路径在这里分岔**：采样走 softmax（不做 clamp），打分走 `log(clamp(p, eps, 1-eps))`。数学上 $$p=0 \Rightarrow \log p = -\infty$$，但 clamp 之后得到的是 $$\log \varepsilon$$。float64 下 $$\varepsilon = 2.22\times 10^{-16}$$，$$\log \varepsilon = -36.04$$；float32 下 $$\varepsilon = 1.19\times 10^{-7}$$，$$\log \varepsilon = -15.94$$。

§5 会给出这条差异的实测后果。

### 1.3 三个可以立刻用的源码定位点

| 位置 | 内容 |
| :--- | :--- |
| `torch/distributions/distribution.py:168-182` | `sample` 转发 `rsample` 并套 `no_grad`；`has_rsample` 默认 `False` |
| `torch/distributions/categorical.py:143-148` | 采样展平 batch 后调 `torch.multinomial`，再 `.T` 转回 |
| `torch/distributions/utils.py:101-137` | `clamp_probs` 与 `probs_to_logits`，logprob 精确性的分界线 |

---

## 2. `torch.multinomial`：一条 fast path 吃掉两种用法

`torch.multinomial` 是整个体系里最值得读的一个函数，因为它同时承担了“有放回”和“无放回”两种语义，而实现上却是**一条 Gumbel 路径 + 两套回退路径**。

### 2.1 守卫条件只有一行

`aten/src/ATen/native/Distributions.cpp` 的 `multinomial_out` 里，真正的分支判断是：

```cpp
  // Fast-path for no replacement or if only one sample is drawn.
  // Reference:
  // https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
  if (!with_replacement || n_sample == 1) {
```

**读法**：`!with_replacement` 为真（无放回）走 fast path；或者 `n_sample == 1` 为真（不管有没有放回）也走 fast path。只有“有放回且要抽多个”才落到 `multinomial_with_replacement_stub`。

这条条件对 LLM 推理是决定性的：**自回归解码每步只抽一个 token**，所以它永远命中 fast path。

### 2.2 fast path 里的算法，注释已经写全

同一段代码的注释逐字给出了推导：

```cpp
    // The algorithm is from gumbel softmax.
    // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
    // Here we can apply exp to the formula which will not affect result of
    // argmax or topk. Then we have
    // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
    // We can also simplify the formula above by
    // s = argmax( p / q ) where q ~ Exp(1)
    Tensor q = at::empty_like(self).exponential_(1, std::move(gen));
    // In theory the probability to generate 0 from exponential distribution is
    // 0. However, on CUDA side there is a protection to avoid 0s, but on CPU
    // side, there is a very low probability to generate 0 from
    // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
    // ignore it here, but there may be some risk to get invalid output on CPU.
    at::div_out(q, self, q);
    if (n_sample == 1) {
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, n_sample);
    }
    return result;
```

三步：`q ~ Exp(1)` → `q ← p / q` → 单样本 `argmax`，多样本 `topk`。

注意注释里那条自我披露的风险：**CPU 上 `exponential_` 有约 $$2^{-53}$$ 的概率生成 0，此时 `p/0 = inf` 会污染 argmax**；CUDA 侧有专门保护，CPU 侧没有。这段注释从 issue 11931 的评论一路沿用至今。

### 2.3 实测：它和 vLLM 手写的采样器逐元素相同

推论很直接：既然 fast path 是 `argmax(p/q)`，而 vLLM 的 `random_sample` 也是 `argmax(p/q)`，两者应该给出**逐元素完全相同**的序列（在同一个随机源下）。

实测（`tools/sampling_bench/l1_sampler_primitives.py`，Probs = `[0.40, 0.30, 0.15, 0.10, 0.05]`，40 万次）：

```text
  A torch.multinomial  max-z= 1.014  TV=0.00088
  B Gumbel-max        max-z= 1.908  TV=0.00136
  C vLLM p/Exp(1)     max-z= 1.014  TV=0.00088

  两两一致性（同一随机种子下逐元素相等的比例）:
    A torch.multinomial vs B Gumbel-max: 0.0360
    A torch.multinomial vs C vLLM p/Exp(1): 1.0000
    B Gumbel-max vs C vLLM p/Exp(1): 0.0360
```

- 三种写法的分布都对（max-z 都远小于 5 的判据阈值；作为对照，把非均匀目标喂给均匀随机流的 max-z 是 **439.28**，判据本身可信）。
- **A 与 C 的一致率是 1.0000**。这不是数值巧合——逐元素相同的唯一解释是走了同一段数学。
- A 与 B 只有 3.6%（约等于 $$1/5$$ 的随机基线）。因为 B 用的是 `-log(-log(U))` 造 Gumbel 后再除以概率，与 `p/q` 形式虽然同分布，随机流不同。

（上面这个块为可读性去掉了每行末尾的样本序列；完整输出见脚本。）

再补一个更硬的证据：**数每行消耗多少个均匀数**。

```text
均匀数预算（每行消耗多少个 U(0,1)，用于反推内部走的是哪条路径）:
multinomial(1x5) 消耗均匀数 = 5
exponential_(1x5) 消耗均匀数 = 5
multinomial(1x5) 反复 3 次后单次增量: 15
对照 torch.rand(1) 消耗均匀数 = 1
```

逆变换采样每行只需要 1 个均匀数（一次二分查找），而实测 `multinomial` 在 5 类分布上消耗 **5** 个——`repeat=3` 时正好是 15，说明消耗量严格线性于调用次数。这一条独立地把结论钉死了。

测量方法本身不依赖任何内部计数器：用同一 seed 先生成参考流 $$r_0, r_1, \dots$$，复位后再执行一次算子，随后取一个均匀数并在参考流里定位它的下标，该下标即消耗量（对照行用 `torch.rand(1)` 验证了方法的分辨率为 1）。封装在 `uniforms_consumed()` 里。

> **为什么这件事重要。** vLLM 的 `random_sample` 注释写着：
> ```
> """Randomly sample from the probabilities.
>
> We use this function instead of torch.multinomial because torch.multinomial
> causes CPU-GPU synchronization.
> """
> ```
> 也就是说，替换的动机是**同步开销**，不是算法不同。理解了这一点，就不会在“换了采样实现”的实验里把等价实现的差异误读成行为变化。

### 2.4 回退路径：有放回且多样本

落到 `multinomial_with_replacement_stub` 之后，CPU 与 CUDA 分道扬镳。

**CPU**（`aten/src/ATen/native/cpu/MultinomialKernel.cpp`）是教科书逆变换：

```cpp
    for (const auto i : c10::irange(n_dist)) {
      scalar_t sum = 0;
      for (const auto j : c10::irange(n_categories)) {
        scalar_t val = self_ptr[i * self_stride_0 + j * self_stride_1];
        TORCH_CHECK(val >= 0, "invalid multinomial distribution (encountering probability entry < 0)");
        TORCH_CHECK(std::isfinite(val), "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
        sum += val;
        cum_dist_ptr[j * cum_dist_stride_0] = sum;
      }
      TORCH_CHECK(sum > 0, "invalid multinomial distribution (sum of probabilities <= 0)");
      ...
      for (const auto j : c10::irange(n_sample)) {
        at::uniform_real_distribution<double> uniform(0, 1);
        double uniform_sample = uniform(gen);
        int64_t left_pointer = 0;
        int64_t right_pointer = n_categories;
        cum_dist_ptr[(n_categories - 1) * cum_dist_stride_0] = 1;
        while (right_pointer - left_pointer > 0) {
          int64_t mid_pointer = std::midpoint(left_pointer, right_pointer);
          scalar_t cum_prob = cum_dist_ptr[mid_pointer * cum_dist_stride_0];
          if (cum_prob < uniform_sample) { left_pointer = mid_pointer + 1; }
          else { right_pointer = mid_pointer; }
        }
        auto sample_idx = left_pointer;
        result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] = sample_idx;
      }
    }
```

两个细节值得记住：

1. **输入不需要归一化**。代码先做累积和再整体除以 `sum`，注释写明“doesn't assume original self row sums to one”。
2. `cum_dist_ptr[(n_categories - 1) * ...] = 1` 是硬赋值。这保证最后一个桶一定接住 $$U$$ 落在 $$(0,1)$$ 开区间外的数值误差。

**CUDA**（`aten/src/ATen/native/cuda/MultinomialKernel.cu`）有两条 kernel：

- `sampleMultinomialOnce`：当 `n_sample == 1` 且共享内存够用时启用，**一个 block 处理一个分布**，块内线程做并行前缀和归约，累加用 `accscalar_t`（即 `at::acc_type<scalar_t, true>`，通常是 double）来压求和误差。若归约得到的 `sum == 0`，直接选索引 0。
- `sampleMultinomialWithReplacement`：**线程按样本切分**，每个线程独立持有 curand 状态，调 `binarySearchForMultinomial` 做 CDF 二分查找。这个函数里有几处针对退化情形的兜底：

```cpp
  int start = 0;
  int end = size;
  // cumdist[size - 1] = 0 => all zero prob dist
  CUDA_KERNEL_ASSERT(cumdist[size - 1] > static_cast<scalar_t>(0));
  ...
  if (start == size) {
    // No probability mass or precision problems; just return the
    // first non-zero element by setting start to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    start = size - 1;
  }
  while(start >= 1 && dist[start] == 0) start--;
  return start;
```

`while(start >= 1 && dist[start] == 0) start--;` 这一行是为了**永不返回零概率索引**：采样值若落在某个零概率桶里（源于浮点误差），就向前回退到最后一个非零概率的索引。

### 2.5 无放回：一个权重竞赛，不是拒绝采样

无放回走 fast path，算法是 `log(U)/w` 之后取 topk——也就是 Efraimidis–Spirakis 的指数竞赛（exponential race）。它等价于“给每个类别算一个独立指数随机数，按 $$w_i / E_i$$ 排序取前 k”。旧版本用的是 `log(U)/p` 形式，2020 年被 PR #47386 改成现在这样，原因写在 PR 描述里：

> torch.Tensor.multinomial may generate invalid output 0 with replacement=False when input distribution is a one-hot vector. The current algorithm is S = topk( log(eps) / p ) where eps ~ U(0, 1). However, when the input is a one-hot vector and the sampled eps is 0 for the one position, then it will get -inf / 1 == -inf. While other positions are also log(eps) / 0 == -inf. **Then the results will be invalid.**

换成 `p/q` 之后，零概率项的 key 变成精确的 `0`（而不是 $$-\infty$$ 或 `NaN`），数值上安全得多。

**正确性验证**：`log(U)/w + topk` 与“每次抽一个、把该项置零、重新归一化”的教科书定义是否同分布？用 6 万次试验统计 2 元子集的联合分布，与精确解逐项对比：

```text
    子集               精确概率  D torch.multin  E 手写 log(U)/w    F 逐次剔除 + 重归一化
    (0, 1)        0.37143         0.37142       0.36888       0.37132
    (0, 2)        0.17059         0.16958       0.16987       0.17000
    (0, 3)        0.11111         0.11282       0.11075       0.11068
    (0, 4)        0.05439         0.05420       0.05412       0.05328
    (1, 2)        0.11723         0.11575       0.11840       0.11898
    (1, 3)        0.07619         0.07703       0.07612       0.07745
    (1, 4)        0.03722         0.03705       0.03818       0.03695
    (2, 3)        0.03431         0.03453       0.03527       0.03425
    (2, 4)        0.01672         0.01668       0.01695       0.01635
    (3, 4)        0.01082         0.01093       0.01147       0.01073

  D torch.multinomial(rep=False)   TV vs 精确 = 0.00288
  E 手写 log(U)/w + topk             TV vs 精确 = 0.00397
  F 逐次剔除 + 重归一化                    TV vs 精确 = 0.00302
```

三者的 TV 距离都在 0.003 量级，与 6 万次试验的统计噪声同阶——**指数竞赛在数学上与逐步剔除完全等价**，但复杂度从 $$O(nm)$$ 降到 $$O(n + m\log m)$$。这也正是 issue #39624 的诉求。

### 2.6 退化输入的行为表（实测）

把 `torch.multinomial` 在各种边界输入上跑一遍，得到一张可以直接当参考的行为表：

| 输入 | 参数 | 实测行为 |
| :--- | :--- | :--- |
| `w = [1, 0, 0]` | 有放回，5 万次 | `bincount = [50000, 0, 0]`，零概率项从不进入样本 |
| `w = [0, 0, 0]` | 有放回 | `RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)` |
| `w = [0, 0, 0]` | 无放回 | 同上 |
| `w = [1, 0, 0]` | `k=2`, 无放回 | 返回 `[[0, 1]]` |
| `w = [1, 0, 0]` | `k=3`, 无放回 | **返回 `[[0, 1, 2]]`** |
| `w = [1, 1, 1]` | `k=4`, 无放回 | `RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement` |
| 含负数 / 含 `inf` | 任一 | `RuntimeError: probability tensor contains either 'inf', 'nan' or element < 0` |

第五行值得单独说。**官方文档写的是**：

> When drawn without replacement, `num_samples` must be lower than number of non-zero elements in input.

**但实现检查的是类别总数，不是非零元素数**：

```cpp
  TORCH_CHECK(with_replacement || (n_sample <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
```

所以 `w = [1, 0, 0]` 配 `k = 3` 不但不报错，还会把两个零概率索引一起选出来。原因是 fast path 换了公式之后，零概率项的 key 是 `0` 而不是 $$-\infty$$，`topk` 无法把它们区分开——当 `k` 超过非零项个数时，零概率项的相对顺序完全由 `topk` 的 tie-break 决定。

**实践含义**：如果业务上依赖“无放回采样出来的索引一定落在有效集合内”，不能靠 `multinomial` 自己保证，必须显式把 `k` 限制在非零项个数以内。在 VLA 里对应的是“被 mask 掉的 action bin 会不会被采出来”——用无放回路径时答案是**会**。

---

## 3. 从 logits 到 token：LLM 采样栈

`torch.multinomial` 之后，token 还是从 logits 来的。这一段在 LLM 里的分工很清晰：**塑形归框架，抽取归原语**。

### 3.1 transformers 的装配顺序

`generation/utils.py:1303-1358` 决定了所有 warper 的施加顺序。这个顺序不是随意的，注释里明确写了 `min_p` 为什么排在那里：

```python
        if generation_config.do_sample:
            min_tokens_to_keep = ...
            # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
            # all samplers can be found in `generation_utils_samplers.py`
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                processors.append(TemperatureLogitsWarper(generation_config.temperature))
            if generation_config.top_h is not None:
                processors.append(TopHLogitsWarper(top_h=generation_config.top_h))
            if generation_config.top_k is not None and generation_config.top_k != 0:
                processors.append(TopKLogitsWarper(top_k=..., min_tokens_to_keep=min_tokens_to_keep))
            if generation_config.top_p is not None and generation_config.top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p=..., min_tokens_to_keep=min_tokens_to_keep))
            if generation_config.min_p is not None:
                # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
                processors.append(MinPLogitsWarper(min_p=..., min_tokens_to_keep=min_tokens_to_keep))
            if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
                processors.append(TypicalLogitsWarper(mass=..., min_tokens_to_keep=min_tokens_to_keep))
            if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
                processors.append(EpsilonLogitsWarper(...))
            if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
                processors.append(EtaLogitsWarper(...))

        # Watermarking should be after all logits processing is finished (see #34630)
        if generation_config.watermarking_config is not None:
            processors.append(...)

        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
```

顺序是：**温度 → top-h → top-k → top-p → min-p → typical → epsilon → eta → 水印 → logit 归一化**。

三个要点：

1. **温度排在最前**。先缩放 logits 再截断，与先截断再缩放不是一回事——后者会让 top-k 的成员集合与温度无关，前者会随温度变化。
2. **min-p 排在 top-p 之后**，注释给了 llama.cpp 的讨论链接作为依据。
3. **水印排在所有塑形之后**。水印做的是“给绿名单 token 的 logits 加一个常数 bias”（`logits_process.py:2385-2460`，默认 `bias=2.0`、`greenlist_ratio=0.25`）。因为它会翻转 argmax，所以不能被放在任何“只影响随机采样”的算子之前，否则 greedy 路径会绕过水印。

顺带说明一件事：**水印不需要触碰采样器**。只要采样器读的是 logits，任何 logits 层面的加性偏移都会等比改变概率。这也是采样栈的一个通用性质——塑形与抽取之间的接口只有一个张量。

### 3.2 `_sample` 的核心只有三行

塑形做完，抽取是（`generation/utils.py:2892-2897`）：

```python
            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = next_token_scores.argmax(dim=1, keepdim=True)
```

注意那条 TODO：`torch.multinomial` 会打断 CUDA graph 捕获。这与 §2.3 的分析一致——`multinomial` 内部要生成随机数并做归约，属于 “incompatible ops”。

### 3.3 vLLM 的九步流水线

vLLM 把同一件事拆得更细，`vllm/v1/sample/sampler.py` 的类 docstring 就是一份规格说明：

```text
    1. If logprobs are requested: ...
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant, i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. ...
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the randomly sampled tokens ...
```

关键的设计决策藏在步骤 5 和 7c 的分类里：**算子被分成 “argmax-invariant” 和 “non-argmax-invariant” 两类**。前者（top-k / top-p / min-p）不改变最大值的位置，只在随机采样路径里施加；后者（min tokens、logit bias、各类惩罚）会影响 greedy，必须在 greedy 之前施加。

这个分类带来一个工程上的便利：一个 batch 里既有 greedy 请求又有采样请求时，可以先算一次 argmax，只在需要的地方做随机采样，最后用 `torch.where` 合并（`sampler.py:194-203`）：

```python
        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs
```

`out=greedy_sampled` 是个细节：把结果写回 `greedy_sampled` 的存储，避免额外分配。而 `apply_temperature` 里也有对应的除零保护（`sampler.py:131-141`）：

```python
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))
```

`_SAMPLING_EPS = 1e-5`，温度为 0 的请求在温度缩放阶段被替换为 1.0，最后靠 `torch.where` 换成 greedy 结果。

### 3.4 三种塑形算子的实现分歧

同一个 top-k / top-p / min-p，在 transformers 与 vLLM 里的写法不同。逐条对照实测。

**top-p：升序累积 vs 降序累积。**

transformers 的写法是升序排列后掩掉“从最小端累积到 $$1-p$$”的部分（`logits_process.py:519-532`）：

```python
    def __call__(self, input_ids, scores):
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
```

vLLM 的 PyTorch 版写法结构相同但用原地操作（`vllm/v1/sample/ops/topk_topp_sampler.py:289-299`）：

```python
    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
```

教科书定义则是“降序排列，保留累积概率达到 $$p$$ 的最小前缀”。把三者放在 $$\text{top\_p} \in \{0.5, 0.9, 0.95\}$$、batch 8 × vocab 40 上逐元素比 mask：

```text
    top_p = 0.9
      transformers  vs 教科书                 mask 完全相同=True  不同元素数=0
      transformers  vs vLLM                mask 完全相同=True  不同元素数=0
      transformers 保留数 = [10, 9, 9, 16, 15, 10, 14, 10]
      教科书      保留数 = [10, 9, 9, 16, 15, 10, 14, 10]
      vLLM        保留数 = [10, 9, 9, 16, 15, 10, 14, 10]

    top_p = 0.95
      ... 均完全相同 ...

    top_p = 0.5
      ... 均完全相同 ...
```

三种写法在所有测试点上都给出完全相同的 mask，包括并列概率的边界样本。原因是代数上恒等：设降序概率 $$q_1 \ge q_2 \ge \cdots \ge q_V$$，升序累积到第 $$j$$ 个即最小的 $$j$$ 个之和 $$\sum_{i>V-j} q_i = 1 - \sum_{i \le V-j} q_i$$，于是 `升序累积 <= 1-p` 与 `降序前缀和 < p` 是同一条件的两种写法，`<=` 与 `<` 的边界由“至少保留一个”和 `min_tokens_to_keep` 一起兜住。

**min-p：概率域 vs log 域，这次真的省了一次 softmax。**

transformers 在概率域（`logits_process.py:751-767`）：

```python
    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores, dim=-1)
        top_probs = probs.amax(dim=-1, keepdim=True)
        scaled_min_p = self.min_p * top_probs
        tokens_to_remove = probs < scaled_min_p
        ...
```

vLLM 的 triton kernel 在 log 域（`vllm/v1/worker/gpu/sample/min_p.py`）：

```python
    max_val = max_val.to(tl.float32)
    threshold = max_val + tl.log(min_p)
    for i in range(0, vocab_size, BLOCK_SIZE):
        ...
        logits = tl.where(logits < threshold, float("-inf"), logits)
```

两者等价，因为

$$
p_i < m \cdot p_{\max} \iff \log p_i < \log m + \log p_{\max} \iff \ell_i - \Lambda < \log m + \ell_{\max} - \Lambda \iff \ell_i < \ell_{\max} + \log m
$$

其中 $$\Lambda = \operatorname{logsumexp}(\ell)$$ 被两边约掉。实测在 $$\text{min\_p} \in \{0.05, 0.1, 0.3\}$$、float64 与 float32 下 mask 均完全一致：

```text
    min_p = 0.05
      概率域 vs log 域 (fp64)                  mask 完全相同=True  不同元素数=0
      概率域 vs log 域 (fp32)                  mask 完全相同=True  不同元素数=0
    min_p = 0.1
      ... 均完全相同 ...
    min_p = 0.3
      ... 均完全相同 ...
```

**但省下的那次 softmax 很值钱**，因为 min-p 只需要一个 `max`，而 top-p 需要一次完整排序。在 batch 32 × vocab 152064 上的实测（本机 CPU，20 轮中位数）：

```text
    batch=32, vocab=152064, 取 20 轮中位数（代表性一次运行）
      torch.sort(全排序)              ~92 ms
      torch.topk(k=50)                 ~2.3 ms   ( ~40x 快于全排序)
      torch.argsort(降序)             ~90 ms
      双 argsort 求秩                 ~155 ms    ( ~1.7x 全排序)
      amax(仅求阈值, min-p 用)          ~0.24 ms   ( ~380x 快于全排序)
```

绝对耗时随机器、线程数、BLAS 后端浮动（同机重跑也会有 5% 上下的抖动），**可以外推的是比值与量级差**，不是某一位小数。

这里有一个需要纠正的常见直觉：**“用秩代替排序”并不便宜**。在一张已有的排序结果上，再求一次秩需要第二次 `argsort`，两次下来反而比一次全排序还慢约 1.7 倍。真正便宜的做法是根本不需要顺序信息——min-p 只要一个 `amax`，这是它相对 top-p 有近两个数量级差距的来源。（vLLM 在批量足够大时走 triton kernel 而不是这套 PyTorch 实现，正是为了绕开 `sort`。）

**top-k：并列值下的真实行为。**

transformers 用“第 k 大的值”做广播比较，vLLM 的 top-k 分支同理：

```python
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
```

严格小于意味着：**所有等于第 k 大值的 token 都会被保留**。这在 logits 有并列时会保留超过 k 个。实测：

```text
    并列值下的行为（logits 含重复最大值）
      k=1: broadcast 保留 3 个, 秩 保留 1 个  -> 不一致
      k=2: broadcast 保留 3 个, 秩 保留 2 个  -> 不一致
```

输入是 `[3.0, 3.0, 3.0, 1.0, 0.0]`。`k=1` 时第 1 大值是 `3.0`，所有 `3.0` 都满足“不小于 3.0”，于是**留下 3 个**。均匀 logits（模型完全不确定）是最极端的例子：`top-k=2` 在 `[0,0,0,0]` 上保留全部 4 个。

这不是 bug，是“阈值式 top-k”的固有语义；但它意味着 **top-k 不是一个精确的基数约束**。需要严格基数时得用秩实现。顺带一提，实测在无并列的随机 logits 上两种实现完全一致（k=5 与 k=20 时 mask 差异均为 0），所以这个问题只在分布出现平坦区时才暴露。

### 3.5 三种塑形算子对最终 token 的影响

同一批 4096 条 logits（vocab 200），施加不同塑形后用同一个种子采样，与“不塑形”的结果比对：

```text
    batch=4096, vocab=200, 与「原始分布」采样结果的一致率
      top-k=20           0.7439
      top-p=0.9 (tf)     0.9045
      top-p=0.9 (教科书)  0.9045
      min-p=0.1 (prob)   0.6543
      min-p=0.1 (log)    0.6543

      top-p 两种写法之间的一致率: 1.0000
      min-p 两种写法之间的一致率: 1.0000
```

两种实现之间的口径是 100% 一致，而与原始分布的一致率落在 65%–90%。**换句话说，实现的差异在这类输入上不影响输出，而“要不要塑形、塑成什么样”才是决定性的。**

---

## 4. 投机解码：采样正确性变成一道数学题

投机解码把采样从“调一个函数”变成了“要证明一个分布不变式”。vLLM 的 `RejectionSampler` docstring 把术语定得很清楚（`vllm/v1/sample/rejection_sampler.py:30-51`）：

```text
class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """
```

核心是拒绝采样：对 draft 提出的 token $$x$$，以概率 $$\min(1, p(x)/q(x))$$ 接受；拒绝时从“残差分布”

$$
p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}
$$

重新采样。这条不变式保证了最终输出严格服从 $$p$$。

三个实践要点：

1. **bonus token 是单独采的**，而且注释明确说明“我们不在这里采，而是把 bonus token 传进来”。原因是为了让 bonus token 能用 top-p / top-k——**投机解码路径本身不支持这些塑形**。这是一个接口设计上的分裂：同一条序列里，draft 覆盖的 token 走拒绝采样（无塑形），最后一个 token 走普通采样（有塑形）。
2. **`bonus_logits` 和 `target_logits` 是索引出来的新张量**。源码注释特意说明这一点，因为 `logits[indices]` 在 PyTorch 里产生独立存储，所以对它们做原地操作不会污染原始 `logits`——这是后续 `apply_logits_processors` 敢用原地修改的前提。
3. 源码里有 `MAX_SPEC_LEN = 128` 的上限，以及 `assert metadata.max_spec_len <= MAX_SPEC_LEN`。

如果只看“采样的数学”，这一节的价值在于：**它是唯一一个把采样正确性写成显式不变式、并且要求实现逐字对齐论文的地方**。其他路径（multinomial、top-p）的正确性是算法自带的，不需要额外证明。

---

## 5. logprob 的五条路径：同一个量，不同的数

采样出 token 只是开始。训练侧要的是这个 token 的 logprob。同一个数学量，PyTorch 生态里有至少五条计算路径：

| 路径 | 实现 | 关键步骤 |
| :--- | :--- | :--- |
| P1 | `Categorical(probs=softmax(logits)).log_prob(x)` | `clamp_probs` → `log` → gather |
| P2 | `Categorical(logits=logits).log_prob(x)` | `logits - logsumexp` → gather |
| P3 | `F.log_softmax(logits)[x]` | 全量物化 log_softmax 再取 |
| P4 | `logits[x] - logsumexp(logits)` | 只算需要的那个（TRL 的做法） |
| P5 | `log(F.softmax(logits)[x])` | 最朴素，数值最不稳 |

### 5.1 常规输入下五条路径一致到浮点极限

先用解析解做自校验。`logits = [0.0, 0.0]` 时两个 token 的 logprob 都应是 $$-\log 2$$；`logits = [3.0, 0.0]` 时 token 0 的 logprob 应是 $$3 - \log(1 + e^3) = -0.048587351574\ldots$$

```text
    logits=[3.0, 0] 时 token0 的解析解 = -0.048587351574
    P1 Categorical(probs)      -0.048587351574  差 8.33e-17
    P2 Categorical(logits)     -0.048587351574  差 0.00e+00
    P3 log_softmax + gather    -0.048587351574  差 9.02e-17
    P4 selective (TRL)         -0.048587351574  差 0.00e+00
    P5 log(softmax)            -0.048587351574  差 1.46e-16
```

在常规 logits、含零概率、含 $$-\infty$$ mask 的场景下，五条路径的差都在 $$10^{-16}$$ 量级。**分歧只出现在概率极小或恰好为零的目标上。**

### 5.2 分岔点：`clamp_probs` 的 $$\varepsilon$$

构造两个场景：

- **S5**：`logits = [1.0, -inf, 0.5]`，目标索引 1（被 mask 掉的项）
- **S6**：`logits = [0.0, -60.0, -120.0]`，目标索引 2（概率约 $$e^{-120}$$ 的项）

```text
  S5 目标为被 mask 项
    logits = [1.0, -inf, 0.5]
    目标索引 = 1
      P1 Categorical(probs)      -36.043653389117  <- 基准
      P2 Categorical(logits)     -inf  不一致(无穷)
      P3 log_softmax + gather    -inf  不一致(无穷)
      P4 selective (TRL)         -inf  不一致(无穷)
      P5 log(softmax)            -inf  不一致(无穷)

  S6 概率极小的项
    logits = [0.0, -60.0, -120.0]
    目标索引 = 2
      P1 Categorical(probs)      -36.043653389117   <- 基准
      P2 Categorical(logits)     -120.000000000000  差 -8.396e+01
      P3 log_softmax + gather    -120.000000000000  差 -8.396e+01
      P4 selective (TRL)         -120.000000000000  差 -8.396e+01
      P5 log(softmax)            -120.000000000000  差 -8.396e+01
```

S6 的差距是 **83.96 nat**，而 S5 是“$$-36.04$$ 还是 $$-\infty$$”的定性区别。

原因回到 §1.2：`Categorical(probs=...)` 的 `log_prob` 走的是 `log(clamp(p, eps, 1-eps))`。float64 的 `eps` 是 `2.220446049250313e-16`，$$\log(2.22\times 10^{-16}) = -36.0436$$。**任何真实概率低于 $$\varepsilon$$ 的 token，其 logprob 都会被抬到这个地板值。**

验证这个解释：

```text
    clamp_probs([0, 0.5, 1]) = [2.220446049250313e-16, 0.5, 0.9999999999999998]  eps=2.220e-16
```

顺带注意 clamp 上界的怪现象：`1.0` 被夹到 `0.9999999999999998`，因为 `1 - eps` 在 float64 里无法精确表示 1 以下的最近值——这也是 `probs_to_logits` 对“概率恰好为 1”的输入会给出一个略小于 0 的 logprob 的原因。

### 5.3 为什么服务端一律 upcast 到 fp32 再算

vLLM 的 logprob 计算只有一行（`vllm/v1/sample/sampler.py:205-207`）：

```python
    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)
```

`dtype=torch.float32` 是显式的——即使 logits 是 bf16 或 fp16，这次 `log_softmax` 也在 fp32 里做。同一文件里还有一句注释解释了另一个容易踩的坑（`:75-78`）：

```python
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
```

**top-k logprobs 用的是未经塑形的原始 logits，而采样用的是塑形后的 logits。** 这意味着 `logprobs` 字段里的数值和“采样时实际用的分布”是两个不同的分布。做 SFT 蒸馏或 RL 打分时如果直接拿返回的 logprobs 当采样分布，就会引入一个系统性的口径错配。

排名是用计数而不是排序算的（`vllm/v1/sample/sampler.py:242`）：

```python
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)
```

`batched_count_greater_than` 就是 `(x >= values).sum(-1)`，被 `torch.compile` 包了一层以避免中间张量（`vllm/v1/sample/ops/logprobs.py`）。对一个只需要排名的需求来说，这比排序便宜得多——与 §3.4 里“秩需要两次 argsort”的结论并不矛盾，因为这里要的是“有多少个比它大”这个计数，不是一个完整的秩映射。

### 5.4 采样得到的 token 用哪条路径回算都一样吗

把后半个词表 mask 成 $$-\infty$$，采样 2 万条，再用五条路径回算：

```text
    batch=20000, vocab=1000（后半被 mask 成 -inf）
    采样结果全部落在前 500 个 token 内: True

    P1 Categorical(probs)      -inf/nan 个数=0      mean=-4.362677 min=-11.900532
    P2 Categorical(logits)     -inf/nan 个数=0      mean=-4.362677 min=-11.900532
    P3 log_softmax + gather    -inf/nan 个数=0      mean=-4.362677 min=-11.900532
    P4 selective (TRL)         -inf/nan 个数=0      mean=-4.362677 min=-11.900532
    P5 log(softmax)            -inf/nan 个数=0      mean=-4.362677 min=-11.900532
```

五条路径逐字节相同。**因为采样只会命中概率足够大的 token，那个 $$\varepsilon$$ 地板根本没有机会生效。** 换句话说：§5.2 的分歧不是“训练时必然会遇到”的问题，而是“一旦遇到就是静默错值”的问题——它不会报错，只会给出一个偏乐观的 logprob。

**实践结论**：传 `logits` 而不是 `probs`。如果手上只有概率，也宁可 `logits = probs.log()` 之后走 `logits - logsumexp`，不要构造 `Categorical(probs=...)`。

---

## 6. RL：采样器与打分器的口径必须对齐

RL 训练把 §5 的数值问题放大成了一个结构问题：**rollout 阶段的采样器和训练阶段的打分器通常是两套代码、两个精度、甚至两个进程。**

### 6.1 TRL 的 `selective_log_softmax`

TRL 没有用 `torch.distributions`，而是实现了一个专门的算子（`trl/trainer/utils.py:495-539`）：

```python
def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
        logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    """
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    ...
```

三点值得记：

1. **只在 fp32/fp64 上用 `logsumexp` 分解**。bf16 下 `logsumexp` 不稳定，回退到逐行 `F.log_softmax` + gather。注释把这条权衡写在代码里了。
2. **逐行循环是为了削峰值内存**（`# loop to reduce peak mem consumption`）。用 `x_i - logsumexp(x)` 而不是物化整个 log_softmax，峰值显存从 $$O(BV)$$ 降到 $$O(B)$$。
3. 它和 `Categorical(logits=...).log_prob` 在数学上是同一件事，但没有任何分布对象、没有 `clamp_probs`、没有 lazy property 的 `torch.enable_grad()` 包裹。

### 6.2 ratio 恒等式的三条误差源

PPO/GRPO 的核心量是 $$r = \exp(\log p_{\text{new}} - \log p_{\text{old}})$$。如果策略权重没变，这个比值应该恒等于 1。实测它能被推离 1 多远：

```text
  batch=8192, vocab=32；采样分布 = softmax(logits)（fp64）

  (a) 同精度、改走 log_softmax+gather 全量物化再取
      max|ratio-1| = 1.776e-15   mean|ratio-1| = 2.717e-16   inf/nan = 0
  (b) 打分侧降到 fp32（推理引擎常见）
      max|ratio-1| = 9.703e-07   mean|ratio-1| = 1.675e-07   inf/nan = 0
  (c) 打分侧降到 bf16（训练 forward 常见）
      max|ratio-1| = 3.519e-02   mean|ratio-1| = 4.582e-03   inf/nan = 0
```

- **(a)** 在浮点极限上：$$1.8\times 10^{-15}$$，可以忽略。
- **(b)** 只把打分侧降到 fp32，最大偏差 $$9.7\times 10^{-7}$$。
- **(c)** 只把打分侧降到 bf16，**最大偏差 $$3.5\times 10^{-2}$$，均值 $$4.6\times 10^{-3}$$**。

实验设计上，这三行的采样侧始终是 fp64，只有打分侧降精度——这正好对应“rollout 由高精度推理引擎产生、训练用 bf16 forward 重算”的真实配置。**策略权重一个字没改，ratio 已经有 0.5% 量级的抖动。** 这类偏差会被 PPO 的裁剪项和 GRPO 的组内标准化当成真实策略漂移吸收掉，成为梯度噪声的固定来源。

### 6.3 截断口径不一致会让 ratio 结构性偏移

更严重的是分布口径不一致。让 proposal 用 top-k=4 截断，target 用完整分布：

```text
  (d) 截断口径不一致：proposal 用 top-k=4 截断，target 用完整分布
      错误做法（proposal 也用 target 打分）
      max|ratio-1| = 0.000e+00   ...   ratio 范围 [1.000000, 1.000000]   inf/nan = 0
      正确做法（proposal 用截断后的分布打分）
      max|ratio-1| = 5.846e-01   mean|ratio-1| = 1.415e-01
      ratio 范围 [0.415402, 0.999809]   inf/nan = 0
     注意：截断后 proposal 的概率质量更集中，w = p_target/p_proposal <= 1 的比例 = 1.0000
```

三件事：

1. **“错误做法”那一行**（用 target 给自己打分）恒为 1，是个空验证——它演示的是“如果两边口径一致，ratio 就是 1”这个恒等式。
2. **正确做法下 ratio 落在 $$[0.4154, 0.9998]$$**，均值偏 14%。这不是噪声，是系统性偏差：截断把概率质量重新分配给了保留的 token，所以 $$p_{\text{proposal}} \ge p_{\text{target}}$$ 恒成立，权重 $$\le 1$$ 的比例是 **100%**。如果训练侧忽略这一点，等价于给所有 token 的梯度乘了一个恒小于 1 的折扣。
3. 现实中的事故形态是反过来的：**proposal 截断、target 不截断，且代码把 target 的 logprob 当作 proposal 的 logprob**。此时被截断掉的 token 若出现在样本里（来自更早的步骤、或来自不同的塑形配置），`log p_proposal = -inf`，ratio 直接变成 `inf` 或 `nan`。

**结论**：`logprobs` 必须和采样时的塑形配置绑定记录。vLLM 的 `logprobs_mode` 提供了 `raw_logprobs` / `raw_logits` / `processed_logits` / `processed_logprobs` 四档，正是为了把这个口径显式化——这也是 §5.3 那条 “V1 与 V0 行为不同” 注释的工程背景。

---

## 7. 重参数化与 Gumbel-Softmax：让梯度穿过采样

前面六节都在处理离散采样。只要模型需要端到端可微（变分推断、VAE、连续松弛），就必须换成重参数化路径。

### 7.1 `sample` 无梯度，`rsample` 有梯度

```text
    has_rsample = True

    sample()  输出 requires_grad = False  (基类走 with torch.no_grad())
    rsample() 输出 requires_grad = True  (loc + eps*scale 保留计算图)

    d(sum rsample)/d(loc)   = [3.0, 3.0]   (解析解 = 采样个数 3.0)
    d(sum rsample)/d(scale) = [-2.3417163106695873, 1.4317019295184301]  (解析解 = sum(eps))
```

`loc = [0.5, -1.0]`、`scale = [1.5, 0.5]`，抽 3 个样本：对 `loc` 的梯度恰好是样本数 3（每个样本对 `loc` 的偏导都是 1），对 `scale` 的梯度是 `eps` 在该 batch 维上的和。

### 7.2 `gradcheck` 的正确用法：把 `eps` 固定下来

一个容易掉的坑：**直接对 `rsample` 做 `torch.autograd.gradcheck` 会失败**。

```text
    gradcheck：直接对 rsample 做会失败（每次调用都重抽 eps，数值 Jacobian 无意义）
      gradcheck(rsample) 直接调用 = 失败（GradcheckError）
      原因：rsample 不是 loc/scale 的确定性函数，扰动输入时不变量 eps 也变了。

    正确做法：固定 eps 后对展开式做 gradcheck
      gradcheck(loc + eps*scale) = True
```

这才是重参数化的实质：**随机性被完整地推到了 $$\varepsilon$$ 上，`loc` / `scale` 走的是普通确定性微分**。用同一个种子复现 $$\varepsilon$$ 之后可以逐元素验证这一点：

```text
    把 rsample 展开：rsample 到底是不是 loc + eps*scale
      重新播种后手动复现 eps，逐元素相等 = True
      eps =
tensor([[-2.6359,  0.0573],
        [ 1.3114, -1.1091],
        [ 0.2368,  0.2579]], dtype=torch.float64)
```

对照解析解：`sum(eps[:,0]) = -2.6359 + 1.3114 + 0.2368 = -1.0877`……而实测梯度是 `-2.3417`。差异来自 `loc`/`scale` 是共享的标量张量，梯度是沿着整个 batch 广播维求和的：`eps` 全部 6 个元素按 batch 维分组累加，得到 `[-2.3417, 1.4317]`。这也说明**一个标量 `scale` 参数承载了 batch 里所有样本的噪声梯度**，这是重参数化训练的方差来源之一。

### 7.3 造 Gumbel 噪声的两条路径

离散松驰的标准做法是 Gumbel-Softmax。PyTorch 里有**两处独立实现**，用不同的方式造 Gumbel 噪声：

`ExpRelaxedCategorical.rsample`（`torch/distributions/relaxed_categorical.py:87-94`）：

```python
    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(
            torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        )
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels) / self.temperature
        return scores - scores.logsumexp(dim=-1, keepdim=True)
```

走 `g = -log(-log(U))`，`U ~ U(0,1)`，并用 `clamp_probs` 把 $$U$$ 夹到开区间。

`F.gumbel_softmax`（`torch/nn/functional.py:2216-2222`）：

```python
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
```

走 `-log(E)`，`E ~ Exp(1)`。顺带注意 `memory_format=torch.legacy_contiguous_format`——因为 `exponential_()` 是原地操作，必须保证布局连续。

**两者分布相同，但逐元素不同。** 实测 10 万个数：

```text
    路径 A  -log(-log(U)), U~U(0,1)   : 均值=0.584665 方差=1.669353
    路径 B  -log(E),       E~Exp(1)   : 均值=0.574374 方差=1.671677
    Gumbel(0,1) 理论值                : 均值=0.577216 方差=1.644934

    A 与 B 的差: 最大 = 1.640e+01, 均值 = 1.945e+00
    A 与 B 逐元素完全相同 = False

    exponential_ 到底做了什么变换（同一 seed 下与候选变换比对）:
      E == -log(U)     : False
      E == -log(1-U)   : True
```

最后两行是决定性的：**PyTorch 的 `exponential_` 在 CPU 上就是 `-log(1-U)`**，于是路径 B 实际是 `-log(-log(1-U))`，与路径 A 的 `-log(-log(U))` 是同一个 $$U$$ 的两种变换。分布一致、样本不一致。“同一分布的不同采样器”这句话在这里有了可复现的实例。

### 7.4 温度的语义，以及 straight-through 的梯度

温度 $$\tau$$ 只影响松驰程度，不影响离散采样的正确性。`logits = [2.0, 0.5, -1.0]`，2 万次：

```text
         tau         max 分量均值       one-hot 比例          argmax 分布 (0/1/2)
         2.0         0.605455           0.0004   [0.7885, 0.1724, 0.0391]
         1.0         0.760688           0.0302   [0.7875, 0.1762, 0.0364]
         0.5         0.877110           0.2777   [0.7863, 0.1747, 0.0389]
         0.1         0.975603           0.8365   [0.7859, 0.1736, 0.0406]
        0.01         0.997476           0.9838   [0.7943, 0.1672, 0.0385]

    真实 softmax 概率 = [0.7856, 0.1753, 0.0391]
```

- `max 分量均值`从 0.605 爬到 0.997，`one-hot 比例`从 0.04% 爬到 98.4%——这是 $$\tau$$ 在控制松驰。
- **`argmax 分布`那一列在五个 $$\tau$$ 上几乎不变，且都贴合真实 softmax 概率**。这是 Gumbel-max 定理的直接实证：离散采样只看 `argmax`，而 `argmax(logits + g)` 与 $$\tau$$ 无关。

`hard=True` 的 straight-through 装修是：

```python
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, ...).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
```

`y_hard - y_soft.detach() + y_soft` 这个式子同时做到两件事：前向值恰好是 `y_hard`（因为加上了又减掉了 `y_soft`），反向梯度等于 `y_soft` 的梯度（`y_soft.detach()` 那一项被剥掉了）。

验证这个性质时有两个陷阱，实测把它们都踩出来了：

```text
    straight-through 的梯度检查（固定同一份噪声 + 非退化目标函数）：
      loss = <y, w>, w = [0.7, -1.3, 2.1]
      重播种后 soft 分支可复现 = True
      y_hard 取值          = [0.0, 0.0, 1.0]  (恰好 one-hot)
      d loss/dlogit (soft) = [0.0485383748, -0.5833761304, 0.5348377556]
      d loss/dlogit (hard) = [0.0485383748, -0.5833761304, 0.5348377556]
      两者相同 = True   (straight-through: 前向 one-hot, 反向等于同一份噪声下 soft 的梯度)
      两个坑: ① 不固定噪声则两次调用抽的是不同 y_soft，梯度当然不同；
              ② 拿 y_soft.sum() 当 loss 会得到全 0（softmax 输出和恒为 1）。
```

**陷阱一**：`F.gumbel_softmax` 每次都重新抽噪声，所以要比较 `hard=True` 与 `hard=False` 的梯度，必须先 `torch.manual_seed` 对齐（实测对齐后两者逐位相同）。
**陷阱二**：用 `y_soft.sum()` 当损失函数会得到全零梯度——`softmax` 的输出和恒为 1，对 logits 的梯度天然为 0。必须用加权目标 `<y, w>` 才能观察到非平凡的梯度。

### 7.5 Gumbel-max 与 `multinomial` 的关系

最后把 §2 的结论在更一般的 K 上复核一遍（K=6，20 万次）：

```text
     idx         理论   multinomial    Gumbel-max      p/Exp(1)
       0    0.38388       0.38410       0.38320       0.38410
       1    0.10462       0.10394       0.10475       0.10394
       2    0.05195       0.05158       0.05157       0.05158
       3    0.19063       0.18987       0.19121       0.18987
       4    0.01159       0.01195       0.01218       0.01195
       5    0.25732       0.25855       0.25710       0.25855

    Gumbel-max 与 multinomial 逐元素一致率 = 0.0331
    p/Exp(1)  与 multinomial 逐元素一致率 = 1.0000
```

四列频率都贴合理论值；`p/Exp(1)` 与 `multinomial` 逐元素完全一致（100%），而手工用 `-log(-log(U))` 造的 Gumbel 只有 3.3%（随机基线 $$1/6 \approx 16.7\%$$ 附近，说明这确实是两条独立的随机流）。**这一条跨规模复现了 §2.3 的结论。**

---

## 8. VLA 的采样：从离散 token 到连续动作

到了具身智能，采样的形态变了两级。VLA 的动作输出有三类实现，对应三种完全不同的“采样”：

| 形态 | 代表模型 | “采样”是什么 | 随机性来源 |
| :--- | :--- | :--- | :--- |
| **离散 action token** | RT-2、OpenVLA | 一次自回归解码 | 每步的 token 采样 |
| **扩散 / 流匹配** | Diffusion Policy、π0、GR00T | 一次 ODE / SDE 积分 | 初始噪声 |
| **CVAE latent** | ACT、Diffusion Policy | 一次重参数化 | latent 采样 |

前两类在 §2–§7 已经铺垫完毕：离散 action token 就是把动作分箱之后接上整套 LLM 采样栈。OpenVLA 的做法是**把连续动作拆成 256 个均匀 bin，每维动作一个 token**（7 维动作对应 `dx, dy, dz, r, p, y, grip` 七个 token），然后走标准自回归。

真正需要新框架的是后两类。

### 8.1 扩散采样：为什么不能直接跳步

DDPM 的祖先采样（ancestral sampling）递推式是：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\right) + \sigma_t z,\qquad \sigma_t^2 = \frac{\beta_t (1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
$$

注意 $$\sigma_t$$ 的表达式中出现的是 $$\bar\alpha_{t-1}$$——**它显式依赖“上一步是谁”**。如果把 500 步的 schedule 抽成 20 步子序列，公式里的 $$t-1$$ 不再对应真正的相邻步，方差就错了。

在双模态目标 $$p_{\text{data}} = 0.35\,\mathcal N(-2, 0.5^2) + 0.65\,\mathcal N(2, 0.4^2)$$ 上（用解析 score 代替神经网络，把算法误差和拟合误差分开），实测：

```text
    DDPM 全 500 步               均值= +0.5908 标准差= 1.9567  q5=-2.523  q50=+1.703  q95=+2.561
                                 与目标直方图 TV = 0.0156
    DDPM 抽 50 步                均值= +0.0845 标准差= 1.2196  q5=-1.858  q50=+0.242  q95=+1.922
                                 与目标直方图 TV = 0.6528
    DDPM 抽 20 步                均值= +0.0331 标准差= 1.0958  q5=-1.740  q50=+0.050  q95=+1.773
                                 与目标直方图 TV = 0.7164
    DDPM 抽 10 步                均值= +0.0182 标准差= 1.0492  q5=-1.692  q50=+0.023  q95=+1.719
                                 与目标直方图 TV = 0.7364
```

**全 500 步的 TV 是 0.0156，抽 10 步变成 0.7364——分布基本被毁掉。** 标准差从 1.96 掉到 1.05，中位数从 1.70 掉到 0.02，双模态信息丢失。

这不是“步数少所以精度低”的正常退化，而是**方差公式被错用**导致的系统性塌缩。DDPM 想加速就不能抽子序列，必须改用 DDIM 那套重排后的确定性公式。

### 8.2 DDIM：η=0 是确定性的

DDIM 的更新式把 $$\bar\alpha_{t}$$ 与 $$\bar\alpha_{t-1}$$ 解耦：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\varepsilon_\theta + \sigma_t z,\qquad
\sigma_t(\eta) = \eta\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}
$$

$$\eta = 1$$ 时退化为 DDPM 的方差。实测：

```text
    同一初始噪声，eta=0 跑两次：逐元素完全相同 = True
    同一初始噪声，eta=1 跑两次：逐元素完全相同 = False
    eta=0 与 eta=1 结果的最大差异 = 5.9818

    DDIM eta=0.0 steps=10      均值= +0.5592 标准差= 1.8696  TV = 0.0937
    DDIM eta=0.0 steps=20      均值= +0.5403 标准差= 1.9236  TV = 0.0477
    DDIM eta=0.0 steps=50      均值= +0.5300 标准差= 1.9548  TV = 0.0249

    DDIM eta=0.5 steps=10      均值= +0.5796 标准差= 1.8593  TV = 0.1052
    DDIM eta=0.5 steps=20      均值= +0.5616 标准差= 1.9181  TV = 0.0566
    DDIM eta=0.5 steps=50      均值= +0.5549 标准差= 1.9471  TV = 0.0267

    DDIM eta=1.0 steps=10      均值= +0.6637 标准差= 1.8281  TV = 0.1434
    DDIM eta=1.0 steps=20      均值= +0.6273 标准差= 1.8942  TV = 0.0893
    DDIM eta=1.0 steps=50      均值= +0.5972 标准差= 1.9366  TV = 0.0449
```

两条结论：

1. **$$\eta=0$$ 完全确定性**：同一初始噪声跑两次逐元素相同；$$\eta=1$$ 则不同。这直接决定了 VLA 的可复现性——**确定性策略（$$\eta=0$$）意味着“同样的观测 → 同样的动作”**，这对工业部署是硬需求。
2. **少步数时 $$\eta$$ 越小越准**：10 步下 TV 从 $$\eta=0$$ 的 0.0937 单调升到 $$\eta=1$$ 的 0.1434。引入随机性会增加离散化误差；只有在步数足够多（50 步，TV 0.0249 vs 0.0449）时，$$\eta=1$$ 的差距才缩小。

自校验里还验证了 $$\eta=1$$ 的 $$\sigma_t$$ 与 DDPM 的 $$\tilde\beta_t^{1/2}$$ 数值相等（相对差 $$4.4\times 10^{-15}$$），确认两种参数化在这一点上一致。

### 8.3 Flow matching：π0 的 τ 方向与 DDPM 相反

π0 用的是线性-Gaussian 概率路径。论文里三件事必须一起看：

- 概率路径：$$\mathbf A_t^\tau = \tau\mathbf A_t + (1-\tau)\epsilon$$，$$\tau \in [0,1]$$，等价地 $$q(\mathbf A_t^\tau \mid \mathbf A_t) = \mathcal N(\tau \mathbf A_t, (1-\tau)\mathbf I)$$
- 训练目标：网络输出 $$\mathbf v_\theta(\mathbf A_t^\tau, \mathbf o_t)$$ 去匹配向量场 $$\mathbf u(\mathbf A_t^\tau \mid \mathbf A_t) = \epsilon - \mathbf A_t$$
- 推理：从纯噪声 $$\mathbf A_t^0 \sim \mathcal N(\mathbf 0, \mathbf I)$$ 出发，用前向 Euler 从 $$\tau = 0$$ 积到 $$\tau = 1$$，10 步，$$\delta = 0.1$$

**记号陷阱在 $$\tau$$ 的方向上。** $$\mathbf A_t^\tau$$ 的定义给出 $$\tau = 0 \Rightarrow \epsilon$$（纯噪声）、$$\tau = 1 \Rightarrow \mathbf A_t$$（干净动作），所以 $$\mathrm d \mathbf A_t^\tau/\mathrm d\tau = \mathbf A_t - \epsilon$$。而论文写的目标场是 $$\mathbf u = \epsilon - \mathbf A_t$$，两者相差一个负号。也就是说，“路径方向”、“速度场方向”、“积分方向”这三者不能同时按字面理解。

实践上只能对着代码核对循环变量和 `dt` 的符号。例如 GigaModels 的 pi0 推理实现用的是：

```python
dt = -1.0 / self.num_steps
timesteps = torch.arange(1.0, -dt/2, dt)   # [1.0, 0.9, ..., 0.1]
for timestep in timesteps:
    v_t = denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
    x_t += dt * v_t                        # x_t += -0.1 * v_t
```

`dt` 为负、循环变量从 1.0 递减，这与 $$\mathbf u = \epsilon - \mathbf A_t$$ 是自洽的（$$x \leftarrow x + \mathrm dt \cdot \mathbf u$$ 沿 $$-\mathbf u = \mathbf A_t - \epsilon$$ 方向移动，即朝向数据）。

本文的实验脚本用自洽的等价形式 $$\mathbf v(x,\tau) = \dfrac{\mathbb E[x_1 \mid x_\tau = x] - x}{1-\tau}$$（即 $$\mathrm d \mathbf A^\tau/\mathrm d\tau$$），并在高斯混合上有解析解，于是可以完全不训练网络地验证积分器。

### 8.4 步数：10 步是默认值，1 步会塌成一点

```text
    同一初始噪声跑两次：逐元素完全相同 = True  (ODE, 无随机性)

    Flow steps=1               均值= +0.6000 标准差= 0.0000  q5=+0.600 ... q95=+0.600
                                 与目标直方图 TV = 0.9999
    Flow steps=2               均值= +0.9658 标准差= 1.2112
                                 与目标直方图 TV = 0.4736
    Flow steps=5               均值= +0.7034 标准差= 1.6935
                                 与目标直方图 TV = 0.1857
    Flow steps=10              均值= +0.6504 标准差= 1.8263
                                 与目标直方图 TV = 0.0911
    Flow steps=20              均值= +0.6259 标准差= 1.8931
                                 与目标直方图 TV = 0.0456
    Flow steps=100             均值= +0.6075 标准差= 1.9470
                                 与目标直方图 TV = 0.0142
```

**1 步 Euler 的标准差是 0.0000。** 原因很直接：起始点是 $$x_0$$，速度为 $$\mathbf v(x_0, 0) = \mathbb E[x_1] - x_0$$（$$\tau = 0$$ 时后验就是先验，退化为常数），于是 $$x_1 = x_0 + 1 \cdot (\mathbb E[x_1] - x_0) = \mathbb E[x_1]$$——**所有样本被推到同一个点，恰好是目标均值 0.6000**。多模态结构在一步积分里被彻底抹平。

这就是流匹配策略的少步数陷阱：**它不会给出“有点模糊”的动作，而是给出“总是同一个动作”的输出**。在 VLA 里这意味着策略退化成确定性回归，失去处理多模态演示的能力。

10 步（TV 0.0911）到 20 步（0.0456）之间收益已经明显递减，100 步才到 0.0142——这是“10 步”成为工程默认值的经验依据。

### 8.5 π0 的 KV cache：积分步只重算 action 后缀

π0 的推理里有一条与生成模型共通的优化：视觉-语言前缀的 KV 只算一次，10 个积分步之间复用。论文的表述是：

> Note that inference can be implemented efficiently by caching the attention keys and values for the prefix $$\mathbf o_t$$ and only recomputing the suffix corresponding to the action tokens for each integration step.

对应到实现，就是先以 `fill_kv_cache=True` 跑一次前缀，之后每步只把 action token 送进网络。这与 LLM 自回归推理的 KV cache 是同一个思路，只是“生成的序列”变成了“积分的轨迹”。

顺带一提，π0 的 action expert 用的是**双向注意力**（论文原文：`The action expert uses a full bidirectional attention mask, so that all action tokens attend to each other`），而不是因果掩码。这与自回归解码有本质区别：**一次前向就输出整条动作 chunk（50 步），不需要逐 token 循环**。代价是 chunk 内部的 token 数固定，无法像 LLM 那样提前停止。

### 8.6 CVAE 的 latent：训练用后验，推理用什么

第三类形态来自 ACT 与 Diffusion Policy 用的条件 VAE。训练时的目标是标准的 ELBO：$$z \sim q_\phi(z \mid x, a)$$，解码器重建动作，外加 KL 项把 $$q_\phi$$ 拉向先验。

推理时的问题是：没有 $$a$$，后验算不出来。可选的做法有三种——用先验 $$z \sim \mathcal N(0, I)$$、直接把 $$z$$ 置零、或者干脆只解码 $$\mathbb E[z]$$。用一个可观测后验的玩具设定量化这三种做法的输出差异（不训练网络，直接构造 $$q_\phi(z \mid x) = \mathcal N(\mu_\phi(x), \sigma_\phi(x)^2)$$）：

```text
    latent 取法                    输出均值        输出标准差       与目标 TV         命中真值误差
    z ~ q(z|x) 训练同款            0.5996       2.0427       0.2317         0.4529
    z = 0       (ACT 做法)       0.0018       2.2877       0.5076         2.3591
    z ~ N(0,1)  (纯先验)          0.0083       2.8157       0.5501         2.8319

    训练时 KL(q(z|x) || N(0,1)) 的均值 = 2.3337 nat
    z=0 与 z~q(z|x) 的输出平均差 = 2.4552
    z~N(0,1) 与 z~q(z|x) 的输出平均差 = 2.8698
```

在这个设定里，训练同款的后验采样给出 0.5996 的输出均值（目标 0.5996），而 $$z=0$$ 和纯先验都掉到 0.002–0.008，与真值的平均误差从 0.45 涨到 2.36 / 2.83。

**要谨慎解读这个数字。** 这里的机制是“如果 $$z$$ 在训练时承载了模态信息、推理时被归零，输出会系统性地偏向平均模态”。它量化的是机制，不是某个具体模型的性能——真实 ACT 的应对手段是在训练时**随机把 $$z$$ 置零**，让解码器对 $$z=0$$ 也稳健；Diffusion Policy 则把 CVAE 换成扩散头，从根上绕开这个问题。

但它解释了一个在 VLA 代码里经常看到的现象：**同一个策略在不同推理路径下的动作方差差异很大**。取 $$z=0$$ 换来的是确定性和可复现，代价是抹掉演示里的多模态。

### 8.7 统一视角

把三类形态放在一起，可以说：**VLA 里的“采样”都是“从一个简单分布到目标分布的运输”**，只是运输的实现方式不同。

| | 离散 token | 扩散 | 流匹配 | CVAE latent |
| :--- | :--- | :--- | :--- | :--- |
| 起点 | logits → softmax | $$x_T \sim \mathcal N(0,I)$$ | $$x_0 \sim \mathcal N(0,I)$$ | $$\varepsilon \sim \mathcal N(0,I)$$ |
| 运输方式 | `argmax` / `multinomial` | 随机 SDE 递推 | 确定性 ODE 积分 | 仿射变换 |
| 步数 | 1（每 token） | 10–1000 | 1–100 | 1 |
| 可复现（固定种子） | 是 | 否（$$\eta>0$$） | 是 | — |
| 少步数失败模式 | 无 | 方差公式错用 → 塌缩 | 退化为条件均值 | 丢失模态 |
| 训练需要梯度 | 否 | 否 | 否 | **是**（重参数化） |

最后一行是 CVAE 与前两者的关键区别：**只有它需要梯度穿过采样**，也就只有它必须用 §7 的 `rsample`。

---

## 9. 让采样可复现：RNG 与确定性的实践

§2 到 §8 反复出现一个主题：**“同一个分布”不等于“同一串样本”**。这一节把可复现性的几个具体问题收一下。

### 9.1 同一分布的不同采样器：一致率是有信息量的

本文给出的几组逐元素一致率：

| 对比 | 一致率 | 结论 |
| :--- | ---: | :--- |
| `torch.multinomial` vs vLLM `p/Exp(1)` | **1.0000** | 同一段数学，RNG 消耗方式也相同 |
| `torch.multinomial` vs 手写 Gumbel-max | 0.0360 | 同分布，不同随机流 |
| 两条 Gumbel 路径（K=6） | 0.0331 | 同上 |
| top-p 升序 vs 降序写法 | 1.0000 | 代数恒等 |
| min-p 概率域 vs log 域 | 1.0000 | 代数恒等 |

**一致率 1.0 是一个很强的断言**——它说明两个实现在每一行上消耗了同样多的随机数、走了同样的比较逻辑。反过来，一致率接近随机基线说明两个实现只是“分布相同”，做替换实验时不能指望输出不变。

做 A/B 实验时这个区分很关键：**如果替换前后逐元素一致，说明改的是性能不是语义；如果只有分布一致，那么任何“输出变好了”的观察都必须用统计口径而不是逐样本比对来支撑。**

### 9.2 三个 `torch.Generator` 相关的坑

**坑一：`get_state()` 的返回值形状在不同版本间变化。**（实测中 `Generator.get_state()` 在 torch 2.10 上返回的不是可切片的两元组结构，直接对它做 tuple 索引会报错。）确定性的正确做法是从一开始就给每次实验显式建 `Generator`：

```python
g = torch.Generator().manual_seed(20260930)
torch.multinomial(probs, 1, generator=g)
```

**坑二：没有“重置”API。** `torch.Generator` 没有 `reset()`；要回到起点只能重新 `manual_seed`。本文实验里三条实现路径要在同一随机流下比对，用的就是“每次用同一个 seed 重建 generator”的方式。

**坑三：消耗量是行为的一部分。** §2.3 里“每行消耗几个均匀数”的测量不依赖任何内部计数器：用同一 seed 先生成一条参考流 $$r_0, r_1, \dots$$，复位后执行一次目标调用，随后取一个均匀数并在参考流里定位它的下标——该下标就是消耗量。对照行用 `torch.rand(1)` 验证了方法的分辨率为 1。它能把“同一算法”与“同分布的不同算法”区分开，实现见 `uniforms_consumed()`。

### 9.3 per-request 种子在服务端的代价

vLLM 的 `random_sample` 里有一段注释直接说明了逐请求种子为什么贵（`vllm/v1/sample/ops/topk_topp_sampler.py:334-346`）：

```python
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)
```

逻辑是：先假定所有请求都没有自己的种子，一次性 `q.exponential_()`；然后对确实带种子的请求逐行覆盖。**逐请求种子必须逐行调用 `exponential_`，这个循环是显式的性能负担**（源码里带着 TODO）。

同一份逻辑在 `forward_cpu` 里也有对应实现（`:161-170`），并且 CPU 路径在没有逐请求种子时走的是 `torch.compile` 包起来的版本。

还有一个连带影响写在 `forward_cuda` 里（`:126-133`）：**FlashInfer 的采样 kernel 不支持 per-request generators**，一旦请求带种子就静默回落到 PyTorch 原生实现。所以“带种子的请求”在 GPU 上不仅慢，还换了 kernel。

---

## 10. Lab 与复现

本文所有数字来自 `tools/sampling_bench/` 下的 5 个脚本，全部**纯 CPU 可跑**，每个都有 `--selftest` 作为前置门禁。运行环境：torch 2.10.0，单线程（成本测试用 4 线程），float64 为主。

| 脚本 | 覆盖 | 关键自校验 |
| :--- | :--- | :--- |
| `l1_sampler_primitives.py` | 有放回三写法、无放回指数竞赛 vs 逐步剔除、退化输入行为表 | 错误分布 max-z = 439 vs 真采样 2.03（判据自身可信） |
| `l2_logprob_paths.py` | 五条 logprob 路径、fp32/fp64 漂移、ratio 的三条误差源 | 与解析解 $$3-\log(1+e^3)$$ 对比，差 $$\le 1.5\times 10^{-16}$$ |
| `l3_warping_ops.py` | top-p 三写法、min-p 概率域 vs log 域、top-k 并列值、成本表 | 均匀 logits 下 top-p=0.5 应保留 2 个（实测 2/2） |
| `l4_reparam_gumbel.py` | `rsample` 展开、gradcheck 的正确用法、两条 Gumbel 路径、straight-through | $$\text{rsample} = \text{loc}+\varepsilon\cdot\text{scale}$$ 逐元素相等 |
| `l5_vla_transport.py` | DDPM 跳步、DDIM 的 $$\eta$$、flow matching 步数、CVAE 先验偏移 | $$\eta=1$$ 的 $$\sigma_t$$ 与 $$\tilde\beta_t^{1/2}$$ 相对差 $$4.4\times 10^{-15}$$ |

复现任一节：

```bash
# 前置门禁：先确认判据本身可用
python tools/sampling_bench/l1_sampler_primitives.py --selftest

# 全量
python tools/sampling_bench/l1_sampler_primitives.py
python tools/sampling_bench/l2_logprob_paths.py
python tools/sampling_bench/l3_warping_ops.py
python tools/sampling_bench/l4_reparam_gumbel.py
python tools/sampling_bench/l5_vla_transport.py
```

### 10.1 一张总结表：本文确认与推翻的东西

| 论断 | 状态 | 依据 |
| :--- | :--- | :--- |
| `torch.multinomial` 单样本走 Gumbel fast path | **源码确认** | `multinomial_out` 的 `!with_replacement \|\| n_sample == 1` |
| 该路径与 vLLM `p/Exp(1)` 逐元素等价 | **实测确认** | 40 万次一致率 1.0000，且每行消耗 $$K$$ 个均匀数 |
| 无放回是 O(n+mlogm) 的指数竞赛 | **源码 + 实测** | PR #47386、issue #11931；与精确联合分布 TV 0.003 |
| 文档“无放回受非零元素数约束” | **实测推翻** | `w=[1,0,0], k=3` 返回 `[[0,1,2]]`，实现检查的是类别总数 |
| `Categorical(probs=…)` 与 `logits=…` 等价 | **实测推翻** | 极端概率下差 83.96 nat（`clamp_probs` 的 $$\varepsilon$$ 地板） |
| top-p 三种写法等价 | **实测确认** | 全部测试点 mask 逐元素相同 |
| min-p 概率域与 log 域等价 | **实测确认** | fp64/fp32 下 mask 逐元素相同 |
| top-k 是严格的基数约束 | **实测推翻** | 并列值下 `top-k=1` 保留 3 个 |
| “用秩代替排序”更快 | **实测推翻** | 双 `argsort` 约为全排序的 1.7 倍耗时 |
| `exponential_` 在 CPU 上等于 `-log(1-U)` | **实测确认** | 同一 seed 下逐元素相等 |
| `ExpRelaxedCategorical` 与 `F.gumbel_softmax` 逐元素相同 | **实测推翻** | 一致率 3.3%（分布相同） |
| DDPM 可以抽子序列加速 | **实测推翻** | TV 从 0.0156 涨到 0.7364 |
| DDIM $$\eta=0$$ 确定性 | **实测确认** | 同初始噪声两次结果逐元素相同 |
| Flow matching 1 步可用 | **实测推翻** | 标准差 0.0000，塌到条件均值 |
| `gradcheck` 可以直接验 `rsample` | **实测推翻** | 每次调用重抽 $$\varepsilon$$，必然 `GradcheckError` |
| bf16 打分只影响**性能**不影响 ratio | **实测推翻** | ratio 偏离 1 达 $$3.5\times 10^{-2}$$ |

### 10.2 给工程实践的三条

1. **要 logprob 就传 `logits`，不要传 `probs`。** 前者走 `x - logsumexp`，后者走 `log(clamp(p, eps, 1-eps))`。这一条在概率极小的 token 上会带来几十 nat 的静默误差。
2. **`logprobs` 必须和采样配置绑定记录。** 塑形（top-k / top-p / min-p）、精度（fp32 / bf16）、是否截断，任何一项不一致，ratio 就不再是“策略变化”的度量。vLLM 的 `logprobs_mode` 四档正是为了显式化这个口径。
3. **连续动作采样的步数是分布性质，不是性能旋钮。** DDPM 不能抽子序列；flow matching 的 1 步会把多模态演示退化成条件均值；DDIM 的 $$\eta=0$$ 是确定性策略，$$\eta>0$$ 不是。选步数之前先确认要保留哪个性质。

---

## 相关与脚本

- 服务侧口径：[vLLM 推理服务 profiling](/2026/09/29/profiling-02-inference/)（`goodput` / `ITL` 的度量边界）
- 数值基础：[GPU 超越函数与 softmax](/2026/09/26/gpu-transcendental-softmax-llm/)（采样之前的 exp/log 代价）
- RL 侧衔接：[RL 训练框架全景横评（2026）](/2026/09/29/rl-training-frameworks-survey/)、[RL 符号与估计器全景](/2026/09/19/rl-notation-and-estimators-deepdive/)、[RL 框架里的那些变量到底怎么算](/2026/09/30/rl-variables-in-frameworks/)
- VLA 侧衔接：[World Model 与 WAM](/2026/09/05/world-model-world-action-model/)
- 脚本：`tools/sampling_bench/` 下的 `l1_sampler_primitives.py`、`l2_logprob_paths.py`、`l3_warping_ops.py`、`l4_reparam_gumbel.py`、`l5_vla_transport.py`

---

*本文所有源码引用对应：torch 2.10.0（`aten/src/ATen/native/Distributions.cpp`、`aten/src/ATen/native/cpu/MultinomialKernel.cpp`、`aten/src/ATen/native/cuda/MultinomialKernel.cu`、`torch/distributions/*`）、transformers 5.3.0.dev0（commit `556312cd`）、vLLM（commit `719735d6`）、TRL。π0 的公式引自 arXiv:2410.24164。*

---

*上一篇：[RL 框架里的那些变量到底怎么算](/2026/09/30/rl-variables-in-frameworks/) ｜ 下一篇：待定*
