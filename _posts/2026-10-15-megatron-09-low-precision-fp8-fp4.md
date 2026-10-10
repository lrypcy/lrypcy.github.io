---
title: "Megatron-LM 深度剖析（09）：低精度——FP8 全链路与 FP4"
date: 2026-10-15 10:00:00 +0800
categories:
  - 分布式训练
tags: [megatron, megatron-core, fp8, fp4, nvfp4, mxfp8, quantization, loss-scaling, grad-clip, transformer-engine, low-precision, source-code]
layout: post
mathjax: true
excerpt: "fp8_recipe 的五个取值在源码里落到三条完全不同的路径。本文逐层拆 fp8_utils.py 的类型判定链与版本抽象层、transformer_engine.py 的 recipe 分派表、clip_grads.py 与 grad_scaler.py 这两个被 05 篇漏掉的孤儿，并手撸 E4M3/E5M2/MX-block/INT8 四类量化器回答一个具体问题：scale 粒度换来的收益，究竟是精度还是量程。"
---

> 源码基准：`megatron-core` **0.20.0**，commit [`60e039626`](https://github.com/NVIDIA/Megatron-LM/tree/60e039626)。本文所有行号均对该 commit 取证。
> 配套实验：`ipynbs` 仓库 `experiments/megatron/low_precision/`（两个脚本 + 固化输出）。
> <!-- TODO-ipynbs-commit: 监工 push 后回填本篇实验目录的 permalink 前缀 -->

低精度这条线在 Megatron 里的复杂度，超出「换个 dtype」很多。

`--fp8-recipe` 一个 flag 底下有五个取值，每个取值在源码里落到**不同的类、不同的构造时机、不同的版本门槛**。而 `fp8_utils.py` 一开篇就是 1056 行里最费解的一段：几十个 `is_*tensor` 谓词、`copy_*_to_quantized_param`、`modify_*_storage`，它们服务的对象是 Transformer Engine 的张量子类，而那些子类的名字**随 TE 大版本变**。

本文要做三件事：

1. 把「五个 recipe 取值」落到源码里真实的那几条路径上，并指出**它们不是同一层的东西** —— `delayed` 有构造函数、有状态，其余四个是每次前向现构造的无状态上下文。
2. 把契约 §2.1 划给本篇的两个孤儿（`clip_grads.py` 277 行、`grad_scaler.py` 165 行）各自写成可独立阅读的一节。它们在第 05 篇写完时提及次数是 1 和 **0**。
3. 用手撸的量化器回答一个具体问题：**scale 粒度换来的收益，是精度还是量程。**

---

> **TL;DR**
>
> 1. **`fp8_recipe` 的五个取值不在同一层。** `delayed` 走 `get_fp8_recipe()`（[`fp8_utils.py` L752-L810](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L752-L810)）构造一个**有状态**的 `TEDelayedScaling`；`tensorwise` / `blockwise` / `mxfp8` / `custom` 走每层现构造的**无状态**上下文。后者的分派表在 `transformer_engine.py` L337-L358，那张表里**根本没有 `delayed`**。
> 2. **版本门槛是逐 recipe 独立的**，不是一个总开关：`tensorwise` 要 TE ≥ 2.2.0.dev0、`blockwise` 要 ≥ 2.3.0.dev0、`mxfp8` 要 ≥ 2.1.0，而 TE < 2.1.0 时源码直接 `assert` 你必须用 `delayed`。
> 3. **`fp8_param_gather` / `fp4_param_gather` 会被静默关掉，但只在 `--use-torch-fsdp2` 下。** `arguments.py` 那两段降级逻辑包在 `if args.use_torch_fsdp2:`（`arguments.py` L1023）里。显式传了 flag 不报错，只打一行 `warn_rank_0` 就改回 `False`。
> 4. **粒度对 FP8 的作用是「量程」不是「精度」。** FP8 是浮点格式，缩小 scale 不增加尾数位。实测：MX 块大小取 1 / 8 / 32 / 128 / 512，RMSE **完全相同**（1.945893e-02）；但在一组「块内混着极小值」的数据上，per-tensor 把 **60.9%** 的极小值清零，MX-block32 清 **47.6%**，块大小降到 8 才降到 **0.0%**。
> 5. **FP8 与 INT8 对 outlier 的反应相反。** INT8 是均匀量化，抬高 scale 等比例放大步长，per-tensor 的 SNR 掉 10 dB；FP8 浮点格式的相对精度与 scale 无关，几乎不掉。**这正是选浮点格式的理由。**
> 6. **`clip_grads.py` 的整块 `torch.norm` 与逐参数循环数学恒等、浮点差 1 ULP。** 更值得记的是 L135-L138 那两条 fallback：`.item()` 把 float32 提升成 Python float，所以**没有 `multi_tensor_scale_tensor` 的那条分支开根精度反而更高**。
> 7. **`grad_scaler.py` 的 `hysteresis` 不是「连续 inf 计数」。** 它只在**成功涨 scale 时**被重置（`grad_scaler.py` L151），一次干净的迭代不会重置它 —— 所以零散的 inf 也会累积到触发退避。
> 8. **`get_fp4_align_size` 的 docstring 推导 32，函数 `return 128`。** 不是笔误：docstring 先给出 TMA 对齐的硬件下限 32，再说明为什么实际取 128（Hadamard 融合 kernel 与 MoE CUDA graph 的 padding 要求）。同一段 docstring 里挂着三条出处链接。

---

## 0. 符号与口径

| 符号 | 含义 | 出处 |
| --- | --- | --- |
| $$a$$ | 一个元素在浮点格式下的实数值 | 本文记法 |
| $$s$$ | scale（缩放因子） | TE 文档 |
| $$X$$ | MX block 的共享 scale（E8M0 编码） | [arXiv:2310.10537](https://arxiv.org/abs/2310.10537) |
| $$k$$ | MX block 的元素数，MXFP8 取 32 | 同上 |
| $$\operatorname{amax}(A)$$ | 张量或块的绝对值最大值 | TE 文档 |
| $$\mathrm{RTNE}$$ | Round To Nearest Even | TE 文档 |

三个「精度」不混用，全文分清：

- **相对精度**：单个值的相对误差。浮点格式下**与 scale 无关**（只要不越出规格化区）。
- **量程利用率**：一组值能同时用满多少动态范围。**这一项才随 scale 粒度变。**
- **block scaling 的真实作用**：让每个块各自用满量程，从而避免小组内的极小值掉进次正规区。

本篇所有量化误差都是自己跑出来的，**数据是合成正态 + 人为注入的极值，不是真实权重**。绝对数值不可外推，只用于比较不同方案之间的相对关系。

---

## 1. 三种 scaling 策略与 amax 归约

### 1.1 配置层：五个取值、两种格式

字段定义在 [`transformer_config.py` L580-L593](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L580-L593)。**格式与 recipe 是两个正交维度**：

```python
fp8: Optional[Literal['e4m3', 'hybrid']] = ...
"""If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

fp8_recipe: Optional[Literal['tensorwise', 'delayed', 'mxfp8', 'blockwise', 'custom']] = (
    "delayed"
)
```

`hybrid` 的选择理由写得很直白：**激活与权重要精度，用 E4M3；输出激活梯度更容易被舍入吃掉、要量程，用 E5M2。** 这与 TE 官方文档的默认一致（[FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)）：*By default, Transformer Engine uses a hybrid approach: Forward pass - activations and weights require more precision, so E4M3; Backward pass - gradients are less susceptible to precision loss but require higher dynamic range, so E5M2 is preferred.*

**默认值是 `delayed`。** 这条默认值本身就是一条信息 —— 它是五个取值里唯一需要跨 iteration 状态的。

（层内 norm 用的是 RMSNorm，见 [`torch.nn.RMSNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)；它没有 bias、也没有均值中心化，所以省下的那一份激活就是 08 篇 §1.3 清单里 k / v 与 FFN 中间量那一类。）

配套的四个调节量（[L606-L617](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L606-L617)）全部只对 `delayed` 有意义：`fp8_margin`、`fp8_interval`、`fp8_amax_history_len`、以及 `fp8_amax_compute_algo`（`most_recent` / `max`）。

### 1.2 `get_fp8_recipe`：五条路与它们的版本门槛

真正构造 recipe 的地方在 [`fp8_utils.py` L752-L810](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L752-L810)。逐字看关键分支：

```python
if is_te_min_version("2.1.0"):
    if config.fp8_recipe == Fp8Recipe.delayed:
        fp8_recipe = TEDelayedScaling(
            config=config, fp8_format=fp8_format,
            override_linear_precision=(False, False, not config.fp8_wgrad))
    elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
        fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
            fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention)
    elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
        fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(fp8_format=fp8_format)
    elif config.fp8_recipe == Fp8Recipe.mxfp8:
        fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
            fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention)
    elif config.fp8_recipe == Fp8Recipe.custom:
        assert config.fp8_quantizer_factory is not None
        fp8_recipe = _get_custom_recipe(config.fp8_quantizer_factory)
    else:
        raise ValueError(...)
else:
    assert config.fp8_recipe == Fp8Recipe.delayed, (...)
    fp8_recipe = TEDelayedScaling(...)
```

四点读数：

| recipe | TE 门槛 | 构造的类 | 有状态 |
| --- | --- | --- | --- |
| `delayed` | 2.1.0 | `TEDelayedScaling` | **是**（amax history） |
| `tensorwise` | 2.2.0.dev0 | `Float8CurrentScaling` | 否 |
| `blockwise` | 2.3.0.dev0 | `Float8BlockScaling` | 否 |
| `mxfp8` | 2.1.0 | `MXFP8BlockScaling` | 否 |
| `custom` | 2.9.0.dev0 | `CustomRecipe` | 由工厂决定 |

**门槛是逐 recipe 独立的，不是一个总开关。** TE 版本落在 2.1.0 与 2.2.0.dev0 之间时，`tensorwise` 会掉进 L792 的 `else` 抛 `ValueError`，而不是静默降级 —— 这是好的失败方式。

第三行的 `override_linear_precision=(False, False, not config.fp8_wgrad)` 值得单说：第三个分量控制 wgrad 是否用高精度。`fp8_wgrad` 为假时它被翻成 `True`，即**权重梯度强制走高精度**。这是 Megatron 在 fp8 recipe 之上唯一的额外精度出口。

`TEDelayedScaling` 本体在 [`transformer_engine.py` L3295-L3328](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/extensions/transformer_engine.py#L3295-L3328)，是 TE `DelayedScaling` 的薄壳，把 config 字段一对一映射过去（`margin` / `fp8_format` / `amax_compute_algo` / `amax_history_len`）。里面还有一处软弃用（`transformer_engine.py` L3316-L3319）：

```python
if get_te_version() < PkgVersion("1.8.0"):
    extra_kwargs["interval"] = config.fp8_interval
elif config.fp8_interval != 1:
    warnings.warn("fp8_interval is deprecated and ignored from Transformer-TE v1.8.0.")
```

### 1.3 第二条路：每层现构造的无状态上下文

`delayed` 之外的四个取值不走 `get_fp8_recipe`。它们在每层前向时现算 recipe 并开 autocast 上下文，分派表在 [`transformer_engine.py` L337-L358](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/extensions/transformer_engine.py#L337-L358)：

```python
elif qrecipe.fp8_quantization_recipe is not None:
    if qrecipe.fp8_format == "e4m3":
        fp8_format = te.common.recipe.Format.E4M3
    elif qrecipe.fp8_format == "hybrid":
        fp8_format = te.common.recipe.Format.HYBRID
    else:
        raise ValueError(f"Unhandled fp8_format {qrecipe.fp8_format}")

    if qrecipe.fp8_quantization_recipe == Fp8Recipe.tensorwise:
        quant_recipe = te.common.recipe.Float8CurrentScaling(fp8_format=fp8_format)
    elif qrecipe.fp8_quantization_recipe == Fp8Recipe.blockwise:
        quant_recipe = te.common.recipe.Float8BlockScaling(fp8_format=fp8_format)
    elif qrecipe.fp8_quantization_recipe == Fp8Recipe.mxfp8:
        quant_recipe = te.common.recipe.MXFP8BlockScaling(fp8_format=fp8_format)
    else:
        raise ValueError(f"Unhandled fp8 recipe: {qrecipe.fp8_quantization_recipe}")
else:
    # Fp4 configured.
    if qrecipe.fp4_quantization_recipe == Fp4Recipe.nvfp4:
        quant_recipe = te.common.recipe.NVFP4BlockScaling()
```

**这张表里没有 `delayed`。** 这不是遗漏 —— `delayed` 的 recipe 对象是有状态的、要跨 iteration 持有 amax 历史，没法在每层前向现造一个。这是本节最值得记住的结构事实：

| | `delayed` | `tensorwise` / `blockwise` / `mxfp8` / `custom` |
| --- | --- | --- |
| recipe 对象 | 模型级，构造一次 | 每层现构造 |
| 是否有状态 | 有（amax history） | 无 |
| 走的函数 | `get_fp8_recipe()` | `_get_fp8_autocast_for_quant_recipe()` |
| autocast 上下文 | 由上层统一开 | 每层 `fp8_autocast(enabled=True, ...)` |

### 1.4 amax 归约组：四个组合

per-tensor 粒度的 recipe（`delayed` / `tensorwise`）必须先把 amax 跨 rank 归约出来，否则各 rank 算出的 scale 不一致，gather 之后的张量会被按错误的量程量化。归约组由 [`parallel_state.py` L1877-L1900](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/parallel_state.py#L1877-L1900) 给出：

```python
def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
```

两个布尔量决定四个组：

| `with_context_parallel` | `tp_only_amax_red` | 返回的组 |
| --- | --- | --- |
| False | False | `_TENSOR_AND_DATA_PARALLEL_GROUP` |
| False | True | `_TENSOR_MODEL_PARALLEL_GROUP` |
| True | False | `_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP` |
| True | True | `_TENSOR_AND_CONTEXT_PARALLEL_GROUP` |

四个分支都以 `assert ... is not None, "FP8 amax reduction group is not initialized"` 收尾 —— **组没初始化就断言失败，不静默退回单卡**。

这个网格就是第 01 篇那套「进程组网格」在低精度上的一个投影：TP / DP / CP 三个轴各自要不要参与 amax 归约，由两个开关正交控制。`tp_only_amax_red=True` 的含义是「只跨 TP 归约」，此时返回的就是 TP 组本身（`parallel_state.py` L1897-L1899）。

两处调用点都传了 `with_context_parallel=True`（[`fp8_utils.py` L859-L861](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L859-L861) 与 `transformer_engine.py` L325-L328），实际的选择由 `config.tp_only_amax_red` 决定落在上表哪一行。

TE 侧对这个机制的说明在 [FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) 的 Distributed training 一节：*Tensors that are gathered across nodes require amax synchronization before quantization. Each node computes its local `amax`, then a reduction produces the global maximum across all nodes. All nodes use this synchronized amax to compute identical scaling factors, enabling quantized all-gather.* —— 顺序是「先归约 amax，再量化，再 all-gather」，不是反过来。

**block 粒度的两个 recipe 不需要这个归约。** `blockwise` 给每 128 个元素一个 scale、`mxfp8` 给每 32 个一个，块间不共享 scale，归约没有对象。`get_amax_reduction_group` 的存在本身就是 per-tensor 与 block 两种粒度的结构分界。

### 1.5 一个只有 delayed 才有的修补

`fp8_utils.py` 里有一段专门解释版本差异的注释（[L404-L408](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L404-L408)）：

> `correct_amax_history_if_needed`
> This function is used to correct the amax history of fp8 tensors. **In TE1.x, some inplace copy operations will write unwanted values to the amax_history of fp8 tensors.** This function corrects the amax_history back. For TE2.x, it's an empty function. Only useful for delayed scaling.

「有些原地拷贝会把不该写的值写进 amax 历史」—— 这是 TE1.x 的一个副作用，也只有 delayed 有 amax 历史可写。第 2.8 节会看到，`--fp8-param-gather` 的实现要大量做原地操作，正好踩在这个副作用上。

### 1.6 本节的一张图

```mermaid
graphLR
  R["fp8_recipe 五取值<br/>落到两条不同的构造路径"]
  P1["delayed<br/>模型级构造 有状态"]
  P2["tensorwise blockwise mxfp8 custom<br/>每层现构造 无状态"]
  G1["get_amax_reduction_group<br/>四个组 需跨 rank 归约 amax"]
  G2["无需归约<br/>块间不共享 scale"]
  D["amax history<br/>仅 delayed 有"]

  R --> P1
  R --> P2
  P1 --> G1
  P1 --> D
  P2 --> G2
```

图注：数据/结论来源（逐条已核）—— `fp8_recipe` 五取值取自 [`transformer_config.py` L587-L593](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L587-L593)；两条构造路径取自 [`fp8_utils.py` L752-L810](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L752-L810) 与 [`transformer_engine.py` L337-L358](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/extensions/transformer_engine.py#L337-L358)；四个 amax 归约组取自 [`parallel_state.py` L1877-L1900](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/parallel_state.py#L1877-L1900)。图为自绘，未复制任何官方或论文图。

---

## 2. param gather 的精度路径

### 2.1 flag 会被静默关掉 —— 但只在 FSDP2 下

这是本篇第一个必须讲的坑。`arguments.py` L1043-L1056：

```python
if args.fp8_param_gather and is_te_min_version("2.0.0"):
    args.fp8_param_gather = False
    warn_rank_0(
        'FSDP2 FP8 param gather is not supported yet in TE 2.0, will fallback to bf16'
        'all_gather instead, turning off fp8_param_gather',
        args.rank,
    )
if args.fp4_param_gather and not is_te_min_version("2.7.0.dev0"):
    args.fp4_param_gather = False
    warn_rank_0(
        'FSDP2 FP4 param gather is not supported yet in TE 2.0, will fallback to bf16'
        'all_gather instead, turning off fp4_param_gather',
        args.rank,
    )
```

**先说清楚触发条件，这一条常被误读。** 往上翻 20 行，这两段包在 [`arguments.py` L1023](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/training/arguments.py#L1023) 的 `if args.use_torch_fsdp2:` 里 —— 那个块里全是 `--use-torch-fsdp2` 的前置断言（要求 PyTorch ≥ 2.4.0、不支持 PP、不支持 EP、不支持 MCore 分布式优化器等）。

所以准确的说法是：**只有开了 `--use-torch-fsdp2` 时，这两段降级逻辑才会执行。** 走 MCore 原生分布式优化器那条路时，这两个 flag 不会被碰。

即便如此，静默降级这件事本身仍值得记：显式传了 flag，程序不报错、不警告退出码，只打一行 `warn_rank_0` 就把 flag 改回 `False`。**训练照跑，显存账照算，只是你以为省下的那部分带宽并没有省。**

还有一处文案与条件不同步：`fp4_param_gather` 那一支的判定是 `not is_te_min_version("2.7.0.dev0")`，但打印的文案写的是 *not supported yet in TE **2.0***。两处版本号对不上 —— 文案是从上一支复制来的。按契约的引文忠实性原则，这里原样引用，不改。

### 2.2 精度路径的入口：`map_dtype`

这一组字段最终要落到具体的搬运与类型转换上，语义见 [`Tensor.to`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html)。低精度相关字段真正被解释成 `torch.dtype` 的地方在 [`arguments.py` L1069-L1077](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/training/arguments.py#L1069-L1077)：

```python
dtype_map = {
    'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16,
    'fp8': torch.uint8, 'auto': None, None: None,
}
map_dtype = lambda d: d if isinstance(d, torch.dtype) else dtype_map[d]

args.main_grads_dtype = map_dtype(args.main_grads_dtype)
args.main_params_dtype = map_dtype(args.main_params_dtype)
args.exp_avg_dtype = map_dtype(args.exp_avg_dtype)
args.exp_avg_sq_dtype = map_dtype(args.exp_avg_sq_dtype)
```

一处值得单独指出：**`'fp8'` 映射到 `torch.uint8`，不是某个浮点类型。** FP8 在 PyTorch 里没有一等公民地位，它借 `uint8` 存储原始位模式。这个选择在 `fp8_utils.py` 里到处都能看到对应 —— `is_grouped_tensor_with_quantized_storage` 判定量化存储的方式就是查 dtype（[L144-L150](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L144-L150)）：

```python
rowwise_data = getattr(tensor, "rowwise_data", None)
return rowwise_data is not None and rowwise_data.dtype == torch.uint8
```

**`dtype == torch.uint8` 就是「这块存储是量化过的」的唯一标记。** 后面所有 `copy_*` / `modify_*` 函数都靠它分流。

### 2.3 三个接口函数：把 TE 的版本差异关进一层

`--fp8-param-gather` 需要在 param gather 阶段把 fp32 主权重量化成低精度再 all-gather。难点在于「fp8 张量的原始数据不在 `.data` 里，且随 TE 版本与 recipe 变」。`fp8_utils.py` 用一段模块级注释交代了这件事（[L385-L409](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L385-L409)）：

> The code below abstracts the functionalities needed for implementing `"--fp8-param-gather"` into several functions. It provides different implementations for each function based on different versions of TE, ensuring compatibility across various TE versions.
>
> Currently, there are three functions:
> - `modify_underlying_storage` — This function is used in DDP to place all parameters into a contiguous buffer. For non-fp8 tensors, replacing their data is simple, just using code like `tensor.data = new_data`. However, **for fp8 tensors, their raw data is not stored in the `.data` attribute**, and it varies with different TE versions and different recipes.
> - `quantize_param_shard` — This function is used in dist-opt to cast fp32 main params to fp8 params. For non-fp8 params, this casting is as simple as `bf16_params.copy_(fp32_main_params)`; but for fp8 params, the casting logic varies with different TE versions and different recipes. This function provides a unified interface to cast fp32 main params to fp8 params, and also **updates the necessary attributes (like amax, scale, scale_inv or transpose cache) of the fp8 model params**.
> - `correct_amax_history_if_needed` — ...

三个对外接口函数本身只有两三行，全是转发（[L666-L684](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L666-L684)）：

```python
# Interface Function
def modify_underlying_storage(tensor, new_raw_data):
    _modify_underlying_storage_impl(tensor, new_raw_data)

# Interface Function
def quantize_param_shard(model_params, main_params, start_offsets, data_parallel_group,
                         fsdp_shard_model_params=None):
    _quantize_param_shard_impl(...)

# Interface Function
def correct_amax_history_if_needed(model):
    _correct_amax_history_if_needed_impl(model)
```

**真正的实现在 `_impl` 里按 TE 版本分叉。** 2.2+ 的那条分叉长这样（[L410-L420](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L410-L420)）：

```python
if HAVE_TE and is_te_min_version("2.2"):
    # Supported TE versions: 2.2+
    from transformer_engine.pytorch.tensor import QuantizedTensor

    def _modify_underlying_storage_impl(fp8_tensor: QuantizedTensor, new_raw_data):
        from transformer_engine.pytorch.tensor.utils import replace_raw_data
        replace_raw_data(fp8_tensor, new_raw_data)
```

注意 `replace_raw_data` 是**从 TE 内部模块路径导入的**（`transformer_engine.pytorch.tensor.utils`），不是公开 API。这是第 6 节要展开的依赖形态的第一个例子。

### 2.4 gather 之后的还原：`post_all_gather_processing`

低精度 param gather 还有一个尾巴：backward 需要转置布局的权重，而转置视图不是 gather 出来的。`fp8_utils.py` L687-L708 处理这件事：

```python
def post_all_gather_processing(model_params):
    """
    Post-processing after all-gather for weights in distributed optimizer.
    - tensorwise: may need to create a transposed view to match backend GEMM.
    - blockwise: create column-wise storage.
    """
```

两步。先把 grouped 量化参数**展开成逐成员的量化张量**（`fp8_utils.py` L696-L701）：

```python
for param in model_params:
    if is_grouped_tensor_with_quantized_storage(param):
        expanded_model_params.extend(get_grouped_quantized_members(param))
    else:
        expanded_model_params.append(param)
```

再交给 TE（`fp8_utils.py` L703-L708）：

```python
if te_post_all_gather_processing is not None:
    te_post_all_gather_processing(expanded_model_params)
else:
    # If the TE version is old and does not have post_all_gather_processing function, this is
    # a no-op, and the transpose/columnwise data will be created in the next forward pass.
    pass
```

**TE 版本太老时的兜底是「推迟到下一次前向再建」**，不是报错。这条注释给出了两个信息：这个函数是后加的；它缺席时功能不会坏，只是延后。

TE 侧对转置的说明在 [FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) 的 Transpose handling 一节：backward 需要转置后的 FP8 张量，columnwise 与 rowwise 布局物理上不同，所以要么显式转置、要么多存一份。块粒度的 `blockwise` 在这里更麻烦 —— 官方文档在 [FP8 Blockwise Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html) 里明说 *transposing a 1D quantized tensor is not supported*，因为转置后行块与列块覆盖的元素集合不同，scale 也不同，只能从高精度源分别量化两遍。

### 2.5 回引 05 §2.5：NVFP4 打包让 numel 减半

第 05 篇已经讲过 param buffer 在 NVFP4 下 numel 减半。本篇接着往下走 —— 具体是哪一行做的。

答案是 [`fp4_utils.py` L70-L77](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp4_utils.py#L70-L77)：

```python
def get_nvfp4_rowwise_packed_shape(shape: torch.Size) -> torch.Size:
    """Return packed byte shape for NVFP4 rowwise storage (last dim // 2)."""
    if len(shape) == 0:
        return shape
    assert shape[-1] % 2 == 0, "NVFP4 requires inner dimension divisible by 2"
    packed = list(shape)
    packed[-1] = packed[-1] // 2
    return torch.Size(packed)
```

**这是纯位打包，不是量化。** 两个 4-bit 值塞进一个字节，所以最后一维减半，numel 也减半。代价有两个，都写在源码里：

1. **`assert shape[-1] % 2 == 0`** —— 最后一维必须是偶数。权重矩阵的 hidden 维通常是 2 的幂，这个条件一般成立；但如果最后一维是奇数，这里直接断言失败。
2. **解包要单独一步**。`dequantize_fp4_tensor`（[L208-L213](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp4_utils.py#L208-L213)）：

```python
def dequantize_fp4_tensor(fp4_tensor):
    if is_te_min_version("2.7.0.dev0"):
        return fp4_tensor.dequantize()
    else:
        raise RuntimeError("FP4 dequantization requires Transformer Engine >= 2.7.0.dev0")
```

**注意与 FP4 对照：FP8 的版本分支是静默的**（[`fp8_utils.py` L303-L308](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L303-L308)）：

```python
def dequantize_fp8_tensor(fp8_tensor):
    if is_te_min_version("2.0"):
        return fp8_tensor.dequantize()
    else:
        return fp8_tensor.from_float8()
```

同一个文件里，FP8 老版本走另一个 API 名，FP4 老版本直接抛异常。两种策略不一致，源码没解释。

### 2.6 类型判定链：为什么有这么多 `is_*tensor`

`fp8_utils.py` 开篇 90 行几乎全是类型判定。要看清它解决什么问题，只需读 `is_float8tensor` 的 docstring（[L122-L131](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L122-L131)）：

> Note that in TE2.x, in order to support more recipes, **the design of the fp8 tensor class has changed. Now Float8Tensor is only used for current scaling and delayed scaling. And mxfp8 and blockwise scaling have their own fp8 tensor classes. These different fp8 tensor classes are both inherited from `QuantizedTensor`. So, for TE1.x, `FP8_TENSOR_CLASS` is `Float8Tensor`, and for TE2.x, `FP8_TENSOR_CLASS` is `QuantizedTensor`.

**类名本身是版本相关的。** 同一份 `fp8_utils.py`，TE1.x 下 `is_float8tensor` 判的是 `Float8Tensor`，TE2.x 下判的是 `QuantizedTensor`。这就是为什么判定函数不能写死一个类 —— `FP8_TENSOR_CLASS` 这个模块级名字在不同 TE 版本下指向不同东西。

整条判定链的入口是两个小工具（[L104-L119](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L104-L119)）：

```python
def _unwrap_parameter_data(tensor):
    """Return underlying tensor data when PyTorch wraps a tensor subclass as a Parameter."""
    if HAVE_TE_GROUPED_TENSOR_CLASS and isinstance(tensor, GroupedTensor):
        # TE GroupedTensor stores its real payload in Python-side metadata fields
        # such as rowwise_data/scale_inv. PyTorch marks tensor-subclass parameters
        # as Parameters, so tensor.data would create a detached wrapper copy. Return
        # the live wrapper so storage metadata mutations update the module parameter.
        return tensor
    return tensor.data if isinstance(tensor, torch.nn.Parameter) else tensor

def _is_instance_or_param_data(tensor, tensor_class) -> bool:
    """Check a tensor subclass, including when wrapped by torch.nn.Parameter."""
    return isinstance(tensor, tensor_class) or isinstance(
        _unwrap_parameter_data(tensor), tensor_class)
```

`_unwrap_parameter_data` 的存在理由是：PyTorch 会把张量子类的实例标成 `Parameter`，此时 `tensor.data` 会造出一个**脱离的包装副本** —— 改了副本，改不到模块参数。对 `GroupedTensor` 尤其致命，因为它的载荷在 `rowwise_data` / `scale_inv` 这些 Python 侧元数据里。

**后面每一个 `_is_instance_or_param_data(...)` 的调用，都是在防这一类「改了但没改到」的静默失效。**

完整的判定链（[L134-L190](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L134-L190)）：

| 谓词 | 判什么 |
| --- | --- |
| `is_float8tensor` | TE1.x 的 `Float8Tensor` / TE2.x 的 `QuantizedTensor` |
| `is_mxfp8tensor` | `MXFP8Tensor`（TE2.x 专有） |
| `is_grouped_tensor` | `GroupedTensor` |
| `is_grouped_tensor_with_quantized_storage` | `GroupedTensor` 且 `rowwise_data.dtype == torch.uint8` |
| `is_grouped_mxfp8tensor` | grouped 且 recipe 报 `mxfp8()` |
| `get_grouped_quantized_members` | 取缓存的逐成员视图，可选缺失时创建 |

`get_grouped_quantized_members` 里有一条**性能契约**（[L182-L189](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L182-L189)）：

```python
if quantized_members is None:
    if not create_if_missing:
        raise RuntimeError(
            "Grouped quantized parameter is missing cached member tensors. "
            "Create them outside the training critical path.")
    quantized_members = grouped_tensor.split_into_quantized_tensors()
    grouped_tensor.quantized_tensors = quantized_members
```

报错文案直接告诉你该在哪儿建：**训练关键路径之外。**

### 2.7 写回量化存储：两条路与它们的守卫

`copy_tensor_to_quantized_param`（[L193-L226](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L193-L226)）是「把高精度值写进量化存储」的唯一入口。它分两条路。

grouped 路先过三道守卫：

```python
if is_grouped_tensor_with_quantized_storage(dst):
    if src.numel() != dst.numel():
        raise ValueError("Grouped quantized parameter copy size mismatch: ...")
    if not dst.all_same_shape():
        raise NotImplementedError(
            "Copying into grouped quantized parameters requires uniform member shapes.")
```

然后逐成员量化写入（`fp8_utils.py` L220-L221）：

```python
for src_member, dst_member in zip(src_members, quantized_members):
    dst.quantizer.update_quantized(src_member, dst_member)
```

**为什么不用 `GroupedTensor.copy_`**，注释给了理由（`fp8_utils.py` L208-L211）：

> Grouped quantized tensors cannot use `GroupedTensor.copy_` here because the generic grouped path can rebuild member tensors through `split_into_quantized_tensors()`, which is **not graph safe**. Update cached member tensors in place instead.

「graph safe」是关键词 —— 重建成员张量会破坏自动微分图。所以只能原地改缓存的成员。

普通张量只要一行（`fp8_utils.py` L224-L226）：

```python
# Plain TE quantized tensors override copy_ to requantize into their backing storage.
dst.copy_(src.view(dst.shape))
```

注释说明了为什么这里 `copy_` 语义是对的：**TE 的量化张量重载了 `copy_`，它内部会做重量化。**

批量版本 `copy_tensors_to_quantized_params`（[L229-L266](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L229-L266)）的 docstring 把动机说得很具体（`fp8_utils.py` L232-L235）：

> Same values, minus the per-param `copy_` and tensor-subclass dispatch: the quantizer is resolved up front and called directly. Cast kernels are unchanged, one per param. **Worth it because those casts are small and issuing them is expensive, and under `--reuse-grad-buf-for-mxfp8-param-ag` they run inside the forward pass.**

**这个 flag 让重量化跑在前向里**，所以省下的是 kernel launch 的开销，不是计算量。这类「优化的是发射开销不是算力」的取舍，在 fp8 路径上反复出现。

`modify_grouped_tensor_rowwise_storage`（[L269-L300](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L269-L300)）是「换掉高精度 grouped 张量的底层存储」，用于把参数塞进 DDP 的连续 buffer。它有一道关键守卫（`fp8_utils.py` L274-L278）：

```python
if is_grouped_tensor_with_quantized_storage(tensor):
    raise ValueError(
        "modify_grouped_tensor_rowwise_storage only supports high-precision GroupedTensor "
        "storage. Quantized grouped storage also owns scale buffers.")
```

**「量化存储还额外持有 scale buffer」** —— 一句话说明了量化张量比高精度张量多带了什么状态。替换完还要清干净（`fp8_utils.py` L296-L300）：

```python
new_storage.detach().copy_(old_rowwise_data)
tensor.rowwise_data = new_storage
tensor.columnwise_data = None
tensor.quantized_tensors = None
del old_rowwise_data
```

`columnwise_data` 与 `quantized_tensors` 被显式置 `None` —— 它们指向旧存储，不清就是悬挂引用。

NVFP4 有个几乎一样的函数（[`fp4_utils.py` L80-L97](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp4_utils.py#L80-L97)），差别在字段名与 dtype 断言：

```python
old_rowwise = getattr(fp4_tensor, "_rowwise_data", None)
...
assert old_rowwise.dtype == new_rowwise_data.dtype == torch.uint8, \
    "Rowwise NVFP4 storage must be uint8"
```

**FP8 用 `rowwise_data`（无下划线），FP4 用 `_rowwise_data`（有下划线）。** 两个库、两种命名，源码没有统一。

### 2.8 本节两张图

```mermaid
graphLR
  Q["fp32 主权重<br/>保持高精度常驻"]
  G["param gather<br/>低精度 all-gather"]
  R["restore<br/>转置视图或列式存储"]

  Q -->|"quantize_param_shard<br/>按 start_offsets 切片"| G
  G -->|"post_all_gather_processing"| R
  R -->|"forward 反向要用转置布局"| Q2["fp8 模型参数"]

  X1["flag 静默降级<br/>仅 FSDP2 下"]
  X2["uint8 作为量化标记<br/>rowwise_data.dtype"]
  X3["TE 类名随版本变<br/>Float8Tensor 到 QuantizedTensor"]
  X4["原地改成员才 graph safe"]
```

图注：数据/结论来源（逐条已核）—— flag 降级取自 [`arguments.py` L1023](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/training/arguments.py#L1023) 与 [L1043-L1056](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/training/arguments.py#L1043-L1056)；`uint8` 映射取自 [L1069-L1071](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/training/arguments.py#L1069-L1071) 与 [`fp8_utils.py` L144-L150](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L144-L150)；类名版本差异取自 [`fp8_utils.py` L122-L131](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L122-L131)；graph safe 取自 [L208-L211](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L208-L211)。图为自绘。

---

## 3. grad reduce 的精度

### 3.1 bf16 归约 + fp32 累加：只引 05 的结论

第 05 篇 §11 已经把这件事讲透了，这里不重复推导，只引用它的结论并说清它与本篇的关系。

bf16 只有 8 位尾数，跨 rank 做 reduce-scatter 时**每一次跨 rank 的加法都在 bf16 里做**，dp 越大参与累加的项越多，误差累积越严重。05 篇给出的解法是把归约拆成两步（[`reduce_scatter_with_fp32_accumulation.py`](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py#L52-L53)）：

| 步骤 | 操作 | 精度 |
| --- | --- | --- |
| 1 | all-to-all | 低精度（只搬运，不累加） |
| 2 | `torch.sum(dim=0, dtype=torch.float32)` | fp32 |
| 3 | 下 cast 回低精度 | 低精度 |

**注意第一步是 all-to-all 而不是 reduce-scatter**，这是 05 篇强调的技术点：每个元素只在本地被累加一次，且那次累加在 fp32 里做。

本篇要补的是**入口**：`args.main_grads_dtype` 经 `map_dtype` 归一化（[L1074](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/training/arguments.py#L1074)），可选 `fp32` / `bf16` / `fp16` / `fp8` / `auto`。也就是说「用 fp32 累加」这件事在配置层的落点是 `main_grads_dtype` 与那条 all-to-all 路径，而不是 clip_grads 那一侧。

### 3.2 `exp_avg_dtype` / `exp_avg_sq_dtype` 也在同一张表里

`map_dtype` 一共处理六个字段（`arguments.py` L1074-L1080），其中 `exp_avg_dtype` 与 `exp_avg_sq_dtype` 是优化器状态的两件。**这意味着优化器状态也可以降到 fp8/bf16** —— 与第 08 篇 §5 讲的 optimizer CPU offload 是两条独立的省显存路径：一条把状态搬到 host，一条把状态本身降精度。

`exp_avg_dtype` / `exp_avg_sq_dtype` 对应的是 AdamW 那两个状态张量（[PyTorch `AdamW` 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)）。`auto` 与 `None` 都映射到 `None`，语义是「交给下游按 recipe 自己定」。`dtype_map` 里 `'fp8'` 紧挨着 `'fp16'`，但映射目标是 `torch.uint8` —— 再一次提醒第 2.2 节那条：FP8 在 PyTorch 里是位模式，不是类型。

---

## 4. grad clip 与 loss scaling

契约 §2.1 把 `clip_grads.py` 与 `grad_scaler.py` 划给本篇，因为它们在第 05 篇写完时的提及次数是 1 和 **0**。下面两节各自独立成节，不依赖前面任何一节。

### 4.1 `clip_grads.py`：三层回退与一条硬断言

`clip_grads.py` 277 行，三个公开函数：`get_grad_norm_fp32`、`clip_grad_by_total_norm_fp32`、`count_zeros_fp32`。开篇 47 行是一组三层回退（`clip_grads.py` L10-L47）：

```python
try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier, multi_tensor_l2norm,
        multi_tensor_scale, multi_tensor_scale_tensor)
    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_impl = multi_tensor_scale
    multi_tensor_scale_tensor_impl = multi_tensor_scale_tensor
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier
        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
        multi_tensor_scale_tensor_impl = None
    except ImportError:
        import warnings
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of ...')
        from megatron.core.utils import (
            local_multi_tensor_applier, local_multi_tensor_l2_norm, local_multi_tensor_scale)
        multi_tensor_applier = local_multi_tensor_applier
        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale
        multi_tensor_scale_tensor_impl = None
```

**TE → Apex → Megatron 本地实现**，三级。第三级只 `warnings.warn` 就继续跑。而第三级里 `multi_tensor_scale_tensor_impl = None` 是**写死的 `None`**（`clip_grads.py` L47）—— 所以下面那处 if/else 的走向，在没有 TE 也没有 Apex 的环境里是被锁死的。

#### 范数怎么算

它自己声明改编自 `torch.nn.utils.clip_grad.clip_grad_norm_`（`clip_grads.py` L60-L63），范数的语义以 [`torch.linalg.vector_norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) 为准；同族的另一个变体 [`clip_grad_value_`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_value_.html) 裁的是逐元素绝对值而非总范数，本文件不用它。

`get_grad_norm_fp32`（[L55-L140](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/clip_grads.py#L55-L140)）分两条：`norm_type == inf` 走 `MAX` 归约（`clip_grads.py` L94-L105），其余走 `SUM` 归约（`clip_grads.py` L107-L134）。L2 那条是重点：

```python
total_norm = torch.zeros(1, dtype=torch.float, device='cuda')
if not grads_for_norm:
    pass
elif norm_type == 2.0:
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
    # Use apex's multi-tensor applier for efficiency reasons.
    # Multi-tensor applier takes a function and a list of list
    # and performs the operation on that list all in one kernel.
    grad_norm, _ = multi_tensor_applier(
        l2_norm_impl, dummy_overflow_buf, [grads_for_norm], False  # no per-parameter norm
    )
    # Since we will be summing across data parallel groups,
    # we need the pow(norm-type).
    total_norm = grad_norm**norm_type
```

第四个参数 `False` 的注释 *no per-parameter norm* 是关键：**只返回一个标量，不返回逐参数范数。** 这就是第 05 篇 §2 那个前提的直接后果 —— 梯度必须落在一块连续内存里，才能对整块做一次归约；如果梯度是散的，就得逐参数循环，kernel launch 次数与参数个数同阶。

L121 那句注释 *Since we will be summing across data parallel groups, we need the pow(norm-type)* 解释了一个容易看漏的动作：**先 `pow(norm_type)` 再跨 rank 求和**，而不是先求和再开根。归约的是平方和，不是范数本身 —— 这样跨 rank 加法才能用 `SUM`。

#### 两条 fallback 分支的差别不止是类型

L135-L138：

```python
if multi_tensor_scale_tensor_impl is not None:
    total_norm = total_norm.pow(1.0 / norm_type)
else:
    total_norm = total_norm.item() ** (1.0 / norm_type)
```

看起来只是「张量版 vs Python 版」。**实际上 `.item()` 会把 float32 提升成 Python float（双精度），所以后一条分支的开根是在 float64 里做的。** 实测两条分支的相对差是 `2.145e-08` 量级 —— 与 float32 的 eps 同阶，方向是**没有 `multi_tensor_scale_tensor` 的那条分支精度更高**，代价是一次 host 侧同步。

这也是一条**软降级**：装了 TE 的环境与没装的环境，clip 用的范数精度差一个 eps。源码没提这件事。

#### clip 系数与那条 fp32 硬断言

`clip_grad_by_total_norm_fp32`（[L143-L192](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/clip_grads.py#L143-L192)）先筛参数，再算系数。**L174 是一条硬断言：**

```python
if param.grad is not None:
    assert param.grad.type() == 'torch.cuda.FloatTensor'
    params.append(param)
    grads.append(to_local_if_dtensor(param.grad).detach())
```

**梯度必须是 fp32 CUDA 张量**，否则直接断言失败。而 `use_decoupled_grad=True` 的那条分支（`clip_grads.py` L167-L171）却宽松得多：

```python
if hasattr(param, "decoupled_grad") and param.decoupled_grad is not None:
    assert param.decoupled_grad.dtype in [torch.float32, torch.bfloat16]
```

decoupled 梯度允许 bf16，普通梯度只允许 fp32。**同一行代码里两条分支的精度要求不一致**，源码没解释原因。

系数与施加（`clip_grads.py` L179-L192）：

```python
clip_coeff = max_norm / (total_norm + 1.0e-6)
dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
if isinstance(clip_coeff, torch.Tensor):
    clip_coeff.clamp_max_(1.0)
    assert multi_tensor_scale_tensor_impl is not None, \
        "clip_coeff is tensor type. But multi_tensor_scale_tensor not available."
    multi_tensor_applier(multi_tensor_scale_tensor_impl, dummy_overflow_buf,
                         [grads, grads], clip_coeff)
elif clip_coeff < 1.0:
    multi_tensor_applier(multi_tensor_scale_impl, dummy_overflow_buf,
                         [grads, grads], clip_coeff)
```

`+ 1.0e-6` 是除零保护。两条路的**判定时机不同**：张量路径先 `clamp_max_(1.0)` 再施加；浮点路径只在 `< 1.0` 时才施加。效果一致，但要注意 `total_norm` 的类型决定了走哪条 —— 而 `total_norm` 的类型正是 §4.1 那个 if/else 决定的。**两处耦合在一起。**

#### `count_zeros_fp32`：与 04 篇的接口

第三个函数数零值，用于 loss scale 自适应（见 §4.2）。它的参数过滤链（`clip_grads.py` L234-L255）里出现了三个谓词：

```python
is_not_shared = param_is_not_shared(param)
is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(
    param, tp_group=tp_group, expert_tp_group=expert_tp_group)
is_not_gtp_duplicate = param_is_not_gtp_duplicate(param)
if grad_not_none and is_not_shared and is_not_tp_duplicate and is_not_gtp_duplicate:
```

`param_is_not_gtp_duplicate` 来自第 04 篇讲的 GTP —— **通用张量并行把同一份权重在多个 rank 上复制，统计梯度统计量时不能重复计数。** 一个「优化器侧的梯度统计」函数，判重逻辑要跨到并行层去，是这一串代码最值得记的地方。

还有一条针对 Megatron-FSDP 的特判（`clip_grads.py` L237-L244）：

```python
if getattr(param, "__fsdp_param__", False) and grad_not_none:
    # If the parameter is managed by Megatron FSDP, the gradient is
    # an FSDP-sharded DTensor, and we should use the local shard.
    use_megatron_fsdp = True
    grad = getattr(param, grad_attr)._local_tensor
```

以及随之而来的一个断言（`clip_grads.py` L257-L263）——FSDP 与 DP 组同时存在时直接报错：

```python
if use_megatron_fsdp and data_parallel_group is not None:
    raise ValueError(
        "Unexpected use of Megatron FSDP with data parallel group. "
        "Please ensure that the parameters are properly managed by Megatron FSDP.")
```

#### 本节实验：整块与逐循环差几个 ULP

配套脚本 `grad_clip_equivalence.py` 验的就是上面这些。float64、12 个 `(256, 512)` 的梯度：

```text
  整块       1254.6424425718765
  逐参数     1254.6424425718762
  是否逐位相同：False
  ULP 差：1
```

**不是逐位相同。** 数学上恒等（$$\sqrt{\textstyle\sum_i \lVert g_i \rVert^2} = \lVert g \rVert$$），浮点上差 1 ULP，来源是求和结合序不同：整块是一趟长线性求和，逐参数是 12 段局部求和再合并。float32 下相对差 `1.069e-08`。

顺带验掉了一个我原本写错的说法：`norm_type=2` 时 `n ** 2` 与 `n * n` 在 IEEE 下是同一条指令序列，所以 L121 的 `pow` 位置**不引入额外差异**，真正带来差异的只有分段 vs 单趟求和。

### 4.2 `grad_scaler.py`：fp16 loss scaling 在 bf16 时代的地位

`grad_scaler.py` 165 行，三个类：一个抽象基类与两个实现。

抽象基类 [`MegatronGradScaler` L11-L40](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/grad_scaler.py#L11-L40) 只有一件不平凡的事 —— scale 是个 GPU 上的 float32 张量：

```python
def __init__(self, initial_scale: float):
    """Initialize scale value with the input initial scale."""
    assert initial_scale > 0.0
    self._scale = torch.tensor([initial_scale], dtype=torch.float, device='cuda')
```

`inv_scale` 用了一个容易被忽略的写法（`grad_scaler.py` L26-L28）：

```python
@property
def inv_scale(self):
    return self._scale.double().reciprocal().float()
```

**先升到 float64 求倒数，再降回 float32。** 直接在 float32 里求倒数会损失精度（除法的舍入），先升再降能让最终 float32 结果更接近真实倒数。这一步与 §4.1 那个 `.item() ** (1/norm_type)` 是同一个思路 —— 代码里多处在自觉规避低精度中间量。

#### `ConstantGradScaler`：一个空 state_dict

`ConstantGradScaler`（[L43-L58](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/grad_scaler.py#L43-L58)）的语义是「永不调整」：

```python
class ConstantGradScaler(MegatronGradScaler):
    """
    Grad scaler with a fixed scale factor.
    The loss scale is never adjusted, regardless of whether NaNs or Infs
    are detected in the gradients.
    """

    def update(self, found_inf: bool):
        pass

    def state_dict(self):
        return dict()
```

`update` 是 `pass`，`state_dict` 返回**空字典**。**scale 不进 checkpoint** —— 对固定 scaler 这是对的（值由配置决定，不需要存），但如果 checkpoint 里没有它、恢复时又用了不同的 `initial_scale`，训练会静默地从另一个 scale 起步。

#### `DynamicGradScaler`：hysteresis 不是「连续 inf 计数」

动态版的语义在 docstring 里写得很清楚（[L64-L69](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/optimizer/grad_scaler.py#L64-L69)）：

> It reduces the loss scale by a `backoff_factor` if a `hysteresis` number of NaNs/Infs are detected in **consecutive** iterations. Conversely, it increases the loss scale by a `growth_factor` if no non-finite gradients are seen for a specified `growth_interval` of iterations.

构造函数里有六个断言（`grad_scaler.py` L109-L125），其中 `backoff_factor` 上下界都要查：

```python
assert backoff_factor < 1.0
assert backoff_factor > 0.0
```

**两个 tracker 的初值不对称**（`grad_scaler.py` L128-L129）：

```python
self._growth_tracker = 0
self._hysteresis_tracker = self.hysteresis
```

`update` 的两个分支（`grad_scaler.py` L131-L153）：

```python
if found_inf:
    self._growth_tracker = 0
    self._hysteresis_tracker -= 1
    # Now if we are out of hysteresis count, scale down the loss.
    if self._hysteresis_tracker <= 0:
        self._scale = torch.max(self._scale * self.backoff_factor, self.min_scale)
else:
    # If there is no nan/inf, increment the growth tracker.
    self._growth_tracker += 1
    # If we have had enough consequitive intervals with no nan/inf:
    if self._growth_tracker == self.growth_interval:
        # Reset the tracker and hysteresis trackers,
        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis
        # and scale up the loss scale.
        self._scale = self._scale * self.growth_factor
```

**「consecutive」这个措辞与实现有一处偏差，值得单独指出。** 干净的迭代（`found_inf=False`）只增加 `_growth_tracker`，**完全不碰 `_hysteresis_tracker`**。后者只在两个地方被重置：

- 构造时（`grad_scaler.py` L129）
- 成功涨 scale 时（`grad_scaler.py` L151）

所以：`inf → 干净 → inf` 这样两个**不连续**的 inf，会让 `_hysteresis_tracker` 减两次并触发一次退避。**`hysteresis` 实际统计的是「距上次成功涨 scale 以来累计的 inf 次数」，而不是「连续 inf 次数」。** 只有当 `growth_interval` 足够小、干净迭代足够频繁时（涨了 scale 就重置），两者才近似等价。

另一处细节：L148 用的是 `== self.growth_interval` 而非 `>=`。由于 L146 每次只 `+1`，从 0 起步单调递增，`==` 能正常命中；但一旦 `growth_interval` 被改成 0（虽然 L120 断言了 `> 0`），或 tracker 因别处改动而跳值，就会永远命不中。

这套 fp16 loss scaling 的原始出处是 *Mixed Precision Training*（[arXiv:1710.03740](https://arxiv.org/abs/1710.03740)），PyTorch 侧的对应实现见 [`torch.cuda.amp.GradScaler`](https://docs.pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler) —— Megatron 这个类是把它按「hysteresis + growth_interval」两个参数重新表达了一遍。

**这个模块在 bf16 时代的位置**：bf16 的指数位与 fp32 相同（8 位），动态范围极宽，训练时几乎不会出现 fp16 那种梯度下溢。所以 fp16 loss scaling 这套机制**在 bf16 路径上基本不启用**，它的存在主要是为了兼容仍走 fp16 的训练与某些 MoE / 路由相关的数值稳定需求。本机没有 GPU，也没有走 fp16 的实测，**「bf16 下 scaler 是否被真正启用」这一点本篇待验证**。

### 4.3 两者的接口：found_inf 从哪来

§4.1 的 `count_zeros_fp32` 与 §4.2 的 `update(found_inf)` 是配对的：前者数零值/非有限值，后者据此调 scale。中间那一环在 Megatron 主训练循环里，不在本篇的两个文件里。**本篇只把两端的语义钉死，中间链路标为待验证。**

---

## 5. NVFP4 / MXFP4 到哪一步

### 5.1 NVFP4 的配置面

[`transformer_config.py` L672](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L672) 只有两个取值：

```python
fp4_recipe: Optional[Literal['nvfp4', 'custom']] = "nvfp4"
```

`nvfp4` 的对齐尺寸由 [`fp4_utils.py` L176-L205](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp4_utils.py#L176-L205) 给出。**这个函数的 docstring 推导 32，函数体却 `return 128`**，是本篇最值得单独讲的一处：

```python
def get_fp4_align_size(fp4_recipe: Fp4Recipe) -> int:
    """
    Get the alignment size required for FP4 GEMM.
    FP4 GEMM requires Blackwell and later architectures.

    The value 32 is a hardware requirement: TMA (Tensor Memory Accelerator) requires
    a 16-byte aligned address for efficient memory access. Since FP4 uses 4 bits per value,
    16 bytes (128 bits) corresponds to 32 FP4 values. Therefore, the alignment size for FP4
    is 32. With this alignment, NVFP4 GEMM can be performed efficiently.

    Note that since we are also random hadamard transform for NVFP4 training, we want
    fused group nvfp4 quantize plus hadamard transform. Hadamard transform will leverage
    tensor core instructions for better performance, while group quantize kernels also
    prefer a more aligned size in token dimension M. The efficiently leverage grouped
    kernels, padding needs to be 64 multiple, but 128 multiple will bring even faster.
    ...
    Paper link: https://arxiv.org/pdf/2509.25149
    Scaling factor layout: https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
    TE NVFP4 Grouped Quantization: https://github.com/NVIDIA/TransformerEngine/pull/2411
    """
    # pylint: disable=unused-argument
    return 128
```

三层理由，一层比一层具体：

| 层 | 约束 | 值 |
| --- | --- | --- |
| 硬件 | TMA 要求 16 字节对齐；FP4 每值 4 bit → 16 B = 32 个值 | 32 |
| kernel | 融合 Hadamard + 分组量化 kernel 偏好 token 维对齐，padding 需 64 的倍数 | 64 |
| 实际取 | 128 的倍数更快 | **128** |

**注意 docstring 第一段的收尾是 *Therefore, the alignment size for FP4 is 32.*，然后函数返回 128。** 单看会以为是笔误。三层理由读下来就知道不是 —— 32 是硬件下限，128 是考虑了融合 kernel 与 padding 后的选择。

同一段 docstring 还挂着三条出处：**NVFP4 预训练论文**（[arXiv:2509.25149](https://arxiv.org/abs/2509.25149)，*Pretraining Large Language Models with NVFP4*，即 `fp4_utils.py` 里那条 `Paper link`）、**cuBLAS 的 block scaling factor 布局**（[官方文档](https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout)）、以及 TE 的 NVFP4 grouped 量化 PR。三条链接里，论文与 cuBLAS 文档本次已核可达。

docstring 里还有一段专门讲 MoE + CUDA graph 的冲突（`fp4_utils.py` L192-L198）：

> When it comes to MOE cuda graph support, the number of tokens for each expert should be a buffer on device memory, which means that we don't know the token dimension for each expert in host, therefore we cannot calculate the zero padded scaling factors shape on host to comply with the NVFP4 GEMM scaling factor layout. However, **if we have already zero padded the tokens to 128 multiple, then there is no need for such padding**, so that host doesn't need to copy the token distribution from device to host (which will break the CUDA graph).

**这一段解释了为什么取 128 而不是 64**：如果已经按 128 对齐了 token 数，host 端就不需要为了凑 scaling factor 布局而把 token 分布从 device 拷回 host —— 那次拷贝会打断 CUDA graph。**对齐尺寸在这里是被 graph 捕获的约束倒推出来的。**

### 5.2 哪些层不能降：bf16 边界与 delayed 的冲突

配置层给了两个「留 bf16」的开关：`first_last_layers_bf16`（[L642](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L642)）与配套的 `num_layers_at_start_in_bf16` / `num_layers_at_end_in_bf16`。判定函数在 [`fp8_utils.py` L711-L727](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L711-L727)，带一条值得记的注释：

```python
# Since layer_no is a global layer index, additional checks on whether
# we are in the first or last pipeline-parallel rank are not needed.
is_first_layer = layer_no < num_bf16_layers_at_start
is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end
```

**用全局层号判定，所以不需要再判断自己在不在首/末 PP rank。** 这个简化成立的前提是层号在所有 PP rank 上连续且偏移已知。

**真正的不兼容在这里**（[L883-L888](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L883-L888)）：

```python
# First / last layer in bf16 isn't supported with delayed scaling since it
# requires entering/exiting fp8 context per layer, causing incorrect amax
# reduction behavior.
assert not (
    config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
), "Delayed scaling does not support first / last layer in BF16."
```

**理由不是精度，是 amax 归约的正确性。** `first_last_layers_bf16` 的实现方式是逐层开关 fp8 上下文；而 `delayed` 的 amax 是跨 layer 归约的历史量，逐层进出上下文会让归约的分组与实际量化的分组对不上，于是 amax 失真。

这个限制与第 8 章那张互斥表是同一类东西 —— **不是「没实现」，是「机制上会错」**。另外三条限制各有出处：

| 限制 | 出处 |
| --- | --- |
| `fp8_output_proj` 必须配 `mxfp8` | [`transformer_config.py` L1551-L1554](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1551-L1554) |
| `fp8_recipe == custom` 必须给 `fp8_quantizer_factory` | [L1540-L1543](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L1540-L1543) |
| `first_last_layers_bf16` + `delayed` | [`fp8_utils.py` L886-L888](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L886-L888) |

`fp8_output_proj` 的判定函数 [`is_mxfp8_output_proj_active` L730-L745](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L730-L745) 写得很有防御性 —— 它不假设 `fp8_recipe` 是枚举还是字符串：

```python
fp8_recipe = getattr(config, "fp8_recipe", None)
recipe_value = getattr(fp8_recipe, "value", fp8_recipe)
return str(recipe_value).lower() == "mxfp8" or str(fp8_recipe).lower().endswith(".mxfp8")
```

`getattr(x, "value", x)` 这个写法是**同时兼容枚举与字符串**：枚举有 `.value`（其 `.name` 是 `"mxfp8"`），字符串则原样返回。

### 5.3 MXFP8 与 blockwise 的实际参数

这两个 recipe 的参数在 TE 侧。官方文档给了确切的数：

**MXFP8**（[Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)）—— block 32、scale 用 E8M0（8 位指数）、元素是 FP8。规格表里列了四个 MX 变体：

| 格式 | block | scale | 元素 |
| --- | --- | --- | --- |
| MXFP8 | 32 | E8M0 | FP8（E4M3 / E5M2） |
| MXFP6 | 32 | E8M0 | FP6（E2M3 / E3M2） |
| MXFP4 | 32 | E8M0 | FP4（E2M1） |
| MXINT8 | 32 | E8M0 | INT8 |

**`blockwise`（TE 的 `Float8BlockScaling`）的 block 是 128，不是 32**（[FP8 Blockwise Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html)）：*Block size is 128. Blocks can be one dimensional – containing 128 consecutive values, two dimensional – containing tiles of 128×128 values.* 并且默认维度不同：激活用 1D、权重用 2D、梯度用 1D。

同一个页面还给出了 scale 的存储方式，对理解第 1.6 节那条修补有用：

> Scaling factors are stored as 32-bit floating point numbers. By default, they are constrained to powers of 2 (utilizing the 8 exponent bits of FP32). On Hopper, this constraint can be relaxed by setting the environment variable `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`. On Blackwell, only powers of 2 are supported.

以及块 scale 的四步算法：

> 1. Find the maximum absolute value (`amax_block`) across all elements in the block. 2. Calculate `s_block = max_fp8 / amax_block`, where `max_fp8` is the maximum representable value in the FP8 format (**448 for E4M3, 57344 for E5M2**). 3. If the power-of-2 constraint is enabled, round down to the nearest power of 2 by zeroing out the mantissa bits, retaining only the sign and exponent. 4. Multiply each element in the block by `s_block` before converting to FP8.

**`448` 与 `57344` 这两个数与本篇实验独立枚举出来的格点最大值完全一致**（见 §5.5）。第 3 步「清掉尾数只留符号与指数」就是 E8M0 编码在 fp32 容器里的实现 —— 与 MX 格式用 E8M0 是同一件事。

### 5.4 能训到哪一步：只陈述源码给出的边界

本机无 GPU，**任何「FP8 训练收敛如何」的数字都是 D 级证据，不写**。能陈述的只有源码与官方文档给出的边界：

- TE 文档对 `blockwise` 的定位是 *inspired by the quantization scheme used to train the DeepSeek-v3 model*（[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)，§3.3 FP8 Training 把这套做法写成 *tile-wise grouping with 1×$$N_c$$ elements or block-wise grouping with $$N_c$$×$$N_c$$ elements*），并说明 *Unlike FP8 Current/Delayed Scaling, E4M3 is used by default for both forward and backward passes. Tensor-scaled recipes used E5M2 for gradients due to its higher dynamic range, but with multiple scaling factors per tensor the dynamic range requirement is lowered, so E4M3 is usually sufficient.* 以及 *Pure E5M2 training is not supported.*
- 同一页面说明 Blackwell 上 `blockwise` 会被模拟成 MXFP8，且 *MXFP8 is the preferred recipe on Blackwell*。
- NVFP4 侧有两条可核的出处：*Pretraining Large Language Models with NVFP4*（[arXiv:2509.25149](https://arxiv.org/abs/2509.25149)）与 FP8-LM（[arXiv:2310.18313](https://arxiv.org/abs/2310.18313)）。
- **一个可引用的量级**：DeepSeek-V3 报告与 BF16 基线相比，*the relative loss error of our FP8-training model remains consistently below **0.25%**, a level well within the acceptable range of training randomness*（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §3.3）。这是**该论文自己报告的数字**，对应它那套细粒度分组 + 提高精度累加的配置；换配置不保证同一量级。

**精度损失量级本篇不给数字。** 手撸量化器能量化「量化本身引入多少误差」，但从量化误差到「训练后精度掉多少」之间隔着优化动力学，本机无法闭合。**待验证。**

### 5.5 本节实验：粒度买到的是量程，不是精度

配套脚本 `quantizers.py` 手撸了四类量化器：E4M3 / E5M2 按 OCP 逐位枚举格点，MX block scaling 用 E8M0（2 的幂）scale，INT8 分 per-tensor / per-channel / per-block 三档。

先说格点自检 —— 这两个数与 §5.3 引的 TE 文档一致：

```text
【0】FP8 格点集合自检（逐位枚举，不是套公式）
  E4M3FN: 有限可表示值 253 个，最大 448，最小正规格值 0.00195312
  E5M2: 有限可表示值 247 个，最大 57344，最小正规格值 1.52588e-05

【0b】指数全 1 那一行的处理差异（枚举错就会多出这个值）
  若把 E5M2 也当「有限-非数」变体枚举，最大值会算成 98304 —— 该值不存在。
```

**E4M3FN 与 E5M2 在指数全 1 那一行上处理不同**，枚举时必须分开：E4M3FN 没有 Inf（指数全 1 且尾数全 1 才是 NaN，其余是正常数，所以最大值 448）；E5M2 指数全 1 整行是 Inf/NaN（最大有限值只能到 57344）。这个区别与 OCP MX 规范里那张 FP8 编码表一致 —— 该表明确 E4M3 的 *Infinities* 一栏是 *N/A*。

枚举出的规格值与官方文档交叉验证通过：

| 量 | 本脚本枚举 | 官方文档 |
| --- | --- | --- |
| E4M3 最大有限 | 448 | 448（[TE](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)） |
| E5M2 最大有限 | 57344 | 57344（同上） |
| E4M3 最小次正规 | $$2^{-9}$$ | $$2^{-9}$$（[OCP MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)） |
| E4M3 最小规格化 | $$2^{-6}$$ | $$2^{-6}$$（同上） |

#### 粒度曲线是平的

```text
【3】MX 块大小的粒度曲线（数据 B，E4M3）—— 曲线是平的，原因见下
     块大小         RMSE    SNR(dB)     scale 字节
----------------------------------------------
       1 1.945893e-02      31.30       4096 B （逐元素，实际不这么用）
       8 1.945893e-02      31.30        512 B （E8M0，1 B/块）
      32 1.945893e-02      31.30        128 B （E8M0，1 B/块）
     128 1.945893e-02      31.30         32 B （E8M0，1 B/块）
     512 1.945893e-02      31.30          8 B （E8M0，1 B/块）
  per-tensor 参照：RMSE 1.889401e-02  SNR 31.56 dB  scale 4 B
```

**块大小从 1 到 512，RMSE 完全一样。** 这不是 bug，是浮点格式的性质：只要一组内的值没有跨出规格化区，缩小 scale 并不增加尾数位，相对精度恒在 $$2^{-3}$$ 量级。E4M3 的规格化区覆盖 $$448 / 2^{-6} \approx 2^{14.8}$$ 的动态范围 —— 跨度不超过这个比值，缩放就换不来精度。

#### 粒度真正起作用的场景

换一组「块内混着极小值」的数据，结论立刻反过来：

```text
【2b】小幅值元素的存活情况（数据 D，阈值 |x| < 1e-3）
方案                              被清零比例         相对误差
--------------------------------------------------
FP8 E4M3 per-tensor            60.9%       0.7751
FP8 E4M3 MX-block32            47.6%       0.6416
FP8 E4M3 MX-block8              0.0%       0.0225
FP8 E4M3 逐元素                    0.0%       0.0225
INT8 per-tensor               100.0%       1.0000
```

**per-tensor 把 60.9% 的极小值直接清零。** 因为那块 scale 是被同一块里的大值顶高的，极小值除以 scale 后落到 E4M3 的规格化区以下，被舍成 0。MX-block32 只好一些（47.6%）—— 因为 32 个元素的块里仍然混着大值。**只有把粒度降到 8 或 1 才救得回来。**

**这张表说明 block scaling 解决的是量程问题，不是精度问题。** 它的价值在于让每个块各自用满 FP8 的量程，从而让组内极小值不至于被大值挤进次正规区。

这个结论不是本脚本的孤证。DeepSeek-V3 技术报告（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）把低精度训练的瓶颈归到 *the presence of outliers in activations, weights, and gradients*，而它给出的解法明确是为了 *effectively **extend the dynamic range** of the FP8 format* —— **扩动态范围，不是提精度。** 独立团队在大规模训练上得到的是同一个判断。

#### FP8 与 INT8 对 outlier 的反应相反

```text
【1】粒度对误差的影响（数据 B：量级差在**块之间**，每块内部一致）
  FP8 E4M3  per-tensor RMSE 1.7971e-02 → MX-block32 1.9020e-02   （降 -5.8%）
  INT8      per-tensor RMSE 9.1083e-03 → per-block32 3.7865e-03   （降 58.4%）
```

- **INT8**：per-tensor 到 per-block32，误差降 58.4%。均匀量化的步长与 scale 成正比，粒度一变步长就变，收益直接。
- **FP8**：per-tensor 到 MX-block32，误差反而**升** 5.8%。因为这组数据的量级差在块之间，MX 给每个块独立 scale 后，那些大值块的 E8M0 scale 取整到 2 的幂时可能比 per-tensor 的浮点 scale 更粗。**浮点 scale 的取整方向是「不溢出」，MX 的取整方向是「必须是 2 的幂」，后者更受限。**

这个对比是本篇最实用的一条结论：**同一笔「粒度换误差」的账，INT8 划算，FP8 不划算。** 选 FP8 的理由在别处 —— 下一段。

「outlier 让量化变难」这件事在推理侧被研究得更透：AWQ（[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)）的核心观察是激活里只占极小比例的**离群通道**却主导了量化误差，并据此做激活感知的权重量化；QLoRA（[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)）则把离群值单独放进更高精度路径来保护。**两条路都是「把离群值从共享 scale 里摘出来」** —— 与本节实验里「per-tensor 被离群值顶高 scale、MX 把离群值圈进自己的块」是同一个思路的两个方向。PyTorch 侧的量化总览见 [量化文档](https://docs.pytorch.org/docs/stable/quantization.html)。

#### 一条度量上的坑：SNR 不能跨数据集比

脚本在数据 C（注入 outlier）上得到：

```text
【2】FP8 与 INT8 对 outlier 的反应完全不同（数据 C）
  FP8 E4M3 per-tensor    SNR 从  31.64 降到  32.19 dB   （掉 -0.55 dB）
  INT8 per-tensor        SNR 从  41.88 降到  31.85 dB   （掉 10.03 dB）
```

**FP8 那一行的 SNR 反而升了。** 这不是量化变好，而是分母变了：outlier 把信号能量 $$\sum x^2$$ 抬高很多，SNR 的分子跟着涨，超过了误差的增长。

**跨数据集比精度只能用 RMSE，或针对某个子集（如极小值）算相对误差，不能用 SNR。** 本篇所有结论都按这个口径写。同理，**极值相对偏差这一列在本组数据里全是 `0.000000e+00`** —— 因为 scale 就是按 amax 算的，最大值必然被精确还原。这一列只在极小值那组（阈值过滤后的子集）上才有信息量。

#### 粒度与归约开销的对应

把实验结果和 §1.4 的归约组对上：

| recipe | scale 粒度 | scale 类型 | 需要跨 rank 归约 |
| --- | --- | --- | --- |
| `tensorwise` | 整张量 | fp32 | **需要** |
| `delayed` | 整张量 + 历史窗口 | fp32 | **需要** |
| `blockwise` | 每 128 个（可 2D） | fp32，默认为 2 的幂 | 不需要 |
| `mxfp8` | 每 32 个 | E8M0 | 不需要 |

**per-tensor 的 scale 必须先归约才能用**，因为各 rank 的 amax 不同，取谁的值都会丢掉一部分张量的量程。块粒度不需要 —— 块内共享、块间独立，归约没有对象。这就是 `get_amax_reduction_group` 只对前两者有意义的结构原因。

### 5.6 本节两张图

```mermaid
graphLR
  Q["量化器四类<br/>E4M3 E5M2 MX-block INT8"]
  F["浮点 vs 均匀<br/>性质不同"]
  R1["FP8 粒度换量程<br/>相对精度不变"]
  R2["INT8 粒度换步长<br/>误差直接降"]
  L["极小值存活率<br/>被清零的比例"]
  A["归约结构分界<br/>per-tensor 需归约 block 不需要"]

  Q --> F
  F --> R1
  F --> R2
  R1 --> L
  R2 --> L
  R1 --> A
```

图注：数据来源 —— 全部来自配套脚本 `quantizers.py` 的固化输出 `results/stdout.txt`（§5.5 四段：格点自检、粒度曲线、极小值存活率、outlier 对比）。格点规格值与 [OCP MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 及 [TE FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) 交叉核对。图为自绘，未复制任何官方或论文图。

---

## 6. TE 集成层：运行期动态解析

### 6.1 3678 行里与低精度相关的部分

`transformer_engine.py` 是本系列最长的单文件之一。低精度相关的符号按位置分布：

| 位置 | 符号 | 作用 |
| --- | --- | --- |
| L29 | `get_amax_reduction_group`（导入） | 从 `parallel_state` 导入 |
| L190 | `if instance.fp8_quantization_recipe == Fp8Recipe.delayed` | delayed 特判 |
| L257 / L303 | `_get_fp8_model_init_for_quant_recipe` / `_for_quant_params` | 建模期 fp8 上下文 |
| L312-L360 | `_get_fp8_autocast_for_quant_recipe` | **前向 recipe 分派表**（§1.3） |
| L363-L369 | `_get_fp8_autocast_for_quant_params` | 按训练/推理选 recipe |
| L372-L380 | `_get_should_context_be_quantized_recipe` | 覆盖判定 |
| L2561-L2563 / L2743-L2745 | `delayed` 才有的全局 fp8 meta | 注释说不用 `recipe.delayed()` 是因为 TE 2.0 才有 |
| L3295-L3328 | `TEDelayedScaling` | delayed recipe 壳 |
| L3366-L3390 | `te_checkpoint` | 08 篇 §2.4 的那条 |

两处 `delayed` 特判的注释值得记（`transformer_engine.py` L2561-L2563）：

```python
# Only delayed scaling has global fp8 meta tensors. We're not using
# self.fp8_meta["recipe"].delayed() because it's available in TE 2.0 and later.
if isinstance(self.fp8_meta["recipe"], te.common.recipe.DelayedScaling):
```

**用 `isinstance` 而不是 recipe 自带的 `.delayed()` 谓词**，理由是后者的存在性有版本门槛。同一判断在 `fp8_utils.py` L886-L888 又出现一次。三处独立地写成 `isinstance(..., TEDelayedScaling)` / `isinstance(..., DelayedScaling)` —— 一致，但也说明这层判断被抄了三遍。

### 6.2 `_resolve_callable_from_python_import_path`：把一个字符串变成可调用对象

`fp8_recipe == custom` 时，用户给的是一个**字符串导入路径**，运行时才解析成工厂函数。解析器在 [`fp8_utils.py` L311-L342](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L311-L342)：

```python
def _resolve_callable_from_python_import_path(dotted_path: str):
    """Resolve a Python import path like 'pkg.mod.func' to a callable.

    Raises ValueError with clear message on failure.
    """
    if not isinstance(dotted_path, str) or not dotted_path:
        raise ValueError(
            "fp8_quantizer_factory must be a non-empty string with format 'pkg.mod.func'.")

    parts = dotted_path.rsplit(".", 1)
    if len(parts) == 1:
        raise ValueError(f"Invalid fp8_quantizer_factory '{dotted_path}'. Expected 'pkg.mod.func'.")
    module_path, attr = parts[0], parts[1]

    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to import module '{module_path}' for fp8_quantizer_factory: {exc}"
        ) from exc

    fn = getattr(mod, attr, None)
    if fn is None:
        raise ValueError(...)
    if not callable(fn):
        raise ValueError(...)
    return fn
```

四处校验、每处一条带上下文的 `ValueError`：**非空字符串** → **至少一个点** → **模块能导入**（异常被包成 `ValueError` 并 `from exc` 保留链）→ **属性存在** → **可调用**。

这个设计本身是好的 —— 把「配置里写错一个字符串」变成启动期的一次清晰报错，而不是训练中途某个模块找不到工厂。

**但它也是一条运行期依赖。** 配置里的字符串 `transformer_engine.common.recipe.CustomRecipe` 要能被导入，TE 的版本必须提供那个符号。`_get_custom_recipe`（[L345-L353](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L345-L353)）把这条依赖写成了显式报错：

```python
def _get_custom_recipe(quantizer_factory_python_path: str) -> Union[Fp8Recipe, Fp4Recipe]:
    quantizer_factory = _resolve_callable_from_python_import_path(quantizer_factory_python_path)
    try:
        custom_recipe = transformer_engine.common.recipe.CustomRecipe(qfactory=quantizer_factory)
    except AttributeError:
        raise ValueError("""CustomRecipe recipe is not available in this version of 
            Transformer Engine. Please make sure you are using TE version 
            >= 2.9.0.dev0.""")
    return custom_recipe
```

`AttributeError` → `ValueError`，并把门槛写成 2.9.0.dev0。**这条门槛比 §1.2 那张表里的任何一格都高**，而配置层的 docstring 只说 custom 需要 `fp8_quantizer_factory`（[`transformer_config.py` L604](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/transformer/transformer_config.py#L604)），没提这个版本要求。**配置层与实现层的版本信息不同步。**

### 6.3 `inspect.signature`：对 TE 的 API 做运行期探测

本篇认为最值得记的一处，在 [`fp8_utils.py` L867-L881](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L867-L881)：

```python
if not is_init:
    fp8_context = transformer_engine.pytorch.fp8_autocast(
        enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
    )
else:
    import inspect

    context_args = {"enabled": True}
    # Check if fp8_model_init supports setting recipe
    if "recipe" in (
        inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
    ):
        context_args["recipe"] = fp8_recipe
    # Check if fp8_model_init supports preserve_high_precision_init_val
    if "preserve_high_precision_init_val" in (
        inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
    ):
        context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()
    fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)
```

**用 `inspect.signature` 在运行时问 TE 的函数「你有哪些参数」，据此决定传不传。** 两个探测点：

- `recipe` —— 早期版本的 `fp8_model_init` 不收 recipe
- `preserve_high_precision_init_val` —— 后加的参数

`preserve_high_precision_init_val=torch.is_grad_enabled()` 这一行尤其微妙：**建模期是否保留高精度初值，取决于当时是否开梯度。** 这是在 `no_grad` 下做初始化时防止初值被量化掉。

**为什么用探测而不是版本号判断？** 因为 `is_te_min_version` 只能回答「这个版本有没有」，答不了「这个具体构建的签名是什么」。包被重打、特性回移、本地补丁，都会让版本号与签名脱节。探测函数签名是唯一直接的办法。

代价也清楚：**每次进入建模路径都要做一次 `inspect.signature`。** `import inspect` 放在函数体内（`fp8_utils.py` L868）说明这是按需路径而非热路径 —— 建模确实不是每步都走。

### 6.4 依赖形态小结

把 §2.3、§6.2、§6.3 里的依赖放在一起：

| 依赖对象 | 形式 | 出处 |
| --- | --- | --- |
| `transformer_engine.pytorch.tensor.QuantizedTensor` | 版本分叉的类导入 | [`fp8_utils.py` L412](https://github.com/NVIDIA/Megatron-LM/blob/60e039626/megatron/core/fp8_utils.py#L412) |
| `transformer_engine.pytorch.tensor.utils.replace_raw_data` | **内部模块路径**导入 | L417 |
| `transformer_engine.common.recipe.CustomRecipe` | 字符串 → 运行期解析 | L348 |
| `fp8_model_init` 的参数 | `inspect.signature` 探测 | L872-L879 |
| `te.common.recipe.DelayedScaling` | `isinstance` 判断（3 处） | `te.py` L190 / L2563 / L2745 |
| 张量子类名 | 版本相关的模块级常量 | L122-L131 的 docstring |

**六种形式，没有一种是版本稳定 API。** 这与第 04 篇「GTP 必须依赖 TE」是同一条线索 —— Megatron 与 TE 的耦合不是「用了 TE 的公开接口」，而是「读了 TE 的内部结构、探了它的函数签名」。TE 一次内部重构就可能让这条链断掉，而 §2.6 那几十个 `is_*tensor` 谓词正是为此布下的防御层。

**代价是这些防御层本身的维护成本。** 1056 行的 `fp8_utils.py`，开头 190 行几乎全在处理「怎么认出这个张量」和「这个 API 在哪个版本有」。

---

## 7. 引用关系网图

```mermaid
graphLR
  CLAIM["低精度全链路的形状<br/>由 recipe 取值与 TE 版本共同决定"]

  A1["五个 recipe 不在同一层<br/>delayed 有状态其余无状态"]
  A2["flag 静默降级<br/>仅 FSDP2 下"]
  A3["uint8 作为量化标记<br/>类名随版本变"]
  A4["粒度换量程不换精度<br/>极小值存活率"]
  A5["clip 整块与逐循环差 1 ULP"]
  A6["hysteresis 统计累计非连续 inf"]
  A7["运行期动态解析<br/>inspect.signature"]

  L1["arXiv 2310.10537 MX 格式"]
  L2["arXiv 2509.25149 NVFP4 预训练"]
  L3["arXiv 2209.05433 FP8 格式"]
  L4["arXiv 2211.10438 SmoothQuant"]
  L5["arXiv 2208.07339 LLM.int8"]
  D1["TE FP8 Current Scaling"]
  D2["TE FP8 Blockwise Scaling"]
  D3["OCP MX 规范"]
  D4["cuBLAS scaling factor 布局"]
  M1["Megatron-Core 用户指南"]
  M2["Megatron-Core API 指南"]
  P1["PyTorch amp 与 GradScaler"]
  P2["PyTorch clip_grad_norm_"]
  N1["NCCL 集合通信"]
  X1["A 级脚本 量化器对比"]
  X2["C 级脚本 clip 等价性"]

  A1 -->|"支撑 recipe 落点不同"| CLAIM
  A2 -->|"支撑 配置可被静默改写"| CLAIM
  A3 -->|"支撑 类型判定链的必要性"| CLAIM
  A4 -->|"支撑 粒度的真实作用"| CLAIM
  A5 -->|"支撑 整块归约的前提"| CLAIM
  A6 -->|"支撑 scaler 的实际语义"| CLAIM
  A7 -->|"支撑 TE 耦合深度"| CLAIM

  L1 -->|"支撑 MX block 32 与 E8M0"| A1
  L2 -->|"支撑 NVFP4 的训练可行性"| A4
  L3 -->|"支撑 FP8 两种格式的规格"| A4
  L4 -->|"支撑 outlier 是量化难点"| A4
  L5 -->|"支撑 均匀量化的粒度敏感"| A4
  D1 -->|"支撑 per-tensor scale 与 amax 同步"| A1
  D2 -->|"支撑 block 128 与 2 的幂 scale"| A4
  D3 -->|"支撑 E4M3 无 Inf 与次正规边界"| A4
  D4 -->|"支撑 FP4 对齐尺寸来源"| A7
  M1 -->|"支撑 特性开关的官方定位"| A2
  M2 -->|"支撑 配置字段的官方签名"| A1
  P1 -->|"支撑 loss scaling 的原始语义"| A6
  P2 -->|"支撑 clip_grad_norm_ 的算法"| A5
  N1 -->|"支撑 归约原语的缓冲区语义"| A1
  X1 -->|"支撑 粒度与极小值存活率"| A4
  X2 -->|"支撑 ULP 与 fallback 精度差"| A5
```
图注：中心论断为「低精度全链路由 recipe 取值与 TE 版本共同决定」；A1-A7 的行号出处见正文各节，X1/X2 的数字来自配套脚本 `results/stdout.txt`。每条外部来源的 URL 见参考节。图自绘，未复制任何官方或论文图。


这张图的结构：中心论断是「低精度全链路由 recipe 取值与 TE 版本共同决定」，七条源码事实（A1-A7）是直接支撑，五篇论文与十一处官方文档给外部锚点，两个实验脚本做交叉验证。**即使读者不接受某一条源码证据，只要接受实验或文献中的任一条，论断仍站得住。**

---

## 8. 小结与下一篇

本篇钉死了八件事：

1. **`fp8_recipe` 五个取值不在同一层。** `delayed` 走 `get_fp8_recipe()` 构造有状态对象；其余四个走每层现构造的无状态上下文，分派表里没有 `delayed`。
2. **版本门槛逐 recipe 独立**（2.1.0 / 2.2.0.dev0 / 2.3.0.dev0 / 2.9.0.dev0），TE < 2.1.0 时直接 `assert` 必须用 `delayed`。
3. **`fp8_param_gather` / `fp4_param_gather` 的静默降级只在 `--use-torch-fsdp2` 下**，且 `fp4` 那支的告警文案写的是 TE 2.0 而条件查的是 2.7.0.dev0。
4. **`uint8` 是量化存储的唯一标记**；`post_all_gather_processing` 负责还原转置/列式布局，TE 太老时推迟到下一次前向。
5. **粒度对 FP8 换的是量程不是精度。** MX 块大小 1→512，RMSE 恒为 `1.945893e-02`；而「块内混极小值」时 per-tensor 清零 60.9%、MX-block32 清 47.6%、block8 清 0.0%。同一笔账在 INT8 上划算（误差降 58.4%），在 FP8 上不划算。
6. **`clip_grads.py` 整块与逐循环差 1 ULP**；L135-L138 两条 fallback 的开根精度不同（`.item()` 提升到 float64），实测相对差 `2.145e-08`。
7. **`grad_scaler.py` 的 `hysteresis` 统计的是累计 inf 而非连续 inf**，因为它只在成功涨 scale 时重置。
8. **`get_fp4_align_size` 的 docstring 推导 32 而 `return 128`**，第三层理由是 CUDA graph：按 128 对齐后 host 端不必把 token 分布从 device 拷回。

四个明确的**待验证**（本篇不编造）：

- **bf16 路径上 `DynamicGradScaler` 是否真被启用**（本机无 GPU，且 §4.2 的分析只到「机制上 bf16 不需要 loss scaling」这一步）
- **`count_zeros_fp32` → `update(found_inf)` 之间的主循环链路**（两端语义已钉死，中间链路不在本篇两个文件里）
- **FP8 / NVFP4 训练后的精度损失量级**（量化误差可算，优化动力学闭合不了）
- **配置层与实现层的版本信息不同步**（`custom` 的 docstring 没提 2.9.0.dev0 门槛，实际影响多少用户路径未量化）

一条容易踩的坑：**PyTorch 文档站与 NVIDIA 文档站是 JS 渲染的，`curl` 只能拿到页面外壳，验不了锚点。** 本篇所有引用都先核过可达性，但**锚点是否存在需在浏览器里再点一遍**。

下一篇写**算子融合与 CUDA Graph**：`fusions/` 家族与 `transformer/cuda_graphs.py` 3311 行。本篇 §5.1 已经埋了一个引子 —— FP4 的对齐尺寸是被 graph 捕获的约束倒推出来的，那篇会把它讲透。

---

## 参考

### 源码（本文对应 commit `60e039626`，`megatron-core` 0.20.0）

仓库 `https://github.com/NVIDIA/Megatron-LM`。本篇引用文件与行号范围：

- `megatron/core/fp8_utils.py` 1056 行 —— L104-L119 `_unwrap_parameter_data` / `_is_instance_or_param_data`；L122-L131 `is_float8tensor`（TE1.x/2.x 类名差异）；L134-L141 `is_mxfp8tensor` / `is_grouped_tensor`；L144-L150 `is_grouped_tensor_with_quantized_storage`（`uint8` 标记）；L153-L170 `_get_grouped_quantized_recipe` / `is_grouped_mxfp8tensor`；L173-L190 `get_grouped_quantized_members`（训练关键路径契约）；L193-L226 `copy_tensor_to_quantized_param`；L229-L266 `copy_tensors_to_quantized_params`；L269-L300 `modify_grouped_tensor_rowwise_storage`；L303-L308 `dequantize_fp8_tensor`；L311-L342 `_resolve_callable_from_python_import_path`；L345-L353 `_get_custom_recipe`（2.9.0.dev0 门槛）；L385-L409 三个接口函数的模块注释；L410-L420 TE 2.2+ 分叉；L666-L684 三个 Interface Function；L687-L708 `post_all_gather_processing`；L711-L727 `is_first_last_bf16_layer`；L730-L745 `is_mxfp8_output_proj_active`；L752-L810 `get_fp8_recipe`；L857-L890 `get_fp8_context` —— L883-L888 delayed 与 bf16 边界互斥
- `megatron/core/fp4_utils.py` 302 行 —— L57-L67 `is_nvfp4tensor` / `is_grouped_nvfp4tensor`；L70-L77 `get_nvfp4_rowwise_packed_shape`（numel 减半）；L80-L97 `modify_nvfp4_rowwise_storage`；L176-L205 `get_fp4_align_size`（推导 32 / 返回 128）；L208-L213 `dequantize_fp4_tensor`
- `megatron/core/optimizer/clip_grads.py` 277 行 —— L10-L47 TE/Apex/本地三层回退；L55-L140 `get_grad_norm_fp32` —— L112-L118 multi-tensor applier `False`；L121 先 pow；L135-L138 两条 fallback；L143-L192 `clip_grad_by_total_norm_fp32` —— L169 decoupled dtype；L174 fp32 硬断言；L179-L192 系数与施加；L195-L277 `count_zeros_fp32` —— L241 FSDP 本地分片；L245-L250 三重判重；L257-L263 FSDP 与 DP 组互斥
- `megatron/core/optimizer/grad_scaler.py` 165 行 —— L11-L40 `MegatronGradScaler`；L26-L28 `inv_scale`；L43-L58 `ConstantGradScaler`；L61-L165 `DynamicGradScaler` —— L109-L125 六个断言；L128-L129 tracker 初值；L131-L153 `update`；L155-L165 状态存取
- `megatron/core/extensions/transformer_engine.py` 3678 行 —— L190 delayed 特判；L303-L309 / L312-L360 `_get_fp8_*` 系列 —— L337-L358 recipe 分派表；L363-L369；L372-L380；L2561-L2563 / L2743-L2745 delayed 全局 meta；L3295-L3328 `TEDelayedScaling` —— L3316-L3319 `fp8_interval` 软弃用；L3366-L3390 `te_checkpoint`
- `megatron/core/transformer/transformer_config.py` 3311 行 —— L580-L585 `fp8`（e4m3 / hybrid）；L587-L593 `fp8_recipe` 五取值；L595-L600 `fp8_param`；L604 `fp8_quantizer_factory`；L606-L617 `fp8_margin` / `fp8_interval` / `fp8_amax_history_len` / `fp8_amax_compute_algo`；L628-L631 `fp8_output_proj`；L639 `tp_only_amax_red`；L642-L651 `first_last_layers_bf16`；L672-L685 `fp4_recipe`；L1540-L1554 custom 与 output_proj 校验
- `megatron/core/parallel_state.py` 2692 行 —— L1877-L1900 `get_amax_reduction_group`（四个组）
- `megatron/training/arguments.py` —— L1023 `if args.use_torch_fsdp2:`（降级逻辑的外层条件）；L1043-L1056 两处静默降级；L1069-L1071 `dtype_map`（`'fp8' → torch.uint8`）；L1074-L1080 `map_dtype` 应用

### 论文

- Rouhani et al., *Microscaling Data Formats for Deep Learning*：https://arxiv.org/abs/2310.10537 —— MX block 32 + E8M0 的原始出处；§5.5 的格点规格与它交叉核对
- *Pretraining Large Language Models with NVFP4*：https://arxiv.org/abs/2509.25149 —— `fp4_utils.py` L200 那条 `Paper link`
- *FP8 Formats for Deep Learning*：https://arxiv.org/abs/2209.05433
- *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*：https://arxiv.org/abs/2208.07339 —— 均匀量化对 outlier 敏感的原始出处
- SmoothQuant, *Accurate and Efficient Post-Training Quantization for Large Language Models*：https://arxiv.org/abs/2211.10438
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*：https://arxiv.org/abs/2205.05198 —— 08 篇已引，本篇的 `override_linear_precision` 与之同源
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*：https://arxiv.org/abs/1604.06174
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*：https://arxiv.org/abs/2104.04473
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*：https://arxiv.org/abs/1910.02054
- DeepSeek-AI et al., *DeepSeek-V3 Technical Report*：https://arxiv.org/abs/2412.19437 —— TE `blockwise` 的灵感来源；§3.3 给出细粒度分组的动机是 *extend the dynamic range*，以及本篇引用的那个 < 0.25% 相对 loss 误差
- *FP8-LM: Training FP8 Large Language Models*：https://arxiv.org/abs/2310.18313
- Micikevicius et al., *Mixed Precision Training*：https://arxiv.org/abs/1710.03740 —— fp16 loss scaling 的原始出处，§4.2 那套机制的原典
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*：https://arxiv.org/abs/2306.00978 —— 离群通道主导量化误差的原始观察
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*：https://arxiv.org/abs/2305.14314 —— 把离群值放进更高精度路径

### 官方文档与规范

- Transformer Engine · FP8 Current Scaling：https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html —— E4M3/E5M2 规格、hybrid 默认、per-tensor fp32 scale、RTNE、两遍读、amax 同步与量化 all-gather
- Transformer Engine · FP8 Blockwise Scaling：https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html —— block 128 / 1D-2D、2 的幂 scale、`NVTE_FP8_BLOCK_SCALING_FP32_SCALES`、块 scale 四步算法、1D 转置不支持
- Transformer Engine 文档总入口：https://docs.nvidia.com/deeplearning/transformer-engine/ —— recipe 类族与 autocast 用法
- OCP Microscaling Formats (MX) Specification 1.0：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf —— FP8 编码表（E4M3 无 Inf、规格化与次正规边界）、MX block 定义
- cuBLAS 文档 · block scaling factors layout：https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout —— `fp4_utils.py` L201 那条 `Scaling factor layout`
- Megatron-Core 用户指南：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html
- Megatron-Core API 指南：https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/index.html —— `TransformerConfig` 等字段页
- Megatron Core 并行策略指南：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html —— amax 归约组所依赖的并行轴
- PyTorch · 自动混合精度与 loss scaling：https://docs.pytorch.org/docs/stable/amp.html —— `GradScaler` 的原始语义
- PyTorch · `torch.cuda.amp.GradScaler`：https://docs.pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler —— §4.2 对照的 PyTorch 侧实现
- PyTorch · 量化总览：https://docs.pytorch.org/docs/stable/quantization.html —— §5.5 离群值处理的推理侧背景
- PyTorch · `torch.linalg.vector_norm`：https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html —— §4.1 范数语义的标准定义
- PyTorch · `torch.nn.utils.clip_grad.clip_grad_value_`：https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_value_.html —— 与 `clip_grad_norm_` 对照的逐元素裁剪变体
- PyTorch · `Tensor.to`：https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html —— §2.2 类型与设备转换语义
- PyTorch · `torch.nn.RMSNorm`：https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html —— §1.1 层内 norm 的语义
- PyTorch · `torch.optim.AdamW`：https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html —— §3.2 两个状态张量的出处
- PyTorch · `torch.nn.utils.clip_grad.clip_grad_norm_`：https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html —— `get_grad_norm_fp32` 自述改编自它
- PyTorch · `Tensor.untyped_storage`：https://docs.pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html —— `resize_(0)` 与 storage 换指针的 API 语义
- PyTorch · autograd：https://docs.pytorch.org/docs/stable/autograd.html —— saved tensor 语义
- PyTorch · 分布式与集合通信：https://docs.pytorch.org/docs/stable/distributed.html
- PyTorch · FSDP：https://docs.pytorch.org/docs/stable/fsdp.html
- PyTorch · 优化器：https://docs.pytorch.org/docs/stable/optim.html
- NCCL 集合通信文档：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html —— 归约原语对缓冲区尺寸与归约类型的要求
- CUDA C++ 编程指南：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- arXiv 官方「Permissions and Reuse」页（本系列判定论文图许可的依据）：https://info.arxiv.org/help/license/reuse.html

### 配套实验与本系列

- 配套实验（`ipynbs` 仓库）：`experiments/megatron/low_precision/`
  - `quantizers.py` —— E4M3 / E5M2 逐位枚举格点、MX block scaling、INT8 三档粒度；四组数据的 RMSE / SNR / 极小值存活率 / 粒度曲线（15 条断言）
  - `grad_clip_equivalence.py` —— 整块与逐参数循环的 ULP 对照、两条 fallback 分支的精度与类型差、clip 系数（7 条断言）
  - `results/stdout.txt` —— 固化输出（22 条断言全过）
- 本系列前作：《（00）代码地图、配置系统与一次迭代的控制流全图》[/2026/09/28/megatron-00-foundation/]、《（01）进程组——网格切分、通信域与计算组织》[/2026/10/08/megatron-01-process-groups-and-comm-overlap/]、《（02）专家并行》[/2026/10/08/megatron-02-expert-parallel/]、《（03）张量并行与序列并行》[/2026/10/09/megatron-03-tensor-parallel-and-sequence-parallel/]、《（04）GTP：新一代张量并行》[/2026/10/11/megatron-04-generalized-tensor-parallelism/]、《（05）数据并行与优化器——三套实现的分野》[/2026/10/10/megatron-05-data-parallel-and-optimizers/]、《（08）显存账本——重计算、卸载与分布保存》[/2026/10/12/megatron-08-memory-ledger/]、《（14）层栈与 spec 机制》[/2026/10/12/megatron-14-module-spec-and-layer-stack/]、《（15）GPT 与 Llama 装配》[/2026/10/13/megatron-15-gpt-and-llama-model-assembly/]