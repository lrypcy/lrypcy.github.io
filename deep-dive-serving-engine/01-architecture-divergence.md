# 01 架构分歧的根：三条独立的设计路线

> 基准：vLLM `v0.31.0`（commit `db9527a4`）／ SGLang `v0.5.21`（commit `e00930c5`）。
> 本文所有行号引用均锁在上述 commit 上。

「vLLM 和 SGLang 谁更强」这个问题问错了。它们不是同一条路线的两个实现，
而是**三个互相独立的设计选择**各自演化出的结果。把这三个选择拆开，
后面所有的功能差异、性能差异、乃至选型结论都能从它们推出来。

| # | 设计选择 | vLLM 的答案 | SGLang 的答案 |
|---|---|---|---|
| 1 | KV 缓存怎么组织 | 固定 block +链式哈希，内容寻址 | radix tree over token 序列 |
| 2 | CPU调度与 GPU 计算怎么重叠 | 默认**不**重叠（`async_scheduling` 需显式开） | 默认**重叠**（overlap scheduler） |
| 3 | 进程与语言边界 | Python 引擎 + 每DP rank 一个 EngineCore 进程 | Python 引擎 + Tokio + 内嵌 Rust 网关／缓存 |

---

## 1. KV 缓存：链式哈希 vs radix tree

这是最容易被讲成「A 比 B 好」、实际上最不适合这么讲的一处。

### 1.1 vLLM：固定 block + 逐块链式哈希

vLLM 把 KV cache 切成固定大小的 block（默认 **16 token**），
每个 block 算一个哈希，哈希**把父块的哈希也算进去**，形成一条链：

$$
h_i = H\!\left(h_{i-1},\; \mathrm{tokens}[i],\; \mathrm{extra\_keys}\right)
$$

实现见 `hash_block_tokens()` —— `vllm/v1/core/kv_cache_utils.py:684`，
哈希输入的三元组 `(parent_block_hash, curr_block_token_ids_tuple, extra_keys)` 在 `kv_cache_utils.py:712-714`。

`extra_keys` 由 `generate_block_hash_extra_keys()` 拼装（`kv_cache_utils.py:614`），
拼接顺序固定为（`kv_cache_utils.py:645-647`）：

```python
extra_keys = lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
```

逐项确认（都带行号，可点开验证）：

| 字段 | 行号 | 备注 |
|:---|:---|:---|
| LoRA 名 **与 adapter 路径** | `kv_cache_utils.py:567`（函数）、`:584`（返回 `[("lora", name, path)]`） | 两项都在哈希里 |
| `cache_salt` | `kv_cache_utils.py:636-640` | **只对第一个 block 生效**（`start_token_idx == 0` 才加） |
| 多模态输入 | `kv_cache_utils.py:499` | |
| prompt embeddings | `kv_cache_utils.py:587` | |

默认 block size = 16，定义在 `DEFAULT_BLOCK_SIZE: ClassVar[int] = 16` —— `vllm/config/cache.py:71`。

**这个设计的直接后果**：前缀复用的最小单位是 16 token，且必须**逐块对齐**。
两个请求的公共前缀长 100 token，能复用的只有 96 token（6 个完整块），
剩下 4 token 的 KV 存不下、也用不上。

### 1.2 SGLang：radix tree over token 序列

SGLang 把所有历史 KV 放进一棵 radix tree（压缩前缀树），
新请求进来时沿树下降，匹配**任意 token 位置**的最长公共前缀，
只为分叉后的后缀做attention。

核心类 `RadixCache` —— `python/sglang/srt/mem_cache/radix_cache.py:280`，
`match_prefix()` 在 `radix_cache.py:354`，真正的匹配循环在 `_match_prefix_helper()` `radix_cache.py:639`。

匹配用的是**数组前缀比较**（`RadixKey.match()` → `match_at()`，`radix_cache.py:177` / `:181`），
不是查哈希表。`match_at`内部还做了指数搜索 + 二分来找首个分歧 token，
注释写明是「避免长公共前缀上的 per-token Python 循环」（`radix_cache.py:192-210`）。

**节点切分规则** —— 这是理解 SGLang 缓存行为的关键，原文在 `_match_prefix_helper`：

```python
prefix_len = child.key.match(key, page_size=self.page_size)
if prefix_len < len(child.key):
    new_node = self._split_node(child.key, child, prefix_len)
    value.append(new_node.value)
    node = new_node
    break
else:
    value.append(child.value)
    node = child
    key = key[prefix_len:]
```

—— `radix_cache.py:649-658`（插入路径的等价判断在 `radix_cache.py:727-734`，
`_split_node()` 实现在 `radix_cache.py:665`）

读法很直接：**匹配长度小于该节点存储的 key 长度时，就把节点切成两级**，
新节点继承 child 的 `priority`（`:668`）、`hit_count`（`:669`）、`lock_ref`（`:672`），
底层 KV 数据不复制，只切引用（`:673-677`）。

`page_size` 的默认值是 **`None`**（`python/sglang/srt/arg_groups/fields/schedule.py:139-141`），
`match_prefix` 匹配前会做 `key.page_aligned(self.page_size)`（`radix_cache.py:397`），
而 `match_at` 在 `page_size=1` 时不向下取整（`radix_cache.py:177-179`）。

> ⚠️ 常见误解要纠正：**SGLang 默认不是「1 token 一块」也不是「按16 对齐」，
> 而是「node 边界完全由实际公共前缀终点决定，只受可选的 `page_size` 对齐约束」**。

### 1.3 精确对照

| | vLLM | SGLang |
|:---|:---|:---|
| 复用粒度 | 固定 16 token | 最长公共前缀（任意 token 位置） |
| 匹配方式 | 逐块链式哈希查表 | 数组前缀比较 + 二分找分歧点 |
| 100token 公共前缀可复用 | 96 token | 100 token |
| 键隔离字段 | lora(name+path) / mm / cache_salt / prompt_embeds | `extra_key` / `cache_salt`（语义见下） |
| 淘汰策略 | 隐式 LRU（`FreeKVCacheBlockQueue`，`kv_cache_utils.py:246`） | **8 种可插拔策略** |
| 默认开关 | 开（`enable_prefix_caching: bool = True`，`vllm/config/cache.py:142`） | 开（`disable_radix_cache = False`，`fields/memory.py:64-70`） |

SGLang 的 `extra_key` 与 `cache_salt` 语义**不同**，源码注释写得很清楚（`radix_cache.py:73-76`）：
`extra_key` 只 namespace 树与存储、**从 KV events 中排除**；
`cache_salt` 同时 namespace 树、存储**和 KV events**。

### 1.4 淘汰策略：这是两边抽象层次最大的差异

vLLM 的 `BlockPool` **没有策略注册表**，`_maybe_evict_cached_block`（`block_pool.py:732-753`）
只处理「有没有 hash」「有没有引用计数」，LRU 由队列隐式承担。

SGLang 的 `RadixCache.evict()` 在 `radix_cache.py:553`，
用最小堆 + 从叶子向上蔓延，父节点变无子且 `lock_ref == 0` 时级联压回堆（`:574-576`）。
策略工厂在 `python/sglang/srt/mem_cache/utils.py:60-69`，共 **8 种**：

| 名称 | 类 | 行号 |
|:---|:---|:---|
| `lru` | `LRUStrategy` | `evict_policy.py:16` |
| `lfu` | `LFUStrategy` | `evict_policy.py:21` |
| `fifo` | `FIFOStrategy` | `evict_policy.py:26` |
| `mru` | `MRUStrategy` | `evict_policy.py:31` |
| `filo` | `FILOStrategy` | `evict_policy.py:36` |
| `priority` | `PriorityStrategy` | `evict_policy.py:41` |
| `slru` | `SLRUStrategy` | `evict_policy.py:84` |
| `tlru` | `TLRUStrategy` | `evict_policy.py:49` |

> **选型含义**：如果你的负载里不同请求的价值差异很大（比如付费用户 vs 免费用户、
> 带SLA 的请求 vs 可重试的批处理），SGLang 的 `priority` / `slru` 是现成的；
> vLLM 侧要自己改 `BlockPool`。

---

## 2. 调度器：默认重叠 vs 默认不重叠

### 2.1 vLLM：调度器里没有 prefill / decode 之分

`Scheduler.schedule()` 起于 `vllm/v1/core/sched/scheduler.py:562`，是约 950 行的单体函数。
源码注释写得很明确：**调度器里没有 prefill 阶段和 decode 阶段之分**，
只有 `num_computed_tokens` 追赶 `num_tokens_with_spec`（`scheduler.py:564-573`）。

准入路径（`waiting` → `running`）：

| 步骤 | 行号 |
|:---|:---|
| 两个队列定义（`waiting` / `running`） | `scheduler.py:212` / `:220` |
| 准入主循环 | `scheduler.py:885` |
| **持 KV 的请求永远优先**：`request_queue = self.kv_holding_waiting or self.waiting` | `scheduler.py:887` |
| 硬闸门 `if num_running >= self.max_num_active_reqs: break` | `scheduler.py:893` |
| 实际迁移 `pop_request()` → `self.running.append()` | `scheduler.py:1290-1292` |
| LoRA 超限则跳过 | `scheduler.py:911-922` |
| 本地 prefix cache 命中查询 | `scheduler.py:930-937` |

值得单独指出的是 `scheduler.py:887`：**已经持有 KV block 的请求
在准入时优先于waiting 队列里的新请求**。这行代码是 v0.31.0 新增的
（release notes 里对应「waiting queue logic revamped so that requests already
holding KV blocks are scheduled preferentially」，目的是「suppress throughput
drops and deadlocks in resource-constrained situations」）。

chunked prefill 的实际截断在 `scheduler.py:679-680`：
`if 0 < long_prefill_token_threshold < num_new_tokens: num_new_tokens = long_prefill_token_threshold`。
阈值 `long_prefill_token_threshold` 默认 **0（不启用）** —— `vllm/config/scheduler.py:80`。

### 2.2 SGLang：overlap 默认开，但也有反例

开关字段 `disable_overlap_schedule: bool`，默认 `False`（即**默认开启**）——
`python/sglang/srt/arg_groups/fields/schedule.py:187-193`，
help 文本 “Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.”

运行时读取在 `scheduler.py:489-491`，`init_overlap()` 定义在 `scheduler.py:1615`。

「CPU 跑在 GPU 前面一个 batch」的具体实现是 `event_loop_overlap()` —— `scheduler.py:1941`。
机制是一个双端队列 `self.result_queue`（`:1943-1945`）缓存上一批结果：
先 `pop_and_process()` 处理上一批结果（`:1947-1950`），
**再 launch 当前批**（`:1988`），并立刻把 `(batch, batch_result)` 压回队列（`:1991`）。

一个容易被忽略的细节：**连续两个 prefill 批次会主动关闭 overlap 以改善 TTFT**，
判定在 `is_disable_overlap_for_batch()` —— `scheduler.py:2015`，
条件在 `:2031-2035`，由环境变量 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 控制。
采样也参与 overlap（`launch_batch_sample_if_needed()`，`scheduler.py:2007`），
因为采样依赖上一批结果，必须排在其后。

### 2.3 overlap 不是无条件赢 —— 这是有issue 证据的反例

SGLang issue [#2558](https://github.com/sgl-project/sglang/issues/2558) 报告：
Qwen2.5-0.5B-Instruct在单张 A100 上跑 `bench_serving` 随机负载（4096 in / 2048 out），
**关掉 overlap 反而更快**（22,025 → 25,555 total tok/s）。社区已复现。

这与 v0.4 官方博客的措辞并不矛盾 —— 那里的限定条件是
「gains most significant on **small models and large tensor parallelism sizes**」。
把两句话合起来读，结论是：

> 小模型 + 单卡 + 低并发时，overlap 引入的 bookkeeping 开销可能超过它隐藏的调度开销。
> 「overlap 默认开启」不等于「always faster」。

vLLM 侧的对应物是 `AsyncScheduler`（`vllm/v1/core/sched/async_scheduler.py:14`，共 78 行），
开关 `async_scheduling: bool | None = None` —— `vllm/config/scheduler.py:209`，
选择逻辑 `get_scheduler_cls()` 在 `:229-237`。**默认 None =关闭。**

vLLM 自己的收益也有 PR 级数据：PR [#32618](https://github.com/vllm-project/vllm/pull/32618)
自测 async scheduling + PP 在 Qwen3-30B-A3B-Thinking-2507-FP8 / PP4 上带来
E2E 吞吐 +30.8%、TPOT +31.8%（**项目方自测，无第三方复现**）。

---

## 3. 进程与语言边界

### 3.1 vLLM：EngineCore 进程隔离 + 故障隔离

vLLM V1 把EngineCore 拆到独立进程（每个 DP rank 一个），
好处之一是**故障可以被兜住而不带崩整个 server**。

`vllm/v1/fault_tolerance/engine_core_sentinel.py`（197 行）把每个 engine core 的busy loop 包起来：

```python
def fault_tolerant_wrapper(busy_loop_func: Callable):
    def run_with_fault_tolerance(self: "EngineCoreProc"):
        while True:
            try:
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as exc:
                if not self.enable_fault_tolerance:
                    raise
                self.ft_sentinel.on_fault(exc)
                recovered = self.ft_sentinel.resumed.wait(
                    timeout=self.ft_sentinel.engine_recovery_timeout_sec
                )
                if recovered:
                    continue
                raise
    return run_with_fault_tolerance
```

—— `engine_core_sentinel.py:173`（返回的闭包在 `:176`）

`on_fault()` 在 `engine_core_sentinel.py:80`，函数体 `:80-106`，按顺序做四件事：

1. 清 `resumed` 事件（`:82`）
2. 用 `finish_requests(None, FINISHED_ABORTED)` 中止所有在飞请求并 `_send_abort_outputs()`（`:88-89`）
3. `batch_queue.clear()`（`:90-91`）
4. `model_executor.is_failed` 则置 `DEAD`，否则置 `UNHEALTHY`（`:92-98`）

状态作为 utility output 推给前端，让上层能立刻绕开降级的 engine，而不是等超时。

**必须说清楚的两点**：

- 恢复指令**当前只实现了 `retry` 一个**。`handle_command()` 用
  `run_method(self, ft_request.instruction, (ft_request,), {})`（`:67`）动态派发，
  sentinel 上唯一可被外部驱动的恢复方法是 `retry()`（`:122-137`）。
- **没有 instruction 白名单**。`FaultToleranceRequest.instruction` 是裸 `str`
  （`vllm/v1/fault_tolerance/utils.py:15`），`api_router.py`（`:30`）只校验类型不做枚举校验。
  派发是通用机制，但目前没有更多恢复策略（部分重启、drain-and-replace）落地。

超时默认 `engine_recovery_timeout_sec: int = 120` —— `vllm/config/fault_tolerance.py:12`。

### 3.2 SGLang：Python + Tokio + 内嵌 Rust

SGLang 的仓库里有一整套 Rust 组件：

| 目录 | 内容 |
|:---|:---|
| `sgl-model-gateway/` | Rust 网关（不是独立仓库，在主仓里） |
| `rust/` | `sglang-server`、`sglang-radix-tree`、`sglang-grpc`、`sglang-mm`、`sglang-renderer` |

> **纠正一个常见说法**：v0.5.21 的 Rust router **不叫 `sgl-router`**，
> 该目录在此 commit 中不存在；实际目录名是 **`sgl-model-gateway/`**。
> 历史上确实有过 `sgl-router` 独立仓库（v0.4 博客提到「Rust router」），
> 后来并入主仓并改名。

cache-aware routing 的实现在 `sgl-model-gateway/src/policies/cache_aware.rs`（1641 行）：
`pub struct CacheAwarePolicy` 在 `:112`，`impl LoadBalancingPolicy for CacheAwarePolicy` 在 `:386`。
底层 radix tree 在 `sgl-model-gateway/src/policies/tree.rs`（2310 行），
`pub struct Tree` 在 `:246`，含自定义 `Hasher for CharHasher`（`:58`）。
router 策略共 12 个文件（`cache_aware` / `tree` / `prefix_hash` / `consistent_hashing` /
`power_of_two` / `round_robin` / `random` / `manual` / `bucket` / `factory` / `registry` / `mod`）。

**调度策略** —— 这是一个 vLLM 侧完全没有的维度。
`python/sglang/srt/managers/schedule_policy.py`（1640 行），主类 `SchedulePolicy` 在 `:238`，
策略被显式二分：

| 类别 | 策略 | 行号 |
|:---|:---|:---|
| **CacheAware** | `lpm`（longest prefix match） | `:223` |
| | `dfs-weight` | `:224` |
| | `hrrn`（highest response ratio next，token-based aging） | `:225` |
| | `shortest-prefill-first` | `:226` |
| **CacheAgnostic** | `fcfs` | `:232` |
| | `lof`（longest output first） | `:233` |
| | `random` | `:234` |
| | `routing-key` | `:235` |

枚举定义 `CacheAwarePolicy` 在 `schedule_policy.py:220-226`，
`CacheAgnosticPolicy` 在 `:229-235`，CLI choices 与之一致（`fields/schedule.py:86-96`）。

> ⚠️ **又一个常见误解要纠正**：**SGLang 默认不是 `lpm`，是 `fcfs`**
> —— `fields/schedule.py:98` 的默认值就是 `"fcfs"`（cache-agnostic 类）。
> 「SGLang 默认用 longest prefix match 调度」是错的，
> 想用 lpm 必须显式传 `--schedule-policy lpm`。

### 3.3 仓库规模（量化工程投入）

用 `git ls-tree -r <sha>`统计：

| 指标 | vLLM | SGLang |
|:---|---:|---:|
| 总文件数 | 7431 | 9912 |
| 主语言目录文件数（`vllm/` vs `python/`） | 3183 | 5247 |
| `.py` 文件数 | 4880 | 7024 |
| 测试文件数 | 2282（`tests/`） | 2495（`test/`） |
| `.rs` 文件数 | — | 567 |

> ⚠️ 注意：`scheduler.py` 行数对比是**同向**的（SGLang 6125行 vs vLLM 3359 行），
> 但这不说明 SGLang 的调度器「更复杂」—— 两者职责范围不同。
> vLLM 把 KV 管理拆到 `kv_cache_manager.py`（962 行）+ `block_pool.py`（904 行），
> SGLang 大量逻辑在radix cache 侧（`radix_cache.py` 823 行 + `schedule_policy.py` 1640 行）。
> 单看scheduler.py 行数会得出错误结论。

---

## 4. 小结：三条路线的代价

| 选择 | 换来什么 | 付出什么 |
|:---|:---|:---|
| 链式哈希 block | 显存账本可预测；块是分配回收的原子单位，实现简单 | 前缀复用受16 对齐约束；缓存策略不可插拔 |
| radix tree token 级 | 任意位置复用；缓存策略可插拔；原生支持分支前缀 | 树结构维护有开销（无复用时是净负担）；碎片化更复杂 |
| 默认 overlap | 隐藏 CPU 调度开销 | 小模型单卡可能反而变慢（issue #2558 证据） |
| 默认不overlap | 行为可预测 | 调度开销裸露 |
| EngineCore 进程隔离 | 单 rank 崩溃不带崩 server | IPC 开销；当前只支持 `retry` 一种恢复 |
| 内嵌 Rust 网关 | 路由与前缀匹配不吃 GIL | 编译链与 Python 侧分离，调试链路更长 |

这三组取舍**互相独立**，所以两个框架在能力上会呈现「交叉优势」而非单边碾压：
SGLang 有 vLLM 没有的可插拔缓存策略和 PD 角色热切换，
vLLM 有 SGLang 没有的 EngineCore 故障隔离和更广的 attention backend 矩阵
（44 vs 24，见 [02](02-feature-matrix.md)）。

## 参考

- SGLang 论文（RadixAttention，6.4× 那组数字的出处）：[arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- SGLang v0.4 overlap scheduler 与 cache-aware LB：<https://openlm.ai/sglang-v0.4/>
- vLLM async scheduling PR：[vllm#32618](https://github.com/vllm-project/vllm/pull/32618)
- SGLang issue #2558（overlap 在小模型上反而慢）：<https://github.com/sgl-project/sglang/issues/2558>