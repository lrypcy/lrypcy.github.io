---
title: "编译器知识体系深度解析（11）：从零开发迷你 AI 编译器：图捕获、融合与代码生成"
date: 2026-08-25 20:55:00 +0800
categories:
  - 编译器技术
tags: [ai-compiler, operator-fusion, memory-planning, triton]
layout: post
mathjax: true
---

> **LLM/编译器技术深度解析系列 · 第 11 篇 / 共 13 篇**（Part C 开发实战 · 2/3）
>
> [第 10 篇 从零开发经典编译器](/2026/08/25/compiler-10-build-classic-compiler/) ← **本篇** → [第 12 篇 编译器工程化](/2026/08/25/compiler-12-engineering/)

**TL;DR**
* 本文用约 **550 行无依赖 Python**（numpy 仅作 eager 对照）实现 MicroAI-Compiler（MAC）：记录式图捕获 → 常量折叠 → elementwise 融合（带闭包合法性检查）→ liveness 内存规划 → 双目标代码生成（可执行 C + Triton 文本工件）。
* 端到端实测（4096×4096 fp32，arm64，clang -O2）：六算子链 `relu((x·1.5+y)·0.25−y·0.5)`，eager numpy **53.57 ms/iter（128 MB 临时量）vs 融合 C 版 2.75 ms/iter（零临时）= 19.45x 加速**，数值 allclose 通过。19 倍的来源不是魔法：融合把 7 次数组往返压成 1 次，让 kernel 跑到接近内存带宽极限。
* 与第 10 篇逐阶段对照——词法↔图捕获、IR pass↔图 pass、寄存器分配↔内存规划、LLVM IR↔C/Triton——你会看到 AI 编译器没有任何新原理，只有新约束。
* 实现过程踩了三个真 bug（融合后悬空引用、flat 成员表漏叶子、基准被文件 I/O 污染），每一个都是真实 AI 编译器框架的著名陷阱的微缩版，见 §7。

---

## 目录

- [1. 设计映射：把第 10 篇的方法论平移到张量世界](#1)
- [2. 图捕获：记录式 Builder](#2)
- [3. 图 Pass 三连](#3)
- [4. C 目标代码生成](#4)
- [5. Triton 工件与双目标的意义](#5)
- [6. 端到端实测：19.45x 从何而来](#6)
- [7. 调试战报：三个真 bug](#7)
- [8. 批判与展望](#8)
- [FAQ](#faq)
- [参考资料](#参考)

<a name="1"></a>
## 1. 设计映射：把第 10 篇的方法论平移到张量世界

| 经典编译器（TinyLang） | 迷你 AI 编译器（MAC） | 本系列出处 |
|----------------------|---------------------|-----------|
| 词法分析（字符→token） | 图捕获（Python 调用→图节点） | 01 |
| AST + 语义检查 | Graph IR + 合法性检查 | 02 |
| 中端 pass（DCE/GVN） | 图 pass（常量折叠/融合） | 03 |
| 寄存器分配（活跃区间复用） | 内存规划（缓冲区间复用） | 04 |
| 指令选择+代码发射 | kernel 代码生成（C/Triton） | 04 |
| 解释器对照组 | numpy eager 对照组 | —— |

关键差异只有两处：
1. **优化对象从标量变为张量**：收益模型从"指令数"变成"内存往返次数"——这是融合成为第一优化的原因；
2. **正确性判据从容差为零变为容差预算**：浮点重结合允许微小偏差，验收用 `allclose(atol=1e-5)`。

<a name="2"></a>
## 2. 图捕获：记录式 Builder

不修改 numpy、不用字节码 hack——最朴素也最诚实的捕获方式是**显式 Builder API**（等价于 FX 的 functional 形式、TVM Relax 的构建器）：

```python
class Node:
    _n = 0

    def __init__(self, op, inputs=(), attrs=None, shape=None):
        self.id = Node._n
        Node._n += 1
        self.op = op                      # 'input'/'const'/'mul'/'add'/'silu'/'fuse'
        self.inputs = list(inputs)        # 上游 Node 列表
        self.attrs = attrs or {}
        self.shape = shape


class Graph:
    def __init__(self):
        self.nodes = []                   # 拓扑序
        self.outputs = []

    def dump(self):
        lines = []
        for nd in self.nodes:
            srcs = ", ".join(f"%{i.id}" for i in nd.inputs)
            lines.append(f"%{nd.id} = {nd.op}({srcs})"
                         if nd.inputs else f"%{nd.id} = {nd.op}")
        outs = ", ".join(f"%{o.id}" for o in self.outputs)
        lines.append(f"return {outs}")
        return "\n".join(lines)


class Builder:
    def __init__(self):
        self.g = Graph()

    def input(self, name, shape):
        nd = Node("input", shape=shape, attrs={"name": name})
        self.g.nodes.append(nd)
        return nd

    def const(self, v):
        nd = Node("const", attrs={"v": float(v)})
        self.g.nodes.append(nd)
        return nd

    def mul(self, a, b):
        nd = Node("mul", inputs=[a, b], shape=a.shape)
        self.g.nodes.append(nd)
        return nd

    def add(self, a, b):
        nd = Node("add", inputs=[a, b], shape=a.shape)
        self.g.nodes.append(nd)
        return nd

    def silu(self, a):
        nd = Node("silu", inputs=[a], shape=a.shape)
        self.g.nodes.append(nd)
        return nd

    def sub(self, a, b):
        nd = Node("sub", inputs=[a, b], shape=a.shape)
        self.g.nodes.append(nd)
        return nd

    def relu(self, a):
        nd = Node("relu", inputs=[a], shape=a.shape)
        self.g.nodes.append(nd)
        return nd

    def output(self, *xs):
        self.g.outputs = list(xs)
```

工作负载（六条 elementwise 算子，故意含一条分支链制造寿命交错）：

```python
def build_sample_graph(shape=(4096, 4096)):
    """out = relu((x*1.5 + y)*0.25 - y*0.5)：六条 elementwise 算子"""
    b = Builder()
    x = b.input("x", shape)
    y = b.input("y", shape)
    t1 = b.mul(x, b.const(1.5))
    t2 = b.add(t1, y)
    t3 = b.mul(t2, b.const(0.25))
    t4 = b.mul(y, b.const(0.5))
    t5 = b.sub(t3, t4)
    t6 = b.relu(t5)
    b.output(t6)
    return b.g
```

**捕获到的原始图（真实 dump）**：

```text
%0 = input()
%1 = input()
%2 = const()
%3 = mul(%0, %2)
%4 = add(%3, %1)
%5 = const()
%6 = mul(%4, %5)
%7 = const()
%8 = mul(%1, %7)
%9 = sub(%6, %8)
%10 = relu(%9)
return %10
```

11 个节点里藏着多少浪费？eager 执行它要分配 6 个中间数组（每片 64 MB），HBM 往返 7 次。接下来三连 pass 逐层消灭。

<a name="3"></a>
## 3. 图 Pass 三连

### 3.1 常量折叠

与第 03 篇 SCCP 的直线代码特例同构：两个 const 进、一个 const 出。

```python
def pass_const_fold(g):
    """编译期折叠 const-op-const"""
    changed = 0
    for nd in g.nodes:
        if nd.op in ELEMENTWISE and len(nd.inputs) == 2 \
           and nd.inputs[0].op == "const" and nd.inputs[1].op == "const":
            a, b = nd.inputs[0].attrs["v"], nd.inputs[1].attrs["v"]
            nd.op, nd.inputs, nd.attrs = "const", [], \
                {"v": {"mul": a * b, "add": a + b, "sub": a - b}[nd.op]}
            changed += 1
    return changed
```

### 3.2 算子融合：本篇的心脏

融合的收益模型一句话讲清：elementwise 链的每个中间量只被消费一次，因此可以完全驻留在寄存器里——**把 N 次数组往返变成 1 次**。但实现有两个必须严肃对待的正确性问题：

**问题一：闭包条件**。中间成员若还被链外消费者引用（fan-out > 1），把它内联进 fuse 组就会重复计算甚至改变语义。解法：入组条件加 `consumer_count == 1`。

**问题二：引用重接**。Python 对象图里，"删节点"不会自动更新别人的 `inputs` 列表；漏掉重接会产生悬空引用和幽灵重复融合。解法：替换后显式遍历全图改写引用。

```python
def consumer_counts(g):
    cc = {}
    for nd in g.nodes:
        for i in nd.inputs:
            cc.setdefault(id(i), 0)
            cc[id(i)] += 1
    for o in g.outputs:
        cc.setdefault(id(o), 0)
    return cc


def pass_const_fold(g):
    """编译期折叠 const-op-const"""
    changed = 0
    for nd in g.nodes:
        if nd.op in ELEMENTWISE and len(nd.inputs) == 2 \
           and nd.inputs[0].op == "const" and nd.inputs[1].op == "const":
            a, b = nd.inputs[0].attrs["v"], nd.inputs[1].attrs["v"]
            nd.op, nd.inputs, nd.attrs = "const", [], \
                {"v": {"mul": a * b, "add": a + b, "sub": a - b}[nd.op]}
            changed += 1
    return changed


def pass_fuse_elementwise(g):
    """自底向上融合 elementwise 链。
    合法性（闭包条件）：中间成员必须只被链内消费者引用（consumer_count==1），
    否则该前驱不并入（保持独立）。融合后把根节点的所有外部引用重接到 fuse 节点。"""
    changed = 0
    processed_roots = set()
    i = len(g.nodes) - 1
    while i >= 0:
        nd = g.nodes[i]
        if nd.op in ELEMENTWISE and id(nd) not in processed_roots:
            cc = consumer_counts(g)
            group = [nd]
            seen = {id(nd)}
            queue = list(nd.inputs)
            while queue:
                cur = queue.pop()
                if id(cur) in seen:
                    continue
                seen.add(id(cur))
                if cur.op in ELEMENTWISE and cc.get(id(cur), 0) == 1:
                    group.append(cur)           # 独占前驱才可安全内联
                    queue.extend(cur.inputs)
            if len(group) > 1:
                # 平铺成员表（供代码生成重建表达式，先深后浅保证拓扑）
                flat = []
                memo = set()

                def rec2(m):
                    mid = id(m)
                    if mid in memo:
                        return
                    memo.add(mid)
                    for inp in m.inputs:
                        if inp in group:
                            rec2(inp)
                        elif id(inp) not in memo:      # 叶子：input/const 也入表
                            memo.add(id(inp))
                            flat.append((inp.id, inp.op, (),
                                         inp.attrs.get("v"),
                                         inp.attrs.get("name")))
                    flat.append((m.id, m.op,
                                 tuple(x.id for x in m.inputs),
                                 m.attrs.get("v"), m.attrs.get("name")))

                for m_ in sorted(group, key=lambda z: z.id):
                    rec2(m_)
                root_id = nd.id
                params = [inp for inp in nd.inputs if inp.op == "input"]
                seen_p = set()
                extra_params = []
                for m_ in group:                 # 收集所有叶上 input（去重保序）
                    for inp in m_.inputs:
                        if inp.op == "input" and id(inp) not in seen_p:
                            seen_p.add(id(inp))
                            extra_params.append(inp)
                params = extra_params if len(params) < len(extra_params) else params
                # 统一用全量收集结果
                params = extra_params
                new_nd = Node("fuse", inputs=params, shape=nd.shape,
                              attrs={"flat": flat, "root": root_id,
                                     "members": [m_.id for m_ in
                                                 sorted(group, key=lambda z: z.id)]})
                idx = g.nodes.index(nd)
                g.nodes[idx] = new_nd
                member_set = set(map(id, group)) - {id(new_nd)}
                g.nodes[:] = [n for n in g.nodes
                              if n is new_nd or id(n) not in member_set]
                # 关键：把外部对根节点 nd 的引用重接到 new_nd
                for other in g.nodes:
                    if other is new_nd:
                        continue
                    other.inputs[:] = [new_nd if x is nd else x
                                       for x in other.inputs]
                g.outputs[:] = [new_nd if x is nd else x for x in g.outputs]
                processed_roots.add(id(new_nd))
                changed += 1
                i = min(i, len(g.nodes) - 1)
                continue
        i -= 1
    return changed
```

**编译后的图（真实 dump）**：

```text
%0 = input()
%1 = input()
%2 = const()
%5 = const()
%7 = const()
%11 = fuse(%1, %0)
return %11
```

六条算子坍缩为一个 `fuse` 节点；`flat` 成员表完整保存了表达式树供后端重建。对照 Inductor 输出里的 `cpp_fused_add_mean_mul_silu_0`（第 09 篇 §3.1），命名习惯都如出一辙。

### 3.3 内存规划：线性扫描的张量版

把每个中间量的生命周期看作区间（定义处出生、最后消费处死亡），做区间复用——算法骨架就是第 04 篇的线性扫描：

```python
def pass_memory_plan(g):
    """基于活跃区间的缓冲复用规划（类比第 04 篇线性扫描）"""
    order = {nd.id: k for k, nd in enumerate(g.nodes)}
    live = {}
    for nd in g.nodes:
        for inp in nd.inputs:
            live.setdefault(inp.id, [order[inp.id], order[inp.id]])
            live[inp.id][1] = max(live[inp.id][1], order[nd.id])
    for o in g.outputs:
        live[o.id] = [order[o.id], len(g.nodes)]
    intervals = sorted(((s, e, nid) for nid, (s, e) in live.items()),
                       key=lambda t: (t[0], t[1]))
    pools, detail = [], []
    for s, e, nid in intervals:
        placed = False
        for pool in pools:
            if pool[-1][1] < s:
                pool.append((s, e, nid))
                detail.append(f"%{nid} 复用 %{pool[0][2]} 的槽位")
                placed = True
                break
        if not placed:
            pools.append([(s, e, nid)])
    naive = len(intervals)
    planned = len(pools)
    return naive, planned, detail
```

对分支寿命交错的测试图（自检阶段5）**真实输出**：

```text
阶段5 内存规划: PASS (8 个分配 -> 4 个缓冲)
      %41 复用 %39 的槽位
      %43 复用 %39 的槽位
      %45 复用 %39 的槽位
      %46 复用 %40 的槽位
```

8 个中间量的图只需 4 个物理缓冲——同样的算法在 GPU 世界里决定的是共享内存池的大小。

<a name="4"></a>
## 4. C 目标代码生成

fuse 节点的 `flat` 表驱动一次递归求值，产出**单循环零临时**的 C kernel。核心生成逻辑（模板见随文脚本）：

```python
def gen_c_from_fuse(fuse_node, out_path="out.bin"):
    """从 fuse 节点的 flat 成员表生成单循环零临时 C kernel。
    compute() 以 restrict 指针收参；--bench 模式下数据驻留内存循环计时。"""
    info = {fid: {"op": op_, "inputs": ins, "v": v_, "name": nm_}
            for fid, op_, ins, v_, nm_ in fuse_node.attrs["flat"]}
    body_lines = []

    def visit(nid, memo):
        if nid in memo:
            return memo[nid]
        d = info[nid]
        if d["op"] == "input":
            memo[nid] = f"p_{d['name']}[i]"
            return memo[nid]
        if d["op"] == "const":
            return repr(np.float32(d["v"])) + "f"
        if d["op"] == "mul":
            a, b = visit(d["inputs"][0], memo), visit(d["inputs"][1], memo)
            e = f"{a} * {b}"
        elif d["op"] == "add":
            a, b = visit(d["inputs"][0], memo), visit(d["inputs"][1], memo)
            e = f"{a} + {b}"
        elif d["op"] == "sub":
            a, b = visit(d["inputs"][0], memo), visit(d["inputs"][1], memo)
            e = f"{a} - {b}"
        elif d["op"] == "relu":
            a = visit(d["inputs"][0], memo)
            e = f"fmaxf({a}, 0.0f)"
        else:                                    # silu
            a = visit(d["inputs"][0], memo)
            e = f"({a} / (1.0f + expf(-({a}))))"
        var = f"t{len(memo)}"
        body_lines.append(f"        const float {var} = {e};")
        memo[nid] = var
        return var

    root_expr = visit(fuse_node.attrs["root"], {})
    body = "\n".join(body_lines) + f"\n        out[i] = {root_expr};"
    names = sorted({d["name"] for d in info.values() if d["op"] == "input"})
    params = ", ".join(f"const float *restrict p_{nm}" for nm in names)
    call_args = ", ".join(f"p_{nm}" for nm in names)
    decls = "\n".join(f'    float *p_{nm} = read_bin("{nm}.bin", N);'
                      for nm in names)
    n_elem = int(np.prod(fuse_node.shape))
    return C_TEMPLATE.format(params=params, body=body, n=n_elem,
                             decls=decls, call_args=call_args,
                             out_path=out_path)
```

**生成的 compute 函数（真实输出，未做任何手工润色）**：

```c
static void compute(const float *restrict p_x, const float *restrict p_y, float *restrict out) {
    for (long i = 0; i < N_G; i++) {
        const float t1 = p_x[i] * 1.5f;
        const float t3 = t1 + p_y[i];
        const float t4 = t3 * 0.25f;
        const float t5 = p_y[i] * 0.5f;
        const float t6 = t4 - t5;
        const float t7 = fmaxf(t6, 0.0f);
        out[i] = t7;
    }
}
```

七个中间值全部是标量局部变量——`restrict` 告诉 clang 无别名混叠，`-O2` 自动向量化为 NEON 指令。对比 eager 版本：同样这六个操作，numpy 要启动六个 kernel、读写七次 64 MB 数组。**这就是"融合消除 HBM 往返"的字面意思。**

<a name="5"></a>
## 5. Triton 工件与双目标的意义

同一个 fuse 组还能发射为 Triton 文本（GPU 上可直接运行）：

```python
@triton.jit
def fused_silu_mul_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    t = x * 1.5
    t = t + y
    t = t * 0.25
    u = y * 0.5
    t = t - u
    result = tl.maximum(t, 0.0)
    tl.store(out_ptr + offs, result, mask=mask)
```

注意 C 目标与 Triton 目标的分工：CPU 版靠 `restrict` + 自动向量化，GPU 版靠块级抽象隐藏线程映射。**同一份图 IR、同一套融合决策，两个世界各自最优的落地形态**——这正是第 09 篇 Inductor 双后端架构的全部逻辑。

<a name="6"></a>
## 6. 端到端实测：19.45x 从何而来

自检全绿（**真实输出**）：

```text
阶段1 图捕获与打印: PASS
阶段2 常量折叠: PASS (mul(2,3) -> const 6)
阶段3 eager 数值 vs numpy 参考: PASS
阶段4 算子融合: PASS (mul+add+silu -> 1 个 fuse 组, 引用重接完成)
阶段5 内存规划: PASS (8 个分配 -> 4 个缓冲)
阶段6 C 编译执行数值闭环: PASS

自检通过项: 6/6
```

基准方法学说明（比数字更重要）：数据以 `.bin` 读入 C 进程后**驻留内存循环计时**（best-of-7），避免磁盘 I/O 污染——本文初版的基准就栽在这上面（§7 bug 3）。

**真实基准结果**：

| 指标 | eager numpy | MAC 融合 C 版 |
|------|------------|--------------|
| 吞吐耗时 | 53.57 ms/iter | **2.75 ms/iter** |
| 中间临时峰值 | ~128 MB | 0 |
| 数组往返次数 | 7 × 256 MB | 3 × 256 MB（读 x/y，写 out）|
| 数值一致性 | —— | allclose(r_eager, r_c, atol=1e-5) = True |

**加速比 19.45x**。拆解一下账本：eager 的 53 ms ≈ 1792 MB 流量 ÷ ~34 GB/s 有效带宽；融合版 2.75 ms ≈ 192 MB ÷ 70 GB/s——后者已经贴近 arm64 单核可达的带宽天花板。**融合的收益上限就是"省掉的流量 ÷ 剩余流量"，当剩余部分贴住带宽极限时加速比自然封顶**。这也解释了为什么 Inductor/TensorRT 在访存受限模型上收益巨大，而在纯 GEMM 大矩阵上收益有限（GEMM 是计算受限，瓶颈不在往返）。

<a name="7"></a>
## 7. 调试战报：三个真 bug

**Bug 1：融合后悬空引用导致重复融合**
症状：小图融合后出现两个 fuse 节点且其中一个引用已被删除的死成员。根因：创建 fuse 节点并删除旧成员后，没有把外部消费者（包括 output）对根节点的引用改接到新节点——旧 add 节点仍被 silu 的 inputs 持有，下一轮又把它当活节点吞了一次。教训：**图重写的完整性 = 结构替换 + 全图引用重写，缺一不可**。Inductor 内部为此维护统一的 mutation 机制，道理相同。

**Bug 2：flat 成员表漏叶子**
症状：C 生成时 KeyError。根因：递归收集成员时只登记了组内 elementwise 节点，作为叶子的 input/const 没进表。教训：**序列化表达式树时叶子也是节点**。"能内联的东西也要出现在 IR 里"是调试期最重要的纪律——优化掉它们是代码生成的职责，不是 IR 构造的职责。

**Bug 3：基准被文件 I/O 污染**
症状：第一版基准显示加速比 0.99x——因为每次迭代都重新从磁盘读 192 MB 输入，I/O 时间完全淹没计算差异。修复：数据驻留进程内、循环计时取 best-of-7。教训：**测融合性能必须隔离数据搬运**；反过来，部署场景中如果输入真的每次都来自磁盘，融合收益确实会被吃掉大半——benchmark 方法论本身就是编译器工程的一部分（第 12 篇展开）。

<a name="8"></a>
## 8. 批判与展望

* **MAC 距离可用产品还差三座山**：自动微分（训练侧）、动态 shape（guard/符号推导）、真正的 tuning 回路（tile 尺寸枚举）。每一座都是工业团队数年投入——但方向全部在本系列的射程内。
* **归约（mean/sum）尚未支持**：跨行归约打破"单循环"结构，需要两层循环或 welford 分段——Inductor 用 template + Reduction hint 处理，是很好的进阶阅读材料。
* **最有价值的迁移练习**：把第 02 篇的支配树/SSA 构建接进来，让 MAC 支持带控制流的图；再把第 03 篇的 DCE 接到融合之后清理死分支。做完这两件事，你对 AI 编译器的理解会超过大多数面试者。

## FAQ

**Q1：为什么不用 torch.fx 直接搭？**
可以用，而且生产上应该用。但 fx 会替你处理 guard、mutation、subgraph 重写——恰恰是本文要教学的内容。自己裸写一遍再看 fx，如同手写完递归下降再读 clang parser，处处似曾相识。

**Q2：19.45x 是否意味着所有模型都能期待这个数量级？**
不能。收益 = 被消除的访存占比。逐元素链（激活、归一化的 affine 部分）收益巨大；大 GEMM 几乎无收益（计算受限）；conv 视 fusion 后能否进入 implicit-gemm 而定。真实模型是混合体，所以端到端常见 1.2~3x。

**Q3：为什么生成 C 而不是直接生成汇编？**
分层原则：让 clang 做 instruction selection/调度（第 04 篇整套机器），MAC 只负责图级决策。除非目标是没有 C 工具链的 NPU，否则永远站在巨人肩膀上。

## 参考资料

[1] 本文随文脚本（单一事实来源）：`experiments_11_miniai.py`
[2] PyTorch TorchDynamo/Inductor 内部机制（与本篇逐模块对应）: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
[3] Triton 官方文档: https://triton-lang.org/main/index.html
[4] Ragan-Kelley et al., Halide (PLDI'13)，算法/调度分离思想的源头: https://doi.org/10.1145/2491956.2462176

---

> **下一篇**：[第 12 篇 编译器工程化](/2026/08/25/compiler-12-engineering/)——lit/FileCheck 测试体系、差分测试与 fuzzing、benchmark 方法学，以及"自研还是复用"的决策框架：编译器项目的成败一半在仓库之外。
