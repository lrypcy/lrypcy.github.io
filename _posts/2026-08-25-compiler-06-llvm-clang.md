---
title: "编译器知识体系深度解析（6）：LLVM/Clang 架构剖析：单一 IR 的胜利与一条被优化掉的循环"
date: 2026-08-25 20:30:00 +0800
categories:
  - 编译器技术
tags: [compiler, llvm, clang, ssa, scev]
layout: post
mathjax: true
---

> **LLM/编译器技术深度解析系列 · 第 6 篇 / 共 13 篇**（Part B 主流编译器剖析 · 2/5）
>
> [第 05 篇 GCC 剖析](/2026/08/25/compiler-05-gcc-anatomy/) ← **本篇** → [第 07 篇 TVM 剖析](/2026/08/25/compiler-07-tvm/)

**TL;DR**
* LLVM 把"中间表示 + pass 基础设施"本身做成了产品。本文在本机（Homebrew clang **22.1.0**，arm64）跑通 `clang → opt → llc` 全链路，所有 IR 与汇编均为真实输出。
* 实测高潮：一个朴素的平方和循环 `for(i<n) acc+=i*i`，在 `-O2` 后**循环整体消失**——LLVM 的标量演化引擎（SCEV）把它识别成闭式公式 $\sum_{i=0}^{n-1} i^2 = \frac{n(n-1)(2n-1)}{6}$，并用魔数乘法替代除以 3。这展示了中端优化能达到的理论高度。
* 过程中踩到一个所有人都踩过的坑：`-O0` 产出的函数带 `optnone` 属性，会让后续 `opt` 全部静默失效，需 `-Xclang -disable-O0-optnone` 解除。
* 架构主线：New PassManager 的可配置管线、MC 层拆分使 JIT 复用汇编器、TableGen 的 DSL 生成思想——三者共同构成"基础设施可编程性"的三根柱子。

---

## 目录

- [1. 定位与生态](#1)
- [2. Clang 前端管线](#2)
- [3. 实测全链路：一条循环的消失](#3)
- [4. New PassManager：管线的可编程化](#4)
- [5. MC 层与 ORC JIT](#5)
- [6. TableGen：用 DSL 对抗重复](#6)
- [7. 批判与展望](#7)
- [FAQ](#faq)
- [参考资料](#参考)

<a name="1"></a>
## 1. 定位与生态

2000 年 Chris Lattner 在伊利诺伊大学香槟分校（UIUC）启动 LLVM，关键决策有二：
1. **授权选 BSD 而非 GPL**——商业公司可以自由嵌入、定制、闭源分发，这是后来 Swift/Metal/Rust 编译器都建在 LLVM 上的根本原因；
2. **定位为"基础设施"而非完整编译器**——前端、链接器（lld）、JIT（ORC）、调试器（LLDB）全部模块化。

时至今日 LLVM 已是事实上的编译器工业标准栈：Rust、Swift、Julia、Zig 的前端都对接 LLVM IR；AMDGPU/NVPTX 后端让 OpenCL/SYCL 生态直接受益。本系列写作时本机安装的是 Homebrew LLVM 22.1.0（`clang --version` 实测），按半年一大版的节奏持续演进。

<a name="2"></a>
## 2. Clang 前端管线

Clang 的四段式前端与第 01~02 篇的理论一一对应：

| 阶段 | 输入 → 输出 | 特色 |
|------|------------|------|
| Preprocessor | 字符流 → 预处理 token 流 | 宏展开、条件编译（C 家族特有负担） |
| Parser | token → AST | 递归下降手写（第 01 篇 §3.3 的工业版） |
| Sema | AST → 标注后的 AST | 类型检查、重载决议、模板实例化 |
| CodeGen | AST → LLVM IR | IRGen 以函数为单位流式产出 |

与 GCC 前端的本质区别在于**库化设计**：clang 的每一阶段都是独立可调用的库（libclang、libtooling），IDE 补全、重构工具、静态分析器（clang-tidy）全部构建其上。GCC 直到近年才补上类似的 `libgccjit`，成熟度不可同日而语。

<a name="3"></a>
## 3. 实测全链路：一条循环的消失

测试代码刻意朴素：

```c
int sum_sq(int n) {
    int acc = 0;
    for (int i = 0; i < n; i++) {
        acc += i * i;
    }
    return acc;
}
```

### 3.1 第一步：看 `-O0` 的原始 IR（以及一个必踩的坑）

```bash
clang -O0 -emit-llvm -S sumsq.c -o raw.ll     # 注意：这样会踩坑，见下文
```

`-O0` 下每个变量都被降级为 alloca（栈槽）+ load/store——变量根本没有寄存器身份，这正是第 04 篇 FAQ 里"-O0 变量都在栈上"的原因。但直接对这个文件跑 `opt` 会发现**什么都没发生**：clang 给每个函数加了 `optnone` 属性（意为"跳过一切优化"，保护调试体验）。解法是显式关闭它：

```bash
clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S sumsq.c -o raw.ll
```

### 3.2 第二步：mem2reg——第 02 篇算法的工业版

```bash
opt -passes=mem2reg raw.ll -S -o ssa.ll && sed -n '/define/,$p' ssa.ll | head -32
```

**真实输出**：

```text
define i32 @sum_sq(i32 noundef %0) #0 {
  br label %2

2:                                                ; preds = %7, %1
  %.01 = phi i32 [ 0, %1 ], [ %6, %7 ]
  %.0 = phi i32 [ 0, %1 ], [ %8, %7 ]
  %3 = icmp slt i32 %.0, %0
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = mul nsw i32 %.0, %.0
  %6 = add nsw i32 %.01, %5
  br label %7

7:                                                ; preds = %4
  %8 = add nsw i32 %.0, 1
  br label %2, !llvm.loop !6

9:                                                ; preds = %2
  ret i32 %.01
}
```

逐点对照第 02 篇的手写实现：
* 所有 `alloca/load/store` 对消失，取而代之的是两个 φ 函数（`.01` 是 acc，`.0` 是 i）；
* φ 的操作数格式 `[值, 来自古块]` 与我们手写的 `x4 = phi(x2 from B2, x3 from B3)` 完全同构；
* 循环头的支配边界放置、支配树 DFS 重命名——工业实现就是那套 Cytron 算法的硬化版。
* 细节彩蛋：`mul nsw` 的 `nsw` 表示 "no signed wrap"，这是给优化器的许可声明（溢出即 UB），中端据此才能安全做归纳变量推断——LLVM IR 把 UB 当作优化资源使用是其最独特的设计。

### 3.3 第三步：`-O2`——循环消失了

```bash
opt -passes="default<O2>" raw.ll -S -o o2.ll
```

**真实输出**（完整函数体）：

```text
define i32 @sum_sq(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 0
  br i1 %2, label %.lr.ph.preheader, label %._crit_edge

.lr.ph.preheader:                                 ; preds = %1
  %3 = add nsw i32 %0, -1
  %4 = zext nneg i32 %3 to i33
  %5 = add nsw i32 %0, -2
  %6 = zext i32 %5 to i33
  %7 = mul i33 %4, %6
  %8 = add nsw i32 %0, -3
  %9 = zext i32 %8 to i33
  %10 = mul i33 %7, %9
  %11 = lshr i33 %10, 1
  %12 = trunc nuw i33 %11 to i32
  %13 = mul i32 %12, 1431655766
  %14 = add i32 %0, %13
  %15 = lshr i33 %7, 1
  %16 = trunc nuw i33 %15 to i32
  %17 = mul i32 %16, 3
  %18 = add i32 %14, %17
  %19 = add i32 %18, -1
  br label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph.preheader, %1
  %.07.lcssa = phi i32 [ 0, %1 ], [ %19, %.lr.ph.preheader ]
  ret i32 %.07.lcssa
}
```

没有循环了。发生了三件教科书级别的事：

1. **闭式公式识别**：SCEV（Scalar Evolution）把归纳变量链推导出精确的多项式求和表达式，识别出 $\sum_{i=0}^{n-1} i^2 = \frac{(n-1)\,n\,(2n-1)}{6}$，循环被替换为常数条数的算术。
2. **i33 的防溢出智慧**：注意中间计算升到了 **33 位整数类型**（`zext ... to i33`, `mul i33`）——$n(n-1)(2n-1)$ 在 $n$ 接近 $2^{31}$ 时会溢出 64 位内的部分中间量？不，33 位恰好容纳 $(2^{31})^2$ 量级的中间积再截回，这是精度与成本的精确权衡，由编译器自动完成。
3. **除以 3 的魔数削减**：$\frac{\cdot}{6} = \frac{\cdot}{2}\cdot\frac{1}{3}$，右移一位搞定 `/2`；`/3` 则变成 `mul i32 ..., 1431655766`（即 $0x55555556 = \lceil 2^{32}/3 \rceil$），配合模 $2^{32}$ 截断天然完成"取高位"——经典的除法强度削减（《Hacker's Delight》第 10 章）。

### 3.4 第四步：llc 落到 arm64 汇编

```bash
llc o2.ll -o o2.s
```

**真实输出**（节选）：

```text
_sum_sq:                                ; @sum_sq
	.cfi_startproc
; %bb.0:
	subs	w8, w0, #1
	b.lt	LBB0_2
; %bb.1:                                ; %.lr.ph.preheader
	sub	w9, w0, #2
	umull	x8, w8, w9
	sub	w9, w0, #3
	mul	w9, w8, w9
	lsr	w9, w9, #1
	mov	w10, #21846                     ; =0x5556
	movk	w10, #21845, lsl #16
	madd	w9, w9, w10, w0
	lsr	x8, x8, #1
	add	w8, w8, w8, lsl #1
	add	w8, w9, w8
	sub	w0, w8, #1
	ret
```

指令选择的痕迹清晰可见：`movk` 双步拼出 32 位魔数常量（arm64 没有"加载任意 32 位立即数"的单指令）；`add w8, w8, w8, lsl #1` 一条指令完成 $3x$（对应 IR 里的 `mul i32 %16, 3`）；`madd` 融合乘加。整段约 14 条指令、零分支、零循环——这就是"编译器替你写数学"的具象形态。

<a name="4"></a>
## 4. New PassManager：管线的可编程化

LLVM 的 pass 组织在 13.x 之后全面切换到 New PassManager，三个关键性质：

1. **显式管线对象**：`opt -passes="mem2reg,default<O2>"` 这样的命令行语法之所以存在，是因为管线本身就是一等数据结构（`PassBuilder` 组装 `ModulePassManager`）；
2. **三层分析粒度**：Module / CGSCC（调用图强连通分量）/ Function 各有专属 pass 类别，过程间分析（如内联）运行在 CGSCC 层并可按需重访 SCC——比 GCC 的静态管线灵活一个量级；
3. **分析与变换分离**：analysis 是缓存化的只读信息（如 AliasAnalysis、MemorySSA），pass 显式声明 invalidate 语义——这让增量重编译与并行 pass 成为可能。

对照第 05 篇的结论：GCC 的管线是"编译期固化的产品配置"，LLVM 的管线是"运行时可组装的工具箱"。前者服务稳定发布，后者服务生态创新——JIT 用户尤其受益（ORC 可以随时 build 一个迷你管线热编译单个函数）。

<a name="5"></a>
## 5. MC 层与 ORC JIT

LLVM 后端被切成两半：**CodeGen**（SelectionDAG/GlobalISel → MachineInstr）和 **MC 层**（MachineInstr → MCInst → 目标码/汇编文本）。这个切分的红利归了 JIT：ORC（On-Request Compilation）可以复用 MC 的汇编器直接从内存中的 MachineInstr 产出目标码并链接执行，不需要落盘 `.s` 再调外部汇编器。

ORC 的分层设计（Layer 堆叠：ObjectLinkingLayer ← IRCompileLayer ← OptimizingMaterializationUnit…）支持懒编译、按函数粒度编译、跨模块符号解析——Kaleidoscope 教程的 JIT 系列三章专门讲它[2]。Kubernetes 圈熟悉的 eBPF 编译、各种语言 REPL（包括 clang 自己的 clang-repl）、数据库 UDF 加速，背后都是这条栈。

<a name="6"></a>
## 6. TableGen：用 DSL 对抗重复

LLVM 内部充满"同类信息的海量实例"：数千条目标指令描述、几十个 target 特性、pass 注册元数据……它们的答案是 TableGen：一种声明式 DSL（`.td` 文件）+ 生成器，把结构化知识写成数据而不是 C++ 代码。例如一条 arm64 指令的模式描述包含汇编助记符、编码位型、与 SelectionDAG 节点的匹配规则，全部集中在一条 `def` 里。

方法论价值超过工具价值：**当某类信息出现 N 千次时，先造 DSL 再写实例**。TVM 的 schedule primitive、MLIR 的 ODS（Operation Definition Specification）、乃至 Triton 的 kernel 元编程，都是同一思想的回声。反例也值得记住：DSL 本身会成为新的复杂度中心（TableGen 的报错体验臭名昭著），需要配套文档与 lint 投入。

<a name="7"></a>
## 7. 批判与展望

* **IR 演进的迁移税**：opaque pointers 迁移（指针不再携带元素类型）横跨 LLVM 15~17 才完成，下游项目集体陪跑。单一全局 IR 的每次语义升级都是生态级工程——这正是 MLIR 独立出去搞"多方言、各自生命周期"的直接动因（第 08 篇展开）。
* **单仓库巨石化**：llvm-project monorepo 体量与 CI 时长持续增长，新贡献者上手成本高企。社区以 issue triage 与 release branch 冻结窗口应对，效果一般。
* **对 AI Infra 工程师的定位建议**：把 LLVM 当"黑盒后端"用（Triton/Inductor 视角）足够应付大多数场景；要读源码的话，优先级是 IR 语法 > PassManager > SCEV > SelectionDAG（最后一项除非做硬件后端否则别碰）。

## FAQ

**Q1：LLVM IR 是 SSA，那内存怎么办？load/store 不是破坏 SSA 吗？**
SSA 只约束寄存器值（虚拟寄存器），内存是独立的语义域。指向内存的分析靠 MemorySSA/AliasAnalysis 这类附加结构补足。mem2reg 的本质就是判断哪些 alloca"够简单"可以提升进 SSA 世界，其余留在内存域。

**Q2：想给自己的芯片加后端，从哪里下手？**
两条路：传统路线写 TableGen 描述 + 实现 SelectionDAG/GlobalISel（成本极高，参考 CHERI/RISCV 扩展的工程量）；现代路线在 MLIR 里自建 dialect 并 lowering 到 LLVM IR 复用现有后端（AI 芯片公司的事实标准）。除非 ISA 有张量原语必须定制，选后者。

**Q3：optnone 这个坑还有别的变体吗？**
有。常见兄弟坑：`-O0` 的 IR 没有 `nsw/nuw` 标记导致某些 pass 行为不同；macOS 上默认 triple 带 SDK 版本号导致跨机器复现 diff 困难。做 IR diff 测试时记得固定 `-mtriple` 和优化属性集合（第 12 篇会系统讲测试基建）。

## 参考资料

[1] LLVM Language Reference: https://llvm.org/docs/LangRef.html
[2] LLVM Tutorial（Kaleidoscope 与 ORC JIT 系列）: https://llvm.org/docs/tutorial/
[3] LLVM New Pass Manager 文档: https://llvm.org/docs/NewPassManager.html
[4] LLVM ORC Design and Implementation: https://llvm.org/docs/ORCv2.html
[5] Warren, *Hacker's Delight*, 2nd ed., Chapter 10（除法魔数原理）

---

> **下一篇**：[第 07 篇 TVM 架构剖析](/2026/08/25/compiler-07-tvm/)——Relax 图 IR、TensorIR 调度代数与 MetaSchedule 自动调优：深度学习编译栈的全景解剖。
