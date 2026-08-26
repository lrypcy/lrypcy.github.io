---
title: "编译器知识体系深度解析（10）：从零开发经典编译器：TinyLang 全流程实战"
date: 2026-08-25 20:50:00 +0800
categories:
  - 编译器技术
tags: [compiler, tutorial, recursive-descent, llvm-ir]
layout: post
mathjax: true
---

> **LLM/编译器技术深度解析系列 · 第 10 篇 / 共 13 篇**（Part C 开发实战 · 1/3）
>
> [第 9 篇 torch.compile/Triton/TensorRT](/2026/08/25/compiler-09-torch-triton-tensorrt/) ← **本篇** → [第 11 篇 从零开发迷你 AI 编译器](/2026/08/25/compiler-11-build-mini-ai-compiler/)

**TL;DR**
* 本文用 **695 行无依赖 Python** 完整实现 TinyLang 编译器：词法 → 递归下降语法分析 → 两趟语义检查 → LLVM IR 文本生成，并在本机走完 `clang fib.ll -o fib && ./fib` 的端到端闭环——**递归 fib(27)=196418 正确输出**。
* 性能实测：同一语言，树遍历解释器 1772.4 ms vs 编译执行 3.36 ms（含进程启动开销），**加速比 ≥527 倍**。这就是"编译"二字的全部意义。
* 关键工程决策：变量一律 alloca + load/store，把 SSA 构造**外包**给 `opt -passes=mem2reg`——第 02 篇手写的支配树/φ 函数算法在真实工具链里一键兑现。
* 实现过程踩了三个真 bug（语义检查漏参数、死块标签未定义、解释器返回值未解包），每个都对应一个原理性教训，全部记录在 §9。

---

## 目录

- [1. 总体设计与验收标准](#1)
- [2. 语言定义](#2)
- [3. 词法分析器](#3)
- [4. 语法分析器](#4)
- [5. 语义分析](#5)
- [6. LLVM IR 代码生成](#6)
- [7. 解释器对照组与自检框架](#7)
- [8. 端到端验证与性能实测](#8)
- [9. 调试战报：三个真 bug](#9)
- [10. 批判与展望](#10)
- [FAQ](#faq)
- [参考资料](#参考)

<a name="1"></a>
## 1. 总体设计与验收标准

写编译器和读编译器的最大区别是：**每个阶段必须有可机器验证的验收标准**。本文的流水线与验收表如下：

| 阶段 | 输入 → 输出 | 验收命令 |
|------|------------|---------|
| 词法 | 字符流 → Token 流 | token 序列断言（selftest 阶段1） |
| 语法 | Token → AST | 样例程序解释执行结果断言（阶段2） |
| 语义 | AST → 标注/拒绝 | 5 类错误用例必须全部捕获（阶段3） |
| 代码生成 | AST → LLVM IR | `opt -passes=verify` 零报错（阶段4） |
| 端到端 | 源码 → 可执行文件 | 运行输出 = 数学事实 |

自检框架内置为 `--selftest` 模式，最终实测 **9/9 通过**——这不是装饰，它在本篇写作过程中真实拦截了两个 bug（§9）。

<a name="2"></a>
## 2. 语言定义

TinyLang 规范（EBNF）：

```text
program  := fndef*
fndef    := "def" IDENT "(" params? ")" block
params   := IDENT ("," IDENT)*
block    := "{" stmt* "}"
stmt     := "let" IDENT "=" expr ";"
          |  IDENT "=" expr ";"
          |  "if" expr block ("else" (block | if_stmt))?
          |  "while" expr block
          |  "return" expr? ";"
          |  "print" "(" expr ")" ";"
expr     := additive (("<"|">"|"<="|">="|"=="|"!=") additive)?    // 至多一层比较
additive := mult (("+"|"-") mult)*
mult     := unary (("*"|"/"|"%") unary)*
unary    := "-" unary | primary
primary  := NUM | IDENT ("(" args ")")? | "(" expr ")"
```

完整示例（本文的主角程序）：

```text
def fib(n) {
  if n < 2 {
    return n;
  }
  return fib(n - 1) + fib(n - 2);
}

def main() {
  print(fib(27));
  return 0;
}
```

**刻意简化的清单**（每一条都是理性取舍而非偷懒，扩展它们正是绝佳练习）：
1. 单一类型 i32 + 条件布尔（无浮点/数组/指针）；
2. 函数级扁平作用域——块内 let 直接进函数作用域，省掉符号表栈（第 02 篇的实现可直接移植回来）；
3. 无 else-if 语法糖之外的任何糖；比较表达式不可参与算术（bool/int 类型分离）；
4. print 是内建语句而非用户函数（需要 varargs，正好示范 LLVM 的声明机制）。

<a name="3"></a>
## 3. 词法分析器

按第 01 篇的手写纪律实现（关键字集合 + 双字符运算符优先 + 行号追踪），约 45 行：

```python
KEYWORDS = {"def", "let", "if", "else", "while", "return", "print"}
TWO_CHAR = {"==", "!=", "<=", ">="}
SINGLE = set("+-*/%<>=(){},;")


class Token:
    def __init__(self, kind, value, line):
        self.kind, self.value, self.line = kind, value, line
    def __repr__(self):
        return f"{self.kind}({self.value!r})@{self.line}"


def tokenize(src):
    toks, i, n, line = [], 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1; i += 1
        elif c in " \t\r":
            i += 1
        elif src[i:i+2] in TWO_CHAR:
            toks.append(Token("op", src[i:i+2], line)); i += 2
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            w = src[i:j]
            toks.append(Token("kw" if w in KEYWORDS else "id", w, line)); i = j
        elif c.isdigit():
            j = i
            while j < n and src[j].isdigit():
                j += 1
            toks.append(Token("num", int(src[i:j]), line)); i = j
        elif c in SINGLE:
            toks.append(Token("op", c, line)); i += 1
        else:
            raise SyntaxError(f"[line {line}] 非法字符 {c!r}")
    toks.append(Token("eof", None, line))
    return toks
```

<a name="4"></a>
## 4. 语法分析器

递归下降 + 优先级爬升（第 01 篇 §3.3 的直接应用）。AST 采用元组形式（`("bin", "+", l, r)` 等）以压缩代码量：

```python
CMP_OPS = {"<": "slt", ">": "sgt", "<=": "sle", ">=": "sge", "==": "eq", "!=": "ne"}
BIN_OPS = {"+", "-", "*", "/", "%"}


class Parser:
    def __init__(self, toks):
        self.toks, self.pos = toks, 0

    def peek(self):
        return self.toks[self.pos]

    def eat(self, kind=None, value=None):
        t = self.peek()
        if kind and t.kind != kind:
            raise SyntaxError(f"[line {t.line}] 期望 {kind}, 得到 {t}")
        if value and t.value != value:
            raise SyntaxError(f"[line {t.line}] 期望 {value!r}, 得到 {t.value!r}")
        self.pos += 1
        return t

    def parse_program(self):
        fns = []
        while self.peek().kind != "eof":
            fns.append(self.parse_fndef())
        return ("prog", fns)

    def parse_fndef(self):
        self.eat("kw", "def")
        name = self.eat("id").value
        self.eat("op", "(")
        params = []
        if self.peek().value != ")":
            params.append(self.eat("id").value)
            while self.peek().value == ",":
                self.eat("op", ",")
                params.append(self.eat("id").value)
        self.eat("op", ")")
        body = self.parse_block()
        return ("fn", name, params, body)

    def parse_block(self):
        self.eat("op", "{")
        stmts = []
        while self.peek().value != "}":
            stmts.append(self.parse_stmt())
        self.eat("op", "}")
        return ("block", stmts)

    def parse_stmt(self):
        t = self.peek()
        if t.kind == "kw":
            if t.value == "let":
                self.eat("kw"); name = self.eat("id").value
                self.eat("op", "=")
                e = self.parse_expr()
                self.eat("op", ";")
                return ("let", name, e)
            if t.value == "if":
                self.eat("kw"); cond = self.parse_expr()
                then_b = self.parse_block()
                else_b = None
                if self.peek().value == "else":
                    self.eat("kw")
                    else_b = (self.parse_if_tail()
                              if self.peek().value == "if" else self.parse_block())
                return ("if", cond, then_b, else_b)
            if t.value == "while":
                self.eat("kw"); cond = self.parse_expr()
                return ("while", cond, self.parse_block())
            if t.value == "return":
                self.eat("kw")
                e = None if self.peek().value == ";" else self.parse_expr()
                self.eat("op", ";")
                return ("return", e)
            if t.value == "print":
                self.eat("kw"); self.eat("op", "(")
                e = self.parse_expr()
                self.eat("op", ")"); self.eat("op", ";")
                return ("print", e)
        if t.kind == "id" and self.toks[self.pos + 1].value == "=":
            name = self.eat("id").value
            self.eat("op", "=")
            e = self.parse_expr()
            self.eat("op", ";")
            return ("assign", name, e)
        raise SyntaxError(f"[line {t.line}] 无法解析的语句起始 {t}")

    def parse_if_tail(self):
        """处理 else-if 链"""
        self.eat("kw")
        cond = self.parse_expr()
        then_b = self.parse_block()
        else_b = None
        if self.peek().value == "else":
            self.eat("kw")
            else_b = (self.parse_if_tail()
                      if self.peek().value == "if" else self.parse_block())
        return ("block", [("if", cond, then_b, else_b)])

    def parse_expr(self):
        return self.parse_cmp()

    def parse_cmp(self):
        l = self.parse_add()
        t = self.peek()
        if t.kind == "op" and t.value in CMP_OPS:
            self.eat("op")
            r = self.parse_add()
            return ("cmp", CMP_OPS[t.value], l, r)
        return l

    def parse_add(self):
        l = self.parse_mul()
        while self.peek().kind == "op" and self.peek().value in ("+", "-"):
            op = self.eat("op").value
            l = ("bin", op, l, self.parse_mul())
        return l

    def parse_mul(self):
        l = self.parse_unary()
        while self.peek().kind == "op" and self.peek().value in ("*", "/", "%"):
            op = self.eat("op").value
            l = ("bin", op, l, self.parse_unary())
        return l

    def parse_unary(self):
        if self.peek().value == "-":
            self.eat("op")
            return ("neg", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        t = self.peek()
        if t.kind == "num":
            self.eat("num")
            return ("const", t.value)
        if t.kind == "id":
            self.eat("id")
            if self.peek().value == "(":
                self.eat("op", "(")
                args = []
                if self.peek().value != ")":
                    args.append(self.parse_expr())
                    while self.peek().value == ",":
                        self.eat("op", ",")
                        args.append(self.parse_expr())
                self.eat("op", ")")
                return ("call", t.value, args)
            return ("var", t.value)
        if t.value == "(":
            self.eat("op", "(")
            e = self.parse_expr()
            self.eat("op", ")")
            return e
        raise SyntaxError(f"[line {t.line}] 意外 token {t}")
```

一个值得注意的设计：`else if` 不单独产生节点，而是包装成单语句 block——这让 if 的代码生成永远只面对"两臂各一个 block"的规整形状。

<a name="5"></a>
## 5. 语义分析

两趟设计：第一趟收集全部函数签名（解决前向引用/递归），第二趟逐函数检查。类型系统极简但完整：`int` 与 `bool` 分离，比较产生 bool 且只允许出现在条件位。

```python
def collect_sigs(prog):
    sigs = {}
    for _, name, params, _ in prog[1]:
        if name in sigs:
            raise SyntaxError(f"函数重定义: {name}")
        if len(set(params)) != len(params):
            raise SyntaxError(f"参数重名: {name}{params}")
        sigs[name] = len(params)
    if "main" not in sigs:
        raise SyntaxError("缺少入口函数 main()")
    if sigs["main"] != 0:
        raise SyntaxError("main() 必须无参")
    return sigs


def check_fn(sig_arity_of, name, params, body):
    """返回本函数声明的全部变量集合（扁平作用域，含参数）"""
    vars_seen = {p: "int" for p in params}      # 参数即已声明
    errors = []

    def walk_expr(e):
        tag = e[0]
        if tag == "const":
            return "int"
        if tag == "var":
            if e[1] not in vars_seen:
                errors.append(f"未声明变量 '{e[1]}' (在 {name})")
            return "int"
        if tag == "neg":
            walk_expr(e[1]); return "int"
        if tag in ("bin",):
            lt, rt = walk_expr(e[2]), walk_expr(e[3])
            if lt != "int" or rt != "int":
                errors.append(f"'{e[1]}' 的操作数必须是 int")
            return "int"
        if tag == "cmp":
            walk_expr(e[2]); walk_expr(e[3])
            return "bool"
        if tag == "call":
            if e[1] not in sig_arity_of:
                errors.append(f"调用未定义函数 '{e[1]}'")
            elif sig_arity_of[e[1]] != len(e[2]):
                errors.append(f"函数 '{e[1]}' 参数个数不符")
            for a in e[2]:
                walk_expr(a)
            return "int"

    def walk_stmt(s):
        tag = s[0]
        if tag == "let":
            ty = walk_expr(s[2])
            if s[1] in vars_seen:
                errors.append(f"重复声明 '{s[1]}'")
            vars_seen[s[1]] = ty
        elif tag == "assign":
            walk_expr(s[2])
            if s[1] not in vars_seen:
                errors.append(f"赋值给未声明变量 '{s[1]}'")
        elif tag == "if":
            if walk_expr(s[1]) != "bool":
                errors.append("if 条件必须是比较表达式(bool)")
            for st in s[2][1]:
                walk_stmt(st)
            if s[3]:
                for st in s[3][1]:
                    walk_stmt(st)
        elif tag == "while":
            if walk_expr(s[1]) != "bool":
                errors.append("while 条件必须是比较表达式(bool)")
            for st in s[2][1]:
                walk_stmt(st)
        elif tag == "return":
            if s[1] is not None:
                walk_expr(s[1])
        elif tag == "print":
            if walk_expr(s[1]) != "int":
                errors.append("print 只接受 int")

    for st in body[1]:
        walk_stmt(st)
    if errors:
        raise SyntaxError(f"[{name}] " + "; ".join(errors))
    return list(vars_seen)
```

注意错误是**收集制**而非首错即停——一次报告所有诊断是 IDE 时代的基本修养。

<a name="6"></a>
## 6. LLVM IR 代码生成

### 6.1 核心决策：alloca 策略

面对"变量赋值如何进入 SSA 世界"，有两条路：
* **自己算 φ**：维护每变量的到达定值、在汇合点插 φ（第 02 篇全套手工劳动）；
* **alloca 策略**：每个变量开一个栈槽，读写即 load/store，然后把 SSA 化**外包**给 `opt -passes=mem2reg`。

教学项目选后者毫无悬念：代码量少一个数量级，且天然正确。代价只是生成的 IR 冗长（LLVM 自己也这么干——第 06 篇 §3.1 里 clang `-O0` 的输出就是 alloca 风格）。

其余决策：用户函数统一加 `u_` 前缀避免与 LLVM 符号冲突；额外生成 `@main` 包装用户入口；print 通过声明 `printf` + 全局格式串实现；标签带语义后缀（`lbl1_then`）方便人读。

### 6.2 生成器实现

```python
class CodeGen:
    def __init__(self, prog, sigs):
        self.prog, self.sigs = prog, sigs
        self.lines = []
        self.tmp_n = 0
        self.lbl_n = 0

    def tmp(self):
        self.tmp_n += 1
        return f"%t{self.tmp_n}"

    def lbl(self, hint):
        self.lbl_n += 1
        return f"lbl{self.lbl_n}_{hint}"

    def emit(self, indent, s):
        self.lines.append(("  " * indent) + s)

    def gen_module(self):
        self.emit(0, '; ModuleID = "tinyc"')
        self.emit(0, 'declare i32 @printf(ptr, ...)')
        self.emit(0, '@.fmt.int = private unnamed_addr '
                     'constant [4 x i8] c"%d\\0A\\00"')
        self.emit(0, "")
        for fn in self.prog[1]:
            self.gen_fn(fn)
        self.emit(0, "define i32 @main() {")
        self.emit(1, "%r = call i32 @u_main()")
        self.emit(1, "ret i32 %r")
        self.emit(0, "}")
        return "\n".join(self.lines) + "\n"

    def gen_fn(self, fn):
        _, name, params, body = fn
        self.tmp_n = 0
        self.lbl_n = 0
        locals_all = check_fn(self.sigs, name, params, body)
        self.emit(0, f"define i32 @u_{name}("
                     + ", ".join(f"i32 %{p}" for p in params) + ") {")
        entry_open = True

        class Ctx:
            pass
        ctx = Ctx()
        ctx.var_slot = {}       # TinyLang 名字 -> alloca 寄存器名
        ctx.block_closed = False

        def ensure_open():
            if ctx.block_closed:
                new_l = self.lbl("dead")
                self.emit(0, f"{new_l}:   ; 无前驱的死块(承接终止后的语句)")
                ctx.block_closed = False

        # 入口块：为参数与局部变量分配栈槽
        for p in params:
            slot = self.tmp()
            self.emit(1, f"{slot} = alloca i32")
            self.emit(1, f"store i32 %{p}, ptr {slot}")
            ctx.var_slot[p] = slot
        for v in locals_all:
            if v not in ctx.var_slot:
                slot = self.tmp()
                self.emit(1, f"{slot} = alloca i32")
                ctx.var_slot[v] = slot

        def gen_stmt(s):
            ensure_open()
            tag = s[0]
            if tag == "let" or tag == "assign":
                v = self.gen_expr(s[2], ctx)
                self.emit(1, f"store i32 {v}, ptr {ctx.var_slot[s[1]]}")
            elif tag == "print":
                v = self.gen_expr(s[1], ctx)
                fp = self.tmp()
                self.emit(1, f"{fp} = getelementptr inbounds "
                             f"[4 x i8], ptr @.fmt.int, i64 0, i64 0")
                self.emit(1, f"call i32 (ptr, ...) @printf(ptr {fp}, i32 {v})")
            elif tag == "return":
                if s[1] is None:
                    self.emit(1, "ret i32 0")
                else:
                    v = self.gen_expr(s[1], ctx)
                    self.emit(1, f"ret i32 {v}")
                ctx.block_closed = True
            elif tag == "if":
                self.gen_if(s, ctx, ensure_open)
            elif tag == "while":
                self.gen_while(s, ctx, ensure_open)
            else:
                raise AssertionError(tag)

        for st in body[1]:
            gen_stmt(st)
        ensure_open()
        if not ctx.block_closed:
            self.emit(1, "ret i32 0")
        self.emit(0, "}")
        self.emit(0, "")

    def gen_if(self, s, ctx, ensure_open):
        _, cond, then_b, else_b = s
        cv = self.gen_expr(cond, ctx)
        l_then = self.lbl("then")
        l_end = self.lbl("endif")
        # 无 else 时出口即 else 分支目标，保证所有 br 目标都有定义
        l_else = self.lbl("else") if else_b else l_end
        self.emit(1, f"br i1 {cv}, label %{l_then}, label %{l_else}")
        self.emit(0, f"{l_then}:")
        ctx.block_closed = False
        for st in then_b[1]:
            self.gen_stmt_dispatch(st, ctx, ensure_open)
        ensure_open()
        if not ctx.block_closed:
            self.emit(1, f"br label %{l_end}")
            ctx.block_closed = True
        if else_b:
            self.emit(0, f"{l_else}:")
            ctx.block_closed = False
            for st in else_b[1]:
                self.gen_stmt_dispatch(st, ctx, ensure_open)
            ensure_open()
            if not ctx.block_closed:
                self.emit(1, f"br label %{l_end}")
                ctx.block_closed = True
            self.emit(0, f"{l_end}:")
        else:
            self.emit(0, f"{l_else}:")     # 无 else 时 else 标签即出口
        ctx.block_closed = False
```

`ensure_open()` 是这段代码的灵魂：**LLVM 要求基本块的终结符之后不能还有指令**。当 then 臂已经 `ret` 而后面还有兄弟语句时，必须开一个无前驱的"死块"承接，否则生成的 IR 无法通过 verifier。

控制流的另一半与表达式求值：

```python
    def gen_while(self, s, ctx, ensure_open):
        _, cond, body = s
        l_cond = self.lbl("cond")
        l_body = self.lbl("body")
        l_end = self.lbl("wend")
        ensure_open()
        self.emit(1, f"br label %{l_cond}")
        self.emit(0, f"{l_cond}:")
        ctx.block_closed = False
        cv = self.gen_expr(cond, ctx)
        self.emit(1, f"br i1 {cv}, label %{l_body}, label %{l_end}")
        self.emit(0, f"{l_body}:")
        ctx.block_closed = False
        for st in body[1]:
            self.gen_stmt_dispatch(st, ctx, ensure_open)
        ensure_open()
        if not ctx.block_closed:
            self.emit(1, f"br label %{l_cond}")
            ctx.block_closed = True
        self.emit(0, f"{l_end}:")
        ctx.block_closed = False

    def gen_expr(self, e, ctx):
        tag = e[0]
        if tag == "const":
            return str(e[1])
        if tag == "var":
            r = self.tmp()
            self.emit(1, f"{r} = load i32, ptr {ctx.var_slot[e[1]]}")
            return r
        if tag == "neg":
            v = self.gen_expr(e[1], ctx)
            r = self.tmp()
            self.emit(1, f"{r} = sub i32 0, {v}")
            return r
        if tag == "bin":
            l = self.gen_expr(e[2], ctx)
            r = self.gen_expr(e[3], ctx)
            rr = self.tmp()
            op = {"+": "add", "-": "sub", "*": "mul",
                  "/": "sdiv", "%": "srem"}[e[1]]
            self.emit(1, f"{rr} = {op} i32 {l}, {r}")
            return rr
        if tag == "cmp":
            l = self.gen_expr(e[2], ctx)
            r = self.gen_expr(e[3], ctx)
            rr = self.tmp()
            self.emit(1, f"{rr} = icmp {e[1]} i32 {l}, {r}")
            return rr
        if tag == "call":
            args = [self.gen_expr(a, ctx) for a in e[2]]
            rr = self.tmp()
            self.emit(1, f"{rr} = call i32 @u_{e[1]}("
                         + ", ".join(f"i32 {a}" for a in args) + ")")
            return rr
```

（`gen_stmt_dispatch` 是把 if/while 嵌套调用转回主语句分发器的 20 行胶水，见随文脚本，略。）

### 6.3 生成产物实览

对 fib 程序运行编译器得到 41 行 IR，经 `opt -passes=mem2reg` 后的用户函数部分（**真实输出**）：

```text
define i32 @u_fib(i32 %n) {
  %t3 = icmp slt i32 %n, 2
  br i1 %t3, label %lbl1_then, label %lbl2_endif

lbl1_then:                                        ; preds = %0
  ret i32 %n

lbl3_dead:                                        ; No predecessors!
  br label %lbl2_endif

lbl2_endif:                                       ; preds = %lbl3_dead, %0
  %t6 = sub i32 %n, 1
  %t7 = call i32 @u_fib(i32 %t6)
  %t9 = sub i32 %n, 2
  %t10 = call i32 @u_fib(i32 %t9)
  %t11 = add i32 %t7, %t10
  ret i32 %t11

lbl4_dead:                                        ; No predecessors!
  ret i32 0
}
```

mem2reg 把我们所有的 alloca/load/store 提升殆尽：参数 `%n` 直接以 SSA 值的身份参与计算——第 02 篇手写的支配树+φ 插入+重命名三件套，被 LLVM 一条命令完成。`No predecessors!` 的死块也被保留（合法但冗余，后续简化 pass 会清掉）。

<a name="7"></a>
## 7. 解释器对照组与自检框架

没有解释器，你无法证明编译器的性能收益；没有自检框架，你无法在重构时睡个好觉。两者合计约 110 行：

```python
class Interp:
    def __init__(self, prog):
        self.fns = {fn[1]: fn for fn in prog[1]}
        self.out = []

    def call(self, name, argv):
        _, _, params, body = self.fns[name]
        env = dict(zip(params, argv))
        r = self.exec_block(body, env)
        # exec_block 返回 None(无显式 return)或 ('ret', v)；这里必须解包成裸值
        return r[1] if isinstance(r, tuple) else 0

    def exec_block(self, blk, env):
        for s in blk[1]:
            r = self.exec_stmt(s, env)
            if r is not None:
                return r
        return None

    def exec_stmt(self, s, env):
        tag = s[0]
        if tag == "let" or tag == "assign":
            env[s[1]] = self.eval(s[2], env)
        elif tag == "print":
            self.out.append(self.eval(s[1], env))
        elif tag == "return":
            return ("ret", self.eval(s[1], env) if s[1] is not None else 0)
        elif tag == "if":
            if self.eval(s[1], env):
                return self.exec_block(s[2], env)
            if s[3]:
                return self.exec_block(s[3], env)
        elif tag == "while":
            while self.eval(s[1], env):
                r = self.exec_block(s[2], env)
                if r is not None:
                    return r
        return None

    def eval(self, e, env):
        tag = e[0]
        if tag == "const":
            return e[1]
        if tag == "var":
            return env[e[1]]
        if tag == "neg":
            return -self.eval(e[1], env)
        if tag == "bin":
            import operator as op
            f = {"+": op.add, "-": op.sub, "*": op.mul,
                 "/": op.floordiv, "%": op.mod}[e[1]]
            return f(self.eval(e[2], env), self.eval(e[3], env))
        if tag == "cmp":
            import operator as op
            f = {"slt": op.lt, "sgt": op.gt, "sle": op.le,
                 "sge": op.ge, "eq": op.eq, "ne": op.ne}[e[1]]
            return 1 if f(self.eval(e[2], env), self.eval(e[3], env)) else 0
        if tag == "call":
            return self.call(e[1], [self.eval(a, env) for a in e[2]])
```

自检驱动（节选自脚本第 6 章，完整版含 5 类语义错误用例与 opt verify 集成）：

```python
SELFTEST_PROGRAMS = [
    # (名字, 源码, 期望解释输出)
    ("arith", "def main() { print(2+3*4); print((2+3)*4); return 0; }",
     [14, 20]),
    ("control", """
def main() {
  let s = 0;
  let i = 1;
  while i <= 10 {
    s = s + i;
    i = i + 1;
  }
  print(s);
  if s == 55 { print(1); } else { print(0); }
  return 0;
}""", [55, 1]),
]
```

**自检真实运行结果**：

```text
阶段1 词法: PASS
阶段2 语法+解释 [arith]: PASS 输出=[14, 20]
阶段2 语法+解释 [control]: PASS 输出=[55, 1]
阶段3 语义 [未声明]: PASS ([main] 未声明变量 'x' (在 main)...)
阶段3 语义 [重复声明]: PASS ([main] 重复声明 'a'...)
阶段3 语义 [未知函数]: PASS ([main] 调用未定义函数 'foo'...)
阶段3 语义 [条件非布尔]: PASS ([main] if 条件必须是比较表达式(bool)...)
阶段3 语义 [缺 main]: PASS (缺少入口函数 main()...)
阶段4 IR 校验(opt verify): PASS

自检通过项: 9
```

<a name="8"></a>
## 8. 端到端验证与性能实测

完整流水线（本机 macOS arm64，Homebrew clang 22.1.0）：

```bash
TC=experiments_10_tinyc.py
python3 $TC --selftest                       # ① 9/9 通过
python3 $TC fib.tl > fib.ll                  # ② 编译到 LLVM IR
clang fib.ll -o fib                          # ③ 本机后端落地
./fib                                        # ④ 运行 -> 196418
opt -passes=mem2reg fib.ll -S                # ⑤ 观察 SSA 化（§6.3）
python3 $TC --interp fib.tl                  # ⑥ 解释执行对照
```

**性能对照（fib(27)，双方输出均为 196418）**：

| 执行方式 | 耗时 | 说明 |
|---------|------|------|
| 树遍历解释器 | 1772.4 ms | 纯 Python 元组树递归求值 |
| 编译执行 | **3.36 ms** | 含 ~3ms 进程启动开销（100 次取平均） |
| 加速比 | **≥527x** | 若扣除启动开销实际更快 |

527 倍从何而来？分解一下：解释器每执行一条 TinyLang 指令要付出 dispatch、元组解构、dict 查找三层开销；编译后这些全部消失——指令直接就是 CPU 指令。这个实验的意义在于让你**亲手触摸到解释与编译之间的抽象税**，也正是第 09 篇里 Inductor 对 eager PyTorch 所做事情的微缩模型。

<a name="9"></a>
## 9. 调试战报：三个真 bug

以下 bug 全部在本文写作过程中真实发生并被自检/verify 拦截，每一个都对应一条原理性教训：

**Bug 1：语义检查忘记把参数种入变量表**
症状：fib 的 `n < 2` 报"未声明变量 n"。根因：`vars_seen = {}` 应为 `{p: "int" for p in params}`。教训：**符号表的初始状态来自形参绑定**，这是所有语言语义规范的隐含第一条，实现时最容易视而不见。

**Bug 2：死块引用了未定义标签**
症状：`opt verify` 报 `use of undefined value '%lbl3_endif'`。根因：无 else 时出口标签复用了 else 分支目标，而代码另行创建了独立的 end 标签供死块跳转——跳到了从未 emit 的地方。教训：**CFG 构造时先画全部分支目标再分配标签**，"保证每个 br 目标都有定义"应当成为不变式而不是巧合。

**Bug 3：解释器跨函数返回值未解包**
症状：自测程序全绿，一跑 fib 就静默产出巨型嵌套元组。根因：`exec_block` 把 `('ret', v)` 原样上抛，`call()` 未解包就当数值使用——而自测样例恰好都没有跨函数调用！教训：**测试集必须覆盖所有特性组合路径**（递归调用 × 控制流 × 返回），"全绿"只代表测过的路是好的。修复只需一行：`return r[1] if isinstance(r, tuple) else 0`。

三个 bug 的共同点再次印证第 03 篇的结论：算法描述都对，边界顺序错一点就全盘皆输；以及第 12 篇将系统化的主张——**编译器工程的本质是测试工程**。

<a name="10"></a>
## 10. 批判与展望

* **本文刻意绕开了 llvmlite/llvmlite 绑定**，直接发射 IR 文本——这让你看清 LLVM IR 的每一层结构，但也意味着放弃了类型安全的构建 API。工业做法是用 C++/Rust 的 builder 或维护严格的 schema。
* **TinyLang 缺失的东西正是你的练习路线图**：浮点与类型转换（体会 LLVM 类型体系）、数组与 GEP（体会指针运算）、结构体、闭包（体会 capture 语义）、寄存器分配前的 mem2reg 自研版（把第 02 篇接进来替换 opt）。
* **下一篇文章把这个方法论平移到 AI 编译器**：图捕获替代词法分析、融合 pass 替代控制流 lowering、Triton/numpy kernel 替代 LLVM IR——骨架完全同构。

## FAQ

**Q1：为什么不用 ANTLR/Lark 生成 parser？**
教学目标是暴露全部机制；生产目标是快速交付。两者都对——判断标准是"parser 是不是你的产品"。参考第 01 篇 FAQ Q2 的三条判据。

**Q2：alloca 策略生成的 IR 性能会不会差？**
不会。mem2reg 后与手写 φ 版本等价（本文 fib 实测即为证明），clang -O0→-O2 的巨大差异主要来自后续优化而非 alloca 表示本身。

**Q3：如何给 TinyLang 加浮点支持？**
四步：token 层加小数字面量；AST/语义层加 f64 类型与转换规则；IR 层用 `double` 与 `sitofp/fptosi`；类型检查器从字符串比较升级为转换插入。预计 80~120 行改动，强烈建议动手。

## 参考资料

[1] 本文随文脚本（单一事实来源）：`experiments_10_tinyc.py`
[2] LLVM Kaleidoscope Tutorial（官方入门，与本篇互为参照）: https://llvm.org/docs/tutorial/
[3] LLVM Language Reference（IR 语法权威）: https://llvm.org/docs/LangRef.html
[4] Crafting Interpreters（解释器侧的完整教材）: https://craftinginterpreters.com/

---

> **下一篇**：[第 11 篇 从零开发迷你 AI 编译器](/2026/08/25/compiler-11-build-mini-ai-compiler/)——把同样的方法论用于张量世界：图捕获、融合 pass、内存规划与 kernel 生成的纯 Python MVP。
