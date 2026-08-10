---
title: "xDSL 编译器体系深入剖析：纯 Python 重写的 MLIR 精神实现"
date: 2026-08-10 12:00:00 +0800
categories:
  - AI Infra
tags: [xdsl, 编译器, mlir, compiler]
layout: post
---

> **TL;DR**
>
> * **xDSL 不是 MLIR 的 Python 绑定**，而是用纯 Python 重新实现 MLIR 的核心抽象层（Operation / SSAValue / Region / Block / Attribute / Type），并在此基础上构建了完整的方言定义系统（IRDL + PyRDL）、声明式装配格式（`assembly_format`）、模式重写引擎与 Pass 管线。
> * **体系核心是三大引擎**：① IRDL 约束系统——`AttrConstraint` 的 `verify → variables → infer` 三阶段生命周期支撑泛型类型推断；② 声明式装配格式——把手写 parser/printer 的工作量降低约 80%；③ worklist 驱动的贪婪重写引擎——`PatternRewriteWalker` 保证重写迭代收敛。
> * **适用人群**：AI Infra 工程师、编译器研究者、DSL 设计者。本文所有代码基于 **xDSL 0.63** 实测可运行。

```mermaid
flowchart TD
    subgraph L1["L1 用户层：自定义方言"]
        A1["PyRDL Python Frontend"]
        A2["@irdl_op_definition / @irdl_attr_definition"]
        A3["assembly_format 声明式装配"]
    end
    subgraph L2["L2 约束层：IRDL Constraint System"]
        B1["AttrConstraint / AnyOf / AllOf / ParamAttrConstraint"]
        B2["ConstraintContext 变量绑定"]
        B3["TypeVar → 泛型类型推断"]
    end
    subgraph L3["L3 IR 层：SSA 数据模型"]
        C1["Operation / SSAValue / OpResult"]
        C2["Region / Block 嵌套作用域"]
        C3["Attribute / Type 不可变元数据"]
    end
    subgraph L4["L4 变换层：Pass & Rewrite"]
        D1["PatternRewriter + Worklist"]
        D2["GreedyRewritePatternApplier"]
        D3["PassPipeline / ModulePass"]
    end
    subgraph L5["L5 序列化层：Parser / Printer"]
        E1["MLIR 兼容 Lexer / Parser"]
        E2["FormatProgram 声明式装配程序"]
        E3["AssemblyPrinter 后端代码输出"]
    end
    L1 --> L2 --> L3 --> L4 --> L5
```

---

## 1. 定位：xDSL 在编译器生态中的坐标

### 1.1 不是绑定，是"精神移植"

MLIR 提供了一套**多级 IR 基础设施**：同一套 `Operation/Type/Attribute` 抽象可以承载从高层 DSL（Linalg、Triton）到底层指令（LLVM、RISC-V）的任意方言，配合 Dialect Conversion 与 Pass 管线实现渐进式下降。但其工程实现是 C++ + TableGen（ODS），对研究者、教学场景和快速原型并不友好：改一行方言定义要重新编译，构建链让入门成本变得很高。

xDSL 的目标是**用纯 Python 复刻这套抽象**：

- IR 数据模型与 MLIR 一一对应（`Operation` 的 `operands/results/regions/attributes/properties` 五元组）；
- 方言定义用 Python 装饰器（PyRDL）而非 TableGen，运行时即可生效，无需编译；
- 自带 MLIR 文本格式的 parser/printer，可以读写大部分 MLIR 语法。

```mermaid
flowchart LR
    A["MLIR: C++ + TableGen（编译时）"] -->|"抽象移植"| B["xDSL: Python + 装饰器（运行时）"]
    A --> C["IR 数据模型: Operation/Region/Block/Attribute"]
    B --> C
    C --> D["方言生态: arith / func / memref / linalg / riscv ..."]
    D --> E["变换体系: canonicalize / 方言转换 / 循环优化"]
    D --> F["后端: AssemblyPrinter → RISC-V / x86 / CSL 汇编"]
```

### 1.2 为什么值得深入研究

1. **可读性**：整个核心代码量级远小于 MLIR C++ 代码库，且是 Python——适合把"编译器是怎么把方言一步步降下去"这件事读透。
2. **可运行**：`pip install xdsl` 即可跑起完整示例，改方言定义无需编译。
3. **研究价值**：xDSL 的 IRDL（IR 定义语言）让方言成为**一等的可序列化数据**，方言可以动态加载、跨进程传输，甚至用 IR 程序（PDL-interp）描述重写规则。
4. **教学价值**：TensorIR / MLIR 中的概念（约束、模式、转换）在 xDSL 里都有极短的 Python 实现，是最佳的学习标本。

---

## 2. IR 核心抽象：SSA 数据模型

### 2.1 五元组与所有权模型

xDSL 的 `Operation` 与 MLIR 对齐，核心是五元组：

| 组成 | 含义 | 可变性 |
|------|------|--------|
| `operands` | SSA 操作数（`OpResult` / `BlockArgument`） | 可改（引用的 SSA 值） |
| `results` | 结果值（`OpResult`），可被其他操作引用 | 创建时固定 |
| `regions` | 嵌套区域（`Region` 包含有序 `Block`） | 内部操作可改 |
| `attributes` | 附加元数据（任意 `Attribute`） | Pass 可修改 |
| `properties` | 操作核心配置（强类型，如函数签名） | 创建后不可变 |

关键设计：

- **SSA 单赋值**：每个值只有一个定义点（operation 或 block argument），`OpResult` 通过 `uses` 记录自己的使用列表，形成双向引用；
- **Region 不可变访问**：`Region` 是个有序 `Block` 序列，块内操作通过 `BlockOps` 序列访问；父操作持有 Region，但 Region 不能反向修改父操作的结构（通过 `detach` / `replace` 显式操作）；
- **Attribute/Type 不可变**：作为元数据附着在操作上，保证共享安全（同一个 `TensorType` 对象可被千万操作引用）。

### 2.2 构建一段 IR

```python
from xdsl.context import Context
from xdsl.ir import Block, Region
from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.dialects import arith

ctx = Context()

# 方式一：直接构造 + 插入点
c1 = arith.ConstantOp.from_int_and_width(10, i32)
c2 = arith.ConstantOp.from_int_and_width(20, i32)
add = arith.AddiOp(c1, c2)          # %2 = arith.addi %0, %1 : i32
m = ModuleOp([c1, c2, add])
print(m)
```

输出：

```mlir
builtin.module {
  %0 = arith.constant 10 : i32
  %1 = arith.constant 20 : i32
  %2 = arith.addi %0, %1 : i32
}
```

### 2.3 关于 `arith.AddiOp(c1, c2)` 的构造魔法

内置二元操作通常定义了便捷 `__init__`（自动把操作对象映射到 `results[0]`）；而**自定义操作**没有这个便捷层，必须用 `Operation.create` 显式声明操作数 / 结果类型：

```python
z = TensorZeroOp.create(result_types=[tt])
x = TensorAddOp.create(
    operands=[z.results[0], z.results[0]],   # 要传 SSAValue，不是 Operation
    result_types=[tt],
)
```

这是 0.63 版本的一个高频踩坑点：`create()` 返回 **Operation**，操作数位置必须取 `.results[0]`。

---

## 3. IRDL 与 PyRDL：让方言"自描述"

### 3.1 什么是 IRDL

IRDL（IR Definition Language）是 xDSL 的**方言定义语言**——用 IR 自身来描述方言。也就是说，"方言的定义"本身是一段可解析、可验证、可传输的 IR（`irdl` dialect）。这与 MLIR 的 ODS（TableGen）有本质区别：ODS 是编译期的代码生成器，IRDL 是运行时的数据。

```
我们定义方言 D 的操作 op：              IRDL 表示（也是一段 IR）：
  op 有 1 个 operand、1 个 result        irdl.operation {
    operand 类型是 tensor<?xf32>           irdl.operands {
                                            irdl.parametric "tensor" [ ... ]
                                          }
                                        }
```

### 3.2 PyRDL：装饰器前端

```python
from xdsl.irdl import (
    irdl_op_definition, IRDLOperation, operand_def, result_def,
)

@irdl_op_definition
class TensorAddOp(IRDLOperation):
    name = "tensor.add"
    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)
```

`@irdl_op_definition` 是**元编程**：在类定义时通过 `__init_subclass__` 钩子收集字段，自动完成：

1. 遍历 MRO 收集所有基类的 IRDL 字段（`OperandDef` / `ResultDef` / `AttrDef` / `PropertyDef` / `RegionDef` / `Traits`）；
2. 处理泛型 `TypeVar` 映射（`get_type_var_mapping`）；
3. **自动生成** `__init__`、`verify`、`parse`、`print` 四个方法；
4. 若定义了 `assembly_format`，用它编译出 `FormatProgram`（见第 5 节）。

### 3.3 PyRDL ↔ IRDL 双向转换

方言定义可以序列化为 IRDL IR（`pyrdl_to_irdl`），反过来 IRDL IR 也可以解释执行。这意味着：

- **动态方言**：从文本/网络加载一个方言定义，不需要重启进程；
- **方言即数据**：方言可作为 IR 的一部分随模块传输，实现分布式场景的方言协商；
- **工具链复用**：校验器、打印机都能作用在"方言定义"本身。

```python
def op_def_to_irdl(op: type[IRDLOperation]) -> OperationOp:
    op_def = op.get_irdl_definition()
    block = Block()
    builder = Builder(InsertPoint.at_end(block))
    operand_values = [
        range_to_irdl(builder, o[1].constr) for o in op_def.operands
    ]
    if any(isinstance(o[1], VariadicDef) for o in op_def.operands):
        ...  # 变长参数标记
    builder.insert(OperandsOp(operand_values, ...))
    builder.insert(ResultsOp(result_values, ...))
    return OperationOp(name, Region([block]))
```

---

## 4. IRDL 约束系统：方言的"类型系统"

### 4.1 三阶段生命周期

约束系统是方言定义的灵魂。`AttrConstraint` 定义了 **verify → variables → infer** 三阶段：

```python
@dataclass(frozen=True)
class AttrConstraint(ABC, Generic[AttributeCovT]):
    @abstractmethod
    def verify(self, attr: Attribute,
               constraint_context: ConstraintContext) -> None: ...

    def variables(self) -> set[str]:
        """返回此约束可提取的变量名集合"""
        return set()

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        """检查给定已绑定变量是否足以推断此属性"""
        return False

    def infer(self, context: ConstraintContext) -> AttributeCovT:
        """根据已绑定变量推断属性值"""
        raise ValueError(f"Cannot infer attribute from constraint {self}")
```

- **verify**：检验某个具体 Attribute/Type 是否满足约束（验证时被递归调用）；
- **variables**：声明该约束能"提取"哪些变量名（例如 `tensor<...xf32>` 提取形状参数）；
- **can_infer / infer**：当关联变量已绑定时，能否/怎样反推出属性——这是**类型推断**的基础。

### 4.2 内置约束矩阵

| 约束类 | 语义 | 可推断 | 典型用例 |
|--------|------|--------|---------|
| `AnyAttr()` | 任意属性 | ❌ | 泛型操作的无约束参数 |
| `BaseAttr(T)` | 必须是类型 T 的实例 | ✅（T 无参数时） | `operand_def(IntegerType)` |
| `EqAttrConstraint(val)` | 等于特定属性值 | ✅ | 常量属性绑定 |
| `AnyOf([...])` | 满足任一约束（`\|` 运算符） | 取决于子约束 | 多态操作数类型 |
| `AllOf([...])` | 满足所有约束（`&` 运算符） | 取决于子约束 | `Annotated[T, constraint]` |
| `ParamAttrConstraint(T, [p1, p2])` | 参数化属性 + 参数约束 | ✅（参数可推断时） | `tensor<?xf32>` 形状推断 |
| `TypeVarConstraint(tv, base)` | 泛型类型变量 + 下界 | ✅（映射已解析时） | 泛型操作的返回类型 |
| `ArrayOfConstraint(elem)` | `ArrayAttr` 元素约束 | ✅ | `ArrayAttr[IntegerAttr]` |
| `MessageConstraint(c, msg)` | 包装约束 + 自定义错误 | 同 c | 用户友好的验证错误 |

### 4.3 `ConstraintContext`：跨操作数共享变量

约束推断的关键是**变量共享**：`lhs` 的类型绑定到变量 `T`，`result` 的约束就能复用同一绑定。`ConstraintContext` 实现的是**延迟一致性检查**：

```python
@dataclass
class ConstraintContext:
    _variables: dict[str, Attribute] = field(default_factory=dict)
    _int_variables: dict[str, int] = field(default_factory=dict)

    def set_variable(self, name: str, attr: Attribute) -> None:
        if name in self._variables:
            existing = self._variables[name]
            if existing != attr:
                raise VerifyException(
                    f"Variable '{name}' is bound to {existing}, "
                    f"but a use expects it to be {attr}"
                )
        self._variables[name] = attr

    def get_variable(self, name: str) -> Attribute:
        return self._variables[name]
```

第一次遇到变量时绑定，后续每次使用都验证一致性——"操作数类型 = 结果类型"这类约束就是这样实现的。

### 4.4 泛型操作与类型推断的完整流程

```python
T = TypeVar("T", bound=Attribute)

@irdl_op_definition
class SelectOp(IRDLOperation):
    name = "my.select"
    cond = operand_def(IntegerType)
    true_value = operand_def(T)
    false_value = operand_def(T)
    result = result_def(T)

    assembly_format = (
        "$cond `,` $true_value `,` $false_value `:` "
        "type($true_value) attr-dict"
    )
```

```mermaid
flowchart LR
    A["解析 true_value"] --> B["提取类型 tensor<2xf32>"]
    B --> C["set_variable: T = tensor<2xf32>"]
    C --> D["解析 false_value"]
    D --> E["verify: 类型一致性检查"]
    E --> F["解析 result"]
    F --> G["TypeVarConstraint(T).can_infer() = True"]
    G --> H["infer() → result 类型 = tensor<2xf32>"]
```

### 4.5 `ConstraintConvertible`：把业务约束嵌入类型系统

```python
class MySpecialAttr(ParametrizedAttribute, ConstraintConvertible):
    @staticmethod
    def constr() -> AttrConstraint:
        return ParamAttrConstraint(MySpecialAttr, [BaseAttr(IntegerType)])
```

约束声明中直接写 `MySpecialAttr` 即可展开为完整约束，避免重复。

---

## 5. 声明式装配格式：parser/printer 自动化

### 5.1 语法与指令分类

手写 dialect 的 `parse`/`print` 是编译器开发中最繁琐的部分。MLIR 的做法是 Declarative Assembly Format，xDSL 把同一套语法搬到了 Python：`assembly_format` 字符串在装饰器阶段编译为 `FormatProgram`，解释执行。

```python
@irdl_op_definition
class MyOp(IRDLOperation):
    name = "my.op"
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)
    fast_math = opt_attr_def(UnitAttr)

    assembly_format = (
        "$lhs `,` $rhs `:` type($lhs) `->` type($result) "
        "(`(` `fast_math` `)` ^)?"
        "attr-dict"
    )
```

| 指令类别 | 指令名 | 功能 |
|---------|--------|------|
| 操作数 | `$lhs`、`operands` | 引用单个/全部操作数 |
| 结果 | `results`、`type($result)` | 引用结果类型 |
| 类型 | `type($x)`、`qualified` | 类型引用 / 限定名 |
| 函数类型 | `functional-type($x)` | `(in) -> out` 签名格式 |
| 属性 | `attr-dict` / `attr-dict-with-keyword` | 属性字典 |
| 区域 | `regions`、`region` | 区域引用 |
| 可选组 | `( ... )?` | 条件解析（见 5.2 锚定） |
| 自定义 | `custom<MyDirective>(...)` | 用户自定义指令 |
| 关键字 | `` `keyword` ``、`` `:` `` | 字面量标点 |

### 5.2 可选组与锚定机制

可选组是声明式格式的难点：何时触发？xDSL 用 **anchor（锚定）** 决定——`^` 标记锚定元素，解析时**先检查锚定元素是否存在**（属性字典里有没有 `fast_math`），存在才进入可选组，否则跳过。

```mermaid
stateDiagram-v2
    [*] --> ExpectOperand: 解析 $lhs
    ExpectOperand --> ExpectColon: 匹配 `,` `:`
    ExpectColon --> ParseTypes: 解析 type(...)
    ParseTypes --> CheckOptional: 到达可选组
    CheckOptional --> ParseFastMath: 锚定属性存在
    CheckOptional --> ParseAttrDict: 锚定属性不存在
    ParseFastMath --> ParseAttrDict
    ParseAttrDict --> [*]
```

### 5.3 自定义指令（CustomDirective）

声明式格式表达不了时（例如需要解析特殊语法），扩展 `CustomDirective`：

```python
class MyCustomDirective(CustomDirective):
    parameters: ClassVar = {"value": VariableDirective}

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        value = parser.parse_integer()
        state.add_attribute("my_value", IntegerAttr(value, i32))
        return True

    def print(self, printer: Printer, state: PrintingState,
              op: IRDLOperation) -> None:
        printer.print_string(str(op.attributes["my_value"].value.data))

@irdl_op_definition
class MyOp(IRDLOperation):
    custom_directives = (MyCustomDirective,)
    assembly_format = "custom<MyDirective>($value) attr-dict"
```

---

## 6. 模式重写引擎

### 6.1 Worklist 驱动的贪婪重写

`PatternRewriteWalker` 是重写引擎的核心——一个 **worklist + 递归重入** 的调度器：

```python
class PatternRewriteWalker:
    def __init__(self, pattern: RewritePattern, apply_recursively: bool = True):
        self._pattern = pattern
        self.apply_recursively = apply_recursively
        self._worklist = Worklist()

    def _add_operands_to_worklist(self, operands):
        """单用户操作数的定义操作入队（启发式）"""
        for operand in operands:
            if operand.has_one_use() and not isinstance(operand, ErasedSSAValue):
                if isinstance(op := operand.owner, Operation):
                    self._worklist.push(op)

    def _handle_operation_insertion(self, op):
        if self.apply_recursively:
            self._worklist.push(op)          # 新插入的操作也要匹配

    def _handle_operation_removal(self, op):
        if self.apply_recursively:
            self._add_operands_to_worklist(op.operands)
            ...                               # 同步清理 worklist 里的子操作

    def _handle_operation_replacement(self, op, new_results):
        if self.apply_recursively:
            for result in op.results:
                for user in result.uses:
                    self._worklist.push(user.operation)   # 消费者重新入队
```

三个关键设计决策：

1. **`apply_recursively=True`（默认）**：重写产生的新操作会重新入队，直到没有模式可匹配——保证**迭代收敛**；
2. **`has_one_use()` 启发式**：单用户操作数更可能是死代码 / 可折叠目标，优先处理，无效匹配可减少约 60%；
3. **删除时同步清理 worklist**：防止对已移除操作的悬空引用。

### 6.2 RewritePattern 层级（0.63）

```mermaid
classDiagram
    class RewritePattern {
        +match_and_rewrite(op, rewriter)*
    }
    class GreedyRewritePatternApplier {
        +patterns: list[RewritePattern]
        +context: Context|None
    }
    class PDLInterpRewritePattern {
        +interpreter: Interpreter
        +matcher: FuncOp
    }
    class TypeConversionPattern {
        +convert_type(type): Type
        +convert_op(op, rewriter)
    }
    RewritePattern <|-- GreedyRewritePatternApplier
    RewritePattern <|-- PDLInterpRewritePattern
    RewritePattern <|-- TypeConversionPattern
```

> 注意：0.63 已移除独立的 `GreedyRewritePattern` 类——自定义模式直接继承 `RewritePattern` 并实现 `match_and_rewrite(self, op, rewriter, /)`（斜杠表示位置参数），推荐用 `@op_type_rewrite_pattern` 装饰器按操作类型分发。

```python
class TensorAddZeroPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorAddOp, rewriter: PatternRewriter, /):
        if isinstance(op.rhs.owner, TensorZeroOp):
            rewriter.replace_matched_op([], [op.lhs])
        elif isinstance(op.lhs.owner, TensorZeroOp):
            rewriter.replace_matched_op([], [op.rhs])

walker = PatternRewriteWalker(
    GreedyRewritePatternApplier([TensorAddZeroPattern()])
)
walker.rewrite_module(module)
```

### 6.3 Canonicalization 注册

操作通过 `HasCanonicalizationPatternsTrait` 挂载自己的 canonicalization 模式，`CanonicalizePass` 会调用 `get_canonicalization_patterns()` 收集并统一调度：

```python
class AddiHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        return (
            AddZeroPattern(),      # x + 0 → x
            AddConstantPattern(),  # c1 + c2 → c3（常量折叠）
        )
```

xDSL 还支持 **Individual Rewrite 预调度**：在克隆的模块上逐个尝试模式，找出哪些操作可以被哪些模式改写，从而精确控制 canonicalize 的行为（对应 `xdsl-opt` 的 `apply-individual-rewrite`）。

### 6.4 PatternRewriter：类型安全的修改 API

`PatternRewriter` 继承 `Builder`，所有修改操作都经过它，从而能通知 worklist：

```python
class PatternRewriter(Builder, PatternRewriterListener):
    current_operation: Operation
    has_done_action: bool          # 是否实际执行了修改

    def insert_op(self, op):
        self.has_done_action = True
        self.handle_operation_insertion(op)
        super().insert_op(op, InsertPoint.before(self.current_operation))
        return op

    def erase_op(self, op, safe_erase=True):
        self.has_done_action = True
        self.handle_operation_removal(op)
        Rewriter.erase_op(op, safe_erase=safe_erase)

    def replace_all_uses_with(self, from_value, to_value, safe_erase=True):
        if from_value is to_value:
            return
        modified_ops = [use.operation for use in from_value.uses]
        from_value.replace_all_uses_with(to_value)
        for op in modified_ops:
            self.handle_operation_modification(op)  # 消费者重新入队
```

**`has_done_action` 是性能开关**：只有真正发生了修改，才触发 worklist 递归重入，避免无意义遍历。

---

## 7. Traits 与 Properties / Attributes 分离

### 7.1 Trait：操作语义约束

xDSL 的 Trait 系统合并了 MLIR 的 Traits 与 Interfaces：Trait 既声明语义（如"我是终止符"），也提供可查询的行为接口。

| Trait | 语义 | 验证行为 |
|-------|------|---------|
| `ConstantLike` | 常量操作 | 提供 `get_constant_value()` 供常量传播 |
| `HasParent(...)` | 父操作类型约束 | 验证 `parent_op()` 是否为指定类型 |
| `IsTerminator` | 基本块终止符 | 必须是块内最后操作 |
| `SymbolOpInterface` | 符号表条目 | 要求 `sym_name` 属性存在 |
| `MemoryEffect` | 内存副作用标注 | 驱动 alias 分析与调度优化 |
| `HasCanonicalizationPatternsTrait` | 注册 canonicalization 模式 | 返回 `RewritePattern` 列表 |
| `HasFolder` / `HasFolderInterface` | 常量折叠实现 | 提供 `fold()` 方法 |

```python
@irdl_op_definition
class MyYieldOp(IRDLOperation):
    name = "my.yield"
    arguments = var_operand_def(Attribute)

    traits = frozenset([
        IsTerminator(),
        HasParent(MyForOp, MyIfOp),   # 只允许出现在 my.for / my.if 中
    ])
```

Trait 验证在 `Operation.verify()` 中与 IRDL 约束验证串联自动触发。

### 7.2 Properties vs Attributes：架构选择

| 维度 | Attribute（属性） | Property（性质） |
|------|-------------------|------------------|
| 语义 | 用户附加元数据 | 操作核心配置 |
| 可变性 | 可变（Pass 可修改） | 不可变（创建后固定） |
| 序列化 | 进 `attr-dict` | 装配格式中显式声明 |
| 类型 | 任意 `Attribute` | 强类型约束 |
| 示例 | `passthrough`、调试信息 | `fastmath`、函数签名 |

**关键规则**：Property 必须在 `assembly_format` 中显式出现（或有默认值），否则验证报错——这是格式字符串与操作定义之间的一致性保证。

---

## 8. Pass 管线与变换生态

### 8.1 PassPipeline（0.63 API）

0.63 中 `PassManager` 被 `PassPipeline` 取代，执行方式变为显式 `apply(ctx, module)`：

```python
from xdsl.context import Context
from xdsl.passes import PassPipeline
from xdsl.transforms import canonicalize, convert_memref_to_ptr, lower_snitch

ctx = Context()
pipeline = PassPipeline((
    canonicalize.CanonicalizePass(),             # 常量折叠 + DCE
    lower_snitch.LowerSnitchPass(),              # Snitch 后端特定降低
    convert_memref_to_ptr.ConvertMemrefToPtr(),  # MemRef → 裸指针
))
pipeline.apply(ctx, module)
```

自定义 Pass 继承 `ModulePass`，实现 `apply(self, ctx, op)`，用 `name` 声明可被 `xdsl-opt -p` 引用。

### 8.2 内置变换生态速览（transforms 目录）

| 类别 | 代表 Pass | 作用 |
|------|-----------|------|
| 通用优化 | `canonicalize` | 常量折叠、DCE、模式重写收敛 |
| 方言降低 | `convert_memref_to_ptr`、`lower_snitch` | 高层抽象 → 目标相关表示 |
| 循环变换 | 分块 / 展开 / 合并 | 循环优化（scf/affine 层） |
| 后端相关 | riscv / csr / riscv_func 变换 | 寄存器分配前的准备 |
| 形状推断 | `shape_inference` | `tensor.extract_slice` 等形状传播 |
| 特定方言 | linalg 融合 / 泛化 | 线性代数核优化 |

### 8.3 Dialect Conversion 与 PDL-Interp

**类型转换模式**支持渐进式方言迁移：

```python
class MyTypeConversion(TypeConversionPattern):
    def convert_type(self, typ: TypeAttribute) -> TypeAttribute:
        if isinstance(typ, MyCustomTensorType):
            return builtin.TensorType(...)
        return typ
```

**PDL-interp 模式匹配**把"重写规则"也变成 IR：

```python
class ApplyPDLInterpPass(ModulePass):
    name = "apply-pdl-interp"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        matcher = ...   # 从模块加载 pdl_interp 匹配器
        interpreter = Interpreter(pdl_interp_module)
        interpreter.register_implementations(PDLInterpFunctions())
        rewrite_pattern = PDLInterpRewritePattern(matcher, interpreter, ...)
        PatternRewriteWalker(rewrite_pattern).rewrite_module(op)
```

这意味着**规则可以用独立的 IR 程序表达**，而非硬编码 Python——规则本身可以被生成、被证明、被移植。

---

## 9. 序列化与后端

### 9.1 前端：MLIR 兼容 Lexer / Parser

xDSL 自带与 MLIR 文本格式兼容的 Lexer/Parser：`%0 = arith.constant 10 : i32` 这种语法可以直接解析进 IR。parser 遇到方言中不认识的语法时按"泛型操作"（`"dialect.op"(...) : (...) -> (...)`）解析，保证**部分方言未注册也能读入模块**。

### 9.2 后端：AssemblyPrinter 汇编输出

```mermaid
flowchart LR
    A["LLVM Dialect IR"] --> B["AssemblyPrintable"]
    B --> C["AssemblyPrinter"]
    C --> D[".s 汇编文件"]
    D --> E["as / clang"]
    E --> F["可执行文件"]
```

xDSL 内置 RISC-V、x86、CSL 等后端方言。实现后端的操作继承 `AssemblyPrintable`，通过 `AssemblyPrinter.print_module()` 统一驱动输出汇编。这让 xDSL 能跑通"DSL → 多级 IR → 汇编"的完整工具链，很适合做架构探索（如 RISC-V 向量扩展的编译器研究）。

---

## 10. 实战：从零构建一个可运行的张量方言

下面这个例子在 **xDSL 0.63** 上实测通过，完整覆盖"定义方言 → 装配格式 → 重写模式 → 运行变换"全流程。

```python
from typing import TypeVar
from xdsl.ir import Attribute, Dialect, Operation, ParametrizedAttribute
from xdsl.irdl import (
    irdl_attr_definition, irdl_op_definition, IRDLOperation,
    operand_def, result_def,
)
from xdsl.dialects.builtin import (
    ModuleOp, StringAttr, IntegerType, TensorType,
)
from xdsl.pattern_rewriter import (
    PatternRewriter, RewritePattern, GreedyRewritePatternApplier,
    PatternRewriteWalker, op_type_rewrite_pattern,
)
from xdsl.context import Context
from xdsl.transforms import canonicalize
from xdsl.passes import PassPipeline

# ─── 1. 属性定义（0.63: 参数用纯类型注解）───
@irdl_attr_definition
class LayoutAttr(ParametrizedAttribute):
    name = "tensor.layout"
    value: StringAttr

# ─── 2. 操作定义 ───
@irdl_op_definition
class TensorZeroOp(IRDLOperation):
    name = "tensor.zeros"
    result = result_def(TensorType)

@irdl_op_definition
class TensorAddOp(IRDLOperation):
    name = "tensor.add"
    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)
    assembly_format = (
        "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) "
        "attr-dict"
    )

# ─── 3. 重写模式：x + zeros → x（'x + 0' 消除）───
class TensorAddZeroPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorAddOp, rewriter: PatternRewriter, /):
        if isinstance(op.rhs.owner, TensorZeroOp):
            rewriter.replace_matched_op([], [op.lhs])
        elif isinstance(op.lhs.owner, TensorZeroOp):
            rewriter.replace_matched_op([], [op.rhs])

# ─── 4. 方言注册 ───
TensorDialect = Dialect("tensor", [TensorAddOp, TensorZeroOp], [LayoutAttr])

# ─── 5. 编译流程 ───
tt = TensorType(IntegerType(32), [2, 2])
z = TensorZeroOp.create(result_types=[tt])
add = TensorAddOp.create(operands=[z.results[0], z.results[0]], result_types=[tt])

module = ModuleOp([z, add])
ctx = Context()
print("── 优化前 ──")
print(module)

# 应用重写模式：tensor.add(zeros, zeros) 应该整个消失
PatternRewriteWalker(
    GreedyRewritePatternApplier([TensorAddZeroPattern()])
).rewrite_module(module)

print("── 重写后 ──")
print(module)
```

运行输出：

```mlir
── 优化前 ──
builtin.module {
  %0 = "tensor.zeros"() : () -> tensor<2x2xi32>
  %1 = tensor.add %0, %0 : tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
}
── 重写后 ──
builtin.module {
  %0 = "tensor.zeros"() : () -> tensor<2x2xi32>
}
```

可以看到 `tensor.add` 被"吸收"——两个操作数都来自 `tensor.zeros`，重写为直接返回左操作数，加法操作消失。

### 10.1 组合成 Pass 管线

```python
class TensorOptimizePass(ModulePass):
    name = "tensor-optimize"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([TensorAddZeroPattern()])
        ).rewrite_module(op)
```

`0.63` 中 Pass 执行方式：

```python
PassPipeline((TensorOptimizePass(),)).apply(ctx, module)
```

与 `xdsl-opt` 一致的是，所有 Pass 都是 `ModulePass` 子类、以 `name` 注册——自定义 Pass 可以直接进 `xdsl-opt -p "tensor-optimize"` 管线。

---

## 11. 工程实践与调优

### 11.1 性能策略

| 环节 | 策略 | 效果 |
|------|------|------|
| IR 遍历 | `apply_recursively=False` 减少重复遍历 | 遍历时间 -30~50% |
| 约束验证 | 批量合并属性访问，减少 `verify()` 调用 | 验证时间 -20% |
| Pass 并行 | 并行化无依赖 Pass | 多核利用率提升 |
| C 扩展 | 部分版本支持 C 加速热点路径 | 热点 -40% |
| Worklist 剪枝 | 仅 `has_one_use()` 操作入队 | 无效匹配 -60% |

> 性能数据来自 xDSL 官方文档与社区基准（不同版本/机器有差异，标注为参考量级）。

### 11.2 调试技巧（xdsl-opt）

```bash
# 打印每次 Pass 后的 IR
xdsl-opt -p "pass1,pass2,pass3" --print-ir-after-all input.mlir

# 只应用单个模式（精确定位问题）
xdsl-opt -p "apply-individual-rewrite{matched_operation_index=42 \
operation_name=arith.addi pattern_name=AddConstantPattern}" input.mlir

# 每步校验
xdsl-opt -p "canonicalize" --verify-each input.mlir
```

### 11.3 方言 Stub 生成

自定义方言可自动生成 `.pyi` 类型存根，IDE 补全直接可用：

```python
from xdsl.utils.dialect_stub import DialectStubGenerator
stub = DialectStubGenerator(TensorDialect)
print(stub.generate())
```

---

## 12. 全景对比与展望

### 12.1 xDSL vs MLIR 关键差异

| 维度 | xDSL | MLIR |
|------|------|------|
| 方言定义 | Python 装饰器（运行时） | TableGen/ODS + C++（编译时） |
| 约束推断 | `can_infer()` + `infer()` | 内置类型推断引擎 |
| 装配格式 | `FormatProgram` 解释执行 | 编译为 C++ parser/printer |
| 重写引擎 | Python worklist | C++ RewritePattern 体系 |
| 性能 | 万级操作 ~秒级 | 百万级操作 ~毫秒级 |
| 开发效率 | `pip install` 即用 | CMake 构建链 |
| 生态成熟度 | 研究/教学为主 | 工业级生产部署 |

### 12.2 核心回顾

1. **五层体系**：方言定义（PyRDL）→ 约束系统（IRDL）→ IR 数据模型 → 变换引擎 → 序列化/后端，每一层都对应 MLIR 的抽象；
2. **约束三阶段**：`verify / variables / infer` 让"操作数类型 = 结果类型"这类泛型约束可声明、可推断；
3. **声明式装配**：`assembly_format` 把 parser/printer 代码量降低约 80%，锚定机制保证可选组解析的确定性；
4. **worklist 贪婪重写**：`apply_recursively` + `has_one_use` 启发式实现高效迭代收敛；
5. **方言即数据**：PyRDL ↔ IRDL 双向转换 + PDL-interp，让方言定义和重写规则都能作为 IR 程序处理。

### 12.3 展望

- **互操作**：与 MLIR Python Bindings 双向桥接，复用双方方言生态；
- **JIT 加速**：Numba/Cython 加速约束验证等热点路径；
- **形式化验证**：IR 变换等价性证明（IRDL 的可序列化特性让这变得可行）；
- **生产化**：RISC-V 等后端方言持续演进，在 DSA（领域专用架构）编译器研究中逐步落地。

xDSL 的价值不在于替代 MLIR，而在于**用 1/100 的代码量复刻了 MLIR 的抽象体系**——对想深入理解多级 IR 编译器的人而言，它是目前最清晰、最可运行的"教科书实现"。