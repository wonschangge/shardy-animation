# L4-12 · `export-named-computations` — 导出命名计算

> 层：**L4 · 导出流水线** ｜ 优先级：P2 ｜ 前置课：`L1-07`、`L1-08`、`L3-06`

## 学习目标

看完这一课，你应该能：

1. **预测 outline 后的函数签名**（TODOLIST 验收点）；
2. 说出**函数参数分片**与 **call sharding** 的**两个来源**；
3. 说出同名 `named_computation` 的命名规则；
4. 说出本课与 L3-06 的**镜像关系**。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 |
|---|---|---|
| `transforms/export/test/export_named_computations.mlir` | **1263** | **93** |

RUN 行：`-sdy-export-named-computations -split-input-file`

## 场景（6 幕）

1. **★ 基本 outline：区域变函数，调用点变 call**
2. **★ 分片的两个来源**
3. **★ 命名：`@baz` 与 `@baz_0`**
4. 93 个用例的族谱
5. 与 L3-06 的镜像关系
6. 练习

## 核心结论

### 基本 outline（三步）

1. `named_computation<"bar">` 的区域 → **`func.func private @bar`**
2. 调用点 → **`call @bar(%arg0)`**
3. 函数带 **`sdy.original_func_name = "bar"`** —— 保留原名

**属性也搬家**：`named_computation` 上的 `random_attr`、`mhlo.frontend_attributes`
全部移到 **call** 上 —— 因为它是「内联的调用点」，属性语义上属于调用点。

### ★ 分片的两个来源

| 位置 | 来源 | 说明 |
|---|---|---|
| **函数参数** | **block argument** 的分片 | **不是** `in_shardings` |
| **call 的 sharding** | **`out_shardings`** | 描述「这个调用返回什么」 |

测试注释点明了规则：
`we don't override the block argument shardings of the function @ignore_operand_shardings,
but we set the argument shardings on the call to @foo.`

**理由**：outline 后的函数**就是那个函数体**，它看到的是 block argument。

### ★ 命名规则

同一个名字出现两次 → **每个实例一个函数**：

| 情形 | 函数名 | call 的 sharding |
|---|---|---|
| **相同分片** | `@baz` / `@baz_0` | 两个都是 `[{"x"}, {}]` |
| **不同分片** | `@baz` / `@baz_0` | `[{"x"}, {}]` / `[{"x"}, {"y"}]` |

- 名字冲突时加 **`_0`** 后缀。
- **两个函数都带 `sdy.original_func_name = "baz"`** —— 下游据此知道它们**同源**。
- 分片相同/不同**不影响命名**，差别只在 call 的 sharding。
- 两例中函数参数的分片**都是** `[{}, {"y"}]` —— 再次印证「参数分片来自 block argument」。

### 93 个用例的 7 类场景

基础 / 分片来源 / 同名去重 / **嵌套** / **与 `manual_axes` 交互** /
**在 `manual_computation` 区域内** / **无 `out_sharding`** / 同源多实例。

**最复杂的交叉**：`named_computation` 出现在 `manual_computation` 区域内时，
它看到的 block argument 是**局部形状**，而 `in/out_shardings` 写的是**全局分片**
—— 需要转换。用例名里 `with_manual_axes` / `without` **成对出现**说明有多个变体。

### 与 L3-06 的镜像关系

```
L3-06（导入）：call  --内联-->  named_computation     （传播期的形态）
L4-12（导出）：named_computation  --outline-->  func + call  （交给后端的形态）
```

**为什么需要这个中间形态**：`call` 是边界（传播默认**不穿过**），
`named_computation` 是「内联的调用点」（传播**可以穿进去**）但保留了名字
—— 所以导出时才能准确 outline 回函数。

## 练习

见第 6 幕。三道题分别考基本 outline、分片来源、命名规则。

## 验收点

- [x] `check_ir_fidelity.py`：19 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：源文件已声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能预测 outline 后的函数签名（含参数分片来源与命名）
