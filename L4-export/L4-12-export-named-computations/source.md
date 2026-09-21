<!-- sdy-coverage
transforms/export/test/export_named_computations.mlir
-->

# L4-12 · export-named-computations — 源 IR

**核心机制**：`sdy.named_computation` 被**反向 outline** 成函数 + call。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `transforms/export/test/export_named_computations.mlir` | **1263** | **93** |

```mlir
// RUN: sdy_opt %s -sdy-export-named-computations -split-input-file | FileCheck %s
```

网格：

```mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>
```

**回顾 L1-08**：`sdy.named_computation` 是"**内联的函数**" ——
传播要能穿过函数调用，所以把函数体**内联**进调用点、但保留一个名字。
本课做的是**反向操作**：导出时把它**重新 outline 成函数**。

---

## 一、★ 基本 outline

### 输入：`named_computation`

```mlir
func.func @vanilla_named_computation(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
```

```mlir
  %0 = sdy.named_computation<"bar">(%arg0) (%arg1: tensor<8x2xi32>) {
    %1 = stablehlo.multiply %arg1, %arg1 : tensor<8x2xi32>
    sdy.return %1 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
```

### 输出：函数 + call

```mlir
// CHECK-LABEL: func @vanilla_named_computation(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
```

```mlir
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
```

```mlir
// CHECK-LABEL: func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> attributes {sdy.original_func_name = "bar"} {
```

**读法**（三步）：
1. `named_computation<"bar">` 的区域 → 变成一个 **`func.func private @bar`**
2. 调用点 → **`call @bar(%arg0)`**
3. 函数带 **`sdy.original_func_name = "bar"`** 属性 —— **保留原名**

**为什么需要 `sdy.original_func_name`**：
函数名在 IR 里必须**唯一**。当同一个名字出现多次时，必须加后缀区分（见第三节）。
但"它原本叫什么"这个信息不能丢 —— 所以用属性记下来。

---

## 二、★ 分片的两个来源

这是本课最需要注意的细节。测试开头的注释直接点明了：

```
// Note we don't override the block argument shardings of the function
// @ignore_operand_shardings, but we set the argument shardings on the call
// to @foo.
```

### 完整例子

```mlir
func.func @ignore_operand_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %2 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %2 : tensor<8x2xi32>
  } {random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", ?}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
```

```mlir
// CHECK-LABEL: func @ignore_operand_shardings(
// CHECK-SAME: %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME: -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%arg0)
  // CHECK-SAME:   {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value",
  // CHECK-SAME:    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-SAME:   : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[MOVE_TO_HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[CALL]]) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", ?}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[MOVE_TO_HOST]] : tensor<8x2xi32>
```

```mlir
// CHECK-LABEL: func private @foo
// CHECK-SAME:    (%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> tensor<8x2xi32> attributes {sdy.original_func_name = "foo"} {
```

### 逐项读

| 位置 | 分片来源 | 值 |
|---|---|---|
| **函数参数** `@foo` 的 `%arg0` | **block argument 的分片** | `[{}, {"y"}]` |
| **call 的 sharding** | **`out_shardings`** | `[{"x"}, {}]` |

**关键**：函数参数的分片**来自 block argument**（`%arg1: tensor<8x2xi32>` 那一侧），
**不是**来自 `in_shardings`。

**为什么**：`in_shardings` 描述的是"**调用者怎么传**"，而函数体看到的是
**局部/块参数**的分片。outline 后的函数**就是那个函数体** ——
所以它的参数分片应该用 block argument 的。

**注意**：本例中两者恰好相同（都是 `[{}, {"y"}]`），但注释明确说
"we **don't override** the block argument shardings" —— 说明**规则是取 block argument**。

### 属性也被搬运

`named_computation` 上的属性**移到了 call 上**：

| 属性 | 原位置 | 新位置 |
|---|---|---|
| `random_attr = "random_value"` | `named_computation` | **call** |
| `mhlo.frontend_attributes = {...}` | `named_computation` | **call** |

**读法**：`named_computation` 是"内联的调用点"，它的属性语义上属于**调用点** ——
所以 outline 后应该留在 call 上。

---

## 三、★ 命名与去重

### 同名两次（**相同分片**）

```mlir
func.func @multiple_same_named_computations_same_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %2 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %2 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = sdy.named_computation<"baz">(%arg0) in_shardings=[<@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>) {
    %3 = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {"y", ?}]>]>} : tensor<8x2xi32>
    sdy.return %3 : tensor<8x2xi32>
  } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
```

```mlir
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %1 : tensor<8x2xi32>
```

```mlir
// CHECK-LABEL: func private @baz(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> tensor<8x2xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "baz"}
```

**读法**：两个同名 `named_computation<"baz">` → 生成 **`@baz`** 与 **`@baz_0`**。
**每个实例一个函数**，名字冲突时加 `_0` 后缀。
两个 call 的 sharding **相同**（都是 `[{"x"}, {}]`）。

### 同名两次（**不同分片**）

```mlir
func.func @multiple_same_named_computations_different_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  // CHECK-NEXT: %0 = call @baz(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %1 = call @baz_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %1 : tensor<8x2xi32>
```

```mlir
// CHECK-LABEL: func private @baz_0(
// CHECK-SAME:    %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
// CHECK-SAME:    -> tensor<8x2xi32>
// CHECK-SAME:  attributes {sdy.original_func_name = "baz"}
```

**读法**：函数命名与上面**一样**（`@baz` / `@baz_0`）；
差别只在 **call 的 sharding**：
- 第一个 call：`[{"x"}, {}]`
- 第二个 call：`[{"x"}, {"y"}]`（**不同**）

**关键观察**：
- **两个函数都带 `sdy.original_func_name = "baz"`** —— 说明它们**同源**。
- 这个属性正是为下游保留"它们来自同一个 `named_computation` 名字"这个信息。
- 测试里还有一族 `three_named_computations_same_origin_func_*` ——
  用例名里的 **`same_origin_func`** 进一步印证了这个概念。

> **注意**：函数参数的分片在两个例子里**都是** `[{}, {"y"}]` ——
> 因为 block argument 的分片本来就相同（差异只在 `out_shardings`）。
> 这再次印证了"函数参数分片来自 block argument"。

---

## 四、93 个用例的族谱

从用例名可以读出覆盖的场景：

| 族 | 代表用例 | 场景 |
|---|---|---|
| **基础** | `vanilla_named_computation`、`single_call` | 最简单的 outline |
| **分片来源** | `ignore_operand_shardings` | 函数参数用 block argument |
| **同名去重** | `multiple_same_named_computations_{same,different}_shardings` | `@baz` / `@baz_0` |
| **嵌套** | `non_flat_nested_named_computations_{same,different,mixed}_shardings` | named_computation **嵌套** |
| **与 manual_axes 交互** | `named_computations_with_manual_axes_*`（多个） | 带 `manual_axes` 的情形 |
| **在 manual_computation 内** | `*_one_inside_manual_computation*`（多个） | 区域内的 named_computation |
| **嵌套 manual_computation** | `nested_manual_computations` | 两层区域 |
| **无 out_sharding** | `single_named_computation_no_out_sharding`、`same_named_computations_one_with_no_out_sharding` | 缺 `out_shardings` 的情形 |
| **同源多实例** | `three_named_computations_same_origin_func_*` | 三个实例共享来源 |
| **其它** | `xla`、`multiple_same_named_computations_same_shardings_named_computations_have_different_manual_computation_calls` | — |

### 最复杂的一族：`manual_axes` 与嵌套

`named_computations_with_manual_axes_*` 与 `*_inside_manual_computation_*` 覆盖
**两个机制的交叉**：

- `named_computation` 是"内联的函数"（L1-08）
- `manual_computation` 是"区域内自己管分片"（L1-07）

当 `named_computation` **出现在 `manual_computation` 区域内**时：
- 它看到的 block argument 是**局部形状**
- outline 出的函数参数分片也应该是**局部的**
- 但 `in_shardings` / `out_shardings` 写的是**全局分片** —— 需要转换

用例名里的 **`one_without_manual_axes`** / **`with_manual_axes`** 成对出现，
说明这个交叉有多个变体需要区分。

---

## 五、与其它课的呼应

| 课 | 讲什么 | 与本课的关系 |
|---|---|---|
| **L1-08** | `named_computation` 的定义与用途 | 本课做**反向**操作（outline） |
| **L1-07** | `manual_computation` 的语义 | 本课有大量交叉用例 |
| **L3-06** | 把 `call` **内联**成 `named_computation` | **正好相反**的方向！ |
| **L3-07** | 提升内联网格 | 导出期不做提升 |

**L3-06 与本课是一对**：

```
L3-06（导入）：call  --内联-->  named_computation
L4-12（导出）：named_computation  --outline-->  func + call
```

**为什么导入要内联、导出要 outline**：
- **导入期**：传播需要"看得见"函数体（否则分片传不进去）→ 内联。
- **导出期**：IR 要交给后端，后端需要真正的函数 → outline 回去。

**一句话总结**：
> **`named_computation` 是"传播期的临时形态"** ——
> 导入时把 call 内联成它，导出时再 outline 回函数。
