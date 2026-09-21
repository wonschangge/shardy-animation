<!-- sdy-coverage
transforms/import/test/remove_size_one_axes.mlir
transforms/import/test/pre_order_funcs.mlir
transforms/import/test/propagate_sharding_from_func_to_call.mlir
-->

# L3-10 · misc-import-cleanup — 源 IR

本课覆盖三个**收尾清理**类 pass：

| 文件 | 行数 | 用例数 | pass |
|---|---|---|---|
| `transforms/import/test/remove_size_one_axes.mlir` | 166 | 10 | `-sdy-remove-size-one-axes` |
| `transforms/import/test/pre_order_funcs.mlir` | 19 | 3 | `-sdy-pre-order-funcs` |
| `transforms/import/test/propagate_sharding_from_func_to_call.mlir` | 151 | 20 | `-sdy-propagate-sharding-from-func-to-call` |

---

## 一、`-sdy-remove-size-one-axes`：删掉大小为 1 的轴

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-remove-size-one-axes 2>&1 | FileCheck %s
```

### 网格定义**不变**，只改分片

```mlir
sdy.mesh @mesh1 = <["a"=1, "b"=2, "c"=1, "d"=4, "e"=1], device_ids=[0, 2, 1, 3, 4, 6, 5, 7]>
sdy.mesh @mesh2 = <["a"=4, "b"=2]>
sdy.mesh @mesh3 = <["x"=1, "y"=1]>
sdy.mesh @mesh4 = <["a"=1, "b"=2, "c"=1]>
```

```mlir
// CHECK: sdy.mesh @mesh1 = <["a"=1, "b"=2, "c"=1, "d"=4, "e"=1], device_ids=[0, 2, 1, 3, 4, 6, 5, 7]>
// CHECK: sdy.mesh @mesh2 = <["a"=4, "b"=2]>
// CHECK: sdy.mesh @mesh3 = <["x"=1, "y"=1]>
// CHECK: sdy.mesh @mesh4 = <["a"=1, "b"=2, "c"=1]>
```

**读法**：`@mesh1` 有 3 个大小为 1 的轴（`a`、`c`、`e`），但**网格定义原样保留**。
改动只发生在**引用这些轴的分片**上。

### 四种位置都会被清理

```mlir
func.func @func_and_op_shardings(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a", "b"}, {"c", ?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"d", "e", ?}, {}], replicated={"b", "c"}>},
  %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"a"}, {"b"}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"e"}, {"c", ?}], unreduced={"a", "d"}>},
      tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a", "b", "c"}, {}], unreduced={"e"}>}) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"d", ?}, {"e", ?}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.add %arg2, %arg2 : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"c"}, {}], replicated={"d"}>]>} : tensor<8x8xf32>
  return %0, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}
```

```mlir
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"b"}, {?}]>},
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"d", ?}, {}], replicated={"b"}>},
// CHECK-SAME:    %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"a"}, {"b"}]>}
// CHECK-SAME:  ) -> (
// CHECK-SAME:    tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {?}], unreduced={"d"}>},
// CHECK-SAME:    tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"b"}, {}]>}) {
```

```mlir
  // CHECK-NEXT:   %[[ADD1:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"d", ?}, {?}]>]>}
  // CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NOT:    sdy.sharding
  // CHECK-NEXT:   %[[ADD3:.*]] = stablehlo.add %[[ADD2]], %[[ADD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {}], replicated={"d"}>]>}
  // CHECK-NEXT:   return %[[ADD1]], %[[ADD3]]
  // CHECK-NEXT: }
```

**逐项读**（`@mesh1` 的 `a`/`c`/`e` 大小为 1；`@mesh2` 没有 size-1 轴）：

| 位置 | 输入 | 输出 | 删了什么 |
|---|---|---|---|
| `%arg0` 维分片 | `[{"a", "b"}, {"c", ?}]` | `[{"b"}, {?}]` | `"a"`（维0）、`"c"`（维1） |
| `%arg1` 维分片 | `[{"d", "e", ?}, {}]` | `[{"d", ?}, {}]` | `"e"` |
| `%arg1` replicated | `replicated={"b", "c"}` | `replicated={"b"}` | `"c"` |
| `%arg2`（`@mesh2`） | `[{"a"}, {"b"}]` | **不变** | 无 size-1 轴 |
| 结果 1 维分片 | `[{"e"}, {"c", ?}]` | `[{}, {?}]` | `"e"`、`"c"` |
| 结果 1 unreduced | `unreduced={"a", "d"}` | `unreduced={"d"}` | `"a"` |
| 结果 2 维分片 | `[{"a", "b", "c"}, {}]` | `[{"b"}, {}]` | `"a"`、`"c"` |
| 结果 2 unreduced | `unreduced={"e"}` | **整个属性消失** | `"e"`（清空后属性不再输出） |
| 算子 `%0` | `[{"d", ?}, {"e", ?}]` | `[{"d", ?}, {?}]` | `"e"` |
| 算子 `%1` | 无分片 | **仍然无分片** | — |
| 算子 `%2` | `[{"c"}, {}], replicated={"d"}` | `[{}, {}], replicated={"d"}` | `"c"` |

**为什么删**：大小为 1 的轴意味着"这条轴上只有 1 台设备" ——
**沿它分片不会真的切开任何东西**。保留它只会：
- 让分片表达式变长、难读
- 可能让传播/导出产生**无意义的通信**（对着一个 1 宽的维度做 all-gather）

> 注意 `@mesh2` 上的 `%arg2` 完全没变 —— 说明规则是**逐轴判断**，不是"有 size-1 轴就整体重写"。

### 内联网格也会被处理

```mlir
func.func @inlined_mesh(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=1, "b"=2, "c"=1, "d"=2], device_ids=[3, 1, 2, 0]>, [{"a", "b"}, {"c", ?}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a", "b"}, {}]>}) {
```

```mlir
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=1, "b"=2, "c"=1, "d"=2], device_ids=[3, 1, 2, 0]>, [{"b"}, {?}]>}
// CHECK-SAME:  ) -> (
// CHECK-SAME:    tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a", "b"}, {}]>}) {
```

**读法**：内联网格同样只改**分片部分**（`[{"a","b"}, {"c",?}]` → `[{"b"}, {?}]`），
网格定义本身原样保留。

---

## 二、`-sdy-pre-order-funcs`：按调用序重排函数

```mlir
// RUN: sdy_opt %s -sdy-pre-order-funcs | FileCheck %s
```

```mlir
func.func @func2() {
```

```mlir
  return
}

func.func @func1() {
  func.call @func2() : () -> ()
  return
}

func.func @main() {
  func.call @func1() : () -> ()
  return
}
```

```mlir
// CHECK: func.func @main
// CHECK: func.func @func1
// CHECK: func.func @func2
```

**读法**：
- 输入顺序是 `func2, func1, main`（被调者在前）。
- 输出顺序是 `main, func1, func2`（**调用者在前**）。
- 即从入口 `main` 出发做**前序遍历**（pre-order），依次访问被调函数。

**为什么要重排**：让后续 pass 能**按顺序单遍处理** ——
处理一个函数时，它调用的函数已经处理过了（或即将处理），无需来回跳转。
这是编译器里常见的"调用图线性化"手法。

---

## 三、`-sdy-propagate-sharding-from-func-to-call`：函数结果分片 → 调用点

```mlir
// RUN: sdy_opt %s -sdy-propagate-sharding-from-func-to-call -split-input-file | FileCheck %s
```

### 调用点没有分片时：从函数**抄过来**

```mlir
func.func @propagate_func_to_call(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
```

```mlir
  %0 = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  return %arg0 : tensor<8x2xi32>
}
```

```mlir
  // CHECK-NEXT: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
```

**读法**：`@foo` 的**结果**上有 `[{"x"}, {"y"}]`，而调用点**没有**分片
→ 把函数结果的分片**抄到调用点上**。

### 调用点已有分片时：**不覆盖**

```mlir
func.func @do_not_overwrite_call_sharding(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
```

```mlir
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  return %arg0 : tensor<8x2xi32>
}
```

```mlir
  // CHECK-NEXT: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
```

**读法**：调用点已有 `[{"y"}, {"x"}]`（与函数的 `[{"x"}, {"y"}]` **不同**）
→ **保持调用点的**，不覆盖。用例名 `do_not_overwrite_call_sharding` 点明了。

### 两者都没有：什么都不加

```mlir
func.func @both_call_and_func_has_empty_result_shardings(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
```

```mlir
  %0 = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
```

```mlir
  // CHECK-NEXT: %0 = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
```

### 为什么需要这个 pass（与 L3-06 的衔接）

L3-06 讲过 `-sdy-import-func-calls` 的规则：

```
// NOTE: we ignore any arg/result shardings on the function.
```

也就是说 **`out_shardings` 只来自调用点**。那么**函数结果上写的分片怎么办**？
—— 如果不管，就会**丢失**。

这个 pass 就是**补救措施**：在 `-sdy-import-func-calls` **之前**运行，
把函数结果上的分片"搬"到调用点上，这样 L3-06 就能正常读到它。

> 顺序：`-sdy-propagate-sharding-from-func-to-call` → `-sdy-import-func-calls`

---

## 四、用例族谱

| pass | 用例 |
|---|---|
| `remove_size_one_axes`（10） | `func_and_op_shardings`、`inlined_mesh`、`shardings_with_priorities`、`manual_computation`、`manual_computation_inlined_mesh`、`single_call` 等 |
| `pre_order_funcs`（3） | 一个 3 函数的调用链 |
| `propagate_sharding_from_func_to_call`（20） | `propagate_func_to_call`、`do_not_overwrite_call_sharding`、`both_call_and_func_has_empty_result_shardings`、`multiple_results`、`keep_empty_call_sharding`、`multiple_results_one_same_one_is_empty` 等 |
