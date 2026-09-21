<!-- sdy-coverage
transforms/propagation/test/user_priority_propagation.mlir
-->

# L2-06 · user-priority-propagation — 源 IR

覆盖：`shardy/dialect/sdy/transforms/propagation/test/user_priority_propagation.mlir`（451 行 / 20 个用例）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-user-priority-propagate 2>&1 | FileCheck %s
```

网格：`sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>`

---

## 一、没有任何优先级时：等同算子优先级传播

```mlir
func.func @no_priorities(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                         %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.divide %0, %arg2 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: stablehlo.divide %[[ADD]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
```

没有 `pN` 时一切照旧 —— 用户优先级是**最外层**策略，没标注就退化成内层。

---

## 二、允许跳号：p1 与 p4 之间没有 p2/p3

```mlir
func.func @skipped_priorities(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}p4]>},
                              %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.divide %0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c", ?}p1]>]>} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>}
  // CHECK-NEXT: stablehlo.divide %[[ADD]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>}
```

**要点**：
- 优先级**允许有空档**（L1-02 的不变量：p0 与 p4 可以同时存在而 p1~p3 缺失）。
- `%arg0` 第 1 维是 `{"b"}p4`，`divide` 第 1 维是 `{"c", ?}p1`。
  **p1 比 p4 更优先** → 结果第 1 维是 `{"c", ?}`，`"b"` 被放弃。

---

## 三、优先级是**逐维**的

`%arg0` 两维都在 p1；`divide` 第 0 维在 p0：

```mlir
func.func @arg_lower_priority_than_return_value(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}p1, {"b"}p1]>},
    %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<8x8xf32>
  %2 = stablehlo.divide %1, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}p0, {?}]>]>} : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: stablehlo.divide %[[ADD_1]], %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}, {"b", ?}]>]>}
```

**读法**（关键）：
- **第 0 维**：`"a"` 是 p1、`"c"` 是 p0 → **p0 赢**，结果是 `{"c", ?}`。
- **第 1 维**：只有 `"b"`（p1），没有竞争者 → **保留** `{"b", ?}`。

**结论**：优先级不是"整个分片二选一"，而是**每个维度各自比较**。

---

## 四、反过来：参数优先级更高时

```mlir
func.func @arg_higher_priority_than_return_value(
      %arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p0, {"b"}p0]>},
      %arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<8x8xf32>
  %2 = stablehlo.divide %1, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}p1, {?}]>]>} : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: stablehlo.divide %[[ADD_1]], %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}, {"b", ?}]>]>}
```

**读法**：
- `%arg1` 是 p0，`divide` 是 p1 → **p0 赢**：前两个算子拿到 `[{"a", ?}, {"b", ?}]`。
- 但 `divide` 自己**用户标注的** `"c"` 仍在它自己的分片上（`[{"c", ?}, {"b", ?}]`）——
  用户标注不会被删掉，只是不向外传播。
- 第 1 维 `"b"` 从 p0 传了过来，补上了 `divide` 原来空着的第 1 维。

---

## 五、低优先级的开维会被高优先级**继续切**

```mlir
func.func @dim_with_lower_priority_gets_further_sharded_by_higher(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}p1, {}]>},
    %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}p0, {}]>]>} : tensor<8x8xf32>
  %2 = stablehlo.divide %0, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}p0, {}]>]>} : tensor<8x8xf32>
  return %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}, {?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %arg0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {}]>]>}
  // CHECK-NEXT: stablehlo.divide %[[ADD_0]], %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}, {}]>]>}
```

**读法**：`%arg0` 第 0 维是 `{"b", ?}p1`（开维），而 `%1` 要求 `{"b", "a", ?}p0`。
高优先级（p0）**在已有 `"b"` 的基础上继续加了 `"a"`** —— 轴序保持，只是变长。

> 这就是用例名的含义："低优先级维被高优先级进一步分片"。

---

## 六、空的**开**维也可以有优先级

```mlir
func.func @open_empty_dim_with_priority(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p0, {"b"}p0]>},
    %arg1: tensor<8x8xf32>,
    %arg2: tensor<8x8xf32>,
    %arg3: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c"}p0]>}) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}p1]>]>} : tensor<8x8xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<8x8xf32>
  %2 = stablehlo.divide %1, %arg3 : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>}
  // CHECK-NEXT: stablehlo.divide %[[ADD_1]], %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>}
```

**`{?}p1` 的含义**：这一维目前没有分片轴，但标注了优先级 p1 ——
意思是"轮到 p1 时，这一维可以被分片"。

**注意区分**（L1-02 的不变量）：
- `{?}`（**开**的空维）可以有优先级 → 表示"以后可以切"。
- `{}`（**闭**的空维）**不能**有优先级 → 表示"这一维就是不切"，优先级无处生效。
