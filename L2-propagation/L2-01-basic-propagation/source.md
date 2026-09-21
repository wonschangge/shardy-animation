<!-- sdy-coverage
transforms/propagation/test/basic_propagation.mlir
-->

# L2-01 · basic-propagation — 源 IR

本课覆盖 **L2 最大的单文件**：`shardy/dialect/sdy/transforms/propagation/test/basic_propagation.mlir`（1106 行）。

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-basic-propagate -verify-diagnostics | FileCheck %s
```

该文件共 **79 个用例**，按算子族可归成十几组。本课选讲 **12 个代表性用例**，
其余（reshape 的 20 余个变体、不可整除的 10 余个变体等）在课件里按族归纳。

网格（节选）：

```mlir
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
```

```mlir
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>
```

```mlir
sdy.mesh @mesh_a_3 = <["a"=3]>
```

```mlir
sdy.mesh @mesh_a_4_b_2 = <["a"=4, "b"=2]>
```

```mlir
sdy.mesh @mesh_a_6 = <["a"=6]>
```

```mlir
sdy.mesh @empty_mesh = <[]>
```

```mlir
sdy.mesh @maximal_mesh = <[], device_ids=[0]>
```

> **关于 `%[[NAME:.*]]`**：CHECK 行里的这种记号是 **FileCheck 变量**，不是字面 IR。
> 它表示"这里捕获一个值名，后面引用"。展示时请把它读作"某个 SSA 值"。

---

## 一、最基本的传播

输入 `%arg0` 已有分片 `[{"a"}, {"b"}]`，`%arg1`、`%arg2` 完全没有分片：

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
```

传播后（CHECK 行）：

```mlir
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
```

```mlir
  // CHECK-NEXT: stablehlo.dot_general %[[ADD]], %arg2
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {?}]>]>}
```

**要点**：`add` 是逐元素算子，两维都继承 `%arg0` 的分片，但都**多了一个 `?`** —— 即变成<b>开维</b>，
表示"已经按 a/b 切了，但还可以继续加轴"。

---

## 二、闭维：锁死就不再加 `?`

`%arg1` 的第 1 维是**闭的** `{"b"}`：

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{}, {?}]>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
```

期望（CHECK 行）：

```mlir
  // CHECK-NEXT: stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{}, {"b", ?}]>]>}
```

**要点**：第 1 维既继承了 `"b"`，又保留了**开维**标记 —— 闭的是"用户指定的那一维"，
传播加上的部分仍然开放。

---

## 三、冲突与消解：只有"兼容前缀"能传过去

三个输入分别要求 `{"b","a"}`、`{"b","c"}`、`{"a","b"}`：

```mlir
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg2 : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
```

期望（CHECK 行）：

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"b", ?}, {?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %arg0, %arg2 : tensor<8x8xf32>
```

**要点**：`%arg0` 与 `%arg1` 的**最长兼容前缀**是 `"b"`，所以只传 `"b"`。
而 `%arg0` 与 `%arg2` 的轴序相反（`{"b","a"}` vs `{"a","b"}`）→ **一个都不传**（注释里没有 sharding）。
这正是"basic 传播不做冲突消解"的含义。

---

## 四、reshape：沿因子传播

把 `2x4` 合并成 `8`，只有**最 major 的因子**被切：

```mlir
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
```

期望（CHECK 行）：

```mlir
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}]>]>}
```

**要点**：`"a"` 切的是 `i` 因子（第 0 维）。合并成 `[ij]` 后，`"a"` 仍然有效，
所以结果第 0 维得到 `{"a", ?}`。若被切的是**非最 major** 因子（如 `j`），就无法直接对应，
需要拆分子轴 —— 这正是 L1-02 讲子轴的动机。

---

## 五、子轴：完整轴落到子轴上

输入第 0 维是完整轴 `{"a", ?}`：

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4_b_2, [{"a":(1)2, ?}, {}]>]>} : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
```

**要点**：该算子的结果**显式标注**了 `{"a":(1)2, ?}`。传播把完整轴 `"a"` 与子轴
`"a":(1)2` 对应起来 —— 因为 `"a"` 的前半段正是 `"a":(1)2`。

---

## 六、不可整除：允许，但维度会变成开维

网格 `a=3`，张量第 1 维大小 4，已按 `"a"` 切（4 不能被 3 整除）：

```mlir
  %0 = stablehlo.add %arg0, %arg0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
```

期望（CHECK 行）：

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_3, [{?}, {"a", ?}]>]>}
```

**要点**：不可整除**不是错误**（L1-02 已说明）。传播照常进行，第 0 维变成 `{?}` 表示
"还没决定，且开放"。是否需要 padding 由导出阶段决定（L5-09）。

---

## 七、跨网格：不传播

输入用 `@mesh_a_3`，函数结果声明用 `@mesh_a_6`：

```mlir
  %0 = stablehlo.tanh %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
```

期望（CHECK 行）：

```mlir
  // CHECK-NEXT: stablehlo.tanh %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_3, [{"a", ?}, {?}]>]>}
```

**要点**：结果只继承了**操作数所在网格**的分片，**不会**跨到 `@mesh_a_6`。
测试里还有一组用例验证：不同网格名但**实际是同一个网格**时**可以**传播
（`different_mesh_names_same_mesh_propagated`）。

---

## 八、空网格与 maximal 网格的特殊处理

- `empty_mesh_replaced_closed_dim_respected`：空网格会被**替换**成真实网格，但闭维仍然被尊重。
- `empty_mesh_all_dims_closed`：所有维都闭时，空网格**不**被替换。
- `maximal_mesh_not_replaced`：maximal 网格（`<[], device_ids=[0]>`）**不会**被替换。
- `do_not_propagate_along_maximal_mesh`：不沿 maximal 网格传播。
- `does_propagate_to_empty_mesh` / `does_not_propagate_to_empty_mesh_with_closed_sharding`：
  空网格的开维可以被填充，闭维不行。
