<!-- sdy-coverage
transforms/import/test/lift_inlined_meshes.mlir
-->

# L3-07 · lift-inlined-meshes — 源 IR

覆盖：`shardy/dialect/sdy/transforms/import/test/lift_inlined_meshes.mlir`（286 行 / 15 个用例）。

```mlir
// RUN: sdy_opt -split-input-file %s -sdy-lift-inlined-meshes | FileCheck %s
```

**这个 pass 做两件事**：
1. **提升（lift）**：把内联写的网格定义 `mesh<["x"=2]>` 提升为**顶层声明**并起名。
2. **去重（dedup）**：内容相同的声明**合并**成一个。

---

## 一、★ 核心变换：内联 → 顶层声明

```mlir
sdy.mesh @mesh = <["x"=2, "y"=4]>
```

```mlir
func.func @inlined_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

期望输出：

```mlir
// CHECK: sdy.mesh @mesh_0 = <["x"=2, "y"=2]>
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"x"}, {}]>]>
```

**读法**：
- 输入里 `add` 的分片用的是**内联 mesh** `<mesh<["x"=2, "y"=2]>, ...>`。
- 输出里多了一条**顶层声明** `sdy.mesh @mesh_0 = <["x"=2, "y"=2]>`，
  内联定义被替换成**命名引用** `@mesh_0`。

**为什么要提升**：内联定义**无法被其它地方引用**。
提升为顶层声明后，同一个网格可以被多个分片共享 —— 这也是去重的前提。

---

## 二、★ 去重：内容相同就复用

```mlir
sdy.mesh @mesh1 = <["x"=2, "y"=2]>
```

```mlir
sdy.mesh @mesh2 = <["x"=2, "y"=4]>
```

```mlir
sdy.mesh @mesh3 = <["x"=2, "y"=2]>
```

```mlir
func.func @no_inlined_meshes_with_duplicates(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}]>}) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{}, {"y"}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

期望输出：

```mlir
// CHECK-NOT: sdy.mesh @mesh3
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"y"}]>]>
```

**读法**：
- `@mesh3` 的内容与 `@mesh1` **完全相同**（都是 `<["x"=2, "y"=2]>`）。
- 输出里 `@mesh3` **被删除**（`CHECK-NOT`），原来的引用改成了 `@mesh1`。

**结论**：去重是**按内容**的，与名字无关。

### 更复杂的一例

```mlir
sdy.mesh @mesh_1 = <["x"=2, "y"=4]>
```

```mlir
sdy.mesh @copy_of_mesh = <["x"=2, "y"=4]>
```

```mlir
sdy.mesh @copy_of_mesh_0 = <["x"=2], device_ids=[1, 0]>
```

```mlir
// CHECK-NOT: @mesh_1
```

```mlir
// CHECK-NOT: @copy_of_mesh
```

```mlir
// CHECK-NOT: @copy_of_mesh_0
```

三条重复声明全部被删除。

---

## 三、命名规则

| 内联定义 | 生成的名字 | 说明 |
|---|---|---|
| `mesh<["x"=2, "y"=2]>` | `@mesh_0` | 基础名 `mesh` + 编号（因为 `@mesh` 已被占用且内容不同） |
| `mesh<[]>` | `@empty_mesh` | **特殊名** |
| `mesh<[], device_ids=[3]>` | `@maximal_mesh_3` | **特殊名 + 设备号** |
| `mesh<[], device_ids=[7]>` | `@maximal_mesh_7` | 设备号不同 → 名字不同 |
| `mesh<["b"=2]>` | `@mesh_0` | 若 `@mesh` 已被占用 |

### 空网格与 maximal 网格

```mlir
func.func @inlined_empty_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<[]>, [{}, {}]>}) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<[]>, [{}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

```mlir
// CHECK: sdy.mesh @empty_mesh = <[]>
```

```mlir
func.func @inlined_maximal_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<[], device_ids=[3]>, []>}) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<[], device_ids=[7]>, []>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

```mlir
// CHECK: sdy.mesh @maximal_mesh_3 = <[], device_ids=[3]>
// CHECK: sdy.mesh @maximal_mesh_7 = <[], device_ids=[7]>
```

**读法**：`<[]>` → `@empty_mesh`；`<[], device_ids=[N]>` → `@maximal_mesh_N`。
**设备号进入名字**，所以 `device_ids=[3]` 与 `[7]` 是两个不同的网格，不会被去重。

---

## 四、覆盖范围：哪些地方的内联 mesh 会被提升

内联 mesh 可以出现在任何带 `sdy.sharding` 的地方，这个 pass 全部覆盖：

| 用例 | 位置 |
|---|---|
| `single_sharding_sdy_ops` | 单个 SDY 算子上的分片 |
| `manual_computation` | `sdy.manual_computation` 的 in/out_shardings |
| `single_call` | 函数调用 |
| `all_reduce_inlined_mesh` | `sdy.all_reduce` |
| `inlined_mesh_on_call_result` | **调用结果**上 |
| `non_flat_graph_inlined_mesh` | 非扁平调用图 |
| `inlined_mesh_with_reduction_op` | 带归约的算子 |
| `tagged_stablehlo_mesh_attribute` | 带 tag 的 stablehlo mesh 属性 |

### `tagged_stablehlo_mesh_attribute`

```mlir
func.func @tagged_stablehlo_mesh_attribute(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["b"=2]>, [{"b"}, {}]>]>} : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}, {}]>]>}
```

**读法**：即使内联 mesh 与 `@mesh` 的**轴名不同**（`"b"` vs `"a"`），
只要 `@mesh` 已被占用，新的就编号成 `@mesh_0`。

---

## 五、用例族谱（15 个）

| 族 | 用例 | 说明 |
|---|---|---|
| **无内联 mesh** | `no_inlined_meshes_or_duplicates`、`no_inlined_meshes_with_duplicates` | 只做去重 |
| **基础提升** | `inlined_mesh`、`inlined_empty_mesh`、`inlined_maximal_mesh` | 三种命名 |
| **多网格与去重** | `many_inlined_meshes_and_duplicates` | 提升 + 去重混合 |
| **作用域** | `another_func_in_module` | 模块内的其它函数 |
| **各类 SDY 算子** | `single_sharding_sdy_ops`、`manual_computation`、`single_call`、`all_reduce_inlined_mesh`、`inlined_mesh_with_reduction_op` | |
| **特殊位置** | `inlined_mesh_on_call_result`、`non_flat_graph_inlined_mesh`、`tagged_stablehlo_mesh_attribute` | |

---

## 六、与其它课的关系

| 课 | 关系 |
|---|---|
| L1-01 | 讲了 `sdy.mesh` 的**语法**（顶层声明） |
| L2-11 | `inlined_mesh` 用例里见过这个现象的**结果** |
| **本课** | 讲**提升与去重的完整规则** |
