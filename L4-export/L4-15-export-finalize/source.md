<!-- sdy-coverage
transforms/export/test/close_shardings.mlir
transforms/export/test/drop_sharding_rules.mlir
transforms/export/test/drop_sharding_and_mesh.mlir
transforms/export/test/inline_meshes.mlir
transforms/export/test/remove_sharding_groups.mlir
transforms/export/test/remove_sub_axes_in_input_output_shardings.mlir
transforms/export/test/update_non_divisible_input_output_shardings.mlir
-->

# L4-15 · export-finalize — 源 IR

**导出流水线的收尾阶段**：7 个 pass 把 IR 清理成交给后端的最终形态。

| pass | 文件 | 行数 | 用例数 |
|---|---|---|---|
| `-sdy-close-shardings` | `close_shardings.mlir` | 92 | 15 |
| `-sdy-update-non-divisible-input-output-shardings` | `update_non_divisible_input_output_shardings.mlir` | 182 | 21 |
| `-sdy-remove-sub-axes-in-input-output-shardings` | `remove_sub_axes_in_input_output_shardings.mlir` | 108 | 6 |
| `-sdy-drop-sharding-rules` | `drop_sharding_rules.mlir` | 35 | 4 |
| `-sdy-remove-sharding-groups` | `remove_sharding_groups.mlir` | 12 | 1 |
| `-sdy-inline-meshes` | `inline_meshes.mlir` | 138 | 9 |
| `-sdy-drop-sharding-and-mesh` | `drop_sharding_and_mesh.mlir` | 26 | 2 |

**合计 593 行 / 58 用例。**

**共同点**：**这 7 个 pass 全都在"删东西"** ——
把传播/导出期用的辅助信息清理掉，交给后端一个**干净**的 IR。

---

## 一、★ 分片的收尾：三个 pass

### ① `close_shardings`：把**开维闭合**

```mlir
// RUN: sdy_opt %s -sdy-close-shardings | FileCheck %s
```

```mlir
sdy.mesh @mesh = <["x"=4, "y"=2]>
```

```mlir
func.func @func_input_sharding_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", ?}, {?}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
```

```mlir
  return %arg0 : tensor<8x16xf32>
}
```

```mlir
// CHECK-LABEL: func @func_input_sharding_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
```

**读法**：
- 输入：`[{"x", ?}, {?}]` —— 第 0 维有 `x` **加一个开维**，第 1 维**只有一个开维**。
- 输出：`[{"x"}, {}]` —— **开维被"闭合"了**。

**什么是开维（`?`）**：
L2-01 讲过 —— `?` 表示"**这一维还有空间可以切**"。
传播期保留它可以让后续的 pass 有调整余地；但**导出时必须定下来** ——
后端需要知道"到底怎么切"。

**"闭合"的含义**：把 `?` 去掉，只保留**实际使用的轴**。
- `{"x", ?}` → `{"x"}`（保留 `x`，丢掉开维）
- `{?}` → `{}`（只有开维 → 变成无分片）

### ② `update_non_divisible_input_output_shardings`：**截断**到可整除前缀

```mlir
// RUN: sdy_opt %s -sdy-update-non-divisible-input-output-shardings -split-input-file | FileCheck %s
```

```mlir
sdy.mesh @mesh_x_4_y_2 = <["x"=4, "y"=2]>
```

```mlir
func.func @only_one_dim_modified(%arg0: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}, {"y"}]>}) -> tensor<2x2xf32> {
```

```mlir
  return %arg0 : tensor<2x2xf32>
}
```

```mlir
// CHECK-LABEL: func @only_one_dim_modified
// CHECK-SAME:    %arg0: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}, {"y"}]>}
```

**读法**（本课最实用的规则）：
- 第 0 维大小是 **2**，但网格上 `x = 4` —— **除不尽**！
- 解法：用**子轴** `{"x":(1)2}` —— 表示"**只用 x 轴的前 2 个设备**"。
- 第 1 维大小 2、`y = 2` —— 整除 ✓ 保持不变。

```mlir
func.func @multiple_dims_modified(%arg0: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}, {"y"}]>}) -> tensor<2x3xf32> {
```

```mlir
  return %arg0 : tensor<2x3xf32>
}
```

```mlir
// CHECK-SAME:    %arg0: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}, {}]>}
```

**读法**（**两种处理方式**）：

| 维 | 大小 | 网格 | 结果 |
|---|---|---|---|
| 第 0 维 | 2 | `x = 4` | **子轴截断** `{"x":(1)2}`（用前 2 个设备） |
| 第 1 维 | 3 | `y = 2` | **直接去掉** `{}`（因为 3 与 2 完全不能整除） |

**关键区分**：
- 能**部分整除**（2 与 4）→ 用**子轴**截断到可整除的前缀。
- **完全不能整除**（3 与 2）→ **整个轴去掉**（变成无分片）。

**为什么必须做**：后端需要知道"每台设备拿多少元素"。
`tensor<2>` 分给 4 台设备是**未定义**的 —— 必须明确成"前 2 台各拿 1 个，后 2 台空闲"。

### ③ `remove_sub_axes_in_input_output_shardings`：**只动边界**

```mlir
// RUN: sdy_opt %s -sdy-remove-sub-axes-in-input-output-shardings -split-input-file | FileCheck %s
```

测试开头的注释点明了规则：

```
// This test check that:
// 1. We remove sub-axes and the trailing axes in input and output shardings.
// 2. We do not modify the shardings for intermediate tensors.
```

```mlir
func.func @main(
    %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
    %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
    -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
        tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
```

```mlir
// CHECK-SAME: %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"z"}]>},
// CHECK-SAME: %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", ?}, {?}]>})
// CHECK-SAME: -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x", "z", ?}]>},
// CHECK-SAME: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>})
```

**逐项读**：

| 位置 | 输入 | 输出 | 删了什么 |
|---|---|---|---|
| `%arg0` 维 0 | `{"x":(1)2, "y", ?}` | `{?}` | `"x":(1)2`（**子轴**）+ `"y"` |
| `%arg0` 维 1 | `{"z"}` | `{"z"}` | **不变**（无子轴） |
| `%arg1` 维 0 | `{"y", "x":(1)2, ?}` | `{"y", ?}` | `"x":(1)2` |
| 结果 0 维 0 | `{"y":(1)2, ?}` | `{?}` | `"y":(1)2` |
| 结果 1 维 0 | `{"y", "z":(4)2}` | `{"y", "z":(4)2}` | **不变** |
| 结果 1 维 1 | `{"z":(2)2, "x"}` | `{"z":(2)2, "x"}` | **不变** |

**规律**（注意 `%arg0` 与 `%arg1` 的差异）：
- `%arg0` 的 `{"x":(1)2, "y", ?}` → `{?}`：**子轴在前**，删掉后只剩 `{?}`。
- `%arg1` 的 `{"y", "x":(1)2, ?}` → `{"y", ?}`：**非子轴的 `"y"` 在前**，被**保留**。

**结论**：移除**子轴**及其**前面**的轴，保留**后面**的轴 ——
即"**只保留最粗的那一段**"。这与"trailing axes"（尾随轴）的表述一致。

**为什么只动输入输出**：
- **中间张量**的分片是导出期**精心算出来的**（L4-02～L4-14），不能动。
- **输入输出**的分片是**函数对外的接口** —— 调用者只需要知道"大概怎么切"，
  不需要知道子轴这种细节。

---

## 二、辅助信息的清理：两个"drop"

### `drop_sharding_rules`：移除分片规则

```mlir
// RUN: sdy_opt %s -sdy-drop-sharding-rules | FileCheck %s
```

```mlir
func.func @dot(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
```

```mlir
  // CHECK: %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
```

**读法**：`sdy.sharding_rule` 属性**被删掉**，`sdy.sharding` **保留**。

**为什么可以删**：分片规则是**传播期**用的（L2-01 讲过"沿规则传播"）。
导出后传播已经结束 —— 规则不再需要。

**回顾 L2-03**：那里有个 `keep-sharding-rules` 选项（**保留**规则便于调试）——
本 pass 就是"最终真的删掉"的那一步。

### `remove_sharding_groups`：移除分片组

```mlir
// RUN: sdy_opt %s -sdy-remove-sharding-groups | FileCheck %s
```

```mlir
func.func @sharding_group_ops(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 : tensor<32x96xf32>
  %1 = stablehlo.add %0, %arg0 : tensor<32x96xf32>
  // CHECK-NOT:   sdy.sharding_group
  sdy.sharding_group %arg0 group_id = 747 : tensor<32x96xf32>
  sdy.sharding_group %0 group_id = 747 : tensor<32x96xf32>
  sdy.sharding_group %1 group_id = 747 : tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
}
```

**读法**：三条 `sdy.sharding_group` 全部**被删**（`CHECK-NOT`）。

**回顾 L3-09**：分片组是"**强制同分片**"的承诺 ——
它在**传播期**作为独立通道生效（L2-07）。传播结束后承诺已经"兑现"，
组本身就没有意义了。

---

## 三、网格的处理：`inline_meshes` 与 `drop_sharding_and_mesh`

### `inline_meshes`：把命名网格**内联**

```mlir
// RUN: sdy_opt -split-input-file %s -sdy-inline-meshes | FileCheck %s
```

```mlir
func.func @no_lifted_meshes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>}) -> tensor<8x8xf32> {
```

```mlir
  return %arg0 : tensor<8x8xf32>
}
```

```mlir
// CHECK-LABEL: func @no_lifted_meshes(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>}
```

```mlir
// CHECK-NOT: sdy.mesh @empty_mesh = <[]>
sdy.mesh @empty_mesh = <[]>
```

**读法**：
- **内联网格** `mesh<["x"=2, "y"=2]>` **保持不变**（本来就是内联的）。
- 顶层的 `sdy.mesh @empty_mesh` 声明**被删除**（`CHECK-NOT`）。

**9 个用例**覆盖：`no_lifted_meshes` / `lifted_empty_mesh` / `lifted_and_inlined_meshes_two_functions` /
`another_function` / `lifted_maximal_mesh` / `single_sharding_sdy_ops` /
`manual_computation` / `named_computation` / `lifted_replica_groups`。

**注意最后几个**：`manual_computation` / `named_computation` 里也有网格引用 ——
说明内联要**递归**处理区域算子内部。

### ★ 与 L3-07 的**镜像关系**

| 课 | pass | 方向 |
|---|---|---|
| **L3-07** | `-sdy-lift-inlined-meshes` | 内联网格 → **顶层声明**（+ 按内容去重） |
| **L4-15** | `-sdy-inline-meshes` | 顶层声明 → **内联** |

**为什么导入要提升、导出要内联**：
- **导入期**：多个内联网格可能**内容相同** → 提升为声明后可以**去重**（L3-07 讲过"按内容去重"）。
- **导出期**：后端要看到**自包含**的 IR —— 每个分片直接写出它用的网格，不依赖外部声明。

### `drop_sharding_and_mesh`：**全部删掉**

```mlir
// RUN: sdy_opt %s -sdy-drop-sharding-and-mesh | FileCheck %s

// CHECK-NOT: sdy.mesh
sdy.mesh @mesh_2 = <["x"=2]>

// CHECK-LABEL: func @drop_sharding
// CHECK-SAME:    %arg0: tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NOT:     sdy.sharding
func.func @drop_sharding(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: stablehlo.add
  // CHECK-NOT:  sdy.sharding
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
```

**读法**：`sdy.mesh` 与所有 `sdy.sharding` **全部消失** ——
IR 变回**纯 StableHLO**。

**这是"完全不用 Shardy"的后端的选项** —— 分片信息已经被后端消化掉了
（比如转换成了具体的集合通信，见 L4-08）。

---

## 四、★ 收尾的**顺序**

TODOLIST 的验收点是"**说出该跑哪几个收尾 pass、顺序如何**"。
按"处理对象"分三个阶段：

| 阶段 | pass | 做什么 |
|---|---|---|
| **① 分片本身** | `close_shardings` | 闭合开维（定下来怎么切） |
| | `update_non_divisible_input_output_shardings` | 不可整除 → 截断到可整除前缀 |
| | `remove_sub_axes_in_input_output_shardings` | 边界上只保留最粗的轴 |
| **② 辅助信息** | `drop_sharding_rules` | 删分片规则（传播已结束） |
| | `remove_sharding_groups` | 删分片组（承诺已兑现） |
| **③ 网格** | `inline_meshes` | 命名网格 → 内联（自包含） |
| | `drop_sharding_and_mesh` | 全部删掉（可选，纯 StableHLO） |

**顺序的逻辑**：
1. **先定分片**（①）—— 因为后续步骤都依赖"分片已经确定"。
2. **再删传播期的辅助信息**（②）—— 它们只在传播中有用。
3. **最后处理网格**（③）—— 因为分片**引用**网格，
   必须先保证分片定下来了，才能安全地内联或删除网格。

**注意 ③ 的两步是"二选一"**：
- 后端**需要**网格信息 → 只做 `inline_meshes`
- 后端**不需要** → 再做 `drop_sharding_and_mesh`

**一句话总结**：
> **收尾阶段全都在"删东西"** —— 先定分片、再删辅助信息、最后处理网格。
> 目标是给后端一个**干净且自包含**的 IR。
