<!-- sdy-coverage
transforms/import/test/manual_axes_cleanup.mlir
transforms/import/test/manual_axes_cleanup_failures.mlir
-->

# L3-08 · manual-axes-cleanup — 源 IR

本课覆盖两个文件：

| 文件 | 行数 | 用例数 | RUN 行 |
|---|---|---|---|
| `transforms/import/test/manual_axes_cleanup.mlir` | 191 | 12 | `-sdy-manual-axes-cleanup` |
| `transforms/import/test/manual_axes_cleanup_failures.mlir` | 10 | 1 | 同上 + `-verify-diagnostics` |

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-manual-axes-cleanup | FileCheck %s
```

**这是 L3-01 里见过的"manual 轴清理"的完整展开**（L1-07 立的不变量：manual 轴必须在所有分片里显式出现）。

网格：

```mlir
sdy.mesh @empty_mesh = <[]>
```

```mlir
sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>
```

```mlir
sdy.mesh @mesh_xyz = <["x"=2, "y"=2, "z"=2]>
```

```mlir
sdy.mesh @mesh_x_8_y_8 = <["x"=8, "y"=8]>
```

> **注意网格的轴序**：`@mesh` 是 **c, a, b**（不是字母序！）。这一点在第三幕很关键。

---

## 一、★ 动作 ①：把没切维度的 manual 轴补进 `replicated`

```mlir
func.func @add_new_replicated(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}]>] out_shardings=[<@mesh, [{"c", ?}]>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
```

**读法**：`manual_axes={"c", "a"}`，但维度分片里只有 `"c"`（`"a"` 没切任何维度）
→ 把 `"a"` 补进 `replicated`。

### 已有 `replicated` 时是**追加**

```mlir
func.func @add_to_existing_replicated(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>] out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>] manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
```

**读法**：已有 `replicated={"a"}`，manual_axes 还多了个 `"b"` → 追加成 `{"a", "b"}`。

### in 与 out 不一致时：**取并集**

```mlir
func.func @add_to_existing_replicated_requires_sorting(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}], replicated={"b"}>] out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>] manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
```

**读法**：in 是 `{"b"}`、out 是 `{"a"}`，两者都缺一个
→ 都补成 `{"a", "b"}`（**并集**，且排序）。

---

## 二、★ 动作 ②：按**网格声明顺序**排序

```mlir
func.func @sort_manual_axes_and_add_out_shardings(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>] out_shardings=[<@mesh, [{"c", ?}], replicated={"b"}>] manual_axes={"b", "a", "c"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
```

**读法**（本课最关键的一处）：
- 输入的 `manual_axes` 是 `{"b", "a", "c"}` —— 看起来像**字母序**。
- 输出的 `manual_axes` 是 `{"c", "a", "b"}`。
- 而 `@mesh = <["c"=2, "a"=2, "b"=2]>` —— **正是网格的声明顺序**！

**结论**：排序依据是**网格里的轴序**，不是字母序。

### 另一个例子确认这一点

```mlir
func.func @inlined_mesh_add_replicated_and_sort(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh_xyz, [{"y", ?}]>]
      out_shardings=[<mesh<["x"=2, "y"=2, "z"=2]>, [{"z", ?}], replicated={"y"}>]
      manual_axes={"y", "x", "z"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_xyz, [{"y", ?}], replicated={"x", "z"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh_xyz, [{"z", ?}], replicated={"x", "y"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"x", "y", "z"} (%arg1: tensor<4xf32>) {
```

**读法**：
- `manual_axes` 输入 `{"y", "x", "z"}` → 输出 `{"x", "y", "z"}`（网格序）。
- **out_sharding 里的内联 mesh 也被提升为 `@mesh_xyz`**（按内容匹配到已有声明）。

---

## 三、动作 ③：空网格被**替换成实际网格**

```mlir
func.func @empty_mesh_by_name_result(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh_xyz, [{"x"}]>]
      out_shardings=[<@empty_mesh, [{}]>]
      manual_axes={"y", "x"} (%arg1: tensor<4xf32>) {
    %1 = "stablehlo.all_gather"(%arg1) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_xyz, [{"x"}], replicated={"y"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh_xyz, [{}], replicated={"x", "y"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"x", "y"} (%arg1: tensor<4xf32>) {
```

**读法**：
- 输入的 `out_shardings` 用的是 `@empty_mesh`，而 `in_shardings` 用 `@mesh_xyz`。
- 输出里 **out 也变成了 `@mesh_xyz`** —— 空网格被替换成"同一个手动计算里实际使用的网格"。
- 替换后才能正确补 `replicated`（因为要按那个网格的轴序排序）。

`empty_mesh_operand` 是**对称的情形**（in 是空网格、out 是实际网格）。

> `inlined_empty_mesh_result` 是同样逻辑但网格**内联**写的 ——
> 输出里 in/out 都变成内联的 `mesh<["x"=2, "y"=2]>`（这个 pass 不做提升，提升是 L3-07 的活）。

---

## 四、三个边界情形

### token 不加 `replicated`

```mlir
func.func @dont_add_manual_axes_to_non_shaped_types(%arg0: !stablehlo.token, %arg1: tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>) {
```

```mlir
  %0:2 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, []>, <@mesh, [{?}]>]
      out_shardings=[<@mesh, []>, <@mesh, [{?}]>]
      manual_axes={"c"} (%arg2: !stablehlo.token, %arg3: tensor<8xf32>) {
    sdy.return %arg2, %arg3 : !stablehlo.token, tensor<8xf32>
  } : (!stablehlo.token, tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>)
  return %0#0, %0#1 : !stablehlo.token, tensor<8xf32>
}
```

```mlir
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, []>, <@mesh, [{?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, []>, <@mesh, [{?}], replicated={"c"}>]
```

**读法**：token 的 sharding 是 `<@mesh, []>`（rank 0）→ **保持原样**，
`replicated` 只加在 **tensor** 的那一项上。

**理由**：rank-0 类型不能有 `replicated`（L1-02 的不变量）。

### 子轴也被正确处理

```mlir
func.func @sub_axes_in_out_shardings(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_x_8_y_8, [{"y":(2)2}]>] out_shardings=[<@mesh_x_8_y_8, [{"y":(1)2}], replicated={"x":(2)2}>] manual_axes={"x", "y"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_x_8_y_8, [{"y":(2)2}], replicated={"x", "y":(1)2, "y":(4)2}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh_x_8_y_8, [{"y":(1)2}], replicated={"x", "y":(2)4}>]
```

**读法**：`@mesh_x_8_y_8 = <["x"=8, "y"=8]>`，`"y"` 被切成 4 个子轴。
- in 用的是子轴 `"y":(2)2` → 其余子轴 `"y":(1)2`、`"y":(4)2` 补进 `replicated`。
- out 用的是 `"y":(1)2` 且已有 `replicated={"x":(2)2}` → 结果 `{"x", "y":(2)4}`。

**要点**：子轴会被**展开成完整的子轴集合**，而不是简单地加一个 `"y"`。

### `unreduced` 被保留

```mlir
func.func @manual_computation_with_reduction_op(%arg0: tensor<8xf32>) -> tensor<8xf32> {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}], unreduced=max{"b"}>] out_shardings=[<@mesh, [{"c", ?}], unreduced=max{"b"}>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
```

```mlir
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}, unreduced=max{"b"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}, unreduced=max{"b"}>]
```

**读法**：补 `replicated={"a"}` 的同时，**`unreduced=max{"b"}` 原样保留**。
三种状态（分片 / 复制 / 未归约）可以共存。

---

## 五、失败情形

```mlir
func.func @manual_computation_no_inputs_or_outputs_with_manual_axes() {
```

```mlir
  // expected-error @+1 {{op has manual_axes when there are no in/out shardings and the body is not empty}}
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={"a"} () {
    %0 = sdy.constant dense<1.000000e+00> : tensor<8xf32>
    sdy.return
  } : () -> ()
  func.return
}
```

**读法**：`in_shardings=[]` 且 `out_shardings=[]`（没有分片可补），
但 `manual_axes={"a"}` 非空，**且函数体非空** → 报错。

**为什么是错误**：manual 轴需要"有地方可写"（in/out_shardings 的 `replicated`）。
两者都空时，`manual_axes` 无处安放 —— 说明用户写错了。

> 注意报文里的 `and the body is not empty` —— 如果**函数体是空的**，
> 这种写法是允许的（那是 L1-07 讲的"空体直接内联"的情形）。
