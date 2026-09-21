<!-- sdy-coverage
transforms/import/test/apply_sharding_constraints.mlir
transforms/import/test/apply_sharding_constraints_preinlined.mlir
-->

# L3-03 · apply-sharding-constraints — 源 IR

本课覆盖两个文件：

| 文件 | 行数 | 用例数 | RUN 行 |
|---|---|---|---|
| `transforms/import/test/apply_sharding_constraints.mlir` | 716 | 39 | `-sdy-apply-sharding-constraints` |
| `transforms/import/test/apply_sharding_constraints_preinlined.mlir` | 68 | 5 | `-sdy-import-func-calls -sdy-apply-sharding-constraints` |

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-apply-sharding-constraints | FileCheck %s
```

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-import-func-calls -sdy-apply-sharding-constraints | FileCheck %s
```

**本课要回答的问题**：`sdy.sharding_constraint` 描述的是"某个使用者看到的分片"（L1-09）。
那么它到底该**贴**在哪个值上？能不能"下沉"到操作数上？

---

## 一、★ 下沉：把约束挪到操作数上

```mlir
func.func @input_has_one_use(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
```

**读法**：`%0` 只有**一个**使用者（就是这个约束），且 `%0` 自己没有分片标注。
→ 流水线把约束的 `[{}, {"b"}]` **直接写到 `%0`（add）上**。

**好处**：传播时不用再"穿过"一条约束算子 —— 分片信息直接在算子上，减少一层间接。

---

## 二、下沉的**三个条件**

### 条件 ①：操作数**没有**自己的分片标注

```mlir
func.func @input_already_has_sharding(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
```

**读法**：`%0` **已经**有分片 `[{"a", ?}, {?}]`，与约束的 `[{}, {"b"}]` 不同。
→ **不下沉**：add 保留自己的分片，约束仍然留在结果上（`return %1`）。

**为什么**：覆盖已有的分片会改变算子的语义 —— 用户明确写了 `{"a", ?}`，不能被约束顶掉。

### 条件 ②：约束必须是**闭维**的

```mlir
func.func @open_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b", ?}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
```

**读法**：约束是 `[{}, {"b", ?}]` —— 第 1 维带 **`?`（开维）**。
→ **不下沉**（`CHECK-NOT: sdy.sharding`）。

**为什么**：开维约束的意思是"**至少**按 b 切，还可以继续加轴"。
下沉到算子上会变成算子的确定分片，**过度约束**了后续传播。

### 条件 ③：操作数**没有其它不同的**约束使用者

**相同的约束 → 可以下沉**：

```mlir
func.func @has_other_identical_sharding_constraint_user(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
```

**不同的约束 → 不下沉**：

```mlir
func.func @has_different_sharding_constraint_user(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
```

**读法**：两个约束分别要求 `[{}, {"b"}]` 与 `[{"a"}, {}]` —— 互不相容。
下沉只能满足一个 → **都不下沉**。

---

## 三、其它使用者**不**妨碍下沉

```mlir
func.func @no_other_sharding_constraint_users(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>,  tensor<8x8xf32>, tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
```

**读法**：`%0` 还有别的使用者（`%2 = add`）以及被 `return` —— 但**照样下沉**。

**关键区分**：妨碍下沉的是「其它**约束**使用者」（且要求不同），
不是「其它使用者」。普通使用者的分片由传播去协调。

---

## 四、约束**链**：逐个下沉

```mlir
func.func @chain_of_two_sharding_constraints(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %3 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[WSC_1]], %[[WSC_1]]
  // CHECK-NEXT: return %[[WSC_1]], %[[ADD_1]]
```

**读法**：
- 链头 `%1`（要求 `[{"a"}, {}]`）下沉到了 `%0` → `%[[ADD_0]]` 拿到 `[{"a"}, {}]`。
- `%2`（要求 `[{}, {"b"}]`）**没有**下沉 —— 它的操作数 `%1` 是另一条约束。
- 两条约束算子**都还在**，链的结构保留。

**结论**：下沉是**逐个**处理的，一次只处理链上的一条。

### 链头已有分片时

```mlir
func.func @input_of_sharding_constraint_chain_head_has_sharding(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %3 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %[[ADD_0]]
```

**读法**：`%0` 已有 `[{"a"}, {}]`，与链头约束**相同** —— 但约束算子**仍然保留**
（因为 `%0` 已带分片，不满足下沉条件 ①；而约束本身也没有被删除）。

---

## 五、悬空约束

```mlir
func.func @dangling_and_no_other_sharding_constraint_users(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %0, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}
```

```mlir
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
```

**读法**：`%1`（约束）**没有被任何算子使用**（悬空），但 `%0` 没有其它**约束**使用者
→ 照样**下沉**。

**含义**：悬空约束表达的是"`%0` **本身**应该是这样切的"（L1-09）。
把它写到 `%0` 上正是它想表达的语义。

---

## 六、`preinlined` 版本：先内联函数调用

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-import-func-calls -sdy-apply-sharding-constraints | FileCheck %s
```

`apply_sharding_constraints_preinlined.mlir`（68 行 / 5 个用例）用**两步**流水线：

1. `-sdy-import-func-calls` —— 先把函数调用内联成 `sdy.named_computation`（L1-08 / L3-06）
2. `-sdy-apply-sharding-constraints` —— 再应用约束

**为什么需要这个变体**：函数调用边界会挡住约束的下沉分析。
先内联之后，约束的输入变成区域内的值，下沉规则才能在**区域内部**正常生效。

> 该文件的用例名（`foo` / `main` / `bar`）说明它们都涉及跨函数的结构。

---

## 七、用例族谱（39 + 5）

| 族 | 代表用例 | 数量 |
|---|---|---|
| **基础下沉** | `input_has_one_use`、`input_already_has_sharding`、`open_sharding_constraint` | 3 |
| **无法附加** | `cannot_attach_sharding_to_input`、`input_is_func_input_with_one_use` | 2 |
| **使用者分析** | `no_other_sharding_constraint_users`、`has_different_/identical_sharding_constraint_user` | 3 |
| **悬空约束** | `dangling_and_*`（3 个） | 3 |
| **约束链** | `chain_of_two/three_sharding_constraints`、`*_in_chain_*`（约 9 个） | 9 |
| **数据流边** | `input_produced_by_data_flow_edge*`、`*_non_owner_target_*`（4 个） | 4 |
| **manual_computation** | `manual_computation*`、`has_other_manual_computation_user_*`（5 个） | 5 |
| **未归约** | `unreduced_sharding_input_*`（2 个） | 2 |
| **其它** | `foo` / `main` / `bar` 等跨函数用例 | 8 |
