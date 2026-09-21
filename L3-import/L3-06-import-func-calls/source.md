<!-- sdy-coverage
transforms/import/test/import_func_calls.mlir
transforms/import/test/import_func_calls_add_data_flow_edges_on_named_computations_false.mlir
transforms/import/test/flatten_call_graph.mlir
-->

# L3-06 · import-func-calls — 源 IR

本课覆盖三个文件：

| 文件 | 行数 | 用例数 | RUN 行 |
|---|---|---|---|
| `transforms/import/test/import_func_calls.mlir` | 634 | 21 | `-sdy-import-func-calls` |
| `transforms/import/test/import_func_calls_add_data_flow_edges_on_named_computations_false.mlir` | 572 | 21 | `-sdy-import-func-calls='add-data-flow-edges-on-named-computations=false'` |
| `transforms/import/test/flatten_call_graph.mlir` | 415 | 13 | `-sdy-flatten-call-graph` |

```mlir
// RUN: sdy_opt --split-input-file %s -sdy-import-func-calls | FileCheck %s
```

**这一课讲的是前面几课反复提到的那个动作**：把 `func.call` **内联**成 `sdy.named_computation`（L1-08 讲过它的语法）。

---

## 一、★ 核心变换：`call` → `named_computation`

```mlir
func.func @backend_config_no_out_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = call @foo(%arg0) {random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
```

```mlir
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
```

```mlir
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
```

期望输出：

```mlir
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg1 : tensor<8x2xi32>
  // CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %[[EDGE_1]], %[[EDGE_1]] {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value"}
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: %[[MOVE_TO_HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[EDGE_2]]) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[MOVE_TO_HOST]] : tensor<8x2xi32>
```

```mlir
// CHECK-NOT: func private @foo
```

**读法**（四件事同时发生）：
1. `call @foo(%arg0)` → `sdy.named_computation<"foo">(%arg0)`，
   名字取自**被调函数名**。
2. **函数体被搬进区域**（`stablehlo.multiply ...`），区域内插了 `data_flow_edge`（L3-04 的规则）。
3. **调用点的属性被搬到 `named_computation` 上**：
   `random_attr`、`mhlo.frontend_attributes`（含 `backend_config`）都跟着过来了。
4. **原来的私有函数被删除**（`CHECK-NOT: func private @foo`）。

---

## 二、★ `in_shardings` 与 `out_shardings` 的**来源不同**

这是本课最容易搞错的一点：

| 字段 | 来自 |
|---|---|
| `in_shardings` | **被调函数**的参数分片标注 |
| `out_shardings` | **调用点**上的 `sdy.sharding` |

### 例子 A：两者都有

```mlir
func.func @single_call_multiple_args_func_all_arguments_with_input_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
```

```mlir
  %0 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
```

```mlir
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg1 sharding=<@mesh, [{"x"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{}, {"y"}]> : tensor<8x2xi32>
```

**读法**：
- `in_shardings` 有 **2 项**（对应 2 个参数）：`[{"x"}, {}]` 与 `[{}, {"y"}]`
  —— 来自**函数 `@foo` 的参数标注**。
- `out_shardings` 有 **1 项**（对应 1 个结果）：`[{"y"}, {"x"}]`
  —— 来自**调用点的 `sdy.sharding`**。
- 区域内的边**继承了 `in_shardings`**（这正是 L3-04 讲的规则）。

### 例子 B：函数上有、调用点没有 → **忽略函数上的**

```mlir
func.func @func_has_out_sharding_call_no_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
```

```mlir
  %0 = call @foo(%arg0, %arg0) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
```

```mlir
// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
```

```mlir
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
```

**读法**：函数结果上有 `[{"x", "y"}, {}]`，但调用点**没有** `sdy.sharding`
→ `named_computation` **没有 `out_shardings`**。函数上的标注被**忽略**。

### 例子 C：函数上没有、调用点有 → **采用调用点的**

```mlir
func.func @func_no_out_sharding_call_has_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
```

```mlir
  %0 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}
```

```mlir
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
```

```mlir
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[NC]] sharding=<@mesh, [{"y"}, {"x"}]> : tensor<8x2xi32>
```

**读法**：调用点有 `[{"y"}, {"x"}]` → `out_shardings` 就是它，且结果上的边也带上它。

> 测试里的 NOTE 点明了这一点：
> `// NOTE: we ignore any arg/result shardings on the function.`

---

## 三、属性搬运

调用点上的属性会**原样搬到 `named_computation` 上**：

```mlir
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}}
```

```mlir
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "true"}}
```

覆盖的用例：
- `backend_config_no_out_shardings` / `backend_config_out_shardings` —— `backend_config` 与 `random_attr`
- `inlineable_false` / `inlineable_true` —— `inlineable` 属性
- `no_backend_config_or_inlineable_attr` —— 两者都没有

---

## 四、多个调用同一个函数

| 用例 | 场景 |
|---|---|
| `multiple_call_ops_same_name` | 同一函数被调多次 → 每个调用点各生成一个 `named_computation`，**同名** |
| `multiple_call_ops_same_name_func_no_input_output_shardings` | 同上，但函数无 in/out 分片 |
| `non_flat_call_graph_all_uninlineable` | 调用图有多层，全不可内联 |
| `non_flat_call_graph_all_inlineable` | 调用图有多层，全可内联 |

---

## 五、函数名与原名的关系

```mlir
func.func @func_name_and_original_names_different(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
```

该用例验证：`named_computation` 的**名字**取自**被调函数名**，
而不是调用点所在函数名或其它名字。

---

## 六、`add-data-flow-edges-on-named-computations=false`

第二个文件的 RUN 行多了一个选项：

```mlir
// RUN: sdy_opt --split-input-file %s -sdy-import-func-calls='add-data-flow-edges-on-named-computations=false' | FileCheck %s
```

**作用**：生成 `named_computation` 时**不**在区域内插数据流边。

**为什么需要这个开关**：插边是有代价的（IR 变大）。
如果后续有单独的 pass 统一插边（如 L3-04 的 `-sdy-add-data-flow-edges`），
这里就可以关掉，避免重复工作。

该文件与主文件**用例一一对应**（都是 21 个），只是输出里少了区域内/结果上的边。

---

## 七、`flatten_call_graph`：调用图展平

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-flatten-call-graph | FileCheck %s
```

415 行 / 13 个用例。这个 pass 处理**调用图结构**：

| 用例 | 场景 |
|---|---|
| `singleton` | 只有一个函数 |
| `simple_call_graph` | `main → foo` |
| `main_calls_foo_twice` | `main` 调 `foo` 两次 |
| `main_calls_foo_calls_bar` | `main → foo → bar` |
| `main_calls_foo_calls_bar_twice` | 链式 + 重复调用 |
| `simple_non_flat` | 非扁平调用图 |
| `main_calls_foo_twice_and_foo_calls_bar_twice` | 双向重复 |
| `main_calls_foo_twice_and_foo_calls_bar_once` | 混合 |
| `simple_non_flat_sharding_on_func_arguments` | 函数参数上有分片 |
| `simple_non_flat_sharding_on_func_results` | 函数结果上有分片 |
| `simple_non_flat_non_matching_sharding_on_func_results_and_second_call_results` | 分片不匹配 |
| `simple_non_flat_non_matching_sharding_on_func_results_and_both_call_results` | 分片不匹配（两个调用） |
| `simple_non_flat_with_manual_computations` | 与 `manual_computation` 共存 |

**关注点**：展平时如何处理**分片标注**（函数上的 vs 调用点上的），
以及**不匹配**时怎么办 —— 这正是 L3-06 与 L3-05 的交汇处。
