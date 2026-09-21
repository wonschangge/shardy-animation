<!-- sdy-coverage
transforms/export/test/export_pipeline.mlir
transforms/export/test/export_pipeline_explicit_collectives.mlir
-->

# L4-01 · export-pipeline-overview — 源 IR

**L4 层的开篇**。L3 讲"传播之前做了什么"，L4 讲"传播之后分片怎么落地"。

| 文件 | 行数 | RUN 行 |
|---|---|---|
| `transforms/export/test/export_pipeline.mlir` | 146 | `-sdy-add-data-flow-edges -sdy-export-pipeline` |
| `transforms/export/test/export_pipeline_explicit_collectives.mlir` | 182 | `-sdy-export-pipeline='enable-insert-explicit-collectives=true remove-all-gather-reduce-scatter-for-cmv1=true mark-partial-result-with-unreduced-axes=true'` |

**两个文件的差异就是本课的主线**：同一个导出流水线，**是否启用"显式插入集合通信"**。

---

## 一、第一个文件：默认导出流水线

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-export-pipeline | FileCheck %s
```

开头的 NOTE 说明了为什么要先跑 `-sdy-add-data-flow-edges`：

```
// NOTE: We apply `sdy-add-data-flow-edges` first, to make sure
// `sdy-sink-data-flow-edges` is applied before any pass that operated on
// `ShardableDataFlowOpInterface` rather than `DataFlowEdgeOp`.
```

**读法**：导出流水线内部的 `-sdy-sink-data-flow-edges` 需要**边已经存在**
（L3-04 负责插边）。这是一条 **pass 顺序的隐式契约** —— 与 L3-01 讲的"先插边、后约束"同类。

### 用例 ①：`manual_computation_free_axes_non_divisible`

```mlir
func.func @main(
    %arg0: tensor<4xf32>, %arg1: tensor<12xf32>, %arg2: tensor<24xf32>,
    %arg3: tensor<48xf32>, %arg4: tensor<96xf32>, %arg5: tensor<192xf32>)
    -> (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>,
        tensor<48xf32>, tensor<96xf32>, tensor<192xf32>) {
```

```mlir
  // CHECK-NEXT: sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
  // CHECK-SAME:   in_shardings=[<mesh<["a"=4, "b"=4, "c"=4]>, [{"a"}]>, <mesh<["a"=4, "b"=4, "c"=4]>, [{"a"}]>,
  // CHECK-SAME:                 <mesh<["a"=4, "b"=4, "c"=4]>, [{"a", "b":(1)2}]>, <mesh<["a"=4, "b"=4, "c"=4]>, [{"a", "b"}]>,
  // CHECK-SAME:                 <mesh<["a"=4, "b"=4, "c"=4]>, [{"a", "b", "c":(1)2}]>, <mesh<["a"=4, "b"=4, "c"=4]>, [{"a", "b", "c"}]>]
  // CHECK-SAME:   out_shardings=[<mesh<["a"=4, "b"=4, "c"=4]>, [{"a"}]>, <mesh<["a"=4, "b"=4, "c"=4]>, [{"a"}]>,
  // CHECK-SAME:                  <mesh<["a"=4, "b"=4, "c"=4]>, [{"a", "b":(1)2}]>, <mesh<["a"=4, "b"=4, "c"=4]>, [{"a", "b"}]>,
  // CHECK-SAME:                  <mesh<["a"=4, "b"=4, "c"=4]>, [{"a", "b", "c":(1)2}]>, <mesh<["a"=4, "b"=4, "c"=4]>, [{"a", "b", "c"}]>]
  // CHECK-SAME:   manual_axes={"a"}
```

**读法**（本课最有信息量的一处）：
- 输入里 **6 个操作数**的 in/out_shardings 全都写成 `[{"a", "b", "c"}]`（完全相同的分片）。
- 输出里它们变成了**六种不同的分片**：
  `[{"a"}]`、`[{"a"}]`、`[{"a", "b":(1)2}]`、`[{"a", "b"}]`、`[{"a", "b", "c":(1)2}]`、`[{"a", "b", "c"}]`
- 而且**网格从命名引用 `@mesh3d` 变成了内联写法** `mesh<["a"=4, "b"=4, "c"=4]>`。

**为什么**：导出流水线要为每个操作数**计算它真正需要的分片**。
用例名里的 `free_axes_non_divisible` 点明了挑战：自由轴上有**不可整除**的情形
（如 `tensor<12>` 沿 `b=4` 切不干净），所以出现 `"b":(1)2` 这样的**子轴**。

> 注意：**内联网格**出现在输出里 —— 导出阶段不像 L3-07 那样做提升。

### 用例 ②：`all_reduce_of_replicated_to_unreduced`

```mlir
func.func @main(
      %arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"c"}, {}], unreduced={"b"}>})
      -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"c"}, {}], unreduced={"b"}>}) {
```

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh3d, [{"c"}, {}], unreduced={"a", "b"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
```

```mlir
  // CHECK-NEXT: %0 = sdy.replicated_to_unreduced
  // CHECK-NEXT: return %arg0 : tensor<16x8xf32>
```

**读法**：
- 输入把 `unreduced={"b"}` 扩大成 `unreduced={"a", "b"}` —— 即"`a` 也从已归约变成未归约"。
- 输出只插了一条 `sdy.replicated_to_unreduced`，且**直接返回 `%arg0`**（没有经过约束）。
- 含义：`"a"` 原本是**复制**状态（不在分片、也不在 unreduced 里），
  现在要变成**未归约** —— 这需要一条"从复制到未归约"的转换算子。

### 用例 ③：`update_input_output_shardings`

```mlir
func.func @main(
    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"a", ?}, {"b"}]>})
    ->    (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"a":(1)2, ?}, {"b", "c"}]>}) {
```

```mlir
// CHECK-LABEL: func @main
// CHECK-SAME:    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=4, "b"=4, "c"=4]>, [{}, {"b"}]>}
// CHECK-SAME:    -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=4, "b"=4, "c"=4]>, [{}, {"b"}]>})
```

**读法**：**函数参数与函数结果的分片被"对齐"了** ——
输入侧 `[{"a", ?}, {"b"}]` 与结果侧 `[{"a":(1)2, ?}, {"b","c"}]` 不一致，
导出后**两者都变成 `[{}, {"b"}]`**。

**为什么**：函数边界上的分片必须**自洽** —— 否则调用者与被调者对不上。

### 用例 ④：`call_with_shardings`

用例名说明它覆盖"调用点带分片"的情形 —— 与 L3-06 / L3-10 讲的
`out_shardings` 来源（调用点）呼应。

---

## 二、第二个文件：显式插入集合通信

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-export-pipeline='enable-insert-explicit-collectives=true remove-all-gather-reduce-scatter-for-cmv1=true mark-partial-result-with-unreduced-axes=true' 2>&1 | FileCheck %s
```

**三个选项**：

| 选项 | 作用 |
|---|---|
| `enable-insert-explicit-collectives=true` | 显式插入集合通信算子 |
| `remove-all-gather-reduce-scatter-for-cmv1=true` | 为 CMV1 移除 all-gather/reduce-scatter |
| `mark-partial-result-with-unreduced-axes=true` | 用 unreduced 轴标记部分结果 |

### ★ 用例 ①：`reduce_scatter_fusion` —— 融合

```mlir
func.func @reduce_scatter_fusion(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<16x8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{"x"}, {"y"}]> : tensor<16x8xf32>
  return %2 : tensor<16x8xf32>
}
```

```mlir
  // CHECK: %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2, "z"=2]>, [{}, {}], unreduced={"x"}>]>} : (tensor<16x8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<mesh<["x"=2, "y"=2, "z"=2]>, [{"x"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %2 = sdy.all_slice [{}, {"y"}] %1 out_sharding=<mesh<["x"=2, "y"=2, "z"=2]>, [{"x"}, {"y"}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %2 : tensor<16x8xf32>
```

**读法**（本课的核心机制）：
- 输入：`reduce`（沿第 1 维归约）+ `sharding_constraint`（要求 `[{"x"}, {"y"}]`）。
- 输出：`reduce` 被标上 `unreduced={"x"}`（表示归约后 `"x"` 上是**部分和**），
  然后接 **`sdy.reduce_scatter [{"x"}, {}]`** —— 把部分和一边归约一边重新切分，
  最后接 **`sdy.all_slice [{}, {"y"}]`** 调整到目标分片。

**关键洞察**：`reduce` 与 `reshard` **被融合成了一条 `reduce_scatter`** ——
而不是"先 all-reduce 再 slice"两步。这是导出期最重要的优化之一。

### 用例 ②：`all_slice_all_gather` —— 配对

```mlir
func.func @all_slice_all_gather(%arg0 : tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
```

```mlir
  // CHECK: %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<mesh<["x"=2, "y"=2, "z"=2]>, [{"y"}, {"x"}]> : tensor<16x2xf32>
```

**读法**：`all_slice` 把 `"x"` 加到第 1 维（从无到有）。
它与 `all_gather` 是**互逆**的一对（L1-06 讲过八个集合通信算子）。

### 其余用例

| 用例 | 看点 |
|---|---|
| `reshard_of_reshard` | 连续 reshard 的合并（L1-10 讲过规范化） |
| `all_to_all_fusion` | `all_to_all` 的融合 |
| `dot_general_with_unreduced_result`（+2 变体） | 未归约结果的三种延迟程度 |
| `reduce_unreduced_to_sharded_max` / `_min` | 未归约 → 已分片，且归约算子为 `max`/`min` |
| `main` | 综合 |

---

## 三、两条分支的对照

| | 默认（文件 1） | 显式 collective（文件 2） |
|---|---|---|
| RUN 选项 | 无 | `enable-insert-explicit-collectives=true` 等 |
| 输出里的通信 | 少（多为 reshard / 分片更新） | **显式的 `sdy.reduce_scatter` / `all_slice` / `all_gather`** |
| 关注点 | 分片对齐、自由轴处理 | **融合**：reduce + reshard → reduce_scatter |

**一句话**：默认分支先把 IR"整理干净"，显式分支再把 reshard **翻译并融合**成具体的集合通信。
