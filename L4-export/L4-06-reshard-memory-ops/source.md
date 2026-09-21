<!-- sdy-coverage
transforms/export/test/insert_explicit_reshards/gather_scatter.mlir
transforms/export/test/insert_explicit_reshards/custom_call.mlir
transforms/export/test/insert_explicit_reshards/collective_ops.mlir
-->

# L4-06 · reshard-memory-ops — 源 IR

**L4 按算子族展开的第四课**：访存、自定义调用、集合通信。覆盖 3 个文件：

| 文件 | 行数 | 用例数 |
|---|---|---|
| `insert_explicit_reshards/gather_scatter.mlir` | 288 | 9 |
| `insert_explicit_reshards/custom_call.mlir` | 318 | 24 |
| `insert_explicit_reshards/collective_ops.mlir` | 107 | 6 |

网格：`sdy.mesh @mesh = <["x"=4, "y"=2]>`、`@mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>`

---

## 一、★ `gather_scatter`：与 `dot` **同构**的处理

### "最有表现力"的那个用例

```mlir
// CHECK-LABEL: @gather
// COM: the most expressive example
func.func @gather(
  %arg0: tensor<2x6x4x26x22xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x":(1)2}, {"x":(2)2}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]>},
  %arg1: tensor<2x22x12x26x14xi64> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x":(1)2}, {"z":(1)2}, {"z":(2)2}, {"y":(2)2}, {"t"}]>}
) -> (tensor<1x6x22x12x26x14xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x":(1)2}, {"x":(2)2}, {"z":(1)2}, {"z":(2)2}, {"y":(2)2}, {"t"}]>}) {
```

```mlir
  // COM: sharding_rule<([i, k, p, n, l], [q, l, m, n, o])->([j, k, l, m, n, o]) {i=2, j=1, k=6, l=22, m=12, n=26, o=14, p=4, q=2} reduction={i, p} need_replication={j, q}>
```

```mlir
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"x":(2)2}, {"y":(1)2}, {"y":(2)2}, {"z":(1)2}]> : tensor
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{}, {"z":(1)2}, {"z":(2)2}, {"y":(2)2}, {"t"}]> : tensor
  // CHECK-NEXT: %[[GATHER:.*]] = "stablehlo.gather"(%[[RESHARD0]], %[[RESHARD1]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"x":(2)2}, {"z":(1)2}, {"z":(2)2}, {"y":(2)2}, {"t"}], unreduced={"y":(1)2}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y":(1)2} %[[GATHER]] out_sharding=<@mesh_xyzt, [{}, {"x":(2)2}, {"z":(1)2}, {"z":(2)2}, {"y":(2)2}, {"t"}]> : tensor
  // CHECK-NEXT: %[[RESHARD_RET:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_xyzt, [{"x":(1)2}, {"x":(2)2}, {"z":(1)2}, {"z":(2)2}, {"y":(2)2}, {"t"}]> : tensor
  // CHECK-NEXT: return %[[RESHARD_RET]] : tensor
```

**读法**（四步，与 L4-04 的 `dot` 完全同构）：
1. **两个操作数各 reshard**（把冲突的轴去掉/对齐）
2. `gather` 执行，结果标 `unreduced={"y":(1)2}` —— **部分结果**
3. `all_reduce {"y":(1)2}` —— 合并部分结果
4. **再 reshard** 到目标分片

**为什么与 `dot` 同构**：两者的规则里都有 `reduction` 因子
（`gather` 是 `reduction={i, p}`）。沿归约因子切 → 部分结果 → 需要 `all_reduce`。

**注意子轴**：这个用例里几乎每个轴都是**子轴**（`{"x":(1)2}`、`{"y":(2)2}`…）——
因为 `@mesh_xyzt` 有 4 个轴共 4×4×4×8 台设备，要用子轴精确切分。

### 与 L3-11 的呼应

L3-11 讲过 `gather` 的**批维显式化**（把 `iota` 从 `start_index_map` 移到
`operand_batching_dims`）。**为什么必须做**在那里已经回答：
隐式批维落在 `blocked_propagation` 因子上，传播无法处理。

这里看到**显式批维之后**的样子 —— 本用例的 gather 已经带了
`operand_batching_dims`，所以它的分片能被正常处理（插 reshard + all_reduce）。

**两课合起来才是 gather 的完整故事**：L3-11 让分片**可表达**，L4-06 让分片**可落地**。

---

## 二、`custom_call`：内置注册 vs 用户自定义

24 个用例，与 L2-10 的 `custom_call_*` **同名** ——
那里验证**规则生成**，这里验证**reshard 插入**。

### 内置注册：按注册表的规则处理

```mlir
func.func @custom_call_compact_wy_helper(%arg0: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
```

```mlir
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
```

```mlir
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @CompactWyHelper(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<128x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<128x128xf32>
```

**读法**（测试里的 NOTE 给出了规则）：
- 规则是**逐元素式**的 `([i, j])->([i, j])` —— 输入输出维度一一对应。
- 输入 `[{"x"}, {"y"}]` 与规则**兼容** → `custom_call` 保持输入的分片。
- 结果要求 `[{"y"}, {"x"}]`（**轴序相反**）→ **之后**插 reshard。

**24 个用例覆盖的 22 个内置算子**（与 L2-10 一致）：

| 类别 | 算子 |
|---|---|
| 线性代数 | `CompactWyHelper`、`Eigh`、`Qr`、`QrDecompositionBlock`、`HouseholderProduct` |
| X64 系列 | `X64Combine`、`X64SplitHigh`、`X64SplitLow` |
| TopK 系列 | `TopK`（1d/2d）、`Top2`、`ApproxTopK` |
| 设备迁移 | `MoveToDevice`、`MoveToHost` |
| 其它 | `InspectSharding`、`LayoutConstraint`、`Erf`、`PartialReduce`、`XlaMegascaleProvideMetadata` |

### 未注册的 `custom_call`：按**用户规则**

```mlir
// CHECK-LABEL: func @unregistered_custom_call_with_existing_rule
func.func @unregistered_custom_call_with_existing_rule(%arg0: tensor<4x2xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<2x4xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}){
```

```mlir
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>]>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
```

**读法**（本课最重要的对比）：
- `@foo` **未注册**，但用户写了 `sdy.sharding_rule`（注意末尾的 `, custom`）。
- 规则是**转置**语义 `([i, j])->([j, i])`。
- 输入 `[{"x"}, {"y"}]`、结果 `[{"x":(1)2}, {"y"}]` —— 按转置规则**完全兼容**
  → **`CHECK-NOT: sdy.reshard`**（一条都不插）！

**这证明了 `custom` 规则是"一等公民"** —— 用户为未注册算子写的规则
与内置注册的规则**受到同等对待**（都会参与兼容性判定与 reshard 插入）。

另一个用例 `unregistered_custom_call_without_existing_rule` 则验证
**没有规则**时的行为（保守处理）。

---

## 三、★ `collective_ops`：集合通信算子**自身**的分片规则

6 个用例：`all_gather` / `all_reduce` / `all_to_all_same_dimension` /
`collective_broadcast` / `collective_permute` / `reduce_scatter`。

### `all_gather`：两个操作数必须同分片

```mlir
func.func @all_gather(%arg0: tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg1: tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
```

```mlir
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{}, {"y"}]>]>
  } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
```
```

```mlir
  // #sdy.op_sharding_rule<([i, j], [i, j])->([i, k], [i, k]) {i=2, j=2, k=4} need_replication={j, k}>
  // CHECK-NEXT: %0 = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<2x2xi64>
  // CHECK-NEXT: %1:2 = "stablehlo.all_gather"(%arg0, %0)
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>}
```

```mlir
  // CHECK-NEXT: %2 = sdy.reshard %1#1 <@mesh, [{}, {"y"}]> : tensor<2x4xi64>
  // CHECK-NEXT: return %1#0, %2
```

**读法**（三步）：
1. **`all_gather` 的两个操作数必须同分片**：`%arg0` 是 `[{"y"}, {}]`、
   `%arg1` 是 `[{}, {"y"}]` → 把 `%arg1` **reshard 到 `[{"y"}, {}]`**。
2. 执行 `all_gather`，两个结果都是 `[{"y"}, {}]`。
3. `%1#1` 要求 `[{}, {"y"}]` → **再 reshard** 回去。

**规则解读**（测试注释给出）：
```
#sdy.op_sharding_rule<([i, j], [i, j])->([i, k], [i, k]) {i=2, j=2, k=4} need_replication={j, k}>
```
- **`need_replication={j, k}`** —— 这两个因子**必须复制**！
- `j` 是输入的非聚合维、`k` 是输出新增的聚合维。
- **为什么**：`all_gather` 在 `all_gather_dim` 上把各设备的数据**拼起来** ——
  这个操作要求每台设备都能看到**完整**的输入（否则拼不出完整结果）。

**这就是"集合通信算子也有分片规则"的含义** —— 它们不是"分片的消费者"，
而是**有自己的约束**的普通算子。

### 6 个用例速览

| 用例 | 规则要点 |
|---|---|
| `all_gather` | `need_replication={j, k}`；两操作数同分片 |
| `all_reduce` | `([i, j], [i, j])->([i, j], [i, j])` —— 逐元素式，无 need_replication |
| `all_to_all_same_dimension` | 同一维上的 all_to_all |
| `collective_broadcast` | 广播 |
| `collective_permute` | 置换 |
| `reduce_scatter` | 归约 + 切分（L4-01 见过融合后的形态） |

**注意 `all_reduce` 的规则没有 `need_replication`** ——
因为它的输出形状与输入**完全相同**，不需要拼接。
而 `all_gather` 的输出在聚合维上**变大了**（`j=2` → `k=4`），所以需要复制。

---

## 四、3 个文件 / 39 个用例的族谱

| 族 | 文件 | 用例数 | 核心规则 |
|---|---|---|---|
| **访存** | `gather_scatter` | 9 | 与 `dot` 同构（`reduction` 因子 → `unreduced` + `all_reduce`） |
| **自定义调用** | `custom_call` | 24 | 内置注册 / 用户 `custom` 规则**同等对待** |
| **集合通信** | `collective_ops` | 6 | **自身有分片规则**；`need_replication` 决定要不要复制 |

**一句话总结**：
> **这三类算子的共同点**：它们都**不是透明的** ——
> 都有自己的分片约束，都需要显式的 reshard 来对齐。
