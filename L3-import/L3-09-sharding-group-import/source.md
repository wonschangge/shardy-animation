<!-- sdy-coverage
transforms/import/test/sharding_group_import.mlir
transforms/import/test/sharding_group_constraints.mlir
-->

# L3-09 · sharding-group-import — 源 IR

本课覆盖两个文件：

| 文件 | 行数 | 用例数 | 作用 |
|---|---|---|---|
| `transforms/import/test/sharding_group_import.mlir` | 74 | 6 | 组 id 的**合并**与**重编号** |
| `transforms/import/test/sharding_group_constraints.mlir` | 185 | 8 | 组的**校验约束** |

```mlir
// RUN: sdy_opt -split-input-file %s -sdy-sharding-group-import | FileCheck %s
```

**这是 L2-07 反复提到的"组 id 传递闭包合并与规范化"的完整规则**
（L3-01 见过"4 条 → 2 条、3 个 id → 1 个"的例子）。

> **注意 RUN 行里的 `-split-input-file`**：它让每个 `// -----` 段**独立**处理。
> 组 id 的计数器是**按段重置**的 —— 漏掉这个选项会看到完全不同的编号。

---

## 一、动作 ①：合并重叠的组

### 不重叠 → 保持不变

```mlir
func.func @sharding_groups_no_overlap(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
```

```mlir
  sdy.sharding_group %arg0 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 1 : tensor<4xf32>
  func.return
}
```

```mlir
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
```

两个张量分属不同的组 → 没有可合并的。

### 全部重叠 → 合并成一个

```mlir
func.func @sharding_groups_all_overlap(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
```

```mlir
  sdy.sharding_group %arg0 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  func.return
}
```

```mlir
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=0 : tensor<4xf32>
```

**读法**：`%arg0` 同时在组 0、1；`%arg1` 同时在组 0、2。
通过**共享的组 0**，两者被关联起来 → **传递闭包**把 0/1/2 合成一个组。

### 通过中间组间接关联

```mlir
func.func @sharding_groups_overlap_min_id_used(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
```

```mlir
  sdy.sharding_group %arg0 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  func.return
}
```

```mlir
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=0 : tensor<4xf32>
```

**读法**：两者通过**组 2** 关联 → 合并。
合并后采用**最小的 id**（0）作为新组的编号 —— 用例名 `overlap_min_id_used` 说的就是这个。

### 部分重叠 → 各自合并

```mlir
func.func @sharding_groups_mixed_overlaps(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
```

```mlir
  sdy.sharding_group %arg0 group_id = 0 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 1 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 3 : tensor<4xf32>
  func.return
}
```

```mlir
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
```

**读法**：`%arg0` 的 {0,1} 与 `%arg1` 的 {2,3} **没有交集** → 合并成**两个**组，
重编号为 0 与 1。

---

## 二、动作 ②：重编号为 `0..N-1`

```mlir
func.func @sharding_groups_reindexes_ids(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
```

```mlir
  sdy.sharding_group %arg0 group_id = 12 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 89 : tensor<4xf32>
  func.return
}
```

```mlir
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
```

**读法**：12、89 这样的任意数字被重编号为 **0、1**。

**为什么必须重编号**：
- 组 id 是用户随手写的（可以是任意 `i64`），但内部处理希望是**紧凑**的 `0..N-1`。
- 紧凑编号让后续的"按组遍历"可以用数组而不是哈希表。

### 排序依据

```mlir
func.func @sharding_groups_reindex_ordering_matches_min_element_ordering(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) {
```

```mlir
  sdy.sharding_group %arg0 group_id = 567 : tensor<4xf32>
  sdy.sharding_group %arg0 group_id = 23 : tensor<4xf32>
  sdy.sharding_group %arg1 group_id = 2 : tensor<4xf32>
  sdy.sharding_group %arg2 group_id = 123456 : tensor<4xf32>
  func.return
}
```

```mlir
  // CHECK: sdy.sharding_group %arg0 group_id=0 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg1 group_id=1 : tensor<4xf32>
  // CHECK: sdy.sharding_group %arg2 group_id=2 : tensor<4xf32>
```

**读法**：三个组分别含 id {567, 23}、{2}、{123456}。
- 每个组的"代表 id"取该组的**最小 id**：`%arg0`→23、`%arg1`→2、`%arg2`→123456。
- 用例名点明了排序依据：**与最小元素顺序一致**（`reindex_ordering_matches_min_element_ordering`）。

> **实测确认**：按 RUN 行加 `-split-input-file` 跑一遍，输出与上面三条 CHECK 完全一致。
> 若**漏掉** `-split-input-file`，所有函数会在同一个模块里处理、共用一个计数器，
> 编号会变成 3、0、4 之类 —— 与 CHECK 不符。**务必按 RUN 行原样复现。**

---

## 三、组的校验约束

`sharding_group_constraints.mlir` 覆盖组的合法性检查（8 个用例，6 条报错）：

### 约束 ①：组的值**不能跨越 `manual_computation` 边界**

```mlir
  // expected-error@below {{ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: 90210}}
  sdy.sharding_group %0 group_id = 90210 : tensor<8x8xf32>
```

```mlir
    // expected-error@below {{ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: 44094}}
    sdy.sharding_group %3 group_id = 44094 : tensor<8x8xf32>
```

```mlir
    // expected-error@below {{ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: 4311}}
    sdy.sharding_group %1 group_id = 4311 : tensor<8x8xf32>
```

```mlir
  // expected-error@below {{ShardingGroupOps values cannot cross ManualComputationOp boundaries for groupId: 7331}}
  sdy.sharding_group %0 group_id = 7331 : tensor<8x8xf32>
```

**读法**：同组的值若一个在 `manual_computation` **内部**、一个在**外部** → 报错。

**为什么**：手动计算内部是"局部世界"（L1-07），内外的值在分片语义上不可比 ——
强制同分片无从谈起。这与 L2-09 讲的"manual 轴冻结"是同一个道理。

**四个用例**覆盖了不同的跨越方向（外→内、内→外、跨多层等）。

### 约束 ②：同组的值**形状必须相同**

```mlir
  sdy.sharding_group %arg0 group_id = 23 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 23 : tensor<8x8xf32>
  // expected-error@below {{ShardingGroupOps values must have the same shape for groupId: 23}}
  sdy.sharding_group %1 group_id = 23 : tensor<8x8x1xf32>
```

**读法**：`tensor<8x8xf32>` 与 `tensor<8x8x1xf32>` 形状不同 → 报错。

**为什么**：组的语义是"强制同分片"。形状不同则分片**根本无法比较** ——
rank 不同，维度分片的项数都不同。

### 合法的情形（2 个用例）

文件里还有 2 个**没有** `expected-error` 的用例，验证合法写法不被误报：
- 组的值全在 `manual_computation` **内部**（不跨界）
- 组的值全在**外部**

---

## 四、与其它课的关系

| 课 | 讲什么 | 关系 |
|---|---|---|
| L1-09 | `sdy.sharding_group` 的**语法** | 本课讲它的**导入期处理** |
| L2-07 | 组在**传播中**的行为 | 那里说"组 id 先规范化再检查" —— 本课就是那个规范化 |
| L3-01 | 见过"4 条 → 2 条"的例子 | 本课给出完整规则 |
| **本课** | 合并 + 重编号 + 校验 | |

**完整的生命周期**：

```
用户写任意 group_id
  → 【本课】传递闭包合并 + 重编号为 0..N-1 + 校验
  → 【L2-07】传播中作为独立通道生效
```
