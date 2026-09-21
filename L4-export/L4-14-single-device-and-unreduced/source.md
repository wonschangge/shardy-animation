<!-- sdy-coverage
transforms/export/test/resolve_single_device_sharding.mlir
transforms/export/test/verify_unreduced_axes.mlir
transforms/export/test/remove_ag_rs_for_cmv1.mlir
-->

# L4-14 · single-device-and-unreduced — 源 IR

导出期的三个"收尾"pass，各处理一类特殊分片。

| 文件 | 行数 | 用例数 | pass |
|---|---|---|---|
| `transforms/export/test/resolve_single_device_sharding.mlir` | 186 | 5 | `-sdy-resolve-single-device-sharding` |
| `transforms/export/test/verify_unreduced_axes.mlir` | 302 | 0（全是 `expected-error`） | `-sdy-verify-unreduced-axes` |
| `transforms/export/test/remove_ag_rs_for_cmv1.mlir` | 130 | 10 | `-sdy-remove-all-gather-reduce-scatter-for-cmv1` |

网格：

```mlir
sdy.mesh @mesh = <["x"=2]>
```

```mlir
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
```

---

## 一、★ 单设备分片 → `stablehlo.if` 守卫

**问题**：单设备网格 `@single_dev_0 = <[], device_ids=[0]>` 表示
"**这个算子只在设备 0 上执行**"。但整个程序要在**所有设备**上运行 ——
其他设备怎么办？

**解法**：用 `stablehlo.if` 按**设备号**守卫。

```mlir
func.func @custom_call_single_device_0(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<4x32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
```

```mlir
  // CHECK-NEXT: %[[IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[PART_ID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[PART_ID_I64:.*]] = stablehlo.convert %[[PART_ID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT: %[[IS_DEV0:.*]] = stablehlo.compare EQ, %[[PART_ID_I64]], %[[C0]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT: %[[IF_RES:.*]]:3 = "stablehlo.if"(%[[IS_DEV0]]) ({
  // CHECK-NEXT:   %[[EXEC:.*]]:3 = stablehlo.custom_call @SomeCustomCall(%[[IN_REPL]]) : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token)
  // CHECK-NEXT:   stablehlo.return %[[EXEC]]#0, %[[EXEC]]#1, %[[EXEC]]#2 : tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %[[ZEROS0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x16xf32>
  // CHECK-NEXT:   %[[ZEROS1:.*]] = stablehlo.constant dense<0> : tensor<4x32xi32>
  // CHECK-NEXT:   %[[TOKEN:.*]] = stablehlo.create_token
  // CHECK-NEXT:   stablehlo.return %[[ZEROS0]], %[[ZEROS1]], %[[TOKEN]] : tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token
```

### 逐项读（五步）

| 步 | 操作 | 作用 |
|---|---|---|
| ① | `sdy.reshard %[[ARG0]] <@mesh, [{}, {}]>` | 输入 reshard 成**全复制** |
| ② | `stablehlo.partition_id` | 取**当前设备号** |
| ③ | `stablehlo.convert` + `constant 0` | 类型对齐、准备比较值 |
| ④ | `stablehlo.compare EQ` | 判断"**我是不是设备 0**" |
| ⑤ | `stablehlo.if` | **是** → 执行 custom_call；**否** → 返回**零值** |

**为什么先全复制**：所有设备都要执行这条 `if`，所以它们都需要**输入数据** ——
即使不参与计算，也要能"走个过场"。

**为什么 else 分支返回零值**：非设备 0 的设备**不参与计算**，
但 `if` 的结果类型必须一致 → 用零值占位。

**注意 token 类型**：第三个结果 `!stablehlo.token` 在 else 分支用
`stablehlo.create_token` 生成 —— token 是**副作用标记**，
即使不执行也要造一个。

> **核心思想**：**单设备分片 = "只有一台设备算，其他设备拿到占位值"**。

---

## 二、★ `verify_unreduced_axes`：未归约轴的**校验**

这个文件用 `-verify-diagnostics`，**全部是 `expected-error`** —— 没有 CHECK-LABEL。

```mlir
// RUN: sdy_opt %s -sdy-verify-unreduced-axes -split-input-file -verify-diagnostics
```

### 规则 ①：不能**悄悄丢掉**未归约轴

```mlir
func.func @dropped_by_add(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
```

```mlir
  // expected-error@+1 {{'stablehlo.add' op dropped unreduced axis 'y' without a blessed operation (e.g., sdy.reshard). This is an invalid transition from unreduced to reduced.}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

**读法**：
- 输入带 `unreduced={"y"}`，但 `add` 的输出**没有** `unreduced` ——
  即"未归约轴 `y` 被丢掉了"。
- 错误信息里的关键词：**`without a blessed operation (e.g., sdy.reshard)`**
  —— **"blessed operation"（受祝福的操作）**。
- 含义：**从"未归约"变成"已归约"必须由特定的算子显式声明**，
  普通算子不能"顺手"完成这个转换。

### 规则 ②：`blessed` 的正确做法

```mlir
func.func @dropped_by_add_blessed(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
```

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
```

**读法**：先插一条 **`sdy.sharding_constraint`**（不带 `unreduced`）——
**显式声明**"这里要把 `y` 归约掉"。之后 `add` 就没问题了。

**`sdy.sharding_constraint` 与 `sdy.reshard` 都是 blessed operation** ——
它们的作用就是"**显式声明分片状态的变化**"。

### 规则 ③：函数调用两端的未归约轴必须**匹配**

```mlir
func.func @call_argument_drops_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
```

```mlir
  // expected-error@+1 {{'func.call' op has unreduced axes mismatch for 'y' at call argument 0.}}
```

**4 个用例**覆盖调用的各种不匹配：

| 用例 | 场景 |
|---|---|
| `call_argument_drops_unreduced` | 实参**丢掉**未归约轴 |
| `call_result_drops_unreduced` | 调用结果**丢掉** |
| `call_result_extra_unreduced` | 结果**多出**未归约轴 |
| `call_operand_missing_unreduced` | 实参**缺少**未归约轴 |

**错误信息统一为**：`'func.call' op has unreduced axes mismatch for 'y' at call argument/result N.`

**为什么必须匹配**：调用点与函数体是**同一个值的两种视角**（L3-05 / L3-10）。
未归约状态不一致意味着"调用者以为已归约、被调者以为未归约" —— 语义错乱。

### 规则 ④：`func.return` 与 `manual_computation` 同样受约束

```mlir
  // expected-error@+1 {{'func.return' op has unreduced axes mismatch for 'y' at return value 0 without a blessed operation (e.g., sdy.reshard). This is an invalid transition from unreduced to reduced.}}
```

```mlir
  // expected-error@+1 {{'sdy.manual_computation' op dropped unreduced axis 'y' without a blessed operation (e.g., sdy.reshard). This is an invalid transition from unreduced to reduced.}}
```

**读法**：`func.return` 与 `sdy.manual_computation` **也**不能悄悄丢掉未归约轴 ——
规则是**统一**的。

### 规则 ⑤：归约算子（`sum` / `max` / `min`）**不能变**

```mlir
func.func @reshard_mismatches_unreduced_kind(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced=max{"x", "y"}>}) -> tensor<8x8xf32> {
```

```mlir
  // expected-error@+1 {{'sdy.reshard' op cannot change the reduction operator of kept unreduced axes from max to sum.}}
```

```mlir
  // expected-error@+1 {{cannot introduce 'max' unreduced axes. Expected 'sum'.}}
```

**读法**（两类错误）：
- **保留**的未归约轴，其归约算子**不能改**（`max` → `sum` 报错）。
- **引入**新的未归约轴时，种类必须正确（`dot` 只能引入 `sum`，
  引入 `max` 报错 —— 因为 `dot` 的语义是求和）。

**为什么重要**：归约算子是**语义的一部分** ——
`unreduced=max{"y"}` 与 `unreduced=sum{"y"}` 表示**完全不同的部分结果**。
改了就改变了程序含义。

### ★ 这回答了 TODOLIST 的验收点

> **能说出 `unreduced` 轴从哪来、必须在哪被消除**

- **从哪来**：`dot` / `reduce` 等算子沿**归约维切分**时**引入**
  （L4-04 的 `dot`、L4-05 的 `reduce` —— 都是 `unreduced` + `all_reduce`）。
- **必须在哪被消除**：由 **blessed operation**（`sdy.reshard` / `sdy.sharding_constraint`）
  **显式**消除。普通算子（`add`、`call`、`return`、`manual_computation`）
  都**不能**悄悄完成这个转换。

---

## 三、`remove_ag_rs_for_cmv1`：为 CMV1 移除通信

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-remove-all-gather-reduce-scatter-for-cmv1 | FileCheck %s
```

**CMV1** 是一个后端 —— 它能**自己处理** `all_gather` / `reduce_scatter`，
所以导出时要把这些算子**移除**。

### `single_all_gather`：独立的 all_gather **保留**

```mlir
func.func @single_all_gather(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = sdy.all_gather [{}, {"y"}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
```

```mlir
  // CHECK: %0 = sdy.all_gather [{}, {"y"}] %arg0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %0
```

**读法**：`all_gather` 是**最终结果**（函数返回它）→ **不能移除**
（否则函数结果就不对了）。

### `all_gather_dot`：可融合的 all_gather **被移除**

```mlir
func.func @all_gather_dot(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = sdy.all_gather [{"x"}, {}] %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<16x32xf32>
  %1 = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}
```

```mlir
  // CHECK: %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: return %0
```

**读法**（**关键对比**）：
- 输入：`all_gather` → `dot`
- 输出：**只剩 `dot`**！`all_gather` 被移除，`dot` 直接读 `%arg1`。

**为什么可以移除**：`all_gather` 的**唯一使用者**是 `dot`，
而 CMV1 能在 `dot` 内部**自己完成**这个聚合 —— 所以显式的 `all_gather` 是冗余的。

**这与 L4-10 的优化同类**：都是"**消除冗余的通信算子**"。
区别：
- **L4-10**：消除 `all_to_all` 前的 `collective_permute`（通用优化）
- **L4-14**：消除 CMV1 能自己处理的 `all_gather`/`reduce_scatter`（**后端特定**）

---

## 四、三个 pass 的共同点

| pass | 处理对象 | 时机 |
|---|---|---|
| `resolve_single_device_sharding` | 单设备分片 | 导出**收尾**（加守卫） |
| `verify_unreduced_axes` | 未归约轴 | 导出**校验**（报错） |
| `remove_ag_rs_for_cmv1` | 通信算子 | 后端**适配**（移除） |

**共同点**：都是导出期的**边界情形处理** ——
前面几课（L4-03～13）处理"常规算子"，本课处理三类**特殊情况**：

1. **只在一台设备上执行**（单设备分片）
2. **部分结果**（未归约轴）
3. **后端能自己做的事**（CMV1 的 all_gather/reduce_scatter）

**一句话总结**：
> 导出期的最后阶段要把所有"**特殊状态**"处理干净 ——
> 单设备分片加守卫、未归约轴校验、后端特定优化。
