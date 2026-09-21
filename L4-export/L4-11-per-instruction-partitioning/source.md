<!-- sdy-coverage
transforms/export/test/per_instruction_partitioning.mlir
transforms/export/test/per_instruction_partitioning_range.mlir
transforms/export/test/per_instruction_partitioning_subroutine.mlir
-->

# L4-11 · per-instruction-partitioning — 源 IR

**只对指定的指令跑分区器**，并把结果包进 `sdy.manual_computation`。

| 文件 | 行数 | 用例数 | filter 语法 |
|---|---|---|---|
| `transforms/export/test/per_instruction_partitioning.mlir` | **578** | **23** | 算子名子串 |
| `transforms/export/test/per_instruction_partitioning_range.mlir` | 49 | 2 | `selectLow` / `selectHigh` |
| `transforms/export/test/per_instruction_partitioning_subroutine.mlir` | 38 | 2 | `func=<名字>` |

---

## 一、★ 核心机制

### RUN 行：`filter` 参数

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter=dot,constant,reshard,all_gather,all_slice,concatenate,convolution,while,call,if" | FileCheck %s
```

**filter 是一个算子名列表**（逗号分隔）。**只有匹配的指令**会被分区。

### 效果：选中的指令被**包进 `manual_computation`**

```mlir
func.func @selective_dot(%lhs: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
                         %rhs: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
```

```mlir
  %dot = stablehlo.dot %lhs, %rhs {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
```

```mlir
  %add = stablehlo.add %dot, %dot {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x16xf32>
```

```mlir
  return %add : tensor<8x16xf32>
}
```

```mlir
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[LHS]], %[[RHS]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {"y"}]>]
  // CHECK-SAME:   manual_axes={"x", "y"} (%arg2: tensor<4x32xf32>, %arg3: tensor<32x8xf32>) {
  // CHECK-NEXT:   %[[LOCAL_DOT:.*]] = stablehlo.dot %arg2, %arg3 : (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:   sdy.return %[[LOCAL_DOT]] : tensor<4x8xf32>
  // CHECK-NEXT: } : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
```

```mlir
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[MANUAL]], %[[MANUAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x16xf32>
```

**读法**（本课最关键的一处对比）：
- `stablehlo.dot` **在 filter 里** → 被**包进 `sdy.manual_computation`**。
- `stablehlo.add` **不在 filter 里** → **保持原样**，直接读 `%[[MANUAL]]`。

**`manual_computation` 的内容**：
- `in_shardings` / `out_shardings` 用的是**全局分片**（与 L1-07 一致）。
- 区域内的 `%arg2: tensor<4x32xf32>` —— **局部形状**！
  全局 `8x32` 沿 `x=2` 切 → 局部 `4x32` ✓
- 区域内的 `dot` **不带任何分片属性** —— 因为它已经是"本地算子"。
- `manual_axes={"x", "y"}` —— 两个轴都冻结（区域内不再分片）。

**为什么用 `manual_computation` 承载**：
与 L4-09 的 HALO 模式同样的道理 ——
`manual_computation` 正是"区域内自己管分片"的机制（L1-07），
天然适合表达"这个算子我已经手动分好了"。

### 不可整除的情形

```mlir
func.func @selective_indivisible_dot(%lhs: tensor<6x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
                                     %rhs: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<5x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
```

```mlir
  %sliced_lhs = stablehlo.slice %lhs [0:5, 0:32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x32xf32>) -> tensor<5x32xf32>
```

```mlir
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[LHS]] [0:5, 0:32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<6x32xf32>) -> tensor<5x32xf32>
```

**读法**：`6` 沿 `x=2` 切 → 每台 3；但输出要 `5`（不可整除）。
→ 先 `slice` 到 `5x32`（`slice` **不在 filter 里**，保持原样），再分区 `dot`。

**这说明 filter 是"选择性"的** —— 未选中的算子完全不受影响。

---

## 二、★ 三种 `filter` 语法

### 语法 ①：算子名子串（主文件 23 个用例）

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter=dot,constant,reshard,all_gather,all_slice,concatenate,convolution,while,call,if" | FileCheck %s
```

**匹配规则**：算子名**子串**匹配。列表里写 `dot` 会匹配 `stablehlo.dot`。

**23 个用例覆盖的算子**（从 filter 串可见）：
`dot` / `constant` / `reshard` / `all_gather` / `all_slice` / `concatenate` /
`convolution` / `while` / `call` / `if` —— 以及主文件里更多的算子。

**注意 `while` / `call` / `if`**：这三个是**区域算子** ——
说明 filter 也能选中带区域的算子，把它们整体包进 `manual_computation`。

### 语法 ②：位置范围（`range` 文件 2 个用例）

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter='selectLow=0, selectHigh=0'" | FileCheck %s
// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter='selectHigh=0, selectLow=0'" | FileCheck %s
```

**两个 RUN 行顺序相反**（`selectLow` 在前 / `selectHigh` 在前）——
**验证参数的顺序不影响结果**。

**`selectLow=N` / `selectHigh=N`**：按**指令位置**选择，选出一个**区间**。
`selectLow=0, selectHigh=0` 表示"只选第 0 条指令"。

**两个用例**：
- `selective_dot_range`：第 0 条是 `dot` → `dot` 被包，`add` 不被包。
- `selective_add_range`：第 0 条是 `add` → **第一条 `add` 被包**，第二条不被包。

```mlir
func.func @selective_add_range(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %add1 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}: tensor<4x8xf32>
```

```mlir
  %add2 = stablehlo.add %add1, %add1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}: tensor<4x8xf32>
```

```mlir
  return %add2 : tensor<4x8xf32>
}
```

```mlir
  // CHECK:      %[[ADD1:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG0]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%arg1: tensor<2x8xf32>, %arg2: tensor<2x8xf32>) {
  // CHECK-NEXT:   %[[LOCAL1:.*]] = stablehlo.add %arg1, %arg2 : tensor<2x8xf32>
  // CHECK-NEXT:   sdy.return %[[LOCAL1]] : tensor<2x8xf32>
  // CHECK-NEXT: } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
```

```mlir
  // CHECK: %[[ADD2:.*]] = stablehlo.add %[[ADD1]], %[[ADD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
```

**读法**（**同一个函数里两个 `add`，只有第一个被包**）：
- `%add1` → 被包进 `manual_computation`
- `%add2` → **保持原样**

**这证明 filter 是【按指令位置】而非"按算子名"选择的** ——
与语法 ① 形成对比。

### 语法 ③：限制在指定函数内（`subroutine` 文件 2 个用例）

```mlir
// RUN: sdy_opt %s -split-input-file -sdy-per-instruction-partitioning="filter=func=subroutine,add" | FileCheck %s
```

**`func=<名字>`**：把选择范围**限制在指定函数内**。

**关键对比**：`@subroutine` 与 `@main` 里**各有一个 `add`**：

```mlir
func.func private @subroutine(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
```

```mlir
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
```

```mlir
  // CHECK:      %[[MANUAL_ADD:.*]] = sdy.manual_computation(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   manual_axes={"x"} (%[[LOCAL_ARG0:.*]]: tensor<4x16xf32>, %[[LOCAL_ARG1:.*]]: tensor<4x16xf32>) {
  // CHECK-NEXT:   %[[LOCAL_ADD:.*]] = stablehlo.add %[[LOCAL_ARG0]], %[[LOCAL_ARG1]] : tensor<4x16xf32>
  // CHECK-NEXT:   sdy.return %[[LOCAL_ADD]] : tensor<4x16xf32>
  // CHECK-NEXT: } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[MANUAL_ADD]] : tensor<8x16xf32>
```

```mlir
  // The add in @main is not partitioned because filter restricts to func=subroutine.
  // CHECK-NOT: sdy.manual_computation
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
  // CHECK: %[[CALL:.*]] = call @subroutine(%[[ADD]], %arg1)
  // CHECK: return %[[CALL]] : tensor<8x16xf32>
```

**读法**（本课最精妙的对比）：
- `@subroutine` 里的 `add` → **被包**（因为 `func=subroutine`）
- `@main` 里的 `add` → **不被包**（`CHECK-NOT: sdy.manual_computation`）

**测试注释把这个设计意图写得很清楚**：
- `The add inside @subroutine is selected and wrapped in sdy.manual_computation.`
- `The add in @main is not partitioned because filter restricts to func=subroutine.`

**`func=` 的用途**：当同一个算子名在多个函数里出现时，**精确定位到某一个函数**。

---

## 三、为什么需要这个 pass：**bisect**

TODOLIST 说这个 pass "**用于 bisect**"。含义：

**场景**：整个导出流水线在某段 IR 上失败（崩溃 / 报错 / 结果不对）。
但 IR 很长、算子很多，**不知道是哪个算子的问题**。

**用法**：用 `filter` **逐个/分组**只对怀疑的算子跑分区器，
看问题是否复现 —— 这就是**二分定位**（bisect）。

| filter 写法 | 用途 |
|---|---|
| `filter=dot` | 只测 `dot` |
| `filter=dot,pad` | 只测 `dot` 与 `pad` |
| `filter='selectLow=0, selectHigh=0'` | 只测第 0 条指令 |
| `filter=func=subroutine,add` | 只测 `@subroutine` 里的 `add` |

**验收点**（TODOLIST）："能写出只分区 dot 与 pad 的 filter 串" →
**`filter=dot,pad`**。

---

## 四、与其它课的呼应

### 与 L1-07（`manual_computation`）的呼应

本 pass 把选中的指令包进 `sdy.manual_computation` —— 这与 **L4-09 的 HALO 模式**
用了同一个机制。

| 课 | 谁包 `manual_computation` | 为什么 |
|---|---|---|
| L4-09 | permutation 因子的消解 | 表达 halo（每台设备多拿边界） |
| L4-11 | 指定的指令 | 表达"这个算子我已经手动分好了" |

**共同点**：`manual_computation` 是 Shardy 里表达"**区域内自己管分片**"的通用机制。

### 与 L4-09 的 `manual_axes` 呼应

两个 pass 生成的 `manual_computation` 都带 `manual_axes`：

```mlir
  // CHECK-SAME:   manual_axes={"x", "y"} (%arg2: tensor<4x32xf32>, %arg3: tensor<32x8xf32>) {
```

**`manual_axes` 列出被冻结的轴** —— 区域内这些轴不再分片，
所以 block argument 是**局部形状**（`4x32` 而非 `8x32`）。

---

## 五、3 个文件 / 27 个用例的族谱

| 文件 | 用例数 | filter 语法 | 关键点 |
|---|---|---|---|
| `per_instruction_partitioning` | 23 | 算子名子串 | 覆盖 10+ 种算子（含 `while`/`call`/`if` 区域算子） |
| `per_instruction_partitioning_range` | 2 | `selectLow` / `selectHigh` | **按指令位置**选；参数顺序无关 |
| `per_instruction_partitioning_subroutine` | 2 | `func=<名字>` | **限制在函数内**；同名算子在别处不受影响 |

**一句话总结**：
> **只对指定的指令跑分区器，结果包进 `sdy.manual_computation`。**
> 三种 filter 语法分别按**算子名**、**指令位置**、**所在函数**选择 ——
> 目的是**二分定位**导出流水线的问题。
