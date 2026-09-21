<!-- sdy-coverage
transforms/export/test/convert_global_to_local/stablehlo_dot.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_dot_general.mlir
-->

# L7-02 · parallelism-strategies — 源 IR

**综合课**：用**同一份矩阵乘程序**演示四种并行策略的**分片标注差异**与**通信代价**。

> **说明**：本课不引入新文件，而是**重新引用** L5-05 与 L6-04 的两个文件 ——
> 它们是下面所有例子的 IR 来源。

| 文件 | 来源课 | 提供什么 |
|---|---|---|
| `convert_global_to_local/stablehlo_dot.mlir` | **L5-05** | 四种分片情形的降级形态 |
| `executable_convert_global_to_local/stablehlo_dot_general.mlir` | **L6-04** | 收缩维分片的**数值验证** |

网格：

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ 四种策略的**分片标注**差异

**同一份程序**：`dot(A, B)`，其中 `A: 8x16`、`B: 16x32`、结果 `C: 8x32`。

| 策略 | 分片标注 | 通信 | 对应用例 |
|---|---|---|---|
| **数据并行**（DP） | 收缩维切 | **`all_reduce`** | `sharded_contracting_dim` |
| **张量并行**（TP） | 非收缩维切 | **无** | `sharded_non_contracting_dims` |
| **ZeRO** | 收缩维切 + **结果未归约** | **无**（延迟） | `..._unreduced_result` |
| **流水线并行**（PP） | 按**层**切（跨算子） | 只在**层边界** | （本课讨论，无单一用例） |

**★ 核心洞察**：
> **策略的差异本质上是「切哪个维」的差异** ——
> 而「切哪个维」直接决定了**需不需要通信**。

---

## 二、数据并行：切**收缩维** → `all_reduce`

```mlir
func.func @sharded_contracting_dim(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<8x32xf32> {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
```

```mlir
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
```

```mlir
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<8x32xf32>
```

```mlir
  return %1 : tensor<8x32xf32>
```

```mlir
}
```

**读法**：
- `%arg0` 的**第 1 维（收缩维）**切 `{"x"}`、`%arg1` 的**第 0 维（收缩维）**切 `{"x"}`。
- 每台设备算一个**部分和** → `unreduced={"x"}` → **`all_reduce`**。

**★ 为什么这是"数据并行"**：
> 收缩维对应**输入数据的方向**（`A` 的列、`B` 的行）——
> 切开它等于**把输入数据分给不同设备**，每台算一部分和，最后归约。

**通信代价**：**一次 `all_reduce`**，通信量 ∝ **结果大小**（`8x32`）。

**★ 在 L6-04 里这个策略被数值验证过**：

```mlir
func.func @parallel_dot(
  %arg0: tensor<2x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>},
  %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}
) -> (tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
```

```mlir
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2, [{}, {}], unreduced={"x"}>]>
  } : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>

  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x2xi32>
  return %1 : tensor<2x2xi32>
}
```

**读法**：与 L5-05 的用例**同构**，但这是**可执行测试** ——
它证明了"收缩维切 + `all_reduce`"的结果**与串行版一致**（L6-04 讲过逐设备推演）。

---

## 三、张量并行：切**非收缩维** → **无通信**

```mlir
func.func @sharded_non_contracting_dims(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>}
```

```mlir
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
```

```mlir
  return %0 : tensor<8x32xf32>
```

```mlir
}
```

**读法**：
- `%arg0` 的**第 0 维（非收缩维，`M`）**切 `{"x"}`、`%arg1` 的**第 1 维（非收缩维，`N`）**切 `{"y"}`。
- **收缩维（`K`）都是完整的 16** → 每台算输出矩阵的**一块** → **无通信** ✓

**★ 为什么这是"张量并行"**：
> 非收缩维对应**权重/输出的方向** ——
> 切开它等于**把权重矩阵分给不同设备**，每台算输出的一部分。

**通信代价**：**零**（在这个算子内）。但注意 ——
**如果后续算子需要完整的输入**，就要在**算子边界**插入通信。

**★ 与数据并行的对照**：

| | 数据并行 | 张量并行 |
|---|---|---|
| 切的维 | **收缩维** | **非收缩维** |
| 每台算什么 | **部分和** | 输出的**一块** |
| 通信 | **`all_reduce`** | **无** |
| 通信量 | ∝ 结果大小 | 0（本算子内） |

---

## 四、ZeRO：切收缩维但**结果未归约** → 延迟

```mlir
func.func @sharded_contracting_dim_unreduced_result(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}], unreduced={"x"}>}) {
```

```mlir
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
```

```mlir
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
```

```mlir
  return %0 : tensor<8x32xf32>
```

```mlir
}
```

**读法**（**与数据并行的唯一区别**）：
- **函数结果也声明了 `unreduced={"x"}`** → 归约责任交给**调用者**。
- → **不插 `all_reduce`**，直接 `return %0` ✓

**★ 为什么这对应 ZeRO 的思想**：
> ZeRO 的核心是**"不要每台设备都保存完整的状态"** ——
> 未归约时每台设备**只保存自己那份部分和**，显存占用**更小**。
> 归约推迟到**真正需要完整结果时**才做（**L4-05** 讲的"延迟归约"）。

**★ 与数据并行的对照**：

| | 数据并行 | ZeRO（延迟归约） |
|---|---|---|
| 收缩维 | 切 | 切 |
| 结果要求 | **完整** | **未归约** |
| `all_reduce` | **插** | **不插** |
| 每台保存 | 完整结果 | **只有部分和**（省显存） |

---

## 五、流水线并行：按**层**切（跨算子）

**流水线并行与前三种的层次不同**：

| 策略 | 切分粒度 |
|---|---|
| 数据 / 张量 / ZeRO | **单个算子内部**的维 |
| **流水线并行** | **算子之间**（按层分组） |

**★ 在 Shardy 里如何表达**：
- 流水线并行**不是**通过"某个算子的分片标注"实现的 ——
  而是**把不同层放到不同的网格/设备上**。
- 这可以用**多个 mesh**（**L7-01** 讲的"网格切换"）或
  **`sdy.manual_computation`**（**L4-11** 讲的"逐指令分区"）来表达。

**通信代价**：
- **只在层边界**通信（把上一层的输出传给下一层）。
- **层内部无通信** —— 这是流水线并行的优势。
- 代价是**流水线气泡**（bubble）—— 前几层在算时后几层空闲。

**★ 与其他三种的本质区别**：
> 数据/张量/ZeRO 是**空间上的切分**（同一个算子的不同维）；
> 流水线并行是**时间/层次上的切分**（不同算子在不同设备）。

---

## 六、★ 通信代价对比与选型

### 通信代价总表

| 策略 | 切的维 | 通信模式 | 通信量 | 显存收益 |
|---|---|---|---|---|
| **数据并行** | 收缩维 | `all_reduce` | ∝ **结果**大小 | 中 |
| **张量并行** | 非收缩维 | **无**（算子内） | 0 | **大** |
| **ZeRO** | 收缩维 + 未归约 | **无**（延迟） | 0（推迟） | **大** |
| **流水线并行** | 按层 | 层边界 | ∝ **层输出**大小 | **大** |

### ★ 选型判据

**给定模型规模与设备数，怎么选**：

| 情形 | 推荐 | 理由 |
|---|---|---|
| **模型能放进单卡，但数据量大** | **数据并行** | 切数据、`all_reduce` 梯度，最成熟 |
| **模型放不进单卡，层内可切** | **张量并行** | 切权重、**无通信**（算子内） |
| **模型放不进单卡，且要省显存** | **ZeRO** | 延迟归约，每台只存部分和 |
| **层数多、层间依赖弱** | **流水线并行** | 层边界通信，层内无通信 |
| **超大规模** | **组合** | 例如 TP × PP × DP（3D 并行） |

**★ 组合的例子**（回顾 **L5-05** 的 `dot_general`）：
- **批维切 `x`**（批并行）+ **收缩维切 `y`**（数据并行）= **2D 并行**
- 那里只有**收缩维**上的分片需要 `all_reduce` —— 批维上的不需要。

### ★ 一个反直觉的点

**张量并行"无通信"是有条件的**：
> 它指的是**单个算子内部**无通信。
> 但如果下游算子需要**完整的**输入，就要在**边界**插入 `all_gather` 之类的通信。
>
> **判据始终是 L5-05 那条**：**只有收缩维上的分片需要 `all_reduce`** ——
> 但"算子之间"的通信由**数据流**决定（**L2-07** 的 data-flow edges）。

---

## 七、本课在 L7 中的位置

| 课 | 主题 |
|---|---|
| `L7-01` | 端到端走查（**流水线**视角） |
| **`L7-02`（本课）** | **并行策略**（**策略**视角） |
| `L7-03` | 调试手册（**排错**视角） |
| `L7-04` | 跨方言集成（**扩展**视角） |

**★ 本课的三条结论**：

1. **四种策略的差异本质上是"切哪个维"的差异** ——
   而"切哪个维"直接决定了**需不需要通信**。
2. **判据仍是 L5-05 那条**：**只有收缩维上的分片需要 `all_reduce`**；
   非收缩维分片在**算子内**无通信。
3. **流水线并行是另一个层次的切分** ——
   它切的是**算子之间**（按层），而不是单个算子内部的维。

**一句话总结**：
> **并行策略 = 在"切哪个维"上做选择** ——
> 切收缩维是数据并行（要 `all_reduce`）、切非收缩维是张量并行（算子内无通信）、
> 不归约是 ZeRO（省显存）、按层切是流水线并行（层边界通信）。
