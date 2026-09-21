<!-- sdy-coverage
transforms/export/test/executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split_2groups.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split_custom_device_ids.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split_gap_2.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split_passthrough_indivisible.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_reshape_2d_split_unrelated_axis.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_reshape_axis_shift.mlir
-->

# L6-07 · exec-reshape — 源 IR

**`reshape` 把分片"拆"到新维度** —— 数值验证 **L4-03** 的 reshape reshard 规则。

| 文件 | 行数 |
|---|---|
| `executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split_2groups.mlir` | 69 |
| `executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split_passthrough_indivisible.mlir` | 56 |
| `executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split_gap_2.mlir` | 48 |
| `executable_partitioner_pipeline/stablehlo_reshape_axis_shift.mlir` | 48 |
| `executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split.mlir` | 39 |
| `executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split_custom_device_ids.mlir` | 39 |
| `executable_partitioner_pipeline/stablehlo_reshape_2d_split_unrelated_axis.mlir` | 39 |

**全部 7 个文件都有两个 RUN 行**：

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
```

→ 验证 **REPL 与 HALO 两种模式的结果一致**（**L4-09**）。

---

## 一、★ 基准：`1d_to_2d_split`

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true

//--- part1.mlir
sdy.mesh @mesh_a_4 = <["b"=2, "c"=2]>

func.func @parallel_reshape_1d_to_2d_split(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"b", "c"}]>}) -> (tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"b", "c"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"b"}, {"c"}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>
  %2 = sdy.reshard %1 <@mesh_a_4, [{}, {}]> : tensor<2x3xi32>
  return %2 : tensor<2x3xi32>
}
```

**读法**（**本课的核心机制**）：
- 网格是 `b=2, c=2` —— **4 台设备**。
- `%arg0: tensor<8xi32>` 切 `[{"b", "c"}]` —— **两个轴都切在第 0 维**（复合分片），
  所以 `8 / (2×2) = 2` 个元素/台。
- **`slice [0:6]`** —— 切到 `6`（因为 `8` 拆不成 `2x3`；`2×3 = 6`）。
- **★ `reshape` 到 `2x3`，分片变成 `[{"b"}, {"c"}]`** ——
  **一维的复合分片被"拆"到两个维度**！
  - `b` 切第 0 维（`2 / 2 = 1`）
  - `c` 切第 1 维（`3 / 2` —— **除不尽**！）
- → **`sdy.reshard` 到全复制 `[{}, {}]`** —— 需要通信。

**★ 这是本课的核心**：
> `reshape` 改变维度结构时，**分片可能不再匹配** → 需要 **`reshard`**。
> 这正是 **L4-03** 讲的 reshape reshard 规则。

### 验证方式

```mlir
//--- part2.mlir
func.func @main() {
```

```mlir
  %input_seq = stablehlo.iota dim = 0 : tensor<8xi32>
  %c1 = stablehlo.constant dense<1> : tensor<8xi32>
  %input = stablehlo.add %input_seq, %c1 : tensor<8xi32>
  %seq = func.call @sequential_reshape_1d_to_2d_split(%input) : (tensor<8xi32>) -> tensor<2x3xi32>
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 6>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_reshape_1d_to_2d_split, @parallel_reshape_1d_to_2d_split, @parallel_reshape_1d_to_2d_split, @parallel_reshape_1d_to_2d_split]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) ->
      (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>)
  "check.expect_eq"(%res#0, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  return
}
```

**读法**（**逐设备推数值**）：
- `%input = iota(8) + 1` = **`[1, 2, 3, 4, 5, 6, 7, 8]`**
- `slice [0:6]` → `[1, 2, 3, 4, 5, 6]`
- `reshape` → **`[[1, 2, 3], [4, 5, 6]]`**（`2x3`）
- `%seq` 就是它（串行版的结果）。

**4 台设备的输入**（`8/4 = 2` 个元素/台）：

| 设备 | `slice` | 输入 |
|---|---|---|
| 0 | `[0:2]` | `[1, 2]` |
| 1 | `[2:4]` | `[3, 4]` |
| 2 | `[4:6]` | `[5, 6]` |
| 3 | `[6:8]` | `[7, 8]` |

**4 台设备的结果都等于 `%seq`** —— 因为最后 **`reshard` 到全复制**，
每台设备都拿到了完整的结果 ✓

**★ 注意 `iota + 1` 这个技巧**：
`iota` 从 `0` 开始，如果直接用它，第 0 个元素是 `0` ——
**可能与"填充值 0"混淆**。`+1` 让所有元素都非零，**便于辨识**。
（与 L6-03 的"缩放技巧"、L6-06 的"填充值用 9"同源。）

---

## 二、★ 7 个文件的差异对照

| 文件 | 网格 | 分片变化 | 特点 |
|---|---|---|---|
| `1d_to_2d_split` | `b=2, c=2` | `[{"b","c"}]` → `[{"b"},{"c"}]` | 基准 |
| `_2groups` | `a=4, b=4` | `[{"a"},{"b"}]` → 四个**子轴** | **子轴** |
| `_custom_device_ids` | `b=2, c=2`，**`device_ids=[3,2,1,0]`** | 同基准 | **设备号倒序** |
| `_gap_2` | `b=2, c=3` | `[{"b","c"}]` → `[{"b"},{"c"}]` | **`c=3` 奇数** |
| `_passthrough_indivisible` | `x=2, b=2, c=2` | `[{"x"},{"b","c"}]` → `[{"x"},{"b"},{"c"}]` | **pass-through 轴** |
| `2d_split_unrelated_axis` | `a=2, b=2` | `[{"a"},{"b"}]` → `[{"a"},{},{"b"}]` | **中间维未分片** |
| `axis_shift` | `a=4` | 单轴 → `[{},{}]` | **轴移位** |

### `_2groups`：用**子轴**拆两组

```mlir
sdy.mesh @mesh_ab_16 = <["a"=4, "b"=4]>
func.func @parallel_reshape_1d_to_2d_split_2groups(%arg0: tensor<8x16xi32> {sdy.sharding = #sdy.sharding<@mesh_ab_16, [{"a"}, {"b"}]>}) -> (tensor<2x3x2x7xi32> {sdy.sharding = #sdy.sharding<@mesh_ab_16, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6, 0:14] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_ab_16, [{"a"}, {"b"}]>]>} : (tensor<8x16xi32>) -> tensor<6x14xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_ab_16, [{"a":(1)2}, {"a":(2)2}, {"b":(1)2}, {"b":(2)2}]>]>} : (tensor<6x14xi32>) -> tensor<2x3x2x7xi32>
```

**读法**：
- 网格是 **`a=4, b=4`**（**16 台设备**）。
- 输入 `8x16` 切 `[{"a"}, {"b"}]`。
- `slice [0:6, 0:14]` → `6x14`。
- **`reshape` 到 `2x3x2x7`** —— **两个维度拆成四个**！
- **分片用子轴**：`[{"a":(1)2}, {"a":(2)2}, {"b":(1)2}, {"b":(2)2}]`
  - `a=4` 被拆成两组 `(1)2` 和 `(2)2` —— **`a` 的后半给了第 1 维**
  - `b=4` 同理
  - → `6 / 2 = 3`（第 1 维）、`2 / 2 = 1`（第 0 维）、`14 / 2 = 7`（第 3 维）、`2 / 2 = 1`（第 2 维）

**★ 这验证了 L2-02 讲的子轴**：当一个轴需要**同时切多个维**时，
必须用**子轴**把轴的大小分配下去。

### `_custom_device_ids`：设备号可以自定义

```mlir
sdy.mesh @mesh_custom = <["b"=2, "c"=2], device_ids=[3, 2, 1, 0]>
func.func @parallel_reshape_1d_to_2d_split_custom(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{"b", "c"}]>}) -> (tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_custom, [{"b", "c"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_custom, [{"b"}, {"c"}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>
```

**读法**：
- **`device_ids=[3, 2, 1, 0]`** —— **设备号倒序**！（回顾 **L1-02** 的 `device_ids`）
- 除设备号外，其余与基准**完全相同**。
- **目的**：验证**设备号的映射顺序不影响结果** ——
  因为 `reshape` 后的 `reshard` 通信依赖"谁和谁通信"，
  而 `device_ids` 决定了**物理设备号**的排列。

**★ 这验证了 L1-02 讲的 `device_ids`** ——
那里讲"逻辑设备号 → 物理设备号的映射"，这里验证它**不影响语义**。

### `_gap_2`：网格是 `b=2, c=3`

```mlir
sdy.mesh @mesh_bc_6 = <["b"=2, "c"=3]>
func.func @parallel_reshape_1d_to_2d_split_gap_2(%arg0: tensor<18xi32> {sdy.sharding = #sdy.sharding<@mesh_bc_6, [{"b", "c"}]>}) -> (tensor<2x7xi32> {sdy.sharding = #sdy.sharding<@mesh_bc_6, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:14] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_bc_6, [{"b", "c"}]>]>} : (tensor<18xi32>) -> tensor<14xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_bc_6, [{"b"}, {"c"}]>]>} : (tensor<14xi32>) -> tensor<2x7xi32>
```

**读法**：
- 网格 `b=2, c=3` —— **6 台设备**（**奇数轴**）。
- 输入 `18` 切 `{"b","c"}` → `18 / 6 = 3` 个元素/台。
- `slice [0:14]` → `14`；`reshape` 到 `2x7`（`2×7 = 14`）。
- **"gap" 的含义**：`c=3` 是奇数，`7 / 3` **除不尽** → 需要处理**间隙**。

### `_passthrough_indivisible`：pass-through 轴

```mlir
sdy.mesh @mesh_xbc_8 = <["x"=2, "b"=2, "c"=2]>
func.func @parallel_reshape_1d_to_2d_split_passthrough_indivisible(%arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xbc_8, [{"x"}, {"b", "c"}]>}) -> (tensor<3x2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_xbc_8, [{}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:3, 0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xbc_8, [{"x"}, {"b", "c"}]>]>} : (tensor<4x8xi32>) -> tensor<3x6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xbc_8, [{"x"}, {"b"}, {"c"}]>]>} : (tensor<3x6xi32>) -> tensor<3x2x3xi32>
```

**读法**：
- 网格 `x=2, b=2, c=2`（**8 台设备**）。
- 输入 `4x8` 切 `[{"x"}, {"b","c"}]` —— **`x` 切第 0 维、`{"b","c"}` 切第 1 维**。
- `slice [0:3, 0:6]` → `3x6`。
- **`reshape` 到 `3x2x3`** —— 第 1 维（`6`）拆成两维（`2x3`）。
- **`x` 轴原样"传递"**（pass-through）：`[{"x"}, {"b"}, {"c"}]` ——
  **`x` 仍在第 0 维，未被 reshape 影响** ✓
- **`indivisible`**：`3` 不能被 `x=2` 整除 → 需要补齐/裁回。

### `2d_split_unrelated_axis`：中间维未分片

```mlir
sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>
func.func @parallel_reshape_2d_split_unrelated_axis(%arg0: tensor<6x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>}) -> (tensor<1x5x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:5, 0:4] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>]>} : (tensor<6x4xi32>) -> tensor<5x4xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a2_b2, [{"a"}, {}, {"b"}]>]>} : (tensor<5x4xi32>) -> tensor<1x5x4xi32>
```

**读法**：
- 输入 `6x4` 切 `[{"a"}, {"b"}]`。
- `slice [0:5, 0:4]` → `5x4`。
- **`reshape` 到 `1x5x4`** —— **在最前面插入一个大小为 1 的维**！
- 分片变成 `[{"a"}, {}, {"b"}]` —— **新插入的中间维未分片** ✓
- **"unrelated axis"**：插入的维（`1`）与分片**无关** ——
  因为它的大小是 1，切不了。

### `axis_shift`：轴移位

```mlir
sdy.mesh @mesh = <["a"=4]>
func.func @parallel_reshape_axis_shift(
  %arg0: tensor<24xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> (tensor<3x6xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
```

**读法**：
- 网格 `a=4`（4 台设备），输入 `24` 切 `{"a"}`。
- 结果 `3x6` 全复制。
- **"axis shift"**：分片的轴从"第 0 维"变成"没有维"（全复制）——
  轴的位置发生了**根本变化**。

---

## 三、★ 共同结构：`slice` → `reshape` → `reshard`

**7 个文件都是同一个三步模式**：

```text
%0 = stablehlo.slice %arg0 [...]   // ① 切到可 reshape 的大小
```

```text
%1 = stablehlo.reshape %0 {...}    // ② reshape 并【重排分片】
```

```text
%2 = sdy.reshard %1 <...>          // ③ reshard（可能通信）
```

**★ 为什么需要 `slice`**：
`reshape` 要求**元素总数不变**。但原形状往往**拆不成**目标形状
（如 `8` 拆不成 `2x3`）→ 先 `slice` 到能拆的大小（`6`）。

**★ 为什么需要 `reshard`**：
`reshape` 后分片可能**不再匹配**（如 `c` 要切 `3/2` 除不尽）→ 需要通信。

**★ 与 L4-03 的呼应**：
L4-03 讲 reshape 的 **reshard 插入规则**（什么时候需要通信）；
本课用**真实数值**验证那些规则的正确性。

---

## 四、7 个文件的族谱

| 族 | 文件 | 验证什么 |
|---|---|---|
| **基准** | `1d_to_2d_split` | 一维复合分片拆到两维 |
| **子轴** | `_2groups` | 一个轴拆给多个维（**L2-02**） |
| **设备号** | `_custom_device_ids` | `device_ids` 倒序不影响语义（**L1-02**） |
| **奇数轴** | `_gap_2` | `c=3` 的间隙处理 |
| **pass-through** | `_passthrough_indivisible` | 未涉及的轴原样传递 |
| **无关轴** | `2d_split_unrelated_axis` | 插入大小为 1 的未分片维 |
| **轴移位** | `axis_shift` | 单轴 → 全复制 |

**★ 本课的三条结论**：

1. **`reshape` 的降级是三步模式**：`slice`（切到可 reshape）→
   `reshape`（重排分片）→ `reshard`（通信）。
2. **分片的重排可以很复杂**：子轴（`_2groups`）、pass-through（`_passthrough`）、
   插入未分片维（`unrelated_axis`）—— 每种都要验证。
3. **两个 RUN 行验证 REPL/HALO 等价** —— 全部 7 个文件都是，
   因为 `reshape` 的 reshard 可能触发 halo exchange。

**一句话总结**：
> **`reshape` 改变维度结构 → 分片必须重排 → 可能触发通信。**
> 7 个文件覆盖了分片重排的各种情形（子轴 / 设备号 / 奇数轴 /
> pass-through / 无关轴 / 轴移位），并用真实数值验证结果与串行版一致。
