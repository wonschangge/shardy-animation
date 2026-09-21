<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/stablehlo_dot_general.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_fft.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_iota.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_dot_general_indivisible.mlir
-->

# L6-04 · exec-matmul-fft-iota — 源 IR

**三个算子 + 一个不可整除的情形** —— 验证 L5-04 / L5-05 / L5-09。

| 文件 | 行数 |
|---|---|
| `executable_convert_global_to_local/stablehlo_dot_general.mlir` | 57 |
| `executable_convert_global_to_local/stablehlo_fft.mlir` | 43 |
| `executable_convert_global_to_local/stablehlo_iota.mlir` | 35 |
| `executable_partitioner_pipeline/stablehlo_dot_general_indivisible.mlir` | — |

网格：

```mlir
sdy.mesh @mesh_2 = <["x"=2]>
```

---

## 一、四个文件与对应课

| 文件 | 验证什么 | 对应课 |
|---|---|---|
| `stablehlo_iota` | **iota 的偏移补偿** | **L5-04** |
| `stablehlo_fft` | FFT 的**作用维未分片** | **L5-04** |
| `stablehlo_dot_general` | 收缩维分片 = **数据并行** | **L5-05** |
| `dot_general_indivisible` | **不可整除时补齐**（3→4） | **L5-09** |

---

## 二、★ `iota`：验证 L5-04 的偏移补偿

```mlir
sdy.mesh @mesh_2 = <["x"=2]>

// Performs the same iota as in sequential_iota, but on 2 devices in parallel.
//
// We will use sdy-opt to convert this to a device local program that
// stablehlo interpreter can execute.
//
func.func @parallel_iota()
  -> (tensor<2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : tensor<2xi32>
  return %0 : tensor<2xi32>
}
```

**读法**：
- `@parallel_iota` **无参数** —— 因为 iota 不需要输入，只生成序号。
- 全局 `tensor<2xi32>` 沿 `{"x"}`（`x=2`）切 → **每台设备 1 个元素**。
- 全局 iota 是 `[0, 1]`；设备 0 该拿 `[0]`、设备 1 该拿 `[1]`。

### 验证方式

```mlir
// Main Orchestrator: executes the sequential and parallel iota and checks that
// they are equivalent.
func.func @main() {
```

```mlir
  %seq = func.call @sequential_iota() : () -> tensor<2xi32>

  %pars:2 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota, @parallel_iota]]
  } : () -> (tensor<1xi32>, tensor<1xi32>)
  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {
    dimension = 0 : i64
  } : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>

  "check.expect_eq"(%seq, %par) : (tensor<2xi32>, tensor<2xi32>) -> ()

  return
}
```

**读法**：
- `run_parallel()` —— **无输入**（iota 不需要输入）✓
- 每台设备各得 `tensor<1xi32>`。
- `concatenate` 拼回 `tensor<2xi32>` → 与 `%seq` 比较。

**★ 这验证了 L5-04 讲的 iota 偏移补偿**：
- **问题**：每台设备本地 `iota` 只会生成 `[0]` —— 但设备 1 该拿 `[1]`！
- **降级后的解法**（L5-04 讲过）：插入 `partition_id` + 查找表 `[0, 1]` + 加法。
- **本课证明**：补偿后 `concatenate` 的结果等于串行版 `[0, 1]` ✓

**★ 如果没做偏移补偿**：两台设备都会输出 `[0]` → 拼接得 `[0, 0]` ≠ `[0, 1]`。
**这个测试恰好能抓住那个 bug** —— 因为 `[0,0]` 与 `[0,1]` 不同。

---

## 三、★ `fft`：作用维**未**分片

```mlir
// This function computes the FFT of a 2x4 complex tensor.
// It is sharded along dimension 0 (batch), so each device performs a 4-point
// FFT.
func.func @parallel_fft(
  %arg0: tensor<2x4xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<2x4xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
```

```mlir
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 4>,
    fft_type = #stablehlo<fft_type FFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
  return %0 : tensor<2x4xcomplex<f64>>
}
```

**读法**（**注释直接给出了结论**）：
```
// It is sharded along dimension 0 (batch), so each device performs a 4-point
// FFT.
```
- **分片在第 0 维（批维）**；**FFT 作用在第 1 维**（`fft_length = 4`）。
- → **作用维未被分片** → **每台设备独立做 4 点 FFT** → **无通信** ✓

**★ 这正是 L5-04 讲的"作用维判据"**：
> **算子的"作用维"是不是被分片的那一维？**
> - FFT 的作用维 = `fft_length` 对应的维（第 1 维）
> - 分片在第 0 维 → **不冲突** → 直接局部化

**注意类型是 `complex<f64>`** —— 本课特有的复数类型。

### 验证方式

```mlir
  // Input tensor with 8 unique complex values.
  %input = stablehlo.constant dense<[
    [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
    [(5.0, 0.0), (7.0, 0.0), (11.0, 0.0), (13.0, 0.0)]
  ]> : tensor<2x4xcomplex<f64>>

  %seq = func.call @sequential_fft(%input) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>

  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 1, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xcomplex<f64>>) -> tensor<1x4xcomplex<f64>>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 1, 0>, limit_indices=array<i64: 2, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xcomplex<f64>>) -> tensor<1x4xcomplex<f64>>
  %pars:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_fft, @parallel_fft]]
  } : (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>) -> (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>)
  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {dimension = 0 : i64}
    : (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>

  "check.expect_eq"(%seq, %par) : (tensor<2x4xcomplex<f64>>, tensor<2x4xcomplex<f64>>) -> ()
```

**读法**：
- 输入是 **8 个各不相同的复数值**（`1/2/3/4` 与 `5/7/11/13`）——
  **都是质数**，避免巧合相等（与 L6-03 的"缩放"技巧同源）。
- **手动 `slice` 成 `%s0`/`%s1`**（按批维）—— L6-03 讲的技巧 ②。
- 各自 FFT → `concatenate` 拼回 → 与串行版比较 ✓

**★ 这验证了 L5-04 的"作用维判据"**：
> 作用维（第 1 维）未分片 → 每台独立算 → **无需通信** → 结果正确。

---

## 四、★ `dot_general`：收缩维分片 = 数据并行

```mlir
func.func @sequential_dot(%arg0: tensor<2x4xi32>, %arg1: tensor<4x2xi32>) -> tensor<2x2xi32> {
```

```mlir
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}
```

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

**读法**：
- `lhs_contracting_dimensions = [1]`、`rhs_contracting_dimensions = [0]`
  —— **收缩维**是 `%arg0` 的第 1 维、`%arg1` 的第 0 维。
- 分片：`%arg0` 的**第 1 维（收缩维）**切 `{"x"}`、`%arg1` 的**第 0 维（收缩维）**切 `{"x"}`
  —— **两个收缩维都被切** → 产生**部分和** → `unreduced={"x"}` + `all_reduce` ✓

**★ 这正是 L5-05 讲的"收缩维分片 = 数据并行"**。

### 验证方式（注释直接标注了每台设备拿什么）

```mlir
  %lhs = stablehlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>
  %rhs = stablehlo.constant dense<[[1, 16], [2, 32], [4, 64], [8, 128]]> : tensor<4x2xi32>

  %seq = func.call @sequential_dot(%lhs, %rhs) : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>

  // Device 0: lhs[:, 0:2], rhs[0:2, :]
  %lhs0 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 2, 2>, strides=array<i64: 1, 1>} : (tensor<2x4xi32>) -> tensor<2x2xi32>
  %rhs0 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 2, 2>, strides=array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  // Device 1: lhs[:, 2:4], rhs[2:4, :]
  %lhs1 = "stablehlo.slice"(%lhs) {start_indices=array<i64: 0, 2>, limit_indices=array<i64: 2, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xi32>) -> tensor<2x2xi32>
  %rhs1 = "stablehlo.slice"(%rhs) {start_indices=array<i64: 2, 0>, limit_indices=array<i64: 4, 2>, strides=array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %pars:2 = "interpreter.run_parallel"(%lhs0, %rhs0, %lhs1, %rhs1) {
    programs = [[@parallel_dot, @parallel_dot]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi32>, tensor<2x2xi32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<2x2xi32>, tensor<2x2xi32>) -> ()
```

**读法**（**注释直接标注了分片方案**）：
```
// Device 0: lhs[:, 0:2], rhs[0:2, :]
// Device 1: lhs[:, 2:4], rhs[2:4, :]
```
- **设备 0** 拿 `lhs` 的第 0~1 列 + `rhs` 的第 0~1 行（收缩维的前半）
- **设备 1** 拿 `lhs` 的第 2~3 列 + `rhs` 的第 2~3 行（收缩维的后半）
- 各算一个**部分和** → `all_reduce` 合并 → **两台设备都得到完整结果**
- 断言 **两个设备**都等于 `%seq` ✓

**★ 这验证了 L5-05 的"数据并行"**：
> 收缩维切分 → 各算部分和 → `all_reduce` → 结果正确。

**注意 `%rhs` 的取值**（`1/16, 2/32, 4/64, 8/128`）——
每行是**倍数关系**，避免不同分片方式的乘积巧合相等。

---

## 五、★ `dot_general_indivisible`：验证 L5-09 的 3→4 补齐

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
```

```mlir
// Contracting dimension is sharded and indivisible (padded 3->4).
func.func @parallel_dot_contracting_indivisible(
  %arg0: tensor<3x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
  %arg1: tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
  -> (tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
```

```mlir
  %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{"x"}, {}], unreduced={"y"}>]>} : (tensor<3x3xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
```

**读法**（**注释直接给出了答案**）：
```
// Contracting dimension is sharded and indivisible (padded 3->4).
```
- **收缩维大小是 3**，沿 `y=2` 分片 → **3 不能被 2 整除**！
- → **补齐 `3 → 4`** ✓
- 网格是 `@mesh_x2_y2`（2x2），结果标 `unreduced={"y"}`。

**★★ 这正是 L5-09 验收点的答案**：
> L5-09 我讲过："补到**下一个能被轴整除的数**"，
> 并给出验收点 `tensor<7x3x8>` 沿 `z=3` 分片 → 补到 `9`。
>
> **本课的测试文件直接证实了这个规则** ——
> 收缩维 `3` 沿 `y=2` 分片 → **补到 `4`**（下一个 2 的倍数）。

**两个 RUN 行**：验证 `enable-halo-exchange` 的 `true`/`false` **结果一致**
（与 L6-03 的 `dual_semantics` 同源，对应 **L4-09**）。

### 验证方式

```mlir
func.func @sequential_dot_contracting_indivisible(%arg0: tensor<3x3xf32>, %arg1: tensor<3x5xf32>) -> tensor<3x5xf32> {
```

```mlir
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<3x3xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
```

```mlir
  "check.expect_eq"(%res#0, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
```

**读法**：
- `@sequential_dot_contracting_indivisible` 是**手写**的串行版（无分片）。
- **4 台设备**（2x2 网格）的结果**都等于** `%seq`。
- 注意结果类型仍是 `tensor<3x5xf32>` —— **补齐是内部的**，
  最终结果**裁回了原始大小**（L5-09 讲的"pad 补齐 + slice 裁回"）✓

**★ 这完整验证了 L5-09 的流程**：
```
收缩维 3 沿 y=2 分片（除不尽）
  -> pad 补齐到 4
  -> 通信 / 计算在可整除的形状上
  -> slice 裁回 3
  -> 结果与串行版一致 ✓
```

---

## 六、四个文件的族谱

| 文件 | 验证什么 | 对应课 |
|---|---|---|
| `stablehlo_iota` | **iota 的偏移补偿**（无偏移会得 `[0,0]`） | **L5-04** |
| `stablehlo_fft` | FFT **作用维未分片** → 无通信 | **L5-04** |
| `stablehlo_dot_general` | 收缩维分片 = **数据并行** | **L5-05** |
| `dot_general_indivisible` | **不可整除补齐 3→4** | **L5-09** |

**★ 本课的一个特殊价值**：
> **`dot_general_indivisible` 直接印证了 L5-09 的推导。**
> 那里我从"补到下一个能被轴整除的数"这条规则出发，
> 给出验收点 `7x3x8` 沿 `z=3` 补到 `9`；
> 本课的测试文件用一个**真实例子**（收缩维 `3` 沿 `y=2` 补到 `4`）证实了它。

**四课验证一条链**：
```
L5-04  iota 偏移补偿 / FFT 作用维判据
L5-05  收缩维分片 = 数据并行
L5-09  不可整除时补齐
  ↓
L6-04  用真实数值全部验证 ✓
```

**一句话总结**：
> **L6-04 用数值验证了 L5-04（位置补偿）、L5-05（数据并行）、L5-09（整除性补齐）三条规律。**
> 其中 `dot_general_indivisible` 的 `3→4` 补齐是 L5-09 那条规则最直接的证据。
