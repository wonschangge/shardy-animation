<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/stablehlo_slice.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_slice_comm_free.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_slice_with_communication.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_slice_indivisible.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_slice_replicated.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_slice_replicated_mesh_2.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_slice_strided.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_reverse_multi_dim_divisible.mlir
transforms/export/test/executable_partitioner_pipeline/stablehlo_reverse_single_dim_indivisible.mlir
-->

# L6-08 · exec-reverse-slice — 源 IR

**`slice` / `reverse` 的执行验证** —— 核心问题是**切片是否跨越设备边界**。

| 族 | 文件数 | 位置 |
|---|---|---|
| `slice` | 7 | 1 个在 `convert_global_to_local/`，6 个在 `partitioner_pipeline/` |
| `reverse` | 2 | `partitioner_pipeline/` |

---

## 一、★ 核心问题：切片是否**跨越设备边界**

`slice` 的降级取决于一个判据：

> **切片的范围是否跨越设备边界？**

| 情形 | 例子 | 通信 |
|---|---|---|
| **不跨越** | `slice [0:4, 0:2]`（第 0 维完整） | **无**（`comm_free`） |
| **跨越** | `slice [1:5, 0:4]`（第 0 维跨了两台） | **需要**（`with_communication`） |

**两个文件的命名直接点明了这个区分**。

---

## 二、`slice_comm_free`：切片**不**跨界

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
```

```mlir
sdy.mesh @mesh = <["x"=2]>
```

```mlir
func.func @parallel_slice_comm_free(
```

```mlir
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
```

```mlir
  -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
```

```mlir
  %0 = stablehlo.slice %arg0 [0:4, 0:2]
```

```mlir
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<4x4xi32>) -> tensor<4x2xi32>
```

**读法**：
- 输入 `4x4` 切 `[{"x"}, {}]` —— 第 0 维分片（`4/2 = 2` 行/台）。
- **`slice [0:4, 0:2]`** —— 第 0 维是 **`0:4`（完整）**、第 1 维切到 `0:2`。
- **第 0 维（分片维）的切片范围是完整的** → **切片本身不跨设备** → **`comm_free`** ✓
- 结果 `4x2` 声明为 `[{}, {}]`（全复制）—— 后续的 reshard 由流水线处理。

**★ 与 L5-04 的"作用维判据"的关系**：
L5-04 讲"**作用维未分片** → 参数不变"。本用例更细一层：
**即使作用维就是分片维，只要切片范围覆盖完整，也不跨设备**。

---

## 三、`slice_with_communication`：切片**跨越**设备边界

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
```

```mlir
sdy.mesh @mesh = <["x"=2]>
```

```mlir
func.func @parallel_slice_with_communication(
```

```mlir
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
```

```mlir
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
```

```mlir
  %0 = stablehlo.slice %arg0 [1:5, 0:4]
```

```mlir
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<4x4xi32>
```

**读法**（**与 `comm_free` 的唯一区别**）：
- 输入 `8x4` 切 `{"x"}` —— 设备 0 有第 **0~3** 行、设备 1 有第 **4~7** 行。
- **`slice [1:5, 0:4]`** —— 第 0 维取第 **1~4** 行。
  - 第 **1、2、3** 行在**设备 0**
  - 第 **4** 行在**设备 1**
  - → **切片范围跨越了两台设备**！
- → **切片本身就需要通信** → 显式的 **`sdy.reshard`** ✓

**★ 这就是 `comm_free` 与 `with_communication` 的区别**：
> **切片的范围是否跨越设备边界。**

---

## 四、`slice_indivisible`：`3` 不能被 `4` 整除

```mlir
sdy.mesh @mesh_x4 = <["x"=4]>
```

```mlir
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{"x"}, {}]>})
```

```mlir
  -> (tensor<3x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}]>}) {
```

```mlir
  %0 = stablehlo.slice %arg0 [0:3, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{"x"}, {}]>]>} : (tensor<4x2xf32>) -> tensor<3x2xf32>
```

```mlir
  %0 = stablehlo.slice %arg0 [0:3, 0:2] : (tensor<4x2xf32>) -> tensor<3x2xf32>
```

**读法**：
- 网格 `x=4`（**4 台设备**），输入 `4x2` 切 `{"x"}` → **每台 1 行**。
- `slice [0:3, 0:2]` —— 第 0 维取前 **3** 行。
- **`3` 不能被 `4` 整除**！→ **不可整除** ✓
- 需要补齐（**L5-09** 的规则）。

**★ 注意第二行**：`%0 = stablehlo.slice %arg0 [0:3, 0:2] : (tensor<4x2xf32>) -> tensor<3x2xf32>`
—— 这是**串行版**的写法（无 `sdy.sharding` 属性）。

---

## 五、`slice_replicated` 与 `slice_replicated_mesh_2`

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2]>
```

```mlir
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>})
```

```mlir
  -> (tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
```

```mlir
  %0 = stablehlo.slice %arg0 [0:3]
```

```mlir
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<3xi32>
```

```mlir
sdy.mesh @mesh = <["x"=2]>
```

```mlir
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
```

```mlir
  -> (tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
```

```mlir
  %0 = stablehlo.slice %arg0 [0:3]
```

```mlir
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}]>]>} : (tensor<8xi32>) -> tensor<3xi32>
```

**读法**（两个文件几乎相同，只差网格）：
- `slice_replicated`：网格 `a=2, b=2`（4 台），**只用 `b` 轴**分片
  —— 所以 `a` 轴是**多余的**（结果 `[{}]` 无分片）。
- `slice_replicated_mesh_2`：网格 `x=2`（2 台），单轴。
- 共同点：输入 `8` 切 `{"b"}`/`{"x"}`（`8/2 = 4` 个/台），
  `slice [0:3]` 取前 3 个，结果声明 `[{}]`（**全复制**）。

**★ 这两个文件验证"结果全复制"的情形** ——
输入有分片、结果无分片 → 需要 reshard 到全复制。
而 `slice_replicated` 的网格有**两个轴但只用一个** —— 验证**未使用的轴**不影响。

---

## 六、`slice_strided`：带**步长**的切片

```mlir
sdy.mesh @mesh = <["x"=2]>
```

```mlir
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
```

```mlir
  -> (tensor<3x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
```

```mlir
  %0 = stablehlo.slice %arg0 [1:7:2, 0:4]
```

```mlir
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>} : (tensor<8x4xi32>) -> tensor<3x4xi32>
```

**读法**：
- `slice [1:7:2, 0:4]` —— 第 0 维：**起始 1、结束 7、步长 2**。
- 取第 **1, 3, 5** 行（3 个元素）✓（结果 `3x4`）
- **步长让切片更复杂**：每台设备要算"自己那段里哪些元素被选中"。
  - 设备 0 有第 0~3 行 → 选中第 1、3 行
  - 设备 1 有第 4~7 行 → 选中第 5 行
- 结果的局部形状**每台不同**（设备 0 得 2 行、设备 1 得 1 行）→ 需要处理。

**★ 这是 `slice` 最复杂的情形** —— 步长让"每台设备拿多少"不再是简单除法。

---

## 七、`reverse` 的两个文件

### `reverse_multi_dim_divisible`：多维 reverse

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2]>
```

```mlir
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
```

```mlir
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
```

```mlir
  %0 = stablehlo.slice %arg0 [0:4, 0:8]
```

```mlir
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>}
```

**读法**：
- 网格 `a=2, b=2`，输入 `4x8` 切 `[{"a"}, {"b"}]`。
- `slice [0:4, 0:8]` —— **完整范围**（不切）—— 这一步只是为了触发后续处理。
- 用例名 `multi_dim_divisible` —— **多维 reverse + 可整除**。

### `reverse_single_dim_indivisible`：单维 reverse + 不可整除

```mlir
sdy.mesh @mesh_abc = <["a"=2, "b"=2, "c"=4]>
```

```mlir
  %arg0: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
```

```mlir
  %0 = stablehlo.slice %arg0 [0:4, 0:6, 0:5]
```

```mlir
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
```

```mlir
  %1 = stablehlo.reverse %0, dims = [0, 2]
```

**读法**：
- 网格 `a=2, b=2, c=4`（**16 台设备**）。
- 输入 `4x6x8`，分片是 **`[{"b"}, {"a"}, {"c"}]`** ——
  **注意轴顺序与维度顺序不同**：第 0 维切 `b`、第 1 维切 `a`、第 2 维切 `c`。
- `slice [0:4, 0:6, 0:5]` —— 第 2 维切到 `5`（`8 → 5`）。
- **`reverse %0, dims = [0, 2]`** —— **反转第 0 维和第 2 维**！
- **`indivisible`**：`5` 不能被 `c=4` 整除 → 不可整除 ✓

**★ `reverse` 的关键问题**：
> 反转一个**被分片**的维时，**分片的顺序也要反转**！
> 设备 0 原本拿第 0 段，反转后应该拿**最后一段** → 需要通信。

---

## 八、9 个文件的族谱

| 族 | 文件 | 验证什么 |
|---|---|---|
| **基准** | `convert_global_to_local/stablehlo_slice` | 切片维未分片 |
| **不跨界** | `slice_comm_free` | 切片范围完整 → **无通信** |
| **跨界** | `slice_with_communication` | 切片跨设备 → **需通信** |
| **不可整除** | `slice_indivisible` | `3` 不能被 `4` 整除 |
| **全复制** | `slice_replicated` | 结果 `[{}]` + 网格有未用轴 |
| **全复制（单轴）** | `slice_replicated_mesh_2` | 同上，网格单轴 |
| **步长** | `slice_strided` | `[1:7:2]` 步长让局部形状不同 |
| **reverse 多维** | `reverse_multi_dim_divisible` | 多维反转 + 可整除 |
| **reverse 单维** | `reverse_single_dim_indivisible` | 单维反转 + 不可整除 + 轴序不同 |

**★ 本课的三条结论**：

1. **`slice` 的核心判据是"切片是否跨越设备边界"** ——
   不跨越则 `comm_free`，跨越则需通信。
2. **`reverse` 的难点是"分片顺序也要反转"** ——
   反转被分片的维时，设备之间的数据要重新分配。
3. **不可整除与步长是两个额外复杂度** ——
   前者要补齐（L5-09），后者让"每台拿多少"不再是简单除法。

**一句话总结**：
> **`slice` / `reverse` 的降级取决于「切片/反转是否改变设备之间的数据归属」** ——
> `slice` 看范围是否跨界、`reverse` 看是否反转了被分片的维。
