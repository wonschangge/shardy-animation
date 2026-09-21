<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/sdy_all_gather.mlir
transforms/export/test/executable_convert_global_to_local/sdy_all_reduce.mlir
transforms/export/test/executable_convert_global_to_local/sdy_all_slice.mlir
transforms/export/test/executable_convert_global_to_local/sdy_all_to_all.mlir
transforms/export/test/executable_convert_global_to_local/sdy_all_to_all_cross_replica.mlir
transforms/export/test/executable_convert_global_to_local/sdy_collective_permute_cross_replica.mlir
transforms/export/test/executable_convert_global_to_local/sdy_collective_permute_without_self_loops.mlir
transforms/export/test/executable_convert_global_to_local/sdy_collective_permute_with_self_loops.mlir
transforms/export/test/executable_convert_global_to_local/sdy_reduce_scatter.mlir
transforms/export/test/executable_convert_global_to_local/sdy_constant.mlir
-->

# L6-01 · exec-sdy-collectives — 源 IR

**用 L6-00 讲的机制，验证 L5-02 降级出的集合通信真的能跑出正确数值。**

| 文件 | 行数 |
|---|---|
| `executable_convert_global_to_local/sdy_all_slice.mlir` | 92 |
| `executable_convert_global_to_local/sdy_all_gather.mlir` | 67 |
| `executable_convert_global_to_local/sdy_all_reduce.mlir` | 65 |
| `executable_convert_global_to_local/sdy_all_to_all.mlir` | 64 |
| `executable_convert_global_to_local/sdy_reduce_scatter.mlir` | 55 |
| `executable_convert_global_to_local/sdy_constant.mlir` | 53 |
| `executable_convert_global_to_local/sdy_collective_permute_without_self_loops.mlir` | 50 |
| `executable_convert_global_to_local/sdy_all_to_all_cross_replica.mlir` | 48 |
| `executable_convert_global_to_local/sdy_collective_permute_cross_replica.mlir` | 48 |
| `executable_convert_global_to_local/sdy_collective_permute_with_self_loops.mlir` | 48 |

**合计 590 行 / 10 个文件。**

RUN 行（L6-00 讲的机制）：

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t
```

---

## 一、★ 可执行测试的结构

以 `sdy_all_reduce.mlir` 为例。

### `part1.mlir`：两个分片函数

```mlir
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// All-reduce across the entire mesh (both "x" and "y" axes).
func.func @all_reduce_xy(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>}) {
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh_2_2, [{}, {}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// All-reduce across only the "x" axis.
// In a 2x2 mesh, this creates two replica groups: {0, 2} and {1, 3}.
func.func @all_reduce_x(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>}) {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_2_2, [{}, {}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
```

**读法**：
- 网格是 `2x2`（**4 台设备**）。
- 两个函数都是 `all_reduce`，但**归约的轴不同**：
  - `@all_reduce_xy` 跨 `{"x", "y"}` —— 全部 4 台
  - `@all_reduce_x` 只跨 `{"x"}` —— 分成两组
- **注释直接给出了分组结果**：`{0, 2}` 和 `{1, 3}`。

### `part2.mlir`：每台设备给不同输入

```mlir
func.func @main() {
```

```mlir
  %c1 = stablehlo.constant dense<1> : tensor<4x4xi32>
  %c10 = stablehlo.constant dense<10> : tensor<4x4xi32>
  %c100 = stablehlo.constant dense<100> : tensor<4x4xi32>
  %c1000 = stablehlo.constant dense<1000> : tensor<4x4xi32>

  // 1. Test All-reduce across both axes ("x" and "y").
  // Sum = 1 + 10 + 100 + 1000 = 1111.
  %res_xy:4 = "interpreter.run_parallel"(%c1, %c10, %c100, %c1000) {
    programs = [[@all_reduce_xy, @all_reduce_xy, @all_reduce_xy, @all_reduce_xy]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) ->
      (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>)

  %expected_xy = stablehlo.constant dense<1111> : tensor<4x4xi32>
  "check.expect_eq"(%res_xy#0, %expected_xy) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_xy#3, %expected_xy) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
```

**读法**（**本课的核心机制**）：
- **`interpreter.run_parallel`** —— 在**4 台设备上并行运行**。
- `programs = [[@all_reduce_xy, @all_reduce_xy, @all_reduce_xy, @all_reduce_xy]]`
  —— 每台设备都运行 `@all_reduce_xy`。
- **四个操作数就是四台设备的输入**：设备 0 拿 `1`、设备 1 拿 `10`、
  设备 2 拿 `100`、设备 3 拿 `1000`。
- **`check.expect_eq`** —— 断言结果等于期望值。

**数值验证**：`1 + 10 + 100 + 1000 = 1111` ✓
（所有设备都得到 `1111` —— 因为 `all_reduce` 让每台设备都有完整的和）

### 只跨 `x` 轴的情形

```mlir
  // 2. Test All-reduce across only "x" axis.
  // Device grid (x, y):
  // (0,0): dev 0, input 1
  // (0,1): dev 1, input 10
  // (1,0): dev 2, input 100
  // (1,1): dev 3, input 1000
  // Replica groups for "x" (dim 0): {0, 2} and {1, 3}.
  %res_x:4 = "interpreter.run_parallel"(%c1, %c10, %c100, %c1000) {
    programs = [[@all_reduce_x, @all_reduce_x, @all_reduce_x, @all_reduce_x]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) ->
      (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>)

  %expected_x_02 = stablehlo.constant dense<101> : tensor<4x4xi32> // 1 + 100
  %expected_x_13 = stablehlo.constant dense<1010> : tensor<4x4xi32> // 10 + 1000

  "check.expect_eq"(%res_x#0, %expected_x_02) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_x#2, %expected_x_02) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_x#1, %expected_x_13) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_x#3, %expected_x_13) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
```

**★ 逐设备读**（TODOLIST 要的"哪台设备算出什么"）：

| 设备 | `(x, y)` | 输入 | 归约组 | 结果 |
|---|---|---|---|---|
| 0 | `(0, 0)` | `1` | `{0, 2}` | **`101`** = 1 + 100 |
| 1 | `(0, 1)` | `10` | `{1, 3}` | **`1010`** = 10 + 1000 |
| 2 | `(1, 0)` | `100` | `{0, 2}` | **`101`** |
| 3 | `(1, 1)` | `1000` | `{1, 3}` | **`1010`** |

**★ 设备号与网格坐标的对应**（注释直接给出）：

```
// Device grid (x, y):
// (0,0): dev 0, input 1
// (0,1): dev 1, input 10
// (1,0): dev 2, input 100
// (1,1): dev 3, input 1000
```

**读法**：
- 设备 0 是 `(x=0, y=0)`、设备 1 是 `(x=0, y=1)`、
  设备 2 是 `(x=1, y=0)`、设备 3 是 `(x=1, y=1)`。
- 即 **`x` 变化最慢**（设备 0、1 的 `x=0`；设备 2、3 的 `x=1`）。
- **`x` 是最 major 的轴** ✓

**★ 这直接验证了 L5-01 的推导**：
那里我从查找表 `[0,0,0,0,2,2,2,2]` **反推**出"设备号按轴序最 major 优先"。
这里测试文件的注释**直接确认**了这个规则 —— 设备 0、1 的 `x=0`，
所以 `x` 是变化最慢（最 major）的。

**为什么 replica groups 是 `{0, 2}` 和 `{1, 3}`**：
- 跨 `x` 归约 → **同一个 `y` 值**的设备归为一组
- `y=0` 的设备是 0 和 2 → `{0, 2}`
- `y=1` 的设备是 1 和 3 → `{1, 3}`
- ✓ 与注释一致

---

## 二、★ 各算子的"结果分布"对比

不同的集合通信算子，**结果的分布方式不同**：

| 算子 | 结果分布 | 例子 |
|---|---|---|
| `all_reduce` | **所有设备相同** | 都得到 `1111` |
| `collective_permute` | **每台设备不同** | `%e0`~`%e3` 各不同 |
| `all_gather` | 每台设备得到**拼接后的完整数据** | — |
| `all_slice` | 每台设备得到**自己那片** | — |

### `collective_permute`：每台设备结果不同

`sdy_collective_permute_with_self_loops.mlir` 的 `part2.mlir`：

```mlir
  "check.expect_eq"(%res#0, %e0) : (tensor<1xi32>, tensor<1xi32>) -> ()
  "check.expect_eq"(%res#1, %e1) : (tensor<1xi32>, tensor<1xi32>) -> ()
  "check.expect_eq"(%res#2, %e2) : (tensor<1xi32>, tensor<1xi32>) -> ()
  "check.expect_eq"(%res#3, %e3) : (tensor<1xi32>, tensor<1xi32>) -> ()
```

**读法**：**四台设备各有不同的期望值** ——
因为 `collective_permute` 是"**设备间交换数据**"的操作，
每台设备拿到的是**别人发来的**数据。

**这解释了用例名的区分**：
| 文件名 | 场景 |
|---|---|
| `..._with_self_loops` | 置换表里有"发给自己"的项 |
| `..._without_self_loops` | 没有"发给自己"的项 |
| `..._cross_replica` | 跨副本的置换 |

**回顾 L5-02**：那里看到 `collective_permute` 的 `source_target_pairs`
是一张**设备收发对照表**，且 `[1,4]`/`[4,1]` **成对出现**（交换语义）。
本课验证了它的**数值效果**。

---

## 三、★ 选项验证：`per-dim-all-gather`

`sdy_all_gather.mlir` 的 `part2.mlir` 有一个很特别的断言：

```mlir
  "check.expect_eq"(%res_comb#0, %res_pdim#0) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_comb#0, %cst) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
```

**读法**：
- `%res_comb` —— **合并模式**（combined，默认）
- `%res_pdim` —— **逐维模式**（`per-dim-all-gather=true`）
- 断言 **两者结果相同**（`%res_comb#0 == %res_pdim#0`）
- 并且**都等于期望常量** `%cst`

**★ 这验证了 L5-02 讲的 `per-dim-all-gather` 选项**：
那里看到它有**四个 RUN 行**（`enable-rgv3` × `per-dim-all-gather` 的 2×2 组合）。
本课证明：**两种模式语义等价** —— 只是 IR 形态不同（合并 vs 逐维）。

---

## 四、★ 常量验证：splat vs dense

`sdy_constant.mlir` 验证了 **L5-01 / L5-03** 讲的常量处理：

```mlir
func.func @sharded_dense_constant()
```

```mlir
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} dense<[
```

```mlir
func.func @sharded_splat_constant()
```

```mlir
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} dense<7> : tensor<4x4xi32>
```

```mlir
  %e_dense_0 = stablehlo.constant dense<[
```

```mlir
  %e_dense_1 = stablehlo.constant dense<[
```

```mlir
  "check.expect_eq"(%res_dense#0, %e_dense_0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
```

```mlir
  "check.expect_eq"(%res_dense#1, %e_dense_1) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
```

```mlir
  %e_splat = stablehlo.constant dense<7> : tensor<2x4xi32>
```

**★ 逐项对比**（**这直接验证了 L5-03 的核心洞察**）：

| 常量类型 | 每台设备的期望值 |
|---|---|
| **dense**（元素各不相同） | `%e_dense_0` 与 `%e_dense_1` —— **两台设备不同**！ |
| **splat**（`dense<7>`） | `%e_splat = dense<7>` —— **两台设备相同** |

**读法**：
- **dense 常量**：每台设备得到**不同的切片**（`%e_dense_0` ≠ `%e_dense_1`）
  → 必须用 `replica_id` + 切片（L5-01 讲的六步）
- **splat 常量**：每台设备得到**相同的值**（都是 `dense<7>`）
  → **不需要切片**（L5-03 讲的"一步搞定"）

**★ 这就是 L5-03 那个洞察的数值验证**：
> 那里我从 IR 形态推断"splat 不需要切片"；
> 这里测试文件用**两个不同的期望值**证明了这个推断 ——
> dense 的两台设备期望值不同、splat 的相同。

**同时注意类型**：
- 局部类型是 `tensor<2x4xi32>`（`4x4` 沿 `x=2` 切第 0 维 → `2x4`）
- **splat 的期望值也是 `tensor<2x4>`** —— 形状是局部的，但**内容处处相同** ✓

---

## 五、10 个文件的族谱

| 族 | 文件 | 验证什么 |
|---|---|---|
| **all_gather** | `sdy_all_gather` | 拼接结果 + `per-dim` 选项等价性 |
| **all_reduce** | `sdy_all_reduce` | 跨轴 vs 跨单轴的分组差异 |
| **all_slice** | `sdy_all_slice` | 每台设备拿自己那片 |
| **all_to_all** | `sdy_all_to_all`、`sdy_all_to_all_cross_replica` | 轴跨维移动的数值效果 |
| **collective_permute** | `..._with_self_loops`、`..._without_self_loops`、`..._cross_replica` | 收发对照表的数值效果（3 个变体） |
| **reduce_scatter** | `sdy_reduce_scatter` | 边归约边切分 |
| **constant** | `sdy_constant` | **splat vs dense 的差异** |

**`collective_permute` 有 3 个文件**（本层最多）：
因为它的语义最"细"—— `source_target_pairs` 的每一种形态都要验证。

**`sdy_constant` 虽然叫 "sdy_"，但不是集合通信** ——
它验证的是**常量处理**（L5-01 / L5-03），
放在这里是因为它同样需要"逐设备不同输入"的验证方式。

---

## 六、本课与前面课的对应

| 本课验证 | 对应课 |
|---|---|
| `all_reduce` 的分组 `{0,2}` / `{1,3}` | **L5-02**（`replica_groups` 的构造） |
| 设备号 ↔ 网格坐标的对应 | **L5-01**（查找表推导的规则） |
| `per-dim-all-gather` 两种模式等价 | **L5-02**（四个 RUN 行的选项） |
| **splat vs dense 的差异** | **L5-03**（"splat 不需要切片"） |
| `collective_permute` 的收发效果 | **L5-02**（`source_target_pairs`） |

**★ 本课的价值**：
> L5-02 / L5-03 是从 **IR 形态**推断规则的；
> 本课用**真正执行的数值**验证了那些推断。
>
> 特别是 **splat vs dense** 那条 —— L5-03 的洞察在这里得到了**直接的数值证据**。
