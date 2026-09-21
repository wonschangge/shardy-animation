<!-- sdy-coverage
transforms/export/test/executable_partitioner_pipeline/single_device_add.mlir
transforms/export/test/executable_partitioner_pipeline/single_device_switch.mlir
transforms/export/test/executable_partitioner_pipeline/sdy_all_to_all_fully_scattered.mlir
transforms/export/test/executable_partitioner_pipeline/sdy_all_to_all_partially_scattered.mlir
-->

# L6-10 · exec-single-device-and-alltoall — 源 IR

**L6 的最后一课**：单设备分片如何被执行 + all-to-all 优化的端到端验证。

> **与 L7-01 的分工**：两课引用**同一批文件**，但视角不同 ——
> **本课**（L6-10）讲**单设备分片如何被执行**（`partition_id` + 单设备网格）；
> **L7-01** 讲**完整流水线的时间轴**。

| 文件 | 行数 | 本课聚焦 |
|---|---|---|
| `executable_partitioner_pipeline/single_device_add.mlir` | 45 | **单设备分片** |
| `executable_partitioner_pipeline/single_device_switch.mlir` | 52 | **网格切换** |
| `executable_partitioner_pipeline/sdy_all_to_all_fully_scattered.mlir` | 95 | all-to-all 优化 |
| `executable_partitioner_pipeline/sdy_all_to_all_partially_scattered.mlir` | 69 | all-to-all 优化 |

---

## 一、★ `partition_id`：让"哪台设备算了什么"可验证

**两个 `single_device_*` 文件的共同手法**：

```mlir
sdy.mesh @mesh = <["x"=2]>
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>

func.func @single_device_add(
    %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %part_id = stablehlo.partition_id : tensor<ui32>
  %part_id_i32 = stablehlo.convert %part_id : (tensor<ui32>) -> tensor<i32>
  %part_id_bc = stablehlo.broadcast_in_dim %part_id_i32, dims=[] : (tensor<i32>) -> tensor<4xi32>
  %0 = stablehlo.add %arg0, %part_id_bc {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : tensor<4xi32>
  return %0 : tensor<4xi32>
}
```

**读法**（**本课的核心洞察**）：
- **`stablehlo.partition_id`** —— 取**当前设备的编号**。
- 转成 `i32` → `broadcast_in_dim` 广播成 `tensor<4xi32>` → **加到 `%arg0`**。
- **★ 效果**：**每台设备的计算结果不同**（设备 0 加 0、设备 1 加 1）。
- → 测试就能验证"**哪台设备算了什么**" ✓

**★ 与前面几课的手法对照**：

| 课 | 手法 | 目的 |
|---|---|---|
| L6-01 / L6-04 | 给每台设备**不同的输入**（`1/10/100/1000`） | 让结果可区分 |
| L6-03 | 把某个 shard **乘以 1000** | 让贡献可辨识 |
| **L6-10（本课）** | 用 **`partition_id`** 加到结果上 | 让**设备号**体现在结果里 |

**共同目的**：**让测试能区分"正确"与"错误"的实现**。

### ★ 分片的变化：从 `@mesh` 到单设备网格

| 位置 | 分片标注 |
|---|---|
| `%arg0` | `@mesh, [{"x"}]` —— **2 设备网格** |
| **`%0`（结果）** | **`@single_dev_1, []`** —— **单设备网格**！ |

**★ 这就是"单设备分片"**：
> 算子被标注在**一个没有轴的单设备网格**上 ——
> 意味着"**这个算子的计算只在一台设备上进行**"。
>
> `-sdy-resolve-single-device-sharding`（**L4-14**）就是处理这种标注的 pass。

**★ 为什么要"单设备"**：
> 有些算子（如某些 `custom_call`）**无法在设备间切分** ——
> 必须**收拢到一台设备**上算。用单设备网格标注它，
> 后续的 reshard 就会**自动插入把数据收拢的通信**。

---

## 二、注释里的"坦诚"

`single_device_add.mlir` 的注释很值得一读：

```mlir
// It doesn't make much sense to run an element-wise op on single-device in
// in real practice. But since we can't hook up an arbitrary custom-call op
// to the StablleHLO interpreter straight-forwardly, we use a simple add op
// here to test the single-device sharding handling in the partitioner
// pipeline.
```

**读法**：
- **"It doesn't make much sense to run an element-wise op on single-device in real practice"**
  —— 承认**现实中不会这么做**（逐元素算子没必要放单设备）。
- **"since we can't hook up an arbitrary custom-call op to the StablleHLO interpreter straight-forwardly"**
  —— **真正的原因**：无法把任意 `custom_call` 接到 StableHLO 解释器上。
- **"we use a simple add op here to test the single-device sharding handling"**
  —— 所以用 `add` **代替** `custom_call` 来**测试单设备分片处理**。

**★ 这是上游测试的一个常见模式**：
> **用"能跑起来的简单算子"代替"想测但跑不起来的算子"** ——
> 测试的是**机制**（单设备分片处理），而不是那个具体算子。
> **L7-03** 讲的 `registry_failures` 用的是 `custom_call`（但只验证警告，不执行）。

---

## 三、★ `single_device_switch`：网格切换

```mlir
sdy.mesh @mesh = <["x"=4]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>
sdy.mesh @single_dev_2 = <[], device_ids=[2]>

func.func @single_device_switch(
    %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %part_id = stablehlo.partition_id : tensor<ui32>
  %part_id_i32 = stablehlo.convert %part_id : (tensor<ui32>) -> tensor<i32>
  %part_id_bc = stablehlo.broadcast_in_dim %part_id_i32, dims=[] : (tensor<i32>) -> tensor<4xi32>

  %c10 = stablehlo.constant dense<10> : tensor<4xi32>
  %part_id_x10 = stablehlo.multiply %part_id_bc, %c10 : tensor<4xi32>

  %0 = stablehlo.add %arg0, %part_id_bc {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_2, []>]>
  } : tensor<4xi32>

  %1 = stablehlo.add %0, %part_id_x10 {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : tensor<4xi32>
```

**读法**（**比 `single_device_add` 多了一次切换**）：

| 位置 | 网格 | 说明 |
|---|---|---|
| `%arg0` | `@mesh`（`x=4`） | 起点：4 台设备 |
| **`%0`** | **`@single_dev_2`** | 第一次加法在**物理设备 2** |
| **`%1`** | **`@single_dev_1`** | 第二次加法在**物理设备 1** |

**★ 数据流**：`@mesh` → `@single_dev_2` → `@single_dev_1`
—— **两次网格切换**，每次都要**收拢/搬运数据**。

**★ `partition_id × 10` 的作用**：
- 第一次加 `partition_id`（设备 2 → 加 2）
- 第二次加 `partition_id × 10`（设备 1 → 加 10）
- → **两次加法的影响可区分** ✓

**★ 这验证了 L4-14 的 `resolve-single-device-sharding`**：
> 从多设备网格切到单设备网格时，pass 要**插入 reshard**
> 把数据**收拢到目标设备**；反向切换则要**广播**。

---

## 四、两个 all-to-all 文件（与 L7-01 同源，本课聚焦"执行"）

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
```

```mlir
// Resharding pattern:
```

```mlir
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
```

```mlir
// OptimizeCollectivesPass eliminates the collective_permute by decomposing
```

```mlir
// The permuted axes ("x" and "y") are fully scattered off dimension 0 to
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
```

```mlir
// Resharding pattern:
```

```mlir
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
```

```mlir
// OptimizeCollectivesPass detects the collective_permute + all_to_all chain and
```

```mlir
// The permuted axes ("x" and "y") are partially scattered off dimension 0 (only
```

**读法**（**本课聚焦"执行"视角**）：
- 两者都有**两个 RUN 行**（`"false"` / `"true"`）—— 验证 **REPL 与 HALO 等价**（**L4-09**）。
- **执行层面的意义**：`collective_permute` 被优化成 `all_to_all` 后，
  **执行结果必须不变** —— 这正是这两个文件验证的。
- **`fully` vs `partially`**：分解的**程度不同**（L7-01 详述）。

**★ 本课与 L7-01 的分工**：
| 课 | 视角 |
|---|---|
| **L6-10（本课）** | **执行**：优化后的通信**跑出来的数值对不对** |
| **L7-01** | **时间轴**：`collective_permute` 在流水线里**怎么被消除** |

---

## 五、四个文件的族谱

| 族 | 文件 | 本课聚焦 |
|---|---|---|
| **单设备分片** | `single_device_add` | `partition_id` + 单设备网格 |
| **网格切换** | `single_device_switch` | `@mesh` → `@single_dev_2` → `@single_dev_1` |
| **all-to-all 优化** | `sdy_all_to_all_fully_scattered` | 优化后**执行结果不变** |
| **all-to-all 优化** | `sdy_all_to_all_partially_scattered` | 同上（部分分解） |

**★ 本课的三条结论**：

1. **`partition_id` 是让"哪台设备算了什么"可验证的关键手法** ——
   加到结果上，设备号就体现在数值里。
2. **单设备网格 `<[], device_ids=[N]>` 表达"这个算子只在一台设备上算"** ——
   从多设备网格切过去时，pass 会**自动插入收拢通信**。
3. **上游测试常用"简单算子代替想测但跑不起来的算子"** ——
   注释坦诚说明了这一点（用 `add` 代替 `custom_call`）。

**一句话总结**：
> **L6-10 讲"单设备分片怎么被执行"** ——
> 用 `partition_id` 让设备号体现在结果里，
> 用**单设备网格**表达"只在一台设备上算"，
> 并验证 all-to-all 优化后的**执行结果不变**。
