# L6-10 · `exec-single-device-and-alltoall` — 单设备与 all-to-all（P2 · L6 最后一课）

> 层：**L6 · 执行与解释器** ｜ 优先级：P2 ｜ 前置课：`L6-00`、`L4-14`、`L4-09`

## 学习目标

看完这一课，你应该能：

1. **说明单设备分片如何被执行**（TODOLIST 验收点）；
2. 说出 `partition_id` 在测试中的**作用**；
3. 说出单设备网格 `<[], device_ids=[N]>` 的**含义**；
4. 说出上游测试「用简单算子代替」的**原因**。

## 覆盖的测试文件（4 个，全在 `executable_partitioner_pipeline/`）

| 文件 | 行数 | 本课聚焦 |
|---|---|---|
| `sdy_all_to_all_fully_scattered.mlir` | 95 | all-to-all 优化（执行结果不变） |
| `sdy_all_to_all_partially_scattered.mlir` | 69 | 同上（部分分解） |
| `single_device_switch.mlir` | 52 | **网格切换** |
| `single_device_add.mlir` | 45 | **单设备分片** |

> **与 L7-01 的分工**：两课引用**同一批文件**，但视角不同 ——
> **本课**讲**执行**（数值对不对），**L7-01** 讲**时间轴**（优化怎么发生）。
> 本课为**主课**。

## 场景（5 幕）

1. **★ `partition_id`：让「哪台设备算了什么」可验证**
2. **★ `single_device_switch`：两次网格切换**
3. 上游注释里的坦诚
4. 两个 all-to-all 文件：本课看「执行」
5. 小结：L6 的最后一课

## 核心结论

### ★ `partition_id`

```mlir
  %part_id = stablehlo.partition_id : tensor<ui32>
  %part_id_i32 = stablehlo.convert %part_id : (tensor<ui32>) -> tensor<i32>
  %part_id_bc = stablehlo.broadcast_in_dim %part_id_i32, dims=[] : (tensor<i32>) -> tensor<4xi32>
  %0 = stablehlo.add %arg0, %part_id_bc {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : tensor<4xi32>
```

**取当前设备编号 → 广播 → 加到结果上** → **每台设备的输出不同**
→ 测试就能验证「**哪台设备算了什么**」✓

**★ 三种「让结果可区分」的手法**：

| 课 | 手法 |
|---|---|
| L6-01 / L6-04 | 给每台设备**不同的输入**（`1/10/100/1000`） |
| L6-03 | 把某个 shard **乘以 1000** |
| **L6-10（本课）** | 用 **`partition_id`** 加到结果上 |

**共同目的**：让测试能区分「正确」与「错误」的实现。

### ★ 单设备网格

| 位置 | 分片标注 |
|---|---|
| `%arg0` | `@mesh, [{"x"}]` —— **2 设备网格** |
| **`%0`（结果）** | **`@single_dev_1, []`** —— **单设备网格**！ |

**「单设备分片」= 算子被标注在一个没有轴的单设备网格上** ——
意味着「**这个算子的计算只在一台设备上进行**」。

**为什么要单设备**：有些算子（如某些 `custom_call`）**无法在设备间切分**，
必须**收拢到一台设备**。用单设备网格标注它，
后续的 reshard 就会**自动插入把数据收拢的通信**。

`-sdy-resolve-single-device-sharding`（**L4-14**）就是处理这种标注的 pass。

### ★ 网格切换

数据流：`@mesh`（`x=4`）→ **`@single_dev_2`** → **`@single_dev_1`**
—— **两次切换**，每次都要**收拢/搬运数据**。

`%part_id_x10`（`partition_id × 10`）让**两次加法的影响可区分**。

### 上游注释的坦诚

```mlir
// It doesn't make much sense to run an element-wise op on single-device in
// in real practice. But since we can't hook up an arbitrary custom-call op
// to the StablleHLO interpreter straight-forwardly, we use a simple add op
// here to test the single-device sharding handling in the partitioner
// pipeline.
```

**用 `add` 代替 `custom_call`** —— 因为无法把任意 `custom_call` 接到解释器上。
**测的是机制，不是那个具体算子。**

**★ 读测试的技巧**：遇到看起来「奇怪」的测试，**先读注释**。

### 两个 all-to-all 文件

两者都有**两个 RUN 行**（`"false"`/`"true"`）—— 验证 **REPL 与 HALO 等价**（L4-09）。

**执行层面的意义**：`collective_permute` 被优化成 `all_to_all` 后，
**执行结果必须不变**。

**为什么「执行」视角重要**：优化 pass 改的是**通信方式** ——
改错了 IR **依然能通过校验**，但**跑出来的数值会错**（L6-00 的核心思想）。

## 验收点

- [x] `check_ir_fidelity.py`：15 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：**241/241 = 100%**，B 组全部通过
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说明单设备分片如何被执行（`partition_id` + 单设备网格 + 收拢通信）
